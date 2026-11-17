"""Engine-free coordinator for the OR-Tools Research V4 stages.

The stage owners deliberately use specialized CP-SAT models.  This module
binds their immutable evidence into one scenario verdict without inventing a
second rules implementation or pretending the models are monolithic.
"""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
from dataclasses import dataclass, field, replace
from functools import partial
from importlib.metadata import version
import json
import math
import multiprocessing
import platform
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence

from ortools.sat.python import cp_model

from baloot_platform.adjudication.research import (
    AdjudicationResearchAuthorityV1,
    AdjudicationResearchReportV1,
)
from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.cards import Card, FULL_DECK
from baloot_platform.core.dealing import DealLayoutV1
from baloot_platform.core.enums import Contract, GamePhase, Seat, Suit, Team
from baloot_platform.core.escalation_ir import EscalationDomainIRV1
from baloot_platform.core.models import LegalPlayActionV1
from baloot_platform.core.play_ir import PlayDomainIRV1
from baloot_platform.core.rule_source_catalog import RuleSourceCatalogV1, load_exact_rule_source_catalog
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.scoring_ir import ScoringDomainIRV1
from baloot_platform.tactics import DELAYED_PAYOFF_TACTIC_IDS, NATIVE_PROFILE_TACTIC_IDS, TacticalDetectorProfileV1
from baloot_platform.config import platform_root

from .campaign_coordinator import (
    CampaignAttemptProgressV1,
    CampaignAttemptReceiptV1,
    CampaignBudgetV1,
    CampaignReceiptV1,
    CampaignShardOutcomeV1,
    CampaignShardRequestV1,
    CampaignStageTimingV1,
    CampaignVariantReceiptV1,
    CampaignVariantV1,
    SEMANTIC_DISTANCE_POLICY_V1,
    DeterministicCpSatCampaignCoordinatorV1,
    build_rules_preserving_diversity_document_v1,
    decision_state_card_vector_v1,
    decision_state_card_hamming_distance_v1,
    semantic_threshold_distance_v1,
    rules_preserving_symmetry_key_v1,
    select_campaign_variants_v1,
)
from .sharding import ResearchShardSpecV1
from .ortools_escalation_stage import (
    OrToolsEscalationStageResultV1,
    OrToolsEscalationStageV1,
)
from .ortools_complete_state_graph import OrToolsCompleteStateGraphCompilerV1
from .ortools_exact_state import (
    ExactNormalizationStopped,
    OrToolsExactStateNormalizerV1,
    OrToolsNormalizedExactStateV1,
    ProjectDecisionNotStrategicError,
    _effective_prefix_hash,
    _native_project_prefix_projection,
)
from .ortools_information_set_compiler import (
    ActorInformationGraphWorldV1,
    ActorInformationSetCompilationV1,
    ActorInformationSetGraphV1,
    InformationSetCompilationStopped,
    OrToolsActorInformationSetCompilerV1,
    prove_actor_information_set_v1,
)
from .ortools_nonplay_complete_graph import compile_nonplay_complete_graph_v1
from .ortools_nonplay_play_graft import OrToolsNonPlayPlayGraphGraftV1
from .ortools_information_set_proof import _solve_cp_sat_with_cancel_v1
from .ortools_nonplay_graphs import (
    AuctionGraphChoiceV1,
    EscalationGraphActionV1,
    NonPlayGraphBudgetV1,
    NonPlayGraphCompilationStoppedV1,
    compile_auction_decision_graph_v1,
    compile_escalation_decision_graph_v1,
    compile_project_declaration_graph_v1,
    enumerate_escalation_opportunity_orders_v1,
    enumerate_escalation_prior_histories_v1,
    bind_escalation_recall_v1,
    normalize_escalation_recall_v1,
    encode_native_auction_model_receipt_v1,
    _escalation_window_key,
)
from .ortools_nonplay_policy_continuation import (
    OrToolsNonPlayPolicyContinuationV1,
)
from .ortools_nonplay_observation import (
    build_nonplay_decision_observer_v1,
    build_nonplay_root_observer_v1,
)
from .ortools_play_state_predicates import (
    CanonicalObserverScopeV1,
    CpSatPlayPrefixActionV1,
    CpSatPlayStatePredicateContextV1,
    _range as _state_range_literal,
)
from .ortools_play_tail import (
    NATIVE_OBSERVER_SAFE_PLAY_POLICY_IDS,
    NATIVE_PARTNER_FEED_POLICY_IDS,
    NATIVE_ESTABLISHED_CONTROL_POLICY_ID,
    NATIVE_SECURE_CONTROL_POLICY_ID,
    NATIVE_GUARD_PROTECTION_POLICY_ID,
    NATIVE_CLOSED_TRICK_FEED_POLICY_ID,
    OrToolsPlayTailResultV1,
    OrToolsPlayTailStageV1,
    PlayTailSolveStopped,
    TailPrefixActionV1,
    _observer_safe_play_score_v1,
    _established_control_blockers_v1,
    _established_control_bonus_v1,
    _guard_discard_cards_v1,
    _guard_discard_bonus_v1,
)
from .ortools_project_fragment import (
    OrToolsProjectLifecycleFragmentV1,
    ProjectPatternVariablesV1,
    attach_project_lifecycle_fragment_v1,
    project_lifecycle_result_v1,
)
from .ortools_project_model import (
    OrToolsProjectLifecycleModelV1,
    OrToolsProjectLifecycleResultV1,
)
from .ortools_scoring_stage import (
    CpSatScoringInputsV1,
    OrToolsScoringStageV1,
    attach_scoring_fragment_v1,
)
from .ortools_proof import (
    ProofGraphSequencePackV1,
    OrToolsGraphProofOutcomeV1,
    OrToolsScenarioGraphProofV1,
    ProofGraphV1,
)
from .ortools_scenario_model import (
    _CONTRACT_IDS,
    _NO_TRUMP,
    _AuctionPlanV1,
    OrToolsAuctionActionV1,
    OrToolsNativeAuctionFragmentV1,
    OrToolsScenarioAuctionFragmentV1,
    OrToolsScenarioModelV1,
)
from .result_authority import ResearchAuthorityV1, validate_trace_quality_evidence_v1
from .result_authority import validate_native_escalation_quality_evidence_v1 as validate_native_escalation_quality_evidence_v1
from .result_authority import validate_native_auction_quality_evidence_v1 as validate_native_auction_quality_evidence_v1
from .result_authority import validate_native_decision_quality_evidence_v1 as validate_native_decision_quality_evidence_v1
from .contracts import canonical_json
from .scenario_contracts import (
    ExactTracePrefixV1,
    ResearchScenarioV1,
    ScenarioExpressionV1,
    ScenarioPredicateV1,
    ScenarioProofRequestV1,
    ScenarioProofBudgetV1,
    ScenarioPlayingQualityPolicyV1,
    ScenarioQualityExecutionBudgetV1,
    validate_proof_boundary_scope_v1,
    validate_native_delayed_payoff_scope_v1,
    hard_continuation_count_request_issue_v1,
    hard_tactic_causal_request_issue_v1,
    canonical_sha256,
)
from .scenario_deal_predicates_v2 import DealPredicateModelContextV2
from .scenario_predicate_auction_escalation_v2 import (
    CpSatAuctionPredicateContextV1,
    CpSatEscalationPredicateContextV1,
)
from .scenario_predicate_lowering_v2 import (
    CpSatActionVariablesV1,
    CpSatPredicateModelContextV1,
    CpSatPredicateLoweringV1,
    CpSatTrickVariablesV1,
)
from .scenario_predicate_manifest_v2 import (
    ScenarioPredicateCatalogLoweringManifestV2,
    shared_scenario_predicate_catalog_lowering_manifest_v2,
)
from .scenario_predicate_capabilities_v3 import (
    NATIVE_V3_ACTOR_ROOT_TACTIC_IDS,
    native_v3_state_decision_supported,
)
from .scenario_adjudication_predicates_v3 import (
    FALSE_EKKA_PREDICATE_V3,
    OUT_OF_TURN_PREDICATE_V3,
    PURCHASE_BEFORE_TURN_PREDICATE_V3,
    ScenarioPredicateCatalogLoweringManifestV3,
    shared_scenario_predicate_catalog_lowering_manifest_v3,
)
from .ortools_adjudication_contexts_v3 import (
    wrap_auction_adjudication_context_v3,
    wrap_play_adjudication_context_v3,
)
from .scenario_project_outcome_lowering_v2 import (
    CpSatOutcomePredicateContextV1,
    CpSatProjectPredicateContextV1,
    PROJECT_PREDICATE_KINDS_V2,
)
from .scenario_proof_tactic_predicates_v2 import (
    PrincipalVariationPlayEvidenceV1,
    PrincipalVariationTrickEvidenceV1,
    ProofTacticPredicateContextV1,
    RegisteredTacticalEvidenceV1,
    lower_native_proof_predicate_v3,
)
from .tail_first_stage_planner import TailFirstStagePlannerV1


def _execute_seed_campaign_process_v1(connection: Any, task: Mapping[str, Any]) -> None:
    """Run one deterministic shard outside the Python GIL."""

    cancelled = False
    cpu_started = time.process_time_ns()

    def sample_cpu() -> None:
        try:
            connection.send(("cpu_sample", max(0, time.process_time_ns() - cpu_started)))
        except (BrokenPipeError, EOFError, OSError):
            pass  # A lost sample must not affect proof or cancellation.

    def cancel_check() -> bool:
        nonlocal cancelled
        try:
            while connection.poll():
                message = connection.recv()
                cancelled = cancelled or message == "cancel"
        except (EOFError, OSError):
            cancelled = True
        return cancelled

    def progress(payload: Mapping[str, Any]) -> None:
        nonlocal cancelled
        if cancel_check():
            return
        sample_cpu()
        try:
            connection.send(("progress", dict(payload)))
        except (BrokenPipeError, EOFError, OSError):
            cancelled = True

    def checkpoint(payload: Mapping[str, Any]) -> None:
        nonlocal cancelled
        # Retain evidence before cancellation acknowledgement can seal the receipt.
        try:
            sample_cpu()
            connection.send(("checkpoint", canonical_json(dict(payload))))
        except (BrokenPipeError, EOFError, OSError):
            cancelled = True

    try:
        executor = OrToolsUnifiedResearchExecutorV1(
            rules=task["rules"],
            bundle_revision_id=str(task["bundleRevisionId"]),
            bundle_hash=str(task["bundleHash"]),
            backend_id=str(task["backendId"]),
            tactical_detector_profile=task.get("tacticalDetectorProfile"),
            adjudication_authority=task.get("adjudicationAuthority"),
            rule_source_catalog=task.get("ruleSourceCatalog"),
            witness_policies=task.get("witnessPolicies"),
            quality_policy=task.get("qualityPolicy"),
            quality_execution=task.get("qualityExecution"),
            quality_proof_profiles=task.get("qualityProofProfiles"),
            quality_search_profiles=task.get("qualitySearchProfiles"),
            runtime_proposals=task.get("runtimeProposals"),
            generation=task.get("generation"),
        )
        if getattr(executor, "bid_quality_cache", None) is not None:
            executor.bid_quality_cache.merge(task.get("bidQualityCache", ()))
        campaign = executor._execute_seed_campaign(
            task["scenario"],
            task["proofRequest"],
            cancel_check=cancel_check,
            progress_callback=progress,
            checkpoint_callback=checkpoint,
            **dict(task["options"]),
        )
        connection.send(("result", campaign))
    except Exception as exc:  # pragma: no cover - exercised through parent receipt
        try:
            connection.send(
                (
                    "error",
                    {
                        "type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                        "detail": str(exc),
                    },
                )
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def _run_seed_campaign_process_v1(
    task: Mapping[str, Any],
    request: Any,
    *,
    cancellation_poll_milliseconds: int,
) -> Any:
    context = multiprocessing.get_context("spawn")
    budget_seconds = (
        task.get("options", {}).get("maximum_campaign_seconds")
        if task.get("backendId") == _V3_BACKEND_ID
        else None
    )
    started = time.monotonic()
    parent, child = context.Pipe(duplex=True)
    process = context.Process(
        target=_execute_seed_campaign_process_v1,
        args=(child, task),
        daemon=False,
    )
    process.start()
    child.close()
    result = None
    error = None
    reported_terminal_ordinals: set[int] = set()
    cancellation_sent_at = None
    budget_exhausted = False
    poll_seconds = max(0.001, cancellation_poll_milliseconds / 1000.0)
    monitor_stopped = threading.Event()
    transfer_expired = threading.Event()

    def acknowledge_cancellation() -> None:
        # Receiving a large evidence frame must not block the control acknowledgement.
        while not monitor_stopped.wait(min(poll_seconds, 0.005)):
            if (request.cancellation_requested() and request.deadline_ns is not None
                    and time.perf_counter_ns() >= request.deadline_ns):
                transfer_expired.set()
                try:
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=1.0)
                        if process.is_alive():
                            process.kill()
                            process.join(timeout=1.0)
                except OSError:
                    pass  # A concurrently exited child closes the same pipe.
                return

    monitor = (
        threading.Thread(target=acknowledge_cancellation, daemon=True)
        if isinstance(request, CampaignShardRequestV1) else None
    )
    try:
        if monitor is not None:
            monitor.start()
        while result is None and error is None:
            if parent.poll(poll_seconds):
                with request.evidence_transfer_lock:
                    if request.cancellation_requested():
                        cancellation_sent_at = time.monotonic()
                        break
                    try:
                        kind, payload = parent.recv()
                    except (BrokenPipeError, EOFError, OSError):
                        if transfer_expired.is_set():
                            cancellation_sent_at = time.monotonic()
                            request.progress.update({
                                "currentStage": "private_evidence_transfer_incomplete",
                                "incompleteCompilation": {"stage": "private_evidence_transfer",
                                    "exhaustionReason": "maximum_seconds"},
                            })
                        break
                    if kind == "checkpoint":
                        request.report_checkpoint(payload)
                if kind == "progress":
                    if (
                        payload.get("currentStage") == "candidate_terminal"
                        and type(payload.get("executionOrdinal")) is int
                    ):
                        reported_terminal_ordinals.add(payload["executionOrdinal"])
                    request.report_progress(payload)
                elif kind == "cpu_sample":
                    request.report_worker_cpu_sample(payload)
                elif kind == "result":
                    result = payload
                elif kind == "error":
                    error = payload
            budget_exhausted = bool(
                budget_seconds is not None
                and time.monotonic() - started >= budget_seconds
            )
            if (request.cancellation_requested() or budget_exhausted) and cancellation_sent_at is None:
                cancellation_sent_at = time.monotonic()
                try:
                    parent.send("cancel")
                except (BrokenPipeError, EOFError, OSError):
                    break
            if not process.is_alive() and not parent.poll():
                break
            if (
                cancellation_sent_at is not None
                and time.monotonic() - cancellation_sent_at >= max(0.05, poll_seconds * 3)
            ):
                break
    except (BrokenPipeError, EOFError, OSError):
        pass  # A closed Windows pipe is handled by the retained terminal receipt below.
    finally:
        monitor_stopped.set()
        if monitor is not None and monitor.ident is not None:
            monitor.join()
        if process.is_alive() and result is None:
            process.terminate()
        process.join(timeout=1.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=1.0)
        parent.close()
    if error is not None:
        raise RuntimeError(f"{error['type']}: {error['detail']}")
    if result is not None:
        for ordinal, execution in enumerate(result.executions, 1):
            if ordinal not in reported_terminal_ordinals:
                request.report_progress(
                    _candidate_terminal_progress_v1(
                        execution,
                        ordinal=ordinal,
                        decision_kind=str(task["scenario"].decision_point["kind"]),
                    )
                )
    if result is None:
        incomplete_receipt = (
            request.progress_snapshot()
            if task.get("backendId") == _V3_BACKEND_ID
            else {}
        )
        if budget_exhausted or (
            incomplete_receipt.get("currentStage") == "candidate_terminal"
            and incomplete_receipt.get("candidateStatus") == "incomplete"
        ):
            if budget_exhausted:
                # Keep timeout telemetry, never send private quality checkpoints live.
                progress = {key: incomplete_receipt[key] for key in (
                    "candidateHash", "completedCandidates", "qualifiedCandidates",
                    "proofCompletionStatus", "qualitySolver",
                ) if key in incomplete_receipt}
                ordinal = progress.get("completedCandidates")
                if (type(ordinal) is int and ordinal > 0
                        and isinstance(progress.get("candidateHash"), str)
                        and ordinal not in reported_terminal_ordinals):
                    progress.update(candidateStatus="incomplete", executionOrdinal=ordinal,
                        candidateAttemptReason="shard_budget_exhausted",
                        incompleteCompilation={"stage": incomplete_receipt.get("currentStage"),
                            "exhaustionReason": "maximum_seconds"})
                request.report_progress({**progress, "currentStage": "shard_budget_exhausted"})
            else:
                request.report_progress({
                    **{key: value for key, value in incomplete_receipt.items()
                       if key not in {"qualityCheckpoints", "workerCpuAccounting"}},
                    "currentStage": "transport_after_incomplete",
                    "transportExitCode": process.exitcode,
                })
            options = task["options"]
            return OrToolsResearchCampaignResultV1(
                "incomplete", (), (), (), (),
                options["result_count"], options["maximum_candidates"],
                options["symmetry_policy_id"],
                "decision_state_card_hamming_v1"
                if options["minimum_principal_variation_action_distance"] is None
                else SEMANTIC_DISTANCE_POLICY_V1,
                options["minimum_decision_state_distance"],
                max(0.0, time.monotonic() - started), 0.0,
            )
        raise RuntimeError(
            "campaign shard process ended without a terminal result"
            if cancellation_sent_at is None
            else "campaign shard process cancelled"
        )
    return result


BACKEND_ID = "ortools_cp_sat_v2"
BACKEND_VERSION = "2.0.0"
_V3_BACKEND_ID = "ortools_cp_sat_v3"
_INPUT_PROJECTION_DEFERRED_STAGES = frozenset({"playPrefix", "outcome", "proof", "tactics"})
_V3_BACKEND_VERSION = "3.1.144"
_KABOUT_TACTIC_IDS_V3 = frozenset({"kabout_creation", "kabout_preservation", "kabout_prevention"})
_ROOT_VALUE_TACTIC_IDS_V3 = frozenset({"only_winning_move", "contract_rescue", "debate", "project_bidding"}) | _KABOUT_TACTIC_IDS_V3
_SEMANTIC_COMPILER_CALLBACKS = (
    "candidate_compiler",
    "context_compiler",
    "graph_compiler",
)


def _reject_v3_semantic_callback_injection(
    backend_id: str,
    *,
    candidate_compiler: Any = None,
    context_compiler: Any = None,
    graph_compiler: Any = None,
) -> None:
    """Keep V3 semantic compilation owned by its frozen backend."""

    if backend_id != _V3_BACKEND_ID:
        return
    supplied = [
        name
        for name, value in (
            ("candidate_compiler", candidate_compiler),
            ("context_compiler", context_compiler),
            ("graph_compiler", graph_compiler),
        )
        if value is not None
    ]
    if supplied:
        raise ValueError(
            "ortools_cp_sat_v3 forbids caller-supplied semantic compiler callbacks: "
            + ", ".join(supplied)
        )


EXECUTION_STATUSES = frozenset(
    {
        "feasible",
        "infeasible",
        "not_found",
        "semantic_postcondition_miss",
        "incomplete",
        "compiler_disagreement",
        "proof_failed",
        "qualified",
    }
)
_EXECUTION_STAGE_TIMING_KEYS = (
    "construction",
    "playingQuality",
    "qualityAuction",
    "qualityProjectPlay",
    "tailConstruction",
    "tailModelBuild",
    "tailModelConfigure",
    "tailSolver",
    "prefixReachability",
    "proofBoundary",
    "proofBoundaryModelBuild",
    "proofBoundarySolver",
    "prefixLift",
    "exactNormalization",
    "finalNormalization",
    "graph",
    "proof",
    "qualification",
)


def _supports_default_nonplay_policy_graph(
    scenario: ResearchScenarioV1,
    request: ScenarioProofRequestV1,
) -> bool:
    return (
        scenario.origin.kind in {"search", "exact_trace_prefix"}
        and scenario.decision_point["kind"] in {"bid", "escalation", "project"}
        and request.mode == "policy_conditioned"
        and request.visibility == "complete_state"
        and request.strategy_class == "frozen_policy"
        and request.policies is not None
    )


def _supports_default_nonplay_complete_graph(
    scenario: ResearchScenarioV1,
    request: ScenarioProofRequestV1,
) -> bool:
    return (
        scenario.origin.kind in {"search", "exact_trace_prefix"}
        and scenario.decision_point["kind"] in {"bid", "escalation", "project"}
        and request.mode in {"exact", "bounded"}
        and request.visibility == "complete_state"
        and request.strategy_class in {None, "pure_deterministic"}
    )


_HASH_ZERO = "0" * 64
_STAGE_PRIORITY = {
    "proof": 0,
    "tactics": 1,
    "playPrefix": 2,
    "outcome": 3,
    "projects": 4,
    "state": 5,
    "escalation": 6,
    "auction": 7,
    "deal": 8,
}


def _campaign_incomplete_compilation_summary_v1(value: Any) -> dict[str, Any] | None:
    """Keep campaign progress bounded; the full receipt remains execution-artifact owned."""
    if not isinstance(value, Mapping):
        return None
    return {
        key: copy.deepcopy(value[key])
        for key in (
            "stage",
            "sourceStage",
            "status",
            "exhaustionReason",
            "completionStatus",
            "sha256",
            "candidateHash",
            "targetActor",
            "generatedActorAtTarget",
            "targetPlayIndex",
            "firstConstructedPlayDivergenceIndex",
            "evidenceBytes",
            "reason",
            "detail",
            "qualityStatus",
            "qualityFailureStage",
            "qualityFailureCauseStage",
            "qualityFailureQueryCount",
            "qualityFailureQueryPhase",
            "qualityFailurePathPhase",
            "qualityFailureReason",
        )
        if key in value
    }


def _quality_failure_cause_v1(value: Any) -> tuple[str | None, int | None, str | None, str | None]:
    """Report bounded cause metadata, not private solver evidence."""
    cause = value
    stage = None
    query_count = None
    for _ in range(8):
        evidence = cause.get("evidence") if isinstance(cause, Mapping) else None
        nested = (evidence.get("causeReceipt") or evidence.get("cause")
                  or evidence.get("generationReceipt")
                  or evidence.get("nativeBidDominance")
                  or evidence.get("nativeObservedBidDominance")) if isinstance(evidence, Mapping) else None
        if not isinstance(nested, Mapping) and isinstance(cause, Mapping):
            nested = cause.get("cause")
        if not isinstance(nested, Mapping):
            break
        if isinstance(nested.get("stage"), str):
            stage = nested["stage"]
        if query_count is None and type(nested.get("queriesUsed")) is int and nested["queriesUsed"] >= 0:
            query_count = nested["queriesUsed"]
        cause = nested
    evidence = cause.get("evidence") if isinstance(cause, Mapping) else None
    if query_count is None and isinstance(evidence, Mapping) and isinstance(evidence.get("queries"), (list, tuple)):
        query_count = len(evidence["queries"])
    phase = evidence.get("queryPhase") if isinstance(evidence, Mapping) else None
    path_phase = evidence.get("pathPhase") if isinstance(evidence, Mapping) else None
    return (stage, query_count, phase if isinstance(phase, str) else None,
            path_phase if isinstance(path_phase, str) else None)


def _hash_document(value: Mapping[str, Any]) -> str:
    return canonical_sha256(copy.deepcopy(dict(value)))


@dataclass(frozen=True, slots=True)
class OrToolsResearchStageReceiptV1:
    stage: str
    model_hash: str
    evidence_hash: str
    semantic_hash: str
    solver_status: str
    branches: int = 0
    conflicts: int = 0
    variable_count: int = 0
    constraint_count: int = 0
    named_assumptions: tuple[Mapping[str, Any], ...] = ()
    infeasible_core: tuple[str, ...] = ()
    evidence_bindings: tuple[Mapping[str, Any], ...] = ()
    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        if type(self.stage) is not str or not self.stage:
            raise ValueError("stage receipt requires a stage")
        for name in ("model_hash", "evidence_hash", "semantic_hash"):
            value = getattr(self, name)
            if type(value) is not str or len(value) != 64:
                raise ValueError(f"stage receipt {name} must be a SHA-256 hash")
        if type(self.solver_status) is not str or not self.solver_status:
            raise ValueError("stage receipt requires a solver status")
        if type(self.branches) is not int or type(self.conflicts) is not int:
            raise TypeError("stage receipt counters must be integers")
        if self.branches < 0 or self.conflicts < 0:
            raise ValueError("stage receipt counters cannot be negative")
        if type(self.variable_count) is not int or type(self.constraint_count) is not int:
            raise TypeError("stage receipt model sizes must be integers")
        if self.variable_count < 0 or self.constraint_count < 0:
            raise ValueError("stage receipt model sizes cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        core = {
            "schemaVersion": 1,
            "stage": self.stage,
            "modelHash": self.model_hash,
            "evidenceHash": self.evidence_hash,
            "semanticHash": self.semantic_hash,
            "solverStatus": self.solver_status,
            "branches": self.branches,
            "conflicts": self.conflicts,
        }
        return {**core, "sha256": canonical_sha256(core)}


@dataclass(frozen=True, slots=True)
class OrToolsResearchStageReceiptV2:
    """Closed native CP-SAT stage receipt for new V3.1 artifacts."""

    stage: str
    model_hash: str
    evidence_hash: str
    semantic_hash: str
    solver_status: str
    variable_count: int = 0
    constraint_count: int = 0
    named_assumptions: tuple[Mapping[str, Any], ...] = ()
    infeasible_core: tuple[str, ...] = ()
    evidence_bindings: tuple[Mapping[str, Any], ...] = ()
    elapsed_seconds: float = 0.0
    branches: int = 0
    conflicts: int = 0

    def __post_init__(self) -> None:
        if type(self.stage) is not str or not self.stage:
            raise ValueError("stage receipt requires a stage")
        for name in ("model_hash", "evidence_hash", "semantic_hash"):
            value = getattr(self, name)
            if type(value) is not str or len(value) != 64:
                raise ValueError(f"stage receipt {name} must be a SHA-256 hash")
        if type(self.solver_status) is not str or not self.solver_status:
            raise ValueError("stage receipt requires a solver status")
        for name in ("variable_count", "constraint_count", "branches", "conflicts"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"stage receipt {name} must be non-negative")
        if (
            type(self.elapsed_seconds) not in {int, float}
            or not math.isfinite(self.elapsed_seconds)
            or self.elapsed_seconds < 0
        ):
            raise ValueError("stage receipt elapsed_seconds must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        core = {
            "schemaVersion": 2,
            "stage": self.stage,
            "modelHash": self.model_hash,
            "evidenceHash": self.evidence_hash,
            "semanticHash": self.semantic_hash,
            "solverStatus": self.solver_status,
            "variableCount": self.variable_count,
            "constraintCount": self.constraint_count,
            "namedAssumptions": [copy.deepcopy(dict(item)) for item in self.named_assumptions],
            "infeasibleCore": list(self.infeasible_core),
            "evidenceBindings": [copy.deepcopy(dict(item)) for item in self.evidence_bindings],
            "elapsedSeconds": float(self.elapsed_seconds),
            "branches": self.branches,
            "conflicts": self.conflicts,
        }
        return {**core, "sha256": canonical_sha256(core)}


@dataclass(frozen=True, slots=True)
class OrToolsNeutralCandidateV1:
    origin: str
    trace: Mapping[str, Any]
    state: Mapping[str, Any]
    no_good_vector: tuple[int, ...]
    stage_receipts: Mapping[
        str, OrToolsResearchStageReceiptV1 | OrToolsResearchStageReceiptV2
    ]
    exact_state_id: str | None = None
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.origin not in {"search", "exact_trace_prefix"}:
            raise ValueError("candidate origin is unsupported")
        if type(self.no_good_vector) is not tuple or (
            self.no_good_vector
            and (
                len(self.no_good_vector) != 32
                or sorted(self.no_good_vector) != list(range(32))
            )
        ):
            raise ValueError("candidate no-good vector must be a canonical deck")
        receipts = dict(self.stage_receipts)
        if not receipts or set(receipts) != {item.stage for item in receipts.values()}:
            raise ValueError("candidate stage receipts must be keyed by stage")
        core = {
            "schemaVersion": 1,
            "origin": self.origin,
            # Hashing is read-only; to_dict still detaches exported evidence.
            "trace": dict(self.trace),
            "state": dict(self.state),
            "noGoodVector": list(self.no_good_vector),
            "stageReceipts": {
                stage: receipt.to_dict() for stage, receipt in sorted(receipts.items())
            },
            "exactStateId": self.exact_state_id,
        }
        object.__setattr__(self, "stage_receipts", receipts)
        object.__setattr__(self, "sha256", canonical_sha256(core))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "origin": self.origin,
            "trace": copy.deepcopy(dict(self.trace)),
            "state": copy.deepcopy(dict(self.state)),
            "noGoodVector": list(self.no_good_vector),
            "stageReceipts": {
                stage: receipt.to_dict()
                for stage, receipt in sorted(self.stage_receipts.items())
            },
            "exactStateId": self.exact_state_id,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class OrToolsStageContextBindingV1:
    stage: str
    source_evidence_hash: str
    context: Any = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.stage) is not str or not self.stage:
            raise ValueError("stage binding requires a stage")
        if (
            type(self.source_evidence_hash) is not str
            or len(self.source_evidence_hash) != 64
        ):
            raise ValueError("stage binding requires its source evidence hash")
        if not hasattr(self.context, "model") or not hasattr(self.context, "sha256"):
            raise TypeError("stage binding context must expose model and sha256")


@dataclass(frozen=True, slots=True)
class OrToolsResearchPreflightV1:
    origin: str
    start_stage: str
    compiled_stages: tuple[str, ...]
    hard_predicate_ids: tuple[str, ...]
    soft_predicate_ids: tuple[str, ...]
    unsupported: tuple[Mapping[str, str], ...]
    lowering_manifest_hash: str
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        core = {
            "schemaVersion": 1,
            "origin": self.origin,
            "startStage": self.start_stage,
            "compiledStages": list(self.compiled_stages),
            "hardPredicateIds": list(self.hard_predicate_ids),
            "softPredicateIds": list(self.soft_predicate_ids),
            "unsupported": [dict(item) for item in self.unsupported],
            "loweringManifestHash": self.lowering_manifest_hash,
        }
        object.__setattr__(self, "sha256", canonical_sha256(core))

    @property
    def supported(self) -> bool:
        return not self.unsupported

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "origin": self.origin,
            "startStage": self.start_stage,
            "compiledStages": list(self.compiled_stages),
            "hardPredicateIds": list(self.hard_predicate_ids),
            "softPredicateIds": list(self.soft_predicate_ids),
            "unsupported": [dict(item) for item in self.unsupported],
            "loweringManifestHash": self.lowering_manifest_hash,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class OrToolsPredicateEvaluationV1:
    status: str
    satisfied: bool
    evidence: tuple[Mapping[str, Any], ...]
    named_core: tuple[str, ...]
    model_hash: str
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.status not in {"satisfied", "target_miss", "infeasible", "incomplete"}:
            raise ValueError("predicate evaluation status is unsupported")
        evidence: list[Mapping[str, Any]] = []
        for raw in self.evidence:
            document = copy.deepcopy(dict(raw))
            known = document.pop("sha256", None)
            digest = canonical_sha256(document)
            if known is not None and known != digest:
                raise ValueError("predicate evidence integrity mismatch")
            evidence.append({**document, "sha256": digest})
        object.__setattr__(self, "evidence", tuple(evidence))
        core = {
            "schemaVersion": 1,
            "status": self.status,
            "satisfied": self.satisfied,
            "evidence": [dict(item) for item in evidence],
            "namedCore": list(self.named_core),
            "modelHash": self.model_hash,
        }
        object.__setattr__(self, "sha256", canonical_sha256(core))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "status": self.status,
            "satisfied": self.satisfied,
            "evidence": [copy.deepcopy(dict(item)) for item in self.evidence],
            "namedCore": list(self.named_core),
            "modelHash": self.model_hash,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class OrToolsResearchArtifactPreimagesV1:
    """Defensive canonical copies of every authority-bound semantic artifact."""

    _canonical_json: str = field(repr=False)

    @classmethod
    def build(
        cls,
        *,
        authority: Mapping[str, Any],
        construction_model: Mapping[str, Any],
        proof_model: Mapping[str, Any],
        proof_graph: Mapping[str, Any],
        trace: Mapping[str, Any],
        proof_outcome: Mapping[str, Any],
    ) -> "OrToolsResearchArtifactPreimagesV1":
        documents = {
            "constructionModel": copy.deepcopy(dict(construction_model)),
            "proofModel": copy.deepcopy(dict(proof_model)),
            "proofGraph": copy.deepcopy(dict(proof_graph)),
            "trace": copy.deepcopy(dict(trace)),
            "proofOutcome": copy.deepcopy(dict(proof_outcome)),
        }
        expected = {
            "constructionModel": authority.get("constructionModelHash"),
            "proofModel": authority.get("proofModelHash"),
            "proofGraph": authority.get("graphHash"),
            "trace": authority.get("traceHash"),
            "proofOutcome": authority.get("proofOutcomeHash"),
        }
        mismatches = sorted(
            name
            for name, document in documents.items()
            if canonical_sha256(document) != expected[name]
        )
        if mismatches:
            raise ValueError(
                "semantic artifact preimage mismatch: " + ", ".join(mismatches)
            )
        return cls(
            json.dumps(
                documents,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._canonical_json)


@dataclass(frozen=True, slots=True)
class OrToolsUnifiedResearchResultV1:
    status: str
    preflight: OrToolsResearchPreflightV1
    candidate: OrToolsNeutralCandidateV1 | None
    predicate_evaluation: OrToolsPredicateEvaluationV1 | None
    proof: Mapping[str, Any] | None
    research_authority: Mapping[str, Any] | None
    artifact_preimages: OrToolsResearchArtifactPreimagesV1 | None
    rejection_hashes: tuple[str, ...]
    execution_receipt: Mapping[str, Any]
    semantic_hash: str
    backend_id: str = BACKEND_ID
    backend_version: str = BACKEND_VERSION
    sha256: str = field(init=False)
    # Same-execution proof ownership survives IPC, never document serialization.
    _quality_verified_artifacts: frozenset[str] = field(
        default=frozenset(), init=False, repr=False, compare=False,
    )

    def __post_init__(self) -> None:
        if self.status not in EXECUTION_STATUSES:
            raise ValueError("Research execution status is unsupported")
        if self.backend_id not in {BACKEND_ID, "ortools_cp_sat_v3"}:
            raise ValueError("Research result backend is unsupported")
        core = {
            "schemaVersion": 1,
            "backendId": self.backend_id,
            "backendVersion": self.backend_version,
            "status": self.status,
            "preflight": self.preflight.to_dict(),
            "candidate": self.candidate.to_dict() if self.candidate else None,
            "predicateEvaluation": (
                self.predicate_evaluation.to_dict()
                if self.predicate_evaluation is not None
                else None
            ),
            "proof": dict(self.proof) if self.proof else None,
            "researchAuthority": (
                dict(self.research_authority)
                if self.research_authority
                else None
            ),
            "rejectionHashes": list(self.rejection_hashes),
            # Hashing is read-only; export below still returns a detached copy.
            "executionReceipt": dict(self.execution_receipt),
            "semanticHash": self.semantic_hash,
        }
        object.__setattr__(self, "sha256", canonical_sha256(core))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "backendId": self.backend_id,
            "backendVersion": self.backend_version,
            "status": self.status,
            "preflight": self.preflight.to_dict(),
            "candidate": self.candidate.to_dict() if self.candidate else None,
            "predicateEvaluation": (
                self.predicate_evaluation.to_dict()
                if self.predicate_evaluation is not None
                else None
            ),
            "proof": copy.deepcopy(dict(self.proof)) if self.proof else None,
            "researchAuthority": (
                copy.deepcopy(dict(self.research_authority))
                if self.research_authority
                else None
            ),
            "rejectionHashes": list(self.rejection_hashes),
            "executionReceipt": copy.deepcopy(dict(self.execution_receipt)),
            "semanticHash": self.semantic_hash,
            "sha256": self.sha256,
        }

    def semantic_artifact_documents(self) -> dict[str, Any]:
        if self.artifact_preimages is None:
            raise RuntimeError("Research result has no semantic artifact preimages")
        return self.artifact_preimages.to_dict()


@dataclass(frozen=True, slots=True)
class OrToolsResearchCampaignResultV1:
    status: str
    executions: tuple[OrToolsUnifiedResearchResultV1, ...]
    accepted_results: tuple[OrToolsUnifiedResearchResultV1, ...]
    variant_receipts: tuple[CampaignVariantReceiptV1, ...]
    no_good_vectors: tuple[tuple[int, ...], ...]
    requested_result_count: int
    maximum_candidates: int
    symmetry_policy_id: str
    distance_policy_id: str
    minimum_distance: int
    wall_time_seconds: float
    cpu_seconds: float
    coordinator_receipt: CampaignReceiptV1 | None = None
    semantic_hash: str = field(init=False)
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.status not in {
            "completed",
            "structurally_infeasible",
            "not_found_within_budget",
            "semantic_postcondition_miss",
            "proof_failed",
            "incomplete",
        }:
            raise ValueError("Research campaign status is unsupported")
        if self.symmetry_policy_id != "rules_preserving_v1":
            raise ValueError("Research campaign symmetry policy is unsupported")
        if self.distance_policy_id not in {
            "decision_state_card_hamming_v1",
            SEMANTIC_DISTANCE_POLICY_V1,
        }:
            raise ValueError("Research campaign distance policy is unsupported")
        if type(self.minimum_distance) is not int or self.minimum_distance < 1:
            raise ValueError("Research campaign minimum distance must be positive")
        semantic_core = {
            "schemaVersion": 1,
            "status": self.status,
            "requestedResultCount": self.requested_result_count,
            "maximumCandidates": self.maximum_candidates,
            "symmetryPolicyId": self.symmetry_policy_id,
            "distancePolicyId": self.distance_policy_id,
            "minimumDistance": self.minimum_distance,
            "executionSemanticHashes": [item.semantic_hash for item in self.executions],
            "acceptedSemanticHashes": [
                item.semantic_hash for item in self.accepted_results
            ],
            "noGoodHashes": [
                canonical_sha256(list(item)) for item in self.no_good_vectors
            ],
            "variantReceipts": [item.to_dict() for item in self.variant_receipts],
            "coordinatorSemanticHash": (
                self.coordinator_receipt.semantic_sha256
                if self.coordinator_receipt is not None
                else None
            ),
        }
        semantic_hash = canonical_sha256(semantic_core)
        receipt_core = {
            **semantic_core,
            "semanticHash": semantic_hash,
            "wallTimeSeconds": self.wall_time_seconds,
            "cpuSeconds": self.cpu_seconds,
            "coordinatorReceipt": (
                self.coordinator_receipt.to_dict()
                if self.coordinator_receipt is not None
                else None
            ),
        }
        object.__setattr__(self, "semantic_hash", semantic_hash)
        object.__setattr__(self, "sha256", canonical_sha256(receipt_core))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "status": self.status,
            "requestedResultCount": self.requested_result_count,
            "maximumCandidates": self.maximum_candidates,
            "symmetryPolicyId": self.symmetry_policy_id,
            "distancePolicyId": self.distance_policy_id,
            "minimumDistance": self.minimum_distance,
            "completedCandidates": sum(
                int(item.execution_receipt["completedCandidates"])
                for item in self.executions
            ),
            "executionHashes": [item.sha256 for item in self.executions],
            "executionSemanticHashes": [item.semantic_hash for item in self.executions],
            "acceptedExecutionHashes": [item.sha256 for item in self.accepted_results],
            "acceptedSemanticHashes": [
                item.semantic_hash for item in self.accepted_results
            ],
            "noGoodVectors": [list(item) for item in self.no_good_vectors],
            "variantReceipts": [item.to_dict() for item in self.variant_receipts],
            "wallTimeSeconds": self.wall_time_seconds,
            "cpuSeconds": self.cpu_seconds,
            "coordinatorReceipt": (
                self.coordinator_receipt.to_dict()
                if self.coordinator_receipt is not None
                else None
            ),
            "semanticHash": self.semantic_hash,
            "sha256": self.sha256,
        }


def _campaign_variant_from_execution_v1(
    execution: OrToolsUnifiedResearchResultV1,
) -> CampaignVariantV1:
    candidate = execution.candidate
    if execution.status != "qualified" or candidate is None:
        raise ValueError("only qualified executions can become campaign variants")
    acting_seat = dict(candidate.trace.get("exactNormalization") or {}).get(
        "nextActor"
    )
    semantic_document = {
            "schemaVersion": 1,
            "semanticResultHash": execution.semantic_hash,
            "candidateHash": candidate.sha256,
            "decisionStateCardVectorRaw": list(
                decision_state_card_vector_v1(
                    candidate.state,
                    acting_seat=acting_seat,
                )
            ),
            "researchAuthorityHash": (
                execution.research_authority["sha256"]
                if execution.research_authority
                else None
            ),
        }
    if execution.backend_id == "ortools_cp_sat_v3":
        semantic_document["principalVariationActionIds"] = [
            str(action)
            for action in dict(execution.proof or {}).get("principalVariation") or ()
        ]
    return CampaignVariantV1.build(
        semantic_document,
        symmetry_document=build_rules_preserving_diversity_document_v1(
            candidate.state,
            acting_seat=acting_seat,
        ),
    )


def operator_candidate_progress_summary_v1(
    execution: OrToolsUnifiedResearchResultV1,
    *,
    decision_kind: str,
) -> dict[str, Any] | None:
    """Return observer-safe candidate facts suitable for live operator progress."""

    candidate = execution.candidate
    if candidate is None:
        return None
    state = dict(candidate.state)
    bidding = state.get("bidding")
    proof = dict(execution.proof or {})
    principal_variation = list(proof.get("principalVariation") or ())
    return {
        "schemaVersion": 1,
        "candidateHash": candidate.sha256,
        "exactStateId": candidate.exact_state_id,
        "decisionKind": decision_kind,
        "contract": candidate.trace.get("contract"),
        "actingSeat": (
            state.get("nextActor")
            or state.get("currentPlayer")
            or (
                dict(bidding).get("actor")
                if isinstance(bidding, Mapping)
                else None
            )
        ),
        "proofCompletionStatus": proof.get("completionStatus"),
        "winningRootActionCount": proof.get("winningRootActionCount"),
        "selectedAction": (
            str(principal_variation[0]) if principal_variation else None
        ),
    }


def _candidate_terminal_progress_v1(
    execution: OrToolsUnifiedResearchResultV1,
    *,
    ordinal: int,
    decision_kind: str,
) -> dict[str, Any]:
    candidate = execution.candidate
    return {
        "currentStage": "candidate_terminal",
        "executionOrdinal": ordinal,
        "candidateStatus": execution.status,
        "completedCandidates": int(execution.execution_receipt["completedCandidates"]),
        "qualifiedCandidates": int(execution.status == "qualified"),
        "candidateHash": candidate.sha256 if candidate is not None else None,
        "proofCompletionStatus": (
            execution.proof.get("completionStatus")
            if execution.proof is not None else None
        ),
        "incompleteCompilation": _campaign_incomplete_compilation_summary_v1(
            execution.execution_receipt.get("incompleteCompilation")
        ),
        "candidateAttemptReason": execution.execution_receipt.get(
            "candidateAttemptReason"
        ),
        "candidateSummary": operator_candidate_progress_summary_v1(
            execution, decision_kind=decision_kind
        ),
    }


def _campaign_variant_decision_cuts_v1(
    variant: CampaignVariantV1,
) -> tuple[tuple[int, ...], ...]:
    return (
        tuple(
            int(status)
            for status in variant.semantic_document["decisionStateCardVectorRaw"]
        ),
    )


@dataclass(frozen=True, slots=True)
class OrToolsCandidateAttemptV1:
    status: str
    candidate: OrToolsNeutralCandidateV1 | None
    reason: str | None = None
    graph: ProofGraphV1 | None = None
    proof_outcome: OrToolsGraphProofOutcomeV1 | None = None
    proof_state_projection_hash: str | None = None
    stage_timings: Mapping[str, float] = field(default_factory=dict)
    partial_receipt: Mapping[str, Any] | None = None
    predicate_evidence: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"feasible", "infeasible", "not_found", "incomplete"}:
            raise ValueError("candidate attempt status is unsupported")
        if (self.status == "feasible") != (self.candidate is not None):
            raise ValueError("only a feasible attempt may contain a candidate")
        if (self.graph is None) != (self.proof_outcome is None):
            raise ValueError(
                "candidate graph and proof outcome must be supplied together"
            )
        if self.proof_outcome is not None and self.status != "feasible":
            raise ValueError("only a feasible candidate may carry proof evidence")
        if self.partial_receipt is not None:
            document = copy.deepcopy(dict(self.partial_receipt))
            receipt_hash = document.pop("sha256", None)
            if receipt_hash != canonical_sha256(document):
                raise ValueError("candidate partial receipt hash mismatch")
            object.__setattr__(
                self,
                "partial_receipt",
                {**document, "sha256": receipt_hash},
            )
        normalized_evidence: dict[str, Mapping[str, Any]] = {}
        for predicate_id, raw in self.predicate_evidence.items():
            document = copy.deepcopy(dict(raw))
            evidence_hash = document.pop("sha256", None)
            if (
                predicate_id != document.get("predicateId")
                or evidence_hash != canonical_sha256(document)
            ):
                raise ValueError("candidate predicate evidence checksum mismatch")
            normalized_evidence[predicate_id] = {
                **document,
                "sha256": evidence_hash,
            }
        object.__setattr__(self, "predicate_evidence", normalized_evidence)


StageContextCompiler = Callable[
    [OrToolsNeutralCandidateV1, OrToolsGraphProofOutcomeV1 | None],
    Mapping[str, OrToolsStageContextBindingV1],
]
GraphCompiler = Callable[
    [OrToolsNeutralCandidateV1, ScenarioProofRequestV1], ProofGraphV1
]


def _canonical_no_good_vectors(
    values: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    normalized = tuple(tuple(item) for item in values)
    if any(
        len(item) != 32
        or any(type(card) is not int for card in item)
        or sorted(item) != list(range(32))
        for item in normalized
    ):
        raise ValueError("candidate no-good vectors must be canonical decks")
    return tuple(sorted(set(normalized)))


def _combined_expression(scenario: ResearchScenarioV1) -> ScenarioExpressionV1:
    return ScenarioExpressionV1.build(
        {
            "op": "all",
            "children": [
                scenario.constraints.to_dict(),
                scenario.outcome.to_dict(),
            ],
        }
    )


def _construction_contracts(
    scenario: ResearchScenarioV1,
) -> tuple[Contract, ...]:
    """Return the exact contract projection of the hard scenario expression."""

    def possible(
        node: ScenarioExpressionV1, contract: Contract
    ) -> frozenset[bool] | None:
        if node.predicate is not None:
            predicate = node.predicate
            if predicate.hardness != "hard":
                return None
            if predicate.kind == "auction.contract":
                return frozenset(
                    {Contract.parse(predicate.arguments["contract"]) == contract}
                )
            return frozenset({False, True})

        children = tuple(
            value
            for child in node.children
            if (value := possible(child, contract)) is not None
        )
        if not children:
            return None
        if len(children) == 1 and node.op in {"all", "any"}:
            return children[0]
        if node.op == "all":
            return frozenset(
                value
                for value, admitted in (
                    (True, all(True in child for child in children)),
                    (False, any(False in child for child in children)),
                )
                if admitted
            )
        if node.op == "any":
            return frozenset(
                value
                for value, admitted in (
                    (True, any(True in child for child in children)),
                    (False, all(False in child for child in children)),
                )
                if admitted
            )
        if node.op == "not":
            return frozenset(not value for value in children[0])
        if node.op == "count":
            if node.minimum is None or node.maximum is None:
                raise ValueError("scenario count expression requires bounds")
            counts = {0}
            for child in children:
                counts = {count + int(value) for count in counts for value in child}
            return frozenset(node.minimum <= count <= node.maximum for count in counts)
        raise ValueError(f"unsupported scenario expression operator: {node.op}")

    expression = _combined_expression(scenario)
    return tuple(
        contract
        for contract in Contract
        if True in (possible(expression, contract) or frozenset({True}))
    )


def _reify_hard_qualification_expression(
    model: cp_model.CpModel,
    expression: ScenarioExpressionV1,
    predicate_literals: Mapping[str, Any],
) -> Any:
    """Project soft/report leaves out before reifying hard qualification."""

    serial = 0

    def next_name(prefix: str, node: ScenarioExpressionV1) -> str:
        nonlocal serial
        serial += 1
        return f"{prefix}_{node.sha256[:12]}_{serial}"

    def lower(node: ScenarioExpressionV1) -> Any | None:
        if node.predicate is not None:
            return predicate_literals.get(node.predicate.predicate_id)

        children = tuple(
            literal for child in node.children if (literal := lower(child)) is not None
        )
        if not children:
            return None
        if len(children) == 1 and node.op in {"all", "any"}:
            return children[0]

        result = model.new_bool_var(next_name("hard_expr", node))
        if node.op == "all":
            model.add_min_equality(result, children)
        elif node.op == "any":
            model.add_max_equality(result, children)
        elif node.op == "not":
            model.add(result + children[0] == 1)
        elif node.op == "count":
            if node.minimum is None or node.maximum is None:
                raise ValueError("scenario count expression requires bounds")
            count = model.new_int_var(0, len(children), next_name("hard_count", node))
            model.add(count == sum(children))
            model.add_allowed_assignments(
                [count, result],
                [
                    (value, int(node.minimum <= value <= node.maximum))
                    for value in range(len(children) + 1)
                ],
            )
        else:
            raise ValueError(f"unsupported scenario expression operator: {node.op}")
        return result

    result = lower(expression)
    return result if result is not None else model.new_constant(1)


def _enforce_hard_stage_projection(
    model: cp_model.CpModel,
    scenario: ResearchScenarioV1,
    *,
    stage: str,
    predicate_literals: Mapping[str, Any],
) -> tuple[str, ...]:
    """Enforce the exact existential projection of one hard stage.

    Hard leaves owned by later stages remain free Boolean variables.  This keeps
    mixed ``any``/``not``/``count`` expressions exact without forcing every leaf
    as though the recipe were a conjunction.
    """

    hard = tuple(
        sorted(
            (
                predicate
                for predicate in scenario.predicates()
                if predicate.hardness == "hard"
            ),
            key=lambda predicate: predicate.predicate_id,
        )
    )
    stage_ids = tuple(
        predicate.predicate_id
        for predicate in hard
        if predicate.predicate_id in predicate_literals
    )
    if not stage_ids:
        return ()
    projected = dict(predicate_literals)
    for predicate in hard:
        projected.setdefault(
            predicate.predicate_id,
            model.new_bool_var(
                f"deferred_{stage}_{predicate.predicate_id}_{predicate.sha256[:8]}"
            ),
        )
    expression = _combined_expression(scenario)
    result = _reify_hard_qualification_expression(model, expression, projected)
    assumption = model.new_bool_var(f"assume_{stage}_{expression.sha256[:16]}")
    model.add(result == 1).only_enforce_if(assumption)
    model.add_assumption(assumption)
    return stage_ids


def _eligible_escalation_fingerprints(
    stage: OrToolsEscalationStageV1,
    scenario: ResearchScenarioV1,
    predicates: Sequence[ScenarioPredicateV1],
    *,
    maximum_seconds: float,
    maximum_conflicts: int,
) -> tuple[str, ...] | None:
    """Return the exact finite escalation projection, or ``None`` on budget."""

    manifest = shared_scenario_predicate_catalog_lowering_manifest_v2()
    candidates = stage._candidates()
    retained: list[str] = []
    per_candidate_seconds = maximum_seconds / max(1, len(candidates))
    for index, candidate in enumerate(candidates):
        context = CpSatEscalationPredicateContextV1.attach(stage)
        literals = {
            predicate.predicate_id: manifest.lower_predicate(predicate, context).literal
            for predicate in predicates
        }
        _enforce_hard_stage_projection(
            context.model,
            scenario,
            stage="escalation",
            predicate_literals=literals,
        )
        context.model.add(context.selected[index] == 1)
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.max_time_in_seconds = per_candidate_seconds
        solver.parameters.max_number_of_conflicts = maximum_conflicts
        status = solver.solve(context.model)
        if status in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
            retained.append(candidate.fingerprint)
        elif status != cp_model.INFEASIBLE:
            return None
    return tuple(retained)


def _project_count_upper_bound(rules: RuleBook, team: Team) -> int:
    cards = tuple(Card(index) for index in range(32))
    return max(
        sum(
            len(rules.project_domain.candidates(seat, cards, contract))
            for seat in Seat
            if seat.team == team
        )
        for contract in Contract
    )


def _project_predicate_is_tautology(
    predicate: ScenarioPredicateV1,
    rules: RuleBook,
) -> bool:
    """Recognize exact non-restrictive project counts before building the fragment."""

    if predicate.kind != "project.count":
        return False
    arguments = dict(predicate.arguments)
    if int(arguments["minimum"]) != 0:
        return False
    team = Team[str(arguments["team"]).upper()]
    return int(arguments["maximum"]) >= _project_count_upper_bound(rules, team)


def _project_count_projection_is_infeasible(
    scenario: ResearchScenarioV1,
    predicates: Sequence[ScenarioPredicateV1],
    rules: RuleBook,
    *,
    maximum_conflicts: int,
    maximum_seconds: float | None = None,
) -> bool:
    """Prove contradictions among shared project counts before deal allocation."""

    count_predicates = tuple(
        predicate for predicate in predicates if predicate.kind == "project.count"
    )
    if not count_predicates:
        return False
    model = cp_model.CpModel()
    counts: dict[tuple[Team, str], Any] = {}
    literals: dict[str, Any] = {}
    for predicate in count_predicates:
        arguments = dict(predicate.arguments)
        team = Team[str(arguments["team"]).upper()]
        state = str(arguments.get("state") or "eligible")
        key = (team, state)
        upper_bound = _project_count_upper_bound(rules, team)
        if key not in counts:
            counts[key] = model.new_int_var(
                0,
                upper_bound,
                f"project_count_{team.name.lower()}_{state}",
            )
        count = counts[key]
        literal = model.new_bool_var(
            f"project_count_range_{predicate.predicate_id}_{predicate.sha256[:8]}"
        )
        minimum = int(arguments["minimum"])
        maximum = int(arguments["maximum"])
        model.add_allowed_assignments(
            [count, literal],
            [
                (value, int(minimum <= value <= maximum))
                for value in range(upper_bound + 1)
            ],
        )
        literals[predicate.predicate_id] = literal
    _enforce_hard_stage_projection(
        model,
        scenario,
        stage="project_count_preflight",
        predicate_literals=literals,
    )
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_number_of_conflicts = maximum_conflicts
    if maximum_seconds is not None:
        solver.parameters.max_time_in_seconds = max(0.001, maximum_seconds)
    return solver.solve(model) == cp_model.INFEASIBLE


def _hard_expression_requires_predicate(
    scenario: ResearchScenarioV1,
    predicate_id: str,
    *,
    maximum_conflicts: int = 250_000,
    maximum_seconds: float | None = None,
    _require_complete: bool = False,
    cancel_check: Callable[[], bool] | None = None,
) -> bool:
    model = cp_model.CpModel()
    literals = {
        predicate.predicate_id: model.new_bool_var(
            f"required_leaf_{predicate.predicate_id}_{predicate.sha256[:8]}"
        )
        for predicate in scenario.predicates()
        if predicate.hardness == "hard"
    }
    target = literals.get(predicate_id)
    if target is None:
        return False
    result = _reify_hard_qualification_expression(
        model, _combined_expression(scenario), literals
    )
    model.add(result == 1)
    model.add(target == 0)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_number_of_conflicts = maximum_conflicts
    if maximum_seconds is not None:
        solver.parameters.max_time_in_seconds = max(0.001, maximum_seconds)
    status = _solve_cp_sat_with_cancel_v1(solver, model, cancel_check)
    if _require_complete and cancel_check is not None and cancel_check():
        raise InformationSetCompilationStopped("cancelled", stage="initial_state_inputs")
    if _require_complete and status not in {cp_model.INFEASIBLE, cp_model.OPTIMAL, cp_model.FEASIBLE}:
        raise InformationSetCompilationStopped(
            "expression_projection_incomplete", stage="initial_state_inputs",
            evidence={"solverStatus": solver.status_name(status).lower()},
        )
    return status == cp_model.INFEASIBLE


def _required_search_initial_inputs_v3(
    scenario: ResearchScenarioV1, *, rules: RuleBook, deadline: float, maximum_conflicts: int,
    cancel_check: Callable[[], bool] | None = None,
    excluded_inputs: Sequence[tuple[int, int, int]] = (),
) -> tuple[Seat, tuple[int, int] | None, bool]:
    dealer = Seat.EAST
    match_score = [0, 0]
    score_bounds: dict[int, list[tuple[int, int]]] = {0: [], 1: []}
    dealer_choices: set[Seat] = set()
    inputs = tuple(predicate for predicate in scenario.predicates()
        if predicate.hardness == "hard" and predicate.kind in {"auction.dealer", "state.match_score"})
    partial_choice = False
    for predicate in inputs:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or (cancel_check is not None and cancel_check()):
            raise InformationSetCompilationStopped(
                "maximum_seconds" if remaining <= 0 else "cancelled", stage="initial_state_inputs")
        required = _hard_expression_requires_predicate(scenario, predicate.predicate_id,
            maximum_conflicts=maximum_conflicts, maximum_seconds=remaining,
            _require_complete=True, cancel_check=cancel_check)
        if time.monotonic() >= deadline:
            raise InformationSetCompilationStopped("maximum_seconds", stage="initial_state_inputs")
        if not required:
            partial_choice = True
        else:
            if predicate.kind == "auction.dealer":
                dealer = Seat.parse(predicate.arguments["seat"])
                dealer_choices.add(dealer)
            else:
                team = Team[str(predicate.arguments["team"]).upper()]
                lower, upper = (int(predicate.arguments[key]) for key in ("minimum", "maximum"))
                score_bounds[int(team)].append((lower, upper))
                match_score[int(team)] = lower
                partial_choice |= lower != upper
    target = ScoringDomainIRV1.from_rules(rules.data).match_target
    score_proposals = {0, EscalationDomainIRV1.from_rules(rules.data).sun_score_threshold + 1} if (
        scenario.decision_point["kind"] == "escalation" or any(predicate.hardness == "hard"
            and predicate.kind.startswith("escalation.") for predicate in scenario.predicates())) else set()
    if (not partial_choice and len(dealer_choices) <= 1
            and all(len(set(bounds)) <= 1 for bounds in score_bounds.values())
            and (max(match_score) < target or match_score[0] == match_score[1])):
        # Unspecified search scores are construction preferences, not an exhaustive universe.
        omitted = any(not bounds for bounds in score_bounds.values())
        if (int(dealer), *match_score) not in excluded_inputs or not (omitted and score_proposals):
            return (dealer, None if (int(dealer), *match_score) in excluded_inputs else tuple(match_score), omitted)
    # shortcut: finite representatives; governed score bounds are needed for exhaustive cross-stage search.
    model = cp_model.CpModel()
    dealer_variable = model.new_int_var(0, 3, "initial_dealer") if any(
        predicate.kind == "auction.dealer" for predicate in inputs) else model.new_constant(int(Seat.EAST))
    score_predicates = tuple(predicate for predicate in inputs if predicate.kind == "state.match_score")
    # One representative beyond every interval covers complements, not the full game universe.
    upper = max(target, *score_proposals, *(int(predicate.arguments["maximum"]) + 1
        for predicate in score_predicates)) if score_predicates or score_proposals else 0
    scores = tuple(model.new_int_var(0, upper, f"initial_match_score_{team}") for team in range(2))
    if score_proposals:
        # Propose threshold crossings; the native escalation model still decides legality.
        representatives = score_proposals | {int(predicate.arguments[key]) + offset
            for predicate in score_predicates for key, offset in (("minimum", 0), ("maximum", 0), ("maximum", 1))
            if int(predicate.arguments[key]) + offset >= target}
        for team, score in enumerate(scores):
            if not any(int(Team[str(predicate.arguments["team"]).upper()]) == team for predicate in score_predicates):
                model.add_allowed_assignments([score], [(value,) for value in sorted(representatives)])
    if excluded_inputs:
        model.add_forbidden_assignments([dealer_variable, *scores], excluded_inputs)
    tied = _state_range_literal(model, scores[0] - scores[1], 0, 0, "initial_match_tied")
    for score in scores:
        model.add(score < target).only_enforce_if(tied.negated())
    literals = {}
    for predicate in inputs:
        if predicate.kind == "auction.dealer":
            value = int(Seat.parse(predicate.arguments["seat"]))
            literal = _state_range_literal(model, dealer_variable, value, value, predicate.predicate_id)
        else:
            literal = _state_range_literal(model, scores[int(Team[str(predicate.arguments["team"]).upper()])],
                int(predicate.arguments["minimum"]), int(predicate.arguments["maximum"]), predicate.predicate_id)
        literals[predicate.predicate_id] = literal
    _enforce_hard_stage_projection(model, scenario, stage="initial_state_inputs", predicate_literals=literals)
    default_dealer = _state_range_literal(model, dealer_variable, int(Seat.EAST), int(Seat.EAST), "initial_default_dealer")
    model.minimize(sum(scores) + 1 - default_dealer)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise InformationSetCompilationStopped("maximum_seconds", stage="initial_state_inputs")
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_number_of_conflicts = maximum_conflicts
    solver.parameters.max_time_in_seconds = remaining
    status = _solve_cp_sat_with_cancel_v1(solver, model, cancel_check)
    if cancel_check is not None and cancel_check():
        raise InformationSetCompilationStopped("cancelled", stage="initial_state_inputs")
    if time.monotonic() >= deadline:
        raise InformationSetCompilationStopped("maximum_seconds", stage="initial_state_inputs")
    if status == cp_model.INFEASIBLE:
        return dealer, None, False
    if status not in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
        raise InformationSetCompilationStopped("expression_projection_incomplete", stage="initial_state_inputs",
            evidence={"solverStatus": solver.status_name(status).lower()})
    return Seat(int(solver.value(dealer_variable))), tuple(int(solver.value(score)) for score in scores), True


def _project_exists_witness_keys(
    predicate: ScenarioPredicateV1,
    rules: RuleBook,
    contract: Contract,
    dealer: Seat,
) -> tuple[tuple[int, str, int, tuple[int, ...]], ...]:
    arguments = dict(predicate.arguments)
    if arguments.get("owner") is not None:
        seats = (Seat.parse(arguments["owner"]),)
    else:
        team = Team[str(arguments["team"]).upper()]
        seats = tuple(seat for seat in Seat if seat.team == team)
    declaration_order = rules.project_domain.declaration_seats(dealer)
    declaration_priority = {
        seat: len(declaration_order) - index
        for index, seat in enumerate(declaration_order)
    }
    matches = tuple(
        project
        for seat in seats
        for project in rules.project_domain.candidates(
            seat, tuple(Card(i) for i in range(32)), contract
        )
        if project.kind == arguments["kind"]
        and project.value == int(arguments["value"])
    )
    ordered = sorted(
        matches,
        key=lambda project: (
            project.value,
            project.category_priority,
            project.high_rank,
            declaration_priority[project.owner],
            tuple(card.id for card in project.cards),
        ),
        reverse=True,
    )
    return tuple(
        (
            int(project.owner),
            project.kind,
            project.value,
            tuple(card.id for card in project.cards),
        )
        for project in ordered
    )


def _walk_not_nodes(
    expression: ScenarioExpressionV1,
) -> tuple[ScenarioExpressionV1, ...]:
    if expression.predicate is not None:
        return ()
    return (
        *((expression,) if expression.op == "not" else ()),
        *(item for child in expression.children for item in _walk_not_nodes(child)),
    )


def _hard_expression_is_cross_stage_conjunction(
    expression: ScenarioExpressionV1,
) -> bool:
    """Whether hard leaves may be enforced independently without changing truth."""

    if expression.predicate is not None:
        return True
    hard_children = tuple(
        child
        for child in expression.children
        if any(predicate.hardness == "hard" for predicate in child.predicates())
    )
    if len(hard_children) <= 1 and expression.op in {"all", "any"}:
        return all(
            _hard_expression_is_cross_stage_conjunction(child)
            for child in hard_children
        )
    return expression.op == "all" and all(
        _hard_expression_is_cross_stage_conjunction(child)
        for child in hard_children
    )


class OrToolsResearchCompilerV1:
    """Preflight and exact cross-stage expression compiler."""

    def __init__(
        self,
        manifest: (
            ScenarioPredicateCatalogLoweringManifestV2
            | ScenarioPredicateCatalogLoweringManifestV3
            | None
        ) = None,
        *,
        native_cp_sat_authority: bool = False,
        tactical_detector_profile: TacticalDetectorProfileV1 | None = None,
    ):
        self.manifest = manifest or shared_scenario_predicate_catalog_lowering_manifest_v2()
        self.native_cp_sat_authority = bool(native_cp_sat_authority)
        self.tactical_detector_profile = tactical_detector_profile

    def preflight(self, scenario: ResearchScenarioV1) -> OrToolsResearchPreflightV1:
        if not isinstance(scenario, ResearchScenarioV1):
            raise TypeError("Research compiler requires ResearchScenarioV1")
        predicates = scenario.predicates()
        unsupported: list[Mapping[str, str]] = []
        definitions: dict[str, Any] = {}
        for predicate in predicates:
            definition = self.manifest.definition(predicate.kind)
            definitions[predicate.predicate_id] = definition
            if definition is None:
                unsupported.append(
                    {"predicateId": predicate.predicate_id, "code": "unknown_predicate"}
                )
                continue
            support = (
                definition.cp_sat_support.search_status
                if scenario.origin.kind == "search"
                else definition.cp_sat_support.exact_input_status
            )
            if support == "unsupported":
                unsupported.append(
                    {
                        "predicateId": predicate.predicate_id,
                        "code": "unsupported_origin",
                    }
                )
            elif predicate.hardness not in definition.allowed_hardness:
                unsupported.append(
                    {
                        "predicateId": predicate.predicate_id,
                        "code": "unsupported_hardness",
                    }
                )
            elif self.native_cp_sat_authority and (
                causal_issue := hard_tactic_causal_request_issue_v1(
                    predicate.kind, predicate.hardness, predicate.arguments,
                    native_delayed_payoff=(self.tactical_detector_profile is not None
                        and self.tactical_detector_profile.native_delayed_gain_metric is not None),
                )
            ) is not None:
                unsupported.append({
                    "predicateId": predicate.predicate_id, "code": causal_issue,
                })
            elif (
                self.native_cp_sat_authority
                and definition.stage == "state"
                and not native_v3_state_decision_supported(
                    definition.kind, scenario.decision_point["kind"],
                )
            ):
                unsupported.append({
                    "predicateId": predicate.predicate_id,
                    "code": "predicate_native_nonplay_state_unavailable",
                })
        combined = _combined_expression(scenario)
        for node in _walk_not_nodes(combined):
            child_predicates = node.children[0].predicates()
            if len(child_predicates) != 1:
                unsupported.append(
                    {
                        "predicateId": f"EXPR-{node.sha256[:16]}",
                        "code": "unsafe_composite_not",
                    }
                )
            elif not getattr(
                definitions.get(child_predicates[0].predicate_id),
                "safe_complement",
                False,
            ):
                unsupported.append(
                    {
                        "predicateId": child_predicates[0].predicate_id,
                        "code": "unsafe_not",
                    }
                )
        hard = tuple(
            sorted(item.predicate_id for item in predicates if item.hardness == "hard")
        )
        soft = tuple(
            sorted(item.predicate_id for item in predicates if item.hardness != "hard")
        )
        stages = {
            definition.stage
            for definition in definitions.values()
            if definition is not None
        }
        ordered = tuple(sorted(stages, key=lambda item: (_STAGE_PRIORITY[item], item)))
        start = ordered[0] if ordered else "deal"
        return OrToolsResearchPreflightV1(
            scenario.origin.kind,
            start,
            ordered,
            hard,
            soft,
            tuple(
                sorted(
                    unsupported, key=lambda item: (item["predicateId"], item["code"])
                )
            ),
            self.manifest.manifest_hash,
        )

    def reject_input_projection(
        self,
        scenario: ResearchScenarioV1,
        candidate: OrToolsNeutralCandidateV1,
        bindings: Mapping[str, OrToolsStageContextBindingV1],
        *,
        maximum_seconds: float,
        cancel_check: Callable[[], bool] | None = None,
    ) -> OrToolsPredicateEvaluationV1 | None:
        """Reject impossible fixed input facts; a partial projection never qualifies."""
        if not self.native_cp_sat_authority:
            raise ValueError("input projection requires native CP-SAT authority")
        if type(maximum_seconds) not in {int, float} or maximum_seconds <= 0:
            raise ValueError("predicate solve budget must be positive")
        deadline = time.monotonic() + float(maximum_seconds)
        groups: dict[int, tuple[cp_model.CpModel, dict[str, Any]]] = {}
        for predicate in scenario.predicates():
            definition = self.manifest.definition(predicate.kind)
            if definition is None:
                raise ValueError(f"predicate has no catalog definition: {predicate.kind}")
            if predicate.hardness != "hard" or definition.stage in _INPUT_PROJECTION_DEFERRED_STAGES:
                continue
            binding = bindings.get(definition.stage)
            if binding is None:
                continue
            receipt = candidate.stage_receipts.get(definition.stage)
            if receipt is None or receipt.evidence_hash != binding.source_evidence_hash:
                raise ValueError(f"predicate stage binding does not own candidate evidence: {definition.stage}")
            lowering = self.manifest.lower_predicate(predicate, binding.context)
            model = binding.context.model
            if id(model) not in groups:
                groups[id(model)] = (model, {})
            groups[id(model)][1][predicate.predicate_id] = lowering.literal
        hashes = []
        for model, literals in groups.values():
            _enforce_hard_stage_projection(model, scenario, stage="input", predicate_literals=literals)
            hashes.append(_hash_document({"proto": str(model.proto)}))
            remaining = deadline - time.monotonic()
            if remaining <= 0 or (cancel_check is not None and cancel_check()):
                return None
            solver = cp_model.CpSolver()
            solver.parameters.num_search_workers = 1
            solver.parameters.max_time_in_seconds = remaining
            status = _solve_cp_sat_with_cancel_v1(solver, model, cancel_check)
            if status == cp_model.INFEASIBLE:
                return OrToolsPredicateEvaluationV1(
                    "target_miss", False, (),
                    (f"EXPR-{_combined_expression(scenario).sha256[:16].upper()}",),
                    canonical_sha256(sorted(hashes)),
                )
            if status != cp_model.OPTIMAL:
                return None
        return None

    def evaluate(
        self,
        scenario: ResearchScenarioV1,
        candidate: OrToolsNeutralCandidateV1,
        bindings: Mapping[str, OrToolsStageContextBindingV1],
        *,
        maximum_seconds: float,
        cancel_check: Callable[[], bool] | None = None,
    ) -> OrToolsPredicateEvaluationV1:
        if type(maximum_seconds) not in {int, float} or maximum_seconds <= 0:
            raise ValueError("predicate solve budget must be positive")
        deadline = time.monotonic() + float(maximum_seconds)

        def remaining_seconds() -> float:
            if not self.native_cp_sat_authority:
                return float(maximum_seconds)
            return max(0.0, deadline - time.monotonic())

        lowerings: list[tuple[ScenarioPredicateV1, Any]] = []
        model_groups: dict[int, tuple[cp_model.CpModel, list[Any]]] = {}
        for predicate in scenario.predicates():
            definition = self.manifest.definition(predicate.kind)
            if definition is None:
                raise ValueError(
                    f"predicate has no catalog definition: {predicate.kind}"
                )
            binding = bindings.get(definition.stage)
            if binding is None:
                return OrToolsPredicateEvaluationV1(
                    "incomplete", False, (), (f"STAGE-{definition.stage}",), _HASH_ZERO
                )
            receipt = candidate.stage_receipts.get(definition.stage)
            if receipt is None or receipt.evidence_hash != binding.source_evidence_hash:
                raise ValueError(
                    f"predicate stage binding does not own candidate evidence: {definition.stage}"
                )
            try:
                lowering = None
                if (self.native_cp_sat_authority and predicate.kind in {"tactic.applicable", "tactic.verified"}
                        and predicate.arguments["kind"] in NATIVE_PROFILE_TACTIC_IDS
                        and getattr(binding.context, "native_cp_sat_authority", None) is not True):
                    raise ValueError("native tactical profile authority binding is unavailable")
                proof = getattr(binding.context, "proof", None)
                if (self.native_cp_sat_authority and predicate.kind in {"tactic.applicable", "tactic.verified"}
                        and predicate.arguments["kind"] in NATIVE_PROFILE_TACTIC_IDS
                        and isinstance(proof, OrToolsGraphProofOutcomeV1) and len(proof.root_actions) > 1):
                    profile = getattr(binding.context, "tactical_detector_profile", None)
                    if (not isinstance(profile, TacticalDetectorProfileV1)
                            or profile != self.tactical_detector_profile):
                        raise ValueError("native tactical profile does not match selected authority")
                    profile.validate()
                actor_tactic = (self.native_cp_sat_authority
                    and predicate.kind in {"tactic.applicable", "tactic.verified"}
                    and isinstance(proof, OrToolsGraphProofOutcomeV1)
                    and proof.proof_engine_id == "ortools_actor_information_set_cegis_v1")
                if actor_tactic and predicate.arguments["kind"] not in NATIVE_V3_ACTOR_ROOT_TACTIC_IDS:
                    raise ValueError("actor tactic has no native producer")
                actor_kabout = actor_tactic and predicate.arguments["kind"].startswith("kabout_")
                actor_small_card = actor_tactic and predicate.arguments["kind"] == "small_card_big_gain"
                if (actor_kabout or actor_small_card) and (proof.completion_status not in {"exact_complete", "proof_failed"}
                        or proof.model_document().get("qualificationSource") != "cp_sat_cegis_root_certificates_v2"):
                    raise ValueError("actor tactic requires exhaustive native root certificates")
                if (
                    self.native_cp_sat_authority
                    and (predicate.hardness == "hard" or actor_kabout or actor_small_card)
                    and (
                        predicate.kind in {
                            "proof.winning_root_action_count",
                            "proof.winning_root_card_rank_count",
                            "proof.winning_root_action_matches",
                        }
                        or predicate.kind in {"tactic.applicable", "tactic.verified"}
                        and predicate.arguments["kind"] in NATIVE_V3_ACTOR_ROOT_TACTIC_IDS
                    )
                    and isinstance(proof, OrToolsGraphProofOutcomeV1)
                    and proof.proof_engine_id == "ortools_actor_information_set_cegis_v1"
                    and proof.completion_status in {"exact_complete", "proof_failed"}
                    and proof.model_document().get("qualificationSource")
                    == "cp_sat_cegis_root_certificates_v2"
                ):
                    # CEGIS certifies root variables, not recipe-specific literals.
                    if receipt.evidence_hash != proof.sha256 or receipt.model_hash != proof.model_hash:
                        raise ValueError("actor root predicate does not own proof evidence")
                    source_model = cp_model.CpModel()
                    if (
                        not getattr(binding.context, "native_proof_bindings", False)
                        or not source_model.proto.parse_text_format(proof.model_document()["modelProto"])
                        or any(
                            tuple(map(str, tuple(getattr(binding.context.model.proto, field))[:len(getattr(source_model.proto, field))]))
                            != tuple(map(str, getattr(source_model.proto, field)))
                            for field in ("variables", "constraints")
                        )
                    ):
                        raise ValueError("actor root predicate model disagrees with proof")
                    winning, values = _proof_root_variables(
                        proof, binding.context.model, require_native=True
                    )
                    if any(
                        len(bound) != len(certified)
                        or any(variable is None for variable in bound)
                        or tuple(variable.index for variable in bound)
                        != tuple(variable.index for variable in certified)
                        for bound, certified in (
                            (binding.context.root_winning_variables, winning),
                            (binding.context.root_value_variables, values),
                        )
                    ):
                        raise ValueError("actor root predicate vectors disagree with proof")
                    if any(
                        tuple(getattr(item, field).index for item in binding.context.root_actions)
                        != tuple(variable.index for variable in variables)
                        for field, variables in (("winning", winning), ("value", values))
                    ) and (binding.context.root_actions or predicate.kind != "proof.winning_root_action_count"):
                        raise ValueError("actor root predicate bindings disagree with proof")
                    for index, action in enumerate(binding.context.root_actions):
                        if action.action_id != f"ROOT-{proof.root_actions[index].action_id}":
                            raise ValueError("actor root action identity disagrees with proof")
                        for field, suffix in (("card", "card"), ("actor", "actor"), ("original_owner", "owner"),
                                ("ekka", "ekka"), ("present", "applicable")):
                            indices = [slot for slot, variable in enumerate(binding.context.model.proto.variables)
                                if variable.name == f"qualification_root_{index}_{suffix}"]
                            if (len(indices) != 1 or getattr(action, field) is None
                                    or getattr(action, field).index != indices[0]):
                                raise ValueError("actor root primitive bindings disagree with proof")
                    if actor_kabout or actor_small_card:
                        context = binding.context
                        indices = tuple(tuple(index for index, variable in enumerate(context.model.proto.variables)
                            if variable.name == f"qualification_base_tricks_{team}") for team in range(2))
                        if ((actor_kabout and context.proof_utility != "kabout") or context.base_tricks is not None
                                or context.native_public_history_prefix is None
                                or any(len(found) != 1 for found in indices)
                                or len(context.base_trick_variables) != 2
                                or any(variable is None for variable in context.base_trick_variables)
                                or tuple(variable.index for variable in context.base_trick_variables)
                                != tuple(found[0] for found in indices)):
                            raise ValueError("actor tactic requires owned native public trick bindings")
                        variable_count, constraint_count, prefix_hash = context.native_public_history_prefix
                        if canonical_sha256({
                            "variables": list(map(str, tuple(context.model.proto.variables)[:variable_count])),
                            "constraints": list(map(str, tuple(context.model.proto.constraints)[:constraint_count])),
                            "cardStrengthById": list(context.card_strength_by_id),
                            **({"tacticalDetectorProfileHash": getattr(context.tactical_detector_profile, "sha256", None)} if context.tactical_detector_profile is not None else {}),
                        }) != prefix_hash:
                            raise ValueError("actor tactic public trick model disagrees with owned bindings")
                        if actor_small_card:
                            continuation = context.principal_variation_has_continuation_variable
                            indices = [index for index, variable in enumerate(context.model.proto.variables)
                                if variable.name == "qualification_principal_has_continuation"]
                            if (continuation is None or len(indices) != 1 or continuation.index != indices[0]
                                    or len(context.card_strength_by_id) != 32):
                                raise ValueError("actor small-card native continuation bindings disagree")
                            selected = context.selected_root_action_index
                            for field, name in (("root_actor_is_target_variable", "qualification_root_actor_is_target"),
                                    ("target_seat_variable", f"qualification_root_{selected}_actor"
                                     if proof.model_document()["rootActorTeam"] == proof.model_document()["targetTeam"]
                                     else "qualification_target_seat_absent")):
                                indices = [slot for slot, variable in enumerate(context.model.proto.variables)
                                    if variable.name == name]
                                variable = getattr(context, field)
                                if variable is None or len(indices) != 1 or variable.index != indices[0]:
                                    raise ValueError("actor small-card target bindings disagree")
                        if predicate.arguments["kind"] == "kabout_prevention":
                            stopping = _proof_stopping_variables(proof, context.model)
                            if (len(context.root_stopping_variables) != len(stopping)
                                    or any(variable is None for variable in context.root_stopping_variables)
                                    or tuple(variable.index for variable in context.root_stopping_variables)
                                    != tuple(variable.index for variable in stopping)):
                                raise ValueError("actor Kabout stopping bindings disagree with proof")
                    lowering = self.manifest.lower_predicate(predicate, binding.context)
                elif (
                    self.native_cp_sat_authority
                    and (predicate.hardness == "hard" or (
                        isinstance(proof, OrToolsGraphProofOutcomeV1)
                        and isinstance(proof.native_qualification, Mapping)
                        and predicate.predicate_id in proof.native_qualification
                    ))
                    and definition.stage in {"proof", "tactics"}
                ):
                    proof = getattr(binding.context, "proof", None)
                    if not isinstance(proof, OrToolsGraphProofOutcomeV1):
                        raise ValueError("native proof outcome binding is unavailable")
                    model_document = proof.model_document()
                    variable_names = model_document.get(
                        "nativeQualificationVariables", {}
                    )
                    variable_name = (
                        variable_names.get(predicate.predicate_id)
                        if isinstance(variable_names, Mapping)
                        else None
                    )
                    native_evidence = (
                        proof.native_qualification.get(predicate.predicate_id)
                        if isinstance(proof.native_qualification, Mapping)
                        else None
                    )
                    variables = {
                        variable.name: index
                        for index, variable in enumerate(
                            binding.context.model.proto.variables
                        )
                    }
                    variable_index = (
                        variables.get(variable_name)
                        if isinstance(variable_name, str)
                        else None
                    )
                    if variable_index is None or not isinstance(
                        native_evidence, Mapping
                    ):
                        raise ValueError(
                            "native proof predicate variable binding is unavailable"
                        )
                    if (predicate.kind in {"tactic.applicable", "tactic.verified"}
                            and predicate.arguments["kind"] in NATIVE_PROFILE_TACTIC_IDS and len(proof.root_actions) > 1
                            and dict(native_evidence.get("observed") or {}).get("detectorProfileHash") != profile.sha256):
                        raise ValueError("native tactical proof profile binding disagrees")
                    literal = binding.context.model.get_bool_var_from_proto_index(
                        variable_index
                    )
                    evidence_snapshot = copy.deepcopy(dict(native_evidence))
                    lowering_hash = canonical_sha256(
                        {
                            "schemaVersion": 1,
                            "predicateHash": predicate.sha256,
                            "proofModelHash": proof.model_hash,
                            "literalVariable": variable_name,
                        }
                    )
                    lowering = CpSatPredicateLoweringV1(
                        predicate.predicate_id,
                        predicate.kind,
                        literal,
                        "cp_sat.native_proof_same_solve_v1",
                        lowering_hash,
                        lambda _solver, value=evidence_snapshot: copy.deepcopy(value),
                    )
                if lowering is None:
                    lowering = self.manifest.lower_predicate(
                        predicate, binding.context
                    )
            except (TypeError, ValueError):
                return OrToolsPredicateEvaluationV1(
                    "incomplete",
                    False,
                    (),
                    (f"PREDICATE-{predicate.predicate_id}",),
                    _HASH_ZERO,
                )
            lowerings.append((predicate, lowering))
            key = id(binding.context.model)
            if key not in model_groups:
                model_groups[key] = (binding.context.model, [])
            model_groups[key][1].append(lowering)

        if self.native_cp_sat_authority:
            expression = _combined_expression(scenario)
            hard_lowerings = {
                predicate.predicate_id: (predicate, lowering)
                for predicate, lowering in lowerings
                if predicate.hardness == "hard"
            }
            hard_models = {
                id(bindings[self.manifest.definition(predicate.kind).stage].context.model)
                for predicate, _lowering in lowerings
                if predicate.hardness == "hard"
            }
            if len(hard_models) <= 1:
                target_model = (
                    model_groups[next(iter(hard_models))][0]
                    if hard_models else next(iter(model_groups.values()))[0]
                )
                result_literal = _reify_hard_qualification_expression(
                    target_model,
                    expression,
                    {
                        predicate_id: lowering.literal
                        for predicate_id, (_predicate, lowering) in hard_lowerings.items()
                    },
                )
                target_model.add(result_literal == 1)
            elif _hard_expression_is_cross_stage_conjunction(expression):
                for predicate, lowering in hard_lowerings.values():
                    definition = self.manifest.definition(predicate.kind)
                    assert definition is not None
                    bindings[definition.stage].context.model.add(
                        lowering.literal == 1
                    )
            else:
                if remaining_seconds() <= 0:
                    return OrToolsPredicateEvaluationV1(
                        "incomplete", False, (), (), _HASH_ZERO
                    )
                return self._evaluate_cross_stage_expression(
                    expression,
                    hard_lowerings,
                    lowerings,
                    model_groups,
                    maximum_seconds=remaining_seconds(),
                    cancel_check=cancel_check,
                )

            evidence_by_id: dict[str, Mapping[str, Any]] = {}
            model_hashes: list[str] = []
            for model, rows in model_groups.values():
                model_hashes.append(_hash_document({"proto": str(model.proto)}))
                remaining = remaining_seconds()
                if remaining <= 0:
                    return OrToolsPredicateEvaluationV1(
                        "incomplete", False, (), (), canonical_sha256(sorted(model_hashes))
                    )
                solver = cp_model.CpSolver()
                solver.parameters.num_search_workers = 1
                solver.parameters.max_time_in_seconds = remaining
                status = _solve_cp_sat_with_cancel_v1(solver, model, cancel_check)
                if status == cp_model.INFEASIBLE:
                    return OrToolsPredicateEvaluationV1(
                        "target_miss",
                        False,
                        (),
                        (f"EXPR-{expression.sha256[:16].upper()}",),
                        canonical_sha256(sorted(model_hashes)),
                    )
                if status != cp_model.OPTIMAL:
                    return OrToolsPredicateEvaluationV1(
                        "incomplete",
                        False,
                        (),
                        (),
                        canonical_sha256(sorted(model_hashes)),
                    )
                for lowering in rows:
                    evidence_by_id[lowering.predicate_id] = lowering.extract(solver)
            return OrToolsPredicateEvaluationV1(
                "satisfied",
                True,
                tuple(evidence_by_id[key] for key in sorted(evidence_by_id)),
                (),
                canonical_sha256(sorted(model_hashes)),
            )

        evidence_by_id: dict[str, Mapping[str, Any]] = {}
        for model, rows in model_groups.values():
            remaining = remaining_seconds()
            if remaining <= 0:
                return OrToolsPredicateEvaluationV1(
                    "incomplete", False, (), (), _HASH_ZERO
                )
            solver = cp_model.CpSolver()
            solver.parameters.num_search_workers = 1
            solver.parameters.max_time_in_seconds = remaining
            status = _solve_cp_sat_with_cancel_v1(solver, model, cancel_check)
            if status == cp_model.INFEASIBLE:
                return OrToolsPredicateEvaluationV1(
                    "infeasible",
                    False,
                    (),
                    ("STAGE-BINDING",),
                    _hash_document({"proto": str(model.proto)}),
                )
            if status != cp_model.OPTIMAL:
                return OrToolsPredicateEvaluationV1(
                    "incomplete",
                    False,
                    (),
                    (),
                    _hash_document({"proto": str(model.proto)}),
                )
            for lowering in rows:
                evidence_by_id[lowering.predicate_id] = lowering.extract(solver)

        expression = _combined_expression(scenario)
        final_model = cp_model.CpModel()
        predicate_literals: dict[str, Any] = {}
        for predicate, _lowering in lowerings:
            evidence = evidence_by_id[predicate.predicate_id]
            actual = bool(evidence["satisfied"])
            if predicate.hardness == "hard":
                predicate_literals[predicate.predicate_id] = final_model.new_constant(
                    int(actual)
                )
        result_literal = _reify_hard_qualification_expression(
            final_model, expression, predicate_literals
        )
        assumption = final_model.new_bool_var(f"assume_expr_{expression.sha256[:16]}")
        final_model.add(result_literal == 1).only_enforce_if(assumption)
        final_model.add_assumption(assumption)
        model_hash = _hash_document(
            {
                "schemaVersion": 1,
                "expressionHash": expression.sha256,
                "predicateEvidenceHashes": {
                    key: value["sha256"]
                    for key, value in sorted(evidence_by_id.items())
                },
                "proto": str(final_model.proto),
            }
        )
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        ordered_evidence = tuple(evidence_by_id[key] for key in sorted(evidence_by_id))
        remaining = remaining_seconds()
        if remaining <= 0:
            return OrToolsPredicateEvaluationV1(
                "incomplete", False, ordered_evidence, (), model_hash
            )
        solver.parameters.max_time_in_seconds = remaining
        status = _solve_cp_sat_with_cancel_v1(solver, final_model, cancel_check)
        if status == cp_model.INFEASIBLE:
            return OrToolsPredicateEvaluationV1(
                "target_miss",
                False,
                ordered_evidence,
                (f"EXPR-{expression.sha256[:16].upper()}",),
                model_hash,
            )
        if status != cp_model.OPTIMAL:
            return OrToolsPredicateEvaluationV1(
                "incomplete", False, ordered_evidence, (), model_hash
            )
        return OrToolsPredicateEvaluationV1(
            "satisfied", True, ordered_evidence, (), model_hash
        )

    def _evaluate_cross_stage_expression(
        self,
        expression: ScenarioExpressionV1,
        hard_lowerings: Mapping[str, tuple[ScenarioPredicateV1, Any]],
        lowerings: Sequence[tuple[ScenarioPredicateV1, Any]],
        model_groups: Mapping[int, tuple[cp_model.CpModel, list[Any]]],
        *,
        maximum_seconds: float,
        cancel_check: Callable[[], bool] | None,
    ) -> OrToolsPredicateEvaluationV1:
        """Join independent native stage models without copying semantic truth."""

        master = cp_model.CpModel()
        master_literals = {
            predicate_id: master.new_bool_var(f"cross_stage_{predicate_id}")
            for predicate_id in sorted(hard_lowerings)
        }
        result = _reify_hard_qualification_expression(
            master, expression, master_literals
        )
        master.add(result == 1)
        deadline = time.monotonic() + maximum_seconds
        predicate_by_id = {predicate.predicate_id: predicate for predicate, _ in lowerings}

        while True:
            if cancel_check is not None and cancel_check():
                return OrToolsPredicateEvaluationV1(
                    "incomplete", False, (), (), _HASH_ZERO
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return OrToolsPredicateEvaluationV1(
                    "incomplete", False, (), (), _HASH_ZERO
                )
            master_solver = cp_model.CpSolver()
            master_solver.parameters.num_search_workers = 1
            master_solver.parameters.max_time_in_seconds = remaining
            master_status = _solve_cp_sat_with_cancel_v1(
                master_solver, master, cancel_check
            )
            master_hash = _hash_document({"proto": str(master.proto)})
            if master_status == cp_model.INFEASIBLE:
                return OrToolsPredicateEvaluationV1(
                    "target_miss",
                    False,
                    (),
                    (f"EXPR-{expression.sha256[:16].upper()}",),
                    master_hash,
                )
            if master_status != cp_model.OPTIMAL:
                return OrToolsPredicateEvaluationV1(
                    "incomplete", False, (), (), master_hash
                )
            assignment = {
                predicate_id: int(master_solver.boolean_value(literal))
                for predicate_id, literal in master_literals.items()
            }
            solved_groups: list[tuple[cp_model.CpSolver, list[Any], str]] = []
            rejected = False
            for source_model, rows in model_groups.values():
                clone = source_model.clone()
                group_ids = [
                    row.predicate_id
                    for row in rows
                    if row.predicate_id in hard_lowerings
                ]
                for row in rows:
                    if row.predicate_id in assignment:
                        clone.add(
                            clone.get_bool_var_from_proto_index(row.literal.index)
                            == assignment[row.predicate_id]
                        )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return OrToolsPredicateEvaluationV1(
                        "incomplete", False, (), (), master_hash
                    )
                solver = cp_model.CpSolver()
                solver.parameters.num_search_workers = 1
                solver.parameters.max_time_in_seconds = remaining
                status = _solve_cp_sat_with_cancel_v1(solver, clone, cancel_check)
                clone_hash = _hash_document({"proto": str(clone.proto)})
                if status == cp_model.INFEASIBLE:
                    if not group_ids:
                        return OrToolsPredicateEvaluationV1(
                            "target_miss", False, (), ("STAGE-BINDING",), clone_hash
                        )
                    master.add_bool_or(
                        [
                            master_literals[predicate_id].negated()
                            if assignment[predicate_id]
                            else master_literals[predicate_id]
                            for predicate_id in group_ids
                        ]
                    )
                    rejected = True
                    break
                if status != cp_model.OPTIMAL:
                    return OrToolsPredicateEvaluationV1(
                        "incomplete", False, (), (), clone_hash
                    )
                solved_groups.append((solver, rows, clone_hash))
            if rejected:
                continue

            evidence_by_id: dict[str, Mapping[str, Any]] = {}
            model_hashes = [master_hash]
            for solver, rows, model_hash in solved_groups:
                model_hashes.append(model_hash)
                for lowering in rows:
                    evidence_by_id[lowering.predicate_id] = lowering.extract(solver)
            if set(evidence_by_id) != set(predicate_by_id):
                raise AssertionError("cross-stage predicate evidence is incomplete")
            return OrToolsPredicateEvaluationV1(
                "satisfied",
                True,
                tuple(evidence_by_id[key] for key in sorted(evidence_by_id)),
                (),
                canonical_sha256(sorted(model_hashes)),
            )


def _stage_receipt(
    stage: str,
    *,
    model_hash: str,
    evidence_hash: str,
    semantic: Mapping[str, Any],
    solver_status: str,
    branches: int = 0,
    conflicts: int = 0,
    variable_count: int | None = None,
    constraint_count: int | None = None,
    named_assumptions: Sequence[Mapping[str, Any]] = (),
    infeasible_core: Sequence[str] = (),
    evidence_bindings: Sequence[Mapping[str, Any]] = (),
    elapsed_seconds: float = 0.0,
) -> OrToolsResearchStageReceiptV1:
    model_size = semantic.get("modelSize")
    model_size = dict(model_size) if isinstance(model_size, Mapping) else {}
    if variable_count is None:
        variable_count = semantic.get(
            "variableCount", model_size.get("variables", 0)
        )
    if constraint_count is None:
        constraint_count = semantic.get(
            "constraintCount", model_size.get("constraints", 0)
        )
    return OrToolsResearchStageReceiptV1(
        stage,
        model_hash,
        evidence_hash,
        _hash_document(semantic),
        solver_status,
        int(branches),
        int(conflicts),
        int(variable_count),
        int(constraint_count),
        tuple(copy.deepcopy(dict(item)) for item in named_assumptions),
        tuple(infeasible_core),
        tuple(copy.deepcopy(dict(item)) for item in evidence_bindings),
        float(elapsed_seconds),
    )


def _cp_sat_receipt_metadata_v1(
    model: cp_model.CpModel,
    *,
    stage: str,
    predicates: Sequence[ScenarioPredicateV1],
    predicate_literals: Mapping[str, Any],
    model_hash: str,
    evidence_hash: str,
    lowering_manifest_hash: str,
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    predicate_ids = tuple(sorted(predicate.predicate_id for predicate in predicates))
    assumptions = []
    for raw_index in model.proto.assumptions:
        literal_index = int(raw_index if raw_index >= 0 else -raw_index - 1)
        literal_name = model.proto.variables[literal_index].name
        if not literal_name.startswith(f"assume_{stage}_"):
            continue
        assumptions.append(
            {
                "assumptionId": f"ASSUME-{stage.upper()}-{literal_index}",
                "literalName": literal_name,
                "literalIndex": literal_index,
                "predicateIds": list(predicate_ids),
            }
        )
    predicate_by_id = {predicate.predicate_id: predicate for predicate in predicates}
    bindings = []
    for predicate_id, literal in sorted(predicate_literals.items()):
        predicate = predicate_by_id.get(predicate_id)
        if predicate is None:
            continue
        literal_index = int(literal.index)
        bindings.append(
            {
                "predicateId": predicate_id,
                "sourceEvidenceHash": evidence_hash,
                "loweringHash": canonical_sha256(
                    {
                        "schemaVersion": 1,
                        "predicateHash": predicate.sha256,
                        "loweringManifestHash": lowering_manifest_hash,
                    }
                ),
                "modelHash": model_hash,
                "literalName": model.proto.variables[literal_index].name,
                "literalIndex": literal_index,
                "evidenceHash": None,
            }
        )
    return tuple(assumptions), tuple(bindings)


def _proof_receipt_evidence_bindings_v3(
    proof: OrToolsGraphProofOutcomeV1,
    request: ScenarioProofRequestV1,
) -> tuple[Mapping[str, Any], ...]:
    if proof.completion_status == "model_error" or (
        proof.completion_status == "proof_incomplete"
        and proof.solver_status != "optimal"
    ):
        # Unsolved model variables are not qualification evidence.
        return ()
    model = proof.model_document()
    raw_claim = model.get("claimEvidenceBinding")
    if raw_claim is None:
        if proof.completion_status in {"exact_complete", "proof_failed"}:
            raise ValueError("complete proof lacks a solver-owned claim certificate")
        return ()
    claim = dict(raw_claim)
    source_hash = proof.sha256
    bindings: list[Mapping[str, Any]] = [
        {
            "predicateId": "PROOF-CLAIM",
            "sourceEvidenceHash": source_hash,
            "loweringHash": canonical_sha256(
                {
                    "schemaVersion": 1,
                    "proofRequestHash": request.sha256,
                    "claim": request.claim,
                    "literalName": claim["variable"],
                    "literalIndex": claim["literalIndex"],
                }
            ),
            "modelHash": proof.model_hash,
            "literalName": claim["variable"],
            "literalIndex": int(claim["literalIndex"]),
            "evidenceHash": None,
        }
    ]
    native_names = dict(model.get("nativeQualificationVariables") or {})
    native_indexes = dict(model.get("nativeQualificationLiteralIndices") or {})
    native_evidence = dict(proof.native_qualification or {})
    for predicate_id, literal_name in sorted(native_names.items()):
        evidence = dict(native_evidence[predicate_id])
        bindings.append(
            {
                "predicateId": predicate_id,
                "sourceEvidenceHash": source_hash,
                "loweringHash": evidence["loweringHash"],
                "modelHash": proof.model_hash,
                "literalName": literal_name,
                "literalIndex": int(native_indexes[predicate_id]),
                "evidenceHash": evidence["sha256"],
            }
        )
    return tuple(bindings)


def _native_same_solve_evaluation_v3(
    scenario: ResearchScenarioV1,
    candidate: OrToolsNeutralCandidateV1,
    proof: OrToolsGraphProofOutcomeV1 | None,
    construction_evidence: Mapping[str, Mapping[str, Any]],
    tactical_detector_profile: TacticalDetectorProfileV1 | None = None,
) -> OrToolsPredicateEvaluationV1 | None:
    """Reuse immutable CP-SAT evidence when every hard leaf is conjunctive."""

    if "proofContinuation" in candidate.trace or "policyContinuation" in candidate.trace:
        # Construction receipts cannot qualify the canonical proof continuation.
        return None
    predicates = tuple(scenario.predicates())
    if (
        not predicates
        or any(predicate.hardness != "hard" for predicate in predicates)
        or not _hard_expression_is_cross_stage_conjunction(
            _combined_expression(scenario)
        )
    ):
        return None
    bindings = [
        (receipt, dict(binding))
        for receipt in candidate.stage_receipts.values()
        for binding in receipt.evidence_bindings
    ]
    proof_document = proof.model_document() if proof is not None else {}
    proof_variable_names = dict(
        proof_document.get("nativeQualificationVariables") or {}
    )
    proof_native = dict(proof.native_qualification or {}) if proof is not None else {}
    evidence: dict[str, Mapping[str, Any]] = {}
    for predicate in predicates:
        if (predicate.kind in {"tactic.applicable", "tactic.verified"}
                and predicate.arguments["kind"] in NATIVE_PROFILE_TACTIC_IDS
                and predicate.predicate_id in construction_evidence):
            return None
        if predicate.predicate_id in construction_evidence:
            item = copy.deepcopy(dict(construction_evidence[predicate.predicate_id]))
            binding_evidence_hash = item.get("sha256")
        else:
            native = proof_native.get(predicate.predicate_id)
            if (predicate.kind in {"tactic.applicable", "tactic.verified"}
                    and predicate.arguments["kind"] in NATIVE_PROFILE_TACTIC_IDS and (proof is None or len(proof.root_actions) > 1)
                    and (tactical_detector_profile is None or not isinstance(native, Mapping)
                         or dict(native.get("observed") or {}).get("detectorProfileHash") != tactical_detector_profile.sha256)):
                return None
            variable_name = proof_variable_names.get(predicate.predicate_id)
            if not isinstance(native, Mapping) or not isinstance(variable_name, str):
                return None
            observed = copy.deepcopy(dict(native))
            lowering_hash = canonical_sha256(
                {
                    "schemaVersion": 1,
                    "predicateHash": predicate.sha256,
                    "proofModelHash": proof.model_hash,
                    "literalVariable": variable_name,
                }
            )
            core = {
                "schemaVersion": 1,
                "predicateId": predicate.predicate_id,
                "kind": predicate.kind,
                "satisfied": bool(observed.get("satisfied")),
                "evidenceExtractor": "cp_sat.native_proof_same_solve_v1",
                "loweringHash": lowering_hash,
                "observed": observed,
            }
            item = {**core, "sha256": canonical_sha256(core)}
            binding_evidence_hash = observed.get("sha256")
        if item.get("satisfied") is not True:
            return None
        if not any(
            binding.get("predicateId") == predicate.predicate_id
            and binding.get("evidenceHash") == binding_evidence_hash
            and binding.get("sourceEvidenceHash") == receipt.evidence_hash
            and binding.get("modelHash") == receipt.model_hash
            for receipt, binding in bindings
        ):
            return None
        evidence[predicate.predicate_id] = item
    model_hash = canonical_sha256(
        {
            "schemaVersion": 1,
            "authority": "cp_sat.native_same_solve_receipts_v1",
            "predicateEvidenceHashes": {
                predicate_id: item["sha256"]
                for predicate_id, item in sorted(evidence.items())
            },
            "stageReceiptHashes": sorted(
                receipt.to_dict()["sha256"]
                for receipt in candidate.stage_receipts.values()
                if receipt.evidence_bindings
            ),
        }
    )
    return OrToolsPredicateEvaluationV1(
        "satisfied",
        True,
        tuple(evidence[key] for key in sorted(evidence)),
        (),
        model_hash,
    )


def _upgrade_candidate_receipts_v3_1(
    candidate: OrToolsNeutralCandidateV1,
) -> OrToolsNeutralCandidateV1:
    """Upgrade new V3.1 semantic receipts; timing stays in executionReceipt."""

    receipts = {
        stage: (
            replace(receipt, elapsed_seconds=0.0)
            if isinstance(receipt, OrToolsResearchStageReceiptV2)
            else OrToolsResearchStageReceiptV2(
                stage=receipt.stage,
                model_hash=receipt.model_hash,
                evidence_hash=receipt.evidence_hash,
                semantic_hash=receipt.semantic_hash,
                solver_status=receipt.solver_status,
                variable_count=receipt.variable_count,
                constraint_count=receipt.constraint_count,
                named_assumptions=receipt.named_assumptions,
                infeasible_core=receipt.infeasible_core,
                evidence_bindings=receipt.evidence_bindings,
                elapsed_seconds=0.0,
                branches=receipt.branches,
                conflicts=receipt.conflicts,
            )
        )
        for stage, receipt in candidate.stage_receipts.items()
    }
    return replace(candidate, stage_receipts=receipts)


def _decision_prefix_length(
    scenario: ResearchScenarioV1,
    plays: Sequence[Any],
) -> int:
    if scenario.decision_point["kind"] != "play":
        return 0
    selector = dict(scenario.decision_point["selector"])
    trick_index = selector.get("trickIndex")
    position = selector.get("position")
    remaining = selector.get("remainingCards")
    actor = Seat.parse(selector["actingSeat"]) if selector.get("actingSeat") else None
    indexes = range(33)
    if trick_index is not None:
        indexes = (index for index in indexes if index // 4 == int(trick_index))
    if position is not None:
        indexes = (index for index in indexes if index % 4 == int(position))
    if remaining is not None:
        indexes = (
            index
            for index in indexes
            if index < len(plays)
            if 8 - sum(
                1 for play in plays[:index]
                if play.actor == (actor if actor is not None else plays[index].actor)
            )
            == int(remaining)
        )
    if actor is not None:
        indexes = (
            index for index in indexes if index < 32 and plays[index].actor == actor
        )
    try:
        return next(iter(indexes))
    except StopIteration as exc:
        raise ValueError(
            "decision point does not occur in the constructed play trace"
        ) from exc


def _static_decision_prefix_length(scenario: ResearchScenarioV1) -> int | None:
    """Return the decision step when it does not depend on the solved play line."""

    if scenario.decision_point["kind"] != "play":
        return None
    selector = dict(scenario.decision_point["selector"])
    trick_index = selector.get("trickIndex")
    if trick_index is None:
        return 0 if not selector else None
    trick_index = int(trick_index)
    if "position" not in selector and selector.get("actingSeat"):
        return None
    position = int(selector.get("position", 0))
    remaining = selector.get("remainingCards")
    if remaining is not None and int(remaining) != 8 - trick_index:
        return None
    return trick_index * 4 + position


def _score_complete_candidate(
    *,
    rules: RuleBook,
    contract: Contract,
    buyer: Seat,
    escalation: OrToolsEscalationStageResultV1,
    projects: OrToolsProjectLifecycleResultV1,
    play: OrToolsPlayTailResultV1,
    starting_match_score: tuple[int, int],
    native_cp_sat: bool = False,
) -> dict[str, Any]:
    if play.raw_team_points is None or play.card_game_points is None:
        raise AssertionError("complete play lacks CP-SAT scoring evidence")
    scoring = ScoringDomainIRV1.from_rules(rules.data)
    observation = escalation.observation
    if observation is None:
        raise AssertionError("satisfied escalation stage lacks evidence")
    level = observation.level
    history = observation.actions
    if native_cp_sat:
        result = OrToolsScoringStageV1(scoring, native_conversion=True).solve(
            contract=contract,
            buyer=buyer,
            escalation=level,
            last_escalating_team=(history[-1].actor.team if history else None),
            raw_points=play.raw_team_points,
            project_points=projects.combined_awarded_value,
            baloot_points=projects.baloot_points,
            trick_winner_teams=tuple(seat.team for seat in play.trick_winners),
            starting_match_score=starting_match_score,
        )
        if result.status != "satisfied" or result.score is None:
            raise ValueError(
                f"native CP-SAT scoring did not complete: {result.solver_status}"
            )
        return {
            **dict(result.score),
            "modelHash": result.model_hash,
            "variableCount": result.variable_count,
            "constraintCount": result.constraint_count,
            "branches": result.branches,
            "conflicts": result.conflicts,
        }
    counting = scoring.resolve_counting_team(
        declarer_team=buyer.team,
        escalation=level,
        last_escalating_team=history[-1].actor.team if history else None,
    )
    raw = tuple(int(item) for item in play.raw_team_points)
    total_cards = scoring.card_game_point_totals[contract.effective.value]
    converted = scoring.convert_raw(raw[int(counting)], contract)
    card_game = [0, 0]
    card_game[int(counting)] = converted
    card_game[int(counting.other)] = total_cards - converted
    if tuple(card_game) != play.card_game_points:
        raise AssertionError("play and scoring Domain IR disagree")
    project_points = projects.combined_awarded_value
    baloot_points = projects.baloot_points
    totals = tuple(
        card_game[index] + project_points[index] + baloot_points[index]
        for index in range(2)
    )
    if totals[0] == totals[1]:
        winner = scoring.tie_winner(contract, raw[int(counting)], counting, buyer.team)
    else:
        winner = Team.NORTH_SOUTH if totals[0] > totals[1] else Team.WEST_EAST
    tricks_won = [0, 0]
    for seat in play.trick_winners:
        tricks_won[int(seat.team)] += 1
    kabout = scoring.kabout_enabled and 0 in tricks_won
    if kabout and scoring.kabout_winner == "trick_sweeper":
        winner = Team.NORTH_SOUTH if tricks_won[0] else Team.WEST_EAST
    gahwa = level == "gahwa"
    awarded = scoring.round_award(
        contract=contract,
        winner=winner,
        declarer_team=buyer.team,
        escalation=level,
        total_before_award=totals,
        project_points=project_points,
        baloot_points=baloot_points,
        kabout=kabout,
    )
    final_match, match_winner = scoring.apply_match_award(
        starting_score=starting_match_score,
        awarded_points=awarded,
        round_winner=winner,
        gahwa=gahwa,
    )
    core = {
        "schemaVersion": 1,
        "scoringDomainHash": scoring.sha256,
        "buyer": buyer.name.lower(),
        "contract": contract.value,
        "rawPoints": list(raw),
        "cardGamePoints": card_game,
        "projectPoints": list(project_points),
        "balootPoints": list(baloot_points),
        "gamePoints": list(awarded),
        "roundWinner": winner.name.lower(),
        "kabout": kabout,
        "gahwa": gahwa,
        "finalMatchScore": list(final_match),
        "matchWinner": match_winner.name.lower() if match_winner is not None else None,
        "matchTarget": scoring.match_target,
    }
    return {**core, "sha256": canonical_sha256(core)}


class _SearchDecisionPrefixMiss(ValueError):
    """A legal generated round that does not expose the selected decision."""


class _PolicyContinuationCompilerDisagreement(ValueError):
    """A proof PV cannot be represented by the retained exact trace authority."""


class _QualityScenarioTargetMiss(ValueError):
    """An intelligent continuation already passed the requested decision root."""

    def __init__(self, message, *, evidence=None):
        super().__init__(message)
        self.evidence = evidence


class _QualityEditorialMiss(_QualityScenarioTargetMiss):
    def __init__(self, evidence):
        super().__init__("native focal suffix fails necessary interestingness checks")
        self.evidence = evidence


def _assert_strategic_project_decision_v1(
    *,
    decision_kind: str,
    declaration_mode: str,
    legal_actions: Sequence[Mapping[str, Any]],
    native_cp_sat: bool = False,
) -> None:
    if (
        decision_kind == "project"
        and declaration_mode == "optional"
        and len(legal_actions) < 2
    ):
        raise AssertionError(
            "optional project normalization admitted a non-strategic decision"
        )


def _search_decision_prefix(
    candidate: OrToolsNeutralCandidateV1,
    scenario: ResearchScenarioV1,
    *,
    project_declaration_mode: str | None = None,
    native_cp_sat: bool = False,
) -> ExactTracePrefixV1:
    """Freeze a generated decision prefix in the canonical exact-input contract."""

    trace = dict(candidate.trace)
    decision_kind = str(scenario.decision_point["kind"])
    selector = dict(scenario.decision_point["selector"])
    play = dict(trace["play"])
    complete_auction_actions = [
        {
            "actor": row["actor"],
            "action": row["action"],
            **({"trumpSuit": row["trumpSuit"]} if row.get("trumpSuit") else {}),
        }
        for row in dict(trace["auction"])["completion"]["actions"]
    ]
    complete_escalation_actions = [
        {
            "actor": row["actor"],
            "action": row["action"],
            **(
                {"requestedMode": row["requestedMode"]}
                if row.get("requestedMode") is not None
                else {}
            ),
        }
        for row in dict(dict(trace["escalation"])["observation"])["actions"]
    ]
    project_rows = [
        row for row in dict(trace["projects"])["projects"] if row["declared"]
    ]
    project_rows.sort(key=lambda row: int(row["declarationOrder"]))
    complete_project_declarations = [
        {
            key: row[key]
            for key in ("owner", "kind", "value", "highRank", "declarationOrder")
        }
        for row in project_rows
    ]
    auction_actions = complete_auction_actions
    escalation_actions = complete_escalation_actions
    project_declarations = complete_project_declarations
    project_actions: list[dict[str, object]] | None = None
    play_actions: list[dict[str, object]] = []
    if decision_kind == "bid":
        action_index = int(selector["actionIndex"])
        if action_index >= len(complete_auction_actions):
            raise _SearchDecisionPrefixMiss(
                "generated auction trace has no live decision at selector.actionIndex"
            )
        auction_actions = complete_auction_actions[:action_index]
        escalation_actions = []
        project_declarations = []
    elif decision_kind == "play":
        declaration_actions = [
            dict(row)
            for row in dict(trace["projects"]).get("declarationActions", ())
        ]
        if declaration_actions:
            if len(declaration_actions) != 4:
                raise _SearchDecisionPrefixMiss(
                    "generated play trace lacks the four governed project actions"
                )
            project_actions = declaration_actions
            project_declarations = []
        solved_plays = tuple(
            TailPrefixActionV1.build(
                row["actor"],
                row["card"],
                ekka=row.get("ekka", False),
            )
            for row in play["plays"]
        )
        prefix_length = _decision_prefix_length(scenario, solved_plays)
        play_actions = [
            {
                "actor": row["actor"],
                "card": row["card"],
                "ekka": row.get("ekka", False),
            }
            for row in play["plays"][:prefix_length]
        ]
    elif decision_kind == "escalation":
        action_index = int(selector["actionIndex"])
        if action_index > len(complete_escalation_actions):
            raise _SearchDecisionPrefixMiss(
                "generated escalation trace is shorter than selector.actionIndex"
            )
        if (
            action_index < len(complete_escalation_actions)
            and complete_escalation_actions[action_index]["actor"]
            != selector["actingSeat"]
        ):
            raise _SearchDecisionPrefixMiss(
                "generated escalation actor does not match selector.actingSeat"
            )
        escalation_actions = complete_escalation_actions[:action_index]
        project_declarations = []
    elif decision_kind == "project":
        owner = str(selector["owner"])
        if int(selector.get("projectIndex", 0)) != 0:
            raise _SearchDecisionPrefixMiss(
                "generated project decision supports only the authoritative project"
            )
        project_evidence = dict(trace["projects"])
        declaration_actions = [
            dict(row) for row in project_evidence.get("declarationActions", ())
        ]
        if len(declaration_actions) != 4:
            raise _SearchDecisionPrefixMiss(
                "generated project trace lacks the four governed declaration actions"
            )
        owner_indexes = [
            index
            for index, row in enumerate(declaration_actions)
            if str(row.get("actor")) == owner
        ]
        if len(owner_indexes) != 1:
            raise _SearchDecisionPrefixMiss(
                "generated project actors do not contain selector.owner exactly once"
            )
        owner_index = owner_indexes[0]
        if not native_cp_sat and owner_index != 3:
            raise _SearchDecisionPrefixMiss(
                "legacy generated project decisions support only the final actor"
            )
        if not native_cp_sat and project_declaration_mode == "optional" and not any(
            str(row.get("owner")) == owner and row.get("selected") is True
            for row in project_evidence.get("projects", ())
        ):
            raise ProjectDecisionNotStrategicError()
        solved_plays = [dict(row) for row in play.get("plays", ())]
        if len(solved_plays) < owner_index:
            raise _SearchDecisionPrefixMiss(
                "generated project trace lacks the preceding interleaved plays"
            )
        if [str(row.get("actor")) for row in solved_plays[:owner_index]] != [
            str(row.get("actor")) for row in declaration_actions[:owner_index]
        ]:
            raise _SearchDecisionPrefixMiss(
                "generated project and first-trick actor order disagree"
            )
        project_actions = declaration_actions[:owner_index]
        project_declarations = []
        play_actions = [
            {
                "actor": row["actor"],
                "card": row["card"],
                "ekka": row.get("ekka", False),
            }
            for row in solved_plays[:owner_index]
        ]
    else:
        raise _SearchDecisionPrefixMiss("generated decision prefix kind is unsupported")
    return ExactTracePrefixV1.build(
        {
            "schemaVersion": 1,
            "deckOrder": list(trace["deckOrder"]),
            "dealer": trace["dealer"],
            "roundNumber": int(trace.get("roundNumber", 1)),
            "matchScore": list(trace.get("matchScore", (0, 0))),
            "auctionActions": auction_actions,
            "escalationActions": escalation_actions,
            **(
                {"projectActions": project_actions}
                if project_actions is not None
                else {}
            ),
            "projectDeclarations": project_declarations,
            "projectAssertions": [],
            "playActions": play_actions,
            "stateAssertions": {},
        }
    )


def _construction_auction_prefix_v1(
    candidate: OrToolsNeutralCandidateV1,
) -> ExactTracePrefixV1:
    """Keep the scenario-constrained auction when generating its quality suffix."""

    trace = candidate.trace
    actions = dict(dict(trace["auction"])["completion"])["actions"]
    return ExactTracePrefixV1.build(
        {
            "schemaVersion": 1,
            "deckOrder": list(trace["deckOrder"]),
            "dealer": trace["dealer"],
            "roundNumber": trace.get("roundNumber", 1),
            "matchScore": list(trace.get("matchScore", (0, 0))),
            "auctionActions": [
                {
                    "actor": row["actor"],
                    "action": row["action"],
                    **(
                        {"trumpSuit": row["trumpSuit"]}
                        if row.get("trumpSuit") is not None
                        else {}
                    ),
                }
                for row in actions
            ],
            "escalationActions": [],
            "projectActions": [],
            "projectDeclarations": [],
            "playActions": [],
        }
    )


def _future_card_count_relaxation_v1(
    initial_hands: Mapping[str, Sequence[str]], prefix: ExactTracePrefixV1,
    predicates: Sequence[ScenarioPredicateV1], *, maximum_seconds: float,
    cancel_check: Callable[[], bool] | None = None,
) -> Mapping[str, Any] | None:
    """Refute impossible future counts; a feasible relaxation proves no scenario."""
    future = tuple(predicates)
    if not future or maximum_seconds <= 0 or cancel_check is not None and cancel_check():
        return None
    deadline = time.monotonic() + maximum_seconds
    hands = {seat: {Card.parse(card).id for card in initial_hands[seat.name.lower()]} for seat in Seat}
    if any(len(cards) != 8 for cards in hands.values()) or set.union(*hands.values()) != set(range(32)):
        raise ValueError("future count relaxation requires complete unique hands")
    for play in prefix.play_actions:
        actor, card = Seat.parse(play["actor"]), Card.parse(play["card"]).id
        if card not in hands[actor]:
            raise ValueError("future count prefix disagrees with initial ownership")
        hands[actor].remove(card)
    model = cp_model.CpModel()
    tricks, selected = {}, []
    for index in sorted({int(predicate.arguments["trickIndex"]) for predicate in future}):
        fixed = {Seat.parse(play["actor"]): Card.parse(play["card"]).id
            for play in prefix.play_actions[index * 4:index * 4 + 4]}
        actions = []
        for seat in Seat:
            card = model.new_int_var_from_domain(cp_model.Domain.from_values(
                [fixed[seat]] if seat in fixed else sorted(hands[seat])),
                                                 f"future_{index}_{int(seat)}")
            selected.append(card)
            actor = model.new_constant(int(seat))
            # Omit follow-suit, order and Ekka legality: every legal continuation must still fit.
            actions.append(CpSatActionVariablesV1(f"FUTURE-{index}-{int(seat)}", card,
                actor, actor, model.new_bool_var(f"future_ekka_{index}_{int(seat)}"),
                model.new_constant(1)))
        tricks[index] = CpSatTrickVariablesV1(index, tuple(actions), actions[0])
    model.add_all_different(selected)
    context = CpSatPredicateModelContextV1(model, "future_card_count_relaxation_v1",
                                          tricks, (), model.new_constant(0))
    manifest = shared_scenario_predicate_catalog_lowering_manifest_v2()
    for predicate in future:
        model.add(manifest.lower_predicate(predicate, context).literal == 1)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None
    solver.parameters.max_time_in_seconds = remaining
    status = _solve_cp_sat_with_cancel_v1(solver, model, cancel_check)
    if status == cp_model.MODEL_INVALID:
        raise ValueError("future card-count relaxation model is invalid")
    if (status != cp_model.INFEASIBLE or time.monotonic() >= deadline
            or cancel_check is not None and cancel_check()):
        return None
    core = {"schemaVersion": 1, "kind": "future_card_count_relaxation_v1",
        "scope": "fixed_initial_hands_and_generated_prefix", "solverStatus": "infeasible",
        "sourcePrefixHash": prefix.sha256, "modelHash": canonical_sha256(str(model.proto)),
        "predicateIds": [predicate.predicate_id for predicate in future]}
    return {**core, "sha256": canonical_sha256(core)}


def _future_card_count_admissible_actions_v1(initial_hands, prefix, normalized, predicates, *,
        maximum_seconds, cancel_check=None):
    """Filter the actual witness, never the actor's hidden-world proof universe."""
    deadline = time.monotonic() + maximum_seconds
    allowed = []
    actions = sorted(row["code"] for row in normalized.legal_actions)
    for code in actions:
        seconds = deadline - time.monotonic()
        if seconds <= 0 or cancel_check is not None and cancel_check():
            raise InformationSetCompilationStopped(
                "cancelled" if cancel_check is not None and cancel_check() else "maximum_seconds",
                stage="scenario_witness_selection")
        action = LegalPlayActionV1.from_code(code)
        raw = prefix.to_dict()
        raw.pop("sha256")
        raw["stateAssertions"], raw["projectAssertions"] = {}, []
        raw["playActions"].append({"actor": normalized.next_actor,
            "card": action.card.id, "ekka": action.ekka})
        trial = ExactTracePrefixV1.build(raw)
        if _future_card_count_relaxation_v1(initial_hands, trial, predicates,
                maximum_seconds=seconds, cancel_check=cancel_check) is None:
            allowed.append(code)
    return tuple(allowed) if len(allowed) != len(actions) else None


def _construction_complete_prefix_v1(
    candidate: OrToolsNeutralCandidateV1,
) -> ExactTracePrefixV1:
    from .construction_trace import construction_trace_prefix_v1

    return construction_trace_prefix_v1(candidate.trace)


def _recorded_play_action_key_v1(
    prefix: ExactTracePrefixV1, play_index: int,
) -> tuple[int, str, str]:
    """Identify a played action by its full, immutable preceding trace."""

    if not 0 <= play_index < len(prefix.play_actions):
        raise ValueError("recorded play index is outside the trace")
    source = prefix.to_dict()
    source.pop("sha256")
    selected = source["playActions"][play_index]
    source["playActions"] = source["playActions"][:play_index]
    action = LegalPlayActionV1(
        Card.parse(selected["card"]), bool(selected.get("ekka", False)),
    ).code
    return play_index, ExactTracePrefixV1.build(source).sha256, action


def _terminal_score_from_state_v1(
    state_value: Mapping[str, Any],
    *,
    match_target: int,
) -> dict[str, Any]:
    state = dict(state_value)
    round_score = state.get("roundScore")
    if not isinstance(round_score, Mapping):
        raise _PolicyContinuationCompilerDisagreement(
            "policy continuation lacks a terminal score"
        )
    return {
        "rawPoints": list(round_score["rawCardPoints"]),
        "cardGamePoints": list(round_score["cardGamePoints"]),
        "projectPoints": list(round_score["projectPoints"]),
        "balootPoints": list(round_score["balootPoints"]),
        "gamePoints": list(round_score["awardedPoints"]),
        "awardedPoints": list(round_score["awardedPoints"]),
        "finalMatchScore": list(state["matchScore"]),
        "roundWinner": str(round_score["winner"]),
        "matchWinner": state.get("matchWinner"),
        "kabout": bool(round_score["kabout"]),
        "gahwa": bool(round_score["gahwa"]),
        "matchTarget": match_target,
    }


def compile_observed_decision_v1(
    rules: RuleBook,
    normalized: OrToolsNormalizedExactStateV1,
    prefix: ExactTracePrefixV1,
    decision_point: Mapping[str, Any],
    request: ScenarioProofRequestV1,
    *,
    maximum_nodes: int,
    maximum_depth: int,
    maximum_worlds: int,
    maximum_seconds: float,
    native_cp_sat: bool,
    cancel_check: Callable[[], bool] | None = None,
    declared_maximum_seconds: float | None = None,
    private_recall: Mapping[str, Sequence[str]] | None = None,
    opportunity_order: Sequence[str] | None = None,
    graph_workers: int = 1,
    worker_cpu_callback: Callable[[float], None] | None = None,
) -> ActorInformationSetCompilationV1:
    """Share observer-safe play/non-play compilation across proof and quality."""
    if type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds) or maximum_seconds <= 0:
        raise ValueError("observed decision time limit must be finite and positive")
    declared_seconds = maximum_seconds if declared_maximum_seconds is None else declared_maximum_seconds
    if (type(declared_seconds) not in {int, float} or not math.isfinite(declared_seconds)
            or declared_seconds < maximum_seconds):
        raise ValueError("declared wall limit must be finite and cover the runtime allowance")
    if cancel_check is not None and not callable(cancel_check):
        raise TypeError("observed decision cancellation check must be callable")
    deadline = time.monotonic() + maximum_seconds
    prior_coverage = None

    def remaining() -> float:
        if cancel_check is not None and cancel_check():
            raise InformationSetCompilationStopped("cancelled", stage="observed_decision",
                evidence={"priorHistoryCoverage": prior_coverage} if prior_coverage is not None else None)
        value = deadline - time.monotonic()
        if value <= 0:
            raise InformationSetCompilationStopped("maximum_seconds", stage="observed_decision",
                evidence={"priorHistoryCoverage": prior_coverage} if prior_coverage is not None else None)
        return value

    remaining()
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=native_cp_sat)
    kind = str(decision_point["kind"])
    validate_proof_boundary_scope_v1(request, kind)
    recorded_history = normalized.recorded_escalation_history()
    if recorded_history is not None:
        if not native_cp_sat or kind != "escalation":
            raise ValueError("recorded escalation proof requires native escalation normalization")
        normalized.require_integrity()
        selector = dict(decision_point.get("selector") or {})
        if (normalized.rules_hash != rules.sha256 or normalized.prefix_hash != _effective_prefix_hash(prefix)
                or (selector.get("actingSeat") is not None
                    and Seat.parse(selector["actingSeat"]).name.lower() != normalized.next_actor)):
            raise ValueError("recorded escalation source context disagrees with the observed decision")
        recorded_recall = {}
        for transition in recorded_history["transitions"]:
            if transition["event"]["visibility"] == "private":
                actions = tuple(EscalationGraphActionV1.build(row["actor"], row["action"], row["requestedMode"])
                    for row in transition["before"]["observation"]["actions"])
                recorded_recall.setdefault(transition["event"]["actor"], []).append(_escalation_window_key(actions))
        if private_recall is not None and normalize_escalation_recall_v1(private_recall) != normalize_escalation_recall_v1(recorded_recall):
            raise ValueError("private recall disagrees with recorded escalation normalization")
        private_recall = recorded_recall
        # Rebuild the public source: an opponent's actual private decline cannot fix an observed proof world.
        public_decision = {**decision_point, "selector": {**selector, "actingSeat": normalized.next_actor}}
        try:
            normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
                prefix, decision_point=public_decision, maximum_seconds=remaining(), cancel_check=cancel_check)
            decision_point = public_decision
        except ExactNormalizationStopped as exc:
            raise InformationSetCompilationStopped(exc.partial_receipt["exhaustionReason"],
                stage="observed_decision", evidence={"normalizationReceipt": exc.partial_receipt}) from exc
    recall = normalize_escalation_recall_v1(private_recall)
    if recall and (not native_cp_sat or kind not in {"escalation", "project", "play"}):
        raise ValueError("persisted recall requires native escalation/project/play compilation")
    private_history_incomplete = (any(seat != normalized.next_actor for seat in recall)
                                  or (native_cp_sat and bool(prefix.escalation_actions)))
    prior_coverage = None
    if native_cp_sat and kind in {"escalation", "project", "play"} and private_recall is not None:
        prior_coverage = enumerate_escalation_prior_histories_v1(
            EscalationDomainIRV1.from_rules(rules.data), declarer=normalized.state["declarer"],
            contract=normalized.state["contract"], match_score=prefix.match_score,
            observer_seat=normalized.next_actor, own_declined_windows=recall.get(normalized.next_actor, ()),
            closed=kind != "escalation",
            prefix=tuple(EscalationGraphActionV1.build(item["actor"], item["action"],
                item.get("mode", item.get("requestedMode"))) for item in prefix.escalation_actions),
            maximum_worlds=maximum_worlds, maximum_seconds=remaining(), declared_maximum_seconds=declared_seconds,
            budget=NonPlayGraphBudgetV1(maximum_nodes, maximum_depth), cancel_check=cancel_check)
        if not prior_coverage["complete"] or prior_coverage["worldCount"] == 0:
            raise InformationSetCompilationStopped(prior_coverage["stopReason"] or "no_compatible_private_histories",
                stage="escalation_history_preflight", evidence=prior_coverage)
        private_history_incomplete = False
    history_count = prior_coverage["worldCount"] if prior_coverage is not None else 1
    def compile_retaining_history(operation, *args, **kwargs):
        try:
            return operation(*args, **kwargs)
        except InformationSetCompilationStopped as exc:
            if prior_coverage is None:
                raise
            raise InformationSetCompilationStopped(exc.reason, stage=exc.partial_receipt["stage"],
                evidence={"causeReceipt": exc.partial_receipt, "priorHistoryCoverage": prior_coverage}) from exc
    if private_history_incomplete and request.mode == "exact":
        raise InformationSetCompilationStopped("private_history_coverage_unavailable", stage="observed_decision",
            evidence={"privateHistoryScopeHash": canonical_sha256(recall)})
    interleaved_play = (
        native_cp_sat and kind == "play" and prefix.project_actions_present
        and len(prefix.project_actions) < len(rules.project_domain.declaration_seats(Seat.parse(prefix.dealer)))
        and (request.mode == "exact" or (request.horizon is not None
            and (request.horizon["unit"] not in {"plies", "decisions"} or request.horizon["value"] > 1)))
    )
    if kind == "play" and not interleaved_play:
        result = compile_retaining_history(compiler.compile,
            normalized, request, maximum_nodes=maximum_nodes,
            maximum_depth=maximum_depth, maximum_worlds=maximum_worlds,
            maximum_seconds=remaining(), cancel_check=cancel_check,
            graph_workers=graph_workers,
            worker_cpu_callback=worker_cpu_callback,
            **({"private_recall": recall} if recall else {}),
            **({"prior_history_worlds": prior_coverage["worlds"]} if prior_coverage is not None else {}),
        )
    else:
        complete_raw = request.to_dict()
        complete_raw.pop("sha256", None)
        complete_raw["visibility"] = "complete_state"
        complete_request = ScenarioProofRequestV1.build(complete_raw)
        order_coverage = None
        if native_cp_sat and kind == "escalation":
            order_coverage = enumerate_escalation_opportunity_orders_v1(
                EscalationDomainIRV1.from_rules(rules.data),
                declarer=normalized.state["declarer"], contract=normalized.state["contract"],
                match_score=prefix.match_score, acting_seat=normalized.next_actor,
                prefix=tuple(EscalationGraphActionV1.build(item["actor"], item["action"],
                    item.get("mode", item.get("requestedMode"))) for item in prefix.escalation_actions),
                maximum_worlds=maximum_worlds // history_count, maximum_seconds=remaining(),
                declared_maximum_seconds=declared_seconds,
                include_private_root_histories=True,
                budget=NonPlayGraphBudgetV1(maximum_nodes, maximum_depth), cancel_check=cancel_check,
            )
            if not order_coverage["complete"]:
                raise InformationSetCompilationStopped(order_coverage["stopReason"],
                    stage="escalation_order_preflight", evidence=order_coverage)

        native_auction_models = []
        compiled_world_nodes = 0

        def single_world_graph(world_normalized, world_prefix, order=None, past=None):
            nonlocal compiled_world_nodes
            nodes_remaining = maximum_nodes - compiled_world_nodes
            if nodes_remaining < 1:
                raise InformationSetCompilationStopped("maximum_nodes", stage="observed_nonplay_graph",
                    evidence={"compiledWorldNodes": compiled_world_nodes, "maximumNodes": maximum_nodes,
                              "requestHash": request.sha256})
            world_recall = past["privateRecall"] if past is not None else recall
            model_receipts = [] if native_cp_sat and kind == "bid" else None
            try:
                compiled = compile_nonplay_complete_graph_v1(
                    rules, world_normalized, world_prefix, decision_point, complete_request,
                    maximum_nodes=nodes_remaining, maximum_depth=maximum_depth,
                    maximum_seconds=remaining(), native_cp_sat=native_cp_sat,
                    _compact_forced=native_cp_sat,
                    cancel_check=cancel_check,
                    **({"native_model_receipts": model_receipts} if model_receipts is not None else {}),
                    **({"opportunity_orders": order["opportunityOrders"],
                        "prior_non_announcing_seats": order.get("priorNonAnnouncingSeats", ()),
                        "prior_window_recall": past["privateRecall"] if past is not None else recall}
                       if order is not None else {}),
                    **({"opportunity_order": opportunity_order}
                       if order is None and opportunity_order is not None
                       and kind in {"bid", "escalation"} else {}),
                    base_observer_factory=lambda observed, observed_prefix: (
                        build_nonplay_root_observer_v1(
                            observed, observed_prefix,
                            "project" if interleaved_play else observed.decision_kind,
                        )
                    ),
                    play_observer_factory=lambda play_normalized, start: (
                        build_nonplay_decision_observer_v1(
                            rules=rules, normalized_exact=play_normalized,
                            exact_prefix=start.prefix, decision_kind="project" if interleaved_play else kind, projects=start.projects,
                            native_cp_sat=native_cp_sat,
                            project_evidence=start.project_evidence if native_cp_sat else None,
                        )
                    ),
                )
            except (NonPlayGraphCompilationStoppedV1, ExactNormalizationStopped) as exc:
                source = ({"nodes": exc.nodes, "depth": exc.depth}
                          if isinstance(exc, NonPlayGraphCompilationStoppedV1)
                          else {"normalizationReceipt": exc.partial_receipt})
                if model_receipts:
                    source["nativeAuctionModels"] = [encode_native_auction_model_receipt_v1(item) for item in model_receipts]
                reason = (exc.reason if isinstance(exc, NonPlayGraphCompilationStoppedV1)
                          else exc.partial_receipt["exhaustionReason"])
                raise InformationSetCompilationStopped(reason, stage="observed_nonplay_graph",
                    evidence={"decisionKind": kind, "requestHash": request.sha256,
                              "exactHash": world_normalized.sha256, "prefixHash": world_prefix.sha256,
                              **source, **({"orderWorldId": order["worldId"],
                                  "orderCoverage": order_coverage} if order is not None else {})}) from exc
            remaining()
            compiled_world_nodes += (len(ProofGraphSequencePackV1.retained_node_ids(
                compiled.graph.nodes, compiled.graph.root_node_id)) if native_cp_sat else len(compiled.graph.nodes))
            if compiled_world_nodes > maximum_nodes:
                raise InformationSetCompilationStopped("maximum_nodes", stage="observed_nonplay_graph",
                    evidence={"compiledWorldNodes": compiled_world_nodes, "maximumNodes": maximum_nodes,
                              "requestHash": request.sha256})
            if model_receipts:
                native_auction_models.append({
                    "sourcePrefixHash": prefix.sha256, "sourceExactHash": normalized.sha256,
                    "exactHash": world_normalized.sha256, "prefixHash": world_prefix.sha256,
                    "requestHash": request.sha256, "graphHash": compiled.graph.sha256,
                    "models": [encode_native_auction_model_receipt_v1(item) for item in model_receipts],
                    **({"boundaryEvaluation": dict(compiled.boundary_evaluation)} if compiled.boundary_evaluation else {}),
                })
            return ActorInformationGraphWorldV1.build(
                world_id=(canonical_sha256({"exact": world_normalized.sha256, "order": order["worldId"],
                                           **({"priorHistory": past["worldId"]} if past is not None else {})})
                          if order is not None else canonical_sha256({"exact": world_normalized.sha256,
                              "priorHistory": past["worldId"]}) if past is not None else world_normalized.sha256), decision_kind=kind,
                graph=compiled.graph, observation_ids={
                    node.node_id: bind_escalation_recall_v1(compiled.decision_observation_ids[node.node_id],
                        world_recall.get(node.actor_seat, ()))
                    for node in compiled.graph.nodes if not node.is_leaf
                } if world_recall and kind != "escalation" else compiled.decision_observation_ids,
            )

        def world_graph(world_normalized, world_prefix, _request):
            if order_coverage is None:
                if prior_coverage is None:
                    return single_world_graph(world_normalized, world_prefix)
                return tuple(single_world_graph(world_normalized, world_prefix, past=past)
                             for past in prior_coverage["worlds"])
            return tuple(single_world_graph(world_normalized, world_prefix, order, past)
                         for order in order_coverage["worlds"]
                         for past in (prior_coverage["worlds"] if prior_coverage is not None else (None,)))

        public_boundary = (
            native_cp_sat and kind == "escalation" and normalized.decision_kind == "escalation"
            and request.visibility == "actor_information_set"
            and OrToolsNonPlayPlayGraphGraftV1._raw_margin_is_invariant_before_first_trick(request)
            and request.horizon["value"] == 1
            and not (prefix.play_actions or prefix.project_actions or prefix.project_declarations)
            and not (normalized.state.get("completedTricks") or normalized.state.get("currentTrick"))
        )
        if public_boundary:
            normalized.require_integrity()
            if normalized.rules_hash != rules.sha256:
                raise ValueError("public escalation boundary requires the frozen normalization rules")
            # Hidden cards cannot change this frontier; keep bounded, not exhaustive, coverage.
            worlds = world_graph(normalized, prefix, request)
            result = compiler.compile_graph_worlds(
                (worlds,) if isinstance(worlds, ActorInformationGraphWorldV1) else worlds,
                request, maximum_nodes=maximum_nodes, maximum_depth=maximum_depth,
                maximum_worlds=maximum_worlds, maximum_seconds=remaining(),
                cancel_check=cancel_check, _compact_forced=True,
            )
            result = replace(result, graph=replace(result.graph, model_document={
                **result.graph.model_document, "publicBoundaryProjection": {
                    "kind": "preplay_raw_margin_v1", "sourcePrefixHash": prefix.sha256,
                    "sourceExactHash": normalized.sha256, "requestHash": request.sha256,
                    "cardAllocationEnumeration": "not_performed",
                    "coverage": "bounded_representatives_only",
                },
            }))
        else:
            result = compile_retaining_history(compiler.compile_nonplay,
                normalized, prefix, decision_point, request, graph_compiler=world_graph,
                maximum_nodes=maximum_nodes, maximum_depth=maximum_depth,
                maximum_worlds=maximum_worlds, maximum_seconds=remaining(),
                cancel_check=cancel_check,
                graph_world_multiplier=(order_coverage["orderWorldCount"] if order_coverage is not None else 1) * history_count,
                _compact_forced=native_cp_sat,
            )
        if order_coverage is not None:
            result = replace(result, graph=replace(result.graph,
                model_document={**result.graph.model_document, "escalationOrderCoverage": order_coverage}))
        if native_auction_models:
            result = replace(result, graph=replace(result.graph,
                model_document={**result.graph.model_document, "nativeAuctionModels": native_auction_models}))
            evaluations = [item["boundaryEvaluation"] for item in native_auction_models if "boundaryEvaluation" in item]
            if evaluations:
                if len(evaluations) != len(native_auction_models) or any(item != evaluations[0] for item in evaluations):
                    raise ValueError("native auction worlds disagree on boundary evaluation authority")
                result = replace(result, graph=replace(result.graph,
                    analysis_scope={**(result.graph.analysis_scope or {}), "boundaryEvaluation": evaluations[0]},
                    model_document={**result.graph.model_document, "boundaryEvaluation": evaluations[0]}))
        if prior_coverage is None and recall:
            result = replace(result, graph=replace(result.graph,
                model_document={**result.graph.model_document, "privateRecallHash": canonical_sha256(recall)}))
    if prior_coverage is not None:
        result = replace(result, graph=replace(result.graph,
            model_document={**result.graph.model_document, "priorHistoryCoverage": prior_coverage}))
    if private_history_incomplete:
        result = replace(result, graph=replace(result.graph,
            hidden_worlds={**result.graph.hidden_worlds, "complete": False},
            analysis_scope={**(result.graph.analysis_scope or {}), "worldCoverageComplete": False,
                "privateHistoryCoverage": "declared_history_only", "privateHistoryScopeHash": canonical_sha256(recall)}))
    remaining()
    return result


def validate_content_interest_rules_v1(rules, quality_request, execution_budget):
    """Reject missing tie semantics before paying for gated actor-visible proofs."""
    if (execution_budget is None
            or (execution_budget.video_interestingness or {}).get("schemaVersion") != 4
            or quality_request is None):
        return
    # Check the eventual tail, not the bounded request used before its boundary.
    request = (quality_request.request_for("play", "north", play_index=31,
        execution_budget=execution_budget)
        if isinstance(quality_request, ScenarioPlayingQualityPolicyV1) else quality_request)
    if not isinstance(request, ScenarioProofRequestV1):
        return  # The caller's existing request validation owns malformed inputs.
    if (request.mode == "exact" and request.visibility == "actor_information_set"
            and rules.data["scoring"]["contract"].get("buyer_tie_outcome")
                not in {"buyer_makes", "buyer_loses"}):
        raise ValueError("content admission requires an explicit buyer tie rule")


def _quality_budget_snapshots(profile_ids, profiles, *, execution_budget=None):
    budgets, snapshots = {}, {}
    if execution_budget is not None and execution_budget.exact_tail_budget_profile is not None:
        profile_ids = {*profile_ids, execution_budget.exact_tail_budget_profile}
        if profiles is None:
            raise ValueError("exact tail requires a resolved proof budget profile")
    if profiles is None:
        return budgets, snapshots
    if not isinstance(profiles, Mapping):
        raise TypeError("quality proof budgets require a resolved profile mapping")
    for profile_id in sorted(profile_ids):
        if profile_id not in profiles:
            raise ValueError("quality proof budget profile is unavailable")
        snapshot = copy.deepcopy(dict(profiles[profile_id]))
        budget = ScenarioProofBudgetV1.build(profile_id, snapshot)
        if budget.maximum_workers != 1 or not all(math.isfinite(value) for value in (
            budget.maximum_wall_seconds, budget.maximum_cpu_seconds,
        )):
            raise ValueError("native quality budgets require one worker and finite time limits")
        budgets[profile_id], snapshots[profile_id] = budget, snapshot
    return budgets, snapshots


class _BidQualityCacheV1:
    """Bounded completed-continuation cache; bid/play hits need native readback."""

    def __init__(self, maximum_entries, maximum_bytes):
        if any(type(value) is not int or value < 1 for value in (maximum_entries, maximum_bytes)):
            raise ValueError("bid cache limits must be positive integers")
        self.maximum_entries, self.maximum_bytes = maximum_entries, maximum_bytes
        self.values, self.size, self.lock = {}, 0, threading.Lock()

    def get(self, key, *, with_checkpoints=False):
        with self.lock:
            value = self.values.get(key)
        if value is None:
            return None
        entry = json.loads(value)
        return entry if with_checkpoints else entry["result"]

    def put(self, key, result, *, checkpoints=None):
        if result.get("status") != "complete":
            return
        core = {name: value for name, value in result.items() if name not in {"sha256", "executionReceipt"}}
        if result.get("sha256") != canonical_sha256(core):
            raise ValueError("bid cache result checksum mismatch")
        entry = {"key": key, "result": result}
        if checkpoints is not None:
            entry["decisionCheckpoints"] = checkpoints
        entry["sha256"] = canonical_sha256(entry)
        value = canonical_json(entry).encode("utf-8")
        if len(value) > self.maximum_bytes:
            return
        with self.lock:
            previous = self.values.pop(key, None)
            self.size -= len(previous) if previous is not None else 0
            while self.values and (len(self.values) >= self.maximum_entries or self.size + len(value) > self.maximum_bytes):
                self.size -= len(self.values.pop(next(iter(self.values))))
            self.values[key] = value
            self.size += len(value)

    def snapshot(self):
        with self.lock:
            return [json.loads(value) for value in self.values.values()]

    def merge(self, entries):
        for entry in entries:
            if entry.get("sha256") != canonical_sha256({key: value for key, value in entry.items() if key != "sha256"}):
                raise ValueError("bid cache entry checksum mismatch")
            self.put(entry["key"], entry["result"], checkpoints=entry.get("decisionCheckpoints"))


def _late_recorded_quality_precheck_v1(
    rules, recorded_trace, policy, execution, profiles, search_profiles, *,
    maximum_seconds, cancel_check=None, checkpoint_callback=None,
):
    """Reuse a cheap suffix audit to refute, never admit, a whole recorded line."""
    if (policy.mode != "exact" or policy.visibility != "complete_state"
            or len(recorded_trace.play_actions) != 32 or not recorded_trace.project_actions_present):
        return None
    budget = ScenarioProofBudgetV1.build(policy.decisions["play"]["budgetProfile"],
        profiles[policy.decisions["play"]["budgetProfile"]])
    started, cpu_started = time.monotonic(), time.process_time()

    def remaining():
        if cancel_check is not None and cancel_check():
            raise InformationSetCompilationStopped("cancelled", stage="late_quality_precheck")
        seconds = min(maximum_seconds - (time.monotonic() - started),
            maximum_seconds - (time.process_time() - cpu_started))
        if seconds <= 0:
            raise InformationSetCompilationStopped("maximum_seconds", stage="late_quality_precheck")
        return seconds

    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    # Validate the untouched input, including terminal assertions, before
    # stripping those assertions from intermediate decision prefixes.
    completed = normalizer.normalize_play_or_project(recorded_trace,
        maximum_seconds=remaining(), cancel_check=cancel_check)
    legal_by_step = completed.normalization_receipt.to_dict()["derivedPlay"]["legalPlayActionsByStep"]
    # ponytail: inspect one latest choice after declarations; the normal audit
    # still owns the opening, first trick, and every unchecked decision.
    for index in range(len(recorded_trace.play_actions) - 1, len(Seat) - 1, -1):
        # Reuse solved legality instead of normalizing every forced suffix step.
        if len(legal_by_step[index]) <= 1:
            continue
        raw = recorded_trace.to_dict()
        raw.pop("sha256")
        raw.update(playActions=raw["playActions"][:index], stateAssertions={}, projectAssertions=[])
        prefix = ExactTracePrefixV1.build(raw)
        exact = normalizer.normalize_play_or_project(prefix,
            maximum_seconds=remaining(), cancel_check=cancel_check)
        if len(exact.legal_actions) <= 1:
            continue
        result = resolve_quality_play_continuation_v1(rules, prefix, policy,
            recorded_trace=recorded_trace, stop_on_recorded_quality_failure=True,
            maximum_nodes=budget.maximum_nodes,
            maximum_depth=min(budget.maximum_depth, budget.maximum_horizon_actions),
            maximum_worlds=1, maximum_seconds=execution.maximum_wall_seconds,
            runtime_maximum_seconds=min(remaining(), execution.maximum_wall_seconds) / 2,
            maximum_decision_seconds=min(budget.maximum_wall_seconds, budget.maximum_cpu_seconds),
            maximum_evidence_bytes=execution.maximum_evidence_bytes,
            proof_budget_profiles=profiles, search_budget_profiles=search_profiles,
            execution_budget=execution, opportunity_order=execution.escalation_opportunity_order,
            cancel_check=cancel_check)
        if result["status"] == "quality_failed":
            validate_native_decision_quality_evidence_v1(result, source_prefix=prefix,
                rules_hash=rules.sha256, request=policy.request_for("play", exact.next_actor),
                policy=policy, rules=rules, search_budget_profiles=search_profiles,
                execution_budget=execution, opportunity_order=execution.escalation_opportunity_order,
                maximum_seconds=remaining(), cancel_check=cancel_check)
            # Only freshly verified failures may seed the campaign's no-good cache.
            if checkpoint_callback is not None:
                for row in result["decisions"]:
                    if row["actualActionOptimal"] is False:
                        checkpoint_callback({"kind": "quality_decision_checkpoint_v1",
                            "stage": "play", "decision": row})
        core = {"schemaVersion": 1, "kind": "late_recorded_quality_precheck_v1",
            "scope": "suffix_only", "sourcePrefix": prefix.to_dict(),
            "recordedPrefixHash": recorded_trace.sha256,
            "receipt": {key: value for key, value in result.items() if key != "executionReceipt"},
            "executionReceipt": result["executionReceipt"]}
        return {**core, "sha256": canonical_sha256(core)}
    return None


def resolve_quality_escalation_opportunity_v1(
    rules: RuleBook, prefix: ExactTracePrefixV1, decision_point: Mapping[str, Any],
    request: ScenarioProofRequestV1, *, maximum_nodes: int, maximum_depth: int,
    maximum_worlds: int, maximum_seconds: float, cancel_check: Callable[[], bool] | None = None,
    proof_budget_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    prior_private_events: Sequence[Mapping[str, Any]] = (),
    prior_opportunity_events: Sequence[Mapping[str, Any]] | None = None,
    search_budget_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    runtime_maximum_seconds: float | None = None,
    private_recall: Mapping[str, Sequence[str]] | None = None,
    opportunity_order: Sequence[str] | None = None,
    _normalizer: OrToolsExactStateNormalizerV1 | None = None,
) -> dict[str, Any]:
    """Prove and record one current-window opportunity; not full-round admission.

    Actual private declines are replayed separately from the actor's proof worlds.
    """
    if (decision_point.get("kind") != "escalation" or request.mode not in {"exact", "bounded"}
            or request.visibility not in {"actor_information_set", "complete_state"}):
        raise ValueError("escalation quality requires an exact/bounded information-scope request")
    if request.visibility == "complete_state" and opportunity_order is None:
        raise ValueError("complete-state escalation quality requires the actual opportunity order")
    if type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds) or maximum_seconds <= 0:
        raise ValueError("escalation quality time limit must be positive and finite")
    runtime_seconds = maximum_seconds if runtime_maximum_seconds is None else runtime_maximum_seconds
    if (type(runtime_seconds) not in {int, float} or not math.isfinite(runtime_seconds)
            or not 0 < runtime_seconds <= maximum_seconds):
        raise ValueError("escalation runtime allowance must fit the declared limit")
    if _normalizer is not None and (not isinstance(_normalizer, OrToolsExactStateNormalizerV1)
            or not _normalizer.native_cp_sat or _normalizer.rules.sha256 != rules.sha256):
        raise ValueError("escalation quality requires an owned native normalizer for these rules")
    budgets, snapshots = _quality_budget_snapshots({request.budget_profile}, proof_budget_profiles)
    budget = budgets.get(request.budget_profile)
    if budget is not None:
        maximum_nodes = min(maximum_nodes, budget.maximum_nodes, budget.maximum_state_space)
        maximum_depth = min(maximum_depth, budget.maximum_depth)
        maximum_worlds = min(maximum_worlds, budget.maximum_hidden_worlds)
        if request.horizon is not None and request.horizon["unit"] != "round_end":
            actions = request.horizon["value"] * (len(Seat) if request.horizon["unit"] == "tricks" else 1)
            if actions > min(maximum_depth, budget.maximum_horizon_actions):
                reason = "maximum_horizon_actions" if actions > budget.maximum_horizon_actions else "maximum_depth"
                raise InformationSetCompilationStopped(reason, stage="escalation_quality_preflight",
                    evidence={"proofBudgetHash": budget.sha256})
        elif request.mode == "bounded":
            maximum_depth = min(maximum_depth, budget.maximum_horizon_actions)
    if maximum_nodes < 2:
        raise InformationSetCompilationStopped("maximum_nodes", stage="escalation_quality_preflight")
    deadline = time.monotonic() + runtime_seconds
    cpu_deadline = math.inf
    def stopped():
        return bool(cancel_check is not None and cancel_check()) or time.process_time() >= cpu_deadline
    def remaining():
        if cancel_check is not None and cancel_check():
            raise InformationSetCompilationStopped("cancelled", stage="escalation_quality")
        if cpu_deadline != math.inf and time.process_time() >= cpu_deadline:
            raise InformationSetCompilationStopped("maximum_cpu_seconds", stage="escalation_quality")
        value = deadline - time.monotonic()
        if value <= 0:
            raise InformationSetCompilationStopped("maximum_seconds", stage="escalation_quality")
        return value
    normalizer = _normalizer if _normalizer is not None else OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    normalized = normalizer.normalize(prefix, decision_point=decision_point,
        maximum_seconds=remaining(), cancel_check=cancel_check)
    actor = Seat.parse(normalized.next_actor)
    if request.target_team != actor.team.name.lower():
        raise ValueError("escalation quality must evaluate the acting team's utility")
    stage = OrToolsEscalationStageV1(EscalationDomainIRV1.from_rules(rules.data),
        declarer=Seat.parse(normalized.state["declarer"]), contract=Contract.parse(normalized.state["contract"]),
        match_score=prefix.match_score, native_cp_sat=True)
    if (not isinstance(prior_private_events, Sequence) or isinstance(prior_private_events, (str, bytes))
            or any(not isinstance(event, Mapping) or event.get("actionCode") != stage.domain.closure_action
                   for event in prior_private_events)):
        raise ValueError("prior private events must contain only current-window declines")
    replay = None
    non_announcements = ()
    if prior_private_events:
        replay = stage.replay_opportunities(required_actions=prefix.escalation_actions,
            events=prior_private_events, maximum_seconds=remaining(),
            expected_final_actions=prefix.escalation_actions,
            declared_maximum_seconds=maximum_seconds, cancel_check=cancel_check)
        if replay["status"] == "incomplete":
            raise InformationSetCompilationStopped(replay.get("stopReason", "native_transition_incomplete"),
                stage="escalation_private_replay", evidence=replay)
        if replay["status"] != "satisfied":
            raise ValueError("prior private escalation history is illegal")
        opportunity = replay["finalState"]["opportunityState"]
        if actor.name.lower() not in opportunity["remainingEligibleSeats"]:
            raise ValueError("acting seat has no remaining private opportunity")
        non_announcements = tuple(opportunity["nonAnnouncingSeats"])
    if budget is not None:
        deadline = min(deadline, time.monotonic() + budget.maximum_wall_seconds)
        cpu_deadline = time.process_time() + budget.maximum_cpu_seconds
    proof_cancel_check = stopped if budget is not None else cancel_check
    declared_proof_seconds = min(maximum_seconds, budget.maximum_wall_seconds) if budget is not None else maximum_seconds
    try:
        if request.visibility == "complete_state":
            artifact = _complete_state_quality_artifact_v1(
                rules, normalized, prefix, decision_point, request,
                maximum_nodes=maximum_nodes, maximum_depth=maximum_depth,
                maximum_seconds=remaining(), cancel_check=proof_cancel_check,
                opportunity_order=opportunity_order,
                prior_non_announcing_seats=non_announcements,
                prior_window_recall=private_recall,
                **({"budget": budget, "escalation_events": prior_opportunity_events,
                    "maximum_conflicts": search_budget_profiles[request.budget_profile]["maximumConflicts"]}
                   if search_budget_profiles is not None and prior_opportunity_events is not None
                   and request.mode == "exact" else {}),
            )
        elif (request.mode == "exact" and request.utility == "awarded_point_margin"
                and budget is not None and budget.native_observed is not None
                and search_budget_profiles is not None and prior_opportunity_events is not None
                and opportunity_order is not None):
            artifact = _observed_play_quality_artifact_v1(rules, normalized, prefix, request,
                budget=budget, opportunity_order=opportunity_order, escalation_events=prior_opportunity_events,
                maximum_conflicts=search_budget_profiles[request.budget_profile]["maximumConflicts"],
                maximum_depth=maximum_depth, maximum_seconds=remaining(), cancel_check=proof_cancel_check,
                decision_kind="escalation")
        else:
            compilation = compile_observed_decision_v1(rules, normalized, prefix, decision_point, request,
                maximum_nodes=maximum_nodes, maximum_depth=maximum_depth, maximum_worlds=maximum_worlds,
                maximum_seconds=remaining(), native_cp_sat=True, cancel_check=proof_cancel_check,
                declared_maximum_seconds=declared_proof_seconds, private_recall=private_recall)
            proof = prove_actor_information_set_v1(compilation, request, maximum_nodes=maximum_nodes,
                maximum_worlds=maximum_worlds, maximum_seconds=remaining(), cancel_check=proof_cancel_check,
                native_action_quality=True, declared_maximum_seconds=declared_proof_seconds,
                policy_search=budget.policy_search if budget is not None else None)
            artifact = {"request": request.to_dict(), "proof": proof.to_dict(),
                "proofModel": proof.model_document(), "graph": compilation.graph.to_dict(),
                "allocationModel": dict(compilation.graph.model_document)}
    except InformationSetCompilationStopped as exc:
        if (cpu_deadline != math.inf and time.process_time() >= cpu_deadline
                and not (cancel_check is not None and cancel_check())):
            raise InformationSetCompilationStopped("maximum_cpu_seconds", stage="escalation_quality",
                evidence={"sourceReceipt": exc.partial_receipt, "proofBudgetHash": budget.sha256}) from exc
        raise
    model, proof = artifact["proofModel"], artifact["proof"]
    core = {"schemaVersion": 1, "kind": "native_escalation_quality_opportunity_v1",
            "scope": "current_window_opportunity_only" if replay is not None else "opening_opportunity_only",
            "status": "incomplete",
            **({"privateHistoryReplay": replay} if replay is not None else {}),
            "limits": {"maximumNodes": maximum_nodes, "maximumDepth": maximum_depth,
                       "maximumWorlds": maximum_worlds, "maximumSeconds": maximum_seconds},
            "sourcePrefixHash": prefix.sha256, "exactStateId": normalized.exact_state_id,
            **({"proofBudgetProfiles": snapshots, "proofBudgetHash": budget.sha256} if budget is not None else {}),
            **artifact,
            "transition": None, "nextPublicPrefix": None}
    expected = "exact_complete" if request.mode == "exact" else "bounded_complete"
    if (cpu_deadline != math.inf and time.process_time() >= cpu_deadline
            and not (cancel_check is not None and cancel_check())):
        core["partialReceipt"] = InformationSetCompilationStopped(
            "maximum_cpu_seconds", stage="escalation_quality").partial_receipt
    if proof["completionStatus"] in {"proof_failed", "model_error"}:
        core["status"] = proof["completionStatus"]
    if proof["completionStatus"] == expected:
        quality = model.get("actionQuality")
        selected_action = proof["selectedAction"]
        if (proof["identities"]["proofRequestHash"] != request.sha256
                or proof["identities"]["graphHash"] != artifact["graph"]["sha256"]
                or model.get("rootActorSeat") != normalized.next_actor
                or not isinstance(quality, Mapping) or quality.get("rootActorRelation") != "target"
                or not isinstance(quality.get("optimalActions"), list) or not quality["optimalActions"]
                or type(quality.get("actualActionOptimal")) is not bool
                or quality.get("actualAction") != selected_action
                or (request.selected_action is not None and selected_action != request.selected_action)
                or selected_action not in {row["code"] for row in normalized.legal_actions}
                or (request.selected_action is None and quality.get("actualActionOptimal") is not True)):
            raise ValueError("escalation action-quality evidence disagrees with selected action")
        try:
            transition = stage.transition_opportunity(required_actions=prefix.escalation_actions,
                non_announcements=non_announcements, actor=actor, action_code=selected_action,
                maximum_seconds=remaining(), cancel_check=proof_cancel_check,
                declared_maximum_seconds=declared_proof_seconds)
        except InformationSetCompilationStopped as exc:
            core["partialReceipt"] = exc.partial_receipt
            return {**core, "sha256": canonical_sha256(core)}
        core["transition"] = transition
        if transition["status"] == "satisfied":
            next_prefix = (prefix if transition["event"]["visibility"] == "private"
                           else normalizer.append_normalized_action(prefix, normalized, selected_action))
            core["nextPublicPrefix"] = next_prefix.to_dict()
            core["status"] = "recorded" if quality.get("actualActionOptimal") is True else "quality_failed"
        elif transition["status"] == "infeasible":
            core["status"] = "model_error"
    return {**core, "sha256": canonical_sha256(core)}


def resolve_quality_escalation_continuation_v1(
    rules: RuleBook, prefix: ExactTracePrefixV1, decision_point: Mapping[str, Any],
    quality_request: ScenarioProofRequestV1, *, opportunity_order: Sequence[str],
    maximum_nodes: int, maximum_depth: int, maximum_worlds: int,
    maximum_seconds: float, maximum_decision_seconds: float, maximum_decisions: int,
    maximum_evidence_bytes: int, recorded_events: Sequence[Mapping[str, Any]] | None = None,
    continue_recorded_prefix: bool = False,
    proof_budget_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    search_budget_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    runtime_maximum_seconds: float | None = None,
    checkpoint_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Generate or audit an escalation suffix; complete is not whole-round admission."""
    from .result_authority import _validate_native_escalation_quality_evidence_v1
    if decision_point.get("kind") != "escalation":
        raise ValueError("escalation continuation requires an escalation decision point")
    if quality_request.mode not in {"exact", "bounded"} or quality_request.visibility not in {"actor_information_set", "complete_state"}:
        raise ValueError("escalation continuation requires exact/bounded information-scope quality")
    if cancel_check is not None and not callable(cancel_check):
        raise TypeError("escalation continuation cancellation check must be callable")
    if quality_request.selected_action is not None:
        raise ValueError("use recorded events rather than a fixed action for every opportunity")
    if (not isinstance(opportunity_order, Sequence) or isinstance(opportunity_order, (str, bytes))
            or any(type(seat) is not str for seat in opportunity_order)):
        raise ValueError("opportunity order must list every seat exactly once")
    order = tuple(Seat.parse(seat).name.lower() for seat in opportunity_order)
    if len(order) != len(Seat) or len(set(order)) != len(Seat):
        raise ValueError("opportunity order must list every seat exactly once")
    if any(type(value) is not int or value < 1 for value in
           (maximum_nodes, maximum_depth, maximum_worlds, maximum_decisions, maximum_evidence_bytes)):
        raise ValueError("escalation continuation limits must be positive integers")
    if any(type(value) not in {int, float} or not math.isfinite(value) or value <= 0
           for value in (maximum_seconds, maximum_decision_seconds)):
        raise ValueError("escalation continuation time limits must be finite and positive")
    if recorded_events is not None and (not isinstance(recorded_events, Sequence)
            or isinstance(recorded_events, (str, bytes)) or len(recorded_events) > maximum_decisions
            or any(not isinstance(row, Mapping) or set(row) != {"actor", "actionCode"}
                   or type(row["actor"]) is not str or type(row["actionCode"]) is not str for row in recorded_events)):
        raise ValueError("recorded opportunities require bounded actor/actionCode events")
    if type(continue_recorded_prefix) is not bool:
        raise TypeError("continue_recorded_prefix must be boolean")
    if continue_recorded_prefix and recorded_events is None:
        raise ValueError("escalation prefix continuation requires explicit recorded events")
    recorded = copy.deepcopy(list(recorded_events)) if recorded_events is not None else None
    runtime_seconds = maximum_seconds if runtime_maximum_seconds is None else runtime_maximum_seconds
    if (type(runtime_seconds) not in {int, float} or not math.isfinite(runtime_seconds)
            or not 0 < runtime_seconds <= maximum_seconds):
        raise ValueError("runtime allowance must be positive and cannot exceed the declared budget")
    _budgets, snapshots = _quality_budget_snapshots({quality_request.budget_profile}, proof_budget_profiles)
    requested_actor = decision_point.get("selector", {}).get("actingSeat")
    if (recorded and not continue_recorded_prefix and requested_actor is not None
            and Seat.parse(recorded[0]["actor"]) != Seat.parse(requested_actor)):
        raise ValueError("recorded opportunity actor disagrees with requested actor")
    actor = (
        Seat.parse(recorded[0]["actor"]).name.lower()
        if recorded
        else Seat.parse(requested_actor).name.lower() if requested_actor is not None else None
    )
    if actor is None and prefix.escalation_actions:
        raise ValueError("automatic escalation actor selection requires the start of escalation")
    initial_actor = actor
    current, private_events, events, decisions, artifacts = prefix, [], [], [], {}
    verified_artifacts: set[str] = set()
    recall: dict[str, list[str]] = {}
    status, partial, final_state, used_bytes = "incomplete", None, None, 0
    initial_state, initial_normalization, handoff = None, None, None
    deadline = time.monotonic() + runtime_seconds
    def remaining_seconds():
        if cancel_check is not None and cancel_check():
            raise InformationSetCompilationStopped("cancelled", stage="escalation_continuation")
        remaining = min(maximum_decision_seconds, deadline - time.monotonic())
        if remaining <= 0:
            raise InformationSetCompilationStopped("maximum_seconds", stage="escalation_continuation")
        return remaining
    def retain_evidence(document):
        nonlocal used_bytes
        size = len(json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
        if used_bytes + size > maximum_evidence_bytes:
            raise InformationSetCompilationStopped("maximum_evidence_bytes", stage="escalation_continuation",
                evidence={"candidateHash": canonical_sha256(document)})
        used_bytes += size
        return document
    try:
        normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
        normalized = normalizer.normalize(prefix, decision_point={"kind": "escalation", "selector": {"actingSeat": actor}} if actor is not None else {"kind": "bid"},
            maximum_seconds=remaining_seconds(), cancel_check=cancel_check)
        initial_normalization = retain_evidence(normalized.normalization_receipt.to_dict())
        if prefix.state_assertions:
            derived = prefix.to_dict()
            derived.pop("sha256")
            derived["stateAssertions"] = {}
            current = ExactTracePrefixV1.build(derived)
        stage = OrToolsEscalationStageV1(EscalationDomainIRV1.from_rules(rules.data),
            declarer=Seat.parse(normalized.state["declarer"]), contract=Contract.parse(normalized.state["contract"]),
            match_score=prefix.match_score, native_cp_sat=True)
        initial_state = retain_evidence(stage.solve(required_actions=prefix.escalation_actions,
            required_actions_complete=True, opportunity_non_announcements=(),
            maximum_seconds=remaining_seconds(), cancel_check=cancel_check).to_dict())
        if initial_state["status"] != "satisfied":
            raise InformationSetCompilationStopped("native_escalation_preflight", stage="escalation_continuation",
                evidence=initial_state)
        final_state = initial_state
        if actor is None and not final_state["observation"]["terminal"]:
            eligible = final_state["opportunityState"]["remainingEligibleSeats"]
            actor = next((seat for seat in order if seat in eligible), None)
            if actor is None:
                raise RuntimeError("native escalation has no eligible initial actor")
            initial_actor = actor
        for index in range(maximum_decisions):
            remaining = remaining_seconds()
            if (continue_recorded_prefix and recorded is not None and index == len(recorded)
                    and requested_actor is not None and actor != Seat.parse(requested_actor).name.lower()):
                raise ValueError("recorded opportunity suffix misses requested actor")
            if final_state["observation"]["terminal"]:
                if recorded:
                    raise ValueError("recorded events supplied for a closed escalation stage")
                status = "complete"
                break
            raw = quality_request.to_dict()
            raw.pop("sha256")
            raw.pop("selectedAction", None)
            raw["targetTeam"] = Seat.parse(actor).team.name.lower()
            if recorded is not None and (not continue_recorded_prefix or index < len(recorded)):
                if index == len(recorded):
                    raise InformationSetCompilationStopped("recorded_events_exhausted", stage="escalation_continuation")
                if Seat.parse(recorded[index]["actor"]).name.lower() != actor:
                    raise ValueError("recorded opportunity actor disagrees with declared order")
                raw["selectedAction"] = recorded[index]["actionCode"]
            current_window = _escalation_window_key(tuple(EscalationGraphActionV1.build(
                row["actor"], row["action"], row.get("mode", row.get("requestedMode")))
                for row in current.escalation_actions))
            prior_recall = {seat: tuple(window for window in remembered if window != current_window)
                            for seat, remembered in recall.items()}
            result = resolve_quality_escalation_opportunity_v1(rules, current,
                {"kind": "escalation", "selector": {"actingSeat": actor}}, ScenarioProofRequestV1.build(raw),
                maximum_nodes=maximum_nodes, maximum_depth=maximum_depth, maximum_worlds=maximum_worlds,
                maximum_seconds=maximum_decision_seconds, runtime_maximum_seconds=remaining,
                prior_private_events=private_events, proof_budget_profiles=snapshots if proof_budget_profiles is not None else None,
                private_recall=normalize_escalation_recall_v1(prior_recall),
                **({"search_budget_profiles": search_budget_profiles,
                    "prior_opportunity_events": copy.deepcopy(events)}
                   if search_budget_profiles is not None and not prefix.escalation_actions else {}),
                opportunity_order=order,
                _normalizer=normalizer,
                cancel_check=cancel_check)
            if result["proofModel"].get("kind") in {
                    "native_escalation_selected_action_quality_model_v1", "native_observed_escalation_quality_model_v1"}:
                verified_artifacts.add(result["sha256"])
            artifacts[result["sha256"]] = retain_evidence(result)
            decisions.append({"index": index, "actor": actor, "sourcePrefixHash": current.sha256,
                              "artifactHash": result["sha256"]})
            if checkpoint_callback is not None:
                checkpoint_callback({"schemaVersion": 1, "kind": "quality_decision_checkpoint_v1",
                    "rulesHash": rules.sha256, "stage": "escalation", "sourcePrefix": current.to_dict(),
                    "decision": decisions[-1], "artifact": result})
            if result["status"] != "recorded":
                status, partial = result["status"], result.get("partialReceipt")
                break
            transition = result["transition"]
            pending = transition["before"]["opportunityState"]["remainingEligibleSeats"]
            if actor != next(seat for seat in order if seat in pending):
                raise ValueError("initial opportunity actor disagrees with declared order")
            event = {key: transition["event"][key] for key in ("actor", "actionCode")}
            if transition["event"]["visibility"] == "private":
                history = [EscalationGraphActionV1.build(row["actor"], row["action"], row["requestedMode"])
                           for row in transition["before"]["observation"]["actions"]]
                recall.setdefault(actor, []).append(_escalation_window_key(history))
            events.append(event)
            current = ExactTracePrefixV1.build(result["nextPublicPrefix"])
            private_events = [*private_events, event] if transition["event"]["visibility"] == "private" else []
            final_state = transition["after"]
            if final_state["observation"]["terminal"]:
                if recorded is not None and (len(events) < len(recorded) or
                        not continue_recorded_prefix and len(events) != len(recorded)):
                    raise ValueError("recorded events continue after escalation closure")
                status = "complete"
                break
            eligible = final_state["opportunityState"]["remainingEligibleSeats"]
            actor = next(seat for seat in order if seat in eligible)
        else:
            partial = {"exhaustionReason": "maximum_decisions"}
        if status == "complete":
            next_state = normalizer.normalize_play_or_project(current,
                maximum_seconds=remaining_seconds(), cancel_check=cancel_check)
            if (next_state.state["escalation"] != final_state["observation"]["level"]
                    or next_state.state["locked"] != (final_state["observation"]["mode"] == "locked")):
                raise RuntimeError("project/play handoff disagrees with native escalation closure")
            handoff = retain_evidence({"exactState": next_state.to_dict(),
                "normalizationReceipt": next_state.normalization_receipt.to_dict()})
    except (InformationSetCompilationStopped, ExactNormalizationStopped) as exc:
        status = "incomplete"
        partial = exc.partial_receipt
    except InterruptedError:
        status = "incomplete"
        partial = InformationSetCompilationStopped("cancelled", stage="escalation_continuation").partial_receipt
    except RuntimeError as exc:
        status = "incomplete"
        partial = {"stage": "escalation_continuation", "failureClass": "runtime_error", "reason": str(exc)}
    core = {"schemaVersion": 1, "kind": "native_quality_escalation_continuation_v1",
        "scope": "escalation_suffix_only", "status": status,
        "operation": "audit_then_generate" if continue_recorded_prefix else "generate" if recorded is None else "audit_recorded",
        "rulesHash": rules.sha256, "sourcePrefixHash": prefix.sha256,
        "qualityRequestHash": quality_request.sha256, "opportunityOrder": list(order),
        "recordedEvents": recorded, "events": events, "decisions": decisions, "artifacts": artifacts,
        "finalPrefix": current.to_dict(), "finalState": final_state, "partialReceipt": partial,
        "initialState": initial_state, "initialNormalization": initial_normalization,
        "projectPlayHandoff": handoff,
        "privateRecall": recall,
        "proofBudgetProfiles": snapshots,
        "limits": {"maximumNodes": maximum_nodes, "maximumDepth": maximum_depth, "maximumWorlds": maximum_worlds,
            "maximumSeconds": maximum_seconds, "maximumDecisionSeconds": maximum_decision_seconds,
            "maximumDecisions": maximum_decisions, "maximumEvidenceBytes": maximum_evidence_bytes}}
    if requested_actor is None:
        core["initialActorSelection"] = {"kind": "native_eligible_order_v1", "actor": initial_actor}
    result = {**core, "sha256": canonical_sha256(core)}
    try:
        _validate_native_escalation_quality_evidence_v1(result, source_prefix=prefix,
            rules_hash=rules.sha256, request=quality_request, rules=rules,
            search_budget_profiles=search_budget_profiles,
            maximum_seconds=remaining_seconds(), cancel_check=cancel_check,
            verified_artifacts=frozenset(verified_artifacts))
    except InformationSetCompilationStopped as exc:
        # Retain completed atomic decisions, but never admit unverified completion.
        if partial is not None:
            exc = InformationSetCompilationStopped(exc.reason, stage=exc.partial_receipt["stage"],
                evidence={"sourceReceipt": partial, "readbackReceipt": exc.partial_receipt})
        core.update(status="incomplete", partialReceipt=exc.partial_receipt)
        result = {**core, "sha256": canonical_sha256(core)}
    return result


def _validate_quality_recorded_trace(prefix, recorded_trace, *, allow_escalation_extension=False,
                                     auction_only=False, allow_auction_extension=False, allow_partial=False):
    if not isinstance(recorded_trace, ExactTracePrefixV1):
        raise TypeError("recorded quality trace requires an exact trace prefix")
    recorded_prefix = ExactTracePrefixV1.build(recorded_trace.to_dict())
    source, recorded = prefix.to_dict(), recorded_prefix.to_dict()
    action_fields = (
        {"auctionActions"}
        if auction_only
        else {"playActions", "projectActions", "projectDeclarations"}
    )
    if allow_escalation_extension:
        action_fields.add("escalationActions")
    if allow_auction_extension:
        action_fields.add("auctionActions")
    for key in sorted(source.keys() | recorded.keys()):
        if key in {"sha256", "stateAssertions", "projectAssertions"}:
            continue
        if key in action_fields:
            actions = source.get(key, [])
            if recorded.get(key, [])[:len(actions)] != actions:
                raise ValueError("recorded quality trace does not extend the source actions")
        elif source.get(key) != recorded.get(key):
            raise ValueError(f"recorded quality trace changes the source context: {key}")
    if prefix.project_actions_present != recorded_prefix.project_actions_present:
        raise ValueError("recorded quality trace changes project action semantics")
    all_pass_candidate = (allow_auction_extension and not recorded_prefix.play_actions
        and len(recorded_prefix.auction_actions) == 8
        and all(row["action"] == "pass" for row in recorded_prefix.auction_actions))
    if not allow_partial and not auction_only and not all_pass_candidate and len(recorded_prefix.play_actions) != 32:
        raise ValueError("recorded quality audit requires a complete 32-play line")
    return recorded_prefix


def _observer_safe_play_policy_observation_v1(
    *,
    actor: Seat,
    contract: Contract,
    trump: Suit | None,
    trick_index: int,
    plays: Sequence[tuple[Seat, Card]],
    own_hand: Sequence[Card | str | int | Mapping[str, Any]],
    legal_actions: Sequence[str],
) -> tuple[dict[str, Any], str]:
    observation = {
        "schemaVersion": 1,
        "actor": actor.name.lower(),
        "contract": contract.value,
        "trumpSuit": trump.name.lower() if trump is not None else None,
        "trickIndex": trick_index,
        "currentTrick": [
            {"actor": seat.name.lower(), "card": card.code}
            for seat, card in plays
        ],
        "ownHand": [Card.parse(card).code for card in own_hand],
        "legalActions": sorted(str(action) for action in legal_actions),
    }
    return observation, canonical_sha256(observation)


def _candidate_play_zero_observation_hash_v1(
    candidate: OrToolsNeutralCandidateV1,
) -> str:
    play = candidate.trace.get("play")
    if not isinstance(play, Mapping):
        raise ValueError("quality candidate requires solved play evidence")
    plays = play.get("plays")
    hands = play.get("initialHands")
    legal_by_step = play.get("legalPlayActionsByStep")
    if (
        not isinstance(plays, list)
        or not plays
        or not isinstance(hands, Mapping)
        or not isinstance(legal_by_step, list)
        or not legal_by_step
    ):
        raise ValueError("quality candidate requires complete play-zero evidence")
    first = plays[0]
    actor = Seat.parse(first["actor"])
    legal_actions = tuple(
        LegalPlayActionV1(
            Card.parse(action["card"]), bool(action.get("ekka", False))
        ).code
        for action in legal_by_step[0]
    )
    _, observation_hash = _observer_safe_play_policy_observation_v1(
        actor=actor,
        contract=Contract.parse(candidate.trace["contract"]),
        trump=(
            Suit.parse(candidate.trace["trumpSuit"])
            if candidate.trace.get("trumpSuit") is not None
            else None
        ),
        trick_index=0,
        plays=(),
        own_hand=hands[actor.name.lower()],
        legal_actions=legal_actions,
    )
    return observation_hash


PUBLIC_PARTNER_ENTRY_BASELINE_ID = "observer_safe_public_partner_entry_v2"
PUBLIC_STRANDED_MASTER_BASELINE_ID = "observer_safe_public_stranded_master_v3"
PUBLIC_MASTER_RETENTION_BASELINE_ID = "observer_safe_public_master_retention_v4"
PUBLIC_EXHAUSTED_SUIT_BASELINE_ID = "observer_safe_public_exhausted_suit_v5"
PUBLIC_TWO_TRICK_BASELINE_ID = "observer_safe_public_two_trick_control_v6"
PUBLIC_ENTRY_BASELINE_IDS = {PUBLIC_PARTNER_ENTRY_BASELINE_ID, PUBLIC_STRANDED_MASTER_BASELINE_ID,
                             PUBLIC_MASTER_RETENTION_BASELINE_ID, PUBLIC_EXHAUSTED_SUIT_BASELINE_ID,
                             PUBLIC_TWO_TRICK_BASELINE_ID}


def _public_remaining_card_domains_v6(state, actor, own, played, public_knowledge):
    """Simple shown-void/card-count deductions; never read another hand."""
    counts = {seat: 8 for seat in Seat}
    for trick in [*state["completedTricks"], state["currentTrick"]]:
        for row in trick["plays"]:
            counts[Seat.parse(row.get("actor") or row.get("seat"))] -= 1
    if counts[actor] != len(own) or any(value < 0 for value in counts.values()):
        return None
    known = {Card.parse(card): Seat.parse(seat)
             for card, seat in public_knowledge["knownRemainingCards"]}
    voids = public_knowledge["shownVoids"]
    domains = {card: {seat for seat in Seat if seat != actor and counts[seat] > 0
        and card.suit.name.lower() not in voids[seat.name.lower()]
        and (card not in known or known[card] == seat)}
        for card in set(map(Card, range(32))) - played - own}
    while True:
        before = {card: frozenset(seats) for card, seats in domains.items()}
        if any(not seats for seats in domains.values()):
            return None
        for seat in Seat:
            if seat == actor:
                continue
            possible = {card for card, seats in domains.items() if seat in seats}
            fixed = {card for card in possible if domains[card] == {seat}}
            if len(possible) < counts[seat] or len(fixed) > counts[seat]:
                return None
            if len(possible) == counts[seat]:
                for card in possible:
                    domains[card] = {seat}
            elif len(fixed) == counts[seat]:
                for card in possible - fixed:
                    domains[card].discard(seat)
        if before == {card: frozenset(seats) for card, seats in domains.items()}:
            return domains


def _public_partner_entry_cards_v2(state, actor, domain, contract, trump):
    """Recognize public winners without consulting any other player's hand."""
    history = [*state["completedTricks"], state["currentTrick"]]
    played = {Card.parse(row["card"]) for trick in history for row in trick["plays"]}
    own = {Card.parse(card) for card in state["hands"][actor.name.lower()]}
    voids = {seat: set() for seat in Seat}
    for trick in history:
        rows = trick["plays"]
        if not rows:
            continue
        led = Card.parse(rows[0]["card"]).suit
        for row in rows:
            if Card.parse(row["card"]).suit != led:
                voids[Seat.parse(row.get("actor") or row.get("seat"))].add(led)
    known = {card: actor for card in own}
    for project in state.get("projects", ()):
        if project.get("revealed") is True:
            for raw in project["cards"]:
                card = Card.parse(raw)
                if card not in played:
                    known[card] = Seat.parse(project["owner"])
    remaining = set(map(Card, range(32))) - played
    for card in remaining - known.keys():
        possible = [seat for seat in Seat if seat != actor and card.suit not in voids[seat]]
        if len(possible) == 1:
            known[card] = possible[0]
    team_cards = {card for card, seat in known.items() if seat.team == actor.team}
    partner_masters = {card for card, seat in known.items()
        if seat != actor and seat.team == actor.team
        and all(other in team_cards for other in _established_control_blockers_v1(
            card, domain=domain, contract=contract, trump=trump, cards=remaining))}
    partner = next(seat for seat in Seat if seat != actor and seat.team == actor.team)
    dead_leads = {card for card in own if card.suit in voids[partner]
        and (contract.effective != Contract.HOKUM or card.suit == trump) and any(
        other.suit == card.suit and other not in own
        and domain.strength(other, contract, trump) > domain.strength(card, contract, trump)
        for other in remaining)}
    observation = {
        "knownRemainingCards": sorted((card.code, seat.name.lower()) for card, seat in known.items()),
        "shownVoids": {seat.name.lower(): sorted(suit.name.lower() for suit in voids[seat]) for seat in Seat},
    }
    return partner_masters, dead_leads, observation


def _observer_safe_play_policy_artifact_v1(
    rules: RuleBook,
    normalized: OrToolsNormalizedExactStateV1,
    request: ScenarioProofRequestV1,
    *,
    actual_action: str | None,
    policy_id: str = "observer_safe_trick_control_v1",
    preferred_completed_trick_winner: Seat | None = None,
    maximum_seconds: float,
    cancel_check: Callable[[], bool] | None,
) -> tuple[str, dict[str, Any]]:
    """Select a fast, visible-information-only play policy with CP-SAT.

    This is policy-conditioned quality evidence, not minimax or exact proof.
    """

    if request.mode != "bounded" or request.visibility != "actor_information_set":
        raise ValueError("observer-safe play policy cannot supply exact or complete-state proof")
    if normalized.decision_kind != "play" or normalized.next_actor is None:
        raise ValueError("observer-safe play policy requires a live play decision")
    if policy_id not in NATIVE_OBSERVER_SAFE_PLAY_POLICY_IDS and policy_id not in PUBLIC_ENTRY_BASELINE_IDS:
        raise ValueError("observer-safe play policy is unsupported")
    actor = Seat.parse(normalized.next_actor)
    state = normalized.state
    current = state.get("currentTrick")
    if not isinstance(current, Mapping) or not isinstance(current.get("plays"), list):
        raise ValueError("observer-safe play policy requires the public current trick")
    contract = Contract.parse(state["contract"])
    trump = Suit.parse(state["trumpSuit"]) if state.get("trumpSuit") is not None else None
    domain = rules.play_domain
    public_played = set()
    if policy_id in {NATIVE_ESTABLISHED_CONTROL_POLICY_ID, NATIVE_GUARD_PROTECTION_POLICY_ID} | PUBLIC_ENTRY_BASELINE_IDS:
        public_played = {Card.parse(row["card"]) for trick in state["completedTricks"]
                         for row in trick["plays"]}
        own_cards = {Card.parse(card) for card in state["hands"][actor.name.lower()]}
    plays = tuple(
        (Seat.parse(row.get("actor") or row.get("seat")), Card.parse(row["card"]))
        for row in current["plays"]
    )
    led_suit = plays[0][1].suit if plays else None
    partner_masters, dead_leads, public_knowledge = (_public_partner_entry_cards_v2(state, actor, domain, contract, trump)
        if policy_id in PUBLIC_ENTRY_BASELINE_IDS else (set(), set(), None))
    guards = (_guard_discard_cards_v1(own_cards=own_cards,
        public_played=public_played | {card for _, card in plays}, pool=tuple(map(Card, range(32))),
        domain=domain, contract=contract, trump=trump)
        if policy_id == NATIVE_GUARD_PROTECTION_POLICY_ID else set())

    def beats(challenger: Card, incumbent: Card) -> bool:
        effective_trump = trump if contract.effective == Contract.HOKUM else None
        if effective_trump is not None:
            if challenger.suit == effective_trump and incumbent.suit != effective_trump:
                return True
            if incumbent.suit == effective_trump and challenger.suit != effective_trump:
                return False
        if challenger.suit != incumbent.suit:
            return challenger.suit == led_suit and incumbent.suit != led_suit
        return domain.strength(challenger, contract, trump) > domain.strength(incumbent, contract, trump)

    winner = plays[0] if plays else None
    for row in plays[1:]:
        if beats(row[1], winner[1]):
            winner = row

    stranded = set()
    if (policy_id in {PUBLIC_STRANDED_MASTER_BASELINE_ID, PUBLIC_MASTER_RETENTION_BASELINE_ID,
                      PUBLIC_EXHAUSTED_SUIT_BASELINE_ID, PUBLIC_TWO_TRICK_BASELINE_ID} and contract.effective == Contract.SUN
            and current["index"] == 6 and len(own_cards) == 2
            and winner is not None and (policy_id in {PUBLIC_EXHAUSTED_SUIT_BASELINE_ID, PUBLIC_TWO_TRICK_BASELINE_ID}
                or (winner[0] != actor and winner[0].team == actor.team))):
        unseen = set(map(Card, range(32))) - public_played - {card for _, card in plays} - own_cards
        if policy_id in {PUBLIC_EXHAUSTED_SUIT_BASELINE_ID, PUBLIC_TWO_TRICK_BASELINE_ID} or not any(beats(card, winner[1]) for card in unseen):
            # An exhausted off-suit cannot be led back on the last trick.
            live_masters = {card for card in own_cards if any(other.suit == card.suit for other in unseen)
                and not _established_control_blockers_v1(card, domain=domain,
                    contract=contract, trump=trump, cards=unseen)}
            if live_masters:
                stranded = {card for card in own_cards if card.suit != led_suit
                    and not any(other.suit == card.suit for other in unseen)}

    preserve = set()
    if (policy_id in {PUBLIC_MASTER_RETENTION_BASELINE_ID, PUBLIC_EXHAUSTED_SUIT_BASELINE_ID, PUBLIC_TWO_TRICK_BASELINE_ID} and contract.effective == Contract.SUN
            and winner is not None):
        unseen = set(map(Card, range(32))) - public_played - {card for _, card in plays} - own_cards
        if winner[0].team == actor.team:
            # Feeding points must not throw away our established winner when a lower
            # card of the same suit can be discarded instead.
            preserve = {card for card in own_cards if card.suit != led_suit
                and not _established_control_blockers_v1(card, domain=domain,
                    contract=contract, trump=trump, cards=unseen)
                and any(other.suit == card.suit and domain.strength(other, contract, trump)
                    < domain.strength(card, contract, trump) for other in own_cards)}
        elif current["index"] == 6 and len(own_cards) == 2 and not any(
                beats(card, winner[1]) for card in unseen | own_cards):
            # Direct shown-void knowledge identifies a one-card forced return.
            # Do not upgrade the basic policy with project-absence deductions.
            history = [*state["completedTricks"], current]
            winner_played = sum(Seat.parse(row.get("actor") or row.get("seat")) == winner[0]
                for trick in history for row in trick["plays"])
            known_return = [Card.parse(code) for code, seat in public_knowledge["knownRemainingCards"]
                            if Seat.parse(seat) == winner[0]]
            if winner_played == 7 and len(known_return) == 1:
                returned = known_return[0]
                preserve = {card for card in own_cards if card.suit == returned.suit
                    and domain.strength(card, contract, trump) > domain.strength(returned, contract, trump)}

    take_entry = set()
    if (policy_id == PUBLIC_TWO_TRICK_BASELINE_ID and contract.effective == Contract.SUN
            and current["index"] == 6 and len(own_cards) == 2 and winner is not None):
        played = public_played | {card for _, card in plays}
        unseen = set(map(Card, range(32))) - played - own_cards
        domains = _public_remaining_card_domains_v6(state, actor, own_cards, played, public_knowledge)
        if domains is not None:
            public_knowledge["remainingCardDomains"] = sorted(
                (card.code, sorted(seat.name.lower() for seat in seats)) for card, seats in domains.items())
            partner = next(seat for seat in Seat if seat != actor and seat.team == actor.team)
            returns = {card for card, seats in domains.items() if partner in seats}
            secure_partner = winner[0] == partner and not any(beats(card, winner[1]) for card in unseen)
            if secure_partner:
                # Retaining a master only helps if partner can actually return its suit.
                preserve = {card for card in preserve if any(other.suit == card.suit for other in returns)}
                reachable = {card for card in own_cards if card.suit != led_suit
                    and any(other.suit == card.suit for other in returns)
                    and not _established_control_blockers_v1(card, domain=domain,
                        contract=contract, trump=trump, cards=unseen)}
                if reachable:
                    preserve |= reachable
                    stranded |= {card for card in own_cards if card.suit != led_suit
                        and not any(other.suit == card.suit for other in returns)}
            # With two cards in the led suit, keep the lead rather than cheaply
            # ducking into an outstanding higher card and stranding our master.
            if all(card.suit == led_suit for card in own_cards):
                take_entry = {card for card in own_cards if beats(card, winner[1])
                    and not _established_control_blockers_v1(card, domain=domain,
                        contract=contract, trump=trump, cards=unseen)}

    scores: dict[str, int] = {}
    action_ids: dict[str, int] = {}
    winner_alignment: dict[str, int] = {}
    for row in normalized.legal_actions:
        code = str(row["code"])
        action = LegalPlayActionV1.from_code(code)
        card = action.card
        strength = domain.strength(card, contract, trump)
        points = domain.points(card, contract, trump)
        action_ids[code] = action.action_id
        if winner is None:
            score = _observer_safe_play_score_v1(strength, points)
            if policy_id in {NATIVE_ESTABLISHED_CONTROL_POLICY_ID} | PUBLIC_ENTRY_BASELINE_IDS and all(
                    other in own_cards or other in public_played
                    for other in _established_control_blockers_v1(card, domain=domain,
                        contract=contract, trump=trump, cards=map(Card, range(32)))):
                score += _established_control_bonus_v1(domain, contract, trump)
            if policy_id in PUBLIC_ENTRY_BASELINE_IDS and any(
                other.suit == card.suit for other in partner_masters):
                score += _established_control_bonus_v1(domain, contract, trump)
            if card in dead_leads:
                score -= _established_control_bonus_v1(domain, contract, trump)
        else:
            score = _observer_safe_play_score_v1(
                strength,
                points,
                partner_winning=winner[0].team == actor.team,
                takes_control=beats(card, winner[1]),
                feed_partner=(policy_id in NATIVE_PARTNER_FEED_POLICY_IDS or policy_id in PUBLIC_ENTRY_BASELINE_IDS) and (
                    policy_id != NATIVE_CLOSED_TRICK_FEED_POLICY_ID or len(plays) == 3),
                secure_control=policy_id == NATIVE_SECURE_CONTROL_POLICY_ID and len(plays) in (1, 2),
            )
            if (card in guards and winner[0].team != actor.team and card.suit != led_suit
                    and (contract.effective != Contract.HOKUM or card.suit != trump)):
                score -= _guard_discard_bonus_v1(domain, contract, trump)
            if card in stranded:
                score += _guard_discard_bonus_v1(domain, contract, trump)
            if card in preserve:
                score -= _guard_discard_bonus_v1(domain, contract, trump)
            if card in take_entry:
                score += _guard_discard_bonus_v1(domain, contract, trump)
        scores[code] = score
        resulting_winner = (
            (actor, card)
            if winner is None or beats(card, winner[1])
            else winner
        )
        winner_alignment[code] = int(
            preferred_completed_trick_winner is not None
            and len(plays) == 3
            and resulting_winner[0] == preferred_completed_trick_winner
        )
    if not scores:
        raise ValueError("observer-safe play policy requires legal actions")
    if policy_id == "observer_safe_tactical_category_v1":
        scores = {
            code: (score + 9_999) // 10_000 for code, score in scores.items()
        }

    model = cp_model.CpModel()
    choices = {code: model.new_bool_var(f"policy_action_{action_ids[code]}") for code in sorted(scores)}
    model.add_exactly_one(choices.values())
    score_var = model.new_int_var(min(scores.values()), max(scores.values()), "observer_safe_policy_score")
    action_var = model.new_int_var(min(action_ids.values()), max(action_ids.values()), "observer_safe_policy_action")
    model.add(score_var == sum(scores[code] * choices[code] for code in choices))
    model.add(action_var == sum(action_ids[code] * choices[code] for code in choices))
    action_span = max(action_ids.values()) - min(action_ids.values())
    tie_weight = action_span + 1
    alignment_var = model.new_int_var(0, 1, "scenario_winner_alignment")
    model.add(alignment_var == sum(winner_alignment[code] * choices[code] for code in choices))
    model.maximize(score_var * (2 * tie_weight) + alignment_var * tie_weight - action_var)
    proto = str(model.proto)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    solver.parameters.max_time_in_seconds = maximum_seconds
    status = _solve_cp_sat_with_cancel_v1(solver, model, cancel_check)
    if status != cp_model.OPTIMAL:
        raise InformationSetCompilationStopped(
            "policy_action_not_optimal", stage="observer_safe_play_policy",
            evidence={"solverStatus": solver.status_name(status).lower()},
        )
    selected = next(code for code, choice in choices.items() if solver.value(choice))
    actual = actual_action or selected
    if actual not in scores:
        raise ValueError("recorded play action is not legal for the observer-safe policy")
    best_score = max(scores.values())
    best = sorted(code for code, score in scores.items() if score == best_score)
    alignment_used = bool(
        preferred_completed_trick_winner is not None
        and len(best) > 1
        and winner_alignment[selected]
        and any(not winner_alignment[code] for code in best)
    )
    observation, observation_hash = _observer_safe_play_policy_observation_v1(
        actor=actor,
        contract=contract,
        trump=trump,
        trick_index=int(current["index"]),
        plays=plays,
        own_hand=state["hands"][actor.name.lower()],
        legal_actions=tuple(scores),
    )
    if public_played:
        observation["playedCards"] = sorted(card.code for card in public_played)
        observation_hash = canonical_sha256(observation)
    if public_knowledge is not None:
        observation["publicKnowledge"] = public_knowledge
        observation_hash = canonical_sha256(observation)
    graph_core = {
        "schemaVersion": 1,
        "kind": "observer_safe_play_policy_graph_v1",
        "observationHash": observation_hash,
        "legalActions": sorted(scores),
        "selectionPreference": (
            {
                "kind": "completed_trick_winner_equal_value_v1",
                "seat": preferred_completed_trick_winner.name.lower(),
            }
            if preferred_completed_trick_winner is not None and len(plays) == 3
            else None
        ),
    }
    graph = {**graph_core, "sha256": canonical_sha256(graph_core)}
    allocation_core = {
        "schemaVersion": 1,
        "kind": "observer_safe_public_projection_v1",
        "observationHash": observation_hash,
    }
    allocation = {**allocation_core, "sha256": canonical_sha256(allocation_core)}
    quality_core = {
        "schemaVersion": 1,
        "authority": "cp_sat",
        "scope": "observer_safe_policy_conditioned",
        "policyId": policy_id,
        "rootActorRelation": "target",
        "actualAction": actual,
        "selectedPolicyAction": selected,
        "selectionTieBreak": (
            "scenario_target_actor_equal_value_v1"
            if alignment_used
            else "stable_action_id" if len(best) > 1 else None
        ),
        "optimalActions": best,
        "bestValue": best_score,
        "actualValue": scores[actual],
        "regret": best_score - scores[actual],
        "actualActionOptimal": scores[actual] == best_score,
        "actionScores": [
            {"action": code, "value": scores[code]} for code in sorted(scores)
        ],
        "observerInformationHash": observation_hash,
        "solverStatus": "optimal",
        "variableCount": len(model.proto.variables),
        "constraintCount": len(model.proto.constraints),
        "modelProto": proto,
    }
    if policy_id in {PUBLIC_MASTER_RETENTION_BASELINE_ID, PUBLIC_EXHAUSTED_SUIT_BASELINE_ID, PUBLIC_TWO_TRICK_BASELINE_ID}:
        remaining = set(map(Card, range(32))) - public_played - {card for _, card in plays}
        quality_core["structuralFeatures"] = {
            "schemaVersion": 1,
            "contract": contract.effective.value,
            "currentWinnerRelation": ("none" if winner is None else
                "self" if winner[0] == actor else
                "partner" if winner[0].team == actor.team else "opponent"),
            "actions": [{"action": code,
                "role": "lead" if led_suit is None else "follow" if card.suit == led_suit else "discard",
                "ownMaster": not bool(_established_control_blockers_v1(card, domain=domain,
                    contract=contract, trump=trump, cards=remaining)),
                "partnerEntry": any(other.suit == card.suit for other in partner_masters),
                "stranded": card in stranded,
                "preserveMaster": card in preserve and winner is not None and winner[0].team == actor.team,
                "forcedReturnGuard": card in preserve and winner is not None and winner[0].team != actor.team,
            } for code in sorted(scores) for card in (LegalPlayActionV1.from_code(code).card,)],
        }
    quality = {**{key: value for key, value in quality_core.items() if key != "modelProto"},
               "sha256": canonical_sha256(quality_core)}
    proof_model_core = {
        "schemaVersion": 1,
        "kind": "observer_safe_play_policy_model_v1",
        "rootActorSeat": actor.name.lower(),
        "requestHash": request.sha256,
        "graphHash": graph["sha256"],
        "actionQuality": quality,
        "modelProto": proto,
    }
    proof_model = dict(proof_model_core)
    model_hash = canonical_sha256(proof_model_core)
    proof_core = {
        "schemaVersion": 1,
        "completionStatus": "bounded_complete",
        "solverStatus": "optimal",
        "claimSatisfied": True,
        "selectedAction": actual,
        "winningRootActionCount": None,
        "rootActions": [
            {"action": code, "actionId": action_ids[code], "value": scores[code],
             "satisfies": scores[code] == best_score}
            for code in sorted(scores)
        ],
        "principalVariation": [],
        "continuationLines": None,
        "policyIdentities": {actor.name.lower(): {
            "policyId": policy_id, "version": "1.0.0",
            "profileHash": canonical_sha256({"policyId": policy_id, "version": "1.0.0"}),
        }},
        "identities": {"graphHash": graph["sha256"], "modelHash": model_hash,
                       "proofRequestHash": request.sha256},
        "modelSize": {"variables": len(model.proto.variables),
                      "constraints": len(model.proto.constraints)},
    }
    proof = {**proof_core, "sha256": canonical_sha256(proof_core)}
    return selected, {"request": request.to_dict(), "proof": proof, "proofModel": proof_model,
                      "graph": graph, "allocationModel": allocation}


def _forced_quality_artifact_v1(
    normalized: OrToolsNormalizedExactStateV1,
    request: ScenarioProofRequestV1,
    *,
    actual_action: str | None,
    maximum_seconds: float,
    cancel_check: Callable[[], bool] | None,
    source_prefix: ExactTracePrefixV1 | None = None,
) -> tuple[str, dict[str, Any]]:
    """Certify a single legal project/play action without building a game tree."""

    if normalized.decision_kind not in {"project", "play"} or normalized.next_actor is None:
        raise ValueError("forced quality requires a live project or play decision")
    complete_state = request.visibility == "complete_state"
    if complete_state:
        normalized.require_integrity()
        if (not isinstance(source_prefix, ExactTracePrefixV1)
                or normalized.prefix_hash != _effective_prefix_hash(source_prefix)
                or request.target_team != Seat.parse(normalized.next_actor).team.name.lower()):
            raise ValueError("complete-state forced quality requires its owned prefix and acting team")
    actions = [str(row["code"]) for row in normalized.legal_actions]
    if len(actions) != 1:
        raise ValueError("forced quality requires exactly one legal action")
    if type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds) or maximum_seconds <= 0:
        raise ValueError("forced quality deadline must be finite and positive")
    selected = actions[0]
    actual = actual_action or selected
    if actual != selected:
        raise ValueError("recorded forced action is not the unique legal action")
    if cancel_check is not None and cancel_check():
        raise InformationSetCompilationStopped("cancelled", stage="forced_action_quality")

    model = cp_model.CpModel()
    chosen = model.new_bool_var("forced_legal_action")
    model.add(chosen == 1)
    proto = str(model.proto)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    solver.parameters.max_time_in_seconds = float(maximum_seconds)
    status = _solve_cp_sat_with_cancel_v1(solver, model, cancel_check)
    if status != cp_model.OPTIMAL or not solver.boolean_value(chosen):
        raise InformationSetCompilationStopped(
            "forced_action_not_certified",
            stage="forced_action_quality",
            evidence={"solverStatus": solver.status_name(status).lower()},
        )

    actor = Seat.parse(normalized.next_actor)
    observation_hash = canonical_sha256({
        "schemaVersion": 1,
        "kind": "forced_visible_action_v1",
        "exactStateId": normalized.exact_state_id,
        "actor": actor.name.lower(),
        "decisionKind": normalized.decision_kind,
        "legalActions": actions,
    })
    graph_core = {
        "schemaVersion": 1,
        "kind": "forced_legal_action_graph_v1",
        "observationHash": observation_hash,
        "legalActions": actions,
    }
    graph = {**graph_core, "sha256": canonical_sha256(graph_core)}
    allocation_core = {
        "schemaVersion": 1,
        "kind": "observer_safe_public_projection_v1",
        "observationHash": observation_hash,
    }
    if complete_state:
        allocation_core = {"schemaVersion": 1, "kind": "native_complete_state_quality_v1",
            "sourcePrefixHash": source_prefix.sha256, "exactStateId": normalized.exact_state_id,
            "rulesHash": normalized.rules_hash}
    allocation = {**allocation_core, "sha256": canonical_sha256(allocation_core)}
    completion = "exact_complete" if request.mode == "exact" else "bounded_complete"
    quality_core = {
        "schemaVersion": 1,
        "authority": "cp_sat",
        "scope": "compiled_complete_state" if complete_state else "compiled_actor_information_set",
        "rootActorRelation": "target",
        "actualAction": actual,
        "optimalActions": actions,
        "bestValue": None,
        "actualValue": None,
        "regret": 0,
        "actualActionOptimal": True,
        "solverStatus": "optimal",
        "variableCount": len(model.proto.variables),
        "constraintCount": len(model.proto.constraints),
        "analysisScope": {
            "mode": request.mode,
            "requestHash": request.sha256,
            "coverage": "single_legal_action",
            "observerInformationHash": observation_hash,
        },
        "modelProto": proto,
    }
    quality = {
        **{key: value for key, value in quality_core.items() if key != "modelProto"},
        "sha256": canonical_sha256(quality_core),
    }
    proof_model = {
        "schemaVersion": 1,
        "kind": "forced_action_optimality_model_v1",
        "rootActorSeat": actor.name.lower(),
        "requestHash": request.sha256,
        "graphHash": graph["sha256"],
        "actionQuality": quality,
        "modelProto": proto,
    }
    model_hash = canonical_sha256(proof_model)
    proof_core = {
        "schemaVersion": 1,
        "completionStatus": completion,
        "completionScope": "action_optimality_only",
        "solverStatus": "optimal",
        "claimSatisfied": None,
        "selectedAction": actual,
        "winningRootActionCount": None,
        "winningRootActionLowerBound": 0,
        "winningRootActionUpperBound": 1,
        "rootActions": [{"action": actual, "actionId": 0, "value": None, "satisfies": None}],
        "principalVariation": [],
        "continuationLines": None,
        "policyIdentities": {},
        "identities": {
            "graphHash": graph["sha256"],
            "modelHash": model_hash,
            "proofRequestHash": request.sha256,
        },
        "modelSize": {
            "variables": len(model.proto.variables),
            "constraints": len(model.proto.constraints),
        },
    }
    proof = {**proof_core, "sha256": canonical_sha256(proof_core)}
    return selected, {
        "request": request.to_dict(),
        "proof": proof,
        "proofModel": proof_model,
        "graph": graph,
        "allocationModel": allocation,
    }


def _select_quality_play_action_v1(
    rules: RuleBook,
    current: ExactTracePrefixV1,
    normalized: OrToolsNormalizedExactStateV1,
    request: ScenarioProofRequestV1,
    *,
    quality_policy: ScenarioPlayingQualityPolicyV1 | None,
    execution_budget: ScenarioQualityExecutionBudgetV1 | None,
    budget: ScenarioProofBudgetV1 | None,
    featured_now: bool,
    actual_action: str | None,
    admissible_root_actions: tuple[str, ...] | None = None,
    preferred_completed_trick_winner: Seat | None = None,
    remaining_seconds: Callable[[], float],
    cancel_check: Callable[[], bool] | None,
    private_recall: Mapping[str, Sequence[str]] | None,
    maximum_worlds: int,
    maximum_depth: int,
    escalation_events: Sequence[Mapping[str, Any]] | None = None,
    opportunity_order: Sequence[str] | None = None,
    maximum_conflicts: int | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[dict[str, Any] | None, OrToolsGraphProofOutcomeV1 | None,
           dict[str, Any] | None, str | None]:
    """Choose a play with the same forced, early-policy, or native proof path."""
    if normalized.decision_kind != "play":
        raise ValueError("quality play selector requires a play decision")
    if len(normalized.legal_actions) == 1:
        _, artifact = _forced_quality_artifact_v1(
            normalized, request, actual_action=actual_action,
            maximum_seconds=remaining_seconds(), cancel_check=cancel_check,
        )
        return artifact, None, None, "forcedSelectionSeconds"
    current_trick = normalized.state.get("currentTrick")
    if (quality_policy is not None and quality_policy.mode == "budgeted"
            and execution_budget is not None
            and execution_budget.early_play_policy_id is not None
            and isinstance(current_trick, Mapping)
            and int(current_trick["index"]) + 1 < int(execution_budget.exact_play_from_trick)
            and not featured_now):
        _, artifact = _observer_safe_play_policy_artifact_v1(
            rules, normalized, request, actual_action=actual_action,
            policy_id=execution_budget.early_play_policy_id,
            preferred_completed_trick_winner=preferred_completed_trick_winner,
            maximum_seconds=remaining_seconds(), cancel_check=cancel_check,
        )
        return artifact, None, None, "policySelectionSeconds"
    if request.mode == "bounded" and budget is not None and budget.native_observed is not None:
        compiler = OrToolsActorInformationSetCompilerV1(
            rules, native_cp_sat=True,
            prune_own_project_pass_worlds=bool(
                execution_budget and execution_budget.prune_own_project_pass_worlds
            ),
        )
        if (not featured_now and request.claim in {"outcome_reachable", "winning_root_action_count"}
                and request.line_universe is None):
            from .result_authority import _project_native_bounded_play_quality_artifact_v1
            try:
                execution = compiler._compile_observed_play_selection(normalized, request, budget=budget,
                    maximum_seconds=remaining_seconds(), cancel_check=cancel_check,
                    private_recall=private_recall, maximum_worlds=maximum_worlds,
                    maximum_depth=maximum_depth, project_prefix=current,
                    **({"admissible_root_actions": admissible_root_actions}
                       if admissible_root_actions is not None else {}))
            except InformationSetCompilationStopped as exc:
                if exc.reason != "no_optimal_witness_action":
                    raise
                raise _QualityScenarioTargetMiss(
                    "no action optimal under the configured quality proof preserves the required future card counts",
                    evidence=exc.partial_receipt) from exc
            artifact = _project_native_bounded_play_quality_artifact_v1(execution, rules=rules,
                source_prefix=current, normalized=normalized, request=request, budget=budget)
            return artifact, None, None, "nativeExecutionSeconds"
        proof, native_execution = compiler._prove_observed_play(
            normalized, request, budget=budget,
            maximum_seconds=remaining_seconds(), cancel_check=cancel_check,
            private_recall=private_recall, maximum_worlds=maximum_worlds,
            maximum_depth=maximum_depth, project_prefix=current,
        )
        return None, proof, native_execution, "nativeExecutionSeconds"
    if (not featured_now and request.mode == "exact" and request.visibility == "actor_information_set"
            and request.utility == "awarded_point_margin" and request.line_universe is None
            and budget is not None and budget.native_observed is not None
            and current.project_actions_present
            and escalation_events is not None and opportunity_order is not None
            and maximum_conflicts is not None):
        artifact = _observed_play_quality_artifact_v1(rules, normalized, current, request,
            budget=budget, opportunity_order=opportunity_order, escalation_events=escalation_events,
            maximum_conflicts=maximum_conflicts, maximum_depth=maximum_depth,
            maximum_seconds=remaining_seconds(), cancel_check=cancel_check,
            progress_callback=progress_callback)
        return artifact, None, None, "nativeExecutionSeconds"
    return None, None, None, None


def _editorial_play_quality_artifact_v1(rules, normalized, prefix, request, *, budget,
        nodes, depth, worlds, remaining_seconds, declared_seconds, cancel_check,
        private_recall, graph_workers, worker_cpu_callback):
    if request.visibility == "complete_state":
        return _complete_state_quality_artifact_v1(rules, normalized, prefix, {"kind": "play"}, request,
            budget=budget, maximum_nodes=nodes, maximum_depth=depth,
            maximum_seconds=remaining_seconds(), cancel_check=cancel_check)
    compilation = compile_observed_decision_v1(rules, normalized, prefix, {"kind": "play"}, request,
        maximum_nodes=nodes, maximum_depth=depth, maximum_worlds=worlds,
        maximum_seconds=remaining_seconds(), declared_maximum_seconds=declared_seconds,
        native_cp_sat=True, private_recall=private_recall, cancel_check=cancel_check,
        graph_workers=graph_workers, worker_cpu_callback=worker_cpu_callback)
    root_witnesses = []
    proof = prove_actor_information_set_v1(compilation, request, maximum_nodes=nodes, maximum_worlds=worlds,
        maximum_seconds=remaining_seconds(), declared_maximum_seconds=declared_seconds,
        native_action_quality=True, cancel_check=cancel_check, policy_search=budget.policy_search if budget else None,
        _root_witnesses=root_witnesses)
    return {"request": request.to_dict(), "proof": proof.to_dict(), "proofModel": proof.model_document(),
        "graph": compilation.graph.to_dict(), "allocationModel": dict(compilation.graph.model_document),
        **({"rootWitnesses": root_witnesses} if root_witnesses else {}),
        "sourcePrefix": prefix.to_dict()}


def _project_inference_quality_evidence_v1(rules, normalized, prefix, artifact, round_win, *,
        budget, maximum_nodes, maximum_depth, maximum_conflicts, remaining_seconds, cancel_check):
    """A public rank excludes a legal no-salvage world, not an optimum ablation."""
    from baloot_platform.core.exact_state_view import build_canonical_observer_information_v1
    from .lesson_tags import _exact_values, _artifact_matches_state
    from .negative_project_prefix import probe_public_project_rank_ownership_v1
    from .result_authority import native_project_counterexample_request_v1, native_round_win_quality_request_v1

    request = ScenarioProofRequestV1.build(artifact['request'])
    action = artifact['proof'].get('selectedAction')
    actions = {row['code'] for row in normalized.legal_actions}
    wins = _exact_values(round_win, action)
    if (request.mode != 'exact' or request.visibility != 'actor_information_set'
            or normalized.decision_kind != 'play' or len(actions) < 2
            or wins is None or set(wins) != actions or wins[action] != 1
            or round_win.get('request') != native_round_win_quality_request_v1(request).to_dict()
            or not _artifact_matches_state(round_win, normalized.exact_state_id)
            or normalized.state['contract'] != 'sun' or prefix.escalation_actions
            or prefix.adjudication_events or len(prefix.project_actions) != 4
            or len(prefix.play_actions) < 8
            or rules.project_domain.declaration_mode != 'mandatory'
            or normalized.state['escalation'] != 'normal'
            or normalized.state['winningBidder'] != normalized.state['buyer']
            or normalized.state['buyer'] != normalized.state['purchaseRecipient']):
        return None
    public = build_canonical_observer_information_v1(
        complete_state=normalized.state, acting_seat=normalized.next_actor)['publicState']
    seen = {Card.parse(row['card']) for row in prefix.play_actions}
    seen.update(map(Card.parse, public['hands'][normalized.next_actor]))
    seen.update(Card.parse(c) for p in public['projects'] for c in (p['cards'] or ()))
    clues = [p for p in public['projects'] if p['owner'] != normalized.next_actor
             and not p['revealed'] and p['highRank'] is not None and p['declarationOrder'] < 999]
    # Only public top-rank cards propose assumptions; native constraints decide ownership.
    for clue in clues:
        for card in FULL_DECK:
            if card in seen or int(card.rank) != clue['highRank']:
                continue
            probe = probe_public_project_rank_ownership_v1(rules=rules, source_prefix=prefix,
                normalized=normalized, project_owner=clue['owner'], card=card, expected_owner=clue['owner'],
                maximum_seconds=remaining_seconds(), maximum_conflicts=maximum_conflicts, cancel_check=cancel_check)
            remaining_seconds()
            if probe['status'] == 'incomplete':
                raise InformationSetCompilationStopped('project_inference_ownership_incomplete',
                    stage='project_inference', evidence={'ownershipProbe': probe})
            if probe['status'] != 'clue_dependent':
                continue
            witness = ExactTracePrefixV1.build(probe['contraryWitness'])
            state = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(
                witness, maximum_seconds=remaining_seconds(), cancel_check=cancel_check)
            contrary = _complete_state_quality_artifact_v1(rules, state, witness, {'kind': 'play'},
                native_project_counterexample_request_v1(request), budget=budget,
                maximum_nodes=maximum_nodes, maximum_depth=maximum_depth,
                maximum_seconds=remaining_seconds(), cancel_check=cancel_check)
            remaining_seconds()
            if contrary['proof']['completionStatus'] != 'exact_complete':
                raise InformationSetCompilationStopped('project_inference_counterexample_incomplete',
                    stage='project_inference', evidence={'contraryRoundWin': contrary})
            losing = _exact_values(contrary, action)
            if losing is None or set(losing) != actions:
                raise ValueError('project inference counterexample root coverage mismatch')
            if set(losing.values()) != {-1}:
                continue
            core = {'schemaVersion': 1, 'kind': 'native_public_project_inference_v1',
                'scope': 'public_rank_excludes_no_salvage_world', 'rulesHash': rules.sha256,
                'sourceArtifactHash': canonical_sha256(artifact), 'sourcePrefixHash': prefix.sha256,
                'exactStateId': normalized.exact_state_id, 'action': action,
                'sourceRoundWinHash': canonical_sha256(round_win), 'ownershipProbe': probe,
                'contraryRoundWin': contrary}
            return {**core, 'sha256': canonical_sha256(core)}
    return None


def _observed_play_quality_artifact_v1(rules, normalized, prefix, request, *, budget,
        opportunity_order, escalation_events, maximum_conflicts, maximum_depth,
        maximum_seconds, cancel_check=None, decision_kind="play", progress_callback=None):
    """Construct and reverify selected and best values under one shared allowance."""
    from .result_authority import (
        _build_native_observed_play_quality_artifact_v1, native_observed_best_strategy_action_v1,
    )
    if (not isinstance(budget, ScenarioProofBudgetV1) or budget.native_observed is None
            or request.mode != "exact" or request.visibility != "actor_information_set"
            or request.utility != "awarded_point_margin" or request.line_universe is not None
            or request.budget_profile != budget.profile_id
            or opportunity_order is None or escalation_events is None
            or type(maximum_seconds) not in {int, float}
            or not math.isfinite(maximum_seconds) or maximum_seconds <= 0):
        raise ValueError("observed play quality requires owned history and frozen exact budgets")
    seconds = min(maximum_seconds, budget.maximum_wall_seconds, budget.maximum_cpu_seconds)
    deadline, cpu_deadline = time.monotonic() + seconds, time.process_time() + seconds
    native = budget.native_observed
    # Reserve the separate-proof worst case; a certified root choice can reuse the best proof.
    queries = (native.maximum_queries - 1) // 4
    if queries < 1:
        raise InformationSetCompilationStopped("maximum_queries", stage="observed_play_quality")

    def stopped():
        return bool(cancel_check and cancel_check()) or time.monotonic() >= deadline or time.process_time() >= cpu_deadline

    def remaining():
        if cancel_check is not None and cancel_check():
            raise InformationSetCompilationStopped("cancelled", stage="observed_play_quality")
        value = min(deadline - time.monotonic(), cpu_deadline - time.process_time())
        if value <= 0:
            raise InformationSetCompilationStopped(
                "maximum_seconds" if time.monotonic() >= deadline else "maximum_cpu_seconds",
                stage="observed_play_quality")
        return value

    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)

    def run_owned(call, *args, **kwargs):
        try:
            return call(*args, **kwargs)
        except InformationSetCompilationStopped as error:
            if error.partial_receipt.get("exhaustionReason") != "cancelled" or cancel_check and cancel_check():
                raise
            reason = ("maximum_seconds" if time.monotonic() >= deadline else
                      "maximum_cpu_seconds" if time.process_time() >= cpu_deadline else None)
            if reason is None:
                raise
            raise InformationSetCompilationStopped(reason, stage="observed_play_quality",
                evidence={"causeReceipt": error.partial_receipt}) from error

    def prove(action):
        phase = "best" if action is None else "selected"
        receipt = run_owned(compiler._solve_native_post_auction_policy_bounds, normalized, prefix,
            Seat.parse(normalized.next_actor).team, selected_root_action=action,
            opportunity_order=opportunity_order, escalation_events=escalation_events,
            maximum_iterations=native.maximum_iterations, maximum_queries=queries,
            maximum_observations=native.maximum_observations, maximum_model_bytes=budget.maximum_cache_bytes,
            maximum_conflicts=maximum_conflicts,
            maximum_depth=min(maximum_depth, budget.maximum_depth, budget.maximum_horizon_actions),
            maximum_seconds=remaining(), cancel_check=stopped, visibility="actor_information_set",
            compact_public_ownership=True,
            observed_play_bounds_version=2, path_floor_version=2,
            progress_callback=(lambda row: progress_callback({**row, "decisionKind": decision_kind,
                "proofPhase": phase})) if progress_callback is not None else None)
        remaining()
        if receipt["status"] != "scoped_bound_converged":
            raise InformationSetCompilationStopped("native_observed_play_quality_incomplete",
                stage="observed_play_quality", evidence={"bounds": receipt})
        return receipt

    if progress_callback is not None:
        try:
            progress_callback({"kind": "quality_solver_progress_v1", "phase": "preparation",
                "masterCount": 0, "queryCount": 0, "bindingCount": 0,
                "decisionKind": decision_kind, "proofPhase": "best"})
        except Exception:
            pass  # Operational progress cannot change proof semantics.
    best = prove(None)
    action = request.selected_action
    if action is None:
        # Equal bounds may converge before the master binds the root. Probe a
        # stable legal action; without a certified choice, prove it separately.
        action = best["strategy"]["policy"].get(best["preparation"]["rootObservationHash"],
            min(row["code"] for row in normalized.legal_actions))
        if action not in {row["code"] for row in normalized.legal_actions}:
            raise InformationSetCompilationStopped("native_root_policy_missing",
                stage="observed_play_quality", evidence={"bounds": best})
    selected = best if native_observed_best_strategy_action_v1(best) == action else prove(action)
    if request.selected_action is None and selected["strategy"]["lowerBound"] != best["strategy"]["upperBound"]:
        raise InformationSetCompilationStopped("native_selected_policy_not_optimal",
            stage="observed_play_quality", evidence={"selectedBounds": selected, "bestBounds": best})
    return run_owned(_build_native_observed_play_quality_artifact_v1, selected, best, rules=rules,
        source_prefix=prefix, normalized=normalized, request=request, budget=budget,
        opportunity_order=opportunity_order, escalation_events=escalation_events,
        maximum_conflicts=maximum_conflicts, maximum_seconds=remaining(), cancel_check=stopped,
        decision_kind=decision_kind, verified_bounds=frozenset(row["sha256"] for row in (selected, best)))


def _observed_bid_quality_artifact_v1(rules, normalized, prefix, request, *, budget,
        execution_budget, maximum_conflicts, maximum_depth, maximum_seconds, cancel_check=None,
        progress_callback=None, _maximum_queries=None, _selected_action=None, _prior_selection_attempts=None,
        _staged_bid_selection=False, _defer_bid_fallback=False):
    """Audit a recorded bid with one governed construction/verification allowance."""
    from .result_authority import _build_native_observed_bid_quality_artifact_v1, _prior_bid_alternative_receipts_v1
    if (not isinstance(budget, ScenarioProofBudgetV1) or budget.native_observed is None
            or not isinstance(execution_budget, ScenarioQualityExecutionBudgetV1)
            or type(maximum_conflicts) is not int or maximum_conflicts < 1
            or request.mode != "exact" or request.visibility != "actor_information_set"
            or request.utility != "awarded_point_margin"
            or request.selected_action is None and _selected_action is None
            or request.budget_profile != budget.profile_id):
        raise ValueError("observed bid quality requires frozen native budgets and a recorded action")
    selected_action = request.selected_action if request.selected_action is not None else _selected_action
    if selected_action not in {row["code"] for row in normalized.legal_actions}:
        raise ValueError("observed bid selection must be legal")
    started = time.monotonic()
    if type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds) or maximum_seconds <= 0:
        raise ValueError("observed bid quality requires finite positive time")
    seconds = min(maximum_seconds, budget.maximum_wall_seconds, budget.maximum_cpu_seconds)
    native = budget.native_observed
    queries = native.maximum_queries if _maximum_queries is None else _maximum_queries
    if type(queries) is not int or not 0 < queries <= native.maximum_queries:
        raise ValueError("observed bid query allowance exceeds the frozen budget")
    order = execution_budget.escalation_opportunity_order
    policy_args = ({"opponent_play_policy_id": execution_budget.bid_opponent_play_policy_id}
                   if execution_budget.bid_opponent_play_policy_id is not None else {})
    incumbent = execution_budget.bid_incumbent_proof
    alternative = execution_budget.bid_alternative_proof
    refutation = execution_budget.bid_threshold_refutation
    # Frozen stage times are ceilings, not minimum admission allowances. The
    # compiler clips each stage to the remaining shared decision deadline.
    if refutation is not None and refutation["maximumQueries"] >= queries:
        raise InformationSetCompilationStopped("refutation_allowance_exceeds_decision_budget",
            stage="observed_bid_quality")
    if alternative is not None and alternative["maximumQueries"] >= queries:
        raise InformationSetCompilationStopped("alternative_allowance_exceeds_decision_budget",
            stage="observed_bid_quality")
    if incumbent is not None and incumbent["schemaVersion"] == 2:
        if incumbent["maximumQueries"] >= queries:
            raise InformationSetCompilationStopped("incumbent_allowance_exceeds_decision_budget",
                stage="observed_bid_quality")
    attempt = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)._prove_observed_bid_dominance(
        normalized, prefix, selected_action=selected_action,
        opponent_policy={"auction_action": "pass", "escalation_action": "decline"},
        maximum_iterations=native.maximum_iterations, maximum_queries=queries,
        maximum_observations=native.maximum_observations, maximum_branches=budget.maximum_state_space,
        maximum_model_bytes=budget.maximum_cache_bytes, maximum_conflicts=maximum_conflicts,
        maximum_depth=min(maximum_depth, budget.maximum_depth, budget.maximum_horizon_actions),
        maximum_seconds=seconds, opportunity_order=order, cancel_check=cancel_check,
        progress_callback=(lambda row: progress_callback({**row, "decisionKind": "bid",
            "proofPhase": "selected"})) if progress_callback is not None else None,
        reserve_artifact_verification=True, **policy_args,
        _defer_bid_fallback=_defer_bid_fallback,
        _prior_alternatives=_prior_bid_alternative_receipts_v1(_prior_selection_attempts,
            source_prefix_hash=prefix.sha256, rules_hash=rules.sha256,
            maximum_seconds=min(budget.maximum_wall_seconds, budget.maximum_cpu_seconds)),
        **({"strategy_play_fallback_id": execution_budget.observed_bid_strategy_play_fallback_id}
           if execution_budget.observed_bid_strategy_play_fallback_id is not None else {}),
        **({"minimax_budget": budget}
           if execution_budget.observed_bid_strategy_play_fallback_id is not None else {}),
        **({"alternative_proof": alternative}
           if alternative is not None else {}),
        **({"threshold_refutation_probe": refutation} if refutation is not None else {}),
        **({"maximum_evidence_bytes": execution_budget.maximum_evidence_bytes}
           if alternative is not None or refutation is not None else {}),
        **({"incumbent_proof": execution_budget.observed_bid_incumbent_proof}
           if incumbent is not None and incumbent["schemaVersion"] == 2 else {}))
    remaining = seconds - (time.monotonic() - started)
    if attempt["status"] != "selected_action_optimal" or remaining <= 0:
        raise InformationSetCompilationStopped("native_observed_bid_quality_incomplete", stage="observed_bid_quality",
            evidence={"nativeObservedBidDominance": attempt})
    if attempt["queriesUsed"] + attempt["certificate"]["limits"]["maximumQueries"] > queries:
        raise ValueError("observed bid artifact verification exceeds the remaining query allowance")
    if request.selected_action is None:
        core = {key: value for key, value in attempt.items() if key != "sha256"}
        core["priorSelectionAttempts"] = list(_prior_selection_attempts or ())
        if _staged_bid_selection:
            core["selectionSchedule"] = "incumbent_then_deep_v1"
        attempt = {**core, "sha256": canonical_sha256(core)}
    return _build_native_observed_bid_quality_artifact_v1(attempt, rules=rules, source_prefix=prefix,
        normalized=normalized, request=request, budget=budget, maximum_conflicts=maximum_conflicts,
        opportunity_order=order, maximum_seconds=remaining, cancel_check=cancel_check,
        execution_budget=execution_budget,
        verified_certificates=frozenset({attempt["certificate"]["sha256"]}), **policy_args)


def _select_observed_bid_quality_artifact_v1(rules, normalized, prefix, request, *, budget,
        execution_budget, maximum_conflicts, maximum_depth, maximum_seconds, cancel_check=None,
        progress_callback=None, maximum_nodes=None, decision=None, opportunity_order=None,
        _search_generated=False):
    """Try legal bids through the existing certificate path, without resetting resources."""
    if (not isinstance(budget, ScenarioProofBudgetV1) or budget.native_observed is None
            or not isinstance(execution_budget, ScenarioQualityExecutionBudgetV1)
            or type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds)
            or maximum_seconds <= 0 or request.selected_action is not None):
        raise ValueError("generated bid selection requires an unselected request")
    started = time.monotonic()
    seconds = min(maximum_seconds, budget.maximum_wall_seconds, budget.maximum_cpu_seconds)
    queries_left = budget.native_observed.maximum_queries
    attempts = []
    actions = [row["code"] for row in normalized.legal_actions]
    staged = bool(_search_generated and execution_budget.bid_incumbent_proof is not None
        and (request.visibility == "complete_state"
             or execution_budget.bid_incumbent_proof["schemaVersion"] == 2))
    schedule = actions * (2 if staged else 1)
    bid_compiler = (OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
        if request.visibility == "complete_state" else None)
    for index, action in enumerate(schedule):
        if cancel_check is not None and cancel_check():
            raise InformationSetCompilationStopped("cancelled", stage="observed_bid_selection",
                evidence={"attempts": attempts})
        remaining = seconds - (time.monotonic() - started)
        quota = queries_left // (len(schedule) - index)
        if remaining <= 0 or quota < 1:
            break
        # Both scopes need alternative bounds and fresh verification. Borrow
        # unused time from later bids without resetting the shared deadline.
        action_seconds = remaining
        try:
            if request.visibility == "complete_state":
                return _complete_state_quality_artifact_v1(rules, normalized, prefix, decision, request,
                    budget=budget, execution_budget=execution_budget, maximum_conflicts=maximum_conflicts,
                    maximum_nodes=maximum_nodes, maximum_depth=maximum_depth,
                    maximum_seconds=action_seconds, opportunity_order=opportunity_order,
                    _maximum_queries=quota, _selected_action=action, _prior_selection_attempts=attempts,
                    _bid_compiler=bid_compiler,
                    _staged_bid_selection=staged,
                    _defer_bid_fallback=staged and index < len(actions),
                    _search_generated=_search_generated,
                    cancel_check=cancel_check, progress_callback=progress_callback)
            return _observed_bid_quality_artifact_v1(rules, normalized, prefix, request,
                budget=budget, execution_budget=execution_budget, maximum_conflicts=maximum_conflicts,
                maximum_depth=maximum_depth, maximum_seconds=action_seconds,
                _maximum_queries=quota, _selected_action=action, _prior_selection_attempts=attempts,
                _staged_bid_selection=staged,
                _defer_bid_fallback=staged and index < len(actions),
                cancel_check=cancel_check, progress_callback=progress_callback)
        except InformationSetCompilationStopped as stopped:
            # Incomplete nested stages may not expose a complete query count.
            # Reserve their entire allowance; never grant it again to the next bid.
            queries_left -= quota
            attempts.append({"action": action, "queryAllowance": quota,
                "queriesCharged": quota, "receipt": stopped.partial_receipt})
            if progress_callback is not None:
                progress_callback({"kind": "quality_bid_selection_attempt_v1",
                    "decisionKind": "bid", "action": action, "status": "incomplete",
                    "attempt": attempts[-1]})
            if cancel_check is not None and cancel_check():
                raise InformationSetCompilationStopped("cancelled", stage="observed_bid_selection",
                    evidence={"attempts": attempts}) from stopped
            continue
    raise InformationSetCompilationStopped("native_observed_bid_selection_incomplete",
        stage="observed_bid_selection", evidence={"attempts": attempts})


def _complete_state_quality_artifact_v1(
    rules, normalized, prefix, decision, request, *, maximum_nodes,
    maximum_depth, maximum_seconds, cancel_check, opportunity_order=None,
    prior_non_announcing_seats=(), prior_window_recall=None,
    budget=None, maximum_conflicts=None, escalation_events=None, execution_budget=None,
    progress_callback=None, _maximum_queries=None, _selected_action=None, _prior_selection_attempts=None,
    _search_generated=False,
    _bid_compiler=None,
    _staged_bid_selection=False, _defer_bid_fallback=False,
):
    """Certify root quality using the same native legal complete-state graph."""
    started = time.monotonic()
    project_dominance_attempt = None

    def remaining():
        if cancel_check is not None and cancel_check():
            raise InformationSetCompilationStopped("cancelled", stage="complete_state_quality")
        seconds = maximum_seconds - (time.monotonic() - started)
        if seconds <= 0:
            raise InformationSetCompilationStopped("maximum_seconds", stage="complete_state_quality")
        return seconds

    if normalized.decision_kind in {"project", "play"} and len(normalized.legal_actions) == 1:
        normalized.require_integrity()
        if normalized.rules_hash != rules.sha256:
            raise ValueError("complete-state forced quality differs from frozen rules")
        _, artifact = _forced_quality_artifact_v1(normalized, request,
            actual_action=request.selected_action, maximum_seconds=remaining(),
            cancel_check=cancel_check, source_prefix=prefix)
        return artifact

    if (normalized.decision_kind == "project" and request.mode == "exact"
            and request.visibility == "complete_state" and request.utility == "awarded_point_margin"
            and normalized.state.get("contract") == "sun" and len(prefix.project_actions) == 3
            and len(prefix.play_actions) == 3
            and budget is not None and budget.native_observed is not None and maximum_conflicts is not None):
        from .result_authority import build_native_project_dominance_quality_artifact_v1
        compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
        prove = (compiler._select_last_project_dominance if request.selected_action is None
                 else compiler._prove_last_project_dominance)
        certificate = prove(
            normalized, prefix, request, maximum_seconds=remaining(), cancel_check=cancel_check,
            budget=budget, maximum_conflicts=maximum_conflicts, maximum_depth=maximum_depth)
        if certificate["status"] == "strictly_dominant":
            return build_native_project_dominance_quality_artifact_v1(certificate, rules=rules,
                source_prefix=prefix, normalized=normalized, request=request, maximum_seconds=remaining(),
                budget=budget, maximum_conflicts=maximum_conflicts, cancel_check=cancel_check, maximum_depth=maximum_depth)
        if certificate["status"] not in {"counterexample_found", "no_strictly_dominant_action"}:
            raise ValueError("project dominance result is unsupported")
        # Failure of this sufficient theorem does not rule out minimax optimality.
        project_dominance_attempt = certificate

    if (normalized.decision_kind == "bid" and request.mode == "exact"
            and request.visibility == "complete_state" and request.utility == "awarded_point_margin"
            and request.line_universe is None and maximum_conflicts is not None):
        from .result_authority import build_native_bid_dominance_quality_artifact_v1, validate_native_bid_dominance_budget_v1
        if not isinstance(budget, ScenarioProofBudgetV1) or budget.native_observed is None:
            raise ValueError("native bid quality requires a frozen native budget")
        if request.selected_action is None and _selected_action is None:
            return _select_observed_bid_quality_artifact_v1(rules, normalized, prefix, request,
                budget=budget, execution_budget=execution_budget, maximum_conflicts=maximum_conflicts,
                maximum_nodes=maximum_nodes, maximum_depth=maximum_depth, maximum_seconds=remaining(),
                decision=decision, opportunity_order=opportunity_order, cancel_check=cancel_check,
                progress_callback=progress_callback, _search_generated=_search_generated)
        native = budget.native_observed
        selected_action = request.selected_action if request.selected_action is not None else _selected_action
        queries = native.maximum_queries if _maximum_queries is None else _maximum_queries
        if (selected_action not in {row["code"] for row in normalized.legal_actions}
                or type(queries) is not int or not 0 < queries <= native.maximum_queries):
            raise ValueError("native bid selection or query allowance is invalid")
        decision_limit = min(maximum_seconds, budget.maximum_wall_seconds, budget.maximum_cpu_seconds)
        decision_seconds = min(remaining(), decision_limit)
        interval_args = {}
        native_incumbent_models = []
        if execution_budget is not None:
            if not isinstance(execution_budget, ScenarioQualityExecutionBudgetV1):
                raise ValueError("native bid quality requires a frozen quality execution configuration")
            execution_budget = ScenarioQualityExecutionBudgetV1.build(execution_budget.to_dict())
            if execution_budget.bid_opponent_play_policy_id is not None:
                interval_args["opponent_play_policy_id"] = execution_budget.bid_opponent_play_policy_id
            incumbent = execution_budget.bid_incumbent_proof
            if incumbent is not None:
                if incumbent["maximumQueries"] >= queries:
                    raise InformationSetCompilationStopped("incumbent_allowance_exceeds_decision_budget",
                        stage="complete_state_quality")
                team = Seat.parse(normalized.next_actor).team
                interval_args.update({"incumbent_maximum_seconds": incumbent["maximumSeconds"],
                    "incumbent_maximum_queries": incumbent["maximumQueries"],
                    **({"incumbent_native_play_policy_id": incumbent["nativePlayPolicyId"],
                        **({"incumbent_nonplay_policy": incumbent["completeStateNonplayPolicy"]}
                           if "completeStateNonplayPolicy" in incumbent else {}),
                        "_owned_incumbent_models": native_incumbent_models,
                        "alternative_first": incumbent.get("completeStateAlternativeFirst", True)} if incumbent["schemaVersion"] == 2 else
                       {"incumbent_play_policies": {seat: raw for seat, raw in incumbent["playPolicies"].items()
                        if Seat.parse(seat).team == team}})})
        compiler = _bid_compiler or OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
        prove_args = dict(
            selected_root_action=selected_action,
            opponent_strategy={"auction_action": "pass", "escalation_action": "decline"},
            maximum_iterations=native.maximum_iterations, maximum_queries=queries,
            maximum_observations=native.maximum_observations,
            maximum_branches=budget.maximum_state_space, maximum_model_bytes=budget.maximum_cache_bytes,
            maximum_conflicts=maximum_conflicts,
            maximum_depth=min(maximum_depth, budget.maximum_depth, budget.maximum_horizon_actions),
            maximum_seconds=decision_seconds,
            opportunity_order=opportunity_order, cancel_check=cancel_check)
        prior_alternatives = []
        for prior in _prior_selection_attempts or ():
            parent = prior.get("receipt", {}).get("evidence", {}).get("nativeBidDominance", {})
            basis = parent.get("incumbentAttempt", parent)
            if (basis.get("kind") == "native_bid_selected_action_dominance_v3"
                    and basis.get("sourcePrefixHash") == prefix.sha256
                    and basis.get("stateHash") == normalized.state_hash
                    and basis.get("rulesHash") == rules.sha256):
                prior_alternatives.extend(row["preparation"] for row in basis.get("alternatives", ())
                    if isinstance(row, Mapping) and isinstance(row.get("preparation"), Mapping))
        certificate = compiler._prove_native_bid_selected_action(
            normalized, prefix, Seat.parse(normalized.next_actor).team,
            **prove_args,
            **({"_prior_alternatives": tuple(prior_alternatives)} if prior_alternatives else {}),
            progress_callback=(lambda row: progress_callback({**row, "decisionKind": "bid",
                "proofPhase": "selected"})) if progress_callback is not None else None,
            **interval_args)
        if (not _defer_bid_fallback and certificate["status"] == "incomplete" and "incumbent_maximum_seconds" in interval_args
                and not (isinstance(certificate.get("cause"), Mapping)
                    and certificate["cause"].get("reason") == "selected_threshold_exceeds_certified_upper")
                and type(certificate.get("queriesUsed")) is int
                and 0 <= certificate["queriesUsed"] < queries):
            validate_native_bid_dominance_budget_v1(certificate, budget,
                maximum_conflicts=maximum_conflicts, opportunity_order=opportunity_order,
                execution_budget=execution_budget, _incomplete_incumbent=True)
            # A policy certificate miss does not refute the action. Both origins
            # may use deeper synthesis within the remaining shared budget.
            attempt = certificate
            if progress_callback is not None:
                progress_callback({"schemaVersion": 1, "kind": "quality_bid_incumbent_checkpoint_v1",
                    "decisionKind": "bid", "action": selected_action, "status": "incomplete",
                    "receipt": attempt})
            native_incumbent_models.clear()
            try:
                # The incumbent allowance is a cap, not time already consumed.
                fallback_seconds = remaining() - (maximum_seconds - decision_limit)
                if fallback_seconds <= 0:
                    raise InformationSetCompilationStopped("maximum_seconds", stage="complete_state_quality")
            except InformationSetCompilationStopped as exc:
                raise InformationSetCompilationStopped(exc.reason, stage="complete_state_quality",
                    evidence={"nativeBidDominance": attempt}) from exc
            prove_args.update(maximum_seconds=fallback_seconds,
                maximum_queries=queries - attempt["queriesUsed"])
            if (attempt.get("kind") == "native_bid_selected_action_dominance_v3"
                    and isinstance(attempt.get("alternatives"), list)
                    and attempt["alternatives"]
                    and [row.get("action") for row in attempt["alternatives"] if isinstance(row, Mapping)]
                        == sorted(row["code"] for row in normalized.legal_actions
                            if row["code"] != prove_args["selected_root_action"])):
                # Reuse complete owned caps; the compiler validates every premise.
                prove_args["_alternative_basis"] = attempt
            certificate = compiler._prove_native_bid_selected_action(
                normalized, prefix, Seat.parse(normalized.next_actor).team, **prove_args,
                **({"opponent_play_policy_id": interval_args["opponent_play_policy_id"]}
                   if "opponent_play_policy_id" in interval_args else {}),
                progress_callback=(lambda row: progress_callback({**row, "decisionKind": "bid",
                    "proofPhase": "fallback"})) if progress_callback is not None else None)
            certificate = {**certificate, "incumbentAttempt": attempt,
                "limits": {**certificate.get("limits", {}), "sharedDecisionMaximumSeconds": decision_seconds}}
            certificate["sha256"] = canonical_sha256({key: value for key, value in certificate.items() if key != "sha256"})
        if certificate["status"] != "selected_action_optimal":
            raise InformationSetCompilationStopped("native_bid_quality_incomplete", stage="complete_state_quality",
                evidence={"nativeBidDominance": certificate})
        if request.selected_action is None:
            certificate = {**certificate, "priorSelectionAttempts": list(_prior_selection_attempts or ())}
            if _staged_bid_selection:
                certificate["selectionSchedule"] = "incumbent_then_deep_v1"
            certificate["sha256"] = canonical_sha256({key: value for key, value in certificate.items() if key != "sha256"})
        validate_native_bid_dominance_budget_v1(certificate, budget,
            maximum_conflicts=maximum_conflicts, opportunity_order=opportunity_order, execution_budget=execution_budget)
        return build_native_bid_dominance_quality_artifact_v1(certificate, rules=rules,
            source_prefix=prefix, normalized=normalized, request=request,
            **({"native_incumbent_models": tuple(native_incumbent_models)} if native_incumbent_models else {}))
    if (normalized.decision_kind == "escalation" and request.mode == "exact"
            and request.visibility == "complete_state" and request.utility == "awarded_point_margin"
            and request.line_universe is None and escalation_events is not None):
        from .result_authority import build_native_constant_escalation_quality_artifact_v1
        if (not isinstance(budget, ScenarioProofBudgetV1) or budget.native_observed is None
                or type(maximum_conflicts) is not int or maximum_conflicts < 1):
            raise ValueError("native escalation quality requires frozen proof and conflict budgets")
        native = budget.native_observed
        certificate = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)._solve_native_post_auction_policy_bounds(
            normalized, prefix, Seat.parse(normalized.next_actor).team,
            selected_root_action=request.selected_action, opportunity_order=opportunity_order,
            escalation_events=escalation_events, maximum_iterations=native.maximum_iterations,
            maximum_queries=native.maximum_queries, maximum_observations=native.maximum_observations,
            maximum_model_bytes=budget.maximum_cache_bytes, maximum_conflicts=maximum_conflicts,
            maximum_depth=min(maximum_depth, budget.maximum_depth, budget.maximum_horizon_actions),
            maximum_seconds=min(remaining(), budget.maximum_wall_seconds, budget.maximum_cpu_seconds),
            cancel_check=cancel_check)
        if certificate["status"] != "scoped_constant_bound_converged":
            raise InformationSetCompilationStopped("native_escalation_quality_incomplete", stage="complete_state_quality",
                evidence={"nativeEscalationBounds": certificate})
        artifact = build_native_constant_escalation_quality_artifact_v1(certificate, rules=rules,
            source_prefix=prefix, normalized=normalized, request=request, budget=budget,
            opportunity_order=opportunity_order, escalation_events=escalation_events,
            maximum_seconds=remaining(), maximum_conflicts=maximum_conflicts, cancel_check=cancel_check)
        allocation = artifact["allocationModel"]
        if (allocation["priorNonAnnouncingSeats"] != list(prior_non_announcing_seats)
                or allocation["priorWindowRecall"] != {seat: list(windows)
                    for seat, windows in (prior_window_recall or {}).items() if windows}):
            raise ValueError("native escalation quality differs from actual private history")
        remaining()
        return artifact
    models = []
    pending_projects = (prefix.project_actions_present and len(prefix.project_actions)
        < len(rules.project_domain.declaration_seats(Seat.parse(prefix.dealer))))
    crosses_project_boundary = pending_projects and not (
        request.mode == "bounded" and request.horizon is not None
        and request.horizon["unit"] in {"plies", "decisions"} and request.horizon["value"] == 1)
    if (normalized.decision_kind == "play" and prefix.project_actions_present
            and request.mode == "exact" and request.visibility == "complete_state"
            and request.line_universe is None and isinstance(budget, ScenarioProofBudgetV1)
            and budget.native_observed is not None):
        from .result_authority import build_native_complete_state_play_quality_artifact_v1
        if 32 - len(prefix.play_actions) + (len(Seat) - len(prefix.project_actions) if pending_projects else 0) > maximum_depth:
            raise InformationSetCompilationStopped("maximum_depth", stage="complete_state_quality")
        compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
        roots = compiler._prove_complete_state_play(normalized, prefix, request, budget=budget,
            maximum_seconds=remaining(), cancel_check=cancel_check, maximum_nodes=maximum_nodes)
        artifact = build_native_complete_state_play_quality_artifact_v1(roots, rules=rules,
            source_prefix=prefix, normalized=normalized, request=request, budget=budget)
        remaining()
        return artifact
    if normalized.decision_kind != "play" or crosses_project_boundary:
        try:
            compiled = compile_nonplay_complete_graph_v1(
                rules, normalized, prefix, decision, request,
                maximum_nodes=maximum_nodes, maximum_depth=maximum_depth,
                maximum_seconds=remaining(), native_cp_sat=True, cancel_check=cancel_check,
                **({"native_model_receipts": models} if normalized.decision_kind == "bid" else {}),
                **({"opportunity_order": opportunity_order} if opportunity_order is not None
                   and normalized.decision_kind in {"bid", "escalation"} else {}),
                prior_non_announcing_seats=prior_non_announcing_seats,
                prior_window_recall=prior_window_recall,
            )
        except NonPlayGraphCompilationStoppedV1 as exc:
            raise InformationSetCompilationStopped(exc.reason, stage="complete_state_quality",
                evidence={"nodes": exc.nodes, "depth": exc.depth,
                          "requestHash": request.sha256, "sourcePrefixHash": prefix.sha256,
                          **({"projectDominanceAttempt": project_dominance_attempt} if project_dominance_attempt is not None else {}),
                          "nativeAuctionModels": [encode_native_auction_model_receipt_v1(item) for item in models]}) from exc
        graph = compiled.graph
    else:
        compiled = OrToolsCompleteStateGraphCompilerV1(
            play_domain=rules.play_domain, scoring_domain=ScoringDomainIRV1.from_rules(rules.data),
            structural_domain=rules.structural_domain, project_domain=rules.project_domain,
            native_conversion=True,
        ).compile(normalized, request, expected_rules_hash=rules.sha256,
            maximum_nodes=maximum_nodes, maximum_depth=maximum_depth,
            maximum_seconds=remaining(), cancelled=cancel_check)
        if not compiled.complete or compiled.graph is None:
            raise InformationSetCompilationStopped(
                compiled.receipt.exhaustion_reason or "incomplete_graph", stage="complete_state_quality")
        graph = compiled.graph
    proof = OrToolsScenarioGraphProofV1().prove(
        graph, request, maximum_seconds=remaining(), cancel_check=cancel_check,
        native_action_quality=True,
    )
    allocation = {"schemaVersion": 1, "kind": "native_complete_state_quality_v1",
        "sourcePrefixHash": prefix.sha256, "exactStateId": normalized.exact_state_id,
        "rulesHash": rules.sha256,
        **({"projectDominanceAttempt": project_dominance_attempt} if project_dominance_attempt is not None else {}),
        **({"opportunityOrder": list(opportunity_order),
            "priorNonAnnouncingSeats": list(prior_non_announcing_seats),
            "priorWindowRecall": {seat: list(windows) for seat, windows in (prior_window_recall or {}).items()}}
           if normalized.decision_kind == "escalation" and opportunity_order is not None else {}),
        **({"nativeAuctionModels": [{"sourcePrefixHash": prefix.sha256,
            "models": [encode_native_auction_model_receipt_v1(item) for item in models]}]}
           if models else {})}
    return {"request": request.to_dict(), "proof": proof.to_dict(),
        "proofModel": proof.model_document(), "graph": graph.to_dict(),
        "allocationModel": {**allocation, "sha256": canonical_sha256(allocation)}}


def _curated_primary_interest_failures_v1(artifact, *, compilation, proof, normalized, request, rules, policy):
    """Reject only owned live primary misses, never admit or bless their evidence."""
    from .decision_selection import actor_interest_primary_failures_v1
    from .result_authority import native_featured_play_proof_supported_v1

    if (not isinstance(compilation, ActorInformationSetCompilationV1)
            or not isinstance(proof, OrToolsGraphProofOutcomeV1)
            or not isinstance(normalized, OrToolsNormalizedExactStateV1)
            or not native_featured_play_proof_supported_v1(artifact)):
        return []
    graph, allocation = compilation.graph, compilation.graph.model_document
    binding = allocation.get("observerScopeBinding") or {}
    roots = artifact["proof"]["rootActions"]
    if (not isinstance(binding, Mapping) or not isinstance(artifact.get("graph"), Mapping)
            or any(not isinstance(row, Mapping) for row in roots)):
        return []
    actions = [row.get("action") for row in roots]
    if (proof.solver_status != "optimal" or proof.request_hash != request.sha256
            or artifact["request"] != request.to_dict() or artifact["proof"] != proof.to_dict()
            or artifact["graph"] != graph.to_dict() or proof.graph_hash != artifact["graph"]["sha256"]
            or artifact["allocationModel"] != dict(allocation)
            or proof.model_hash != canonical_sha256(artifact["proofModel"])
            or artifact["proofModel"].get("modelStatus") != "valid"
            or graph.root_actor != normalized.next_actor or graph.root_actor_team != request.target_team
            or graph.root_actor_relation != "target" or allocation.get("exactStateId") != normalized.exact_state_id
            or binding.get("exactStateId") != normalized.exact_state_id
            or binding.get("sourceStateHash") != normalized.state_hash or binding.get("rulesHash") != rules.sha256
            or any(not isinstance(action, str) for action in actions)
            or len(set(actions)) != len(actions)
            or set(actions) != {row["code"] for row in normalized.legal_actions}
            or any(type(row.get("value")) is not int for row in roots)):
        return []
    normalized.require_integrity()
    failures = actor_interest_primary_failures_v1(proof.selected_action, artifact, policy)
    definite = {"interestingness_next_best_regret_insufficient", "interestingness_consistent_worlds_insufficient"}
    return failures if failures and set(failures) <= definite else []


def _resolve_quality_continuation_v1(
    rules: RuleBook,
    prefix: ExactTracePrefixV1,
    quality_request: ScenarioProofRequestV1 | ScenarioPlayingQualityPolicyV1,
    *,
    maximum_nodes: int,
    maximum_depth: int,
    maximum_worlds: int,
    maximum_seconds: float,
    maximum_decision_seconds: float,
    maximum_evidence_bytes: int,
    cancel_check: Callable[[], bool] | None = None,
    recorded_actions: Sequence[Mapping[str, Any]] | None = None,
    recorded_trace: ExactTracePrefixV1 | None = None,
    proof_budget_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    search_budget_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    runtime_maximum_seconds: float | None = None,
    private_recall: Mapping[str, Sequence[str]] | None = None,
    auction_only: bool = False,
    escalation_events: Sequence[Mapping[str, Any]] | None = None,
    continue_recorded_prefix: bool = False,
    stop_on_recorded_quality_failure: bool = False,
    checkpoint_callback: Callable[[Mapping[str, Any]], None] | None = None,
    final_normalized_callback: Callable[[OrToolsNormalizedExactStateV1], None] | None = None,
    execution_budget: ScenarioQualityExecutionBudgetV1 | None = None,
    opportunity_order: Sequence[str] | None = None,
    featured_play_index: int | None = None,
    featured_decision_point: Mapping[str, Any] | None = None,
    featured_selector_dynamic: bool = False,
    paired_policy_proofs: Mapping[str, Any] | None = None,
    _verified_artifacts: set[str] | None = None,
    _verified_normalizations: dict[str, OrToolsNormalizedExactStateV1] | None = None,
    search_generated: bool = False,
    graph_workers: int = 1,
    _curated_focal_only: bool = False,
    _admissible_play_actions: Callable[..., tuple[str, ...] | None] | None = None,
) -> dict[str, Any]:
    """Generate or audit choices within the request's declared information scope.

    The authored prefix is immutable. This certifies only the requested suffix,
    not its preceding decisions, and cannot qualify a round
    on its own. Each actor maximizes its own team's declared utility.
    Resolved budget snapshots are supplied by the caller's frozen bundle;
    this low-level helper never loads a process-active or default profile.
    """
    if not isinstance(rules, RuleBook) or not isinstance(prefix, ExactTracePrefixV1):
        raise TypeError("quality continuation requires rules and an exact trace prefix")
    validate_content_interest_rules_v1(rules, quality_request, execution_budget)
    if type(graph_workers) is not int or not 1 <= graph_workers <= 4:
        raise ValueError("graph_workers must be an integer within the granted 1..4 worker ceiling")
    if type(continue_recorded_prefix) is not bool:
        raise TypeError("continue_recorded_prefix must be boolean")
    if type(stop_on_recorded_quality_failure) is not bool:
        raise TypeError("stop_on_recorded_quality_failure must be boolean")
    if type(featured_selector_dynamic) is not bool:
        raise TypeError("featured_selector_dynamic must be boolean")
    if type(search_generated) is not bool:
        raise TypeError("search_generated must be boolean")
    if type(_curated_focal_only) is not bool:
        raise TypeError("_curated_focal_only must be boolean")
    if featured_play_index is not None and (
        auction_only or type(featured_play_index) is not int
        or not 0 <= featured_play_index < 32
        or not isinstance(featured_decision_point, Mapping)
        or featured_decision_point.get("kind") != "play"
    ):
        raise ValueError("featured native play requires one valid play decision")
    if continue_recorded_prefix and (recorded_trace is None or recorded_actions is not None):
        raise ValueError("prefix continuation requires a recorded trace")
    if auction_only and (prefix.escalation_actions or prefix.project_actions
            or prefix.project_declarations or prefix.play_actions or private_recall):
        raise ValueError("auction quality must start before escalation, projects and play")
    if auction_only and recorded_actions is not None:
        raise ValueError("recorded auction quality requires a trace, not a play suffix")
    quality_policy = None
    recall = normalize_escalation_recall_v1(private_recall)
    if escalation_events is not None:
        if (auction_only or not isinstance(escalation_events, (list, tuple))
                or any(not isinstance(row, Mapping) or set(row) != {"actor", "actionCode"}
                    or row["actor"] not in {seat.name.lower() for seat in Seat}
                    or type(row["actionCode"]) is not str or not row["actionCode"] for row in escalation_events)):
            raise ValueError("play quality escalation history requires canonical actor/actionCode events")
        escalation_events = copy.deepcopy(list(escalation_events))
    if isinstance(quality_request, ScenarioPlayingQualityPolicyV1):
        quality_policy = ScenarioPlayingQualityPolicyV1.build(quality_request.to_dict())
        quality_request = quality_policy.request_for("bid" if auction_only else "play", "north")
    if (
        not isinstance(quality_request, ScenarioProofRequestV1)
        or quality_request.mode not in {"exact", "bounded"}
        or quality_request.visibility not in {"actor_information_set", "complete_state"}
        or quality_request.claim != "winning_root_action_count"
        or quality_request.selected_action is not None
    ):
        raise ValueError("quality continuation requires declared root-value analysis")
    if len(prefix.play_actions) >= 32:
        raise ValueError("quality continuation requires an unfinished play prefix")
    if paired_policy_proofs is not None and (auction_only or recorded_actions is not None
            or recorded_trace is not None
            or quality_request.mode != "exact" or quality_request.visibility != "complete_state"
            or featured_play_index is not None):
        raise ValueError("paired quality requires an exact generated terminal play suffix without a focal count claim")
    includes_project_decisions = not auction_only and prefix.project_actions_present and len(prefix.play_actions) < 4
    if includes_project_decisions and quality_policy is None:
        raise ValueError("interleaved quality continuation requires a per-decision policy")
    if includes_project_decisions and recorded_actions is not None:
        raise ValueError("recorded play-only audit cannot supply project decisions")
    if recorded_trace is not None and recorded_actions is not None:
        raise ValueError("supply one recorded trace or play suffix, not both")
    if any(type(value) is not int or value < 1 for value in (
        maximum_nodes, maximum_depth, maximum_worlds, maximum_evidence_bytes,
    )):
        raise ValueError("quality continuation limits must be positive integers")
    if any(type(value) not in {int, float} or not math.isfinite(value) or value <= 0
           for value in (maximum_seconds, maximum_decision_seconds)):
        raise ValueError("quality continuation time limits must be finite and positive")
    if cancel_check is not None and not callable(cancel_check):
        raise TypeError("quality continuation cancellation check must be callable")
    if _curated_focal_only:
        if (auction_only or quality_policy is None
                or not isinstance(execution_budget, ScenarioQualityExecutionBudgetV1)
                or not execution_budget.exact_tail_proof or execution_budget.exact_play_from_trick is None
                or (execution_budget.video_interestingness or {}).get("schemaVersion") != 4
                or len(prefix.play_actions) < 4 * (execution_budget.exact_play_from_trick - 1)
                or recorded_trace is not None or recorded_actions is not None):
            raise ValueError("curated focal precheck requires an immutable exact-tail play prefix")
        focal_request = quality_policy.request_for("play", "north",
            play_index=len(prefix.play_actions), execution_budget=execution_budget)
        if (focal_request.mode != "exact" or focal_request.visibility != "actor_information_set"
                or focal_request.strategy_class != "pure_deterministic"
                or focal_request.utility != "awarded_point_margin"):
            raise ValueError("curated focal precheck requires exact actor-information-set margin proof")
    if execution_budget is not None:
        execution_budget = ScenarioQualityExecutionBudgetV1.build(execution_budget.to_dict())
    runtime_seconds = maximum_seconds if runtime_maximum_seconds is None else runtime_maximum_seconds
    if (type(runtime_seconds) not in {int, float} or not math.isfinite(runtime_seconds)
            or not 0 < runtime_seconds <= maximum_seconds):
        raise ValueError("runtime allowance must be positive and cannot exceed the declared budget")
    recorded_prefix = None
    if recorded_trace is not None:
        recorded_prefix = _validate_quality_recorded_trace(prefix, recorded_trace, auction_only=auction_only,
            allow_partial=continue_recorded_prefix)
    if recorded_actions is not None:
        recorded_document = prefix.to_dict()
        recorded_document.pop("sha256")
        recorded_document["stateAssertions"] = {}
        recorded_document["playActions"] = [*prefix.play_actions, *recorded_actions]
        recorded_prefix = ExactTracePrefixV1.build(recorded_document)
    if not continue_recorded_prefix and not auction_only and recorded_prefix is not None and len(recorded_prefix.play_actions) != 32:
        raise ValueError("recorded quality audit requires a complete 32-play line")

    requested_profiles = ({row["budgetProfile"] for row in quality_policy.decisions.values()}
                          if quality_policy is not None else {quality_request.budget_profile})
    budgets, budget_snapshots = _quality_budget_snapshots(requested_profiles, proof_budget_profiles,
        execution_budget=execution_budget)

    started = time.monotonic()
    deadline = started + runtime_seconds
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    def normalize_current(value, **kwargs):
        if auction_only:
            return normalizer.normalize(value, decision_point={"kind": "bid"}, **kwargs)
        return normalizer.normalize_play_or_project(value, **kwargs)
    current = prefix
    normalized = None
    decisions: list[dict[str, Any]] = []
    decision_timings: list[dict[str, Any]] = []
    artifacts: dict[str, dict[str, Any]] = {}
    paired_evidence = {}
    pair_synthesis = None
    pair_attempted = paired_policy_proofs is not None
    evidence_bytes = 0
    initial_normalization_seconds = 0.0
    raw_context = None
    raw_evidence = {}
    round_win_evidence = {}
    baseline_evidence = {}
    tie_rule_evidence = {}
    project_inference_evidence = {}
    raw_timing = None
    checkpoint_callback_seconds = 0.0
    partial: dict[str, Any] | None = None
    stage = "normalization"
    cpu_deadline = math.inf
    worker_cpu_seconds = 0.0

    def add_worker_cpu(seconds: float) -> None:
        nonlocal worker_cpu_seconds
        worker_cpu_seconds += seconds

    def cpu_time() -> float:
        return time.process_time() + worker_cpu_seconds

    def stopped() -> bool:
        return bool(cancel_check is not None and cancel_check()) or cpu_time() >= cpu_deadline

    def remaining(limit: float) -> float:
        if cancel_check is not None and cancel_check():
            raise InformationSetCompilationStopped("cancelled", stage=stage)
        if cpu_deadline != math.inf and cpu_time() >= cpu_deadline:
            raise InformationSetCompilationStopped("maximum_cpu_seconds", stage=stage)
        if evidence_bytes >= maximum_evidence_bytes:
            raise InformationSetCompilationStopped("maximum_evidence_bytes", stage=stage)
        value = min(deadline, limit) - time.monotonic()
        if value <= 0:
            raise InformationSetCompilationStopped("maximum_seconds", stage=stage)
        return value

    status = "incomplete"
    featured_matches = 0
    try:
        normalization_started = time.monotonic()
        normalization_seconds = maximum_decision_seconds
        if execution_budget is not None and execution_budget.exact_tail_budget_profile is not None:
            initial_request = (quality_policy.request_for("bid" if auction_only else "play", "north",
                play_index=None if auction_only else len(current.play_actions), execution_budget=execution_budget)
                if quality_policy is not None else quality_request)
            normalization_seconds = min(normalization_seconds, budgets[initial_request.budget_profile].maximum_wall_seconds)
        normalized = normalize_current(
            current,
            maximum_seconds=remaining(time.monotonic() + normalization_seconds),
            cancel_check=cancel_check,
        )
        initial_normalization_seconds = time.monotonic() - normalization_started
        while (bool(normalized.legal_actions) if auction_only else len(current.play_actions) < 32):
            decision_deadline = time.monotonic() + maximum_decision_seconds
            remaining(decision_deadline)
            if normalized.decision_kind not in ({"bid"} if auction_only else {"project", "play"}) or normalized.next_actor is None:
                raise ValueError("quality continuation requires a legal decision in its declared stage")
            actor = Seat.parse(normalized.next_actor)
            decision: dict[str, Any] = {"kind": normalized.decision_kind}
            if normalized.decision_kind == "project":
                decision["selector"] = {"owner": actor.name.lower()}
            request_value = (
                quality_policy.request_for(normalized.decision_kind, actor.name.lower(),
                    play_index=len(current.play_actions) if normalized.decision_kind == "play" else None,
                    execution_budget=execution_budget).to_dict()
                if quality_policy is not None else quality_request.to_dict()
            )
            request_value.pop("sha256")
            request_value["targetTeam"] = actor.team.name.lower()
            auditing = recorded_prefix is not None and (not continue_recorded_prefix or
                (len(current.auction_actions) < len(recorded_prefix.auction_actions) if auction_only else
                 len(current.play_actions) < len(recorded_prefix.play_actions) or
                 len(current.project_actions) < len(recorded_prefix.project_actions)))
            if continue_recorded_prefix and not auditing and current.sha256 != recorded_prefix.sha256 and not decisions:
                raise ValueError("recorded prefix boundary differs from source state")
            if auditing:
                if auction_only:
                    if len(current.auction_actions) >= len(recorded_prefix.auction_actions):
                        raise ValueError("recorded quality trace is missing a bid action")
                    recorded = recorded_prefix.auction_actions[len(current.auction_actions)]
                    matches = [row["code"] for row in normalized.legal_actions
                               if row["action"] == recorded["action"]
                               and (recorded.get("trumpSuit") is None
                                    or row.get("trumpSuit") == recorded["trumpSuit"])]
                    if len(matches) != 1:
                        raise ValueError("recorded quality bid action is not uniquely legal")
                    recorded_code = matches[0]
                elif normalized.decision_kind == "project":
                    if len(current.project_actions) >= len(recorded_prefix.project_actions):
                        raise ValueError("recorded quality trace is missing a project action")
                    recorded = recorded_prefix.project_actions[len(current.project_actions)]
                    matches = [row["code"] for row in normalized.legal_actions
                               if row.get("action") == recorded["action"]
                               and row.get("projectId") == recorded.get("projectId")]
                    if len(matches) != 1:
                        raise ValueError("recorded quality project action is not legal")
                    recorded_code = matches[0]
                else:
                    if len(current.play_actions) >= len(recorded_prefix.play_actions):
                        raise ValueError("recorded quality trace is missing a play action")
                    recorded = recorded_prefix.play_actions[len(current.play_actions)]
                    recorded_code = LegalPlayActionV1(
                        Card.parse(recorded["card"]), recorded.get("ekka", False),
                    ).code
                if recorded.get("actor", actor.name.lower()) != actor.name.lower():
                    raise ValueError("recorded quality action has the wrong actor")
                if recorded_code not in {row["code"] for row in normalized.legal_actions}:
                    raise ValueError("recorded quality action is not legal")
                request_value["selectedAction"] = recorded_code
            request = ScenarioProofRequestV1.build(request_value)
            budget = budgets.get(request.budget_profile)
            nodes, depth, worlds = maximum_nodes, maximum_depth, maximum_worlds
            if execution_budget is not None:
                worlds = execution_budget.resolve_hidden_world_cap(worlds,
                    play_index=len(current.play_actions) if normalized.decision_kind == "play" else None)
            if budget is not None:
                nodes = min(nodes, budget.maximum_nodes, budget.maximum_state_space)
                depth = min(depth, budget.maximum_depth)
                worlds = min(worlds, budget.maximum_hidden_worlds)
                decision_deadline = min(decision_deadline, time.monotonic() + budget.maximum_wall_seconds)
                cpu_deadline = cpu_time() + budget.maximum_cpu_seconds
                if request.horizon is not None and request.horizon["unit"] != "round_end":
                    actions = request.horizon["value"] * (len(Seat) if request.horizon["unit"] == "tricks" else 1)
                    if actions > budget.maximum_horizon_actions:
                        raise InformationSetCompilationStopped("maximum_horizon_actions", stage="action_analysis")
                    if actions > depth:
                        raise InformationSetCompilationStopped("maximum_depth", stage="action_analysis")
                elif request.mode == "bounded":
                    depth = min(depth, budget.maximum_horizon_actions)
            if nodes < 2:
                raise InformationSetCompilationStopped("maximum_nodes", stage="action_analysis")
            if normalized.decision_kind == "bid" and request.boundary_evaluator == "auction_hand_margin_v1":
                worlds = 1
            proof_cancel_check = stopped if budget is not None else cancel_check
            stage = "action_analysis"
            declared_decision_seconds = min(maximum_decision_seconds, budget.maximum_wall_seconds) if budget is not None else maximum_decision_seconds
            timing = {
                "decisionIndex": len(decisions),
                "decisionKind": normalized.decision_kind,
                "compilationSeconds": 0.0,
                "proofSeconds": 0.0,
            }
            decision_timings.append(timing)
            stage_started = time.monotonic()
            proof = None
            native_execution = None
            policy_artifact = None
            root_witnesses = []
            current_trick = normalized.state.get("currentTrick")
            selector = dict(featured_decision_point.get("selector") or {}) if featured_play_index is not None else {}
            trick_index = (
                current_trick.get("index")
                if isinstance(current_trick, Mapping)
                else len(normalized.state.get("completedTricks") or ())
            )
            position = (
                len(current_trick.get("plays") or ())
                if isinstance(current_trick, Mapping) else 0
            )
            hand = dict(normalized.state.get("hands") or {}).get(actor.name.lower()) or ()
            featured_selector_matches = not any((
                selector.get("actingSeat") not in {None, actor.name.lower()},
                selector.get("trickIndex") not in {None, trick_index},
                selector.get("position") not in {None, position},
                selector.get("remainingCards") not in {None, len(hand)},
            ))
            featured_now = bool(
                normalized.decision_kind == "play"
                and featured_play_index is not None
                and (
                    featured_selector_matches and featured_matches == 0
                    if featured_selector_dynamic
                    else featured_play_index == len(current.play_actions)
                )
            )
            if featured_now:
                if not featured_selector_matches:
                    raise InformationSetCompilationStopped(
                        "featured_decision_mismatch",
                        stage="action_analysis",
                        evidence={
                            "actualActor": actor.name.lower(),
                            "actualTrickIndex": trick_index,
                            "actualPosition": position,
                            "actualRemainingCards": len(hand),
                        },
                    )
                featured_matches += 1
            if request.visibility == "complete_state":
                if featured_now:
                    # The focal root-count claim needs its independent roots;
                    # preceding action-quality certificates remain usable.
                    paired_policy_proofs = None
                    if auditing and not continue_recorded_prefix and request.mode == "exact" and request.horizon is None:
                        # Audit descendants independently instead of restarting pair synthesis after a full focal proof.
                        pair_attempted = True
                if (paired_policy_proofs is None and not pair_attempted and execution_budget is not None
                        and execution_budget.complete_state_play_policy_pair
                        and (featured_play_index is None or not featured_now) and not auction_only
                        and request.mode == "exact" and request.utility == "awarded_point_margin"
                        and normalized.decision_kind == "play" and current.project_actions_present):
                    if budget is None:
                        raise ValueError("paired synthesis requires a frozen proof budget")
                    if 32 - len(current.play_actions) + len(Seat) - len(current.project_actions) > depth:
                        raise InformationSetCompilationStopped("maximum_depth", stage="action_analysis")
                    try:
                        pair_attempted = auditing
                        pair_request = ScenarioProofRequestV1.build({key: value for key, value
                            in request.to_dict().items() if key not in {"sha256", "selectedAction"}})
                        incumbent = execution_budget.bid_incumbent_proof
                        pair_synthesis = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)._synthesize_complete_state_policy_pair(
                            normalized, current, pair_request, budget=budget,
                            maximum_seconds=remaining(decision_deadline), maximum_nodes=nodes,
                            maximum_evidence_bytes=maximum_evidence_bytes - evidence_bytes,
                            cancel_check=proof_cancel_check,
                            **({"play_policies": incumbent["playPolicies"]}
                               if incumbent is not None and incumbent["schemaVersion"] == 1 else
                               {"native_play_policy_ids": {
                                   request.target_team: incumbent["nativePlayPolicyId"],
                                   Team[request.target_team.upper()].other.name.lower():
                                       execution_budget.bid_opponent_play_policy_id or incumbent["nativePlayPolicyId"]}}
                               if incumbent is not None and incumbent["schemaVersion"] == 2 else {}))
                    except InformationSetCompilationStopped as exc:
                        if exc.reason != "paired_project_branch_choices":
                            raise
                        timing["pairedPreflight"] = exc.partial_receipt
                    else:
                        pair_attempted = True
                        paired_policy_proofs = pair_synthesis["proofs"]
                if paired_policy_proofs is not None and normalized.decision_kind == "project":
                    _, policy_artifact = _forced_quality_artifact_v1(normalized, request,
                        actual_action=request.selected_action, maximum_seconds=remaining(decision_deadline),
                        cancel_check=proof_cancel_check, source_prefix=current)
                elif paired_policy_proofs is not None:
                    from .result_authority import (verify_native_complete_state_policy_pair_v1,
                        build_native_paired_play_action_artifact_v1)
                    if budget is None:
                        raise ValueError("paired quality requires a frozen proof budget")
                    if not paired_evidence:
                        pair_request = ScenarioProofRequestV1.build({key: value for key, value
                            in request.to_dict().items() if key not in {"sha256", "selectedAction"}})
                        if 32 - len(current.play_actions) + len(Seat) - len(current.project_actions) > depth:
                            raise InformationSetCompilationStopped("maximum_depth", stage="action_analysis")
                        if any(len(proof.get("domains", {})) > nodes for proof in paired_policy_proofs.values()):
                            raise InformationSetCompilationStopped("maximum_nodes", stage="action_analysis")
                        query_allowance = budget.native_observed.maximum_queries - (
                            pair_synthesis["queriesUsed"] if pair_synthesis is not None else 0)
                        if query_allowance <= 0:
                            raise InformationSetCompilationStopped("native_pair_budget_exhausted", stage="action_analysis",
                                evidence={"synthesis": pair_synthesis})
                        pair = verify_native_complete_state_policy_pair_v1(paired_policy_proofs,
                            rules=rules, normalized=normalized, source_prefix=current,
                            request=pair_request, budget=budget,
                            maximum_queries=query_allowance, maximum_nodes=nodes,
                            cancel_check=lambda: bool(proof_cancel_check and proof_cancel_check())
                                or time.monotonic() >= min(deadline, decision_deadline))
                        evidence = {"receipt": pair, "proofs": copy.deepcopy(dict(paired_policy_proofs)),
                            "sourcePrefix": current.to_dict(), "request": pair_request.to_dict()}
                        if pair_synthesis is not None:
                            evidence["synthesis"] = pair_synthesis
                        pair_bytes = len(json.dumps(evidence, ensure_ascii=False,
                            sort_keys=True, separators=(",", ":")).encode("utf-8"))
                        if evidence_bytes + pair_bytes > maximum_evidence_bytes:
                            raise InformationSetCompilationStopped("maximum_evidence_bytes", stage=stage,
                                evidence={"retainedBytes": evidence_bytes, "candidateBytes": pair_bytes,
                                    "maximumBytes": maximum_evidence_bytes})
                        pair_hash = canonical_sha256(evidence)
                        paired_evidence[pair_hash] = evidence
                        if _verified_artifacts is not None:
                            _verified_artifacts.add(canonical_sha256({
                                "kind": "verified_native_complete_state_policy_pair_v1",
                                "evidenceHash": pair_hash}))
                        evidence_bytes += pair_bytes
                    pair_hash, evidence = next(iter(paired_evidence.items()))
                    if budget.sha256 != evidence["receipt"]["proofBudgetHash"]:
                        raise ValueError("paired quality cannot switch proof budgets mid-continuation")
                    paired_decision = next(row for row in evidence["receipt"]["continuation"]["decisions"]
                        if row["playIndex"] == len(current.play_actions))
                    paired_action = LegalPlayActionV1(Card.parse(paired_decision["action"]["card"]),
                        paired_decision["action"].get("ekka", False)).code
                    if auditing and recorded_code != paired_action:
                        # A different legal move may also be optimal. Prove it
                        # independently; never replace or reject it by mismatch.
                        paired_policy_proofs = None
                        if not any(item["proofModel"].get("pairEvidenceHash") == pair_hash
                                   for item in artifacts.values()):
                            evidence_bytes -= pair_bytes
                            paired_evidence.clear()
                    else:
                        policy_artifact = build_native_paired_play_action_artifact_v1(evidence["receipt"],
                            pair_hash=pair_hash, source_prefix=current, normalized=normalized, request=request)
                if policy_artifact is None:
                    policy_artifact = _complete_state_quality_artifact_v1(
                        rules, normalized, current, decision, request,
                        maximum_nodes=nodes, maximum_depth=depth,
                        maximum_seconds=remaining(decision_deadline), cancel_check=proof_cancel_check,
                        opportunity_order=opportunity_order, budget=budget,
                        execution_budget=execution_budget, progress_callback=checkpoint_callback,
                        _search_generated=search_generated,
                        **({"maximum_conflicts": search_budget_profiles[request.budget_profile]["maximumConflicts"]}
                           if search_budget_profiles is not None and (normalized.decision_kind == "project"
                               or normalized.decision_kind == "bid") else {}),
                    )
                timing["compilationAndProofSeconds"] = time.monotonic() - stage_started
            elif (normalized.decision_kind == "bid" and request.mode == "exact"
                    and request.utility == "awarded_point_margin" and request.line_universe is None
                    and len(normalized.legal_actions) > 1
                    and budget is not None and budget.native_observed is not None):
                if search_budget_profiles is None or request.budget_profile not in search_budget_profiles:
                    raise ValueError("observed bid quality requires frozen search limits")
                select_bid = _observed_bid_quality_artifact_v1 if auditing else _select_observed_bid_quality_artifact_v1
                policy_artifact = select_bid(rules, normalized, current, request,
                    budget=budget, execution_budget=execution_budget,
                    maximum_conflicts=search_budget_profiles[request.budget_profile]["maximumConflicts"],
                    maximum_depth=depth, maximum_seconds=remaining(decision_deadline), cancel_check=proof_cancel_check,
                    progress_callback=checkpoint_callback,
                    **({"_search_generated": search_generated} if not auditing else {}))
                timing["nativeExecutionSeconds"] = time.monotonic() - stage_started
            elif normalized.decision_kind == "play" and not (
                    execution_budget is not None and execution_budget.exact_tail_proof
                    and request.mode == "exact" and request.visibility == "actor_information_set"
                    and len(normalized.legal_actions) > 1):
                preferred_completed_trick_winner = None
                if (
                    not auditing
                    and featured_play_index is not None
                    and isinstance(current_trick, Mapping)
                    and len(current_trick.get("plays") or ()) == 3
                    and selector.get("trickIndex") == int(current_trick["index"]) + 1
                    and type(selector.get("position")) is int
                    and isinstance(selector.get("actingSeat"), str)
                ):
                    preferred_completed_trick_winner = Seat.parse(selector["actingSeat"])
                    for _ in range(selector["position"]):
                        preferred_completed_trick_winner = preferred_completed_trick_winner.previous()
                admissible_actions = (_admissible_play_actions(
                    current, normalized, lambda: remaining(decision_deadline))
                    if _admissible_play_actions is not None and not auditing
                    and request.mode == "bounded" and not featured_now else None)
                policy_artifact, proof, native_execution, timing_kind = _select_quality_play_action_v1(
                    rules, current, normalized, request,
                    quality_policy=quality_policy, execution_budget=execution_budget,
                    budget=budget, featured_now=featured_now,
                    actual_action=recorded_code if auditing else None,
                    **({"admissible_root_actions": admissible_actions}
                       if admissible_actions is not None else {}),
                    preferred_completed_trick_winner=preferred_completed_trick_winner,
                    remaining_seconds=lambda: remaining(decision_deadline),
                    cancel_check=proof_cancel_check, private_recall=recall,
                    maximum_worlds=worlds, maximum_depth=depth,
                    escalation_events=escalation_events, opportunity_order=opportunity_order,
                    maximum_conflicts=(search_budget_profiles[request.budget_profile]["maximumConflicts"]
                        if search_budget_profiles is not None and request.budget_profile in search_budget_profiles else None),
                    progress_callback=checkpoint_callback,
                )
                if timing_kind is not None:
                    timing[timing_kind] = time.monotonic() - stage_started
            elif normalized.decision_kind == "project" and len(normalized.legal_actions) == 1:
                selected, policy_artifact = _forced_quality_artifact_v1(
                    normalized,
                    request,
                    actual_action=recorded_code if auditing else None,
                    maximum_seconds=remaining(decision_deadline),
                    cancel_check=proof_cancel_check,
                )
                timing["forcedSelectionSeconds"] = time.monotonic() - stage_started
            elif (normalized.decision_kind == "project" and request.mode == "exact"
                    and request.visibility == "actor_information_set" and request.utility == "awarded_point_margin"
                    and request.line_universe is None and budget is not None and budget.native_observed is not None
                    and current.project_actions_present and escalation_events is not None and opportunity_order is not None
                    and search_budget_profiles is not None and request.budget_profile in search_budget_profiles):
                policy_artifact = _observed_play_quality_artifact_v1(rules, normalized, current, request,
                    budget=budget, opportunity_order=opportunity_order, escalation_events=escalation_events,
                    maximum_conflicts=search_budget_profiles[request.budget_profile]["maximumConflicts"],
                    maximum_depth=depth, maximum_seconds=remaining(decision_deadline),
                    cancel_check=proof_cancel_check, decision_kind="project",
                    progress_callback=checkpoint_callback)
                timing["nativeExecutionSeconds"] = time.monotonic() - stage_started
            elif (budget is not None and budget.native_observed is not None
                  and normalized.decision_kind == "project"
                  and request.horizon is not None and request.horizon["unit"] == "tricks"):
                proof, native_execution = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)._prove_observed_project(
                    normalized, current, request, budget=budget, maximum_seconds=remaining(decision_deadline),
                    cancel_check=proof_cancel_check, private_recall=recall,
                    maximum_worlds=worlds, maximum_depth=depth)
                timing["nativeExecutionSeconds"] = time.monotonic() - stage_started
            else:
                proof = None
            if native_execution is None and policy_artifact is None:
                compilation = compile_observed_decision_v1(
                    rules, normalized, current, decision, request,
                    maximum_nodes=nodes, native_cp_sat=True,
                    maximum_depth=depth, maximum_worlds=worlds,
                    maximum_seconds=remaining(decision_deadline), cancel_check=proof_cancel_check,
                    declared_maximum_seconds=declared_decision_seconds,
                    opportunity_order=opportunity_order,
                    graph_workers=graph_workers,
                    worker_cpu_callback=add_worker_cpu,
                    **({"private_recall": recall} if private_recall is not None else {}),
                )
                timing["compilationSeconds"] = time.monotonic() - stage_started
                stage_started = time.monotonic()
                proof = prove_actor_information_set_v1(
                    compilation, request, maximum_nodes=nodes,
                    maximum_worlds=worlds, maximum_seconds=remaining(decision_deadline),
                    cancel_check=proof_cancel_check, native_action_quality=True,
                    declared_maximum_seconds=declared_decision_seconds,
                    policy_search=budget.policy_search if budget is not None else None,
                    **({"_root_witnesses": root_witnesses} if normalized.decision_kind == "play" else {}),
                )
                timing["proofSeconds"] = time.monotonic() - stage_started
            proof_document = policy_artifact["proofModel"] if policy_artifact is not None else proof.model_document()
            artifact = policy_artifact if policy_artifact is not None else {
                "request": request.to_dict(), "proof": proof.to_dict(),
                "proofModel": proof_document,
                **({"rootWitnesses": root_witnesses} if root_witnesses else {}),
                "graph": native_execution["rootArtifact"] if native_execution is not None else compilation.graph.to_dict(),
                "allocationModel": ({"reference": "graph.preparations",
                    "sha256": canonical_sha256(native_execution["preparation"])}
                    if native_execution is not None and normalized.decision_kind == "project" else
                    native_execution["preparation"] if native_execution is not None else dict(compilation.graph.model_document)),
                **({"nativeExecution": {key: value for key, value in native_execution.items()
                    if key not in {"rootArtifact", "preparation", "outcome"}},
                    "nativeExecutionBindings": {"rootArtifact": "graph", "preparation": "allocationModel", "outcome": "proof"}}
                    if native_execution is not None else {}),
            }
            if featured_now and request.visibility == "actor_information_set" and len(normalized.legal_actions) > 1:
                from .result_authority import (native_featured_play_proof_supported_v1,
                    reverify_native_exact_observed_play_artifact_v1, _exact_observed_verification_token_v1)
                if not native_featured_play_proof_supported_v1(artifact):
                    coverage = artifact["proof"].get("hiddenWorlds") or {}
                    raise InformationSetCompilationStopped("featured_native_proof_unavailable", stage="action_analysis",
                        evidence={"proofModelKind": artifact["proofModel"].get("kind"),
                            "proofCompletionStatus": artifact["proof"].get("completionStatus"),
                            "proofModelStatus": artifact["proofModel"].get("modelStatus"),
                            "quarantineReason": artifact["proofModel"].get("quarantineReason"),
                            "rootActionCount": len(artifact["proof"].get("rootActions") or ()),
                            "hiddenWorldCount": coverage.get("count"),
                            "hiddenWorldCoverageComplete": coverage.get("complete"),
                            "allocationReachability": artifact["allocationModel"].get("reachability")})
                if request.mode == "exact":
                    reverify_native_exact_observed_play_artifact_v1(artifact, rules=rules, source_prefix=current,
                        normalized=normalized, request=request, maximum_nodes=nodes, maximum_depth=depth,
                        maximum_worlds=worlds, maximum_seconds=remaining(decision_deadline),
                        declared_maximum_seconds=declared_decision_seconds,
                        private_recall=recall if private_recall is not None else None, cancel_check=proof_cancel_check,
                        graph_workers=graph_workers, worker_cpu_callback=add_worker_cpu,
                        policy_search=budget.policy_search if budget is not None else None)
                    if _verified_artifacts is not None:
                        _verified_artifacts.add(_exact_observed_verification_token_v1(artifact, rules=rules,
                            source_prefix=current, normalized=normalized, request=request,
                            maximum_nodes=nodes, maximum_depth=depth, maximum_worlds=worlds,
                            declared_maximum_seconds=declared_decision_seconds,
                            private_recall=recall if private_recall is not None else None,
                            policy_search=budget.policy_search if budget is not None else None))
            artifact_encoding = canonical_json(artifact).encode("utf-8")
            artifact_hash = hashlib.sha256(artifact_encoding).hexdigest()
            artifact_bytes = len(artifact_encoding)
            del artifact_encoding
            if _verified_artifacts is not None and artifact["proofModel"].get("kind") in {
                    "native_observed_play_quality_model_v1", "native_observed_project_quality_model_v1",
                    "native_bounded_play_quality_model_v1",
                    "native_complete_state_root_outcome_v1", "native_bid_selected_action_quality_model_v1"}:
                _verified_artifacts.add(artifact_hash)
            if _verified_artifacts is not None and artifact["proofModel"].get("kind") == "native_observed_bid_quality_model_v1":
                _verified_artifacts.add(canonical_sha256({
                    "artifactHash": artifact_hash, "proofBudgetHash": budget.sha256,
                    "executionBudgetHash": execution_budget.sha256,
                    "maximumConflicts": search_budget_profiles[request.budget_profile]["maximumConflicts"],
                }))
            if evidence_bytes + artifact_bytes > maximum_evidence_bytes:
                raise InformationSetCompilationStopped("maximum_evidence_bytes", stage=stage,
                    evidence={"retainedBytes": evidence_bytes, "candidateBytes": artifact_bytes,
                        "maximumBytes": maximum_evidence_bytes,
                        "sectionBytes": {key: len(json.dumps(value, ensure_ascii=False, sort_keys=True,
                            separators=(",", ":")).encode("utf-8")) for key, value in artifact.items()}})
            artifacts[artifact_hash] = artifact
            if _curated_focal_only and policy_artifact is None and native_execution is None:
                remaining(decision_deadline)
                failures = _curated_primary_interest_failures_v1(artifact, compilation=compilation,
                    proof=proof, normalized=normalized, request=request, rules=rules,
                    policy=execution_budget.video_interestingness)
                if failures:
                    raise _QualityEditorialMiss({"sourcePrefix": current.to_dict(), "primaryArtifact": artifact,
                        "blockers": failures, "researchClaimStatus": "not_evaluated",
                        "verificationStatus": "builder_preview_only"})
            baseline = None
            win_companion = None
            tie_companion = None
            project_companion = None
            if (normalized.decision_kind == "play" and request.mode == "exact"
                    and len(normalized.legal_actions) > 1 and execution_budget is not None
                    and (execution_budget.video_interestingness or {}).get("schemaVersion") == 4
                    and artifact["proof"]["completionStatus"] == "exact_complete"):
                from .result_authority import native_basic_policy_quality_request_v1, native_round_win_quality_request_v1
                baseline_started = time.monotonic()
                baseline_request = native_basic_policy_quality_request_v1(request)
                _, baseline = _observer_safe_play_policy_artifact_v1(rules, normalized, baseline_request,
                    actual_action=None, policy_id=execution_budget.video_interestingness["baselinePolicyId"],
                    maximum_seconds=remaining(decision_deadline), cancel_check=proof_cancel_check)
                baseline = {**baseline, "sourcePrefix": current.to_dict()}
                size = len(canonical_json(baseline).encode("utf-8"))
                if evidence_bytes + artifact_bytes + size > maximum_evidence_bytes:
                    raise InformationSetCompilationStopped("maximum_evidence_bytes", stage="basic_policy_comparison")
                evidence_bytes += size
                timing["baselineComparisonSeconds"] = time.monotonic() - baseline_started
                # Solved editorial misses need no winner or tie counterfactual proofs.
                interest = execution_budget.video_interestingness
                from .decision_selection import actor_interest_prerequisite_failures_v1
                eligible = not actor_interest_prerequisite_failures_v1(
                    artifact["proof"]["selectedAction"], artifact, baseline, interest)
                if featured_now and eligible:
                    win_started = time.monotonic()
                    win_request = native_round_win_quality_request_v1(request)
                    companion_options = dict(budget=budget, nodes=nodes, depth=depth, worlds=worlds,
                        remaining_seconds=lambda: remaining(decision_deadline), declared_seconds=declared_decision_seconds,
                        cancel_check=proof_cancel_check, private_recall=recall if private_recall is not None else None,
                        graph_workers=graph_workers, worker_cpu_callback=add_worker_cpu)
                    companion = _editorial_play_quality_artifact_v1(rules, normalized, current,
                        win_request, **companion_options)
                    remaining(decision_deadline)
                    size = len(canonical_json(companion).encode("utf-8"))
                    if evidence_bytes + artifact_bytes + size > maximum_evidence_bytes:
                        raise InformationSetCompilationStopped("maximum_evidence_bytes", stage="round_win_proof")
                    if companion["proof"]["completionStatus"] == "exact_complete":
                        win_companion = companion
                        evidence_bytes += size
                    timing["roundWinProofSeconds"] = time.monotonic() - win_started
                    if win_companion is not None and request.visibility == "actor_information_set":
                        from .result_authority import native_tie_rule_profile_v1, native_tie_margin_quality_request_v1, native_tie_allocation_universe_hash_v1
                        tie_started = time.monotonic()
                        source_rule = rules.data["scoring"]["contract"].get("buyer_tie_outcome")
                        if source_rule not in {"buyer_makes", "buyer_loses"}:
                            raise ValueError("content admission requires an explicit buyer tie rule")
                        universe_hash = native_tie_allocation_universe_hash_v1(artifact)
                        comparison = {"margin": {**artifact, "sourcePrefix": current.to_dict()}, "roundWin": win_companion,
                            "rulesSnapshot": copy.deepcopy(rules.data), "rulesHash": rules.sha256,
                            "exactStateId": normalized.exact_state_id, "allocationUniverseHash": universe_hash}
                        alternate_rule = "buyer_loses" if source_rule == "buyer_makes" else "buyer_makes"
                        variant = native_tie_rule_profile_v1(rules, alternate_rule)
                        variant_state = OrToolsExactStateNormalizerV1(variant, native_cp_sat=True).normalize_play_or_project(
                            current, maximum_seconds=remaining(decision_deadline), cancel_check=proof_cancel_check)
                        _, variant_baseline = _observer_safe_play_policy_artifact_v1(variant, variant_state,
                            baseline_request, actual_action=None,
                            policy_id=execution_budget.video_interestingness["baselinePolicyId"],
                            maximum_seconds=remaining(decision_deadline), cancel_check=proof_cancel_check)
                        if variant_baseline["proof"]["selectedAction"] != baseline["proof"]["selectedAction"]:
                            raise ValueError("tie comparison changed the observer-safe baseline")
                        alternative = {"rulesSnapshot": variant.data, "rulesHash": variant.sha256,
                            "exactStateId": variant_state.exact_state_id}
                        for key, comparison_request in (("margin", native_tie_margin_quality_request_v1(request)),
                                ("roundWin", win_request)):
                            alternative[key] = _editorial_play_quality_artifact_v1(variant, variant_state, current,
                                comparison_request, **companion_options)
                        remaining(decision_deadline)
                        if all(alternative[key]["proof"]["completionStatus"] == "exact_complete" for key in ("margin", "roundWin")):
                            alternative["allocationUniverseHash"] = native_tie_allocation_universe_hash_v1(alternative["margin"])
                            if alternative["allocationUniverseHash"] != universe_hash:
                                raise ValueError("tie comparison changed the legal allocation universe")
                            tie_core = {"sourceArtifactHash": artifact_hash, "sourcePrefixHash": current.sha256,
                                "baselineAction": baseline["proof"]["selectedAction"],
                                source_rule: comparison, alternate_rule: alternative}
                            tie_companion = {**tie_core, "sha256": canonical_sha256(tie_core)}
                            size = len(canonical_json(tie_companion).encode("utf-8"))
                            if evidence_bytes + artifact_bytes + size > maximum_evidence_bytes:
                                raise InformationSetCompilationStopped("maximum_evidence_bytes", stage="tie_rule_proof")
                            evidence_bytes += size
                        timing["tieRuleProofSeconds"] = time.monotonic() - tie_started
                        if tie_companion is not None:
                            inference_started = time.monotonic()
                            project_companion = _project_inference_quality_evidence_v1(rules, normalized, current,
                                artifact, win_companion, budget=budget, maximum_nodes=nodes, maximum_depth=depth,
                                maximum_conflicts=search_budget_profiles[request.budget_profile]['maximumConflicts'],
                                remaining_seconds=lambda: remaining(decision_deadline), cancel_check=proof_cancel_check)
                            if project_companion is not None:
                                size = len(canonical_json(project_companion).encode('utf-8'))
                                if evidence_bytes + artifact_bytes + size > maximum_evidence_bytes:
                                    raise InformationSetCompilationStopped('maximum_evidence_bytes', stage='project_inference')
                                evidence_bytes += size
                            timing['projectInferenceSeconds'] = time.monotonic() - inference_started
            if (_verified_normalizations is not None and _verified_artifacts is not None
                    and artifact_hash in _verified_artifacts):
                # Immediate readback owns this solved state. Do not retain CP models
                # or offer this reuse to stored/external evidence readers.
                _verified_normalizations[artifact_hash] = replace(normalized, _play_template=None)
            evidence_bytes += artifact_bytes
            if checkpoint_callback is not None:
                checkpoint_started = time.monotonic()
                checkpoint_callback({"schemaVersion": 1, "kind": "quality_proof_checkpoint_v1",
                    "rulesHash": rules.sha256, "stage": normalized.decision_kind,
                    "sourcePrefix": current.to_dict(), "artifactHash": artifact_hash,
                    "completionStatus": artifact["proof"]["completionStatus"], "artifact": artifact})
                checkpoint_callback_seconds += time.monotonic() - checkpoint_started
            expected_completion = "exact_complete" if request.mode == "exact" else "bounded_complete"
            proof_completion = artifact["proof"]["completionStatus"]
            if proof_completion != expected_completion:
                partial = {"stage": stage, "proofArtifactHash": artifact_hash,
                           "completionStatus": proof_completion}
                status = "model_error" if proof_completion == "model_error" else "incomplete"
                break
            quality = proof_document.get("actionQuality")
            selected_bid_quality = (
                proof_document.get("kind") in {"native_bid_selected_action_quality_model_v1", "native_observed_bid_quality_model_v1"}
                and normalized.decision_kind == "bid"
                and request.mode == "exact"
            )
            paired_play_quality = proof_document.get("kind") == "native_complete_state_policy_pair_quality_v1"
            selected_observed_quality = proof_document.get("kind") in {
                "native_observed_play_quality_model_v1", "native_observed_project_quality_model_v1",
                "native_bounded_play_quality_model_v1"}
            if (
                not isinstance(quality, Mapping)
                or not isinstance(quality.get("optimalActions"), list)
                or (not quality.get("optimalActions") and not (selected_observed_quality and auditing
                    and quality.get("actualActionOptimal") is False))
                or (not selected_bid_quality and not paired_play_quality and not selected_observed_quality
                    and quality.get("optimalActionsComplete", True) is not True)
                or quality.get("rootActorRelation") != "target"
                or (not auditing and quality.get("actualActionOptimal") is not True)
                or quality.get("actualAction") != artifact["proof"]["selectedAction"]
                or (request.selected_action is not None and artifact["proof"]["selectedAction"] != request.selected_action)
                or artifact["proof"]["selectedAction"] not in {row["code"] for row in normalized.legal_actions}
            ):
                status = "model_error"
                partial = {"stage": stage, "reason": "action_quality_binding_disagreement",
                           "proofArtifactHash": artifact_hash}
                break
            if not auditing:
                next_prefix = normalizer.append_normalized_action(current, normalized, artifact["proof"]["selectedAction"])
            else:
                next_document = current.to_dict()
                next_document.pop("sha256")
                next_document["stateAssertions"] = {}
                next_document["projectAssertions"] = []
                field = "auctionActions" if auction_only else "projectActions" if normalized.decision_kind == "project" else "playActions"
                next_document[field] = [*next_document[field], dict(recorded)]
                if (len(next_document["auctionActions"]) == len(recorded_prefix.auction_actions)
                        if auction_only else len(next_document["playActions"]) == len(recorded_prefix.play_actions)
                        and len(next_document.get("projectActions", [])) == len(recorded_prefix.project_actions)):
                    next_document["stateAssertions"] = dict(recorded_prefix.state_assertions)
                    next_document["projectAssertions"] = [dict(item) for item in recorded_prefix.project_assertions]
                next_prefix = ExactTracePrefixV1.build(next_document)
            stage = "selected_action_normalization"
            stage_started = time.monotonic()
            transition_deadline = time.monotonic() + maximum_decision_seconds
            if execution_budget is not None and execution_budget.exact_tail_budget_profile is not None:
                transition_deadline = min(transition_deadline, time.monotonic() + declared_decision_seconds)
            try:
                next_state = normalize_current(
                    next_prefix,
                    maximum_seconds=remaining(transition_deadline),
                    cancel_check=proof_cancel_check,
                )
            except (ValueError, AssertionError) as exc:
                status = "model_error"
                partial = {"stage": stage, "reason": str(exc), "proofArtifactHash": artifact_hash}
                break
            timing["selectedActionNormalizationSeconds"] = time.monotonic() - stage_started
            remaining(transition_deadline)
            row = {
                ("bidIndex" if auction_only else "playIndex"): len(current.auction_actions) if auction_only else len(current.play_actions), "actor": actor.name.lower(),
                "targetTeam": actor.team.name.lower(), "action": artifact["proof"]["selectedAction"],
                **({"decisionKind": normalized.decision_kind} if includes_project_decisions or auction_only or paired_play_quality else {}),
                "sourcePrefixHash": current.sha256, "exactStateId": normalized.exact_state_id,
                "nextPrefixHash": next_prefix.sha256, "nextExactStateId": next_state.exact_state_id,
                "proofArtifactHash": artifact_hash, "actionQualityHash": quality["sha256"],
                **({"proofBudgetHash": budget.sha256} if budget is not None else {}),
                **(
                    {"observerInformationHash": quality["observerInformationHash"]}
                    if isinstance(quality.get("observerInformationHash"), str)
                    else {}
                ),
                "actualActionOptimal": quality["actualActionOptimal"], "regret": quality["regret"],
                "optimalActionCount": None if selected_bid_quality or paired_play_quality or selected_observed_quality else len(quality["optimalActions"]),
                **({key: quality[key] for key in ("optimalActionCountLowerBound", "optimalActionCountUpperBound")}
                   if selected_bid_quality or paired_play_quality or selected_observed_quality else {}),
                "selectionTieBreak": (
                    quality.get("selectionTieBreak")
                    if not auditing and len(quality["optimalActions"]) > 1 else None
                ),
                "selectionReason": (
                    "native_complete_state_equilibrium_path" if paired_play_quality and not auditing else
                    "native_observer_safe_policy_value"
                    if policy_artifact is not None
                    and proof_document.get("kind") == "observer_safe_play_policy_model_v1"
                    and not auditing
                    else ("native_complete_state_best_value" if request.visibility == "complete_state"
                          else "native_actor_information_set_best_value") if not auditing else "recorded_action"
                ),
            }
            decisions.append({**row, "sha256": canonical_sha256(row)})
            if baseline is not None:
                baseline_evidence[artifact_hash] = baseline
            if win_companion is not None:
                round_win_evidence[artifact_hash] = win_companion
            if tie_companion is not None:
                tie_rule_evidence[artifact_hash] = tie_companion
            if project_companion is not None:
                project_inference_evidence[artifact_hash] = project_companion
            if (featured_now and request.mode == "exact"
                    and (request.visibility == "complete_state" or proof_document.get("kind") == "actor_information_set_cegis_model_v1")
                    and len(normalized.legal_actions) > 1 and current.project_actions_present
                    and budget is not None and budget.native_observed is not None
                    and execution_budget is not None
                    and (execution_budget.video_interestingness or {}).get("schemaVersion") == 2):
                raw_context = (current, replace(normalized, _play_template=None), request, budget, nodes, depth,
                    artifact_hash, worlds,
                    recall if private_recall is not None else None)
            if checkpoint_callback is not None:
                timing["elapsedSeconds"] = time.monotonic() - started
                timing["checkpointCallbackSeconds"] = checkpoint_callback_seconds
                checkpoint_started = time.monotonic()
                checkpoint_callback({"schemaVersion": 1, "kind": "quality_decision_checkpoint_v1",
                    "rulesHash": rules.sha256, "stage": normalized.decision_kind,
                    "sourcePrefix": current.to_dict(), "nextPrefix": next_prefix.to_dict(),
                    "decision": decisions[-1], "artifactHash": artifact_hash,
                    **({"publicAwardedProjects": [
                        {key: project[key] for key in ("owner", "team", "kind", "value")}
                        for project in next_state.state["projects"]
                        if project["scoring"] and project["revealed"]
                    ]} if normalized.decision_kind == "play"
                        and len(normalized.state.get("completedTricks") or ()) == 1
                        and len(next_state.state.get("completedTricks") or ()) == 2 else {}),
                    "timing": dict(timing)})
                checkpoint_callback_seconds += time.monotonic() - checkpoint_started
            current, normalized = next_prefix, next_state
            if (
                stop_on_recorded_quality_failure
                and auditing
                and decisions[-1]["actualActionOptimal"] is False
            ):
                status = "quality_failed"
                partial = {
                    "stage": "action_analysis",
                    "reason": "recorded_action_not_optimal",
                    "proofArtifactHash": artifact_hash,
                }
                break
            if _curated_focal_only:
                # A verified focal rejection needs no proof of later actors' choices.
                partial = {"stage": "curated_focal_precheck", "reason": "focal_decision_complete"}
                break
            cpu_deadline = math.inf
        else:
            status = "complete" if (not normalized.legal_actions and normalized.next_actor is None
                if auction_only else normalized.phase == "scored") else "model_error"
            if status == "complete" and featured_play_index is not None and featured_matches != 1:
                status = "model_error"
                partial = {"stage": "action_analysis", "reason": "featured_decision_not_covered"}
            if recorded_prefix is not None and not continue_recorded_prefix and current.sha256 != recorded_prefix.sha256:
                status = "model_error"
                partial = {"stage": "recorded_trace_completion", "reason": "recorded_trace_not_consumed"}
            if status == "complete" and any(row["actualActionOptimal"] is not True for row in decisions):
                status = "quality_failed"
                failed = next(row for row in decisions if row["actualActionOptimal"] is not True)
                partial = {
                    "stage": "action_analysis",
                    "reason": "recorded_action_not_optimal",
                    "proofArtifactHash": failed["proofArtifactHash"],
                }
    except (InformationSetCompilationStopped, ExactNormalizationStopped) as exc:
        partial = exc.partial_receipt
    except (RuntimeError, TimeoutError) as exc:
        partial = {"stage": stage, "reason": str(exc)}

    if partial is not None and cpu_deadline != math.inf and cpu_time() >= cpu_deadline:
        partial = {**partial, "exhaustionReason": "maximum_cpu_seconds"}

    if partial is not None and pair_synthesis is not None:
        partial = {**partial, "synthesis": pair_synthesis}

    if status == "complete" and raw_context is not None:
        from .result_authority import native_raw_point_quality_request_v1
        raw_prefix, raw_state, raw_request, raw_budget, raw_nodes, raw_depth, source_hash, raw_worlds, raw_recall = raw_context
        raw_request = native_raw_point_quality_request_v1(raw_request)
        raw_started = time.monotonic()
        raw_cpu_started = cpu_time()
        raw_timing = {"status": "incomplete", "sourceArtifactHash": source_hash}
        cpu_deadline = raw_cpu_started + raw_budget.maximum_cpu_seconds
        raw_deadline = min(deadline, raw_started + maximum_decision_seconds,
            raw_started + raw_budget.maximum_wall_seconds)
        try:
            if raw_request.visibility == "actor_information_set":
                declared = min(maximum_decision_seconds, raw_budget.maximum_wall_seconds)
                raw_compilation = compile_observed_decision_v1(rules, raw_state, raw_prefix, {"kind": "play"}, raw_request,
                    maximum_nodes=raw_nodes, maximum_depth=raw_depth, maximum_worlds=raw_worlds,
                    maximum_seconds=remaining(raw_deadline), declared_maximum_seconds=declared,
                    native_cp_sat=True, private_recall=raw_recall, cancel_check=stopped,
                    graph_workers=graph_workers, worker_cpu_callback=add_worker_cpu)
                raw_proof = prove_actor_information_set_v1(raw_compilation, raw_request,
                    maximum_nodes=raw_nodes, maximum_worlds=raw_worlds,
                    maximum_seconds=remaining(raw_deadline), declared_maximum_seconds=declared,
                    native_action_quality=True, cancel_check=stopped, policy_search=raw_budget.policy_search)
                companion = {"request": raw_request.to_dict(), "proof": raw_proof.to_dict(),
                    "proofModel": raw_proof.model_document(), "graph": raw_compilation.graph.to_dict(),
                    "allocationModel": dict(raw_compilation.graph.model_document), "sourcePrefix": raw_prefix.to_dict()}
            else:
                companion = _complete_state_quality_artifact_v1(rules, raw_state, raw_prefix, {"kind": "play"},
                    raw_request, budget=raw_budget, maximum_nodes=raw_nodes, maximum_depth=raw_depth,
                    maximum_seconds=remaining(raw_deadline), cancel_check=stopped)
            remaining(raw_deadline)
            size = len(json.dumps(companion, ensure_ascii=False, sort_keys=True,
                separators=(",", ":")).encode("utf-8"))
            if evidence_bytes + size > maximum_evidence_bytes:
                raise InformationSetCompilationStopped("maximum_evidence_bytes", stage="raw_point_editorial_proof")
            if companion["proof"]["completionStatus"] == "exact_complete":
                raw_evidence[source_hash] = companion
                evidence_bytes += size
                raw_timing.update(status="complete", evidenceBytes=size)
        except (InformationSetCompilationStopped, ExactNormalizationStopped) as exc:
            reason = getattr(exc, "reason", exc.partial_receipt.get("exhaustionReason"))
            if cpu_time() >= cpu_deadline and not (cancel_check is not None and cancel_check()):
                reason = "maximum_cpu_seconds"
            if reason == "cancelled":
                raise
            raw_timing.update(reason=reason,
                partialReceiptHash=canonical_sha256(exc.partial_receipt))
        except (RuntimeError, TimeoutError) as exc:
            raw_timing.update(reason=type(exc).__name__, reasonHash=canonical_sha256(str(exc)))
        raw_timing.update(elapsedSeconds=time.monotonic() - raw_started,
            cpuSeconds=cpu_time() - raw_cpu_started)

    core = {
        "schemaVersion": 1, "kind": "native_quality_auction_continuation_v1" if auction_only else "native_quality_play_continuation_v1",
        "status": status, "qualificationAuthority": "cp_sat",
        "scope": "auction_suffix_only" if auction_only else "project_play_suffix" if includes_project_decisions else "play_suffix_only",
        "operation": "audit_then_generate" if continue_recorded_prefix else "generate" if recorded_prefix is None else "audit_recorded",
        **({"submittedPrefix": recorded_prefix.to_dict()} if continue_recorded_prefix else {}),
        "recordedPrefixHash": recorded_prefix.sha256 if recorded_prefix is not None else None,
        "rulesHash": rules.sha256, "qualityRequestHash": quality_request.sha256,
        "sourcePrefixHash": prefix.sha256,
        ("coveredFromBidIndex" if auction_only else "coveredFromPlayIndex"): len(prefix.auction_actions) if auction_only else len(prefix.play_actions),
        ("coveredToBidIndex" if auction_only else "coveredToPlayIndex"): len(current.auction_actions) if auction_only else len(current.play_actions), "decisions": decisions,
        "finalPrefix": current.to_dict(),
        "finalExactStateId": normalized.exact_state_id if normalized is not None else None,
        "artifacts": artifacts, "partialReceipt": partial,
        "limits": {"maximumNodesPerDecision": maximum_nodes, "maximumDepth": maximum_depth,
                   "maximumWorldsPerDecision": maximum_worlds, "maximumSeconds": maximum_seconds,
                   "maximumDecisionSeconds": maximum_decision_seconds,
                   "maximumEvidenceBytes": maximum_evidence_bytes},
    }
    if quality_policy is not None:
        core["playingQuality"] = quality_policy.to_dict()
    if auction_only and normalized is not None:
        core["auctionOutcome"] = {"allPassed": normalized.state.get("allPassed", False),
                                  "buyer": normalized.state.get("buyer"),
                                  "contract": normalized.state.get("contract")}
    if recall:
        core["privateRecallHash"] = canonical_sha256(recall)
    if escalation_events is not None:
        core["escalationHistoryHash"] = canonical_sha256(escalation_events)
    if budget_snapshots:
        core["proofBudgetProfiles"] = budget_snapshots
    if paired_evidence:
        core["pairedPolicyEvidence"] = paired_evidence
    if raw_evidence:
        core["rawPointEvidence"] = raw_evidence
    if round_win_evidence:
        core["roundWinEvidence"] = round_win_evidence
    if baseline_evidence:
        core["baselineEvidence"] = baseline_evidence
    if tie_rule_evidence:
        core["tieRuleEvidence"] = tie_rule_evidence
    if project_inference_evidence:
        core['projectInferenceEvidence'] = project_inference_evidence
    result = {**core, "sha256": canonical_sha256(core),
              "executionReceipt": {"elapsedSeconds": time.monotonic() - started, "evidenceBytes": evidence_bytes,
                                   "decisionTimings": decision_timings,
                                   "initialNormalizationSeconds": initial_normalization_seconds,
                                   "checkpointCallbackSeconds": checkpoint_callback_seconds}}
    if raw_timing is not None:
        result["executionReceipt"]["rawPointProof"] = raw_timing
    if not auction_only and status == "complete" and final_normalized_callback is not None:
        if not isinstance(normalized, OrToolsNormalizedExactStateV1):
            raise ValueError("completed play quality is missing its normalized terminal state")
        final_normalized_callback(normalized)
    return result


def _read_back_generated_quality_v1(result, remaining, **kwargs):
    """Preserve owned decisions when final readback exhausts its shared allowance."""
    from .result_authority import _validate_native_decision_quality_evidence_v1
    started = time.monotonic()
    try:
        _validate_native_decision_quality_evidence_v1(result, maximum_seconds=remaining(), **kwargs)
    except (InformationSetCompilationStopped, ExactNormalizationStopped) as stopped:
        if isinstance(stopped, ExactNormalizationStopped):
            stopped = InformationSetCompilationStopped(stopped.partial_receipt["exhaustionReason"],
                stage="quality_readback", evidence={"normalizationReceipt": stopped.partial_receipt})
        if result.get("partialReceipt") is not None:
            stopped = InformationSetCompilationStopped(stopped.reason, stage="quality_readback", evidence={
                "generationReceipt": result["partialReceipt"], "readbackReceipt": stopped.partial_receipt})
        core = {key: value for key, value in result.items() if key not in {"sha256", "executionReceipt"}}
        # Later cancellation cannot erase a suboptimal recorded move already
        # refuted by this owned generation path. It still cannot qualify a round.
        core.update(status="quality_failed" if result.get("status") == "quality_failed" else "incomplete",
            partialReceipt=stopped.partial_receipt)
        result = {**core, "sha256": canonical_sha256(core), "executionReceipt": result["executionReceipt"]}
    result["executionReceipt"]["readbackSeconds"] = time.monotonic() - started
    return result


def resolve_quality_play_continuation_v1(rules, prefix, quality_request, *, _play_quality_cache=None,
                                        _verified_artifacts=None, **kwargs):
    """Generate or audit the project/play suffix under declared quality bounds."""
    from .result_authority import _quality_readback_clock
    started = time.monotonic()
    remaining = _quality_readback_clock(kwargs["maximum_seconds"]
        if kwargs.get("runtime_maximum_seconds") is None else kwargs["runtime_maximum_seconds"],
        kwargs.get("cancel_check"))
    # Keep freshly reverified context tokens owned by this execution, never its JSON/cache.
    verified_artifacts = _verified_artifacts if _verified_artifacts is not None else set()
    verified_normalizations: dict[str, OrToolsNormalizedExactStateV1] = {}
    cache_key, result, decision_checkpoints = None, None, []
    if _play_quality_cache is not None and kwargs.get("_admissible_play_actions") is None:
        if not isinstance(_play_quality_cache, _BidQualityCacheV1):
            raise TypeError("play reuse requires the private bounded continuation cache")
        execution = kwargs.get("execution_budget")
        execution = execution.to_dict() if execution is not None else None
        if execution is not None:
            for key in ("sha256", "familyTarget", "familyCatalog", "candidateSource", "proposalSeedCount",
                        "proposalMaximumSeconds", "proposalPlayIndices", "acceptedLessonLibrary"):
                execution.pop(key, None)
        recorded = kwargs.get("recorded_trace")
        cache_key = canonical_sha256({"stage": "play", "rulesHash": rules.sha256,
            "sourcePrefixHash": prefix.sha256, "request": quality_request.to_dict(),
            "recordedTraceHash": recorded.sha256 if recorded is not None else None,
            "execution": execution, "proofProfiles": kwargs.get("proof_budget_profiles"),
            "searchProfiles": kwargs.get("search_budget_profiles"),
            "limits": {key: kwargs.get(key) for key in ("maximum_nodes", "maximum_depth", "maximum_worlds",
                "maximum_seconds", "maximum_decision_seconds", "maximum_evidence_bytes", "private_recall",
                "escalation_events", "opportunity_order", "continue_recorded_prefix",
                "stop_on_recorded_quality_failure", "featured_play_index", "featured_decision_point",
                "featured_selector_dynamic", "recorded_actions", "paired_policy_proofs", "graph_workers",
                "search_generated", "_curated_focal_only")}})
        cached = _play_quality_cache.get(cache_key, with_checkpoints=True)
        if cached is not None:
            result = cached["result"]
            decision_checkpoints = cached.get("decisionCheckpoints", [])
    cache_hit = result is not None
    if cache_hit:
        previous = result["executionReceipt"]
        result["executionReceipt"] = {"elapsedSeconds": 0, "evidenceBytes": previous["evidenceBytes"],
            "cachedSourceElapsedSeconds": previous.get("cachedSourceElapsedSeconds", previous["elapsedSeconds"]),
            "cachedDecisionTimings": previous.get("cachedDecisionTimings", previous.get("decisionTimings", [])),
            "decisionTimings": [], "initialNormalizationSeconds": 0, "checkpointCallbackSeconds": 0}
    else:
        original_checkpoint = kwargs.get("checkpoint_callback")
        def retain_checkpoint(payload):
            if payload.get("kind") == "quality_decision_checkpoint_v1":
                decision_checkpoints.append(copy.deepcopy(dict(payload)))
            if original_checkpoint is not None:
                original_checkpoint(payload)
        resolve_kwargs = {**kwargs, "checkpoint_callback": retain_checkpoint} if _play_quality_cache is not None else kwargs
        result = _resolve_quality_continuation_v1(rules, prefix, quality_request, auction_only=False,
            _verified_artifacts=verified_artifacts, _verified_normalizations=verified_normalizations,
            **resolve_kwargs)
    policy = quality_request if isinstance(quality_request, ScenarioPlayingQualityPolicyV1) else None
    request = policy.request_for("play", "north") if policy is not None else quality_request
    readback_artifacts = set() if cache_hit else verified_artifacts
    result = _read_back_generated_quality_v1(result, remaining, source_prefix=prefix, rules_hash=rules.sha256,
        verified_artifacts=readback_artifacts, verified_normalizations=verified_normalizations,
        request=request, policy=policy, rules=rules,
        search_budget_profiles=kwargs.get("search_budget_profiles"), private_recall=kwargs.get("private_recall"),
        execution_budget=kwargs.get("execution_budget"), escalation_events=kwargs.get("escalation_events"),
        opportunity_order=kwargs.get("opportunity_order"), cancel_check=kwargs.get("cancel_check"))
    if result["status"] == "complete":
        verified_artifacts.update(readback_artifacts)
    result["executionReceipt"]["playCacheHit"] = cache_hit
    if cache_hit and result["status"] == "complete" and kwargs.get("checkpoint_callback") is not None:
        for checkpoint in decision_checkpoints:
            replay = copy.deepcopy(checkpoint)
            replay["timing"] = {"cachedSourceTiming": replay.get("timing"),
                "elapsedSeconds": time.monotonic() - started, "cacheHit": True}
            kwargs["checkpoint_callback"](replay)
    if cache_hit and result["status"] == "complete" and kwargs.get("final_normalized_callback") is not None:
        try:
            final = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(
                ExactTracePrefixV1.build(result["finalPrefix"]), maximum_seconds=remaining(),
                cancel_check=kwargs.get("cancel_check"))
        except (InformationSetCompilationStopped, ExactNormalizationStopped) as stopped:
            core = {key: value for key, value in result.items() if key not in {"sha256", "executionReceipt"}}
            core.update(status="incomplete", partialReceipt=stopped.partial_receipt)
            result = {**core, "sha256": canonical_sha256(core), "executionReceipt": result["executionReceipt"]}
        else:
            kwargs["final_normalized_callback"](final)
    result["executionReceipt"]["elapsedSeconds"] = time.monotonic() - started
    if _play_quality_cache is not None and cache_key is not None:
        _play_quality_cache.put(cache_key, result, checkpoints=decision_checkpoints)
        if result["status"] == "complete" and kwargs.get("checkpoint_callback") is not None:
            entry = {"key": cache_key, "result": result, "decisionCheckpoints": decision_checkpoints}
            kwargs["checkpoint_callback"]({"kind": "bid_quality_cache_entry_v1",
                "entry": {**entry, "sha256": canonical_sha256(entry)}})
    return result


def resolve_quality_auction_continuation_v1(rules, prefix, quality_request, *, _bid_quality_cache=None, **kwargs):
    """Generate or audit native bid decisions; the round composer owns admission."""
    from .result_authority import _quality_readback_clock
    started = time.monotonic()
    remaining = _quality_readback_clock(kwargs["maximum_seconds"]
        if kwargs.get("runtime_maximum_seconds") is None else kwargs["runtime_maximum_seconds"],
        kwargs.get("cancel_check"))
    request = (quality_request.request_for("bid", "north")
               if isinstance(quality_request, ScenarioPlayingQualityPolicyV1) else quality_request)
    cache_key, result = None, None
    if _bid_quality_cache is not None:
        if not isinstance(_bid_quality_cache, _BidQualityCacheV1):
            raise TypeError("auction reuse requires the private bounded bid cache")
        execution = kwargs.get("execution_budget")
        execution = execution.to_dict() if execution is not None else None
        if execution is not None:
            for key in ("sha256", "familyTarget", "familyCatalog", "candidateSource", "proposalSeedCount",
                        "proposalMaximumSeconds", "proposalPlayIndices", "acceptedLessonLibrary"):
                execution.pop(key, None)
        recorded = kwargs.get("recorded_trace")
        cache_key = canonical_sha256({"rulesHash": rules.sha256, "sourcePrefixHash": prefix.sha256,
            "recordedAuctionHash": recorded.sha256 if recorded is not None else None,
            "request": quality_request.to_dict(), "execution": execution,
            "proofProfiles": kwargs.get("proof_budget_profiles"),
            "searchProfiles": kwargs.get("search_budget_profiles"),
            "limits": {key: kwargs.get(key) for key in ("maximum_nodes", "maximum_depth", "maximum_worlds",
                "maximum_seconds", "maximum_decision_seconds", "maximum_evidence_bytes", "opportunity_order",
                "continue_recorded_prefix", "stop_on_recorded_quality_failure", "search_generated")}})
        result = _bid_quality_cache.get(cache_key)
    cache_hit = result is not None
    if cache_hit:
        previous = result["executionReceipt"]
        result["executionReceipt"] = {"elapsedSeconds": 0, "evidenceBytes": previous["evidenceBytes"],
            "cachedSourceElapsedSeconds": previous.get("cachedSourceElapsedSeconds", previous["elapsedSeconds"]),
            "cachedDecisionTimings": previous.get("cachedDecisionTimings", previous.get("decisionTimings", [])),
            "decisionTimings": [], "initialNormalizationSeconds": 0, "checkpointCallbackSeconds": 0}
    verified_artifacts: set[str] = set()
    if result is None:
        result = _resolve_quality_continuation_v1(rules, prefix, quality_request, auction_only=True,
            _verified_artifacts=verified_artifacts, **kwargs)
    # A cache hit never carries an immediate-worker verification token.
    result = _read_back_generated_quality_v1(result, remaining, source_prefix=prefix, rules_hash=rules.sha256,
        request=request, rules=rules, verified_artifacts=frozenset(verified_artifacts),
        search_budget_profiles=kwargs.get("search_budget_profiles"), execution_budget=kwargs.get("execution_budget"),
        cancel_check=kwargs.get("cancel_check"))
    result["executionReceipt"]["elapsedSeconds"] = time.monotonic() - started
    result["executionReceipt"]["bidCacheHit"] = cache_hit
    if _bid_quality_cache is not None:
        _bid_quality_cache.put(cache_key, result)
        if result["status"] == "complete" and kwargs.get("checkpoint_callback") is not None:
            entry = {"key": cache_key, "result": result}
            kwargs["checkpoint_callback"]({"kind": "bid_quality_cache_entry_v1",
                "entry": {**entry, "sha256": canonical_sha256(entry)}})
    return result


def resolve_quality_post_auction_continuation_v1(
    rules: RuleBook, prefix: ExactTracePrefixV1, decision_point: Mapping[str, Any],
    quality_policy: ScenarioPlayingQualityPolicyV1, *, opportunity_order: Sequence[str],
    maximum_nodes: int, maximum_depth: int, maximum_worlds: int,
    maximum_seconds: float, maximum_decision_seconds: float, maximum_escalation_decisions: int,
    maximum_evidence_bytes: int, recorded_events: Sequence[Mapping[str, Any]] | None = None,
    recorded_trace: ExactTracePrefixV1 | None = None,
    proof_budget_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    search_budget_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    runtime_maximum_seconds: float | None = None,
    continue_recorded_prefix: bool = False,
    stop_on_recorded_quality_failure: bool = False,
    checkpoint_callback: Callable[[Mapping[str, Any]], None] | None = None,
    final_normalized_callback: Callable[[OrToolsNormalizedExactStateV1], None] | None = None,
    execution_budget: ScenarioQualityExecutionBudgetV1 | None = None,
    featured_play_index: int | None = None,
    featured_decision_point: Mapping[str, Any] | None = None,
    featured_selector_dynamic: bool = False,
    _play_quality_cache=None,
    _verified_artifacts: set[str] | None = None,
    _curated_focal_precheck: bool = False,
    graph_workers: int = 1,
    _admissible_play_actions: Callable[..., tuple[str, ...] | None] | None = None,
) -> dict[str, Any]:
    """Connect existing stage proofs; this neither audits bids nor admits a round.

    Evidence limits count retained proof/stage payloads, not envelope metadata.
    Individual stage proof boundaries remain authoritative, including world limits.
    """
    if type(graph_workers) is not int or not 1 <= graph_workers <= 4:
        raise ValueError("graph_workers must be within the granted 1..4 worker ceiling")
    if not isinstance(prefix, ExactTracePrefixV1) or not isinstance(quality_policy, ScenarioPlayingQualityPolicyV1):
        raise TypeError("post-auction quality requires an exact prefix and a per-decision policy")
    if (not prefix.project_actions_present or prefix.project_actions or prefix.play_actions
            or prefix.escalation_actions):
        raise ValueError("post-auction quality must start before escalation, with explicit empty project actions")
    if (recorded_events is None) != (recorded_trace is None):
        raise ValueError("recorded audit requires both escalation events and a complete trace")
    if type(continue_recorded_prefix) is not bool:
        raise TypeError("continue_recorded_prefix must be boolean")
    if continue_recorded_prefix and recorded_trace is None:
        raise ValueError("post-auction prefix continuation requires a recorded prefix")
    escalation_prefix_only = bool(continue_recorded_prefix and not (
        recorded_trace.project_actions or recorded_trace.play_actions))
    if escalation_prefix_only and (recorded_trace.state_assertions or recorded_trace.project_assertions):
        raise ValueError("partial escalation boundary assertions are not yet supported")
    if recorded_trace is not None:
        recorded_trace = _validate_quality_recorded_trace(prefix, recorded_trace, allow_escalation_extension=True,
            allow_partial=continue_recorded_prefix)
    policy = ScenarioPlayingQualityPolicyV1.build(quality_policy.to_dict())
    _budgets, snapshots = _quality_budget_snapshots(
        {row["budgetProfile"] for row in policy.decisions.values()}, proof_budget_profiles,
        execution_budget=execution_budget)
    profiles = snapshots if proof_budget_profiles is not None else None
    runtime_seconds = maximum_seconds if runtime_maximum_seconds is None else runtime_maximum_seconds
    if (type(runtime_seconds) not in {int, float} or not math.isfinite(runtime_seconds)
            or not 0 < runtime_seconds <= maximum_seconds):
        raise ValueError("runtime allowance must be positive and cannot exceed the declared budget")
    started = time.monotonic()
    if execution_budget is not None:
        maximum_worlds = execution_budget.resolve_hidden_world_cap(maximum_worlds, play_index=None)
    escalation = resolve_quality_escalation_continuation_v1(rules, prefix, decision_point,
        # The template target is rebound to each native-derived actor in the driver.
        policy.request_for("escalation", Seat.parse(decision_point.get("selector", {}).get("actingSeat") or "north").name.lower()),
        opportunity_order=opportunity_order, maximum_nodes=maximum_nodes, maximum_depth=maximum_depth,
        maximum_worlds=maximum_worlds, maximum_seconds=maximum_seconds,
        maximum_decision_seconds=maximum_decision_seconds, maximum_decisions=maximum_escalation_decisions,
        maximum_evidence_bytes=maximum_evidence_bytes, recorded_events=recorded_events,
        continue_recorded_prefix=continue_recorded_prefix,
        runtime_maximum_seconds=runtime_seconds,
        proof_budget_profiles=profiles, cancel_check=cancel_check,
        **({"search_budget_profiles": search_budget_profiles} if search_budget_profiles is not None else {}),
        checkpoint_callback=checkpoint_callback)
    if checkpoint_callback is not None:
        checkpoint_callback({"schemaVersion": 1, "kind": "quality_stage_checkpoint_v1",
            "rulesHash": rules.sha256, "stage": "escalation",
            "elapsedSeconds": time.monotonic() - started})
    documents = [escalation[key] for key in ("initialNormalization", "initialState", "projectPlayHandoff")
                 if escalation[key] is not None] + list(escalation["artifacts"].values())
    evidence_bytes = sum(len(json.dumps(document, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")) for document in documents)
    status, partial = escalation["status"], escalation["partialReceipt"]
    final_prefix, play, play_execution = escalation["finalPrefix"], None, None
    if status == "complete":
        if escalation["projectPlayHandoff"] is None:
            raise ValueError("complete escalation is missing its verified project/play handoff")
        if recorded_trace is not None:
            boundary = (escalation["artifacts"][escalation["decisions"][len(recorded_events) - 1]["artifactHash"]]["nextPublicPrefix"]
                if escalation_prefix_only and recorded_events else prefix.to_dict() if escalation_prefix_only else final_prefix)
            if (stop_on_recorded_quality_failure and continue_recorded_prefix
                    and not escalation_prefix_only
                    and ExactTracePrefixV1.build(boundary).escalation_actions != recorded_trace.escalation_actions):
                raise _QualityScenarioTargetMiss("recorded escalation actions differ from quality continuation")
            _validate_quality_recorded_trace(ExactTracePrefixV1.build(boundary), recorded_trace,
                allow_partial=continue_recorded_prefix)
        if (_curated_focal_precheck and recorded_trace is not None
                and execution_budget is not None and execution_budget.exact_tail_proof
                and execution_budget.exact_play_from_trick is not None
                and featured_play_index == len(recorded_trace.play_actions)
                and 4 * (execution_budget.exact_play_from_trick - 1) <= featured_play_index < 32):
            # Only reject immutable proposals here; a potential passer still needs the full audit.
            remaining = runtime_seconds - (time.monotonic() - started)
            if remaining > 0 and evidence_bytes < maximum_evidence_bytes and not (cancel_check and cancel_check()):
                precheck_started, precheck_partial = time.monotonic(), None
                try:
                    precheck = resolve_quality_play_continuation_v1(rules, recorded_trace, policy,
                        maximum_nodes=maximum_nodes, maximum_depth=maximum_depth, maximum_worlds=maximum_worlds,
                        maximum_seconds=maximum_seconds, runtime_maximum_seconds=remaining,
                        maximum_decision_seconds=maximum_decision_seconds,
                        maximum_evidence_bytes=maximum_evidence_bytes - evidence_bytes,
                        private_recall=escalation.get("privateRecall"), escalation_events=escalation.get("events"),
                        opportunity_order=opportunity_order, proof_budget_profiles=profiles,
                        search_budget_profiles=search_budget_profiles, cancel_check=cancel_check,
                        checkpoint_callback=checkpoint_callback,
                        execution_budget=execution_budget, _play_quality_cache=_play_quality_cache,
                        _verified_artifacts=_verified_artifacts, _curated_focal_only=True,
                        graph_workers=graph_workers)
                except (InformationSetCompilationStopped, ExactNormalizationStopped) as stopped:
                    precheck = None
                    precheck_partial = stopped.partial_receipt
                if checkpoint_callback is not None:
                    checkpoint_callback({"schemaVersion": 1, "kind": "quality_stage_checkpoint_v1",
                        "rulesHash": rules.sha256, "stage": "curated_focal_precheck",
                        "elapsedSeconds": time.monotonic() - precheck_started,
                        "status": precheck["status"] if precheck is not None else "incomplete",
                        "partialReceipt": precheck.get("partialReceipt") if precheck is not None else precheck_partial})
                if precheck is not None:
                    evidence_bytes += precheck["executionReceipt"]["evidenceBytes"]
                    focal_checked = (precheck["status"] == "incomplete"
                        and precheck.get("partialReceipt") == {
                            "stage": "curated_focal_precheck", "reason": "focal_decision_complete"}
                        and len(precheck.get("decisions", ())) == 1
                        and precheck["decisions"][0].get("playIndex") == featured_play_index)
                    if precheck["status"] == "complete" or focal_checked:
                        from .decision_selection import actor_interest_prerequisite_failures_v1
                        row = next((row for row in precheck.get("decisions", ())
                                    if row.get("playIndex") == featured_play_index), {})
                        proof_hash = row.get("proofArtifactHash")
                        artifact = precheck.get("artifacts", {}).get(proof_hash, {})
                        proof = artifact.get("proof") or {}
                        # A native-proved forced choice cannot supply a non-obvious alternative.
                        forced = (artifact.get("request", {}).get("mode") == "exact"
                            and artifact.get("request", {}).get("visibility") == "actor_information_set"
                            and proof.get("completionStatus") == "exact_complete"
                            and proof.get("solverStatus") == "optimal"
                            and proof.get("completionScope") == "action_optimality_only"
                            and artifact.get("proofModel", {}).get("kind") == "forced_action_optimality_model_v1"
                            and artifact.get("graph", {}).get("kind") == "forced_legal_action_graph_v1"
                            and artifact["graph"].get("legalActions") == [row.get("action")]
                            and proof.get("selectedAction") == row.get("action")
                            and [root.get("action") for root in proof.get("rootActions", ())] == [row.get("action")])
                        failures = (["interestingness_forced_legal_action"] if forced else
                            actor_interest_prerequisite_failures_v1(row.get("action"), artifact,
                                precheck.get("baselineEvidence", {}).get(proof_hash, {}),
                                execution_budget.video_interestingness))
                        if failures and not any(reason.endswith("_unavailable") for reason in failures):
                            raise _QualityEditorialMiss({"sourcePrefix": recorded_trace.to_dict(),
                                "receipt": precheck, "blockers": failures,
                                "researchClaimStatus": "not_evaluated"})
        remaining = runtime_seconds - (time.monotonic() - started)
        reason = ("cancelled" if cancel_check is not None and cancel_check() else
                  "maximum_seconds" if remaining <= 0 else
                  "maximum_evidence_bytes" if evidence_bytes >= maximum_evidence_bytes else None)
        if reason is not None:
            status = "incomplete"
            partial = InformationSetCompilationStopped(reason, stage="post_auction_handoff").partial_receipt
        else:
            play_result = resolve_quality_play_continuation_v1(rules, ExactTracePrefixV1.build(final_prefix), policy,
                maximum_nodes=maximum_nodes, maximum_depth=maximum_depth, maximum_worlds=maximum_worlds,
                maximum_seconds=maximum_seconds, runtime_maximum_seconds=remaining,
                maximum_decision_seconds=maximum_decision_seconds,
                maximum_evidence_bytes=maximum_evidence_bytes - evidence_bytes,
                private_recall=escalation.get("privateRecall"),
                escalation_events=escalation.get("events"), opportunity_order=opportunity_order,
                continue_recorded_prefix=continue_recorded_prefix and not escalation_prefix_only,
                stop_on_recorded_quality_failure=stop_on_recorded_quality_failure,
                recorded_trace=None if escalation_prefix_only else recorded_trace,
                proof_budget_profiles=profiles, search_budget_profiles=search_budget_profiles, cancel_check=cancel_check,
                checkpoint_callback=checkpoint_callback,
                final_normalized_callback=final_normalized_callback,
                execution_budget=execution_budget,
                featured_play_index=featured_play_index,
                featured_decision_point=featured_decision_point,
                featured_selector_dynamic=featured_selector_dynamic,
                _play_quality_cache=_play_quality_cache, _verified_artifacts=_verified_artifacts,
                **({"_admissible_play_actions": _admissible_play_actions}
                   if _admissible_play_actions is not None else {}),
                graph_workers=graph_workers)
            play_execution = play_result["executionReceipt"]
            play_execution["playStageMaximumSeconds"] = remaining
            play_execution["playStageStatus"] = "UNKNOWN" if play_result["status"] == "incomplete" else play_result["status"]
            play = {key: value for key, value in play_result.items() if key != "executionReceipt"}
            evidence_bytes += play_execution["evidenceBytes"]
            status, partial, final_prefix = play["status"], play["partialReceipt"], play["finalPrefix"]
    core = {"schemaVersion": 1, "kind": "native_quality_post_auction_continuation_v1",
        "scope": "post_auction_only", "status": status,
        "operation": "audit_then_generate" if continue_recorded_prefix else "generate" if recorded_trace is None else "audit_recorded",
        **({"submittedPrefix": recorded_trace.to_dict()} if continue_recorded_prefix else {}),
        "recordedPrefixHash": recorded_trace.sha256 if recorded_trace is not None else None,
        "rulesHash": rules.sha256, "sourcePrefixHash": prefix.sha256,
        "playingQuality": policy.to_dict(), "escalation": escalation, "projectPlay": play,
        "finalPrefix": final_prefix, "partialReceipt": partial,
        "limits": {"maximumSeconds": maximum_seconds, "maximumEvidenceBytes": maximum_evidence_bytes}}
    result = {**core, "sha256": canonical_sha256(core), "executionReceipt": {
        "elapsedSeconds": time.monotonic() - started, "evidenceBytes": evidence_bytes,
        "projectPlay": play_execution}}
    return result


def resolve_quality_round_continuation_v1(
    rules, prefix, decision_point, quality_policy, *, opportunity_order,
    maximum_nodes, maximum_depth, maximum_worlds, maximum_seconds,
    maximum_decision_seconds, maximum_escalation_decisions, maximum_evidence_bytes,
    recorded_events=None, recorded_trace=None, proof_budget_profiles=None, cancel_check=None,
    search_budget_profiles=None,
    execution_budget: ScenarioQualityExecutionBudgetV1 | None = None,
    runtime_maximum_seconds: float | None = None,
    continue_recorded_prefix: bool = False,
    stop_on_recorded_quality_failure: bool = False,
    checkpoint_callback: Callable[[Mapping[str, Any]], None] | None = None,
    final_normalized_callback: Callable[[OrToolsNormalizedExactStateV1], None] | None = None,
    featured_play_index: int | None = None,
    featured_decision_point: Mapping[str, Any] | None = None,
    featured_selector_dynamic: bool = False,
    search_generated: bool = False,
    _bid_quality_cache=None,
    _verified_artifacts: set[str] | None = None,
    _curated_focal_precheck: bool = False,
    graph_workers: int = 1,
    _admissible_play_actions: Callable[..., tuple[str, ...] | None] | None = None,
):
    """Compose native all-decision evidence for guarded whole-round admission."""
    if type(graph_workers) is not int or not 1 <= graph_workers <= 4:
        raise ValueError("graph_workers must be within the granted 1..4 worker ceiling")
    runtime_seconds = maximum_seconds if runtime_maximum_seconds is None else runtime_maximum_seconds
    if (type(runtime_seconds) not in {int, float} or not math.isfinite(runtime_seconds)
            or not 0 < runtime_seconds <= maximum_seconds):
        raise ValueError("runtime allowance must be positive and cannot exceed the declared budget")
    if not isinstance(prefix, ExactTracePrefixV1) or not isinstance(quality_policy, ScenarioPlayingQualityPolicyV1):
        raise TypeError("round quality requires an exact prefix and per-decision policy")
    if execution_budget is not None:
        if not isinstance(execution_budget, ScenarioQualityExecutionBudgetV1):
            raise TypeError("round quality execution budget must be typed")
        execution_budget = ScenarioQualityExecutionBudgetV1.build(execution_budget.to_dict())
        if (maximum_seconds > execution_budget.maximum_wall_seconds
                or maximum_evidence_bytes > execution_budget.maximum_evidence_bytes
                or maximum_escalation_decisions > execution_budget.maximum_escalation_decisions):
            raise ValueError("round quality request exceeds its frozen execution budget")
    if (prefix.auction_actions or prefix.escalation_actions or prefix.project_actions
            or prefix.project_declarations or prefix.play_actions or not prefix.project_actions_present):
        raise ValueError("round quality must start before bidding with explicit empty project actions")
    if (recorded_events is None) != (recorded_trace is None):
        raise ValueError("recorded round audit requires escalation events and a complete trace")
    if type(continue_recorded_prefix) is not bool:
        raise TypeError("continue_recorded_prefix must be boolean")
    if continue_recorded_prefix and recorded_trace is None:
        raise ValueError("round prefix continuation requires a recorded prefix")
    auction_prefix_only = bool(continue_recorded_prefix and not (
        recorded_trace.escalation_actions or recorded_trace.project_actions
        or recorded_trace.play_actions or recorded_events))
    policy = ScenarioPlayingQualityPolicyV1.build(quality_policy.to_dict())
    _, snapshots = _quality_budget_snapshots(
        {row["budgetProfile"] for row in policy.decisions.values()}, proof_budget_profiles,
        execution_budget=execution_budget)
    profiles = snapshots if proof_budget_profiles is not None else None
    auction_recorded = None
    if recorded_trace is not None:
        recorded_trace = _validate_quality_recorded_trace(prefix, recorded_trace,
            allow_escalation_extension=True, allow_auction_extension=True,
            allow_partial=continue_recorded_prefix)
        auction_raw = prefix.to_dict()
        auction_raw.pop("sha256")
        auction_raw["auctionActions"] = [dict(row) for row in recorded_trace.auction_actions]
        auction_raw["stateAssertions"] = dict(recorded_trace.state_assertions) if not (
            recorded_trace.play_actions or recorded_trace.project_actions or recorded_trace.escalation_actions) else {}
        auction_recorded = ExactTracePrefixV1.build(auction_raw)
    started = time.monotonic()
    limits = dict(maximum_nodes=maximum_nodes, maximum_depth=maximum_depth,
        maximum_worlds=maximum_worlds, maximum_seconds=maximum_seconds,
        maximum_decision_seconds=maximum_decision_seconds, proof_budget_profiles=profiles,
        cancel_check=cancel_check, checkpoint_callback=checkpoint_callback)
    bid_budget = ScenarioProofBudgetV1.build(policy.decisions["bid"]["budgetProfile"],
        snapshots[policy.decisions["bid"]["budgetProfile"]]) if snapshots else None
    bid_seconds = min(runtime_seconds / 2, bid_budget.maximum_wall_seconds, bid_budget.maximum_cpu_seconds) \
        if execution_budget is not None and execution_budget.exact_tail_proof and bid_budget is not None else runtime_seconds
    auction_result = resolve_quality_auction_continuation_v1(rules, prefix, policy,
        **({"search_budget_profiles": search_budget_profiles} if search_budget_profiles is not None else {}),
        maximum_evidence_bytes=maximum_evidence_bytes, recorded_trace=auction_recorded,
        continue_recorded_prefix=auction_prefix_only,
        stop_on_recorded_quality_failure=stop_on_recorded_quality_failure,
        opportunity_order=opportunity_order,
        runtime_maximum_seconds=bid_seconds, execution_budget=execution_budget,
        _bid_quality_cache=_bid_quality_cache,
        search_generated=search_generated, **limits)
    auction = {key: value for key, value in auction_result.items() if key != "executionReceipt"}
    used = auction_result["executionReceipt"]["evidenceBytes"]
    if checkpoint_callback is not None:
        checkpoint_callback({"schemaVersion": 1, "kind": "quality_stage_checkpoint_v1",
            "rulesHash": rules.sha256, "stage": "auction",
            "elapsedSeconds": time.monotonic() - started,
            "executionReceipt": auction_result["executionReceipt"]})
    status, partial, final_prefix = auction["status"], auction["partialReceipt"], auction["finalPrefix"]
    post, post_execution, suffix_audit = None, None, None
    if status == "complete" or (
        status == "quality_failed" and not stop_on_recorded_quality_failure
    ):
        if auction["auctionOutcome"]["allPassed"]:
            if recorded_trace is not None and not auction_prefix_only and (recorded_events or recorded_trace.to_dict() != final_prefix):
                raise ValueError("recorded all-pass round contains unconsumed actions or assertions")
        else:
            remaining = runtime_seconds - (time.monotonic() - started)
            reason = ("cancelled" if cancel_check is not None and cancel_check() else
                      "maximum_seconds" if remaining <= 0 else
                      "maximum_evidence_bytes" if used >= maximum_evidence_bytes else None)
            if reason:
                status = "incomplete"
                partial = InformationSetCompilationStopped(reason, stage="auction_quality_handoff").partial_receipt
            else:
                post_result = resolve_quality_post_auction_continuation_v1(
                    rules, ExactTracePrefixV1.build(final_prefix),
                    (decision_point if decision_point.get("kind") == "escalation"
                     else {"kind": "escalation", "selector": {}}), policy,
                    opportunity_order=opportunity_order, maximum_escalation_decisions=maximum_escalation_decisions,
                    maximum_evidence_bytes=maximum_evidence_bytes - used, runtime_maximum_seconds=remaining,
                    continue_recorded_prefix=continue_recorded_prefix and not auction_prefix_only,
                    stop_on_recorded_quality_failure=stop_on_recorded_quality_failure,
                    recorded_events=None if auction_prefix_only else recorded_events,
                    recorded_trace=None if auction_prefix_only else recorded_trace,
                    execution_budget=execution_budget,
                    **({"search_budget_profiles": search_budget_profiles} if search_budget_profiles is not None else {}),
                    featured_play_index=featured_play_index,
                    featured_decision_point=featured_decision_point,
                    featured_selector_dynamic=featured_selector_dynamic,
                    final_normalized_callback=final_normalized_callback,
                    _play_quality_cache=_bid_quality_cache, _verified_artifacts=_verified_artifacts,
                    _curated_focal_precheck=_curated_focal_precheck,
                    **({"_admissible_play_actions": _admissible_play_actions}
                       if _admissible_play_actions is not None else {}),
                    graph_workers=graph_workers, **limits)
                post_execution = post_result["executionReceipt"]
                post = {key: value for key, value in post_result.items() if key != "executionReceipt"}
                used += post_execution["evidenceBytes"]
                status, partial, final_prefix = post["status"], post["partialReceipt"], post["finalPrefix"]
                if status == "complete" and auction["status"] == "quality_failed":
                    status = "quality_failed"
                    partial = auction["partialReceipt"]
    if (auction["status"] == "incomplete" and recorded_trace is not None
            and featured_play_index is not None and len(recorded_trace.play_actions) >= featured_play_index
            and execution_budget is not None and execution_budget.exact_tail_proof
            and not (cancel_check and cancel_check())):
        remaining = runtime_seconds - (time.monotonic() - started)
        if remaining > 0 and used < maximum_evidence_bytes:
            saved = recorded_trace.to_dict()
            saved.pop("sha256", None)
            saved["stateAssertions"] = {}
            saved["playActions"] = saved["playActions"][:featured_play_index]
            saved = ExactTracePrefixV1.build(saved)
            if checkpoint_callback is not None:
                checkpoint_callback({"kind": "quality_stage_checkpoint_v1", "stage": "play_suffix_audit",
                    "bidStageStatus": "UNKNOWN", "proofStartIndex": featured_play_index})
            # Suffix evidence cannot admit the round with an unproved auction.
            try:
                suffix_audit = resolve_quality_play_continuation_v1(rules, saved, policy,
                    graph_workers=graph_workers,
                    maximum_nodes=maximum_nodes, maximum_depth=maximum_depth, maximum_worlds=maximum_worlds,
                    maximum_seconds=maximum_seconds, runtime_maximum_seconds=remaining,
                    maximum_decision_seconds=maximum_decision_seconds,
                    maximum_evidence_bytes=maximum_evidence_bytes - used,
                    proof_budget_profiles=profiles, search_budget_profiles=search_budget_profiles,
                    execution_budget=execution_budget, escalation_events=recorded_events,
                    opportunity_order=opportunity_order, cancel_check=cancel_check,
                    featured_play_index=featured_play_index, featured_decision_point=featured_decision_point,
                    featured_selector_dynamic=featured_selector_dynamic,
                    checkpoint_callback=checkpoint_callback,
                    _play_quality_cache=_bid_quality_cache)
            except (InformationSetCompilationStopped, ExactNormalizationStopped) as stopped:
                suffix_audit = {"status": "incomplete", "partialReceipt": stopped.partial_receipt}
            used += suffix_audit.get("executionReceipt", {}).get("evidenceBytes", 0)
    core = {"schemaVersion": 1, "kind": "native_quality_round_continuation_v1",
        "scope": "all_decisions", "status": status, "rulesHash": rules.sha256,
        "sourcePrefixHash": prefix.sha256, "playingQuality": policy.to_dict(),
        "operation": "audit_then_generate" if continue_recorded_prefix else "generate" if recorded_trace is None else "audit_recorded",
        **({"submittedPrefix": recorded_trace.to_dict()} if continue_recorded_prefix else {}),
        "recordedPrefixHash": recorded_trace.sha256 if recorded_trace is not None else None,
        "auction": auction, "postAuction": post, "finalPrefix": final_prefix, "partialReceipt": partial,
        "limits": {"maximumSeconds": maximum_seconds, "maximumEvidenceBytes": maximum_evidence_bytes}}
    if execution_budget is not None:
        core["qualityExecution"] = execution_budget.to_dict()
    result = {**core, "sha256": canonical_sha256(core), "executionReceipt": {
        "elapsedSeconds": time.monotonic() - started, "evidenceBytes": used,
        "auction": auction_result["executionReceipt"], "postAuction": post_execution,
        "bidStageStatus": "UNKNOWN" if auction["status"] == "incomplete" else auction["status"],
        "bidStageMaximumSeconds": bid_seconds,
        **({"playSuffixAudit": suffix_audit} if suffix_audit is not None else {})}}
    return result


def _policy_terminal_utility_v1(
    score: Mapping[str, Any],
    request: ScenarioProofRequestV1,
) -> int:
    target = Team[str(request.target_team).upper()]
    if request.utility == "awarded_point_margin":
        awarded = tuple(int(value) for value in score["awardedPoints"])
        return awarded[int(target)] - awarded[int(target.other)]
    if request.utility in {"contract_success", "round_win"}:
        loss = 0 if request.utility == "contract_success" and request.utility_encoding is None else -1
        return 1 if score["roundWinner"] == target.name.lower() else loss
    if request.utility == "match_win":
        winner = score.get("matchWinner")
        return 0 if winner is None else (1 if winner == target.name.lower() else -1)
    if request.utility == "kabout":
        return (
            1
            if bool(score["kabout"]) and score["roundWinner"] == target.name.lower()
            else -1
        )
    raise _PolicyContinuationCompilerDisagreement(
        "policy continuation uses an unsupported terminal utility"
    )


def _policy_play_principal_variation_v1(
    graph: ProofGraphV1,
    proof: OrToolsGraphProofOutcomeV1,
) -> tuple[tuple[dict[str, Any], ...], int]:
    graph.validate()
    if proof.graph_hash != graph.sha256:
        raise _PolicyContinuationCompilerDisagreement(
            "policy proof graph authority does not match the retained graph"
        )
    nodes = {node.node_id: node for node in graph.nodes}
    node = nodes[graph.root_node_id]
    rows: list[dict[str, Any]] = []
    for code in proof.principal_variation:
        if node.is_leaf or node.actor_seat is None:
            raise _PolicyContinuationCompilerDisagreement(
                "policy principal variation has no typed actor path"
            )
        matches = [edge for edge in node.edges if edge.action_code == code]
        if len(matches) != 1:
            raise _PolicyContinuationCompilerDisagreement(
                "policy principal variation disagrees with the proof graph"
            )
        try:
            action = LegalPlayActionV1.from_code(code)
        except ValueError as exc:
            raise _PolicyContinuationCompilerDisagreement(
                "policy principal variation contains a non-play action"
            ) from exc
        rows.append(
            {
                "actor": node.actor_seat,
                "card": action.card.code,
                "ekka": action.ekka,
            }
        )
        node = nodes[matches[0].child_node_id]
    if not node.is_leaf or node.leaf_kind != "terminal":
        raise _PolicyContinuationCompilerDisagreement(
            "policy principal variation does not reach a terminal graph node"
        )
    assert node.utility is not None
    return tuple(rows), node.utility


def _play_action_identity_v1(value: Mapping[str, Any]) -> tuple[str, str, bool]:
    return (
        Seat.parse(value["actor"]).name.lower(),
        Card.parse(value["card"]).code,
        bool(value.get("ekka", False)),
    )


def _materialize_play_policy_continuation_v1(
    *,
    rules: RuleBook,
    candidate: OrToolsNeutralCandidateV1,
    source_normalization: Any,
    graph: ProofGraphV1,
    proof: OrToolsGraphProofOutcomeV1,
    request: ScenarioProofRequestV1,
    maximum_seconds: float,
    native_cp_sat: bool = False,
    cancel_check: Callable[[], bool] | None = None,
) -> OrToolsNeutralCandidateV1:
    """Bind one complete-state proof PV to one immutable exact decision prefix."""

    if "playingQualityEvidence" in candidate.trace:
        raise _PolicyContinuationCompilerDisagreement(
            "audited full rounds cannot be rematerialized"
        )

    if request.mode == "policy_conditioned":
        continuation_key = "policyContinuation"
        continuation_kind = "play_policy_principal_variation_continuation_v1"
    elif request.mode == "exact" and request.visibility == "complete_state":
        continuation_key = "proofContinuation"
        continuation_kind = "play_exact_principal_variation_continuation_v1"
    else:
        raise _PolicyContinuationCompilerDisagreement(
            "play continuation materialization requires a complete-state proof"
        )

    raw_prefix = candidate.trace.get("exactTracePrefix")
    raw_source = candidate.trace.get("exactNormalization")
    if not isinstance(raw_prefix, Mapping) or not isinstance(raw_source, Mapping):
        raise _PolicyContinuationCompilerDisagreement(
            "play policy proof lacks an exact decision prefix"
        )
    prefix = ExactTracePrefixV1.build(raw_prefix)
    if (
        candidate.exact_state_id != source_normalization.exact_state_id
        or candidate.state != source_normalization.state
        or raw_source != source_normalization.to_dict()
        or raw_source.get("exactStateId") != source_normalization.exact_state_id
        or raw_source.get("sha256") != source_normalization.sha256
    ):
        raise _PolicyContinuationCompilerDisagreement(
            "play policy proof source state authority mismatch"
        )
    suffix, graph_utility = _policy_play_principal_variation_v1(graph, proof)
    if len(prefix.play_actions) + len(suffix) != 32:
        raise _PolicyContinuationCompilerDisagreement(
            "policy principal variation does not complete exactly 32 plays"
        )

    retained_play = candidate.trace.get("play")
    retained_rows = (
        list(retained_play.get("plays") or ())
        if isinstance(retained_play, Mapping)
        else []
    )
    construction_play_hash = (
        canonical_sha256(retained_play) if len(retained_rows) == 32 else None
    )
    construction_score = candidate.trace.get("score")
    construction_score_hash = (
        canonical_sha256(construction_score)
        if construction_play_hash is not None
        and isinstance(construction_score, Mapping)
        else None
    )

    complete_document = prefix.to_dict()
    complete_document.pop("sha256", None)
    complete_document["playActions"] = [
        *[dict(item) for item in prefix.play_actions],
        *[dict(item) for item in suffix],
    ]
    complete_document["stateAssertions"] = {}
    complete_prefix = ExactTracePrefixV1.build(complete_document)
    try:
        complete = OrToolsExactStateNormalizerV1(
            rules, native_cp_sat=native_cp_sat
        ).normalize(
            complete_prefix,
            decision_point={"kind": "play"},
            maximum_seconds=maximum_seconds,
            cancel_check=cancel_check,
        )
    except ValueError as exc:
        raise _PolicyContinuationCompilerDisagreement(
            "policy principal variation is not a legal exact continuation"
        ) from exc
    complete_receipt = complete.normalization_receipt.to_dict()
    complete_play = complete_receipt.get("derivedPlay")
    if (
        complete.phase != "scored"
        or complete.state.get("phase") != "scored"
        or len(complete.state.get("completedTricks", ())) != 8
        or not isinstance(complete_play, Mapping)
        or dict(complete_play.get("decisionState") or {}).get("prefixLength") != 32
    ):
        raise _PolicyContinuationCompilerDisagreement(
            "policy continuation did not normalize to eight scored tricks"
        )
    if native_cp_sat:
        # Keep authored actor assertions until the native solve validates them.
        complete_document["playActions"] = [dict(item) for item in complete_play["plays"]]
        complete_prefix = ExactTracePrefixV1.build(complete_document)
    score = _terminal_score_from_state_v1(
        complete.state,
        match_target=ScoringDomainIRV1.from_rules(rules.data).match_target,
    )
    if graph_utility != _policy_terminal_utility_v1(score, request):
        raise _PolicyContinuationCompilerDisagreement(
            "policy proof utility disagrees with the normalized terminal score"
        )

    pv_core = {
        "schemaVersion": 1,
        "actions": list(proof.principal_variation),
    }
    source_hash = canonical_sha256(raw_source)
    complete_normalization = complete.to_dict()
    continuation_core = {
        "schemaVersion": 1,
        "kind": continuation_kind,
        "proofOutcomeHash": proof.sha256,
        "graphHash": graph.sha256,
        "principalVariation": list(proof.principal_variation),
        "principalVariationHash": canonical_sha256(pv_core),
        "sourceExactStateId": source_normalization.exact_state_id,
        "sourceExactNormalizationHash": source_hash,
        "sourceConstructionPlayHash": construction_play_hash,
        "sourceConstructionScoreHash": construction_score_hash,
        "completeExactTracePrefixHash": complete_prefix.sha256,
        "completeExactTracePrefixDocumentHash": canonical_sha256(
            complete_prefix.to_dict()
        ),
        "completeExactNormalizationHash": canonical_sha256(complete_normalization),
        "completeExactStateId": complete.exact_state_id,
        "completeExactStateHash": complete.state_hash,
        "terminalScore": score,
        "terminalScoreHash": canonical_sha256(score),
        "policyIdentities": proof.to_dict()["policyIdentities"],
        "playable": True,
        "completionStage": "scored",
    }
    continuation = {
        **continuation_core,
        "sha256": canonical_sha256(continuation_core),
    }
    scoring_receipt = dict(complete_receipt.get("derivedScoring") or {})
    proof_size = {
        "variableCount": proof.variable_count,
        "constraintCount": proof.constraint_count,
    }
    receipts = {
        **candidate.stage_receipts,
        "playPrefix": _stage_receipt(
            "playPrefix",
            model_hash=str(complete_play["modelHash"]),
            evidence_hash=str(complete_play["evidenceHash"]),
            semantic=complete_play,
            solver_status=str(complete_play["solverStatus"]),
        ),
        "outcome": _stage_receipt(
            "outcome",
            model_hash=str(complete_receipt["modelHash"]),
            evidence_hash=canonical_sha256(score),
            semantic=score,
            solver_status=str(complete_receipt["solverStatus"]),
            variable_count=int(scoring_receipt.get("variableCount", 0)),
            constraint_count=int(scoring_receipt.get("constraintCount", 0)),
        ),
        "completeExactNormalization": _stage_receipt(
            "completeExactNormalization",
            model_hash=str(complete_receipt["modelHash"]),
            evidence_hash=canonical_sha256(complete_normalization),
            semantic=complete_normalization,
            solver_status=str(complete_receipt["solverStatus"]),
            variable_count=int(complete_receipt.get("variableCount", 0)),
            constraint_count=int(complete_receipt.get("constraintCount", 0)),
        ),
        continuation_key: _stage_receipt(
            continuation_key,
            model_hash=proof.model_hash,
            evidence_hash=str(continuation["sha256"]),
            semantic=continuation,
            solver_status=proof.solver_status,
            variable_count=int(proof_size["variableCount"]),
            constraint_count=int(proof_size["constraintCount"]),
        ),
    }
    trace = {
        **candidate.trace,
        "score": score,
        **(
            {"constructionScore": copy.deepcopy(dict(construction_score))}
            if construction_score_hash is not None
            else {}
        ),
        "completePlay": copy.deepcopy(dict(complete_play)),
        continuation_key: continuation,
        "completeExactTracePrefix": complete_prefix.to_dict(),
        "completeExactNormalization": complete_normalization,
    }
    return OrToolsNeutralCandidateV1(
        candidate.origin,
        trace,
        candidate.state,
        candidate.no_good_vector,
        receipts,
        candidate.exact_state_id,
    )


def _validate_play_policy_continuation_binding_v1(
    candidate: OrToolsNeutralCandidateV1,
    *,
    graph: ProofGraphV1 | None,
    proof: OrToolsGraphProofOutcomeV1 | None,
) -> None:
    continuation = candidate.trace.get("policyContinuation")
    continuation_key = "policyContinuation"
    if not isinstance(continuation, Mapping):
        continuation = candidate.trace.get("proofContinuation")
        continuation_key = "proofContinuation"
    expected_kind = {
        "policyContinuation": "play_policy_principal_variation_continuation_v1",
        "proofContinuation": "play_exact_principal_variation_continuation_v1",
    }[continuation_key]
    if (
        not isinstance(continuation, Mapping)
        or continuation.get("kind") != expected_kind
    ):
        return
    if graph is None or proof is None:
        raise ValueError("play policy continuation lacks bound proof artifacts")
    core = {key: copy.deepcopy(value) for key, value in continuation.items()}
    declared_hash = core.pop("sha256", None)
    if declared_hash != canonical_sha256(core):
        raise ValueError("play policy continuation checksum mismatch")
    if (
        continuation.get("proofOutcomeHash") != proof.sha256
        or continuation.get("graphHash") != graph.sha256
        or continuation.get("principalVariation") != list(proof.principal_variation)
    ):
        raise ValueError("play policy continuation proof binding mismatch")
    expected_suffix, _ = _policy_play_principal_variation_v1(graph, proof)
    expected_pv_hash = canonical_sha256(
        {"schemaVersion": 1, "actions": list(proof.principal_variation)}
    )
    if continuation.get("principalVariationHash") != expected_pv_hash:
        raise ValueError("play policy continuation PV hash mismatch")

    construction_play_hash = continuation.get("sourceConstructionPlayHash")
    if construction_play_hash is not None:
        construction_play = candidate.trace.get("play")
        construction_score = candidate.trace.get("constructionScore")
        if (
            not isinstance(construction_play, Mapping)
            or len(construction_play.get("plays", ())) != 32
            or canonical_sha256(construction_play) != construction_play_hash
            or not isinstance(construction_score, Mapping)
            or canonical_sha256(construction_score)
            != continuation.get("sourceConstructionScoreHash")
        ):
            raise ValueError("play policy construction witness binding mismatch")

    source = candidate.trace.get("exactNormalization")
    complete_prefix_raw = candidate.trace.get("completeExactTracePrefix")
    complete = candidate.trace.get("completeExactNormalization")
    complete_play = candidate.trace.get("completePlay")
    if not all(
        isinstance(item, Mapping)
        for item in (source, complete_prefix_raw, complete, complete_play)
    ):
        raise ValueError("play policy continuation artifacts are incomplete")
    complete_prefix = ExactTracePrefixV1.build(complete_prefix_raw)
    if len(complete_prefix.play_actions) != 32:
        raise ValueError("play policy continuation is not a complete round")
    source_prefix = ExactTracePrefixV1.build(candidate.trace["exactTracePrefix"])
    if tuple(
        _play_action_identity_v1(item)
        for item in complete_prefix.play_actions[len(source_prefix.play_actions) :]
    ) != tuple(_play_action_identity_v1(item) for item in expected_suffix):
        raise ValueError("play policy continuation suffix mismatch")
    score = candidate.trace.get("score")
    if not isinstance(score, Mapping):
        raise ValueError("play policy continuation terminal score is missing")
    complete_state = complete.get("state")
    if not isinstance(
        complete_state, Mapping
    ) or score != _terminal_score_from_state_v1(
        complete_state,
        match_target=int(score["matchTarget"]),
    ):
        raise ValueError("play policy continuation terminal score mismatch")
    if (
        source.get("exactStateId") != candidate.exact_state_id
        or source.get("state") != candidate.state
        or continuation.get("sourceExactStateId") != candidate.exact_state_id
        or continuation.get("sourceExactNormalizationHash") != canonical_sha256(source)
        or continuation.get("completeExactTracePrefixHash") != complete_prefix.sha256
        or continuation.get("completeExactTracePrefixDocumentHash")
        != canonical_sha256(complete_prefix.to_dict())
        or continuation.get("completeExactNormalizationHash")
        != canonical_sha256(complete)
        or continuation.get("completeExactStateId") != complete.get("exactStateId")
        or continuation.get("completeExactStateHash") != complete.get("stateHash")
        or continuation.get("terminalScore") != score
        or continuation.get("terminalScoreHash") != canonical_sha256(score)
    ):
        raise ValueError("play policy continuation exact-state binding mismatch")
    play_rows = complete_play.get("plays")
    if (
        complete.get("phase") != "scored"
        or complete_state.get("phase") != "scored"
        or len(complete_state.get("completedTricks", ())) != 8
        or not isinstance(play_rows, Sequence)
        or len(play_rows) != 32
        or tuple(_play_action_identity_v1(item) for item in play_rows)
        != tuple(
            _play_action_identity_v1(item) for item in complete_prefix.play_actions
        )
    ):
        raise ValueError("play policy continuation terminal trace mismatch")
    continuation_receipt = candidate.stage_receipts.get(continuation_key)
    play_receipt = candidate.stage_receipts.get("playPrefix")
    outcome_receipt = candidate.stage_receipts.get("outcome")
    if (
        continuation_receipt is None
        or continuation_receipt.evidence_hash != declared_hash
        or continuation_receipt.semantic_hash != canonical_sha256(continuation)
        or play_receipt is None
        or play_receipt.semantic_hash != canonical_sha256(complete_play)
        or outcome_receipt is None
        or outcome_receipt.semantic_hash != canonical_sha256(score)
    ):
        raise ValueError("play policy continuation stage receipt mismatch")


def _validate_bid_decision_prefix_v1(
    prefix: ExactTracePrefixV1,
    decision_point: Mapping[str, object],
) -> None:
    if decision_point.get("kind") != "bid":
        return
    selector = decision_point.get("selector")
    if not isinstance(selector, Mapping):
        raise ValueError("bid decision requires a selector")
    action_index = selector.get("actionIndex")
    if type(action_index) is not int or action_index < 0:
        raise ValueError("bid selector.actionIndex must be a non-negative integer")
    if action_index != len(prefix.auction_actions):
        raise ValueError(
            "bid selector.actionIndex must equal the auction prefix length"
        )


def _rebind_search_candidate_to_decision(
    candidate: OrToolsNeutralCandidateV1,
    prefix: ExactTracePrefixV1,
    normalized: Any,
    *,
    preserve_construction_outcome: bool = False,
    native_cp_sat: bool = False,
) -> OrToolsNeutralCandidateV1:
    """Replace generated terminal-stage evidence with the selected decision prefix."""

    receipt = normalized.normalization_receipt.to_dict()
    derived_auction = dict(receipt["derivedAuction"])
    derived_escalation = receipt["derivedEscalation"]
    derived_projects = receipt["derivedProjects"]
    derived_play = receipt["derivedPlay"]
    replaced_stages = {"auction", "escalation", "projects", "playPrefix", "state"}
    if not preserve_construction_outcome:
        replaced_stages.add("outcome")
    receipts = {
        key: value
        for key, value in candidate.stage_receipts.items()
        if key not in replaced_stages
    }
    receipts.update(
        {
            "auction": _stage_receipt(
                "auction",
                model_hash=receipt["modelHash"],
                evidence_hash=receipt["solutionHash"],
                semantic=derived_auction,
                solver_status=receipt["solverStatus"],
                variable_count=int(receipt.get("variableCount", 0)),
                constraint_count=int(receipt.get("constraintCount", 0)),
            ),
            "state": _stage_receipt(
                "state",
                model_hash=receipt["modelHash"],
                evidence_hash=normalized.state_hash,
                semantic={
                    **normalized.state,
                    "variableCount": receipt.get("variableCount", 0),
                    "constraintCount": receipt.get("constraintCount", 0),
                },
                solver_status=receipt["solverStatus"],
            ),
            "exactNormalization": _stage_receipt(
                "exactNormalization",
                model_hash=receipt["modelHash"],
                evidence_hash=normalized.sha256,
                semantic={
                    **normalized.to_dict(),
                    "variableCount": receipt.get("variableCount", 0),
                    "constraintCount": receipt.get("constraintCount", 0),
                },
                solver_status=receipt["solverStatus"],
            ),
        }
    )
    if derived_escalation is not None:
        receipts["escalation"] = _stage_receipt(
            "escalation",
            model_hash=derived_escalation["modelHash"],
            evidence_hash=derived_escalation["evidenceHash"],
            semantic=derived_escalation,
            solver_status="optimal",
        )
    if derived_projects is not None:
        receipts["projects"] = _stage_receipt(
            "projects",
            model_hash=derived_projects["modelHash"],
            evidence_hash=derived_projects["evidenceHash"],
            semantic=derived_projects,
            solver_status=derived_projects["solverStatus"],
            branches=derived_projects["branches"],
            conflicts=derived_projects["conflicts"],
        )
    if derived_play is not None:
        receipts["playPrefix"] = _stage_receipt(
            "playPrefix",
            model_hash=derived_play["modelHash"],
            evidence_hash=derived_play["evidenceHash"],
            semantic=derived_play,
            solver_status=derived_play["solverStatus"],
        )
    trace = {
        **candidate.trace,
        "constructionPlay": candidate.trace.get("play"),
        "constructionScore": candidate.trace.get("score"),
        **({"constructionEscalation": candidate.trace.get("escalation")} if native_cp_sat else {}),
        "exactTracePrefix": prefix.to_dict(),
        "exactNormalization": normalized.to_dict(),
        "normalizationReceipt": receipt,
        "dealer": normalized.state["dealer"],
        "buyer": normalized.state.get("buyer"),
        "contract": normalized.state.get("contract"),
        "trumpSuit": normalized.state.get("trumpSuit"),
        "auction": {"prefix": derived_auction},
        "escalation": derived_escalation,
        "projects": derived_projects,
        "play": derived_play,
    }
    return OrToolsNeutralCandidateV1(
        candidate.origin,
        trace,
        normalized.state,
        candidate.no_good_vector,
        receipts,
        normalized.exact_state_id,
    )


def _validated_candidate_partition_v1(
    value: tuple[int, int, int] | None,
) -> tuple[int, int, int] | None:
    if value is None:
        return None
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or any(type(item) is not int for item in value)
    ):
        raise ValueError("candidate partition must be a three-integer tuple")
    modulus, residue_start, residue_count = value
    if (
        modulus < 1
        or residue_start < 0
        or residue_count < 1
        or residue_start + residue_count > modulus
    ):
        raise ValueError("candidate partition range is outside its modulus")
    return value


def _apply_candidate_partition_v1(
    model: cp_model.CpModel,
    variables: Sequence[cp_model.IntVar],
    partition: tuple[int, int, int] | None,
    *,
    name: str,
) -> None:
    partition = _validated_candidate_partition_v1(partition)
    if partition is None or partition[0] == 1 or partition[2] == partition[0]:
        return
    selected = tuple(variables[:12])
    if not selected:
        raise ValueError("candidate partition requires at least one finite variable")
    modulus, residue_start, residue_count = partition
    residue = model.new_int_var(0, modulus - 1, f"{name}_partition_residue")
    model.add_modulo_equality(
        residue,
        sum(variable * (index + 1) for index, variable in enumerate(selected)),
        modulus,
    )
    model.add(residue >= residue_start)
    model.add(residue < residue_start + residue_count)


class GenericCompleteTraceCompilerV1:
    """Deterministically construct one complete legal round per no-good epoch."""

    def __init__(
        self,
        *,
        rules: RuleBook,
        dealer: Seat | str | int = Seat.EAST,
        match_score: tuple[int, int] = (0, 0),
        minimum_deck_hamming_distance: int = 1,
        excluded_decision_status_vectors: Sequence[Sequence[int]] = (),
        minimum_decision_status_hamming_distance: int = 1,
        candidate_partition: tuple[int, int, int] | None = None,
        native_cp_sat_scoring: bool = False,
        optimal_auction_quality: bool = False,
        structural_only: bool = False,
        play_model_configurator: Callable[..., None] | None = None,
        play_model_refiner: Callable[..., Any] | None = None,
        auction_model_refiner: Callable[..., Any] | None = None,
        canonicalize_deal_packets: bool = False,
        manifest: (
            ScenarioPredicateCatalogLoweringManifestV2
            | ScenarioPredicateCatalogLoweringManifestV3
            | None
        ) = None,
    ) -> None:
        if not isinstance(rules, RuleBook):
            raise TypeError("complete-trace compiler requires RuleBook")
        self.rules = rules
        self.dealer = Seat.parse(dealer)
        self.match_score = match_score
        if (
            type(minimum_deck_hamming_distance) is not int
            or not 1 <= minimum_deck_hamming_distance <= 32
        ):
            raise ValueError("deck Hamming distance must be in 1..32")
        self.minimum_deck_hamming_distance = minimum_deck_hamming_distance
        self.excluded_decision_status_vectors = tuple(
            tuple(vector) for vector in excluded_decision_status_vectors
        )
        if any(
            len(vector) != 32
            or any(type(status) is not int or not 0 <= status <= 9 for status in vector)
            for vector in self.excluded_decision_status_vectors
        ):
            raise ValueError("decision-state cuts must contain 32 statuses in 0..9")
        if (
            type(minimum_decision_status_hamming_distance) is not int
            or not 1 <= minimum_decision_status_hamming_distance <= 32
        ):
            raise ValueError("decision-state Hamming distance must be in 1..32")
        self.minimum_decision_status_hamming_distance = (
            minimum_decision_status_hamming_distance
        )
        self.candidate_partition = _validated_candidate_partition_v1(
            candidate_partition
        )
        self.native_cp_sat_scoring = bool(native_cp_sat_scoring)
        self.optimal_auction_quality = bool(optimal_auction_quality)
        self.structural_only = bool(structural_only)
        if play_model_configurator is not None and not callable(play_model_configurator):
            raise TypeError("play_model_configurator must be callable")
        self.play_model_configurator = play_model_configurator
        if play_model_refiner is not None and (not callable(play_model_refiner) or not self.native_cp_sat_scoring):
            raise TypeError("play_model_refiner requires native scoring and a callable")
        self.play_model_refiner = play_model_refiner
        if auction_model_refiner is not None and (not callable(auction_model_refiner) or not self.native_cp_sat_scoring):
            raise TypeError("auction_model_refiner requires native scoring and a callable")
        self.auction_model_refiner = auction_model_refiner
        if type(canonicalize_deal_packets) is not bool:
            raise TypeError("deal-packet canonicalization must be boolean")
        self.canonicalize_deal_packets = canonicalize_deal_packets
        if play_model_refiner is not None and self.excluded_decision_status_vectors:
            raise ValueError("policy refinement cannot use the separate diversity-model rebuild path")
        self.manifest = manifest or shared_scenario_predicate_catalog_lowering_manifest_v2()

    def __call__(
        self,
        scenario: ResearchScenarioV1,
        excluded_decks: Sequence[Sequence[int]],
        *,
        seed: int,
        maximum_seconds: float,
        maximum_conflicts: int = 250_000,
    ) -> OrToolsCandidateAttemptV1:
        manifest = self.manifest
        candidate_deadline = time.monotonic() + maximum_seconds

        def remaining() -> float:
            return candidate_deadline - time.monotonic()

        deal_predicates = tuple(
            predicate
            for predicate in scenario.predicates()
            if predicate.hardness == "hard"
            and (definition := manifest.definition(predicate.kind)) is not None
            and definition.stage == "deal"
            and definition.cp_sat_support.qualification_source == "construction"
        )
        project_stage_predicates = tuple(
            predicate
            for predicate in scenario.predicates()
            if predicate.hardness == "hard"
            and (definition := manifest.definition(predicate.kind)) is not None
            and definition.stage == "projects"
            and definition.cp_sat_support.qualification_source == "construction"
        )
        project_tautology_ids = (
            frozenset()
            if self.native_cp_sat_scoring
            else frozenset(
                predicate.predicate_id
                for predicate in project_stage_predicates
                if _project_predicate_is_tautology(predicate, self.rules)
            )
        )
        project_predicates = tuple(
            predicate
            for predicate in project_stage_predicates
            if predicate.predicate_id not in project_tautology_ids
        )
        project_eligibility_only = (
            self.structural_only
            and self.native_cp_sat_scoring
            and isinstance(manifest, ScenarioPredicateCatalogLoweringManifestV3)
            and all(
                predicate.kind in PROJECT_PREDICATE_KINDS_V2
                and predicate.kind != "project.baloot"
                and (predicate.kind == "project.overlap_count"
                     or predicate.arguments.get("state", "eligible") == "eligible")
                for predicate in project_predicates
            )
        )
        project_tautologies = tuple(
            predicate
            for predicate in project_stage_predicates
            if predicate.predicate_id in project_tautology_ids
        )
        native_project_opening = (
            self.native_cp_sat_scoring
            and self.rules.project_domain.tie_break_order == "second_trick_reveal"
            and (not self.structural_only or (
                bool(project_predicates) and not project_eligibility_only
            ))
        )
        escalation_predicates = tuple(
            predicate
            for predicate in scenario.predicates()
            if predicate.hardness == "hard"
            and (definition := manifest.definition(predicate.kind)) is not None
            and definition.stage == "escalation"
            and definition.cp_sat_support.qualification_source == "construction"
        )
        preflight_seconds = remaining()
        if preflight_seconds <= 0:
            return OrToolsCandidateAttemptV1(
                "incomplete", None, "deal_auction_project_budget"
            )
        if _project_count_projection_is_infeasible(
            scenario,
            project_stage_predicates,
            self.rules,
            maximum_conflicts=maximum_conflicts,
            maximum_seconds=preflight_seconds,
        ):
            return OrToolsCandidateAttemptV1(
                "infeasible", None, "deal_auction_project_infeasible"
            )
        if remaining() <= 0:
            return OrToolsCandidateAttemptV1(
                "incomplete", None, "deal_auction_project_budget"
            )
        auction_predicates = tuple(
            predicate
            for predicate in scenario.predicates()
            if predicate.hardness == "hard"
            and getattr(manifest.definition(predicate.kind), "stage", None) == "auction"
        )
        bid_zero_score_predicates = tuple(
            predicate
            for predicate in scenario.predicates()
            if self.native_cp_sat_scoring
            and scenario.decision_point["kind"] == "bid"
            and predicate.hardness == "hard"
            and predicate.kind in {
                "state.current_raw_points",
                "state.current_project_points",
                "state.current_game_points",
            }
        )
        owned_play_attempt_predicates = tuple(
            predicate
            for predicate in scenario.predicates()
            if predicate.hardness == "hard"
            and predicate.kind
            in {OUT_OF_TURN_PREDICATE_V3, FALSE_EKKA_PREDICATE_V3}
        )
        auction_view = (
            "prefix" if scenario.decision_point["kind"] == "bid" else "completion"
        )
        auction_prefix_length = (
            int(dict(scenario.decision_point["selector"]).get("actionIndex", 0))
            if auction_view == "prefix"
            else None
        )
        contracts = _construction_contracts(scenario)
        strategic_escalation_pairs: tuple[tuple[int, int], ...] | None = None
        if scenario.decision_point["kind"] == "escalation":
            actor_raw = dict(scenario.decision_point["selector"]).get("actingSeat")
            if actor_raw is None:
                raise ValueError("escalation search requires selector.actingSeat")
            actor = Seat.parse(actor_raw)
            escalation_domain = EscalationDomainIRV1.from_rules(self.rules.data)
            if self.native_cp_sat_scoring:
                pairs = []
                for declarer in Seat:
                    for contract in contracts:
                        stage_seconds = remaining()
                        if stage_seconds <= 0:
                            return OrToolsCandidateAttemptV1(
                                "incomplete",
                                None,
                                "escalation_decision_structurally_incomplete",
                            )
                        result = OrToolsEscalationStageV1(
                            escalation_domain,
                            declarer=declarer,
                            contract=contract,
                            match_score=self.match_score,
                            native_cp_sat=True,
                        ).solve(
                            required_actions_complete=True,
                            required_eligible_seat=actor,
                            seed=seed,
                            maximum_seconds=stage_seconds,
                            maximum_conflicts=maximum_conflicts,
                        )
                        if result.status == "incomplete":
                            return OrToolsCandidateAttemptV1(
                                "incomplete",
                                None,
                                "escalation_decision_structurally_incomplete",
                            )
                        if result.status == "satisfied":
                            pairs.append((int(declarer), _CONTRACT_IDS[contract]))
                strategic_escalation_pairs = tuple(pairs)
            else:
                strategic_escalation_pairs = tuple(
                    (int(declarer), _CONTRACT_IDS[contract])
                    for declarer in Seat
                    for contract in contracts
                    if escalation_domain.legal_decisions(
                        escalation_domain.initial_state(),
                        declarer=declarer,
                        contract=contract,
                        match_score=self.match_score,
                        actors=(actor,),
                        include_closure=False,
                    )
                )
            if not strategic_escalation_pairs:
                return OrToolsCandidateAttemptV1(
                    "infeasible",
                    None,
                    "escalation_decision_structurally_infeasible",
                )
        project_witness_predicate = None
        for predicate in project_predicates:
            if predicate.kind != "project.exists":
                continue
            preflight_seconds = remaining()
            if preflight_seconds <= 0:
                return OrToolsCandidateAttemptV1(
                    "incomplete", None, "deal_auction_project_budget"
                )
            if _hard_expression_requires_predicate(
                scenario,
                predicate.predicate_id,
                maximum_conflicts=maximum_conflicts,
                maximum_seconds=preflight_seconds,
            ):
                project_witness_predicate = predicate
                break
        if remaining() <= 0:
            return OrToolsCandidateAttemptV1(
                "incomplete", None, "deal_auction_project_budget"
            )
        base_construction_choices = (
            tuple(
                (contract, trump)
                for contract in contracts
                for trump in (
                    tuple(Suit) if contract.effective == Contract.HOKUM else (None,)
                )
            )
            if project_predicates or native_project_opening
            else tuple((contract, None) for contract in contracts)
            if self.native_cp_sat_scoring and escalation_predicates
            else ((None, None),)
        )
        construction_choices = tuple(
            (contract, trump, witness_key, use_witness_template, opening_buyer, opening_locked)
            for contract, trump in base_construction_choices
            for witness_key in (
                (
                    _project_exists_witness_keys(
                        project_witness_predicate,
                        self.rules,
                        contract,
                        self.dealer,
                    )
                    or (None,)
                )
                if project_witness_predicate is not None and contract is not None
                and not project_eligibility_only
                and not native_project_opening
                else (None,)
            )
            for use_witness_template in (
                (True, False)
                if self.structural_only
                and witness_key is not None
                and len(project_predicates) == 1
                else (False,)
                if self.native_cp_sat_scoring
                else (True, False)
                if witness_key is not None and len(project_predicates) == 1
                else (False,)
            )
            for opening_buyer in (
                tuple(Seat)
                if native_project_opening and self.rules.play_domain.first_leader == "buyer"
                else (None,)
            )
            for opening_locked in (
                (False, True)
                if native_project_opening and contract.effective == Contract.HOKUM
                and self.rules.play_domain.locked_hokum_blocks_trump_lead
                else (None,)
            )
        )
        exact_infeasible = True
        deal = None
        model = None
        deal_literals: dict[str, Any] = {}
        auction_literals: dict[str, Any] = {}
        bid_zero_literals: dict[str, Any] = {}
        prequalified_projects: OrToolsProjectLifecycleResultV1 | None = None
        prequalified_escalation = None
        opening_prefix = ()
        project_construction_hash: str | None = None
        project_construction_ids = tuple(
            predicate.predicate_id for predicate in project_stage_predicates
        )
        deal_started = time.perf_counter()
        for (
            required_contract,
            required_trump,
            required_project_witness,
            use_project_witness_template,
            opening_buyer,
            opening_locked,
        ) in construction_choices:
            if candidate_deadline <= time.monotonic():
                exact_infeasible = False
                break
            current = OrToolsScenarioModelV1(
                rules=self.rules,
                dealer=self.dealer,
                excluded_deck_orders=excluded_decks,
                minimum_deck_hamming_distance=self.minimum_deck_hamming_distance,
                canonicalize_deal_packets=self.canonicalize_deal_packets,
            )
            _apply_candidate_partition_v1(
                current.model,
                current.card_at_position,
                self.candidate_partition,
                name="complete_trace",
            )
            auction_kwargs = {
                "allowed_contracts": (required_contract,) if required_contract is not None else contracts,
                "prefix_length": auction_prefix_length,
            }
            if self.native_cp_sat_scoring:
                auction_kwargs["require_optimal_actions"] = self.optimal_auction_quality
            auction_fragment = (
                current.add_native_auction_fragment if self.native_cp_sat_scoring
                else current.add_auction_fragment
            )(**auction_kwargs)
            if strategic_escalation_pairs is not None:
                current.model.add_allowed_assignments(
                    [auction_fragment.declarer, auction_fragment.contract],
                    strategic_escalation_pairs,
                )
            if required_trump is not None:
                current.model.add(auction_fragment.trump == int(required_trump))
            if opening_buyer is not None:
                current.model.add(current.buyer == int(opening_buyer))
            if isinstance(manifest, ScenarioPredicateCatalogLoweringManifestV3):
                for predicate in owned_play_attempt_predicates:
                    current.model.add(
                        current.owner_of_card[
                            int(predicate.arguments["attemptedCardId"])
                        ]
                        == int(Seat.parse(predicate.arguments["attemptedActor"]))
                    )
            if auction_predicates:
                auction_context = wrap_auction_adjudication_context_v3(
                    CpSatAuctionPredicateContextV1(
                        current.model,
                        auction_fragment,
                        self.dealer,
                        view=auction_view,
                    ),
                    auction_predicates,
                    origin="search",
                    authority_id=manifest.adjudication_authority_id,
                    authority_sha256=manifest.adjudication_authority_sha256,
                ) if isinstance(
                    manifest, ScenarioPredicateCatalogLoweringManifestV3
                ) else CpSatAuctionPredicateContextV1(
                    current.model,
                    auction_fragment,
                    self.dealer,
                    view=auction_view,
                )
                auction_literals = {
                    predicate.predicate_id: manifest.lower_predicate(
                        predicate, auction_context
                    ).literal
                    for predicate in auction_predicates
                }
                _enforce_hard_stage_projection(
                    current.model,
                    scenario,
                    stage="auction",
                    predicate_literals=auction_literals,
                )
            if bid_zero_score_predicates:
                zero = current.model.new_constant(0)
                bid_zero_literals = {
                    predicate.predicate_id: _state_range_literal(
                        current.model,
                        zero,
                        int(predicate.arguments["minimum"]),
                        int(predicate.arguments["maximum"]),
                        predicate.predicate_id,
                    )
                    for predicate in bid_zero_score_predicates
                }
                _enforce_hard_stage_projection(
                    current.model,
                    scenario,
                    stage="bid_zero_score",
                    predicate_literals=bid_zero_literals,
                )
            if deal_predicates:
                deal_context = DealPredicateModelContextV2(current)
                deal_literals = {
                    predicate.predicate_id: manifest.lower_predicate(
                        predicate, deal_context
                    ).literal
                    for predicate in deal_predicates
                }
                _enforce_hard_stage_projection(
                    current.model,
                    scenario,
                    stage="deal",
                    predicate_literals=deal_literals,
                )
            project_literals = {
                predicate.predicate_id: current.model.new_constant(1)
                for predicate in project_tautologies
            }
            project_witness_hint_ids: list[str] = []
            project_fragment = None
            opening_build = None
            opening_escalation = None
            if native_project_opening or (self.native_cp_sat_scoring and escalation_predicates):
                assert required_contract is not None
                opening_escalation = CpSatEscalationPredicateContextV1.attach(
                    OrToolsEscalationStageV1(
                        EscalationDomainIRV1.from_rules(self.rules.data),
                        declarer=self.dealer,
                        contract=required_contract,
                        match_score=self.match_score,
                        native_cp_sat=True,
                    ),
                    model=current.model,
                    context_id="construction_escalation",
                    declarer_variable=auction_fragment.declarer,
                )
                current.model.add(opening_escalation.native["terminal"] == 1)
                if opening_locked is not None:
                    current.model.add(
                        opening_escalation.native["locked"] == int(opening_locked)
                    )
                _enforce_hard_stage_projection(
                    current.model, scenario, stage="escalation",
                    predicate_literals={
                        predicate.predicate_id: manifest.lower_predicate(
                            predicate, opening_escalation
                        ).literal
                        for predicate in escalation_predicates
                    },
                )
            if native_project_opening:
                opening_build = OrToolsPlayTailStageV1(native_conversion=True).build_model(
                    cards=tuple(Card(index) for index in range(32)),
                    ownership=None,
                    contract=required_contract,
                    trump=required_trump,
                    first_leader=self.rules.play_domain.resolve_first_leader(
                        dealer=self.dealer,
                        buyer=opening_buyer if opening_buyer is not None else self.dealer,
                        structural_domain=self.rules.structural_domain,
                    ),
                    locked=bool(opening_locked),
                    play_domain=self.rules.play_domain,
                    structural_domain=self.rules.structural_domain,
                    play_step_count=4,
                    shared_model=current.model,
                    shared_owner_of_card=current.owner_of_card,
                )
            if project_predicates or native_project_opening:
                assert required_contract is not None
                target_owner = None
                target_cards: tuple[Card, ...] = ()
                if use_project_witness_template:
                    assert project_witness_predicate is not None
                    assert required_project_witness is not None
                    project_literals[project_witness_predicate.predicate_id] = (
                        current.model.new_constant(1)
                    )
                    target_owner = Seat(required_project_witness[0])
                    target_cards = tuple(
                        Card(card_id) for card_id in required_project_witness[3]
                    )
                else:
                    project_fragment = attach_project_lifecycle_fragment_v1(
                        current.model,
                        owner_of_card=current.owner_of_card,
                        contract=required_contract,
                        dealer=self.dealer,
                        trump=required_trump,
                        domain=self.rules.project_domain,
                        name="construction_projects",
                        second_trick_leader=(
                            opening_build._extract.winner_seats[0]
                            if opening_build is not None else None
                        ),
                        **({"eligibility_only": True} if project_eligibility_only else {}),
                    )
                    project_context = CpSatProjectPredicateContextV1(
                        current.model,
                        "construction_projects",
                        project_fragment,
                    )
                    project_literals.update(
                        {
                            predicate.predicate_id: manifest.lower_predicate(
                                predicate, project_context
                            ).literal
                            for predicate in project_predicates
                        }
                    )
                    for literal in project_literals.values():
                        current.model.add_hint(literal, 1)
                    hint_predicate = project_witness_predicate or next(
                        (
                            predicate
                            for predicate in project_predicates
                            if predicate.kind == "project.exists"
                        ),
                        None,
                    )
                    if hint_predicate is not None:
                        arguments = dict(hint_predicate.arguments)
                        matches = project_fragment.matching_projects(
                            owner=arguments.get("owner"),
                            team=arguments["team"],
                            kind=str(arguments["kind"]),
                            value=int(arguments["value"]),
                        )
                        target = next(
                            (
                                project
                                for project in matches
                                if required_project_witness is not None
                                and (
                                    int(project.owner),
                                    project.kind,
                                    project.value,
                                    tuple(card.id for card in project.cards),
                                )
                                == required_project_witness
                            ),
                            max(
                                matches,
                                key=lambda project: (
                                    project.value,
                                    project.category_priority,
                                    project.high_rank,
                                    project.project_id,
                                ),
                                default=None,
                            ),
                        )
                        if target is not None:
                            target_owner = target.owner
                            target_cards = target.cards
                            state = str(arguments.get("state") or "eligible")
                            if required_project_witness is not None:
                                witness_assumption = current.model.new_bool_var(
                                    "assume_project_witness_"
                                    f"{hint_predicate.predicate_id}_{target.project_id}"
                                )
                                current.model.add(
                                    target.literal(state) == 1
                                ).only_enforce_if(witness_assumption)
                                current.model.add_assumption(witness_assumption)
                            current.model.add_hint(target.literal(state), 1)
                if target_owner is not None:
                    hinted_hands: dict[Seat, list[Card]] = {seat: [] for seat in Seat}
                    hinted_hands[target_owner].extend(target_cards)
                    remaining_cards = [
                        Card(card_id)
                        for card_id in range(32)
                        if Card(card_id) not in target_cards
                    ]
                    seat_order = (target_owner, *tuple(
                        seat for seat in Seat if seat != target_owner
                    ))
                    remaining_hint_cards = iter(remaining_cards)
                    while any(len(hinted_hands[seat]) < 8 for seat in seat_order):
                        for seat in seat_order:
                            if len(hinted_hands[seat]) < 8:
                                hinted_hands[seat].append(next(remaining_hint_cards))
                    layout = DealLayoutV1.build(
                        self.rules.structural_domain,
                        dealer=self.dealer,
                        buyer=target_owner,
                    )
                    template_assumption = (
                        current.model.new_bool_var(
                            "assume_project_witness_template_"
                            + canonical_sha256(required_project_witness)[:16]
                        )
                        if use_project_witness_template
                        else None
                    )
                    if template_assumption is not None:
                        current.model.add(
                            current.buyer == int(target_owner)
                        ).only_enforce_if(template_assumption)
                        for card in target_cards:
                            current.model.add(
                                current.owner_of_card[card.id] == int(target_owner)
                            ).only_enforce_if(template_assumption)
                    else:
                        current.model.add_hint(current.buyer, int(target_owner))
                    for seat in Seat:
                        for position, card in zip(
                            layout.positions_for(seat, final=True),
                            hinted_hands[seat],
                            strict=True,
                        ):
                            current.model.add_hint(
                                current.card_at_position[position], card.id
                            )
                    if template_assumption is not None:
                        current.model.add_assumption(template_assumption)
                    project_witness_hint_ids.append(
                        project_witness_predicate.predicate_id
                        if project_witness_predicate is not None
                        else hint_predicate.predicate_id
                    )
            if project_stage_predicates:
                _enforce_hard_stage_projection(
                    current.model,
                    scenario,
                    stage="projects",
                    predicate_literals=project_literals,
                )
                project_construction_hash = canonical_sha256(
                    {
                        "schemaVersion": 1,
                        "stage": "projects",
                        "expressionHash": _combined_expression(scenario).sha256,
                        "predicateIds": list(project_construction_ids),
                        "fragmentHash": (
                            project_fragment.fragment_hash
                            if project_fragment is not None
                            else None
                        ),
                        "hintPolicy": (
                            "fixed_project_witness_template_v1"
                            if use_project_witness_template
                            else "prefer_hard_project_literal_and_interleaved_witness_v2"
                            if project_fragment is not None
                            else None
                        ),
                        "witnessHintPredicateIds": project_witness_hint_ids,
                        "requiredProjectWitness": required_project_witness,
                        "usesProjectWitnessTemplate": use_project_witness_template,
                        "tautologyPredicateIds": [
                            predicate.predicate_id for predicate in project_tautologies
                        ],
                    }
                )
            remaining_seconds = remaining()
            if remaining_seconds <= 0:
                exact_infeasible = False
                break
            construction_solver = cp_model.CpSolver() if self.native_cp_sat_scoring else None
            if construction_solver is not None:
                # Match native tail witnesses: probing can consume the deadline before search.
                construction_solver.parameters.cp_model_probing_level = 0
            attempt = current.solve(
                seed=seed,
                maximum_seconds=remaining_seconds,
                maximum_conflicts=maximum_conflicts,
                search_branching=(
                    "automatic"
                    if project_fragment is not None or self.optimal_auction_quality
                    else "fixed"
                ),
                **({"solver": construction_solver} if construction_solver is not None else {}),
            )
            if self.auction_model_refiner is not None and attempt.status == "satisfied":
                attempt = self.auction_model_refiner(current, attempt, construction_solver, remaining, maximum_conflicts)
            if remaining() <= 0:
                return OrToolsCandidateAttemptV1(
                    "incomplete",
                    None,
                    "deal_auction_project_budget"
                    if project_stage_predicates
                    else "deal_auction_budget",
                )
            if attempt.status == "satisfied":
                if opening_build is not None:
                    solved_projects = project_lifecycle_result_v1(
                        project_fragment, construction_solver
                    )
                    prequalified_projects = replace(
                        solved_projects,
                        model_hash=attempt.model_hash,
                        evidence_hash=canonical_sha256({
                            "constructionModelHash": attempt.model_hash,
                            "fragmentEvidenceHash": solved_projects.evidence_hash,
                        }),
                        variable_count=attempt.variable_count,
                        constraint_count=attempt.constraint_count,
                    )
                    opening_prefix = opening_build._extract(
                        construction_solver,
                        construction_solver.response_proto.status,
                        opening_build.model_hash,
                    ).plays
                if opening_escalation is not None:
                    prequalified_escalation = opening_escalation.extract_stage_result(
                        construction_solver,
                        model_hash=attempt.model_hash,
                        solver_status=attempt.solver_status,
                        branches=attempt.branches,
                        conflicts=attempt.conflicts,
                    )
                if (
                    use_project_witness_template
                    and attempt.auction is not None
                    and attempt.owner_of_card is not None
                    and project_witness_predicate is not None
                    and required_project_witness is not None
                ):
                    template_hands = {
                        seat: tuple(
                            Card(card)
                            for card, owner in enumerate(attempt.owner_of_card)
                            if owner == seat
                        )
                        for seat in Seat
                    }
                    remaining_seconds = remaining()
                    if remaining_seconds <= 0:
                        exact_infeasible = False
                        break
                    try:
                        prequalified_projects = OrToolsProjectLifecycleModelV1().solve(
                            hands=template_hands,
                            contract=attempt.auction.completion_contract,
                            dealer=self.dealer,
                            trump=attempt.auction.completion_trump,
                            domain=self.rules.project_domain,
                            timeout_seconds=remaining_seconds,
                            maximum_conflicts=maximum_conflicts,
                            construction_plan_hash=project_construction_hash,
                            construction_predicate_ids=project_construction_ids,
                            required_witness=required_project_witness,
                            required_witness_state=str(
                                dict(project_witness_predicate.arguments).get("state")
                                or "eligible"
                            ),
                        )
                    except RuntimeError:
                        exact_infeasible = False
                        continue
                model, deal = current, attempt
                break
            exact_infeasible = exact_infeasible and attempt.status == "infeasible"
        if deal is None or model is None:
            if exact_infeasible and self.native_cp_sat_scoring and self.optimal_auction_quality:
                # The visible-hand policy excludes legal auctions; exhausting it
                # does not establish infeasibility of the requested scenario.
                return OrToolsCandidateAttemptV1("not_found", None, "auction_policy_no_witness")
            return OrToolsCandidateAttemptV1(
                "infeasible" if exact_infeasible else "incomplete",
                None,
                (
                    (
                        "deal_auction_project_infeasible"
                        if project_stage_predicates
                        else "deal_auction_infeasible"
                    )
                    if exact_infeasible
                    else (
                        "deal_auction_project_budget"
                        if project_stage_predicates
                        else "deal_auction_budget"
                    )
                ),
            )
        if deal.status == "infeasible":
            return OrToolsCandidateAttemptV1(
                "infeasible", None, "deal_auction_infeasible"
            )
        if (
            deal.status != "satisfied"
            or deal.auction is None
            or deal.owner_of_card is None
        ):
            return OrToolsCandidateAttemptV1("not_found", None, "deal_auction_budget")
        deal_elapsed = max(0.0, time.perf_counter() - deal_started)
        contract = deal.auction.completion_contract
        trump = deal.auction.completion_trump
        buyer = deal.auction.completion_buyer
        hands = {
            seat: tuple(
                Card(card)
                for card, owner in enumerate(deal.owner_of_card)
                if owner == seat
            )
            for seat in Seat
        }
        if self.structural_only:
            trace = {
                "schemaVersion": 1,
                "constructionScope": "deal_auction_project_ownership",
                "deckOrder": list(deal.deck_order or ()),
                "dealer": self.dealer.name.lower(),
                "roundNumber": 1,
                "matchScore": list(self.match_score),
                "buyer": buyer.name.lower(),
                "contract": contract.value,
                "trumpSuit": trump.name.lower() if trump is not None else None,
                "auction": deal.auction.to_dict(),
                "projectConstructionHash": project_construction_hash,
            }
            evidence_hash = _hash_document(trace)
            receipt = _stage_receipt(
                "structural",
                model_hash=deal.model_hash,
                evidence_hash=evidence_hash,
                semantic=trace,
                solver_status=deal.solver_status,
                branches=deal.branches,
                conflicts=deal.conflicts,
                elapsed_seconds=deal_elapsed,
            )
            return OrToolsCandidateAttemptV1(
                "feasible",
                OrToolsNeutralCandidateV1(
                    "search",
                    trace,
                    {
                        "schemaVersion": 1,
                        "constructionScope": trace["constructionScope"],
                        "traceHash": evidence_hash,
                    },
                    tuple(deal.deck_order or ()),
                    {"structural": receipt},
                ),
            )
        escalation_stage = OrToolsEscalationStageV1(
            EscalationDomainIRV1.from_rules(self.rules.data),
            declarer=deal.auction.completion_declarer if self.native_cp_sat_scoring else buyer,
            contract=contract,
            match_score=self.match_score,
            native_cp_sat=self.native_cp_sat_scoring,
        )
        allowed_escalation_fingerprints = None
        if escalation_predicates and not self.native_cp_sat_scoring:
            escalation_fingerprint_seconds = remaining()
            if escalation_fingerprint_seconds <= 0:
                return OrToolsCandidateAttemptV1(
                    "incomplete", None, "escalation_predicate_budget"
                )
            allowed_escalation_fingerprints = _eligible_escalation_fingerprints(
                escalation_stage,
                scenario,
                escalation_predicates,
                maximum_seconds=escalation_fingerprint_seconds,
                maximum_conflicts=maximum_conflicts,
            )
        if remaining() <= 0:
            return OrToolsCandidateAttemptV1(
                "incomplete", None, "escalation_predicate_budget"
            )
        if escalation_predicates and not self.native_cp_sat_scoring and allowed_escalation_fingerprints is None:
            return OrToolsCandidateAttemptV1(
                "incomplete", None, "escalation_predicate_budget"
            )
        escalation_started = time.perf_counter()
        def configure_escalation_model(cp_model_value, variables):
            context = CpSatEscalationPredicateContextV1(cp_model_value, escalation_stage, (), (), variables)
            literals = {predicate.predicate_id: manifest.lower_predicate(predicate, context).literal
                        for predicate in escalation_predicates}
            _enforce_hard_stage_projection(cp_model_value, scenario, stage="escalation", predicate_literals=literals)

        escalation_seconds = remaining()
        if escalation_seconds <= 0:
            return OrToolsCandidateAttemptV1(
                "incomplete", None, "escalation_incomplete"
            )
        escalation = prequalified_escalation or escalation_stage.solve(
            required_terminal=True,
            allowed_candidate_fingerprints=allowed_escalation_fingerprints,
            seed=seed,
            maximum_seconds=escalation_seconds,
            maximum_conflicts=maximum_conflicts,
            configure_model=configure_escalation_model if self.native_cp_sat_scoring and escalation_predicates else None,
        )
        if remaining() <= 0:
            return OrToolsCandidateAttemptV1(
                "incomplete", None, "escalation_incomplete"
            )
        escalation_elapsed = (
            0.0 if prequalified_escalation is not None
            else max(0.0, time.perf_counter() - escalation_started)
        )
        if escalation.status != "satisfied":
            return OrToolsCandidateAttemptV1(
                "incomplete" if escalation.status == "incomplete" else "infeasible",
                None,
                "escalation_" + escalation.status,
            )
        projects_started = time.perf_counter()
        project_seconds = remaining()
        if project_seconds <= 0:
            return OrToolsCandidateAttemptV1(
                "incomplete", None, "project_budget"
            )
        projects = prequalified_projects or OrToolsProjectLifecycleModelV1().solve(
            hands=hands,
            contract=contract,
            dealer=self.dealer,
            trump=trump,
            domain=self.rules.project_domain,
            timeout_seconds=project_seconds,
            maximum_conflicts=maximum_conflicts,
            construction_plan_hash=project_construction_hash,
            construction_predicate_ids=project_construction_ids,
        )
        if remaining() <= 0:
            return OrToolsCandidateAttemptV1(
                "incomplete", None, "project_budget"
            )
        projects_elapsed = (
            0.0 if prequalified_projects is not None
            else max(0.0, time.perf_counter() - projects_started)
        )
        ownership = {card: seat for seat, hand in hands.items() for card in hand}
        first_leader = self.rules.play_domain.resolve_first_leader(
            dealer=self.dealer,
            buyer=buyer,
            structural_domain=self.rules.structural_domain,
        )
        scoring = ScoringDomainIRV1.from_rules(self.rules.data)
        static_prefix_length = _static_decision_prefix_length(scenario)
        play_predicates = tuple(
            predicate
            for predicate in scenario.predicates()
            if predicate.hardness == "hard"
            and (definition := manifest.definition(predicate.kind)) is not None
            and definition.stage == "playPrefix"
            and definition.cp_sat_support.qualification_source == "construction"
        )
        outcome_predicates = tuple(
            predicate
            for predicate in scenario.predicates()
            if predicate.hardness == "hard"
            and (definition := manifest.definition(predicate.kind)) is not None
            and definition.stage == "outcome"
        )
        play_action_kinds = {
            "play.action_matches",
            "play.trick_card_count",
            "play.trick_winner_matches",
        }
        play_construction_plan_hash = (
            canonical_sha256(
                {
                    "schemaVersion": 1,
                    "stage": "playPrefix",
                    "expressionHash": _combined_expression(scenario).sha256,
                    "predicateIds": [
                        predicate.predicate_id
                        for predicate in (*play_predicates, *outcome_predicates)
                    ],
                    "nativeScoring": self.native_cp_sat_scoring,
                }
            )
            if play_predicates or self.native_cp_sat_scoring
            else None
        )
        play_build: Any | None = None
        play_literals: dict[str, Any] = {}
        play_project_fragment: OrToolsProjectLifecycleFragmentV1 | None = None

        def configure_play_model(build) -> None:
            nonlocal play_build, play_literals, play_project_fragment
            play_build = build
            if self.play_model_configurator is not None:
                self.play_model_configurator(build, {
                    "deckOrder": list(deal.deck_order), "dealer": self.dealer.name.lower(),
                    "matchScore": list(self.match_score), "auction": deal.auction.to_dict(),
                    "escalation": escalation.to_dict(),
                })
            play_adjudication_context = (
                wrap_play_adjudication_context_v3(
                    build.play_state_context,
                    build.initial_owner_variables,
                    play_predicates,
                    origin="search",
                    authority_id=manifest.adjudication_authority_id,
                    authority_sha256=manifest.adjudication_authority_sha256,
                )
                if isinstance(manifest, ScenarioPredicateCatalogLoweringManifestV3)
                else build.play_state_context
            )
            literals: dict[str, Any] = {
                predicate.predicate_id: manifest.lower_predicate(
                    predicate,
                    (
                        build.predicate_context
                        if predicate.kind in play_action_kinds
                        else play_adjudication_context
                        if predicate.kind
                        in {OUT_OF_TURN_PREDICATE_V3, FALSE_EKKA_PREDICATE_V3}
                        else build.play_state_context
                    ),
                ).literal
                for predicate in play_predicates
            }
            if self.native_cp_sat_scoring:
                if len(build.raw_team_variables) != 2 or len(
                    build.winner_team_variables
                ) != 8:
                    raise ValueError("native scoring requires a complete play model")

                project_fragment = attach_project_lifecycle_fragment_v1(
                    build.model,
                    owner_of_card=build.initial_owner_variables,
                    fixed_card_owners=tuple(ownership[Card(index)] for index in range(32)),
                    contract=contract,
                    dealer=self.dealer,
                    trump=trump,
                    domain=self.rules.project_domain,
                    name="play_score_projects",
                    second_trick_leader=(
                        build._extract.winner_seats[0]
                        if self.rules.project_domain.tie_break_order == "second_trick_reveal"
                        else None
                    ),
                    compact_fixed_states=True,
                    tighten_value_bounds=True,
                )
                play_project_fragment = project_fragment
                if self.play_model_refiner is not None and project_stage_predicates:
                    project_context = CpSatProjectPredicateContextV1(build.model, "play_score_projects", project_fragment)
                    _enforce_hard_stage_projection(build.model, scenario, stage="projects", predicate_literals={
                        predicate.predicate_id: manifest.lower_predicate(predicate, project_context).literal
                        for predicate in project_stage_predicates})
                recorded_projects = {row.project_id: row for row in projects.projects}
                active_projects = {
                    row.project_id: row
                    for row in project_fragment.projects
                    if tuple(row.declared.proto.domain) != (0, 0)
                }
                if set(active_projects) != set(recorded_projects):
                    raise ValueError("play scoring project catalog differs from construction")
                for project_id, row in active_projects.items():
                    build.model.add(row.declared == int(recorded_projects[project_id].declared))
                baloot_points = tuple(
                    build.model.new_int_var(
                        0, project_fragment.baloot_value, f"play_score_baloot_{int(team)}"
                    )
                    for team in Team
                )
                for team in Team:
                    build.model.add(
                        baloot_points[int(team)]
                        == project_fragment.baloot_value
                        * project_fragment.baloot_team[int(team)]
                    )

                def fixed_input(name: str, value: int, upper: int) -> cp_model.IntVar:
                    variable = build.model.new_int_var(0, max(value, upper), name)
                    build.model.add(variable == value)
                    return variable

                level = escalation.observation.level
                actions = escalation.observation.actions
                score_source_hash = canonical_sha256(
                    {
                        "schemaVersion": 1,
                        "stage": "play_scoring",
                        "scenarioHash": _combined_expression(scenario).sha256,
                        "projectEvidenceHash": projects.evidence_hash,
                        "projectFragmentHash": project_fragment.fragment_hash,
                        "escalationEvidenceHash": escalation.evidence_hash,
                        "playRequestHash": build.request_hash,
                        "scoringDomainHash": scoring.sha256,
                    }
                )
                scoring_fragment = attach_scoring_fragment_v1(
                    build.model,
                    scoring,
                    CpSatScoringInputsV1(
                        contract=fixed_input(
                            "score_input_contract",
                            0 if contract.effective == Contract.SUN else 1,
                            1,
                        ),
                        buyer_seat=fixed_input(
                            "score_input_buyer", int(buyer), 3
                        ),
                        escalation=fixed_input(
                            "score_input_escalation",
                            {
                                "normal": 0,
                                "double": 1,
                                "triple": 2,
                                "four": 3,
                                "gahwa": 4,
                            }[level],
                            4,
                        ),
                        last_escalating_team=fixed_input(
                            "score_input_last_team",
                            int(actions[-1].actor.team) if actions else 2,
                            2,
                        ),
                        raw_points=build.raw_team_variables,
                        project_points=project_fragment.combined_awarded_value,
                        baloot_points=baloot_points,
                        trick_winner_teams=build.winner_team_variables,
                        starting_match_score=tuple(
                            fixed_input(
                                f"score_input_match_{team}", value, scoring.match_target
                            )
                            for team, value in enumerate(self.match_score)
                        ),
                    ),
                    source_hash=score_source_hash,
                    native_conversion=True,
                )
                literals.update(
                    {
                        predicate.predicate_id: manifest.lower_predicate(
                            predicate, scoring_fragment.outcome_context
                        ).literal
                        for predicate in outcome_predicates
                    }
                )
            _enforce_hard_stage_projection(
                build.model,
                scenario,
                stage="play_scoring",
                predicate_literals=literals,
            )
            play_literals = dict(literals)

        def verify_play_project_scores(solver: cp_model.CpSolver | None) -> None:
            nonlocal projects
            if solver is None:
                return
            if play_project_fragment is None:
                raise AssertionError("native play project fragment is missing")
            if self.play_model_refiner is not None:
                projects = project_lifecycle_result_v1(play_project_fragment, solver)
                return
            solved_projects = play_project_fragment.extract(solver)
            if (
                solved_projects.combined_awarded_value
                != projects.combined_awarded_value
                or solved_projects.baloot_points != projects.baloot_points
            ):
                raise AssertionError("native play and project stages disagree on scoring")

        play_started = time.perf_counter()
        play_seconds = remaining()
        if play_seconds <= 0:
            return OrToolsCandidateAttemptV1("incomplete", None, "play_budget")
        first_solver = cp_model.CpSolver() if self.native_cp_sat_scoring else None
        first_play_infeasibility = None
        try:
            first_play = OrToolsPlayTailStageV1(native_conversion=self.native_cp_sat_scoring).solve(
                cards=tuple(Card(index) for index in range(32)),
                ownership=ownership,
                contract=contract,
                trump=trump,
                first_leader=first_leader,
                # A policy-refined line may legally change the opening winner,
                # hence project comparison; its live fragment is authoritative.
                prefix=() if self.play_model_refiner is not None else opening_prefix,
                locked=escalation.observation.mode == "locked",
                play_domain=self.rules.play_domain,
                structural_domain=self.rules.structural_domain,
                scoring_domain=scoring,
                decision_prefix_length=static_prefix_length,
                decision_actor=(
                    dict(scenario.decision_point["selector"]).get("actingSeat")
                    if static_prefix_length is not None
                    else None
                ),
                decision_purchase_card=deal.purchase_card,
                excluded_decision_status_vectors=(
                    self.excluded_decision_status_vectors
                    if static_prefix_length is not None
                    else ()
                ),
                minimum_decision_status_hamming_distance=(
                    self.minimum_decision_status_hamming_distance
                ),
                construction_plan_hash=play_construction_plan_hash,
                model_configurator=(
                    configure_play_model
                    if play_predicates or self.native_cp_sat_scoring
                    else None
                ),
                maximum_time_seconds=play_seconds,
                maximum_conflicts=maximum_conflicts,
                solver=first_solver,
            )
            if self.play_model_refiner is not None:
                first_play = self.play_model_refiner(play_build, {
                    "deckOrder": list(deal.deck_order), "dealer": self.dealer.name.lower(),
                    "matchScore": list(self.match_score), "auction": deal.auction.to_dict(),
                    "escalation": escalation.to_dict(),
                }, first_play, first_solver, remaining, maximum_conflicts)
            verify_play_project_scores(first_solver)
        except PlayTailSolveStopped as stopped:
            if stopped.solver_status != "infeasible":
                return OrToolsCandidateAttemptV1(
                    "incomplete",
                    None,
                    (
                        "decision_diversity_budget"
                        if self.excluded_decision_status_vectors
                        else "play_budget"
                    ),
                    partial_receipt=stopped.partial_receipt,
                )
            first_play = None
            first_play_infeasibility = stopped.partial_receipt
        if remaining() <= 0:
            return OrToolsCandidateAttemptV1("incomplete", None, "play_budget")
        if (
            first_play is None
            and static_prefix_length is not None
            and self.excluded_decision_status_vectors
        ):
            play_seconds = remaining()
            if play_seconds <= 0:
                return OrToolsCandidateAttemptV1(
                    "incomplete", None, "decision_diversity_budget"
                )
            try:
                first_solver = cp_model.CpSolver() if self.native_cp_sat_scoring else None
                first_play = OrToolsPlayTailStageV1(native_conversion=self.native_cp_sat_scoring).solve(
                    cards=tuple(Card(index) for index in range(32)),
                    ownership=ownership,
                    contract=contract,
                    trump=trump,
                    first_leader=first_leader,
                    prefix=opening_prefix,
                    locked=escalation.observation.mode == "locked",
                    play_domain=self.rules.play_domain,
                    structural_domain=self.rules.structural_domain,
                    scoring_domain=scoring,
                    decision_prefix_length=static_prefix_length,
                    decision_actor=dict(scenario.decision_point["selector"]).get(
                        "actingSeat"
                    ),
                    decision_purchase_card=deal.purchase_card,
                    excluded_decision_status_vectors=(),
                    minimum_decision_status_hamming_distance=(
                        self.minimum_decision_status_hamming_distance
                    ),
                    construction_plan_hash=play_construction_plan_hash,
                    model_configurator=(
                        configure_play_model
                        if play_predicates or self.native_cp_sat_scoring
                        else None
                    ),
                    maximum_time_seconds=play_seconds,
                    maximum_conflicts=maximum_conflicts,
                    solver=first_solver,
                )
                verify_play_project_scores(first_solver)
            except PlayTailSolveStopped:
                return OrToolsCandidateAttemptV1("incomplete", None, "play_budget")
            if remaining() <= 0:
                return OrToolsCandidateAttemptV1("incomplete", None, "play_budget")
        elif first_play is None:
            return OrToolsCandidateAttemptV1("incomplete", None, "fixed_deal_play_infeasible",
                partial_receipt=first_play_infeasibility)
        if first_play.solver_status not in {"optimal", "feasible"}:
            return OrToolsCandidateAttemptV1("incomplete", None, "play_budget")
        first_play_build = play_build
        first_play_literals = play_literals
        prefix_length = _decision_prefix_length(scenario, first_play.plays)
        if (
            self.excluded_decision_status_vectors
            and static_prefix_length is None
            and scenario.decision_point["kind"] == "play"
        ):
            play_seconds = remaining()
            if play_seconds <= 0:
                return OrToolsCandidateAttemptV1(
                    "incomplete", None, "decision_diversity_budget"
                )
            try:
                diversity_solver = cp_model.CpSolver() if self.native_cp_sat_scoring else None
                play = OrToolsPlayTailStageV1(native_conversion=self.native_cp_sat_scoring).solve(
                    cards=tuple(Card(index) for index in range(32)),
                    ownership=ownership,
                    contract=contract,
                    trump=trump,
                    first_leader=first_leader,
                    prefix=opening_prefix,
                    locked=escalation.observation.mode == "locked",
                    play_domain=self.rules.play_domain,
                    structural_domain=self.rules.structural_domain,
                    scoring_domain=scoring,
                    decision_prefix_length=prefix_length,
                    decision_actor=dict(scenario.decision_point["selector"]).get(
                        "actingSeat"
                    ),
                    decision_purchase_card=deal.purchase_card,
                    excluded_decision_status_vectors=(
                        self.excluded_decision_status_vectors
                    ),
                    minimum_decision_status_hamming_distance=(
                        self.minimum_decision_status_hamming_distance
                    ),
                    construction_plan_hash=play_construction_plan_hash,
                    model_configurator=(
                        configure_play_model
                        if play_predicates or self.native_cp_sat_scoring
                        else None
                    ),
                    maximum_time_seconds=play_seconds,
                    maximum_conflicts=maximum_conflicts,
                    solver=diversity_solver,
                )
                verify_play_project_scores(diversity_solver)
            except PlayTailSolveStopped as stopped:
                if stopped.solver_status != "infeasible":
                    return OrToolsCandidateAttemptV1(
                        "incomplete", None, "decision_diversity_budget"
                    )
                play = first_play
                play_build = first_play_build
                play_literals = first_play_literals
            if remaining() <= 0:
                return OrToolsCandidateAttemptV1(
                    "incomplete", None, "decision_diversity_budget"
                )
            if play.solver_status not in {"optimal", "feasible"}:
                return OrToolsCandidateAttemptV1(
                    "incomplete", None, "decision_diversity_budget"
                )
            play = play.at_prefix(prefix_length)
        else:
            play = first_play.at_prefix(prefix_length)
        play_elapsed = max(0.0, time.perf_counter() - play_started)
        score_started = time.perf_counter()
        score = _score_complete_candidate(
            rules=self.rules,
            contract=contract,
            buyer=buyer,
            escalation=escalation,
            projects=projects,
            play=play,
            starting_match_score=self.match_score,
            native_cp_sat=self.native_cp_sat_scoring,
        )
        score_elapsed = max(0.0, time.perf_counter() - score_started)
        auction_doc = deal.auction.to_dict()
        trace = {
            "schemaVersion": 1,
            "deckOrder": list(deal.deck_order or ()),
            "dealer": self.dealer.name.lower(),
            "roundNumber": 1,
            "matchScore": list(self.match_score),
            "buyer": buyer.name.lower(),
            "contract": contract.value,
            "trumpSuit": trump.name.lower() if trump is not None else None,
            "auctionPredicateView": auction_view,
            "auction": auction_doc,
            "escalation": escalation.to_dict(),
            "projects": projects.to_dict(),
            "play": play.to_dict(),
            "score": score,
        }
        trace_hash = _hash_document(trace)
        deal_assumptions, deal_bindings = _cp_sat_receipt_metadata_v1(
            model.model,
            stage="deal",
            predicates=deal_predicates,
            predicate_literals=deal_literals,
            model_hash=deal.model_hash,
            evidence_hash=deal.solution_hash or _HASH_ZERO,
            lowering_manifest_hash=manifest.manifest_hash,
        )
        auction_assumptions, auction_bindings = _cp_sat_receipt_metadata_v1(
            model.model,
            stage="auction",
            predicates=auction_predicates,
            predicate_literals=auction_literals,
            model_hash=deal.model_hash,
            evidence_hash=deal.solution_hash or _HASH_ZERO,
            lowering_manifest_hash=manifest.manifest_hash,
        )
        bid_zero_assumptions, bid_zero_bindings = _cp_sat_receipt_metadata_v1(
            model.model,
            stage="bid_zero_score",
            predicates=bid_zero_score_predicates,
            predicate_literals=bid_zero_literals,
            model_hash=deal.model_hash,
            evidence_hash=deal.solution_hash or _HASH_ZERO,
            lowering_manifest_hash=manifest.manifest_hash,
        )
        play_assumptions: tuple[Mapping[str, Any], ...] = ()
        play_bindings: tuple[Mapping[str, Any], ...] = ()
        outcome_assumptions: tuple[Mapping[str, Any], ...] = ()
        outcome_bindings: tuple[Mapping[str, Any], ...] = ()
        if play_build is not None:
            if play_build.model_hash != play.model_hash:
                raise AssertionError("play receipt model differs from selected candidate")
            play_assumptions, play_bindings = _cp_sat_receipt_metadata_v1(
                play_build.model,
                stage="play_scoring",
                predicates=play_predicates,
                predicate_literals={
                    predicate.predicate_id: play_literals[predicate.predicate_id]
                    for predicate in play_predicates
                    if predicate.predicate_id in play_literals
                },
                model_hash=play.model_hash,
                evidence_hash=play.evidence_hash,
                lowering_manifest_hash=manifest.manifest_hash,
            )
            outcome_assumptions, outcome_bindings = _cp_sat_receipt_metadata_v1(
                play_build.model,
                stage="play_scoring",
                predicates=outcome_predicates,
                predicate_literals={
                    predicate.predicate_id: play_literals[predicate.predicate_id]
                    for predicate in outcome_predicates
                    if predicate.predicate_id in play_literals
                },
                model_hash=play.model_hash,
                evidence_hash=score["sha256"],
                lowering_manifest_hash=manifest.manifest_hash,
            )
        receipts = {
            "deal": _stage_receipt(
                "deal",
                model_hash=deal.model_hash,
                evidence_hash=deal.solution_hash or _HASH_ZERO,
                semantic={
                    "variableCount": deal.variable_count,
                    "constraintCount": deal.constraint_count,
                    "deckOrder": list(deal.deck_order or ()),
                    "owners": [item.name.lower() for item in deal.owner_of_card],
                    "noGoodHashes": list(deal.no_good_hashes),
                    "minimumDeckHammingDistance": (deal.minimum_deck_hamming_distance),
                    "constructionPredicateIds": [
                        predicate.predicate_id for predicate in deal_predicates
                    ],
                    "constructionExpressionHash": (
                        _combined_expression(scenario).sha256
                        if deal_predicates
                        else None
                    ),
                },
                solver_status=deal.solver_status,
                branches=deal.branches,
                conflicts=deal.conflicts,
                named_assumptions=deal_assumptions,
                evidence_bindings=deal_bindings,
                elapsed_seconds=deal_elapsed,
            ),
            "auction": _stage_receipt(
                "auction",
                model_hash=deal.model_hash,
                evidence_hash=deal.solution_hash or _HASH_ZERO,
                semantic={
                    "variableCount": deal.variable_count,
                    "constraintCount": deal.constraint_count,
                    **auction_doc,
                    "constructionPredicateIds": [
                        predicate.predicate_id for predicate in auction_predicates
                    ],
                    "bidZeroScorePredicateIds": [
                        predicate.predicate_id for predicate in bid_zero_score_predicates
                    ],
                    "constructionExpressionHash": (
                        _combined_expression(scenario).sha256
                        if auction_predicates
                        else None
                    ),
                },
                solver_status=deal.solver_status,
                branches=deal.branches,
                conflicts=deal.conflicts,
                named_assumptions=auction_assumptions + bid_zero_assumptions,
                evidence_bindings=auction_bindings + bid_zero_bindings,
                elapsed_seconds=deal_elapsed,
            ),
            "escalation": _stage_receipt(
                "escalation",
                model_hash=escalation.model_hash,
                evidence_hash=escalation.evidence_hash or _HASH_ZERO,
                semantic={
                    **escalation.to_dict(),
                    "constructionPredicateIds": [
                        predicate.predicate_id for predicate in escalation_predicates
                    ],
                    "constructionExpressionHash": (
                        _combined_expression(scenario).sha256
                        if escalation_predicates
                        else None
                    ),
                    "terminalBoundaryRequired": True,
                },
                solver_status=escalation.solver_status,
                branches=escalation.branches,
                conflicts=escalation.conflicts,
                elapsed_seconds=escalation_elapsed,
            ),
            "projects": _stage_receipt(
                "projects",
                model_hash=projects.model_hash,
                evidence_hash=projects.evidence_hash,
                semantic={
                    **projects.to_dict(),
                    "constructionPredicateIds": list(project_construction_ids),
                    "constructionPlanHash": project_construction_hash,
                },
                solver_status=projects.solver_status,
                branches=projects.branches,
                conflicts=projects.conflicts,
                elapsed_seconds=projects_elapsed,
            ),
            "playPrefix": _stage_receipt(
                "playPrefix",
                model_hash=play.model_hash,
                evidence_hash=play.evidence_hash,
                semantic=play.to_dict(),
                solver_status=play.solver_status,
                named_assumptions=play_assumptions,
                evidence_bindings=play_bindings,
                elapsed_seconds=play_elapsed,
            ),
            "state": _stage_receipt(
                "state",
                model_hash=play.model_hash,
                evidence_hash=play.evidence_hash,
                semantic={
                    **play.to_dict()["decisionState"],
                    "variableCount": play.variable_count,
                    "constraintCount": play.constraint_count,
                },
                solver_status=play.solver_status,
                elapsed_seconds=play_elapsed,
            ),
            "outcome": _stage_receipt(
                "outcome",
                model_hash=(
                    play.model_hash
                    if self.native_cp_sat_scoring
                    else canonical_sha256({"scoringDomainHash": scoring.sha256})
                ),
                evidence_hash=score["sha256"],
                semantic=score,
                solver_status="optimal",
                branches=int(score.get("branches", 0)),
                conflicts=int(score.get("conflicts", 0)),
                variable_count=(
                    play.variable_count if self.native_cp_sat_scoring else None
                ),
                constraint_count=(
                    play.constraint_count if self.native_cp_sat_scoring else None
                ),
                named_assumptions=outcome_assumptions,
                evidence_bindings=outcome_bindings,
                elapsed_seconds=score_elapsed,
            ),
        }
        state = {
            "schemaVersion": 1,
            "decisionPoint": copy.deepcopy(dict(scenario.decision_point)),
            "decisionState": play.to_dict()["decisionState"],
            "score": score,
            "traceHash": trace_hash,
        }
        return OrToolsCandidateAttemptV1(
            "feasible",
            OrToolsNeutralCandidateV1(
                "search",
                trace,
                state,
                tuple(deal.deck_order or ()),
                receipts,
            ),
        )


class _PlayQualificationContextV1:
    """One fixed play model serving both legacy and reusable play predicates."""

    __slots__ = (
        "_base",
        "model",
        "context_id",
        "tricks",
        "root_actions",
        "proof_complete",
        "sha256",
        "project_context",
        "native_build",
    )

    def __init__(
        self,
        base: CpSatPlayStatePredicateContextV1,
        tricks: Mapping[int, CpSatTrickVariablesV1],
        *,
        project_context: CpSatProjectPredicateContextV1 | None = None,
        native_build: Any = None,
    ) -> None:
        self._base = base
        self.model = base.model
        self.context_id = base.context_id
        self.tricks = dict(tricks)
        self.root_actions = ()
        self.proof_complete = base.model.new_constant(1)
        self.project_context = project_context
        self.native_build = native_build
        self.sha256 = canonical_sha256(
            {
                "schemaVersion": 1,
                "baseContextHash": base.sha256,
                "tricks": {
                    str(index): [action.action_id for action in trick.actions]
                    for index, trick in sorted(self.tricks.items())
                },
            }
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


class _ProofQualificationContextV1:
    """Validated proof/tactic facts plus native root-action proof variables."""

    __slots__ = (
        "model",
        "context_id",
        "proof",
        "principal_variation_tricks",
        "tactical_evidence",
        "root_actions",
        "root_winning_variables",
        "root_value_variables",
        "root_stopping_variables",
        "root_action_codes",
        "root_actor_seat",
        "root_actor_team",
        "proof_utility",
        "decision_kind",
        "selected_root_action_index",
        "root_actor_is_target",
        "root_actor_is_target_variable",
        "target_seat_variable",
        "card_strength_by_id",
        "first_principal_trick_winner_team",
        "first_principal_trick_winner_team_variable",
        "trump_suit",
        "trump_suit_variable",
        "base_tricks",
        "base_trick_variables",
        "principal_trick_winner_seat_variables",
        "principal_variation_has_continuation_variable",
        "proof_complete",
        "native_proof_bindings",
        "native_cp_sat_authority",
        "tactical_detector_profile",
        "native_public_history_prefix",
        "sha256",
    )

    def __init__(
        self,
        base: ProofTacticPredicateContextV1,
        root_actions: tuple[CpSatActionVariablesV1, ...],
        *,
        root_winning_variables: tuple[Any, ...],
        root_value_variables: tuple[Any, ...],
        root_actor_seat: Seat | None,
        root_actor_team: Team | None,
        proof_utility: str | None,
        decision_kind: str | None,
        selected_root_action_index: int | None,
        root_actor_is_target: bool | None,
        root_actor_is_target_variable: Any | None,
        target_seat_variable: Any | None,
        card_strength_by_id: tuple[int, ...],
        first_principal_trick_winner_team: Team | None,
        first_principal_trick_winner_team_variable: Any | None,
        trump_suit: Suit | None,
        trump_suit_variable: Any | None,
        base_tricks: tuple[int, int] | None,
        base_trick_variables: tuple[Any, Any],
        principal_trick_winner_seat_variables: tuple[Any, ...],
        principal_variation_has_continuation_variable: Any,
        native_proof_bindings: bool = False,
        native_cp_sat_authority: bool = False,
        tactical_detector_profile: TacticalDetectorProfileV1 | None = None,
        root_stopping_variables: tuple[Any, ...] = (),
    ) -> None:
        self.model = base.model
        self.context_id = base.context_id
        self.proof = base.proof
        self.principal_variation_tricks = base.principal_variation_tricks
        self.tactical_evidence = base.tactical_evidence
        self.root_actions = root_actions
        self.root_winning_variables = root_winning_variables
        self.root_value_variables = root_value_variables
        self.root_stopping_variables = root_stopping_variables
        self.root_action_codes = tuple(row.action_code for row in base.proof.root_actions)
        self.root_actor_seat = int(root_actor_seat) if root_actor_seat is not None else None
        self.root_actor_team = int(root_actor_team) if root_actor_team is not None else None
        self.proof_utility = proof_utility
        self.decision_kind = decision_kind
        self.selected_root_action_index = selected_root_action_index
        self.root_actor_is_target = root_actor_is_target
        self.root_actor_is_target_variable = root_actor_is_target_variable
        self.target_seat_variable = target_seat_variable
        self.card_strength_by_id = card_strength_by_id
        self.first_principal_trick_winner_team = (
            int(first_principal_trick_winner_team)
            if first_principal_trick_winner_team is not None
            else None
        )
        self.first_principal_trick_winner_team_variable = (
            first_principal_trick_winner_team_variable
        )
        self.trump_suit = int(trump_suit) if trump_suit is not None else None
        self.trump_suit_variable = trump_suit_variable
        self.base_tricks = base_tricks
        self.base_trick_variables = base_trick_variables
        self.principal_trick_winner_seat_variables = (
            principal_trick_winner_seat_variables
        )
        self.principal_variation_has_continuation_variable = (
            principal_variation_has_continuation_variable
        )
        self.native_proof_bindings = native_proof_bindings
        self.native_cp_sat_authority = native_cp_sat_authority
        self.tactical_detector_profile = tactical_detector_profile
        self.proof_complete = base.model.new_bool_var("qualification_proof_complete")
        base.model.add(
            self.proof_complete
            == int(
                base.proof.solver_status == "optimal"
                and base.proof.completion_status
                in ({"exact_complete", "proof_failed"} if native_proof_bindings
                    else {"exact_complete", "bounded_complete", "policy_complete"})
            )
        )
        self.native_public_history_prefix = (
            (len(base.model.proto.variables), len(base.model.proto.constraints), canonical_sha256({
                "variables": list(map(str, base.model.proto.variables)),
                "constraints": list(map(str, base.model.proto.constraints)),
                "cardStrengthById": list(card_strength_by_id),
                **({"tacticalDetectorProfileHash": tactical_detector_profile.sha256} if tactical_detector_profile is not None else {}),
            })) if native_proof_bindings and base_tricks is None else None
        )
        self.sha256 = canonical_sha256(
            {
                "schemaVersion": 1,
                "baseContextHash": base.sha256,
                "rootActionIds": [item.action_id for item in root_actions],
                "rootActionCodes": list(self.root_action_codes),
                "rootWinningVariables": [item.name for item in root_winning_variables],
                "rootValueVariables": [item.name for item in root_value_variables],
                "rootActorSeat": (
                    root_actor_seat.name.lower()
                    if root_actor_seat is not None
                    else None
                ),
                "rootActorTeam": (
                    root_actor_team.name.lower()
                    if root_actor_team is not None
                    else None
                ),
                "proofUtility": proof_utility,
                "decisionKind": decision_kind,
                "selectedRootActionIndex": selected_root_action_index,
                "rootActorIsTarget": root_actor_is_target,
                "rootActorIsTargetVariable": (
                    root_actor_is_target_variable.name
                    if root_actor_is_target_variable is not None
                    else None
                ),
                "targetSeatVariable": (
                    target_seat_variable.name
                    if target_seat_variable is not None
                    else None
                ),
                "cardStrengthById": list(card_strength_by_id),
                "firstPrincipalTrickWinnerTeam": (
                    first_principal_trick_winner_team.name.lower()
                    if first_principal_trick_winner_team is not None
                    else None
                ),
                "firstPrincipalTrickWinnerTeamVariable": (
                    first_principal_trick_winner_team_variable.name
                    if first_principal_trick_winner_team_variable is not None
                    else None
                ),
                "trumpSuit": trump_suit.name.lower() if trump_suit is not None else None,
                "trumpSuitVariable": (
                    trump_suit_variable.name if trump_suit_variable is not None else None
                ),
                "baseTricks": list(base_tricks) if base_tricks is not None else None,
                "baseTrickVariables": [item.name for item in base_trick_variables],
                "principalTrickWinnerSeatVariables": [
                    item.name for item in principal_trick_winner_seat_variables
                ],
                "principalVariationHasContinuationVariable": (
                    principal_variation_has_continuation_variable.name
                ),
                "proofCompleteVariable": self.proof_complete.name,
                **({"nativeProofBindings": True} if native_proof_bindings else {}),
                **({"nativeCpSatAuthority": True} if native_cp_sat_authority else {}),
                **({"tacticalDetectorProfileHash": tactical_detector_profile.sha256} if tactical_detector_profile is not None else {}),
                **({"nativePublicHistoryPrefixHash": self.native_public_history_prefix[2]}
                   if self.native_public_history_prefix is not None else {}),
                **({"rootStoppingVariables": [variable.name for variable in root_stopping_variables]}
                   if root_stopping_variables else {}),
            }
        )


def _candidate_play_rows(
    candidate: OrToolsNeutralCandidateV1,
    *,
    complete: bool = False,
) -> tuple[dict, list]:
    raw_play = (
        candidate.trace.get("completePlay") if complete else candidate.trace.get("play")
    )
    if not isinstance(raw_play, Mapping):
        raw_play = candidate.trace["play"]
    play = dict(raw_play)
    rows = list(play.get("plays") or ())
    if candidate.origin == "exact_trace_prefix" and not complete:
        prefix = candidate.trace.get("exactTracePrefix")
        if isinstance(prefix, Mapping):
            rows = rows[: len(prefix.get("playActions", ()))]
    return play, rows


def _initial_owner_by_card(
    candidate: OrToolsNeutralCandidateV1,
    play: Mapping[str, Any],
) -> dict[int, Seat]:
    result: dict[int, Seat] = {}
    for seat, cards in dict(play.get("initialHands") or {}).items():
        for card in cards:
            result[Card.parse(card).id] = Seat.parse(seat)
    if len(result) == 32:
        return result
    hands = candidate.state.get("hands")
    if isinstance(hands, Mapping):
        for seat, cards in hands.items():
            for card in cards:
                result[Card.parse(card).id] = Seat.parse(seat)
    for row in play.get("plays", ()):
        result[Card.parse(row["card"]).id] = Seat.parse(row["actor"])
    return result


def _constant_general_tricks(
    candidate: OrToolsNeutralCandidateV1,
    *,
    model: cp_model.CpModel,
    play: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[int, CpSatTrickVariablesV1]:
    owners = _initial_owner_by_card(candidate, play)
    trick_rows = {int(item["index"]): item for item in (play.get("tricks") or ())}
    result: dict[int, CpSatTrickVariablesV1] = {}
    for trick_index in range(8):
        actions: list[CpSatActionVariablesV1] = []
        for position in range(4):
            index = trick_index * 4 + position
            present = index < len(rows)
            row = rows[index] if present else {"card": "D7", "actor": "north"}
            card = Card.parse(row["card"])
            actor = Seat.parse(row["actor"])
            actions.append(
                CpSatActionVariablesV1(
                    f"PLAY-{index}",
                    model.new_constant(card.id),
                    model.new_constant(int(actor)),
                    model.new_constant(int(owners.get(card.id, actor))),
                    model.new_constant(int(bool(row.get("ekka", False)))),
                    model.new_constant(int(present)),
                )
            )
        winner = actions[0]
        observed = trick_rows.get(trick_index)
        if observed is not None and (trick_index + 1) * 4 <= len(rows):
            winner_seat = Seat.parse(observed["winner"])
            winning_card = Card.parse(observed["winningCard"])
            winner_position = next(
                position
                for position, row in enumerate(
                    rows[trick_index * 4 : trick_index * 4 + 4]
                )
                if Seat.parse(row["actor"]) == winner_seat
                and Card.parse(row["card"]) == winning_card
            )
            winner = actions[winner_position]
        result[trick_index] = CpSatTrickVariablesV1(trick_index, tuple(actions), winner)
    return result


def _visible_cards_by_observer(
    candidate: OrToolsNeutralCandidateV1,
) -> Mapping[Seat, frozenset[int]]:
    state = candidate.state
    hands = state.get("hands")
    play = dict(candidate.trace.get("play") or {})
    if not isinstance(hands, Mapping):
        hands = dict(play.get("decisionState") or {}).get("remainingHands")
    if not isinstance(hands, Mapping):
        return {}
    public: set[int] = set()
    for event in state.get("events", ()):
        if event.get("type") == "card_played":
            public.add(Card.parse(event["card"]).id)
    prefix_length = int(dict(play.get("decisionState") or {}).get("prefixLength", 0))
    for row in (play.get("plays") or ())[:prefix_length]:
        public.add(Card.parse(row["card"]).id)
    purchase = state.get("purchaseCard")
    deck_order = candidate.trace.get("deckOrder")
    if purchase is None and isinstance(deck_order, Sequence) and len(deck_order) > 20:
        purchase = deck_order[20]
    if purchase is not None:
        public.add(Card.parse(purchase).id)
    revealed_by_seat: dict[Seat, set[int]] = {seat: set() for seat in Seat}
    project_rows = state.get("projects")
    if not isinstance(project_rows, Sequence):
        project_rows = dict(candidate.trace.get("projects") or {}).get("projects") or ()
    for row in project_rows:
        cards = {Card.parse(card).id for card in (row.get("cards") or ())}
        if not cards or row.get("owner") is None:
            continue
        if row.get("revealed") is True:
            public.update(cards)
        else:
            revealed_by_seat[Seat.parse(row["owner"])].update(cards)
    return {
        observer: frozenset(
            public
            | revealed_by_seat[observer]
            | {Card.parse(card).id for card in hands[observer.name.lower()]}
        )
        for observer in Seat
    }


def _observer_scope_from_candidate(
    candidate: OrToolsNeutralCandidateV1,
    *,
    acting_observer: Seat,
    visible: Mapping[Seat, frozenset[int]],
) -> CanonicalObserverScopeV1:
    state = candidate.state
    if all(
        key in state for key in ("hands", "initialHands", "completedTricks", "events")
    ):
        return CanonicalObserverScopeV1.build(
            visibility_scope="complete_state",
            acting_observer=acting_observer,
            complete_states=(state,),
        )
    projection_hash = canonical_sha256(
        {
            "schemaVersion": 1,
            "visibilityScope": "complete_state",
            "actingObserver": acting_observer.name.lower(),
            "visibleCards": sorted(visible[acting_observer]),
            "decisionState": dict(dict(candidate.trace["play"])["decisionState"]),
        }
    )
    hidden_scope_hash = canonical_sha256(
        {
            "schemaVersion": 1,
            "candidateHash": candidate.sha256,
            "actingObserver": acting_observer.name.lower(),
        }
    )
    safe = {
        "schemaVersion": 1,
        "visibilityScope": "complete_state",
        "actingObserver": acting_observer.name.lower(),
        "classification": "public_fact",
        "observerProjectionHash": projection_hash,
        "hiddenWorldCount": 1,
        "hiddenWorldScopeHash": hidden_scope_hash,
    }
    return CanonicalObserverScopeV1(
        "complete_state",
        acting_observer,
        "public_fact",
        projection_hash,
        hidden_scope_hash,
        1,
        canonical_sha256(safe),
    )


def _constant_play_context(
    candidate: OrToolsNeutralCandidateV1,
    rules: RuleBook,
    *,
    complete: bool = False,
) -> _PlayQualificationContextV1:
    play, qualification_rows = _candidate_play_rows(candidate, complete=complete)
    decision_play = (
        dict(candidate.trace["play"])
        if complete and isinstance(candidate.trace.get("play"), Mapping)
        else play
    )
    raw_decision = decision_play.get("decisionState")
    if isinstance(raw_decision, Mapping):
        decision = dict(raw_decision)
    else:
        state_hands = candidate.state.get("hands") or {
            seat.name.lower(): [] for seat in Seat
        }
        current = candidate.state.get("currentTrick")
        current_rows = (
            list(current.get("plays", ())) if isinstance(current, Mapping) else []
        )
        legal_rows = []
        for row in candidate.state.get("legalActions", ()):
            if isinstance(row, Mapping) and row.get("card") is not None:
                legal_rows.append(
                    {
                        "card": Card.parse(row["card"]).code,
                        "ekka": bool(row.get("ekka", False)),
                    }
                )
        decision = {
            "prefixLength": len(qualification_rows),
            "remainingHands": {
                seat: [Card.parse(card).code for card in cards]
                for seat, cards in dict(state_hands).items()
            },
            "currentTrick": [
                {
                    "card": Card.parse(row["card"]).code,
                    "actor": row.get("actor") or row.get("seat"),
                    "ekka": bool(row.get("ekka", False)),
                }
                for row in current_rows
            ],
            "nextActor": candidate.state.get("currentPlayer"),
            "legalActions": [row["card"] for row in legal_rows],
            "legalPlayActions": legal_rows,
        }
    model = cp_model.CpModel()
    prefix_length = int(decision["prefixLength"])
    rows = list(play.get("plays") or ())

    def slot(index: int, *, present: bool | None = None) -> CpSatPlayPrefixActionV1:
        row = (
            rows[index]
            if index < len(rows)
            else {"card": "D7", "actor": "north", "ekka": False}
        )
        return CpSatPlayPrefixActionV1(
            model.new_constant(Card.parse(row["card"]).id),
            model.new_constant(int(Seat.parse(row["actor"]))),
            model.new_constant(int(bool(row.get("ekka", False)))),
            model.new_constant(
                int(index < prefix_length if present is None else present)
            ),
        )

    play_slots = tuple(slot(index) for index in range(32))
    current_rows = list(decision["currentTrick"])
    current_slots = tuple(
        CpSatPlayPrefixActionV1(
            model.new_constant(
                Card.parse(current_rows[index]["card"]).id
                if index < len(current_rows)
                else 0
            ),
            model.new_constant(
                int(Seat.parse(current_rows[index]["actor"]))
                if index < len(current_rows)
                else 0
            ),
            model.new_constant(
                int(bool(current_rows[index].get("ekka", False)))
                if index < len(current_rows)
                else 0
            ),
            model.new_constant(int(index < len(current_rows))),
        )
        for index in range(4)
    )
    next_actor = decision["nextActor"]
    trump = candidate.trace.get("trumpSuit")
    tricks = list(play.get("tricks") or ())
    score = dict(candidate.trace["score"])
    completed_count = prefix_length // 4
    current_raw = [0, 0]
    for trick in tricks[:completed_count]:
        current_raw[int(Seat.parse(trick["winner"]).team)] += int(trick["rawPoints"])
    comparison_complete = any(
        rules.project_domain.should_compare(index) for index in range(completed_count)
    )
    current_projects = [0, 0]
    if comparison_complete:
        for project in dict(candidate.trace.get("projects") or {}).get("projects", ()):
            if project.get("scoring") is True:
                current_projects[int(Team[str(project["team"]).upper()])] += int(
                    project["value"]
                )
    scoring = ScoringDomainIRV1.from_rules(rules.data)
    contract = Contract.parse(candidate.trace["contract"])
    current_game = [
        scoring.convert_raw(current_raw[int(team)], contract) for team in Team
    ]
    current_match = list(candidate.trace.get("matchScore", (0, 0)))
    if prefix_length == 32:
        current_game = list(score["cardGamePoints"])
        current_match = list(score["finalMatchScore"])
    observer_scope = None
    current_player = candidate.state.get("currentPlayer") or decision.get("nextActor")
    visible = _visible_cards_by_observer(candidate)
    if current_player is not None and visible:
        observer_scope = _observer_scope_from_candidate(
            candidate,
            acting_observer=Seat.parse(current_player),
            visible=visible,
        )
    base = CpSatPlayStatePredicateContextV1(
        model=model,
        context_id=f"candidate-{candidate.sha256[:16]}",
        play_slots=play_slots,
        prefix_length=model.new_constant(prefix_length),
        trick_index=model.new_constant(min(prefix_length // 4, 8)),
        next_actor=model.new_constant(int(Seat.parse(next_actor)) if next_actor else 4),
        remaining_hand_sizes={
            seat: model.new_constant(len(decision["remainingHands"][seat.name.lower()]))
            for seat in Seat
        },
        current_trick_slots=current_slots,
        current_trick_size=model.new_constant(len(current_rows)),
        current_trick_active=model.new_constant(int(bool(current_rows))),
        current_trick_leader=model.new_constant(
            int(Seat.parse(current_rows[0]["actor"])) if current_rows else 0
        ),
        current_trick_suit_led=model.new_constant(
            int(Card.parse(current_rows[0]["card"]).suit) if current_rows else 0
        ),
        trump_suit=model.new_constant(int(Suit.parse(trump)) if trump else 4),
        completed_trick_literals={
            index: model.new_constant(int(index < prefix_length // 4))
            for index in range(8)
        },
        completed_trick_winners={
            index: model.new_constant(
                int(Seat.parse(tricks[index]["winner"])) if index < len(tricks) else 0
            )
            for index in range(8)
        },
        legal_card_literals={
            index: model.new_constant(
                int(Card(index).code in set(decision["legalActions"]))
            )
            for index in range(32)
        },
        legal_action_literals={
            (index, ekka): model.new_constant(
                int(
                    any(
                        Card.parse(row["card"]).id == index
                        and bool(row.get("ekka", False)) == ekka
                        for row in decision["legalPlayActions"]
                    )
                )
            )
            for index in range(32)
            for ekka in (False, True)
        },
        phase=model.new_constant(
            list(GamePhase).index(
                GamePhase(
                    candidate.state.get(
                        "phase",
                        "scored" if prefix_length == 32 else "playing",
                    )
                )
            )
        ),
        match_scores={
            team: model.new_constant(current_match[int(team)]) for team in Team
        },
        raw_points={team: model.new_constant(current_raw[int(team)]) for team in Team},
        project_points={
            team: model.new_constant(current_projects[int(team)]) for team in Team
        },
        game_points={
            team: model.new_constant(current_game[int(team)]) for team in Team
        },
        card_visibility_literals=(
            {
                (observer, card): model.new_constant(int(card in visible[observer]))
                for observer in Seat
                for card in range(32)
            }
            if visible
            else None
        ),
        observer_scope=observer_scope,
    )
    return _PlayQualificationContextV1(
        base,
        _constant_general_tricks(
            candidate,
            model=model,
            play=play,
            rows=qualification_rows,
        ),
    )


def _native_play_context_v3(
    candidate: OrToolsNeutralCandidateV1,
    rules: RuleBook,
    *,
    dealer: Seat,
    complete: bool = False,
    observer_scope: CanonicalObserverScopeV1 | None = None,
) -> _PlayQualificationContextV1:
    """Rebind one retained play to the native transition model."""

    play, qualification_rows = _candidate_play_rows(candidate, complete=complete)
    decision_play = (
        dict(candidate.trace["play"])
        if complete and isinstance(candidate.trace.get("play"), Mapping)
        else play
    )
    decision = decision_play.get("decisionState")
    if not isinstance(decision, Mapping):
        raise ValueError("native play binding requires a canonical decision state")
    recorded_projects = candidate.trace.get("projects")
    if not isinstance(recorded_projects, Mapping):
        raise ValueError("native play score binding requires project declarations")
    prefix = candidate.trace.get("exactTracePrefix")
    spec = ExactTracePrefixV1.build(prefix) if isinstance(prefix, Mapping) else None
    rows = list(play.get("plays") or ())
    project_decision = dict(candidate.trace.get("exactNormalization") or {}).get("decisionKind") == "project"
    if not rows and not project_decision:
        raise ValueError("native play binding requires at least one play action")
    initial_hands = play.get("initialHands")
    if not isinstance(initial_hands, Mapping):
        initial_hands = dict(candidate.trace.get("play") or {}).get("initialHands")
    if not isinstance(initial_hands, Mapping):
        raise ValueError("native play binding requires canonical initial hands")
    ownership = {
        Card.parse(card): Seat.parse(seat)
        for seat, cards in initial_hands.items()
        for card in cards
    }
    if len(ownership) != 32:
        raise ValueError("native play binding requires ownership of all 32 cards")
    deck = candidate.trace.get("deckOrder")
    if not isinstance(deck, Sequence) or len(deck) != 32:
        raise ValueError("native play binding requires a canonical 32-card deck")
    layout = DealLayoutV1.build(rules.structural_domain, dealer=dealer)
    purchase_card = Card.parse(deck[layout.purchase_position])
    escalation = dict(dict(candidate.trace.get("escalation") or {}).get("observation") or {})
    build = OrToolsPlayTailStageV1(native_conversion=True).build_model(
        cards=tuple(Card(index) for index in range(32)),
        ownership=ownership,
        contract=Contract.parse(candidate.trace["contract"]),
        trump=(
            Suit.parse(candidate.trace["trumpSuit"])
            if candidate.trace.get("trumpSuit") is not None
            else None
        ),
        first_leader=rules.play_domain.resolve_first_leader(
            dealer=dealer, buyer=Seat.parse(candidate.trace["buyer"]),
            structural_domain=rules.structural_domain,
        ),
        play_domain=rules.play_domain,
        structural_domain=rules.structural_domain,
        scoring_domain=ScoringDomainIRV1.from_rules(rules.data),
        prefix=tuple(
            TailPrefixActionV1.build(
                row["actor"], row["card"], ekka=bool(row.get("ekka", False))
            )
            for row in rows
        ),
        locked=escalation.get("mode") == "locked",
        decision_prefix_length=int(decision["prefixLength"]),
        decision_actor=decision.get("nextActor"),
        decision_purchase_card=purchase_card,
        play_step_count=4 if project_decision and not complete else None,
        _export_predicate_contexts=True,
    )
    base = build.play_state_context
    model = build.model
    projects = attach_project_lifecycle_fragment_v1(
        model,
        owner_of_card=build.initial_owner_variables,
        fixed_card_owners=tuple(ownership[Card(index)] for index in range(32)),
        contract=Contract.parse(candidate.trace["contract"]),
        dealer=dealer,
        trump=Suit.parse(candidate.trace["trumpSuit"])
        if candidate.trace.get("trumpSuit") is not None else None,
        domain=rules.project_domain,
        name="state_projects",
        second_trick_leader=build._extract.winner_seats[0],
        include_declaration_actions=project_decision,
        compact_fixed_states=True,
        tighten_value_bounds=True,
    )
    project_binding = None
    if spec is not None:
        project_binding = _native_project_prefix_projection(
            build=build, spec=spec, fragment=projects, dealer=dealer,
            contract=Contract.parse(candidate.trace["contract"]),
            trump=Suit.parse(candidate.trace["trumpSuit"])
            if candidate.trace.get("trumpSuit") is not None else None,
            domain=rules.project_domain,
        )
        current_winning_team = tuple(
            model.new_bool_var(f"state_project_winning_team_{int(team)}") for team in Team
        )
        for team, literal in zip(Team, current_winning_team, strict=True):
            model.add(project_binding.winning_team == int(team)).only_enforce_if(literal)
            model.add(project_binding.winning_team != int(team)).only_enforce_if(literal.Not())
        # Prefix observations must not bind pending declarations or future awards.
        current_project_fragment = replace(
            projects,
            projects=tuple(replace(
                project,
                declared=project_binding.declared_now[project.project_id],
                declaration_order=project_binding.declaration_order_now[project.project_id],
                comparison_winner=project_binding.comparison_now[project.project_id],
                awarded=project_binding.awarded_now[project.project_id],
            ) for project in projects.projects),
            combined_awarded_value=project_binding.combined_awarded_value,
            winning_team=current_winning_team,
            baloot_team=project_binding.team_baloot,
            fragment_hash=canonical_sha256({
                "fragmentHash": projects.fragment_hash, "prefixHash": spec.sha256,
                "scope": "project_prefix",
            }),
        )
    else:
        declared = {
            str(row["projectId"]): bool(row["declared"])
            for row in recorded_projects.get("projects", ())
        }
        for project in projects.projects:
            model.add(project.declared == int(declared.get(project.project_id, False)))
        current_project_fragment = projects
    comparison_complete = model.new_bool_var("state_project_comparison_complete")
    model.add_max_equality(
        comparison_complete,
        [base.completed_trick_literals[index] for index in range(8)
         if rules.project_domain.should_compare(index)] or [0],
    )
    current_projects = {}
    for team in Team:
        current = model.new_int_var(
            0, 2_147_483_647, f"state_project_points_{int(team)}"
        )
        model.add(current == projects.combined_awarded_value[int(team)]).only_enforce_if(
            comparison_complete
        )
        model.add(current == 0).only_enforce_if(comparison_complete.Not())
        current_projects[team] = current
    starting_match = tuple(
        model.new_constant(int(value))
        for value in (
            prefix["matchScore"] if isinstance(prefix, Mapping)
            else candidate.trace.get("matchScore", (0, 0))
        )
    )
    match_scores = starting_match
    game_points = base.game_points
    if int(decision["prefixLength"]) == 32:
        baloot_points = tuple(
            model.new_int_var(0, projects.baloot_value, f"state_baloot_{int(team)}")
            for team in Team
        )
        for team in Team:
            model.add(baloot_points[int(team)]
                      == projects.baloot_value * projects.baloot_team[int(team)])
        actions = tuple(escalation.get("actions", ()))
        last_team = (
            Seat.parse(actions[-1].get("actor", actions[-1].get("seat"))).team
            if actions else None
        )
        scoring = attach_scoring_fragment_v1(
            model,
            ScoringDomainIRV1.from_rules(rules.data),
            CpSatScoringInputsV1(
                contract=model.new_constant(
                    int(Contract.parse(candidate.trace["contract"]).effective != Contract.SUN)
                ),
                buyer_seat=model.new_constant(int(Seat.parse(candidate.trace["buyer"]))),
                escalation=model.new_constant(
                    {"normal": 0, "double": 1, "triple": 2, "four": 3, "gahwa": 4}[
                        escalation.get("level", "normal")
                    ]
                ),
                last_escalating_team=model.new_constant(int(last_team) if last_team is not None else 2),
                raw_points=build.raw_team_variables,
                project_points=projects.combined_awarded_value,
                baloot_points=baloot_points,
                trick_winner_teams=build.winner_team_variables,
                starting_match_score=starting_match,
            ),
            source_hash=candidate.sha256,
            native_conversion=True,
        )
        match_scores = scoring.final_match_score
        game_points = dict(zip(Team, scoring.card_game_points, strict=True))
    base = replace(
        base,
        context_id=f"candidate-native-play-{candidate.sha256[:16]}",
        match_scores=dict(zip(Team, match_scores, strict=True)),
        project_points=current_projects,
        game_points=game_points,
    )
    # Observation truth is derived from the decision prefix, never its state projection.
    public_cards = {card: [] for card in range(32)}
    purchase_variable = model.new_int_var(0, 31, "state_visible_purchase_card")
    model.add(purchase_variable == purchase_card.id)
    for card in range(32):
        purchase_visible = model.new_bool_var(f"state_visible_purchase_{card}")
        model.add(purchase_variable == card).only_enforce_if(purchase_visible)
        model.add(purchase_variable != card).only_enforce_if(purchase_visible.Not())
        public_cards[card].append(purchase_visible)
        for index, slot in enumerate(base.play_slots[:int(decision["prefixLength"])]):
            matches = model.new_bool_var(f"state_visible_play_match_{index}_{card}")
            played = model.new_bool_var(f"state_visible_play_{index}_{card}")
            model.add(slot.card == card).only_enforce_if(matches)
            model.add(slot.card != card).only_enforce_if(matches.Not())
            model.add_min_equality(played, [matches, slot.present])
            public_cards[card].append(played)
    revealed_by_owner = {}
    for seat in Seat:
        turns = []
        for index, slot in enumerate(base.play_slots):
            if not rules.project_domain.should_reveal(index // 4):
                continue
            owner_turn = model.new_bool_var(f"state_project_reveal_actor_{int(seat)}_{index}")
            revealed_turn = model.new_bool_var(f"state_project_reveal_turn_{int(seat)}_{index}")
            model.add(slot.actor == int(seat)).only_enforce_if(owner_turn)
            model.add(slot.actor != int(seat)).only_enforce_if(owner_turn.Not())
            model.add_min_equality(revealed_turn, [owner_turn, slot.present])
            turns.append(revealed_turn)
        revealed = model.new_bool_var(f"state_project_revealed_owner_{int(seat)}")
        model.add_max_equality(revealed, turns or [0])
        revealed_by_owner[seat] = revealed
    for project in projects.projects:
        revealed = model.new_bool_var(f"state_project_revealed_{project.project_id}")
        if project_binding is not None:
            model.add(revealed == project_binding.revealed_now[project.project_id])
        else:
            model.add_min_equality(
                revealed, [project.comparison_winner if rules.project_domain.reveal_scope == "comparison_winner"
                           else project.awarded, comparison_complete, revealed_by_owner[project.owner]]
            )
        for card in project.cards:
            public_cards[card.id].append(revealed)
    public_visibility = {}
    for card, sources in public_cards.items():
        public = model.new_bool_var(f"state_visible_public_{card}")
        model.add_max_equality(public, sources)
        public_visibility[card] = public
    card_visibility = {}
    for seat in Seat:
        for card in range(32):
            owned = model.new_bool_var(f"state_visible_owned_{int(seat)}_{card}")
            visible_card = model.new_bool_var(f"state_visible_card_{int(seat)}_{card}")
            model.add(build.initial_owner_variables[card] == int(seat)).only_enforce_if(owned)
            model.add(build.initial_owner_variables[card] != int(seat)).only_enforce_if(owned.Not())
            model.add_max_equality(visible_card, [owned, public_visibility[card]])
            card_visibility[seat, card] = visible_card
    base = replace(base, card_visibility_literals=card_visibility, native_observer_scope=True)
    visible = _visible_cards_by_observer(candidate)
    current_player = candidate.state.get("currentPlayer") or decision.get("nextActor")
    if current_player is not None and visible:
        observer = Seat.parse(current_player)
        base = replace(
            base,
            observer_scope=observer_scope or _observer_scope_from_candidate(
                candidate,
                acting_observer=observer,
                visible=visible,
            ),
        )
    return _PlayQualificationContextV1(
        base, build.predicate_context.tricks,
        project_context=CpSatProjectPredicateContextV1(
            model, f"candidate-native-projects-{candidate.sha256[:12]}",
            current_project_fragment,
        ),
        native_build=build,
    )


def _constant_nonplay_state_context(
    candidate: OrToolsNeutralCandidateV1,
    *,
    native_bid: tuple[cp_model.CpModel, OrToolsNativeAuctionFragmentV1] | None = None,
    native_escalation: CpSatEscalationPredicateContextV1 | None = None,
    native_project: tuple[cp_model.CpModel, Seat, Mapping[str, cp_model.IntVar]] | None = None,
    native_card_visibility: tuple[
        Sequence[cp_model.IntVar], cp_model.IntVar,
        Sequence[tuple[Seat, int]], Sequence[cp_model.IntVar],
    ] | None = None,
) -> CpSatPlayStatePredicateContextV1:
    """Bind legacy state or a native non-play decision state."""

    if sum(value is not None for value in (native_bid, native_escalation, native_project)) > 1:
        raise ValueError("non-play state cannot bind two decision models")
    if native_card_visibility is not None and all(
        value is None for value in (native_bid, native_escalation, native_project)
    ):
        raise ValueError("native card visibility requires a deal-stage decision model")

    normalized = candidate.trace.get("exactNormalization")
    if not isinstance(normalized, Mapping):
        raise ValueError("non-play state qualification requires exact normalization")
    next_actor = normalized.get("nextActor")
    if next_actor is None:
        raise ValueError("non-play decision has no next actor")
    legal_actions = normalized.get("legalActions")
    if not isinstance(legal_actions, Sequence):
        raise ValueError("non-play decision has no canonical legal actions")
    state = candidate.state
    hands = state.get("hands")
    if not isinstance(hands, Mapping):
        raise ValueError("non-play decision state has no canonical hands")
    model = (
        native_bid[0] if native_bid is not None
        else native_escalation.model if native_escalation is not None
        else native_project[0] if native_project is not None
        else cp_model.CpModel()
    )

    def empty_slot(index: int) -> CpSatPlayPrefixActionV1:
        return CpSatPlayPrefixActionV1(
            model.new_constant(index % 32),
            model.new_constant(0),
            model.new_constant(0),
            model.new_constant(0),
        )

    if any(value is not None for value in (native_bid, native_escalation, native_project)):
        authored_prefix = candidate.trace.get("exactTracePrefix")
        scores = authored_prefix.get("matchScore") if isinstance(authored_prefix, Mapping) else None
        if (
            not isinstance(scores, (list, tuple))
            or len(scores) != 2
            or any(type(value) is not int or value < 0 for value in scores)
        ):
            raise ValueError("native non-play match score requires exact authored input")
        current_match = tuple(scores)
        if tuple(state.get("matchScore", ())) != current_match:
            raise ValueError("native match score disagrees with normalized decision")
    else:
        current_match = tuple(int(value) for value in state.get("matchScore", (0, 0)))
    visible = _visible_cards_by_observer(candidate)
    actor = Seat.parse(next_actor)
    observer_scope = (
        CanonicalObserverScopeV1.build(
            visibility_scope="complete_state",
            acting_observer=actor,
            complete_states=({**state, "currentPlayer": actor.name.lower()},),
        )
        if visible
        else None
    )
    if native_card_visibility is not None:
        owners, purchase_card, requested, played_cards = native_card_visibility
        if len(owners) != 32:
            raise ValueError("native card visibility requires 32 card owners")
        if any(
            variable.model_proto is not model.proto
            for variable in (*owners, purchase_card, *played_cards)
        ):
            raise ValueError("native card visibility variables belong to another model")
        native_visible = {}
        for observer, card in requested:
            owned = model.new_bool_var(f"visible_owned_{int(observer)}_{card}")
            visible_card = model.new_bool_var(f"visible_card_{int(observer)}_{card}")
            model.add(owners[card] == int(observer)).only_enforce_if(owned)
            model.add(owners[card] != int(observer)).only_enforce_if(owned.Not())
            sources = [owned]
            for index, source_card in enumerate((purchase_card, *played_cards)):
                public = model.new_bool_var(
                    f"visible_public_{int(observer)}_{card}_{index}"
                )
                model.add(source_card == card).only_enforce_if(public)
                model.add(source_card != card).only_enforce_if(public.Not())
                sources.append(public)
            model.add_max_equality(visible_card, sources)
            native_visible[observer, card] = visible_card
    else:
        native_visible = None
    phase = GamePhase(str(state["phase"]))
    if any(value is not None for value in (native_bid, native_escalation, native_project)):
        if state.get("completedTricks") or any(
            project.get("scoring") for project in state.get("projects", ())
        ):
            raise ValueError("native non-play score boundary is not before first trick")
        if native_bid is not None and state.get("contract") is not None:
            raise ValueError("native bid score boundary unexpectedly has a contract")
        if (native_escalation is not None or native_project is not None) and state.get("contract") is None:
            raise ValueError("native non-play game points require a contract")
    if native_bid is not None:
        fragment = native_bid[1]
        bid_state = fragment.states[fragment.prefix_length]
        model.add(bid_state["terminal"] == 0)
        native_count = model.new_int_var(0, len(fragment.legal_next_choice_literals), "bid_legal_action_count")
        model.add(native_count == sum(fragment.legal_next_choice_literals.values()))
        next_actor_var = bid_state["actor"]
        phase_var = model.new_constant(list(GamePhase).index(GamePhase.BIDDING))
        legal_count_var = native_count
    elif native_escalation is not None:
        native = native_escalation.native
        closure_action = native_escalation.stage.domain.closure_action
        choices = native["choiceLiterals"]
        if closure_action in {action for _seat, action, _mode in choices}:
            raise ValueError("native escalation closure overlaps announcement choices")
        closure = native["legal"][closure_action, None]
        actor_choices = [
            literal for (seat, _action, _mode), literal in choices.items()
            if seat == actor
        ]
        native_count = model.new_int_var(
            0, len(actor_choices) + 1, "escalation_legal_action_count"
        )
        model.add(native_count == sum(actor_choices) + closure)
        model.add(native_count >= 1)
        next_actor_var = model.new_constant(int(actor))
        phase_var = model.new_constant(list(GamePhase).index(GamePhase.FINAL_DEALT))
        legal_count_var = native_count
    elif native_project is not None:
        expected_actor = native_project[1]
        next_actor_var = model.new_int_var(0, 3, "project_decision_actor")
        model.add(next_actor_var == int(expected_actor))
        phase_var = model.new_constant(list(GamePhase).index(GamePhase.PLAYING))
        action_literals = tuple(native_project[2].values())
        native_count = model.new_int_var(0, len(action_literals), "project_legal_action_count")
        model.add(native_count == sum(action_literals))
        model.add(native_count >= 1)
        legal_count_var = native_count
    else:
        next_actor_var = model.new_constant(int(actor))
        phase_var = model.new_constant(list(GamePhase).index(phase))
        legal_count_var = model.new_constant(len(legal_actions))
    return CpSatPlayStatePredicateContextV1(
        model=model,
        context_id=(
            f"candidate-{'native-bid' if native_bid else 'native-escalation' if native_escalation else 'native-project' if native_project else 'nonplay'}-"
            f"{candidate.sha256[:16]}"
        ),
        play_slots=tuple(empty_slot(index) for index in range(32)),
        prefix_length=model.new_constant(0),
        trick_index=model.new_constant(0),
        next_actor=next_actor_var,
        remaining_hand_sizes={
            seat: model.new_constant(len(hands[seat.name.lower()])) for seat in Seat
        },
        current_trick_slots=tuple(empty_slot(index) for index in range(4)),
        current_trick_size=model.new_constant(0),
        current_trick_active=model.new_constant(0),
        current_trick_leader=model.new_constant(0),
        current_trick_suit_led=model.new_constant(0),
        trump_suit=model.new_constant(
            int(Suit.parse(state["trumpSuit"]))
            if state.get("trumpSuit") is not None
            else 4
        ),
        completed_trick_literals={index: model.new_constant(0) for index in range(8)},
        completed_trick_winners={index: model.new_constant(0) for index in range(8)},
        legal_card_literals={index: model.new_constant(0) for index in range(32)},
        legal_action_literals={
            (index, ekka): model.new_constant(0)
            for index in range(32)
            for ekka in (False, True)
        },
        phase=phase_var,
        match_scores={
            team: model.new_constant(current_match[int(team)]) for team in Team
        },
        raw_points={team: model.new_constant(0) for team in Team},
        project_points={team: model.new_constant(0) for team in Team},
        game_points={team: model.new_constant(0) for team in Team},
        generic_legal_action_count=legal_count_var,
        card_visibility_literals=(
            native_visible
            if native_visible is not None
            else
            {
                (observer, card): model.new_constant(int(card in visible[observer]))
                for observer in Seat
                for card in range(32)
            }
            if visible
            else None
        ),
        observer_scope=observer_scope,
    )


def _constant_outcome_context(
    candidate: OrToolsNeutralCandidateV1,
    buyer: Seat,
) -> CpSatOutcomePredicateContextV1:
    score = dict(candidate.trace["score"])
    model = cp_model.CpModel()
    match_winner = score["matchWinner"]
    return CpSatOutcomePredicateContextV1(
        model=model,
        context_id=f"candidate-outcome-{candidate.sha256[:12]}",
        source_stage="full_play",
        source_hash=candidate.stage_receipts["outcome"].evidence_hash,
        buyer_seat=model.new_constant(int(buyer)),
        round_winner=model.new_constant(int(Team[score["roundWinner"].upper()])),
        match_winner=model.new_constant(
            int(Team[match_winner.upper()]) if match_winner else 2
        ),
        raw_points=tuple(model.new_constant(value) for value in score["rawPoints"]),
        game_points=tuple(model.new_constant(value) for value in score["gamePoints"]),
        project_points=tuple(
            model.new_constant(value) for value in score["projectPoints"]
        ),
        final_match_score=tuple(
            model.new_constant(value) for value in score["finalMatchScore"]
        ),
        kabout=model.new_constant(int(score["kabout"])),
        gahwa=model.new_constant(int(score["gahwa"])),
        match_target=int(score["matchTarget"]),
    )


def _native_outcome_context_v3(
    candidate: OrToolsNeutralCandidateV1,
    rules: RuleBook,
    buyer: Seat,
) -> CpSatOutcomePredicateContextV1:
    trace = dict(candidate.trace)
    complete = trace.get("completeExactNormalization")
    complete_state = (
        dict(complete["state"])
        if isinstance(complete, Mapping) and isinstance(complete.get("state"), Mapping)
        else {}
    )
    # Terminal play, score and escalation must come from the same continuation.
    play_raw = trace.get("completePlay")
    construction_outcome = not isinstance(play_raw, Mapping) and isinstance(trace.get("constructionPlay"), Mapping)
    if not isinstance(play_raw, Mapping):
        complete_state = {}
        play_raw = trace.get("constructionPlay") if construction_outcome else trace.get("play")
    score = dict(trace["constructionScore"] if construction_outcome else trace["score"])
    escalation_stage = trace.get("constructionEscalation" if construction_outcome else "escalation")
    escalation_observation = (
        escalation_stage.get("observation")
        if isinstance(escalation_stage, Mapping)
        else None
    )
    if construction_outcome and not isinstance(escalation_observation, Mapping):
        raise ValueError("native construction outcome requires its terminal escalation")
    escalation = (
        {
            "level": complete_state.get("escalation", "normal"),
            "actions": list(complete_state.get("escalationHistory", ())),
        }
        if complete_state
        else dict(escalation_observation) if isinstance(escalation_observation, Mapping)
        else {"level": "normal", "actions": []}
    )
    actions = tuple(escalation.get("actions", ()))
    last_escalating_team = None
    if actions:
        last_action = actions[-1]
        last_escalating_team = (
            Team[str(last_action["team"]).upper()]
            if last_action.get("team") is not None
            else Seat.parse(last_action.get("actor", last_action.get("seat"))).team
        )
    if not isinstance(play_raw, Mapping):
        raise ValueError("native outcome binding requires a complete play trace")
    play = dict(play_raw)
    winner_rows = play.get("trickWinners")
    winners = (
        tuple(Seat.parse(item).team for item in winner_rows)
        if isinstance(winner_rows, Sequence) and not isinstance(winner_rows, str)
        else tuple(
            Seat.parse(row["winner"]).team for row in play.get("tricks", ())
        )
    )
    if len(winners) != 8:
        raise ValueError("native outcome binding requires eight trick winners")
    result = OrToolsScoringStageV1(
        ScoringDomainIRV1.from_rules(rules.data), native_conversion=True
    ).solve(
        contract=Contract.parse(
            complete_state.get("contract")
            or score.get("contract")
            or trace.get("contract")
        ),
        buyer=Seat.parse(complete_state.get("buyer") or score.get("buyer"))
        if complete_state.get("buyer") or score.get("buyer") else buyer,
        escalation=str(escalation["level"]),
        last_escalating_team=last_escalating_team,
        raw_points=tuple(int(item) for item in score["rawPoints"]),
        project_points=tuple(int(item) for item in score["projectPoints"]),
        baloot_points=tuple(int(item) for item in score.get("balootPoints", (0, 0))),
        trick_winner_teams=winners,
        starting_match_score=tuple(int(item) for item in trace.get("matchScore", (0, 0))),
    )
    if result.status != "satisfied" or result.context is None:
        raise ValueError("native outcome evidence cannot be rebound")
    return result.context


def _principal_variation_tricks(
    candidate: OrToolsNeutralCandidateV1,
    proof: OrToolsGraphProofOutcomeV1,
    rules: RuleBook,
    *,
    require_native: bool = False,
) -> tuple[PrincipalVariationTrickEvidenceV1, ...]:
    if not proof.principal_variation:
        return ()
    try:
        principal_actions = tuple(
            LegalPlayActionV1.from_code(code) for code in proof.principal_variation
        )
    except ValueError:
        return ()
    play = dict(candidate.trace["play"])
    decision_raw = play.get("decisionState")
    initial_raw = play.get("initialHands")
    rows = list(play.get("plays") or ())
    if not isinstance(decision_raw, Mapping) or not isinstance(initial_raw, Mapping):
        return ()
    decision = dict(decision_raw)
    prefix_length = int(decision["prefixLength"])
    if prefix_length > len(rows):
        return ()
    contract = Contract.parse(candidate.trace["contract"])
    trump_raw = candidate.trace.get("trumpSuit")
    trump = Suit.parse(trump_raw) if trump_raw else None
    ownership = {
        Card.parse(card): Seat.parse(seat)
        for seat, cards in initial_raw.items()
        for card in cards
    }
    if set(card.id for card in ownership) != set(range(32)):
        return ()
    prefix = [
        TailPrefixActionV1.build(
            row.get("seat") or row.get("actor"),
            row["card"],
            ekka=bool(row.get("ekka", False)),
        )
        for row in rows[:prefix_length]
    ]
    first_leader_raw = play.get("firstLeader") or (
        rows[0].get("seat") or rows[0].get("actor") if rows else None
    )
    if first_leader_raw is None:
        return ()
    stage = OrToolsPlayTailStageV1(native_conversion=require_native)

    def solve_prefix() -> OrToolsPlayTailResultV1:
        return stage.solve(
            cards=tuple(sorted(ownership, key=lambda card: card.id)),
            ownership=ownership,
            contract=contract,
            trump=trump,
            first_leader=Seat.parse(first_leader_raw),
            play_domain=rules.play_domain,
            structural_domain=rules.structural_domain,
            scoring_domain=ScoringDomainIRV1.from_rules(rules.data),
            prefix=tuple(prefix),
            locked=bool(candidate.state.get("locked", False)),
            decision_prefix_length=len(prefix),
            maximum_time_seconds=5.0,
        )

    try:
        solved = solve_prefix()
    except (PlayTailSolveStopped, ValueError):
        return ()
    for action in principal_actions:
        actor = solved.next_actor
        if actor is None or action not in {
            LegalPlayActionV1(item.card, item.ekka)
            for item in solved.legal_play_actions
        }:
            return ()
        prefix.append(TailPrefixActionV1(actor, action.card, action.ekka))
        try:
            solved = solve_prefix()
        except (PlayTailSolveStopped, ValueError):
            return ()
    if len(prefix) % 4:
        return ()
    first_trick = prefix_length // 4
    completed = []
    for trick_index in range(first_trick, len(prefix) // 4):
        trick = prefix[trick_index * 4 : trick_index * 4 + 4]
        winner = solved.trick_winners[trick_index]
        completed.append(
            PrincipalVariationTrickEvidenceV1.build(
                trick_index=trick_index,
                plays=tuple(
                    PrincipalVariationPlayEvidenceV1.build(
                        LegalPlayActionV1(item.card, item.ekka).code,
                        item.actor.name.lower(),
                    )
                    for item in trick
                ),
                winner_position=next(
                    index for index, item in enumerate(trick) if item.actor == winner
                ),
            )
        )
    return tuple(completed)


def _proof_root_actions(
    candidate: OrToolsNeutralCandidateV1,
    proof: OrToolsGraphProofOutcomeV1,
    model: cp_model.CpModel,
    *,
    require_native: bool = False,
) -> tuple[CpSatActionVariablesV1, ...]:
    raw_play = candidate.trace.get("play")
    if not isinstance(raw_play, Mapping):
        return ()
    play = dict(raw_play)
    if not isinstance(play.get("decisionState"), Mapping):
        return ()
    decision = dict(play["decisionState"])
    actor_raw = candidate.state.get("currentPlayer") or decision.get("nextActor")
    if actor_raw is None:
        return ()
    actor = Seat.parse(actor_raw)
    owners = _initial_owner_by_card(candidate, play)
    result = []
    winning_variables, value_variables = _proof_root_variables(
        proof, model, require_native=require_native
    )

    def fixed(
        name: str,
        value: int,
        lower: int,
        upper: int,
    ) -> Any:
        if not require_native:
            return model.new_constant(value)
        variable = model.new_int_var(lower, upper, name)
        model.add(variable == value)
        return variable

    for index, row in enumerate(proof.root_actions):
        try:
            action = LegalPlayActionV1.from_code(row.action_code)
        except ValueError:
            return ()
        result.append(
            CpSatActionVariablesV1(
                f"ROOT-{row.action_id}",
                fixed(
                    f"qualification_root_{index}_card",
                    action.card.id,
                    0,
                    31,
                ),
                fixed(
                    f"qualification_root_{index}_actor",
                    int(actor),
                    0,
                    3,
                ),
                fixed(
                    f"qualification_root_{index}_owner",
                    int(owners.get(action.card.id, actor)),
                    0,
                    3,
                ),
                fixed(
                    f"qualification_root_{index}_ekka",
                    int(action.ekka),
                    0,
                    1,
                ),
                fixed(f"qualification_root_{index}_applicable", 1, 0, 1),
                winning_variables[index],
                value_variables[index],
            )
        )
    return tuple(result)


def _proof_root_variables(
    proof: OrToolsGraphProofOutcomeV1,
    model: cp_model.CpModel,
    *,
    require_native: bool = False,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    variables = {
        variable.name: index for index, variable in enumerate(model.proto.variables)
    }
    raw_bindings = proof.model_document().get("rootActionVariables")
    bindings = (
        {
            (int(item["actionId"]), str(item["action"])): dict(item)
            for item in raw_bindings
        }
        if isinstance(raw_bindings, list)
        else {}
    )
    winning = []
    values = []
    for index, row in enumerate(proof.root_actions):
        binding = bindings.get((row.action_id, row.action_code), {})
        winning_index = variables.get(
            str(binding.get("winningVariable") or f"root_wins_{index}")
        )
        value_index = variables.get(str(binding.get("valueVariable") or ""))
        if require_native and (winning_index is None or value_index is None):
            raise ValueError(
                "native proof model is missing root-action variable bindings: "
                f"{row.action_id}:{row.action_code}"
            )
        winning.append(
            model.get_bool_var_from_proto_index(winning_index)
            if winning_index is not None
            else model.new_constant(int(row.satisfies))
        )
        values.append(
            model.get_int_var_from_proto_index(value_index)
            if value_index is not None
            else model.new_constant(row.value)
        )
    return tuple(winning), tuple(values)


def _registered_tactical_evidence(
    candidate: OrToolsNeutralCandidateV1,
    proof: OrToolsGraphProofOutcomeV1,
) -> tuple[RegisteredTacticalEvidenceV1, ...]:
    complete = proof.solver_status == "optimal" and proof.completion_status in {
        "exact_complete",
        "bounded_complete",
        "policy_complete",
    }
    if not complete or not proof.root_actions:
        return ()
    raw_play = candidate.trace.get("play")
    if not isinstance(raw_play, Mapping):
        return ()
    play = dict(raw_play)
    decision = play.get("decisionState")
    actor_raw = candidate.state.get("currentPlayer") or (
        dict(decision).get("nextActor") if isinstance(decision, Mapping) else None
    )
    if actor_raw is None:
        return ()
    actor = Seat.parse(actor_raw)
    verified = proof.winning_root_action_count == 1
    return (
        RegisteredTacticalEvidenceV1.build(
            detector_id="only_winning_move",
            detector_version="1.0.0",
            evidence_id=f"TACTIC-{proof.sha256[:20].upper()}",
            applicable=True,
            verified=verified,
            score=100 if verified else 0,
            acting_seat=actor.name.lower(),
            observer=actor.name.lower(),
            target_team=actor.team.name.lower(),
            causal_proof_hash=proof.sha256 if verified else None,
            source_proof_hash=proof.sha256,
        ),
    )


def _native_proof_tactic_configurer_v3(
    *,
    scenario: ResearchScenarioV1,
    candidate: OrToolsNeutralCandidateV1,
    graph: ProofGraphV1,
    request: ScenarioProofRequestV1,
    rules: RuleBook,
    manifest: ScenarioPredicateCatalogLoweringManifestV2,
    tactical_detector_profile: TacticalDetectorProfileV1 | None = None,
) -> Callable[[Any], Mapping[str, Any]] | None:
    """Lower hard claims and selected-root tactic reports in the proof model."""

    root = next(node for node in graph.nodes if node.node_id == graph.root_node_id)
    forced_root = len(root.edges) == 1
    predicates = tuple(
        predicate
        for predicate in scenario.predicates()
        if (predicate.hardness == "hard" or (
            predicate.kind in {"tactic.applicable", "tactic.verified"}
            and (forced_root or predicate.arguments["kind"] in _ROOT_VALUE_TACTIC_IDS_V3
                 or predicate.arguments["kind"] in NATIVE_PROFILE_TACTIC_IDS)
            and (predicate.arguments["kind"] not in _KABOUT_TACTIC_IDS_V3 or request.utility == "kabout")
        ))
        and manifest.definition(predicate.kind).stage in {"proof", "tactics"}
    )
    if not predicates:
        return None
    validate_native_delayed_payoff_scope_v1(scenario, request, tactical_detector_profile)
    if root.actor_seat is None or root.actor_team is None:
        raise ValueError("native proof qualification requires an actor-owned root")
    actor = Seat.parse(root.actor_seat)
    root_team = Team[str(root.actor_team).upper()]
    target_team = Team[str(request.target_team).upper()]
    completion = dict(dict(candidate.trace.get("auction") or {}).get("completion") or {})
    contract_raw = completion.get("contract") or candidate.state.get("contract")
    trump_raw = completion.get("trumpSuit") or candidate.state.get("trumpSuit")
    root_only = all(item.kind in {"proof.winning_root_action_count",
        "proof.winning_continuation_line_count"} or (
            item.kind in {"tactic.applicable", "tactic.verified"}
            and (forced_root or item.arguments["kind"] in _ROOT_VALUE_TACTIC_IDS_V3)
        ) for item in predicates)
    if (contract_raw is None or graph.tactic_context is None) and not root_only:
        raise ValueError("native proof qualification requires the solved contract")
    contract = Contract.parse(contract_raw) if contract_raw is not None else None
    trump = Suit.parse(trump_raw) if trump_raw is not None else None
    tactic_context = dict(graph.tactic_context or {})
    if contract is not None and graph.tactic_context is not None:
        expected_trump = int(trump) if contract.effective is Contract.HOKUM and trump is not None else None
        if (contract.effective is Contract.HOKUM and trump is None) or tactic_context.get("trumpSuit") != expected_trump:
            raise ValueError("native proof trump binding does not match the solved contract")
    base_tricks = tuple(int(value) for value in tactic_context.get("baseTricks", (0, 0)))
    if len(base_tricks) != 2:
        raise ValueError("native proof qualification requires two base trick counts")
    preplay_kabout = graph.tactic_context is None and any(
        item.kind in {"tactic.applicable", "tactic.verified"}
        and item.arguments["kind"] in _KABOUT_TACTIC_IDS_V3 for item in predicates
    )
    if preplay_kabout:
        phase = {"bid": "bidding", "escalation": "final_dealt"}.get(scenario.decision_point["kind"])
        if (phase is None or candidate.state.get("phase") != phase
                or type(candidate.state.get("completedTricks")) is not list
                or candidate.state["completedTricks"]
                or candidate.state.get("currentTrick", {}) is not None):
            raise ValueError("native pre-play Kabout requires an unplayed normalized state")

    def configure(bindings: Any) -> Mapping[str, Any]:
        # Pre-root public cards cannot be an extraction caused by this choice.
        trump_suffix = tuple(zip(bindings.principal_action_card_variables,
            bindings.principal_action_actor_variables, bindings.principal_action_present_variables,
            bindings.principal_action_trick_indexes, strict=True)) if any(
                item.kind in {"tactic.applicable", "tactic.verified"}
                and item.arguments["kind"] == "trump_control" for item in predicates) else ()
        if any(item.kind == 'proof.principal_variation_trick_team_rank_count' for item in predicates):
            trick = candidate.state.get('currentTrick') or {}
            if trick.get('plays'):
                bindings = bindings.with_current_trick(trick['index'], tuple(
                    (Card.parse(play['card']).id, int(Seat.parse(play.get('actor') or play.get('seat'))))
                    for play in trick['plays']))
        def fixed_bool(name: str, value: bool) -> Any:
            variable = bindings.model.new_bool_var(name)
            bindings.model.add(variable == int(value))
            return variable

        root_actions = tuple(
            SimpleNamespace(actor=bindings.root_actor_variables[index],
                            winning=bindings.root_winning_variables[index],
                            value=bindings.root_value_variables[index])
            if root_only else CpSatActionVariablesV1(
                f"ROOT-{edge.action_id}",
                bindings.root_card_variables[index],
                bindings.root_actor_variables[index],
                bindings.root_owner_variables[index],
                fixed_bool(f"native_root_ekka_{index}", edge.action_id >= 32),
                fixed_bool(f"native_root_present_{index}", True),
                bindings.root_winning_variables[index],
                bindings.root_value_variables[index],
            )
            for index, edge in enumerate(root.edges)
        )
        proof_plan = SimpleNamespace(
            completion_status="exact_complete",
            solver_status="optimal",
            root_actions=tuple(
                SimpleNamespace(action_code=edge.action_code, satisfies=False)
                for edge in root.edges
            ),
            winning_root_action_count=None,
            selected_action=request.selected_action,
            sha256=graph.sha256,
            principal_variation=(),
        )
        base_trick_variables = bindings.base_trick_variables
        raw_payoffs = ()
        delayed_control = (isinstance(tactical_detector_profile, TacticalDetectorProfileV1)
            and tactical_detector_profile.native_delayed_gain_metric is not None
            and len(root.edges) > 1
            and any(item.kind in {"tactic.applicable", "tactic.verified"}
                and item.arguments["kind"] in DELAYED_PAYOFF_TACTIC_IDS for item in predicates))
        if (request.mode == "exact" and request.visibility == "complete_state" and contract is not None
                and any(item.kind in {"tactic.applicable", "tactic.verified"} and (
                        item.arguments["kind"] in {"partner_enablement", "entry_unblocking"} and len(root.edges) > 1
                        or item.arguments["kind"] in DELAYED_PAYOFF_TACTIC_IDS
                        and (delayed_control or item.kind == "tactic.verified" and item.hardness == "report"))
                        for item in predicates)
                and all(node.actor_seat is not None and all(edge.play_trick_index is not None for edge in node.edges)
                        for node in graph.nodes if not node.is_leaf)):
            current_trick = candidate.state.get("currentTrick") or {}
            raw_payoffs = bindings.bind_root_raw_payoffs(
                tuple(rules.play_domain.points(Card(card), contract, trump) for card in range(32)),
                ScoringDomainIRV1.from_rules(rules.data).last_trick_bonus_raw,
                tuple(Card.parse(play["card"]).id for play in current_trick.get("plays", ())),
                **({"first_leader": int(Seat.parse(current_trick["leader"])) if current_trick.get("plays") else int(actor)}
                   if len(root.edges) > 1 and (delayed_control or any(item.kind in {"tactic.applicable", "tactic.verified"}
                       and item.arguments["kind"] == "entry_unblocking" for item in predicates)) else {}),
                **({"delayed_control": True} if delayed_control else {}),
            )
        if preplay_kabout:
            # No trick has occurred at a validated bid/escalation root.
            base_trick_variables = tuple(bindings.model.new_int_var(0, 0,
                f"native_preplay_base_tricks_{index}") for index in range(2))
        context = SimpleNamespace(
            model=bindings.model,
            context_id="native-proof-" + graph.sha256[:16],
            sha256=canonical_sha256(
                {
                    "graphHash": graph.sha256,
                    "predicateIds": [item.predicate_id for item in predicates],
                    **({"tacticalDetectorProfileHash": tactical_detector_profile.sha256} if tactical_detector_profile is not None else {}),
                }
            ),
            proof=proof_plan,
            native_proof_bindings=True,
            native_cp_sat_authority=True,
            tactical_detector_profile=tactical_detector_profile,
            root_raw_payoff_variables=raw_payoffs,
            principal_variation_tricks=(),
            tactical_evidence=(),
            root_actions=root_actions,
            root_winning_variables=bindings.root_winning_variables,
            root_value_variables=bindings.root_value_variables,
            root_project_consequence_variables=bindings.root_project_consequence_variables,
            project_consequence_scale=graph.utility_scale,
            causal_gain_scale=graph.utility_scale,
            causal_gain_terminal_awarded_scope=(request.mode == "exact"
                and request.visibility == "complete_state" and graph.scope == "exact"
                and request.utility == "awarded_point_margin"),
            root_actor_seat=int(actor),
            root_actor_team=int(root_team),
            proof_utility=request.utility,
            decision_kind=str(scenario.decision_point["kind"]),
            selected_root_action_index=bindings.selected_root_action_index,
            root_actor_is_target=root_team == target_team,
            root_actor_is_target_variable=bindings.root_actor_is_target_variable,
            target_seat_variable=bindings.target_seat_variable,
            card_strength_by_id=tuple(
                rules.play_domain.strength(Card(card_id), contract, trump)
                for card_id in (range(32) if contract is not None else ())
            ),
            first_principal_trick_winner_team=None,
            first_principal_trick_winner_team_variable=(
                bindings.first_principal_trick_winner_team_variable
            ),
            trump_suit=trump,
            trump_suit_variable=bindings.trump_suit_variable,
            base_tricks=(base_tricks[0], base_tricks[1]),
            base_trick_variables=base_trick_variables,
            principal_trick_winner_seat_variables=(
                bindings.principal_trick_winner_seat_variables
            ),
            principal_trick_winning_card_variables=(
                bindings.principal_trick_winning_card_variables
            ),
            principal_trick_indexes=bindings.principal_trick_indexes,
            principal_action_card_variables=bindings.principal_action_card_variables,
            principal_action_actor_variables=(
                bindings.principal_action_actor_variables
            ),
            principal_action_present_variables=(
                bindings.principal_action_present_variables
            ),
            principal_action_trick_indexes=bindings.principal_action_trick_indexes,
            trump_control_suffix_actions=trump_suffix,
            principal_variation_has_continuation_variable=(
                bindings.principal_variation_has_continuation_variable
            ),
            proof_complete=bindings.proof_complete_variable,
            continuation_line_digits=bindings.continuation_line_digits,
            continuation_line_universe=bindings.continuation_line_universe,
        )
        return {
            predicate.predicate_id: lower_native_proof_predicate_v3(
                predicate, context, manifest=manifest,
                root_selection_variables=bindings.root_selection_variables,
            )
            for predicate in predicates
        }

    return configure


def _proof_stopping_variables(proof, model):
    certificate = proof.model_document().get("opponentStopCertificate")
    if (not isinstance(certificate, Mapping) or certificate.get("completionStatus") != "exact_complete"
            or certificate.get("solverStatus") != "optimal"):
        raise ValueError("native opponent stopping certificate is incomplete or unavailable")
    rows = certificate.get("rootActionVariables")
    if rows != [{"action": row.action_code, "actionId": row.action_id,
            "valueVariable": f"opponent_stop_root_value_{index}",
            "winningVariable": f"opponent_stop_root_wins_{index}"}
            for index, row in enumerate(proof.root_actions)]:
        raise ValueError("native opponent stopping root identity disagrees with proof")
    result = []
    for row in rows:
        for variable_field in ("valueVariable", "winningVariable"):
            indices = [index for index, variable in enumerate(model.proto.variables) if variable.name == row[variable_field]]
            if len(indices) != 1:
                raise ValueError("native opponent stopping CP variables are unavailable")
            if variable_field == "winningVariable":
                result.append(model.get_bool_var_from_proto_index(indices[0]))
    return tuple(result)


def _owned_actor_observer_scope_v3(candidate, proof, rules, information_graph):
    if (not isinstance(information_graph, ActorInformationSetGraphV1)
            or information_graph.observer_scope is None or proof is None
            or proof.completion_status not in {"exact_complete", "proof_failed"}
            or proof.graph_hash != information_graph.to_dict()["sha256"]
            or information_graph.analysis_scope is not None
            or information_graph.hidden_worlds.get("complete") is not True):
        raise ValueError("native observer scope requires an owned exhaustive actor proof")
    observer_scope = information_graph.observer_scope
    binding = information_graph.model_document.get("observerScopeBinding")
    if (binding != {"scope": observer_scope.to_safe_dict(),
            "sourceStateHash": canonical_sha256(candidate.state),
            "exactStateId": candidate.exact_state_id, "rulesHash": rules.sha256,
            "graphWorldCount": len(information_graph.problem.worlds)}
            or observer_scope.visibility_scope != "actor_information_set"
            or observer_scope.acting_observer != Seat.parse(candidate.state.get("currentPlayer"))
            or information_graph.hidden_worlds.get("count") != len(information_graph.problem.worlds)
            or proof.hidden_worlds != information_graph.hidden_worlds):
        raise ValueError("native observer scope disagrees with candidate or proof ownership")
    return observer_scope


def _proof_stage_bindings_v1(
    candidate: OrToolsNeutralCandidateV1,
    proof: OrToolsGraphProofOutcomeV1 | None,
    rules: RuleBook,
    *,
    require_native: bool = False,
    proof_request: ScenarioProofRequestV1 | None = None,
    information_graph: ActorInformationSetGraphV1 | None = None,
    tactical_detector_profile: TacticalDetectorProfileV1 | None = None,
) -> dict[str, OrToolsStageContextBindingV1]:
    if proof_request is not None:
        if not isinstance(proof_request, ScenarioProofRequestV1):
            raise TypeError("native proof request binding requires ScenarioProofRequestV1")
        if not require_native:
            raise ValueError("proof request binding requires native CP-SAT authority")
        ScenarioProofRequestV1.build(proof_request.to_dict())
    if proof is None:
        return {}
    proof_receipt = candidate.stage_receipts.get("proof")
    if proof_receipt is None:
        raise ValueError("proof outcome lacks candidate stage receipt")
    proof_model = cp_model.CpModel()
    model_document = proof.model_document()
    if proof_request is not None and (
        proof_request.sha256 != proof.request_hash
        or model_document.get("utility") not in {None, proof_request.utility}
    ):
        raise ValueError("native proof request or utility binding disagrees with proof")
    native_actor_roots = (
        require_native
        and proof.proof_engine_id == "ortools_actor_information_set_cegis_v1"
        and proof.completion_status in {"exact_complete", "proof_failed"}
        and proof.solver_status == "optimal"
        and model_document.get("qualificationSource") == "cp_sat_cegis_root_certificates_v2"
    )
    if native_actor_roots and (
        proof_receipt.evidence_hash != proof.sha256
        or proof_receipt.model_hash != proof.model_hash
        or canonical_sha256(model_document) != proof.model_hash
        or model_document.get("analysisScope") is not None
        or not proof.hidden_worlds
        or proof.hidden_worlds.get("complete") is not True
    ):
        raise ValueError("native actor root bindings do not own exhaustive proof evidence")
    native_public_history = native_actor_roots and proof_request is not None and (
        proof_request.utility == "kabout" or isinstance(information_graph, ActorInformationSetGraphV1)
        and information_graph.decision_kind == "play")
    if native_public_history:
        _owned_actor_observer_scope_v3(candidate, proof, rules, information_graph)
    proto_text = model_document.get("modelProto")
    if proto_text and (
        type(proto_text) is not str
        or not proof_model.proto.parse_text_format(proto_text)
    ):
        raise ValueError("proof CP-SAT model could not be reconstructed")
    root_stopping = ()
    certificate = model_document.get("opponentStopCertificate")
    if native_public_history and certificate is not None:
        expected_scope = {"graphHash": proof.graph_hash, "problemHash": information_graph.problem.sha256,
            "requestHash": proof_request.sha256, "qualificationModelHash": model_document.get("qualificationModelHash"),
            "worldIds": [world.world_id for world in information_graph.problem.worlds],
            "roots": [[row.action_code, row.action_id] for row in proof.root_actions],
            "transform": "swap_policy_roles_negate_native_kabout_v1"}
        if (not isinstance(certificate, Mapping) or certificate.get("scope") != expected_scope
                or certificate.get("budgetHash") != model_document.get("budgetHash")
                or certificate.get("sha256") != canonical_sha256({key: value
                    for key, value in certificate.items() if key != "sha256"})):
            raise ValueError("native opponent stopping ownership disagrees with graph or request")
        if (certificate.get("completionStatus") == "exact_complete"
                and certificate.get("modelHash") != canonical_sha256({
                    "core": certificate.get("modelCore"), "modelProto": proto_text})):
            raise ValueError("native opponent stopping certificate model disagrees with proof")
        if certificate.get("completionStatus") == "exact_complete":
            core, roots = certificate.get("modelCore"), certificate.get("roots")
            if (not isinstance(core, Mapping) or not isinstance(roots, list)
                    or len(roots) != len(proof.root_actions)
                    or any(not isinstance(row, Mapping) or row.get("completionStatus") != "exact"
                        or type(row.get("robustValue")) is not int or row["robustValue"] not in {-1, 1}
                        for row in roots)
                    or core.get("problemHash") != certificate.get("problemHash")
                    or core.get("successThreshold") != 1 or core.get("claim") != "winning_root_action_count"
                    or core.get("roots") != [{"action": row.action_code, "robustValue": source["robustValue"],
                        "sourceModelHash": source.get("modelHash"), "valueVariable": f"opponent_stop_root_value_{index}",
                        "winningVariable": f"opponent_stop_root_wins_{index}"}
                        for index, (row, source) in enumerate(zip(proof.root_actions, roots))]
                    or [row.get("rootAction") for row in roots] != [row.action_code for row in proof.root_actions]):
                raise ValueError("native opponent stopping root certificates disagree with model")
        prefix = certificate.get("sourcePrefix", {})
        if (not isinstance(prefix, Mapping)
                or type(prefix.get("variables")) is not int or type(prefix.get("constraints")) is not int
                or not 0 < prefix["variables"] <= len(proof_model.proto.variables)
                or not 0 < prefix["constraints"] <= len(proof_model.proto.constraints)
                or prefix.get("sha256") != canonical_sha256({
                    "variables": list(map(str, tuple(proof_model.proto.variables)[:prefix["variables"]])),
                    "constraints": list(map(str, tuple(proof_model.proto.constraints)[:prefix["constraints"]]))})):
            raise ValueError("native opponent stopping source model disagrees with proof")
        root_stopping = _proof_stopping_variables(proof, proof_model)
    tricks = _principal_variation_tricks(
        candidate, proof, rules, require_native=require_native
    )
    base = ProofTacticPredicateContextV1(
        proof_model,
        f"candidate-proof-{candidate.sha256[:12]}",
        proof,
        tricks,
        _registered_tactical_evidence(candidate, proof),
    )
    root_actions = _proof_root_actions(
        candidate, proof, proof_model, require_native=require_native
    )
    selected_index = next(
        (
            index
            for index, row in enumerate(proof.root_actions)
            if row.action_code == proof.selected_action
        ),
        None,
    )
    raw_play = candidate.trace.get("play")
    decision = (
        dict(raw_play).get("decisionState")
        if isinstance(raw_play, Mapping)
        else None
    )
    actor_raw = candidate.state.get("currentPlayer") or (
        dict(decision).get("nextActor") if isinstance(decision, Mapping) else None
    )
    actor_raw = model_document.get("rootActorSeat") or actor_raw
    actor = Seat.parse(actor_raw) if actor_raw is not None else None
    root_team_raw = model_document.get("rootActorTeam")
    root_team = (
        Team[str(root_team_raw).upper()]
        if root_team_raw is not None
        else actor.team if actor is not None else None
    )
    target_raw = model_document.get("targetTeam")
    target = Team[str(target_raw).upper()] if target_raw is not None else None
    completion = dict(dict(candidate.trace.get("auction") or {}).get("completion") or {})
    contract_raw = completion.get("contract") or candidate.state.get("contract")
    trump_raw = completion.get("trumpSuit") or candidate.state.get("trumpSuit")
    if native_public_history:
        contract_raw, trump_raw = candidate.state.get("contract"), candidate.state.get("trumpSuit")
    strengths: tuple[int, ...] = ()
    trump = None
    if contract_raw is not None:
        contract = Contract.parse(contract_raw)
        trump = Suit.parse(trump_raw) if trump_raw is not None else None
        play_domain = PlayDomainIRV1.from_rules(rules.data)
        strengths = tuple(
            play_domain.strength(Card(card_id), contract, trump)
            for card_id in range(32)
        )
    first_winner = (
        Seat.parse(min(tricks, key=lambda item: item.trick_index).winner.actor_seat).team
        if tricks
        else None
    )
    base_tricks = [0, 0]
    if not native_public_history and isinstance(raw_play, Mapping) and isinstance(decision, Mapping):
        completed_count = int(dict(decision).get("prefixLength", 0)) // 4
        for trick in list(dict(raw_play).get("tricks") or ())[:completed_count]:
            if isinstance(trick, Mapping) and trick.get("winner") is not None:
                base_tricks[int(Seat.parse(trick["winner"]).team)] += 1
    root_winning, root_values = _proof_root_variables(
        proof, proof_model, require_native=require_native
    )

    def bind_int(
        name: str,
        value: int,
        lower: int,
        upper: int,
    ) -> Any:
        variable = proof_model.new_int_var(lower, upper, name)
        proof_model.add(variable == value)
        return variable

    base_trick_variables = tuple(
        proof_model.new_int_var(0, 8, f"qualification_base_tricks_{team}") if native_public_history
        else bind_int(f"qualification_base_tricks_{team}", value, 0, 8)
        for team, value in enumerate(base_tricks)
    )
    if native_public_history:
        completed = candidate.state.get("completedTricks")
        current = candidate.state.get("currentTrick")
        if (type(completed) is not list or len(completed) > 8 or not isinstance(current, Mapping)
                or type(current.get("index")) is not int or current["index"] != len(completed)):
            raise ValueError("native public trick history is unavailable or inconsistent")
        team_winners = [[], []]
        for index, trick in enumerate(completed):
            if (not isinstance(trick, Mapping) or type(trick.get("index")) is not int or trick["index"] != index
                    or type(trick.get("winner")) is not str or trick["winner"] not in {seat.name.lower() for seat in Seat}
                    or type(trick.get("plays")) is not list or len(trick["plays"]) != 4):
                raise ValueError("native public trick winner is unavailable or invalid")
            winner = bind_int(f"qualification_base_winner_seat_{index}", int(Seat.parse(trick["winner"])), 0, 3)
            winner_team = proof_model.new_int_var(0, 1, f"qualification_base_winner_team_{index}")
            proof_model.add_modulo_equality(winner_team, winner, 2)
            for team in range(2):
                won = proof_model.new_bool_var(f"qualification_base_won_{team}_{index}")
                proof_model.add(winner_team == team).only_enforce_if(won)
                proof_model.add(winner_team != team).only_enforce_if(won.Not())
                team_winners[team].append(won)
        for team, count in enumerate(base_trick_variables):
            proof_model.add(count == sum(team_winners[team]))
    ordered_tricks = tuple(sorted(tricks, key=lambda item: item.trick_index))
    principal_winner_variables = tuple(
        bind_int(
            f"qualification_principal_winner_seat_{index}",
            int(Seat.parse(trick.winner.actor_seat)),
            0,
            3,
        )
        for index, trick in enumerate(ordered_tricks)
    )
    has_continuation_variable = proof_model.new_bool_var(
        "qualification_principal_has_continuation"
    )
    if native_public_history:
        current_plays = candidate.state["currentTrick"].get("plays")
        if candidate.state.get("phase") != "playing" or type(current_plays) is not list or len(current_plays) > 3:
            raise ValueError("native public play continuation is unavailable")
        current_present = []
        for index, play in enumerate(current_plays):
            card = play.get("card") if isinstance(play, Mapping) else None
            if not isinstance(card, Mapping) or type(card.get("id")) is not int or not 0 <= card["id"] < 32:
                raise ValueError("native public current-trick card is unavailable")
            identity = bind_int(f"qualification_current_card_{index}", card["id"], 0, 31)
            present = proof_model.new_bool_var(f"qualification_current_present_{index}")
            proof_model.add(identity >= 0).only_enforce_if(present)
            proof_model.add(identity < 0).only_enforce_if(present.Not())
            current_present.append(present)
        remaining = proof_model.new_int_var(1, 32, "qualification_remaining_plays")
        # Ekka changes obligations, not the one-card-per-action round length.
        proof_model.add(remaining == 32 - 4 * sum(base_trick_variables) - sum(current_present))
        proof_model.add(remaining > 1).only_enforce_if(has_continuation_variable)
        proof_model.add(remaining <= 1).only_enforce_if(has_continuation_variable.Not())
    else:
        proof_model.add(has_continuation_variable == int(len(proof.principal_variation) > 1))
    trump_variable = (
        bind_int("qualification_trump_suit", int(trump), 0, 3)
        if trump is not None
        else None
    )
    first_winner_team_variable = None
    if principal_winner_variables:
        first_winner_team_variable = proof_model.new_int_var(
            0, 1, "qualification_first_principal_winner_team"
        )
        proof_model.add_modulo_equality(
            first_winner_team_variable,
            principal_winner_variables[0],
            2,
        )
    root_actor_is_target_variable = None
    target_seat_variable = None
    if (
        selected_index is not None
        and selected_index < len(root_actions)
        and target is not None
    ):
        selected_actor = root_actions[selected_index].actor
        selected_actor_team = proof_model.new_int_var(
            0, 1, "qualification_root_actor_team"
        )
        proof_model.add_modulo_equality(selected_actor_team, selected_actor, 2)
        root_actor_is_target_variable = proof_model.new_bool_var(
            "qualification_root_actor_is_target"
        )
        proof_model.add(selected_actor_team == int(target)).only_enforce_if(
            root_actor_is_target_variable
        )
        proof_model.add(selected_actor_team != int(target)).only_enforce_if(
            root_actor_is_target_variable.Not()
        )
        if root_team == target:
            target_seat_variable = selected_actor
        elif require_native and root_team is not None:
            target_seat_variable = bind_int("qualification_target_seat_absent", -1, -1, -1)
    decision_point = candidate.state.get("decisionPoint")
    decision_kind = (
        str(dict(decision_point).get("kind"))
        if isinstance(decision_point, Mapping)
        else "play" if root_actions else None
    )
    context = _ProofQualificationContextV1(
        base,
        root_actions,
        root_winning_variables=root_winning,
        root_value_variables=root_values,
        root_actor_seat=actor,
        root_actor_team=root_team,
        proof_utility=(
            proof_request.utility if proof_request is not None else (
                str(model_document.get("utility"))
                if model_document.get("utility") is not None else None
            )
        ),
        decision_kind=decision_kind,
        selected_root_action_index=selected_index,
        root_actor_is_target=(
            root_team == target
            if root_team is not None and target is not None
            else None
        ),
        root_actor_is_target_variable=root_actor_is_target_variable,
        target_seat_variable=target_seat_variable,
        card_strength_by_id=strengths,
        first_principal_trick_winner_team=first_winner,
        first_principal_trick_winner_team_variable=(
            first_winner_team_variable
        ),
        trump_suit=trump,
        trump_suit_variable=trump_variable,
        base_tricks=None if native_public_history else (base_tricks[0], base_tricks[1]),
        base_trick_variables=(
            base_trick_variables[0],
            base_trick_variables[1],
        ),
        principal_trick_winner_seat_variables=principal_winner_variables,
        principal_variation_has_continuation_variable=(
            has_continuation_variable
        ),
        native_proof_bindings=native_actor_roots,
        native_cp_sat_authority=require_native,
        tactical_detector_profile=tactical_detector_profile,
        root_stopping_variables=root_stopping,
    )
    binding = OrToolsStageContextBindingV1(
        "proof", proof_receipt.evidence_hash, context
    )
    return {
        "proof": binding,
        "tactics": OrToolsStageContextBindingV1(
            "tactics", proof_receipt.evidence_hash, context
        ),
    }


def _constant_auction_context_v1(
    candidate: OrToolsNeutralCandidateV1,
    rules: RuleBook,
    dealer: Seat,
    deck: Sequence[int],
) -> CpSatAuctionPredicateContextV1:
    raw = dict(candidate.trace["auction"])
    completion = dict(raw["completion"])
    buyer = Seat.parse(completion["buyer"])
    contract = Contract.parse(completion["contract"])
    trump_raw = completion.get("trumpSuit")
    trump = Suit.parse(trump_raw) if trump_raw is not None else None
    purchase_round = int(completion["purchaseRound"])
    domain = AuctionDomainIRV1.from_rules(rules.data)
    layout = DealLayoutV1.build(rules.structural_domain, dealer=dealer)
    purchase_card = Card(int(deck[layout.purchase_position]))
    purchase_suit = purchase_card.suit
    actions = []
    for row in completion["actions"]:
        action = str(row["action"])
        actor = Seat.parse(row["actor"])
        round_index = int(row["roundIndex"])
        action_trump_raw = row.get("trumpSuit")
        action_trump = (
            Suit.parse(action_trump_raw) if action_trump_raw is not None else None
        )
        resolution = domain.resolve_action(
            action,
            round_index=round_index,
            actor=actor,
            purchase_suit=purchase_suit,
            trump=action_trump,
        )
        actions.append(
            OrToolsAuctionActionV1(
                round_index,
                actor,
                action,
                action_trump,
                resolution.contract,
                resolution.buyer,
                resolution.purchase_recipient,
                resolution.caller,
                resolution.declarer,
            )
        )
    actions_tuple = tuple(actions)
    if not actions_tuple or not any(
        action.contract == contract and action.buyer == buyer
        for action in actions_tuple
    ):
        raise ValueError("candidate auction completion cannot be rebound")
    core = {
        "schemaVersion": 1,
        "auctionDomainIrHash": domain.sha256,
        "dealer": dealer.name.lower(),
        "purchaseSuit": purchase_suit.name.lower(),
        "actions": [action.to_dict() for action in actions_tuple],
        "buyer": buyer.name.lower(),
        "contract": contract.value,
        "trumpSuit": trump.name.lower() if trump is not None else None,
        "prefixLength": len(actions_tuple),
        "prefixTerminal": True,
        "legalNextActions": [],
        "legalNextChoices": [],
    }
    purchased_action = next(
        action
        for action in reversed(actions_tuple)
        if action.contract == contract and action.buyer == buyer
    )
    if (
        purchased_action.caller is None
        or purchased_action.purchase_recipient is None
        or purchased_action.declarer is None
    ):
        raise ValueError("candidate auction completion lacks purchased roles")
    caller = purchased_action.caller
    purchase_recipient = purchased_action.purchase_recipient
    declarer = purchased_action.declarer
    plan = _AuctionPlanV1(
        purchase_suit=purchase_suit,
        purchase_rank=(
            purchase_card.rank
            if domain.exposed_ace_hokum_sun_overcall == "dealer_right_only"
            else None
        ),
        actions=actions_tuple,
        caller=caller,
        buyer=buyer,
        purchase_recipient=purchase_recipient,
        declarer=declarer,
        contract=contract,
        trump=trump,
        purchase_round=purchase_round,
        prefix_actions=actions_tuple,
        prefix_caller=caller,
        prefix_buyer=buyer,
        prefix_purchase_recipient=purchase_recipient,
        prefix_declarer=declarer,
        prefix_contract=contract,
        prefix_trump=trump,
        prefix_purchase_round=purchase_round,
        current_actor=None,
        current_round=actions_tuple[-1].round_index,
        prefix_terminal=True,
        legal_next_actions=(),
        legal_next_choices=(),
        fingerprint=canonical_sha256(core),
    )
    model = cp_model.CpModel()
    fragment = OrToolsScenarioAuctionFragmentV1(
        (plan,),
        (model.new_constant(1),),
        model.new_constant(_CONTRACT_IDS[contract]),
        model.new_constant(_NO_TRUMP if trump is None else list(Suit).index(trump)),
        model.new_constant(purchase_round),
        model.new_constant(int(caller)),
        model.new_constant(int(buyer)),
        model.new_constant(int(purchase_recipient)),
        model.new_constant(int(declarer)),
        model.new_constant(1),
        {},
        {},
        domain.sha256,
    )
    return CpSatAuctionPredicateContextV1(
        model,
        fragment,
        dealer,
        view="completion",
        context_id=f"candidate-auction-{candidate.sha256[:12]}",
    )


def _constant_project_context_v1(
    candidate: OrToolsNeutralCandidateV1,
    rules: RuleBook,
) -> CpSatProjectPredicateContextV1:
    raw = dict(candidate.trace["projects"])
    play = dict(candidate.trace["play"])
    owner_by_card = {
        Card.parse(card).id: Seat.parse(seat)
        for seat, cards in dict(play["initialHands"]).items()
        for card in cards
    }
    if set(owner_by_card) != set(range(32)):
        raise ValueError("candidate project evidence requires every initial card")
    model = cp_model.CpModel()
    owners = tuple(
        model.new_constant(int(owner_by_card[card])) for card in range(32)
    )
    projects = tuple(
        ProjectPatternVariablesV1(
            str(row["projectId"]),
            Seat.parse(row["owner"]),
            str(row["kind"]),
            tuple(Card.parse(card) for card in row["cards"]),
            int(row["value"]),
            int(row["highRank"]),
            int(row["categoryPriority"]),
            model.new_constant(int(bool(row["eligible"]))),
            model.new_constant(int(bool(row["selected"]))),
            model.new_constant(int(bool(row["declarable"]))),
            model.new_constant(int(bool(row["declared"]))),
            model.new_constant(int(row["declarationOrder"] or 0)),
            model.new_constant(int(bool(row["comparisonWinner"]))),
            model.new_constant(int(bool(row["awarded"]))),
        )
        for row in raw["projects"]
    )
    winning_raw = raw.get("winningTeam")
    winning = Team[str(winning_raw).upper()] if winning_raw is not None else None
    awarded = tuple(int(value) for value in raw["combinedAwardedValue"])
    baloot_seats = {Seat.parse(value) for value in raw["balootSeats"]}
    baloot_teams = {Team[str(value).upper()] for value in raw["balootTeams"]}
    fragment = OrToolsProjectLifecycleFragmentV1(
        projects,
        owners,
        tuple(model.new_constant(int(team == winning)) for team in Team),
        tuple(model.new_constant(awarded[int(team)]) for team in Team),
        tuple(model.new_constant(int(seat in baloot_seats)) for seat in Seat),
        tuple(model.new_constant(int(team in baloot_teams)) for team in Team),
        rules.project_domain.baloot_value,
        str(raw["modelHash"]),
    )
    return CpSatProjectPredicateContextV1(
        model,
        f"candidate-projects-{candidate.sha256[:12]}",
        fragment,
    )


class DefaultStageContextCompilerV1:
    """Rebind a neutral candidate to exact fixed-value lowering contexts."""

    def __init__(
        self,
        rules: RuleBook,
        dealer: Seat | str | int = Seat.EAST,
        required_stages: Sequence[str] | None = None,
        native_cp_sat_authority: bool = False,
        visibility_requests: Sequence[tuple[Seat, int]] = (),
        actor_observer_scope_required: bool = False,
        proof_request: ScenarioProofRequestV1 | None = None,
        tactical_detector_profile: TacticalDetectorProfileV1 | None = None,
    ):
        if proof_request is not None:
            if not isinstance(proof_request, ScenarioProofRequestV1):
                raise TypeError("native proof request binding requires ScenarioProofRequestV1")
            if not native_cp_sat_authority:
                raise ValueError("proof request binding requires native CP-SAT authority")
        self.rules = rules
        self.dealer = Seat.parse(dealer)
        self.required_stages = (
            frozenset(required_stages) if required_stages is not None else None
        )
        self.native_cp_sat_authority = bool(native_cp_sat_authority)
        self.visibility_requests = tuple(sorted(set(visibility_requests)))
        self.actor_observer_scope_required = actor_observer_scope_required
        self.proof_request = proof_request
        self.tactical_detector_profile = tactical_detector_profile

    def __call__(
        self,
        candidate: OrToolsNeutralCandidateV1,
        proof: OrToolsGraphProofOutcomeV1 | None,
        *,
        information_graph: ActorInformationSetGraphV1 | None = None,
    ) -> Mapping[str, OrToolsStageContextBindingV1]:
        observer_scope = None
        if self.native_cp_sat_authority and self.actor_observer_scope_required:
            observer_scope = _owned_actor_observer_scope_v3(candidate, proof, self.rules, information_graph)
        trace = dict(candidate.trace)
        exact = candidate.origin == "exact_trace_prefix"
        decision_normalized = isinstance(trace.get("exactNormalization"), Mapping)
        normalized_decision_kind = (
            str(dict(trace["exactNormalization"]).get("decisionKind") or "")
            if decision_normalized
            else ""
        )
        nonplay_decision = normalized_decision_kind in {
            "bid",
            "escalation",
            "project",
        }
        if exact or decision_normalized:
            raw_prefix = dict(trace["exactTracePrefix"])
            deck = tuple(int(item) for item in raw_prefix["deckOrder"])
            dealer = Seat.parse(raw_prefix["dealer"])
            auction_actions = list(raw_prefix["auctionActions"])
            required_complete = bool(candidate.state.get("contract"))
            auction_view = "completion" if required_complete else "prefix"
        else:
            deck = tuple(int(item) for item in trace["deckOrder"])
            dealer = self.dealer
            auction_document = dict(trace["auction"])
            auction_view = str(trace.get("auctionPredicateView") or "completion")
            if auction_view not in {"prefix", "completion"}:
                raise ValueError("candidate auction predicate view is invalid")
            required_complete = auction_view == "completion"
            auction_actions = list(
                auction_document["completion"]["actions"]
                if required_complete
                else auction_document["actions"]
            )
        auction_actions = [
            {
                key: value
                for key, value in dict(action).items()
                if key
                in {"ordinal", "roundIndex", "actor", "action", "trumpSuit"}
            }
            for action in auction_actions
        ]
        contract_raw = trace.get("contract") or candidate.state.get("contract")
        buyer_raw = trace.get("buyer") or candidate.state.get("buyer")
        native_play_contexts: dict[bool, _PlayQualificationContextV1] = {}

        def play_context(*, complete: bool = False) -> _PlayQualificationContextV1:
            if not self.native_cp_sat_authority:
                return _constant_play_context(
                    candidate, self.rules, complete=complete
                )
            if complete not in native_play_contexts:
                native_play_contexts[complete] = _native_play_context_v3(
                    candidate,
                    self.rules,
                    dealer=dealer,
                    complete=complete,
                    observer_scope=observer_scope,
                )
            return native_play_contexts[complete]

        constant_rebind_stages = {
            "auction",
            "projects",
            "playPrefix",
            "state",
            "outcome",
            "proof",
            "tactics",
        }
        raw_projects = trace.get("projects")
        if (
            not self.native_cp_sat_authority
            and self.required_stages is not None
            and self.required_stages <= constant_rebind_stages
            and auction_view == "completion"
            and isinstance(trace.get("auction"), Mapping)
            and "completion" in trace["auction"]
            and isinstance(raw_projects, Mapping)
            and isinstance(trace.get("play"), Mapping)
            and contract_raw is not None
            and buyer_raw is not None
            and all(
                "declarable" in row for row in dict(raw_projects).get("projects", ())
            )
        ):
            bindings: dict[str, OrToolsStageContextBindingV1] = {}
            if "auction" in self.required_stages:
                bindings["auction"] = OrToolsStageContextBindingV1(
                    "auction",
                    candidate.stage_receipts["auction"].evidence_hash,
                    _constant_auction_context_v1(candidate, self.rules, dealer, deck),
                )
            if "projects" in self.required_stages:
                bindings["projects"] = OrToolsStageContextBindingV1(
                    "projects",
                    candidate.stage_receipts["projects"].evidence_hash,
                    _constant_project_context_v1(candidate, self.rules),
                )
            if "playPrefix" in self.required_stages:
                bindings["playPrefix"] = OrToolsStageContextBindingV1(
                    "playPrefix",
                    candidate.stage_receipts["playPrefix"].evidence_hash,
                    _constant_play_context(
                        candidate,
                        self.rules,
                        complete=isinstance(trace.get("completePlay"), Mapping),
                    ),
                )
            if "state" in self.required_stages:
                bindings["state"] = OrToolsStageContextBindingV1(
                    "state",
                    candidate.stage_receipts["state"].evidence_hash,
                    (
                        _constant_nonplay_state_context(candidate)
                        if nonplay_decision
                        else _constant_play_context(candidate, self.rules)
                    ),
                )
            if "outcome" in self.required_stages:
                bindings["outcome"] = OrToolsStageContextBindingV1(
                    "outcome",
                    candidate.stage_receipts["outcome"].evidence_hash,
                    _constant_outcome_context(candidate, Seat.parse(buyer_raw)),
                )
            proof_bindings = _proof_stage_bindings_v1(
                candidate,
                proof,
                self.rules,
                require_native=self.native_cp_sat_authority,
                proof_request=self.proof_request,
                information_graph=information_graph,
                tactical_detector_profile=self.tactical_detector_profile,
            )
            bindings.update(
                {
                    stage: binding
                    for stage, binding in proof_bindings.items()
                    if stage in self.required_stages
                }
            )
            if self.required_stages <= set(bindings):
                return bindings
        scenario = OrToolsScenarioModelV1(rules=self.rules, dealer=dealer)
        for position, card in enumerate(deck):
            scenario.model.add(scenario.card_at_position[position] == card)
        native_auction_fragment = None
        bind_deal = self.required_stages is None or "deal" in self.required_stages
        if self.native_cp_sat_authority and bind_deal:
            native_auction_fragment = scenario.add_native_auction_fragment(
                required_actions=auction_actions,
                required_actions_complete=required_complete,
            )
        remaining_owner_of_card = None
        trace_play = trace.get("play")
        if isinstance(trace_play, Mapping) and self.native_cp_sat_authority and bind_deal:
            play, rows = _candidate_play_rows(candidate)
            prefix_length = play["decisionState"]["prefixLength"]
            if type(prefix_length) is not int or not 0 <= prefix_length <= len(rows):
                raise ValueError("native remaining ownership requires a valid play prefix")
            played_cards = []
            for index, row in enumerate(rows[:prefix_length]):
                card = scenario.model.new_int_var(0, 31, f"deal_play_card_{index}")
                scenario.model.add(card == Card.parse(row["card"]).id)
                scenario.model.add_element(
                    card, scenario.owner_of_card, int(Seat.parse(row["actor"]))
                )
                played_cards.append(card)
            scenario.model.add_all_different(played_cards)
            remaining_owners = []
            for card, original_owner in enumerate(scenario.owner_of_card):
                matches = []
                for index, played_card in enumerate(played_cards):
                    match = scenario.model.new_bool_var(f"deal_played_{card}_{index}")
                    scenario.model.add(played_card == card).only_enforce_if(match)
                    scenario.model.add(played_card != card).only_enforce_if(match.Not())
                    matches.append(match)
                played = scenario.model.new_bool_var(f"deal_played_{card}")
                scenario.model.add_max_equality(played, matches or [0])
                owner = scenario.model.new_int_var(0, 4, f"deal_remaining_owner_{card}")
                scenario.model.add(owner == 4).only_enforce_if(played)
                scenario.model.add(owner == original_owner).only_enforce_if(played.Not())
                remaining_owners.append(owner)
            remaining_owner_of_card = tuple(remaining_owners)
        elif isinstance(trace_play, Mapping) and not self.native_cp_sat_authority:
            remaining = dict(dict(trace_play)["decisionState"])["remainingHands"]
            owner_by_remaining_card = {
                Card.parse(card).id: int(Seat.parse(seat))
                for seat, cards in dict(remaining).items()
                for card in cards
            }
            remaining_owner_of_card = tuple(
                scenario.model.new_constant(owner_by_remaining_card.get(card, 4))
                for card in range(32)
            )
        deal_context = DealPredicateModelContextV2(
            scenario,
            remaining_owner_of_card=remaining_owner_of_card,
        )
        constant_stages = {
            "deal",
            "playPrefix",
            "state",
            "outcome",
            "proof",
            "tactics",
        }
        if (
            self.required_stages is not None
            and self.required_stages <= constant_stages
            and ("outcome" not in self.required_stages or buyer_raw is not None)
            and not (
                self.native_cp_sat_authority
                and normalized_decision_kind in {"bid", "escalation", "project"}
                and "state" in self.required_stages
            )
        ):
            bindings: dict[str, OrToolsStageContextBindingV1] = {}
            if "deal" in self.required_stages:
                bindings["deal"] = OrToolsStageContextBindingV1(
                    "deal", candidate.stage_receipts["deal"].evidence_hash, deal_context
                )
            if "playPrefix" in self.required_stages:
                play_context_value = play_context(
                    complete=isinstance(trace.get("completePlay"), Mapping)
                )
                bindings["playPrefix"] = OrToolsStageContextBindingV1(
                    "playPrefix",
                    candidate.stage_receipts["playPrefix"].evidence_hash,
                    play_context_value,
                )
            if "state" in self.required_stages:
                state_context = (
                    _constant_nonplay_state_context(candidate)
                    if nonplay_decision
                    or not isinstance(trace.get("play"), Mapping)
                    else play_context()
                )
                bindings["state"] = OrToolsStageContextBindingV1(
                    "state",
                    candidate.stage_receipts["state"].evidence_hash,
                    state_context,
                )
            if "outcome" in self.required_stages:
                if buyer_raw is not None:
                    outcome = (
                        _native_outcome_context_v3(
                            candidate, self.rules, Seat.parse(buyer_raw)
                        )
                        if self.native_cp_sat_authority
                        else _constant_outcome_context(
                            candidate, Seat.parse(buyer_raw)
                        )
                    )
                    bindings["outcome"] = OrToolsStageContextBindingV1(
                        "outcome",
                        candidate.stage_receipts["outcome"].evidence_hash,
                        outcome,
                    )
            bindings.update(
                _proof_stage_bindings_v1(
                    candidate,
                    proof,
                    self.rules,
                    require_native=self.native_cp_sat_authority,
                    proof_request=self.proof_request,
                    information_graph=information_graph,
                    tactical_detector_profile=self.tactical_detector_profile,
                )
            )
            return bindings
        auction_fragment = native_auction_fragment or (
            scenario.add_native_auction_fragment
            if self.native_cp_sat_authority
            else scenario.add_auction_fragment
        )(
            required_actions=auction_actions,
            required_actions_complete=required_complete,
        )
        auction_context = CpSatAuctionPredicateContextV1(
            scenario.model,
            auction_fragment,
            dealer,
            view=auction_view,
        )
        if self.required_stages is not None and self.required_stages <= {"deal", "auction"}:
            return {
                stage: OrToolsStageContextBindingV1(stage, candidate.stage_receipts[stage].evidence_hash, context)
                for stage, context in (("deal", deal_context), ("auction", auction_context))
                if stage in self.required_stages
            }
        if contract_raw is None or buyer_raw is None:
            initial_owners = None
            if (
                self.native_cp_sat_authority
                and normalized_decision_kind == "bid"
                and self.visibility_requests
            ):
                layout = DealLayoutV1.build(self.rules.structural_domain, dealer=dealer)
                owner_by_position = [4] * 32
                for seat in Seat:
                    for position in layout.initial_positions[int(seat)]:
                        owner_by_position[position] = int(seat)
                initial_owners = tuple(
                    scenario.model.new_int_var(0, 4, f"bid_initial_owner_{card}")
                    for card in range(32)
                )
                for card, owner in enumerate(initial_owners):
                    scenario.model.add_element(
                        scenario.position_of_card[card], owner_by_position, owner,
                    )
            bindings = {
                "deal": OrToolsStageContextBindingV1(
                    "deal", candidate.stage_receipts["deal"].evidence_hash, deal_context
                ),
                "auction": OrToolsStageContextBindingV1(
                    "auction",
                    candidate.stage_receipts["auction"].evidence_hash,
                    auction_context,
                ),
            }
            if "state" in candidate.stage_receipts:
                bindings["state"] = OrToolsStageContextBindingV1(
                    "state",
                    candidate.stage_receipts["state"].evidence_hash,
                    _constant_nonplay_state_context(
                        candidate,
                        native_bid=(scenario.model, auction_fragment)
                        if self.native_cp_sat_authority
                        and normalized_decision_kind == "bid"
                        else None,
                        native_card_visibility=(
                            initial_owners, scenario.purchase_card,
                            self.visibility_requests, (),
                        )
                        if initial_owners is not None else None,
                    ),
                )
            complete = trace.get("completeExactNormalization")
            complete_state = (
                complete.get("state") if isinstance(complete, Mapping) else None
            )
            completed_buyer = (
                complete_state.get("buyer")
                if isinstance(complete_state, Mapping)
                else None
            )
            if (completed_buyer is None and self.native_cp_sat_authority
                    and isinstance(trace.get("constructionPlay"), Mapping)):
                completed_buyer = dict(trace.get("constructionScore") or {}).get("buyer")
            if completed_buyer is not None and "outcome" in candidate.stage_receipts:
                outcome_context = (
                    _native_outcome_context_v3(
                        candidate, self.rules, Seat.parse(completed_buyer)
                    )
                    if self.native_cp_sat_authority
                    else _constant_outcome_context(
                        candidate, Seat.parse(completed_buyer)
                    )
                )
                bindings["outcome"] = OrToolsStageContextBindingV1(
                    "outcome",
                    candidate.stage_receipts["outcome"].evidence_hash,
                    outcome_context,
                )
            bindings.update(
                _proof_stage_bindings_v1(
                    candidate,
                    proof,
                    self.rules,
                    require_native=self.native_cp_sat_authority,
                    proof_request=self.proof_request,
                    information_graph=information_graph,
                    tactical_detector_profile=self.tactical_detector_profile,
                )
            )
            return bindings
        contract = Contract.parse(contract_raw)
        trump_raw = trace.get("trumpSuit") or candidate.state.get("trumpSuit")
        if trump_raw:
            Suit.parse(trump_raw)
        buyer = Seat.parse(buyer_raw)
        escalation_declarer = buyer
        if self.native_cp_sat_authority:
            declarer_raw = candidate.state.get("declarer") or dict(dict(trace["auction"]).get("completion") or {}).get("declarer")
            if declarer_raw is None:
                raise ValueError("native escalation evidence requires the auction declarer")
            escalation_declarer = Seat.parse(declarer_raw)
        escalation_stage = OrToolsEscalationStageV1(
            EscalationDomainIRV1.from_rules(self.rules.data),
            declarer=escalation_declarer,
            contract=contract,
            match_score=tuple(candidate.state.get("matchScore", (0, 0))),
            native_cp_sat=self.native_cp_sat_authority,
        )
        escalation_context = CpSatEscalationPredicateContextV1.attach(
            escalation_stage,
            model=scenario.model if self.native_cp_sat_authority and self.visibility_requests else None,
            declarer_variable=auction_fragment.declarer
            if self.native_cp_sat_authority and self.visibility_requests else None,
        )
        observed_escalation = dict(dict(trace["escalation"])["observation"])
        matching = [
            index
            for index, row in enumerate(escalation_context.candidates)
            if [item.to_dict() for item in row.actions]
            == observed_escalation["actions"]
            and row.level == observed_escalation["level"]
            and row.locked == (observed_escalation["mode"] == "locked")
            and row.terminal is observed_escalation["terminal"]
        ]
        if self.native_cp_sat_authority:
            escalation_context.bind_native_observation(observed_escalation)
        else:
            if len(matching) != 1:
                raise ValueError("escalation evidence cannot be rebound uniquely")
            escalation_context.model.add(escalation_context.selected[matching[0]] == 1)
        bindings: dict[str, OrToolsStageContextBindingV1] = {
            "deal": OrToolsStageContextBindingV1(
                "deal", candidate.stage_receipts["deal"].evidence_hash, deal_context
            ),
            "auction": OrToolsStageContextBindingV1(
                "auction",
                candidate.stage_receipts["auction"].evidence_hash,
                auction_context,
            ),
            "escalation": OrToolsStageContextBindingV1(
                "escalation",
                candidate.stage_receipts["escalation"].evidence_hash,
                escalation_context,
            ),
        }
        if trace.get("projects") is None or trace.get("play") is None:
            if (
                self.native_cp_sat_authority
                and normalized_decision_kind == "project"
                and self.required_stages is not None
                and "state" in self.required_stages
            ):
                raise ValueError("native project state requires project and play models")
            bindings["state"] = OrToolsStageContextBindingV1(
                "state",
                candidate.stage_receipts["state"].evidence_hash,
                _constant_nonplay_state_context(
                    candidate,
                    native_bid=(scenario.model, auction_fragment)
                    if self.native_cp_sat_authority
                    and normalized_decision_kind == "bid"
                    else None,
                    native_escalation=escalation_context
                    if self.native_cp_sat_authority
                    and normalized_decision_kind == "escalation"
                    else None,
                    native_card_visibility=(
                        scenario.owner_of_card,
                        scenario.purchase_card,
                        self.visibility_requests,
                        (),
                    )
                    if self.native_cp_sat_authority
                    and normalized_decision_kind == "escalation"
                    and self.visibility_requests
                    else None,
                ),
            )
            if "outcome" in candidate.stage_receipts:
                outcome_context = (
                    _native_outcome_context_v3(candidate, self.rules, buyer)
                    if self.native_cp_sat_authority
                    else _constant_outcome_context(candidate, buyer)
                )
                bindings["outcome"] = OrToolsStageContextBindingV1(
                    "outcome",
                    candidate.stage_receipts["outcome"].evidence_hash,
                    outcome_context,
                )
            bindings.update(
                _proof_stage_bindings_v1(
                    candidate,
                    proof,
                    self.rules,
                    require_native=self.native_cp_sat_authority,
                    proof_request=self.proof_request,
                    information_graph=information_graph,
                    tactical_detector_profile=self.tactical_detector_profile,
                )
            )
            return bindings
        if self.native_cp_sat_authority:
            prefix_context = play_context()
            project_context = prefix_context.project_context
            project_actor = None
            project_visibility = None
            if normalized_decision_kind == "project":
                project_actions = raw_prefix.get("projectActions")
                if not isinstance(project_actions, list):
                    raise ValueError("native project state requires typed action prefix")
                declaration_order = self.rules.project_domain.declaration_seats(dealer)
                if len(project_actions) >= len(declaration_order):
                    raise ValueError("project decision follows all declaration turns")
                project_actor = declaration_order[len(project_actions)]
                if self.visibility_requests:
                    played = raw_prefix.get("playActions")
                    if not isinstance(played, list) or len(played) >= 4:
                        raise ValueError("project visibility requires a partial first trick")
                    if any(row.get("revealed") for row in candidate.state.get("projects", ())):
                        raise ValueError("project visibility cannot bind an early reveal")
                    layout = DealLayoutV1.build(self.rules.structural_domain, dealer=dealer)
                    project_visibility = (
                        prefix_context.native_build.initial_owner_variables,
                        prefix_context.model.new_constant(deck[layout.purchase_position]),
                        self.visibility_requests,
                        tuple(slot.card for slot in prefix_context.play_slots[:len(played)]),
                    )
            state_context = (
                _constant_nonplay_state_context(
                    candidate,
                    native_bid=(scenario.model, auction_fragment)
                    if normalized_decision_kind == "bid" else None,
                    native_escalation=escalation_context
                    if normalized_decision_kind == "escalation" else None,
                    native_project=(
                        prefix_context.model, project_actor,
                        project_context.projects.declaration_actions[int(project_actor)],
                    ) if project_actor is not None else None,
                    native_card_visibility=project_visibility or (
                        scenario.owner_of_card, scenario.purchase_card,
                        self.visibility_requests, (),
                    ) if project_visibility is not None or (
                        normalized_decision_kind == "escalation" and self.visibility_requests
                    ) else None,
                ) if nonplay_decision else prefix_context
            )
            bindings.update({
                "projects": OrToolsStageContextBindingV1(
                    "projects", candidate.stage_receipts["projects"].evidence_hash, project_context,
                ),
                "playPrefix": OrToolsStageContextBindingV1(
                    "playPrefix", candidate.stage_receipts["playPrefix"].evidence_hash,
                    play_context(complete=isinstance(trace.get("completePlay"), Mapping)),
                ),
                "state": OrToolsStageContextBindingV1(
                    "state", candidate.stage_receipts["state"].evidence_hash, state_context,
                ),
            })
            if "outcome" in candidate.stage_receipts:
                bindings["outcome"] = OrToolsStageContextBindingV1(
                    "outcome", candidate.stage_receipts["outcome"].evidence_hash,
                    _native_outcome_context_v3(candidate, self.rules, buyer),
                )
            bindings.update(_proof_stage_bindings_v1(
                candidate, proof, self.rules, require_native=True,
                proof_request=self.proof_request,
                information_graph=information_graph,
                tactical_detector_profile=self.tactical_detector_profile,
            ))
            return bindings
        # Historical candidates keep their frozen lifecycle evidence, not a new rule model.
        project_context = _constant_project_context_v1(candidate, self.rules)
        play_context_value = play_context(
            complete=isinstance(trace.get("completePlay"), Mapping)
        )
        state_context = (
            _constant_nonplay_state_context(candidate)
            if nonplay_decision
            else play_context()
        )
        bindings.update(
            {
                "projects": OrToolsStageContextBindingV1(
                    "projects",
                    candidate.stage_receipts["projects"].evidence_hash,
                    project_context,
                ),
                "playPrefix": OrToolsStageContextBindingV1(
                    "playPrefix",
                    candidate.stage_receipts["playPrefix"].evidence_hash,
                    play_context_value,
                ),
                "state": OrToolsStageContextBindingV1(
                    "state",
                    candidate.stage_receipts["state"].evidence_hash,
                    state_context,
                ),
            }
        )
        if "outcome" in candidate.stage_receipts:
            outcome_context = (
                _native_outcome_context_v3(candidate, self.rules, buyer)
                if self.native_cp_sat_authority
                else _constant_outcome_context(candidate, buyer)
            )
            bindings["outcome"] = OrToolsStageContextBindingV1(
                "outcome",
                candidate.stage_receipts["outcome"].evidence_hash,
                outcome_context,
            )
        bindings.update(
            _proof_stage_bindings_v1(
                candidate,
                proof,
                self.rules,
                require_native=self.native_cp_sat_authority,
                proof_request=self.proof_request,
                information_graph=information_graph,
                tactical_detector_profile=self.tactical_detector_profile,
            )
        )
        return bindings


def _adjudication_predicates_v3(
    scenario: ResearchScenarioV1,
) -> tuple[ScenarioPredicateV1, ...]:
    return tuple(
        predicate
        for predicate in scenario.predicates()
        if predicate.kind
        in {
            OUT_OF_TURN_PREDICATE_V3,
            PURCHASE_BEFORE_TURN_PREDICATE_V3,
            FALSE_EKKA_PREDICATE_V3,
        }
    )


def _wrap_adjudication_stage_bindings_v3(
    scenario: ResearchScenarioV1,
    candidate: OrToolsNeutralCandidateV1,
    bindings: Mapping[str, OrToolsStageContextBindingV1],
    authority: AdjudicationResearchAuthorityV1,
) -> Mapping[str, OrToolsStageContextBindingV1]:
    """Bind attempted-action predicates beside, never inside, legal history."""

    predicates = _adjudication_predicates_v3(scenario)
    if not predicates:
        return bindings
    if not isinstance(authority, AdjudicationResearchAuthorityV1):
        raise TypeError("V3 adjudication predicates require their selected authority")
    exact_prefix = scenario.origin.exact_trace_prefix
    exact_events = (
        exact_prefix.adjudication_events if exact_prefix is not None else ()
    )
    result = dict(bindings)

    auction_predicates = tuple(
        predicate
        for predicate in predicates
        if predicate.kind == PURCHASE_BEFORE_TURN_PREDICATE_V3
    )
    if auction_predicates:
        binding = result.get("auction")
        if binding is None:
            raise ValueError("purchase-before-turn qualification requires auction evidence")
        context = wrap_auction_adjudication_context_v3(
            binding.context,
            auction_predicates,
            origin=candidate.origin,
            exact_events=exact_events,
            action_sequence=(
                len(exact_prefix.auction_actions)
                if exact_prefix is not None and binding.context.view == "prefix"
                else None
            ),
            authority_id=authority.profile_id,
            authority_sha256=authority.sha256,
        )
        result["auction"] = OrToolsStageContextBindingV1(
            binding.stage,
            binding.source_evidence_hash,
            context,
        )

    play_predicates = tuple(
        predicate
        for predicate in predicates
        if predicate.kind
        in {OUT_OF_TURN_PREDICATE_V3, FALSE_EKKA_PREDICATE_V3}
    )
    if play_predicates:
        binding = result.get("playPrefix")
        if binding is None:
            raise ValueError("out-of-turn qualification requires play-prefix evidence")
        play, _ = _candidate_play_rows(candidate)
        owners = _initial_owner_by_card(candidate, play)
        if set(owners) != set(range(32)):
            raise ValueError("out-of-turn qualification requires all initial card owners")
        owner_variables = tuple(
            binding.context.model.new_constant(int(owners[card_id]))
            for card_id in range(32)
        )
        context = wrap_play_adjudication_context_v3(
            binding.context,
            owner_variables,
            play_predicates,
            origin=candidate.origin,
            exact_events=exact_events,
            action_sequence=(
                len(exact_prefix.play_actions) if exact_prefix is not None else None
            ),
            authority_id=authority.profile_id,
            authority_sha256=authority.sha256,
        )
        result["playPrefix"] = OrToolsStageContextBindingV1(
            binding.stage,
            binding.source_evidence_hash,
            context,
        )
    return result


def _tail_early_policy_steps_v1(
    prefix_length: int,
    quality_policy: ScenarioPlayingQualityPolicyV1 | None,
    execution_budget: ScenarioQualityExecutionBudgetV1 | None,
) -> int:
    if (quality_policy is None or quality_policy.mode != "budgeted"
            or execution_budget is None
            or execution_budget.early_play_policy_id is None):
        return 0
    assert execution_budget.exact_play_from_trick is not None
    return min(prefix_length, 4 * (execution_budget.exact_play_from_trick - 1))


class OrToolsUnifiedResearchExecutorV1:
    """Execute exact input or deterministic no-good search without runtime imports."""

    def __init__(
        self,
        rules: RuleBook,
        *,
        bundle_revision_id: str,
        bundle_hash: str,
        backend_id: str = BACKEND_ID,
        tactical_detector_profile: TacticalDetectorProfileV1 | None = None,
        adjudication_authority: Mapping[str, Any] | None = None,
        rule_source_catalog: Mapping[str, Any] | None = None,
        witness_policies: Mapping[str, Any] | None = None,
        quality_policy: ScenarioPlayingQualityPolicyV1 | None = None,
        quality_execution: ScenarioQualityExecutionBudgetV1 | None = None,
        quality_proof_profiles: Mapping[str, Mapping[str, Any]] | None = None,
        quality_search_profiles: Mapping[str, Mapping[str, Any]] | None = None,
        runtime_proposals: Mapping[int, Mapping[str, Any]] | None = None,
        bid_quality_cache: _BidQualityCacheV1 | None = None,
        generation: Mapping[str, Any] | None = None,
    ):
        if not isinstance(rules, RuleBook):
            raise TypeError("unified Research executor requires RuleBook")
        if type(bundle_revision_id) is not str or not bundle_revision_id.strip():
            raise ValueError("unified Research executor requires a bundle revision")
        if type(bundle_hash) is not str or len(bundle_hash) != 64:
            raise ValueError("unified Research executor requires a bundle hash")
        if backend_id not in {BACKEND_ID, "ortools_cp_sat_v3"}:
            raise ValueError("unified Research executor backend is unsupported")
        self.rules = rules
        self.bundle_revision_id = bundle_revision_id
        self.bundle_hash = bundle_hash
        self.backend_id = backend_id
        if tactical_detector_profile is not None:
            if backend_id != _V3_BACKEND_ID or not isinstance(tactical_detector_profile, TacticalDetectorProfileV1):
                raise TypeError("native tactical detector configuration requires a frozen V3 profile")
            tactical_detector_profile.validate()
        self.tactical_detector_profile = tactical_detector_profile
        quality_values = (quality_policy, quality_execution, quality_proof_profiles)
        if any(value is not None for value in quality_values):
            if backend_id != _V3_BACKEND_ID or any(value is None for value in quality_values):
                raise ValueError("campaign quality requires native V3 and complete frozen settings")
            if not isinstance(quality_policy, ScenarioPlayingQualityPolicyV1) or not isinstance(
                quality_execution, ScenarioQualityExecutionBudgetV1,
            ):
                raise TypeError("campaign quality requires typed policy and execution budget")
            quality_policy = ScenarioPlayingQualityPolicyV1.build(quality_policy.to_dict())
            quality_execution = ScenarioQualityExecutionBudgetV1.build(quality_execution.to_dict())
            validate_content_interest_rules_v1(rules, quality_policy, quality_execution)
            if quality_execution.escalation_opportunity_order is None:
                raise ValueError("campaign quality requires frozen escalation opportunity order")
            quality_budgets, quality_proof_profiles = _quality_budget_snapshots(
                {entry["budgetProfile"] for entry in quality_policy.decisions.values()},
                quality_proof_profiles,
                execution_budget=quality_execution,
            )
            if quality_execution.featured_play_native_proof:
                play_request = quality_policy.request_for("play", "north")
                if quality_budgets[play_request.budget_profile].native_observed is None:
                    raise ValueError("featured play requires a native observed proof budget")
        self.quality_policy = quality_policy
        self.quality_execution = quality_execution
        self.generation = None
        if generation is not None:
            from .family_catalog import build_family_generation_request_v1
            if quality_execution is None or quality_execution.family_catalog is None:
                raise ValueError("generation requires a frozen native familyCatalog")
            self.generation = build_family_generation_request_v1(generation)
            if self.generation["catalogHash"] != quality_execution.family_catalog["sha256"]:
                raise ValueError("generation catalog does not match the frozen authority")
        if bid_quality_cache is not None and not isinstance(bid_quality_cache, _BidQualityCacheV1):
            raise TypeError("bid-quality reuse requires a private bounded cache")
        self.bid_quality_cache = bid_quality_cache
        if quality_policy is not None and self.bid_quality_cache is None:
            bid_budget = quality_budgets[quality_policy.decisions["bid"]["budgetProfile"]]
            self.bid_quality_cache = _BidQualityCacheV1(bid_budget.maximum_cache_entries, bid_budget.maximum_cache_bytes)
        self.runtime_proposals = None
        if runtime_proposals is not None:
            if (quality_execution is None or quality_execution.candidate_source not in {"runtime_proposal_v1", "runtime_proposal_stream_v1"}
                    or backend_id != _V3_BACKEND_ID):
                raise ValueError("runtime proposals require their frozen native candidateSource")
            self.runtime_proposals = {}
            for ordinal, proposal in runtime_proposals.items():
                row = copy.deepcopy(dict(proposal))
                digest = row.pop("sha256", None)
                prefix = ExactTracePrefixV1.build(row["prefix"])
                native_opening = quality_execution.family_opening_mode is not None
                if (type(ordinal) is not int or ordinal < 0
                        or digest != canonical_sha256(row)
                        or set(row) != ({"prefix", "prefixHash", "rulesHash", "policyHash", "playIndex", "actor", "action", "sourceHash", "sourceSeed"}
                                       | ({'familyOpening'} if native_opening else set()))
                        or type(row["sourceSeed"]) is not int or row["sourceSeed"] < 0
                        or row["rulesHash"] != rules.sha256
                        or row["policyHash"] != canonical_sha256(quality_execution.video_interestingness)
                        or row["prefixHash"] != prefix.sha256
                        or type(row["playIndex"]) is not int or row["playIndex"] != len(prefix.play_actions)
                        or not 0 <= row["playIndex"] <= 31
                        or row["actor"] not in {"north", "east", "south", "west"}
                        or not isinstance(row["action"], str)
                        or not isinstance(row["sourceHash"], str) or len(row["sourceHash"]) != 64):
                    raise ValueError("runtime proposal data identity mismatch")
                if native_opening:
                    from .scenario_contracts import build_family_opening_evidence_v1
                    row['familyOpening'] = build_family_opening_evidence_v1(row['familyOpening'],
                        prefix=prefix, rules_hash=rules.sha256, source_seed=row['sourceSeed'])
                self.runtime_proposals[ordinal] = {**row, "sha256": digest}
        self.quality_proof_profiles = quality_proof_profiles
        if quality_search_profiles is not None:
            if (quality_proof_profiles is None or not isinstance(quality_search_profiles, Mapping)
                    or any(not isinstance(quality_search_profiles.get(key), Mapping)
                        or type(quality_search_profiles[key].get("maximumConflicts")) is not int
                        or quality_search_profiles[key]["maximumConflicts"] < 1
                        for key in quality_proof_profiles)):
                raise ValueError("campaign quality requires matching frozen search profiles")
        self.quality_search_profiles = copy.deepcopy(quality_search_profiles)
        if witness_policies is not None and backend_id != "ortools_cp_sat_v3":
            raise ValueError("witness policies are supported only by V3")
        self.witness_policies = (
            copy.deepcopy(dict(witness_policies))
            if witness_policies is not None
            else None
        )
        if adjudication_authority is not None and not isinstance(
            adjudication_authority, Mapping
        ):
            raise TypeError("adjudication authority must be a frozen mapping")
        if adjudication_authority is not None and backend_id != "ortools_cp_sat_v3":
            raise ValueError("adjudication authority is supported only by V3")
        self.adjudication_authority_document = (
            copy.deepcopy(dict(adjudication_authority))
            if adjudication_authority is not None
            else None
        )
        self.adjudication_authority = (
            AdjudicationResearchAuthorityV1.from_mapping(
                self.adjudication_authority_document
            )
            if self.adjudication_authority_document is not None
            else None
        )
        if self.adjudication_authority is not None:
            out_of_turn = self.adjudication_authority.capability("out_of_turn")
            invalid_bid = self.adjudication_authority.capability("invalid_bid")
            if (
                out_of_turn.covered_subtype != "play_before_turn"
                or invalid_bid.covered_subtype != "purchase_before_turn"
            ):
                raise ValueError(
                    "selected adjudication authority lacks native attempted-action subtypes"
                )
        self.backend_version = (
            BACKEND_VERSION if backend_id == BACKEND_ID else _V3_BACKEND_VERSION
        )
        if rule_source_catalog is not None:
            if backend_id != _V3_BACKEND_ID:
                raise ValueError("rule source catalog is supported only by V3")
            if not isinstance(rule_source_catalog, Mapping) or set(rule_source_catalog) != {"data", "sha256"}:
                raise ValueError("rule source catalog requires a frozen data and hash snapshot")
            catalog = RuleSourceCatalogV1(rule_source_catalog["data"], rules)
            if catalog.sha256 != rule_source_catalog["sha256"]:
                raise ValueError("rule source catalog snapshot hash mismatch")
            self.rule_source_catalog = catalog
        else:
            self.rule_source_catalog = (
                load_exact_rule_source_catalog(rules, platform_root())
                if backend_id == _V3_BACKEND_ID else None
            )
        native_delayed_payoff = (self.tactical_detector_profile is not None
            and self.tactical_detector_profile.native_delayed_gain_metric is not None)
        selected_manifest = (
            shared_scenario_predicate_catalog_lowering_manifest_v3(
                self.adjudication_authority.profile_id,
                self.adjudication_authority.sha256,
                "false_ekka_attempt" in self.adjudication_authority.capabilities,
                native_delayed_payoff=native_delayed_payoff,
            )
            if backend_id == "ortools_cp_sat_v3"
            and self.adjudication_authority is not None
            else shared_scenario_predicate_catalog_lowering_manifest_v3(native_delayed_payoff=native_delayed_payoff)
            if backend_id == "ortools_cp_sat_v3"
            else shared_scenario_predicate_catalog_lowering_manifest_v2()
        )
        self.lowering_manifest = selected_manifest
        self.compiler = OrToolsResearchCompilerV1(
            selected_manifest,
            native_cp_sat_authority=backend_id == "ortools_cp_sat_v3",
            tactical_detector_profile=self.tactical_detector_profile,
        )

    def _candidate_from_normalized_exact(
        self, normalized, prefix: ExactTracePrefixV1, *, origin: str = "exact_trace_prefix",
    ) -> OrToolsNeutralCandidateV1:
        receipt = normalized.normalization_receipt.to_dict()
        derived_auction = dict(receipt["derivedAuction"])
        derived_escalation = receipt["derivedEscalation"]
        derived_projects = receipt["derivedProjects"]
        derived_play = receipt["derivedPlay"]
        stage_receipts = {
            "deal": _stage_receipt(
                "deal",
                model_hash=receipt["modelHash"],
                evidence_hash=receipt["solutionHash"],
                semantic={
                    "variableCount": receipt.get("variableCount", 0),
                    "constraintCount": receipt.get("constraintCount", 0),
                    "deckOrder": list(
                        prefix.deck_order
                    )
                },
                solver_status=receipt["solverStatus"],
            ),
            "auction": _stage_receipt(
                "auction",
                model_hash=receipt["modelHash"],
                evidence_hash=receipt["solutionHash"],
                semantic=derived_auction,
                solver_status=receipt["solverStatus"],
                variable_count=int(receipt.get("variableCount", 0)),
                constraint_count=int(receipt.get("constraintCount", 0)),
            ),
            "state": _stage_receipt(
                "state",
                model_hash=receipt["modelHash"],
                evidence_hash=normalized.state_hash,
                semantic={
                    **normalized.state,
                    "variableCount": receipt.get("variableCount", 0),
                    "constraintCount": receipt.get("constraintCount", 0),
                },
                solver_status=receipt["solverStatus"],
            ),
        }
        if derived_escalation is not None:
            stage_receipts["escalation"] = _stage_receipt(
                "escalation",
                model_hash=derived_escalation["modelHash"],
                evidence_hash=derived_escalation["evidenceHash"],
                semantic=derived_escalation,
                solver_status="optimal",
            )
        if derived_projects is not None:
            stage_receipts["projects"] = _stage_receipt(
                "projects",
                model_hash=derived_projects["modelHash"],
                evidence_hash=derived_projects["evidenceHash"],
                semantic=derived_projects,
                solver_status=derived_projects["solverStatus"],
                branches=derived_projects["branches"],
                conflicts=derived_projects["conflicts"],
            )
        if derived_play is not None:
            stage_receipts["playPrefix"] = _stage_receipt(
                "playPrefix",
                model_hash=derived_play["modelHash"],
                evidence_hash=derived_play["evidenceHash"],
                semantic=derived_play,
                solver_status=derived_play["solverStatus"],
            )
        round_score = normalized.state.get("roundScore")
        if round_score is not None:
            outcome_hash = canonical_sha256(round_score)
            stage_receipts["outcome"] = _stage_receipt(
                "outcome",
                model_hash=receipt["modelHash"],
                evidence_hash=outcome_hash,
                semantic={
                    **round_score,
                    "variableCount": int(
                        dict(receipt.get("derivedScoring") or {}).get(
                            "variableCount", 0
                        )
                    ),
                    "constraintCount": int(
                        dict(receipt.get("derivedScoring") or {}).get(
                            "constraintCount", 0
                        )
                    ),
                },
                solver_status="optimal",
            )
        current_raw = [0, 0]
        for trick in normalized.state.get("completedTricks", []):
            current_raw[int(Seat.parse(trick["winner"]).team)] += int(
                trick["rawPoints"]
            )
        current_projects = [0, 0]
        winning_team_raw = normalized.state.get("projectWinningTeam")
        if winning_team_raw is not None:
            winning_team = Team[str(winning_team_raw).upper()]
            current_projects[int(winning_team)] = sum(
                int(item["value"])
                for item in normalized.state.get("projects", [])
                if item.get("scoring") is True
            )
        score_snapshot = round_score or {
            "rawPoints": current_raw,
            "projectPoints": current_projects,
            "gamePoints": [0, 0],
            "finalMatchScore": list(normalized.state["matchScore"]),
            "roundWinner": Team.NORTH_SOUTH.name.lower(),
            "matchWinner": normalized.state.get("matchWinner"),
            "kabout": False,
            "gahwa": False,
            "matchTarget": ScoringDomainIRV1.from_rules(
                self.rules.data
            ).match_target,
        }
        exact_trace = {
            "schemaVersion": 1,
            "roundNumber": prefix.round_number,
            "matchScore": list(prefix.match_score),
            "deckOrder": list(
                prefix.deck_order
            ),
            "exactTracePrefix": prefix.to_dict(),
            "exactNormalization": normalized.to_dict(),
            "normalizationReceipt": receipt,
            "dealer": normalized.state["dealer"],
            "buyer": normalized.state.get("buyer"),
            "contract": normalized.state.get("contract"),
            "trumpSuit": normalized.state.get("trumpSuit"),
            "auction": {"prefix": derived_auction},
            "escalation": derived_escalation,
            "projects": derived_projects,
            "play": derived_play,
            "score": score_snapshot,
        }
        exact_prefix = prefix
        assert exact_prefix is not None
        if exact_prefix.adjudication_events:
            if self.adjudication_authority is None:
                raise ValueError(
                    "exact adjudication evidence requires the selected authority"
                )
            exact_trace["adjudicationReport"] = (
                AdjudicationResearchReportV1.build(
                    self.adjudication_authority,
                    prefix_sha256=exact_prefix.sha256,
                    events=exact_prefix.adjudication_events,
                    state_sha256=normalized.state_hash,
                    exact_state_id=normalized.exact_state_id,
                ).to_dict()
            )
        candidate = OrToolsNeutralCandidateV1(
            origin,
            exact_trace,
            normalized.state,
            tuple(prefix.deck_order),
            stage_receipts,
            normalized.exact_state_id,
        )
        return candidate

    def generate_quality_candidate(
        self, candidate: OrToolsNeutralCandidateV1, *, maximum_seconds: float,
        scenario: ResearchScenarioV1 | None = None,
        cancel_check: Callable[[], bool] | None = None,
        checkpoint_callback: Callable[[Mapping[str, Any]], None] | None = None,
        _verified_artifacts: set[str] | None = None,
        graph_workers: int = 1,
    ) -> OrToolsCandidateAttemptV1:
        """Select certified bids unless the scenario explicitly owns the action history."""
        select_auction = (
            self.backend_id == _V3_BACKEND_ID
            and candidate.origin == "search"
            and scenario is not None and scenario.origin.kind == "search"
            and not any(predicate.hardness == "hard" and predicate.kind == "auction.action_sequence"
                        for predicate in scenario.predicates())
        )
        return self._resolve_quality_candidate(
            candidate,
            maximum_seconds=maximum_seconds,
            cancel_check=cancel_check,
            recorded_trace=None if select_auction else _construction_auction_prefix_v1(candidate),
            recorded_events=None if select_auction else (),
            decision_point={"kind": "escalation", "selector": {}},
            continue_recorded_prefix=not select_auction,
            scenario=scenario,
            checkpoint_callback=checkpoint_callback,
            _verified_artifacts=_verified_artifacts,
            graph_workers=graph_workers,
        )

    def construct_qualified_auction(self, prefix, *, maximum_seconds, cancel_check=None):
        """Generate and read back native bid choices before constructing play motifs."""
        if type(maximum_seconds) not in (int, float) or not math.isfinite(maximum_seconds) or maximum_seconds <= 0:
            raise ValueError('qualified auction allowance must be finite and positive')
        if self.backend_id != _V3_BACKEND_ID or self.quality_policy is None or self.quality_execution is None:
            raise ValueError('qualified auction construction requires frozen native quality settings')
        budgets, snapshots = _quality_budget_snapshots(
            {row['budgetProfile'] for row in self.quality_policy.decisions.values()},
            self.quality_proof_profiles, execution_budget=self.quality_execution)
        execution = self.quality_execution
        return resolve_quality_auction_continuation_v1(
            self.rules, prefix, self.quality_policy,
            maximum_nodes=max(row.maximum_nodes for row in budgets.values()),
            maximum_depth=max(row.maximum_depth for row in budgets.values()),
            maximum_worlds=max(row.maximum_hidden_worlds for row in budgets.values()),
            maximum_seconds=execution.maximum_wall_seconds,
            runtime_maximum_seconds=min(maximum_seconds, execution.maximum_wall_seconds),
            maximum_decision_seconds=max(row.maximum_wall_seconds for row in budgets.values()),
            maximum_evidence_bytes=execution.maximum_evidence_bytes,
            proof_budget_profiles=snapshots, search_budget_profiles=self.quality_search_profiles,
            execution_budget=execution, opportunity_order=execution.escalation_opportunity_order,
            _bid_quality_cache=self.bid_quality_cache, search_generated=True, cancel_check=cancel_check)

    def audit_quality_candidate(
        self, candidate: OrToolsNeutralCandidateV1, *, recorded_trace: ExactTracePrefixV1,
        recorded_events: Sequence[Mapping[str, Any]], maximum_seconds: float,
        decision_point: Mapping[str, Any],
        cancel_check: Callable[[], bool] | None = None,
        continue_recorded_prefix: bool = False,
        stop_on_recorded_quality_failure: bool = False,
        checkpoint_callback: Callable[[Mapping[str, Any]], None] | None = None,
        _verified_artifacts: set[str] | None = None,
        graph_workers: int = 1,
    ) -> OrToolsCandidateAttemptV1:
        """Audit recorded actions and optionally generate only their continuation."""
        if not isinstance(recorded_trace, ExactTracePrefixV1) or recorded_events is None:
            raise ValueError("recorded candidate audit requires a trace and explicit opportunity history")
        return self._resolve_quality_candidate(candidate, maximum_seconds=maximum_seconds,
            cancel_check=cancel_check, recorded_trace=recorded_trace, recorded_events=recorded_events,
            decision_point=decision_point, continue_recorded_prefix=continue_recorded_prefix,
            audit_recorded=True,
            stop_on_recorded_quality_failure=stop_on_recorded_quality_failure,
            checkpoint_callback=checkpoint_callback, _verified_artifacts=_verified_artifacts,
            graph_workers=graph_workers)

    def _resolve_quality_candidate(
        self, candidate: OrToolsNeutralCandidateV1, *, maximum_seconds: float,
        cancel_check: Callable[[], bool] | None = None,
        recorded_trace: ExactTracePrefixV1 | None = None,
        recorded_events: Sequence[Mapping[str, Any]] | None = None,
        decision_point: Mapping[str, Any] | None = None,
        continue_recorded_prefix: bool = False,
        audit_recorded: bool = False,
        stop_on_recorded_quality_failure: bool = False,
        scenario: ResearchScenarioV1 | None = None,
        checkpoint_callback: Callable[[Mapping[str, Any]], None] | None = None,
        _verified_artifacts: set[str] | None = None,
        graph_workers: int = 1,
    ) -> OrToolsCandidateAttemptV1:
        if self.quality_policy is None or self.quality_execution is None:
            raise ValueError("quality generation requires frozen campaign settings")
        started = time.perf_counter()
        source = ExactTracePrefixV1.build({
            "schemaVersion": 1, "deckOrder": list(candidate.trace["deckOrder"]),
            "dealer": candidate.trace["dealer"],
            "roundNumber": candidate.trace.get("roundNumber", 1),
            "matchScore": list(candidate.trace.get("matchScore", (0, 0))),
            "auctionActions": [], "escalationActions": [], "projectActions": [],
            "projectDeclarations": [], "playActions": [],
        })
        if recorded_trace is not None:
            _validate_quality_recorded_trace(source, recorded_trace,
                allow_auction_extension=True, allow_escalation_extension=True,
                allow_partial=continue_recorded_prefix)
        recorded_input = ({"recordedTrace": {"completeExactTracePrefix": recorded_trace.to_dict()},
                           "recordedEscalationOpportunities": [dict(row) for row in (recorded_events or ())]}
                          if recorded_trace is not None else {})
        budgets, snapshots = _quality_budget_snapshots(
            {entry["budgetProfile"] for entry in self.quality_policy.decisions.values()},
            self.quality_proof_profiles,
            execution_budget=self.quality_execution,
        )
        execution = self.quality_execution
        quality_worlds = max(item.maximum_hidden_worlds for item in budgets.values())
        late_precheck = None
        precheck_seconds = 0.0
        if (self.backend_id == _V3_BACKEND_ID and audit_recorded and stop_on_recorded_quality_failure
                and recorded_trace is not None and self.quality_policy.mode == "exact"
                and self.quality_policy.visibility == "complete_state"
                and len(recorded_trace.play_actions) == 32 and recorded_trace.project_actions_present):
            precheck_started = time.perf_counter()
            try:
                late_precheck = _late_recorded_quality_precheck_v1(
                    self.rules, recorded_trace, self.quality_policy, execution, snapshots,
                    self.quality_search_profiles,
                    maximum_seconds=min(maximum_seconds, execution.maximum_wall_seconds,
                        budgets[self.quality_policy.decisions["play"]["budgetProfile"]].maximum_wall_seconds,
                        budgets[self.quality_policy.decisions["play"]["budgetProfile"]].maximum_cpu_seconds),
                    cancel_check=cancel_check, checkpoint_callback=checkpoint_callback)
            except (InformationSetCompilationStopped, ExactNormalizationStopped) as exc:
                core = {"schemaVersion": 1, "kind": "late_recorded_quality_precheck_v1",
                    "scope": "suffix_only", "recordedPrefixHash": recorded_trace.sha256,
                    "status": "incomplete", "partialReceipt": exc.partial_receipt}
                late_precheck = {**core, "sha256": canonical_sha256(core)}
            precheck_seconds = time.perf_counter() - precheck_started
            if late_precheck is not None and late_precheck.get("receipt", {}).get("status") == "quality_failed":
                core = {"schemaVersion": 1, "stage": "playingQuality", "qualityStatus": "quality_failed",
                    "qualityFailureStage": "action_analysis", "qualityFailureReason": "recorded_action_not_optimal",
                    "sourcePrefix": source.to_dict(), "lateQualityPrecheck": late_precheck,
                    "recordedTrace": {"completeExactTracePrefix": recorded_trace.to_dict()}}
                return OrToolsCandidateAttemptV1("not_found", None, reason="quality_recorded_audit_target_miss",
                    stage_timings={"qualityProjectPlay": precheck_seconds},
                    partial_receipt={**core, "sha256": canonical_sha256(core)})
            if (maximum_seconds - (time.perf_counter() - started) <= 0
                    or cancel_check is not None and cancel_check()):
                core = {"schemaVersion": 1, "stage": "playingQuality", "qualityStatus": "incomplete",
                    "qualityFailureReason": "cancelled" if cancel_check is not None and cancel_check() else "maximum_seconds",
                    "lateQualityPrecheck": late_precheck, "sourcePrefix": source.to_dict(), **recorded_input}
                return OrToolsCandidateAttemptV1("incomplete", None, reason="quality_recorded_audit_incomplete",
                    partial_receipt={**core, "sha256": canonical_sha256(core)})
        decision = copy.deepcopy(dict(decision_point)) if decision_point is not None else {"kind": "escalation", "selector": {}}
        target_play_index = None
        target_actor = None
        featured_play_index = None
        constructed_decision_prefix = None
        first_play_divergence = None
        constructed_plays = (
            candidate.trace["play"].get("plays")
            if isinstance(candidate.trace.get("play"), Mapping) else None
        )
        track_constructed_play = bool(
            scenario is not None and scenario.origin.kind == "search" and not audit_recorded
            and isinstance(constructed_plays, (list, tuple)) and constructed_plays
        )
        generated_target_actor = None
        required_awards = tuple(
            predicate for predicate in scenario.predicates()
            if predicate.kind == "project.exists"
            and predicate.hardness == "hard"
            and predicate.arguments["state"] == "awarded"
            and _hard_expression_requires_predicate(scenario, predicate.predicate_id)
        ) if scenario is not None and scenario.origin.kind == "search" and not audit_recorded else ()
        project_award_miss: dict[str, Any] | None = None
        future_count_miss: Mapping[str, Any] | None = None
        initial_hands = (candidate.trace["play"].get("initialHands")
                         if isinstance(candidate.trace.get("play"), Mapping) else None)
        future_counts = tuple(predicate for predicate in scenario.predicates()
            if predicate.hardness == "hard" and predicate.kind == "play.trick_card_count"
            and maximum_seconds > time.perf_counter() - started
            and _hard_expression_requires_predicate(scenario, predicate.predicate_id,
                maximum_seconds=maximum_seconds - (time.perf_counter() - started), cancel_check=cancel_check)
        ) if (self.backend_id == _V3_BACKEND_ID and scenario is not None
              and scenario.origin.kind == "search" and not audit_recorded
              and self.quality_policy.mode == "budgeted"
              and self.quality_policy.visibility == "actor_information_set"
              and isinstance(initial_hands, Mapping)) else ()
        construction_auction = (_construction_auction_prefix_v1(candidate).auction_actions
                                if future_counts else ())

        def admissible_play_actions(prefix, normalized, remaining_seconds):
            if prefix.auction_actions != construction_auction:
                return None
            return _future_card_count_admissible_actions_v1(initial_hands, prefix, normalized,
                future_counts, maximum_seconds=remaining_seconds(), cancel_check=cancel_check)

        if scenario is not None and scenario.decision_point["kind"] == "play":
            selector = scenario.decision_point["selector"]
            explicit_play_coordinate = "position" in selector and "trickIndex" in selector
            if explicit_play_coordinate:
                target_play_index = int(selector["trickIndex"]) * 4 + int(selector["position"])
            elif isinstance(candidate.trace.get("play"), Mapping):
                constructed_decision_prefix = _search_decision_prefix(
                    candidate, scenario,
                    project_declaration_mode=self.rules.project_domain.declaration_mode,
                    native_cp_sat=True,
                )
                target_play_index = len(constructed_decision_prefix.play_actions)
            if execution.featured_play_native_proof:
                featured_play_index = target_play_index if target_play_index is not None else 0
            if explicit_play_coordinate:
                target_actor = selector.get("actingSeat")
        elif execution.featured_play_native_proof and decision.get("kind") == "play":
            if recorded_trace is None:
                raise ValueError("featured native play requires an exact recorded prefix")
            selector = dict(decision.get("selector") or {})
            featured_play_index = (
                int(selector["trickIndex"]) * 4 + int(selector["position"])
                if "trickIndex" in selector and "position" in selector
                else len(recorded_trace.play_actions)
            )

        last_quality_checkpoint: Mapping[str, Any] | None = None

        def quality_checkpoint(row: Mapping[str, Any]) -> None:
            nonlocal first_play_divergence, generated_target_actor, project_award_miss, last_quality_checkpoint, future_count_miss
            last_quality_checkpoint = row
            if checkpoint_callback is not None:
                checkpoint_callback(row)
            awards = row.get("publicAwardedProjects")
            if (required_awards and row.get("stage") == "play"
                    and isinstance(row.get("decision"), Mapping)
                    and row["decision"].get("playIndex") == 7
                    and isinstance(awards, list)):
                missing = [
                    predicate.predicate_id for predicate in required_awards
                    if not any(
                        all(award.get(key) == predicate.arguments[key]
                            for key in ("team", "kind", "value"))
                        and (predicate.arguments.get("owner") is None
                             or award.get("owner") == predicate.arguments["owner"])
                        for award in awards
                    )
                ]
                if missing:
                    project_award_miss = {
                        "requiredPredicateIds": missing,
                        "publicAwardedProjects": awards,
                    }
                    raise _QualityScenarioTargetMiss(
                        "quality project declarations miss a required awarded project"
                    )
            selected = row.get("decision")
            if (
                track_constructed_play
                and first_play_divergence is None
                and row.get("kind") == "quality_decision_checkpoint_v1"
                and row.get("stage") == "play"
                and isinstance(selected, Mapping)
            ):
                play_index = selected.get("playIndex")
                if type(play_index) is int and 0 <= play_index < len(constructed_plays):
                    constructed = constructed_plays[play_index]
                    constructed_action = LegalPlayActionV1(
                        Card.parse(constructed["card"]), bool(constructed.get("ekka", False)),
                    ).code
                    if (selected.get("actor"), selected.get("action")) != (
                        constructed["actor"], constructed_action,
                    ):
                        first_play_divergence = play_index
            if (
                target_play_index is not None
                and row.get("kind") == "quality_decision_checkpoint_v1"
                and row.get("stage") == "play"
                and isinstance(selected, Mapping)
                and type(selected.get("playIndex")) is int
                and selected["playIndex"] <= target_play_index
                and selected["playIndex"] // 4 == target_play_index // 4
                and target_actor is not None
                and selected.get("actor") is not None
            ):
                actor_at_target = self.rules.structural_domain.offset(
                    Seat.parse(selected["actor"]), target_play_index - selected["playIndex"],
                ).name.lower()
                if actor_at_target != target_actor:
                    generated_target_actor = actor_at_target
                    raise _QualityScenarioTargetMiss(
                        "quality continuation misses the requested play actor"
                    )
            if (future_counts and row.get("kind") == "quality_decision_checkpoint_v1"
                    and row.get("stage") == "play" and isinstance(row.get("nextPrefix"), Mapping)):
                generated = ExactTracePrefixV1.build(row["nextPrefix"])
                if generated.auction_actions == construction_auction:
                    future_count_miss = _future_card_count_relaxation_v1(
                        initial_hands, generated, future_counts,
                        maximum_seconds=maximum_seconds - (time.perf_counter() - started),
                        cancel_check=cancel_check)
                    if future_count_miss is not None:
                        raise _QualityScenarioTargetMiss("generated prefix cannot satisfy a required future card count")

        final_normalized: list[OrToolsNormalizedExactStateV1] = []

        def retain_final_normalized(value: OrToolsNormalizedExactStateV1) -> None:
            if final_normalized:
                raise ValueError("quality continuation returned multiple terminal states")
            final_normalized.append(value)

        result: Mapping[str, Any] | None = None
        try:
            quality_remaining = maximum_seconds - (time.perf_counter() - started)
            if quality_remaining <= 0 or (cancel_check is not None and cancel_check()):
                partial_core = {"schemaVersion": 1, "stage": "playingQuality", "qualityStatus": "incomplete",
                    "qualityFailureReason": "maximum_seconds" if quality_remaining <= 0 else "cancelled",
                    "sourcePrefix": source.to_dict(), **recorded_input}
                return OrToolsCandidateAttemptV1("incomplete", None,
                    reason="quality_recorded_audit_incomplete" if audit_recorded else "quality_generation_incomplete",
                    partial_receipt={**partial_core, "sha256": canonical_sha256(partial_core)})
            result = resolve_quality_round_continuation_v1(
                self.rules, source, decision, self.quality_policy,
                graph_workers=graph_workers,
                opportunity_order=execution.escalation_opportunity_order,
                maximum_nodes=max(item.maximum_nodes for item in budgets.values()),
                maximum_depth=max(item.maximum_depth for item in budgets.values()),
                maximum_worlds=quality_worlds,
                maximum_seconds=execution.maximum_wall_seconds,
                runtime_maximum_seconds=min(quality_remaining, execution.maximum_wall_seconds),
                maximum_decision_seconds=max(item.maximum_wall_seconds for item in budgets.values()),
                maximum_escalation_decisions=execution.maximum_escalation_decisions,
                maximum_evidence_bytes=execution.maximum_evidence_bytes,
                proof_budget_profiles=snapshots, execution_budget=execution, cancel_check=cancel_check,
                search_budget_profiles=self.quality_search_profiles,
                recorded_trace=recorded_trace, recorded_events=recorded_events,
                continue_recorded_prefix=continue_recorded_prefix,
                search_generated=candidate.origin == "search" and not audit_recorded,
                featured_play_index=featured_play_index,
                featured_decision_point=(
                    scenario.decision_point
                    if featured_play_index is not None and scenario is not None
                    else decision
                    if featured_play_index is not None
                    else None
                ),
                featured_selector_dynamic=bool(
                    scenario is not None
                    and scenario.origin.kind == "search"
                    and not (
                        "position" in scenario.decision_point["selector"]
                        and "trickIndex" in scenario.decision_point["selector"]
                    )
                ),
                stop_on_recorded_quality_failure=stop_on_recorded_quality_failure,
                checkpoint_callback=quality_checkpoint,
                final_normalized_callback=retain_final_normalized,
                _bid_quality_cache=self.bid_quality_cache,
                _verified_artifacts=_verified_artifacts,
                **({"_admissible_play_actions": admissible_play_actions} if future_counts else {}),
                _curated_focal_precheck=bool(
                    self.backend_id == _V3_BACKEND_ID and candidate.origin == "search"
                    and audit_recorded and continue_recorded_prefix and stop_on_recorded_quality_failure
                    and execution.video_interestingness is not None
                    and execution.video_interestingness.get("schemaVersion") == 4
                    and self.quality_policy.visibility == "actor_information_set"
                    and (execution.family_target is not None or self.generation is not None)),
            )
            partial = result.get("partialReceipt")
            if (
                scenario is not None
                and featured_play_index is not None
                and isinstance(partial, Mapping)
                and partial.get("exhaustionReason")
                == "featured_decision_mismatch"
            ):
                evidence = partial.get("evidence")
                if isinstance(evidence, Mapping):
                    generated_target_actor = evidence.get("actualActor")
                raise _QualityScenarioTargetMiss(
                    "quality continuation misses the requested play decision"
                )
        except _QualityEditorialMiss as exc:
            partial_core = {"schemaVersion": 1, "stage": "curatedSelection",
                "qualityFailureReason": ", ".join(exc.evidence["blockers"]),
                "researchClaimStatus": "not_evaluated", "candidateHash": candidate.sha256,
                "focalPrecheck": exc.evidence, "qualityElapsedSeconds": time.perf_counter() - started,
                **recorded_input}
            return OrToolsCandidateAttemptV1("not_found", None, reason="curated_interest_miss",
                partial_receipt={**partial_core, "sha256": canonical_sha256(partial_core)})
        except _QualityScenarioTargetMiss as exc:
            retained_execution = dict(result.get("executionReceipt") or {}) if result is not None else {}
            retained_post = dict(retained_execution.get("postAuction") or {})
            retained_timings = {
                "qualityAuction": float(dict(retained_execution.get("auction") or {}).get("elapsedSeconds", 0)),
                "qualityProjectPlay": precheck_seconds + float(
                    dict(retained_post.get("projectPlay") or {}).get("elapsedSeconds", 0)),
            }
            partial_core = {
                "schemaVersion": 1,
                "stage": "playingQuality",
                "reason": ("hard_project_award_miss" if project_award_miss else
                           "hard_future_card_count_miss" if future_count_miss or exc.evidence else "decision_point_target_miss"),
                "detail": str(exc),
                "candidateHash": candidate.sha256,
                "targetPlayIndex": target_play_index,
                "targetActor": target_actor,
                "firstConstructedPlayDivergenceIndex": first_play_divergence,
                "generatedActorAtTarget": generated_target_actor,
                "qualityElapsedSeconds": time.perf_counter() - started,
                **({"qualityCheckpoint": copy.deepcopy(last_quality_checkpoint)}
                   if last_quality_checkpoint is not None else {}),
                **({"projectAwardMiss": project_award_miss} if project_award_miss else {}),
                **({"futureCardCountPrecheck": future_count_miss} if future_count_miss else {}),
                **({"witnessActionSelection": exc.evidence} if exc.evidence else {}),
                "sourcePrefix": source.to_dict(),
                **({"qualityEvidence": {key: value for key, value in result.items() if key != "executionReceipt"},
                    "executionReceipt": retained_execution} if result is not None else {}),
                **({"lateQualityPrecheck": late_precheck} if late_precheck is not None else {}),
            }
            return OrToolsCandidateAttemptV1(
                "not_found", None, reason=("quality_recorded_audit_target_miss"
                    if audit_recorded else "quality_generation_target_miss"),
                stage_timings=retained_timings,
                partial_receipt={**partial_core, "sha256": canonical_sha256(partial_core)},
            )
        execution = result["executionReceipt"]
        post_execution = execution.get("postAuction")
        quality_stage_timings = {
            "qualityAuction": float(execution["auction"]["elapsedSeconds"]),
            "qualityProjectPlay": float(post_execution["projectPlay"]["elapsedSeconds"])
            if isinstance(post_execution, Mapping)
            and isinstance(post_execution.get("projectPlay"), Mapping)
            else 0.0,
        }
        quality_stage_timings["qualityProjectPlay"] += precheck_seconds
        receipt = {key: value for key, value in result.items() if key != "executionReceipt"}
        trace = {"completeExactTracePrefix": recorded_trace.to_dict() if recorded_trace is not None and not continue_recorded_prefix else result["finalPrefix"], "playingQualityEvidence": {
            "sourcePrefix": source.to_dict(), "decisionPoint": decision, "receipt": receipt,
        }}
        recorded_evidence = {"recordedTrace": trace} if recorded_trace is not None else {}
        remaining = maximum_seconds - (time.perf_counter() - started)
        if result["status"] != "complete" or len(result["finalPrefix"]["playActions"]) != 32 or remaining <= 0:
            # This path has no candidate for final admission to validate.
            try:
                if remaining <= 0:
                    raise InformationSetCompilationStopped("maximum_seconds", stage="quality_readback")
                from .result_authority import _validate_trace_quality_evidence_v1
                _validate_trace_quality_evidence_v1(trace, rules_hash=self.rules.sha256, rules=self.rules,
                    verified_artifacts=_verified_artifacts if _verified_artifacts is not None else frozenset(),
                    required_search_profiles=self.quality_search_profiles,
                    maximum_seconds=remaining, cancel_check=cancel_check)
            except InformationSetCompilationStopped as stopped:
                partial_core = {"schemaVersion": 1, "stage": "playingQuality",
                    "qualityEvidence": receipt, "sourcePrefix": source.to_dict(), **recorded_evidence,
                    "executionReceipt": result.get("executionReceipt"), "qualityStatus": "incomplete",
                    "qualityFailureReason": stopped.reason,
                    "qualityValidationIncomplete": stopped.partial_receipt}
                return OrToolsCandidateAttemptV1("incomplete", None,
                    reason="quality_recorded_audit_incomplete" if audit_recorded else "quality_generation_incomplete",
                    stage_timings=quality_stage_timings,
                    partial_receipt={**partial_core, "sha256": canonical_sha256(partial_core)})
            remaining = maximum_seconds - (time.perf_counter() - started)
            quality_failure = receipt.get("partialReceipt")
            quality_failure_reason = (
                quality_failure.get("reason") or quality_failure.get("exhaustionReason")
                if isinstance(quality_failure, Mapping) else None
            )
            cause_stage, cause_queries, cause_phase, path_phase = _quality_failure_cause_v1(quality_failure)
            partial_core = {"schemaVersion": 1, "stage": "playingQuality",
                            "qualityEvidence": receipt, "sourcePrefix": source.to_dict(),
                            "executionReceipt": result.get("executionReceipt"), **recorded_evidence,
                            "qualityStatus": result["status"],
                            **({"lateQualityPrecheck": late_precheck} if late_precheck is not None else {}),
                            **({"qualityFailureStage": quality_failure["stage"]}
                               if isinstance(quality_failure, Mapping) and isinstance(quality_failure.get("stage"), str) else {}),
                            **({"qualityFailureCauseStage": cause_stage} if cause_stage else {}),
                            **({"qualityFailureQueryCount": cause_queries} if cause_queries is not None else {}),
                            **({"qualityFailureQueryPhase": cause_phase} if cause_phase else {}),
                            **({"qualityFailurePathPhase": path_phase} if path_phase else {}),
                            **({"qualityFailureReason": quality_failure_reason}
                               if isinstance(quality_failure_reason, str) else {})}
            # A freshly verified counterexample remains a rejection even if
            # bookkeeping crosses the deadline; unfinished readback returns above.
            incomplete = result["status"] == "incomplete" or (
                remaining <= 0 and result["status"] != "quality_failed"
            )
            return OrToolsCandidateAttemptV1(
                "incomplete" if incomplete else "not_found",
                None, reason=("quality_recorded_audit_incomplete" if audit_recorded else "quality_generation_incomplete")
                if incomplete else ("quality_recorded_audit_target_miss" if audit_recorded else "quality_generation_target_miss"),
                stage_timings=quality_stage_timings,
                partial_receipt={**partial_core, "sha256": canonical_sha256(partial_core)},
            )
        # Complete candidates remain feasible until _result validates their proof.
        if len(final_normalized) != 1:
            raise ValueError("completed quality continuation did not retain one terminal state")
        prefix = ExactTracePrefixV1.build(trace["completeExactTracePrefix"])
        normalized_candidate = self._candidate_from_terminal_quality_trace(
            trace, prefix, final_normalized[0], origin=candidate.origin,
        )
        handoff = None
        if track_constructed_play:
            handoff_core = {
                "schemaVersion": 1,
                "kind": "quality_handoff_source_v1",
                "constructionCandidateHash": candidate.sha256,
                "generatedCandidateHash": normalized_candidate.sha256,
                "firstConstructedPlayDivergenceIndex": first_play_divergence,
            }
            handoff = {
                **handoff_core,
                "sha256": canonical_sha256(handoff_core),
            }
        if late_precheck is not None:
            core = {**({key: value for key, value in handoff.items() if key != "sha256"}
                       if handoff is not None else {}),
                    "schemaVersion": 1, "lateQualityPrecheck": late_precheck}
            handoff = {**core, "sha256": canonical_sha256(core)}
        return OrToolsCandidateAttemptV1(
            "feasible",
            normalized_candidate,
            stage_timings=quality_stage_timings,
            partial_receipt=handoff,
        )

    def candidate_from_quality_trace(
        self, trace: Mapping[str, Any], *, maximum_seconds: float,
        cancel_check: Callable[[], bool] | None = None,
        origin: str | None = None,
    ) -> OrToolsNeutralCandidateV1:
        """Normalize a quality-checked full round without changing its provenance."""
        if self.backend_id != _V3_BACKEND_ID:
            raise ValueError("quality candidates require native V3")
        if type(maximum_seconds) not in {int, float} or not math.isfinite(maximum_seconds) or maximum_seconds <= 0:
            raise ValueError("quality readback deadline must be finite and positive")
        deadline = time.monotonic() + maximum_seconds

        def remaining():
            if cancel_check is not None and cancel_check():
                raise InformationSetCompilationStopped("cancelled", stage="quality_readback")
            seconds = deadline - time.monotonic()
            if seconds <= 0:
                raise InformationSetCompilationStopped("maximum_seconds", stage="quality_readback")
            return seconds

        remaining()
        validate_trace_quality_evidence_v1(trace, rules_hash=self.rules.sha256, rules=self.rules,
            required_policy=self.quality_policy, required_execution=self.quality_execution,
            require_family_opening=True,
            required_proof_profiles=self.quality_proof_profiles,
            required_search_profiles=self.quality_search_profiles,
            maximum_seconds=remaining(), cancel_check=cancel_check)
        remaining()
        evidence = trace.get("playingQualityEvidence")
        if not isinstance(evidence, Mapping):
            raise ValueError("quality candidate requires immutable quality evidence")
        receipt = evidence["receipt"]
        if receipt["status"] != "complete" or receipt["operation"] not in {"generate", "audit_recorded", "audit_then_generate"}:
            raise ValueError("quality candidate requires completed generation or recorded audit")
        prefix = ExactTracePrefixV1.build(trace["completeExactTracePrefix"])
        if len(prefix.play_actions) != 32:
            raise ValueError("quality candidate requires a completed purchased round")
        normalized = OrToolsExactStateNormalizerV1(self.rules, native_cp_sat=True).normalize(
            prefix, decision_point={"kind": "play", "selector": {}},
            maximum_seconds=remaining(), cancel_check=cancel_check,
        )
        remaining()
        candidate = self._candidate_from_terminal_quality_trace(
            trace, prefix, normalized, origin=origin,
        )
        remaining()
        return candidate

    def _candidate_from_terminal_quality_trace(
        self,
        trace: Mapping[str, Any],
        prefix: ExactTracePrefixV1,
        normalized: OrToolsNormalizedExactStateV1,
        *,
        origin: str | None = None,
    ) -> OrToolsNeutralCandidateV1:
        """Build a feasible candidate; proof qualification belongs to _result."""
        if self.backend_id != _V3_BACKEND_ID:
            raise ValueError("quality candidates require native V3")
        if not isinstance(normalized, OrToolsNormalizedExactStateV1):
            raise TypeError("quality candidate requires a typed terminal normalization")
        normalized.require_integrity()
        disagreements = [
            name
            for name, matches in (
                ("play_count", len(prefix.play_actions) == 32),
                ("prefix_hash", normalized.prefix_hash == _effective_prefix_hash(prefix)),
                ("rules_hash", normalized.rules_hash == self.rules.sha256),
                ("phase", normalized.phase == "scored"),
                ("next_actor", normalized.next_actor is None),
                ("legal_actions", not normalized.legal_actions),
            )
            if not matches
        ]
        if disagreements:
            raise ValueError(
                "quality terminal normalization disagrees with the completed trace: "
                + ", ".join(disagreements)
            )
        evidence = trace.get("playingQualityEvidence")
        if not isinstance(evidence, Mapping) or not isinstance(evidence.get("receipt"), Mapping):
            raise ValueError("quality candidate requires immutable quality evidence")
        receipt = evidence["receipt"]
        post_auction = receipt.get("postAuction")
        project_play = post_auction.get("projectPlay") if isinstance(post_auction, Mapping) else None
        if (
            receipt.get("status") != "complete"
            or receipt.get("operation") not in {"generate", "audit_recorded", "audit_then_generate"}
            or receipt.get("finalPrefix") != prefix.to_dict()
            or not isinstance(project_play, Mapping)
            or project_play.get("finalExactStateId") != normalized.exact_state_id
        ):
            raise ValueError("quality evidence disagrees with its terminal normalization")
        candidate = self._candidate_from_normalized_exact(
            normalized,
            prefix,
            origin=origin or (
                "search"
                if receipt["operation"] == "generate"
                else "exact_trace_prefix"
            ),
        )
        receipt_document = normalized.normalization_receipt.to_dict()
        generated_trace = {
            **candidate.trace,
            "roundNumber": prefix.round_number,
            "matchScore": list(prefix.match_score),
            "score": _terminal_score_from_state_v1(
                normalized.state, match_target=ScoringDomainIRV1.from_rules(self.rules.data).match_target,
            ),
            "auction": {"completion": {
                **receipt_document["derivedAuction"],
                "actions": [dict(action) for action in prefix.auction_actions],
            }},
            "projects": {**candidate.trace["projects"],
                         "declarationActions": [dict(action) for action in prefix.project_actions]},
            "completeExactTracePrefix": prefix.to_dict(),
            "completeExactNormalization": normalized.to_dict(),
            "completePlay": receipt_document["derivedPlay"],
            "playingQualityEvidence": copy.deepcopy(dict(evidence)),
        }
        return replace(candidate, trace=generated_trace)

    def execute(
        self,
        scenario: ResearchScenarioV1,
        proof_request: ScenarioProofRequestV1,
        *,
        candidate_compiler: Callable[..., OrToolsCandidateAttemptV1] | None = None,
        context_compiler: StageContextCompiler | None = None,
        graph_compiler: GraphCompiler | None = None,
        maximum_proof_nodes: int | None = None,
        maximum_proof_depth: int | None = None,
        maximum_hidden_worlds: int | None = None,
        policy_search: str | None = None,
        maximum_line_count_limbs: int = 64,
        maximum_conflicts: int = 250_000,
        maximum_prefix_lift_attempts: int | None = None,
        seed: int = 0,
        maximum_candidates: int = 1,
        maximum_seconds: float = 15.0,
        maximum_stage_seconds: float | None = None,
        initial_excluded_decks: Sequence[Sequence[int]] = (),
        initial_quality_rejected_candidates: Sequence[OrToolsNeutralCandidateV1] = (),
        minimum_deck_hamming_distance: int = 1,
        excluded_decision_status_vectors: Sequence[Sequence[int]] = (),
        minimum_decision_status_hamming_distance: int = 1,
        candidate_partition: tuple[int, int, int] | None = None,
        cancel_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
        terminal_worker_count: int = 1,
        checkpoint_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> OrToolsUnifiedResearchResultV1:
        _reject_v3_semantic_callback_injection(
            self.backend_id,
            candidate_compiler=candidate_compiler,
            context_compiler=context_compiler,
            graph_compiler=graph_compiler,
        )
        if not isinstance(proof_request, ScenarioProofRequestV1):
            raise TypeError("unified Research executor requires ScenarioProofRequestV1")
        validate_native_delayed_payoff_scope_v1(scenario, proof_request, self.tactical_detector_profile,
            native_authority=self.backend_id == _V3_BACKEND_ID)
        self._validate_runtime_proposal_scope(scenario)
        proposal = self.runtime_proposals.get(seed) if self.runtime_proposals is not None else None
        proposal_prefix = ExactTracePrefixV1.build(proposal["prefix"]) if proposal is not None else None
        if policy_search is not None and (policy_search != "time_budgeted_symbolic_v1"
                or self.backend_id != "ortools_cp_sat_v3"):
            raise ValueError("symbolic policy search requires the native V3 backend")
        if self.quality_policy is not None and scenario.origin.kind != "search":
            recorded = scenario.origin.exact_trace_prefix
            events = scenario.origin.recorded_escalation_opportunities
            if (recorded is None or events is None or self.quality_execution is None
                    or (self.quality_policy.mode == "exact" and not self.quality_execution.allow_exact_input_quality)
                    or not recorded.project_actions_present or recorded.project_declarations
                    or (not (recorded.project_actions or recorded.play_actions)
                        and bool(recorded.escalation_actions or events)
                        and (self.backend_id != "ortools_cp_sat_v3"
                            or scenario.decision_point["kind"] != "escalation"
                            or proof_request.mode not in {"exact", "bounded"}
                            or bool(recorded.state_assertions or recorded.project_assertions)))
                    or len(events) > self.quality_execution.maximum_escalation_decisions):
                raise ValueError("exact-input quality requires an auction or project/play prefix and recorded escalation history")
        if type(maximum_candidates) is not int or maximum_candidates < 1:
            raise ValueError("maximum_candidates must be positive")
        if type(terminal_worker_count) is not int or terminal_worker_count < 1:
            raise ValueError("terminal_worker_count must be positive")
        if (
            type(maximum_seconds) not in {int, float}
            or not math.isfinite(maximum_seconds)
            or maximum_seconds <= 0
        ):
            raise ValueError("maximum_seconds must be finite and positive")
        stage_seconds = (
            maximum_seconds if maximum_stage_seconds is None else maximum_stage_seconds
        )
        if (
            type(stage_seconds) not in {int, float}
            or not math.isfinite(stage_seconds)
            or stage_seconds <= 0
        ):
            raise ValueError("maximum_stage_seconds must be finite and positive")
        if (maximum_proof_nodes is None) != (maximum_proof_depth is None):
            raise ValueError("default graph proof requires both governed graph limits")
        if maximum_proof_nodes is not None and (
            type(maximum_proof_nodes) is not int
            or type(maximum_proof_depth) is not int
            or maximum_proof_nodes < 1
            or maximum_proof_depth < 1
        ):
            raise ValueError("governed graph limits must be positive integers")
        if maximum_hidden_worlds is not None and (
            type(maximum_hidden_worlds) is not int or maximum_hidden_worlds < 1
        ):
            raise ValueError("maximum_hidden_worlds must be a positive integer")
        if type(maximum_line_count_limbs) is not int or maximum_line_count_limbs < 1:
            raise ValueError("maximum_line_count_limbs must be a positive integer")
        if type(maximum_conflicts) is not int or maximum_conflicts < 1:
            raise ValueError("maximum_conflicts must be a positive integer")
        if maximum_prefix_lift_attempts is not None and (
            type(maximum_prefix_lift_attempts) is not int
            or maximum_prefix_lift_attempts < 1
        ):
            raise ValueError("maximum_prefix_lift_attempts must be a positive integer")
        if (
            type(minimum_deck_hamming_distance) is not int
            or not 1 <= minimum_deck_hamming_distance <= 32
        ):
            raise ValueError("minimum_deck_hamming_distance must be in 1..32")
        decision_status_cuts = tuple(
            tuple(vector) for vector in excluded_decision_status_vectors
        )
        if any(
            len(vector) != 32
            or any(type(status) is not int or not 0 <= status <= 9 for status in vector)
            for vector in decision_status_cuts
        ):
            raise ValueError("decision-state cuts must contain 32 statuses in 0..9")
        if (
            type(minimum_decision_status_hamming_distance) is not int
            or not 1 <= minimum_decision_status_hamming_distance <= 32
        ):
            raise ValueError("decision-state Hamming distance must be in 1..32")
        initial_excluded = _canonical_no_good_vectors(initial_excluded_decks)
        candidate_partition = _validated_candidate_partition_v1(candidate_partition)
        if scenario.origin.kind == "exact_trace_prefix" and initial_excluded:
            raise ValueError(
                "exact trace-prefix execution cannot use candidate no-goods"
            )
        started = time.perf_counter()
        cpu_started = time.process_time()
        stage_timings = {stage: 0.0 for stage in _EXECUTION_STAGE_TIMING_KEYS}
        # Local to this solve; never stored or accepted from a caller/result read.
        quality_verified_artifacts: set[str] = set()

        def record_stage(stage: str, stage_started: float) -> None:
            stage_timings[stage] += max(0.0, time.perf_counter() - stage_started)

        def stage_allowance() -> float:
            return max(
                0.001,
                min(stage_seconds, maximum_seconds - (time.perf_counter() - started)),
            )

        finish = partial(
            self._result,
            stage_timings=stage_timings,
            scenario_hash=scenario.sha256,
            proof_request_hash=proof_request.sha256,
            _quality_verified_artifacts=quality_verified_artifacts,
            _family_opening=proposal.get('familyOpening') if proposal is not None else None,
        )
        if progress_callback is not None:
            progress_callback(
                {"currentStage": "preflight", "stageTimings": dict(stage_timings)}
            )
        preflight = self.compiler.preflight(scenario)
        if self.backend_id == "ortools_cp_sat_v3":
            count_errors = tuple(
                {
                    "predicateId": predicate.predicate_id,
                    "code": code,
                }
                for predicate in scenario.predicates()
                if (code := hard_continuation_count_request_issue_v1(
                    predicate.kind,
                    predicate.hardness,
                    predicate.arguments,
                    proof_request.selected_action,
                    proof_request.line_universe,
                )) is not None
            )
            if count_errors:
                preflight = OrToolsResearchPreflightV1(
                    preflight.origin,
                    preflight.start_stage,
                    preflight.compiled_stages,
                    preflight.hard_predicate_ids,
                    preflight.soft_predicate_ids,
                    tuple(sorted(
                        (*preflight.unsupported, *count_errors),
                        key=lambda item: (item["predicateId"], item["code"]),
                    )),
                    preflight.lowering_manifest_hash,
                )
        adjudication_predicates = _adjudication_predicates_v3(scenario)
        if adjudication_predicates and self.adjudication_authority is None:
            preflight = OrToolsResearchPreflightV1(
                preflight.origin,
                preflight.start_stage,
                preflight.compiled_stages,
                preflight.hard_predicate_ids,
                preflight.soft_predicate_ids,
                tuple(
                    sorted(
                        (
                            *preflight.unsupported,
                            *(
                                {
                                    "predicateId": predicate.predicate_id,
                                    "code": "adjudication_authority_unavailable",
                                }
                                for predicate in adjudication_predicates
                            ),
                        ),
                        key=lambda item: (item["predicateId"], item["code"]),
                    )
                ),
                preflight.lowering_manifest_hash,
            )
        if cancel_check is not None and cancel_check():
            cancellation_core = {
                "schemaVersion": 1,
                "status": "incomplete",
                "stage": "execution",
                "exhaustionReason": "cancelled",
                "evidence": {"completedCandidates": 0},
            }
            cancellation_receipt = {
                **cancellation_core,
                "sha256": canonical_sha256(cancellation_core),
            }
            return finish(
                "incomplete",
                preflight,
                None,
                None,
                None,
                (),
                started,
                cpu_started,
                0,
                incomplete_receipt=cancellation_receipt,
            )
        if graph_compiler is None and maximum_proof_nodes is not None:
            unsupported = list(preflight.unsupported)
            actor_mode_supported = proof_request.mode == "exact" or (
                self.backend_id == "ortools_cp_sat_v3" and proof_request.mode == "bounded"
            )
            if (
                scenario.decision_point["kind"] != "play"
                and not _supports_default_nonplay_policy_graph(scenario, proof_request)
                and not _supports_default_nonplay_complete_graph(
                    scenario, proof_request
                )
                and not (
                    proof_request.visibility == "actor_information_set"
                    and actor_mode_supported
                    and proof_request.claim != "winning_continuation_line_count"
                )
            ):
                unsupported.append(
                    {
                        "predicateId": "PROOF-REQUEST",
                        "code": "nonplay_terminal_continuation_unavailable",
                    }
                )
            if proof_request.visibility == "actor_information_set":
                if not actor_mode_supported:
                    unsupported.append(
                        {
                            "predicateId": "PROOF-REQUEST",
                            "code": "actor_information_set_requires_exact_mode",
                        }
                    )
                if proof_request.claim == "winning_continuation_line_count":
                    unsupported.append(
                        {
                            "predicateId": "PROOF-REQUEST",
                            "code": "actor_information_set_continuation_count_unsupported",
                        }
                    )
                if maximum_hidden_worlds is None:
                    unsupported.append(
                        {
                            "predicateId": "PROOF-REQUEST",
                            "code": "actor_information_set_requires_world_budget",
                        }
                    )
            elif proof_request.visibility != "complete_state":
                unsupported.append(
                    {
                        "predicateId": "PROOF-REQUEST",
                        "code": "default_proof_requires_complete_state",
                    }
                )
            if unsupported != list(preflight.unsupported):
                preflight = OrToolsResearchPreflightV1(
                    preflight.origin,
                    preflight.start_stage,
                    preflight.compiled_stages,
                    preflight.hard_predicate_ids,
                    preflight.soft_predicate_ids,
                    tuple(
                        sorted(
                            unsupported,
                            key=lambda item: (item["predicateId"], item["code"]),
                        )
                    ),
                    preflight.lowering_manifest_hash,
                )
        if not preflight.supported:
            return finish(
                "incomplete", preflight, None, None, None, (), started, cpu_started, 0
            )
        if scenario.origin.kind == "search" and self.backend_id == _V3_BACKEND_ID:
            from .ortools_flagship_research import OrToolsFlagshipResearchV1

            counterstrategy_started = time.perf_counter()
            counterstrategy = OrToolsFlagshipResearchV1(rules=self.rules).certify_round_win_counterstrategy(
                scenario, proof_request, maximum_seconds=stage_allowance(),
                maximum_conflicts=maximum_conflicts, cancel_check=cancel_check,
            )
            if counterstrategy is not None:
                record_stage("proof", counterstrategy_started)
                finish = partial(finish, counterstrategy_receipt=counterstrategy)
                if counterstrategy["status"] == "certified":
                    return finish("proof_failed", preflight, None, None, None, (), started, cpu_started, 0,
                        candidate_attempt_reason="native_round_win_counterstrategy")
                if time.perf_counter() - started >= maximum_seconds or (cancel_check is not None and cancel_check()):
                    return finish("incomplete", preflight, None, None, None, (), started, cpu_started, 0,
                        candidate_attempt_reason="counterstrategy_budget")
        if scenario.origin.kind == "search" and self.backend_id != "ortools_cp_sat_v3":
            from .ortools_flagship_research import OrToolsFlagshipResearchV1

            flagship_research = OrToolsFlagshipResearchV1(rules=self.rules)
            contradiction_core = flagship_research.contradiction_core_for_scenario(
                scenario, proof_request
            )
            if contradiction_core is not None:
                theorem = flagship_research.contradiction_receipt()
                structural_core = {
                    "schemaVersion": 1,
                    "status": "exact_structural_infeasibility",
                    "stage": "preflight",
                    "theorem": theorem.to_dict(),
                    "namedCorePredicateIds": list(contradiction_core),
                    "scenarioHash": scenario.sha256,
                    "proofRequestHash": proof_request.sha256,
                }
                structural_receipt = {
                    **structural_core,
                    "sha256": canonical_sha256(structural_core),
                }
                return finish(
                    "infeasible",
                    preflight,
                    None,
                    None,
                    None,
                    (),
                    started,
                    cpu_started,
                    0,
                    structural_infeasibility_receipt=structural_receipt,
                )
        tail_plan = (
            TailFirstStagePlannerV1(
                native_cp_sat=self.backend_id == "ortools_cp_sat_v3",
            ).plan(scenario)
            if scenario.origin.kind == "search"
            else None
        )
        tail_first_active = (
            tail_plan is not None
            and self.runtime_proposals is None
            and candidate_compiler is None
            and context_compiler is None
            and graph_compiler is None
            and not any(
                predicate.kind
                in {
                    OUT_OF_TURN_PREDICATE_V3,
                    PURCHASE_BEFORE_TURN_PREDICATE_V3,
                    FALSE_EKKA_PREDICATE_V3,
                }
                for predicate in scenario.predicates()
            )
            and tail_plan.planner == "tail_first"
            and maximum_proof_nodes is not None
            and maximum_proof_depth is not None
            and proof_request.visibility == "complete_state"
        )
        # Construction must use only the configured bid boundary evaluator.
        require_auction_hand_policy = (
            self.quality_policy is not None and self.quality_policy.uses_auction_hand_boundary
        )
        partial_initial_input_choice = False
        generic_factory = None
        excluded_initial_inputs = []
        if tail_first_active:
            if maximum_prefix_lift_attempts is None:
                raise ValueError(
                    "tail-first Research requires maximum_prefix_lift_attempts"
                )
            from .ortools_tail_first_candidate import (
                OrToolsTailFirstCandidateCompilerV1,
            )

            candidate_factory = OrToolsTailFirstCandidateCompilerV1(
                rules=self.rules,
                plan=tail_plan,
                proof_request=proof_request,
                maximum_proof_nodes=maximum_proof_nodes,
                maximum_proof_depth=maximum_proof_depth,
                maximum_line_count_limbs=maximum_line_count_limbs,
                maximum_conflicts=maximum_conflicts,
                maximum_prefix_lift_attempts=maximum_prefix_lift_attempts,
                excluded_decision_status_vectors=decision_status_cuts,
                minimum_decision_status_hamming_distance=(
                    minimum_decision_status_hamming_distance
                ),
                candidate_partition=candidate_partition,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
                native_cp_sat_scoring=self.backend_id == "ortools_cp_sat_v3",
                tactical_detector_profile=self.tactical_detector_profile,
                policy_prefix_steps=_tail_early_policy_steps_v1(
                    tail_plan.prefix_length, self.quality_policy,
                    self.quality_execution,
                ),
                policy_id=(
                    self.quality_execution.early_play_policy_id
                    if self.quality_execution is not None
                    and self.quality_execution.early_play_policy_id is not None
                    else "observer_safe_trick_control_v1"
                ),
                require_optimal_auction_actions=require_auction_hand_policy,
            )
            for rejected in initial_quality_rejected_candidates:
                candidate_factory._reject_quality_candidate(rejected)
        else:
            if initial_quality_rejected_candidates:
                raise ValueError("quality play cuts require tail-first search")
            if scenario.origin.kind == "exact_trace_prefix":
                candidate_factory = candidate_compiler
            else:
                initial_dealer = Seat.EAST
                initial_match_score = (0, 0)
                if candidate_compiler is None and self.backend_id == _V3_BACKEND_ID:
                    try:
                        initial_dealer, initial_match_score, partial_initial_input_choice = _required_search_initial_inputs_v3(
                            scenario, rules=self.rules, deadline=time.monotonic() + maximum_seconds - (time.perf_counter() - started),
                            maximum_conflicts=maximum_conflicts, cancel_check=cancel_check)
                    except InformationSetCompilationStopped as stopped:
                        return finish("incomplete", preflight, None, None, None, (), started,
                            cpu_started, 0, incomplete_receipt=stopped.partial_receipt)
                    if initial_match_score is None:
                        return finish("infeasible", preflight, None, None, None, (), started,
                            cpu_started, 0, candidate_attempt_reason="initial_score_infeasible")
                if candidate_compiler is None:
                    generic_factory = partial(GenericCompleteTraceCompilerV1,
                        rules=self.rules,
                        minimum_deck_hamming_distance=minimum_deck_hamming_distance,
                        excluded_decision_status_vectors=decision_status_cuts,
                        minimum_decision_status_hamming_distance=(
                            minimum_decision_status_hamming_distance
                        ),
                        candidate_partition=candidate_partition,
                        native_cp_sat_scoring=(
                            self.backend_id == "ortools_cp_sat_v3"
                        ),
                        optimal_auction_quality=require_auction_hand_policy,
                        structural_only=self.quality_policy is not None,
                        manifest=self.lowering_manifest,
                    )
                candidate_factory = candidate_compiler if candidate_compiler is not None else generic_factory(
                    dealer=initial_dealer, match_score=initial_match_score)
        binding_factory = context_compiler or DefaultStageContextCompilerV1(
            self.rules,
            required_stages=preflight.compiled_stages,
            native_cp_sat_authority=self.backend_id == "ortools_cp_sat_v3",
            tactical_detector_profile=self.tactical_detector_profile,
            proof_request=proof_request if self.backend_id == "ortools_cp_sat_v3" else None,
            visibility_requests=tuple(
                (Seat.parse(predicate.arguments["observer"]), int(predicate.arguments["card"]))
                for predicate in scenario.predicates()
                if predicate.kind == "state.observer_card_visibility"
            ),
            actor_observer_scope_required=(proof_request.visibility == "actor_information_set"
                and any(predicate.kind == "state.observer_visibility"
                        or scenario.decision_point["kind"] == "play"
                        and self.lowering_manifest.definition(predicate.kind).stage in {"proof", "tactics"}
                        for predicate in scenario.predicates())),
        )
        excluded = list(initial_excluded)
        rejection_hashes: list[str] = []
        last_candidate = None
        last_evaluation = None
        last_proof = None
        last_graph = None
        last_proof_outcome = None
        last_failure_status = "not_found"
        last_candidate_receipt = None
        last_candidate_attempt_reason = None
        last_quality_handoff_diagnostic = None
        normalized_exact = None
        completed_candidates = 0
        nonoptimal_recorded_prefixes: dict[tuple[int, str, str], str] = {}
        refined_quality_attempts: dict[str, OrToolsCandidateAttemptV1] = {}

        def report_quality_checkpoint(row: Mapping[str, Any]) -> None:
            if row.get("kind") == "quality_proof_checkpoint_v1":
                artifact = row.get("artifact")
                if (isinstance(artifact, Mapping)
                        and artifact.get("proofModel", {}).get("kind") in {
                            "native_complete_state_root_outcome_v1",
                            "native_bounded_play_quality_model_v1",
                            "native_bid_selected_action_quality_model_v1"}
                        and row.get("artifactHash") == canonical_sha256(artifact)):
                    quality_verified_artifacts.add(row["artifactHash"])
            if row.get("kind") == "quality_solver_progress_v1":
                if progress_callback is not None:
                    progress_callback({"currentStage": "playingQuality",
                        "candidateHash": last_candidate.sha256,
                        "completedCandidates": completed_candidates,
                        "qualifiedCandidates": 0,
                        "proofCompletionStatus": "incomplete",
                        "qualitySolver": dict(row)})
            else:
                if row.get("kind") == "quality_decision_checkpoint_v1" and progress_callback is not None:
                    # Refresh progress without exposing private cards, legal choices or values.
                    progress_callback({"currentStage": "playingQuality",
                        "candidateHash": last_candidate.sha256,
                        "completedCandidates": completed_candidates,
                        "qualifiedCandidates": 0,
                        "proofCompletionStatus": "incomplete"})
                if checkpoint_callback is not None:
                    checkpoint_callback(row)

        def refine_quality_candidate(candidate, seconds):
            nonlocal last_candidate
            if seconds <= 0:
                return None
            last_candidate = candidate
            quality_verified_artifacts.clear()
            quality_started = time.perf_counter()
            try:
                quality_attempt = self.generate_quality_candidate(candidate, scenario=scenario,
                    maximum_seconds=seconds, cancel_check=cancel_check,
                    graph_workers=min(4, terminal_worker_count),
                    checkpoint_callback=report_quality_checkpoint,
                    _verified_artifacts=quality_verified_artifacts)
            finally:
                record_stage("playingQuality", quality_started)
            failure = quality_attempt.partial_receipt or {}
            if (quality_attempt.status == "not_found"
                    and quality_attempt.reason == "quality_generation_target_miss"
                    and failure.get("reason") == "hard_future_card_count_miss"
                    and failure.get("witnessActionSelection")
                    and failure.get("candidateHash") == candidate.sha256
                    and candidate.trace.get("roundNumber", 1) == 1):
                rejection_hashes.append(candidate.sha256)
                for stage, seconds in quality_attempt.stage_timings.items():
                    if stage in stage_timings:
                        stage_timings[stage] += max(0.0, float(seconds))
                return failure
            refined_quality_attempts[candidate.sha256] = quality_attempt
            return None

        if (tail_first_active and self.backend_id == _V3_BACKEND_ID
                and self.quality_policy is not None and self.quality_policy.mode == "budgeted"
                and self.quality_policy.visibility == "actor_information_set"
                and any(predicate.hardness == "hard" and predicate.kind == "play.trick_card_count"
                        for predicate in scenario.predicates())):
            candidate_factory._candidate_refinement = refine_quality_candidate

        for attempt_index in range(maximum_candidates):
            if self.runtime_proposals is not None and (proposal is None or attempt_index):
                break
            remaining = maximum_seconds - (time.perf_counter() - started)
            if remaining <= 0 or (cancel_check is not None and cancel_check()):
                reason = "maximum_seconds" if remaining <= 0 else "cancelled"
                receipt = {"schemaVersion": 1, "status": "incomplete",
                    "stage": "construction", "exhaustionReason": reason,
                    "evidence": {"completedCandidates": completed_candidates}}
                return finish("incomplete", preflight, last_candidate, last_evaluation, last_proof,
                    tuple(rejection_hashes), started, cpu_started, completed_candidates,
                    incomplete_receipt={**receipt, "sha256": canonical_sha256(receipt)},
                    candidate_attempt_reason=reason)
            if scenario.origin.kind == "exact_trace_prefix" or proposal_prefix is not None:
                if attempt_index:
                    break
                _validate_bid_decision_prefix_v1(
                    proposal_prefix or scenario.origin.exact_trace_prefix,
                    scenario.decision_point,
                )
                stage_started = time.perf_counter()
                try:
                    normalized = OrToolsExactStateNormalizerV1(
                        self.rules,
                        native_cp_sat=self.backend_id == "ortools_cp_sat_v3",
                    ).normalize(
                        proposal_prefix or scenario.origin.exact_trace_prefix,
                        decision_point=scenario.decision_point,
                        maximum_seconds=min(remaining, stage_seconds),
                        cancel_check=cancel_check,
                        **({"recorded_escalation_opportunities": scenario.origin.recorded_escalation_opportunities,
                            "opportunity_order": self.quality_execution.escalation_opportunity_order}
                           if self.backend_id == "ortools_cp_sat_v3" and self.quality_execution is not None
                           and scenario.decision_point["kind"] == "escalation"
                           and scenario.origin.recorded_escalation_opportunities else {}),
                    )
                except ExactNormalizationStopped as exc:
                    return finish(
                        "incomplete", preflight, None, None, None,
                        tuple(rejection_hashes), started, cpu_started, completed_candidates,
                        incomplete_receipt=exc.partial_receipt,
                    )
                finally:
                    record_stage("exactNormalization", stage_started)
                normalized_exact = normalized
                if proposal is not None and (normalized.next_actor != proposal["actor"]
                        or proposal["action"] not in {row["code"] for row in normalized.legal_actions}):
                    raise ValueError("runtime proposal focal action disagrees with native normalization")
                if (
                    scenario.decision_point["kind"] == "bid"
                    and normalized.state.get("phase") != GamePhase.BIDDING.value
                ):
                    raise ValueError("bid prefix must stop at a live auction decision")
                candidate = self._candidate_from_normalized_exact(
                    normalized, proposal_prefix or scenario.origin.exact_trace_prefix,
                    origin="search" if proposal_prefix is not None else "exact_trace_prefix",
                )
                attempt = OrToolsCandidateAttemptV1("feasible", candidate)
            else:
                assert candidate_factory is not None
                stage_started = time.perf_counter()
                try:
                    construction_seconds = min(remaining, stage_seconds)
                    construction_deadline = (time.monotonic() + construction_seconds
                        if generic_factory is not None and partial_initial_input_choice else None)
                    while True:
                        attempt = candidate_factory(
                            scenario, tuple(excluded), seed=seed + attempt_index,
                            maximum_seconds=construction_seconds, maximum_conflicts=maximum_conflicts,
                        )
                        if construction_deadline is None or attempt.status != "infeasible":
                            break
                        excluded_initial_inputs.append((int(initial_dealer), *initial_match_score))
                        try:
                            initial_dealer, initial_match_score, _partial = _required_search_initial_inputs_v3(
                                scenario, rules=self.rules, deadline=construction_deadline,
                                maximum_conflicts=maximum_conflicts, cancel_check=cancel_check,
                                excluded_inputs=excluded_initial_inputs)
                            construction_seconds = construction_deadline - time.monotonic()
                            if construction_seconds <= 0 or (cancel_check is not None and cancel_check()):
                                raise InformationSetCompilationStopped(
                                    "maximum_seconds" if construction_seconds <= 0 else "cancelled",
                                    stage="initial_state_inputs")
                        except InformationSetCompilationStopped as stopped:
                            attempt = OrToolsCandidateAttemptV1("incomplete", None, stopped.reason,
                                partial_receipt=stopped.partial_receipt)
                            break
                        if initial_match_score is None:
                            attempt = replace(attempt, status="not_found", reason="initial_input_choice_no_witness")
                            break
                        candidate_factory = generic_factory(dealer=initial_dealer, match_score=initial_match_score)
                        # A generated deck cut belongs to its starting state, not every score/dealer tuple.
                        excluded = list(initial_excluded)
                finally:
                    if not tail_first_active:
                        record_stage("construction", stage_started)
                for stage, seconds in attempt.stage_timings.items():
                    if stage in stage_timings:
                        stage_timings[stage] += max(0.0, float(seconds))
            if (attempt.status == "feasible" and self.backend_id == _V3_BACKEND_ID
                    and scenario.decision_point["kind"] in {"bid", "escalation", "project"}
                    and scenario.origin.kind == "exact_trace_prefix"
                    and not adjudication_predicates):
                input_stages = {
                    definition.stage for predicate in scenario.predicates()
                    if predicate.hardness == "hard"
                    and (definition := self.lowering_manifest.definition(predicate.kind)) is not None
                    and definition.stage not in _INPUT_PROJECTION_DEFERRED_STAGES
                    and definition.stage in attempt.candidate.stage_receipts
                }
                if input_stages:
                    stage_started = time.perf_counter()
                    try:
                        input_bindings = DefaultStageContextCompilerV1(
                            self.rules, required_stages=input_stages, native_cp_sat_authority=True,
                            proof_request=proof_request,
                            visibility_requests=tuple(
                                (Seat.parse(predicate.arguments["observer"]), int(predicate.arguments["card"]))
                                for predicate in scenario.predicates()
                                if predicate.kind == "state.observer_card_visibility"
                            ),
                        )(attempt.candidate, None)
                        input_miss = self.compiler.reject_input_projection(
                            scenario, attempt.candidate, input_bindings,
                            maximum_seconds=stage_allowance(), cancel_check=cancel_check,
                        )
                    finally:
                        record_stage("qualification", stage_started)
                    if input_miss is not None:
                        return finish(
                            "semantic_postcondition_miss", preflight, attempt.candidate,
                            input_miss, None, (*rejection_hashes, attempt.candidate.sha256),
                            started, cpu_started, completed_candidates + 1,
                        )
            if attempt.status == "feasible" and self.quality_policy is not None:
                if checkpoint_callback is not None:
                    checkpoint_callback({"schemaVersion": 1, "kind": "quality_stage_checkpoint_v1",
                        "rulesHash": self.rules.sha256, "stage": "construction",
                        "elapsedSeconds": time.perf_counter() - started,
                        "stageTimings": dict(stage_timings)})
                refined_quality_attempt = refined_quality_attempts.pop(attempt.candidate.sha256, None)
                if refined_quality_attempt is None:
                    quality_verified_artifacts.clear()
                construction_candidate = attempt.candidate
                construction_proof = attempt.proof_outcome
                last_candidate = attempt.candidate
                completed_candidates += 1

                if progress_callback is not None:
                    progress_callback({
                        "currentStage": "playingQuality",
                        "candidateHash": last_candidate.sha256,
                        "completedCandidates": completed_candidates,
                        "qualifiedCandidates": 0,
                        "proofCompletionStatus": "incomplete",
                    })
                quality_started = time.perf_counter()
                observed_nonoptimal: tuple[int, str, str, str] | None = None
                try:
                    quality_seconds = min(
                        self.quality_execution.maximum_wall_seconds,
                        maximum_seconds - (time.perf_counter() - started),
                    )
                    if ((quality_seconds <= 0 or (cancel_check is not None and cancel_check()))
                            and (refined_quality_attempt is None or refined_quality_attempt.status == "feasible")):
                        reason = "maximum_seconds" if quality_seconds <= 0 else "cancelled"
                        receipt = {"schemaVersion": 1, "status": "incomplete",
                            "stage": "playingQuality", "exhaustionReason": reason,
                            "candidateHash": attempt.candidate.sha256,
                            **({"qualityAttempt": refined_quality_attempt.partial_receipt}
                               if refined_quality_attempt is not None else {})}
                        return finish("incomplete", preflight, last_candidate, None, None,
                            tuple(rejection_hashes), started, cpu_started, completed_candidates,
                            incomplete_receipt={**receipt, "sha256": canonical_sha256(receipt)},
                            candidate_attempt_reason=reason)
                    if refined_quality_attempt is not None:
                        attempt = refined_quality_attempt
                    elif proposal_prefix is not None:
                        attempt = self.audit_quality_candidate(attempt.candidate,
                            graph_workers=min(4, terminal_worker_count),
                            recorded_trace=proposal_prefix, recorded_events=(),
                            decision_point=scenario.decision_point, continue_recorded_prefix=True,
                            maximum_seconds=quality_seconds, cancel_check=cancel_check,
                            stop_on_recorded_quality_failure=True,
                            checkpoint_callback=report_quality_checkpoint,
                            _verified_artifacts=quality_verified_artifacts)
                    elif scenario.origin.kind == "exact_trace_prefix":
                        attempt = self.audit_quality_candidate(attempt.candidate,
                            graph_workers=min(4, terminal_worker_count),
                            recorded_trace=scenario.origin.exact_trace_prefix,
                            recorded_events=scenario.origin.recorded_escalation_opportunities,
                            decision_point=scenario.decision_point,
                            continue_recorded_prefix=len(scenario.origin.exact_trace_prefix.play_actions) < 32,
                            maximum_seconds=quality_seconds, cancel_check=cancel_check,
                            stop_on_recorded_quality_failure=True,
                            checkpoint_callback=report_quality_checkpoint,
                            _verified_artifacts=quality_verified_artifacts)
                    else:
                        if ("tailFirstStagePlan" in attempt.candidate.trace
                            and (self.quality_execution.audit_constructed_trace or (not (self.backend_id == _V3_BACKEND_ID
                                     and attempt.candidate.origin == "search")
                            and not (
                            self.quality_policy.mode == "exact"
                            and self.quality_policy.visibility == "complete_state"
                            and self.quality_execution.complete_state_play_policy_pair
                        )))):
                            recorded = _construction_complete_prefix_v1(attempt.candidate)
                            repeated = next((
                                (key, proof_hash)
                                for key, proof_hash in nonoptimal_recorded_prefixes.items()
                                if key == _recorded_play_action_key_v1(recorded, key[0])
                            ), None)
                            if repeated is not None:
                                partial_core = {
                                    "schemaVersion": 1, "stage": "playingQuality",
                                    "qualityStatus": "quality_failed",
                                    "qualityFailureStage": "action_analysis",
                                    "qualityFailureReason": "recorded_action_not_optimal",
                                    "knownFailureProofArtifactHash": repeated[1],
                                    "candidateHash": attempt.candidate.sha256,
                                }
                                attempt = OrToolsCandidateAttemptV1(
                                    "not_found", None,
                                    reason="quality_known_nonoptimal_prefix",
                                    partial_receipt={**partial_core,
                                        "sha256": canonical_sha256(partial_core)},
                                )
                            else:
                                def capture_quality_checkpoint(row: Mapping[str, Any]) -> None:
                                    nonlocal observed_nonoptimal
                                    decision = row.get("decision")
                                    if (row.get("kind") == "quality_decision_checkpoint_v1"
                                            and row.get("stage") == "play"
                                            and isinstance(decision, Mapping)
                                            and decision.get("actualActionOptimal") is False
                                            and type(decision.get("playIndex")) is int
                                            and isinstance(decision.get("sourcePrefixHash"), str)
                                            and isinstance(decision.get("action"), str)
                                            and isinstance(decision.get("proofArtifactHash"), str)):
                                        observed_nonoptimal = (
                                            decision["playIndex"],
                                            decision["sourcePrefixHash"],
                                            decision["action"],
                                            decision["proofArtifactHash"],
                                        )
                                    report_quality_checkpoint(row)

                                attempt = self.audit_quality_candidate(
                                    attempt.candidate,
                                    graph_workers=min(4, terminal_worker_count),
                                    recorded_trace=recorded,
                                    recorded_events=(),
                                    continue_recorded_prefix=True,
                                    maximum_seconds=quality_seconds,
                                    decision_point=scenario.decision_point,
                                    cancel_check=cancel_check,
                                    stop_on_recorded_quality_failure=True,
                                    checkpoint_callback=capture_quality_checkpoint,
                                    _verified_artifacts=quality_verified_artifacts,
                                )
                        else:
                            # A constructed tail is a structural witness, not a
                            # player policy. V3 search chooses a fresh whole line
                            # under the frozen quality objective, then requalifies
                            # its scenario; exact inputs remain audited above.
                            attempt = self.generate_quality_candidate(
                                attempt.candidate,
                                graph_workers=min(4, terminal_worker_count),
                                maximum_seconds=quality_seconds,
                                scenario=scenario,
                                cancel_check=cancel_check,
                                checkpoint_callback=report_quality_checkpoint,
                                _verified_artifacts=quality_verified_artifacts,
                            )
                finally:
                    record_stage("playingQuality", quality_started)
                for stage, seconds in attempt.stage_timings.items():
                    if stage in stage_timings:
                        stage_timings[stage] += max(0.0, float(seconds))
                if (observed_nonoptimal is not None
                        and attempt.reason == "quality_recorded_audit_target_miss"
                        and isinstance(attempt.partial_receipt, Mapping)
                        and attempt.partial_receipt.get("qualityStatus") == "quality_failed"
                        and attempt.partial_receipt.get("qualityFailureReason") == "recorded_action_not_optimal"):
                    nonoptimal_recorded_prefixes[observed_nonoptimal[:3]] = observed_nonoptimal[3]
                if (
                    attempt.status == "feasible"
                    and construction_candidate is not None
                    and construction_proof is not None
                    and attempt.partial_receipt is not None
                    and attempt.partial_receipt.get("kind")
                    == "quality_handoff_source_v1"
                ):
                    last_quality_handoff_diagnostic = {
                        "schemaVersion": 1,
                        "kind": "quality_handoff_diagnostic_v1",
                        "qualificationAuthority": "generated_quality_trace",
                        "constructionCandidateHash": construction_candidate.sha256,
                        "constructionProofHash": construction_proof.sha256,
                        "constructionWinningRootActionCount": (
                            construction_proof.winning_root_action_count
                        ),
                        "generatedCandidateHash": attempt.candidate.sha256,
                        "firstConstructedPlayDivergenceIndex": (
                            attempt.partial_receipt.get(
                                "firstConstructedPlayDivergenceIndex"
                            )
                        ),
                    }
            if attempt.status != "feasible":
                if attempt.status == "not_found" and attempt.reason == "curated_interest_miss":
                    return finish("not_found", preflight, last_candidate, None, None,
                        tuple(rejection_hashes), started, cpu_started, completed_candidates,
                        incomplete_receipt=attempt.partial_receipt,
                        candidate_attempt_reason="curated_interest_miss")
                quality_miss = (
                    scenario.origin.kind == "search"
                    and attempt.status == "not_found"
                    and attempt.reason in {
                        "quality_generation_target_miss",
                        "quality_recorded_audit_target_miss",
                        "quality_known_nonoptimal_prefix",
                    }
                )
                if quality_miss and last_candidate is not None and last_candidate.no_good_vector:
                    if tail_first_active:
                        selector = scenario.decision_point["selector"]
                        failure = attempt.partial_receipt or {}
                        source = failure.get("sourcePrefix")
                        if (self.backend_id == _V3_BACKEND_ID
                                and self.quality_policy is not None
                                and self.quality_policy.mode == "budgeted"
                                and self.quality_policy.visibility == "actor_information_set"
                                and attempt.reason == "quality_generation_target_miss"
                                and failure.get("reason") in {"decision_point_target_miss", "hard_future_card_count_miss"}
                                and failure.get("candidateHash") == last_candidate.sha256
                                and isinstance(source, Mapping)
                                and source.get("deckOrder") == last_candidate.trace.get("deckOrder")
                                and source.get("dealer") == last_candidate.trace.get("dealer")
                                and source.get("roundNumber") == last_candidate.trace.get("roundNumber", 1)
                                and source.get("matchScore") == last_candidate.trace.get("matchScore", [0, 0])
                                and "trickIndex" in selector and "position" in selector
                                and not any(predicate.hardness == "hard"
                                    and predicate.kind == "auction.action_sequence"
                                    for predicate in scenario.predicates())):
                            # Fresh deterministic generation depends on the initial input, not its constructed play.
                            candidate_factory._reject_quality_candidate(last_candidate, reject_initial_input=True)
                        else:
                            candidate_factory._reject_quality_candidate(last_candidate)
                    last_failure_status = "semantic_postcondition_miss"
                    last_candidate_receipt = attempt.partial_receipt
                    last_candidate_attempt_reason = attempt.reason
                    rejection_hashes.append(last_candidate.sha256)
                    if not tail_first_active:
                        excluded.append(last_candidate.no_good_vector)
                    if progress_callback is not None:
                        progress_callback({
                            "currentStage": "candidate_terminal",
                            "candidateStatus": "semantic_postcondition_miss",
                            "candidateHash": last_candidate.sha256,
                            "rejectionReason": attempt.reason,
                            "completedCandidates": completed_candidates,
                            "qualifiedCandidates": 0,
                        })
                    continue
                status = (
                    "semantic_postcondition_miss"
                    if attempt.status == "not_found" and attempt.reason in {
                        "quality_generation_target_miss", "quality_recorded_audit_target_miss"}
                    else
                    "not_found"
                    if attempt.status == "infeasible" and (
                        partial_initial_input_choice or rejection_hashes
                        or self.backend_id == _V3_BACKEND_ID
                        and scenario.origin.kind == "search"
                        and (initial_excluded or decision_status_cuts
                             or initial_quality_rejected_candidates)
                    )
                    else attempt.status
                )
                return finish(
                    status,
                    preflight,
                    last_candidate,
                    last_evaluation,
                    last_proof,
                    tuple(rejection_hashes),
                    started,
                    cpu_started,
                    completed_candidates,
                    incomplete_receipt=attempt.partial_receipt,
                    candidate_attempt_reason=("initial_input_choice_no_witness"
                        if partial_initial_input_choice and attempt.status == "infeasible" else attempt.reason),
                )
            candidate = attempt.candidate
            assert candidate is not None
            if scenario.origin.kind == "exact_trace_prefix" and self.quality_policy is not None:
                candidate = _rebind_search_candidate_to_decision(candidate,
                    scenario.origin.exact_trace_prefix, normalized_exact,
                    preserve_construction_outcome=True,
                    native_cp_sat=self.backend_id == _V3_BACKEND_ID)
            completed_candidates += int(self.quality_policy is None)
            if (
                scenario.origin.kind == "search"
                and graph_compiler is None
                and maximum_proof_nodes is not None
            ):
                remaining = stage_allowance()
                try:
                    exact_prefix = _search_decision_prefix(
                        candidate,
                        scenario,
                        project_declaration_mode=(
                            self.rules.project_domain.declaration_mode
                            if scenario.decision_point["kind"] == "project"
                            else None
                        ),
                        native_cp_sat=self.backend_id == "ortools_cp_sat_v3",
                    )
                    _validate_bid_decision_prefix_v1(
                        exact_prefix,
                        scenario.decision_point,
                    )
                    stage_started = time.perf_counter()
                    try:
                        normalized_exact = OrToolsExactStateNormalizerV1(
                            self.rules,
                            native_cp_sat=self.backend_id == "ortools_cp_sat_v3",
                        ).normalize(
                            exact_prefix,
                            decision_point=scenario.decision_point,
                            maximum_seconds=remaining,
                            cancel_check=cancel_check,
                        )
                        _assert_strategic_project_decision_v1(
                            decision_kind=str(scenario.decision_point["kind"]),
                            declaration_mode=(
                                self.rules.project_domain.declaration_mode
                            ),
                            legal_actions=normalized_exact.legal_actions,
                            native_cp_sat=self.backend_id == "ortools_cp_sat_v3",
                        )
                        if (
                            scenario.decision_point["kind"] == "escalation"
                            and len(normalized_exact.legal_actions) < 2
                        ):
                            raise _SearchDecisionPrefixMiss(
                                "generated escalation root is not strategic"
                            )
                    finally:
                        record_stage("exactNormalization", stage_started)
                except ValueError:
                    last_candidate = candidate
                    last_failure_status = "semantic_postcondition_miss"
                    rejection_hashes.append(candidate.sha256)
                    if candidate.no_good_vector:
                        excluded.append(candidate.no_good_vector)
                        continue
                    return finish(
                        "incomplete",
                        preflight,
                        candidate,
                        None,
                        None,
                        tuple(rejection_hashes),
                        started,
                        cpu_started,
                        completed_candidates,
                    )
                except RuntimeError as exc:
                    return finish(
                        "incomplete",
                        preflight,
                        candidate,
                        None,
                        None,
                        tuple(rejection_hashes),
                        started,
                        cpu_started,
                        completed_candidates,
                        incomplete_receipt=(
                            exc.partial_receipt if isinstance(exc, ExactNormalizationStopped) else None
                        ),
                    )
                normalization = normalized_exact.to_dict()
                normalization_receipt = normalized_exact.normalization_receipt.to_dict()
                if (
                    scenario.decision_point["kind"] == "bid"
                    and normalized_exact.state.get("phase") != GamePhase.BIDDING.value
                ):
                    last_candidate = candidate
                    last_failure_status = "semantic_postcondition_miss"
                    rejection_hashes.append(candidate.sha256)
                    if candidate.no_good_vector:
                        excluded.append(candidate.no_good_vector)
                        continue
                    return finish(
                        "incomplete",
                        preflight,
                        candidate,
                        None,
                        None,
                        tuple(rejection_hashes),
                        started,
                        cpu_started,
                        completed_candidates,
                    )
                if self.quality_policy is not None or scenario.decision_point["kind"] in {
                    "bid",
                    "escalation",
                    "project",
                }:
                    candidate = _rebind_search_candidate_to_decision(
                        candidate,
                        exact_prefix,
                        normalized_exact,
                        preserve_construction_outcome=(
                            "playingQualityEvidence" in candidate.trace
                            or proof_request.mode in {"exact", "bounded"}
                        ),
                        native_cp_sat=self.backend_id == _V3_BACKEND_ID,
                    )
                else:
                    receipts = {
                        **candidate.stage_receipts,
                        "exactNormalization": _stage_receipt(
                            "exactNormalization",
                            model_hash=normalization_receipt["modelHash"],
                            evidence_hash=normalized_exact.sha256,
                            semantic={
                                **normalization,
                                "variableCount": normalization_receipt.get(
                                    "variableCount", 0
                                ),
                                "constraintCount": normalization_receipt.get(
                                    "constraintCount", 0
                                ),
                            },
                            solver_status="optimal",
                        ),
                    }
                    candidate = OrToolsNeutralCandidateV1(
                        candidate.origin,
                        {
                            **candidate.trace,
                            "exactTracePrefix": exact_prefix.to_dict(),
                            "exactNormalization": normalization,
                            "normalizationReceipt": normalization_receipt,
                        },
                        normalized_exact.state,
                        candidate.no_good_vector,
                        receipts,
                        normalized_exact.exact_state_id,
                    )
                if attempt.proof_state_projection_hash is not None:
                    if attempt.graph is None or maximum_proof_nodes is None:
                        return finish(
                            "incomplete",
                            preflight,
                            candidate,
                            None,
                            None,
                            tuple(rejection_hashes),
                            started,
                            cpu_started,
                            completed_candidates,
                        )
                    stage_started = time.perf_counter()
                    projection_hash = candidate_factory.complete_state_graph_compiler.proof_state_projection_hash(
                        normalized_exact,
                        expected_rules_hash=self.rules.sha256,
                    )
                    record_stage("graph", stage_started)
                    if projection_hash != attempt.proof_state_projection_hash:
                        return finish(
                            "incomplete",
                            preflight,
                            candidate,
                            None,
                            None,
                            tuple(rejection_hashes),
                            started,
                            cpu_started,
                            completed_candidates,
                        )
            interest_policy = getattr(self.quality_execution, "video_interestingness", None)
            if (self.backend_id == _V3_BACKEND_ID and scenario.origin.kind == "search"
                    and self.quality_policy is not None and isinstance(interest_policy, Mapping)
                    and interest_policy.get("schemaVersion") == 4
                    and (getattr(self.quality_execution, "family_target", None) is not None
                         or self.generation is not None)):
                from .decision_selection import evaluate_round_interestingness_v1

                interest = evaluate_round_interestingness_v1(
                    dict(candidate.trace.get("playingQualityEvidence") or {}).get("receipt", {}),
                    candidate.trace.get("score") or {}, interest_policy,
                    featured_state_id=candidate.exact_state_id or "")
                if not interest["passed"]:
                    # Editorial rejection is not a false or impossible Research claim.
                    candidate = replace(candidate, trace={**candidate.trace, "curatedSelection": {
                        "researchClaimStatus": "not_evaluated", "evidence": interest}})
                    return finish("not_found", preflight, candidate, None, None,
                        tuple(rejection_hashes), started, cpu_started, completed_candidates,
                        candidate_attempt_reason="curated_interest_miss")
            last_candidate = candidate
            proof_outcome = attempt.proof_outcome
            effective_graph_compiler = graph_compiler
            nonplay_continuation = None
            project_counterfactual_graph = None
            project_compilation_evidence = ()
            witness_root_action = None
            if proof_outcome is not None:
                graph = attempt.graph
            if (
                effective_graph_compiler is None
                and maximum_proof_nodes is not None
                and proof_request.visibility == "actor_information_set"
            ):
                if normalized_exact is None or maximum_hidden_worlds is None:
                    return finish(
                        "incomplete",
                        preflight,
                        candidate,
                        None,
                        None,
                        tuple(rejection_hashes),
                        started,
                        cpu_started,
                        completed_candidates,
                    )
                try:
                    expired = time.perf_counter() - started >= maximum_seconds
                    if expired or (cancel_check is not None and cancel_check()):
                        raise InformationSetCompilationStopped(
                            "maximum_seconds" if expired else "cancelled",
                            stage="information_graph_handoff")
                    remaining = stage_allowance()
                    stage_started = time.perf_counter()
                    try:
                        information_prefix = scenario.origin.exact_trace_prefix
                        if information_prefix is None:
                            raw_prefix = candidate.trace.get("exactTracePrefix")
                            if not isinstance(raw_prefix, Mapping):
                                raise ValueError("information proof requires an exact prefix")
                            information_prefix = ExactTracePrefixV1.build(raw_prefix)
                        compilation = compile_observed_decision_v1(
                            self.rules, normalized_exact, information_prefix,
                            scenario.decision_point, proof_request,
                            maximum_nodes=maximum_proof_nodes,
                            maximum_depth=maximum_proof_depth,
                            maximum_worlds=maximum_hidden_worlds,
                            maximum_seconds=remaining,
                            native_cp_sat=self.backend_id == "ortools_cp_sat_v3",
                            cancel_check=cancel_check,
                        )
                    finally:
                        record_stage("graph", stage_started)
                    expired = time.perf_counter() - started >= maximum_seconds
                    if expired or (cancel_check is not None and cancel_check()):
                        raise InformationSetCompilationStopped(
                            "maximum_seconds" if expired else "cancelled",
                            stage="information_proof_handoff")
                    remaining = stage_allowance()
                    stage_started = time.perf_counter()
                    try:
                        proof_outcome = prove_actor_information_set_v1(
                            compilation,
                            proof_request,
                            maximum_nodes=maximum_proof_nodes,
                            maximum_worlds=maximum_hidden_worlds,
                            maximum_seconds=remaining,
                            cancel_check=cancel_check,
                            native_action_quality=self.backend_id == "ortools_cp_sat_v3",
                            **({"native_opponent_stop": True} if self.backend_id == "ortools_cp_sat_v3"
                               and any(predicate.kind in {"tactic.applicable", "tactic.verified"}
                                   and predicate.arguments.get("kind") == "kabout_prevention"
                                   for predicate in scenario.predicates()) else {}),
                            policy_search=policy_search,
                            **({"declared_maximum_seconds": maximum_seconds}
                               if self.backend_id == "ortools_cp_sat_v3" else {}),
                        )
                    finally:
                        record_stage("proof", stage_started)
                    graph = compilation.graph
                except InformationSetCompilationStopped as exc:
                    return finish(
                        "incomplete",
                        preflight,
                        candidate,
                        None,
                        None,
                        tuple(rejection_hashes),
                        started,
                        cpu_started,
                        completed_candidates,
                        incomplete_receipt=exc.partial_receipt,
                    )
                except (RuntimeError, ValueError):
                    return finish(
                        "incomplete",
                        preflight,
                        candidate,
                        None,
                        None,
                        tuple(rejection_hashes),
                        started,
                        cpu_started,
                        completed_candidates,
                    )
            elif effective_graph_compiler is None and maximum_proof_nodes is not None:

                def compile_default_graph(
                    _candidate: OrToolsNeutralCandidateV1,
                    request: ScenarioProofRequestV1,
                ) -> ProofGraphV1:
                    nonlocal nonplay_continuation, project_counterfactual_graph, project_compilation_evidence
                    if normalized_exact is None:
                        raise RuntimeError("exact state normalization is unavailable")
                    decision_kind = str(scenario.decision_point["kind"])
                    if decision_kind in {"bid", "escalation", "project"}:
                        exact_prefix = scenario.origin.exact_trace_prefix
                        if exact_prefix is None:
                            raw_prefix = _candidate.trace.get("exactTracePrefix")
                            if not isinstance(raw_prefix, Mapping):
                                raise RuntimeError(
                                    "non-play policy proof requires a normalized prefix"
                                )
                            exact_prefix = ExactTracePrefixV1.build(raw_prefix)
                        selector = dict(scenario.decision_point.get("selector") or {})
                        budget = NonPlayGraphBudgetV1(
                            maximum_nodes=maximum_proof_nodes,
                            maximum_depth=maximum_proof_depth,
                        )
                        if request.mode in {"exact", "bounded"}:
                            paired_project = self.backend_id == "ortools_cp_sat_v3" and any(
                                item.kind in {"tactic.verified", "tactic.applicable"}
                                and item.arguments.get("kind") == "project_bidding"
                                for item in scenario.predicates()
                            )
                            compiled = compile_nonplay_complete_graph_v1(
                                self.rules,
                                normalized_exact,
                                exact_prefix,
                                scenario.decision_point,
                                request,
                                maximum_nodes=maximum_proof_nodes,
                                maximum_depth=maximum_proof_depth,
                                maximum_seconds=stage_allowance(),
                                native_cp_sat=(
                                    self.backend_id == "ortools_cp_sat_v3"
                                ),
                                cancel_check=cancel_check,
                                **({"project_counterfactual": True} if paired_project else {}),
                            )
                            if paired_project:
                                project_counterfactual_graph = compiled.project_counterfactual_graph
                                if project_counterfactual_graph is None:
                                    raise ValueError("native project proof omitted its counterfactual graph")
                                project_compilation_evidence = compiled.project_compilation_evidence
                                if not project_compilation_evidence:
                                    raise ValueError("native project proof omitted its compilation evidence")
                            return compiled.graph
                        continuation = OrToolsNonPlayPolicyContinuationV1(
                            self.rules,
                            exact_prefix,
                            request,
                            normalized_exact.state,
                            native_cp_sat=self.backend_id == "ortools_cp_sat_v3",
                            legal_actions=normalized_exact.legal_actions,
                            maximum_seconds=stage_allowance(),
                        )
                        nonplay_continuation = continuation
                        if decision_kind == "bid":
                            layout = DealLayoutV1.build(
                                self.rules.structural_domain,
                                dealer=Seat.parse(exact_prefix.dealer),
                            )
                            purchase_suit = Card.parse(
                                exact_prefix.deck_order[layout.purchase_position]
                            ).suit
                            prefix = tuple(
                                AuctionGraphChoiceV1.build(
                                    item["action"], item.get("trumpSuit")
                                )
                                for item in exact_prefix.auction_actions
                            )
                            auction_remaining = stage_allowance()
                            if self.backend_id == "ortools_cp_sat_v3" and auction_remaining <= 0:
                                raise NonPlayGraphCompilationStoppedV1("maximum_seconds", nodes=0, depth=0)
                            return compile_auction_decision_graph_v1(
                                AuctionDomainIRV1.from_rules(self.rules.data),
                                dealer=exact_prefix.dealer,
                                purchase_suit=purchase_suit,
                                purchase_rank=Card.parse(exact_prefix.deck_order[layout.purchase_position]).rank,
                                terminal_utility=continuation.auction_utility,
                                prefix=prefix,
                                scope="exact",
                                budget=budget,
                                continuation_action_selector=(
                                    continuation.continuation_action
                                ),
                                root_action_code=witness_root_action,
                                root_worker_count=terminal_worker_count,
                                native_cp_sat=self.backend_id == "ortools_cp_sat_v3",
                                maximum_seconds=auction_remaining if self.backend_id == "ortools_cp_sat_v3" else None,
                                cancel_check=cancel_check,
                            )
                        if decision_kind == "escalation":
                            buyer = normalized_exact.state.get("buyer")
                            contract = normalized_exact.state.get("contract")
                            actor = selector.get("actingSeat")
                            if buyer is None or contract is None or actor is None:
                                raise ValueError("escalation proof state is incomplete")
                            prefix = tuple(
                                EscalationGraphActionV1.build(
                                    item["actor"],
                                    item["action"],
                                    item.get("mode", item.get("requestedMode")),
                                )
                                for item in exact_prefix.escalation_actions
                            )
                            return compile_escalation_decision_graph_v1(
                                EscalationDomainIRV1.from_rules(self.rules.data),
                                declarer=buyer,
                                contract=contract,
                                match_score=exact_prefix.match_score,
                                acting_seat=actor,
                                terminal_utility=continuation.escalation_utility,
                                prefix=prefix,
                                native_cp_sat=self.backend_id == "ortools_cp_sat_v3",
                                scope="exact",
                                budget=budget,
                                continuation_action_selector=(
                                    continuation.continuation_action
                                ),
                                root_action_code=witness_root_action,
                                root_worker_count=terminal_worker_count,
                            )
                        actor = selector.get("owner")
                        if actor is None:
                            raise ValueError("project proof actor is missing")
                        return compile_project_declaration_graph_v1(
                            self.rules.project_domain,
                            projects=continuation.detected_projects(),
                            actor=actor,
                            dealer=exact_prefix.dealer,
                            prefix=exact_prefix.project_actions,
                            trump_suit=normalized_exact.state.get("trumpSuit"),
                            terminal_utility=continuation.project_utility,
                            scope="exact",
                            budget=budget,
                            continuation_action_selector=(
                                continuation.continuation_action
                            ),
                            root_action_code=witness_root_action,
                        )
                    graph_remaining = stage_allowance()
                    compiled = OrToolsCompleteStateGraphCompilerV1(
                        native_conversion=self.backend_id == "ortools_cp_sat_v3",
                        play_domain=self.rules.play_domain,
                        scoring_domain=ScoringDomainIRV1.from_rules(self.rules.data),
                        structural_domain=self.rules.structural_domain,
                        project_domain=self.rules.project_domain,
                    ).compile(
                        normalized_exact,
                        request,
                        expected_rules_hash=self.rules.sha256,
                        maximum_nodes=maximum_proof_nodes,
                        maximum_depth=maximum_proof_depth,
                        maximum_seconds=graph_remaining,
                        cancelled=cancel_check,
                    )
                    if not compiled.complete or compiled.graph is None:
                        if compiled.receipt.exhaustion_reason == "cancelled":
                            raise InterruptedError("complete-state graph compilation cancelled")
                        raise RuntimeError(
                            "complete-state graph compilation did not finish: "
                            + str(compiled.receipt.exhaustion_reason)
                        )
                    return compiled.graph

                effective_graph_compiler = compile_default_graph
            if proof_outcome is not None or effective_graph_compiler is not None:
                proof_stage = "graph"
                try:
                    if proof_outcome is None:
                        assert effective_graph_compiler is not None
                        if time.perf_counter() - started >= maximum_seconds:
                            raise TimeoutError("campaign expired before graph construction")
                        if cancel_check is not None and cancel_check():
                            raise InterruptedError("campaign cancelled before graph construction")
                        stage_started = time.perf_counter()
                        try:
                            graph = effective_graph_compiler(candidate, proof_request)
                        finally:
                            record_stage("graph", stage_started)
                        proof_stage = "proof"
                        if time.perf_counter() - started >= maximum_seconds:
                            raise TimeoutError("campaign expired before graph proof")
                        if cancel_check is not None and cancel_check():
                            raise InterruptedError("campaign cancelled before graph proof")
                        remaining = stage_allowance()
                        stage_started = time.perf_counter()
                        try:
                            native_configure = (
                                _native_proof_tactic_configurer_v3(
                                    scenario=scenario, candidate=candidate, graph=graph, request=proof_request,
                                    rules=self.rules, manifest=self.compiler.manifest,
                                    tactical_detector_profile=self.tactical_detector_profile)
                                if self.backend_id == "ortools_cp_sat_v3" else None
                            )
                            proof_outcome = OrToolsScenarioGraphProofV1().prove(
                                graph,
                                proof_request,
                                **({"project_counterfactual_graph": project_counterfactual_graph,
                                    "project_compilation_evidence": project_compilation_evidence}
                                   if project_counterfactual_graph is not None else {}),
                                maximum_seconds=remaining,
                                cancel_check=cancel_check,
                                maximum_line_count_limbs=maximum_line_count_limbs,
                                tactic_configure=native_configure,
                                native_root_only=native_configure is not None and graph.tactic_context is None,
                                native_line_count=(
                                    self.backend_id == "ortools_cp_sat_v3"
                                    and any(
                                        predicate.hardness == "hard"
                                        and predicate.kind == "proof.winning_continuation_line_count"
                                        for predicate in scenario.predicates()
                                    )
                                ),
                            )
                        finally:
                            record_stage("proof", stage_started)
                except (RuntimeError, ValueError, TimeoutError, InterruptedError) as exc:
                    incomplete_core = {
                        "schemaVersion": 1,
                        "status": "incomplete",
                        "stage": proof_stage,
                        "exhaustionReason": (exc.partial_receipt["exhaustionReason"]
                            if isinstance(exc, ExactNormalizationStopped)
                            else "maximum_seconds" if isinstance(exc, TimeoutError)
                            else "cancelled" if isinstance(exc, InterruptedError)
                            else f"{proof_stage}_compilation_failed"),
                        "evidence": {
                            "exceptionType": type(exc).__name__,
                            "detail": str(exc),
                            **({"causeReceipt": copy.deepcopy(exc.partial_receipt)}
                               if isinstance(exc, ExactNormalizationStopped) else {}),
                        },
                    }
                    return finish(
                        "incomplete",
                        preflight,
                        candidate,
                        None,
                        None,
                        tuple(rejection_hashes),
                        started,
                        cpu_started,
                        completed_candidates,
                        incomplete_receipt={
                            **incomplete_core,
                            "sha256": canonical_sha256(incomplete_core),
                        },
                    )
                proof_evidence_bindings = _proof_receipt_evidence_bindings_v3(
                    proof_outcome,
                    proof_request,
                )
                proof_receipt = _stage_receipt(
                    "proof",
                    model_hash=proof_outcome.model_hash,
                    evidence_hash=proof_outcome.sha256,
                    semantic=proof_outcome.to_dict(),
                    solver_status=proof_outcome.solver_status,
                    variable_count=proof_outcome.variable_count,
                    constraint_count=proof_outcome.constraint_count,
                    evidence_bindings=proof_evidence_bindings,
                )
                tactic_receipt = _stage_receipt(
                    "tactics",
                    model_hash=proof_outcome.model_hash,
                    evidence_hash=proof_outcome.sha256,
                    semantic=proof_outcome.to_dict(),
                    solver_status=proof_outcome.solver_status,
                    variable_count=proof_outcome.variable_count,
                    constraint_count=proof_outcome.constraint_count,
                    evidence_bindings=tuple(
                        item
                        for item in proof_evidence_bindings
                        if str(item.get("predicateId", "")).startswith("PRED-")
                    ),
                )
                witness_only = False
                if (
                    nonplay_continuation is None
                    and "playingQualityEvidence" not in candidate.trace
                    and self.witness_policies is not None
                    and scenario.decision_point["kind"]
                    in {"bid", "escalation", "project"}
                    and proof_request.mode in {"exact", "bounded"}
                    and proof_request.visibility == "complete_state"
                    and proof_outcome.solver_status == "optimal"
                    and proof_outcome.principal_variation
                ):
                    witness_root_action = proof_outcome.principal_variation[0]
                    witness_request = ScenarioProofRequestV1.build(
                        {
                            "schemaVersion": 1,
                            "mode": "policy_conditioned",
                            "visibility": "complete_state",
                            "strategyClass": "frozen_policy",
                            "claim": "outcome_reachable",
                            "targetTeam": proof_request.target_team,
                            "utility": proof_request.utility,
                            "successThreshold": proof_request.success_threshold,
                            "policies": self.witness_policies,
                            "budgetProfile": proof_request.budget_profile,
                        }
                    )
                    stage_started = time.perf_counter()
                    try:
                        compile_default_graph(candidate, witness_request)
                    except (RuntimeError, ValueError) as exc:
                        incomplete_core = {
                            "schemaVersion": 1,
                            "status": "incomplete",
                            "stage": "finalNormalization",
                            "exhaustionReason": "witness_continuation_failed",
                            "evidence": {
                                "exceptionType": type(exc).__name__,
                                "detail": str(exc),
                            },
                        }
                        return finish(
                            "incomplete",
                            preflight,
                            candidate,
                            None,
                            proof_outcome.to_dict(),
                            tuple(rejection_hashes),
                            started,
                            cpu_started,
                            completed_candidates,
                            graph=graph,
                            proof_outcome=proof_outcome,
                            incomplete_receipt={
                                **incomplete_core,
                                "sha256": canonical_sha256(incomplete_core),
                            },
                        )
                    finally:
                        record_stage("finalNormalization", stage_started)
                    witness_only = True
                if (
                    scenario.decision_point["kind"] == "play"
                    # Audited full rounds already own their continuation. Final
                    # result admission still validates their complete evidence.
                    and "playingQualityEvidence" not in candidate.trace
                    and proof_outcome.solver_status == "optimal"
                    and normalized_exact is not None
                    and graph is not None
                    and (
                        (
                            proof_request.mode == "policy_conditioned"
                            and proof_outcome.completion_status == "policy_complete"
                        )
                        or (
                            proof_request.mode == "exact"
                            and proof_request.visibility == "complete_state"
                            and proof_outcome.completion_status == "exact_complete"
                        )
                    )
                ):
                    stage_started = time.perf_counter()
                    try:
                        candidate = _materialize_play_policy_continuation_v1(
                            rules=self.rules,
                            candidate=candidate,
                            source_normalization=normalized_exact,
                            graph=graph,
                            proof=proof_outcome,
                            request=proof_request,
                            native_cp_sat=self.backend_id == "ortools_cp_sat_v3",
                            cancel_check=cancel_check,
                            maximum_seconds=stage_allowance(),
                        )
                    except _PolicyContinuationCompilerDisagreement:
                        return finish(
                            "compiler_disagreement",
                            preflight,
                            candidate,
                            None,
                            proof_outcome.to_dict(),
                            tuple(rejection_hashes),
                            started,
                            cpu_started,
                            completed_candidates,
                            graph=graph,
                            proof_outcome=proof_outcome,
                        )
                    except RuntimeError as exc:
                        return finish(
                            "incomplete",
                            preflight,
                            candidate,
                            None,
                            proof_outcome.to_dict(),
                            tuple(rejection_hashes),
                            started,
                            cpu_started,
                            completed_candidates,
                            graph=graph,
                            proof_outcome=proof_outcome,
                            incomplete_receipt=(
                                exc.partial_receipt if isinstance(exc, ExactNormalizationStopped) else None
                            ),
                        )
                    finally:
                        record_stage("exactNormalization", stage_started)
                trace = candidate.trace
                continuation_receipt = None
                complete_exact_normalization = None
                selected_nonplay_continuation = None
                if nonplay_continuation is not None:
                    selected_nonplay_continuation = (
                        nonplay_continuation.selected_root_continuation(
                            proof_outcome.principal_variation[0]
                        )
                        if witness_only
                        else nonplay_continuation.selected_continuation(
                            proof_outcome.principal_variation
                        )
                    )
                if selected_nonplay_continuation is not None and (
                    "playingQualityEvidence" not in candidate.trace
                    or selected_nonplay_continuation.get("allPassed") is True
                ):
                    continuation_receipt = selected_nonplay_continuation
                    continuation_key = (
                        "witnessContinuation"
                        if witness_only
                        else "policyContinuation"
                    )
                    if continuation_receipt.get("allPassed") is True:
                        trace = {
                            **candidate.trace,
                            continuation_key: continuation_receipt,
                            "auctionCompletion": continuation_receipt,
                        }
                    else:
                        stage_started = time.perf_counter()
                        try:
                            complete_exact_normalization = (
                                OrToolsExactStateNormalizerV1(
                                    self.rules,
                                    native_cp_sat=(
                                        self.backend_id == "ortools_cp_sat_v3"
                                    ),
                                ).normalize(
                                    continuation_receipt["exactTracePrefix"],
                                    decision_point={"kind": "play"},
                                    cancel_check=cancel_check,
                                    maximum_seconds=stage_allowance(),
                                )
                            )
                        except ExactNormalizationStopped as exc:
                            return finish(
                                "incomplete", preflight, candidate, None, proof_outcome.to_dict(),
                                tuple(rejection_hashes), started, cpu_started, completed_candidates,
                                graph=graph, proof_outcome=proof_outcome,
                                incomplete_receipt=exc.partial_receipt,
                            )
                        finally:
                            record_stage("finalNormalization", stage_started)
                        complete_play = (
                            complete_exact_normalization.normalization_receipt.to_dict().get(
                                "derivedPlay"
                            )
                        )
                        if not isinstance(complete_play, Mapping):
                            raise _PolicyContinuationCompilerDisagreement(
                                "non-play continuation lacks a complete play trace"
                            )
                        trace = {
                            **candidate.trace,
                            "score": dict(continuation_receipt["score"]),
                            "completePlay": copy.deepcopy(dict(complete_play)),
                            continuation_key: continuation_receipt,
                            "auctionCompletion": continuation_receipt,
                            "completeExactTracePrefix": continuation_receipt[
                                "exactTracePrefix"
                            ],
                            "completeExactNormalization": (
                                complete_exact_normalization.to_dict()
                            ),
                        }
                receipts = {
                    **candidate.stage_receipts,
                    "proof": proof_receipt,
                    "tactics": tactic_receipt,
                }
                if continuation_receipt is not None:
                    continuation_key = (
                        "witnessContinuation"
                        if witness_only
                        else "policyContinuation"
                    )
                    if continuation_receipt.get("allPassed") is not True:
                        continuation_score = dict(continuation_receipt["score"])
                        score_hash = canonical_sha256(continuation_score)
                        receipts["outcome"] = _stage_receipt(
                            "outcome",
                            model_hash=str(continuation_receipt["sha256"]),
                            evidence_hash=score_hash,
                            semantic=continuation_score,
                            solver_status="optimal",
                        )
                    receipts[continuation_key] = _stage_receipt(
                        continuation_key,
                        model_hash=canonical_sha256(
                            {
                                "kind": (
                                    "declarative_policy_witness_continuation_v1"
                                    if witness_only
                                    else "declarative_policy_terminal_continuation_v1"
                                ),
                                "proofRequestHash": proof_request.sha256,
                            }
                        ),
                        evidence_hash=str(continuation_receipt["sha256"]),
                        semantic=continuation_receipt,
                        solver_status="optimal",
                    )
                    if scenario.decision_point["kind"] == "bid":
                        receipts["auctionCompletion"] = _stage_receipt(
                            "auctionCompletion",
                            model_hash=str(continuation_receipt["sha256"]),
                            evidence_hash=str(continuation_receipt["sha256"]),
                            semantic=continuation_receipt,
                            solver_status="optimal",
                        )
                candidate = OrToolsNeutralCandidateV1(
                    candidate.origin,
                    trace,
                    candidate.state,
                    candidate.no_good_vector,
                    receipts,
                    candidate.exact_state_id,
                )
                last_candidate = candidate
                last_proof = proof_outcome.to_dict()
                last_graph = graph
                last_proof_outcome = proof_outcome
                if proof_outcome.completion_status == "model_error":
                    return finish(
                        "compiler_disagreement",
                        preflight,
                        candidate,
                        None,
                        last_proof,
                        tuple(rejection_hashes),
                        started,
                        cpu_started,
                        completed_candidates,
                        graph=last_graph,
                        proof_outcome=last_proof_outcome,
                    )
                stopping = proof_outcome.model_document().get("opponentStopCertificate")
                stopping_incomplete = (
                    self.backend_id == "ortools_cp_sat_v3"
                    and proof_outcome.proof_engine_id == "ortools_actor_information_set_cegis_v1"
                    and isinstance(stopping, Mapping)
                    and stopping.get("completionStatus") == "proof_incomplete"
                )
                stopping_receipt = None
                if stopping_incomplete:
                    stopping_receipt = {"schemaVersion": 1, "status": "incomplete",
                        "stage": "opponent_stopping_proof",
                        "exhaustionReason": stopping.get("exhaustionReason", "proof_incomplete"),
                        "evidence": {"proofHash": proof_outcome.sha256, "certificateHash": stopping["sha256"]}}
                    stopping_receipt["sha256"] = canonical_sha256(stopping_receipt)
                if proof_outcome.completion_status == "proof_incomplete" or stopping_incomplete:
                    return finish(
                        "incomplete",
                        preflight,
                        candidate,
                        None,
                        last_proof,
                        tuple(rejection_hashes),
                        started,
                        cpu_started,
                        completed_candidates,
                        graph=last_graph,
                        proof_outcome=last_proof_outcome,
                        incomplete_receipt=stopping_receipt,
                    )
            stage_started = time.perf_counter()
            try:
                evaluation = (
                    _native_same_solve_evaluation_v3(
                        scenario,
                        candidate,
                        proof_outcome,
                        attempt.predicate_evidence,
                        tactical_detector_profile=self.tactical_detector_profile,
                    )
                    if self.backend_id == "ortools_cp_sat_v3"
                    and not adjudication_predicates
                    else None
                )
                if evaluation is None:
                    bindings = binding_factory(candidate, proof_outcome,
                        **({"information_graph": graph if proof_outcome is not None
                            and isinstance(graph, ActorInformationSetGraphV1) else None}
                           if isinstance(binding_factory, DefaultStageContextCompilerV1) else {}))
                    if adjudication_predicates:
                        assert self.adjudication_authority is not None
                        bindings = _wrap_adjudication_stage_bindings_v3(
                            scenario,
                            candidate,
                            bindings,
                            self.adjudication_authority,
                        )
                    evaluation = self.compiler.evaluate(
                        scenario,
                        candidate,
                        bindings,
                        maximum_seconds=remaining,
                        cancel_check=cancel_check,
                    )
            finally:
                record_stage("qualification", stage_started)
            last_evaluation = evaluation
            if evaluation.status == "incomplete":
                return finish(
                    "incomplete",
                    preflight,
                    candidate,
                    evaluation,
                    last_proof,
                    tuple(rejection_hashes),
                    started,
                    cpu_started,
                    completed_candidates,
                    graph=last_graph,
                    proof_outcome=last_proof_outcome,
                )
            if not evaluation.satisfied:
                last_failure_status = "semantic_postcondition_miss"
            elif proof_outcome is None:
                return finish(
                    "feasible",
                    preflight,
                    candidate,
                    evaluation,
                    None,
                    tuple(rejection_hashes),
                    started,
                    cpu_started,
                    completed_candidates,
                )
            elif proof_outcome.completion_status == "proof_incomplete":
                return finish(
                    "incomplete",
                    preflight,
                    candidate,
                    evaluation,
                    last_proof,
                    tuple(rejection_hashes),
                    started,
                    cpu_started,
                    completed_candidates,
                    graph=last_graph,
                    proof_outcome=last_proof_outcome,
                )
            elif not proof_outcome.claim_satisfied:
                last_failure_status = "proof_failed"
            else:
                return finish(
                    "qualified",
                    preflight,
                    candidate,
                    evaluation,
                    last_proof,
                    tuple(rejection_hashes),
                    started,
                    cpu_started,
                    completed_candidates,
                    graph=last_graph,
                    proof_outcome=last_proof_outcome,
                    _lesson_normalized=normalized_exact,
                )
            rejection_hashes.append(candidate.sha256)
            if candidate.no_good_vector:
                excluded.append(candidate.no_good_vector)
            elif scenario.origin.kind == "search":
                return finish(
                    "incomplete",
                    preflight,
                    candidate,
                    evaluation,
                    last_proof,
                    tuple(rejection_hashes),
                    started,
                    cpu_started,
                    completed_candidates,
                    graph=last_graph,
                    proof_outcome=last_proof_outcome,
                )
        return finish(
            last_failure_status,
            preflight,
            last_candidate,
            last_evaluation,
            last_proof,
            tuple(rejection_hashes),
            started,
            cpu_started,
            completed_candidates,
            graph=last_graph,
            proof_outcome=last_proof_outcome,
            incomplete_receipt=last_candidate_receipt,
            candidate_attempt_reason=last_candidate_attempt_reason,
            quality_handoff_diagnostic=last_quality_handoff_diagnostic,
        )

    def _validate_runtime_proposal_scope(self, scenario: ResearchScenarioV1) -> None:
        if self.generation is not None:
            from .family_catalog import ResearchFamilyCatalogV1, matching_generation_families_v1
            matching_generation_families_v1(self.generation,
                ResearchFamilyCatalogV1.build(self.quality_execution.family_catalog), scenario,
                play_indices=self.quality_execution.proposal_play_indices,
                rules=self.rules, execution=self.quality_execution, playing_quality=self.quality_policy)
        search_play = scenario.origin.kind == "search" and scenario.decision_point["kind"] == "play"
        if self.runtime_proposals is not None and not search_play:
            raise ValueError("runtime proposals require a search play scenario")
        if (search_play and self.quality_execution is not None
                and self.quality_execution.candidate_source is not None
                and self.runtime_proposals is None):
            raise ValueError("runtime proposal queue is required by candidateSource")

    def execute_proposal_stream(
        self, scenario, proof_request, *, proposals: Iterable[tuple[int, Mapping[str, Any]]],
        proof_worker_count: int, maximum_campaign_seconds: float,
        maximum_candidates: int, result_count: int, symmetry_policy_id: str,
        allow_symmetry_only_variants: bool, minimum_decision_state_distance: int,
        minimum_principal_variation_action_distance: int | None = None,
        cancel_check=None, progress_callback=None, cancellation_poll_milliseconds=25,
        **proof_options,
    ) -> OrToolsResearchCampaignResultV1:
        """Opt-in bounded pipeline; the caller reserves producer CPU separately.

        The parent alone advances the cooperative proposal iterator. Generation
        time consumes the same deadline as proof; only validated singleton
        proposals enter isolated children. Final selection is semantic-hash
        ordered, never first-finisher ordered.
        """
        if (self.backend_id != _V3_BACKEND_ID or self.runtime_proposals is not None
                or self.quality_execution is None
                or self.quality_execution.candidate_source not in {"runtime_proposal_v1", "runtime_proposal_stream_v1"}
                or scenario.origin.kind != "search"
                or scenario.decision_point["kind"] != "play"):
            raise ValueError("proposal streaming requires native search-play candidateSource")
        for name, value in (("proof_worker_count", proof_worker_count),
                            ("maximum_candidates", maximum_candidates),
                            ("result_count", result_count),
                            ("cancellation_poll_milliseconds", cancellation_poll_milliseconds)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (type(maximum_campaign_seconds) not in {int, float}
                or not math.isfinite(maximum_campaign_seconds) or maximum_campaign_seconds < 0):
            raise ValueError("maximum_campaign_seconds must be finite and nonnegative")
        allowed = {"maximum_proof_nodes", "maximum_proof_depth", "maximum_hidden_worlds",
                   "policy_search", "maximum_line_count_limbs", "maximum_conflicts",
                   "maximum_prefix_lift_attempts", "maximum_seconds"}
        if set(proof_options) - allowed:
            raise ValueError("unsupported proposal proof options")
        if symmetry_policy_id != "rules_preserving_v1":
            raise ValueError("Research campaign symmetry policy is unsupported")
        if type(minimum_decision_state_distance) is not int or minimum_decision_state_distance < 1:
            raise ValueError("minimum decision-state distance must be positive")
        candidate_workers = min(proof_worker_count, maximum_candidates)
        # Divide the existing grant; candidate and world pools must not multiply it.
        terminal_workers = min(4, proof_worker_count // candidate_workers)
        started, cpu_started = time.perf_counter(), time.process_time()
        deadline = started + maximum_campaign_seconds
        stopped = threading.Event()
        def cancelled():
            return stopped.is_set() or time.perf_counter() >= deadline or bool(cancel_check and cancel_check())
        common = dict(bundle_revision_id=self.bundle_revision_id, bundle_hash=self.bundle_hash,
            backend_id=self.backend_id, adjudication_authority=self.adjudication_authority_document,
            tactical_detector_profile=self.tactical_detector_profile,
            rule_source_catalog=({"data": self.rule_source_catalog.data, "sha256": self.rule_source_catalog.sha256}
                                 if self.rule_source_catalog is not None else None),
            witness_policies=self.witness_policies, quality_policy=self.quality_policy,
            quality_execution=self.quality_execution, quality_proof_profiles=self.quality_proof_profiles,
            quality_search_profiles=self.quality_search_profiles)
        if self.generation is not None:
            common["generation"] = self.generation
        worker_samples, worker_progress, worker_seconds, proposal_shards = {}, {}, {}, {}
        bid_cache = getattr(self, "bid_quality_cache", None)
        def emit(ordinal, payload):
            value = {**payload, "proposalOrdinal": ordinal,
                "shardId": proposal_shards[ordinal].shard_id,
                "shardIndex": ordinal, "attempt": 1}
            if progress_callback:
                try:
                    progress_callback(value)
                except Exception:  # Telemetry cannot change proof or cancellation.
                    pass
            return value
        def prove(ordinal, proposal):
            worker_started = time.perf_counter()
            remaining = deadline - time.perf_counter()
            progress = CampaignAttemptProgressV1()
            native_started = False
            def report(payload):
                value = emit(ordinal, {**payload, "nativeAttemptStarted": native_started})
                progress.update(value)
            def disposition(status, reason=None, **details):
                # Checkpoints and private hands stay in the owned receipt, not live progress.
                snapshot = progress.snapshot()
                report({**{key: value for key, value in snapshot.items()
                            if key not in {"qualityCheckpoints", "workerCpuAccounting"}},
                    "proposalDisposition": status, "dispositionReason": reason, **details})
            def checkpoint(payload):
                progress.checkpoint(payload)
                document = json.loads(payload) if isinstance(payload, str) else payload
                if bid_cache is not None and document.get("kind") == "bid_quality_cache_entry_v1":
                    bid_cache.merge((document["entry"],))
            request = SimpleNamespace(cancellation_requested=cancelled,
                report_progress=report, report_checkpoint=checkpoint,
                report_worker_cpu_sample=progress.sample_worker_cpu,
                evidence_transfer_lock=threading.RLock(),
                progress_snapshot=progress.snapshot)
            task = {"rules": self.rules, "bundleRevisionId": self.bundle_revision_id,
                "bundleHash": self.bundle_hash, "backendId": self.backend_id,
                "tacticalDetectorProfile": self.tactical_detector_profile,
                "adjudicationAuthority": common["adjudication_authority"],
                "ruleSourceCatalog": common["rule_source_catalog"],
                "witnessPolicies": self.witness_policies, "qualityPolicy": self.quality_policy,
                "qualityExecution": self.quality_execution, "qualityProofProfiles": self.quality_proof_profiles,
                "qualitySearchProfiles": self.quality_search_profiles,
                **({"generation": self.generation} if self.generation is not None else {}),
                "runtimeProposals": {ordinal: proposal}, "scenario": scenario,
                "bidQualityCache": bid_cache.snapshot() if bid_cache is not None else (),
                "proofRequest": proof_request, "options": {**proof_options,
                    "result_count": 1, "maximum_candidates": 1, "seed": ordinal,
                    "symmetry_policy_id": symmetry_policy_id,
                    "allow_symmetry_only_variants": allow_symmetry_only_variants,
                    "minimum_decision_state_distance": minimum_decision_state_distance,
                    "minimum_principal_variation_action_distance": minimum_principal_variation_action_distance,
                    "maximum_campaign_seconds": remaining, "terminal_worker_count": terminal_workers}}
            try:
                if cancelled() or remaining <= 0:
                    disposition("rejected_before_dispatch", "campaign_budget_exhausted"
                        if time.perf_counter() >= deadline else "cancelled")
                    return None
                native_started = True
                report({"currentStage": "queued"})
                campaign = _run_seed_campaign_process_v1(task, request,
                    cancellation_poll_milliseconds=cancellation_poll_milliseconds)
                reasons = tuple(dict.fromkeys(item.execution_receipt.get("candidateAttemptReason") or item.status
                    for item in campaign.executions if item.status != "qualified"))
                snapshot = progress.snapshot()
                disposition("incomplete" if campaign.status == "incomplete" else "executed",
                    reasons[0] if reasons else snapshot.get("candidateAttemptReason") or (
                    "shard_budget_exhausted" if snapshot["currentStage"] == "shard_budget_exhausted" else
                    campaign.status if campaign.status != "completed" else None),
                    campaignStatus=campaign.status, completedExecutions=len(campaign.executions),
                    rejectionReasons=list(reasons))
                return campaign
            except RuntimeError as error:
                if not cancelled() or str(error) != "campaign shard process cancelled":
                    disposition("failed", "native_attempt_error")
                    raise
                disposition("incomplete", "campaign_budget_exhausted"
                    if time.perf_counter() >= deadline else "cancelled")
                return None
            except Exception:
                disposition("failed", "native_attempt_error")
                raise
            finally:
                worker_progress[ordinal] = progress.snapshot()
                worker_seconds[ordinal] = time.perf_counter() - worker_started
                worker_samples[ordinal] = worker_progress[ordinal].get("workerCpuAccounting", {}).get("cpuTimeNs", 0) / 1e9
        iterator = iter(proposals)
        completed, pending = {}, {}
        submitted, exhausted = 0, False
        pool = ThreadPoolExecutor(max_workers=candidate_workers)
        try:
            while not cancelled() and (pending or not exhausted):
                while not exhausted and len(pending) < candidate_workers and not cancelled():
                    if submitted >= maximum_candidates:
                        exhausted = True
                        break
                    try:
                        ordinal, proposal = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    if type(ordinal) is not int or ordinal < 0 or ordinal in pending.values() or ordinal in completed:
                        raise ValueError("proposal ordinals must be unique nonnegative integers")
                    proposal_shards[ordinal] = ResearchShardSpecV1.build(
                        run_code="proposal_stream", run_hash=canonical_sha256({
                            "bundleHash": self.bundle_hash, "scenarioHash": scenario.sha256,
                            "proposal": proposal}), recipe_hash=scenario.sha256,
                        shard_index=ordinal, seed_start=ordinal, seed_count=1, maximum_attempts=1)
                    if cancelled():
                        worker_progress[ordinal] = emit(ordinal, {"currentStage": "proposal_validation",
                            "nativeAttemptStarted": False, "proposalDisposition": "rejected_before_dispatch",
                            "dispositionReason": "campaign_budget_exhausted"
                                if time.perf_counter() >= deadline else "cancelled"})
                        completed[ordinal] = None
                        break
                    try:
                        validated = type(self)(self.rules, runtime_proposals={ordinal: proposal}, **common)
                    except Exception:
                        emit(ordinal, {"currentStage": "proposal_validation", "nativeAttemptStarted": False,
                            "proposalDisposition": "rejected_before_dispatch", "dispositionReason": "invalid_proposal"})
                        raise
                    pending[pool.submit(prove, ordinal, validated.runtime_proposals[ordinal])] = ordinal
                    submitted += 1
                if pending:
                    ready, _ = wait(pending, timeout=cancellation_poll_milliseconds / 1000,
                                    return_when=FIRST_COMPLETED)
                    for future in ready:
                        completed[pending.pop(future)] = future.result()
        finally:
            stopped.set()
            try:
                pool.shutdown(wait=True, cancel_futures=True)
                for future, ordinal in pending.items():
                    if not future.cancelled():
                        completed[ordinal] = future.result()
                    else:
                        completed[ordinal] = None
                        worker_progress[ordinal] = emit(ordinal, {"currentStage": "queued",
                            "nativeAttemptStarted": False, "proposalDisposition": "rejected_before_dispatch",
                            "dispositionReason": "cancelled"})
            finally:
                close = getattr(iterator, "close", None)
                if close:
                    close()
        campaigns = tuple((ordinal, campaign) for ordinal, campaign in sorted(completed.items()) if campaign is not None)
        distance = decision_state_card_hamming_distance_v1
        minimum_distance = minimum_decision_state_distance
        distance_id = "decision_state_card_hamming_v1"
        if minimum_principal_variation_action_distance is not None:
            distance = partial(semantic_threshold_distance_v1,
                minimum_decision_state_distance=minimum_decision_state_distance,
                minimum_principal_variation_distance=minimum_principal_variation_action_distance)
            minimum_distance, distance_id = 2, SEMANTIC_DISTANCE_POLICY_V1
        occurrences = tuple((ordinal, 1, _campaign_variant_from_execution_v1(execution))
            for ordinal, campaign in campaigns for execution in campaign.accepted_results)
        accepted, receipts = select_campaign_variants_v1(occurrences, result_count=result_count,
            symmetry_key=None if allow_symmetry_only_variants else rules_preserving_symmetry_key_v1,
            distance=distance, minimum_distance=minimum_distance)
        by_hash = {_campaign_variant_from_execution_v1(execution).semantic_hash: execution
                   for _, campaign in reversed(campaigns) for execution in campaign.accepted_results}
        incomplete = (not exhausted or len(campaigns) != submitted
                      or any(campaign.status == "incomplete" for _, campaign in campaigns))
        status = "completed" if len(accepted) == result_count else "incomplete" if incomplete else "not_found_within_budget"
        wall_seconds = time.perf_counter() - started
        cpu_seconds = time.process_time() - cpu_started + sum(
            max(worker_samples.get(ordinal, 0), campaign.cpu_seconds if campaign is not None else 0)
            for ordinal, campaign in completed.items())
        coordinator_receipt = None
        if proposal_shards:
            attempts = tuple(CampaignAttemptReceiptV1(
                ordinal, proposal_shards[ordinal].shard_id, 1,
                campaign.status if campaign is not None else "incomplete",
                tuple(_campaign_variant_from_execution_v1(item).semantic_hash
                      for item in campaign.accepted_results) if campaign is not None else (),
                (CampaignStageTimingV1("seed_campaigns",
                    int(worker_seconds.get(ordinal, 0) * 1e9),
                    int(max(worker_samples.get(ordinal, 0), campaign.cpu_seconds if campaign is not None else 0) * 1e9)),),
                partial_evidence=worker_progress.get(ordinal))
                for ordinal, campaign in sorted(completed.items()))
            coordinator_receipt = CampaignReceiptV1(
                status, tuple(proposal_shards[key] for key in sorted(proposal_shards)),
                CampaignBudgetV1(result_count, proof_worker_count, maximum_campaign_seconds,
                    cancellation_poll_milliseconds, 1),
                "semantic_hash_v1", None if allow_symmetry_only_variants else symmetry_policy_id,
                distance_id, minimum_distance, attempts, receipts, accepted, None,
                int(wall_seconds * 1e9), int(cpu_seconds * 1e9), "",
                cpu_time_coverage="sampled_lower_bound")
            semantic_core = coordinator_receipt._semantic_core_document()
            coordinator_receipt = replace(coordinator_receipt,
                semantic_sha256=canonical_sha256(semantic_core) if semantic_core else None)
            coordinator_receipt = replace(coordinator_receipt,
                receipt_sha256=canonical_sha256(coordinator_receipt._receipt_core()))
        return OrToolsResearchCampaignResultV1(
            status,
            tuple(execution for _, campaign in campaigns for execution in campaign.executions),
            tuple(by_hash[item.semantic_hash] for item in accepted), receipts,
            tuple(dict.fromkeys(vector for _, campaign in campaigns for vector in campaign.no_good_vectors)),
            result_count, maximum_candidates, symmetry_policy_id, distance_id, minimum_distance,
            wall_seconds, cpu_seconds, coordinator_receipt)

    def execute_campaign(
        self,
        scenario: ResearchScenarioV1,
        proof_request: ScenarioProofRequestV1,
        *,
        result_count: int,
        symmetry_policy_id: str,
        allow_symmetry_only_variants: bool,
        minimum_decision_state_distance: int,
        minimum_principal_variation_action_distance: int | None = None,
        candidate_compiler: Callable[..., OrToolsCandidateAttemptV1] | None = None,
        context_compiler: StageContextCompiler | None = None,
        graph_compiler: GraphCompiler | None = None,
        maximum_proof_nodes: int | None = None,
        maximum_proof_depth: int | None = None,
        maximum_hidden_worlds: int | None = None,
        policy_search: str | None = None,
        maximum_line_count_limbs: int = 64,
        maximum_conflicts: int = 250_000,
        maximum_prefix_lift_attempts: int | None = None,
        seed: int = 0,
        maximum_candidates: int = 1,
        maximum_seconds: float = 15.0,
        maximum_campaign_seconds: float | None = None,
        cancel_check: Callable[[], bool] | None = None,
        shards: Sequence[ResearchShardSpecV1] | None = None,
        shard_worker_count: int = 1,
        cancellation_poll_milliseconds: int = 25,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> OrToolsResearchCampaignResultV1:
        """Run fixed seed shards and merge their semantic results canonically."""

        self._validate_runtime_proposal_scope(scenario)
        _reject_v3_semantic_callback_injection(
            self.backend_id,
            candidate_compiler=candidate_compiler,
            context_compiler=context_compiler,
            graph_compiler=graph_compiler,
        )
        if minimum_principal_variation_action_distance is not None and (
            type(minimum_principal_variation_action_distance) is not int
            or minimum_principal_variation_action_distance < 1
        ):
            raise ValueError(
                "campaign minimum principal-variation distance must be positive"
            )
        if self.runtime_proposals is not None and (
            maximum_campaign_seconds == 0 or not self.runtime_proposals
        ):
            # Screening can consume the budget; never start a nominal solver retry.
            return OrToolsResearchCampaignResultV1(
                "incomplete" if maximum_campaign_seconds == 0 else "not_found_within_budget",
                (), (), (), (), result_count, maximum_candidates, symmetry_policy_id,
                SEMANTIC_DISTANCE_POLICY_V1 if minimum_principal_variation_action_distance is not None
                else "decision_state_card_hamming_v1",
                2 if minimum_principal_variation_action_distance is not None
                else minimum_decision_state_distance, 0.0, 0.0,
            )
        if shards is None:
            return self._execute_seed_campaign(
                scenario,
                proof_request,
                result_count=result_count,
                symmetry_policy_id=symmetry_policy_id,
                allow_symmetry_only_variants=allow_symmetry_only_variants,
                minimum_decision_state_distance=minimum_decision_state_distance,
                minimum_principal_variation_action_distance=(
                    minimum_principal_variation_action_distance
                ),
                candidate_compiler=candidate_compiler,
                context_compiler=context_compiler,
                graph_compiler=graph_compiler,
                maximum_proof_nodes=maximum_proof_nodes,
                maximum_proof_depth=maximum_proof_depth,
                maximum_hidden_worlds=maximum_hidden_worlds,
                policy_search=policy_search,
                maximum_line_count_limbs=maximum_line_count_limbs,
                maximum_conflicts=maximum_conflicts,
                maximum_prefix_lift_attempts=maximum_prefix_lift_attempts,
                seed=seed,
                maximum_candidates=maximum_candidates,
                maximum_seconds=maximum_seconds,
                maximum_campaign_seconds=maximum_campaign_seconds,
                cancel_check=cancel_check,
                progress_callback=progress_callback,
                terminal_worker_count=shard_worker_count,
            )

        fixed_shards = tuple(shards)
        if not fixed_shards:
            raise ValueError("sharded campaign requires at least one shard")
        if scenario.origin.kind == "exact_trace_prefix" and (
            len(fixed_shards) != 1 or fixed_shards[0].seed_count != 1
        ):
            raise ValueError("exact trace-prefix execution requires one one-seed shard")
        candidate_universe = sum(item.seed_count for item in fixed_shards)
        partition_origin = min(item.seed_start for item in fixed_shards)
        ordered_ranges = sorted(
            (item.seed_start, item.seed_end_exclusive) for item in fixed_shards
        )
        if ordered_ranges[0][0] != partition_origin or any(
            left[1] != right[0]
            for left, right in zip(ordered_ranges, ordered_ranges[1:], strict=False)
        ):
            raise ValueError("campaign shard seed ranges must form one contiguous partition")
        if ordered_ranges[-1][1] - partition_origin != candidate_universe:
            raise ValueError("campaign shard partition does not cover its candidate universe")
        if maximum_candidates != candidate_universe:
            raise ValueError(
                "campaign candidate budget must equal its fixed seed universe"
            )
        if result_count > candidate_universe:
            raise ValueError("campaign seed universe cannot cover requested results")
        campaign_seconds = (
            maximum_seconds
            if maximum_campaign_seconds is None
            else maximum_campaign_seconds
        )
        if (
            type(campaign_seconds) not in {int, float}
            or not math.isfinite(campaign_seconds)
            or campaign_seconds <= 0
        ):
            raise ValueError("maximum_campaign_seconds must be finite and positive")
        ordered_partition_shards = tuple(
            sorted(fixed_shards, key=lambda item: item.shard_index)
        )
        partition_ordinal_by_shard_id = {
            item.shard_id: ordinal
            for ordinal, item in enumerate(ordered_partition_shards)
        }

        started = time.perf_counter()
        cpu_started = time.process_time()
        lock = threading.Lock()
        effective_shard_worker_count = min(shard_worker_count, len(fixed_shards))
        # Untried candidates do not reserve time from active proof work.
        maximum_shard_seconds = (
            campaign_seconds
            if self.backend_id == _V3_BACKEND_ID
            else min(maximum_seconds, campaign_seconds)
        )
        process_shards = bool(
            self.backend_id == "ortools_cp_sat_v3"
            and candidate_compiler is None
            and context_compiler is None
            and graph_compiler is None
        )
        shard_campaigns: dict[
            tuple[int, int], tuple[tuple[int, OrToolsResearchCampaignResultV1], ...]
        ] = {}

        def solve_shard(request) -> CampaignShardOutcomeV1:
            shard_started = time.perf_counter()
            shard_cpu_started = time.thread_time()
            with lock:
                prior_campaigns = tuple(
                    (attempt, campaign)
                    for (shard_index, attempt), campaigns in shard_campaigns.items()
                    if shard_index == request.shard.shard_index
                    for _seed_value, campaign in campaigns
                )
            remaining_shard_seconds = maximum_shard_seconds - (
                sum(campaign.wall_time_seconds for _attempt, campaign in prior_campaigns)
                if self.backend_id == _V3_BACKEND_ID else 0.0
            )
            if self.backend_id == _V3_BACKEND_ID and remaining_shard_seconds <= 0:
                return CampaignShardOutcomeV1(
                    (), partial_evidence={
                        **request.progress_snapshot(),
                        "shardBudgetExhausted": True,
                        "completedCandidates": 0,
                    },
                )
            exhausted_proof_no_goods = tuple(
                execution.candidate.no_good_vector
                for attempt, campaign in prior_campaigns
                if attempt > 1
                for execution in campaign.executions
                if execution.status == "incomplete"
                and execution.candidate is not None
                and execution.proof is not None
                and execution.proof.get("completionStatus") == "proof_incomplete"
                and execution.candidate.no_good_vector
            )
            prior_no_goods = tuple(
                dict.fromkeys(
                    no_good
                    for no_good in (
                        *exhausted_proof_no_goods,
                        *(
                            no_good
                            for _attempt, campaign in prior_campaigns
                            for no_good in campaign.no_good_vectors
                        ),
                    )
                )
            )
            prior_quality_rejections = tuple({
                execution.candidate.sha256: execution.candidate
                for _attempt, campaign in prior_campaigns
                for execution in campaign.executions
                if self.backend_id == _V3_BACKEND_ID
                and execution.candidate is not None
                and execution.status == "semantic_postcondition_miss"
                and execution.execution_receipt.get("candidateAttemptReason") in {
                    "quality_generation_target_miss",
                    "quality_recorded_audit_target_miss",
                    "quality_known_nonoptimal_prefix",
                }
                and isinstance(execution.candidate.trace.get("tailFirstStagePlan"), Mapping)
                and Contract.parse(execution.candidate.trace["contract"]).effective is not Contract.HOKUM
            }.values())
            incomplete_search_attempts = sum(
                (
                    sum(
                        execution.status == "incomplete" and execution.candidate is None
                        for execution in campaign.executions
                    )
                    if campaign.executions else int(campaign.status == "incomplete")
                )
                if self.backend_id == _V3_BACKEND_ID else int(
                    campaign.status == "incomplete"
                    and not any(execution.candidate is not None for execution in campaign.executions)
                )
                for _attempt, campaign in prior_campaigns
            )
            remaining_candidate_budget = (
                request.shard.seed_count
                - len(set(prior_no_goods))
                - len(prior_quality_rejections)
                - incomplete_search_attempts
            )
            consumed_proposals = sum(len(campaign.executions) for _attempt, campaign in prior_campaigns)
            if self.runtime_proposals is not None:
                remaining_candidate_budget = request.shard.seed_count - consumed_proposals
            if remaining_candidate_budget <= 0:
                return CampaignShardOutcomeV1(
                    (),
                    partial_evidence={
                        **request.progress_snapshot(),
                        "candidateBudgetExhausted": True,
                        "completedCandidates": 0,
                    },
                )
            replenishment_decision_vectors = tuple(
                vector
                for variant in request.accepted_variants
                for vector in _campaign_variant_decision_cuts_v1(variant)
            )
            effective_proof_nodes = maximum_proof_nodes
            if (
                maximum_proof_nodes is not None
                and self.backend_id != _V3_BACKEND_ID
                and request.shard.maximum_attempts > 1
                and request.attempt == 1
            ):
                effective_proof_nodes = max(1, maximum_proof_nodes // 4)
            campaign_options = dict(
                result_count=(
                    1
                    if len(fixed_shards) > 1
                    else min(result_count, remaining_candidate_budget)
                ),
                symmetry_policy_id=symmetry_policy_id,
                allow_symmetry_only_variants=allow_symmetry_only_variants,
                minimum_decision_state_distance=minimum_decision_state_distance,
                minimum_principal_variation_action_distance=(
                    minimum_principal_variation_action_distance
                ),
                candidate_compiler=candidate_compiler,
                context_compiler=context_compiler,
                graph_compiler=graph_compiler,
                maximum_proof_nodes=effective_proof_nodes,
                maximum_proof_depth=maximum_proof_depth,
                maximum_hidden_worlds=maximum_hidden_worlds,
                policy_search=policy_search,
                maximum_line_count_limbs=maximum_line_count_limbs,
                maximum_conflicts=maximum_conflicts,
                maximum_prefix_lift_attempts=maximum_prefix_lift_attempts,
                seed=(
                    request.shard.seed_start
                    + (consumed_proposals if self.runtime_proposals is not None else
                       len(prior_no_goods) + len(prior_quality_rejections) + incomplete_search_attempts)
                ),
                maximum_candidates=remaining_candidate_budget,
                maximum_seconds=maximum_seconds,
                **(
                    {"maximum_campaign_seconds": max(
                        0.001,
                        min(
                            remaining_shard_seconds,
                            campaign_seconds - (time.perf_counter() - started),
                        ),
                    )}
                    if self.backend_id == _V3_BACKEND_ID
                    else {}
                ),
                candidate_partition=(
                    None
                    if result_count == 1 and request.shard.shard_index == 0
                    else (
                        len(fixed_shards),
                        partition_ordinal_by_shard_id[request.shard.shard_id],
                        1,
                    )
                ),
                initial_excluded_decks=prior_no_goods,
                initial_quality_rejected_candidates=prior_quality_rejections,
                initial_excluded_decision_status_vectors=(
                    replenishment_decision_vectors
                ),
                terminal_worker_count=(
                    effective_shard_worker_count if len(fixed_shards) == 1 else 1
                ),
            )
            if process_shards:
                task = {
                    "rules": self.rules,
                    "bundleRevisionId": self.bundle_revision_id,
                    "bundleHash": self.bundle_hash,
                    "backendId": self.backend_id,
                    "tacticalDetectorProfile": self.tactical_detector_profile,
                    "adjudicationAuthority": self.adjudication_authority_document,
                    "ruleSourceCatalog": (
                        {"data": self.rule_source_catalog.data, "sha256": self.rule_source_catalog.sha256}
                        if self.rule_source_catalog is not None else None
                    ),
                    "witnessPolicies": self.witness_policies,
                    "qualityPolicy": self.quality_policy,
                    "qualityExecution": self.quality_execution,
                    "qualityProofProfiles": self.quality_proof_profiles,
                    "qualitySearchProfiles": self.quality_search_profiles,
                    "runtimeProposals": self.runtime_proposals,
                    **({"generation": self.generation} if self.generation is not None else {}),
                    "scenario": scenario,
                    "proofRequest": proof_request,
                    "options": campaign_options,
                }
                for transport_attempt in (1, 2):
                    remaining_transport_seconds = min(
                        remaining_shard_seconds - (time.perf_counter() - shard_started),
                        campaign_seconds - (time.perf_counter() - started),
                    )
                    if remaining_transport_seconds <= 0:
                        request.report_progress({"currentStage": "shard_budget_exhausted"})
                        return CampaignShardOutcomeV1(
                            (), partial_evidence={
                                **request.progress_snapshot(),
                                "shardBudgetExhausted": True,
                            },
                        )
                    task["options"]["maximum_campaign_seconds"] = max(
                        0.001, remaining_transport_seconds
                    )
                    try:
                        campaign = _run_seed_campaign_process_v1(
                            task,
                            request,
                            cancellation_poll_milliseconds=(
                                cancellation_poll_milliseconds
                            ),
                        )
                        break
                    except (BrokenPipeError, EOFError):
                        if transport_attempt == 2 or request.cancellation_requested():
                            raise
                        request.report_progress(
                            {
                                "currentStage": "transport_retry_scheduled",
                                "transportAttempt": 2,
                            }
                        )
            else:
                campaign = self._execute_seed_campaign(
                    scenario,
                    proof_request,
                    cancel_check=lambda: request.cancellation_requested()
                    or (cancel_check is not None and cancel_check()),
                    progress_callback=request.report_progress,
                    checkpoint_callback=request.report_checkpoint,
                    **campaign_options,
                )
            local = ((campaign_options["seed"], campaign),)
            variants = tuple(
                _campaign_variant_from_execution_v1(execution)
                for execution in campaign.accepted_results
            )
            incomplete_executions = tuple(
                execution
                for execution in campaign.executions
                if execution.status == "incomplete"
            )
            partial_timings = tuple(
                CampaignStageTimingV1(
                    stage,
                    int(max(0.0, float(seconds)) * 1e9),
                )
                for execution in incomplete_executions
                for stage, seconds in dict(
                    execution.execution_receipt.get("stageTimings") or {}
                ).items()
            )
            partial_evidence = None
            terminal_execution = campaign.executions[-1] if campaign.executions else None
            progress_snapshot = request.progress_snapshot()
            if (
                self.backend_id == _V3_BACKEND_ID
                and campaign.executions
                and all(
                    execution.candidate is not None
                    and isinstance(
                        execution.candidate.trace.get("playingQualityEvidence"),
                        Mapping,
                    )
                    for execution in campaign.executions
                )
            ):
                # Full decision evidence is retained in each execution artifact.
                checkpoints = progress_snapshot.pop("qualityCheckpoints", None)
                if checkpoints:
                    progress_snapshot["qualityCheckpointHashes"] = sorted(checkpoints)
            proof_incomplete = bool(
                terminal_execution is not None
                and terminal_execution.status == "incomplete"
                and terminal_execution.proof is not None
                and terminal_execution.proof.get("completionStatus")
                == "proof_incomplete"
            )
            remaining_after_campaign = (
                remaining_candidate_budget - len(campaign.executions)
            )
            shard_budget_exhausted = bool(
                self.backend_id == _V3_BACKEND_ID
                and time.perf_counter() - shard_started >= remaining_shard_seconds
            )
            if incomplete_executions or campaign.status == "incomplete":
                partial_evidence = {
                    **progress_snapshot,
                    "campaignStatus": campaign.status,
                    "shardBudgetExhausted": shard_budget_exhausted,
                    "proofNodeLimit": effective_proof_nodes,
                    "proofNodeCeiling": maximum_proof_nodes,
                    "proofRetryEligible": bool(
                        not shard_budget_exhausted
                        and proof_incomplete
                        and request.attempt < request.shard.maximum_attempts
                        and maximum_proof_nodes is not None
                        and effective_proof_nodes is not None
                        and effective_proof_nodes < maximum_proof_nodes
                    ),
                    "attemptRetryEligible": bool(
                        not shard_budget_exhausted
                        and request.attempt < request.shard.maximum_attempts
                        and remaining_after_campaign > 0
                        and (
                            not proof_incomplete
                            or effective_proof_nodes == maximum_proof_nodes
                        )
                    ),
                    "completedCandidates": max(
                        sum(int(execution.execution_receipt.get("completedCandidates", 0))
                            for execution in campaign.executions),
                        int(progress_snapshot.get("completedCandidates", 0)),
                    ),
                    "executions": [
                        {
                            "status": execution.status,
                            "stageTimings": dict(
                                execution.execution_receipt.get("stageTimings") or {}
                            ),
                            "incompleteCompilation": (
                                _campaign_incomplete_compilation_summary_v1(
                                    execution.execution_receipt.get(
                                        "incompleteCompilation"
                                    )
                                )
                            ),
                            "candidateAttemptReason": execution.execution_receipt.get(
                                "candidateAttemptReason"
                            ),
                        }
                        for execution in incomplete_executions
                    ],
                }
            exhausted_candidate_budget = remaining_after_campaign <= 0
            if exhausted_candidate_budget:
                partial_evidence = {
                    **dict(partial_evidence or progress_snapshot),
                    "candidateBudgetExhausted": True,
                }
            with lock:
                if not request.cancellation_requested():
                    shard_campaigns[(request.shard.shard_index, request.attempt)] = (
                        local
                    )
            return CampaignShardOutcomeV1(
                variants,
                (
                    CampaignStageTimingV1(
                        "seed_campaigns",
                        int(max(0.0, time.perf_counter() - shard_started) * 1e9),
                        int(
                            max(
                                0.0,
                                campaign.cpu_seconds
                                if process_shards
                                else time.thread_time() - shard_cpu_started,
                            )
                            * 1e9
                        ),
                    ),
                    *partial_timings,
                ),
                partial_evidence,
            )

        distance = decision_state_card_hamming_distance_v1
        distance_policy_id = "decision_state_card_hamming_v1"
        minimum_distance = minimum_decision_state_distance
        if minimum_principal_variation_action_distance is not None:
            distance = partial(
                semantic_threshold_distance_v1,
                minimum_decision_state_distance=minimum_decision_state_distance,
                minimum_principal_variation_distance=(
                    minimum_principal_variation_action_distance
                ),
            )
            distance_policy_id = SEMANTIC_DISTANCE_POLICY_V1
            minimum_distance = 2
        coordinator = DeterministicCpSatCampaignCoordinatorV1(
            symmetry_key=rules_preserving_symmetry_key_v1,
            symmetry_policy_id=symmetry_policy_id,
            distance=distance,
            minimum_distance=minimum_distance,
            distance_policy_id=distance_policy_id,
        )
        receipt = coordinator.run(
            shards=fixed_shards,
            solver=solve_shard,
            budget=CampaignBudgetV1(
                result_count,
                effective_shard_worker_count,
                campaign_seconds,
                cancellation_poll_milliseconds,
                1,
            ),
            replenish_diversity_rejections=True,
            cancellation_check=cancel_check,
            progress_callback=progress_callback,
        )
        if process_shards:
            child_cpu_ns = 0
            cpu_complete = True
            for attempt in receipt.attempts:
                terminal_cpu = sum(timing.cpu_time_ns for timing in attempt.stage_timings
                                   if timing.stage == "seed_campaigns")
                if terminal_cpu:
                    child_cpu_ns += terminal_cpu
                else:
                    # Killed workers cannot return terminal CPU totals. Preserve the
                    # latest received sample, never present it as a complete total.
                    sample = (attempt.partial_evidence or {}).get("workerCpuAccounting", {})
                    child_cpu_ns += sample.get("cpuTimeNs", 0)
                    cpu_complete = False
            receipt = replace(
                receipt,
                cpu_time_ns=receipt.cpu_time_ns + child_cpu_ns,
                cpu_time_coverage="terminal_measurements" if cpu_complete else "sampled_lower_bound",
                receipt_sha256="",
            )
            receipt = replace(
                receipt,
                receipt_sha256=canonical_sha256(receipt._receipt_core()),
            )
        completed_attempts = {
            (item.shard_index, item.attempt)
            for item in receipt.attempts
            if item.status == "completed"
        }
        ordered_campaigns = tuple(
            (shard_index, attempt, seed_value, campaign)
            for (shard_index, attempt), campaigns in sorted(shard_campaigns.items())
            if (shard_index, attempt) in completed_attempts
            for seed_value, campaign in campaigns
        )
        executions = tuple(
            execution
            for _shard, _attempt, _seed, campaign in ordered_campaigns
            for execution in campaign.executions
        )
        execution_by_variant: dict[str, OrToolsUnifiedResearchResultV1] = {}
        for _shard, _attempt, _seed, campaign in ordered_campaigns:
            for execution in campaign.accepted_results:
                execution_by_variant.setdefault(
                    _campaign_variant_from_execution_v1(execution).semantic_hash,
                    execution,
                )
        accepted_results = tuple(
            execution_by_variant[item.semantic_hash]
            for item in receipt.accepted_variants
        )
        no_goods = tuple(
            dict.fromkeys(
                vector
                for _shard, _attempt, _seed, campaign in ordered_campaigns
                for vector in campaign.no_good_vectors
            )
        )
        inner_statuses = tuple(
            campaign.status for _shard, _attempt, _seed, campaign in ordered_campaigns
        )
        if len(accepted_results) == result_count:
            status = "completed"
        elif receipt.status in {"cancelled", "budget_exhausted", "incomplete"} or any(
            status == "incomplete" for status in inner_statuses
        ):
            status = "incomplete"
        elif inner_statuses and all(
            item == "structurally_infeasible" for item in inner_statuses
        ):
            status = "structurally_infeasible"
        elif not accepted_results and "proof_failed" in inner_statuses:
            status = "proof_failed"
        elif not accepted_results and "semantic_postcondition_miss" in inner_statuses:
            status = "semantic_postcondition_miss"
        else:
            status = "not_found_within_budget"
        return OrToolsResearchCampaignResultV1(
            status,
            executions,
            accepted_results,
            receipt.variants,
            no_goods,
            result_count,
            candidate_universe,
            symmetry_policy_id,
            distance_policy_id,
            minimum_distance,
            max(0.0, time.perf_counter() - started),
            (
                receipt.cpu_time_ns / 1e9
                if process_shards
                else max(0.0, time.process_time() - cpu_started)
            ),
            receipt,
        )

    def _execute_seed_campaign(
        self,
        scenario: ResearchScenarioV1,
        proof_request: ScenarioProofRequestV1,
        *,
        result_count: int,
        symmetry_policy_id: str,
        allow_symmetry_only_variants: bool,
        minimum_decision_state_distance: int,
        minimum_principal_variation_action_distance: int | None = None,
        candidate_compiler: Callable[..., OrToolsCandidateAttemptV1] | None = None,
        context_compiler: StageContextCompiler | None = None,
        graph_compiler: GraphCompiler | None = None,
        maximum_proof_nodes: int | None = None,
        maximum_proof_depth: int | None = None,
        maximum_hidden_worlds: int | None = None,
        policy_search: str | None = None,
        maximum_line_count_limbs: int = 64,
        maximum_conflicts: int = 250_000,
        maximum_prefix_lift_attempts: int | None = None,
        seed: int = 0,
        maximum_candidates: int = 1,
        maximum_seconds: float = 15.0,
        maximum_campaign_seconds: float | None = None,
        cancel_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
        candidate_partition: tuple[int, int, int] | None = None,
        checkpoint_callback: Callable[[Mapping[str, Any]], None] | None = None,
        initial_excluded_decks: Sequence[tuple[int, ...]] = (),
        initial_quality_rejected_candidates: Sequence[OrToolsNeutralCandidateV1] = (),
        initial_excluded_decision_status_vectors: Sequence[tuple[int, ...]] = (),
        terminal_worker_count: int = 1,
    ) -> OrToolsResearchCampaignResultV1:
        """Collect distinct qualified results under one deterministic no-good budget."""

        if type(result_count) is not int or result_count < 1:
            raise ValueError("campaign result_count must be positive")
        if symmetry_policy_id != "rules_preserving_v1":
            raise ValueError("campaign symmetry policy is unsupported")
        if allow_symmetry_only_variants is not False:
            raise ValueError("campaign symmetry-only variants are forbidden")
        if (
            type(minimum_decision_state_distance) is not int
            or minimum_decision_state_distance < 1
        ):
            raise ValueError(
                "campaign minimum decision-state distance must be positive"
            )
        if minimum_principal_variation_action_distance is not None and (
            type(minimum_principal_variation_action_distance) is not int
            or minimum_principal_variation_action_distance < 1
        ):
            raise ValueError(
                "campaign minimum principal-variation distance must be positive"
            )
        distance = decision_state_card_hamming_distance_v1
        distance_policy_id = "decision_state_card_hamming_v1"
        minimum_distance = minimum_decision_state_distance
        if minimum_principal_variation_action_distance is not None:
            distance = partial(
                semantic_threshold_distance_v1,
                minimum_decision_state_distance=minimum_decision_state_distance,
                minimum_principal_variation_distance=(
                    minimum_principal_variation_action_distance
                ),
            )
            distance_policy_id = SEMANTIC_DISTANCE_POLICY_V1
            minimum_distance = 2
        if type(maximum_candidates) is not int or maximum_candidates < result_count:
            raise ValueError("campaign candidate budget must cover requested results")
        if scenario.origin.kind == "exact_trace_prefix" and result_count != 1:
            raise ValueError(
                "an exact trace-prefix campaign can return only one result"
            )
        if (
            type(maximum_seconds) not in {int, float}
            or not math.isfinite(maximum_seconds)
            or maximum_seconds <= 0
        ):
            raise ValueError("maximum_seconds must be finite and positive")

        campaign_seconds = (
            maximum_seconds if maximum_campaign_seconds is None
            else maximum_campaign_seconds
        )
        if (
            type(campaign_seconds) not in {int, float}
            or not math.isfinite(campaign_seconds)
            or campaign_seconds <= 0
        ):
            raise ValueError("maximum_campaign_seconds must be finite and positive")
        continue_incomplete = (
            self.backend_id == _V3_BACKEND_ID
            and scenario.origin.kind == "search"
            and maximum_campaign_seconds is not None
        )

        started = time.perf_counter()
        cpu_started = time.process_time()
        executions: list[OrToolsUnifiedResearchResultV1] = []
        variants: list[tuple[int, int, CampaignVariantV1]] = []
        execution_by_variant_hash: dict[str, OrToolsUnifiedResearchResultV1] = {}
        excluded = [tuple(int(card) for card in item) for item in initial_excluded_decks]
        quality_rejected = list(initial_quality_rejected_candidates)
        incomplete_decision_vectors: list[tuple[int, ...]] = []
        accepted_variants: tuple[CampaignVariantV1, ...] = ()
        variant_receipts: tuple[CampaignVariantReceiptV1, ...] = ()
        terminal_status = "not_found_within_budget"
        last_failure_status: str | None = None
        completed_candidates = 0
        attempted_candidates = 0

        while (
            attempted_candidates if continue_incomplete else completed_candidates
        ) < maximum_candidates:
            if cancel_check is not None and cancel_check():
                terminal_status = "incomplete"
                break
            remaining = campaign_seconds - (time.perf_counter() - started)
            if remaining <= 0:
                break
            execution = self.execute(
                scenario,
                proof_request,
                candidate_compiler=candidate_compiler,
                context_compiler=context_compiler,
                graph_compiler=graph_compiler,
                maximum_proof_nodes=maximum_proof_nodes,
                maximum_proof_depth=maximum_proof_depth,
                maximum_hidden_worlds=maximum_hidden_worlds,
                policy_search=policy_search,
                maximum_line_count_limbs=maximum_line_count_limbs,
                maximum_conflicts=maximum_conflicts,
                maximum_prefix_lift_attempts=maximum_prefix_lift_attempts,
                seed=seed + (
                    attempted_candidates if continue_incomplete else completed_candidates
                ),
                maximum_candidates=1,
                maximum_seconds=remaining,
                maximum_stage_seconds=min(maximum_seconds, remaining),
                initial_excluded_decks=excluded,
                initial_quality_rejected_candidates=tuple(quality_rejected),
                minimum_deck_hamming_distance=minimum_decision_state_distance,
                excluded_decision_status_vectors=(
                    tuple(initial_excluded_decision_status_vectors)
                    + tuple(incomplete_decision_vectors)
                    + tuple(
                        tuple(
                            int(status)
                            for status in variant.symmetry_document[
                                "decisionStateCardVector"
                            ]
                        )
                        for variant in accepted_variants
                    )
                    if scenario.decision_point["kind"] == "play"
                    else ()
                ),
                minimum_decision_status_hamming_distance=(
                    minimum_decision_state_distance
                ),
                candidate_partition=candidate_partition,
                cancel_check=cancel_check,
                progress_callback=progress_callback,
                checkpoint_callback=checkpoint_callback,
                terminal_worker_count=terminal_worker_count,
            )
            executions.append(execution)
            attempted_candidates += 1
            attempt_candidates = int(execution.execution_receipt["completedCandidates"])
            completed_candidates += attempt_candidates
            candidate = execution.candidate
            if progress_callback is not None:
                progress_callback(
                    _candidate_terminal_progress_v1(
                        execution,
                        ordinal=len(executions),
                        decision_kind=str(scenario.decision_point["kind"]),
                    )
                )
            if self.runtime_proposals is None and candidate is not None and (
                execution.status != "incomplete" or continue_incomplete
            ):
                quality_play_miss = (
                    self.backend_id == _V3_BACKEND_ID
                    and execution.status == "semantic_postcondition_miss"
                    and execution.execution_receipt.get("candidateAttemptReason") in {
                        "quality_generation_target_miss",
                        "quality_recorded_audit_target_miss",
                        "quality_known_nonoptimal_prefix",
                    }
                    and isinstance(candidate.trace.get("tailFirstStagePlan"), Mapping)
                    and Contract.parse(candidate.trace["contract"]).effective is not Contract.HOKUM
                )
                if quality_play_miss:
                    if any(previous.sha256 == candidate.sha256 for previous in quality_rejected):
                        terminal_status = "incomplete"
                        break
                    quality_rejected.append(candidate)
                no_good = () if quality_play_miss else candidate.no_good_vector
                if scenario.origin.kind == "search" and not no_good and not quality_play_miss:
                    terminal_status = "incomplete"
                    break
                if no_good:
                    if no_good in excluded:
                        terminal_status = "incomplete"
                        break
                    excluded.append(no_good)

            if execution.status == "qualified":
                variant = _campaign_variant_from_execution_v1(execution)
                variants.append((0, len(executions), variant))
                execution_by_variant_hash[variant.semantic_hash] = execution
                accepted_variants, variant_receipts = select_campaign_variants_v1(
                    tuple(variants),
                    result_count=result_count,
                    symmetry_key=rules_preserving_symmetry_key_v1,
                    distance=distance,
                    minimum_distance=minimum_distance,
                )
                if len(accepted_variants) == result_count:
                    terminal_status = "completed"
                    break
                continue
            if (execution.status == "proof_failed"
                    and execution.execution_receipt.get("counterstrategy", {}).get("status") == "certified"):
                terminal_status = "proof_failed"
                break
            if execution.status == "infeasible":
                terminal_status = (
                    "incomplete"
                    if continue_incomplete and any(
                        item.status == "incomplete" for item in executions[:-1]
                    ) else
                    "structurally_infeasible"
                    if completed_candidates == 0
                    and not (continue_incomplete and attempted_candidates > 1)
                    and not initial_excluded_decks
                    else "not_found_within_budget"
                )
                if completed_candidates:
                    last_failure_status = None
                break
            if execution.status == "incomplete":
                terminal_status = "incomplete"
                if continue_incomplete:
                    incomplete_receipt = execution.execution_receipt.get(
                        "incompleteCompilation"
                    ) or {}
                    vector = (incomplete_receipt.get("evidence") or {}).get(
                        "decisionStateStatusVector"
                    ) if incomplete_receipt.get("stage") == "proofBoundary" else None
                    if vector is not None:
                        if (
                            type(vector) is not list
                            or len(vector) != 32
                            or any(type(value) is not int or not 0 <= value <= 8
                                   for value in vector)
                        ):
                            raise ValueError("incomplete boundary has an invalid decision-state cut")
                        cut = tuple(vector)
                        if (
                            cut in incomplete_decision_vectors
                            or cut in initial_excluded_decision_status_vectors
                        ):
                            break
                        incomplete_decision_vectors.append(cut)
                    # An incomplete candidate spends a candidate slot, not a worker retry.
                    continue
                break
            if execution.status == "compiler_disagreement":
                terminal_status = "incomplete"
                break
            if execution.status == "not_found":
                last_failure_status = None
                if (attempt_candidates > 0 and execution.execution_receipt.get(
                        "candidateAttemptReason") == "curated_interest_miss"):
                    continue
                break
            if attempt_candidates == 0:
                terminal_status = "incomplete"
                break
            if execution.status in {"feasible", "semantic_postcondition_miss"}:
                last_failure_status = "semantic_postcondition_miss"
            elif execution.status == "proof_failed":
                last_failure_status = "proof_failed"

        accepted_results = tuple(
            execution_by_variant_hash[item.semantic_hash] for item in accepted_variants
        )
        if not accepted_results and terminal_status == "not_found_within_budget":
            terminal_status = last_failure_status or terminal_status
        return OrToolsResearchCampaignResultV1(
            terminal_status,
            tuple(executions),
            accepted_results,
            variant_receipts,
            tuple(excluded),
            result_count,
            maximum_candidates,
            symmetry_policy_id,
            distance_policy_id,
            minimum_distance,
            max(0.0, time.perf_counter() - started),
            max(0.0, time.process_time() - cpu_started),
        )

    def _result(
        self,
        status: str,
        preflight: OrToolsResearchPreflightV1,
        candidate: OrToolsNeutralCandidateV1 | None,
        evaluation: OrToolsPredicateEvaluationV1 | None,
        proof: Mapping[str, Any] | None,
        rejection_hashes: tuple[str, ...],
        started: float,
        cpu_started: float,
        completed_candidates: int,
        *,
        graph: Any | None = None,
        proof_outcome: OrToolsGraphProofOutcomeV1 | None = None,
        stage_timings: Mapping[str, float] | None = None,
        scenario_hash: str | None = None,
        proof_request_hash: str | None = None,
        incomplete_receipt: Mapping[str, Any] | None = None,
        structural_infeasibility_receipt: Mapping[str, Any] | None = None,
        counterstrategy_receipt: Mapping[str, Any] | None = None,
        candidate_attempt_reason: str | None = None,
        quality_handoff_diagnostic: Mapping[str, Any] | None = None,
        _quality_verified_artifacts: set[str] | None = None,
        _lesson_normalized: OrToolsNormalizedExactStateV1 | None = None,
        _family_opening: Mapping[str, Any] | None = None,
    ) -> OrToolsUnifiedResearchResultV1:
        authority = None
        artifact_preimages = None
        counterstrategy_core = None
        if counterstrategy_receipt is not None:
            counterstrategy_core = copy.deepcopy(dict(counterstrategy_receipt))
            counterstrategy_core.pop("executionReceipt", None)
            counterstrategy_hash = counterstrategy_core.pop("sha256", None)
            if (counterstrategy_hash != canonical_sha256(counterstrategy_core)
                    or counterstrategy_core.get("rulesHash") != self.rules.sha256
                    or counterstrategy_core.get("scenarioHash") != scenario_hash
                    or counterstrategy_core.get("proofRequestHash") != proof_request_hash
                    or self.backend_id != _V3_BACKEND_ID):
                raise ValueError("native counterstrategy identity mismatch")
            counterstrategy_core["sha256"] = counterstrategy_hash
            if counterstrategy_core["status"] == "certified" and (status != "proof_failed" or candidate is not None):
                raise ValueError("certified counterstrategy must fail the request without a candidate")
        if self.backend_id == _V3_BACKEND_ID and candidate is not None and candidate.exact_state_id is None:
            if status == "qualified" or candidate.trace.get("exactNormalization") is not None:
                raise RuntimeError("qualified or normalized candidate requires exact-state identity")
            candidate = replace(candidate, trace={**candidate.trace,
                "partialConstructionState": copy.deepcopy(candidate.state)})
        if candidate is not None:
            if _family_opening is not None:
                candidate = replace(candidate, trace={**candidate.trace,
                    'familyOpening': copy.deepcopy(dict(_family_opening))})
            if self.generation is not None:
                candidate = replace(candidate, trace={**candidate.trace,
                    "generationRequest": copy.deepcopy(self.generation)})
            from .result_authority import _validate_trace_quality_evidence_v1
            validator = (_validate_trace_quality_evidence_v1
                if _quality_verified_artifacts else validate_trace_quality_evidence_v1)
            validator(
                candidate.trace, rules_hash=self.rules.sha256, rules=self.rules,
                require_family_opening=True,
                required_policy=self.quality_policy if status == "qualified" else None,
                required_proof_profiles=self.quality_proof_profiles if status == "qualified" else None,
                required_search_profiles=self.quality_search_profiles,
                required_execution=self.quality_execution if status == "qualified" else None,
                **({"verified_artifacts": frozenset(_quality_verified_artifacts)}
                   if _quality_verified_artifacts else {}),
            )
            if status == "qualified":
                from .lesson_tags import derive_trace_lesson_tags_v1
                from .family_catalog import TagQueryV1
                lesson_tags = derive_trace_lesson_tags_v1(
                    candidate.trace, normalized=_lesson_normalized, rules=self.rules)
                target = getattr(self.quality_execution, "family_target", None)
                if lesson_tags is not None:
                    candidate = replace(candidate, trace={**candidate.trace, "lessonTags": lesson_tags})
                if target is not None and (lesson_tags is None or not
                        TagQueryV1.build(target["tagQuery"]).matches(lesson_tags["tags"])):
                    status = "semantic_postcondition_miss"
                    candidate_attempt_reason = "family_target_result_tags_mismatch"
                if self.generation is not None and (lesson_tags is None or not
                        TagQueryV1.build(self.generation["tagQuery"]).matches(lesson_tags["tags"])):
                    status = "semantic_postcondition_miss"
                    candidate_attempt_reason = "generation_result_tags_mismatch"
        if self.backend_id == _V3_BACKEND_ID and candidate is not None:
            candidate = _upgrade_candidate_receipts_v3_1(candidate)
        if candidate is not None:
            _validate_play_policy_continuation_binding_v1(
                candidate,
                graph=graph,
                proof=proof_outcome,
            )
            if not (
                (proof is None and proof_outcome is None and graph is None)
                or (
                    proof is not None
                    and proof_outcome is not None
                    and graph is not None
                )
            ):
                raise RuntimeError("proof artifacts must be supplied as one bound set")
            construction_document = {
                stage: item.model_hash
                for stage, item in sorted(candidate.stage_receipts.items())
                if stage not in {"proof", "tactics"}
            }
            if self.rule_source_catalog is not None:
                construction_document["ruleSourceCatalog"] = copy.deepcopy(
                    self.rule_source_catalog.data
                )
            proof_model_document = (
                proof_outcome.model_document()
                if proof_outcome is not None
                else {"schemaVersion": 1, "status": "not_proven"}
            )
            if graph is not None:
                graph_document = graph.to_dict()
                graph_document.pop("sha256", None)
            else:
                graph_document = {
                    "schemaVersion": 1,
                    "status": "graph_not_compiled",
                }
            if proof is not None:
                proof_outcome_document = copy.deepcopy(dict(proof))
                proof_outcome_document.pop("sha256")
            else:
                proof_outcome_document = {
                    "schemaVersion": 1,
                    "status": "proof_not_run",
                }
            trace_document = copy.deepcopy(dict(candidate.trace))
        else:
            # Terminal executions still own truthful stage artifacts.  These
            # documents are receipts, not a fabricated playable trace.
            construction_document = {
                "schemaVersion": 1,
                "status": (
                    "not_constructed"
                    if completed_candidates == 0
                    else "candidate_not_retained"
                ),
                "preflightHash": preflight.sha256,
                "compiledStages": list(preflight.compiled_stages),
                "namedCore": (
                    list(evaluation.named_core) if evaluation is not None else []
                ),
            }
            if structural_infeasibility_receipt is not None:
                construction_document["structuralInfeasibility"] = copy.deepcopy(
                    dict(structural_infeasibility_receipt)
                )
            if self.rule_source_catalog is not None:
                construction_document["ruleSourceCatalog"] = copy.deepcopy(
                    self.rule_source_catalog.data
                )
            proof_model_document = {
                "schemaVersion": 1,
                "status": "not_compiled",
                "reason": status,
            }
            graph_document = {
                "schemaVersion": 1,
                "status": "graph_not_compiled",
                "reason": status,
            }
            trace_document = {
                "schemaVersion": 1,
                "status": "trace_not_available",
                "reason": status,
                "playable": False,
            }
            proof_outcome_document = {
                "schemaVersion": 1,
                "status": "proof_not_run",
                "reason": status,
            }
            if counterstrategy_core is not None:
                proof_model_document = {"schemaVersion": 1, "kind": "native_counterstrategy_models",
                    "certificate": counterstrategy_core}
                proof_outcome_document = {"schemaVersion": 1, "kind": "native_counterstrategy_outcome",
                    "status": status, "certificateHash": counterstrategy_core["sha256"],
                    "claimSatisfied": False if counterstrategy_core["status"] == "certified" else None,
                    "stateSpecificRootActionsEvaluated": False}

        if self.generation is not None:
            trace_document["generationRequest"] = copy.deepcopy(self.generation)
        if candidate is not None or status in EXECUTION_STATUSES:
            domain_ir_hashes = {
                "auction": AuctionDomainIRV1.from_rules(self.rules.data).sha256,
                "escalation": EscalationDomainIRV1.from_rules(self.rules.data).sha256,
                "play": self.rules.play_domain.sha256,
                "projects": self.rules.project_domain.sha256,
                "scoring": ScoringDomainIRV1.from_rules(self.rules.data).sha256,
                "structural": self.rules.structural_domain.sha256,
                **(
                    {"ruleSources": self.rule_source_catalog.sha256}
                    if self.rule_source_catalog is not None
                    else {}
                ),
            }
            if counterstrategy_core is not None and counterstrategy_core.get("domainIrHashes") != {
                name: domain_ir_hashes[name] for name in ("play", "structural", "projects", "scoring")
            }:
                raise ValueError("native counterstrategy domain authority mismatch")
            if self.backend_id == _V3_BACKEND_ID:
                if scenario_hash is None:
                    raise RuntimeError("V3.1 compilation receipt requires scenario identity")
                stage_receipts = (
                    {
                        stage: receipt.to_dict()
                        for stage, receipt in sorted(candidate.stage_receipts.items())
                    }
                    if candidate is not None
                    else {}
                )
                candidate_identity = None
                if candidate is not None:
                    candidate_identity = {
                        "schemaVersion": 1,
                        "origin": candidate.origin,
                        "noGoodVector": list(candidate.no_good_vector),
                        "exactStateId": candidate.exact_state_id,
                        "stateHash": canonical_sha256(candidate.state),
                    }
                construction_core = {
                    "schemaVersion": 2,
                    "kind": "ortools_research_compilation_receipt_v3",
                    "backendId": self.backend_id,
                    "backendVersion": self.backend_version,
                    "status": status,
                    "scenarioHash": scenario_hash,
                    "preflightHash": preflight.sha256,
                    "compiledStages": list(preflight.compiled_stages),
                    "executedStages": sorted(stage_receipts),
                    "configurationAuthority": {
                        "bundleRevisionId": self.bundle_revision_id,
                        "bundleHash": self.bundle_hash,
                        "domainIrHashes": domain_ir_hashes,
                        "loweringManifestHash": preflight.lowering_manifest_hash,
                    },
                    "stageReceipts": stage_receipts,
                    "candidateIdentity": candidate_identity,
                    "predicateEvaluation": (
                        evaluation.to_dict() if evaluation is not None else None
                    ),
                    "semanticArtifactHashes": {
                        "proofModel": canonical_sha256(proof_model_document),
                        "proofGraph": canonical_sha256(graph_document),
                        "completeTrace": canonical_sha256(trace_document),
                        "proofOutcome": canonical_sha256(proof_outcome_document),
                    },
                }
                construction_document = {
                    **construction_core,
                    "sha256": canonical_sha256(construction_core),
                }
            construction_model_hash = canonical_sha256(construction_document)
            proof_model_hash = canonical_sha256(proof_model_document)
            graph_hash = canonical_sha256(graph_document)
            trace_hash = canonical_sha256(trace_document)
            proof_hash = canonical_sha256(proof_outcome_document)
            authority = ResearchAuthorityV1.build(
                backend_id=self.backend_id,
                backend_version=self.backend_version,
                solver_version=version("ortools"),
                bundle_revision_id=self.bundle_revision_id,
                bundle_hash=self.bundle_hash,
                domain_ir_hashes=domain_ir_hashes,
                lowering_manifest_hash=preflight.lowering_manifest_hash,
                construction_model_hash=construction_model_hash,
                proof_model_hash=proof_model_hash,
                graph_hash=graph_hash,
                trace_hash=trace_hash,
                proof_outcome_hash=proof_hash,
            ).to_dict()
            artifact_preimages = OrToolsResearchArtifactPreimagesV1.build(
                authority=authority,
                construction_model=construction_document,
                proof_model=proof_model_document,
                proof_graph=graph_document,
                trace=trace_document,
                proof_outcome=proof_outcome_document,
            )
        semantic_core = {
            "schemaVersion": 1,
            "backendId": self.backend_id,
            "backendVersion": self.backend_version,
            "status": status,
            "preflightHash": preflight.sha256,
            "candidateHash": candidate.sha256 if candidate else None,
            "predicateEvaluationHash": evaluation.sha256 if evaluation else None,
            "proofHash": proof.get("sha256") if proof else None,
            "researchAuthorityHash": authority.get("sha256") if authority else None,
            "rejectionHashes": list(rejection_hashes),
        }
        if structural_infeasibility_receipt is not None:
            semantic_core["structuralInfeasibilityHash"] = (
                structural_infeasibility_receipt["sha256"]
            )
        if counterstrategy_core is not None:
            semantic_core["counterstrategyHash"] = counterstrategy_core["sha256"]
        semantic_hash = canonical_sha256(semantic_core)
        timings = {
            stage: max(0.0, float((stage_timings or {}).get(stage, 0.0)))
            for stage in _EXECUTION_STAGE_TIMING_KEYS
        }
        timings["total"] = max(0.0, time.perf_counter() - started)
        receipt_core = {
            "schemaVersion": 1,
            "semanticResultHash": semantic_hash,
            "stageTimings": timings,
            "cpuSeconds": max(0.0, time.process_time() - cpu_started),
            "completedCandidates": completed_candidates,
            "solverVersion": version("ortools"),
            "machine": {"system": platform.system(), "machine": platform.machine()},
        }
        if _family_opening is not None and candidate is None:
            receipt_core['familyOpening'] = copy.deepcopy(dict(_family_opening))
        if incomplete_receipt is not None:
            partial = copy.deepcopy(dict(incomplete_receipt))
            partial_hash = partial.pop("sha256", None)
            if partial_hash != canonical_sha256(partial):
                raise ValueError("incomplete compilation receipt hash mismatch")
            receipt_core["incompleteCompilation"] = {
                **partial,
                "sha256": partial_hash,
            }
        if candidate_attempt_reason is not None:
            receipt_core["candidateAttemptReason"] = candidate_attempt_reason
        if counterstrategy_core is not None:
            receipt_core["counterstrategy"] = copy.deepcopy(dict(counterstrategy_receipt))
        if (
            quality_handoff_diagnostic is not None
            and status in {"semantic_postcondition_miss", "proof_failed"}
        ):
            handoff_core = {
                **copy.deepcopy(dict(quality_handoff_diagnostic)),
                "finalProofHash": (
                    proof_outcome.sha256 if proof_outcome is not None else None
                ),
                "finalWinningRootActionCount": (
                    proof_outcome.winning_root_action_count
                    if proof_outcome is not None
                    else None
                ),
            }
            receipt_core["qualityHandoffDiagnostic"] = {
                **handoff_core,
                "sha256": canonical_sha256(handoff_core),
            }
        if structural_infeasibility_receipt is not None:
            structural = copy.deepcopy(dict(structural_infeasibility_receipt))
            structural_hash = structural.pop("sha256", None)
            if structural_hash != canonical_sha256(structural):
                raise ValueError("structural infeasibility receipt hash mismatch")
            receipt_core["structuralInfeasibility"] = {
                **structural,
                "sha256": structural_hash,
            }
        receipt = {**receipt_core, "sha256": canonical_sha256(receipt_core)}
        result = OrToolsUnifiedResearchResultV1(
            status,
            preflight,
            candidate,
            evaluation,
            copy.deepcopy(dict(proof)) if proof else None,
            authority,
            artifact_preimages,
            rejection_hashes,
            receipt,
            semantic_hash,
            self.backend_id,
            self.backend_version,
        )
        if candidate is not None and _quality_verified_artifacts:
            object.__setattr__(result, "_quality_verified_artifacts",
                frozenset(_quality_verified_artifacts))
        return result


__all__ = [
    "BACKEND_ID",
    "BACKEND_VERSION",
    "EXECUTION_STATUSES",
    "DefaultStageContextCompilerV1",
    "GenericCompleteTraceCompilerV1",
    "resolve_quality_play_continuation_v1",
    "resolve_quality_auction_continuation_v1",
    "resolve_quality_round_continuation_v1",
    "resolve_quality_escalation_opportunity_v1",
    "resolve_quality_escalation_continuation_v1",
    "resolve_quality_post_auction_continuation_v1",
    "OrToolsCandidateAttemptV1",
    "OrToolsNeutralCandidateV1",
    "OrToolsPredicateEvaluationV1",
    "OrToolsResearchArtifactPreimagesV1",
    "OrToolsResearchCampaignResultV1",
    "OrToolsResearchCompilerV1",
    "OrToolsResearchPreflightV1",
    "OrToolsResearchStageReceiptV1",
    "OrToolsResearchStageReceiptV2",
    "OrToolsStageContextBindingV1",
    "OrToolsUnifiedResearchExecutorV1",
    "OrToolsUnifiedResearchResultV1",
]
