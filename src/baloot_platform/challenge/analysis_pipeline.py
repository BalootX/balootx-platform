"""Staged actor-safe analysis over every decision in one verified trace."""
from __future__ import annotations

import concurrent.futures
import copy
import math
import multiprocessing
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Mapping

from baloot_platform.analysis.bidding import analyze_bid
from baloot_platform.analysis.belief_v7 import (
    MAXIMUM_HIDDEN_DEAL_ATTEMPTS,
    BeliefPolicy,
)
from baloot_platform.analysis.evaluator import (
    ANALYSIS_OBJECTIVES,
    AnalysisPolicy,
    analyze,
)
from baloot_platform.analysis.models import InformationState
from baloot_platform.analysis.decision_scores import (
    BID_DECISION_OBJECTIVE_V1,
    BID_DECISION_TIE_BREAK_V1,
    PLAY_DECISION_TIE_BREAK_V1,
    PRIMARY_DECISION_COHORT_SEED_POLICY_V1,
    PrimaryDecisionScoreEvidenceV1,
    bid_score_rows_v1,
    canonical_json_bytes_v1,
    legal_action_set_hash_v1,
    play_score_rows_v1,
    primary_decision_score_evidence_v1,
    primary_decision_score_confidence_v1,
    validate_primary_decision_score_evidence_v1,
)
from baloot_platform.analysis.search_backend import ExactSearchBackend
from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Seat
from baloot_platform.core.rules import RuleBook
from baloot_platform.plugins.base import PluginDescriptor

from .decision_cohort import (
    PRIMARY_DECISION_COHORT_SALT_V1,
    derive_actor_observer_cohort_seed_v1,
)
from .evidence_v3 import (
    DecisionLedgerRecordV1,
    DecisionLedgerV1,
    LaffaEvidenceV2,
    canonical_sha256,
)
from .finalist_competence import (
    canonical_finalist_cohort_v1,
    canonical_score_confidence_v1,
    canonical_score_support_classification_v1,
    canonical_surrounding_score_projection_v1,
    finalist_adjudication_target_required_v2,
    normalize_finalist_adjudication_profile_v2,
    project_finalist_adjudication_v2,
)
from .foundry import MultiPerspectiveFoundryV1
from .profiles import SelectionProfileV1, decision_materiality_v1
from .replay import ReplayDecisionStateV1, TraceDecisionProjectionStoreV1
from .trace import CompleteGameTraceV1


_FORCED_LEGALITY_PROOF_KEYS_V1 = frozenset({
    "schemaVersion",
    "proofType",
    "decisionKind",
    "actor",
    "publicPrefixHash",
    "rulesHash",
    "legalActionCount",
    "legalActionSetHash",
    "selectedAction",
    "selectedActionHash",
    "sha256",
})
_TRACE_DECISION_ANALYSIS_KEYS_V3 = frozenset({
    "schemaVersion",
    "decisionId",
    "ordinal",
    "actor",
    "decisionKind",
    "forced",
    "actualAction",
    "bestAction",
    "alternatives",
    "engineMethod",
    "hiddenDeals",
    "effectiveSampleSize",
    "auditConfidence",
    "predictedDifficulty",
    "debateScore",
    "expectedRegret",
    "winProbabilityDrop",
    "catastrophicCrossing",
    "qualityScore",
    "reasons",
    "supportedMetrics",
    "unsupportedMetrics",
    "primaryDecisionScoreEvidence",
    "forcedLegalityProof",
    "analysisHash",
})
_TRACE_ANALYSIS_RESULT_KEYS_V1 = frozenset({
    "schemaVersion",
    "traceHash",
    "profileHash",
    "analyzerId",
    "analyzerVersion",
    "decisions",
    "laffaEvidence",
    "sha256",
})
_TRACE_ANALYSIS_RESULT_KEYS_V2 = frozenset({
    *_TRACE_ANALYSIS_RESULT_KEYS_V1,
    "confirmedDecisionIds",
    "screeningProfileHash",
    "confirmationProfileHash",
})
_TRACE_ANALYSIS_RESULT_KEYS_V3 = frozenset({
    *_TRACE_ANALYSIS_RESULT_KEYS_V2,
    "adjudicationEvidence",
    "unresolvedDecisionIds",
})
_FORCED_DECISION_ALTERNATIVE_KEYS_V3 = frozenset({
    "action",
    "rank",
    "expectedMargin",
    "winProbability",
    "confidence95",
    "downsideCvar10",
    "volatility",
    "bestShare",
    "exactShare",
    "principalVariation",
})
_BID_DECISION_ALTERNATIVE_KEYS_V3 = frozenset({
    *_FORCED_DECISION_ALTERNATIVE_KEYS_V3,
    "contractSuccessProbability",
})
_PLAY_DECISION_ALTERNATIVE_KEYS_V3 = frozenset({
    *_BID_DECISION_ALTERNATIVE_KEYS_V3,
    "expectedTeamAwardedPoints",
    "expectedOpponentAwardedPoints",
    "matchWinProbability",
    "matchWinConfidence95",
    "searchNodes",
    "searchCompletionStatus",
    "samples",
    "effectiveSamples",
})
_DECISION_ALTERNATIVE_REQUIRED_KEYS_V3 = frozenset({
    "action",
    "rank",
    "expectedMargin",
    "winProbability",
    "bestShare",
})


def _canonical_sha256_digest(value: Any, *, field: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{field} must be a canonical lowercase SHA-256 digest")
    return digest


def _forced_legality_proof_v1(
    snapshot: ReplayDecisionStateV1,
    *,
    rules_hash: str,
) -> dict[str, Any]:
    """Bind the independently replay-verified singleton legal action set."""

    record = snapshot.decision
    legal = tuple(dict(item) for item in record.legal_actions)
    selected = dict(record.chosen_action)
    if not record.forced or len(legal) != 1:
        raise ValueError(
            "forced decision proof requires exactly one replay-verified legal action"
        )
    if canonical_json_bytes_v1(legal[0]) != canonical_json_bytes_v1(selected):
        raise ValueError("forced decision selected action is not its only legal action")
    core = {
        "schemaVersion": 1,
        "proofType": "authoritative_forced_legal_action_v1",
        "decisionKind": record.decision_kind,
        "actor": record.acting_seat.name.lower(),
        "publicPrefixHash": record.public_prefix_hash,
        "rulesHash": str(rules_hash),
        "legalActionCount": 1,
        "legalActionSetHash": legal_action_set_hash_v1(legal),
        "selectedAction": selected,
        "selectedActionHash": canonical_sha256(selected),
    }
    return {**core, "sha256": canonical_sha256(core)}


def _validate_forced_legality_proof_v1(
    value: Mapping[str, Any],
    *,
    record: DecisionLedgerRecordV1 | None = None,
    rules_hash: str | None = None,
) -> dict[str, Any]:
    raw = dict(value)
    if set(raw) != _FORCED_LEGALITY_PROOF_KEYS_V1:
        raise ValueError(
            "forced legality proof is incomplete or contains unknown keys"
        )
    if raw["schemaVersion"] != 1 or isinstance(raw["schemaVersion"], bool):
        raise ValueError("unsupported forced legality proof schema")
    if raw["proofType"] != "authoritative_forced_legal_action_v1":
        raise ValueError("unsupported forced legality proof authority")
    if raw["decisionKind"] not in {"bid", "play"}:
        raise ValueError("forced legality proof decision kind is unsupported")
    if raw["actor"] not in {seat.name.lower() for seat in Seat}:
        raise ValueError("forced legality proof actor is invalid")
    if raw["legalActionCount"] != 1 or isinstance(
        raw["legalActionCount"], bool
    ):
        raise ValueError("forced legality proof must bind one legal action")
    selected = dict(raw.get("selectedAction") or {})
    if not selected:
        raise ValueError("forced legality proof selected action is required")
    if _canonical_sha256_digest(
        raw["selectedActionHash"], field="forced selected action hash"
    ) != canonical_sha256(selected):
        raise ValueError("forced legality proof selected action checksum mismatch")
    if raw["legalActionSetHash"] != legal_action_set_hash_v1((selected,)):
        raise ValueError("forced legality proof singleton action-set checksum mismatch")
    for key in (
        "publicPrefixHash",
        "rulesHash",
        "legalActionSetHash",
        "sha256",
    ):
        _canonical_sha256_digest(raw[key], field=f"forced legality {key}")
    core = {key: item for key, item in raw.items() if key != "sha256"}
    if raw["sha256"] != canonical_sha256(core):
        raise ValueError("forced legality proof checksum mismatch")
    if record is not None:
        if not record.forced or len(record.legal_actions) != 1:
            raise ValueError("forced legality proof does not bind a forced ledger decision")
        expected = {
            "decisionKind": record.decision_kind,
            "actor": record.acting_seat.name.lower(),
            "publicPrefixHash": record.public_prefix_hash,
            "legalActionSetHash": legal_action_set_hash_v1(record.legal_actions),
            "selectedActionHash": canonical_sha256(dict(record.chosen_action)),
        }
        if any(raw[key] != expected_value for key, expected_value in expected.items()):
            raise ValueError("forced legality proof ledger binding mismatch")
        if canonical_json_bytes_v1(selected) != canonical_json_bytes_v1(
            dict(record.chosen_action)
        ):
            raise ValueError("forced legality proof selected action binding mismatch")
    if rules_hash is not None and raw["rulesHash"] != rules_hash:
        raise ValueError("forced legality proof rules binding mismatch")
    return raw


@dataclass(frozen=True, slots=True)
class DecisionAnalysisExecutorProfileV1:
    worker_count: int
    maximum_pending_per_worker: int
    maximum_seconds: float
    poll_seconds: float
    sha256: str

    @classmethod
    def build(
        cls,
        *,
        worker_count: int = 1,
        maximum_pending_per_worker: int = 1,
        maximum_seconds: float = 900.0,
        poll_seconds: float = 0.05,
    ) -> "DecisionAnalysisExecutorProfileV1":
        workers = int(worker_count)
        pending = int(maximum_pending_per_worker)
        seconds = float(maximum_seconds)
        poll = float(poll_seconds)
        if not 1 <= workers <= 4:
            raise ValueError("decision analysis worker count must be between one and four")
        if not 1 <= pending <= 2:
            raise ValueError(
                "decision analysis pending tasks per worker must be one or two"
            )
        if not 0.05 <= seconds <= 3_600.0:
            raise ValueError(
                "decision analysis timeout must be between 0.05 and 3600 seconds"
            )
        if not 0.01 <= poll <= 1.0:
            raise ValueError("decision analysis poll interval must be between 0.01 and one second")
        core = {
            "schemaVersion": 1,
            "workerCount": workers,
            "maximumPendingPerWorker": pending,
            "maximumSeconds": seconds,
            "pollSeconds": poll,
        }
        return cls(workers, pending, seconds, poll, canonical_sha256(core))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "workerCount": self.worker_count,
            "maximumPendingPerWorker": self.maximum_pending_per_worker,
            "maximumSeconds": self.maximum_seconds,
            "pollSeconds": self.poll_seconds,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class DecisionAnalysisExecutorSetV1:
    """Versioned executor settings for every post-trace analysis stage."""

    profile_id: str
    revision: str
    screening: DecisionAnalysisExecutorProfileV1
    confirmation: DecisionAnalysisExecutorProfileV1
    finalist_reanalysis: DecisionAnalysisExecutorProfileV1
    sha256: str

    @classmethod
    def build(
        cls, value: Mapping[str, Any]
    ) -> "DecisionAnalysisExecutorSetV1":
        allowed = {
            "schemaVersion",
            "id",
            "revision",
            "screening",
            "confirmation",
            "finalistReanalysis",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                f"unknown decision analysis executor keys: {sorted(unknown)}"
            )
        if int(value.get("schemaVersion", 0)) != 1:
            raise ValueError("unsupported decision analysis executor schema")
        profile_id = str(value.get("id") or "").strip()
        revision = str(value.get("revision") or "").strip()
        if not profile_id or not revision:
            raise ValueError("decision analysis executor identity is required")

        def stage(name: str) -> DecisionAnalysisExecutorProfileV1:
            raw = value.get(name)
            if not isinstance(raw, Mapping):
                raise ValueError(
                    f"decision analysis executor {name} stage is required"
                )
            stage_value = dict(raw)
            expected = {
                "workerCount",
                "maximumPendingPerWorker",
                "maximumSeconds",
                "pollSeconds",
            }
            if set(stage_value) != expected:
                raise ValueError(
                    f"decision analysis executor {name} stage is incomplete "
                    "or contains unknown keys"
                )
            return DecisionAnalysisExecutorProfileV1.build(
                worker_count=int(stage_value["workerCount"]),
                maximum_pending_per_worker=int(
                    stage_value["maximumPendingPerWorker"]
                ),
                maximum_seconds=float(stage_value["maximumSeconds"]),
                poll_seconds=float(stage_value["pollSeconds"]),
            )

        screening = stage("screening")
        confirmation = stage("confirmation")
        finalist = stage("finalistReanalysis")
        core = {
            "schemaVersion": 1,
            "id": profile_id,
            "revision": revision,
            "screening": screening.to_dict(),
            "confirmation": confirmation.to_dict(),
            "finalistReanalysis": finalist.to_dict(),
        }
        return cls(
            profile_id,
            revision,
            screening,
            confirmation,
            finalist,
            canonical_sha256(core),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "id": self.profile_id,
            "revision": self.revision,
            "screening": self.screening.to_dict(),
            "confirmation": self.confirmation.to_dict(),
            "finalistReanalysis": self.finalist_reanalysis.to_dict(),
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class TraceAnalysisProfileV1:
    bid_samples: int
    play_samples: int
    exact_allocation_limit: int
    exact_search_max_remaining: int
    maximum_sampling_attempts: int
    exact_search_max_nodes: int
    exact_search_max_seconds: float
    exact_search_max_depth: int | None
    exact_search_max_cache_entries: int
    search_backend_id: str
    objective: str
    use_conventions: bool
    sha256: str

    @classmethod
    def build(
        cls,
        *,
        bid_samples: int = 16,
        play_samples: int = 16,
        exact_allocation_limit: int = 5_000,
        exact_search_max_remaining: int = 8,
        maximum_sampling_attempts: int = 20_000,
        exact_search_max_nodes: int = 250_000,
        exact_search_max_seconds: float = 2.0,
        exact_search_max_depth: int | None = None,
        exact_search_max_cache_entries: int = 100_000,
        search_backend_id: str = "python_reference_exact_search_v1",
        objective: str = "match_win_probability",
        use_conventions: bool = False,
        belief_policy: BeliefPolicy | None = None,
    ) -> "TraceAnalysisProfileV1":
        bid_count = int(bid_samples)
        play_count = int(play_samples)
        attempts = int(maximum_sampling_attempts)
        allocation_limit = int(exact_allocation_limit)
        resolved_objective = str(objective).strip()
        if min(bid_count, play_count) < 2:
            raise ValueError("paired trace analysis requires at least two samples")
        if max(bid_count, play_count) > MAXIMUM_HIDDEN_DEAL_ATTEMPTS:
            raise ValueError("trace analysis sample count exceeds the safety bound")
        if allocation_limit < 1 or attempts < 1:
            raise ValueError("trace allocation and sampling limits must be positive")
        if not max(bid_count, play_count) <= attempts <= MAXIMUM_HIDDEN_DEAL_ATTEMPTS:
            raise ValueError("trace analysis sampling attempts violate safety bounds")
        if belief_policy is not None:
            if max(bid_count, play_count) > belief_policy.maximum_particles:
                raise ValueError("trace analysis sample count violates belief bounds")
            if allocation_limit > belief_policy.maximum_exact_allocation_limit:
                raise ValueError(
                    "trace analysis exact-allocation limit violates belief bounds"
                )
        if not 0 <= int(exact_search_max_remaining) <= 32:
            raise ValueError("trace exact-search remaining threshold must be in [0, 32]")
        if int(exact_search_max_nodes) < 1 or float(exact_search_max_seconds) <= 0:
            raise ValueError("trace exact-search budgets must be positive")
        if exact_search_max_depth is not None and int(exact_search_max_depth) < 1:
            raise ValueError("trace exact-search depth must be positive when configured")
        if int(exact_search_max_cache_entries) < 0:
            raise ValueError("trace exact-search cache bound cannot be negative")
        if not str(search_backend_id).strip():
            raise ValueError("trace exact-search backend identity is required")
        if resolved_objective not in ANALYSIS_OBJECTIVES:
            raise ValueError(
                f"unsupported trace analysis objective: {resolved_objective}"
            )
        core = {
            "schemaVersion": 1,
            "bidSamples": bid_count,
            "playSamples": play_count,
            "exactAllocationLimit": allocation_limit,
            "exactSearchMaxRemaining": int(exact_search_max_remaining),
            "maximumSamplingAttempts": attempts,
            "exactSearchMaxNodes": int(exact_search_max_nodes),
            "exactSearchMaxSeconds": float(exact_search_max_seconds),
            "exactSearchMaxDepth": (
                int(exact_search_max_depth) if exact_search_max_depth is not None else None
            ),
            "exactSearchMaxCacheEntries": int(exact_search_max_cache_entries),
            "searchBackendId": str(search_backend_id),
            "objective": resolved_objective,
            "useConventions": bool(use_conventions),
        }
        return cls(
            bid_count,
            play_count,
            allocation_limit,
            int(exact_search_max_remaining),
            attempts,
            int(exact_search_max_nodes),
            float(exact_search_max_seconds),
            int(exact_search_max_depth) if exact_search_max_depth is not None else None,
            int(exact_search_max_cache_entries),
            str(search_backend_id),
            resolved_objective,
            bool(use_conventions),
            canonical_sha256(core),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "bidSamples": self.bid_samples,
            "playSamples": self.play_samples,
            "exactAllocationLimit": self.exact_allocation_limit,
            "exactSearchMaxRemaining": self.exact_search_max_remaining,
            "maximumSamplingAttempts": self.maximum_sampling_attempts,
            "exactSearchMaxNodes": self.exact_search_max_nodes,
            "exactSearchMaxSeconds": self.exact_search_max_seconds,
            "exactSearchMaxDepth": self.exact_search_max_depth,
            "exactSearchMaxCacheEntries": self.exact_search_max_cache_entries,
            "searchBackendId": self.search_backend_id,
            "objective": self.objective,
            "useConventions": self.use_conventions,
            "sha256": self.sha256,
        }


def build_finalist_reanalysis_trace_profile_v1(
    stability: Mapping[str, Any],
    confirmation: Mapping[str, Any],
    *,
    search_backend_id: str,
    belief_policy: BeliefPolicy | None = None,
) -> TraceAnalysisProfileV1:
    """Compile the one governed finalist reanalysis profile.

    Service assembly, Research context reconstruction, and publication
    validation must all use this projection.  In particular, finalist
    reanalysis is intentionally distinct from the deep-confirmation profile.
    """

    return TraceAnalysisProfileV1.build(
        bid_samples=int(stability.get("reanalysis_samples", 0)),
        play_samples=int(stability.get("reanalysis_samples", 0)),
        exact_allocation_limit=int(stability.get("exact_allocation_limit", 0)),
        exact_search_max_remaining=int(
            stability.get("exact_search_remaining", -1)
        ),
        maximum_sampling_attempts=int(
            stability.get("maximum_sampling_attempts", 0)
        ),
        exact_search_max_nodes=int(stability.get("exact_search_max_nodes", 0)),
        exact_search_max_seconds=float(
            stability.get("exact_search_max_seconds", 0.0)
        ),
        exact_search_max_depth=int(
            confirmation.get("exact_search_max_depth", 0)
        ),
        exact_search_max_cache_entries=int(
            confirmation.get("exact_search_max_cache_entries", -1)
        ),
        search_backend_id=str(search_backend_id),
        objective=str(confirmation.get("objective") or ""),
        use_conventions=False,
        belief_policy=belief_policy,
    )


def trace_analysis_composite_identity_hash_v1(
    *,
    mode: str,
    descriptor: Mapping[str, Any],
    profile_hash: str,
    analysis_policy: AnalysisPolicy,
    belief_policy: BeliefPolicy,
    match_value_profile_hash: str,
    rules_hash: str,
    convention_hash: str,
    challenge_hash: str,
    performance_hash: str,
    search_backend_descriptor: Mapping[str, Any],
) -> str:
    """Hash every authority that can change one trace-analysis result."""

    return canonical_sha256(
        {
            "schemaVersion": 1,
            "mode": str(mode),
            "descriptor": copy.deepcopy(dict(descriptor)),
            "profileHash": str(profile_hash),
            "analysisPolicy": asdict(analysis_policy),
            "beliefPolicy": asdict(belief_policy),
            "matchValueProfileHash": str(match_value_profile_hash),
            "rulesHash": str(rules_hash),
            "conventionHash": str(convention_hash),
            "challengeHash": str(challenge_hash),
            "performanceHash": str(performance_hash),
            "searchBackend": copy.deepcopy(dict(search_backend_descriptor)),
        }
    )


@dataclass(frozen=True, slots=True)
class CascadedTraceAnalysisProfileV1:
    screening_profile_hash: str
    confirmation_profile_hash: str
    maximum_confirmation_decisions_per_trace: int
    minimum_screening_score: float
    sha256: str

    @classmethod
    def build(
        cls,
        screening_profile: TraceAnalysisProfileV1,
        confirmation_profile: TraceAnalysisProfileV1,
        *,
        maximum_confirmation_decisions_per_trace: int,
        minimum_screening_score: float,
    ) -> "CascadedTraceAnalysisProfileV1":
        maximum = int(maximum_confirmation_decisions_per_trace)
        minimum = float(minimum_screening_score)
        if not 1 <= maximum <= 64:
            raise ValueError("confirmation decision bound must be in [1, 64]")
        if not 0.0 <= minimum <= 100.0:
            raise ValueError("minimum screening score must be in [0, 100]")
        core = {
            "schemaVersion": 1,
            "screeningProfileHash": screening_profile.sha256,
            "confirmationProfileHash": confirmation_profile.sha256,
            "maximumConfirmationDecisionsPerTrace": maximum,
            "minimumScreeningScore": minimum,
        }
        return cls.from_mapping({**core, "sha256": canonical_sha256(core)})

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "CascadedTraceAnalysisProfileV1":
        raw = dict(value)
        if set(raw) != {
            "schemaVersion",
            "screeningProfileHash",
            "confirmationProfileHash",
            "maximumConfirmationDecisionsPerTrace",
            "minimumScreeningScore",
            "sha256",
        }:
            raise ValueError(
                "cascaded trace analysis profile is incomplete or contains unknown keys"
            )
        if raw["schemaVersion"] != 1 or isinstance(
            raw["schemaVersion"], bool
        ):
            raise ValueError("unsupported cascaded trace analysis profile schema")
        screening_hash = _canonical_sha256_digest(
            raw["screeningProfileHash"],
            field="cascaded screening profile hash",
        )
        confirmation_hash = _canonical_sha256_digest(
            raw["confirmationProfileHash"],
            field="cascaded confirmation profile hash",
        )
        maximum = raw["maximumConfirmationDecisionsPerTrace"]
        if (
            isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or not 1 <= maximum <= 64
        ):
            raise ValueError("confirmation decision bound must be in [1, 64]")
        minimum = raw["minimumScreeningScore"]
        if not isinstance(minimum, float) or not 0.0 <= minimum <= 100.0:
            raise ValueError(
                "minimum screening score must be a normalized float in [0, 100]"
            )
        core = {
            "schemaVersion": 1,
            "screeningProfileHash": screening_hash,
            "confirmationProfileHash": confirmation_hash,
            "maximumConfirmationDecisionsPerTrace": maximum,
            "minimumScreeningScore": minimum,
        }
        declared_hash = _canonical_sha256_digest(
            raw["sha256"], field="cascaded trace analysis profile sha256"
        )
        if canonical_sha256(core) != declared_hash:
            raise ValueError("cascaded trace analysis profile checksum mismatch")
        return cls(
            screening_hash,
            confirmation_hash,
            maximum,
            minimum,
            declared_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "screeningProfileHash": self.screening_profile_hash,
            "confirmationProfileHash": self.confirmation_profile_hash,
            "maximumConfirmationDecisionsPerTrace": (
                self.maximum_confirmation_decisions_per_trace
            ),
            "minimumScreeningScore": self.minimum_screening_score,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class TraceDecisionAnalysisV3:
    decision_id: str
    ordinal: int
    acting_seat: str
    decision_kind: str
    forced: bool
    actual_action: Mapping[str, Any]
    best_action: Mapping[str, Any]
    alternatives: tuple[Mapping[str, Any], ...]
    method: str
    hidden_deals: int
    effective_sample_size: float
    confidence: float
    difficulty: float
    debate_score: float
    expected_regret: float
    probability_drop: float
    catastrophic_crossing: bool
    supported_metrics: tuple[str, ...]
    unsupported_metrics: tuple[str, ...]
    sha256: str
    primary_decision_score_evidence: PrimaryDecisionScoreEvidenceV1 | None = None
    forced_legality_proof: Mapping[str, Any] | None = None

    def audit_mapping(self) -> dict[str, Any]:
        quality = max(
            0.0,
            100.0 - self.expected_regret * 5.0 - self.probability_drop * 100.0,
        )
        return {
            "decisionId": self.decision_id,
            "ordinal": self.ordinal,
            "actor": self.acting_seat,
            "decisionKind": self.decision_kind,
            "forced": self.forced,
            "actualAction": dict(self.actual_action),
            "bestAction": dict(self.best_action),
            "alternatives": [dict(item) for item in self.alternatives],
            "engineMethod": self.method,
            "hiddenDeals": self.hidden_deals,
            "effectiveSampleSize": self.effective_sample_size,
            "auditConfidence": self.confidence,
            "predictedDifficulty": self.difficulty,
            "debateScore": self.debate_score,
            "expectedRegret": self.expected_regret,
            "winProbabilityDrop": self.probability_drop,
            "catastrophicCrossing": self.catastrophic_crossing,
            "qualityScore": quality,
            "reasons": (
                [{"code": "paired_counterfactual_evidence"}]
                if not self.forced
                else [{"code": "only_legal_action"}]
            ),
            "supportedMetrics": list(self.supported_metrics),
            "unsupportedMetrics": list(self.unsupported_metrics),
            "primaryDecisionScoreEvidence": (
                self.primary_decision_score_evidence.to_dict()
                if self.primary_decision_score_evidence is not None
                else None
            ),
            "forcedLegalityProof": (
                dict(self.forced_legality_proof)
                if self.forced_legality_proof is not None
                else None
            ),
            "analysisHash": self.sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.audit_mapping(), "schemaVersion": 3}

    def validate_ledger_binding(
        self,
        record: DecisionLedgerRecordV1,
        *,
        trace_analysis_profile_hash: str,
        rules_hash: str | None = None,
    ) -> None:
        if (
            self.decision_id != record.decision_id
            or self.ordinal != record.ordinal
            or self.acting_seat != record.acting_seat.name.lower()
            or self.decision_kind != record.decision_kind
            or self.forced != record.forced
            or canonical_json_bytes_v1(self.actual_action)
            != canonical_json_bytes_v1(record.chosen_action)
        ):
            raise ValueError("trace decision analysis ledger binding mismatch")
        if self.forced:
            if self.primary_decision_score_evidence is not None:
                raise ValueError(
                    "forced trace decision cannot contain primary score evidence"
                )
            if self.forced_legality_proof is None:
                raise ValueError(
                    "forced trace decision requires an explicit forced legality proof"
                )
            _validate_forced_legality_proof_v1(
                self.forced_legality_proof,
                record=record,
                rules_hash=rules_hash,
            )
            return
        if self.primary_decision_score_evidence is None:
            raise ValueError(
                "non-forced trace decision requires primary decision score evidence"
            )
        if self.forced_legality_proof is not None:
            raise ValueError(
                "non-forced trace decision cannot contain a forced legality proof"
            )
        expected: dict[str, Any] = {
            "decision_kind": record.decision_kind,
            "actor": record.acting_seat,
            "public_prefix_hash": record.public_prefix_hash,
            "legal_action_set_hash": legal_action_set_hash_v1(
                record.legal_actions
            ),
            "trace_analysis_profile_hash": trace_analysis_profile_hash,
        }
        if rules_hash is not None:
            expected["rules_hash"] = rules_hash
        validate_primary_decision_score_evidence_v1(
            self.primary_decision_score_evidence,
            **expected,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TraceDecisionAnalysisV3":
        if value.get("schemaVersion") != 3 or isinstance(
            value.get("schemaVersion"), bool
        ):
            raise ValueError("unsupported trace decision analysis schema")
        if set(value) != _TRACE_DECISION_ANALYSIS_KEYS_V3:
            raise ValueError(
                "trace decision analysis is incomplete or contains unknown keys"
            )
        for field in ("decisionId", "actor", "decisionKind", "engineMethod"):
            if not isinstance(value[field], str) or not value[field]:
                raise ValueError(f"trace decision {field} must be a non-empty string")
        if value["decisionKind"] not in {"bid", "play"}:
            raise ValueError("trace decision kind is unsupported")
        if (
            isinstance(value["ordinal"], bool)
            or not isinstance(value["ordinal"], int)
            or value["ordinal"] < 0
        ):
            raise ValueError("trace decision ordinal must be a non-negative integer")
        if not isinstance(value.get("forced"), bool):
            raise ValueError("trace decision forced flag must be boolean")
        if not isinstance(value.get("catastrophicCrossing"), bool):
            raise ValueError(
                "trace decision catastrophic-crossing flag must be boolean"
            )
        if not isinstance(value["actualAction"], Mapping) or not isinstance(
            value["bestAction"], Mapping
        ):
            raise ValueError("trace decision actions must be objects")
        if (
            isinstance(value["hiddenDeals"], bool)
            or not isinstance(value["hiddenDeals"], int)
            or value["hiddenDeals"] < 0
        ):
            raise ValueError(
                "trace decision hidden-deal support must be a non-negative integer"
            )
        for field in (
            "effectiveSampleSize",
            "auditConfidence",
            "predictedDifficulty",
            "debateScore",
            "expectedRegret",
            "winProbabilityDrop",
            "qualityScore",
        ):
            if not isinstance(value[field], float) or not math.isfinite(value[field]):
                raise ValueError(
                    f"trace decision {field} must be a normalized finite float"
                )
        if not isinstance(value["analysisHash"], str):
            raise ValueError("trace decision analysis hash must be a string")
        _canonical_sha256_digest(
            value["analysisHash"], field="trace decision analysis hash"
        )
        forced = bool(value["forced"])
        evidence_value = value.get("primaryDecisionScoreEvidence")
        proof_value = value.get("forcedLegalityProof")
        if forced:
            if evidence_value is not None:
                raise ValueError(
                    "forced trace decision cannot contain primary score evidence"
                )
            if not isinstance(proof_value, Mapping):
                raise ValueError(
                    "forced trace decision requires an explicit forced legality proof"
                )
            score_evidence = None
            forced_proof = _validate_forced_legality_proof_v1(proof_value)
        else:
            if not isinstance(evidence_value, Mapping):
                raise ValueError(
                    "non-forced trace decision requires primary decision score evidence"
                )
            if proof_value is not None:
                raise ValueError(
                    "non-forced trace decision cannot contain a forced legality proof"
                )
            score_evidence = validate_primary_decision_score_evidence_v1(
                evidence_value,
                decision_kind=str(value.get("decisionKind") or ""),
                actor=str(value.get("actor") or ""),
            )
            forced_proof = None
        supported_raw = value.get("supportedMetrics")
        unsupported_raw = value.get("unsupportedMetrics")
        if not isinstance(supported_raw, list) or not all(
            isinstance(item, str) and item for item in supported_raw
        ):
            raise ValueError("trace decision supported metrics must be strings")
        if not isinstance(unsupported_raw, list) or not all(
            isinstance(item, str) and item for item in unsupported_raw
        ):
            raise ValueError("trace decision unsupported metrics must be strings")
        if supported_raw != sorted(set(supported_raw)):
            raise ValueError("trace decision supported metrics are not canonical")
        if unsupported_raw != sorted(set(unsupported_raw)):
            raise ValueError("trace decision unsupported metrics are not canonical")
        if set(supported_raw) & set(unsupported_raw):
            raise ValueError(
                "trace decision supported and unsupported metrics must be disjoint"
            )
        alternatives_raw = value.get("alternatives")
        if not isinstance(alternatives_raw, list) or not alternatives_raw or not all(
            isinstance(item, Mapping) for item in alternatives_raw
        ):
            raise ValueError("trace decision alternatives must be a non-empty list")
        if any(
            not _DECISION_ALTERNATIVE_REQUIRED_KEYS_V3.issubset(item)
            or not set(item).issubset(_PLAY_DECISION_ALTERNATIVE_KEYS_V3)
            for item in alternatives_raw
        ):
            raise ValueError(
                "trace decision alternative is incomplete or contains unknown keys"
            )
        core = {
            "schemaVersion": 3,
            "decisionId": str(value.get("decisionId") or ""),
            "ordinal": int(value.get("ordinal", 0)),
            "actingSeat": str(value.get("actor") or ""),
            "decisionKind": str(value.get("decisionKind") or ""),
            "forced": forced,
            "actualAction": dict(value.get("actualAction") or {}),
            "bestAction": dict(value.get("bestAction") or {}),
            "alternatives": [dict(item) for item in alternatives_raw],
            "method": str(value.get("engineMethod") or ""),
            "hiddenDeals": int(value.get("hiddenDeals", 0)),
            "effectiveSampleSize": float(value.get("effectiveSampleSize", 0.0)),
            "difficulty": float(value.get("predictedDifficulty", 0.0)),
            "debateScore": float(value.get("debateScore", 0.0)),
            "expectedRegret": float(value.get("expectedRegret", 0.0)),
            "probabilityDrop": float(value.get("winProbabilityDrop", 0.0)),
            "catastrophicCrossing": bool(value.get("catastrophicCrossing")),
            "supportedMetrics": list(supported_raw),
            "unsupportedMetrics": list(unsupported_raw),
            "primaryDecisionScoreEvidenceHash": (
                score_evidence.sha256 if score_evidence is not None else None
            ),
            "forcedLegalityProofHash": (
                str(forced_proof["sha256"]) if forced_proof is not None else None
            ),
        }
        if score_evidence is not None:
            if canonical_json_bytes_v1(score_evidence.recommended_action) != (
                canonical_json_bytes_v1(core["bestAction"])
            ):
                raise ValueError(
                    "primary decision score recommendation does not match best action"
                )
            if (
                score_evidence.method != core["method"]
                or score_evidence.hidden_deal_support != core["hiddenDeals"]
                or float(score_evidence.effective_sample_size)
                != core["effectiveSampleSize"]
            ):
                raise ValueError(
                    "primary decision score evidence analysis projection mismatch"
                )
            score_actions = {
                canonical_json_bytes_v1(dict(row.get("action") or {}))
                for row in score_evidence.rows
            }
            if canonical_json_bytes_v1(core["actualAction"]) not in score_actions:
                raise ValueError(
                    "primary decision score evidence does not contain actual action"
                )
            if len(core["alternatives"]) != len(score_evidence.rows):
                raise ValueError(
                    "trace decision alternatives do not mirror canonical score rows"
                )
            for alternative, row in zip(
                core["alternatives"], score_evidence.rows, strict=True
            ):
                expected_mirrors = {
                    "rank": int(row["rank"]),
                    "expectedMargin": float(row["expectedRoundMargin"]),
                    "winProbability": float(row["roundLeadProbability"]),
                    "bestShare": float(row["bestShare"]),
                }
                if (
                    canonical_json_bytes_v1(alternative.get("action"))
                    != canonical_json_bytes_v1(row["action"])
                    or any(
                        canonical_json_bytes_v1(alternative.get(field))
                        != canonical_json_bytes_v1(expected)
                        for field, expected in expected_mirrors.items()
                    )
                ):
                    raise ValueError(
                        "trace decision alternatives do not mirror canonical score rows"
                    )
            canonical_projection = canonical_surrounding_score_projection_v1(
                score_evidence,
                actual_action=core["actualAction"],
                best_action=core["bestAction"],
            )
            if (
                core["expectedRegret"]
                != canonical_projection["expectedRegret"]
                or core["probabilityDrop"]
                != canonical_projection["probabilityDrop"]
                or core["catastrophicCrossing"]
                is not canonical_projection["catastrophicCrossing"]
            ):
                raise ValueError(
                    "trace decision qualification metrics drift from canonical score rows"
                )
        else:
            assert forced_proof is not None
            if (
                forced_proof["decisionKind"] != core["decisionKind"]
                or forced_proof["actor"] != core["actingSeat"]
                or canonical_json_bytes_v1(forced_proof["selectedAction"])
                != canonical_json_bytes_v1(core["actualAction"])
            ):
                raise ValueError("forced legality proof decision binding mismatch")
        confidence = (
            primary_decision_score_confidence_v1(score_evidence)
            if score_evidence is not None
            else StagedTraceAnalyzerV1._confidence(
                core["effectiveSampleSize"],
                kind=core["decisionKind"],
                method=core["method"],
                alternatives=core["alternatives"],
            )
        )
        serialized_confidence = value.get("auditConfidence")
        if canonical_json_bytes_v1(serialized_confidence) != canonical_json_bytes_v1(
            confidence
        ):
            raise ValueError("trace decision audit confidence mismatch")
        quality = max(
            0.0,
            100.0
            - core["expectedRegret"] * 5.0
            - core["probabilityDrop"] * 100.0,
        )
        serialized_quality = value.get("qualityScore")
        if canonical_json_bytes_v1(serialized_quality) != canonical_json_bytes_v1(
            quality
        ):
            raise ValueError("trace decision quality score mismatch")
        expected_reasons = [
            {
                "code": (
                    "only_legal_action"
                    if forced
                    else "paired_counterfactual_evidence"
                )
            }
        ]
        if value.get("reasons") != expected_reasons:
            raise ValueError("trace decision reason projection mismatch")
        digest = canonical_sha256(core)
        if str(value.get("analysisHash") or "") != digest:
            raise ValueError("trace decision analysis checksum mismatch")
        return cls(
            core["decisionId"], core["ordinal"], core["actingSeat"],
            core["decisionKind"], core["forced"], core["actualAction"],
            core["bestAction"], tuple(core["alternatives"]), core["method"],
            core["hiddenDeals"], core["effectiveSampleSize"],
            confidence,
            core["difficulty"], core["debateScore"], core["expectedRegret"],
            core["probabilityDrop"], core["catastrophicCrossing"],
            tuple(core["supportedMetrics"]), tuple(core["unsupportedMetrics"]), digest,
            score_evidence, forced_proof,
        )


_ADJUDICATION_STATUSES_V2 = frozenset({
    "strict_confirmed",
    "primary_robust_competent",
    "unresolved",
})
_ADJUDICATION_STRICT_REASONS_V2 = frozenset({
    "adjudication_bound_exceeded",
    "analysis_not_converged",
    "budget_exhausted_low_support",
    "catastrophic_classification_disagreement",
    "cohort_reversal",
    "expected_margin_drift",
    "insufficient_qualified_agreement",
    "invalid_best_share",
    "invalid_confidence",
    "invalid_support_truth_classification",
    "low_best_share",
    "low_confidence",
    "low_hidden_deal_support",
    "missing_primary_decision_score_evidence",
    "missing_majority_margin",
    "missing_primary_boundary_metrics",
    "population_solver_best_known_only",
    "primary_action_not_confirmed",
    "primary_recommendation_not_confirmed",
})
_COMPETENCE_COHORT_STRICT_REASONS_V1 = frozenset({
    "analysis_not_converged",
    "budget_exhausted_low_support",
    "invalid_best_share",
    "invalid_confidence",
    "invalid_support_truth_classification",
    "low_best_share",
    "low_confidence",
    "low_hidden_deal_support",
    "population_solver_best_known_only",
})
_COMPETENCE_COHORT_REASONS_V1 = frozenset({
    "competence_match_win_drop",
    "competence_regret",
    "failure_boundary_crossing",
    "incomplete_failure_boundary_metric",
    "incomplete_match_win_metric",
    "invalid_competence_utility",
    "invalid_confidence",
    "invalid_support_truth_classification",
    "invalid_value_interval",
    "low_confidence",
    "low_hidden_deal_support",
    "missing_competence_utility",
    "primary_action_drift",
})
_COMPETENCE_RESOLUTION_REASONS_V1 = frozenset({
    "competence_expected_margin_drift",
    "competence_value_interval_instability",
    "insufficient_competence_agreement",
})
_ADJUDICATION_REASONS_V2 = frozenset(
    _ADJUDICATION_STRICT_REASONS_V2
    | _COMPETENCE_COHORT_REASONS_V1
    | _COMPETENCE_RESOLUTION_REASONS_V1
)
_ADJUDICATION_EVIDENCE_KEYS_V2 = frozenset({
    "schemaVersion",
    "verdictSchemaVersion",
    "profileHash",
    "adjudicationProfile",
    "selectionProfile",
    "primaryProfile",
    "primaryProfileHash",
    "reanalysisProfileHash",
    "seedNamespace",
    "seedOffset",
    "thirdSeedNamespace",
    "thirdSeedOffset",
    "targetDecisionIds",
    "boundedDecisionIds",
    "overflowDecisionIds",
    "thirdCohortDecisionIds",
    "evaluations",
    "unresolvedDecisionIds",
    "competentDecisionIds",
    "sha256",
})
_ADJUDICATION_EVALUATION_KEYS_V2 = frozenset({
    "decisionId",
    "ordinal",
    "status",
    "strictRecommendationStatus",
    "strictPrimaryActionConfirmed",
    "reasons",
    "strictReasons",
    "competenceEvidence",
    "initialReasons",
    "cohortCount",
    "agreementCount",
    "supporterCohortIndices",
    "majorityActionHash",
    "cohortAnalysisHashes",
    "cohortAnalyses",
    "cohortBestActionHashes",
    "cohortHiddenDeals",
    "cohortConfidence",
    "cohortExhaustiveExactSupport",
    "cohortPopulationCoverageComplete",
    "cohortSolverCompletion",
    "cohortTruthClass",
    "primaryAnalysisHash",
    "reanalysisHash",
    "primaryAnalysis",
    "reanalysis",
    "primaryBestActionHash",
    "reanalysisBestActionHash",
    "selectedAnalysisHash",
    "retainedPrimaryAnalysisHash",
    "primaryAnalysisRetained",
    "featuredTruthEligible",
    "surroundingVerified",
    "primaryHiddenDeals",
    "reanalysisHiddenDeals",
    "primaryConfidence",
    "reanalysisConfidence",
    "primaryExhaustiveExactSupport",
    "reanalysisExhaustiveExactSupport",
    "expectedMarginDrift",
})
_COMPETENCE_EVIDENCE_KEYS_V1 = frozenset({
    "schemaVersion",
    "policy",
    "status",
    "primaryActionHash",
    "agreementCount",
    "minimumAgreement",
    "expectedMarginRegretDrift",
    "maximumExpectedMarginDrift",
    "valueIntervalsStable",
    "resolutionReasons",
    "cohortEvaluations",
    "sha256",
})
_COMPETENCE_COHORT_KEYS_V1 = frozenset({
    "cohortIndex",
    "evidenceHash",
    "traceSeed",
    "bestActionHash",
    "primaryActionHash",
    "expectedMarginRegret",
    "matchWinDrop",
    "matchWinMetricSupported",
    "valueConfidence95",
    "budgetExhausted",
    "populationCoverageComplete",
    "solverCompletion",
    "truthClass",
    "exhaustiveExactSupport",
    "strictReasons",
    "reasons",
    "qualified",
})
_SOLVER_COMPLETION_VALUES_V2 = frozenset({
    "budget_exhausted",
    "cancelled",
    "exact",
    "failed",
    "heuristic_complete",
    "mixed_or_unknown",
    "sampled",
    "unknown",
})
_TRUTH_CLASS_VALUES_V2 = frozenset({
    "exact_optimal",
    "full_population_best_known",
    "unresolved",
})


def _require_exact_keys_v1(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    field: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{field} is incomplete or contains unknown keys")


def _require_bool_v1(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


def _finite_float_v1(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field} must be finite")
    return normalized


def _canonical_reason_list_v1(
    value: Any,
    *,
    allowed: frozenset[str],
    field: str,
    require_sorted: bool = True,
) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{field} must be a list of reason codes")
    if len(value) != len(set(value)):
        raise ValueError(f"{field} contains duplicate reason codes")
    if require_sorted and value != sorted(value):
        raise ValueError(f"{field} is not in canonical order")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{field} contains unsupported reason codes")
    return list(value)


def _canonical_decision_id_list_v1(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{field} must be a list of decision ids")
    if value != sorted(set(value)):
        raise ValueError(f"{field} is not a canonical decision-id set")
    return list(value)


def _semantic_version_v1(value: str) -> tuple[int, int, int]:
    parts = str(value).split(".")
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        raise ValueError("trace analyzer version must be semantic x.y.z")
    return tuple(int(part) for part in parts)  # type: ignore[return-value]


def _validate_competence_evidence_v1(
    raw: Mapping[str, Any],
    *,
    cohorts: tuple[TraceDecisionAnalysisV3, ...],
    primary: TraceDecisionAnalysisV3,
) -> dict[str, Any]:
    evidence = dict(raw)
    _require_exact_keys_v1(
        evidence,
        _COMPETENCE_EVIDENCE_KEYS_V1,
        field="finalist competence evidence",
    )
    if evidence["schemaVersion"] != 1 or isinstance(
        evidence["schemaVersion"], bool
    ):
        raise ValueError("unsupported finalist competence evidence schema")
    if evidence["policy"] != "recorded_primary_robust_competence_v1":
        raise ValueError("unsupported finalist competence policy")
    status = str(evidence["status"])
    if status not in {"primary_robust_competent", "unresolved"}:
        raise ValueError("unsupported finalist competence status")
    declared_hash = _canonical_sha256_digest(
        evidence["sha256"], field="finalist competence sha256"
    )
    core = {key: item for key, item in evidence.items() if key != "sha256"}
    if canonical_sha256(core) != declared_hash:
        raise ValueError("finalist competence evidence checksum mismatch")
    primary_action_hash = canonical_sha256(dict(primary.actual_action))
    if _canonical_sha256_digest(
        evidence["primaryActionHash"], field="finalist primary action hash"
    ) != primary_action_hash:
        raise ValueError("finalist competence primary action binding mismatch")
    cohort_values = evidence["cohortEvaluations"]
    if not isinstance(cohort_values, list) or len(cohort_values) != len(cohorts):
        raise ValueError("finalist competence cohort count mismatch")
    qualified_count = 0
    qualified_regrets: list[float] = []
    qualified_intervals: list[list[float]] = []
    for index, (raw_cohort, analysis) in enumerate(zip(cohort_values, cohorts)):
        if not isinstance(raw_cohort, Mapping):
            raise ValueError("finalist competence cohort must be an object")
        cohort = dict(raw_cohort)
        _require_exact_keys_v1(
            cohort,
            _COMPETENCE_COHORT_KEYS_V1,
            field="finalist competence cohort",
        )
        if cohort["cohortIndex"] != index or isinstance(
            cohort["cohortIndex"], bool
        ):
            raise ValueError("finalist competence cohort index mismatch")
        if _canonical_sha256_digest(
            cohort["evidenceHash"], field="finalist cohort evidence hash"
        ) != analysis.sha256:
            raise ValueError("finalist competence cohort evidence binding mismatch")
        trace_seed = cohort["traceSeed"]
        if trace_seed is not None and (
            isinstance(trace_seed, bool) or not isinstance(trace_seed, int)
        ):
            raise ValueError("finalist competence trace seed must be an integer or null")
        if _canonical_sha256_digest(
            cohort["bestActionHash"], field="finalist cohort best action hash"
        ) != canonical_sha256(dict(analysis.best_action)):
            raise ValueError("finalist competence best action binding mismatch")
        if _canonical_sha256_digest(
            cohort["primaryActionHash"], field="finalist cohort primary action hash"
        ) != primary_action_hash:
            raise ValueError("finalist competence cohort primary action mismatch")
        strict_reasons = _canonical_reason_list_v1(
            cohort["strictReasons"],
            allowed=_COMPETENCE_COHORT_STRICT_REASONS_V1,
            field="finalist competence cohort strict reasons",
        )
        reasons = _canonical_reason_list_v1(
            cohort["reasons"],
            allowed=_COMPETENCE_COHORT_REASONS_V1,
            field="finalist competence cohort reasons",
        )
        qualified = _require_bool_v1(
            cohort["qualified"], field="finalist competence qualified flag"
        )
        if qualified != (not reasons):
            raise ValueError("finalist competence qualified flag mismatch")
        for flag in (
            "matchWinMetricSupported",
            "budgetExhausted",
            "populationCoverageComplete",
            "exhaustiveExactSupport",
        ):
            _require_bool_v1(
                cohort[flag], field=f"finalist competence {flag}"
            )
        if cohort["solverCompletion"] not in _SOLVER_COMPLETION_VALUES_V2:
            raise ValueError("unsupported finalist competence solver completion")
        if cohort["truthClass"] not in _TRUTH_CLASS_VALUES_V2:
            raise ValueError("unsupported finalist competence truth class")
        score_evidence = analysis.primary_decision_score_evidence
        if score_evidence is None:
            raise ValueError(
                "finalist competence cohort lacks canonical score evidence"
            )
        if trace_seed != score_evidence.cohort_seed:
            raise ValueError("finalist competence trace seed binding mismatch")
        classification = canonical_score_support_classification_v1(
            score_evidence
        )
        if (
            cohort["budgetExhausted"]
            != bool(classification["budgetExhausted"])
            or cohort["populationCoverageComplete"]
            != bool(classification["populationCoverageComplete"])
            or cohort["solverCompletion"]
            != str(classification["solverCompletion"])
            or cohort["truthClass"] != str(classification["truthClass"])
            or cohort["exhaustiveExactSupport"]
            != bool(classification["exhaustiveExactSupport"])
        ):
            raise ValueError("finalist competence support projection mismatch")
        regret = cohort["expectedMarginRegret"]
        if regret is not None:
            regret = _finite_float_v1(
                regret, field="finalist competence expected-margin regret"
            )
            if regret < 0.0:
                raise ValueError("finalist competence regret must be non-negative")
        match_drop = cohort["matchWinDrop"]
        if match_drop is not None:
            match_drop = _finite_float_v1(
                match_drop, field="finalist competence match-win drop"
            )
            if match_drop < 0.0:
                raise ValueError("finalist competence match-win drop must be non-negative")
        if bool(cohort["matchWinMetricSupported"]) != (match_drop is not None):
            raise ValueError("finalist competence match-win support mismatch")
        interval = cohort["valueConfidence95"]
        normalized_interval: list[float] | None = None
        if interval is not None:
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError("finalist competence value interval is invalid")
            normalized_interval = [
                _finite_float_v1(
                    item, field="finalist competence value interval bound"
                )
                for item in interval
            ]
            if normalized_interval[0] > normalized_interval[1]:
                raise ValueError("finalist competence value interval is reversed")
        if qualified:
            if regret is None:
                raise ValueError("qualified finalist competence requires utility regret")
            qualified_count += 1
            qualified_regrets.append(regret)
            if normalized_interval is not None:
                qualified_intervals.append(normalized_interval)
        # Touch the validated strict reasons so an omitted list cannot be hidden
        # behind the competence-only verdict.
        assert isinstance(strict_reasons, list)
    agreement_count = evidence["agreementCount"]
    minimum_agreement = evidence["minimumAgreement"]
    if (
        isinstance(agreement_count, bool)
        or not isinstance(agreement_count, int)
        or agreement_count != qualified_count
    ):
        raise ValueError("finalist competence agreement count mismatch")
    if (
        isinstance(minimum_agreement, bool)
        or not isinstance(minimum_agreement, int)
        or not 2 <= minimum_agreement <= len(cohorts)
    ):
        raise ValueError("finalist competence minimum agreement is invalid")
    maximum_drift = _finite_float_v1(
        evidence["maximumExpectedMarginDrift"],
        field="finalist competence maximum expected-margin drift",
    )
    if maximum_drift < 0.0:
        raise ValueError("finalist competence maximum drift must be non-negative")
    expected_drift = (
        max(qualified_regrets) - min(qualified_regrets)
        if qualified_regrets else 0.0
    )
    if _finite_float_v1(
        evidence["expectedMarginRegretDrift"],
        field="finalist competence expected-margin regret drift",
    ) != expected_drift:
        raise ValueError("finalist competence expected-margin drift mismatch")
    intervals_stable = bool(
        len(qualified_intervals) < minimum_agreement
        or max(item[0] for item in qualified_intervals)
        <= min(item[1] for item in qualified_intervals)
    )
    if _require_bool_v1(
        evidence["valueIntervalsStable"],
        field="finalist competence interval-stability flag",
    ) != intervals_stable:
        raise ValueError("finalist competence interval-stability mismatch")
    resolution_reasons = _canonical_reason_list_v1(
        evidence["resolutionReasons"],
        allowed=_COMPETENCE_RESOLUTION_REASONS_V1,
        field="finalist competence resolution reasons",
        require_sorted=False,
    )
    expected_resolution: list[str] = []
    if qualified_count < minimum_agreement:
        expected_resolution.append("insufficient_competence_agreement")
    if expected_drift > maximum_drift:
        expected_resolution.append("competence_expected_margin_drift")
    if not intervals_stable:
        expected_resolution.append("competence_value_interval_instability")
    if resolution_reasons != expected_resolution:
        raise ValueError("finalist competence resolution reason mismatch")
    expected_status = (
        "primary_robust_competent" if not expected_resolution else "unresolved"
    )
    if status != expected_status:
        raise ValueError("finalist competence status mismatch")
    return evidence


def _validate_adjudication_evaluation_v2(
    raw: Mapping[str, Any],
    *,
    decisions_by_id: Mapping[str, TraceDecisionAnalysisV3],
    ledger_by_id: Mapping[str, DecisionLedgerRecordV1],
    adjudication_profile: Mapping[str, Any],
) -> tuple[str, str, str, str | None]:
    evaluation = dict(raw)
    _require_exact_keys_v1(
        evaluation,
        _ADJUDICATION_EVALUATION_KEYS_V2,
        field="finalist adjudication evaluation",
    )
    decision_id = str(evaluation["decisionId"])
    if not decision_id or decision_id not in decisions_by_id:
        raise ValueError("finalist adjudication evaluation decision is unknown")
    top_level = decisions_by_id[decision_id]
    ledger_record = ledger_by_id[decision_id]
    reanalysis_profile_hash = str(adjudication_profile["reanalysisProfileHash"])
    if evaluation["ordinal"] != top_level.ordinal or isinstance(
        evaluation["ordinal"], bool
    ):
        raise ValueError("finalist adjudication evaluation ordinal mismatch")
    status = str(evaluation["status"])
    if status not in _ADJUDICATION_STATUSES_V2:
        raise ValueError("unsupported finalist adjudication status")
    reasons = _canonical_reason_list_v1(
        evaluation["reasons"],
        allowed=_ADJUDICATION_REASONS_V2,
        field="finalist adjudication reasons",
    )
    strict_reasons = _canonical_reason_list_v1(
        evaluation["strictReasons"],
        allowed=_ADJUDICATION_STRICT_REASONS_V2,
        field="finalist adjudication strict reasons",
    )
    _canonical_reason_list_v1(
        evaluation["initialReasons"],
        allowed=_ADJUDICATION_STRICT_REASONS_V2,
        field="finalist adjudication initial reasons",
    )
    cohort_count = evaluation["cohortCount"]
    if (
        isinstance(cohort_count, bool)
        or not isinstance(cohort_count, int)
        or not 1 <= cohort_count <= 3
    ):
        raise ValueError("finalist adjudication cohort count is invalid")
    vector_fields = (
        "cohortAnalysisHashes",
        "cohortAnalyses",
        "cohortBestActionHashes",
        "cohortHiddenDeals",
        "cohortConfidence",
        "cohortExhaustiveExactSupport",
        "cohortPopulationCoverageComplete",
        "cohortSolverCompletion",
        "cohortTruthClass",
    )
    if any(
        not isinstance(evaluation[field], list)
        or len(evaluation[field]) != cohort_count
        for field in vector_fields
    ):
        raise ValueError("finalist adjudication cohort vector length mismatch")
    parsed_cohort_items: list[TraceDecisionAnalysisV3] = []
    for raw_cohort in evaluation["cohortAnalyses"]:
        if not isinstance(raw_cohort, Mapping):
            raise ValueError(
                "finalist adjudication cohort analysis must be an object"
            )
        parsed_cohort_items.append(TraceDecisionAnalysisV3.from_dict(raw_cohort))
    parsed_cohorts = tuple(parsed_cohort_items)
    if any(
        item.decision_id != decision_id
        or item.ordinal != top_level.ordinal
        or item.acting_seat != top_level.acting_seat
        or item.decision_kind != top_level.decision_kind
        or canonical_json_bytes_v1(item.actual_action)
        != canonical_json_bytes_v1(top_level.actual_action)
        for item in parsed_cohorts
    ):
        raise ValueError("finalist adjudication cohort decision binding mismatch")
    declared_hashes = [
        _canonical_sha256_digest(
            item, field="finalist adjudication cohort analysis hash"
        )
        for item in evaluation["cohortAnalysisHashes"]
    ]
    if declared_hashes != [item.sha256 for item in parsed_cohorts]:
        raise ValueError("finalist adjudication cohort analysis hash mismatch")
    if parsed_cohorts[0].to_dict() != top_level.to_dict():
        raise ValueError("finalist adjudication primary analysis was replaced")
    if _canonical_sha256_digest(
        evaluation["primaryAnalysisHash"],
        field="finalist adjudication primary analysis hash",
    ) != top_level.sha256:
        raise ValueError("finalist adjudication primary analysis hash mismatch")
    if not isinstance(evaluation["primaryAnalysis"], Mapping) or (
        TraceDecisionAnalysisV3.from_dict(evaluation["primaryAnalysis"]).to_dict()
        != top_level.to_dict()
    ):
        raise ValueError("finalist adjudication primary analysis binding mismatch")
    for field in ("selectedAnalysisHash", "retainedPrimaryAnalysisHash"):
        if _canonical_sha256_digest(
            evaluation[field], field=f"finalist adjudication {field}"
        ) != top_level.sha256:
            raise ValueError("finalist adjudication did not retain primary analysis")
    if not _require_bool_v1(
        evaluation["primaryAnalysisRetained"],
        field="finalist adjudication primary-retention flag",
    ):
        raise ValueError("finalist adjudication primary analysis must be retained")
    best_hashes = [canonical_sha256(dict(item.best_action)) for item in parsed_cohorts]
    if [
        _canonical_sha256_digest(
            item, field="finalist adjudication cohort best-action hash"
        )
        for item in evaluation["cohortBestActionHashes"]
    ] != best_hashes:
        raise ValueError("finalist adjudication cohort best-action hash mismatch")
    if _canonical_sha256_digest(
        evaluation["primaryBestActionHash"],
        field="finalist adjudication primary best-action hash",
    ) != best_hashes[0]:
        raise ValueError("finalist adjudication primary best-action hash mismatch")
    if evaluation["cohortHiddenDeals"] != [
        item.hidden_deals for item in parsed_cohorts
    ]:
        raise ValueError("finalist adjudication hidden-deal mirror mismatch")
    canonical_confidences = [
        canonical_score_confidence_v1(item.primary_decision_score_evidence)
        for item in parsed_cohorts
        if item.primary_decision_score_evidence is not None
    ]
    if (
        len(canonical_confidences) != cohort_count
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in evaluation["cohortConfidence"]
        )
        or [float(item) for item in evaluation["cohortConfidence"]]
        != canonical_confidences
    ):
        raise ValueError("finalist adjudication confidence mirror mismatch")
    for field in (
        "cohortExhaustiveExactSupport",
        "cohortPopulationCoverageComplete",
    ):
        if not all(isinstance(item, bool) for item in evaluation[field]):
            raise ValueError(f"finalist adjudication {field} must be boolean")
    if any(
        item not in _SOLVER_COMPLETION_VALUES_V2
        for item in evaluation["cohortSolverCompletion"]
    ):
        raise ValueError("unsupported finalist adjudication solver completion")
    if any(
        item not in _TRUTH_CLASS_VALUES_V2
        for item in evaluation["cohortTruthClass"]
    ):
        raise ValueError("unsupported finalist adjudication truth class")
    classifications: list[dict[str, Any]] = []
    for item in parsed_cohorts:
        if item.primary_decision_score_evidence is None:
            raise ValueError(
                "finalist adjudication cohort lacks canonical score evidence"
            )
        classifications.append(
            canonical_score_support_classification_v1(
                item.primary_decision_score_evidence
            )
        )
    primary_score_evidence = parsed_cohorts[0].primary_decision_score_evidence
    assert primary_score_evidence is not None
    if (
        primary_score_evidence.cohort_seed_policy
        != PRIMARY_DECISION_COHORT_SEED_POLICY_V1
        or primary_score_evidence.cohort_index != 0
    ):
        raise ValueError("finalist adjudication primary cohort role mismatch")
    if primary_score_evidence.cohort_seed != _decision_seed(
        ledger_record,
        primary_score_evidence.analysis_identity_hash,
        seed_namespace="",
        seed_offset=0,
    ):
        raise ValueError("finalist adjudication primary cohort seed mismatch")
    reanalysis_identities: set[str] = set()
    cohort_seeds = {primary_score_evidence.cohort_seed}
    cohort_analysis_hashes = {parsed_cohorts[0].sha256}
    for cohort_position, item in enumerate(parsed_cohorts[1:], start=1):
        score_evidence = item.primary_decision_score_evidence
        assert score_evidence is not None
        if score_evidence.trace_analysis_profile_hash != reanalysis_profile_hash:
            raise ValueError("finalist adjudication reanalysis profile mismatch")
        if (
            score_evidence.cohort_seed_policy
            != PRIMARY_DECISION_COHORT_SEED_POLICY_V1
            or score_evidence.cohort_index != 1
        ):
            raise ValueError("finalist adjudication reanalysis cohort role mismatch")
        seed_namespace = (
            "finalist_surrounding_adjudication_v2"
            if cohort_position == 1
            else "finalist_surrounding_adjudication_v2:cohort3"
        )
        seed_offset = int(adjudication_profile["seedOffset"]) * cohort_position
        if score_evidence.cohort_seed != _decision_seed(
            ledger_record,
            score_evidence.analysis_identity_hash,
            seed_namespace=seed_namespace,
            seed_offset=seed_offset,
        ):
            raise ValueError("finalist adjudication reanalysis cohort seed mismatch")
        item.validate_ledger_binding(
            ledger_record,
            trace_analysis_profile_hash=reanalysis_profile_hash,
            rules_hash=primary_score_evidence.rules_hash,
        )
        if (
            score_evidence.public_prefix_hash
            != primary_score_evidence.public_prefix_hash
            or score_evidence.legal_action_set_hash
            != primary_score_evidence.legal_action_set_hash
            or score_evidence.objective != primary_score_evidence.objective
            or score_evidence.tie_break != primary_score_evidence.tie_break
            or score_evidence.backend != primary_score_evidence.backend
        ):
            raise ValueError("finalist adjudication cohort authority drift")
        if score_evidence.cohort_seed in cohort_seeds or item.sha256 in (
            cohort_analysis_hashes
        ):
            raise ValueError("finalist adjudication cohort evidence is not distinct")
        cohort_seeds.add(score_evidence.cohort_seed)
        cohort_analysis_hashes.add(item.sha256)
        reanalysis_identities.add(score_evidence.analysis_identity_hash)
    if len(reanalysis_identities) > 1:
        raise ValueError("finalist adjudication reanalysis identity drift")
    reanalysis_identity = (
        next(iter(reanalysis_identities)) if reanalysis_identities else None
    )
    if reanalysis_identity == primary_score_evidence.analysis_identity_hash:
        raise ValueError("finalist adjudication lane identities must be distinct")
    if evaluation["cohortExhaustiveExactSupport"] != [
        bool(item["exhaustiveExactSupport"]) for item in classifications
    ]:
        raise ValueError("finalist adjudication exact-support mirror mismatch")
    if evaluation["cohortPopulationCoverageComplete"] != [
        bool(item["populationCoverageComplete"]) for item in classifications
    ]:
        raise ValueError("finalist adjudication population-coverage mirror mismatch")
    if evaluation["cohortSolverCompletion"] != [
        str(item["solverCompletion"]) for item in classifications
    ]:
        raise ValueError("finalist adjudication solver-completion mirror mismatch")
    if evaluation["cohortTruthClass"] != [
        str(item["truthClass"]) for item in classifications
    ]:
        raise ValueError("finalist adjudication truth-class mirror mismatch")
    if evaluation["primaryHiddenDeals"] != parsed_cohorts[0].hidden_deals:
        raise ValueError("finalist adjudication primary hidden-deal mirror mismatch")
    if _finite_float_v1(
        evaluation["primaryConfidence"],
        field="finalist adjudication primary confidence",
    ) != canonical_confidences[0]:
        raise ValueError("finalist adjudication primary confidence mismatch")
    if evaluation["primaryExhaustiveExactSupport"] != evaluation[
        "cohortExhaustiveExactSupport"
    ][0] or not isinstance(evaluation["primaryExhaustiveExactSupport"], bool):
        raise ValueError("finalist adjudication primary exact-support mismatch")
    overflow = strict_reasons == ["adjudication_bound_exceeded"]
    if overflow:
        if (
            cohort_count != 1
            or status != "unresolved"
            or reasons != ["adjudication_bound_exceeded"]
            or evaluation["competenceEvidence"] is not None
            or evaluation["reanalysis"] is not None
            or evaluation["reanalysisHash"] is not None
            or evaluation["reanalysisBestActionHash"] is not None
            or evaluation["reanalysisHiddenDeals"] is not None
            or evaluation["reanalysisConfidence"] is not None
            or evaluation["reanalysisExhaustiveExactSupport"] is not None
            or evaluation["agreementCount"] != 0
            or evaluation["majorityActionHash"] is not None
            or evaluation["expectedMarginDrift"] is not None
            or evaluation["initialReasons"] != []
        ):
            raise ValueError("finalist adjudication overflow evidence is invalid")
    else:
        if cohort_count not in {2, 3}:
            raise ValueError("finalist adjudication requires two or three cohorts")
        reanalysis = parsed_cohorts[1]
        if _canonical_sha256_digest(
            evaluation["reanalysisHash"],
            field="finalist adjudication reanalysis hash",
        ) != reanalysis.sha256:
            raise ValueError("finalist adjudication reanalysis hash mismatch")
        if not isinstance(evaluation["reanalysis"], Mapping) or (
            TraceDecisionAnalysisV3.from_dict(evaluation["reanalysis"]).to_dict()
            != reanalysis.to_dict()
        ):
            raise ValueError("finalist adjudication reanalysis binding mismatch")
        if _canonical_sha256_digest(
            evaluation["reanalysisBestActionHash"],
            field="finalist adjudication reanalysis best-action hash",
        ) != best_hashes[1]:
            raise ValueError("finalist adjudication reanalysis best-action mismatch")
        if evaluation["reanalysisHiddenDeals"] != reanalysis.hidden_deals:
            raise ValueError("finalist adjudication reanalysis hidden-deal mismatch")
        if _finite_float_v1(
            evaluation["reanalysisConfidence"],
            field="finalist adjudication reanalysis confidence",
        ) != canonical_confidences[1]:
            raise ValueError("finalist adjudication reanalysis confidence mismatch")
        if evaluation["reanalysisExhaustiveExactSupport"] != evaluation[
            "cohortExhaustiveExactSupport"
        ][1] or not isinstance(
            evaluation["reanalysisExhaustiveExactSupport"], bool
        ):
            raise ValueError("finalist adjudication reanalysis exact-support mismatch")
    agreement_count = evaluation["agreementCount"]
    if (
        isinstance(agreement_count, bool)
        or not isinstance(agreement_count, int)
        or not 0 <= agreement_count <= cohort_count
    ):
        raise ValueError("finalist adjudication agreement count is invalid")
    supporter_indices = evaluation["supporterCohortIndices"]
    if (
        not isinstance(supporter_indices, list)
        or len(supporter_indices) != agreement_count
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or not 0 <= item < cohort_count
            for item in supporter_indices
        )
        or len(supporter_indices) != len(set(supporter_indices))
    ):
        raise ValueError("finalist adjudication supporter quorum is invalid")
    majority_hash = evaluation["majorityActionHash"]
    if majority_hash is not None:
        majority_hash = _canonical_sha256_digest(
            majority_hash, field="finalist adjudication majority action hash"
        )
    drift = evaluation["expectedMarginDrift"]
    if drift is not None and _finite_float_v1(
        drift, field="finalist adjudication expected-margin drift"
    ) < 0.0:
        raise ValueError("finalist adjudication margin drift must be non-negative")
    featured = _require_bool_v1(
        evaluation["featuredTruthEligible"],
        field="finalist adjudication featured-truth flag",
    )
    surrounding = _require_bool_v1(
        evaluation["surroundingVerified"],
        field="finalist adjudication surrounding-verification flag",
    )
    strict_primary = _require_bool_v1(
        evaluation["strictPrimaryActionConfirmed"],
        field="finalist adjudication strict-primary flag",
    )
    competence_raw = evaluation["competenceEvidence"]
    competence: dict[str, Any] | None = None
    if competence_raw is not None:
        if not isinstance(competence_raw, Mapping):
            raise ValueError("finalist competence evidence must be an object")
        competence = _validate_competence_evidence_v1(
            competence_raw,
            cohorts=parsed_cohorts,
            primary=top_level,
        )
    recommendation_status = evaluation["strictRecommendationStatus"]
    if status == "strict_confirmed":
        if (
            competence is not None
            or reasons
            or strict_reasons
            or recommendation_status not in {
                "confirmed_primary", "confirmed_mistake"
            }
            or strict_primary != (recommendation_status == "confirmed_primary")
            or not featured
            or not surrounding
            or agreement_count < 2
            or majority_hash not in set(best_hashes)
        ):
            raise ValueError("strict finalist adjudication verdict is inconsistent")
        actual_hash = canonical_sha256(dict(top_level.actual_action))
        if strict_primary and majority_hash != actual_hash:
            raise ValueError("strict finalist primary action was not confirmed")
        if not strict_primary and majority_hash == actual_hash:
            raise ValueError("strict finalist mistake classification is inconsistent")
    elif status == "primary_robust_competent":
        if (
            competence is None
            or competence["status"] != "primary_robust_competent"
            or reasons
            or not strict_reasons
            or recommendation_status is not None
            or strict_primary
            or featured
            or not surrounding
            or majority_hash != canonical_sha256(dict(top_level.actual_action))
            or agreement_count != competence["agreementCount"]
        ):
            raise ValueError("robust finalist competence verdict is inconsistent")
    else:
        if (
            not reasons
            or recommendation_status is not None
            or strict_primary
            or featured
            or surrounding
        ):
            raise ValueError("unresolved finalist adjudication verdict is inconsistent")
        if not overflow:
            if (
                not strict_reasons
                or competence is None
                or competence["status"] != "unresolved"
            ):
                raise ValueError("unresolved finalist competence evidence is missing")
            expected_reasons = set(strict_reasons)
            expected_reasons.update(competence["resolutionReasons"])
            for cohort in competence["cohortEvaluations"]:
                expected_reasons.update(cohort["reasons"])
            if set(reasons) != expected_reasons:
                raise ValueError("unresolved finalist reason projection mismatch")
    if not overflow:
        normalized_cohorts = tuple(
            canonical_finalist_cohort_v1(
                item.primary_decision_score_evidence,
                cohort_index=index,
                evidence_hash=item.sha256,
                actual_action=item.actual_action,
                best_action=item.best_action,
            )
            for index, item in enumerate(parsed_cohorts)
        )
        pair_projection = project_finalist_adjudication_v2(
            normalized_cohorts[:2],
            primary_best_action=top_level.best_action,
            primary_actual_action=top_level.actual_action,
            adjudication_profile=adjudication_profile,
        )
        pair_requires_third = bool(pair_projection["strictReasons"]) and (
            int(adjudication_profile["maximumCohorts"]) >= 3
            and bool(adjudication_profile["thirdCohortOnDisagreement"])
        )
        if cohort_count != (3 if pair_requires_third else 2):
            raise ValueError(
                "finalist adjudication third-cohort coverage does not match policy"
            )
        expected_projection = project_finalist_adjudication_v2(
            normalized_cohorts,
            primary_best_action=top_level.best_action,
            primary_actual_action=top_level.actual_action,
            adjudication_profile=adjudication_profile,
            initial_reasons=pair_projection["strictReasons"],
        )
        if any(
            evaluation[field] != expected_value
            for field, expected_value in expected_projection.items()
        ):
            raise ValueError(
                "finalist adjudication verdict does not match canonical score rows"
            )
    return (
        decision_id,
        status,
        primary_score_evidence.analysis_identity_hash,
        reanalysis_identity,
    )


def _validate_adjudication_evidence_v2(
    raw: Mapping[str, Any],
    *,
    decisions: tuple[TraceDecisionAnalysisV3, ...],
    ledger: DecisionLedgerV1,
    analyzer_id: str,
    analyzer_version: str,
    effective_profile_hash: str,
    screening_profile_hash: str,
    confirmation_profile_hash: str,
    confirmed_decision_ids: tuple[str, ...],
) -> None:
    evidence = dict(raw)
    _require_exact_keys_v1(
        evidence,
        _ADJUDICATION_EVIDENCE_KEYS_V2,
        field="finalist adjudication evidence",
    )
    if analyzer_id != "adaptive_finalist_trace_analyzer_v1" or (
        analyzer_version != "1.3.0"
    ):
        raise ValueError("adjudication-v2 requires adaptive finalist analyzer 1.3.0")
    if (
        evidence["schemaVersion"] != 2
        or isinstance(evidence["schemaVersion"], bool)
        or evidence["verdictSchemaVersion"] != 2
        or isinstance(evidence["verdictSchemaVersion"], bool)
    ):
        raise ValueError("unsupported finalist adjudication evidence schema")
    for field in ("profileHash", "primaryProfileHash", "reanalysisProfileHash"):
        _canonical_sha256_digest(
            evidence[field], field=f"finalist adjudication {field}"
        )
    raw_primary_profile = evidence["primaryProfile"]
    if not isinstance(raw_primary_profile, Mapping):
        raise ValueError("finalist primary profile snapshot must be an object")
    primary_profile = CascadedTraceAnalysisProfileV1.from_mapping(
        raw_primary_profile
    )
    if (
        primary_profile.to_dict() != dict(raw_primary_profile)
        or evidence["primaryProfileHash"] != primary_profile.sha256
        or primary_profile.screening_profile_hash != screening_profile_hash
        or primary_profile.confirmation_profile_hash != confirmation_profile_hash
    ):
        raise ValueError("finalist primary profile snapshot binding mismatch")
    if len(confirmed_decision_ids) > (
        primary_profile.maximum_confirmation_decisions_per_trace
    ):
        raise ValueError(
            "confirmed decisions exceed the reconstructed primary profile bound"
        )
    raw_profile = evidence["adjudicationProfile"]
    if not isinstance(raw_profile, Mapping):
        raise ValueError("finalist adjudication profile snapshot must be an object")
    adjudication_profile = normalize_finalist_adjudication_profile_v2(raw_profile)
    if dict(raw_profile) != adjudication_profile:
        raise ValueError("finalist adjudication profile snapshot is not normalized")
    if (
        evidence["profileHash"] != adjudication_profile["sha256"]
        or evidence["primaryProfileHash"]
        != adjudication_profile["primaryProfileHash"]
        or evidence["reanalysisProfileHash"]
        != adjudication_profile["reanalysisProfileHash"]
    ):
        raise ValueError("finalist adjudication profile snapshot binding mismatch")
    raw_selection = evidence["selectionProfile"]
    if not isinstance(raw_selection, Mapping):
        raise ValueError("finalist selection profile snapshot must be an object")
    selection_value = dict(raw_selection)
    declared_selection_hash = _canonical_sha256_digest(
        selection_value.pop("sha256", ""),
        field="finalist selection profile sha256",
    )
    selection_profile = SelectionProfileV1.build(selection_value)
    if (
        selection_profile.sha256 != declared_selection_hash
        or canonical_json_bytes_v1(selection_profile.to_dict())
        != canonical_json_bytes_v1(dict(raw_selection))
        or float(adjudication_profile["minimumConfidence"])
        != selection_profile.minimum_surrounding_confidence
        or float(adjudication_profile["minimumQuality"])
        != selection_profile.minimum_surrounding_quality
    ):
        raise ValueError("finalist selection profile snapshot binding mismatch")
    expected_effective_profile_hash = canonical_sha256({
        "schemaVersion": 2,
        "primaryProfileHash": evidence["primaryProfileHash"],
        "adjudicationProfileHash": evidence["profileHash"],
        "selectionProfileHash": selection_profile.sha256,
    })
    if effective_profile_hash != expected_effective_profile_hash:
        raise ValueError("finalist effective profile binding mismatch")
    if evidence["seedNamespace"] != "finalist_surrounding_adjudication_v2" or (
        evidence["thirdSeedNamespace"]
        != "finalist_surrounding_adjudication_v2:cohort3"
    ):
        raise ValueError("unsupported finalist adjudication seed namespace")
    seed_offset = evidence["seedOffset"]
    third_seed_offset = evidence["thirdSeedOffset"]
    if (
        isinstance(seed_offset, bool)
        or not isinstance(seed_offset, int)
        or seed_offset <= 0
        or isinstance(third_seed_offset, bool)
        or not isinstance(third_seed_offset, int)
        or third_seed_offset != seed_offset * 2
        or seed_offset != adjudication_profile["seedOffset"]
    ):
        raise ValueError("finalist adjudication seed offsets are inconsistent")
    known = {item.decision_id for item in decisions}

    decisions_by_id = {item.decision_id: item for item in decisions}
    ledger_by_id = {item.decision_id: item for item in ledger.records}

    def ordered_target_ids(field: str) -> list[str]:
        raw_ids = evidence[field]
        if not isinstance(raw_ids, list) or not all(
            isinstance(item, str) and item for item in raw_ids
        ):
            raise ValueError(f"finalist adjudication {field} must contain decision ids")
        if len(raw_ids) != len(set(raw_ids)) or not set(raw_ids).issubset(known):
            raise ValueError(f"finalist adjudication {field} identities are invalid")
        expected = sorted(
            raw_ids,
            key=lambda item: (decisions_by_id[item].ordinal, item),
        )
        if raw_ids != expected:
            raise ValueError(f"finalist adjudication {field} order is not canonical")
        return list(raw_ids)

    target_ids = ordered_target_ids("targetDecisionIds")
    bounded_ids = ordered_target_ids("boundedDecisionIds")
    overflow_ids = ordered_target_ids("overflowDecisionIds")
    third_ids = ordered_target_ids("thirdCohortDecisionIds")
    expected_target_ids = [
        item.decision_id
        for item in sorted(decisions, key=lambda value: (value.ordinal, value.decision_id))
        if finalist_adjudication_target_required_v2(
            item,
            adjudication_profile=adjudication_profile,
            selection_profile=selection_profile.to_dict(),
        )
    ]
    if target_ids != expected_target_ids:
        raise ValueError("finalist adjudication target coverage is incomplete")
    maximum_decisions = int(adjudication_profile["maximumDecisionsPerTrace"])
    if (
        target_ids != [*bounded_ids, *overflow_ids]
        or bounded_ids != target_ids[:maximum_decisions]
        or overflow_ids != target_ids[maximum_decisions:]
        or not set(third_ids).issubset(bounded_ids)
        or (
            int(adjudication_profile["maximumCohorts"]) < 3
            and bool(third_ids)
        )
    ):
        raise ValueError("finalist adjudication target coverage is inconsistent")
    unresolved = _canonical_decision_id_list_v1(
        evidence["unresolvedDecisionIds"],
        field="finalist adjudication unresolved decisions",
    )
    competent = _canonical_decision_id_list_v1(
        evidence["competentDecisionIds"],
        field="finalist adjudication competent decisions",
    )
    if not set(unresolved).issubset(known) or not set(competent).issubset(known):
        raise ValueError("finalist adjudication verdict decision is unknown")
    if set(unresolved) & set(competent):
        raise ValueError(
            "finalist adjudication unresolved precedence was not applied"
        )
    evaluations_raw = evidence["evaluations"]
    if not isinstance(evaluations_raw, list) or not all(
        isinstance(item, Mapping) for item in evaluations_raw
    ):
        raise ValueError("finalist adjudication evaluations must be objects")
    evaluations = [
        _validate_adjudication_evaluation_v2(
            item,
            decisions_by_id=decisions_by_id,
            ledger_by_id=ledger_by_id,
            adjudication_profile=adjudication_profile,
        )
        for item in evaluations_raw
    ]
    evaluation_ids = [item[0] for item in evaluations]
    if len(evaluation_ids) != len(set(evaluation_ids)):
        raise ValueError("finalist adjudication evaluation decisions are not unique")
    if evaluation_ids != target_ids:
        raise ValueError("finalist adjudication evaluation coverage is incomplete")
    evaluation_values = [dict(item) for item in evaluations_raw]
    for decision_id, evaluation_value in zip(
        evaluation_ids, evaluation_values, strict=True
    ):
        expected_count = (
            1
            if decision_id in overflow_ids
            else (3 if decision_id in third_ids else 2)
        )
        if evaluation_value["cohortCount"] != expected_count:
            raise ValueError("finalist adjudication cohort target coverage mismatch")
    primary_identities_by_stage: dict[str, set[str]] = {}
    for decision_id, _status, primary_identity, _reanalysis_identity in evaluations:
        stage = (
            "confirmation"
            if decision_id in confirmed_decision_ids
            else "screening"
        )
        primary_identities_by_stage.setdefault(stage, set()).add(primary_identity)
    if any(
        len(identities) > 1
        for identities in primary_identities_by_stage.values()
    ):
        raise ValueError(
            "finalist adjudication receipt primary stage identity drift"
        )
    reanalysis_identities = {
        item[3] for item in evaluations if item[3] is not None
    }
    if len(reanalysis_identities) > 1:
        raise ValueError("finalist adjudication receipt reanalysis identity drift")
    unresolved_from_status = sorted(
        decision_id
        for decision_id, status, _primary_identity, _reanalysis_identity in evaluations
        if status == "unresolved"
    )
    competent_from_status = sorted(
        decision_id
        for decision_id, status, _primary_identity, _reanalysis_identity in evaluations
        if status == "primary_robust_competent"
    )
    if unresolved != unresolved_from_status:
        raise ValueError("finalist adjudication unresolved status map mismatch")
    if competent != competent_from_status:
        raise ValueError("finalist adjudication competence status map mismatch")


@dataclass(frozen=True, slots=True)
class TraceAnalysisResultV1:
    trace_hash: str
    profile_hash: str
    analyzer_id: str
    analyzer_version: str
    decisions: tuple[TraceDecisionAnalysisV3, ...]
    laffa_evidence: LaffaEvidenceV2
    sha256: str
    confirmed_decision_ids: tuple[str, ...] = ()
    screening_profile_hash: str | None = None
    confirmation_profile_hash: str | None = None
    adjudication_evidence: Mapping[str, Any] | None = None
    unresolved_decision_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        cascaded = self.screening_profile_hash is not None
        adjudicated = self.adjudication_evidence is not None
        value = {
            "schemaVersion": 3 if adjudicated else (2 if cascaded else 1),
            "traceHash": self.trace_hash,
            "profileHash": self.profile_hash,
            "analyzerId": self.analyzer_id,
            "analyzerVersion": self.analyzer_version,
            "decisions": [item.to_dict() for item in self.decisions],
            "laffaEvidence": self.laffa_evidence.to_dict(),
            "sha256": self.sha256,
        }
        if cascaded:
            value.update({
                "confirmedDecisionIds": list(self.confirmed_decision_ids),
                "screeningProfileHash": self.screening_profile_hash,
                "confirmationProfileHash": self.confirmation_profile_hash,
            })
        if adjudicated:
            value.update({
                "adjudicationEvidence": dict(self.adjudication_evidence or {}),
                "unresolvedDecisionIds": list(self.unresolved_decision_ids),
            })
        return value

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        ledger: DecisionLedgerV1,
        expected_adaptive_profile_hash: str | None = None,
    ) -> "TraceAnalysisResultV1":
        schema_raw = value.get("schemaVersion")
        if isinstance(schema_raw, bool) or not isinstance(schema_raw, int):
            raise ValueError("unsupported trace analysis result schema")
        schema_version = schema_raw
        expected_keys = {
            1: _TRACE_ANALYSIS_RESULT_KEYS_V1,
            2: _TRACE_ANALYSIS_RESULT_KEYS_V2,
            3: _TRACE_ANALYSIS_RESULT_KEYS_V3,
        }.get(schema_version)
        if expected_keys is None:
            raise ValueError("unsupported trace analysis result schema")
        _require_exact_keys_v1(
            value,
            expected_keys,
            field="trace analysis result",
        )
        for field in ("traceHash", "profileHash", "sha256"):
            if not isinstance(value[field], str):
                raise ValueError(
                    f"trace analysis result {field} must be a string"
                )
            _canonical_sha256_digest(
                value[field], field=f"trace analysis result {field}"
            )
        if not isinstance(value["analyzerId"], str) or not value["analyzerId"]:
            raise ValueError("trace analysis analyzer id must be a non-empty string")
        if not isinstance(value["analyzerVersion"], str) or not value["analyzerVersion"]:
            raise ValueError(
                "trace analysis analyzer version must be a non-empty string"
            )
        analyzer_id = str(value.get("analyzerId") or "")
        analyzer_version = str(value.get("analyzerVersion") or "")
        current_adaptive = (
            analyzer_id == "adaptive_finalist_trace_analyzer_v1"
            and _semantic_version_v1(analyzer_version) >= (1, 3, 0)
        )
        if (
            current_adaptive
            and schema_version != 3
        ):
            raise ValueError(
                "adaptive finalist analyzer 1.3 requires adjudication-v2 evidence"
            )
        if current_adaptive:
            if expected_adaptive_profile_hash is None:
                raise ValueError(
                    "current adaptive finalist analysis requires an expected "
                    "effective profile authority"
                )
            if not isinstance(expected_adaptive_profile_hash, str):
                raise ValueError(
                    "expected adaptive finalist profile hash must be a string"
                )
            expected_profile_hash = _canonical_sha256_digest(
                expected_adaptive_profile_hash,
                field="expected adaptive finalist profile hash",
            )
            if expected_profile_hash != value["profileHash"]:
                raise ValueError(
                    "adaptive finalist effective profile does not match expected "
                    "authority"
                )
        decisions_raw = value["decisions"]
        if not isinstance(decisions_raw, list) or not all(
            isinstance(item, Mapping) for item in decisions_raw
        ):
            raise ValueError("trace analysis decisions must be a list of objects")
        decisions = tuple(
            TraceDecisionAnalysisV3.from_dict(item) for item in decisions_raw
        )
        decision_ids = [item.decision_id for item in decisions]
        ledger_ids = [item.decision_id for item in ledger.records]
        if decision_ids != ledger_ids:
            raise ValueError(
                "trace analysis does not preserve the exact canonical decision ledger"
            )
        if not isinstance(value["laffaEvidence"], Mapping):
            raise ValueError("trace analysis laffa evidence must be an object")
        laffa = LaffaEvidenceV2.from_dict(
            dict(value["laffaEvidence"]),
            ledger=ledger,
        )
        core = {
            "schemaVersion": schema_version,
            "traceHash": str(value.get("traceHash") or ""),
            "profileHash": str(value.get("profileHash") or ""),
            "analyzerId": str(value.get("analyzerId") or ""),
            "analyzerVersion": str(value.get("analyzerVersion") or ""),
            "decisionAnalysisHashes": [item.sha256 for item in decisions],
            "laffaEvidenceHash": laffa.sha256,
        }
        confirmed: tuple[str, ...] = ()
        screening_profile_hash = (
            str(value.get("screeningProfileHash") or "") or None
        )
        confirmation_profile_hash = (
            str(value.get("confirmationProfileHash") or "") or None
        )
        if schema_version >= 2:
            confirmed = tuple(_canonical_decision_id_list_v1(
                value["confirmedDecisionIds"],
                field="trace analysis confirmed decisions",
            ))
            if not screening_profile_hash or not confirmation_profile_hash:
                raise ValueError("cascaded trace analysis requires both stage profiles")
            if not isinstance(value["screeningProfileHash"], str) or not isinstance(
                value["confirmationProfileHash"], str
            ):
                raise ValueError("cascaded trace analysis profile hashes must be strings")
            _canonical_sha256_digest(
                screening_profile_hash,
                field="trace analysis screening profile hash",
            )
            _canonical_sha256_digest(
                confirmation_profile_hash,
                field="trace analysis confirmation profile hash",
            )
            by_id = {item.decision_id: item for item in decisions}
            known = set(by_id)
            if not set(confirmed).issubset(known):
                raise ValueError("confirmed decisions must belong to the trace analysis")
            if any(by_id[decision_id].forced for decision_id in confirmed):
                raise ValueError("confirmed decisions must be non-forced")
            core.update({
                "confirmedDecisionIds": list(confirmed),
                "screeningProfileHash": screening_profile_hash,
                "confirmationProfileHash": confirmation_profile_hash,
            })
        records = {item.decision_id: item for item in ledger.records}
        rules_hashes: set[str] = set()
        primary_identities_by_stage: dict[str, set[str]] = {}
        for item in decisions:
            expected_profile_hash = core["profileHash"]
            if schema_version >= 2:
                expected_profile_hash = (
                    confirmation_profile_hash
                    if item.decision_id in confirmed
                    else screening_profile_hash
                )
            assert expected_profile_hash is not None
            item.validate_ledger_binding(
                records[item.decision_id],
                trace_analysis_profile_hash=expected_profile_hash,
            )
            if item.primary_decision_score_evidence is not None:
                score_evidence = item.primary_decision_score_evidence
                rules_hashes.add(score_evidence.rules_hash)
                if (
                    score_evidence.cohort_seed_policy
                    != PRIMARY_DECISION_COHORT_SEED_POLICY_V1
                    or score_evidence.cohort_index != 0
                ):
                    raise ValueError(
                        "trace decision primary cohort role mismatch"
                    )
                if score_evidence.cohort_seed != _decision_seed(
                    records[item.decision_id],
                    score_evidence.analysis_identity_hash,
                    seed_namespace="",
                    seed_offset=0,
                ):
                    raise ValueError(
                        "trace decision primary cohort seed mismatch"
                    )
                stage = (
                    "confirmation"
                    if schema_version >= 2 and item.decision_id in confirmed
                    else ("screening" if schema_version >= 2 else "primary")
                )
                primary_identities_by_stage.setdefault(stage, set()).add(
                    score_evidence.analysis_identity_hash
                )
            if item.forced_legality_proof is not None:
                rules_hashes.add(str(item.forced_legality_proof["rulesHash"]))
        if len(rules_hashes) != 1:
            raise ValueError(
                "trace decision analysis must retain one identical rules authority"
            )
        if any(
            len(identities) > 1
            for identities in primary_identities_by_stage.values()
        ):
            raise ValueError("trace decision primary stage identity drift")
        adjudication_evidence = None
        unresolved_decision_ids: tuple[str, ...] = ()
        if schema_version == 3:
            if not isinstance(value["adjudicationEvidence"], Mapping):
                raise ValueError("trace analysis adjudication evidence must be an object")
            unresolved_decision_ids = tuple(
                _canonical_decision_id_list_v1(
                    value["unresolvedDecisionIds"],
                    field="trace analysis unresolved adjudication decisions",
                )
            )
            adjudication_evidence = dict(value["adjudicationEvidence"])
            declared_adjudication_hash = _canonical_sha256_digest(
                adjudication_evidence.get("sha256") or "",
                field="trace analysis adjudication sha256",
            )
            adjudication_core = {
                key: item
                for key, item in adjudication_evidence.items()
                if key != "sha256"
            }
            if canonical_sha256(adjudication_core) != declared_adjudication_hash:
                raise ValueError("trace analysis adjudication checksum mismatch")
            evidence_schema = adjudication_evidence.get("schemaVersion")
            if isinstance(evidence_schema, bool) or not isinstance(
                evidence_schema, int
            ):
                raise ValueError("unsupported finalist adjudication evidence schema")
            if evidence_schema == 2:
                _validate_adjudication_evidence_v2(
                    adjudication_evidence,
                    decisions=decisions,
                    ledger=ledger,
                    analyzer_id=core["analyzerId"],
                    analyzer_version=core["analyzerVersion"],
                    effective_profile_hash=core["profileHash"],
                    screening_profile_hash=screening_profile_hash,
                    confirmation_profile_hash=confirmation_profile_hash,
                    confirmed_decision_ids=confirmed,
                )
            elif evidence_schema == 1:
                if core["analyzerId"] != "adaptive_finalist_trace_analyzer_v1" or (
                    _semantic_version_v1(core["analyzerVersion"])
                    >= (1, 3, 0)
                ):
                    raise ValueError(
                        "historical adjudication-v1 evidence cannot be interpreted "
                        "by the current finalist analyzer"
                    )
                legacy_competent_ids = adjudication_evidence.get(
                    "competentDecisionIds"
                ) or ()
                legacy_evaluations = adjudication_evidence.get("evaluations") or ()
                if (
                    legacy_competent_ids
                    or any(
                        isinstance(item, Mapping)
                        and (
                            item.get("status") == "primary_robust_competent"
                            or item.get("featuredTruthEligible") is True
                            or (
                                isinstance(item.get("competenceEvidence"), Mapping)
                                and (
                                    item["competenceEvidence"].get("status")
                                    == "primary_robust_competent"
                                    or item["competenceEvidence"].get("qualified")
                                    is True
                                    or any(
                                        isinstance(cohort, Mapping)
                                        and cohort.get("qualified") is True
                                        for cohort in item[
                                            "competenceEvidence"
                                        ].get("cohortEvaluations")
                                        or ()
                                    )
                                )
                            )
                        )
                        for item in legacy_evaluations
                    )
                ):
                    raise ValueError(
                        "historical adjudication-v1 evidence cannot carry current "
                        "competent qualification semantics"
                    )
                historical_unresolved = tuple(sorted(
                    item.decision_id for item in decisions if not item.forced
                ))
                if unresolved_decision_ids != historical_unresolved:
                    raise ValueError(
                        "historical finalist evidence must remain fail-closed"
                    )
            else:
                raise ValueError("unsupported finalist adjudication evidence schema")
            known = {item.decision_id for item in decisions}
            if not set(unresolved_decision_ids).issubset(known):
                raise ValueError("unresolved adjudication decisions must belong to the trace")
            evidence_unresolved = tuple(sorted(
                str(item)
                for item in adjudication_evidence.get("unresolvedDecisionIds") or ()
            ))
            if unresolved_decision_ids != evidence_unresolved:
                raise ValueError("trace analysis unresolved adjudication identity mismatch")
            core.update({
                "adjudicationEvidenceHash": declared_adjudication_hash,
                "unresolvedDecisionIds": list(unresolved_decision_ids),
            })
        elif (
            analyzer_id == "adaptive_finalist_trace_analyzer_v1"
            and _semantic_version_v1(analyzer_version) < (1, 3, 0)
        ):
            unresolved_decision_ids = tuple(sorted(
                item.decision_id for item in decisions if not item.forced
            ))
        digest = canonical_sha256(core)
        if str(value.get("sha256") or "") != digest:
            raise ValueError("trace analysis result checksum mismatch")
        return cls(
            core["traceHash"], core["profileHash"], core["analyzerId"],
            core["analyzerVersion"], decisions, laffa, digest, confirmed,
            screening_profile_hash, confirmation_profile_hash,
            adjudication_evidence, unresolved_decision_ids,
        )


_PROCESS_ANALYZER: Any = None
_PROCESS_CANCEL_EVENT: Any = None
_SECONDARY_DECISION_COHORT_SALT = "secondary-decision-policy:v1"
_SECONDARY_COHORT_AUTHORITY_POLICY = "trace_analysis_secondary_cohort_v1"


def _decision_seed_and_index(
    record: DecisionLedgerRecordV1,
    cohort_authority_hash: str,
    *,
    seed_namespace: str,
    seed_offset: int,
) -> tuple[int, int]:
    """Derive primary or compatibility-secondary actor-observer cohort seed.

    Legacy namespace/offset pairs identify a separate governed authority.  The
    numeric offset is never added to the resulting uint64 seed.
    """

    authority_hash = str(cohort_authority_hash)
    cohort_index = 0
    governed_salt = PRIMARY_DECISION_COHORT_SALT_V1
    normalized_offset = int(seed_offset)
    if seed_namespace or normalized_offset:
        authority_hash = canonical_sha256({
            "schemaVersion": 1,
            "policy": _SECONDARY_COHORT_AUTHORITY_POLICY,
            "primaryCohortAuthorityHash": authority_hash,
            "seedNamespace": str(seed_namespace),
            "seedOffset": normalized_offset,
        })
        cohort_index = 1
        governed_salt = _SECONDARY_DECISION_COHORT_SALT
    seed = derive_actor_observer_cohort_seed_v1(
        public_prefix_hash=record.public_prefix_hash,
        decision_kind=record.decision_kind,
        acting_seat=record.acting_seat,
        cohort_authority_hash=authority_hash,
        cohort_index=cohort_index,
        governed_salt=governed_salt,
    )
    return seed, cohort_index


def _decision_seed(
    record: DecisionLedgerRecordV1,
    cohort_authority_hash: str,
    *,
    seed_namespace: str,
    seed_offset: int,
) -> int:
    return _decision_seed_and_index(
        record,
        cohort_authority_hash,
        seed_namespace=seed_namespace,
        seed_offset=seed_offset,
    )[0]


def _canonical_analysis_results(
    count: int,
    indexed: Mapping[int, "TraceDecisionAnalysisV3"],
) -> tuple["TraceDecisionAnalysisV3", ...]:
    expected = set(range(int(count)))
    actual = {int(index) for index in indexed}
    if actual != expected:
        raise RuntimeError("parallel trace analysis returned incomplete decision ordinals")
    return tuple(indexed[index] for index in range(int(count)))


def _canonical_worker_failure(errors: Mapping[int, BaseException]) -> BaseException | None:
    if not errors:
        return None
    return errors[min(int(index) for index in errors)]


def _abort_process_executor(
    executor: concurrent.futures.ProcessPoolExecutor,
    pending: Iterable[concurrent.futures.Future[Any]],
    *,
    maximum_wait_seconds: float,
) -> None:
    """Bound cancellation to processes owned by one decision executor.

    Python 3.12 and 3.13 cannot stop a running ``ProcessPoolExecutor`` future.
    Snapshotting this executor's workers before non-waiting shutdown gives the
    cancellation path an explicit ownership boundary without enumerating or
    disturbing unrelated child processes.
    """

    wait_seconds = float(maximum_wait_seconds)
    if wait_seconds <= 0:
        raise ValueError("process executor abort wait must be positive")

    failures: list[str] = []
    for future in tuple(pending):
        try:
            future.cancel()
        except BaseException as exc:
            failures.append(f"future_cancel:{type(exc).__name__}")

    owned_mapping = getattr(executor, "_processes", None)
    owned_processes = tuple(
        dict(owned_mapping or {}).values()
    )
    manager_thread = getattr(executor, "_executor_manager_thread", None)
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except BaseException as exc:
        failures.append(f"executor_shutdown:{type(exc).__name__}")

    def process_alive(process: Any) -> bool:
        try:
            return bool(process.is_alive())
        except BaseException as exc:
            failures.append(f"process_alive:{type(exc).__name__}")
            return True

    def join_until(processes: Iterable[Any], deadline: float) -> None:
        for process in processes:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            try:
                process.join(timeout=remaining)
            except BaseException as exc:
                failures.append(f"process_join:{type(exc).__name__}")

    join_until(owned_processes, time.monotonic() + wait_seconds)

    survivors = tuple(process for process in owned_processes if process_alive(process))
    for process in survivors:
        try:
            process.terminate()
        except BaseException as exc:
            failures.append(f"process_terminate:{type(exc).__name__}")
    join_until(survivors, time.monotonic() + wait_seconds)

    survivors = tuple(process for process in survivors if process_alive(process))
    for process in survivors:
        try:
            process.kill()
        except BaseException as exc:
            failures.append(f"process_kill:{type(exc).__name__}")
    join_until(survivors, time.monotonic() + wait_seconds)

    survivors = tuple(process for process in survivors if process_alive(process))
    if manager_thread is not None:
        try:
            manager_thread.join(timeout=wait_seconds)
        except BaseException as exc:
            failures.append(f"manager_join:{type(exc).__name__}")
        try:
            if manager_thread.is_alive():
                failures.append("manager_join:still_alive")
        except BaseException as exc:
            failures.append(f"manager_alive:{type(exc).__name__}")
    if survivors:
        failures.append(f"worker_processes_still_alive:{len(survivors)}")
    if failures:
        raise RuntimeError(
            "decision-analysis executor cleanup was incomplete: "
            + ",".join(failures)
        )


def _initialize_decision_analysis_worker(
    rules_data: Mapping[str, Any],
    convention: Mapping[str, Any],
    profile: TraceAnalysisProfileV1,
    match_value_profile: Mapping[str, Any],
    critical_threshold: float,
    minimum_critical_confidence: float,
    analysis_policy: AnalysisPolicy | None,
    belief_policy: BeliefPolicy | None,
    backend_id: str,
    cancel_event: Any,
    cohort_authority_hash: str | None = None,
) -> None:
    global _PROCESS_ANALYZER, _PROCESS_CANCEL_EVENT
    rules = RuleBook(dict(rules_data))
    exact_search_backend: ExactSearchBackend | None = None
    if backend_id == "native_exact_v3_python_fallback_v1":
        from baloot_platform.analysis.search_backend import NativeExactSearchBackendV3, PythonReferenceSearchBackend
        from baloot_platform.native.accelerator import NativeAccelerator, NativeUnavailable

        try:
            exact_search_backend = NativeExactSearchBackendV3(NativeAccelerator(rules))
        except (NativeUnavailable, OSError):
            exact_search_backend = PythonReferenceSearchBackend()
    elif backend_id != "python_reference_exact_search_v1":
        raise ValueError(
            f"process decision analysis does not support backend: {backend_id}"
        )
    _PROCESS_CANCEL_EVENT = cancel_event
    _PROCESS_ANALYZER = StagedTraceAnalyzerV1(
        rules,
        convention,
        profile,
        match_value_profile=match_value_profile,
        critical_threshold=critical_threshold,
        minimum_critical_confidence=minimum_critical_confidence,
        analysis_policy=analysis_policy,
        belief_policy=belief_policy,
        exact_search_backend=exact_search_backend,
        executor_profile=DecisionAnalysisExecutorProfileV1.build(worker_count=1),
        cohort_authority_hash=cohort_authority_hash,
    )


def _analyze_decision_in_worker(
    index: int,
    snapshot: ReplayDecisionStateV1,
    seed: int,
    cohort_index: int,
) -> tuple[int, "TraceDecisionAnalysisV3"]:
    analyzer = _PROCESS_ANALYZER
    cancel_event = _PROCESS_CANCEL_EVENT
    if analyzer is None or cancel_event is None:
        raise RuntimeError("decision analysis worker was not initialized")
    if cancel_event.is_set():
        raise InterruptedError("trace analysis cancelled")
    item = (
        analyzer._bid(
            snapshot,
            int(seed),
            cancel_event.is_set,
            cohort_index=int(cohort_index),
        )
        if snapshot.decision.decision_kind == "bid"
        else analyzer._play(
            snapshot,
            int(seed),
            cancel_event.is_set,
            cohort_index=int(cohort_index),
        )
    )
    return int(index), item


class StagedTraceAnalyzerV1:
    descriptor = PluginDescriptor(
        id="staged_trace_analyzer_v1",
        kind="trace_analyzer",
        version="1.3.0",
        description="Paired observer-safe analysis of every decision in a verified natural trace",
        capabilities=(
            "all_decisions",
            "actor_observer_truth",
            "paired_hidden_deals",
            "canonical_decision_score_evidence",
            "forced_legality_proof",
            "exact_information_set_confidence",
            "four_observer_projections",
            "labeled_unsupported_metrics",
        ),
        resource_class="cpu",
        deterministic=True,
        timeout_seconds=900,
        maximum_memory_mb=4096,
    )

    def __init__(
        self,
        rules: RuleBook,
        convention: Mapping[str, Any],
        profile: TraceAnalysisProfileV1,
        *,
        match_value_profile: Mapping[str, Any] | None = None,
        critical_threshold: float = 70.0,
        minimum_critical_confidence: float = 0.75,
        analysis_policy: AnalysisPolicy | None = None,
        belief_policy: BeliefPolicy | None = None,
        exact_search_backend: ExactSearchBackend | None = None,
        executor_profile: DecisionAnalysisExecutorProfileV1 | None = None,
        cohort_authority_hash: str | None = None,
    ):
        if not 0.0 <= float(critical_threshold) <= 100.0:
            raise ValueError("analysis critical threshold must be between zero and 100")
        self.rules = rules
        self.convention = dict(convention) if profile.use_conventions else {}
        self.profile = profile
        self.match_value_profile = dict(match_value_profile or {})
        self.critical_threshold = float(critical_threshold)
        if not 0.0 <= float(minimum_critical_confidence) <= 1.0:
            raise ValueError("analysis minimum confidence must be between zero and one")
        self.minimum_critical_confidence = float(minimum_critical_confidence)
        if analysis_policy is not None and belief_policy is not None:
            analysis_policy.validate_against(belief_policy)
        self.analysis_policy = analysis_policy
        self.belief_policy = belief_policy
        if (
            exact_search_backend is not None
            and exact_search_backend.descriptor.id != profile.search_backend_id
        ):
            raise ValueError("trace exact-search backend identity mismatch")
        self.exact_search_backend = exact_search_backend
        resolved_cohort_authority = (
            profile.sha256
            if cohort_authority_hash is None
            else str(cohort_authority_hash)
        )
        if (
            len(resolved_cohort_authority) != 64
            or any(
                character not in "0123456789abcdef"
                for character in resolved_cohort_authority
            )
        ):
            raise ValueError(
                "trace analysis cohort authority must be a canonical lowercase "
                "SHA-256 digest"
            )
        self.cohort_authority_hash = resolved_cohort_authority
        self.executor_profile = (
            executor_profile
            if executor_profile is not None
            else DecisionAnalysisExecutorProfileV1.build(worker_count=1)
        )

    @staticmethod
    def _action_key(value: Mapping[str, Any]) -> str:
        if "action" in value:
            return f"{value['action']}:{value.get('trumpSuit') or ''}"
        return f"card:{Card.parse(dict(value)).id}"

    @staticmethod
    def _confidence(
        effective_samples: float,
        *,
        kind: str,
        method: str = "",
        alternatives: Iterable[Mapping[str, Any]] = (),
    ) -> float:
        ranked = tuple(alternatives)
        denominator = 8.0 if kind == "bid" else 10.0
        sampled_confidence = min(
            1.0,
            math.sqrt(max(1.0, float(effective_samples))) / denominator,
        )
        if str(method).startswith("exhaustive_") and ranked and all(
            str(item.get("searchCompletionStatus") or "")
            == "heuristic_complete"
            for item in ranked
        ):
            # Confidence describes hidden-population coverage.  An exhaustive
            # compatible-deal population is complete even when the declared
            # continuation policy is heuristic.  Strategic truth remains
            # policy-conditioned and is never promoted to exact by this value.
            return 1.0
        if str(method).startswith("exhaustive_") and ranked and all(
            str(item.get("searchCompletionStatus") or "") == "exact"
            for item in ranked
        ):
            exact_coverage = min(
                max(0.0, min(1.0, float(item.get("exactShare", 0.0))))
                for item in ranked
            )
            return max(sampled_confidence, exact_coverage)
        return sampled_confidence

    @staticmethod
    def _finish(
        snapshot: ReplayDecisionStateV1,
        candidates: Iterable[Mapping[str, Any]],
        *,
        method: str,
        hidden_deals: int,
        effective_samples: float,
        difficulty: float,
        debate: float,
        supported_metrics: Iterable[str],
        unsupported_metrics: Iterable[str],
        primary_decision_score_evidence: (
            PrimaryDecisionScoreEvidenceV1 | None
        ) = None,
        forced_legality_proof: Mapping[str, Any] | None = None,
    ) -> TraceDecisionAnalysisV3:
        record = snapshot.decision
        ranked = tuple(dict(item) for item in candidates)
        if not ranked:
            raise ValueError("trace decision analysis produced no legal alternatives")
        actual_key = StagedTraceAnalyzerV1._action_key(record.chosen_action)
        selected = next(
            (item for item in ranked if StagedTraceAnalyzerV1._action_key(item["action"]) == actual_key),
            None,
        )
        if selected is None:
            raise ValueError("trace selected action is absent from paired analysis")
        best = ranked[0]
        if record.forced:
            if primary_decision_score_evidence is not None:
                raise ValueError(
                    "forced trace decision cannot contain primary score evidence"
                )
            if forced_legality_proof is None:
                raise ValueError(
                    "forced trace decision requires an explicit forced legality proof"
                )
            normalized_forced_proof = _validate_forced_legality_proof_v1(
                forced_legality_proof,
                record=record,
            )
        else:
            if primary_decision_score_evidence is None:
                raise ValueError(
                    "non-forced trace decision requires primary decision score evidence"
                )
            if forced_legality_proof is not None:
                raise ValueError(
                    "non-forced trace decision cannot contain a forced legality proof"
                )
            primary_decision_score_evidence = (
                validate_primary_decision_score_evidence_v1(
                    primary_decision_score_evidence,
                    decision_kind=record.decision_kind,
                    actor=record.acting_seat,
                    public_prefix_hash=record.public_prefix_hash,
                    legal_action_set_hash=legal_action_set_hash_v1(
                        record.legal_actions
                    ),
                )
            )
            normalized_forced_proof = None
            if canonical_json_bytes_v1(
                primary_decision_score_evidence.recommended_action
            ) != canonical_json_bytes_v1(best["action"]):
                raise ValueError(
                    "primary decision score recommendation does not match best action"
                )
        regret = max(0.0, float(best["expectedMargin"]) - float(selected["expectedMargin"]))
        probability_drop = max(
            0.0,
            float(best["winProbability"]) - float(selected["winProbability"]),
        )
        catastrophic = (
            float(best["expectedMargin"]) > 0 > float(selected["expectedMargin"])
            or float(best["winProbability"]) >= 0.5 > float(selected["winProbability"])
        )
        if primary_decision_score_evidence is not None:
            canonical_projection = canonical_surrounding_score_projection_v1(
                primary_decision_score_evidence,
                actual_action=record.chosen_action,
                best_action=best["action"],
            )
            regret = float(canonical_projection["expectedRegret"])
            probability_drop = float(canonical_projection["probabilityDrop"])
            catastrophic = bool(canonical_projection["catastrophicCrossing"])
        core = {
            "schemaVersion": 3,
            "decisionId": record.decision_id,
            "ordinal": record.ordinal,
            "actingSeat": record.acting_seat.name.lower(),
            "decisionKind": record.decision_kind,
            "forced": record.forced,
            "actualAction": dict(record.chosen_action),
            "bestAction": dict(best["action"]),
            "alternatives": list(ranked),
            "method": method,
            "hiddenDeals": int(hidden_deals),
            "effectiveSampleSize": float(effective_samples),
            "difficulty": float(difficulty),
            "debateScore": float(debate),
            "expectedRegret": regret,
            "probabilityDrop": probability_drop,
            "catastrophicCrossing": catastrophic,
            "supportedMetrics": sorted(set(str(item) for item in supported_metrics)),
            "unsupportedMetrics": sorted(set(str(item) for item in unsupported_metrics)),
            "primaryDecisionScoreEvidenceHash": (
                primary_decision_score_evidence.sha256
                if primary_decision_score_evidence is not None
                else None
            ),
            "forcedLegalityProofHash": (
                str(normalized_forced_proof["sha256"])
                if normalized_forced_proof is not None
                else None
            ),
        }
        return TraceDecisionAnalysisV3(
            record.decision_id,
            record.ordinal,
            record.acting_seat.name.lower(),
            record.decision_kind,
            record.forced,
            dict(record.chosen_action),
            dict(best["action"]),
            ranked,
            method,
            int(hidden_deals),
            float(effective_samples),
            (
                primary_decision_score_confidence_v1(
                    primary_decision_score_evidence
                )
                if primary_decision_score_evidence is not None
                else StagedTraceAnalyzerV1._confidence(
                    effective_samples,
                    kind=record.decision_kind,
                    method=method,
                    alternatives=ranked,
                )
            ),
            float(difficulty),
            float(debate),
            regret,
            probability_drop,
            catastrophic,
            tuple(core["supportedMetrics"]),
            tuple(core["unsupportedMetrics"]),
            canonical_sha256(core),
            primary_decision_score_evidence,
            normalized_forced_proof,
        )

    def _forced(self, snapshot: ReplayDecisionStateV1) -> TraceDecisionAnalysisV3:
        selected = dict(snapshot.decision.chosen_action)
        candidate = {
            "action": selected,
            "rank": 1,
            "expectedMargin": 0.0,
            "winProbability": 0.5,
            "confidence95": [0.5, 0.5],
            "downsideCvar10": 0.0,
            "volatility": 0.0,
            "bestShare": 1.0,
            "exactShare": 1.0,
            "principalVariation": [],
        }
        result = self._finish(
            snapshot,
            (candidate,),
            method="authoritative_forced_legal_action_v1",
            hidden_deals=0,
            effective_samples=100.0,
            difficulty=0.0,
            debate=0.0,
            supported_metrics=("legality", "forcedness"),
            unsupported_metrics=(
                "counterfactual_outcomes",
                "kabout_probability",
                "expected_future_rounds",
            ),
            forced_legality_proof=_forced_legality_proof_v1(
                snapshot,
                rules_hash=self.rules.sha256,
            ),
        )
        return result

    def _bid(
        self,
        snapshot: ReplayDecisionStateV1,
        seed: int,
        cancel_check: Callable[[], bool] | None,
        *,
        cohort_index: int = 0,
    ) -> TraceDecisionAnalysisV3:
        if snapshot.bidding is None:
            raise ValueError("bid analysis requires the replayed bidding machine")
        result = analyze_bid(
            snapshot.state,
            snapshot.bidding,
            snapshot.decision.acting_seat,
            self.rules,
            requested_samples=self.profile.bid_samples,
            seed=seed,
            cancel_check=cancel_check,
            exact_search_backend=self.exact_search_backend,
        )
        if result.actor != snapshot.decision.acting_seat:
            raise ValueError("bid analyzer actor does not match the trace decision")
        score_rows = bid_score_rows_v1(result.choices)
        score_evidence = primary_decision_score_evidence_v1(
            decision_kind="bid",
            actor=snapshot.decision.acting_seat,
            public_prefix_hash=snapshot.decision.public_prefix_hash,
            rules_hash=self.rules.sha256,
            analysis_identity_hash=self.cohort_authority_hash,
            trace_analysis_profile_hash=self.profile.sha256,
            cohort_seed_policy=PRIMARY_DECISION_COHORT_SEED_POLICY_V1,
            cohort_index=int(cohort_index),
            cohort_seed=int(seed),
            objective=BID_DECISION_OBJECTIVE_V1,
            tie_break=BID_DECISION_TIE_BREAK_V1,
            backend=self.profile.search_backend_id,
            method=result.method,
            hidden_deal_support=result.hidden_deals,
            effective_sample_size=float(result.hidden_deals),
            recommended_action=result.recommended.to_dict(),
            rows=score_rows,
            legal_action_set_hash=legal_action_set_hash_v1(
                snapshot.decision.legal_actions
            ),
        )
        candidates = tuple(
            {
                "action": item.choice.to_dict(),
                "rank": index + 1,
                "expectedMargin": item.expected_match_margin,
                "winProbability": item.team_win_probability,
                "contractSuccessProbability": item.contract_success_probability,
                "confidence95": [0.0, 1.0],
                "downsideCvar10": item.downside_cvar10,
                "volatility": item.volatility,
                "bestShare": item.best_share,
                "exactShare": 0.0,
                "principalVariation": [],
            }
            for index, item in enumerate(result.choices)
        )
        return self._finish(
            snapshot,
            candidates,
            method=result.method,
            hidden_deals=result.hidden_deals,
            effective_samples=float(result.hidden_deals),
            difficulty=result.difficulty,
            debate=result.debate_score,
            supported_metrics=(
                "expected_round_margin",
                "round_lead_probability",
                "contract_success_probability",
                "downside_cvar",
                "volatility",
                "best_deal_share",
            ),
            unsupported_metrics=(
                "true_match_win_probability",
                "expected_future_rounds",
                "kabout_probability",
                "exact_solve_share",
                "principal_continuation",
            ),
            primary_decision_score_evidence=score_evidence,
        )

    def _play(
        self,
        snapshot: ReplayDecisionStateV1,
        seed: int,
        cancel_check: Callable[[], bool] | None,
        *,
        cohort_index: int = 0,
    ) -> TraceDecisionAnalysisV3:
        info = InformationState.from_complete(
            snapshot.state,
            snapshot.decision.acting_seat,
        )
        result = analyze(
            info,
            self.rules,
            self.convention,
            requested_samples=self.profile.play_samples,
            exact_allocation_limit=self.profile.exact_allocation_limit,
            exact_search_max_remaining=self.profile.exact_search_max_remaining,
            maximum_sampling_attempts=self.profile.maximum_sampling_attempts,
            exact_search_max_nodes=self.profile.exact_search_max_nodes,
            exact_search_max_seconds=self.profile.exact_search_max_seconds,
            exact_search_max_depth=self.profile.exact_search_max_depth,
            exact_search_max_cache_entries=self.profile.exact_search_max_cache_entries,
            seed=seed,
            objective=self.profile.objective,
            match_value_profile=self.match_value_profile or None,
            cancel_check=cancel_check,
            exact_search_backend=self.exact_search_backend,
            analysis_policy=self.analysis_policy,
            belief_policy=self.belief_policy,
        )
        if result.observer != snapshot.decision.acting_seat:
            raise ValueError("play analyzer observer does not match the trace decision")
        if result.objective != self.profile.objective:
            raise ValueError("play analyzer objective does not match the trace profile")
        score_rows = play_score_rows_v1(result.actions)
        score_evidence = primary_decision_score_evidence_v1(
            decision_kind="play",
            actor=snapshot.decision.acting_seat,
            public_prefix_hash=snapshot.decision.public_prefix_hash,
            rules_hash=self.rules.sha256,
            analysis_identity_hash=self.cohort_authority_hash,
            trace_analysis_profile_hash=self.profile.sha256,
            cohort_seed_policy=PRIMARY_DECISION_COHORT_SEED_POLICY_V1,
            cohort_index=int(cohort_index),
            cohort_seed=int(seed),
            objective=result.objective,
            tie_break=PLAY_DECISION_TIE_BREAK_V1,
            backend=self.profile.search_backend_id,
            method=result.method,
            hidden_deal_support=result.hidden_deals,
            effective_sample_size=result.effective_sample_size,
            recommended_action=result.recommended_card.to_dict(),
            rows=score_rows,
            legal_action_set_hash=legal_action_set_hash_v1(
                snapshot.decision.legal_actions
            ),
        )
        candidates = tuple(
            {
                "action": item.card.to_dict(),
                "rank": index + 1,
                "expectedMargin": item.expected_round_margin,
                "expectedTeamAwardedPoints": item.expected_team_awarded_points,
                "expectedOpponentAwardedPoints": item.expected_opponent_awarded_points,
                "winProbability": item.round_lead_probability,
                "contractSuccessProbability": item.contract_success_probability,
                "matchWinProbability": item.match_win_probability,
                "matchWinConfidence95": [item.match_confidence_low, item.match_confidence_high],
                "confidence95": [item.confidence_low, item.confidence_high],
                "downsideCvar10": item.downside_cvar,
                "volatility": item.volatility,
                "bestShare": item.best_share,
                "exactShare": item.exact_share,
                "principalVariation": list(item.principal_variation),
                "searchNodes": item.search_nodes,
                "searchCompletionStatus": item.search_completion_status,
                "samples": item.samples,
                "effectiveSamples": item.effective_samples,
            }
            for index, item in enumerate(result.actions)
        )
        return self._finish(
            snapshot,
            candidates,
            method=result.method,
            hidden_deals=result.hidden_deals,
            effective_samples=result.effective_sample_size,
            difficulty=result.difficulty,
            debate=result.debate_score,
            supported_metrics=(
                "expected_team_awarded_points",
                "expected_opponent_awarded_points",
                "expected_round_margin",
                "contract_success_probability",
                "round_lead_probability",
                "match_win_probability",
                "downside_cvar",
                "volatility",
                "best_deal_share",
                "exact_solve_share",
                "confidence_interval",
                "principal_continuation",
                "search_completion",
            ),
            unsupported_metrics=(
                "expected_future_rounds",
                "kabout_probability",
                "project_consequence",
                "solver_model_agreement",
            ),
            primary_decision_score_evidence=score_evidence,
        )

    def analyze_trace(
        self,
        trace: CompleteGameTraceV1,
        *,
        cancel_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> TraceAnalysisResultV1:
        store = TraceDecisionProjectionStoreV1(trace, self.rules)
        analyses = self._analyze_snapshots(
            trace,
            store.replay.decisions,
            cancel_check=cancel_check,
            progress_callback=progress_callback,
        )
        return self._build_result(trace, store, analyses)

    def analyze_decisions(
        self,
        trace: CompleteGameTraceV1,
        decision_ids: Iterable[str],
        *,
        seed_namespace: str = "",
        seed_offset: int = 0,
        cancel_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> tuple[TraceDecisionAnalysisV3, ...]:
        """Analyze only governed confirmation targets in canonical trace order."""
        requested = {str(item) for item in decision_ids}
        store = TraceDecisionProjectionStoreV1(trace, self.rules)
        known = {item.decision.decision_id for item in store.replay.decisions}
        unknown = requested - known
        if unknown:
            raise ValueError(f"confirmation targets are absent from the trace: {sorted(unknown)}")
        snapshots = tuple(
            item for item in store.replay.decisions
            if item.decision.decision_id in requested
        )
        return self._analyze_snapshots(
            trace,
            snapshots,
            seed_namespace=seed_namespace,
            seed_offset=seed_offset,
            cancel_check=cancel_check,
            progress_callback=progress_callback,
        )

    def _analyze_snapshots(
        self,
        trace: CompleteGameTraceV1,
        snapshots: Iterable[ReplayDecisionStateV1],
        *,
        seed_namespace: str = "",
        seed_offset: int = 0,
        cancel_check: Callable[[], bool] | None,
        progress_callback: Callable[[float], None] | None,
    ) -> tuple[TraceDecisionAnalysisV3, ...]:
        decisions = tuple(snapshots)
        if self.executor_profile.worker_count == 1 or len(decisions) < 2:
            return self._analyze_snapshots_serial(
                trace,
                decisions,
                seed_namespace=seed_namespace,
                seed_offset=seed_offset,
                cancel_check=cancel_check,
                progress_callback=progress_callback,
            )
        return self._analyze_snapshots_parallel(
            trace,
            decisions,
            seed_namespace=seed_namespace,
            seed_offset=seed_offset,
            cancel_check=cancel_check,
            progress_callback=progress_callback,
        )

    def _analyze_snapshots_serial(
        self,
        trace: CompleteGameTraceV1,
        decisions: tuple[ReplayDecisionStateV1, ...],
        *,
        seed_namespace: str,
        seed_offset: int,
        cancel_check: Callable[[], bool] | None,
        progress_callback: Callable[[float], None] | None,
    ) -> tuple[TraceDecisionAnalysisV3, ...]:
        analyses: list[TraceDecisionAnalysisV3] = []
        for index, snapshot in enumerate(decisions):
            if cancel_check is not None and cancel_check():
                raise InterruptedError("trace analysis cancelled")
            record = snapshot.decision
            if record.forced:
                item = self._forced(snapshot)
            else:
                seed, cohort_index = _decision_seed_and_index(
                    record,
                    self.cohort_authority_hash,
                    seed_namespace=seed_namespace,
                    seed_offset=seed_offset,
                )
                item = (
                    self._bid(
                        snapshot,
                        seed,
                        cancel_check,
                        cohort_index=cohort_index,
                    )
                    if record.decision_kind == "bid"
                    else self._play(
                        snapshot,
                        seed,
                        cancel_check,
                        cohort_index=cohort_index,
                    )
                )
            analyses.append(item)
            if progress_callback is not None:
                progress_callback((index + 1) / max(1, len(decisions)))
        return tuple(analyses)

    def _analyze_snapshots_parallel(
        self,
        trace: CompleteGameTraceV1,
        decisions: tuple[ReplayDecisionStateV1, ...],
        *,
        seed_namespace: str,
        seed_offset: int,
        cancel_check: Callable[[], bool] | None,
        progress_callback: Callable[[float], None] | None,
    ) -> tuple[TraceDecisionAnalysisV3, ...]:
        profile = self.executor_profile
        indexed: dict[int, TraceDecisionAnalysisV3] = {}
        work: list[tuple[int, ReplayDecisionStateV1, int, int]] = []
        completed_count = 0
        for index, snapshot in enumerate(decisions):
            if cancel_check is not None and cancel_check():
                raise InterruptedError("trace analysis cancelled")
            record = snapshot.decision
            if record.forced:
                indexed[index] = self._forced(snapshot)
                completed_count += 1
                if progress_callback is not None:
                    progress_callback(completed_count / max(1, len(decisions)))
                continue
            work.append((
                index,
                snapshot,
                *_decision_seed_and_index(
                    record,
                    self.cohort_authority_hash,
                    seed_namespace=seed_namespace,
                    seed_offset=seed_offset,
                ),
            ))
        if not work:
            return _canonical_analysis_results(len(decisions), indexed)

        backend_id = (
            "python_reference_exact_search_v1"
            if self.exact_search_backend is None
            else str(self.exact_search_backend.descriptor.id)
        )
        if backend_id not in {
            "python_reference_exact_search_v1",
            "native_exact_v3_python_fallback_v1",
        }:
            raise ValueError(
                f"process decision analysis does not support backend: {backend_id}"
            )
        context = multiprocessing.get_context("spawn")
        cancel_event = context.Event()
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=profile.worker_count,
            mp_context=context,
            initializer=_initialize_decision_analysis_worker,
            initargs=(
                self.rules.data,
                self.convention,
                self.profile,
                self.match_value_profile,
                self.critical_threshold,
                self.minimum_critical_confidence,
                self.analysis_policy,
                self.belief_policy,
                backend_id,
                cancel_event,
                self.cohort_authority_hash,
            ),
        )
        maximum_pending = (
            profile.worker_count * profile.maximum_pending_per_worker
        )
        pending: dict[concurrent.futures.Future[Any], int] = {}
        errors: dict[int, BaseException] = {}
        cursor = 0
        started = time.monotonic()

        def submit_available() -> None:
            nonlocal cursor
            while cursor < len(work) and len(pending) < maximum_pending:
                index, snapshot, seed, cohort_index = work[cursor]
                cursor += 1
                try:
                    future = executor.submit(
                        _analyze_decision_in_worker,
                        index,
                        snapshot,
                        seed,
                        cohort_index,
                    )
                except BaseException as exc:
                    errors[index] = exc
                    continue
                pending[future] = index

        try:
            submit_available()
            while pending or cursor < len(work):
                if cancel_check is not None and cancel_check():
                    cancel_event.set()
                    raise InterruptedError("trace analysis cancelled")
                remaining_seconds = profile.maximum_seconds - (
                    time.monotonic() - started
                )
                if remaining_seconds <= 0:
                    cancel_event.set()
                    raise TimeoutError("parallel trace analysis exceeded its governed timeout")
                if not pending:
                    submit_available()
                    continue
                done, _not_done = concurrent.futures.wait(
                    pending,
                    timeout=min(profile.poll_seconds, remaining_seconds),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                if not done:
                    continue
                for future in sorted(done, key=lambda item: pending[item]):
                    expected_index = pending.pop(future)
                    try:
                        returned_index, item = future.result()
                        if int(returned_index) != expected_index:
                            raise RuntimeError(
                                "parallel trace analysis returned a mismatched ordinal"
                            )
                        indexed[expected_index] = item
                    except BaseException as exc:
                        errors[expected_index] = exc
                    completed_count += 1
                    if progress_callback is not None:
                        progress_callback(
                            completed_count / max(1, len(decisions))
                        )
                submit_available()
            failure = _canonical_worker_failure(errors)
            if failure is not None:
                raise failure
        except BaseException as exc:
            cancel_event.set()
            try:
                _abort_process_executor(
                    executor,
                    pending,
                    maximum_wait_seconds=profile.poll_seconds,
                )
            except BaseException as cleanup_error:
                exc.add_note(
                    "decision-analysis executor cleanup failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise
        executor.shutdown(wait=True, cancel_futures=True)
        return _canonical_analysis_results(len(decisions), indexed)

    def _build_result(
        self,
        trace: CompleteGameTraceV1,
        store: TraceDecisionProjectionStoreV1,
        analyses: Iterable[TraceDecisionAnalysisV3],
    ) -> TraceAnalysisResultV1:
        analyses = tuple(analyses)
        if {item.decision_id for item in analyses} != {
            item.decision_id for item in trace.decision_ledger.records
        }:
            raise ValueError("complete trace analysis must cover every ledger decision")
        audits = [item.audit_mapping() for item in analyses]
        laffa = MultiPerspectiveFoundryV1(
            store,
            critical_threshold=self.critical_threshold,
            minimum_critical_confidence=self.minimum_critical_confidence,
        ).analyze_laffa(trace.game_hash, {"decisions": audits})
        core = {
            "schemaVersion": 1,
            "traceHash": trace.sha256,
            "profileHash": self.profile.sha256,
            "analyzerId": self.descriptor.id,
            "analyzerVersion": self.descriptor.version,
            "decisionAnalysisHashes": [item.sha256 for item in analyses],
            "laffaEvidenceHash": laffa.sha256,
        }
        return TraceAnalysisResultV1(
            trace.sha256,
            self.profile.sha256,
            self.descriptor.id,
            self.descriptor.version,
            tuple(analyses),
            laffa,
            canonical_sha256(core),
        )


class CascadedTraceAnalyzerV1:
    """Screen every decision, then deeply confirm only the strongest moments."""

    descriptor = PluginDescriptor(
        id="cascaded_trace_analyzer_v1",
        kind="trace_analyzer",
        version="1.3.0",
        description="All-decision observer-safe screening with bounded deep confirmation",
        capabilities=(
            "all_decisions",
            "all_seats",
            "four_observer_projections",
            "bounded_deep_confirmation",
            "canonical_decision_score_evidence",
            "forced_legality_proof",
            "exact_information_set_confidence",
            "stage_provenance",
        ),
        resource_class="cpu",
        deterministic=True,
        timeout_seconds=900,
        maximum_memory_mb=4096,
    )

    def __init__(
        self,
        screening_analyzer: StagedTraceAnalyzerV1,
        confirmation_analyzer: StagedTraceAnalyzerV1,
        profile: CascadedTraceAnalysisProfileV1,
    ):
        if screening_analyzer.rules.sha256 != confirmation_analyzer.rules.sha256:
            raise ValueError("cascaded analyzers must use identical rules")
        if profile.screening_profile_hash != screening_analyzer.profile.sha256:
            raise ValueError("cascaded screening profile identity mismatch")
        if profile.confirmation_profile_hash != confirmation_analyzer.profile.sha256:
            raise ValueError("cascaded confirmation profile identity mismatch")
        self.screening_analyzer = screening_analyzer
        self.confirmation_analyzer = confirmation_analyzer
        self.profile = profile
        self.rules = confirmation_analyzer.rules
        self.critical_threshold = confirmation_analyzer.critical_threshold
        self.minimum_critical_confidence = (
            confirmation_analyzer.minimum_critical_confidence
        )

    @staticmethod
    def screening_score(item: TraceDecisionAnalysisV3) -> float:
        return decision_materiality_v1(item)

    def _confirmation_targets(
        self,
        screening: TraceAnalysisResultV1,
    ) -> tuple[str, ...]:
        ranked = sorted(
            (
                (self.screening_score(item), item.ordinal, item.decision_id)
                for item in screening.decisions
                if not item.forced
            ),
            key=lambda item: (-item[0], item[1], item[2]),
        )
        return tuple(
            decision_id
            for score, _, decision_id in ranked
            if score >= self.profile.minimum_screening_score
        )[: self.profile.maximum_confirmation_decisions_per_trace]

    def analyze_trace(
        self,
        trace: CompleteGameTraceV1,
        *,
        cancel_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> TraceAnalysisResultV1:
        screening = self.screening_analyzer.analyze_trace(
            trace,
            cancel_check=cancel_check,
            progress_callback=(
                None
                if progress_callback is None
                else lambda fraction: progress_callback(0.55 * fraction)
            ),
        )
        targets = self._confirmation_targets(screening)
        confirmations = self.confirmation_analyzer.analyze_decisions(
            trace,
            targets,
            cancel_check=cancel_check,
            progress_callback=(
                None
                if progress_callback is None
                else lambda fraction: progress_callback(0.55 + 0.45 * fraction)
            ),
        )
        if progress_callback is not None and not targets:
            progress_callback(1.0)
        replacements = {item.decision_id: item for item in confirmations}
        analyses = tuple(
            replacements.get(item.decision_id, item)
            for item in screening.decisions
        )
        store = TraceDecisionProjectionStoreV1(trace, self.rules)
        audits = [item.audit_mapping() for item in analyses]
        laffa = MultiPerspectiveFoundryV1(
            store,
            critical_threshold=self.critical_threshold,
            minimum_critical_confidence=self.minimum_critical_confidence,
        ).analyze_laffa(trace.game_hash, {"decisions": audits})
        confirmed = tuple(sorted(replacements))
        core = {
            "schemaVersion": 2,
            "traceHash": trace.sha256,
            "profileHash": self.profile.sha256,
            "analyzerId": self.descriptor.id,
            "analyzerVersion": self.descriptor.version,
            "decisionAnalysisHashes": [item.sha256 for item in analyses],
            "laffaEvidenceHash": laffa.sha256,
            "confirmedDecisionIds": list(confirmed),
            "screeningProfileHash": self.profile.screening_profile_hash,
            "confirmationProfileHash": self.profile.confirmation_profile_hash,
        }
        return TraceAnalysisResultV1(
            trace.sha256,
            self.profile.sha256,
            self.descriptor.id,
            self.descriptor.version,
            analyses,
            laffa,
            canonical_sha256(core),
            confirmed,
            self.profile.screening_profile_hash,
            self.profile.confirmation_profile_hash,
        )
