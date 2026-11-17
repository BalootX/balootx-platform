from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Mapping


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def identifier(prefix: str, value: Any) -> str:
    return f"{prefix}-{digest(value)[:20].upper()}"


class ReviewCaseStatus(StrEnum):
    DRAFT = "draft"
    SUBMITTED = "submitted"
    BLIND_PEER_REVIEW = "blind_peer_review"
    CONSENSUS_CANDIDATE = "consensus_candidate"
    EXPERT_ADJUDICATION = "expert_adjudication"
    TRAINING_CANDIDATE = "training_candidate"
    CHALLENGER_EVALUATED = "challenger_evaluated"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    DISPUTED = "disputed"
    SUPERSEDED = "superseded"


class AssignmentStatus(StrEnum):
    ACTIVE = "active"
    SUBMITTED = "submitted"
    EXPIRED = "expired"
    REVOKED = "revoked"


class SubmissionStatus(StrEnum):
    SUBMITTED = "submitted"
    LEGALITY_VALIDATED = "legality_validated"
    INFORMATION_SET_VALIDATED = "information_set_validated"
    ENGINE_CHECKED = "engine_checked"
    BLOCKED = "blocked"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class ConsentRecord:
    subject_id: str
    version: str = "1"
    allow_training: bool = False
    allow_research: bool = False
    allow_public_challenge: bool = False
    allow_voice: bool = False
    allow_identity_attribution: bool = False
    allow_private_coaching: bool = True
    effective_at: str = field(default_factory=utc_now)
    revoked_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "subjectId": self.subject_id, "version": self.version,
            "allowTraining": self.allow_training, "allowResearch": self.allow_research,
            "allowPublicChallenge": self.allow_public_challenge, "allowVoice": self.allow_voice,
            "allowIdentityAttribution": self.allow_identity_attribution,
            "allowPrivateCoaching": self.allow_private_coaching,
            "effectiveAt": self.effective_at, "revokedAt": self.revoked_at,
        }


@dataclass(frozen=True, slots=True)
class ReviewerProfile:
    subject_id: str
    verified_role: str = "player"
    role_verified: bool = False
    legality_accuracy: float = 0.5
    information_set_discipline: float = 0.5
    strategic_accuracy: float = 0.5
    calibration: float = 0.5
    independence: float = 0.5
    adjudicated_cases: int = 0
    updated_at: str = field(default_factory=utc_now)

    @property
    def reputation(self) -> float:
        value = (
            0.22 * self.legality_accuracy + 0.23 * self.information_set_discipline +
            0.25 * self.strategic_accuracy + 0.15 * self.calibration + 0.15 * self.independence
        )
        return max(0.1, min(1.0, value))

    @property
    def role_prior(self) -> float:
        role = self.verified_role if self.role_verified else "player"
        return {"player": 1.0, "experienced_player": 1.2, "coach": 1.35, "expert": 1.55, "referee": 1.55, "admin": 1.4}.get(role, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subjectId": self.subject_id, "verifiedRole": self.verified_role,
            "roleVerified": self.role_verified, "reputation": self.reputation,
            "rolePrior": self.role_prior, "legalityAccuracy": self.legality_accuracy,
            "informationSetDiscipline": self.information_set_discipline,
            "strategicAccuracy": self.strategic_accuracy, "calibration": self.calibration,
            "independence": self.independence, "adjudicatedCases": self.adjudicated_cases,
            "updatedAt": self.updated_at,
        }


@dataclass(frozen=True, slots=True)
class GovernedReviewCase:
    case_id: str
    decision_id: str
    created_by: str
    public_context: Mapping[str, Any]
    legal_actions: tuple[Any, ...]
    production_action: Any
    engine_version: str
    bundle_hash: str
    correction_type: str = "analysis_defect"
    status: str = ReviewCaseStatus.BLIND_PEER_REVIEW.value
    high_impact: bool = False
    completed: bool = True
    hidden_revealed_at: str | None = None
    result_revealed_at: str | None = None
    created_at: str = field(default_factory=utc_now)
    case_hash: str = ""

    @classmethod
    def create(cls, **kwargs: Any) -> "GovernedReviewCase":
        base = dict(kwargs)
        base.setdefault("created_at", utc_now())
        payload = {
            "decisionId": base["decision_id"], "createdBy": base["created_by"],
            "publicContext": base["public_context"], "legalActions": list(base["legal_actions"]),
            "productionAction": base.get("production_action"), "engineVersion": base.get("engine_version", ""),
            "bundleHash": base.get("bundle_hash", ""), "highImpact": bool(base.get("high_impact", False)),
            "correctionType": base.get("correction_type", "analysis_defect"),
            "completed": bool(base.get("completed", True)), "createdAt": base["created_at"],
        }
        h = digest(payload)
        base["case_id"] = base.get("case_id") or f"HITL-{h[:20].upper()}"
        base["case_hash"] = h
        base["legal_actions"] = tuple(base["legal_actions"])
        return cls(**base)

    def to_dict(self, *, blind: bool = False) -> dict[str, Any]:
        value = {
            "schemaVersion": 1, "contract": "CorrectionCaseV1",
            "caseId": self.case_id, "decisionId": self.decision_id, "createdBy": self.created_by,
            "publicContext": dict(self.public_context), "legalActions": list(self.legal_actions),
            "engineVersion": self.engine_version, "bundleHash": self.bundle_hash,
            "correctionType": self.correction_type,
            "status": self.status, "highImpact": self.high_impact, "completed": self.completed,
            "hiddenRevealedAt": self.hidden_revealed_at, "resultRevealedAt": self.result_revealed_at,
            "createdAt": self.created_at, "caseHash": self.case_hash, "blind": blind,
        }
        if not blind:
            value["productionAction"] = self.production_action
        return value


@dataclass(frozen=True, slots=True)
class BlindAssignment:
    assignment_id: str
    case_id: str
    reviewer_id: str
    token_hash: str
    device_cluster_hash: str
    network_cluster_hash: str
    status: str = AssignmentStatus.ACTIVE.value
    assigned_at: str = field(default_factory=utc_now)
    expires_at: str | None = None

    def to_dict(self, *, include_token: str | None = None) -> dict[str, Any]:
        value = {
            "assignmentId": self.assignment_id, "caseId": self.case_id, "reviewerId": self.reviewer_id,
            "tokenHash": self.token_hash, "deviceClusterHash": self.device_cluster_hash,
            "networkClusterHash": self.network_cluster_hash, "status": self.status,
            "assignedAt": self.assigned_at, "expiresAt": self.expires_at,
        }
        if include_token is not None:
            value["assignmentToken"] = include_token
        return value


@dataclass(frozen=True, slots=True)
class ReviewSubmission:
    submission_id: str
    case_id: str
    assignment_id: str
    reviewer_id: str
    preferred_action: Any
    confidence: float
    reason_codes: tuple[str, ...]
    rationale_ar: str
    claims: tuple[Mapping[str, Any], ...]
    evidence: tuple[Mapping[str, Any], ...]
    view_mode: str
    hidden_cards_seen: bool
    final_result_seen: bool
    disclosure_timing: str
    status: str = SubmissionStatus.SUBMITTED.value
    submitted_at: str = field(default_factory=utc_now)
    submission_hash: str = ""

    @classmethod
    def create(cls, **kwargs: Any) -> "ReviewSubmission":
        base = dict(kwargs)
        base.setdefault("submitted_at", utc_now())
        rationale = " ".join(str(base.get("rationale_ar", "")).split())
        if len(rationale) < 3:
            raise ValueError("review rationale is too short")
        if len(rationale) > 4000:
            raise ValueError("review rationale is too long")
        base["rationale_ar"] = rationale
        base["confidence"] = max(0.0, min(1.0, float(base.get("confidence", 0.5))))
        base["reason_codes"] = tuple(str(x) for x in base.get("reason_codes") or ("other",))
        base["claims"] = tuple(dict(x) for x in base.get("claims") or ())
        base["evidence"] = tuple(dict(x) for x in base.get("evidence") or ())
        payload = {
            "caseId": base["case_id"], "assignmentId": base["assignment_id"], "reviewerId": base["reviewer_id"],
            "preferredAction": base["preferred_action"], "confidence": base["confidence"],
            "reasonCodes": list(base["reason_codes"]), "rationaleAr": rationale,
            "claims": list(base["claims"]), "evidence": list(base["evidence"]),
            "viewMode": base["view_mode"], "hiddenCardsSeen": bool(base["hidden_cards_seen"]),
            "finalResultSeen": bool(base["final_result_seen"]), "disclosureTiming": base["disclosure_timing"],
            "submittedAt": base["submitted_at"],
        }
        h = digest(payload)
        base["submission_id"] = base.get("submission_id") or f"SUB-{h[:20].upper()}"
        base["submission_hash"] = h
        return cls(**base)

    def to_dict(self) -> dict[str, Any]:
        return {
            "submissionId": self.submission_id, "caseId": self.case_id, "assignmentId": self.assignment_id,
            "reviewerId": self.reviewer_id, "preferredAction": self.preferred_action,
            "confidence": self.confidence, "reasonCodes": list(self.reason_codes), "rationaleAr": self.rationale_ar,
            "claims": [dict(x) for x in self.claims], "evidence": [dict(x) for x in self.evidence],
            "viewMode": self.view_mode, "hiddenCardsSeen": self.hidden_cards_seen,
            "finalResultSeen": self.final_result_seen, "disclosureTiming": self.disclosure_timing,
            "status": self.status, "submittedAt": self.submitted_at, "submissionHash": self.submission_hash,
        }


@dataclass(frozen=True, slots=True)
class ValidationResult:
    validation_id: str
    submission_id: str
    legal: bool
    information_set_safe: bool
    hindsight_safe: bool
    engine_status: str
    engine_regret: float | None = None
    engine_win_drop: float | None = None
    deep_solver_status: str = "not_run"
    challenger_status: str = "not_run"
    reasons: tuple[str, ...] = ()
    validated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "validationId": self.validation_id, "submissionId": self.submission_id,
            "legal": self.legal, "informationSetSafe": self.information_set_safe,
            "hindsightSafe": self.hindsight_safe, "engineStatus": self.engine_status,
            "engineRegret": self.engine_regret, "engineWinDrop": self.engine_win_drop,
            "deepSolverStatus": self.deep_solver_status, "challengerStatus": self.challenger_status,
            "reasons": list(self.reasons), "validatedAt": self.validated_at,
        }


@dataclass(frozen=True, slots=True)
class ConsensusSnapshot:
    consensus_id: str
    case_id: str
    preferred_action: Any | None
    reviewer_count: int
    independent_reviewer_count: int
    weighted_support: float
    wilson_lower_bound: float
    expert_coverage: bool
    expert_opposition: bool
    cluster_diversity: int
    eligible: bool
    reasons: tuple[str, ...]
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "consensusId": self.consensus_id, "caseId": self.case_id,
            "preferredAction": self.preferred_action, "reviewerCount": self.reviewer_count,
            "independentReviewerCount": self.independent_reviewer_count,
            "weightedSupport": self.weighted_support, "wilsonLowerBound": self.wilson_lower_bound,
            "expertCoverage": self.expert_coverage, "expertOpposition": self.expert_opposition,
            "clusterDiversity": self.cluster_diversity, "eligible": self.eligible,
            "reasons": list(self.reasons), "createdAt": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class ModerationFlag:
    flag_id: str
    case_id: str
    submission_ids: tuple[str, ...]
    code: str
    severity: str
    details: Mapping[str, Any]
    status: str = "open"
    created_at: str = field(default_factory=utc_now)
    resolved_at: str | None = None
    resolved_by: str | None = None
    resolution: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "flagId": self.flag_id, "caseId": self.case_id,
            "submissionIds": list(self.submission_ids), "code": self.code,
            "severity": self.severity, "details": dict(self.details), "status": self.status,
            "createdAt": self.created_at, "resolvedAt": self.resolved_at,
            "resolvedBy": self.resolved_by, "resolution": self.resolution,
        }


@dataclass(frozen=True, slots=True)
class AdjudicationRecord:
    adjudication_id: str
    case_id: str
    adjudicator_id: str
    outcome: str
    preferred_action: Any | None
    rationale_ar: str
    preserve_dissent: bool = True
    certification_manifest: Mapping[str, Any] | None = None
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "adjudicationId": self.adjudication_id, "caseId": self.case_id,
            "adjudicatorId": self.adjudicator_id, "outcome": self.outcome,
            "preferredAction": self.preferred_action, "rationaleAr": self.rationale_ar,
            "preserveDissent": self.preserve_dissent,
            "certificationManifest": dict(self.certification_manifest or {}),
            "createdAt": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class TrainingCandidate:
    candidate_id: str
    case_id: str
    label_provenance: Mapping[str, Any]
    preferred_action: Any
    consent_subjects: tuple[str, ...]
    status: str = "training_candidate"
    automatic_production_promotion: bool = False
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidateId": self.candidate_id, "caseId": self.case_id,
            "labelProvenance": dict(self.label_provenance), "preferredAction": self.preferred_action,
            "consentSubjects": list(self.consent_subjects), "status": self.status,
            "automaticProductionPromotion": False, "createdAt": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class ChallengerEvaluation:
    evaluation_id: str
    candidate_id: str
    validator_id: str
    validator_type: str
    status: str
    metrics: Mapping[str, Any]
    independent: bool = True
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluationId": self.evaluation_id, "candidateId": self.candidate_id,
            "validatorId": self.validator_id, "validatorType": self.validator_type,
            "status": self.status, "metrics": dict(self.metrics),
            "independent": self.independent, "createdAt": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class ImprovementProposal:
    proposal_id: str
    created_by: str
    target: str
    title_ar: str
    rationale_ar: str
    source_case_ids: tuple[str, ...]
    requested_change: Mapping[str, Any]
    status: str = "proposed"
    automatic_production_apply: bool = False
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposalId": self.proposal_id, "createdBy": self.created_by,
            "target": self.target, "titleAr": self.title_ar, "rationaleAr": self.rationale_ar,
            "sourceCaseIds": list(self.source_case_ids), "requestedChange": dict(self.requested_change),
            "status": self.status, "automaticProductionApply": False, "createdAt": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class ImprovementEvaluation:
    evaluation_id: str
    proposal_id: str
    validator_id: str
    status: str
    evidence: Mapping[str, Any]
    independent: bool = True
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluationId": self.evaluation_id, "proposalId": self.proposal_id,
            "validatorId": self.validator_id, "status": self.status,
            "evidence": dict(self.evidence), "independent": self.independent,
            "createdAt": self.created_at,
        }
