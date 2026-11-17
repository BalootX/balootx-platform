from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from baloot_platform.hitl import ReviewGovernanceServiceV18
from baloot_platform.identity import IdentityService
from baloot_platform.security import AuthorizationPolicy, Scope
from baloot_platform.video.hitl_review_v18 import HitlReviewClipRendererV18

from .auth import require_identity


class CaseRequest(BaseModel):
    decisionId: str
    correctionType: str = "analysis_defect"
    highImpact: bool = False
    completed: bool = True


class AssignmentRequest(BaseModel):
    reviewerId: str | None = None
    deviceId: str = ""
    networkId: str = ""
    ttlHours: int = Field(default=72, ge=1, le=720)


class SubmissionRequest(BaseModel):
    assignmentId: str
    assignmentToken: str
    preferredAction: Any
    confidence: float = Field(ge=0, le=1)
    reasonCodes: list[str] = Field(min_length=1)
    rationaleAr: str = Field(min_length=3, max_length=4000)
    claims: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    viewMode: str = "observer"
    hiddenCardsSeen: bool = False
    finalResultSeen: bool = False
    disclosureTiming: str = "before_reveal"


class ConsentRequest(BaseModel):
    allowTraining: bool = False
    allowResearch: bool = False
    allowPublicChallenge: bool = False
    allowVoice: bool = False
    allowIdentityAttribution: bool = False
    allowPrivateCoaching: bool = True
    revoke: bool = False


class ReviewerProfileRequest(BaseModel):
    verifiedRole: str = "player"
    roleVerified: bool = False
    legalityAccuracy: float = Field(default=0.5, ge=0, le=1)
    informationSetDiscipline: float = Field(default=0.5, ge=0, le=1)
    strategicAccuracy: float = Field(default=0.5, ge=0, le=1)
    calibration: float = Field(default=0.5, ge=0, le=1)
    independence: float = Field(default=0.5, ge=0, le=1)
    adjudicatedCases: int = Field(default=0, ge=0)


class ModerationRequest(BaseModel):
    submissionIds: list[str] = Field(default_factory=list)
    code: str
    severity: str = "medium"
    details: dict[str, Any] = Field(default_factory=dict)


class ResolveFlagRequest(BaseModel):
    resolution: str = Field(min_length=2, max_length=1000)


class AdjudicationRequest(BaseModel):
    outcome: str
    preferredAction: Any | None = None
    rationaleAr: str = Field(min_length=3, max_length=4000)
    certificationManifest: dict[str, Any] | None = None


class ChallengerEvaluationRequest(BaseModel):
    validatorId: str
    validatorType: str
    status: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    independent: bool = True


class ImprovementRequest(BaseModel):
    target: str
    titleAr: str = Field(min_length=3, max_length=300)
    rationaleAr: str = Field(min_length=3, max_length=4000)
    sourceCaseIds: list[str] = Field(default_factory=list)
    requestedChange: dict[str, Any] = Field(default_factory=dict)


class ImprovementEvaluationRequest(BaseModel):
    validatorId: str
    status: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    independent: bool = True


class ChangeManifestRequest(BaseModel):
    changeManifest: dict[str, Any]


class ClipRequest(BaseModel):
    width: int | None = Field(default=None, ge=320, le=2160)
    height: int | None = Field(default=None, ge=320, le=3840)
    fps: int | None = None
    durationSeconds: float | None = Field(default=None, gt=0, le=60)

    @model_validator(mode="after")
    def validate_mobile_canvas(self) -> "ClipRequest":
        if (self.width is None) != (self.height is None):
            raise ValueError("review clip width and height must be supplied together")
        if (
            self.width is not None
            and self.height is not None
            and self.width * 16 != self.height * 9
        ):
            raise ValueError("review clip resolution must use an exact 9:16 aspect ratio")
        return self


def _role(request: Request) -> str:
    claims = getattr(request.state, "access_claims", None)
    if claims is not None:
        return str(getattr(claims, "role", "player")).lower()
    if bool(getattr(request.state, "api_key_valid", False)):
        return request.headers.get("x-baloot-role", "service").lower()
    return "player"


def _privileged(request: Request, policy: AuthorizationPolicy, scope: Scope) -> None:
    claims = getattr(request.state, "access_claims", None)
    try:
        policy.require(role=_role(request), required=scope, claims=claims)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc


def build_hitl_router(service: ReviewGovernanceServiceV18, identities: IdentityService,
                      policy: AuthorizationPolicy, renderer: HitlReviewClipRendererV18,
                      outputs: str | Path, profile: dict[str, Any]) -> APIRouter:
    router = APIRouter(prefix="/api/v1/hitl", tags=["human-in-the-loop"])
    output_root = Path(outputs)

    @router.get("/profile")
    def get_profile(request: Request):
        require_identity(request, identities)
        return {"profile": profile, "reasonTaxonomy": list(service.REASON_TAXONOMY),
                "correctionTaxonomy": list(service.CORRECTION_TAXONOMY),
                "automaticProductionApply": False, "automaticModelPromotion": False}

    @router.get("/cases")
    def list_cases(request: Request, limit: int = 100):
        require_identity(request, identities)
        return {"items": service.list_cases(limit=max(1, min(500, limit)))}

    @router.post("/cases", status_code=201)
    def create_case(body: CaseRequest, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            return service.create_case(body.decisionId, created_by=identity.subject_id,
                                       correction_type=body.correctionType,
                                       high_impact=body.highImpact, completed=body.completed)
        except KeyError as exc:
            raise HTTPException(404, "decision not found") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/cases/{case_id}")
    def get_case(case_id: str, request: Request, blind: bool = False):
        require_identity(request, identities)
        try:
            return service.case(case_id, blind=blind)
        except KeyError as exc:
            raise HTTPException(404, "review case not found") from exc

    @router.post("/cases/{case_id}/assignments", status_code=201)
    def assign(case_id: str, body: AssignmentRequest, request: Request):
        identity, _ = require_identity(request, identities)
        reviewer = body.reviewerId or identity.subject_id
        if reviewer != identity.subject_id:
            _privileged(request, policy, Scope.MODEL_ADMIN)
        try:
            return service.assign(case_id, reviewer_id=reviewer, device_id=body.deviceId,
                                  network_id=body.networkId, ttl_hours=body.ttlHours)
        except KeyError as exc:
            raise HTTPException(404, "review case not found") from exc

    @router.post("/cases/{case_id}/submissions", status_code=201)
    def submit(case_id: str, body: SubmissionRequest, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            return service.submit(
                case_id, assignment_id=body.assignmentId, assignment_token=body.assignmentToken,
                reviewer_id=identity.subject_id, preferred_action=body.preferredAction,
                confidence=body.confidence, reason_codes=body.reasonCodes, rationale_ar=body.rationaleAr,
                claims=body.claims, evidence=body.evidence, view_mode=body.viewMode,
                hidden_cards_seen=body.hiddenCardsSeen, final_result_seen=body.finalResultSeen,
                disclosure_timing=body.disclosureTiming,
            )
        except KeyError as exc:
            raise HTTPException(404, "review case or decision not found") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post("/cases/{case_id}/consensus")
    def consensus(case_id: str, request: Request):
        require_identity(request, identities)
        try:
            return service.compute_consensus(case_id)
        except KeyError as exc:
            raise HTTPException(404, "review case not found") from exc

    @router.post("/cases/{case_id}/moderation-flags", status_code=201)
    def flag(case_id: str, body: ModerationRequest, request: Request):
        identity, _ = require_identity(request, identities)
        _privileged(request, policy, Scope.MODEL_ADMIN)
        try:
            return service.open_moderation_flag(case_id, opened_by=identity.subject_id,
                submission_ids=body.submissionIds, code=body.code, severity=body.severity, details=body.details)
        except KeyError as exc:
            raise HTTPException(404, "review case not found") from exc

    @router.post("/moderation-flags/{flag_id}:resolve")
    def resolve_flag(flag_id: str, body: ResolveFlagRequest, request: Request):
        identity, _ = require_identity(request, identities)
        _privileged(request, policy, Scope.MODEL_ADMIN)
        try:
            return service.resolve_moderation_flag(flag_id, resolved_by=identity.subject_id,
                                                    resolution=body.resolution)
        except KeyError as exc:
            raise HTTPException(404, "moderation flag not found") from exc

    @router.post("/cases/{case_id}/adjudications", status_code=201)
    def adjudicate(case_id: str, body: AdjudicationRequest, request: Request):
        identity, _ = require_identity(request, identities)
        _privileged(request, policy, Scope.REFEREE)
        try:
            return service.adjudicate(case_id, adjudicator_id=identity.subject_id, outcome=body.outcome,
                                      preferred_action=body.preferredAction, rationale_ar=body.rationaleAr,
                                      certification_manifest=body.certificationManifest)
        except KeyError as exc:
            raise HTTPException(404, "review case not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.post("/cases/{case_id}/training-candidates", status_code=201)
    def training_candidate(case_id: str, request: Request):
        require_identity(request, identities)
        try:
            return service.create_training_candidate(case_id)
        except KeyError as exc:
            raise HTTPException(404, "review case not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.post("/training-candidates/{candidate_id}/evaluations", status_code=201)
    def challenger(candidate_id: str, body: ChallengerEvaluationRequest, request: Request):
        require_identity(request, identities); _privileged(request, policy, Scope.MODEL_ADMIN)
        try:
            return service.record_challenger_evaluation(candidate_id, validator_id=body.validatorId,
                validator_type=body.validatorType, status=body.status, metrics=body.metrics,
                independent=body.independent)
        except KeyError as exc:
            raise HTTPException(404, "training candidate not found") from exc

    @router.post("/training-candidates/{candidate_id}:accept")
    def accept_candidate(candidate_id: str, request: Request):
        identity, _ = require_identity(request, identities); _privileged(request, policy, Scope.MODEL_ADMIN)
        try:
            return service.accept_training_candidate(candidate_id, accepted_by=identity.subject_id)
        except KeyError as exc:
            raise HTTPException(404, "training candidate not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.get("/consent")
    def consent(request: Request):
        identity, _ = require_identity(request, identities)
        return service.get_consent(identity.subject_id)

    @router.put("/consent")
    def set_consent(body: ConsentRequest, request: Request):
        identity, _ = require_identity(request, identities)
        return service.set_consent(identity.subject_id, allow_training=body.allowTraining,
            allow_research=body.allowResearch, allow_public_challenge=body.allowPublicChallenge,
            allow_voice=body.allowVoice, allow_identity_attribution=body.allowIdentityAttribution,
            allow_private_coaching=body.allowPrivateCoaching, revoke=body.revoke)

    @router.get("/reviewers/{subject_id}")
    def reviewer(subject_id: str, request: Request):
        identity, _ = require_identity(request, identities)
        if subject_id != identity.subject_id:
            _privileged(request, policy, Scope.MODEL_ADMIN)
        return service.get_reviewer_profile(subject_id)

    @router.put("/reviewers/{subject_id}")
    def set_reviewer(subject_id: str, body: ReviewerProfileRequest, request: Request):
        require_identity(request, identities); _privileged(request, policy, Scope.MODEL_ADMIN)
        return service.set_reviewer_profile(subject_id, verified_role=body.verifiedRole,
            role_verified=body.roleVerified, legality_accuracy=body.legalityAccuracy,
            information_set_discipline=body.informationSetDiscipline,
            strategic_accuracy=body.strategicAccuracy, calibration=body.calibration,
            independence=body.independence, adjudicated_cases=body.adjudicatedCases)

    @router.post("/cases/{case_id}/review-clip", status_code=201)
    def review_clip(case_id: str, body: ClipRequest, request: Request):
        require_identity(request, identities)
        try:
            context = service.case(case_id, blind=True)
            artifacts = renderer.render(case=context, output_dir=output_root,
                basename=f"baloot-hitl-{case_id}", width=body.width, height=body.height,
                fps=body.fps, duration=body.durationSeconds)
            return {"videoUrl": f"/outputs/{artifacts.video.name}", "posterUrl": f"/outputs/{artifacts.poster.name}",
                    "timelineUrl": f"/outputs/{artifacts.timeline.name}", "manifestUrl": f"/outputs/{artifacts.manifest.name}"}
        except KeyError as exc:
            raise HTTPException(404, "review case not found") from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/improvements")
    def improvements(request: Request):
        require_identity(request, identities)
        return {"items": service.list_proposals(), "automaticProductionApply": False}

    @router.post("/improvements", status_code=201)
    def create_improvement(body: ImprovementRequest, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            return service.create_improvement_proposal(created_by=identity.subject_id, target=body.target,
                title_ar=body.titleAr, rationale_ar=body.rationaleAr, source_case_ids=body.sourceCaseIds,
                requested_change=body.requestedChange)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/improvements/{proposal_id}")
    def improvement(proposal_id: str, request: Request):
        require_identity(request, identities)
        try: return service.proposal(proposal_id)
        except KeyError as exc: raise HTTPException(404, "improvement proposal not found") from exc

    @router.post("/improvements/{proposal_id}/evaluations", status_code=201)
    def evaluate(proposal_id: str, body: ImprovementEvaluationRequest, request: Request):
        require_identity(request, identities); _privileged(request, policy, Scope.MODEL_ADMIN)
        try: return service.evaluate_improvement(proposal_id, validator_id=body.validatorId,
                    status=body.status, evidence=body.evidence, independent=body.independent)
        except KeyError as exc: raise HTTPException(404, "improvement proposal not found") from exc

    @router.post("/improvements/{proposal_id}:approve")
    def approve(proposal_id: str, request: Request):
        identity, _ = require_identity(request, identities)
        try: return service.approve_improvement(proposal_id, approved_by=identity.subject_id, role=_role(request))
        except KeyError as exc: raise HTTPException(404, "improvement proposal not found") from exc
        except (PermissionError, ValueError) as exc: raise HTTPException(403 if isinstance(exc, PermissionError) else 409, str(exc)) from exc

    @router.post("/improvements/{proposal_id}:finalize")
    def finalize(proposal_id: str, body: ChangeManifestRequest, request: Request):
        identity, _ = require_identity(request, identities); _privileged(request, policy, Scope.MODEL_ADMIN)
        try: return service.finalize_improvement(proposal_id, release_governor_id=identity.subject_id,
                                                  change_manifest=body.changeManifest)
        except KeyError as exc: raise HTTPException(404, "improvement proposal not found") from exc
        except ValueError as exc: raise HTTPException(409, str(exc)) from exc

    @router.get("/cases/{case_id}/verification")
    def verification(case_id: str, request: Request):
        require_identity(request, identities)
        try: return service.case(case_id, blind=False).get("verification")
        except KeyError as exc: raise HTTPException(404, "review case not found") from exc

    return router
