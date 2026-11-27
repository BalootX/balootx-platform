from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from baloot_platform.adjudication import AdjudicationService, RulingOutcome, ViolationCode
from baloot_platform.identity import IdentityService
from baloot_platform.security import AuthorizationPolicy, Scope

from .auth import require_identity


class OpenCaseRequest(BaseModel):
    gameId: str
    violation: str
    description: str = Field(min_length=3, max_length=4000)
    roundNumber: int | None = Field(default=None, ge=1)
    eventSequence: int | None = Field(default=None, ge=0)
    accusedSubjectId: str | None = None
    accusedSeat: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceRequest(BaseModel):
    kind: str
    description: str = Field(min_length=2, max_length=4000)
    uri: str | None = None
    sha256: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RulingRequest(BaseModel):
    outcome: str
    reasonAr: str = Field(min_length=3, max_length=4000)
    pointsDelta: tuple[int, int] = (0, 0)
    replayFromEventSequence: int | None = Field(default=None, ge=0)
    final: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class AppealRequest(BaseModel):
    reason: str = Field(min_length=3, max_length=4000)


def _role(request: Request) -> str:
    claims = getattr(request.state, "access_claims", None)
    if claims is not None:
        return str(getattr(claims, "role", "player")).strip().lower()
    configured = request.headers.get("x-baloot-role", "player").strip().lower()
    if configured in {"admin", "referee", "service"} and not bool(getattr(request.state, "api_key_valid", False)):
        return "player"
    return configured


def build_adjudication_router(service: AdjudicationService, identities: IdentityService,
                              policy: AuthorizationPolicy) -> APIRouter:
    router = APIRouter(prefix="/api/v1/adjudication", tags=["adjudication"])

    @router.get("/cases")
    def list_cases(request: Request, gameId: str | None = None, status: str | None = None, limit: int = 100):
        require_identity(request, identities)
        return {"items": service.store.list(game_id=gameId, status=status, limit=limit)}

    @router.post("/cases", status_code=201)
    def open_case(body: OpenCaseRequest, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            case = service.open_case(
                game_id=body.gameId, violation=ViolationCode(body.violation), reported_by=identity.subject_id,
                description=body.description, round_number=body.roundNumber, event_sequence=body.eventSequence,
                accused_subject_id=body.accusedSubjectId, accused_seat=body.accusedSeat, metadata=body.metadata,
            )
            return case.to_dict()
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/cases/{case_id}")
    def get_case(case_id: str, request: Request):
        require_identity(request, identities)
        case = service.store.get(case_id)
        if case is None:
            raise HTTPException(404, "adjudication case not found")
        return case.to_dict()

    @router.post("/cases/{case_id}/evidence", status_code=201)
    def add_evidence(case_id: str, body: EvidenceRequest, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            return service.add_evidence(
                case_id, kind=body.kind, submitted_by=identity.subject_id, description=body.description,
                uri=body.uri, sha256=body.sha256, metadata=body.metadata,
            ).to_dict()
        except KeyError as exc:
            raise HTTPException(404, "adjudication case not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.get("/cases/{case_id}/recommendation")
    def recommendation(case_id: str, request: Request):
        require_identity(request, identities)
        try:
            return service.recommended_ruling(case_id)
        except KeyError as exc:
            raise HTTPException(404, "adjudication case not found") from exc

    @router.post("/cases/{case_id}/rulings", status_code=201)
    def ruling(case_id: str, body: RulingRequest, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            policy.require(
                role=_role(request), required=Scope.REFEREE,
                claims=getattr(request.state, "access_claims", None),
            )
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        try:
            return service.rule(
                case_id, referee_id=identity.subject_id, outcome=RulingOutcome(body.outcome),
                reason_ar=body.reasonAr, points_delta=body.pointsDelta,
                replay_from_event_sequence=body.replayFromEventSequence, final=body.final, metadata=body.metadata,
            ).to_dict()
        except KeyError as exc:
            raise HTTPException(404, "adjudication case not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.post("/cases/{case_id}/appeals", status_code=201)
    def appeal(case_id: str, body: AppealRequest, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            return service.appeal(case_id, appealed_by=identity.subject_id, reason=body.reason).to_dict()
        except KeyError as exc:
            raise HTTPException(404, "adjudication case not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.post("/cases/{case_id}/appeals:resolve")
    def resolve(case_id: str, body: RulingRequest, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            policy.require(
                role=_role(request), required=Scope.REFEREE,
                claims=getattr(request.state, "access_claims", None),
            )
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        try:
            return service.resolve_appeal(
                case_id, referee_id=identity.subject_id, outcome=body.outcome, reason_ar=body.reasonAr,
                points_delta=body.pointsDelta, replay_from_event_sequence=body.replayFromEventSequence,
                metadata=body.metadata,
            ).to_dict()
        except KeyError as exc:
            raise HTTPException(404, "adjudication case not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.get("/cases/{case_id}/verification")
    def verify(case_id: str, request: Request):
        require_identity(request, identities)
        try:
            return service.verify(case_id)
        except KeyError as exc:
            raise HTTPException(404, "adjudication case not found") from exc

    @router.get("/cases/{case_id}/directive")
    def directive(case_id: str, request: Request):
        require_identity(request, identities)
        try:
            return service.directive(case_id)
        except KeyError as exc:
            raise HTTPException(404, "adjudication case not found") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    return router
