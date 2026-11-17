from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from baloot_platform.history.canonical_store_v17 import CanonicalHistoryStore
from baloot_platform.identity import IdentityService
from baloot_platform.service.game import GameManager

from .auth import require_identity


class BranchRequest(BaseModel):
    childGameId: str = Field(min_length=1, max_length=200)
    reason: str = Field(min_length=2, max_length=1000)


class CorrectionRequest(BaseModel):
    correctedGameId: str = Field(min_length=1, max_length=200)
    reason: str = Field(min_length=2, max_length=1000)
    adjudicationCaseId: str | None = None


def build_canonical_history_router(store: CanonicalHistoryStore, identities: IdentityService,
                                   games: GameManager) -> APIRouter:
    router = APIRouter(prefix="/api/v1/canonical-history", tags=["canonical-history"])

    def stream(identifier: str) -> str:
        try:
            game = games.get(identifier)
            return f"{game.id}:round:{game.state.round_number}"
        except KeyError:
            return identifier

    @router.get("/{identifier}/verification")
    def verify(identifier: str, request: Request):
        require_identity(request, identities)
        return store.verify(stream(identifier))

    @router.get("/{identifier}/snapshot")
    def snapshot(identifier: str, request: Request):
        require_identity(request, identities)
        value = store.latest_snapshot(stream(identifier))
        if value is None:
            raise HTTPException(404, "canonical snapshot not found")
        return value.to_dict()

    @router.get("/{identifier}/lineage")
    def lineage(identifier: str, request: Request):
        require_identity(request, identities)
        return store.lineage(stream(identifier))

    @router.post("/{identifier}/branches", status_code=201)
    def branch(identifier: str, body: BranchRequest, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            return store.create_branch(
                parent_game_id=stream(identifier), child_game_id=body.childGameId,
                created_by=identity.subject_id, reason=body.reason,
            ).to_dict()
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.post("/{identifier}/corrections", status_code=201)
    def correction(identifier: str, body: CorrectionRequest, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            return store.create_correction(
                source_game_id=stream(identifier), corrected_game_id=body.correctedGameId,
                created_by=identity.subject_id, reason=body.reason,
                adjudication_case_id=body.adjudicationCaseId,
            ).to_dict()
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    return router
