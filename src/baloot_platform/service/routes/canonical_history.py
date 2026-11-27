from __future__ import annotations

from collections.abc import Callable
from typing import Any

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


def _lineage_fields(record: dict[str, Any], *fields: str) -> dict[str, Any]:
    return {field: record[field] for field in fields}


def build_canonical_history_router(
    store: CanonicalHistoryStore,
    identities: IdentityService,
    games: GameManager,
    *,
    require_trusted: Callable[[Request], None],
) -> APIRouter:
    router = APIRouter(prefix="/api/v1/canonical-history", tags=["canonical-history"])

    def current_stream(game_id: str) -> str | None:
        try:
            game = games.get(game_id)
        except KeyError:
            return None
        return f"{game.id}:round:{game.state.round_number}"

    def stream_identity(identifier: str) -> tuple[str, str | None]:
        current = current_stream(identifier)
        if current is not None:
            return current, identifier
        game_id, separator, _ = identifier.rpartition(":round:")
        # Only the current stream can inherit a game-level destination row.
        if separator and current_stream(game_id) == identifier:
            return identifier, game_id
        return identifier, None

    @router.get("/{identifier}/verification")
    def verify(identifier: str, request: Request):
        require_trusted(request)
        stream_id, destination_alias = stream_identity(identifier)
        return store.verify(stream_id, destination_alias=destination_alias)

    @router.get("/{identifier}/snapshot")
    def snapshot(identifier: str, request: Request):
        require_trusted(request)
        value = store.latest_snapshot(current_stream(identifier) or identifier)
        if value is None:
            raise HTTPException(404, "canonical snapshot not found")
        return value.to_dict()

    @router.get("/{identifier}/lineage")
    def lineage(identifier: str, request: Request):
        require_identity(request, identities)
        stream_id, destination_alias = stream_identity(identifier)
        record = store.lineage(stream_id, destination_alias=destination_alias)
        parent = record["parentBranch"]
        correction = record["parentCorrection"]
        return {
            "gameId": record["gameId"],
            "parentBranch": _lineage_fields(
                parent, "branch_id", "parent_game_id", "parent_event_count", "parent_head_hash",
                "child_game_id", "created_at",
            ) if parent is not None else None,
            "parentCorrection": _lineage_fields(
                correction, "correction_id", "source_game_id", "source_event_count", "source_head_hash",
                "corrected_game_id", "created_at",
            ) if correction is not None else None,
            "branches": [
                _lineage_fields(child, "child_game_id", "branch_id", "created_at")
                for child in record["branches"]
            ],
            "corrections": [
                _lineage_fields(child, "corrected_game_id", "correction_id", "created_at")
                for child in record["corrections"]
            ],
        }

    @router.post("/{identifier}/branches", status_code=201)
    def branch(identifier: str, body: BranchRequest, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            return store.create_branch(
                parent_game_id=current_stream(identifier) or identifier, child_game_id=body.childGameId,
                created_by=identity.subject_id, reason=body.reason,
            ).to_dict()
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.post("/{identifier}/corrections", status_code=201)
    def correction(identifier: str, body: CorrectionRequest, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            return store.create_correction(
                source_game_id=current_stream(identifier) or identifier, corrected_game_id=body.correctedGameId,
                created_by=identity.subject_id, reason=body.reason,
                adjudication_case_id=body.adjudicationCaseId,
            ).to_dict()
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    return router
