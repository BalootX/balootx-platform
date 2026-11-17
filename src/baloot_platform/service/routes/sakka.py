from __future__ import annotations

import asyncio
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, model_validator

from baloot_platform.core.enums import Seat
from baloot_platform.identity import IdentityService
from baloot_platform.sakka import SakkaService
from .auth import require_identity


class CreateSakkaBoardRequest(BaseModel):
    source: Literal["new", "game"] = "new"
    gameId: str | None = None
    seed: int = 1
    dealer: Literal["north", "west", "south", "east"] = "east"
    rulesProfile: Literal["full", "sun_only", "hokum_only"] = "full"
    usScore: int = Field(0, ge=0, le=500)
    themScore: int = Field(0, ge=0, le=500)
    observerSeat: Literal["north", "west", "south", "east"] = "south"
    scope: Literal["round", "match"] = "round"
    maximumRounds: int = Field(16, ge=1, le=64)
    label: str | None = Field(default=None, max_length=80)
    visibility: Literal["private", "unlisted", "public"] = "unlisted"
    agentProfiles: dict[str, str] | None = None

    @model_validator(mode="after")
    def validate_source(self):
        if self.source == "game" and not self.gameId:
            raise ValueError("gameId is required when source=game")
        return self


class CreateRoomRequest(BaseModel):
    boardIdentifier: str
    mode: Literal["async", "live", "practice"] = "async"
    maxPlayers: int = Field(16, ge=2, le=200)
    allowGuests: bool = True
    settings: dict[str, Any] = Field(default_factory=dict)


class SakkaActionRequest(BaseModel):
    kind: Literal["bid", "play"]
    action: str | None = None
    trumpSuit: str | None = None
    card: str | None = None
    ekka: bool = False

    @model_validator(mode="after")
    def validate_action(self):
        if self.kind == "bid" and not self.action:
            raise ValueError("bid action is required")
        if self.kind == "play" and not self.card:
            raise ValueError("card is required")
        return self


class ShareDecisionRequest(BaseModel):
    scope: Literal["round", "match"] = "round"
    maximumRounds: int = Field(16, ge=1, le=64)
    label: str | None = Field(default=None, max_length=80)
    visibility: Literal["private", "unlisted", "public"] = "unlisted"


class AuditRunRequest(BaseModel):
    deep: bool = False
    force: bool = False
    maximumDecisions: int = Field(32, ge=1, le=256)


class RunFeedbackRequest(BaseModel):
    preferredAction: Any
    confidence: float = Field(0.8, ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=2000)
    interestingness: float = Field(0.0, ge=0.0, le=1.0)


class CompareRunsRequest(BaseModel):
    runA: str = Field(min_length=1, max_length=80)
    runB: str = Field(min_length=1, max_length=80)


def build_sakka_router(service: SakkaService, identities: IdentityService) -> APIRouter:
    router = APIRouter(prefix="/api/v1/sakka", tags=["sakka"])

    @router.get("/boards")
    def list_boards(request: Request, limit: int = 100):
        identity, _ = require_identity(request, identities)
        return {"items": service.store.list_boards(subject_id=identity.subject_id, limit=limit)}

    @router.post("/boards", status_code=201)
    def create_board(body: CreateSakkaBoardRequest, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            if body.source == "game":
                board = service.create_board_from_game(
                    body.gameId or "", created_by=identity.subject_id,
                    observer_seat=Seat.parse(body.observerSeat), scope=body.scope,
                    maximum_rounds=body.maximumRounds, label=body.label,
                    visibility=body.visibility,
                )
            else:
                board = service.create_board_from_new_game(
                    created_by=identity.subject_id, seed=body.seed, dealer=Seat.parse(body.dealer),
                    mode=body.rulesProfile, match_score=(body.usScore, body.themScore),
                    observer_seat=Seat.parse(body.observerSeat), scope=body.scope,
                    maximum_rounds=body.maximumRounds, label=body.label,
                    visibility=body.visibility, agent_profiles=body.agentProfiles,
                )
            return board.to_dict()
        except (ValueError, KeyError) as exc:
            raise HTTPException(422 if isinstance(exc, ValueError) else 404, str(exc)) from exc

    @router.get("/boards/{identifier}")
    def get_board(identifier: str, request: Request):
        identity, _ = require_identity(request, identities)
        item = service.store.get_board(identifier)
        if item is None:
            raise HTTPException(404, "Sakka board not found")
        board = item[0]
        if board.visibility == "private" and board.created_by != identity.subject_id:
            raise HTTPException(404, "Sakka board not found")
        return board.to_dict(include_hashes=board.created_by == identity.subject_id)

    @router.get("/boards/{identifier}/verification")
    def verify_board(identifier: str, request: Request):
        identity, _ = require_identity(request, identities)
        item = service.store.get_board(identifier)
        if item is None:
            raise HTTPException(404, "Sakka board not found")
        board = item[0]
        if board.visibility == "private" and board.created_by != identity.subject_id:
            raise HTTPException(404, "Sakka board not found")
        return service.verify_board(identifier)

    @router.post("/rooms", status_code=201)
    def create_room(body: CreateRoomRequest, request: Request):
        identity, _ = require_identity(request, identities)
        item = service.store.get_board(body.boardIdentifier)
        if item is None:
            raise HTTPException(404, "Sakka board not found")
        board = item[0]
        if board.visibility == "private" and board.created_by != identity.subject_id:
            raise HTTPException(403, "only the board owner may create a room")
        try:
            room = service.open_room(
                body.boardIdentifier, created_by=identity.subject_id, mode=body.mode,
                max_players=body.maxPlayers, allow_guests=body.allowGuests,
                settings=body.settings,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"room": room.to_dict(), "shareUrl": f"/play?code={room.room_code}"}

    @router.get("/rooms/{identifier}")
    def get_room(identifier: str):
        try:
            return service.room(identifier)
        except KeyError as exc:
            raise HTTPException(404, "Sakka room not found") from exc

    @router.get("/rooms/{identifier}/events")
    def room_events(identifier: str, afterSequence: int = 0, limit: int = 200):
        try:
            return service.events(identifier, after_sequence=afterSequence, limit=limit)
        except KeyError as exc:
            raise HTTPException(404, "Sakka room not found") from exc

    @router.websocket("/rooms/{identifier}/stream")
    async def room_stream(websocket: WebSocket, identifier: str):
        token = websocket.cookies.get("baloot_session")
        authorization = websocket.headers.get("authorization", "")
        if authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        identity, _session = identities.resolve(token)
        if identity is None:
            await websocket.close(code=4401, reason="authentication required")
            return
        try:
            room = service.store.get_room(identifier)
            if room is None:
                await websocket.close(code=4404, reason="room not found")
                return
            await websocket.accept()
            cursor = max(0, int(websocket.query_params.get("afterSequence", "0")))
            await websocket.send_json({"type": "connected", "roomId": room.room_id, "cursor": cursor})
            while True:
                events = service.store.room_events(room.room_id, after_sequence=cursor, limit=200)
                if events:
                    cursor = max(int(item["sequence"]) for item in events)
                    await websocket.send_json({"type": "events", "cursor": cursor, "events": events})
                try:
                    message = await asyncio.wait_for(websocket.receive_json(), timeout=0.75)
                    if message.get("type") == "ping":
                        await websocket.send_json({"type": "pong", "cursor": cursor})
                except asyncio.TimeoutError:
                    continue
        except WebSocketDisconnect:
            return

    @router.post("/rooms/{identifier}:join")
    def join_room(identifier: str, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            return service.join_room(
                identifier, subject_id=identity.subject_id,
                display_name=identity.display_name, is_guest=identity.is_guest,
            )
        except KeyError as exc:
            raise HTTPException(404, "Sakka room not found") from exc
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.post("/rooms/{identifier}/runs", status_code=201)
    def start_run(identifier: str, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            return service.start_run(
                identifier, subject_id=identity.subject_id,
                display_name=identity.display_name, is_guest=identity.is_guest,
            )
        except KeyError as exc:
            raise HTTPException(404, "Sakka room not found") from exc
        except (PermissionError, ValueError) as exc:
            raise HTTPException(403 if isinstance(exc, PermissionError) else 409, str(exc)) from exc

    @router.get("/runs/{run_id}")
    def reopen_run(run_id: str, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            return service.reopen_run(run_id, subject_id=identity.subject_id)
        except KeyError as exc:
            raise HTTPException(404, "Sakka run not found") from exc
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc

    @router.post("/runs/{run_id}/actions")
    def apply_action(run_id: str, body: SakkaActionRequest, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            return service.apply_action(run_id, subject_id=identity.subject_id, action=body.model_dump(exclude_none=True))
        except KeyError as exc:
            raise HTTPException(404, "Sakka run not found") from exc
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post("/runs/{run_id}/actions/{sequence}:share", status_code=201)
    def share_from_action(run_id: str, sequence: int, body: ShareDecisionRequest, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            board = service.create_board_from_run_action(
                run_id, sequence, created_by=identity.subject_id, scope=body.scope,
                maximum_rounds=body.maximumRounds, label=body.label,
                visibility=body.visibility,
            )
        except KeyError as exc:
            raise HTTPException(404, "Sakka decision point not found") from exc
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        return board.to_dict()

    @router.get("/runs/{run_id}/audit")
    def get_run_audit(run_id: str, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            value = service.get_run_audit(run_id, subject_id=identity.subject_id)
        except KeyError:
            raise HTTPException(404, "Sakka run not found")
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        if value is None:
            raise HTTPException(404, "run audit has not been generated")
        return value

    @router.get("/runs/{run_id}/verification")
    def verify_run(run_id: str, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            return service.verify_run(run_id, subject_id=identity.subject_id)
        except KeyError as exc:
            raise HTTPException(404, "Sakka run not found") from exc
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc

    @router.post("/runs/{run_id}:audit")
    def audit_run(run_id: str, body: AuditRunRequest, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            return service.audit_run(
                run_id, subject_id=identity.subject_id, deep=body.deep,
                force=body.force, maximum_decisions=body.maximumDecisions,
            )
        except KeyError as exc:
            raise HTTPException(404, "Sakka run not found") from exc
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc

    @router.post("/runs/{run_id}/actions/{sequence}/feedback", status_code=201)
    def run_feedback(run_id: str, sequence: int, body: RunFeedbackRequest, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            return service.feedback(
                run_id, sequence, subject_id=identity.subject_id,
                preferred_action=body.preferredAction, confidence=body.confidence,
                rationale=body.rationale, interestingness=body.interestingness,
            )
        except KeyError as exc:
            raise HTTPException(404, "Sakka action or decision not found") from exc
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc

    @router.get("/rooms/{identifier}/leaderboard")
    def leaderboard(identifier: str):
        try:
            return service.leaderboard(identifier)
        except KeyError as exc:
            raise HTTPException(404, "Sakka room not found") from exc

    @router.post("/comparisons")
    def compare_runs(body: CompareRunsRequest, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            return service.compare_runs(
                body.runA, body.runB, requester_subject_id=identity.subject_id,
            )
        except KeyError as exc:
            raise HTTPException(404, "Sakka run not found") from exc
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    return router
