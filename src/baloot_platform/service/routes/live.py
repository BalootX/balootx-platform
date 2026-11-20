from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, model_validator

from baloot_platform.core.enums import Seat
from baloot_platform.identity.ownership import owns_subject
from baloot_platform.identity import IdentityService
from baloot_platform.live import LiveTableService
from baloot_platform.service.game import GameManager
from baloot_platform.security import WebSocketTicketService

from .auth import require_identity


class CreateLiveRoomRequest(BaseModel):
    gameId: str | None = None
    seed: int = 1
    dealer: str = "east"
    rulesProfile: str = "full"
    usScore: int = Field(0, ge=0, le=1000)
    themScore: int = Field(0, ge=0, le=1000)
    turnSeconds: int = Field(45, ge=5, le=600)
    reconnectGraceSeconds: int = Field(120, ge=15, le=3600)
    allowSpectators: bool = True
    assistancePolicy: str = "none"

    @model_validator(mode="after")
    def validate_profile(self):
        Seat.parse(self.dealer)
        if self.rulesProfile not in {"full", "sun_only", "hokum_only"}:
            raise ValueError("unsupported rules profile")
        if self.assistancePolicy not in {"none", "post_action_own", "training"}:
            raise ValueError("unsupported assistance policy")
        return self


class ClaimSeatRequest(BaseModel):
    seat: str
    displayName: str = Field(min_length=1, max_length=40)

    @model_validator(mode="after")
    def validate_seat(self):
        Seat.parse(self.seat); return self


class ReadyRequest(BaseModel):
    ready: bool = True


class LiveActionRequest(BaseModel):
    kind: str
    action: str | None = None
    trumpSuit: str | None = None
    card: str | None = None
    ekka: bool = False
    expectedVersion: int | None = None

    @model_validator(mode="after")
    def validate_action(self):
        if self.kind not in {"bid", "play"}:
            raise ValueError("kind must be bid or play")
        if self.kind == "bid" and not self.action:
            raise ValueError("bid action is required")
        if self.kind == "play" and not self.card:
            raise ValueError("card is required")
        return self


class ReconnectRequest(BaseModel):
    reconnectToken: str = Field(min_length=20, max_length=200)


class SpectatorRequest(BaseModel):
    displayName: str = Field(min_length=1, max_length=40)


class PauseRequest(BaseModel):
    reason: str = Field(min_length=2, max_length=500)


class AssignRefereeRequest(BaseModel):
    refereeSubjectId: str = Field(min_length=3, max_length=100)


def build_live_router(service: LiveTableService, identities: IdentityService, games: GameManager, websocket_tickets: WebSocketTicketService) -> APIRouter:
    router = APIRouter(prefix="/api/v1/live", tags=["live-four-human"])

    @router.get("/rooms")
    def list_rooms(request: Request, status: str | None = None, limit: int = 100):
        identity, _ = require_identity(request, identities)
        return {"items": service.store.list(subject_id=identity.subject_id, status=status, limit=limit)}

    @router.post("/rooms", status_code=201)
    def create_room(body: CreateLiveRoomRequest, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            game = games.get(body.gameId) if body.gameId else games.create(
                seed=body.seed, dealer=Seat.parse(body.dealer), mode=body.rulesProfile,
                match_score=(body.usScore, body.themScore),
            )
            room = service.create_room(
                game_id=game.id, owner_subject_id=identity.subject_id,
                turn_seconds=body.turnSeconds, reconnect_grace_seconds=body.reconnectGraceSeconds,
                allow_spectators=body.allowSpectators, assistance_policy=body.assistancePolicy,
            )
            return {"room": room.to_dict(include_events=False), "joinCode": room.room_code, "gameId": game.id}
        except (KeyError, ValueError) as exc:
            raise HTTPException(422 if isinstance(exc, ValueError) else 404, str(exc)) from exc

    @router.get("/rooms/{identifier}")
    def get_room(identifier: str, request: Request, sinceSequence: int = 0):
        identity, _ = require_identity(request, identities)
        try:
            return service.view(identifier, subject_id=identity.subject_id, since_sequence=sinceSequence)
        except KeyError as exc:
            raise HTTPException(404, "live room not found") from exc

    @router.post("/rooms/{identifier}/seats", status_code=201)
    def claim_seat(identifier: str, body: ClaimSeatRequest, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            return service.claim_seat(
                identifier, subject_id=identity.subject_id,
                display_name=body.displayName or identity.display_name, seat=body.seat,
            )
        except KeyError as exc:
            raise HTTPException(404, "live room not found") from exc
        except (ValueError, PermissionError) as exc:
            raise HTTPException(409 if "occupied" in str(exc) else 422, str(exc)) from exc

    @router.post("/rooms/{identifier}/ready")
    def ready(identifier: str, body: ReadyRequest, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            return {"room": service.set_ready(identifier, subject_id=identity.subject_id, ready=body.ready).to_dict(include_events=False)}
        except KeyError as exc:
            raise HTTPException(404, "live room not found") from exc
        except (ValueError, PermissionError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post("/rooms/{identifier}/referee")
    def assign_referee(identifier: str, body: AssignRefereeRequest, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            return {"room": service.assign_referee(
                identifier, owner_subject_id=identity.subject_id,
                referee_subject_id=body.refereeSubjectId,
            ).to_dict(include_events=False)}
        except KeyError as exc:
            raise HTTPException(404, "live room not found") from exc
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc

    @router.post("/rooms/{identifier}/start")
    def start(identifier: str, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            return {"room": service.start(identifier, requested_by=identity.subject_id).to_dict(include_events=False)}
        except KeyError as exc:
            raise HTTPException(404, "live room not found") from exc
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.post("/rooms/{identifier}/actions")
    def action(identifier: str, body: LiveActionRequest, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            return service.submit_action(identifier, subject_id=identity.subject_id, action=body.model_dump(exclude_none=True))
        except KeyError as exc:
            raise HTTPException(404, "live room or game not found") from exc
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post("/rooms/{identifier}/disconnect")
    def disconnect(identifier: str, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            return {"room": service.disconnect(identifier, subject_id=identity.subject_id).to_dict(include_events=False)}
        except KeyError as exc:
            raise HTTPException(404, "live room not found") from exc

    @router.post("/rooms/{identifier}/reconnect")
    def reconnect(identifier: str, body: ReconnectRequest, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            return {"room": service.reconnect(
                identifier, subject_id=identity.subject_id, reconnect_token=body.reconnectToken,
            ).to_dict(include_events=False)}
        except KeyError as exc:
            raise HTTPException(404, "live room not found") from exc
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc

    @router.post("/rooms/{identifier}/spectators", status_code=201)
    def spectate(identifier: str, body: SpectatorRequest, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            return {"room": service.add_spectator(
                identifier, subject_id=identity.subject_id, display_name=body.displayName,
            ).to_dict(include_events=False)}
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc

    @router.post("/rooms/{identifier}/pause")
    def pause(identifier: str, body: PauseRequest, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            return {"room": service.pause(identifier, requested_by=identity.subject_id, reason=body.reason).to_dict(include_events=False)}
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.post("/rooms/{identifier}/resume")
    def resume(identifier: str, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            return {"room": service.resume(identifier, requested_by=identity.subject_id).to_dict(include_events=False)}
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.post("/rooms/{identifier}/deadline:enforce")
    def enforce_deadline(identifier: str, request: Request):
        identity, _ = require_identity(request, identities)
        room = service.store.get(identifier)
        if room is None:
            raise HTTPException(404, "live room not found")
        if not (
            owns_subject(identity.subject_id, room.owner_subject_id)
            or owns_subject(identity.subject_id, room.referee_subject_id)
        ):
            raise HTTPException(403, "only owner or referee may enforce deadlines")
        return service.enforce_deadline(identifier)

    @router.post("/rooms/{identifier}/disconnects:enforce")
    def enforce_disconnects(identifier: str, request: Request):
        identity, _ = require_identity(request, identities)
        room = service.store.get(identifier)
        if room is None:
            raise HTTPException(404, "live room not found")
        if not (
            owns_subject(identity.subject_id, room.owner_subject_id)
            or owns_subject(identity.subject_id, room.referee_subject_id)
        ):
            raise HTTPException(403, "only owner or referee may enforce reconnect grace")
        return service.enforce_reconnect_grace(identifier)

    @router.get("/rooms/{identifier}/verification")
    def verify(identifier: str, request: Request):
        require_identity(request, identities)
        try:
            return service.verify(identifier)
        except KeyError as exc:
            raise HTTPException(404, "live room not found") from exc

    @router.websocket("/rooms/{identifier}/events")
    async def events(identifier: str, websocket: WebSocket):
        # Never accept long-lived access tokens in the URL. Clients exchange
        # their authenticated HTTPS session for a single-use room-bound ticket.
        ticket = websocket.query_params.get("ticket")
        if not ticket or websocket.query_params.get("access_token"):
            await websocket.close(code=4401); return
        try:
            claims = websocket_tickets.consume(ticket, room_id=identifier)
        except ValueError:
            await websocket.close(code=4401); return
        identity = identities.resolve_subject(claims.subject_id)
        if identity is None:
            await websocket.close(code=4401); return
        origin = websocket.headers.get("origin")
        configured_origin = __import__("os").environ.get("BALOOT_WEBAUTHN_ORIGIN", "").rstrip("/")
        if configured_origin and origin and origin.rstrip("/") != configured_origin:
            await websocket.close(code=4403); return
        await websocket.accept()
        sequence = int(websocket.query_params.get("since", "0"))
        try:
            while True:
                try:
                    value = service.view(identifier, subject_id=identity.subject_id, since_sequence=sequence)
                except KeyError:
                    await websocket.close(code=4404); return
                if value["events"]:
                    sequence = value["nextSequence"]
                    await websocket.send_json(value)
                await service.broker.wait(value["room"]["roomId"], sequence, timeout=5.0)
        except WebSocketDisconnect:
            return

    return router
