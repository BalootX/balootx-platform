from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, model_validator

from baloot_platform.core.enums import Seat
from baloot_platform.explain import ExplanationService
from baloot_platform.identity.ownership import owns_subject
from baloot_platform.identity import IdentityService
from baloot_platform.review import DecisionReviewService
from baloot_platform.sakka import SakkaService
from baloot_platform.shared import (
    SharedScope,
    observer_relation,
    private_decision_review_allowed,
    public_shared_environment,
)
from baloot_platform.storage.codec_v7 import decode

from .auth import require_identity


class CreateSharedBoardRequest(BaseModel):
    source: str = "new"
    gameId: str | None = None
    seed: int = 1
    dealer: str = "east"
    rulesProfile: str = "full"
    usScore: int = Field(0, ge=0, le=500)
    themScore: int = Field(0, ge=0, le=500)
    observerSeat: str = "south"
    scope: str = "laffa"
    maximumRounds: int = Field(16, ge=1, le=64)
    label: str | None = Field(default=None, max_length=80)
    visibility: str = "unlisted"
    agentProfiles: dict[str, str] | None = None

    @model_validator(mode="after")
    def validate_request(self):
        SharedScope.parse(self.scope)
        Seat.parse(self.dealer)
        Seat.parse(self.observerSeat)
        if self.source not in {"new", "game"}:
            raise ValueError("source must be new or game")
        if self.source == "game" and not self.gameId:
            raise ValueError("gameId is required when source=game")
        if self.rulesProfile not in {"full", "sun_only", "hokum_only"}:
            raise ValueError("unsupported rules profile")
        if self.visibility not in {"private", "unlisted", "public"}:
            raise ValueError("unsupported visibility")
        return self


class CreateSharedRoomRequest(BaseModel):
    boardIdentifier: str
    mode: str = "async"
    maxPlayers: int = Field(16, ge=2, le=200)
    allowGuests: bool = True
    settings: dict[str, Any] = Field(default_factory=dict)


class SharedActionRequest(BaseModel):
    kind: str
    action: str | None = None
    trumpSuit: str | None = None
    card: str | None = None
    ekka: bool = False

    @model_validator(mode="after")
    def validate_action(self):
        if self.kind not in {"bid", "play"}:
            raise ValueError("kind must be bid or play")
        if self.kind == "bid" and not self.action:
            raise ValueError("bid action is required")
        if self.kind == "play" and not self.card:
            raise ValueError("card is required")
        return self


class ShareSharedDecisionRequest(BaseModel):
    scope: str = "laffa"
    maximumRounds: int = Field(16, ge=1, le=64)
    label: str | None = Field(default=None, max_length=80)
    visibility: str = "unlisted"

    @model_validator(mode="after")
    def validate_scope(self):
        SharedScope.parse(self.scope)
        return self


class ReviewRequest(BaseModel):
    preferredAction: Any
    confidence: float = Field(0.8, ge=0.0, le=1.0)
    reasonCodes: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=3, max_length=3000)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    reviewerRole: str = "player"
    viewMode: str = "observer"


class CompareRequest(BaseModel):
    runA: str
    runB: str


def _owned_run(service: SakkaService, run_id: str, subject_id: str):
    loaded = service.store.get_run(run_id)
    if loaded is None:
        raise HTTPException(404, "shared run not found")
    run, payload = loaded
    if not owns_subject(subject_id, run.subject_id):
        raise HTTPException(403, "run belongs to another player")
    board_item = service.store.get_board(run.board_id)
    room = service.store.get_room(run.room_id)
    if board_item is None or room is None:
        raise HTTPException(500, "shared run references missing resources")
    return run, payload, board_item[0], room


def _board_payload(service: SakkaService, board, *, owner: bool) -> dict[str, Any]:
    result = board.to_dict(include_hashes=owner)
    item = service.store.get_board(board.board_id)
    if item is not None:
        try:
            session = decode(item[1], root=service.root)
            if session.shared_environment:
                result["sharedEnvironment"] = public_shared_environment(session.shared_environment)
        except Exception:
            result["sharedEnvironment"] = None
    result["scope"] = SharedScope.parse(board.scope).value
    result["scopeLegacy"] = SharedScope.parse(board.scope).legacy
    result["scopeLabelAr"] = SharedScope.parse(board.scope).label_ar
    return result


def build_shared_router(
    service: SakkaService,
    identities: IdentityService,
    explanations: ExplanationService,
    reviews: DecisionReviewService,
) -> APIRouter:
    router = APIRouter(prefix="/api/v1/shared", tags=["shared-laffa-sakka"])

    @router.get("/boards")
    def list_boards(request: Request, limit: int = 100):
        identity, _ = require_identity(request, identities)
        values = service.store.list_boards(subject_id=identity.subject_id, limit=limit)
        for item in values:
            item["scope"] = SharedScope.parse(item.get("scope", "round")).value
            item["scopeLegacy"] = SharedScope.parse(item.get("scopeLegacy", item.get("scope", "round"))).legacy
            item["scopeLabelAr"] = SharedScope.parse(item["scope"]).label_ar
        return {"items": values, "terminology": {"matchAr": "صكّة", "roundAr": "لفة", "trickAr": "أكلة"}}

    @router.post("/boards", status_code=201)
    def create_board(body: CreateSharedBoardRequest, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            scope = SharedScope.parse(body.scope)
            if body.source == "game":
                board = service.create_board_from_game(
                    body.gameId or "", created_by=identity.subject_id,
                    observer_seat=Seat.parse(body.observerSeat), scope=scope.value,
                    maximum_rounds=body.maximumRounds, label=body.label,
                    visibility=body.visibility,
                )
            else:
                board = service.create_board_from_new_game(
                    created_by=identity.subject_id, seed=body.seed, dealer=Seat.parse(body.dealer),
                    mode=body.rulesProfile, match_score=(body.usScore, body.themScore),
                    observer_seat=Seat.parse(body.observerSeat), scope=scope.value,
                    maximum_rounds=body.maximumRounds, label=body.label,
                    visibility=body.visibility, agent_profiles=body.agentProfiles,
                )
            return _board_payload(service, board, owner=True)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/boards/{identifier}")
    def get_board(identifier: str, request: Request):
        identity, _ = require_identity(request, identities)
        item = service.store.get_board(identifier)
        if item is None:
            raise HTTPException(404, "shared board not found")
        board = item[0]
        if board.visibility == "private" and not owns_subject(identity.subject_id, board.created_by):
            raise HTTPException(404, "shared board not found")
        return _board_payload(service, board, owner=owns_subject(identity.subject_id, board.created_by))

    @router.get("/boards/{identifier}/verification")
    def verify_board(identifier: str, request: Request):
        identity, _ = require_identity(request, identities)
        item = service.store.get_board(identifier)
        if item is None:
            raise HTTPException(404, "shared board not found")
        board = item[0]
        if board.visibility == "private" and not owns_subject(identity.subject_id, board.created_by):
            raise HTTPException(404, "shared board not found")
        return service.verify_board(identifier)

    @router.post("/rooms", status_code=201)
    def create_room(body: CreateSharedRoomRequest, request: Request):
        identity, _ = require_identity(request, identities)
        item = service.store.get_board(body.boardIdentifier)
        if item is None:
            raise HTTPException(404, "shared board not found")
        board = item[0]
        if board.visibility == "private" and not owns_subject(identity.subject_id, board.created_by):
            raise HTTPException(403, "only the board owner may create a room")
        try:
            room = service.open_room(
                body.boardIdentifier, created_by=identity.subject_id, mode=body.mode,
                max_players=body.maxPlayers, allow_guests=body.allowGuests, settings=body.settings,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"room": room.to_dict(), "shareUrl": f"/play?code={room.room_code}"}

    @router.get("/rooms/{identifier}")
    def get_room(identifier: str):
        try:
            return service.room(identifier)
        except KeyError as exc:
            raise HTTPException(404, "shared room not found") from exc

    @router.post("/rooms/{identifier}:join")
    def join_room(identifier: str, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            return service.join_room(identifier, subject_id=identity.subject_id,
                                     display_name=identity.display_name, is_guest=identity.is_guest)
        except KeyError as exc:
            raise HTTPException(404, "shared room not found") from exc
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.get("/rooms/{identifier}/events")
    def room_events(identifier: str, afterSequence: int = 0, limit: int = 200):
        try:
            return service.events(identifier, after_sequence=afterSequence, limit=limit)
        except KeyError as exc:
            raise HTTPException(404, "shared room not found") from exc

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
        room = service.store.get_room(identifier)
        if room is None:
            await websocket.close(code=4404, reason="room not found")
            return
        await websocket.accept()
        cursor = max(0, int(websocket.query_params.get("afterSequence", "0")))
        await websocket.send_json({"type": "connected", "roomId": room.room_id, "cursor": cursor})
        try:
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

    @router.post("/rooms/{identifier}/runs", status_code=201)
    def start_run(identifier: str, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            return service.start_run(identifier, subject_id=identity.subject_id,
                                     display_name=identity.display_name, is_guest=identity.is_guest)
        except KeyError as exc:
            raise HTTPException(404, "shared room not found") from exc
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.get("/runs/{run_id}")
    def reopen_run(run_id: str, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            return service.reopen_run(run_id, subject_id=identity.subject_id)
        except KeyError as exc:
            raise HTTPException(404, "shared run not found") from exc
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc

    @router.post("/runs/{run_id}/actions")
    def apply_action(run_id: str, body: SharedActionRequest, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            return service.apply_action(run_id, subject_id=identity.subject_id,
                                        action=body.model_dump(exclude_none=True))
        except KeyError as exc:
            raise HTTPException(404, "shared run not found") from exc
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/runs/{run_id}/verification")
    def verify_run(run_id: str, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            return service.verify_run(run_id, subject_id=identity.subject_id)
        except KeyError as exc:
            raise HTTPException(404, "shared run not found") from exc
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc

    @router.get("/runs/{run_id}/review")
    def review_package(run_id: str, request: Request, ensureAudit: bool = False):
        identity, _ = require_identity(request, identities)
        run, payload, board, room = _owned_run(service, run_id, identity.subject_id)
        session = decode(payload, root=service.root)
        actions = service.store.actions(run_id)
        sequence_by_decision = {str(item.get("decisionId")): int(item["sequence"])
                                for item in actions if item.get("decisionId")}
        decisions = service.games.learning_store.list_decisions(session.id, limit=10_000) if service.games.learning_store else []
        rows: list[dict[str, Any]] = []
        completed = run.status == "completed"
        for decision in decisions:
            relation = observer_relation(str(decision.get("actor")), board.observer_seat)
            can_review = private_decision_review_allowed(
                actor=str(decision.get("actor") or ""),
                observer=board.observer_seat,
                run_completed=completed,
            )
            explanation = explanations.explain(
                str(decision["decisionId"]), observer=board.observer_seat.name.lower(), level="basic",
                surface="replay", run_completed=completed, challenge_revealed=completed,
                view_mode="observer", ensure_audit=bool(ensureAudit and can_review),
            )
            rows.append({
                "decisionId": decision["decisionId"], "ordinal": decision["ordinal"],
                "decisionKind": decision["decisionKind"], "actor": decision["actor"],
                "actorRelation": relation, "selectedAction": decision.get("selectedAction"),
                "legalActions": (decision.get("legalActions") or []) if can_review else [],
                "actionSequence": sequence_by_decision.get(str(decision["decisionId"])),
                "reviewable": can_review, "simpleExplanation": explanation,
                "audits": (decision.get("audits") or []) if can_review else [],
                "reviewConsensus": (decision.get("reviewConsensus") or {}) if can_review else {},
            })
        return {
            "runId": run_id, "status": run.status, "scope": SharedScope.parse(board.scope).value,
            "scopeLabelAr": SharedScope.parse(board.scope).label_ar, "boardId": board.board_id,
            "boardSerial": board.serial, "boardHash": board.board_hash,
            "environmentHash": board.environment_hash, "decisions": rows,
            "terminology": {"matchAr": "صكّة", "roundAr": "لفة", "trickAr": "أكلة"},
        }

    @router.get("/runs/{run_id}/decisions/{decision_id}/explanation")
    def explain_decision(
        run_id: str, decision_id: str, request: Request,
        level: str = Query("basic", pattern="^(basic|coach|expert)$"),
        surface: str = Query("replay", pattern="^(active|replay|video|challenge)$"),
        challengeRevealed: bool = False, ensureAudit: bool = False,
        viewMode: str = Query("observer", pattern="^(observer|public|oracle_after_completion)$"),
    ):
        identity, _ = require_identity(request, identities)
        run, _payload, board, _room = _owned_run(service, run_id, identity.subject_id)
        decision = service.games.learning_store.get_decision(decision_id) if service.games.learning_store else None
        if decision is None:
            raise HTTPException(404, "decision not found")
        if str(decision.get("gameId")) not in {
            str(decode(service.store.get_run(run_id)[1], root=service.root).id),
        }:
            raise HTTPException(404, "decision does not belong to this run")
        completed = run.status == "completed"
        if viewMode == "oracle_after_completion" and not completed:
            raise HTTPException(403, "oracle explanation is available after completion only")
        return explanations.explain(
            decision_id, observer=board.observer_seat.name.lower(), level=level,
            surface=surface, run_completed=completed,
            challenge_revealed=bool(challengeRevealed and completed), view_mode=viewMode,
            ensure_audit=ensureAudit,
        )

    @router.get("/runs/{run_id}/decisions/{decision_id}/reviews")
    def review_context(run_id: str, decision_id: str, request: Request):
        identity, _ = require_identity(request, identities)
        run, payload, board, _room = _owned_run(service, run_id, identity.subject_id)
        session = decode(payload, root=service.root)
        decision = service.games.learning_store.get_decision(decision_id) if service.games.learning_store else None
        if decision is None or str(decision.get("gameId")) != str(session.id):
            raise HTTPException(404, "decision not found")
        if not private_decision_review_allowed(
            actor=str(decision.get("actor") or ""),
            observer=board.observer_seat,
            run_completed=run.status == "completed",
        ):
            raise HTTPException(403, "other players' reviews are available after completion")
        value = reviews.context(decision_id, run_id=run_id, view_mode="observer")
        consensus = dict(value.get("consensus") or {})
        consensus["agreement"] = consensus.get("supportShare", consensus.get("agreement", 0.0))
        value["consensus"] = consensus
        return value

    @router.post("/runs/{run_id}/decisions/{decision_id}/reviews", status_code=201)
    def submit_review(run_id: str, decision_id: str, body: ReviewRequest, request: Request):
        identity, _ = require_identity(request, identities)
        run, payload, board, _room = _owned_run(service, run_id, identity.subject_id)
        session = decode(payload, root=service.root)
        decision = service.games.learning_store.get_decision(decision_id) if service.games.learning_store else None
        if decision is None or str(decision.get("gameId")) != str(session.id):
            raise HTTPException(404, "decision not found")
        if not private_decision_review_allowed(
            actor=str(decision.get("actor") or ""),
            observer=board.observer_seat,
            run_completed=run.status == "completed",
        ):
            raise HTTPException(403, "other players' decisions can be reviewed after completion")
        try:
            value = reviews.submit(
                decision_id, reviewer_id=identity.subject_id,
                reviewer_role=(body.reviewerRole if identity.kind == "service" else "player"),
                preferred_action=body.preferredAction, confidence=body.confidence,
                reason_codes=body.reasonCodes or ["other"], rationale=body.rationale,
                evidence=body.evidence, view_mode=body.viewMode, source="decision_workbench",
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        consensus = dict(value.get("consensus") or {})
        consensus["agreement"] = consensus.get("supportShare", consensus.get("agreement", 0.0))
        value["consensus"] = consensus
        return value

    @router.post("/runs/{run_id}/actions/{sequence}:share", status_code=201)
    def share_from_action(run_id: str, sequence: int, body: ShareSharedDecisionRequest, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            board = service.create_board_from_run_action(
                run_id, sequence, created_by=identity.subject_id,
                scope=SharedScope.parse(body.scope).value, maximum_rounds=body.maximumRounds,
                label=body.label, visibility=body.visibility,
            )
        except KeyError as exc:
            raise HTTPException(404, "decision point not found") from exc
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        return _board_payload(service, board, owner=True)

    @router.get("/rooms/{identifier}/leaderboard")
    def leaderboard(identifier: str):
        try:
            return service.leaderboard(identifier)
        except KeyError as exc:
            raise HTTPException(404, "shared room not found") from exc

    @router.post("/comparisons")
    def compare(body: CompareRequest, request: Request):
        identity, _ = require_identity(request, identities)
        try:
            value = service.compare_runs(body.runA, body.runB, requester_subject_id=identity.subject_id)
            value["terminology"] = {"matchAr": "صكّة", "roundAr": "لفة", "trickAr": "أكلة"}
            return value
        except KeyError as exc:
            raise HTTPException(404, "shared run not found") from exc
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    return router
