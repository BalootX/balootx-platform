from __future__ import annotations

import hashlib
import secrets
import threading
from datetime import datetime, timezone
from typing import Any, Mapping, TYPE_CHECKING

from baloot_platform.core.cards import Card, iter_mask
from baloot_platform.identity.ownership import owns_subject
from baloot_platform.core.enums import Seat, Suit
from baloot_platform.service.game import GameManager

from .broker import InMemoryLiveEventBroker, LiveEventBroker
from .models import LiveRoom, LiveStatus, SeatAssignment, future_time, utc_now
from .store import SQLiteLiveStore

if TYPE_CHECKING:
    from baloot_platform.adjudication import AdjudicationService


class LiveTableService:
    """Server-authoritative four-human Baloot table.

    Shared identical-start competition remains a separate product. This service
    owns one synchronous table where each connected subject controls exactly one
    seat. It delegates all legal transitions to GameManager/BalootEngine.
    """

    def __init__(self, store: SQLiteLiveStore, games: GameManager, broker: LiveEventBroker | None = None,
                 adjudication: "AdjudicationService | None" = None):
        self.store = store
        self.games = games
        self.broker = broker or InMemoryLiveEventBroker()
        self.adjudication = adjudication
        self._lock = threading.RLock()

    def _save(self, room: LiveRoom) -> None:
        expected = room.version if room.version > 0 else None
        self.store.save(room, expected_version=expected)
        sequence = room.events[-1].sequence if room.events else 0
        self.broker.publish(room.room_id, sequence)

    def _open_adjudication(self, room: LiveRoom, *, violation: str, description: str,
                           accused_subject_id: str | None = None, accused_seat: Seat | None = None) -> str | None:
        if self.adjudication is None:
            return None
        case = self.adjudication.open_case(
            game_id=room.game_id, violation=violation, reported_by=room.referee_subject_id or room.owner_subject_id,
            description=description, event_sequence=room.events[-1].sequence if room.events else None,
            accused_subject_id=accused_subject_id,
            accused_seat=accused_seat.name.lower() if accused_seat is not None else None,
            metadata={"liveRoomId": room.room_id, "liveRoomHash": room.room_hash},
        )
        return case.case_id

    @staticmethod
    def _hash_token(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    def create_room(self, *, game_id: str, owner_subject_id: str, turn_seconds: int = 45,
                    reconnect_grace_seconds: int = 120, allow_spectators: bool = True,
                    assistance_policy: str = "none", metadata: Mapping[str, Any] | None = None) -> LiveRoom:
        self.games.get(game_id)
        room = LiveRoom.create(
            game_id=game_id, owner_subject_id=owner_subject_id, turn_seconds=turn_seconds,
            reconnect_grace_seconds=reconnect_grace_seconds, allow_spectators=allow_spectators,
            assistance_policy=assistance_policy, metadata=metadata,
        )
        room.append_event("ROOM_CREATED", subject_id=owner_subject_id, payload={"gameId": game_id})
        self._save(room)
        return room

    def claim_seat(self, identifier: str, *, subject_id: str, display_name: str, seat: Seat | str) -> dict[str, Any]:
        with self._lock:
            room = self._require(identifier)
            if room.status != LiveStatus.WAITING:
                raise ValueError("seats may be claimed only before the table starts")
            parsed = Seat.parse(seat)
            key = parsed.name.lower()
            current = room.seats.get(key)
            if current and not owns_subject(subject_id, current.subject_id):
                raise ValueError("seat is already occupied")
            for other_key, assignment in list(room.seats.items()):
                if owns_subject(subject_id, assignment.subject_id) and other_key != key:
                    del room.seats[other_key]
            token = secrets.token_urlsafe(32)
            room.seats[key] = SeatAssignment(
                parsed, subject_id, display_name.strip()[:40] or "لاعب",
                reconnect_token_hash=self._hash_token(token),
            )
            room.append_event("SEAT_CLAIMED", subject_id=subject_id, seat=parsed, payload={"displayName": display_name[:40]})
            self._save(room)
            return {"room": room.to_dict(), "reconnectToken": token}

    def set_ready(self, identifier: str, *, subject_id: str, ready: bool = True) -> LiveRoom:
        with self._lock:
            room = self._require(identifier)
            assignment = self._assignment(room, subject_id)
            if room.status != LiveStatus.WAITING:
                raise ValueError("ready status is locked after start")
            assignment.ready = bool(ready); assignment.last_seen_at = utc_now()
            room.append_event("PLAYER_READY" if ready else "PLAYER_NOT_READY", subject_id=subject_id, seat=assignment.seat)
            self._save(room)
            return room

    def assign_referee(self, identifier: str, *, owner_subject_id: str, referee_subject_id: str) -> LiveRoom:
        room = self._require(identifier)
        if not owns_subject(owner_subject_id, room.owner_subject_id):
            raise PermissionError("only the room owner may assign the referee")
        room.referee_subject_id = referee_subject_id
        room.append_event("REFEREE_ASSIGNED", subject_id=owner_subject_id, payload={"refereeSubjectId": referee_subject_id})
        self._save(room)
        return room

    def start(self, identifier: str, *, requested_by: str) -> LiveRoom:
        with self._lock:
            room = self._require(identifier)
            if not (owns_subject(requested_by, room.owner_subject_id) or owns_subject(requested_by, room.referee_subject_id)):
                raise PermissionError("only owner or referee may start")
            if room.status != LiveStatus.WAITING:
                raise ValueError("room is not waiting")
            if set(room.seats) != {seat.name.lower() for seat in Seat}:
                raise ValueError("all four seats must be occupied")
            if not all(item.ready and item.connected for item in room.seats.values()):
                raise ValueError("all four players must be ready and connected")
            room.status = LiveStatus.ACTIVE
            room.turn_deadline = future_time(room.turn_seconds)
            room.append_event("TABLE_STARTED", subject_id=requested_by, payload={"turnDeadline": room.turn_deadline})
            self._save(room)
            return room

    def submit_action(self, identifier: str, *, subject_id: str, action: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            room = self._require(identifier)
            if room.status != LiveStatus.ACTIVE:
                raise ValueError("table is not active")
            assignment = self._assignment(room, subject_id)
            game = self.games.get(room.game_id)
            actor = game.bidding.actor if game.bidding is not None and not game.bidding.finished else game.state.current_player
            if actor is None:
                room.status = LiveStatus.COMPLETED
                room.turn_deadline = None
                room.append_event("TABLE_COMPLETED", payload={"gameVersion": game.version})
                self._save(room)
                return self.view(room.room_id, subject_id=subject_id)
            if actor != assignment.seat:
                raise PermissionError("it is another seat's turn")
            kind = str(action.get("kind") or "")
            if kind == "bid":
                trump = Suit.parse(action["trumpSuit"]) if action.get("trumpSuit") else None
                game = self.games.bid(
                    game.id, seat=assignment.seat, action=str(action.get("action")), trump=trump,
                    expected_version=game.version if action.get("expectedVersion") is None else int(action["expectedVersion"]),
                    decision_metadata={"schemaVersion": 1, "source": "human_live", "roomId": room.room_id},
                )
            elif kind == "play":
                game = self.games.play(
                    game.id, seat=assignment.seat, card=Card.parse(action.get("card")),
                    ekka=bool(action.get("ekka", False)),
                    expected_version=game.version if action.get("expectedVersion") is None else int(action["expectedVersion"]),
                    decision_metadata={"schemaVersion": 1, "source": "human_live", "roomId": room.room_id},
                )
            else:
                raise ValueError("action kind must be bid or play")
            room.turn_deadline = future_time(room.turn_seconds)
            room.append_event("ACTION_APPLIED", subject_id=subject_id, seat=assignment.seat, payload={
                "action": dict(action), "gameVersion": game.version, "turnDeadline": room.turn_deadline,
            })
            if game.state.current_player is None and (game.bidding is None or game.bidding.finished):
                room.status = LiveStatus.COMPLETED; room.turn_deadline = None
                room.append_event("TABLE_COMPLETED", payload={"gameVersion": game.version})
            self._save(room)
            return self.view(room.room_id, subject_id=subject_id)

    def disconnect(self, identifier: str, *, subject_id: str) -> LiveRoom:
        room = self._require(identifier); assignment = self._assignment(room, subject_id)
        assignment.connected = False; assignment.last_seen_at = utc_now()
        room.append_event("PLAYER_DISCONNECTED", subject_id=subject_id, seat=assignment.seat)
        self._save(room); return room

    def reconnect(self, identifier: str, *, subject_id: str, reconnect_token: str) -> LiveRoom:
        room = self._require(identifier); assignment = self._assignment(room, subject_id)
        if not secrets.compare_digest(assignment.reconnect_token_hash, self._hash_token(reconnect_token)):
            raise PermissionError("invalid reconnect token")
        assignment.connected = True; assignment.last_seen_at = utc_now()
        room.append_event("PLAYER_RECONNECTED", subject_id=subject_id, seat=assignment.seat)
        self._save(room); return room

    def add_spectator(self, identifier: str, *, subject_id: str, display_name: str) -> LiveRoom:
        room = self._require(identifier)
        if not room.allow_spectators:
            raise PermissionError("spectators are disabled")
        room.spectators[subject_id] = display_name.strip()[:40] or "مشاهد"
        room.append_event("SPECTATOR_JOINED", subject_id=subject_id)
        self._save(room); return room

    def pause(self, identifier: str, *, requested_by: str, reason: str) -> LiveRoom:
        room = self._require(identifier)
        if not (owns_subject(requested_by, room.owner_subject_id) or owns_subject(requested_by, room.referee_subject_id)):
            raise PermissionError("only owner or referee may pause")
        if room.status != LiveStatus.ACTIVE:
            raise ValueError("only an active table may be paused")
        room.status = LiveStatus.PAUSED; room.paused_reason = reason.strip(); room.turn_deadline = None
        room.append_event("TABLE_PAUSED", subject_id=requested_by, payload={"reason": room.paused_reason})
        self._save(room); return room

    def resume(self, identifier: str, *, requested_by: str) -> LiveRoom:
        room = self._require(identifier)
        if not (owns_subject(requested_by, room.owner_subject_id) or owns_subject(requested_by, room.referee_subject_id)):
            raise PermissionError("only owner or referee may resume")
        if room.status != LiveStatus.PAUSED:
            raise ValueError("table is not paused")
        room.status = LiveStatus.ACTIVE; room.paused_reason = None; room.turn_deadline = future_time(room.turn_seconds)
        room.append_event("TABLE_RESUMED", subject_id=requested_by, payload={"turnDeadline": room.turn_deadline})
        self._save(room); return room

    def enforce_deadline(self, identifier: str) -> dict[str, Any]:
        with self._lock:
            room = self._require(identifier)
            if room.status != LiveStatus.ACTIVE or not room.turn_deadline:
                return {"expired": False, "room": room.to_dict()}
            deadline = datetime.fromisoformat(room.turn_deadline)
            if deadline > datetime.now(timezone.utc):
                return {"expired": False, "room": room.to_dict()}
            game = self.games.get(room.game_id)
            actor = game.bidding.actor if game.bidding is not None and not game.bidding.finished else game.state.current_player
            room.status = LiveStatus.PAUSED; room.paused_reason = "turn_timeout"; room.turn_deadline = None
            room.append_event("TURN_TIMEOUT", seat=actor, payload={"gameVersion": game.version})
            accused = next((item.subject_id for item in room.seats.values() if item.seat == actor), None) if actor else None
            case_id = self._open_adjudication(
                room, violation="timeout", description="انتهت مهلة الدور في الطاولة المباشرة.",
                accused_subject_id=accused, accused_seat=actor,
            )
            if case_id:
                room.append_event("ADJUDICATION_CASE_OPENED", payload={"caseId": case_id, "violation": "timeout"})
            self._save(room)
            return {"expired": True, "actorSeat": actor.name.lower() if actor else None, "adjudicationCaseId": case_id, "room": room.to_dict()}

    def enforce_reconnect_grace(self, identifier: str) -> dict[str, Any]:
        with self._lock:
            room = self._require(identifier)
            now = datetime.now(timezone.utc)
            expired: list[SeatAssignment] = []
            for assignment in room.seats.values():
                if assignment.connected or not assignment.last_seen_at:
                    continue
                last = datetime.fromisoformat(assignment.last_seen_at)
                if (now - last).total_seconds() >= room.reconnect_grace_seconds:
                    expired.append(assignment)
            case_ids: list[str] = []
            if expired and room.status == LiveStatus.ACTIVE:
                room.status = LiveStatus.PAUSED
                room.paused_reason = "reconnect_grace_expired"
                room.turn_deadline = None
            for assignment in expired:
                room.append_event("RECONNECT_GRACE_EXPIRED", subject_id=assignment.subject_id, seat=assignment.seat)
                case_id = self._open_adjudication(
                    room, violation="disconnect", description="انتهت مهلة العودة بعد انقطاع اللاعب.",
                    accused_subject_id=assignment.subject_id, accused_seat=assignment.seat,
                )
                if case_id:
                    case_ids.append(case_id)
                    room.append_event("ADJUDICATION_CASE_OPENED", payload={"caseId": case_id, "violation": "disconnect"})
            if expired:
                self._save(room)
            return {
                "expired": bool(expired),
                "seats": [item.seat.name.lower() for item in expired],
                "adjudicationCaseIds": case_ids,
                "room": room.to_dict(),
            }

    def view(self, identifier: str, *, subject_id: str | None = None, since_sequence: int = 0) -> dict[str, Any]:
        room = self._require(identifier)
        game = self.games.get(room.game_id)
        assignment = None
        if subject_id:
            assignment = next((item for item in room.seats.values() if owns_subject(subject_id, item.subject_id)), None)
        state = game.to_dict(reveal_hands=False)
        own_hand: list[str] = []
        if assignment is not None:
            mask = game.state.initial_hands[int(assignment.seat)]
            for played in game.state.played_cards:
                if played.seat == assignment.seat:
                    mask &= ~(1 << played.card.id)
            own_hand = [card.code for card in iter_mask(mask)]
        events = [item.to_dict() for item in room.events if item.sequence > int(since_sequence)]
        return {
            "room": room.to_dict(include_events=False),
            "game": state,
            "viewerSeat": assignment.seat.name.lower() if assignment else None,
            "ownHand": own_hand,
            "events": events,
            "nextSequence": room.events[-1].sequence if room.events else 0,
        }

    def verify(self, identifier: str) -> dict[str, Any]:
        room = self._require(identifier)
        issues: list[str] = []
        previous = ""
        for expected, event in enumerate(room.events, start=1):
            if event.sequence != expected:
                issues.append(f"sequence:{expected}")
            if event.previous_hash != previous:
                issues.append(f"previous_hash:{expected}")
            from .models import canonical_hash
            if event.event_hash != canonical_hash(event.to_dict(include_hash=False)):
                issues.append(f"event_hash:{expected}")
            previous = event.event_hash
        current = room.room_hash
        valid_room = current == room.computed_hash()
        if not valid_room:
            issues.append("room_hash")
        return {"roomId": room.room_id, "valid": not issues, "issues": issues, "roomHash": current, "eventHead": previous}

    def _require(self, identifier: str) -> LiveRoom:
        room = self.store.get(identifier)
        if room is None:
            raise KeyError(identifier)
        return room

    @staticmethod
    def _assignment(room: LiveRoom, subject_id: str) -> SeatAssignment:
        assignment = next((item for item in room.seats.values() if owns_subject(subject_id, item.subject_id)), None)
        if assignment is None:
            raise PermissionError("subject does not own a seat")
        return assignment
