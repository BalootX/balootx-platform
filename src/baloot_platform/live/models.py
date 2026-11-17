from __future__ import annotations

import hashlib
import json
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import Any, Mapping

from baloot_platform.core.enums import Seat


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def future_time(seconds: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=max(1, int(seconds)))).isoformat(timespec="seconds")


def canonical_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class LiveStatus(StrEnum):
    WAITING = "waiting"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class SeatAssignment:
    seat: Seat
    subject_id: str
    display_name: str
    ready: bool = False
    connected: bool = True
    joined_at: str = field(default_factory=utc_now)
    last_seen_at: str = field(default_factory=utc_now)
    reconnect_token_hash: str = ""
    takeover: str | None = None

    def to_dict(self, *, include_private: bool = False) -> dict[str, Any]:
        value = {
            "seat": self.seat.name.lower(),
            "subjectId": self.subject_id,
            "displayName": self.display_name,
            "ready": self.ready,
            "connected": self.connected,
            "joinedAt": self.joined_at,
            "lastSeenAt": self.last_seen_at,
            "takeover": self.takeover,
        }
        if include_private:
            value["reconnectTokenHash"] = self.reconnect_token_hash
        return value


@dataclass(slots=True)
class LiveEvent:
    sequence: int
    event_type: str
    actor_subject_id: str | None
    actor_seat: str | None
    created_at: str
    payload: dict[str, Any]
    previous_hash: str
    event_hash: str = ""

    @classmethod
    def create(cls, *, sequence: int, event_type: str, actor_subject_id: str | None,
               actor_seat: str | None, payload: Mapping[str, Any] | None, previous_hash: str) -> "LiveEvent":
        item = cls(sequence, event_type, actor_subject_id, actor_seat, utc_now(), dict(payload or {}), previous_hash)
        item.event_hash = canonical_hash(item.to_dict(include_hash=False))
        return item

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "sequence": self.sequence,
            "eventType": self.event_type,
            "actorSubjectId": self.actor_subject_id,
            "actorSeat": self.actor_seat,
            "createdAt": self.created_at,
            "payload": dict(self.payload),
            "previousHash": self.previous_hash,
        }
        if include_hash:
            value["eventHash"] = self.event_hash
        return value


@dataclass(slots=True)
class LiveRoom:
    room_id: str
    room_code: str
    game_id: str
    owner_subject_id: str
    status: LiveStatus
    created_at: str
    updated_at: str
    turn_seconds: int
    reconnect_grace_seconds: int
    allow_spectators: bool
    assistance_policy: str
    referee_subject_id: str | None = None
    seats: dict[str, SeatAssignment] = field(default_factory=dict)
    spectators: dict[str, str] = field(default_factory=dict)
    turn_deadline: str | None = None
    paused_reason: str | None = None
    events: list[LiveEvent] = field(default_factory=list)
    room_hash: str = ""
    version: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, *, game_id: str, owner_subject_id: str, turn_seconds: int = 45,
               reconnect_grace_seconds: int = 120, allow_spectators: bool = True,
               assistance_policy: str = "none", metadata: Mapping[str, Any] | None = None) -> "LiveRoom":
        now = utc_now()
        room = cls(
            room_id=f"LIV-{uuid.uuid4().hex[:20].upper()}",
            room_code=secrets.token_urlsafe(9).replace("-", "").replace("_", "")[:12].upper(),
            game_id=game_id,
            owner_subject_id=owner_subject_id,
            status=LiveStatus.WAITING,
            created_at=now,
            updated_at=now,
            turn_seconds=max(5, min(600, int(turn_seconds))),
            reconnect_grace_seconds=max(15, min(3600, int(reconnect_grace_seconds))),
            allow_spectators=bool(allow_spectators),
            assistance_policy=str(assistance_policy),
            metadata=dict(metadata or {}),
        )
        room.rehash()
        return room

    def append_event(self, event_type: str, *, subject_id: str | None = None,
                     seat: Seat | str | None = None, payload: Mapping[str, Any] | None = None) -> LiveEvent:
        previous = self.events[-1].event_hash if self.events else ""
        seat_value = Seat.parse(seat).name.lower() if seat is not None else None
        event = LiveEvent.create(
            sequence=len(self.events) + 1, event_type=event_type,
            actor_subject_id=subject_id, actor_seat=seat_value,
            payload=payload, previous_hash=previous,
        )
        self.events.append(event)
        self.rehash()
        return event

    def computed_hash(self) -> str:
        """Return the room hash without changing updated_at."""
        payload = self.to_dict(include_events=False, include_private=True, include_hash=False, include_version=False)
        return canonical_hash(payload)

    def rehash(self) -> str:
        self.updated_at = utc_now()
        self.room_hash = self.computed_hash()
        return self.room_hash

    def to_dict(self, *, include_events: bool = True, include_private: bool = False,
                include_hash: bool = True, include_version: bool = True) -> dict[str, Any]:
        value = {
            "roomId": self.room_id,
            "roomCode": self.room_code,
            "gameId": self.game_id,
            "ownerSubjectId": self.owner_subject_id,
            "status": self.status.value,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "turnSeconds": self.turn_seconds,
            "reconnectGraceSeconds": self.reconnect_grace_seconds,
            "allowSpectators": self.allow_spectators,
            "assistancePolicy": self.assistance_policy,
            "refereeSubjectId": self.referee_subject_id,
            "seats": {name: item.to_dict(include_private=include_private) for name, item in self.seats.items()},
            "spectators": dict(self.spectators),
            "turnDeadline": self.turn_deadline,
            "pausedReason": self.paused_reason,
            "metadata": dict(self.metadata),
        }
        if include_version:
            value["version"] = self.version
        if include_events:
            value["events"] = [event.to_dict() for event in self.events]
        if include_hash:
            value["roomHash"] = self.room_hash
        return value
