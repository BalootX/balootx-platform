from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from baloot_platform.core.enums import Seat


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class BoardDefinition:
    board_id: str
    version: int
    board_hash: str
    serial: str
    bundle_hash: str
    observer_seat: Seat
    starting_match_score: tuple[int, int]
    public_event_prefix_hash: str
    bot_policy_hashes: Mapping[str, str]
    scoring_profile: str
    time_control: Mapping[str, Any]
    reveal_policy: str
    trusted_payload_hash: str
    status: str = "published"
    label: str | None = None
    created_at: str = field(default_factory=now)

    @property
    def comparison_key(self) -> str:
        return digest({
            "boardHash": self.board_hash,
            "bundleHash": self.bundle_hash,
            "observerSeat": self.observer_seat.name.lower(),
            "startingMatchScore": list(self.starting_match_score),
            "publicEventPrefixHash": self.public_event_prefix_hash,
            "botPolicyHashes": dict(sorted(self.bot_policy_hashes.items())),
            "scoringProfile": self.scoring_profile,
            "timeControl": dict(self.time_control),
            "revealPolicy": self.reveal_policy,
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "boardId": self.board_id, "version": self.version, "boardHash": self.board_hash,
            "serial": self.serial, "bundleHash": self.bundle_hash,
            "observerSeat": self.observer_seat.name.lower(),
            "startingMatchScore": list(self.starting_match_score),
            "publicEventPrefixHash": self.public_event_prefix_hash,
            "botPolicyHashes": dict(self.bot_policy_hashes), "scoringProfile": self.scoring_profile,
            "timeControl": dict(self.time_control), "revealPolicy": self.reveal_policy,
            "trustedPayloadHash": self.trusted_payload_hash, "comparisonKey": self.comparison_key,
            "status": self.status, "label": self.label, "createdAt": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class DuplicateRoom:
    room_id: str
    board_id: str
    room_code: str
    mode: str
    status: str
    settings: Mapping[str, Any]
    created_at: str = field(default_factory=now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "roomId": self.room_id, "boardId": self.board_id, "roomCode": self.room_code,
            "mode": self.mode, "status": self.status, "settings": dict(self.settings),
            "createdAt": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class DuplicateRun:
    run_id: str
    room_id: str
    board_id: str
    subject_id: str
    comparison_key: str
    status: str
    version: int
    state_hash: str | None = None
    result: Mapping[str, Any] | None = None
    started_at: str = field(default_factory=now)
    completed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "runId": self.run_id, "roomId": self.room_id, "boardId": self.board_id,
            "subjectId": self.subject_id, "comparisonKey": self.comparison_key,
            "status": self.status, "version": self.version, "stateHash": self.state_hash,
            "result": dict(self.result) if self.result is not None else None,
            "startedAt": self.started_at, "completedAt": self.completed_at,
        }
