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
class SakkaBoard:
    board_id: str
    version: int
    board_hash: str
    serial: str
    share_code: str
    bundle_hash: str
    observer_seat: Seat
    scope: str
    maximum_rounds: int
    starting_match_score: tuple[int, int]
    starting_round_number: int
    public_event_prefix_hash: str
    bot_policy_hashes: Mapping[str, str]
    scoring_profile: str
    time_control: Mapping[str, Any]
    reveal_policy: str
    trusted_payload_hash: str
    created_by: str
    visibility: str = "unlisted"
    status: str = "published"
    label: str | None = None
    created_at: str = field(default_factory=now)
    signature_algorithm: str | None = None
    signature_key_id: str | None = None
    board_signature: str | None = None
    environment_hash: str | None = None
    deal_schedule_hash: str | None = None
    agent_snapshot_hash: str | None = None
    shared_profile_hash: str | None = None
    terminology_version: str = "ar_sa_baloot_v2"

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": 3,
            "payloadHash": self.trusted_payload_hash,
            "bundleHash": self.bundle_hash,
            "observer": self.observer_seat.name.lower(),
            "scope": self.scope,
            "maximumRounds": self.maximum_rounds,
            "startingMatchScore": list(self.starting_match_score),
            "startingRoundNumber": self.starting_round_number,
            "eventPrefix": self.public_event_prefix_hash,
            "bots": dict(sorted(self.bot_policy_hashes.items())),
            "scoring": self.scoring_profile,
            "time": dict(self.time_control),
            "reveal": self.reveal_policy,
            "environmentHash": self.environment_hash,
            "dealScheduleHash": self.deal_schedule_hash,
            "agentSnapshotHash": self.agent_snapshot_hash,
            "sharedProfileHash": self.shared_profile_hash,
            "terminologyVersion": self.terminology_version,
        }

    @property
    def comparison_key(self) -> str:
        return digest({
            "boardHash": self.board_hash,
            "bundleHash": self.bundle_hash,
            "observerSeat": self.observer_seat.name.lower(),
            "scope": self.scope,
            "scopeLegacy": "round" if self.scope in {"laffa", "round"} else "match",
            "scopeLabelAr": "لفة واحدة" if self.scope in {"laffa", "round"} else "صكّة كاملة",
            "maximumRounds": self.maximum_rounds,
            "startingMatchScore": list(self.starting_match_score),
            "startingRoundNumber": self.starting_round_number,
            "publicEventPrefixHash": self.public_event_prefix_hash,
            "botPolicyHashes": dict(sorted(self.bot_policy_hashes.items())),
            "scoringProfile": self.scoring_profile,
            "timeControl": dict(self.time_control),
            "revealPolicy": self.reveal_policy,
        })

    def to_dict(self, *, include_hashes: bool = True) -> dict[str, Any]:
        value = {
            "boardId": self.board_id,
            "version": self.version,
            "serial": self.serial,
            "shareCode": self.share_code,
            "observerSeat": self.observer_seat.name.lower(),
            "scope": self.scope,
            "scopeLegacy": "round" if self.scope in {"laffa", "round"} else "match",
            "scopeLabelAr": "لفة واحدة" if self.scope in {"laffa", "round"} else "صكّة كاملة",
            "maximumRounds": self.maximum_rounds,
            "startingMatchScore": list(self.starting_match_score),
            "startingRoundNumber": self.starting_round_number,
            "scoringProfile": self.scoring_profile,
            "timeControl": dict(self.time_control),
            "revealPolicy": self.reveal_policy,
            "createdBy": self.created_by,
            "visibility": self.visibility,
            "status": self.status,
            "label": self.label,
            "createdAt": self.created_at,
            "signature": {
                "algorithm": self.signature_algorithm,
                "keyId": self.signature_key_id,
                "value": self.board_signature,
            } if self.board_signature else None,
        }
        if include_hashes:
            value.update({
                "boardHash": self.board_hash,
                "bundleHash": self.bundle_hash,
                "publicEventPrefixHash": self.public_event_prefix_hash,
                "botPolicyHashes": dict(self.bot_policy_hashes),
                "trustedPayloadHash": self.trusted_payload_hash,
                "comparisonKey": self.comparison_key,
                "environmentHash": self.environment_hash,
                "dealScheduleHash": self.deal_schedule_hash,
                "agentSnapshotHash": self.agent_snapshot_hash,
                "sharedProfileHash": self.shared_profile_hash,
                "terminologyVersion": self.terminology_version,
            })
        return value


@dataclass(frozen=True, slots=True)
class SakkaRoom:
    room_id: str
    board_id: str
    room_code: str
    created_by: str
    mode: str
    status: str
    max_players: int
    allow_guests: bool
    settings: Mapping[str, Any]
    created_at: str = field(default_factory=now)
    expires_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "roomId": self.room_id,
            "boardId": self.board_id,
            "roomCode": self.room_code,
            "createdBy": self.created_by,
            "mode": self.mode,
            "status": self.status,
            "maxPlayers": self.max_players,
            "allowGuests": self.allow_guests,
            "settings": dict(self.settings),
            "createdAt": self.created_at,
            "expiresAt": self.expires_at,
        }


@dataclass(frozen=True, slots=True)
class SakkaRun:
    run_id: str
    room_id: str
    board_id: str
    subject_id: str
    comparison_key: str
    status: str
    version: int
    rounds_completed: int = 0
    state_hash: str | None = None
    result: Mapping[str, Any] | None = None
    analysis_status: str = "not_requested"
    started_at: str = field(default_factory=now)
    completed_at: str | None = None
    game_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "runId": self.run_id,
            "roomId": self.room_id,
            "boardId": self.board_id,
            "subjectId": self.subject_id,
            "comparisonKey": self.comparison_key,
            "status": self.status,
            "version": self.version,
            "roundsCompleted": self.rounds_completed,
            "stateHash": self.state_hash,
            "result": dict(self.result) if self.result is not None else None,
            "analysisStatus": self.analysis_status,
            "startedAt": self.started_at,
            "completedAt": self.completed_at,
            "gameId": self.game_id,
        }
