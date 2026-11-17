from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Mapping


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ViolationCode(StrEnum):
    MISDEAL = "misdeal"
    EXPOSED_CARD = "exposed_card"
    ILLEGAL_CARD = "illegal_card"
    OUT_OF_TURN = "out_of_turn"
    WRONG_BID = "wrong_bid"
    COMMUNICATION = "communication"
    TIMEOUT = "timeout"
    DISCONNECT = "disconnect"
    CLAIM = "claim"
    CONCESSION = "concession"
    SCORE_DISPUTE = "score_dispute"
    TECHNICAL_FAILURE = "technical_failure"
    OTHER = "other"


class CaseStatus(StrEnum):
    OPEN = "open"
    UNDER_REVIEW = "under_review"
    RULED = "ruled"
    APPEALED = "appealed"
    FINAL = "final"
    VOID = "void"


class RulingOutcome(StrEnum):
    NO_VIOLATION = "no_violation"
    WARNING = "warning"
    CORRECT_ACTION = "correct_action"
    REPLAY_FROM_EVENT = "replay_from_event"
    REDEAL_ROUND = "redeal_round"
    AWARD_POINTS = "award_points"
    DEDUCT_POINTS = "deduct_points"
    FORFEIT_ROUND = "forfeit_round"
    FORFEIT_MATCH = "forfeit_match"
    TECHNICAL_DRAW = "technical_draw"
    CUSTOM = "custom"


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    evidence_id: str
    kind: str
    uri: str | None
    description: str
    submitted_by: str
    created_at: str
    sha256: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        kind: str,
        submitted_by: str,
        description: str,
        uri: str | None = None,
        sha256: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "EvidenceItem":
        return cls(
            evidence_id=f"EVD-{uuid.uuid4().hex[:20].upper()}",
            kind=str(kind),
            uri=uri,
            description=str(description).strip(),
            submitted_by=submitted_by,
            created_at=utc_now(),
            sha256=sha256,
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidenceId": self.evidence_id,
            "kind": self.kind,
            "uri": self.uri,
            "description": self.description,
            "submittedBy": self.submitted_by,
            "createdAt": self.created_at,
            "sha256": self.sha256,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class Ruling:
    ruling_id: str
    outcome: RulingOutcome
    referee_id: str
    reason_ar: str
    created_at: str
    points_delta: tuple[int, int] = (0, 0)
    replay_from_event_sequence: int | None = None
    profile_id: str = ""
    profile_hash: str = ""
    final: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        outcome: RulingOutcome | str,
        referee_id: str,
        reason_ar: str,
        points_delta: tuple[int, int] = (0, 0),
        replay_from_event_sequence: int | None = None,
        profile_id: str = "",
        profile_hash: str = "",
        final: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Ruling":
        return cls(
            ruling_id=f"RUL-{uuid.uuid4().hex[:20].upper()}",
            outcome=RulingOutcome(outcome),
            referee_id=referee_id,
            reason_ar=reason_ar.strip(),
            created_at=utc_now(),
            points_delta=(int(points_delta[0]), int(points_delta[1])),
            replay_from_event_sequence=replay_from_event_sequence,
            profile_id=profile_id,
            profile_hash=profile_hash,
            final=bool(final),
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rulingId": self.ruling_id,
            "outcome": self.outcome.value,
            "refereeId": self.referee_id,
            "reasonAr": self.reason_ar,
            "createdAt": self.created_at,
            "pointsDelta": list(self.points_delta),
            "replayFromEventSequence": self.replay_from_event_sequence,
            "profileId": self.profile_id,
            "profileHash": self.profile_hash,
            "final": self.final,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class AdjudicationCase:
    case_id: str
    game_id: str
    round_number: int | None
    event_sequence: int | None
    violation: ViolationCode
    reported_by: str
    accused_subject_id: str | None
    accused_seat: str | None
    description: str
    status: CaseStatus
    created_at: str
    updated_at: str
    evidence: list[EvidenceItem] = field(default_factory=list)
    rulings: list[Ruling] = field(default_factory=list)
    appeal_reason: str | None = None
    appealed_by: str | None = None
    appeal_created_at: str | None = None
    previous_hash: str = ""
    case_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        game_id: str,
        violation: ViolationCode | str,
        reported_by: str,
        description: str,
        round_number: int | None = None,
        event_sequence: int | None = None,
        accused_subject_id: str | None = None,
        accused_seat: str | None = None,
        previous_hash: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> "AdjudicationCase":
        now = utc_now()
        case = cls(
            case_id=f"ADJ-{uuid.uuid4().hex[:20].upper()}",
            game_id=game_id,
            round_number=round_number,
            event_sequence=event_sequence,
            violation=ViolationCode(violation),
            reported_by=reported_by,
            accused_subject_id=accused_subject_id,
            accused_seat=accused_seat,
            description=description.strip(),
            status=CaseStatus.OPEN,
            created_at=now,
            updated_at=now,
            previous_hash=previous_hash,
            metadata=dict(metadata or {}),
        )
        case.rehash()
        return case

    def public_payload(self) -> dict[str, Any]:
        return {
            "caseId": self.case_id,
            "gameId": self.game_id,
            "roundNumber": self.round_number,
            "eventSequence": self.event_sequence,
            "violation": self.violation.value,
            "reportedBy": self.reported_by,
            "accusedSubjectId": self.accused_subject_id,
            "accusedSeat": self.accused_seat,
            "description": self.description,
            "status": self.status.value,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "evidence": [item.to_dict() for item in self.evidence],
            "rulings": [item.to_dict() for item in self.rulings],
            "appealReason": self.appeal_reason,
            "appealedBy": self.appealed_by,
            "appealCreatedAt": self.appeal_created_at,
            "previousHash": self.previous_hash,
            "metadata": dict(self.metadata),
        }

    def computed_hash(self) -> str:
        """Return the canonical hash without mutating timestamps or state."""
        return canonical_hash(self.public_payload())

    def rehash(self) -> str:
        self.updated_at = utc_now()
        self.case_hash = self.computed_hash()
        return self.case_hash

    def to_dict(self) -> dict[str, Any]:
        value = self.public_payload()
        value["caseHash"] = self.case_hash
        return value
