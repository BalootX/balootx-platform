from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .models import (
    AdjudicationCase,
    CaseStatus,
    EvidenceItem,
    Ruling,
    RulingOutcome,
    ViolationCode,
)
from .research import AdjudicationEventEvidenceV1
from .store import SQLiteAdjudicationStore


class AdjudicationProfile:
    def __init__(self, data: Mapping[str, Any]):
        self.data = dict(data)
        profile = dict(self.data.get("profile") or {})
        self.profile_id = str(profile.get("id") or "adjudication_unconfigured")
        self.version = str(profile.get("version") or "0")
        canonical = json.dumps(self.data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.sha256 = hashlib.sha256(canonical).hexdigest()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AdjudicationProfile":
        return cls(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})

    def default_for(self, violation: ViolationCode | str) -> dict[str, Any]:
        key = ViolationCode(violation).value
        return dict((self.data.get("violations") or {}).get(key) or {})

    @property
    def appeal_enabled(self) -> bool:
        return bool((self.data.get("appeals") or {}).get("enabled", True))


class AdjudicationService:
    """Versioned tournament/adjudication workflow.

    The service never silently edits authoritative history. A ruling either
    produces a correction/replay instruction or a score/forfeit directive that
    another transactional game service applies explicitly.
    """

    def __init__(self, store: SQLiteAdjudicationStore, profile: AdjudicationProfile):
        self.store = store
        self.profile = profile

    def open_case(
        self,
        *,
        game_id: str,
        violation: ViolationCode | str,
        reported_by: str,
        description: str,
        round_number: int | None = None,
        event_sequence: int | None = None,
        accused_subject_id: str | None = None,
        accused_seat: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AdjudicationCase:
        case = AdjudicationCase.create(
            game_id=game_id,
            violation=violation,
            reported_by=reported_by,
            description=description,
            round_number=round_number,
            event_sequence=event_sequence,
            accused_subject_id=accused_subject_id,
            accused_seat=accused_seat,
            previous_hash=self.store.latest_hash(game_id),
            metadata=metadata,
        )
        self.store.save(case)
        return case

    def open_event_case(
        self,
        event: AdjudicationEventEvidenceV1,
        *,
        reported_by: str,
        description: str,
    ) -> AdjudicationCase:
        case = self.open_case(
            game_id=event.game_id,
            violation=event.violation,
            reported_by=reported_by,
            description=description,
            round_number=event.round_number,
            event_sequence=event.event_sequence,
            accused_seat=event.actor_seat,
            metadata={"adjudicationEventV1": event.to_dict()},
        )
        case.evidence.append(
            event.to_evidence_item(
                submitted_by=reported_by,
                description=description,
            )
        )
        case.status = CaseStatus.UNDER_REVIEW
        case.rehash()
        self.store.save(case)
        return case

    def add_evidence(self, case_id: str, *, kind: str, submitted_by: str, description: str,
                     uri: str | None = None, sha256: str | None = None,
                     metadata: Mapping[str, Any] | None = None) -> AdjudicationCase:
        case = self._require(case_id)
        if case.status in {CaseStatus.FINAL, CaseStatus.VOID}:
            raise ValueError("case is closed")
        case.evidence.append(EvidenceItem.create(
            kind=kind, submitted_by=submitted_by, description=description,
            uri=uri, sha256=sha256, metadata=metadata,
        ))
        case.status = CaseStatus.UNDER_REVIEW
        case.rehash()
        self.store.save(case)
        return case

    def recommended_ruling(self, case_id: str) -> dict[str, Any]:
        case = self._require(case_id)
        event = case.metadata.get("adjudicationEventV1")
        if isinstance(event, Mapping):
            return {
                "caseId": case.case_id,
                "violation": case.violation.value,
                "profileId": str(event.get("authorityProfileId") or ""),
                "profileHash": str(event.get("authoritySha256") or ""),
                "recommendedOutcome": "referee_choice_required",
                "availableRemedies": list(event.get("availableRemedies") or ()),
                "requiresReferee": True,
                "requiresVideoReview": True,
                "notesAr": case.description,
            }
        default = self.profile.default_for(case.violation)
        return {
            "caseId": case.case_id,
            "violation": case.violation.value,
            "profileId": self.profile.profile_id,
            "profileHash": self.profile.sha256,
            "recommendedOutcome": default.get("default_outcome", "warning"),
            "pointsDelta": list(default.get("points_delta", [0, 0])),
            "requiresReferee": bool(default.get("requires_referee", True)),
            "requiresVideoReview": bool(default.get("requires_video_review", False)),
            "notesAr": str(default.get("notes_ar") or "يحتاج مراجعة حكم مع سجل اللعب."),
        }

    def rule(self, case_id: str, *, referee_id: str, outcome: RulingOutcome | str,
             reason_ar: str, points_delta: tuple[int, int] = (0, 0),
             replay_from_event_sequence: int | None = None, final: bool = False,
             metadata: Mapping[str, Any] | None = None) -> AdjudicationCase:
        case = self._require(case_id)
        if case.status in {CaseStatus.FINAL, CaseStatus.VOID}:
            raise ValueError("case is closed")
        event = case.metadata.get("adjudicationEventV1")
        profile_id = self.profile.profile_id
        profile_hash = self.profile.sha256
        if isinstance(event, Mapping):
            selected = metadata.get("selectedRemedy") if isinstance(metadata, Mapping) else None
            remedies = tuple(str(item) for item in event.get("availableRemedies") or ())
            if RulingOutcome(outcome) != RulingOutcome.CUSTOM or selected not in remedies:
                raise ValueError("adjudication events require a supported referee remedy")
            profile_id = str(event.get("authorityProfileId") or "")
            profile_hash = str(event.get("authoritySha256") or "")
        ruling = Ruling.create(
            outcome=outcome, referee_id=referee_id, reason_ar=reason_ar,
            points_delta=points_delta, replay_from_event_sequence=replay_from_event_sequence,
            profile_id=profile_id, profile_hash=profile_hash,
            final=final, metadata=metadata,
        )
        case.rulings.append(ruling)
        case.status = CaseStatus.FINAL if final else CaseStatus.RULED
        case.rehash()
        self.store.save(case)
        return case

    def rule_event(
        self,
        case_id: str,
        *,
        referee_id: str,
        selected_remedy: str,
        reason_ar: str,
        final: bool = False,
    ) -> AdjudicationCase:
        case = self._require(case_id)
        event = case.metadata.get("adjudicationEventV1")
        if not isinstance(event, Mapping):
            raise ValueError("case is not an adjudication event")
        remedies = tuple(str(item) for item in event.get("availableRemedies") or ())
        if selected_remedy not in remedies:
            raise ValueError(f"unsupported adjudication remedy: {selected_remedy}")
        return self.rule(
            case_id,
            referee_id=referee_id,
            outcome=RulingOutcome.CUSTOM,
            reason_ar=reason_ar,
            final=final,
            metadata={
                "schemaVersion": 1,
                "adjudicationEventId": event.get("evidenceId"),
                "selectedRemedy": selected_remedy,
            },
        )

    def appeal(self, case_id: str, *, appealed_by: str, reason: str) -> AdjudicationCase:
        case = self._require(case_id)
        if not self.profile.appeal_enabled:
            raise ValueError("appeals are disabled")
        if case.status != CaseStatus.RULED:
            raise ValueError("only a ruled non-final case may be appealed")
        case.status = CaseStatus.APPEALED
        case.appeal_reason = reason.strip()
        case.appealed_by = appealed_by
        from .models import utc_now
        case.appeal_created_at = utc_now()
        case.rehash()
        self.store.save(case)
        return case

    def resolve_appeal(self, case_id: str, *, referee_id: str, outcome: RulingOutcome | str,
                       reason_ar: str, points_delta: tuple[int, int] = (0, 0),
                       replay_from_event_sequence: int | None = None,
                       metadata: Mapping[str, Any] | None = None) -> AdjudicationCase:
        case = self._require(case_id)
        if case.status != CaseStatus.APPEALED:
            raise ValueError("case is not appealed")
        return self.rule(
            case_id, referee_id=referee_id, outcome=outcome, reason_ar=reason_ar,
            points_delta=points_delta, replay_from_event_sequence=replay_from_event_sequence,
            final=True, metadata=metadata,
        )

    def verify(self, case_id: str) -> dict[str, Any]:
        case = self._require(case_id)
        current = case.case_hash
        valid = current == case.computed_hash()
        return {
            "caseId": case.case_id,
            "valid": valid,
            "caseHash": current,
            "previousHash": case.previous_hash,
            "profileHash": self.profile.sha256,
        }

    def directive(self, case_id: str) -> dict[str, Any]:
        case = self._require(case_id)
        if not case.rulings:
            raise ValueError("case has no ruling")
        ruling = case.rulings[-1]
        return {
            "schemaVersion": 1,
            "caseId": case.case_id,
            "gameId": case.game_id,
            "eventSequence": case.event_sequence,
            "status": case.status.value,
            "outcome": ruling.outcome.value,
            "pointsDelta": list(ruling.points_delta),
            "replayFromEventSequence": ruling.replay_from_event_sequence,
            "rulingId": ruling.ruling_id,
            "rulingHash": hashlib.sha256(json.dumps(ruling.to_dict(), sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "adjudicationEventId": ruling.metadata.get("adjudicationEventId"),
            "selectedRemedy": ruling.metadata.get("selectedRemedy"),
            "mustCreateCorrectionLineage": ruling.outcome in {
                RulingOutcome.CORRECT_ACTION,
                RulingOutcome.REPLAY_FROM_EVENT,
                RulingOutcome.REDEAL_ROUND,
            },
        }

    def _require(self, case_id: str) -> AdjudicationCase:
        case = self.store.get(case_id)
        if case is None:
            raise KeyError(case_id)
        return case
