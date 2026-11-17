from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Mapping, Sequence


class ReviewRole(StrEnum):
    PLAYER = "player"
    EXPERIENCED_PLAYER = "experienced_player"
    EXPERT = "expert"
    REFEREE = "referee"
    COACH = "coach"

    @property
    def weight(self) -> float:
        return {"player": 1.0, "experienced_player": 1.4, "coach": 1.8, "expert": 2.2, "referee": 2.5}[self.value]


class FeedbackReason(StrEnum):
    LEGAL_OBLIGATION = "legal_obligation"
    CARD_COUNTING = "card_counting"
    PARTNER_SIGNAL = "partner_signal"
    OPPONENT_SIGNAL = "opponent_signal"
    TRUMP_CONTROL = "trump_control"
    ENTRY_MANAGEMENT = "entry_management"
    PROTECT_POINTS = "protect_points"
    PROJECT_INFORMATION = "project_information"
    SCORE_STRATEGY = "score_strategy"
    RISK_MANAGEMENT = "risk_management"
    DECEPTION = "deception"
    OTHER = "other"

    @property
    def label_ar(self) -> str:
        return {
            "legal_obligation": "الحركة كانت مفروضة",
            "card_counting": "عدّ الورق",
            "partner_signal": "إشارة من الخوي",
            "opponent_signal": "قراءة الخصم",
            "trump_control": "التحكم بالحكم",
            "entry_management": "حفظ الدخول",
            "protect_points": "حماية النقاط",
            "project_information": "معلومات المشاريع",
            "score_strategy": "وضع نتيجة الصكّة",
            "risk_management": "تقليل أو قبول المخاطرة",
            "deception": "خداع أو تنويع اللعب",
            "other": "سبب آخر",
        }[self.value]


@dataclass(frozen=True, slots=True)
class StructuredDecisionReview:
    review_id: str
    decision_id: str
    reviewer_id: str
    reviewer_role: str
    preferred_action: Any
    confidence: float
    reason_codes: tuple[str, ...]
    rationale: str
    evidence: tuple[Mapping[str, Any], ...]
    view_mode: str
    source: str
    engine_validation: str
    engine_regret: float | None
    engine_win_drop: float | None
    created_at: str
    review_hash: str

    @classmethod
    def create(cls, *, decision_id: str, reviewer_id: str, reviewer_role: str,
               preferred_action: Any, confidence: float, reason_codes: Sequence[str],
               rationale: str, evidence: Sequence[Mapping[str, Any]], view_mode: str,
               source: str, engine_validation: str, engine_regret: float | None,
               engine_win_drop: float | None) -> "StructuredDecisionReview":
        role = ReviewRole(str(reviewer_role))
        reasons = tuple(FeedbackReason(str(code)).value for code in reason_codes)
        if not reasons:
            reasons = (FeedbackReason.OTHER.value,)
        cleaned = " ".join(str(rationale).strip().split())
        if len(cleaned) < 3:
            raise ValueError("review rationale is too short")
        if len(cleaned) > 3000:
            raise ValueError("review rationale is too long")
        created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        payload = {
            "decisionId": decision_id, "reviewerId": reviewer_id,
            "reviewerRole": role.value, "preferredAction": preferred_action,
            "confidence": round(max(0.0, min(1.0, float(confidence))), 6),
            "reasonCodes": list(reasons), "rationale": cleaned,
            "evidence": [dict(item) for item in evidence], "viewMode": view_mode,
            "source": source, "engineValidation": engine_validation,
            "engineRegret": engine_regret, "engineWinDrop": engine_win_drop,
            "createdAt": created,
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return cls(
            f"REV-{digest[:20].upper()}", decision_id, reviewer_id, role.value,
            preferred_action, payload["confidence"], reasons, cleaned,
            tuple(dict(item) for item in evidence), view_mode, source,
            engine_validation, engine_regret, engine_win_drop, created, digest,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reviewId": self.review_id, "decisionId": self.decision_id,
            "reviewerId": self.reviewer_id, "reviewerRole": self.reviewer_role,
            "reviewerWeight": ReviewRole(self.reviewer_role).weight,
            "preferredAction": self.preferred_action, "confidence": self.confidence,
            "reasonCodes": list(self.reason_codes), "rationale": self.rationale,
            "evidence": [dict(item) for item in self.evidence], "viewMode": self.view_mode,
            "source": self.source, "engineValidation": self.engine_validation,
            "engineRegret": self.engine_regret, "engineWinDrop": self.engine_win_drop,
            "createdAt": self.created_at, "reviewHash": self.review_hash,
        }
