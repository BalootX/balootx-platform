from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ArabicReason:
    code: str
    label_ar: str
    text_ar: str
    certainty: str
    evidence: tuple[str, ...] = ()
    category: str = "computed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "labelAr": self.label_ar,
            "textAr": self.text_ar,
            "certainty": self.certainty,
            "evidence": list(self.evidence),
            "category": self.category,
        }


@dataclass(frozen=True, slots=True)
class ArabicDecisionExplanation:
    decision_id: str
    available: bool
    title_ar: str
    summary_ar: str
    attribution_ar: str
    visibility: str
    reasons: tuple[ArabicReason, ...] = ()
    caveat_ar: str = ""
    selected_action: Any = None
    best_action: Any = None
    level: str = "basic"
    surface: str = "replay"
    evidence_version: str = "arabic_explanation_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "decisionId": self.decision_id,
            "available": self.available,
            "titleAr": self.title_ar,
            "summaryAr": self.summary_ar,
            "attributionAr": self.attribution_ar,
            "visibility": self.visibility,
            "reasons": [item.to_dict() for item in self.reasons],
            "caveatAr": self.caveat_ar,
            "selectedAction": self.selected_action,
            "bestAction": self.best_action,
            "level": self.level,
            "surface": self.surface,
            "evidenceVersion": self.evidence_version,
        }
