from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

from baloot_platform.core.enums import Seat, Team


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def canonical_action(value: Any) -> dict[str, Any]:
    """Convert cards, bids, declarations, and scalar actions to stable JSON.

    The audit layer never relies on object identity. Every comparison uses this
    canonical shape so historical rows remain comparable after code upgrades.
    """
    if value is None:
        return {"kind": "none"}
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if isinstance(value, dict):
        if "card" in value and isinstance(value["card"], dict):
            value = value["card"]
        if "code" in value or ("suit" in value and "rank" in value):
            return {
                "kind": "card",
                "id": value.get("id"),
                "code": value.get("code"),
                "suit": value.get("suit"),
                "rank": value.get("rank"),
            }
        if "action" in value or "trumpSuit" in value:
            return {
                "kind": "bid",
                "action": str(value.get("action", "pass")),
                "trumpSuit": value.get("trumpSuit"),
            }
        return {"kind": "object", "value": json.loads(canonical_json(value))}
    if isinstance(value, (str, int, float, bool)):
        return {"kind": "scalar", "value": value}
    return {"kind": "scalar", "value": str(value)}


def action_key(value: Any) -> str:
    return canonical_json(canonical_action(value))


@dataclass(frozen=True, slots=True)
class AuditedAlternative:
    action: dict[str, Any]
    rank: int
    expected_margin: float
    win_probability: float
    confidence_low: float
    confidence_high: float
    downside_cvar10: float
    volatility: float
    best_share: float
    regret: float
    win_probability_drop: float
    exact_share: float
    principal_variation: tuple[str, ...] = ()
    method_notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "rank": self.rank,
            "expectedMargin": self.expected_margin,
            "winProbability": self.win_probability,
            "confidence95": [self.confidence_low, self.confidence_high],
            "downsideCvar10": self.downside_cvar10,
            "volatility": self.volatility,
            "bestShare": self.best_share,
            "regret": self.regret,
            "winProbabilityDrop": self.win_probability_drop,
            "exactShare": self.exact_share,
            "principalVariation": list(self.principal_variation),
            "methodNotes": list(self.method_notes),
        }


@dataclass(frozen=True, slots=True)
class ReasonEvidence:
    code: str
    category: str
    certainty: str
    title_ar: str
    detail_ar: str
    title_en: str
    detail_en: str
    weight: float = 1.0
    source_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "category": self.category,
            "certainty": self.certainty,
            "titleAr": self.title_ar,
            "detailAr": self.detail_ar,
            "titleEn": self.title_en,
            "detailEn": self.detail_en,
            "weight": self.weight,
            "sourceIds": list(self.source_ids),
        }


@dataclass(frozen=True, slots=True)
class DecisionAudit:
    decision_id: str
    decision_hash: str
    game_id: str
    game_hash: str | None
    ordinal: int
    decision_kind: str
    actor: str
    phase: str
    actual_action: dict[str, Any]
    best_action: dict[str, Any]
    selected_rank: int
    quality_label: str
    quality_score: float
    forced: bool
    expected_regret: float
    win_probability_drop: float
    catastrophic_crossing: bool
    alternatives: tuple[AuditedAlternative, ...]
    hard_facts: tuple[dict[str, Any], ...]
    signal_hypotheses: tuple[dict[str, Any], ...]
    reasons: tuple[ReasonEvidence, ...]
    actual_intent: dict[str, Any]
    engine_method: str
    engine_version: str
    bundle_hash: str
    analysis_id: str | None
    model_id: str | None
    conventions_applied: bool
    hidden_deals: int
    effective_sample_size: float
    predicted_difficulty: float
    debate_score: float
    audit_confidence: float
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decisionId": self.decision_id,
            "decisionHash": self.decision_hash,
            "gameId": self.game_id,
            "gameHash": self.game_hash,
            "ordinal": self.ordinal,
            "decisionKind": self.decision_kind,
            "actor": self.actor,
            "phase": self.phase,
            "actualAction": self.actual_action,
            "bestAction": self.best_action,
            "selectedRank": self.selected_rank,
            "qualityLabel": self.quality_label,
            "qualityScore": self.quality_score,
            "forced": self.forced,
            "expectedRegret": self.expected_regret,
            "winProbabilityDrop": self.win_probability_drop,
            "catastrophicCrossing": self.catastrophic_crossing,
            "alternatives": [item.to_dict() for item in self.alternatives],
            "hardFacts": list(self.hard_facts),
            "signalHypotheses": list(self.signal_hypotheses),
            "reasons": [item.to_dict() for item in self.reasons],
            "actualIntent": self.actual_intent,
            "engineMethod": self.engine_method,
            "engineVersion": self.engine_version,
            "bundleHash": self.bundle_hash,
            "analysisId": self.analysis_id,
            "modelId": self.model_id,
            "conventionsApplied": self.conventions_applied,
            "hiddenDeals": self.hidden_deals,
            "effectiveSampleSize": self.effective_sample_size,
            "predictedDifficulty": self.predicted_difficulty,
            "debateScore": self.debate_score,
            "auditConfidence": self.audit_confidence,
            "createdAt": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class GameAuditReport:
    audit_id: str
    game_id: str
    game_hash: str | None
    game_serial: str | None
    mode: str
    bundle_hash: str
    engine_version: str
    config_hash: str
    decisions: tuple[DecisionAudit, ...]
    summary: dict[str, Any]
    critical_decisions: tuple[str, ...]
    report_hash: str
    created_at: str = field(default_factory=utc_now)

    @classmethod
    def build(
        cls,
        *,
        game_id: str,
        game_hash: str | None,
        game_serial: str | None,
        mode: str,
        bundle_hash: str,
        engine_version: str,
        config: dict[str, Any],
        decisions: Iterable[DecisionAudit],
    ) -> "GameAuditReport":
        items = tuple(decisions)
        label_counts: dict[str, int] = {}
        total_regret = 0.0
        quality_sum = 0.0
        for item in items:
            label_counts[item.quality_label] = label_counts.get(item.quality_label, 0) + 1
            total_regret += item.expected_regret
            quality_sum += item.quality_score
        critical = tuple(
            item.decision_id
            for item in sorted(
                items,
                key=lambda value: (
                    value.catastrophic_crossing,
                    value.win_probability_drop,
                    value.expected_regret,
                ),
                reverse=True,
            )[: min(12, len(items))]
            if item.forced is False
        )
        def journey(group: tuple[DecisionAudit, ...]) -> dict[str, Any]:
            meaningful = tuple(item for item in group if not item.forced)
            regrets = sorted((item.expected_regret for item in meaningful), reverse=True)
            return {
                "decisionCount": len(group),
                "forcedCount": sum(1 for item in group if item.forced),
                "nonForcedCount": len(meaningful),
                "maximumExpectedRegret": regrets[0] if regrets else None,
                "top3ExpectedRegretMean": sum(regrets[:3]) / len(regrets[:3]) if regrets else None,
                "minimumActionQuality": min((item.quality_score for item in meaningful), default=None),
                "criticalCount": sum(1 for item in meaningful if item.decision_id in critical),
                "status": "evaluated" if meaningful else "not_applicable",
            }
        by_seat = {
            seat.name.lower(): journey(tuple(item for item in items if Seat.parse(item.actor) == seat))
            for seat in Seat
        }
        by_team = {
            team.name.lower(): journey(tuple(item for item in items if Seat.parse(item.actor).team == team))
            for team in Team
        }
        summary = {
            "decisionCount": len(items),
            "forcedCount": sum(1 for item in items if item.forced),
            "qualityCounts": label_counts,
            "averageQuality": quality_sum / len(items) if items else 100.0,
            "totalExpectedRegret": total_regret,
            "catastrophicCount": sum(1 for item in items if item.catastrophic_crossing),
            "criticalCount": len(critical),
            "bySeat": by_seat,
            "byTeam": by_team,
        }
        config_hash = content_hash(config)
        identity = {
            "gameId": game_id,
            "gameHash": game_hash,
            "mode": mode,
            "bundleHash": bundle_hash,
            "engineVersion": engine_version,
            "configHash": config_hash,
            "decisions": [item.to_dict() for item in items],
        }
        report_hash = content_hash(identity)
        return cls(
            audit_id=f"AUD-{report_hash[:20].upper()}",
            game_id=game_id,
            game_hash=game_hash,
            game_serial=game_serial,
            mode=mode,
            bundle_hash=bundle_hash,
            engine_version=engine_version,
            config_hash=config_hash,
            decisions=items,
            summary=summary,
            critical_decisions=critical,
            report_hash=report_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "auditId": self.audit_id,
            "gameId": self.game_id,
            "gameHash": self.game_hash,
            "gameSerial": self.game_serial,
            "mode": self.mode,
            "bundleHash": self.bundle_hash,
            "engineVersion": self.engine_version,
            "configHash": self.config_hash,
            "decisions": [item.to_dict() for item in self.decisions],
            "summary": self.summary,
            "criticalDecisions": list(self.critical_decisions),
            "reportHash": self.report_hash,
            "createdAt": self.created_at,
        }
