from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from baloot_platform.core.cards import Card
from baloot_platform.core.scenario import EpisodeSpec


def _card_action(code: str) -> dict[str, Any]:
    card = Card.parse(code)
    return {"kind": "card", **card.to_dict()}


def build_comparison_episode(
    base: EpisodeSpec,
    comparison: Mapping[str, Any],
    *,
    player_a_name: str = "اللاعب الأول",
    player_b_name: str = "اللاعب الثاني",
    title_ar: str | None = None,
    question_ar: str = "نفس الورق ونفس البداية، أي قرار يغيّر النتيجة؟",
    simple_arabic_explanation: Mapping[str, Any] | None = None,
) -> EpisodeSpec:
    """Bind a verified standardized لفة/صكّة comparison to EpisodeSpec.

    ``sharedComparison`` is the canonical V9 payload. ``sakkaComparison`` is
    retained as a deprecated V8 compatibility alias. The renderer remains
    presentation-only and cannot alter any engine result.
    """
    first = comparison.get("firstDivergence") or {}
    left_action = ((first.get("runA") or {}).get("action") or {})
    right_action = ((first.get("runB") or {}).get("action") or {})
    left_code = str(left_action.get("card") or "")
    right_code = str(right_action.get("card") or "")
    if not left_code or not right_code:
        raise ValueError("comparison must contain a card-play divergence")
    left_result = dict((comparison.get("runA") or {}).get("result") or {})
    right_result = dict((comparison.get("runB") or {}).get("result") or {})
    winner_id = comparison.get("winnerRunId")
    left_wins = winner_id == (comparison.get("runA") or {}).get("runId")
    best_code = left_code if left_wins else right_code
    alternative_code = right_code if left_wins else left_code
    scope = str(comparison.get("sharedScope") or "laffa").lower()
    label = "لفة" if scope in {"laffa", "round"} else "صكّة"
    shared = {
        "schemaVersion": 2,
        "comparisonHash": comparison.get("comparisonHash"),
        "boardId": comparison.get("boardId"),
        "boardSerial": comparison.get("boardSerial"),
        "boardHash": comparison.get("boardHash"),
        "comparisonKey": comparison.get("comparisonKey"),
        "sameStartVerified": bool(comparison.get("sameStartVerified")),
        "sharedScope": "laffa" if scope in {"laffa", "round"} else "sakka",
        "scopeLabelAr": label,
        "targetMatchPoints": comparison.get("targetMatchPoints", 152),
        "environmentHash": comparison.get("environmentHash"),
        "dealScheduleHash": comparison.get("dealScheduleHash"),
        "agentSnapshotHash": comparison.get("agentSnapshotHash"),
        "modelSnapshotHash": comparison.get("modelSnapshotHash"),
        "sharedProfileHash": comparison.get("sharedProfileHash"),
        "winnerRunId": winner_id,
        "firstDivergence": first,
        "playerA": {
            "name": player_a_name,
            "runId": (comparison.get("runA") or {}).get("runId"),
            "choice": left_code,
            "result": left_result,
            "winner": left_wins,
        },
        "playerB": {
            "name": player_b_name,
            "runId": (comparison.get("runB") or {}).get("runId"),
            "choice": right_code,
            "result": right_result,
            "winner": not left_wins,
        },
        "verification": comparison.get("verification"),
        "simpleArabicExplanation": dict(simple_arabic_explanation or {
            "summaryAr": "أعدنا القرارين من نفس اللحظة وبنفس الورق والإعدادات. القرار الأول حافظ على مسار الفوز، بينما القرار الآخر غيّر النتيجة للأسوأ.",
            "attributionAr": "السبب مبني على إعادة فعلية للمسارين، وليس على تخمين نية اللاعب.",
        }),
    }
    existing = dict(base.audit_summary or {})
    existing.update({
        "schemaVersion": "9.0.0",
        "actualAction": _card_action(alternative_code),
        "bestAction": _card_action(best_code),
        "qualityLabel": "blunder",
        "qualityScore": 0.0,
        "expectedRegret": abs(float(left_result.get("lastRoundMargin", 0.0)) - float(right_result.get("lastRoundMargin", 0.0))),
        "winProbabilityDrop": 1.0,
        "auditConfidence": 1.0,
        "sharedComparison": shared,
        "sakkaComparison": shared,  # Deprecated compatibility alias.
    })
    serial = str(comparison.get("boardSerial") or base.game_serial or base.episode_id)
    comparison_hash = str(comparison.get("comparisonHash") or base.game_record_hash or base.bundle_hash)
    return replace(
        base,
        schema_version=9,
        episode_id=f"shared-v9-{comparison_hash[:16]}",
        title_ar=title_ar or f"{label} واحدة، قراران، نتيجتان",
        question_ar=question_ar,
        game_serial=serial,
        game_record_hash=comparison_hash,
        audit_summary=existing,
        playback_mode="shared_same_start_comparison",
    )
