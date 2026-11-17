from __future__ import annotations

from typing import Any

from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Suit

from .models import ReasonEvidence, action_key, canonical_action


_LABELS = {
    "forced": ("إجباري", "Forced"),
    "excellent": ("ممتاز", "Excellent"),
    "good": ("جيد", "Good"),
    "inaccuracy": ("غير دقيق", "Inaccuracy"),
    "mistake": ("خطأ", "Mistake"),
    "blunder": ("خطأ حاسم", "Blunder"),
}


def quality_label(
    *,
    forced: bool,
    regret: float,
    win_drop: float,
    catastrophic: bool,
    thresholds: dict[str, Any],
) -> tuple[str, float]:
    if forced:
        return "forced", 100.0
    margin = thresholds.get("expected_margin_regret", {})
    win = thresholds.get("win_probability_drop", {})
    if catastrophic or regret >= float(margin.get("mistake", 14.0)) or win_drop >= float(win.get("mistake", 0.12)):
        label = "blunder"
    elif regret >= float(margin.get("inaccuracy", 6.0)) or win_drop >= float(win.get("inaccuracy", 0.05)):
        label = "mistake"
    elif regret >= float(margin.get("good", 2.5)) or win_drop >= float(win.get("good", 0.02)):
        label = "inaccuracy"
    elif regret >= float(margin.get("excellent", 0.75)) or win_drop >= float(win.get("excellent", 0.005)):
        label = "good"
    else:
        label = "excellent"
    # A stable, easy-to-read 0–100 move quality score. Both expected utility
    # regret and probability drop contribute, but neither can create a negative
    # result or exceed 100.
    score = 100.0 - min(100.0, regret * 3.2 + win_drop * 240.0 + (18.0 if catastrophic else 0.0))
    return label, round(max(0.0, score), 2)


def _action_text(action: Any, *, arabic: bool) -> str:
    value = canonical_action(action)
    if value["kind"] == "card":
        code = value.get("code") or f"#{value.get('id')}"
        try:
            card = Card.parse(code)
            return f"{card.rank.short}{card.suit.symbol}" if arabic else card.code
        except Exception:
            return str(code)
    if value["kind"] == "bid":
        action_name = str(value.get("action", "pass"))
        trump = value.get("trumpSuit")
        if arabic:
            translated = {"pass": "تمرير", "sun": "صن", "hokum": "حكم", "ashkal": "أشكل"}.get(action_name, action_name)
            if trump:
                try:
                    trump = Suit.parse(trump).symbol
                except ValueError:
                    trump = str(trump)
            return translated + (f" {trump}" if trump else "")
        return action_name + (f" — {trump}" if trump else "")
    return str(value.get("value", value))


def _arabic_seat(value: Any) -> str:
    return {
        "north": "شمال",
        "west": "غرب",
        "south": "جنوب",
        "east": "شرق",
    }.get(str(value).strip().lower(), str(value))


def _arabic_suit(value: Any) -> str:
    try:
        return Suit.parse(value).symbol
    except ValueError:
        return str(value or "")


def _arabic_signal(value: Any) -> str:
    return {
        "low_then_high": "صغيرة ثم كبيرة",
        "high_then_low": "كبيرة ثم صغيرة",
        "lowest_winner": "الأكل بأصغر ورقة كافية",
        "return_partner_suit": "ترجيع نوع الشريك",
        "partner": "الشريك",
        "opponent": "الخصم",
        "meaningful_choice": "اختيار فعلي",
        "forced": "إجباري",
        "unknown": "غير محسوم",
    }.get(str(value).strip().lower(), str(value))


def build_reasons(
    *,
    decision: dict[str, Any],
    analysis: dict[str, Any],
    actual_action: Any,
    best_action: Any,
    forced: bool,
    regret: float,
    win_drop: float,
    catastrophic: bool,
    maximum_reasons: int = 5,
) -> tuple[ReasonEvidence, ...]:
    reasons: list[ReasonEvidence] = []
    actual_ar = _action_text(actual_action, arabic=True)
    best_ar = _action_text(best_action, arabic=True)
    actual_en = _action_text(actual_action, arabic=False)
    best_en = _action_text(best_action, arabic=False)

    if forced:
        reasons.append(
            ReasonEvidence(
                "FORCED_MOVE", "legality", "fact",
                "لا يوجد قرار تكتيكي هنا",
                "كان هذا هو الخيار القانوني الوحيد؛ لذلك لا يصنّف كخطأ أو مهارة.",
                "No tactical choice",
                "This was the only legal action, so it is neither a mistake nor a skill signal.",
                1.0,
            )
        )
    elif action_key(actual_action) == action_key(best_action):
        reasons.append(
            ReasonEvidence(
                "ENGINE_MATCH", "counterfactual", "computed",
                "القرار يطابق أفضل بديل محسوب",
                f"اختيار {actual_ar} حقق أعلى قيمة متوقعة ضمن المعلومات المتاحة للاعب.",
                "Decision matches the best evaluated alternative",
                f"{actual_en} had the highest expected value under the player's available information.",
                1.0,
            )
        )
    else:
        reasons.append(
            ReasonEvidence(
                "COUNTERFACTUAL_REGRET", "counterfactual", "computed",
                "كان هناك بديل أقوى",
                f"مقارنةً بـ {actual_ar}، أعطى {best_ar} هامشاً متوقعاً أفضل بنحو {regret:.1f} نقطة.",
                "A stronger alternative existed",
                f"Compared with {actual_en}, {best_en} improved expected margin by about {regret:.1f} points.",
                min(1.0, 0.45 + regret / 20.0),
            )
        )
    if win_drop >= 0.01:
        reasons.append(
            ReasonEvidence(
                "WIN_PROBABILITY_DROP", "outcome", "computed",
                "القرار غيّر احتمال الفوز",
                f"الفرق بين القرار الفعلي والأفضل بلغ قرابة {win_drop * 100:.1f} نقطة مئوية في احتمال الفوز.",
                "The action changed winning chances",
                f"The selected action trailed the best action by about {win_drop * 100:.1f} percentage points.",
                min(1.0, 0.5 + win_drop * 2.5),
            )
        )
    if catastrophic:
        reasons.append(
            ReasonEvidence(
                "OUTCOME_CROSSING", "outcome", "computed",
                "القرار قلب اتجاه النتيجة",
                "أحد البدائل أبقى التوقع في صالح الفريق، بينما القرار الآخر نقل التوقع إلى خسارة.",
                "The action crossed the outcome boundary",
                "One alternative kept the expected result favorable while the other moved it into a loss.",
                1.0,
            )
        )

    deductions = list(analysis.get("deductions", []))
    if deductions:
        item = deductions[0]
        seat = _arabic_seat(item.get("seat", ""))
        suit = _arabic_suit(item.get("suit"))
        reasons.append(
            ReasonEvidence(
                "HARD_DEDUCTION", "hidden_information", "fact",
                "استنتاج مؤكد من سجل اللعب",
                f"السجل يثبت قيداً على يد {seat} في {suit}؛ وقد استُخدم لاستبعاد التوزيعات المستحيلة.",
                "A proven deduction from the play history",
                f"The replay proves a constraint on {seat} in {suit}; impossible hidden deals were removed.",
                0.9,
                tuple(str(item.get(key, "")) for key in ("source",)),
            )
        )

    signals = list(analysis.get("signals", []))
    if signals:
        signal = signals[0]
        signal_kind = _arabic_signal(signal.get("kind", ""))
        relationship = _arabic_signal(signal.get("relationship", ""))
        forcedness = _arabic_signal(signal.get("forcedness", "unknown"))
        reasons.append(
            ReasonEvidence(
                "SIGNAL_HYPOTHESIS", "intention", "hypothesis",
                "قراءة محتملة لإشارة لعب",
                f"رُصد نمط {signal_kind} من {relationship}. هذه قراءة محتملة وليست حقيقة، وحالة الإلزام: {forcedness}.",
                "Possible interpretation of a play signal",
                f"A {signal.get('kind', '')} pattern was observed from {signal.get('relationship', '')}. It is a hypothesis, not a fact; forcedness is {signal.get('forcedness', 'unknown')}.",
                float(signal.get("confidence", 0.45)),
            )
        )

    metadata = decision.get("decisionMetadata") or {}
    source = str(metadata.get("source", "human"))
    if source == "bot":
        profile = metadata.get("profileId", "unknown")
        strategy = metadata.get("strategyId", "unknown")
        selected_rank = metadata.get("selectedRank")
        detail = f"اللاعب الآلي استخدم ملف {profile} عبر {strategy}"
        if selected_rank is not None:
            detail += f" واختار المرتبة {int(selected_rank) + 1}"
        if metadata.get("mistakeApplied"):
            detail += " بعد تطبيق خطأ مضبوط ومقصود للاختبار"
        reasons.append(
            ReasonEvidence(
                "AGENT_POLICY_TRACE", "provenance", "recorded",
                "سبب القرار الآلي مسجّل",
                detail + ".",
                "The bot's decision provenance is recorded",
                f"Agent profile {profile} used {strategy}; selected rank was {selected_rank} and controlled mistake={bool(metadata.get('mistakeApplied'))}.",
                1.0,
            )
        )
    elif not decision.get("feedback"):
        reasons.append(
            ReasonEvidence(
                "HUMAN_INTENT_UNKNOWN", "intention", "unknown",
                "نية اللاعب غير معروفة",
                "يمكن تفسير أثر القرار حسابياً، لكن لا يجوز الادعاء بمعرفة نية اللاعب من الورقة وحدها. أضف مراجعته لتوثيق السبب الحقيقي.",
                "The player's actual intent is unknown",
                "The action's effect can be evaluated, but intent cannot be inferred with certainty from the card alone. Add player or expert feedback to record it.",
                0.7,
            )
        )

    return tuple(sorted(reasons, key=lambda item: item.weight, reverse=True)[: max(1, int(maximum_reasons))])


def actual_intent_summary(decision: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(decision.get("decisionMetadata") or {})
    feedback = list(decision.get("feedback") or [])
    if metadata.get("source") == "bot":
        return {
            "status": "recorded_policy_trace",
            "source": "bot",
            "profileId": metadata.get("profileId"),
            "strategyId": metadata.get("strategyId"),
            "decisionSource": metadata.get("decisionSource"),
            "modelId": metadata.get("modelId"),
            "selectedRank": metadata.get("selectedRank"),
            "mistakeApplied": bool(metadata.get("mistakeApplied", False)),
            "trace": metadata,
        }
    if feedback:
        latest = feedback[0]
        return {
            "status": "reviewed_statement",
            "source": latest.get("source", "human"),
            "reviewer": latest.get("reviewer"),
            "confidence": latest.get("confidence"),
            "rationale": latest.get("rationale"),
            "preferredAction": latest.get("preferredAction"),
        }
    return {
        "status": "unknown",
        "source": "human_or_external",
        "statement": "No player-authored rationale was recorded; only counterfactual effects and hypotheses are available.",
    }
