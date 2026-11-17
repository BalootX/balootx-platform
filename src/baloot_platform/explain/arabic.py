from __future__ import annotations

from typing import Any, Mapping

from baloot_platform.core.cards import Card
from baloot_platform.shared.access import (
    observer_relation,
    private_decision_review_allowed,
)
from .models import ArabicDecisionExplanation, ArabicReason

_RANK_AR = {"7": "سبعة", "8": "ثمانية", "9": "تسعة", "10": "عشرة", "J": "ولد", "Q": "بنت", "K": "شيخ", "A": "آص"}
_SUIT_AR = {"spades": "سباتي", "hearts": "هاص", "diamonds": "ديناري", "clubs": "كلب"}
_BID_AR = {"pass": "مرّر", "sun": "اشترى صن", "hokum": "اشترى حكم", "ashkal": "اشترى أشكل"}
_REL_AR = {"self": "أنت", "partner": "خويك", "left_opponent": "الخصم اليسار", "right_opponent": "الخصم اليمين", "unknown": "اللاعب"}


def _action_label(action: Any) -> str:
    if action is None:
        return "هذا القرار"
    if isinstance(action, str):
        try:
            card = Card.parse(action)
            return f"{_RANK_AR.get(card.rank.label, card.rank.label)} {_SUIT_AR.get(card.suit.name.lower(), card.suit.name)}"
        except Exception:
            return _BID_AR.get(action.lower(), action)
    if isinstance(action, Mapping):
        code = action.get("code") or action.get("card")
        if code:
            return _action_label(str(code))
        bid = str(action.get("action") or "").lower()
        if bid:
            label = _BID_AR.get(bid, bid)
            if bid == "hokum" and action.get("trumpSuit"):
                label += f" على {_SUIT_AR.get(str(action['trumpSuit']).lower(), action['trumpSuit'])}"
            return label
    return str(action)


def _certainty(value: Any, category: str = "computed") -> str:
    if category == "fact":
        return "مؤكد"
    try:
        score = float(value)
    except Exception:
        return "محتمل"
    if score >= .85:
        return "ثقة عالية"
    if score >= .60:
        return "ثقة متوسطة"
    return "قراءة محتملة"


def _audit_values(audit: Mapping[str, Any] | None) -> tuple[Any, float, float, bool]:
    if not audit:
        return None, 0.0, 0.0, False
    best = audit.get("bestAction")
    regret = float(audit.get("expectedRegret", audit.get("regret", 0.0)) or 0.0)
    drop = float(audit.get("winProbabilityDrop", 0.0) or 0.0)
    catastrophic = bool(audit.get("catastrophic") or audit.get("outcomeCrossing"))
    return best, regret, drop, catastrophic


def explain_decision_ar(
    decision: Mapping[str, Any], *, audit: Mapping[str, Any] | None,
    observer: str | None, level: str, surface: str, run_completed: bool,
    challenge_revealed: bool, view_mode: str, config: Mapping[str, Any] | None = None,
) -> ArabicDecisionExplanation:
    """Build a plain-Arabic, observer-safe explanation.

    Explanations are intentionally unavailable before a challenge reveal.
    Facts, computed counterfactuals, recorded bot traces and hypotheses are
    labelled separately.  The function never claims to know an unrecorded
    human intention.
    """
    decision_id = str(decision.get("decisionId") or "")
    selected = decision.get("selectedAction")
    kind = str(decision.get("decisionKind") or "play")
    actor = str(decision.get("actor") or "unknown")
    relation = observer_relation(actor, observer or "")
    attribution = _REL_AR.get(relation, "اللاعب")

    if surface == "challenge" and not challenge_revealed:
        return ArabicDecisionExplanation(
            decision_id, False, "الشرح بعد كشف الإجابة",
            "اختر أولاً بدون تلميحات. بعد انتهاء الوقت يظهر سبب القرار وماذا كان سيحدث مع البدائل.",
            attribution, "challenge_hidden", (),
            "لا تُعرض أي إشارة أو مقارنة أو توصية قبل كشف الإجابة.",
            selected_action=selected, level=level, surface=surface,
        )

    own_or_complete = private_decision_review_allowed(
        actor=actor,
        observer=observer or "",
        run_completed=run_completed,
    )
    visibility = "trusted" if view_mode == "oracle_after_completion" and run_completed else "public_observation_only"
    title = f"ليش {attribution} {_action_label(selected)}؟"
    reasons: list[ArabicReason] = []
    metadata = dict(decision.get("decisionMetadata") or {})
    best, regret, drop, catastrophic = _audit_values(audit)

    if kind == "bid":
        public = dict(decision.get("publicState") or {})
        purchase = public.get("purchaseCard")
        hand_count = len(public.get("observerHand") or [])
        reasons.append(ArabicReason(
            "PUBLIC_BUY_CONTEXT", "حسب ورقه والشراء الظاهر",
            f"القرار اتخذ من {hand_count} ورقات مع ورقة الشراء الظاهرة{(' ' + _action_label(purchase)) if purchase else ''}، ومن نتيجة الصكّة وقتها.",
            "مؤكد", category="fact",
        ))
    elif own_or_complete:
        legal_count = len(decision.get("legalActions") or [])
        if legal_count <= 1:
            reasons.append(ArabicReason(
                "FORCED_MOVE", "ما كان عنده خيار ثاني",
                "هذه كانت الورقة القانونية الوحيدة، لذلك اللعب هنا إجباري وليس قراءة أو مخاطرة.",
                "مؤكد", category="fact",
            ))
        else:
            reasons.append(ArabicReason(
                "LEGAL_CHOICE", "كان عنده أكثر من خيار",
                f"كان أمامه {legal_count} خيارات قانونية، لذلك قارَن المحرك أثر كل خيار من نفس اللحظة.",
                "مؤكد", category="fact",
            ))

    if audit and own_or_complete:
        if best is not None and _action_label(best) == _action_label(selected):
            reasons.append(ArabicReason(
                "BEST_COUNTERFACTUAL", "النتيجة دعمت هذا اللعب",
                "عند إعادة نفس الحالة وتجربة البدائل، هذا القرار أعطى أفضل نتيجة متوقعة ضمن المعلومات التي كانت ظاهرة للاعب.",
                _certainty(audit.get("auditConfidence", .8)), category="computed",
            ))
        elif best is not None:
            reasons.append(ArabicReason(
                "BETTER_ALTERNATIVE", "كان يوجد لعب أقوى",
                f"إعادة الحالة أظهرت أن {_action_label(best)} كان أفضل من {_action_label(selected)} بفارق متوقع يقارب {regret:.1f} نقطة.",
                _certainty(audit.get("auditConfidence", .75)), category="computed",
            ))
        if drop >= .01:
            reasons.append(ArabicReason(
                "WIN_SWING", "أثر على فرصة الفوز",
                f"الفرق بين القرار وأفضل بديل غيّر فرصة الفوز بنحو {drop*100:.0f} نقطة مئوية.",
                _certainty(audit.get("auditConfidence", .75)), category="computed",
            ))
        if catastrophic:
            reasons.append(ArabicReason(
                "OUTCOME_CROSSING", "القرار قلب النتيجة",
                "أحد المسارين أبقى الصكّة في جهة الفوز، بينما المسار الآخر نقلها إلى الخسارة.",
                "ثقة عالية", category="computed",
            ))

        for item in list(audit.get("reasons") or [])[:3]:
            if not isinstance(item, Mapping):
                continue
            category = str(item.get("category") or item.get("evidenceType") or "computed")
            text = item.get("detailAr") or item.get("textAr") or item.get("descriptionAr")
            label = item.get("titleAr") or item.get("labelAr") or "سبب إضافي"
            if text:
                reasons.append(ArabicReason(
                    str(item.get("code") or "AUDIT_REASON"), str(label), str(text),
                    _certainty(item.get("confidence", .65), "fact" if category == "fact" else "computed"),
                    tuple(str(x) for x in item.get("evidence", []) if x), category=category,
                ))

    # The bot's recorded policy trace is authoritative provenance, but only
    # reveal it for the actor or after completion.
    if metadata.get("source") == "bot" and own_or_complete:
        profile = str(metadata.get("profileId") or "الملف الافتراضي")
        source = str(metadata.get("decisionSource") or "السياسة المبرمجة")
        rank = metadata.get("selectedRank")
        sentence = f"اللاعب الآلي استخدم إعداد {profile} وطريقة {source}."
        if rank is not None:
            sentence += f" هذا الخيار كان ترتيبه {int(rank)+1} عنده."
        if metadata.get("mistakeApplied"):
            sentence += " ملف الصعوبة سمح بخطأ مضبوط، لذلك لم يختر الأفضل."
        elif metadata.get("deceptionApplied"):
            sentence += " اختار بديلاً قريباً بقصد التنويع أو الخداع من دون تجاوز حد الخسارة المسموح."
        reasons.append(ArabicReason(
            "BOT_POLICY_TRACE", "سبب اللاعب الآلي مسجّل", sentence,
            "مؤكد", category="recorded",
        ))
    elif metadata.get("source") != "bot" and own_or_complete:
        feedback = list(decision.get("feedback") or [])
        if feedback:
            latest = feedback[0]
            reasons.append(ArabicReason(
                "HUMAN_RECORDED_REASON", "اللاعب أو المراجع سجّل سببه",
                str(latest.get("rationale") or "تم تسجيل سبب القرار."),
                _certainty(latest.get("confidence", .7)), category="recorded",
            ))
        else:
            reasons.append(ArabicReason(
                "INTENT_UNKNOWN", "نية اللاعب غير معروفة",
                "نقدر نفسر ما ظهر على الطاولة وما نتج عن القرار، لكن ما نقدر نجزم بنية اللاعب إلا إذا سجّلها بنفسه.",
                "مؤكد", category="unknown",
            ))

    # Public signal observations are hypotheses and are never shown during a
    # hidden challenge.  They may appear in replay with explicit caveats.
    if own_or_complete:
        signals = list((audit or {}).get("signals") or [])
        for signal in signals[:2]:
            if not isinstance(signal, Mapping):
                continue
            reasons.append(ArabicReason(
                "SIGNAL_HYPOTHESIS", "قراءة محتملة من ترتيب الورق",
                f"ظهر نمط {signal.get('kind','لعب متتابع')} من {signal.get('relationship','اللاعب')}. هذه قراءة محتملة فقط، وقد يكون اللعب إجبارياً أو تكتيكياً.",
                _certainty(signal.get("confidence", .45)), category="hypothesis",
            ))

    # Keep the basic view short; coach/expert can show more evidence.
    max_reasons = 3 if level == "basic" else (5 if level == "coach" else 8)
    reasons = reasons[:max_reasons]
    if audit and best is not None and _action_label(best) != _action_label(selected) and own_or_complete:
        summary = f"{attribution} لعب {_action_label(selected)}، لكن تجربة نفس الموقف أظهرت أن {_action_label(best)} كان أقوى."
    elif metadata.get("source") == "bot" and own_or_complete:
        summary = f"{attribution} اختار {_action_label(selected)} حسب ملفه الآلي والمعلومات التي كانت ظاهرة له وقت القرار."
    else:
        summary = f"{attribution} لعب {_action_label(selected)}. الشرح يعتمد على اللعب الظاهر والنتيجة المحسوبة، وليس على تخمين نية غير مسجلة."

    caveat = "الحقائق القانونية مؤكدة. المقارنات ناتجة من المحرك. الإشارات المحتملة لا تُعامل كحقيقة، ولا تُستخدم معلومات مخفية في عرض المشاهد."
    return ArabicDecisionExplanation(
        decision_id, True, title, summary, attribution, visibility,
        tuple(reasons), caveat, selected_action=selected,
        best_action=best if own_or_complete else None,
        level=level, surface=surface,
    )
