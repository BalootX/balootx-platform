"""Internal construction of isolated observer projections.

The builder may derive all four projections inside trusted analysis, but callers
receive one requested observer at a time. This prevents a collection of safe
views from becoming an unsafe omniscient public response.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Protocol

from baloot_platform.analysis.deductions import derive_hard_deductions
from baloot_platform.analysis.models import InformationState
from baloot_platform.analysis.signals import detect_signals
from baloot_platform.core.cards import mask_to_cards
from baloot_platform.core.enums import Seat
from baloot_platform.core.play_ir import PlayDomainIRV1

from .evidence_v3 import (
    DecisionPerspectiveEvidenceV3,
    ObserverProjectionV1,
    TruthLabel,
    TruthMode,
)


class DecisionProjectionStore(Protocol):
    play_domain: PlayDomainIRV1

    def get_decision(self, decision_id: str) -> dict[str, Any] | None: ...
    def load_decision_payload(self, decision_id: str) -> dict[str, Any]: ...


_SEAT_EXPERIENCE_PROMPTS = frozenset({
    "predict_opponent_play",
    "read_partner_signal",
    "identify_threat",
    "plan_next_response",
    "infer_distribution",
    "post_reveal_analysis",
})


def _bid_projection(state: Any, bidding: Any, observer: Seat) -> dict[str, Any]:
    history = (
        getattr(bidding, "records", ())
        if bidding is not None
        else state.bid_history
    )
    return {
        "rulesHash": state.rules_hash,
        "observer": observer.name.lower(),
        "observerHand": [card.to_dict() for card in mask_to_cards(state.initial_five[int(observer)])],
        "purchaseCard": state.purchase_card.to_dict() if state.purchase_card else None,
        "dealer": state.dealer.name.lower(),
        "matchScore": list(state.match_score),
        "biddingHistory": [item.to_dict() for item in history],
        "biddingRound": getattr(bidding, "round_index", None),
        "biddingActor": bidding.actor.name.lower() if getattr(bidding, "actor", None) is not None else None,
        "publicDeductions": [],
        "signals": [],
    }


def build_observer_projection(
    store: DecisionProjectionStore,
    decision_id: str,
    observer: Seat | str,
) -> ObserverProjectionV1:
    decision = store.get_decision(decision_id)
    if decision is None:
        raise KeyError(decision_id)
    trusted = store.load_decision_payload(decision_id)
    seat = observer if isinstance(observer, Seat) else Seat.parse(observer)
    state = trusted["state"]
    if str(decision["decisionKind"]) == "play":
        information_state = InformationState.from_complete(state, seat)
        payload = information_state.to_dict()
        payload["publicDeductions"] = [
            item.to_dict() for item in derive_hard_deductions(information_state)
        ]
        payload["signals"] = [
            item.to_dict()
            for item in detect_signals(information_state, store.play_domain)
        ]
    else:
        payload = _bid_projection(state, trusted.get("bidding"), seat)
    return ObserverProjectionV1.from_public_mapping(decision_id, seat, payload)


def build_all_observer_projections(
    store: DecisionProjectionStore,
    decision_id: str,
) -> Mapping[Seat, ObserverProjectionV1]:
    """Trusted-only helper for analysis; never serialize this mapping publicly."""
    return {seat: build_observer_projection(store, decision_id, seat) for seat in Seat}


def observer_projection_timeline(
    store: DecisionProjectionStore,
    decisions: list[Mapping[str, Any]],
    observer: Seat | str,
) -> tuple[ObserverProjectionV1, ...]:
    seat = observer if isinstance(observer, Seat) else Seat.parse(observer)
    return tuple(build_observer_projection(store, str(item["decisionId"]), seat) for item in decisions)


def _actor_truth_label(audit: Mapping[str, Any]) -> TruthLabel:
    alternatives = tuple(audit.get("alternatives") or ())
    hidden_deals = int(audit.get("hiddenDeals", 0))
    effective_samples = float(audit.get("effectiveSampleSize", 0.0))
    method = str(audit.get("engineMethod", ""))
    exhaustive = method.startswith("exhaustive_constrained_hidden_deals")
    if (
        alternatives
        and hidden_deals > 0
        and effective_samples >= hidden_deals
        and exhaustive
        and float(audit.get("auditConfidence", 0.0)) >= 0.999999
        and all(
            float(item.get("exactShare", 0.0)) >= 0.999999
            and str(item.get("searchCompletionStatus") or "") == "exact"
            for item in alternatives
        )
    ):
        return TruthLabel.EXACT_OPTIMAL
    confidence = float(audit.get("auditConfidence", 0.0))
    if confidence >= 0.75:
        return TruthLabel.ROBUST_BEST_KNOWN
    if confidence >= 0.50 and float(audit.get("debateScore", 0.0)) >= 65.0:
        return TruthLabel.DEBATABLE
    return TruthLabel.UNRESOLVED


def derive_seat_experience_evidence(
    projection: ObserverProjectionV1,
    *,
    acting_seat: Seat,
    allowed_prompts: Iterable[str] = _SEAT_EXPERIENCE_PROMPTS,
) -> tuple[TruthLabel, float, dict[str, float], str]:
    """Score only facts and qualified signals visible in one isolated view."""

    observer = projection.observer_seat
    if observer == acting_seat:
        raise ValueError("seat-experience evidence requires a non-acting observer")
    allowed = frozenset(str(item).strip() for item in allowed_prompts if str(item).strip())
    unknown = allowed - _SEAT_EXPERIENCE_PROMPTS
    if unknown:
        raise ValueError(f"unsupported seat-experience prompts: {sorted(unknown)}")
    if not allowed:
        raise ValueError("seat-experience prompt policy cannot be empty")

    public = projection.information_state
    hand_sizes = tuple(int(item) for item in public.get("handSizes") or ())
    observer_cards = len(tuple(public.get("observerHand") or ()))
    known_by_seat = dict(public.get("knownUnplayed") or {})
    known_other_cards = sum(
        len(tuple(cards or ()))
        for seat, cards in known_by_seat.items()
        if str(seat) != observer.name.lower()
    )
    played_cards = len(tuple(public.get("plays") or ()))
    total_unplayed = (
        sum(hand_sizes)
        if len(hand_sizes) == 4
        else max(observer_cards + known_other_cards, 32 - played_cards)
    )
    hidden_unplayed = max(0, total_unplayed - observer_cards - known_other_cards)
    hidden_capacity = max(1, total_unplayed - observer_cards)
    uncertainty = 100.0 * hidden_unplayed / hidden_capacity

    deductions = tuple(dict(item) for item in public.get("publicDeductions") or ())
    signals = tuple(dict(item) for item in public.get("signals") or ())
    acting_name = acting_seat.name.lower()
    actor_deductions = tuple(
        item for item in deductions if str(item.get("seat") or "") == acting_name
    )
    actor_signals = tuple(
        item for item in signals if str(item.get("seat") or "") == acting_name
    )
    strongest_signal = max(
        (max(0.0, min(1.0, float(item.get("confidence", 0.0)))) for item in actor_signals),
        default=0.0,
    )
    exact_inference = min(100.0, 100.0 * len(actor_deductions) / 4.0)
    known_card_inference = min(100.0, 100.0 * known_other_cards / hidden_capacity)
    inference = max(exact_inference, known_card_inference)

    projects = tuple(dict(item) for item in public.get("projects") or ())
    public_threat = bool(projects) or str(public.get("escalation") or "normal") != "normal"
    threat_visibility = 100.0 if public_threat else 0.0
    partner_signal = strongest_signal * 100.0 if acting_seat == observer.partner else 0.0
    opponent_signal = strongest_signal * 100.0 if acting_seat.team != observer.team else 0.0
    next_decision_urgency = 100.0 if observer == acting_seat.next() else 0.0
    educational_value = max(inference, partner_signal, opponent_signal, threat_visibility)
    visual_value = 100.0 if projects or actor_signals else threat_visibility

    if actor_deductions or public_threat:
        label = TruthLabel.ROBUST_BEST_KNOWN
        confidence = 1.0
    elif strongest_signal >= 0.5:
        label = TruthLabel.DEBATABLE
        confidence = strongest_signal
    else:
        label = TruthLabel.UNRESOLVED
        confidence = strongest_signal

    preferred = (
        "infer_distribution" if actor_deductions else None,
        "read_partner_signal" if partner_signal > 0.0 else None,
        "predict_opponent_play" if opponent_signal > 0.0 else None,
        "identify_threat" if public_threat else None,
        "plan_next_response" if next_decision_urgency > 0.0 else None,
        "post_reveal_analysis",
    )
    prompt = next((item for item in preferred if item is not None and item in allowed), None)
    if prompt is None:
        raise ValueError("perspective policy cannot represent this observer experience")
    return label, confidence, {
        "publicStateUncertainty": uncertainty,
        "inferenceAbility": inference,
        "partnerSignalSignificance": partner_signal,
        "opponentThreatVisibility": max(threat_visibility, opponent_signal),
        "threatVisibility": threat_visibility,
        "predictionConfidence": strongest_signal * 100.0,
        "nextDecisionUrgency": next_decision_urgency,
        "educationalValue": educational_value,
        "visualValue": visual_value,
    }, prompt


def build_decision_perspective_evidence(
    store: DecisionProjectionStore,
    decision: Mapping[str, Any],
    audit: Mapping[str, Any],
    *,
    seat_experience_prompts: Iterable[str] = _SEAT_EXPERIENCE_PROMPTS,
) -> tuple[DecisionPerspectiveEvidenceV3, ...]:
    """Derive typed actor truth and three non-acting experience records."""
    decision_id = str(decision["decisionId"])
    actor = Seat.parse(str(decision["actor"]))
    projections = build_all_observer_projections(store, decision_id)
    forced = bool(audit.get("forced", len(decision.get("legalActions") or ()) <= 1))
    confidence = max(0.0, min(1.0, float(audit.get("auditConfidence", 0.0))))
    alternatives = tuple(dict(item) for item in audit.get("alternatives") or ())
    best = alternatives[0] if alternatives else {}
    second = alternatives[1] if len(alternatives) > 1 else best
    worst = alternatives[-1] if alternatives else best
    best_margin = float(best.get("expectedMargin", 0.0))
    second_margin = float(second.get("expectedMargin", best_margin))
    worst_margin = float(worst.get("expectedMargin", best_margin))
    best_probability = float(best.get("winProbability", 0.5))
    second_probability = float(second.get("winProbability", best_probability))
    worst_probability = float(worst.get("winProbability", best_probability))
    outcome_impact = min(100.0, max(
        max(0.0, best_margin - worst_margin) * 4.0,
        max(0.0, best_probability - worst_probability) * 100.0,
    ))
    uniqueness = min(100.0, max(
        max(0.0, best_margin - second_margin) * 8.0,
        max(0.0, best_probability - second_probability) * 100.0,
    ))
    catastrophic_alternative = (
        best_margin > 0.0 > worst_margin
        or best_probability >= 0.5 > worst_probability
    )
    principal_depth = max(
        (len(tuple(item.get("principalVariation") or ())) for item in alternatives),
        default=0,
    )
    best_action = audit.get("bestAction")
    if not forced and not isinstance(best_action, Mapping):
        raise ValueError("non-forced actor evidence requires a supported best action")
    actor_evidence = DecisionPerspectiveEvidenceV3(
        decision_id=decision_id,
        acting_seat=actor,
        observer_seat=actor,
        truth_mode=TruthMode.ACTOR_OBSERVER,
        projection_hash=projections[actor].sha256,
        forced=forced,
        truth_label=_actor_truth_label(audit),
        confidence=confidence,
        evidence={
            "outcomeImpact": outcome_impact,
            "uniqueness": uniqueness,
            "disasterAvoidance": 100.0 if catastrophic_alternative else outcome_impact,
            "tacticalDepth": min(100.0, principal_depth * 12.5),
            "informationBurden": max(0.0, min(100.0, float(audit.get("predictedDifficulty", 0.0)))),
            "executionDifficulty": min(
                100.0,
                max(0, len(alternatives) - 1) * 20.0 + principal_depth * 7.5,
            ),
            "surprise": max(0.0, min(100.0, 100.0 - float(audit.get("qualityScore", 100.0)))),
            "robustness": confidence * 100.0,
            "confidence": confidence * 100.0,
            "explanationFidelity": 100.0 if audit.get("reasons") else 0.0,
            "visualClarity": 85.0 if decision.get("decisionKind") == "play" else 70.0,
        },
        recommended_action=None if forced else dict(best_action),
        prompt_contract="choose_acting_action",
    )
    values = [actor_evidence]
    for observer, projection in projections.items():
        if observer == actor:
            continue
        label, observer_confidence, observer_evidence, prompt = (
            derive_seat_experience_evidence(
                projection,
                acting_seat=actor,
                allowed_prompts=seat_experience_prompts,
            )
        )
        values.append(
            DecisionPerspectiveEvidenceV3(
                decision_id=decision_id,
                acting_seat=actor,
                observer_seat=observer,
                truth_mode=TruthMode.SEAT_EXPERIENCE,
                projection_hash=projection.sha256,
                forced=forced,
                truth_label=label,
                confidence=observer_confidence,
                evidence=observer_evidence,
                prompt_contract=prompt,
            )
        )
    return tuple(values)
