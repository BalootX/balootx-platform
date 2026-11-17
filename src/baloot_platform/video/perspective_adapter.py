"""Observer-safe adapter from foundry perspective evidence to video scenes."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from baloot_platform.challenge.evidence_v3 import (
    DecisionPerspectiveEvidenceV3,
    ObserverProjectionV1,
    RenderVariantV1,
    TruthMode,
    canonical_sha256,
)
from baloot_platform.core.enums import Seat

from .scene import build_scene_timeline, reduce_scene_document
from .scene_adapter import DecisionScenePackageV1


_PROMPT_KEYS = {
    "choose_acting_action": "video.decision.choose_best_move",
    "predict_opponent_play": "video.observer.predict_opponent_play",
    "read_partner_signal": "video.observer.read_partner_signal",
    "identify_threat": "video.observer.identify_threat",
    "plan_next_response": "video.observer.plan_next_response",
    "infer_distribution": "video.observer.infer_distribution",
    "post_reveal_analysis": "video.observer.post_reveal_analysis",
}


def _public_play(value: Mapping[str, Any]) -> dict[str, Any]:
    card = value.get("card")
    if not isinstance(card, Mapping):
        raise ValueError("observer projection play is missing its public card")
    seat = str(value.get("seat") or "")
    if seat not in {item.name.lower() for item in Seat}:
        raise ValueError("observer projection play has an invalid seat")
    return {
        "seat": seat,
        "card": dict(card),
        "trickIndex": int(value.get("trickIndex", 0)),
        "position": int(value.get("position", 0)),
        "ekka": bool(value.get("ekka", False)),
        "baloot": bool(value.get("baloot", False)),
    }


def _scene_inputs(projection: ObserverProjectionV1) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], dict[str, Any], dict[str, Any]]:
    information = dict(projection.information_state)
    if canonical_sha256(information) != projection.sha256:
        raise ValueError("observer projection checksum mismatch")
    observer = projection.observer_seat.name.lower()
    if information.get("observer") != observer:
        raise ValueError("observer projection and payload identities differ")
    observer_hand = information.get("observerHand")
    if not isinstance(observer_hand, list):
        raise ValueError("observer projection must contain one visible observer hand")

    raw_sizes = information.get("handSizes")
    if raw_sizes is None:
        hand_sizes = [len(observer_hand)] * 4
    elif isinstance(raw_sizes, list) and len(raw_sizes) == 4:
        hand_sizes = [int(value) for value in raw_sizes]
    else:
        raise ValueError("observer projection hand sizes must contain four seats")
    if any(value < 0 or value > 8 for value in hand_sizes):
        raise ValueError("observer projection hand size is out of bounds")
    if hand_sizes[int(projection.observer_seat)] != len(observer_hand):
        raise ValueError("observer projection hand count does not match its visible cards")

    plays = tuple(_public_play(item) for item in information.get("plays") or ())
    current_trick: list[dict[str, Any]] = []
    if plays:
        last_trick = max(int(item["trickIndex"]) for item in plays)
        candidate = [item for item in plays if int(item["trickIndex"]) == last_trick]
        if len(candidate) < 4:
            current_trick = candidate

    hands: dict[str, Any] = {}
    for seat in Seat:
        key = seat.name.lower()
        hands[key] = list(observer_hand) if seat == projection.observer_seat else {"count": hand_sizes[int(seat)]}
    table_keys = (
        "dealer", "buyer", "declarer", "contract", "trumpSuit", "firstLeader",
        "currentPlayer", "matchScore", "projects", "balootTeams",
        "biddingHistory", "escalationHistory", "purchaseCard", "completedTricks",
    )
    observer_view: dict[str, Any] = {
        "observer": observer,
        "phase": "playing" if "plays" in information else "bidding",
        "hands": hands,
        "currentTrick": {"plays": current_trick},
        "completedTricks": [{"index": index // 4, "plays": list(plays[index:index + 4])}
                            for index in range(0, len(plays) - len(current_trick), 4)],
        **{key: information[key] for key in table_keys if key in information},
    }
    public_events = tuple({"type": "card_played", **item} for item in plays)
    initial_hand = [*observer_hand, *(item["card"] for item in plays if item["seat"] == observer)]
    initial_hand.sort(key=lambda card: int(card["id"]))
    public_lifecycle = {
        "dealer": information.get("dealer"),
        "buyer": information.get("buyer"),
        "contract": information.get("contract"),
        "trumpSuit": information.get("trumpSuit"),
        "firstLeader": information.get("firstLeader"),
        "preDecisionMatchScore": list(information.get("matchScore") or (0, 0)),
        "matchScore": list(information.get("matchScore") or (0, 0)),
        "pauseBeforePlayIndex": len(plays),
        "plays": list(plays),
        "projects": list(information.get("projects") or ()),
        "bidHistory": list(information.get("biddingHistory") or ()),
    }
    observer_lifecycle = {"initialHandCards": initial_hand}
    return observer_view, public_events, public_lifecycle, observer_lifecycle


def _validate_variant(
    projection: ObserverProjectionV1,
    evidence: DecisionPerspectiveEvidenceV3,
    variant: RenderVariantV1,
) -> None:
    if len({projection.decision_id, evidence.decision_id, variant.decision_id}) != 1:
        raise ValueError("perspective render inputs must reference one decision")
    if projection.observer_seat != evidence.observer_seat or projection.observer_seat != variant.observer_seat:
        raise ValueError("perspective render inputs must reference one observer")
    if evidence.projection_hash != projection.sha256:
        raise ValueError("perspective evidence does not reference the supplied projection")
    if evidence.prompt_contract != variant.prompt_contract:
        raise ValueError("perspective evidence and render prompt contracts differ")
    if variant.rendition_type == "actor_challenge":
        if evidence.truth_mode != TruthMode.ACTOR_OBSERVER or evidence.observer_seat != evidence.acting_seat:
            raise ValueError("actor challenge requires acting-player truth")
    else:
        if evidence.truth_mode != TruthMode.SEAT_EXPERIENCE or evidence.observer_seat == evidence.acting_seat:
            raise ValueError("non-actor rendition requires seat-experience truth")
    if variant.rendition_type == "partner_inference" and evidence.observer_seat != evidence.acting_seat.partner:
        raise ValueError("partner inference must use the acting player's partner")
    if variant.rendition_type == "opponent_prediction" and evidence.observer_seat.team == evidence.acting_seat.team:
        raise ValueError("opponent prediction must use an opposing observer")


def build_perspective_decision_scene_package(
    *,
    projection: ObserverProjectionV1,
    evidence: DecisionPerspectiveEvidenceV3,
    variant: RenderVariantV1,
    profile: Mapping[str, Any],
    effective_config_sha256: str,
    legal_actions: Sequence[Mapping[str, Any]] = (),
    reveal_reason_key: str | None = None,
    complete_public_lifecycle: Mapping[str, Any] | None = None,
    complete_observer_lifecycle: Mapping[str, Any] | None = None,
) -> DecisionScenePackageV1:
    """Build a typed actor or non-actor decision clip without hidden state."""

    _validate_variant(projection, evidence, variant)
    observer_view, public_events, public_lifecycle, observer_lifecycle = _scene_inputs(projection)
    if complete_public_lifecycle is not None:
        public_lifecycle = dict(complete_public_lifecycle)
    if complete_observer_lifecycle is not None:
        observer_lifecycle = dict(complete_observer_lifecycle)
    if str(observer_lifecycle.get("seat", projection.observer_seat.name.lower())) != projection.observer_seat.name.lower():
        raise ValueError("complete observer lifecycle belongs to another seat")
    actor_prompt = evidence.truth_mode == TruthMode.ACTOR_OBSERVER
    safe_actions = tuple(dict(item) for item in legal_actions)
    if actor_prompt:
        if not evidence.forced and not safe_actions:
            raise ValueError("non-forced actor render requires its legal actions")
        if evidence.recommended_action is not None and dict(evidence.recommended_action) not in safe_actions:
            raise ValueError("actor recommendation must be present in the legal action set")
    elif safe_actions:
        raise ValueError("non-acting render cannot receive acting-player legal actions")

    prompt_key = _PROMPT_KEYS.get(str(evidence.prompt_contract))
    if prompt_key is None:
        raise ValueError("perspective prompt has no governed localization key")
    reason_key = reveal_reason_key or (
        "video.analysis.expert_reason" if actor_prompt else "video.observer.insight"
    )
    analysis: dict[str, Any] = {
        "truthMode": evidence.truth_mode.value,
        "truthLabel": evidence.truth_label.value,
        "confidence": evidence.confidence,
        "reasonKey": reason_key,
        "promptContract": evidence.prompt_contract,
        "actingSeat": evidence.acting_seat.name.lower(),
    }
    if actor_prompt and evidence.recommended_action is not None:
        analysis["recommendedAction"] = dict(evidence.recommended_action)
    else:
        analysis["observerInterpretation"] = {
            "promptContract": evidence.prompt_contract,
            "actingSeat": evidence.acting_seat.name.lower(),
        }

    render = profile["render"]
    timeline_profile = profile["timeline"]
    common = {
        "observer_view": observer_view,
        "public_events": public_events,
        "effective_config_sha256": effective_config_sha256,
        "locale": str(render["locale"]),
        "direction": str(render["direction"]),
    }
    prompt_semantics = {
        "acting_seat": evidence.acting_seat.name.lower(),
        "prompt_contract": str(evidence.prompt_contract),
        "rendition_type": variant.rendition_type,
    }
    question = reduce_scene_document(
        stage="question", prompt_key=prompt_key, legal_choices=safe_actions,
        **prompt_semantics, **common,
    )
    reveal = reduce_scene_document(
        stage="reveal", prompt_key=prompt_key, legal_choices=safe_actions,
        analysis_facts=analysis, **prompt_semantics, **common,
    )
    continuation = reduce_scene_document(
        stage="continuation", public_lifecycle_facts=public_lifecycle,
        observer_lifecycle_facts=observer_lifecycle, **common,
    )
    context = reduce_scene_document(
        stage="context",
        public_lifecycle_facts=public_lifecycle,
        observer_lifecycle_facts=observer_lifecycle,
        **common,
    )
    scenes = (context, question, reveal, continuation)
    timeline = build_scene_timeline(
        fps=int(render["fps"]), width=int(render["width"]), height=int(render["height"]),
        segments=(
            ("question", question.scene_id, timeline_profile["pause_seconds"]),
            ("reveal", reveal.scene_id, timeline_profile["reveal_seconds"]),
            ("continuation", continuation.scene_id, timeline_profile["continuation_seconds"]),
        ),
    )
    return DecisionScenePackageV1(scenes=scenes, timeline=timeline)
