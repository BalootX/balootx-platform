from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from baloot_platform.challenge.evidence_v3 import (
    DecisionPerspectiveEvidenceV3,
    ObserverProjectionV1,
    RenderVariantV1,
    TruthLabel,
    TruthMode,
)
from baloot_platform.core.enums import Seat
from baloot_platform.video.perspective_adapter import build_perspective_decision_scene_package
from baloot_platform.video.remotion_renderer import RemotionProfessionalRenderer


def _projection(observer: Seat, actor: Seat) -> ObserverProjectionV1:
    return ObserverProjectionV1.from_public_mapping(
        "DECISION-7",
        observer,
        {
            "rulesHash": "r" * 64,
            "observer": observer.name.lower(),
            "observerHand": [
                {"id": 0, "rank": "seven", "suit": "spades"},
                {"id": 9, "rank": "eight", "suit": "hearts"},
            ],
            "knownUnplayed": {seat.name.lower(): [] for seat in Seat},
            "handSizes": [2, 2, 2, 2],
            "dealer": "west",
            "buyer": "north",
            "declarer": "north",
            "contract": "sun",
            "trumpSuit": None,
            "firstLeader": "north",
            "currentPlayer": actor.name.lower(),
            "matchScore": [44, 39],
            "plays": [
                {
                    "seat": "west",
                    "card": {"id": 18, "rank": "nine", "suit": "diamonds"},
                    "trickIndex": 4,
                    "position": 0,
                    "legalCards": [],
                    "ekka": False,
                    "baloot": False,
                }
            ],
            "projects": [],
            "balootTeams": [],
            "biddingHistory": [],
        },
    )


def _evidence(observer: Seat, actor: Seat, prompt: str) -> DecisionPerspectiveEvidenceV3:
    projection = _projection(observer, actor)
    actor_truth = observer == actor
    action = {
        "kind": "play_card",
        "card": {"id": 0, "rank": "seven", "suit": "spades"},
    }
    return DecisionPerspectiveEvidenceV3(
        decision_id=projection.decision_id,
        acting_seat=actor,
        observer_seat=observer,
        truth_mode=TruthMode.ACTOR_OBSERVER if actor_truth else TruthMode.SEAT_EXPERIENCE,
        projection_hash=projection.sha256,
        forced=False,
        truth_label=TruthLabel.ROBUST_BEST_KNOWN if actor_truth else TruthLabel.UNRESOLVED,
        confidence=0.91 if actor_truth else 0.0,
        evidence={"outcomeImpact": 88} if actor_truth else {"threatVisibility": 72},
        recommended_action=action if actor_truth else None,
        prompt_contract=prompt,
    )


def _variant(observer: Seat, rendition: str, prompt: str) -> RenderVariantV1:
    return RenderVariantV1(
        laffa_evidence_hash="b" * 64,
        decision_id="DECISION-7",
        observer_seat=observer,
        rendition_type=rendition,
        prompt_contract=prompt,
        reveal_policy="answer_then_reason" if rendition == "actor_challenge" else "interpretation_only",
        challenge_locator=f"https://balootx.local/c/{observer.name.lower()}-7",
    )


def _prompt(scene) -> dict:
    return next(layer["content"] for layer in scene.value["layers"] if layer["id"] == "decision-prompt")


def _analysis(scene) -> dict:
    return next(layer["content"] for layer in scene.value["layers"] if layer["id"] == "analysis")


def test_actor_perspective_emits_legal_choices_and_recommendation(root: Path) -> None:
    profile = yaml.safe_load((root / "configs/video/premium.yaml").read_text(encoding="utf-8"))
    projection = _projection(Seat.SOUTH, Seat.SOUTH)
    evidence = _evidence(Seat.SOUTH, Seat.SOUTH, "choose_acting_action")
    action = dict(evidence.recommended_action or {})
    package = build_perspective_decision_scene_package(
        projection=projection,
        evidence=evidence,
        variant=_variant(Seat.SOUTH, "actor_challenge", "choose_acting_action"),
        profile=profile,
        effective_config_sha256="a" * 64,
        legal_actions=(action,),
    )
    prompt = _prompt(package.scene("question"))
    analysis = _analysis(package.scene("reveal"))
    assert prompt["actingSeat"] == "south"
    assert prompt["legalChoices"] == [action]
    assert analysis["recommendedAction"] == action
    assert "observerInterpretation" not in analysis


def test_dynamic_arabic_reason_isolates_inline_card_labels(root: Path) -> None:
    profile = yaml.safe_load((root / "configs/video/premium.yaml").read_text(encoding="utf-8"))
    projection = _projection(Seat.SOUTH, Seat.SOUTH)
    evidence = _evidence(Seat.SOUTH, Seat.SOUTH, "choose_acting_action")
    variant = _variant(Seat.SOUTH, "actor_challenge", "choose_acting_action")
    renderer = RemotionProfessionalRenderer(profile, repository_root=root)

    bundle = renderer.build_perspective_render_request(
        projection=projection,
        evidence=evidence,
        variant=variant,
        effective_config_sha256="a" * 64,
        game_serial="BIDI-7",
        legal_actions=(dict(evidence.recommended_action or {}),),
        reveal_reason_key="video.analysis.only_winning_contract_rescue",
        reveal_reason_values={"best": "A♥", "alternative": "9♦"},
    )

    assert renderer._reveal_reason(bundle) == (
        "\u2066A♥\u2069 هو الوحيد اللي ينقذ المشتري؛ "
        "\u20669♦\u2069 تطيّر اللعب."
    )


def test_partner_perspective_has_typed_inference_and_no_actor_answer(root: Path) -> None:
    profile = yaml.safe_load((root / "configs/video/premium.yaml").read_text(encoding="utf-8"))
    projection = _projection(Seat.SOUTH, Seat.NORTH)
    evidence = _evidence(Seat.SOUTH, Seat.NORTH, "read_partner_signal")
    variant = _variant(Seat.SOUTH, "partner_inference", "read_partner_signal")
    package = build_perspective_decision_scene_package(
        projection=projection,
        evidence=evidence,
        variant=variant,
        profile=profile,
        effective_config_sha256="a" * 64,
    )
    prompt = _prompt(package.scene("question"))
    analysis = _analysis(package.scene("reveal"))
    assert prompt == {
        "textKey": "video.observer.read_partner_signal",
        "legalChoices": [],
        "actingSeat": "north",
        "promptContract": "read_partner_signal",
        "renditionType": "partner_inference",
    }
    assert "recommendedAction" not in analysis
    assert analysis["observerInterpretation"]["actingSeat"] == "north"

    renderer = RemotionProfessionalRenderer(profile, repository_root=root)
    bundle = renderer.build_perspective_render_request(
        projection=projection,
        evidence=evidence,
        variant=variant,
        effective_config_sha256="a" * 64,
        game_serial="PERSPECTIVE-7",
    )
    assert bundle.verify()
    assert bundle.value["observerSeat"] == "south"
    assert "recommendedAction" not in str(bundle.value)
    assert renderer._reveal_reason(bundle) == profile["localizations"]["video.observer.insight"]


def test_opponent_semantics_and_private_action_rejection(root: Path) -> None:
    profile = yaml.safe_load((root / "configs/video/premium.yaml").read_text(encoding="utf-8"))
    projection = _projection(Seat.EAST, Seat.NORTH)
    evidence = _evidence(Seat.EAST, Seat.NORTH, "predict_opponent_play")
    variant = _variant(Seat.EAST, "opponent_prediction", "predict_opponent_play")
    package = build_perspective_decision_scene_package(
        projection=projection,
        evidence=evidence,
        variant=variant,
        profile=profile,
        effective_config_sha256="a" * 64,
    )
    assert _prompt(package.scene("question"))["actingSeat"] == "north"
    with pytest.raises(ValueError, match="cannot receive"):
        build_perspective_decision_scene_package(
            projection=projection,
            evidence=evidence,
            variant=variant,
            profile=profile,
            effective_config_sha256="a" * 64,
            legal_actions=({"kind": "play_card", "card": {"id": 0}},),
        )


def test_perspective_adapter_rejects_wrong_relationship_and_projection() -> None:
    projection = _projection(Seat.WEST, Seat.NORTH)
    evidence = _evidence(Seat.WEST, Seat.NORTH, "read_partner_signal")
    with pytest.raises(ValueError, match="partner"):
        build_perspective_decision_scene_package(
            projection=projection,
            evidence=evidence,
            variant=_variant(Seat.WEST, "partner_inference", "read_partner_signal"),
            profile={"render": {"fps": 60, "width": 1080, "height": 1920, "locale": "ar-SA", "direction": "rtl"}, "timeline": {}},
            effective_config_sha256="a" * 64,
        )
