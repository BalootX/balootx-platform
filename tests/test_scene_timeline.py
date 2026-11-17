from __future__ import annotations

import copy
import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import jsonschema
import pytest
import yaml

from baloot_platform.core.engine import BalootEngine
from baloot_platform.audit.models import content_hash
from baloot_platform.core.enums import Contract, Seat
from baloot_platform.config import _semantic_validate, build_bundle
from baloot_platform.core.episode_io import load_episode_spec
from baloot_platform.video.scene_adapter import (
    build_legacy_episode_scene_package,
    observer_safe_episode_payload,
)
from baloot_platform.video.captions import derive_caption_files
from baloot_platform.video.remotion_renderer import RemotionProfessionalRenderer
from baloot_platform.video.render_projection import (
    build_observer_safe_render_episode,
    rebuild_render_episode_from_continuation_scene,
)
from baloot_platform.video.scene import (
    SceneDocumentV1,
    _walk_leaves,
    build_scene_timeline,
    create_signed_deep_link_payload,
    reduce_scene_document,
    verify_signed_deep_link_payload,
)


def _observer_view() -> dict:
    return {
        "rulesHash": "1" * 64,
        "roundNumber": 3,
        "phase": "playing",
        "observer": "south",
        "dealer": "west",
        "buyer": "north",
        "declarer": "north",
        "contract": "sun",
        "trumpSuit": None,
        "firstLeader": "north",
        "currentPlayer": "south",
        "matchScore": [24, 19],
        "hands": {
            "north": {"count": 3},
            "east": {"count": 3},
            "south": [
                {"id": 0, "rank": "7", "suit": "spades"},
                {"id": 9, "rank": "8", "suit": "hearts"},
                {"id": 18, "rank": "9", "suit": "diamonds"},
            ],
            "west": {"count": 3},
        },
        "currentTrick": {
            "leader": "north",
            "plays": [
                {"seat": "north", "card": {"id": 1, "rank": "8", "suit": "spades"}}
            ],
        },
        "completedTricks": [],
        "projects": [],
        "events": [],
    }


def _scene(root: Path, *, stage: str = "question"):
    document = reduce_scene_document(
        observer_view=_observer_view(),
        public_events=({"type": "card_played", "seat": "north", "cardId": 1},),
        effective_config_sha256="a" * 64,
        stage=stage,
        locale="ar-SA",
        direction="rtl",
        prompt_key="video.decision.choose_best_move",
        acting_seat="south",
        prompt_contract="choose_acting_action",
        rendition_type="actor_challenge",
        legal_choices=(
            {"kind": "play_card", "cardId": 0},
            {"kind": "play_card", "cardId": 9},
        ),
        analysis_facts=(
            {
                "recommendedAction": {"kind": "play_card", "cardId": 9},
                "reasonKey": "analysis.preserve_control",
            }
            if stage == "reveal"
            else None
        ),
    )
    schema = json.loads(
        (root / "schemas/contracts/scene-document-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator(schema).validate(document.to_dict())
    assert document.verify()
    return document


def test_scene_document_is_schema_valid_and_fully_classified(root: Path) -> None:
    document = _scene(root)
    value = document.to_dict()
    assert value["stage"] == "question"
    assert value["fieldDisclosures"]
    assert all(
        "analysis" not in layer["id"]
        for layer in value["layers"]
    )


def test_same_observer_view_produces_identical_pre_reveal_scene(root: Path) -> None:
    first = _scene(root)
    second_view = copy.deepcopy(_observer_view())
    # This represents a different trusted completion outside the reducer.  The
    # reducer accepts only the identical observer view, so it cannot distinguish
    # or leak the alternative hidden deal.
    second = reduce_scene_document(
        observer_view=second_view,
        public_events=({"type": "card_played", "seat": "north", "cardId": 1},),
        effective_config_sha256="a" * 64,
        stage="question",
        locale="ar-SA",
        direction="rtl",
        prompt_key="video.decision.choose_best_move",
        acting_seat="south",
        prompt_contract="choose_acting_action",
        rendition_type="actor_challenge",
        legal_choices=(
            {"kind": "play_card", "cardId": 0},
            {"kind": "play_card", "cardId": 9},
        ),
    )
    assert first.scene_id == second.scene_id
    assert first.to_dict() == second.to_dict()


def test_reducer_accepts_authoritative_engine_observer_projection(rules) -> None:
    engine = BalootEngine(rules)
    state = engine.create_round(seed=21_008, dealer=Seat.EAST)
    engine.shuffle_and_initial_deal(state, cut_position=11)
    engine.bid_and_complete_deal(
        state,
        forced_contract=Contract.SUN,
        forced_buyer=Seat.SOUTH,
    )
    engine.prepare_projects(state)
    view = engine.observer_view(state, Seat.SOUTH)
    document = reduce_scene_document(
        observer_view=view,
        public_events=tuple(view["events"]),
        effective_config_sha256="a" * 64,
        stage="context",
        locale="ar-SA",
        direction="rtl",
    )
    assert document.verify()
    assert document.to_dict()["observer"] == "south"


def test_scene_rejects_hidden_or_future_information() -> None:
    leaked = _observer_view()
    leaked["seed"] = 42
    with pytest.raises(ValueError, match="forbidden field"):
        reduce_scene_document(
            observer_view=leaked,
            public_events=(),
            effective_config_sha256="a" * 64,
            stage="context",
            locale="ar-SA",
            direction="rtl",
        )

    leaked_hand = _observer_view()
    leaked_hand["hands"]["north"] = [{"id": 6, "rank": "A", "suit": "spades"}]
    with pytest.raises(ValueError, match="non-observer hand"):
        reduce_scene_document(
            observer_view=leaked_hand,
            public_events=(),
            effective_config_sha256="a" * 64,
            stage="context",
            locale="ar-SA",
            direction="rtl",
        )


@pytest.mark.parametrize("leak", ("future_play", "terminal_score", "future_deal", "future_own_card", "future_event", "terminal_event", "future_trick", "known_card_metadata"))
def test_context_lifecycle_is_a_verified_decision_prefix(leak: str) -> None:
    view = _observer_view()
    past_card = {"id": 27, "rank": "J", "suit": "clubs"}
    known_plays = [{"seat": "south", "card": past_card},
                   *view["currentTrick"]["plays"]]
    view["completedTricks"] = [{"plays": known_plays[:1]}]
    events = [{"type": "card_played", **play} for play in known_plays]
    future_card = {"id": 31, "rank": "A", "suit": "clubs"}
    future_play = {"seat": "south", "card": future_card}
    document = reduce_scene_document(
        observer_view=view, public_events=events, effective_config_sha256="a" * 64,
        stage="context", locale="ar-SA", direction="rtl",
        public_lifecycle_facts={"plays": [*known_plays, future_play],
                                "roundScore": {"awardedPoints": [0, 38]},
                                "dealSequence": [{"card": future_card}]},
        observer_lifecycle_facts={"initialHandCards": [future_card],
                                  "initialFiveCards": [{**view["hands"]["south"][0], "futureScore": [0, 38]}, future_card]},
    )
    assert document.verify()
    value = document.to_dict()
    layers = {layer["id"]: layer["content"] for layer in value["layers"]}
    assert layers["public-lifecycle"]["plays"] == known_plays
    assert layers["public-lifecycle"]["pauseBeforePlayIndex"] == 2
    assert "roundScore" not in layers["public-lifecycle"]
    assert "dealSequence" not in layers["public-lifecycle"]
    assert {card["id"] for card in layers["observer-lifecycle"]["initialHandCards"]} == {0, 9, 18, 27}
    assert layers["observer-lifecycle"]["initialFiveCards"] == [view["hands"]["south"][0]]
    if leak == "future_play":
        layers["public-lifecycle"]["plays"].append(future_play)
    elif leak == "terminal_score":
        layers["public-lifecycle"]["roundScore"] = {"awardedPoints": [0, 38]}
    elif leak == "future_deal":
        layers["public-lifecycle"]["dealSequence"] = [{"card": future_card}]
    elif leak == "future_own_card":
        layers["observer-lifecycle"]["initialHandCards"].append(future_card)
    elif leak == "future_event":
        layers["public-events"]["items"].append({"type": "card_played", **future_play})
        layers["public-lifecycle"]["plays"].append(future_play)
        layers["public-lifecycle"]["pauseBeforePlayIndex"] += 1
    elif leak == "terminal_event":
        layers["public-events"]["items"].append({"type": "round_scored", "score": {"awardedPoints": [0, 38]}})
    elif leak == "future_trick":
        layers["public-events"]["items"].append({"type": "trick_completed", "trickIndex": 7,
                                                "winner": "east", "rawPoints": 42})
    else:
        layers["observer-lifecycle"]["initialFiveCards"][0]["futureScore"] = [0, 38]
    core = {key: item for key, item in value.items() if key not in {"sceneId", "fieldDisclosures"}}
    value["sceneId"] = content_hash(core)
    value["fieldDisclosures"] = {
        path: core["layers"][int(path.split("/")[2])]["disclosure"]
        if path.startswith("/layers/") else "public" for path in _walk_leaves(core)
    }
    assert not SceneDocumentV1(value).verify()


def test_analysis_is_forbidden_before_reveal() -> None:
    with pytest.raises(ValueError, match="forbidden before reveal"):
        reduce_scene_document(
            observer_view=_observer_view(),
            public_events=(),
            effective_config_sha256="a" * 64,
            stage="question",
            locale="ar-SA",
            direction="rtl",
            analysis_facts={"recommendedAction": {"cardId": 9}},
        )


def test_timeline_is_frame_authoritative_and_9_by_16_only(root: Path) -> None:
    question = _scene(root)
    reveal = _scene(root, stage="reveal")
    timeline = build_scene_timeline(
        fps=60,
        width=1080,
        height=1920,
        segments=(
            ("question", question.scene_id, Decimal("6.5")),
            ("reveal", reveal.scene_id, Decimal("3.0")),
        ),
    )
    value = timeline.to_dict()
    assert value["segments"][0]["durationFrames"] == 390
    assert value["segments"][1]["startFrame"] == 390
    assert value["totalFrames"] == 570
    assert timeline.verify()
    schema = json.loads(
        (root / "schemas/contracts/scene-timeline-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator(schema).validate(value)

    for width, height in ((1920, 1080), (1080, 1080)):
        with pytest.raises(ValueError, match="9:16"):
            build_scene_timeline(
                fps=60,
                width=width,
                height=height,
                segments=(("question", question.scene_id, 1),),
            )


def test_release_video_profile_schema_rejects_landscape_and_square(root: Path) -> None:
    schema = json.loads(
        (root / "schemas/profiles/video.schema.json").read_text(encoding="utf-8")
    )
    profile = yaml.safe_load(
        (root / "configs/video/premium.yaml").read_text(encoding="utf-8")
    )
    validator = jsonschema.Draft202012Validator(schema)
    validator.validate(profile)
    for width, height in ((1920, 1080), (1080, 1080)):
        invalid = copy.deepcopy(profile)
        invalid["render"]["width"] = width
        invalid["render"]["height"] = height
        with pytest.raises(jsonschema.ValidationError):
            validator.validate(invalid)
        with pytest.raises(ValueError, match="9:16"):
            _semantic_validate("video", invalid)

    player_leak = copy.deepcopy(profile)
    player_leak["presentation"]["show_state_locator"] = True
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(player_leak)

    expert = copy.deepcopy(profile)
    expert["presentation"]["mode"] = "expert"
    expert["presentation"]["show_identity_pill"] = True
    expert["presentation"]["show_developer_card_codes"] = True
    validator.validate(expert)


def test_hitl_player_profile_hides_developer_codes_and_stays_9_by_16(
    root: Path,
) -> None:
    schema = json.loads(
        (root / "schemas/profiles/hitl.schema.json").read_text(encoding="utf-8")
    )
    profile = yaml.safe_load(
        (root / "configs/hitl/governance_v18.yaml").read_text(encoding="utf-8")
    )
    validator = jsonschema.Draft202012Validator(schema)
    validator.validate(profile)

    player_leak = copy.deepcopy(profile)
    player_leak["review_clip"]["show_developer_card_codes"] = True
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(player_leak)

    expert = copy.deepcopy(profile)
    expert["review_clip"]["presentation_mode"] = "expert"
    expert["review_clip"]["show_developer_card_codes"] = True
    validator.validate(expert)


def test_deep_link_is_minimal_signed_and_tamper_evident(root: Path) -> None:
    scene = _scene(root)
    key = bytes(range(32))
    token = create_signed_deep_link_payload(
        scene_id=scene.scene_id,
        decision_id="decision-42",
        observer="south",
        signing_key=key,
    )
    value = verify_signed_deep_link_payload(token, signing_key=key)
    assert value == {
        "schemaVersion": 1,
        "sceneId": scene.scene_id,
        "decisionId": "decision-42",
        "observer": "south",
    }
    changed = ("A" if token[0] != "A" else "B") + token[1:]
    with pytest.raises(ValueError):
        verify_signed_deep_link_payload(changed, signing_key=key)


def test_legacy_episode_adapter_emits_only_observer_safe_semantic_sidecars(
    root: Path,
) -> None:
    bundle = build_bundle(root)
    episode = load_episode_spec(
        root / "examples/v7-outcome-flip-source.json",
        bundle.rules,
    )
    profile = yaml.safe_load(
        (root / "configs/video/premium.yaml").read_text(
            encoding="utf-8"
        )
    )
    package = build_legacy_episode_scene_package(episode, profile)
    assert package.timeline.verify()
    context_layers = {layer["id"]: layer["content"] for layer in package.scene("context").value["layers"]}
    question_layers = {layer["id"]: layer["content"] for layer in package.scene("question").value["layers"]}
    for layer_id in ("table", "public-events", "observer-hand", "seats"):
        assert context_layers[layer_id] == question_layers[layer_id]
    assert "roundScore" not in context_layers["public-lifecycle"]
    assert context_layers["table"]["roundScore"] is None
    continuation = package.scene("continuation").value
    assert all(layer["disclosure"] == "post_reveal_analysis" for layer in continuation["layers"]
               if layer["id"] in {"public-lifecycle", "observer-lifecycle"})
    full = next(layer["content"] for layer in continuation["layers"] if layer["id"] == "public-lifecycle")
    assert len(full["completedTricks"]) == 8
    assert full["roundScore"] is not None
    assert [item.value["stage"] for item in package.scenes] == [
        "context",
        "question",
        "reveal",
        "explanation",
        "continuation",
        "outcome",
    ]
    for scene in package.scenes:
        assert scene.verify()
        value = scene.to_dict()
        encoded = json.dumps(value, ensure_ascii=False)
        assert '"seed"' not in encoded
        assert '"initialHands"' not in encoded
        assert '"deckOrder"' not in encoded
        assert '"principalVariation"' not in encoded
        observer_layers = [
            item for item in value["layers"] if item["id"] == "observer-hand"
        ]
        assert len(observer_layers) == 1
        assert observer_layers[0]["content"]["seat"] == "south"


def test_decision_pixels_read_the_frozen_scene_package(root: Path) -> None:
    bundle = build_bundle(root)
    episode = load_episode_spec(
        root / "examples/v7-outcome-flip-source.json",
        bundle.rules,
    )
    profile = yaml.safe_load(
        (root / "configs/video/premium.yaml").read_text(
            encoding="utf-8"
        )
    )
    package = build_legacy_episode_scene_package(episode, profile)
    original = episode.analysis.recommended_card
    alternative = next(
        item.card for item in episode.analysis.actions if item.card != original
    )
    changed = replace(
        episode,
        analysis=replace(episode.analysis, recommended_card=alternative),
    )
    changed_package = build_legacy_episode_scene_package(changed, profile)
    frozen_reveal = package.scene("reveal").to_dict()
    changed_reveal = changed_package.scene("reveal").to_dict()

    assert package.scene("reveal").scene_id != changed_package.scene("reveal").scene_id
    assert frozen_reveal != changed_reveal
    assert package.scene("reveal").to_dict() == frozen_reveal


def test_v21_public_episode_payload_excludes_trusted_replay_state(
    root: Path,
) -> None:
    bundle = build_bundle(root)
    episode = load_episode_spec(
        root / "examples/v7-outcome-flip-source.json",
        bundle.rules,
    )
    profile = yaml.safe_load(
        (root / "configs/video/premium.yaml").read_text(
            encoding="utf-8"
        )
    )
    package = build_legacy_episode_scene_package(episode, profile)
    payload = observer_safe_episode_payload(episode, package)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)

    assert payload["observerSafe"] is True
    assert payload["semanticScenes"]["questionSceneId"] == package.scene(
        "question"
    ).scene_id
    for forbidden in (
        '"seed"',
        '"deckOrder"',
        '"dealTrace"',
        '"initialFive"',
        '"initialHands"',
        '"futureCards"',
        '"analysis"',
    ):
        assert forbidden not in encoded


def test_v21_question_scene_excludes_events_after_the_visible_decision(root: Path) -> None:
    bundle = build_bundle(root)
    episode = load_episode_spec(
        root / "examples/v7-outcome-flip-source.json",
        bundle.rules,
    )
    profile = yaml.safe_load(
        (root / "configs/video/premium.yaml").read_text(encoding="utf-8")
    )
    package = build_legacy_episode_scene_package(episode, profile)
    question = package.scene("question").to_dict()
    events = next(
        layer["content"]["items"]
        for layer in question["layers"]
        if layer["id"] == "public-events"
    )
    table = next(
        layer["content"]
        for layer in question["layers"]
        if layer["id"] == "table"
    )
    completed_count = len(table["completedTricks"])
    visible_plays = len(table["currentTrick"]["plays"])

    assert not any(item["type"] == "round_scored" for item in events)
    assert not any(
        item["type"] == "card_played"
        and (
            item["trickIndex"] > completed_count
            or (
                item["trickIndex"] == completed_count
                and item["position"] >= visible_plays
            )
        )
        for item in events
    )


def test_v21_renderer_projection_excludes_trusted_episode_state(
    root: Path,
) -> None:
    bundle = build_bundle(root)
    episode = load_episode_spec(
        root / "examples/v7-outcome-flip-source.json",
        bundle.rules,
    )
    profile = yaml.safe_load(
        (root / "configs/video/premium.yaml").read_text(
            encoding="utf-8"
        )
    )
    package = build_legacy_episode_scene_package(episode, profile)
    projection = build_observer_safe_render_episode(
        episode,
        package.scene("reveal"),
    )
    state = projection.final_state

    assert not hasattr(projection, "seed")
    assert not hasattr(state, "seed")
    assert not hasattr(state, "hands")
    assert not hasattr(state.shuffle, "seed")
    assert not hasattr(state.shuffle, "deck_order")
    for seat in Seat:
        if seat == episode.viewer:
            assert state.initial_hands[int(seat)]
            assert state.initial_five[int(seat)]
        else:
            assert state.initial_hands[int(seat)] == 0
            assert state.initial_five[int(seat)] == 0
    for record in state.deal_trace:
        if record.seat not in {episode.viewer, None}:
            assert all(card is None for card in record.cards)
    assert all(
        not play.legal_cards
        for trick in state.completed_tricks
        for play in trick.plays
    )
    assert set(projection.audit_summary) <= {
        "actualAction",
        "bestAction",
        "expectedRegret",
        "winProbabilityDrop",
        "qualityLabel",
        "qualityScore",
        "auditConfidence",
        "actualIntent",
        "sharedComparison",
    }


def test_v21_lifecycle_renderer_state_round_trips_through_context_scene(
    root: Path,
) -> None:
    bundle = build_bundle(root)
    episode = load_episode_spec(
        root / "examples/v7-outcome-flip-source.json",
        bundle.rules,
    )
    profile = yaml.safe_load(
        (root / "configs/video/premium.yaml").read_text(
            encoding="utf-8"
        )
    )
    package = build_legacy_episode_scene_package(episode, profile)
    envelope = build_observer_safe_render_episode(
        episode,
        package.scene("reveal"),
    )
    changed_envelope = replace(
        envelope,
        final_state=replace(
            envelope.final_state,
            match_score=(999, 999),
        ),
    )
    rebuilt = rebuild_render_episode_from_continuation_scene(
        changed_envelope,
        package.scene("continuation"),
    )

    assert rebuilt.state_before_pause == envelope.state_before_pause
    assert rebuilt.final_state == envelope.final_state
    context = package.scene("context").to_dict()
    encoded = json.dumps(context, ensure_ascii=False, sort_keys=True)
    assert '"seed"' not in encoded
    assert '"deckOrder"' not in encoded
    assert '"initialHands"' not in encoded
    assert '"legalCards"' not in encoded


def test_remotion_prepares_scene_backed_lifecycle_request(root: Path) -> None:
    bundle = build_bundle(root)
    episode = load_episode_spec(
        root / "examples/v7-outcome-flip-source.json",
        bundle.rules,
    )
    profile = yaml.safe_load(
        (root / "configs/video/premium.yaml").read_text(
            encoding="utf-8"
        )
    )
    renderer = RemotionProfessionalRenderer(profile, repository_root=root)
    render_bundle = renderer.build_render_request(episode, export_preset="hooked_flagship")
    context = next(scene for scene in render_bundle.value["scenes"] if scene["stage"] == "context")
    encoded = json.dumps(context, ensure_ascii=False, sort_keys=True)
    assert render_bundle.verify()
    assert '"hands"' not in encoded
    assert '"initialHands"' not in encoded


def test_export_frame_timing_is_resolved_by_scene_timeline(root: Path) -> None:
    bundle = build_bundle(root)
    episode = load_episode_spec(
        root / "examples/v7-outcome-flip-source.json",
        bundle.rules,
    )
    profile = yaml.safe_load(
        (root / "configs/video/premium.yaml").read_text(
            encoding="utf-8"
        )
    )
    renderer = RemotionProfessionalRenderer(profile, repository_root=root)
    render_bundle = renderer.build_render_request(episode, export_preset="hooked_flagship")
    timeline = render_bundle.value["timeline"]

    assert timeline["frameRate"] == {"numerator": 60, "denominator": 1}
    assert timeline["totalFrames"] == 4320
    assert [(item["stage"], item["startFrame"], item["durationFrames"]) for item in timeline["segments"]] == [
        ("hook", 0, 108),
        ("rewind", 108, 24),
        ("replay", 132, 2748),
        ("question", 2880, 300),
        ("reveal", 3180, 300),
        ("continuation", 3480, 600),
        ("outcome", 4080, 240),
    ]


def test_v21_authority_map_records_the_current_renderer_boundary(
    root: Path,
) -> None:
    authority_map = json.loads(
        (root / "docs/architecture/authority-map.json").read_text(
            encoding="utf-8"
        )
    )
    scenes = authority_map["authorities"]["scenesAndTimelines"]
    boundary = scenes["currentRendererBoundary"]

    assert scenes["owner"] == (
        "baloot_platform.video.scene.reduce_scene_document"
    )
    assert boundary == {
        "decisionPixelsConsumeCanonicalScenes": True,
        "fullLifecyclePixelsConsumeCanonicalScenes": True,
        "fullLifecycleTimingConsumesCanonicalTimeline": True,
        "rendererReceivesTrustedEpisode": False,
        "rendererInputProjection": "ObserverSafeRenderEpisodeProjectionV1",
        "publicEpisodePayloadIsObserverSafe": True,
        "remainingWork": (
            "No renderer state or timing authority remains outside "
            "SceneDocumentV1 and SceneTimelineV1."
        ),
    }


def test_caption_derivatives_share_one_srt_authority(tmp_path: Path) -> None:
    srt = tmp_path / "clip.srt"
    vtt = tmp_path / "clip.vtt"
    transcript = tmp_path / "clip.txt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:03,500\nاختر الورقة الأفضل\n\n"
        "2\n00:00:04,000 --> 00:00:06,000\nالاختيار الأقوى\n",
        encoding="utf-8",
    )
    derive_caption_files(srt, vtt, transcript)
    assert vtt.read_text(encoding="utf-8") == (
        "WEBVTT\n\n"
        "1\n00:00:01.000 --> 00:00:03.500\nاختر الورقة الأفضل\n\n"
        "2\n00:00:04.000 --> 00:00:06.000\nالاختيار الأقوى\n"
    )
    assert transcript.read_text(encoding="utf-8") == (
        "اختر الورقة الأفضل\nالاختيار الأقوى\n"
    )
