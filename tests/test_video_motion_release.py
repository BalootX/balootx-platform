from __future__ import annotations

import json
import re
import wave
from pathlib import Path

import yaml
import pytest

from baloot_platform.audit.models import content_hash
from baloot_platform.config import build_bundle
from baloot_platform.config import validate_profile_payload
from baloot_platform.core.episode_io import load_episode_spec
from baloot_platform.video.remotion_renderer import RemotionProfessionalRenderer
from baloot_platform.video.media_validation import (
    REMOTION_RENDERER_VERSION,
    compare_images,
    renderer_source_files,
    renderer_source_sha256,
    validate_loudness,
    validate_media_probe,
)
from baloot_platform.video.audio_postprocess import (
    AUDIO_FINISHER_ID,
    build_audio_cue_plan,
    generate_original_foley,
)
from baloot_platform.video.render_bundle import RenderBundleV1


def _profile(root: Path) -> dict:
    return yaml.safe_load(
        (root / "configs/video/premium.yaml").read_text(encoding="utf-8")
    )


def test_professional_video_is_the_closed_player_default(root: Path) -> None:
    bundle = build_bundle(root)
    video = bundle.video
    assert video["renderer_plugin"] == "premium_video"
    assert video["render"]["fps"] == 60
    assert video["presentation"]["show_trick_winner_text"] is False
    assert video["presentation"]["physical_trick_collection"] is True
    assert video["presentation"]["mode"] == "player"
    assert video["presentation"]["show_identity_pill"] is True
    assert video["presentation"]["show_state_locator"] is False
    assert video["presentation"]["show_developer_card_codes"] is False


def test_motion_is_spatial_restrained_and_reduced_motion_is_supported(
    root: Path,
) -> None:
    configured = _profile(root)
    assert 0.3 <= configured["timeline"]["between_cards_seconds"] <= 0.4
    assert configured["motion"]["reduced"]["collection_steps"] == 2
    source = (
        root / "src/baloot_platform/video/remotion/src/professional-prototype.tsx"
    ).read_text(encoding="utf-8")
    assert "translate" in source
    assert "reducedMotion" in source


def test_research_audit_video_badge_is_governed_persistent_and_qr_safe(
    root: Path,
) -> None:
    profile = _profile(root)
    key = "video.governance.research_audit_badge"
    assert profile["profile"]["version"] == "21.3.5"
    assert profile["localizations"][key] == "للبحث والتدقيق فقط — غير مؤهل للنشر"

    missing = json.loads(json.dumps(profile))
    del missing["localizations"][key]
    with pytest.raises(ValueError):
        validate_profile_payload("video", missing, root=root)

    source = (
        root / "src/baloot_platform/video/remotion/src/professional-prototype.tsx"
    ).read_text(encoding="utf-8")
    types = (root / "src/baloot_platform/video/remotion/src/types.ts").read_text(
        encoding="utf-8"
    )
    styles = (root / "src/baloot_platform/video/remotion/src/style.css").read_text(
        encoding="utf-8"
    )
    assert f'localized("{key}")' in source
    assert "bundle.source !== undefined" in source
    assert 'source?.qualificationStatus === "qualified"' in source
    assert 'source.renderPurpose === "content_candidate"' in source
    assert "source.publicationEligible === true" in source
    assert "&& !auditMedia" in source
    assert "bundle.source.qrActionable === true" in source
    assert "show_identity_pill) || auditMedia" in source
    assert "bundle.source?.sourceSchemaVersion === 3" in source
    assert "data-decision-reference={publicDecisionReference}" in source
    assert "data-exact-state-id={legacyExactStateId}" in source
    assert "bundle.identity.gameSerial" in source
    assert "exactStateId: string" in types
    assert "qrActionable: boolean" in types
    assert "publicationBlockers: string[]" in types
    assert "HistoricalRenderSourceV1" in types
    assert "CurrentRenderSourceV2" in types
    assert "ObserverSafeScenarioRenderSourceV3" in types
    assert "sourceSchemaVersion?: never" in types
    assert "sourceSchemaVersion: 2" in types
    assert "sourceSchemaVersion: 3" in types
    assert "publicProjectionHash: string" in types
    assert "exactStateId?: never" in types
    for lineage_field in (
        "resultHash",
        "traceHash",
        "analysisHash",
        "auditHash",
        "analysisProfileHash",
        "analyzerId",
        "analyzerVersion",
    ):
        assert f"{lineage_field}: string" in types
    assert ".audit-media-badge" in styles


def test_video_and_playable_copy_have_no_second_authority_in_source(root: Path) -> None:
    profile = _profile(root)
    configured_keys = (
        set(profile["localizations"])
        | set(profile["challenge_copy"])
        | set(profile["preview_copy"])
    )
    sources = [
        root / "src/baloot_platform/video/remotion/src/root.tsx",
        root / "src/baloot_platform/video/remotion/src/professional-prototype.tsx",
        root / "src/baloot_platform/service/static/player/index.html",
        root / "src/baloot_platform/service/static/player/app.js",
        root / "src/baloot_platform/service/static/challenge/index.html",
        root / "src/baloot_platform/service/static/challenge/app.js",
        root / "src/baloot_platform/challenge/analysis_catalog.py",
    ]
    for path in sources:
        source = path.read_text(encoding="utf-8")
        assert not re.search(r"[\u0600-\u06ff]", source), path

    preview_sources = [
        root / "src/baloot_platform/service/static/scene-preview/index.html",
        root / "src/baloot_platform/service/static/scene-preview/app.js",
        root / "scripts/generate_scene_preview.py",
    ]
    for path in preview_sources:
        source = path.read_text(encoding="utf-8")
        assert not re.search(r"[\u0600-\u06ff]", source), path

    episode_sources = [
        root / "src/baloot_platform/core/scenario.py",
        root / "src/baloot_platform/core/bid_scenario.py",
        root / "src/baloot_platform/core/episode_copy.py",
        root / "src/baloot_platform/core/episode_io.py",
        root / "src/baloot_platform/learning/service.py",
        root / "scripts/build_v6_challenge_episode.py",
    ]
    for path in episode_sources:
        source = path.read_text(encoding="utf-8")
        assert not re.search(r"[\u0600-\u06ff]", source), path

    preview_html = preview_sources[0].read_text(encoding="utf-8")
    preview_keys = set(re.findall(r'data-copy-key="([a-zA-Z0-9_]+)"', preview_html))
    preview_script = preview_sources[1].read_text(encoding="utf-8")
    preview_keys |= set(re.findall(r'copy\("([a-zA-Z0-9_]+)"\)', preview_script))
    assert preview_keys <= set(profile["preview_copy"])
    assert all(str(value).strip() for value in profile["preview_copy"].values())

    for template in (
        sources[2].read_text(encoding="utf-8"),
        sources[4].read_text(encoding="utf-8"),
    ):
        assert 'lang="ar-SA"' in template
        assert 'translate="no"' in template
        assert '<meta name="google" content="notranslate">' in template

    remotion = sources[1].read_text(encoding="utf-8")
    direct_video_keys = set(re.findall(r'localized\("([^"]+)"\)', remotion))
    assert direct_video_keys <= configured_keys

    player_template = sources[2].read_text(encoding="utf-8")
    placeholders = set(re.findall(r"\{\{([a-zA-Z0-9_]+)\}\}", player_template))
    assert (
        placeholders - {"player_copy_json", "player_asset_version"} <= configured_keys
    )

    challenge_template = sources[4].read_text(encoding="utf-8")
    challenge_placeholders = set(
        re.findall(r"\{\{([a-zA-Z0-9_]+)\}\}", challenge_template)
    )
    assert challenge_placeholders == set()
    challenge_source = sources[5].read_text(encoding="utf-8")
    assert "bindFrozenCopy" in challenge_source
    assert "copyProfile" in challenge_source

    player_script = sources[3].read_text(encoding="utf-8")
    challenge_script = sources[5].read_text(encoding="utf-8")
    assert "data?.detail" not in player_script
    assert "value.detail ||" not in challenge_script
    assert "copy('request_error'" in player_script
    assert "copy('request_error'" in challenge_script


def test_mobile_choice_breakdown_uses_bounded_card_geometry(root: Path) -> None:
    styles = (root / "src/baloot_platform/service/static/challenge/app.css").read_text(
        encoding="utf-8"
    )
    assert ".choice-card .answer-card" in styles
    assert "grid-template-columns: 44px minmax(0, 1fr) 48px" in styles
    assert ".choice-card { width: 40px; height: 56px" in styles


def test_renderer_release_decodes_the_rendered_qr_before_accepting_media(
    root: Path,
) -> None:
    renderer = RemotionProfessionalRenderer(_profile(root), repository_root=root)
    assert renderer.descriptor.version == REMOTION_RENDERER_VERSION
    assert "qr_decode_validation" in renderer.descriptor.capabilities
    assert "decode_validation" in renderer.descriptor.capabilities
    assert "bt709_tagging" in renderer.descriptor.capabilities

    renderer_source = (
        root / "src/baloot_platform/video/remotion_renderer.py"
    ).read_text(encoding="utf-8")
    verifier_source = (
        root / "src/baloot_platform/video/remotion/scripts/verify-qr.mjs"
    ).read_text(encoding="utf-8")
    package = json.loads(
        (root / "src/baloot_platform/video/remotion/package.json").read_text(
            encoding="utf-8"
        )
    )

    assert '"verify-qr.mjs"' in renderer_source
    assert '"qrValidation": qr_validation' in renderer_source
    assert '"mediaProbe": media_probe' in renderer_source
    assert "result.data !== expectedValue" in verifier_source
    assert '["required", "forbidden"].includes(policy)' in verifier_source
    assert 'policy === "required"' in verifier_source
    assert 'policy === "forbidden"' in verifier_source
    assert '"BALOOTX_QR_POLICY": qr_policy' in renderer_source
    assert '"BALOOTX_QR_REASON": qr_reason' in renderer_source
    renderer_sources = renderer_source_files(root)
    assert (
        root / "src/baloot_platform/video/remotion/src/style.css"
    ) in renderer_sources
    assert package["dependencies"]["jpeg-js"] == "0.4.4"
    assert package["dependencies"]["jsqr"] == "1.4.0"

    ffmpeg_profile = (
        root / "src/baloot_platform/video/remotion/scripts/ffmpeg-profile.mjs"
    ).read_text(encoding="utf-8")
    assert '"-color_primaries"' in ffmpeg_profile
    assert '"-color_trc"' in ffmpeg_profile
    assert (
        '"colorprim=bt709:transfer=bt709:colormatrix=bt709:range=limited"'
        in ffmpeg_profile
    )
    assert "...args.slice(0, -1)" in ffmpeg_profile
    assert "args.at(-1)" in ffmpeg_profile
    for name in (
        "render-decision-clip.mjs",
        "render-flagship.mjs",
        "render-multi-angle.mjs",
        "render-prototypes.mjs",
    ):
        source = (
            root / f"src/baloot_platform/video/remotion/scripts/{name}"
        ).read_text(encoding="utf-8")
        assert "ffmpegOverride: bt709FfmpegOverride" in source
        assert "governedH264Options" in source

    governed_options = (
        root / "src/baloot_platform/video/remotion/scripts/render-options.mjs"
    ).read_text(encoding="utf-8")
    assert 'x264Preset: render.preset' in governed_options
    assert 'concurrency: Math.min(render.concurrency, availableParallelism(), RenderInternals.getMaxConcurrency(), cgroupCpuLimit())' in governed_options
    assert '"-AttachRootId", String(process.pid)' in governed_options
    assert 'sampleRate: render.audio_rate' in governed_options


def test_media_gate_requires_complete_bt709_and_exact_frame_count() -> None:
    probe = {
        "durationSeconds": 28.0,
        "video": {
            "codec": "h264",
            "width": 1080,
            "height": 1920,
            "frameRate": "60/1",
            "pixelFormat": "yuv420p",
            "colorSpace": "bt709",
            "colorPrimaries": "bt709",
            "colorTransfer": "bt709",
            "decodedFrames": 1680,
        },
        "audio": {"codec": "aac", "sampleRate": 48000, "channels": 2},
    }
    validate_media_probe(
        probe,
        width=1080,
        height=1920,
        fps=60,
        minimum_frames=1680,
        expected_frames=1680,
        audio_rate=48000,
    )

    missing_primary = json.loads(json.dumps(probe))
    missing_primary["video"]["colorPrimaries"] = None
    with pytest.raises(ValueError, match="media profile"):
        validate_media_probe(
            missing_primary,
            width=1080,
            height=1920,
            fps=60,
            minimum_frames=1680,
            expected_frames=1680,
            audio_rate=48000,
        )


def test_audio_loudness_and_source_frame_parity_are_release_gates(
    tmp_path: Path,
) -> None:
    validate_loudness(
        {
            "integratedLufs": -18.8,
            "truePeakDbtp": -1.6,
            "targetLufs": -18.0,
        }
    )
    with pytest.raises(ValueError, match="loudness"):
        validate_loudness(
            {
                "integratedLufs": -21.0,
                "truePeakDbtp": -1.6,
                "targetLufs": -18.0,
            }
        )
    from PIL import Image

    source = tmp_path / "source.png"
    encoded = tmp_path / "encoded.png"
    Image.new("RGB", (32, 32), (20, 80, 50)).save(source)
    Image.new("RGB", (32, 32), (21, 81, 51)).save(encoded)
    evidence = compare_images(source, encoded)
    assert evidence["psnrDb"] > 28

    Image.new("RGB", (32, 32), (240, 20, 220)).save(encoded)
    with pytest.raises(ValueError, match="source-versus-encoded"):
        compare_images(source, encoded)


def test_observer_safe_audio_plan_and_original_foley_are_deterministic(
    root: Path, tmp_path: Path
) -> None:
    historical = {"renderer": {}, "invalidHistoricalEvidence": True}
    # Restamping old evidence cannot turn a leaked context into a safe bundle.
    historical["renderer"]["version"] = REMOTION_RENDERER_VERSION
    historical["renderer"]["sourceSha256"] = renderer_source_sha256(root)
    historical["bundleSha256"] = content_hash({key: value for key, value in historical.items() if key != "bundleSha256"})
    assert not RenderBundleV1(historical).verify()
    with pytest.raises(ValueError, match="verified render bundle"):
        build_audio_cue_plan(RenderBundleV1(historical))
    configuration = build_bundle(root)
    episode = load_episode_spec(root / "examples/v7-outcome-flip-source.json", configuration.rules)
    renderer = RemotionProfessionalRenderer(_profile(root), repository_root=root)
    bundle = renderer.build_render_request(episode, export_preset="decision_clip")
    assert any(
        path.name == "audio_postprocess.py" for path in renderer_source_files(root)
    )
    first_plan = build_audio_cue_plan(bundle)
    second_plan = build_audio_cue_plan(bundle)
    assert first_plan == second_plan
    assert first_plan["finisher"] == AUDIO_FINISHER_ID
    assert first_plan["music"] is False
    assert first_plan["targetLufs"] == -18
    assert first_plan["normalizationTruePeakDbtp"] == -2.5
    assert {event["kind"] for event in first_plan["events"]} == {
        "card",
        "collect",
        "reveal",
        "timer",
        "outcome",
    }
    encoded = json.dumps(first_plan, ensure_ascii=False, sort_keys=True)
    assert all(
        forbidden not in encoded
        for forbidden in ("initialHands", "deckOrder", "trustedState", "futureCards")
    )

    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    first_evidence = generate_original_foley(first_plan, first)
    second_evidence = generate_original_foley(first_plan, second)
    assert first_evidence["sha256"] == second_evidence["sha256"]
    assert first_evidence["eventCount"] == len(first_plan["events"])
    with wave.open(str(first), "rb") as stream:
        assert stream.getframerate() == 48_000
        assert stream.getnchannels() == 2
        assert stream.getnframes() == 20 * 48_000


def test_governed_copy_uses_concise_saudi_baloot_terms(root: Path) -> None:
    profile = _profile(root)
    video = profile["localizations"]
    playable = profile["challenge_copy"]

    assert video["video.decision.choose_best_move"] == "وش تلعب؟"
    assert video["video.decision.generated_question"] == "وش تلعب؟ وليه؟"
    assert video["video.bid.challenge_question"] == "وش تشوف الشراء؟"
    assert video["video.review.play_title"] == "مراجعة اللفّة"
    assert video["video.role.dealer"] == "الموزع"
    assert video["video.role.buyer"] == "المشتري"
    assert video["video.bid.sun"] == "صن"
    assert video["video.bid.hokum"] == "حكم"
    assert video["video.analysis.one_winning_action"] == (
        "إثبات OR-Tools CP-SAT أكد أن فيه لعبة واحدة فقط تحقق النتيجة المطلوبة."
    )
    assert "المحرك" not in video["video.analysis.one_winning_action"]
    assert video["video.purchase.hokum_summary"] == "{contract} {round} {trump}"
    assert video["video.purchase.plain_summary"] == "{contract} {round}"
    assert playable["player_your_turn"] == "دورك"
    assert playable["player_decision_play"] == "وش بتلعب؟"
    assert playable["player_bid_pass"] == "بس"
    assert playable["purchase_hokum_summary"] == "{contract} {round} {trump}"
    assert playable["purchase_plain_summary"] == "{contract} {round}"

    remotion = (
        root / "src/baloot_platform/video/remotion/src/professional-prototype.tsx"
    ).read_text(encoding="utf-8")
    player = (root / "src/baloot_platform/service/static/player/app.js").read_text(
        encoding="utf-8"
    )
    challenge = (
        root / "src/baloot_platform/service/static/challenge/app.js"
    ).read_text(encoding="utf-8")
    assert "video.purchase.hokum_summary" in remotion
    assert "purchase_hokum_summary" in player
    assert "purchase_hokum_summary" in challenge
    assert "primaryDecisionReason" in player
    assert "HUMAN_INTENT_UNKNOWN" in player


def test_editable_copy_templates_preserve_required_dynamic_values(root: Path) -> None:
    profile = _profile(root)
    valid = json.loads(json.dumps(profile))
    valid["localizations"]["video.purchase.hokum_summary"] = (
        "{contract} · {trump} · {round}"
    )
    validate_profile_payload("video", valid, root=root)

    missing = json.loads(json.dumps(profile))
    missing["localizations"]["video.purchase.hokum_summary"] = "{contract} {round}"
    with pytest.raises(ValueError, match="must contain exactly"):
        validate_profile_payload("video", missing, root=root)

    unknown = json.loads(json.dumps(profile))
    unknown["challenge_copy"]["buyer_with_name"] = "{buyer} {seat}"
    with pytest.raises(ValueError, match="must contain exactly"):
        validate_profile_payload("video", unknown, root=root)

    legacy = json.loads(json.dumps(profile))
    del legacy["localizations"]["video.analysis.unique_awarded_margin"]
    validate_profile_payload("video", legacy, root=root)

    scope_legacy = json.loads(json.dumps(profile))
    for key in list(scope_legacy["localizations"]):
        if key.startswith("video.analysis.proof_scope.") or key == "video.analysis.worst_case_awarded_margin":
            del scope_legacy["localizations"][key]
    validate_profile_payload("video", scope_legacy, root=root)
    partial_scope = json.loads(json.dumps(profile))
    del partial_scope["localizations"]["video.analysis.proof_scope.unproven"]
    with pytest.raises(ValueError):
        validate_profile_payload("video", partial_scope, root=root)
    blank_scope = json.loads(json.dumps(profile))
    blank_scope["localizations"]["video.analysis.proof_scope.complete_state"] = " "
    with pytest.raises(ValueError):
        validate_profile_payload("video", blank_scope, root=root)

    bad_margin = json.loads(json.dumps(profile))
    bad_margin["localizations"]["video.analysis.unique_awarded_margin"] = "{selectedMargin}"
    with pytest.raises(ValueError, match="must contain exactly"):
        validate_profile_payload("video", bad_margin, root=root)


def test_voice_pack_is_original_and_media_correct(root: Path) -> None:
    directory = root / "assets/audio/voice_packs/ar_sa_baloot_v2_smooth"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    required = {
        "pass",
        "sun",
        "hokum",
        "hokum_second",
        "ashkal",
        "wala",
        "double",
        "triple",
        "four",
        "gahwa",
        "siri",
        "fifty",
        "hundred",
        "four_hundred",
        "baloot",
        "explanation",
    }
    assert required <= set(manifest["clips"])
    assert manifest["copiedReferenceAudio"] is False
    assert 8 <= manifest["clips"]["explanation"]["durationSeconds"] <= 14
    for key in required:
        with wave.open(str(directory / manifest["clips"][key]["file"]), "rb") as wav:
            assert wav.getframerate() == 48_000 and wav.getnchannels() == 1

    source = (
        root / "src/baloot_platform/video/remotion/src/professional-prototype.tsx"
    ).read_text(encoding="utf-8")
    assert (
        'staticFile("audio/voice_packs/ar_sa_baloot_v2_smooth/explanation.wav")'
        not in source
    )
    assert "bundle.audioCues.find" in source
    profile = yaml.safe_load(
        (root / "configs/video/premium.yaml").read_text(encoding="utf-8")
    )
    assert profile["audio"]["explanation_voice_mode"] == "exact_transcript_only"
    assert profile["audio"]["one_piece_explanation"] is False
    assert (
        profile["audio"]["explanation_voice_transcript"]
        == manifest["clips"]["explanation"]["textAr"]
    )


def test_physical_collection_has_no_winner_copy(root: Path) -> None:
    configured = _profile(root)
    renderer = RemotionProfessionalRenderer(configured, repository_root=root)
    assert configured["presentation"]["physical_trick_collection"] is True
    assert "physical_trick_collection" in renderer.descriptor.capabilities
    source = (
        root / "src/baloot_platform/video/remotion/src/professional-prototype.tsx"
    ).read_text(encoding="utf-8")
    assert "winnerText" not in source
    assert "collect" in source.lower()


def test_show_justification_text_false_hides_reason_and_proof_text_only(
    root: Path,
) -> None:
    source = (
        root / "src/baloot_platform/video/remotion/src/professional-prototype.tsx"
    ).read_text(encoding="utf-8")
    # Absent or true keeps existing exports; only an explicit false suppresses.
    assert (
        "const showJustification = presentation.show_justification_text !== false;"
        in source
    )
    reveal_start = source.index("const revealPanel = showReveal ? (")
    reveal = source[reveal_start : source.index(") : null;", reveal_start)]
    guard_start = reveal.index('{showJustification ? <div className="reason-text">')
    guarded = reveal[guard_start : reveal.index("</div> : null}", guard_start)]
    for text in ("<p>{reason}</p>", "localized(proofScopeKey)", "localized(sunTieKey)"):
        assert text in guarded
    # Selected action stays visible in the reveal regardless of the flag.
    assert reveal.index('className="winning-move-code"') < guard_start
    assert "data-selected-action={selectedActionCode" in reveal[:guard_start]
    proof = source[: source.index('className="reveal-reason proof-boundary-only"')]
    assert proof.rstrip().endswith(
        "{showOutcome && !terminalOutcomeAvailable && showJustification ? (\n        <section"
    )
    # Every rendered reason/proof/sun-tie string sits behind the guard.
    assert source.count("{reason}") == 2
    assert source.count("localized(proofScopeKey)") == 1
    assert source.count("localized(sunTieKey)") == 1
    assert "audioEnabled && showJustification" in source
    # Public session code box is red independent of the research notice.
    assert '<div className="idnum game-number is-notice">' in source
