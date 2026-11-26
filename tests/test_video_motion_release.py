from __future__ import annotations

import json
import re
import shutil
import wave
from pathlib import Path

import yaml
import pytest

from baloot_platform.audit.models import content_hash
from baloot_platform.config import build_bundle
from baloot_platform.config import validate_profile_payload
from baloot_platform.core.episode_io import load_episode_spec
from baloot_platform.video import media_validation, render_bundle
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
        root / "src/baloot_platform/service/static/app/index.html",
        root / "src/baloot_platform/service/static/app/app.js",
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

    app_template = sources[7].read_text(encoding="utf-8")
    assert '<html lang="ar" dir="rtl">' in app_template
    assert 'id="shareCode" dir="ltr"' in app_template
    app_placeholders = set(re.findall(r"\{\{([a-zA-Z0-9_]+)\}\}", app_template))
    assert app_placeholders - {"app_copy_json"} <= set(profile["challenge_copy"])
    app_keys = set(
        re.findall(r"copy\('([a-zA-Z0-9_]+)'", sources[8].read_text(encoding="utf-8"))
    )
    assert app_keys and app_keys <= set(profile["challenge_copy"])

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


def test_active_browser_api_helpers_share_one_request_deadline(root: Path) -> None:
    service = root / "src/baloot_platform/service"
    shared = (service / "static/shared/request.js").read_text(encoding="utf-8")
    assert shared.count("REQUEST_DEADLINE_MS =") == 1
    assert "AbortSignal.timeout(REQUEST_DEADLINE_MS)" in shared
    assert "AbortSignal.any([caller, deadline])" in shared

    assert "AbortSignal.timeout(LONG_CALL_DEADLINE_MS)" in shared
    assert shared.count("AbortSignal.timeout(") == 2

    # Pages reach the helper only through the loader, which drops a failed load.
    loader = (service / "static/shared/request-loader.js").read_text(encoding="utf-8")
    assert "'/shared-assets/request.js'" in loader
    assert "pending = null;" in loader
    loader_tag = '<script src="/shared-assets/request-loader.js" defer></script>'
    bare_fetch = re.compile(r"(?<![.\w])fetch\(")
    for page, assets in (
        ("app", "/app-assets/"),
        ("challenge", "/challenge-assets/"),
        ("player", "/play-assets/"),
    ):
        script = (service / f"static/{page}/app.js").read_text(encoding="utf-8")
        assert "await loadRequestHelper()" in script, page
        assert "import(" not in script, page
        assert "request_timeout" in script, page
        assert "request_unavailable" in script, page
        assert not bare_fetch.search(script), page
        template = (service / f"static/{page}/index.html").read_text(encoding="utf-8")
        assert template.count(loader_tag) == 1, page
        assert template.index(loader_tag) < template.index(f"{assets}app.js"), page

    foundry = service / "foundry_dashboard/src"
    for source in sorted(foundry.rglob("*.ts*")):
        if "generated" not in source.parts:
            assert not bare_fetch.search(source.read_text(encoding="utf-8")), source
    assert "defaultClient.setConfig({fetch: fetchWithDeadline})" in (
        foundry / "api/research.ts"
    ).read_text(encoding="utf-8")
    assert '"static/shared/*.js"' in (root / "pyproject.toml").read_text(
        encoding="utf-8"
    )


def _request_constant(shared: str, name: str) -> int:
    declared = re.findall(rf"^export const {name} = ([0-9_]+);$", shared, re.M)
    assert len(declared) == 1, name
    return int(declared[0].replace("_", ""))


def _interactive_wall_seconds(value: object) -> list[float]:
    if isinstance(value, dict):
        found = [
            float(profile["maximumWallSeconds"])
            for key, profile in value.items()
            if key == "interactive_validation_v1"
            and isinstance(profile, dict)
            and "maximumWallSeconds" in profile
        ]
        return found + [
            seconds
            for item in value.values()
            for seconds in _interactive_wall_seconds(item)
        ]
    if isinstance(value, list):
        return [
            seconds for item in value for seconds in _interactive_wall_seconds(item)
        ]
    return []


def test_browser_deadlines_exceed_the_server_budgets_they_cover(root: Path) -> None:
    import inspect

    from baloot_platform.challenge.analysis_catalog import (
        FoundryAnalysisCandidateServiceV1,
    )
    from baloot_platform.research.play_compiler import OrToolsCpSatPlayCompilerV1
    from baloot_platform.research.project_compiler import (
        OrToolsCpSatProjectCompilerV1,
    )
    from baloot_platform.research.render_service import (
        ResearchRenderPreparationServiceV1,
    )

    service = root / "src/baloot_platform/service"
    shared = (service / "static/shared/request.js").read_text(encoding="utf-8")
    request_ms = _request_constant(shared, "REQUEST_DEADLINE_MS")
    long_ms = _request_constant(shared, "LONG_CALL_DEADLINE_MS")
    margin_ms = _request_constant(shared, "LONG_CALL_MARGIN_MS")
    assert margin_ms >= 10_000

    def solver_seconds(compiler: type) -> float:
        # The solver stops at compile's default; the descriptor declares the same budget.
        default = inspect.signature(compiler.compile).parameters["timeout_seconds"]
        return max(float(default.default), float(compiler.descriptor.timeout_seconds))

    # :verify-play-compiler answers after the project compiler and then the play compiler.
    research = (root / "src/baloot_platform/research/service.py").read_text(
        encoding="utf-8"
    )
    verify = research.split("    def verify_play_compiler(", 1)[1]
    verify = verify.split("    def list_disagreements(", 1)[0]
    assert verify.count(".verify_trace(") == 2
    verify_ms = 1000 * (
        solver_seconds(OrToolsCpSatProjectCompilerV1)
        + solver_seconds(OrToolsCpSatPlayCompilerV1)
    )
    prepare_ms = 1000 * max(
        float(FoundryAnalysisCandidateServiceV1.descriptor.timeout_seconds),
        float(ResearchRenderPreparationServiceV1.descriptor.timeout_seconds),
    )
    assert long_ms >= max(verify_ms, prepare_ms) + margin_ms

    # Every call the Foundry client makes to a covered endpoint uses the long deadline.
    main = (service / "foundry_dashboard/src/main.tsx").read_text(encoding="utf-8")
    assert "api<T>(path, init, fetchWithLongDeadline)" in main
    assert main.count("await longApi<") == 9
    assert "await api<ResearchPreparedRender>(" not in main
    assert not re.search(
        r"\bapi<[^(]*\(`[^`]*:(?:prepare|prepare-best|prepare-render"
        r"|verify-play-compiler|compile)`",
        main,
    )

    # Interactive calls stay inside the default deadline with the same margin.
    interactive = [
        seconds
        for path in sorted((root / "configs").rglob("*.yaml"))
        if "interactive_validation_v1" in path.read_text(encoding="utf-8")
        for seconds in _interactive_wall_seconds(
            yaml.safe_load(path.read_text(encoding="utf-8"))
        )
    ]
    assert interactive
    assert request_ms >= 1000 * max(interactive) + margin_ms


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
        '"colorprim=bt709:transfer=bt709:colormatrix=bt709:range=limited:cpu-independent=1"'
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
        if name == "render-decision-clip.mjs":
            assert 'const measure = args[0] === "--measure-layout";' in source
            assert 'const outputBase = measure ? "NEVER_POST-layout-measurement" :' in source
            assert '...(measure ? {disposition: "NEVER_POST", postingAuthorized: false} : {})' in source
            assert (
                "ffmpegOverride: measure ? ({args}) => {\n"
                "      const governed = bt709FfmpegOverride({args});\n"
                '      return [...governed.slice(0, -1), "-metadata", '
                '`comment=${layoutMeasurementMarker}`, governed.at(-1)];\n'
                "    } : bt709FfmpegOverride,"
            ) in source
        else:
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


@pytest.mark.parametrize(("integrated", "peak", "passes"), [
    (-20.9, -1.5, True), (-17.9, -1.5, True),
    (-20.91, -2.0, False), (-17.89, -2.0, False), (-19.4, -1.49, False),
    (float("nan"), -2.0, False), (-19.4, float("-inf"), False),
])
def test_release_loudness_boundaries(integrated, peak, passes) -> None:
    evidence = {"integratedLufs": integrated, "truePeakDbtp": peak, "targetLufs": -19.4}
    if passes:
        validate_loudness(evidence)
    else:
        with pytest.raises(ValueError, match="loudness"):
            validate_loudness(evidence)


def test_source_frame_parity_is_a_release_gate(
    tmp_path: Path,
) -> None:
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
    assert first_plan["seedDerivationVersion"] == 2
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


def test_renderer_provenance_changes_leave_foley_and_identity_unchanged(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import render_research_notice as notice

    fingerprint_root = tmp_path / "fingerprint"
    for path in renderer_source_files(root):
        target = fingerprint_root / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    fingerprint = renderer_source_sha256
    assert fingerprint(fingerprint_root) == fingerprint(root)
    scripts = fingerprint_root / "src/baloot_platform/video/remotion/scripts"
    for name in ("approved-layout-adjacencies.json", "ffmpeg-runtime.json"):
        source = scripts / name
        original = source.read_bytes()
        source.write_bytes(original + b"\n")
        assert fingerprint(fingerprint_root) != fingerprint(root)
        source.write_bytes(original)
    monkeypatch.setattr(media_validation, "renderer_source_sha256", lambda _: fingerprint(fingerprint_root))
    monkeypatch.setattr(render_bundle, "renderer_source_sha256", lambda _: fingerprint(fingerprint_root))

    rules = build_bundle(root).rules
    episode = load_episode_spec(root / "examples/v7-outcome-flip-source.json", rules)
    renderer = RemotionProfessionalRenderer(_profile(root), repository_root=root)

    def capture() -> dict:
        bundles = {
            "notice": notice.build_notice_bundle(rules, notice.notice_profile(public=True)),
            "v7-decision": renderer.build_render_request(episode, export_preset="decision_clip"),
            "flagship": renderer.build_render_request(episode, export_preset="hooked_flagship"),
        }
        return {
            key: (bundle.value["identity"], build_audio_cue_plan(bundle))
            for key, bundle in bundles.items()
        }

    baseline = capture()
    baseline_foley = {
        key: generate_original_foley(plan, tmp_path / f"{key}-base.wav")["sha256"]
        for key, (_, plan) in baseline.items()
    }
    source = fingerprint_root / "src/baloot_platform/video/audio_postprocess.py"
    lock = source.parent / "remotion/package-lock.json"
    source_bytes, lock_bytes = source.read_bytes(), lock.read_bytes()
    lock_value = json.loads(lock_bytes)
    package = lock_value["packages"]["node_modules/source-map-js"]
    major, minor, patch = package["version"].split(".")
    package["version"] = f"{major}.{minor}.{int(patch) + 1}"
    for mutation, path, changed, original in (
        ("comment", source, source_bytes + b"\n# fingerprint test\n", source_bytes),
        ("lock", lock, json.dumps(lock_value).encode(), lock_bytes),
    ):
        path.write_bytes(changed)
        for key, (identity, plan) in capture().items():
            base_identity, base_plan = baseline[key]
            assert plan["bundleSha256"] != base_plan["bundleSha256"]
            assert plan["planSha256"] != base_plan["planSha256"]
            assert identity == base_identity
            paired = generate_original_foley(plan, tmp_path / f"{key}-{mutation}.wav")
            assert paired["sha256"] == baseline_foley[key], (key, mutation)
        path.write_bytes(original)


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


def test_voice_pack_manifest_keeps_the_governed_transcript_contract(root: Path) -> None:
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
    source = (
        root / "src/baloot_platform/video/remotion/src/professional-prototype.tsx"
    ).read_text(encoding="utf-8")
    assert (
        'staticFile("audio/voice_packs/ar_sa_baloot_v2_smooth/explanation.wav")'
        not in source
    )
    profile = yaml.safe_load(
        (root / "configs/video/premium.yaml").read_text(encoding="utf-8")
    )
    audio = profile["audio"]
    assert audio["one_piece_explanation"] is False
    assert audio["voice_callouts"] is False
    assert audio["spoken_voice_pack"] is False
    assert audio["voice_provider"] == audio["voice_pack_directory"] == ""
    assert audio["explanation_voice_localization_key"] == audio["explanation_voice_transcript"] == ""


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
    # Public session code box is red independent of the research notice.
    assert '<div className="idnum game-number is-notice">' in source
