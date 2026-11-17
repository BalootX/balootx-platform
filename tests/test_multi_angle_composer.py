from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

import baloot_platform.video.multi_angle_composer as composer_module
from baloot_platform.video.multi_angle_composer import MultiAngleComposerV1
from baloot_platform.video.render_bundle import build_export_timeline, build_render_bundle
from baloot_platform.video.render_plan import MultiAngleRenderPlanV1
from baloot_platform.video.scene import reduce_scene_document
from baloot_platform.config import build_bundle
from baloot_platform.plugins.registry import PluginPolicy, PluginRegistry


def _bundle(root: Path, observer: str, serial: str, *, profile=None):
    profile = profile or yaml.safe_load(
        (root / "configs/video/premium.yaml").read_text(encoding="utf-8")
    )
    hand = [
        {"id": 0, "rank": "7", "suit": "spades"},
        {"id": 9, "rank": "8", "suit": "hearts"},
    ]
    view = {
        "observer": observer,
        "currentPlayer": "south",
        "dealer": "west",
        "buyer": "north",
        "contract": "sun",
        "trumpSuit": None,
        "matchScore": [24, 19],
        "currentTrick": {"leader": "north", "plays": []},
        "hands": {
            seat: hand if seat == observer else {"count": 3}
            for seat in ("north", "east", "south", "west")
        },
    }
    common = {
        "observer_view": view,
        "public_events": (),
        "effective_config_sha256": "a" * 64,
        "locale": "ar-SA",
        "direction": "rtl",
        "acting_seat": "south",
        "prompt_contract": (
            "choose_acting_action" if observer == "south" else "read_partner_signal"
        ),
        "rendition_type": (
            "actor_challenge" if observer == "south" else "partner_inference"
        ),
    }
    question = reduce_scene_document(
        stage="question",
        prompt_key=(
            "video.decision.choose_best_move"
            if observer == "south"
            else "video.observer.read_partner_signal"
        ),
        legal_choices=(
            ({"kind": "play_card", "card": hand[0]},)
            if observer == "south"
            else ()
        ),
        **common,
    )
    reveal = reduce_scene_document(
        stage="reveal",
        prompt_key="video.decision.best_move",
        legal_choices=(
            ({"kind": "play_card", "card": hand[0]},)
            if observer == "south"
            else ()
        ),
        analysis_facts={
            **(
                {"recommendedAction": {"kind": "play_card", "card": hand[0]}}
                if observer == "south"
                else {}
            ),
            "reasonKey": "video.analysis.expert_reason",
        },
        **common,
    )
    scenes = (question, reveal)
    return build_render_bundle(
        scenes=scenes,
        timeline=build_export_timeline(
            scenes=scenes, profile=profile, export_preset="decision_clip"
        ),
        profile=profile,
        export_preset="decision_clip",
        localizations={
            "video.analysis.expert_reason": "يحفظ ورقة الدخول للفة الأخيرة.",
        },
        identity={
            "gameSerial": serial,
            "sceneId": question.scene_id,
            "enrollmentUri": f"https://balootx.local/c/{serial}",
            "players": {
                "north": "Player 01",
                "west": "Player 04",
                "east": "Player 02",
                "south": "Player 03",
            },
        },
        repository_root=root,
    )


def _probe(frames: int, fps: int) -> dict[str, object]:
    return {
        "durationSeconds": frames / fps,
        "video": {
            "codec": "h264",
            "width": 1080,
            "height": 1920,
            "frameRate": f"{fps}/1",
            "pixelFormat": "yuv420p",
            "colorSpace": "bt709",
            "colorPrimaries": "bt709",
            "colorTransfer": "bt709",
            "decodedFrames": frames,
        },
        "audio": {"codec": "aac", "sampleRate": 48000, "channels": 2},
    }


def _source_manifest(directory: Path, bundle, basename: str) -> Path:
    video = directory / f"{basename}.mp4"
    video.write_bytes(f"observer-safe-{basename}".encode("ascii"))
    value = {
        "schemaVersion": 1,
        "bundleSha256": bundle.bundle_sha256,
        "observerSafe": True,
        "files": {
            "video": {
                "name": video.name,
                "bytes": video.stat().st_size,
                "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            }
        },
    }
    value["manifestSha256"] = composer_module._canonical_manifest_sha(value)
    path = directory / f"{basename}-manifest.json"
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


@pytest.mark.parametrize("fps", [30, 60])
def test_multi_angle_composer_preserves_isolation_and_validates_output(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fps: int
) -> None:
    profile = yaml.safe_load(
        (root / "configs/video/premium.yaml").read_text(encoding="utf-8")
    )
    profile["exports"]["decision_clip"]["fps"] = fps
    south = _bundle(root, "south", "SOUTH1", profile=profile)
    north = _bundle(root, "north", "NORTH1", profile=profile)
    bundles = {item.bundle_sha256: item for item in (south, north)}
    plan = MultiAngleRenderPlanV1.create((
        (south, "actor_challenge", fps * 10, fps * 18, "cut", 0),
        (north, "partner_inference", fps * 12, fps * 20, "fade", 12),
    ))
    manifests = {
        south.bundle_sha256: _source_manifest(tmp_path, south, "south"),
        north.bundle_sha256: _source_manifest(tmp_path, north, "north"),
    }
    calls: list[dict[str, str]] = []
    staged_values: list[dict[str, object]] = []

    def fake_probe(path: Path, *, ffprobe: Path):
        del ffprobe
        return _probe(
            plan.total_frames if path.name == "angle.mp4"
            else south.value["timeline"]["totalFrames"],
            fps,
        )

    def fake_render(*, script: Path, environment, cancel_event):
        del script, cancel_event
        calls.append(dict(environment))
        work = Path(environment["BALOOTX_RENDER_OUTPUT"])
        staged_values.append(json.loads(
            Path(environment["BALOOTX_MULTI_ANGLE_INPUT"]).read_text(encoding="utf-8")
        ))
        (work / "render.mp4").write_bytes(b"composed-observer-safe-media")
        for index in range(2):
            Image.new("RGB", (1080, 1920), (4 + index, 20, 17)).save(
                work / f"checkpoint-{index:02d}.png"
            )

    monkeypatch.setattr(composer_module, "_probe_media", fake_probe)
    composer = MultiAngleComposerV1(profile, repository_root=root)
    monkeypatch.setattr(composer, "_run_renderer", fake_render)
    artifacts = composer.execute(
        plan,
        bundles=bundles,
        source_manifests=manifests,
        output_dir=tmp_path,
        basename="angle",
    )

    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    record = json.loads(artifacts.episode_json.read_text(encoding="utf-8"))
    assert manifest["observerSafe"] is True
    assert manifest["observerIsolation"] == "separately_rendered_media_only"
    assert manifest["planSha256"] == plan.sha256
    assert manifest["mediaProbe"]["video"]["decodedFrames"] == plan.total_frames
    assert manifest["mediaProbe"]["video"]["frameRate"] == f"{fps}/1"
    assert manifest["mediaProbe"]["durationSeconds"] == plan.total_frames / fps
    assert {item["observerSeat"] for item in manifest["sourceIdentities"]} == {"south", "north"}
    assert len(manifest["sourceMedia"]) == 2
    assert record["observerIsolation"] == "separately_rendered_media_only"
    assert "scenes" not in record and "hands" not in record
    assert "يحفظ ورقة الدخول" in artifacts.captions.read_text(encoding="utf-8")
    assert artifacts.video.is_file() and artifacts.poster.is_file()
    assert artifacts.contact_sheet.is_file() and len(calls) == 1
    assert [
        item["perspectiveName"]
        for item in staged_values[0]["plan"]["segments"]  # type: ignore[index]
    ] == ["Player 03", "Player 01"]
    assert not Path(calls[0]["BALOOTX_MULTI_ANGLE_INPUT"]).exists()

    reused = composer.execute(
        plan,
        bundles=bundles,
        source_manifests=manifests,
        output_dir=tmp_path,
        basename="angle",
    )
    assert reused.manifest == artifacts.manifest
    assert len(calls) == 1


def test_multi_angle_composer_rejects_tampered_source(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    south = _bundle(root, "south", "SOUTH2")
    north = _bundle(root, "north", "NORTH2")
    bundles = {item.bundle_sha256: item for item in (south, north)}
    plan = MultiAngleRenderPlanV1.create((
        (south, "actor_challenge", 0, 300, "cut", 0),
        (north, "partner_inference", 0, 300, "cut", 0),
    ))
    south_manifest = _source_manifest(tmp_path, south, "south-tamper")
    north_manifest = _source_manifest(tmp_path, north, "north-tamper")
    (tmp_path / "north-tamper.mp4").write_bytes(b"tampered")
    monkeypatch.setattr(
        composer_module, "_probe_media",
        lambda path, ffprobe: _probe(
            south.value["timeline"]["totalFrames"], south.value["render"]["fps"]
        ),
    )
    composer = MultiAngleComposerV1.from_yaml(root / "configs/video/premium.yaml")

    with pytest.raises(ValueError, match="hash or size"):
        composer.execute(
            plan,
            bundles=bundles,
            source_manifests={
                south.bundle_sha256: south_manifest,
                north.bundle_sha256: north_manifest,
            },
            output_dir=tmp_path,
        )


def test_multi_angle_composer_rejects_ungoverned_export_rate(root: Path) -> None:
    south = _bundle(root, "south", "SOUTH-RATE")
    north = _bundle(root, "north", "NORTH-RATE")
    plan = MultiAngleRenderPlanV1.create((
        (south, "actor_challenge", 0, 300, "cut", 0),
        (north, "partner_inference", 0, 300, "cut", 0),
    ))
    composer = MultiAngleComposerV1.from_yaml(root / "configs/video/premium.yaml")
    composer.profile["exports"]["decision_clip"]["fps"] = 60

    with pytest.raises(ValueError, match="governed video profile"):
        composer.execute(
            plan, bundles={item.bundle_sha256: item for item in (south, north)},
            source_manifests={},
        )


def test_multi_angle_plan_rejects_mixed_export_rates(root: Path) -> None:
    south = _bundle(root, "south", "SOUTH-MIXED")
    profile = yaml.safe_load(
        (root / "configs/video/premium.yaml").read_text(encoding="utf-8")
    )
    profile["exports"]["decision_clip"]["fps"] = 60
    north = _bundle(root, "north", "NORTH-MIXED", profile=profile)

    with pytest.raises(ValueError, match="same media profile"):
        MultiAngleRenderPlanV1.create((
            (south, "actor_challenge", 0, 300, "cut", 0),
            (north, "partner_inference", 0, 300, "cut", 0),
        ))


def test_multi_angle_transition_contract_is_fail_closed(root: Path) -> None:
    south = _bundle(root, "south", "SOUTH3")
    north = _bundle(root, "north", "NORTH3")
    with pytest.raises(ValueError, match="cut transitions"):
        MultiAngleRenderPlanV1.create((
            (south, "actor_challenge", 0, 300, "cut", 0),
            (north, "partner_inference", 0, 300, "cut", 4),
        ))
    with pytest.raises(ValueError, match="positive frame count"):
        MultiAngleRenderPlanV1.create((
            (south, "actor_challenge", 0, 300, "cut", 0),
            (north, "partner_inference", 0, 300, "fade", 0),
        ))
    with pytest.raises(ValueError, match="shorter than adjacent"):
        MultiAngleRenderPlanV1.create((
            (south, "actor_challenge", 0, 300, "cut", 0),
            (north, "partner_inference", 0, 300, "fade", 300),
        ))


def test_multi_angle_composer_has_a_separate_governed_plugin_boundary(root: Path) -> None:
    bundle = build_bundle(root)
    registry = PluginRegistry(
        policy=PluginPolicy.from_profile(bundle.component("runtime")["plugin_host"])
    )
    composer = MultiAngleComposerV1.from_yaml(root / "configs/video/premium.yaml")
    registry.register(composer)
    assert registry.get("video_compositor", "multi_angle_composer_v1") is composer
    with pytest.raises(KeyError):
        registry.get("video_renderer", "multi_angle_composer_v1")
