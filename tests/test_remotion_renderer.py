from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest
import yaml
from PIL import Image

import baloot_platform.video.remotion_renderer as remotion_renderer_module
from baloot_platform.audit.models import content_hash
from baloot_platform.config import build_bundle
from baloot_platform.core.cards import FULL_DECK
from baloot_platform.core.enums import Contract, Seat
from baloot_platform.core.episode_io import load_episode_spec
from baloot_platform.core.projects import project_candidates
from baloot_platform.plugins.registry import PluginPolicy, PluginRegistry
from baloot_platform.video.remotion_renderer import RemotionProfessionalRenderer
from baloot_platform.video.render_bundle import RenderBundleV1
from baloot_platform.video.scene import SceneDocumentV1, _walk_leaves


def _renderer(root: Path) -> RemotionProfessionalRenderer:
    profile = yaml.safe_load(
        (root / "configs/video/premium.yaml").read_text(encoding="utf-8")
    )
    return RemotionProfessionalRenderer(profile, repository_root=root)


def _episode(root: Path):
    bundle = build_bundle(root)
    return load_episode_spec(
        root / "examples/v7-outcome-flip-source.json", bundle.rules
    )


def _publication_policy(*, blockers: list[str]) -> dict[str, object]:
    decision_core = {"schemaVersion": 1, "gatePassed": not blockers}
    video_core = {"schemaVersion": 1, "gatePassed": not blockers}
    policy_core = {
        "schemaVersion": 1,
        "decisionSelection": {
            **decision_core,
            "sha256": content_hash(decision_core),
        },
        "videoProduction": {
            **video_core,
            "sha256": content_hash(video_core),
        },
        "blockers": blockers,
    }
    return {**policy_core, "sha256": content_hash(policy_core)}


def _governed_bundle(
    bundle: RenderBundleV1,
    *,
    audit_only: bool,
    show_qr: bool = True,
) -> RenderBundleV1:
    raw = bundle.to_dict()
    raw["presentation"]["presentation"]["show_enrollment_qr"] = show_qr
    blockers = ["semantic_quarantine"] if audit_only else []
    raw["source"] = {
        "sourceSchemaVersion": 2,
        "kind": "research_result",
        "code": "LAF-REMOTION01",
        "decisionOrdinal": 1,
        "decisionId": "D1",
        "resultHash": "1" * 64,
        "traceHash": "2" * 64,
        "analysisHash": "3" * 64,
        "auditHash": "4" * 64,
        "analysisProfileHash": "5" * 64,
        "analyzerId": "adaptive_finalist_trace_analyzer_v1",
        "analyzerVersion": "1.3.0",
        "rulesHash": "6" * 64,
        "exactStateId": "D1",
        "qrActionable": not audit_only,
        "qualificationStatus": "rejected" if audit_only else "qualified",
        "renderPurpose": "research_audit" if audit_only else "content_candidate",
        "qualificationFailures": blockers,
        "publicationBlockers": blockers,
        "observerSeat": raw["observerSeat"],
        "auditOnly": audit_only,
        "publicationEligible": not audit_only,
        "publicationPolicy": _publication_policy(blockers=blockers),
    }
    core = {key: value for key, value in raw.items() if key != "bundleSha256"}
    raw["bundleSha256"] = content_hash(core)
    governed = RenderBundleV1(raw)
    assert governed.verify()
    return governed


def test_remotion_plugin_builds_only_verified_observer_safe_bundles(root: Path) -> None:
    renderer = _renderer(root)
    episode = _episode(root)
    flagship = renderer.build_render_request(episode, export_preset="hooked_flagship")
    decision = renderer.build_render_request(episode, export_preset="decision_clip")
    encoded = json.dumps(flagship.to_dict(), ensure_ascii=False, sort_keys=True)

    assert flagship.verify() and decision.verify()
    assert flagship.value["timeline"]["totalFrames"] == 4320
    assert decision.value["timeline"]["totalFrames"] == 600
    assert decision.value["render"]["fps"] == 30
    players = flagship.value["identity"]["players"]
    pool = set(renderer.profile["identity"]["player_name_pool"])
    assert set(players) == {"north", "west", "east", "south"}
    assert len(set(players.values())) == 4
    assert set(players.values()) <= pool
    assert flagship.value["localizations"]["video.project.siri"] == "سرا"
    for forbidden in (
        "deckOrder",
        "initialHands",
        "legalCards",
        "futureCards",
        "trustedState",
        '"seed"',
    ):
        assert forbidden not in encoded


@pytest.mark.parametrize("leak", ["future_play", "terminal_score"])
def test_remotion_rejects_rehashed_context_leaks_before_encoding(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, leak: str
) -> None:
    renderer = _renderer(root)
    bundle = renderer.build_render_request(_episode(root), export_preset="hooked_flagship")
    assert bundle.verify()
    raw = bundle.to_dict()
    context = next(scene for scene in raw["scenes"] if scene["stage"] == "context")
    lifecycle = next(layer["content"] for layer in context["layers"]
                     if layer["id"] == "public-lifecycle")
    if leak == "future_play":
        played = {play["card"]["id"] for play in lifecycle["plays"]}
        card = next(card for card in FULL_DECK if card.id not in played)
        lifecycle["plays"].append({"seat": "north", "card": card.to_dict()})
    else:
        lifecycle["roundScore"] = {"awardedPoints": [0, 38]}
    old_id = context["sceneId"]
    core = {key: value for key, value in context.items()
            if key not in {"sceneId", "fieldDisclosures"}}
    context["sceneId"] = content_hash(core)
    context["fieldDisclosures"] = {
        path: core["layers"][int(path.split("/")[2])]["disclosure"]
        if path.startswith("/layers/") else "public" for path in _walk_leaves(core)
    }
    assert not SceneDocumentV1(context).verify()
    raw = json.loads(json.dumps(raw).replace(old_id, context["sceneId"]))
    raw["timeline"]["timelineSha256"] = content_hash(
        {key: value for key, value in raw["timeline"].items() if key != "timelineSha256"}
    )
    raw["bundleSha256"] = content_hash(
        {key: value for key, value in raw.items() if key != "bundleSha256"}
    )
    monkeypatch.setattr(renderer, "_run_renderer", lambda **_kwargs: pytest.fail(
        "unsafe context must be rejected before starting an encoder"))
    output = tmp_path / "must-not-be-created"
    with pytest.raises(ValueError, match="integrity verification failed"):
        renderer.execute(RenderBundleV1(raw), output_dir=output)
    assert not output.exists()


@pytest.mark.parametrize("viewer", [Seat.SOUTH, Seat.WEST])
def test_playback_evidence_derives_all_complete_trace_boundaries(root: Path, viewer: Seat) -> None:
    renderer = _renderer(root)
    bundle = renderer.build_render_request(
        replace(_episode(root), viewer=viewer),
        export_preset="hooked_flagship",
    )
    evidence = renderer._playback_state_evidence(
        bundle,
        {
            "schemaVersion": 1,
            "items": [
                {
                    "completedPlayCount": count,
                    "frame": count,
                    "source": f"playback-{count:02d}.png",
                }
                for count in range(33)
            ],
        },
    )

    assert len(evidence) == 33
    assert evidence[0]["handCounts"] == {
        "north": 8,
        "west": 8,
        "south": 8,
        "east": 8,
    }
    assert evidence[1]["playedCard"] is not None
    assert evidence[3]["currentTrickCardIds"]
    assert evidence[4]["currentTrickCardIds"] == []
    assert evidence[4]["completedTrickIndex"] == 0
    assert evidence[-1]["handCounts"] == {
        "north": 0,
        "west": 0,
        "south": 0,
        "east": 0,
    }
    assert evidence[-1]["observerHandCardIds"] == []
    assert len(evidence[-1]["logicalTableCardIds"]) == 4
    assert all("tableCardIds" not in row for row in evidence)
    assert all(row["visibleCardStateVerified"] is False for row in evidence)
    assert all(row["evidenceScope"] == "trace_state_and_encoded_frame_parity" for row in evidence)
    outcome = next(row for row in bundle.value["timeline"]["segments"] if row["stage"] == "outcome")
    outcome_checkpoint = {
        "phase": "outcome", "completedPlayCount": 32,
        "frame": outcome["startFrame"] + outcome["durationFrames"] // 2,
        "source": "playback-outcome.png",
    }
    with pytest.raises(ValueError, match="DOM checkpoint is missing"):
        renderer._playback_state_evidence(
            bundle,
            {"items": [
                {"completedPlayCount": count, "frame": count, "source": f"playback-{count:02d}.png"}
                for count in range(33)
            ] + [outcome_checkpoint]},
            require_rendered_dom=True,
        )
    paused_value = bundle.to_dict()
    context = next(scene for scene in paused_value["scenes"] if scene["stage"] == "continuation")
    lifecycle = next(layer for layer in context["layers"] if layer["id"] == "public-lifecycle")
    lifecycle["content"]["pauseBeforePlayIndex"] = 28
    paused = renderer._playback_state_evidence(
        RenderBundleV1(paused_value),
        {"items": [
            {"completedPlayCount": count, "frame": count, "source": f"playback-{count:02d}.png"}
            for count in range(33)
        ]},
    )
    assert len(paused[28]["logicalTableCardIds"]) == 4
    assert paused[28]["currentTrickCardIds"] == []
    checkpoints = {"items": [
        {"completedPlayCount": row["completedPlayCount"], "frame": row["frame"], "source": row["source"],
         "renderedDom": {"frame": row["frame"], "completedPlayCount": row["completedPlayCount"],
                         "hands": [{"seat": seat, "count": count,
                                    "faceCardIds": row["observerHandCardIds"] if seat == bundle.value["observerSeat"] else []}
                                   for seat, count in row["handCounts"].items()],
                         "cards": [{"cardId": card, "cardCode": FULL_DECK[card].code,
                                    "imageLoaded": True,
                                    "opacity": 1 if card in row["currentTrickCardIds"] else 0,
                                    "visible": card in row["currentTrickCardIds"]}
                                   for card in row["logicalTableCardIds"]]}}
        for row in evidence
    ]}
    terminal_score = list(lifecycle["content"]["matchScore"])
    if viewer in (Seat.WEST, Seat.EAST):
        terminal_score.reverse()
    checkpoints["items"].append({
        **outcome_checkpoint,
        "renderedDom": {
            "frame": outcome_checkpoint["frame"], "completedPlayCount": 32,
            "displayedScore": terminal_score, "cards": [],
            "hands": [{"seat": seat, "count": 0, "faceCardIds": []}
                      for seat in ("north", "west", "south", "east")],
        },
    })
    observed = renderer._playback_state_evidence(
        bundle, checkpoints, require_rendered_dom=True,
    )
    assert len(observed) == 34
    assert len(observed[-2]["logicalTableCardIds"]) == 4
    assert observed[-1]["checkpointPhase"] == "outcome"
    assert observed[-1]["logicalTableCardIds"] == []
    assert observed[-1]["renderedDom"]["displayedScore"] == terminal_score
    with pytest.raises(ValueError, match="full playback evidence must cover outcome"):
        renderer._playback_state_evidence(bundle, {"items": checkpoints["items"][:-1]}, require_rendered_dom=True)
    checkpoints["items"][-1]["frame"] += 1
    with pytest.raises(ValueError, match="outcome at its midpoint"):
        renderer._playback_state_evidence(bundle, checkpoints, require_rendered_dom=True)
    checkpoints["items"][-1]["frame"] -= 1
    checkpoints["items"][-1]["source"] = checkpoints["items"][-2]["source"]
    with pytest.raises(ValueError, match="unique source"):
        renderer._playback_state_evidence(bundle, checkpoints, require_rendered_dom=True)
    checkpoints["items"][-1]["source"] = outcome_checkpoint["source"]
    checkpoints["items"][-1]["renderedDom"]["displayedScore"] = [-1, -1]
    with pytest.raises(ValueError, match="outcome HUD"):
        renderer._playback_state_evidence(bundle, checkpoints, require_rendered_dom=True)
    checkpoints["items"][-1]["renderedDom"]["displayedScore"] = terminal_score
    checkpoints["items"][-1]["renderedDom"]["cards"] = checkpoints["items"][-2]["renderedDom"]["cards"]
    with pytest.raises(ValueError, match="cards disagree with trace"):
        renderer._playback_state_evidence(bundle, checkpoints, require_rendered_dom=True)
    checkpoints["items"][-1]["renderedDom"]["cards"] = []
    checkpoints["items"][-1]["renderedDom"]["hands"][0]["count"] = 1
    with pytest.raises(ValueError, match="observer safety"):
        renderer._playback_state_evidence(bundle, checkpoints, require_rendered_dom=True)
    checkpoints["items"][-1]["renderedDom"]["hands"][0]["count"] = 0
    paused_observed = renderer._playback_state_evidence(
        RenderBundleV1(paused_value), checkpoints, require_rendered_dom=True,
    )
    assert len(paused_observed[28]["logicalTableCardIds"]) == 4
    assert paused_observed[28]["observedVisibleTableCardIds"] == []
    checkpoints["items"][28]["renderedDom"]["cards"][0]["opacity"] = 1
    checkpoints["items"][28]["renderedDom"]["cards"][0]["visible"] = True
    with pytest.raises(ValueError, match="visible cards disagree"):
        renderer._playback_state_evidence(RenderBundleV1(paused_value), checkpoints)
    checkpoints["items"][28]["renderedDom"]["cards"][0]["opacity"] = 0
    checkpoints["items"][28]["renderedDom"]["cards"][0]["visible"] = False
    assert observed[-1]["observedVisibleTableCardIds"] == []
    assert observed[1]["observedVisibleTableCardIds"] == observed[1]["currentTrickCardIds"]
    checkpoints["items"][1]["renderedDom"]["cards"][0]["visible"] = False
    with pytest.raises(ValueError, match="visible cards disagree"):
        renderer._playback_state_evidence(bundle, checkpoints)
    checkpoints["items"][1]["renderedDom"]["cards"][0]["visible"] = True
    checkpoints["items"][1]["renderedDom"]["cards"][0]["imageLoaded"] = False
    with pytest.raises(ValueError, match="visibility evidence"):
        renderer._playback_state_evidence(bundle, checkpoints)
    checkpoints["items"][1]["renderedDom"]["cards"][0]["imageLoaded"] = True
    checkpoints["items"][1]["renderedDom"]["cards"][0]["cardCode"] = "WRONG"
    with pytest.raises(ValueError, match="visibility evidence"):
        renderer._playback_state_evidence(bundle, checkpoints)
    checkpoints["items"][1]["renderedDom"]["cards"][0]["cardCode"] = FULL_DECK[
        checkpoints["items"][1]["renderedDom"]["cards"][0]["cardId"]
    ].code
    checkpoints["items"][-1]["renderedDom"]["hands"][0]["faceCardIds"] = [0]
    with pytest.raises(ValueError, match="observer safety"):
        renderer._playback_state_evidence(bundle, checkpoints)
    checkpoints["items"][-1]["renderedDom"]["hands"][0]["faceCardIds"] = []
    checkpoints["items"][-2]["renderedDom"]["cards"][0]["visible"] = True
    with pytest.raises(ValueError, match="visibility evidence"):
        renderer._playback_state_evidence(bundle, checkpoints)
    checkpoints["items"][-2]["renderedDom"]["cards"][0]["visible"] = False
    checkpoints["items"][-1]["renderedDom"]["frame"] = -1
    with pytest.raises(ValueError, match="checkpoint identity"):
        renderer._playback_state_evidence(bundle, checkpoints)


@pytest.mark.parametrize("widened", [False, True], ids=["approved", "widened"])
def test_layout_receipt_binds_raw_mp4_bundle_table_and_checkpoint_bytes(root, tmp_path, widened):
    def retain(name, contents):
        target = tmp_path / name
        target.write_bytes(contents)
        return target, remotion_renderer_module._sha256(target)

    approved = (root / "src/baloot_platform/video/remotion/scripts/approved-layout-adjacencies.json").read_bytes()
    if widened:
        altered = json.loads(approved)
        altered["rows"].append({
            "phase": "hook", "pair": ["promptPill:0", "broadcastHud:0"],
            "observedMaxWidth": 200, "observedMaxHeight": 5, "marginEachAxis": 2,
        })
        approved = json.dumps(altered).encode("utf-8")
    table, table_sha = retain("layout-table.json", approved)
    bundle, bundle_sha = retain("render-bundle.json", b'{"schemaVersion": 1}')
    video, video_sha = retain("render.mp4", b"rendered media")
    binding = {
        "mode": "qualification", "tableSha256": table_sha,
        "bundleSha256": "bundle-sha", "bundleFileSha256": bundle_sha,
    }
    rows = []
    for phase in ("hook", "question", "reveal"):
        png, png_sha = retain(f"{phase}.png", b"checkpoint")
        rows.append({"phase": phase, "source": png.name, "pngSha256": png_sha, "binding": binding,
            "validation": {"status": "passed", "tableSha256": table_sha}})
    evidence = {"schemaVersion": 1, "status": "passed", "binding": binding, "items": rows, "videoSha256": video_sha}
    receipt, _ = retain("layout-evidence.json", json.dumps(evidence).encode("utf-8"))

    def read():
        return remotion_renderer_module._qualified_layout_evidence(
            tmp_path, bundle_path=bundle, bundle_sha256="bundle-sha",
        )

    if widened:
        with pytest.raises(ValueError, match="layout table, bundle and rendered MP4"):
            read()
        return

    assert read() == evidence
    for file in (table, bundle, video, png):
        original = file.read_bytes()
        file.write_bytes(original + b"changed")
        with pytest.raises(ValueError, match="layout table, bundle and rendered MP4|checkpoint binding"):
            read()
        file.write_bytes(original)
    binding["tableSha256"] = "0" * 64
    receipt.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(ValueError, match="layout table, bundle and rendered MP4"):
        read()
    binding["tableSha256"] = table_sha
    binding["mode"] = "measurement"
    receipt.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(ValueError, match="permanently NEVER_POST"):
        read()


@pytest.mark.parametrize("pause", [24, 25, 26])
def test_decision_clip_checks_selected_states_without_claiming_full_replay(root: Path, pause) -> None:
    script = (root / "src/baloot_platform/video/remotion/scripts/render-decision-clip.mjs").read_text(encoding="utf-8")
    # Remotion renders resolved composition props, not just inputProps.
    assert "props: {...composition.props, ...stillProps}" in script
    assert "{capturePlaybackDom: true}" in script
    renderer = _renderer(root)
    raw = renderer.build_render_request(_episode(root), export_preset="decision_clip").to_dict()
    continuation_scene = next(scene for scene in raw["scenes"] if scene["stage"] == "continuation")
    next(layer for layer in continuation_scene["layers"] if layer["id"] == "public-lifecycle")["content"]["pauseBeforePlayIndex"] = pause
    candidate = RenderBundleV1(raw)
    segments = {row["stage"]: row for row in raw["timeline"]["segments"]}
    continuation = segments["continuation"]
    frames = [segments["context"]["startFrame"] + segments["context"]["durationFrames"] - 1,
        *[segments[stage]["startFrame"] + segments[stage]["durationFrames"] // 2 for stage in ("question", "reveal")],
        continuation["startFrame"] + (continuation["durationFrames"] + 31 - pause) // (32 - pause) - 1,
        continuation["startFrame"] + continuation["durationFrames"] - 1,
        segments["outcome"]["startFrame"] + segments["outcome"]["durationFrames"] // 2]
    checkpoints = {"items": [dict(phase=phase, completedPlayCount=count, frame=frame, source=f"{phase}.png")
        for phase, count, frame in zip(("context", "question", "reveal", "continuation", "continuation_end", "outcome"),
            (pause, pause, pause, pause + 1, 32, 32), frames, strict=True)]}
    rows = renderer._playback_state_evidence(candidate, checkpoints, clip_evidence=True)
    assert len(rows) == 6 and all(row["evidenceScope"] == "decision_clip_state_and_encoded_frame_parity" for row in rows)
    lifecycle = next(layer for layer in continuation_scene["layers"] if layer["id"] == "public-lifecycle")["content"]
    terminal_score = list(lifecycle["matchScore"])
    if raw["observerSeat"] in ("west", "east"):
        terminal_score.reverse()
    for item, row in zip(checkpoints["items"], rows, strict=True):
        item["renderedDom"] = dict(frame=row["frame"], completedPlayCount=row["completedPlayCount"],
            displayedScore=terminal_score if item["phase"] == "outcome" else [0, 0],
            hands=[dict(seat=seat, count=count, faceCardIds=row["observerHandCardIds"] if seat == raw["observerSeat"] else [])
                for seat, count in row["handCounts"].items()],
            cards=[dict(cardId=card, cardCode=FULL_DECK[card].code, imageLoaded=True,
                opacity=1 if card in row["currentTrickCardIds"] else 0, visible=card in row["currentTrickCardIds"])
                for card in row["logicalTableCardIds"]])
    observed = renderer._playback_state_evidence(candidate, checkpoints, clip_evidence=True, require_rendered_dom=True)
    assert observed[-1]["observerHandCardIds"] == []
    checkpoints["items"][-1]["renderedDom"]["displayedScore"] = [-1, -1]
    with pytest.raises(ValueError, match="outcome HUD"):
        renderer._playback_state_evidence(candidate, checkpoints, clip_evidence=True, require_rendered_dom=True)
    checkpoints["items"][-1]["renderedDom"]["displayedScore"] = terminal_score
    assert all(row["handCounts"] == dict(north=0, west=0, south=0, east=0) for row in observed[-1:])
    checkpoints["items"][0]["renderedDom"]["hands"][0]["faceCardIds"] = [31]
    with pytest.raises(ValueError, match="observer safety"):
        renderer._playback_state_evidence(candidate, checkpoints, clip_evidence=True, require_rendered_dom=True)
    checkpoints["items"][0]["frame"] += 1
    with pytest.raises(ValueError, match="cover context"):
        renderer._playback_state_evidence(candidate, checkpoints, clip_evidence=True)


def test_video_profile_localizes_every_runtime_project_kind(root: Path) -> None:
    profile = yaml.safe_load(
        (root / "configs/video/premium.yaml").read_text(encoding="utf-8")
    )
    rules = build_bundle(root).rules
    runtime_kinds = {
        project.kind
        for contract in (Contract.SUN, Contract.HOKUM)
        for project in project_candidates(Seat.NORTH, FULL_DECK, contract, rules)
    }

    assert runtime_kinds == {
        "siri",
        "fifty",
        "hundred_sequence",
        "hundred_four_kind",
        "four_hundred",
    }
    assert {
        kind
        for kind in runtime_kinds
        if f"video.project.{kind}" not in profile["localizations"]
    } == set()


def test_remotion_plugin_is_admitted_by_the_governed_registry(root: Path) -> None:
    bundle = build_bundle(root)
    registry = PluginRegistry(
        policy=PluginPolicy.from_profile(bundle.component("runtime")["plugin_host"])
    )
    renderer = _renderer(root)
    registry.register(renderer)

    assert registry.get("video_renderer", "premium_video") is renderer
    descriptor = registry.list("video_renderer")[0]
    assert descriptor["isolation"] == "process"
    assert descriptor["deterministic"] is True
    assert "render_bundle_v1" in descriptor["capabilities"]


def test_remotion_renderer_uses_exact_governed_runtime_timeout(root: Path) -> None:
    profile = yaml.safe_load(
        (root / "configs/video/premium.yaml").read_text(encoding="utf-8")
    )
    renderer = RemotionProfessionalRenderer(
        profile,
        repository_root=root,
        timeout_seconds=3600,
    )

    assert renderer.descriptor.timeout_seconds == 3600
    assert RemotionProfessionalRenderer.descriptor.timeout_seconds == 900
    for invalid in (True, 0, -1, 1.5, "3600"):
        with pytest.raises(ValueError, match="positive integer"):
            RemotionProfessionalRenderer(
                profile,
                repository_root=root,
                timeout_seconds=invalid,
            )


def test_remotion_plugin_rejects_unknown_configuration(root: Path) -> None:
    profile = yaml.safe_load(
        (root / "configs/video/premium.yaml").read_text(encoding="utf-8")
    )
    invalid = copy.deepcopy(profile)
    invalid["exports"]["decision_clip"]["unknown_timing"] = 1

    with pytest.raises(ValueError, match="unknown_timing"):
        RemotionProfessionalRenderer(invalid, repository_root=root)


def test_remotion_plugin_packages_complete_artifacts_from_bundle(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    renderer = _renderer(root)
    bundle = renderer.build_render_request(
        _episode(root), export_preset="decision_clip"
    )

    def fake_render(*, script, environment, cancel_event):
        del cancel_event
        if script.name == "verify-qr.mjs":
            policy = environment["BALOOTX_QR_POLICY"]
            Path(environment["BALOOTX_QR_RESULT"]).write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "policy": policy,
                        "reason": environment["BALOOTX_QR_REASON"],
                        "passed": True,
                        "decodedValue": (
                            environment["BALOOTX_QR_EXPECTED"]
                            if policy == "required"
                            else None
                        ),
                    }
                ),
                encoding="utf-8",
            )
            return
        output = Path(environment["BALOOTX_RENDER_OUTPUT"])
        bundle_path = Path(environment["BALOOTX_RENDER_BUNDLE"])
        assert bundle_path.is_file() and bundle_path.parent != output
        output.mkdir()
        (output / "render.mp4").write_bytes(b"deterministic-test-video")
        for index, name in enumerate(
            (
                "01-hook",
                "02-rewind",
                "03-context",
                "04-decision",
                "05-reveal",
                "06-continuation",
            )
        ):
            Image.new("RGB", (1080, 1920), (4 + index, 20, 17)).save(
                output / f"{name}.png"
            )
        (output / "render.ar.vtt").write_text("WEBVTT\n\n", encoding="utf-8")

    monkeypatch.setattr(renderer, "_run_renderer", fake_render)
    monkeypatch.setattr(renderer, "_ffprobe", lambda: Path("ffprobe"))
    audio_evidence = {
        "status": "passed",
        "finisher": "balootx_original_foley_v1",
        "music": False,
        "videoStreamPreserved": True,
    }
    monkeypatch.setattr(
        remotion_renderer_module,
        "finish_remotion_audio",
        lambda *_args, **_kwargs: audio_evidence,
    )
    render_profile = bundle.value["render"]
    monkeypatch.setattr(
        remotion_renderer_module,
        "probe_media",
        lambda *_args, **_kwargs: {
            "durationSeconds": bundle.value["timeline"]["totalFrames"]
            / render_profile["fps"],
            "video": {
                "codec": "h264",
                "width": render_profile["width"],
                "height": render_profile["height"],
                "frameRate": f"{render_profile['fps']}/1",
                "pixelFormat": "yuv420p",
                "colorSpace": "bt709",
                "colorPrimaries": "bt709",
                "colorTransfer": "bt709",
                "decodedFrames": bundle.value["timeline"]["totalFrames"],
            },
            "audio": {
                "codec": "aac",
                "sampleRate": render_profile["audio_rate"],
                "channels": 2,
            },
        },
    )
    artifacts = renderer.execute(
        bundle,
        export_preset="decision_clip",
        output_dir=tmp_path,
        basename="decision-test",
    )
    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    bundle_output = artifacts.episode_json.read_text(encoding="utf-8")

    assert manifest["observerSafe"] is True
    assert manifest["bundleSha256"] == bundle.bundle_sha256
    assert manifest["timelineSha256"] == bundle.value["timeline"]["timelineSha256"]
    assert manifest["audioFinishing"] == audio_evidence
    assert manifest["source"] is None
    assert manifest["sourceSha256"] is None
    assert "exactStateId" not in manifest
    assert manifest["decisionReference"] is None
    assert manifest["publicProjectionHash"] is None
    assert manifest["qrValidation"] == {
        "schemaVersion": 1,
        "policy": "forbidden",
        "reason": "profile_hides_qr",
        "passed": True,
        "decodedValue": None,
    }
    assert manifest["files"]["audioEvidence"]["sha256"]
    assert artifacts.video.read_bytes() == b"deterministic-test-video"
    assert artifacts.poster.is_file() and artifacts.contact_sheet.is_file()
    assert artifacts.captions.is_file() and artifacts.captions_vtt.is_file()
    assert artifacts.captions_vtt.read_text(encoding="utf-8") == "WEBVTT\n\n"
    assert not artifacts.transcript.read_text(encoding="utf-8").strip()
    assert artifacts.captions.read_text(encoding="utf-8").splitlines()[-1] == ""
    assert artifacts.timeline.is_file() and len(artifacts.semantic_files) == len(
        bundle.value["scenes"]
    )
    for forbidden in (
        "deckOrder",
        "initialHands",
        "legalCards",
        "futureCards",
        "trustedState",
        '"seed"',
    ):
        assert forbidden not in bundle_output

    audit_bundle = _governed_bundle(bundle, audit_only=True)
    audit_artifacts = renderer.execute(
        audit_bundle,
        export_preset="decision_clip",
        output_dir=tmp_path,
        basename="decision-audit-test",
    )
    audit_manifest = json.loads(audit_artifacts.manifest.read_text(encoding="utf-8"))
    assert audit_manifest["source"] == audit_bundle.value["source"]
    assert audit_manifest["sourceSha256"] == content_hash(audit_bundle.value["source"])
    assert "exactStateId" not in audit_manifest
    assert audit_manifest["qrValidation"] == {
        "schemaVersion": 1,
        "policy": "forbidden",
        "reason": "research_audit",
        "passed": True,
        "decodedValue": None,
    }

    profile_disabled = _governed_bundle(
        bundle,
        audit_only=False,
        show_qr=False,
    )
    assert renderer._qr_expectation(profile_disabled) == (
        "forbidden",
        "profile_disabled",
    )
    qr_bundle = bundle.to_dict()
    qr_bundle["presentation"]["majlis"]["id_box"] = "qr"
    qr_bundle.pop("bundleSha256")
    qr_bundle["bundleSha256"] = content_hash(qr_bundle)
    assert renderer._qr_expectation(RenderBundleV1(qr_bundle)) == ("required", "profile_enabled")


def test_remotion_subprocess_honors_cancellation(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    renderer = _renderer(root)
    monkeypatch.setattr(
        renderer,
        "_command",
        lambda script: [sys.executable, "-c", "import time; time.sleep(30)"],
    )
    cancel = Event()
    cancel.set()

    with pytest.raises(RuntimeError, match="cancelled"):
        renderer._run_renderer(
            script=Path("ignored"),
            environment={},
            cancel_event=cancel,
        )


def test_remotion_subprocess_honors_timeout(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ShortTimeoutRenderer(RemotionProfessionalRenderer):
        descriptor = replace(RemotionProfessionalRenderer.descriptor, timeout_seconds=1)

    profile = yaml.safe_load(
        (root / "configs/video/premium.yaml").read_text(encoding="utf-8")
    )
    renderer = ShortTimeoutRenderer(profile, repository_root=root)
    child_pid_path = tmp_path / "child.pid"
    parent = (
        "import pathlib, subprocess, sys, time; "
        "child=subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(30)']); "
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid)); "
        "time.sleep(30)"
    )
    monkeypatch.setattr(
        renderer, "_command", lambda script: [sys.executable, "-c", parent]
    )

    with pytest.raises(TimeoutError, match="governed timeout"):
        renderer._run_renderer(
            script=Path("ignored"),
            environment={},
            cancel_event=None,
        )

    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if not remotion_renderer_module._process_exists(child_pid):
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"owned renderer child process {child_pid} survived timeout")

@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_remotion_subprocess_completes_with_verbose_output(
    root: Path, monkeypatch: pytest.MonkeyPatch, stream: str
) -> None:
    renderer = _renderer(root)
    # Check pipe handling, not cold process-start timing on a CPU-capped host.
    monkeypatch.setattr(renderer, "descriptor", replace(renderer.descriptor, timeout_seconds=10))
    monkeypatch.setattr(
        renderer,
        "_command",
        lambda script: [
            sys.executable,
            "-c",
            f"import sys; sys.{stream}.write('render progress\\n' * 32768)",
        ],
    )

    renderer._run_renderer(script=Path("ignored"), environment={}, cancel_event=None)


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_remotion_subprocess_retains_bounded_failure_diagnostics(
    root: Path, monkeypatch: pytest.MonkeyPatch, stream: str
) -> None:
    renderer = _renderer(root)
    monkeypatch.setattr(renderer, "descriptor", replace(renderer.descriptor, timeout_seconds=10))
    message = "render failed: تفاصيل"
    monkeypatch.setattr(
        renderer,
        "_command",
        lambda script: [
            sys.executable,
            "-c",
            f"import sys; sys.{stream}.write('x' * 65536 + {message!r}); sys.exit(3)",
        ],
    )

    with pytest.raises(RuntimeError) as error:
        renderer._run_renderer(script=Path("ignored"), environment={}, cancel_event=None)

    assert len(str(error.value)) == 4000
    assert str(error.value).endswith(message)


def test_remotion_subprocess_prefers_stderr_diagnostics(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    renderer = _renderer(root)
    monkeypatch.setattr(
        renderer,
        "_command",
        lambda script: [
            sys.executable,
            "-c",
            "import sys; print('progress'); print('render failed', file=sys.stderr); sys.exit(3)",
        ],
    )

    with pytest.raises(RuntimeError, match="^render failed$"):
        renderer._run_renderer(script=Path("ignored"), environment={}, cancel_event=None)


@pytest.mark.skipif(os.name == "nt", reason="Windows terminate cannot be ignored")
@pytest.mark.parametrize("cancelled", [False, True], ids=["timeout", "cancellation"])
def test_remotion_subprocess_kills_child_that_ignores_termination(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancelled: bool
) -> None:
    renderer = _renderer(root)
    monkeypatch.setattr(renderer, "descriptor", replace(renderer.descriptor, timeout_seconds=1))
    ready = tmp_path / "ready"
    monkeypatch.setattr(
        renderer,
        "_command",
        lambda script: [
            sys.executable,
            "-c",
            (
                "import signal, time; from pathlib import Path; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                f"Path({str(ready)!r}).touch(); time.sleep(30)"
            ),
        ],
    )
    real_popen = subprocess.Popen
    children = []

    def start_child(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        real_wait = child.wait
        # Exercise the real termination and kill path without a ten-second
        # shutdown grace period in each test case.
        monkeypatch.setattr(
            child,
            "wait",
            lambda timeout=None: real_wait(timeout=min(timeout, 0.1) if timeout is not None else None),
        )
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", start_child)
    cancel = Event() if cancelled else None
    if cancel is not None:
        monkeypatch.setattr(cancel, "is_set", ready.is_file)

    try:
        error = RuntimeError if cancelled else TimeoutError
        message = "cancelled" if cancelled else "governed timeout"
        with pytest.raises(error, match=message):
            renderer._run_renderer(script=Path("ignored"), environment={}, cancel_event=cancel)
        assert ready.is_file(), "child must install its signal handler before shutdown"
        assert children[0].poll() is not None
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
