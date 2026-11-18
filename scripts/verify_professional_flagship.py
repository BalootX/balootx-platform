#!/usr/bin/env python3
# ruff: noqa: E402
"""Verify the complete professional Remotion laffah and write deterministic evidence."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baloot_platform.video.render_bundle import RenderBundleV1
from baloot_platform.video.remotion_renderer import _timestamp
from baloot_platform.video.media_validation import run_media_command
from baloot_platform.filesystem import atomic_write_text

OUTPUT = ROOT / "outputs/professional-video-flagship"
VIDEO = OUTPUT / "balootx-complete-laffah-audit.mp4"
CAPTIONS = OUTPUT / "balootx-complete-laffah-audit.ar.vtt"
BUNDLE = ROOT / "outputs/professional-video-prototypes/render-bundle-hooked_flagship.json"
AUDIO_EVIDENCE = OUTPUT / "balootx-complete-laffah-audit.audio.json"
STILLS = (
    "01-hook.png",
    "01b-rewind.png",
    "02-opening-play.png",
    "02-project-mention-cleared.png",
    "02a-second-trick-start.png",
    "02b-project-reveal.png",
    "02b2-project-reveal-late.png",
    "02c-project-reveal-cleared.png",
    "03-mid-play.png",
    "04-critical-pause.png",
    "05-reveal.png",
    "06-continuation.png",
    "07-outcome.png",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_captions(bundle: dict, captions: str) -> None:
    expected = "WEBVTT\n\n"
    if bundle["presentation"]["presentation"].get("show_justification_text") is not False:
        reveal = next(scene for scene in bundle["scenes"] if scene["stage"] == "reveal")
        analysis = next((layer["content"] for layer in reveal["layers"] if layer["id"] == "analysis"), {})
        reason = bundle["localizations"].get(analysis.get("reasonKey", "video.analysis.expert_reason"))
        if not reason:
            raise ValueError("flagship Arabic captions lack governed text")
        start = next(item["startFrame"] for item in bundle["timeline"]["segments"] if item["stage"] == "reveal")
        fps = bundle["render"]["fps"]
        expected += f"{_timestamp(start, fps)} --> {_timestamp(start + 240, fps)}\n{reason}\n"
    if captions != expected:
        raise ValueError("flagship Arabic captions differ from the governed export")


def main() -> None:
    bundle = RenderBundleV1(json.loads(BUNDLE.read_text(encoding="utf-8")))
    if not bundle.verify():
        raise ValueError("flagship render bundle verification failed")
    audio_finishing = json.loads(AUDIO_EVIDENCE.read_text(encoding="utf-8"))
    if (
        audio_finishing.get("status") != "passed"
        or audio_finishing.get("bundleSha256") != bundle.bundle_sha256
        or audio_finishing.get("music") is not False
        or audio_finishing.get("videoStreamPreserved") is not True
        or audio_finishing.get("output", {}).get("sha256") != _sha(VIDEO)
    ):
        raise ValueError("flagship audio-finishing evidence is missing or stale")
    timeline = bundle.value["timeline"]
    segments = {item["stage"]: item for item in timeline["segments"]}
    if [item["stage"] for item in timeline["segments"]] != [
        "hook", "rewind", "replay", "question", "reveal", "continuation", "outcome"
    ] or timeline["totalFrames"] != 4320:
        raise ValueError("flagship must use the governed 72-second export timeline")
    context = next(scene for scene in bundle.value["scenes"] if scene["stage"] == "continuation")
    lifecycle = next(
        layer["content"] for layer in context["layers"] if layer["id"] == "public-lifecycle"
    )
    trick_count = len(lifecycle["completedTricks"])
    play_count = sum(len(trick["plays"]) for trick in lifecycle["completedTricks"])
    if trick_count != 8 or play_count != 32:
        raise ValueError("flagship source must contain exactly eight complete tricks")
    projects = lifecycle.get("projects", [])
    round_score = lifecycle.get("roundScore") or {}
    if not projects or any(not project.get("revealed") for project in projects):
        raise ValueError("flagship must carry the authoritative revealed projects")
    missing_project_labels = [
        project["kind"]
        for project in projects
        if not bundle.value["localizations"].get(f"video.project.{project['kind']}")
    ]
    if missing_project_labels:
        raise ValueError(f"project names must be localized: {missing_project_labels}")
    layout = bundle.value["presentation"]["layout"]
    if layout.get("project_announcement_trick") != 1 or layout.get("project_reveal_trick") != 2:
        raise ValueError("project timing must announce on trick one and reveal on trick two")
    timeline_profile = bundle.value["presentation"]["timeline"]
    if (
        timeline_profile.get("project_call_seconds") != 6.0
        or timeline_profile.get("project_reveal_seconds") != 6.0
        or layout.get("persist_revealed_projects") is not False
    ):
        raise ValueError("project mention and reveal must each last six seconds without persistence")
    project_timing = []
    for project in projects:
        owner = project["owner"]
        owner_turns = [
            next(
                (index for index, play in enumerate(trick["plays"]) if play["seat"] == owner),
                None,
            )
            for trick in lifecycle["completedTricks"][:2]
        ]
        if any(turn is None for turn in owner_turns):
            raise ValueError(f"project owner lacks a turn in the first two tricks: {owner}")
        project_timing.append(
            {
                "owner": owner,
                "kind": project["kind"],
                "announcementTrick": 1,
                "announcementSeconds": 6.0,
                "revealTrick": 2,
                "revealSeconds": 6.0,
                "playContinues": True,
                "persistentBadge": False,
                "ownerTurnPositions": owner_turns,
            }
        )
    if not all(
        isinstance(round_score.get(field), list) and len(round_score[field]) == 2
        for field in ("cardGamePoints", "projectPoints", "awardedPoints")
    ):
        raise ValueError("flagship must carry a complete two-team score breakdown")
    calculated_match_score = [
        previous + awarded
        for previous, awarded in zip(
            lifecycle["preDecisionMatchScore"], round_score["awardedPoints"], strict=True
        )
    ]
    if calculated_match_score != lifecycle["matchScore"]:
        raise ValueError("score breakdown does not reconcile to the final match score")

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        candidates = sorted(
            (ROOT / "src/baloot_platform/video/remotion/node_modules/@remotion").glob(
                "compositor-*/ffprobe*"
            )
        )
        if not candidates:
            raise ValueError("ffprobe is required to verify flagship media")
        ffprobe = str(candidates[0])
    result = run_media_command(
        [
            ffprobe,
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=index,codec_name,width,height,r_frame_rate,pix_fmt,color_space,color_primaries,color_transfer,sample_rate,channels,nb_read_frames:format=duration",
            "-of",
            "json",
            str(VIDEO),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    probe = json.loads(result.stdout)
    video_stream, audio_stream = probe["streams"]
    duration = float(probe["format"]["duration"])
    if not (
        video_stream["codec_name"] == "h264"
        and video_stream["width"] == 1080
        and video_stream["height"] == 1920
        and video_stream["pix_fmt"] == "yuv420p"
        and video_stream["color_space"] == "bt709"
        and video_stream["color_primaries"] == "bt709"
        and video_stream["color_transfer"] == "bt709"
        and video_stream["r_frame_rate"] == "60/1"
        and int(video_stream["nb_read_frames"]) == timeline["totalFrames"]
        and 72.0 <= duration <= 72.1
        and audio_stream["codec_name"] == "aac"
        and audio_stream["sample_rate"] == "48000"
        and audio_stream["channels"] == 2
    ):
        raise ValueError(f"flagship media validation failed: {probe}")

    stills: dict[str, dict[str, object]] = {}
    for name in STILLS:
        path = OUTPUT / name
        with Image.open(path) as image:
            if image.size != (1080, 1920):
                raise ValueError(f"invalid flagship checkpoint: {name} {image.size}")
        stills[name] = {"bytes": path.stat().st_size, "sha256": _sha(path)}

    verify_captions(bundle.value, CAPTIONS.read_text(encoding="utf-8"))

    manifest = {
        "schemaVersion": 1,
        "bundleSha256": bundle.bundle_sha256,
        "sourceCoverage": {
            "bids": len(lifecycle["bidHistory"]),
            "plays": play_count,
            "tricks": trick_count,
            "shuffleOrDealRendered": False,
            "projects": len(projects),
        },
        "scoreBreakdown": {
            "cardGamePoints": round_score["cardGamePoints"],
            "projectPoints": round_score["projectPoints"],
            "awardedPoints": round_score["awardedPoints"],
            "matchScore": lifecycle["matchScore"],
        },
        "projectTiming": project_timing,
        "story": {
            "timelineSha256": timeline["timelineSha256"],
            "durationFrames": timeline["totalFrames"],
            "hookFrames": segments["hook"]["durationFrames"],
            "rewindStartsAtFrame": segments["rewind"]["startFrame"],
            "playStartsAtFrame": segments["replay"]["startFrame"],
            "criticalPauseStartsAtFrame": segments["question"]["startFrame"],
            "revealStartsAtFrame": segments["reveal"]["startFrame"],
            "continuationStartsAtFrame": segments["continuation"]["startFrame"],
            "outcomeStartsAtFrame": segments["outcome"]["startFrame"],
        },
        "video": {
            "bytes": VIDEO.stat().st_size,
            "sha256": _sha(VIDEO),
            "mediaProbe": probe,
            "validationBoundary": "all 4320 encoded video frames counted by ffprobe; bundled ffmpeg lacks null/raw output muxers",
        },
        "audioFinishing": audio_finishing,
        "captions": {"bytes": CAPTIONS.stat().st_size, "sha256": _sha(CAPTIONS)},
        "stills": stills,
    }
    path = OUTPUT / "manifest.json"
    atomic_write_text(
        path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(path)


if __name__ == "__main__":
    main()
