#!/usr/bin/env python3
# ruff: noqa: E402
"""Verify the governed short decision export and write deterministic evidence."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baloot_platform.video.render_bundle import RenderBundleV1
from baloot_platform.video.media_validation import run_media_command
from baloot_platform.video.remotion_renderer import _timestamp
from baloot_platform.filesystem import atomic_write_text

OUTPUT = Path(os.environ.get("BALOOTX_RENDER_OUTPUT", ROOT / "outputs/professional-video-decision"))
BASENAME = os.environ.get("BALOOTX_RENDER_BASENAME", "balootx-decision-clip")
VIDEO = OUTPUT / f"{BASENAME}.mp4"
CAPTIONS = OUTPUT / f"{BASENAME}.ar.vtt"
BUNDLE = Path(os.environ.get("BALOOTX_RENDER_BUNDLE", ROOT / "outputs/professional-video-prototypes/render-bundle-decision.json"))
AUDIO_EVIDENCE = OUTPUT / f"{BASENAME}.audio.json"
STILLS = (
    "01-hook.png",
    "02-rewind.png",
    "03-context.png",
    "04-decision.png",
    "05-reveal.png",
    "06-continuation.png",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    bundle = RenderBundleV1(json.loads(BUNDLE.read_text(encoding="utf-8")))
    if not bundle.verify() or bundle.value["exportPreset"] != "decision_clip":
        raise ValueError("decision render bundle verification failed")
    audio_finishing = json.loads(AUDIO_EVIDENCE.read_text(encoding="utf-8"))
    if (
        audio_finishing.get("status") != "passed"
        or audio_finishing.get("bundleSha256") != bundle.bundle_sha256
        or audio_finishing.get("music") is not False
        or audio_finishing.get("videoStreamPreserved") is not True
        or audio_finishing.get("output", {}).get("sha256") != _sha(VIDEO)
    ):
        raise ValueError("decision audio-finishing evidence is missing or stale")
    timeline = bundle.value["timeline"]
    segments = {item["stage"]: item for item in timeline["segments"]}
    exports = bundle.value["presentation"]["exports"]["decision_clip"]
    if exports["include_complete_replay"] is not False:
        raise ValueError("decision clip must be a governed derived export")
    fps = bundle.value["render"]["fps"]
    expected_duration = timeline["totalFrames"] / fps
    context = next(scene for scene in bundle.value["scenes"] if scene["stage"] == "continuation")
    lifecycle = next(
        layer["content"] for layer in context["layers"] if layer["id"] == "public-lifecycle"
    )
    play_count = sum(len(trick["plays"]) for trick in lifecycle["completedTricks"])
    if play_count != 32:
        raise ValueError("decision export must derive from the same complete authoritative laffah")

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        candidates = sorted(
            (ROOT / "src/baloot_platform/video/remotion/node_modules/@remotion").glob(
                "compositor-*/ffprobe*"
            )
        )
        if not candidates:
            raise ValueError("ffprobe is required to verify decision media")
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
        and video_stream["r_frame_rate"] == f"{fps}/1"
        and int(video_stream["nb_read_frames"]) == timeline["totalFrames"]
        and expected_duration <= duration <= expected_duration + 0.1
        and audio_stream["codec_name"] == "aac"
        and audio_stream["sample_rate"] == "48000"
        and audio_stream["channels"] == 2
    ):
        raise ValueError(f"decision media validation failed: {probe}")

    stills: dict[str, dict[str, object]] = {}
    for name in STILLS:
        path = OUTPUT / name
        with Image.open(path) as image:
            if image.size != (1080, 1920):
                raise ValueError(f"invalid decision checkpoint: {name} {image.size}")
        stills[name] = {"bytes": path.stat().st_size, "sha256": _sha(path)}

    if bundle.value["presentation"]["presentation"].get("show_justification_text") is False:
        if CAPTIONS.exists():
            raise ValueError("justification captions are forbidden by the frozen bundle")
        captions_evidence = {"policy": "no_justification_text", "present": False}
    else:
        captions = CAPTIONS.read_text(encoding="utf-8")
        reveal = segments["reveal"]
        caption_start = reveal["startFrame"]
        caption_end = caption_start + min(240, reveal["durationFrames"])
        if f"{_timestamp(caption_start, fps)} --> {_timestamp(caption_end, fps)}" not in captions:
            raise ValueError("decision Arabic captions are missing or mistimed")
        captions_evidence = {"bytes": CAPTIONS.stat().st_size, "sha256": _sha(CAPTIONS)}

    manifest = {
        "schemaVersion": 1,
        "bundleSha256": bundle.bundle_sha256,
        "sceneId": bundle.value["identity"]["sceneId"],
        "sourceCoverage": {
            "sourcePlays": play_count,
            "includeCompleteReplay": False,
            "derivedFromAuthoritativeLaffah": True,
        },
        "story": {
            "timelineSha256": timeline["timelineSha256"],
            "durationFrames": timeline["totalFrames"],
            "hookFrames": segments["hook"]["durationFrames"],
            "rewindFrames": segments["rewind"]["durationFrames"],
            "contextStartsAtFrame": segments["context"]["startFrame"],
            "decisionStartsAtFrame": segments["question"]["startFrame"],
            "revealStartsAtFrame": segments["reveal"]["startFrame"],
            "continuationStartsAtFrame": segments["continuation"]["startFrame"],
        },
        "video": {"bytes": VIDEO.stat().st_size, "sha256": _sha(VIDEO), "mediaProbe": probe},
        "audioFinishing": audio_finishing,
        "captions": captions_evidence,
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
