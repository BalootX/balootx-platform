#!/usr/bin/env python3
"""Generate governed decode, color, loudness, freshness, and frame-parity evidence."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baloot_platform.video.media_validation import (  # noqa: E402
    compare_source_frame,
    media_tool,
    probe_loudness,
    probe_media,
    validate_loudness,
    validate_media_probe,
    validate_release_loudness_target,
)
from baloot_platform.video.render_bundle import RenderBundleV1  # noqa: E402
from baloot_platform.filesystem import atomic_write_text  # noqa: E402

OUTPUT = ROOT / "release/media-probe.json"
MEDIA = (
    {
        "video": ROOT / "outputs/professional-video-flagship/balootx-complete-laffah-audit.mp4",
        "bundle": ROOT / "outputs/professional-video-prototypes/render-bundle-hooked_flagship.json",
        "manifest": ROOT / "outputs/professional-video-flagship/manifest.json",
        "checkpoints": (("01-hook.png", "hook", 0), ("05-reveal.png", "reveal", 120)),
    },
    {
        "video": ROOT / "outputs/professional-video-decision/balootx-decision-clip.mp4",
        "bundle": ROOT / "outputs/professional-video-prototypes/render-bundle-decision.json",
        "manifest": ROOT / "outputs/professional-video-decision/manifest.json",
        "checkpoints": (("01-hook.png", "hook", 0), ("05-reveal.png", "reveal", 88)),
    },
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_media(item: dict[str, Any], *, ffmpeg: Path, ffprobe: Path) -> dict[str, Any]:
    video: Path = item["video"]
    bundle_path: Path = item["bundle"]
    manifest_path: Path = item["manifest"]
    for required in (video, bundle_path, manifest_path):
        if not required.is_file():
            raise RuntimeError(f"missing release media evidence input: {required}")

    bundle = RenderBundleV1(json.loads(bundle_path.read_text(encoding="utf-8")))
    if not bundle.verify():
        raise RuntimeError(f"stale or invalid render bundle: {bundle_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    video_hash = sha256(video)
    if (
        manifest.get("bundleSha256") != bundle.bundle_sha256
        or manifest.get("video", {}).get("sha256") != video_hash
        or manifest.get("video", {}).get("bytes") != video.stat().st_size
    ):
        raise RuntimeError(f"render manifest does not pin the current bundle and video: {manifest_path}")
    audio_finishing = manifest.get("audioFinishing")
    validate_release_loudness_target(
        audio_finishing.get("loudness") if isinstance(audio_finishing, dict) else None
    )

    timeline = bundle.value["timeline"]
    probe = probe_media(video, ffprobe=ffprobe)
    validate_media_probe(
        probe,
        width=int(bundle.value["render"]["width"]),
        height=int(bundle.value["render"]["height"]),
        fps=int(bundle.value["render"]["fps"]),
        minimum_frames=int(timeline["totalFrames"]),
        expected_frames=int(timeline["totalFrames"]),
        audio_rate=48_000,
    )
    loudness = probe_loudness(video, ffmpeg=ffmpeg)
    validate_loudness(loudness)

    segments = {segment["stage"]: segment for segment in timeline["segments"]}
    comparisons = []
    for filename, stage, offset in item["checkpoints"]:
        source = video.parent / filename
        if not source.is_file():
            raise RuntimeError(f"missing source checkpoint: {source}")
        frame = int(segments[stage]["startFrame"]) + int(offset)
        comparison = compare_source_frame(
            video, source, frame=frame, fps=int(bundle.value["render"]["fps"]), ffmpeg=ffmpeg,
        )
        comparison["sourceSha256"] = sha256(source)
        comparisons.append(comparison)

    return {
        "path": video.relative_to(ROOT).as_posix(),
        "bytes": video.stat().st_size,
        "sha256": video_hash,
        "width": int(probe["video"]["width"]),
        "height": int(probe["video"]["height"]),
        "bundlePath": bundle_path.relative_to(ROOT).as_posix(),
        "bundleSha256": bundle.bundle_sha256,
        "renderer": bundle.value["renderer"],
        "mediaProbe": probe,
        "loudness": loudness,
        "sourceComparisons": comparisons,
    }


def main() -> None:
    ffmpeg = media_tool(ROOT, "ffmpeg")
    ffprobe = media_tool(ROOT, "ffprobe")
    value = {
        "schemaVersion": 2,
        "release": "BalootX",
        "status": "passed",
        "requiredAspectRatio": "9:16",
        "forbiddenReleaseAspectRatios": ["16:9", "1:1"],
        "media": [
            verify_media(item, ffmpeg=ffmpeg, ffprobe=ffprobe)
            for item in MEDIA
        ],
        "proofBoundary": (
            "The report pins current renderer source, bundle, encoded bytes, full decode, "
            "BT.709 tags, audio loudness, and selected source-versus-encoded frames. "
            "It does not replace human Arabic or professional visual acceptance."
        ),
    }
    atomic_write_text(
        OUTPUT,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
