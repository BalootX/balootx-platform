#!/usr/bin/env python3
# ruff: noqa: E402
"""Verify and inventory the professional video prototype approval package."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baloot_platform.video.render_bundle import RenderBundleV1
from baloot_platform.filesystem import atomic_write_text

OUTPUT = ROOT / "outputs/professional-video-prototypes"
STILLS = ("normal.png", "hook.png", "reveal.png")
MOTION = ("card-motion.mp4", "card-motion-reduced.mp4")
FORBIDDEN_PLAYER_COPY = (
    "يمينك",
    "يسارك",
    "شريكك",
    "الخصم الأيمن",
    "سير اللعب",
    "آمن للمشاهد",
    "renderer version",
    "scene hash",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _probe(path: Path) -> dict[str, object]:
    executable = shutil.which("ffprobe")
    if executable is None:
        candidates = sorted(
            (ROOT / "src/baloot_platform/video/remotion/node_modules/@remotion").glob(
                "compositor-*/ffprobe*"
            )
        )
        if not candidates:
            raise ValueError("ffprobe is required to verify prototype media")
        executable = str(candidates[0])
    result = subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,pix_fmt:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(result.stdout)


def main() -> None:
    decision_path = OUTPUT / "render-bundle-decision.json"
    flagship_path = OUTPUT / "render-bundle-hooked_flagship.json"
    decision = RenderBundleV1(json.loads(decision_path.read_text(encoding="utf-8")))
    flagship = RenderBundleV1(json.loads(flagship_path.read_text(encoding="utf-8")))
    if not decision.verify() or not flagship.verify():
        raise ValueError("render bundle verification failed")

    files: dict[str, dict[str, object]] = {}
    for name in STILLS:
        path = OUTPUT / name
        with Image.open(path) as image:
            if image.size != (1080, 1920) or image.mode not in {"RGB", "RGBA"}:
                raise ValueError(f"invalid prototype still: {name} {image.size} {image.mode}")
            files[name] = {
                "sha256": _sha(path),
                "bytes": path.stat().st_size,
                "width": image.width,
                "height": image.height,
            }
    for name in MOTION:
        path = OUTPUT / name
        probe = _probe(path)
        stream = probe["streams"][0]
        duration = float(probe["format"]["duration"])
        if (
            stream["codec_name"] != "h264"
            or stream["width"] != 1080
            or stream["height"] != 1920
            or stream["r_frame_rate"] != "60/1"
            or stream["pix_fmt"] != "yuv420p"
            or not 1.9 <= duration <= 2.1
        ):
            raise ValueError(f"invalid card motion sample: {name} {probe}")
        files[name] = {
            "sha256": _sha(path),
            "bytes": path.stat().st_size,
            "mediaProbe": probe,
        }
    if files[MOTION[0]]["sha256"] == files[MOTION[1]]["sha256"]:
        raise ValueError("normal and reduced-motion samples must be visually distinct")

    authored = "\n".join(
        (ROOT / relative).read_text(encoding="utf-8")
        for relative in (
            "src/baloot_platform/video/remotion/src/professional-prototype.tsx",
            "src/baloot_platform/video/remotion/src/style.css",
        )
    ).casefold()
    found = [term for term in FORBIDDEN_PLAYER_COPY if term.casefold() in authored]
    if found:
        raise ValueError(f"obvious or technical player copy remains: {found}")

    question = next(scene for scene in decision.value["scenes"] if scene["stage"] == "question")
    reveal = next(scene for scene in decision.value["scenes"] if scene["stage"] == "reveal")
    manifest = {
        "schemaVersion": 1,
        "status": "owner_visual_approval_required",
        "observerSafe": True,
        "domainCalculations": False,
        "canvas": {"width": 1080, "height": 1920, "fps": 60},
        "exactState": {
            "questionSceneId": question["sceneId"],
            "revealSceneId": reveal["sceneId"],
            "timelineSha256": decision.value["timeline"]["timelineSha256"],
        },
        "renderBundles": {
            "decision_clip": decision.bundle_sha256,
            "hooked_flagship": flagship.bundle_sha256,
        },
        "files": dict(sorted(files.items())),
        "approvalGate": (
            "Owner approves normal play, unanswered hook, reveal, card motion, "
            "and reduced motion before the full renderer replaces Pillow."
        ),
    }
    target = OUTPUT / "manifest.json"
    atomic_write_text(
        target,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(target)


if __name__ == "__main__":
    main()
