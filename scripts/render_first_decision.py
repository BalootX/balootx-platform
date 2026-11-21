#!/usr/bin/env python3
# ruff: noqa: E402
"""Render and validate only the frozen first vertical decision clip."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image

from baloot_platform.video.audio_postprocess import finish_remotion_audio
from baloot_platform.video.media_validation import media_tool, run_media_command
from baloot_platform.video.render_bundle import RenderBundleV1

try:
    from .generate_professional_video_prototypes import generate, _write
except ImportError:
    from generate_professional_video_prototypes import generate, _write


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    source = ROOT / "examples/curated-decision-20261114.json"
    curation = json.loads(source.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(source.read_bytes()).hexdigest() != curation["sourceSha256"]:
        raise ValueError("curated source hash mismatch")
    _, bundle_path = generate(output / "inputs", source=source, seed=20261114)
    bundle = RenderBundleV1(json.loads(bundle_path.read_text(encoding="utf-8")))
    if not bundle.verify() or bundle.value["presentation"]["presentation"].get("show_justification_text") is not False:
        raise ValueError("first clip requires a verified no-justification-text bundle")
    os.environ.update({
        "BALOOTX_RENDER_BUNDLE": str(bundle_path),
        "BALOOTX_PYTHON": sys.executable,
        "BALOOTX_RENDER_OUTPUT": str(output),
        "BALOOTX_RENDER_BASENAME": "balootx-first-decision",
        "BALOOTX_PLAYBACK_EVIDENCE": "1",
    })
    started = time.monotonic()
    subprocess.run(
        ["node", "scripts/render-decision-clip.mjs"], check=True,
        cwd=ROOT / "src/baloot_platform/video/remotion", timeout=1800,
    )
    video = output / "balootx-first-decision.mp4"
    finish_remotion_audio(
        video, bundle, repository_root=ROOT,
        evidence_path=output / "balootx-first-decision.audio.json",
    )
    try:
        from . import verify_professional_decision_clip as verifier
    except ImportError:
        import verify_professional_decision_clip as verifier
    verifier.main()
    ffmpeg = media_tool(ROOT, "ffmpeg")
    run_media_command([
        str(ffmpeg), "-v", "error", "-threads", "4", "-i", str(video),
        "-filter_threads", "4", "-filter_complex_threads", "4", "-threads", "4",
        "-c:v", "rawvideo", "-c:a", "pcm_s16le", "-f", "null", "-",
    ], check=True)
    sheet = Image.new("RGB", (810, 960))
    for index, name in enumerate(verifier.STILLS):
        with Image.open(output / name) as still:
            sheet.paste(still.resize((270, 480)), ((index % 3) * 270, (index // 3) * 480))
    sheet.save(output / "contact-sheet.png")
    _write(output / "curation.json", curation)
    _write(output / "acceptance.json", {
        "schemaVersion": 1, "status": "MEDIA_CHECKS_PASSED_EDITORIAL_REVIEW_PENDING",
        "sourceHead": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "sourceTree": subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, text=True).strip(),
        "bundleSha256": bundle.bundle_sha256,
        "sourceSha256": curation["sourceSha256"],
        "videoSha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "videoBytes": video.stat().st_size, "elapsedSeconds": time.monotonic() - started,
        "legalReplay": "passed", "observerSafeBundle": "passed",
        "ffprobeStillsAudioAndDecode": "passed", "publication": "NOT_RUN",
        "tools": {name: subprocess.check_output(command, text=True).splitlines()[0] for name, command in {
            "python": [sys.executable, "--version"], "node": ["node", "--version"],
            "ffmpeg": [str(ffmpeg), "-version"],
            "ffprobe": [str(media_tool(ROOT, "ffprobe")), "-version"],
        }.items()},
    })
    total = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    if total > 500_000_000:
        raise ValueError(f"single artifact exceeds 500 MB: {total}")
    print(video)


if __name__ == "__main__":
    main()
