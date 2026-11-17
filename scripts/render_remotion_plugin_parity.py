#!/usr/bin/env python3
# ruff: noqa: E402
"""Render the governed plugin outputs used by the byte-parity release gate."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from baloot_platform.config import build_bundle
from baloot_platform.video.remotion_renderer import RemotionProfessionalRenderer
from baloot_platform.video.render_bundle import RenderBundleV1
from scripts.render_release import render_timeout_seconds


OUTPUT = ROOT / "outputs/remotion-plugin-parity"
EXPORTS = (
    (
        "hooked_flagship",
        ROOT
        / "outputs/professional-video-prototypes/render-bundle-hooked_flagship.json",
        "flagship-plugin-parity",
    ),
    (
        "decision_clip",
        ROOT / "outputs/professional-video-prototypes/render-bundle-decision.json",
        "decision-plugin-parity",
    ),
)


def main() -> None:
    renderer = RemotionProfessionalRenderer(
        build_bundle(ROOT).video,
        repository_root=ROOT,
        timeout_seconds=render_timeout_seconds(),
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for preset, bundle_path, basename in EXPORTS:
        bundle = RenderBundleV1(json.loads(bundle_path.read_text(encoding="utf-8")))
        artifacts = renderer.execute(
            bundle,
            export_preset=preset,
            output_dir=OUTPUT,
            basename=basename,
        )
        print(artifacts.manifest)


if __name__ == "__main__":
    main()
