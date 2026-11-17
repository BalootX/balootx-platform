#!/usr/bin/env python3
"""Build the content-first player from exact professional media outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from baloot_platform.filesystem import atomic_write_text  # noqa: E402

STATIC = ROOT / "src/baloot_platform/service/static/scene-preview"
VIDEO_PROFILE = ROOT / "configs/video/premium.yaml"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _governed_copy(profile_path: Path = VIDEO_PROFILE) -> tuple[dict[str, str], str]:
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    raw = profile.get("preview_copy")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("video profile preview_copy must be a non-empty object")
    copy = {str(key): str(value) for key, value in raw.items()}
    if any(not value.strip() for value in copy.values()):
        raise ValueError("video profile preview_copy values must be non-empty")
    payload = json.dumps(copy, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return copy, hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _chapters(bundle: dict[str, Any], labels: dict[str, str]) -> list[dict[str, Any]]:
    fps = int(bundle["render"]["fps"])
    return [
        {"label": labels[item["stage"]], "startSeconds": item["startFrame"] / fps}
        for item in bundle["timeline"]["segments"]
        if item["stage"] in labels
    ]


def _artifact(
    *,
    bundle: dict[str, Any],
    scenes: list[dict[str, str]],
    captions: str,
    chapter_labels: dict[str, str],
) -> dict[str, Any]:
    question = next(scene for scene in bundle["scenes"] if scene["stage"] == "question")
    layers = [
        {"name": str(layer["kind"]), "disclosure": str(layer["disclosure"])}
        for layer in question["layers"]
    ]
    return {
        "width": int(bundle["render"]["width"]),
        "height": int(bundle["render"]["height"]),
        "observerSafe": True,
        "identity": bundle["identity"]["sceneId"],
        "layers": layers,
        "chapters": _chapters(bundle, chapter_labels),
        "captions": captions,
        "scenes": scenes,
    }


def _professional_artifacts(
    input_dir: Path,
    copy: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Path]]:
    paths = {
        "flagshipManifest": input_dir / "professional-video-flagship/manifest.json",
        "flagshipVideo": input_dir / "professional-video-flagship/balootx-complete-laffah-audit.mp4",
        "decisionManifest": input_dir / "professional-video-decision/manifest.json",
        "decisionVideo": input_dir / "professional-video-decision/balootx-decision-clip.mp4",
        "flagshipBundle": input_dir / "professional-video-prototypes/render-bundle-hooked_flagship.json",
        "decisionBundle": input_dir / "professional-video-prototypes/render-bundle-decision.json",
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError(f"professional player artifacts are missing: {missing}")
    flagship = _read(paths["flagshipBundle"])
    decision = _read(paths["decisionBundle"])
    if flagship.get("exportPreset") != "hooked_flagship" or decision.get("exportPreset") != "decision_clip":
        raise ValueError("professional player bundles have invalid export presets")
    for bundle in (flagship, decision):
        if (int(bundle["render"]["width"]), int(bundle["render"]["height"])) != (1080, 1920):
            raise ValueError("professional player media must be 1080x1920")
    artifacts = {
        "flagship": _artifact(
            bundle=flagship,
            captions="../professional-video-flagship/balootx-complete-laffah-audit.ar.vtt",
            chapter_labels={
                "hook": copy["chapter_hook"],
                "replay": copy["chapter_replay"],
                "question": copy["chapter_question"],
                "reveal": copy["chapter_reveal"],
                "continuation": copy["chapter_continuation"],
                "outcome": copy["chapter_outcome"],
            },
            scenes=[
                {"kind": "video", "label": copy["scene_video"], "src": "../professional-video-flagship/balootx-complete-laffah-audit.mp4"},
                {"kind": "poster", "label": copy["scene_poster"], "src": "../professional-video-flagship/04-critical-pause.png"},
                {"kind": "reveal", "label": copy["scene_reveal"], "src": "../professional-video-flagship/05-reveal.png"},
                {"kind": "outcome", "label": copy["scene_outcome"], "src": "../professional-video-flagship/07-outcome.png"},
            ],
        ),
        "hitl": _artifact(
            bundle=decision,
            captions="../professional-video-decision/balootx-decision-clip.ar.vtt",
            chapter_labels={
                "hook": copy["chapter_hook"],
                "context": copy["chapter_context"],
                "question": copy["chapter_question"],
                "reveal": copy["chapter_reveal"],
                "continuation": copy["chapter_continuation"],
            },
            scenes=[
                {"kind": "video", "label": copy["scene_video"], "src": "../professional-video-decision/balootx-decision-clip.mp4"},
                {"kind": "poster", "label": copy["scene_poster"], "src": "../professional-video-decision/04-decision.png"},
                {"kind": "reveal", "label": copy["scene_reveal"], "src": "../professional-video-decision/05-reveal.png"},
            ],
        ),
    }
    return artifacts, paths


def _build_unchecked(input_dir: Path, output_dir: Path, *, profile_path: Path = VIDEO_PROFILE) -> None:
    copy, copy_sha256 = _governed_copy(profile_path)
    artifacts, sources = _professional_artifacts(input_dir, copy)
    output_dir.mkdir(parents=True, exist_ok=False)
    for source in sorted(STATIC.iterdir()):
        shutil.copyfile(source, output_dir / source.name)
    data = {
        "schemaVersion": 1,
        "purpose": "static_scene_and_media_qa_only",
        "domainCalculations": False,
        "copy": copy,
        "artifacts": artifacts,
    }
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    atomic_write_text(
        output_dir / "qa-data.js",
        f"window.BALOOTX_QA={payload};\n",
    )
    manifest = {
        "schemaVersion": 1,
        "observerSafe": True,
        "domainCalculations": False,
        "copySha256": copy_sha256,
        "flagshipManifestSha256": _sha(sources["flagshipManifest"]),
        "decisionManifestSha256": _sha(sources["decisionManifest"]),
        "flagshipVideoSha256": _sha(sources["flagshipVideo"]),
        "decisionVideoSha256": _sha(sources["decisionVideo"]),
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha(path)}
            for path in sorted(output_dir.iterdir())
            if path.is_file()
        },
    }
    manifest["previewIdentity"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    atomic_write_text(
        output_dir / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def build(
    input_dir: Path,
    output_dir: Path,
    *,
    check: bool = False,
    refresh: bool = False,
    profile_path: Path = VIDEO_PROFILE,
) -> None:
    if check and refresh:
        raise ValueError("check and refresh are mutually exclusive")
    if not check and not refresh:
        _build_unchecked(input_dir, output_dir, profile_path=profile_path)
        return
    if check and not output_dir.is_dir():
        raise ValueError("checked-in scene preview is missing")
    with tempfile.TemporaryDirectory(prefix="scene-preview-check-") as temporary:
        generated = Path(temporary) / "preview"
        _build_unchecked(input_dir, generated, profile_path=profile_path)
        if refresh:
            output_dir.mkdir(parents=True, exist_ok=True)
            generated_names = {path.name for path in generated.iterdir()}
            output_names = {path.name for path in output_dir.iterdir()}
            if output_names - generated_names:
                raise ValueError("refusing to refresh preview with unexpected files")
            for source in generated.iterdir():
                shutil.copyfile(source, output_dir / source.name)
            return
        expected = {path.name: path.read_bytes() for path in generated.iterdir()}
        actual = {path.name: path.read_bytes() for path in output_dir.iterdir()}
        if actual != expected:
            raise ValueError("checked-in scene preview is stale")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "outputs")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/scene-preview")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    build(args.input.resolve(), args.output.resolve(), check=args.check, refresh=args.refresh)


if __name__ == "__main__":
    main()
