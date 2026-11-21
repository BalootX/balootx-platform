"""Single governed media-probe authority for rendered BalootX video."""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageChops, ImageStat

from baloot_platform.video.media_probe import (
    probe_media as probe_media,
    run_media_command as run_media_command,
    validate_media_probe as validate_media_probe,
)


REMOTION_RENDERER_VERSION = "21.10.7"
APPROVED_LAYOUT_TABLE = "scripts/approved-layout-adjacencies.json"
APPROVED_LAYOUT_TABLE_SHA256 = "7c1aeee912e4b032d8fb35b50cacde5f464ebf279cbb34464e4054e322ae6a1c"
MAXIMUM_LOUDNESS_DEVIATION_LUFS = 2.0
MAXIMUM_TRUE_PEAK_DBTP = -1.0
MINIMUM_SOURCE_COMPARISON_PSNR_DB = 28.0
MAXIMUM_SOURCE_COMPARISON_MAE = 6.0


def renderer_source_files(root: Path) -> tuple[Path, ...]:
    package = root / "src/baloot_platform/video/remotion"
    values = [
        root / "src/baloot_platform/video/media_validation.py",
        root / "src/baloot_platform/video/media_probe.py",
        root / "src/baloot_platform/video/audio_postprocess.py",
        root / "src/baloot_platform/video/remotion_renderer.py",
        root / "src/baloot_platform/video/render_bundle.py",
        package / "package.json",
        package / "package-lock.json",
        package / "tsconfig.json",
        package / APPROVED_LAYOUT_TABLE,
    ]
    values.extend(sorted((package / "scripts").glob("*.mjs")))
    values.extend(sorted((package / "src").glob("*.ts")))
    values.extend(sorted((package / "src").glob("*.tsx")))
    values.extend(sorted((package / "src").glob("*.css")))
    missing = [path for path in values if not path.is_file()]
    if missing:
        raise RuntimeError(f"renderer source is incomplete: {missing}")
    return tuple(
        sorted(set(values), key=lambda item: item.relative_to(root).as_posix())
    )


def renderer_source_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in renderer_source_files(root):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def validate_renderer_identity(renderer: Mapping[str, Any], *, root: Path) -> None:
    expected = {
        "plugin": "remotion_professional",
        "contractVersion": 1,
        "version": REMOTION_RENDERER_VERSION,
        "sourceSha256": renderer_source_sha256(root),
    }
    if dict(renderer) != expected:
        raise ValueError("render bundle does not pin the current renderer source")


def media_tool(root: Path, name: str) -> Path:
    system = shutil.which(name)
    if system:
        return Path(system).resolve()
    candidates = sorted(
        (root / "src/baloot_platform/video/remotion/node_modules/@remotion").glob(
            f"compositor-*/{name}*"
        )
    )
    executables = [path.resolve() for path in candidates if path.is_file()]
    if not executables:
        raise RuntimeError(f"{name} is required for governed media validation")
    return executables[0]


def probe_loudness(
    path: Path,
    *,
    ffmpeg: Path,
    target_lufs: float,
    true_peak_dbtp: float = -1.5,
) -> dict[str, float]:
    process = run_media_command(
        [
            str(ffmpeg),
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-vn",
            "-af",
            f"loudnorm=I={target_lufs}:TP={true_peak_dbtp}:LRA=11:print_format=json",
            "-f",
            "wav",
            "-y",
            os.devnull,
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    matches = re.findall(r"\{\s*\"input_i\"[\s\S]*?\}", process.stderr)
    if not matches:
        raise ValueError("FFmpeg loudness analysis did not return measured evidence")
    raw = json.loads(matches[-1])
    return {
        "integratedLufs": float(raw["input_i"]),
        "truePeakDbtp": float(raw["input_tp"]),
        "loudnessRangeLu": float(raw["input_lra"]),
        "thresholdLufs": float(raw["input_thresh"]),
        "targetLufs": float(target_lufs),
        "targetOffset": float(raw["target_offset"]),
    }


def validate_loudness(evidence: Mapping[str, Any]) -> None:
    if (
        abs(float(evidence["integratedLufs"]) - float(evidence["targetLufs"]))
        > MAXIMUM_LOUDNESS_DEVIATION_LUFS
        or float(evidence["truePeakDbtp"]) > MAXIMUM_TRUE_PEAK_DBTP
    ):
        raise ValueError(f"governed audio loudness validation failed: {evidence}")


def compare_images(source: Path, encoded: Path) -> dict[str, float | int | str]:
    with Image.open(source) as source_image, Image.open(encoded) as encoded_image:
        first = source_image.convert("RGB")
        second = encoded_image.convert("RGB")
        if first.size != second.size:
            raise ValueError("source and encoded checkpoint dimensions differ")
        difference = ImageChops.difference(first, second)
        stats = ImageStat.Stat(difference)
        mae = sum(stats.mean) / len(stats.mean)
        mse = sum(value * value for value in stats.rms) / len(stats.rms)
        psnr = math.inf if mse == 0 else 10 * math.log10((255 * 255) / mse)
        evidence: dict[str, float | int | str] = {
            "source": source.name,
            "encoded": encoded.name,
            "width": first.width,
            "height": first.height,
            "meanAbsoluteError": round(mae, 6),
            "psnrDb": 999.0 if math.isinf(psnr) else round(psnr, 6),
        }
    if mae > MAXIMUM_SOURCE_COMPARISON_MAE or psnr < MINIMUM_SOURCE_COMPARISON_PSNR_DB:
        raise ValueError(f"source-versus-encoded frame comparison failed: {evidence}")
    return evidence


def compare_source_frame(
    video: Path,
    source: Path,
    *,
    frame: int,
    fps: int,
    ffmpeg: Path,
) -> dict[str, float | int | str]:
    if frame < 0 or fps <= 0:
        raise ValueError("checkpoint frame and fps must be valid")
    with tempfile.TemporaryDirectory(prefix="baloot-media-frame-") as directory:
        encoded = Path(directory) / f"frame-{frame}.png"
        # ponytail: this seeks validated zero-start CFR renders; use indexed decode if VFR is added.
        run_media_command(
            [
                str(ffmpeg),
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{frame / fps:.12f}",
                "-i",
                str(video),
                "-an",
                "-frames:v",
                "1",
                "-y",
                str(encoded),
            ],
            check=True,
            capture_output=True,
            timeout=180,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if not encoded.is_file():
            raise ValueError("encoded video is missing a playback checkpoint frame")
        evidence = compare_images(source, encoded)
    evidence["frame"] = frame
    return evidence
