#!/usr/bin/env python3
"""Fail-fast prerequisite check for the V20 exact verifier."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import ImageFont, features

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "release/verifier-preflight-v20.json"
COMMAND_TIMEOUT_ENV = "BALOOT_VERIFY_COMMAND_TIMEOUT_SECONDS"
COMMAND_TIMEOUT_SECONDS = max(
    1, int(os.environ.get(COMMAND_TIMEOUT_ENV, "10"))
)
OUTPUT_LIMIT = 4096
ARABIC_SAMPLE = "صكّة البلوت جولة أكلة شرق غرب شمال جنوب ٠١٢٣٤٥٦٧٨٩"
LATIN_SAMPLE = "BalootX A K Q J 10 0123456789 ♠ ♥ ♦ ♣"

FONT_REQUIREMENTS = {
    "notoSansArabicRegular": (
        "NotoSansArabic-Regular.ttf",
        ARABIC_SAMPLE,
    ),
    "notoSansArabicBold": (
        "NotoSansArabic-Bold.ttf",
        ARABIC_SAMPLE,
    ),
    "notoKufiArabicRegular": (
        "NotoKufiArabic-Regular.ttf",
        ARABIC_SAMPLE,
    ),
    "notoKufiArabicBold": (
        "NotoKufiArabic-Bold.ttf",
        ARABIC_SAMPLE,
    ),
    "dejavuSansRegular": (
        "DejaVuSans.ttf",
        LATIN_SAMPLE,
    ),
    "dejavuSansBold": (
        "DejaVuSans-Bold.ttf",
        LATIN_SAMPLE,
    ),
}

FONT_ROOTS = (
    ROOT / "assets/fonts",
    ROOT / "src/baloot_platform/resources/assets/fonts",
    Path("/usr/share/fonts/truetype/noto"),
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("C:/Windows/Fonts"),
)
FONT_MANIFEST = ROOT / "assets/fonts/manifest.json"
MIN_FREE_DISK_BYTES = max(
    1, int(os.environ.get("BALOOT_VERIFY_MIN_FREE_DISK_BYTES", str(20 * 1024**3)))
)

TOOLS = {
    "cmake": ["cmake", "--version"],
    "ctest": ["ctest", "--version"],
    "gcc": ["gcc", "--version"],
    "g++": ["g++", "--version"],
    "ffmpeg": ["ffmpeg", "-version"],
    "ffprobe": ["ffprobe", "-version"],
    "node": ["node", "--version"],
    "tsc": ["tsc", "--version"],
    "go": ["go", "version"],
}

PYTHON_PACKAGES = (
    "build",
    "fastapi",
    "grpcio",
    "httpx",
    "jsonschema",
    "numpy",
    "Pillow",
    "PyYAML",
    "setuptools",
    "wheel",
)


def utc_now() -> str:
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "315532800"))
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def _bounded(value: str) -> str:
    encoded = value.encode("utf-8", "replace")
    return encoded[-OUTPUT_LIMIT:].decode("utf-8", "replace").strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_check(name: str, command: list[str]) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if not executable:
        return {
            "name": name,
            "status": "failed",
            "error": f"required executable is unavailable: {command[0]}",
        }
    try:
        process = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "name": name,
            "status": "failed",
            "path": executable,
            "error": _bounded(str(exc)),
        }
    output = _bounded(process.stdout or process.stderr)
    return {
        "name": name,
        "status": "passed" if process.returncode == 0 else "failed",
        "path": executable,
        "returnCode": process.returncode,
        "version": output.splitlines()[0] if output else "",
    }


def package_check(distribution: str) -> dict[str, Any]:
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return {
            "name": distribution,
            "status": "failed",
            "error": f"required Python package is unavailable: {distribution}",
        }
    return {"name": distribution, "status": "passed", "version": version}


def resolve_font(filename: str, roots: tuple[Path, ...] = FONT_ROOTS) -> Path | None:
    configured = os.environ.get("BALOOT_VERIFY_FONT_ROOT")
    search_roots = (Path(configured),) if configured else roots
    for root in search_roots:
        candidate = root / filename
        if candidate.is_file():
            return candidate.resolve()
    return None


def _glyph_signature(font: ImageFont.FreeTypeFont, character: str) -> tuple[tuple[int, int], bytes]:
    mask = font.getmask(character)
    return mask.size, bytes(mask)


def unsupported_characters(
    font: ImageFont.FreeTypeFont, text: str
) -> tuple[str, ...]:
    notdef = _glyph_signature(font, "\u0378")
    missing: list[str] = []
    for character in text:
        if character.isspace():
            continue
        signature = _glyph_signature(font, character)
        if signature == notdef or signature[0] == (0, 0):
            missing.append(character)
    return tuple(dict.fromkeys(missing))


def font_check(
    name: str,
    filename: str,
    sample: str,
    roots: tuple[Path, ...] = FONT_ROOTS,
) -> dict[str, Any]:
    path = resolve_font(filename, roots)
    if path is None:
        return {
            "name": name,
            "status": "failed",
            "filename": filename,
            "error": f"required font is unavailable: {filename}",
        }
    try:
        font = ImageFont.truetype(str(path), 40)
        missing = unsupported_characters(font, sample)
    except Exception as exc:
        return {
            "name": name,
            "status": "failed",
            "path": str(path),
            "error": _bounded(str(exc)),
        }
    return {
        "name": name,
        "status": "passed" if not missing else "failed",
        "path": str(path),
        "missingGlyphs": [f"U+{ord(value):04X}" for value in missing],
    }


def font_manifest_check() -> dict[str, Any]:
    if not FONT_MANIFEST.is_file():
        return {
            "name": "bundled-font-manifest",
            "status": "failed",
            "error": f"required font manifest is unavailable: {FONT_MANIFEST}",
        }
    try:
        manifest = json.loads(FONT_MANIFEST.read_text(encoding="utf-8"))
        font_root = FONT_MANIFEST.parent
        for filename, metadata in manifest["fonts"].items():
            path = font_root / filename
            if (
                not path.is_file()
                or path.stat().st_size != int(metadata["bytes"])
                or sha256(path) != metadata["sha256"]
            ):
                raise ValueError(f"bundled font identity mismatch: {filename}")
        for metadata in manifest["packages"].values():
            license_path = font_root / metadata["license"]
            if not license_path.is_file() or not license_path.read_text(
                encoding="utf-8", errors="replace"
            ).strip():
                raise ValueError(
                    f"font license is missing or empty: {metadata['license']}"
                )
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "name": "bundled-font-manifest",
            "status": "failed",
            "error": _bounded(str(exc)),
        }
    return {
        "name": "bundled-font-manifest",
        "status": "passed",
        "path": str(FONT_MANIFEST.resolve()),
        "sha256": sha256(FONT_MANIFEST),
        "fonts": len(manifest["fonts"]),
        "licenses": sorted(
            metadata["license"] for metadata in manifest["packages"].values()
        ),
    }


def verification_workspace() -> Path:
    configured = os.environ.get("BALOOT_VERIFY_WORKSPACE")
    return Path(configured).resolve() if configured else ROOT


def disk_check(path: Path | None = None) -> dict[str, Any]:
    path = path or verification_workspace()
    usage = shutil.disk_usage(path)
    passed = usage.free >= MIN_FREE_DISK_BYTES
    return {
        "name": "workspace-disk",
        "status": "passed" if passed else "failed",
        "path": str(path.resolve()),
        "freeBytes": usage.free,
        "requiredFreeBytes": MIN_FREE_DISK_BYTES,
        "error": (
            None
            if passed
            else "insufficient free disk space for two clean builds and verifications"
        ),
    }


def package_manifest() -> list[dict[str, str]]:
    if not shutil.which("dpkg-query"):
        return []
    packages = (
        "build-essential",
        "cmake",
        "ffmpeg",
        "fonts-dejavu-core",
        "fonts-noto-core",
        "fonts-noto-extra",
        "golang-go",
        "node-typescript",
        "nodejs",
        "ninja-build",
    )
    results: list[dict[str, str]] = []
    for package in packages:
        process = subprocess.run(
            ["dpkg-query", "-W", "-f=${Version}", package],
            text=True,
            capture_output=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
            check=False,
        )
        if process.returncode == 0:
            results.append({"name": package, "version": _bounded(process.stdout)})
    return results


def build_report() -> dict[str, Any]:
    started = time.perf_counter()
    tools = [command_check(name, command) for name, command in TOOLS.items()]
    packages = [package_check(name) for name in PYTHON_PACKAGES]
    fonts = [
        font_check(name, filename, sample)
        for name, (filename, sample) in FONT_REQUIREMENTS.items()
    ]
    platform_checks = [font_manifest_check(), disk_check()]
    raqm = bool(features.check_feature("raqm"))
    failures = [
        {"category": category, "name": item["name"], "error": item.get("error")}
        for category, items in (
            ("tool", tools),
            ("pythonPackage", packages),
            ("font", fonts),
            ("platform", platform_checks),
        )
        for item in items
        if item["status"] != "passed"
    ]
    if not raqm:
        failures.append(
            {
                "category": "pillowFeature",
                "name": "raqm",
                "error": "Pillow was built without RAQM complex-text support",
            }
        )
    if sys.version_info < (3, 12):
        failures.append(
            {
                "category": "runtime",
                "name": "python",
                "error": "Python 3.12 or newer is required",
            }
        )
    return {
        "schemaVersion": 1,
        "project": "baloot-platform",
        "release": "20.0.0",
        "generatedAt": utc_now(),
        "status": "passed" if not failures else "failed",
        "durationSeconds": round(time.perf_counter() - started, 3),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.splitlines()[0],
            "pythonExecutable": sys.executable,
            "sourceDateEpoch": os.environ.get(
                "SOURCE_DATE_EPOCH", "315532800"
            ),
            "verifierImageDigest": os.environ.get(
                "BALOOT_VERIFY_IMAGE_DIGEST"
            ),
        },
        "tools": tools,
        "pythonPackages": packages,
        "fonts": fonts,
        "platformChecks": platform_checks,
        "pillowFeatures": {"raqm": raqm},
        "systemPackages": package_manifest(),
        "failures": failures,
        "policy": {
            "networkInstallationDuringVerification": False,
            "commandTimeoutSeconds": COMMAND_TIMEOUT_SECONDS,
            "capturedOutputLimitBytes": OUTPUT_LIMIT,
            "minimumFreeDiskBytes": MIN_FREE_DISK_BYTES,
            "verificationWorkspace": str(verification_workspace()),
            "supportedVerifierImageDigest": os.environ.get(
                "BALOOT_VERIFY_IMAGE_DIGEST"
            ),
            "bundledFontsRequired": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_REPORT))
    arguments = parser.parse_args()
    report = build_report()
    output = Path(arguments.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
