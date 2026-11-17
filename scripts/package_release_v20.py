#!/usr/bin/env python3
"""Build the deterministic, cross-platform V20 repository archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import stat
import tempfile
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = str(
    tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["version"]
)
PREFIX = f"baloot-platform-{VERSION}"
OUTPUT_PREFIX = "baloot-platform-v20-premium-outcome-flip"
MANIFEST_REL = Path("release/source-manifest-v20.json")
EXCLUDED_DIRS = {
    "private-inputs",
    ".local-state",
    ".codex-local",
    ".git",
    ".pytest_cache",
    ".runtime",
    ".release-runtime",
    "build",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    ".venv",
    "venv",
    "baloot_platform.egg-info",
    "studio-js",
    "studio-js-v6",
    "studio-js-v7.1",
    "studio-js-v8",
    "studio-js-v9",
    "studio-js-v20",
    "test-lanes",
    "test-lanes-v7.1",
    "test-lanes-v8",
    "test-lanes-v9",
    "test-lanes-v20",
    "archive",
    "work",
}
EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pkl",
    ".sqlite",
    ".sqlite3",
    ".sqlite-shm",
    ".sqlite-wal",
    ".sqlite-journal",
    ".sqlite3-shm",
    ".sqlite3-wal",
    ".sqlite3-journal",
    ".db-shm",
    ".db-wal",
    ".db-journal",
    ".pid",
    ".log",
    ".tmp",
}
ALLOWED_RELEASE_FILES = {
    "verification-report-v20.json",
    "verification-report.json",
    "property-check-v20.json",
    "property-check.json",
    "pytest-lanes-v20.json",
    "media-probe-v20.json",
    "audio-loudness-v20.json",
    "cleanroom-report-v20.json",
    "cleanroom-report.json",
    "sbom.cdx.json",
    "license-inventory.json",
    "selfplay-league-v20.json",
    "external-validation-boundaries-v20.json",
    "hitl-review-probe-v20.json",
    "rules-certification-candidate-v20.json",
    "intelligence-benchmark-v20.json",
    "slo-evaluation-v20.json",
    "platform-v20-evidence.json",
    "configuration-registry-v20.json",
    "plugin-host-v20.json",
    "performance-benchmark-v20.json",
    "visual-qa-v20.json",
    "test-inventory-v20.json",
    "wheel-reproducibility-v20.json",
    "verifier-preflight-v20.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_name(path: Path) -> str:
    """Return one canonical archive/manifest path on every host OS."""
    return path.relative_to(ROOT).as_posix()


def include(path: Path) -> bool:
    if not path.is_file():
        return False
    relative = path.relative_to(ROOT)
    if any(
        part in EXCLUDED_DIRS or part.endswith("-runtime")
        for part in relative.parts
    ):
        return False
    if path.suffix in EXCLUDED_SUFFIXES or relative == MANIFEST_REL:
        return False
    if relative.parts and relative.parts[0] == "outputs":
        return path.name.startswith(OUTPUT_PREFIX) or path.name.startswith(
            "baloot-platform-v20-hitl-review-clip"
        )
    if relative.parts and relative.parts[0] == "dist":
        return path.name.startswith(f"baloot_platform-{VERSION}-")
    if relative.parts and relative.parts[0] == "release":
        if len(relative.parts) > 1 and relative.parts[1] == "models":
            return True
        if (
            path.name not in ALLOWED_RELEASE_FILES
            and not path.name.startswith("source-manifest-v20")
        ):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(
            Path(tempfile.gettempdir())
            / f"{PREFIX}-software-complete.zip"
        ),
    )
    arguments = parser.parse_args()
    output = Path(arguments.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)

    files = sorted(
        (path for path in ROOT.rglob("*") if include(path)),
        key=relative_name,
    )
    manifest = {
        "schemaVersion": 1,
        "project": "baloot-platform",
        "version": VERSION,
        "archivePrefix": PREFIX,
        "terminology": {
            "matchAr": "صكّة",
            "roundAr": "لفة",
            "trickAr": "أكلة",
        },
        "fileCount": len(files),
        "files": {
            relative_name(path): {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in files
        },
    }
    manifest_path = ROOT / MANIFEST_REL
    manifest_path.parent.mkdir(exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    files.append(manifest_path)
    files.sort(key=relative_name)

    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in files:
            info = zipfile.ZipInfo(
                f"{PREFIX}/{relative_name(path)}",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            mode = 0o755 if path.stat().st_mode & stat.S_IXUSR else 0o644
            info.external_attr = (mode & 0xFFFF) << 16
            archive.writestr(
                info,
                path.read_bytes(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    checksum = output.with_suffix(output.suffix + ".sha256")
    checksum.write_text(
        f"{sha256(output)}  {output.name}\n", encoding="utf-8"
    )
    print(output)
    print(checksum)


if __name__ == "__main__":
    main()
