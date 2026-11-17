#!/usr/bin/env python3
"""Build the deterministic current BalootX repository archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import sys
import tempfile
import tomllib
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from baloot_platform.filesystem import atomic_write_text  # noqa: E402

try:
    from scripts.generate_performance_evidence import git_identity
    from scripts.generate_security_report import verify_packaged as verify_security
except ModuleNotFoundError:
    from generate_performance_evidence import git_identity
    from generate_security_report import verify_packaged as verify_security

VERSION = str(
    tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["version"]
)
PREFIX = f"baloot-platform-{VERSION}"
MANIFEST_RELATIVE = Path("release/source-manifest.json")
SECURITY_REPORT_RELATIVE = Path("release/security-report.json")
CURRENT_OUTPUT_DIRECTORIES = {
    "ortools-v4-five-video-evidence",
    "professional-video-flagship",
    "professional-video-decision",
    "professional-video-prototypes",
    "research-scenario-v4-goldens",
    "remotion-plugin-parity",
    "scene-preview",
}
EXCLUDED_DIRECTORIES = {
    "private-inputs", ".codex-local", ".local-state",
    ".codex-tmp",
    ".playwright-cli",
    ".git",
    ".pytest_cache",
    ".release-runtime",
    ".ruff_cache",
    ".runtime",
    ".venv",
    "__pycache__",
    "baloot_platform.egg-info",
    "build",
    "node_modules",
    "output",
    "venv",
    "work",
}
EXCLUDED_SUFFIXES = {
    ".log",
    ".pid",
    ".pkl",
    ".pyc",
    ".pyo",
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
    ".tmp",
}
EXCLUDED_PATH_PREFIXES = (
    "assets/cards/packs/poker_double_backmarked_baloot_32/",
    "src/baloot_platform/resources/assets/cards/packs/poker_double_backmarked_baloot_32/",
)
EXCLUDED_RELEASE_EVIDENCE_PREFIXES = (
    "live-postgres-research-",
    "repository-archive-verification",
    "verifier-container-",
    "verifier-image-build-",
    "verifier-preflight",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_name(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def include(path: Path) -> bool:
    if not path.is_file():
        return False
    relative = path.relative_to(ROOT)
    relative_posix = relative.as_posix()
    if any(
        part in EXCLUDED_DIRECTORIES or part.endswith("-runtime")
        for part in relative.parts
    ):
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES or relative == MANIFEST_RELATIVE:
        return False
    if any(relative_posix.startswith(prefix) for prefix in EXCLUDED_PATH_PREFIXES):
        return False
    if relative.parts[0] == "outputs":
        return len(relative.parts) > 1 and relative.parts[1] in CURRENT_OUTPUT_DIRECTORIES
    if relative.parts[0] == "dist":
        return path.name == f"baloot_platform-{VERSION}-py3-none-any.whl"
    if relative.parts[0] == "release":
        if path.suffix.lower() in {".zip", ".whl"} or path.name.endswith(".sha256"):
            return False
        if any(
            path.name.startswith(prefix) for prefix in EXCLUDED_RELEASE_EVIDENCE_PREFIXES
        ):
            return False
        if re.fullmatch(r"source-manifest-v\d+\.json", path.name):
            return False
    return True


def build(output: Path) -> Path:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    security_path = ROOT / SECURITY_REPORT_RELATIVE
    if not security_path.is_file():
        raise FileNotFoundError(
            "repository package requires release/security-report.json"
        )
    try:
        security = json.loads(security_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("repository package security report is invalid JSON") from exc
    source_identity = git_identity(ROOT)
    verify_security(
        security,
        root=ROOT,
        source_identity=source_identity,
    )
    files = sorted(
        (path for path in ROOT.rglob("*") if include(path)),
        key=relative_name,
    )
    manifest = {
        "schemaVersion": 1,
        "project": "baloot-platform",
        "version": VERSION,
        "archivePrefix": PREFIX,
        "source": source_identity,
        "fileCount": len(files),
        "files": {
            relative_name(path): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        },
    }
    manifest_path = ROOT / MANIFEST_RELATIVE
    atomic_write_text(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
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
    atomic_write_text(
        output.with_suffix(output.suffix + ".sha256"),
        f"{sha256(output)}  {output.name}\n",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(Path(tempfile.gettempdir()) / f"{PREFIX}-release-candidate.zip"),
    )
    output = build(Path(parser.parse_args().output))
    print(output)
    print(output.with_suffix(output.suffix + ".sha256"))


if __name__ == "__main__":
    main()
