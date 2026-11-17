#!/usr/bin/env python3
"""Verify an exact V20 repository archive in a portable clean workspace."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import tomllib
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

OUTPUT_PREFIX = "baloot-platform-v20-premium-outcome-flip"
MANIFEST = Path("release/source-manifest-v20.json")
WORKSPACE_ENV = "BALOOT_VERIFY_WORKSPACE"
MAX_ARCHIVE_BYTES_ENV = "BALOOT_VERIFY_MAX_ARCHIVE_BYTES"
DEFAULT_MAX_ARCHIVE_BYTES = 4 * 1024 * 1024 * 1024


class ArchiveVerificationError(RuntimeError):
    """Raised when an archive or verifier workspace is unsafe."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_workspace_parent(cli_value: str | None, env: dict[str, str] | None = None) -> Path:
    """Resolve and create the parent used for the disposable clean workspace."""
    values = os.environ if env is None else env
    raw = cli_value or values.get(WORKSPACE_ENV) or tempfile.gettempdir()
    parent = Path(raw).expanduser()
    if not parent.is_absolute():
        parent = Path.cwd() / parent
    if parent.exists() and (not parent.is_dir() or parent.is_symlink()):
        raise ArchiveVerificationError(
            f"verifier workspace parent must be a real directory: {parent}"
        )
    try:
        parent.mkdir(parents=True, exist_ok=True)
        resolved = parent.resolve(strict=True)
    except OSError as exc:
        raise ArchiveVerificationError(
            f"unable to create verifier workspace parent {parent}: {exc}"
        ) from exc
    if not os.access(resolved, os.W_OK | os.X_OK):
        raise ArchiveVerificationError(
            f"verifier workspace parent is not writable: {resolved}"
        )
    return resolved


def _safe_member_path(info: zipfile.ZipInfo) -> PurePosixPath:
    # On Windows, zipfile normalizes backslashes in ``filename`` before this
    # verifier sees the entry. ``orig_filename`` preserves the central-
    # directory spelling and is therefore the security boundary.
    raw = getattr(info, "orig_filename", info.filename)
    if not raw or "\x00" in raw or "\\" in raw:
        raise ArchiveVerificationError(f"unsafe archive entry name: {raw!r}")
    member = PurePosixPath(raw)
    if member.is_absolute() or any(part in {"", ".", ".."} for part in member.parts):
        raise ArchiveVerificationError(f"unsafe archive entry path: {raw!r}")
    if member.parts and ":" in member.parts[0]:
        raise ArchiveVerificationError(f"drive-qualified archive entry: {raw!r}")
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    if unix_mode and stat.S_ISLNK(unix_mode):
        raise ArchiveVerificationError(f"symbolic-link archive entry is forbidden: {raw!r}")
    file_type = stat.S_IFMT(unix_mode)
    if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
        raise ArchiveVerificationError(
            f"special archive entry is forbidden: {raw!r}"
        )
    if info.flag_bits & 0x1:
        raise ArchiveVerificationError(
            f"encrypted archive entry is unsupported: {raw!r}"
        )
    return member


def validate_archive(
    archive: zipfile.ZipFile,
    *,
    max_uncompressed_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES,
) -> list[tuple[zipfile.ZipInfo, PurePosixPath]]:
    """Validate every entry before any archive content is written."""
    if max_uncompressed_bytes <= 0:
        raise ArchiveVerificationError("maximum archive size must be positive")
    validated: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
    seen: set[str] = set()
    total = 0
    for info in archive.infolist():
        member = _safe_member_path(info)
        normalized = "/".join(member.parts).rstrip("/").casefold()
        if normalized in seen:
            raise ArchiveVerificationError(
                f"duplicate or case-colliding archive entry: {info.filename!r}"
            )
        seen.add(normalized)
        total += int(info.file_size)
        if total > max_uncompressed_bytes:
            raise ArchiveVerificationError(
                f"archive expands beyond the configured {max_uncompressed_bytes} byte limit"
            )
        validated.append((info, member))
    if archive.testzip() is not None:
        raise ArchiveVerificationError("archive CRC validation failed")
    return validated


def extract_archive_safely(
    archive: zipfile.ZipFile,
    destination: Path,
    *,
    max_uncompressed_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES,
) -> None:
    """Extract validated regular files and directories without link traversal."""
    members = validate_archive(
        archive, max_uncompressed_bytes=max_uncompressed_bytes
    )
    root = destination.resolve(strict=True)
    for info, member in members:
        target = destination.joinpath(*member.parts)
        resolved = target.resolve(strict=False)
        if resolved != root and root not in resolved.parents:
            raise ArchiveVerificationError(
                f"archive entry escapes the clean workspace: {info.filename!r}"
            )
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info, "r") as source, target.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)


def run(
    name: str,
    command: list[str],
    *,
    cwd: Path,
    timeout: int = 7200,
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(cwd / "src"),
            "BALOOT_ROOT": str(cwd),
            "SOURCE_DATE_EPOCH": "315532800",
        }
    )
    started = time.perf_counter()
    process = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    result = {
        "name": name,
        "status": "passed" if process.returncode == 0 else "failed",
        "returnCode": process.returncode,
        "durationSeconds": round(time.perf_counter() - started, 3),
        "stdout": process.stdout[-20000:],
        "stderr": process.stderr[-20000:],
    }
    if process.returncode:
        raise RuntimeError(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--workspace",
        help=f"Parent for the disposable clean workspace; overrides {WORKSPACE_ENV}.",
    )
    parser.add_argument("--keep", action="store_true")
    arguments = parser.parse_args()

    archive_path = Path(arguments.archive).resolve()
    report_path = Path(arguments.output).resolve()
    generated = datetime.fromtimestamp(315532800, timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    parent: Path | None = None
    clean_root: Path | None = None
    preserve = bool(arguments.keep)
    checks: list[dict[str, Any]] = []
    try:
        workspace_parent = resolve_workspace_parent(arguments.workspace)
        parent = Path(
            tempfile.mkdtemp(prefix="baloot-v20-cleanroom-", dir=workspace_parent)
        )
        max_archive_bytes = int(
            os.environ.get(MAX_ARCHIVE_BYTES_ENV, str(DEFAULT_MAX_ARCHIVE_BYTES))
        )
        with zipfile.ZipFile(archive_path) as archive:
            extract_archive_safely(
                archive, parent, max_uncompressed_bytes=max_archive_bytes
            )
        roots = [path for path in parent.iterdir() if path.is_dir()]
        if len(roots) != 1:
            raise ArchiveVerificationError(
                f"archive must contain exactly one repository root; found {len(roots)}"
            )
        clean_root = roots[0]
        version = str(
            tomllib.loads(
                (clean_root / "pyproject.toml").read_text(encoding="utf-8")
            )["project"]["version"]
        )
        if version != "20.0.0":
            raise ArchiveVerificationError(
                f"unexpected repository version: {version!r}"
            )
        manifest = json.loads(
            (clean_root / MANIFEST).read_text(encoding="utf-8")
        )
        if manifest["version"] != version:
            raise ArchiveVerificationError("source manifest version mismatch")
        for relative, metadata in manifest["files"].items():
            path = clean_root / relative
            if (
                not path.is_file()
                or path.stat().st_size != metadata["bytes"]
                or sha256(path) != metadata["sha256"]
            ):
                raise ArchiveVerificationError(
                    f"source manifest mismatch: {relative}"
                )
        checks.append(
            {
                "name": "source-manifest",
                "status": "passed",
                "files": len(manifest["files"]),
            }
        )

        output_directory = clean_root / "outputs"
        before = {
            path.name: sha256(path)
            for path in output_directory.iterdir()
            if path.is_file()
            and (
                path.name.startswith(OUTPUT_PREFIX)
                or path.name.startswith("baloot-platform-v20-hitl-review-clip")
            )
        }
        before_wheels = {
            path.name: sha256(path)
            for path in (clean_root / "dist").glob("baloot_platform-20.0.0-*.whl")
        }
        inventory_path = clean_root / "release/test-inventory-v20.json"
        before_inventory = (
            json.loads(inventory_path.read_text(encoding="utf-8"))
            if inventory_path.is_file()
            else None
        )
        visual_paths = [
            clean_root / "release/visual-qa-v20.json",
            *sorted((clean_root / "generated/visual-qa-v20").glob("*")),
        ]
        before_visual = {
            str(path.relative_to(clean_root)): sha256(path)
            for path in visual_paths
            if path.is_file()
        }
        checks.append(
            run(
                "cleanroom-verification",
                [sys.executable, "scripts/verify_release_v20.py"],
                cwd=clean_root,
                timeout=7200,
            )
        )
        after_inventory = json.loads(
            inventory_path.read_text(encoding="utf-8")
        )
        if (
            before_inventory is None
            or before_inventory.get("inventorySha256")
            != after_inventory.get("inventorySha256")
            or before_inventory.get("nodeIds") != after_inventory.get("nodeIds")
        ):
            raise ArchiveVerificationError(
                "source and packaged pytest inventories do not match"
            )
        checks.append(
            {
                "name": "source-package-test-inventory",
                "status": "passed",
                "testFiles": after_inventory["testFiles"],
                "testCases": after_inventory["testCases"],
                "inventorySha256": after_inventory["inventorySha256"],
            }
        )
        after = {
            path.name: sha256(path)
            for path in output_directory.iterdir()
            if path.is_file()
            and (
                path.name.startswith(OUTPUT_PREFIX)
                or path.name.startswith("baloot-platform-v20-hitl-review-clip")
            )
        }
        if before != after:
            changed = {
                name: {"before": before.get(name), "after": after.get(name)}
                for name in sorted(set(before) | set(after))
                if before.get(name) != after.get(name)
            }
            raise ArchiveVerificationError(
                "rendered media is not byte-identical: "
                + json.dumps(changed, ensure_ascii=False, sort_keys=True)
            )
        checks.append(
            {
                "name": "byte-identical-render-artifacts",
                "status": "passed",
                "files": len(after),
                "sha256": after,
            }
        )
        after_visual = {
            str(path.relative_to(clean_root)): sha256(path)
            for path in visual_paths
            if path.is_file()
        }
        if before_visual != after_visual:
            changed = {
                name: {
                    "before": before_visual.get(name),
                    "after": after_visual.get(name),
                }
                for name in sorted(set(before_visual) | set(after_visual))
                if before_visual.get(name) != after_visual.get(name)
            }
            raise ArchiveVerificationError(
                "rendered visual-QA artifacts are not byte-identical: "
                + json.dumps(changed, ensure_ascii=False, sort_keys=True)
            )
        checks.append(
            {
                "name": "byte-identical-visual-qa-artifacts",
                "status": "passed",
                "files": len(after_visual),
                "sha256": after_visual,
            }
        )
        after_wheels = {
            path.name: sha256(path)
            for path in (clean_root / "dist").glob("baloot_platform-20.0.0-*.whl")
        }
        if before_wheels and before_wheels != after_wheels:
            changed = {
                name: {
                    "before": before_wheels.get(name),
                    "after": after_wheels.get(name),
                }
                for name in sorted(set(before_wheels) | set(after_wheels))
                if before_wheels.get(name) != after_wheels.get(name)
            }
            raise ArchiveVerificationError(
                "wheel rebuild is not deterministic: "
                + json.dumps(changed, ensure_ascii=False, sort_keys=True)
            )
        checks.append(
            {
                "name": "deterministic-wheel",
                "status": "passed",
                "sha256": after_wheels,
            }
        )
        report = {
            "schemaVersion": 2,
            "project": "baloot-platform",
            "version": version,
            "archive": archive_path.name,
            "archiveSha256": sha256(archive_path),
            "generatedAt": generated,
            "status": "passed",
            "checks": checks,
            "terminology": {
                "matchAr": "صكّة",
                "roundAr": "لفة",
                "trickAr": "أكلة",
            },
            "workspaceParent": str(workspace_parent),
            "workspaceSource": (
                "cli"
                if arguments.workspace
                else "environment"
                if os.environ.get(WORKSPACE_ENV)
                else "system-temp"
            ),
            "cleanroomRoot": str(clean_root) if preserve else None,
        }
        _write_report(report_path, report)
        print(report_path)
        if preserve:
            print(clean_root)
    except Exception as exc:
        failure = {
            "schemaVersion": 2,
            "project": "baloot-platform",
            "version": "20.0.0",
            "archive": archive_path.name,
            "archiveSha256": sha256(archive_path) if archive_path.is_file() else None,
            "generatedAt": generated,
            "status": "failed",
            "checks": checks,
            "error": {"type": type(exc).__name__, "message": str(exc)[:20000]},
            "cleanroomRoot": str(clean_root) if arguments.keep and clean_root else None,
        }
        _write_report(report_path, failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr)
        raise
    finally:
        if parent is not None and not preserve:
            shutil.rmtree(parent, ignore_errors=True)


if __name__ == "__main__":
    main()
