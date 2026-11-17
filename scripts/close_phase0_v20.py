#!/usr/bin/env python3
"""Close the isolated V20 baseline with repeatable exact-build evidence."""
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
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATE_EPOCH = 315532800
BASELINE_VERSION = "20.0.0"
WHEEL_NAME = f"baloot_platform-{BASELINE_VERSION}-py3-none-any.whl"


def utc_now() -> str:
    return datetime.fromtimestamp(
        SOURCE_DATE_EPOCH, timezone.utc
    ).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path, base: Path | None = None) -> dict[str, Any]:
    return {
        "path": (
            path.relative_to(base).as_posix()
            if base is not None and path.is_relative_to(base)
            else path.name
        ),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def _git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(
        f"blob {len(data)}\0".encode("ascii") + data,
        usedforsecurity=False,
    ).hexdigest()


def validate_git_tree(
    source_root: Path,
    repository: Path,
    commit: str,
    expected_tree: str,
) -> dict[str, Any]:
    git = [
        "git",
        "-c",
        f"safe.directory={repository}",
        "-C",
        str(repository),
    ]
    resolved_commit = subprocess.run(
        [*git, "rev-parse", f"{commit}^{{commit}}"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout.strip()
    resolved_tree = subprocess.run(
        [*git, "rev-parse", f"{resolved_commit}^{{tree}}"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout.strip()
    if resolved_commit != commit or resolved_tree != expected_tree:
        raise RuntimeError("declared baseline commit or tree does not resolve exactly")
    listing = subprocess.run(
        [*git, "ls-tree", "-rz", "--full-tree", resolved_commit],
        capture_output=True,
        timeout=60,
        check=True,
    ).stdout
    expected: dict[str, str] = {}
    for entry in listing.split(b"\0"):
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        mode, kind, blob = metadata.decode("ascii").split(" ", 2)
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise RuntimeError(
                f"unsupported baseline Git entry: {mode} {kind} "
                f"{raw_path.decode('utf-8', 'replace')}"
            )
        expected[raw_path.decode("utf-8")] = blob
    actual = {
        path.relative_to(source_root).as_posix(): path
        for path in source_root.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(source_root).parts
    }
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    mismatched = sorted(
        name
        for name in set(expected) & set(actual)
        if _git_blob_sha1(actual[name]) != expected[name]
    )
    if missing or extra or mismatched:
        raise RuntimeError(
            json.dumps(
                {
                    "reason": "baseline source does not match declared Git tree",
                    "missing": missing[:100],
                    "extra": extra[:100],
                    "mismatched": mismatched[:100],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return {
        "gitCommit": resolved_commit,
        "gitTree": resolved_tree,
        "gitTreeFiles": len(expected),
        "gitTreeValidation": "passed",
    }


def source_identity(
    immutable_git_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "scripts"))
    import package_release_v20 as package

    files = sorted(
        (path for path in ROOT.rglob("*") if package.include(path)),
        key=package.relative_name,
    )
    rows = [
        f"{package.relative_name(path)}\0{path.stat().st_size}\0{sha256(path)}\n"
        for path in files
    ]
    digest = hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()
    identity: dict[str, Any] = {
        "kind": "canonical-packaged-tree",
        "sha256": digest,
        "files": len(files),
    }
    if immutable_git_identity is not None:
        identity.update(immutable_git_identity)
        return identity
    git = ["git", "-c", f"safe.directory={ROOT}", "-C", str(ROOT)]
    try:
        commit = subprocess.run(
            [*git, "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        tree = subprocess.run(
            [*git, "rev-parse", "HEAD^{tree}"],
            text=True,
            capture_output=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        identity.update({"gitCommit": commit, "gitTree": tree})
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        identity.update(
            {
                "gitCommit": None,
                "gitTree": None,
                "reasonGitUnavailable": str(exc)[:1000],
            }
        )
    return identity


def next_attempt(output_root: Path, source_hash: str) -> Path:
    run_root = output_root / source_hash[:16]
    run_root.mkdir(parents=True, exist_ok=True)
    numbers = [
        int(path.name.split("-", 1)[1])
        for path in run_root.glob("attempt-*")
        if path.is_dir() and path.name.split("-", 1)[1].isdigit()
    ]
    attempt = run_root / f"attempt-{max(numbers, default=0) + 1:03d}"
    attempt.mkdir()
    return attempt


def run(
    name: str,
    command: list[str],
    *,
    cwd: Path,
    attempt: Path,
    timeout: int,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(cwd / "src"),
            "BALOOT_ROOT": str(cwd),
            "BALOOT_ENV": "test",
            "SOURCE_DATE_EPOCH": str(SOURCE_DATE_EPOCH),
            "PYTHONHASHSEED": "0",
        }
    )
    if extra_env:
        environment.update(extra_env)
    started = time.perf_counter()
    process = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    log = attempt / f"{name}.log"
    log.write_text(
        process.stdout + ("\n" if process.stdout and process.stderr else "") + process.stderr,
        encoding="utf-8",
        newline="\n",
    )
    result = {
        "name": name,
        "status": "passed" if process.returncode == 0 else "failed",
        "returnCode": process.returncode,
        "durationSeconds": round(time.perf_counter() - started, 3),
        "command": command,
        "log": log.name,
        "outputTail": (process.stdout + "\n" + process.stderr)[-20000:],
    }
    if process.returncode:
        raise RuntimeError(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def dependencies() -> list[dict[str, str]]:
    names = (
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
    result = []
    for name in names:
        try:
            result.append({"name": name, "version": importlib.metadata.version(name)})
        except importlib.metadata.PackageNotFoundError:
            result.append({"name": name, "version": "unavailable"})
    return result


def main() -> None:
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "release/phase0-v20"),
    )
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--git-repository", type=Path)
    parser.add_argument("--baseline-commit")
    parser.add_argument("--baseline-tree")
    arguments = parser.parse_args()
    if arguments.source_root is not None:
        ROOT = arguments.source_root.resolve()
    identity_arguments = (
        arguments.git_repository,
        arguments.baseline_commit,
        arguments.baseline_tree,
    )
    if any(identity_arguments) and not all(identity_arguments):
        raise SystemExit(
            "--git-repository, --baseline-commit, and --baseline-tree "
            "must be provided together"
        )
    package_version = str(
        tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
            "project"
        ]["version"]
    )
    if package_version != BASELINE_VERSION:
        raise SystemExit(
            f"baseline source must declare {BASELINE_VERSION}, got {package_version}"
        )
    immutable_git_identity = None
    if all(identity_arguments):
        immutable_git_identity = validate_git_tree(
            ROOT,
            arguments.git_repository.resolve(),
            str(arguments.baseline_commit),
            str(arguments.baseline_tree),
        )
    output_root = Path(arguments.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    bootstrap_identity = source_identity(immutable_git_identity)
    attempt = next_attempt(output_root, bootstrap_identity["sha256"])
    checks: list[dict[str, Any]] = []
    ledger: dict[str, Any] = {
        "schemaVersion": 1,
        "project": "baloot-platform",
        "version": "20.0.0",
        "phase": "isolated V20 baseline closure",
        "generatedAt": utc_now(),
        "status": "running",
        "sourceDateEpoch": SOURCE_DATE_EPOCH,
        "bootstrapSourceIdentity": bootstrap_identity,
        "environment": {
            "os": platform.platform(),
            "python": sys.version.splitlines()[0],
            "pythonExecutable": sys.executable,
            "cpu": platform.processor() or platform.machine(),
            "machine": platform.machine(),
            "logicalCpuCount": os.cpu_count(),
            "sourceStorage": shutil.disk_usage(ROOT)._asdict(),
            "workspaceStorage": shutil.disk_usage(output_root)._asdict(),
            "verificationWorkspace": str(output_root),
            "verifierImageDigest": os.environ.get("BALOOT_VERIFY_IMAGE_DIGEST"),
        },
        "dependencies": dependencies(),
        "checks": checks,
        "artifacts": {},
        "proofBoundaries": {
            "officialRulesCertification": False,
            "humanPremiumAcceptance": False,
            "deploymentProof": False,
            "scope": "V20 reproducible baseline only",
        },
    }
    ledger_path = attempt / "baseline-ledger-v20.json"

    def persist() -> None:
        ledger_path.write_text(
            json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    persist()
    try:
        checks.append(
            run(
                "source-verification",
                [sys.executable, "scripts/verify_release_v20.py"],
                cwd=ROOT,
                attempt=attempt,
                timeout=7200,
            )
        )
        frozen_identity = source_identity(immutable_git_identity)
        ledger["frozenSourceIdentity"] = frozen_identity

        retained_wheels = [
            ROOT / "release/reproducibility-build-a.whl",
            ROOT / "release/reproducibility-build-b.whl",
        ]
        if not all(path.is_file() for path in retained_wheels):
            raise RuntimeError("two retained clean wheel builds are unavailable")
        wheel_identities = [file_identity(path) for path in retained_wheels]
        if wheel_identities[0]["sha256"] != wheel_identities[1]["sha256"]:
            raise RuntimeError("retained clean wheel builds are not byte-identical")
        for index, path in enumerate(retained_wheels, start=1):
            target = attempt / f"wheel-build-{index}" / WHEEL_NAME
            target.parent.mkdir()
            shutil.copyfile(path, target)
        ledger["artifacts"]["wheels"] = wheel_identities

        archives = [
            attempt / "repository-build-1.zip",
            attempt / "repository-build-2.zip",
        ]
        for index, archive in enumerate(archives, start=1):
            checks.append(
                run(
                    f"repository-build-{index}",
                    [
                        sys.executable,
                        "scripts/package_release_v20.py",
                        "--output",
                        str(archive),
                    ],
                    cwd=ROOT,
                    attempt=attempt,
                    timeout=1800,
                )
            )
        archive_identities = [file_identity(path, attempt) for path in archives]
        if archive_identities[0]["sha256"] != archive_identities[1]["sha256"]:
            raise RuntimeError("two repository builds are not byte-identical")
        ledger["artifacts"]["repositoryZips"] = archive_identities

        for index, archive in enumerate(archives, start=1):
            workspace = attempt / f"clean-workspace-{index}"
            workspace.mkdir()
            report = attempt / f"exact-verification-{index}.json"
            checks.append(
                run(
                    f"exact-verification-{index}",
                    [
                        sys.executable,
                        "scripts/verify_archive_v20.py",
                        str(archive),
                        "--workspace",
                        str(workspace),
                        "--output",
                        str(report),
                    ],
                    cwd=ROOT,
                    attempt=attempt,
                    timeout=10800,
                )
            )
            exact = json.loads(report.read_text(encoding="utf-8"))
            if exact.get("status") != "passed":
                raise RuntimeError(f"exact verification {index} did not pass")
        ledger["artifacts"]["exactVerificationReports"] = [
            file_identity(attempt / f"exact-verification-{index}.json", attempt)
            for index in (1, 2)
        ]

        inventory = json.loads(
            (ROOT / "release/test-inventory-v20.json").read_text(encoding="utf-8")
        )
        pytest_report = json.loads(
            (ROOT / "release/pytest-lanes-v20.json").read_text(encoding="utf-8")
        )
        if (
            inventory["testFiles"] != pytest_report["testFiles"]
            or inventory["testCases"] != pytest_report["testsPassed"]
        ):
            raise RuntimeError("collected and executed test inventories do not match")
        ledger["testInventory"] = {
            "testFiles": inventory["testFiles"],
            "testCases": inventory["testCases"],
            "inventorySha256": inventory["inventorySha256"],
            "executedTestsPassed": pytest_report["testsPassed"],
        }
        ledger["sourceManifest"] = file_identity(
            ROOT / "release/source-manifest-v20.json", ROOT
        )
        ledger["artifacts"]["baselineLedger"] = {
            "path": ledger_path.name,
        }
        ledger["status"] = "passed"
        persist()
        print(ledger_path)
    except Exception as exc:
        ledger["status"] = "failed"
        ledger["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc)[:20000],
        }
        persist()
        raise


if __name__ == "__main__":
    main()
