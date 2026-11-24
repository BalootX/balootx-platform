#!/usr/bin/env python3
"""Execute the current BalootX verifier-container admission gate."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

from baloot_platform.config import build_bundle
from baloot_platform.service.container import build_container
from baloot_platform.version import __version__


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "release/verifier-preflight.json"
REMOTION = ROOT / "src/baloot_platform/video/remotion"
FOUNDRY = ROOT / "src/baloot_platform/service/foundry_dashboard"
SOURCE_ONLY_TESTS = (
    "tests/test_performance_evidence.py",
    "tests/test_security_report.py",
    "tests/test_repository_structure.py::test_runtime_and_build_directories_are_not_tracked",
    "tests/test_repository_structure.py::test_workspace_root_is_the_repository_root",
)
ISOLATED_CONTAINER_TESTS = (
    "tests/test_exact_holdout_labels.py::test_exact_holdout_labels_replay_and_preserve_all_optimal_ties",
    "tests/test_search_performance.py::test_v21_search_benchmark_is_reproducible_and_state_safe",
)
CONTAINER_SUITE_SHARD_COUNT = 3


class CheckFailure(RuntimeError):
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        super().__init__(json.dumps(result, ensure_ascii=False, sort_keys=True))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(
    name: str,
    command: list[str],
    *,
    cwd: Path = ROOT,
    timeout: int = 1200,
) -> dict[str, Any]:
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT / "src"),
            "BALOOT_ROOT": str(ROOT),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=os.name != "nt",
    )
    timed_out = False
    cleanup_errors: list[str] = []
    cleanup_complete = True
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        tree_terminated = False
        if os.name == "nt":
            try:
                cleanup = subprocess.run(
                    ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                )
                tree_terminated = cleanup.returncode == 0
                if not tree_terminated:
                    cleanup_errors.append(
                        f"taskkill_return_code:{cleanup.returncode}"
                    )
            except (OSError, subprocess.TimeoutExpired) as exc:
                cleanup_errors.append(f"taskkill:{type(exc).__name__}")
                process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
                tree_terminated = True
            except OSError as exc:
                cleanup_errors.append(f"killpg:{type(exc).__name__}")
                process.kill()
        try:
            stdout, stderr = process.communicate(timeout=30)
        except subprocess.TimeoutExpired as exc:
            cleanup_errors.append("communicate:TimeoutExpired")
            process.kill()
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
        cleanup_complete = (
            tree_terminated and process.poll() is not None and not cleanup_errors
        )
    result = {
        "name": name,
        "status": "passed" if process.returncode == 0 and not timed_out else "failed",
        "returnCode": 124 if timed_out else process.returncode,
        "timedOut": timed_out,
        "durationSeconds": round(time.perf_counter() - started, 3),
        "stdout": stdout[-20000:],
        "stderr": stderr[-20000:],
    }
    if timed_out:
        result["cleanupComplete"] = cleanup_complete
        result["cleanupErrors"] = cleanup_errors
    if result["status"] != "passed":
        raise CheckFailure(result)
    return result


def authority_check() -> dict[str, Any]:
    project_version = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["version"]
    if __version__ != project_version:
        raise RuntimeError(f"unexpected package version: {__version__}")
    bundle = build_bundle(ROOT)
    container = build_container(root=ROOT)
    try:
        video_plugins = [
            item
            for item in container.registry.list()
            if item["kind"] == "video_renderer"
        ]
    finally:
        container.shutdown()
    if [item["id"] for item in video_plugins] != ["premium_video"]:
        raise RuntimeError("premium_video must be the sole video renderer")
    package_lock = REMOTION / "package-lock.json"
    if not package_lock.is_file() or not (REMOTION / "node_modules").is_dir():
        raise RuntimeError("the exact Remotion runtime is unavailable")
    return {
        "name": "authorities",
        "status": "passed",
        "version": __version__,
        "bundleSha256": bundle.sha256,
        "videoPlugins": [item["id"] for item in video_plugins],
        "remotionLockSha256": sha256(package_lock),
    }


def container_suite_commands() -> tuple[list[str], ...]:
    excluded_files = {
        selection for selection in SOURCE_ONLY_TESTS if "::" not in selection
    }
    test_files = tuple(
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in sorted((ROOT / "tests").glob("test_*.py"))
        if str(path.relative_to(ROOT)).replace("\\", "/") not in excluded_files
    )
    shards = tuple(
        test_files[index::CONTAINER_SUITE_SHARD_COUNT]
        for index in range(CONTAINER_SUITE_SHARD_COUNT)
    )
    commands: list[list[str]] = []
    deselections = tuple(
        selection
        for selection in (*SOURCE_ONLY_TESTS, *ISOLATED_CONTAINER_TESTS)
        if "::" in selection
    )
    for files in shards:
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-vv",
            "--override-ini=addopts=",
            "--durations=25",
            *files,
        ]
        selected_files = set(files)
        command.extend(
            f"--deselect={selection}"
            for selection in deselections
            if selection.split("::", 1)[0] in selected_files
        )
        commands.append(command)
    return tuple(commands)


def isolated_container_test_command(selection: str) -> list[str]:
    if selection not in ISOLATED_CONTAINER_TESTS:
        raise ValueError("isolated container test is not governed")
    return [sys.executable, "-m", "pytest", "-q", selection, "--durations=5"]


def container_suite_timeout_seconds() -> int:
    raw = os.environ.get("BALOOT_VERIFIER_TEST_TIMEOUT_SECONDS")
    if raw is None:
        raise RuntimeError("BALOOT_VERIFIER_TEST_TIMEOUT_SECONDS is required")
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            "BALOOT_VERIFIER_TEST_TIMEOUT_SECONDS must be an integer"
        ) from exc
    if not 300 <= value <= 3600:
        raise RuntimeError(
            "BALOOT_VERIFIER_TEST_TIMEOUT_SECONDS must be between 300 and 3600"
        )
    return value


def main() -> None:
    checks: list[dict[str, Any]] = []
    try:
        suite_timeout = container_suite_timeout_seconds()
        checks.append(authority_check())
        checks.append(run("node", ["node", "--version"], timeout=30))
        checks.append(
            run(
                "remotion-typecheck",
                ["node", "node_modules/typescript/bin/tsc", "--noEmit"],
                cwd=REMOTION,
                timeout=300,
            )
        )
        checks.append(
            run(
                "foundry-check",
                ["node", "scripts/check-generated-api.mjs"],
                cwd=FOUNDRY,
                timeout=300,
            )
        )
        checks.append(
            run(
                "foundry-typecheck",
                ["node", "node_modules/typescript/bin/tsc", "--noEmit"],
                cwd=FOUNDRY,
                timeout=300,
            )
        )
        checks.append(
            run(
                "foundry-research-tests",
                [
                    "node",
                    "node_modules/vitest/vitest.mjs",
                    "run",
                    "--config",
                    "vitest.config.mts",
                ],
                cwd=FOUNDRY,
                timeout=300,
            )
        )
        checks.append(
            run(
                "native-tests",
                ["baloot_native_tests"],
                timeout=120,
            )
        )
        checks.append(
            {
                "name": "source-only-test-boundary",
                "status": "passed",
                "excluded": list(SOURCE_ONLY_TESTS),
                "reason": "These checks require Git history and already pass in the source-tree release gate.",
            }
        )
        checks.append(
            {
                "name": "container-suite-operational-timeout",
                "status": "passed",
                "seconds": suite_timeout,
                "minimum": 300,
                "maximum": 3600,
                "scope": "per_shard",
                "shardCount": CONTAINER_SUITE_SHARD_COUNT,
            }
        )
        for selection in ISOLATED_CONTAINER_TESTS:
            checks.append(
                run(
                    "isolated-container-test",
                    isolated_container_test_command(selection),
                    timeout=suite_timeout,
                )
                | {"selection": selection}
            )
        shard_failures: list[dict[str, Any]] = []
        suite_commands = container_suite_commands()
        for index, command in enumerate(suite_commands, start=1):
            try:
                checks.append(
                    run(
                        f"container-python-suite-{index}-of-{len(suite_commands)}",
                        command,
                        timeout=suite_timeout,
                    )
                )
            except CheckFailure as exc:
                checks.append(exc.result)
                shard_failures.append(exc.result)
        if shard_failures:
            raise RuntimeError(
                "container Python suite shards failed: "
                + json.dumps(shard_failures, ensure_ascii=False, sort_keys=True)
            )
        report = {
            "schemaVersion": 1,
            "release": __version__,
            "status": "passed",
            "checks": checks,
            "proofBoundary": (
                "Container admission covers current package authorities, Node/Remotion "
                "and Foundry contract/research checks, native parity, explicitly isolated timing-sensitive tests, "
                "and all remaining tests that do not require Git history. "
                "Exact encoded media is covered by separate byte-bound media and archive gates."
            ),
        }
    except Exception as exc:
        if isinstance(exc, CheckFailure) and exc.result not in checks:
            checks.append(exc.result)
        report = {
            "schemaVersion": 1,
            "release": __version__,
            "status": "failed",
            "checks": checks,
            "error": {"type": type(exc).__name__, "message": str(exc)[:20000]},
        }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
