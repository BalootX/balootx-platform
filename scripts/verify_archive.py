#!/usr/bin/env python3
"""Verify the exact current repository archive in a clean workspace."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import signal
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


MAX_UNCOMPRESSED_BYTES = 4 * 1024 * 1024 * 1024
ARCHIVE_PYTHON_FILE_TIMEOUT_SECONDS = 1200
ARCHIVE_LONG_PYTHON_FILE_TIMEOUT_SECONDS = 3600
ARCHIVE_NODE_TIMEOUT_SECONDS = 600
ARCHIVE_NATIVE_TIMEOUT_SECONDS = 600
ARCHIVE_LONG_PYTHON_FILES = {
    "test_research_audit_trace_service.py",
    "test_research_foundry.py",
    "test_research_sharding.py",
    "test_research_strategic_play.py",
}


class ArchiveVerificationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_packaged_security(
    clean_root: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    try:
        from scripts.generate_security_report import verify_packaged
    except ModuleNotFoundError:
        from generate_security_report import verify_packaged

    relative = "release/security-report.json"
    files = manifest.get("files")
    source = manifest.get("source")
    if not isinstance(files, dict) or relative not in files:
        raise ArchiveVerificationError(
            "archive source manifest omits the required security report"
        )
    if not isinstance(source, dict):
        raise ArchiveVerificationError(
            "archive source manifest omits its canonical source identity"
        )
    path = clean_root / relative
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("security report must be a JSON object")
        verify_packaged(value, root=clean_root, source_identity=source)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ArchiveVerificationError(
            "archive security report is stale or invalid"
        ) from exc
    return {
        "name": "security-evidence",
        "status": "passed",
        "binding": source.get("binding"),
        "evidenceSha256": value.get("evidenceSha256"),
    }


def _verify_packaged_wheel(
    clean_root: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    relative = "release/wheel-reproducibility.json"
    files = manifest.get("files")
    source = manifest.get("source")
    if not isinstance(files, dict) or relative not in files:
        raise ArchiveVerificationError(
            "archive source manifest omits wheel reproducibility evidence"
        )
    try:
        value = json.loads((clean_root / relative).read_text(encoding="utf-8"))
        wheel_relative = value["wheel"]
        wheel_path = PurePosixPath(wheel_relative)
        builds = value["builds"]
        if (
            not isinstance(source, dict)
            or value.get("source") != source
            or value.get("status") != "passed"
            or value.get("byteIdentical") is not True
            or not isinstance(wheel_relative, str)
            or wheel_path.is_absolute()
            or ".." in wheel_path.parts
            or wheel_path.as_posix() != wheel_relative
            or wheel_relative not in files
            or not isinstance(builds, list)
            or len(builds) != 2
            or builds[0] != builds[1]
        ):
            raise ValueError("wheel evidence contract drifted")
        wheel = clean_root.joinpath(*wheel_path.parts)
        if (
            not wheel.is_file()
            or wheel.stat().st_size != builds[0]["bytes"]
            or sha256(wheel) != builds[0]["sha256"]
        ):
            raise ValueError("wheel artifact identity drifted")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ArchiveVerificationError(
            "archive wheel evidence is stale or invalid"
        ) from exc
    return {
        "name": "wheel-source-binding",
        "status": "passed",
        "wheel": wheel_relative,
        "sha256": builds[0]["sha256"],
    }


def _native_tool(name: str) -> str:
    available = shutil.which(name)
    if available:
        return available
    if os.name == "nt":
        program_files = os.environ.get("ProgramFiles(x86)")
        if program_files:
            vswhere = Path(program_files) / "Microsoft Visual Studio/Installer/vswhere.exe"
            if vswhere.is_file():
                discovered = subprocess.run(
                    [
                        str(vswhere),
                        "-latest",
                        "-products",
                        "*",
                        "-find",
                        f"Common7\\IDE\\CommonExtensions\\Microsoft\\CMake\\CMake\\bin\\{name}.exe",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                for value in discovered.stdout.splitlines():
                    candidate = Path(value.strip())
                    if candidate.is_file():
                        return str(candidate)
    raise ArchiveVerificationError(f"required native build tool is unavailable: {name}")


def _member_path(info: zipfile.ZipInfo) -> PurePosixPath:
    raw = getattr(info, "orig_filename", info.filename)
    if not raw or "\x00" in raw or "\\" in raw:
        raise ArchiveVerificationError(f"unsafe archive entry: {raw!r}")
    member = PurePosixPath(raw)
    if member.is_absolute() or any(part in {"", ".", ".."} for part in member.parts):
        raise ArchiveVerificationError(f"unsafe archive path: {raw!r}")
    if member.parts and ":" in member.parts[0]:
        raise ArchiveVerificationError(f"drive-qualified archive entry: {raw!r}")
    mode = (info.external_attr >> 16) & 0xFFFF
    if mode and stat.S_ISLNK(mode):
        raise ArchiveVerificationError(f"archive links are forbidden: {raw!r}")
    if stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR}:
        raise ArchiveVerificationError(f"archive special file is forbidden: {raw!r}")
    if info.flag_bits & 0x1:
        raise ArchiveVerificationError(f"encrypted archive entry is forbidden: {raw!r}")
    return member


def extract_safely(archive: zipfile.ZipFile, destination: Path) -> None:
    members: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
    seen: set[str] = set()
    total = 0
    for info in archive.infolist():
        member = _member_path(info)
        normalized = member.as_posix().rstrip("/").casefold()
        if normalized in seen:
            raise ArchiveVerificationError(f"duplicate archive entry: {info.filename}")
        seen.add(normalized)
        total += int(info.file_size)
        if total > MAX_UNCOMPRESSED_BYTES:
            raise ArchiveVerificationError("archive exceeds the extraction budget")
        members.append((info, member))
    if archive.testzip() is not None:
        raise ArchiveVerificationError("archive CRC validation failed")
    root = destination.resolve(strict=True)
    for info, member in members:
        target = destination.joinpath(*member.parts)
        resolved = target.resolve(strict=False)
        if resolved != root and root not in resolved.parents:
            raise ArchiveVerificationError(f"archive entry escapes workspace: {member}")
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source, target.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)


def _run_complete(
    name: str,
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    timeout: int,
) -> dict[str, Any]:
    """Run one bounded check and return its evidence instead of failing early."""
    merged = {**os.environ, **environment}
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=merged,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=os.name != "nt",
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return {
            "name": name,
            "status": "passed" if process.returncode == 0 else "failed",
            "returnCode": process.returncode,
            "timedOut": False,
            "durationSeconds": round(time.perf_counter() - started, 3),
            "stdout": stdout[-20000:],
            "stderr": stderr[-20000:],
        }
    except subprocess.TimeoutExpired:
        cleanup_errors: list[str] = []
        tree_termination_succeeded = False
        if os.name == "nt":
            # Popen.kill() terminates only the direct process on Windows. Archive
            # tests legitimately create ProcessPool workers, and abandoning those
            # descendants can poison every later Python launch in the same wave.
            # Kill the still-live parent and its owned descendants as one tree.
            try:
                cleanup = subprocess.run(
                    ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                    check=False,
                    text=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                )
                tree_termination_succeeded = cleanup.returncode == 0
                if not tree_termination_succeeded:
                    cleanup_errors.append(
                        f"taskkill_return_code:{cleanup.returncode}"
                    )
            except (OSError, subprocess.TimeoutExpired) as exc:
                cleanup_errors.append(f"taskkill:{type(exc).__name__}")
                process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
                tree_termination_succeeded = True
            except OSError as exc:
                cleanup_errors.append(f"killpg:{type(exc).__name__}")
                process.kill()
        try:
            stdout, stderr = process.communicate(timeout=30)
        except subprocess.TimeoutExpired as cleanup_timeout:
            cleanup_errors.append("communicate:TimeoutExpired")
            process.kill()
            stdout = cleanup_timeout.stdout or ""
            stderr = cleanup_timeout.stderr or ""
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
        direct_process_exited = process.poll() is not None
        if not direct_process_exited:
            cleanup_errors.append("direct_process_still_running")
        return {
            "name": name,
            "status": "failed",
            "returnCode": 124,
            "timedOut": True,
            "durationSeconds": round(time.perf_counter() - started, 3),
            "stdout": stdout[-20000:],
            "stderr": stderr[-20000:],
            "cleanupComplete": (
                tree_termination_succeeded
                and direct_process_exited
                and not cleanup_errors
            ),
            "cleanupErrors": cleanup_errors,
        }


def _archive_python_files(clean_root: Path) -> tuple[Path, ...]:
    excluded = {
        "test_performance_evidence.py",
        "test_security_report.py",
    }
    return tuple(
        path
        for path in sorted((clean_root / "tests").glob("test_*.py"))
        if path.name not in excluded
    )


def _run_archive_python_files(
    clean_root: Path,
    parent: Path,
    native_library: Path,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, test_file in enumerate(_archive_python_files(clean_root)):
        relative_test = str(Path("tests") / test_file.name)
        timeout = (
            ARCHIVE_LONG_PYTHON_FILE_TIMEOUT_SECONDS
            if test_file.name in ARCHIVE_LONG_PYTHON_FILES
            else ARCHIVE_PYTHON_FILE_TIMEOUT_SECONDS
        )
        command = [
            sys.executable,
            "-B",
            str(clean_root / "scripts/run_archive_pytest_file.py"),
            "--diagnostic-after-seconds",
            str(max(1, timeout - 30)),
            "--",
            "-q",
            "--override-ini=addopts=",
            relative_test,
        ]
        if test_file.name == "test_repository_structure.py":
            command.extend(
                [
                    "--deselect=tests/test_repository_structure.py::test_runtime_and_build_directories_are_not_tracked",
                    "--deselect=tests/test_repository_structure.py::test_workspace_root_is_the_repository_root",
                ]
            )
        runtime = parent / "python-file-runtimes" / f"{index:03d}-{test_file.stem}"
        environment = {
            "PYTHONPATH": os.pathsep.join(
                (
                    str(clean_root),
                    str(clean_root / "src"),
                    str(clean_root / "sdk/python"),
                )
            ),
            "BALOOT_ROOT": str(clean_root),
            "BALOOT_DATA_DIR": str(runtime),
            "BALOOT_RUNTIME": str(runtime),
            "BALOOT_OUTPUTS": str(runtime / "outputs"),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "SOURCE_DATE_EPOCH": "315532800",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "ORTOOLS_NUM_THREADS": "1",
            "BALOOT_NATIVE_LIBRARY": str(native_library),
        }
        result = _run_complete(
            test_file.name,
            command,
            cwd=clean_root,
            environment=environment,
            timeout=timeout,
        )
        results.append(result)
    passed = sum(result["status"] == "passed" for result in results)
    return {
        "name": "archive-python-files",
        "status": "passed" if passed == len(results) else "failed",
        "testFiles": len(results),
        "passedTestFiles": passed,
        "failedTestFiles": len(results) - passed,
        "durationSeconds": round(time.perf_counter() - started, 3),
        "results": results,
    }


def _node_command(command: str, cwd: Path) -> list[str]:
    if os.name != "nt":
        return ["bash", "-lc", command]
    wrapper = Path(os.environ.get("BALOOT_HIDDEN_NODE_WRAPPER", ""))
    if not wrapper.is_file():
        return ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command]
    encoded = base64.b64encode(command.encode("utf-8")).decode("ascii")
    return [
        "powershell.exe",
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(wrapper),
        "-EncodedCommand",
        encoded,
        "-WorkingDirectory",
        str(cwd),
    ]


def _node_environment() -> dict[str, str]:
    if os.name != "nt":
        return {}
    node = shutil.which("node")
    npm = shutil.which("npm.cmd")
    system_root = os.environ.get("SystemRoot")
    if not node or not npm or not system_root:
        raise ArchiveVerificationError(
            "required Windows Node or system runtime path is unavailable"
        )
    directories = (
        Path(node).parent,
        Path(npm).parent,
        Path(sys.executable).parent,
        Path(system_root) / "System32",
        Path(system_root),
        Path(system_root) / "System32/WindowsPowerShell/v1.0",
    )
    return {
        "PATH": os.pathsep.join(
            dict.fromkeys(str(path) for path in directories)
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workspace")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    archive_path = Path(args.archive).resolve()
    report_path = Path(args.output).resolve()
    workspace = Path(args.workspace).resolve() if args.workspace else Path(tempfile.gettempdir())
    workspace.mkdir(parents=True, exist_ok=True)
    parent = Path(tempfile.mkdtemp(prefix="baloot-release-verify-", dir=workspace))
    checks: list[dict[str, Any]] = []
    clean_root: Path | None = None
    native_build: Path | None = None
    try:
        with zipfile.ZipFile(archive_path) as archive:
            extract_safely(archive, parent)
        roots = [path for path in parent.iterdir() if path.is_dir()]
        if len(roots) != 1:
            raise ArchiveVerificationError("archive must contain one repository root")
        clean_root = roots[0]
        manifest = json.loads(
            (clean_root / "release/source-manifest.json").read_text(encoding="utf-8")
        )
        for relative, metadata in manifest["files"].items():
            path = clean_root / relative
            if (
                not path.is_file()
                or path.stat().st_size != metadata["bytes"]
                or sha256(path) != metadata["sha256"]
            ):
                raise ArchiveVerificationError(f"source manifest mismatch: {relative}")
        checks.append(
            {"name": "source-manifest", "status": "passed", "files": manifest["fileCount"]}
        )
        checks.append(_verify_packaged_security(clean_root, manifest))
        checks.append(_verify_packaged_wheel(clean_root, manifest))
        native_build_parent = (
            Path(os.environ["LOCALAPPDATA"]) / "BalootPlatform/native-verification"
            if os.name == "nt" and os.environ.get("LOCALAPPDATA")
            else Path(tempfile.gettempdir())
        )
        native_build_parent.mkdir(parents=True, exist_ok=True)
        native_build = Path(
            tempfile.mkdtemp(prefix="baloot-native-", dir=native_build_parent)
        )
        cmake = _native_tool("cmake")
        ctest = _native_tool("ctest")
        native_checks = (
            (
                "native-release-configure",
                [
                    cmake,
                    "-S",
                    "native",
                    "-B",
                    str(native_build),
                    "-DCMAKE_BUILD_TYPE=Release",
                ],
            ),
            (
                "native-release-build",
                [
                    cmake,
                    "--build",
                    str(native_build),
                    "--config",
                    "Release",
                    "--parallel",
                    "1",
                ],
            ),
            (
                "native-release-test",
                [
                    ctest,
                    "--test-dir",
                    str(native_build),
                    "-C",
                    "Release",
                    "--output-on-failure",
                    "--timeout",
                    "150",
                ],
            ),
        )
        for name, command in native_checks:
            result = _run_complete(
                name,
                command,
                cwd=clean_root,
                environment={"CMAKE_BUILD_PARALLEL_LEVEL": "1"},
                timeout=ARCHIVE_NATIVE_TIMEOUT_SECONDS,
            )
            checks.append(result)
            if result["status"] != "passed":
                raise ArchiveVerificationError(
                    f"archive native check failed: {json.dumps(result, ensure_ascii=False, sort_keys=True)}"
                )
        native_library = (
            native_build / "Release/baloot_native.dll"
            if os.name == "nt"
            else native_build / (
                "libbaloot_native.dylib"
                if sys.platform == "darwin"
                else "libbaloot_native.so"
            )
        )
        if not native_library.is_file():
            raise ArchiveVerificationError(
                f"archive native library is missing after build: {native_library}"
            )
        native_runtime = parent / "native-preflight-runtime"
        native_environment = {
            "PYTHONPATH": os.pathsep.join(
                (
                    str(clean_root),
                    str(clean_root / "src"),
                    str(clean_root / "sdk/python"),
                )
            ),
            "BALOOT_ROOT": str(clean_root),
            "BALOOT_DATA_DIR": str(native_runtime),
            "BALOOT_RUNTIME": str(native_runtime),
            "BALOOT_OUTPUTS": str(native_runtime / "outputs"),
            "BALOOT_NATIVE_LIBRARY": str(native_library),
        }
        native_preflight = _run_complete(
            "native-backend-preflight",
            [
                sys.executable,
                "-B",
                "-c",
                (
                    "from baloot_platform.service.container import build_container; "
                    "container = build_container(); "
                    "backend = container.research.analyzers['screening'].exact_search_backend; "
                    "actual = None if backend is None else backend.descriptor.id; "
                    "container.shutdown(); "
                    "assert actual == 'native_exact_v3_python_fallback_v1', actual; "
                    "print(actual)"
                ),
            ],
            cwd=clean_root,
            environment=native_environment,
            timeout=ARCHIVE_NATIVE_TIMEOUT_SECONDS,
        )
        checks.append(native_preflight)
        if native_preflight["status"] != "passed":
            raise ArchiveVerificationError(
                "archive native backend preflight failed: "
                + json.dumps(native_preflight, ensure_ascii=False, sort_keys=True)
            )
        remotion = clean_root / "src/baloot_platform/video/remotion"
        foundry = clean_root / "src/baloot_platform/service/foundry_dashboard"
        node_checks = (
            (
                "remotion-install-and-typecheck",
                remotion,
                "npm.cmd ci --ignore-scripts --prefer-offline --no-audit --no-fund; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; npm.cmd run typecheck"
                if os.name == "nt"
                else "npm ci --ignore-scripts --prefer-offline --no-audit --no-fund && npm run typecheck",
            ),
            (
                "foundry-install-and-check",
                foundry,
                "npm.cmd ci --ignore-scripts --prefer-offline --no-audit --no-fund; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; npm.cmd run check"
                if os.name == "nt"
                else "npm ci --ignore-scripts --prefer-offline --no-audit --no-fund && npm run check",
            ),
            (
                "foundry-research-tests",
                foundry,
                "npm.cmd run test:research"
                if os.name == "nt"
                else "npm run test:research",
            ),
        )
        node_environment = _node_environment()
        node_failures: list[dict[str, Any]] = []
        for name, directory, command in node_checks:
            result = _run_complete(
                name,
                _node_command(command, directory),
                cwd=directory,
                environment=node_environment,
                timeout=ARCHIVE_NODE_TIMEOUT_SECONDS,
            )
            checks.append(result)
            if result["status"] != "passed":
                node_failures.append(result)
        if node_failures:
            raise ArchiveVerificationError(
                "archive Node checks failed: "
                + json.dumps(node_failures, ensure_ascii=False, sort_keys=True)
            )
        source_only_tests = (
            "tests/test_performance_evidence.py",
            "tests/test_security_report.py",
            "tests/test_repository_structure.py::test_runtime_and_build_directories_are_not_tracked",
            "tests/test_repository_structure.py::test_workspace_root_is_the_repository_root",
        )
        checks.append(
            {
                "name": "source-only-test-boundary",
                "status": "passed",
                "excluded": list(source_only_tests),
                "reason": "These checks require Git history, which is intentionally excluded from the deterministic source archive and passed in the source-tree release gate.",
            }
        )
        python_files = _run_archive_python_files(
            clean_root,
            parent,
            native_library,
        )
        checks.append(python_files)
        if python_files["status"] != "passed":
            failures = [
                result
                for result in python_files["results"]
                if result["status"] != "passed"
            ]
            raise ArchiveVerificationError(
                "archive Python files failed: "
                + json.dumps(failures, ensure_ascii=False, sort_keys=True)
            )
        for relative, metadata in manifest["files"].items():
            path = clean_root / relative
            if (
                not path.is_file()
                or path.stat().st_size != metadata["bytes"]
                or sha256(path) != metadata["sha256"]
            ):
                raise ArchiveVerificationError(
                    f"post-test source manifest mismatch: {relative}"
                )
        checks.append(
            {
                "name": "post-test-source-manifest",
                "status": "passed",
                "files": manifest["fileCount"],
            }
        )
        report = {
            "schemaVersion": 1,
            "project": "baloot-platform",
            "version": manifest["version"],
            "status": "passed",
            "archive": archive_path.name,
            "archiveSha256": sha256(archive_path),
            "checks": checks,
            "cleanRoot": str(clean_root) if args.keep else None,
        }
    except Exception as exc:
        report = {
            "schemaVersion": 1,
            "project": "baloot-platform",
            "status": "failed",
            "archive": archive_path.name,
            "archiveSha256": sha256(archive_path) if archive_path.is_file() else None,
            "checks": checks,
            "error": {"type": type(exc).__name__, "message": str(exc)[:20000]},
            "cleanRoot": str(clean_root) if args.keep and clean_root else None,
        }
    finally:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        if not args.keep:
            shutil.rmtree(parent, ignore_errors=True)
        if native_build is not None:
            shutil.rmtree(native_build, ignore_errors=True)
    print(report_path)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
