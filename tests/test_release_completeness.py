from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import pytest
from fastapi.testclient import TestClient

from baloot_platform.config import build_bundle
from baloot_platform.core.enums import Contract, Seat, Suit
from baloot_platform.core.scenario import ScenarioRequest, generate_episode
from baloot_platform.service import api as api_module
from baloot_platform.service.api import (
    ChallengeRenderRequest,
    HistoricalReplayRenderRequest,
    VideoJobRequest,
    _rules_for,
    app,
)
from baloot_platform.service.routes.hitl import ClipRequest


def _pid_is_running(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            return bool(
                kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
                and int(exit_code.value) == 259
            )
        finally:
            kernel32.CloseHandle(handle)
    stat = Path(f"/proc/{pid}/stat")
    if stat.is_file():
        try:
            return stat.read_text(encoding="utf-8").rsplit(")", 1)[1].split()[0] != "Z"
        except (OSError, IndexError):
            pass
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def test_current_archive_excludes_local_runtime_and_operator_evidence(root: Path):
    spec = importlib.util.spec_from_file_location(
        "balootx_package_release",
        root / "scripts/package_release.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert {
        "private-inputs", ".codex-local", ".local-state",
        ".codex-tmp",
        ".playwright-cli",
        "output",
    } <= module.EXCLUDED_DIRECTORIES
    assert "research-scenario-v4-goldens" in module.CURRENT_OUTPUT_DIRECTORIES
    assert "ortools-v4-five-video-evidence" in module.CURRENT_OUTPUT_DIRECTORIES


def test_release_verifier_keeps_agent_expert_review_external(
    root: Path, tmp_path: Path
):
    spec = importlib.util.spec_from_file_location(
        "balootx_verify_release_external_gate",
        root / "scripts/verify_release.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    gate = module.agent_qualification_boundary(tmp_path, None)

    assert gate == {
        "status": "pending_external_review",
        "releaseStatus": "not_assembled",
        "benchmarkEvidenceComplete": False,
        "professionalReviewComplete": False,
        "expertGenerationAllowed": False,
        "issues": [
            "permanent_holdout_missing",
            "professional_review_missing",
        ],
    }


def test_release_verifier_refreshes_and_binds_current_test_inventory(root: Path):
    source = (root / "scripts/verify_release.py").read_text(encoding="utf-8")

    assert '"test-inventory"' in source
    assert '"scripts/generate_test_inventory_v20.py"' in source
    assert '"release/test-inventory.json"' in source


def test_release_verifier_refreshes_and_binds_current_performance_evidence(
    root: Path,
):
    source = (root / "scripts/verify_release.py").read_text(encoding="utf-8")

    for required in (
        '"release/search-performance.json"',
        '"release/research-scenario-v4-performance.json"',
        '"release/performance-benchmark.json"',
        '"scripts/benchmark_search.py"',
        '"scripts/benchmark_research_scenario_v4.py"',
        '"scripts/generate_performance_evidence.py"',
        'performance.get("researchV4", {}).get("evidenceSha256")',
        'current_source = git_identity(ROOT)',
        'search_performance.get("source")',
        'search_performance.get("engine")',
        'bundle_path="configs/bundle_ortools_research_v4.yaml"',
        '"bundleSha256": ortools_bundle.sha256',
        'research_performance.get("source")',
        'research_bundle.get("componentSha256")',
        'wheel.get("source")',
    ):
        assert required in source
    assert source.index('"scripts/benchmark_search.py"') < source.index(
        '"python-suite"'
    )
    assert source.index('"scripts/benchmark_research_scenario_v4.py"') < source.index(
        '"python-suite"'
    )
    assert source.index('"generate_media_probe.py"') < source.index(
        '"scripts/generate_performance_evidence.py"'
    )


def test_release_verifier_binds_installed_wheel_predicate_catalog(root: Path):
    source = (root / "scripts/verify_release.py").read_text(encoding="utf-8")

    assert "ScenarioPredicateRegistryV1().catalog()" in source
    assert 'installed_runtime.get("scenarioPredicateCount")' in source
    assert 'installed_runtime.get("scenarioPredicateCatalogSha256")' in source


def test_release_verifier_has_a_measured_source_suite_budget_and_node_progress(
    root: Path,
):
    spec = importlib.util.spec_from_file_location(
        "balootx_verify_release_source_suite",
        root / "scripts/verify_release.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = (root / "scripts/verify_release.py").read_text(encoding="utf-8")

    assert module.PYTHON_SUITE_TIMEOUT_SECONDS == 7200
    assert '"-vv"' in source
    assert 'timeout=PYTHON_SUITE_TIMEOUT_SECONDS' in source


def test_release_verifier_preserves_structured_failure_tail(
    root: Path, tmp_path: Path
):
    spec = importlib.util.spec_from_file_location(
        "balootx_verify_release_failure_evidence",
        root / "scripts/verify_release.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summary = (
        "FAILED tests/test_engine_guide.py::test_generated_guide - assertion\n"
        "ERROR tests/test_other.py::test_setup - fixture failed\n"
        "1 failed, 1 error\n"
    )
    failure = RuntimeError(
        json.dumps(
            {
                "name": "python-suite",
                "status": "failed",
                "returnCode": 1,
                "stdout": "diff-prefix\n" * 3000 + summary,
                "stderr": "",
            },
            ensure_ascii=False,
        )
    )

    evidence = module._error_evidence(failure)

    assert evidence["message"] == "python-suite: failed"
    assert evidence["details"]["stdout"].endswith(summary)
    assert len(evidence["details"]["stdout"]) <= 20000
    assert evidence["details"]["returnCode"] == 1
    assert module._pytest_failure_nodes("diff-prefix\n" * 3000 + summary) == [
        "tests/test_engine_guide.py::test_generated_guide",
        "tests/test_other.py::test_setup",
    ]

    failure_script = tmp_path / "test-failure-nodes.py"
    failure_script.write_text(
        "print('FAILED tests/test_engine_guide.py::test_generated_guide - assertion')\n"
        "raise SystemExit(1)\n",
        encoding="utf-8",
    )
    try:
        with pytest.raises(RuntimeError) as caught:
            module.run("python-suite", [sys.executable, "-B", str(failure_script)])
        assert json.loads(str(caught.value))["failureNodes"] == [
            "tests/test_engine_guide.py::test_generated_guide"
        ]
    finally:
        failure_script.unlink(missing_ok=True)


@pytest.mark.real_clock
def test_release_verifier_is_single_flight_and_cleans_timed_out_tree(
    root: Path, tmp_path: Path
):
    spec = importlib.util.spec_from_file_location(
        "balootx_verify_release_process_tree",
        root / "scripts/verify_release.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.LOCK = tmp_path / "verify-release.lock"
    first_lock = module._acquire_single_flight_lock()
    try:
        with pytest.raises(RuntimeError, match="already active"):
            module._acquire_single_flight_lock()
    finally:
        first_lock.close()

    child_pid_path = tmp_path / "release-child.pid"
    child_script = tmp_path / "release-child.py"
    parent_script = tmp_path / "release-parent.py"
    child_script.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    parent_script.write_text(
        "import pathlib, subprocess, sys, time\n"
        f"process = subprocess.Popen([sys.executable, '-B', {str(child_script)!r}])\n"
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(process.pid), encoding='utf-8')\n"
        "print('LAST-NODE: test_release_timeout_diagnostic', flush=True)\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match='"timedOut": true') as caught:
        module.run(
            "release-owned-tree-timeout",
            [sys.executable, "-B", str(parent_script)],
            cwd=tmp_path,
            timeout=1,
        )
    evidence = json.loads(str(caught.value))
    assert evidence["stdout"].endswith(
        "LAST-NODE: test_release_timeout_diagnostic\n"
    )
    assert evidence["stderr"] == ""
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 5
    while _pid_is_running(child_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _pid_is_running(child_pid) is False


def test_archive_verifier_selects_the_complete_python_scope_once(root: Path):
    spec = importlib.util.spec_from_file_location(
        "balootx_verify_archive",
        root / "scripts/verify_archive.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    selected = [path.name for path in module._archive_python_files(root)]
    assert len(selected) == len(set(selected))
    assert set(selected) == {
        path.name
        for path in (root / "tests").glob("test_*.py")
        if path.name
        not in {
            "test_performance_evidence.py",
            "test_security_report.py",
        }
    }


def test_archive_node_checks_avoid_redundant_advisory_fetch_without_weakening_security_evidence(
    root: Path,
):
    spec = importlib.util.spec_from_file_location(
        "balootx_verify_archive_node_install",
        root / "scripts/verify_archive.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = (root / "scripts/verify_archive.py").read_text(encoding="utf-8")

    install_flags = "--ignore-scripts --prefer-offline --no-audit --no-fund"
    assert module.ARCHIVE_NODE_TIMEOUT_SECONDS == 600
    assert source.count(install_flags) == 4
    assert "timeout=ARCHIVE_NODE_TIMEOUT_SECONDS" in source
    assert "node_failures" in source
    assert "_run_complete(" in source

    # The implicit npm-ci advisory POST never failed the install on findings.
    # Supply-chain and security evidence remain explicit release requirements.
    readiness = (root / "scripts/check_release_readiness.py").read_text(
        encoding="utf-8"
    )
    security = (root / "scripts/generate_security_report.py").read_text(
        encoding="utf-8"
    )
    assert '"release/sbom.cdx.json"' in readiness
    assert '"release/security-report.json"' in readiness
    assert 'sbom = load(root, "release/sbom.cdx.json")' in security


def test_release_verifier_admits_current_configuration_and_benchmark_authorities(
    root: Path,
):
    source = (root / "scripts/verify_release.py").read_text(encoding="utf-8")

    assert "admit_release_evidence" in source
    for relative in (
        "release/configuration-registry.json",
        "release/intelligence-benchmark.json",
        "release/rules-certification-candidate.json",
    ):
        assert source.count(f'"{relative}"') >= 2


@pytest.mark.real_clock
def test_archive_verifier_timeout_terminates_owned_process_tree(
    root: Path, tmp_path: Path
):
    spec = importlib.util.spec_from_file_location(
        "balootx_verify_archive_process_tree",
        root / "scripts/verify_archive.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    child_pid_path = tmp_path / "child.pid"
    grandchild_pid_path = tmp_path / "grandchild.pid"
    grandchild_script = tmp_path / "grandchild.py"
    child_script = tmp_path / "child.py"
    parent_script = tmp_path / "parent.py"
    grandchild_script.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    child_script.write_text(
        "import pathlib, subprocess, sys, time\n"
        f"process = subprocess.Popen([sys.executable, '-B', {str(grandchild_script)!r}])\n"
        f"pathlib.Path({str(grandchild_pid_path)!r}).write_text(str(process.pid), encoding='utf-8')\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    parent_script.write_text(
        "import pathlib, subprocess, sys, time\n"
        f"process = subprocess.Popen([sys.executable, '-B', {str(child_script)!r}])\n"
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(process.pid), encoding='utf-8')\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    unrelated = subprocess.Popen(
        [sys.executable, "-B", "-c", "import time;time.sleep(60)"],
        cwd=tmp_path,
    )
    try:
        result = module._run_complete(
            "owned-tree-timeout",
            [sys.executable, "-B", str(parent_script)],
            cwd=tmp_path,
            environment={},
            timeout=1,
        )
        assert result["status"] == "failed"
        assert result["timedOut"] is True
        assert result["returnCode"] == 124
        assert result["cleanupComplete"] is True
        assert result["cleanupErrors"] == []
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        grandchild_pid = int(grandchild_pid_path.read_text(encoding="utf-8"))

        deadline = time.monotonic() + 5
        while (
            _pid_is_running(child_pid) or _pid_is_running(grandchild_pid)
        ) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert _pid_is_running(child_pid) is False
        assert _pid_is_running(grandchild_pid) is False
        assert unrelated.poll() is None
        runtime_probe = module._run_complete(
            "post-timeout-python-probe",
            [sys.executable, "-B", "-c", "print('python-runtime-ready')"],
            cwd=tmp_path,
            environment={},
            timeout=10,
        )
        assert runtime_probe["status"] == "passed"
        assert runtime_probe["stdout"].strip() == "python-runtime-ready"
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=10)


def test_archive_verifier_launcher_uses_os_owned_single_flight(
    root: Path, tmp_path: Path
):
    launcher = root / "scripts/launch_archive_verification.ps1"
    source = launcher.read_text(encoding="utf-8")

    for required in (
        "Register-ScheduledTask",
        "Start-ScheduledTask",
        "-MultipleInstances IgnoreNew",
        "[Threading.Mutex]::new",
        ".WaitOne(0)",
        ".ProcessorAffinity = [IntPtr]$AffinityMask",
        "'run_local_capped.ps1') -AttachRootId $PID",
        "$AffinityMask = $AffinityMask -band",
        "cpuCeiling = $cpuCeilingReceipt",
        "$processInfo.RedirectStandardOutput = $true",
        "$processInfo.RedirectStandardError = $true",
        "'/PID', [string]$verifier.Id, '/T', '/F'",
        "[IO.File]::Move($temporaryReceipt, $Receipt)",
        "-ExecutionTimeLimit ([TimeSpan]::Zero)",
        "-DisallowHardTerminate",
        "[Diagnostics.ProcessStartInfo]::new()",
        "$exitCode = [int]$verifier.ExitCode",
    ):
        assert required in source
    assert source.index("'run_local_capped.ps1') -AttachRootId $PID") < source.index("function Get-Sha256")
    assert "Get-FileHash" not in source
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "ORTOOLS_NUM_THREADS",
    ):
        assert f"'{variable}'" in source

    if os.name == "nt":
        parser = (
            "$tokens=$null;$errors=$null;"
            f"[Management.Automation.Language.Parser]::ParseFile('{launcher}',"
            "[ref]$tokens,[ref]$errors)|Out-Null;"
            "if($errors.Count){$errors|ForEach-Object{$_.ToString()}|Write-Error;exit 1}"
        )
        completed = subprocess.run(
            [
                "powershell.exe",
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                parser,
            ],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        archive = tmp_path / "candidate.zip"
        archive.write_bytes(b"not-the-requested-archive")
        receipt = tmp_path / "terminal-receipt.json"
        completed = subprocess.run(
            [
                "powershell.exe",
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(launcher),
                "-Worker",
                "-Archive",
                str(archive),
                "-ArchiveSha256",
                "0" * 64,
                "-Report",
                str(tmp_path / "report.json"),
                "-Stdout",
                str(tmp_path / "stdout.log"),
                "-Stderr",
                str(tmp_path / "stderr.log"),
                "-Receipt",
                str(receipt),
                "-Workspace",
                str(tmp_path / "workspace"),
                "-TaskName",
                f"BalootX-ArchiveVerify-Test-{os.getpid()}",
                "-PythonExecutable",
                sys.executable,
            ],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        assert completed.returncode == 1
        terminal = json.loads(receipt.read_text(encoding="utf-8"))
        assert terminal["status"] == "failed"
        assert terminal["exitCode"] == 1
        assert terminal["verifierPid"] is None
        assert terminal["expectedArchiveSha256"] == "0" * 64
        assert terminal["actualArchiveSha256"] != terminal["expectedArchiveSha256"]

        public_archive = tmp_path / "public-preflight.zip"
        public_archive.write_bytes(b"public-launcher-scalar-python-probe")
        public_sha = hashlib.sha256(public_archive.read_bytes()).hexdigest()
        completed = subprocess.run(
            [
                "powershell.exe",
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(launcher),
                "-Archive",
                str(public_archive),
                "-ArchiveSha256",
                public_sha,
                "-Report",
                str(tmp_path / "public-report.json"),
                "-Stdout",
                str(tmp_path / "public-stdout.log"),
                "-Stderr",
                str(tmp_path / "public-stderr.log"),
                "-Receipt",
                str(tmp_path / "public-receipt.json"),
                "-Workspace",
                str(tmp_path),
            ],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        assert completed.returncode == 1
        assert "Refusing to reuse verifier workspace" in completed.stderr
        assert "GetFullPath" not in completed.stderr
        assert not (tmp_path / "public-report.json").exists()
        assert not (tmp_path / "public-receipt.json").exists()

        fake_root = tmp_path / "fake-repository"
        fake_scripts = fake_root / "scripts"
        fake_scripts.mkdir(parents=True)
        fake_launcher = fake_scripts / launcher.name
        fake_launcher.write_text(source, encoding="utf-8")
        (fake_scripts / "run_local_capped.ps1").write_text(
            (root / "scripts/run_local_capped.ps1").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        fake_config = fake_root / "configs/runtime"
        fake_config.mkdir(parents=True)
        (fake_config / "local_cpu.json").write_text(
            (root / "configs/runtime/local_cpu.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (fake_scripts / "verify_archive.py").write_text(
            "import json, pathlib, sys\n"
            "output = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
            "output.write_text(json.dumps({'status': 'failed'}), encoding='utf-8')\n"
            "raise SystemExit(7)\n",
            encoding="utf-8",
        )
        fake_archive = fake_root / "candidate.zip"
        fake_archive.write_bytes(b"exact-worker-exit-code-probe")
        fake_sha = hashlib.sha256(fake_archive.read_bytes()).hexdigest()
        fake_receipt = fake_root / "exit.json"
        completed = subprocess.run(
            [
                "powershell.exe",
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(fake_launcher),
                "-Worker",
                "-Archive",
                str(fake_archive),
                "-ArchiveSha256",
                fake_sha,
                "-Report",
                str(fake_root / "report.json"),
                "-Stdout",
                str(fake_root / "stdout.log"),
                "-Stderr",
                str(fake_root / "stderr.log"),
                "-Receipt",
                str(fake_receipt),
                "-Workspace",
                str(fake_root / "workspace"),
                "-TaskName",
                f"BalootX-ArchiveVerify-Exit-Test-{os.getpid()}",
                "-PythonExecutable",
                sys.executable,
            ],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        assert completed.returncode == 7
        terminal = json.loads(fake_receipt.read_text(encoding="utf-8"))
        assert terminal["exitCode"] == 7
        assert terminal["reportStatus"] == "failed"
        assert terminal["status"] == "failed"


def test_archive_verifier_rebuilds_and_requires_native_backend(root: Path):
    source = (root / "scripts/verify_archive.py").read_text(encoding="utf-8")

    configure = source.index('"native-release-configure"')
    preflight = source.index('"native-backend-preflight"')
    python_files = source.index("python_files = _run_archive_python_files")
    assert configure < preflight < python_files
    for required in (
        '"native-release-build"',
        '"native-release-test"',
        "native_exact_v3_python_fallback_v1",
        '"CMAKE_BUILD_PARALLEL_LEVEL": "1"',
        '"BALOOT_NATIVE_LIBRARY": str(native_library)',
        '"BalootPlatform/native-verification"',
        'tempfile.mkdtemp(prefix="baloot-native-", dir=native_build_parent)',
        'vswhere.exe',
    ):
        assert required in source


def test_release_zip_cannot_include_local_browser_or_runtime_state(
    root: Path, tmp_path: Path
):
    spec = importlib.util.spec_from_file_location(
        "balootx_package_release_isolated",
        root / "scripts/package_release.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path
    (tmp_path / "src").mkdir()
    (tmp_path / "src/kept.txt").write_text("kept", encoding="utf-8")
    (tmp_path / "release").mkdir()
    with pytest.raises(FileNotFoundError, match="security-report.json"):
        module.build(tmp_path / "missing-security.zip")
    (tmp_path / "release/security-report.json").write_text(
        "{}\n", encoding="utf-8"
    )
    source_identity = {
        "revision": "a" * 40,
        "tree": "b" * 40,
        "scope": "git_visible_product_inputs_v1",
        "excludedPaths": ["release/"],
        "binding": "exact_git_head",
    }
    module.git_identity = lambda _root: source_identity
    module.verify_security = lambda _value, **_kwargs: None
    (tmp_path / "release/repository-archive-verification-failed.json").write_text(
        "{}\n", encoding="utf-8"
    )
    (tmp_path / "release/verification-report.json").write_text("{}\n", encoding="utf-8")
    for name in (
        "live-postgres-research-final.json",
        "verifier-container-run-final.exit.json",
        "verifier-image-build-final.exit.json",
        "verifier-preflight-scenario-final.json",
        "verifier-preflight-v20.json",
        "verifier-preflight.json",
    ):
        (tmp_path / "release" / name).write_text("{}\n", encoding="utf-8")
    for directory in ("private-inputs", ".codex-local", ".local-state", ".codex-tmp", ".playwright-cli", "output"):
        path = tmp_path / directory
        path.mkdir()
        (path / "private.txt").write_text("excluded", encoding="utf-8")
    for relative in (
        "assets/cards/packs/poker_double_backmarked_baloot_32/card.png",
        "src/baloot_platform/resources/assets/cards/packs/poker_double_backmarked_baloot_32/card.png",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("unresolved asset", encoding="utf-8")
    archive = module.build(tmp_path / "candidate.zip")
    with zipfile.ZipFile(archive) as zipped:
        names = zipped.namelist()
        manifest = json.loads(
            zipped.read(
                next(
                    name
                    for name in names
                    if name.endswith("/release/source-manifest.json")
                )
            )
        )
    assert any(name.endswith("/src/kept.txt") for name in names)
    assert any(name.endswith("/release/verification-report.json") for name in names)
    assert any(name.endswith("/release/security-report.json") for name in names)
    assert manifest["source"] == source_identity
    assert not any(
        f"/{directory}/" in name
        for name in names
        for directory in ("private-inputs", ".codex-local", ".local-state", ".codex-tmp", ".playwright-cli", "output")
    )
    assert not any("repository-archive-verification" in name for name in names)
    assert not any("live-postgres-research-" in name for name in names)
    assert not any("verifier-container-" in name for name in names)
    assert not any("verifier-image-build-" in name for name in names)
    assert not any("verifier-preflight" in name for name in names)
    assert not any("poker_double_backmarked_baloot_32" in name for name in names)


def test_supply_chain_inventory_covers_released_npm_and_bundled_assets(
    root: Path, tmp_path: Path
):
    sbom = tmp_path / "sbom.json"
    inventory = tmp_path / "licenses.json"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/generate_supply_chain.py"),
            "--sbom-output",
            str(sbom),
            "--license-output",
            str(inventory),
        ],
        cwd=root,
        check=True,
    )
    value = json.loads(inventory.read_text(encoding="utf-8"))
    assert value["status"] == "passed"
    assert value["unresolvedReleasedComponents"] == []
    assert value["componentCount"] == len(value["components"])
    components = value["components"]
    ecosystems = {item.get("baloot:ecosystem") for item in components}
    asset_kinds = {item.get("baloot:asset-kind") for item in components}
    assert {"python", "npm"} <= ecosystems
    assert {
        "font",
        "playing-card-pack",
        "professional-video-svg",
        "voice-pack",
    } <= asset_kinds
    goodall = next(item for item in components if item["name"] == "goodall_modern_baloot_32")
    assert goodall["baloot:released"] == "true"
    unresolved_pack = next(
        (
            item
            for item in components
            if item["name"] == "poker_double_backmarked_baloot_32"
        ),
        None,
    )
    unresolved_manifest = (
        root / "assets/cards/packs/poker_double_backmarked_baloot_32/manifest.json"
    )
    if unresolved_manifest.is_file():
        assert unresolved_pack is not None
        assert unresolved_pack["baloot:released"] == "false"
        assert unresolved_pack["scope"] == "excluded-unresolved-asset"
    else:
        assert unresolved_pack is None


def test_authored_and_packaged_card_pack_profiles_disable_unresolved_assets(root: Path):
    import yaml

    paths = (
        root / "configs/assets/card_packs.yaml",
        root / "src/baloot_platform/resources/configs/assets/card_packs.yaml",
    )
    profiles = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in paths]
    assert profiles[0] == profiles[1]
    for profile in profiles:
        assert profile["default_pack"] == "goodall_modern_baloot_32"
        assert profile["packs"]["goodall_modern_baloot_32"]["enabled"] is True
        assert profile["packs"]["poker_double_backmarked_baloot_32"]["enabled"] is False


def _episode(rules, convention):
    return generate_episode(
        rules,
        convention,
        bundle_hash=rules.sha256,
        request=ScenarioRequest.governed(
            seed=73,
            contract=Contract.HOKUM,
            buyer=Seat.SOUTH,
            trump_suit=Suit.HEARTS,
            require_project=False,
            requested_samples=3,
            exact_allocation_limit=120,
            exact_search_max_remaining=3,
            max_seed_attempts=30,
        ),
    )


def test_episode_spec_covers_shuffle_buy_projects_play_and_score(rules, convention):
    data = _episode(rules, convention).to_dict()
    assert data["shuffle"]["algorithm"]
    assert len(data["shuffle"]["deckOrder"]) == 32
    assert len(data["shuffle"]["deckCommitment"]) == 64
    assert data["purchaseCard"] and data["buyer"] and data["declarer"]
    assert data["firstLeader"] in {"north", "west", "south", "east"}
    assert data["contract"] == "hokum" and data["trumpSuit"] == "hearts"
    assert len(data["initialFive"]) == 4 and all(len(cards) == 5 for cards in data["initialFive"].values())
    assert len(data["initialHands"]) == 4 and all(len(cards) == 8 for cards in data["initialHands"].values())
    assert len(data["completedTricks"]) == 8
    assert sum(len(trick["plays"]) for trick in data["completedTricks"]) == 32
    assert data["roundScore"]["awardedPoints"]
    event_types = [event["type"] for event in data["events"]]
    for required in (
        "round_created", "deck_shuffled", "purchase_card_exposed", "initial_deal_complete",
        "contract_purchased", "final_deal_complete", "project_holdings_detected",
        "play_started", "round_scored",
    ):
        assert required in event_types


def test_effective_bundle_hash_tracks_rules_conventions_and_video_settings():
    base = VideoJobRequest(contract="hokum", trumpSuit="hearts")
    _, _, _, base_hash = _rules_for(base)
    variants = (
        VideoJobRequest(contract="hokum", trumpSuit="hearts", theme="royal"),
        VideoJobRequest(contract="hokum", trumpSuit="hearts", firstLeaderPolicy="buyer"),
        VideoJobRequest(contract="hokum", trumpSuit="hearts", conventionProfile="experimental_weighting"),
        VideoJobRequest(contract="hokum", trumpSuit="hearts", fps=30),
    )
    assert all(_rules_for(item)[3] != base_hash for item in variants)
    assert len({base_hash, *(_rules_for(item)[3] for item in variants)}) == 5


def test_video_job_visible_defaults_come_from_governed_video_copy():
    request = VideoJobRequest()
    copy = api_module.ACTIVE_BUNDLE.video["localizations"]
    assert request.titleAr == copy["video.decision.challenge_title"]
    assert request.questionAr == copy["video.decision.challenge_question"]


def test_theme_selection_changes_resolved_colors():
    _, _, emerald, _ = _rules_for(VideoJobRequest(contract="hokum", trumpSuit="hearts", theme="emerald"))
    _, _, royal, _ = _rules_for(VideoJobRequest(contract="hokum", trumpSuit="hearts", theme="royal"))
    assert emerald["selected_theme"] == "emerald"
    assert royal["selected_theme"] == "royal"
    assert emerald["colors"] != royal["colors"]


def test_friendly_configuration_normalizes_disabled_dependencies():
    candidate = VideoJobRequest.model_validate({"contract": "sun", "rulesProfile": "sun_only", "projectsEnabled": False})
    assert candidate.trumpSuit == "auto"
    assert not candidate.projectsEnabled and not candidate.balootEnabled and not candidate.requireProject


@pytest.mark.parametrize(
    ("model", "payload"),
    (
        (ChallengeRenderRequest, {"width": 1080, "height": 1080}),
        (ChallengeRenderRequest, {"width": 1080, "height": 1280}),
        (HistoricalReplayRenderRequest, {"width": 720, "height": 1920}),
        (ClipRequest, {"width": 1920, "height": 1080}),
        (ClipRequest, {"width": 1080}),
    ),
)
def test_all_release_request_models_reject_non_mobile_canvas(model, payload):
    with pytest.raises(ValueError):
        model.model_validate(payload)


def test_all_release_request_models_accept_governed_mobile_canvas():
    assert ChallengeRenderRequest(width=1080, height=1920)
    assert HistoricalReplayRenderRequest(width=720, height=1280)
    assert ClipRequest(width=1080, height=1920)
    assert ClipRequest()


def test_studio_exposes_supported_profiles_and_configured_leader(root: Path):
    text = (root / "src/baloot_platform/service/static/studio.html").read_text(encoding="utf-8")
    assert 'value="hard_facts_only"' in text
    assert 'value="experimental_display"' in text
    assert 'value="experimental_weighting"' in text
    assert 'id="configuredFirstLeader"' in text
    assert "premium_video" in text
    assert "premium_match_v4" not in text
    assert "premium_pillow" not in text


def test_local_rate_limiter_and_metrics(monkeypatch):
    monkeypatch.setenv("BALOOT_ENV", "production")
    monkeypatch.setenv("BALOOT_RATE_LIMIT_PER_MINUTE", "1")
    with api_module._RATE_LOCK:
        api_module._RATE_EVENTS.clear()
    client = TestClient(app)
    guest = api_module.IDENTITY.create_guest("rate-limit-test")
    client.cookies.set("baloot_session", guest["session"]["token"])
    assert client.get("/api/v1/plugins").status_code == 200
    limited = client.get("/api/v1/plugins")
    assert limited.status_code == 429 and int(limited.headers["retry-after"]) >= 1
    assert client.get("/healthz").status_code == 200
    metrics = client.get("/metrics")
    assert metrics.status_code == 200 and "baloot_http_requests_total" in metrics.text


def test_bundle_is_immutable_content_addressed_and_complete(root: Path):
    bundle = build_bundle(root)
    assert len(bundle.sha256) == 64
    expected = {"rules", "conventions", "analysis", "video", "game", "product", "security", "runtime", "strategy", "ml", "storage", "agents", "learning", "challenge", "history", "audit", "difficulty", "belief", "challenge_contracts", "database", "duplicate", "match_value", "identity", "shared", "live", "adjudication", "hitl", "certification", "benchmark", "operations", "configuration", "performance"}
    assert set(bundle.components) == expected
    assert bundle.rules.data["profile"]["status"] == "researched_default_unratified"
    assert bundle.video["renderer_plugin"] == "premium_video"


def test_zero_valued_enums_serialize_without_becoming_null(rules):
    from baloot_platform.analysis.models import HardDeduction, InformationState
    from baloot_platform.core.engine import BalootEngine
    from baloot_platform.core.models import BidRecord

    record = BidRecord(
        round_index=1,
        turn_index=0,
        seat=Seat.NORTH,
        action="hokum",
        contract=Contract.HOKUM,
        trump=Suit.SPADES,
        bidder=Seat.NORTH,
        buyer=Seat.NORTH,
    ).to_dict()
    assert record["trumpSuit"] == "spades"
    assert record["bidder"] == "north"
    assert record["buyer"] == "north"

    deduction = HardDeduction("void_suit", Seat.NORTH, Suit.SPADES, 1.0, "regression").to_dict()
    assert deduction["seat"] == "north" and deduction["suit"] == "spades"

    engine = BalootEngine(rules)
    state = engine.create_round(seed=211, dealer=Seat.SOUTH)
    engine.shuffle_and_initial_deal(state)
    assert engine.bid_and_complete_deal(
        state,
        forced_contract=Contract.HOKUM,
        forced_buyer=Seat.NORTH,
        forced_trump=Suit.SPADES,
    )
    assert state.buyer == Seat.NORTH
    assert state.trump_suit == Suit.SPADES
    engine.prepare_projects(state)
    info = InformationState.from_complete(state, state.current_player)
    assert info.to_dict()["trumpSuit"] == "spades"


def test_explicit_north_seat_is_not_replaced_by_current_player(rules):
    from baloot_platform.core.engine import BalootEngine

    engine = BalootEngine(rules)
    state = engine.create_round(seed=217, dealer=Seat.EAST)
    engine.shuffle_and_initial_deal(state)
    assert engine.bid_and_complete_deal(
        state,
        forced_contract=Contract.SUN,
        forced_buyer=Seat.SOUTH,
    )
    engine.prepare_projects(state)
    assert state.current_player == Seat.NORTH
    first = engine.legal_moves(state)[0]
    engine.play_card(state, first)
    assert state.current_player == Seat.WEST
    assert engine.legal_moves(state, Seat.NORTH) == ()
    assert engine.legal_moves(state, Seat.WEST)


def test_episode_exposes_all_purchase_and_contract_roles(rules, convention):
    data = _episode(rules, convention).to_dict()
    assert data["winningBidder"] in {"north", "west", "south", "east"}
    assert data["contractOwner"] == data["buyer"]
    assert data["purchaseRecipient"] == data["buyer"]
    assert data["roundResult"]["winner"] == data["roundScore"]["winner"]
    purchased = next(event for event in data["events"] if event["type"] == "contract_purchased")
    assert purchased["winningBidder"] == data["winningBidder"]
    assert purchased["contractOwner"] == data["contractOwner"]
    assert purchased["purchaseRecipient"] == data["purchaseRecipient"]
