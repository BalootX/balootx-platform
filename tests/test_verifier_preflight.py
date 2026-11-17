from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import pytest

from scripts import build_reproducible_wheel, verifier_preflight, verify_archive
from baloot_platform.version import __version__
from baloot_platform.video.remotion_renderer import _process_exists


ROOT = Path(__file__).resolve().parents[1]


def test_container_suite_timeout_is_explicit_and_bounded(monkeypatch) -> None:
    monkeypatch.delenv("BALOOT_VERIFIER_TEST_TIMEOUT_SECONDS", raising=False)
    with pytest.raises(RuntimeError, match="is required"):
        verifier_preflight.container_suite_timeout_seconds()

    for value in ("invalid", "299", "3601"):
        monkeypatch.setenv("BALOOT_VERIFIER_TEST_TIMEOUT_SECONDS", value)
        with pytest.raises(RuntimeError, match="must be"):
            verifier_preflight.container_suite_timeout_seconds()

    monkeypatch.setenv("BALOOT_VERIFIER_TEST_TIMEOUT_SECONDS", "1800")
    assert verifier_preflight.container_suite_timeout_seconds() == 1800


def test_timing_sensitive_exact_oracle_is_run_once_in_isolation() -> None:
    suites = verifier_preflight.container_suite_commands()

    assert len(suites) == verifier_preflight.CONTAINER_SUITE_SHARD_COUNT == 3
    assert verifier_preflight.ISOLATED_CONTAINER_TESTS == (
        "tests/test_exact_holdout_labels.py::test_exact_holdout_labels_replay_and_preserve_all_optimal_ties",
        "tests/test_search_performance.py::test_v21_search_benchmark_is_reproducible_and_state_safe",
    )
    for selection in verifier_preflight.ISOLATED_CONTAINER_TESTS:
        isolated = verifier_preflight.isolated_container_test_command(selection)
        assert sum(f"--deselect={selection}" in suite for suite in suites) == 1
        assert isolated[0:4] == [
            verifier_preflight.sys.executable,
            "-m",
            "pytest",
            "-q",
        ]
        assert isolated.count(selection) == 1

    with pytest.raises(ValueError, match="not governed"):
        verifier_preflight.isolated_container_test_command("tests/test_other.py")


def test_container_suite_shards_cover_every_file_once_with_verbose_diagnostics() -> None:
    suites = verifier_preflight.container_suite_commands()
    selected = [
        argument
        for suite in suites
        for argument in suite
        if argument.startswith("tests/test_") and "::" not in argument
    ]
    excluded = {
        selection
        for selection in verifier_preflight.SOURCE_ONLY_TESTS
        if "::" not in selection
    }
    expected = {
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in (ROOT / "tests").glob("test_*.py")
        if str(path.relative_to(ROOT)).replace("\\", "/") not in excluded
    }

    assert len(selected) == len(set(selected))
    assert set(selected) == expected
    for suite in suites:
        assert "-vv" in suite
        assert "--override-ini=addopts=" in suite
        assert "--durations=25" in suite


def test_verifier_timeout_preserves_progress_and_terminates_owned_tree(
    tmp_path: Path,
) -> None:
    child_pid_path = tmp_path / "child.pid"
    child_script = tmp_path / "child.py"
    parent_script = tmp_path / "parent.py"
    child_script.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    parent_script.write_text(
        "import pathlib, subprocess, sys, time\n"
        f"child = subprocess.Popen([sys.executable, '-B', {str(child_script)!r}])\n"
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid), encoding='utf-8')\n"
        "print('CURRENT_TEST=tests/test_probe.py::test_timeout', flush=True)\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )

    with pytest.raises(verifier_preflight.CheckFailure) as failure:
        verifier_preflight.run(
            "container-timeout-probe",
            [sys.executable, "-B", str(parent_script)],
            cwd=tmp_path,
            timeout=1,
        )
    result = failure.value.result
    assert result["timedOut"] is True
    assert result["returnCode"] == 124
    assert result["cleanupComplete"] is True
    assert result["cleanupErrors"] == []
    assert "CURRENT_TEST=tests/test_probe.py::test_timeout" in result["stdout"]

    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if not _process_exists(child_pid):
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"owned timeout child remained alive: {child_pid}")


def test_verifier_collects_the_complete_shard_failure_wave(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "verifier-preflight.json"
    monkeypatch.setattr(verifier_preflight, "OUTPUT", output)
    monkeypatch.setattr(
        verifier_preflight,
        "container_suite_timeout_seconds",
        lambda: 600,
    )
    monkeypatch.setattr(
        verifier_preflight,
        "authority_check",
        lambda: {"name": "authorities", "status": "passed"},
    )
    monkeypatch.setattr(
        verifier_preflight,
        "container_suite_commands",
        lambda: (["shard-1"], ["shard-2"], ["shard-3"]),
    )
    seen_shards: list[str] = []

    def fake_run(name: str, _command: list[str], **_kwargs):
        result = {"name": name, "status": "passed", "returnCode": 0}
        if name.startswith("container-python-suite-"):
            seen_shards.append(name)
            if name.endswith(("1-of-3", "3-of-3")):
                result |= {"status": "failed", "returnCode": 1}
                raise verifier_preflight.CheckFailure(result)
        return result

    monkeypatch.setattr(verifier_preflight, "run", fake_run)

    with pytest.raises(SystemExit):
        verifier_preflight.main()

    assert seen_shards == [
        "container-python-suite-1-of-3",
        "container-python-suite-2-of-3",
        "container-python-suite-3-of-3",
    ]
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    failed = [check["name"] for check in report["checks"] if check["status"] == "failed"]
    assert failed == [
        "container-python-suite-1-of-3",
        "container-python-suite-3-of-3",
    ]




def test_reproducible_wheel_help_exits_before_building() -> None:
    completed = subprocess.run(
        [sys.executable, "-B", "scripts/build_reproducible_wheel.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0
    assert "usage:" in completed.stdout


@pytest.mark.skipif(os.name != "nt", reason="Windows PATH limit regression")
def test_archive_node_checks_use_a_bounded_runtime_path() -> None:
    environment = verify_archive._node_environment()
    paths = environment["PATH"].split(os.pathsep)

    assert len(paths) <= 6
    assert len(environment["PATH"]) < 1024
    assert str(Path(verify_archive.shutil.which("node")).parent) in paths
    assert str(Path(verify_archive.shutil.which("npm.cmd")).parent) in paths
    assert str(Path(sys.executable).parent) in paths


def test_archive_runner_decodes_utf8_diagnostics(tmp_path: Path) -> None:
    result = verify_archive._run_complete(
        "utf8-probe",
        [sys.executable, "-c", "print('وش تلعب؟')"],
        cwd=tmp_path,
        environment={"PYTHONIOENCODING": "utf-8"},
        timeout=10,
    )

    assert result["status"] == "passed"
    assert result["stdout"] == "وش تلعب؟\n"


def test_reproducible_wheel_install_ignores_ambient_source_tree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured.update(command=command, **kwargs)

    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src"))
    monkeypatch.setenv("BALOOT_ROOT", str(ROOT))
    monkeypatch.setattr(build_reproducible_wheel, "_run", fake_run)

    python = tmp_path / "python"
    wheel = tmp_path / "baloot_platform.whl"
    build_reproducible_wheel._install_wheel(python, wheel, cwd=tmp_path)

    assert captured["cwd"] == tmp_path
    assert captured["environment"] == {"PYTHONPATH": "", "BALOOT_ROOT": ""}
    assert captured["command"] == [
        str(python),
        "-m",
        "pip",
        "install",
        str(wheel),
    ]


def test_reproducible_wheel_probe_preserves_failure_output(capsys) -> None:
    completed = subprocess.CompletedProcess(
        ["python", "-c", "probe"],
        1,
        stdout="probe stdout\n",
        stderr="probe stderr\n",
    )

    with pytest.raises(subprocess.CalledProcessError):
        build_reproducible_wheel._probe_stdout(completed)

    output = capsys.readouterr()
    assert output.out == "probe stdout\n"
    assert output.err == "probe stderr\n"


def test_reproducible_wheel_probe_uses_registry_as_predicate_inventory() -> None:
    source = (ROOT / "scripts" / "build_reproducible_wheel.py").read_text(
        encoding="utf-8"
    )
    assert "assert len(catalog['definitions'])==" not in source
    assert "'scenarioPredicateCount':len(catalog['definitions'])" in source


def test_reproducible_wheel_uses_canonical_migration_hash() -> None:
    assert build_reproducible_wheel.MIGRATION_028 == (
        ROOT / "migrations/028_research_authority_projection.sql"
    )
    assert "MIGRATION_028_SHA256" not in (
        ROOT / "scripts/build_reproducible_wheel.py"
    ).read_text(encoding="utf-8")


def test_archive_rejects_wheel_evidence_from_another_source(tmp_path: Path) -> None:
    wheel = tmp_path / "dist/baloot_platform.whl"
    wheel.parent.mkdir()
    wheel.write_bytes(b"wheel")
    identity = {
        "bytes": wheel.stat().st_size,
        "sha256": verify_archive.sha256(wheel),
    }
    report = tmp_path / "release/wheel-reproducibility.json"
    report.parent.mkdir()
    report.write_text(
        json.dumps(
            {
                "status": "passed",
                "source": {"revision": "a" * 40},
                "byteIdentical": True,
                "builds": [identity, identity],
                "wheel": "dist/baloot_platform.whl",
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "source": {"revision": "a" * 40},
        "files": {
            "release/wheel-reproducibility.json": {},
            "dist/baloot_platform.whl": identity,
        },
    }
    assert verify_archive._verify_packaged_wheel(tmp_path, manifest)["status"] == "passed"

    manifest["source"] = {"revision": "b" * 40}
    with pytest.raises(
        verify_archive.ArchiveVerificationError,
        match="wheel evidence is stale or invalid",
    ):
        verify_archive._verify_packaged_wheel(tmp_path, manifest)


@pytest.mark.parametrize(
    "script",
    ("scripts/generate_security_report.py", "scripts/check_release_readiness.py"),
)
def test_release_evidence_help_works_without_pythonpath(script: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-B", script, "--help"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": ""},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0
    assert "usage:" in completed.stdout


def test_verifier_delivery_uses_current_version_and_retries_apt() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    project_version = project["version"]
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    remotion_root = ROOT / "src/baloot_platform/video/remotion"
    remotion_package = (remotion_root / "package.json").read_text(encoding="utf-8")
    remotion_lock = json.loads(
        (remotion_root / "package-lock.json").read_text(encoding="utf-8")
    )

    assert __version__ == project_version
    assert "ortools==9.15.6755" in project["dependencies"]
    assert '"release": __version__' in (
        ROOT / "scripts/verifier_preflight.py"
    ).read_text(encoding="utf-8")
    assert 'Acquire::Retries "5"' in dockerfile
    assert dockerfile.count("target=/var/cache/apt,sharing=locked") == 4
    assert dockerfile.count(
        "id=balootx-remotion-npm,target=/root/.npm,sharing=locked"
    ) == 2
    assert dockerfile.count(
        "id=balootx-foundry-npm,target=/root/.npm,sharing=locked"
    ) == 1
    assert "--mount=type=cache,target=/root/.npm" not in dockerfile
    assert "security,optimization]" in dockerfile
    assert "FROM remotion-build AS foundry-build" in dockerfile
    build_stage = dockerfile.split("FROM python-base AS build", 1)[1].split(
        "FROM python-base AS runtime", 1
    )[0]
    verifier_stage = dockerfile.split("FROM python-base AS verifier", 1)[1]
    assert " cmake ninja-build git ffmpeg " in build_stage
    assert (
        "apt-get install -y --no-install-recommends git"
        in verifier_stage.split("COPY --from=build", 1)[0]
    )
    runtime_stage = dockerfile.split("FROM python-base AS runtime", 1)[1].split(
        "FROM production-runtime AS video-worker", 1
    )[0]
    assert "--no-install-recommends git" not in runtime_stage
    assert (
        "COPY --from=foundry-build /foundry/node_modules "
        "./src/baloot_platform/service/foundry_dashboard/node_modules"
    ) in dockerfile
    assert "pip install --no-cache-dir pytest" not in dockerfile
    assert '"nanoid": "3.3.18"' in remotion_package
    assert remotion_lock["packages"]["node_modules/nanoid"]["version"] == "3.3.18"
    preflight = (ROOT / "scripts/verifier_preflight.py").read_text(encoding="utf-8")
    assert '"foundry-check"' in preflight
    assert '"foundry-typecheck"' in preflight
    assert '"foundry-research-tests"' in preflight
    archive = (ROOT / "scripts/verify_archive.py").read_text(encoding="utf-8")
    assert '"foundry-install-and-check"' in archive
    assert '"foundry-research-tests"' in archive
    assert "npm run test:research" in archive
