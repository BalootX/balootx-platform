from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "scripts" / "verify_release_v20.py"
CMAKE = ROOT / "native" / "CMakeLists.txt"


def _load_verifier():
    spec = importlib.util.spec_from_file_location("verify_release_v20_portability", VERIFY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_asan_runtime_is_linked_once_without_ld_preload() -> None:
    verifier = VERIFY.read_text(encoding="utf-8")
    cmake = CMAKE.read_text(encoding="utf-8")

    assert "-DBALOOT_ASAN_RUNTIME=" not in verifier
    assert "gcc','-print-file-name=libasan.so" not in verifier
    assert "env={'LD_PRELOAD':None,'ASAN_OPTIONS':None}" in verifier
    assert "'orchestratorLdPreload':False" in verifier
    assert "'childLdPreload':False" in verifier
    assert "'asanRuntimeLinkedByCompiler':True" in verifier
    assert "'actualSanitizedBinaryProcess':True" in verifier
    assert "run_bounded(" in verifier
    assert "f'LD_PRELOAD=" not in verifier

    assert '"${CMAKE_COMMAND}" -E env' in cmake
    assert "BALOOT_ASAN_RUNTIME" not in cmake
    assert "LD_PRELOAD=" not in cmake
    assert '"$<TARGET_FILE:baloot_native_tests>"' in cmake
    assert "set_tests_properties(native PROPERTIES TIMEOUT 120" in cmake


def test_bounded_sanitizer_runner_caps_repeated_output() -> None:
    verifier = _load_verifier()
    result = verifier.run_bounded(
        "asan-output-bound-regression",
        [sys.executable, "-c", "import sys; sys.stdout.write('X' * 131072)"],
        cwd=ROOT,
        timeout=15,
        limit_bytes=4096,
    )
    assert result["status"] == "passed"
    assert result["stdoutTruncated"] is True
    assert result["stdoutBytesProduced"] == 131072
    assert len(result["stdout"].encode("utf-8")) <= 4096
    assert result["processGroupTerminationOnTimeout"] is True


def test_preflight_lane_timeout_is_bounded_and_configurable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BALOOT_VERIFY_PREFLIGHT_TIMEOUT_SECONDS", "211")

    verifier = _load_verifier()

    assert verifier.PREFLIGHT_TIMEOUT_SECONDS == 211


def _asan_artifact(filename: str, skip_reason: str) -> Path:
    configured = os.environ.get("BALOOT_ASAN_BUILD_DIR")
    build = Path(configured) if configured else ROOT / "build" / "native-asan"
    artifact = build / filename
    if artifact.is_file():
        return artifact
    if configured:
        pytest.fail(f"configured ASan build is missing {filename}")
    pytest.skip(skip_reason)


def test_generated_asan_ctest_invokes_sanitized_binary_without_preload() -> None:
    ctest_file = _asan_artifact(
        "CTestTestfile.cmake", "release verifier configures build/native-asan before the complete test lane"
    )

    generated = ctest_file.read_text(encoding="utf-8", errors="replace")
    assert "cmake" in generated
    assert "-E" in generated and "env" in generated
    assert "ASAN_OPTIONS=" in generated
    assert "LD_PRELOAD=" not in generated
    assert "baloot_native_tests" in generated

    env = os.environ.copy()
    env.pop("LD_PRELOAD", None)
    env.pop("ASAN_OPTIONS", None)
    probe = subprocess.run(
        ["ctest", "--test-dir", str(ctest_file.parent), "-N", "-V"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )
    assert "baloot_native_tests" in probe.stdout
    assert "LD_PRELOAD=" not in probe.stdout


def test_configured_sanitized_binary_passes_without_preload() -> None:
    executable = _asan_artifact(
        "baloot_native_tests", "release verifier configures build/native-asan before this test"
    )

    env = os.environ.copy()
    env.pop("LD_PRELOAD", None)
    env["ASAN_OPTIONS"] = (
        "detect_leaks=1:halt_on_error=1:abort_on_error=1:disable_coredump=1"
    )
    probe = subprocess.run(
        [str(executable)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )
    assert "tests passed" in probe.stdout
    assert "AddressSanitizer:DEADLYSIGNAL" not in probe.stderr
