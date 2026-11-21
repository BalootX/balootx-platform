#!/usr/bin/env python3
"""Verify the current BalootX local release candidate without rendering again."""

from __future__ import annotations

import argparse
import base64
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


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
from baloot_platform.filesystem import atomic_write_text  # noqa: E402

VERSION = str(
    tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]
)
REPORT = ROOT / "release/verification-report.json"
REMOTION = ROOT / "src/baloot_platform/video/remotion"
LOCK = ROOT / ".release-runtime/release/verify-release.lock"
PYTHON_SUITE_TIMEOUT_SECONDS = 7200


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def node_command(command: str) -> list[str]:
    if os.name != "nt":
        return ["bash", "-lc", command]
    wrapper = Path(os.environ.get("BALOOT_HIDDEN_NODE_WRAPPER", ""))
    if not wrapper.is_file():
        raise RuntimeError("the required hidden Node command wrapper is unavailable")
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
        base64.b64encode(command.encode("utf-8")).decode("ascii"),
        "-WorkingDirectory",
        str(REMOTION),
    ]


def _terminate_process_tree(
    process: subprocess.Popen[str],
) -> tuple[list[str], str, str]:
    errors: list[str] = []
    if process.poll() is not None:
        return errors, "", ""
    try:
        if os.name == "nt":
            killed = subprocess.run(
                ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if killed.returncode and process.poll() is None:
                errors.append(killed.stderr.strip() or killed.stdout.strip())
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        if process.poll() is None:
            process.kill()
    try:
        stdout, stderr = process.communicate(timeout=30)
    except subprocess.TimeoutExpired as exc:
        errors.append("owned process tree did not exit after termination")
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", errors="replace")
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    return errors, stdout, stderr


def _acquire_single_flight_lock():
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    handle = LOCK.open("a+b")
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError(
            "another verify_release.py process is already active"
        ) from exc
    return handle


def _pytest_failure_nodes(output: str) -> list[str]:
    nodes: list[str] = []
    for line in output.splitlines():
        for prefix in ("FAILED ", "ERROR "):
            if line.startswith(prefix):
                node = line[len(prefix) :].split(" - ", 1)[0].strip()
                if node and node not in nodes:
                    nodes.append(node)
    return nodes


def run(
    name: str,
    command: list[str],
    *,
    cwd: Path = ROOT,
    timeout: int = 1200,
) -> dict[str, Any]:
    started = time.perf_counter()
    environment = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "BALOOT_ROOT": str(ROOT),
        "SOURCE_DATE_EPOCH": "315532800",
    }
    print(f"[verify] START {name}", flush=True)
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=os.name != "nt",
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        cleanup_errors, cleanup_stdout, cleanup_stderr = _terminate_process_tree(
            process
        )
        stdout = cleanup_stdout or stdout
        stderr = cleanup_stderr or stderr
        failure_nodes = _pytest_failure_nodes(stdout)
        raise RuntimeError(
            json.dumps(
                {
                    "name": name,
                    "status": "failed",
                    "timedOut": True,
                    "timeoutSeconds": timeout,
                    "cleanupComplete": process.poll() is not None,
                    "cleanupErrors": cleanup_errors,
                    "failureNodes": failure_nodes,
                    "stdout": stdout[-20000:],
                    "stderr": stderr[-20000:],
                },
                ensure_ascii=False,
                indent=2,
            )
        ) from exc
    result = {
        "name": name,
        "status": "passed" if process.returncode == 0 else "failed",
        "returnCode": process.returncode,
        "durationSeconds": round(time.perf_counter() - started, 3),
        "stdout": stdout[-20000:],
        "stderr": stderr[-20000:],
    }
    failure_nodes = _pytest_failure_nodes(stdout)
    if failure_nodes:
        result["failureNodes"] = failure_nodes
    if process.returncode:
        raise RuntimeError(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[verify] PASS {name} ({result['durationSeconds']}s)", flush=True)
    return result


def _error_evidence(exc: Exception) -> dict[str, Any]:
    raw = str(exc)
    try:
        details = json.loads(raw)
    except json.JSONDecodeError:
        return {"type": type(exc).__name__, "message": raw[-20000:]}
    if not isinstance(details, dict):
        return {"type": type(exc).__name__, "message": raw[-20000:]}
    bounded = dict(details)
    for field in ("stdout", "stderr"):
        if isinstance(bounded.get(field), str):
            bounded[field] = bounded[field][-20000:]
    name = str(bounded.get("name") or "release check")
    status = str(bounded.get("status") or "failed")
    return {
        "type": type(exc).__name__,
        "message": f"{name}: {status}",
        "details": bounded,
    }


def agent_qualification_boundary(root: Path, bundle) -> dict[str, Any]:
    """Report the external expert gate without converting it into software proof."""

    path = root / "release/agent-qualification.json"
    if not path.is_file():
        return {
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

    from baloot_platform.agents.profiles import AgentProfileRegistry
    from baloot_platform.agents.qualification import FoundryAgentQualificationBundleV1

    benchmark = bundle.component("benchmark")
    policy_id = str(benchmark["paired_comparison"]["candidate_profile_id"])
    registry = AgentProfileRegistry.from_bundle(root)
    qualification = FoundryAgentQualificationBundleV1.load(
        path,
        profile=registry.get(policy_id),
        lineup_snapshot=registry.snapshot_lineup(
            {seat: policy_id for seat in ("north", "west", "south", "east")}
        ),
        benchmark_profile=benchmark,
    )
    release_decision = qualification.release
    return {
        "status": (
            "passed"
            if release_decision.expert_generation_allowed
            else "pending_external_review"
        ),
        "releaseStatus": release_decision.status,
        "benchmarkEvidenceComplete": release_decision.benchmark_evidence_complete,
        "professionalReviewComplete": release_decision.professional_review_complete,
        "expertGenerationAllowed": release_decision.expert_generation_allowed,
        "issues": list(release_decision.issues),
        "sha256": qualification.sha256,
    }


def artifact_check() -> dict[str, Any]:
    from baloot_platform.benchmark import canonical_sha256
    from baloot_platform.config import build_bundle
    from baloot_platform.research import ScenarioPredicateRegistryV1
    try:
        from scripts.generate_performance_evidence import git_identity
    except ModuleNotFoundError:
        from generate_performance_evidence import git_identity
    try:
        from scripts.generate_security_report import verify as verify_security
    except ModuleNotFoundError:
        from generate_security_report import verify as verify_security
    try:
        from scripts.check_release_readiness import (
            _evidence_artifact as admit_release_evidence,
        )
    except ModuleNotFoundError:
        from check_release_readiness import (
            _evidence_artifact as admit_release_evidence,
        )

    required = (
        "release/bounded-campaign-proof.json",
        "release/chrome-qa.json",
        "release/configuration-registry.json",
        "release/intelligence-benchmark.json",
        "release/media-probe.json",
        "release/performance-benchmark.json",
        "release/plugin-host.json",
        "release/property-check-professional-video.json",
        "release/research-scenario-v4-performance.json",
        "release/rules-certification-candidate.json",
        "release/search-performance.json",
        "release/security-report.json",
        "release/test-inventory.json",
        "release/visual-qa.json",
        "release/wheel-reproducibility.json",
        "outputs/professional-video-flagship/manifest.json",
        "outputs/professional-video-decision/manifest.json",
        "outputs/remotion-plugin-parity/parity-evidence.json",
    )
    records: dict[str, Any] = {}
    for relative in required:
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"missing release artifact: {relative}")
        value = json.loads(path.read_text(encoding="utf-8"))
        records[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        if relative.startswith("release/") and relative not in {
            "release/bounded-campaign-proof.json",
            "release/property-check-professional-video.json",
            "release/wheel-reproducibility.json",
        } and value.get("status") != "passed":
            raise RuntimeError(f"release artifact is not passed: {relative}")
        if relative in {
            "release/performance-benchmark.json",
            "release/research-scenario-v4-performance.json",
            "release/search-performance.json",
        }:
            payload = {
                key: item for key, item in value.items() if key != "evidenceSha256"
            }
            if value.get("evidenceSha256") != canonical_sha256(payload):
                raise RuntimeError(f"release artifact hash is invalid: {relative}")
    for relative in (
        "release/configuration-registry.json",
        "release/intelligence-benchmark.json",
        "release/media-probe.json",
        "release/rules-certification-candidate.json",
    ):
        admission = admit_release_evidence(ROOT, relative)
        if admission.get("status") != "passed":
            raise RuntimeError(
                f"release authority evidence is stale or invalid: {relative}"
            )
    bundle = build_bundle(ROOT)
    ortools_bundle = build_bundle(
        ROOT,
        bundle_path="configs/bundle_ortools_research_v4.yaml",
    )
    search_performance = json.loads(
        (ROOT / "release/search-performance.json").read_text(encoding="utf-8")
    )
    research_performance = json.loads(
        (ROOT / "release/research-scenario-v4-performance.json").read_text(
            encoding="utf-8"
        )
    )
    performance = json.loads(
        (ROOT / "release/performance-benchmark.json").read_text(encoding="utf-8")
    )
    security = json.loads(
        (ROOT / "release/security-report.json").read_text(encoding="utf-8")
    )
    current_source = git_identity(ROOT)
    current_components = {
        family: component.sha256
        for family, component in sorted(bundle.components.items())
    }
    research_bundle = research_performance.get("bundle") or {}
    if (
        canonical_sha256(performance.get("source") or {})
        != canonical_sha256(current_source)
        or performance.get("bundleSha256") != bundle.sha256
        or performance.get("rulesSha256") != bundle.rules.sha256
        or canonical_sha256(search_performance.get("source") or {})
        != canonical_sha256(current_source)
        or search_performance.get("engine")
        != {
            "bundleSha256": ortools_bundle.sha256,
            "rulesSha256": ortools_bundle.rules.sha256,
        }
        or canonical_sha256(research_performance.get("source") or {})
        != canonical_sha256(current_source)
        or research_bundle.get("sha256") != bundle.sha256
        or research_bundle.get("rulesSha256") != bundle.rules.sha256
        or research_bundle.get("componentSha256") != current_components
        or performance.get("search", {}).get("evidenceSha256")
        != search_performance.get("evidenceSha256")
        or performance.get("researchV4", {}).get("evidenceSha256")
        != research_performance.get("evidenceSha256")
        or performance.get("researchV4", {}).get("bundle")
        != research_performance.get("bundle")
    ):
        raise RuntimeError("current-source performance evidence is stale or invalid")
    try:
        verify_security(security, root=ROOT)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "current-source security evidence is stale or invalid"
        ) from exc
    campaign_proof = json.loads(
        (ROOT / "release/bounded-campaign-proof.json").read_text(encoding="utf-8")
    )["document"]
    agent_gate = agent_qualification_boundary(ROOT, bundle)
    if (
        agent_gate["status"] == "pending_external_review"
        and agent_gate["expertGenerationAllowed"] is not False
    ):
        raise RuntimeError("pending agent qualification must disable expert generation")
    executions = campaign_proof.get("executions") or []
    proof_profile = bundle.component("challenge")["campaign_proof"]
    if (
        campaign_proof.get("status") != "qualified"
        or campaign_proof.get("unmetQualificationGates")
        or len(executions) != max(
            len(proof_profile["requiredWorkerCounts"]),
            len(proof_profile["requiredFailureSchedules"]),
        )
        or {item.get("declaredWorkerCount") for item in executions}
        != set(proof_profile["requiredWorkerCounts"])
        or {item.get("failureSchedule") for item in executions}
        != set(proof_profile["requiredFailureSchedules"])
        or any(
            item.get("organicRoundCount")
            != int(proof_profile["minimumOrganicRounds"])
            for item in executions
        )
        or any(item.get("failureScheduleAttested") is not True for item in executions)
        or len({item.get("normalizedEvidenceHash") for item in executions}) != 1
    ):
        raise RuntimeError("bounded campaign equivalence proof is stale or invalid")
    wheel = json.loads((ROOT / "release/wheel-reproducibility.json").read_text(encoding="utf-8"))
    wheel_path = ROOT / wheel["wheel"]
    builds = wheel.get("builds") or []
    installed_runtime = wheel.get("installedRuntime") or {}
    scenario_catalog = ScenarioPredicateRegistryV1().catalog()
    if (
        wheel.get("status") != "passed"
        or canonical_sha256(wheel.get("source") or {})
        != canonical_sha256(current_source)
        or wheel.get("byteIdentical") is not True
        or len(builds) != 2
        or builds[0] != builds[1]
        or not wheel_path.is_file()
        or sha256(wheel_path) != builds[0]["sha256"]
        or installed_runtime.get("scenarioPredicateCount")
        != len(scenario_catalog["definitions"])
        or installed_runtime.get("scenarioPredicateCatalogSha256")
        != scenario_catalog["sha256"]
    ):
        raise RuntimeError("exact wheel reproducibility evidence is stale or invalid")
    plugins = json.loads((ROOT / "release/plugin-host.json").read_text(encoding="utf-8"))["plugins"]
    video_plugins = [item for item in plugins if item["kind"] == "video_renderer"]
    if [item["id"] for item in video_plugins] != ["premium_video"]:
        raise RuntimeError("premium_video is not the sole video renderer")
    return {
        "name": "release-artifacts",
        "status": "passed",
        "files": records,
        "externalGates": {"agentQualification": agent_gate},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-python-suite", action="store_true")
    args = parser.parse_args()
    _single_flight_lock = _acquire_single_flight_lock()
    checks: list[dict[str, Any]] = []
    try:
        checks.append(
            run(
                "search-performance",
                [
                    sys.executable,
                    "scripts/benchmark_search.py",
                    "--output",
                    "release/search-performance.json",
                ],
                timeout=600,
            )
        )
        checks.append(
            run(
                "research-v4-performance",
                [
                    sys.executable,
                    "scripts/benchmark_research_scenario_v4.py",
                    "--output",
                    "release/research-scenario-v4-performance.json",
                ],
                timeout=600,
            )
        )
        if not args.skip_python_suite:
            checks.append(
                run(
                    "python-suite",
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-vv",
                        "-r",
                        "fE",
                        "--color=no",
                        "--override-ini=addopts=",
                    ],
                    timeout=PYTHON_SUITE_TIMEOUT_SECONDS,
                )
            )
        checks.append(
            run(
                "test-inventory",
                [
                    sys.executable,
                    "scripts/generate_test_inventory_v20.py",
                    "--version",
                    VERSION,
                    "--output",
                    "release/test-inventory.json",
                ],
            )
        )
        checks.append(run("remotion-typecheck", node_command("npm.cmd run typecheck" if os.name == "nt" else "npm run typecheck"), cwd=REMOTION, timeout=300))
        for name, script in (
            ("professional-prototypes", "verify_professional_video_prototypes.py"),
            ("professional-flagship", "verify_professional_flagship.py"),
            ("professional-decision", "verify_professional_decision_clip.py"),
            ("remotion-plugin-parity", "verify_remotion_plugin_parity.py"),
            ("media-evidence", "generate_media_probe.py"),
        ):
            checks.append(run(name, [sys.executable, f"scripts/{script}"]))
        checks.append(
            run(
                "scene-preview-refresh",
                [sys.executable, "scripts/generate_scene_preview.py", "--refresh"],
            )
        )
        checks.append(
            run(
                "scene-preview-check",
                [sys.executable, "scripts/generate_scene_preview.py", "--check"],
            )
        )
        checks.append(run("visual-evidence", [sys.executable, "scripts/generate_visual_qa.py"]))
        checks.append(
            run(
                "performance-evidence",
                [sys.executable, "scripts/generate_performance_evidence.py"],
            )
        )
        configured_contract = os.environ.get("BALOOT_DEPLOYMENT_CONTRACT_DIR", "").strip()
        if not configured_contract:
            raise RuntimeError("private deployment contract directory must be supplied explicitly")
        contract_root = Path(configured_contract).resolve(strict=True)
        checks.append(run("compose-contract", ["docker", "compose", "-f", str(contract_root / "docker-compose.yml"), "-f", str(contract_root / "docker-compose.production.yml"), "config", "--quiet"], timeout=60))
        checks.append(artifact_check())
        report = {
            "schemaVersion": 1,
            "project": "baloot-platform",
            "version": VERSION,
            "status": "passed",
            "checks": checks,
            "proofBoundary": (
                "Local source, exact media, plugin parity, browser evidence, and exact wheel passed. "
                "Container execution, independent permanent-holdout adjudication, professional "
                "agent review, Saudi rules ratification, and owner media acceptance remain "
                "separate gates. Expert generation stays disabled until its external evidence "
                "is complete."
            ),
        }
    except Exception as exc:
        error = _error_evidence(exc)
        failed_check = error.get("details")
        if isinstance(failed_check, dict) and {
            "name",
            "status",
        } <= failed_check.keys():
            checks.append(failed_check)
        report = {
            "schemaVersion": 1,
            "project": "baloot-platform",
            "version": VERSION,
            "status": "failed",
            "checks": checks,
            "error": error,
        }
    atomic_write_text(
        REPORT,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _single_flight_lock.close()
    print(REPORT)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
