from __future__ import annotations

import json
import os
import subprocess
import sys

from baloot_platform.benchmark import canonical_sha256
from baloot_platform.release_scope import (
    SOURCE_EXCLUDED_PATHS,
    SOURCE_SCOPE,
)
from scripts import benchmark_search, generate_performance_evidence


def _run_benchmark_child(root):
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            (
                "import json,sys;from pathlib import Path;"
                "from scripts.benchmark_search import run_benchmark;"
                "print(json.dumps(run_benchmark(Path(sys.argv[1])),sort_keys=True))"
            ),
            str(root),
        ],
        cwd=root,
        env={
            **os.environ,
            "PYTHONPATH": str(root / "src"),
        },
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return json.loads(completed.stdout)


def test_v21_search_benchmark_is_reproducible_and_state_safe(root):
    report = _run_benchmark_child(root)
    repeated = _run_benchmark_child(root)
    for current in (report, repeated):
        crossings = [
            (
                item["name"],
                item["p95Ms"],
                item["budgetMs"],
                item["passed"],
            )
            for item in current["measurements"]
            if not item["passed"]
        ]
        assert current["status"] == "passed", crossings
    assert report["evidenceSha256"] == canonical_sha256(
        {
            key: value
            for key, value in report.items()
            if key != "evidenceSha256"
        }
    )
    assert report["source"]["binding"] == "unbound_test_invocation"
    assert report["source"]["scope"] == SOURCE_SCOPE
    assert report["source"]["excludedPaths"] == list(SOURCE_EXCLUDED_PATHS)
    assert report["sourceFixture"]["stateSha256"]
    assert report["engine"]["bundleSha256"]
    assert report["researchAuthority"] == {
        "backendId": "ortools_cp_sat_v2",
        "qualificationAuthority": "cp_sat",
        "bundleSource": "configs/bundle_ortools_research_v4.yaml",
    }
    assert report["searchBudget"]["maxNodes"] > 0
    assert report["exactSearchObservation"]["completionStatus"] == "exact"
    assert report["exactSearchObservation"]["nodes"] > 0
    assert report["exactSearchObservation"]["principalVariation"]
    identity = report["reproducibility"]["semanticIdentity"]
    assert identity["initialStateSha256"] == identity["finalStateSha256"]
    assert report["reproducibility"]["semanticIdentitySha256"] == (
        canonical_sha256(identity)
    )
    reproducibility_receipt = {
        key: value
        for key, value in report["reproducibility"].items()
        if key != "sha256"
    }
    assert report["reproducibility"]["sha256"] == canonical_sha256(
        reproducibility_receipt
    )
    assert repeated["reproducibility"]["semanticIdentitySha256"] == report[
        "reproducibility"
    ]["semanticIdentitySha256"]
    assert report["promotionEvaluation"]["decision"] == "not_evaluated"
    assert report["throughput"]["reversibleApplyUndoPerSecond"] > 0
    assert report["throughput"]["exactFourCardSearchesPerSecond"] > 0
    assert all(
        item["budgetMs"] is not None
        for item in report["measurements"]
    )
    assert report["runtimeScope"]["measurementClock"] == (
        "time.perf_counter_ns_elapsed_v1"
    )
    assert report["runtimeScope"]["schedulerDelayIncluded"] is True
    assert report["runtimeScope"]["cgroup"]["status"] in {
        "measured",
        "not_applicable",
        "unavailable",
    }
    assert "deployment benchmark" in report["proofBoundary"]


def test_search_benchmark_uses_canonical_dirty_snapshot_identity(
    tmp_path,
    monkeypatch,
):
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("first\n", encoding="utf-8")
    for relative in ("release/search-performance.json", "outputs/video.mp4"):
        generated = tmp_path / relative
        generated.parent.mkdir(parents=True, exist_ok=True)
        generated.write_text("first\n", encoding="utf-8")

    def run(command, **_kwargs):
        arguments = tuple(command[1:])
        if arguments == ("rev-parse", "HEAD"):
            output = ("a" * 40 + "\n").encode()
        elif arguments == ("rev-parse", "HEAD^{tree}"):
            output = ("b" * 40 + "\n").encode()
        elif arguments[0] == "status":
            output = b" M tracked.txt\0"
        elif arguments[:2] == ("ls-files", "-z"):
            output = b"tracked.txt\0"
        elif arguments[:3] == ("ls-files", "-s", "-z"):
            output = (f"100644 {'c' * 40} 0\ttracked.txt\0").encode()
        else:
            raise AssertionError(arguments)
        return subprocess.CompletedProcess(command, 0, stdout=output)

    monkeypatch.setattr(generate_performance_evidence.subprocess, "run", run)
    first = benchmark_search.git_identity(tmp_path)
    assert first["binding"] == "git_head_plus_worktree_snapshot"

    (tmp_path / "release/search-performance.json").write_text(
        "changed\n",
        encoding="utf-8",
    )
    (tmp_path / "outputs/video.mp4").write_text("changed\n", encoding="utf-8")
    assert benchmark_search.git_identity(tmp_path) == first

    tracked.write_text("changed\n", encoding="utf-8")
    assert benchmark_search.git_identity(tmp_path)["worktreeSnapshot"][
        "sha256"
    ] != first["worktreeSnapshot"]["sha256"]


def test_performance_measurements_report_tail_and_warmup():
    from baloot_platform.performance import PerformanceService

    service = PerformanceService(
        {
            "profile": {"id": "v21-test"},
            "measurement": {
                "warmup_iterations": 1,
                "measured_iterations": 3,
                "percentile": 95,
            },
            "budgets": {"noop": 100.0},
        }
    )
    measurement = service.measure("noop", lambda: None)
    payload = measurement.to_dict()
    assert payload["warmupIterations"] == 1
    assert payload["p99Ms"] >= payload["p95Ms"]
    assert payload["meanMs"] >= 0
    assert "samplesMs" not in payload

    retained = service.measure(
        "noop",
        lambda: None,
        iterations=3,
        warmup=0,
        retain_samples=True,
    ).to_dict()
    assert len(retained["samplesMs"]) == 3
