#!/usr/bin/env python3
"""Measure the reversible kernel and deterministic exact-search path."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
from typing import Any

from baloot_platform.analysis.search import exact_team_value
from baloot_platform.benchmark import canonical_sha256
from baloot_platform.config import build_bundle
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Seat
from baloot_platform.core.scenario import ScenarioRequest
from baloot_platform.filesystem import atomic_write_text
from baloot_platform.performance import PerformanceService
from baloot_platform.release_scope import (
    SOURCE_EXCLUDED_PATHS,
    SOURCE_SCOPE,
)
try:
    from scripts.generate_performance_evidence import git_identity
except ModuleNotFoundError:  # direct ``python scripts/...`` execution
    from generate_performance_evidence import git_identity

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("release/search-performance.json")
ORTOOLS_RESEARCH_BUNDLE = "configs/bundle_ortools_research_v4.yaml"


def _read_optional_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="ascii").strip() or None
    except (OSError, UnicodeError):
        return None


def _cgroup_scope(root: Path = Path("/sys/fs/cgroup")) -> dict[str, Any]:
    if platform.system() != "Linux":
        return {
            "status": "not_applicable",
            "version": None,
            "reason": "cgroups_are_linux_specific",
        }
    if not (root / "cgroup.controllers").is_file():
        return {
            "status": "unavailable",
            "version": None,
            "reason": "cgroup_v2_not_detected",
        }

    cpu_max = _read_optional_text(root / "cpu.max")
    quota_micros: int | None = None
    period_micros: int | None = None
    quota_cpu_count: float | None = None
    cpu_status = "unavailable"
    if cpu_max is not None:
        parts = cpu_max.split()
        if len(parts) == 2:
            try:
                period_micros = int(parts[1])
                if parts[0] == "max":
                    cpu_status = "unlimited"
                else:
                    quota_micros = int(parts[0])
                    if quota_micros >= 0 and period_micros > 0:
                        quota_cpu_count = quota_micros / period_micros
                        cpu_status = "limited"
            except ValueError:
                pass

    memory_max = _read_optional_text(root / "memory.max")
    memory_max_bytes: int | None = None
    memory_status = "unavailable"
    if memory_max == "max":
        memory_status = "unlimited"
    elif memory_max is not None:
        try:
            memory_max_bytes = int(memory_max)
            memory_status = "limited" if memory_max_bytes >= 0 else "unavailable"
        except ValueError:
            pass

    return {
        "status": "measured",
        "version": 2,
        "cpu": {
            "status": cpu_status,
            "quotaMicros": quota_micros,
            "periodMicros": period_micros,
            "quotaCpuCount": quota_cpu_count,
        },
        "cpusetCpusEffective": _read_optional_text(
            root / "cpuset.cpus.effective"
        ),
        "memory": {
            "status": memory_status,
            "maximumBytes": memory_max_bytes,
        },
    }


def _runtime_scope() -> dict[str, Any]:
    affinity: list[int] | None = None
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = sorted(int(value) for value in os.sched_getaffinity(0))
        except OSError:
            affinity = None
    return {
        "schemaVersion": 1,
        "processModel": "single_process_serial_v1",
        "measurementClock": "time.perf_counter_ns_elapsed_v1",
        "schedulerDelayIncluded": True,
        "processCpuAffinity": {
            "status": "measured" if affinity is not None else "unavailable",
            "logicalCpuIds": affinity,
            "logicalCpuCount": len(affinity) if affinity is not None else None,
        },
        "cgroup": _cgroup_scope(),
    }


def _comparison(
    current: dict[str, Any],
    baseline: dict[str, Any],
    *,
    baseline_revision: str,
    baseline_sha256: str,
) -> dict[str, Any]:
    current_by_name = {
        str(item["name"]): item
        for item in current["measurements"]
    }
    baseline_by_name = {
        str(item["name"]): item
        for item in baseline["measurements"]
    }
    metrics = {}
    for name in sorted(set(current_by_name) & set(baseline_by_name)):
        current_item = current_by_name[name]
        baseline_item = baseline_by_name[name]
        baseline_median = max(
            float(baseline_item["medianMs"]),
            1e-12,
        )
        baseline_p95 = max(float(baseline_item["p95Ms"]), 1e-12)
        metrics[name] = {
            "baselineMedianMs": float(baseline_item["medianMs"]),
            "currentMedianMs": float(current_item["medianMs"]),
            "medianImprovementPercent": round(
                (
                    1.0
                    - float(current_item["medianMs"])
                    / baseline_median
                )
                * 100.0,
                4,
            ),
            "baselineP95Ms": float(baseline_item["p95Ms"]),
            "currentP95Ms": float(current_item["p95Ms"]),
            "p95ImprovementPercent": round(
                (
                    1.0
                    - float(current_item["p95Ms"])
                    / baseline_p95
                )
                * 100.0,
                4,
            ),
        }
    return {
        "status": "local_same_hardware_comparison",
        "baselineRevision": baseline_revision,
        "baselineArtifactSha256": baseline_sha256,
        "metrics": metrics,
        "automaticPromotion": False,
    }


def _fixture_module(root: Path):
    path = root / "scripts/generate_exact_endgames.py"
    spec = importlib.util.spec_from_file_location(
        "generate_exact_endgames_benchmark",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load exact endgame generator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_benchmark(
    root: Path = ROOT,
    *,
    iterations: int | None = None,
    warmup: int | None = None,
    source_identity: dict[str, Any] | None = None,
    retain_samples: bool = False,
    baseline: dict[str, Any] | None = None,
    baseline_revision: str | None = None,
    baseline_sha256: str | None = None,
) -> dict[str, Any]:
    bundle = build_bundle(root, bundle_path=ORTOOLS_RESEARCH_BUNDLE)
    service = PerformanceService(bundle.component("performance"))
    engine = BalootEngine(bundle.rules)
    fixture = json.loads(
        (root / "tests/fixtures/synthetic-exact-endgames.json").read_text(
            encoding="utf-8"
        )
    )
    case = min(
        fixture["cases"],
        key=lambda item: (int(item["remainingCards"]), str(item["id"])),
    )
    state = _fixture_module(root).prepare(engine, case)
    original_hash = engine.state_hash(state)
    legal = engine.legal_actions(state)
    if not legal:
        raise RuntimeError("benchmark fixture has no legal action")
    request = ScenarioRequest.governed()
    search_observation: dict[str, Any] = {}
    search_observation_count = 0

    def reversible_roundtrip() -> None:
        token = engine.apply(
            state,
            legal[0],
            verify_identity=False,
        )
        engine.undo(state, token)

    def exact_search() -> None:
        nonlocal search_observation_count
        result = exact_team_value(
            state,
            engine,
            Seat.SOUTH.team,
            max_nodes=request.exact_search_max_nodes,
            max_seconds=request.exact_search_max_seconds,
            max_depth=request.exact_search_max_depth,
            max_cache_entries=request.exact_search_max_cache_entries,
        )
        if result.completion_status != "exact":
            raise RuntimeError(
                f"exact-search benchmark was {result.completion_status}"
            )
        observation = {
            "completionStatus": result.completion_status,
            "nodes": result.nodes,
            "principalVariation": list(result.principal_variation),
            "principalVariationLength": len(result.principal_variation),
            "value": result.value,
            "ownPoints": result.own_points,
            "opponentPoints": result.opponent_points,
            "contractSucceeded": result.contract_succeeded,
            "roundLead": result.round_lead,
            "scoreAfterRound": list(result.score_after_round),
            "matchWinner": (
                result.match_winner.name.lower()
                if result.match_winner is not None
                else None
            ),
        }
        if search_observation and observation != search_observation:
            raise RuntimeError("exact-search benchmark observation drifted")
        search_observation.update(observation)
        search_observation_count += 1

    reversible = service.measure(
        "reversible_apply_undo",
        reversible_roundtrip,
        iterations=iterations,
        warmup=warmup,
        budget_key="apply_action_p95_ms",
        retain_samples=retain_samples,
    )
    search = service.measure(
        "exact_search_four_card",
        exact_search,
        iterations=iterations,
        warmup=warmup,
        budget_key="shallow_analysis_p95_ms",
        retain_samples=retain_samples,
    )
    final_hash = engine.state_hash(state)
    if final_hash != original_hash:
        raise RuntimeError("benchmark mutated its frozen source state")

    report = service.report([reversible, search])
    source_fixture = {
        "id": case["id"],
        "stateSha256": case["stateSha256"],
        "remainingCards": case["remainingCards"],
        "oracle": case["oracle"],
    }
    engine_identity = {
        "bundleSha256": bundle.sha256,
        "rulesSha256": bundle.rules.sha256,
    }
    research_authority = {
        "backendId": "ortools_cp_sat_v2",
        "qualificationAuthority": "cp_sat",
        "bundleSource": ORTOOLS_RESEARCH_BUNDLE,
    }
    search_budget = {
        "maxNodes": request.exact_search_max_nodes,
        "maxSecondsSafetyCancellation": request.exact_search_max_seconds,
        "maxDepth": request.exact_search_max_depth,
        "maxCacheEntries": request.exact_search_max_cache_entries,
    }
    semantic_identity = {
        "schemaVersion": 1,
        "kind": "exact_search_semantic_reproducibility_v1",
        "benchmarkVersion": "21.1.0",
        "sourceFixture": source_fixture,
        "initialStateSha256": original_hash,
        "finalStateSha256": final_hash,
        "engine": engine_identity,
        "researchAuthority": research_authority,
        "searchBudget": search_budget,
        "exactSearchObservation": dict(search_observation),
    }
    reproducibility_core = {
        "schemaVersion": 1,
        "status": "passed",
        "observationCount": search_observation_count,
        "semanticIdentity": semantic_identity,
        "semanticIdentitySha256": canonical_sha256(semantic_identity),
    }
    report.update(
        {
            "schemaVersion": 2,
            "version": "21.1.0",
            "source": source_identity or {
                "revision": None,
                "tree": None,
                "scope": SOURCE_SCOPE,
                "excludedPaths": list(SOURCE_EXCLUDED_PATHS),
                "binding": "unbound_test_invocation",
            },
            "hardware": {
                "logicalCpuCount": os.cpu_count(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "platform": platform.platform(),
            },
            "runtimeScope": _runtime_scope(),
            "sourceFixture": source_fixture,
            "engine": engine_identity,
            "researchAuthority": research_authority,
            "searchBudget": search_budget,
            "exactSearchObservation": dict(search_observation),
            "reproducibility": {
                **reproducibility_core,
                "sha256": canonical_sha256(reproducibility_core),
            },
            "promotionEvaluation": {
                "performanceBudgetApplied": True,
                "decision": "not_evaluated",
                "reason": (
                    "Local budget conformance and same-hardware improvement "
                    "are not release promotion or cross-hardware proof."
                ),
            },
            "throughput": {
                "reversibleApplyUndoPerSecond": round(
                    1000.0 / max(reversible.median_ms, 1e-9),
                    2,
                ),
                "exactFourCardSearchesPerSecond": round(
                    1000.0 / max(search.median_ms, 1e-9),
                    2,
                ),
            },
            "proofBoundary": (
                "Single-process elapsed-time measurement, including scheduler delay, "
                "within the declared process-affinity and cgroup scope. It is not a "
                "cross-hardware comparison, deployment benchmark, or proof of the "
                "promotion targets."
            ),
        }
    )
    if baseline is not None:
        if not baseline_revision or not baseline_sha256:
            raise ValueError(
                "baseline revision and artifact SHA-256 are required"
            )
        report["baselineComparison"] = _comparison(
            report,
            baseline,
            baseline_revision=baseline_revision,
            baseline_sha256=baseline_sha256,
        )
    report["evidenceSha256"] = canonical_sha256(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--baseline-revision")
    args = parser.parse_args()
    output = (args.output or ROOT / DEFAULT_OUTPUT).resolve()
    source_identity = git_identity(ROOT)
    baseline = None
    baseline_sha256 = None
    if args.baseline is not None:
        baseline_bytes = args.baseline.read_bytes()
        baseline = json.loads(baseline_bytes)
        baseline_sha256 = hashlib.sha256(baseline_bytes).hexdigest()
    report = run_benchmark(
        iterations=args.iterations,
        warmup=args.warmup,
        source_identity=source_identity,
        retain_samples=True,
        baseline=baseline,
        baseline_revision=args.baseline_revision,
        baseline_sha256=baseline_sha256,
    )
    atomic_write_text(
        output,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(output)
    if report["status"] != "passed":
        raise SystemExit("search performance budgets failed")


if __name__ == "__main__":
    main()
