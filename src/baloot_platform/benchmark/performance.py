from __future__ import annotations

import math
import platform
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = max(0.0, min(100.0, percentile)) / 100.0 * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - rank) + ordered[high] * (rank - low)


@dataclass(frozen=True, slots=True)
class PerformanceResult:
    name: str
    iterations: int
    warmup_iterations: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    maximum_ms: float
    target_p95_ms: float | None
    passed: bool
    environment: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "warmupIterations": self.warmup_iterations,
            "p50Ms": self.p50_ms,
            "p95Ms": self.p95_ms,
            "p99Ms": self.p99_ms,
            "meanMs": self.mean_ms,
            "maximumMs": self.maximum_ms,
            "targetP95Ms": self.target_p95_ms,
            "passed": self.passed,
            "environment": dict(self.environment),
        }


class PerformanceHarnessV19:
    """Deterministic latency harness; measurement never mutates production tuning."""

    def __init__(self, profile: dict[str, Any]):
        self.profile = dict(profile)
        measurement = dict(profile.get("measurement") or {})
        self.warmup_iterations = int(measurement.get("warmup_iterations", 25))
        self.measured_iterations = int(measurement.get("measured_iterations", 200))
        self.budgets = {str(key): float(value) for key, value in dict(profile.get("budgets") or {}).items()}

    def run(self, name: str, operation: Callable[[], T], *, iterations: int | None = None,
            warmup: int | None = None, target_key: str | None = None) -> PerformanceResult:
        warmup_count = max(0, int(self.warmup_iterations if warmup is None else warmup))
        measured = max(1, int(self.measured_iterations if iterations is None else iterations))
        for _ in range(warmup_count):
            operation()
        values: list[float] = []
        for _ in range(measured):
            start = time.perf_counter_ns()
            operation()
            values.append((time.perf_counter_ns() - start) / 1_000_000.0)
        p95 = _percentile(values, 95)
        target = self.budgets.get(target_key or name)
        environment = {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        }
        return PerformanceResult(
            name=name, iterations=measured, warmup_iterations=warmup_count,
            p50_ms=round(_percentile(values, 50), 6), p95_ms=round(p95, 6),
            p99_ms=round(_percentile(values, 99), 6), mean_ms=round(statistics.fmean(values), 6),
            maximum_ms=round(max(values), 6), target_p95_ms=target,
            passed=target is None or p95 <= target, environment=environment,
        )

    def compare(self, baseline: PerformanceResult, challenger: PerformanceResult) -> dict[str, Any]:
        base = max(baseline.p95_ms, 1e-9)
        regression = (challenger.p95_ms - baseline.p95_ms) / base * 100.0
        limit = float(self.profile.get("measurement", {}).get("fail_on_regression_percent", 20.0))
        return {
            "baseline": baseline.to_dict(), "challenger": challenger.to_dict(),
            "p95RegressionPercent": round(regression, 4), "allowedRegressionPercent": limit,
            "passed": regression <= limit and challenger.passed,
            "automaticTuningApplied": False,
        }
