from __future__ import annotations

import math
import os
import platform
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from baloot_platform.observability import TELEMETRY


@dataclass(frozen=True, slots=True)
class BenchmarkMeasurement:
    name: str
    iterations: int
    warmup_iterations: int
    minimum_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    maximum_ms: float
    budget_ms: float | None
    passed: bool
    samples_ms: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = {
            "name": self.name, "iterations": self.iterations,
            "warmupIterations": self.warmup_iterations,
            "minimumMs": self.minimum_ms, "medianMs": self.median_ms,
            "p95Ms": self.p95_ms, "p99Ms": self.p99_ms,
            "meanMs": self.mean_ms, "maximumMs": self.maximum_ms,
            "budgetMs": self.budget_ms, "passed": self.passed,
        }
        if self.samples_ms:
            value["samplesMs"] = list(self.samples_ms)
        return value


class PerformanceService:
    def __init__(self, profile: Mapping[str, Any]):
        self.profile = dict(profile)
        self.budgets = {str(k): float(v) for k, v in dict(self.profile.get("budgets") or {}).items()}
        measurement = dict(self.profile.get("measurement") or {})
        self.warmup = int(measurement.get("warmup_iterations", 25))
        self.iterations = int(measurement.get("measured_iterations", 200))
        self.percentile = float(measurement.get("percentile", 95))

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        ordered = sorted(values)
        if not ordered:
            return 0.0
        position = (len(ordered) - 1) * percentile / 100.0
        lower, upper = math.floor(position), math.ceil(position)
        if lower == upper:
            return ordered[lower]
        low, high = ordered[lower], ordered[upper]
        # Interpolate from the lower sample and keep rounding within the bracket.
        return min(high, low + (position - lower) * (high - low))

    def measure(self, name: str, operation: Callable[[], Any], *, iterations: int | None = None,
                warmup: int | None = None, budget_key: str | None = None,
                retain_samples: bool = False) -> BenchmarkMeasurement:
        warmup_count = self.warmup if warmup is None else max(0, int(warmup))
        for _ in range(warmup_count):
            operation()
        values: list[float] = []
        count = self.iterations if iterations is None else max(1, int(iterations))
        with TELEMETRY.span("performance.measure", benchmark=name):
            for _ in range(count):
                start = time.perf_counter_ns()
                operation()
                values.append(
                    (time.perf_counter_ns() - start) / 1_000_000.0
                )
        key = budget_key or name
        budget = self.budgets.get(key)
        p95 = self._percentile(values, self.percentile)
        TELEMETRY.observe(f"performance.{name}.p95_ms", p95)
        return BenchmarkMeasurement(
            name=name,
            iterations=count,
            warmup_iterations=warmup_count,
            minimum_ms=min(values),
            median_ms=statistics.median(values),
            p95_ms=p95,
            p99_ms=self._percentile(values, 99),
            mean_ms=statistics.fmean(values),
            maximum_ms=max(values),
            budget_ms=budget,
            passed=budget is None or p95 <= budget,
            samples_ms=(
                tuple(round(value, 6) for value in values)
                if retain_samples
                else ()
            ),
        )

    def environment(self) -> dict[str, Any]:
        return {
            "python": platform.python_version(),
            "pythonImplementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logicalCpuCount": os.cpu_count(),
            "profile": self.profile.get("profile", {}),
        }

    def report(self, measurements: list[BenchmarkMeasurement]) -> dict[str, Any]:
        return {"status": "passed" if all(item.passed for item in measurements) else "failed", "environment": self.environment(), "measurements": [item.to_dict() for item in measurements]}
