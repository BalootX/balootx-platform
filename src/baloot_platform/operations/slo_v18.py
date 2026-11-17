from __future__ import annotations

from typing import Any, Mapping


class SloEvaluatorV18:
    def __init__(self, profile: Mapping[str, Any]):
        self.profile = dict(profile)

    def evaluate(self, metrics: Mapping[str, float]) -> dict[str, Any]:
        objectives = dict(self.profile.get("objectives") or {})
        checks = []
        for name, target in objectives.items():
            rule = dict(target or {})
            observed = metrics.get(name)
            comparator = str(rule.get("comparator", "lte"))
            threshold = float(rule.get("target", 0.0))
            if observed is None:
                passed = False
            elif comparator == "gte":
                passed = float(observed) >= threshold
            else:
                passed = float(observed) <= threshold
            checks.append({"name": name, "observed": observed, "target": threshold,
                           "comparator": comparator, "passed": passed})
        return {"status": "passed" if checks and all(x["passed"] for x in checks) else "failed",
                "checks": checks, "automaticProductionChange": False}
