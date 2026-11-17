#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from baloot_platform.config import build_bundle
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.episode_io import load_episode_spec
from baloot_platform.performance import PerformanceService
from baloot_platform.service.container import build_container

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    profile = yaml.safe_load((ROOT / "configs/performance/budgets_v19.yaml").read_text(encoding="utf-8"))
    service = PerformanceService(profile)
    bundle = build_bundle(ROOT)
    episode = load_episode_spec(ROOT / "examples/v7-outcome-flip-source.json", bundle.rules)
    engine = BalootEngine(bundle.rules)
    state = episode.state_before_pause.clone()
    runtime = ROOT / ".release-runtime/v19-benchmark"
    runtime.mkdir(parents=True, exist_ok=True)
    container = build_container(root=ROOT, writable_root=runtime)
    try:
        current = container.configuration.store.list_bundles(status="published", limit=1)[0]
        measurements = [
            service.measure("legal_moves_p95_ms", lambda: engine.legal_moves(state), iterations=1000, warmup=100),
            service.measure("configuration_resolve_p95_ms", lambda: container.configuration.resolve(revision_id=current.revision_id), iterations=200, warmup=25),
            service.measure("plugin_registry_list_p95_ms", lambda: container.registry.list(), iterations=500, warmup=50),
        ]
        # API health is measured through the application contract in a child import
        # friendly environment, while avoiding long-running external services.
        os.environ.setdefault("BALOOT_ENV", "test")
        from baloot_platform.service.api import app
        client = TestClient(app)
        measurements.append(service.measure("api_health_p95_ms", lambda: client.get("/api/v1/health").raise_for_status(), iterations=50, warmup=5))
        report = service.report(measurements)
        report.update({
            "schemaVersion": 1,
            "version": "19.0.0",
            "bundleHash": bundle.sha256,
            "legalPositionsPerSecond": round(1000.0 / max(0.000001, measurements[0].median_ms), 2),
        })
        output = ROOT / "release/performance-benchmark-v19.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(output)
        if report["status"] != "passed":
            raise SystemExit("performance budgets failed")
    finally:
        container.shutdown()


if __name__ == "__main__":
    main()
