"""Operational process for fenced Foundry deep-analysis jobs."""
from __future__ import annotations

import os
import socket
import time

from baloot_platform.challenge.deep_analysis import FoundryDeepAnalysisWorkerV1
from baloot_platform.service.container import build_container


def _truthy(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:  # pragma: no cover - operational process
    container = build_container()
    worker = FoundryDeepAnalysisWorkerV1(
        container.deep_analysis,
        container.learning.artifact_store,
        worker_id=os.environ.get(
            "BALOOT_WORKER_ID",
            f"{socket.gethostname()}-{os.getpid()}-foundry-analysis",
        ),
        lease_seconds=int(os.environ.get("BALOOT_FOUNDRY_ANALYSIS_LEASE_SECONDS", "120")),
        heartbeat_seconds=int(
            os.environ.get("BALOOT_FOUNDRY_ANALYSIS_HEARTBEAT_SECONDS", "30")
        ),
    )
    poll_seconds = float(os.environ.get("BALOOT_WORKER_POLL_SECONDS", "0.5"))
    exit_when_idle = _truthy("BALOOT_WORKER_EXIT_WHEN_IDLE")
    try:
        while True:
            result = worker.run_once()
            if result.status == "idle":
                if exit_when_idle:
                    return
                time.sleep(poll_seconds)
    finally:
        container.shutdown()


if __name__ == "__main__":  # pragma: no cover - module execution
    main()
