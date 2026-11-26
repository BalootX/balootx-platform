"""Operational process for fenced generated Foundry simulation shards."""
from __future__ import annotations

import os
import socket
import time

from baloot_platform.challenge.worker import (
    DisagreementFoundryWorkerV1,
    GuidedFoundryWorkerV1,
    OrganicFoundryWorkerV1,
)
from baloot_platform.observability import log_worker_result, run_worker
from baloot_platform.service.container import build_container


_GENERATED_LANES = frozenset({"organic", "guided", "disagreement"})


def _truthy(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _build_worker(container, *, lane: str, worker_id: str, lease_seconds: int,
                  heartbeat_seconds: int):
    selected_lane = str(lane).strip().lower()
    if selected_lane not in _GENERATED_LANES:
        raise ValueError(
            "BALOOT_FOUNDRY_LANE must be organic, guided, or disagreement"
        )
    common = {
        "worker_id": worker_id,
        "lease_seconds": lease_seconds,
        "heartbeat_seconds": heartbeat_seconds,
    }
    if selected_lane == "organic":
        return OrganicFoundryWorkerV1(
            container.campaign_control,
            container.learning.artifact_store,
            container.registry.get("generator", "natural_laffa_generator_v1"),
            screening_profile=container.organic_screening_profile,
            **common,
        )
    if selected_lane == "guided":
        return GuidedFoundryWorkerV1(
            container.campaign_control,
            container.learning.artifact_store,
            container.registry.get("generator", "guided_legal_search_v1"),
            **common,
        )
    return DisagreementFoundryWorkerV1(
        container.campaign_control,
        container.learning.artifact_store,
        container.registry.get("generator", "observer_safe_disagreement_search_v1"),
        **common,
    )


def _run() -> None:
    container = build_container()
    lane = os.environ.get("BALOOT_FOUNDRY_LANE", "organic").strip().lower()
    worker_id = os.environ.get(
            "BALOOT_WORKER_ID",
            f"{socket.gethostname()}-{os.getpid()}-foundry-{lane}",
        )
    worker = _build_worker(
        container,
        lane=lane,
        worker_id=worker_id,
        lease_seconds=int(
            os.environ.get("BALOOT_FOUNDRY_SIMULATION_LEASE_SECONDS", "900")
        ),
        heartbeat_seconds=int(
            os.environ.get("BALOOT_FOUNDRY_SIMULATION_HEARTBEAT_SECONDS", "60")
        ),
    )
    poll_seconds = float(os.environ.get("BALOOT_WORKER_POLL_SECONDS", "0.5"))
    campaign_hash = os.environ.get("BALOOT_FOUNDRY_CAMPAIGN_HASH", "").strip() or None
    exit_when_idle = _truthy("BALOOT_WORKER_EXIT_WHEN_IDLE")
    try:
        while True:
            result = worker.run_once(campaign_hash=campaign_hash)
            if result.status == "idle":
                if exit_when_idle:
                    return
                time.sleep(poll_seconds)
            else:
                log_worker_result("foundry_simulation", result.shard_hash, result.status)
    finally:
        container.shutdown()


def main() -> None:
    run_worker("foundry_simulation", _run)


if __name__ == "__main__":  # pragma: no cover - module execution
    main()
