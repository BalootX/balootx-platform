from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from baloot_platform.challenge.campaign import SimulationCampaignSpecV1
from baloot_platform.challenge.evidence_v3 import canonical_sha256
from baloot_platform.challenge.worker import OrganicFoundryWorkerV1
from baloot_platform.service.container import build_container


_EXECUTION_MATRIX = (
    (1, "none"),
    (2, "out_of_order"),
    (4, "retry_reclaim"),
)

_STORAGE_METRIC_KEYS = {
    "schemaVersion",
    "campaignHash",
    "occurrenceCount",
    "uniqueTraceCount",
    "uniqueArtifactCount",
    "logicalBytes",
    "uniqueBytes",
}


def _progress(event: str, **values: Any) -> None:
    print(
        json.dumps(
            {"event": event, **values},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


def _ratio(numerator: int, denominator: int) -> float:
    return round(float(numerator) / float(denominator), 12)


def _normalized_storage_metrics(
    store: Any,
    campaign_hash: str,
    *,
    requested_rounds: int,
) -> dict[str, Any]:
    if (
        isinstance(requested_rounds, bool)
        or not isinstance(requested_rounds, int)
        or requested_rounds < 1
    ):
        raise RuntimeError("requested round count is invalid")
    raw = dict(store.audit_trace_storage_metrics(campaign_hash))
    if set(raw) != _STORAGE_METRIC_KEYS:
        raise RuntimeError("campaign audit-trace storage metrics contract drifted")
    if (
        isinstance(raw["schemaVersion"], bool)
        or raw["schemaVersion"] != 1
        or not isinstance(raw["campaignHash"], str)
        or raw["campaignHash"] != campaign_hash
    ):
        raise RuntimeError("campaign audit-trace storage metrics identity mismatch")
    names = (
        "occurrenceCount",
        "uniqueTraceCount",
        "uniqueArtifactCount",
        "logicalBytes",
        "uniqueBytes",
    )
    values: dict[str, int] = {}
    for name in names:
        value = raw[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"campaign audit-trace storage metric {name} is invalid")
        values[name] = value
    occurrences = values["occurrenceCount"]
    unique_traces = values["uniqueTraceCount"]
    unique_artifacts = values["uniqueArtifactCount"]
    logical_bytes = values["logicalBytes"]
    unique_bytes = values["uniqueBytes"]
    if occurrences < requested_rounds:
        raise RuntimeError("campaign audit-trace occurrences do not cover every requested round")
    if not 1 <= unique_traces <= occurrences:
        raise RuntimeError("campaign audit-trace unique trace count is incoherent")
    if not 1 <= unique_artifacts <= occurrences:
        raise RuntimeError("campaign audit-trace unique artifact count is incoherent")
    if logical_bytes < occurrences or not 1 <= unique_bytes <= logical_bytes:
        raise RuntimeError("campaign audit-trace storage byte counts are incoherent")
    if unique_bytes < unique_artifacts:
        raise RuntimeError("campaign audit-trace unique artifact bytes are incoherent")
    deduplicated_bytes = logical_bytes - unique_bytes
    return {
        "schemaVersion": 1,
        **values,
        "deduplicatedBytes": deduplicated_bytes,
        "dedupeRatio": _ratio(deduplicated_bytes, logical_bytes),
        "logicalToUniqueRatio": _ratio(logical_bytes, unique_bytes),
        "logicalBytesPerRequestedRound": _ratio(logical_bytes, requested_rounds),
        "uniqueBytesPerRequestedRound": _ratio(unique_bytes, requested_rounds),
    }


def _bounded_execution_evidence(
    executions: list[dict[str, Any]],
    *,
    requested_rounds: int,
) -> dict[str, Any]:
    if (
        isinstance(requested_rounds, bool)
        or not isinstance(requested_rounds, int)
        or requested_rounds < 1
    ):
        raise RuntimeError("requested round count is invalid")
    normalized = [dict(item["storage"]) for item in executions]
    if not normalized or any(item != normalized[0] for item in normalized[1:]):
        raise RuntimeError(
            "campaign audit-trace storage metrics differ across worker schedules"
        )
    common = normalized[0]
    execution_count = len(executions)
    matrix_logical_bytes = int(common["logicalBytes"]) * execution_count
    matrix_unique_bytes = int(common["uniqueBytes"])
    matrix_deduplicated_bytes = matrix_logical_bytes - matrix_unique_bytes
    return {
        "schemaVersion": 1,
        "requestedRounds": int(requested_rounds),
        "executionCount": execution_count,
        "normalizedStorageMetrics": common,
        "normalizedStorageSha256": canonical_sha256(common),
        "matrixStorageMetrics": {
            "schemaVersion": 1,
            "requestedExecutionRounds": int(requested_rounds) * execution_count,
            "occurrenceCount": int(common["occurrenceCount"]) * execution_count,
            "uniqueTraceCount": int(common["uniqueTraceCount"]),
            "uniqueArtifactCount": int(common["uniqueArtifactCount"]),
            "logicalBytes": matrix_logical_bytes,
            "uniqueBytes": matrix_unique_bytes,
            "deduplicatedBytes": matrix_deduplicated_bytes,
            "dedupeRatio": _ratio(
                matrix_deduplicated_bytes, matrix_logical_bytes
            ),
            "logicalToUniqueRatio": _ratio(
                matrix_logical_bytes, matrix_unique_bytes
            ),
            "logicalBytesPerExecutedRound": _ratio(
                matrix_logical_bytes, requested_rounds * execution_count
            ),
            "uniqueBytesPerLogicalRound": _ratio(
                matrix_unique_bytes, requested_rounds
            ),
        },
        "executions": executions,
    }


def _campaign(container: Any, *, seed_start: int, rounds: int, workers: int) -> SimulationCampaignSpecV1:
    runtime = container.foundry_profile
    shard_count = 1 if workers == 1 else max(4, workers * 2)
    seeds_per_shard = max(1, math.ceil(rounds / shard_count))
    lane_hash = str(runtime["staticLaneProfileHashes"]["organic"])
    return SimulationCampaignSpecV1.build(
        rules_hash=str(runtime["rulesHash"]),
        bundle_hash=str(runtime["bundleHash"]),
        engine_version=str(runtime["engineVersion"]),
        analysis_profile_hash=str(runtime["analysisProfileHash"]),
        perspective_profile_hash=str(runtime["perspectiveProfileHash"]),
        selection_profile_hash=str(runtime["selectionProfileHash"]),
        agent_release=str(runtime["agentRelease"]),
        seat_agents=tuple(
            str(runtime["seatAgents"][seat])
            for seat in ("north", "west", "south", "east")
        ),
        lanes=("organic",),
        lane_profile_hashes={"organic": lane_hash},
        seed_start=seed_start,
        seed_count=rounds,
        seeds_per_shard=seeds_per_shard,
    )


def _workers(container: Any, *, count: int, schedule: str) -> tuple[OrganicFoundryWorkerV1, ...]:
    return tuple(
        OrganicFoundryWorkerV1(
            container.campaign_control,
            container.learning.artifact_store,
            container.registry.get("generator", "natural_laffa_generator_v1"),
            screening_profile=container.organic_screening_profile,
            worker_id=f"bounded-{schedule}-w{index + 1}",
            lease_seconds=7_200,
            heartbeat_seconds=60,
        )
        for index in range(count)
    )


def _require_fresh_campaign(store: Any, spec: SimulationCampaignSpecV1) -> None:
    row = store.get_campaign(spec.sha256)
    if row is None:
        store.create_campaign(spec)
        return
    if row["status"] != "completed":
        raise RuntimeError(
            "bounded proof found a partial campaign; choose a new --seed-start "
            "so failure evidence cannot be reinterpreted"
        )


def _run_none(container: Any, spec: SimulationCampaignSpecV1) -> None:
    worker = _workers(container, count=1, schedule="none")[0]
    while container.campaign_control.get_campaign(spec.sha256)["status"] != "completed":
        result = worker.run_once(campaign_hash=spec.sha256)
        if result.status != "completed":
            raise RuntimeError(f"serial proof worker returned {result.status}: {result.error}")
        _progress("shard_completed", campaignHash=spec.sha256, schedule="none", shardHash=result.shard_hash)


def _run_out_of_order(container: Any, spec: SimulationCampaignSpecV1) -> None:
    workers = _workers(container, count=2, schedule="out_of_order")
    claims = []
    for index in range(len(spec.shards())):
        worker = workers[index % len(workers)]
        claim = container.campaign_control.claim_shard(
            worker.worker_id,
            lease_seconds=7_200,
            campaign_hash=spec.sha256,
            supported_lanes=("organic",),
        )
        if claim is None:
            raise RuntimeError("out-of-order proof could not fence every immutable shard")
        claims.append((worker, claim))
    for worker, claim in reversed(claims):
        result = worker.run_claim(claim)
        if result.status != "completed":
            raise RuntimeError(
                f"out-of-order proof worker returned {result.status}: {result.error}"
            )
        _progress(
            "shard_completed",
            campaignHash=spec.sha256,
            schedule="out_of_order",
            shardHash=result.shard_hash,
        )


def _run_retry_reclaim(container: Any, spec: SimulationCampaignSpecV1) -> None:
    workers = _workers(container, count=4, schedule="retry_reclaim")
    failed = container.campaign_control.claim_shard(
        workers[0].worker_id,
        lease_seconds=7_200,
        campaign_hash=spec.sha256,
        supported_lanes=("organic",),
    )
    if failed is None:
        raise RuntimeError("retry proof could not claim the injected failure shard")
    container.campaign_control.fail_shard(
        failed.spec.input_hash,
        worker_id=failed.worker_id,
        fencing_token=failed.fencing_token,
        error="bounded proof injected retryable failure",
        retryable=True,
    )
    _progress(
        "retry_injected",
        campaignHash=spec.sha256,
        shardHash=failed.spec.input_hash,
    )
    worker_index = 0
    while container.campaign_control.get_campaign(spec.sha256)["status"] != "completed":
        worker = workers[worker_index % len(workers)]
        worker_index += 1
        result = worker.run_once(campaign_hash=spec.sha256)
        if result.status != "completed":
            raise RuntimeError(f"retry proof worker returned {result.status}: {result.error}")
        _progress(
            "shard_completed",
            campaignHash=spec.sha256,
            schedule="retry_reclaim",
            shardHash=result.shard_hash,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the governed bounded Foundry partition-equivalence proof."
    )
    parser.add_argument("--seed-start", type=int, default=910_000)
    parser.add_argument("--rounds", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/campaign-proof/bounded-proof.json"),
    )
    args = parser.parse_args()
    if args.seed_start < 0:
        parser.error("--seed-start must be non-negative")

    container = build_container()
    try:
        profile = container.campaign_proofs.profile
        rounds = profile.minimum_organic_rounds if args.rounds is None else args.rounds
        if not 8 <= rounds <= 5_000:
            parser.error(
                "--rounds must be between 8 and 5000 so the 1/2/4-worker "
                "matrix has distinct shard layouts"
            )
        required_matrix = {
            (workers, schedule) for workers, schedule in _EXECUTION_MATRIX
        }
        if set(profile.required_worker_counts) != {1, 2, 4} or set(
            profile.required_failure_schedules
        ) != {"none", "out_of_order", "retry_reclaim"}:
            raise RuntimeError(
                "the published campaign-proof profile does not match the supported "
                "bounded execution matrix"
            )
        _progress(
            "proof_started",
            profileHash=profile.sha256,
            rounds=rounds,
            seedStart=args.seed_start,
            executionMatrix=sorted(required_matrix),
        )
        executions = []
        execution_evidence = []
        for workers, schedule in _EXECUTION_MATRIX:
            spec = _campaign(
                container,
                seed_start=args.seed_start,
                rounds=rounds,
                workers=workers,
            )
            _require_fresh_campaign(container.campaign_control, spec)
            _progress(
                "execution_started",
                campaignHash=spec.sha256,
                workers=workers,
                schedule=schedule,
                shardCount=len(spec.shards()),
            )
            reused_completed_campaign = (
                container.campaign_control.get_campaign(spec.sha256)["status"]
                == "completed"
            )
            started_wall = time.perf_counter()
            started_cpu = time.process_time()
            if not reused_completed_campaign:
                if schedule == "none":
                    _run_none(container, spec)
                elif schedule == "out_of_order":
                    _run_out_of_order(container, spec)
                else:
                    _run_retry_reclaim(container, spec)
            timing = {
                "schemaVersion": 1,
                "reusedCompletedCampaign": reused_completed_campaign,
                "wallSeconds": round(time.perf_counter() - started_wall, 6),
                "processCpuSeconds": round(time.process_time() - started_cpu, 6),
                "wallMeasurementBoundary": "campaign_execution_perf_counter_v1",
                "cpuMeasurementBoundary": "campaign_execution_process_time_v1",
            }
            storage = _normalized_storage_metrics(
                container.campaign_control,
                spec.sha256,
                requested_rounds=rounds,
            )
            executions.append(
                {
                    "campaignHash": spec.sha256,
                    "declaredWorkerCount": workers,
                    "failureSchedule": schedule,
                }
            )
            execution_evidence.append(
                {
                    "campaignHash": spec.sha256,
                    "declaredWorkerCount": workers,
                    "failureSchedule": schedule,
                    "storage": storage,
                    "timing": timing,
                }
            )
            _progress(
                "execution_completed",
                campaignHash=spec.sha256,
                workers=workers,
                schedule=schedule,
                storage=storage,
                timing=timing,
            )
        proof = container.campaign_proofs.create(tuple(executions))
        document = dict(proof["document"])
        if rounds >= profile.minimum_organic_rounds and document["status"] != "qualified":
            raise RuntimeError(
                "bounded campaign proof did not qualify: "
                + ", ".join(document["unmetQualificationGates"])
            )
        bounded_evidence = _bounded_execution_evidence(
            execution_evidence,
            requested_rounds=rounds,
        )
        output_document = {
            **proof,
            "boundedExecutionEvidence": bounded_evidence,
        }
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(output_document, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        _progress(
            "proof_completed",
            status=document["status"],
            normalizedEvidenceHash=document["normalizedEvidenceHash"],
            normalizedStorageSha256=bounded_evidence["normalizedStorageSha256"],
            output=str(output),
        )
        return 0
    finally:
        container.shutdown()


if __name__ == "__main__":
    sys.exit(main())
