"""Bounded multi-process launcher for generated Foundry simulation workers."""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Mapping

from baloot_platform.service.container import build_container


_HASH = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[A-Za-z0-9._-]{1,80}$")


@dataclass(frozen=True, slots=True)
class FoundrySimulationFleetSpecV1:
    campaign_hash: str
    worker_count: int
    run_id: str
    host_id: str
    timeout_seconds: int

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None
    ) -> "FoundrySimulationFleetSpecV1":
        values = os.environ if environment is None else environment
        campaign_hash = str(values.get("BALOOT_FOUNDRY_CAMPAIGN_HASH") or "").strip()
        if not _HASH.fullmatch(campaign_hash):
            raise ValueError("BALOOT_FOUNDRY_CAMPAIGN_HASH must be a lowercase SHA-256")
        worker_count = int(values.get("BALOOT_FOUNDRY_WORKER_COUNT") or "0")
        if not 1 <= worker_count <= 256:
            raise ValueError("BALOOT_FOUNDRY_WORKER_COUNT must be between 1 and 256")
        run_id = str(values.get("BALOOT_FOUNDRY_RUN_ID") or "").strip()
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError(
                "BALOOT_FOUNDRY_RUN_ID must be 1-80 letters, numbers, dots, dashes, or underscores"
            )
        host_id = str(values.get("BALOOT_FOUNDRY_HOST_ID") or socket.gethostname()).strip()
        if not _RUN_ID.fullmatch(host_id):
            raise ValueError("BALOOT_FOUNDRY_HOST_ID is invalid")
        timeout_seconds = int(
            values.get("BALOOT_FOUNDRY_FLEET_TIMEOUT_SECONDS") or "0"
        )
        if not 60 <= timeout_seconds <= 604_800:
            raise ValueError(
                "BALOOT_FOUNDRY_FLEET_TIMEOUT_SECONDS must be between 60 and 604800"
            )
        return cls(campaign_hash, worker_count, run_id, host_id, timeout_seconds)

    def worker_environment(
        self, base: Mapping[str, str], slot: int, lanes: tuple[str, ...] = ("organic",)
    ) -> dict[str, str]:
        if not 1 <= int(slot) <= self.worker_count:
            raise ValueError("Foundry fleet worker slot is out of range")
        normalized_lanes = tuple(sorted(set(str(item) for item in lanes)))
        unsupported = set(normalized_lanes) - {"organic", "guided", "disagreement"}
        if not normalized_lanes or unsupported:
            raise ValueError(
                "Foundry simulation fleet supports organic, guided, and disagreement lanes"
            )
        if self.worker_count < len(normalized_lanes):
            raise ValueError(
                "Foundry worker count must be at least the generated lane count"
            )
        lane = normalized_lanes[(int(slot) - 1) % len(normalized_lanes)]
        result = dict(base)
        result.update({
            "BALOOT_FOUNDRY_CAMPAIGN_HASH": self.campaign_hash,
            "BALOOT_FOUNDRY_LANE": lane,
            "BALOOT_WORKER_EXIT_WHEN_IDLE": "true",
            "BALOOT_WORKER_ID": (
                f"{self.host_id}-{self.run_id}-{lane}-w{int(slot):03d}"
            ),
        })
        return result

    def expected_worker_ids(
        self, base: Mapping[str, str], lanes: tuple[str, ...]
    ) -> tuple[str, ...]:
        return tuple(sorted(
            self.worker_environment(base, slot, lanes)["BALOOT_WORKER_ID"]
            for slot in range(1, self.worker_count + 1)
        ))


def _campaign_snapshot(
    campaign_hash: str,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    container = build_container()
    try:
        row = container.campaign_control.get_campaign(campaign_hash)
        if row is None:
            raise KeyError(f"unknown Foundry campaign {campaign_hash}")
        attempts = container.campaign_control.list_attempts(campaign_hash)
        workers = tuple(sorted({str(item["workerId"]) for item in attempts}))
        lanes = tuple(sorted(str(item) for item in row["spec"]["lanes"]))
        return str(row["status"]), workers, lanes
    finally:
        container.shutdown()


def _release_owned_claims(
    campaign_control: Any,
    campaign_hash: str,
    expected_workers: tuple[str, ...],
    *,
    error: str,
) -> int:
    """Return terminated fleet claims to the governed retry queue."""
    released = 0
    allowed = set(expected_workers)
    for shard in campaign_control.list_shards(campaign_hash):
        worker_id = str(shard.get("workerId") or "")
        if shard.get("status") != "running" or worker_id not in allowed:
            continue
        campaign_control.fail_shard(
            str(shard["specHash"]),
            worker_id=worker_id,
            fencing_token=int(shard["fencingToken"]),
            error=error,
            retryable=True,
        )
        released += 1
    return released


def main() -> None:  # pragma: no cover - operational process
    spec = FoundrySimulationFleetSpecV1.from_environment()
    initial_status, existing_workers, lanes = _campaign_snapshot(spec.campaign_hash)
    expected_workers = spec.expected_worker_ids(os.environ, lanes)
    unexpected_workers = sorted(set(existing_workers) - set(expected_workers))
    if unexpected_workers:
        raise ValueError(
            "Foundry campaign already contains worker identities outside this fleet: "
            f"{unexpected_workers}"
        )
    if initial_status in {"failed", "cancelled"}:
        raise RuntimeError(
            f"Foundry campaign cannot run from terminal state {initial_status}"
        )
    if initial_status == "completed":
        print(json.dumps({
            "schemaVersion": 1,
            "campaignHash": spec.campaign_hash,
            "workerCount": 0,
            "status": "already_completed",
        }, sort_keys=True, separators=(",", ":")))
        return

    command = [
        sys.executable,
        "-m",
        "baloot_platform.service.foundry_simulation_worker",
    ]
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    processes: list[subprocess.Popen[bytes]] = []
    try:
        for slot in range(1, spec.worker_count + 1):
            processes.append(subprocess.Popen(
                command,
                env=spec.worker_environment(os.environ, slot, lanes),
                creationflags=creation_flags,
            ))
        deadline = time.monotonic() + spec.timeout_seconds
        exit_codes: list[int | None] = [None] * len(processes)
        while True:
            for index, process in enumerate(processes):
                if exit_codes[index] is None:
                    exit_codes[index] = process.poll()
            failures = [
                {"slot": index + 1, "exitCode": code}
                for index, code in enumerate(exit_codes)
                if code not in {None, 0}
            ]
            if failures:
                raise RuntimeError(
                    f"Foundry simulation fleet worker failures: {failures}"
                )
            if all(code == 0 for code in exit_codes):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Foundry simulation fleet exceeded its declared timeout"
                )
            time.sleep(0.1)
    except BaseException as exc:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=30)
        cleanup = build_container()
        try:
            _release_owned_claims(
                cleanup.campaign_control,
                spec.campaign_hash,
                expected_workers,
                error=f"fleet stopped before completion: {type(exc).__name__}",
            )
        except Exception as cleanup_error:
            raise RuntimeError(
                "Foundry fleet stopped and could not release its fenced shard claims"
            ) from cleanup_error
        finally:
            cleanup.shutdown()
        raise
    final_status, attested_workers, final_lanes = _campaign_snapshot(
        spec.campaign_hash
    )
    if final_lanes != lanes:
        raise RuntimeError("Foundry campaign lane identity changed during execution")
    if final_status != "completed":
        raise RuntimeError(
            f"Foundry simulation fleet exited with campaign status {final_status}"
        )
    if attested_workers != expected_workers:
        raise RuntimeError(
            "Foundry simulation fleet did not attest every declared worker: "
            f"expected={list(expected_workers)} actual={list(attested_workers)}"
        )
    print(json.dumps({
        "schemaVersion": 1,
        "campaignHash": spec.campaign_hash,
        "workerCount": spec.worker_count,
        "lanes": list(lanes),
        "runId": spec.run_id,
        "timeoutSeconds": spec.timeout_seconds,
        "attestedWorkers": list(attested_workers),
        "status": final_status,
    }, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":  # pragma: no cover - module execution
    main()
