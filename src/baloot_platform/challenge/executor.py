"""Reference deterministic batch executor for immutable simulation shards."""
from __future__ import annotations

from typing import Any, Callable, Mapping

from baloot_platform.plugins.base import PluginDescriptor

from .campaign import SimulationShardResultV1, SimulationShardSpecV1


class DeterministicBatchExecutorV1:
    descriptor = PluginDescriptor(
        id="deterministic_batch_executor_v1",
        kind="batch_executor",
        version="1.0.0",
        description="Ordered, cancellable execution of immutable Foundry seed shards",
        capabilities=("deterministic_seed_order", "cooperative_cancellation", "partial_rejection"),
        resource_class="cpu",
        deterministic=True,
        timeout_seconds=900,
        maximum_memory_mb=4096,
    )

    def execute_batch(
        self,
        shard: SimulationShardSpecV1,
        *,
        simulate_seed: Callable[[int], Mapping[str, Any]],
        cancel_check: Callable[[], bool] | None = None,
    ) -> SimulationShardResultV1:
        processed: list[int] = []
        traces: list[str] = []
        candidates: list[str] = []
        coverage: list[str] = []
        artifacts: list[dict[str, Any]] = []
        status = "completed"
        for seed in range(shard.seed_start, shard.seed_end_exclusive):
            if cancel_check is not None and cancel_check():
                status = "cancelled"
                break
            output = dict(simulate_seed(seed))
            trace_hash = str(output.get("traceHash") or "")
            if len(trace_hash) != 64:
                raise ValueError("simulation seed output requires a complete trace hash")
            processed.append(seed)
            seed_traces = tuple(str(item) for item in output.get("traceHashes") or (trace_hash,))
            if trace_hash not in seed_traces or any(len(item) != 64 for item in seed_traces):
                raise ValueError("simulation attempt traces must contain valid complete trace hashes")
            traces.extend(seed_traces)
            candidates.extend(str(item) for item in output.get("candidateHashes") or ())
            coverage.extend(str(item) for item in output.get("coverageCells") or ())
            artifacts.extend(dict(item) for item in output.get("artifactReferences") or ())
        return SimulationShardResultV1.build(
            shard,
            processed_seeds=processed,
            trace_hashes=traces,
            candidate_hashes=candidates,
            coverage_cells=coverage,
            artifact_references=artifacts,
            completion_status=status,
        )
