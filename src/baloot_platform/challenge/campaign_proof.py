"""Partition-independent proof contracts for deterministic Foundry campaigns."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .campaign import (
    SimulationCampaignSpecV1,
    SimulationShardResultV1,
    canonical_sha256,
    deterministic_campaign_reduction,
)


@dataclass(frozen=True, slots=True)
class CampaignProofProfileV1:
    profile_id: str
    revision: str
    minimum_organic_rounds: int
    required_worker_counts: tuple[int, ...]
    required_failure_schedules: tuple[str, ...]
    require_attempt_ledger: bool
    sha256: str

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "CampaignProofProfileV1":
        allowed = {
            "schemaVersion", "id", "revision", "minimumOrganicRounds",
            "requiredWorkerCounts", "requiredFailureSchedules",
            "requireAttemptLedger",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown campaign proof profile keys: {sorted(unknown)}")
        if int(value.get("schemaVersion", 0)) != 1:
            raise ValueError("unsupported campaign proof profile schema")
        workers = tuple(sorted(set(int(item) for item in value.get("requiredWorkerCounts") or ())))
        schedules = tuple(sorted({
            str(item).strip() for item in value.get("requiredFailureSchedules") or ()
            if str(item).strip()
        }))
        core = {
            "schemaVersion": 1,
            "id": str(value.get("id") or ""),
            "revision": str(value.get("revision") or ""),
            "minimumOrganicRounds": int(value.get("minimumOrganicRounds", 0)),
            "requiredWorkerCounts": list(workers),
            "requiredFailureSchedules": list(schedules),
            "requireAttemptLedger": bool(value.get("requireAttemptLedger", True)),
        }
        if not core["id"] or not core["revision"]:
            raise ValueError("campaign proof profile identity is required")
        if not 1 <= core["minimumOrganicRounds"] <= 1_000_000_000:
            raise ValueError("campaign proof round target is out of range")
        if not workers or any(item < 1 or item > 100_000 for item in workers):
            raise ValueError("campaign proof worker counts are invalid")
        if not schedules:
            raise ValueError("campaign proof failure schedules are required")
        return cls(
            core["id"], core["revision"], core["minimumOrganicRounds"],
            workers, schedules, core["requireAttemptLedger"], canonical_sha256(core),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "id": self.profile_id,
            "revision": self.revision,
            "minimumOrganicRounds": self.minimum_organic_rounds,
            "requiredWorkerCounts": list(self.required_worker_counts),
            "requiredFailureSchedules": list(self.required_failure_schedules),
            "requireAttemptLedger": self.require_attempt_ledger,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class CampaignWorkloadIdentityV1:
    rules_hash: str
    bundle_hash: str
    engine_version: str
    analysis_profile_hash: str
    perspective_profile_hash: str
    selection_profile_hash: str
    agent_release: str
    seat_agents: tuple[str, ...]
    lanes: tuple[str, ...]
    lane_profile_hashes: tuple[tuple[str, str], ...]
    seed_start: int
    seed_count: int
    sha256: str

    @classmethod
    def from_campaign(cls, campaign: SimulationCampaignSpecV1) -> "CampaignWorkloadIdentityV1":
        core = {
            "schemaVersion": 1,
            "rulesHash": campaign.rules_hash,
            "bundleHash": campaign.bundle_hash,
            "engineVersion": campaign.engine_version,
            "analysisProfileHash": campaign.analysis_profile_hash,
            "perspectiveProfileHash": campaign.perspective_profile_hash,
            "selectionProfileHash": campaign.selection_profile_hash,
            "agentRelease": campaign.agent_release,
            "seatAgents": list(campaign.seat_agents),
            "lanes": list(campaign.lanes),
            "laneProfileHashes": dict(campaign.lane_profile_hashes),
            "seedStart": campaign.seed_start,
            "seedCount": campaign.seed_count,
        }
        return cls(
            campaign.rules_hash, campaign.bundle_hash, campaign.engine_version,
            campaign.analysis_profile_hash, campaign.perspective_profile_hash,
            campaign.selection_profile_hash, campaign.agent_release,
            campaign.seat_agents, campaign.lanes, campaign.lane_profile_hashes,
            campaign.seed_start, campaign.seed_count, canonical_sha256(core),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "rulesHash": self.rules_hash,
            "bundleHash": self.bundle_hash,
            "engineVersion": self.engine_version,
            "analysisProfileHash": self.analysis_profile_hash,
            "perspectiveProfileHash": self.perspective_profile_hash,
            "selectionProfileHash": self.selection_profile_hash,
            "agentRelease": self.agent_release,
            "seatAgents": list(self.seat_agents),
            "lanes": list(self.lanes),
            "laneProfileHashes": dict(self.lane_profile_hashes),
            "seedStart": self.seed_start,
            "seedCount": self.seed_count,
            "sha256": self.sha256,
        }


def _ranges(values: Iterable[int]) -> tuple[tuple[int, int], ...]:
    ordered = tuple(sorted(set(int(item) for item in values)))
    if not ordered:
        return ()
    result: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        result.append((start, previous + 1))
        start = previous = value
    result.append((start, previous + 1))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class CampaignExecutionEvidenceV1:
    workload: CampaignWorkloadIdentityV1
    campaign_hash: str
    seeds_per_shard: int
    shard_count: int
    declared_worker_count: int
    failure_schedule: str
    attested_worker_ids: tuple[str, ...]
    attempt_ledger_hash: str | None
    failure_schedule_attested: bool
    shard_attempt_counts: tuple[tuple[str, int], ...]
    coverage_by_lane: tuple[tuple[str, tuple[tuple[int, int], ...]], ...]
    organic_round_count: int
    normalized_reduction: Mapping[str, Any]
    normalized_evidence_hash: str
    sha256: str

    @classmethod
    def build(
        cls,
        campaign: SimulationCampaignSpecV1,
        results: Sequence[SimulationShardResultV1],
        *,
        declared_worker_count: int,
        failure_schedule: str,
        shard_attempt_counts: Mapping[str, int] | None = None,
        attested_worker_ids: Iterable[str] = (),
        attempt_ledger_hash: str | None = None,
        failure_schedule_attested: bool = False,
    ) -> "CampaignExecutionEvidenceV1":
        workers = int(declared_worker_count)
        schedule = str(failure_schedule).strip()
        if workers < 1 or not schedule:
            raise ValueError("campaign execution requires a worker count and failure schedule")
        attested_workers = tuple(sorted({
            str(item).strip() for item in attested_worker_ids if str(item).strip()
        }))
        ledger_hash = str(attempt_ledger_hash or "") or None
        if attested_workers and len(attested_workers) != workers:
            raise ValueError("attested worker identities do not match the declared worker count")
        if (attested_workers or ledger_hash) and (
            not attested_workers
            or ledger_hash is None
            or len(ledger_hash) != 64
            or any(char not in "0123456789abcdef" for char in ledger_hash)
        ):
            raise ValueError("worker attestation requires identities and an attempt-ledger hash")
        reduction = deterministic_campaign_reduction(campaign, results)
        shards = {item.input_hash: item for item in campaign.shards()}
        coverage: dict[str, set[int]] = {lane: set() for lane in campaign.lanes}
        for result in results:
            spec = shards[result.spec_hash]
            lane_values = coverage[spec.lane]
            overlap = lane_values.intersection(result.processed_seeds)
            if overlap:
                raise ValueError("campaign execution contains duplicate lane-seed coverage")
            lane_values.update(result.processed_seeds)
        expected = set(range(campaign.seed_start, campaign.seed_start + campaign.seed_count))
        if any(values != expected for values in coverage.values()):
            raise ValueError("campaign execution coverage is incomplete")
        coverage_rows = tuple(
            (lane, _ranges(coverage[lane])) for lane in sorted(coverage)
        )
        attempts = tuple(sorted(
            (spec_hash, int((shard_attempt_counts or {}).get(spec_hash, 1)))
            for spec_hash in shards
        ))
        if any(value < 1 for _, value in attempts):
            raise ValueError("campaign shard attempt counts must be positive")
        workload = CampaignWorkloadIdentityV1.from_campaign(campaign)
        normalized = {
            "schemaVersion": 1,
            "workloadHash": workload.sha256,
            "traceHashes": list(reduction["traceHashes"]),
            "candidateHashes": list(reduction["candidateHashes"]),
            "coverageCells": list(reduction["coverageCells"]),
            "traceHashesByLane": dict(reduction["traceHashesByLane"]),
            "coverageCellsByLane": dict(reduction["coverageCellsByLane"]),
            "coverageByLane": {
                lane: [list(item) for item in ranges] for lane, ranges in coverage_rows
            },
            "distinctTraceCount": len(reduction["traceHashes"]),
            "distinctCandidateCount": len(reduction["candidateHashes"]),
        }
        normalized_hash = canonical_sha256(normalized)
        organic_rounds = campaign.seed_count if "organic" in campaign.lanes else 0
        core = {
            "schemaVersion": 1,
            "workloadHash": workload.sha256,
            "campaignHash": campaign.sha256,
            "seedsPerShard": campaign.seeds_per_shard,
            "shardCount": len(shards),
            "declaredWorkerCount": workers,
            "failureSchedule": schedule,
            "attestedWorkerIds": list(attested_workers),
            "attemptLedgerHash": ledger_hash,
            "failureScheduleAttested": bool(failure_schedule_attested),
            "shardAttemptCounts": dict(attempts),
            "organicRoundCount": organic_rounds,
            "normalizedEvidenceHash": normalized_hash,
            "shardResultHashes": sorted(item.sha256 for item in results),
        }
        return cls(
            workload, campaign.sha256, campaign.seeds_per_shard, len(shards),
            workers, schedule, attested_workers, ledger_hash,
            bool(failure_schedule_attested), attempts,
            coverage_rows, organic_rounds,
            normalized, normalized_hash, canonical_sha256(core),
        )

    @classmethod
    def from_store(
        cls,
        campaign_store: Any,
        campaign_hash: str,
        *,
        declared_worker_count: int,
        failure_schedule: str,
    ) -> "CampaignExecutionEvidenceV1":
        row = campaign_store.get_campaign(str(campaign_hash))
        if row is None:
            raise KeyError(campaign_hash)
        if row.get("status") != "completed" or bool(row.get("cancelRequested")):
            raise ValueError("campaign proof requires a completed non-cancelled campaign")
        campaign = SimulationCampaignSpecV1.from_dict(dict(row["spec"]))
        shard_rows = tuple(campaign_store.list_shards(campaign.sha256))
        specs = {item.input_hash: item for item in campaign.shards()}
        if len(shard_rows) != len(specs):
            raise ValueError("campaign proof requires every immutable shard")
        results = []
        attempts = {}
        for shard in shard_rows:
            spec_hash = str(shard.get("specHash") or "")
            spec = specs.get(spec_hash)
            if spec is None or shard.get("status") != "completed":
                raise ValueError("campaign proof contains an incomplete or unknown shard")
            results.append(SimulationShardResultV1.from_dict(spec, dict(shard.get("result") or {})))
            attempts[spec_hash] = int(shard.get("attempts", 0))
        attested_workers: tuple[str, ...] = ()
        attempt_ledger_hash: str | None = None
        schedule_attested = False
        load_attempts = getattr(campaign_store, "list_attempts", None)
        if callable(load_attempts):
            ledger = tuple(dict(item) for item in load_attempts(campaign.sha256))
            if not ledger or any(item.get("terminalStatus") is None for item in ledger):
                raise ValueError("campaign worker-attempt ledger is incomplete")
            attested_workers = tuple(sorted({str(item["workerId"]) for item in ledger}))
            if len(attested_workers) != int(declared_worker_count):
                raise ValueError("declared worker count differs from the attempt ledger")
            ledger_document = {
                "schemaVersion": 1,
                "campaignHash": campaign.sha256,
                "attempts": sorted(ledger, key=lambda item: (
                    str(item["specHash"]), int(item["fencingToken"])
                )),
            }
            attempt_ledger_hash = canonical_sha256(ledger_document)
            statuses = {str(item["terminalStatus"]) for item in ledger}
            if failure_schedule == "none":
                schedule_attested = statuses == {"completed"} and all(
                    int(item["attempt"]) == 1 for item in ledger
                )
            elif failure_schedule == "retry_reclaim":
                schedule_attested = bool(
                    statuses.intersection({"retryable_failure", "lease_expired"})
                )
            elif failure_schedule == "out_of_order":
                completed = sorted(
                    (item for item in ledger if item["terminalStatus"] == "completed"),
                    key=lambda item: (float(item["finishedAt"]), str(item["specHash"])),
                )
                schedule_attested = [int(item["shardIndex"]) for item in completed] != sorted(
                    int(item["shardIndex"]) for item in completed
                )
            if not schedule_attested:
                raise ValueError("declared failure schedule is not proven by the attempt ledger")
        return cls.build(
            campaign,
            tuple(results),
            declared_worker_count=declared_worker_count,
            failure_schedule=failure_schedule,
            shard_attempt_counts=attempts,
            attested_worker_ids=attested_workers,
            attempt_ledger_hash=attempt_ledger_hash,
            failure_schedule_attested=schedule_attested,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "workload": self.workload.to_dict(),
            "campaignHash": self.campaign_hash,
            "seedsPerShard": self.seeds_per_shard,
            "shardCount": self.shard_count,
            "declaredWorkerCount": self.declared_worker_count,
            "failureSchedule": self.failure_schedule,
            "attestedWorkerIds": list(self.attested_worker_ids),
            "attemptLedgerHash": self.attempt_ledger_hash,
            "failureScheduleAttested": self.failure_schedule_attested,
            "shardAttemptCounts": dict(self.shard_attempt_counts),
            "coverageByLane": {
                lane: [list(item) for item in ranges]
                for lane, ranges in self.coverage_by_lane
            },
            "organicRoundCount": self.organic_round_count,
            "normalizedReduction": dict(self.normalized_reduction),
            "normalizedEvidenceHash": self.normalized_evidence_hash,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class CampaignEquivalenceProofV1:
    profile_hash: str
    workload_hash: str
    executions: tuple[CampaignExecutionEvidenceV1, ...]
    normalized_evidence_hash: str
    status: str
    unmet_qualification_gates: tuple[str, ...]
    sha256: str

    @classmethod
    def build(
        cls,
        executions: Sequence[CampaignExecutionEvidenceV1],
        profile: CampaignProofProfileV1,
    ) -> "CampaignEquivalenceProofV1":
        values = tuple(sorted(executions, key=lambda item: item.sha256))
        if len(values) < 2 or len({item.campaign_hash for item in values}) < 2:
            raise ValueError("equivalence proof requires at least two distinct campaign executions")
        workload_hashes = {item.workload.sha256 for item in values}
        if len(workload_hashes) != 1:
            raise ValueError("equivalence proof executions do not share one logical workload")
        evidence_hashes = {item.normalized_evidence_hash for item in values}
        if len(evidence_hashes) != 1:
            raise ValueError("campaign outputs differ across partitions or schedules")
        gates: list[str] = []
        organic_rounds = min(item.organic_round_count for item in values)
        if organic_rounds < profile.minimum_organic_rounds:
            gates.append("minimum_organic_rounds")
        workers = {item.declared_worker_count for item in values}
        missing_workers = set(profile.required_worker_counts) - workers
        if missing_workers:
            gates.append("worker_counts:" + ",".join(str(item) for item in sorted(missing_workers)))
        schedules = {item.failure_schedule for item in values}
        missing_schedules = set(profile.required_failure_schedules) - schedules
        if missing_schedules:
            gates.append("failure_schedules:" + ",".join(sorted(missing_schedules)))
        if profile.require_attempt_ledger and any(
            not item.attested_worker_ids or item.attempt_ledger_hash is None
            for item in values
        ):
            gates.append("worker_attempt_attestation")
        if profile.require_attempt_ledger and any(
            not item.failure_schedule_attested for item in values
        ):
            gates.append("failure_schedule_attestation")
        status = "qualified" if not gates else "pilot_only"
        workload_hash = next(iter(workload_hashes))
        normalized_hash = next(iter(evidence_hashes))
        core = {
            "schemaVersion": 1,
            "profileHash": profile.sha256,
            "workloadHash": workload_hash,
            "executionHashes": [item.sha256 for item in values],
            "normalizedEvidenceHash": normalized_hash,
            "status": status,
            "unmetQualificationGates": gates,
        }
        return cls(
            profile.sha256, workload_hash, values, normalized_hash, status,
            tuple(gates), canonical_sha256(core),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "profileHash": self.profile_hash,
            "workloadHash": self.workload_hash,
            "executions": [item.to_dict() for item in self.executions],
            "normalizedEvidenceHash": self.normalized_evidence_hash,
            "status": self.status,
            "unmetQualificationGates": list(self.unmet_qualification_gates),
            "sha256": self.sha256,
        }


class CampaignProofServiceV1:
    def __init__(self, campaign_store: Any, catalogs: Any, profile: CampaignProofProfileV1):
        self.campaign_store = campaign_store
        self.catalogs = catalogs
        self.profile = profile

    def create(self, executions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        values = tuple(
            CampaignExecutionEvidenceV1.from_store(
                self.campaign_store,
                str(item.get("campaignHash") or ""),
                declared_worker_count=int(item.get("declaredWorkerCount", 0)),
                failure_schedule=str(item.get("failureSchedule") or ""),
            )
            for item in executions
        )
        proof = CampaignEquivalenceProofV1.build(values, self.profile)
        return self.catalogs.create_campaign_equivalence_proof(proof)
