"""Deterministic Foundry campaign and shard contracts."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .evidence_v3 import canonical_sha256


DISCOVERY_LANES = frozenset({
    "organic", "guided", "disagreement", "historical", "expert",
})


def _is_sha256(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", str(value)))


@dataclass(frozen=True, slots=True)
class SimulationShardSpecV1:
    campaign_hash: str
    shard_index: int
    lane: str
    lane_profile_hash: str
    seed_start: int
    seed_count: int
    input_hash: str

    @property
    def seed_end_exclusive(self) -> int:
        return self.seed_start + self.seed_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "campaignHash": self.campaign_hash,
            "shardIndex": self.shard_index,
            "lane": self.lane,
            "laneProfileHash": self.lane_profile_hash,
            "seedStart": self.seed_start,
            "seedCount": self.seed_count,
            "seedEndExclusive": self.seed_end_exclusive,
            "inputHash": self.input_hash,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SimulationShardSpecV1":
        identity = {
            "campaignHash": str(value["campaignHash"]),
            "shardIndex": int(value["shardIndex"]),
            "lane": str(value.get("lane") or "organic"),
            "laneProfileHash": str(value.get("laneProfileHash") or ""),
            "seedStart": int(value["seedStart"]),
            "seedCount": int(value["seedCount"]),
        }
        if identity["lane"] not in DISCOVERY_LANES:
            raise ValueError("simulation shard has an unsupported discovery lane")
        if value.get("laneProfileHash") and not _is_sha256(identity["laneProfileHash"]):
            raise ValueError("simulation shard lane profile must be a SHA-256 identity")
        hash_identity = dict(identity)
        if "lane" not in value:
            hash_identity.pop("lane")
        if "laneProfileHash" not in value:
            hash_identity.pop("laneProfileHash")
        expected = canonical_sha256(hash_identity)
        supplied = str(value.get("inputHash") or expected)
        if supplied != expected:
            raise ValueError("simulation shard input hash mismatch")
        return cls(
            identity["campaignHash"], identity["shardIndex"], identity["lane"],
            identity["laneProfileHash"],
            identity["seedStart"], identity["seedCount"], supplied,
        )


@dataclass(frozen=True, slots=True)
class SimulationCampaignSpecV1:
    rules_hash: str
    bundle_hash: str
    engine_version: str
    analysis_profile_hash: str
    perspective_profile_hash: str
    selection_profile_hash: str
    agent_release: str
    seat_agents: tuple[str, str, str, str]
    lanes: tuple[str, ...]
    lane_profile_hashes: tuple[tuple[str, str], ...]
    seed_start: int
    seed_count: int
    seeds_per_shard: int
    sha256: str

    @classmethod
    def build(
        cls,
        *,
        rules_hash: str,
        bundle_hash: str,
        engine_version: str,
        analysis_profile_hash: str,
        perspective_profile_hash: str,
        selection_profile_hash: str,
        agent_release: str,
        seat_agents: Iterable[str],
        lanes: Iterable[str],
        lane_profile_hashes: Mapping[str, str],
        seed_start: int,
        seed_count: int,
        seeds_per_shard: int,
    ) -> "SimulationCampaignSpecV1":
        agents = tuple(str(item) for item in seat_agents)
        if len(agents) != 4 or len(set(agents)) != 1 or agents[0] != str(agent_release):
            raise ValueError("campaigns require one equal signed agent release for all four seats")
        lane_values = tuple(sorted(set(str(item) for item in lanes)))
        if not lane_values or set(lane_values) - DISCOVERY_LANES:
            raise ValueError("campaign contains an unsupported or empty discovery lane")
        lane_profiles = tuple(sorted(
            (str(lane), str(profile_hash))
            for lane, profile_hash in lane_profile_hashes.items()
        ))
        if {lane for lane, _ in lane_profiles} != set(lane_values):
            raise ValueError("campaign requires exactly one profile hash per discovery lane")
        if any(not _is_sha256(profile_hash) for _, profile_hash in lane_profiles):
            raise ValueError("campaign lane profiles must be SHA-256 identities")
        if int(seed_start) < 0 or int(seed_count) < 1 or int(seeds_per_shard) < 1:
            raise ValueError("campaign seed range and shard size must be positive")
        hashes = (
            rules_hash,
            bundle_hash,
            analysis_profile_hash,
            perspective_profile_hash,
            selection_profile_hash,
        )
        if any(not _is_sha256(str(item)) for item in hashes):
            raise ValueError("campaign configuration references must be SHA-256 identities")
        core = {
            "schemaVersion": 1,
            "rulesHash": rules_hash,
            "bundleHash": bundle_hash,
            "engineVersion": str(engine_version),
            "analysisProfileHash": analysis_profile_hash,
            "perspectiveProfileHash": perspective_profile_hash,
            "selectionProfileHash": selection_profile_hash,
            "agentRelease": str(agent_release),
            "seatAgents": list(agents),
            "lanes": list(lane_values),
            "laneProfileHashes": dict(lane_profiles),
            "seedStart": int(seed_start),
            "seedCount": int(seed_count),
            "seedsPerShard": int(seeds_per_shard),
        }
        return cls(
            str(rules_hash), str(bundle_hash), str(engine_version),
            str(analysis_profile_hash), str(perspective_profile_hash),
            str(selection_profile_hash), str(agent_release), agents,
            lane_values, lane_profiles,
            int(seed_start), int(seed_count), int(seeds_per_shard),
            canonical_sha256(core),
        )

    def shards(self) -> tuple[SimulationShardSpecV1, ...]:
        result: list[SimulationShardSpecV1] = []
        index = 0
        profiles = dict(self.lane_profile_hashes)
        for lane in self.lanes:
            remaining = self.seed_count
            start = self.seed_start
            while remaining:
                count = min(self.seeds_per_shard, remaining)
                identity = {
                    "campaignHash": self.sha256,
                    "shardIndex": index,
                    "lane": lane,
                    "laneProfileHash": profiles[lane],
                    "seedStart": start,
                    "seedCount": count,
                }
                result.append(
                    SimulationShardSpecV1(
                        self.sha256, index, lane, profiles[lane], start, count,
                        canonical_sha256(identity),
                    )
                )
                start += count
                remaining -= count
                index += 1
        return tuple(result)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SimulationCampaignSpecV1":
        result = cls.build(
            rules_hash=str(value["rulesHash"]),
            bundle_hash=str(value["bundleHash"]),
            engine_version=str(value["engineVersion"]),
            analysis_profile_hash=str(value["analysisProfileHash"]),
            perspective_profile_hash=str(value["perspectiveProfileHash"]),
            selection_profile_hash=str(value["selectionProfileHash"]),
            agent_release=str(value["agentRelease"]),
            seat_agents=tuple(value["seatAgents"]),
            lanes=tuple(value["lanes"]),
            lane_profile_hashes=dict(
                value.get("laneProfileHashes")
                or {lane: value["selectionProfileHash"] for lane in value["lanes"]}
            ),
            seed_start=int(value["seedStart"]),
            seed_count=int(value["seedCount"]),
            seeds_per_shard=int(value["seedsPerShard"]),
        )
        if value.get("sha256") and str(value["sha256"]) != result.sha256:
            raise ValueError("simulation campaign hash mismatch")
        return result

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
            "seedsPerShard": self.seeds_per_shard,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class SimulationShardResultV1:
    spec_hash: str
    lane: str
    lane_profile_hash: str
    processed_seeds: tuple[int, ...]
    trace_hashes: tuple[str, ...]
    candidate_hashes: tuple[str, ...]
    coverage_cells: tuple[str, ...]
    artifact_references: tuple[dict[str, Any], ...]
    completion_status: str
    sha256: str

    @classmethod
    def build(
        cls,
        spec: SimulationShardSpecV1,
        *,
        processed_seeds: Iterable[int],
        trace_hashes: Iterable[str],
        candidate_hashes: Iterable[str],
        coverage_cells: Iterable[str],
        artifact_references: Iterable[dict[str, Any]] = (),
        completion_status: str,
    ) -> "SimulationShardResultV1":
        seeds = tuple(sorted(set(int(item) for item in processed_seeds)))
        expected = tuple(range(spec.seed_start, spec.seed_end_exclusive))
        status = str(completion_status)
        if status not in {"completed", "cancelled", "failed"}:
            raise ValueError("unsupported shard completion status")
        if status == "completed" and seeds != expected:
            raise ValueError("a completed shard must prove every assigned seed exactly once")
        if any(seed not in expected for seed in seeds):
            raise ValueError("shard result contains a seed outside its immutable assignment")
        traces = tuple(sorted(set(str(item) for item in trace_hashes)))
        candidates = tuple(sorted(set(str(item) for item in candidate_hashes)))
        coverage = tuple(sorted(set(str(item) for item in coverage_cells)))
        artifacts = tuple(sorted(
            (dict(item) for item in artifact_references),
            key=lambda item: (
                str(item.get("ownerKind", "")),
                str(item.get("ownerHash", "")),
                str(item.get("sha256", "")),
            ),
        ))
        for artifact in artifacts:
            if not artifact.get("ownerKind") or not _is_sha256(str(artifact.get("ownerHash", ""))):
                raise ValueError("shard artifact requires a typed content-addressed owner")
            if not artifact.get("uri") or not _is_sha256(str(artifact.get("sha256", ""))):
                raise ValueError("shard artifact requires a URI and SHA-256 identity")
            if int(artifact.get("bytes", -1)) < 0 or not artifact.get("mediaType"):
                raise ValueError("shard artifact requires size and media type")
        if any(not _is_sha256(item) for item in (*traces, *candidates)):
            raise ValueError("trace and candidate results require SHA-256 identities")
        core = {
            "schemaVersion": 1,
            "specHash": spec.input_hash,
            "lane": spec.lane,
            "laneProfileHash": spec.lane_profile_hash,
            "processedSeeds": list(seeds),
            "traceHashes": list(traces),
            "candidateHashes": list(candidates),
            "coverageCells": list(coverage),
            "artifactReferences": list(artifacts),
            "completionStatus": status,
        }
        return cls(
            spec.input_hash, spec.lane, spec.lane_profile_hash,
            seeds, traces, candidates, coverage, artifacts, status,
            canonical_sha256(core),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "specHash": self.spec_hash,
            "lane": self.lane,
            "laneProfileHash": self.lane_profile_hash,
            "processedSeeds": list(self.processed_seeds),
            "traceHashes": list(self.trace_hashes),
            "candidateHashes": list(self.candidate_hashes),
            "coverageCells": list(self.coverage_cells),
            "artifactReferences": [dict(item) for item in self.artifact_references],
            "completionStatus": self.completion_status,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(
        cls, spec: SimulationShardSpecV1, value: dict[str, Any]
    ) -> "SimulationShardResultV1":
        result = cls.build(
            spec,
            processed_seeds=tuple(value.get("processedSeeds") or ()),
            trace_hashes=tuple(value.get("traceHashes") or ()),
            candidate_hashes=tuple(value.get("candidateHashes") or ()),
            coverage_cells=tuple(value.get("coverageCells") or ()),
            artifact_references=tuple(value.get("artifactReferences") or ()),
            completion_status=str(value["completionStatus"]),
        )
        if value.get("lane") and str(value["lane"]) != spec.lane:
            raise ValueError("simulation shard result lane mismatch")
        if value.get("laneProfileHash") and str(value["laneProfileHash"]) != spec.lane_profile_hash:
            raise ValueError("simulation shard result lane profile mismatch")
        if value.get("sha256") and str(value["sha256"]) != result.sha256:
            raise ValueError("simulation shard result hash mismatch")
        return result


def deterministic_campaign_reduction(
    campaign: SimulationCampaignSpecV1,
    results: Iterable[SimulationShardResultV1],
) -> dict[str, Any]:
    items = tuple(results)
    expected = {item.input_hash for item in campaign.shards()}
    indexed = {item.spec_hash: item for item in items}
    if set(indexed) != expected or len(indexed) != len(items):
        raise ValueError("campaign reduction requires exactly one result per immutable shard")
    if any(item.completion_status != "completed" for item in items):
        raise ValueError("partial, failed, or cancelled shards cannot become campaign evidence")
    core = {
        "schemaVersion": 1,
        "campaignHash": campaign.sha256,
        "shardResultHashes": sorted(item.sha256 for item in items),
        "traceHashes": sorted({value for item in items for value in item.trace_hashes}),
        "candidateHashes": sorted({value for item in items for value in item.candidate_hashes}),
        "coverageCells": sorted({value for item in items for value in item.coverage_cells}),
        "traceHashesByLane": {
            lane: sorted({
                value for item in items if item.lane == lane for value in item.trace_hashes
            })
            for lane in campaign.lanes
        },
        "coverageCellsByLane": {
            lane: sorted({
                value for item in items if item.lane == lane for value in item.coverage_cells
            })
            for lane in campaign.lanes
        },
    }
    return {**core, "sha256": canonical_sha256(core)}
