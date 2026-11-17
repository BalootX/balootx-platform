from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

from .scenario_contracts import canonical_sha256


@dataclass(frozen=True, slots=True)
class ResearchShardSpecV1:
    """Immutable, deterministic seed partition for one Research run."""

    run_code: str
    run_hash: str
    recipe_hash: str
    shard_index: int
    seed_start: int
    seed_count: int
    maximum_attempts: int
    shard_id: str
    sha256: str

    @classmethod
    def build(
        cls,
        *,
        run_code: str,
        run_hash: str,
        recipe_hash: str,
        shard_index: int,
        seed_start: int,
        seed_count: int,
        maximum_attempts: int = 3,
    ) -> "ResearchShardSpecV1":
        index = int(shard_index)
        start = int(seed_start)
        count = int(seed_count)
        attempts = int(maximum_attempts)
        if index < 0 or start < 0 or count < 1:
            raise ValueError("research shard index/start/count are invalid")
        if not 1 <= attempts <= 1_000_000:
            raise ValueError(
                "research shard maximum attempts must be in [1, 1000000]"
            )
        identity = {
            "schemaVersion": 1,
            "runCode": str(run_code),
            "runHash": str(run_hash),
            "recipeHash": str(recipe_hash),
            "shardIndex": index,
            "seedStart": start,
            "seedCount": count,
            "maximumAttempts": attempts,
        }
        digest = canonical_sha256(identity)
        return cls(
            str(run_code),
            str(run_hash),
            str(recipe_hash),
            index,
            start,
            count,
            attempts,
            f"RSH-{digest[:16].upper()}",
            digest,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResearchShardSpecV1":
        raw = copy.deepcopy(dict(value))
        expected = {
            "schemaVersion", "shardId", "sha256", "runCode", "runHash",
            "recipeHash", "shardIndex", "seedStart", "seedCount",
            "seedEndExclusive", "maximumAttempts",
        }
        unknown = set(raw) - expected
        if unknown:
            raise ValueError(f"unknown research shard keys: {sorted(unknown)}")
        if int(raw.get("schemaVersion", 1)) != 1:
            raise ValueError("unsupported research shard schema version")
        built = cls.build(
            run_code=str(raw["runCode"]),
            run_hash=str(raw["runHash"]),
            recipe_hash=str(raw["recipeHash"]),
            shard_index=int(raw["shardIndex"]),
            seed_start=int(raw["seedStart"]),
            seed_count=int(raw["seedCount"]),
            maximum_attempts=int(raw.get("maximumAttempts", 3)),
        )
        if raw.get("shardId") not in {None, built.shard_id}:
            raise ValueError("research shard identity does not match its contents")
        if raw.get("sha256") not in {None, built.sha256}:
            raise ValueError("research shard hash does not match its contents")
        if int(raw.get("seedEndExclusive", built.seed_end_exclusive)) != built.seed_end_exclusive:
            raise ValueError("research shard seed boundary is invalid")
        return built

    @property
    def seed_end_exclusive(self) -> int:
        return self.seed_start + self.seed_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "shardId": self.shard_id,
            "sha256": self.sha256,
            "runCode": self.run_code,
            "runHash": self.run_hash,
            "recipeHash": self.recipe_hash,
            "shardIndex": self.shard_index,
            "seedStart": self.seed_start,
            "seedCount": self.seed_count,
            "seedEndExclusive": self.seed_end_exclusive,
            "maximumAttempts": self.maximum_attempts,
        }


def partition_research_shards(
    *,
    run_code: str,
    run_hash: str,
    recipe_hash: str,
    seed_start: int,
    seed_count: int,
    seeds_per_shard: int,
    maximum_attempts: int = 3,
) -> tuple[ResearchShardSpecV1, ...]:
    start = int(seed_start)
    total = int(seed_count)
    width = int(seeds_per_shard)
    if start < 0 or total < 1 or width < 1:
        raise ValueError("research shard partition inputs are invalid")
    rows: list[ResearchShardSpecV1] = []
    consumed = 0
    while consumed < total:
        count = min(width, total - consumed)
        rows.append(ResearchShardSpecV1.build(
            run_code=run_code,
            run_hash=run_hash,
            recipe_hash=recipe_hash,
            shard_index=len(rows),
            seed_start=start + consumed,
            seed_count=count,
            maximum_attempts=maximum_attempts,
        ))
        consumed += count
    return tuple(rows)


__all__ = ["ResearchShardSpecV1", "partition_research_shards"]
