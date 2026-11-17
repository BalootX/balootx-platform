from __future__ import annotations

import copy
import math
from datetime import datetime, timezone
from typing import Any, Mapping

from .contracts import canonical_sha256
from .sequential_candidate_search import SequentialCandidateSearchReceiptV1


def strict_positive_integer(value: int, *, field: str) -> int:
    """Return an exact positive integer without accepting coercive aliases."""

    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def normalize_lease_duration(lease_seconds: float) -> float:
    if isinstance(lease_seconds, bool):
        raise ValueError("research shard lease must be positive")
    try:
        seconds = float(lease_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("research shard lease must be positive") from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("research shard lease must be positive")
    return seconds


def normalize_lease_moment(moment: datetime | None) -> datetime:
    selected = datetime.now(timezone.utc) if moment is None else moment
    if not isinstance(selected, datetime) or selected.tzinfo is None:
        raise ValueError("research shard lease time must be timezone-aware")
    try:
        offset = selected.utcoffset()
    except (OverflowError, ValueError) as exc:
        raise ValueError("research shard lease time must be timezone-aware") from exc
    if offset is None:
        raise ValueError("research shard lease time must be timezone-aware")
    return selected.astimezone(timezone.utc)


def normalize_fenced_attempt(
    *,
    shard_id: str | None,
    attempt_count: int | None,
    fencing_token: int | None,
    worker_id: str | None,
) -> tuple[str, int, int, str] | None:
    """Validate a write without ever inferring its execution fence.

    The all-``None`` branch is reserved for legacy or genuinely unsharded
    writes. Stores reject that branch when the target run already has shards.
    """

    if shard_id is None:
        if (
            attempt_count is not None
            or fencing_token is not None
            or worker_id is not None
        ):
            raise ValueError(
                "attempt_count, fencing_token, and worker_id require a research shard"
            )
        return None
    if attempt_count is None or fencing_token is None or worker_id is None:
        raise ValueError(
            "new sharded writes require explicit attempt_count, fencing_token, "
            "and worker_id"
        )
    normalized_attempt = strict_positive_integer(
        attempt_count, field="research attempt_count"
    )
    normalized_fence = strict_positive_integer(
        fencing_token, field="research fencing_token"
    )
    normalized_shard = str(shard_id).strip()
    if not normalized_shard:
        raise ValueError("research shard id must not be empty")
    normalized_worker = str(worker_id).strip()
    if not normalized_worker:
        raise ValueError("research worker id must not be empty")
    return (
        normalized_shard,
        normalized_attempt,
        normalized_fence,
        normalized_worker,
    )


def build_fenced_occurrence(
    *,
    kind: str,
    semantic_hash: str,
    run_code: str,
    shard_id: str,
    attempt_count: int,
    fencing_token: int,
    seed: int | None = None,
) -> dict[str, Any]:
    core: dict[str, Any] = {
        "schemaVersion": 1,
        "kind": str(kind),
        "semanticHash": str(semantic_hash),
        "runCode": str(run_code),
        "shardId": str(shard_id),
        "attemptCount": int(attempt_count),
        "fencingToken": int(fencing_token),
    }
    if seed is not None:
        core["seed"] = int(seed)
    return {
        **core,
        "occurrenceHash": canonical_sha256(core),
        "admitted": False,
    }


def build_sequential_search_receipt_bundle(
    receipt: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    normalized_seed = int(seed)
    if normalized_seed < 0:
        raise ValueError("research seed must be non-negative")
    validated = SequentialCandidateSearchReceiptV1.from_dict(
        copy.deepcopy(dict(receipt))
    ).verify()
    bundle = validated.to_dict()
    if int(bundle["rootSeed"]) != normalized_seed:
        raise ValueError("sequential search receipt seed does not match rootSeed")
    return bundle


__all__ = [
    "build_fenced_occurrence",
    "build_sequential_search_receipt_bundle",
    "normalize_lease_duration",
    "normalize_lease_moment",
    "normalize_fenced_attempt",
    "strict_positive_integer",
]
