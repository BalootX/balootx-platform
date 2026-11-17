"""Leased and fenced local control plane for deterministic Foundry shards."""
from __future__ import annotations

import json
import hashlib
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .campaign import (
    DISCOVERY_LANES,
    SimulationCampaignSpecV1,
    SimulationShardResultV1,
    SimulationShardSpecV1,
)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _audit_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _audit_record(payload: Mapping[str, Any]) -> Any:
    # Import lazily: research is an application layer that imports challenge modules.
    from baloot_platform.research.audit_trace import ResearchAuditTraceRecordV1

    return ResearchAuditTraceRecordV1.build(payload)


def _audit_source_occurrence_hash(record: Any) -> str:
    return hashlib.sha256(_canonical(record.source).encode("utf-8")).hexdigest()


def _audit_identity(record: Any) -> tuple[Any, ...]:
    return (
        record.record_hash,
        record.trace_hash,
        record.audit_trace_code,
        record.complete_trace["sha256"],
        record.complete_trace["bytes"],
        record.complete_trace["mediaType"],
        record.source,
        record.replay_validation,
        record.observer_safety,
        record.configuration_provenance,
    )


def _audit_index_record(row: Mapping[str, Any]) -> Any:
    raw = row["record_json"]
    record = _audit_record(json.loads(raw) if isinstance(raw, str) else raw)
    source = record.source
    expected = {
        "record_hash": record.record_hash,
        "audit_trace_code": record.audit_trace_code,
        "trace_hash": record.trace_hash,
        "artifact_sha256": record.complete_trace["sha256"],
        "artifact_bytes": int(record.complete_trace["bytes"]),
        "source_occurrence_hash": _audit_source_occurrence_hash(record),
        "campaign_hash": source["runCode"],
        "spec_hash": source["shardId"],
        "attempt": int(source["attemptCount"]),
        "fencing_token": int(source["fencingToken"]),
        "candidate_hash": source["candidateHash"],
        "candidate_ordinal": int(source["candidateOrdinal"]),
        "seed": int(source["seed"]),
        "disposition": record.disposition,
        "status": record.status,
        "attempt_status": record.attempt_status,
    }
    if any(str(row[key]) != str(value) for key, value in expected.items()):
        raise ValueError("Foundry audit trace index does not match its record")
    return record


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _audit_with_attempt_lifecycle(
    record: Any,
    *,
    attempt_status: str,
    failure_categories: Iterable[str] = (),
    updated_at: str,
) -> Any:
    return _audit_record(
        {
            **record.to_dict(),
            "attemptStatus": str(attempt_status),
            "failureCategories": sorted(
                {
                    *record.failure_categories,
                    *(str(value).strip() for value in failure_categories if str(value).strip()),
                }
            ),
            "updatedAt": str(updated_at),
        }
    )


def _validate_campaign_audit_source(
    record: Any,
    *,
    campaign_hash: str,
    campaign_spec: Mapping[str, Any],
    shard_spec: Mapping[str, Any],
    spec_hash: str,
    shard_index: int,
    attempt: int,
    worker_id: str,
    fencing_token: int,
) -> str:
    source = record.source
    common = (
        str(source["runCode"]) == str(campaign_hash)
        and str(source["runHash"]) == str(campaign_hash)
        and str(source["shardId"]) == str(spec_hash)
        and str(source["shardHash"]) == str(spec_hash)
        and int(source["shardIndex"]) == int(shard_index)
        and int(source["attemptCount"]) == int(attempt)
        and str(source["workerId"]) == str(worker_id)
        and int(source["fencingToken"]) == int(fencing_token)
    )
    if not common:
        raise ValueError("Foundry audit trace source identity mismatch")
    seed = int(source["seed"])
    seed_start = int(shard_spec["seedStart"])
    seed_end = seed_start + int(shard_spec["seedCount"])
    if seed < seed_start or seed >= seed_end:
        raise ValueError("Foundry audit trace seed is outside its claimed shard")
    kind = str(source["kind"])
    if kind == "organic_foundry_trace":
        if (
            str(source["recipeCode"])
            not in {"organic-screening-v1", "organic-unscreened-v1"}
            or str(source["recipeHash"]) != str(shard_spec["laneProfileHash"])
            or str(source["generationLane"]) != "organic"
            or str(shard_spec["lane"]) != "organic"
        ):
            raise ValueError("Foundry audit trace organic profile identity mismatch")
    elif kind == "organic_finalist_trace":
        if (
            str(source["recipeCode"]) != "organic-finalist-validation-v1"
            or str(source["recipeHash"]) != str(source["stageProfileHash"])
            or str(source["generationLane"]) != "organic_finalist"
            or str(source["traceRole"]) != "finalist_descendant"
            or not _is_sha256(source["stageRunHash"])
        ):
            raise ValueError("Foundry finalist audit trace stage identity mismatch")
    else:
        raise ValueError("unsupported Foundry audit trace source kind")
    if (
        str(record.configuration_provenance["bundleHash"])
        != str(campaign_spec["bundleHash"])
        or str(record.configuration_provenance["rulesHash"])
        != str(campaign_spec["rulesHash"])
        or str(record.configuration_provenance["engineVersion"])
        != str(campaign_spec["engineVersion"])
    ):
        raise ValueError("Foundry audit trace campaign configuration mismatch")
    return kind


def _effective_campaign_status(
    stored_status: str,
    *,
    cancel_requested: bool,
    shard_counts: dict[str, int],
) -> str:
    if stored_status in {"completed", "failed", "cancelled"}:
        return stored_status
    if shard_counts.get("failed", 0) > 0:
        return "failed"
    if cancel_requested or shard_counts.get("cancelled", 0) > 0:
        return "cancelled"
    total = sum(shard_counts.values())
    if total > 0 and shard_counts.get("completed", 0) == total:
        return "completed"
    if shard_counts.get("running", 0) > 0:
        return "running"
    return "running" if stored_status == "running" else "queued"


@dataclass(frozen=True, slots=True)
class ClaimedSimulationShardV1:
    spec: SimulationShardSpecV1
    worker_id: str
    fencing_token: int
    attempt: int
    lease_until: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "spec": self.spec.to_dict(),
            "workerId": self.worker_id,
            "fencingToken": self.fencing_token,
            "attempt": self.attempt,
            "leaseUntil": self.lease_until,
        }


class SQLiteCampaignControlStore:
    """Development/pilot adapter with the same lease and fencing semantics required at scale."""

    def __init__(self, path: str | Path, *, maximum_attempts: int):
        if int(maximum_attempts) < 1:
            raise ValueError("campaign maximum attempts must be positive")
        self.path = str(Path(path))
        self.maximum_attempts = int(maximum_attempts)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS foundry_campaigns_v1(
                    campaign_hash TEXT PRIMARY KEY,
                    spec_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS foundry_shards_v1(
                    spec_hash TEXT PRIMARY KEY,
                    campaign_hash TEXT NOT NULL,
                    shard_index INTEGER NOT NULL,
                    spec_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    worker_id TEXT,
                    lease_until REAL,
                    fencing_token INTEGER NOT NULL DEFAULT 0,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    result_json TEXT,
                    error TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(campaign_hash, shard_index),
                    FOREIGN KEY(campaign_hash) REFERENCES foundry_campaigns_v1(campaign_hash)
                );
                CREATE INDEX IF NOT EXISTS idx_foundry_shards_claim_v1
                  ON foundry_shards_v1(status,lease_until,campaign_hash,shard_index);
                CREATE TABLE IF NOT EXISTS foundry_shard_attempts_v1(
                    spec_hash TEXT NOT NULL,
                    campaign_hash TEXT NOT NULL,
                    fencing_token INTEGER NOT NULL,
                    attempt INTEGER NOT NULL,
                    worker_id TEXT NOT NULL,
                    claimed_at REAL NOT NULL,
                    finished_at REAL,
                    terminal_status TEXT,
                    error TEXT,
                    PRIMARY KEY(spec_hash,fencing_token),
                    FOREIGN KEY(spec_hash) REFERENCES foundry_shards_v1(spec_hash),
                    FOREIGN KEY(campaign_hash) REFERENCES foundry_campaigns_v1(campaign_hash)
                );
                CREATE INDEX IF NOT EXISTS idx_foundry_attempts_campaign_v1
                  ON foundry_shard_attempts_v1(campaign_hash,spec_hash,fencing_token);
                CREATE TABLE IF NOT EXISTS foundry_audit_traces_v1(
                    record_hash TEXT PRIMARY KEY,
                    audit_trace_code TEXT NOT NULL UNIQUE,
                    trace_hash TEXT NOT NULL,
                    artifact_sha256 TEXT NOT NULL,
                    artifact_bytes INTEGER NOT NULL,
                    source_occurrence_hash TEXT NOT NULL UNIQUE,
                    campaign_hash TEXT NOT NULL,
                    spec_hash TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    fencing_token INTEGER NOT NULL,
                    candidate_hash TEXT NOT NULL,
                    candidate_ordinal INTEGER NOT NULL,
                    seed INTEGER NOT NULL,
                    disposition TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt_status TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    admitted INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(campaign_hash) REFERENCES foundry_campaigns_v1(campaign_hash),
                    FOREIGN KEY(spec_hash) REFERENCES foundry_shards_v1(spec_hash)
                );
                CREATE INDEX IF NOT EXISTS idx_foundry_audit_library_v1
                  ON foundry_audit_traces_v1(
                    campaign_hash,admitted,disposition,created_at DESC,trace_hash
                  );
                CREATE INDEX IF NOT EXISTS idx_foundry_audit_trace_v1
                  ON foundry_audit_traces_v1(trace_hash,created_at DESC,record_hash);
                CREATE INDEX IF NOT EXISTS idx_foundry_audit_artifact_v1
                  ON foundry_audit_traces_v1(artifact_sha256);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def close(self) -> None:
        return None

    def create_campaign(self, spec: SimulationCampaignSpecV1) -> dict[str, Any]:
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT spec_json FROM foundry_campaigns_v1 WHERE campaign_hash=?",
                (spec.sha256,),
            ).fetchone()
            if existing is not None:
                if json.loads(existing[0]) != spec.to_dict():
                    raise RuntimeError("immutable campaign hash collision")
                return self.get_campaign(spec.sha256) or {}
            conn.execute(
                "INSERT INTO foundry_campaigns_v1(campaign_hash,spec_json,status) VALUES(?,?,'queued')",
                (spec.sha256, _canonical(spec.to_dict())),
            )
            for shard in spec.shards():
                conn.execute(
                    """INSERT INTO foundry_shards_v1(
                           spec_hash,campaign_hash,shard_index,spec_json,status
                       ) VALUES(?,?,?,?,'queued')""",
                    (shard.input_hash, spec.sha256, shard.shard_index, _canonical(shard.to_dict())),
                )
        return self.get_campaign(spec.sha256) or {}

    def get_campaign(self, campaign_hash: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM foundry_campaigns_v1 WHERE campaign_hash=?", (campaign_hash,)
            ).fetchone()
            if row is None:
                return None
            counts = conn.execute(
                """SELECT status,COUNT(*) AS count FROM foundry_shards_v1
                   WHERE campaign_hash=? GROUP BY status""",
                (campaign_hash,),
            ).fetchall()
        shard_counts = {str(item["status"]): int(item["count"]) for item in counts}
        cancel_requested = bool(row["cancel_requested"])
        return {
            "campaignHash": row["campaign_hash"],
            "spec": json.loads(row["spec_json"]),
            "status": _effective_campaign_status(
                str(row["status"]),
                cancel_requested=cancel_requested,
                shard_counts=shard_counts,
            ),
            "cancelRequested": cancel_requested,
            "shardCounts": shard_counts,
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
        }

    def list_campaigns(
        self, *, after: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        size = max(1, min(500, int(limit)))
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT campaign_hash FROM foundry_campaigns_v1
                   WHERE (? IS NULL OR campaign_hash>?)
                   ORDER BY campaign_hash LIMIT ?""",
                (after, after, size),
            ).fetchall()
        return [
            value for row in rows
            if (value := self.get_campaign(str(row["campaign_hash"]))) is not None
        ]

    def list_shards(self, campaign_hash: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM foundry_shards_v1 WHERE campaign_hash=? ORDER BY shard_index",
                (campaign_hash,),
            ).fetchall()
        return [
            {
                "specHash": row["spec_hash"],
                "campaignHash": row["campaign_hash"],
                "shardIndex": int(row["shard_index"]),
                "spec": json.loads(row["spec_json"]),
                "status": row["status"],
                "workerId": row["worker_id"],
                "leaseUntil": row["lease_until"],
                "fencingToken": int(row["fencing_token"]),
                "attempts": int(row["attempts"]),
                "result": json.loads(row["result_json"]) if row["result_json"] else None,
                "error": row["error"],
            }
            for row in rows
        ]

    def get_shard(self, spec_hash: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT campaign_hash FROM foundry_shards_v1 WHERE spec_hash=?",
                (spec_hash,),
            ).fetchone()
        if row is None:
            return None
        return next(
            item for item in self.list_shards(str(row["campaign_hash"]))
            if item["specHash"] == spec_hash
        )

    @staticmethod
    def _update_audit_trace_attempts_locked(
        conn: sqlite3.Connection,
        *,
        spec_hash: str,
        fencing_token: int,
        attempt_status: str,
        failure_categories: Iterable[str] = (),
        updated_at: str,
    ) -> None:
        rows = conn.execute(
            """SELECT * FROM foundry_audit_traces_v1
               WHERE spec_hash=? AND fencing_token=?""",
            (str(spec_hash), int(fencing_token)),
        ).fetchall()
        for row in rows:
            current = _audit_index_record(row)
            value = _audit_with_attempt_lifecycle(
                current,
                attempt_status=attempt_status,
                failure_categories=failure_categories,
                updated_at=updated_at,
            )
            conn.execute(
                """UPDATE foundry_audit_traces_v1
                   SET attempt_status=?,disposition=?,status=?,record_json=?,updated_at=?
                   WHERE record_hash=? AND spec_hash=? AND fencing_token=?""",
                (
                    value.attempt_status,
                    value.disposition,
                    value.status,
                    _canonical(value.to_dict()),
                    updated_at,
                    value.record_hash,
                    str(spec_hash),
                    int(fencing_token),
                ),
            )

    @staticmethod
    def _validate_finalist_parent_locked(
        conn: sqlite3.Connection,
        record: Any,
    ) -> None:
        source = record.source
        row = conn.execute(
            "SELECT * FROM foundry_audit_traces_v1 WHERE record_hash=?",
            (str(source["parentRecordHash"]),),
        ).fetchone()
        if row is None:
            raise ValueError("Foundry finalist audit trace parent is unavailable")
        parent = _audit_index_record(row)
        common_keys = (
            "runCode",
            "runHash",
            "shardId",
            "shardHash",
            "shardIndex",
            "attemptCount",
            "fencingToken",
            "workerId",
        )
        if (
            str(parent.source["kind"]) != "organic_foundry_trace"
            or parent.trace_hash != str(source["parentTraceHash"])
            or parent.attempt_status != "completed"
            or any(parent.source[key] != source[key] for key in common_keys)
        ):
            raise ValueError("Foundry finalist audit trace parent identity mismatch")

    def _validate_audit_source_locked(
        self,
        conn: sqlite3.Connection,
        record: Any,
        *,
        worker_id: str,
        fencing_token: int,
        insertion: bool,
    ) -> tuple[sqlite3.Row, str, str]:
        source = record.source
        row = conn.execute(
            """SELECT s.*,c.spec_json AS campaign_spec_json
               FROM foundry_shards_v1 s
               JOIN foundry_campaigns_v1 c ON c.campaign_hash=s.campaign_hash
               WHERE s.spec_hash=?""",
            (str(source["shardId"]),),
        ).fetchone()
        if row is None:
            raise KeyError(str(source["shardId"]))
        shard_spec = json.loads(row["spec_json"])
        campaign_spec = json.loads(row["campaign_spec_json"])
        kind = _validate_campaign_audit_source(
            record,
            campaign_hash=str(row["campaign_hash"]),
            campaign_spec=campaign_spec,
            shard_spec=shard_spec,
            spec_hash=str(row["spec_hash"]),
            shard_index=int(row["shard_index"]),
            attempt=int(row["attempts"]),
            worker_id=str(worker_id),
            fencing_token=int(fencing_token),
        )
        current = time.time()
        active = (
            str(row["status"]) == "running"
            and str(row["worker_id"] or "") == str(worker_id)
            and int(row["fencing_token"]) == int(fencing_token)
            and float(row["lease_until"] or 0) >= current
        )
        attempt_row = conn.execute(
            """SELECT terminal_status FROM foundry_shard_attempts_v1
               WHERE spec_hash=? AND fencing_token=? AND attempt=? AND worker_id=?""",
            (
                str(row["spec_hash"]),
                int(fencing_token),
                int(source["attemptCount"]),
                str(worker_id),
            ),
        ).fetchone()
        completed = (
            attempt_row is not None
            and str(attempt_row["terminal_status"]) == "completed"
        )
        permitted = completed if kind == "organic_finalist_trace" else active
        if not insertion:
            permitted = active or completed
        if not permitted:
            raise RuntimeError("stale Foundry shard audit-trace admission")
        if kind == "organic_finalist_trace":
            self._validate_finalist_parent_locked(conn, record)
        return row, kind, "completed" if completed else "running"

    def put_audit_trace(
        self,
        payload: Mapping[str, Any],
        *,
        shard_id: str,
        worker_id: str,
        fencing_token: int,
    ) -> dict[str, Any]:
        requested = _audit_record(payload)
        if str(shard_id) != str(requested.source["shardId"]):
            raise ValueError("Foundry audit trace shard identity mismatch")
        if (
            str(worker_id) != str(requested.source["workerId"])
            or int(fencing_token) != int(requested.source["fencingToken"])
        ):
            raise ValueError("Foundry audit trace attempt identity mismatch")
        now = _audit_now()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                _, kind, _ = self._validate_audit_source_locked(
                    conn,
                    requested,
                    worker_id=worker_id,
                    fencing_token=fencing_token,
                    insertion=True,
                )
                value = _audit_record(
                    {
                        **requested.to_dict(),
                        "attemptStatus": (
                            "completed"
                            if kind == "organic_finalist_trace"
                            else "running"
                        ),
                        "createdAt": now,
                        "updatedAt": now,
                    }
                )
                occurrence_hash = _audit_source_occurrence_hash(value)
                rows = conn.execute(
                    """SELECT * FROM foundry_audit_traces_v1
                       WHERE record_hash=? OR source_occurrence_hash=?
                          OR audit_trace_code=?""",
                    (
                        value.record_hash,
                        occurrence_hash,
                        value.audit_trace_code,
                    ),
                ).fetchall()
                if rows:
                    for row in rows:
                        current = _audit_index_record(row)
                        if (
                            current.record_hash != value.record_hash
                            or _audit_source_occurrence_hash(current) != occurrence_hash
                            or current.audit_trace_code != value.audit_trace_code
                            or _audit_identity(current) != _audit_identity(value)
                        ):
                            raise ValueError(
                                "Foundry audit trace source occurrence is immutable"
                            )
                    conn.commit()
                    return _audit_index_record(rows[0]).to_dict()
                conn.execute(
                    """INSERT INTO foundry_audit_traces_v1(
                       record_hash,audit_trace_code,trace_hash,artifact_sha256,
                       artifact_bytes,source_occurrence_hash,campaign_hash,spec_hash,
                       attempt,fencing_token,candidate_hash,candidate_ordinal,seed,
                       disposition,status,attempt_status,record_json,admitted,
                       created_at,updated_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        value.record_hash,
                        value.audit_trace_code,
                        value.trace_hash,
                        value.complete_trace["sha256"],
                        value.complete_trace["bytes"],
                        occurrence_hash,
                        value.source["runCode"],
                        value.source["shardId"],
                        value.source["attemptCount"],
                        value.source["fencingToken"],
                        value.source["candidateHash"],
                        value.source["candidateOrdinal"],
                        value.source["seed"],
                        value.disposition,
                        value.status,
                        value.attempt_status,
                        _canonical(value.to_dict()),
                        1,
                        now,
                        now,
                    ),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return value.to_dict()

    def update_audit_trace(
        self,
        payload: Mapping[str, Any],
        *,
        worker_id: str,
        fencing_token: int,
    ) -> dict[str, Any]:
        requested = _audit_record(payload)
        if (
            str(worker_id) != str(requested.source["workerId"])
            or int(fencing_token) != int(requested.source["fencingToken"])
        ):
            raise ValueError("Foundry audit trace attempt identity mismatch")
        now = _audit_now()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                _, _, lifecycle = self._validate_audit_source_locked(
                    conn,
                    requested,
                    worker_id=worker_id,
                    fencing_token=fencing_token,
                    insertion=False,
                )
                row = conn.execute(
                    "SELECT * FROM foundry_audit_traces_v1 WHERE record_hash=?",
                    (requested.record_hash,),
                ).fetchone()
                if row is None:
                    raise KeyError(requested.record_hash)
                current = _audit_index_record(row)
                if _audit_identity(current) != _audit_identity(requested):
                    raise ValueError("Foundry audit trace identity is immutable")
                if requested.status != "finalized":
                    raise ValueError("Foundry audit trace update must finalize its lifecycle")
                value = _audit_record(
                    {
                        **requested.to_dict(),
                        "failureCategories": sorted(
                            {*current.failure_categories, *requested.failure_categories}
                        ),
                        "attemptStatus": lifecycle,
                        "createdAt": current.created_at,
                        "updatedAt": now,
                    }
                )
                if current.status == "finalized":
                    comparable_current = current.to_dict()
                    comparable_value = value.to_dict()
                    comparable_current.pop("updatedAt")
                    comparable_value.pop("updatedAt")
                    if comparable_current != comparable_value:
                        raise ValueError("finalized Foundry audit trace is immutable")
                    conn.commit()
                    return current.to_dict()
                conn.execute(
                    """UPDATE foundry_audit_traces_v1
                       SET disposition=?,status=?,attempt_status=?,record_json=?,updated_at=?
                       WHERE record_hash=?""",
                    (
                        value.disposition,
                        value.status,
                        value.attempt_status,
                        _canonical(value.to_dict()),
                        now,
                        value.record_hash,
                    ),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return value.to_dict()

    def get_audit_trace(
        self, identifier: str, *, include_unadmitted: bool = False
    ) -> dict[str, Any] | None:
        admitted = "" if include_unadmitted else " AND admitted=1"
        with self._connect() as conn:
            row = conn.execute(
                f"""SELECT * FROM foundry_audit_traces_v1
                    WHERE (record_hash=? OR trace_hash=? OR audit_trace_code=?)
                    {admitted} ORDER BY created_at DESC,record_hash LIMIT 1""",
                (str(identifier), str(identifier), str(identifier)),
            ).fetchone()
        return None if row is None else _audit_index_record(row).to_dict()

    def list_audit_traces(
        self,
        *,
        run_code: str | None = None,
        disposition: str | None = None,
        status: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        clauses = ["admitted=1"]
        parameters: list[Any] = []
        for column, value in (
            ("campaign_hash", run_code),
            ("disposition", disposition),
            ("status", status),
        ):
            if value is not None:
                clauses.append(f"{column}=?")
                parameters.append(str(value))
        where = " WHERE " + " AND ".join(clauses)
        start = max(0, int(offset))
        size = max(1, min(500, int(limit)))
        with self._connect() as conn:
            total = int(
                conn.execute(
                    f"SELECT COUNT(*) AS count FROM foundry_audit_traces_v1{where}",
                    parameters,
                ).fetchone()["count"]
            )
            rows = conn.execute(
                f"""SELECT * FROM foundry_audit_traces_v1{where}
                    ORDER BY created_at DESC,trace_hash LIMIT ? OFFSET ?""",
                [*parameters, size, start],
            ).fetchall()
        items = [_audit_index_record(row).to_dict() for row in rows]
        next_offset = start + len(items)
        return {
            "items": items,
            "total": total,
            "offset": start,
            "limit": size,
            "nextOffset": next_offset if next_offset < total else None,
        }

    def audit_trace_storage_metrics(self, campaign_hash: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                """WITH selected AS (
                     SELECT trace_hash,artifact_sha256,artifact_bytes
                     FROM foundry_audit_traces_v1
                     WHERE campaign_hash=? AND admitted=1
                   ), unique_artifacts AS (
                     SELECT artifact_sha256,MAX(artifact_bytes) AS artifact_bytes
                     FROM selected GROUP BY artifact_sha256
                   )
                   SELECT COUNT(*) AS occurrences,
                     COUNT(DISTINCT trace_hash) AS unique_traces,
                     COUNT(DISTINCT artifact_sha256) AS unique_artifacts,
                     COALESCE(SUM(artifact_bytes),0) AS logical_bytes,
                     COALESCE((SELECT SUM(artifact_bytes) FROM unique_artifacts),0)
                       AS unique_bytes
                   FROM selected""",
                (str(campaign_hash),),
            ).fetchone()
        return {
            "schemaVersion": 1,
            "campaignHash": str(campaign_hash),
            "occurrenceCount": int(row["occurrences"]),
            "uniqueTraceCount": int(row["unique_traces"]),
            "uniqueArtifactCount": int(row["unique_artifacts"]),
            "logicalBytes": int(row["logical_bytes"]),
            "uniqueBytes": int(row["unique_bytes"]),
        }

    def claim_shard(
        self,
        worker_id: str,
        *,
        lease_seconds: float,
        campaign_hash: str | None = None,
        supported_lanes: tuple[str, ...] = ("organic",),
        now: float | None = None,
    ) -> ClaimedSimulationShardV1 | None:
        worker = str(worker_id).strip()
        if not worker or float(lease_seconds) <= 0:
            raise ValueError("worker identity and positive lease are required")
        lanes = tuple(sorted(set(str(item) for item in supported_lanes)))
        if not lanes or set(lanes) - DISCOVERY_LANES:
            raise ValueError("worker must declare supported discovery lanes")
        current = time.time() if now is None else float(now)
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            query = """SELECT s.* FROM foundry_shards_v1 s
                       JOIN foundry_campaigns_v1 c ON c.campaign_hash=s.campaign_hash
                       WHERE c.cancel_requested=0 AND c.status IN ('queued','running')
                         AND NOT EXISTS (
                           SELECT 1 FROM foundry_shards_v1 failed
                           WHERE failed.campaign_hash=s.campaign_hash AND failed.status='failed'
                         )
                         AND s.attempts<?
                         AND (s.status='queued' OR (s.status='running' AND s.lease_until<?))"""
            params: list[Any] = [self.maximum_attempts, current]
            placeholders = ",".join("?" for _ in lanes)
            query += (
                " AND COALESCE(json_extract(s.spec_json,'$.lane'),'organic') "
                f"IN ({placeholders})"
            )
            params.extend(lanes)
            if campaign_hash is not None:
                query += " AND s.campaign_hash=?"
                params.append(campaign_hash)
            query += " ORDER BY s.campaign_hash,s.shard_index LIMIT 1"
            row = conn.execute(query, tuple(params)).fetchone()
            if row is None:
                return None
            token = int(row["fencing_token"]) + 1
            attempt = int(row["attempts"]) + 1
            lease_until = current + float(lease_seconds)
            if str(row["status"]) == "running":
                self._update_audit_trace_attempts_locked(
                    conn,
                    spec_hash=str(row["spec_hash"]),
                    fencing_token=int(row["fencing_token"]),
                    attempt_status="reclaimed",
                    failure_categories=(
                        "foundry_shard_attempt_reclaimed",
                        "foundry_shard_lease_expired",
                    ),
                    updated_at=_audit_now(),
                )
            conn.execute(
                """UPDATE foundry_shard_attempts_v1 SET terminal_status='lease_expired',
                   finished_at=? WHERE spec_hash=? AND terminal_status IS NULL""",
                (current, row["spec_hash"]),
            )
            conn.execute(
                """UPDATE foundry_shards_v1 SET status='running',worker_id=?,lease_until=?,
                   fencing_token=?,attempts=?,error=NULL,updated_at=CURRENT_TIMESTAMP
                   WHERE spec_hash=?""",
                (worker, lease_until, token, attempt, row["spec_hash"]),
            )
            conn.execute(
                "UPDATE foundry_campaigns_v1 SET status='running',updated_at=CURRENT_TIMESTAMP WHERE campaign_hash=?",
                (row["campaign_hash"],),
            )
            conn.execute(
                """INSERT INTO foundry_shard_attempts_v1(
                       spec_hash,campaign_hash,fencing_token,attempt,worker_id,claimed_at
                   ) VALUES(?,?,?,?,?,?)""",
                (row["spec_hash"], row["campaign_hash"], token, attempt, worker, current),
            )
        return ClaimedSimulationShardV1(
            SimulationShardSpecV1.from_dict(json.loads(row["spec_json"])),
            worker, token, attempt, lease_until,
        )

    def heartbeat(
        self,
        spec_hash: str,
        *,
        worker_id: str,
        fencing_token: int,
        lease_seconds: float,
        now: float | None = None,
    ) -> bool:
        if float(lease_seconds) <= 0:
            raise ValueError("heartbeat lease must be positive")
        current = time.time() if now is None else float(now)
        with self._connect() as conn:
            result = conn.execute(
                """UPDATE foundry_shards_v1 SET lease_until=?,updated_at=CURRENT_TIMESTAMP
                   WHERE spec_hash=? AND status='running' AND worker_id=? AND fencing_token=?
                     AND lease_until>=?""",
                (current + float(lease_seconds), spec_hash, worker_id, int(fencing_token), current),
            )
        return result.rowcount == 1

    def commit_result(
        self,
        result: SimulationShardResultV1,
        *,
        worker_id: str,
        fencing_token: int,
        now: float | None = None,
    ) -> dict[str, Any]:
        if result.completion_status != "completed":
            raise ValueError("only complete shard results may enter campaign evidence")
        current = time.time() if now is None else float(now)
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM foundry_shards_v1 WHERE spec_hash=?", (result.spec_hash,)
            ).fetchone()
            if row is None:
                raise KeyError(result.spec_hash)
            if (
                row["status"] != "running"
                or row["worker_id"] != worker_id
                or int(row["fencing_token"]) != int(fencing_token)
                or float(row["lease_until"] or 0) < current
            ):
                raise PermissionError("stale or expired shard worker cannot commit")
            spec = SimulationShardSpecV1.from_dict(json.loads(row["spec_json"]))
            if spec.input_hash != result.spec_hash:
                raise ValueError("shard result does not match the claimed immutable input")
            conn.execute(
                """UPDATE foundry_shards_v1 SET status='completed',result_json=?,worker_id=NULL,
                   lease_until=NULL,updated_at=CURRENT_TIMESTAMP WHERE spec_hash=?""",
                (_canonical(result.to_dict()), result.spec_hash),
            )
            attempt_update = conn.execute(
                """UPDATE foundry_shard_attempts_v1 SET terminal_status='completed',
                   finished_at=? WHERE spec_hash=? AND fencing_token=? AND worker_id=?
                     AND terminal_status IS NULL""",
                (current, result.spec_hash, int(fencing_token), worker_id),
            )
            if attempt_update.rowcount != 1:
                raise RuntimeError("campaign attempt ledger rejected a valid completion")
            self._update_audit_trace_attempts_locked(
                conn,
                spec_hash=result.spec_hash,
                fencing_token=int(fencing_token),
                attempt_status="completed",
                updated_at=_audit_now(),
            )
            remaining = int(conn.execute(
                "SELECT COUNT(*) FROM foundry_shards_v1 WHERE campaign_hash=? AND status!='completed'",
                (row["campaign_hash"],),
            ).fetchone()[0])
            if remaining == 0:
                conn.execute(
                    "UPDATE foundry_campaigns_v1 SET status='completed',updated_at=CURRENT_TIMESTAMP WHERE campaign_hash=?",
                    (row["campaign_hash"],),
                )
        return self.list_shards(str(row["campaign_hash"]))[int(row["shard_index"])]

    def fail_shard(
        self,
        spec_hash: str,
        *,
        worker_id: str,
        fencing_token: int,
        error: str,
        retryable: bool,
    ) -> None:
        current = time.time()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM foundry_shards_v1 WHERE spec_hash=?", (spec_hash,)
            ).fetchone()
            if row is None:
                raise KeyError(spec_hash)
            if row["status"] != "running" or row["worker_id"] != worker_id or int(row["fencing_token"]) != int(fencing_token):
                raise PermissionError("stale shard worker cannot report failure")
            retry = bool(retryable) and int(row["attempts"]) < self.maximum_attempts
            status = "queued" if retry else "failed"
            conn.execute(
                """UPDATE foundry_shards_v1 SET status=?,worker_id=NULL,lease_until=NULL,
                   error=?,updated_at=CURRENT_TIMESTAMP WHERE spec_hash=?""",
                (status, str(error)[:2000], spec_hash),
            )
            attempt_update = conn.execute(
                """UPDATE foundry_shard_attempts_v1 SET terminal_status=?,finished_at=?,error=?
                   WHERE spec_hash=? AND fencing_token=? AND worker_id=?
                     AND terminal_status IS NULL""",
                (
                    "retryable_failure" if retry else "failed",
                    current,
                    str(error)[:2000],
                    spec_hash,
                    int(fencing_token),
                    worker_id,
                ),
            )
            if attempt_update.rowcount != 1:
                raise RuntimeError("campaign attempt ledger rejected a valid failure")
            self._update_audit_trace_attempts_locked(
                conn,
                spec_hash=spec_hash,
                fencing_token=int(fencing_token),
                attempt_status="retry_scheduled" if retry else "failed",
                failure_categories=(
                    (
                        "foundry_shard_attempt_failed",
                        "foundry_shard_retry_scheduled",
                    )
                    if retry
                    else (
                        "foundry_shard_attempt_failed",
                        "foundry_shard_terminal_failure",
                    )
                ),
                updated_at=_audit_now(),
            )
            if not retry:
                conn.execute(
                    "UPDATE foundry_campaigns_v1 SET status='failed',updated_at=CURRENT_TIMESTAMP WHERE campaign_hash=?",
                    (row["campaign_hash"],),
                )

    def request_cancel(self, campaign_hash: str) -> bool:
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            changed = conn.execute(
                """UPDATE foundry_campaigns_v1 SET cancel_requested=1,status='cancelled',
                   updated_at=CURRENT_TIMESTAMP WHERE campaign_hash=? AND status NOT IN ('completed','failed','cancelled')""",
                (campaign_hash,),
            )
            if changed.rowcount:
                attempts = conn.execute(
                    """SELECT spec_hash,fencing_token FROM foundry_shards_v1
                       WHERE campaign_hash=? AND status IN ('queued','running')""",
                    (campaign_hash,),
                ).fetchall()
                conn.execute(
                    """UPDATE foundry_shards_v1 SET status='cancelled',worker_id=NULL,lease_until=NULL,
                       updated_at=CURRENT_TIMESTAMP WHERE campaign_hash=? AND status IN ('queued','running')""",
                    (campaign_hash,),
                )
                conn.execute(
                    """UPDATE foundry_shard_attempts_v1 SET terminal_status='cancelled',
                       finished_at=? WHERE campaign_hash=? AND terminal_status IS NULL""",
                    (time.time(), campaign_hash),
                )
                for attempt in attempts:
                    self._update_audit_trace_attempts_locked(
                        conn,
                        spec_hash=str(attempt["spec_hash"]),
                        fencing_token=int(attempt["fencing_token"]),
                        attempt_status="cancelled",
                        failure_categories=("foundry_shard_attempt_cancelled",),
                        updated_at=_audit_now(),
                    )
        return changed.rowcount == 1

    def list_attempts(self, campaign_hash: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT a.*,s.shard_index FROM foundry_shard_attempts_v1 a
                   JOIN foundry_shards_v1 s ON s.spec_hash=a.spec_hash
                   WHERE a.campaign_hash=?
                   ORDER BY a.spec_hash,a.fencing_token""",
                (campaign_hash,),
            ).fetchall()
        return [{
            "specHash": str(row["spec_hash"]),
            "campaignHash": str(row["campaign_hash"]),
            "shardIndex": int(row["shard_index"]),
            "fencingToken": int(row["fencing_token"]),
            "attempt": int(row["attempt"]),
            "workerId": str(row["worker_id"]),
            "claimedAt": float(row["claimed_at"]),
            "finishedAt": float(row["finished_at"]) if row["finished_at"] is not None else None,
            "terminalStatus": row["terminal_status"],
            "error": row["error"],
        } for row in rows]


class PostgresCampaignControlStore:
    """Multi-node campaign control plane using PostgreSQL row locks and fencing."""

    def __init__(self, dsn: str, *, maximum_attempts: int):
        if int(maximum_attempts) < 1:
            raise ValueError("campaign maximum attempts must be positive")
        try:
            from psycopg_pool import ConnectionPool
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError('Install with: pip install "baloot-platform[postgres]"') from exc
        self.maximum_attempts = int(maximum_attempts)
        self.pool = ConnectionPool(dsn, min_size=1, max_size=16, open=True)

    def close(self) -> None:
        self.pool.close()

    def _one(self, query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        from psycopg.rows import dict_row
        with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(query, params)
            return cursor.fetchone()

    def _all(self, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        from psycopg.rows import dict_row
        with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(query, params)
            return list(cursor.fetchall())

    @staticmethod
    def _json(value: Any) -> Any:
        return json.loads(value) if isinstance(value, str) else value

    def create_campaign(self, spec: SimulationCampaignSpecV1) -> dict[str, Any]:
        from psycopg.rows import dict_row
        with self.pool.connection() as conn, conn.transaction(), conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                "SELECT spec_json FROM foundry_campaigns_v1 WHERE campaign_hash=%s FOR UPDATE",
                (spec.sha256,),
            )
            existing = cursor.fetchone()
            if existing is not None:
                if self._json(existing["spec_json"]) != spec.to_dict():
                    raise RuntimeError("immutable campaign hash collision")
            else:
                cursor.execute(
                    """INSERT INTO foundry_campaigns_v1(campaign_hash,spec_json,status)
                       VALUES(%s,%s::jsonb,'queued')""",
                    (spec.sha256, _canonical(spec.to_dict())),
                )
                for shard in spec.shards():
                    cursor.execute(
                        """INSERT INTO foundry_shards_v1(
                               spec_hash,campaign_hash,shard_index,spec_json,status
                           ) VALUES(%s,%s,%s,%s::jsonb,'queued')""",
                        (shard.input_hash, spec.sha256, shard.shard_index, _canonical(shard.to_dict())),
                    )
        return self.get_campaign(spec.sha256) or {}

    def get_campaign(self, campaign_hash: str) -> dict[str, Any] | None:
        row = self._one(
            "SELECT * FROM foundry_campaigns_v1 WHERE campaign_hash=%s", (campaign_hash,)
        )
        if row is None:
            return None
        counts = self._all(
            """SELECT status,COUNT(*) AS count FROM foundry_shards_v1
               WHERE campaign_hash=%s GROUP BY status""",
            (campaign_hash,),
        )
        shard_counts = {str(item["status"]): int(item["count"]) for item in counts}
        cancel_requested = bool(row["cancel_requested"])
        return {
            "campaignHash": row["campaign_hash"],
            "spec": self._json(row["spec_json"]),
            "status": _effective_campaign_status(
                str(row["status"]),
                cancel_requested=cancel_requested,
                shard_counts=shard_counts,
            ),
            "cancelRequested": cancel_requested,
            "shardCounts": shard_counts,
            "createdAt": str(row["created_at"]),
            "updatedAt": str(row["updated_at"]),
        }

    def list_campaigns(
        self, *, after: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        size = max(1, min(500, int(limit)))
        rows = self._all(
            """SELECT campaign_hash FROM foundry_campaigns_v1
               WHERE (%s IS NULL OR campaign_hash>%s)
               ORDER BY campaign_hash LIMIT %s""",
            (after, after, size),
        )
        return [
            value for row in rows
            if (value := self.get_campaign(str(row["campaign_hash"]))) is not None
        ]

    def list_shards(self, campaign_hash: str) -> list[dict[str, Any]]:
        rows = self._all(
            "SELECT * FROM foundry_shards_v1 WHERE campaign_hash=%s ORDER BY shard_index",
            (campaign_hash,),
        )
        return [
            {
                "specHash": row["spec_hash"], "campaignHash": row["campaign_hash"],
                "shardIndex": int(row["shard_index"]), "spec": self._json(row["spec_json"]),
                "status": row["status"], "workerId": row["worker_id"],
                "leaseUntil": row["lease_until"], "fencingToken": int(row["fencing_token"]),
                "attempts": int(row["attempts"]),
                "result": self._json(row["result_json"]) if row["result_json"] else None,
                "error": row["error"],
            }
            for row in rows
        ]

    def get_shard(self, spec_hash: str) -> dict[str, Any] | None:
        row = self._one(
            "SELECT campaign_hash FROM foundry_shards_v1 WHERE spec_hash=%s",
            (spec_hash,),
        )
        if row is None:
            return None
        return next(
            item for item in self.list_shards(str(row["campaign_hash"]))
            if item["specHash"] == spec_hash
        )

    @staticmethod
    def _update_audit_trace_attempts_locked(
        cursor: Any,
        *,
        spec_hash: str,
        fencing_token: int,
        attempt_status: str,
        failure_categories: Iterable[str] = (),
        updated_at: str,
    ) -> None:
        cursor.execute(
            """SELECT * FROM foundry_audit_traces_v1
               WHERE spec_hash=%s AND fencing_token=%s FOR UPDATE""",
            (str(spec_hash), int(fencing_token)),
        )
        for row in cursor.fetchall():
            current = _audit_index_record(row)
            value = _audit_with_attempt_lifecycle(
                current,
                attempt_status=attempt_status,
                failure_categories=failure_categories,
                updated_at=updated_at,
            )
            cursor.execute(
                """UPDATE foundry_audit_traces_v1
                   SET attempt_status=%s,disposition=%s,status=%s,
                       record_json=%s::jsonb,updated_at=%s::timestamptz
                   WHERE record_hash=%s AND spec_hash=%s AND fencing_token=%s""",
                (
                    value.attempt_status,
                    value.disposition,
                    value.status,
                    _canonical(value.to_dict()),
                    updated_at,
                    value.record_hash,
                    str(spec_hash),
                    int(fencing_token),
                ),
            )

    @staticmethod
    def _validate_finalist_parent_locked(cursor: Any, record: Any) -> None:
        source = record.source
        cursor.execute(
            """SELECT * FROM foundry_audit_traces_v1
               WHERE record_hash=%s FOR UPDATE""",
            (str(source["parentRecordHash"]),),
        )
        row = cursor.fetchone()
        if row is None:
            raise ValueError("Foundry finalist audit trace parent is unavailable")
        parent = _audit_index_record(row)
        common_keys = (
            "runCode",
            "runHash",
            "shardId",
            "shardHash",
            "shardIndex",
            "attemptCount",
            "fencingToken",
            "workerId",
        )
        if (
            str(parent.source["kind"]) != "organic_foundry_trace"
            or parent.trace_hash != str(source["parentTraceHash"])
            or parent.attempt_status != "completed"
            or any(parent.source[key] != source[key] for key in common_keys)
        ):
            raise ValueError("Foundry finalist audit trace parent identity mismatch")

    def _validate_audit_source_locked(
        self,
        cursor: Any,
        record: Any,
        *,
        worker_id: str,
        fencing_token: int,
        insertion: bool,
    ) -> tuple[dict[str, Any], str, str]:
        source = record.source
        cursor.execute(
            """SELECT s.*,c.spec_json AS campaign_spec_json
               FROM foundry_shards_v1 s
               JOIN foundry_campaigns_v1 c ON c.campaign_hash=s.campaign_hash
               WHERE s.spec_hash=%s FOR UPDATE OF s""",
            (str(source["shardId"]),),
        )
        row = cursor.fetchone()
        if row is None:
            raise KeyError(str(source["shardId"]))
        shard_spec = self._json(row["spec_json"])
        campaign_spec = self._json(row["campaign_spec_json"])
        kind = _validate_campaign_audit_source(
            record,
            campaign_hash=str(row["campaign_hash"]),
            campaign_spec=campaign_spec,
            shard_spec=shard_spec,
            spec_hash=str(row["spec_hash"]),
            shard_index=int(row["shard_index"]),
            attempt=int(row["attempts"]),
            worker_id=str(worker_id),
            fencing_token=int(fencing_token),
        )
        active = (
            str(row["status"]) == "running"
            and str(row["worker_id"] or "") == str(worker_id)
            and int(row["fencing_token"]) == int(fencing_token)
            and float(row["lease_until"] or 0) >= time.time()
        )
        cursor.execute(
            """SELECT terminal_status FROM foundry_shard_attempts_v1
               WHERE spec_hash=%s AND fencing_token=%s AND attempt=%s
                 AND worker_id=%s FOR UPDATE""",
            (
                str(row["spec_hash"]),
                int(fencing_token),
                int(source["attemptCount"]),
                str(worker_id),
            ),
        )
        attempt_row = cursor.fetchone()
        completed = (
            attempt_row is not None
            and str(attempt_row["terminal_status"]) == "completed"
        )
        permitted = completed if kind == "organic_finalist_trace" else active
        if not insertion:
            permitted = active or completed
        if not permitted:
            raise RuntimeError("stale Foundry shard audit-trace admission")
        if kind == "organic_finalist_trace":
            self._validate_finalist_parent_locked(cursor, record)
        return row, kind, "completed" if completed else "running"

    def put_audit_trace(
        self,
        payload: Mapping[str, Any],
        *,
        shard_id: str,
        worker_id: str,
        fencing_token: int,
    ) -> dict[str, Any]:
        requested = _audit_record(payload)
        if str(shard_id) != str(requested.source["shardId"]):
            raise ValueError("Foundry audit trace shard identity mismatch")
        if (
            str(worker_id) != str(requested.source["workerId"])
            or int(fencing_token) != int(requested.source["fencingToken"])
        ):
            raise ValueError("Foundry audit trace attempt identity mismatch")
        occurrence_hash = _audit_source_occurrence_hash(requested)
        now = _audit_now()
        from psycopg.rows import dict_row

        with (
            self.pool.connection() as conn,
            conn.transaction(),
            conn.cursor(row_factory=dict_row) as cursor,
        ):
            cursor.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
                (f"foundry:audit-trace:{occurrence_hash}",),
            )
            _, kind, _ = self._validate_audit_source_locked(
                cursor,
                requested,
                worker_id=worker_id,
                fencing_token=fencing_token,
                insertion=True,
            )
            value = _audit_record(
                {
                    **requested.to_dict(),
                    "attemptStatus": (
                        "completed" if kind == "organic_finalist_trace" else "running"
                    ),
                    "createdAt": now,
                    "updatedAt": now,
                }
            )
            cursor.execute(
                """INSERT INTO foundry_audit_traces_v1(
                   record_hash,audit_trace_code,trace_hash,artifact_sha256,
                   artifact_bytes,source_occurrence_hash,campaign_hash,spec_hash,
                   attempt,fencing_token,candidate_hash,candidate_ordinal,seed,
                   disposition,status,attempt_status,record_json,admitted,
                   created_at,updated_at)
                   VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                          %s::jsonb,true,%s::timestamptz,%s::timestamptz)
                   ON CONFLICT DO NOTHING RETURNING *""",
                (
                    value.record_hash,
                    value.audit_trace_code,
                    value.trace_hash,
                    value.complete_trace["sha256"],
                    value.complete_trace["bytes"],
                    occurrence_hash,
                    value.source["runCode"],
                    value.source["shardId"],
                    value.source["attemptCount"],
                    value.source["fencingToken"],
                    value.source["candidateHash"],
                    value.source["candidateOrdinal"],
                    value.source["seed"],
                    value.disposition,
                    value.status,
                    value.attempt_status,
                    _canonical(value.to_dict()),
                    now,
                    now,
                ),
            )
            row = cursor.fetchone()
            if row is None:
                cursor.execute(
                    """SELECT * FROM foundry_audit_traces_v1
                       WHERE record_hash=%s OR source_occurrence_hash=%s
                          OR audit_trace_code=%s FOR UPDATE""",
                    (value.record_hash, occurrence_hash, value.audit_trace_code),
                )
                rows = cursor.fetchall()
            else:
                rows = [row]
            if not rows:
                raise RuntimeError("Foundry audit trace index write failed")
            for indexed in rows:
                current = _audit_index_record(indexed)
                if (
                    current.record_hash != value.record_hash
                    or _audit_source_occurrence_hash(current) != occurrence_hash
                    or current.audit_trace_code != value.audit_trace_code
                    or _audit_identity(current) != _audit_identity(value)
                ):
                    raise ValueError("Foundry audit trace source occurrence is immutable")
            return _audit_index_record(rows[0]).to_dict()

    def update_audit_trace(
        self,
        payload: Mapping[str, Any],
        *,
        worker_id: str,
        fencing_token: int,
    ) -> dict[str, Any]:
        requested = _audit_record(payload)
        if (
            str(worker_id) != str(requested.source["workerId"])
            or int(fencing_token) != int(requested.source["fencingToken"])
        ):
            raise ValueError("Foundry audit trace attempt identity mismatch")
        occurrence_hash = _audit_source_occurrence_hash(requested)
        now = _audit_now()
        from psycopg.rows import dict_row

        with (
            self.pool.connection() as conn,
            conn.transaction(),
            conn.cursor(row_factory=dict_row) as cursor,
        ):
            cursor.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
                (f"foundry:audit-trace:{occurrence_hash}",),
            )
            _, _, lifecycle = self._validate_audit_source_locked(
                cursor,
                requested,
                worker_id=worker_id,
                fencing_token=fencing_token,
                insertion=False,
            )
            cursor.execute(
                """SELECT * FROM foundry_audit_traces_v1
                   WHERE record_hash=%s FOR UPDATE""",
                (requested.record_hash,),
            )
            row = cursor.fetchone()
            if row is None:
                raise KeyError(requested.record_hash)
            current = _audit_index_record(row)
            if _audit_identity(current) != _audit_identity(requested):
                raise ValueError("Foundry audit trace identity is immutable")
            if requested.status != "finalized":
                raise ValueError("Foundry audit trace update must finalize its lifecycle")
            value = _audit_record(
                {
                    **requested.to_dict(),
                    "failureCategories": sorted(
                        {*current.failure_categories, *requested.failure_categories}
                    ),
                    "attemptStatus": lifecycle,
                    "createdAt": current.created_at,
                    "updatedAt": now,
                }
            )
            if current.status == "finalized":
                comparable_current = current.to_dict()
                comparable_value = value.to_dict()
                comparable_current.pop("updatedAt")
                comparable_value.pop("updatedAt")
                if comparable_current != comparable_value:
                    raise ValueError("finalized Foundry audit trace is immutable")
                return current.to_dict()
            cursor.execute(
                """UPDATE foundry_audit_traces_v1
                   SET disposition=%s,status=%s,attempt_status=%s,
                       record_json=%s::jsonb,updated_at=%s::timestamptz
                   WHERE record_hash=%s""",
                (
                    value.disposition,
                    value.status,
                    value.attempt_status,
                    _canonical(value.to_dict()),
                    now,
                    value.record_hash,
                ),
            )
            return value.to_dict()

    def get_audit_trace(
        self, identifier: str, *, include_unadmitted: bool = False
    ) -> dict[str, Any] | None:
        admitted = "" if include_unadmitted else " AND admitted=true"
        row = self._one(
            f"""SELECT * FROM foundry_audit_traces_v1
                WHERE (record_hash=%s OR trace_hash=%s OR audit_trace_code=%s)
                {admitted} ORDER BY created_at DESC,record_hash LIMIT 1""",
            (str(identifier), str(identifier), str(identifier)),
        )
        return None if row is None else _audit_index_record(row).to_dict()

    def list_audit_traces(
        self,
        *,
        run_code: str | None = None,
        disposition: str | None = None,
        status: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        clauses = ["admitted=true"]
        parameters: list[Any] = []
        for column, value in (
            ("campaign_hash", run_code),
            ("disposition", disposition),
            ("status", status),
        ):
            if value is not None:
                clauses.append(f"{column}=%s")
                parameters.append(str(value))
        where = " WHERE " + " AND ".join(clauses)
        start = max(0, int(offset))
        size = max(1, min(500, int(limit)))
        count = self._one(
            f"SELECT COUNT(*) AS count FROM foundry_audit_traces_v1{where}",
            tuple(parameters),
        )
        rows = self._all(
            f"""SELECT * FROM foundry_audit_traces_v1{where}
                ORDER BY created_at DESC,trace_hash LIMIT %s OFFSET %s""",
            (*parameters, size, start),
        )
        total = int(count["count"] if count is not None else 0)
        items = [_audit_index_record(row).to_dict() for row in rows]
        next_offset = start + len(items)
        return {
            "items": items,
            "total": total,
            "offset": start,
            "limit": size,
            "nextOffset": next_offset if next_offset < total else None,
        }

    def audit_trace_storage_metrics(self, campaign_hash: str) -> dict[str, Any]:
        row = self._one(
            """WITH selected AS (
                 SELECT trace_hash,artifact_sha256,artifact_bytes
                 FROM foundry_audit_traces_v1
                 WHERE campaign_hash=%s AND admitted=true
               ), unique_artifacts AS (
                 SELECT artifact_sha256,MAX(artifact_bytes) AS artifact_bytes
                 FROM selected GROUP BY artifact_sha256
               )
               SELECT COUNT(*) AS occurrences,
                 COUNT(DISTINCT trace_hash) AS unique_traces,
                 COUNT(DISTINCT artifact_sha256) AS unique_artifacts,
                 COALESCE(SUM(artifact_bytes),0) AS logical_bytes,
                 COALESCE((SELECT SUM(artifact_bytes) FROM unique_artifacts),0)
                   AS unique_bytes
               FROM selected""",
            (str(campaign_hash),),
        )
        return {
            "schemaVersion": 1,
            "campaignHash": str(campaign_hash),
            "occurrenceCount": int(row["occurrences"] if row is not None else 0),
            "uniqueTraceCount": int(row["unique_traces"] if row is not None else 0),
            "uniqueArtifactCount": int(row["unique_artifacts"] if row is not None else 0),
            "logicalBytes": int(row["logical_bytes"] if row is not None else 0),
            "uniqueBytes": int(row["unique_bytes"] if row is not None else 0),
        }

    def claim_shard(
        self,
        worker_id: str,
        *,
        lease_seconds: float,
        campaign_hash: str | None = None,
        supported_lanes: tuple[str, ...] = ("organic",),
        now: float | None = None,
    ) -> ClaimedSimulationShardV1 | None:
        worker = str(worker_id).strip()
        if not worker or float(lease_seconds) <= 0:
            raise ValueError("worker identity and positive lease are required")
        lanes = tuple(sorted(set(str(item) for item in supported_lanes)))
        if not lanes or set(lanes) - DISCOVERY_LANES:
            raise ValueError("worker must declare supported discovery lanes")
        current = time.time() if now is None else float(now)
        from psycopg.rows import dict_row
        with self.pool.connection() as conn, conn.transaction(), conn.cursor(row_factory=dict_row) as cursor:
            query = """SELECT s.* FROM foundry_shards_v1 s
                       JOIN foundry_campaigns_v1 c ON c.campaign_hash=s.campaign_hash
                       WHERE c.cancel_requested=FALSE AND c.status IN ('queued','running')
                         AND NOT EXISTS (
                           SELECT 1 FROM foundry_shards_v1 failed
                           WHERE failed.campaign_hash=s.campaign_hash AND failed.status='failed'
                         )
                         AND s.attempts<%s
                         AND (s.status='queued' OR (s.status='running' AND s.lease_until<%s))"""
            params: list[Any] = [self.maximum_attempts, current]
            query += " AND COALESCE(s.spec_json->>'lane','organic')=ANY(%s)"
            params.append(list(lanes))
            if campaign_hash is not None:
                query += " AND s.campaign_hash=%s"
                params.append(campaign_hash)
            query += " ORDER BY s.campaign_hash,s.shard_index FOR UPDATE OF s SKIP LOCKED LIMIT 1"
            cursor.execute(query, tuple(params))
            row = cursor.fetchone()
            if row is None:
                return None
            token = int(row["fencing_token"]) + 1
            attempt = int(row["attempts"]) + 1
            lease_until = current + float(lease_seconds)
            if str(row["status"]) == "running":
                self._update_audit_trace_attempts_locked(
                    cursor,
                    spec_hash=str(row["spec_hash"]),
                    fencing_token=int(row["fencing_token"]),
                    attempt_status="reclaimed",
                    failure_categories=(
                        "foundry_shard_attempt_reclaimed",
                        "foundry_shard_lease_expired",
                    ),
                    updated_at=_audit_now(),
                )
            cursor.execute(
                """UPDATE foundry_shard_attempts_v1 SET terminal_status='lease_expired',
                   finished_at=%s WHERE spec_hash=%s AND terminal_status IS NULL""",
                (current, row["spec_hash"]),
            )
            cursor.execute(
                """UPDATE foundry_shards_v1 SET status='running',worker_id=%s,lease_until=%s,
                   fencing_token=%s,attempts=%s,error=NULL,updated_at=now() WHERE spec_hash=%s""",
                (worker, lease_until, token, attempt, row["spec_hash"]),
            )
            cursor.execute(
                "UPDATE foundry_campaigns_v1 SET status='running',updated_at=now() WHERE campaign_hash=%s",
                (row["campaign_hash"],),
            )
            cursor.execute(
                """INSERT INTO foundry_shard_attempts_v1(
                       spec_hash,campaign_hash,fencing_token,attempt,worker_id,claimed_at
                   ) VALUES(%s,%s,%s,%s,%s,%s)""",
                (row["spec_hash"], row["campaign_hash"], token, attempt, worker, current),
            )
        return ClaimedSimulationShardV1(
            SimulationShardSpecV1.from_dict(self._json(row["spec_json"])),
            worker, token, attempt, lease_until,
        )

    def heartbeat(
        self,
        spec_hash: str,
        *,
        worker_id: str,
        fencing_token: int,
        lease_seconds: float,
        now: float | None = None,
    ) -> bool:
        if float(lease_seconds) <= 0:
            raise ValueError("heartbeat lease must be positive")
        current = time.time() if now is None else float(now)
        with self.pool.connection() as conn:
            result = conn.execute(
                """UPDATE foundry_shards_v1 SET lease_until=%s,updated_at=now()
                   WHERE spec_hash=%s AND status='running' AND worker_id=%s
                     AND fencing_token=%s AND lease_until>=%s""",
                (current + float(lease_seconds), spec_hash, worker_id, int(fencing_token), current),
            )
        return result.rowcount == 1

    def commit_result(
        self,
        result: SimulationShardResultV1,
        *,
        worker_id: str,
        fencing_token: int,
        now: float | None = None,
    ) -> dict[str, Any]:
        if result.completion_status != "completed":
            raise ValueError("only complete shard results may enter campaign evidence")
        current = time.time() if now is None else float(now)
        from psycopg.rows import dict_row
        with self.pool.connection() as conn, conn.transaction(), conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                "SELECT * FROM foundry_shards_v1 WHERE spec_hash=%s FOR UPDATE",
                (result.spec_hash,),
            )
            row = cursor.fetchone()
            if row is None:
                raise KeyError(result.spec_hash)
            if (
                row["status"] != "running" or row["worker_id"] != worker_id
                or int(row["fencing_token"]) != int(fencing_token)
                or float(row["lease_until"] or 0) < current
            ):
                raise PermissionError("stale or expired shard worker cannot commit")
            spec = SimulationShardSpecV1.from_dict(self._json(row["spec_json"]))
            if spec.input_hash != result.spec_hash:
                raise ValueError("shard result does not match the claimed immutable input")
            cursor.execute(
                """UPDATE foundry_shards_v1 SET status='completed',result_json=%s::jsonb,
                   worker_id=NULL,lease_until=NULL,updated_at=now() WHERE spec_hash=%s""",
                (_canonical(result.to_dict()), result.spec_hash),
            )
            cursor.execute(
                """UPDATE foundry_shard_attempts_v1 SET terminal_status='completed',
                   finished_at=%s WHERE spec_hash=%s AND fencing_token=%s AND worker_id=%s
                     AND terminal_status IS NULL""",
                (current, result.spec_hash, int(fencing_token), worker_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("campaign attempt ledger rejected a valid completion")
            self._update_audit_trace_attempts_locked(
                cursor,
                spec_hash=result.spec_hash,
                fencing_token=int(fencing_token),
                attempt_status="completed",
                updated_at=_audit_now(),
            )
            cursor.execute(
                "SELECT COUNT(*) AS count FROM foundry_shards_v1 WHERE campaign_hash=%s AND status!='completed'",
                (row["campaign_hash"],),
            )
            if int(cursor.fetchone()["count"]) == 0:
                cursor.execute(
                    "UPDATE foundry_campaigns_v1 SET status='completed',updated_at=now() WHERE campaign_hash=%s",
                    (row["campaign_hash"],),
                )
        return self.list_shards(str(row["campaign_hash"]))[int(row["shard_index"])]

    def fail_shard(
        self,
        spec_hash: str,
        *,
        worker_id: str,
        fencing_token: int,
        error: str,
        retryable: bool,
    ) -> None:
        current = time.time()
        from psycopg.rows import dict_row
        with self.pool.connection() as conn, conn.transaction(), conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute("SELECT * FROM foundry_shards_v1 WHERE spec_hash=%s FOR UPDATE", (spec_hash,))
            row = cursor.fetchone()
            if row is None:
                raise KeyError(spec_hash)
            if row["status"] != "running" or row["worker_id"] != worker_id or int(row["fencing_token"]) != int(fencing_token):
                raise PermissionError("stale shard worker cannot report failure")
            retry = bool(retryable) and int(row["attempts"]) < self.maximum_attempts
            cursor.execute(
                """UPDATE foundry_shards_v1 SET status=%s,worker_id=NULL,lease_until=NULL,
                   error=%s,updated_at=now() WHERE spec_hash=%s""",
                ("queued" if retry else "failed", str(error)[:2000], spec_hash),
            )
            cursor.execute(
                """UPDATE foundry_shard_attempts_v1 SET terminal_status=%s,
                   finished_at=%s,error=%s WHERE spec_hash=%s AND fencing_token=%s
                     AND worker_id=%s AND terminal_status IS NULL""",
                (
                    "retryable_failure" if retry else "failed",
                    current,
                    str(error)[:2000],
                    spec_hash,
                    int(fencing_token),
                    worker_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("campaign attempt ledger rejected a valid failure")
            self._update_audit_trace_attempts_locked(
                cursor,
                spec_hash=spec_hash,
                fencing_token=int(fencing_token),
                attempt_status="retry_scheduled" if retry else "failed",
                failure_categories=(
                    (
                        "foundry_shard_attempt_failed",
                        "foundry_shard_retry_scheduled",
                    )
                    if retry
                    else (
                        "foundry_shard_attempt_failed",
                        "foundry_shard_terminal_failure",
                    )
                ),
                updated_at=_audit_now(),
            )
            if not retry:
                cursor.execute(
                    "UPDATE foundry_campaigns_v1 SET status='failed',updated_at=now() WHERE campaign_hash=%s",
                    (row["campaign_hash"],),
                )

    def request_cancel(self, campaign_hash: str) -> bool:
        from psycopg.rows import dict_row

        with (
            self.pool.connection() as conn,
            conn.transaction(),
            conn.cursor(row_factory=dict_row) as cursor,
        ):
            cursor.execute(
                """UPDATE foundry_campaigns_v1 SET cancel_requested=TRUE,status='cancelled',updated_at=now()
                   WHERE campaign_hash=%s AND status NOT IN ('completed','failed','cancelled')""",
                (campaign_hash,),
            )
            changed = cursor.rowcount
            if changed:
                cursor.execute(
                    """SELECT spec_hash,fencing_token FROM foundry_shards_v1
                       WHERE campaign_hash=%s AND status IN ('queued','running')
                       FOR UPDATE""",
                    (campaign_hash,),
                )
                attempts = cursor.fetchall()
                cursor.execute(
                    """UPDATE foundry_shards_v1 SET status='cancelled',worker_id=NULL,lease_until=NULL,
                       updated_at=now() WHERE campaign_hash=%s AND status IN ('queued','running')""",
                    (campaign_hash,),
                )
                cursor.execute(
                    """UPDATE foundry_shard_attempts_v1 SET terminal_status='cancelled',
                       finished_at=%s WHERE campaign_hash=%s AND terminal_status IS NULL""",
                    (time.time(), campaign_hash),
                )
                for attempt in attempts:
                    self._update_audit_trace_attempts_locked(
                        cursor,
                        spec_hash=str(attempt["spec_hash"]),
                        fencing_token=int(attempt["fencing_token"]),
                        attempt_status="cancelled",
                        failure_categories=("foundry_shard_attempt_cancelled",),
                        updated_at=_audit_now(),
                    )
        return changed == 1

    def list_attempts(self, campaign_hash: str) -> list[dict[str, Any]]:
        rows = self._all(
            """SELECT a.*,s.shard_index FROM foundry_shard_attempts_v1 a
               JOIN foundry_shards_v1 s ON s.spec_hash=a.spec_hash
               WHERE a.campaign_hash=%s ORDER BY a.spec_hash,a.fencing_token""",
            (campaign_hash,),
        )
        return [{
            "specHash": str(row["spec_hash"]),
            "campaignHash": str(row["campaign_hash"]),
            "shardIndex": int(row["shard_index"]),
            "fencingToken": int(row["fencing_token"]),
            "attempt": int(row["attempt"]),
            "workerId": str(row["worker_id"]),
            "claimedAt": float(row["claimed_at"]),
            "finishedAt": float(row["finished_at"]) if row["finished_at"] is not None else None,
            "terminalStatus": row["terminal_status"],
            "error": row["error"],
        } for row in rows]
