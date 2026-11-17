from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from baloot_platform.security.encryption import PayloadCipher

from baloot_platform.core.models import Event

from .event_store_v7 import EventEnvelope, SQLiteEventStore


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class SnapshotRecord:
    snapshot_id: str
    game_id: str
    event_count: int
    head_hash: str | None
    state_hash: str
    state_payload: Mapping[str, Any]
    schema_version: int
    created_at: str
    engine_version: str
    profile_bundle_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshotId": self.snapshot_id, "gameId": self.game_id,
            "eventCount": self.event_count, "headHash": self.head_hash,
            "stateHash": self.state_hash, "state": dict(self.state_payload),
            "schemaVersion": self.schema_version, "createdAt": self.created_at,
            "engineVersion": self.engine_version, "profileBundleHash": self.profile_bundle_hash,
        }


@dataclass(frozen=True, slots=True)
class BranchRecord:
    branch_id: str
    parent_game_id: str
    parent_event_count: int
    parent_head_hash: str | None
    child_game_id: str
    created_by: str
    reason: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "branchId": self.branch_id, "parentGameId": self.parent_game_id,
            "parentEventCount": self.parent_event_count, "parentHeadHash": self.parent_head_hash,
            "childGameId": self.child_game_id, "createdBy": self.created_by,
            "reason": self.reason, "createdAt": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class CorrectionRecord:
    correction_id: str
    source_game_id: str
    source_event_count: int
    source_head_hash: str | None
    corrected_game_id: str
    adjudication_case_id: str | None
    reason: str
    created_by: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "correctionId": self.correction_id, "sourceGameId": self.source_game_id,
            "sourceEventCount": self.source_event_count, "sourceHeadHash": self.source_head_hash,
            "correctedGameId": self.corrected_game_id, "adjudicationCaseId": self.adjudication_case_id,
            "reason": self.reason, "createdBy": self.created_by, "createdAt": self.created_at,
        }


class CanonicalHistoryStore:
    """One authoritative event history plus disposable snapshots/projections."""

    def __init__(self, path: str | Path, *, cipher: PayloadCipher | None = None):
        self.path = Path(path)
        self.cipher = cipher
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.events = SQLiteEventStore(self.path)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS canonical_snapshots_v17(
                  snapshot_id TEXT PRIMARY KEY,
                  game_id TEXT NOT NULL,
                  event_count INTEGER NOT NULL,
                  head_hash TEXT,
                  state_hash TEXT NOT NULL,
                  state_json TEXT NOT NULL,
                  schema_version INTEGER NOT NULL,
                  created_at TEXT NOT NULL,
                  engine_version TEXT NOT NULL,
                  profile_bundle_hash TEXT NOT NULL,
                  UNIQUE(game_id,event_count,state_hash)
                );
                CREATE INDEX IF NOT EXISTS idx_canonical_snapshots_game
                  ON canonical_snapshots_v17(game_id,event_count DESC);
                CREATE TABLE IF NOT EXISTS canonical_branches_v17(
                  branch_id TEXT PRIMARY KEY,
                  parent_game_id TEXT NOT NULL,
                  parent_event_count INTEGER NOT NULL,
                  parent_head_hash TEXT,
                  child_game_id TEXT NOT NULL UNIQUE,
                  created_by TEXT NOT NULL,
                  reason TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS canonical_corrections_v17(
                  correction_id TEXT PRIMARY KEY,
                  source_game_id TEXT NOT NULL,
                  source_event_count INTEGER NOT NULL,
                  source_head_hash TEXT,
                  corrected_game_id TEXT NOT NULL UNIQUE,
                  adjudication_case_id TEXT,
                  reason TEXT NOT NULL,
                  created_by TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );
                """
            )

    def append(self, *, game_id: str, events: Iterable[Event], profile_bundle_hash: str,
               engine_version: str) -> tuple[EventEnvelope, ...]:
        return self.events.append_core_events(
            game_id=game_id, events=events,
            profile_bundle_hash=profile_bundle_hash, engine_version=engine_version,
        )

    def snapshot(self, *, game_id: str, state: Mapping[str, Any], engine_version: str,
                 profile_bundle_hash: str, schema_version: int = 1) -> SnapshotRecord:
        verification = self.events.verify(game_id)
        if not verification["valid"]:
            raise RuntimeError("cannot snapshot an invalid event chain")
        payload = dict(state)
        state_hash = canonical_hash(payload)
        stored_payload = (
            self.cipher.encrypt_json(payload, aad=f"snapshot:{game_id}".encode("utf-8"))
            if self.cipher is not None else
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
        record = SnapshotRecord(
            snapshot_id=f"SNP-{uuid.uuid4().hex[:20].upper()}", game_id=game_id,
            event_count=int(verification["eventCount"]), head_hash=verification.get("headHash"),
            state_hash=state_hash, state_payload=payload, schema_version=int(schema_version),
            created_at=utc_now(), engine_version=engine_version, profile_bundle_hash=profile_bundle_hash,
        )
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO canonical_snapshots_v17(
                     snapshot_id,game_id,event_count,head_hash,state_hash,state_json,schema_version,
                     created_at,engine_version,profile_bundle_hash
                   ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    record.snapshot_id, record.game_id, record.event_count, record.head_hash,
                    record.state_hash, stored_payload,
                    record.schema_version, record.created_at, record.engine_version, record.profile_bundle_hash,
                ),
            )
        return record

    def latest_snapshot(self, game_id: str) -> SnapshotRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM canonical_snapshots_v17 WHERE game_id=? ORDER BY event_count DESC,created_at DESC LIMIT 1",
                (game_id,),
            ).fetchone()
        if row is None:
            return None
        stored = str(row["state_json"])
        payload = (
            self.cipher.decrypt_json(stored, aad=f"snapshot:{game_id}".encode("utf-8"))
            if stored.startswith("enc:v1:") and self.cipher is not None else
            json.loads(stored)
        )
        if stored.startswith("enc:v1:") and self.cipher is None:
            raise RuntimeError("canonical snapshot is encrypted but no payload cipher is configured")
        return SnapshotRecord(
            row["snapshot_id"], row["game_id"], int(row["event_count"]), row["head_hash"],
            row["state_hash"], payload, int(row["schema_version"]),
            row["created_at"], row["engine_version"], row["profile_bundle_hash"],
        )

    def create_branch(self, *, parent_game_id: str, child_game_id: str,
                      created_by: str, reason: str) -> BranchRecord:
        verification = self.events.verify(parent_game_id)
        if not verification["valid"]:
            raise RuntimeError("parent game event chain is invalid")
        record = BranchRecord(
            branch_id=f"BRN-{uuid.uuid4().hex[:20].upper()}", parent_game_id=parent_game_id,
            parent_event_count=int(verification["eventCount"]), parent_head_hash=verification.get("headHash"),
            child_game_id=child_game_id, created_by=created_by, reason=reason.strip(), created_at=utc_now(),
        )
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO canonical_branches_v17 VALUES(?,?,?,?,?,?,?,?)",
                (record.branch_id, record.parent_game_id, record.parent_event_count, record.parent_head_hash,
                 record.child_game_id, record.created_by, record.reason, record.created_at),
            )
        return record

    def create_correction(self, *, source_game_id: str, corrected_game_id: str,
                          created_by: str, reason: str,
                          adjudication_case_id: str | None = None) -> CorrectionRecord:
        verification = self.events.verify(source_game_id)
        if not verification["valid"]:
            raise RuntimeError("source game event chain is invalid")
        record = CorrectionRecord(
            correction_id=f"COR-{uuid.uuid4().hex[:20].upper()}", source_game_id=source_game_id,
            source_event_count=int(verification["eventCount"]), source_head_hash=verification.get("headHash"),
            corrected_game_id=corrected_game_id, adjudication_case_id=adjudication_case_id,
            reason=reason.strip(), created_by=created_by, created_at=utc_now(),
        )
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO canonical_corrections_v17 VALUES(?,?,?,?,?,?,?,?,?)",
                (record.correction_id, record.source_game_id, record.source_event_count, record.source_head_hash,
                 record.corrected_game_id, record.adjudication_case_id, record.reason, record.created_by, record.created_at),
            )
        return record

    def lineage(self, game_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            parent = conn.execute(
                "SELECT * FROM canonical_branches_v17 WHERE child_game_id=?", (game_id,)
            ).fetchone()
            correction = conn.execute(
                "SELECT * FROM canonical_corrections_v17 WHERE corrected_game_id=?", (game_id,)
            ).fetchone()
            children = conn.execute(
                "SELECT child_game_id,branch_id,created_at FROM canonical_branches_v17 WHERE parent_game_id=? ORDER BY created_at",
                (game_id,),
            ).fetchall()
            corrected = conn.execute(
                "SELECT corrected_game_id,correction_id,created_at FROM canonical_corrections_v17 WHERE source_game_id=? ORDER BY created_at",
                (game_id,),
            ).fetchall()
        return {
            "gameId": game_id,
            "parentBranch": dict(parent) if parent else None,
            "parentCorrection": dict(correction) if correction else None,
            "branches": [dict(row) for row in children],
            "corrections": [dict(row) for row in corrected],
        }

    def verify(self, game_id: str) -> dict[str, Any]:
        event_result = self.events.verify(game_id)
        snapshot = self.latest_snapshot(game_id)
        issues: list[str] = []
        if not event_result["valid"]:
            issues.append("event_chain")
        if snapshot is not None and snapshot.event_count > event_result.get("eventCount", 0):
            issues.append("snapshot_ahead_of_event_chain")
        return {
            "gameId": game_id, "valid": not issues, "issues": issues,
            "event": event_result, "latestSnapshot": snapshot.to_dict() if snapshot else None,
            "lineage": self.lineage(game_id),
        }
