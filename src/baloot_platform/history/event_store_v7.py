"""Append-only, tamper-evident Baloot event stores.

The event log is the authoritative durable history. Snapshots are projections
that can be rebuilt and discarded. Both SQLite development and PostgreSQL
production implementations use the same canonical envelope and hash chain.
"""
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

from baloot_platform.core.models import Event


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    event_id: str
    game_id: str
    sequence: int
    event_type: str
    schema_version: int
    payload: Mapping[str, Any]
    profile_bundle_hash: str
    engine_version: str
    previous_event_hash: str | None
    event_hash: str
    occurred_at: str
    actor_seat: str | None = None
    actor_subject_id: str | None = None

    @classmethod
    def create(
        cls,
        *,
        game_id: str,
        sequence: int,
        event_type: str,
        payload: Mapping[str, Any],
        profile_bundle_hash: str,
        engine_version: str,
        previous_event_hash: str | None,
        occurred_at: str | None = None,
        actor_seat: str | None = None,
        actor_subject_id: str | None = None,
    ) -> "EventEnvelope":
        when = occurred_at or _now()
        identity = {
            "gameId": game_id,
            "sequence": int(sequence),
            "eventType": event_type,
            "schemaVersion": 1,
            "payload": dict(payload),
            "profileBundleHash": profile_bundle_hash,
            "engineVersion": engine_version,
            "previousEventHash": previous_event_hash,
            "occurredAt": when,
            "actorSeat": actor_seat,
            "actorSubjectId": actor_subject_id,
        }
        digest = hashlib.sha256(_canonical(identity)).hexdigest()
        return cls(
            f"EVT-{digest[:24].upper()}", game_id, int(sequence), event_type, 1,
            dict(payload), profile_bundle_hash, engine_version, previous_event_hash,
            digest, when, actor_seat, actor_subject_id,
        )

    def verify(self, previous: str | None) -> bool:
        rebuilt = self.create(
            game_id=self.game_id,
            sequence=self.sequence,
            event_type=self.event_type,
            payload=self.payload,
            profile_bundle_hash=self.profile_bundle_hash,
            engine_version=self.engine_version,
            previous_event_hash=previous,
            occurred_at=self.occurred_at,
            actor_seat=self.actor_seat,
            actor_subject_id=self.actor_subject_id,
        )
        return self.previous_event_hash == previous and rebuilt.event_hash == self.event_hash and rebuilt.event_id == self.event_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "eventId": self.event_id,
            "gameId": self.game_id,
            "sequence": self.sequence,
            "eventType": self.event_type,
            "schemaVersion": self.schema_version,
            "payload": dict(self.payload),
            "profileBundleHash": self.profile_bundle_hash,
            "engineVersion": self.engine_version,
            "previousEventHash": self.previous_event_hash,
            "eventHash": self.event_hash,
            "occurredAt": self.occurred_at,
            "actorSeat": self.actor_seat,
            "actorSubjectId": self.actor_subject_id,
        }


class SQLiteEventStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS game_events_v7(
                    event_id TEXT PRIMARY KEY,
                    game_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    profile_bundle_hash TEXT NOT NULL,
                    engine_version TEXT NOT NULL,
                    previous_event_hash TEXT,
                    event_hash TEXT NOT NULL UNIQUE,
                    occurred_at TEXT NOT NULL,
                    actor_seat TEXT,
                    actor_subject_id TEXT,
                    UNIQUE(game_id,sequence)
                );
                CREATE INDEX IF NOT EXISTS idx_game_events_v7_game_sequence
                    ON game_events_v7(game_id,sequence);
                CREATE TABLE IF NOT EXISTS game_event_heads_v7(
                    game_id TEXT PRIMARY KEY,
                    event_count INTEGER NOT NULL,
                    head_hash TEXT,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _from_row(row: sqlite3.Row) -> EventEnvelope:
        return EventEnvelope(
            row["event_id"], row["game_id"], int(row["sequence"]), row["event_type"],
            int(row["schema_version"]), json.loads(row["payload_json"]),
            row["profile_bundle_hash"], row["engine_version"], row["previous_event_hash"],
            row["event_hash"], row["occurred_at"], row["actor_seat"], row["actor_subject_id"],
        )

    def append_core_events(
        self,
        *,
        game_id: str,
        events: Iterable[Event],
        profile_bundle_hash: str,
        engine_version: str,
    ) -> tuple[EventEnvelope, ...]:
        appended: list[EventEnvelope] = []
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            head = conn.execute(
                "SELECT event_count,head_hash FROM game_event_heads_v7 WHERE game_id=?", (game_id,)
            ).fetchone()
            count = int(head["event_count"]) if head else 0
            previous = str(head["head_hash"]) if head and head["head_hash"] is not None else None
            for event in sorted(events, key=lambda item: item.sequence):
                existing = conn.execute(
                    "SELECT * FROM game_events_v7 WHERE game_id=? AND sequence=?", (game_id, int(event.sequence))
                ).fetchone()
                if existing is not None:
                    envelope = self._from_row(existing)
                    expected = EventEnvelope.create(
                        game_id=game_id, sequence=event.sequence, event_type=event.type,
                        payload=event.data, profile_bundle_hash=profile_bundle_hash,
                        engine_version=engine_version, previous_event_hash=envelope.previous_event_hash,
                        occurred_at=envelope.occurred_at, actor_seat=envelope.actor_seat,
                        actor_subject_id=envelope.actor_subject_id,
                    )
                    if envelope.event_hash != expected.event_hash:
                        raise RuntimeError(f"immutable event conflict at {game_id}:{event.sequence}")
                    previous = envelope.event_hash
                    count = max(count, event.sequence + 1)
                    continue
                if event.sequence != count:
                    raise RuntimeError(f"event sequence gap for {game_id}: expected {count}, got {event.sequence}")
                envelope = EventEnvelope.create(
                    game_id=game_id, sequence=event.sequence, event_type=event.type,
                    payload=event.data, profile_bundle_hash=profile_bundle_hash,
                    engine_version=engine_version, previous_event_hash=previous,
                    actor_seat=str(event.data.get("seat")) if event.data.get("seat") is not None else None,
                )
                conn.execute(
                    """INSERT INTO game_events_v7(
                           event_id,game_id,sequence,event_type,schema_version,payload_json,
                           profile_bundle_hash,engine_version,previous_event_hash,event_hash,
                           occurred_at,actor_seat,actor_subject_id
                       ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        envelope.event_id,envelope.game_id,envelope.sequence,envelope.event_type,
                        envelope.schema_version,json.dumps(envelope.payload,ensure_ascii=False,sort_keys=True,separators=(",",":")),
                        envelope.profile_bundle_hash,envelope.engine_version,envelope.previous_event_hash,
                        envelope.event_hash,envelope.occurred_at,envelope.actor_seat,envelope.actor_subject_id,
                    ),
                )
                appended.append(envelope)
                previous = envelope.event_hash
                count += 1
            conn.execute(
                """INSERT INTO game_event_heads_v7(game_id,event_count,head_hash,updated_at)
                   VALUES(?,?,?,?) ON CONFLICT(game_id) DO UPDATE SET
                   event_count=excluded.event_count,head_hash=excluded.head_hash,updated_at=excluded.updated_at""",
                (game_id,count,previous,_now()),
            )
            conn.commit()
        return tuple(appended)

    def list(self, game_id: str) -> tuple[EventEnvelope, ...]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM game_events_v7 WHERE game_id=? ORDER BY sequence", (game_id,)
            ).fetchall()
        return tuple(self._from_row(row) for row in rows)

    def verify(self, game_id: str) -> dict[str, Any]:
        previous: str | None = None
        records = self.list(game_id)
        for index, event in enumerate(records):
            if event.sequence != index or not event.verify(previous):
                return {"valid": False, "gameId": game_id, "failedSequence": index}
            previous = event.event_hash
        return {"valid": True, "gameId": game_id, "eventCount": len(records), "headHash": previous}


class PostgresEventStore:
    """PostgreSQL append-only event store with transaction/advisory locking."""

    def __init__(self, dsn: str):
        try:
            from psycopg_pool import ConnectionPool
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError('Install with: pip install "baloot-platform[postgres]"') from exc
        self.pool = ConnectionPool(dsn, min_size=1, max_size=16, open=True)

    def append_core_events(self, *, game_id: str, events: Iterable[Event], profile_bundle_hash: str, engine_version: str) -> tuple[EventEnvelope, ...]:
        appended: list[EventEnvelope] = []
        with self.pool.connection() as conn, conn.transaction():
            # Stable per-game transaction lock; snapshots and events may be
            # written by different API nodes without interleaving sequences.
            conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (game_id,))
            head = conn.execute(
                "SELECT event_count,head_hash FROM game_event_heads_v7 WHERE game_id=%s FOR UPDATE", (game_id,)
            ).fetchone()
            count = int(head[0]) if head else 0
            previous = str(head[1]) if head and head[1] is not None else None
            for event in sorted(events, key=lambda item: item.sequence):
                existing = conn.execute(
                    "SELECT event_hash,previous_event_hash,occurred_at,actor_seat,actor_subject_id FROM game_events_v7 WHERE game_id=%s AND sequence=%s",
                    (game_id,int(event.sequence)),
                ).fetchone()
                if existing:
                    expected = EventEnvelope.create(
                        game_id=game_id,sequence=event.sequence,event_type=event.type,payload=event.data,
                        profile_bundle_hash=profile_bundle_hash,engine_version=engine_version,
                        previous_event_hash=existing[1],occurred_at=existing[2].isoformat(),
                        actor_seat=existing[3],actor_subject_id=existing[4],
                    )
                    if str(existing[0]) != expected.event_hash:
                        raise RuntimeError(f"immutable event conflict at {game_id}:{event.sequence}")
                    previous = str(existing[0]); count=max(count,event.sequence+1); continue
                if event.sequence != count:
                    raise RuntimeError(f"event sequence gap for {game_id}: expected {count}, got {event.sequence}")
                envelope = EventEnvelope.create(
                    game_id=game_id,sequence=event.sequence,event_type=event.type,payload=event.data,
                    profile_bundle_hash=profile_bundle_hash,engine_version=engine_version,previous_event_hash=previous,
                    actor_seat=str(event.data.get("seat")) if event.data.get("seat") is not None else None,
                )
                conn.execute(
                    """INSERT INTO game_events_v7(event_id,game_id,sequence,event_type,schema_version,payload_json,
                           profile_bundle_hash,engine_version,previous_event_hash,event_hash,occurred_at,actor_seat,actor_subject_id)
                       VALUES(%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s::timestamptz,%s,%s)""",
                    (envelope.event_id,game_id,envelope.sequence,envelope.event_type,envelope.schema_version,
                     json.dumps(envelope.payload,ensure_ascii=False,sort_keys=True,separators=(",",":")),
                     profile_bundle_hash,engine_version,previous,envelope.event_hash,envelope.occurred_at,
                     envelope.actor_seat,envelope.actor_subject_id),
                )
                appended.append(envelope); previous=envelope.event_hash; count+=1
            conn.execute(
                """INSERT INTO game_event_heads_v7(game_id,event_count,head_hash,updated_at)
                   VALUES(%s,%s,%s,now()) ON CONFLICT(game_id) DO UPDATE SET
                   event_count=excluded.event_count,head_hash=excluded.head_hash,updated_at=excluded.updated_at""",
                (game_id,count,previous),
            )
        return tuple(appended)
