from __future__ import annotations

import json
import sqlite3
import threading
import uuid
import weakref
from pathlib import Path
from typing import Any

from baloot_platform.storage.request_context import contextual_connection

from baloot_platform.core.enums import Seat

from .models import LiveEvent, LiveRoom, LiveStatus, SeatAssignment, utc_now


class LiveVersionConflict(RuntimeError):
    pass


class SQLiteLiveStore:
    """Normalized event-sourced live-room store with compare-and-swap heads."""

    def __init__(self, path: str | Path, *, synchronous: str = "FULL"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.synchronous = str(synchronous).strip().upper()
        if self.synchronous not in {"NORMAL", "FULL", "EXTRA"}:
            raise ValueError("SQLite live-store synchronous mode must be NORMAL, FULL, or EXTRA")
        self._lock = threading.RLock()
        self._write_connection = self._open_connection(enable_wal=True)
        self._write_connection_finalizer = weakref.finalize(
            self,
            self._write_connection.close,
        )
        self._initialize()

    def _open_connection(self, *, enable_wal: bool = False) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.path,
            timeout=30,
            isolation_level=None,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        if enable_wal:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(f"PRAGMA synchronous={self.synchronous}")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _connect(self) -> sqlite3.Connection:
        return self._open_connection()

    def _initialize(self) -> None:
        with self._write_connection as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS live_room_heads_v20(
                  room_id TEXT PRIMARY KEY,
                  room_code TEXT NOT NULL UNIQUE,
                  game_id TEXT NOT NULL,
                  status TEXT NOT NULL,
                  version INTEGER NOT NULL,
                  current_sequence INTEGER NOT NULL,
                  updated_at TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  room_hash TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_live_room_heads_game_v20 ON live_room_heads_v20(game_id,updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_live_room_heads_status_v20 ON live_room_heads_v20(status,updated_at DESC);
                CREATE TABLE IF NOT EXISTS live_events_v20(
                  room_id TEXT NOT NULL REFERENCES live_room_heads_v20(room_id) ON DELETE CASCADE,
                  sequence INTEGER NOT NULL,
                  event_type TEXT NOT NULL,
                  actor_subject_id TEXT,
                  actor_seat TEXT,
                  created_at TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  previous_hash TEXT NOT NULL,
                  event_hash TEXT NOT NULL,
                  PRIMARY KEY(room_id,sequence)
                );
                CREATE TABLE IF NOT EXISTS live_outbox_v20(
                  outbox_id TEXT PRIMARY KEY,
                  room_id TEXT NOT NULL,
                  sequence INTEGER NOT NULL,
                  topic TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  published_at TEXT,
                  UNIQUE(room_id,sequence,topic)
                );
                CREATE INDEX IF NOT EXISTS idx_live_outbox_unpublished_v20 ON live_outbox_v20(published_at,created_at);
                """
            )

    @staticmethod
    def _deserialize(payload: dict[str, Any], events: list[dict[str, Any]] | None = None,
                     *, version: int | None = None) -> LiveRoom:
        seats = {
            name: SeatAssignment(
                seat=Seat.parse(item["seat"]), subject_id=item["subjectId"], display_name=item["displayName"],
                ready=bool(item.get("ready")), connected=bool(item.get("connected", True)),
                joined_at=item.get("joinedAt", ""), last_seen_at=item.get("lastSeenAt", ""),
                reconnect_token_hash=item.get("reconnectTokenHash", ""), takeover=item.get("takeover"),
            )
            for name, item in dict(payload.get("seats") or {}).items()
        }
        event_values = events if events is not None else list(payload.get("events") or [])
        parsed = [
            LiveEvent(
                sequence=int(item["sequence"]), event_type=item["eventType"],
                actor_subject_id=item.get("actorSubjectId"), actor_seat=item.get("actorSeat"),
                created_at=item["createdAt"], payload=dict(item.get("payload") or {}),
                previous_hash=item.get("previousHash", ""), event_hash=item.get("eventHash", ""),
            )
            for item in event_values
        ]
        return LiveRoom(
            room_id=payload["roomId"], room_code=payload["roomCode"], game_id=payload["gameId"],
            owner_subject_id=payload["ownerSubjectId"], status=LiveStatus(payload["status"]),
            created_at=payload["createdAt"], updated_at=payload["updatedAt"],
            turn_seconds=int(payload["turnSeconds"]), reconnect_grace_seconds=int(payload["reconnectGraceSeconds"]),
            allow_spectators=bool(payload["allowSpectators"]), assistance_policy=payload["assistancePolicy"],
            referee_subject_id=payload.get("refereeSubjectId"), seats=seats,
            spectators=dict(payload.get("spectators") or {}), turn_deadline=payload.get("turnDeadline"),
            paused_reason=payload.get("pausedReason"), events=parsed, room_hash=payload.get("roomHash", ""),
            version=int(version if version is not None else payload.get("version", 0)),
            metadata=dict(payload.get("metadata") or {}),
        )

    @staticmethod
    def _event_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "sequence": int(row["sequence"]), "eventType": row["event_type"],
            "actorSubjectId": row["actor_subject_id"], "actorSeat": row["actor_seat"],
            "createdAt": row["created_at"], "payload": json.loads(row["payload_json"]),
            "previousHash": row["previous_hash"], "eventHash": row["event_hash"],
        }

    def save(self, room: LiveRoom, *, expected_version: int | None = None) -> int:
        payload = room.to_dict(include_events=False, include_private=True, include_version=False)
        with self._lock, self._write_connection as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute(
                "SELECT version,current_sequence FROM live_room_heads_v20 WHERE room_id=?", (room.room_id,)
            ).fetchone()
            if current is None:
                if expected_version not in {None, 0}:
                    conn.execute("ROLLBACK"); raise LiveVersionConflict("live room does not exist")
                current_version = 0; current_sequence = 0; new_version = 1
                conn.execute(
                    """INSERT INTO live_room_heads_v20(room_id,room_code,game_id,status,version,current_sequence,updated_at,payload_json,room_hash)
                       VALUES(?,?,?,?,?,?,?,?,?)""",
                    (room.room_id, room.room_code, room.game_id, room.status.value, new_version,
                     len(room.events), room.updated_at, json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")), room.room_hash),
                )
            else:
                current_version = int(current["version"]); current_sequence = int(current["current_sequence"])
                expected = current_version if expected_version is None else int(expected_version)
                if expected != current_version:
                    conn.execute("ROLLBACK"); raise LiveVersionConflict(
                        f"live room version conflict: expected={expected} current={current_version}"
                    )
                new_version = current_version + 1
                result = conn.execute(
                    """UPDATE live_room_heads_v20 SET room_code=?,game_id=?,status=?,version=?,current_sequence=?,
                       updated_at=?,payload_json=?,room_hash=? WHERE room_id=? AND version=?""",
                    (room.room_code, room.game_id, room.status.value, new_version, len(room.events), room.updated_at,
                     json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")), room.room_hash,
                     room.room_id, current_version),
                )
                if result.rowcount != 1:
                    conn.execute("ROLLBACK"); raise LiveVersionConflict("live room compare-and-swap failed")
            for event in room.events[current_sequence:]:
                conn.execute(
                    """INSERT INTO live_events_v20(room_id,sequence,event_type,actor_subject_id,actor_seat,created_at,
                       payload_json,previous_hash,event_hash) VALUES(?,?,?,?,?,?,?,?,?)""",
                    (room.room_id, event.sequence, event.event_type, event.actor_subject_id, event.actor_seat,
                     event.created_at, json.dumps(event.payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                     event.previous_hash, event.event_hash),
                )
                outbox_id = f"LOB-{uuid.uuid5(uuid.NAMESPACE_URL, f'{room.room_id}:{event.sequence}').hex.upper()}"
                conn.execute(
                    """INSERT OR IGNORE INTO live_outbox_v20(outbox_id,room_id,sequence,topic,payload_json,created_at)
                       VALUES(?,?,?,?,?,?)""",
                    (outbox_id, room.room_id, event.sequence, "live.room.event",
                     json.dumps({"roomId": room.room_id, "sequence": event.sequence}, separators=(",", ":")), utc_now()),
                )
            conn.execute("COMMIT")
        room.version = new_version
        return new_version

    def _events(self, conn: sqlite3.Connection, room_id: str, *, since: int = 0) -> list[dict[str, Any]]:
        rows = conn.execute(
            "SELECT * FROM live_events_v20 WHERE room_id=? AND sequence>? ORDER BY sequence", (room_id, int(since))
        ).fetchall()
        return [self._event_dict(row) for row in rows]

    def get(self, identifier: str) -> LiveRoom | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM live_room_heads_v20 WHERE room_id=? OR room_code=?", (identifier, identifier.upper())
            ).fetchone()
            if not row: return None
            events = self._events(conn, str(row["room_id"]))
        payload = json.loads(row["payload_json"]); payload["roomHash"] = row["room_hash"]
        return self._deserialize(payload, events, version=int(row["version"]))

    def events_since(self, identifier: str, sequence: int) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT room_id,version,current_sequence FROM live_room_heads_v20 WHERE room_id=? OR room_code=?", (identifier, identifier.upper())).fetchone()
            if not row: raise KeyError(identifier)
            events = self._events(conn, str(row["room_id"]), since=sequence)
        return {"roomId": row["room_id"], "version": int(row["version"]),
                "events": events, "nextSequence": int(row["current_sequence"])}

    def list(self, *, subject_id: str | None = None, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        clauses=[]; args=[]
        if status: clauses.append("status=?"); args.append(LiveStatus(status).value)
        sql="SELECT payload_json,version,room_hash FROM live_room_heads_v20"+(" WHERE "+" AND ".join(clauses) if clauses else "")+" ORDER BY updated_at DESC LIMIT ?"; args.append(max(1,min(1000,int(limit))))
        with self._connect() as conn: rows=conn.execute(sql,args).fetchall()
        values=[]
        for row in rows:
            value=json.loads(row["payload_json"]); value["version"]=int(row["version"]); value["roomHash"]=row["room_hash"]; values.append(value)
        if subject_id:
            values=[v for v in values if v.get("ownerSubjectId")==subject_id or any(i.get("subjectId")==subject_id for i in v.get("seats",{}).values()) or subject_id in v.get("spectators",{})]
        return values

    def claim_outbox(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock, self._write_connection as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows=conn.execute("SELECT * FROM live_outbox_v20 WHERE published_at IS NULL ORDER BY created_at LIMIT ?",(max(1,min(1000,int(limit))),)).fetchall()
            now=utc_now()
            for row in rows: conn.execute("UPDATE live_outbox_v20 SET published_at=? WHERE outbox_id=?",(now,row["outbox_id"]))
            conn.execute("COMMIT")
        return [{"outboxId":r["outbox_id"],"roomId":r["room_id"],"sequence":int(r["sequence"]),"topic":r["topic"],"payload":json.loads(r["payload_json"])} for r in rows]

    def close(self) -> None:
        with self._lock:
            self._write_connection_finalizer()


class PostgresLiveStore:
    """Normalized PostgreSQL live store with row-lock CAS and outbox."""
    def __init__(self, dsn: str):
        try:
            from psycopg_pool import ConnectionPool
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError('Install with: pip install "baloot-platform[postgres]"') from exc
        self.pool = ConnectionPool(dsn, min_size=1, max_size=24, open=True)

    def save(self, room: LiveRoom, *, expected_version: int | None = None) -> int:
        payload=room.to_dict(include_events=False,include_private=True,include_version=False)
        with contextual_connection(self.pool) as conn:
            row=conn.execute("SELECT version,current_sequence FROM live.room_heads WHERE room_id=%s FOR UPDATE",(room.room_id,)).fetchone()
            if row is None:
                if expected_version not in {None,0}: raise LiveVersionConflict("live room does not exist")
                current_sequence=0; new_version=1
                conn.execute("INSERT INTO live.room_heads(room_id,room_code,game_id,status,version,current_sequence,updated_at,payload_json,room_hash) VALUES(%s,%s,%s,%s,%s,%s,%s::timestamptz,%s::jsonb,%s)",(room.room_id,room.room_code,room.game_id,room.status.value,new_version,len(room.events),room.updated_at,json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(",",":")),room.room_hash))
            else:
                current_version=int(row[0]); current_sequence=int(row[1]); expected=current_version if expected_version is None else int(expected_version)
                if expected!=current_version: raise LiveVersionConflict(f"live room version conflict: expected={expected} current={current_version}")
                new_version=current_version+1
                updated=conn.execute("UPDATE live.room_heads SET room_code=%s,game_id=%s,status=%s,version=%s,current_sequence=%s,updated_at=%s::timestamptz,payload_json=%s::jsonb,room_hash=%s WHERE room_id=%s AND version=%s",(room.room_code,room.game_id,room.status.value,new_version,len(room.events),room.updated_at,json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(",",":")),room.room_hash,room.room_id,current_version))
                if updated.rowcount!=1: raise LiveVersionConflict("live room compare-and-swap failed")
            for event in room.events[current_sequence:]:
                conn.execute("INSERT INTO live.events(room_id,sequence,event_type,actor_subject_id,actor_seat,created_at,payload_json,previous_hash,event_hash) VALUES(%s,%s,%s,%s,%s,%s::timestamptz,%s::jsonb,%s,%s)",(room.room_id,event.sequence,event.event_type,event.actor_subject_id,event.actor_seat,event.created_at,json.dumps(event.payload,ensure_ascii=False,sort_keys=True,separators=(",",":")),event.previous_hash,event.event_hash))
                conn.execute("INSERT INTO live.outbox(outbox_id,room_id,sequence,topic,payload_json,created_at) VALUES(%s,%s,%s,'live.room.event',%s::jsonb,now()) ON CONFLICT(room_id,sequence,topic) DO NOTHING",(f"LOB-{uuid.uuid5(uuid.NAMESPACE_URL,f'{room.room_id}:{event.sequence}').hex.upper()}",room.room_id,event.sequence,json.dumps({"roomId":room.room_id,"sequence":event.sequence},separators=(",",":"))))
        room.version=new_version; return new_version

    def get(self, identifier: str) -> LiveRoom | None:
        with contextual_connection(self.pool) as conn:
            row=conn.execute("SELECT room_id,version,payload_json,room_hash FROM live.room_heads WHERE room_id=%s OR room_code=upper(%s)",(identifier,identifier)).fetchone()
            if not row: return None
            events=conn.execute("SELECT sequence,event_type,actor_subject_id,actor_seat,created_at,payload_json,previous_hash,event_hash FROM live.events WHERE room_id=%s ORDER BY sequence",(row[0],)).fetchall()
        payload=dict(row[2]); payload["roomHash"]=row[3]
        values=[{"sequence":int(e[0]),"eventType":e[1],"actorSubjectId":e[2],"actorSeat":e[3],"createdAt":str(e[4]),"payload":dict(e[5]),"previousHash":e[6],"eventHash":e[7]} for e in events]
        return SQLiteLiveStore._deserialize(payload,values,version=int(row[1]))

    def events_since(self, identifier: str, sequence: int) -> dict[str, Any]:
        room=self.get(identifier)
        if room is None: raise KeyError(identifier)
        events=[e.to_dict() for e in room.events if e.sequence>int(sequence)]
        return {"roomId":room.room_id,"version":room.version,"events":events,"nextSequence":room.events[-1].sequence if room.events else 0}

    def list(self, *, subject_id: str|None=None, status: str|None=None, limit: int=100) -> list[dict[str,Any]]:
        clauses=[]; args=[]
        if status: clauses.append("status=%s"); args.append(LiveStatus(status).value)
        sql="SELECT payload_json,version,room_hash FROM live.room_heads"+(" WHERE "+" AND ".join(clauses) if clauses else "")+" ORDER BY updated_at DESC LIMIT %s"; args.append(max(1,min(1000,int(limit))))
        with contextual_connection(self.pool) as conn: rows=conn.execute(sql,args).fetchall()
        values=[]
        for row in rows:
            value=dict(row[0]); value["version"]=int(row[1]); value["roomHash"]=row[2]; values.append(value)
        if subject_id: values=[v for v in values if v.get("ownerSubjectId")==subject_id or any(i.get("subjectId")==subject_id for i in v.get("seats",{}).values()) or subject_id in v.get("spectators",{})]
        return values

    def claim_outbox(self, *, limit: int=100) -> list[dict[str,Any]]:
        with contextual_connection(self.pool) as conn:
            rows=conn.execute("SELECT outbox_id,room_id,sequence,topic,payload_json FROM live.outbox WHERE published_at IS NULL ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT %s",(max(1,min(1000,int(limit))),)).fetchall()
            if rows: conn.execute("UPDATE live.outbox SET published_at=now() WHERE outbox_id=ANY(%s)",([r[0] for r in rows],))
        return [{"outboxId":r[0],"roomId":r[1],"sequence":int(r[2]),"topic":r[3],"payload":dict(r[4])} for r in rows]

    def close(self) -> None: self.pool.close()
