from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from .request_context import CURRENT_DATABASE_IDENTITY


class JobStore:
    """Durable SQLite job queue for local and single-node deployments.

    The schema mirrors the production PostgreSQL queue and supports priorities,
    idempotency, leases, retries, cancellation and progress events. SQLite is
    intentionally the zero-configuration backend; PostgreSQL is the horizontal
    production backend.
    """

    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS jobs(
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL DEFAULT 'default',
                    kind TEXT NOT NULL DEFAULT 'video',
                    status TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 100,
                    progress REAL NOT NULL,
                    request_json TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    lease_owner TEXT,
                    lease_until REAL,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    idempotency_key TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS job_events(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            # Backward-compatible migrations for databases created by older builds.
            columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
            additions = {
                "tenant_id": "TEXT NOT NULL DEFAULT 'default'",
                "kind": "TEXT NOT NULL DEFAULT 'video'",
                "priority": "INTEGER NOT NULL DEFAULT 100",
                "attempts": "INTEGER NOT NULL DEFAULT 0",
                "max_attempts": "INTEGER NOT NULL DEFAULT 3",
                "lease_owner": "TEXT",
                "lease_until": "REAL",
                "cancel_requested": "INTEGER NOT NULL DEFAULT 0",
                "idempotency_key": "TEXT",
            }
            for name, definition in additions.items():
                if name not in columns:
                    conn.execute(f"ALTER TABLE jobs ADD COLUMN {name} {definition}")
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS jobs_idempotency ON jobs(tenant_id,kind,idempotency_key) WHERE idempotency_key IS NOT NULL"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS jobs_claim ON jobs(status,priority,created_at)"
            )

    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def create(
        self,
        job_id: str,
        request: dict[str, Any],
        *,
        kind: str = "video",
        tenant_id: str = "default",
        priority: int = 100,
        max_attempts: int = 3,
        idempotency_key: str | None = None,
    ) -> str:
        with self._lock, self._connect() as conn:
            if idempotency_key:
                row = conn.execute(
                    "SELECT id FROM jobs WHERE tenant_id=? AND kind=? AND idempotency_key=?",
                    (tenant_id, kind, idempotency_key),
                ).fetchone()
                if row:
                    return str(row["id"])
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """INSERT INTO jobs(id,tenant_id,kind,status,priority,progress,request_json,max_attempts,idempotency_key)
                            VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    job_id,
                    tenant_id,
                    kind,
                    "queued",
                    priority,
                    0.0,
                    json.dumps(request, ensure_ascii=False),
                    max_attempts,
                    idempotency_key,
                ),
            )
            conn.execute(
                "INSERT INTO job_events(job_id,type,payload_json) VALUES(?,?,?)",
                (job_id, "queued", "{}"),
            )
            conn.commit()
        return job_id

    def claim(
        self,
        worker_id: str,
        *,
        kinds: tuple[str, ...] = ("video",),
        lease_seconds: int = 60,
    ) -> dict[str, Any] | None:
        now = time.time()
        placeholders = ",".join("?" for _ in kinds)
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                f"""SELECT * FROM jobs WHERE kind IN ({placeholders}) AND cancel_requested=0
               AND attempts < max_attempts AND (status='queued' OR (status='running' AND COALESCE(lease_until,0)<?))
               ORDER BY priority ASC, created_at ASC LIMIT 1""",
                (*kinds, now),
            ).fetchone()
            if row is None:
                conn.commit()
                return None
            lease = now + lease_seconds
            conn.execute(
                "UPDATE jobs SET status='running',attempts=attempts+1,lease_owner=?,lease_until=?,updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (worker_id, lease, row["id"]),
            )
            conn.execute(
                "INSERT INTO job_events(job_id,type,payload_json) VALUES(?,?,?)",
                (
                    row["id"],
                    "claimed",
                    json.dumps({"workerId": worker_id, "leaseUntil": lease}),
                ),
            )
            conn.commit()
        return self.get(str(row["id"]))

    def heartbeat(
        self,
        job_id: str,
        worker_id: str,
        *,
        lease_seconds: int = 60,
        fencing_token: int | None = None,
    ) -> bool:
        where = "id=? AND lease_owner=? AND status='running'"
        values = [time.time() + lease_seconds, job_id, worker_id]
        if fencing_token is not None:
            where += " AND attempts=?"
            values.append(int(fencing_token))
        with self._lock, self._connect() as conn:
            result = conn.execute(
                f"UPDATE jobs SET lease_until=?,updated_at=CURRENT_TIMESTAMP WHERE {where}",
                values,
            )
            return result.rowcount == 1

    def update(
        self,
        job_id: str,
        *,
        status: str | None = None,
        progress: float | None = None,
        result: dict | None = None,
        error: str | None = None,
        worker_id: str | None = None,
        fencing_token: int | None = None,
    ) -> bool:
        fields = []
        values = []
        if status is not None:
            fields.append("status=?")
            values.append(status)
        if progress is not None:
            fields.append("progress=?")
            values.append(max(0.0, min(1.0, float(progress))))
        if result is not None:
            fields.append("result_json=?")
            values.append(json.dumps(result, ensure_ascii=False))
        if error is not None:
            fields.append("error=?")
            values.append(error)
        elif status == "completed":
            fields.append("error=NULL")
        if status in {"queued", "completed", "failed", "cancelled", "exhausted"}:
            fields.extend(["lease_owner=NULL", "lease_until=NULL"])
        fields.append("updated_at=CURRENT_TIMESTAMP")
        values.append(job_id)
        where = "id=?"
        if worker_id is not None:
            where += " AND lease_owner=?"
            values.append(worker_id)
        if fencing_token is not None:
            where += " AND attempts=?"
            values.append(int(fencing_token))
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            updated = conn.execute(
                f"UPDATE jobs SET {','.join(fields)} WHERE {where}", values
            )
            if updated.rowcount != 1:
                return False
            event = status or ("progress" if progress is not None else "updated")
            conn.execute(
                "INSERT INTO job_events(job_id,type,payload_json) VALUES(?,?,?)",
                (job_id, event, json.dumps({"progress": progress}, ensure_ascii=False)),
            )
            return True

    def request_cancel(self, job_id: str) -> bool:
        with self._lock, self._connect() as conn:
            result = conn.execute(
                "UPDATE jobs SET cancel_requested=1,status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END,updated_at=CURRENT_TIMESTAMP WHERE id=? AND status NOT IN ('completed','failed','cancelled','exhausted')",
                (job_id,),
            )
            if result.rowcount:
                conn.execute(
                    "INSERT INTO job_events(job_id,type,payload_json) VALUES(?,?,?)",
                    (job_id, "cancel_requested", "{}"),
                )
            return result.rowcount == 1

    def cancelled(self, job_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT cancel_requested,status FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
        return bool(row and (row["cancel_requested"] or row["status"] == "cancelled"))

    def get(self, job_id: str) -> dict[str, Any] | None:
        tenant = CURRENT_DATABASE_IDENTITY.get().tenant_id
        with self._connect() as conn:
            if tenant:
                row = conn.execute(
                    "SELECT * FROM jobs WHERE id=? AND tenant_id=?", (job_id, tenant),
                ).fetchone()
            else:
                row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            return None
        return {
            "jobId": row["id"],
            "tenantId": row["tenant_id"],
            "kind": row["kind"],
            "status": row["status"],
            "priority": row["priority"],
            "progress": row["progress"],
            "request": json.loads(row["request_json"]),
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
            "error": row["error"],
            "attempts": row["attempts"],
            "maxAttempts": row["max_attempts"],
            "leaseOwner": row["lease_owner"],
            "leaseUntil": row["lease_until"],
            "cancelRequested": bool(row["cancel_requested"]),
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
        }

    def list_jobs(
        self,
        *,
        kinds: tuple[str, ...] | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        size = max(1, min(500, int(limit)))
        clauses: list[str] = []
        values: list[Any] = []
        tenant = CURRENT_DATABASE_IDENTITY.get().tenant_id
        if tenant:
            clauses.append("tenant_id=?")
            values.append(tenant)
        if kinds:
            placeholders = ",".join("?" for _ in kinds)
            clauses.append(f"kind IN ({placeholders})")
            values.extend(str(item) for item in kinds)
        if status:
            clauses.append("status=?")
            values.append(str(status))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT id FROM jobs{where} ORDER BY created_at DESC,rowid DESC LIMIT ?",
                (*values, size),
            ).fetchall()
        return [item for row in rows if (item := self.get(str(row["id"]))) is not None]

    def events(self, job_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT type,payload_json,created_at FROM job_events WHERE job_id=? ORDER BY id",
                (job_id,),
            ).fetchall()
        return [
            {
                "type": r["type"],
                "payload": json.loads(r["payload_json"]),
                "createdAt": r["created_at"],
            }
            for r in rows
        ]
