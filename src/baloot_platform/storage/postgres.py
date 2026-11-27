from __future__ import annotations

import hashlib
import json
from typing import Any

from baloot_platform.storage.codec_v7 import (
    decode as decode_payload,
    encode as encode_payload,
)
from baloot_platform.storage.request_context import CURRENT_DATABASE_IDENTITY


class _PostgresBase:
    def __init__(self, dsn: str):
        try:
            from psycopg_pool import ConnectionPool
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                'Install with: pip install "baloot-platform[postgres]"'
            ) from exc
        self.pool = ConnectionPool(dsn, min_size=1, max_size=10, open=True)


class PostgresJobStore(_PostgresBase):
    """Horizontally scalable durable queue using `FOR UPDATE SKIP LOCKED`."""

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
        payload = dict(request)
        if idempotency_key is not None:
            payload["idempotencyKey"] = idempotency_key
        with self.pool.connection() as conn, conn.transaction():
            if idempotency_key:
                row = conn.execute(
                    "SELECT job_id FROM jobs WHERE tenant_id=%s AND kind=%s AND request_json->>'idempotencyKey'=%s",
                    (tenant_id, kind, idempotency_key),
                ).fetchone()
                if row:
                    return str(row[0])
            conn.execute(
                """INSERT INTO jobs(job_id,tenant_id,kind,status,priority,progress,request_json,max_attempts)
                   VALUES(%s,%s,%s,'queued',%s,0,%s::jsonb,%s)""",
                (job_id, tenant_id, kind, priority, json.dumps(payload), max_attempts),
            )
            conn.execute(
                "INSERT INTO job_events(job_id,type,payload_json) VALUES(%s,'queued','{}'::jsonb)",
                (job_id,),
            )
        return job_id

    def claim(
        self,
        worker_id: str,
        *,
        kinds: tuple[str, ...] = ("video",),
        lease_seconds: int = 60,
    ) -> dict[str, Any] | None:
        with self.pool.connection() as conn, conn.transaction():
            row = conn.execute(
                """SELECT job_id FROM jobs
                   WHERE kind=ANY(%s) AND cancel_requested=FALSE AND attempts<max_attempts
                     AND (status='queued' OR (status='running' AND (lease_until IS NULL OR lease_until<now())))
                   ORDER BY priority,created_at FOR UPDATE SKIP LOCKED LIMIT 1""",
                (list(kinds),),
            ).fetchone()
            if not row:
                return None
            conn.execute(
                """UPDATE jobs SET status='running',attempts=attempts+1,lease_owner=%s,
                   lease_until=now()+(%s||' seconds')::interval,updated_at=now() WHERE job_id=%s""",
                (worker_id, lease_seconds, row[0]),
            )
            conn.execute(
                "INSERT INTO job_events(job_id,type,payload_json) VALUES(%s,'claimed',%s::jsonb)",
                (
                    row[0],
                    json.dumps({"workerId": worker_id, "leaseSeconds": lease_seconds}),
                ),
            )
        return self.get(str(row[0]))

    def heartbeat(
        self,
        job_id: str,
        worker_id: str,
        *,
        lease_seconds: int = 60,
        fencing_token: int | None = None,
    ) -> bool:
        where = "job_id=%s AND lease_owner=%s AND status='running'"
        values: list[Any] = [lease_seconds, job_id, worker_id]
        if fencing_token is not None:
            where += " AND attempts=%s"
            values.append(int(fencing_token))
        with self.pool.connection() as conn:
            result = conn.execute(
                """UPDATE jobs SET lease_until=now()+(%s||' seconds')::interval,updated_at=now()
                   WHERE """
                + where,
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
        assignments = ["updated_at=now()"]
        values: list[Any] = []
        if status is not None:
            assignments.append("status=%s")
            values.append(status)
        if progress is not None:
            assignments.append("progress=%s")
            values.append(max(0.0, min(1.0, float(progress))))
        if result is not None:
            assignments.append("result_json=%s::jsonb")
            values.append(json.dumps(result))
        if error is not None:
            assignments.append("error=%s")
            values.append(error)
        elif status == "completed":
            assignments.append("error=NULL")
        if status in {"queued", "completed", "failed", "cancelled", "exhausted"}:
            assignments.extend(("lease_owner=NULL", "lease_until=NULL"))
        where = "job_id=%s"
        values.append(job_id)
        if worker_id is not None:
            where += " AND lease_owner=%s"
            values.append(worker_id)
        if fencing_token is not None:
            where += " AND attempts=%s"
            values.append(int(fencing_token))
        with self.pool.connection() as conn, conn.transaction():
            updated = conn.execute(
                f"UPDATE jobs SET {','.join(assignments)} WHERE {where}", values
            )
            if updated.rowcount != 1:
                return False
            event = status or ("progress" if progress is not None else "updated")
            conn.execute(
                "INSERT INTO job_events(job_id,type,payload_json) VALUES(%s,%s,%s::jsonb)",
                (job_id, event, json.dumps({"progress": progress})),
            )
            return True

    def request_cancel(self, job_id: str) -> bool:
        with self.pool.connection() as conn, conn.transaction():
            result = conn.execute(
                """UPDATE jobs SET cancel_requested=TRUE,
                   status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END,updated_at=now()
                   WHERE job_id=%s AND status NOT IN ('completed','failed','cancelled','exhausted')""",
                (job_id,),
            )
            if result.rowcount:
                conn.execute(
                    "INSERT INTO job_events(job_id,type,payload_json) VALUES(%s,'cancel_requested','{}'::jsonb)",
                    (job_id,),
                )
            return result.rowcount == 1

    def cancelled(self, job_id: str) -> bool:
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT cancel_requested,status FROM jobs WHERE job_id=%s", (job_id,)
            ).fetchone()
        return bool(row and (row[0] or row[1] == "cancelled"))

    def get(self, job_id: str) -> dict[str, Any] | None:
        tenant = CURRENT_DATABASE_IDENTITY.get().tenant_id
        tenant_filter = " AND tenant_id=%s" if tenant else ""
        values = (job_id, tenant) if tenant else (job_id,)
        with self.pool.connection() as conn:
            row = conn.execute(
                """SELECT job_id,tenant_id,kind,status,priority,progress,request_json,result_json,error,
                   attempts,max_attempts,lease_owner,lease_until,cancel_requested,created_at,updated_at
                   FROM jobs WHERE job_id=%s""" + tenant_filter,
                values,
            ).fetchone()
        if not row:
            return None
        return {
            "jobId": row[0],
            "tenantId": row[1],
            "kind": row[2],
            "status": row[3],
            "priority": row[4],
            "progress": float(row[5]),
            "request": row[6],
            "result": row[7],
            "error": row[8],
            "attempts": row[9],
            "maxAttempts": row[10],
            "leaseOwner": row[11],
            "leaseUntil": float(row[12].timestamp()) if row[12] else None,
            "cancelRequested": bool(row[13]),
            "createdAt": row[14].isoformat(),
            "updatedAt": row[15].isoformat(),
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
            clauses.append("tenant_id=%s")
            values.append(tenant)
        if kinds:
            clauses.append("kind = ANY(%s)")
            values.append(list(kinds))
        if status:
            clauses.append("status=%s")
            values.append(str(status))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.pool.connection() as conn:
            rows = conn.execute(
                f"SELECT job_id FROM jobs{where} ORDER BY created_at DESC,job_id DESC LIMIT %s",
                (*values, size),
            ).fetchall()
        return [item for row in rows if (item := self.get(str(row[0]))) is not None]

    def events(self, job_id: str) -> list[dict[str, Any]]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT type,payload_json,created_at FROM job_events WHERE job_id=%s ORDER BY id",
                (job_id,),
            ).fetchall()
        return [
            {"type": row[0], "payload": row[1], "createdAt": row[2].isoformat()}
            for row in rows
        ]


class PostgresGameStore(_PostgresBase):
    """Transactional trusted game snapshots for horizontally scaled API nodes."""

    @staticmethod
    def _encode(session: Any) -> tuple[bytes, str]:
        payload = encode_payload(session)
        return payload, hashlib.sha256(payload).hexdigest()

    def save(self, session: Any) -> str:
        payload, digest = self._encode(session)
        with self.pool.connection() as conn, conn.transaction():
            conn.execute(
                """INSERT INTO game_snapshots(game_id,version,payload,state_hash,event_count)
                   VALUES(%s,%s,%s,%s,%s)
                   ON CONFLICT(game_id) DO UPDATE SET version=excluded.version,payload=excluded.payload,
                   state_hash=excluded.state_hash,event_count=excluded.event_count,updated_at=now()""",
                (
                    session.id,
                    session.version,
                    payload,
                    digest,
                    len(session.state.events),
                ),
            )
            conn.execute(
                "INSERT INTO game_audit(game_id,version,state_hash,event_count) VALUES(%s,%s,%s,%s)",
                (session.id, session.version, digest, len(session.state.events)),
            )
        return digest

    def load(self, game_id: str) -> Any | None:
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT payload,state_hash FROM game_snapshots WHERE game_id=%s",
                (game_id,),
            ).fetchone()
        if row is None:
            return None
        payload = bytes(row[0])
        if hashlib.sha256(payload).hexdigest() != row[1]:
            raise RuntimeError("stored game snapshot hash mismatch")
        return decode_payload(payload)

    def delete(self, game_id: str) -> None:
        with self.pool.connection() as conn:
            conn.execute("DELETE FROM game_snapshots WHERE game_id=%s", (game_id,))

    def audit(self, game_id: str) -> list[dict[str, Any]]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT version,state_hash,event_count,created_at FROM game_audit WHERE game_id=%s ORDER BY id",
                (game_id,),
            ).fetchall()
        return [
            {
                "version": row[0],
                "stateHash": row[1],
                "eventCount": row[2],
                "createdAt": row[3].isoformat(),
            }
            for row in rows
        ]
