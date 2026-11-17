from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .models import AdjudicationCase, CaseStatus, EvidenceItem, Ruling, RulingOutcome, ViolationCode


class SQLiteAdjudicationStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS adjudication_cases_v17(
                  case_id TEXT PRIMARY KEY,
                  game_id TEXT NOT NULL,
                  status TEXT NOT NULL,
                  violation TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  case_hash TEXT NOT NULL,
                  previous_hash TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_adjudication_cases_game
                  ON adjudication_cases_v17(game_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_adjudication_cases_status
                  ON adjudication_cases_v17(status, updated_at DESC);
                """
            )

    @staticmethod
    def _deserialize(data: dict[str, Any]) -> AdjudicationCase:
        evidence = [
            EvidenceItem(
                evidence_id=item["evidenceId"], kind=item["kind"], uri=item.get("uri"),
                description=item["description"], submitted_by=item["submittedBy"],
                created_at=item["createdAt"], sha256=item.get("sha256"), metadata=dict(item.get("metadata") or {}),
            )
            for item in data.get("evidence", [])
        ]
        rulings = [
            Ruling(
                ruling_id=item["rulingId"], outcome=RulingOutcome(item["outcome"]), referee_id=item["refereeId"],
                reason_ar=item["reasonAr"], created_at=item["createdAt"],
                points_delta=tuple(item.get("pointsDelta") or (0, 0)),
                replay_from_event_sequence=item.get("replayFromEventSequence"),
                profile_id=item.get("profileId", ""), profile_hash=item.get("profileHash", ""),
                final=bool(item.get("final", False)), metadata=dict(item.get("metadata") or {}),
            )
            for item in data.get("rulings", [])
        ]
        return AdjudicationCase(
            case_id=data["caseId"], game_id=data["gameId"], round_number=data.get("roundNumber"),
            event_sequence=data.get("eventSequence"), violation=ViolationCode(data["violation"]),
            reported_by=data["reportedBy"], accused_subject_id=data.get("accusedSubjectId"),
            accused_seat=data.get("accusedSeat"), description=data["description"],
            status=CaseStatus(data["status"]), created_at=data["createdAt"], updated_at=data["updatedAt"],
            evidence=evidence, rulings=rulings, appeal_reason=data.get("appealReason"),
            appealed_by=data.get("appealedBy"), appeal_created_at=data.get("appealCreatedAt"),
            previous_hash=data.get("previousHash", ""), case_hash=data.get("caseHash", ""),
            metadata=dict(data.get("metadata") or {}),
        )

    def save(self, case: AdjudicationCase) -> None:
        payload = case.to_dict()
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO adjudication_cases_v17(
                     case_id,game_id,status,violation,updated_at,payload_json,case_hash,previous_hash
                   ) VALUES(?,?,?,?,?,?,?,?)
                   ON CONFLICT(case_id) DO UPDATE SET
                     status=excluded.status,violation=excluded.violation,updated_at=excluded.updated_at,
                     payload_json=excluded.payload_json,case_hash=excluded.case_hash,previous_hash=excluded.previous_hash""",
                (
                    case.case_id, case.game_id, case.status.value, case.violation.value, case.updated_at,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                    case.case_hash, case.previous_hash,
                ),
            )

    def get(self, case_id: str) -> AdjudicationCase | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM adjudication_cases_v17 WHERE case_id=?", (case_id,)
            ).fetchone()
        if row is None:
            return None
        return self._deserialize(json.loads(row["payload_json"]))

    def list(self, *, game_id: str | None = None, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if game_id:
            clauses.append("game_id=?")
            params.append(game_id)
        if status:
            clauses.append("status=?")
            params.append(CaseStatus(status).value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(1000, int(limit))))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT payload_json FROM adjudication_cases_v17 {where} ORDER BY updated_at DESC LIMIT ?",
                params,
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def latest_hash(self, game_id: str) -> str:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT case_hash FROM adjudication_cases_v17 WHERE game_id=? ORDER BY updated_at DESC LIMIT 1",
                (game_id,),
            ).fetchone()
        return str(row["case_hash"]) if row else ""

    def close(self) -> None:
        return None


class PostgresAdjudicationStore:
    """PostgreSQL production store using the same service contract."""

    def __init__(self, dsn: str):
        try:
            from psycopg_pool import ConnectionPool
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError('Install with: pip install "baloot-platform[postgres]"') from exc
        self.pool = ConnectionPool(dsn, min_size=1, max_size=16, open=True)

    def save(self, case: AdjudicationCase) -> None:
        payload = case.to_dict()
        with self.pool.connection() as conn:
            conn.execute(
                """INSERT INTO adjudication.cases(case_id,game_id,status,violation,updated_at,payload_json,case_hash,previous_hash)
                   VALUES(%s,%s,%s,%s,%s::timestamptz,%s::jsonb,%s,%s)
                   ON CONFLICT(case_id) DO UPDATE SET status=excluded.status,violation=excluded.violation,
                     updated_at=excluded.updated_at,payload_json=excluded.payload_json,
                     case_hash=excluded.case_hash,previous_hash=excluded.previous_hash""",
                (case.case_id, case.game_id, case.status.value, case.violation.value, case.updated_at,
                 json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                 case.case_hash, case.previous_hash),
            )

    def get(self, case_id: str) -> AdjudicationCase | None:
        with self.pool.connection() as conn:
            row = conn.execute("SELECT payload_json FROM adjudication.cases WHERE case_id=%s", (case_id,)).fetchone()
        return SQLiteAdjudicationStore._deserialize(dict(row[0])) if row else None

    def list(self, *, game_id: str | None = None, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if game_id:
            clauses.append("game_id=%s"); params.append(game_id)
        if status:
            clauses.append("status=%s"); params.append(CaseStatus(status).value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(1000, int(limit))))
        with self.pool.connection() as conn:
            rows = conn.execute(
                f"SELECT payload_json FROM adjudication.cases {where} ORDER BY updated_at DESC LIMIT %s", params
            ).fetchall()
        return [dict(row[0]) for row in rows]

    def latest_hash(self, game_id: str) -> str:
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT case_hash FROM adjudication.cases WHERE game_id=%s ORDER BY updated_at DESC LIMIT 1", (game_id,)
            ).fetchone()
        return str(row[0]) if row else ""

    def close(self) -> None:
        self.pool.close()
