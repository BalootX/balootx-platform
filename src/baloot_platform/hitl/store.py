from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .models import (
    AdjudicationRecord, BlindAssignment, ChallengerEvaluation, ConsensusSnapshot,
    ConsentRecord, GovernedReviewCase, ImprovementEvaluation, ImprovementProposal,
    ModerationFlag, ReviewSubmission, ReviewerProfile, TrainingCandidate, ValidationResult,
    canonical, digest, utc_now,
)


def _loads(value: str | None, default: Any = None) -> Any:
    return default if value is None else json.loads(value)


class SQLiteReviewGovernanceStore:
    """Local append-only governance store with hash-chained audit events."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as conn:
            self._migrate(conn)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS hitl_cases(
                case_id TEXT PRIMARY KEY,
                decision_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_hitl_cases_decision
                ON hitl_cases(decision_id, created_at DESC);

            CREATE TABLE IF NOT EXISTS hitl_assignments(
                assignment_id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                reviewer_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(case_id, reviewer_id)
            );
            CREATE TABLE IF NOT EXISTS hitl_submissions(
                submission_id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                reviewer_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(case_id, reviewer_id)
            );
            CREATE TABLE IF NOT EXISTS hitl_validations(
                validation_id TEXT PRIMARY KEY,
                submission_id TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS hitl_consensus(
                consensus_id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS hitl_reviewer_profiles(
                subject_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS hitl_consent(
                subject_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS hitl_moderation_flags(
                flag_id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS hitl_adjudications(
                adjudication_id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS hitl_training_candidates(
                candidate_id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS hitl_challenger_evaluations(
                evaluation_id TEXT PRIMARY KEY,
                candidate_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS hitl_improvement_proposals(
                proposal_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS hitl_improvement_evaluations(
                evaluation_id TEXT PRIMARY KEY,
                proposal_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS hitl_events(
                event_id TEXT PRIMARY KEY,
                aggregate_type TEXT NOT NULL,
                aggregate_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                previous_hash TEXT NOT NULL,
                event_hash TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                UNIQUE(aggregate_type, aggregate_id, sequence)
            );
            CREATE INDEX IF NOT EXISTS idx_hitl_events_aggregate
                ON hitl_events(aggregate_type, aggregate_id, sequence);
            """
        )

    def _upsert(
        self,
        table: str,
        key_col: str,
        key: str,
        payload: dict[str, Any],
        *,
        time_col: str = "created_at",
    ) -> dict[str, Any]:
        now = (
            payload.get("createdAt")
            or payload.get("submittedAt")
            or payload.get("updatedAt")
            or utc_now()
        )
        with self._lock, self._connect() as conn:
            conn.execute(
                f"INSERT INTO {table}({key_col},payload_json,{time_col}) VALUES(?,?,?) "
                f"ON CONFLICT({key_col}) DO UPDATE SET payload_json=excluded.payload_json,{time_col}=excluded.{time_col}",
                (key, canonical(payload), now),
            )
        return payload

    def append_event(
        self,
        aggregate_type: str,
        aggregate_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT sequence,event_hash FROM hitl_events "
                "WHERE aggregate_type=? AND aggregate_id=? "
                "ORDER BY sequence DESC LIMIT 1",
                (aggregate_type, aggregate_id),
            ).fetchone()
            sequence = 0 if row is None else int(row["sequence"]) + 1
            previous = "" if row is None else str(row["event_hash"])
            created = utc_now()
            identity = {
                "aggregateType": aggregate_type,
                "aggregateId": aggregate_id,
                "sequence": sequence,
                "eventType": event_type,
                "payload": payload,
                "previousHash": previous,
                "createdAt": created,
            }
            event_hash = digest(identity)
            event_id = f"HEV-{event_hash[:20].upper()}"
            conn.execute(
                "INSERT INTO hitl_events VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    event_id,
                    aggregate_type,
                    aggregate_id,
                    sequence,
                    event_type,
                    canonical(payload),
                    previous,
                    event_hash,
                    created,
                ),
            )
        return {**identity, "eventId": event_id, "eventHash": event_hash}

    def list_events(self, aggregate_type: str, aggregate_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM hitl_events WHERE aggregate_type=? AND aggregate_id=? ORDER BY sequence",
                (aggregate_type, aggregate_id),
            ).fetchall()
        return [
            {
                "eventId": row["event_id"],
                "aggregateType": row["aggregate_type"],
                "aggregateId": row["aggregate_id"],
                "sequence": row["sequence"],
                "eventType": row["event_type"],
                "payload": _loads(row["payload_json"], {}),
                "previousHash": row["previous_hash"],
                "eventHash": row["event_hash"],
                "createdAt": row["created_at"],
            }
            for row in rows
        ]

    def verify_events(self, aggregate_type: str, aggregate_id: str) -> dict[str, Any]:
        previous = ""
        events = self.list_events(aggregate_type, aggregate_id)
        for event in events:
            identity = {
                key: event[key]
                for key in (
                    "aggregateType",
                    "aggregateId",
                    "sequence",
                    "eventType",
                    "payload",
                    "previousHash",
                    "createdAt",
                )
            }
            if event["previousHash"] != previous or digest(identity) != event["eventHash"]:
                return {"valid": False, "sequence": event["sequence"], "events": len(events)}
            previous = event["eventHash"]
        return {"valid": True, "events": len(events), "headHash": previous}

    def put_case(self, case: GovernedReviewCase) -> dict[str, Any]:
        payload = case.to_dict()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO hitl_cases(case_id,decision_id,payload_json,created_at) VALUES(?,?,?,?)",
                (case.case_id, case.decision_id, canonical(payload), case.created_at),
            )
        self.append_event("case", case.case_id, "case_created", payload)
        return payload

    def get_case(self, case_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM hitl_cases WHERE case_id=?", (case_id,)
            ).fetchone()
        return None if row is None else _loads(row[0], {})

    def list_cases(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM hitl_cases ORDER BY created_at DESC LIMIT ?",
                (max(1, min(1000, int(limit))),),
            ).fetchall()
        return [_loads(row[0], {}) for row in rows]

    def update_case(self, case_id: str, **changes: Any) -> dict[str, Any]:
        payload = self.get_case(case_id)
        if payload is None:
            raise KeyError(case_id)
        payload.update(changes)
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE hitl_cases SET payload_json=? WHERE case_id=?",
                (canonical(payload), case_id),
            )
        self.append_event("case", case_id, "case_updated", changes)
        return payload

    def put_assignment(self, assignment: BlindAssignment) -> dict[str, Any]:
        payload = assignment.to_dict()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO hitl_assignments(assignment_id,case_id,reviewer_id,payload_json,created_at) VALUES(?,?,?,?,?)",
                (
                    assignment.assignment_id,
                    assignment.case_id,
                    assignment.reviewer_id,
                    canonical(payload),
                    assignment.assigned_at,
                ),
            )
        self.append_event("case", assignment.case_id, "assignment_created", payload)
        return payload

    def get_assignment(self, assignment_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM hitl_assignments WHERE assignment_id=?",
                (assignment_id,),
            ).fetchone()
        return None if row is None else _loads(row[0], {})

    def find_assignment(self, case_id: str, reviewer_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM hitl_assignments WHERE case_id=? AND reviewer_id=?",
                (case_id, reviewer_id),
            ).fetchone()
        return None if row is None else _loads(row[0], {})

    def update_assignment(self, assignment_id: str, **changes: Any) -> dict[str, Any]:
        payload = self.get_assignment(assignment_id)
        if payload is None:
            raise KeyError(assignment_id)
        payload.update(changes)
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE hitl_assignments SET payload_json=? WHERE assignment_id=?",
                (canonical(payload), assignment_id),
            )
        return payload

    def put_submission(self, submission: ReviewSubmission) -> dict[str, Any]:
        payload = submission.to_dict()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO hitl_submissions(submission_id,case_id,reviewer_id,payload_json,created_at) VALUES(?,?,?,?,?)",
                (
                    submission.submission_id,
                    submission.case_id,
                    submission.reviewer_id,
                    canonical(payload),
                    submission.submitted_at,
                ),
            )
        self.append_event("case", submission.case_id, "review_submitted", payload)
        return payload

    def get_submission(self, submission_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM hitl_submissions WHERE submission_id=?",
                (submission_id,),
            ).fetchone()
        return None if row is None else _loads(row[0], {})

    def list_submissions(self, case_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM hitl_submissions WHERE case_id=? ORDER BY created_at",
                (case_id,),
            ).fetchall()
        return [_loads(row[0], {}) for row in rows]

    def update_submission(self, submission_id: str, **changes: Any) -> dict[str, Any]:
        payload = self.get_submission(submission_id)
        if payload is None:
            raise KeyError(submission_id)
        payload.update(changes)
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE hitl_submissions SET payload_json=? WHERE submission_id=?",
                (canonical(payload), submission_id),
            )
        return payload

    def put_validation(self, validation: ValidationResult) -> dict[str, Any]:
        payload = validation.to_dict()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO hitl_validations(validation_id,submission_id,payload_json,created_at) VALUES(?,?,?,?)",
                (
                    validation.validation_id,
                    validation.submission_id,
                    canonical(payload),
                    validation.validated_at,
                ),
            )
        return payload

    def get_validation_for_submission(self, submission_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM hitl_validations WHERE submission_id=?",
                (submission_id,),
            ).fetchone()
        return None if row is None else _loads(row[0], {})

    def put_consensus(self, consensus: ConsensusSnapshot) -> dict[str, Any]:
        payload = consensus.to_dict()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO hitl_consensus(consensus_id,case_id,payload_json,created_at) VALUES(?,?,?,?)",
                (consensus.consensus_id, consensus.case_id, canonical(payload), consensus.created_at),
            )
        self.append_event("case", consensus.case_id, "consensus_computed", payload)
        return payload

    def latest_consensus(self, case_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM hitl_consensus WHERE case_id=? ORDER BY created_at DESC, rowid DESC LIMIT 1",
                (case_id,),
            ).fetchone()
        return None if row is None else _loads(row[0], {})

    def put_profile(self, profile: ReviewerProfile) -> dict[str, Any]:
        return self._upsert(
            "hitl_reviewer_profiles",
            "subject_id",
            profile.subject_id,
            profile.to_dict(),
            time_col="updated_at",
        )

    def get_profile(self, subject_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM hitl_reviewer_profiles WHERE subject_id=?",
                (subject_id,),
            ).fetchone()
        return None if row is None else _loads(row[0], {})

    def put_consent(self, consent: ConsentRecord) -> dict[str, Any]:
        return self._upsert(
            "hitl_consent",
            "subject_id",
            consent.subject_id,
            consent.to_dict(),
            time_col="updated_at",
        )

    def get_consent(self, subject_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM hitl_consent WHERE subject_id=?", (subject_id,)
            ).fetchone()
        return None if row is None else _loads(row[0], {})

    def put_flag(self, flag: ModerationFlag) -> dict[str, Any]:
        payload = flag.to_dict()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO hitl_moderation_flags(flag_id,case_id,payload_json,created_at) VALUES(?,?,?,?)",
                (flag.flag_id, flag.case_id, canonical(payload), flag.created_at),
            )
        self.append_event("case", flag.case_id, "moderation_flag_opened", payload)
        return payload

    def list_flags(self, case_id: str, *, open_only: bool = False) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM hitl_moderation_flags WHERE case_id=? ORDER BY created_at",
                (case_id,),
            ).fetchall()
        values = [_loads(row[0], {}) for row in rows]
        return [value for value in values if not open_only or value.get("status") == "open"]

    def update_flag(self, flag_id: str, **changes: Any) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM hitl_moderation_flags WHERE flag_id=?", (flag_id,)
            ).fetchone()
        if row is None:
            raise KeyError(flag_id)
        payload = _loads(row[0], {})
        payload.update(changes)
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE hitl_moderation_flags SET payload_json=? WHERE flag_id=?",
                (canonical(payload), flag_id),
            )
        return payload

    def put_adjudication(self, adjudication: AdjudicationRecord) -> dict[str, Any]:
        payload = adjudication.to_dict()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO hitl_adjudications(adjudication_id,case_id,payload_json,created_at) VALUES(?,?,?,?)",
                (adjudication.adjudication_id, adjudication.case_id, canonical(payload), adjudication.created_at),
            )
        self.append_event("case", adjudication.case_id, "adjudicated", payload)
        return payload

    def latest_adjudication(self, case_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM hitl_adjudications WHERE case_id=? ORDER BY created_at DESC LIMIT 1",
                (case_id,),
            ).fetchone()
        return None if row is None else _loads(row[0], {})

    def put_candidate(self, candidate: TrainingCandidate) -> dict[str, Any]:
        payload = candidate.to_dict()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO hitl_training_candidates(candidate_id,case_id,payload_json,created_at) VALUES(?,?,?,?)",
                (candidate.candidate_id, candidate.case_id, canonical(payload), candidate.created_at),
            )
        self.append_event("case", candidate.case_id, "training_candidate_created", payload)
        return payload

    def get_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM hitl_training_candidates WHERE candidate_id=?",
                (candidate_id,),
            ).fetchone()
        return None if row is None else _loads(row[0], {})

    def list_candidates(self, status: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM hitl_training_candidates ORDER BY created_at DESC"
            ).fetchall()
        values = [_loads(row[0], {}) for row in rows]
        return [value for value in values if status is None or value.get("status") == status]

    def update_candidate(self, candidate_id: str, **changes: Any) -> dict[str, Any]:
        payload = self.get_candidate(candidate_id)
        if payload is None:
            raise KeyError(candidate_id)
        payload.update(changes)
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE hitl_training_candidates SET payload_json=? WHERE candidate_id=?",
                (canonical(payload), candidate_id),
            )
        return payload

    def put_challenger_evaluation(
        self, evaluation: ChallengerEvaluation
    ) -> dict[str, Any]:
        payload = evaluation.to_dict()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO hitl_challenger_evaluations(evaluation_id,candidate_id,payload_json,created_at) VALUES(?,?,?,?)",
                (evaluation.evaluation_id, evaluation.candidate_id, canonical(payload), evaluation.created_at),
            )
        return payload

    def list_challenger_evaluations(self, candidate_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM hitl_challenger_evaluations WHERE candidate_id=? ORDER BY created_at",
                (candidate_id,),
            ).fetchall()
        return [_loads(row[0], {}) for row in rows]

    def put_proposal(self, proposal: ImprovementProposal) -> dict[str, Any]:
        payload = proposal.to_dict()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO hitl_improvement_proposals(proposal_id,payload_json,created_at) VALUES(?,?,?)",
                (proposal.proposal_id, canonical(payload), proposal.created_at),
            )
        self.append_event("proposal", proposal.proposal_id, "proposal_created", payload)
        return payload

    def get_proposal(self, proposal_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM hitl_improvement_proposals WHERE proposal_id=?",
                (proposal_id,),
            ).fetchone()
        return None if row is None else _loads(row[0], {})

    def list_proposals(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM hitl_improvement_proposals ORDER BY created_at DESC"
            ).fetchall()
        return [_loads(row[0], {}) for row in rows]

    def update_proposal(self, proposal_id: str, **changes: Any) -> dict[str, Any]:
        payload = self.get_proposal(proposal_id)
        if payload is None:
            raise KeyError(proposal_id)
        payload.update(changes)
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE hitl_improvement_proposals SET payload_json=? WHERE proposal_id=?",
                (canonical(payload), proposal_id),
            )
        self.append_event("proposal", proposal_id, "proposal_updated", changes)
        return payload

    def put_improvement_evaluation(
        self, evaluation: ImprovementEvaluation
    ) -> dict[str, Any]:
        payload = evaluation.to_dict()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO hitl_improvement_evaluations(evaluation_id,proposal_id,payload_json,created_at) VALUES(?,?,?,?)",
                (evaluation.evaluation_id, evaluation.proposal_id, canonical(payload), evaluation.created_at),
            )
        self.append_event("proposal", evaluation.proposal_id, "proposal_evaluated", payload)
        return payload

    def list_improvement_evaluations(self, proposal_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM hitl_improvement_evaluations WHERE proposal_id=? ORDER BY created_at",
                (proposal_id,),
            ).fetchall()
        return [_loads(row[0], {}) for row in rows]

    def close(self) -> None:
        return None


class PostgresReviewGovernanceStore:
    """PostgreSQL implementation for governed HITL state and event lineage.

    The store mirrors :class:`SQLiteReviewGovernanceStore` while using the
    normalized JSONB tables created by migration ``010_hitl_governance_v18``.
    Connections are intentionally short lived so the class works with normal
    psycopg connections and external pool/proxy deployments alike.
    """

    def __init__(self, dsn: str):
        try:
            import psycopg  # type: ignore
            from psycopg.rows import dict_row  # type: ignore
            from psycopg.types.json import Jsonb  # type: ignore
        except ImportError as exc:
            raise RuntimeError("install baloot-platform[postgres]") from exc
        self.psycopg = psycopg
        self.dict_row = dict_row
        self.Jsonb = Jsonb
        self.dsn = dsn
        # Fail early instead of exposing a partially configured production
        # backend. Migrations are still owned by the migration runner.
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT to_regclass('hitl.review_cases') AS table_name")
                row = cur.fetchone()
                if not row or row["table_name"] is None:
                    raise RuntimeError("HITL PostgreSQL schema is not migrated; run migration 010")

    def _connect(self):
        return self.psycopg.connect(self.dsn, row_factory=self.dict_row)

    @staticmethod
    def _payload_value(row: Any) -> dict[str, Any] | None:
        if row is None:
            return None
        value = row.get("payload") if isinstance(row, dict) else row[0]
        if isinstance(value, str):
            return json.loads(value)
        return dict(value or {})

    def _put_payload(
        self,
        table: str,
        key_col: str,
        key: str,
        payload: dict[str, Any],
        *,
        insert_columns: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        columns = {key_col: key, "payload": self.Jsonb(payload)}
        columns.update(insert_columns or {})
        names = list(columns)
        values = [columns[name] for name in names]
        placeholders = ",".join(["%s"] * len(names))
        updates = "payload=EXCLUDED.payload"
        if "status" in columns:
            updates += ",status=EXCLUDED.status"
        if "updated_at" in columns:
            updates += ",updated_at=EXCLUDED.updated_at"
        query = (
            f"INSERT INTO {table}({','.join(names)}) VALUES({placeholders}) "
            f"ON CONFLICT({key_col}) DO UPDATE SET {updates}"
        )
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, values)
            conn.commit()
        return payload

    def _get_payload(self, table: str, key_col: str, key: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT payload FROM {table} WHERE {key_col}=%s", (key,))
                return self._payload_value(cur.fetchone())

    def _list_payloads(
        self,
        table: str,
        *,
        where: str = "",
        params: tuple[Any, ...] = (),
        order: str = "created_at ASC",
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        query = f"SELECT payload FROM {table}"
        if where:
            query += " WHERE " + where
        query += " ORDER BY " + order
        values: list[Any] = list(params)
        if limit is not None:
            query += " LIMIT %s"
            values.append(limit)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, values)
                return [self._payload_value(row) or {} for row in cur.fetchall()]

    def _update_payload(
        self,
        table: str,
        key_col: str,
        key: str,
        **changes: Any,
    ) -> dict[str, Any]:
        payload = self._get_payload(table, key_col, key)
        if payload is None:
            raise KeyError(key)
        payload.update(changes)
        assignments = ["payload=%s"]
        values: list[Any] = [self.Jsonb(payload)]
        if "status" in changes:
            assignments.append("status=%s")
            values.append(changes["status"])
        if table in {"hitl.review_cases", "hitl.reviewer_profiles", "hitl.consent"}:
            assignments.append("updated_at=now()")
        values.append(key)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE {table} SET {','.join(assignments)} WHERE {key_col}=%s",
                    values,
                )
            conn.commit()
        return payload

    def append_event(
        self,
        aggregate_type: str,
        aggregate_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                # Advisory transaction lock serializes event heads without
                # requiring an aggregate-head table.
                cur.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (f"{aggregate_type}:{aggregate_id}",),
                )
                cur.execute(
                    "SELECT sequence,event_hash FROM hitl.events "
                    "WHERE aggregate_type=%s AND aggregate_id=%s "
                    "ORDER BY sequence DESC LIMIT 1",
                    (aggregate_type, aggregate_id),
                )
                row = cur.fetchone()
                sequence = 0 if row is None else int(row["sequence"]) + 1
                previous = "" if row is None else str(row["event_hash"])
                created = utc_now()
                identity = {
                    "aggregateType": aggregate_type,
                    "aggregateId": aggregate_id,
                    "sequence": sequence,
                    "eventType": event_type,
                    "payload": payload,
                    "previousHash": previous,
                    "createdAt": created,
                }
                event_hash = digest(identity)
                event_id = f"HEV-{event_hash[:20].upper()}"
                cur.execute(
                    "INSERT INTO hitl.events(aggregate_type,aggregate_id,sequence,event_type,payload,previous_hash,event_hash,occurred_at) "
                    "VALUES(%s,%s,%s,%s,%s,%s,%s,%s)",
                    (
                        aggregate_type,
                        aggregate_id,
                        sequence,
                        event_type,
                        self.Jsonb(payload),
                        previous,
                        event_hash,
                        created,
                    ),
                )
            conn.commit()
        return {**identity, "eventId": event_id, "eventHash": event_hash}

    def list_events(self, aggregate_type: str, aggregate_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT event_id,aggregate_type,aggregate_id,sequence,event_type,payload,previous_hash,event_hash,occurred_at "
                    "FROM hitl.events WHERE aggregate_type=%s AND aggregate_id=%s ORDER BY sequence",
                    (aggregate_type, aggregate_id),
                )
                rows = cur.fetchall()
        return [
            {
                "eventId": f"HEV-{str(row['event_hash'])[:20].upper()}",
                "aggregateType": row["aggregate_type"],
                "aggregateId": row["aggregate_id"],
                "sequence": row["sequence"],
                "eventType": row["event_type"],
                "payload": dict(row["payload"] or {}),
                "previousHash": row["previous_hash"],
                "eventHash": row["event_hash"],
                "createdAt": row["occurred_at"].isoformat() if hasattr(row["occurred_at"], "isoformat") else str(row["occurred_at"]),
            }
            for row in rows
        ]

    def verify_events(self, aggregate_type: str, aggregate_id: str) -> dict[str, Any]:
        previous = ""
        events = self.list_events(aggregate_type, aggregate_id)
        for event in events:
            identity = {
                key: event[key]
                for key in (
                    "aggregateType",
                    "aggregateId",
                    "sequence",
                    "eventType",
                    "payload",
                    "previousHash",
                    "createdAt",
                )
            }
            if event["previousHash"] != previous or digest(identity) != event["eventHash"]:
                return {"valid": False, "sequence": event["sequence"], "events": len(events)}
            previous = event["eventHash"]
        return {"valid": True, "events": len(events), "headHash": previous}

    def put_case(self, case: GovernedReviewCase) -> dict[str, Any]:
        payload = case.to_dict()
        result = self._put_payload(
            "hitl.review_cases", "case_id", case.case_id, payload,
            insert_columns={
                "decision_id": case.decision_id,
                "created_by": case.created_by,
                "status": case.status,
                "case_hash": case.case_hash,
                "created_at": case.created_at,
                "updated_at": case.created_at,
            },
        )
        self.append_event("case", case.case_id, "case_created", payload)
        return result

    def get_case(self, case_id: str) -> dict[str, Any] | None:
        return self._get_payload("hitl.review_cases", "case_id", case_id)

    def list_cases(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._list_payloads(
            "hitl.review_cases", order="created_at DESC", limit=max(1, min(1000, int(limit)))
        )

    def update_case(self, case_id: str, **changes: Any) -> dict[str, Any]:
        result = self._update_payload("hitl.review_cases", "case_id", case_id, **changes)
        self.append_event("case", case_id, "case_updated", changes)
        return result

    def put_assignment(self, assignment: BlindAssignment) -> dict[str, Any]:
        payload = assignment.to_dict()
        result = self._put_payload(
            "hitl.blind_assignments", "assignment_id", assignment.assignment_id, payload,
            insert_columns={
                "case_id": assignment.case_id,
                "reviewer_id": assignment.reviewer_id,
                "token_hash": assignment.token_hash,
                "status": assignment.status,
                "device_cluster_hash": assignment.device_cluster_hash,
                "network_cluster_hash": assignment.network_cluster_hash,
                "created_at": assignment.assigned_at,
            },
        )
        self.append_event("case", assignment.case_id, "assignment_created", payload)
        return result

    def get_assignment(self, assignment_id: str) -> dict[str, Any] | None:
        return self._get_payload("hitl.blind_assignments", "assignment_id", assignment_id)

    def find_assignment(self, case_id: str, reviewer_id: str) -> dict[str, Any] | None:
        values = self._list_payloads(
            "hitl.blind_assignments",
            where="case_id=%s AND reviewer_id=%s",
            params=(case_id, reviewer_id),
            order="created_at DESC",
            limit=1,
        )
        return values[0] if values else None

    def update_assignment(self, assignment_id: str, **changes: Any) -> dict[str, Any]:
        return self._update_payload("hitl.blind_assignments", "assignment_id", assignment_id, **changes)

    def put_submission(self, submission: ReviewSubmission) -> dict[str, Any]:
        payload = submission.to_dict()
        result = self._put_payload(
            "hitl.review_submissions", "submission_id", submission.submission_id, payload,
            insert_columns={
                "case_id": submission.case_id,
                "assignment_id": submission.assignment_id,
                "reviewer_id": submission.reviewer_id,
                "preferred_action": self.Jsonb(submission.preferred_action),
                "status": submission.status,
                "submission_hash": submission.submission_hash,
                "created_at": submission.submitted_at,
            },
        )
        self.append_event("case", submission.case_id, "review_submitted", payload)
        return result

    def get_submission(self, submission_id: str) -> dict[str, Any] | None:
        return self._get_payload("hitl.review_submissions", "submission_id", submission_id)

    def list_submissions(self, case_id: str) -> list[dict[str, Any]]:
        return self._list_payloads("hitl.review_submissions", where="case_id=%s", params=(case_id,))

    def update_submission(self, submission_id: str, **changes: Any) -> dict[str, Any]:
        return self._update_payload("hitl.review_submissions", "submission_id", submission_id, **changes)

    def put_validation(self, validation: ValidationResult) -> dict[str, Any]:
        payload = validation.to_dict()
        return self._put_payload(
            "hitl.validation_results", "validation_id", validation.validation_id, payload,
            insert_columns={"submission_id": validation.submission_id, "created_at": validation.validated_at},
        )

    def get_validation_for_submission(self, submission_id: str) -> dict[str, Any] | None:
        values = self._list_payloads(
            "hitl.validation_results", where="submission_id=%s", params=(submission_id,), limit=1
        )
        return values[0] if values else None

    def put_consensus(self, consensus: ConsensusSnapshot) -> dict[str, Any]:
        payload = consensus.to_dict()
        result = self._put_payload(
            "hitl.consensus_snapshots", "consensus_id", consensus.consensus_id, payload,
            insert_columns={
                "case_id": consensus.case_id,
                "status": "eligible" if consensus.eligible else "ineligible",
                "created_at": consensus.created_at,
            },
        )
        self.append_event("case", consensus.case_id, "consensus_computed", payload)
        return result

    def latest_consensus(self, case_id: str) -> dict[str, Any] | None:
        values = self._list_payloads(
            "hitl.consensus_snapshots", where="case_id=%s", params=(case_id,), order="created_at DESC, ((payload->>'reviewerCount')::int) DESC", limit=1
        )
        return values[0] if values else None

    def put_profile(self, profile: ReviewerProfile) -> dict[str, Any]:
        return self._put_payload(
            "hitl.reviewer_profiles", "subject_id", profile.subject_id, profile.to_dict(),
            insert_columns={"updated_at": profile.updated_at},
        )

    def get_profile(self, subject_id: str) -> dict[str, Any] | None:
        return self._get_payload("hitl.reviewer_profiles", "subject_id", subject_id)

    def put_consent(self, consent: ConsentRecord) -> dict[str, Any]:
        return self._put_payload(
            "hitl.consent", "subject_id", consent.subject_id, consent.to_dict(),
            insert_columns={"updated_at": utc_now()},
        )

    def get_consent(self, subject_id: str) -> dict[str, Any] | None:
        return self._get_payload("hitl.consent", "subject_id", subject_id)

    def put_flag(self, flag: ModerationFlag) -> dict[str, Any]:
        payload = flag.to_dict()
        result = self._put_payload(
            "hitl.moderation_flags", "flag_id", flag.flag_id, payload,
            insert_columns={"case_id": flag.case_id, "status": flag.status, "created_at": flag.created_at},
        )
        self.append_event("case", flag.case_id, "moderation_flag_opened", payload)
        return result

    def list_flags(self, case_id: str, *, open_only: bool = False) -> list[dict[str, Any]]:
        where = "case_id=%s" + (" AND status='open'" if open_only else "")
        return self._list_payloads("hitl.moderation_flags", where=where, params=(case_id,))

    def update_flag(self, flag_id: str, **changes: Any) -> dict[str, Any]:
        return self._update_payload("hitl.moderation_flags", "flag_id", flag_id, **changes)

    def put_adjudication(self, adjudication: AdjudicationRecord) -> dict[str, Any]:
        payload = adjudication.to_dict()
        result = self._put_payload(
            "hitl.adjudications", "adjudication_id", adjudication.adjudication_id, payload,
            insert_columns={"case_id": adjudication.case_id, "created_at": adjudication.created_at},
        )
        self.append_event("case", adjudication.case_id, "adjudicated", payload)
        return result

    def latest_adjudication(self, case_id: str) -> dict[str, Any] | None:
        values = self._list_payloads(
            "hitl.adjudications", where="case_id=%s", params=(case_id,), order="created_at DESC", limit=1
        )
        return values[0] if values else None

    def put_candidate(self, candidate: TrainingCandidate) -> dict[str, Any]:
        payload = candidate.to_dict()
        result = self._put_payload(
            "hitl.training_candidates", "candidate_id", candidate.candidate_id, payload,
            insert_columns={
                "case_id": candidate.case_id,
                "status": candidate.status,
                "created_at": candidate.created_at,
            },
        )
        self.append_event("case", candidate.case_id, "training_candidate_created", payload)
        return result

    def get_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        return self._get_payload("hitl.training_candidates", "candidate_id", candidate_id)

    def list_candidates(self, status: str | None = None) -> list[dict[str, Any]]:
        return self._list_payloads(
            "hitl.training_candidates",
            where="status=%s" if status else "",
            params=(status,) if status else (),
            order="created_at DESC",
        )

    def update_candidate(self, candidate_id: str, **changes: Any) -> dict[str, Any]:
        return self._update_payload("hitl.training_candidates", "candidate_id", candidate_id, **changes)

    def put_challenger_evaluation(self, evaluation: ChallengerEvaluation) -> dict[str, Any]:
        payload = evaluation.to_dict()
        return self._put_payload(
            "hitl.challenger_evaluations", "evaluation_id", evaluation.evaluation_id, payload,
            insert_columns={"candidate_id": evaluation.candidate_id, "created_at": evaluation.created_at},
        )

    def list_challenger_evaluations(self, candidate_id: str) -> list[dict[str, Any]]:
        return self._list_payloads(
            "hitl.challenger_evaluations", where="candidate_id=%s", params=(candidate_id,)
        )

    def put_proposal(self, proposal: ImprovementProposal) -> dict[str, Any]:
        payload = proposal.to_dict()
        result = self._put_payload(
            "hitl.improvement_proposals", "proposal_id", proposal.proposal_id, payload,
            insert_columns={
                "target": proposal.target,
                "status": proposal.status,
                "created_at": proposal.created_at,
            },
        )
        self.append_event("proposal", proposal.proposal_id, "proposal_created", payload)
        return result

    def get_proposal(self, proposal_id: str) -> dict[str, Any] | None:
        return self._get_payload("hitl.improvement_proposals", "proposal_id", proposal_id)

    def list_proposals(self) -> list[dict[str, Any]]:
        return self._list_payloads("hitl.improvement_proposals", order="created_at DESC")

    def update_proposal(self, proposal_id: str, **changes: Any) -> dict[str, Any]:
        result = self._update_payload("hitl.improvement_proposals", "proposal_id", proposal_id, **changes)
        self.append_event("proposal", proposal_id, "proposal_updated", changes)
        return result

    def put_improvement_evaluation(self, evaluation: ImprovementEvaluation) -> dict[str, Any]:
        payload = evaluation.to_dict()
        result = self._put_payload(
            "hitl.improvement_evaluations", "evaluation_id", evaluation.evaluation_id, payload,
            insert_columns={"proposal_id": evaluation.proposal_id, "created_at": evaluation.created_at},
        )
        self.append_event("proposal", evaluation.proposal_id, "proposal_evaluated", payload)
        return result

    def list_improvement_evaluations(self, proposal_id: str) -> list[dict[str, Any]]:
        return self._list_payloads(
            "hitl.improvement_evaluations", where="proposal_id=%s", params=(proposal_id,)
        )

    def close(self) -> None:
        return None
