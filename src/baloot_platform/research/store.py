from __future__ import annotations

import copy
import hashlib
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from .audit_trace import ResearchAuditTraceRecordV1
from .contracts import effective_disagreement_category
from .domain_artifact import DomainCompilationArtifactV1
from .fenced_occurrence import (
    build_fenced_occurrence,
    build_sequential_search_receipt_bundle,
    normalize_lease_duration,
    normalize_lease_moment,
    normalize_fenced_attempt,
    strict_positive_integer,
)
from .recipe_api import (
    RecipeListCursorV1,
    ResearchMutationReceiptCorruptionError,
    replay_recipe_mutation_receipt,
    validate_idempotency_key,
    validate_request_intent_hash,
)
from .result_library import (
    RESULT_PROJECTION_VERSION,
    ResearchResultQueryV1,
    normalize_timestamp,
    result_facets,
    result_library_projection,
    result_library_safe_payload,
    stored_result_library_projection_hash_v1,
    stored_result_payload_hash_v1,
    validate_stored_result_identity_v1,
    validate_stored_result_library_projection_hash_v1,
    validate_stored_result_record_v1,
)
from .serialization import decode_json as _decode
from .serialization import encode_json as _encode

# Avoid decoding private proof receipts on every Foundry poll. Full reads stay default.
_OPERATOR_RUN_PAYLOAD_SQL = (
    "CASE WHEN json_type(payload_json, '$.performanceReceipt') = 'object' "
    "AND json_type(payload_json, '$.performanceReceipt.coordinatorReceipt') IS NOT NULL "
    "THEN json_remove(payload_json, '$.performanceReceipt') ELSE payload_json END"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _sqlite_utc_text(moment: datetime) -> str:
    return normalize_lease_moment(moment).isoformat(timespec="microseconds")


def _sqlite_lease_is_live(value: Any, *, moment: datetime) -> bool:
    if value is None:
        return False
    try:
        expiry = datetime.fromisoformat(str(value))
        return normalize_lease_moment(expiry) > normalize_lease_moment(moment)
    except (TypeError, ValueError):
        return False


_DATABASE_INTEGER_MAX = (1 << 63) - 1
_SEQUENTIAL_RECEIPT_PAGE_LIMIT_MAX = 1000
_SEQUENTIAL_RECEIPT_SNAPSHOT_BATCH_MAX = 1000


def _strict_nonnegative_database_integer(value: Any, *, field: str) -> int:
    if type(value) is not int or not 0 <= value <= _DATABASE_INTEGER_MAX:
        raise ValueError(f"{field} must be a non-negative database integer")
    return value


def _strict_receipt_text(value: Any, *, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


class SQLiteResearchStoreV1:
    """Local durable control plane; large evidence stays in object storage."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(
            self.path,
            check_same_thread=False,
            timeout=30.0,
        )
        self._db.row_factory = sqlite3.Row
        self._db.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;
            PRAGMA busy_timeout=30000;
            CREATE TABLE IF NOT EXISTS research_recipes(
              recipe_code TEXT PRIMARY KEY, recipe_hash TEXT NOT NULL UNIQUE,
              status TEXT NOT NULL, payload_json TEXT NOT NULL,
              created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS ix_research_recipes_created_page
              ON research_recipes(created_at DESC,recipe_code COLLATE BINARY ASC);
            CREATE INDEX IF NOT EXISTS ix_research_recipes_status_page
              ON research_recipes(
                status,created_at DESC,recipe_code COLLATE BINARY ASC);
            CREATE TABLE IF NOT EXISTS research_recipe_mutation_receipts(
              idempotency_key TEXT PRIMARY KEY,
              mutation_kind TEXT NOT NULL
                CHECK(mutation_kind IN ('create','retire')),
              request_intent_hash TEXT NOT NULL
                CHECK(length(request_intent_hash)=64),
              recipe_code TEXT NOT NULL,
              response_status INTEGER NOT NULL
                CHECK(response_status IN (200,201)),
              response_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              CHECK(
                (mutation_kind='create' AND response_status=201) OR
                (mutation_kind='retire' AND response_status=200)),
              FOREIGN KEY(recipe_code) REFERENCES research_recipes(recipe_code));
            CREATE TABLE IF NOT EXISTS research_runs(
              run_code TEXT PRIMARY KEY, run_hash TEXT NOT NULL UNIQUE,
              recipe_code TEXT NOT NULL, recipe_hash TEXT NOT NULL,
              status TEXT NOT NULL, progress REAL NOT NULL,
              payload_json TEXT NOT NULL, created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              FOREIGN KEY(recipe_code) REFERENCES research_recipes(recipe_code));
            CREATE TABLE IF NOT EXISTS research_results(
              result_code TEXT PRIMARY KEY, result_hash TEXT NOT NULL UNIQUE,
              run_code TEXT NOT NULL, truth_label TEXT NOT NULL,
              confidence REAL NOT NULL, score REAL NOT NULL,
              disposition TEXT NOT NULL, payload_json TEXT NOT NULL,
              payload_hash TEXT,
              library_projection_json TEXT,
              library_projection_hash TEXT,
              attempt_count INTEGER, fencing_token INTEGER,
              created_at TEXT NOT NULL,
              CONSTRAINT ck_research_results_legacy_fence CHECK(
                (attempt_count IS NULL AND fencing_token IS NULL) OR
                (attempt_count IS NOT NULL AND fencing_token IS NOT NULL AND
                 attempt_count >= 1 AND fencing_token >= 1)),
              FOREIGN KEY(run_code) REFERENCES research_runs(run_code));
            CREATE INDEX IF NOT EXISTS ix_research_results_rank
              ON research_results(run_code,disposition,score DESC,result_code);
            CREATE TABLE IF NOT EXISTS research_result_facets(
              result_code TEXT NOT NULL,
              facet_kind TEXT NOT NULL,
              facet_value TEXT NOT NULL,
              PRIMARY KEY(result_code,facet_kind,facet_value),
              FOREIGN KEY(result_code) REFERENCES research_results(result_code));
            CREATE INDEX IF NOT EXISTS ix_research_result_facets_lookup
              ON research_result_facets(facet_kind,facet_value,result_code);
            CREATE TABLE IF NOT EXISTS semantic_disagreements(
              disagreement_id TEXT PRIMARY KEY,
              disagreement_hash TEXT NOT NULL UNIQUE,
              run_code TEXT NOT NULL, category TEXT NOT NULL,
              status TEXT NOT NULL, payload_json TEXT NOT NULL,
              attempt_count INTEGER, fencing_token INTEGER,
              created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
              CONSTRAINT ck_research_disagreements_legacy_fence CHECK(
                (attempt_count IS NULL AND fencing_token IS NULL) OR
                (attempt_count IS NOT NULL AND fencing_token IS NOT NULL AND
                 attempt_count >= 1 AND fencing_token >= 1)),
              FOREIGN KEY(run_code) REFERENCES research_runs(run_code));
            CREATE TABLE IF NOT EXISTS comparison_campaigns(
              campaign_code TEXT PRIMARY KEY, campaign_hash TEXT NOT NULL UNIQUE,
              base_recipe_code TEXT NOT NULL, payload_json TEXT NOT NULL,
              created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
              FOREIGN KEY(base_recipe_code) REFERENCES research_recipes(recipe_code));
            CREATE TABLE IF NOT EXISTS research_domain_compilation_artifacts(
              compilation_hash TEXT PRIMARY KEY,
              artifact_sha256 TEXT NOT NULL UNIQUE,
              record_json TEXT NOT NULL,
              created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS research_audit_traces(
              record_hash TEXT PRIMARY KEY,
              audit_trace_code TEXT NOT NULL UNIQUE,
              trace_hash TEXT NOT NULL,
              artifact_sha256 TEXT NOT NULL,
              run_code TEXT NOT NULL,
              shard_id TEXT,
               attempt_count INTEGER NOT NULL,
               fencing_token INTEGER NOT NULL,
               candidate_hash TEXT NOT NULL,
               candidate_ordinal INTEGER NOT NULL,
               seed INTEGER NOT NULL,
               source_occurrence_hash TEXT NOT NULL UNIQUE,
               disposition TEXT NOT NULL,
              status TEXT NOT NULL,
              attempt_status TEXT NOT NULL,
              record_json TEXT NOT NULL,
              admitted INTEGER NOT NULL DEFAULT 1,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              FOREIGN KEY(run_code) REFERENCES research_runs(run_code));
            CREATE TABLE IF NOT EXISTS research_shards(
              shard_id TEXT PRIMARY KEY, shard_hash TEXT NOT NULL UNIQUE,
              run_code TEXT NOT NULL, shard_index INTEGER NOT NULL,
              status TEXT NOT NULL, attempt_count INTEGER NOT NULL DEFAULT 0,
              retry_count INTEGER NOT NULL DEFAULT 0,
              reclaim_count INTEGER NOT NULL DEFAULT 0,
              operation_counters_available INTEGER NOT NULL DEFAULT 1,
              fencing_token INTEGER NOT NULL DEFAULT 0,
              lease_owner TEXT, lease_expires_at TEXT, heartbeat_at TEXT,
              spec_json TEXT NOT NULL, result_json TEXT, error_json TEXT,
              created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
              UNIQUE(run_code,shard_index),
              FOREIGN KEY(run_code) REFERENCES research_runs(run_code));
            CREATE UNIQUE INDEX IF NOT EXISTS ux_research_shards_identity_run
              ON research_shards(shard_id,run_code);
            CREATE INDEX IF NOT EXISTS ix_research_shards_claim
              ON research_shards(run_code,status,lease_expires_at,shard_index);
            CREATE TABLE IF NOT EXISTS research_result_occurrences(
              occurrence_hash TEXT PRIMARY KEY,
              result_code TEXT NOT NULL,
              run_code TEXT NOT NULL,
              shard_id TEXT NOT NULL,
              attempt_count INTEGER NOT NULL CHECK(attempt_count >= 1),
              fencing_token INTEGER NOT NULL CHECK(fencing_token >= 1),
              admitted INTEGER NOT NULL DEFAULT 0,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              UNIQUE(result_code,shard_id,attempt_count,fencing_token),
              FOREIGN KEY(result_code) REFERENCES research_results(result_code),
              FOREIGN KEY(run_code) REFERENCES research_runs(run_code),
              CONSTRAINT fk_research_result_occurrence_shard_run
                FOREIGN KEY(shard_id,run_code)
                REFERENCES research_shards(shard_id,run_code));
            CREATE INDEX IF NOT EXISTS ix_research_result_occurrences_admission
              ON research_result_occurrences(
                shard_id,attempt_count,fencing_token,admitted);
            CREATE TABLE IF NOT EXISTS semantic_disagreement_occurrences(
              occurrence_hash TEXT PRIMARY KEY,
              disagreement_id TEXT NOT NULL,
              run_code TEXT NOT NULL,
              shard_id TEXT NOT NULL,
              attempt_count INTEGER NOT NULL CHECK(attempt_count >= 1),
              fencing_token INTEGER NOT NULL CHECK(fencing_token >= 1),
              admitted INTEGER NOT NULL DEFAULT 0,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              UNIQUE(disagreement_id,shard_id,attempt_count,fencing_token),
              FOREIGN KEY(disagreement_id)
                REFERENCES semantic_disagreements(disagreement_id),
              FOREIGN KEY(run_code) REFERENCES research_runs(run_code),
              CONSTRAINT fk_research_disagreement_occurrence_shard_run
                FOREIGN KEY(shard_id,run_code)
                REFERENCES research_shards(shard_id,run_code));
            CREATE INDEX IF NOT EXISTS ix_research_disagreement_occurrences_admission
              ON semantic_disagreement_occurrences(
                shard_id,attempt_count,fencing_token,admitted);
            CREATE TABLE IF NOT EXISTS sequential_search_receipt_bundles(
              receipt_hash TEXT PRIMARY KEY,
              seed INTEGER NOT NULL,
              receipt_json TEXT NOT NULL,
              created_at TEXT NOT NULL);
            CREATE UNIQUE INDEX IF NOT EXISTS ux_research_receipt_bundle_hash_seed
              ON sequential_search_receipt_bundles(receipt_hash,seed);
            CREATE TABLE IF NOT EXISTS sequential_search_receipt_occurrences(
              occurrence_hash TEXT PRIMARY KEY,
              receipt_hash TEXT NOT NULL,
              run_code TEXT NOT NULL,
              shard_id TEXT NOT NULL,
              attempt_count INTEGER NOT NULL CHECK(attempt_count >= 1),
              fencing_token INTEGER NOT NULL CHECK(fencing_token >= 1),
              seed INTEGER NOT NULL,
              admitted INTEGER NOT NULL DEFAULT 0,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              UNIQUE(run_code,shard_id,attempt_count,fencing_token,seed),
              CONSTRAINT fk_research_receipt_occurrence_bundle_seed
                FOREIGN KEY(receipt_hash,seed)
                REFERENCES sequential_search_receipt_bundles(receipt_hash,seed),
              FOREIGN KEY(run_code) REFERENCES research_runs(run_code),
              CONSTRAINT fk_research_receipt_occurrence_shard_run
                FOREIGN KEY(shard_id,run_code)
                REFERENCES research_shards(shard_id,run_code));
            CREATE INDEX IF NOT EXISTS ix_research_search_receipts_library
              ON sequential_search_receipt_occurrences(
                admitted,run_code,created_at DESC,occurrence_hash);
            CREATE INDEX IF NOT EXISTS ix_research_search_receipts_admission
              ON sequential_search_receipt_occurrences(
                shard_id,attempt_count,fencing_token,admitted);
            CREATE UNIQUE INDEX IF NOT EXISTS ux_research_search_receipts_admitted_seed
              ON sequential_search_receipt_occurrences(run_code,seed)
              WHERE admitted=1;
            """
        )
        result_columns = {
            str(row["name"])
            for row in self._db.execute(
                "PRAGMA table_info(research_results)"
            ).fetchall()
        }
        if "shard_id" not in result_columns:
            self._db.execute("ALTER TABLE research_results ADD COLUMN shard_id TEXT")
        if "admitted" not in result_columns:
            self._db.execute(
                "ALTER TABLE research_results ADD COLUMN admitted INTEGER NOT NULL DEFAULT 1"
            )
        if "projection_version" not in result_columns:
            self._db.execute(
                "ALTER TABLE research_results ADD COLUMN projection_version INTEGER NOT NULL DEFAULT 5"
            )
        if "library_projection_json" not in result_columns:
            self._db.execute(
                "ALTER TABLE research_results ADD COLUMN library_projection_json TEXT"
            )
        if "payload_hash" not in result_columns:
            self._db.execute(
                "ALTER TABLE research_results ADD COLUMN payload_hash TEXT"
            )
        if "library_projection_hash" not in result_columns:
            self._db.execute(
                "ALTER TABLE research_results ADD COLUMN library_projection_hash TEXT"
            )
        self._db.execute(
            """CREATE INDEX IF NOT EXISTS ix_research_results_library
               ON research_results(admitted,created_at DESC,result_code ASC)"""
        )
        if "attempt_count" not in result_columns:
            self._db.execute(
                "ALTER TABLE research_results ADD COLUMN attempt_count INTEGER"
            )
        if "fencing_token" not in result_columns:
            self._db.execute(
                "ALTER TABLE research_results ADD COLUMN fencing_token INTEGER"
            )
        rows = self._db.execute(
            """SELECT result_code,result_hash,run_code,payload_json,payload_hash,created_at FROM research_results
               WHERE projection_version<? OR library_projection_json IS NULL
                  OR payload_hash IS NULL
               ORDER BY result_code""",
            (RESULT_PROJECTION_VERSION,),
        ).fetchall()
        for row in rows:
            decoded = _decode(row["payload_json"])
            payload_hash = row["payload_hash"] or stored_result_payload_hash_v1(decoded)
            payload = validate_stored_result_record_v1(
                decoded,
                result_code=row["result_code"],
                result_hash=row["result_hash"],
                run_code=row["run_code"],
                payload_hash=payload_hash,
            )
            library_value = result_library_safe_payload(payload)
            self._replace_result_facets(str(row["result_code"]), library_value)
            try:
                fallback_created_at = normalize_timestamp(row["created_at"]) or str(
                    row["created_at"]
                )
            except ValueError:
                fallback_created_at = str(row["created_at"])
            try:
                created_at = normalize_timestamp(
                    payload.get("createdAt"), fallback=fallback_created_at
                )
            except ValueError:
                created_at = fallback_created_at
            projection = result_library_projection(
                library_value, created_at=created_at
            )
            self._db.execute(
                """UPDATE research_results SET created_at=?,disposition=?,payload_hash=?,projection_version=?,
                   library_projection_json=?,library_projection_hash=?
                   WHERE result_code=?""",
                (
                    created_at,
                    library_value["disposition"],
                    payload_hash,
                    RESULT_PROJECTION_VERSION,
                    _encode(projection),
                    stored_result_library_projection_hash_v1(
                        projection, payload_hash=payload_hash
                    ),
                    str(row["result_code"]),
                ),
            )
        projection_rows = self._db.execute(
            """SELECT result_code,result_hash,run_code,payload_hash,library_projection_json
               FROM research_results WHERE library_projection_hash IS NULL
               ORDER BY result_code"""
        ).fetchall()
        for row in projection_rows:
            projection = validate_stored_result_identity_v1(
                _decode(row["library_projection_json"]),
                result_code=row["result_code"],
                result_hash=row["result_hash"],
                run_code=row["run_code"],
            )
            self._db.execute(
                "UPDATE research_results SET library_projection_hash=? WHERE result_code=?",
                (
                    stored_result_library_projection_hash_v1(
                        projection, payload_hash=row["payload_hash"]
                    ),
                    str(row["result_code"]),
                ),
            )
        disagreement_columns = {
            str(row["name"])
            for row in self._db.execute(
                "PRAGMA table_info(semantic_disagreements)"
            ).fetchall()
        }
        if "shard_id" not in disagreement_columns:
            self._db.execute(
                "ALTER TABLE semantic_disagreements ADD COLUMN shard_id TEXT"
            )
        if "admitted" not in disagreement_columns:
            self._db.execute(
                "ALTER TABLE semantic_disagreements ADD COLUMN admitted INTEGER NOT NULL DEFAULT 1"
            )
        if "attempt_count" not in disagreement_columns:
            self._db.execute(
                "ALTER TABLE semantic_disagreements ADD COLUMN attempt_count INTEGER"
            )
        if "fencing_token" not in disagreement_columns:
            self._db.execute(
                "ALTER TABLE semantic_disagreements ADD COLUMN fencing_token INTEGER"
            )
        shard_columns = {
            str(row["name"])
            for row in self._db.execute("PRAGMA table_info(research_shards)").fetchall()
        }
        if "retry_count" not in shard_columns:
            self._db.execute(
                "ALTER TABLE research_shards ADD COLUMN retry_count INTEGER NOT NULL DEFAULT 0"
            )
        if "reclaim_count" not in shard_columns:
            self._db.execute(
                "ALTER TABLE research_shards ADD COLUMN reclaim_count INTEGER NOT NULL DEFAULT 0"
            )
        if "operation_counters_available" not in shard_columns:
            self._db.execute(
                "ALTER TABLE research_shards ADD COLUMN operation_counters_available INTEGER"
            )
        # The audit-table rebuild owns its transaction so a failed row validation
        # leaves the legacy table untouched.
        self._db.commit()
        self._migrate_audit_trace_schema()
        self._db.commit()

    def _migrate_audit_trace_schema(self) -> None:
        """Upgrade the unreleased trace-hash draft without dropping its rows."""
        columns = self._db.execute(
            "PRAGMA table_info(research_audit_traces)"
        ).fetchall()
        required = {
            "record_hash",
            "audit_trace_code",
            "trace_hash",
            "artifact_sha256",
            "run_code",
            "shard_id",
            "attempt_count",
            "fencing_token",
            "candidate_hash",
            "candidate_ordinal",
            "seed",
            "source_occurrence_hash",
            "disposition",
            "status",
            "attempt_status",
            "record_json",
            "admitted",
            "created_at",
            "updated_at",
        }
        names = {str(row["name"]) for row in columns}
        primary_key = tuple(
            str(row["name"])
            for row in sorted(columns, key=lambda item: int(item["pk"]))
            if int(row["pk"]) > 0
        )
        if not required.issubset(names) or primary_key != ("record_hash",):
            self._rebuild_audit_trace_table(columns)

        rows = self._db.execute(
            """SELECT record_hash,record_json FROM research_audit_traces
               WHERE candidate_ordinal IS NULL OR seed IS NULL
                  OR source_occurrence_hash IS NULL"""
        ).fetchall()
        for row in rows:
            record = ResearchAuditTraceRecordV1.build(_decode(row["record_json"]))
            self._db.execute(
                """UPDATE research_audit_traces
                   SET candidate_ordinal=?,seed=?,source_occurrence_hash=?
                   WHERE record_hash=?""",
                (
                    int(record.source["candidateOrdinal"]),
                    int(record.source["seed"]),
                    self._audit_trace_source_occurrence_hash(record),
                    str(row["record_hash"]),
                ),
            )
        self._db.execute(
            """CREATE INDEX IF NOT EXISTS ix_research_audit_traces_library
               ON research_audit_traces(
                 admitted,disposition,created_at DESC,trace_hash)"""
        )
        self._db.execute(
            """CREATE INDEX IF NOT EXISTS ix_research_audit_traces_trace
               ON research_audit_traces(trace_hash,created_at DESC,record_hash)"""
        )
        self._db.execute(
            """CREATE UNIQUE INDEX IF NOT EXISTS
               ux_research_audit_traces_source_occurrence
               ON research_audit_traces(source_occurrence_hash)"""
        )

    def _rebuild_audit_trace_table(self, columns: list[sqlite3.Row]) -> None:
        names = {str(row["name"]) for row in columns}
        staging = "research_audit_traces_v1_upgrade"
        if (
            self._db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (staging,)
            ).fetchone()
            is not None
        ):
            raise RuntimeError(
                "Research audit trace schema upgrade is already in progress"
            )
        self._db.execute("BEGIN IMMEDIATE")
        try:
            self._db.execute(
                f"""CREATE TABLE {staging}(
                  record_hash TEXT PRIMARY KEY,
                  audit_trace_code TEXT NOT NULL UNIQUE,
                  trace_hash TEXT NOT NULL,
                  artifact_sha256 TEXT NOT NULL,
                  run_code TEXT NOT NULL,
                  shard_id TEXT,
                  attempt_count INTEGER NOT NULL,
                  fencing_token INTEGER NOT NULL,
                  candidate_hash TEXT NOT NULL,
                  candidate_ordinal INTEGER NOT NULL,
                  seed INTEGER NOT NULL,
                  source_occurrence_hash TEXT NOT NULL UNIQUE,
                  disposition TEXT NOT NULL,
                  status TEXT NOT NULL,
                  attempt_status TEXT NOT NULL,
                  record_json TEXT NOT NULL,
                  admitted INTEGER NOT NULL DEFAULT 1,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  FOREIGN KEY(run_code) REFERENCES research_runs(run_code))"""
            )
            for row in self._db.execute(
                "SELECT * FROM research_audit_traces ORDER BY trace_hash"
            ).fetchall():
                try:
                    record = ResearchAuditTraceRecordV1.build(
                        _decode(row["record_json"])
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        "legacy Research audit trace row cannot be migrated safely"
                    ) from exc
                indexed = {
                    "trace_hash": record.trace_hash,
                    "audit_trace_code": record.audit_trace_code,
                    "artifact_sha256": record.complete_trace["sha256"],
                    "run_code": record.source["runCode"],
                    "shard_id": record.source["shardId"],
                    "candidate_hash": record.source["candidateHash"],
                    "disposition": record.disposition,
                    "status": record.status,
                }
                if any(
                    key in names
                    and row[key] is not None
                    and str(row[key]) != str(expected)
                    for key, expected in indexed.items()
                ):
                    raise ValueError(
                        "legacy Research audit trace index identity mismatch"
                    )
                self._db.execute(
                    f"""INSERT INTO {staging}(
                       record_hash,audit_trace_code,trace_hash,artifact_sha256,
                       run_code,shard_id,attempt_count,fencing_token,
                       candidate_hash,candidate_ordinal,seed,
                       source_occurrence_hash,disposition,status,attempt_status,
                       record_json,admitted,created_at,updated_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        record.record_hash,
                        record.audit_trace_code,
                        record.trace_hash,
                        record.complete_trace["sha256"],
                        record.source["runCode"],
                        record.source["shardId"],
                        int(record.source["attemptCount"]),
                        int(record.source["fencingToken"]),
                        record.source["candidateHash"],
                        int(record.source["candidateOrdinal"]),
                        int(record.source["seed"]),
                        self._audit_trace_source_occurrence_hash(record),
                        record.disposition,
                        record.status,
                        record.attempt_status,
                        _encode(record.to_dict()),
                        int(row["admitted"]) if "admitted" in names else 1,
                        str(row["created_at"]),
                        str(row["updated_at"]),
                    ),
                )
            self._db.execute("DROP TABLE research_audit_traces")
            self._db.execute(f"ALTER TABLE {staging} RENAME TO research_audit_traces")
            self._db.commit()
        except Exception:
            self._db.rollback()
            raise

    def _replace_result_facets(
        self, result_code: str, payload: Mapping[str, Any]
    ) -> None:
        self._db.execute(
            "DELETE FROM research_result_facets WHERE result_code=?",
            (str(result_code),),
        )
        for facet_kind, values in result_facets(payload).items():
            self._db.executemany(
                """INSERT OR IGNORE INTO research_result_facets(
                   result_code,facet_kind,facet_value) VALUES(?,?,?)""",
                [(str(result_code), str(facet_kind), str(value)) for value in values],
            )

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def put_domain_compilation_artifact(
        self, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        requested = DomainCompilationArtifactV1.build(payload)
        created_at = requested.created_at or _now()
        value = DomainCompilationArtifactV1(
            requested.compilation_hash,
            requested.artifact,
            requested.bundle_hash,
            requested.effective_runtime_config_hash,
            requested.context_hash,
            requested.domain_hash,
            created_at,
        )
        with self._lock:
            row = self._db.execute(
                """SELECT record_json
                   FROM research_domain_compilation_artifacts
                   WHERE compilation_hash=?""",
                (value.compilation_hash,),
            ).fetchone()
            if row is not None:
                current = DomainCompilationArtifactV1.build(_decode(row["record_json"]))
                if (
                    current.artifact["sha256"] != value.artifact["sha256"]
                    or current.artifact["bytes"] != value.artifact["bytes"]
                    or current.artifact["mediaType"] != value.artifact["mediaType"]
                ):
                    raise ValueError(
                        "domain-compilation artifact identity is immutable"
                    )
                return current.to_dict()
            self._db.execute(
                """INSERT INTO research_domain_compilation_artifacts(
                   compilation_hash,artifact_sha256,record_json,created_at)
                   VALUES(?,?,?,?)""",
                (
                    value.compilation_hash,
                    value.artifact["sha256"],
                    _encode(value.to_dict()),
                    created_at,
                ),
            )
            self._db.commit()
        return value.to_dict()

    def get_domain_compilation_artifact(
        self, compilation_hash: str
    ) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                """SELECT record_json
                   FROM research_domain_compilation_artifacts
                   WHERE compilation_hash=?""",
                (str(compilation_hash),),
            ).fetchone()
        if row is None:
            return None
        record = DomainCompilationArtifactV1.build(_decode(row["record_json"]))
        if record.compilation_hash != str(compilation_hash):
            raise ValueError("domain-compilation artifact index key mismatch")
        return record.to_dict()

    @staticmethod
    def _audit_trace_identity(record: ResearchAuditTraceRecordV1) -> tuple[Any, ...]:
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

    @staticmethod
    def _audit_trace_source_occurrence_hash(
        record: ResearchAuditTraceRecordV1,
    ) -> str:
        return hashlib.sha256(_encode(record.source).encode("utf-8")).hexdigest()

    def put_audit_trace(
        self,
        payload: Mapping[str, Any],
        *,
        shard_id: str,
        worker_id: str,
        fencing_token: int,
    ) -> dict[str, Any]:
        requested = ResearchAuditTraceRecordV1.build(payload)
        source_shard = str(requested.source["shardId"])
        if str(shard_id) != source_shard:
            raise ValueError("Research audit trace shard identity mismatch")
        if str(worker_id) != str(requested.source["workerId"]) or int(
            fencing_token
        ) != int(requested.source["fencingToken"]):
            raise ValueError("Research audit trace attempt identity mismatch")
        now = _now()
        value = ResearchAuditTraceRecordV1.build(
            {**requested.to_dict(), "createdAt": now, "updatedAt": now}
        )
        source_occurrence_hash = self._audit_trace_source_occurrence_hash(value)
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                lease = self._db.execute(
                    """SELECT status,attempt_count,fencing_token,lease_owner,
                       lease_expires_at,run_code,shard_hash,shard_index
                       FROM research_shards WHERE shard_id=?""",
                    (source_shard,),
                ).fetchone()
                if (
                    lease is None
                    or str(lease["status"]) != "running"
                    or int(lease["attempt_count"])
                    != int(requested.source["attemptCount"])
                    or int(lease["fencing_token"]) != int(fencing_token)
                    or str(lease["lease_owner"] or "") != str(worker_id)
                    or str(lease["lease_expires_at"] or "") <= now
                    or str(lease["run_code"]) != str(requested.source["runCode"])
                    or str(lease["shard_hash"]) != str(requested.source["shardHash"])
                    or int(lease["shard_index"]) != int(requested.source["shardIndex"])
                ):
                    raise RuntimeError("stale Research shard audit-trace admission")
                row = self._db.execute(
                    """SELECT record_json,source_occurrence_hash
                       FROM research_audit_traces
                       WHERE source_occurrence_hash=? OR record_hash=?
                       ORDER BY source_occurrence_hash=? DESC LIMIT 1""",
                    (
                        source_occurrence_hash,
                        value.record_hash,
                        source_occurrence_hash,
                    ),
                ).fetchone()
                if row is not None:
                    current = ResearchAuditTraceRecordV1.build(
                        _decode(row["record_json"])
                    )
                    if str(row["source_occurrence_hash"]) != (
                        self._audit_trace_source_occurrence_hash(current)
                    ):
                        raise ValueError(
                            "Research audit trace source occurrence index mismatch"
                        )
                    if self._audit_trace_identity(
                        current
                    ) != self._audit_trace_identity(value):
                        raise ValueError(
                            "Research audit trace source occurrence is immutable"
                        )
                    self._db.commit()
                    return current.to_dict()
                self._db.execute(
                    """INSERT INTO research_audit_traces(
                       record_hash,audit_trace_code,trace_hash,artifact_sha256,
                       run_code,shard_id,attempt_count,fencing_token,candidate_hash,
                       candidate_ordinal,seed,source_occurrence_hash,
                       disposition,status,attempt_status,record_json,admitted,
                       created_at,updated_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        value.record_hash,
                        value.audit_trace_code,
                        value.trace_hash,
                        value.complete_trace["sha256"],
                        value.source["runCode"],
                        source_shard,
                        value.source["attemptCount"],
                        value.source["fencingToken"],
                        value.source["candidateHash"],
                        value.source["candidateOrdinal"],
                        value.source["seed"],
                        source_occurrence_hash,
                        value.disposition,
                        value.status,
                        value.attempt_status,
                        _encode(value.to_dict()),
                        1,
                        now,
                        now,
                    ),
                )
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
        return value.to_dict()

    def update_audit_trace(
        self,
        payload: Mapping[str, Any],
        *,
        worker_id: str,
        fencing_token: int,
    ) -> dict[str, Any]:
        requested = ResearchAuditTraceRecordV1.build(payload)
        if str(worker_id) != str(requested.source["workerId"]) or int(
            fencing_token
        ) != int(requested.source["fencingToken"]):
            raise ValueError("Research audit trace attempt identity mismatch")
        with self._lock:
            now = _now()
            self._db.execute("BEGIN IMMEDIATE")
            try:
                lease = self._db.execute(
                    """SELECT status,attempt_count,fencing_token,lease_owner,
                       lease_expires_at,run_code,shard_hash,shard_index
                       FROM research_shards WHERE shard_id=?""",
                    (str(requested.source["shardId"]),),
                ).fetchone()
                if (
                    lease is None
                    or str(lease["status"]) != "running"
                    or int(lease["attempt_count"])
                    != int(requested.source["attemptCount"])
                    or int(lease["fencing_token"]) != int(fencing_token)
                    or str(lease["lease_owner"] or "") != str(worker_id)
                    or str(lease["lease_expires_at"] or "") <= now
                    or str(lease["run_code"]) != str(requested.source["runCode"])
                    or str(lease["shard_hash"]) != str(requested.source["shardHash"])
                    or int(lease["shard_index"]) != int(requested.source["shardIndex"])
                ):
                    raise RuntimeError("stale Research shard audit-trace finalization")
                row = self._db.execute(
                    """SELECT record_json,source_occurrence_hash
                       FROM research_audit_traces WHERE record_hash=?""",
                    (requested.record_hash,),
                ).fetchone()
                if row is None:
                    raise KeyError(requested.record_hash)
                current = ResearchAuditTraceRecordV1.build(_decode(row["record_json"]))
                expected_occurrence = self._audit_trace_source_occurrence_hash(current)
                if str(row["source_occurrence_hash"]) != expected_occurrence:
                    raise ValueError(
                        "Research audit trace source occurrence index mismatch"
                    )
                if self._audit_trace_identity(current) != self._audit_trace_identity(
                    requested
                ):
                    raise ValueError("Research audit trace identity is immutable")
                if current.status == "finalized":
                    comparable_current = current.to_dict()
                    comparable_requested = requested.to_dict()
                    comparable_current.pop("updatedAt")
                    comparable_requested.pop("updatedAt")
                    if comparable_current != comparable_requested:
                        raise ValueError("finalized Research audit trace is immutable")
                    self._db.commit()
                    return current.to_dict()
                if requested.status != "finalized":
                    raise ValueError(
                        "Research audit trace update must finalize its lifecycle"
                    )
                value = ResearchAuditTraceRecordV1.build(
                    {
                        **requested.to_dict(),
                        "createdAt": current.created_at,
                        "updatedAt": now,
                    }
                )
                self._db.execute(
                    """UPDATE research_audit_traces
                       SET disposition=?,status=?,attempt_status=?,record_json=?,updated_at=?
                       WHERE record_hash=?""",
                    (
                        value.disposition,
                        value.status,
                        value.attempt_status,
                        _encode(value.to_dict()),
                        now,
                        value.record_hash,
                    ),
                )
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
        return value.to_dict()

    def get_audit_trace(
        self, identifier: str, *, include_unadmitted: bool = False
    ) -> dict[str, Any] | None:
        admitted = "" if include_unadmitted else " AND admitted=1"
        with self._lock:
            row = self._db.execute(
                f"""SELECT record_json FROM research_audit_traces
                    WHERE (record_hash=? OR trace_hash=? OR audit_trace_code=?)
                    {admitted} ORDER BY created_at DESC,record_hash LIMIT 1""",
                (str(identifier), str(identifier), str(identifier)),
            ).fetchone()
        if row is None:
            return None
        return ResearchAuditTraceRecordV1.build(_decode(row["record_json"])).to_dict()

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
            ("run_code", run_code),
            ("disposition", disposition),
            ("status", status),
        ):
            if value is not None:
                clauses.append(f"{column}=?")
                parameters.append(str(value))
        where = " WHERE " + " AND ".join(clauses)
        start = max(0, int(offset))
        size = max(1, min(500, int(limit)))
        with self._lock:
            total = int(
                self._db.execute(
                    f"SELECT COUNT(*) AS count FROM research_audit_traces{where}",
                    parameters,
                ).fetchone()["count"]
            )
            rows = self._db.execute(
                f"""SELECT record_json FROM research_audit_traces{where}
                    ORDER BY created_at DESC,trace_hash LIMIT ? OFFSET ?""",
                [*parameters, size, start],
            ).fetchall()
        items = [
            ResearchAuditTraceRecordV1.build(_decode(row["record_json"])).to_dict()
            for row in rows
        ]
        next_offset = start + len(items)
        return {
            "items": items,
            "total": total,
            "offset": start,
            "limit": size,
            "nextOffset": next_offset if next_offset < total else None,
        }

    @staticmethod
    def _recipe_mutation_receipt(row: Mapping[str, Any]) -> dict[str, Any]:
        try:
            created_at = normalize_timestamp(row["created_at"])
            if created_at is None:
                raise ValueError("receipt timestamp is missing")
            return {
                "schemaVersion": 1,
                "idempotencyKey": str(row["idempotency_key"]),
                "mutationKind": str(row["mutation_kind"]),
                "requestIntentHash": str(row["request_intent_hash"]),
                "recipeCode": str(row["recipe_code"]),
                "responseStatus": int(row["response_status"]),
                "response": _decode(row["response_json"]),
                "createdAt": created_at,
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ResearchMutationReceiptCorruptionError(
                "stored research recipe mutation receipt is invalid"
            ) from exc

    def get_recipe_mutation_receipt(
        self, idempotency_key: str
    ) -> dict[str, Any] | None:
        key = validate_idempotency_key(idempotency_key)
        with self._lock:
            row = self._db.execute(
                """SELECT idempotency_key,mutation_kind,request_intent_hash,
                          recipe_code,response_status,response_json,created_at
                   FROM research_recipe_mutation_receipts
                   WHERE idempotency_key=?""",
                (key,),
            ).fetchone()
        return None if row is None else self._recipe_mutation_receipt(row)

    def put_recipe(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        value = copy.deepcopy(dict(payload))
        now = _now()
        with self._lock:
            row = self._db.execute(
                "SELECT payload_json FROM research_recipes WHERE recipe_code=?",
                (str(value["code"]),),
            ).fetchone()
            if row is not None:
                current = _decode(row["payload_json"])
                if current != value:
                    raise ValueError(
                        "research recipe code is immutable; create a new revision"
                    )
                return current
            self._db.execute(
                "INSERT INTO research_recipes VALUES(?,?,?,?,?,?)",
                (
                    value["code"],
                    value["sha256"],
                    value["status"],
                    _encode(value),
                    now,
                    now,
                ),
            )
            self._db.commit()
        return value

    def put_recipe_with_receipt(
        self,
        payload: Mapping[str, Any],
        *,
        idempotency_key: str,
        request_intent_hash: str,
    ) -> dict[str, Any]:
        value = copy.deepcopy(dict(payload))
        key = validate_idempotency_key(idempotency_key)
        intent_hash = validate_request_intent_hash(request_intent_hash)
        with self._lock:
            try:
                self._db.execute("BEGIN IMMEDIATE")
                receipt_row = self._db.execute(
                    """SELECT idempotency_key,mutation_kind,request_intent_hash,
                              recipe_code,response_status,response_json,created_at
                       FROM research_recipe_mutation_receipts
                       WHERE idempotency_key=?""",
                    (key,),
                ).fetchone()
                if receipt_row is not None:
                    response = replay_recipe_mutation_receipt(
                        self._recipe_mutation_receipt(receipt_row),
                        mutation_kind="create",
                        request_intent_hash=intent_hash,
                    )
                    self._db.commit()
                    return response

                recipe_row = self._db.execute(
                    """SELECT status,payload_json FROM research_recipes
                       WHERE recipe_code=?""",
                    (str(value["code"]),),
                ).fetchone()
                if recipe_row is None:
                    now = _now()
                    self._db.execute(
                        """INSERT INTO research_recipes(
                               recipe_code,recipe_hash,status,payload_json,
                               created_at,updated_at)
                           VALUES(?,?,?,?,?,?)""",
                        (
                            value["code"],
                            value["sha256"],
                            value["status"],
                            _encode(value),
                            now,
                            now,
                        ),
                    )
                    current = value
                    lifecycle_status = str(value["status"])
                else:
                    current = _decode(recipe_row["payload_json"])
                    if current != value:
                        raise ValueError(
                            "research recipe code is immutable; create a new revision"
                        )
                    lifecycle_status = str(recipe_row["status"])
                response = {
                    **copy.deepcopy(current),
                    "lifecycleStatus": lifecycle_status,
                }
                self._db.execute(
                    """INSERT INTO research_recipe_mutation_receipts(
                           idempotency_key,mutation_kind,request_intent_hash,
                           recipe_code,response_status,response_json,created_at)
                       VALUES(?,?,?,?,?,?,?)""",
                    (
                        key,
                        "create",
                        intent_hash,
                        value["code"],
                        201,
                        _encode(response),
                        _now(),
                    ),
                )
                self._db.commit()
                return response
            except Exception:
                self._db.rollback()
                raise

    def get_recipe(self, code: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT payload_json FROM research_recipes WHERE recipe_code=?",
                (str(code),),
            ).fetchone()
        return None if row is None else _decode(row["payload_json"])

    def recipe_status(self, code: str) -> str | None:
        with self._lock:
            row = self._db.execute(
                "SELECT status FROM research_recipes WHERE recipe_code=?", (str(code),)
            ).fetchone()
        return None if row is None else str(row["status"])

    def retire_recipe(self, code: str) -> dict[str, Any]:
        with self._lock:
            row = self._db.execute(
                "SELECT status,payload_json FROM research_recipes WHERE recipe_code=?",
                (str(code),),
            ).fetchone()
            if row is None:
                raise KeyError(code)
            if str(row["status"]) == "draft":
                raise ValueError("a draft recipe cannot be retired")
            if str(row["status"]) != "retired":
                self._db.execute(
                    """UPDATE research_recipes
                       SET status='retired',updated_at=? WHERE recipe_code=?""",
                    (_now(), str(code)),
                )
            self._db.commit()
        return {**_decode(row["payload_json"]), "lifecycleStatus": "retired"}

    def retire_recipe_with_receipt(
        self,
        code: str,
        *,
        idempotency_key: str,
        request_intent_hash: str,
    ) -> dict[str, Any]:
        key = validate_idempotency_key(idempotency_key)
        intent_hash = validate_request_intent_hash(request_intent_hash)
        recipe_code = str(code)
        with self._lock:
            try:
                self._db.execute("BEGIN IMMEDIATE")
                receipt_row = self._db.execute(
                    """SELECT idempotency_key,mutation_kind,request_intent_hash,
                              recipe_code,response_status,response_json,created_at
                       FROM research_recipe_mutation_receipts
                       WHERE idempotency_key=?""",
                    (key,),
                ).fetchone()
                if receipt_row is not None:
                    response = replay_recipe_mutation_receipt(
                        self._recipe_mutation_receipt(receipt_row),
                        mutation_kind="retire",
                        request_intent_hash=intent_hash,
                    )
                    self._db.commit()
                    return response

                row = self._db.execute(
                    """SELECT status,payload_json FROM research_recipes
                       WHERE recipe_code=?""",
                    (recipe_code,),
                ).fetchone()
                if row is None:
                    raise KeyError(recipe_code)
                if str(row["status"]) == "draft":
                    raise ValueError("a draft recipe cannot be retired")
                if str(row["status"]) != "retired":
                    self._db.execute(
                        """UPDATE research_recipes
                           SET status='retired',updated_at=? WHERE recipe_code=?""",
                        (_now(), recipe_code),
                    )
                response = {
                    **_decode(row["payload_json"]),
                    "lifecycleStatus": "retired",
                }
                self._db.execute(
                    """INSERT INTO research_recipe_mutation_receipts(
                           idempotency_key,mutation_kind,request_intent_hash,
                           recipe_code,response_status,response_json,created_at)
                       VALUES(?,?,?,?,?,?,?)""",
                    (
                        key,
                        "retire",
                        intent_hash,
                        recipe_code,
                        200,
                        _encode(response),
                        _now(),
                    ),
                )
                self._db.commit()
                return response
            except Exception:
                self._db.rollback()
                raise

    def list_recipes(
        self, *, status: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        page = self.list_recipe_page(status=status, limit=limit)
        return [copy.deepcopy(item["recipe"]) for item in page["items"]]

    def list_recipe_page(
        self,
        *,
        status: str | None = None,
        limit: int = 100,
        after_created_at: str | None = None,
        after_recipe_code: str | None = None,
    ) -> dict[str, Any]:
        if (after_created_at is None) != (after_recipe_code is None):
            raise ValueError("research recipe cursor anchor is incomplete")
        anchor = None
        if after_created_at is not None and after_recipe_code is not None:
            anchor = RecipeListCursorV1.from_anchor(
                after_created_at,
                after_recipe_code,
                status=status,
            )
        elif status not in {None, "draft", "published", "retired"}:
            raise ValueError("research recipe status scope is invalid")
        size = max(1, min(500, int(limit)))
        clauses: list[str] = []
        parameters: list[Any] = []
        if status is not None:
            clauses.append("status=?")
            parameters.append(status)
        if anchor is not None:
            clauses.append(
                """(created_at < ? OR
                    (created_at = ? AND recipe_code COLLATE BINARY > ?))"""
            )
            parameters.extend(
                [anchor.created_at, anchor.created_at, anchor.recipe_code]
            )
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.append(size + 1)
        with self._lock:
            rows = self._db.execute(
                f"""SELECT recipe_code,status,payload_json,created_at
                    FROM research_recipes{where}
                    ORDER BY created_at DESC,recipe_code COLLATE BINARY ASC
                    LIMIT ?""",
                parameters,
            ).fetchall()
        has_more = len(rows) > size
        selected = rows[:size]
        items = [
            {
                "recipe": _decode(row["payload_json"]),
                "lifecycleStatus": str(row["status"]),
            }
            for row in selected
        ]
        next_anchor = None
        if has_more and selected:
            last = selected[-1]
            next_anchor = {
                "createdAt": normalize_timestamp(last["created_at"]),
                "recipeCode": str(last["recipe_code"]),
            }
        return {"items": items, "nextAnchor": next_anchor}

    def create_run(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        value = copy.deepcopy(dict(payload))
        now = _now()
        value["createdAt"] = normalize_timestamp(value.get("createdAt"), fallback=now)
        value["updatedAt"] = normalize_timestamp(value.get("updatedAt"), fallback=now)
        with self._lock:
            row = self._db.execute(
                "SELECT payload_json FROM research_runs WHERE run_hash=?",
                (value["runHash"],),
            ).fetchone()
            if row is not None:
                return _decode(row["payload_json"])
            self._db.execute(
                "INSERT INTO research_runs VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    value["runCode"],
                    value["runHash"],
                    value["recipeCode"],
                    value["recipeHash"],
                    value["status"],
                    float(value["progress"]),
                    _encode(value),
                    now,
                    now,
                ),
            )
            self._db.commit()
        return value

    def update_run(self, run_code: str, **changes: Any) -> dict[str, Any]:
        with self._lock:
            row = self._db.execute(
                "SELECT payload_json FROM research_runs WHERE run_code=?",
                (str(run_code),),
            ).fetchone()
            if row is None:
                raise KeyError(run_code)
            current = _decode(row["payload_json"])
            value = copy.deepcopy(current)
            value.update(copy.deepcopy(changes))
            if current.get("status") in {
                "completed",
                "failed",
                "cancelled",
            } and value.get("status") != current.get("status"):
                return current
            self._db.execute(
                "UPDATE research_runs SET status=?,progress=?,payload_json=?,updated_at=? WHERE run_code=?",
                (
                    value["status"],
                    float(value["progress"]),
                    _encode(value),
                    _now(),
                    str(run_code),
                ),
            )
            self._db.commit()
        return value

    def get_run(self, run_code: str, *, include_coordinator_evidence: bool = True) -> dict[str, Any] | None:
        payload = "payload_json" if include_coordinator_evidence else _OPERATOR_RUN_PAYLOAD_SQL
        with self._lock:
            row = self._db.execute(
                f"SELECT {payload} AS payload_json FROM research_runs WHERE run_code=?",
                (str(run_code),),
            ).fetchone()
        return None if row is None else _decode(row["payload_json"])

    def run_status(self, run_code: str) -> str | None:
        with self._lock:
            row = self._db.execute(
                "SELECT status FROM research_runs WHERE run_code=?",
                (str(run_code),),
            ).fetchone()
        return None if row is None else str(row["status"])

    def list_runs(
        self, *, status: str | None = None, limit: int | None = 100,
        include_coordinator_evidence: bool = True,
    ) -> list[dict[str, Any]]:
        payload = "payload_json" if include_coordinator_evidence else _OPERATOR_RUN_PAYLOAD_SQL
        sql = f"SELECT {payload} AS payload_json FROM research_runs"
        params: list[Any] = []
        if status is not None:
            sql += " WHERE status=?"
            params.append(status)
        sql += " ORDER BY created_at DESC,run_code"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._lock:
            rows = self._db.execute(sql, params).fetchall()
        return [_decode(row["payload_json"]) for row in rows]

    @staticmethod
    def _shard_row(row: sqlite3.Row) -> dict[str, Any]:
        value = {
            **_decode(row["spec_json"]),
            "status": str(row["status"]),
            "attemptCount": int(row["attempt_count"]),
            "retryCount": (
                int(row["retry_count"])
                if row["operation_counters_available"] == 1
                else None
            ),
            "reclaimCount": (
                int(row["reclaim_count"])
                if row["operation_counters_available"] == 1
                else None
            ),
            "fencingToken": int(row["fencing_token"]),
            "leaseOwner": row["lease_owner"],
            "leaseExpiresAt": row["lease_expires_at"],
            "heartbeatAt": row["heartbeat_at"],
            "result": None
            if row["result_json"] is None
            else _decode(row["result_json"]),
            "error": None if row["error_json"] is None else _decode(row["error_json"]),
        }
        return value

    def create_shards(
        self, specs: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...]
    ) -> list[dict[str, Any]]:
        """Insert immutable shard specs; repeated identical creation is idempotent."""
        values = [copy.deepcopy(dict(item)) for item in specs]
        if not values:
            return []
        run_codes = {str(value["runCode"]) for value in values}
        if len(run_codes) != 1:
            raise ValueError("research shard batch must belong to one run")
        run_code = next(iter(run_codes))
        now = _now()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                for value in values:
                    row = self._db.execute(
                        "SELECT spec_json FROM research_shards WHERE shard_id=?",
                        (str(value["shardId"]),),
                    ).fetchone()
                    if row is not None:
                        if _decode(row["spec_json"]) != value:
                            raise ValueError("research shard identity is immutable")
                        continue
                    self._db.execute(
                        """INSERT INTO research_shards(
                           shard_id,shard_hash,run_code,shard_index,status,
                           operation_counters_available,spec_json,created_at,updated_at)
                           VALUES(?,?,?,?,?,?,?,?,?)""",
                        (
                            value["shardId"],
                            value["sha256"],
                            value["runCode"],
                            int(value["shardIndex"]),
                            "queued",
                            1,
                            _encode(value),
                            now,
                            now,
                        ),
                    )
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
        return self.list_shards(run_code)

    def get_shard(self, shard_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM research_shards WHERE shard_id=?", (str(shard_id),)
            ).fetchone()
        return None if row is None else self._shard_row(row)

    def list_shards(self, run_code: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM research_shards WHERE run_code=? ORDER BY shard_index",
                (str(run_code),),
            ).fetchall()
        return [self._shard_row(row) for row in rows]

    def list_performance_shards(
        self, *, run_code: str | None = None
    ) -> list[dict[str, Any]]:
        where = " WHERE run_code=?" if run_code is not None else ""
        args: tuple[Any, ...] = (str(run_code),) if run_code is not None else ()
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM research_shards{where} ORDER BY run_code,shard_index",
                args,
            ).fetchall()
        return [self._shard_row(row) for row in rows]

    def performance_evidence_snapshot(
        self, *, run_code: str | None = None
    ) -> dict[str, list[dict[str, Any]]]:
        """Read all run-level and shard-level performance evidence atomically."""

        run_where = " WHERE run_code=?" if run_code is not None else ""
        shard_where = " WHERE run_code=?" if run_code is not None else ""
        args: tuple[Any, ...] = (str(run_code),) if run_code is not None else ()
        with self._lock:
            self._db.execute("BEGIN")
            try:
                run_rows = self._db.execute(
                    f"SELECT payload_json FROM research_runs{run_where} "
                    "ORDER BY created_at DESC,run_code",
                    args,
                ).fetchall()
                shard_rows = self._db.execute(
                    f"SELECT * FROM research_shards{shard_where} "
                    "ORDER BY run_code,shard_index",
                    args,
                ).fetchall()
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
        return {
            "runs": [_decode(row["payload_json"]) for row in run_rows],
            "shards": [self._shard_row(row) for row in shard_rows],
        }

    def _update_audit_trace_attempts_locked(
        self,
        shard_id: str,
        fencing_token: int,
        *,
        attempt_status: str,
        failure_categories: tuple[str, ...] = (),
        updated_at: str,
    ) -> None:
        rows = self._db.execute(
            """SELECT record_hash,record_json FROM research_audit_traces
               WHERE shard_id=? AND fencing_token=?""",
            (str(shard_id), int(fencing_token)),
        ).fetchall()
        for row in rows:
            current = ResearchAuditTraceRecordV1.build(_decode(row["record_json"]))
            if str(current.source["shardId"]) != str(shard_id) or int(
                current.source["fencingToken"]
            ) != int(fencing_token):
                raise ValueError("Research audit trace attempt index mismatch")
            value = ResearchAuditTraceRecordV1.build(
                {
                    **current.to_dict(),
                    "attemptStatus": str(attempt_status),
                    "failureCategories": sorted(
                        {*current.failure_categories, *failure_categories}
                    ),
                    "updatedAt": str(updated_at),
                }
            )
            self._db.execute(
                """UPDATE research_audit_traces
                   SET attempt_status=?,record_json=?,updated_at=?
                   WHERE record_hash=? AND fencing_token=?""",
                (
                    value.attempt_status,
                    _encode(value.to_dict()),
                    str(updated_at),
                    value.record_hash,
                    int(fencing_token),
                ),
            )

    def claim_shard(
        self,
        run_code: str,
        *,
        worker_id: str,
        lease_seconds: float,
        now: datetime | None = None,
    ) -> dict[str, Any] | None:
        seconds = normalize_lease_duration(lease_seconds)
        if now is not None:
            normalize_lease_moment(now)
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                moment = normalize_lease_moment(now)
                now_text = _sqlite_utc_text(moment)
                expiry = _sqlite_utc_text(moment + timedelta(seconds=seconds))
                candidates = self._db.execute(
                    """SELECT * FROM research_shards
                       WHERE run_code=? AND status IN ('queued','running')
                       ORDER BY shard_index""",
                    (str(run_code),),
                ).fetchall()
                row = next(
                    (
                        candidate
                        for candidate in candidates
                        if str(candidate["status"]) == "queued"
                        or not _sqlite_lease_is_live(
                            candidate["lease_expires_at"], moment=moment
                        )
                    ),
                    None,
                )
                if row is None:
                    self._db.commit()
                    return None
                attempt = int(row["attempt_count"]) + 1
                maximum = int(_decode(row["spec_json"])["maximumAttempts"])
                reclaim_increment = 1 if str(row["status"]) == "running" else 0
                if attempt > maximum:
                    if reclaim_increment:
                        self._update_audit_trace_attempts_locked(
                            str(row["shard_id"]),
                            int(row["fencing_token"]),
                            attempt_status="reclaimed",
                            failure_categories=(
                                "research_shard_attempt_reclaimed",
                                "research_shard_lease_expired",
                                "research_shard_maximum_attempts_exhausted",
                            ),
                            updated_at=now_text,
                        )
                    else:
                        self._update_audit_trace_attempts_locked(
                            str(row["shard_id"]),
                            int(row["fencing_token"]),
                            attempt_status="failed",
                            failure_categories=(
                                "research_shard_attempt_failed",
                                "research_shard_terminal_failure",
                                "research_shard_maximum_attempts_exhausted",
                            ),
                            updated_at=now_text,
                        )
                    self._db.execute(
                        """UPDATE research_shards SET status='failed',
                           reclaim_count=reclaim_count+?,updated_at=? WHERE shard_id=?""",
                        (reclaim_increment, now_text, str(row["shard_id"])),
                    )
                    self._db.commit()
                    return None
                if reclaim_increment:
                    self._update_audit_trace_attempts_locked(
                        str(row["shard_id"]),
                        int(row["fencing_token"]),
                        attempt_status="reclaimed",
                        failure_categories=(
                            "research_shard_attempt_reclaimed",
                            "research_shard_lease_expired",
                        ),
                        updated_at=now_text,
                    )
                fence = int(row["fencing_token"]) + 1
                self._db.execute(
                    """UPDATE research_shards SET status='running',attempt_count=?,
                       reclaim_count=reclaim_count+?,fencing_token=?,lease_owner=?,
                       lease_expires_at=?,heartbeat_at=?,
                       error_json=NULL,updated_at=? WHERE shard_id=?""",
                    (
                        attempt,
                        reclaim_increment,
                        fence,
                        str(worker_id),
                        expiry,
                        now_text,
                        now_text,
                        str(row["shard_id"]),
                    ),
                )
                claimed = self._db.execute(
                    "SELECT * FROM research_shards WHERE shard_id=?",
                    (str(row["shard_id"]),),
                ).fetchone()
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
        return self._shard_row(claimed)

    def heartbeat_shard(
        self,
        shard_id: str,
        *,
        worker_id: str,
        fencing_token: int,
        lease_seconds: float,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        fence = strict_positive_integer(fencing_token, field="research fencing_token")
        seconds = normalize_lease_duration(lease_seconds)
        if now is not None:
            normalize_lease_moment(now)
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                moment = normalize_lease_moment(now)
                row = self._db.execute(
                    """SELECT status,lease_owner,fencing_token,lease_expires_at
                       FROM research_shards WHERE shard_id=?""",
                    (str(shard_id),),
                ).fetchone()
                if (
                    row is None
                    or str(row["status"]) != "running"
                    or str(row["lease_owner"] or "") != str(worker_id)
                    or int(row["fencing_token"]) != fence
                    or not _sqlite_lease_is_live(row["lease_expires_at"], moment=moment)
                ):
                    raise RuntimeError("stale research shard lease")
                now_text = _sqlite_utc_text(moment)
                expiry = _sqlite_utc_text(moment + timedelta(seconds=seconds))
                self._db.execute(
                    """UPDATE research_shards
                       SET heartbeat_at=?,lease_expires_at=?,updated_at=?
                       WHERE shard_id=?""",
                    (now_text, expiry, now_text, str(shard_id)),
                )
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
        value = self.get_shard(shard_id)
        if value is None:
            raise KeyError(shard_id)
        return value

    def complete_shard(
        self,
        shard_id: str,
        *,
        worker_id: str,
        fencing_token: int,
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        fence = strict_positive_integer(fencing_token, field="research fencing_token")
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                moment = normalize_lease_moment(None)
                now = _sqlite_utc_text(moment)
                current = self._db.execute(
                    """SELECT status,lease_owner,fencing_token,lease_expires_at
                       FROM research_shards WHERE shard_id=?""",
                    (str(shard_id),),
                ).fetchone()
                if (
                    current is None
                    or str(current["status"]) != "running"
                    or str(current["lease_owner"] or "") != str(worker_id)
                    or int(current["fencing_token"]) != fence
                    or not _sqlite_lease_is_live(
                        current["lease_expires_at"], moment=moment
                    )
                ):
                    raise RuntimeError("stale research shard completion")
                cursor = self._db.execute(
                    """UPDATE research_shards SET status='completed',result_json=?,
                       error_json=NULL,lease_owner=NULL,lease_expires_at=NULL,
                       heartbeat_at=?,updated_at=?
                        WHERE shard_id=? AND status='running'
                          AND lease_owner=? AND fencing_token=?""",
                    (
                        _encode(dict(result)),
                        now,
                        now,
                        str(shard_id),
                        str(worker_id),
                        fence,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("stale research shard completion")
                shard_row = self._db.execute(
                    "SELECT attempt_count FROM research_shards WHERE shard_id=?",
                    (str(shard_id),),
                ).fetchone()
                if shard_row is None:
                    raise KeyError(shard_id)
                attempt_count = int(shard_row["attempt_count"])
                self._update_audit_trace_attempts_locked(
                    str(shard_id),
                    fence,
                    attempt_status="completed",
                    updated_at=now,
                )
                self._db.execute(
                    """UPDATE research_result_occurrences
                       SET admitted=1,updated_at=?
                       WHERE shard_id=? AND attempt_count=?
                         AND fencing_token=?""",
                    (
                        now,
                        str(shard_id),
                        attempt_count,
                        fence,
                    ),
                )
                # The semantic row and its first-origin projections stay
                # immutable.  Exact occurrence admission controls visibility.
                self._db.execute(
                    """UPDATE research_results SET admitted=1
                       WHERE result_code IN (
                         SELECT result_code FROM research_result_occurrences
                         WHERE shard_id=? AND attempt_count=?
                           AND fencing_token=? AND admitted=1
                       )""",
                    (str(shard_id), attempt_count, fence),
                )
                self._db.execute(
                    """UPDATE semantic_disagreement_occurrences
                       SET admitted=1,updated_at=?
                       WHERE shard_id=? AND attempt_count=?
                         AND fencing_token=?""",
                    (
                        now,
                        str(shard_id),
                        attempt_count,
                        fence,
                    ),
                )
                self._db.execute(
                    """UPDATE semantic_disagreements SET admitted=1
                       WHERE disagreement_id IN (
                         SELECT disagreement_id
                         FROM semantic_disagreement_occurrences
                         WHERE shard_id=? AND attempt_count=?
                           AND fencing_token=? AND admitted=1
                       )""",
                    (str(shard_id), attempt_count, fence),
                )
                self._db.execute(
                    """UPDATE sequential_search_receipt_occurrences
                       SET admitted=1,updated_at=?
                       WHERE shard_id=? AND attempt_count=?
                         AND fencing_token=?""",
                    (
                        now,
                        str(shard_id),
                        attempt_count,
                        fence,
                    ),
                )
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
        value = self.get_shard(shard_id)
        if value is None:
            raise KeyError(shard_id)
        return value

    def fail_shard(
        self,
        shard_id: str,
        *,
        worker_id: str,
        fencing_token: int,
        error: Mapping[str, Any],
        retryable: bool = True,
    ) -> dict[str, Any]:
        fence = strict_positive_integer(fencing_token, field="research fencing_token")
        now = _now()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute(
                    "SELECT * FROM research_shards WHERE shard_id=?",
                    (str(shard_id),),
                ).fetchone()
                if row is None:
                    raise KeyError(shard_id)
                maximum = int(_decode(row["spec_json"])["maximumAttempts"])
                next_status = (
                    "queued"
                    if retryable and int(row["attempt_count"]) < maximum
                    else "failed"
                )
                retry_increment = 1 if next_status == "queued" else 0
                cursor = self._db.execute(
                    """UPDATE research_shards SET status=?,
                       retry_count=retry_count+?,error_json=?,lease_owner=NULL,
                       lease_expires_at=NULL,heartbeat_at=?,updated_at=?
                       WHERE shard_id=? AND status='running'
                         AND lease_owner=? AND fencing_token=?""",
                    (
                        next_status,
                        retry_increment,
                        _encode(dict(error)),
                        now,
                        now,
                        str(shard_id),
                        str(worker_id),
                        fence,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("stale research shard failure")
                if next_status == "queued":
                    attempt_status = "retry_scheduled"
                    failure_categories = (
                        "research_shard_attempt_failed",
                        "research_shard_retry_scheduled",
                    )
                else:
                    attempt_status = "failed"
                    failure_categories = (
                        "research_shard_attempt_failed",
                        "research_shard_terminal_failure",
                    )
                self._update_audit_trace_attempts_locked(
                    str(shard_id),
                    fence,
                    attempt_status=attempt_status,
                    failure_categories=failure_categories,
                    updated_at=now,
                )
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
        value = self.get_shard(shard_id)
        if value is None:
            raise KeyError(shard_id)
        return value

    def cancel_shards(self, run_code: str) -> int:
        now = _now()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                rows = self._db.execute(
                    """SELECT shard_id,fencing_token FROM research_shards
                       WHERE run_code=? AND status IN ('queued','running')""",
                    (str(run_code),),
                ).fetchall()
                cursor = self._db.execute(
                    """UPDATE research_shards SET status='cancelled',lease_owner=NULL,
                       lease_expires_at=NULL,updated_at=?
                       WHERE run_code=? AND status IN ('queued','running')""",
                    (now, str(run_code)),
                )
                for row in rows:
                    self._update_audit_trace_attempts_locked(
                        str(row["shard_id"]),
                        int(row["fencing_token"]),
                        attempt_status="cancelled",
                        failure_categories=("research_shard_attempt_cancelled",),
                        updated_at=now,
                    )
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
        return int(cursor.rowcount)

    def recover_orphaned_shards(self) -> list[str]:
        """A new process invalidates all prior in-process worker leases."""
        now = _now()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                rows = self._db.execute(
                    """SELECT run_code,shard_id,fencing_token
                       FROM research_shards WHERE status='running'"""
                ).fetchall()
                self._db.execute(
                    """UPDATE research_shards SET status='queued',lease_owner=NULL,
                       lease_expires_at=NULL,reclaim_count=reclaim_count+1,
                       updated_at=? WHERE status='running'""",
                    (now,),
                )
                for row in rows:
                    self._update_audit_trace_attempts_locked(
                        str(row["shard_id"]),
                        int(row["fencing_token"]),
                        attempt_status="reclaimed",
                        failure_categories=(
                            "research_shard_attempt_reclaimed",
                            "research_shard_process_recovered",
                        ),
                        updated_at=now,
                    )
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
        return sorted({str(row["run_code"]) for row in rows})

    def put_comparison_campaign(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        value = copy.deepcopy(dict(payload))
        now = _now()
        with self._lock:
            row = self._db.execute(
                "SELECT payload_json FROM comparison_campaigns WHERE campaign_hash=?",
                (str(value["campaignHash"]),),
            ).fetchone()
            if row is not None:
                return _decode(row["payload_json"])
            self._db.execute(
                "INSERT INTO comparison_campaigns VALUES(?,?,?,?,?,?)",
                (
                    value["campaignCode"],
                    value["campaignHash"],
                    value["baseRecipeCode"],
                    _encode(value),
                    now,
                    now,
                ),
            )
            self._db.commit()
        return value

    def get_comparison_campaign(self, code: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT payload_json FROM comparison_campaigns WHERE campaign_code=?",
                (str(code),),
            ).fetchone()
        return None if row is None else _decode(row["payload_json"])

    def list_comparison_campaigns(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT payload_json FROM comparison_campaigns ORDER BY created_at DESC,campaign_code LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [_decode(row["payload_json"]) for row in rows]

    def _validate_occurrence_shard_locked(
        self,
        *,
        run_code: str,
        shard_id: str,
        attempt_count: int,
        fencing_token: int,
        worker_id: str,
        seed: int | None = None,
    ) -> None:
        row = self._db.execute(
            """SELECT run_code,spec_json,status,attempt_count,fencing_token,
               lease_owner,lease_expires_at
               FROM research_shards WHERE shard_id=?""",
            (str(shard_id),),
        ).fetchone()
        if row is None:
            raise KeyError(shard_id)
        moment = normalize_lease_moment(None)
        if (
            str(row["run_code"]) != str(run_code)
            or str(row["status"]) != "running"
            or int(row["attempt_count"]) != int(attempt_count)
            or int(row["fencing_token"]) != int(fencing_token)
            or str(row["lease_owner"] or "") != str(worker_id)
            or not _sqlite_lease_is_live(row["lease_expires_at"], moment=moment)
        ):
            raise RuntimeError("stale research shard occurrence admission")
        if seed is not None:
            spec = _decode(row["spec_json"])
            start = int(spec["seedStart"])
            end = start + int(spec["seedCount"])
            if not start <= int(seed) < end:
                raise ValueError("research receipt seed is outside its shard")

    def _validate_unsharded_write_locked(self, run_code: str) -> None:
        row = self._db.execute(
            "SELECT 1 FROM research_shards WHERE run_code=? LIMIT 1",
            (str(run_code),),
        ).fetchone()
        if row is not None:
            raise ValueError(
                "sharded research writes require an explicit current lease"
            )

    def _put_fenced_occurrence_locked(
        self,
        *,
        table: str,
        semantic_column: str,
        semantic_id: str,
        occurrence: Mapping[str, Any],
        now: str,
    ) -> None:
        if table not in {
            "research_result_occurrences",
            "semantic_disagreement_occurrences",
        }:
            raise ValueError("unsupported fenced occurrence table")
        row = self._db.execute(
            f"""SELECT occurrence_hash,{semantic_column},run_code
                FROM {table}
                WHERE {semantic_column}=? AND shard_id=?
                  AND attempt_count=? AND fencing_token=?""",
            (
                str(semantic_id),
                str(occurrence["shardId"]),
                int(occurrence["attemptCount"]),
                int(occurrence["fencingToken"]),
            ),
        ).fetchone()
        if row is not None:
            if (
                str(row["occurrence_hash"]) != str(occurrence["occurrenceHash"])
                or str(row[semantic_column]) != str(semantic_id)
                or str(row["run_code"]) != str(occurrence["runCode"])
            ):
                raise RuntimeError("research occurrence identity is immutable")
            return
        collision = self._db.execute(
            f"SELECT {semantic_column} FROM {table} WHERE occurrence_hash=?",
            (str(occurrence["occurrenceHash"]),),
        ).fetchone()
        if collision is not None:
            raise RuntimeError("research occurrence hash collision")
        self._db.execute(
            f"""INSERT INTO {table}(
                occurrence_hash,{semantic_column},run_code,shard_id,
                attempt_count,fencing_token,admitted,created_at,updated_at)
                VALUES(?,?,?,?,?,?,0,?,?)""",
            (
                str(occurrence["occurrenceHash"]),
                str(semantic_id),
                str(occurrence["runCode"]),
                str(occurrence["shardId"]),
                int(occurrence["attemptCount"]),
                int(occurrence["fencingToken"]),
                now,
                now,
            ),
        )

    @staticmethod
    def _sequential_search_receipt_row(row: Mapping[str, Any]) -> dict[str, Any]:
        occurrence_hash = _strict_receipt_text(
            row["occurrence_hash"], field="sequential receipt occurrence hash"
        )
        receipt_hash = _strict_receipt_text(
            row["receipt_hash"], field="sequential receipt hash"
        )
        run_code = _strict_receipt_text(
            row["run_code"], field="sequential receipt run code"
        )
        shard_id = _strict_receipt_text(
            row["shard_id"], field="sequential receipt shard id"
        )
        attempt_count = strict_positive_integer(
            row["attempt_count"], field="sequential receipt attempt count"
        )
        fencing_token = strict_positive_integer(
            row["fencing_token"], field="sequential receipt fencing token"
        )
        seed = _strict_nonnegative_database_integer(
            row["seed"], field="sequential receipt seed"
        )
        admitted_value = row["admitted"]
        if type(admitted_value) is not int or admitted_value not in (0, 1):
            raise ValueError("sequential receipt admitted flag is invalid")
        created_at = _strict_receipt_text(
            row["created_at"], field="sequential receipt creation time"
        )
        return {
            "schemaVersion": 1,
            "occurrenceHash": occurrence_hash,
            "receiptHash": receipt_hash,
            "runCode": run_code,
            "shardId": shard_id,
            "attemptCount": attempt_count,
            "fencingToken": fencing_token,
            "seed": seed,
            "admitted": admitted_value == 1,
            "createdAt": created_at,
            "receiptBundle": _decode(row["receipt_json"]),
        }

    def put_sequential_search_receipt(
        self,
        receipt: Mapping[str, Any],
        *,
        run_code: str,
        shard_id: str,
        attempt_count: int,
        fencing_token: int,
        worker_id: str,
        seed: int,
    ) -> dict[str, Any]:
        normalized_run_code = _strict_receipt_text(
            run_code, field="sequential receipt run code"
        )
        normalized_shard_id = _strict_receipt_text(
            shard_id, field="sequential receipt shard id"
        )
        normalized_worker_id = _strict_receipt_text(
            worker_id, field="sequential receipt worker id"
        )
        normalized_seed = _strict_nonnegative_database_integer(
            seed, field="sequential receipt seed"
        )
        fenced = normalize_fenced_attempt(
            shard_id=normalized_shard_id,
            attempt_count=attempt_count,
            fencing_token=fencing_token,
            worker_id=normalized_worker_id,
        )
        if fenced is None:  # pragma: no cover - guarded by the supplied shard
            raise ValueError("sequential search receipts require a shard")
        bundle = build_sequential_search_receipt_bundle(receipt, seed=normalized_seed)
        occurrence = build_fenced_occurrence(
            kind="sequential_search_receipt",
            semantic_hash=str(bundle["sha256"]),
            run_code=normalized_run_code,
            shard_id=fenced[0],
            attempt_count=fenced[1],
            fencing_token=fenced[2],
            seed=normalized_seed,
        )
        now = _now()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                self._validate_occurrence_shard_locked(
                    run_code=normalized_run_code,
                    shard_id=fenced[0],
                    attempt_count=fenced[1],
                    fencing_token=fenced[2],
                    worker_id=fenced[3],
                    seed=normalized_seed,
                )
                bundle_row = self._db.execute(
                    """SELECT seed,receipt_json FROM sequential_search_receipt_bundles
                       WHERE receipt_hash=?""",
                    (str(bundle["sha256"]),),
                ).fetchone()
                if bundle_row is None:
                    self._db.execute(
                        """INSERT INTO sequential_search_receipt_bundles(
                           receipt_hash,seed,receipt_json,created_at)
                           VALUES(?,?,?,?)""",
                        (
                            str(bundle["sha256"]),
                            normalized_seed,
                            _encode(bundle),
                            now,
                        ),
                    )
                elif (
                    _strict_nonnegative_database_integer(
                        bundle_row["seed"], field="sequential receipt bundle seed"
                    )
                    != normalized_seed
                    or _decode(bundle_row["receipt_json"]) != bundle
                ):
                    raise RuntimeError("sequential search receipt CAS payload mismatch")
                existing = self._db.execute(
                    """SELECT occurrence_hash,receipt_hash
                       FROM sequential_search_receipt_occurrences
                       WHERE run_code=? AND shard_id=? AND attempt_count=?
                         AND fencing_token=? AND seed=?""",
                    (
                        normalized_run_code,
                        fenced[0],
                        fenced[1],
                        fenced[2],
                        normalized_seed,
                    ),
                ).fetchone()
                if existing is not None:
                    if str(existing["occurrence_hash"]) != str(
                        occurrence["occurrenceHash"]
                    ) or str(existing["receipt_hash"]) != str(bundle["sha256"]):
                        raise RuntimeError(
                            "sequential search seed occurrence is immutable"
                        )
                else:
                    collision = self._db.execute(
                        """SELECT receipt_hash
                           FROM sequential_search_receipt_occurrences
                           WHERE occurrence_hash=?""",
                        (str(occurrence["occurrenceHash"]),),
                    ).fetchone()
                    if collision is not None:
                        raise RuntimeError(
                            "sequential search occurrence hash collision"
                        )
                    self._db.execute(
                        """INSERT INTO sequential_search_receipt_occurrences(
                           occurrence_hash,receipt_hash,run_code,shard_id,
                           attempt_count,fencing_token,seed,admitted,
                           created_at,updated_at)
                           VALUES(?,?,?,?,?,?,?,0,?,?)""",
                        (
                            str(occurrence["occurrenceHash"]),
                            str(bundle["sha256"]),
                            normalized_run_code,
                            fenced[0],
                            fenced[1],
                            fenced[2],
                            normalized_seed,
                            now,
                            now,
                        ),
                    )
                row = self._db.execute(
                    """SELECT occurrence.*,bundle.receipt_json
                       FROM sequential_search_receipt_occurrences occurrence
                       JOIN sequential_search_receipt_bundles bundle
                         ON bundle.receipt_hash=occurrence.receipt_hash
                       WHERE occurrence.occurrence_hash=?""",
                    (str(occurrence["occurrenceHash"]),),
                ).fetchone()
                if row is None:  # pragma: no cover - protected by this transaction
                    raise RuntimeError("sequential search receipt was not retained")
                value = self._sequential_search_receipt_row(row)
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
        return value

    @staticmethod
    def _search_receipt_filters(
        *,
        run_code: str | None,
        shard_id: str | None,
        admitted_only: bool,
    ) -> tuple[str, list[Any]]:
        if type(admitted_only) is not bool:
            raise ValueError("sequential receipt admitted_only must be a boolean")
        clauses: list[str] = []
        params: list[Any] = []
        if admitted_only:
            clauses.append("occurrence.admitted=1")
        if run_code is not None:
            run_code = _strict_receipt_text(
                run_code, field="sequential receipt run code"
            )
            clauses.append("occurrence.run_code=?")
            params.append(run_code)
        if shard_id is not None:
            shard_id = _strict_receipt_text(
                shard_id, field="sequential receipt shard id"
            )
            clauses.append("occurrence.shard_id=?")
            params.append(shard_id)
        return (" WHERE " + " AND ".join(clauses) if clauses else "", params)

    def count_sequential_search_receipts(
        self,
        *,
        run_code: str | None = None,
        shard_id: str | None = None,
        admitted_only: bool = True,
    ) -> int:
        where, params = self._search_receipt_filters(
            run_code=run_code,
            shard_id=shard_id,
            admitted_only=admitted_only,
        )
        with self._lock:
            row = self._db.execute(
                f"""SELECT COUNT(*) AS count
                    FROM sequential_search_receipt_occurrences occurrence{where}""",
                params,
            ).fetchone()
        return (
            0
            if row is None
            else _strict_nonnegative_database_integer(
                row["count"], field="sequential receipt count"
            )
        )

    @contextmanager
    def iter_admitted_sequential_search_receipts(
        self,
        *,
        run_code: str | None = None,
        batch_size: int = 1000,
    ) -> Iterator[Iterator[dict[str, Any]]]:
        """Stream one fixed admitted-receipt snapshot in canonical stats order."""

        normalized_batch_size = strict_positive_integer(
            batch_size, field="sequential receipt snapshot batch size"
        )
        if normalized_batch_size > _SEQUENTIAL_RECEIPT_SNAPSHOT_BATCH_MAX:
            raise ValueError("sequential receipt snapshot batch size is too large")
        where, params = self._search_receipt_filters(
            run_code=run_code,
            shard_id=None,
            admitted_only=True,
        )
        order = (
            "occurrence.seed ASC,occurrence.occurrence_hash ASC"
            if run_code is not None
            else (
                "occurrence.run_code ASC,occurrence.seed ASC,"
                "occurrence.occurrence_hash ASC"
            )
        )
        cursor: sqlite3.Cursor | None = None
        self._lock.acquire()
        try:
            self._db.execute("BEGIN")
            cursor = self._db.execute(
                f"""SELECT occurrence.*,bundle.receipt_json
                    FROM sequential_search_receipt_occurrences occurrence
                    JOIN sequential_search_receipt_bundles bundle
                      ON bundle.receipt_hash=occurrence.receipt_hash{where}
                    ORDER BY {order}""",
                params,
            )
            cursor.arraysize = normalized_batch_size

            def rows() -> Iterator[dict[str, Any]]:
                while True:
                    batch = cursor.fetchmany(normalized_batch_size)
                    if not batch:
                        return
                    for row in batch:
                        yield self._sequential_search_receipt_row(row)

            yield rows()
        finally:
            if cursor is not None:
                cursor.close()
            if self._db.in_transaction:
                self._db.rollback()
            self._lock.release()

    def list_sequential_search_receipts(
        self,
        *,
        run_code: str | None = None,
        shard_id: str | None = None,
        admitted_only: bool = True,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        normalized_offset = _strict_nonnegative_database_integer(
            offset, field="sequential receipt page offset"
        )
        normalized_limit = strict_positive_integer(
            limit, field="sequential receipt page limit"
        )
        if normalized_limit > _SEQUENTIAL_RECEIPT_PAGE_LIMIT_MAX:
            raise ValueError("sequential receipt page limit is too large")
        where, params = self._search_receipt_filters(
            run_code=run_code,
            shard_id=shard_id,
            admitted_only=admitted_only,
        )
        with self._lock:
            rows = self._db.execute(
                f"""WITH filtered AS (
                      SELECT occurrence.*,bundle.receipt_json
                      FROM sequential_search_receipt_occurrences occurrence
                      JOIN sequential_search_receipt_bundles bundle
                        ON bundle.receipt_hash=occurrence.receipt_hash{where}
                    ), page AS (
                      SELECT * FROM filtered ORDER BY created_at DESC,
                        occurrence_hash ASC LIMIT ? OFFSET ?
                    )
                    SELECT page.*,totals.total
                    FROM (SELECT COUNT(*) AS total FROM filtered) totals
                    LEFT JOIN page ON 1=1
                    ORDER BY page.created_at DESC,page.occurrence_hash ASC""",
                [*params, normalized_limit, normalized_offset],
            ).fetchall()
        total = (
            0
            if not rows
            else _strict_nonnegative_database_integer(
                rows[0]["total"], field="sequential receipt page total"
            )
        )
        page_rows = [row for row in rows if row["occurrence_hash"] is not None]
        next_offset = normalized_offset + len(page_rows)
        return {
            "items": [self._sequential_search_receipt_row(row) for row in page_rows],
            "total": total,
            "offset": normalized_offset,
            "limit": normalized_limit,
            "nextOffset": next_offset if next_offset < total else None,
            "hasMore": next_offset < total,
        }

    def put_result(
        self,
        payload: Mapping[str, Any],
        *,
        attempt_count: int | None = None,
        fencing_token: int | None = None,
        worker_id: str | None = None,
    ) -> dict[str, Any]:
        value = copy.deepcopy(dict(payload))
        now = _now()
        created_at = normalize_timestamp(value.get("createdAt"), fallback=now)
        shard_id = dict(value.get("provenance") or {}).get("researchShardId")
        fenced = normalize_fenced_attempt(
            shard_id=None if shard_id is None else str(shard_id),
            attempt_count=attempt_count,
            fencing_token=fencing_token,
            worker_id=worker_id,
        )
        occurrence = (
            None
            if fenced is None
            else build_fenced_occurrence(
                kind="research_result",
                semantic_hash=str(value["resultHash"]),
                run_code=str(value["runCode"]),
                shard_id=fenced[0],
                attempt_count=fenced[1],
                fencing_token=fenced[2],
            )
        )
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                if fenced is None:
                    self._validate_unsharded_write_locked(str(value["runCode"]))
                row = self._db.execute(
                    """SELECT result_code,run_code,payload_json,payload_hash,shard_id,attempt_count,
                       fencing_token FROM research_results WHERE result_hash=?""",
                    (value["resultHash"],),
                ).fetchone()
                if row is not None:
                    stored = validate_stored_result_record_v1(
                        _decode(row["payload_json"]),
                        result_code=row["result_code"],
                        result_hash=value["resultHash"],
                        run_code=row["run_code"],
                        payload_hash=row["payload_hash"],
                    )
                    if (
                        str(row["result_code"]) != str(value["resultCode"])
                        or stored != value
                    ):
                        raise RuntimeError("research result CAS payload mismatch")
                    result_code = str(row["result_code"])
                else:
                    library_value = result_library_safe_payload(value)
                    payload_hash = stored_result_payload_hash_v1(value)
                    projection = result_library_projection(
                        library_value, created_at=created_at
                    )
                    self._db.execute(
                        """INSERT INTO research_results(
                           result_code,result_hash,run_code,truth_label,
                           confidence,score,disposition,payload_json,payload_hash,created_at,
                           shard_id,admitted,projection_version,
                           library_projection_json,library_projection_hash,
                           attempt_count,fencing_token)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            value["resultCode"],
                            value["resultHash"],
                            value["runCode"],
                            value["truthLabel"],
                            float(value["confidence"]),
                            float(value["score"]),
                            library_value["disposition"],
                            _encode(value),
                            payload_hash,
                            created_at,
                            None if fenced is None else fenced[0],
                            1 if fenced is None else 0,
                            RESULT_PROJECTION_VERSION,
                            _encode(projection),
                            stored_result_library_projection_hash_v1(
                                projection, payload_hash=payload_hash
                            ),
                            None if fenced is None else fenced[1],
                            None if fenced is None else fenced[2],
                        ),
                    )
                    result_code = str(value["resultCode"])
                    self._replace_result_facets(result_code, library_value)
                if occurrence is not None:
                    self._validate_occurrence_shard_locked(
                        run_code=str(value["runCode"]),
                        shard_id=str(occurrence["shardId"]),
                        attempt_count=fenced[1],
                        fencing_token=fenced[2],
                        worker_id=fenced[3],
                    )
                    self._put_fenced_occurrence_locked(
                        table="research_result_occurrences",
                        semantic_column="result_code",
                        semantic_id=result_code,
                        occurrence=occurrence,
                        now=now,
                    )
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
        return value

    def get_result(self, code: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                """SELECT result_code,result_hash,run_code,payload_json,payload_hash
                   FROM research_results WHERE result_code=? AND admitted=1""",
                (str(code),),
            ).fetchone()
        return (
            None
            if row is None
            else result_library_safe_payload(
                validate_stored_result_record_v1(
                    _decode(row["payload_json"]),
                    result_code=row["result_code"],
                    result_hash=row["result_hash"],
                    run_code=row["run_code"],
                    payload_hash=row["payload_hash"],
                )
            )
        )

    def list_results(
        self,
        *,
        run_code: str | None = None,
        truth_label: str | None = None,
        disposition: str | None = None,
        minimum_score: float | None = None,
        search: str | None = None,
        tactical_tag: str | None = None,
        acting_seat: str | None = None,
        observer_seat: str | None = None,
        contract: str | None = None,
        project: str | None = None,
        generation_lane: str | None = None,
        scenario_template: str | None = None,
        scenario_origin: str | None = None,
        scenario_status: str | None = None,
        requested_proof_mode: str | None = None,
        achieved_proof_mode: str | None = None,
        proof_completion: str | None = None,
        proof_visibility: str | None = None,
        target_team: str | None = None,
        contract_outcome: str | None = None,
        project_kind: str | None = None,
        project_value: str | None = None,
        project_state: str | None = None,
        winning_root_action_band: str | None = None,
        exact_state_available: str | None = None,
        decision_kind: str | None = None,
        research_backend: str | None = None,
        qualification_authority: str | None = None,
        runtime_compatibility_status: str | None = None,
        configuration_bundle: str | None = None,
        exact_state_id: str | None = None,
        scenario_category: str | None = None,
        variant_collection: str | None = None,
        tag_query: Mapping[str, Any] | None = None,
        created_from: str | None = None,
        created_to: str | None = None,
        group_by: str | None = None,
        sort: str = "score_desc",
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        query = ResearchResultQueryV1(
            run_code=run_code,
            truth_label=truth_label,
            disposition=disposition,
            minimum_score=minimum_score,
            search=search,
            tactical_tag=tactical_tag,
            acting_seat=acting_seat,
            observer_seat=observer_seat,
            contract=contract,
            project=project,
            generation_lane=generation_lane,
            scenario_template=scenario_template,
            scenario_origin=scenario_origin,
            scenario_status=scenario_status,
            requested_proof_mode=requested_proof_mode,
            achieved_proof_mode=achieved_proof_mode,
            proof_completion=proof_completion,
            proof_visibility=proof_visibility,
            target_team=target_team,
            contract_outcome=contract_outcome,
            project_kind=project_kind,
            project_value=project_value,
            project_state=project_state,
            winning_root_action_band=winning_root_action_band,
            exact_state_available=exact_state_available,
            decision_kind=decision_kind,
            research_backend=research_backend,
            qualification_authority=qualification_authority,
            runtime_compatibility_status=runtime_compatibility_status,
            configuration_bundle=configuration_bundle,
            exact_state_id=exact_state_id,
            scenario_category=scenario_category,
            variant_collection=variant_collection,
            tag_query=tag_query,
            created_from=created_from,
            created_to=created_to,
            group_by=group_by,
            sort=sort,
            offset=offset,
            limit=limit,
        )
        clauses: list[str] = ["r.admitted=1"]
        params: list[Any] = []
        for column, value in (
            ("run_code", query.run_code),
            ("truth_label", query.truth_label),
            ("disposition", query.disposition),
        ):
            if value is not None:
                clauses.append(f"r.{column}=?")
                params.append(value)
        if query.minimum_score is not None:
            clauses.append("r.score>=?")
            params.append(float(query.minimum_score))
        if query.search:
            clauses.append("r.payload_json LIKE ?")
            params.append(f"%{query.search}%")
        if query.created_from is not None:
            clauses.append("r.created_at>=?")
            params.append(query.created_from)
        if query.created_to is not None:
            clauses.append("r.created_at<=?")
            params.append(query.created_to)
        for index, (facet_kind, facet_value) in enumerate(query.facet_filters):
            clauses.append(
                f"""EXISTS(SELECT 1 FROM research_result_facets f{index}
                    WHERE f{index}.result_code=r.result_code
                    AND f{index}.facet_kind=? AND f{index}.facet_value=?)"""
            )
            params.extend((facet_kind, facet_value))
        for index, (facet_kind, values) in enumerate(query.tag_facet_filters):
            placeholders = ",".join("?" for _ in values)
            clauses.append(
                f"""EXISTS(SELECT 1 FROM research_result_facets tag{index}
                    WHERE tag{index}.result_code=r.result_code
                    AND tag{index}.facet_kind=?
                    AND tag{index}.facet_value IN ({placeholders}))"""
            )
            params.extend((facet_kind, *values))
        where = " WHERE " + " AND ".join(clauses)
        with self._lock:
            total = int(
                self._db.execute(
                    f"SELECT COUNT(*) AS count FROM research_results r{where}", params
                ).fetchone()["count"]
            )
            rows = self._db.execute(
                f"SELECT r.result_code,r.result_hash,r.run_code,r.payload_json,r.payload_hash FROM research_results r{where} ORDER BY {query.order_sql('r')} LIMIT ? OFFSET ?",
                [*params, int(query.limit), int(query.offset)],
            ).fetchall()
            groups: list[dict[str, Any]] = []
            if query.group_by in {"truth_label", "disposition"}:
                group_rows = self._db.execute(
                    f"""SELECT r.{query.group_by} AS value,COUNT(*) AS count
                        FROM research_results r{where}
                        GROUP BY r.{query.group_by}
                        ORDER BY count DESC,value ASC""",
                    params,
                ).fetchall()
                groups = [
                    {"value": str(row["value"]), "count": int(row["count"])}
                    for row in group_rows
                ]
            elif query.group_by is not None:
                group_rows = self._db.execute(
                    f"""SELECT grouped.facet_value AS value,
                        COUNT(DISTINCT r.result_code) AS count
                        FROM research_results r
                        JOIN research_result_facets grouped
                          ON grouped.result_code=r.result_code
                         AND grouped.facet_kind=?
                        {where}
                        GROUP BY grouped.facet_value
                        ORDER BY count DESC,value ASC""",
                    [query.group_by, *params],
                ).fetchall()
                groups = [
                    {"value": str(row["value"]), "count": int(row["count"])}
                    for row in group_rows
                ]
        next_offset = int(query.offset) + len(rows)
        return {
            "items": [
                result_library_safe_payload(
                    validate_stored_result_record_v1(
                        _decode(row["payload_json"]),
                        result_code=row["result_code"],
                        result_hash=row["result_hash"],
                        run_code=row["run_code"],
                        payload_hash=row["payload_hash"],
                    )
                )
                for row in rows
            ],
            "total": total,
            "offset": int(query.offset),
            "limit": int(query.limit),
            "nextOffset": next_offset if next_offset < total else None,
            "hasMore": next_offset < total,
            "groupBy": query.group_by,
            "groups": groups,
        }

    def list_result_library_projections(
        self, *, offset: int = 0, limit: int = 500
    ) -> dict[str, Any]:
        """Return the compact immutable fields needed by the Foundry library."""

        with self._lock:
            total = int(
                self._db.execute(
                    "SELECT COUNT(*) AS count FROM research_results WHERE admitted=1"
                ).fetchone()["count"]
            )
            rows = self._db.execute(
                """SELECT result_code,result_hash,run_code,payload_hash,
                          library_projection_json,library_projection_hash
                   FROM research_results
                   WHERE admitted=1
                   ORDER BY created_at DESC,result_code ASC
                   LIMIT ? OFFSET ?""",
                (max(1, min(500, int(limit))), max(0, int(offset))),
            ).fetchall()

        items = [
            validate_stored_result_library_projection_hash_v1(
                _decode(row["library_projection_json"]),
                result_code=row["result_code"],
                result_hash=row["result_hash"],
                run_code=row["run_code"],
                payload_hash=row["payload_hash"],
                projection_hash=row["library_projection_hash"],
            )
            for row in rows
        ]
        next_offset = max(0, int(offset)) + len(items)
        return {
            "items": items,
            "total": total,
            "nextOffset": next_offset if next_offset < total else None,
        }

    def put_disagreement(
        self,
        payload: Mapping[str, Any],
        *,
        shard_id: str | None = None,
        attempt_count: int | None = None,
        fencing_token: int | None = None,
        worker_id: str | None = None,
    ) -> dict[str, Any]:
        value = copy.deepcopy(dict(payload))
        now = _now()
        fenced = normalize_fenced_attempt(
            shard_id=shard_id,
            attempt_count=attempt_count,
            fencing_token=fencing_token,
            worker_id=worker_id,
        )
        occurrence = (
            None
            if fenced is None
            else build_fenced_occurrence(
                kind="semantic_disagreement",
                semantic_hash=str(value["sha256"]),
                run_code=str(value["runCode"]),
                shard_id=fenced[0],
                attempt_count=fenced[1],
                fencing_token=fenced[2],
            )
        )
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                if fenced is None:
                    self._validate_unsharded_write_locked(str(value["runCode"]))
                row = self._db.execute(
                    """SELECT disagreement_id,payload_json
                       FROM semantic_disagreements
                       WHERE disagreement_hash=?""",
                    (value["sha256"],),
                ).fetchone()
                if row is not None:
                    if _decode(row["payload_json"]) != value:
                        raise RuntimeError("semantic disagreement CAS payload mismatch")
                    disagreement_id = str(row["disagreement_id"])
                else:
                    self._db.execute(
                        """INSERT INTO semantic_disagreements(
                           disagreement_id,disagreement_hash,run_code,category,
                           status,payload_json,created_at,updated_at,shard_id,
                           admitted,attempt_count,fencing_token)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            value["disagreementId"],
                            value["sha256"],
                            value["runCode"],
                            value["category"],
                            value["status"],
                            _encode(value),
                            now,
                            now,
                            None if fenced is None else fenced[0],
                            1 if fenced is None else 0,
                            None if fenced is None else fenced[1],
                            None if fenced is None else fenced[2],
                        ),
                    )
                    disagreement_id = str(value["disagreementId"])
                if occurrence is not None:
                    self._validate_occurrence_shard_locked(
                        run_code=str(value["runCode"]),
                        shard_id=str(occurrence["shardId"]),
                        attempt_count=fenced[1],
                        fencing_token=fenced[2],
                        worker_id=fenced[3],
                    )
                    self._put_fenced_occurrence_locked(
                        table="semantic_disagreement_occurrences",
                        semantic_column="disagreement_id",
                        semantic_id=disagreement_id,
                        occurrence=occurrence,
                        now=now,
                    )
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
        return value

    def list_disagreements(
        self,
        *,
        run_code: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = ["admitted=1"]
        params: list[Any] = []
        if run_code:
            clauses.append("run_code=?")
            params.append(run_code)
        if status:
            clauses.append("status=?")
            params.append(status)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._lock:
            rows = self._db.execute(
                f"SELECT payload_json FROM semantic_disagreements{where} ORDER BY created_at DESC LIMIT ?",
                [*params, int(limit)],
            ).fetchall()
        return [_decode(row["payload_json"]) for row in rows]

    def aggregate_stats(self, *, run_code: str | None = None) -> dict[str, Any]:
        result_where = " WHERE admitted=1" + (
            " AND run_code=?" if run_code is not None else ""
        )
        result_args: list[Any] = [str(run_code)] if run_code is not None else []
        run_where = " WHERE run_code=?" if run_code is not None else ""
        with self._lock:
            run_rows = self._db.execute(
                f"SELECT status,COUNT(*) AS count FROM research_runs{run_where} GROUP BY status",
                result_args,
            ).fetchall()
            disposition_rows = self._db.execute(
                f"SELECT disposition,COUNT(*) AS count FROM research_results{result_where} GROUP BY disposition",
                result_args,
            ).fetchall()
            truth_rows = self._db.execute(
                f"SELECT truth_label,COUNT(*) AS count FROM research_results{result_where} GROUP BY truth_label",
                result_args,
            ).fetchall()
            score_row = self._db.execute(
                f"SELECT COUNT(*) AS count,AVG(score) AS average_score,AVG(confidence) AS average_confidence FROM research_results{result_where}",
                result_args,
            ).fetchone()
            defect_rows = self._db.execute(
                f"SELECT category,COUNT(*) AS count FROM semantic_disagreements{result_where} GROUP BY category",
                result_args,
            ).fetchall()
        semantic_defects: dict[str, int] = {}
        historical_records: dict[str, int] = {}
        for row in defect_rows:
            category, semantic = effective_disagreement_category(str(row["category"]))
            target = semantic_defects if semantic else historical_records
            target[category] = target.get(category, 0) + int(row["count"])
        return {
            "runStatus": {str(row["status"]): int(row["count"]) for row in run_rows},
            "resultDisposition": {
                str(row["disposition"]): int(row["count"]) for row in disposition_rows
            },
            "truth": {str(row["truth_label"]): int(row["count"]) for row in truth_rows},
            "resultCount": int(score_row["count"]),
            "averageScore": float(score_row["average_score"] or 0.0),
            "averageConfidence": float(score_row["average_confidence"] or 0.0),
            "semanticDefects": semantic_defects,
            "historicalRecords": historical_records,
        }


from .postgres_store import PostgresResearchStoreV1  # noqa: E402

__all__ = ["SQLiteResearchStoreV1", "PostgresResearchStoreV1"]
