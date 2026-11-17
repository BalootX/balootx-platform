from __future__ import annotations

import copy
import hashlib
import uuid

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Iterator, Mapping

from baloot_platform.storage.request_context import (
    apply_postgres_identity_context,
    contextual_connection,
)

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

_OPERATOR_RUN_PAYLOAD_SQL = (
    "CASE WHEN jsonb_typeof(payload_json->'performanceReceipt') = 'object' "
    "AND (payload_json->'performanceReceipt') ? 'coordinatorReceipt' "
    "THEN payload_json - 'performanceReceipt' ELSE payload_json END"
)

_BOOTSTRAP_LOCK_KEY = "balootx:research-store-v1:bootstrap"
_BOOTSTRAP_TIMEOUT = "60s"
_SERIALIZABLE_RETRY_SQLSTATES = frozenset({"40001", "40P01"})
_SERIALIZABLE_ATTEMPTS = 3
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


class PostgresResearchStoreV1:
    """PostgreSQL implementation of the narrow research-store contract."""

    def __init__(self, dsn: str):
        try:
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                'Install with: pip install "baloot-platform[postgres]"'
            ) from exc
        self.pool = ConnectionPool(
            str(dsn),
            min_size=1,
            max_size=24,
            open=True,
            kwargs={"row_factory": dict_row},
        )
        try:
            self._bootstrap_schema()
        except BaseException:
            self.pool.close()
            raise

    def _bootstrap_schema(self) -> None:
        with contextual_connection(self.pool) as conn:
            for setting in ("lock_timeout", "statement_timeout"):
                conn.execute(
                    "SELECT set_config(%s,%s,true)",
                    (setting, _BOOTSTRAP_TIMEOUT),
                )
            conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
                (_BOOTSTRAP_LOCK_KEY,),
            )
            conn.execute("CREATE SCHEMA IF NOT EXISTS research")
            conn.execute("""CREATE TABLE IF NOT EXISTS research.recipes(
              recipe_code text PRIMARY KEY,recipe_hash text UNIQUE NOT NULL,
              status text NOT NULL,payload_json jsonb NOT NULL,
              created_at timestamptz NOT NULL DEFAULT now(),updated_at timestamptz NOT NULL DEFAULT now())""")
            conn.execute("""CREATE INDEX IF NOT EXISTS ix_research_recipes_page
              ON research.recipes(created_at DESC,recipe_code COLLATE "C" ASC)""")
            conn.execute("""CREATE INDEX IF NOT EXISTS ix_research_recipes_status_page
              ON research.recipes(status,created_at DESC,recipe_code COLLATE "C" ASC)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS research.recipe_mutation_receipts(
              idempotency_key text PRIMARY KEY,mutation_kind text NOT NULL,
              request_intent_hash text NOT NULL,
              recipe_code text NOT NULL REFERENCES research.recipes(recipe_code),
              response_status integer NOT NULL,response_json jsonb NOT NULL,
              created_at timestamptz NOT NULL DEFAULT now())""")
            conn.execute(
                """DO $$
                BEGIN
                  IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conname='ck_research_recipe_receipt_mutation_kind'
                      AND conrelid='research.recipe_mutation_receipts'::regclass
                  ) THEN
                    ALTER TABLE research.recipe_mutation_receipts
                      ADD CONSTRAINT ck_research_recipe_receipt_mutation_kind
                      CHECK(mutation_kind IN ('create','retire'));
                  END IF;
                  IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conname='ck_research_recipe_receipt_intent_hash'
                      AND conrelid='research.recipe_mutation_receipts'::regclass
                  ) THEN
                    ALTER TABLE research.recipe_mutation_receipts
                      ADD CONSTRAINT ck_research_recipe_receipt_intent_hash
                      CHECK(request_intent_hash ~ '^[0-9a-f]{64}$');
                  END IF;
                  IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conname='ck_research_recipe_receipt_response_status'
                      AND conrelid='research.recipe_mutation_receipts'::regclass
                  ) THEN
                    ALTER TABLE research.recipe_mutation_receipts
                      ADD CONSTRAINT ck_research_recipe_receipt_response_status
                      CHECK(
                        (mutation_kind='create' AND response_status=201)
                        OR (mutation_kind='retire' AND response_status=200)
                      );
                  END IF;
                END $$"""
            )
            conn.execute("""CREATE TABLE IF NOT EXISTS research.runs(
              run_code text PRIMARY KEY,run_hash text UNIQUE NOT NULL,
              recipe_code text NOT NULL REFERENCES research.recipes(recipe_code),
              recipe_hash text NOT NULL,status text NOT NULL,progress double precision NOT NULL,
              payload_json jsonb NOT NULL,created_at timestamptz NOT NULL DEFAULT now(),
              updated_at timestamptz NOT NULL DEFAULT now())""")
            conn.execute("""CREATE TABLE IF NOT EXISTS research.results(
              result_code text PRIMARY KEY,result_hash text UNIQUE NOT NULL,
              run_code text NOT NULL REFERENCES research.runs(run_code),truth_label text NOT NULL,
              confidence double precision NOT NULL,score double precision NOT NULL,
              disposition text NOT NULL,payload_json jsonb NOT NULL,
              payload_hash text,
              library_projection_json jsonb,
              library_projection_hash text,
              attempt_count integer,fencing_token bigint,
              CONSTRAINT ck_research_results_legacy_fence CHECK(
                (attempt_count IS NULL AND fencing_token IS NULL) OR
                (attempt_count IS NOT NULL AND fencing_token IS NOT NULL AND
                 attempt_count >= 1 AND fencing_token >= 1)),
              created_at timestamptz NOT NULL DEFAULT now())""")
            conn.execute("""CREATE INDEX IF NOT EXISTS ix_research_results_rank
              ON research.results(run_code,disposition,score DESC,result_code)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS research.result_facets(
              result_code text NOT NULL REFERENCES research.results(result_code),
              facet_kind text NOT NULL,facet_value text NOT NULL,
              PRIMARY KEY(result_code,facet_kind,facet_value))""")
            conn.execute("""CREATE INDEX IF NOT EXISTS ix_research_result_facets_lookup
              ON research.result_facets(facet_kind,facet_value,result_code)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS research.semantic_disagreements(
              disagreement_id text PRIMARY KEY,disagreement_hash text UNIQUE NOT NULL,
              run_code text NOT NULL REFERENCES research.runs(run_code),category text NOT NULL,
              status text NOT NULL,payload_json jsonb NOT NULL,
              attempt_count integer,fencing_token bigint,
              CONSTRAINT ck_research_disagreements_legacy_fence CHECK(
                (attempt_count IS NULL AND fencing_token IS NULL) OR
                (attempt_count IS NOT NULL AND fencing_token IS NOT NULL AND
                 attempt_count >= 1 AND fencing_token >= 1)),
              created_at timestamptz NOT NULL DEFAULT now(),updated_at timestamptz NOT NULL DEFAULT now())""")
            conn.execute("""CREATE TABLE IF NOT EXISTS research.comparison_campaigns(
              campaign_code text PRIMARY KEY,campaign_hash text UNIQUE NOT NULL,
              base_recipe_code text NOT NULL REFERENCES research.recipes(recipe_code),payload_json jsonb NOT NULL,
              created_at timestamptz NOT NULL DEFAULT now(),updated_at timestamptz NOT NULL DEFAULT now())""")
            conn.execute("""CREATE TABLE IF NOT EXISTS research.domain_compilation_artifacts(
              compilation_hash text PRIMARY KEY,
              artifact_sha256 text UNIQUE NOT NULL,
              record_json jsonb NOT NULL,
              created_at timestamptz NOT NULL DEFAULT now())""")
            conn.execute("""CREATE TABLE IF NOT EXISTS research.audit_traces(
              record_hash text PRIMARY KEY,
              audit_trace_code text UNIQUE NOT NULL,
              trace_hash text NOT NULL,
              artifact_sha256 text NOT NULL,
              source_occurrence_hash text NOT NULL,
              run_code text NOT NULL,
              shard_id text,
              attempt_count integer NOT NULL,
              fencing_token bigint NOT NULL,
              candidate_hash text NOT NULL,
              candidate_ordinal integer NOT NULL,
              seed bigint NOT NULL,
              disposition text NOT NULL,
              status text NOT NULL,
              attempt_status text NOT NULL,
              record_json jsonb NOT NULL,
              admitted boolean NOT NULL DEFAULT true,
              created_at timestamptz NOT NULL DEFAULT now(),
              updated_at timestamptz NOT NULL DEFAULT now(),
              CONSTRAINT fk_research_audit_traces_run
                FOREIGN KEY(run_code) REFERENCES research.runs(run_code)
                ON DELETE RESTRICT)""")
            conn.execute(
                """DO $$
                BEGIN
                  IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint AS existing_fk
                    WHERE existing_fk.contype='f'
                      AND existing_fk.conrelid='research.audit_traces'::regclass
                      AND existing_fk.confrelid='research.runs'::regclass
                      AND existing_fk.conkey=ARRAY[(
                        SELECT attnum FROM pg_attribute
                        WHERE attrelid='research.audit_traces'::regclass
                          AND attname='run_code' AND NOT attisdropped
                      )]::smallint[]
                      AND existing_fk.confkey=ARRAY[(
                        SELECT attnum FROM pg_attribute
                        WHERE attrelid='research.runs'::regclass
                          AND attname='run_code' AND NOT attisdropped
                      )]::smallint[]
                  ) THEN
                    ALTER TABLE research.audit_traces
                      ADD CONSTRAINT fk_research_audit_traces_run
                      FOREIGN KEY(run_code) REFERENCES research.runs(run_code)
                      ON DELETE RESTRICT;
                  END IF;
                END $$"""
            )
            conn.execute("""CREATE TABLE IF NOT EXISTS research.shards(
              shard_id text PRIMARY KEY,shard_hash text UNIQUE NOT NULL,
              run_code text NOT NULL REFERENCES research.runs(run_code),shard_index integer NOT NULL,
              status text NOT NULL,attempt_count integer NOT NULL DEFAULT 0,
              retry_count integer NOT NULL DEFAULT 0,
              reclaim_count integer NOT NULL DEFAULT 0,
              operation_counters_available boolean NOT NULL DEFAULT true,
              fencing_token bigint NOT NULL DEFAULT 0,lease_owner text,
              lease_expires_at timestamptz,heartbeat_at timestamptz,
              spec_json jsonb NOT NULL,result_json jsonb,error_json jsonb,
              created_at timestamptz NOT NULL DEFAULT now(),updated_at timestamptz NOT NULL DEFAULT now(),
              UNIQUE(run_code,shard_index))""")
            conn.execute("""CREATE INDEX IF NOT EXISTS ix_research_shards_claim
              ON research.shards(run_code,status,lease_expires_at,shard_index)""")
            conn.execute("""CREATE UNIQUE INDEX IF NOT EXISTS ux_research_shards_identity_run
              ON research.shards(shard_id,run_code)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS research.result_occurrences(
              occurrence_hash text PRIMARY KEY,
              result_code text NOT NULL REFERENCES research.results(result_code) ON DELETE RESTRICT,
              run_code text NOT NULL REFERENCES research.runs(run_code) ON DELETE RESTRICT,
              shard_id text NOT NULL,
              attempt_count integer NOT NULL CHECK(attempt_count >= 1),
              fencing_token bigint NOT NULL CHECK(fencing_token >= 1),
              admitted boolean NOT NULL DEFAULT false,
              created_at timestamptz NOT NULL DEFAULT now(),
              updated_at timestamptz NOT NULL DEFAULT now(),
              UNIQUE(result_code,shard_id,attempt_count,fencing_token),
              CONSTRAINT fk_research_result_occurrence_shard_run
                FOREIGN KEY(shard_id,run_code)
                REFERENCES research.shards(shard_id,run_code) ON DELETE RESTRICT)""")
            conn.execute("""CREATE INDEX IF NOT EXISTS ix_research_result_occurrences_admission
              ON research.result_occurrences(
                shard_id,attempt_count,fencing_token,admitted)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS research.semantic_disagreement_occurrences(
              occurrence_hash text PRIMARY KEY,
              disagreement_id text NOT NULL REFERENCES research.semantic_disagreements(disagreement_id) ON DELETE RESTRICT,
              run_code text NOT NULL REFERENCES research.runs(run_code) ON DELETE RESTRICT,
              shard_id text NOT NULL,
              attempt_count integer NOT NULL CHECK(attempt_count >= 1),
              fencing_token bigint NOT NULL CHECK(fencing_token >= 1),
              admitted boolean NOT NULL DEFAULT false,
              created_at timestamptz NOT NULL DEFAULT now(),
              updated_at timestamptz NOT NULL DEFAULT now(),
              UNIQUE(disagreement_id,shard_id,attempt_count,fencing_token),
              CONSTRAINT fk_research_disagreement_occurrence_shard_run
                FOREIGN KEY(shard_id,run_code)
                REFERENCES research.shards(shard_id,run_code) ON DELETE RESTRICT)""")
            conn.execute("""CREATE INDEX IF NOT EXISTS ix_research_disagreement_occurrences_admission
              ON research.semantic_disagreement_occurrences(
                shard_id,attempt_count,fencing_token,admitted)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS research.sequential_search_receipt_bundles(
              receipt_hash text PRIMARY KEY,seed bigint NOT NULL,
              receipt_json jsonb NOT NULL,
              created_at timestamptz NOT NULL DEFAULT now())""")
            conn.execute("""CREATE UNIQUE INDEX IF NOT EXISTS ux_research_receipt_bundle_hash_seed
              ON research.sequential_search_receipt_bundles(receipt_hash,seed)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS research.sequential_search_receipt_occurrences(
              occurrence_hash text PRIMARY KEY,
              receipt_hash text NOT NULL,
              run_code text NOT NULL REFERENCES research.runs(run_code) ON DELETE RESTRICT,
              shard_id text NOT NULL,
              attempt_count integer NOT NULL CHECK(attempt_count >= 1),
              fencing_token bigint NOT NULL CHECK(fencing_token >= 1),
              seed bigint NOT NULL,admitted boolean NOT NULL DEFAULT false,
              created_at timestamptz NOT NULL DEFAULT now(),
              updated_at timestamptz NOT NULL DEFAULT now(),
              UNIQUE(run_code,shard_id,attempt_count,fencing_token,seed),
              CONSTRAINT fk_research_receipt_occurrence_bundle_seed
                FOREIGN KEY(receipt_hash,seed)
                REFERENCES research.sequential_search_receipt_bundles(receipt_hash,seed)
                ON DELETE RESTRICT,
              CONSTRAINT fk_research_receipt_occurrence_shard_run
                FOREIGN KEY(shard_id,run_code)
                REFERENCES research.shards(shard_id,run_code) ON DELETE RESTRICT)""")
            conn.execute("""CREATE INDEX IF NOT EXISTS ix_research_search_receipts_library
              ON research.sequential_search_receipt_occurrences(
                admitted,run_code,created_at DESC,occurrence_hash)""")
            conn.execute("""CREATE INDEX IF NOT EXISTS ix_research_search_receipts_admission
              ON research.sequential_search_receipt_occurrences(
                shard_id,attempt_count,fencing_token,admitted)""")
            conn.execute("""CREATE UNIQUE INDEX IF NOT EXISTS ux_research_search_receipts_admitted_seed
              ON research.sequential_search_receipt_occurrences(run_code,seed)
              WHERE admitted""")
            conn.execute(
                "ALTER TABLE research.results ADD COLUMN IF NOT EXISTS shard_id text"
            )
            conn.execute(
                "ALTER TABLE research.results ADD COLUMN IF NOT EXISTS admitted boolean NOT NULL DEFAULT true"
            )
            conn.execute(
                "ALTER TABLE research.results ADD COLUMN IF NOT EXISTS projection_version integer NOT NULL DEFAULT 5"
            )
            conn.execute(
                "ALTER TABLE research.results ALTER COLUMN projection_version SET DEFAULT 5"
            )
            conn.execute(
                "ALTER TABLE research.results ADD COLUMN IF NOT EXISTS library_projection_json jsonb"
            )
            conn.execute(
                "ALTER TABLE research.results ADD COLUMN IF NOT EXISTS payload_hash text"
            )
            conn.execute(
                "ALTER TABLE research.results ADD COLUMN IF NOT EXISTS library_projection_hash text"
            )
            conn.execute(
                """CREATE INDEX IF NOT EXISTS ix_research_results_library
                   ON research.results(admitted,created_at DESC,result_code ASC)"""
            )
            conn.execute(
                "ALTER TABLE research.results ADD COLUMN IF NOT EXISTS attempt_count integer"
            )
            conn.execute(
                "ALTER TABLE research.results ADD COLUMN IF NOT EXISTS fencing_token bigint"
            )
            conn.execute(
                "ALTER TABLE research.semantic_disagreements ADD COLUMN IF NOT EXISTS shard_id text"
            )
            conn.execute(
                "ALTER TABLE research.semantic_disagreements ADD COLUMN IF NOT EXISTS admitted boolean NOT NULL DEFAULT true"
            )
            conn.execute(
                "ALTER TABLE research.semantic_disagreements ADD COLUMN IF NOT EXISTS attempt_count integer"
            )
            conn.execute(
                "ALTER TABLE research.semantic_disagreements ADD COLUMN IF NOT EXISTS fencing_token bigint"
            )
            conn.execute(
                """DO $$ BEGIN
                  IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conname='ck_research_results_legacy_fence'
                      AND conrelid='research.results'::regclass
                  ) THEN
                    ALTER TABLE research.results
                      ADD CONSTRAINT ck_research_results_legacy_fence CHECK(
                        (attempt_count IS NULL AND fencing_token IS NULL) OR
                        (attempt_count IS NOT NULL AND fencing_token IS NOT NULL AND
                         attempt_count >= 1 AND fencing_token >= 1));
                  END IF;
                  IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conname='ck_research_disagreements_legacy_fence'
                      AND conrelid='research.semantic_disagreements'::regclass
                  ) THEN
                    ALTER TABLE research.semantic_disagreements
                      ADD CONSTRAINT ck_research_disagreements_legacy_fence CHECK(
                        (attempt_count IS NULL AND fencing_token IS NULL) OR
                        (attempt_count IS NOT NULL AND fencing_token IS NOT NULL AND
                         attempt_count >= 1 AND fencing_token >= 1));
                  END IF;
                END $$"""
            )
            conn.execute(
                "ALTER TABLE research.shards ADD COLUMN IF NOT EXISTS retry_count integer NOT NULL DEFAULT 0"
            )
            conn.execute(
                "ALTER TABLE research.shards ADD COLUMN IF NOT EXISTS reclaim_count integer NOT NULL DEFAULT 0"
            )
            conn.execute(
                "ALTER TABLE research.shards ADD COLUMN IF NOT EXISTS operation_counters_available boolean"
            )
            conn.execute(
                "ALTER TABLE research.audit_traces "
                "ADD COLUMN IF NOT EXISTS source_occurrence_hash text"
            )
            conn.execute(
                "ALTER TABLE research.audit_traces "
                "ADD COLUMN IF NOT EXISTS candidate_ordinal integer"
            )
            conn.execute(
                "ALTER TABLE research.audit_traces ADD COLUMN IF NOT EXISTS seed bigint"
            )
            audit_rows = conn.execute(
                """SELECT record_hash,record_json FROM research.audit_traces
                   WHERE source_occurrence_hash IS NULL
                      OR candidate_ordinal IS NULL OR seed IS NULL
                   FOR UPDATE"""
            ).fetchall()
            for audit_row in audit_rows:
                audit_record = ResearchAuditTraceRecordV1.build(
                    _decode(audit_row["record_json"])
                )
                conn.execute(
                    """UPDATE research.audit_traces
                       SET source_occurrence_hash=%s,candidate_ordinal=%s,seed=%s
                       WHERE record_hash=%s""",
                    (
                        self._audit_trace_source_occurrence_hash(audit_record),
                        int(audit_record.source["candidateOrdinal"]),
                        int(audit_record.source["seed"]),
                        str(audit_row["record_hash"]),
                    ),
                )
            conn.execute(
                "ALTER TABLE research.audit_traces "
                "ALTER COLUMN source_occurrence_hash SET NOT NULL"
            )
            conn.execute(
                "ALTER TABLE research.audit_traces "
                "ALTER COLUMN candidate_ordinal SET NOT NULL"
            )
            conn.execute(
                "ALTER TABLE research.audit_traces ALTER COLUMN seed SET NOT NULL"
            )
            conn.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS ux_research_audit_trace_source
                   ON research.audit_traces(source_occurrence_hash)"""
            )
            conn.execute("""CREATE INDEX IF NOT EXISTS ix_research_audit_traces_library
              ON research.audit_traces(admitted,disposition,created_at DESC,trace_hash)""")
            conn.execute("""CREATE INDEX IF NOT EXISTS ix_research_audit_traces_trace
              ON research.audit_traces(trace_hash,created_at DESC,record_hash)""")
            rows = conn.execute(
                """SELECT result_code,result_hash,run_code,payload_json,payload_hash,created_at FROM research.results
                   WHERE projection_version<%s OR library_projection_json IS NULL
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
                self._replace_result_facets(
                    conn, str(row["result_code"]), library_value
                )
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
                conn.execute(
                    """UPDATE research.results
                       SET created_at=%s,disposition=%s,payload_hash=%s,projection_version=%s,
                            library_projection_json=%s::jsonb,library_projection_hash=%s
                       WHERE result_code=%s""",
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
            projection_rows = conn.execute(
                """SELECT result_code,result_hash,run_code,payload_hash,library_projection_json
                   FROM research.results WHERE library_projection_hash IS NULL
                   ORDER BY result_code"""
            ).fetchall()
            for row in projection_rows:
                projection = validate_stored_result_identity_v1(
                    _decode(row["library_projection_json"]),
                    result_code=row["result_code"],
                    result_hash=row["result_hash"],
                    run_code=row["run_code"],
                )
                conn.execute(
                    """UPDATE research.results SET library_projection_hash=%s
                       WHERE result_code=%s""",
                    (
                        stored_result_library_projection_hash_v1(
                            projection, payload_hash=row["payload_hash"]
                        ),
                        str(row["result_code"]),
                    ),
                )

    def _run_serializable(
        self,
        operation: Callable[[Any], Any],
        *,
        advisory_lock_key: str,
    ) -> Any:
        """Run one bounded transaction behind a pre-snapshot session lock."""
        with self.pool.connection() as conn:
            try:
                conn.execute(
                    "SELECT pg_advisory_lock(hashtextextended(%s,0))",
                    (str(advisory_lock_key),),
                )
                conn.commit()
                for attempt in range(_SERIALIZABLE_ATTEMPTS):
                    try:
                        with conn.transaction():
                            conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
                            apply_postgres_identity_context(conn)
                            return operation(conn)
                    except Exception as exc:
                        sqlstate = getattr(exc, "sqlstate", None)
                        if (
                            sqlstate not in _SERIALIZABLE_RETRY_SQLSTATES
                            or attempt + 1 >= _SERIALIZABLE_ATTEMPTS
                        ):
                            raise
                raise RuntimeError("unreachable PostgreSQL serializable retry state")
            finally:
                try:
                    conn.rollback()
                    conn.execute(
                        "SELECT pg_advisory_unlock(hashtextextended(%s,0))",
                        (str(advisory_lock_key),),
                    )
                    conn.commit()
                except Exception:
                    conn.close()
                    raise

    @staticmethod
    def _replace_result_facets(
        conn: Any, result_code: str, payload: Mapping[str, Any]
    ) -> None:
        conn.execute(
            "DELETE FROM research.result_facets WHERE result_code=%s",
            (str(result_code),),
        )
        rows = [
            (str(result_code), str(facet_kind), str(value))
            for facet_kind, values in result_facets(payload).items()
            for value in values
        ]
        if rows:
            with conn.cursor() as cursor:
                cursor.executemany(
                    """INSERT INTO research.result_facets(
                       result_code,facet_kind,facet_value) VALUES(%s,%s,%s)
                       ON CONFLICT DO NOTHING""",
                    rows,
                )

    def close(self) -> None:
        self.pool.close()

    def put_domain_compilation_artifact(
        self, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        requested = DomainCompilationArtifactV1.build(payload)
        created_at = requested.created_at or datetime.now(timezone.utc).isoformat(
            timespec="microseconds"
        )
        value = DomainCompilationArtifactV1(
            requested.compilation_hash,
            requested.artifact,
            requested.bundle_hash,
            requested.effective_runtime_config_hash,
            requested.context_hash,
            requested.domain_hash,
            created_at,
        )

        def _write(conn: Any) -> dict[str, Any]:
            row = conn.execute(
                """INSERT INTO research.domain_compilation_artifacts(
                   compilation_hash,artifact_sha256,record_json,created_at)
                   VALUES(%s,%s,%s::jsonb,%s::timestamptz)
                   ON CONFLICT (compilation_hash) DO NOTHING
                   RETURNING record_json""",
                (
                    value.compilation_hash,
                    value.artifact["sha256"],
                    _encode(value.to_dict()),
                    created_at,
                ),
            ).fetchone()
            if row is None:
                row = conn.execute(
                    """SELECT record_json
                       FROM research.domain_compilation_artifacts
                       WHERE compilation_hash=%s FOR UPDATE""",
                    (value.compilation_hash,),
                ).fetchone()
            if row is None:
                raise RuntimeError("domain-compilation artifact index write failed")
            current = DomainCompilationArtifactV1.build(_decode(row["record_json"]))
            if (
                current.artifact["sha256"] != value.artifact["sha256"]
                or current.artifact["bytes"] != value.artifact["bytes"]
                or current.artifact["mediaType"] != value.artifact["mediaType"]
            ):
                raise ValueError("domain-compilation artifact identity is immutable")
            return current.to_dict()

        return self._run_serializable(
            _write,
            advisory_lock_key=(f"research:domain-compilation:{value.compilation_hash}"),
        )

    def get_domain_compilation_artifact(
        self, compilation_hash: str
    ) -> dict[str, Any] | None:
        with contextual_connection(self.pool) as conn:
            row = conn.execute(
                """SELECT record_json
                   FROM research.domain_compilation_artifacts
                   WHERE compilation_hash=%s""",
                (str(compilation_hash),),
            ).fetchone()
        if row is None:
            return None
        record = DomainCompilationArtifactV1.build(_decode(row["record_json"]))
        if record.compilation_hash != str(compilation_hash):
            raise ValueError("domain-compilation artifact index key mismatch")
        return record.to_dict()

    @staticmethod
    def _audit_trace_source_occurrence_hash(
        record: ResearchAuditTraceRecordV1,
    ) -> str:
        return hashlib.sha256(_encode(record.source).encode("utf-8")).hexdigest()

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
    def _update_audit_trace_attempts(
        conn: Any,
        *,
        shard_id: str,
        fencing_token: int,
        attempt_status: str,
        failure_categories: Iterable[str] = (),
    ) -> None:
        rows = conn.execute(
            """SELECT record_hash,record_json FROM research.audit_traces
               WHERE shard_id=%s AND fencing_token=%s FOR UPDATE""",
            (str(shard_id), int(fencing_token)),
        ).fetchall()
        now = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        failures = tuple(
            sorted(
                {
                    str(value).strip()
                    for value in failure_categories
                    if str(value).strip()
                }
            )
        )
        for row in rows:
            current = ResearchAuditTraceRecordV1.build(_decode(row["record_json"]))
            if str(current.source["shardId"]) != str(shard_id) or int(
                current.source["fencingToken"]
            ) != int(fencing_token):
                raise ValueError("Research audit trace attempt index mismatch")
            value = current.with_attempt_status(str(attempt_status))
            for failure_category in failures:
                value = value.with_attempt_status(
                    str(attempt_status), failure_category=failure_category
                )
            value = ResearchAuditTraceRecordV1.build(
                {
                    **value.to_dict(),
                    "createdAt": current.created_at,
                    "updatedAt": now,
                }
            )
            conn.execute(
                """UPDATE research.audit_traces
                   SET attempt_status=%s,record_json=%s::jsonb,
                       updated_at=%s::timestamptz
                   WHERE record_hash=%s AND fencing_token=%s""",
                (
                    value.attempt_status,
                    _encode(value.to_dict()),
                    now,
                    value.record_hash,
                    int(fencing_token),
                ),
            )

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
        now = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        value = ResearchAuditTraceRecordV1.build(
            {**requested.to_dict(), "createdAt": now, "updatedAt": now}
        )
        source_occurrence_hash = self._audit_trace_source_occurrence_hash(value)

        def _write(conn: Any) -> dict[str, Any]:
            lease = conn.execute(
                """SELECT status,attempt_count,fencing_token,lease_owner,
                   lease_expires_at,run_code,shard_hash,shard_index
                   FROM research.shards
                   WHERE shard_id=%s FOR UPDATE""",
                (source_shard,),
            ).fetchone()
            moment = datetime.now(timezone.utc)
            if (
                lease is None
                or str(lease["status"]) != "running"
                or int(lease["attempt_count"]) != int(requested.source["attemptCount"])
                or int(lease["fencing_token"]) != int(fencing_token)
                or str(lease["lease_owner"] or "") != str(worker_id)
                or lease["lease_expires_at"] is None
                or lease["lease_expires_at"] <= moment
                or str(lease["run_code"]) != str(requested.source["runCode"])
                or str(lease["shard_hash"]) != str(requested.source["shardHash"])
                or int(lease["shard_index"]) != int(requested.source["shardIndex"])
            ):
                raise RuntimeError("stale Research shard audit-trace admission")
            row = conn.execute(
                """INSERT INTO research.audit_traces(
                   record_hash,audit_trace_code,trace_hash,artifact_sha256,
                   source_occurrence_hash,run_code,shard_id,attempt_count,
                   fencing_token,candidate_hash,candidate_ordinal,seed,
                   disposition,status,attempt_status,record_json,admitted,
                   created_at,updated_at)
                   VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,true,
                           %s::timestamptz,%s::timestamptz)
                   ON CONFLICT DO NOTHING
                   RETURNING record_json,source_occurrence_hash""",
                (
                    value.record_hash,
                    value.audit_trace_code,
                    value.trace_hash,
                    value.complete_trace["sha256"],
                    source_occurrence_hash,
                    value.source["runCode"],
                    source_shard,
                    value.source["attemptCount"],
                    value.source["fencingToken"],
                    value.source["candidateHash"],
                    value.source["candidateOrdinal"],
                    value.source["seed"],
                    value.disposition,
                    value.status,
                    value.attempt_status,
                    _encode(value.to_dict()),
                    now,
                    now,
                ),
            ).fetchone()
            if row is None:
                row = conn.execute(
                    """SELECT record_json,source_occurrence_hash
                       FROM research.audit_traces
                       WHERE record_hash=%s OR source_occurrence_hash=%s
                       FOR UPDATE""",
                    (value.record_hash, source_occurrence_hash),
                ).fetchone()
            if row is None:
                raise RuntimeError("Research audit trace index write failed")
            current = ResearchAuditTraceRecordV1.build(_decode(row["record_json"]))
            expected_occurrence = self._audit_trace_source_occurrence_hash(current)
            if str(row["source_occurrence_hash"]) != expected_occurrence:
                raise ValueError(
                    "Research audit trace source occurrence index mismatch"
                )
            if self._audit_trace_identity(current) != self._audit_trace_identity(value):
                raise ValueError("Research audit trace source occurrence is immutable")
            return current.to_dict()

        return self._run_serializable(
            _write,
            advisory_lock_key=f"research:audit-trace:{source_occurrence_hash}",
        )

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
        source_occurrence_hash = self._audit_trace_source_occurrence_hash(requested)
        updated_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")

        def _write(conn: Any) -> dict[str, Any]:
            lease = conn.execute(
                """SELECT status,attempt_count,fencing_token,lease_owner,
                   lease_expires_at,run_code,shard_hash,shard_index
                   FROM research.shards
                   WHERE shard_id=%s FOR UPDATE""",
                (str(requested.source["shardId"]),),
            ).fetchone()
            moment = datetime.now(timezone.utc)
            if (
                lease is None
                or str(lease["status"]) != "running"
                or int(lease["attempt_count"]) != int(requested.source["attemptCount"])
                or int(lease["fencing_token"]) != int(fencing_token)
                or str(lease["lease_owner"] or "") != str(worker_id)
                or lease["lease_expires_at"] is None
                or lease["lease_expires_at"] <= moment
                or str(lease["run_code"]) != str(requested.source["runCode"])
                or str(lease["shard_hash"]) != str(requested.source["shardHash"])
                or int(lease["shard_index"]) != int(requested.source["shardIndex"])
            ):
                raise RuntimeError("stale Research shard audit-trace finalization")
            row = conn.execute(
                """SELECT record_json,source_occurrence_hash
                   FROM research.audit_traces
                   WHERE record_hash=%s FOR UPDATE""",
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
                return current.to_dict()
            if requested.status != "finalized":
                raise ValueError(
                    "Research audit trace update must finalize its lifecycle"
                )
            value = ResearchAuditTraceRecordV1.build(
                {
                    **requested.to_dict(),
                    "createdAt": current.created_at,
                    "updatedAt": updated_at,
                }
            )
            conn.execute(
                """UPDATE research.audit_traces
                   SET disposition=%s,status=%s,attempt_status=%s,
                       record_json=%s::jsonb,updated_at=%s::timestamptz
                   WHERE record_hash=%s""",
                (
                    value.disposition,
                    value.status,
                    value.attempt_status,
                    _encode(value.to_dict()),
                    updated_at,
                    value.record_hash,
                ),
            )
            return value.to_dict()

        return self._run_serializable(
            _write,
            advisory_lock_key=f"research:audit-trace:{source_occurrence_hash}",
        )

    def get_audit_trace(
        self, identifier: str, *, include_unadmitted: bool = False
    ) -> dict[str, Any] | None:
        admitted = "" if include_unadmitted else " AND admitted=true"
        with contextual_connection(self.pool) as conn:
            row = conn.execute(
                f"""SELECT record_json FROM research.audit_traces
                    WHERE (record_hash=%s OR trace_hash=%s OR audit_trace_code=%s)
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
        clauses = ["admitted=true"]
        parameters: list[Any] = []
        for column, value in (
            ("run_code", run_code),
            ("disposition", disposition),
            ("status", status),
        ):
            if value is not None:
                clauses.append(f"{column}=%s")
                parameters.append(str(value))
        where = " WHERE " + " AND ".join(clauses)
        start = max(0, int(offset))
        size = max(1, min(500, int(limit)))
        with contextual_connection(self.pool) as conn:
            total = int(
                conn.execute(
                    f"SELECT COUNT(*) AS count FROM research.audit_traces{where}",
                    parameters,
                ).fetchone()["count"]
            )
            rows = conn.execute(
                f"""SELECT record_json FROM research.audit_traces{where}
                    ORDER BY created_at DESC,trace_hash LIMIT %s OFFSET %s""",
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
        with contextual_connection(self.pool) as conn:
            row = conn.execute(
                """SELECT idempotency_key,mutation_kind,request_intent_hash,
                          recipe_code,response_status,response_json,created_at
                   FROM research.recipe_mutation_receipts
                   WHERE idempotency_key=%s""",
                (key,),
            ).fetchone()
        return None if row is None else self._recipe_mutation_receipt(row)

    def put_recipe(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        value = copy.deepcopy(dict(payload))
        with contextual_connection(self.pool) as conn:
            row = conn.execute(
                """SELECT payload_json FROM research.recipes
                   WHERE recipe_code=%s FOR UPDATE""",
                (value["code"],),
            ).fetchone()
            if row is not None:
                current = _decode(row["payload_json"])
                if current != value:
                    raise ValueError(
                        "research recipe code is immutable; create a new revision"
                    )
                return current
            conn.execute(
                """INSERT INTO research.recipes(
                     recipe_code,recipe_hash,status,payload_json)
                   VALUES(%s,%s,%s,%s::jsonb)""",
                (value["code"], value["sha256"], value["status"], _encode(value)),
            )
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
        with contextual_connection(self.pool) as conn:
            conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
                (key,),
            )
            receipt_row = conn.execute(
                """SELECT idempotency_key,mutation_kind,request_intent_hash,
                          recipe_code,response_status,response_json,created_at
                   FROM research.recipe_mutation_receipts
                   WHERE idempotency_key=%s""",
                (key,),
            ).fetchone()
            if receipt_row is not None:
                return replay_recipe_mutation_receipt(
                    self._recipe_mutation_receipt(receipt_row),
                    mutation_kind="create",
                    request_intent_hash=intent_hash,
                )

            recipe_row = conn.execute(
                """SELECT status,payload_json FROM research.recipes
                   WHERE recipe_code=%s FOR UPDATE""",
                (str(value["code"]),),
            ).fetchone()
            if recipe_row is None:
                conn.execute(
                    """INSERT INTO research.recipes(
                         recipe_code,recipe_hash,status,payload_json)
                       VALUES(%s,%s,%s,%s::jsonb)
                       ON CONFLICT(recipe_code) DO NOTHING""",
                    (
                        value["code"],
                        value["sha256"],
                        value["status"],
                        _encode(value),
                    ),
                )
                recipe_row = conn.execute(
                    """SELECT status,payload_json FROM research.recipes
                       WHERE recipe_code=%s FOR UPDATE""",
                    (str(value["code"]),),
                ).fetchone()
            if recipe_row is None:  # pragma: no cover - insert/select invariant
                raise RuntimeError("research recipe insert was not observable")
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
            conn.execute(
                """INSERT INTO research.recipe_mutation_receipts(
                     idempotency_key,mutation_kind,request_intent_hash,
                     recipe_code,response_status,response_json)
                   VALUES(%s,%s,%s,%s,%s,%s::jsonb)""",
                (
                    key,
                    "create",
                    intent_hash,
                    value["code"],
                    201,
                    _encode(response),
                ),
            )
        return response

    def get_recipe(self, code: str) -> dict[str, Any] | None:
        with contextual_connection(self.pool) as conn:
            row = conn.execute(
                "SELECT payload_json FROM research.recipes WHERE recipe_code=%s",
                (str(code),),
            ).fetchone()
        return None if row is None else _decode(row["payload_json"])

    def recipe_status(self, code: str) -> str | None:
        with contextual_connection(self.pool) as conn:
            row = conn.execute(
                "SELECT status FROM research.recipes WHERE recipe_code=%s", (str(code),)
            ).fetchone()
        return None if row is None else str(row["status"])

    def retire_recipe(self, code: str) -> dict[str, Any]:
        with contextual_connection(self.pool) as conn:
            row = conn.execute(
                """SELECT status,payload_json FROM research.recipes
                   WHERE recipe_code=%s FOR UPDATE""",
                (str(code),),
            ).fetchone()
            if row is None:
                raise KeyError(code)
            if str(row["status"]) == "draft":
                raise ValueError("a draft recipe cannot be retired")
            if str(row["status"]) != "retired":
                conn.execute(
                    """UPDATE research.recipes
                       SET status='retired',updated_at=now()
                       WHERE recipe_code=%s""",
                    (str(code),),
                )
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
        with contextual_connection(self.pool) as conn:
            conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
                (key,),
            )
            receipt_row = conn.execute(
                """SELECT idempotency_key,mutation_kind,request_intent_hash,
                          recipe_code,response_status,response_json,created_at
                   FROM research.recipe_mutation_receipts
                   WHERE idempotency_key=%s""",
                (key,),
            ).fetchone()
            if receipt_row is not None:
                return replay_recipe_mutation_receipt(
                    self._recipe_mutation_receipt(receipt_row),
                    mutation_kind="retire",
                    request_intent_hash=intent_hash,
                )

            row = conn.execute(
                """SELECT status,payload_json FROM research.recipes
                   WHERE recipe_code=%s FOR UPDATE""",
                (recipe_code,),
            ).fetchone()
            if row is None:
                raise KeyError(recipe_code)
            if str(row["status"]) == "draft":
                raise ValueError("a draft recipe cannot be retired")
            if str(row["status"]) != "retired":
                conn.execute(
                    """UPDATE research.recipes
                       SET status='retired',updated_at=now()
                       WHERE recipe_code=%s""",
                    (recipe_code,),
                )
            response = {
                **_decode(row["payload_json"]),
                "lifecycleStatus": "retired",
            }
            conn.execute(
                """INSERT INTO research.recipe_mutation_receipts(
                     idempotency_key,mutation_kind,request_intent_hash,
                     recipe_code,response_status,response_json)
                   VALUES(%s,%s,%s,%s,%s,%s::jsonb)""",
                (
                    key,
                    "retire",
                    intent_hash,
                    recipe_code,
                    200,
                    _encode(response),
                ),
            )
        return response

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
            clauses.append("status=%s")
            parameters.append(status)
        if anchor is not None:
            clauses.append(
                """(created_at < %s::timestamptz OR
                   (created_at = %s::timestamptz AND
                    recipe_code COLLATE "C" > %s))"""
            )
            parameters.extend(
                [anchor.created_at, anchor.created_at, anchor.recipe_code]
            )
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.append(size + 1)
        with contextual_connection(self.pool) as conn:
            rows = conn.execute(
                f"""SELECT recipe_code,status,payload_json,created_at
                    FROM research.recipes{where}
                    ORDER BY created_at DESC,recipe_code COLLATE "C" ASC
                    LIMIT %s""",
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
        now = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        value["createdAt"] = normalize_timestamp(value.get("createdAt"), fallback=now)
        value["updatedAt"] = normalize_timestamp(value.get("updatedAt"), fallback=now)
        with contextual_connection(self.pool) as conn:
            row = conn.execute(
                "SELECT payload_json FROM research.runs WHERE run_hash=%s",
                (value["runHash"],),
            ).fetchone()
            if row is not None:
                return _decode(row["payload_json"])
            conn.execute(
                """INSERT INTO research.runs(run_code,run_hash,recipe_code,recipe_hash,status,progress,payload_json)
              VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb)""",
                (
                    value["runCode"],
                    value["runHash"],
                    value["recipeCode"],
                    value["recipeHash"],
                    value["status"],
                    float(value["progress"]),
                    _encode(value),
                ),
            )
        return value

    def update_run(self, run_code: str, **changes: Any) -> dict[str, Any]:
        with contextual_connection(self.pool) as conn:
            row = conn.execute(
                "SELECT payload_json FROM research.runs WHERE run_code=%s FOR UPDATE",
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
            conn.execute(
                "UPDATE research.runs SET status=%s,progress=%s,payload_json=%s::jsonb,updated_at=now() WHERE run_code=%s",
                (
                    value["status"],
                    float(value["progress"]),
                    _encode(value),
                    str(run_code),
                ),
            )
        return value

    def get_run(self, run_code: str, *, include_coordinator_evidence: bool = True) -> dict[str, Any] | None:
        payload = "payload_json" if include_coordinator_evidence else _OPERATOR_RUN_PAYLOAD_SQL
        with contextual_connection(self.pool) as conn:
            row = conn.execute(
                f"SELECT {payload} AS payload_json FROM research.runs WHERE run_code=%s",
                (str(run_code),),
            ).fetchone()
        return None if row is None else _decode(row["payload_json"])

    def run_status(self, run_code: str) -> str | None:
        with contextual_connection(self.pool) as conn:
            row = conn.execute(
                "SELECT status FROM research.runs WHERE run_code=%s",
                (str(run_code),),
            ).fetchone()
        return None if row is None else str(row["status"])

    def list_runs(
        self, *, status: str | None = None, limit: int | None = 100,
        include_coordinator_evidence: bool = True,
    ) -> list[dict[str, Any]]:
        payload = "payload_json" if include_coordinator_evidence else _OPERATOR_RUN_PAYLOAD_SQL
        sql = f"SELECT {payload} AS payload_json FROM research.runs"
        args: list[Any] = []
        if status is not None:
            sql += " WHERE status=%s"
            args.append(status)
        sql += " ORDER BY created_at DESC,run_code"
        if limit is not None:
            sql += " LIMIT %s"
            args.append(int(limit))
        with contextual_connection(self.pool) as conn:
            rows = conn.execute(sql, args).fetchall()
        return [_decode(row["payload_json"]) for row in rows]

    @staticmethod
    def _shard_row(row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            **_decode(row["spec_json"]),
            "status": str(row["status"]),
            "attemptCount": int(row["attempt_count"]),
            "retryCount": (
                int(row["retry_count"])
                if row["operation_counters_available"] is True
                else None
            ),
            "reclaimCount": (
                int(row["reclaim_count"])
                if row["operation_counters_available"] is True
                else None
            ),
            "fencingToken": int(row["fencing_token"]),
            "leaseOwner": row["lease_owner"],
            "leaseExpiresAt": (
                row["lease_expires_at"].isoformat() if row["lease_expires_at"] else None
            ),
            "heartbeatAt": row["heartbeat_at"].isoformat()
            if row["heartbeat_at"]
            else None,
            "result": None
            if row["result_json"] is None
            else _decode(row["result_json"]),
            "error": None if row["error_json"] is None else _decode(row["error_json"]),
        }

    def create_shards(
        self, specs: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...]
    ) -> list[dict[str, Any]]:
        values = [copy.deepcopy(dict(item)) for item in specs]
        if not values:
            return []
        run_codes = {str(value["runCode"]) for value in values}
        if len(run_codes) != 1:
            raise ValueError("research shard batch must belong to one run")
        run_code = next(iter(run_codes))
        with contextual_connection(self.pool) as conn:
            run = conn.execute(
                "SELECT run_code FROM research.runs WHERE run_code=%s FOR UPDATE",
                (run_code,),
            ).fetchone()
            if run is None:
                raise KeyError(run_code)
            for value in values:
                row = conn.execute(
                    "SELECT spec_json FROM research.shards WHERE shard_id=%s FOR UPDATE",
                    (str(value["shardId"]),),
                ).fetchone()
                if row is not None:
                    if _decode(row["spec_json"]) != value:
                        raise ValueError("research shard identity is immutable")
                    continue
                conn.execute(
                    """INSERT INTO research.shards(
                       shard_id,shard_hash,run_code,shard_index,status,
                       operation_counters_available,spec_json)
                       VALUES(%s,%s,%s,%s,'queued',true,%s::jsonb)""",
                    (
                        value["shardId"],
                        value["sha256"],
                        value["runCode"],
                        int(value["shardIndex"]),
                        _encode(value),
                    ),
                )
        return self.list_shards(run_code)

    def get_shard(self, shard_id: str) -> dict[str, Any] | None:
        with contextual_connection(self.pool) as conn:
            row = conn.execute(
                "SELECT * FROM research.shards WHERE shard_id=%s", (str(shard_id),)
            ).fetchone()
        return None if row is None else self._shard_row(row)

    def list_shards(self, run_code: str) -> list[dict[str, Any]]:
        with contextual_connection(self.pool) as conn:
            rows = conn.execute(
                "SELECT * FROM research.shards WHERE run_code=%s ORDER BY shard_index",
                (str(run_code),),
            ).fetchall()
        return [self._shard_row(row) for row in rows]

    def list_performance_shards(
        self, *, run_code: str | None = None
    ) -> list[dict[str, Any]]:
        where = " WHERE run_code=%s" if run_code is not None else ""
        args: tuple[Any, ...] = (str(run_code),) if run_code is not None else ()
        with contextual_connection(self.pool) as conn:
            rows = conn.execute(
                f"SELECT * FROM research.shards{where} ORDER BY run_code,shard_index",
                args,
            ).fetchall()
        return [self._shard_row(row) for row in rows]

    def performance_evidence_snapshot(
        self, *, run_code: str | None = None
    ) -> dict[str, list[dict[str, Any]]]:
        """Read all run-level and shard-level performance evidence atomically."""

        run_where = " WHERE run_code=%s" if run_code is not None else ""
        shard_where = " WHERE run_code=%s" if run_code is not None else ""
        args: tuple[Any, ...] = (str(run_code),) if run_code is not None else ()
        with self.pool.connection() as conn:
            with conn.transaction():
                conn.execute(
                    "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"
                )
                apply_postgres_identity_context(conn)
                run_rows = conn.execute(
                    f"SELECT payload_json FROM research.runs{run_where} "
                    "ORDER BY created_at DESC,run_code",
                    args,
                ).fetchall()
                shard_rows = conn.execute(
                    f"SELECT * FROM research.shards{shard_where} "
                    "ORDER BY run_code,shard_index",
                    args,
                ).fetchall()
        return {
            "runs": [_decode(row["payload_json"]) for row in run_rows],
            "shards": [self._shard_row(row) for row in shard_rows],
        }

    def claim_shard(
        self,
        run_code: str,
        *,
        worker_id: str,
        lease_seconds: float,
        now: datetime | None = None,
    ) -> dict[str, Any] | None:
        seconds = normalize_lease_duration(lease_seconds)
        moment = normalize_lease_moment(now)
        expiry = moment + timedelta(seconds=seconds)
        with contextual_connection(self.pool) as conn:
            row = conn.execute(
                """SELECT * FROM research.shards WHERE run_code=%s AND
                   (status='queued' OR (status='running' AND lease_expires_at<=%s))
                   ORDER BY shard_index FOR UPDATE SKIP LOCKED LIMIT 1""",
                (str(run_code), moment),
            ).fetchone()
            if row is None:
                return None
            attempt = int(row["attempt_count"]) + 1
            maximum = int(_decode(row["spec_json"])["maximumAttempts"])
            reclaim_increment = 1 if str(row["status"]) == "running" else 0
            if attempt > maximum:
                terminal_attempt_status = "reclaimed" if reclaim_increment else "failed"
                terminal_failures = ["research_shard_maximum_attempts_exhausted"]
                if reclaim_increment:
                    terminal_failures.extend(
                        (
                            "research_shard_attempt_reclaimed",
                            "research_shard_lease_expired",
                        )
                    )
                else:
                    terminal_failures.extend(
                        (
                            "research_shard_attempt_failed",
                            "research_shard_terminal_failure",
                        )
                    )
                self._update_audit_trace_attempts(
                    conn,
                    shard_id=str(row["shard_id"]),
                    fencing_token=int(row["fencing_token"]),
                    attempt_status=terminal_attempt_status,
                    failure_categories=terminal_failures,
                )
                conn.execute(
                    """UPDATE research.shards SET status='failed',
                       reclaim_count=reclaim_count+%s,updated_at=now() WHERE shard_id=%s""",
                    (reclaim_increment, str(row["shard_id"])),
                )
                return None
            if reclaim_increment:
                self._update_audit_trace_attempts(
                    conn,
                    shard_id=str(row["shard_id"]),
                    fencing_token=int(row["fencing_token"]),
                    attempt_status="reclaimed",
                    failure_categories=(
                        "research_shard_attempt_reclaimed",
                        "research_shard_lease_expired",
                    ),
                )
            fence = int(row["fencing_token"]) + 1
            updated = conn.execute(
                """UPDATE research.shards SET status='running',attempt_count=%s,
                   reclaim_count=reclaim_count+%s,fencing_token=%s,lease_owner=%s,
                   lease_expires_at=%s,heartbeat_at=%s,
                   error_json=NULL,updated_at=now() WHERE shard_id=%s RETURNING *""",
                (
                    attempt,
                    reclaim_increment,
                    fence,
                    str(worker_id),
                    expiry,
                    moment,
                    str(row["shard_id"]),
                ),
            ).fetchone()
        return self._shard_row(updated)

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
        with contextual_connection(self.pool) as conn:
            current = conn.execute(
                "SELECT * FROM research.shards WHERE shard_id=%s FOR UPDATE",
                (str(shard_id),),
            ).fetchone()
            moment = normalize_lease_moment(now)
            if (
                current is None
                or str(current["status"]) != "running"
                or str(current["lease_owner"] or "") != str(worker_id)
                or int(current["fencing_token"]) != fence
                or current["lease_expires_at"] is None
                or current["lease_expires_at"] <= moment
            ):
                raise RuntimeError("stale research shard lease")
            expiry = moment + timedelta(seconds=seconds)
            row = conn.execute(
                """UPDATE research.shards SET heartbeat_at=%s,lease_expires_at=%s,updated_at=now()
                   WHERE shard_id=%s
                   RETURNING *""",
                (
                    moment,
                    expiry,
                    str(shard_id),
                ),
            ).fetchone()
            if row is None:  # pragma: no cover - protected by the row lock
                raise RuntimeError("stale research shard lease")
        return self._shard_row(row)

    def complete_shard(
        self,
        shard_id: str,
        *,
        worker_id: str,
        fencing_token: int,
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        fence = strict_positive_integer(fencing_token, field="research fencing_token")
        with contextual_connection(self.pool) as conn:
            current = conn.execute(
                "SELECT * FROM research.shards WHERE shard_id=%s FOR UPDATE",
                (str(shard_id),),
            ).fetchone()
            moment = normalize_lease_moment(None)
            if (
                current is None
                or str(current["status"]) != "running"
                or str(current["lease_owner"] or "") != str(worker_id)
                or int(current["fencing_token"]) != fence
                or current["lease_expires_at"] is None
                or current["lease_expires_at"] <= moment
            ):
                raise RuntimeError("stale research shard completion")
            row = conn.execute(
                """UPDATE research.shards SET status='completed',result_json=%s::jsonb,
                   error_json=NULL,lease_owner=NULL,lease_expires_at=NULL,
                   heartbeat_at=%s,updated_at=now()
                   WHERE shard_id=%s
                   RETURNING *""",
                (_encode(dict(result)), moment, str(shard_id)),
            ).fetchone()
            if row is None:  # pragma: no cover - protected by the row lock
                raise RuntimeError("stale research shard completion")
            self._update_audit_trace_attempts(
                conn,
                shard_id=str(shard_id),
                fencing_token=fence,
                attempt_status="completed",
            )
            attempt_count = int(row["attempt_count"])
            conn.execute(
                """UPDATE research.result_occurrences
                   SET admitted=true,updated_at=now()
                   WHERE shard_id=%s AND attempt_count=%s
                     AND fencing_token=%s""",
                (str(shard_id), attempt_count, fence),
            )
            # The semantic row and its first-origin projections stay
            # immutable. Exact occurrence admission controls visibility.
            conn.execute(
                """UPDATE research.results SET admitted=true
                   WHERE result_code IN (
                     SELECT result_code FROM research.result_occurrences
                     WHERE shard_id=%s AND attempt_count=%s
                       AND fencing_token=%s AND admitted=true
                   )""",
                (str(shard_id), attempt_count, fence),
            )
            conn.execute(
                """UPDATE research.semantic_disagreement_occurrences
                   SET admitted=true,updated_at=now()
                   WHERE shard_id=%s AND attempt_count=%s
                     AND fencing_token=%s""",
                (str(shard_id), attempt_count, fence),
            )
            conn.execute(
                """UPDATE research.semantic_disagreements SET admitted=true
                   WHERE disagreement_id IN (
                     SELECT disagreement_id
                     FROM research.semantic_disagreement_occurrences
                     WHERE shard_id=%s AND attempt_count=%s
                       AND fencing_token=%s AND admitted=true
                   )""",
                (str(shard_id), attempt_count, fence),
            )
            conn.execute(
                """UPDATE research.sequential_search_receipt_occurrences
                   SET admitted=true,updated_at=now()
                   WHERE shard_id=%s AND attempt_count=%s
                     AND fencing_token=%s""",
                (str(shard_id), attempt_count, fence),
            )
        return self._shard_row(row)

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
        with contextual_connection(self.pool) as conn:
            current = conn.execute(
                "SELECT * FROM research.shards WHERE shard_id=%s FOR UPDATE",
                (str(shard_id),),
            ).fetchone()
            if current is None:
                raise KeyError(shard_id)
            maximum = int(_decode(current["spec_json"])["maximumAttempts"])
            status = (
                "queued"
                if retryable and int(current["attempt_count"]) < maximum
                else "failed"
            )
            retry_increment = 1 if status == "queued" else 0
            row = conn.execute(
                """UPDATE research.shards SET status=%s,retry_count=retry_count+%s,
                   error_json=%s::jsonb,lease_owner=NULL,lease_expires_at=NULL,
                   heartbeat_at=now(),updated_at=now()
                   WHERE shard_id=%s AND status='running' AND lease_owner=%s AND fencing_token=%s
                   RETURNING *""",
                (
                    status,
                    retry_increment,
                    _encode(dict(error)),
                    str(shard_id),
                    str(worker_id),
                    fence,
                ),
            ).fetchone()
            if row is None:
                raise RuntimeError("stale research shard failure")
            failure_categories = ["research_shard_attempt_failed"]
            attempt_status = "retry_scheduled" if status == "queued" else "failed"
            if attempt_status == "retry_scheduled":
                failure_categories.append("research_shard_retry_scheduled")
            else:
                failure_categories.append("research_shard_terminal_failure")
            self._update_audit_trace_attempts(
                conn,
                shard_id=str(shard_id),
                fencing_token=fence,
                attempt_status=attempt_status,
                failure_categories=failure_categories,
            )
        return self._shard_row(row)

    def cancel_shards(self, run_code: str) -> int:
        with contextual_connection(self.pool) as conn:
            pending = conn.execute(
                """SELECT shard_id,fencing_token FROM research.shards
                   WHERE run_code=%s AND status IN ('queued','running')
                   ORDER BY shard_index FOR UPDATE""",
                (str(run_code),),
            ).fetchall()
            rows = conn.execute(
                """UPDATE research.shards SET status='cancelled',lease_owner=NULL,
                   lease_expires_at=NULL,updated_at=now()
                   WHERE run_code=%s AND status IN ('queued','running')
                   RETURNING shard_id""",
                (str(run_code),),
            ).fetchall()
            for shard in pending:
                self._update_audit_trace_attempts(
                    conn,
                    shard_id=str(shard["shard_id"]),
                    fencing_token=int(shard["fencing_token"]),
                    attempt_status="cancelled",
                    failure_categories=("research_shard_attempt_cancelled",),
                )
        return len(rows)

    def recover_orphaned_shards(self) -> list[str]:
        # In multi-process mode, only an expired lease is known to be orphaned.
        with contextual_connection(self.pool) as conn:
            orphaned = conn.execute(
                """SELECT shard_id,run_code,fencing_token FROM research.shards
                   WHERE status='running' AND lease_expires_at<=now()
                   ORDER BY run_code,shard_index FOR UPDATE"""
            ).fetchall()
            conn.execute(
                """UPDATE research.shards SET status='queued',lease_owner=NULL,
                   lease_expires_at=NULL,reclaim_count=reclaim_count+1,updated_at=now()
                   WHERE status='running' AND lease_expires_at<=now()"""
            )
            for shard in orphaned:
                self._update_audit_trace_attempts(
                    conn,
                    shard_id=str(shard["shard_id"]),
                    fencing_token=int(shard["fencing_token"]),
                    attempt_status="reclaimed",
                    failure_categories=(
                        "research_shard_attempt_reclaimed",
                        "research_shard_process_recovered",
                    ),
                )
        return sorted({str(row["run_code"]) for row in orphaned})

    def put_comparison_campaign(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        value = copy.deepcopy(dict(payload))
        with contextual_connection(self.pool) as conn:
            row = conn.execute(
                "SELECT payload_json FROM research.comparison_campaigns WHERE campaign_hash=%s",
                (str(value["campaignHash"]),),
            ).fetchone()
            if row is not None:
                return _decode(row["payload_json"])
            conn.execute(
                """INSERT INTO research.comparison_campaigns(campaign_code,campaign_hash,base_recipe_code,payload_json)
              VALUES(%s,%s,%s,%s::jsonb)""",
                (
                    value["campaignCode"],
                    value["campaignHash"],
                    value["baseRecipeCode"],
                    _encode(value),
                ),
            )
        return value

    def get_comparison_campaign(self, code: str) -> dict[str, Any] | None:
        with contextual_connection(self.pool) as conn:
            row = conn.execute(
                "SELECT payload_json FROM research.comparison_campaigns WHERE campaign_code=%s",
                (str(code),),
            ).fetchone()
        return None if row is None else _decode(row["payload_json"])

    def list_comparison_campaigns(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with contextual_connection(self.pool) as conn:
            rows = conn.execute(
                "SELECT payload_json FROM research.comparison_campaigns ORDER BY created_at DESC,campaign_code LIMIT %s",
                (int(limit),),
            ).fetchall()
        return [_decode(row["payload_json"]) for row in rows]

    @staticmethod
    def _validate_occurrence_shard(
        conn: Any,
        *,
        run_code: str,
        shard_id: str,
        attempt_count: int,
        fencing_token: int,
        worker_id: str,
        seed: int | None = None,
    ) -> None:
        row = conn.execute(
            """SELECT run_code,spec_json,status,attempt_count,fencing_token,
               lease_owner,lease_expires_at FROM research.shards
               WHERE shard_id=%s FOR UPDATE""",
            (str(shard_id),),
        ).fetchone()
        if row is None:
            raise KeyError(shard_id)
        moment = datetime.now(timezone.utc)
        if (
            str(row["run_code"]) != str(run_code)
            or str(row["status"]) != "running"
            or int(row["attempt_count"]) != int(attempt_count)
            or int(row["fencing_token"]) != int(fencing_token)
            or str(row["lease_owner"] or "") != str(worker_id)
            or row["lease_expires_at"] is None
            or row["lease_expires_at"] <= moment
        ):
            raise RuntimeError("stale research shard occurrence admission")
        if seed is not None:
            spec = _decode(row["spec_json"])
            start = int(spec["seedStart"])
            end = start + int(spec["seedCount"])
            if not start <= int(seed) < end:
                raise ValueError("research receipt seed is outside its shard")

    @staticmethod
    def _validate_unsharded_write(conn: Any, run_code: str) -> None:
        run = conn.execute(
            "SELECT run_code FROM research.runs WHERE run_code=%s FOR UPDATE",
            (str(run_code),),
        ).fetchone()
        if run is None:
            raise KeyError(run_code)
        row = conn.execute(
            "SELECT 1 FROM research.shards WHERE run_code=%s LIMIT 1",
            (str(run_code),),
        ).fetchone()
        if row is not None:
            raise ValueError(
                "sharded research writes require an explicit current lease"
            )

    @staticmethod
    def _put_fenced_occurrence(
        conn: Any,
        *,
        table: str,
        semantic_column: str,
        semantic_id: str,
        occurrence: Mapping[str, Any],
    ) -> None:
        if table not in {
            "research.result_occurrences",
            "research.semantic_disagreement_occurrences",
        }:
            raise ValueError("unsupported fenced occurrence table")
        inserted = conn.execute(
            f"""INSERT INTO {table}(
                occurrence_hash,{semantic_column},run_code,shard_id,
                attempt_count,fencing_token,admitted)
                VALUES(%s,%s,%s,%s,%s,%s,false)
                ON CONFLICT DO NOTHING
                RETURNING occurrence_hash,{semantic_column},run_code,shard_id,
                  attempt_count,fencing_token""",
            (
                str(occurrence["occurrenceHash"]),
                str(semantic_id),
                str(occurrence["runCode"]),
                str(occurrence["shardId"]),
                int(occurrence["attemptCount"]),
                int(occurrence["fencingToken"]),
            ),
        ).fetchone()
        if inserted is not None:
            return
        rows = conn.execute(
            f"""SELECT occurrence_hash,{semantic_column},run_code,shard_id,
                  attempt_count,fencing_token
                FROM {table}
                WHERE occurrence_hash=%s OR (
                  {semantic_column}=%s AND shard_id=%s
                  AND attempt_count=%s AND fencing_token=%s
                ) FOR UPDATE""",
            (
                str(occurrence["occurrenceHash"]),
                str(semantic_id),
                str(occurrence["shardId"]),
                int(occurrence["attemptCount"]),
                int(occurrence["fencingToken"]),
            ),
        ).fetchall()
        if len(rows) != 1:
            raise RuntimeError("research occurrence uniqueness conflict")
        row = rows[0]
        if (
            str(row["occurrence_hash"]) != str(occurrence["occurrenceHash"])
            or str(row[semantic_column]) != str(semantic_id)
            or str(row["run_code"]) != str(occurrence["runCode"])
            or str(row["shard_id"]) != str(occurrence["shardId"])
            or int(row["attempt_count"]) != int(occurrence["attemptCount"])
            or int(row["fencing_token"]) != int(occurrence["fencingToken"])
        ):
            raise RuntimeError("research occurrence identity is immutable")

    @staticmethod
    def _sequential_search_receipt_row(row: Mapping[str, Any]) -> dict[str, Any]:
        created_at = row["created_at"]
        if isinstance(created_at, datetime):
            created_at_text = created_at.isoformat(timespec="microseconds")
        else:
            created_at_text = _strict_receipt_text(
                created_at, field="sequential receipt creation time"
            )
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
        admitted = row["admitted"]
        if type(admitted) is not bool:
            raise ValueError("sequential receipt admitted flag is invalid")
        return {
            "schemaVersion": 1,
            "occurrenceHash": occurrence_hash,
            "receiptHash": receipt_hash,
            "runCode": run_code,
            "shardId": shard_id,
            "attemptCount": attempt_count,
            "fencingToken": fencing_token,
            "seed": seed,
            "admitted": admitted,
            "createdAt": created_at_text,
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
        with contextual_connection(self.pool) as conn:
            self._validate_occurrence_shard(
                conn,
                run_code=normalized_run_code,
                shard_id=fenced[0],
                attempt_count=fenced[1],
                fencing_token=fenced[2],
                worker_id=fenced[3],
                seed=normalized_seed,
            )
            bundle_row = conn.execute(
                """INSERT INTO research.sequential_search_receipt_bundles(
                   receipt_hash,seed,receipt_json)
                   VALUES(%s,%s,%s::jsonb)
                   ON CONFLICT DO NOTHING
                   RETURNING receipt_hash,seed,receipt_json""",
                (str(bundle["sha256"]), normalized_seed, _encode(bundle)),
            ).fetchone()
            if bundle_row is None:
                bundle_row = conn.execute(
                    """SELECT receipt_hash,seed,receipt_json
                       FROM research.sequential_search_receipt_bundles
                       WHERE receipt_hash=%s FOR UPDATE""",
                    (str(bundle["sha256"]),),
                ).fetchone()
            if (
                bundle_row is None
                or str(bundle_row["receipt_hash"]) != str(bundle["sha256"])
                or _strict_nonnegative_database_integer(
                    bundle_row["seed"], field="sequential receipt bundle seed"
                )
                != normalized_seed
                or _decode(bundle_row["receipt_json"]) != bundle
            ):
                raise RuntimeError("sequential search receipt CAS payload mismatch")
            inserted = conn.execute(
                """INSERT INTO research.sequential_search_receipt_occurrences(
                   occurrence_hash,receipt_hash,run_code,shard_id,
                   attempt_count,fencing_token,seed,admitted)
                   VALUES(%s,%s,%s,%s,%s,%s,%s,false)
                   ON CONFLICT DO NOTHING
                   RETURNING occurrence_hash,receipt_hash,run_code,shard_id,
                     attempt_count,fencing_token,seed""",
                (
                    str(occurrence["occurrenceHash"]),
                    str(bundle["sha256"]),
                    normalized_run_code,
                    fenced[0],
                    fenced[1],
                    fenced[2],
                    normalized_seed,
                ),
            ).fetchone()
            if inserted is None:
                rows = conn.execute(
                    """SELECT occurrence_hash,receipt_hash,run_code,shard_id,
                       attempt_count,fencing_token,seed
                       FROM research.sequential_search_receipt_occurrences
                       WHERE occurrence_hash=%s OR (
                         run_code=%s AND shard_id=%s AND attempt_count=%s
                         AND fencing_token=%s AND seed=%s
                       ) FOR UPDATE""",
                    (
                        str(occurrence["occurrenceHash"]),
                        normalized_run_code,
                        fenced[0],
                        fenced[1],
                        fenced[2],
                        normalized_seed,
                    ),
                ).fetchall()
                if len(rows) != 1:
                    raise RuntimeError(
                        "sequential search occurrence uniqueness conflict"
                    )
                inserted = rows[0]
            if (
                _strict_receipt_text(
                    inserted["occurrence_hash"],
                    field="sequential receipt occurrence hash",
                )
                != str(occurrence["occurrenceHash"])
                or _strict_receipt_text(
                    inserted["receipt_hash"], field="sequential receipt hash"
                )
                != str(bundle["sha256"])
                or _strict_receipt_text(
                    inserted["run_code"], field="sequential receipt run code"
                )
                != normalized_run_code
                or _strict_receipt_text(
                    inserted["shard_id"], field="sequential receipt shard id"
                )
                != fenced[0]
                or strict_positive_integer(
                    inserted["attempt_count"],
                    field="sequential receipt attempt count",
                )
                != fenced[1]
                or strict_positive_integer(
                    inserted["fencing_token"],
                    field="sequential receipt fencing token",
                )
                != fenced[2]
                or _strict_nonnegative_database_integer(
                    inserted["seed"], field="sequential receipt occurrence seed"
                )
                != normalized_seed
            ):
                raise RuntimeError("sequential search seed occurrence is immutable")
            row = conn.execute(
                """SELECT occurrence.*,bundle.receipt_json
                   FROM research.sequential_search_receipt_occurrences occurrence
                   JOIN research.sequential_search_receipt_bundles bundle
                     ON bundle.receipt_hash=occurrence.receipt_hash
                   WHERE occurrence.occurrence_hash=%s""",
                (str(occurrence["occurrenceHash"]),),
            ).fetchone()
            if row is None:  # pragma: no cover - protected by this transaction
                raise RuntimeError("sequential search receipt was not retained")
        return self._sequential_search_receipt_row(row)

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
        args: list[Any] = []
        if admitted_only:
            clauses.append("occurrence.admitted=true")
        if run_code is not None:
            run_code = _strict_receipt_text(
                run_code, field="sequential receipt run code"
            )
            clauses.append("occurrence.run_code=%s")
            args.append(run_code)
        if shard_id is not None:
            shard_id = _strict_receipt_text(
                shard_id, field="sequential receipt shard id"
            )
            clauses.append("occurrence.shard_id=%s")
            args.append(shard_id)
        return (" WHERE " + " AND ".join(clauses) if clauses else "", args)

    def count_sequential_search_receipts(
        self,
        *,
        run_code: str | None = None,
        shard_id: str | None = None,
        admitted_only: bool = True,
    ) -> int:
        where, args = self._search_receipt_filters(
            run_code=run_code,
            shard_id=shard_id,
            admitted_only=admitted_only,
        )
        with contextual_connection(self.pool) as conn:
            row = conn.execute(
                f"""SELECT COUNT(*) AS count
                    FROM research.sequential_search_receipt_occurrences occurrence{where}""",
                args,
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
        """Stream one server-cursor snapshot in canonical statistics order."""

        normalized_batch_size = strict_positive_integer(
            batch_size, field="sequential receipt snapshot batch size"
        )
        if normalized_batch_size > _SEQUENTIAL_RECEIPT_SNAPSHOT_BATCH_MAX:
            raise ValueError("sequential receipt snapshot batch size is too large")
        where, args = self._search_receipt_filters(
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
        cursor_name = f"balootx_sequential_receipts_{uuid.uuid4().hex}"
        with self.pool.connection() as conn:
            with conn.transaction():
                conn.execute(
                    "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"
                )
                apply_postgres_identity_context(conn)
                with conn.cursor(name=cursor_name) as cursor:
                    cursor.itersize = normalized_batch_size
                    cursor.execute(
                        f"""SELECT occurrence.*,bundle.receipt_json
                            FROM research.sequential_search_receipt_occurrences occurrence
                            JOIN research.sequential_search_receipt_bundles bundle
                              ON bundle.receipt_hash=occurrence.receipt_hash{where}
                            ORDER BY {order}""",
                        args,
                    )

                    def rows() -> Iterator[dict[str, Any]]:
                        while True:
                            batch = cursor.fetchmany(normalized_batch_size)
                            if not batch:
                                return
                            for row in batch:
                                yield self._sequential_search_receipt_row(row)

                    yield rows()

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
        where, args = self._search_receipt_filters(
            run_code=run_code,
            shard_id=shard_id,
            admitted_only=admitted_only,
        )
        with contextual_connection(self.pool) as conn:
            rows = conn.execute(
                f"""WITH filtered AS (
                      SELECT occurrence.*,bundle.receipt_json
                      FROM research.sequential_search_receipt_occurrences occurrence
                      JOIN research.sequential_search_receipt_bundles bundle
                        ON bundle.receipt_hash=occurrence.receipt_hash{where}
                    ), page AS (
                      SELECT * FROM filtered ORDER BY created_at DESC,
                        occurrence_hash ASC LIMIT %s OFFSET %s
                    )
                    SELECT page.*,totals.total
                    FROM (SELECT COUNT(*) AS total FROM filtered) totals
                    LEFT JOIN page ON true
                    ORDER BY page.created_at DESC,page.occurrence_hash ASC""",
                [*args, normalized_limit, normalized_offset],
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
        library_value = result_library_safe_payload(value)
        payload_hash = stored_result_payload_hash_v1(value)
        created_at = normalize_timestamp(
            value.get("createdAt"),
            fallback=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        )
        projection = result_library_projection(
            library_value, created_at=created_at
        )
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
        with contextual_connection(self.pool) as conn:
            if occurrence is not None:
                self._validate_occurrence_shard(
                    conn,
                    run_code=str(value["runCode"]),
                    shard_id=str(occurrence["shardId"]),
                    attempt_count=fenced[1],
                    fencing_token=fenced[2],
                    worker_id=fenced[3],
                )
            else:
                self._validate_unsharded_write(conn, str(value["runCode"]))
            row = conn.execute(
                """INSERT INTO research.results(
                   result_code,result_hash,run_code,truth_label,confidence,
                   score,disposition,payload_json,payload_hash,created_at,shard_id,
                   admitted,projection_version,library_projection_json,
                   library_projection_hash,
                   attempt_count,fencing_token)
                   VALUES(%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,
                      COALESCE(%s::timestamptz,now()),%s,%s,%s,%s::jsonb,%s,%s,%s)
                   ON CONFLICT DO NOTHING
                   RETURNING result_code,result_hash,run_code,payload_json,payload_hash""",
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
                    fenced is None,
                    RESULT_PROJECTION_VERSION,
                    _encode(projection),
                    stored_result_library_projection_hash_v1(
                        projection, payload_hash=payload_hash
                    ),
                    None if fenced is None else fenced[1],
                    None if fenced is None else fenced[2],
                ),
            ).fetchone()
            inserted = row is not None
            if row is None:
                rows = conn.execute(
                    """SELECT result_code,result_hash,run_code,payload_json,payload_hash
                       FROM research.results
                       WHERE result_hash=%s OR result_code=%s FOR UPDATE""",
                    (value["resultHash"], value["resultCode"]),
                ).fetchall()
                if len(rows) != 1:
                    raise RuntimeError("research result uniqueness conflict")
                row = rows[0]
            stored = validate_stored_result_record_v1(
                _decode(row["payload_json"]),
                result_code=row["result_code"],
                result_hash=row["result_hash"],
                run_code=row["run_code"],
                payload_hash=row["payload_hash"],
            )
            if (
                str(row["result_code"]) != str(value["resultCode"])
                or str(row["result_hash"]) != str(value["resultHash"])
                or str(row["run_code"]) != str(value["runCode"])
                or stored != value
            ):
                raise RuntimeError("research result CAS payload mismatch")
            result_code = str(row["result_code"])
            if inserted:
                self._replace_result_facets(conn, result_code, library_value)
            if occurrence is not None:
                self._put_fenced_occurrence(
                    conn,
                    table="research.result_occurrences",
                    semantic_column="result_code",
                    semantic_id=result_code,
                    occurrence=occurrence,
                )
        return value

    def get_result(self, code: str) -> dict[str, Any] | None:
        with contextual_connection(self.pool) as conn:
            row = conn.execute(
                """SELECT result_code,result_hash,run_code,payload_json,payload_hash FROM research.results
                   WHERE result_code=%s AND admitted=true""",
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
        clauses: list[str] = ["r.admitted=true"]
        args: list[Any] = []
        for column, value in (
            ("run_code", query.run_code),
            ("truth_label", query.truth_label),
            ("disposition", query.disposition),
        ):
            if value is not None:
                clauses.append(f"r.{column}=%s")
                args.append(value)
        if query.minimum_score is not None:
            clauses.append("r.score>=%s")
            args.append(float(query.minimum_score))
        if query.search:
            clauses.append("r.payload_json::text ILIKE %s")
            args.append(f"%{query.search}%")
        if query.created_from is not None:
            clauses.append("r.created_at>=%s::timestamptz")
            args.append(query.created_from)
        if query.created_to is not None:
            clauses.append("r.created_at<=%s::timestamptz")
            args.append(query.created_to)
        for index, (facet_kind, facet_value) in enumerate(query.facet_filters):
            clauses.append(
                f"""EXISTS(SELECT 1 FROM research.result_facets f{index}
                    WHERE f{index}.result_code=r.result_code
                    AND f{index}.facet_kind=%s AND f{index}.facet_value=%s)"""
            )
            args.extend((facet_kind, facet_value))
        for index, (facet_kind, values) in enumerate(query.tag_facet_filters):
            placeholders = ",".join("%s" for _ in values)
            clauses.append(
                f"""EXISTS(SELECT 1 FROM research.result_facets tag{index}
                    WHERE tag{index}.result_code=r.result_code
                    AND tag{index}.facet_kind=%s
                    AND tag{index}.facet_value IN ({placeholders}))"""
            )
            args.extend((facet_kind, *values))
        where = " WHERE " + " AND ".join(clauses)
        with contextual_connection(self.pool) as conn:
            total = int(
                conn.execute(
                    f"SELECT COUNT(*) AS count FROM research.results r{where}", args
                ).fetchone()["count"]
            )
            rows = conn.execute(
                f"SELECT r.result_code,r.result_hash,r.run_code,r.payload_json,r.payload_hash FROM research.results r{where} ORDER BY {query.order_sql('r')} LIMIT %s OFFSET %s",
                [*args, int(query.limit), int(query.offset)],
            ).fetchall()
            groups: list[dict[str, Any]] = []
            if query.group_by in {"truth_label", "disposition"}:
                group_rows = conn.execute(
                    f"""SELECT r.{query.group_by} AS value,COUNT(*) AS count
                        FROM research.results r{where}
                        GROUP BY r.{query.group_by}
                        ORDER BY count DESC,value ASC""",
                    args,
                ).fetchall()
                groups = [
                    {"value": str(row["value"]), "count": int(row["count"])}
                    for row in group_rows
                ]
            elif query.group_by is not None:
                group_rows = conn.execute(
                    f"""SELECT grouped.facet_value AS value,
                        COUNT(DISTINCT r.result_code) AS count
                        FROM research.results r
                        JOIN research.result_facets grouped
                          ON grouped.result_code=r.result_code
                         AND grouped.facet_kind=%s
                        {where}
                        GROUP BY grouped.facet_value
                        ORDER BY count DESC,value ASC""",
                    [query.group_by, *args],
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

        with contextual_connection(self.pool) as conn:
            total = int(
                conn.execute(
                    "SELECT COUNT(*) AS count FROM research.results WHERE admitted=true"
                ).fetchone()["count"]
            )
            rows = conn.execute(
                """SELECT result_code,result_hash,run_code,payload_hash,
                          library_projection_json,library_projection_hash
                   FROM research.results
                   WHERE admitted=true
                   ORDER BY created_at DESC,result_code ASC
                   LIMIT %s OFFSET %s""",
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
        with contextual_connection(self.pool) as conn:
            if occurrence is not None:
                self._validate_occurrence_shard(
                    conn,
                    run_code=str(value["runCode"]),
                    shard_id=str(occurrence["shardId"]),
                    attempt_count=fenced[1],
                    fencing_token=fenced[2],
                    worker_id=fenced[3],
                )
            else:
                self._validate_unsharded_write(conn, str(value["runCode"]))
            row = conn.execute(
                """INSERT INTO research.semantic_disagreements(
                   disagreement_id,disagreement_hash,run_code,category,status,
                   payload_json,shard_id,admitted,attempt_count,fencing_token)
                   VALUES(%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s)
                   ON CONFLICT DO NOTHING
                   RETURNING disagreement_id,disagreement_hash,payload_json""",
                (
                    value["disagreementId"],
                    value["sha256"],
                    value["runCode"],
                    value["category"],
                    value["status"],
                    _encode(value),
                    None if fenced is None else fenced[0],
                    fenced is None,
                    None if fenced is None else fenced[1],
                    None if fenced is None else fenced[2],
                ),
            ).fetchone()
            if row is None:
                rows = conn.execute(
                    """SELECT disagreement_id,disagreement_hash,payload_json
                       FROM research.semantic_disagreements
                       WHERE disagreement_hash=%s OR disagreement_id=%s
                       FOR UPDATE""",
                    (value["sha256"], value["disagreementId"]),
                ).fetchall()
                if len(rows) != 1:
                    raise RuntimeError("semantic disagreement uniqueness conflict")
                row = rows[0]
            if (
                str(row["disagreement_id"]) != str(value["disagreementId"])
                or str(row["disagreement_hash"]) != str(value["sha256"])
                or _decode(row["payload_json"]) != value
            ):
                raise RuntimeError("semantic disagreement CAS payload mismatch")
            disagreement_id = str(row["disagreement_id"])
            if occurrence is not None:
                self._put_fenced_occurrence(
                    conn,
                    table="research.semantic_disagreement_occurrences",
                    semantic_column="disagreement_id",
                    semantic_id=disagreement_id,
                    occurrence=occurrence,
                )
        return value

    def list_disagreements(
        self,
        *,
        run_code: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = ["admitted=true"]
        args: list[Any] = []
        if run_code:
            clauses.append("run_code=%s")
            args.append(run_code)
        if status:
            clauses.append("status=%s")
            args.append(status)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        args.append(int(limit))
        with contextual_connection(self.pool) as conn:
            rows = conn.execute(
                f"SELECT payload_json FROM research.semantic_disagreements{where} ORDER BY created_at DESC LIMIT %s",
                args,
            ).fetchall()
        return [_decode(row["payload_json"]) for row in rows]

    def aggregate_stats(self, *, run_code: str | None = None) -> dict[str, Any]:
        run_where = " WHERE run_code=%s" if run_code is not None else ""
        admitted_where = " WHERE admitted=true" + (
            " AND run_code=%s" if run_code is not None else ""
        )
        args: list[Any] = [str(run_code)] if run_code is not None else []
        with contextual_connection(self.pool) as conn:
            run_rows = conn.execute(
                f"""SELECT status,COUNT(*) AS count
                    FROM research.runs{run_where} GROUP BY status""",
                args,
            ).fetchall()
            disposition_rows = conn.execute(
                f"""SELECT disposition,COUNT(*) AS count
                    FROM research.results{admitted_where} GROUP BY disposition""",
                args,
            ).fetchall()
            truth_rows = conn.execute(
                f"""SELECT truth_label,COUNT(*) AS count
                    FROM research.results{admitted_where} GROUP BY truth_label""",
                args,
            ).fetchall()
            score_row = conn.execute(
                f"""SELECT COUNT(*) AS count,AVG(score) AS average_score,
                    AVG(confidence) AS average_confidence
                    FROM research.results{admitted_where}""",
                args,
            ).fetchone()
            defect_rows = conn.execute(
                f"""SELECT category,COUNT(*) AS count
                    FROM research.semantic_disagreements{admitted_where}
                    GROUP BY category""",
                args,
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
