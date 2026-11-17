BEGIN;

SELECT pg_advisory_xact_lock(
  hashtextextended('balootx:research-store-v1:bootstrap', 0)
);

CREATE SCHEMA IF NOT EXISTS research;

CREATE TABLE IF NOT EXISTS research.audit_traces(
  record_hash TEXT PRIMARY KEY,
  audit_trace_code TEXT UNIQUE NOT NULL,
  trace_hash TEXT NOT NULL,
  artifact_sha256 TEXT NOT NULL,
  source_occurrence_hash TEXT NOT NULL,
  run_code TEXT NOT NULL,
  shard_id TEXT,
  attempt_count INTEGER NOT NULL,
  fencing_token BIGINT NOT NULL,
  candidate_hash TEXT NOT NULL,
  candidate_ordinal INTEGER NOT NULL,
  seed BIGINT NOT NULL,
  disposition TEXT NOT NULL,
  status TEXT NOT NULL,
  attempt_status TEXT NOT NULL,
  record_json JSONB NOT NULL,
  admitted BOOLEAN NOT NULL DEFAULT TRUE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_research_audit_trace_source
  ON research.audit_traces(source_occurrence_hash);

CREATE INDEX IF NOT EXISTS ix_research_audit_traces_library
  ON research.audit_traces(
    admitted,
    disposition,
    created_at DESC,
    trace_hash
  );

CREATE INDEX IF NOT EXISTS ix_research_audit_traces_trace
  ON research.audit_traces(
    trace_hash,
    created_at DESC,
    record_hash
  );

COMMIT;
