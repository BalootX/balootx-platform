BEGIN;

SELECT pg_advisory_xact_lock(
  hashtextextended('balootx:foundry-audit-traces-v1:bootstrap', 0)
);

CREATE TABLE IF NOT EXISTS foundry_audit_traces_v1(
  record_hash CHAR(64) PRIMARY KEY,
  audit_trace_code TEXT NOT NULL UNIQUE,
  trace_hash CHAR(64) NOT NULL,
  artifact_sha256 CHAR(64) NOT NULL,
  artifact_bytes BIGINT NOT NULL,
  source_occurrence_hash CHAR(64) NOT NULL UNIQUE,
  campaign_hash CHAR(64) NOT NULL
    REFERENCES foundry_campaigns_v1(campaign_hash) ON DELETE RESTRICT,
  spec_hash CHAR(64) NOT NULL
    REFERENCES foundry_shards_v1(spec_hash) ON DELETE RESTRICT,
  attempt INTEGER NOT NULL,
  fencing_token BIGINT NOT NULL,
  candidate_hash CHAR(64) NOT NULL,
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

CREATE INDEX IF NOT EXISTS foundry_audit_traces_library_v1_idx
  ON foundry_audit_traces_v1(
    campaign_hash,
    admitted,
    disposition,
    created_at DESC,
    trace_hash
  );

CREATE INDEX IF NOT EXISTS foundry_audit_traces_trace_v1_idx
  ON foundry_audit_traces_v1(trace_hash,created_at DESC,record_hash);

CREATE INDEX IF NOT EXISTS foundry_audit_traces_artifact_v1_idx
  ON foundry_audit_traces_v1(artifact_sha256);

COMMIT;
