BEGIN;

CREATE SCHEMA IF NOT EXISTS research;

CREATE TABLE IF NOT EXISTS research.domain_compilation_artifacts(
  compilation_hash TEXT PRIMARY KEY,
  artifact_sha256 TEXT UNIQUE NOT NULL,
  record_json JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMIT;
