CREATE TABLE IF NOT EXISTS profile_artifacts(
  sha256 CHAR(64) PRIMARY KEY, family TEXT NOT NULL, profile_id TEXT NOT NULL,
  version TEXT NOT NULL, canonical_json JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS game_snapshots(
  game_id TEXT PRIMARY KEY, version BIGINT NOT NULL, payload BYTEA NOT NULL,
  state_hash CHAR(64) NOT NULL, event_count INTEGER NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS game_audit(
  id BIGSERIAL PRIMARY KEY, game_id TEXT NOT NULL, version BIGINT NOT NULL,
  state_hash CHAR(64) NOT NULL, event_count INTEGER NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS game_audit_game_version_idx ON game_audit(game_id,version);
CREATE TABLE IF NOT EXISTS jobs(
  job_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL DEFAULT 'default', kind TEXT NOT NULL,
  status TEXT NOT NULL, priority INTEGER NOT NULL DEFAULT 100, progress DOUBLE PRECISION NOT NULL DEFAULT 0,
  request_json JSONB NOT NULL, result_json JSONB, error TEXT, attempts INTEGER NOT NULL DEFAULT 0,
  max_attempts INTEGER NOT NULL DEFAULT 3, lease_owner TEXT, lease_until TIMESTAMPTZ,
  cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS jobs_claim_idx ON jobs(status,priority,created_at);
CREATE TABLE IF NOT EXISTS job_events(
  id BIGSERIAL PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
  type TEXT NOT NULL, payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS audit_events(
  id BIGSERIAL PRIMARY KEY, tenant_id TEXT NOT NULL, subject_id TEXT,
  action TEXT NOT NULL, resource_type TEXT NOT NULL, resource_id TEXT,
  details_json JSONB NOT NULL DEFAULT '{}'::jsonb, created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
