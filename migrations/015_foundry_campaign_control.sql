BEGIN;

CREATE TABLE IF NOT EXISTS foundry_campaigns_v1(
  campaign_hash CHAR(64) PRIMARY KEY,
  tenant_id TEXT NOT NULL DEFAULT 'default',
  spec_json JSONB NOT NULL,
  status TEXT NOT NULL,
  cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS foundry_shards_v1(
  spec_hash CHAR(64) PRIMARY KEY,
  campaign_hash CHAR(64) NOT NULL REFERENCES foundry_campaigns_v1(campaign_hash) ON DELETE CASCADE,
  shard_index INTEGER NOT NULL,
  spec_json JSONB NOT NULL,
  status TEXT NOT NULL,
  worker_id TEXT,
  lease_until DOUBLE PRECISION,
  fencing_token BIGINT NOT NULL DEFAULT 0,
  attempts INTEGER NOT NULL DEFAULT 0,
  result_json JSONB,
  error TEXT,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(campaign_hash,shard_index)
);

CREATE INDEX IF NOT EXISTS foundry_shards_claim_v1_idx
  ON foundry_shards_v1(status,lease_until,campaign_hash,shard_index);

COMMIT;
