BEGIN;

CREATE TABLE IF NOT EXISTS foundry_shard_attempts_v1(
  spec_hash CHAR(64) NOT NULL REFERENCES foundry_shards_v1(spec_hash) ON DELETE CASCADE,
  campaign_hash CHAR(64) NOT NULL REFERENCES foundry_campaigns_v1(campaign_hash) ON DELETE CASCADE,
  fencing_token BIGINT NOT NULL,
  attempt INTEGER NOT NULL,
  worker_id TEXT NOT NULL,
  claimed_at DOUBLE PRECISION NOT NULL,
  finished_at DOUBLE PRECISION,
  terminal_status TEXT,
  error TEXT,
  PRIMARY KEY(spec_hash,fencing_token)
);

CREATE INDEX IF NOT EXISTS foundry_attempts_campaign_v1_idx
  ON foundry_shard_attempts_v1(campaign_hash,spec_hash,fencing_token);

COMMIT;
