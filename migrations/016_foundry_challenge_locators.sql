BEGIN;

CREATE TABLE IF NOT EXISTS challenge_locators_v1(
  code VARCHAR(32) PRIMARY KEY,
  tenant_id TEXT NOT NULL DEFAULT 'default',
  binding_json JSONB NOT NULL,
  public_json JSONB NOT NULL,
  created_at BIGINT NOT NULL,
  expires_at BIGINT,
  revoked_at BIGINT
);

CREATE INDEX IF NOT EXISTS challenge_locators_expiry_v1_idx
  ON challenge_locators_v1(expires_at)
  WHERE revoked_at IS NULL;

COMMIT;
