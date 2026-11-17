BEGIN;

CREATE TABLE IF NOT EXISTS foundry_catalogs_v1(
  catalog_hash CHAR(64) PRIMARY KEY,
  tenant_id TEXT NOT NULL DEFAULT 'default',
  kind TEXT NOT NULL,
  artifact_uri TEXT NOT NULL,
  artifact_sha256 CHAR(64) NOT NULL,
  artifact_bytes BIGINT NOT NULL,
  media_type TEXT NOT NULL,
  metadata_json JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS foundry_catalogs_kind_v1_idx
  ON foundry_catalogs_v1(kind,catalog_hash);

COMMIT;
