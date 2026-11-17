BEGIN;

SELECT pg_advisory_xact_lock(
  hashtextextended('balootx:research-store-v1:bootstrap', 0)
);

CREATE SCHEMA IF NOT EXISTS research;

CREATE TABLE IF NOT EXISTS research.comparison_campaigns(
  campaign_code TEXT PRIMARY KEY,
  campaign_hash TEXT UNIQUE NOT NULL,
  base_recipe_code TEXT NOT NULL
    REFERENCES research.recipes(recipe_code),
  payload_json JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS research.results(
  result_code TEXT PRIMARY KEY,
  result_hash TEXT UNIQUE NOT NULL,
  run_code TEXT NOT NULL
    REFERENCES research.runs(run_code) ON DELETE RESTRICT,
  truth_label TEXT NOT NULL,
  confidence DOUBLE PRECISION NOT NULL,
  score DOUBLE PRECISION NOT NULL,
  disposition TEXT NOT NULL,
  payload_json JSONB NOT NULL,
  shard_id TEXT,
  admitted BOOLEAN NOT NULL DEFAULT TRUE,
  projection_version INTEGER NOT NULL DEFAULT 0,
  attempt_count INTEGER,
  fencing_token BIGINT,
  CONSTRAINT ck_research_results_legacy_fence CHECK(
    (attempt_count IS NULL AND fencing_token IS NULL) OR
    (attempt_count IS NOT NULL AND fencing_token IS NOT NULL AND
     attempt_count >= 1 AND fencing_token >= 1)
  ),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_research_results_rank
  ON research.results(run_code,disposition,score DESC,result_code);

CREATE TABLE IF NOT EXISTS research.result_facets(
  result_code TEXT NOT NULL
    REFERENCES research.results(result_code) ON DELETE RESTRICT,
  facet_kind TEXT NOT NULL,
  facet_value TEXT NOT NULL,
  PRIMARY KEY(result_code,facet_kind,facet_value)
);

CREATE INDEX IF NOT EXISTS ix_research_result_facets_lookup
  ON research.result_facets(facet_kind,facet_value,result_code);

CREATE TABLE IF NOT EXISTS research.semantic_disagreements(
  disagreement_id TEXT PRIMARY KEY,
  disagreement_hash TEXT UNIQUE NOT NULL,
  run_code TEXT NOT NULL
    REFERENCES research.runs(run_code) ON DELETE RESTRICT,
  category TEXT NOT NULL,
  status TEXT NOT NULL,
  payload_json JSONB NOT NULL,
  shard_id TEXT,
  admitted BOOLEAN NOT NULL DEFAULT TRUE,
  attempt_count INTEGER,
  fencing_token BIGINT,
  CONSTRAINT ck_research_disagreements_legacy_fence CHECK(
    (attempt_count IS NULL AND fencing_token IS NULL) OR
    (attempt_count IS NOT NULL AND fencing_token IS NOT NULL AND
     attempt_count >= 1 AND fencing_token >= 1)
  ),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS research.shards(
  shard_id TEXT PRIMARY KEY,
  shard_hash TEXT UNIQUE NOT NULL,
  run_code TEXT NOT NULL
    REFERENCES research.runs(run_code) ON DELETE RESTRICT,
  shard_index INTEGER NOT NULL,
  status TEXT NOT NULL,
  attempt_count INTEGER NOT NULL DEFAULT 0,
  retry_count INTEGER NOT NULL DEFAULT 0,
  reclaim_count INTEGER NOT NULL DEFAULT 0,
  operation_counters_available BOOLEAN NOT NULL DEFAULT TRUE,
  fencing_token BIGINT NOT NULL DEFAULT 0,
  lease_owner TEXT,
  lease_expires_at TIMESTAMPTZ,
  heartbeat_at TIMESTAMPTZ,
  spec_json JSONB NOT NULL,
  result_json JSONB,
  error_json JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(run_code,shard_index)
);

CREATE INDEX IF NOT EXISTS ix_research_shards_claim
  ON research.shards(run_code,status,lease_expires_at,shard_index);

CREATE UNIQUE INDEX IF NOT EXISTS ux_research_shards_identity_run
  ON research.shards(shard_id,run_code);

ALTER TABLE research.results
  ADD COLUMN IF NOT EXISTS attempt_count INTEGER;
ALTER TABLE research.results
  ADD COLUMN IF NOT EXISTS fencing_token BIGINT;

ALTER TABLE research.semantic_disagreements
  ADD COLUMN IF NOT EXISTS attempt_count INTEGER;
ALTER TABLE research.semantic_disagreements
  ADD COLUMN IF NOT EXISTS fencing_token BIGINT;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conname='ck_research_results_legacy_fence'
      AND conrelid='research.results'::regclass
  ) THEN
    ALTER TABLE research.results
      ADD CONSTRAINT ck_research_results_legacy_fence CHECK(
        (attempt_count IS NULL AND fencing_token IS NULL) OR
        (attempt_count IS NOT NULL AND fencing_token IS NOT NULL AND
         attempt_count >= 1 AND fencing_token >= 1)
      );
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
         attempt_count >= 1 AND fencing_token >= 1)
      );
  END IF;
END $$;

CREATE TABLE IF NOT EXISTS research.result_occurrences(
  occurrence_hash TEXT PRIMARY KEY,
  result_code TEXT NOT NULL
    REFERENCES research.results(result_code) ON DELETE RESTRICT,
  run_code TEXT NOT NULL
    REFERENCES research.runs(run_code) ON DELETE RESTRICT,
  shard_id TEXT NOT NULL,
  attempt_count INTEGER NOT NULL CHECK(attempt_count >= 1),
  fencing_token BIGINT NOT NULL CHECK(fencing_token >= 1),
  admitted BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(result_code,shard_id,attempt_count,fencing_token),
  CONSTRAINT fk_research_result_occurrence_shard_run
    FOREIGN KEY(shard_id,run_code)
    REFERENCES research.shards(shard_id,run_code) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS ix_research_result_occurrences_admission
  ON research.result_occurrences(
    shard_id,
    attempt_count,
    fencing_token,
    admitted
  );

CREATE TABLE IF NOT EXISTS research.semantic_disagreement_occurrences(
  occurrence_hash TEXT PRIMARY KEY,
  disagreement_id TEXT NOT NULL
    REFERENCES research.semantic_disagreements(disagreement_id)
    ON DELETE RESTRICT,
  run_code TEXT NOT NULL
    REFERENCES research.runs(run_code) ON DELETE RESTRICT,
  shard_id TEXT NOT NULL,
  attempt_count INTEGER NOT NULL CHECK(attempt_count >= 1),
  fencing_token BIGINT NOT NULL CHECK(fencing_token >= 1),
  admitted BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(disagreement_id,shard_id,attempt_count,fencing_token),
  CONSTRAINT fk_research_disagreement_occurrence_shard_run
    FOREIGN KEY(shard_id,run_code)
    REFERENCES research.shards(shard_id,run_code) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS ix_research_disagreement_occurrences_admission
  ON research.semantic_disagreement_occurrences(
    shard_id,
    attempt_count,
    fencing_token,
    admitted
  );

CREATE TABLE IF NOT EXISTS research.sequential_search_receipt_bundles(
  receipt_hash TEXT PRIMARY KEY,
  seed BIGINT NOT NULL,
  receipt_json JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_research_receipt_bundle_hash_seed
  ON research.sequential_search_receipt_bundles(receipt_hash,seed);

CREATE TABLE IF NOT EXISTS research.sequential_search_receipt_occurrences(
  occurrence_hash TEXT PRIMARY KEY,
  receipt_hash TEXT NOT NULL,
  run_code TEXT NOT NULL
    REFERENCES research.runs(run_code) ON DELETE RESTRICT,
  shard_id TEXT NOT NULL,
  attempt_count INTEGER NOT NULL CHECK(attempt_count >= 1),
  fencing_token BIGINT NOT NULL CHECK(fencing_token >= 1),
  seed BIGINT NOT NULL,
  admitted BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(run_code,shard_id,attempt_count,fencing_token,seed),
  CONSTRAINT fk_research_receipt_occurrence_bundle_seed
    FOREIGN KEY(receipt_hash,seed)
    REFERENCES research.sequential_search_receipt_bundles(receipt_hash,seed)
    ON DELETE RESTRICT,
  CONSTRAINT fk_research_receipt_occurrence_shard_run
    FOREIGN KEY(shard_id,run_code)
    REFERENCES research.shards(shard_id,run_code) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS ix_research_search_receipts_library
  ON research.sequential_search_receipt_occurrences(
    admitted,
    run_code,
    created_at DESC,
    occurrence_hash
  );

CREATE INDEX IF NOT EXISTS ix_research_search_receipts_admission
  ON research.sequential_search_receipt_occurrences(
    shard_id,
    attempt_count,
    fencing_token,
    admitted
  );

CREATE UNIQUE INDEX IF NOT EXISTS ux_research_search_receipts_admitted_seed
  ON research.sequential_search_receipt_occurrences(run_code,seed)
  WHERE admitted;

COMMIT;
