BEGIN;

SELECT pg_advisory_xact_lock(
  hashtextextended('balootx:research-store-v1:bootstrap', 0)
);

CREATE SCHEMA IF NOT EXISTS research;

CREATE TABLE IF NOT EXISTS research.recipes(
  recipe_code TEXT PRIMARY KEY,
  recipe_hash TEXT UNIQUE NOT NULL,
  status TEXT NOT NULL,
  payload_json JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_research_recipes_page
  ON research.recipes(created_at DESC,recipe_code COLLATE "C" ASC);

CREATE INDEX IF NOT EXISTS ix_research_recipes_status_page
  ON research.recipes(
    status,
    created_at DESC,
    recipe_code COLLATE "C" ASC
  );

CREATE TABLE IF NOT EXISTS research.runs(
  run_code TEXT PRIMARY KEY,
  run_hash TEXT UNIQUE NOT NULL,
  recipe_code TEXT NOT NULL REFERENCES research.recipes(recipe_code),
  recipe_hash TEXT NOT NULL,
  status TEXT NOT NULL,
  progress DOUBLE PRECISION NOT NULL,
  payload_json JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS research.recipe_mutation_receipts(
  idempotency_key TEXT PRIMARY KEY,
  mutation_kind TEXT NOT NULL,
  request_intent_hash TEXT NOT NULL,
  recipe_code TEXT NOT NULL
    REFERENCES research.recipes(recipe_code) ON DELETE RESTRICT,
  response_status INTEGER NOT NULL,
  response_json JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

DO $$
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
END $$;

DO $$
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
END $$;

COMMIT;
