BEGIN;

SELECT pg_advisory_xact_lock(
  hashtextextended('balootx:research-store-v1:bootstrap', 0)
);

ALTER TABLE research.results
  ADD COLUMN IF NOT EXISTS library_projection_json JSONB;

ALTER TABLE research.results
  ALTER COLUMN projection_version SET DEFAULT 4;

UPDATE research.results
SET projection_version = 0
WHERE projection_version < 4;

CREATE INDEX IF NOT EXISTS ix_research_results_library
  ON research.results(admitted,created_at DESC,result_code ASC);

COMMIT;
