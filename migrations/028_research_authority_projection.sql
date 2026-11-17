BEGIN;

SELECT pg_advisory_xact_lock(
  hashtextextended('balootx:research-store-v1:bootstrap', 0)
);

ALTER TABLE research.results
  ALTER COLUMN projection_version SET DEFAULT 5;

ALTER TABLE research.results
  ADD COLUMN IF NOT EXISTS payload_hash TEXT;

ALTER TABLE research.results
  ADD COLUMN IF NOT EXISTS library_projection_hash TEXT;

UPDATE research.results
SET projection_version = 0
WHERE projection_version < 5;

COMMIT;
