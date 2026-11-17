BEGIN;

SELECT pg_advisory_xact_lock(
  hashtextextended('balootx:shared-run-access-v1:bootstrap', 0)
);

ALTER TABLE duplicate.runs
  ADD COLUMN IF NOT EXISTS game_id text;

CREATE UNIQUE INDEX IF NOT EXISTS idx_duplicate_runs_game
  ON duplicate.runs(game_id)
  WHERE game_id IS NOT NULL;

COMMIT;
