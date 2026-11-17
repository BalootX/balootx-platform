-- V9 canonical shared لفة/صكّة environment commitments
ALTER TABLE duplicate.boards ADD COLUMN IF NOT EXISTS environment_hash text;
ALTER TABLE duplicate.boards ADD COLUMN IF NOT EXISTS deal_schedule_hash text;
ALTER TABLE duplicate.boards ADD COLUMN IF NOT EXISTS agent_snapshot_hash text;
ALTER TABLE duplicate.boards ADD COLUMN IF NOT EXISTS shared_profile_hash text;
ALTER TABLE duplicate.boards ADD COLUMN IF NOT EXISTS terminology_version text NOT NULL DEFAULT 'ar_sa_baloot_v2';
CREATE INDEX IF NOT EXISTS idx_duplicate_boards_environment_hash ON duplicate.boards(environment_hash);
