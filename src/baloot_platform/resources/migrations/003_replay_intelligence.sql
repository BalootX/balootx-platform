ALTER TABLE learning_decisions
  ADD COLUMN IF NOT EXISTS decision_metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb;

CREATE TABLE IF NOT EXISTS learning_decision_audits(
  decision_audit_id TEXT PRIMARY KEY,
  decision_id TEXT NOT NULL REFERENCES learning_decisions(decision_id) ON DELETE CASCADE,
  audit_hash CHAR(64) NOT NULL UNIQUE,
  quality_label TEXT NOT NULL,
  quality_score DOUBLE PRECISION NOT NULL,
  report_json JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS learning_decision_audits_decision_idx
  ON learning_decision_audits(decision_id, created_at DESC);

CREATE TABLE IF NOT EXISTS learning_game_audits(
  audit_id TEXT PRIMARY KEY,
  game_id TEXT NOT NULL,
  game_hash CHAR(64),
  report_hash CHAR(64) NOT NULL UNIQUE,
  mode TEXT NOT NULL,
  bundle_hash CHAR(64) NOT NULL,
  engine_version TEXT NOT NULL,
  report_json JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS learning_game_audits_game_idx
  ON learning_game_audits(game_id, created_at DESC);
CREATE INDEX IF NOT EXISTS learning_game_audits_hash_idx
  ON learning_game_audits(game_hash, created_at DESC);

CREATE TABLE IF NOT EXISTS learning_difficulty_items(
  challenge_id TEXT PRIMARY KEY,
  engine_difficulty DOUBLE PRECISION NOT NULL,
  latent_difficulty DOUBLE PRECISION NOT NULL,
  discrimination DOUBLE PRECISION NOT NULL,
  response_count BIGINT NOT NULL,
  distinct_players BIGINT NOT NULL,
  standard_error DOUBLE PRECISION NOT NULL,
  calibrated_score DOUBLE PRECISION NOT NULL,
  label TEXT NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS learning_difficulty_players(
  player_id TEXT PRIMARY KEY,
  ability DOUBLE PRECISION NOT NULL,
  response_count BIGINT NOT NULL,
  standard_error DOUBLE PRECISION NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS learning_difficulty_responses(
  response_id TEXT PRIMARY KEY,
  challenge_id TEXT NOT NULL REFERENCES learning_difficulty_items(challenge_id) ON DELETE CASCADE,
  player_id TEXT NOT NULL REFERENCES learning_difficulty_players(player_id) ON DELETE CASCADE,
  correct BOOLEAN NOT NULL,
  response_seconds DOUBLE PRECISION,
  confidence DOUBLE PRECISION,
  chosen_action_json JSONB,
  trace_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS learning_difficulty_responses_item_idx
  ON learning_difficulty_responses(challenge_id, created_at DESC);
CREATE INDEX IF NOT EXISTS learning_difficulty_responses_player_idx
  ON learning_difficulty_responses(player_id, created_at DESC);

CREATE TABLE IF NOT EXISTS learning_video_records(
  video_id TEXT PRIMARY KEY,
  owner_kind TEXT NOT NULL,
  owner_id TEXT NOT NULL,
  game_hash CHAR(64),
  episode_hash CHAR(64) NOT NULL,
  renderer_id TEXT NOT NULL,
  manifest_json JSONB NOT NULL,
  timeline_json JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS learning_video_records_owner_idx
  ON learning_video_records(owner_kind, owner_id, created_at DESC);
CREATE INDEX IF NOT EXISTS learning_video_records_game_idx
  ON learning_video_records(game_hash, created_at DESC);
