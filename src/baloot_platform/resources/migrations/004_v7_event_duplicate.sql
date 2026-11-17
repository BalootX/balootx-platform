-- V7 authoritative typed event history and Duplicate Baloot.
CREATE TABLE IF NOT EXISTS game_events_v7(
  event_id TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL DEFAULT 'default',
  game_id TEXT NOT NULL,
  round_id TEXT,
  sequence BIGINT NOT NULL,
  event_type TEXT NOT NULL,
  schema_version INTEGER NOT NULL,
  payload_json JSONB NOT NULL,
  payload_protobuf BYTEA,
  profile_bundle_hash CHAR(64) NOT NULL,
  engine_version TEXT NOT NULL,
  previous_event_hash CHAR(64),
  event_hash CHAR(64) NOT NULL UNIQUE,
  occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  actor_seat TEXT,
  actor_subject_id TEXT,
  UNIQUE(game_id,sequence)
);
CREATE INDEX IF NOT EXISTS game_events_v7_game_sequence_idx ON game_events_v7(game_id,sequence);
CREATE INDEX IF NOT EXISTS game_events_v7_occurred_idx ON game_events_v7(occurred_at);
CREATE TABLE IF NOT EXISTS game_event_heads_v7(
  game_id TEXT PRIMARY KEY,
  event_count BIGINT NOT NULL,
  head_hash CHAR(64),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS game_snapshots_v7(
  game_id TEXT NOT NULL,
  version BIGINT NOT NULL,
  through_sequence BIGINT NOT NULL,
  state_hash CHAR(64) NOT NULL,
  payload BYTEA NOT NULL,
  payload_format TEXT NOT NULL DEFAULT 'baloot.durable.v1',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY(game_id,version)
);

CREATE TABLE IF NOT EXISTS duplicate_boards_v7(
  board_id TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL DEFAULT 'default',
  version INTEGER NOT NULL,
  board_hash CHAR(64) NOT NULL UNIQUE,
  serial TEXT NOT NULL UNIQUE,
  profile_bundle_hash CHAR(64) NOT NULL,
  observer_seat TEXT NOT NULL,
  starting_match_score JSONB NOT NULL,
  public_event_prefix JSONB NOT NULL,
  trusted_payload BYTEA NOT NULL,
  bot_policy_hashes JSONB NOT NULL,
  scoring_profile TEXT NOT NULL,
  time_control JSONB NOT NULL,
  reveal_policy TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS duplicate_rooms_v7(
  room_id TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL DEFAULT 'default',
  board_id TEXT NOT NULL REFERENCES duplicate_boards_v7(board_id),
  room_code TEXT NOT NULL UNIQUE,
  mode TEXT NOT NULL,
  status TEXT NOT NULL,
  settings_json JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  closes_at TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS duplicate_runs_v7(
  run_id TEXT PRIMARY KEY,
  room_id TEXT NOT NULL REFERENCES duplicate_rooms_v7(room_id),
  board_id TEXT NOT NULL REFERENCES duplicate_boards_v7(board_id),
  subject_id TEXT NOT NULL,
  comparison_key CHAR(64) NOT NULL,
  state_hash CHAR(64),
  result_json JSONB,
  status TEXT NOT NULL,
  started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  completed_at TIMESTAMPTZ,
  UNIQUE(room_id,subject_id)
);
CREATE INDEX IF NOT EXISTS duplicate_runs_v7_board_idx ON duplicate_runs_v7(board_id,comparison_key,status);
CREATE TABLE IF NOT EXISTS duplicate_run_actions_v7(
  run_id TEXT NOT NULL REFERENCES duplicate_runs_v7(run_id) ON DELETE CASCADE,
  sequence BIGINT NOT NULL,
  action_json JSONB NOT NULL,
  state_hash CHAR(64) NOT NULL,
  occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY(run_id,sequence)
);

CREATE TABLE IF NOT EXISTS challenge_contract_verifications_v7(
  verification_id TEXT PRIMARY KEY,
  challenge_id TEXT NOT NULL,
  contract TEXT NOT NULL,
  accepted BOOLEAN NOT NULL,
  truth_level TEXT NOT NULL,
  copy_policy TEXT NOT NULL,
  winning_language_allowed BOOLEAN NOT NULL,
  metrics_json JSONB NOT NULL,
  reasons_json JSONB NOT NULL,
  profile_hash CHAR(64) NOT NULL,
  analysis_hash CHAR(64) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS challenge_contract_v7_challenge_idx ON challenge_contract_verifications_v7(challenge_id,created_at DESC);

-- Existing bytea columns now carry the typed V7 codec, never Python pickle.
ALTER TABLE game_snapshots ADD COLUMN IF NOT EXISTS payload_format TEXT NOT NULL DEFAULT 'baloot.durable.v1';
ALTER TABLE archived_games ADD COLUMN IF NOT EXISTS payload_format TEXT NOT NULL DEFAULT 'baloot.durable.v1';
ALTER TABLE learning_decisions ADD COLUMN IF NOT EXISTS payload_format TEXT NOT NULL DEFAULT 'baloot.durable.v1';
ALTER TABLE learning_challenges ADD COLUMN IF NOT EXISTS episode_format TEXT NOT NULL DEFAULT 'baloot.durable.v1';
