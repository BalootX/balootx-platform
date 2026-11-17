CREATE TABLE IF NOT EXISTS archived_games(
  game_hash CHAR(64) PRIMARY KEY,
  serial TEXT NOT NULL UNIQUE,
  game_id TEXT NOT NULL,
  parent_game_hash CHAR(64),
  bundle_hash CHAR(64) NOT NULL,
  engine_version TEXT NOT NULL,
  state_hash CHAR(64) NOT NULL,
  public_json JSONB NOT NULL,
  trusted_payload BYTEA NOT NULL,
  status TEXT NOT NULL,
  label TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS archived_games_game_id_idx ON archived_games(game_id);
CREATE INDEX IF NOT EXISTS archived_games_created_idx ON archived_games(created_at DESC);

CREATE TABLE IF NOT EXISTS learning_decisions(
  decision_id TEXT PRIMARY KEY,
  game_id TEXT NOT NULL,
  game_hash CHAR(64),
  game_version BIGINT NOT NULL,
  ordinal INTEGER NOT NULL,
  decision_kind TEXT NOT NULL,
  actor TEXT NOT NULL,
  phase TEXT NOT NULL,
  legal_actions_json JSONB NOT NULL,
  selected_action_json JSONB,
  public_state_json JSONB NOT NULL,
  trusted_payload BYTEA NOT NULL,
  decision_hash CHAR(64) NOT NULL UNIQUE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(game_id, ordinal)
);
CREATE INDEX IF NOT EXISTS learning_decisions_game_idx ON learning_decisions(game_id, ordinal);
CREATE INDEX IF NOT EXISTS learning_decisions_hash_idx ON learning_decisions(game_hash, ordinal);

CREATE TABLE IF NOT EXISTS learning_analyses(
  analysis_id TEXT PRIMARY KEY,
  decision_id TEXT NOT NULL REFERENCES learning_decisions(decision_id) ON DELETE CASCADE,
  engine_version TEXT NOT NULL,
  bundle_hash CHAR(64) NOT NULL,
  config_hash CHAR(64) NOT NULL,
  result_hash CHAR(64) NOT NULL,
  config_json JSONB NOT NULL,
  result_json JSONB NOT NULL,
  supersedes_analysis_id TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS learning_analyses_decision_idx ON learning_analyses(decision_id, created_at DESC);

CREATE TABLE IF NOT EXISTS learning_feedback(
  feedback_id TEXT PRIMARY KEY,
  decision_id TEXT NOT NULL REFERENCES learning_decisions(decision_id) ON DELETE CASCADE,
  preferred_action_json JSONB NOT NULL,
  engine_action_json JSONB,
  confidence DOUBLE PRECISION NOT NULL CHECK(confidence BETWEEN 0 AND 1),
  rationale TEXT NOT NULL,
  source TEXT NOT NULL,
  reviewer TEXT,
  interestingness DOUBLE PRECISION NOT NULL DEFAULT 0,
  model_error DOUBLE PRECISION NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS learning_feedback_decision_idx ON learning_feedback(decision_id, created_at DESC);

CREATE TABLE IF NOT EXISTS learning_datasets(
  dataset_id TEXT PRIMARY KEY,
  purpose TEXT NOT NULL,
  artifact_path TEXT NOT NULL,
  manifest_json JSONB NOT NULL,
  example_count BIGINT NOT NULL,
  sha256 CHAR(64) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS learning_models(
  model_id TEXT PRIMARY KEY,
  plugin_id TEXT NOT NULL,
  version TEXT NOT NULL,
  artifact_path TEXT NOT NULL,
  manifest_json JSONB NOT NULL,
  metrics_json JSONB,
  dataset_id TEXT,
  status TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS learning_models_status_idx ON learning_models(status, updated_at DESC);

CREATE TABLE IF NOT EXISTS learning_challenges(
  challenge_id TEXT PRIMARY KEY,
  decision_id TEXT,
  fingerprint CHAR(64) NOT NULL UNIQUE,
  archetype TEXT NOT NULL,
  publishability DOUBLE PRECISION NOT NULL,
  status TEXT NOT NULL,
  payload_json JSONB NOT NULL,
  episode_payload BYTEA,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS learning_challenges_status_idx ON learning_challenges(status, publishability DESC);

CREATE TABLE IF NOT EXISTS learning_artifacts(
  artifact_id TEXT PRIMARY KEY,
  owner_kind TEXT NOT NULL,
  owner_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  uri TEXT NOT NULL,
  sha256 CHAR(64),
  metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS learning_artifacts_owner_idx ON learning_artifacts(owner_kind, owner_id);
