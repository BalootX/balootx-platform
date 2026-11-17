-- V8 optional identity and shareable same-start Sakka.
CREATE SCHEMA IF NOT EXISTS identity;
CREATE SCHEMA IF NOT EXISTS duplicate;

CREATE TABLE IF NOT EXISTS identity.subjects(
  subject_id text PRIMARY KEY,
  kind text NOT NULL CHECK(kind IN ('guest','local','oidc','service')),
  username text UNIQUE,
  display_name text NOT NULL,
  password_salt bytea,
  password_hash bytea,
  external_issuer text,
  external_subject text,
  created_at timestamptz NOT NULL DEFAULT now(),
  disabled_at timestamptz,
  UNIQUE(external_issuer,external_subject)
);

CREATE TABLE IF NOT EXISTS identity.sessions(
  session_id text PRIMARY KEY,
  subject_id text NOT NULL REFERENCES identity.subjects(subject_id) ON DELETE CASCADE,
  token_hash text NOT NULL UNIQUE,
  csrf_token text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  expires_at timestamptz NOT NULL,
  last_seen_at timestamptz NOT NULL,
  user_agent_hash text,
  revoked_at timestamptz
);
CREATE INDEX IF NOT EXISTS idx_identity_sessions_subject ON identity.sessions(subject_id,expires_at);

CREATE TABLE IF NOT EXISTS identity.friendships(
  subject_low text NOT NULL REFERENCES identity.subjects(subject_id) ON DELETE CASCADE,
  subject_high text NOT NULL REFERENCES identity.subjects(subject_id) ON DELETE CASCADE,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY(subject_low,subject_high),
  CHECK(subject_low < subject_high)
);

CREATE TABLE IF NOT EXISTS identity.friend_requests(
  request_id text PRIMARY KEY,
  sender_id text NOT NULL REFERENCES identity.subjects(subject_id) ON DELETE CASCADE,
  recipient_id text NOT NULL REFERENCES identity.subjects(subject_id) ON DELETE CASCADE,
  status text NOT NULL CHECK(status IN ('pending','accepted','declined','cancelled')),
  created_at timestamptz NOT NULL DEFAULT now(),
  responded_at timestamptz
);

CREATE TABLE IF NOT EXISTS duplicate.boards(
  board_id text PRIMARY KEY,
  version integer NOT NULL,
  board_hash text NOT NULL UNIQUE,
  serial text NOT NULL UNIQUE,
  share_code text NOT NULL UNIQUE,
  bundle_hash text NOT NULL,
  observer_seat text NOT NULL,
  scope text NOT NULL CHECK(scope IN ('round','match')),
  maximum_rounds integer NOT NULL,
  starting_match_score jsonb NOT NULL,
  starting_round_number integer NOT NULL,
  public_event_prefix_hash text NOT NULL,
  bot_policy_hashes jsonb NOT NULL,
  scoring_profile text NOT NULL,
  time_control jsonb NOT NULL,
  reveal_policy text NOT NULL,
  trusted_payload bytea NOT NULL,
  trusted_payload_hash text NOT NULL,
  created_by text NOT NULL,
  visibility text NOT NULL CHECK(visibility IN ('private','unlisted','public')),
  status text NOT NULL,
  label text,
  created_at timestamptz NOT NULL DEFAULT now(),
  signature_algorithm text,
  signature_key_id text,
  board_signature text
);
ALTER TABLE duplicate.boards ADD COLUMN IF NOT EXISTS signature_algorithm text;
ALTER TABLE duplicate.boards ADD COLUMN IF NOT EXISTS signature_key_id text;
ALTER TABLE duplicate.boards ADD COLUMN IF NOT EXISTS board_signature text;

CREATE TABLE IF NOT EXISTS duplicate.rooms(
  room_id text PRIMARY KEY,
  board_id text NOT NULL REFERENCES duplicate.boards(board_id),
  room_code text NOT NULL UNIQUE,
  created_by text NOT NULL,
  mode text NOT NULL,
  status text NOT NULL,
  max_players integer NOT NULL,
  allow_guests boolean NOT NULL,
  settings_json jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  expires_at timestamptz
);

CREATE TABLE IF NOT EXISTS duplicate.members(
  room_id text NOT NULL REFERENCES duplicate.rooms(room_id) ON DELETE CASCADE,
  subject_id text NOT NULL,
  display_name text NOT NULL,
  joined_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY(room_id,subject_id)
);

CREATE TABLE IF NOT EXISTS duplicate.runs(
  run_id text PRIMARY KEY,
  room_id text NOT NULL REFERENCES duplicate.rooms(room_id),
  board_id text NOT NULL REFERENCES duplicate.boards(board_id),
  subject_id text NOT NULL,
  comparison_key text NOT NULL,
  status text NOT NULL,
  version integer NOT NULL,
  rounds_completed integer NOT NULL,
  session_payload bytea NOT NULL,
  state_hash text,
  result_json jsonb,
  analysis_status text NOT NULL,
  started_at timestamptz NOT NULL DEFAULT now(),
  completed_at timestamptz,
  UNIQUE(room_id,subject_id)
);

CREATE TABLE IF NOT EXISTS duplicate.run_actions(
  run_id text NOT NULL REFERENCES duplicate.runs(run_id) ON DELETE CASCADE,
  sequence integer NOT NULL,
  action_json jsonb NOT NULL,
  state_hash text NOT NULL,
  pre_state_payload bytea NOT NULL,
  legal_actions_json jsonb NOT NULL,
  decision_id text,
  pre_state_payload_hash text NOT NULL DEFAULT '',
  previous_action_hash text NOT NULL DEFAULT '',
  action_hash text NOT NULL DEFAULT '',
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY(run_id,sequence)
);
ALTER TABLE duplicate.run_actions ADD COLUMN IF NOT EXISTS decision_id text;
ALTER TABLE duplicate.run_actions ADD COLUMN IF NOT EXISTS pre_state_payload_hash text NOT NULL DEFAULT '';
ALTER TABLE duplicate.run_actions ADD COLUMN IF NOT EXISTS previous_action_hash text NOT NULL DEFAULT '';
ALTER TABLE duplicate.run_actions ADD COLUMN IF NOT EXISTS action_hash text NOT NULL DEFAULT '';

CREATE TABLE IF NOT EXISTS duplicate.room_events(
  room_id text NOT NULL REFERENCES duplicate.rooms(room_id) ON DELETE CASCADE,
  sequence integer NOT NULL,
  event_type text NOT NULL,
  subject_id text,
  payload_json jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY(room_id,sequence)
);

CREATE TABLE IF NOT EXISTS duplicate.run_audits(
  run_id text PRIMARY KEY REFERENCES duplicate.runs(run_id) ON DELETE CASCADE,
  audit_hash text NOT NULL,
  audit_json jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_duplicate_runs_leaderboard ON duplicate.runs(board_id,comparison_key,status);
CREATE INDEX IF NOT EXISTS idx_duplicate_room_events_stream ON duplicate.room_events(room_id,sequence);
