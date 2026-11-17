CREATE SCHEMA IF NOT EXISTS adjudication;
CREATE TABLE IF NOT EXISTS adjudication.cases(
  case_id text PRIMARY KEY,
  game_id text NOT NULL,
  status text NOT NULL,
  violation text NOT NULL,
  updated_at timestamptz NOT NULL,
  payload_json jsonb NOT NULL,
  case_hash text NOT NULL,
  previous_hash text NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_adjudication_cases_game ON adjudication.cases(game_id,updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_adjudication_cases_status ON adjudication.cases(status,updated_at DESC);

CREATE SCHEMA IF NOT EXISTS live;
CREATE TABLE IF NOT EXISTS live.rooms(
  room_id text PRIMARY KEY,
  room_code text NOT NULL UNIQUE,
  game_id text NOT NULL,
  status text NOT NULL,
  updated_at timestamptz NOT NULL,
  payload_json jsonb NOT NULL,
  room_hash text NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_live_rooms_game ON live.rooms(game_id,updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_live_rooms_status ON live.rooms(status,updated_at DESC);
