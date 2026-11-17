BEGIN;

CREATE TABLE IF NOT EXISTS challenge_locator_attempts_v1(
  attempt_id VARCHAR(40) PRIMARY KEY,
  code VARCHAR(32) NOT NULL REFERENCES challenge_locators_v1(code),
  participant_key VARCHAR(40) NOT NULL,
  chosen_action_json JSONB NOT NULL,
  response_seconds DOUBLE PRECISION,
  confidence DOUBLE PRECISION,
  changed_answer BOOLEAN NOT NULL DEFAULT FALSE,
  created_at BIGINT NOT NULL,
  UNIQUE(code, participant_key),
  CHECK(response_seconds IS NULL OR (response_seconds >= 0 AND response_seconds <= 3600)),
  CHECK(confidence IS NULL OR (confidence >= 0 AND confidence <= 1))
);

CREATE INDEX IF NOT EXISTS challenge_locator_attempts_code_v1_idx
  ON challenge_locator_attempts_v1(code, created_at, attempt_id);

COMMIT;
