-- V9 structured human/expert review evidence
CREATE TABLE IF NOT EXISTS learning_structured_reviews(
  review_id text PRIMARY KEY,
  decision_id text NOT NULL REFERENCES learning_decisions(decision_id),
  reviewer_id text NOT NULL,
  reviewer_role text NOT NULL,
  preferred_action_json jsonb NOT NULL,
  confidence double precision NOT NULL CHECK(confidence BETWEEN 0 AND 1),
  reason_codes_json jsonb NOT NULL,
  rationale text NOT NULL,
  evidence_json jsonb NOT NULL,
  view_mode text NOT NULL,
  source text NOT NULL,
  engine_validation text NOT NULL,
  engine_regret double precision,
  engine_win_drop double precision,
  review_hash text NOT NULL UNIQUE,
  created_at timestamptz NOT NULL,
  UNIQUE(decision_id, reviewer_id, review_hash)
);
CREATE INDEX IF NOT EXISTS idx_learning_structured_reviews_decision
  ON learning_structured_reviews(decision_id, created_at DESC);
