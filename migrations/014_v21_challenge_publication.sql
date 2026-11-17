BEGIN;

CREATE TABLE IF NOT EXISTS learning_challenge_publication_reviews_v21(
    review_id TEXT PRIMARY KEY,
    challenge_id TEXT NOT NULL REFERENCES learning_challenges(challenge_id),
    reviewer_id TEXT NOT NULL,
    reviewer_role TEXT NOT NULL,
    decision TEXT NOT NULL CHECK(decision IN ('approve','reject')),
    evidence_sha256 CHAR(64) NOT NULL CHECK(evidence_sha256 ~ '^[0-9a-f]{64}$'),
    comment TEXT NOT NULL CHECK(length(comment) <= 2000),
    created_at TIMESTAMPTZ NOT NULL,
    UNIQUE(challenge_id, reviewer_id)
);
CREATE INDEX IF NOT EXISTS ix_learning_challenge_publication_reviews_v21
    ON learning_challenge_publication_reviews_v21(challenge_id, created_at);

CREATE TABLE IF NOT EXISTS learning_challenge_publication_events_v21(
    event_id TEXT PRIMARY KEY,
    challenge_id TEXT NOT NULL REFERENCES learning_challenges(challenge_id),
    actor_id TEXT NOT NULL,
    actor_role TEXT NOT NULL,
    from_status TEXT NOT NULL,
    to_status TEXT NOT NULL CHECK(to_status IN ('approved','published','retired')),
    evidence_sha256 CHAR(64) NOT NULL CHECK(evidence_sha256 ~ '^[0-9a-f]{64}$'),
    created_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_learning_challenge_publication_events_v21
    ON learning_challenge_publication_events_v21(challenge_id, created_at);

COMMIT;
