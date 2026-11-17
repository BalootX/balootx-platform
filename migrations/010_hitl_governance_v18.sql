BEGIN;
CREATE SCHEMA IF NOT EXISTS hitl;
CREATE TABLE IF NOT EXISTS hitl.review_cases (
 case_id TEXT PRIMARY KEY, decision_id TEXT NOT NULL, created_by TEXT NOT NULL,
 status TEXT NOT NULL, case_hash TEXT NOT NULL UNIQUE, payload JSONB NOT NULL,
 created_at TIMESTAMPTZ NOT NULL DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_hitl_cases_decision ON hitl.review_cases(decision_id);
CREATE TABLE IF NOT EXISTS hitl.blind_assignments (
 assignment_id TEXT PRIMARY KEY, case_id TEXT NOT NULL REFERENCES hitl.review_cases(case_id),
 reviewer_id TEXT NOT NULL, token_hash TEXT NOT NULL, status TEXT NOT NULL,
 device_cluster_hash TEXT NOT NULL, network_cluster_hash TEXT NOT NULL,
 payload JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now(), UNIQUE(case_id, reviewer_id)
);
CREATE TABLE IF NOT EXISTS hitl.review_submissions (
 submission_id TEXT PRIMARY KEY, case_id TEXT NOT NULL REFERENCES hitl.review_cases(case_id),
 assignment_id TEXT NOT NULL REFERENCES hitl.blind_assignments(assignment_id), reviewer_id TEXT NOT NULL,
 preferred_action JSONB NOT NULL, status TEXT NOT NULL, submission_hash TEXT NOT NULL UNIQUE,
 payload JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS hitl.validation_results (
 validation_id TEXT PRIMARY KEY, submission_id TEXT NOT NULL REFERENCES hitl.review_submissions(submission_id),
 payload JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS hitl.consensus_snapshots (
 consensus_id TEXT PRIMARY KEY, case_id TEXT NOT NULL REFERENCES hitl.review_cases(case_id),
 status TEXT NOT NULL, payload JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS hitl.reviewer_profiles (subject_id TEXT PRIMARY KEY, payload JSONB NOT NULL, updated_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS hitl.consent (subject_id TEXT PRIMARY KEY, payload JSONB NOT NULL, updated_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS hitl.moderation_flags (flag_id TEXT PRIMARY KEY, case_id TEXT NOT NULL REFERENCES hitl.review_cases(case_id), status TEXT NOT NULL, payload JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS hitl.adjudications (adjudication_id TEXT PRIMARY KEY, case_id TEXT NOT NULL REFERENCES hitl.review_cases(case_id), payload JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS hitl.training_candidates (candidate_id TEXT PRIMARY KEY, case_id TEXT NOT NULL REFERENCES hitl.review_cases(case_id), status TEXT NOT NULL, payload JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS hitl.challenger_evaluations (evaluation_id TEXT PRIMARY KEY, candidate_id TEXT NOT NULL REFERENCES hitl.training_candidates(candidate_id), payload JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS hitl.improvement_proposals (proposal_id TEXT PRIMARY KEY, target TEXT NOT NULL, status TEXT NOT NULL, payload JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS hitl.improvement_evaluations (evaluation_id TEXT PRIMARY KEY, proposal_id TEXT NOT NULL REFERENCES hitl.improvement_proposals(proposal_id), payload JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS hitl.events (
 event_id BIGSERIAL PRIMARY KEY, aggregate_type TEXT NOT NULL, aggregate_id TEXT NOT NULL,
 sequence INTEGER NOT NULL, event_type TEXT NOT NULL, payload JSONB NOT NULL,
 previous_hash TEXT NOT NULL, event_hash TEXT NOT NULL UNIQUE, occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),
 UNIQUE(aggregate_type, aggregate_id, sequence)
);
ALTER TABLE hitl.review_cases ENABLE ROW LEVEL SECURITY;
ALTER TABLE hitl.blind_assignments ENABLE ROW LEVEL SECURITY;
ALTER TABLE hitl.review_submissions ENABLE ROW LEVEL SECURITY;
ALTER TABLE hitl.consent ENABLE ROW LEVEL SECURITY;
-- Applications set baloot.subject_id. Service/admin connections may bypass through a dedicated role.
DROP POLICY IF EXISTS hitl_case_owner ON hitl.review_cases;
CREATE POLICY hitl_case_owner ON hitl.review_cases USING (created_by = current_setting('baloot.subject_id', true));
DROP POLICY IF EXISTS hitl_assignment_reviewer ON hitl.blind_assignments;
CREATE POLICY hitl_assignment_reviewer ON hitl.blind_assignments USING (reviewer_id = current_setting('baloot.subject_id', true));
DROP POLICY IF EXISTS hitl_submission_reviewer ON hitl.review_submissions;
CREATE POLICY hitl_submission_reviewer ON hitl.review_submissions USING (reviewer_id = current_setting('baloot.subject_id', true));
DROP POLICY IF EXISTS hitl_consent_subject ON hitl.consent;
CREATE POLICY hitl_consent_subject ON hitl.consent USING (subject_id = current_setting('baloot.subject_id', true));
COMMIT;
