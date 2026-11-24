BEGIN;

SELECT pg_advisory_xact_lock(
  hashtextextended('balootx:identity-list-pagination-v1:bootstrap', 0)
);

CREATE INDEX IF NOT EXISTS idx_identity_friendships_high_low
  ON identity.friendships(subject_high,subject_low);
CREATE INDEX IF NOT EXISTS idx_identity_pending_requests
  ON identity.friend_requests(recipient_id,created_at DESC,request_id DESC)
  WHERE status='pending';

COMMIT;
