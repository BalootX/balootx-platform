CREATE INDEX IF NOT EXISTS idx_identity_active_search
  ON identity.subjects(kind DESC,display_name,subject_id) WHERE disabled_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_identity_friendships_high_low
  ON identity.friendships(subject_high,subject_low);
CREATE INDEX IF NOT EXISTS idx_identity_pending_requests
  ON identity.friend_requests(recipient_id,created_at DESC,request_id DESC)
  WHERE status='pending';
