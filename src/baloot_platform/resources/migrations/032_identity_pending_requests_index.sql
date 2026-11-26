CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_identity_pending_requests
  ON identity.friend_requests(recipient_id,created_at DESC,request_id DESC)
  WHERE status='pending';
