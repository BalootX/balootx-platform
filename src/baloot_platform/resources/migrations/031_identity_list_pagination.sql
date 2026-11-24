CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_identity_friendships_high_low
  ON identity.friendships(subject_high,subject_low);
