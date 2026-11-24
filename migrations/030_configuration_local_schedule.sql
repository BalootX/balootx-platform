BEGIN;

ALTER TABLE configuration.profile_revisions
  ADD COLUMN IF NOT EXISTS local_schedule jsonb;
ALTER TABLE configuration.bundle_revisions
  ADD COLUMN IF NOT EXISTS local_schedule jsonb;

CREATE OR REPLACE FUNCTION configuration.guard_local_schedule()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  RAISE EXCEPTION 'local_schedule is immutable; create a new configuration revision';
END;
$$;

DROP TRIGGER IF EXISTS guard_local_schedule ON configuration.profile_revisions;
CREATE TRIGGER guard_local_schedule
  BEFORE UPDATE OF local_schedule ON configuration.profile_revisions
  FOR EACH ROW WHEN (NEW.local_schedule IS DISTINCT FROM OLD.local_schedule)
  EXECUTE FUNCTION configuration.guard_local_schedule();

DROP TRIGGER IF EXISTS guard_local_schedule ON configuration.bundle_revisions;
CREATE TRIGGER guard_local_schedule
  BEFORE UPDATE OF local_schedule ON configuration.bundle_revisions
  FOR EACH ROW WHEN (NEW.local_schedule IS DISTINCT FROM OLD.local_schedule)
  EXECUTE FUNCTION configuration.guard_local_schedule();

COMMIT;
