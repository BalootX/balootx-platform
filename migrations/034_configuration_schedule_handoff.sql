BEGIN;

SET LOCAL lock_timeout = '5s';

ALTER TABLE configuration.bundle_revisions
  ADD COLUMN IF NOT EXISTS scheduled_predecessor_id text,
  ADD COLUMN IF NOT EXISTS scheduled_successor_id text;

CREATE INDEX IF NOT EXISTS ix_configuration_scheduled_successor
  ON configuration.bundle_revisions(scheduled_successor_id)
  WHERE status = 'published' AND scheduled_successor_id IS NOT NULL;

CREATE OR REPLACE FUNCTION configuration.guard_scheduled_predecessor()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  RAISE EXCEPTION 'scheduled_predecessor_id is immutable after publication';
END;
$$;

DROP TRIGGER IF EXISTS trg_configuration_scheduled_predecessor_immutable ON configuration.bundle_revisions;
CREATE TRIGGER trg_configuration_scheduled_predecessor_immutable
  BEFORE UPDATE OF scheduled_predecessor_id ON configuration.bundle_revisions
  FOR EACH ROW WHEN (OLD.status <> 'draft' AND NEW.scheduled_predecessor_id IS DISTINCT FROM OLD.scheduled_predecessor_id)
  EXECUTE FUNCTION configuration.guard_scheduled_predecessor();

COMMIT;
