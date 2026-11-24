BEGIN;

SET LOCAL lock_timeout = '5s';

ALTER TABLE configuration.profile_revisions
  ADD COLUMN IF NOT EXISTS local_schedule jsonb;
ALTER TABLE configuration.bundle_revisions
  ADD COLUMN IF NOT EXISTS local_schedule jsonb;

DROP INDEX IF EXISTS configuration.uq_configuration_profile_content;
CREATE UNIQUE INDEX IF NOT EXISTS uq_configuration_profile_schedule
  ON configuration.profile_revisions (
    family,scope,scope_key,tenant_id,environment,payload_sha256,
    effective_from,effective_until,(local_schedule - 'zoneDataVersion')
  ) NULLS NOT DISTINCT;

CREATE OR REPLACE FUNCTION configuration.guard_local_schedule()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  RAISE EXCEPTION 'local_schedule is immutable; create a new configuration revision';
END;
$$;

DROP TRIGGER IF EXISTS trg_configuration_profile_local_schedule_immutable ON configuration.profile_revisions;
CREATE TRIGGER trg_configuration_profile_local_schedule_immutable
  BEFORE UPDATE OF local_schedule ON configuration.profile_revisions
  FOR EACH ROW WHEN (NEW.local_schedule IS DISTINCT FROM OLD.local_schedule)
  EXECUTE FUNCTION configuration.guard_local_schedule();

DROP TRIGGER IF EXISTS trg_configuration_bundle_local_schedule_immutable ON configuration.bundle_revisions;
CREATE TRIGGER trg_configuration_bundle_local_schedule_immutable
  BEFORE UPDATE OF local_schedule ON configuration.bundle_revisions
  FOR EACH ROW WHEN (NEW.local_schedule IS DISTINCT FROM OLD.local_schedule)
  EXECUTE FUNCTION configuration.guard_local_schedule();

COMMIT;
