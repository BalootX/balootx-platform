BEGIN;

SELECT pg_advisory_xact_lock(
  hashtextextended('balootx:configuration-store-v25:migration',0)
);
SELECT set_config('app.role','service',true);

-- The V11 trigger does not know every V12 column, while an earlier attempted
-- V25 can already guard them.  Suspend both guards transactionally while the
-- deterministic legacy normalization below runs; final guards are recreated.
DROP TRIGGER IF EXISTS trg_configuration_profile_immutable
  ON configuration.profile_revisions;
DROP TRIGGER IF EXISTS trg_configuration_bundle_immutable
  ON configuration.bundle_revisions;

-- V25 closes configuration runtime/schema drift without rewriting any
-- previously published migration.  The rollout percentage remains strictly
-- positive, but NUMERIC preserves fractional percentages exactly.
ALTER TABLE configuration.profile_revisions
  DROP CONSTRAINT IF EXISTS profile_revisions_scope_check,
  DROP CONSTRAINT IF EXISTS configuration_profile_scope_v25,
  DROP CONSTRAINT IF EXISTS configuration_profile_dimensions_v25,
  DROP CONSTRAINT IF EXISTS configuration_profile_rollout_v25;
ALTER TABLE configuration.bundle_revisions
  DROP CONSTRAINT IF EXISTS bundle_revisions_scope_check,
  DROP CONSTRAINT IF EXISTS configuration_bundle_scope_v25,
  DROP CONSTRAINT IF EXISTS configuration_bundle_dimensions_v25,
  DROP CONSTRAINT IF EXISTS configuration_bundle_rollout_v25;

DROP INDEX IF EXISTS configuration.uq_configuration_profile_content;
DROP INDEX IF EXISTS configuration.uq_configuration_profile_content_v20;
DROP INDEX IF EXISTS configuration.uq_configuration_profile_v20;
DROP INDEX IF EXISTS configuration.ix_configuration_profile_lookup;
DROP INDEX IF EXISTS configuration.ix_configuration_profile_lookup_v25;
DROP INDEX IF EXISTS configuration.uq_configuration_profile_content_v25;
DROP INDEX IF EXISTS configuration.uq_configuration_profile_revision_v25;
DROP INDEX IF EXISTS configuration.uq_configuration_profile_published_v25;

DROP INDEX IF EXISTS configuration.uq_configuration_bundle_content;
DROP INDEX IF EXISTS configuration.uq_configuration_bundle_content_v20;
DROP INDEX IF EXISTS configuration.uq_configuration_bundle_v20;
DROP INDEX IF EXISTS configuration.ix_configuration_bundle_lookup;
DROP INDEX IF EXISTS configuration.ix_configuration_bundle_content_v21;
DROP INDEX IF EXISTS configuration.ix_configuration_bundle_lookup_v25;
DROP INDEX IF EXISTS configuration.ix_configuration_bundle_content_v25;
DROP INDEX IF EXISTS configuration.uq_configuration_bundle_revision_v25;
DROP INDEX IF EXISTS configuration.uq_configuration_bundle_published_v25;

ALTER TABLE configuration.profile_revisions
  ALTER COLUMN rollout_percentage TYPE NUMERIC
    USING rollout_percentage::NUMERIC,
  ALTER COLUMN rollout_percentage SET DEFAULT 100::NUMERIC,
  ALTER COLUMN rollout_seed SET DEFAULT 'default';
ALTER TABLE configuration.bundle_revisions
  ALTER COLUMN rollout_percentage TYPE NUMERIC
    USING rollout_percentage::NUMERIC,
  ALTER COLUMN rollout_percentage SET DEFAULT 100::NUMERIC,
  ALTER COLUMN rollout_seed SET DEFAULT 'default';

-- V11 rows predate scope_key.  V12 assigned '*', so the tenant key can be
-- recovered losslessly from its required tenant_id.  Other legacy scopes do
-- not have a recoverable external identifier and retain their non-empty '*'.
UPDATE configuration.profile_revisions
SET scope_key=tenant_id
WHERE scope='tenant'
  AND tenant_id IS NOT NULL
  AND btrim(tenant_id)<>''
  AND scope_key='*';
UPDATE configuration.bundle_revisions
SET scope_key=tenant_id
WHERE scope='tenant'
  AND tenant_id IS NOT NULL
  AND btrim(tenant_id)<>''
  AND scope_key='*';

-- V11 uniqueness included version, although sequence ownership does not.
-- Retain one existing sequence and move only colliding rows above that exact
-- logical identity's prior maximum.  No row, payload, revision id, or
-- supersession link is discarded.
WITH marked AS (
  SELECT
    revision_id,family,profile_id,scope,scope_key,tenant_id,environment,
    sequence,created_at,
    row_number() OVER (
      PARTITION BY
        family,profile_id,scope,scope_key,tenant_id,environment,sequence
      ORDER BY created_at,revision_id
    ) AS duplicate_rank,
    max(sequence) OVER (
      PARTITION BY family,profile_id,scope,scope_key,tenant_id,environment
    ) AS prior_maximum
  FROM configuration.profile_revisions
), reassigned AS (
  SELECT
    revision_id,
    prior_maximum+row_number() OVER (
      PARTITION BY family,profile_id,scope,scope_key,tenant_id,environment
      ORDER BY sequence,created_at,revision_id
    ) AS new_sequence
  FROM marked
  WHERE duplicate_rank>1
)
UPDATE configuration.profile_revisions AS revision
SET sequence=reassigned.new_sequence
FROM reassigned
WHERE revision.revision_id=reassigned.revision_id;

WITH marked AS (
  SELECT
    revision_id,bundle_id,scope,scope_key,tenant_id,environment,
    sequence,created_at,
    row_number() OVER (
      PARTITION BY
        bundle_id,scope,scope_key,tenant_id,environment,sequence
      ORDER BY created_at,revision_id
    ) AS duplicate_rank,
    max(sequence) OVER (
      PARTITION BY bundle_id,scope,scope_key,tenant_id,environment
    ) AS prior_maximum
  FROM configuration.bundle_revisions
), reassigned AS (
  SELECT
    revision_id,
    prior_maximum+row_number() OVER (
      PARTITION BY bundle_id,scope,scope_key,tenant_id,environment
      ORDER BY sequence,created_at,revision_id
    ) AS new_sequence
  FROM marked
  WHERE duplicate_rank>1
)
UPDATE configuration.bundle_revisions AS revision
SET sequence=reassigned.new_sequence
FROM reassigned
WHERE revision.revision_id=reassigned.revision_id;

ALTER TABLE configuration.profile_revisions
  ADD CONSTRAINT configuration_profile_scope_v25 CHECK (
    scope IN (
      'global','environment','tenant','user_experiment',
      'room','shared_game','video_profile'
    )
  ),
  ADD CONSTRAINT configuration_profile_dimensions_v25 CHECK (
    (scope='global' AND scope_key='*' AND tenant_id IS NULL)
    OR (scope='environment' AND tenant_id IS NULL
        AND environment IS NOT NULL AND btrim(environment)<>''
        AND scope_key=environment)
    OR (scope='tenant' AND tenant_id IS NOT NULL
        AND btrim(tenant_id)<>'' AND scope_key=tenant_id)
    OR (scope IN ('user_experiment','room','shared_game','video_profile')
        AND btrim(scope_key)<>'')
  ),
  ADD CONSTRAINT configuration_profile_rollout_v25 CHECK (
    rollout_percentage>0::NUMERIC AND rollout_percentage<=100::NUMERIC
  );
ALTER TABLE configuration.bundle_revisions
  ADD CONSTRAINT configuration_bundle_scope_v25 CHECK (
    scope IN (
      'global','environment','tenant','user_experiment',
      'room','shared_game','video_profile'
    )
  ),
  ADD CONSTRAINT configuration_bundle_dimensions_v25 CHECK (
    (scope='global' AND scope_key='*' AND tenant_id IS NULL)
    OR (scope='environment' AND tenant_id IS NULL
        AND environment IS NOT NULL AND btrim(environment)<>''
        AND scope_key=environment)
    OR (scope='tenant' AND tenant_id IS NOT NULL
        AND btrim(tenant_id)<>'' AND scope_key=tenant_id)
    OR (scope IN ('user_experiment','room','shared_game','video_profile')
        AND btrim(scope_key)<>'')
  ),
  ADD CONSTRAINT configuration_bundle_rollout_v25 CHECK (
    rollout_percentage>0::NUMERIC AND rollout_percentage<=100::NUMERIC
  );

-- Remove only the V11 inline UNIQUE constraints.  Their generated names can
-- be truncated differently, so identify them by the ordered column vector.
DO $$
DECLARE old_constraint record;
BEGIN
  FOR old_constraint IN
    SELECT constraint_row.conname
    FROM pg_constraint AS constraint_row
    WHERE constraint_row.conrelid='configuration.profile_revisions'::regclass
      AND constraint_row.contype='u'
      AND ARRAY(
        SELECT attribute.attname::text
        FROM unnest(constraint_row.conkey)
          WITH ORDINALITY AS key_row(attnum,position)
        JOIN pg_attribute AS attribute
          ON attribute.attrelid=constraint_row.conrelid
         AND attribute.attnum=key_row.attnum
        ORDER BY key_row.position
      )=ARRAY['family','profile_id','version','scope','tenant_id','sequence']::text[]
  LOOP
    EXECUTE format(
      'ALTER TABLE configuration.profile_revisions DROP CONSTRAINT %I',
      old_constraint.conname
    );
  END LOOP;

  FOR old_constraint IN
    SELECT constraint_row.conname
    FROM pg_constraint AS constraint_row
    WHERE constraint_row.conrelid='configuration.bundle_revisions'::regclass
      AND constraint_row.contype='u'
      AND ARRAY(
        SELECT attribute.attname::text
        FROM unnest(constraint_row.conkey)
          WITH ORDINALITY AS key_row(attnum,position)
        JOIN pg_attribute AS attribute
          ON attribute.attrelid=constraint_row.conrelid
         AND attribute.attnum=key_row.attnum
        ORDER BY key_row.position
      )=ARRAY['bundle_id','version','scope','tenant_id','sequence']::text[]
  LOOP
    EXECUTE format(
      'ALTER TABLE configuration.bundle_revisions DROP CONSTRAINT %I',
      old_constraint.conname
    );
  END LOOP;
END $$;

-- Earlier schemas could admit concurrent published rows.  Preserve every row
-- and deterministically retain the newest exact logical identity as published.
WITH ranked AS (
  SELECT
    revision_id,
    row_number() OVER (
      PARTITION BY family,profile_id,scope,scope_key,tenant_id,environment
      ORDER BY sequence DESC,created_at DESC,revision_id DESC
    ) AS published_rank
  FROM configuration.profile_revisions
  WHERE status='published'
)
UPDATE configuration.profile_revisions AS revision
SET status='superseded'
FROM ranked
WHERE revision.revision_id=ranked.revision_id
  AND ranked.published_rank>1;

WITH ranked AS (
  SELECT
    revision_id,
    row_number() OVER (
      PARTITION BY bundle_id,scope,scope_key,tenant_id,environment
      ORDER BY sequence DESC,created_at DESC,revision_id DESC
    ) AS published_rank
  FROM configuration.bundle_revisions
  WHERE status='published'
)
UPDATE configuration.bundle_revisions AS revision
SET status='superseded'
FROM ranked
WHERE revision.revision_id=ranked.revision_id
  AND ranked.published_rank>1;

-- NULLS NOT DISTINCT gives PostgreSQL the same exact nullable identity used by
-- runtime IS NOT DISTINCT FROM predicates without conflating NULL and ''.
-- Corrected unique indexes retain the historical names as replay sentinels:
-- the migration runner replays 001..025, so the older IF NOT EXISTS statements
-- must see the corrected definitions rather than recreate their obsolete ones.
CREATE UNIQUE INDEX uq_configuration_profile_content
  ON configuration.profile_revisions(
    family,scope,scope_key,tenant_id,environment,payload_sha256
  ) NULLS NOT DISTINCT;
CREATE UNIQUE INDEX uq_configuration_profile_v20
  ON configuration.profile_revisions(
    family,profile_id,scope,scope_key,tenant_id,environment,sequence
  ) NULLS NOT DISTINCT;
CREATE INDEX ix_configuration_profile_lookup_v25
  ON configuration.profile_revisions(
    family,profile_id,scope,scope_key,tenant_id,environment,
    status,sequence DESC,revision_id
  );
CREATE UNIQUE INDEX uq_configuration_profile_published_v25
  ON configuration.profile_revisions(
    family,profile_id,scope,scope_key,tenant_id,environment
  ) NULLS NOT DISTINCT
  WHERE status='published';

CREATE UNIQUE INDEX uq_configuration_bundle_v20
  ON configuration.bundle_revisions(
    bundle_id,scope,scope_key,tenant_id,environment,sequence
  ) NULLS NOT DISTINCT;
-- V13 deliberately makes bundle-content lookup non-unique so a governed
-- rollback may retain a new revision with identical content.  Keeping the V11
-- name prevents its obsolete unique definition from being recreated before
-- V13 is reached during a populated full-history replay.
CREATE INDEX uq_configuration_bundle_content
  ON configuration.bundle_revisions(
    bundle_id,scope,scope_key,tenant_id,environment,
    content_sha256,sequence DESC,revision_id
  );
CREATE INDEX ix_configuration_bundle_lookup_v25
  ON configuration.bundle_revisions(
    bundle_id,scope,scope_key,tenant_id,environment,
    status,sequence DESC,revision_id
  );
CREATE UNIQUE INDEX uq_configuration_bundle_published_v25
  ON configuration.bundle_revisions(
    bundle_id,scope,scope_key,tenant_id,environment
  ) NULLS NOT DISTINCT
  WHERE status='published';

CREATE OR REPLACE FUNCTION configuration.guard_published_profile_payload()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF OLD.status IN ('published','superseded','retired') AND (
    NEW.revision_id IS DISTINCT FROM OLD.revision_id OR
    NEW.family IS DISTINCT FROM OLD.family OR
    NEW.profile_id IS DISTINCT FROM OLD.profile_id OR
    NEW.version IS DISTINCT FROM OLD.version OR
    NEW.scope IS DISTINCT FROM OLD.scope OR
    NEW.scope_key IS DISTINCT FROM OLD.scope_key OR
    NEW.tenant_id IS DISTINCT FROM OLD.tenant_id OR
    NEW.environment IS DISTINCT FROM OLD.environment OR
    NEW.payload IS DISTINCT FROM OLD.payload OR
    NEW.payload_sha256 IS DISTINCT FROM OLD.payload_sha256 OR
    NEW.sequence IS DISTINCT FROM OLD.sequence OR
    NEW.effective_from IS DISTINCT FROM OLD.effective_from OR
    NEW.effective_until IS DISTINCT FROM OLD.effective_until OR
    NEW.rollout_percentage IS DISTINCT FROM OLD.rollout_percentage OR
    NEW.rollout_seed IS DISTINCT FROM OLD.rollout_seed OR
    NEW.created_by IS DISTINCT FROM OLD.created_by OR
    NEW.created_at IS DISTINCT FROM OLD.created_at OR
    NEW.published_by IS DISTINCT FROM OLD.published_by OR
    NEW.published_at IS DISTINCT FROM OLD.published_at OR
    NEW.supersedes_revision_id IS DISTINCT FROM OLD.supersedes_revision_id
  ) THEN
    RAISE EXCEPTION 'published configuration profile revisions are immutable';
  END IF;
  RETURN NEW;
END $$;
DROP TRIGGER IF EXISTS trg_configuration_profile_immutable
  ON configuration.profile_revisions;
CREATE TRIGGER trg_configuration_profile_immutable
BEFORE UPDATE ON configuration.profile_revisions
FOR EACH ROW EXECUTE FUNCTION configuration.guard_published_profile_payload();

CREATE OR REPLACE FUNCTION configuration.guard_published_bundle_payload()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF OLD.status IN ('published','superseded','retired') AND (
    NEW.revision_id IS DISTINCT FROM OLD.revision_id OR
    NEW.bundle_id IS DISTINCT FROM OLD.bundle_id OR
    NEW.version IS DISTINCT FROM OLD.version OR
    NEW.mode IS DISTINCT FROM OLD.mode OR
    NEW.scope IS DISTINCT FROM OLD.scope OR
    NEW.scope_key IS DISTINCT FROM OLD.scope_key OR
    NEW.tenant_id IS DISTINCT FROM OLD.tenant_id OR
    NEW.environment IS DISTINCT FROM OLD.environment OR
    NEW.components IS DISTINCT FROM OLD.components OR
    NEW.content_sha256 IS DISTINCT FROM OLD.content_sha256 OR
    NEW.sequence IS DISTINCT FROM OLD.sequence OR
    NEW.effective_from IS DISTINCT FROM OLD.effective_from OR
    NEW.effective_until IS DISTINCT FROM OLD.effective_until OR
    NEW.rollout_percentage IS DISTINCT FROM OLD.rollout_percentage OR
    NEW.rollout_seed IS DISTINCT FROM OLD.rollout_seed OR
    NEW.created_by IS DISTINCT FROM OLD.created_by OR
    NEW.created_at IS DISTINCT FROM OLD.created_at OR
    NEW.published_by IS DISTINCT FROM OLD.published_by OR
    NEW.published_at IS DISTINCT FROM OLD.published_at OR
    NEW.supersedes_revision_id IS DISTINCT FROM OLD.supersedes_revision_id
  ) THEN
    RAISE EXCEPTION 'published configuration bundle revisions are immutable';
  END IF;
  RETURN NEW;
END $$;
DROP TRIGGER IF EXISTS trg_configuration_bundle_immutable
  ON configuration.bundle_revisions;
CREATE TRIGGER trg_configuration_bundle_immutable
BEFORE UPDATE ON configuration.bundle_revisions
FOR EACH ROW EXECUTE FUNCTION configuration.guard_published_bundle_payload();

-- Permissive PostgreSQL policies combine with OR.  Remove both generations of
-- legacy policies before installing one explicit V25 authority per operation.
DROP POLICY IF EXISTS configuration_profile_read
  ON configuration.profile_revisions;
DROP POLICY IF EXISTS configuration_profile_admin
  ON configuration.profile_revisions;
DROP POLICY IF EXISTS configuration_profile_read_v20
  ON configuration.profile_revisions;
DROP POLICY IF EXISTS configuration_profile_write_v20
  ON configuration.profile_revisions;
DROP POLICY IF EXISTS configuration_profile_context_read_v25
  ON configuration.profile_revisions;
DROP POLICY IF EXISTS configuration_profile_write_v25
  ON configuration.profile_revisions;
CREATE POLICY configuration_profile_context_read_v25
  ON configuration.profile_revisions FOR SELECT
  USING (
    scope='global'
    OR current_setting('app.role',true) IN ('admin','service')
    OR (
      current_setting('app.subject_id',true)<>''
      AND NULLIF(current_setting('app.tenant_id',true),'') IS NOT NULL
      AND tenant_id IS NOT DISTINCT FROM
        NULLIF(current_setting('app.tenant_id',true),'')
    )
  );
CREATE POLICY configuration_profile_write_v25
  ON configuration.profile_revisions FOR ALL
  USING (current_setting('app.role',true) IN ('admin','service'))
  WITH CHECK (current_setting('app.role',true) IN ('admin','service'));

DROP POLICY IF EXISTS configuration_bundle_read
  ON configuration.bundle_revisions;
DROP POLICY IF EXISTS configuration_bundle_admin
  ON configuration.bundle_revisions;
DROP POLICY IF EXISTS configuration_bundle_read_v20
  ON configuration.bundle_revisions;
DROP POLICY IF EXISTS configuration_bundle_write_v20
  ON configuration.bundle_revisions;
DROP POLICY IF EXISTS configuration_bundle_context_read_v25
  ON configuration.bundle_revisions;
DROP POLICY IF EXISTS configuration_bundle_write_v25
  ON configuration.bundle_revisions;
CREATE POLICY configuration_bundle_context_read_v25
  ON configuration.bundle_revisions FOR SELECT
  USING (
    scope='global'
    OR current_setting('app.role',true) IN ('admin','service')
    OR (
      current_setting('app.subject_id',true)<>''
      AND NULLIF(current_setting('app.tenant_id',true),'') IS NOT NULL
      AND tenant_id IS NOT DISTINCT FROM
        NULLIF(current_setting('app.tenant_id',true),'')
    )
  );
CREATE POLICY configuration_bundle_write_v25
  ON configuration.bundle_revisions FOR ALL
  USING (current_setting('app.role',true) IN ('admin','service'))
  WITH CHECK (current_setting('app.role',true) IN ('admin','service'));

DROP POLICY IF EXISTS configuration_events_read ON configuration.events;
DROP POLICY IF EXISTS configuration_events_admin ON configuration.events;
DROP POLICY IF EXISTS configuration_events_read_v25 ON configuration.events;
DROP POLICY IF EXISTS configuration_events_write_v25 ON configuration.events;
CREATE POLICY configuration_events_read_v25
  ON configuration.events FOR SELECT
  USING (
    current_setting('app.subject_id',true)<>''
    OR current_setting('app.role',true) IN ('admin','service')
  );
CREATE POLICY configuration_events_write_v25
  ON configuration.events FOR INSERT
  WITH CHECK (current_setting('app.role',true) IN ('admin','service'));

ALTER TABLE configuration.profile_revisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE configuration.profile_revisions FORCE ROW LEVEL SECURITY;
ALTER TABLE configuration.bundle_revisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE configuration.bundle_revisions FORCE ROW LEVEL SECURITY;
ALTER TABLE configuration.events ENABLE ROW LEVEL SECURITY;
ALTER TABLE configuration.events FORCE ROW LEVEL SECURITY;

INSERT INTO configuration.invalidation_epochs(namespace,epoch,updated_at)
VALUES('configuration',1,TIMESTAMPTZ '1980-01-01T00:00:00+00:00')
ON CONFLICT(namespace) DO NOTHING;

COMMIT;
