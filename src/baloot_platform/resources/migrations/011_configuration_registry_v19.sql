BEGIN;

CREATE SCHEMA IF NOT EXISTS configuration;

CREATE TABLE IF NOT EXISTS configuration.profile_revisions(
    revision_id TEXT PRIMARY KEY,
    family TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    version TEXT NOT NULL,
    scope TEXT NOT NULL CHECK(scope IN ('global','tenant','room','shared_game','video_profile')),
    tenant_id TEXT,
    status TEXT NOT NULL CHECK(status IN ('draft','published','rejected','retired','superseded')),
    payload JSONB NOT NULL,
    payload_sha256 CHAR(64) NOT NULL,
    sequence BIGINT NOT NULL,
    created_by TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    published_by TEXT,
    published_at TIMESTAMPTZ,
    supersedes_revision_id TEXT REFERENCES configuration.profile_revisions(revision_id),
    etag BIGINT NOT NULL DEFAULT 1,
    UNIQUE(family,profile_id,version,scope,tenant_id,sequence)
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_configuration_profile_content
    ON configuration.profile_revisions(family,scope,COALESCE(tenant_id,''),payload_sha256);
CREATE INDEX IF NOT EXISTS ix_configuration_profile_lookup
    ON configuration.profile_revisions(family,profile_id,scope,tenant_id,status,sequence DESC);

CREATE TABLE IF NOT EXISTS configuration.bundle_revisions(
    revision_id TEXT PRIMARY KEY,
    bundle_id TEXT NOT NULL,
    version TEXT NOT NULL,
    mode TEXT NOT NULL,
    scope TEXT NOT NULL CHECK(scope IN ('global','tenant','room','shared_game','video_profile')),
    tenant_id TEXT,
    status TEXT NOT NULL CHECK(status IN ('draft','published','rejected','retired','superseded')),
    components JSONB NOT NULL,
    content_sha256 CHAR(64) NOT NULL,
    sequence BIGINT NOT NULL,
    created_by TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    published_by TEXT,
    published_at TIMESTAMPTZ,
    supersedes_revision_id TEXT REFERENCES configuration.bundle_revisions(revision_id),
    etag BIGINT NOT NULL DEFAULT 1,
    UNIQUE(bundle_id,version,scope,tenant_id,sequence)
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_configuration_bundle_content
    ON configuration.bundle_revisions(scope,COALESCE(tenant_id,''),content_sha256);
CREATE INDEX IF NOT EXISTS ix_configuration_bundle_lookup
    ON configuration.bundle_revisions(bundle_id,scope,tenant_id,status,sequence DESC);

CREATE TABLE IF NOT EXISTS configuration.events(
    event_id TEXT PRIMARY KEY,
    aggregate_type TEXT NOT NULL CHECK(aggregate_type IN ('profile','bundle')),
    aggregate_id TEXT NOT NULL,
    sequence BIGINT NOT NULL,
    event_type TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    payload JSONB NOT NULL,
    previous_hash TEXT NOT NULL,
    event_hash CHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(aggregate_type,aggregate_id,sequence)
);
CREATE INDEX IF NOT EXISTS ix_configuration_events_aggregate
    ON configuration.events(aggregate_type,aggregate_id,sequence);

CREATE OR REPLACE FUNCTION configuration.guard_published_profile_payload() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  IF OLD.status IN ('published','superseded','retired') AND
     (NEW.payload IS DISTINCT FROM OLD.payload OR NEW.payload_sha256 IS DISTINCT FROM OLD.payload_sha256 OR
      NEW.family IS DISTINCT FROM OLD.family OR NEW.profile_id IS DISTINCT FROM OLD.profile_id OR
      NEW.version IS DISTINCT FROM OLD.version OR NEW.scope IS DISTINCT FROM OLD.scope OR
      NEW.tenant_id IS DISTINCT FROM OLD.tenant_id) THEN
    RAISE EXCEPTION 'published configuration profile revisions are immutable';
  END IF;
  RETURN NEW;
END $$;
DROP TRIGGER IF EXISTS trg_configuration_profile_immutable ON configuration.profile_revisions;
CREATE TRIGGER trg_configuration_profile_immutable
BEFORE UPDATE ON configuration.profile_revisions
FOR EACH ROW EXECUTE FUNCTION configuration.guard_published_profile_payload();

CREATE OR REPLACE FUNCTION configuration.guard_published_bundle_payload() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  IF OLD.status IN ('published','superseded','retired') AND
     (NEW.components IS DISTINCT FROM OLD.components OR NEW.content_sha256 IS DISTINCT FROM OLD.content_sha256 OR
      NEW.bundle_id IS DISTINCT FROM OLD.bundle_id OR NEW.version IS DISTINCT FROM OLD.version OR
      NEW.mode IS DISTINCT FROM OLD.mode OR NEW.scope IS DISTINCT FROM OLD.scope OR
      NEW.tenant_id IS DISTINCT FROM OLD.tenant_id) THEN
    RAISE EXCEPTION 'published configuration bundle revisions are immutable';
  END IF;
  RETURN NEW;
END $$;
DROP TRIGGER IF EXISTS trg_configuration_bundle_immutable ON configuration.bundle_revisions;
CREATE TRIGGER trg_configuration_bundle_immutable
BEFORE UPDATE ON configuration.bundle_revisions
FOR EACH ROW EXECUTE FUNCTION configuration.guard_published_bundle_payload();

ALTER TABLE configuration.profile_revisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE configuration.bundle_revisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE configuration.events ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS configuration_profile_read ON configuration.profile_revisions;
CREATE POLICY configuration_profile_read ON configuration.profile_revisions FOR SELECT
USING (
  scope='global' OR tenant_id IS NULL OR tenant_id=current_setting('app.tenant_id', true) OR
  current_setting('app.role', true) IN ('admin','service')
);
DROP POLICY IF EXISTS configuration_profile_admin ON configuration.profile_revisions;
CREATE POLICY configuration_profile_admin ON configuration.profile_revisions FOR ALL
USING (current_setting('app.role', true) IN ('admin','service'))
WITH CHECK (current_setting('app.role', true) IN ('admin','service'));

DROP POLICY IF EXISTS configuration_bundle_read ON configuration.bundle_revisions;
CREATE POLICY configuration_bundle_read ON configuration.bundle_revisions FOR SELECT
USING (
  scope='global' OR tenant_id IS NULL OR tenant_id=current_setting('app.tenant_id', true) OR
  current_setting('app.role', true) IN ('admin','service')
);
DROP POLICY IF EXISTS configuration_bundle_admin ON configuration.bundle_revisions;
CREATE POLICY configuration_bundle_admin ON configuration.bundle_revisions FOR ALL
USING (current_setting('app.role', true) IN ('admin','service'))
WITH CHECK (current_setting('app.role', true) IN ('admin','service'));

DROP POLICY IF EXISTS configuration_events_read ON configuration.events;
CREATE POLICY configuration_events_read ON configuration.events FOR SELECT
USING (current_setting('app.subject_id', true) <> '' OR current_setting('app.role', true) IN ('admin','service'));
DROP POLICY IF EXISTS configuration_events_admin ON configuration.events;
CREATE POLICY configuration_events_admin ON configuration.events FOR INSERT
WITH CHECK (current_setting('app.role', true) IN ('admin','service'));

COMMIT;
