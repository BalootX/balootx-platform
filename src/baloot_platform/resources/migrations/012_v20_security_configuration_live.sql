BEGIN;

CREATE SCHEMA IF NOT EXISTS security;

ALTER TABLE configuration.profile_revisions
  ADD COLUMN IF NOT EXISTS scope_key text NOT NULL DEFAULT '*',
  ADD COLUMN IF NOT EXISTS environment text,
  ADD COLUMN IF NOT EXISTS effective_from timestamptz,
  ADD COLUMN IF NOT EXISTS effective_until timestamptz,
  ADD COLUMN IF NOT EXISTS rollout_percentage integer NOT NULL DEFAULT 100,
  ADD COLUMN IF NOT EXISTS rollout_seed text NOT NULL DEFAULT '';
ALTER TABLE configuration.bundle_revisions
  ADD COLUMN IF NOT EXISTS scope_key text NOT NULL DEFAULT '*',
  ADD COLUMN IF NOT EXISTS environment text,
  ADD COLUMN IF NOT EXISTS effective_from timestamptz,
  ADD COLUMN IF NOT EXISTS effective_until timestamptz,
  ADD COLUMN IF NOT EXISTS rollout_percentage integer NOT NULL DEFAULT 100,
  ADD COLUMN IF NOT EXISTS rollout_seed text NOT NULL DEFAULT '';

CREATE TABLE IF NOT EXISTS configuration.aggregate_heads(
  aggregate_type text NOT NULL,
  aggregate_id text NOT NULL,
  next_sequence bigint NOT NULL DEFAULT 1,
  etag bigint NOT NULL DEFAULT 1,
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY(aggregate_type,aggregate_id)
);
CREATE TABLE IF NOT EXISTS configuration.invalidation_epochs(
  namespace text PRIMARY KEY,
  epoch bigint NOT NULL DEFAULT 1,
  updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_configuration_profile_v20
 ON configuration.profile_revisions(family,profile_id,version,scope,scope_key,COALESCE(tenant_id,''),sequence);
CREATE UNIQUE INDEX IF NOT EXISTS uq_configuration_bundle_v20
 ON configuration.bundle_revisions(bundle_id,version,scope,scope_key,COALESCE(tenant_id,''),sequence);

ALTER TABLE live.rooms ADD COLUMN IF NOT EXISTS version bigint NOT NULL DEFAULT 1;
CREATE TABLE IF NOT EXISTS live.room_events(
  room_id text NOT NULL REFERENCES live.rooms(room_id) ON DELETE CASCADE,
  sequence bigint NOT NULL,
  event_type text NOT NULL,
  payload_json jsonb NOT NULL,
  event_hash char(64) NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY(room_id,sequence)
);
CREATE TABLE IF NOT EXISTS live.outbox(
  outbox_id bigserial PRIMARY KEY,
  room_id text NOT NULL,
  sequence bigint NOT NULL,
  payload_json jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  published_at timestamptz,
  UNIQUE(room_id,sequence)
);

CREATE TABLE IF NOT EXISTS security.oidc_transactions(
  state text PRIMARY KEY, payload_json jsonb NOT NULL, expires_at timestamptz NOT NULL,
  consumed_at timestamptz
);
CREATE TABLE IF NOT EXISTS security.websocket_tickets(
  nonce_hash char(64) PRIMARY KEY, subject_id text NOT NULL, room_id text NOT NULL,
  audience text NOT NULL, expires_at timestamptz NOT NULL, consumed_at timestamptz
);
CREATE TABLE IF NOT EXISTS security.passkey_challenges(
  challenge_hash char(64) PRIMARY KEY, subject_id text NOT NULL, purpose text NOT NULL,
  payload_json jsonb NOT NULL, expires_at timestamptz NOT NULL, consumed_at timestamptz
);
CREATE TABLE IF NOT EXISTS security.passkey_credentials(
  credential_id text PRIMARY KEY, subject_id text NOT NULL, public_key_cose bytea NOT NULL,
  sign_count bigint NOT NULL DEFAULT 0, transports_json jsonb NOT NULL DEFAULT '[]'::jsonb,
  created_at timestamptz NOT NULL, last_used_at timestamptz, label text NOT NULL
);
CREATE TABLE IF NOT EXISTS security.external_identities(
  provider_id text NOT NULL, issuer text NOT NULL, external_subject text NOT NULL,
  local_subject_id text NOT NULL, claims_json jsonb NOT NULL, linked_at timestamptz NOT NULL,
  PRIMARY KEY(provider_id,issuer,external_subject), UNIQUE(local_subject_id,provider_id)
);

ALTER TABLE configuration.profile_revisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE configuration.bundle_revisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE configuration.events ENABLE ROW LEVEL SECURITY;
ALTER TABLE configuration.profile_revisions FORCE ROW LEVEL SECURITY;
ALTER TABLE configuration.bundle_revisions FORCE ROW LEVEL SECURITY;
ALTER TABLE configuration.events FORCE ROW LEVEL SECURITY;
ALTER TABLE live.rooms ENABLE ROW LEVEL SECURITY;
ALTER TABLE live.room_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE live.outbox ENABLE ROW LEVEL SECURITY;
ALTER TABLE live.rooms FORCE ROW LEVEL SECURITY;
ALTER TABLE live.room_events FORCE ROW LEVEL SECURITY;
ALTER TABLE live.outbox FORCE ROW LEVEL SECURITY;
ALTER TABLE security.websocket_tickets ENABLE ROW LEVEL SECURITY;
ALTER TABLE security.passkey_credentials ENABLE ROW LEVEL SECURITY;
ALTER TABLE security.external_identities ENABLE ROW LEVEL SECURITY;
ALTER TABLE security.websocket_tickets FORCE ROW LEVEL SECURITY;
ALTER TABLE security.passkey_credentials FORCE ROW LEVEL SECURITY;
ALTER TABLE security.external_identities FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS configuration_profile_read_v20 ON configuration.profile_revisions;
CREATE POLICY configuration_profile_read_v20 ON configuration.profile_revisions FOR SELECT USING (
 scope='global' OR current_setting('app.role',true) IN ('admin','service') OR
 (tenant_id=current_setting('app.tenant_id',true) AND current_setting('app.subject_id',true)<>'')
);
DROP POLICY IF EXISTS configuration_profile_write_v20 ON configuration.profile_revisions;
CREATE POLICY configuration_profile_write_v20 ON configuration.profile_revisions FOR ALL
 USING (current_setting('app.role',true) IN ('admin','service'))
 WITH CHECK (current_setting('app.role',true) IN ('admin','service'));
DROP POLICY IF EXISTS configuration_bundle_read_v20 ON configuration.bundle_revisions;
CREATE POLICY configuration_bundle_read_v20 ON configuration.bundle_revisions FOR SELECT USING (
 scope='global' OR current_setting('app.role',true) IN ('admin','service') OR
 (tenant_id=current_setting('app.tenant_id',true) AND current_setting('app.subject_id',true)<>'')
);
DROP POLICY IF EXISTS configuration_bundle_write_v20 ON configuration.bundle_revisions;
CREATE POLICY configuration_bundle_write_v20 ON configuration.bundle_revisions FOR ALL
 USING (current_setting('app.role',true) IN ('admin','service'))
 WITH CHECK (current_setting('app.role',true) IN ('admin','service'));

DROP POLICY IF EXISTS live_room_read_v20 ON live.rooms;
CREATE POLICY live_room_read_v20 ON live.rooms FOR SELECT USING (
 current_setting('app.subject_id',true)<>'' OR current_setting('app.role',true) IN ('admin','service')
);
DROP POLICY IF EXISTS live_room_write_v20 ON live.rooms;
CREATE POLICY live_room_write_v20 ON live.rooms FOR ALL USING (
 current_setting('app.subject_id',true)<>'' OR current_setting('app.role',true) IN ('admin','service')
) WITH CHECK (current_setting('app.subject_id',true)<>'' OR current_setting('app.role',true) IN ('admin','service'));
DROP POLICY IF EXISTS live_event_read_v20 ON live.room_events;
CREATE POLICY live_event_read_v20 ON live.room_events FOR SELECT USING (current_setting('app.subject_id',true)<>'');
DROP POLICY IF EXISTS live_event_write_v20 ON live.room_events;
CREATE POLICY live_event_write_v20 ON live.room_events FOR INSERT WITH CHECK (current_setting('app.subject_id',true)<>'' OR current_setting('app.role',true) IN ('admin','service'));
DROP POLICY IF EXISTS live_outbox_service_v20 ON live.outbox;
CREATE POLICY live_outbox_service_v20 ON live.outbox FOR ALL USING (current_setting('app.role',true) IN ('admin','service')) WITH CHECK (current_setting('app.role',true) IN ('admin','service'));

DROP POLICY IF EXISTS passkey_owner_v20 ON security.passkey_credentials;
CREATE POLICY passkey_owner_v20 ON security.passkey_credentials FOR ALL
 USING (subject_id=current_setting('app.subject_id',true) OR current_setting('app.role',true) IN ('admin','service'))
 WITH CHECK (subject_id=current_setting('app.subject_id',true) OR current_setting('app.role',true) IN ('admin','service'));
DROP POLICY IF EXISTS external_identity_owner_v20 ON security.external_identities;
CREATE POLICY external_identity_owner_v20 ON security.external_identities FOR SELECT
 USING (local_subject_id=current_setting('app.subject_id',true) OR current_setting('app.role',true) IN ('admin','service'));
DROP POLICY IF EXISTS websocket_ticket_service_v20 ON security.websocket_tickets;
CREATE POLICY websocket_ticket_service_v20 ON security.websocket_tickets FOR ALL
 USING (current_setting('app.role',true) IN ('admin','service'))
 WITH CHECK (current_setting('app.role',true) IN ('admin','service'));

COMMIT;
