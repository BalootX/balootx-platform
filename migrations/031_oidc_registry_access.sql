BEGIN;

DROP POLICY IF EXISTS oidc_registry_access_v31 ON configuration.profile_revisions;
CREATE POLICY oidc_registry_access_v31 ON configuration.profile_revisions AS RESTRICTIVE
  USING (
    family <> 'oidc_providers'
    OR current_setting('app.configuration_admin', true) = 'on'
    OR (status = 'published' AND current_setting('app.oidc_registry_read', true) = 'on')
  )
  WITH CHECK (
    family <> 'oidc_providers'
    OR current_setting('app.configuration_admin', true) = 'on'
  );

DROP POLICY IF EXISTS oidc_registry_audit_v31 ON configuration.events;
CREATE POLICY oidc_registry_audit_v31 ON configuration.events AS RESTRICTIVE FOR SELECT
  USING (
    aggregate_type <> 'profile'
    OR current_setting('app.configuration_admin', true) = 'on'
    OR EXISTS (
      SELECT 1 FROM configuration.profile_revisions AS profile
      WHERE profile.revision_id = events.aggregate_id AND profile.family <> 'oidc_providers'
    )
  );

COMMIT;
