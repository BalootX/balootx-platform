BEGIN;

DROP POLICY IF EXISTS external_identity_service_v30 ON security.external_identities;
DROP POLICY IF EXISTS external_identity_authority_v32 ON security.external_identities;
CREATE POLICY external_identity_authority_v32 ON security.external_identities FOR ALL
  USING (
    (current_setting('app.role', true), current_setting('app.oidc_identity_authority', true))
    IN (('oidc_link', 'callback'), ('service', 'migration'))
  )
  WITH CHECK (
    (current_setting('app.role', true), current_setting('app.oidc_identity_authority', true))
    IN (('oidc_link', 'callback'), ('service', 'migration'))
  );

COMMIT;
