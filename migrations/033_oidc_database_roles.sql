BEGIN;

REVOKE EXECUTE ON FUNCTION security.mirror_oidc_issuer_link() FROM PUBLIC;

DROP POLICY IF EXISTS external_identity_authority_v32 ON security.external_identities;
DROP POLICY IF EXISTS external_identity_authority_v33 ON security.external_identities;
CREATE POLICY external_identity_authority_v33 ON security.external_identities FOR ALL
  USING (
    (current_setting('app.role', true), current_setting('app.oidc_identity_authority', true),
     has_table_privilege(session_user, 'security.oidc_issuer_transitions', 'UPDATE'))
    IN (('oidc_link', 'callback', false), ('service', 'migration', true))
  )
  WITH CHECK (
    (current_setting('app.role', true), current_setting('app.oidc_identity_authority', true),
     has_table_privilege(session_user, 'security.oidc_issuer_transitions', 'UPDATE'))
    IN (('oidc_link', 'callback', false), ('service', 'migration', true))
  );

COMMIT;
