BEGIN;

CREATE OR REPLACE FUNCTION security.oidc_identity_authorized() RETURNS boolean
LANGUAGE sql STABLE SECURITY INVOKER SET search_path = pg_catalog AS $$
  SELECT (
    current_setting('app.role', true),
    current_setting('app.oidc_identity_authority', true),
    has_table_privilege(session_user, 'security.oidc_issuer_transitions', 'UPDATE')
  ) IN (('oidc_link', 'callback', false), ('service', 'migration', true))
$$;

ALTER POLICY external_identity_authority_v33 ON security.external_identities
  USING (security.oidc_identity_authorized())
  WITH CHECK (security.oidc_identity_authorized());

-- Older migration runners can replay the permissive owner policy.
DROP POLICY IF EXISTS external_identity_read_scope_v35 ON security.external_identities;
CREATE POLICY external_identity_read_scope_v35 ON security.external_identities AS RESTRICTIVE FOR SELECT
  USING (
    local_subject_id = NULLIF(current_setting('app.subject_id', true), '')
    OR security.oidc_identity_authorized()
  );

COMMIT;
