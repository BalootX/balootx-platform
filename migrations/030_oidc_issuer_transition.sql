BEGIN;

CREATE TABLE IF NOT EXISTS security.oidc_issuer_transitions (
  revision_id text PRIMARY KEY REFERENCES configuration.profile_revisions(revision_id),
  profile_id text NOT NULL,
  provider_id text NOT NULL,
  legacy_issuer text NOT NULL,
  issuer text NOT NULL,
  applied_at timestamptz NOT NULL DEFAULT now(),
  rolled_back_at timestamptz,
  finalized_at timestamptz
);
CREATE UNIQUE INDEX IF NOT EXISTS oidc_active_provider_transition
  ON security.oidc_issuer_transitions (provider_id) WHERE rolled_back_at IS NULL;
CREATE UNIQUE INDEX IF NOT EXISTS oidc_active_profile_transition
  ON security.oidc_issuer_transitions (profile_id) WHERE rolled_back_at IS NULL;

ALTER TABLE security.external_identities
  ADD COLUMN IF NOT EXISTS issuer_alias boolean NOT NULL DEFAULT false;
CREATE UNIQUE INDEX IF NOT EXISTS external_identity_owner_and_issuer_kind
  ON security.external_identities (local_subject_id, provider_id, issuer_alias);
ALTER TABLE security.external_identities
  DROP CONSTRAINT IF EXISTS external_identities_local_subject_id_provider_id_key;

DROP POLICY IF EXISTS external_identity_service_v30 ON security.external_identities;
CREATE POLICY external_identity_service_v30 ON security.external_identities FOR ALL
  USING (current_setting('app.role', true) IN ('oidc_link', 'service'))
  WITH CHECK (current_setting('app.role', true) IN ('oidc_link', 'service'));

CREATE OR REPLACE FUNCTION security.mirror_oidc_issuer_link() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog AS $$
DECLARE
  transition security.oidc_issuer_transitions%ROWTYPE;
  peer_issuer text;
BEGIN
  IF pg_trigger_depth() > 1 THEN
    RETURN NEW;
  END IF;
  NEW.issuer_alias := false;
  SELECT * INTO transition FROM security.oidc_issuer_transitions
    WHERE provider_id = NEW.provider_id AND rolled_back_at IS NULL AND finalized_at IS NULL
      AND legacy_issuer <> issuer AND NEW.issuer IN (legacy_issuer, issuer);
  IF NOT FOUND THEN
    RETURN NEW;
  END IF;

  -- Old and new APIs may write different keys for the same link during a rollout.
  PERFORM pg_advisory_xact_lock(hashtextextended(NEW.provider_id, 30));
  NEW.issuer_alias := NEW.issuer = transition.legacy_issuer;
  peer_issuer := CASE WHEN NEW.issuer_alias THEN transition.issuer ELSE transition.legacy_issuer END;
  INSERT INTO security.external_identities AS existing
    (provider_id, issuer, external_subject, local_subject_id, claims_json, linked_at, issuer_alias)
  VALUES
    (NEW.provider_id, peer_issuer, NEW.external_subject, NEW.local_subject_id,
     NEW.claims_json, NEW.linked_at, NOT NEW.issuer_alias)
  ON CONFLICT (provider_id, issuer, external_subject) DO UPDATE
    SET claims_json = EXCLUDED.claims_json
    WHERE existing.local_subject_id = EXCLUDED.local_subject_id;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'OIDC external identity is already linked to another account';
  END IF;
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS mirror_oidc_issuer_link ON security.external_identities;
CREATE TRIGGER mirror_oidc_issuer_link BEFORE INSERT ON security.external_identities
  FOR EACH ROW EXECUTE FUNCTION security.mirror_oidc_issuer_link();

COMMIT;
