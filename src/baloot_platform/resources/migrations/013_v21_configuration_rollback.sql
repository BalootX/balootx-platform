BEGIN;

DROP INDEX IF EXISTS configuration.uq_configuration_bundle_content;
DROP INDEX IF EXISTS configuration.uq_configuration_bundle_content_v20;

CREATE INDEX IF NOT EXISTS ix_configuration_bundle_content_v21
    ON configuration.bundle_revisions(
        scope,
        scope_key,
        COALESCE(environment, ''),
        content_sha256,
        sequence DESC
    );

COMMIT;
