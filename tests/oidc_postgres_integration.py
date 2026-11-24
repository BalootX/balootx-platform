from __future__ import annotations

import os
from copy import deepcopy

import pytest

from scripts import migrate_postgres



def test_oidc_issuer_transition_on_postgres(root, monkeypatch, caplog):
    import httpx
    import psycopg
    from psycopg.conninfo import make_conninfo
    from psycopg.rows import dict_row

    from baloot_platform.config import build_bundle
    from baloot_platform.configuration import PostgresConfigurationStore, VersionedConfigurationService
    from baloot_platform.security.state_v20 import PostgresSecurityStateStore
    from baloot_platform.security.oidc_migration import (
        finalize_legacy_oidc,
        legacy_oidc_profile,
        migrate_legacy_oidc,
        rollback_legacy_oidc,
    )
    from baloot_platform.storage.request_context import (
        CURRENT_DATABASE_IDENTITY,
        DatabaseIdentityContext,
        apply_postgres_identity_context,
    )

    dsn = os.environ.get("BALOOT_TEST_OIDC_POSTGRES_DSN")
    assert dsn, "hosted PostgreSQL proof requires BALOOT_TEST_OIDC_POSTGRES_DSN"
    monkeypatch.setenv("BALOOT_DATABASE_URL", dsn)
    monkeypatch.setenv("BALOOT_ROOT", str(root))
    migrate_postgres.main()
    with psycopg.connect(dsn) as conn:
        conn.execute("CREATE ROLE oidc_ci_app NOLOGIN NOSUPERUSER NOBYPASSRLS")
        conn.execute("GRANT USAGE ON SCHEMA security TO oidc_ci_app")
        conn.execute("GRANT SELECT, INSERT, UPDATE ON security.external_identities TO oidc_ci_app")
    app_dsn = make_conninfo(dsn, options="-c role=oidc_ci_app")
    legacy = "https://signin.example/"
    canonical = "https://signin.example"
    for field, value in {
        "ISSUER": legacy,
        "AUTHORIZATION_ENDPOINT": canonical + "/authorize",
        "TOKEN_ENDPOINT": canonical + "/token",
        "CLIENT_ID": "fixture-client",
        "REDIRECT_URI": "https://app.example/callback",
        "CLIENT_SECRET": "synthetic-client-fixture",
    }.items():
        monkeypatch.setenv(f"BALOOT_OIDC_{field}", value)
    monkeypatch.delenv("BALOOT_OIDC_PROVIDER_ID", raising=False)
    monkeypatch.setenv("BALOOT_OIDC_JWKS_URI", canonical + "/legacy-keys")

    def discovery(request):
        assert str(request.url) == canonical + "/.well-known/openid-configuration"
        return httpx.Response(200, json={"issuer": canonical, "jwks_uri": canonical + "/keys"})

    with httpx.Client(transport=httpx.MockTransport(discovery)) as client:
        source, payload = legacy_oidc_profile(
            client, profile_id="signin", version="1.0.0",
            reason="Migrate fixture links", algorithms=["RS256"],
        )
    assert source == legacy
    assert payload["providers"]["default"]["issuer"] == canonical
    assert payload["providers"]["default"]["client_secret_env"] == "BALOOT_OIDC_CLIENT_SECRET"
    assert "synthetic-client-fixture" not in str(payload)
    assert sum("legacy JWKS override is ignored" in record.message for record in caplog.records) == 1

    def bind(conn):
        apply_postgres_identity_context(conn)
        return VersionedConfigurationService(
            PostgresConfigurationStore(dsn, transaction_connection=conn),
            root=root, profile=build_bundle(root).component("configuration"),
        )

    def link(conn, issuer, subject):
        conn.execute(
            "INSERT INTO security.external_identities "
            "(provider_id, issuer, external_subject, local_subject_id, claims_json, linked_at) "
            "VALUES ('default', %s, %s, %s, '{}'::jsonb, '2026-01-01T00:00:00Z') "
            "ON CONFLICT (provider_id, issuer, external_subject) DO UPDATE "
            "SET local_subject_id=excluded.local_subject_id, claims_json=excluded.claims_json",
            (issuer, subject, f"local-{subject}"),
        )

    def lookup(conn, issuer, subject):
        return conn.execute(
            "SELECT local_subject_id, issuer_alias FROM security.external_identities "
            "WHERE provider_id='default' AND issuer=%s AND external_subject=%s",
            (issuer, subject),
        ).fetchone()

    token = CURRENT_DATABASE_IDENTITY.set(DatabaseIdentityContext(subject_id="operator-1", role="service"))
    try:
        with psycopg.connect(dsn, row_factory=dict_row) as conn:
            configuration = bind(conn)
            with pytest.raises(ValueError, match="migration was not found"):
                rollback_legacy_oidc(
                    conn, configuration, profile_id="signin", revision_id="CPR-UNKNOWN",
                    actor="operator-1", reason="Refuse an unrelated migration",
                )
            with pytest.raises(ValueError, match="must preserve the legacy issuer"):
                migrate_legacy_oidc(
                    conn, configuration, payload=payload, legacy_issuer="https://other.example",
                    actor="operator-1",
                )
            already_published = deepcopy(payload)
            already_published["profile"]["id"] = "already-published"
            configuration.create_profile(
                family="oidc_providers", payload=already_published, created_by="operator-1", publish=True,
            )
            with pytest.raises(ValueError, match="already has a published revision"):
                migrate_legacy_oidc(
                    conn, configuration, payload=already_published, legacy_issuer=legacy, actor="operator-1",
                )
            link(conn, canonical, "collision")
            with pytest.raises(ValueError, match="discovered issuer already has identity links"):
                migrate_legacy_oidc(
                    conn, configuration, payload=payload, legacy_issuer=legacy, actor="operator-1",
                )
            conn.execute("DELETE FROM security.external_identities WHERE external_subject='collision'")
            link(conn, legacy, "existing")
            migration = migrate_legacy_oidc(
                conn, configuration, payload=payload, legacy_issuer=legacy, actor="operator-1"
            )
            assert migration["links"] == 1
        revision_id = migration["revisionId"]
        with psycopg.connect(dsn, row_factory=dict_row) as conn:
            configuration = bind(conn)
            different_reason = {**payload, "change_reason": "Repeat the existing command"}
            assert migrate_legacy_oidc(
                conn, configuration, payload=different_reason, legacy_issuer=legacy, actor="operator-1"
            )["status"] == "already-applied"
            assert configuration.store.get_profile(revision_id).payload["change_reason"] == payload["change_reason"]
            changed = deepcopy(payload)
            changed["providers"]["default"]["client_id"] = "different-fixture-client"
            with pytest.raises(ValueError, match="different binding"):
                migrate_legacy_oidc(
                    conn, configuration, payload=changed, legacy_issuer=legacy, actor="operator-1",
                )
            assert lookup(conn, legacy, "existing") == {
                "local_subject_id": "local-existing", "issuer_alias": True,
            }
            assert lookup(conn, canonical, "existing") == {
                "local_subject_id": "local-existing", "issuer_alias": False,
            }
            link(conn, legacy, "old-callback")
            assert lookup(conn, canonical, "old-callback")["local_subject_id"] == "local-old-callback"
        store = PostgresSecurityStateStore(app_dsn)
        bypassing_store = PostgresSecurityStateStore(dsn)
        try:
            store.require_oidc_rls()
            with pytest.raises(RuntimeError, match="subject to row-level security"):
                bypassing_store.require_oidc_rls()
            with psycopg.connect(app_dsn) as conn:
                assert conn.execute("SELECT count(*) FROM security.external_identities").fetchone() == (0,)
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                with psycopg.connect(app_dsn) as conn:
                    link(conn, canonical, "unverified")
            assert store.get_external_identity("default", canonical, "old-callback")["local_subject_id"] == "local-old-callback"
            values = {
                "provider_id": "default", "issuer": canonical, "external_subject": "new-callback",
                "local_subject_id": "local-new-callback", "claims": {},
                "linked_at": "2026-01-01T00:00:00Z",
            }
            store.link_external_identity(**values)
            with pytest.raises(psycopg.errors.RaiseException, match="already linked"):
                store.link_external_identity(**{**values, "local_subject_id": "wrong-account"})
            assert store.get_external_identity("default", legacy, "new-callback")["local_subject_id"] == "local-new-callback"
        finally:
            store.close()
            bypassing_store.close()

        with psycopg.connect(dsn, row_factory=dict_row) as conn:
            assert lookup(conn, legacy, "new-callback")["local_subject_id"] == "local-new-callback"
            newer = deepcopy(payload)
            newer["profile"]["version"] = "1.0.1"
            newer_revision = bind(conn).create_profile(
                family="oidc_providers", payload=newer, created_by="operator-1", publish=True,
            )
        with psycopg.connect(dsn, row_factory=dict_row) as conn:
            configuration = bind(conn)
            assert rollback_legacy_oidc(
                conn, configuration, profile_id="signin", revision_id=revision_id,
                actor="operator-1", reason="Return to the previous release",
            )["links"] == 3
            assert rollback_legacy_oidc(
                conn, configuration, profile_id="signin", revision_id=revision_id,
                actor="operator-1", reason="Return to the previous release",
            )["status"] == "already-rolled-back"
            assert configuration.store.get_profile(newer_revision["revisionId"]).status == "retired"
            with pytest.raises(ValueError, match="was rolled back"):
                finalize_legacy_oidc(
                    conn, configuration, profile_id="signin", revision_id=revision_id,
                    actor="operator-1", reason="Close rollback window",
                )
            for subject in ("existing", "old-callback", "new-callback"):
                assert lookup(conn, canonical, subject) is None
                assert lookup(conn, legacy, subject) == {
                    "local_subject_id": f"local-{subject}", "issuer_alias": False,
                }
            audit = configuration.store.events_with_verification("profile", revision_id)
            assert audit["verification"]["valid"]
            assert audit["items"][-1]["eventType"] == "oidc_issuer_rolled_back"
        payload["profile"]["version"] = "1.0.2"
        with psycopg.connect(dsn, row_factory=dict_row) as conn:
            configuration = bind(conn)
            repeated = migrate_legacy_oidc(
                conn, configuration, payload=payload, legacy_issuer=legacy, actor="operator-1"
            )
            revision_id = repeated["revisionId"]
            for expected in ("finalized", "already-finalized"):
                assert finalize_legacy_oidc(
                    conn, configuration, profile_id="signin", revision_id=revision_id,
                    actor="operator-1", reason="Close rollback window",
                )["status"] == expected
            with pytest.raises(ValueError, match="rollback window has closed"):
                rollback_legacy_oidc(
                    conn, configuration, profile_id="signin", revision_id=revision_id,
                    actor="operator-1", reason="Repeat rollback",
                )
            assert lookup(conn, legacy, "existing") is None
            assert lookup(conn, canonical, "existing")["local_subject_id"] == "local-existing"
    finally:
        CURRENT_DATABASE_IDENTITY.reset(token)
