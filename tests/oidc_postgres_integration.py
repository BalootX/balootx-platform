from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from datetime import datetime
from io import BytesIO
from types import ModuleType, SimpleNamespace
from urllib.parse import parse_qs, urlsplit
from urllib.request import OpenerDirector

import pytest

from scripts import migrate_postgres


def _previous_module(root, monkeypatch, ref, module_name):
    path = f"src/{module_name.replace('.', '/')}.py"
    source = subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=root)
    module = ModuleType(f"{module_name}_previous")
    module.__package__ = module_name.rpartition(".")[0]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(source, path, "exec"), module.__dict__)
    return module


def test_oidc_issuer_transition_on_postgres(root, monkeypatch, caplog):
    import httpx
    import jwt
    import psycopg
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from psycopg.conninfo import make_conninfo
    from psycopg.rows import dict_row

    from baloot_platform.config import build_bundle
    from baloot_platform.configuration import PostgresConfigurationStore, VersionedConfigurationService
    from baloot_platform.identity import IdentityService
    from baloot_platform.identity.store import PostgresIdentityStore
    from baloot_platform.security import oidc
    from baloot_platform.security.oidc_migration import (
        finalize_legacy_oidc,
        legacy_oidc_profile,
        migrate_legacy_oidc,
        rollback_legacy_oidc,
    )
    from baloot_platform.security.state_v20 import PostgresSecurityStateStore
    from baloot_platform.service.routes.security import build_security_router
    from baloot_platform.storage.request_context import (
        CURRENT_DATABASE_IDENTITY,
        DatabaseIdentityContext,
        apply_postgres_identity_context,
    )

    dsn = os.environ.get("BALOOT_TEST_OIDC_POSTGRES_DSN")
    assert dsn, "hosted PostgreSQL proof requires BALOOT_TEST_OIDC_POSTGRES_DSN"
    legacy_ref = os.environ.get("BALOOT_TEST_OIDC_LEGACY_REF", "")
    assert re.fullmatch(r"[0-9a-f]{40}", legacy_ref), "hosted proof requires an exact previous release reference"
    old_oidc = _previous_module(root, monkeypatch, legacy_ref, "baloot_platform.security.oidc")
    old_state = _previous_module(root, monkeypatch, legacy_ref, "baloot_platform.security.state_v20")
    old_routes = _previous_module(root, monkeypatch, legacy_ref, "baloot_platform.service.routes.security")
    monkeypatch.setenv("BALOOT_DATABASE_URL", dsn)
    monkeypatch.setenv("BALOOT_ROOT", str(root))
    migrate_postgres.main()
    with psycopg.connect(dsn) as conn:
        conn.execute("CREATE ROLE oidc_ci_app LOGIN NOSUPERUSER NOBYPASSRLS")
        conn.execute("GRANT USAGE ON SCHEMA configuration, identity, security TO oidc_ci_app")
        conn.execute("GRANT SELECT ON configuration.profile_revisions, configuration.invalidation_epochs TO oidc_ci_app")
        conn.execute("GRANT SELECT, INSERT ON identity.subjects TO oidc_ci_app")
        conn.execute("GRANT INSERT ON identity.sessions TO oidc_ci_app")
        conn.execute("GRANT SELECT, INSERT, UPDATE ON security.oidc_transactions, security.external_identities TO oidc_ci_app")
    app_dsn = make_conninfo(dsn, user="oidc_ci_app")
    with psycopg.connect(app_dsn) as conn:
        assert conn.execute("SELECT current_user, session_user").fetchone() == ("oidc_ci_app", "oidc_ci_app")
    legacy = "https://signin.example/"
    canonical = "https://signin.example"
    legacy_metadata = {
        "ISSUER": legacy,
        "AUTHORIZATION_ENDPOINT": canonical + "/authorize",
        "TOKEN_ENDPOINT": canonical + "/token",
        "CLIENT_ID": "fixture-client",
        "REDIRECT_URI": "https://app.example/callback",
        "CLIENT_SECRET": "synthetic-client-fixture",
    }
    for field, value in legacy_metadata.items():
        monkeypatch.setenv(f"BALOOT_OIDC_{field}", value)
    monkeypatch.delenv("BALOOT_OIDC_PROVIDER_ID", raising=False)
    monkeypatch.setenv("BALOOT_OIDC_JWKS_URI", canonical + "/legacy-keys")

    keys_uri = canonical + "/keys"
    signing_key = Ed25519PrivateKey.from_private_bytes(b"a" * 32)
    jwk = {**json.loads(jwt.algorithms.OKPAlgorithm.to_jwk(signing_key.public_key())), "kid": "fixture"}
    now = 1_767_225_600
    outstanding_codes = {}

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls.fromtimestamp(now, tz)

    monkeypatch.setattr(jwt.api_jwt, "datetime", FixedDateTime)
    monkeypatch.setattr(oidc, "time", SimpleNamespace(time=lambda: now))
    monkeypatch.setattr(old_oidc, "time", SimpleNamespace(time=lambda: now))

    def fetch_keys(_opener, request, timeout=None):
        assert request.full_url == keys_uri
        return BytesIO(json.dumps({"keys": [jwk]}).encode())

    monkeypatch.setattr(OpenerDirector, "open", fetch_keys)

    def provider_http(request):
        if request.method == "GET":
            assert str(request.url) == canonical + "/.well-known/openid-configuration"
            return httpx.Response(200, json={"issuer": canonical, "jwks_uri": keys_uri})
        assert CURRENT_DATABASE_IDENTITY.get().role == "anonymous"
        assert str(request.url) == legacy_metadata["TOKEN_ENDPOINT"]
        form = parse_qs(request.content.decode())
        nonce, challenge, subject, issuer = outstanding_codes.pop(form["code"][0])
        assert oidc.PkceTransactionStore._challenge(form["code_verifier"][0]) == challenge
        assert form["client_id"] == [legacy_metadata["CLIENT_ID"]]
        assert form["redirect_uri"] == [legacy_metadata["REDIRECT_URI"]]
        assert form["client_secret"] == [legacy_metadata["CLIENT_SECRET"]]
        claims = {"iss": issuer, "aud": form["client_id"][0], "sub": subject,
                  "nonce": nonce, "iat": now, "exp": now + 60}
        encoded = jwt.encode(claims, signing_key, algorithm="EdDSA", headers={"kid": "fixture"})
        return httpx.Response(200, json={"id_token": encoded})

    with httpx.Client(transport=httpx.MockTransport(provider_http)) as client:
        source, payload = legacy_oidc_profile(
            client, profile_id="signin", version="1.0.0",
            reason="Migrate fixture links", algorithms=["EdDSA"],
        )
    assert source == legacy
    assert payload["providers"]["default"]["issuer"] == canonical
    assert payload["providers"]["default"]["client_secret_env"] == "BALOOT_OIDC_CLIENT_SECRET"
    assert "synthetic-client-fixture" not in str(payload)
    assert sum("legacy JWKS override is ignored" in record.message for record in caplog.records) == 1

    profile = build_bundle(root).component("configuration")

    @contextmanager
    def operator_connection():
        token = CURRENT_DATABASE_IDENTITY.set(DatabaseIdentityContext(subject_id="operator-1", role="service"))
        try:
            with psycopg.connect(dsn, row_factory=dict_row) as conn:
                yield conn
        finally:
            CURRENT_DATABASE_IDENTITY.reset(token)

    def bind(conn):
        apply_postgres_identity_context(conn)
        return VersionedConfigurationService(
            PostgresConfigurationStore(dsn, transaction_connection=conn),
            root=root, profile=profile,
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

    def begin(client, *, code, subject, issuer=canonical):
        response = client.post("/api/v1/security/oidc:begin", json={"providerId": "default", "returnTo": "/app"})
        assert response.status_code == 200, response.text
        started = response.json()
        query = parse_qs(urlsplit(started["authorizationUrl"]).query)
        outstanding_codes[code] = (query["nonce"][0], query["code_challenge"][0], subject, issuer)
        return {"providerId": "default", "state": started["state"], "code": code}

    def callback(client, transaction):
        return client.post("/api/v1/security/oidc:callback", json=transaction)

    def complete(client, transaction):
        response = callback(client, transaction)
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["returnTo"] == "/app"
        assert result["session"]["sessionId"].startswith("SES-")
        replay = callback(client, transaction)
        assert replay.status_code == 400 and "reused OIDC state" in replay.json()["detail"]
        return result["identity"]["subjectId"]

    with ExitStack() as stack:
        provider_client = stack.enter_context(httpx.Client(transport=httpx.MockTransport(provider_http)))
        legacy_store = old_state.PostgresSecurityStateStore(app_dsn)
        stack.callback(legacy_store.close)
        identity_store = PostgresIdentityStore(app_dsn)
        stack.callback(identity_store.pool.close)
        identities = IdentityService(identity_store)
        identities.create_external(kind="oidc", subject_id="local-existing", display_name="Existing fixture")
        legacy_provider = old_oidc.OidcProviderConfig(
            provider_id="default", jwks_uri=keys_uri,
            **{name.lower(): value for name, value in legacy_metadata.items()},
        )
        legacy_transactions = old_oidc.PkceTransactionStore(backend=legacy_store)
        legacy_providers = {"default": legacy_provider}
        previous = FastAPI()
        previous.include_router(old_routes.build_security_router(
            identities, None, None, legacy_transactions, legacy_providers,
            old_oidc.OidcCallbackService(legacy_providers, legacy_transactions, http_client=provider_client),
            None, None, legacy_store,
        ))
        previous_client = stack.enter_context(TestClient(previous))
        legacy_existing = begin(previous_client, code="old-existing", subject="existing")
        legacy_new = begin(previous_client, code="old-new", subject="old-callback")

        with operator_connection() as conn:
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
        with operator_connection() as conn:
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
        store = PostgresSecurityStateStore(app_dsn)
        bypassing_store = PostgresSecurityStateStore(dsn)
        stack.callback(store.close)
        stack.callback(bypassing_store.close)
        store.require_oidc_rls()
        with pytest.raises(RuntimeError, match="subject to row-level security"):
            bypassing_store.require_oidc_rls()
        with psycopg.connect(app_dsn) as conn:
            assert conn.execute("SELECT count(*) FROM security.external_identities").fetchone() == (0,)
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            with psycopg.connect(app_dsn) as conn:
                link(conn, canonical, "unverified")

        configuration_store = PostgresConfigurationStore(app_dsn)
        stack.callback(configuration_store.close)
        monkeypatch.setenv("BALOOT_OIDC_PROVIDERS_PROFILE", "signin")
        runtime = oidc.HttpOidcRuntime(
            VersionedConfigurationService(configuration_store, root=root, profile=profile), store,
        )
        runtime.client.close()
        runtime.client = provider_client
        runtime.current(startup=True)
        app = FastAPI()
        app.state.oidc = runtime
        app.include_router(build_security_router(identities, None, None, None, None, store))
        current_client = stack.enter_context(TestClient(app))
        assert complete(current_client, legacy_existing) == "local-existing"
        legacy_subject = complete(current_client, legacy_new)
        assert complete(current_client, begin(current_client, code="new-existing", subject="existing")) == "local-existing"
        new_subject = complete(current_client, begin(current_client, code="new-new", subject="new-callback"))
        assert new_subject not in {legacy_subject, "local-existing"}
        assert store.get_external_identity("default", canonical, "old-callback")["local_subject_id"] == legacy_subject
        assert store.get_external_identity("default", legacy, "new-callback")["local_subject_id"] == new_subject
        with pytest.raises(psycopg.errors.RaiseException, match="already linked"):
            store.link_external_identity(
                provider_id="default", issuer=canonical, external_subject="new-callback",
                local_subject_id="wrong-account", claims={}, linked_at="2026-01-01T00:00:00Z",
            )

        # An old callback can consume a new transaction and verify PKCE, but RLS still blocks its write.
        wrong_route = begin(current_client, code="wrong-route", subject="legacy-write-denied", issuer=legacy)
        with pytest.raises(psycopg.errors.InsufficientPrivilege, match="row-level security"):
            callback(previous_client, wrong_route)
        assert wrong_route["code"] not in outstanding_codes
        replay = callback(current_client, wrong_route)
        assert replay.status_code == 400 and "reused OIDC state" in replay.json()["detail"]
        assert store.get_external_identity("default", legacy, "legacy-write-denied") is None

        drain = begin(current_client, code="drain", subject="existing")
        expired = begin(current_client, code="expire", subject="existing")
        with operator_connection() as conn:
            assert lookup(conn, legacy, "new-callback")["local_subject_id"] == new_subject
            newer = deepcopy(payload)
            newer["profile"]["version"] = "1.0.1"
            newer_revision = bind(conn).create_profile(
                family="oidc_providers", payload=newer, created_by="operator-1", publish=True,
            )
        assert complete(current_client, drain) == "local-existing"
        now += runtime.transactions.ttl_seconds
        response = callback(current_client, expired)
        assert response.status_code == 400 and "OIDC state" in response.json()["detail"]
        assert expired["code"] in outstanding_codes

        with operator_connection() as conn:
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
            for subject, local_subject in {
                "existing": "local-existing", "old-callback": legacy_subject, "new-callback": new_subject,
            }.items():
                assert lookup(conn, canonical, subject) is None
                assert lookup(conn, legacy, subject) == {
                    "local_subject_id": local_subject, "issuer_alias": False,
                }
            audit = configuration.store.events_with_verification("profile", revision_id)
            assert audit["verification"]["valid"]
            assert audit["items"][-1]["eventType"] == "oidc_issuer_rolled_back"
        payload["profile"]["version"] = "1.0.2"
        assert current_client.post("/api/v1/security/oidc:begin", json={"providerId": "default"}).status_code == 404
        assert callback(current_client, drain).json()["detail"] == "OIDC provider is not configured"
        previous_existing = begin(previous_client, code="rollback-legacy", subject="existing", issuer=legacy)
        with pytest.raises(psycopg.errors.InsufficientPrivilege, match="row-level security"):
            callback(previous_client, previous_existing)
        assert store.get_external_identity("default", legacy, "existing")["local_subject_id"] == "local-existing"
        with operator_connection() as conn:
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
