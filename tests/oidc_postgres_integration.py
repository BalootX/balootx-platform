from __future__ import annotations

import ast
import gc
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing, contextmanager
from copy import deepcopy
from datetime import datetime
from io import BytesIO
from threading import Barrier
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


def _previous_middleware(root, ref, api, identities):
    path = "src/baloot_platform/service/api.py"
    source = subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=root)
    middleware, = [node for node in ast.parse(source).body
                   if isinstance(node, ast.AsyncFunctionDef) and node.name == "platform_security"]
    middleware.decorator_list = []
    namespace = {**vars(api), "IDENTITY": identities}
    exec(compile(ast.Module(body=[middleware], type_ignores=[]), path, "exec"), namespace)
    return namespace["platform_security"]


def test_sqlite_oidc_storage_survives_release_roundtrip(root, tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from baloot_platform.config import build_bundle
    from baloot_platform.configuration import SQLiteConfigurationStore, VersionedConfigurationService
    from baloot_platform.identity import IdentityService
    from baloot_platform.identity.store import SQLiteIdentityStore
    from baloot_platform.security.oidc import http_oidc_runtime
    from baloot_platform.security.state_v20 import (
        SQLiteOidcIssuerChange,
        SQLiteSecurityStateStore,
        complete_oidc_sign_in,
    )
    from baloot_platform.service.routes.security import build_security_router

    ref = os.environ.get("BALOOT_TEST_OIDC_LEGACY_REF", "")
    assert re.fullmatch(r"[0-9a-f]{40}", ref)
    old_state = _previous_module(root, monkeypatch, ref, "baloot_platform.security.state_v20")
    old_identity = _previous_module(root, monkeypatch, ref, "baloot_platform.identity.store")
    old_service = _previous_module(root, monkeypatch, ref, "baloot_platform.identity.service")
    security_path, identity_path = tmp_path / "security.sqlite3", tmp_path / "identity.sqlite3"
    provider, issuer = "default", "https://signin.example"

    def previous_login(subject):
        store = old_state.SQLiteSecurityStateStore(security_path)
        service = old_service.IdentityService(old_identity.SQLiteIdentityStore(identity_path))
        link = store.get_external_identity(provider, issuer, subject)
        session = service.create_external(
            kind="oidc", display_name="Fixture", subject_id=link["local_subject_id"] if link else None,
        )
        owner = session["identity"]["subjectId"]
        store.link_external_identity(
            provider_id=provider, issuer=issuer, external_subject=subject,
            local_subject_id=owner, claims={}, linked_at="2026-01-01T00:00:00Z",
        )
        return owner

    def current_login(subject):
        gc.collect()  # The legacy store relies on connection destruction on process exit.
        identities = IdentityService(SQLiteIdentityStore(identity_path))
        store = SQLiteSecurityStateStore(security_path)
        session = complete_oidc_sign_in(
            store, identities, {"providerId": provider, "issuer": issuer, "externalSubject": subject, "claims": {}},
            display_name="Fixture", user_agent=None,
        )
        return session["identity"]["subjectId"]

    existing = previous_login("existing")
    assert current_login("existing") == existing
    upgraded = current_login("upgraded")
    assert previous_login("upgraded") == upgraded
    downgraded = previous_login("downgraded")
    assert current_login("downgraded") == downgraded
    assert current_login("upgraded") == upgraded
    assert len({existing, upgraded, downgraded}) == 3

    identities = SQLiteIdentityStore(identity_path)
    state = SQLiteSecurityStateStore(security_path)
    configuration_store = SQLiteConfigurationStore(tmp_path / "configuration.sqlite3")
    configuration = VersionedConfigurationService(
        configuration_store, root=root, profile=build_bundle(root).component("configuration"),
    )
    payload = {
        "schema_version": 1,
        "profile": {"id": "signin", "version": "1.0.0"},
        "change_reason": "Configure fixture provider",
        "providers": {provider: {
            "issuer": issuer,
            "authorization_endpoint": issuer + "/authorize",
            "token_endpoint": issuer + "/token",
            "client_id": "fixture-client",
            "redirect_uri": "https://app.example/callback",
            "allowed_algorithms": ["EdDSA"],
        }},
    }
    monkeypatch.setenv("BALOOT_OIDC_PROVIDERS_PROFILE", "signin")
    monkeypatch.setenv("BALOOT_OIDC_ISSUER", issuer)
    monkeypatch.delenv("BALOOT_OIDC_PROVIDER_ID", raising=False)
    configuration.create_profile(family="oidc_providers", payload=payload, created_by="fixture", publish=True)
    with http_oidc_runtime(configuration, state) as runtime:
        assert runtime.current().providers[provider].issuer == issuer
        app = FastAPI()
        app.state.oidc = runtime
        app.include_router(build_security_router(IdentityService(identities), None, None, None, None, state))
        with closing(identities._connect()) as conn:
            before = conn.execute("SELECT count(*) FROM identity_sessions_v8").fetchone()[0]
        payload["profile"]["version"] = "1.0.1"
        payload["providers"][provider]["issuer"] = issuer + "/"
        configuration.create_profile(family="oidc_providers", payload=payload, created_by="fixture", publish=True)
        with pytest.raises(ValueError, match="requires its original issuer"):
            runtime.current()
        monkeypatch.delenv("BALOOT_OIDC_ISSUER")
        with pytest.raises(SQLiteOidcIssuerChange, match="requires its original issuer"):
            runtime.current()
        unavailable = {"detail": "OIDC provider configuration is unavailable. Contact support to continue."}
        callback_body = {"providerId": provider, "state": "fixture-state", "code": "fixture-code"}
        with TestClient(app) as client:
            response = client.post("/api/v1/security/oidc:begin", json={"providerId": provider})
            assert response.status_code == 503 and response.json() == unavailable
            response = client.post("/api/v1/security/oidc:callback", json=callback_body)
            assert response.status_code == 503 and response.json() == unavailable

            # The store must also reject a verified issuer change after runtime selection.
            verified = {"providerId": provider, "issuer": issuer + "/", "externalSubject": "existing", "claims": {}}
            callback = SimpleNamespace(complete=lambda **_body: verified)
            app.state.oidc = SimpleNamespace(current=lambda: callback)
            response = client.post("/api/v1/security/oidc:callback", json=callback_body)
            assert response.status_code == 503 and response.json() == unavailable
            assert "set-cookie" not in response.headers
        with closing(state._connect()) as conn:
            assert conn.execute("SELECT count(*) FROM security_oidc_transactions_v20").fetchone()[0] == 0
        with pytest.raises(SQLiteOidcIssuerChange, match="requires its original issuer"):
            complete_oidc_sign_in(
                state, IdentityService(identities),
                verified,
                display_name="Fixture", user_agent=None,
            )
        with closing(identities._connect()) as conn:
            assert conn.execute("SELECT count(*) FROM identity_sessions_v8").fetchone()[0] == before
            assert conn.execute("SELECT count(*) FROM identity_subjects_v8").fetchone()[0] == 3
        assert state.get_external_identity(provider, issuer, "existing")["local_subject_id"] == existing
        assert state.get_external_identity(provider, issuer + "/", "existing") is None
    with pytest.raises(SQLiteOidcIssuerChange, match="requires its original issuer"):
        with http_oidc_runtime(configuration, state):
            pytest.fail("SQLite must refuse startup with the incompatible issuer")
    configuration_store.close()


def _sqlite_oidc_login(directory, subject):
    from pathlib import Path

    from baloot_platform.identity import IdentityService, SQLiteIdentityStore
    from baloot_platform.security.state_v20 import SQLiteSecurityStateStore, complete_oidc_sign_in

    directory = Path(directory)
    store = SQLiteSecurityStateStore(directory / "security.sqlite3")
    identities = IdentityService(SQLiteIdentityStore(directory / "identity.sqlite3"))
    return complete_oidc_sign_in(
        store, identities,
        {"providerId": "default", "issuer": "https://signin.example", "externalSubject": subject, "claims": {}},
        display_name="Fixture", user_agent=None,
    )


def test_sqlite_oidc_recovers_across_super_journal_commit(root, tmp_path, monkeypatch):
    from baloot_platform.identity import SQLiteIdentityStore
    from baloot_platform.security.state_v20 import SQLiteSecurityStateStore

    ref = os.environ.get("BALOOT_TEST_OIDC_LEGACY_REF", "")
    assert re.fullmatch(r"[0-9a-f]{40}", ref)
    old_state = _previous_module(root, monkeypatch, ref, "baloot_platform.security.state_v20")
    old_identity = _previous_module(root, monkeypatch, ref, "baloot_platform.identity.store")
    interrupt = tmp_path / "interrupt.so"
    subprocess.run(
        ["cc", "-shared", "-fPIC", "-o", str(interrupt),
         str(root / "tests/fixtures/sqlite_super_journal_interrupt.c"), "-ldl"],
        check=True, timeout=30,
    )
    for phase, exit_code, committed in (("before", 81, False), ("after", 82, True)):
        for reader, identity_store, security_store in (
            ("current", SQLiteIdentityStore, SQLiteSecurityStateStore),
            ("previous", old_identity.SQLiteIdentityStore, old_state.SQLiteSecurityStateStore),
        ):
            directory = tmp_path / f"{phase}-{reader}"
            existing = _sqlite_oidc_login(directory, "existing")["identity"]["subjectId"]
            child = subprocess.run(
                [sys.executable, "-B", "-c",
                 "import runpy, sys; runpy.run_path(sys.argv[1])['_sqlite_oidc_login'](sys.argv[2], 'interrupted')",
                 __file__, str(directory)],
                cwd=root, timeout=30,
                env={**os.environ, "LD_PRELOAD": str(interrupt), "BALOOT_TEST_SQLITE_COMMIT_INTERRUPTION": phase},
            )
            assert child.returncode == exit_code, f"child did not stop at the {phase} commit boundary"
            assert (directory / "identity.sqlite3-journal").is_file()
            assert (directory / "security.sqlite3-journal").is_file()
            assert bool(list(directory.glob("identity.sqlite3-mj*"))) is not committed

            links = security_store(directory / "security.sqlite3")
            identities = identity_store(directory / "identity.sqlite3")
            owner = links.get_external_identity("default", "https://signin.example", "interrupted")
            assert (owner is not None) is committed
            subject_ids = {existing}
            if owner is not None:
                subject_ids.add(owner["local_subject_id"])
            with closing(identities._connect()) as conn:
                assert [row[0] for row in conn.execute("PRAGMA integrity_check")] == ["ok"]
                assert list(conn.execute("PRAGMA foreign_key_check")) == []
                assert {row[0] for row in conn.execute("SELECT subject_id FROM identity_subjects_v8")} == subject_ids
                assert list(map(tuple, conn.execute(
                    "SELECT subject_id, count(*) FROM identity_sessions_v8 GROUP BY subject_id ORDER BY subject_id"
                ))) == [(subject, 1) for subject in sorted(subject_ids)]
            with closing(links._connect()) as conn:
                assert [row[0] for row in conn.execute("PRAGMA integrity_check")] == ["ok"]
                assert {row[0] for row in conn.execute(
                    "SELECT local_subject_id FROM security_external_identities_v20"
                )} == subject_ids
            gc.collect()
            retried = _sqlite_oidc_login(directory, "interrupted")["identity"]["subjectId"]
            assert retried != existing
            if owner is not None:
                assert retried == owner["local_subject_id"]


def test_oidc_issuer_transition_on_postgres(root, tmp_path, monkeypatch, caplog, capsys):
    import httpx
    import jwt
    import psycopg
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from psycopg import sql
    from psycopg.conninfo import make_conninfo
    from psycopg.rows import dict_row

    from baloot_platform.config import build_bundle
    from baloot_platform.configuration import PostgresConfigurationStore, VersionedConfigurationService
    from baloot_platform.identity import IdentityService
    from baloot_platform.identity.store import PostgresIdentityStore
    from baloot_platform.security import oidc
    from baloot_platform.security.access import Scope
    from baloot_platform.security.oidc_migration import (
        finalize_legacy_oidc,
        legacy_oidc_profile,
        main as oidc_migration_command,
        migrate_legacy_oidc,
        rollback_legacy_oidc,
    )
    from baloot_platform.security.state_v20 import OidcIdentityConflict, PostgresSecurityStateStore
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
    old_registry = _previous_module(root, monkeypatch, legacy_ref, "baloot_platform.service.routes.configuration")
    monkeypatch.setenv("BALOOT_DATABASE_URL", dsn)
    monkeypatch.setenv("BALOOT_ROOT", str(root))
    migrate_postgres.main()
    with psycopg.connect(dsn) as conn:
        conn.execute("CREATE ROLE oidc_ci_app LOGIN NOSUPERUSER NOBYPASSRLS")
        conn.execute("GRANT USAGE ON SCHEMA configuration, hitl, identity, security TO oidc_ci_app")
        conn.execute(
            "GRANT SELECT, INSERT, UPDATE ON configuration.profile_revisions, configuration.invalidation_epochs,"
            " configuration.bundle_revisions, configuration.aggregate_heads TO oidc_ci_app"
        )
        conn.execute("GRANT SELECT, INSERT ON configuration.events TO oidc_ci_app")
        conn.execute("GRANT SELECT, INSERT ON identity.subjects TO oidc_ci_app")
        conn.execute("GRANT SELECT, INSERT, UPDATE ON identity.sessions TO oidc_ci_app")
        conn.execute("GRANT SELECT, INSERT, UPDATE ON security.oidc_transactions, security.external_identities TO oidc_ci_app")
        conn.execute("CREATE ROLE oidc_ci_migrator LOGIN NOSUPERUSER NOBYPASSRLS")
        conn.execute("GRANT USAGE ON SCHEMA configuration, security TO oidc_ci_migrator")
        conn.execute(
            "GRANT SELECT, INSERT, UPDATE ON configuration.profile_revisions, configuration.aggregate_heads,"
            " configuration.invalidation_epochs, security.oidc_issuer_transitions TO oidc_ci_migrator"
        )
        conn.execute("GRANT SELECT, INSERT ON configuration.events TO oidc_ci_migrator")
        conn.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON security.external_identities TO oidc_ci_migrator")
        # The assembled application also bootstraps its unrelated Research store.
        database, = conn.execute("SELECT current_database()").fetchone()
        conn.execute(sql.SQL("GRANT CREATE ON DATABASE {} TO oidc_ci_app").format(sql.Identifier(database)))
        conn.execute("ALTER SCHEMA research OWNER TO oidc_ci_app")
        for (relation,) in conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname='research'"
        ).fetchall():
            conn.execute(sql.SQL("ALTER TABLE research.{} OWNER TO oidc_ci_app").format(sql.Identifier(relation)))
    app_dsn = make_conninfo(dsn, user="oidc_ci_app")
    migration_dsn = make_conninfo(dsn, user="oidc_ci_migrator")
    for login_dsn, login, can_migrate in (
        (app_dsn, "oidc_ci_app", False), (migration_dsn, "oidc_ci_migrator", True),
    ):
        with psycopg.connect(login_dsn) as conn:
            assert conn.execute(
                "SELECT current_user, session_user, "
                "row_security_active('security.external_identities'), "
                "row_security_active('configuration.profile_revisions'), "
                "has_table_privilege('security.oidc_issuer_transitions', 'UPDATE'), "
                "has_table_privilege('security.external_identities', 'DELETE'), "
                "has_function_privilege('security.mirror_oidc_issuer_link()', 'EXECUTE')"
            ).fetchone() == (login, login, True, True, can_migrate, can_migrate, False)
    with psycopg.connect(migration_dsn) as conn:
        assert not conn.execute(
            "SELECT has_schema_privilege('identity', 'USAGE')"
        ).fetchone()[0]
    legacy = "https://signin.example/"
    canonical = "https://signin.example"
    service_key = "synthetic-oidc-api-key"
    legacy_headers = {"x-api-key": service_key, "x-baloot-role": "service"}
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
    race = Barrier(2)

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
        assert str(request.url) == legacy_metadata["TOKEN_ENDPOINT"]
        form = parse_qs(request.content.decode())
        nonce, challenge, subject, issuer = outstanding_codes.pop(form["code"][0])
        identity_context = CURRENT_DATABASE_IDENTITY.get()
        assert identity_context.role == ("service" if issuer == legacy else "anonymous")
        assert identity_context.oidc_identity_authority == ""
        assert oidc.PkceTransactionStore._challenge(form["code_verifier"][0]) == challenge
        assert form["client_id"] == [legacy_metadata["CLIENT_ID"]]
        assert form["redirect_uri"] == [legacy_metadata["REDIRECT_URI"]]
        assert form["client_secret"] == [legacy_metadata["CLIENT_SECRET"]]
        if form["code"][0] in {"race-first", "race-second"}:
            race.wait(timeout=15)
        claims = {"iss": issuer, "aud": form["client_id"][0], "sub": subject,
                  "nonce": nonce, "iat": now, "exp": now + 60}
        key = Ed25519PrivateKey.from_private_bytes(b"b" * 32) if form["code"][0] == "invalid-signature" else signing_key
        encoded = jwt.encode(claims, key, algorithm="EdDSA", headers={"kid": "fixture"})
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
    def migration_connection():
        context = DatabaseIdentityContext(
            subject_id="operator-1", role="service", configuration_admin=True, oidc_identity_authority="migration",
        )
        token = CURRENT_DATABASE_IDENTITY.set(context)
        try:
            with psycopg.connect(migration_dsn, row_factory=dict_row) as conn:
                apply_postgres_identity_context(conn, context)
                yield conn
        finally:
            CURRENT_DATABASE_IDENTITY.reset(token)

    def bind(conn):
        return VersionedConfigurationService(
            PostgresConfigurationStore(migration_dsn, transaction_connection=conn),
            root=root, profile=profile,
        )

    def command(*arguments, reason="Migrate fixture links", login_dsn=migration_dsn):
        with monkeypatch.context() as invocation:
            invocation.setenv("BALOOT_DATABASE_URL", login_dsn)
            invocation.setenv("BALOOT_OIDC_PROVIDERS_PROFILE", "signin")
            invocation.setattr(sys, "argv", [
                "baloot-migrate-oidc", "--actor", "operator-1", "--reason", reason, *arguments,
            ])
            capsys.readouterr()
            oidc_migration_command()
            return json.loads(capsys.readouterr().out)

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

    def begin(client, *, code, subject, issuer=canonical, headers=None):
        response = client.post(
            "/api/v1/security/oidc:begin", headers=headers, json={"providerId": "default", "returnTo": "/app"},
        )
        assert response.status_code == 200, response.text
        started = response.json()
        query = parse_qs(urlsplit(started["authorizationUrl"]).query)
        outstanding_codes[code] = (query["nonce"][0], query["code_challenge"][0], subject, issuer)
        return {"providerId": "default", "state": started["state"], "code": code}

    def callback(client, transaction, *, headers=None):
        return client.post("/api/v1/security/oidc:callback", headers=headers, json=transaction)

    def linked_owner(issuer, subject):
        with psycopg.connect(dsn, row_factory=dict_row) as conn:
            item = lookup(conn, issuer, subject)
            return item["local_subject_id"] if item else None

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
        monkeypatch.setenv("BALOOT_DATABASE_URL", app_dsn)
        monkeypatch.setenv("BALOOT_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("BALOOT_RUNTIME", str(tmp_path / "runtime"))
        monkeypatch.setenv("BALOOT_CHALLENGE_LOCATOR_KEY", "fixture-" * 5)
        monkeypatch.setenv("BALOOT_CURATED_SAKKA_SIGNING_KEY", "fixture-" * 5)
        monkeypatch.setenv("BALOOT_API_KEY", service_key)
        monkeypatch.setenv("BALOOT_ENFORCE_SESSION_CSRF", "true")
        monkeypatch.setenv("BALOOT_RATE_LIMIT_PER_MINUTE", "0")
        monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda _self, request: provider_http(request))
        from baloot_platform.service import api

        stack.callback(api.CONTAINER.shutdown)
        provider_client = stack.enter_context(httpx.Client(transport=httpx.MockTransport(provider_http)))
        legacy_store = old_state.PostgresSecurityStateStore(app_dsn)
        stack.callback(legacy_store.close)
        identity_store = PostgresIdentityStore(app_dsn)
        stack.callback(identity_store.pool.close)
        identities = IdentityService(identity_store)
        identities.create_external(kind="oidc", subject_id="local-existing", display_name="Existing fixture")
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            with psycopg.connect(migration_dsn) as conn:
                identity_store.insert_session({
                    "session_id": "SES-refused-migrator",
                    "subject_id": "local-existing",
                    "token_hash": "fixture-migrator-token-hash",
                    "csrf_token": "fixture-migrator-csrf",
                    "created_at": "2026-01-01T00:00:00Z",
                    "expires_at": "2026-01-01T01:00:00Z",
                    "last_seen_at": "2026-01-01T00:00:00Z",
                }, connection=conn)
        legacy_provider = old_oidc.OidcProviderConfig(
            provider_id="default", jwks_uri=keys_uri,
            **{name.lower(): value for name, value in legacy_metadata.items()},
        )
        legacy_transactions = old_oidc.PkceTransactionStore(backend=legacy_store)
        legacy_providers = {"default": legacy_provider}
        previous = FastAPI()
        previous.middleware("http")(_previous_middleware(root, legacy_ref, api, identities))
        previous.include_router(old_routes.build_security_router(
            identities, None, None, legacy_transactions, legacy_providers,
            old_oidc.OidcCallbackService(legacy_providers, legacy_transactions, http_client=provider_client),
            None, None, legacy_store,
        ))
        legacy_configuration = PostgresConfigurationStore(app_dsn)
        stack.callback(legacy_configuration.close)
        previous.include_router(old_registry.build_configuration_router(
            VersionedConfigurationService(legacy_configuration, root=root, profile=profile),
            identities, None, profile, {},
        ))
        previous_client = stack.enter_context(TestClient(previous))
        assert previous_client.post("/api/v1/security/oidc:begin", json={"providerId": "default"}).status_code == 401
        legacy_existing = begin(previous_client, code="old-existing", subject="existing", headers=legacy_headers)
        legacy_new = begin(previous_client, code="old-new", subject="old-callback", headers=legacy_headers)

        with psycopg.connect(dsn) as conn:
            link(conn, canonical, "collision")
            link(conn, legacy, "existing")
        with migration_connection() as conn:
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
            with pytest.raises(ValueError, match="discovered issuer already has identity links"):
                migrate_legacy_oidc(
                    conn, configuration, payload=payload, legacy_issuer=legacy, actor="operator-1",
                )
        with psycopg.connect(dsn) as conn:
            conn.execute("DELETE FROM security.external_identities WHERE external_subject='collision'")
        migration = command("--version", "1.0.0", "--algorithm", "EdDSA")
        assert migration["status"] == "applied" and migration["links"] == 1
        revision_id = migration["revisionId"]
        for refused_dsn in (app_dsn, dsn):
            with pytest.raises(SystemExit, match="requires a dedicated login"):
                command("--rollback", revision_id, login_dsn=refused_dsn)
        assert command(
            "--version", "1.0.0", "--algorithm", "EdDSA", reason="Repeat the existing command",
        ) == {"status": "already-applied", "revisionId": revision_id}
        with migration_connection() as conn:
            configuration = bind(conn)
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
        migration_store = PostgresSecurityStateStore(migration_dsn)
        stack.callback(store.close)
        stack.callback(bypassing_store.close)
        stack.callback(migration_store.close)
        store.require_oidc_rls()
        for invalid_store in (bypassing_store, migration_store):
            with pytest.raises(RuntimeError, match="subject to row-level security"):
                invalid_store.require_oidc_rls()
        with psycopg.connect(app_dsn) as conn:
            assert conn.execute("SELECT count(*) FROM security.external_identities").fetchone() == (0,)
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            with psycopg.connect(app_dsn) as conn:
                link(conn, canonical, "unverified")

        monkeypatch.setenv("BALOOT_OIDC_PROVIDERS_PROFILE", "signin")
        current_client = stack.enter_context(TestClient(api.app, base_url="https://app.example"))
        runtime = api.app.state.oidc
        assert current_client.get("/readyz").status_code == 200
        guest = current_client.post("/api/v1/auth/guest", json={"displayName": "Fixture guest"})
        assert guest.status_code == 201
        cookie = current_client.cookies.pop("baloot_session")
        session_headers = {"cookie": f"baloot_session={cookie}"}
        assert current_client.post("/api/v1/security/tokens", json={}).status_code == 401
        assert current_client.post("/api/v1/security/oidc:begin/extra", json={}).status_code == 401
        assert current_client.post("/api/v1/security/oidc:begin", content="{}").status_code == 415
        for operation, body in (("begin", {"providerId": "default"}), ("callback", legacy_existing)):
            route = f"/api/v1/security/oidc:{operation}"
            assert current_client.post(route, json=body, headers=session_headers).status_code == 403
        accepted = current_client.post(
            "/api/v1/security/oidc:begin", json={"providerId": "default"},
            headers={**session_headers, "x-csrf-token": guest.json()["csrfToken"]},
        )
        assert accepted.status_code == 200
        assert "set-cookie" not in accepted.headers

        def assert_registry_boundary(revision):
            prefix = "/api/v1/configuration-registry/profiles"
            paths = (
                f"{prefix}/{revision}",
                f"{prefix}:compare?left={revision}&right={revision}",
                f"{prefix}/{revision}/events",
            )
            admin = {"x-api-key": service_key, "x-baloot-role": "admin"}
            ordinary = {**session_headers, "x-baloot-role": "admin", "x-api-key": "incorrect"}
            narrowed = api.ACCESS_TOKENS.issue(subject_id="fixture-operator", scopes=(Scope.READ,), role="admin")
            signed_admin = api.ACCESS_TOKENS.issue(
                subject_id="fixture-operator", scopes=(Scope.CONFIG_ADMIN,), role="admin",
            )
            for path in paths:
                assert current_client.get(path, headers=admin).status_code == 200
                assert current_client.get(path, headers={"authorization": f"Bearer {signed_admin}"}).status_code == 200
                assert current_client.get(path, headers=ordinary).status_code == 404
                assert current_client.get(path, headers={"authorization": f"Bearer {narrowed}"}).status_code == 404
                assert previous_client.get(path, headers=session_headers).status_code == 404
                assert previous_client.get(path, headers=legacy_headers).status_code == 404
            with psycopg.connect(app_dsn) as conn:
                apply_postgres_identity_context(conn, DatabaseIdentityContext(subject_id="fixture", role="admin"))
                assert conn.execute(
                    "SELECT count(*) FROM configuration.events WHERE aggregate_id=%s", (revision,)
                ).fetchone() == (0,)

        assert_registry_boundary(revision_id)
        unsigned_identity = begin(current_client, code="invalid-signature", subject="invalid-identity")
        refused = callback(current_client, unsigned_identity)
        assert refused.status_code == 400 and refused.json()["detail"] == "OIDC ID token validation failed"
        assert linked_owner(canonical, "invalid-identity") is None
        assert complete(current_client, legacy_existing) == "local-existing"
        legacy_subject = complete(current_client, legacy_new)
        assert complete(current_client, begin(current_client, code="new-existing", subject="existing")) == "local-existing"
        new_subject = complete(current_client, begin(current_client, code="new-new", subject="new-callback"))
        assert new_subject not in {legacy_subject, "local-existing"}
        assert linked_owner(canonical, "old-callback") == legacy_subject
        assert linked_owner(legacy, "new-callback") == new_subject

        def assert_owner_cannot_change(subject):
            with pytest.raises(OidcIdentityConflict, match="already linked"):
                with store.external_identity_transaction(identity_store, ("default", canonical, subject)) as conn:
                    store.link_external_identity(
                        provider_id="default", issuer=canonical, external_subject=subject,
                        local_subject_id="wrong-account", claims={}, linked_at="2026-01-01T00:00:00Z", connection=conn,
                    )

        assert_owner_cannot_change("new-callback")
        conflict = begin(current_client, code="ownership-conflict", subject="new-callback")
        with monkeypatch.context() as stale_lookup:
            stale_lookup.setattr(api.SECURITY_STATE, "get_external_identity", lambda *_args, **_kwargs: None)
            refused = callback(current_client, conflict)
        assert refused.status_code == 409
        assert refused.json() == {"detail": "Unable to complete sign-in for this account. Contact support to continue."}
        assert "set-cookie" not in refused.headers
        assert callback(current_client, conflict).status_code == 400
        assert linked_owner(canonical, "new-callback") == new_subject

        # An old callback can consume a new transaction and verify PKCE, but RLS still blocks its write.
        wrong_route = begin(current_client, code="wrong-route", subject="legacy-write-denied", issuer=legacy)
        assert callback(previous_client, wrong_route).status_code == 401
        with pytest.raises(psycopg.errors.InsufficientPrivilege, match="row-level security"):
            callback(previous_client, wrong_route, headers=legacy_headers)
        assert wrong_route["code"] not in outstanding_codes
        replay = callback(current_client, wrong_route)
        assert replay.status_code == 400 and "reused OIDC state" in replay.json()["detail"]
        assert linked_owner(legacy, "legacy-write-denied") is None
        assert linked_owner(canonical, "legacy-write-denied") is None

        drain = begin(current_client, code="drain", subject="existing")
        expired = begin(current_client, code="expire", subject="existing")
        with migration_connection() as conn:
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

        assert command("--rollback", revision_id, reason="Return to the previous release")["links"] == 3
        assert command(
            "--rollback", revision_id, reason="Return to the previous release",
        )["status"] == "already-rolled-back"
        with migration_connection() as conn:
            configuration = bind(conn)
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
        assert current_client.post("/api/v1/security/oidc:begin", json={"providerId": "default"}).status_code == 404
        assert callback(current_client, drain).json()["detail"] == "OIDC provider is not configured"
        assert_registry_boundary(revision_id)
        assert_registry_boundary(newer_revision["revisionId"])
        previous_existing = begin(
            previous_client, code="rollback-legacy", subject="existing", issuer=legacy, headers=legacy_headers,
        )
        with pytest.raises(psycopg.errors.InsufficientPrivilege, match="row-level security"):
            callback(previous_client, previous_existing, headers=legacy_headers)
        assert previous_existing["code"] not in outstanding_codes
        assert linked_owner(legacy, "existing") == "local-existing"
        repeated = command("--version", "1.0.2", "--algorithm", "EdDSA")
        revision_id = repeated["revisionId"]
        for expected in ("finalized", "already-finalized"):
            assert command("--finalize", revision_id, reason="Close rollback window")["status"] == expected
        with pytest.raises(SystemExit, match="rollback window has closed"):
            command("--rollback", revision_id, reason="Repeat rollback")
        with migration_connection() as conn:
            assert lookup(conn, legacy, "existing") is None
            assert lookup(conn, canonical, "existing")["local_subject_id"] == "local-existing"

        assert_registry_boundary(revision_id)
        concurrent = [begin(current_client, code=f"race-{suffix}", subject="same-account") for suffix in ("first", "second")]
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda value: callback(current_client, value), concurrent))
        assert all(result.status_code == 200 for result in results)
        assert all("set-cookie" not in result.headers for result in results)
        accounts = {result.json()["identity"]["subjectId"] for result in results}
        assert len(accounts) == 1
        owner, = accounts
        assert complete(current_client, begin(current_client, code="race-retry", subject="same-account")) == owner
        assert linked_owner(canonical, "same-account") == owner
        assert_owner_cannot_change("same-account")
        with psycopg.connect(dsn, row_factory=dict_row) as conn:
            assert conn.execute(
                "SELECT count(*) AS n FROM identity.sessions WHERE subject_id=%s", (owner,),
            ).fetchone()["n"] == 3

        def counts():
            with psycopg.connect(dsn, row_factory=dict_row) as conn:
                return conn.execute(
                    "SELECT (SELECT count(*) FROM identity.subjects) AS subjects, "
                    "(SELECT count(*) FROM identity.sessions) AS sessions"
                ).fetchone()

        previous_counts = counts()
        failed = begin(current_client, code="link-failure", subject="retry-account")

        def reject_link(**_values):
            raise psycopg.OperationalError("internal fixture database detail")

        with monkeypatch.context() as failure:
            failure.setattr(api.SECURITY_STATE, "link_external_identity", reject_link)
            refused = callback(current_client, failed)
        assert refused.status_code == 503
        assert refused.json() == {"detail": "Sign-in is temporarily unavailable. Start a new sign-in and try again."}
        assert "set-cookie" not in refused.headers
        assert counts() == previous_counts
        assert callback(current_client, failed).status_code == 400
        assert linked_owner(canonical, "retry-account") is None
        account = complete(current_client, begin(current_client, code="link-retry", subject="retry-account"))
        assert linked_owner(canonical, "retry-account") == account
        assert counts() == {name: value + 1 for name, value in previous_counts.items()}
