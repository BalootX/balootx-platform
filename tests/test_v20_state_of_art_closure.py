from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
from datetime import datetime
from email.message import Message
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from urllib.request import HTTPSHandler, OpenerDirector
from urllib.response import addinfourl

import httpx
import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from hypothesis import given, strategies as st

from baloot_platform.config import build_bundle, platform_root
from baloot_platform.configuration import SQLiteConfigurationStore, VersionedConfigurationService
from baloot_platform.live import LiveRoom, LiveVersionConflict, SQLiteLiveStore
from baloot_platform.plugins.base import PluginDescriptor
from baloot_platform.plugins.registry import ExternalPluginManifest, PluginPolicy, PluginRegistry
from baloot_platform.security import (
    OidcCallbackService, OidcProviderConfig, PasskeyChallengeService, PkceTransactionStore,
    SQLiteSecurityStateStore, WebSocketTicketService,
)
from baloot_platform.security.oidc import (
    OIDC_PROVIDER_FAMILY,
    http_oidc_runtime,
    load_oidc_providers,
    validate_oidc_provider_profile,
)
from baloot_platform.service.container import build_container
from baloot_platform.service.routes.configuration import build_configuration_router
from baloot_platform.service.routes.security import build_security_router


def test_v20_multi_scope_configuration_precedence_and_freeze(tmp_path: Path):
    root = platform_root()
    profile = yaml.safe_load((root / "configs/configuration/control-plane.yaml").read_text(encoding="utf-8"))
    store = SQLiteConfigurationStore(tmp_path / "configuration.sqlite3")
    service = VersionedConfigurationService(store, root=root, profile=profile)
    baseline = service.bootstrap_packaged_bundle()
    global_bundle = baseline["bundle"]
    components = dict(global_bundle["componentRevisions"])
    tenant = service.create_bundle(bundle_id=global_bundle["bundleId"], version="20.0.1", mode="full",
        component_revisions=components, created_by="ADMIN", scope="tenant", tenant_id="TEN-1", publish=True)
    room = service.create_bundle(bundle_id=global_bundle["bundleId"], version="20.0.2", mode="full",
        component_revisions=components, created_by="ADMIN", scope="room", scope_key="ROOM-1", tenant_id="TEN-1", publish=True)
    shared = service.create_bundle(bundle_id=global_bundle["bundleId"], version="20.0.3", mode="full",
        component_revisions=components, created_by="ADMIN", scope="shared_game", scope_key="BOARD-1", tenant_id="TEN-1", publish=True)
    assert service.resolve(bundle_id=global_bundle["bundleId"], tenant_id="TEN-1")["bundle"]["revisionId"] == tenant["revisionId"]
    assert service.resolve(bundle_id=global_bundle["bundleId"], tenant_id="TEN-1", room_id="ROOM-1")["bundle"]["revisionId"] == room["revisionId"]
    resolved = service.resolve(bundle_id=global_bundle["bundleId"], tenant_id="TEN-1", room_id="ROOM-1", shared_game_id="BOARD-1")
    assert resolved["bundle"]["revisionId"] == shared["revisionId"]
    frozen = service.freeze(bundle_id=global_bundle["bundleId"], tenant_id="TEN-1", shared_game_id="BOARD-1")
    assert frozen["bundleRevisionId"] == shared["revisionId"] and len(frozen["componentRevisionIds"]) == len(components)
    assert store.get_invalidation_epoch() >= 1


def test_v20_live_store_rejects_stale_writes_and_preserves_events(tmp_path: Path):
    store = SQLiteLiveStore(tmp_path / "live.sqlite3")
    assert store.synchronous == "FULL"
    write_connection = store._write_connection
    room = LiveRoom.create(game_id="G-1", owner_subject_id="U-1")
    room.append_event("room_created", subject_id="U-1")
    store.save(room)
    first = store.get(room.room_id); stale = copy.deepcopy(first)
    assert first is not None and stale is not None
    first.append_event("first", subject_id="U-1"); store.save(first, expected_version=first.version)
    stale.append_event("stale", subject_id="U-2")
    with pytest.raises(LiveVersionConflict): store.save(stale, expected_version=stale.version)
    stream = store.events_since(room.room_id, 0)
    assert [item["eventType"] for item in stream["events"]] == ["room_created", "first"]
    assert store.claim_outbox(limit=10)
    assert store._write_connection is write_connection
    store.close()


def test_v20_live_store_durability_mode_is_explicit_and_validated(tmp_path: Path):
    store = SQLiteLiveStore(tmp_path / "live-normal.sqlite3", synchronous="normal")
    try:
        assert store.synchronous == "NORMAL"
        assert store._write_connection.execute("PRAGMA synchronous").fetchone()[0] == 1
    finally:
        store.close()
    with pytest.raises(ValueError, match="synchronous mode"):
        SQLiteLiveStore(tmp_path / "live-invalid.sqlite3", synchronous="off")


def test_v20_websocket_ticket_is_room_bound_and_one_time(tmp_path: Path):
    state = SQLiteSecurityStateStore(tmp_path / "security.sqlite3")
    service = WebSocketTicketService(b"x" * 48, state, lifetime_seconds=30)
    value = service.issue(subject_id="USR-1", room_id="ROOM-1", tenant_id="TEN-1", now=100)
    claims = service.consume(value["ticket"], room_id="ROOM-1", now=101)
    assert claims.subject_id == "USR-1" and claims.tenant_id == "TEN-1"
    with pytest.raises(ValueError): service.consume(value["ticket"], room_id="ROOM-1", now=102)
    other = service.issue(subject_id="USR-1", room_id="ROOM-1", now=100)
    with pytest.raises(ValueError): service.consume(other["ticket"], room_id="ROOM-2", now=101)


def test_v20_oidc_callback_pkce_nonce_and_claim_validation(tmp_path: Path):
    state = SQLiteSecurityStateStore(tmp_path / "security.sqlite3")
    transactions = PkceTransactionStore(backend=state)
    provider = OidcProviderConfig("demo", "https://issuer.example", "https://issuer.example/auth",
        "https://issuer.example/token", "client-1", "https://app.example/callback",
        allowed_algorithms=("RS256",))
    transaction = transactions.create("demo", return_to="/app")
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"id_token": "signed", "access_token": "access"}))
    client = httpx.Client(transport=transport)
    service = OidcCallbackService({"demo": provider}, transactions, http_client=client,
        claims_verifier=lambda token, config, nonce: {"sub": "external-1", "iss": config.issuer, "aud": config.client_id, "nonce": nonce, "iat": 1, "exp": 9999999999})
    result = service.complete(provider_id="demo", state=transaction.state, code="code")
    assert result["externalSubject"] == "external-1" and result["returnTo"] == "/app"
    with pytest.raises(ValueError): service.complete(provider_id="demo", state=transaction.state, code="code")


def _oidc_row(key: str) -> dict:
    return {
        "issuer": f"https://{key}.example/oidc/",
        "authorization_endpoint": f"https://{key}.example/authorize?prompt=login",
        "token_endpoint": f"https://{key}.example/token",
        "client_id": f"{key}-client",
        "redirect_uri": "http://localhost:8080/callback",
        "allowed_algorithms": ["EdDSA"],
    }


def _oidc_profile(providers: dict) -> dict:
    return {
        "schema_version": 1,
        "profile": {"id": "signin", "version": "1.0.0"},
        "change_reason": "Migrate provider onboarding to the registry",
        "providers": providers,
    }


@pytest.fixture
def oidc_configuration(tmp_path):
    root = platform_root()
    config = yaml.safe_load((root / "configs/configuration/control-plane.yaml").read_text(encoding="utf-8"))
    runtime = tmp_path / "data" / "runtime"
    runtime.mkdir(parents=True)
    store = SQLiteConfigurationStore(runtime / "configuration.sqlite3")
    try:
        yield VersionedConfigurationService(store, root=root, profile=config)
    finally:
        store.close()


def test_v20_oidc_registry_drives_api_and_preserves_audit(
    tmp_path, monkeypatch, caplog, oidc_configuration
):
    monkeypatch.setattr(
        "baloot_platform.configuration.store.utc_now", lambda: "2026-01-01T00:00:00+00:00"
    )
    beta = {
        **_oidc_row("beta"),
        "client_secret_env": "BALOOT_OIDC_CLIENT_SECRET_BETA",
        "trusted_additional_audiences": ["trusted-client"],
    }
    payload = _oidc_profile({"alpha": _oidc_row("alpha"), "beta": beta})
    draft = oidc_configuration.create_profile(
        family=OIDC_PROVIDER_FAMILY, payload=payload, created_by="operator-1"
    )
    revision = oidc_configuration.publish_profile(
        draft["revisionId"], expected_etag=draft["etag"], actor_id="operator-2"
    )
    assert revision["payload"]["change_reason"] == payload["change_reason"]
    assert revision["payload"] == validate_oidc_provider_profile(payload).model_dump(mode="json", exclude_none=True)
    audit = oidc_configuration.store.events_with_verification("profile", revision["revisionId"])
    assert audit["verification"]["valid"]
    assert [
        (event["eventType"], event["actorId"], event["createdAt"])
        for event in audit["items"]
    ] == [
        ("created", "operator-1", "2026-01-01T00:00:00+00:00"),
        ("published", "operator-2", "2026-01-01T00:00:00+00:00"),
    ]
    assert audit["items"][0]["payload"]["payloadSha256"] == revision["payloadSha256"]

    monkeypatch.setenv("BALOOT_OIDC_PROVIDERS_PROFILE", "signin")
    monkeypatch.setenv("BALOOT_OIDC_CLIENT_SECRET_BETA", "synthetic-fixture")
    providers = load_oidc_providers(oidc_configuration)
    assert providers["alpha"].trusted_additional_audiences == ()
    assert providers["beta"].trusted_additional_audiences == ("trusted-client",)
    monkeypatch.setenv("BALOOT_OIDC_ISSUER", "https://legacy.example")
    monkeypatch.setenv("BALOOT_OIDC_PROVIDER_ID", "legacy")
    container = build_container(writable_root=tmp_path / "data", database_url="")
    try:
        api = FastAPI()

        @api.middleware("http")
        async def mark_api_key(request: Request, call_next):
            request.state.api_key_valid = request.headers.get("x-api-key") == "synthetic-test-key"
            return await call_next(request)

        api.include_router(
            build_configuration_router(
                container.configuration,
                container.identity,
                container.authorization_policy,
                container.configuration_profile,
                container.performance_profile,
            )
        )
        api.include_router(
            build_security_router(
                container.identity,
                container.access_tokens,
                container.authorization_policy,
                container.passkeys,
                container.websocket_tickets,
                container.security_state,
            )
        )
        caplog.set_level("INFO", logger="baloot_platform.security.oidc")
        with http_oidc_runtime(container.configuration, container.security_state) as runtime, TestClient(api) as client:
            api.state.oidc = runtime
            oidc = runtime.current()
            assert oidc is not None
            assert oidc.providers["alpha"].client_secret is None
            assert oidc.providers["beta"].client_secret == "synthetic-fixture"
            assert "synthetic-fixture" not in repr(oidc.providers)
            assert oidc.providers["beta"].issuer == "https://beta.example/oidc/"
            assert sum("Ignoring legacy OIDC provider metadata" in record.message for record in caplog.records) == 1
            assert any(revision["revisionId"] in record.message for record in caplog.records)
            prefix = "/api/v1/configuration-registry"
            revision_id = revision["revisionId"]
            profile_path = f"{prefix}/profiles/{revision_id}"
            for path in (
                profile_path,
                f"{profile_path}/events",
                f"{prefix}/profiles:resolve?family=oidc_providers&profileId=signin",
                f"{prefix}/profiles:compare?left={revision_id}&right={revision_id}",
            ):
                headers = {"x-api-key": "synthetic-test-key", "x-baloot-role": "player"}
                assert client.get(path, headers=headers).status_code == 403
                headers["x-baloot-role"] = "admin"
                assert client.get(path, headers=headers).status_code == 200
            assert client.get("/api/v1/security/capabilities").json()["oidcProviders"] == ["alpha", "beta"]
            states = {}
            for key in ("alpha", "beta"):
                begin = client.post("/api/v1/security/oidc:begin", json={"providerId": key})
                assert begin.status_code == 200
                states[key] = begin.json()["state"]
                url = urlsplit(begin.json()["authorizationUrl"])
                assert url.hostname == f"{key}.example"
                params = parse_qs(url.query)
                assert params["client_id"] == [f"{key}-client"]
                assert params["prompt"] == ["login"]
                assert params["code_challenge_method"] == ["S256"]
            unknown = client.post("/api/v1/security/oidc:begin", json={"providerId": "legacy"})
            assert unknown.status_code == 404
            updated_alpha = {**_oidc_row("alpha"), "token_endpoint": "https://alpha.example/new-token",
                             "scopes": ["openid", "fixture:read"]}
            replacement = _oidc_profile({"alpha": updated_alpha})
            replacement["profile"]["version"] = "1.0.1"
            replacement_revision = oidc_configuration.create_profile(
                family=OIDC_PROVIDER_FAMILY, payload=replacement, created_by="operator-1", publish=True,
            )
            assert client.get("/api/v1/security/capabilities").json()["oidcProviders"] == ["alpha"]
            stale = client.post("/api/v1/security/oidc:callback", json={
                "providerId": "alpha", "state": states["alpha"], "code": "fixture-code",
            })
            assert stale.status_code == 400 and "provider changed" in stale.json()["detail"]
            begin = client.post("/api/v1/security/oidc:begin", json={"providerId": "alpha"})
            assert parse_qs(urlsplit(begin.json()["authorizationUrl"]).query)["scope"] == ["openid fixture:read"]
            oidc_configuration.retire_profile(
                replacement_revision["revisionId"], expected_etag=replacement_revision["etag"],
                actor_id="operator-1", reason="Disable the registry",
            )
            assert client.get("/api/v1/security/capabilities").json()["oidcProviders"] == []
    finally:
        container.shutdown()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("client_id", None),
        ("client_id", 123),
        ("client_id", " "),
        ("issuer", "https://beta.example/oidc/?"),
        ("token_endpoint", "http://beta.example/token"),
        ("token_endpoint", "https://beta.example:synthetic-fixture/token"),
        ("authorization_endpoint", "https://synthetic-fixture@localhost/authorize"),
        ("redirect_uri", "http://beta.example/callback"),
        ("client_secret", "synthetic-fixture"),
        ("client_secret_env", "UNRELATED_SECRET"),
        ("jwks_uri", "https://beta.example/keys"),
        ("allowed_algorithms", []),
        ("allowed_algorithms", ["HS256"]),
        ("allowed_algorithms", ["EdDSA", "EdDSA"]),
        ("trusted_additional_audiences", [" "]),
        ("trusted_additional_audiences", ["peer", "peer"]),
        ("trusted_additional_audiences", ["beta-client"]),
    ],
)
def test_v20_oidc_registry_rejects_rows_on_write(oidc_configuration, field, value):
    beta = {**_oidc_row("beta"), field: value}
    with pytest.raises(ValueError, match="OIDC provider 'beta'") as exc:
        oidc_configuration.create_profile(
            family=OIDC_PROVIDER_FAMILY,
            payload=_oidc_profile({"alpha": _oidc_row("alpha"), "beta": beta}),
            created_by="operator-1",
        )
    assert "synthetic-fixture" not in str(exc.value)
    assert oidc_configuration.store.list_profiles(family=OIDC_PROVIDER_FAMILY) == []


def test_v20_oidc_schema_and_writes_agree_on_profile_version(oidc_configuration):
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(oidc_configuration.profile_schema(OIDC_PROVIDER_FAMILY))
    payload = _oidc_profile({})
    assert validator.is_valid(payload)
    payload["profile"]["version"] = "release"
    assert not validator.is_valid(payload)
    with pytest.raises(ValueError, match=r"OIDC registry profile\.version"):
        oidc_configuration.create_profile(
            family=OIDC_PROVIDER_FAMILY, payload=payload, created_by="operator-1",
        )


def test_v20_oidc_registry_fails_closed_at_startup(tmp_path, monkeypatch, oidc_configuration):
    monkeypatch.delenv("BALOOT_OIDC_PROVIDERS_PROFILE", raising=False)
    monkeypatch.delenv("BALOOT_OIDC_ISSUER", raising=False)
    assert load_oidc_providers(oidc_configuration) == {}
    monkeypatch.setenv("BALOOT_OIDC_ISSUER", "https://legacy.example")
    with pytest.raises(ValueError, match="migrate legacy OIDC"):
        load_oidc_providers(oidc_configuration)

    alpha = {**_oidc_row("alpha"), "client_secret_env": "BALOOT_OIDC_CLIENT_SECRET_ALPHA"}
    draft = oidc_configuration.create_profile(
        family=OIDC_PROVIDER_FAMILY, payload=_oidc_profile({"alpha": alpha}), created_by="operator-1"
    )
    monkeypatch.setenv("BALOOT_OIDC_PROVIDERS_PROFILE", "signin")
    with pytest.raises(ValueError, match="published OIDC provider profile"):
        load_oidc_providers(oidc_configuration)
    oidc_configuration.publish_profile(
        draft["revisionId"], expected_etag=draft["etag"], actor_id="operator-1"
    )
    monkeypatch.delenv("BALOOT_OIDC_CLIENT_SECRET_ALPHA", raising=False)
    worker = build_container(writable_root=tmp_path / "data", database_url="")
    worker.shutdown()
    with pytest.raises(ValueError, match="OIDC provider 'alpha': missing deployment secret"):
        load_oidc_providers(oidc_configuration)


def test_v20_oidc_startup_follows_current_published_revision(monkeypatch, oidc_configuration):
    monkeypatch.setenv("BALOOT_OIDC_PROVIDERS_PROFILE", "signin")
    previous = oidc_configuration.create_profile(
        family=OIDC_PROVIDER_FAMILY, payload=_oidc_profile({"alpha": _oidc_row("alpha")}),
        created_by="operator-1", publish=True,
    )
    assert set(load_oidc_providers(oidc_configuration)) == {"alpha"}
    revised = _oidc_profile({"beta": _oidc_row("beta")})
    revised["profile"]["version"] = "1.0.1"
    current = oidc_configuration.create_profile(
        family=OIDC_PROVIDER_FAMILY, payload=revised, created_by="operator-1",
        supersedes_revision_id=previous["revisionId"], publish=True,
    )
    assert set(load_oidc_providers(oidc_configuration)) == {"beta"}
    oidc_configuration.retire_profile(
        current["revisionId"], expected_etag=current["etag"], actor_id="operator-1",
        reason="Retire current sign-in profile",
    )
    with pytest.raises(ValueError, match="no active published"):
        load_oidc_providers(oidc_configuration)


def test_v20_oidc_registry_refuses_rollout_conditions(monkeypatch, oidc_configuration):
    monkeypatch.setenv("BALOOT_OIDC_PROVIDERS_PROFILE", "signin")
    oidc_configuration.create_profile(
        family=OIDC_PROVIDER_FAMILY, payload=_oidc_profile({"alpha": _oidc_row("alpha")}),
        created_by="operator-1", effective_from="2000-01-01T00:00:00+00:00", publish=True,
    )
    with pytest.raises(ValueError, match="without rollout conditions"):
        load_oidc_providers(oidc_configuration)


@given(st.lists(st.from_regex(r"[a-z][a-z0-9-]{0,10}", fullmatch=True), max_size=4, unique=True))
def test_v20_oidc_registry_preserves_provider_keys(keys):
    validated = validate_oidc_provider_profile(
        _oidc_profile({key: _oidc_row(key) for key in keys})
    )
    assert list(validated.providers) == keys


@pytest.mark.parametrize("algorithm", ["EdDSA", "RS256"])
def test_v20_oidc_callback_uses_real_signature_and_discovered_keys(monkeypatch, algorithm):
    import jwt

    provider = OidcProviderConfig(
        "demo",
        "https://issuer.example/oidc/",
        "https://issuer.example/authorize",
        "https://issuer.example/token",
        "demo-client",
        "https://app.example/callback",
        allowed_algorithms=(algorithm,),
        client_secret="synthetic-client-fixture",
        trusted_additional_audiences=("trusted-client",),
    )
    timestamp = 1_767_225_600

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls.fromtimestamp(timestamp, tz)

    monkeypatch.setattr(jwt.api_jwt, "datetime", FixedDateTime)
    if algorithm == "EdDSA":
        private_key = Ed25519PrivateKey.from_private_bytes(b"a" * 32)
        other_key = Ed25519PrivateKey.from_private_bytes(b"b" * 32)
        jwk = jwt.algorithms.OKPAlgorithm.to_jwk(private_key.public_key())
    else:
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        jwk = jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key())
    ec_key = ec.derive_private_key(7, ec.SECP256R1())
    keys = [
        {**json.loads(jwk), "kid": "signing"},
        {**json.loads(jwt.algorithms.ECAlgorithm.to_jwk(ec_key.public_key())), "kid": "ec"},
    ]
    discovery_url = "https://issuer.example/oidc/.well-known/openid-configuration"
    jwks_url = "https://keys.example/oidc/public"
    rotation_clock = 0.0
    monkeypatch.setattr("baloot_platform.security.oidc.monotonic", lambda: rotation_clock)
    discovery_reads = []
    jwks_reads = []
    metadata = {"issuer": provider.issuer, "jwks_uri": jwks_url}
    token = ""

    def fetch_jwks(_opener, request, timeout=None):
        jwks_reads.append(request.full_url)
        assert request.full_url == metadata["jwks_uri"]
        return BytesIO(json.dumps({"keys": keys}).encode())

    monkeypatch.setattr(OpenerDirector, "open", fetch_jwks)

    def exchange(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            discovery_reads.append(str(request.url))
            assert str(request.url) == discovery_url
            return httpx.Response(200, json=metadata)
        assert str(request.url) == provider.token_endpoint
        params = parse_qs(request.content.decode())
        assert params["client_id"] == [provider.client_id]
        assert params["client_secret"] == ["synthetic-client-fixture"]
        return httpx.Response(200, json={"id_token": token})

    transactions = PkceTransactionStore()
    with httpx.Client(transport=httpx.MockTransport(exchange)) as client:
        service = OidcCallbackService({"demo": provider}, transactions, http_client=client)

        def complete(*, key=private_key, alg=algorithm, kid="signing", **overrides):
            nonlocal token
            transaction = transactions.create("demo")
            claims = {
                "iss": provider.issuer,
                "aud": provider.client_id,
                "sub": "external-1",
                "nonce": transaction.nonce,
                "iat": timestamp - 1,
                "exp": timestamp + 60,
                **overrides,
            }
            token = jwt.encode(claims, key, algorithm=alg, headers={"kid": kid} if kid is not None else {})
            return service.complete(provider_id="demo", state=transaction.state, code="code")

        with pytest.raises(ValueError, match="signing algorithm is not allowed"):
            complete(key=ec_key, alg="ES256", kid="ec")
        with pytest.raises(ValueError, match="signing algorithm is not allowed"):
            complete(key=None, alg="none")
        with pytest.raises(ValueError, match="signing key ID is missing"):
            complete(kid=None)
        assert discovery_reads == jwks_reads == []
        metadata["issuer"] = provider.issuer.rstrip("/")
        with pytest.raises(ValueError, match="discovery issuer mismatch"):
            complete()
        metadata["issuer"] = provider.issuer
        metadata["jwks_uri"] = "http://keys.example/oidc/public"
        with pytest.raises(ValueError, match="absolute HTTPS URL"):
            complete()
        metadata["jwks_uri"] = jwks_url
        assert jwks_reads == []
        discovery_reads.clear()
        assert complete()["externalSubject"] == "external-1"
        assert complete(azp=provider.client_id)["externalSubject"] == "external-1"
        trusted_audiences = [provider.client_id, "trusted-client"]
        assert complete(aud=trusted_audiences, azp=provider.client_id)["externalSubject"] == "external-1"
        with pytest.raises(ValueError, match="untrusted audience"):
            complete(aud=[provider.client_id, "other-client"], azp=provider.client_id)
        with pytest.raises(ValueError, match="authorized party mismatch"):
            complete(aud=trusted_audiences)
        with pytest.raises(ValueError, match="authorized party mismatch"):
            complete(aud=trusted_audiences, azp="trusted-client")
        with pytest.raises(ValueError, match="authorized party mismatch"):
            complete(azp="other-client")
        with pytest.raises(ValueError, match="authorized party mismatch"):
            complete(azp=None)
        assert complete(iat=timestamp + provider.clock_skew_seconds)["issuer"] == provider.issuer
        assert discovery_reads == [discovery_url]
        assert jwks_reads == [jwks_url]
        for overrides in (
            {"kid": "unknown"},
            {"iss": provider.issuer.rstrip("/")},
            {"aud": "different-client"},
            {"exp": timestamp - provider.clock_skew_seconds - 1},
            {"iat": timestamp + provider.clock_skew_seconds + 1},
            {"key": other_key},
        ):
            with pytest.raises(ValueError, match="OIDC ID token validation failed"):
                complete(**overrides)
        assert discovery_reads == [discovery_url]
        metadata["jwks_uri"] = "https://keys.example/oidc/rotated"
        rotation_clock += provider.discovery_cache_seconds
        assert complete()["externalSubject"] == "external-1"
        assert discovery_reads == [discovery_url, discovery_url]
        assert jwks_reads[-1] == metadata["jwks_uri"]


def test_v20_oidc_jwks_transport_rejects_redirect(monkeypatch):
    import jwt

    requests = []
    jwks_url = "https://keys.example/oidc/public"

    def redirect(_handler, request):
        requests.append(request.full_url)
        headers = Message()
        headers["Location"] = "http://keys.example/other"
        response = addinfourl(BytesIO(), headers, request.full_url, 302)
        response.msg = "Found"
        return response

    monkeypatch.setattr(HTTPSHandler, "https_open", redirect)
    with pytest.raises(jwt.exceptions.PyJWKClientConnectionError):
        jwt.PyJWKClient(jwks_url).get_signing_key("signing")
    assert requests == [jwks_url]


def test_v20_passkey_state_is_durable_and_replay_safe(tmp_path: Path):
    state = SQLiteSecurityStateStore(tmp_path / "security.sqlite3")
    passkeys = PasskeyChallengeService(rp_id="localhost", origin="https://localhost", backend=state)
    begin = passkeys.begin(subject_id="USR-1", purpose="register")
    item = passkeys.register_verified(challenge=begin["challenge"], subject_id="USR-1", credential_id="cred-1", public_key_cose=b"key")
    assert item.credential_id == "cred-1" and passkeys.list_credentials("USR-1")[0]["credentialId"] == "cred-1"
    with pytest.raises(ValueError): passkeys.register_verified(challenge=begin["challenge"], subject_id="USR-1", credential_id="cred-2", public_key_cose=b"key")


def test_v20_external_plugin_manifest_verifies_hash_and_signature(tmp_path: Path):
    artifact = tmp_path / "plugin.bin"; artifact.write_bytes(b"plugin-v20")
    descriptor = PluginDescriptor("external-demo", "analysis", "1.0.0", "demo", api_version="1",
        host_api_range=">=1,<2", profile_schema_range=">=1,<2", isolation="process",
        artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(), signed=True, trust_class="external")
    key = Ed25519PrivateKey.generate(); public = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    policy = PluginPolicy.from_profile({
        "host_api_version": "1", "profile_schema_version": "1",
        "allowed_isolation": ["process", "wasm_component"],
        "allowed_permissions": ["none"], "require_signed_external": True,
        "require_content_addressed_external": True,
        "maximum_timeout_seconds": 30, "maximum_response_bytes": 1048576,
        "maximum_memory_mb": 512, "maximum_cpu_seconds": 10,
        "maximum_stderr_bytes": 65536,
        "trusted_signers": {
            "reviewer": base64.b64encode(public).decode()
        },
    })
    unsigned = {"descriptor": descriptor.to_dict(), "artifact": artifact.name, "command": [str(artifact)], "signerId": "reviewer"}
    temp = tmp_path / "temp.json"; temp.write_text(json.dumps(unsigned), encoding="utf-8")
    parsed = ExternalPluginManifest.load(temp); signature = base64.b64encode(key.sign(parsed.signed_payload())).decode()
    unsigned["signature"] = signature; manifest = tmp_path / "baloot-plugin.json"; manifest.write_text(json.dumps(unsigned), encoding="utf-8")
    registry = PluginRegistry(policy=policy); assert registry.register_manifest(manifest) == "external-demo"
    assert registry.list()[0]["external"] is True
    artifact.write_bytes(b"tampered")
    with pytest.raises(PermissionError): PluginRegistry(policy=policy).register_manifest(manifest)


def test_v20_native_apply_undo_restores_exact_state():
    if os.name == "nt":
        pytest.skip("ELF native parity is verified by the WSL release gate on Windows hosts")
    from baloot_platform.native import NativeAccelerator
    root = platform_root()
    bundle = build_bundle(root)
    episode = __import__('baloot_platform.core.episode_io', fromlist=['load_episode_spec']).load_episode_spec(
        root / 'examples/v7-outcome-flip-source.json', bundle.rules
    )
    state = episode.state_before_pause.clone()
    native = NativeAccelerator(bundle.rules, root / 'build/native-release/libbaloot_native.so')
    legal = native.legal_moves(state)
    assert legal
    proof = native.apply_undo_roundtrip(state, legal[0])
    assert proof['restored'] is True and proof['beforeHash'] == proof['afterHash']


def test_v20_security_contract_removes_query_bearer_and_forces_rls():
    root = platform_root()
    live_route = (root / 'src/baloot_platform/service/routes/live.py').read_text(encoding='utf-8')
    assert 'ticket = websocket.query_params.get("ticket")' in live_route
    assert 'or websocket.query_params.get("access_token")' in live_route
    assert 'access_token = websocket.query_params.get' not in live_route
    migration = (root / 'migrations/012_v20_security_configuration_live.sql').read_text(encoding='utf-8').upper()
    assert 'FORCE ROW LEVEL SECURITY' in migration
    context = (root / 'src/baloot_platform/storage/request_context.py').read_text(encoding='utf-8')
    for setting in ('app.tenant_id', 'app.subject_id', 'app.role'):
        assert setting in context
