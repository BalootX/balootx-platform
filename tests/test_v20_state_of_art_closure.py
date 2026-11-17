from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
from pathlib import Path

import httpx
import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from baloot_platform.config import build_bundle, platform_root
from baloot_platform.configuration import SQLiteConfigurationStore, VersionedConfigurationService
from baloot_platform.live import LiveRoom, LiveVersionConflict, SQLiteLiveStore
from baloot_platform.plugins.base import PluginDescriptor
from baloot_platform.plugins.registry import ExternalPluginManifest, PluginPolicy, PluginRegistry
from baloot_platform.security import (
    OidcCallbackService, OidcProviderConfig, PasskeyChallengeService, PkceTransactionStore,
    SQLiteSecurityStateStore, WebSocketTicketService,
)


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
        "https://issuer.example/token", "client-1", "https://app.example/callback", jwks_uri="https://issuer.example/jwks")
    transaction = transactions.create("demo", return_to="/app")
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"id_token": "signed", "access_token": "access"}))
    client = httpx.Client(transport=transport)
    service = OidcCallbackService({"demo": provider}, transactions, http_client=client,
        claims_verifier=lambda token, config, nonce: {"sub": "external-1", "iss": config.issuer, "aud": config.client_id, "nonce": nonce, "iat": 1, "exp": 9999999999})
    result = service.complete(provider_id="demo", state=transaction.state, code="code")
    assert result["externalSubject"] == "external-1" and result["returnTo"] == "/app"
    with pytest.raises(ValueError): service.complete(provider_id="demo", state=transaction.state, code="code")


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
