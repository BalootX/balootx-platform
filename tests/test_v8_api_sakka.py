from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from security_probe_helpers import security_probe_app

from baloot_platform.service.api import app


def _guest(client: TestClient, name: str) -> dict:
    response = client.post('/api/v1/auth/guest', json={'displayName': name})
    assert response.status_code in {200, 201}, response.text
    return response.json()


def test_player_app_and_guest_first_sakka_round_flow():
    with TestClient(app) as client:
        page = client.get('/play')
        css = client.get('/play-assets/app.css')
        script = client.get('/play-assets/app.js')
        assert page.status_code == css.status_code == script.status_code == 200
        assert page.headers['cache-control'] == 'no-store'
        player_root = Path('src/baloot_platform/service/static/player')
        digest = hashlib.sha256()
        for asset_name in ('app.css', 'app.js'):
            digest.update(asset_name.encode('utf-8'))
            digest.update((player_root / asset_name).read_bytes())
        asset_version = digest.hexdigest()[:12]
        assert f'/play-assets/app.css?v={asset_version}' in page.text
        assert f'/play-assets/app.js?v={asset_version}' in page.text
        assert 'match-table' in page.text
        assert '<div class="hero-card red"' not in page.text
        assert '<img class="hero-card"' in page.text
        for card_asset in (
            '/card-assets/A_hearts.svg',
            '/card-assets/J_spades.svg',
            '/card-assets/10_diamonds.svg',
        ):
            assert card_asset in page.text
            assert client.get(card_asset).status_code == 200
        assert 'لنا' in page.text and 'لهم' in page.text
        assert '{{' not in page.text and '}}' not in page.text
        assert 'id="player-copy"' in page.text
        assert 'Missing governed player copy' in script.text
        assert not any('\u0600' <= character <= '\u06ff' for character in script.text)
        assert 'touch-action:none' in css.text
        assert 'grid-template-areas:"exit code contract score"' in css.text
        assert '.game-view:not(.play-phase) .playing-card:not(.legal)' in css.text
        assert 'onCardPointerMove' in script.text
        assert 'animateCardToTable' in script.text
        assert 'sourceRect' in script.text
        assert 'collectTrick' in script.text
        assert 'cardFromInternalCode' in script.text
        assert 'audit-card-face' in script.text and 'audit-card-face' in css.text
        guest = _guest(client, f'ضيف {uuid.uuid4().hex[:6]}')
        assert guest['identity']['guest'] is True

        board = client.post('/api/v1/sakka/boards', json={
            'source': 'new', 'seed': 88321, 'dealer': 'east',
            'rulesProfile': 'sun_only', 'observerSeat': 'south',
            'scope': 'round', 'visibility': 'unlisted',
            'label': 'صكّة اختبار واجهة',
        })
        assert board.status_code == 201, board.text
        board_data = board.json()
        assert board_data['scope'] == 'round'
        assert board_data['shareCode']
        verification = client.get(f"/api/v1/sakka/boards/{board_data['boardId']}/verification")
        assert verification.status_code == 200
        assert verification.json()['valid'] is True
        assert verification.json()['signatureValid'] is True

        room = client.post('/api/v1/sakka/rooms', json={
            'boardIdentifier': board_data['boardId'], 'mode': 'async',
            'maxPlayers': 8, 'allowGuests': True,
        })
        assert room.status_code == 201, room.text
        room_data = room.json()['room']
        assert len(room_data['roomCode']) >= 8

        run = client.post(f"/api/v1/sakka/rooms/{room_data['roomCode']}/runs")
        assert run.status_code == 201, run.text
        run_data = run.json()
        assert run_data['run']['status'] in {'active', 'completed'}
        assert run_data['run']['comparisonKey']
        assert run_data['playerNames']['south'] == guest['identity']['displayName']
        assert all(run_data['playerNames'].get(seat) for seat in ('north', 'west', 'south', 'east'))

        reopened = client.get(f"/api/v1/sakka/runs/{run_data['run']['runId']}")
        assert reopened.status_code == 200
        assert reopened.json()['run']['stateHash'] == run_data['run']['stateHash']
        run_verification = client.get(f"/api/v1/sakka/runs/{run_data['run']['runId']}/verification")
        assert run_verification.status_code == 200
        assert run_verification.json()['valid'] is True


def test_two_guests_receive_identical_sakka_start_state_and_events():
    creator = TestClient(app)
    player = TestClient(app)
    _guest(creator, f'منشئ {uuid.uuid4().hex[:5]}')
    _guest(player, f'لاعب {uuid.uuid4().hex[:5]}')

    board = creator.post('/api/v1/sakka/boards', json={
        'seed': 99217, 'dealer': 'west', 'rulesProfile': 'hokum_only',
        'observerSeat': 'south', 'scope': 'round', 'visibility': 'unlisted',
    }).json()
    room = creator.post('/api/v1/sakka/rooms', json={
        'boardIdentifier': board['boardId'], 'mode': 'async', 'allowGuests': True,
    }).json()['room']

    first = creator.post(f"/api/v1/sakka/rooms/{room['roomCode']}/runs").json()
    second_response = player.post(f"/api/v1/sakka/rooms/{room['roomCode']}/runs")
    assert second_response.status_code == 201, second_response.text
    second = second_response.json()
    assert first['run']['stateHash'] == second['run']['stateHash']
    assert first['legalActions'] == second['legalActions']
    assert first['state']['viewerSeat'] == second['state']['viewerSeat'] == 'south'

    events = creator.get(f"/api/v1/sakka/rooms/{room['roomCode']}/events").json()
    kinds = {item['eventType'] for item in events['events']}
    assert {'room_created', 'member_joined', 'run_started'} <= kinds

    leaderboard = creator.get(f"/api/v1/sakka/rooms/{room['roomCode']}/leaderboard")
    assert leaderboard.status_code == 200


def test_guest_can_upgrade_in_place_without_losing_identity():
    with TestClient(app) as client:
        guest = _guest(client, f'ضيف مستمر {uuid.uuid4().hex[:5]}')
        subject_id = guest['identity']['subjectId']
        username = f'lasting_{uuid.uuid4().hex[:8]}'
        registered = client.post('/api/v1/auth/register', json={
            'username': username, 'password': 'correct-horse', 'displayName': 'لاعب مستمر',
        })
        assert registered.status_code == 201, registered.text
        assert registered.json()['identity']['subjectId'] == subject_id
        assert registered.json()['identity']['authenticated'] is True
        current = client.get('/api/v1/auth/me').json()
        assert current['identity']['subjectId'] == subject_id


_SERVICE_KEY = "synthetic-service-key"
_KEY_SERVICE = "SVC-A748F51287ECF77FE4FA"
# The SHA-256 subject IDs used by the pre-upgrade server for these fixtures.
_LEGACY_SERVICE = "SVC-FC90545E2119AD22B93F"
_BOARDS = "/api/v1/sakka/boards"
_ROOMS = "/api/v1/sakka/rooms"
_RUNS = "/api/v1/sakka/runs"


_CONFIG_READ_ROLES = ("guest", "player", "expert", "referee")


def _oidc_test_payload(profile_id: str, version: str = "1.0.0") -> dict:
    return {
        "schema_version": 1,
        "profile": {"id": profile_id, "version": version},
        "change_reason": "Exercise registry authorization",
        "providers": {},
    }


def test_http_oidc_profile_lifecycle_requires_config_admin(monkeypatch):
    monkeypatch.setenv("BALOOT_API_KEY", _SERVICE_KEY)
    monkeypatch.setenv("BALOOT_RATE_LIMIT_PER_MINUTE", "0")
    prefix = "/api/v1/configuration-registry"

    with TestClient(app) as client:
        def check(method, path, role, *, body=None, status=200):
            for denied in _CONFIG_READ_ROLES:
                response = client.request(method, prefix + path, json=body, headers={
                    "x-api-key": _SERVICE_KEY, "x-baloot-role": denied,
                })
                assert response.status_code == (404 if method == "GET" else 403), response.text
            response = client.request(method, prefix + path, json=body, headers={
                "x-api-key": "wrong-fixture-key", "x-baloot-role": role,
            })
            assert response.status_code == 401, response.text
            response = client.request(method, prefix + path, json=body, headers={
                "x-api-key": _SERVICE_KEY, "x-baloot-role": role,
            })
            assert response.status_code == status, response.text
            return response.json()

        for role in ("admin", "service"):
            profile_id = f"http-oidc-{role}"
            draft = check("POST", "/profiles", role, status=201, body={
                "family": "oidc_providers", "payload": _oidc_test_payload(profile_id),
            })
            path = f"/profiles/{draft['revisionId']}"
            updated = check("PATCH", path, role, body={
                "payload": _oidc_test_payload(profile_id, "1.0.1"), "expectedEtag": draft["etag"],
            })
            published = check("POST", path + ":publish", role, body={"expectedEtag": updated["etag"]})
            assert check("GET", f"/profiles:resolve?family=oidc_providers&profileId={profile_id}", role)["status"] == "published"
            revision_id = published["revisionId"]
            check("GET", f"/profiles:compare?left={revision_id}&right={revision_id}", role)
            rejected = check("POST", "/profiles", role, status=201, body={
                "family": "oidc_providers", "payload": _oidc_test_payload(profile_id, "1.0.2"),
            })
            check("POST", f"/profiles/{rejected['revisionId']}:reject", role, body={
                "expectedEtag": rejected["etag"], "reason": "Reject draft",
            })
            check("POST", path + ":retire", role, body={
                "expectedEtag": published["etag"], "reason": "Retire revision",
            })


def test_http_oidc_registry_key_only_changes_record_the_service_actor(monkeypatch):
    monkeypatch.setenv("BALOOT_API_KEY", _SERVICE_KEY)
    monkeypatch.setenv("BALOOT_RATE_LIMIT_PER_MINUTE", "0")
    prefix = "/api/v1/configuration-registry/profiles"
    headers = {"x-api-key": _SERVICE_KEY}

    with TestClient(app) as client:
        created = client.post(prefix, headers=headers, json={
            "family": "oidc_providers", "payload": _oidc_test_payload("key-only-audit-fixture"),
        })
        assert created.status_code == 201, created.text
        draft = created.json()
        assert draft["createdBy"] == _KEY_SERVICE
        revision = f"{prefix}/{draft['revisionId']}"
        published = client.post(revision + ":publish", headers=headers, json={"expectedEtag": draft["etag"]})
        assert published.status_code == 200, published.text
        assert published.json()["publishedBy"] == _KEY_SERVICE
        response = client.get(revision + "/events", headers=headers)
        assert response.status_code == 200, response.text
        audit = response.json()
        assert audit["verification"]["valid"]
        assert [(event["eventType"], event["actorId"]) for event in audit["items"]] == [
            ("created", _KEY_SERVICE), ("published", _KEY_SERVICE),
        ]


def test_http_configuration_revision_readers_share_sqlite_visibility(monkeypatch):
    from baloot_platform.configuration import SQLiteConfigurationStore
    from baloot_platform.security import Scope, SignedAccessTokenService
    from baloot_platform.service import api, mcp_server

    assert isinstance(api.CONFIGURATION.store, SQLiteConfigurationStore)
    monkeypatch.setenv("BALOOT_API_KEY", _SERVICE_KEY)
    monkeypatch.setenv("BALOOT_RATE_LIMIT_PER_MINUTE", "0")
    monkeypatch.setenv("BALOOT_ALLOW_TRUSTED_CONTEXT", "true")
    fixture_secret = "synthetic-mcp-reader-secret-" * 2
    tokens = SignedAccessTokenService(fixture_secret.encode())
    monkeypatch.setenv("BALOOT_ACCESS_TOKEN_SECRET", fixture_secret)
    monkeypatch.setenv("BALOOT_MCP_TOKEN_AUDIENCE", tokens.audience)
    monkeypatch.setenv("BALOOT_TOKEN_ISSUER", tokens.issuer)
    monkeypatch.delenv("BALOOT_MCP_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("BALOOT_API_URL", "http://testserver")
    monkeypatch.setattr(api, "ACCESS_TOKENS", tokens)
    v1 = "/api/v1/configuration-registry/profiles"
    v2 = "/api/v2/configuration/profiles"
    profile_id = "oidc-reader-fixture"

    def headers(role):
        return {"x-api-key": _SERVICE_KEY, "x-baloot-role": role}

    with TestClient(app) as client, TestClient(mcp_server.create_http_app()) as mcp:
        monkeypatch.setitem(mcp_server._API_CLIENTS, "http://testserver", client)
        created = client.post(v1, headers=headers("admin"), json={
            "family": "oidc_providers", "payload": _oidc_test_payload(profile_id), "publish": True,
        })
        assert created.status_code == 201, created.text
        revision = created.json()["revisionId"]
        public = api.CONFIGURATION.store.list_profiles(scope="global", limit=1)[0].revision_id
        operator_listings = (v2, f"{v2}?family=oidc_providers", "/api/v2/config-revisions?kind=profile")
        details = (
            f"{v1}/{revision}", f"{v1}/{revision}/events",
            f"{v1}:compare?left={public}&right={revision}",
            f"{v1}:resolve?family=oidc_providers&profileId={profile_id}",
        )
        for role in ("admin", "service", *_CONFIG_READ_ROLES):
            authorized = role in {"admin", "service"}
            response = client.get(v1, headers=headers(role))
            assert response.status_code == 200, response.text
            ids = {item["revisionId"] for item in response.json()["items"]}
            assert public in ids
            assert (revision in ids) is authorized
            response = client.get(f"{v1}?family=oidc_providers", headers=headers(role))
            assert response.status_code == 200, response.text
            items = response.json()["items"]
            assert any(item["revisionId"] == revision for item in items) is authorized
            if not authorized:
                assert items == []
            for url in operator_listings:
                response = client.get(url, headers=headers(role))
                assert response.status_code == (200 if authorized else 403), response.text
                if authorized:
                    assert revision in {item["revisionId"] for item in response.json()["items"]}
            response = client.get(f"{v2}/{revision}?includePayload=false", headers=headers(role))
            assert response.status_code == (200 if authorized else 403), response.text
            for url in details:
                assert client.get(url, headers=headers(role)).status_code == (200 if authorized else 404)
            for base in (v1, v2):
                response = client.post(f"{base}/{revision}:validate", headers=headers(role))
                denied_status = 404 if base == v1 else 403
                assert response.status_code == (200 if authorized else denied_status), response.text

            scopes = (Scope.READ, Scope.CONFIG_ADMIN) if authorized else (Scope.READ,)
            token = tokens.issue(subject_id="fixture-reader", role=role, scopes=scopes)
            for tool, args in (
                ("list_configuration_profiles", {"family": "oidc_providers"}),
                ("resolve_configuration_profile", {"family": "oidc_providers", "profile_id": profile_id}),
                ("compare_configuration_profiles", {"left": public, "right": revision}),
            ):
                response = mcp.post("/mcp", headers={"authorization": f"Bearer {token}"}, json={
                    "jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": tool, "arguments": args},
                })
                assert response.status_code == 200
                result = response.json()["result"]
                if tool == "list_configuration_profiles":
                    assert not result["isError"]
                    items = result["structuredContent"]["items"]
                    assert any(item["revisionId"] == revision for item in items) is authorized
                else:
                    assert result["isError"] is not authorized
                    if not authorized:
                        assert "404 Not Found" in result["content"][0]["text"]

        _guest(client, "Fixture reader")
        response = client.get(operator_listings[-1], headers={"x-baloot-role": "admin"})
        assert response.status_code == 403, response.text
        response = client.get(v1, headers={"x-baloot-role": "admin"})
        assert response.status_code == 200, response.text
        assert all(item["revisionId"] != revision for item in response.json()["items"])


def test_http_database_role_ignores_keyless_role_header(monkeypatch):
    from baloot_platform.storage.request_context import CURRENT_DATABASE_IDENTITY

    probe = security_probe_app()

    @probe.get("/healthz")
    def database_role():
        return {"role": CURRENT_DATABASE_IDENTITY.get().role}

    monkeypatch.delenv("BALOOT_API_KEY", raising=False)
    with TestClient(probe) as client:
        headers = {"x-baloot-role": "admin"}
        assert client.get("/healthz", headers=headers).json() == {"role": "anonymous"}
        monkeypatch.setenv("BALOOT_API_KEY", _SERVICE_KEY)
        assert client.get("/healthz", headers={**headers, "x-api-key": "wrong-fixture-key"}).json() == {"role": "anonymous"}
        assert client.get("/healthz", headers={**headers, "x-api-key": _SERVICE_KEY}).json() == {"role": "admin"}

@pytest.fixture
def legacy_service_records(monkeypatch):
    monkeypatch.setenv("BALOOT_API_KEY", _SERVICE_KEY)
    monkeypatch.delenv("BALOOT_API_SUBJECT_ID", raising=False)
    with TestClient(app) as client:
        headers = {"x-api-key": _SERVICE_KEY, "x-baloot-subject-id": _LEGACY_SERVICE}
        board = client.post(_BOARDS, json={"visibility": "private", "seed": 6029}, headers=headers)
        assert board.status_code == 201, board.text
        board = board.json()
        assert board["createdBy"] == _LEGACY_SERVICE
        room = client.post(_ROOMS, json={"boardIdentifier": board["boardId"]}, headers=headers)
        assert room.status_code == 201, room.text
        code = room.json()["room"]["roomCode"]
        run = client.post(f"{_ROOMS}/{code}/runs", headers=headers)
        assert run.status_code == 201, run.text
        assert run.json()["run"]["subjectId"] == _LEGACY_SERVICE
        return board, code, run.json()["run"]


def _grpc_probe(tmp_path, monkeypatch, probe, *, stub_type=None):
    from baloot_platform.generated import sakka_pb2_grpc
    from service_test_client import grpc_probe

    grpc_probe(tmp_path, monkeypatch, probe, stub_type=stub_type or sakka_pb2_grpc.SakkaPlatformStub)


def test_grpc_oidc_profile_lifecycle_requires_config_admin(tmp_path, monkeypatch):
    import asyncio

    import grpc
    from service_test_client import grpc_probe

    from baloot_platform.generated import configuration_pb2, configuration_pb2_grpc
    from baloot_platform.security import Scope
    from baloot_platform.service import grpc_service
    from baloot_platform.service.api import ACCESS_TOKENS, CONFIGURATION
    from baloot_platform.service.authorization import DEFAULT_TENANT
    from baloot_platform.storage import request_context

    monkeypatch.delenv("BALOOT_API_KEY", raising=False)
    database_identity = request_context.CURRENT_DATABASE_IDENTITY
    contexts, resets = [], []
    audit_expectations = []
    completions = asyncio.Queue()

    class ObserveServerCompletion(grpc.aio.ServerInterceptor):
        async def intercept_service(self, continuation, details):
            handler = await continuation(details)

            async def invoke(request, context):
                finished = asyncio.Event()
                completions.put_nowait(finished)
                try:
                    return await handler.unary_unary(request, context)
                finally:
                    finished.set()

            return grpc.unary_unary_rpc_method_handler(
                invoke, request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )

    class TrackingContext:
        def get(self):
            return database_identity.get()

        def set(self, identity):
            token = database_identity.set(identity)
            contexts.append((identity, token))
            return token

        def reset(self, token):
            database_identity.reset(token)
            resets.append(token)

    monkeypatch.setattr(grpc_service, "CURRENT_DATABASE_IDENTITY", TrackingContext())

    async def probe(grpc, stub):
        async def call(rpc, request, metadata=()):
            try:
                return await rpc(request, metadata=metadata)
            finally:
                # Error status can arrive before the server unwinds the handler.
                await completions.get_nowait().wait()

        request = configuration_pb2.ProfileDraft(family="oidc_providers", canonical_json=b"{}")
        with pytest.raises(grpc.aio.AioRpcError) as error:
            await call(stub.CreateProfile, request)
        assert error.value.code() == grpc.StatusCode.UNAUTHENTICATED
        assert contexts == []
        monkeypatch.setenv("BALOOT_API_KEY", _SERVICE_KEY)

        async def check(rpc, request, role, *, write=False):
            for denied in _CONFIG_READ_ROLES:
                start = len(contexts)
                with pytest.raises(grpc.aio.AioRpcError) as error:
                    await call(rpc, request, metadata=(("x-api-key", _SERVICE_KEY), ("x-baloot-role", denied)))
                expected = grpc.StatusCode.PERMISSION_DENIED if write else grpc.StatusCode.NOT_FOUND
                assert error.value.code() == expected
                if write:
                    assert len(contexts) == start
                else:
                    assert_context(start, denied, _KEY_SERVICE, admin=False)
            start = len(contexts)
            for invalid in ((), (("x-api-key", "wrong-fixture-key"),)):
                with pytest.raises(grpc.aio.AioRpcError) as error:
                    await call(rpc, request, metadata=(*invalid, ("x-baloot-role", role)))
                assert error.value.code() == grpc.StatusCode.UNAUTHENTICATED
            assert len(contexts) == start
            headers = [
                ("x-api-key", _SERVICE_KEY), ("x-baloot-role", role),
                ("x-baloot-subject-id", "fixture-operator"),
            ]
            result = await call(rpc, request, metadata=headers)
            assert_context(start, role, "fixture-operator")
            return json.loads(result.canonical_json)

        def assert_context(start, role, subject, *, tenant=DEFAULT_TENANT, admin=True):
            (identity, token), = contexts[start:]
            assert identity.configuration_admin is admin
            assert (identity.role, identity.subject_id, identity.tenant_id) == (role, subject, tenant)
            assert resets[start:] == [token]

        for role in ("admin", "service"):
            profile_id = f"grpc-oidc-{role}"
            for version in ("1.0.0", "1.0.1"):
                request = configuration_pb2.ProfileDraft(
                    family="oidc_providers",
                    canonical_json=json.dumps(_oidc_test_payload(profile_id, version)).encode(),
                )
                draft = await check(stub.CreateProfile, request, role, write=True)
                assert draft["createdBy"] == "fixture-operator"
                if version == "1.0.0":
                    published = await check(stub.PublishProfile, configuration_pb2.Identifier(
                        value=draft["revisionId"], expected_etag=draft["etag"],
                    ), role, write=True)
                    assert published["publishedBy"] == "fixture-operator"
                else:
                    await check(stub.RejectProfile, configuration_pb2.TransitionRequest(
                        value=draft["revisionId"], expected_etag=draft["etag"], reason="Reject draft",
                    ), role, write=True)
                    audit_expectations.append((draft["revisionId"], ("created", "rejected", "rejection_reason"), "Reject draft"))
            await check(stub.ResolveProfile, configuration_pb2.ProfileResolution(
                family="oidc_providers", profile_id=profile_id,
            ), role)
            await check(stub.CompareProfiles, configuration_pb2.ProfileCompare(
                left=published["revisionId"], right=published["revisionId"],
            ), role)
            for read_role in (role, *_CONFIG_READ_ROLES):
                for family in ("", "oidc_providers"):
                    start = len(contexts)
                    result = await call(
                        stub.ListProfiles,
                        configuration_pb2.ProfileQuery(family=family),
                        metadata=(("x-api-key", _SERVICE_KEY), ("x-baloot-role", read_role)),
                    )
                    items = json.loads(result.canonical_json)["items"]
                    listed = any(item["revisionId"] == published["revisionId"] for item in items)
                    assert listed is (read_role == role)
                    assert_context(start, read_role, _KEY_SERVICE, admin=listed)
            for tenant, scopes in (
                (DEFAULT_TENANT, (Scope.READ,)),
                ("fixture-tenant", (Scope.READ, Scope.CONFIG_ADMIN)),
            ):
                token = ACCESS_TOKENS.issue(
                    subject_id="fixture-reader", role="admin", tenant_id=tenant, scopes=scopes,
                )
                metadata = (("authorization", f"Bearer {token}"),)
                start = len(contexts)
                with pytest.raises(grpc.aio.AioRpcError) as error:
                    await call(stub.CompareProfiles, configuration_pb2.ProfileCompare(
                        left=published["revisionId"], right=published["revisionId"],
                    ), metadata=metadata)
                assert error.value.code() == grpc.StatusCode.NOT_FOUND
                assert_context(start, "admin", "fixture-reader", tenant=tenant, admin=False)
                with pytest.raises(grpc.aio.AioRpcError) as error:
                    await call(stub.CreateProfile, request, metadata=metadata)
                assert error.value.code() == grpc.StatusCode.PERMISSION_DENIED
                assert len(contexts) == start + 1
            await check(stub.RetireProfile, configuration_pb2.TransitionRequest(
                value=published["revisionId"], expected_etag=published["etag"], reason="Retire revision",
            ), role, write=True)
            audit_expectations.append((
                published["revisionId"], ("created", "published", "retired", "retirement_reason"), "Retire revision",
            ))

    grpc_probe(
        tmp_path,
        monkeypatch,
        probe,
        stub_type=configuration_pb2_grpc.ConfigurationControlStub,
        interceptors=(ObserveServerCompletion(),),
    )
    assert completions.empty()
    assert resets == [token for _, token in contexts]
    token = database_identity.set(request_context.DatabaseIdentityContext(configuration_admin=True))
    try:
        for revision, expected_types, reason in audit_expectations:
            snapshot = CONFIGURATION.store.events_with_verification("profile", revision)
            assert snapshot["verification"]["valid"]
            events = snapshot["items"]
            assert tuple(event["eventType"] for event in events) == expected_types
            assert all(event["actorId"] == "fixture-operator" and event["createdAt"] for event in events)
            assert events[-1]["payload"]["reason"] == reason
    finally:
        database_identity.reset(token)


def test_grpc_policy_covers_every_descriptor_method():
    from importlib import import_module
    from pkgutil import iter_modules

    from baloot_platform import generated
    from baloot_platform.service.authorization import RPC_POLICY

    declared = {
        (f"/{service.full_name}", method.name)
        for module in iter_modules(generated.__path__, f"{generated.__name__}.")
        if module.name.endswith("_pb2")
        for service in import_module(module.name).DESCRIPTOR.services_by_name.values()
        for method in service.methods
    }
    assert declared == {
        (service, method) for service, methods in RPC_POLICY.items() for method in methods
    }


def test_http_policy_covers_every_registered_route():
    from fastapi.staticfiles import StaticFiles
    from starlette.routing import Mount, WebSocketRoute

    from baloot_platform.service.authorization import HTTP_POLICY, http_routes

    registered = set()
    for route in http_routes(app.router):
        if isinstance(route, Mount):
            assert isinstance(route.app, StaticFiles)
            methods = {"GET", "HEAD"}
        elif isinstance(route, WebSocketRoute):
            continue
        else:
            methods = route.methods
            assert methods, route
        registered.update((method, route.path) for method in methods)

    assert registered <= HTTP_POLICY.keys()


@pytest.mark.parametrize(("role", "operator_scope"), [
    ("anonymous", None),
    ("guest", None),
    ("player", None),
    ("admin", "baloot.model.admin"),
    ("admin", "baloot.config.admin"),
])
def test_http_operator_route_scopes(monkeypatch, role, operator_scope):
    from starlette.responses import Response
    from starlette.routing import Route

    from baloot_platform.service import api

    learning = "/api/v1/learning"
    operations = {
        ("GET", f"{learning}/queue"): "baloot.model.admin",
        ("GET", f"{learning}/datasets"): "baloot.model.admin",
        ("POST", f"{learning}/datasets"): "baloot.model.admin",
        ("GET", f"{learning}/models"): "baloot.model.admin",
        ("POST", f"{learning}/models:train"): "baloot.model.admin",
        ("POST", f"{learning}/models/{{model_id}}:evaluate"): "baloot.model.admin",
        ("POST", f"{learning}/models/{{model_id}}:promote"): "baloot.model.admin",
        ("POST", "/api/v1/challenges/{challenge_id}:recalibrate"): "baloot.config.admin",
    }
    # Exercise dispatch through the policy while keeping the service operations inert.
    for method, path in operations:
        route = next(
            route for route in app.routes
            if getattr(route, "path", None) == path and method in route.methods
        )
        monkeypatch.setattr(route, "app", Response(status_code=204))

    unlisted = Route(
        "/api/v1/games/policy-fixture", lambda _: Response(status_code=204), methods=["GET"],
    )
    monkeypatch.setattr(app.router, "routes", [unlisted, *app.routes])
    operations[("GET", unlisted.path)] = "baloot.config.admin"

    headers = {}
    if role == "guest":
        token = api.IDENTITY.create_guest("Policy fixture")["session"]["token"]
    elif role == "player":
        account = api.IDENTITY.register(f"fixture_{uuid.uuid4().hex[:8]}", "fixture-password")
        token = account["session"]["token"]
    elif operator_scope is not None:
        token = api.ACCESS_TOKENS.issue(
            subject_id="fixture-operator", role=role, scopes=[operator_scope],
        )
    else:
        token = None
    if token is not None:
        headers["authorization"] = f"Bearer {token}"

    with TestClient(app, headers=headers) as client:
        for (method, path), scope in operations.items():
            path = path.format(model_id="fixture-model", challenge_id="fixture-challenge")
            if role == "anonymous":
                expected = 401
            else:
                expected = 204 if operator_scope == scope else 403
            response = client.request(method, path)
            assert response.status_code == expected, (method, path, response.text)


@pytest.mark.parametrize(("path", "body", "role", "scope", "missing_detail"), [
    (
        "/api/v1/configuration-registry/profiles/CPR-POLICY-FIXTURE:publish",
        {"expectedEtag": 1}, "admin", "baloot.config.admin", "configuration profile revision not found",
    ),
    (
        "/api/v1/adjudication/cases/CASE-POLICY-FIXTURE/rulings",
        {"outcome": "no_violation", "reasonAr": "Fixture ruling"}, "referee",
        "baloot.referee", "adjudication case not found",
    ),
])
def test_http_privileged_operations_accept_their_own_scope(path, body, role, scope, missing_detail):
    from baloot_platform.service.api import ACCESS_TOKENS

    with TestClient(app) as client:
        for grant, expected in ((scope, 404), ("baloot.play", 403)):
            token = ACCESS_TOKENS.issue(subject_id="fixture-operator", role=role, scopes=[grant])
            response = client.post(path, headers={"authorization": f"Bearer {token}"}, json=body)
            assert response.status_code == expected, response.text
            detail = missing_detail if grant == scope else f"missing required scope: {scope}"
            assert response.json()["detail"] == detail


@pytest.mark.parametrize(("rpc_service", "http_service"), [
    ("baloot.v8.SakkaPlatform", "sakka"),
    ("baloot.v9.SharedPlatform", "shared"),
])
def test_run_comparison_uses_the_same_scope_over_http_and_grpc(rpc_service, http_service):
    from baloot_platform.security import Scope
    from baloot_platform.service.authorization import http_scope, rpc_scope

    assert rpc_scope(f"/{rpc_service}/CompareRuns") == http_scope(
        "POST", f"/api/v1/{http_service}/comparisons",
    ) == Scope.PLAY


def test_api_key_reads_real_legacy_board_and_run_over_http_and_grpc(
    legacy_service_records, tmp_path, monkeypatch,
):
    from baloot_platform.generated import sakka_pb2

    legacy_board, _, legacy_run = legacy_service_records
    board_id, run_id = legacy_board["boardId"], legacy_run["runId"]
    headers = {"x-api-key": _SERVICE_KEY}

    with TestClient(app) as client:
        assert client.get(f"{_BOARDS}/{board_id}", headers=headers).status_code == 200
        run = client.get(f"{_RUNS}/{run_id}", headers=headers)
        assert run.status_code == 200, run.text
        assert run.json()["run"]["runId"] == run_id
        verified = client.get(f"{_BOARDS}/{board_id}/verification", headers=headers)
        assert verified.status_code == 200 and verified.json()["signatureValid"]
        created = client.post(_BOARDS, json={"visibility": "private", "seed": 6030}, headers=headers)
        assert created.status_code == 201, created.text
        assert created.json()["createdBy"] == "SVC-A748F51287ECF77FE4FA"

    async def probe(grpc, stub):
        for method, value, key in ((stub.GetBoard, board_id, "boardId"), (stub.GetRun, run_id, "runId")):
            wire = await method(sakka_pb2.Identifier(value=value), metadata=tuple(headers.items()), timeout=5)
            output = json.loads(wire.canonical_json)
            assert output.get("run", output)[key] == value
        with pytest.raises(grpc.aio.AioRpcError) as rejected:
            await stub.GetBoard(
                sakka_pb2.Identifier(value=board_id), metadata=(("x-api-key", "wrong"),), timeout=5,
            )
        assert rejected.value.code() == grpc.StatusCode.UNAUTHENTICATED

    _grpc_probe(tmp_path, monkeypatch, probe)


def test_service_ownership_survives_repeated_startup(legacy_service_records):
    from baloot_platform.service.api import SAKKA

    board, code, run = legacy_service_records
    board_id, run_id = board["boardId"], run["runId"]
    room = SAKKA.store.get_room(code)
    before = SAKKA.store.members(room.room_id)
    for _ in range(2):
        with TestClient(app) as client:
            headers = {"x-api-key": _SERVICE_KEY}
            listed = client.get(_BOARDS, headers=headers)
            assert listed.status_code == 200
            assert any(item["boardId"] == board_id for item in listed.json()["items"])
            reopened = client.post(f"{_ROOMS}/{code}/runs", headers=headers)
            assert reopened.status_code == 201, reopened.text
            assert reopened.json()["run"]["runId"] == run_id
    assert SAKKA.store.members(room.room_id) == before
    assert SAKKA.store.get_board(board_id)[0].created_by == _LEGACY_SERVICE
    assert SAKKA.store.get_run(run_id)[0].subject_id == _LEGACY_SERVICE


def test_service_ownership_leaves_other_subjects_and_explicit_callers_separate(legacy_service_records):
    from baloot_platform.service.api import SAKKA

    legacy_board, _, _ = legacy_service_records
    legacy_id = legacy_board["boardId"]
    default = {"x-api-key": _SERVICE_KEY}
    explicit = {**default, "x-baloot-subject-id": "SVC-ANOTHER-OWNER"}
    with TestClient(app) as client:
        created = client.post(_BOARDS, json={"visibility": "private", "seed": 6031}, headers=explicit)
        assert created.status_code == 201, created.text
        other_id = created.json()["boardId"]
        before = SAKKA.store.get_board(other_id)
        assert client.get(f"{_BOARDS}/{legacy_id}", headers=default).status_code == 200
        assert client.get(f"{_BOARDS}/{other_id}", headers=default).status_code == 404
        assert client.get(f"{_BOARDS}/{legacy_id}", headers=explicit).status_code == 404
        assert client.get(f"{_BOARDS}/{other_id}", headers=explicit).status_code == 200
        assert SAKKA.store.get_board(other_id) == before


def test_grpc_requires_a_real_credential_without_configured_service_key(tmp_path, monkeypatch):
    from baloot_platform.core.enums import Seat
    from baloot_platform.generated import sakka_pb2
    from baloot_platform.service.api import SAKKA, IDENTITY

    monkeypatch.delenv("BALOOT_API_KEY", raising=False)
    owner = IDENTITY.create_guest("Board owner")
    viewer = IDENTITY.create_guest("Board viewer")
    board = SAKKA.create_board_from_new_game(
        created_by=owner["identity"]["subjectId"], seed=6040, dealer=Seat.EAST,
        visibility="private",
    )

    async def probe(grpc, stub):
        identifier = sakka_pb2.Identifier(value=board.board_id)
        for metadata in ((), (("x-api-key", "synthetic-unconfigured"),)):
            with pytest.raises(grpc.aio.AioRpcError) as rejected:
                await stub.GetBoard(identifier, metadata=metadata, timeout=5)
            assert rejected.value.code() == grpc.StatusCode.UNAUTHENTICATED
        owner_metadata = (("authorization", f"Bearer {owner['session']['token']}"),)
        result = await stub.GetBoard(identifier, metadata=owner_metadata, timeout=5)
        assert json.loads(result.canonical_json)["boardId"] == board.board_id
        viewer_metadata = (("authorization", f"Bearer {viewer['session']['token']}"),)
        with pytest.raises(grpc.aio.AioRpcError) as rejected:
            await stub.GetBoard(identifier, metadata=viewer_metadata, timeout=5)
        assert rejected.value.code() == grpc.StatusCode.NOT_FOUND

    _grpc_probe(tmp_path, monkeypatch, probe)


def test_grpc_unbound_mutations_require_a_configured_service_key(tmp_path, monkeypatch):
    from baloot_platform.generated import platform_pb2, platform_pb2_grpc
    from baloot_platform.security import Scope
    from baloot_platform.service.api import ACCESS_TOKENS, IDENTITY

    mutations = (
        ("CreateGame", platform_pb2.CreateGameRequest, platform_pb2.GameState),
        ("SubmitBid", platform_pb2.BidAction, platform_pb2.GameState),
        ("PlayCard", platform_pb2.PlayAction, platform_pb2.GameState),
        ("ForkDecision", platform_pb2.ForkDecisionRequest, platform_pb2.HistoryOperationResult),
        ("RecordDecisionAttempt", platform_pb2.DifficultyAttempt, platform_pb2.DifficultyResult),
        ("RecordChallengeAttempt", platform_pb2.ChallengeAttempt, platform_pb2.DifficultyResult),
    )
    register = platform_pb2_grpc.add_BalootPlatformServicer_to_server
    fixture_payload = b'{"fixture":true}'

    def register_fixture(servicer, server):
        for method, _, response_type in mutations:
            async def inert_mutation(_request, _context, response_type=response_type):
                return response_type(canonical_json=fixture_payload)

            setattr(servicer, method, inert_mutation)
        register(servicer, server)

    monkeypatch.setattr(platform_pb2_grpc, "add_BalootPlatformServicer_to_server", register_fixture)
    monkeypatch.setenv("BALOOT_API_KEY", _SERVICE_KEY)
    guest = IDENTITY.create_guest("Game authorization fixture")["session"]["token"]
    signed = ACCESS_TOKENS.issue(subject_id="fixture-player", role="player", scopes=[Scope.PLAY])
    service_key = (("x-api-key", _SERVICE_KEY),)
    denied_metadata = (
        (),
        (("x-api-key", "unconfigured-fixture-key"),),
        (("authorization", f"Bearer {guest}"),),
        (("authorization", f"Bearer {signed}"),),
    )

    async def probe(grpc, stub):
        calls = [(getattr(stub, method), request_type()) for method, request_type, _ in mutations]
        for call, request in calls:
            for metadata in denied_metadata:
                with pytest.raises(grpc.aio.AioRpcError) as rejected:
                    await call(request, metadata=metadata, timeout=5)
                assert rejected.value.code() == grpc.StatusCode.UNAUTHENTICATED
            assert (await call(request, metadata=service_key, timeout=5)).canonical_json == fixture_payload

        monkeypatch.delenv("BALOOT_API_KEY")
        for call, request in calls:
            with pytest.raises(grpc.aio.AioRpcError) as rejected:
                await call(request, metadata=service_key, timeout=5)
            assert rejected.value.code() == grpc.StatusCode.UNAUTHENTICATED

    _grpc_probe(tmp_path, monkeypatch, probe, stub_type=platform_pb2_grpc.BalootPlatformStub)


def test_grpc_improvement_approval_uses_the_verified_principal(tmp_path, monkeypatch):
    from baloot_platform.generated import hitl_pb2, hitl_pb2_grpc
    from baloot_platform.security import Scope
    from baloot_platform.service import api

    monkeypatch.setenv("BALOOT_API_KEY", _SERVICE_KEY)
    proposal = "IMP-FIXTURE"
    required = api.HITL.profile["improvement"]["minimum_independent_validators"]
    evaluations = [
        {"validatorId": f"FIXTURE-{number}", "independent": True, "status": "passed"}
        for number in range(required)
    ]
    update = MagicMock(return_value={"status": "approved_for_challenger"})
    monkeypatch.setattr(api.HITL.store, "get_proposal", lambda _id: {"proposalId": proposal})
    monkeypatch.setattr(api.HITL.store, "list_improvement_evaluations", lambda _id: evaluations)
    monkeypatch.setattr(api.HITL.store, "update_proposal", update)

    def bearer(subject, role):
        token = api.ACCESS_TOKENS.issue(subject_id=subject, role=role, scopes=[Scope.MODEL_ADMIN])
        return (("authorization", f"Bearer {token}"),)

    service_key = (("x-api-key", _SERVICE_KEY), ("x-baloot-subject-id", "FIXTURE-KEY"))
    scoped_service = bearer("FIXTURE-SIGNED-SERVICE", "service")
    guest = api.IDENTITY.create_guest("Approval fixture")["session"]["token"]
    scoped_admin = bearer("FIXTURE-ADMIN", "admin")

    def request(role):
        return hitl_pb2.JsonRequest(canonical_json=json.dumps({"proposalId": proposal, "role": role}).encode())

    async def probe(grpc, stub):
        declared_role = request("expert")
        for metadata in (
            scoped_service,
            scoped_service + service_key,
            (("authorization", f"Bearer {guest}"),) + service_key,
        ):
            with pytest.raises(grpc.aio.AioRpcError) as denied:
                await stub.ApproveImprovement(declared_role, metadata=metadata, timeout=5)
            assert denied.value.code() == grpc.StatusCode.PERMISSION_DENIED
        update.assert_not_called()

        for approval, metadata, actor in (
            (request("player"), scoped_admin, "FIXTURE-ADMIN"),
            (declared_role, service_key, "FIXTURE-KEY"),
        ):
            approved = await stub.ApproveImprovement(approval, metadata=metadata, timeout=5)
            assert json.loads(approved.canonical_json)["status"] == "approved_for_challenger"
            assert update.call_args.kwargs["approvedBy"] == actor

    _grpc_probe(tmp_path, monkeypatch, probe, stub_type=hitl_pb2_grpc.HumanLearningStub)


@pytest.fixture(params=["sqlite", "postgres"])
def tenant_jobs(request, tmp_path):
    from baloot_platform.storage.jobs import JobStore
    from baloot_platform.storage.postgres import PostgresJobStore

    tenants = {"job-home": "default", "job-other": "tenant-other"}
    if request.param == "sqlite":
        store = JobStore(tmp_path / "jobs.sqlite3")
        for job_id, tenant in tenants.items():
            store.create(job_id, {"fixture": True}, tenant_id=tenant)
        return store

    # Exercise the production PostgreSQL queries and row mapping at the pool boundary.
    store = object.__new__(PostgresJobStore)
    store.pool = MagicMock()
    connection = store.pool.connection.return_value.__enter__.return_value
    created = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def execute(sql, params):
        if "FROM job_events" in sql:
            return SimpleNamespace(fetchall=lambda: [("queued", {}, created)])
        if "ORDER BY created_at" in sql:
            assert "WHERE tenant_id=%s" in sql
            tenant, limit = params
            rows = [(job_id,) for job_id, owner in tenants.items() if owner == tenant][:limit]
            return SimpleNamespace(fetchall=lambda: rows)
        assert "WHERE job_id=%s AND tenant_id=%s" in sql
        job_id, tenant = params
        row = None
        if tenants.get(job_id) == tenant:
            row = (job_id, tenant, "video", "queued", 100, 0, {}, None, None,
                   0, 3, None, None, False, created, created)
        return SimpleNamespace(fetchone=lambda: row)

    connection.execute.side_effect = execute
    return store


def test_operator_jobs_and_media_require_role_and_tenant_over_http_and_grpc(tmp_path, monkeypatch, tenant_jobs):
    from fastapi.staticfiles import StaticFiles

    from baloot_platform.generated import platform_pb2, platform_pb2_grpc
    from baloot_platform.learning.store import LearningStore
    from baloot_platform.security import Scope
    from baloot_platform.service import api
    from baloot_platform.storage.request_context import CURRENT_DATABASE_IDENTITY, DatabaseIdentityContext

    monkeypatch.setattr(api, "STORE", tenant_jobs)
    for job_id, tenant in (("job-home", "default"), ("job-other", "tenant-other")):
        identity_context = CURRENT_DATABASE_IDENTITY.set(DatabaseIdentityContext(tenant_id=tenant))
        try:
            assert [job["jobId"] for job in tenant_jobs.list_jobs()] == [job_id]
        finally:
            CURRENT_DATABASE_IDENTITY.reset(identity_context)
    videos = LearningStore(tmp_path / "learning.sqlite3")
    video = videos.put_video_record({
        "ownerKind": "challenge", "ownerId": "demo", "episodeHash": "fixture",
        "rendererId": "fixture", "manifest": {}, "timeline": {"segments": []},
    })
    video_id = video["videoId"]
    monkeypatch.setattr(api, "LEARNING_STORE", videos)
    outputs = next(route for route in app.routes if route.name == "outputs")
    (tmp_path / "demo.mp4").write_bytes(b"Synthetic media")
    monkeypatch.setattr(outputs, "app", StaticFiles(directory=tmp_path))

    def bearer(role, scope, tenant=None):
        token = api.ACCESS_TOKENS.issue(
            subject_id="fixture-subject", role=role, scopes=[scope], tenant_id=tenant,
        )
        return {"authorization": f"Bearer {token}"}

    admin = bearer("admin", Scope.CONFIG_ADMIN)
    other = bearer("admin", Scope.CONFIG_ADMIN, "tenant-other")
    unscoped_admin = bearer("admin", Scope.READ)
    player = {**bearer("player", Scope.READ), "x-baloot-role": "admin"}
    guest = bearer("guest", Scope.READ)
    expert = bearer("expert", Scope.READ)
    referee = bearer("referee", Scope.HIDDEN_READ)
    service = bearer("service", Scope.CONFIG_ADMIN)
    denied_roles = (unscoped_admin, guest, player, expert, referee)
    job_path = "/api/v1/jobs"
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/api/v1/auth/me").json()["identity"] is None
        for path in (
            f"{job_path}/job-home", "/api/v1/video-jobs/job-home",
            "/api/v1/video-jobs/job-home/events", f"/api/v1/videos/{video_id}",
            f"/api/v1/videos/{video_id}/timeline", f"/api/v1/videos/{video_id}/audit-map",
            "/outputs/demo.mp4",
        ):
            assert client.get(path).status_code == 401
            for headers in denied_roles:
                assert client.get(path, headers=headers).status_code == 403
            assert client.get(path, headers=admin).status_code == 200
        assert client.get(f"{job_path}/job-home", headers=service).status_code == 200
        assert client.get(f"{job_path}/job-home", headers=other).status_code == 404
        assert client.get(f"{job_path}/job-other", headers=admin).status_code == 404
        assert client.get(f"{job_path}/job-other", headers=other).status_code == 200
        assert client.get("/api/v1/video-jobs/job-other", headers=other).status_code == 200
        assert client.get("/api/v1/video-jobs/job-other/events", headers=other).status_code == 200
        assert client.get("/api/v1/video-jobs/job-home/events", headers=other).status_code == 404
        assert client.get(f"/api/v1/videos/{video_id}", headers=other).status_code == 403
        assert client.get("/outputs/demo.mp4", headers=other).status_code == 403
        ranged = client.get("/outputs/demo.mp4", headers={**admin, "range": "bytes=0-3"})
        assert ranged.status_code == 206 and ranged.content == b"Synt"

    async def probe(grpc, stub):
        async def call(method, request, headers):
            return await method(request, metadata=tuple(headers.items()), timeout=5)

        async def rejected(method, request, headers, status):
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await call(method, request, headers)
            assert error.value.code() == status

        home_job = platform_pb2.JobRef(job_id="job-home")
        other_job = platform_pb2.JobRef(job_id="job-other")
        video_ref = platform_pb2.VideoRecordRef(video_id=video_id)
        assert (await call(stub.HealthCheck, platform_pb2.Empty(), {})).status == "ok"
        for method, request in (
            (stub.GetJob, home_job), (stub.GetVideoRecord, video_ref),
            (stub.GetVideoTimeline, video_ref), (stub.GetVideoAuditMap, video_ref),
        ):
            await rejected(method, request, {}, grpc.StatusCode.UNAUTHENTICATED)
            for headers in denied_roles:
                await rejected(method, request, headers, grpc.StatusCode.PERMISSION_DENIED)
            assert (await call(method, request, admin)).canonical_json

        await rejected(stub.GetJob, home_job, other, grpc.StatusCode.NOT_FOUND)
        await rejected(stub.GetJob, other_job, admin, grpc.StatusCode.NOT_FOUND)
        assert (await call(stub.GetJob, home_job, service)).job.job_id == "job-home"
        assert (await call(stub.GetJob, other_job, other)).job.job_id == "job-other"
        await rejected(stub.GetVideoRecord, video_ref, other, grpc.StatusCode.PERMISSION_DENIED)

    _grpc_probe(tmp_path, monkeypatch, probe, stub_type=platform_pb2_grpc.BalootPlatformStub)


def test_grpc_archived_replay_preserves_completion_ownership_and_media_scopes(tmp_path, monkeypatch):
    from baloot_platform.core.enums import Seat
    from baloot_platform.generated import platform_pb2, platform_pb2_grpc
    from baloot_platform.security import Scope
    from baloot_platform.service import api

    run = SimpleNamespace(run_id="run", status="active", subject_id="archive-owner", board_id="board")
    board = SimpleNamespace(observer_seat=Seat.SOUTH)
    shared_store = SimpleNamespace(
        get_run_for_game=lambda _: (run, b""),
        get_run_for_decision=lambda _: (run, b""),
        get_board=lambda _: (board, b""),
    )
    videos = [{"videoId": "archive-video"}]
    decision = {"decisionId": "decision", "gameId": "game", "videoRecords": videos}
    game = {"gameId": "game", "gameHash": "hash", "decisions": [decision], "videoRecords": videos}
    learning_store = SimpleNamespace(
        get_game=lambda identifier: game if identifier == "hash" else None,
        get_decision=lambda _: decision,
        verify_game=lambda _: {"valid": True},
        list_game_audits=lambda *_, **__: [],
        list_video_records=lambda *_, **__: videos,
    )
    monkeypatch.setattr(api, "SAKKA", SimpleNamespace(store=shared_store))
    monkeypatch.setattr(api, "LEARNING_STORE", learning_store)

    def metadata(subject, role, *scopes, tenant=None):
        token = api.ACCESS_TOKENS.issue(subject_id=subject, role=role, scopes=scopes, tenant_id=tenant)
        return (("authorization", f"Bearer {token}"),)

    owner = metadata("archive-owner", "guest", Scope.READ)
    viewer = metadata("archive-viewer", "player", Scope.READ)
    operator = metadata("archive-operator", "admin", Scope.READ, Scope.CONFIG_ADMIN)
    foreign = metadata("archive-owner", "player", Scope.READ, tenant="tenant-other")

    async def probe(grpc, stub):
        ref = platform_pb2.Identifier(value="hash")

        async def denied(credentials, code, request=ref):
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await stub.GetReplay(request, metadata=credentials, timeout=5)
            assert error.value.code() == code
            return error.value.details()

        await denied((), grpc.StatusCode.UNAUTHENTICATED)
        assert "active shared run" in await denied(owner, grpc.StatusCode.PERMISSION_DENIED)
        missing = await denied(viewer, grpc.StatusCode.NOT_FOUND, platform_pb2.Identifier(value="missing"))
        assert await denied(viewer, grpc.StatusCode.NOT_FOUND) == missing
        run.status = "completed"
        await denied(viewer, grpc.StatusCode.NOT_FOUND)
        await denied(foreign, grpc.StatusCode.PERMISSION_DENIED)
        replay = json.loads((await stub.GetReplay(ref, metadata=owner, timeout=5)).canonical_json)
        assert replay["videos"] == replay["game"]["videoRecords"] == []
        assert "videoRecords" not in replay["decisions"][0]
        replay = json.loads((await stub.GetReplay(ref, metadata=operator, timeout=5)).canonical_json)
        assert replay["videos"] == replay["game"]["videoRecords"] == videos

    _grpc_probe(tmp_path, monkeypatch, probe, stub_type=platform_pb2_grpc.BalootPlatformStub)


def test_grpc_reopen_and_audit_require_operator(root, tmp_path, monkeypatch):
    from baloot_platform.audit import ReplayAuditService
    from baloot_platform.core.enums import Seat
    from baloot_platform.generated import platform_pb2, platform_pb2_grpc
    from baloot_platform.learning.service import LearningService
    from baloot_platform.service import api
    from baloot_platform.service.game import GameManager
    from baloot_platform.storage.game_store import SQLiteGameStore

    game_store = SQLiteGameStore(tmp_path / "games.sqlite3")
    manager = GameManager(root, game_store)
    learning = LearningService(root, tmp_path, game_manager=manager)
    store = learning.store
    session = GameManager(root).create(seed=6051, dealer=Seat.EAST)
    identifier = learning.archive_session(session, status="manual_archive")["gameHash"]
    archived = store.get_game(identifier)
    monkeypatch.setattr(api, "LEARNING_STORE", store)
    monkeypatch.setattr(api, "LEARNING", learning)
    monkeypatch.setattr(api, "AUDIT", ReplayAuditService(root, store, learning))
    monkeypatch.setenv("BALOOT_API_KEY", _SERVICE_KEY)

    guest = api.IDENTITY.create_guest("Archive guest")
    player = api.IDENTITY.register(f"archive_{uuid.uuid4().hex[:12]}", "synthetic-password", "Archive player")
    operator = (("x-api-key", _SERVICE_KEY),)

    async def probe(grpc, stub):
        reopen = platform_pb2.ReopenGameRequest(identifier=identifier)
        audit = platform_pb2.ReplayAuditRequest(identifier=identifier, mode="observer_safe")
        for identity in (guest, player):
            metadata = (("authorization", f"Bearer {identity['session']['token']}"),)
            for method, request in ((stub.ReopenGame, reopen), (stub.AuditGame, audit)):
                with pytest.raises(grpc.aio.AioRpcError) as error:
                    await method(request, metadata=metadata, timeout=5)
                assert error.value.code() == grpc.StatusCode.PERMISSION_DENIED
                assert store.get_game(identifier) == archived
                assert game_store.load(session.id) is None
                assert store.list_game_audits(identifier) == []

        reopened = await stub.ReopenGame(reopen, metadata=operator, timeout=5)
        assert reopened.game_id == session.id
        assert game_store.load(session.id).id == session.id
        report = await stub.AuditGame(audit, metadata=operator, timeout=5)
        assert report.audit_id
        assert store.get_game_audit(report.audit_id)["gameHash"] == identifier

    _grpc_probe(tmp_path, monkeypatch, probe, stub_type=platform_pb2_grpc.BalootPlatformStub)


def test_service_ownership_keeps_legacy_live_seat_and_event_chain(monkeypatch):
    from baloot_platform.service.api import LIVE

    monkeypatch.setenv("BALOOT_API_KEY", _SERVICE_KEY)
    monkeypatch.delenv("BALOOT_API_SUBJECT_ID", raising=False)
    default = {"x-api-key": _SERVICE_KEY}
    legacy = {**default, "x-baloot-subject-id": _LEGACY_SERVICE}
    rooms = "/api/v1/live/rooms"
    with TestClient(app) as client:
        response = client.post(rooms, json={"seed": 6050}, headers=legacy)
        assert response.status_code == 201, response.text
        room_id = response.json()["room"]["roomId"]
        url = f"{rooms}/{room_id}"
        claimed = client.post(
            f"{url}/seats", json={"seat": "south", "displayName": "Legacy fixture"}, headers=legacy,
        )
        assert claimed.status_code == 201, claimed.text
        before = LIVE.verify(room_id)
        assert before["valid"]

        listed = client.get(rooms, headers=default)
        assert any(room["roomId"] == room_id for room in listed.json()["items"])
        legacy_projection = client.get(url, headers=legacy).json()
        assert legacy_projection["viewerSeat"] == "south"
        assert client.get(url, headers=default).json() == legacy_projection
        assert LIVE.verify(room_id) == before
        ready = client.post(f"{url}/ready", json={"ready": True}, headers=default)
        assert ready.status_code == 200, ready.text
        assert LIVE.verify(room_id)["valid"]
