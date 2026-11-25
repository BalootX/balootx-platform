from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

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
        public = api.CONFIGURATION.store.list_profiles(limit=1)[0].revision_id
        listings = (v1, v2, "/api/v2/config-revisions?kind=profile")
        details = (
            f"{v1}/{revision}", f"{v2}/{revision}?includePayload=false", f"{v1}/{revision}/events",
            f"{v1}:compare?left={public}&right={revision}",
            f"{v1}:resolve?family=oidc_providers&profileId={profile_id}",
        )
        for role in ("admin", "service", *_CONFIG_READ_ROLES):
            authorized = role in {"admin", "service"}
            for url in listings:
                response = client.get(url, headers=headers(role))
                assert response.status_code == 200, response.text
                ids = {item["revisionId"] for item in response.json()["items"]}
                assert public in ids
                assert (revision in ids) is authorized
            for url in (f"{v1}?family=oidc_providers", f"{v2}?family=oidc_providers"):
                items = client.get(url, headers=headers(role)).json()["items"]
                assert any(item["revisionId"] == revision for item in items) is authorized
                if not authorized:
                    assert items == []
            for url in details:
                assert client.get(url, headers=headers(role)).status_code == (200 if authorized else 404)
            for base in (v1, v2):
                response = client.post(f"{base}/{revision}:validate", headers=headers(role))
                assert response.status_code == (200 if authorized else 404), response.text

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
        response = client.get(listings[-1], headers={"x-baloot-role": "admin"})
        assert response.status_code == 200, response.text
        assert all(item["revisionId"] != revision for item in response.json()["items"])


def test_http_database_role_ignores_keyless_role_header(monkeypatch):
    from fastapi import FastAPI

    from baloot_platform.service.api import platform_security
    from baloot_platform.storage.request_context import CURRENT_DATABASE_IDENTITY

    probe = FastAPI()
    probe.middleware("http")(platform_security)

    @probe.get("/database-role")
    def database_role():
        return {"role": CURRENT_DATABASE_IDENTITY.get().role}

    monkeypatch.delenv("BALOOT_API_KEY", raising=False)
    with TestClient(probe) as client:
        headers = {"x-baloot-role": "admin"}
        assert client.get("/database-role", headers=headers).json() == {"role": "anonymous"}
        monkeypatch.setenv("BALOOT_API_KEY", _SERVICE_KEY)
        assert client.get("/database-role", headers={**headers, "x-api-key": "wrong-fixture-key"}).json() == {"role": "anonymous"}
        assert client.get("/database-role", headers={**headers, "x-api-key": _SERVICE_KEY}).json() == {"role": "admin"}

# The SHA-256 subject IDs used by the pre-upgrade server for these fixtures.
_LEGACY_SERVICE = "SVC-FC90545E2119AD22B93F"
_LEGACY_KEYLESS_ONE = "SVC-F04BE659D2BE04575D83"
_BOARDS = "/api/v1/sakka/boards"
_ROOMS = "/api/v1/sakka/rooms"
_RUNS = "/api/v1/sakka/runs"


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


def _grpc_probe(tmp_path, monkeypatch, probe, *, stub_factory=None):
    import grpc

    from baloot_platform.generated import sakka_pb2_grpc
    from baloot_platform.service.grpc_service import serve

    stub_factory = stub_factory or sakka_pb2_grpc.SakkaPlatformStub
    monkeypatch.setenv("BALOOT_RUNTIME", str(tmp_path))
    monkeypatch.delenv("BALOOT_GRPC_TLS_CERT", raising=False)
    monkeypatch.delenv("BALOOT_GRPC_TLS_KEY", raising=False)
    make_server = grpc.aio.server

    async def run():
        server = None

        def create_server(**options):
            nonlocal server
            server = make_server(**options)
            port = server.add_insecure_port("127.0.0.1:0")

            async def after_start():
                async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
                    await probe(grpc, stub_factory(channel))

            monkeypatch.setattr(server, "add_insecure_port", lambda _address: port)
            monkeypatch.setattr(server, "wait_for_termination", after_start)
            return server

        monkeypatch.setattr(grpc.aio, "server", create_server)
        try:
            await serve()
        finally:
            if server is not None:
                await server.stop(0)

    asyncio.run(run())


def test_grpc_oidc_profile_lifecycle_requires_config_admin(tmp_path, monkeypatch):
    from baloot_platform.generated import configuration_pb2, configuration_pb2_grpc
    from baloot_platform.storage import request_context

    monkeypatch.delenv("BALOOT_API_KEY", raising=False)
    database_identity = request_context.CURRENT_DATABASE_IDENTITY
    contexts, resets = [], []

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

    monkeypatch.setattr(request_context, "CURRENT_DATABASE_IDENTITY", TrackingContext())

    async def probe(grpc, stub):
        request = configuration_pb2.ProfileDraft(family="oidc_providers", canonical_json=b"{}")
        with pytest.raises(grpc.aio.AioRpcError) as error:
            await stub.CreateProfile(request)
        assert error.value.code() == grpc.StatusCode.PERMISSION_DENIED
        monkeypatch.setenv("BALOOT_API_KEY", _SERVICE_KEY)

        async def check(rpc, request, role, *, tenant=None, create=False):
            for denied in _CONFIG_READ_ROLES:
                with pytest.raises(grpc.aio.AioRpcError) as error:
                    await rpc(request, metadata=(("x-api-key", _SERVICE_KEY), ("x-baloot-role", denied)))
                expected = grpc.StatusCode.PERMISSION_DENIED if create else grpc.StatusCode.NOT_FOUND
                assert error.value.code() == expected
            for invalid in ((), (("x-api-key", "wrong-fixture-key"),)):
                with pytest.raises(grpc.aio.AioRpcError) as error:
                    await rpc(request, metadata=(*invalid, ("x-baloot-role", role)))
                assert error.value.code() == grpc.StatusCode.UNAUTHENTICATED
            headers = [
                ("x-api-key", _SERVICE_KEY), ("x-baloot-role", role),
                ("x-baloot-subject-id", "fixture-operator"),
            ]
            if tenant is not None:
                headers.append(("x-baloot-tenant-id", tenant))
            result = await rpc(request, metadata=headers)
            return json.loads(result.canonical_json)

        for role in ("admin", "service"):
            profile_id = f"grpc-oidc-{role}"
            for version in ("1.0.0", "1.0.1"):
                request = configuration_pb2.ProfileDraft(
                    family="oidc_providers",
                    canonical_json=json.dumps(_oidc_test_payload(profile_id, version)).encode(),
                )
                draft = await check(stub.CreateProfile, request, role, create=True)
                if version == "1.0.0":
                    published = await check(stub.PublishProfile, configuration_pb2.Identifier(
                        value=draft["revisionId"], expected_etag=draft["etag"],
                    ), role)
                else:
                    await check(stub.RejectProfile, configuration_pb2.TransitionRequest(
                        value=draft["revisionId"], expected_etag=draft["etag"], reason="Reject draft",
                    ), role)
            await check(stub.ResolveProfile, configuration_pb2.ProfileResolution(
                family="oidc_providers", profile_id=profile_id,
            ), role)
            await check(stub.CompareProfiles, configuration_pb2.ProfileCompare(
                left=published["revisionId"], right=published["revisionId"],
            ), role, tenant="fixture-tenant")
            for read_role in (role, *_CONFIG_READ_ROLES):
                for family in ("", "oidc_providers"):
                    result = await stub.ListProfiles(
                        configuration_pb2.ProfileQuery(family=family),
                        metadata=(("x-api-key", _SERVICE_KEY), ("x-baloot-role", read_role)),
                    )
                    items = json.loads(result.canonical_json)["items"]
                    listed = any(item["revisionId"] == published["revisionId"] for item in items)
                    assert listed is (read_role == role)
            await check(stub.RetireProfile, configuration_pb2.TransitionRequest(
                value=published["revisionId"], expected_etag=published["etag"], reason="Retire revision",
            ), role)

    _grpc_probe(
        tmp_path,
        monkeypatch,
        probe,
        stub_factory=configuration_pb2_grpc.ConfigurationControlStub,
    )
    assert resets == [token for _, token in contexts]
    assert {identity.role for identity, _ in contexts} == {"admin", "service"}
    assert {identity.subject_id for identity, _ in contexts} == {"fixture-operator"}
    assert any(identity.tenant_id == "fixture-tenant" for identity, _ in contexts)


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


def test_keyless_grpc_keeps_caller_ownership_separate(tmp_path, monkeypatch):
    from baloot_platform.generated import sakka_pb2
    from baloot_platform.service.api import SAKKA
    from baloot_platform.core.enums import Seat

    monkeypatch.delenv("BALOOT_API_KEY", raising=False)
    old_board = SAKKA.create_board_from_new_game(
        created_by=_LEGACY_KEYLESS_ONE, seed=6040, dealer=Seat.EAST, visibility="private",
    )

    async def probe(grpc, stub):
        one = (("x-api-key", "keyless-one"),)
        two = (("x-api-key", "keyless-two"),)
        identifier = sakka_pb2.Identifier(value=old_board.board_id)
        legacy = await stub.GetBoard(identifier, metadata=one, timeout=5)
        assert json.loads(legacy.canonical_json)["boardId"] == old_board.board_id
        with pytest.raises(grpc.aio.AioRpcError) as rejected:
            await stub.GetBoard(identifier, metadata=two, timeout=5)
        assert rejected.value.code() == grpc.StatusCode.NOT_FOUND

        payload = sakka_pb2.JsonPayload(canonical_json=b'{"visibility":"private","seed":6041}')
        created = await stub.CreateBoard(payload, metadata=one, timeout=5)
        board = json.loads(created.canonical_json)
        identifier = sakka_pb2.Identifier(value=board["boardId"])
        assert board["createdBy"].startswith("SVC-KL-")
        assert (await stub.GetBoard(identifier, metadata=one, timeout=5)).canonical_json
        with pytest.raises(grpc.aio.AioRpcError) as rejected:
            await stub.GetBoard(identifier, metadata=two, timeout=5)
        assert rejected.value.code() == grpc.StatusCode.NOT_FOUND

    _grpc_probe(tmp_path, monkeypatch, probe)


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
