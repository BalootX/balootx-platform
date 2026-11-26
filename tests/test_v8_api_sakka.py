from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

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
# The SHA-256 subject IDs used by the pre-upgrade server for these fixtures.
_LEGACY_SERVICE = "SVC-FC90545E2119AD22B93F"
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


def _grpc_probe(tmp_path, monkeypatch, probe, *, stub_type=None):
    import grpc

    from baloot_platform.generated import sakka_pb2_grpc
    from baloot_platform.service.grpc_service import serve

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
                    await probe(grpc, (stub_type or sakka_pb2_grpc.SakkaPlatformStub)(channel))

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
