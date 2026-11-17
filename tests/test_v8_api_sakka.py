from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

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
