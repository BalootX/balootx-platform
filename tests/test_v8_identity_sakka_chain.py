from __future__ import annotations

from pathlib import Path


from baloot_platform.core.enums import Seat

from sakka_identity_helpers import _services


def test_sakka_action_chain_detects_tampering(root: Path, tmp_path: Path):
    import sqlite3
    service, _manager, _learning = _services(root, tmp_path)
    board = service.create_board_from_new_game(
        created_by='USR-X', seed=44771, dealer=Seat.EAST,
        observer_seat=Seat.SOUTH, scope='round',
    )
    room = service.open_room(board.board_id, created_by='USR-X')
    payload = service.start_run(room.room_code, subject_id='USR-X', display_name='المدقق', is_guest=False)
    payload = service.apply_action(
        payload['run']['runId'], subject_id='USR-X', action=payload['legalActions'][0],
    )
    assert service.verify_run(payload['run']['runId'], subject_id='USR-X')['valid'] is True
    with sqlite3.connect(service.store.path) as conn:
        conn.execute(
            "UPDATE sakka_run_actions_v8 SET action_json=? WHERE run_id=? AND sequence=0",
            ('{"kind":"bid","action":"tampered"}', payload['run']['runId']),
        )
    result = service.verify_run(payload['run']['runId'], subject_id='USR-X')
    assert result['valid'] is False
    assert any(item['code'] in {'action_hash_mismatch', 'replay_error'} for item in result['issues'])
