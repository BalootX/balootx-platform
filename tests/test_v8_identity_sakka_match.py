from __future__ import annotations

from pathlib import Path


from baloot_platform.core.enums import Seat

from sakka_identity_helpers import _services


def test_sakka_full_match_same_start(root: Path, tmp_path: Path):
    service, _manager, _learning = _services(root, tmp_path)
    board = service.create_board_from_new_game(
        created_by="USR-A", seed=9911, dealer=Seat.NORTH, observer_seat=Seat.SOUTH,
        match_score=(140, 140), scope="match", maximum_rounds=3,
        agent_profiles={seat: "beginner" for seat in ("north", "west", "south", "east")},
    )
    room = service.open_room(board.board_id, created_by="USR-A")
    payload = service.start_run(room.room_code, subject_id="USR-A", display_name="الأول", is_guest=False)
    for _ in range(120):
        if payload["run"]["status"] == "completed":
            break
        payload = service.apply_action(
            payload["run"]["runId"], subject_id="USR-A", action=payload["legalActions"][0],
        )
    assert payload["run"]["status"] == "completed"
    assert 1 <= payload["run"]["roundsCompleted"] <= 3
    assert payload["run"]["result"]["scope"] == "match"
