from __future__ import annotations

from pathlib import Path

import pytest

from baloot_platform.core.enums import Seat

from sakka_identity_helpers import _services


def test_active_sakka_run_analysis_fails_closed_until_completion(
    root: Path,
    tmp_path: Path,
):
    service, _manager, _learning_store = _services(root, tmp_path)
    board = service.create_board_from_new_game(
        created_by="USR-A",
        seed=80809,
        dealer=Seat.EAST,
        observer_seat=Seat.SOUTH,
        scope="round",
    )
    room = service.open_room(board.board_id, created_by="USR-A")
    payload = service.start_run(
        room.room_code,
        subject_id="USR-A",
        display_name="الأول",
        is_guest=False,
    )
    run_id = payload["run"]["runId"]

    with pytest.raises(PermissionError, match="after completion"):
        service.audit_run(run_id, subject_id="USR-A")
    service.store.put_run_audit(run_id, {"runId": run_id, "secret": "ACTIVE-AUDIT"})
    with pytest.raises(PermissionError, match="after completion"):
        service.get_run_audit(run_id, subject_id="USR-A")

    for _ in range(40):
        if payload["run"]["status"] == "completed":
            break
        payload = service.apply_action(
            run_id,
            subject_id="USR-A",
            action=payload["legalActions"][0],
        )
    assert payload["run"]["status"] == "completed"
    assert service.get_run_audit(run_id, subject_id="USR-A")["secret"] == "ACTIVE-AUDIT"
    generated = service.audit_run(run_id, subject_id="USR-A", force=True)
    assert generated["runId"] == run_id
