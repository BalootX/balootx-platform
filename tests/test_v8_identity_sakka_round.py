from __future__ import annotations

import copy
from pathlib import Path


from baloot_platform.core.enums import Seat

from sakka_identity_helpers import _services


def test_sakka_same_start_round_events_branch_and_audit(root: Path, tmp_path: Path):
    service, _manager, learning_store = _services(root, tmp_path)
    board = service.create_board_from_new_game(
        created_by="USR-A", seed=80808, dealer=Seat.EAST, observer_seat=Seat.SOUTH,
        scope="round", label="لفة اختبار", visibility="unlisted",
        agent_profiles={seat: "beginner" for seat in ("north", "west", "south", "east")},
    )
    room = service.open_room(board.board_id, created_by="USR-A", max_players=4)
    one = service.start_run(room.room_code, subject_id="USR-A", display_name="الأول", is_guest=False)
    two = service.start_run(room.room_code, subject_id="USR-B", display_name="الثاني", is_guest=False)
    assert one["run"]["stateHash"] == two["run"]["stateHash"]
    assert one["state"]["viewerSeat"] == two["state"]["viewerSeat"] == "south"
    assert one["state"]["hands"]["south"] == two["state"]["hands"]["south"]

    # Apply the same observer choices and confirm semantic state equality.
    for _ in range(40):
        if one["run"]["status"] == "completed":
            break
        action = copy.deepcopy(one["legalActions"][0])
        one = service.apply_action(one["run"]["runId"], subject_id="USR-A", action=action)
        two = service.apply_action(two["run"]["runId"], subject_id="USR-B", action=action)
        assert one["run"]["stateHash"] == two["run"]["stateHash"]
    assert one["run"]["status"] == two["run"]["status"] == "completed"
    assert one["run"]["result"] == two["run"]["result"]

    events = service.events(room.room_code)["events"]
    assert any(item["eventType"] == "room_created" for item in events)
    assert sum(item["eventType"] == "run_completed" for item in events) == 2

    verification = service.verify_run(one["run"]["runId"], subject_id="USR-A")
    assert verification["valid"] is True
    assert verification["actionCount"] > 0 and verification["headActionHash"]

    actions = service.store.actions(one["run"]["runId"])
    assert actions and all(item["decisionId"] for item in actions)
    assert all(item["actionHash"] for item in actions)
    assert actions[0]["previousActionHash"] == ""
    assert all(actions[index]["previousActionHash"] == actions[index - 1]["actionHash"] for index in range(1, len(actions)))
    branch = service.create_board_from_run_action(
        one["run"]["runId"], actions[0]["sequence"], created_by="USR-A",
        scope="round", label="فرع من القرار",
    )
    assert branch.board_hash != board.board_hash
    assert service.store.get_board(branch.share_code) is not None

    audit = service.audit_run(one["run"]["runId"], subject_id="USR-A")
    assert audit["decisionCount"] == len(actions)
    assert audit["averageDecisionQuality"] == 84.0
    feedback = service.feedback(
        one["run"]["runId"], actions[0]["sequence"], subject_id="USR-A",
        preferred_action={"code": "8S"}, confidence=0.9,
        rationale="أريد تعليم المحرك هذا المسار.", interestingness=0.7,
    )
    assert feedback["decisionId"] == actions[0]["decisionId"]
    assert learning_store.list_feedback(actions[0]["decisionId"])
