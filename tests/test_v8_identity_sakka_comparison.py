from __future__ import annotations

from pathlib import Path


from baloot_platform.core.enums import Seat

from sakka_identity_helpers import _services


def test_sakka_comparison_reports_verified_outcome_flip(root: Path, tmp_path: Path):
    from baloot_platform.config import build_bundle
    from baloot_platform.core.engine import BalootEngine
    from baloot_platform.core.episode_io import load_episode_spec
    from baloot_platform.service.game import GameSession

    service, manager, _learning = _services(root, tmp_path)
    bundle = build_bundle(root)
    episode = load_episode_spec(root / "examples/v7-outcome-flip-source.json", bundle.rules)
    session = GameSession(
        "OUTCOME-FLIP-BASE", "full", BalootEngine(bundle.rules),
        episode.state_before_pause.clone(), None, 1, bundle.sha256,
        agent_profiles=dict(manager.agent_registry.default_lineup),
    )
    manager.import_session(session)
    board = service.create_board(
        session, created_by="USR-A", observer_seat=Seat.SOUTH,
        scope="round", label="لفة تحسم الصكّة",
    )
    room = service.open_room(board.board_id, created_by="USR-A")
    first = service.start_run(room.room_code, subject_id="USR-A", display_name="الأول", is_guest=False)
    second = service.start_run(room.room_code, subject_id="USR-B", display_name="الثاني", is_guest=False)

    first = service.apply_action(first["run"]["runId"], subject_id="USR-A", action={"kind": "play", "card": "D10"})
    second = service.apply_action(second["run"]["runId"], subject_id="USR-B", action={"kind": "play", "card": "CA"})
    while first["run"]["status"] == "active":
        first = service.apply_action(first["run"]["runId"], subject_id="USR-A", action=first["legalActions"][0])
    while second["run"]["status"] == "active":
        second = service.apply_action(second["run"]["runId"], subject_id="USR-B", action=second["legalActions"][0])

    comparison = service.compare_runs(
        first["run"]["runId"], second["run"]["runId"], requester_subject_id="USR-A",
    )
    assert comparison["sameStartVerified"] is True
    assert comparison["verification"]["runA"]["valid"] is True
    assert comparison["verification"]["runB"]["valid"] is True
    assert comparison["runA"]["result"]["matchScore"] == [158, 140]
    assert comparison["runB"]["result"]["matchScore"] == [146, 152]
    assert comparison["winnerRunId"] == first["run"]["runId"]
    assert comparison["firstDivergence"]["sequence"] == 0
    assert comparison["firstDivergence"]["runA"]["action"]["card"] == "D10"
    assert comparison["firstDivergence"]["runB"]["action"]["card"] == "CA"
    assert len(comparison["comparisonHash"]) == 64
