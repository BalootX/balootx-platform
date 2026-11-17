from __future__ import annotations

import copy
from pathlib import Path

import pytest

from baloot_platform.core.enums import Seat
from baloot_platform.history.event_store_v7 import EventEnvelope
from baloot_platform.identity import IdentityService, SQLiteIdentityStore
from baloot_platform.learning.service import LearningService
from baloot_platform.learning.store import LearningStore
from baloot_platform.sakka import SakkaService, SQLiteSakkaStore
from baloot_platform.service.game import GameManager
from baloot_platform.storage.game_store import SQLiteGameStore


class _Audit:
    def audit_decision(self, decision_id: str, *, deep: bool, force: bool):
        return {
            "decisionId": decision_id,
            "decisionKind": "play",
            "actualAction": {"code": "7S"},
            "bestAction": {"code": "8S"},
            "qualityLabel": "good",
            "qualityScore": 84.0,
            "forced": False,
            "expectedRegret": 2.0,
            "winProbabilityDrop": 0.03,
            "catastrophicCrossing": False,
            "reasons": [{"detailAr": "بديل محسوب أفضل بقليل."}],
        }


def _services(root: Path, tmp_path: Path):
    learning_store = LearningStore(tmp_path / "learning.sqlite3")
    manager = GameManager(
        root,
        SQLiteGameStore(tmp_path / "games.sqlite3"),
        learning_store=learning_store,
    )
    learning = LearningService(root, tmp_path / "learning", store=learning_store, game_manager=manager)
    manager.learning_service = learning
    service = SakkaService(
        root,
        SQLiteSakkaStore(tmp_path / "sakka.sqlite3"),
        game_manager=manager,
        audit_service=_Audit(),
    )
    return service, manager, learning_store


def test_optional_identity_guest_local_and_friends(tmp_path: Path):
    store = SQLiteIdentityStore(tmp_path / "identity.sqlite3")
    service = IdentityService(store, password_minimum=8)
    guest = service.create_guest("ضيف أ")
    resolved, session = service.resolve(guest["session"]["token"])
    assert resolved is not None and resolved.is_guest and session is not None

    first = service.register("player.one", "correct-horse", "لاعب أول")
    second = service.register("player.two", "correct-battery", "لاعب ثان")
    with pytest.raises(ValueError):
        service.login("player.one", "wrong-password")
    logged = service.login("player.one", "correct-horse")
    assert logged["identity"]["authenticated"] is True

    request = service.request_friend(first["identity"]["subjectId"], second["identity"]["subjectId"])
    service.respond_friend(request["requestId"], second["identity"]["subjectId"], accept=True)
    friends = service.friends(first["identity"]["subjectId"])
    assert [item["subjectId"] for item in friends] == [second["identity"]["subjectId"]]


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



def test_guest_upgrade_preserves_subject_and_history(tmp_path: Path):
    store = SQLiteIdentityStore(tmp_path / "identity-upgrade.sqlite3")
    service = IdentityService(store, password_minimum=8)
    guest = service.create_guest("ضيف مستمر")
    guest_id = guest["identity"]["subjectId"]
    upgraded = service.register(
        "lasting.player", "correct-horse", "لاعب مستمر",
        upgrade_guest_id=guest_id,
    )
    assert upgraded["identity"]["subjectId"] == guest_id
    assert upgraded["identity"]["authenticated"] is True
    original_token_identity, _ = service.resolve(guest["session"]["token"])
    assert original_token_identity is not None
    assert original_token_identity.subject_id == guest_id
    assert original_token_identity.authenticated is True
    logged = service.login("lasting.player", "correct-horse")
    assert logged["identity"]["subjectId"] == guest_id


def test_signed_sakka_board_detects_payload_and_signature_tampering(root: Path, tmp_path: Path):
    from baloot_platform.sakka.signing import HmacBoardSigner
    import sqlite3

    learning_store = LearningStore(tmp_path / "signed-learning.sqlite3")
    manager = GameManager(root, SQLiteGameStore(tmp_path / "signed-games.sqlite3"), learning_store=learning_store)
    signer = HmacBoardSigner.from_secret(b"s" * 32)
    store = SQLiteSakkaStore(tmp_path / "signed-sakka.sqlite3")
    service = SakkaService(root, store, game_manager=manager, board_signer=signer)
    board = service.create_board_from_new_game(
        created_by="USR-S", seed=12345, dealer=Seat.EAST, observer_seat=Seat.SOUTH,
    )
    verified = service.verify_board(board.board_id)
    assert verified["valid"] is True and verified["signatureValid"] is True
    with sqlite3.connect(store.path) as conn:
        conn.execute(
            "UPDATE sakka_boards_v8 SET board_signature=? WHERE board_id=?",
            ("0" * 64, board.board_id),
        )
    failed = service.verify_board(board.board_id)
    assert failed["valid"] is False
    assert any(item["code"] == "board_signature_invalid" for item in failed["issues"])

def test_event_timestamp_is_part_of_tamper_evident_identity():
    envelope = EventEnvelope.create(
        game_id="GAME-1", sequence=0, event_type="round_created", payload={"seed": 1},
        profile_bundle_hash="a" * 64, engine_version="8.0.0", previous_event_hash=None,
        occurred_at="2026-07-27T00:00:00+00:00",
    )
    assert envelope.verify(None)
    changed = EventEnvelope(
        envelope.event_id, envelope.game_id, envelope.sequence, envelope.event_type,
        envelope.schema_version, envelope.payload, envelope.profile_bundle_hash,
        envelope.engine_version, envelope.previous_event_hash, envelope.event_hash,
        "2026-07-28T00:00:00+00:00", envelope.actor_seat, envelope.actor_subject_id,
    )
    assert not changed.verify(None)


def test_nested_profile_fields_and_mode_overlays_are_closed(tmp_path, root):
    import yaml
    from baloot_platform.config import build_bundle
    from baloot_platform.core.rules import RuleBook

    profile = yaml.safe_load((root / 'configs/identity/optional_simple_v1.yaml').read_text(encoding='utf-8'))
    profile['guest']['typo_session_hours'] = 10
    bad = tmp_path / 'identity.yaml'
    bad.write_text(yaml.safe_dump(profile, allow_unicode=True), encoding='utf-8')
    bundle = yaml.safe_load((root / 'configs/bundle.yaml').read_text(encoding='utf-8'))
    bundle['components']['identity'] = str(bad)
    bundle_path = tmp_path / 'bundle.yaml'
    bundle_path.write_text(yaml.safe_dump(bundle, allow_unicode=True), encoding='utf-8')
    try:
        build_bundle(root, bundle_path=bundle_path)
    except ValueError as exc:
        assert 'typo_session_hours' in str(exc) or 'Additional properties' in str(exc)
    else:
        raise AssertionError('unknown nested identity setting was accepted')

    base = root / 'configs/rules/saudi_common_v1.yaml'
    overlay = tmp_path / 'bad-overlay.yaml'
    overlay.write_text('play:\n  typo_obligation: true\n', encoding='utf-8')
    try:
        RuleBook.load(base, overlay)
    except ValueError as exc:
        assert 'typo_obligation' in str(exc)
    else:
        raise AssertionError('unknown rule overlay setting was accepted')


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
