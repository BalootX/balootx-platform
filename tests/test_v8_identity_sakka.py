from __future__ import annotations

from pathlib import Path

import pytest

from baloot_platform.core.enums import Seat
from baloot_platform.history.event_store_v7 import EventEnvelope
from baloot_platform.identity import IdentityService, SQLiteIdentityStore
from baloot_platform.learning.store import LearningStore
from baloot_platform.sakka import SakkaService, SQLiteSakkaStore
from baloot_platform.service.game import GameManager
from baloot_platform.storage.game_store import SQLiteGameStore


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
