from __future__ import annotations

import os
from contextlib import closing
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from hypothesis import given, strategies as st

from baloot_platform.core.enums import Seat
from baloot_platform.history.event_store_v7 import EventEnvelope
from baloot_platform.identity import Identity, IdentityService, PostgresIdentityStore, SQLiteIdentityStore
from baloot_platform.identity.ownership import SERVICE_SUBJECT_OWNER
from baloot_platform.identity.pagination import (
    IdentityPaginationConfig,
    InvalidIdentityCursor,
    cursor_scope,
    decode_cursor,
    encode_cursor,
)
from baloot_platform.learning.store import LearningStore
from baloot_platform.sakka import SakkaService, SQLiteSakkaStore
from baloot_platform.service.game import GameManager
from baloot_platform.service.routes.auth import build_auth_router
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
    friends = service.friends(first["identity"]["subjectId"], limit=1)
    assert [item["subjectId"] for item in friends["items"]] == [second["identity"]["subjectId"]]
    assert friends["nextCursor"] is None


@given(st.text(alphabet=st.characters(codec="utf-8", exclude_characters="\x00"), max_size=40))
def test_identity_cursor_round_trips_and_rejects_another_scope(display_name: str):
    scope = cursor_scope("search", "sample")
    anchor = ("local", display_name, "SUB-EXAMPLE")
    cursor = encode_cursor(scope, anchor)
    keys = ("kind", "display_name", "subject_id")
    assert decode_cursor(cursor, scope, keys=keys) == anchor
    with pytest.raises(InvalidIdentityCursor):
        decode_cursor(cursor, cursor_scope("search", "different"), keys=keys)


def test_identity_pagination_configuration_is_validated(root):
    assert IdentityPaginationConfig.load(root).maximum_page_size > 0
    with pytest.raises(ValueError, match="must not exceed"):
        IdentityPaginationConfig(default_page_size=2, maximum_page_size=1)


def test_python_sdk_forwards_identity_pagination_options():
    from sdk.python.client import BalootClient

    observed = []

    def response(request):
        observed.append((request.url.path, dict(request.url.params)))
        return httpx.Response(200, json={"items": [], "nextCursor": None})

    with BalootClient() as client:
        client.client.close()
        client.client = httpx.Client(base_url="https://example.com", transport=httpx.MockTransport(response))
        assert client.friends()["nextCursor"] is None
        client.friend_requests(cursor="encoded-_", limit=5)
        client.search_users("Fixture + sample", cursor="encoded-_", limit=7)
    assert observed == [
        ("/api/v1/friends", {}),
        ("/api/v1/friends/requests", {"cursor": "encoded-_", "limit": "5"}),
        ("/api/v1/users:search", {"query": "Fixture + sample", "cursor": "encoded-_", "limit": "7"}),
    ]


def test_identity_pagination_large_lists_and_plans(tmp_path, root, monkeypatch):
    dsn = os.environ.get("BALOOT_IDENTITY_TEST_POSTGRES")
    postgres = dsn is not None
    database = tmp_path / "identity.sqlite3"
    if postgres:
        store = PostgresIdentityStore(dsn)
        connect = store.pool.connection
        dialect = store._postgres_page_sql
    else:
        store = SQLiteIdentityStore(database)
        connect = store._connect

        def dialect(sql):
            return sql

    owner, legacy = "SUB-0120-OWNER", "SUB-0120-LEGACY"
    candidates = [f"SUB-{index:04}" for index in range(256)]
    unrelated = [f"SUB-OTHER-{index:04}" for index in range(8_192)]
    started = "2026-01-01T00:00:00+00:00"
    newer = "2026-01-02T00:00:00+00:00"
    request_keys = [(newer if index % 2 else started, f"REQ-{index:04}") for index in range(len(candidates))]
    indexes = {
        "search_subjects": "idx_identity_active_search",
        "list_friends": "idx_identity_friendships_high_low",
        "list_friend_requests": "idx_identity_pending_requests",
    }
    subject_rows = [(subject, "local" if index % 2 else "guest", None, "Page Fixture", started, None)
                    for index, subject in enumerate(candidates)]
    subject_rows.extend((subject, "local", None, "Background Fixture", started, None) for subject in unrelated)
    subject_rows.extend([
        (owner, "service", None, "Owner Fixture", started, None),
        (legacy, "service", None, "Legacy Fixture", started, None),
        ("SUB-DISABLED", "local", None, "Page Fixture", started, started),
        ("SUB-USERNAME", "local", "page fixture", "Username Fixture", started, None),
    ])
    friendships = [(*sorted((owner if index % 2 else legacy, subject)), started)
                   for index, subject in enumerate(candidates)]
    friendships.extend((*sorted((owner, subject)), started) for subject in candidates[:20:2])
    friendships.extend((subject, unrelated[index + 1], started) for index, subject in enumerate(unrelated[:-1]))
    friendships.append((legacy, owner, started))
    requests = [(key, subject, owner if index % 2 else legacy, "pending", timestamp)
                for index, (subject, (timestamp, key)) in enumerate(zip(candidates, request_keys, strict=True))]
    requests.extend((f"REQ-OTHER-{index:04}", subject, unrelated[index + 1], "pending", started)
                    for index, subject in enumerate(unrelated[:-1]))
    requests.append(("REQ-DECLINED", candidates[0], owner, "declined", newer))

    try:
        with connect() as conn, closing(conn.cursor()) as cursor:
            if postgres:
                cursor.execute((root / "migrations/005_identity_sakka.sql").read_text(encoding="utf-8"))
            else:
                for index in indexes.values():
                    cursor.execute(f"DROP INDEX {index}")
            cursor.executemany(dialect("""INSERT INTO identity_subjects_v8
                (subject_id,kind,username,display_name,created_at,disabled_at) VALUES(?,?,?,?,?,?)"""), subject_rows)
            cursor.executemany(dialect("""INSERT INTO identity_friendships_v8
                (subject_low,subject_high,created_at) VALUES(?,?,?)"""), friendships)
            cursor.executemany(dialect("""INSERT INTO identity_friend_requests_v8
                (request_id,sender_id,recipient_id,status,created_at) VALUES(?,?,?,?,?)"""), requests)
            cursor.execute("ANALYZE")

        operations = (
            (store.search_subjects, "Page Fixture", ("local", "Page Fixture", candidates[119])),
            (store.list_friends, owner, (candidates[119],)),
            (store.list_friend_requests, owner, (newer, request_keys[119][1])),
        )

        def plans(stage):
            with connect() as conn, monkeypatch.context() as patch:
                prefix = "EXPLAIN (ANALYZE, BUFFERS) " if postgres else "EXPLAIN QUERY PLAN "

                def explain(sql, params):
                    rows = conn.execute(prefix + dialect(sql), params).fetchall()
                    nonlocal plan
                    plan = "\n".join(row[0 if postgres else 3] for row in rows)
                    return []

                plan = ""
                patch.setattr(store, "_read_page_rows", explain)
                for fetch, value, after in operations:
                    for anchor in ((), after):
                        fetch(value, limit=14, after=anchor)
                        print(f"{stage} {fetch.__name__} {'continuation' if anchor else 'first'}\n{plan}")
                        if stage == "indexed":
                            assert indexes[fetch.__name__] in plan, plan
                        else:
                            assert indexes[fetch.__name__] not in plan, plan

        token = SERVICE_SUBJECT_OWNER.set((owner, legacy))
        try:
            plans("baseline")
            if postgres:
                with connect() as conn:
                    conn.execute((root / "migrations/031_identity_list_pagination.sql").read_text(encoding="utf-8"))
                    conn.execute("ANALYZE")
            else:
                SQLiteIdentityStore(database)
                with connect() as conn:
                    conn.execute("ANALYZE")
            plans("indexed")

            service = IdentityService(store)
            auth = {"fixture-owner": owner, "fixture-other": "SUB-USERNAME"}
            monkeypatch.setattr(service, "resolve", lambda token: (
                (Identity(auth[token], "service", "Synthetic"), {}) if token in auth else (None, None)
            ))
            app = FastAPI()
            app.include_router(build_auth_router(
                service, pagination=IdentityPaginationConfig(default_page_size=13, maximum_page_size=21),
            ))

            with TestClient(app) as client:
                def collect(path, key, *, query=None):
                    params = {} if query is None else {"query": query}
                    payload = client.get(path, params=params).json()
                    assert len(payload["items"]) == 13
                    cursor = payload["nextCursor"]
                    assert cursor
                    assert client.get(path, params={**params, "limit": 1000}).json()["items"] == (
                        client.get(path, params={**params, "limit": 21}).json()["items"]
                    )
                    assert len(client.get(path, params={**params, "limit": 21}).json()["items"]) == 21
                    seen = set()
                    items = []
                    while True:
                        items.extend(item[key] for item in payload["items"])
                        if not payload["nextCursor"]:
                            return items, cursor
                        assert payload["nextCursor"] not in seen
                        seen.add(payload["nextCursor"])
                        response = client.get(path, params={**params, "cursor": payload["nextCursor"], "limit": 17})
                        assert response.status_code == 200, response.json()
                        payload = response.json()
                        assert 0 < len(payload["items"]) <= 17

                assert client.get("/api/v1/friends").status_code == 401
                client.headers["Authorization"] = "Bearer fixture-owner"
                friends, friend_cursor = collect("/api/v1/friends", "subjectId")
                assert friends == candidates
                assert list(service.friend_ids(owner, limit=13)) == friends
                pending, request_cursor = collect("/api/v1/friends/requests", "request_id")
                assert pending == [key for _, key in sorted(request_keys, reverse=True)]
                request_item = client.get("/api/v1/friends/requests", params={"limit": 1}).json()["items"][0]
                assert ("responded_at" in request_item) == (not postgres)
                users, search_cursor = collect("/api/v1/users:search", "subjectId", query="Page Fixture")
                assert users == candidates[1::2] + ["SUB-USERNAME"] + candidates[::2]
                assert client.get("/api/v1/users:search", params={"query": "Absent"}).json() == {
                    "items": [], "nextCursor": None,
                }
                assert client.get("/api/v1/users:search", params={"query": "Changed", "cursor": search_cursor}).status_code == 422
                assert client.get("/api/v1/friends", params={"limit": 0}).status_code == 422
                for bad in ("", "@invalid", request_cursor, "A" * 2_049):
                    assert client.get("/api/v1/friends", params={"cursor": bad}).status_code == 422
                client.headers["Authorization"] = "Bearer fixture-other"
                assert client.get("/api/v1/friends", params={"cursor": friend_cursor}).status_code == 422
                assert client.get("/api/v1/friends").json() == {"items": [], "nextCursor": None}

                service.respond_friend(pending[12], owner, accept=False)
                assert service.requests(owner, limit=17, cursor=request_cursor)["items"][0]["request_id"] == pending[13]
        finally:
            SERVICE_SUBJECT_OWNER.reset(token)
    finally:
        if postgres:
            store.pool.close()


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
