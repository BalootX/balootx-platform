from __future__ import annotations

import os
import shutil
from contextlib import closing
from functools import partial
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
from baloot_platform.security import AuthorizationPolicy, Scope, SignedAccessTokenService
from baloot_platform.service.game import GameManager
from baloot_platform.service.routes.auth import PendingRequestItem, build_auth_router
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


@given(st.text(alphabet=st.characters(codec="utf-8", exclude_characters="\x00"), max_size=40))
def test_identity_cursor_round_trips_and_rejects_another_scope(subject_id: str):
    scope = cursor_scope("search", "sample")
    anchor = (subject_id,)
    cursor = encode_cursor(scope, anchor)
    keys = ("subject_id",)
    assert decode_cursor(cursor, scope, keys=keys) == anchor
    with pytest.raises(InvalidIdentityCursor):
        decode_cursor(cursor, cursor_scope("search", "different"), keys=keys)


def test_identity_pagination_configuration_is_validated(root):
    from baloot_platform.config import load_profile_payload, validate_profile_payload

    path = root / "configs/identity/optional_simple_v1.yaml"
    profile = load_profile_payload("identity", path, root=root)
    settings = profile.pop("pagination")
    with pytest.raises(ValueError, match="'pagination' is a required property"):
        validate_profile_payload("identity", profile, root=root)
    with pytest.raises(ValueError, match="pagination: Additional properties are not allowed"):
        validate_profile_payload(
            "identity", {**profile, "pagination": {**settings, "unexpected": True}}, root=root
        )
    with pytest.raises(ValueError, match="must not exceed"):
        validate_profile_payload(
            "identity", {**profile, "pagination": {**settings, "default_page_size": settings["maximum_page_size"] + 1}},
            root=root,
        )


def test_python_sdk_forwards_identity_pagination_options():
    from sdk.python.client import BalootClient

    observed = []

    def response(request):
        observed.append((request.url.path, dict(request.url.params)))
        return httpx.Response(200, json={"items": [], "nextCursor": None})

    with BalootClient() as client:
        client.client.close()
        client.client = httpx.Client(base_url="https://example.com", transport=httpx.MockTransport(response))
        assert client.friends()["items"] == []
        assert client.friends_page()["nextCursor"] is None
        client.friend_requests(cursor="encoded-_", limit=5)
        client.search_users("Fixture + sample", cursor="encoded-_", limit=7)
    assert observed == [
        ("/api/v1/friends", {}),
        ("/api/v2/friends", {}),
        ("/api/v2/friends/requests", {"cursor": "encoded-_", "limit": "5"}),
        ("/api/v2/users:search", {"query": "Fixture + sample", "cursor": "encoded-_", "limit": "7"}),
    ]


def test_friend_routes_preserve_service_delegation_and_apply_v2_read_scope(tmp_path, monkeypatch):
    from baloot_platform.service import api
    from baloot_platform.service.client_identity import service_subject_ownership

    key = "synthetic-service-key"
    monkeypatch.setenv("BALOOT_API_KEY", key)
    store = SQLiteIdentityStore(tmp_path / "transport-identity.sqlite3")
    service = IdentityService(store)
    tokens = SignedAccessTokenService(b"synthetic-identity-transport-signing-key")
    now = 1_767_225_600
    monkeypatch.setattr(tokens, "verify", partial(tokens.verify, now=now))
    monkeypatch.setattr(api, "IDENTITY", service)
    monkeypatch.setattr(api, "ACCESS_TOKENS", tokens)
    _, legacy = service_subject_ownership({}, configured_key=key)
    for subject_id in (legacy, "SUB-DELEGATED", "SUB-FRIEND", "SUB-SENDER"):
        store.insert_subject(
            subject_id=subject_id, kind="local", username=None, display_name=subject_id,
            password_salt=None, password_hash=None, created_at="2026-01-01T00:00:00+00:00",
        )
    for owner in (legacy, "SUB-DELEGATED"):
        friendship = service.request_friend("SUB-FRIEND", owner)
        service.respond_friend(friendship["requestId"], owner, accept=True)
        service.request_friend("SUB-SENDER", owner)

    app = FastAPI()
    app.middleware("http")(api.platform_security)
    app.include_router(build_auth_router(api.IDENTITY, pagination=api.CONTAINER.identity_pagination, policy=api.AUTHORIZATION_POLICY))
    delegated = {"x-api-key": key, "x-baloot-subject-id": "SUB-DELEGATED"}

    def bearer(role, scope):
        return {"Authorization": "Bearer " + tokens.issue(subject_id="SUB-DELEGATED", role=role, scopes=[scope], now=now)}

    with TestClient(app) as client:
        v2_search = "/api/v2/users:search"
        query = {"query": "SUB"}
        assert client.get(v2_search, params=query).status_code == 401
        for headers in ({"x-api-key": key}, delegated):
            assert client.get(v2_search, params=query, headers=headers).status_code == 200
        for role in ("guest", "player", "expert", "referee", "admin", "service", "unknown"):
            expected = 403 if role in {"guest", "unknown"} else 200
            assert client.get(v2_search, params=query, headers=bearer(role, Scope.READ)).status_code == expected
            assert client.get(v2_search, params=query, headers=bearer(role, Scope.PLAY)).status_code == 403
        for version in ("v1", "v2"):
            for resource in ("friends", "friends/requests"):
                path = f"/api/{version}/{resource}"
                assert client.get(path).status_code == 401
                assert client.get(path, headers={**delegated, "x-api-key": "invalid"}).status_code == 401
                for headers, owner in (({"x-api-key": key}, legacy), (delegated, "SUB-DELEGATED")):
                    response = client.get(path, headers=headers)
                    assert response.status_code == 200
                    row, = response.json()["items"]
                    if resource == "friends":
                        assert row["subjectId"] == "SUB-FRIEND"
                    elif version == "v1":
                        assert row["recipient_id"] == owner
                        assert row["responded_at"] is None
                    else:
                        assert row["recipientId"] == owner
                for role in ("guest", "player", "expert", "referee", "admin", "service"):
                    allowed = client.get(path, headers=bearer(role, Scope.READ))
                    assert allowed.status_code == 200
                    assert len(allowed.json()["items"]) == 1
                    assert client.get(path, headers=bearer(role, Scope.PLAY)).status_code == (200 if version == "v1" else 403)
                assert client.get(path, headers=bearer("unknown", Scope.READ)).status_code == (200 if version == "v1" else 403)

        outgoing = client.post("/api/v1/friends/requests", json={"recipientId": "SUB-SENDER"}, headers=delegated)
        assert outgoing.status_code == 201
        assert outgoing.json()["senderId"] == "SUB-DELEGATED"
        pending, = client.get("/api/v1/friends/requests", headers=delegated).json()["items"]
        response = client.post(f"/api/v1/friends/requests/{pending['request_id']}:respond", json={"accept": False}, headers=delegated)
        assert response.status_code == 200
        assert client.get("/api/v1/friends/requests", headers=delegated).json() == {"items": []}


def test_identity_pagination_large_lists_and_plans(tmp_path, root, monkeypatch):
    dsn = os.environ.get("BALOOT_IDENTITY_TEST_POSTGRES")
    postgres = dsn is not None
    database = tmp_path / "identity.sqlite3"
    if postgres:
        store = PostgresIdentityStore(dsn)
        connect = store.pool.connection
    else:
        store = SQLiteIdentityStore(database)
        connect = store._connect

    owner, legacy = "SUB-0120-OWNER", "SUB-0120-LEGACY"
    candidates = [f"SUB-{index:04}" for index in range(256)]
    unrelated = [f"SUB-OTHER-{index:04}" for index in range(8_192)]
    started = "2026-01-01T00:00:00+00:00"
    newer = "2026-01-02T00:00:00+00:00"
    request_keys = [(newer if index % 2 else started, f"REQ-{index:04}") for index in range(len(candidates))]
    migrations = {"031_identity_list_pagination.sql", "032_identity_pending_requests_index.sql"}

    def migrate(source_root):
        from scripts.migrate_postgres import main

        with monkeypatch.context() as patch:
            patch.setenv("BALOOT_DATABASE_URL", dsn)
            patch.setenv("BALOOT_ROOT", str(source_root))
            main()

    indexes = {
        "list_friends": "idx_identity_friendships_high_low",
        "list_friend_requests": "idx_identity_pending_requests",
    }
    subject_rows = [(subject, "local" if index % 2 else "guest", None,
                     "Page-Fixture A" if index % 2 else "Page-Fixture Z", started, None)
                    for index, subject in enumerate(candidates)]
    subject_rows.extend((subject, "local", None, "Background Fixture", started, None) for subject in unrelated)
    subject_rows.extend([
        (owner, "service", None, "Owner Fixture", started, None),
        (legacy, "service", None, "Legacy Fixture", started, None),
        ("SUB-DISABLED", "local", None, "Page-Fixture", started, started),
        ("SUB-USERNAME", "local", "page-fixture", "Username Fixture", started, None),
        ("SUB-LITERAL", "local", None, "Literal %_! token", started, None),
        ("SUB-SIMILAR", "local", None, "Literal AB! token", started, None),
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
        if postgres:
            baseline_root = tmp_path / "previous"
            baseline_migrations = baseline_root / "migrations"
            baseline_migrations.mkdir(parents=True)
            for path in (root / "migrations").glob("*.sql"):
                if path.name not in migrations:
                    shutil.copyfile(path, baseline_migrations / path.name)
            migrate(baseline_root)

        p = store.placeholder
        with connect() as conn, closing(conn.cursor()) as cursor:
            if postgres:
                provider, locale = cursor.execute(
                    "SELECT datlocprovider,datlocale FROM pg_database WHERE datname = current_database()"
                ).fetchone()
                assert provider == "i" and locale == "en-US"
            else:
                for index in indexes.values():
                    cursor.execute(f"DROP INDEX {index}")
            cursor.executemany(
                f"""INSERT INTO {store.subjects}
                    (subject_id,kind,username,display_name,created_at,disabled_at)
                    VALUES({p},{p},{p},{p},{p},{p})""", subject_rows,
            )
            cursor.executemany(
                f"INSERT INTO {store.friendships} (subject_low,subject_high,created_at) VALUES({p},{p},{p})",
                friendships,
            )
            cursor.executemany(
                f"""INSERT INTO {store.requests}
                    (request_id,sender_id,recipient_id,status,created_at) VALUES({p},{p},{p},{p},{p})""",
                requests,
            )
            cursor.execute("ANALYZE")

        operations = (
            (store.search_subjects, "Fixture", ("1", candidates[39]), {"exact_username": "fixture", "maximum_results": 64}),
            (store.list_friends, owner, (candidates[119],), {}),
            (store.list_friend_requests, owner, (newer, request_keys[119][1]), {}),
        )
        search_index = "subjects_pkey" if postgres else "sqlite_autoindex_identity_subjects_v8_1"

        def plans(stage):
            with connect() as conn, monkeypatch.context() as patch:
                prefix = "EXPLAIN (ANALYZE, BUFFERS) " if postgres else "EXPLAIN QUERY PLAN "

                def explain(sql, params):
                    rows = conn.execute(prefix + sql, params).fetchall()
                    nonlocal plan
                    plan = "\n".join(row[0 if postgres else 3] for row in rows)
                    return []

                plan = ""
                patch.setattr(store, "_read_page_rows", explain)
                for fetch, value, after, options in operations:
                    for anchor in ((), after):
                        fetch(value, limit=14, after=anchor, **options)
                        print(f"{stage} {fetch.__name__} {'continuation' if anchor else 'first'}\n{plan}")
                        if fetch == store.search_subjects:
                            assert search_index in plan, plan
                        elif stage == "indexed":
                            assert indexes[fetch.__name__] in plan, plan
                        else:
                            assert indexes[fetch.__name__] not in plan, plan

        ownership_context = SERVICE_SUBJECT_OWNER.set((owner, legacy))
        try:
            plans("baseline")
            if postgres:
                import psycopg

                with psycopg.connect(dsn, autocommit=True) as conn:
                    with pytest.raises(psycopg.errors.UniqueViolation):
                        conn.execute(
                            "CREATE UNIQUE INDEX CONCURRENTLY idx_identity_pending_requests "
                            "ON identity.friend_requests(recipient_id)"
                        )
                migrate(root)
                migrate(root)
                with connect() as conn:
                    conn.execute("ANALYZE")
                timestamp_row = store._read_page_rows(
                    "SELECT %s::timestamptz AS created_at, %s::timestamp AS observed_at", (started, newer)
                )
                assert timestamp_row == [{"created_at": started, "observed_at": newer}]
            else:
                SQLiteIdentityStore(database)
                with connect() as conn:
                    conn.execute("ANALYZE")
            plans("indexed")

            service = IdentityService(store)
            auth = {
                "fixture-owner": Identity(owner, "service", "Synthetic"),
                "fixture-other": Identity("SUB-USERNAME", "local", "Synthetic"),
                "fixture-guest": Identity(candidates[0], "guest", "Synthetic"),
            }
            monkeypatch.setattr(service, "resolve", lambda token: (
                (auth[token], {}) if token in auth else (None, None)
            ))
            pagination = IdentityPaginationConfig(
                default_page_size=13, maximum_page_size=21,
                minimum_search_query_length=3, maximum_search_query_length=20, maximum_search_results=64,
                allow_guest_search=False,
            )
            app = FastAPI()
            policy = AuthorizationPolicy()
            app.include_router(build_auth_router(service, pagination=pagination, policy=policy))

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

                search_path = "/api/v2/users:search"
                friends_path = "/api/v2/friends"
                requests_path = "/api/v2/friends/requests"
                assert client.get(friends_path).status_code == 401
                assert client.get(requests_path).status_code == 401
                assert client.get(search_path, params={"query": "Page-Fixture"}).status_code == 401
                client.headers["Authorization"] = "Bearer fixture-guest"
                assert client.get(friends_path).status_code == 200
                assert client.get(search_path, params={"query": "Page-Fixture"}).status_code == 403

                def legacy_search(query, limit=None):
                    params = {"query": query}
                    if limit is not None:
                        params["limit"] = limit
                    response = client.get("/api/v1/users:search", params=params)
                    assert response.status_code == 200
                    assert set(response.json()) == {"items"}
                    return response.json()["items"]

                assert len(legacy_search("")) == 20
                assert len(legacy_search("  P ")) == 20
                assert legacy_search("x" * 129) == []
                assert len(legacy_search("", 1000)) == 100
                for limit in (0, -1):
                    assert legacy_search("Page-Fixture", limit) == legacy_search("Page-Fixture", 1)
                assert len(legacy_search("Page-Fixture", 0)) == 1
                assert {row["subjectId"] for row in legacy_search("Literal %_!")} == {"SUB-LITERAL", "SUB-SIMILAR"}
                legacy_order = legacy_search("Page-Fixture", 100)
                assert {(row["kind"], row["displayName"]) for row in legacy_order} == {("local", "Page-Fixture A")}
                assert [row["subjectId"] for row in legacy_search("Fixture", 2)] == [legacy, owner]

                guest_app = FastAPI()
                guest_app.include_router(build_auth_router(
                    service, pagination=pagination.model_copy(update={"allow_guest_search": True}), policy=policy,
                ))
                with TestClient(guest_app) as guest_client:
                    guest_search = guest_client.get(
                        search_path, params={"query": "Page-Fixture"}, headers=client.headers,
                    )
                    assert guest_search.status_code == 200
                    assert len(guest_search.json()["items"]) == pagination.default_page_size

                client.headers["Authorization"] = "Bearer fixture-owner"
                friends, friend_cursor = collect(friends_path, "subjectId")
                assert friends == candidates
                assert sorted(service.friend_ids(owner)) == friends
                legacy_friends = client.get("/api/v1/friends").json()
                assert set(legacy_friends) == {"items"}
                assert sorted(row["subjectId"] for row in legacy_friends["items"]) == candidates
                names = [row["displayName"] for row in legacy_friends["items"]]
                assert names == sorted(names)

                pending, request_cursor = collect(requests_path, "requestId")
                assert pending == [key for _, key in sorted(request_keys, reverse=True)]
                request_item = client.get(requests_path, params={"limit": 1}).json()["items"][0]
                request_keys_expected = {
                    "requestId", "senderId", "recipientId", "status", "createdAt",
                    "senderDisplayName", "senderUsername",
                }
                legacy_keys_expected = {
                    "request_id", "sender_id", "recipient_id", "status", "created_at",
                    "sender_display_name", "sender_username",
                }
                if not postgres:
                    legacy_keys_expected.add("responded_at")
                assert set(request_item) == request_keys_expected
                unpaged_requests = client.get("/api/v1/friends/requests").json()["items"]
                assert len(unpaged_requests) == len(request_keys)
                assert all(set(item) == legacy_keys_expected for item in unpaged_requests)
                if postgres:
                    sqlite_store = SQLiteIdentityStore(database)
                    for subject_id in (candidates[0], owner):
                        sqlite_store.insert_subject(
                            subject_id=subject_id, kind="local", username=None, display_name="Synthetic",
                            password_salt=None, password_hash=None, created_at=started,
                        )
                    sqlite_store.create_friend_request(
                        request_id=request_keys[0][1], sender_id=candidates[0], recipient_id=owner, created_at=started,
                    )
                    sqlite_item = sqlite_store.list_friend_requests(owner, limit=1)[0]
                    assert set(PendingRequestItem.model_validate(sqlite_item).model_dump()) == set(request_item)

                users, search_cursor = collect(search_path, "subjectId", query="Page-Fixture")
                assert users == ["SUB-USERNAME", *candidates[:63]]
                exact = service.search_page("  PAGE-FIXTURE  ", limit=2, maximum_results=64)["items"]
                assert [row["subjectId"] for row in exact] == ["SUB-USERNAME", candidates[0]]
                literal = client.get(search_path, params={"query": "Literal %_!"}).json()
                assert [row["subjectId"] for row in literal["items"]] == ["SUB-LITERAL"]
                assert client.get(search_path, params={"query": "Absent"}).json() == {
                    "items": [], "nextCursor": None,
                }
                outside_cap = encode_cursor(cursor_scope("search", "Page-Fixture"), ("1", candidates[100]))
                assert client.get(search_path, params={"query": "Page-Fixture", "cursor": outside_cap}).json() == {
                    "items": [], "nextCursor": None,
                }
                wildcard_only = client.get(search_path, params={"query": "%__%"})
                assert wildcard_only.status_code == 200
                assert wildcard_only.json() == {"items": [], "nextCursor": None}
                for invalid in ("", "  ", "%", "_", "ab", "abc\0"):
                    rejection = client.get(search_path, params={"query": invalid})
                    assert rejection.status_code == 422
                    assert "at least 3 characters" in rejection.json()["detail"][0]["msg"]
                oversized = client.get(search_path, params={"query": "x" * 21})
                assert oversized.status_code == 422
                assert "at most 20 characters" in oversized.json()["detail"][0]["msg"]
                wrong_query = client.get(search_path, params={"query": "Changed", "cursor": search_cursor})
                assert wrong_query.status_code == 422
                assert "Restart from the first page" in wrong_query.json()["detail"][0]["msg"]
                for params in ({"limit": 0}, {"limit": -1}, {"limit": "many"}, {"cursor": "@invalid"}, {"cursor": "A" * 2_049}):
                    rejected = client.get(search_path, params={"query": "Page-Fixture", **params})
                    assert rejected.status_code == 422
                    assert rejected.json()["detail"][0]["loc"] == ["query", next(iter(params))]
                assert client.get(search_path).status_code == 422
                assert client.get(friends_path, params={"limit": 0}).status_code == 422
                for bad in ("", "@invalid", request_cursor, "A" * 2_049):
                    assert client.get(friends_path, params={"cursor": bad}).status_code == 422

                client.headers["Authorization"] = "Bearer fixture-other"
                assert client.get(search_path, params={"query": "Page-Fixture"}).status_code == 200
                assert client.get(friends_path, params={"cursor": friend_cursor}).status_code == 422
                assert client.get(friends_path).json() == {"items": [], "nextCursor": None}
                assert client.get(requests_path, params={"cursor": request_cursor}).status_code == 422
                assert client.get(requests_path).json() == {"items": [], "nextCursor": None}

                service.respond_friend(pending[12], owner, accept=False)
                store.create_friend_request(request_id="REQ-AHEAD", sender_id="SUB-LITERAL", recipient_id=owner,
                                            created_at="2026-01-03T00:00:00+00:00")
                store.create_friend_request(request_id="REQ-BEHIND", sender_id="SUB-SIMILAR", recipient_id=owner,
                                            created_at="2025-12-31T00:00:00+00:00")
                continuation = []
                while request_cursor:
                    request_page = service.request_page(owner, limit=17, cursor=request_cursor)
                    continuation.extend(row["request_id"] for row in request_page["items"])
                    request_cursor = request_page["nextCursor"]
                assert continuation == [*pending[13:], "REQ-BEHIND"]
                assert service.request_page(owner, limit=1)["items"][0]["request_id"] == "REQ-AHEAD"
        finally:
            SERVICE_SUBJECT_OWNER.reset(ownership_context)
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
