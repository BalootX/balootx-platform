from __future__ import annotations

import copy
import inspect
import json
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest
from hypothesis import example, given, strategies as st
from tzdata import IANA_VERSION

from baloot_platform.configuration.models import canonical, digest
from baloot_platform.configuration.schedule import _zone, normalize_schedule
from baloot_platform.configuration.store import (
    PostgresConfigurationStore,
    SQLiteConfigurationStore,
    _aggregate_identity,
)


class _Result:
    def __init__(
        self,
        row: dict[str, Any] | None = None,
        rows: list[dict[str, Any]] | None = None,
    ) -> None:
        self._row = row
        self._rows = rows or []

    def fetchone(self) -> dict[str, Any] | None:
        return self._row

    def fetchall(self) -> list[dict[str, Any]]:
        return list(self._rows)


class _ScriptedConnection:
    def __init__(self, handler: Callable[[str, Any], _Result]) -> None:
        self.handler = handler
        self.calls: list[tuple[str, Any]] = []

    def execute(self, sql: str, args: Any = None) -> _Result:
        normalized = " ".join(sql.split())
        self.calls.append((normalized, args))
        if normalized.startswith("SELECT set_config"):
            return _Result()
        return self.handler(normalized, args)


class _Pool:
    def __init__(self, connection: _ScriptedConnection) -> None:
        self.connection_value = connection
        self.commits = 0
        self.rollbacks = 0

    @contextmanager
    def connection(self):
        try:
            yield self.connection_value
        except Exception:
            self.rollbacks += 1
            raise
        else:
            self.commits += 1


def _profile_row(**changes: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "revision_id": "CPR-ONE",
        "family": "rules",
        "profile_id": "saudi",
        "version": "1.0.0",
        "scope": "global",
        "scope_key": "*",
        "tenant_id": None,
        "environment": None,
        "status": "draft",
        "payload": {"profile": {"id": "saudi", "version": "1.0.0"}},
        "payload_sha256": "a" * 64,
        "sequence": 1,
        "effective_from": None,
        "effective_until": None,
        "rollout_percentage": 100.0,
        "rollout_seed": "default",
        "created_by": "TEST",
        "created_at": datetime(
            2026, 1, 1, 3, 0, tzinfo=timezone(timedelta(hours=3))
        ),
        "updated_at": datetime(
            2026, 1, 1, 3, 0, tzinfo=timezone(timedelta(hours=3))
        ),
        "published_by": None,
        "published_at": None,
        "supersedes_revision_id": None,
        "etag": 1,
    }
    row.update(changes)
    return row


def _bundle_row(**changes: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "revision_id": "CBR-ONE",
        "bundle_id": "main",
        "version": "1.0.0",
        "mode": "full",
        "scope": "global",
        "scope_key": "*",
        "tenant_id": None,
        "environment": None,
        "status": "draft",
        "components": {"rules": "CPR-ONE"},
        "content_sha256": "b" * 64,
        "sequence": 1,
        "effective_from": None,
        "effective_until": None,
        "rollout_percentage": 100.0,
        "rollout_seed": "default",
        "created_by": "TEST",
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "published_by": None,
        "published_at": None,
        "supersedes_revision_id": None,
        "etag": 1,
    }
    row.update(changes)
    return row


def _postgres_store(connection: _ScriptedConnection) -> tuple[PostgresConfigurationStore, _Pool]:
    store = PostgresConfigurationStore.__new__(PostgresConfigurationStore)
    pool = _Pool(connection)
    store.pool = pool
    return store, pool


def test_postgres_uses_migration_columns_named_rows_and_utc_timestamps() -> None:
    profile = PostgresConfigurationStore._profile(_profile_row())
    bundle = PostgresConfigurationStore._bundle(_bundle_row())
    assert profile is not None and bundle is not None
    assert profile.payload["profile"]["id"] == "saudi"
    assert bundle.component_revisions == {"rules": "CPR-ONE"}
    assert profile.created_at == "2026-01-01T00:00:00+00:00"

    source = inspect.getsource(PostgresConfigurationStore)
    assert "payload_json" not in source
    assert "components_json" not in source
    assert "aggregate_key" not in source
    assert "SELECT *" not in source
    assert "RETURNING *" not in source
    assert "configuration.events" in source and "payload," in source


@pytest.mark.parametrize(
    ("instant", "local", "message"),
    [
        ("2026-07-01T12:00", None, "effectiveFrom requires an explicit UTC offset"),
        (None, {"effectiveFrom": "2026-07-01T12:00"}, r"localSchedule\.timeZone"),
        (None, {"timeZone": "Mars/Olympus", "effectiveFrom": "2026-07-01T12:00"}, r"localSchedule\.timeZone"),
        (None, {"timeZone": "localtime", "effectiveFrom": "2026-07-01T12:00"}, r"localSchedule\.timeZone"),
        (None, {"timeZone": "America/New_York", "effectiveFrom": "2026-03-08T02:30"}, r"localSchedule\.effectiveFrom does not exist"),
        (None, {"timeZone": "America/New_York", "effectiveUntil": "2026-11-01T01:30"}, r"localSchedule\.effectiveUntil is ambiguous"),
        (None, {"timeZone": "Australia/Lord_Howe", "effectiveUntil": "2026-10-04T02:15"}, r"localSchedule\.effectiveUntil does not exist"),
        ("2026-07-01T12:00Z", {"timeZone": "UTC", "effectiveFrom": "2026-07-01T12:00"}, "cannot both be specified"),
    ],
)
def test_schedule_rejects_invalid_bounds(instant, local, message) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_schedule(instant, None, local)


@given(
    wall_time=st.datetimes(min_value=datetime(2025, 1, 1), max_value=datetime(2027, 12, 31)),
    offset_minutes=st.integers(min_value=-14 * 60, max_value=14 * 60),
)
def test_explicit_offset_schedule_normalizes_to_utc(wall_time, offset_minutes) -> None:
    offset = timedelta(minutes=offset_minutes)
    explicit = wall_time.replace(tzinfo=timezone(offset)).isoformat()
    utc = (wall_time - offset).replace(tzinfo=timezone.utc).isoformat()
    assert normalize_schedule(explicit, None, None) == (utc, None, None)


@given(
    wall=st.datetimes(min_value=datetime(2026, 1, 2), max_value=datetime(2026, 12, 30)),
    name=st.sampled_from(["America/New_York", "Europe/Berlin", "Australia/Lord_Howe"]),
)
@example(wall=datetime(2026, 3, 8, 2, 30), name="America/New_York")
@example(wall=datetime(2026, 4, 5, 1, 45), name="Australia/Lord_Howe")
def test_local_schedule_matches_pinned_zone_round_trip(wall: datetime, name: str) -> None:
    zone = _zone(name)
    adjacent_offsets = {
        (wall + timedelta(days=days)).replace(tzinfo=zone).utcoffset()
        for days in (-1, 1)
    }
    candidates = {
        (wall - offset).replace(tzinfo=timezone.utc)
        for offset in adjacent_offsets if offset is not None
    }
    valid = {instant for instant in candidates if instant.astimezone(zone).replace(tzinfo=None) == wall}
    local = {"timeZone": name, "effectiveFrom": wall.isoformat()}
    if len(valid) != 1:
        with pytest.raises(ValueError, match="does not exist|is ambiguous"):
            normalize_schedule(None, None, local)
    else:
        instant, _, saved = normalize_schedule(None, None, local)
        assert instant == valid.pop().isoformat()
        assert saved == {**local, "zoneDataVersion": IANA_VERSION}


def test_sqlite_round_trips_local_schedule_and_preserves_older_rows(tmp_path: Path) -> None:
    path = tmp_path / "configuration.sqlite3"
    store = SQLiteConfigurationStore(path)
    profile_args = {
        "family": "rules", "profile_id": "example", "version": "1.0.0",
        "scope": "global", "tenant_id": None, "payload": {"example": True},
        "created_by": "TEST", "status": "published",
    }
    old = store.create_profile(**profile_args, effective_from="2026-07-01T12:00Z")
    with store._connect() as connection:
        connection.execute("DROP TRIGGER trg_configuration_profile_local_schedule_immutable")
        connection.execute("DROP TRIGGER trg_configuration_bundle_local_schedule_immutable")
        connection.execute("ALTER TABLE configuration_profile_revisions DROP COLUMN local_schedule")
        connection.execute("ALTER TABLE configuration_bundle_revisions DROP COLUMN local_schedule")
        connection.execute(
            "UPDATE configuration_profile_revisions SET effective_from=?, effective_until='' WHERE revision_id=?",
            ("2026-07-01T12:00", old.revision_id),
        )

    reopened = SQLiteConfigurationStore(path)
    assert reopened.get_profile(old.revision_id).to_dict() == old.to_dict()
    assert "localSchedule" not in old.to_dict()
    local = {"timeZone": "America/New_York", "effectiveFrom": "2026-07-01T12:00"}
    profile_args.update(profile_id="scheduled", payload={"scheduled": True})
    profile = reopened.create_profile(**profile_args, local_schedule=local)
    assert reopened.create_profile(**profile_args, local_schedule=local) == profile
    with pytest.raises(ValueError, match="localSchedule differs from existing revision"):
        reopened.create_profile(**profile_args, effective_from="2026-07-01T16:00Z")
    bundle = reopened.create_bundle(
        bundle_id="scheduled", version="1.0.0", mode="full", scope="global",
        tenant_id=None, component_revisions={"rules": profile.revision_id},
        status="draft", created_by="TEST", local_schedule=local,
        effective_until="2026-07-01T13:00-04:00",
    )
    cold = SQLiteConfigurationStore(path)
    assert cold.get_profile(profile.revision_id) == profile
    assert cold.get_bundle(bundle.revision_id) == bundle
    assert profile.effective_from == bundle.effective_from == "2026-07-01T16:00:00+00:00"
    assert bundle.effective_until == "2026-07-01T17:00:00+00:00"
    assert profile.local_schedule == bundle.local_schedule == {
        "timeZone": "America/New_York", "effectiveFrom": "2026-07-01T12:00:00",
        "zoneDataVersion": IANA_VERSION,
    }
    with cold._connect() as connection:
        assert connection.execute(
            "SELECT effective_until FROM configuration_profile_revisions WHERE revision_id=?",
            (old.revision_id,),
        ).fetchone()[0] is None
        for kind, revision in (("profile", profile), ("bundle", bundle)):
            with pytest.raises(sqlite3.IntegrityError, match="local_schedule is immutable"):
                connection.execute(
                    f"UPDATE configuration_{kind}_revisions SET local_schedule=NULL WHERE revision_id=?",
                    (revision.revision_id,),
                )
    audit = cold.events_with_verification("bundle", bundle.revision_id)
    assert audit["verification"]["valid"] is True
    assert audit["items"][0]["payload"]["localSchedule"] == bundle.local_schedule


def test_postgres_pool_installs_dict_row_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    marker = object()
    captured: dict[str, Any] = {}

    class FakeConnectionPool:
        def __init__(self, dsn: str, **kwargs: Any) -> None:
            captured.update({"dsn": dsn, **kwargs})

        def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setitem(sys.modules, "psycopg.rows", SimpleNamespace(dict_row=marker))
    monkeypatch.setitem(
        sys.modules, "psycopg_pool", SimpleNamespace(ConnectionPool=FakeConnectionPool)
    )
    store = PostgresConfigurationStore("postgresql://example")
    assert captured["kwargs"]["row_factory"] is marker
    store.close()
    assert captured["closed"] is True


def test_postgres_exact_dimension_and_bundle_identity_predicates() -> None:
    def handler(sql: str, _args: Any) -> _Result:
        if "LIMIT 1" in sql:
            return _Result()
        return _Result(rows=[])

    connection = _ScriptedConnection(handler)
    store, _ = _postgres_store(connection)
    assert (
        store.find_profile_by_content(
            family="rules",
            scope="global",
            scope_key="*",
            tenant_id=None,
            environment=None,
            payload_sha256="a" * 64,
        )
        is None
    )
    assert (
        store.find_bundle_by_content(
            bundle_id="main",
            scope="global",
            scope_key="*",
            tenant_id=None,
            environment=None,
            content_sha256="b" * 64,
        )
        is None
    )
    store.list_profiles(tenant_id=None, environment=None, exact_dimensions=True)
    store.list_bundles(tenant_id=None, environment=None, exact_dimensions=True)

    statements = [sql for sql, _ in connection.calls if "set_config" not in sql]
    assert all("IS NOT DISTINCT FROM" in sql for sql in statements)
    assert "bundle_id=%s" in statements[1]
    parameter = inspect.signature(
        PostgresConfigurationStore.find_bundle_by_content
    ).parameters["bundle_id"]
    assert parameter.default is inspect.Parameter.empty


@pytest.mark.parametrize("kind", ["profile", "bundle"])
def test_postgres_supersession_exclusion_uses_one_typed_revision_parameter(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def ignore_event(
        _cls: type[PostgresConfigurationStore],
        _conn: Any,
        _aggregate_type: str,
        _aggregate_id: str,
        _event_type: str,
        _actor_id: str,
        _payload: dict[str, Any],
    ) -> dict[str, Any]:
        return {}

    monkeypatch.setattr(
        PostgresConfigurationStore,
        "_append_event_tx",
        classmethod(ignore_event),
    )
    if kind == "profile":
        rows = [
            _profile_row(revision_id="CPR-A", status="published"),
            _profile_row(revision_id="CPR-B", status="published"),
        ]
        method = PostgresConfigurationStore._supersede_published_profiles_tx
        method_kwargs = {
            "family": "rules",
            "profile_id": "saudi",
            "scope": "tenant",
            "scope_key": "TEN-1",
            "tenant_id": "TEN-1",
            "environment": "production",
            "actor_id": "TEST",
        }
        expected_prefix = (
            "rules",
            "saudi",
            "tenant",
            "TEN-1",
            "TEN-1",
            "production",
        )
    else:
        rows = [
            _bundle_row(revision_id="CBR-A", status="published"),
            _bundle_row(revision_id="CBR-B", status="published"),
        ]
        method = PostgresConfigurationStore._supersede_published_bundles_tx
        method_kwargs = {
            "bundle_id": "main",
            "scope": "tenant",
            "scope_key": "TEN-1",
            "tenant_id": "TEN-1",
            "environment": "production",
            "actor_id": "TEST",
        }
        expected_prefix = (
            "main",
            "tenant",
            "TEN-1",
            "TEN-1",
            "production",
        )

    revision_ids = [row["revision_id"] for row in rows]
    cases = (
        (None, revision_ids),
        (revision_ids[0], revision_ids[1:]),
        (f"{kind.upper()}-MISSING", revision_ids),
    )
    expected_suffix = (
        "AND revision_id IS DISTINCT FROM %s ORDER BY revision_id FOR UPDATE"
    )
    for except_revision_id, expected_changed in cases:
        select_calls: list[tuple[str, Any]] = []

        def handler(sql: str, args: Any) -> _Result:
            if sql.startswith("SELECT "):
                select_calls.append((sql, args))
                return _Result(
                    rows=[
                        dict(row)
                        for row in rows
                        if row["revision_id"] != except_revision_id
                    ]
                )
            if sql.startswith("UPDATE "):
                current = next(row for row in rows if row["revision_id"] == args[1])
                return _Result(
                    {
                        **current,
                        "status": "superseded",
                        "updated_at": args[0],
                        "etag": current["etag"] + 1,
                    }
                )
            raise AssertionError(sql)

        connection = _ScriptedConnection(handler)
        changed = method(
            connection,
            except_revision_id=except_revision_id,
            **method_kwargs,
        )

        assert changed == expected_changed
        assert len(select_calls) == 1
        sql, args = select_calls[0]
        assert sql.endswith(expected_suffix)
        assert "%s IS NULL" not in sql
        assert args == (*expected_prefix, except_revision_id)
        assert sql.count("%s") == len(args)


def test_postgres_event_snapshot_is_one_named_select_and_hashes_utc() -> None:
    created = datetime(2026, 1, 1, tzinfo=timezone.utc)
    body = {
        "aggregateType": "profile",
        "aggregateId": "CPR-ONE",
        "sequence": 1,
        "eventType": "created",
        "actorId": "TEST",
        "payload": {"value": 1},
        "previousHash": "",
        "createdAt": created.isoformat(),
    }
    row = {
        "event_id": "CFE-ONE",
        "aggregate_type": "profile",
        "aggregate_id": "CPR-ONE",
        "sequence": 1,
        "event_type": "created",
        "actor_id": "TEST",
        "payload": {"value": 1},
        "previous_hash": "",
        "event_hash": digest(body),
        "created_at": created,
    }

    def handler(sql: str, _args: Any) -> _Result:
        assert "event_id,aggregate_type" in sql
        return _Result(rows=[row])

    connection = _ScriptedConnection(handler)
    store, _ = _postgres_store(connection)
    snapshot = store.events_with_verification("profile", "CPR-ONE")
    assert snapshot["verification"] == {
        "valid": True,
        "issues": [],
        "eventCount": 1,
        "eventHead": row["event_hash"],
    }
    event_queries = [
        sql
        for sql, _ in connection.calls
        if "FROM configuration.events" in sql
    ]
    assert len(event_queries) == 1


def test_postgres_empty_event_stream_is_serialized_by_a_head_row() -> None:
    head = {
        "next_sequence": 1,
        "etag": 1,
        "updated_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
    }
    inserted: list[tuple[str, Any]] = []

    def handler(sql: str, args: Any) -> _Result:
        if sql.startswith("INSERT INTO configuration.aggregate_heads"):
            return _Result()
        if "FROM configuration.aggregate_heads" in sql:
            return _Result(dict(head))
        if "SELECT sequence,event_hash" in sql:
            return _Result()
        if sql.startswith("UPDATE configuration.aggregate_heads"):
            return _Result(
                {
                    "next_sequence": 2,
                    "etag": 2,
                    "updated_at": args[1],
                }
            )
        if sql.startswith("INSERT INTO configuration.events"):
            inserted.append((sql, args))
            return _Result()
        raise AssertionError(sql)

    connection = _ScriptedConnection(handler)
    event = PostgresConfigurationStore._append_event_tx(
        connection, "profile", "CPR-ONE", "created", "TEST", {"value": 1}
    )
    assert event["sequence"] == 1
    statements = [sql for sql, _ in connection.calls]
    assert statements[0].startswith("INSERT INTO configuration.aggregate_heads")
    assert statements[1].endswith("FOR UPDATE")
    assert "payload,previous_hash" in inserted[0][0]
    assert "payload_json" not in inserted[0][0]


def test_postgres_sequence_reservation_reconciles_legacy_maximum() -> None:
    head = {
        "next_sequence": 1,
        "etag": 4,
        "updated_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
    }
    update_args: list[Any] = []

    def handler(sql: str, args: Any) -> _Result:
        if sql.startswith("INSERT INTO configuration.aggregate_heads"):
            return _Result()
        if "FROM configuration.aggregate_heads" in sql:
            return _Result(dict(head))
        if sql.startswith("UPDATE configuration.aggregate_heads"):
            update_args.extend(args)
            return _Result(
                {"sequence": 7, "etag": 5, "updated_at": args[1]}
            )
        raise AssertionError(sql)

    connection = _ScriptedConnection(handler)
    assert (
        PostgresConfigurationStore._next_sequence(
            connection,
            "profile",
            _aggregate_identity(("rules", "saudi", "global", "*", None, None)),
            minimum_next_sequence=7,
        )
        == 7
    )
    assert update_args[0] == 8
    assert update_args[2] == "profile"
    assert update_args[4] == 4


def _profile_payload(identifier: str, version: str, value: int) -> dict[str, Any]:
    return {
        "profile": {"id": identifier, "version": version, "status": "published"},
        "value": value,
    }


def test_sqlite_exact_tenant_environment_identity_and_event_parity(
    tmp_path: Path,
) -> None:
    store = SQLiteConfigurationStore(tmp_path / "configuration.sqlite3")
    global_revision = store.create_profile(
        family="rules",
        profile_id="saudi",
        version="1.0.0",
        scope="global",
        scope_key="*",
        tenant_id=None,
        environment=None,
        payload=_profile_payload("saudi", "1.0.0", 1),
        status="published",
        created_by="TEST",
    )
    tenant_revision = store.create_profile(
        family="rules",
        profile_id="saudi",
        version="1.0.0",
        scope="tenant",
        scope_key="TEN-1",
        tenant_id="TEN-1",
        environment=None,
        payload=_profile_payload("saudi", "1.0.0", 1),
        status="published",
        created_by="TEST",
    )
    assert global_revision.revision_id != tenant_revision.revision_id
    assert global_revision.sequence == tenant_revision.sequence == 1
    assert (
        store.find_profile_by_content(
            family="rules",
            scope="global",
            scope_key="*",
            tenant_id=None,
            environment=None,
            payload_sha256=global_revision.payload_sha256,
        ).revision_id
        == global_revision.revision_id
    )
    assert store.list_profiles(
        tenant_id=None, environment=None, exact_dimensions=True
    ) == [global_revision]
    assert [
        event["eventType"]
        for event in store.events("profile", global_revision.revision_id)
    ] == ["created", "published"]
    assert store.verify_events("profile", global_revision.revision_id)["valid"] is True


def test_sqlite_publication_supersession_is_atomic_and_bundle_id_is_exact(
    tmp_path: Path,
) -> None:
    store = SQLiteConfigurationStore(tmp_path / "configuration.sqlite3")
    first = store.create_profile(
        family="rules",
        profile_id="saudi",
        version="1.0.0",
        scope="global",
        tenant_id=None,
        payload=_profile_payload("saudi", "1.0.0", 1),
        status="published",
        created_by="TEST",
    )
    second = store.create_profile(
        family="rules",
        profile_id="saudi",
        version="2.0.0",
        scope="global",
        tenant_id=None,
        payload=_profile_payload("saudi", "2.0.0", 2),
        status="published",
        created_by="TEST",
    )
    assert store.get_profile(first.revision_id).status == "superseded"
    assert store.get_profile(second.revision_id).status == "published"
    assert [
        event["eventType"] for event in store.events("profile", first.revision_id)
    ] == ["created", "published", "superseded"]

    components = {"rules": second.revision_id}
    main = store.create_bundle(
        bundle_id="main",
        version="1.0.0",
        mode="full",
        scope="global",
        tenant_id=None,
        component_revisions=components,
        status="published",
        created_by="TEST",
    )
    alternate = store.create_bundle(
        bundle_id="alternate",
        version="1.0.0",
        mode="full",
        scope="global",
        tenant_id=None,
        component_revisions=components,
        status="published",
        created_by="TEST",
    )
    assert main.revision_id != alternate.revision_id
    assert (
        store.find_bundle_by_content(
            bundle_id="main",
            scope="global",
            tenant_id=None,
            content_sha256=main.content_sha256,
        ).revision_id
        == main.revision_id
    )


def test_sqlite_idempotent_create_is_serialized_under_concurrency(
    tmp_path: Path,
) -> None:
    store = SQLiteConfigurationStore(tmp_path / "configuration.sqlite3")
    kwargs = {
        "family": "rules",
        "profile_id": "saudi",
        "version": "1.0.0",
        "scope": "global",
        "tenant_id": None,
        "payload": _profile_payload("saudi", "1.0.0", 1),
        "status": "draft",
        "created_by": "TEST",
    }
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: store.create_profile(**kwargs), range(2)))
    assert len({result.revision_id for result in results}) == 1
    revision_id = results[0].revision_id
    assert [
        event["eventType"] for event in store.events("profile", revision_id)
    ] == ["created"]
    assert store.get_invalidation_epoch() == 2


def test_sqlite_revision_event_and_epoch_roll_back_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SQLiteConfigurationStore(tmp_path / "configuration.sqlite3")

    def fail_event(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("event write failed")

    monkeypatch.setattr(
        SQLiteConfigurationStore, "_append_event_tx", staticmethod(fail_event)
    )
    with pytest.raises(RuntimeError, match="event write failed"):
        store.create_profile(
            family="rules",
            profile_id="saudi",
            version="1.0.0",
            scope="global",
            tenant_id=None,
            payload=_profile_payload("saudi", "1.0.0", 1),
            status="draft",
            created_by="TEST",
        )
    assert store.list_profiles() == []
    assert store.get_invalidation_epoch() == 1


def test_aggregate_identity_is_unambiguous() -> None:
    assert _aggregate_identity(("a:b", "c", None, "")) != _aggregate_identity(
        ("a", "b:c", "", None)
    )
    assert json.loads(_aggregate_identity(("a:b", None, ""))) == ["a:b", None, ""]
    assert canonical(json.loads(_aggregate_identity(("a:b", None, "")))) == (
        _aggregate_identity(("a:b", None, ""))
    )


def test_fake_pool_marks_transaction_rollback() -> None:
    connection = _ScriptedConnection(
        lambda sql, _args: (_ for _ in ()).throw(RuntimeError(sql))
    )
    store, pool = _postgres_store(connection)
    with pytest.raises(RuntimeError):
        store.get_invalidation_epoch()
    assert pool.commits == 0
    assert pool.rollbacks == 1
    assert copy.deepcopy(connection.calls)
