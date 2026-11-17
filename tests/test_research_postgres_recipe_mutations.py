from __future__ import annotations

import inspect
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pytest

from baloot_platform.research import postgres_store as postgres_module
from baloot_platform.research.fenced_occurrence import build_fenced_occurrence
from baloot_platform.research.postgres_store import PostgresResearchStoreV1
from baloot_platform.research.recipe_api import ResearchMutationConflictError
from baloot_platform.research.store import SQLiteResearchStoreV1


_INTENT_HASH = "a" * 64
_RECIPE = {
    "code": "RCP-PG000001",
    "sha256": "b" * 64,
    "status": "published",
}


class _Cursor:
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = rows or []

    def fetchone(self) -> dict[str, Any] | None:
        return self.rows[0] if self.rows else None

    def fetchall(self) -> list[dict[str, Any]]:
        return list(self.rows)


class _Connection:
    def __init__(
        self,
        handler: Callable[[str, Any], list[dict[str, Any]] | None],
    ) -> None:
        self.handler = handler
        self.statements: list[tuple[str, Any]] = []

    def execute(self, statement: str, parameters: Any = None) -> _Cursor:
        sql = " ".join(statement.split())
        self.statements.append((sql, parameters))
        return _Cursor(self.handler(sql, parameters))


def _store(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[str, Any], list[dict[str, Any]] | None],
) -> tuple[PostgresResearchStoreV1, _Connection]:
    connection = _Connection(handler)

    @contextmanager
    def fake_contextual_connection(_pool: object):
        yield connection

    monkeypatch.setattr(
        postgres_module,
        "contextual_connection",
        fake_contextual_connection,
    )
    store = object.__new__(PostgresResearchStoreV1)
    store.pool = object()
    return store, connection


def test_postgres_recipe_mutation_signatures_match_sqlite() -> None:
    for method_name in (
        "get_recipe_mutation_receipt",
        "put_recipe_with_receipt",
        "retire_recipe_with_receipt",
        "list_recipe_page",
        "get_run",
        "list_runs",
    ):
        assert inspect.signature(
            getattr(PostgresResearchStoreV1, method_name)
        ) == inspect.signature(getattr(SQLiteResearchStoreV1, method_name))


def test_postgres_operator_reads_project_receipt_in_database(monkeypatch):
    store, connection = _store(monkeypatch, lambda _sql, _args: [])
    assert store.get_run("RUN-OPERATOR", include_coordinator_evidence=False) is None
    assert store.list_runs(status="completed", limit=1, include_coordinator_evidence=False) == []
    for sql, _args in connection.statements:
        assert "jsonb_typeof(payload_json->'performanceReceipt') = 'object'" in sql
        assert "? 'coordinatorReceipt'" in sql
        assert "THEN payload_json - 'performanceReceipt' ELSE payload_json END" in sql
    assert connection.statements[0][1] == ("RUN-OPERATOR",)
    assert connection.statements[1][1] == ["completed", 1]
    connection.statements.clear()
    store.get_run("RUN-OPERATOR")
    store.list_runs()
    assert all(sql.startswith("SELECT payload_json AS payload_json") for sql, _args in connection.statements)


def test_postgres_recipe_mutation_migration_is_packaged_and_complete() -> None:
    authored = Path("migrations/023_research_recipe_mutations.sql")
    packaged = Path(
        "src/baloot_platform/resources/migrations/"
        "023_research_recipe_mutations.sql"
    )
    assert authored.read_bytes() == packaged.read_bytes()
    sql = authored.read_text(encoding="utf-8")
    assert sql.index("CREATE TABLE IF NOT EXISTS research.recipes") < sql.index(
        "CREATE TABLE IF NOT EXISTS research.recipe_mutation_receipts"
    )
    assert sql.index("CREATE TABLE IF NOT EXISTS research.runs") < sql.index(
        "ADD CONSTRAINT fk_research_audit_traces_run"
    )
    assert "idempotency_key TEXT PRIMARY KEY" in sql
    assert "REFERENCES research.recipes(recipe_code)" in sql
    assert "recipe_code COLLATE \"C\" ASC" in sql
    assert "ix_research_recipes_page" in sql
    assert "ix_research_recipes_status_page" in sql
    assert "ck_research_recipe_receipt_mutation_kind" in sql
    assert "CHECK(mutation_kind IN ('create','retire'))" in sql
    assert "ck_research_recipe_receipt_intent_hash" in sql
    assert "CHECK(request_intent_hash ~ '^[0-9a-f]{64}$')" in sql
    assert "ck_research_recipe_receipt_response_status" in sql
    assert "mutation_kind='create' AND response_status=201" in sql
    assert "mutation_kind='retire' AND response_status=200" in sql
    assert "conrelid='research.audit_traces'::regclass" in sql
    assert "confrelid='research.runs'::regclass" in sql
    assert "existing_fk.conkey" in sql
    assert "existing_fk.confkey" in sql
    assert "FOREIGN KEY(run_code) REFERENCES research.runs(run_code)" in sql
    assert "ON DELETE RESTRICT" in sql

    bootstrap = inspect.getsource(PostgresResearchStoreV1._bootstrap_schema)
    assert "research.recipe_mutation_receipts" in bootstrap
    assert "ck_research_recipe_receipt_response_status" in bootstrap
    assert "fk_research_audit_traces_run" in bootstrap
    assert "pg_constraint" in bootstrap
    assert "conrelid='research.audit_traces'::regclass" in bootstrap
    assert "confrelid='research.runs'::regclass" in bootstrap


def test_postgres_create_receipt_is_atomic_and_replays_globally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe_row: dict[str, Any] | None = None
    receipt_row: dict[str, Any] | None = None

    def handler(sql: str, parameters: Any) -> list[dict[str, Any]] | None:
        nonlocal recipe_row, receipt_row
        if "pg_advisory_xact_lock" in sql:
            return []
        if "FROM research.recipe_mutation_receipts" in sql:
            return [] if receipt_row is None else [receipt_row]
        if "SELECT status,payload_json FROM research.recipes" in sql:
            return [] if recipe_row is None else [recipe_row]
        if sql.startswith("INSERT INTO research.recipes"):
            assert "ON CONFLICT(recipe_code) DO NOTHING" in sql
            recipe_row = {"status": parameters[2], "payload_json": parameters[3]}
            return []
        if sql.startswith("INSERT INTO research.recipe_mutation_receipts"):
            receipt_row = {
                "idempotency_key": parameters[0],
                "mutation_kind": parameters[1],
                "request_intent_hash": parameters[2],
                "recipe_code": parameters[3],
                "response_status": parameters[4],
                "response_json": parameters[5],
                "created_at": datetime(2026, 8, 9, tzinfo=timezone.utc),
            }
            return []
        raise AssertionError(sql)

    store, connection = _store(monkeypatch, handler)
    expected = {**_RECIPE, "lifecycleStatus": "published"}
    assert store.put_recipe_with_receipt(
        _RECIPE,
        idempotency_key="create-recipe-1",
        request_intent_hash=_INTENT_HASH,
    ) == expected
    assert store.put_recipe_with_receipt(
        _RECIPE,
        idempotency_key="create-recipe-1",
        request_intent_hash=_INTENT_HASH,
    ) == expected
    assert store.get_recipe_mutation_receipt("create-recipe-1") == {
        "schemaVersion": 1,
        "idempotencyKey": "create-recipe-1",
        "mutationKind": "create",
        "requestIntentHash": _INTENT_HASH,
        "recipeCode": _RECIPE["code"],
        "responseStatus": 201,
        "response": expected,
        "createdAt": "2026-08-09T00:00:00.000000+00:00",
    }
    lock_calls = [
        parameters
        for sql, parameters in connection.statements
        if "pg_advisory_xact_lock" in sql
    ]
    assert lock_calls == [("create-recipe-1",), ("create-recipe-1",)]
    assert sum(
        sql.startswith("INSERT INTO research.recipe_mutation_receipts")
        for sql, _ in connection.statements
    ) == 1
    with pytest.raises(ResearchMutationConflictError):
        store.retire_recipe_with_receipt(
            _RECIPE["code"],
            idempotency_key="create-recipe-1",
            request_intent_hash=_INTENT_HASH,
        )


def test_postgres_retiring_an_already_retired_recipe_keeps_its_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(sql: str, parameters: Any) -> list[dict[str, Any]] | None:
        if "pg_advisory_xact_lock" in sql:
            return []
        if "FROM research.recipe_mutation_receipts" in sql:
            return []
        if "SELECT status,payload_json FROM research.recipes" in sql:
            return [{"status": "retired", "payload_json": json.dumps(_RECIPE)}]
        if sql.startswith("INSERT INTO research.recipe_mutation_receipts"):
            assert parameters[1] == "retire"
            assert parameters[4] == 200
            return []
        raise AssertionError(sql)

    store, connection = _store(monkeypatch, handler)
    assert store.retire_recipe(_RECIPE["code"]) == {
        **_RECIPE,
        "lifecycleStatus": "retired",
    }
    assert store.retire_recipe_with_receipt(
        _RECIPE["code"],
        idempotency_key="retire-recipe-1",
        request_intent_hash=_INTENT_HASH,
    ) == {**_RECIPE, "lifecycleStatus": "retired"}
    assert not any(
        sql.startswith("UPDATE research.recipes")
        for sql, _ in connection.statements
    )


def test_postgres_recipe_page_uses_stable_keyset_order_and_limit_plus_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_at = datetime(2026, 8, 9, 12, tzinfo=timezone.utc)
    rows = [
        {
            "recipe_code": f"RCP-PG00000{index}",
            "status": "published",
            "payload_json": json.dumps({**_RECIPE, "code": f"RCP-PG00000{index}"}),
            "created_at": created_at,
        }
        for index in range(1, 4)
    ]

    def handler(sql: str, parameters: Any) -> list[dict[str, Any]] | None:
        assert "recipe_code COLLATE \"C\" > %s" in sql
        assert "ORDER BY created_at DESC,recipe_code COLLATE \"C\" ASC" in sql
        assert parameters[-1] == 3
        return rows

    store, _ = _store(monkeypatch, handler)
    page = store.list_recipe_page(
        status="published",
        limit=2,
        after_created_at="2026-08-10T00:00:00.000000+00:00",
        after_recipe_code="RCP-PG000000",
    )
    assert [item["recipe"]["code"] for item in page["items"]] == [
        "RCP-PG000001",
        "RCP-PG000002",
    ]
    assert page["nextAnchor"] == {
        "createdAt": "2026-08-09T12:00:00.000000+00:00",
        "recipeCode": "RCP-PG000002",
    }


def test_postgres_aggregate_stats_never_filters_runs_by_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(sql: str, _parameters: Any) -> list[dict[str, Any]] | None:
        if "FROM research.runs" in sql:
            return [{"status": "completed", "count": 1}]
        if "GROUP BY disposition" in sql:
            return [{"disposition": "qualified", "count": 1}]
        if "GROUP BY truth_label" in sql:
            return [{"truth_label": "verified", "count": 1}]
        if "AVG(score)" in sql:
            return [{"count": 1, "average_score": 0.8, "average_confidence": 0.9}]
        if "FROM research.semantic_disagreements" in sql:
            return []
        raise AssertionError(sql)

    store, connection = _store(monkeypatch, handler)
    stats = store.aggregate_stats(run_code="RUN-PG0000001")
    assert stats["runStatus"] == {"completed": 1}
    run_sql = next(sql for sql, _ in connection.statements if "FROM research.runs" in sql)
    assert "admitted" not in run_sql
    admitted_sql = [
        sql
        for sql, _ in connection.statements
        if "FROM research.results" in sql
        or "FROM research.semantic_disagreements" in sql
    ]
    assert admitted_sql
    assert all("admitted=true" in sql for sql in admitted_sql)


def test_postgres_sequential_receipt_concurrent_winner_replays_atomically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = {
        "schemaVersion": 1,
        "rootSeed": 41,
        "sha256": "c" * 64,
    }
    monkeypatch.setattr(
        postgres_module,
        "build_sequential_search_receipt_bundle",
        lambda _receipt, *, seed: dict(bundle),
    )
    retained_bundle: dict[str, Any] | None = None
    retained_occurrence: dict[str, Any] | None = None
    created_at = datetime(2026, 8, 9, tzinfo=timezone.utc)

    def handler(sql: str, parameters: Any) -> list[dict[str, Any]] | None:
        nonlocal retained_bundle, retained_occurrence
        if "FROM research.shards" in sql:
            return [{
                "run_code": "RUN-PG-FENCED",
                "spec_json": json.dumps({"seedStart": 41, "seedCount": 1}),
                "status": "running",
                "attempt_count": 1,
                "fencing_token": 7,
                "lease_owner": "worker-pg",
                "lease_expires_at": datetime(2027, 8, 9, tzinfo=timezone.utc),
            }]
        if sql.startswith(
            "INSERT INTO research.sequential_search_receipt_bundles"
        ):
            assert "ON CONFLICT DO NOTHING" in sql
            if retained_bundle is not None:
                return []
            retained_bundle = {
                "receipt_hash": parameters[0],
                "seed": parameters[1],
                "receipt_json": parameters[2],
            }
            return [retained_bundle]
        if sql.startswith(
            "SELECT receipt_hash,seed,receipt_json FROM "
            "research.sequential_search_receipt_bundles"
        ):
            assert "FOR UPDATE" in sql
            return [retained_bundle] if retained_bundle is not None else []
        if sql.startswith(
            "INSERT INTO research.sequential_search_receipt_occurrences"
        ):
            assert "ON CONFLICT DO NOTHING" in sql
            if retained_occurrence is not None:
                return []
            retained_occurrence = {
                "occurrence_hash": parameters[0],
                "receipt_hash": parameters[1],
                "run_code": parameters[2],
                "shard_id": parameters[3],
                "attempt_count": parameters[4],
                "fencing_token": parameters[5],
                "seed": parameters[6],
                "admitted": False,
                "created_at": created_at,
            }
            return [retained_occurrence]
        if sql.startswith(
            "SELECT occurrence_hash,receipt_hash,run_code,shard_id,"
        ):
            assert "FOR UPDATE" in sql
            return [retained_occurrence] if retained_occurrence is not None else []
        if sql.startswith("SELECT occurrence.*,bundle.receipt_json"):
            if retained_occurrence is None or retained_bundle is None:
                return []
            return [{
                **retained_occurrence,
                "receipt_json": retained_bundle["receipt_json"],
            }]
        raise AssertionError(sql)

    store, connection = _store(monkeypatch, handler)
    arguments = {
        "run_code": "RUN-PG-FENCED",
        "shard_id": "RSH-PG-FENCED",
        "attempt_count": 1,
        "fencing_token": 7,
        "worker_id": "worker-pg",
        "seed": 41,
    }
    first = store.put_sequential_search_receipt({}, **arguments)
    second = store.put_sequential_search_receipt({}, **arguments)
    assert first == second
    assert first["receiptHash"] == bundle["sha256"]
    assert sum(
        sql.startswith(
            "INSERT INTO research.sequential_search_receipt_bundles"
        )
        for sql, _ in connection.statements
    ) == 2
    assert sum(
        sql.startswith(
            "INSERT INTO research.sequential_search_receipt_occurrences"
        )
        for sql, _ in connection.statements
    ) == 2
    assert retained_occurrence is not None
    retained_occurrence["seed"] = 42
    with pytest.raises(RuntimeError, match="immutable"):
        store.put_sequential_search_receipt({}, **arguments)


def test_postgres_generic_occurrence_conflict_verifies_the_full_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    occurrence = build_fenced_occurrence(
        kind="research_result",
        semantic_hash="a" * 64,
        run_code="RUN-PG-FENCED",
        shard_id="RSH-PG-FENCED",
        attempt_count=2,
        fencing_token=9,
    )

    def handler(sql: str, _parameters: Any) -> list[dict[str, Any]]:
        if sql.startswith("INSERT INTO research.result_occurrences"):
            return []
        if sql.startswith("SELECT occurrence_hash,result_code,run_code,shard_id"):
            return [{
                "occurrence_hash": occurrence["occurrenceHash"],
                "result_code": "LAF-PG-FENCED",
                "run_code": occurrence["runCode"],
                "shard_id": "RSH-WRONG",
                "attempt_count": occurrence["attemptCount"],
                "fencing_token": occurrence["fencingToken"],
            }]
        raise AssertionError(sql)

    store, connection = _store(monkeypatch, handler)
    with pytest.raises(RuntimeError, match="immutable"):
        store._put_fenced_occurrence(
            connection,
            table="research.result_occurrences",
            semantic_column="result_code",
            semantic_id="LAF-PG-FENCED",
            occurrence=occurrence,
        )


def test_postgres_shard_creation_and_unsharded_validation_share_run_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statements: list[str] = []

    def handler(sql: str, _parameters: Any) -> list[dict[str, Any]]:
        statements.append(sql)
        if sql.startswith("SELECT run_code FROM research.runs"):
            assert sql.endswith("FOR UPDATE")
            return [{"run_code": "RUN-PG-LOCKED"}]
        if sql.startswith("SELECT 1 FROM research.shards"):
            return []
        if sql.startswith("SELECT spec_json FROM research.shards"):
            return []
        if sql.startswith("INSERT INTO research.shards"):
            return []
        if sql.startswith("SELECT * FROM research.shards"):
            return []
        raise AssertionError(sql)

    store, connection = _store(monkeypatch, handler)
    store._validate_unsharded_write(connection, "RUN-PG-LOCKED")
    assert statements[:2] == [
        "SELECT run_code FROM research.runs WHERE run_code=%s FOR UPDATE",
        "SELECT 1 FROM research.shards WHERE run_code=%s LIMIT 1",
    ]
    statements.clear()
    store.create_shards(({
        "shardId": "RSH-PG-LOCKED",
        "sha256": "b" * 64,
        "runCode": "RUN-PG-LOCKED",
        "shardIndex": 0,
    },))
    assert statements[0] == (
        "SELECT run_code FROM research.runs WHERE run_code=%s FOR UPDATE"
    )


def test_postgres_sequential_receipt_rejects_expired_lease_before_cas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        postgres_module,
        "build_sequential_search_receipt_bundle",
        lambda _receipt, *, seed: {
            "schemaVersion": 1,
            "rootSeed": seed,
            "sha256": "d" * 64,
        },
    )

    def handler(sql: str, _parameters: Any) -> list[dict[str, Any]] | None:
        if "FROM research.shards" in sql:
            assert "FOR UPDATE" in sql
            return [{
                "run_code": "RUN-PG-FENCED",
                "spec_json": json.dumps({"seedStart": 41, "seedCount": 1}),
                "status": "running",
                "attempt_count": 1,
                "fencing_token": 7,
                "lease_owner": "worker-pg",
                "lease_expires_at": datetime(2020, 8, 9, tzinfo=timezone.utc),
            }]
        raise AssertionError(sql)

    store, connection = _store(monkeypatch, handler)
    with pytest.raises(RuntimeError, match="stale"):
        store.put_sequential_search_receipt(
            {},
            run_code="RUN-PG-FENCED",
            shard_id="RSH-PG-FENCED",
            attempt_count=1,
            fencing_token=7,
            worker_id="worker-pg",
            seed=41,
        )
    assert not any(
        sql.startswith("INSERT INTO research.sequential_search")
        for sql, _ in connection.statements
    )


def test_postgres_expired_lease_cannot_heartbeat_or_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(sql: str, _parameters: Any) -> list[dict[str, Any]]:
        if sql == "SELECT * FROM research.shards WHERE shard_id=%s FOR UPDATE":
            return [{
                "status": "running",
                "lease_owner": "worker-pg",
                "fencing_token": 7,
                "lease_expires_at": datetime(2020, 8, 9, tzinfo=timezone.utc),
            }]
        raise AssertionError(sql)

    store, connection = _store(monkeypatch, handler)
    with pytest.raises(RuntimeError, match="stale"):
        store.heartbeat_shard(
            "RSH-PG-FENCED",
            worker_id="worker-pg",
            fencing_token=7,
            lease_seconds=30,
            now=datetime(2026, 8, 9, tzinfo=timezone.utc),
        )
    with pytest.raises(RuntimeError, match="stale"):
        store.complete_shard(
            "RSH-PG-FENCED",
            worker_id="worker-pg",
            fencing_token=7,
            result={"attemptedSeeds": 1},
    )
    assert len(connection.statements) == 2
    assert all(sql.startswith("SELECT * FROM research.shards") for sql, _ in connection.statements)
