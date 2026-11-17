from __future__ import annotations

import hashlib
import inspect
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from baloot_platform.challenge.constraint_deal_search import (
    DealConstraintSpecV1,
    OrToolsCpSatDealGeneratorV1,
)
from baloot_platform.config import build_bundle
from baloot_platform.core.enums import Seat
from baloot_platform.research.contracts import canonical_sha256
from baloot_platform.research.fenced_occurrence import (
    build_fenced_occurrence,
    normalize_fenced_attempt,
)
from baloot_platform.research.postgres_store import PostgresResearchStoreV1
from baloot_platform.research.result_library import stored_result_payload_hash_v1
from baloot_platform.research.sequential_candidate_search import (
    SequentialCandidateSearchReceiptV1,
    run_sequential_candidate_search_v1,
)
from baloot_platform.research.sharding import partition_research_shards
from baloot_platform.research.store import SQLiteResearchStoreV1


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


_RECEIPT_RULES = build_bundle().rules
_RECEIPT_SPEC = DealConstraintSpecV1.build({
    "schemaVersion": 1,
    "id": "storage_test_v1",
    "revision": "1",
    "enabled": True,
    "backend": "ortools_cp_sat_v1",
    "assumedBuyer": None,
    "purchaseSuits": [],
    "initialRequiredCards": {},
    "finalRequiredCards": {},
    "finalForbiddenCards": {},
    "initialSuitCountRanges": {},
    "finalSuitCountRanges": {},
    "projectPattern": None,
    "maximumSolutions": 2,
    "maximumConflicts": 1000,
    "safetyTimeoutSeconds": 1.0,
    "solverWorkers": 1,
    "seedOffset": 0,
    "tags": ["engine_validation_required"],
})


class _EmptyReceiptGenerator:
    descriptor = OrToolsCpSatDealGeneratorV1.descriptor

    @staticmethod
    def generate_next(*_args, **_kwargs):
        return None

    @staticmethod
    def probe_continuation(
        spec,
        *,
        root_seed,
        dealer,
        rules,
        rejected_candidates,
        **_kwargs,
    ):
        assert rejected_candidates == ()
        model_hash = _hash("storage-receipt-model")
        backend_version = "9.15.6755"
        no_good_cut_type = "complete_deck_assignment_v1"
        search_identity = OrToolsCpSatDealGeneratorV1.search_identity_document(
            spec=spec,
            model_hash=model_hash,
            root_seed=root_seed,
            dealer=dealer,
            rules=rules,
            backend_version=backend_version,
            no_good_cut_type=no_good_cut_type,
        )
        return OrToolsCpSatDealGeneratorV1._build_continuation_receipt(
            profile_hash=spec.sha256,
            model_hash=model_hash,
            backend_version=backend_version,
            solver_status="infeasible",
            solution_ordinal=0,
            search_identity_hash=search_identity["sha256"],
            preceding_no_good_cut_hashes=(),
            no_good_cut_type=no_good_cut_type,
            exhausted=True,
            witness=None,
            structural_objective_score=0,
        )


def _receipt(seed: int) -> dict[str, object]:
    result = run_sequential_candidate_search_v1(
        _EmptyReceiptGenerator(),
        _RECEIPT_SPEC,
        root_seed=int(seed),
        dealer=Seat.SOUTH,
        rules=_RECEIPT_RULES,
        evaluate_candidate=lambda *_args: pytest.fail(
            "empty storage receipt may not evaluate a candidate"
        ),
    )
    return result.receipt.verify().to_dict()


def _store_with_claim(tmp_path, *, now: datetime | None = None):
    store = SQLiteResearchStoreV1(tmp_path / "research.sqlite3")
    recipe = {
        "code": "RCP-FENCED01",
        "sha256": _hash("recipe"),
        "status": "published",
    }
    run = {
        "runCode": "RUN-FENCED0001",
        "runHash": _hash("run"),
        "recipeCode": recipe["code"],
        "recipeHash": recipe["sha256"],
        "status": "running",
        "progress": 0.0,
        "events": [],
    }
    store.put_recipe(recipe)
    store.create_run(run)
    spec = partition_research_shards(
        run_code=run["runCode"],
        run_hash=run["runHash"],
        recipe_hash=recipe["sha256"],
        seed_start=41,
        seed_count=1,
        seeds_per_shard=1,
    )[0]
    store.create_shards((spec.to_dict(),))
    claim = store.claim_shard(
        run["runCode"], worker_id="worker-1", lease_seconds=30, now=now
    )
    assert claim is not None
    return store, run, claim


def _result(run_code: str, shard_id: str) -> dict[str, object]:
    return {
        "resultCode": "LAF-FENCED",
        "resultHash": _hash("same-deterministic-result"),
        "runCode": run_code,
        "truthLabel": "provisional",
        "confidence": 0.5,
        "score": 50,
        "disposition": "rejected",
        "provenance": {"researchShardId": shard_id},
    }


def _disagreement(run_code: str) -> dict[str, object]:
    return {
        "disagreementId": "DSG-FENCED",
        "sha256": _hash("same-deterministic-disagreement"),
        "runCode": run_code,
        "category": "compiler_parity_defect",
        "status": "open",
    }


def test_fenced_store_contracts_match_and_migration_024_is_packaged() -> None:
    for method_name in (
        "put_result",
        "put_disagreement",
        "put_sequential_search_receipt",
        "count_sequential_search_receipts",
        "iter_admitted_sequential_search_receipts",
        "list_sequential_search_receipts",
    ):
        assert inspect.signature(
            getattr(SQLiteResearchStoreV1, method_name)
        ) == inspect.signature(getattr(PostgresResearchStoreV1, method_name))

    authored = Path("migrations/024_research_fenced_search_receipts.sql")
    packaged = Path(
        "src/baloot_platform/resources/migrations/"
        "024_research_fenced_search_receipts.sql"
    )
    assert authored.read_bytes() == packaged.read_bytes()
    sql = authored.read_text(encoding="utf-8")
    for required in (
        "ALTER TABLE research.results",
        "ALTER TABLE research.semantic_disagreements",
        "ADD COLUMN IF NOT EXISTS attempt_count INTEGER",
        "ADD COLUMN IF NOT EXISTS fencing_token BIGINT",
        "CREATE TABLE IF NOT EXISTS research.comparison_campaigns",
        "REFERENCES research.recipes(recipe_code)",
        "research.result_occurrences",
        "research.semantic_disagreement_occurrences",
        "research.sequential_search_receipt_bundles",
        "research.sequential_search_receipt_occurrences",
        "UNIQUE(run_code,shard_id,attempt_count,fencing_token,seed)",
        "FOREIGN KEY(shard_id,run_code)",
        "FOREIGN KEY(receipt_hash,seed)",
        "ck_research_results_legacy_fence",
        "ck_research_disagreements_legacy_fence",
        "ux_research_shards_identity_run",
        "ux_research_receipt_bundle_hash_seed",
    ):
        assert required in sql

    bootstrap = inspect.getsource(PostgresResearchStoreV1._bootstrap_schema)
    assert "research.comparison_campaigns" in bootstrap
    assert "research.sequential_search_receipt_bundles" in bootstrap
    assert "research.result_occurrences" in bootstrap
    assert "research.semantic_disagreement_occurrences" in bootstrap
    receipt_source = inspect.getsource(
        PostgresResearchStoreV1.put_sequential_search_receipt
    )
    assert receipt_source.count("ON CONFLICT DO NOTHING") == 2
    assert "FOR UPDATE" in receipt_source
    assert "ux_research_search_receipts_admitted_seed" in bootstrap
    for method in (
        PostgresResearchStoreV1.put_result,
        PostgresResearchStoreV1.put_disagreement,
    ):
        source = inspect.getsource(method)
        assert "ON CONFLICT DO NOTHING" in source
        assert "worker_id" in source
    for store_type in (SQLiteResearchStoreV1, PostgresResearchStoreV1):
        list_source = inspect.getsource(
            store_type.list_sequential_search_receipts
        )
        assert list_source.count(".execute(") == 1
        assert "WITH filtered AS" in list_source
        snapshot_source = inspect.getsource(
            store_type.iter_admitted_sequential_search_receipts
        )
        assert "fetchmany" in snapshot_source
        assert "fetchall" not in snapshot_source
        assert "occurrence.seed ASC,occurrence.occurrence_hash ASC" in snapshot_source
        assert "occurrence.run_code ASC,occurrence.seed ASC" in snapshot_source
    postgres_snapshot_source = inspect.getsource(
        PostgresResearchStoreV1.iter_admitted_sequential_search_receipts
    )
    assert "READ ONLY" in postgres_snapshot_source
    assert "REPEATABLE READ" in postgres_snapshot_source
    assert "cursor(name=" in postgres_snapshot_source


class _FakePostgresSnapshotCursor:
    def __init__(self, rows):
        self.rows = list(rows)
        self.position = 0
        self.itersize = None
        self.executed = []
        self.fetch_sizes = []
        self.closed = False
        self.exit_type = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, _exc, _traceback):
        self.exit_type = exc_type
        self.closed = True
        return False

    def execute(self, sql, args):
        self.executed.append((sql, list(args)))
        return self

    def fetchmany(self, size):
        self.fetch_sizes.append(size)
        batch = self.rows[self.position:self.position + size]
        self.position += len(batch)
        return batch


class _FakePostgresSnapshotConnection:
    def __init__(self, rows):
        self.rows = rows
        self.executed = []
        self.cursor_names = []
        self.cursors = []
        self.transaction_exit_type = None
        self.connection_exit_type = None

    class _Transaction:
        def __init__(self, connection):
            self.connection = connection

        def __enter__(self):
            return self

        def __exit__(self, exc_type, _exc, _traceback):
            self.connection.transaction_exit_type = exc_type
            return False

    def transaction(self):
        return self._Transaction(self)

    def execute(self, sql, args=None):
        self.executed.append((sql, args))
        return self

    def cursor(self, *, name):
        self.cursor_names.append(name)
        cursor = _FakePostgresSnapshotCursor(self.rows)
        self.cursors.append(cursor)
        return cursor


class _FakePostgresSnapshotPool:
    def __init__(self, rows):
        self.connection_value = _FakePostgresSnapshotConnection(rows)

    class _ConnectionContext:
        def __init__(self, connection):
            self.connection = connection

        def __enter__(self):
            return self.connection

        def __exit__(self, exc_type, _exc, _traceback):
            self.connection.connection_exit_type = exc_type
            return False

    def connection(self):
        return self._ConnectionContext(self.connection_value)


def _postgres_snapshot_row(*, admitted=True):
    receipt = _receipt(41)
    occurrence = build_fenced_occurrence(
        kind="sequential_search_receipt",
        semantic_hash=receipt["sha256"],
        run_code="RUN-FENCED0001",
        shard_id="RSH-FENCED-SNAPSHOT",
        attempt_count=1,
        fencing_token=1,
        seed=41,
    )
    return {
        "occurrence_hash": occurrence["occurrenceHash"],
        "receipt_hash": receipt["sha256"],
        "run_code": "RUN-FENCED0001",
        "shard_id": "RSH-FENCED-SNAPSHOT",
        "attempt_count": 1,
        "fencing_token": 1,
        "seed": 41,
        "admitted": admitted,
        "created_at": datetime(2026, 8, 9, tzinfo=timezone.utc),
        "receipt_json": receipt,
    }


def test_postgres_receipt_snapshot_uses_named_read_only_cursor_and_cleans_up():
    pool = _FakePostgresSnapshotPool((_postgres_snapshot_row(),))
    store = object.__new__(PostgresResearchStoreV1)
    store.pool = pool

    with store.iter_admitted_sequential_search_receipts(
        run_code="RUN-FENCED0001", batch_size=1
    ) as rows:
        projected = list(rows)

    connection = pool.connection_value
    cursor = connection.cursors[0]
    assert projected[0]["seed"] == 41
    assert projected[0]["admitted"] is True
    assert connection.executed[0][0] == (
        "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"
    )
    assert len([sql for sql, _args in connection.executed if "set_config" in sql]) == 6
    assert len(connection.cursor_names) == 1
    assert cursor.itersize == 1
    assert cursor.fetch_sizes == [1, 1]
    assert "ORDER BY occurrence.seed ASC,occurrence.occurrence_hash ASC" in (
        cursor.executed[0][0]
    )
    assert cursor.executed[0][1] == ["RUN-FENCED0001"]
    assert cursor.closed is True
    assert cursor.exit_type is None
    assert connection.transaction_exit_type is None
    assert connection.connection_exit_type is None


def test_postgres_receipt_snapshot_cleans_up_on_validation_error_and_cancellation():
    invalid_pool = _FakePostgresSnapshotPool(
        (_postgres_snapshot_row(admitted=1),)
    )
    store = object.__new__(PostgresResearchStoreV1)
    store.pool = invalid_pool
    with pytest.raises(ValueError, match="admitted flag"):
        with store.iter_admitted_sequential_search_receipts(batch_size=1) as rows:
            next(rows)
    invalid_connection = invalid_pool.connection_value
    assert invalid_connection.cursors[0].closed is True
    assert invalid_connection.cursors[0].exit_type is ValueError
    assert invalid_connection.transaction_exit_type is ValueError
    assert invalid_connection.connection_exit_type is ValueError

    cancelled_pool = _FakePostgresSnapshotPool((_postgres_snapshot_row(),))
    store.pool = cancelled_pool
    with pytest.raises(KeyboardInterrupt):
        with store.iter_admitted_sequential_search_receipts(batch_size=1) as rows:
            next(rows)
            raise KeyboardInterrupt
    cancelled_connection = cancelled_pool.connection_value
    assert cancelled_connection.cursors[0].closed is True
    assert cancelled_connection.cursors[0].exit_type is KeyboardInterrupt
    assert cancelled_connection.transaction_exit_type is KeyboardInterrupt
    assert cancelled_connection.connection_exit_type is KeyboardInterrupt


@pytest.mark.parametrize("invalid", [True, 1.0, "1"])
def test_fenced_attempt_identity_rejects_coercive_integer_aliases(invalid) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        normalize_fenced_attempt(
            shard_id="RSH-STRICT",
            attempt_count=invalid,
            fencing_token=1,
            worker_id="worker",
        )
    with pytest.raises(ValueError, match="positive integer"):
        normalize_fenced_attempt(
            shard_id="RSH-STRICT",
            attempt_count=1,
            fencing_token=invalid,
            worker_id="worker",
        )


@pytest.mark.parametrize("invalid", [True, 41.0, "41"])
def test_sequential_receipt_store_contract_rejects_coercive_aliases(
    tmp_path,
    invalid,
) -> None:
    sqlite_store, run, claim = _store_with_claim(tmp_path)
    postgres_store = object.__new__(PostgresResearchStoreV1)
    receipt = _receipt(41)
    try:
        for store in (sqlite_store, postgres_store):
            with pytest.raises(ValueError, match="seed"):
                store.put_sequential_search_receipt(
                    receipt,
                    run_code=run["runCode"],
                    shard_id=claim["shardId"],
                    attempt_count=claim["attemptCount"],
                    fencing_token=claim["fencingToken"],
                    worker_id="worker-1",
                    seed=invalid,
                )
            admitted_alias = 1 if type(invalid) is bool else invalid
            with pytest.raises(ValueError, match="admitted_only"):
                store.count_sequential_search_receipts(
                    admitted_only=admitted_alias
                )
            with pytest.raises(ValueError):
                store.list_sequential_search_receipts(offset=invalid)
            with pytest.raises(ValueError):
                store.list_sequential_search_receipts(limit=invalid)
            with pytest.raises(ValueError):
                with store.iter_admitted_sequential_search_receipts(
                    batch_size=invalid
                ):
                    pytest.fail("invalid snapshot input must fail before iteration")
    finally:
        sqlite_store.close()


def test_sequential_receipt_store_contract_rejects_out_of_bounds_pages_and_batches(
    tmp_path,
) -> None:
    sqlite_store, _run, _claim = _store_with_claim(tmp_path)
    postgres_store = object.__new__(PostgresResearchStoreV1)
    try:
        for store in (sqlite_store, postgres_store):
            for offset in (-1, 1 << 63):
                with pytest.raises(ValueError, match="offset"):
                    store.list_sequential_search_receipts(offset=offset)
            for limit in (0, 1001):
                with pytest.raises(ValueError, match="limit"):
                    store.list_sequential_search_receipts(limit=limit)
            for batch_size in (0, 1001):
                with pytest.raises(ValueError, match="batch size"):
                    with store.iter_admitted_sequential_search_receipts(
                        batch_size=batch_size
                    ):
                        pytest.fail("invalid snapshot input must fail before iteration")
    finally:
        sqlite_store.close()


def test_sqlite_exact_fence_admission_and_deterministic_retry(tmp_path) -> None:
    store, run, first = _store_with_claim(tmp_path)
    result = _result(run["runCode"], first["shardId"])
    disagreement = _disagreement(run["runCode"])
    receipt = _receipt(41)
    try:
        with pytest.raises(ValueError, match="explicit attempt_count"):
            store.put_result(result)
        with pytest.raises(ValueError, match="explicit attempt_count"):
            store.put_disagreement(
                disagreement,
                shard_id=first["shardId"],
            )
        with pytest.raises(ValueError, match="explicit current lease"):
            store.put_result({**result, "provenance": {}})
        with pytest.raises(ValueError, match="explicit current lease"):
            store.put_disagreement(disagreement)

        first_receipt = store.put_sequential_search_receipt(
            receipt,
            run_code=run["runCode"],
            shard_id=first["shardId"],
            attempt_count=first["attemptCount"],
            fencing_token=first["fencingToken"],
            worker_id="worker-1",
            seed=41,
        )
        assert store.put_sequential_search_receipt(
            receipt,
            run_code=run["runCode"],
            shard_id=first["shardId"],
            attempt_count=first["attemptCount"],
            fencing_token=first["fencingToken"],
            worker_id="worker-1",
            seed=41,
        ) == first_receipt
        assert first_receipt["receiptHash"] == receipt["sha256"]
        assert first_receipt["admitted"] is False

        store.put_result(
            result,
            attempt_count=first["attemptCount"],
            fencing_token=first["fencingToken"],
            worker_id="worker-1",
        )
        store.put_disagreement(
            disagreement,
            shard_id=first["shardId"],
            attempt_count=first["attemptCount"],
            fencing_token=first["fencingToken"],
            worker_id="worker-1",
        )
        for table, identity_column, identity in (
            ("research_results", "result_code", result["resultCode"]),
            (
                "semantic_disagreements",
                "disagreement_id",
                disagreement["disagreementId"],
            ),
        ):
            with pytest.raises(sqlite3.IntegrityError):
                store._db.execute(
                    f"UPDATE {table} SET fencing_token=NULL WHERE {identity_column}=?",
                    (identity,),
                )
            store._db.rollback()
        receipt_foreign_keys = store._db.execute(
            "PRAGMA foreign_key_list(sequential_search_receipt_occurrences)"
        ).fetchall()
        receipt_pairs: dict[int, set[tuple[str, str]]] = {}
        for row in receipt_foreign_keys:
            receipt_pairs.setdefault(int(row["id"]), set()).add(
                (str(row["from"]), str(row["to"]))
            )
        assert {("receipt_hash", "receipt_hash"), ("seed", "seed")} in (
            receipt_pairs.values()
        )
        assert {("shard_id", "shard_id"), ("run_code", "run_code")} in (
            receipt_pairs.values()
        )
        semantic_before_retry = store._db.execute(
            """SELECT payload_json,attempt_count,fencing_token
               FROM research_results WHERE result_code=?""",
            (result["resultCode"],),
        ).fetchone()
        disagreement_before_retry = store._db.execute(
            """SELECT payload_json,attempt_count,fencing_token
               FROM semantic_disagreements WHERE disagreement_id=?""",
            (disagreement["disagreementId"],),
        ).fetchone()
        assert store.count_sequential_search_receipts(
            run_code=run["runCode"]
        ) == 0
        assert store.count_sequential_search_receipts(
            run_code=run["runCode"], admitted_only=False
        ) == 1
        traced: list[str] = []
        store._db.set_trace_callback(traced.append)
        page = store.list_sequential_search_receipts(
            run_code=run["runCode"], admitted_only=False
        )
        store._db.set_trace_callback(None)
        assert page["total"] == 1 and page["items"] == [first_receipt]
        assert len([sql for sql in traced if sql.lstrip().startswith("WITH filtered")]) == 1

        store.fail_shard(
            first["shardId"],
            worker_id="worker-1",
            fencing_token=first["fencingToken"],
            error={"category": "transient"},
            retryable=True,
        )
        second = store.claim_shard(
            run["runCode"], worker_id="worker-2", lease_seconds=30
        )
        assert second is not None
        assert second["attemptCount"] == first["attemptCount"] + 1
        assert second["fencingToken"] == first["fencingToken"] + 1

        with pytest.raises(RuntimeError, match="stale"):
            store.put_sequential_search_receipt(
                receipt,
                run_code=run["runCode"],
                shard_id=first["shardId"],
                attempt_count=first["attemptCount"],
                fencing_token=first["fencingToken"],
                worker_id="worker-1",
                seed=41,
            )
        with pytest.raises(RuntimeError, match="stale"):
            store.put_result(
                result,
                attempt_count=first["attemptCount"],
                fencing_token=first["fencingToken"],
                worker_id="worker-1",
            )
        with pytest.raises(RuntimeError, match="stale"):
            store.put_disagreement(
                disagreement,
                shard_id=first["shardId"],
                attempt_count=first["attemptCount"],
                fencing_token=first["fencingToken"],
                worker_id="worker-1",
            )

        second_receipt = store.put_sequential_search_receipt(
            receipt,
            run_code=run["runCode"],
            shard_id=second["shardId"],
            attempt_count=second["attemptCount"],
            fencing_token=second["fencingToken"],
            worker_id="worker-2",
            seed=41,
        )
        assert second_receipt["receiptHash"] == first_receipt["receiptHash"]
        assert second_receipt["occurrenceHash"] != first_receipt["occurrenceHash"]
        store.put_result(
            result,
            attempt_count=second["attemptCount"],
            fencing_token=second["fencingToken"],
            worker_id="worker-2",
        )
        store.put_disagreement(
            disagreement,
            shard_id=second["shardId"],
            attempt_count=second["attemptCount"],
            fencing_token=second["fencingToken"],
            worker_id="worker-2",
        )

        with pytest.raises(RuntimeError, match="stale"):
            store.complete_shard(
                first["shardId"],
                worker_id="worker-1",
                fencing_token=first["fencingToken"],
                result={"attemptedSeeds": 1},
            )
        store.complete_shard(
            second["shardId"],
            worker_id="worker-2",
            fencing_token=second["fencingToken"],
            result={"attemptedSeeds": 1},
        )

        admitted = store.list_sequential_search_receipts(
            run_code=run["runCode"]
        )
        assert admitted["total"] == 1
        assert admitted["items"][0]["occurrenceHash"] == second_receipt[
            "occurrenceHash"
        ]
        all_receipts = store.list_sequential_search_receipts(
            run_code=run["runCode"], admitted_only=False
        )
        assert all_receipts["total"] == 2
        by_attempt = {
            item["attemptCount"]: item["admitted"]
            for item in all_receipts["items"]
        }
        assert by_attempt == {
            first["attemptCount"]: False,
            second["attemptCount"]: True,
        }
        assert store.get_result(result["resultCode"]) == result
        assert store.list_disagreements(run_code=run["runCode"]) == [
            disagreement
        ]

        result_projection = store._db.execute(
            """SELECT attempt_count,fencing_token,admitted
               FROM research_results WHERE result_code=?""",
            (result["resultCode"],),
        ).fetchone()
        disagreement_projection = store._db.execute(
            """SELECT attempt_count,fencing_token,admitted
               FROM semantic_disagreements WHERE disagreement_id=?""",
            (disagreement["disagreementId"],),
        ).fetchone()
        assert tuple(result_projection) == (
            first["attemptCount"],
            first["fencingToken"],
            1,
        )
        assert tuple(disagreement_projection) == (
            first["attemptCount"],
            first["fencingToken"],
            1,
        )
        semantic_after_retry = store._db.execute(
            """SELECT payload_json,attempt_count,fencing_token
               FROM research_results WHERE result_code=?""",
            (result["resultCode"],),
        ).fetchone()
        disagreement_after_retry = store._db.execute(
            """SELECT payload_json,attempt_count,fencing_token
               FROM semantic_disagreements WHERE disagreement_id=?""",
            (disagreement["disagreementId"],),
        ).fetchone()
        assert tuple(semantic_after_retry) == tuple(semantic_before_retry)
        assert tuple(disagreement_after_retry) == tuple(
            disagreement_before_retry
        )
        result_occurrences = store._db.execute(
            """SELECT attempt_count,admitted FROM research_result_occurrences
               ORDER BY attempt_count"""
        ).fetchall()
        disagreement_occurrences = store._db.execute(
            """SELECT attempt_count,admitted
               FROM semantic_disagreement_occurrences
               ORDER BY attempt_count"""
        ).fetchall()
        assert [tuple(row) for row in result_occurrences] == [(1, 0), (2, 1)]
        assert [tuple(row) for row in disagreement_occurrences] == [
            (1, 0),
            (2, 1),
        ]
        assert store._db.execute(
            "SELECT COUNT(*) FROM sequential_search_receipt_bundles"
        ).fetchone()[0] == 1
    finally:
        store.close()


def test_sqlite_receipt_occurrence_is_immutable_and_seed_is_fenced(tmp_path) -> None:
    store, run, claim = _store_with_claim(tmp_path)
    receipt = _receipt(41)
    try:
        retained = store.put_sequential_search_receipt(
            receipt,
            run_code=run["runCode"],
            shard_id=claim["shardId"],
            attempt_count=claim["attemptCount"],
            fencing_token=claim["fencingToken"],
            worker_id="worker-1",
            seed=41,
        )
        with pytest.raises(ValueError, match="rootSeed"):
            store.put_sequential_search_receipt(
                receipt,
                run_code=run["runCode"],
                shard_id=claim["shardId"],
                attempt_count=claim["attemptCount"],
                fencing_token=claim["fencingToken"],
                worker_id="worker-1",
                seed=42,
            )
        different = _receipt(41)
        different_core = dict(different)
        different_core.pop("sha256")
        different_core["maximumSolutions"] = 3
        different_request = dict(different_core["searchRequest"])
        different_request.pop("sha256")
        different_request["maximumSolutions"] = 3
        different_core["searchRequest"] = {
            **different_request,
            "sha256": canonical_sha256(different_request),
        }
        different = SequentialCandidateSearchReceiptV1._build(
            different_core
        ).to_dict()
        with pytest.raises(RuntimeError, match="immutable"):
            store.put_sequential_search_receipt(
                different,
                run_code=run["runCode"],
                shard_id=claim["shardId"],
                attempt_count=claim["attemptCount"],
                fencing_token=claim["fencingToken"],
                worker_id="worker-1",
                seed=41,
            )
        assert store.list_sequential_search_receipts(
            run_code=run["runCode"], admitted_only=False
        )["items"] == [retained]
        with pytest.raises(RuntimeError, match="stale"):
            store.put_sequential_search_receipt(
                receipt,
                run_code=run["runCode"],
                shard_id=claim["shardId"],
                attempt_count=claim["attemptCount"],
                fencing_token=claim["fencingToken"],
                worker_id="wrong-worker",
                seed=41,
            )
        store._db.execute(
            """UPDATE research_shards
               SET lease_expires_at='2000-01-01T00:00:00.000000+00:00'
               WHERE shard_id=?""",
            (claim["shardId"],),
        )
        store._db.commit()
        with pytest.raises(RuntimeError, match="stale"):
            store.put_sequential_search_receipt(
                receipt,
                run_code=run["runCode"],
                shard_id=claim["shardId"],
                attempt_count=claim["attemptCount"],
                fencing_token=claim["fencingToken"],
                worker_id="worker-1",
                seed=41,
            )
        with pytest.raises(RuntimeError, match="stale"):
            store.heartbeat_shard(
                claim["shardId"],
                worker_id="worker-1",
                fencing_token=claim["fencingToken"],
                lease_seconds=30,
                now=datetime(2026, 8, 9, tzinfo=timezone.utc),
            )
        with pytest.raises(RuntimeError, match="stale"):
            store.complete_shard(
                claim["shardId"],
                worker_id="worker-1",
                fencing_token=claim["fencingToken"],
                result={"attemptedSeeds": 1},
            )
        occurrence = store._db.execute(
            """SELECT admitted FROM sequential_search_receipt_occurrences
               WHERE occurrence_hash=?""",
            (retained["occurrenceHash"],),
        ).fetchone()
        assert occurrence is not None and int(occurrence["admitted"]) == 0
        current = store.get_shard(claim["shardId"])
        assert current is not None and current["status"] == "running"
        assert str(current["leaseExpiresAt"]).startswith("2000-01-01")
    finally:
        store.close()


@pytest.mark.real_clock
def test_sqlite_leases_are_utc_validated_and_expiry_is_evaluated_after_lock(
    tmp_path,
) -> None:
    offset = timezone(timedelta(hours=3))
    started = datetime(2026, 8, 9, 12, tzinfo=offset)
    store, run, claim = _store_with_claim(tmp_path, now=started)
    contender = SQLiteResearchStoreV1(store.path)
    try:
        assert claim["heartbeatAt"] == "2026-08-09T09:00:00.000000+00:00"
        assert claim["leaseExpiresAt"] == "2026-08-09T09:00:30.000000+00:00"
        renewed = store.heartbeat_shard(
            claim["shardId"],
            worker_id="worker-1",
            fencing_token=claim["fencingToken"],
            lease_seconds=30,
            now=started + timedelta(seconds=5),
        )
        assert renewed["heartbeatAt"] == "2026-08-09T09:00:05.000000+00:00"
        assert renewed["leaseExpiresAt"] == "2026-08-09T09:00:35.000000+00:00"

        with pytest.raises(ValueError, match="timezone-aware"):
            store.heartbeat_shard(
                claim["shardId"],
                worker_id="worker-1",
                fencing_token=claim["fencingToken"],
                lease_seconds=30,
                now=datetime(2026, 8, 9),
            )
        for duration in (0, -1, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="lease must be positive"):
                store.claim_shard(
                    run["runCode"],
                    worker_id="worker-1",
                    lease_seconds=duration,
                )

        def wait_past_expiry(operation):
            store._db.execute("BEGIN IMMEDIATE")
            expiry = datetime.now(timezone.utc) + timedelta(milliseconds=150)
            store._db.execute(
                "UPDATE research_shards SET lease_expires_at=? WHERE shard_id=?",
                (expiry.isoformat(timespec="microseconds"), claim["shardId"]),
            )
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(operation)
                time.sleep(0.35)
                waited_for_lock = not future.done()
                store._db.commit()
                assert waited_for_lock
                with pytest.raises(RuntimeError, match="stale"):
                    future.result(timeout=5)

        wait_past_expiry(
            lambda: contender.heartbeat_shard(
                claim["shardId"],
                worker_id="worker-1",
                fencing_token=claim["fencingToken"],
                lease_seconds=30,
            )
        )
        wait_past_expiry(
            lambda: contender.complete_shard(
                claim["shardId"],
                worker_id="worker-1",
                fencing_token=claim["fencingToken"],
                result={"attemptedSeeds": 1},
            )
        )
        current = store.get_shard(claim["shardId"])
        assert current is not None and current["status"] == "running"
    finally:
        contender.close()
        store.close()


def test_sqlite_legacy_null_fence_reads_are_preserved_not_reinterpreted(
    tmp_path,
) -> None:
    store, run, claim = _store_with_claim(tmp_path)
    admitted_result = {
        **_result(run["runCode"], claim["shardId"]),
        "resultCode": "LAF-LEGACY",
        "resultHash": _hash("legacy-result"),
    }
    admitted_disagreement = {
        **_disagreement(run["runCode"]),
        "disagreementId": "DSG-LEGACY",
        "sha256": _hash("legacy-disagreement"),
    }
    staged_result = {
        **admitted_result,
        "resultCode": "LAF-LEGACY-STAGED",
        "resultHash": _hash("legacy-staged-result"),
    }
    try:
        now = "2026-08-09T00:00:00.000000+00:00"
        store._db.execute(
            """INSERT INTO research_results(
                   result_code,result_hash,run_code,truth_label,confidence,score,
                   disposition,payload_json,payload_hash,created_at,shard_id,admitted,
                   projection_version,attempt_count,fencing_token)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,1,1,NULL,NULL)""",
            (
                admitted_result["resultCode"],
                admitted_result["resultHash"],
                admitted_result["runCode"],
                admitted_result["truthLabel"],
                admitted_result["confidence"],
                admitted_result["score"],
                admitted_result["disposition"],
                json.dumps(
                    admitted_result,
                    sort_keys=True,
                        separators=(",", ":"),
                    ),
                    stored_result_payload_hash_v1(admitted_result),
                    now,
                claim["shardId"],
            ),
        )
        store._db.execute(
            """INSERT INTO semantic_disagreements(
               disagreement_id,disagreement_hash,run_code,category,status,
               payload_json,created_at,updated_at,shard_id,admitted,
               attempt_count,fencing_token)
               VALUES(?,?,?,?,?,?,?,?,?,1,NULL,NULL)""",
            (
                admitted_disagreement["disagreementId"],
                admitted_disagreement["sha256"],
                admitted_disagreement["runCode"],
                admitted_disagreement["category"],
                admitted_disagreement["status"],
                json.dumps(
                    admitted_disagreement,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                now,
                now,
                claim["shardId"],
            ),
        )
        store._db.execute(
            """INSERT INTO research_results(
                   result_code,result_hash,run_code,truth_label,confidence,score,
                   disposition,payload_json,payload_hash,created_at,shard_id,admitted,
                   projection_version,attempt_count,fencing_token)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,0,1,NULL,NULL)""",
            (
                staged_result["resultCode"],
                staged_result["resultHash"],
                staged_result["runCode"],
                staged_result["truthLabel"],
                staged_result["confidence"],
                staged_result["score"],
                staged_result["disposition"],
                json.dumps(
                    staged_result,
                    sort_keys=True,
                        separators=(",", ":"),
                    ),
                    stored_result_payload_hash_v1(staged_result),
                    now,
                claim["shardId"],
            ),
        )
        store._db.commit()
        assert store.get_result(admitted_result["resultCode"]) == admitted_result
        assert store.list_disagreements(run_code=run["runCode"]) == [
            admitted_disagreement
        ]
        store.complete_shard(
            claim["shardId"],
            worker_id="worker-1",
            fencing_token=claim["fencingToken"],
            result={"attemptedSeeds": 1},
        )
        assert store.get_result(staged_result["resultCode"]) is None
        null_fences = store._db.execute(
            """SELECT attempt_count,fencing_token FROM research_results
               WHERE result_code IN (?,?) ORDER BY result_code""",
            (admitted_result["resultCode"], staged_result["resultCode"]),
        ).fetchall()
        assert [tuple(row) for row in null_fences] == [(None, None), (None, None)]
    finally:
        store.close()


def test_sqlite_concurrent_receipt_cas_replay_is_idempotent(tmp_path) -> None:
    store, run, claim = _store_with_claim(tmp_path)
    path = store.path
    receipt = _receipt(41)
    barrier = threading.Barrier(2, timeout=10)

    def retain() -> dict[str, object]:
        local = SQLiteResearchStoreV1(path)
        try:
            barrier.wait()
            return local.put_sequential_search_receipt(
                receipt,
                run_code=run["runCode"],
                shard_id=claim["shardId"],
                attempt_count=claim["attemptCount"],
                fencing_token=claim["fencingToken"],
                worker_id="worker-1",
                seed=41,
            )
        finally:
            local.close()

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            values = list(executor.map(lambda _index: retain(), range(2)))
        assert values[0] == values[1]
        assert store.count_sequential_search_receipts(
            run_code=run["runCode"], admitted_only=False
        ) == 1
        assert store._db.execute(
            "SELECT COUNT(*) FROM sequential_search_receipt_bundles"
        ).fetchone()[0] == 1
    finally:
        store.close()


def test_sqlite_only_one_occurrence_per_run_seed_can_be_admitted(
    tmp_path,
) -> None:
    store, run, first = _store_with_claim(tmp_path)
    receipt = _receipt(41)
    try:
        store.put_sequential_search_receipt(
            receipt,
            run_code=run["runCode"],
            shard_id=first["shardId"],
            attempt_count=first["attemptCount"],
            fencing_token=first["fencingToken"],
            worker_id="worker-1",
            seed=41,
        )
        store.complete_shard(
            first["shardId"],
            worker_id="worker-1",
            fencing_token=first["fencingToken"],
            result={"attemptedSeeds": 1},
        )
        duplicate_seed = partition_research_shards(
            run_code=run["runCode"],
            run_hash=run["runHash"],
            recipe_hash=run["recipeHash"],
            seed_start=41,
            seed_count=2,
            seeds_per_shard=1,
        )[1]
        # The second immutable shard is deliberately malformed to overlap the
        # first seed.  The admission index remains the final fail-closed gate.
        duplicate = duplicate_seed.__class__.build(
            run_code=duplicate_seed.run_code,
            run_hash=duplicate_seed.run_hash,
            recipe_hash=duplicate_seed.recipe_hash,
            shard_index=duplicate_seed.shard_index,
            seed_start=41,
            seed_count=1,
        )
        store.create_shards((duplicate.to_dict(),))
        second = store.claim_shard(
            run["runCode"], worker_id="worker-2", lease_seconds=30
        )
        assert second is not None
        store.put_sequential_search_receipt(
            receipt,
            run_code=run["runCode"],
            shard_id=second["shardId"],
            attempt_count=second["attemptCount"],
            fencing_token=second["fencingToken"],
            worker_id="worker-2",
            seed=41,
        )
        with pytest.raises(sqlite3.IntegrityError):
            store.complete_shard(
                second["shardId"],
                worker_id="worker-2",
                fencing_token=second["fencingToken"],
                result={"attemptedSeeds": 1},
            )
        assert store.count_sequential_search_receipts(
            run_code=run["runCode"]
        ) == 1
        current = store.get_shard(second["shardId"])
        assert current is not None and current["status"] == "running"
    finally:
        store.close()
