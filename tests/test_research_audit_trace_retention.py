from __future__ import annotations

import copy
import hashlib
import inspect
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import pytest
import yaml

from baloot_platform.agents.profiles import AgentProfileRegistry
from baloot_platform.challenge.simulation import NaturalLaffaSimulatorV1
from baloot_platform.challenge.trace import CompleteGameTraceV1
from baloot_platform.config import build_bundle
from baloot_platform.core.enums import GamePhase, Seat
from baloot_platform.core.rules import RuleBook
from baloot_platform.research.audit_trace import (
    COMPLETE_TRACE_MEDIA_TYPE,
    ResearchAuditTraceRecordV1,
    ResearchAuditTraceRepositoryV1,
)
from baloot_platform.research.postgres_store import PostgresResearchStoreV1
from baloot_platform.research.sharding import partition_research_shards
from baloot_platform.research.store import SQLiteResearchStoreV1
from baloot_platform.storage.artifact_store import (
    LocalContentAddressedArtifactStore,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(slots=True)
class _ReplayFixture:
    rules: RuleBook
    traces: tuple[CompleteGameTraceV1, CompleteGameTraceV1]
    configuration_snapshot_hash: str


@dataclass(slots=True)
class _Harness:
    store: SQLiteResearchStoreV1
    artifacts: LocalContentAddressedArtifactStore
    repository: ResearchAuditTraceRepositoryV1
    recipe: dict[str, Any]


@dataclass(slots=True)
class _OldDraftAuditDatabase:
    path: Path
    record: dict[str, Any]


class _FakePostgresResult:
    def fetchone(self) -> None:
        return None

    def fetchall(self) -> list[Any]:
        return []


class _FakeContext:
    def __init__(self, value: Any) -> None:
        self.value = value

    def __enter__(self) -> Any:
        return self.value

    def __exit__(self, *args: Any) -> bool:
        return False


class _FakePostgresConnection:
    def __init__(self) -> None:
        self.statements: list[tuple[str, Any]] = []
        self.transaction_count = 0
        self.commit_count = 0
        self.rollback_count = 0
        self.closed = False

    def execute(
        self,
        statement: str,
        parameters: Any = None,
    ) -> _FakePostgresResult:
        self.statements.append((statement, parameters))
        return _FakePostgresResult()

    def executemany(self, statement: str, parameters: Any) -> None:
        self.statements.append((statement, parameters))

    def transaction(self) -> _FakeContext:
        self.transaction_count += 1
        return _FakeContext(self)

    def commit(self) -> None:
        self.commit_count += 1

    def rollback(self) -> None:
        self.rollback_count += 1

    def close(self) -> None:
        self.closed = True


class _FakePostgresPool:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.connections: list[_FakePostgresConnection] = []
        self.closed = False

    def connection(self) -> _FakeContext:
        connection = _FakePostgresConnection()
        self.connections.append(connection)
        return _FakeContext(connection)

    def close(self) -> None:
        self.closed = True


class _BootstrapPostgresPool(_FakePostgresPool):
    latest: "_BootstrapPostgresPool | None" = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        type(self).latest = self


class _LockNotAvailable(RuntimeError):
    pass


class _FailingBootstrapConnection(_FakePostgresConnection):
    def execute(
        self,
        statement: str,
        parameters: Any = None,
    ) -> _FakePostgresResult:
        result = super().execute(statement, parameters)
        if "pg_advisory_xact_lock" in statement:
            raise _LockNotAvailable("canceling statement due to lock timeout")
        return result


class _FailingBootstrapPool(_BootstrapPostgresPool):
    def connection(self) -> _FakeContext:
        connection = _FailingBootstrapConnection()
        self.connections.append(connection)
        return _FakeContext(connection)


class _SqlStateError(RuntimeError):
    def __init__(self, sqlstate: str) -> None:
        super().__init__(f"PostgreSQL test error {sqlstate}")
        self.sqlstate = sqlstate


@pytest.fixture(scope="module")
def replay_fixture(root: Path) -> _ReplayFixture:
    bundle = build_bundle(root)
    rules = RuleBook.load(root / "configs/rules/saudi_common_v1.yaml")
    convention = yaml.safe_load(
        (root / "configs/conventions/conservative_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    simulator = NaturalLaffaSimulatorV1(
        rules,
        convention,
        bundle_hash=bundle.sha256,
        configuration_snapshot=bundle.configuration_snapshot(),
        agent_profile=AgentProfileRegistry.from_bundle(root).get("strong"),
        maximum_deal_attempts=4,
    )
    first = simulator.simulate(1, dealer=Seat.EAST)
    second = simulator.simulate(2, dealer=Seat.EAST)
    assert first.publishable_laffa is True
    assert second.publishable_laffa is True
    assert first.final_trace.sha256 != second.final_trace.sha256
    assert first.final_trace.trusted_final_state["phase"] == GamePhase.SCORED.value
    assert second.final_trace.trusted_final_state["phase"] == GamePhase.SCORED.value
    return _ReplayFixture(
        rules=rules,
        traces=(first.final_trace, second.final_trace),
        configuration_snapshot_hash=simulator.configuration_snapshot_hash,
    )


@pytest.fixture
def harness(tmp_path: Path) -> _Harness:
    store = SQLiteResearchStoreV1(tmp_path / "research.sqlite3")
    artifacts = LocalContentAddressedArtifactStore(tmp_path / "artifacts")
    recipe = {
        "code": "RCP-AUDIT01",
        "sha256": _hash("audit-recipe"),
        "status": "published",
    }
    store.put_recipe(recipe)
    value = _Harness(
        store=store,
        artifacts=artifacts,
        repository=ResearchAuditTraceRepositoryV1(
            store=store,
            artifact_store=artifacts,
        ),
        recipe=recipe,
    )
    try:
        yield value
    finally:
        store.close()


def _claim(
    harness: _Harness,
    label: str,
    *,
    worker_id: str | None = None,
    maximum_attempts: int = 3,
    now: datetime | None = None,
    lease_seconds: float = 600,
) -> dict[str, Any]:
    run_code = f"RUN-AUDIT-{label.upper()}"
    run = {
        "runCode": run_code,
        "runHash": _hash(f"run:{label}"),
        "recipeCode": harness.recipe["code"],
        "recipeHash": harness.recipe["sha256"],
        "status": "running",
        "progress": 0.0,
        "events": [],
    }
    harness.store.create_run(run)
    spec = partition_research_shards(
        run_code=run_code,
        run_hash=run["runHash"],
        recipe_hash=harness.recipe["sha256"],
        seed_start=1,
        seed_count=1,
        seeds_per_shard=1,
        maximum_attempts=maximum_attempts,
    )[0]
    harness.store.create_shards((spec.to_dict(),))
    claimed = harness.store.claim_shard(
        run_code,
        worker_id=worker_id or f"worker-{label}",
        lease_seconds=lease_seconds,
        now=now,
    )
    assert claimed is not None
    return claimed


def _source(
    harness: _Harness,
    claim: Mapping[str, Any],
    *,
    candidate_label: str = "candidate",
) -> dict[str, Any]:
    return {
        "kind": "research_candidate",
        "runCode": claim["runCode"],
        "runHash": claim["runHash"],
        "recipeCode": harness.recipe["code"],
        "recipeHash": harness.recipe["sha256"],
        "shardId": claim["shardId"],
        "shardHash": claim["sha256"],
        "shardIndex": claim["shardIndex"],
        "seed": claim["seedStart"],
        "candidateHash": _hash(
            f"{claim['runCode']}:{claim['attemptCount']}:{candidate_label}"
        ),
        "candidateOrdinal": 0,
        "generationLane": "organic",
        "attemptCount": claim["attemptCount"],
        "fencingToken": claim["fencingToken"],
        "workerId": claim["leaseOwner"],
    }


def _retain(
    harness: _Harness,
    replay_fixture: _ReplayFixture,
    claim: Mapping[str, Any],
    *,
    trace_index: int = 0,
    candidate_label: str = "candidate",
) -> ResearchAuditTraceRecordV1:
    return harness.repository.retain(
        replay_fixture.traces[trace_index],
        rules=replay_fixture.rules,
        source=_source(
            harness,
            claim,
            candidate_label=candidate_label,
        ),
        research_configuration={
            "schemaVersion": 1,
            "configurationHash": _hash("frozen-research-configuration"),
            "generationLane": "organic",
        },
    )


@pytest.fixture
def old_draft_audit_database(
    harness: _Harness,
    replay_fixture: _ReplayFixture,
) -> _OldDraftAuditDatabase:
    claim = _claim(harness, "old-draft-schema")
    record = _retain(harness, replay_fixture, claim).to_dict()
    database_path = harness.store.path
    harness.store.close()

    with sqlite3.connect(database_path) as database:
        database.execute("PRAGMA foreign_keys=OFF")
        database.execute("DROP TABLE research_audit_traces")
        database.execute(
            """CREATE TABLE research_audit_traces(
                 trace_hash TEXT PRIMARY KEY,
                 audit_trace_code TEXT NOT NULL UNIQUE,
                 artifact_sha256 TEXT NOT NULL,
                 run_code TEXT NOT NULL,
                 shard_id TEXT,
                 candidate_hash TEXT NOT NULL,
                 disposition TEXT NOT NULL,
                 status TEXT NOT NULL,
                 record_json TEXT NOT NULL,
                 admitted INTEGER NOT NULL DEFAULT 1,
                 created_at TEXT NOT NULL,
                 updated_at TEXT NOT NULL,
                 FOREIGN KEY(run_code) REFERENCES research_runs(run_code))"""
        )
        database.execute(
            """INSERT INTO research_audit_traces(
                 trace_hash,audit_trace_code,artifact_sha256,run_code,shard_id,
                 candidate_hash,disposition,status,record_json,admitted,
                 created_at,updated_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                record["traceHash"],
                record["auditTraceCode"],
                record["completeTrace"]["sha256"],
                record["source"]["runCode"],
                record["source"]["shardId"],
                record["source"]["candidateHash"],
                record["disposition"],
                record["status"],
                _canonical_bytes(record).decode("utf-8"),
                1,
                record["createdAt"],
                record["updatedAt"],
            ),
        )

    return _OldDraftAuditDatabase(path=database_path, record=record)


def _all_keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        return {
            *(str(key) for key in value),
            *(
                child
                for item in value.values()
                for child in _all_keys(item)
            ),
        }
    if isinstance(value, list | tuple):
        return {child for item in value for child in _all_keys(item)}
    return set()


def test_replay_valid_trace_is_immediately_visible_and_audit_only(
    harness: _Harness,
    replay_fixture: _ReplayFixture,
) -> None:
    claim = _claim(harness, "immediate")
    record = _retain(harness, replay_fixture, claim)

    shard = harness.store.get_shard(claim["shardId"])
    assert shard is not None and shard["status"] == "running"
    page = harness.repository.list_records(run_code=claim["runCode"])
    assert page["total"] == 1
    assert page["items"] == [record.to_dict()]
    assert record.status == "provisional"
    assert record.disposition == "provisional"
    assert record.attempt_status == "running"
    assert record.to_dict()["auditOnly"] is True
    assert record.to_dict()["publicationEligible"] is False
    assert set(record.failure_categories) == {
        "content_qualification_not_established",
        "research_analysis_not_completed",
    }

    assert record.replay_validation["status"] == "verified"
    assert (
        record.replay_validation["validator"]
        == "authoritative_complete_trace_replay_v1"
    )
    assert record.replay_validation["decisionCount"] == len(
        replay_fixture.traces[0].decision_ledger.records
    )
    assert record.observer_safety == {
        "schemaVersion": 1,
        "status": "verified",
        "validator": "isolated_observer_projection_v1",
        "traceHash": replay_fixture.traces[0].sha256,
        "projectedDecisionCount": len(
            replay_fixture.traces[0].decision_ledger.records
        ),
        "trustedTraceExposure": "internal_only",
    }
    assert (
        record.configuration_provenance["configurationSnapshotHash"]
        == replay_fixture.configuration_snapshot_hash
    )
    assert record.configuration_provenance["researchConfiguration"] == {
        "schemaVersion": 1,
        "configurationHash": _hash("frozen-research-configuration"),
        "generationLane": "organic",
    }

    public = record.to_dict()
    assert public["completeTrace"] == record.complete_trace
    assert set(public["completeTrace"]) == {
        "uri",
        "sha256",
        "bytes",
        "mediaType",
        "name",
    }
    assert public["completeTrace"]["mediaType"] == COMPLETE_TRACE_MEDIA_TYPE
    assert not {
        "trustedFinalState",
        "completeState",
        "hands",
        "initialHands",
        "observerHand",
        "dealTrace",
    }.intersection(_all_keys(public))


def test_retry_is_idempotent_and_same_trace_bytes_are_deduplicated(
    harness: _Harness,
    replay_fixture: _ReplayFixture,
) -> None:
    claim = _claim(harness, "idempotent")
    first = _retain(harness, replay_fixture, claim)
    repeated = _retain(harness, replay_fixture, claim)

    assert repeated == first
    assert harness.repository.list_records()["total"] == 1
    payloads = [
        path
        for path in (harness.artifacts.root / "sha256").rglob("*")
        if path.is_file() and len(path.name) == 64
    ]
    assert payloads == [
        harness.artifacts.materialize(first.complete_trace["uri"])
    ]


def test_same_trace_keeps_distinct_source_occurrences_with_one_cas_object(
    harness: _Harness,
    replay_fixture: _ReplayFixture,
) -> None:
    first_claim = _claim(harness, "occurrence-a")
    second_claim = _claim(harness, "occurrence-b")
    first = _retain(harness, replay_fixture, first_claim)
    second = _retain(harness, replay_fixture, second_claim)

    assert first.trace_hash == second.trace_hash
    assert first.record_hash != second.record_hash
    assert first.audit_trace_code != second.audit_trace_code
    assert first.complete_trace["sha256"] == second.complete_trace["sha256"]
    assert harness.repository.list_records()["total"] == 2
    assert {
        item["source"]["runCode"]
        for item in harness.repository.list_records()["items"]
    } == {first_claim["runCode"], second_claim["runCode"]}


def test_same_candidate_occurrence_cannot_be_rebound_to_another_trace(
    harness: _Harness,
    replay_fixture: _ReplayFixture,
) -> None:
    claim = _claim(harness, "source-conflict")
    first = _retain(harness, replay_fixture, claim, trace_index=0)

    with pytest.raises((ValueError, RuntimeError)):
        _retain(harness, replay_fixture, claim, trace_index=1)

    page = harness.repository.list_records(run_code=claim["runCode"])
    assert page["total"] == 1
    assert page["items"][0]["recordHash"] == first.record_hash


def test_stale_fencing_can_write_cas_bytes_but_cannot_index_the_trace(
    harness: _Harness,
    replay_fixture: _ReplayFixture,
) -> None:
    expired_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    first = _claim(
        harness,
        "stale-fence",
        worker_id="worker-old",
        now=expired_at,
        lease_seconds=1,
    )
    current = harness.store.claim_shard(
        first["runCode"],
        worker_id="worker-current",
        lease_seconds=600,
        now=datetime.now(timezone.utc),
    )
    assert current is not None
    assert current["fencingToken"] > first["fencingToken"]

    with pytest.raises(RuntimeError, match="stale"):
        _retain(harness, replay_fixture, first)

    assert harness.repository.list_records(run_code=first["runCode"])["total"] == 0
    payloads = [
        path
        for path in (harness.artifacts.root / "sha256").rglob("*")
        if path.is_file() and len(path.name) == 64
    ]
    assert len(payloads) == 1


def test_retry_reclaim_and_failure_keep_all_occurrences_visible(
    harness: _Harness,
    replay_fixture: _ReplayFixture,
) -> None:
    first_claim = _claim(
        harness,
        "attempt-lifecycle",
        worker_id="worker-first",
        maximum_attempts=3,
    )
    first = _retain(harness, replay_fixture, first_claim)
    harness.store.fail_shard(
        first_claim["shardId"],
        worker_id="worker-first",
        fencing_token=first_claim["fencingToken"],
        error={"category": "temporary_worker_failure"},
        retryable=True,
    )
    first_after = harness.repository.get_record(first.record_hash)
    assert first_after is not None
    assert first_after.attempt_status == "retry_scheduled"
    assert {
        "research_shard_attempt_failed",
        "research_shard_retry_scheduled",
    }.issubset(first_after.failure_categories)

    second_claim = harness.store.claim_shard(
        first_claim["runCode"],
        worker_id="worker-second",
        lease_seconds=600,
    )
    assert second_claim is not None
    second = _retain(harness, replay_fixture, second_claim)
    assert harness.store.recover_orphaned_shards() == [first_claim["runCode"]]
    second_after = harness.repository.get_record(second.record_hash)
    assert second_after is not None
    assert second_after.attempt_status == "reclaimed"
    assert "research_shard_attempt_reclaimed" in second_after.failure_categories

    third_claim = harness.store.claim_shard(
        first_claim["runCode"],
        worker_id="worker-third",
        lease_seconds=600,
    )
    assert third_claim is not None
    third = _retain(harness, replay_fixture, third_claim)
    harness.store.fail_shard(
        third_claim["shardId"],
        worker_id="worker-third",
        fencing_token=third_claim["fencingToken"],
        error={"category": "terminal_worker_failure"},
        retryable=False,
    )
    third_after = harness.repository.get_record(third.record_hash)
    assert third_after is not None
    assert third_after.attempt_status == "failed"
    assert "research_shard_attempt_failed" in third_after.failure_categories

    page = harness.repository.list_records(run_code=first_claim["runCode"])
    assert page["total"] == 3
    assert {item["attemptStatus"] for item in page["items"]} == {
        "retry_scheduled",
        "reclaimed",
        "failed",
    }


def test_finalized_rejected_and_quarantined_records_preserve_disagreements(
    harness: _Harness,
    replay_fixture: _ReplayFixture,
) -> None:
    rejected_claim = _claim(harness, "final-rejected")
    rejected = _retain(harness, replay_fixture, rejected_claim)
    rejected = harness.repository.finalize(
        rejected.audit_trace_code,
        disposition="rejected",
        failure_categories=("content_gate_failed",),
    )
    assert rejected.status == "finalized"
    assert rejected.disposition == "rejected"
    assert rejected.failure_categories == ("content_gate_failed",)
    assert rejected.to_dict()["auditOnly"] is True
    assert rejected.to_dict()["publicationEligible"] is False

    quarantined_claim = _claim(harness, "final-quarantined")
    quarantined = _retain(harness, replay_fixture, quarantined_claim)
    disagreement = {
        "disagreementId": "DSG-AUDIT-001",
        "sha256": _hash("semantic-disagreement"),
        "category": "solver_runtime_disagreement",
        "status": "open",
    }
    quarantined = harness.repository.finalize(
        quarantined.record_hash,
        disposition="quarantined",
        failure_categories=("semantic_parity_disagreement",),
        disagreement_references=(disagreement,),
    )
    assert quarantined.status == "finalized"
    assert quarantined.disposition == "quarantined"
    assert quarantined.failure_categories == ("semantic_parity_disagreement",)
    assert quarantined.disagreement_references == (disagreement,)
    assert harness.repository.get_record(quarantined.trace_hash) == quarantined


@pytest.mark.parametrize("failure_mode", ["missing", "corrupt", "tampered_trace"])
def test_missing_corrupt_or_tampered_trace_bytes_fail_closed(
    harness: _Harness,
    replay_fixture: _ReplayFixture,
    failure_mode: str,
) -> None:
    claim = _claim(harness, f"artifact-{failure_mode}")
    record = _retain(harness, replay_fixture, claim)
    payload_path = harness.artifacts.materialize(record.complete_trace["uri"])

    if failure_mode == "missing":
        payload_path.unlink()
        expected_error: type[Exception] | tuple[type[Exception], ...] = (
            FileNotFoundError
        )
    elif failure_mode == "corrupt":
        payload_path.write_bytes(b"corrupt complete trace")
        expected_error = RuntimeError
    else:
        trusted_payload = copy.deepcopy(
            replay_fixture.traces[0].to_dict(trusted=True)
        )
        trusted_payload["trustedFinalState"]["matchScore"][0] += 1
        tampered_reference = harness.artifacts.put(
            _canonical_bytes(trusted_payload),
            media_type=COMPLETE_TRACE_MEDIA_TYPE,
            name="tampered.complete-trace.json",
        )
        public = record.to_dict()
        public["completeTrace"] = tampered_reference
        harness.store._db.execute(
            """UPDATE research_audit_traces
               SET artifact_sha256=?,record_json=? WHERE record_hash=?""",
            (
                tampered_reference["sha256"],
                json.dumps(public, ensure_ascii=False, sort_keys=True),
                record.record_hash,
            ),
        )
        harness.store._db.commit()
        expected_error = ValueError

    with pytest.raises(expected_error):
        harness.repository.load_trace(record.audit_trace_code)


def test_sqlite_old_draft_audit_schema_upgrades_without_losing_the_row(
    old_draft_audit_database: _OldDraftAuditDatabase,
) -> None:
    expected = old_draft_audit_database.record
    with sqlite3.connect(old_draft_audit_database.path) as old_database:
        old_database.row_factory = sqlite3.Row
        old_columns = {
            str(row["name"]): row
            for row in old_database.execute(
                "PRAGMA table_info(research_audit_traces)"
            ).fetchall()
        }
        assert set(old_columns) == {
            "trace_hash",
            "audit_trace_code",
            "artifact_sha256",
            "run_code",
            "shard_id",
            "candidate_hash",
            "disposition",
            "status",
            "record_json",
            "admitted",
            "created_at",
            "updated_at",
        }
        assert old_columns["trace_hash"]["pk"] == 1
        assert old_database.execute(
            "SELECT COUNT(*) FROM research_audit_traces"
        ).fetchone()[0] == 1

    expected_occurrence_hash = hashlib.sha256(
        _canonical_bytes(expected["source"])
    ).hexdigest()

    def upgraded_snapshot(store: SQLiteResearchStoreV1) -> dict[str, Any]:
        columns = {
            str(row["name"]): dict(row)
            for row in store._db.execute(
                "PRAGMA table_info(research_audit_traces)"
            ).fetchall()
        }
        indexes: list[tuple[str, int, tuple[str, ...]]] = []
        for index in store._db.execute(
            "PRAGMA index_list(research_audit_traces)"
        ).fetchall():
            index_name = str(index["name"])
            indexes.append(
                (
                    index_name,
                    int(index["unique"]),
                    tuple(
                        str(row["name"])
                        for row in store._db.execute(
                            f"PRAGMA index_info({index_name})"
                        ).fetchall()
                    ),
                )
            )
        rows = [
            dict(row)
            for row in store._db.execute(
                "SELECT * FROM research_audit_traces ORDER BY record_hash"
            ).fetchall()
        ]
        return {
            "columns": columns,
            "indexes": sorted(indexes),
            "rows": rows,
        }

    first_store = SQLiteResearchStoreV1(old_draft_audit_database.path)
    try:
        first_snapshot = upgraded_snapshot(first_store)
        columns = first_snapshot["columns"]
        assert columns["record_hash"]["pk"] == 1
        assert columns["trace_hash"]["pk"] == 0
        assert {
            "source_occurrence_hash",
            "attempt_count",
            "fencing_token",
            "candidate_ordinal",
            "seed",
            "attempt_status",
        }.issubset(columns)

        assert len(first_snapshot["rows"]) == 1
        row = first_snapshot["rows"][0]
        assert row["record_hash"] == expected["recordHash"]
        assert row["trace_hash"] == expected["traceHash"]
        assert row["source_occurrence_hash"] == expected_occurrence_hash
        assert row["attempt_count"] == expected["source"]["attemptCount"]
        assert row["fencing_token"] == expected["source"]["fencingToken"]
        assert row["candidate_ordinal"] == expected["source"]["candidateOrdinal"]
        assert row["seed"] == expected["source"]["seed"]
        assert row["attempt_status"] == expected["attemptStatus"]
        assert json.loads(row["record_json"]) == expected
        assert first_store.get_audit_trace(expected["recordHash"]) == expected

        unique_index_columns = {
            index_columns
            for _, unique, index_columns in first_snapshot["indexes"]
            if unique
        }
        assert ("source_occurrence_hash",) in unique_index_columns
        assert ("trace_hash",) not in unique_index_columns
    finally:
        first_store.close()

    second_store = SQLiteResearchStoreV1(old_draft_audit_database.path)
    try:
        assert upgraded_snapshot(second_store) == first_snapshot
        assert second_store.get_audit_trace(expected["recordHash"]) == expected
    finally:
        second_store.close()


def test_sqlite_old_draft_upgrade_rolls_back_on_index_identity_tampering(
    old_draft_audit_database: _OldDraftAuditDatabase,
) -> None:
    tampered_candidate_hash = _hash("tampered-old-draft-candidate")
    with sqlite3.connect(old_draft_audit_database.path) as database:
        database.row_factory = sqlite3.Row
        database.execute(
            "UPDATE research_audit_traces SET candidate_hash=?",
            (tampered_candidate_hash,),
        )
        before_sql = database.execute(
            """SELECT sql FROM sqlite_master
               WHERE type='table' AND name='research_audit_traces'"""
        ).fetchone()["sql"]
        before_row = dict(
            database.execute("SELECT * FROM research_audit_traces").fetchone()
        )

    with pytest.raises(
        ValueError,
        match="legacy Research audit trace index identity mismatch",
    ):
        SQLiteResearchStoreV1(old_draft_audit_database.path)

    with sqlite3.connect(old_draft_audit_database.path) as database:
        database.row_factory = sqlite3.Row
        after_sql = database.execute(
            """SELECT sql FROM sqlite_master
               WHERE type='table' AND name='research_audit_traces'"""
        ).fetchone()["sql"]
        after_row = dict(
            database.execute("SELECT * FROM research_audit_traces").fetchone()
        )
        staging_table = database.execute(
            """SELECT name FROM sqlite_master
               WHERE type='table'
                 AND name='research_audit_traces_v1_upgrade'"""
        ).fetchone()

    assert after_sql == before_sql
    assert after_row == before_row
    assert after_row["trace_hash"] == old_draft_audit_database.record["traceHash"]
    assert after_row["candidate_hash"] == tampered_candidate_hash
    assert staging_table is None


def test_sqlite_and_postgres_audit_trace_contracts_and_migration_match(
    harness: _Harness,
    root: Path,
) -> None:
    for method_name in (
        "put_audit_trace",
        "update_audit_trace",
        "get_audit_trace",
        "list_audit_traces",
    ):
        assert inspect.signature(
            getattr(SQLiteResearchStoreV1, method_name)
        ) == inspect.signature(getattr(PostgresResearchStoreV1, method_name))

    authored = (root / "migrations/021_research_audit_traces.sql").read_bytes()
    packaged = (
        root
        / "src/baloot_platform/resources/migrations/021_research_audit_traces.sql"
    ).read_bytes()
    assert packaged == authored
    sql = authored.decode("utf-8")
    assert "record_hash TEXT PRIMARY KEY" in sql
    assert "trace_hash TEXT NOT NULL" in sql
    assert "artifact_sha256 TEXT NOT NULL" in sql
    assert "source_occurrence_hash TEXT NOT NULL" in sql
    assert "attempt_count INTEGER NOT NULL" in sql
    assert "fencing_token BIGINT NOT NULL" in sql
    assert "candidate_ordinal INTEGER NOT NULL" in sql
    assert "seed BIGINT NOT NULL" in sql
    assert "ux_research_audit_trace_source" in sql
    assert "ON research.audit_traces(source_occurrence_hash)" in sql
    assert "balootx:research-store-v1:bootstrap" in sql
    assert "pg_advisory_xact_lock" in sql
    assert "FOREIGN KEY" not in sql

    columns = {
        row["name"]: row
        for row in harness.store._db.execute(
            "PRAGMA table_info(research_audit_traces)"
        ).fetchall()
    }
    assert columns["record_hash"]["pk"] == 1
    assert {
        "trace_hash",
        "artifact_sha256",
        "source_occurrence_hash",
        "run_code",
        "shard_id",
        "attempt_count",
        "fencing_token",
        "candidate_hash",
        "candidate_ordinal",
        "seed",
        "attempt_status",
    }.issubset(columns)
    unique_index_columns = {
        tuple(
            row["name"]
            for row in harness.store._db.execute(
                f"PRAGMA index_info({index['name']})"
            ).fetchall()
        )
        for index in harness.store._db.execute(
            "PRAGMA index_list(research_audit_traces)"
        ).fetchall()
        if index["unique"]
    }
    assert ("source_occurrence_hash",) in unique_index_columns
    assert ("trace_hash",) not in unique_index_columns
    assert ("artifact_sha256",) not in unique_index_columns


def test_postgres_bootstrap_takes_one_deterministic_transaction_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    psycopg_module = ModuleType("psycopg")
    rows_module = ModuleType("psycopg.rows")
    rows_module.dict_row = object()  # type: ignore[attr-defined]
    psycopg_module.rows = rows_module  # type: ignore[attr-defined]
    pool_module = ModuleType("psycopg_pool")
    pool_module.ConnectionPool = _BootstrapPostgresPool  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "psycopg", psycopg_module)
    monkeypatch.setitem(sys.modules, "psycopg.rows", rows_module)
    monkeypatch.setitem(sys.modules, "psycopg_pool", pool_module)

    store = PostgresResearchStoreV1("postgresql://offline-test")
    pool = _BootstrapPostgresPool.latest
    assert pool is not None
    assert len(pool.connections) == 1
    statements = pool.connections[0].statements
    lock_calls = [
        (index, parameters)
        for index, (statement, parameters) in enumerate(statements)
        if "pg_advisory_xact_lock(hashtextextended(%s,0))" in statement
    ]
    assert len(lock_calls) == 1
    lock_index, lock_parameters = lock_calls[0]
    assert lock_parameters == ("balootx:research-store-v1:bootstrap",)
    timeout_calls = [
        (index, parameters)
        for index, (statement, parameters) in enumerate(statements)
        if "SELECT set_config(%s,%s,true)" in statement
        and parameters[0] in {"lock_timeout", "statement_timeout"}
    ]
    assert [parameters for _, parameters in timeout_calls] == [
        ("lock_timeout", "60s"),
        ("statement_timeout", "60s"),
    ]
    assert all(index < lock_index for index, _ in timeout_calls)
    create_schema_index = next(
        index
        for index, (statement, _) in enumerate(statements)
        if "CREATE SCHEMA IF NOT EXISTS research" in statement
    )
    assert lock_index < create_schema_index
    store.close()
    assert pool.closed is True


def test_postgres_bootstrap_closes_pool_when_lock_acquisition_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    psycopg_module = ModuleType("psycopg")
    rows_module = ModuleType("psycopg.rows")
    rows_module.dict_row = object()  # type: ignore[attr-defined]
    psycopg_module.rows = rows_module  # type: ignore[attr-defined]
    pool_module = ModuleType("psycopg_pool")
    pool_module.ConnectionPool = _FailingBootstrapPool  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "psycopg", psycopg_module)
    monkeypatch.setitem(sys.modules, "psycopg.rows", rows_module)
    monkeypatch.setitem(sys.modules, "psycopg_pool", pool_module)

    with pytest.raises(
        _LockNotAvailable,
        match="canceling statement due to lock timeout",
    ):
        PostgresResearchStoreV1("postgresql://offline-test")

    pool = _FailingBootstrapPool.latest
    assert pool is not None
    assert pool.closed is True


@pytest.mark.parametrize("sqlstate", ["40001", "40P01"])
def test_postgres_serializable_retry_is_exactly_three_attempts(
    sqlstate: str,
) -> None:
    store = object.__new__(PostgresResearchStoreV1)
    success_pool = _FakePostgresPool()
    store.pool = success_pool
    success_attempts = 0

    def succeed_on_third(connection: _FakePostgresConnection) -> str:
        nonlocal success_attempts
        success_attempts += 1
        if success_attempts < 3:
            raise _SqlStateError(sqlstate)
        return "committed"

    assert store._run_serializable(
        succeed_on_third,
        advisory_lock_key="research:audit-trace:test-occurrence",
    ) == "committed"
    assert success_attempts == 3
    assert len(success_pool.connections) == 1
    success_connection = success_pool.connections[0]
    assert success_connection.transaction_count == 3
    assert success_connection.statements[0] == (
        "SELECT pg_advisory_lock(hashtextextended(%s,0))",
        ("research:audit-trace:test-occurrence",),
    )
    isolation_statements = [
        value
        for value in success_connection.statements
        if value
        == (
            "SET TRANSACTION ISOLATION LEVEL SERIALIZABLE",
            None,
        )
    ]
    assert len(isolation_statements) == 3
    assert success_connection.statements[-1] == (
        "SELECT pg_advisory_unlock(hashtextextended(%s,0))",
        ("research:audit-trace:test-occurrence",),
    )
    assert success_connection.commit_count == 2
    assert success_connection.rollback_count == 1

    exhausted_pool = _FakePostgresPool()
    store.pool = exhausted_pool
    exhausted_attempts = 0

    def always_fail(connection: _FakePostgresConnection) -> None:
        nonlocal exhausted_attempts
        exhausted_attempts += 1
        raise _SqlStateError(sqlstate)

    with pytest.raises(_SqlStateError) as raised:
        store._run_serializable(
            always_fail,
            advisory_lock_key="research:audit-trace:test-occurrence",
        )
    assert raised.value.sqlstate == sqlstate
    assert exhausted_attempts == 3
    assert len(exhausted_pool.connections) == 1
    exhausted_connection = exhausted_pool.connections[0]
    assert exhausted_connection.transaction_count == 3
    assert exhausted_connection.statements[-1][0].startswith(
        "SELECT pg_advisory_unlock"
    )


def test_postgres_serializable_does_not_retry_unrelated_errors() -> None:
    store = object.__new__(PostgresResearchStoreV1)
    pool = _FakePostgresPool()
    store.pool = pool
    attempts = 0
    expected = _SqlStateError("23505")

    def fail_once(connection: _FakePostgresConnection) -> None:
        nonlocal attempts
        attempts += 1
        raise expected

    with pytest.raises(_SqlStateError) as raised:
        store._run_serializable(
            fail_once,
            advisory_lock_key="research:audit-trace:test-occurrence",
        )
    assert raised.value is expected
    assert attempts == 1
    assert len(pool.connections) == 1


def test_postgres_audit_writes_use_serializable_occurrence_locks() -> None:
    put_source = inspect.getsource(PostgresResearchStoreV1.put_audit_trace)
    update_source = inspect.getsource(PostgresResearchStoreV1.update_audit_trace)
    domain_source = inspect.getsource(
        PostgresResearchStoreV1.put_domain_compilation_artifact
    )
    bootstrap_source = inspect.getsource(PostgresResearchStoreV1._bootstrap_schema)
    serializable_source = inspect.getsource(
        PostgresResearchStoreV1._run_serializable
    )

    for method_source in (put_source, update_source):
        assert "research:audit-trace:{source_occurrence_hash}" in method_source
        assert "return self._run_serializable(" in method_source
        assert "advisory_lock_key=" in method_source
    assert "pg_advisory_lock(hashtextextended(%s,0))" in serializable_source
    assert "pg_advisory_unlock(hashtextextended(%s,0))" in serializable_source
    assert "SET TRANSACTION ISOLATION LEVEL SERIALIZABLE" in serializable_source
    assert "ON CONFLICT DO NOTHING" in put_source
    assert "RETURNING record_json,source_occurrence_hash" in put_source
    assert "SELECT record_json,source_occurrence_hash" in put_source
    assert "record_hash=%s OR source_occurrence_hash=%s" in put_source
    assert (
        'str(row["source_occurrence_hash"]) != expected_occurrence'
        in put_source
    )
    assert "Research audit trace source occurrence index mismatch" in put_source
    assert "CREATE UNIQUE INDEX IF NOT EXISTS ux_research_audit_trace_source" in (
        bootstrap_source
    )
    assert "return self._run_serializable(" in domain_source
    assert "advisory_lock_key=" in domain_source
    assert (
        "research:domain-compilation:{value.compilation_hash}" in domain_source
    )
