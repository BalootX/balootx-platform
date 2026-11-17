from __future__ import annotations

import copy
import hashlib
import inspect
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from baloot_platform.research.contracts import canonical_json, canonical_sha256
from baloot_platform.research.domain_artifact import (
    DOMAIN_COMPILATION_MEDIA_TYPE,
    DomainCompilationArtifactRepositoryV1,
    DomainCompilationArtifactV1,
)
from baloot_platform.research.domain_compiler import CompiledDomainBundleV1
from baloot_platform.research.postgres_store import PostgresResearchStoreV1
from baloot_platform.research.service import ResearchFoundryServiceV1
from baloot_platform.research.store import SQLiteResearchStoreV1
from baloot_platform.service.container import build_container
from baloot_platform.storage.artifact_store import LocalContentAddressedArtifactStore


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _variant(
    compiled: CompiledDomainBundleV1, label: str
) -> CompiledDomainBundleV1:
    payloads = copy.deepcopy(dict(compiled.target_payloads))
    documentation = copy.deepcopy(dict(payloads["documentationModel"]))
    documentation["testVariant"] = label
    payloads["documentationModel"] = documentation
    documentation_hash = canonical_sha256(documentation)
    compilation_hash = canonical_sha256(
        {
            "schemaVersion": 1,
            "domainIrHash": compiled.domain_ir.sha256,
            "coverageHash": compiled.coverage.sha256,
            "targets": [
                compiled.runtime_plan_hash,
                compiled.reference_plan_hash,
                compiled.cp_sat_blueprint_hash,
                compiled.policy_plan_hash,
                compiled.ui_schema_hash,
                documentation_hash,
            ],
        }
    )
    return CompiledDomainBundleV1(
        domain_ir=compiled.domain_ir,
        coverage=compiled.coverage,
        definitions=compiled.definitions,
        runtime_plan_hash=compiled.runtime_plan_hash,
        reference_plan_hash=compiled.reference_plan_hash,
        cp_sat_blueprint_hash=compiled.cp_sat_blueprint_hash,
        policy_plan_hash=compiled.policy_plan_hash,
        ui_schema_hash=compiled.ui_schema_hash,
        documentation_model_hash=documentation_hash,
        target_payloads=payloads,
        publishable=compiled.publishable,
        sha256=compilation_hash,
    )


@pytest.fixture(scope="module")
def persisted_compilation(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    writable_root = tmp_path_factory.mktemp("compiled-domain-container")
    container = build_container(writable_root=writable_root, database_url="")
    try:
        active_compilation = container.research.compiled_domain.compiled
        assert container.research.store.get_domain_compilation_artifact(
            active_compilation.sha256
        ) is not None
        revision_id = str(
            container.configuration.resolve()["bundle"]["revisionId"]
        )
        _effective, context = (
            container.research.execution_context_factory.resolve_bundle_revision(
                revision_id
            )
        )
        compilation = copy.deepcopy(context.compiled_domain.compiled)
        assert compilation.sha256 == active_compilation.sha256
        record = container.research.store.get_domain_compilation_artifact(
            compilation.sha256
        )
        assert record is not None
        return {
            "compiled": compilation,
            "record": record,
            "database": Path(container.research.store.path),
            "artifacts": Path(container.research.artifact_store.root),
        }
    finally:
        container.shutdown()


def test_compilation_reloads_after_restart_and_cache_eviction(
    persisted_compilation: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled = persisted_compilation["compiled"]
    store = SQLiteResearchStoreV1(persisted_compilation["database"])
    artifacts = LocalContentAddressedArtifactStore(
        persisted_compilation["artifacts"]
    )
    repository = DomainCompilationArtifactRepositoryV1(
        store=store,
        artifact_store=artifacts,
        cache_size=1,
    )
    try:
        loaded = repository.get(compiled.sha256)
        assert loaded == compiled

        original_get = artifacts.get
        monkeypatch.setattr(
            artifacts,
            "get",
            lambda _uri: (_ for _ in ()).throw(
                AssertionError("hot lookup must not read the durable artifact")
            ),
        )
        assert repository.get(compiled.sha256) == compiled

        monkeypatch.setattr(artifacts, "get", original_get)
        variant = _variant(compiled, "eviction")
        repository.register(
            variant,
            bundle_hash=_hash("variant-bundle"),
            effective_runtime_config_hash=_hash("variant-effective"),
            context_hash=_hash("variant-context"),
            domain_hash=_hash("variant-domain"),
        )
        assert repository.cached_count == 1
        assert repository.get(compiled.sha256) == compiled

        service = SimpleNamespace(
            compiled_domain=None,
            execution_context_factory=SimpleNamespace(
                compiled_domain_for_hash=repository.get
            ),
        )
        default = ResearchFoundryServiceV1.domain_compilation(
            service, compiled.sha256
        )
        full = ResearchFoundryServiceV1.domain_compilation(
            service, compiled.sha256, include_payloads=True
        )
        assert default == compiled.to_dict()
        assert full == compiled.to_dict(include_payloads=True)
        assert "targetPayloads" not in default
        assert "targetPayloads" in full
    finally:
        store.close()


def _register_reference(
    root: Path,
    compiled: CompiledDomainBundleV1,
    *,
    payload: bytes,
) -> tuple[
    SQLiteResearchStoreV1,
    LocalContentAddressedArtifactStore,
    DomainCompilationArtifactV1,
]:
    store = SQLiteResearchStoreV1(root / "research.sqlite3")
    artifacts = LocalContentAddressedArtifactStore(root / "artifacts")
    artifact = artifacts.put(
        payload,
        media_type=DOMAIN_COMPILATION_MEDIA_TYPE,
        name=f"{compiled.sha256}.json",
    )
    record = DomainCompilationArtifactV1.build(
        {
            "schemaVersion": 1,
            "compilationHash": compiled.sha256,
            "artifact": artifact,
            "bundleHash": _hash("bundle"),
            "effectiveRuntimeConfigHash": _hash("effective"),
            "contextHash": _hash("context"),
            "domainHash": _hash("domain"),
            "createdAt": None,
        }
    )
    stored = DomainCompilationArtifactV1.build(
        store.put_domain_compilation_artifact(record.to_dict())
    )
    return store, artifacts, stored


def test_compilation_lookup_fails_closed_for_missing_and_corrupt_artifacts(
    tmp_path: Path,
    persisted_compilation: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled = persisted_compilation["compiled"]
    payload = canonical_json(compiled.to_dict(include_payloads=True)).encode(
        "utf-8"
    )
    store, artifacts, record = _register_reference(
        tmp_path / "outer-integrity", compiled, payload=payload
    )
    repository = DomainCompilationArtifactRepositoryV1(
        store=store, artifact_store=artifacts, cache_size=1
    )
    try:
        assert repository.get(_hash("missing")) is None
        monkeypatch.setattr(
            artifacts,
            "get",
            lambda _uri: (_ for _ in ()).throw(FileNotFoundError("missing")),
        )
        with pytest.raises(FileNotFoundError):
            repository.get(compiled.sha256)

        original = payload
        corrupt = bytes([original[0] ^ 1]) + original[1:]
        monkeypatch.setattr(artifacts, "get", lambda _uri: corrupt)
        with pytest.raises(ValueError, match="checksum mismatch"):
            repository.get(compiled.sha256)
        assert record.artifact["sha256"] == hashlib.sha256(payload).hexdigest()
    finally:
        store.close()


def test_compilation_lookup_rejects_internal_target_and_compilation_mismatch(
    tmp_path: Path,
    persisted_compilation: dict[str, Any],
) -> None:
    compiled = persisted_compilation["compiled"]
    target_tamper = compiled.to_dict(include_payloads=True)
    target_tamper["targetPayloads"]["documentationModel"]["tampered"] = True
    target_bytes = canonical_json(target_tamper).encode("utf-8")
    target_store, target_artifacts, _record = _register_reference(
        tmp_path / "target-identity", compiled, payload=target_bytes
    )
    try:
        repository = DomainCompilationArtifactRepositoryV1(
            store=target_store,
            artifact_store=target_artifacts,
        )
        with pytest.raises(ValueError, match="documentationModel identity mismatch"):
            repository.get(compiled.sha256)
    finally:
        target_store.close()

    proof_tamper = compiled.to_dict(include_payloads=True)
    proof_id = next(iter(proof_tamper["targetPayloads"]["behavioralProofs"]))
    proof = proof_tamper["targetPayloads"]["behavioralProofs"][proof_id]
    proof["proofBoundary"] = "substituted-after-compilation"
    proof_core = dict(proof)
    proof_core.pop("proofHash")
    proof["proofHash"] = canonical_sha256(proof_core)
    proof_bytes = canonical_json(proof_tamper).encode("utf-8")
    proof_store, proof_artifacts, _record = _register_reference(
        tmp_path / "proof-identity",
        compiled,
        payload=proof_bytes,
    )
    try:
        repository = DomainCompilationArtifactRepositoryV1(
            store=proof_store,
            artifact_store=proof_artifacts,
        )
        with pytest.raises(ValueError, match="behavioral proof identity mismatch"):
            repository.get(compiled.sha256)
    finally:
        proof_store.close()

    compilation_tamper = compiled.to_dict(include_payloads=True)
    compilation_tamper["sha256"] = _hash("forged-compilation")
    compilation_bytes = canonical_json(compilation_tamper).encode("utf-8")
    identity_store, identity_artifacts, _record = _register_reference(
        tmp_path / "compilation-identity",
        compiled,
        payload=compilation_bytes,
    )
    try:
        repository = DomainCompilationArtifactRepositoryV1(
            store=identity_store,
            artifact_store=identity_artifacts,
        )
        with pytest.raises(ValueError, match="internal identity mismatch"):
            repository.get(compiled.sha256)
    finally:
        identity_store.close()


def test_invented_rule_receipt_is_rejected_without_replacing_immutable_artifact(
    tmp_path: Path, persisted_compilation: dict[str, Any],
) -> None:
    compiled = persisted_compilation["compiled"]
    value = compiled.to_dict(include_payloads=True)
    proof_core = {
        "schemaVersion": 1,
        "primitiveId": "rules.projects.tie_break_order",
        "status": "proven_invariant",
        "rulesHash": canonical_sha256(compiled.domain_ir.rules),
        "proofKind": "runtime_rulebook_counterexample_rejection",
        "targetsObserved": ["runtime", "reference", "cpSat"],
        "proofBoundary": "shared_rulebook_admission_gate",
    }
    assert proof_core["primitiveId"] not in {
        row.primitive_id for row in compiled.coverage.rows
    }
    value["targetPayloads"]["behavioralProofs"][proof_core["primitiveId"]] = {
        **proof_core, "proofHash": canonical_sha256(proof_core),
    }
    store, artifacts, record = _register_reference(
        tmp_path, compiled, payload=canonical_json(value).encode("utf-8"),
    )
    try:
        repository = DomainCompilationArtifactRepositoryV1(store=store, artifact_store=artifacts)
        with pytest.raises(ValueError, match="behavioral proof coverage mismatch"):
            repository.get(compiled.sha256)
        with pytest.raises(ValueError, match="identity is immutable"):
            repository.register(
                compiled, bundle_hash=record.bundle_hash,
                effective_runtime_config_hash=record.effective_runtime_config_hash,
                context_hash=record.context_hash, domain_hash=record.domain_hash,
            )
        assert store.get_domain_compilation_artifact(compiled.sha256) == record.to_dict()
    finally:
        store.close()


class _Result:
    def __init__(self, row: dict[str, Any] | None):
        self.row = row

    def fetchone(self) -> dict[str, Any] | None:
        return self.row


class _FakePostgresConnection:
    def __init__(self) -> None:
        self.record: dict[str, Any] | None = None

    def execute(
        self, sql: str, parameters: tuple[Any, ...] | None = None
    ) -> _Result:
        if any(
            marker in sql
            for marker in (
                "pg_advisory_lock",
                "pg_advisory_unlock",
                "SET TRANSACTION ISOLATION LEVEL SERIALIZABLE",
                "set_config",
            )
        ):
            return _Result(None)
        assert parameters is not None
        if "INSERT INTO research.domain_compilation_artifacts" in sql:
            if self.record is not None:
                return _Result(None)
            self.record = json.loads(str(parameters[2]))
            return _Result({"record_json": self.record})
        if "FROM research.domain_compilation_artifacts" in sql:
            requested = str(parameters[0])
            if self.record is None or self.record["compilationHash"] != requested:
                return _Result(None)
            return _Result({"record_json": self.record})
        raise AssertionError(sql)

    @contextmanager
    def transaction(self):
        yield self

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass

    def close(self) -> None:
        pass


class _FakePostgresPool:
    def __init__(self, connection: _FakePostgresConnection) -> None:
        self._connection = connection

    @contextmanager
    def connection(self):
        yield self._connection


def test_sqlite_postgres_compilation_artifact_contract_parity(
    tmp_path: Path,
    persisted_compilation: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert inspect.signature(
        SQLiteResearchStoreV1.put_domain_compilation_artifact
    ) == inspect.signature(PostgresResearchStoreV1.put_domain_compilation_artifact)
    assert inspect.signature(
        SQLiteResearchStoreV1.get_domain_compilation_artifact
    ) == inspect.signature(PostgresResearchStoreV1.get_domain_compilation_artifact)

    compiled = persisted_compilation["compiled"]
    payload = canonical_json(compiled.to_dict(include_payloads=True)).encode(
        "utf-8"
    )
    sqlite_store, sqlite_artifacts, sqlite_record = _register_reference(
        tmp_path / "sqlite", compiled, payload=payload
    )
    changed_provenance = {
        **sqlite_record.to_dict(),
        "bundleHash": _hash("other-bundle"),
        "effectiveRuntimeConfigHash": _hash("other-effective"),
        "contextHash": _hash("other-context"),
        "domainHash": _hash("other-domain"),
        "createdAt": None,
    }
    try:
        assert sqlite_store.put_domain_compilation_artifact(
            changed_provenance
        ) == sqlite_record.to_dict()
        conflicting = {
            **sqlite_record.to_dict(),
            "artifact": sqlite_artifacts.put(
                payload + b"x",
                media_type=DOMAIN_COMPILATION_MEDIA_TYPE,
                name="conflicting.json",
            ),
            "createdAt": None,
        }
        with pytest.raises(ValueError, match="identity is immutable"):
            sqlite_store.put_domain_compilation_artifact(conflicting)
    finally:
        sqlite_store.close()

    connection = _FakePostgresConnection()

    @contextmanager
    def fake_connection(_pool: object):
        yield connection

    monkeypatch.setattr(
        "baloot_platform.research.postgres_store.contextual_connection",
        fake_connection,
    )
    postgres = PostgresResearchStoreV1.__new__(PostgresResearchStoreV1)
    postgres.pool = _FakePostgresPool(connection)
    postgres_record = postgres.put_domain_compilation_artifact(
        {**sqlite_record.to_dict(), "createdAt": None}
    )
    assert postgres.get_domain_compilation_artifact(compiled.sha256) == postgres_record

    assert postgres.put_domain_compilation_artifact(changed_provenance) == (
        postgres_record
    )
    with pytest.raises(ValueError, match="identity is immutable"):
        postgres.put_domain_compilation_artifact(conflicting)

    source = Path("migrations/020_research_domain_compilation_artifacts.sql")
    packaged = Path(
        "src/baloot_platform/resources/migrations/"
        "020_research_domain_compilation_artifacts.sql"
    )
    assert source.read_bytes() == packaged.read_bytes()
    assert "PRIMARY KEY" in source.read_text(encoding="utf-8").upper()
