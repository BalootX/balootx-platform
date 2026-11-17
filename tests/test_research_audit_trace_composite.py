from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import pytest

from baloot_platform.research.audit_trace import (
    COMPLETE_TRACE_MEDIA_TYPE,
    CompositeResearchAuditTraceRepositoryV1,
    ResearchAuditTraceRecordV1,
    ResearchAuditTraceRepositoryV1,
    _source,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _source_value(
    kind: str,
    *,
    run_code: str = "RUN-ORGANIC-1",
    candidate_ordinal: int = 0,
) -> dict[str, Any]:
    source: dict[str, Any] = {
        "kind": kind,
        "runCode": run_code,
        "runHash": _hash(f"run:{run_code}"),
        "recipeCode": "RCP-ORGANIC-1",
        "recipeHash": _hash("recipe"),
        "shardId": f"{run_code}-shard-0",
        "shardHash": _hash(f"shard:{run_code}"),
        "shardIndex": 0,
        "seed": candidate_ordinal + 1,
        "candidateHash": _hash(f"candidate:{run_code}:{candidate_ordinal}"),
        "candidateOrdinal": candidate_ordinal,
        "generationLane": "organic",
        "attemptCount": 1,
        "fencingToken": 1,
        "workerId": "organic-worker-1",
    }
    if kind == "organic_foundry_trace":
        source["traceRole"] = "scored_final"
    elif kind == "organic_finalist_trace":
        source.update(
            {
                "traceRole": "finalist_descendant",
                "parentRecordHash": _hash("parent-record"),
                "parentTraceHash": _hash("parent-trace"),
                "stageProfileHash": _hash("stage-profile"),
                "stageRunHash": _hash("stage-run"),
            }
        )
    return source


def _record(
    source: Mapping[str, Any],
    *,
    trace_hash: str,
    created_at: str,
) -> ResearchAuditTraceRecordV1:
    normalized_source = _source(source)
    record_hash = hashlib.sha256(
        json.dumps(
            {
                "schemaVersion": 1,
                "traceHash": trace_hash,
                "source": normalized_source,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    artifact_hash = _hash(f"artifact:{trace_hash}")
    return ResearchAuditTraceRecordV1.build(
        {
            "schemaVersion": 1,
            "recordHash": record_hash,
            "auditTraceCode": f"LAF-AUD-{record_hash[:10].upper()}",
            "traceHash": trace_hash,
            "completeTrace": {
                "uri": f"cas://sha256/{artifact_hash}",
                "sha256": artifact_hash,
                "bytes": 1,
                "mediaType": COMPLETE_TRACE_MEDIA_TYPE,
                "name": "trace.json",
            },
            "source": normalized_source,
            "disposition": "qualified",
            "status": "finalized",
            "failureCategories": [],
            "disagreementReferences": [],
            "replayValidation": {
                "schemaVersion": 1,
                "status": "verified",
                "validator": "authoritative_complete_trace_replay_v1",
                "traceHash": trace_hash,
                "gameHash": _hash(f"game:{trace_hash}"),
                "finalStateHash": _hash(f"state:{trace_hash}"),
                "eventChainHash": _hash(f"events:{trace_hash}"),
                "decisionCount": 1,
            },
            "observerSafety": {
                "schemaVersion": 1,
                "status": "verified",
                "validator": "isolated_observer_projection_v1",
                "traceHash": trace_hash,
                "projectedDecisionCount": 1,
                "trustedTraceExposure": "internal_only",
            },
            "configurationProvenance": {
                "schemaVersion": 1,
                "traceHash": trace_hash,
                "bundleHash": _hash("bundle"),
                "rulesHash": _hash("rules"),
                "engineVersion": "test-engine",
                "configurationSnapshotHash": _hash("snapshot"),
                "researchConfiguration": None,
            },
            "attemptStatus": "completed",
            "auditOnly": True,
            "publicationEligible": False,
            "createdAt": created_at,
            "updatedAt": created_at,
        }
    )


class _FakeRepository:
    def __init__(
        self,
        name: str,
        records: list[ResearchAuditTraceRecordV1],
        *,
        page_cap: int = 500,
    ) -> None:
        self.name = name
        self.records = records
        self.page_cap = page_cap
        self.retain_calls: list[tuple[Any, dict[str, Any]]] = []
        self.finalize_calls: list[tuple[str, dict[str, Any]]] = []
        self.load_calls: list[str] = []

    def retain(self, trace: Any, **kwargs: Any) -> ResearchAuditTraceRecordV1:
        self.retain_calls.append((trace, kwargs))
        return self.records[0]

    def finalize(self, identifier: str, **kwargs: Any) -> ResearchAuditTraceRecordV1:
        self.finalize_calls.append((identifier, kwargs))
        return self.records[0]

    def get_record(
        self, identifier: str, *, include_unadmitted: bool = False
    ) -> ResearchAuditTraceRecordV1 | None:
        del include_unadmitted
        matches = [
            record
            for record in self.records
            if identifier
            in {record.record_hash, record.trace_hash, record.audit_trace_code}
        ]
        return max(matches, key=lambda item: item.created_at or "") if matches else None

    def list_records(
        self,
        *,
        run_code: str | None = None,
        disposition: str | None = None,
        status: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        records = [
            record
            for record in self.records
            if (run_code is None or record.source["runCode"] == run_code)
            and (disposition is None or record.disposition == disposition)
            and (status is None or record.status == status)
        ]
        records.sort(key=lambda item: item.record_hash)
        records.sort(key=lambda item: item.created_at or "", reverse=True)
        size = min(int(limit), self.page_cap)
        items = records[offset : offset + size]
        next_offset = offset + len(items)
        return {
            "items": [item.to_dict() for item in items],
            "total": len(records),
            "offset": offset,
            "limit": size,
            "nextOffset": next_offset if next_offset < len(records) else None,
        }

    def load_trace(self, identifier: str) -> str:
        self.load_calls.append(identifier)
        return f"{self.name}:{identifier}"


class _PagedStore:
    def __init__(self, records: list[ResearchAuditTraceRecordV1]) -> None:
        self.records = records

    def list_audit_traces(
        self,
        *,
        run_code: str | None = None,
        disposition: str | None = None,
        status: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        records = [
            record
            for record in self.records
            if (run_code is None or record.source["runCode"] == run_code)
            and (disposition is None or record.disposition == disposition)
            and (status is None or record.status == status)
        ]
        size = min(limit, 1)
        items = records[offset : offset + size]
        next_offset = offset + len(items)
        return {
            "items": [item.to_dict() for item in items],
            "total": len(records),
            "offset": offset,
            "limit": size,
            "nextOffset": next_offset if next_offset < len(records) else None,
        }


@pytest.mark.parametrize("trace_role", ["scored_final", "all_pass_attempt"])
def test_organic_foundry_source_contract_is_closed(trace_role: str) -> None:
    source = _source_value("organic_foundry_trace")
    source["traceRole"] = trace_role

    assert _source(source) == source

    with pytest.raises(ValueError, match="keys are invalid"):
        _source({**source, "parentTraceHash": _hash("not-allowed")})
    with pytest.raises(ValueError, match="role is invalid"):
        _source({**source, "traceRole": "finalist_descendant"})


def test_candidate_and_finalist_source_contracts_are_strict_and_hash_bound() -> None:
    candidate = _source_value("research_candidate")
    finalist = _source_value("organic_finalist_trace")

    assert _source(candidate) == candidate
    assert _source(finalist) == finalist

    with pytest.raises(ValueError, match="keys are invalid"):
        _source({**candidate, "traceRole": "scored_final"})
    with pytest.raises(ValueError, match="keys are invalid"):
        missing = dict(finalist)
        missing.pop("parentRecordHash")
        _source(missing)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _source({**finalist, "stageRunHash": finalist["stageRunHash"].upper()})
    with pytest.raises(ValueError, match="role is invalid"):
        _source({**finalist, "traceRole": "scored_final"})


def test_composite_repository_merges_pages_and_resolves_owning_repository() -> None:
    shared_trace_hash = _hash("shared-trace")
    primary_occurrence = _record(
        _source_value("research_candidate", run_code="RUN-A", candidate_ordinal=0),
        trace_hash=shared_trace_hash,
        created_at="2026-08-09T10:00:00+00:00",
    )
    secondary_occurrence = _record(
        _source_value(
            "organic_foundry_trace", run_code="RUN-B", candidate_ordinal=1
        ),
        trace_hash=shared_trace_hash,
        created_at="2026-08-09T12:00:00+00:00",
    )
    middle = _record(
        _source_value(
            "organic_finalist_trace", run_code="RUN-A", candidate_ordinal=2
        ),
        trace_hash=_hash("finalist-trace"),
        created_at="2026-08-09T11:00:00+00:00",
    )
    primary = _FakeRepository("primary", [primary_occurrence])
    secondary = _FakeRepository(
        "secondary",
        [secondary_occurrence, middle, primary_occurrence],
        page_cap=1,
    )
    repository = CompositeResearchAuditTraceRepositoryV1(
        primary,
        secondary_repositories=(secondary,),
    )

    page = repository.list_records(offset=1, limit=1)
    assert page == {
        "items": [middle.to_dict()],
        "total": 3,
        "offset": 1,
        "limit": 1,
        "nextOffset": 2,
    }
    assert [
        item["recordHash"]
        for item in repository.list_records(run_code="RUN-A")["items"]
    ] == [middle.record_hash, primary_occurrence.record_hash]

    assert repository.get_record(secondary_occurrence.record_hash) == secondary_occurrence
    assert repository.get_record(secondary_occurrence.audit_trace_code) == secondary_occurrence
    assert repository.get_record(shared_trace_hash) == primary_occurrence
    assert repository.load_trace(secondary_occurrence.record_hash) == (
        f"secondary:{secondary_occurrence.record_hash}"
    )
    assert secondary.load_calls == [secondary_occurrence.record_hash]
    assert repository.find_record(shared_trace_hash, run_code="RUN-B") == (
        secondary_occurrence
    )
    assert repository.find_record_by_trace_hash_and_run_code(
        shared_trace_hash, run_code="RUN-B"
    ) == secondary_occurrence


def test_single_store_find_record_pages_and_breaks_newest_ties_deterministically() -> None:
    trace_hash = _hash("single-store-trace")
    first = _record(
        _source_value(
            "organic_foundry_trace", run_code="RUN-SINGLE", candidate_ordinal=1
        ),
        trace_hash=trace_hash,
        created_at="2026-08-09T12:00:00+00:00",
    )
    second = _record(
        _source_value(
            "organic_foundry_trace", run_code="RUN-SINGLE", candidate_ordinal=2
        ),
        trace_hash=trace_hash,
        created_at="2026-08-09T12:00:00+00:00",
    )
    older = _record(
        _source_value(
            "organic_foundry_trace", run_code="RUN-SINGLE", candidate_ordinal=3
        ),
        trace_hash=trace_hash,
        created_at="2026-08-09T11:00:00+00:00",
    )
    repository = ResearchAuditTraceRepositoryV1(
        store=_PagedStore(
            [
                older,
                max(first, second, key=lambda item: item.record_hash),
                min(first, second, key=lambda item: item.record_hash),
            ]
        ),
        artifact_store=object(),
    )

    assert repository.find_record(trace_hash, run_code="RUN-SINGLE") == min(
        first, second, key=lambda item: item.record_hash
    )
    assert repository.find_record(trace_hash, run_code="RUN-MISSING") is None


def test_composite_mutations_are_primary_only() -> None:
    record = _record(
        _source_value("research_candidate"),
        trace_hash=_hash("primary-only"),
        created_at="2026-08-09T10:00:00+00:00",
    )
    primary = _FakeRepository("primary", [record])
    secondary = _FakeRepository("secondary", [record])
    repository = CompositeResearchAuditTraceRepositoryV1(
        primary,
        secondary_repositories=(secondary,),
    )
    trace = object()

    assert repository.retain(
        trace,
        rules="rules",
        source=record.source,
        research_configuration=None,
    ) == record
    assert repository.finalize(
        record.record_hash,
        disposition="qualified",
    ) == record
    assert len(primary.retain_calls) == 1
    assert len(primary.finalize_calls) == 1
    assert secondary.retain_calls == []
    assert secondary.finalize_calls == []
