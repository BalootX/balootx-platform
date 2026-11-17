from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from baloot_platform.challenge.evidence_v3 import canonical_sha256
from baloot_platform.challenge.finalist_validation_service import (
    FinalistValidationJobServiceV1,
)


def _digest(label: str) -> str:
    return canonical_sha256({"label": label})


class _AuditRepository:
    def __init__(self, parent: Mapping[str, Any] | None, artifact_sha256: str):
        self.parent = parent
        self.artifact_sha256 = artifact_sha256
        self.find_calls: list[tuple[str, str]] = []
        self.retain_calls: list[dict[str, Any]] = []
        self.finalize_calls: list[dict[str, Any]] = []

    def find_record(self, trace_hash: str, *, run_code: str):
        self.find_calls.append((trace_hash, run_code))
        return self.parent

    def retain(self, trace, *, rules, source, research_configuration):
        self.retain_calls.append(
            {
                "trace": trace,
                "rules": rules,
                "source": dict(source),
                "researchConfiguration": research_configuration,
            }
        )
        return {
            "recordHash": _digest("finalist-record"),
            "completeTrace": {"sha256": self.artifact_sha256},
        }

    def finalize(self, record_hash: str, **values: Any):
        self.finalize_calls.append({"recordHash": record_hash, **values})


def _service(repository: _AuditRepository) -> FinalistValidationJobServiceV1:
    service = object.__new__(FinalistValidationJobServiceV1)
    service.audit_repository = repository
    service.profile = SimpleNamespace(sha256=_digest("finalist-profile"))
    service.replay = SimpleNamespace(
        simulator=SimpleNamespace(rules=SimpleNamespace(sha256=_digest("rules")))
    )
    return service


def _parent(campaign_hash: str, trace_hash: str) -> dict[str, Any]:
    return {
        "recordHash": _digest("parent-record"),
        "traceHash": trace_hash,
        "source": {
            "kind": "organic_foundry_trace",
            "runCode": campaign_hash,
            "runHash": campaign_hash,
            "recipeCode": "organic-campaign-v1",
            "recipeHash": _digest("campaign-profile"),
            "shardId": _digest("shard"),
            "shardHash": _digest("shard"),
            "shardIndex": 2,
            "seed": 41,
            "candidateHash": _digest("parent-candidate"),
            "candidateOrdinal": 3,
            "generationLane": "organic_campaign",
            "attemptCount": 2,
            "fencingToken": 7,
            "workerId": "organic-worker",
        },
        "configurationProvenance": {
            "researchConfiguration": {"schemaVersion": 1, "frozen": True}
        },
    }


def test_finalist_projection_links_parent_reuses_cas_and_preserves_gate_outcome():
    campaign_hash = _digest("campaign")
    parent_trace_hash = _digest("parent-trace")
    artifact_sha256 = _digest("trace-artifact")
    repository = _AuditRepository(
        _parent(campaign_hash, parent_trace_hash),
        artifact_sha256,
    )
    service = _service(repository)
    descendant = SimpleNamespace(sha256=_digest("descendant-trace"))
    source_parent = SimpleNamespace(sha256=parent_trace_hash)
    request = {
        "schemaVersion": 1,
        "analysisJobId": "analysis-1",
        "candidateHashes": [_digest("candidate")],
        "profileHash": service.profile.sha256,
    }

    record = service._retain_finalist_audit_trace(
        descendant,
        source_parent=source_parent,
        campaign_hash=campaign_hash,
        request=request,
        trace_reference={"sha256": artifact_sha256},
    )
    blockers = (
        "professional_agent_qualification_missing",
        "laffa_selection_failed",
    )
    service._finalize_finalist_audit_trace(
        record,
        publication_ready=False,
        blockers=blockers,
    )

    assert repository.find_calls == [(parent_trace_hash, campaign_hash)]
    retained = repository.retain_calls[0]
    source = retained["source"]
    stage_run_hash = canonical_sha256(request)
    assert source == {
        "kind": "organic_finalist_trace",
        "runCode": campaign_hash,
        "runHash": campaign_hash,
        "shardId": _digest("shard"),
        "shardHash": _digest("shard"),
        "shardIndex": 2,
        "seed": 41,
        "candidateOrdinal": 3,
        "attemptCount": 2,
        "fencingToken": 7,
        "workerId": "organic-worker",
        "recipeCode": "organic-finalist-validation-v1",
        "recipeHash": service.profile.sha256,
        "candidateHash": canonical_sha256(
            {
                "schemaVersion": 1,
                "parentRecordHash": _digest("parent-record"),
                "stageProfileHash": service.profile.sha256,
                "stageRunHash": stage_run_hash,
            }
        ),
        "generationLane": "organic_finalist",
        "traceRole": "finalist_descendant",
        "parentRecordHash": _digest("parent-record"),
        "parentTraceHash": parent_trace_hash,
        "stageProfileHash": service.profile.sha256,
        "stageRunHash": stage_run_hash,
    }
    assert retained["trace"] is descendant
    assert retained["rules"] is service.replay.simulator.rules
    assert retained["researchConfiguration"] == {
        "schemaVersion": 1,
        "frozen": True,
    }
    assert repository.finalize_calls == [
        {
            "recordHash": _digest("finalist-record"),
            "disposition": "rejected",
            "failure_categories": blockers,
        }
    ]


def test_finalist_projection_skips_legacy_parent_and_fails_closed_on_cas_drift():
    campaign_hash = _digest("campaign")
    parent_trace_hash = _digest("parent-trace")
    source_parent = SimpleNamespace(sha256=parent_trace_hash)
    descendant = SimpleNamespace(sha256=_digest("descendant-trace"))
    request = {"schemaVersion": 1, "candidateHashes": [_digest("candidate")]}
    legacy_repository = _AuditRepository(None, _digest("trace-artifact"))
    legacy_service = _service(legacy_repository)

    assert (
        legacy_service._retain_finalist_audit_trace(
            descendant,
            source_parent=source_parent,
            campaign_hash=campaign_hash,
            request=request,
            trace_reference={"sha256": _digest("trace-artifact")},
        )
        is None
    )
    assert legacy_repository.retain_calls == []

    invalid_parent = _parent(campaign_hash, parent_trace_hash)
    invalid_parent["source"]["kind"] = "organic_finalist_trace"
    invalid_repository = _AuditRepository(
        invalid_parent,
        _digest("trace-artifact"),
    )
    with pytest.raises(ValueError, match="parent identity mismatch"):
        _service(invalid_repository)._retain_finalist_audit_trace(
            descendant,
            source_parent=source_parent,
            campaign_hash=campaign_hash,
            request=request,
            trace_reference={"sha256": _digest("trace-artifact")},
        )
    assert invalid_repository.retain_calls == []

    drift_repository = _AuditRepository(
        _parent(campaign_hash, parent_trace_hash),
        _digest("different-artifact"),
    )
    drift_service = _service(drift_repository)
    with pytest.raises(ValueError, match="reuse the retained CAS bytes"):
        drift_service._retain_finalist_audit_trace(
            descendant,
            source_parent=source_parent,
            campaign_hash=campaign_hash,
            request=request,
            trace_reference={"sha256": _digest("trace-artifact")},
        )


def test_finalist_projection_qualifies_only_publication_ready_trace():
    repository = _AuditRepository(None, _digest("trace-artifact"))
    service = _service(repository)
    record = {"recordHash": _digest("finalist-record")}

    service._finalize_finalist_audit_trace(
        record,
        publication_ready=True,
        blockers=(),
    )

    assert repository.finalize_calls == [
        {
            "recordHash": _digest("finalist-record"),
            "disposition": "qualified",
            "failure_categories": (),
        }
    ]
    with pytest.raises(ValueError, match="requires exact blockers"):
        service._finalize_finalist_audit_trace(
            record,
            publication_ready=False,
            blockers=(),
        )
