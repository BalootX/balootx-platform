from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from baloot_platform.challenge.evidence_v3 import (
    DecisionLedgerV1,
    TruthMode,
    canonical_sha256,
)
from baloot_platform.challenge.trace import CompleteGameTraceV1
from baloot_platform.core.enums import Seat
from baloot_platform.research import render_service as render_module
from baloot_platform.research.audit_trace import (
    COMPLETE_TRACE_MEDIA_TYPE,
    ResearchAuditTraceRecordV1,
)
from baloot_platform.research.render_service import ResearchRenderPreparationServiceV1
from baloot_platform.service.laffa_library import LaffaLibraryServiceV1


def _digest(character: str) -> str:
    return character * 64


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _trace(*, game_character: str = "a") -> CompleteGameTraceV1:
    trusted = {
        "rulesHash": _digest("b"),
        "seed": 7,
        "dealer": "east",
        "shuffle": {
            "algorithm": "splitmix64_fisher_yates_v1",
            "seed": 7,
            "dealer": "east",
            "cutter": "south",
            "cutPosition": 9,
            "deckCommitment": _digest("c"),
            "deckOrder": list(range(32)),
        },
        "agentProfiles": {
            seat: "legacy" for seat in ("north", "west", "south", "east")
        },
        "privateHands": {"east": ["SECRET-HIDDEN-CARD"]},
        "contract": "hokum",
        "buyer": "south",
        "events": [
            {"sequence": 0, "type": "round_created", "dealer": "east"},
            {"sequence": 1, "type": "card_played", "seat": "north", "card": "S7"},
        ],
    }
    ledger = DecisionLedgerV1.build(
        _digest(game_character),
        (
            {
                "decisionId": "D1",
                "ordinal": 1,
                "actor": "north",
                "decisionKind": "play",
                "phase": "playing",
                "publicState": {"observer": "north"},
                "completeStateHash": _digest("d"),
                "legalActions": [
                    {"suit": "spades", "rank": "seven"},
                    {"suit": "spades", "rank": "eight"},
                ],
                "selectedAction": {"suit": "spades", "rank": "seven"},
            },
        ),
    )
    return CompleteGameTraceV1.build(
        game_record={
            "gameHash": _digest(game_character),
            "gameId": f"GAME-{game_character}",
            "stateHash": canonical_sha256(trusted),
            "bundleHash": _digest("e"),
            "engineVersion": "1.0.0",
        },
        trusted_final_state=trusted,
        decision_ledger=ledger,
    )


def _trace_payload(trace: CompleteGameTraceV1) -> tuple[bytes, dict[str, Any]]:
    payload = _canonical_bytes(trace.to_dict(trusted=True))
    digest = hashlib.sha256(payload).hexdigest()
    return payload, {
        "uri": f"cas://sha256/{digest}",
        "sha256": digest,
        "bytes": len(payload),
        "mediaType": COMPLETE_TRACE_MEDIA_TYPE,
        "name": f"{trace.sha256}.complete-trace.json",
    }


def _audit_record(
    trace: CompleteGameTraceV1,
    *,
    attempt: int,
    disposition: str = "rejected",
    failures: tuple[str, ...] = ("decision_quality_gate_failed",),
) -> dict[str, Any]:
    _, reference = _trace_payload(trace)
    source = {
        "kind": "research_candidate",
        "runCode": "RUN-AUDIT",
        "runHash": _digest("1"),
        "recipeCode": "RCP-AUDIT",
        "recipeHash": _digest("2"),
        "shardId": "SHARD-AUDIT",
        "shardHash": _digest("3"),
        "shardIndex": 0,
        "seed": 7,
        "candidateHash": hashlib.sha256(f"candidate-{attempt}".encode()).hexdigest(),
        "candidateOrdinal": attempt - 1,
        "generationLane": "organic_runtime",
        "attemptCount": attempt,
        "fencingToken": attempt,
        "workerId": f"worker-{attempt}",
    }
    record_hash = hashlib.sha256(
        _canonical_bytes(
            {"schemaVersion": 1, "traceHash": trace.sha256, "source": source}
        )
    ).hexdigest()
    return ResearchAuditTraceRecordV1.build(
        {
            "schemaVersion": 1,
            "auditTraceCode": f"LAF-AUD-{record_hash[:10].upper()}",
            "recordHash": record_hash,
            "traceHash": trace.sha256,
            "completeTrace": reference,
            "source": source,
            "disposition": disposition,
            "status": "provisional" if disposition == "provisional" else "finalized",
            "failureCategories": list(failures),
            "disagreementReferences": [],
            "replayValidation": {
                "schemaVersion": 1,
                "status": "verified",
                "validator": "authoritative_complete_trace_replay_v1",
                "traceHash": trace.sha256,
                "gameHash": trace.game_hash,
                "finalStateHash": trace.state_hash,
                "eventChainHash": _digest("4"),
                "decisionCount": len(trace.decision_ledger.records),
            },
            "observerSafety": {
                "schemaVersion": 1,
                "status": "verified",
                "validator": "isolated_observer_projection_v1",
                "traceHash": trace.sha256,
                "projectedDecisionCount": len(trace.decision_ledger.records),
                "trustedTraceExposure": "internal_only",
            },
            "configurationProvenance": {
                "schemaVersion": 1,
                "traceHash": trace.sha256,
                "bundleHash": trace.bundle_hash,
                "rulesHash": trace.rules_hash,
                "engineVersion": trace.engine_version,
                "configurationSnapshotHash": _digest("5"),
                "researchConfiguration": {"recipeRevision": "published-v1"},
            },
            "attemptStatus": "completed",
            "auditOnly": True,
            "publicationEligible": False,
            "createdAt": f"2026-08-09T00:00:0{attempt}+00:00",
            "updatedAt": f"2026-08-09T00:00:0{attempt}+00:00",
        }
    ).to_dict()


class _Artifacts:
    def __init__(self, values: Mapping[str, bytes]) -> None:
        self.values = dict(values)

    def get(self, uri: str) -> bytes:
        if uri not in self.values:
            raise FileNotFoundError(uri)
        return self.values[uri]


class _Research:
    def __init__(
        self,
        records: list[dict[str, Any]],
        trace: CompleteGameTraceV1,
        *,
        results: list[dict[str, Any]] | None = None,
        load_mode: str = "valid",
    ) -> None:
        self.records = copy.deepcopy(records)
        self.trace = trace
        self.results = copy.deepcopy(results or [])
        self.load_mode = load_mode
        payload, reference = _trace_payload(trace)
        self.artifact_store = _Artifacts({str(reference["uri"]): payload})

    def list_results(self, *, sort: str, offset: int, limit: int) -> dict[str, Any]:
        assert sort == "newest"
        items = self.results[offset : offset + limit]
        return {"items": copy.deepcopy(items), "nextOffset": None}

    def list_audit_traces(self, *, offset: int, limit: int) -> dict[str, Any]:
        items = self.records[offset : offset + limit]
        return {"items": copy.deepcopy(items), "nextOffset": None}

    def get_audit_trace(self, identifier: str) -> dict[str, Any] | None:
        for record in self.records:
            if identifier in {
                record["auditTraceCode"],
                record["recordHash"],
                record["traceHash"],
            }:
                return copy.deepcopy(record)
        return None

    def load_audit_trace(self, _identifier: str) -> CompleteGameTraceV1:
        if self.load_mode == "missing":
            raise FileNotFoundError("retained trace is missing")
        if self.load_mode == "corrupt":
            return _trace(game_character="f")
        return self.trace


class _Learning:
    def list_games(self, **_filters: Any) -> list[dict[str, Any]]:
        return []

    def list_video_records(self, **_filters: Any) -> list[dict[str, Any]]:
        return []


def _library(research: _Research) -> LaffaLibraryServiceV1:
    return LaffaLibraryServiceV1(
        learning_store=_Learning(),
        research=research,
        foundry_catalogs=None,
        render_batches=None,
    )


def test_audit_traces_are_discoverable_and_duplicate_occurrences_remain_visible() -> None:
    trace = _trace()
    first = _audit_record(trace, attempt=1)
    second = _audit_record(trace, attempt=2)
    service = _library(_Research([first, second], trace))

    response = service.list(source_kind="research_audit_trace", limit=10)

    assert response["total"] == 1
    item = response["items"][0]
    assert item["sourceKind"] == "research_audit_trace"
    assert item["sourceKinds"] == ["research_audit_trace"]
    assert item["replayValid"] is True
    assert item["qualificationStatus"] == "rejected"
    assert item["auditOnly"] is True
    assert item["publicationEligible"] is False
    assert {
        "audit_trace_publication_not_authorized",
        "decision_quality_gate_failed",
    } <= set(item["qualificationFailures"])
    assert item["traceHash"] == trace.sha256
    audit_references = [
        reference
        for reference in item["sourceReferences"]
        if reference["kind"] == "research_audit_trace"
    ]
    assert {reference["identity"] for reference in audit_references} == {
        first["recordHash"],
        second["recordHash"],
    }
    assert "completeTrace" not in json.dumps(item)
    assert service.get(first["auditTraceCode"]) == item
    assert service.get(second["recordHash"]) == item


def test_full_result_replaces_the_entire_audit_governance_tuple() -> None:
    trace = _trace()
    record = _audit_record(trace, attempt=1)
    payload, reference = _trace_payload(trace)
    result = {
        "resultCode": "LAF-QUALIFIED-RESULT",
        "resultHash": _digest("6"),
        "traceHash": trace.sha256,
        "gameHash": trace.game_hash,
        "disposition": "qualified",
        "publicationEligible": True,
        "auditOnly": False,
        "qualification": {"failures": []},
        "truthLabel": "exact_optimal",
        "confidence": 0.99,
        "score": 98.0,
        "libraryProjection": {
            "contract": "hokum",
            "actingSeats": ["north"],
            "projectKinds": [],
        },
        "artifacts": {"completeTrace": reference},
        "createdAt": "2026-08-09T01:00:00+00:00",
    }
    research = _Research([record], trace, results=[result])
    research.artifact_store = _Artifacts({str(reference["uri"]): payload})

    item = _library(research).list(limit=10)["items"][0]

    assert item["sourceKind"] == "research_result"
    assert item["sourceKinds"] == ["research_audit_trace", "research_result"]
    assert item["qualificationStatus"] == "qualified"
    assert item["auditOnly"] is False
    assert item["publicationEligible"] is True
    assert item["qualificationFailures"] == []
    assert item["truthLabel"] == "exact_optimal"
    assert "decision_quality_gate_failed" not in item["qualificationFailures"]


def test_audit_trace_resolves_by_human_code_record_hash_and_trace_hash() -> None:
    trace = _trace()
    record = _audit_record(trace, attempt=1)
    service = _library(_Research([record], trace))

    for identifier in (
        record["auditTraceCode"],
        record["recordHash"],
        record["traceHash"],
    ):
        assert service.resolve_audit_trace(identifier) == record
    assert service.resolve_audit_trace("LAF-AUD-MISSING") is None


@pytest.mark.parametrize(
    ("load_mode", "failure"),
    (
        ("missing", "audit_trace_artifact_unavailable"),
        ("corrupt", "audit_trace_integrity_check_failed"),
    ),
)
def test_missing_or_corrupt_audit_trace_remains_visible_but_fails_closed(
    load_mode: str, failure: str
) -> None:
    trace = _trace()
    record = _audit_record(trace, attempt=1)
    service = _library(_Research([record], trace, load_mode=load_mode))

    item = service.list(limit=10)["items"][0]

    assert item["replayValid"] is False
    assert item["auditOnly"] is True
    assert item["publicationEligible"] is False
    assert failure in item["qualificationFailures"]
    with pytest.raises(ValueError, match="checksum-bound replay loading"):
        service.resolve_audit_trace(record["auditTraceCode"])


class _LocatorService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def publish(self, *_args: Any, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(dict(kwargs))
        return SimpleNamespace(
            code="BX-AUDIT-LOCATOR",
            universal_link="https://example.test/c/BX-AUDIT-LOCATOR",
            to_dict=lambda: {
                "code": "BX-AUDIT-LOCATOR",
                "universalLink": "https://example.test/c/BX-AUDIT-LOCATOR",
            },
        )


def test_prepare_audit_trace_uses_requested_observer_and_freezes_audit_governance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _trace()
    record = _audit_record(trace, attempt=1)
    payload, reference = _trace_payload(trace)
    research = _Research([record], trace)
    artifacts = _Artifacts({str(reference["uri"]): payload})
    locator = _LocatorService()
    actor = SimpleNamespace(
        decision_id="D1",
        acting_seat=Seat.NORTH,
        observer_seat=Seat.NORTH,
        truth_mode=TruthMode.ACTOR_OBSERVER,
        prompt_contract="choose_acting_action",
        recommended_action={"suit": "spades", "rank": "seven"},
    )
    observer = SimpleNamespace(
        decision_id="D1",
        acting_seat=Seat.NORTH,
        observer_seat=Seat.SOUTH,
        truth_mode=TruthMode.SEAT_EXPERIENCE,
        prompt_contract="predict_opponent_play",
        recommended_action=None,
    )
    monkeypatch.setattr(
        render_module,
        "TraceDecisionProjectionStoreV1",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        render_module,
        "build_decision_perspective_evidence",
        lambda *_args, **_kwargs: (actor, observer),
    )
    service = ResearchRenderPreparationServiceV1(
        research=research,
        artifact_store=artifacts,
        publication_service=SimpleNamespace(),
        renderer=SimpleNamespace(),
        rules=SimpleNamespace(sha256=trace.rules_hash),
        locator_service=locator,
    )
    captured: dict[str, Any] = {}

    def build_and_store(**kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        captured.update(kwargs)
        captured["source_metadata"] = copy.deepcopy(kwargs["source_metadata"])
        return (
            {"ownerKind": "render_bundle_v1", "ownerHash": _digest("7")},
            {"bundleHash": _digest("7"), "sceneCount": 3, "scenes": []},
        )

    monkeypatch.setattr(service, "_build_and_store", build_and_store)
    input_record = copy.deepcopy(record)

    prepared = service.prepare_audit_trace(
        input_record,
        decision_ordinal=1,
        observer_seat="south",
    )
    from baloot_platform.service.api import ResearchAuditTraceRenderPrepareResponse

    ResearchAuditTraceRenderPrepareResponse.model_validate(prepared)
    input_record["failureCategories"].append("late_input_mutation")
    input_record["source"]["runCode"] = "MUTATED"

    assert captured["evidence"] is observer
    assert locator.calls[-1]["observer_seat"] is Seat.SOUTH
    assert locator.calls[-1]["prompt_contract"] == "predict_opponent_play"
    assert prepared["observerSeat"] == "south"
    assert prepared["laffaCode"] == record["auditTraceCode"]
    assert prepared["qualificationStatus"] == "rejected"
    assert prepared["auditOnly"] is True
    assert prepared["publicationEligible"] is False
    assert prepared["renderPurpose"] == "research_audit"
    assert "decision_quality_gate_failed" in prepared["qualificationFailures"]
    assert "audit_trace_publication_not_authorized" in prepared["publicationBlockers"]
    assert prepared["source"] == captured["source_metadata"]
    assert prepared["source"]["sourceSchemaVersion"] == 2
    assert prepared["source"]["exactStateId"] == "D1"
    assert prepared["source"]["qrActionable"] is False
    assert {
        "resultHash",
        "analysisHash",
        "auditHash",
        "analysisProfileHash",
        "analyzerId",
        "analyzerVersion",
    }.isdisjoint(prepared["source"])
    assert prepared["source"]["kind"] == "laffa"
    assert prepared["source"]["originKind"] == "research_audit_trace"
    assert prepared["source"]["code"] == record["auditTraceCode"]
    assert prepared["source"]["recordHash"] == record["recordHash"]
    assert prepared["source"]["auditOnly"] is True
    assert prepared["source"]["publicationEligible"] is False
    assert prepared["source"]["qualificationFailures"] == prepared[
        "qualificationFailures"
    ]
    assert prepared["source"]["observerSeat"] == "south"
    assert "late_input_mutation" not in prepared["qualificationFailures"]
    assert prepared["source"]["recordHash"] == record["recordHash"]
    assert prepared["source"]["attemptStatus"] == "completed"
    serialized = json.dumps(prepared, sort_keys=True)
    assert "SECRET-HIDDEN-CARD" not in serialized
    assert "privateHands" not in serialized
    assert "completeTrace" not in serialized
