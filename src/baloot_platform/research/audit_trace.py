"""Durable audit-only retention for replay-valid Research traces."""
from __future__ import annotations

import copy
import hashlib
import json
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

from baloot_platform.challenge.perspective_pipeline import build_observer_projection
from baloot_platform.challenge.replay import TraceDecisionProjectionStoreV1
from baloot_platform.challenge.trace import CompleteGameTraceV1
from baloot_platform.core.enums import Seat


COMPLETE_TRACE_MEDIA_TYPE = "application/vnd.balootx.complete-trace-v1+json"
_DISPOSITIONS = frozenset({"provisional", "qualified", "rejected", "quarantined"})
_STATUSES = frozenset({"provisional", "finalized"})
_ATTEMPT_STATUSES = frozenset(
    {"running", "completed", "retry_scheduled", "failed", "reclaimed", "cancelled"}
)
_COMMON_SOURCE_KEYS = frozenset(
    {
        "kind",
        "runCode",
        "runHash",
        "recipeCode",
        "recipeHash",
        "shardId",
        "shardHash",
        "shardIndex",
        "seed",
        "candidateHash",
        "candidateOrdinal",
        "generationLane",
        "attemptCount",
        "fencingToken",
        "workerId",
    }
)
_SOURCE_EXTRA_KEYS = {
    "research_candidate": frozenset(),
    "organic_foundry_trace": frozenset({"traceRole"}),
    "organic_finalist_trace": frozenset(
        {
            "traceRole",
            "parentRecordHash",
            "parentTraceHash",
            "stageProfileHash",
            "stageRunHash",
        }
    ),
}
_ARTIFACT_PUT_LOCK = threading.RLock()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _closed(
    value: Mapping[str, Any], expected: set[str], label: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    raw = copy.deepcopy(dict(value))
    unknown = set(raw) - expected
    missing = expected - set(raw)
    if unknown or missing:
        raise ValueError(
            f"{label} keys are invalid; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    return raw


def _hash(value: Any, label: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _text(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} cannot be empty")
    return text


def _timestamp(value: Any, label: str) -> str | None:
    if value is None:
        return None
    text = _text(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return text


def _artifact(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = _closed(
        value,
        {"uri", "sha256", "bytes", "mediaType", "name"},
        "Research audit complete-trace reference",
    )
    digest = _hash(raw["sha256"], "complete-trace artifact sha256")
    uri = _text(raw["uri"], "complete-trace artifact URI")
    parsed = urlparse(uri)
    local = (
        parsed.scheme == "cas"
        and parsed.netloc == "sha256"
        and parsed.path == f"/{digest}"
    )
    remote = (
        parsed.scheme == "s3"
        and bool(parsed.netloc)
        and parsed.path not in {"", "/"}
        and parsed.path.rsplit("/", 1)[-1] == digest
    )
    if parsed.params or parsed.query or parsed.fragment or not (local or remote):
        raise ValueError("complete-trace artifact URI does not match its hash")
    byte_count = int(raw["bytes"])
    if byte_count < 1:
        raise ValueError("complete-trace artifact must contain bytes")
    if str(raw["mediaType"]) != COMPLETE_TRACE_MEDIA_TYPE:
        raise ValueError("complete-trace artifact media type is invalid")
    name = raw["name"]
    if name is not None and not str(name).strip():
        raise ValueError("complete-trace artifact name is invalid")
    return {
        "uri": uri,
        "sha256": digest,
        "bytes": byte_count,
        "mediaType": COMPLETE_TRACE_MEDIA_TYPE,
        "name": None if name is None else str(name).strip(),
    }


def _source(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Research audit trace source must be an object")
    kind = str(value.get("kind") or "")
    extra_keys = _SOURCE_EXTRA_KEYS.get(kind)
    if extra_keys is None:
        raise ValueError("Research audit trace source kind is invalid")
    raw = _closed(
        value,
        set(_COMMON_SOURCE_KEYS | extra_keys),
        "Research audit trace source",
    )
    shard_index = int(raw["shardIndex"])
    seed = int(raw["seed"])
    candidate_ordinal = int(raw["candidateOrdinal"])
    attempt_count = int(raw["attemptCount"])
    fencing_token = int(raw["fencingToken"])
    if (
        shard_index < 0
        or seed < 0
        or candidate_ordinal < 0
        or attempt_count < 1
        or fencing_token < 1
    ):
        raise ValueError("Research audit trace source ordinals must be nonnegative")
    normalized = {
        "kind": kind,
        "runCode": _text(raw["runCode"], "Research run code"),
        "runHash": _hash(raw["runHash"], "Research run hash"),
        "recipeCode": _text(raw["recipeCode"], "Research recipe code"),
        "recipeHash": _hash(raw["recipeHash"], "Research recipe hash"),
        "shardId": _text(raw["shardId"], "Research shard ID"),
        "shardHash": _hash(raw["shardHash"], "Research shard hash"),
        "shardIndex": shard_index,
        "seed": seed,
        "candidateHash": _hash(raw["candidateHash"], "Research candidate hash"),
        "candidateOrdinal": candidate_ordinal,
        "generationLane": _text(raw["generationLane"], "Research generation lane"),
        "attemptCount": attempt_count,
        "fencingToken": fencing_token,
        "workerId": _text(raw["workerId"], "Research worker ID"),
    }
    if kind == "organic_foundry_trace":
        trace_role = str(raw["traceRole"])
        if trace_role not in {"scored_final", "all_pass_attempt"}:
            raise ValueError("Organic Foundry audit trace role is invalid")
        normalized["traceRole"] = trace_role
    elif kind == "organic_finalist_trace":
        if raw["traceRole"] != "finalist_descendant":
            raise ValueError("Organic finalist audit trace role is invalid")
        normalized.update(
            {
                "traceRole": "finalist_descendant",
                "parentRecordHash": _hash(
                    raw["parentRecordHash"], "Organic finalist parent record hash"
                ),
                "parentTraceHash": _hash(
                    raw["parentTraceHash"], "Organic finalist parent trace hash"
                ),
                "stageProfileHash": _hash(
                    raw["stageProfileHash"], "Organic finalist stage profile hash"
                ),
                "stageRunHash": _hash(
                    raw["stageRunHash"], "Organic finalist stage run hash"
                ),
            }
        )
    return normalized


def _record_hash(trace_hash: str, source: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        _canonical_bytes(
            {
                "schemaVersion": 1,
                "traceHash": trace_hash,
                "source": copy.deepcopy(dict(source)),
            }
        )
    ).hexdigest()


def _validation_receipt(
    value: Mapping[str, Any], *, observer: bool, trace_hash: str
) -> dict[str, Any]:
    if observer:
        expected = {
            "schemaVersion",
            "status",
            "validator",
            "traceHash",
            "projectedDecisionCount",
            "trustedTraceExposure",
        }
        raw = _closed(value, expected, "Research observer-safety receipt")
        if raw["validator"] != "isolated_observer_projection_v1":
            raise ValueError("Research observer-safety validator is invalid")
        if raw["trustedTraceExposure"] != "internal_only":
            raise ValueError("Research trusted trace must remain internal-only")
        count_key = "projectedDecisionCount"
    else:
        expected = {
            "schemaVersion",
            "status",
            "validator",
            "traceHash",
            "gameHash",
            "finalStateHash",
            "eventChainHash",
            "decisionCount",
        }
        raw = _closed(value, expected, "Research replay-validation receipt")
        if raw["validator"] != "authoritative_complete_trace_replay_v1":
            raise ValueError("Research replay validator is invalid")
        _hash(raw["gameHash"], "replay game hash")
        _hash(raw["finalStateHash"], "replay final-state hash")
        _hash(raw["eventChainHash"], "replay event-chain hash")
        count_key = "decisionCount"
    if int(raw["schemaVersion"]) != 1 or raw["status"] != "verified":
        raise ValueError("Research trace validation receipt is not verified")
    if _hash(raw["traceHash"], "validation trace hash") != trace_hash:
        raise ValueError("Research trace validation receipt identity mismatch")
    count = int(raw[count_key])
    if count < 0:
        raise ValueError("Research trace validation decision count is invalid")
    normalized = copy.deepcopy(raw)
    normalized["schemaVersion"] = 1
    normalized[count_key] = count
    return normalized


def _configuration(value: Mapping[str, Any], trace_hash: str) -> dict[str, Any]:
    raw = _closed(
        value,
        {
            "schemaVersion",
            "traceHash",
            "bundleHash",
            "rulesHash",
            "engineVersion",
            "configurationSnapshotHash",
            "researchConfiguration",
        },
        "Research audit configuration provenance",
    )
    if int(raw["schemaVersion"]) != 1:
        raise ValueError("unsupported Research audit configuration provenance version")
    if _hash(raw["traceHash"], "configuration trace hash") != trace_hash:
        raise ValueError("Research audit configuration trace identity mismatch")
    research_configuration = raw["researchConfiguration"]
    if research_configuration is not None and not isinstance(
        research_configuration, Mapping
    ):
        raise ValueError("Research configuration provenance must be an object")
    return {
        "schemaVersion": 1,
        "traceHash": trace_hash,
        "bundleHash": _hash(raw["bundleHash"], "configuration bundle hash"),
        "rulesHash": _hash(raw["rulesHash"], "configuration rules hash"),
        "engineVersion": _text(raw["engineVersion"], "configuration engine version"),
        "configurationSnapshotHash": _hash(
            raw["configurationSnapshotHash"], "configuration snapshot hash"
        ),
        "researchConfiguration": (
            None
            if research_configuration is None
            else copy.deepcopy(dict(research_configuration))
        ),
    }


def _disagreement(value: Mapping[str, Any]) -> dict[str, str]:
    raw = _closed(
        value,
        {"disagreementId", "sha256", "category", "status"},
        "Research audit disagreement reference",
    )
    return {
        "disagreementId": _text(raw["disagreementId"], "disagreement ID"),
        "sha256": _hash(raw["sha256"], "disagreement hash"),
        "category": _text(raw["category"], "disagreement category"),
        "status": _text(raw["status"], "disagreement status"),
    }


@dataclass(frozen=True, slots=True)
class ResearchAuditTraceRecordV1:
    record_hash: str
    audit_trace_code: str
    trace_hash: str
    complete_trace: Mapping[str, Any]
    source: Mapping[str, Any]
    disposition: str
    status: str
    failure_categories: tuple[str, ...]
    disagreement_references: tuple[Mapping[str, str], ...]
    replay_validation: Mapping[str, Any]
    observer_safety: Mapping[str, Any]
    configuration_provenance: Mapping[str, Any]
    attempt_status: str
    created_at: str | None = None
    updated_at: str | None = None

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "ResearchAuditTraceRecordV1":
        raw = _closed(
            value,
            {
                "schemaVersion",
                "auditTraceCode",
                "recordHash",
                "traceHash",
                "completeTrace",
                "source",
                "disposition",
                "status",
                "failureCategories",
                "disagreementReferences",
                "replayValidation",
                "observerSafety",
                "configurationProvenance",
                "attemptStatus",
                "auditOnly",
                "publicationEligible",
                "createdAt",
                "updatedAt",
            },
            "Research audit trace record",
        )
        if int(raw["schemaVersion"]) != 1:
            raise ValueError("unsupported Research audit trace record version")
        trace_hash = _hash(raw["traceHash"], "Research audit trace hash")
        source = _source(raw["source"])
        record_hash = _record_hash(trace_hash, source)
        if _hash(raw["recordHash"], "Research audit record hash") != record_hash:
            raise ValueError("Research audit record hash does not match its source")
        expected_code = f"LAF-AUD-{record_hash[:10].upper()}"
        if raw["auditTraceCode"] != expected_code:
            raise ValueError("Research audit trace code does not match its trace hash")
        disposition = str(raw["disposition"])
        status = str(raw["status"])
        if disposition not in _DISPOSITIONS or status not in _STATUSES:
            raise ValueError("Research audit trace lifecycle is invalid")
        if (disposition == "provisional") != (status == "provisional"):
            raise ValueError("Research audit trace disposition/status mismatch")
        if not isinstance(raw["failureCategories"], (list, tuple)):
            raise ValueError("Research audit trace failure categories must be an array")
        failures = tuple(
            sorted({_text(item, "failure category") for item in raw["failureCategories"]})
        )
        if disposition in {"rejected", "quarantined"} and not failures:
            raise ValueError("rejected or quarantined audit traces require failures")
        if not isinstance(raw["disagreementReferences"], (list, tuple)):
            raise ValueError("Research audit disagreement references must be an array")
        disagreements = tuple(
            sorted(
                (_disagreement(item) for item in raw["disagreementReferences"]),
                key=lambda item: (item["sha256"], item["disagreementId"]),
            )
        )
        if len(
            {(item["sha256"], item["disagreementId"]) for item in disagreements}
        ) != len(disagreements):
            raise ValueError("Research audit disagreement references must be unique")
        if raw["auditOnly"] is not True or raw["publicationEligible"] is not False:
            raise ValueError("Research audit traces must remain non-publishable audit media")
        attempt_status = str(raw["attemptStatus"])
        if attempt_status not in _ATTEMPT_STATUSES:
            raise ValueError("Research audit trace attempt status is invalid")
        return cls(
            record_hash,
            expected_code,
            trace_hash,
            _artifact(raw["completeTrace"]),
            source,
            disposition,
            status,
            failures,
            disagreements,
            _validation_receipt(
                raw["replayValidation"], observer=False, trace_hash=trace_hash
            ),
            _validation_receipt(
                raw["observerSafety"], observer=True, trace_hash=trace_hash
            ),
            _configuration(raw["configurationProvenance"], trace_hash),
            attempt_status,
            _timestamp(raw["createdAt"], "Research audit trace createdAt"),
            _timestamp(raw["updatedAt"], "Research audit trace updatedAt"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "recordHash": self.record_hash,
            "auditTraceCode": self.audit_trace_code,
            "traceHash": self.trace_hash,
            "completeTrace": copy.deepcopy(dict(self.complete_trace)),
            "source": copy.deepcopy(dict(self.source)),
            "disposition": self.disposition,
            "status": self.status,
            "failureCategories": list(self.failure_categories),
            "disagreementReferences": [
                copy.deepcopy(dict(item)) for item in self.disagreement_references
            ],
            "replayValidation": copy.deepcopy(dict(self.replay_validation)),
            "observerSafety": copy.deepcopy(dict(self.observer_safety)),
            "configurationProvenance": copy.deepcopy(
                dict(self.configuration_provenance)
            ),
            "attemptStatus": self.attempt_status,
            "auditOnly": True,
            "publicationEligible": False,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }

    def finalized(
        self,
        *,
        disposition: str,
        failure_categories: Iterable[str] = (),
        disagreement_references: Iterable[Mapping[str, Any]] = (),
    ) -> "ResearchAuditTraceRecordV1":
        return self.build(
            {
                **self.to_dict(),
                "disposition": str(disposition),
                "status": "finalized",
                "failureCategories": list(failure_categories),
                "disagreementReferences": [
                    copy.deepcopy(dict(item)) for item in disagreement_references
                ],
                "updatedAt": None,
            }
        )

    def with_attempt_status(
        self, attempt_status: str, *, failure_category: str | None = None
    ) -> "ResearchAuditTraceRecordV1":
        failures = list(self.failure_categories)
        if failure_category is not None:
            failures.append(str(failure_category))
        return self.build(
            {
                **self.to_dict(),
                "attemptStatus": str(attempt_status),
                "failureCategories": failures,
                "updatedAt": None,
            }
        )


class ResearchAuditTraceRepositoryV1:
    """Checksum-verified CAS and durable index for audit-only Research traces."""

    def __init__(self, *, store: Any, artifact_store: Any) -> None:
        self.store = store
        self.artifact_store = artifact_store

    def retain(
        self,
        trace: CompleteGameTraceV1,
        *,
        rules: Any,
        source: Mapping[str, Any],
        research_configuration: Mapping[str, Any] | None,
    ) -> ResearchAuditTraceRecordV1:
        payload = trace.to_dict(trusted=True)
        round_tripped = CompleteGameTraceV1.from_dict(payload)
        if round_tripped.sha256 != trace.sha256:
            raise ValueError("Research complete-trace round trip changed its identity")
        projection_store = TraceDecisionProjectionStoreV1(round_tripped, rules)
        for decision in round_tripped.decision_ledger.records:
            for observer_seat in Seat:
                build_observer_projection(
                    projection_store, decision.decision_id, observer_seat
                )
        canonical_payload = _canonical_bytes(round_tripped.to_dict(trusted=True))
        with _ARTIFACT_PUT_LOCK:
            reference = self.artifact_store.put(
                canonical_payload,
                media_type=COMPLETE_TRACE_MEDIA_TYPE,
                name=f"{round_tripped.sha256}.complete-trace.json",
            )
        if reference.get("sha256") != hashlib.sha256(canonical_payload).hexdigest():
            raise ValueError("Research complete-trace artifact digest mismatch")
        generation = dict(round_tripped.trusted_final_state.get("generation") or {})
        snapshot_hash = str(generation.get("configurationSnapshotHash") or "")
        replay = projection_store.replay
        event_chain_hash = (
            replay.event_chain_hash
            if replay.event_chain_hash
            else hashlib.sha256(b"[]").hexdigest()
        )
        normalized_source = _source(source)
        record_hash = _record_hash(round_tripped.sha256, normalized_source)
        record = ResearchAuditTraceRecordV1.build(
            {
                "schemaVersion": 1,
                "recordHash": record_hash,
                "auditTraceCode": f"LAF-AUD-{record_hash[:10].upper()}",
                "traceHash": round_tripped.sha256,
                "completeTrace": reference,
                "source": normalized_source,
                "disposition": "provisional",
                "status": "provisional",
                "failureCategories": [
                    "content_qualification_not_established",
                    "research_analysis_not_completed",
                ],
                "disagreementReferences": [],
                "replayValidation": {
                    "schemaVersion": 1,
                    "status": "verified",
                    "validator": "authoritative_complete_trace_replay_v1",
                    "traceHash": round_tripped.sha256,
                    "gameHash": round_tripped.game_hash,
                    "finalStateHash": replay.final_state_hash,
                    "eventChainHash": event_chain_hash,
                    "decisionCount": len(replay.decisions),
                },
                "observerSafety": {
                    "schemaVersion": 1,
                    "status": "verified",
                    "validator": "isolated_observer_projection_v1",
                    "traceHash": round_tripped.sha256,
                    "projectedDecisionCount": len(replay.decisions),
                    "trustedTraceExposure": "internal_only",
                },
                "configurationProvenance": {
                    "schemaVersion": 1,
                    "traceHash": round_tripped.sha256,
                    "bundleHash": round_tripped.bundle_hash,
                    "rulesHash": round_tripped.rules_hash,
                    "engineVersion": round_tripped.engine_version,
                    "configurationSnapshotHash": snapshot_hash,
                    "researchConfiguration": (
                        None
                        if research_configuration is None
                        else copy.deepcopy(dict(research_configuration))
                    ),
                },
                "attemptStatus": "running",
                "auditOnly": True,
                "publicationEligible": False,
                "createdAt": None,
                "updatedAt": None,
            }
        )
        stored = self.store.put_audit_trace(
            record.to_dict(),
            shard_id=str(record.source["shardId"]),
            worker_id=str(record.source["workerId"]),
            fencing_token=int(record.source["fencingToken"]),
        )
        return ResearchAuditTraceRecordV1.build(stored)

    def finalize(
        self,
        identifier: str,
        *,
        disposition: str,
        failure_categories: Iterable[str] = (),
        disagreement_references: Iterable[Mapping[str, Any]] = (),
    ) -> ResearchAuditTraceRecordV1:
        current = self.store.get_audit_trace(str(identifier), include_unadmitted=True)
        if current is None:
            raise KeyError(identifier)
        record = ResearchAuditTraceRecordV1.build(current).finalized(
            disposition=disposition,
            failure_categories=failure_categories,
            disagreement_references=disagreement_references,
        )
        return ResearchAuditTraceRecordV1.build(
            self.store.update_audit_trace(
                record.to_dict(),
                worker_id=str(record.source["workerId"]),
                fencing_token=int(record.source["fencingToken"]),
            )
        )

    def get_record(
        self, identifier: str, *, include_unadmitted: bool = False
    ) -> ResearchAuditTraceRecordV1 | None:
        value = self.store.get_audit_trace(
            identifier, include_unadmitted=include_unadmitted
        )
        return None if value is None else ResearchAuditTraceRecordV1.build(value)

    def list_records(
        self,
        *,
        run_code: str | None = None,
        disposition: str | None = None,
        status: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        page = self.store.list_audit_traces(
            run_code=run_code,
            disposition=disposition,
            status=status,
            offset=offset,
            limit=limit,
        )
        return {
            **page,
            "items": [
                ResearchAuditTraceRecordV1.build(item).to_dict()
                for item in page["items"]
            ],
        }

    def find_record(
        self, trace_hash: str, *, run_code: str
    ) -> ResearchAuditTraceRecordV1 | None:
        expected_trace_hash = _hash(trace_hash, "Research audit trace hash")
        expected_run_code = _text(run_code, "Research run code")
        matches: list[ResearchAuditTraceRecordV1] = []
        offset = 0
        while True:
            page = self.list_records(
                run_code=expected_run_code,
                offset=offset,
                limit=500,
            )
            for item in page["items"]:
                record = ResearchAuditTraceRecordV1.build(item)
                if record.trace_hash == expected_trace_hash:
                    matches.append(record)
            next_offset = page.get("nextOffset")
            if next_offset is None:
                break
            next_value = int(next_offset)
            if next_value <= offset:
                raise ValueError("Research audit repository pagination did not advance")
            offset = next_value
        matches.sort(key=lambda item: item.record_hash)
        matches.sort(
            key=lambda item: (
                float("-inf")
                if item.created_at is None
                else datetime.fromisoformat(
                    item.created_at.replace("Z", "+00:00")
                ).timestamp()
            ),
            reverse=True,
        )
        return matches[0] if matches else None

    def load_trace(self, identifier: str) -> CompleteGameTraceV1:
        record = self.get_record(identifier)
        if record is None:
            raise KeyError(identifier)
        reference = record.complete_trace
        data = self.artifact_store.get(str(reference["uri"]))
        if len(data) != int(reference["bytes"]):
            raise ValueError("Research audit complete-trace byte count mismatch")
        if hashlib.sha256(data).hexdigest() != str(reference["sha256"]):
            raise ValueError("Research audit complete-trace checksum mismatch")
        try:
            decoded = data.decode("utf-8")
            value = json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Research audit complete-trace is not valid JSON") from exc
        if not isinstance(value, Mapping):
            raise ValueError("Research audit complete-trace must contain one object")
        if _canonical_bytes(value) != data:
            raise ValueError("Research audit complete-trace is not canonical JSON")
        trace = CompleteGameTraceV1.from_dict(value)
        if trace.sha256 != record.trace_hash:
            raise ValueError("Research audit complete-trace internal identity mismatch")
        return trace


class CompositeResearchAuditTraceRepositoryV1:
    """One read view over the primary and additional audit-trace repositories."""

    def __init__(
        self,
        primary: ResearchAuditTraceRepositoryV1,
        *,
        secondary_repositories: Iterable[ResearchAuditTraceRepositoryV1] = (),
    ) -> None:
        self.primary = primary
        self.secondary_repositories = tuple(secondary_repositories)
        self._repositories = (self.primary, *self.secondary_repositories)

    def retain(
        self,
        trace: CompleteGameTraceV1,
        *,
        rules: Any,
        source: Mapping[str, Any],
        research_configuration: Mapping[str, Any] | None,
    ) -> ResearchAuditTraceRecordV1:
        return self.primary.retain(
            trace,
            rules=rules,
            source=source,
            research_configuration=research_configuration,
        )

    def finalize(
        self,
        identifier: str,
        *,
        disposition: str,
        failure_categories: Iterable[str] = (),
        disagreement_references: Iterable[Mapping[str, Any]] = (),
    ) -> ResearchAuditTraceRecordV1:
        return self.primary.finalize(
            identifier,
            disposition=disposition,
            failure_categories=failure_categories,
            disagreement_references=disagreement_references,
        )

    def _resolve_record(
        self, identifier: str, *, include_unadmitted: bool = False
    ) -> tuple[ResearchAuditTraceRepositoryV1, ResearchAuditTraceRecordV1] | None:
        for repository in self._repositories:
            record = repository.get_record(
                str(identifier), include_unadmitted=include_unadmitted
            )
            if record is not None:
                return repository, record
        return None

    def get_record(
        self, identifier: str, *, include_unadmitted: bool = False
    ) -> ResearchAuditTraceRecordV1 | None:
        resolved = self._resolve_record(
            identifier, include_unadmitted=include_unadmitted
        )
        return None if resolved is None else resolved[1]

    @staticmethod
    def _page_records(
        repository: ResearchAuditTraceRepositoryV1,
        *,
        run_code: str | None,
        disposition: str | None,
        status: str | None,
    ) -> list[ResearchAuditTraceRecordV1]:
        records: list[ResearchAuditTraceRecordV1] = []
        offset = 0
        while True:
            page = repository.list_records(
                run_code=run_code,
                disposition=disposition,
                status=status,
                offset=offset,
                limit=500,
            )
            records.extend(
                ResearchAuditTraceRecordV1.build(item) for item in page["items"]
            )
            next_offset = page.get("nextOffset")
            if next_offset is None:
                return records
            next_value = int(next_offset)
            if next_value <= offset:
                raise ValueError("Research audit repository pagination did not advance")
            offset = next_value

    def _merged_records(
        self,
        *,
        run_code: str | None = None,
        disposition: str | None = None,
        status: str | None = None,
    ) -> list[ResearchAuditTraceRecordV1]:
        by_record_hash: dict[str, ResearchAuditTraceRecordV1] = {}
        for repository in self._repositories:
            for record in self._page_records(
                repository,
                run_code=run_code,
                disposition=disposition,
                status=status,
            ):
                if run_code is not None and record.source["runCode"] != run_code:
                    continue
                if disposition is not None and record.disposition != disposition:
                    continue
                if status is not None and record.status != status:
                    continue
                by_record_hash.setdefault(record.record_hash, record)
        records = list(by_record_hash.values())
        records.sort(key=lambda item: (item.trace_hash, item.record_hash))
        records.sort(
            key=lambda item: (
                float("-inf")
                if item.created_at is None
                else datetime.fromisoformat(
                    item.created_at.replace("Z", "+00:00")
                ).timestamp()
            ),
            reverse=True,
        )
        return records

    def list_records(
        self,
        *,
        run_code: str | None = None,
        disposition: str | None = None,
        status: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        start = max(0, int(offset))
        size = max(1, min(500, int(limit)))
        records = self._merged_records(
            run_code=run_code,
            disposition=disposition,
            status=status,
        )
        items = records[start : start + size]
        next_offset = start + len(items)
        return {
            "items": [item.to_dict() for item in items],
            "total": len(records),
            "offset": start,
            "limit": size,
            "nextOffset": next_offset if next_offset < len(records) else None,
        }

    def find_record(
        self, trace_hash: str, *, run_code: str
    ) -> ResearchAuditTraceRecordV1 | None:
        expected_trace_hash = _hash(trace_hash, "Research audit trace hash")
        records = self._merged_records(run_code=_text(run_code, "Research run code"))
        return next(
            (record for record in records if record.trace_hash == expected_trace_hash),
            None,
        )

    def find_record_by_trace_hash_and_run_code(
        self, trace_hash: str, *, run_code: str
    ) -> ResearchAuditTraceRecordV1 | None:
        return self.find_record(trace_hash, run_code=run_code)

    def load_trace(self, identifier: str) -> CompleteGameTraceV1:
        resolved = self._resolve_record(identifier)
        if resolved is None:
            raise KeyError(identifier)
        repository, record = resolved
        return repository.load_trace(record.record_hash)


__all__ = [
    "CompositeResearchAuditTraceRepositoryV1",
    "COMPLETE_TRACE_MEDIA_TYPE",
    "ResearchAuditTraceRecordV1",
    "ResearchAuditTraceRepositoryV1",
]
