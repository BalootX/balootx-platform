"""Governed historical-game and expert-proposal trace ingestion."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from baloot_platform.core.rules import RuleBook
from baloot_platform.plugins.base import PluginDescriptor

from .campaign import DISCOVERY_LANES
from .evidence_v3 import canonical_sha256
from .replay import CompleteTraceReplayerV1
from .trace import CompleteGameTraceV1


def _sha256(value: Any, field: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(item not in "0123456789abcdef" for item in digest):
        raise ValueError(f"{field} must be a SHA-256 digest")
    return digest


@dataclass(frozen=True, slots=True)
class TraceImportPolicyV1:
    maximum_entries: int
    allowed_rights_statuses: tuple[str, ...]
    historical_verification_statuses: tuple[str, ...]
    expert_verification_statuses: tuple[str, ...]
    require_complete_hidden_state: bool
    sha256: str

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "TraceImportPolicyV1":
        allowed = {
            "schemaVersion", "maximumEntries", "allowedRightsStatuses",
            "historicalVerificationStatuses", "expertVerificationStatuses",
            "requireCompleteHiddenState",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown trace import policy keys: {sorted(unknown)}")
        if int(value.get("schemaVersion", 0)) != 1:
            raise ValueError("unsupported trace import policy schema version")
        maximum = int(value.get("maximumEntries", 0))
        rights = tuple(sorted({str(item) for item in value.get("allowedRightsStatuses") or ()}))
        historical = tuple(sorted({
            str(item) for item in value.get("historicalVerificationStatuses") or ()
        }))
        expert = tuple(sorted({
            str(item) for item in value.get("expertVerificationStatuses") or ()
        }))
        if not 1 <= maximum <= 1_000_000:
            raise ValueError("trace import maximum entries must be in [1, 1000000]")
        if not rights or set(rights) - {"cleared", "research_only"}:
            raise ValueError("trace import rights statuses are unsupported")
        if not historical or set(historical) - {"verified_action_order"}:
            raise ValueError("historical trace verification statuses are unsupported")
        if not expert or set(expert) - {
            "unreviewed", "verified_expert", "consensus_approved",
        }:
            raise ValueError("expert trace verification statuses are unsupported")
        core = {
            "schemaVersion": 1,
            "maximumEntries": maximum,
            "allowedRightsStatuses": list(rights),
            "historicalVerificationStatuses": list(historical),
            "expertVerificationStatuses": list(expert),
            "requireCompleteHiddenState": bool(value.get("requireCompleteHiddenState", True)),
        }
        return cls(
            maximum, rights, historical, expert,
            core["requireCompleteHiddenState"], canonical_sha256(core),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "maximumEntries": self.maximum_entries,
            "allowedRightsStatuses": list(self.allowed_rights_statuses),
            "historicalVerificationStatuses": list(self.historical_verification_statuses),
            "expertVerificationStatuses": list(self.expert_verification_statuses),
            "requireCompleteHiddenState": self.require_complete_hidden_state,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class TraceImportProvenanceV1:
    lane: str
    source_record_id: str
    source_evidence_hash: str
    rights_status: str
    verification_status: str
    complete_hidden_state_attested: bool
    submitted_by_hash: str | None
    sha256: str

    @classmethod
    def build(
        cls,
        value: Mapping[str, Any],
        *,
        policy: TraceImportPolicyV1,
    ) -> "TraceImportProvenanceV1":
        allowed = {
            "schemaVersion", "lane", "sourceRecordId", "sourceEvidenceHash",
            "rightsStatus", "verificationStatus", "completeHiddenStateAttested",
            "submittedByHash",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown trace import provenance keys: {sorted(unknown)}")
        if int(value.get("schemaVersion", 0)) != 1:
            raise ValueError("unsupported trace import provenance schema version")
        lane = str(value.get("lane") or "")
        if lane not in {"historical", "expert"} or lane not in DISCOVERY_LANES:
            raise ValueError("trace import provenance requires historical or expert lane")
        record_id = str(value.get("sourceRecordId") or "").strip()
        if not record_id or len(record_id) > 500:
            raise ValueError("trace import source record identity is invalid")
        rights = str(value.get("rightsStatus") or "")
        if rights not in policy.allowed_rights_statuses:
            raise ValueError("trace import rights status is not allowed")
        verification = str(value.get("verificationStatus") or "")
        allowed_statuses = (
            policy.historical_verification_statuses
            if lane == "historical" else policy.expert_verification_statuses
        )
        if verification not in allowed_statuses:
            raise ValueError("trace import verification status is not allowed for its lane")
        complete = bool(value.get("completeHiddenStateAttested", False))
        if policy.require_complete_hidden_state and not complete:
            raise ValueError("trace import requires complete hidden-state attestation")
        submitter_raw = value.get("submittedByHash")
        submitter = _sha256(submitter_raw, "submittedByHash") if submitter_raw else None
        if lane == "expert" and submitter is None:
            raise ValueError("expert trace proposal requires a pseudonymous submitter hash")
        core = {
            "schemaVersion": 1,
            "lane": lane,
            "sourceRecordId": record_id,
            "sourceEvidenceHash": _sha256(value.get("sourceEvidenceHash"), "sourceEvidenceHash"),
            "rightsStatus": rights,
            "verificationStatus": verification,
            "completeHiddenStateAttested": complete,
            "submittedByHash": submitter,
        }
        return cls(
            lane, record_id, core["sourceEvidenceHash"], rights, verification,
            complete, submitter, canonical_sha256(core),
        )

    @property
    def review_state(self) -> str:
        if self.lane == "historical":
            return "source_verified_review_pending"
        return self.verification_status

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "lane": self.lane,
            "sourceRecordId": self.source_record_id,
            "sourceEvidenceHash": self.source_evidence_hash,
            "rightsStatus": self.rights_status,
            "verificationStatus": self.verification_status,
            "completeHiddenStateAttested": self.complete_hidden_state_attested,
            "submittedByHash": self.submitted_by_hash,
            "reviewState": self.review_state,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class TraceImportCatalogEntryV1:
    ordinal: int
    trace_reference: Mapping[str, Any]
    provenance: TraceImportProvenanceV1
    sha256: str

    @classmethod
    def build(
        cls,
        *,
        ordinal: int,
        trace_reference: Mapping[str, Any],
        provenance: TraceImportProvenanceV1,
    ) -> "TraceImportCatalogEntryV1":
        number = int(ordinal)
        if number < 0:
            raise ValueError("trace import catalog ordinal cannot be negative")
        reference = dict(trace_reference)
        if reference.get("ownerKind") != "complete_trace_v1":
            raise ValueError("trace import catalog requires complete trace artifacts")
        if reference.get("mediaType") != "application/vnd.balootx.complete-trace-v1+json":
            raise ValueError("trace import catalog trace media type is invalid")
        _sha256(reference.get("ownerHash"), "trace ownerHash")
        _sha256(reference.get("sha256"), "trace artifact sha256")
        if not str(reference.get("uri") or "") or int(reference.get("bytes", -1)) < 1:
            raise ValueError("trace import catalog reference is incomplete")
        core = {
            "schemaVersion": 1,
            "ordinal": number,
            "traceReference": reference,
            "provenanceHash": provenance.sha256,
        }
        return cls(number, reference, provenance, canonical_sha256(core))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "ordinal": self.ordinal,
            "traceReference": dict(self.trace_reference),
            "provenance": self.provenance.to_dict(),
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class TraceImportCatalogV1:
    lane: str
    policy_hash: str
    entries: tuple[TraceImportCatalogEntryV1, ...]
    sha256: str

    @classmethod
    def build(
        cls,
        lane: str,
        entries: Sequence[TraceImportCatalogEntryV1],
        *,
        policy: TraceImportPolicyV1,
    ) -> "TraceImportCatalogV1":
        selected_lane = str(lane)
        if selected_lane not in {"historical", "expert"}:
            raise ValueError("trace import catalog lane is unsupported")
        values = tuple(sorted(entries, key=lambda item: item.ordinal))
        if not values or len(values) > policy.maximum_entries:
            raise ValueError("trace import catalog size is outside policy")
        if tuple(item.ordinal for item in values) != tuple(range(len(values))):
            raise ValueError("trace import catalog ordinals must be contiguous from zero")
        if any(item.provenance.lane != selected_lane for item in values):
            raise ValueError("trace import catalog contains another lane's provenance")
        if len({item.sha256 for item in values}) != len(values):
            raise ValueError("trace import catalog contains duplicate entries")
        core = {
            "schemaVersion": 1,
            "lane": selected_lane,
            "policyHash": policy.sha256,
            "entryHashes": [item.sha256 for item in values],
        }
        return cls(selected_lane, policy.sha256, values, canonical_sha256(core))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "lane": self.lane,
            "policyHash": self.policy_hash,
            "entries": [item.to_dict() for item in self.entries],
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class TraceImportResultV1:
    catalog_hash: str
    catalog_ordinal: int
    lane: str
    trace_hash: str
    provenance_hash: str
    replay_final_state_hash: str
    replay_event_chain_hash: str
    review_state: str
    coverage_cells: tuple[str, ...]
    sha256: str
    trace: CompleteGameTraceV1
    trace_reference: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "catalogHash": self.catalog_hash,
            "catalogOrdinal": self.catalog_ordinal,
            "lane": self.lane,
            "traceHash": self.trace_hash,
            "provenanceHash": self.provenance_hash,
            "replayFinalStateHash": self.replay_final_state_hash,
            "replayEventChainHash": self.replay_event_chain_hash,
            "reviewState": self.review_state,
            "coverageCells": list(self.coverage_cells),
            "sha256": self.sha256,
        }


class TraceImportIngestorV1:
    descriptor = PluginDescriptor(
        id="governed_trace_import_v1",
        kind="generator",
        version="1.0.0",
        description="Replay-verified historical and expert complete-trace ingestion",
        capabilities=(
            "historical", "expert", "content_addressed_input",
            "authoritative_replay", "review_pending",
        ),
        resource_class="io",
        deterministic=True,
        timeout_seconds=300,
        maximum_memory_mb=1024,
    )

    def __init__(
        self,
        artifact_store: Any,
        rules: RuleBook,
        *,
        bundle_hash: str,
        engine_version: str,
        catalog: TraceImportCatalogV1,
        policy: TraceImportPolicyV1,
    ):
        if catalog.policy_hash != policy.sha256:
            raise ValueError("trace import catalog policy identity mismatch")
        self.artifact_store = artifact_store
        self.rules = rules
        self.bundle_hash = _sha256(bundle_hash, "bundleHash")
        self.engine_version = str(engine_version)
        self.catalog = catalog
        self.policy = policy

    def ingest(self, ordinal: int) -> TraceImportResultV1:
        number = int(ordinal)
        if number < 0 or number >= len(self.catalog.entries):
            raise KeyError(number)
        entry = self.catalog.entries[number]
        reference = dict(entry.trace_reference)
        data = self.artifact_store.get(str(reference["uri"]))
        if len(data) != int(reference["bytes"]):
            raise ValueError("imported trace artifact size mismatch")
        if hashlib.sha256(data).hexdigest() != str(reference["sha256"]):
            raise ValueError("imported trace artifact checksum mismatch")
        document = json.loads(data)
        if not isinstance(document, Mapping):
            raise ValueError("imported trace artifact must contain a JSON object")
        trace = CompleteGameTraceV1.from_dict(document)
        if trace.sha256 != str(reference["ownerHash"]):
            raise ValueError("imported trace owner identity mismatch")
        if (
            trace.rules_hash != self.rules.sha256
            or trace.bundle_hash != self.bundle_hash
            or trace.engine_version != self.engine_version
        ):
            raise ValueError("imported trace runtime identity is incompatible")
        replay = CompleteTraceReplayerV1(self.rules).replay(trace)
        coverage = tuple(sorted({
            f"source:{self.catalog.lane}",
            f"rights:{entry.provenance.rights_status}",
            f"verification:{entry.provenance.verification_status}",
            f"review:{entry.provenance.review_state}",
            "publication:pending_analysis_and_review",
        }))
        core = {
            "schemaVersion": 1,
            "catalogHash": self.catalog.sha256,
            "catalogOrdinal": number,
            "lane": self.catalog.lane,
            "traceHash": trace.sha256,
            "provenanceHash": entry.provenance.sha256,
            "replayFinalStateHash": replay.final_state_hash,
            "replayEventChainHash": replay.event_chain_hash,
            "reviewState": entry.provenance.review_state,
            "coverageCells": list(coverage),
        }
        return TraceImportResultV1(
            self.catalog.sha256,
            number,
            self.catalog.lane,
            trace.sha256,
            entry.provenance.sha256,
            replay.final_state_hash,
            replay.event_chain_hash,
            entry.provenance.review_state,
            coverage,
            canonical_sha256(core),
            trace,
            reference,
        )


class TraceImportCatalogFactoryV1:
    """Registered factory; catalogs remain row-driven immutable runtime data."""

    descriptor = PluginDescriptor(
        id="trace_import_catalog_factory_v1",
        kind="generator_factory",
        version="1.0.0",
        description="Build governed historical/expert import catalogs and ingestors",
        capabilities=("historical", "expert", "row_driven_catalog"),
        resource_class="io",
        deterministic=True,
        timeout_seconds=60,
        maximum_memory_mb=256,
    )

    def __init__(
        self,
        artifact_store: Any,
        rules: RuleBook,
        *,
        bundle_hash: str,
        engine_version: str,
        policy: TraceImportPolicyV1,
    ):
        self.artifact_store = artifact_store
        self.rules = rules
        self.bundle_hash = bundle_hash
        self.engine_version = engine_version
        self.policy = policy

    def build_ingestor(
        self,
        lane: str,
        entries: Sequence[TraceImportCatalogEntryV1],
    ) -> TraceImportIngestorV1:
        catalog = TraceImportCatalogV1.build(lane, entries, policy=self.policy)
        return TraceImportIngestorV1(
            self.artifact_store,
            self.rules,
            bundle_hash=self.bundle_hash,
            engine_version=self.engine_version,
            catalog=catalog,
            policy=self.policy,
        )
