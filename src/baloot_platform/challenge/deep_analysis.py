"""Leased and fenced deep-analysis jobs over completed Foundry campaigns."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Mapping

from baloot_platform.plugins.base import PluginDescriptor

from .campaign import SimulationCampaignSpecV1
from .evidence_v3 import canonical_sha256
from .funnel import FoundryAnalysisFunnelV1


DEEP_ANALYSIS_JOB_KIND = "foundry_deep_analysis_v1"


class NoRetainedCandidateTracesError(ValueError):
    """A completed campaign had no screened trace eligible for deep analysis."""


def _is_sha256(value: Any) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", str(value)))


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class DeepAnalysisProfileV1:
    profile_id: str
    revision: str
    maximum_oracle_decisions_per_trace: int
    maximum_oracle_remaining_cards: int
    maximum_confirmation_decisions_per_trace: int
    minimum_screening_score: float
    allow_pending_surrounding_nominations: bool
    job_priority: int
    sha256: str

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "DeepAnalysisProfileV1":
        allowed = {
            "schemaVersion", "id", "revision",
            "maximumOracleDecisionsPerTrace",
            "maximumOracleRemainingCards",
            "maximumConfirmationDecisionsPerTrace", "minimumScreeningScore",
            "allowPendingSurroundingNominations",
            "jobPriority",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown deep analysis profile keys: {sorted(unknown)}")
        if int(value.get("schemaVersion", 0)) != 1:
            raise ValueError("unsupported deep analysis profile schema")
        core = {
            "schemaVersion": 1,
            "id": str(value.get("id") or ""),
            "revision": str(value.get("revision") or ""),
            "maximumOracleDecisionsPerTrace": int(
                value.get("maximumOracleDecisionsPerTrace", -1)
            ),
            "maximumOracleRemainingCards": int(
                value.get("maximumOracleRemainingCards", -1)
            ),
            "maximumConfirmationDecisionsPerTrace": int(
                value.get("maximumConfirmationDecisionsPerTrace", -1)
            ),
            "minimumScreeningScore": float(value.get("minimumScreeningScore", -1.0)),
            "allowPendingSurroundingNominations": value.get(
                "allowPendingSurroundingNominations", False
            ),
            "jobPriority": int(value.get("jobPriority", -1)),
        }
        if not core["id"] or not core["revision"]:
            raise ValueError("deep analysis profile identity is required")
        if not 0 <= core["maximumOracleDecisionsPerTrace"] <= 128:
            raise ValueError("deep analysis oracle decision bound is out of range")
        if not 1 <= core["maximumOracleRemainingCards"] <= 32:
            raise ValueError("deep analysis oracle endgame window is out of range")
        if not 1 <= core["maximumConfirmationDecisionsPerTrace"] <= 32:
            raise ValueError("deep analysis confirmation decision bound is out of range")
        if not 0.0 <= core["minimumScreeningScore"] <= 100.0:
            raise ValueError("deep analysis screening score is out of range")
        if not isinstance(core["allowPendingSurroundingNominations"], bool):
            raise ValueError("deep analysis pending nomination policy must be boolean")
        if not 0 <= core["jobPriority"] <= 10_000:
            raise ValueError("deep analysis job priority is out of range")
        return cls(
            core["id"],
            core["revision"],
            core["maximumOracleDecisionsPerTrace"],
            core["maximumOracleRemainingCards"],
            core["maximumConfirmationDecisionsPerTrace"],
            core["minimumScreeningScore"],
            core["allowPendingSurroundingNominations"],
            core["jobPriority"],
            canonical_sha256(core),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "id": self.profile_id,
            "revision": self.revision,
            "maximumOracleDecisionsPerTrace": self.maximum_oracle_decisions_per_trace,
            "maximumOracleRemainingCards": self.maximum_oracle_remaining_cards,
            "maximumConfirmationDecisionsPerTrace": (
                self.maximum_confirmation_decisions_per_trace
            ),
            "minimumScreeningScore": self.minimum_screening_score,
            "allowPendingSurroundingNominations": (
                self.allow_pending_surrounding_nominations
            ),
            "jobPriority": self.job_priority,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class DeepAnalysisBatchSpecV1:
    campaign_hash: str
    shard_result_hashes: tuple[str, ...]
    rules_hash: str
    bundle_hash: str
    engine_version: str
    analysis_profile_hash: str
    perspective_profile_hash: str
    selection_profile_hash: str
    diversity_profile_hash: str
    deep_analysis_profile_hash: str
    funnel_id: str
    funnel_version: str
    sha256: str

    @classmethod
    def build(
        cls,
        campaign_store: Any,
        campaign_hash: str,
        *,
        funnel: FoundryAnalysisFunnelV1,
        profile: DeepAnalysisProfileV1,
    ) -> "DeepAnalysisBatchSpecV1":
        campaign = campaign_store.get_campaign(str(campaign_hash))
        if campaign is None:
            raise KeyError(campaign_hash)
        if campaign.get("status") != "completed" or bool(campaign.get("cancelRequested")):
            raise ValueError("deep analysis requires a completed non-cancelled campaign")
        campaign_spec = SimulationCampaignSpecV1.from_dict(dict(campaign["spec"]))
        shards = tuple(campaign_store.list_shards(campaign_spec.sha256))
        if not shards or any(item.get("status") != "completed" for item in shards):
            raise ValueError("deep analysis requires every campaign shard to be completed")
        result_hashes = tuple(sorted(
            str(dict(item.get("result") or {}).get("sha256") or "") for item in shards
        ))
        if any(not _is_sha256(item) for item in result_hashes):
            raise ValueError("deep analysis requires immutable shard result hashes")
        retained_candidates = {
            str(candidate_hash)
            for shard in shards
            for candidate_hash in tuple(
                dict(shard.get("result") or {}).get("candidateHashes") or ()
            )
        }
        if not retained_candidates:
            raise NoRetainedCandidateTracesError(
                "deep analysis requires at least one retained candidate trace"
            )
        if any(not _is_sha256(item) for item in retained_candidates):
            raise ValueError("deep analysis campaign contains an invalid candidate trace")
        if campaign_spec.analysis_profile_hash != funnel.analyzer.profile.sha256:
            raise ValueError("campaign and deep analyzer profile identities differ")
        if campaign_spec.selection_profile_hash != funnel.selection_profile.sha256:
            raise ValueError("campaign and selection profile identities differ")
        core = {
            "schemaVersion": 1,
            "campaignHash": campaign_spec.sha256,
            "shardResultHashes": list(result_hashes),
            "rulesHash": campaign_spec.rules_hash,
            "bundleHash": campaign_spec.bundle_hash,
            "engineVersion": campaign_spec.engine_version,
            "analysisProfileHash": funnel.analyzer.profile.sha256,
            "perspectiveProfileHash": campaign_spec.perspective_profile_hash,
            "selectionProfileHash": funnel.selection_profile.sha256,
            "diversityProfileHash": funnel.diversity_profile.sha256,
            "deepAnalysisProfileHash": profile.sha256,
            "funnelId": funnel.descriptor.id,
            "funnelVersion": funnel.descriptor.version,
        }
        return cls(
            core["campaignHash"],
            result_hashes,
            core["rulesHash"],
            core["bundleHash"],
            core["engineVersion"],
            core["analysisProfileHash"],
            core["perspectiveProfileHash"],
            core["selectionProfileHash"],
            core["diversityProfileHash"],
            core["deepAnalysisProfileHash"],
            core["funnelId"],
            core["funnelVersion"],
            canonical_sha256(core),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DeepAnalysisBatchSpecV1":
        core = {
            "schemaVersion": 1,
            "campaignHash": str(value.get("campaignHash") or ""),
            "shardResultHashes": sorted(
                str(item) for item in value.get("shardResultHashes") or ()
            ),
            "rulesHash": str(value.get("rulesHash") or ""),
            "bundleHash": str(value.get("bundleHash") or ""),
            "engineVersion": str(value.get("engineVersion") or ""),
            "analysisProfileHash": str(value.get("analysisProfileHash") or ""),
            "perspectiveProfileHash": str(value.get("perspectiveProfileHash") or ""),
            "selectionProfileHash": str(value.get("selectionProfileHash") or ""),
            "diversityProfileHash": str(value.get("diversityProfileHash") or ""),
            "deepAnalysisProfileHash": str(value.get("deepAnalysisProfileHash") or ""),
            "funnelId": str(value.get("funnelId") or ""),
            "funnelVersion": str(value.get("funnelVersion") or ""),
        }
        hashes = (
            core["campaignHash"], *core["shardResultHashes"], core["rulesHash"],
            core["bundleHash"], core["analysisProfileHash"],
            core["perspectiveProfileHash"], core["selectionProfileHash"],
            core["diversityProfileHash"], core["deepAnalysisProfileHash"],
        )
        if not core["shardResultHashes"] or any(not _is_sha256(item) for item in hashes):
            raise ValueError("deep analysis batch contains invalid content identities")
        digest = canonical_sha256(core)
        if str(value.get("sha256") or digest) != digest:
            raise ValueError("deep analysis batch hash mismatch")
        return cls(
            core["campaignHash"], tuple(core["shardResultHashes"]),
            core["rulesHash"], core["bundleHash"], core["engineVersion"],
            core["analysisProfileHash"], core["perspectiveProfileHash"],
            core["selectionProfileHash"], core["diversityProfileHash"],
            core["deepAnalysisProfileHash"], core["funnelId"],
            core["funnelVersion"], digest,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "campaignHash": self.campaign_hash,
            "shardResultHashes": list(self.shard_result_hashes),
            "rulesHash": self.rules_hash,
            "bundleHash": self.bundle_hash,
            "engineVersion": self.engine_version,
            "analysisProfileHash": self.analysis_profile_hash,
            "perspectiveProfileHash": self.perspective_profile_hash,
            "selectionProfileHash": self.selection_profile_hash,
            "diversityProfileHash": self.diversity_profile_hash,
            "deepAnalysisProfileHash": self.deep_analysis_profile_hash,
            "funnelId": self.funnel_id,
            "funnelVersion": self.funnel_version,
            "sha256": self.sha256,
        }


class FoundryDeepAnalysisServiceV1:
    def __init__(
        self,
        job_store: Any,
        campaign_store: Any,
        funnel: FoundryAnalysisFunnelV1,
        profile: DeepAnalysisProfileV1,
        *,
        maximum_attempts: int,
        rules_hash: str,
        bundle_hash: str,
        engine_version: str,
        perspective_profile_hash: str,
    ):
        if int(maximum_attempts) < 1:
            raise ValueError("deep analysis maximum attempts must be positive")
        self.job_store = job_store
        self.campaign_store = campaign_store
        self.funnel = funnel
        self.profile = profile
        self.maximum_attempts = int(maximum_attempts)
        identities = {
            "rules": str(rules_hash),
            "bundle": str(bundle_hash),
            "perspective": str(perspective_profile_hash),
        }
        if any(not _is_sha256(value) for value in identities.values()):
            raise ValueError("deep analysis runtime identities must be SHA-256 values")
        if not str(engine_version).strip():
            raise ValueError("deep analysis engine version is required")
        self.rules_hash = identities["rules"]
        self.bundle_hash = identities["bundle"]
        self.engine_version = str(engine_version)
        self.perspective_profile_hash = identities["perspective"]

    def validate_batch(self, spec: DeepAnalysisBatchSpecV1) -> None:
        expected = {
            "rulesHash": self.rules_hash,
            "bundleHash": self.bundle_hash,
            "engineVersion": self.engine_version,
            "perspectiveProfileHash": self.perspective_profile_hash,
            "analysisProfileHash": self.funnel.analyzer.profile.sha256,
            "selectionProfileHash": self.funnel.selection_profile.sha256,
            "diversityProfileHash": self.funnel.diversity_profile.sha256,
            "deepAnalysisProfileHash": self.profile.sha256,
            "funnelId": self.funnel.descriptor.id,
            "funnelVersion": self.funnel.descriptor.version,
        }
        actual = {
            "rulesHash": spec.rules_hash,
            "bundleHash": spec.bundle_hash,
            "engineVersion": spec.engine_version,
            "perspectiveProfileHash": spec.perspective_profile_hash,
            "analysisProfileHash": spec.analysis_profile_hash,
            "selectionProfileHash": spec.selection_profile_hash,
            "diversityProfileHash": spec.diversity_profile_hash,
            "deepAnalysisProfileHash": spec.deep_analysis_profile_hash,
            "funnelId": spec.funnel_id,
            "funnelVersion": spec.funnel_version,
        }
        if actual != expected:
            raise ValueError("deep analysis batch does not match this worker runtime")

    def submit(self, campaign_hash: str) -> dict[str, Any]:
        spec = DeepAnalysisBatchSpecV1.build(
            self.campaign_store,
            campaign_hash,
            funnel=self.funnel,
            profile=self.profile,
        )
        self.validate_batch(spec)
        job_id = f"FDA-{spec.sha256[:24].upper()}"
        self.job_store.create(
            job_id,
            spec.to_dict(),
            kind=DEEP_ANALYSIS_JOB_KIND,
            priority=self.profile.job_priority,
            max_attempts=self.maximum_attempts,
            idempotency_key=spec.sha256,
        )
        return self.job_store.get(job_id) or {}

    def get(self, job_id: str) -> dict[str, Any] | None:
        item = self.job_store.get(job_id)
        return item if item is not None and item.get("kind") == DEEP_ANALYSIS_JOB_KIND else None

    def list(self, *, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        return self.job_store.list_jobs(
            kinds=(DEEP_ANALYSIS_JOB_KIND,),
            status=status,
            limit=limit,
        )

    def cancel(self, job_id: str) -> bool:
        if self.get(job_id) is None:
            return False
        return bool(self.job_store.request_cancel(job_id))


@dataclass(frozen=True, slots=True)
class DeepAnalysisWorkerRunV1:
    worker_id: str
    job_id: str | None
    batch_hash: str | None
    status: str
    result_hash: str | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "workerId": self.worker_id,
            "jobId": self.job_id,
            "batchHash": self.batch_hash,
            "status": self.status,
            "resultHash": self.result_hash,
            "error": self.error,
        }


class FoundryDeepAnalysisWorkerV1:
    descriptor = PluginDescriptor(
        id="foundry_deep_analysis_worker_v1",
        kind="analysis_worker",
        version="1.0.0",
        description="Leased and fenced deep analysis of complete Foundry campaigns",
        capabilities=(
            "all_decisions", "four_observer_projections", "exact_causal_oracle",
            "quality_diversity", "heartbeat", "fencing", "cancellation",
        ),
        resource_class="cpu",
        deterministic=True,
        timeout_seconds=3600,
        maximum_memory_mb=4096,
    )

    def __init__(
        self,
        service: FoundryDeepAnalysisServiceV1,
        artifact_store: Any,
        *,
        worker_id: str,
        lease_seconds: int,
        heartbeat_seconds: int,
    ):
        worker = str(worker_id).strip()
        if not worker:
            raise ValueError("deep analysis worker identity is required")
        if int(lease_seconds) < 3 or int(heartbeat_seconds) < 1:
            raise ValueError("deep analysis lease and heartbeat must be positive")
        if int(heartbeat_seconds) * 3 > int(lease_seconds):
            raise ValueError("deep analysis heartbeat must be no slower than one-third of its lease")
        self.service = service
        self.artifact_store = artifact_store
        self.worker_id = worker
        self.lease_seconds = int(lease_seconds)
        self.heartbeat_seconds = int(heartbeat_seconds)

    def _validate_runtime(self, spec: DeepAnalysisBatchSpecV1) -> None:
        self.service.validate_batch(spec)

    def run_once(self) -> DeepAnalysisWorkerRunV1:
        job = self.service.job_store.claim(
            self.worker_id,
            kinds=(DEEP_ANALYSIS_JOB_KIND,),
            lease_seconds=self.lease_seconds,
        )
        if job is None:
            return DeepAnalysisWorkerRunV1(self.worker_id, None, None, "idle", None, None)
        job_id = str(job["jobId"])
        fencing_token = int(job["attempts"])
        spec: DeepAnalysisBatchSpecV1 | None = None
        next_heartbeat = time.time() + self.heartbeat_seconds
        last_progress = 0.0

        def maintain_lease() -> bool:
            nonlocal next_heartbeat
            if self.service.job_store.cancelled(job_id):
                return True
            now = time.time()
            if now >= next_heartbeat:
                accepted = self.service.job_store.heartbeat(
                    job_id,
                    self.worker_id,
                    lease_seconds=self.lease_seconds,
                    fencing_token=fencing_token,
                )
                if not accepted:
                    raise PermissionError("deep analysis worker lost its fenced lease")
                next_heartbeat = now + self.heartbeat_seconds
            return False

        def report_progress(value: float) -> None:
            nonlocal last_progress
            if maintain_lease():
                raise InterruptedError("deep analysis job was cancelled")
            progress = min(0.99, max(last_progress, float(value)))
            if progress <= last_progress and progress < 0.99:
                return
            accepted = self.service.job_store.update(
                job_id,
                status="running",
                progress=progress,
                worker_id=self.worker_id,
                fencing_token=fencing_token,
            )
            if not accepted:
                raise PermissionError("deep analysis progress lost its fenced lease")
            last_progress = progress

        try:
            spec = DeepAnalysisBatchSpecV1.from_dict(dict(job["request"]))
            self._validate_runtime(spec)
            current = DeepAnalysisBatchSpecV1.build(
                self.service.campaign_store,
                spec.campaign_hash,
                funnel=self.service.funnel,
                profile=self.service.profile,
            )
            if current.sha256 != spec.sha256:
                raise ValueError("deep analysis campaign evidence changed after submission")
            result = self.service.funnel.process_completed_campaign(
                self.service.campaign_store,
                spec.campaign_hash,
                cancel_check=maintain_lease,
                progress_callback=report_progress,
            )
            if result.analysis_profile_hash != spec.analysis_profile_hash:
                raise ValueError(
                    "deep analysis funnel used a different analyzer profile"
                )
            referenced_analysis_hashes = tuple(sorted(
                str(item.get("ownerHash") or "")
                for item in result.artifact_references
                if item.get("ownerKind") == "trace_analysis_v1"
            ))
            if referenced_analysis_hashes != tuple(result.analysis_hashes):
                raise ValueError(
                    "deep analysis funnel analysis membership is incomplete"
                )
            if maintain_lease():
                raise InterruptedError("deep analysis job was cancelled")
            manifest = self.artifact_store.put(
                _canonical_bytes(result.to_dict()),
                media_type="application/vnd.balootx.foundry-analysis-funnel-v1+json",
                name=f"{result.sha256}.foundry-analysis.json",
            )
            reference = {
                "ownerKind": "foundry_analysis_funnel_v1",
                "ownerHash": result.sha256,
                "uri": str(manifest["uri"]),
                "sha256": str(manifest["sha256"]),
                "bytes": int(manifest["bytes"]),
                "mediaType": str(manifest["mediaType"]),
                "name": manifest.get("name"),
            }
            output = {
                "schemaVersion": 1,
                "batchHash": spec.sha256,
                "campaignHash": spec.campaign_hash,
                "funnelResultHash": result.sha256,
                "candidateHashes": list(result.candidate_hashes),
                "selectionHash": result.selection.sha256,
                "artifactReferences": [*result.artifact_references, reference],
            }
            committed = self.service.job_store.update(
                job_id,
                status="completed",
                progress=1.0,
                result=output,
                worker_id=self.worker_id,
                fencing_token=fencing_token,
            )
            if not committed:
                raise PermissionError("deep analysis completion lost its fenced lease")
            return DeepAnalysisWorkerRunV1(
                self.worker_id, job_id, spec.sha256, "completed", result.sha256, None
            )
        except Exception as exc:
            if isinstance(exc, PermissionError):
                return DeepAnalysisWorkerRunV1(
                    self.worker_id,
                    job_id,
                    spec.sha256 if spec is not None else None,
                    "fenced_out",
                    None,
                    f"{exc.__class__.__name__}: {exc}",
                )
            retryable = not isinstance(
                exc, (ValueError, KeyError, InterruptedError)
            )
            latest = self.service.job_store.get(job_id) or job
            status = "cancelled" if isinstance(exc, InterruptedError) else (
                "queued"
                if retryable and int(latest["attempts"]) < int(latest["maxAttempts"])
                else "failed"
            )
            self.service.job_store.update(
                job_id,
                status=status,
                error=f"{exc.__class__.__name__}: {exc}",
                worker_id=self.worker_id,
                fencing_token=fencing_token,
            )
            return DeepAnalysisWorkerRunV1(
                self.worker_id,
                job_id,
                spec.sha256 if spec is not None else None,
                status,
                None,
                f"{exc.__class__.__name__}: {exc}",
            )
