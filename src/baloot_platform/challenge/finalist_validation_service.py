"""Durable local-only finalist replay jobs for selected complete Laffat."""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import uuid
from time import perf_counter as _perf_counter
from typing import Any, Mapping

from .candidate_resimulation import CandidateResimulationV1
from .evidence_v3 import canonical_sha256
from .evidence_v3 import LaffaEvidenceV2
from .finalist_validation import (
    FinalistAnalysisEvidenceV1,
    FinalistLaffaReplayV1,
    FinalistValidationProfileV1,
)
from .oracle import ActualDealOracleVerifierV1, TacticalViabilityProbeV1
from .profiles import ThrillProfileV1
from .scoring import ProfileChallengeScorerV1


FINALIST_VALIDATION_JOB_KIND = "foundry_finalist_validation_v1"
_TRACE_MEDIA_TYPE = "application/vnd.balootx.complete-trace-v1+json"
_EVIDENCE_MEDIA_TYPE = "application/vnd.balootx.finalist-replay-evidence-v1+json"
_ANALYSIS_EVIDENCE_MEDIA_TYPE = (
    "application/vnd.balootx.finalist-analysis-evidence-v1+json"
)
_FUNNEL_MEDIA_TYPE = "application/vnd.balootx.foundry-analysis-funnel-v1+json"
_MANIFEST_MEDIA_TYPE = "application/vnd.balootx.finalist-validation-manifest-v1+json"
_CANDIDATE_RESIMULATION_EVIDENCE_MEDIA_TYPE = (
    "application/vnd.balootx.candidate-resimulation-evidence-v1+json"
)
_TACTICAL_VIABILITY_MEDIA_TYPE = (
    "application/vnd.balootx.tactical-viability-probe-v1+json"
)
_LAFFA_SELECTION_MEDIA_TYPE = (
    "application/vnd.balootx.laffa-content-selection-v1+json"
)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class _JobCancellation:
    def __init__(self, jobs: Any, job_id: str, *, timeout_seconds: int):
        self.jobs = jobs
        self.job_id = job_id
        self.deadline = time.monotonic() + int(timeout_seconds)
        self.timed_out = False

    def __call__(self) -> bool:
        if time.monotonic() >= self.deadline:
            self.timed_out = True
            return True
        return bool(self.jobs.cancelled(self.job_id))


class FinalistValidationJobServiceV1:
    def __init__(
        self,
        *,
        jobs: Any,
        background_executor: Any,
        replay: FinalistLaffaReplayV1,
        profile: FinalistValidationProfileV1,
        candidates: Any,
        funnel: Any,
        artifact_store: Any,
        maximum_attempts: int,
        thrill_profile: ThrillProfileV1,
        scorer: ProfileChallengeScorerV1 | None = None,
        candidate_resimulation: CandidateResimulationV1 | None = None,
        tactical_viability_probe: ActualDealOracleVerifierV1 | None = None,
        audit_repository: Any | None = None,
    ) -> None:
        if maximum_attempts < 1:
            raise ValueError("maximum_attempts must be positive")
        self.jobs = jobs
        self.background_executor = background_executor
        self.replay = replay
        self.profile = profile
        self.candidates = candidates
        self.funnel = funnel
        self.artifact_store = artifact_store
        self.maximum_attempts = int(maximum_attempts)
        self.thrill_profile = thrill_profile
        self.scorer = scorer or ProfileChallengeScorerV1()
        self.candidate_resimulation = candidate_resimulation
        self.tactical_viability_probe = tactical_viability_probe
        self.audit_repository = audit_repository
        self._active_lock = threading.Lock()
        self._active_jobs: set[str] = set()

    def _finalist_analysis_profile_hash(self) -> str:
        profile_hash = str(
            getattr(
                getattr(getattr(self.funnel, "analyzer", None), "profile", None),
                "sha256",
                "",
            )
        )
        if not re.fullmatch(r"[0-9a-f]{64}", profile_hash):
            raise ValueError("finalist analysis funnel profile identity is invalid")
        return profile_hash

    def _validate_request_authority(self, request: Mapping[str, Any]) -> None:
        if (
            request.get("schemaVersion") != 1
            or isinstance(request.get("schemaVersion"), bool)
            or str(request.get("profileHash") or "") != self.profile.sha256
            or str(request.get("thrillProfileHash") or "")
            != self.thrill_profile.sha256
            or str(request.get("finalistAnalysisProfileHash") or "")
            != self._finalist_analysis_profile_hash()
            or dict(request.get("scoringPlugin") or {})
            != {
                "id": str(self.scorer.descriptor.id),
                "version": str(self.scorer.descriptor.version),
            }
            or dict(request.get("replayPlugin") or {})
            != {
                "id": str(self.replay.descriptor.id),
                "version": str(self.replay.descriptor.version),
            }
        ):
            raise ValueError(
                "finalist validation request does not match its frozen runtime authority"
            )

    @staticmethod
    def _audit_record_value(record: Any) -> dict[str, Any]:
        if hasattr(record, "to_dict"):
            return dict(record.to_dict())
        if isinstance(record, Mapping):
            return dict(record)
        raise TypeError("finalist audit repository returned an invalid record")

    def _retain_finalist_audit_trace(
        self,
        descendant: Any,
        *,
        source_parent: Any,
        campaign_hash: str | None,
        request: Mapping[str, Any],
        trace_reference: Mapping[str, Any],
    ) -> Any | None:
        repository = self.audit_repository
        if repository is None or not campaign_hash:
            return None
        parent_record = repository.find_record(
            source_parent.sha256,
            run_code=campaign_hash,
        )
        if parent_record is None:
            # Historical/non-organic candidates predate governed audit retention.
            return None
        parent = self._audit_record_value(parent_record)
        parent_source = dict(parent.get("source") or {})
        parent_record_hash = str(parent.get("recordHash") or "")
        if (
            str(parent.get("traceHash") or "") != source_parent.sha256
            or str(parent_source.get("kind") or "") != "organic_foundry_trace"
            or str(parent_source.get("runCode") or "") != campaign_hash
            or str(parent_source.get("runHash") or "") != campaign_hash
            or not parent_record_hash
        ):
            raise ValueError("finalist audit parent identity mismatch")
        stage_profile_hash = self.profile.sha256
        stage_run_hash = canonical_sha256(request)
        stage_candidate_hash = canonical_sha256(
            {
                "schemaVersion": 1,
                "parentRecordHash": parent_record_hash,
                "stageProfileHash": stage_profile_hash,
                "stageRunHash": stage_run_hash,
            }
        )
        base_keys = (
            "runCode",
            "runHash",
            "shardId",
            "shardHash",
            "shardIndex",
            "seed",
            "candidateOrdinal",
            "attemptCount",
            "fencingToken",
            "workerId",
        )
        missing = [key for key in base_keys if key not in parent_source]
        if missing:
            raise ValueError(
                f"finalist audit parent source is incomplete: {sorted(missing)}"
            )
        source = {
            "kind": "organic_finalist_trace",
            **{key: parent_source[key] for key in base_keys},
            "recipeCode": "organic-finalist-validation-v1",
            "recipeHash": stage_profile_hash,
            "candidateHash": stage_candidate_hash,
            "generationLane": "organic_finalist",
            "traceRole": "finalist_descendant",
            "parentRecordHash": parent_record_hash,
            "parentTraceHash": source_parent.sha256,
            "stageProfileHash": stage_profile_hash,
            "stageRunHash": stage_run_hash,
        }
        provenance = dict(parent.get("configurationProvenance") or {})
        retained = repository.retain(
            descendant,
            rules=self.replay.simulator.rules,
            source=source,
            research_configuration=provenance.get("researchConfiguration"),
        )
        retained_value = self._audit_record_value(retained)
        retained_reference = dict(retained_value.get("completeTrace") or {})
        if str(retained_reference.get("sha256") or "") != str(
            trace_reference.get("sha256") or ""
        ):
            raise ValueError("finalist audit trace did not reuse the retained CAS bytes")
        return retained

    def _finalize_finalist_audit_trace(
        self,
        record: Any | None,
        *,
        publication_ready: bool,
        blockers: tuple[str, ...],
    ) -> None:
        if record is None:
            return
        record_value = self._audit_record_value(record)
        record_hash = str(record_value.get("recordHash") or "")
        if not record_hash:
            raise ValueError("finalist audit trace record lacks its identity")
        failures = tuple(
            dict.fromkeys(str(value).strip() for value in blockers if str(value).strip())
        )
        if not publication_ready and not failures:
            raise ValueError("rejected finalist audit trace requires exact blockers")
        self.audit_repository.finalize(
            record_hash,
            disposition="qualified" if publication_ready else "rejected",
            failure_categories=() if publication_ready else failures,
        )

    def submit(
        self,
        *,
        analysis_job_id: str,
        candidate_hashes: tuple[str, ...],
        idempotency_key: str,
        tenant_id: str = "default",
        candidate_resimulation: bool = False,
        required_decision_kind: str | None = None,
    ) -> dict[str, Any]:
        job = str(analysis_job_id).strip()
        values = tuple(str(item).strip().lower() for item in candidate_hashes)
        if not job or len(job) > 200:
            raise ValueError("finalist validation analysis job identity is invalid")
        if not 1 <= len(values) <= self.profile.maximum_traces_per_job:
            raise ValueError("finalist validation candidate count is out of range")
        if len(set(values)) != len(values) or any(
            not re.fullmatch(r"[0-9a-f]{64}", item) for item in values
        ):
            raise ValueError("finalist validation candidates must be unique SHA-256 values")
        if not idempotency_key or len(idempotency_key) > 200:
            raise ValueError("finalist validation requires a bounded idempotency key")
        if candidate_resimulation and self.candidate_resimulation is None:
            raise ValueError(
                "candidate resimulation was requested but is not configured"
            )
        normalized_required_kind = (
            str(required_decision_kind).strip().lower()
            if required_decision_kind is not None
            else None
        )
        if normalized_required_kind not in {None, "play"}:
            raise ValueError(
                "finalist validation required decision kind is unsupported"
            )
        if (
            normalized_required_kind == "play"
            and self.profile.enable_exact_tactical_viability_precheck
            and self.tactical_viability_probe is None
        ):
            raise ValueError(
                "exact tactical viability precheck is enabled but unavailable"
            )
        resimulation_identity = (
            {
                "profileHash": self.candidate_resimulation.profile.sha256,
                "agentRelease": self.candidate_resimulation.simulator.agent_release,
                "plugin": {
                    "id": self.candidate_resimulation.descriptor.id,
                    "version": self.candidate_resimulation.descriptor.version,
                },
            }
            if candidate_resimulation and self.candidate_resimulation is not None
            else None
        )
        request = {
            "schemaVersion": 1,
            "analysisJobId": job,
            "candidateHashes": list(values),
            "profileHash": self.profile.sha256,
            "finalistAnalysisProfileHash": (
                self._finalist_analysis_profile_hash()
            ),
            "thrillProfileHash": self.thrill_profile.sha256,
            "scoringPlugin": {
                "id": str(self.scorer.descriptor.id),
                "version": str(self.scorer.descriptor.version),
            },
            "replayPlugin": {
                "id": str(self.replay.descriptor.id),
                "version": str(self.replay.descriptor.version),
            },
            "candidateResimulation": resimulation_identity,
            "requiredDecisionKind": normalized_required_kind,
        }
        self._validate_request_authority(request)
        job_id = uuid.uuid4().hex
        actual_id = self.jobs.create(
            job_id,
            request,
            kind=FINALIST_VALIDATION_JOB_KIND,
            tenant_id=tenant_id,
            max_attempts=self.maximum_attempts,
            idempotency_key=idempotency_key,
        )
        if actual_id == job_id:
            self.background_executor.submit(self._execute, job_id, request)
        result = self.jobs.get(actual_id)
        if result is None:
            raise RuntimeError("finalist validation job was not persisted")
        persisted_request = dict(result.get("request") or {})
        persisted_request.pop("idempotencyKey", None)
        if persisted_request != request:
            raise ValueError(
                "Idempotency-Key was already used for a different finalist validation"
            )
        return result

    def get(self, job_id: str) -> dict[str, Any] | None:
        item = self.jobs.get(str(job_id))
        if item is None or item.get("kind") != FINALIST_VALIDATION_JOB_KIND:
            return None
        return item

    def list(
        self,
        *,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return self.jobs.list_jobs(
            kinds=(FINALIST_VALIDATION_JOB_KIND,),
            status=status,
            limit=limit,
        )

    def cancel(self, job_id: str) -> bool:
        if self.get(job_id) is None:
            return False
        accepted = bool(self.jobs.request_cancel(job_id))
        if not accepted:
            return False
        with self._active_lock:
            active = job_id in self._active_jobs
        if not active:
            self.jobs.update(
                job_id,
                status="cancelled",
                progress=1.0,
                error="cancelled before finalist validation resumed",
            )
        return True

    def events(self, job_id: str) -> list[dict[str, Any]] | None:
        if self.get(job_id) is None:
            return None
        return list(self.jobs.events(job_id))

    def resume(self, job_id: str) -> dict[str, Any]:
        item = self.get(job_id)
        if item is None:
            raise KeyError(job_id)
        if item.get("cancelRequested"):
            self.jobs.update(
                job_id,
                status="cancelled",
                progress=1.0,
                error="cancelled before finalist validation resumed",
            )
            return self.get(job_id) or item
        if item["status"] in {"queued", "running"}:
            self._validate_request_authority(dict(item["request"]))
            self.background_executor.submit(self._execute, job_id, dict(item["request"]))
        return self.get(job_id) or item

    def recover_pending(self) -> tuple[str, ...]:
        """Resume local jobs after restart; callers must gate this to single-node mode."""
        recovered: set[str] = set()
        for status in ("queued", "running"):
            for item in self.list(status=status, limit=1_000):
                job_id = str(item["jobId"])
                if item.get("cancelRequested"):
                    self.jobs.update(
                        job_id,
                        status="cancelled",
                        progress=1.0,
                        error="cancelled during finalist validation recovery",
                    )
                else:
                    try:
                        self._validate_request_authority(dict(item["request"]))
                    except ValueError as exc:
                        self.jobs.update(
                            job_id,
                            status="failed",
                            progress=1.0,
                            error=f"ValueError: {exc}",
                        )
                    else:
                        self.background_executor.submit(
                            self._execute, job_id, dict(item["request"])
                        )
                recovered.add(job_id)
        return tuple(sorted(recovered))

    def _put_reference(
        self,
        document: Mapping[str, Any],
        *,
        owner_kind: str,
        owner_hash: str,
        media_type: str,
        name: str,
    ) -> dict[str, Any]:
        payload = _canonical_bytes(document)
        stored = self.artifact_store.put(payload, media_type=media_type, name=name)
        if str(stored["sha256"]) != hashlib.sha256(payload).hexdigest():
            raise ValueError("finalist validation artifact checksum mismatch")
        return {
            "ownerKind": owner_kind,
            "ownerHash": owner_hash,
            "uri": str(stored["uri"]),
            "sha256": str(stored["sha256"]),
            "bytes": int(stored["bytes"]),
            "mediaType": str(stored["mediaType"]),
            "name": stored.get("name"),
        }

    def _load_single_document(
        self,
        references: tuple[Mapping[str, Any], ...],
        *,
        owner_kind: str,
    ) -> Mapping[str, Any]:
        matches = [item for item in references if item.get("ownerKind") == owner_kind]
        if len(matches) != 1:
            raise ValueError(
                f"finalist analysis requires exactly one {owner_kind} artifact"
            )
        reference = matches[0]
        payload = self.artifact_store.get(str(reference.get("uri") or ""))
        if (
            len(payload) != int(reference.get("bytes", -1))
            or hashlib.sha256(payload).hexdigest()
            != str(reference.get("sha256") or "")
        ):
            raise ValueError(f"finalist {owner_kind} artifact integrity failure")
        document = json.loads(payload.decode("utf-8"))
        if not isinstance(document, Mapping):
            raise ValueError(f"finalist {owner_kind} artifact must be an object")
        if str(document.get("sha256") or "") != str(
            reference.get("ownerHash") or ""
        ):
            raise ValueError(f"finalist {owner_kind} owner identity mismatch")
        return document

    def _execute(self, job_id: str, request: Mapping[str, Any]) -> None:
        with self._active_lock:
            if job_id in self._active_jobs:
                return
            self._active_jobs.add(job_id)
        try:
            self._execute_once(job_id, request)
        finally:
            with self._active_lock:
                self._active_jobs.discard(job_id)

    def _execute_once(self, job_id: str, request: Mapping[str, Any]) -> None:
        self._validate_request_authority(request)
        if self.jobs.cancelled(job_id):
            self.jobs.update(
                job_id,
                status="cancelled",
                progress=1.0,
                error="cancelled before finalist validation execution",
            )
            return
        job_started = _perf_counter()
        stage_seconds: dict[str, float] = {}

        def record_stage(name: str, started: float) -> None:
            stage_seconds[name] = stage_seconds.get(name, 0.0) + max(
                0.0,
                _perf_counter() - started,
            )

        self.jobs.update(job_id, status="running", progress=0.02)
        hashes = tuple(str(item) for item in request["candidateHashes"])
        audit_campaign_hash: str | None = None
        replayed: dict[str, dict[str, Any]] = {}
        items: list[dict[str, Any]] = []
        cancellation: _JobCancellation | None = None
        try:
            if self.audit_repository is not None:
                list_candidates = getattr(self.candidates, "list_candidates", None)
                if callable(list_candidates):
                    candidate_page = list_candidates(str(request["analysisJobId"]))
                    if not isinstance(candidate_page, Mapping):
                        raise ValueError("finalist audit candidate catalog is invalid")
                    audit_campaign_hash = str(
                        candidate_page.get("campaignHash") or ""
                    ).strip() or None
            for index, candidate_hash in enumerate(hashes):
                if self.jobs.cancelled(job_id):
                    raise InterruptedError("finalist validation cancelled")
                trace_load_started = _perf_counter()
                parent = self.candidates.candidate_trace(
                    str(request["analysisJobId"]), candidate_hash
                )
                record_stage("candidateTraceLoad", trace_load_started)
                cached = replayed.get(parent.sha256)
                if cached is None:
                    cancellation = _JobCancellation(
                        self.jobs,
                        job_id,
                        timeout_seconds=self.profile.maximum_trace_seconds,
                    )
                    source_parent = parent
                    resimulation_evidence = None
                    resimulation_trace_reference = None
                    resimulation_evidence_reference = None
                    if request.get("candidateResimulation") is not None:
                        if self.candidate_resimulation is None:
                            raise ValueError(
                                "candidate resimulation runtime is unavailable"
                            )
                        resimulation_started = _perf_counter()
                        try:
                            parent, resimulation_evidence = (
                                self.candidate_resimulation.resimulate(
                                    source_parent,
                                    cancel_check=cancellation,
                                )
                            )
                        except InterruptedError:
                            raise
                        except Exception as exc:
                            raise RuntimeError(
                                f"candidate resimulation failed: {exc}"
                            ) from exc
                        finally:
                            record_stage(
                                "candidateResimulation",
                                resimulation_started,
                            )
                        resimulation_trace_reference = self._put_reference(
                            parent.to_dict(trusted=True),
                            owner_kind="candidate_resimulation_trace_v1",
                            owner_hash=parent.sha256,
                            media_type=_TRACE_MEDIA_TYPE,
                            name=f"{parent.sha256}.candidate-resimulation-trace.json",
                        )
                        resimulation_evidence_reference = self._put_reference(
                            resimulation_evidence.to_dict(),
                            owner_kind="candidate_resimulation_evidence_v1",
                            owner_hash=resimulation_evidence.sha256,
                            media_type=_CANDIDATE_RESIMULATION_EVIDENCE_MEDIA_TYPE,
                            name=(
                                f"{resimulation_evidence.sha256}."
                                "candidate-resimulation-evidence.json"
                            ),
                        )
                    replay_started = _perf_counter()
                    descendant, evidence = self.replay.replay(
                        parent,
                        cancel_check=cancellation,
                    )
                    record_stage("finalistReplay", replay_started)
                    artifact_started = _perf_counter()
                    trace_reference = self._put_reference(
                        descendant.to_dict(trusted=True),
                        owner_kind="complete_trace_v1",
                        owner_hash=descendant.sha256,
                        media_type=_TRACE_MEDIA_TYPE,
                        name=f"{descendant.sha256}.finalist-trace.json",
                    )
                    evidence_reference = self._put_reference(
                        evidence.to_dict(),
                        owner_kind="finalist_replay_evidence_v1",
                        owner_hash=evidence.sha256,
                        media_type=_EVIDENCE_MEDIA_TYPE,
                        name=f"{evidence.sha256}.finalist-evidence.json",
                    )
                    audit_trace_record = self._retain_finalist_audit_trace(
                        descendant,
                        source_parent=source_parent,
                        campaign_hash=audit_campaign_hash,
                        request=request,
                        trace_reference=trace_reference,
                    )
                    record_stage("replayArtifactIo", artifact_started)
                    viability_probe: TacticalViabilityProbeV1 | None = None
                    viability_reference = None
                    if (
                        request.get("requiredDecisionKind") == "play"
                        and self.profile.enable_exact_tactical_viability_precheck
                    ):
                        assert self.tactical_viability_probe is not None
                        viability_started = _perf_counter()
                        viability_probe = (
                            self.tactical_viability_probe
                            .probe_verified_tactical_play(
                                descendant,
                                cancel_check=cancellation,
                            )
                        )
                        record_stage("tacticalViabilityProbe", viability_started)
                        artifact_started = _perf_counter()
                        viability_reference = self._put_reference(
                            viability_probe.to_dict(),
                            owner_kind="tactical_viability_probe_v1",
                            owner_hash=viability_probe.sha256,
                            media_type=_TACTICAL_VIABILITY_MEDIA_TYPE,
                            name=(
                                f"{viability_probe.sha256}."
                                "tactical-viability.json"
                            ),
                        )
                        record_stage("viabilityArtifactIo", artifact_started)
                    if viability_probe is not None and viability_probe.can_reject:
                        replay_blockers = tuple(evidence.blockers)
                        item_blockers = tuple(dict.fromkeys(
                            replay_blockers
                            + ("no_verified_tactical_play_opportunity",)
                        ))
                        cached = {
                            "sourceTraceHash": source_parent.sha256,
                            "parentTraceHash": parent.sha256,
                            "candidateResimulationTraceHash": (
                                parent.sha256
                                if resimulation_evidence is not None
                                else None
                            ),
                            "descendantTraceHash": descendant.sha256,
                            "candidateResimulation": (
                                resimulation_evidence.to_dict()
                                if resimulation_evidence is not None
                                else {
                                    "schemaVersion": 1,
                                    "stage": "candidate_resimulation",
                                    "status": "not_requested",
                                }
                            ),
                            "evidence": evidence.to_dict(),
                            "analysisEvidence": None,
                            "tacticalViabilityProbe": viability_probe.to_dict(),
                            "blockers": list(item_blockers),
                            "agentAgreementResolvedByAudit": False,
                            "localPreviewReady": False,
                            "publicationReady": False,
                            "traceArtifact": trace_reference,
                            "evidenceArtifact": evidence_reference,
                            "analysisEvidenceArtifact": None,
                            "analysisFunnelArtifact": None,
                            "laffaSelectionScoreArtifact": None,
                            "analysisArtifacts": [],
                            "tacticalViabilityArtifact": viability_reference,
                            "candidateResimulationTraceArtifact": (
                                resimulation_trace_reference
                            ),
                            "candidateResimulationEvidenceArtifact": (
                                resimulation_evidence_reference
                            ),
                        }
                        self._finalize_finalist_audit_trace(
                            audit_trace_record,
                            publication_ready=False,
                            blockers=item_blockers,
                        )
                        replayed[source_parent.sha256] = cached
                        items.append({"candidateHash": candidate_hash, **cached})
                        self.jobs.update(
                            job_id,
                            progress=min(
                                0.92,
                                0.05 + 0.87 * (index + 1) / len(hashes),
                            ),
                        )
                        continue
                    funnel_started = _perf_counter()
                    funnel_kwargs: dict[str, Any] = {
                        "cancel_check": cancellation,
                        "source_by_trace": {
                            descendant.sha256: "finalist_validation_descendant"
                        },
                        "progress_callback": lambda fraction, item_index=index: self.jobs.update(
                            job_id,
                            progress=min(
                                0.92,
                                0.08
                                + 0.82
                                * (item_index + max(0.0, min(1.0, fraction)))
                                / len(hashes),
                            ),
                        ),
                    }
                    if viability_probe is not None:
                        funnel_kwargs["oracle_action_cache_by_trace"] = {
                            descendant.sha256: (
                                viability_probe.action_oracles_by_decision
                            )
                        }
                    funnel = self.funnel.process(
                        (trace_reference,),
                        **funnel_kwargs,
                    )
                    if (
                        funnel.analysis_profile_hash
                        != str(request["finalistAnalysisProfileHash"])
                    ):
                        raise ValueError(
                            "finalist funnel analyzer profile changed after submission"
                        )
                    if tuple(funnel.trace_hashes) != (descendant.sha256,):
                        raise ValueError(
                            "finalist funnel trace membership is inconsistent"
                        )
                    record_stage("finalistFunnel", funnel_started)
                    analysis_references = tuple(
                        dict(item) for item in funnel.artifact_references
                    )
                    analysis_document = self._load_single_document(
                        analysis_references,
                        owner_kind="trace_analysis_v1",
                    )
                    if tuple(funnel.analysis_hashes) != (
                        str(analysis_document.get("sha256") or ""),
                    ):
                        raise ValueError(
                            "finalist funnel analysis membership is inconsistent"
                        )
                    laffa_document = self._load_single_document(
                        analysis_references,
                        owner_kind="verified_laffa_evidence_v2",
                    )
                    verified_laffa = LaffaEvidenceV2.from_dict(
                        laffa_document,
                        ledger=descendant.decision_ledger,
                    )
                    laffa_selection_score = self.scorer.score(
                        verified_laffa,
                        self.thrill_profile,
                    )
                    laffa_selection_reference = self._put_reference(
                        laffa_selection_score.to_dict(),
                        owner_kind="laffa_content_selection_v1",
                        owner_hash=laffa_selection_score.sha256,
                        media_type=_LAFFA_SELECTION_MEDIA_TYPE,
                        name=(
                            f"{laffa_selection_score.sha256}."
                            "laffa-content-selection.json"
                        ),
                    )
                    analysis_evidence = FinalistAnalysisEvidenceV1.build(
                        trace=descendant,
                        funnel=funnel,
                        analysis_document=analysis_document,
                        verified_laffa=verified_laffa,
                        laffa_selection_score=laffa_selection_score,
                        expected_thrill_profile_hash=self.thrill_profile.sha256,
                    )
                    analysis_evidence_reference = self._put_reference(
                        analysis_evidence.to_dict(),
                        owner_kind="finalist_analysis_evidence_v1",
                        owner_hash=analysis_evidence.sha256,
                        media_type=_ANALYSIS_EVIDENCE_MEDIA_TYPE,
                        name=f"{analysis_evidence.sha256}.finalist-analysis.json",
                    )
                    funnel_reference = self._put_reference(
                        funnel.to_dict(),
                        owner_kind="foundry_analysis_funnel_v1",
                        owner_hash=funnel.sha256,
                        media_type=_FUNNEL_MEDIA_TYPE,
                        name=f"{funnel.sha256}.finalist-funnel.json",
                    )
                    agreement_resolved_by_audit = bool(
                        int(getattr(
                            evidence, "decision_agreement_unresolved_count", 0
                        )) > 0
                        and analysis_evidence.unresolved_decision_count == 0
                        and analysis_evidence.challenge_ready
                    )
                    replay_blockers = tuple(
                        item for item in evidence.blockers
                        if not (
                            agreement_resolved_by_audit
                            and item == "unresolved_agent_decisions"
                        )
                    )
                    replay_publication_ready = bool(
                        evidence.publication_ready
                        or (
                            agreement_resolved_by_audit
                            and evidence.replay_complete
                            and not replay_blockers
                        )
                    )
                    item_blockers = tuple(
                        dict.fromkeys(
                            replay_blockers + tuple(analysis_evidence.blockers)
                        )
                    )
                    cached = {
                        "sourceTraceHash": source_parent.sha256,
                        "parentTraceHash": parent.sha256,
                        "candidateResimulationTraceHash": (
                            parent.sha256
                            if resimulation_evidence is not None
                            else None
                        ),
                        "descendantTraceHash": descendant.sha256,
                        "candidateResimulation": (
                            resimulation_evidence.to_dict()
                            if resimulation_evidence is not None
                            else {
                                "schemaVersion": 1,
                                "stage": "candidate_resimulation",
                                "status": "not_requested",
                            }
                        ),
                        "evidence": evidence.to_dict(),
                        "analysisEvidence": analysis_evidence.to_dict(),
                        "tacticalViabilityProbe": (
                            viability_probe.to_dict()
                            if viability_probe is not None
                            else {
                                "schemaVersion": 1,
                                "status": "not_requested",
                            }
                        ),
                        "blockers": list(item_blockers),
                        "agentAgreementResolvedByAudit": agreement_resolved_by_audit,
                        "localPreviewReady": bool(
                            evidence.local_preview_allowed
                            and analysis_evidence.challenge_ready
                        ),
                        "publicationReady": bool(
                            replay_publication_ready
                            and analysis_evidence.challenge_ready
                        ),
                        "traceArtifact": trace_reference,
                        "evidenceArtifact": evidence_reference,
                        "analysisEvidenceArtifact": analysis_evidence_reference,
                        "analysisFunnelArtifact": funnel_reference,
                        "laffaSelectionScoreArtifact": (
                            laffa_selection_reference
                        ),
                        "analysisArtifacts": [
                            *analysis_references,
                            laffa_selection_reference,
                        ],
                        "tacticalViabilityArtifact": viability_reference,
                        "candidateResimulationTraceArtifact": (
                            resimulation_trace_reference
                        ),
                        "candidateResimulationEvidenceArtifact": (
                            resimulation_evidence_reference
                        ),
                    }
                    self._finalize_finalist_audit_trace(
                        audit_trace_record,
                        publication_ready=bool(cached["publicationReady"]),
                        blockers=item_blockers,
                    )
                    replayed[source_parent.sha256] = cached
                items.append({"candidateHash": candidate_hash, **cached})
                self.jobs.update(
                    job_id,
                    progress=min(0.92, 0.05 + 0.87 * (index + 1) / len(hashes)),
                )
            core = {
                "schemaVersion": 2,
                "analysisJobId": str(request["analysisJobId"]),
                "profileHash": self.profile.sha256,
                "finalistAnalysisProfileHash": str(
                    request["finalistAnalysisProfileHash"]
                ),
                "thrillProfileHash": self.thrill_profile.sha256,
                "scoringPlugin": {
                    "id": str(self.scorer.descriptor.id),
                    "version": str(self.scorer.descriptor.version),
                },
                "items": items,
                "publicationReady": bool(items) and all(
                    bool(item["publicationReady"])
                    for item in items
                ),
                "localPreviewReady": bool(items) and all(
                    bool(item["localPreviewReady"])
                    for item in items
                ),
            }
            manifest = {**core, "sha256": canonical_sha256(core)}
            manifest_reference = self._put_reference(
                manifest,
                owner_kind="finalist_validation_manifest_v1",
                owner_hash=manifest["sha256"],
                media_type=_MANIFEST_MEDIA_TYPE,
                name=f"{manifest['sha256']}.finalist-validation.json",
            )
            self.jobs.update(
                job_id,
                status="completed",
                progress=1.0,
                result={
                    "schemaVersion": 1,
                    "manifestHash": manifest["sha256"],
                    "validatedLaffaCount": len(replayed),
                    "candidateCount": len(items),
                    "publicationReady": manifest["publicationReady"],
                    "localPreviewReady": manifest["localPreviewReady"],
                    "manifestArtifact": manifest_reference,
                    "items": items,
                    "operationalMetrics": {
                        "schemaVersion": 1,
                        "stageWallSeconds": {
                            key: round(value, 6)
                            for key, value in sorted(stage_seconds.items())
                        },
                        "totalWallSeconds": round(
                            max(0.0, _perf_counter() - job_started),
                            6,
                        ),
                    },
                },
            )
        except InterruptedError as exc:
            if cancellation is not None and cancellation.timed_out:
                self.jobs.update(
                    job_id,
                    status="failed",
                    progress=1.0,
                    error=(
                        "finalist validation exceeded the governed "
                        f"{self.profile.maximum_trace_seconds}-second trace limit"
                    ),
                )
            else:
                self.jobs.update(
                    job_id, status="cancelled", progress=1.0, error=str(exc)
                )
        except Exception as exc:
            status = "cancelled" if self.jobs.cancelled(job_id) else "failed"
            self.jobs.update(
                job_id,
                status=status,
                progress=1.0,
                error=f"{type(exc).__name__}: {exc}",
            )
