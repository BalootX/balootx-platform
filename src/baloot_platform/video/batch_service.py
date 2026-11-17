"""Asynchronous admin service for governed observer-safe render batches."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from .batch_executor import RenderBatchExecutorV1
from .render_bundle import (
    RENDER_AUDIT_VIDEO_CANDIDATE_EVIDENCE_KEYS,
    RENDER_DECISION_SELECTION_EVIDENCE_KEYS,
    RENDER_PUBLICATION_POLICY_KEYS,
    RENDER_SOURCE_KEYS,
    RENDER_VIDEO_CANDIDATE_EVIDENCE_KEYS,
    RENDER_VIDEO_PRODUCTION_EVIDENCE_KEYS,
    RenderBundleV1,
    normalize_render_source,
    validate_render_interest_admission,
)
from .render_plan import RenderBatchV1


RENDER_BUNDLE_MEDIA_TYPE = "application/vnd.balootx.render-bundle-v1+json"
_BASENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_RESULT_CODE = re.compile(r"^LAF-[A-F0-9]{10}$")
RENDER_BATCH_REQUEST_KEYS = frozenset(
    {"schemaVersion", "items", "playbackEvidence"}
)
RENDER_BATCH_ITEM_KEYS = frozenset(
    {
        "bundleArtifact",
        "basename",
        "source",
        "proofArtifact",
        "authorityReceipt",
        "locator",
    }
)
RENDER_BUNDLE_ARTIFACT_KEYS = frozenset(
    {"ownerKind", "ownerHash", "uri", "sha256", "bytes", "mediaType", "name"}
)
RENDER_PROOF_ARTIFACT_KEYS = frozenset(
    {"uri", "sha256", "bytes", "mediaType", "name"}
)
RENDER_PLAYER_LOCATOR_KEYS = frozenset({"schemaVersion", "code", "universalLink"})
RENDER_SCENARIO_AUTHORITY_RECEIPT_KEYS = frozenset(
    {
        "schemaVersion",
        "audience",
        "resultCode",
        "resultHash",
        "decisionOrdinal",
        "exactStateId",
        "stateHash",
        "exactStateHash",
        "rulesHash",
        "configurationBundleHash",
        "scenarioHash",
        "proofSemanticSha256",
        "proofArtifactSha256",
        "constraintEvidenceHash",
        "sha256",
    }
)
RESEARCH_PROOF_OUTCOME_MEDIA_TYPE = (
    "application/vnd.baloot.research-proof-outcome-v1+json"
)
ORTOOLS_PROOF_OUTCOME_MEDIA_TYPE = (
    "application/vnd.balootx.ortools-proof-outcome-v1+json"
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _checked_mapping(
    value: Any,
    *,
    allowed_keys: frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    unknown = sorted(str(key) for key in value if key not in allowed_keys)
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {unknown}")
    return value


def _check_source_keys(value: Mapping[str, Any]) -> None:
    source = _checked_mapping(
        value,
        allowed_keys=RENDER_SOURCE_KEYS,
        label="render batch source",
    )
    policy_value = source.get("publicationPolicy")
    if policy_value is None:
        return
    policy = _checked_mapping(
        policy_value,
        allowed_keys=RENDER_PUBLICATION_POLICY_KEYS,
        label="render batch publicationPolicy",
    )
    if "interestAdmission" in policy:
        validate_render_interest_admission(policy["interestAdmission"])
    decision_value = policy.get("decisionSelection")
    if decision_value is not None:
        _checked_mapping(
            decision_value,
            allowed_keys=RENDER_DECISION_SELECTION_EVIDENCE_KEYS,
            label="render batch publicationPolicy.decisionSelection",
        )
    video_value = policy.get("videoProduction")
    if video_value is not None:
        video = _checked_mapping(
            video_value,
            allowed_keys=RENDER_VIDEO_PRODUCTION_EVIDENCE_KEYS,
            label="render batch publicationPolicy.videoProduction",
        )
        candidate_value = video.get("candidate")
        if candidate_value is not None:
            candidate_allowed_keys = (
                RENDER_AUDIT_VIDEO_CANDIDATE_EVIDENCE_KEYS
                if isinstance(candidate_value, Mapping)
                and candidate_value.get("auditOnly") is True
                else RENDER_VIDEO_CANDIDATE_EVIDENCE_KEYS
            )
            _checked_mapping(
                candidate_value,
                allowed_keys=candidate_allowed_keys,
                label="render batch publicationPolicy.videoProduction.candidate",
            )


def _validate_source_publication_claim(value: Mapping[str, Any]) -> None:
    """Keep legacy and governed batch labels fail-closed and unambiguous."""

    audit_present = "auditOnly" in value
    eligible_present = "publicationEligible" in value
    if audit_present != eligible_present:
        raise ValueError(
            "render batch source must declare auditOnly and publicationEligible together"
        )
    if not audit_present:
        if value.get("qualificationStatus") == "quarantined":
            raise ValueError("quarantined render batch source must be audit-only")
        return

    audit_only = value.get("auditOnly")
    eligible = value.get("publicationEligible")
    if not isinstance(audit_only, bool) or not isinstance(eligible, bool):
        raise ValueError(
            "render batch source auditOnly and publicationEligible must be boolean"
        )
    if audit_only != (not eligible):
        raise ValueError(
            "render batch source auditOnly contradicts publication eligibility"
        )
    purpose = value.get("renderPurpose")
    if purpose is not None and eligible != (purpose == "content_candidate"):
        raise ValueError(
            "render batch source purpose contradicts publication eligibility"
        )
    status = value.get("qualificationStatus")
    if status == "quarantined" and not audit_only:
        raise ValueError("quarantined render batch source must be audit-only")
    if eligible and status not in {None, "qualified"}:
        raise ValueError("only a qualified render batch source may be publishable")


def _checked_item(
    value: Any,
) -> tuple[
    str,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    item = _checked_mapping(
        value,
        allowed_keys=RENDER_BATCH_ITEM_KEYS,
        label="render batch item",
    )
    reference = dict(
        _checked_mapping(
            item.get("bundleArtifact"),
            allowed_keys=RENDER_BUNDLE_ARTIFACT_KEYS,
            label="render batch bundleArtifact",
        )
    )
    source_value = item.get("source")
    if source_value is None:
        source: dict[str, Any] = {}
    else:
        source = dict(
            _checked_mapping(
                source_value,
                allowed_keys=RENDER_SOURCE_KEYS,
                label="render batch source",
            )
        )
        _check_source_keys(source)
    proof_value = item.get("proofArtifact")
    proof_artifact = (
        None
        if proof_value is None
        else dict(
            _checked_mapping(
                proof_value,
                allowed_keys=RENDER_PROOF_ARTIFACT_KEYS,
                label="render batch proofArtifact",
            )
        )
    )
    authority_value = item.get("authorityReceipt")
    authority_receipt = (
        None
        if authority_value is None
        else dict(
            _checked_mapping(
                authority_value,
                allowed_keys=RENDER_SCENARIO_AUTHORITY_RECEIPT_KEYS,
                label="render batch authorityReceipt",
            )
        )
    )
    locator_value = item.get("locator")
    locator = (
        None
        if locator_value is None
        else dict(
            _checked_mapping(
                locator_value,
                allowed_keys=RENDER_PLAYER_LOCATOR_KEYS,
                label="render batch locator",
            )
        )
    )
    if locator is not None:
        code = locator.get("code")
        if (
            locator.get("schemaVersion") != 1
            or not isinstance(code, str)
            or _RESULT_CODE.fullmatch(code) is None
            or locator.get("universalLink") != f"/c/{code}"
        ):
            raise ValueError("render batch locator is invalid")
        if (
            source.get("kind") != "research_scenario_result"
            or source.get("code") != code
            or source.get("auditOnly") is not False
            or source.get("publicationEligible") is not True
            or source.get("renderPurpose") != "content_candidate"
            or authority_receipt is None
            or authority_receipt.get("resultCode") != code
        ):
            raise ValueError(
                "render batch locator requires its qualified prepared result authority"
            )
    return (
        str(item.get("basename") or ""),
        reference,
        source,
        proof_artifact,
        authority_receipt,
        locator,
    )


class _JobCancellation:
    def __init__(self, jobs: Any, job_id: str):
        self.jobs = jobs
        self.job_id = job_id

    def is_set(self) -> bool:
        return bool(self.jobs.cancelled(self.job_id))


class RenderBatchServiceV1:
    """Submit, inspect, and cancel content-addressed render batches."""

    def __init__(
        self,
        *,
        jobs: Any,
        background_executor: Any,
        batch_executor: RenderBatchExecutorV1,
        artifact_store: Any,
        output_root: str | Path,
        maximum_attempts: int,
    ) -> None:
        self.jobs = jobs
        self.background_executor = background_executor
        self.batch_executor = batch_executor
        self.artifact_store = artifact_store
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        if maximum_attempts < 1:
            raise ValueError("maximum_attempts must be positive")
        self.maximum_attempts = maximum_attempts
        self._active_lock = threading.Lock()
        self._active_jobs: set[str] = set()

    def submit(
        self,
        items: Sequence[Mapping[str, Any]],
        *,
        idempotency_key: str,
        tenant_id: str = "default",
        playback_evidence: bool = False,
    ) -> dict[str, Any]:
        if not 1 <= len(items) <= 100:
            raise ValueError("render batch requires between one and one hundred items")
        if not idempotency_key or len(idempotency_key) > 200:
            raise ValueError(
                "render batch requires an idempotency key of at most 200 characters"
            )
        if type(playback_evidence) is not bool:
            raise TypeError("render batch playback evidence flag must be boolean")
        request_items: list[dict[str, Any]] = []
        basenames: set[str] = set()
        for item in items:
            (
                basename,
                reference,
                source,
                proof_artifact,
                authority_receipt,
                locator,
            ) = _checked_item(item)
            if not _BASENAME.fullmatch(basename) or basename in {".", ".."}:
                raise ValueError("render batch basename is invalid")
            if basename in basenames:
                raise ValueError("render batch basenames must be unique")
            basenames.add(basename)
            request_item = {"basename": basename, "bundleArtifact": reference}
            bundle = self._load_bundle(reference)
            resolved_source = self._validated_source_metadata(
                source,
                bundle,
            )
            proof_artifact_sha256 = self._validated_proof_artifact(
                proof_artifact,
                bundle,
                authority_receipt=authority_receipt,
            )
            if resolved_source:
                _check_source_keys(resolved_source)
                request_item["source"] = resolved_source
            if proof_artifact_sha256 is not None:
                request_item["proofArtifact"] = proof_artifact
                request_item["authorityReceipt"] = authority_receipt
            if locator is not None:
                request_item["locator"] = locator
            request_items.append(request_item)
        request = {
            "schemaVersion": 1,
            "items": request_items,
            "playbackEvidence": playback_evidence,
        }
        job_id = uuid.uuid4().hex
        actual_id = self.jobs.create(
            job_id,
            request,
            kind="render_batch_v1",
            tenant_id=tenant_id,
            max_attempts=self.maximum_attempts,
            idempotency_key=idempotency_key,
        )
        if actual_id == job_id:
            self.background_executor.submit(self._execute, job_id, request)
        result = self.jobs.get(actual_id)
        if result is None:
            raise RuntimeError("render batch job was not persisted")
        return result

    def get(self, job_id: str) -> dict[str, Any] | None:
        item = self.jobs.get(job_id)
        if item is None or item.get("kind") != "render_batch_v1":
            return None
        return item

    def list(
        self,
        *,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return self.jobs.list_jobs(
            kinds=("render_batch_v1",),
            status=status,
            limit=limit,
        )

    def list_for_source(
        self,
        *,
        source_kind: str,
        source_code: str,
        limit: int = 1_000,
    ) -> list[dict[str, Any]]:
        """Return durable jobs linked to one semantic source object."""
        matches: list[dict[str, Any]] = []
        for job in self.list(limit=limit):
            items = dict(job.get("request") or {}).get("items") or ()
            for item in items:
                source = dict(item.get("source") or {})
                if str(source.get("kind") or "") == str(source_kind) and str(
                    source.get("code") or ""
                ) == str(source_code):
                    matches.append(job)
                    break
        return matches

    def cancel(self, job_id: str) -> bool:
        if self.get(job_id) is None:
            return False
        accepted = bool(self.jobs.request_cancel(job_id))
        if not accepted:
            return False
        # A process restart can leave a durable job marked as running even
        # though no executor owns it anymore.  Cancellation must still reach a
        # terminal state instead of waiting for a thread that no longer exists.
        with self._active_lock:
            active = job_id in self._active_jobs
        if not active:
            self.jobs.update(
                job_id,
                status="cancelled",
                progress=1.0,
                error="cancelled before render execution resumed",
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
                error="cancelled before render execution resumed",
            )
            return self.get(job_id) or item
        if item["status"] in {"queued", "running"}:
            self.background_executor.submit(
                self._execute, job_id, dict(item["request"])
            )
        return self.get(job_id) or item

    def recover_pending(self) -> tuple[str, ...]:
        """Resume durable jobs that lost their in-process executor on restart."""
        recovered: set[str] = set()
        for status in ("queued", "running"):
            for item in self.list(status=status, limit=1_000):
                job_id = str(item["jobId"])
                if item.get("cancelRequested"):
                    self.jobs.update(
                        job_id,
                        status="cancelled",
                        progress=1.0,
                        error="cancelled during render-service recovery",
                    )
                else:
                    self.background_executor.submit(
                        self._execute,
                        job_id,
                        dict(item["request"]),
                    )
                recovered.add(job_id)
        return tuple(sorted(recovered))

    def _load_bundle(self, reference: Mapping[str, Any]) -> RenderBundleV1:
        if (
            reference.get("ownerKind") != "render_bundle_v1"
            or reference.get("mediaType") != RENDER_BUNDLE_MEDIA_TYPE
        ):
            raise ValueError("render batch artifact kind or media type is invalid")
        data = self.artifact_store.get(str(reference.get("uri") or ""))
        if len(data) != int(reference.get("bytes", -1)) or hashlib.sha256(
            data
        ).hexdigest() != reference.get("sha256"):
            raise ValueError("render batch artifact size or checksum mismatch")
        try:
            bundle = RenderBundleV1(json.loads(data.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
            raise ValueError(
                "render batch artifact is not valid RenderBundleV1 JSON"
            ) from exc
        if (
            not bundle.verify()
            or reference.get("ownerHash") != bundle.bundle_sha256
            or reference.get("sha256") != hashlib.sha256(data).hexdigest()
        ):
            raise ValueError("render batch bundle identity or integrity check failed")
        return bundle

    def _validated_proof_artifact(
        self,
        reference: Mapping[str, Any] | None,
        bundle: RenderBundleV1,
        *,
        authority_receipt: Mapping[str, Any] | None,
    ) -> str | None:
        """Bind full proof receipts without making bundle identity time-sensitive."""

        source = bundle.value.get("source")
        is_scenario = (
            isinstance(source, Mapping)
            and source.get("kind") == "research_scenario_result"
        )
        if not is_scenario:
            if reference is not None or authority_receipt is not None:
                raise ValueError(
                    "render proof authority is supported only for Research V4 scenarios"
                )
            return None
        if reference is None or authority_receipt is None:
            raise ValueError(
                "Research V4 scenario render requires its immutable proof artifact "
                "and operator authority receipt"
            )
        receipt = dict(
            _checked_mapping(
                authority_receipt,
                allowed_keys=RENDER_SCENARIO_AUTHORITY_RECEIPT_KEYS,
                label="render batch authorityReceipt",
            )
        )
        receipt_core = dict(receipt)
        declared_receipt_hash = str(receipt_core.pop("sha256", ""))
        if (
            receipt_core.get("schemaVersion") != 1
            or receipt_core.get("audience") != "operator_audit"
            or declared_receipt_hash
            != hashlib.sha256(
                _canonical_json(receipt_core).encode("utf-8")
            ).hexdigest()
            or receipt_core.get("resultCode") != source.get("code")
            or receipt_core.get("decisionOrdinal") != source.get("decisionOrdinal")
            or receipt_core.get("rulesHash") != source.get("rulesHash")
        ):
            raise ValueError("render scenario authority receipt is invalid")
        for key in (
            "resultHash",
            "stateHash",
            "exactStateHash",
            "rulesHash",
            "configurationBundleHash",
            "scenarioHash",
            "proofSemanticSha256",
            "proofArtifactSha256",
            "constraintEvidenceHash",
        ):
            value = receipt_core.get(key)
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("render scenario authority receipt hash is invalid")
        if (
            not isinstance(receipt_core.get("exactStateId"), str)
            or re.fullmatch(
                r"STATE-[A-F0-9]{20}",
                str(receipt_core["exactStateId"]),
            )
            is None
        ):
            raise ValueError("render scenario authority exact-state identity is invalid")
        checked = dict(
            _checked_mapping(
                reference,
                allowed_keys=RENDER_PROOF_ARTIFACT_KEYS,
                label="render batch proofArtifact",
            )
        )
        media_type = checked.get("mediaType")
        if media_type not in {
            RESEARCH_PROOF_OUTCOME_MEDIA_TYPE,
            ORTOOLS_PROOF_OUTCOME_MEDIA_TYPE,
        }:
            raise ValueError("render proof artifact media type is invalid")
        data = self.artifact_store.get(str(checked.get("uri") or ""))
        artifact_sha256 = hashlib.sha256(data).hexdigest()
        if (
            len(data) != int(checked.get("bytes", -1))
            or artifact_sha256 != checked.get("sha256")
        ):
            raise ValueError("render proof artifact size or checksum mismatch")
        try:
            proof = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
            raise ValueError("render proof artifact is invalid JSON") from exc
        if not isinstance(proof, Mapping):
            raise ValueError("render proof artifact must be an object")
        if media_type == ORTOOLS_PROOF_OUTCOME_MEDIA_TYPE:
            if (
                _canonical_json(proof).encode("utf-8") != data
                or artifact_sha256 != receipt_core.get("proofSemanticSha256")
                or artifact_sha256 != receipt_core.get("proofArtifactSha256")
                or proof.get("completionStatus")
                != source.get("proofCompletionStatus")
                or (proof.get("claimSatisfied") is True)
                is not source.get("proofClaimSatisfied")
            ):
                raise ValueError("render OR-Tools proof artifact authority mismatch")
            return artifact_sha256
        proof_receipt_core = dict(proof)
        proof_receipt_hash = str(proof_receipt_core.pop("sha256", ""))
        semantic_hash = str(proof_receipt_core.get("semanticSha256") or "")
        semantic_core = json.loads(_canonical_json(proof_receipt_core))
        semantic_core.pop("semanticSha256", None)
        resources = semantic_core.get("resources")
        if isinstance(resources, Mapping):
            stable_resources = dict(resources)
            stable_resources.pop("elapsedWallSeconds", None)
            stable_resources.pop("elapsedCpuSeconds", None)
            semantic_core["resources"] = stable_resources
        recomputed_semantic = hashlib.sha256(
            _canonical_json(semantic_core).encode("utf-8")
        ).hexdigest()
        recomputed_receipt = hashlib.sha256(
            _canonical_json(proof_receipt_core).encode("utf-8")
        ).hexdigest()
        if (
            semantic_hash != recomputed_semantic
            or proof_receipt_hash != recomputed_receipt
            or semantic_hash != receipt_core.get("proofSemanticSha256")
            or artifact_sha256 != receipt_core.get("proofArtifactSha256")
        ):
            raise ValueError("render proof artifact semantic authority mismatch")
        return artifact_sha256

    @staticmethod
    def _validated_source_metadata(
        caller_source: Mapping[str, Any],
        bundle: RenderBundleV1,
    ) -> dict[str, Any]:
        """Reject caller labels that contradict hash-bound bundle governance."""

        try:
            supplied = json.loads(_canonical_json(dict(caller_source)))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "render batch caller source is not canonical JSON"
            ) from exc
        bundled = bundle.value.get("source")
        # Previously generated bundles had no source block. Preserve their
        # renderability while requiring every new governed bundle to be exact.
        if bundled is None:
            _validate_source_publication_claim(supplied)
            return supplied
        if not isinstance(bundled, Mapping):
            raise ValueError("render batch bundle source metadata is invalid")
        contradictions = sorted(
            key
            for key, value in supplied.items()
            if key not in bundled
            or _canonical_json(bundled[key]) != _canonical_json(value)
        )
        if contradictions:
            raise ValueError(
                "render batch caller source contradicts the immutable bundle: "
                f"{contradictions}"
            )
        resolved = normalize_render_source(
            bundled,
            observer=str(bundle.value.get("observerSeat") or ""),
        )
        _validate_source_publication_claim(resolved)
        return resolved

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
        if self.jobs.cancelled(job_id):
            self.jobs.update(
                job_id,
                status="cancelled",
                progress=1.0,
                error="cancelled before render execution",
            )
            return
        self.jobs.update(job_id, status="running", progress=0.05)
        try:
            checked_request = _checked_mapping(
                request,
                allowed_keys=RENDER_BATCH_REQUEST_KEYS,
                label="render batch request",
            )
            if checked_request.get("schemaVersion") != 1:
                raise ValueError("render batch request schemaVersion must be 1")
            playback_evidence = checked_request.get("playbackEvidence", False)
            if type(playback_evidence) is not bool:
                raise TypeError("render batch playback evidence flag must be boolean")
            raw_items = checked_request.get("items")
            if (
                not isinstance(raw_items, (list, tuple))
                or not 1 <= len(raw_items) <= 100
            ):
                raise ValueError(
                    "render batch request items must contain between one and one hundred objects"
                )
            pairs: list[tuple[RenderBundleV1, str]] = []
            proof_artifact_hashes: dict[str, str] = {}
            basenames: set[str] = set()
            for item in raw_items:
                (
                    basename,
                    reference,
                    source,
                    proof_artifact,
                    authority_receipt,
                    _locator,
                ) = _checked_item(item)
                if not _BASENAME.fullmatch(basename) or basename in {".", ".."}:
                    raise ValueError("render batch basename is invalid")
                if basename in basenames:
                    raise ValueError("render batch basenames must be unique")
                basenames.add(basename)
                bundle = self._load_bundle(reference)
                resolved_source = self._validated_source_metadata(source, bundle)
                proof_artifact_sha256 = self._validated_proof_artifact(
                    proof_artifact,
                    bundle,
                    authority_receipt=authority_receipt,
                )
                if bundle.value.get("source") is not None and (
                    _canonical_json(source) != _canonical_json(resolved_source)
                ):
                    raise ValueError(
                        "persisted governed render batch source does not exactly match "
                        "the immutable bundle"
                    )
                pairs.append(
                    (
                        bundle,
                        basename,
                    )
                )
                if proof_artifact_sha256 is not None:
                    proof_artifact_hashes[
                        bundle.bundle_sha256
                    ] = proof_artifact_sha256
            bundles = {bundle.bundle_sha256: bundle for bundle, _ in pairs}
            batch = RenderBatchV1.create(
                pairs,
                proof_artifact_hashes=proof_artifact_hashes,
            )
            directory = f"batch-{batch.sha256[:16]}"
            output_base_url = f"/outputs/render-batches/{directory}"
            self.jobs.update(job_id, progress=0.15)
            item_progress = {
                "total": len(pairs),
                "finished": 0,
                "ready": 0,
                "failed": 0,
                "cancelled": 0,
                "remaining": len(pairs),
            }
            ready_items: dict[int, dict[str, Any]] = {}
            progress_lock = threading.Lock()

            def record_item(
                result: Any, completed_count: int, total_count: int
            ) -> None:
                with progress_lock:
                    item_progress["finished"] = completed_count
                    item_progress["ready"] += 1 if result.status == "completed" else 0
                    item_progress["failed"] += 1 if result.status == "failed" else 0
                    item_progress["cancelled"] += 1 if result.status == "cancelled" else 0
                    item_progress["remaining"] = total_count - completed_count
                    if result.status == "completed":
                        ready_items[result.ordinal] = {
                            "ordinal": result.ordinal,
                            "basename": pairs[result.ordinal][1],
                            "manifestName": result.manifest_name,
                            "manifestSha256": result.manifest_sha256,
                        }
                    self.jobs.update(
                        job_id,
                        progress=0.15 + 0.8 * completed_count / max(1, total_count),
                        result={
                            "schemaVersion": 1,
                            "batch": batch.to_dict(),
                            "outputBaseUrl": output_base_url,
                            "readyItems": [ready_items[key] for key in sorted(ready_items)],
                            "itemProgress": dict(item_progress),
                        },
                    )

            execution = self.batch_executor.execute_batch(
                batch,
                bundles=bundles,
                output_root=self.output_root,
                cancel_event=_JobCancellation(self.jobs, job_id),
                item_completed=record_item,
                playback_evidence=playback_evidence,
            )
            result = {
                "schemaVersion": 1,
                "batch": batch.to_dict(),
                "execution": execution.to_dict(),
                "outputBaseUrl": output_base_url,
                "manifests": [
                    f"/outputs/render-batches/{directory}/{item.manifest_name}"
                    for item in execution.items
                    if item.manifest_name is not None
                ],
                "readyItems": [ready_items[key] for key in sorted(ready_items)],
                "itemProgress": item_progress,
            }
            if execution.status == "completed":
                self.jobs.update(
                    job_id, status="completed", progress=1.0, result=result
                )
            elif execution.status == "cancelled":
                self.jobs.update(
                    job_id, status="cancelled", progress=1.0, result=result
                )
            else:
                self.jobs.update(
                    job_id,
                    status="failed",
                    progress=1.0,
                    result=result,
                    error="one or more render batch items failed",
                )
        except Exception as exc:
            status = "cancelled" if self.jobs.cancelled(job_id) else "failed"
            self.jobs.update(
                job_id,
                status=status,
                progress=1.0,
                error=f"{type(exc).__name__}: {exc}",
            )
