from __future__ import annotations

import hashlib
import json
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest
from service_test_client import OperatorClient as TestClient

from baloot_platform.audit.models import content_hash
from baloot_platform.service import api
from baloot_platform.storage.artifact_store import LocalContentAddressedArtifactStore
from baloot_platform.storage.jobs import JobStore
from baloot_platform.video.batch_executor import RenderBatchExecutorV1
from baloot_platform.video.batch_service import (
    ORTOOLS_PROOF_OUTCOME_MEDIA_TYPE,
    RENDER_BUNDLE_MEDIA_TYPE,
    RenderBatchServiceV1,
)
from baloot_platform.video.media_validation import (
    REMOTION_RENDERER_VERSION,
    renderer_source_sha256,
)
from baloot_platform.video.render_bundle import RenderBundleV1


class InlineExecutor:
    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except Exception as exc:  # pragma: no cover - mirrors Future behavior
            future.set_exception(exc)
        return future


class DeferredExecutor:
    def __init__(self):
        self.calls = []

    def submit(self, function, *args):
        self.calls.append((function, args))
        return Future()


class FakeRenderer:
    def __init__(self):
        self.calls: list[str] = []
        self.playback_flags: list[bool] = []

    def execute(self, bundle, **kwargs):
        basename = str(kwargs["basename"])
        output = Path(kwargs["output_dir"])
        video = output / f"{basename}.mp4"
        video.write_bytes(bundle.bundle_sha256.encode("ascii"))
        manifest = output / f"{basename}-manifest.json"
        value = {
            "bundleSha256": bundle.bundle_sha256,
            "files": {
                "video": {
                    "name": video.name,
                    "bytes": video.stat().st_size,
                    "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                }
            },
        }
        if kwargs.get("playback_evidence"):
            value["playbackEvidence"] = [dict(checkpointPhase=phase,
                evidenceScope="decision_clip_state_and_encoded_frame_parity")
                for phase in ("context", "question", "reveal", "continuation", "continuation_end") +
                    (("outcome",) if any(row["stage"] == "outcome" for row in bundle.value["timeline"]["segments"]) else ())]
        manifest.write_text(json.dumps(value), encoding="utf-8")
        self.calls.append(basename)
        self.playback_flags.append(bool(kwargs.get("playback_evidence")))
        return SimpleNamespace(manifest=manifest)


def _bundle(root: Path) -> RenderBundleV1:
    from test_remotion_renderer import _episode, _renderer

    raw = _renderer(root).build_render_request(_episode(root), export_preset="decision_clip").to_dict()
    raw["render"]["concurrency"] = 1
    raw["renderer"]["version"] = REMOTION_RENDERER_VERSION
    raw["renderer"]["sourceSha256"] = renderer_source_sha256(root)
    core = {key: value for key, value in raw.items() if key != "bundleSha256"}
    raw["bundleSha256"] = content_hash(core)
    bundle = RenderBundleV1(raw)
    assert bundle.verify()
    return bundle


def _publication_policy(*, blockers: list[str]) -> dict[str, object]:
    decision_core = {"schemaVersion": 1, "gatePassed": not blockers}
    video_core = {"schemaVersion": 1, "gatePassed": not blockers}
    policy_core = {
        "schemaVersion": 1,
        "decisionSelection": {
            **decision_core,
            "sha256": content_hash(decision_core),
        },
        "videoProduction": {
            **video_core,
            "sha256": content_hash(video_core),
        },
        "blockers": blockers,
    }
    return {**policy_core, "sha256": content_hash(policy_core)}


def _governed_bundle(root: Path) -> RenderBundleV1:
    raw = _bundle(root).to_dict()
    raw["source"] = {
        "sourceSchemaVersion": 2,
        "kind": "research_result",
        "code": "LAF-GOVERNED01",
        "decisionOrdinal": 1,
        "decisionId": "D1",
        "resultHash": "1" * 64,
        "traceHash": "2" * 64,
        "analysisHash": "3" * 64,
        "auditHash": "4" * 64,
        "analysisProfileHash": "5" * 64,
        "analyzerId": "adaptive_finalist_trace_analyzer_v1",
        "analyzerVersion": "1.3.0",
        "rulesHash": "6" * 64,
        "exactStateId": "D1",
        "qrActionable": True,
        "qualificationStatus": "qualified",
        "renderPurpose": "content_candidate",
        "qualificationFailures": [],
        "publicationBlockers": [],
        "observerSeat": raw["observerSeat"],
        "auditOnly": False,
        "publicationEligible": True,
        "publicationPolicy": _publication_policy(blockers=[]),
    }
    core = {key: value for key, value in raw.items() if key != "bundleSha256"}
    raw["bundleSha256"] = content_hash(core)
    bundle = RenderBundleV1(raw)
    assert bundle.verify()
    return bundle


def _reference(store: LocalContentAddressedArtifactStore, bundle: RenderBundleV1):
    stored = store.put(
        json.dumps(bundle.to_dict(), ensure_ascii=False, sort_keys=True).encode(
            "utf-8"
        ),
        media_type=RENDER_BUNDLE_MEDIA_TYPE,
        name="bundle.json",
    )
    return {
        "ownerKind": "render_bundle_v1",
        "ownerHash": bundle.bundle_sha256,
        **stored,
    }


def test_render_batch_service_is_idempotent_and_content_addressed(
    root: Path, tmp_path: Path
) -> None:
    jobs = JobStore(tmp_path / "jobs.sqlite3")
    artifacts = LocalContentAddressedArtifactStore(tmp_path / "artifacts")
    renderer = FakeRenderer()
    service = RenderBatchServiceV1(
        jobs=jobs,
        background_executor=InlineExecutor(),
        batch_executor=RenderBatchExecutorV1(renderer, maximum_concurrency=1),
        artifact_store=artifacts,
        output_root=tmp_path / "outputs" / "render-batches",
        maximum_attempts=3,
    )
    bundle = _bundle(root)
    request = (
        {
            "basename": "decision",
            "bundleArtifact": _reference(artifacts, bundle),
            "source": {
                "kind": "research_result",
                "code": "LAF-RENDERTEST01",
                "decisionOrdinal": 3,
                "traceHash": "a" * 64,
                "renderPurpose": "research_audit",
            },
        },
    )

    first = service.submit(
        request, idempotency_key="batch-1", playback_evidence=True
    )
    second = service.submit(
        request, idempotency_key="batch-1", playback_evidence=True
    )
    assert first["status"] == "completed"
    assert second["jobId"] == first["jobId"]
    assert renderer.calls == ["decision"]
    assert renderer.playback_flags == [True]
    assert first["request"]["playbackEvidence"] is True
    assert first["request"]["items"][0]["source"]["code"] == "LAF-RENDERTEST01"
    assert [
        item["jobId"]
        for item in service.list_for_source(
            source_kind="research_result", source_code="LAF-RENDERTEST01"
        )
    ] == [first["jobId"]]
    assert (
        service.list_for_source(
            source_kind="research_result", source_code="LAF-NOT-THERE"
        )
        == []
    )
    laffa = service.submit(
        (
            {
                "basename": "legacy-laffa-audit",
                "bundleArtifact": request[0]["bundleArtifact"],
                "source": {
                    "kind": "laffa",
                    "code": "BX-LEGACY01",
                    "traceHash": "b" * 64,
                    "renderPurpose": "research_audit",
                },
            },
        ),
        idempotency_key="batch-legacy-laffa",
    )
    assert [
        item["jobId"]
        for item in service.list_for_source(
            source_kind="laffa", source_code="BX-LEGACY01"
        )
    ] == [laffa["jobId"]]
    assert first["result"]["execution"]["status"] == "completed"
    assert first["result"]["itemProgress"] == {
        "total": 1,
        "finished": 1,
        "ready": 1,
        "failed": 0,
        "cancelled": 0,
        "remaining": 0,
    }
    assert first["result"]["readyItems"] == [
        {
            "ordinal": 0,
            "basename": "decision",
            "manifestName": "decision-manifest.json",
            "manifestSha256": first["result"]["execution"]["items"][0][
                "manifestSha256"
            ],
        }
    ]
    assert first["result"]["outputBaseUrl"].startswith("/outputs/render-batches/batch-")
    with pytest.raises(ValueError, match="basename is invalid"):
        service.submit(
            ({"basename": "..", "bundleArtifact": request[0]["bundleArtifact"]},),
            idempotency_key="unsafe-path",
        )


def test_render_batch_service_cancels_queued_work_and_rejects_bad_artifacts(
    root: Path, tmp_path: Path
) -> None:
    jobs = JobStore(tmp_path / "jobs.sqlite3")
    artifacts = LocalContentAddressedArtifactStore(tmp_path / "artifacts")
    deferred = DeferredExecutor()
    service = RenderBatchServiceV1(
        jobs=jobs,
        background_executor=deferred,
        batch_executor=RenderBatchExecutorV1(FakeRenderer(), maximum_concurrency=1),
        artifact_store=artifacts,
        output_root=tmp_path / "outputs",
        maximum_attempts=3,
    )
    bundle = _bundle(root)
    reference = _reference(artifacts, bundle)
    queued = service.submit(
        ({"basename": "decision", "bundleArtifact": reference},),
        idempotency_key="cancel-me",
    )
    assert service.cancel(queued["jobId"])
    function, args = deferred.calls.pop()
    function(*args)
    assert service.get(queued["jobId"])["status"] == "cancelled"

    broken = dict(reference)
    broken["sha256"] = "f" * 64
    failed_service = RenderBatchServiceV1(
        jobs=jobs,
        background_executor=InlineExecutor(),
        batch_executor=RenderBatchExecutorV1(FakeRenderer(), maximum_concurrency=1),
        artifact_store=artifacts,
        output_root=tmp_path / "failed",
        maximum_attempts=3,
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        failed_service.submit(
            ({"basename": "broken", "bundleArtifact": broken},),
            idempotency_key="broken",
        )


def test_render_batch_derives_and_persists_exact_governed_bundle_source(
    root: Path,
    tmp_path: Path,
) -> None:
    jobs = JobStore(tmp_path / "jobs.sqlite3")
    artifacts = LocalContentAddressedArtifactStore(tmp_path / "artifacts")
    deferred = DeferredExecutor()
    service = RenderBatchServiceV1(
        jobs=jobs,
        background_executor=deferred,
        batch_executor=RenderBatchExecutorV1(FakeRenderer(), maximum_concurrency=1),
        artifact_store=artifacts,
        output_root=tmp_path / "outputs",
        maximum_attempts=3,
    )
    bundle = _governed_bundle(root)
    submitted = service.submit(
        ({"basename": "governed", "bundleArtifact": _reference(artifacts, bundle)},),
        idempotency_key="governed-source",
    )

    assert submitted["request"]["items"][0]["source"] == bundle.value["source"]

    tampered = json.loads(json.dumps(submitted["request"]))
    tampered["items"][0]["source"]["code"] = "LAF-FOREIGN01"
    service._execute_once(submitted["jobId"], tampered)
    failed = service.get(submitted["jobId"])
    assert failed["status"] == "failed"
    assert "contradicts the immutable bundle" in failed["error"]


def test_render_batch_persists_only_authority_bound_same_origin_player_locator(
    root: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    jobs = JobStore(tmp_path / "jobs.sqlite3")
    artifacts = LocalContentAddressedArtifactStore(tmp_path / "artifacts")
    service = RenderBatchServiceV1(
        jobs=jobs,
        background_executor=DeferredExecutor(),
        batch_executor=RenderBatchExecutorV1(FakeRenderer(), maximum_concurrency=1),
        artifact_store=artifacts,
        output_root=tmp_path / "outputs",
        maximum_attempts=3,
    )
    bundle = _bundle(root)
    code = "LAF-A1B2C3D4E5"
    source = {
        "kind": "research_scenario_result",
        "code": code,
        "qualificationStatus": "qualified",
        "renderPurpose": "content_candidate",
        "auditOnly": False,
        "publicationEligible": True,
    }
    authority = {"resultCode": code}
    locator = {
        "schemaVersion": 1,
        "code": code,
        "universalLink": f"/c/{code}",
    }
    monkeypatch.setattr(service, "_load_bundle", lambda _reference: bundle)
    monkeypatch.setattr(
        service,
        "_validated_source_metadata",
        lambda caller_source, _bundle: dict(caller_source),
    )
    monkeypatch.setattr(
        service,
        "_validated_proof_artifact",
        lambda _reference, _bundle, *, authority_receipt: (
            "f" * 64 if authority_receipt is not None else None
        ),
    )
    base_item = {
        "basename": "qualified-scenario",
        "bundleArtifact": _reference(artifacts, bundle),
        "source": source,
        "proofArtifact": {
            "uri": "cas://sha256/" + "f" * 64,
            "sha256": "f" * 64,
            "bytes": 1,
            "mediaType": ORTOOLS_PROOF_OUTCOME_MEDIA_TYPE,
            "name": "proof.json",
        },
        "authorityReceipt": authority,
        "locator": locator,
    }

    submitted = service.submit((base_item,), idempotency_key="player-locator")
    assert submitted["request"]["items"][0]["locator"] == locator

    for invalid_item in (
        {**base_item, "locator": {**locator, "code": "LAF-FFFFFFFFFF"}},
        {
            **base_item,
            "source": {
                **source,
                "auditOnly": True,
                "publicationEligible": False,
                "renderPurpose": "research_audit",
            },
        },
    ):
        with pytest.raises(ValueError, match="render batch locator"):
            service.submit((invalid_item,), idempotency_key="invalid-player-locator")

    legacy_item = dict(base_item)
    legacy_item.pop("locator")
    legacy = service.submit((legacy_item,), idempotency_key="no-player-locator")
    assert "locator" not in legacy["request"]["items"][0]


def test_render_batch_service_recovers_interrupted_jobs_and_terminalizes_stale_cancellation(
    root: Path, tmp_path: Path
) -> None:
    jobs = JobStore(tmp_path / "jobs.sqlite3")
    artifacts = LocalContentAddressedArtifactStore(tmp_path / "artifacts")
    deferred = DeferredExecutor()
    bundle = _bundle(root)
    reference = _reference(artifacts, bundle)
    original = RenderBatchServiceV1(
        jobs=jobs,
        background_executor=deferred,
        batch_executor=RenderBatchExecutorV1(FakeRenderer(), maximum_concurrency=1),
        artifact_store=artifacts,
        output_root=tmp_path / "outputs",
        maximum_attempts=3,
    )
    interrupted = original.submit(
        ({"basename": "recovered", "bundleArtifact": reference},),
        idempotency_key="recover-me",
    )
    jobs.update(interrupted["jobId"], status="running", progress=0.15)

    renderer = FakeRenderer()
    restarted = RenderBatchServiceV1(
        jobs=jobs,
        background_executor=InlineExecutor(),
        batch_executor=RenderBatchExecutorV1(renderer, maximum_concurrency=1),
        artifact_store=artifacts,
        output_root=tmp_path / "outputs",
        maximum_attempts=3,
    )
    assert restarted.recover_pending() == (interrupted["jobId"],)
    assert restarted.get(interrupted["jobId"])["status"] == "completed"
    assert renderer.calls == ["recovered"]

    stale = original.submit(
        ({"basename": "stale", "bundleArtifact": reference},),
        idempotency_key="cancel-stale",
    )
    jobs.update(stale["jobId"], status="running", progress=0.15)
    assert restarted.cancel(stale["jobId"])
    terminal = restarted.get(stale["jobId"])
    assert terminal["status"] == "cancelled"
    assert terminal["progress"] == 1.0


class FakeApiRenderBatches:
    def __init__(self):
        self.submissions = []
        self.item = {
            "jobId": "render-api-1",
            "tenantId": "default",
            "kind": "render_batch_v1",
            "status": "completed",
            "priority": 100,
            "progress": 1.0,
            "request": {"schemaVersion": 1, "items": []},
            "result": {"execution": {"status": "completed"}},
            "error": None,
            "attempts": 0,
            "maxAttempts": 1,
            "leaseOwner": None,
            "leaseUntil": None,
            "cancelRequested": False,
            "createdAt": "2026-08-03T00:00:00Z",
            "updatedAt": "2026-08-03T00:00:00Z",
        }

    def submit(self, items, *, idempotency_key, playback_evidence=False):
        self.submissions.append((items, idempotency_key, playback_evidence))
        return self.item

    def get(self, job_id):
        return self.item if job_id == self.item["jobId"] else None

    def list(self, *, status=None, limit=100):
        return (
            [self.item] if status in {None, self.item["status"]} and limit > 0 else []
        )

    def cancel(self, job_id):
        return job_id == self.item["jobId"]

    def events(self, job_id):
        return [{"type": "completed", "payload": {"progress": 1.0}, "createdAt": "now"}]


def test_render_batch_admin_api_requires_idempotency_and_streams_events(
    monkeypatch,
) -> None:
    fake = FakeApiRenderBatches()
    monkeypatch.setattr(api, "RENDER_BATCHES", fake)
    client = TestClient(api.app)
    artifact = {
        "ownerKind": "render_bundle_v1",
        "ownerHash": "a" * 64,
        "uri": "cas://sha256/" + "b" * 64,
        "sha256": "b" * 64,
        "bytes": 12,
        "mediaType": RENDER_BUNDLE_MEDIA_TYPE,
        "name": "bundle.json",
    }
    body = {"items": [{"basename": "angle", "bundleArtifact": artifact}]}
    assert client.post("/api/v2/render-batches", json=body).status_code == 428
    created = client.post(
        "/api/v2/render-batches",
        json=body,
        headers={"Idempotency-Key": "api-batch-1"},
    )
    assert created.status_code == 202
    assert fake.submissions[0][1] == "api-batch-1"
    listed = client.get("/api/v2/render-batches", params={"limit": 20})
    assert listed.status_code == 200
    assert listed.json()["items"][0]["jobId"] == "render-api-1"
    assert client.get("/api/v2/render-batches/render-api-1").status_code == 200
    events = client.get("/api/v2/render-batches/render-api-1/events")
    assert events.status_code == 200
    assert events.headers["content-type"].startswith("text/event-stream")
    assert "event: completed" in events.text
    assert client.get("/api/v2/render-batches/missing").status_code == 404


def test_render_batch_admin_api_accepts_ortools_proof_artifact(monkeypatch) -> None:
    fake = FakeApiRenderBatches()
    monkeypatch.setattr(api, "RENDER_BATCHES", fake)
    proof_hash = "c" * 64
    body = {
        "items": [
            {
                "basename": "ortools-result",
                "bundleArtifact": {
                    "ownerKind": "render_bundle_v1",
                    "ownerHash": "a" * 64,
                    "uri": "cas://sha256/" + "b" * 64,
                    "sha256": "b" * 64,
                    "bytes": 12,
                    "mediaType": RENDER_BUNDLE_MEDIA_TYPE,
                    "name": "bundle.json",
                },
                "proofArtifact": {
                    "uri": "cas://sha256/" + proof_hash,
                    "sha256": proof_hash,
                    "bytes": 12,
                    "mediaType": ORTOOLS_PROOF_OUTCOME_MEDIA_TYPE,
                    "name": "proof.json",
                },
            }
        ]
    }

    response = TestClient(api.app).post(
        "/api/v2/render-batches",
        json=body,
        headers={"Idempotency-Key": "ortools-render"},
    )

    assert response.status_code == 202
    assert fake.submissions[0][0][0]["proofArtifact"]["mediaType"] == (
        ORTOOLS_PROOF_OUTCOME_MEDIA_TYPE
    )
