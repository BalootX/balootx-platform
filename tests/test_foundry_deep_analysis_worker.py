from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from baloot_platform.challenge.campaign import SimulationCampaignSpecV1
from baloot_platform.challenge.deep_analysis import (
    DeepAnalysisBatchSpecV1,
    DeepAnalysisProfileV1,
    FoundryDeepAnalysisServiceV1,
    FoundryDeepAnalysisWorkerV1,
)
from baloot_platform.challenge.funnel import FoundryAnalysisFunnelV1
from baloot_platform.plugins.base import PluginDescriptor
from baloot_platform.storage.artifact_store import LocalContentAddressedArtifactStore
from baloot_platform.storage.jobs import JobStore


@dataclass(frozen=True)
class _FakeResult:
    sha256: str = "d" * 64
    analysis_profile_hash: str = "1" * 64
    analysis_hashes: tuple[str, ...] = ("9" * 64,)
    candidate_hashes: tuple[str, ...] = ("e" * 64,)
    artifact_references: tuple[dict, ...] = field(
        default_factory=lambda: ({
            "ownerKind": "trace_analysis_v1",
            "ownerHash": "9" * 64,
        },)
    )
    selection: object = field(
        default_factory=lambda: SimpleNamespace(sha256="f" * 64)
    )

    def to_dict(self):
        return {
            "schemaVersion": 1,
            "candidateHashes": list(self.candidate_hashes),
            "selectionHash": self.selection.sha256,
            "sha256": self.sha256,
        }


class _FakeCampaignStore:
    def __init__(self, spec: SimulationCampaignSpecV1, *, completed: bool = True):
        self.spec = spec
        self.completed = completed

    def get_campaign(self, campaign_hash: str):
        if campaign_hash != self.spec.sha256:
            return None
        return {
            "campaignHash": campaign_hash,
            "spec": self.spec.to_dict(),
            "status": "completed" if self.completed else "running",
            "cancelRequested": False,
        }

    def list_shards(self, campaign_hash: str):
        return [{
            "status": "completed" if self.completed else "running",
            "spec": self.spec.shards()[0].to_dict(),
            "result": {
                "sha256": "a" * 64,
                "candidateHashes": ["b" * 64],
                "artifactReferences": [{
                    "ownerKind": "complete_trace_v1",
                    "ownerHash": "b" * 64,
                    "sha256": "c" * 64,
                }],
            } if self.completed else None,
        }]


class _FakeFunnel:
    descriptor = PluginDescriptor(
        id="foundry_analysis_funnel_v1",
        kind="trace_analysis_funnel",
        version="1.0.0",
        description="test funnel",
    )

    def __init__(self, analysis_hash: str, selection_hash: str, diversity_hash: str):
        self.analyzer = SimpleNamespace(profile=SimpleNamespace(sha256=analysis_hash))
        self.selection_profile = SimpleNamespace(sha256=selection_hash)
        self.diversity_profile = SimpleNamespace(sha256=diversity_hash)
        self.calls = 0

    def process_completed_campaign(
        self,
        campaign_store,
        campaign_hash,
        *,
        cancel_check=None,
        progress_callback=None,
    ):
        self.calls += 1
        if cancel_check is not None and cancel_check():
            raise InterruptedError("cancelled")
        if progress_callback is not None:
            progress_callback(0.5)
            progress_callback(1.0)
        return _FakeResult(analysis_profile_hash=self.analyzer.profile.sha256)


def _service(tmp_path, *, completed: bool = True):
    analysis_hash, perspective_hash, selection_hash = "1" * 64, "2" * 64, "3" * 64
    spec = SimulationCampaignSpecV1.build(
        rules_hash="4" * 64,
        bundle_hash="5" * 64,
        engine_version="test",
        analysis_profile_hash=analysis_hash,
        perspective_profile_hash=perspective_hash,
        selection_profile_hash=selection_hash,
        agent_release="expert-test",
        seat_agents=("expert-test",) * 4,
        lanes=("organic",),
        lane_profile_hashes={"organic": "6" * 64},
        seed_start=0,
        seed_count=1,
        seeds_per_shard=1,
    )
    profile = DeepAnalysisProfileV1.build({
        "schemaVersion": 1,
        "id": "deep-test",
        "revision": "1",
        "maximumOracleDecisionsPerTrace": 0,
        "maximumOracleRemainingCards": 12,
        "maximumConfirmationDecisionsPerTrace": 1,
        "minimumScreeningScore": 25.0,
        "jobPriority": 10,
    })
    jobs = JobStore(tmp_path / "jobs.sqlite3")
    campaigns = _FakeCampaignStore(spec, completed=completed)
    funnel = _FakeFunnel(analysis_hash, selection_hash, "7" * 64)
    service = FoundryDeepAnalysisServiceV1(
        jobs,
        campaigns,
        funnel,
        profile,
        maximum_attempts=3,
        rules_hash="4" * 64,
        bundle_hash="5" * 64,
        engine_version="test",
        perspective_profile_hash=perspective_hash,
    )
    return service, spec, funnel


def test_deep_analysis_batch_is_idempotent_and_worker_commits_artifact(tmp_path):
    service, spec, funnel = _service(tmp_path)
    first = service.submit(spec.sha256)
    second = service.submit(spec.sha256)
    assert first["jobId"] == second["jobId"]

    artifacts = LocalContentAddressedArtifactStore(tmp_path / "artifacts")
    worker = FoundryDeepAnalysisWorkerV1(
        service,
        artifacts,
        worker_id="worker-a",
        lease_seconds=30,
        heartbeat_seconds=10,
    )
    run = worker.run_once()
    assert run.status == "completed"
    assert funnel.calls == 1
    stored = service.get(first["jobId"])
    assert stored["status"] == "completed"
    assert stored["progress"] == 1.0
    assert stored["result"]["batchHash"] == run.batch_hash
    assert stored["result"]["funnelResultHash"] == "d" * 64
    result_reference = stored["result"]["artifactReferences"][-1]
    assert artifacts.get(result_reference["uri"])
    assert worker.run_once().status == "idle"


@pytest.mark.parametrize(
    ("result", "message"),
    (
        (
            _FakeResult(analysis_profile_hash="0" * 64),
            "different analyzer profile",
        ),
        (
            _FakeResult(analysis_hashes=("8" * 64,)),
            "analysis membership is incomplete",
        ),
    ),
)
def test_deep_worker_rejects_funnel_authority_or_membership_substitution(
    tmp_path,
    result,
    message,
):
    service, spec, funnel = _service(tmp_path)
    submitted = service.submit(spec.sha256)
    funnel.process_completed_campaign = lambda *_args, **_kwargs: result
    worker = FoundryDeepAnalysisWorkerV1(
        service,
        LocalContentAddressedArtifactStore(tmp_path / "artifacts"),
        worker_id="worker-authority",
        lease_seconds=30,
        heartbeat_seconds=10,
    )

    run = worker.run_once()

    assert run.status == "failed"
    failed = service.get(submitted["jobId"])
    assert failed["status"] == "failed"
    assert message in failed["error"]


def test_deep_analysis_rejects_incomplete_campaign_and_cancels_queued_job(tmp_path):
    incomplete, spec, _ = _service(tmp_path / "incomplete", completed=False)
    try:
        incomplete.submit(spec.sha256)
    except ValueError as exc:
        assert "completed" in str(exc)
    else:
        raise AssertionError("incomplete campaign entered deep analysis")

    service, spec, _ = _service(tmp_path / "complete")
    job = service.submit(spec.sha256)
    assert service.cancel(job["jobId"])
    assert service.get(job["jobId"])["status"] == "cancelled"


def test_deep_analysis_rejects_completed_campaign_without_retained_candidate(tmp_path):
    service, spec, _ = _service(tmp_path)
    original = service.campaign_store.list_shards

    def without_candidates(campaign_hash):
        shards = original(campaign_hash)
        shards[0]["result"]["candidateHashes"] = []
        return shards

    service.campaign_store.list_shards = without_candidates
    try:
        service.submit(spec.sha256)
    except ValueError as exc:
        assert "retained candidate" in str(exc)
    else:
        raise AssertionError("an empty campaign entered deep analysis")


def test_job_store_fencing_rejects_stale_worker_completion(tmp_path):
    store = JobStore(tmp_path / "fencing.sqlite3")
    store.create("job-1", {"value": 1}, kind="foundry_deep_analysis_v1")
    first = store.claim("worker-a", kinds=("foundry_deep_analysis_v1",), lease_seconds=30)
    assert first["attempts"] == 1
    with store._connect() as connection:
        connection.execute("UPDATE jobs SET lease_until=0 WHERE id='job-1'")
    second = store.claim("worker-b", kinds=("foundry_deep_analysis_v1",), lease_seconds=30)
    assert second["attempts"] == 2
    assert not store.update(
        "job-1", status="completed", worker_id="worker-a", fencing_token=1
    )
    assert store.update(
        "job-1", status="completed", worker_id="worker-b", fencing_token=2
    )
    assert store.get("job-1")["status"] == "completed"


def test_completed_campaign_filters_non_trace_artifacts_and_preserves_lane(monkeypatch):
    funnel = object.__new__(FoundryAnalysisFunnelV1)
    captured = {}
    sentinel = object()

    def process(
        references,
        *,
        cancel_check=None,
        source_by_trace=None,
        progress_callback=None,
    ):
        captured["references"] = tuple(references)
        captured["sources"] = dict(source_by_trace or {})
        captured["progressCallback"] = progress_callback
        return sentinel

    monkeypatch.setattr(funnel, "process", process)
    trace = {
        "ownerKind": "complete_trace_v1",
        "ownerHash": "8" * 64,
        "sha256": "9" * 64,
    }
    campaign_store = SimpleNamespace(
        get_campaign=lambda _: {"status": "completed", "cancelRequested": False},
        list_shards=lambda _: [
            {
                "status": "completed",
                "spec": {"lane": "guided"},
                "result": {"candidateHashes": ["8" * 64], "artifactReferences": [
                    trace,
                    {"ownerKind": "guided_search_result_v1", "ownerHash": "a" * 64},
                ]},
            },
            {
                "status": "completed",
                "spec": {"lane": "expert"},
                "result": {
                    "candidateHashes": ["8" * 64],
                    "artifactReferences": [trace],
                },
            },
        ],
    )
    assert funnel.process_completed_campaign(campaign_store, "campaign") is sentinel
    assert captured["references"] == (trace,)
    assert captured["sources"] == {"8" * 64: "expert+guided"}
    assert captured["progressCallback"] is None


def test_batch_parser_rejects_non_hex_content_identity(tmp_path):
    service, spec, _ = _service(tmp_path)
    document = DeepAnalysisBatchSpecV1.build(
        service.campaign_store,
        spec.sha256,
        funnel=service.funnel,
        profile=service.profile,
    ).to_dict()
    document["rulesHash"] = "z" * 64
    try:
        DeepAnalysisBatchSpecV1.from_dict(document)
    except ValueError as exc:
        assert "content identities" in str(exc)
    else:
        raise AssertionError("non-hex content identity was accepted")


def test_typed_deep_analysis_api_submits_reads_and_cancels(tmp_path, monkeypatch):
    from service_test_client import OperatorClient as TestClient
    from baloot_platform.service import api

    service, spec, _ = _service(tmp_path)
    monkeypatch.setattr(api, "FOUNDRY_DEEP_ANALYSIS", service)
    with TestClient(api.app) as client:
        created = client.post(
            "/api/v2/analysis-batches", json={"campaignHash": spec.sha256}
        )
        assert created.status_code == 202, created.text
        job_id = created.json()["jobId"]
        fetched = client.get(f"/api/v2/analysis-batches/{job_id}")
        assert fetched.status_code == 200
        assert fetched.json()["request"]["campaignHash"] == spec.sha256
        cancelled = client.post(f"/api/v2/analysis-batches/{job_id}:cancel")
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"


def test_typed_candidate_catalog_api_lists_and_prepares(monkeypatch):
    from service_test_client import OperatorClient as TestClient
    from baloot_platform.service import api

    candidate_hash = "a" * 64
    artifact = {
        "ownerKind": "render_bundle_v1",
        "ownerHash": "b" * 64,
        "uri": "cas://sha256/" + "c" * 64,
        "sha256": "c" * 64,
        "bytes": 10,
        "mediaType": "application/vnd.balootx.render-bundle-v1+json",
        "name": "bundle.json",
    }
    candidate = {
        "candidateHash": candidate_hash,
        "shortCode": "AAAAAAAA",
        "playReady": True,
    }
    service = SimpleNamespace(
        list_candidates=lambda job_id: {
            "schemaVersion": 1,
            "jobId": job_id,
            "campaignHash": "d" * 64,
            "count": 1,
            "items": [candidate],
        },
        prepare_actor_candidate=lambda job_id, candidate_id, reveal_policy, export_preset: {
            "schemaVersion": 1,
            "exportPreset": export_preset,
            "candidate": candidate,
            "locator": {"code": "BX-TEST", "revealPolicy": reveal_policy},
            "bundleArtifact": artifact,
            "basename": "balootx-bx-test",
        },
    )
    monkeypatch.setattr(api, "FOUNDRY_ANALYSIS_CANDIDATES", service)
    with TestClient(api.app) as client:
        listed = client.get("/api/v2/analysis-batches/analysis-1/candidates")
        assert listed.status_code == 200, listed.text
        assert listed.json()["items"][0]["shortCode"] == "AAAAAAAA"
        prepared = client.post(
            f"/api/v2/analysis-batches/analysis-1/candidates/{candidate_hash}:prepare",
            json={
                "revealPolicy": "after_completion",
                "exportPreset": "hooked_flagship",
            },
        )
        assert prepared.status_code == 200, prepared.text
        assert prepared.json()["locator"]["revealPolicy"] == "after_completion"
        assert prepared.json()["exportPreset"] == "hooked_flagship"
        invalid = client.post(
            "/api/v2/analysis-batches/analysis-1/candidates/not-a-hash:prepare",
            json={},
        )
        assert invalid.status_code == 422


def test_typed_finalist_validation_api_requires_idempotency_and_is_bounded(
    monkeypatch,
):
    from service_test_client import OperatorClient as TestClient
    from baloot_platform.service import api

    candidate_hash = "a" * 64
    job = {
        "jobId": "FINALIST-1",
        "tenantId": "default",
        "kind": "foundry_finalist_validation_v1",
        "status": "queued",
        "priority": 100,
        "progress": 0.0,
        "request": {
            "schemaVersion": 1,
            "analysisJobId": "analysis-1",
            "candidateHashes": [candidate_hash],
        },
        "result": None,
        "error": None,
        "attempts": 0,
        "maxAttempts": 2,
        "leaseOwner": None,
        "leaseUntil": None,
        "cancelRequested": False,
        "createdAt": "2026-08-04 00:00:00",
        "updatedAt": "2026-08-04 00:00:00",
    }

    class Service:
        def __init__(self):
            self.submissions = []

        def submit(self, **value):
            self.submissions.append(value)
            return job

        def get(self, job_id):
            return job if job_id == job["jobId"] else None

        def list(self, *, status=None, limit=100):
            return [job] if status in {None, job["status"]} and limit > 0 else []

        def cancel(self, job_id):
            if job_id != job["jobId"]:
                return False
            job["status"] = "cancelled"
            job["cancelRequested"] = True
            return True

    service = Service()
    class CandidateMaterializer:
        def __init__(self):
            self.calls = []

        def prepare_best_finalist_candidate(
            self, finalist_job, *, reveal_policy, export_preset
        ):
            self.calls.append(
                (finalist_job["jobId"], reveal_policy, export_preset)
            )
            return {
                "schemaVersion": 1,
                "exportPreset": export_preset,
                "candidate": {"shortCode": "BEST0001"},
                "locator": {"code": "BEST0001"},
                "bundleArtifact": {
                    "ownerKind": "render_bundle_v1",
                    "ownerHash": "b" * 64,
                    "uri": "cas://sha256/" + "c" * 64,
                    "sha256": "c" * 64,
                    "bytes": 10,
                    "mediaType": "application/vnd.balootx.render-bundle-v1+json",
                    "name": "best.render-bundle.json",
                },
                "basename": "balootx-best0001",
            }

    materializer = CandidateMaterializer()
    monkeypatch.setattr(api, "FOUNDRY_FINALIST_VALIDATIONS", service)
    monkeypatch.setattr(api, "FOUNDRY_ANALYSIS_CANDIDATES", materializer)
    with TestClient(api.app) as client:
        missing_key = client.post(
            "/api/v2/finalist-validations",
            json={
                "analysisJobId": "analysis-1",
                "candidateHashes": [candidate_hash],
            },
        )
        assert missing_key.status_code == 428

        created = client.post(
            "/api/v2/finalist-validations",
            headers={"Idempotency-Key": "finalist-api-proof"},
            json={
                "analysisJobId": "analysis-1",
                "candidateHashes": [candidate_hash.upper()],
            },
        )
        assert created.status_code == 202, created.text
        assert service.submissions[0]["candidate_hashes"] == (candidate_hash,)
        fetched = client.get("/api/v2/finalist-validations/FINALIST-1")
        assert fetched.status_code == 200
        listed = client.get("/api/v2/finalist-validations", params={"limit": 20})
        assert listed.status_code == 200
        assert listed.json()["items"][0]["jobId"] == "FINALIST-1"
        prepared = client.post(
            "/api/v2/finalist-validations/FINALIST-1:prepare-best",
            json={"revealPolicy": "after_commit"},
        )
        assert prepared.status_code == 200, prepared.text
        assert prepared.json()["locator"]["code"] == "BEST0001"
        assert materializer.calls == [
            ("FINALIST-1", "after_commit", "decision_clip")
        ]
        cancelled = client.post(
            "/api/v2/finalist-validations/FINALIST-1:cancel"
        )
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"

        invalid = client.post(
            "/api/v2/finalist-validations",
            headers={"Idempotency-Key": "invalid-finalist-api-proof"},
            json={
                "analysisJobId": "analysis-1",
                "candidateHashes": ["not-a-hash"],
            },
        )
        assert invalid.status_code == 422
