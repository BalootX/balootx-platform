from __future__ import annotations

import hashlib

from baloot_platform.challenge.campaign import (
    SimulationCampaignSpecV1,
    SimulationShardResultV1,
)
from baloot_platform.challenge.campaign_proof import (
    CampaignEquivalenceProofV1,
    CampaignExecutionEvidenceV1,
    CampaignProofProfileV1,
    CampaignProofServiceV1,
)
from baloot_platform.challenge.catalog_control import (
    FoundryCatalogServiceV1,
    SQLiteFoundryCatalogStoreV1,
)
from baloot_platform.challenge.campaign_control import SQLiteCampaignControlStore
from baloot_platform.storage.artifact_store import LocalContentAddressedArtifactStore


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _campaign(seeds_per_shard: int) -> SimulationCampaignSpecV1:
    return SimulationCampaignSpecV1.build(
        rules_hash="1" * 64,
        bundle_hash="2" * 64,
        engine_version="proof-test",
        analysis_profile_hash="3" * 64,
        perspective_profile_hash="4" * 64,
        selection_profile_hash="5" * 64,
        agent_release="equal-expert-test",
        seat_agents=("equal-expert-test",) * 4,
        lanes=("organic",),
        lane_profile_hashes={"organic": "6" * 64},
        seed_start=100,
        seed_count=6,
        seeds_per_shard=seeds_per_shard,
    )


def _results(campaign: SimulationCampaignSpecV1, *, changed_seed: int | None = None):
    values = []
    for shard in campaign.shards():
        seeds = tuple(range(shard.seed_start, shard.seed_end_exclusive))
        traces = tuple(
            _digest(f"trace-{seed}-changed" if seed == changed_seed else f"trace-{seed}")
            for seed in seeds
        )
        values.append(SimulationShardResultV1.build(
            shard,
            processed_seeds=seeds,
            trace_hashes=traces,
            candidate_hashes=tuple(_digest(f"candidate-{seed}") for seed in seeds),
            coverage_cells=tuple(f"trick:{seed % 3}" for seed in seeds),
            completion_status="completed",
        ))
    return tuple(values)


def _shard_result(shard):
    seeds = tuple(range(shard.seed_start, shard.seed_end_exclusive))
    return SimulationShardResultV1.build(
        shard,
        processed_seeds=seeds,
        trace_hashes=tuple(_digest(f"trace-{seed}") for seed in seeds),
        candidate_hashes=tuple(_digest(f"candidate-{seed}") for seed in seeds),
        coverage_cells=tuple(f"trick:{seed % 3}" for seed in seeds),
        completion_status="completed",
    )


def _profile(*, minimum_rounds: int = 6) -> CampaignProofProfileV1:
    return CampaignProofProfileV1.build({
        "schemaVersion": 1,
        "id": "campaign-proof-test",
        "revision": "1",
        "minimumOrganicRounds": minimum_rounds,
        "requiredWorkerCounts": [1, 4],
        "requiredFailureSchedules": ["none", "retry_reclaim"],
        "requireAttemptLedger": True,
    })


def _execution(
    campaign: SimulationCampaignSpecV1,
    *,
    workers: int,
    schedule: str,
    attempts: int = 1,
    changed_seed: int | None = None,
) -> CampaignExecutionEvidenceV1:
    return CampaignExecutionEvidenceV1.build(
        campaign,
        _results(campaign, changed_seed=changed_seed),
        declared_worker_count=workers,
        failure_schedule=schedule,
        shard_attempt_counts={item.input_hash: attempts for item in campaign.shards()},
        attested_worker_ids=tuple(f"worker-{index}" for index in range(workers)),
        attempt_ledger_hash=_digest(f"ledger-{workers}-{schedule}"),
        failure_schedule_attested=True,
    )


def test_partition_independent_campaign_proof_is_qualified_and_order_independent():
    serial = _execution(_campaign(6), workers=1, schedule="none")
    partitioned = _execution(
        _campaign(2), workers=4, schedule="retry_reclaim", attempts=2
    )
    assert serial.campaign_hash != partitioned.campaign_hash
    assert serial.workload.sha256 == partitioned.workload.sha256
    assert serial.normalized_evidence_hash == partitioned.normalized_evidence_hash
    assert dict(serial.coverage_by_lane) == {"organic": ((100, 106),)}

    first = CampaignEquivalenceProofV1.build((serial, partitioned), _profile())
    second = CampaignEquivalenceProofV1.build((partitioned, serial), _profile())
    assert first.sha256 == second.sha256
    assert first.status == "qualified"
    assert not first.unmet_qualification_gates


def test_small_campaign_stays_pilot_only_below_configured_round_profile():
    serial = _execution(_campaign(6), workers=1, schedule="none")
    partitioned = _execution(
        _campaign(2), workers=4, schedule="retry_reclaim", attempts=2
    )
    proof = CampaignEquivalenceProofV1.build(
        (serial, partitioned), _profile(minimum_rounds=10_000)
    )
    assert proof.status == "pilot_only"
    assert "minimum_organic_rounds" in proof.unmet_qualification_gates


def test_campaign_proof_rejects_any_partition_output_drift():
    serial = _execution(_campaign(6), workers=1, schedule="none")
    changed = _execution(
        _campaign(2),
        workers=4,
        schedule="retry_reclaim",
        changed_seed=103,
    )
    try:
        CampaignEquivalenceProofV1.build((serial, changed), _profile())
    except ValueError as exc:
        assert "outputs differ" in str(exc)
    else:
        raise AssertionError("partition output drift entered an equivalence proof")


class _CampaignStore:
    def __init__(self, campaigns):
        self.campaigns = {item.sha256: item for item in campaigns}

    def get_campaign(self, campaign_hash):
        campaign = self.campaigns.get(campaign_hash)
        if campaign is None:
            return None
        return {
            "campaignHash": campaign.sha256,
            "spec": campaign.to_dict(),
            "status": "completed",
            "cancelRequested": False,
        }

    def list_shards(self, campaign_hash):
        campaign = self.campaigns[campaign_hash]
        results = {item.spec_hash: item for item in _results(campaign)}
        return [{
            "specHash": shard.input_hash,
            "status": "completed",
            "attempts": 2 if campaign.seeds_per_shard == 2 else 1,
            "result": results[shard.input_hash].to_dict(),
        } for shard in campaign.shards()]


def test_campaign_proof_service_persists_immutable_catalog(tmp_path):
    serial, partitioned = _campaign(6), _campaign(2)
    store = _CampaignStore((serial, partitioned))
    artifacts = LocalContentAddressedArtifactStore(tmp_path / "artifacts")
    catalogs = FoundryCatalogServiceV1(
        SQLiteFoundryCatalogStoreV1(tmp_path / "catalogs.sqlite3"),
        artifacts,
        None,
        None,
    )
    service = CampaignProofServiceV1(store, catalogs, _profile())
    request = (
        {
            "campaignHash": serial.sha256,
            "declaredWorkerCount": 1,
            "failureSchedule": "none",
        },
        {
            "campaignHash": partitioned.sha256,
            "declaredWorkerCount": 4,
            "failureSchedule": "retry_reclaim",
        },
    )
    first = service.create(request)
    second = service.create(tuple(reversed(request)))
    assert first["record"]["catalogHash"] == second["record"]["catalogHash"]
    assert first["record"]["kind"] == "campaign_equivalence_proof_v1"
    assert first["document"]["status"] == "pilot_only"
    assert "worker_attempt_attestation" in first["document"]["unmetQualificationGates"]
    assert artifacts.get(first["record"]["artifact"]["uri"])


def test_typed_campaign_equivalence_api_persists_proof(tmp_path, monkeypatch):
    from service_test_client import OperatorClient as TestClient
    from baloot_platform.service import api

    serial, partitioned = _campaign(6), _campaign(2)
    artifacts = LocalContentAddressedArtifactStore(tmp_path / "api-artifacts")
    catalogs = FoundryCatalogServiceV1(
        SQLiteFoundryCatalogStoreV1(tmp_path / "api-catalogs.sqlite3"),
        artifacts,
        None,
        None,
    )
    service = CampaignProofServiceV1(
        _CampaignStore((serial, partitioned)), catalogs, _profile()
    )
    monkeypatch.setattr(api, "FOUNDRY_CAMPAIGN_PROOFS", service)
    with TestClient(api.app) as client:
        response = client.post(
            "/api/v2/campaign-equivalence-proofs",
            json={"executions": [
                {
                    "campaignHash": serial.sha256,
                    "declaredWorkerCount": 1,
                    "failureSchedule": "none",
                },
                {
                    "campaignHash": partitioned.sha256,
                    "declaredWorkerCount": 4,
                    "failureSchedule": "retry_reclaim",
                },
            ]},
        )
        assert response.status_code == 201, response.text
        assert response.json()["record"]["kind"] == "campaign_equivalence_proof_v1"
        assert response.json()["document"]["status"] == "pilot_only"


def test_sqlite_attempt_ledger_attests_workers_retries_and_qualified_proof(
    tmp_path, monkeypatch
):
    store = SQLiteCampaignControlStore(tmp_path / "campaigns.sqlite3", maximum_attempts=3)
    serial = _campaign(6)
    retry = _campaign(2)
    store.create_campaign(serial)
    serial_claim = store.claim_shard(
        "serial-worker", lease_seconds=30, campaign_hash=serial.sha256, now=10
    )
    store.commit_result(
        _shard_result(serial_claim.spec),
        worker_id="serial-worker",
        fencing_token=serial_claim.fencing_token,
        now=11,
    )

    store.create_campaign(retry)
    failed = store.claim_shard(
        "retry-worker-a", lease_seconds=30, campaign_hash=retry.sha256, now=20
    )
    store.fail_shard(
        failed.spec.input_hash,
        worker_id="retry-worker-a",
        fencing_token=failed.fencing_token,
        error="injected retry",
        retryable=True,
    )
    worker_index = 0
    while store.get_campaign(retry.sha256)["status"] != "completed":
        worker = ("retry-worker-b", "retry-worker-a")[worker_index % 2]
        worker_index += 1
        claim = store.claim_shard(
            worker, lease_seconds=30, campaign_hash=retry.sha256, now=30 + worker_index
        )
        assert claim is not None
        store.commit_result(
            _shard_result(claim.spec),
            worker_id=worker,
            fencing_token=claim.fencing_token,
            now=40 + worker_index,
        )

    profile = CampaignProofProfileV1.build({
        "schemaVersion": 1,
        "id": "attested-proof-test",
        "revision": "1",
        "minimumOrganicRounds": 6,
        "requiredWorkerCounts": [1, 2],
        "requiredFailureSchedules": ["none", "retry_reclaim"],
        "requireAttemptLedger": True,
    })
    artifacts = LocalContentAddressedArtifactStore(tmp_path / "attested-artifacts")
    catalogs = FoundryCatalogServiceV1(
        SQLiteFoundryCatalogStoreV1(tmp_path / "attested-catalogs.sqlite3"),
        artifacts,
        None,
        None,
    )
    proof = CampaignProofServiceV1(store, catalogs, profile).create((
        {
            "campaignHash": serial.sha256,
            "declaredWorkerCount": 1,
            "failureSchedule": "none",
        },
        {
            "campaignHash": retry.sha256,
            "declaredWorkerCount": 2,
            "failureSchedule": "retry_reclaim",
        },
    ))
    assert proof["document"]["status"] == "qualified"
    assert all(
        item["failureScheduleAttested"] for item in proof["document"]["executions"]
    )

    from service_test_client import OperatorClient as TestClient
    from baloot_platform.service import api

    monkeypatch.setattr(api, "FOUNDRY_CAMPAIGNS", store)
    with TestClient(api.app) as client:
        response = client.get(f"/api/v2/campaigns/{retry.sha256}/attempts")
        assert response.status_code == 200, response.text
        attempts = response.json()["attempts"]
        assert {item["terminalStatus"] for item in attempts} >= {
            "completed", "retryable_failure"
        }


def test_attempt_ledger_migration_is_packaged(root):
    source = (root / "migrations/019_foundry_attempt_ledger.sql").read_text(
        encoding="utf-8"
    )
    packaged = (
        root
        / "src/baloot_platform/resources/migrations/019_foundry_attempt_ledger.sql"
    ).read_text(encoding="utf-8")
    assert source == packaged
    assert "foundry_shard_attempts_v1" in source
    assert "PRIMARY KEY(spec_hash,fencing_token)" in source


def test_attempt_ledger_attests_out_of_order_completion(tmp_path):
    store = SQLiteCampaignControlStore(tmp_path / "out-of-order.sqlite3", maximum_attempts=3)
    campaign = _campaign(2)
    store.create_campaign(campaign)
    claims = [
        store.claim_shard(
            f"worker-{index % 2}",
            lease_seconds=200,
            campaign_hash=campaign.sha256,
            now=10 + index,
        )
        for index in range(3)
    ]
    for index, claim in enumerate(reversed(claims)):
        store.commit_result(
            _shard_result(claim.spec),
            worker_id=claim.worker_id,
            fencing_token=claim.fencing_token,
            now=100 + index,
        )
    evidence = CampaignExecutionEvidenceV1.from_store(
        store,
        campaign.sha256,
        declared_worker_count=2,
        failure_schedule="out_of_order",
    )
    assert evidence.failure_schedule_attested
    assert len(evidence.attested_worker_ids) == 2
