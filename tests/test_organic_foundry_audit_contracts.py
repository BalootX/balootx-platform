from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pytest

from baloot_platform.challenge.campaign import SimulationCampaignSpecV1
from baloot_platform.challenge.campaign_control import (
    ClaimedSimulationShardV1,
    SQLiteCampaignControlStore,
)
from baloot_platform.challenge.simulation import NaturalLaffaSimulatorV1
from baloot_platform.research.audit_trace import (
    CompositeResearchAuditTraceRepositoryV1,
    ResearchAuditTraceRecordV1,
    ResearchAuditTraceRepositoryV1,
)
from baloot_platform.service.laffa_library import LaffaLibraryServiceV1
from baloot_platform.storage.artifact_store import (
    LocalContentAddressedArtifactStore,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _record_hash(trace_hash: str, source: Mapping[str, Any]) -> str:
    payload = {
        "schemaVersion": 1,
        "traceHash": trace_hash,
        "source": dict(source),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class _ReplayFixture:
    simulator: NaturalLaffaSimulatorV1
    trace: Any


@dataclass(frozen=True, slots=True)
class _RecordTemplate:
    replay: _ReplayFixture
    record: ResearchAuditTraceRecordV1
    artifacts: LocalContentAddressedArtifactStore


@pytest.fixture(scope="module")
def replay_fixture(root: Path) -> _ReplayFixture:
    simulator = NaturalLaffaSimulatorV1.from_root(root)
    simulation = simulator.simulate(1)
    assert simulation.publishable_laffa is True
    return _ReplayFixture(simulator=simulator, trace=simulation.final_trace)


def _campaign(
    replay: _ReplayFixture,
    *,
    seed_start: int,
    seed_count: int,
    seeds_per_shard: int,
) -> SimulationCampaignSpecV1:
    simulator = replay.simulator
    return SimulationCampaignSpecV1.build(
        rules_hash=simulator.rules.sha256,
        bundle_hash=simulator.bundle_hash,
        engine_version=simulator.engine_version,
        analysis_profile_hash="3" * 64,
        perspective_profile_hash="4" * 64,
        selection_profile_hash="5" * 64,
        agent_release=simulator.agent_release,
        seat_agents=simulator.equal_seat_agents,
        lanes=("organic",),
        lane_profile_hashes={"organic": "6" * 64},
        seed_start=seed_start,
        seed_count=seed_count,
        seeds_per_shard=seeds_per_shard,
    )


def _source(
    claim: ClaimedSimulationShardV1,
    *,
    seed: int,
    candidate_ordinal: int = 0,
) -> dict[str, Any]:
    return {
        "kind": "organic_foundry_trace",
        "runCode": claim.spec.campaign_hash,
        "runHash": claim.spec.campaign_hash,
        "recipeCode": "organic-screening-v1",
        "recipeHash": claim.spec.lane_profile_hash,
        "shardId": claim.spec.input_hash,
        "shardHash": claim.spec.input_hash,
        "shardIndex": claim.spec.shard_index,
        "seed": seed,
        "candidateHash": _digest(
            f"{claim.spec.campaign_hash}:{claim.spec.input_hash}:"
            f"{seed}:{candidate_ordinal}"
        ),
        "candidateOrdinal": candidate_ordinal,
        "generationLane": claim.spec.lane,
        "attemptCount": claim.attempt,
        "fencingToken": claim.fencing_token,
        "workerId": claim.worker_id,
        "traceRole": "scored_final",
    }


def _clone_record(
    template: ResearchAuditTraceRecordV1,
    claim: ClaimedSimulationShardV1,
    *,
    seed: int,
    candidate_ordinal: int = 0,
) -> dict[str, Any]:
    source = _source(
        claim,
        seed=seed,
        candidate_ordinal=candidate_ordinal,
    )
    record_hash = _record_hash(template.trace_hash, source)
    return ResearchAuditTraceRecordV1.build(
        {
            **template.to_dict(),
            "recordHash": record_hash,
            "auditTraceCode": f"LAF-AUD-{record_hash[:10].upper()}",
            "source": source,
            "attemptStatus": "running",
            "createdAt": None,
            "updatedAt": None,
        }
    ).to_dict()


def _create_control(
    path: Path,
    replay: _ReplayFixture,
    *,
    seed_start: int,
    seed_count: int,
    seeds_per_shard: int,
) -> tuple[SQLiteCampaignControlStore, SimulationCampaignSpecV1]:
    campaign = _campaign(
        replay,
        seed_start=seed_start,
        seed_count=seed_count,
        seeds_per_shard=seeds_per_shard,
    )
    control = SQLiteCampaignControlStore(path, maximum_attempts=3)
    control.create_campaign(campaign)
    return control, campaign


def _claim_all(
    control: SQLiteCampaignControlStore,
    campaign: SimulationCampaignSpecV1,
    *,
    worker_count: int,
) -> tuple[ClaimedSimulationShardV1, ...]:
    claims: list[ClaimedSimulationShardV1] = []
    for index in range(len(campaign.shards())):
        claim = control.claim_shard(
            f"layout-{worker_count}-worker-{index % worker_count}",
            lease_seconds=600,
            campaign_hash=campaign.sha256,
        )
        assert claim is not None
        claims.append(claim)
    assert (
        control.claim_shard(
            "unexpected-worker",
            lease_seconds=600,
            campaign_hash=campaign.sha256,
        )
        is None
    )
    return tuple(claims)


@pytest.fixture(scope="module")
def record_template(
    tmp_path_factory: pytest.TempPathFactory,
    replay_fixture: _ReplayFixture,
) -> _RecordTemplate:
    base = tmp_path_factory.mktemp("organic-audit-record-template")
    control, campaign = _create_control(
        base / "campaign.sqlite3",
        replay_fixture,
        seed_start=1,
        seed_count=1,
        seeds_per_shard=1,
    )
    claim = control.claim_shard(
        "template-worker",
        lease_seconds=600,
        campaign_hash=campaign.sha256,
    )
    assert claim is not None
    artifacts = LocalContentAddressedArtifactStore(base / "artifacts")
    repository = ResearchAuditTraceRepositoryV1(
        store=control,
        artifact_store=artifacts,
    )
    record = repository.retain(
        replay_fixture.trace,
        rules=replay_fixture.simulator.rules,
        source=_source(claim, seed=claim.spec.seed_start),
        research_configuration=None,
    )
    return _RecordTemplate(
        replay=replay_fixture,
        record=record,
        artifacts=artifacts,
    )


def _payload_files(artifacts: LocalContentAddressedArtifactStore) -> list[Path]:
    return sorted(
        path
        for path in (artifacts.root / "sha256").glob("*/*/*")
        if path.is_file() and len(path.name) == 64
    )


def test_same_trace_occurrences_share_one_cas_object(
    tmp_path: Path,
    replay_fixture: _ReplayFixture,
) -> None:
    control, campaign = _create_control(
        tmp_path / "dedupe.sqlite3",
        replay_fixture,
        seed_start=100,
        seed_count=2,
        seeds_per_shard=2,
    )
    claim = control.claim_shard(
        "dedupe-worker",
        lease_seconds=600,
        campaign_hash=campaign.sha256,
    )
    assert claim is not None
    artifacts = LocalContentAddressedArtifactStore(tmp_path / "dedupe-artifacts")
    repository = ResearchAuditTraceRepositoryV1(
        store=control,
        artifact_store=artifacts,
    )

    first = repository.retain(
        replay_fixture.trace,
        rules=replay_fixture.simulator.rules,
        source=_source(claim, seed=100),
        research_configuration=None,
    )
    second = repository.retain(
        replay_fixture.trace,
        rules=replay_fixture.simulator.rules,
        source=_source(claim, seed=101),
        research_configuration=None,
    )

    assert first.record_hash != second.record_hash
    assert first.trace_hash == second.trace_hash
    assert first.complete_trace == second.complete_trace
    assert control.list_audit_traces(run_code=campaign.sha256)["total"] == 2
    assert _payload_files(artifacts) == [
        artifacts.materialize(first.complete_trace["uri"])
    ]
    assert control.audit_trace_storage_metrics(campaign.sha256) == {
        "schemaVersion": 1,
        "campaignHash": campaign.sha256,
        "occurrenceCount": 2,
        "uniqueTraceCount": 1,
        "uniqueArtifactCount": 1,
        "logicalBytes": 2 * int(first.complete_trace["bytes"]),
        "uniqueBytes": int(first.complete_trace["bytes"]),
    }


def test_stale_fence_cannot_index_trace_after_cas_write(
    tmp_path: Path,
    replay_fixture: _ReplayFixture,
) -> None:
    control, campaign = _create_control(
        tmp_path / "stale.sqlite3",
        replay_fixture,
        seed_start=200,
        seed_count=1,
        seeds_per_shard=1,
    )
    expired = control.claim_shard(
        "expired-worker",
        lease_seconds=1,
        campaign_hash=campaign.sha256,
        now=time.time() - 60,
    )
    assert expired is not None
    current = control.claim_shard(
        "current-worker",
        lease_seconds=600,
        campaign_hash=campaign.sha256,
    )
    assert current is not None
    assert current.spec.input_hash == expired.spec.input_hash
    assert current.fencing_token > expired.fencing_token
    artifacts = LocalContentAddressedArtifactStore(tmp_path / "stale-artifacts")
    repository = ResearchAuditTraceRepositoryV1(
        store=control,
        artifact_store=artifacts,
    )

    with pytest.raises((RuntimeError, ValueError), match="stale|identity"):
        repository.retain(
            replay_fixture.trace,
            rules=replay_fixture.simulator.rules,
            source=_source(expired, seed=200),
            research_configuration=None,
        )
    assert control.list_audit_traces(run_code=campaign.sha256)["total"] == 0
    assert len(_payload_files(artifacts)) == 1

    admitted = repository.retain(
        replay_fixture.trace,
        rules=replay_fixture.simulator.rules,
        source=_source(current, seed=200),
        research_configuration=None,
    )
    assert control.list_audit_traces(run_code=campaign.sha256)["items"] == [
        admitted.to_dict()
    ]
    assert len(_payload_files(artifacts)) == 1


def _layout_evidence(
    path: Path,
    template: _RecordTemplate,
    *,
    worker_count: int,
    seeds_per_shard: int,
) -> tuple[str, tuple[tuple[Any, ...], ...], frozenset[str], dict[str, Any]]:
    control, campaign = _create_control(
        path,
        template.replay,
        seed_start=300,
        seed_count=8,
        seeds_per_shard=seeds_per_shard,
    )
    claims = _claim_all(control, campaign, worker_count=worker_count)
    for claim in claims:
        for seed in range(claim.spec.seed_start, claim.spec.seed_end_exclusive):
            payload = _clone_record(template.record, claim, seed=seed)
            control.put_audit_trace(
                payload,
                shard_id=claim.spec.input_hash,
                worker_id=claim.worker_id,
                fencing_token=claim.fencing_token,
            )
    page = control.list_audit_traces(run_code=campaign.sha256, limit=500)
    normalized = tuple(
        sorted(
            (
                int(item["source"]["seed"]),
                item["traceHash"],
                item["completeTrace"]["sha256"],
                item["disposition"],
                item["status"],
            )
            for item in page["items"]
        )
    )
    workers = frozenset(item["source"]["workerId"] for item in page["items"])
    metrics = control.audit_trace_storage_metrics(campaign.sha256)
    metrics.pop("campaignHash")
    return campaign.sha256, normalized, workers, metrics


def test_occurrences_and_storage_are_deterministic_with_one_two_four_workers(
    tmp_path: Path,
    record_template: _RecordTemplate,
) -> None:
    evidence = (
        _layout_evidence(
            tmp_path / "layout-1.sqlite3",
            record_template,
            worker_count=1,
            seeds_per_shard=8,
        ),
        _layout_evidence(
            tmp_path / "layout-2.sqlite3",
            record_template,
            worker_count=2,
            seeds_per_shard=4,
        ),
        _layout_evidence(
            tmp_path / "layout-4.sqlite3",
            record_template,
            worker_count=4,
            seeds_per_shard=2,
        ),
    )

    assert len({item[0] for item in evidence}) == 3
    assert all(item[1] == evidence[0][1] for item in evidence)
    assert [len(item[2]) for item in evidence] == [1, 2, 4]
    assert all(item[3] == evidence[0][3] for item in evidence)
    assert evidence[0][3] == {
        "schemaVersion": 1,
        "occurrenceCount": 8,
        "uniqueTraceCount": 1,
        "uniqueArtifactCount": 1,
        "logicalBytes": 8 * int(record_template.record.complete_trace["bytes"]),
        "uniqueBytes": int(record_template.record.complete_trace["bytes"]),
    }


def test_thousand_occurrence_storage_metrics_use_one_repeated_trace_fixture(
    tmp_path: Path,
    record_template: _RecordTemplate,
) -> None:
    control, campaign = _create_control(
        tmp_path / "thousand.sqlite3",
        record_template.replay,
        seed_start=1_000,
        seed_count=1_000,
        seeds_per_shard=1_000,
    )
    claim = control.claim_shard(
        "metrics-worker",
        lease_seconds=600,
        campaign_hash=campaign.sha256,
    )
    assert claim is not None

    for seed in range(claim.spec.seed_start, claim.spec.seed_end_exclusive):
        payload = _clone_record(record_template.record, claim, seed=seed)
        control.put_audit_trace(
            payload,
            shard_id=claim.spec.input_hash,
            worker_id=claim.worker_id,
            fencing_token=claim.fencing_token,
        )

    byte_count = int(record_template.record.complete_trace["bytes"])
    assert control.list_audit_traces(
        run_code=campaign.sha256,
        limit=1,
    )["total"] == 1_000
    assert control.audit_trace_storage_metrics(campaign.sha256) == {
        "schemaVersion": 1,
        "campaignHash": campaign.sha256,
        "occurrenceCount": 1_000,
        "uniqueTraceCount": 1,
        "uniqueArtifactCount": 1,
        "logicalBytes": 1_000 * byte_count,
        "uniqueBytes": byte_count,
    }


class _CompositeResearchAdapter:
    def __init__(
        self,
        repository: CompositeResearchAuditTraceRepositoryV1,
        artifact_store: LocalContentAddressedArtifactStore,
    ) -> None:
        self.repository = repository
        self.artifact_store = artifact_store

    def list_results(
        self,
        *,
        sort: str,
        offset: int,
        limit: int,
    ) -> dict[str, Any]:
        assert sort == "newest"
        return {"items": [], "offset": offset, "limit": limit, "nextOffset": None}

    def list_audit_traces(self, **filters: Any) -> dict[str, Any]:
        return self.repository.list_records(**filters)

    def get_audit_trace(self, identifier: str) -> dict[str, Any] | None:
        record = self.repository.get_record(identifier)
        return None if record is None else record.to_dict()

    def load_audit_trace(self, identifier: str) -> Any:
        return self.repository.load_trace(identifier)


class _EmptyLearningStore:
    def list_games(self, **_filters: Any) -> list[dict[str, Any]]:
        return []

    def list_video_records(self, **_filters: Any) -> list[dict[str, Any]]:
        return []


def test_composite_library_deduplicates_replicas_not_source_occurrences(
    tmp_path: Path,
    record_template: _RecordTemplate,
) -> None:
    primary_store, campaign = _create_control(
        tmp_path / "primary.sqlite3",
        record_template.replay,
        seed_start=500,
        seed_count=2,
        seeds_per_shard=2,
    )
    secondary_store, replicated_campaign = _create_control(
        tmp_path / "secondary.sqlite3",
        record_template.replay,
        seed_start=500,
        seed_count=2,
        seeds_per_shard=2,
    )
    assert replicated_campaign.sha256 == campaign.sha256
    primary_claim = primary_store.claim_shard(
        "replica-worker",
        lease_seconds=600,
        campaign_hash=campaign.sha256,
    )
    secondary_claim = secondary_store.claim_shard(
        "replica-worker",
        lease_seconds=600,
        campaign_hash=campaign.sha256,
    )
    assert primary_claim is not None and secondary_claim is not None

    first = _clone_record(record_template.record, primary_claim, seed=500)
    replica = _clone_record(record_template.record, secondary_claim, seed=500)
    second = _clone_record(record_template.record, secondary_claim, seed=501)
    assert replica["recordHash"] == first["recordHash"]
    for store, claim, payload in (
        (primary_store, primary_claim, first),
        (secondary_store, secondary_claim, replica),
        (secondary_store, secondary_claim, second),
    ):
        store.put_audit_trace(
            payload,
            shard_id=claim.spec.input_hash,
            worker_id=claim.worker_id,
            fencing_token=claim.fencing_token,
        )

    primary = ResearchAuditTraceRepositoryV1(
        store=primary_store,
        artifact_store=record_template.artifacts,
    )
    secondary = ResearchAuditTraceRepositoryV1(
        store=secondary_store,
        artifact_store=record_template.artifacts,
    )
    composite = CompositeResearchAuditTraceRepositoryV1(
        primary,
        secondary_repositories=(secondary,),
    )
    page = composite.list_records(run_code=campaign.sha256)
    assert page["total"] == 2
    assert {item["recordHash"] for item in page["items"]} == {
        first["recordHash"],
        second["recordHash"],
    }
    located = composite.find_record(
        record_template.record.trace_hash,
        run_code=campaign.sha256,
    )
    assert located is not None
    assert located.record_hash in {first["recordHash"], second["recordHash"]}
    assert composite.load_trace(second["recordHash"]).sha256 == second["traceHash"]

    library = LaffaLibraryServiceV1(
        learning_store=_EmptyLearningStore(),
        research=_CompositeResearchAdapter(composite, record_template.artifacts),
        foundry_catalogs=None,
        render_batches=None,
    )
    response = library.list(source_kind="research_audit_trace", limit=10)
    assert response["total"] == 1
    item = response["items"][0]
    assert item["traceHash"] == record_template.record.trace_hash
    assert item["replayValid"] is True
    assert item["auditOnly"] is True
    assert item["publicationEligible"] is False
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
