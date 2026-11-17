from __future__ import annotations

import hashlib
import inspect

import pytest

from baloot_platform.challenge.campaign import (
    SimulationCampaignSpecV1,
    SimulationShardResultV1,
)
from baloot_platform.challenge.campaign_control import SQLiteCampaignControlStore
from baloot_platform.challenge.campaign_control import PostgresCampaignControlStore
from baloot_platform.challenge.simulation import NaturalLaffaSimulatorV1
from baloot_platform.research.audit_trace import ResearchAuditTraceRepositoryV1
from baloot_platform.storage.artifact_store import LocalContentAddressedArtifactStore


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_completed_campaign_attempt_accepts_and_finalizes_linked_finalist_trace(
    root, tmp_path
):
    simulator = NaturalLaffaSimulatorV1.from_root(root)
    simulation = simulator.simulate(1)
    campaign = SimulationCampaignSpecV1.build(
        rules_hash=simulator.rules.sha256,
        bundle_hash=simulator.bundle_hash,
        engine_version=simulator.engine_version,
        analysis_profile_hash=_digest("analysis"),
        perspective_profile_hash=_digest("perspective"),
        selection_profile_hash=_digest("selection"),
        agent_release=simulator.agent_release,
        seat_agents=simulator.equal_seat_agents,
        lanes=("organic",),
        lane_profile_hashes={"organic": _digest("organic-screening")},
        seed_start=1,
        seed_count=1,
        seeds_per_shard=1,
    )
    store = SQLiteCampaignControlStore(
        tmp_path / "campaign.sqlite3", maximum_attempts=2
    )
    store.create_campaign(campaign)
    claim = store.claim_shard(
        "campaign-worker",
        lease_seconds=600,
        campaign_hash=campaign.sha256,
    )
    assert claim is not None
    repository = ResearchAuditTraceRepositoryV1(
        store=store,
        artifact_store=LocalContentAddressedArtifactStore(tmp_path / "artifacts"),
    )
    parent_source = {
        "kind": "organic_foundry_trace",
        "runCode": campaign.sha256,
        "runHash": campaign.sha256,
        "recipeCode": "organic-screening-v1",
        "recipeHash": claim.spec.lane_profile_hash,
        "shardId": claim.spec.input_hash,
        "shardHash": claim.spec.input_hash,
        "shardIndex": claim.spec.shard_index,
        "seed": 1,
        "candidateHash": _digest("parent-candidate"),
        "candidateOrdinal": 0,
        "generationLane": "organic",
        "attemptCount": claim.attempt,
        "fencingToken": claim.fencing_token,
        "workerId": claim.worker_id,
        "traceRole": "scored_final",
    }
    with pytest.raises(ValueError, match="seed is outside"):
        repository.retain(
            simulation.final_trace,
            rules=simulator.rules,
            source={
                **parent_source,
                "seed": claim.spec.seed_end_exclusive,
                "candidateHash": _digest("outside-shard-candidate"),
            },
            research_configuration=None,
        )
    assert store.list_audit_traces(run_code=campaign.sha256)["total"] == 0
    parent = repository.retain(
        simulation.final_trace,
        rules=simulator.rules,
        source=parent_source,
        research_configuration=None,
    )
    result = SimulationShardResultV1.build(
        claim.spec,
        processed_seeds=(1,),
        trace_hashes=(parent.trace_hash,),
        candidate_hashes=(parent.source["candidateHash"],),
        coverage_cells=(),
        completion_status="completed",
    )
    store.commit_result(
        result,
        worker_id=claim.worker_id,
        fencing_token=claim.fencing_token,
    )
    parent = repository.finalize(
        parent.record_hash,
        disposition="rejected",
        failure_categories=("organic_screening_nonselected",),
    )
    assert parent.attempt_status == "completed"

    finalist_profile_hash = _digest("finalist-profile")
    finalist = repository.retain(
        simulation.final_trace,
        rules=simulator.rules,
        source={
            **parent_source,
            "kind": "organic_finalist_trace",
            "recipeCode": "organic-finalist-validation-v1",
            "recipeHash": finalist_profile_hash,
            "candidateHash": _digest("finalist-candidate"),
            "candidateOrdinal": 1,
            "generationLane": "organic_finalist",
            "traceRole": "finalist_descendant",
            "parentRecordHash": parent.record_hash,
            "parentTraceHash": parent.trace_hash,
            "stageProfileHash": finalist_profile_hash,
            "stageRunHash": _digest("finalist-run"),
        },
        research_configuration=None,
    )
    assert finalist.attempt_status == "completed"
    finalized = repository.finalize(
        finalist.record_hash,
        disposition="rejected",
        failure_categories=("finalist_validation_not_qualified",),
    )
    assert finalized.status == "finalized"
    assert finalized.attempt_status == "completed"
    assert finalized.source["parentRecordHash"] == parent.record_hash
    assert finalized.source["parentTraceHash"] == parent.trace_hash

    metrics = store.audit_trace_storage_metrics(campaign.sha256)
    assert metrics["occurrenceCount"] == 2
    assert metrics["uniqueTraceCount"] == 1
    assert metrics["uniqueArtifactCount"] == 1
    assert metrics["logicalBytes"] == 2 * metrics["uniqueBytes"]


def test_postgres_audit_trace_contract_and_migration_match_sqlite(root):
    authored = (root / "migrations/022_foundry_audit_traces.sql").read_bytes()
    packaged = (
        root
        / "src/baloot_platform/resources/migrations/022_foundry_audit_traces.sql"
    ).read_bytes()
    assert authored == packaged
    sql = authored.decode("utf-8")
    for field in (
        "record_hash CHAR(64) PRIMARY KEY",
        "audit_trace_code TEXT NOT NULL UNIQUE",
        "trace_hash CHAR(64) NOT NULL",
        "artifact_sha256 CHAR(64) NOT NULL",
        "source_occurrence_hash CHAR(64) NOT NULL UNIQUE",
        "campaign_hash CHAR(64) NOT NULL",
        "spec_hash CHAR(64) NOT NULL",
        "attempt INTEGER NOT NULL",
        "fencing_token BIGINT NOT NULL",
        "candidate_hash CHAR(64) NOT NULL",
        "record_json JSONB NOT NULL",
        "admitted BOOLEAN NOT NULL DEFAULT TRUE",
        "ON DELETE RESTRICT",
    ):
        assert field in sql
    assert "UNIQUE(trace_hash" not in sql
    assert "UNIQUE(artifact_sha256" not in sql

    methods = (
        "put_audit_trace",
        "update_audit_trace",
        "get_audit_trace",
        "list_audit_traces",
        "audit_trace_storage_metrics",
    )
    for name in methods:
        sqlite_parameters = tuple(
            (item.name, item.kind)
            for item in inspect.signature(
                getattr(SQLiteCampaignControlStore, name)
            ).parameters.values()
        )
        postgres_parameters = tuple(
            (item.name, item.kind)
            for item in inspect.signature(
                getattr(PostgresCampaignControlStore, name)
            ).parameters.values()
        )
        assert postgres_parameters == sqlite_parameters
    postgres_put = inspect.getsource(PostgresCampaignControlStore.put_audit_trace)
    postgres_update = inspect.getsource(
        PostgresCampaignControlStore.update_audit_trace
    )
    assert "pg_advisory_xact_lock" in postgres_put
    assert "FOR UPDATE" in postgres_put
    assert "ON CONFLICT DO NOTHING RETURNING *" in postgres_put
    assert "insertion=True" in postgres_put
    assert "insertion=False" in postgres_update
    for adapter in (SQLiteCampaignControlStore, PostgresCampaignControlStore):
        source = inspect.getsource(adapter._validate_audit_source_locked)
        assert "_validate_campaign_audit_source(" in source
