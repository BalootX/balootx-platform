from __future__ import annotations

import json

import pytest

from baloot_platform.challenge.campaign import SimulationCampaignSpecV1
from baloot_platform.challenge.campaign_control import SQLiteCampaignControlStore
from baloot_platform.challenge.simulation import NaturalLaffaSimulatorV1
from baloot_platform.challenge.trace_import import (
    TraceImportCatalogEntryV1,
    TraceImportCatalogV1,
    TraceImportIngestorV1,
    TraceImportPolicyV1,
    TraceImportProvenanceV1,
)
from baloot_platform.challenge.worker import ExpertFoundryWorkerV1, HistoricalFoundryWorkerV1
from baloot_platform.storage.artifact_store import LocalContentAddressedArtifactStore


def _policy():
    return TraceImportPolicyV1.build({
        "schemaVersion": 1,
        "maximumEntries": 100,
        "allowedRightsStatuses": ["cleared", "research_only"],
        "historicalVerificationStatuses": ["verified_action_order"],
        "expertVerificationStatuses": [
            "unreviewed", "verified_expert", "consensus_approved",
        ],
        "requireCompleteHiddenState": True,
    })


def _reference(store, trace):
    reference = store.put(
        json.dumps(
            trace.to_dict(trusted=True),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
        media_type="application/vnd.balootx.complete-trace-v1+json",
        name=f"{trace.sha256}.complete-trace.json",
    )
    return {"ownerKind": "complete_trace_v1", "ownerHash": trace.sha256, **reference}


def _provenance(lane, *, verification, submitter=None, rights="cleared"):
    value = {
        "schemaVersion": 1,
        "lane": lane,
        "sourceRecordId": f"source-{lane}-001",
        "sourceEvidenceHash": "d" * 64,
        "rightsStatus": rights,
        "verificationStatus": verification,
        "completeHiddenStateAttested": True,
    }
    if submitter is not None:
        value["submittedByHash"] = submitter
    return TraceImportProvenanceV1.build(value, policy=_policy())


def _ingestor(root, store, trace, lane, provenance):
    simulator = NaturalLaffaSimulatorV1.from_root(root)
    entry = TraceImportCatalogEntryV1.build(
        ordinal=0,
        trace_reference=_reference(store, trace),
        provenance=provenance,
    )
    catalog = TraceImportCatalogV1.build(lane, (entry,), policy=_policy())
    return simulator, TraceImportIngestorV1(
        store,
        simulator.rules,
        bundle_hash=simulator.bundle_hash,
        engine_version=simulator.engine_version,
        catalog=catalog,
        policy=_policy(),
    )


def _campaign(simulator, ingestor, lane):
    return SimulationCampaignSpecV1.build(
        rules_hash=simulator.rules.sha256,
        bundle_hash=simulator.bundle_hash,
        engine_version=simulator.engine_version,
        analysis_profile_hash="1" * 64,
        perspective_profile_hash="2" * 64,
        selection_profile_hash="3" * 64,
        agent_release=simulator.agent_release,
        seat_agents=simulator.equal_seat_agents,
        lanes=(lane,),
        lane_profile_hashes={lane: ingestor.catalog.sha256},
        seed_start=0,
        seed_count=1,
        seeds_per_shard=1,
    )


def test_historical_trace_import_is_exact_replay_and_review_pending(root, tmp_path):
    store = LocalContentAddressedArtifactStore(tmp_path / "artifacts")
    trace = NaturalLaffaSimulatorV1.from_root(root).simulate(7).final_trace
    simulator, ingestor = _ingestor(
        root,
        store,
        trace,
        "historical",
        _provenance("historical", verification="verified_action_order"),
    )
    first = ingestor.ingest(0)
    second = ingestor.ingest(0)
    assert first.sha256 == second.sha256
    assert first.trace_hash == trace.sha256
    assert first.review_state == "source_verified_review_pending"
    assert "publication:pending_analysis_and_review" in first.coverage_cells
    assert first.replay_event_chain_hash
    assert simulator.rules.sha256 == trace.rules_hash


def test_expert_import_requires_submitter_and_preserves_unreviewed_status(root, tmp_path):
    with pytest.raises(ValueError, match="submitter"):
        _provenance("expert", verification="unreviewed")
    store = LocalContentAddressedArtifactStore(tmp_path / "artifacts")
    trace = NaturalLaffaSimulatorV1.from_root(root).simulate(8).final_trace
    _simulator, ingestor = _ingestor(
        root,
        store,
        trace,
        "expert",
        _provenance(
            "expert",
            verification="unreviewed",
            submitter="e" * 64,
            rights="research_only",
        ),
    )
    result = ingestor.ingest(0)
    assert result.review_state == "unreviewed"
    assert "rights:research_only" in result.coverage_cells
    assert "review:unreviewed" in result.coverage_cells


@pytest.mark.parametrize(
    ("lane", "worker_type", "verification", "submitter"),
    [
        ("historical", HistoricalFoundryWorkerV1, "verified_action_order", None),
        ("expert", ExpertFoundryWorkerV1, "verified_expert", "f" * 64),
    ],
)
def test_import_workers_claim_only_their_lane_and_commit_no_candidates(
    root, tmp_path, lane, worker_type, verification, submitter
):
    store = LocalContentAddressedArtifactStore(tmp_path / f"artifacts-{lane}")
    trace = NaturalLaffaSimulatorV1.from_root(root).simulate(9).final_trace
    simulator, ingestor = _ingestor(
        root,
        store,
        trace,
        lane,
        _provenance(lane, verification=verification, submitter=submitter),
    )
    campaign = _campaign(simulator, ingestor, lane)
    control = SQLiteCampaignControlStore(tmp_path / f"{lane}.sqlite3", maximum_attempts=2)
    control.create_campaign(campaign)
    worker = worker_type(
        control,
        store,
        ingestor,
        expected_agent_release=simulator.agent_release,
        worker_id=f"{lane}-worker",
        lease_seconds=9,
        heartbeat_seconds=3,
    )
    completed = worker.run_once(campaign_hash=campaign.sha256)
    assert completed.status == "completed"
    shard = control.list_shards(campaign.sha256)[0]
    result = shard["result"]
    assert result["candidateHashes"] == []
    assert result["traceHashes"] == [trace.sha256]
    assert f"source:{lane}" in result["coverageCells"]
    assert any(
        item["ownerKind"] == "trace_import_result_v1"
        for item in result["artifactReferences"]
    )


def test_import_rejects_wrong_lane_and_artifact_identity(root, tmp_path):
    store = LocalContentAddressedArtifactStore(tmp_path / "artifacts")
    trace = NaturalLaffaSimulatorV1.from_root(root).simulate(10).final_trace
    provenance = _provenance("historical", verification="verified_action_order")
    reference = _reference(store, trace)
    with pytest.raises(ValueError, match="another lane"):
        TraceImportCatalogV1.build(
            "expert",
            (TraceImportCatalogEntryV1.build(
                ordinal=0,
                trace_reference=reference,
                provenance=provenance,
            ),),
            policy=_policy(),
        )
    bad_entry = TraceImportCatalogEntryV1.build(
        ordinal=0,
        trace_reference={**reference, "ownerHash": "0" * 64},
        provenance=provenance,
    )
    catalog = TraceImportCatalogV1.build("historical", (bad_entry,), policy=_policy())
    simulator = NaturalLaffaSimulatorV1.from_root(root)
    ingestor = TraceImportIngestorV1(
        store,
        simulator.rules,
        bundle_hash=simulator.bundle_hash,
        engine_version=simulator.engine_version,
        catalog=catalog,
        policy=_policy(),
    )
    with pytest.raises(ValueError, match="owner identity"):
        ingestor.ingest(0)
