from __future__ import annotations

import copy
import json

import pytest

from baloot_platform.agents.profiles import AgentProfile
from baloot_platform.challenge.campaign import (
    SimulationCampaignSpecV1,
    SimulationShardResultV1,
    deterministic_campaign_reduction,
)
from baloot_platform.challenge.campaign_control import SQLiteCampaignControlStore
from baloot_platform.challenge.screening import (
    OrganicScreeningProfileV1,
    OrganicShardScreeningManifestV1,
    screen_organic_trace_v1,
)
from baloot_platform.challenge.evidence_v3 import canonical_sha256
from baloot_platform.challenge.simulation import NaturalLaffaSimulatorV1
from baloot_platform.challenge.worker import OrganicFoundryWorkerV1
from baloot_platform.config import build_bundle
from baloot_platform.core.rules import RuleBook
from baloot_platform.storage.artifact_store import LocalContentAddressedArtifactStore


def _profile(**updates) -> OrganicScreeningProfileV1:
    value = {
        "schemaVersion": 1,
        "id": "screening-test",
        "revision": "1",
        "agentProfileId": "strong",
        "minimumLegalActions": 2,
        "baseRetainBasisPoints": 0,
        "maximumRetainBasisPoints": 0,
        "eliteScore": 100.0,
        "eliteRetainBasisPoints": 0,
        "rejectedAuditModulus": 1_000_000,
        "retainKabout": False,
        "retainEscalated": False,
        "compressionLevel": 9,
    }
    value.update(updates)
    return OrganicScreeningProfileV1.build(value)


def _campaign(simulator, profile, *, seeds=2, seeds_per_shard=None):
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
        lane_profile_hashes={
            "organic": (
                simulator.bundle_hash if profile is None else profile.sha256
            )
        },
        seed_start=1,
        seed_count=seeds,
        seeds_per_shard=seeds if seeds_per_shard is None else seeds_per_shard,
    )


def _always_pass_simulator(root) -> NaturalLaffaSimulatorV1:
    bundle = build_bundle(root)
    raw_rules = copy.deepcopy(bundle.rules.data)
    for key in (
        "sun_threshold",
        "hokum_threshold",
        "ashkal_partner_threshold",
        "second_sun_threshold",
        "second_hokum_threshold",
    ):
        raw_rules["bidding"]["heuristics"][key] = 999.0
    rules = RuleBook(raw_rules)
    profile = AgentProfile.from_mapping(
        "always-pass-baseline",
        {
            "skill": 1.0,
            "buying_strength": 0.0,
            "play_search_budget": 0,
            "mistake_probability": 0.0,
            "deception_probability": 0.0,
            "use_production_model": False,
            "bid_samples": 0,
        },
    )
    snapshot = {
        "schemaVersion": 1,
        "bundleHash": "b" * 64,
        "mode": "test",
        "profile": {"id": "test", "version": "1", "status": "draft"},
        "components": {
            "rules": {"sha256": rules.sha256, "payload": rules.data},
            "conventions": {
                "sha256": canonical_sha256(bundle.convention),
                "payload": bundle.convention,
            },
        },
    }
    return NaturalLaffaSimulatorV1(
        rules,
        bundle.convention,
        bundle_hash="b" * 64,
        configuration_snapshot={
            **snapshot,
            "sha256": canonical_sha256(snapshot),
        },
        agent_profile=profile,
        maximum_deal_attempts=2,
    )


def _run_campaign(root, tmp_path, profile, *, seeds, seeds_per_shard):
    simulator = NaturalLaffaSimulatorV1.from_root(root)
    campaign = _campaign(
        simulator,
        profile,
        seeds=seeds,
        seeds_per_shard=seeds_per_shard,
    )
    control = SQLiteCampaignControlStore(
        tmp_path / f"campaign-{seeds_per_shard}.sqlite3", maximum_attempts=2
    )
    control.create_campaign(campaign)
    worker = OrganicFoundryWorkerV1(
        control,
        LocalContentAddressedArtifactStore(
            tmp_path / f"artifacts-{seeds_per_shard}"
        ),
        simulator,
        worker_id=f"worker-{seeds_per_shard}",
        lease_seconds=60,
        heartbeat_seconds=10,
        screening_profile=profile,
    )
    while worker.run_once(campaign_hash=campaign.sha256).status == "completed":
        pass
    results = tuple(
        SimulationShardResultV1.from_dict(shard, row["result"])
        for shard in campaign.shards()
        for row in control.list_shards(campaign.sha256)
        if row["specHash"] == shard.input_hash
    )
    return campaign, deterministic_campaign_reduction(campaign, results)


def test_screening_profile_is_closed_and_monotonic():
    with pytest.raises(ValueError, match="unknown"):
        _profile(hiddenSwitch=True)
    with pytest.raises(ValueError, match="monotonic"):
        _profile(baseRetainBasisPoints=100, maximumRetainBasisPoints=50)


def test_runtime_uses_one_governed_intelligent_research_agent_for_every_seat(root):
    from baloot_platform.service.container import build_container

    container = build_container(root=root)
    try:
        simulator = container.registry.get("generator", "natural_laffa_generator_v1")
        assert simulator.agent_profile.id == "foundry_balanced"
        assert simulator.agent_profile.skill == 1.0
        assert simulator.agent_profile.mistake_probability == 0.0
        assert simulator.agent_profile.deception_probability == 0.0
        assert simulator.agent_profile.signal_discipline == 1.0
        assert simulator.agent_profile.play_search_mode == "paired_information_set"
        assert simulator.agent_profile.play_search_budget > 0
        assert simulator.agent_profile.play_search_samples > 0
        assert simulator.agent_profile.bid_samples > 0
        assert simulator.agent_profile.use_production_model is False
        assert simulator.agent_profile.model_weight == 0.0
        assert len(set(simulator.equal_seat_agents)) == 1
        assert (
            container.organic_screening_profile.agent_profile_id
            == simulator.agent_profile.id
        )
        trace = simulator.simulate(23).final_trace
        assert len(set(dict(trace.agent_profiles).values())) == 1
        assert trace.trusted_final_state["agentProfileHash"] == simulator.profile_hash
        assert (
            trace.trusted_final_state["configurationSnapshot"]["sha256"]
            == simulator.configuration_snapshot_hash
        )
        policy_traces = tuple(
            item["trace"]
            for item in trace.trusted_final_state["decisionPolicyTraces"]
        )
        assert not any(item.get("mistakeApplied") for item in policy_traces)
        assert not any(item.get("deceptionApplied") for item in policy_traces)
    finally:
        container.shutdown()


def test_structural_screen_is_deterministic_all_seat_and_compressible(root):
    simulator = NaturalLaffaSimulatorV1.from_root(root)
    simulation = simulator.simulate(1)
    profile = _profile()
    first = screen_organic_trace_v1(
        root_seed=1,
        attempt_trace_hashes=(item.sha256 for item in simulation.attempt_traces),
        final_trace=simulation.final_trace,
        publishable_laffa=simulation.publishable_laffa,
        profile=profile,
    )
    second = screen_organic_trace_v1(
        root_seed=1,
        attempt_trace_hashes=(item.sha256 for item in simulation.attempt_traces),
        final_trace=simulation.final_trace,
        publishable_laffa=simulation.publishable_laffa,
        profile=profile,
    )
    assert first == second
    assert first.features["actingSeatBreadth"] == 4
    assert 0.0 <= first.structural_score <= 100.0
    manifest = OrganicShardScreeningManifestV1.build("a" * 64, profile, (first,))
    decoded = manifest.decode_compressed(
        manifest.compressed_bytes(level=profile.compression_level)
    )
    assert decoded["sha256"] == manifest.sha256
    assert decoded["records"][0]["finalTraceHash"] == simulation.final_trace.sha256


def test_screened_worker_keeps_summary_only_traces_audit_only_and_out_of_candidates(
    root, tmp_path
):
    simulator = NaturalLaffaSimulatorV1.from_root(root)
    profile = _profile()
    campaign = _campaign(simulator, profile)
    control = SQLiteCampaignControlStore(
        tmp_path / "screened-campaign.sqlite3", maximum_attempts=2
    )
    control.create_campaign(campaign)
    artifacts = LocalContentAddressedArtifactStore(tmp_path / "artifacts")
    worker = OrganicFoundryWorkerV1(
        control,
        artifacts,
        simulator,
        worker_id="screened-worker",
        lease_seconds=60,
        heartbeat_seconds=10,
        screening_profile=profile,
    )
    result = worker.run_once(campaign_hash=campaign.sha256)
    assert result.status == "completed"
    shard_result = control.list_shards(campaign.sha256)[0]["result"]
    references = shard_result["artifactReferences"]
    assert shard_result["candidateHashes"] == []
    assert len(shard_result["traceHashes"]) >= 2
    assert [item["ownerKind"] for item in references] == [
        "organic_screening_manifest_v1"
    ]
    payload = artifacts.get(references[0]["uri"])
    manifest = OrganicShardScreeningManifestV1.decode_compressed(payload)
    assert manifest["recordCount"] == 2
    assert manifest["retainedCount"] == 0
    assert len(payload) < 20_000

    page = control.list_audit_traces(run_code=campaign.sha256)
    records = page["items"]
    shard_spec = campaign.shards()[0]
    assert page["total"] == len(shard_result["traceHashes"])
    assert {item["traceHash"] for item in records} == set(
        shard_result["traceHashes"]
    )
    assert all(item["auditOnly"] is True for item in records)
    assert all(item["publicationEligible"] is False for item in records)
    assert all(item["status"] == "finalized" for item in records)
    assert all(item["disposition"] == "rejected" for item in records)
    assert all(item["source"]["traceRole"] == "scored_final" for item in records)
    assert all(
        {
            "stage_organic_screening",
            "organic_screening_nonselected",
            "organic_screening_summary_only",
        }.issubset(item["failureCategories"])
        for item in records
    )
    for item in records:
        source = item["source"]
        assert set(source) == {
            "kind",
            "runCode",
            "runHash",
            "recipeCode",
            "recipeHash",
            "shardId",
            "shardHash",
            "shardIndex",
            "seed",
            "candidateHash",
            "candidateOrdinal",
            "generationLane",
            "attemptCount",
            "fencingToken",
            "workerId",
            "traceRole",
        }
        assert source["kind"] == "organic_foundry_trace"
        assert source["runCode"] == campaign.sha256
        assert source["runHash"] == campaign.sha256
        assert source["recipeCode"] == "organic-screening-v1"
        assert source["recipeHash"] == profile.sha256
        assert source["shardId"] == shard_spec.input_hash
        assert source["shardHash"] == shard_spec.input_hash
        assert source["generationLane"] == "organic"
        assert source["attemptCount"] == 1
        assert source["fencingToken"] == 1
        assert source["workerId"] == "screened-worker"
        assert source["candidateHash"] == canonical_sha256({
            "schemaVersion": 1,
            "kind": "organic_foundry_candidate_occurrence_v1",
            "campaignHash": campaign.sha256,
            "shardHash": shard_spec.input_hash,
            "seed": source["seed"],
            "candidateOrdinal": source["candidateOrdinal"],
        })
        research_configuration = item["configurationProvenance"][
            "researchConfiguration"
        ]
        assert research_configuration["campaign"]["sha256"] == campaign.sha256
        assert research_configuration["screeningProfile"]["sha256"] == (
            profile.sha256
        )
        document = json.loads(artifacts.get(item["completeTrace"]["uri"]))
        assert document["sha256"] == item["traceHash"]

    metrics = control.audit_trace_storage_metrics(campaign.sha256)
    assert metrics["occurrenceCount"] == len(records)
    assert metrics["uniqueTraceCount"] == len(records)
    assert metrics["uniqueArtifactCount"] == len(records)
    assert metrics["logicalBytes"] == metrics["uniqueBytes"]


def test_screened_worker_exposes_only_retained_final_traces_to_deep_analysis(
    root, tmp_path
):
    simulator = NaturalLaffaSimulatorV1.from_root(root)
    profile = _profile(
        baseRetainBasisPoints=10_000,
        maximumRetainBasisPoints=10_000,
        eliteRetainBasisPoints=10_000,
    )
    campaign = _campaign(simulator, profile)
    control = SQLiteCampaignControlStore(
        tmp_path / "retained-campaign.sqlite3", maximum_attempts=2
    )
    control.create_campaign(campaign)
    artifacts = LocalContentAddressedArtifactStore(tmp_path / "retained-artifacts")
    worker = OrganicFoundryWorkerV1(
        control,
        artifacts,
        simulator,
        worker_id="retaining-worker",
        lease_seconds=60,
        heartbeat_seconds=10,
        screening_profile=profile,
    )
    assert worker.run_once(campaign_hash=campaign.sha256).status == "completed"
    shard_result = control.list_shards(campaign.sha256)[0]["result"]
    complete = [
        item for item in shard_result["artifactReferences"]
        if item["ownerKind"] == "complete_trace_v1"
    ]
    assert len(complete) == 2
    assert sorted(shard_result["candidateHashes"]) == sorted(
        item["ownerHash"] for item in complete
    )
    for reference in complete:
        document = json.loads(artifacts.get(reference["uri"]))
        assert document["sha256"] == reference["ownerHash"]

    audit_records = control.list_audit_traces(run_code=campaign.sha256)["items"]
    assert {item["traceHash"] for item in audit_records} == set(
        shard_result["traceHashes"]
    )
    scored = [
        item
        for item in audit_records
        if item["source"]["traceRole"] == "scored_final"
    ]
    assert {item["traceHash"] for item in scored} == set(
        shard_result["candidateHashes"]
    )
    assert all(item["status"] == "provisional" for item in scored)
    assert all(item["disposition"] == "provisional" for item in scored)
    assert all(item["attemptStatus"] == "completed" for item in scored)
    assert all(item["auditOnly"] is True for item in scored)
    assert all(item["publicationEligible"] is False for item in scored)
    audit_by_trace = {item["traceHash"]: item for item in audit_records}
    assert all(
        audit_by_trace[item["ownerHash"]]["completeTrace"]["sha256"]
        == item["sha256"]
        for item in complete
    )


def test_screened_worker_finalizes_all_pass_attempts_without_candidates(
    root, tmp_path
):
    simulator = _always_pass_simulator(root)
    profile = _profile(agentProfileId="always-pass-baseline")
    campaign = _campaign(simulator, profile, seeds=1)
    control = SQLiteCampaignControlStore(
        tmp_path / "all-pass-campaign.sqlite3", maximum_attempts=2
    )
    control.create_campaign(campaign)
    artifacts = LocalContentAddressedArtifactStore(tmp_path / "all-pass-artifacts")
    worker = OrganicFoundryWorkerV1(
        control,
        artifacts,
        simulator,
        worker_id="all-pass-worker",
        lease_seconds=60,
        heartbeat_seconds=10,
        screening_profile=profile,
    )

    assert worker.run_once(campaign_hash=campaign.sha256).status == "completed"
    shard_result = control.list_shards(campaign.sha256)[0]["result"]
    assert shard_result["candidateHashes"] == []
    assert len(shard_result["traceHashes"]) == 2
    assert [
        item["ownerKind"] for item in shard_result["artifactReferences"]
    ] == ["organic_screening_manifest_v1"]

    records = control.list_audit_traces(run_code=campaign.sha256)["items"]
    assert {item["traceHash"] for item in records} == set(
        shard_result["traceHashes"]
    )
    assert all(
        item["source"]["traceRole"] == "all_pass_attempt" for item in records
    )
    assert all(item["status"] == "finalized" for item in records)
    assert all(item["disposition"] == "rejected" for item in records)
    assert all(item["attemptStatus"] == "completed" for item in records)
    assert all(
        {
            "stage_organic_simulation",
            "organic_all_pass_complete_trace",
        }.issubset(item["failureCategories"])
        for item in records
    )
    assert not any(
        item["traceHash"] in shard_result["candidateHashes"] for item in records
    )


def test_unscreened_worker_audit_indexes_every_trace_without_changing_outputs(
    root, tmp_path
):
    simulator = NaturalLaffaSimulatorV1.from_root(root)
    campaign = _campaign(simulator, None, seeds=1)
    control = SQLiteCampaignControlStore(
        tmp_path / "unscreened-campaign.sqlite3", maximum_attempts=2
    )
    control.create_campaign(campaign)
    artifacts = LocalContentAddressedArtifactStore(tmp_path / "unscreened-artifacts")
    worker = OrganicFoundryWorkerV1(
        control,
        artifacts,
        simulator,
        worker_id="unscreened-worker",
        lease_seconds=60,
        heartbeat_seconds=10,
    )

    assert worker.run_once(campaign_hash=campaign.sha256).status == "completed"
    shard_spec = campaign.shards()[0]
    shard_result = control.list_shards(campaign.sha256)[0]["result"]
    assert shard_result["candidateHashes"] == []
    assert len(shard_result["artifactReferences"]) == len(
        shard_result["traceHashes"]
    )
    assert all(
        item["ownerKind"] == "complete_trace_v1"
        for item in shard_result["artifactReferences"]
    )

    records = control.list_audit_traces(run_code=campaign.sha256)["items"]
    assert {item["traceHash"] for item in records} == set(
        shard_result["traceHashes"]
    )
    assert any(item["source"]["traceRole"] == "scored_final" for item in records)
    for item in records:
        source = item["source"]
        assert source["runCode"] == campaign.sha256
        assert source["runHash"] == campaign.sha256
        assert source["recipeCode"] == "organic-unscreened-v1"
        assert source["recipeHash"] == simulator.bundle_hash
        assert source["shardId"] == shard_spec.input_hash
        assert source["shardHash"] == shard_spec.input_hash
        assert source["workerId"] == "unscreened-worker"
        assert source["candidateHash"] == canonical_sha256({
            "schemaVersion": 1,
            "kind": "organic_foundry_candidate_occurrence_v1",
            "campaignHash": campaign.sha256,
            "shardHash": shard_spec.input_hash,
            "seed": source["seed"],
            "candidateOrdinal": source["candidateOrdinal"],
        })
        provenance = item["configurationProvenance"]
        assert provenance["researchConfiguration"]["campaign"]["sha256"] == (
            campaign.sha256
        )
        assert provenance["researchConfiguration"]["screeningProfile"] is None
        assert item["attemptStatus"] == "completed"
        assert item["auditOnly"] is True
        assert item["publicationEligible"] is False
        if source["traceRole"] == "scored_final":
            assert item["status"] == "provisional"
            assert item["disposition"] == "provisional"
        else:
            assert source["traceRole"] == "all_pass_attempt"
            assert item["status"] == "finalized"
            assert item["disposition"] == "rejected"
            assert {
                "stage_organic_simulation",
                "organic_all_pass_complete_trace",
            }.issubset(item["failureCategories"])


def test_screened_reduction_is_identical_across_shard_layouts(root, tmp_path):
    profile = _profile(
        baseRetainBasisPoints=5_000,
        maximumRetainBasisPoints=5_000,
        eliteRetainBasisPoints=5_000,
    )
    _, single = _run_campaign(
        root, tmp_path, profile, seeds=4, seeds_per_shard=4
    )
    _, partitioned = _run_campaign(
        root, tmp_path, profile, seeds=4, seeds_per_shard=1
    )
    for field in (
        "traceHashes",
        "candidateHashes",
        "coverageCells",
        "traceHashesByLane",
        "coverageCellsByLane",
    ):
        assert single[field] == partitioned[field]


def test_simulation_worker_is_a_packaged_entrypoint(root):
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert "baloot-foundry-simulation-worker" in pyproject
    assert "baloot_platform.service.foundry_simulation_worker:main" in pyproject
    assert "baloot-foundry-simulation-fleet" in pyproject
    assert "baloot_platform.service.foundry_simulation_fleet:main" in pyproject


def test_simulation_fleet_requires_explicit_bounded_identity():
    from baloot_platform.service.foundry_simulation_fleet import (
        FoundrySimulationFleetSpecV1,
    )

    environment = {
        "BALOOT_FOUNDRY_CAMPAIGN_HASH": "a" * 64,
        "BALOOT_FOUNDRY_WORKER_COUNT": "4",
        "BALOOT_FOUNDRY_RUN_ID": "proof-4w-none",
        "BALOOT_FOUNDRY_HOST_ID": "host-a",
        "BALOOT_FOUNDRY_FLEET_TIMEOUT_SECONDS": "3600",
    }
    spec = FoundrySimulationFleetSpecV1.from_environment(environment)
    assert spec.worker_count == 4
    assert spec.timeout_seconds == 3600
    identities = {
        spec.worker_environment(environment, slot)["BALOOT_WORKER_ID"]
        for slot in range(1, 5)
    }
    assert len(identities) == 4
    assert all("proof-4w-none" in item for item in identities)

    multi_lane = tuple(
        spec.worker_environment(
            environment, slot, ("organic", "guided", "disagreement")
        )
        for slot in range(1, 5)
    )
    assert [item["BALOOT_FOUNDRY_LANE"] for item in multi_lane] == [
        "disagreement", "guided", "organic", "disagreement"
    ]
    assert len({item["BALOOT_WORKER_ID"] for item in multi_lane}) == 4
    expected_workers = spec.expected_worker_ids(
        environment, ("organic", "guided")
    )
    assert expected_workers == tuple(sorted(expected_workers))
    assert set(expected_workers) == {
        spec.worker_environment(environment, slot, ("organic", "guided"))[
            "BALOOT_WORKER_ID"
        ]
        for slot in range(1, 5)
    }

    with pytest.raises(ValueError, match="at least the generated lane count"):
        FoundrySimulationFleetSpecV1.from_environment({
            **environment,
            "BALOOT_FOUNDRY_WORKER_COUNT": "2",
        }).worker_environment(
            environment, 1, ("organic", "guided", "disagreement")
        )

    with pytest.raises(ValueError, match="between 1 and 256"):
        FoundrySimulationFleetSpecV1.from_environment({
            **environment,
            "BALOOT_FOUNDRY_WORKER_COUNT": "0",
        })
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        FoundrySimulationFleetSpecV1.from_environment({
            **environment,
            "BALOOT_FOUNDRY_CAMPAIGN_HASH": "A" * 64,
        })


def test_simulation_fleet_releases_only_its_terminated_claims():
    from baloot_platform.service.foundry_simulation_fleet import (
        _release_owned_claims,
    )

    class Control:
        def __init__(self):
            self.failures = []

        def list_shards(self, campaign_hash):
            assert campaign_hash == "a" * 64
            return [
                {"specHash": "1" * 64, "status": "running", "workerId": "mine", "fencingToken": 3},
                {"specHash": "2" * 64, "status": "running", "workerId": "other", "fencingToken": 4},
                {"specHash": "3" * 64, "status": "completed", "workerId": None, "fencingToken": 1},
            ]

        def fail_shard(self, spec_hash, **values):
            self.failures.append((spec_hash, values))

    control = Control()
    released = _release_owned_claims(
        control,
        "a" * 64,
        ("mine",),
        error="fleet stopped before completion: TimeoutError",
    )
    assert released == 1
    assert control.failures == [(
        "1" * 64,
        {
            "worker_id": "mine",
            "fencing_token": 3,
            "error": "fleet stopped before completion: TimeoutError",
            "retryable": True,
        },
    )]
