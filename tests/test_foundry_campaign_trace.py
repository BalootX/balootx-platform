from __future__ import annotations

import sqlite3

import pytest

from baloot_platform.agents.profiles import AgentProfileRegistry
from baloot_platform.challenge.campaign import (
    SimulationCampaignSpecV1,
    SimulationShardResultV1,
    deterministic_campaign_reduction,
)
from baloot_platform.challenge.campaign_control import SQLiteCampaignControlStore
from baloot_platform.challenge.executor import DeterministicBatchExecutorV1
from baloot_platform.challenge.evidence_v3 import DecisionLedgerV1, canonical_sha256
from baloot_platform.challenge.trace import CompleteGameTraceV1, TraceEventV1
from baloot_platform.core.enums import Seat
from baloot_platform.learning.store import LearningStore
from baloot_platform.service.game import GameManager
from baloot_platform.storage.game_store import SQLiteGameStore


def _trace_inputs():
    game_hash = "a" * 64
    trusted = {
        "rulesHash": "b" * 64,
        "seed": 19,
        "dealer": "east",
        "shuffle": {
            "algorithm": "splitmix64_fisher_yates_v1",
            "seed": 19,
            "dealer": "east",
            "cutter": "south",
            "cutPosition": 9,
            "deckCommitment": "c" * 64,
            "deckOrder": list(range(32)),
        },
        "agentProfiles": {
            "north": "expert-v1", "west": "expert-v1",
            "south": "expert-v1", "east": "expert-v1",
        },
        "events": [
            {"sequence": 0, "type": "round_created", "dealer": "east"},
            {"sequence": 1, "type": "card_played", "seat": "north", "card": "S7"},
        ],
    }
    ledger = DecisionLedgerV1.build(game_hash, ({
        "decisionId": "D1",
        "ordinal": 1,
        "actor": "north",
        "decisionKind": "play",
        "phase": "playing",
        "publicState": {"observer": "north"},
        "completeStateHash": "d" * 64,
        "legalActions": [{"code": "S7"}, {"code": "S8"}],
        "selectedAction": {"code": "S7"},
    },))
    game = {
        "gameHash": game_hash,
        "gameId": "GAME-1",
        "stateHash": canonical_sha256(trusted),
        "bundleHash": "e" * 64,
        "engineVersion": "1.0.0",
    }
    return game, trusted, ledger


def test_complete_game_trace_is_hash_bound_and_public_projection_omits_hands():
    game, trusted, ledger = _trace_inputs()
    first = CompleteGameTraceV1.build(
        game_record=game,
        trusted_final_state=trusted,
        decision_ledger=ledger,
        model_releases=("model-v1",),
    )
    second = CompleteGameTraceV1.build(
        game_record=game,
        trusted_final_state=trusted,
        decision_ledger=ledger,
        model_releases=("model-v1",),
    )
    assert first.sha256 == second.sha256
    assert first.events[1].previous_event_hash == first.events[0].event_hash
    assert "trustedFinalState" not in first.to_dict()
    assert first.to_dict(trusted=True)["trustedFinalState"]["shuffle"]["deckOrder"] == list(range(32))

    tampered = {**trusted, "seed": 20}
    with pytest.raises(ValueError, match="state hash"):
        CompleteGameTraceV1.build(
            game_record=game,
            trusted_final_state=tampered,
            decision_ledger=ledger,
        )


def test_trace_event_chain_rejects_missing_sequences():
    with pytest.raises(ValueError, match="contiguous"):
        TraceEventV1.chain(({"sequence": 1, "type": "late"},))


def test_completed_game_store_builds_one_verified_complete_trace(root, tmp_path):
    store = LearningStore(tmp_path / "learning.sqlite3")
    manager = GameManager(
        root,
        SQLiteGameStore(tmp_path / "games.sqlite3"),
        learning_store=store,
        agent_registry=AgentProfileRegistry.from_yaml(
            root / "configs/agents/adaptive_lineup_v1.yaml"
        ),
    )
    game = manager.create(
        seed=501,
        dealer=Seat.EAST,
        agent_profiles={seat: "beginner" for seat in ("north", "west", "south", "east")},
    )
    game = manager.advance_bots(game.id, stop_before=None, maximum_actions=100)
    trace = store.complete_game_trace(str(game.game_hash))
    assert trace.game_hash == game.game_hash
    assert len(trace.decision_ledger.records) >= 32
    assert all(record.complete_state_hash for record in trace.decision_ledger.records)
    assert trace.events[-1].event_type in {"round_scored", "match_completed"}
    assert trace.sha256 == store.complete_game_trace(str(game.game_hash)).sha256


def test_all_pass_redeal_uses_a_new_immutable_event_stream(root, tmp_path):
    game_store = SQLiteGameStore(tmp_path / "redeal-games.sqlite3")
    manager = GameManager(root, game_store)
    game = manager.create(seed=509, dealer=Seat.EAST)
    initial_round = game.state.round_number
    for _ in range(12):
        actor = game.bidding.actor
        game = manager.bid(game.id, seat=actor, action="pass", expected_version=game.version)
        if game.state.round_number != initial_round:
            break
    assert game.state.round_number == initial_round + 1
    verification = game_store.verify(game.id)
    assert verification["valid"] is True
    assert verification["eventVerification"]["streamCount"] == 2


def _campaign() -> SimulationCampaignSpecV1:
    return SimulationCampaignSpecV1.build(
        rules_hash="1" * 64,
        bundle_hash="2" * 64,
        engine_version="1.0.0",
        analysis_profile_hash="3" * 64,
        perspective_profile_hash="4" * 64,
        selection_profile_hash="5" * 64,
        agent_release="expert-v1",
        seat_agents=("expert-v1",) * 4,
        lanes=("organic",),
        lane_profile_hashes={"organic": "5" * 64},
        seed_start=100,
        seed_count=10,
        seeds_per_shard=4,
    )


def test_campaign_shards_and_reduction_are_order_independent():
    campaign = _campaign()
    shards = campaign.shards()
    assert [(item.seed_start, item.seed_count) for item in shards] == [(100, 4), (104, 4), (108, 2)]
    results = tuple(
        SimulationShardResultV1.build(
            shard,
            processed_seeds=range(shard.seed_start, shard.seed_end_exclusive),
            trace_hashes=(f"{shard.shard_index + 6:x}" * 64,),
            candidate_hashes=(f"{shard.shard_index + 9:x}" * 64,),
            coverage_cells=(f"contract:sun:shard:{shard.shard_index}",),
            completion_status="completed",
        )
        for shard in shards
    )
    forward = deterministic_campaign_reduction(campaign, results)
    reverse = deterministic_campaign_reduction(campaign, reversed(results))
    assert forward == reverse
    assert len(forward["traceHashes"]) == 3
    assert forward["traceHashesByLane"] == {"organic": forward["traceHashes"]}


def test_campaign_partitions_each_discovery_lane_and_workers_claim_capabilities(tmp_path):
    campaign = SimulationCampaignSpecV1.build(
        rules_hash="1" * 64,
        bundle_hash="2" * 64,
        engine_version="1.0.0",
        analysis_profile_hash="3" * 64,
        perspective_profile_hash="4" * 64,
        selection_profile_hash="5" * 64,
        agent_release="expert-v1",
        seat_agents=("expert-v1",) * 4,
        lanes=("organic", "guided", "disagreement"),
        lane_profile_hashes={
            "organic": "6" * 64,
            "guided": "7" * 64,
            "disagreement": "8" * 64,
        },
        seed_start=100,
        seed_count=4,
        seeds_per_shard=2,
    )
    shards = campaign.shards()
    assert len(shards) == 6
    assert {item.lane for item in shards} == {"organic", "guided", "disagreement"}
    assert all(
        [(item.seed_start, item.seed_count) for item in shards if item.lane == lane]
        == [(100, 2), (102, 2)]
        for lane in campaign.lanes
    )
    store = SQLiteCampaignControlStore(tmp_path / "lane-routing.sqlite3", maximum_attempts=2)
    store.create_campaign(campaign)
    organic = store.claim_shard(
        "organic-worker",
        lease_seconds=60,
        supported_lanes=("organic",),
        now=100,
    )
    guided = store.claim_shard(
        "guided-worker",
        lease_seconds=60,
        supported_lanes=("guided",),
        now=100,
    )
    assert organic is not None and organic.spec.lane == "organic"
    assert guided is not None and guided.spec.lane == "guided"
    with pytest.raises(ValueError, match="supported discovery lanes"):
        store.claim_shard(
            "bad-worker",
            lease_seconds=60,
            supported_lanes=(),
            now=100,
        )


def test_campaign_rejects_unequal_agents_and_partial_shard_completion():
    with pytest.raises(ValueError, match="equal signed agent"):
        SimulationCampaignSpecV1.build(
            rules_hash="1" * 64,
            bundle_hash="2" * 64,
            engine_version="1",
            analysis_profile_hash="3" * 64,
            perspective_profile_hash="4" * 64,
            selection_profile_hash="5" * 64,
            agent_release="expert-v1",
            seat_agents=("expert-v1", "weak", "expert-v1", "expert-v1"),
            lanes=("organic",),
            lane_profile_hashes={"organic": "5" * 64},
            seed_start=0,
            seed_count=4,
            seeds_per_shard=4,
        )
    shard = _campaign().shards()[0]
    with pytest.raises(ValueError, match="every assigned seed"):
        SimulationShardResultV1.build(
            shard,
            processed_seeds=(100, 101),
            trace_hashes=(),
            candidate_hashes=(),
            coverage_cells=(),
            completion_status="completed",
        )


def _completed_result(shard, marker: str):
    return SimulationShardResultV1.build(
        shard,
        processed_seeds=range(shard.seed_start, shard.seed_end_exclusive),
        trace_hashes=(marker * 64,),
        candidate_hashes=(),
        coverage_cells=(f"cell:{shard.shard_index}",),
        completion_status="completed",
    )


def test_campaign_control_uses_leases_fencing_and_exact_completion(tmp_path):
    store = SQLiteCampaignControlStore(tmp_path / "campaigns.sqlite3", maximum_attempts=3)
    campaign = _campaign()
    created = store.create_campaign(campaign)
    assert created["status"] == "queued" and created["shardCounts"] == {"queued": 3}

    first = store.claim_shard("worker-1", lease_seconds=10, now=100)
    second = store.claim_shard("worker-2", lease_seconds=10, now=100)
    assert first is not None and second is not None
    assert first.spec.shard_index == 0 and second.spec.shard_index == 1

    reclaimed = store.claim_shard("worker-3", lease_seconds=10, now=111)
    assert reclaimed is not None and reclaimed.spec.input_hash == first.spec.input_hash
    assert reclaimed.fencing_token > first.fencing_token
    assert store.heartbeat(
        first.spec.input_hash,
        worker_id="worker-1",
        fencing_token=first.fencing_token,
        lease_seconds=10,
        now=111,
    ) is False
    with pytest.raises(PermissionError, match="stale"):
        store.commit_result(
            _completed_result(first.spec, "6"),
            worker_id="worker-1",
            fencing_token=first.fencing_token,
            now=105,
        )

    store.commit_result(
        _completed_result(reclaimed.spec, "6"),
        worker_id="worker-3",
        fencing_token=reclaimed.fencing_token,
        now=115,
    )
    store.commit_result(
        _completed_result(second.spec, "7"),
        worker_id="worker-2",
        fencing_token=second.fencing_token,
        now=105,
    )
    third = store.claim_shard("worker-4", lease_seconds=10, now=112)
    assert third is not None and third.spec.shard_index == 2
    store.commit_result(
        _completed_result(third.spec, "8"),
        worker_id="worker-4",
        fencing_token=third.fencing_token,
        now=115,
    )
    completed = store.get_campaign(campaign.sha256)
    assert completed is not None and completed["status"] == "completed"
    assert completed["shardCounts"] == {"completed": 3}


def test_cancelled_campaign_cannot_issue_more_work(tmp_path):
    store = SQLiteCampaignControlStore(tmp_path / "cancel.sqlite3", maximum_attempts=2)
    campaign = _campaign()
    store.create_campaign(campaign)
    assert store.request_cancel(campaign.sha256) is True
    assert store.claim_shard("worker", lease_seconds=10, now=0) is None
    assert {item["status"] for item in store.list_shards(campaign.sha256)} == {"cancelled"}


def test_failed_campaign_cannot_issue_remaining_queued_work(tmp_path):
    store = SQLiteCampaignControlStore(tmp_path / "failed.sqlite3", maximum_attempts=2)
    campaign = _campaign()
    store.create_campaign(campaign)
    failed = store.claim_shard("worker-1", lease_seconds=10, now=0)
    assert failed is not None
    store.fail_shard(
        failed.spec.input_hash,
        worker_id="worker-1",
        fencing_token=failed.fencing_token,
        error="terminal injected failure",
        retryable=False,
    )

    snapshot = store.get_campaign(campaign.sha256)
    assert snapshot is not None and snapshot["status"] == "failed"
    assert store.claim_shard("worker-2", lease_seconds=10, now=1) is None
    assert snapshot["shardCounts"] == {"failed": 1, "queued": 2}

    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE foundry_campaigns_v1 SET status='running' WHERE campaign_hash=?",
            (campaign.sha256,),
        )
    repaired_projection = store.get_campaign(campaign.sha256)
    assert repaired_projection is not None
    assert repaired_projection["status"] == "failed"
    assert store.claim_shard("worker-3", lease_seconds=10, now=2) is None


def test_postgres_campaign_migration_matches_packaged_runtime(root):
    source = root / "migrations/015_foundry_campaign_control.sql"
    packaged = root / "src/baloot_platform/resources/migrations" / source.name
    assert source.read_bytes() == packaged.read_bytes()
    sql = source.read_text(encoding="utf-8")
    assert "foundry_campaigns_v1" in sql
    assert "foundry_shards_v1" in sql
    assert "fencing_token" in sql and "lease_until" in sql
    runtime = (root / "src/baloot_platform/challenge/campaign_control.py").read_text(encoding="utf-8")
    assert "class PostgresCampaignControlStore" in runtime
    assert "SKIP LOCKED" in runtime
    assert runtime.count("c.status IN ('queued','running')") == 2
    assert runtime.count("failed.campaign_hash=s.campaign_hash") == 2


def test_batch_executor_is_ordered_deterministic_and_rejects_partial_completion():
    shard = _campaign().shards()[0]
    executor = DeterministicBatchExecutorV1()

    def simulate(seed: int):
        return {
            "traceHash": f"{seed % 10}" * 64,
            "candidateHashes": ("a" * 64,),
            "coverageCells": (f"seed-mod:{seed % 2}",),
        }

    first = executor.execute_batch(shard, simulate_seed=simulate)
    second = executor.execute_batch(shard, simulate_seed=simulate)
    assert first == second and first.completion_status == "completed"
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        return calls > 2

    partial = executor.execute_batch(shard, simulate_seed=simulate, cancel_check=cancelled)
    assert partial.completion_status == "cancelled"
    assert len(partial.processed_seeds) == 2


def test_typed_campaign_api_creates_claims_and_commits_exact_shards(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from baloot_platform.service import api

    store = SQLiteCampaignControlStore(tmp_path / "api-campaigns.sqlite3", maximum_attempts=3)
    monkeypatch.setattr(api, "FOUNDRY_CAMPAIGNS", store)
    body = {
        "rulesHash": "1" * 64,
        "bundleHash": "2" * 64,
        "engineVersion": "1.0.0",
        "analysisProfileHash": "3" * 64,
        "perspectiveProfileHash": "4" * 64,
        "selectionProfileHash": "5" * 64,
        "agentRelease": "expert-v1",
        "seatAgents": {
            "north": "expert-v1", "west": "expert-v1",
            "south": "expert-v1", "east": "expert-v1",
        },
        "lanes": ["organic", "guided"],
        "laneProfileHashes": {"organic": "a" * 64, "guided": "b" * 64},
        "seedStart": 20,
        "seedCount": 2,
        "seedsPerShard": 2,
    }
    with TestClient(api.app) as client:
        created = client.post("/api/v2/campaigns", json=body)
        assert created.status_code == 201, created.text
        campaign_hash = created.json()["campaignHash"]
        claim_response = client.post(
            "/api/v2/shards:claim",
            json={"workerId": "worker-api", "campaignHash": campaign_hash, "leaseSeconds": 60},
        )
        assert claim_response.status_code == 200, claim_response.text
        claim = claim_response.json()["claim"]
        spec = claim["spec"]
        committed = client.post(
            f"/api/v2/shards/{spec['inputHash']}:commit",
            json={
                "workerId": "worker-api",
                "fencingToken": claim["fencingToken"],
                "processedSeeds": [20, 21],
                "traceHashes": ["6" * 64],
                "candidateHashes": ["7" * 64],
                "coverageCells": ["sun:play:north"],
                "completionStatus": "completed",
            },
        )
        assert committed.status_code == 200, committed.text
        status = client.get(f"/api/v2/campaigns/{campaign_hash}").json()
        assert status["status"] == "running"
        guided_claim = client.post(
            "/api/v2/shards:claim",
            json={
                "workerId": "guided-api",
                "campaignHash": campaign_hash,
                "leaseSeconds": 60,
                "supportedLanes": ["guided"],
            },
        ).json()["claim"]
        assert guided_claim["spec"]["lane"] == "guided"
        guided_spec = guided_claim["spec"]
        guided_commit = client.post(
            f"/api/v2/shards/{guided_spec['inputHash']}:commit",
            json={
                "workerId": "guided-api",
                "fencingToken": guided_claim["fencingToken"],
                "processedSeeds": [20, 21],
                "traceHashes": ["8" * 64],
                "candidateHashes": ["9" * 64],
                "coverageCells": ["guided:hokum:play:east"],
                "completionStatus": "completed",
            },
        )
        assert guided_commit.status_code == 200, guided_commit.text
        assert client.get(f"/api/v2/campaigns/{campaign_hash}").json()["status"] == "completed"
