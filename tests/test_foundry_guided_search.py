from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from baloot_platform.challenge.campaign import SimulationCampaignSpecV1
from baloot_platform.challenge.campaign_control import SQLiteCampaignControlStore
from baloot_platform.challenge.guided_search import (
    GuidedLegalSearchV1,
    GuidedSearchProfileV1,
)
from baloot_platform.challenge.simulation import NaturalLaffaSimulatorV1
from baloot_platform.challenge.worker import GuidedFoundryWorkerV1
from baloot_platform.storage.artifact_store import LocalContentAddressedArtifactStore


def _profile():
    return GuidedSearchProfileV1.build({
        "schemaVersion": 1,
        "id": "guided_structural_pilot_v1",
        "revision": "1",
        "agentProfileId": "strong",
        "candidatesPerSeed": 3,
        "targetContracts": ["sun", "hokum"],
        "targetEscalations": ["normal"],
        "requireProjects": False,
        "minimumNonForcedDecisions": 4,
        "minimumScore": 50.0,
        "weights": {
            "contractMatch": 2.0,
            "decisionOpportunity": 1.0,
            "scoreCloseness": 1.0,
        },
    })


def test_guided_legal_search_is_deterministic_complete_and_provenanced(root):
    simulator = NaturalLaffaSimulatorV1.from_root(root, maximum_deal_attempts=4)
    searcher = GuidedLegalSearchV1(simulator, _profile())
    first = searcher.search(900)
    second = searcher.search(900)
    assert first.sha256 == second.sha256
    assert first.to_dict() == second.to_dict()
    assert first.provenance == "guided_legal_search"
    assert len(first.evaluated) == 3
    assert len({item["candidateSeed"] for item in first.evaluated}) == 3
    assert first.selected_score == max(item["score"] for item in first.evaluated)
    assert first.selected_trace_hash == first.selected_simulation.final_trace.sha256
    assert first.selected_simulation.publishable_laffa
    for decision in first.selected_simulation.final_trace.decision_ledger.records:
        assert dict(decision.chosen_action) in [dict(item) for item in decision.legal_actions]


def test_guided_worker_claims_only_guided_shard_and_persists_search_manifest(root, tmp_path):
    simulator = NaturalLaffaSimulatorV1.from_root(root, maximum_deal_attempts=4)
    profile = GuidedSearchProfileV1.build({
        **{key: value for key, value in _profile().to_dict().items() if key != "sha256"},
        "candidatesPerSeed": 2,
    })
    searcher = GuidedLegalSearchV1(simulator, profile)
    campaign = SimulationCampaignSpecV1.build(
        rules_hash=simulator.rules.sha256,
        bundle_hash=simulator.bundle_hash,
        engine_version=simulator.engine_version,
        analysis_profile_hash="3" * 64,
        perspective_profile_hash="4" * 64,
        selection_profile_hash="5" * 64,
        agent_release=simulator.agent_release,
        seat_agents=simulator.equal_seat_agents,
        lanes=("guided",),
        lane_profile_hashes={"guided": profile.sha256},
        seed_start=10,
        seed_count=1,
        seeds_per_shard=1,
    )
    control = SQLiteCampaignControlStore(tmp_path / "guided.sqlite3", maximum_attempts=2)
    control.create_campaign(campaign)
    artifacts = LocalContentAddressedArtifactStore(tmp_path / "artifacts")
    worker = GuidedFoundryWorkerV1(
        control,
        artifacts,
        searcher,
        worker_id="guided-worker",
        lease_seconds=60,
        heartbeat_seconds=10,
    )
    run = worker.run_once(campaign_hash=campaign.sha256)
    assert run.status == "completed"
    shard = control.list_shards(campaign.sha256)[0]
    assert shard["spec"]["lane"] == "guided"
    assert shard["result"]["lane"] == "guided"
    references = shard["result"]["artifactReferences"]
    manifests = [item for item in references if item["ownerKind"] == "guided_search_v1"]
    assert len(manifests) == 1
    manifest = json.loads(artifacts.get(manifests[0]["uri"]))
    assert manifest["provenance"] == "guided_legal_search"
    assert manifest["profileHash"] == profile.sha256
    assert shard["result"]["candidateHashes"] == (
        [manifest["selectedTraceHash"]] if manifest["targetSatisfied"] else []
    )
    assert "source:guided" in shard["result"]["coverageCells"]


def test_guided_profile_rejects_unknown_targets_and_unbounded_search():
    with pytest.raises(ValueError, match="unsupported contract"):
        GuidedSearchProfileV1.build({
            **{key: value for key, value in _profile().to_dict().items() if key != "sha256"},
            "targetContracts": ["invented"],
        })

    with pytest.raises(ValueError, match="agent profile identity"):
        GuidedSearchProfileV1.build({
            **{
                key: value
                for key, value in _profile().to_dict().items()
                if key not in {"sha256", "agentProfileId"}
            },
        })


def test_guided_search_rejects_agent_profile_mismatch(root):
    simulator = NaturalLaffaSimulatorV1.from_root(root, maximum_deal_attempts=1)
    profile = GuidedSearchProfileV1.build({
        **{key: value for key, value in _profile().to_dict().items() if key != "sha256"},
        "agentProfileId": "finalist_candidate",
    })
    with pytest.raises(ValueError, match="governed agent profile"):
        GuidedLegalSearchV1(simulator, profile)


def test_guided_tactical_opportunity_gate_is_explicit_and_fail_closed(root):
    simulator = NaturalLaffaSimulatorV1.from_root(root, maximum_deal_attempts=4)
    profile = GuidedSearchProfileV1.build({
        **{key: value for key, value in _profile().to_dict().items() if key != "sha256"},
        "requireActualDealOpportunity": True,
        "minimumActualDealMarginGap": 5.0,
        "maximumActualDealDecisions": 2,
        "weights": {"actualDealOutcomeGap": 1.0},
    })
    with pytest.raises(ValueError, match="requires an exact scanner"):
        GuidedLegalSearchV1(simulator, profile)

    class Scanner:
        def discover_opportunities(self, trace, *, maximum_decisions, cancel_check):
            assert maximum_decisions == 2
            return (
                SimpleNamespace(
                    outcome_gap=10.0,
                    complete=True,
                    to_dict=lambda: {
                        "schemaVersion": 1,
                        "decisionId": "decision-test",
                        "outcomeGap": 10.0,
                        "complete": True,
                    },
                ),
            )

    searcher = GuidedLegalSearchV1(simulator, profile, Scanner())
    row = searcher._evaluate(simulator.simulate(902))
    assert row["actualDealOpportunityPassed"]
    assert row["components"]["actualDealOutcomeGap"] == 50.0
    assert row["actualDealOpportunities"][0]["decisionId"] == "decision-test"
    with pytest.raises(ValueError, match="candidates per seed"):
        GuidedSearchProfileV1.build({
            **{key: value for key, value in _profile().to_dict().items() if key != "sha256"},
            "candidatesPerSeed": 0,
        })


def test_guided_search_marks_all_pass_attempt_as_ineligible(root):
    simulator = NaturalLaffaSimulatorV1.from_root(root, maximum_deal_attempts=1)
    searcher = GuidedLegalSearchV1(simulator, _profile())
    trace = SimpleNamespace(
        sha256="a" * 64,
        trusted_final_state={
            "contract": None,
            "escalation": "normal",
            "projects": [],
            "roundScore": None,
        },
        decision_ledger=SimpleNamespace(records=()),
    )
    row = searcher._evaluate(SimpleNamespace(
        root_seed=1,
        final_trace=trace,
        publishable_laffa=False,
    ))
    assert row["analysisStatus"] == "ineligible_all_pass"
    assert not row["hardConstraintsPassed"]
    assert row["score"] == 0.0
