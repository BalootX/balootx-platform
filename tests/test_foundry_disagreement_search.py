from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from baloot_platform.challenge.analysis_pipeline import TraceDecisionAnalysisV3
from baloot_platform.challenge.campaign import SimulationCampaignSpecV1
from baloot_platform.challenge.campaign_control import SQLiteCampaignControlStore
from baloot_platform.challenge.disagreement_search import (
    DisagreementSearchProfileV1,
    DisagreementSearchV1,
)
from baloot_platform.challenge.evidence_v3 import canonical_sha256
from baloot_platform.challenge.simulation import NaturalLaffaSimulatorV1
from baloot_platform.challenge.worker import DisagreementFoundryWorkerV1
from baloot_platform.storage.artifact_store import LocalContentAddressedArtifactStore


class _FakeAnalyzer:
    def __init__(self, rules, profile_hash: str, mode: str):
        self.rules = rules
        self.profile = SimpleNamespace(sha256=profile_hash)
        self.mode = mode

    def analyze_trace(self, trace, *, cancel_check=None):
        if cancel_check is not None and cancel_check():
            raise InterruptedError
        decision = next(
            item for item in trace.decision_ledger.records
            if not item.forced and len(item.legal_actions) >= 2
        )
        first = dict(decision.legal_actions[0])
        second = dict(decision.legal_actions[1])
        if self.mode == "shallow":
            best = first
            alternatives = (
                {"action": first, "rank": 1, "expectedMargin": 8.0, "winProbability": 0.60},
                {"action": second, "rank": 2, "expectedMargin": 7.0, "winProbability": 0.58},
            )
            method = "shallow_paired_v1"
            confidence = 0.55
        else:
            best = second
            alternatives = (
                {"action": second, "rank": 1, "expectedMargin": 14.0, "winProbability": 0.76},
                {"action": first, "rank": 2, "expectedMargin": 4.0, "winProbability": 0.42},
            )
            method = "deep_paired_v1"
            confidence = 0.90
        analysis = TraceDecisionAnalysisV3(
            decision.decision_id,
            decision.ordinal,
            decision.acting_seat.name.lower(),
            decision.decision_kind,
            False,
            dict(decision.chosen_action),
            best,
            alternatives,
            method,
            64,
            64.0,
            confidence,
            80.0,
            20.0,
            0.0,
            0.0,
            False,
            ("expected_round_margin", "round_lead_probability"),
            (),
            canonical_sha256({"trace": trace.sha256, "mode": self.mode}),
        )
        return SimpleNamespace(
            decisions=(analysis,),
            sha256=canonical_sha256({"analysis": analysis.sha256}),
        )

    def analyze_decisions(self, trace, decision_ids, *, cancel_check=None):
        result = self.analyze_trace(trace, cancel_check=cancel_check)
        requested = set(decision_ids)
        return tuple(
            item for item in result.decisions if item.decision_id in requested
        )


class _UnresolvedAnalyzer(_FakeAnalyzer):
    def analyze_trace(self, trace, *, cancel_check=None):
        raise RuntimeError(
            "no legal hidden deals could reproduce the public history"
        )

    def analyze_decisions(self, trace, decision_ids, *, cancel_check=None):
        return self.analyze_trace(trace, cancel_check=cancel_check)


def _profile():
    return DisagreementSearchProfileV1.build({
        "schemaVersion": 1,
        "id": "test_disagreement_v1",
        "revision": "1",
        "candidatesPerSeed": 1,
        "maximumDeepDecisionsPerCandidate": 8,
        "minimumDeepConfidence": 0.75,
        "minimumExpectedMarginGap": 2.0,
        "minimumProbabilityGap": 0.10,
        "minimumScore": 30.0,
        "weights": {
            "disagreementCount": 1.0,
            "peakMarginGap": 1.0,
            "peakProbabilityGap": 1.0,
            "deepConfidence": 1.0,
            "methodDiversity": 1.0,
        },
    })


def _searcher(root):
    simulator = NaturalLaffaSimulatorV1.from_root(root, maximum_deal_attempts=4)
    return DisagreementSearchV1(
        simulator,
        _FakeAnalyzer(simulator.rules, "a" * 64, "shallow"),
        _FakeAnalyzer(simulator.rules, "b" * 64, "deep"),
        _profile(),
    )


def test_disagreement_search_is_actor_safe_deterministic_and_consequence_gated(root):
    searcher = _searcher(root)
    first = searcher.search(44)
    second = searcher.search(44)
    assert first.to_dict() == second.to_dict()
    assert first.target_satisfied
    assert first.provenance == "observer_safe_disagreement_search"
    row = first.evaluated[0]
    assert row["targetSatisfied"]
    disagreement = row["disagreements"][0]
    assert disagreement["shallowAction"] != disagreement["deepAction"]
    assert disagreement["expectedMarginGap"] == 10.0
    assert disagreement["probabilityGap"] == pytest.approx(0.34)
    assert disagreement["qualifies"]
    assert first.shallow_profile_hash != first.deep_profile_hash
    assert row["analysisStatus"] == "completed"
    assert row["rejectionReason"] is None
    assert row["deepTargetDecisionIds"]


def test_disagreement_search_records_sparse_public_history_as_unresolved(root):
    simulator = NaturalLaffaSimulatorV1.from_root(root, maximum_deal_attempts=4)
    searcher = DisagreementSearchV1(
        simulator,
        _UnresolvedAnalyzer(simulator.rules, "a" * 64, "shallow"),
        _FakeAnalyzer(simulator.rules, "b" * 64, "deep"),
        _profile(),
    )
    result = searcher.search(44)
    assert not result.target_satisfied
    assert result.selected_score == 0.0
    assert result.evaluated[0]["analysisStatus"] == "unresolved_hidden_deals"
    assert result.evaluated[0]["rejectionReason"] == (
        "public_history_conditioning_exhausted"
    )


def test_disagreement_worker_persists_both_analysis_provenance_hashes(root, tmp_path):
    searcher = _searcher(root)
    simulator = searcher.simulator
    campaign = SimulationCampaignSpecV1.build(
        rules_hash=simulator.rules.sha256,
        bundle_hash=simulator.bundle_hash,
        engine_version=simulator.engine_version,
        analysis_profile_hash=searcher.deep_analyzer.profile.sha256,
        perspective_profile_hash="4" * 64,
        selection_profile_hash="5" * 64,
        agent_release=simulator.agent_release,
        seat_agents=simulator.equal_seat_agents,
        lanes=("disagreement",),
        lane_profile_hashes={"disagreement": searcher.profile.sha256},
        seed_start=11,
        seed_count=1,
        seeds_per_shard=1,
    )
    control = SQLiteCampaignControlStore(tmp_path / "disagreement.sqlite3", maximum_attempts=2)
    control.create_campaign(campaign)
    artifacts = LocalContentAddressedArtifactStore(tmp_path / "artifacts")
    worker = DisagreementFoundryWorkerV1(
        control,
        artifacts,
        searcher,
        worker_id="disagreement-worker",
        lease_seconds=60,
        heartbeat_seconds=10,
    )
    run = worker.run_once(campaign_hash=campaign.sha256)
    assert run.status == "completed", run.error
    shard = control.list_shards(campaign.sha256)[0]
    manifests = [
        item for item in shard["result"]["artifactReferences"]
        if item["ownerKind"] == "disagreement_search_v1"
    ]
    assert len(manifests) == 1
    manifest = json.loads(artifacts.get(manifests[0]["uri"]))
    assert manifest["shallowProfileHash"] == "a" * 64
    assert manifest["deepProfileHash"] == "b" * 64
    assert shard["result"]["candidateHashes"] == [
        manifest["selectedTraceHash"]
    ]
    assert "source:disagreement" in shard["result"]["coverageCells"]


def test_disagreement_search_rejects_same_analysis_profile(root):
    simulator = NaturalLaffaSimulatorV1.from_root(root)
    analyzer = _FakeAnalyzer(simulator.rules, "a" * 64, "shallow")
    with pytest.raises(ValueError, match="distinct shallow and deep"):
        DisagreementSearchV1(simulator, analyzer, analyzer, _profile())
