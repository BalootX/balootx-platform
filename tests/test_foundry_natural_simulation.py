from __future__ import annotations

import copy
from dataclasses import replace
import json
from types import SimpleNamespace

import yaml
import pytest

from baloot_platform.agents.profiles import AgentProfile, AgentProfileRegistry
from baloot_platform.challenge.analysis_pipeline import (
    CascadedTraceAnalysisProfileV1,
    CascadedTraceAnalyzerV1,
    StagedTraceAnalyzerV1,
    TraceAnalysisProfileV1,
    TraceAnalysisResultV1,
)
from baloot_platform.challenge.campaign import SimulationCampaignSpecV1
from baloot_platform.challenge.campaign_control import SQLiteCampaignControlStore
from baloot_platform.challenge.evidence_v3 import DecisionLedgerV1, canonical_sha256
from baloot_platform.challenge.executor import DeterministicBatchExecutorV1
from baloot_platform.challenge.perspective_pipeline import build_all_observer_projections
from baloot_platform.challenge.profiles import (
    ContentSelectionScoreV1,
    SelectionProfileV1,
    SurroundingCompetenceEvidenceV1,
    ThrillProfileV1,
)
from baloot_platform.challenge.quality_diversity import (
    QualityDiversityCandidateV1,
    QualityDiversityProfileV1,
    QualityDiversitySelectorV1,
)
from baloot_platform.challenge.oracle import ActualDealOracleVerifierV1
from baloot_platform.challenge.replay import CompleteTraceReplayerV1, TraceDecisionProjectionStoreV1
from baloot_platform.challenge.simulation import NaturalLaffaSimulatorV1
from baloot_platform.challenge.trace import CompleteGameTraceV1
from baloot_platform.challenge.worker import OrganicFoundryWorkerV1
from baloot_platform.challenge.funnel import (
    FoundryAnalysisFunnelResultV1,
    FoundryAnalysisFunnelV1,
)
from baloot_platform.config import build_bundle
from baloot_platform.core.cards import Card
from baloot_platform.core.enums import GamePhase, Seat
from baloot_platform.core.rules import RuleBook
from baloot_platform.challenge.evidence_v3 import TruthLabel
from baloot_platform.plugins.base import (
    AnalysisFunnelPlugin,
    GeneratorPlugin,
    TacticalVerifierPlugin,
    TraceAnalyzerPlugin,
)
from baloot_platform.storage.artifact_store import LocalContentAddressedArtifactStore


def _simulator(root, *, maximum_deal_attempts: int = 4) -> NaturalLaffaSimulatorV1:
    registry = AgentProfileRegistry.from_bundle(root)
    bundle = build_bundle(root)
    return NaturalLaffaSimulatorV1(
        RuleBook.load(root / "configs/rules/saudi_common_v1.yaml"),
        yaml.safe_load(
            (root / "configs/conventions/conservative_v1.yaml").read_text(encoding="utf-8")
        ),
        bundle_hash=bundle.sha256,
        configuration_snapshot=bundle.configuration_snapshot(),
        agent_profile=registry.get("strong"),
        maximum_deal_attempts=maximum_deal_attempts,
    )


def test_default_actor_selection_weights_use_supported_decision_dimensions(root):
    challenge = build_bundle(root).component("challenge")
    profile = SelectionProfileV1.build(challenge["selection_policy"])
    decision_evidence_dimensions = {
        "outcomeImpact", "uniqueness", "disasterAvoidance", "kaboutImpact",
        "sacrificeValue", "delayedGain", "tacticalDepth", "informationBurden",
        "executionDifficulty", "surprise", "robustness", "confidence",
        "explanationFidelity", "visualClarity", "novelty", "coverage",
    }
    assert set(dict(profile.weights)).issubset(decision_evidence_dimensions)
    thrill = ThrillProfileV1.build(challenge["thrill_profile"])
    assert {
        "overallPeak",
        "causalChainStrength",
        "outcomeImportance",
        "narrativeArc",
    }.issubset(dict(thrill.weights))


def test_natural_laffa_is_byte_stable_complete_and_all_seat(root):
    simulator = _simulator(root)
    assert isinstance(simulator, GeneratorPlugin)
    first = simulator.simulate(1, dealer=Seat.EAST)
    second = simulator.simulate(1, dealer=Seat.EAST)

    assert first.sha256 == second.sha256
    assert first.to_dict(trusted=True) == second.to_dict(trusted=True)
    assert first.publishable_laffa is True
    assert first.final_trace.trusted_final_state["phase"] == GamePhase.SCORED.value
    records = first.final_trace.decision_ledger.records
    play_records = [item for item in records if item.decision_kind == "play"]
    assert len(play_records) == 32
    assert {item.acting_seat for item in play_records} == set(Seat)
    assert all(item.chosen_action in item.legal_actions for item in records)
    assert all(item.complete_state_hash for item in records)
    assert all(item.resulting_event_hash for item in records)
    assert all(
        profile == simulator.agent_release
        for profile in dict(first.final_trace.agent_profiles).values()
    )
    snapshot = first.final_trace.trusted_final_state["configurationSnapshot"]
    assert snapshot["sha256"] == simulator.configuration_snapshot_hash
    assert snapshot["bundleHash"] == simulator.bundle_hash
    assert snapshot["components"]["rules"]["sha256"] == simulator.rules.sha256
    assert set(snapshot["components"]) == set(build_bundle(root).components)
    assert "trustedFinalState" not in first.final_trace.to_dict()


def test_all_pass_attempts_are_retained_as_separate_non_publishable_traces(root, rules, convention):
    raw = copy.deepcopy(rules.data)
    for key in (
        "sun_threshold",
        "hokum_threshold",
        "ashkal_partner_threshold",
        "second_sun_threshold",
        "second_hokum_threshold",
    ):
        raw["bidding"]["heuristics"][key] = 999.0
    pass_profile = AgentProfile.from_mapping(
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
    pass_rules = RuleBook(raw)
    snapshot_core = {
        "schemaVersion": 1,
        "bundleHash": "b" * 64,
        "mode": "test",
        "profile": {"id": "test", "version": "1", "status": "draft"},
        "components": {
            "rules": {"sha256": pass_rules.sha256, "payload": pass_rules.data},
            "conventions": {"sha256": "c" * 64, "payload": convention},
        },
    }
    simulator = NaturalLaffaSimulatorV1(
        pass_rules,
        convention,
        bundle_hash="b" * 64,
        configuration_snapshot={
            **snapshot_core,
            "sha256": canonical_sha256(snapshot_core),
        },
        agent_profile=pass_profile,
        maximum_deal_attempts=2,
    )
    result = simulator.simulate(91, dealer=Seat.EAST)

    assert result.publishable_laffa is False
    assert len(result.attempt_traces) == 2
    assert len({item.sha256 for item in result.attempt_traces}) == 2
    assert all(item.trusted_final_state["phase"] == GamePhase.COMPLETE.value for item in result.attempt_traces)
    assert all(
        {record.decision_kind for record in item.decision_ledger.records} == {"bid"}
        for item in result.attempt_traces
    )
    assert "auction:all_pass" in result.coverage_cells


def test_batch_executor_retains_every_attempt_trace(root):
    simulator = _simulator(root, maximum_deal_attempts=2)
    campaign = SimulationCampaignSpecV1.build(
        rules_hash=simulator.rules.sha256,
        bundle_hash=simulator.bundle_hash,
        engine_version=simulator.engine_version,
        analysis_profile_hash="3" * 64,
        perspective_profile_hash="4" * 64,
        selection_profile_hash="5" * 64,
        agent_release=simulator.agent_release,
        seat_agents=simulator.equal_seat_agents,
        lanes=("organic",),
        lane_profile_hashes={"organic": simulator.bundle_hash},
        seed_start=1,
        seed_count=2,
        seeds_per_shard=2,
    )
    shard = campaign.shards()[0]
    result = DeterministicBatchExecutorV1().execute_batch(
        shard,
        simulate_seed=lambda seed: simulator.simulate(seed).to_batch_output(),
    )

    assert result.completion_status == "completed"
    assert result.processed_seeds == (1, 2)
    assert len(result.trace_hashes) >= 2
    assert "decision:play" in result.coverage_cells


def test_complete_trace_replay_reconstructs_every_decision_and_projection(root):
    simulator = _simulator(root)
    simulation = simulator.simulate(3)
    trace = simulation.final_trace
    replay = CompleteTraceReplayerV1(simulator.rules).replay(trace)
    assert len(replay.decisions) == len(trace.decision_ledger.records)

    store = TraceDecisionProjectionStoreV1(trace, simulator.rules)
    later_bid = next(
        item for item in trace.decision_ledger.records
        if item.decision_kind == "bid" and item.ordinal > 1
    )
    bid_views = build_all_observer_projections(store, later_bid.decision_id)
    assert set(bid_views) == set(Seat)
    assert all(
        view.information_state["biddingHistory"]
        for view in bid_views.values()
    )
    assert all(
        len(view.information_state["observerHand"]) == 5
        for view in bid_views.values()
    )

    play_decision = next(
        item for item in trace.decision_ledger.records
        if item.decision_kind == "play"
    )
    play_views = build_all_observer_projections(store, play_decision.decision_id)
    assert len({view.sha256 for view in play_views.values()}) == 4
    assert all(
        view.information_state["observer"] == seat.name.lower()
        for seat, view in play_views.items()
    )


def test_complete_trace_replay_rejects_a_tampered_selected_card(root):
    simulator = _simulator(root)
    trace = simulator.simulate(2).final_trace
    records = list(trace.decision_ledger.records)
    play_index = next(index for index, item in enumerate(records) if item.decision_kind == "play")
    play = records[play_index]
    illegal = next(
        card for card in range(32)
        if all(int(action["id"]) != card for action in play.legal_actions)
    )
    records[play_index] = replace(play, chosen_action={"id": illegal})
    tampered = replace(
        trace,
        decision_ledger=DecisionLedgerV1.build(trace.game_hash, records),
    )

    with pytest.raises(ValueError, match="selected card"):
        CompleteTraceReplayerV1(simulator.rules).replay(tampered)


def test_staged_trace_analysis_covers_every_decision_without_fake_metrics(root):
    simulator = _simulator(root)
    trace = simulator.simulate(2).final_trace
    bundle = build_bundle(root)
    profile = TraceAnalysisProfileV1.build(
        bid_samples=2,
        play_samples=2,
        exact_allocation_limit=50,
        exact_search_max_remaining=2,
        maximum_sampling_attempts=2_000,
        exact_search_max_nodes=10_000,
        exact_search_max_seconds=0.5,
        exact_search_max_cache_entries=5_000,
    )
    analyzer = StagedTraceAnalyzerV1(
        bundle.rules,
        bundle.convention,
        profile,
        match_value_profile=bundle.component("match_value"),
    )
    assert isinstance(analyzer, TraceAnalyzerPlugin)
    first = analyzer.analyze_trace(trace)
    second = analyzer.analyze_trace(trace)

    assert first.sha256 == second.sha256
    assert len(first.decisions) == len(trace.decision_ledger.records)
    assert len(first.laffa_evidence.perspectives) == 4 * len(first.decisions)
    assert all(
        decision_id in {item.decision_id for item in first.decisions}
        for decision_id in first.laffa_evidence.critical_decision_ids
    )
    assert first.laffa_evidence.classification
    bid = next(item for item in first.decisions if item.decision_kind == "bid")
    assert "true_match_win_probability" in bid.unsupported_metrics
    play = next(
        item for item in first.decisions
        if item.decision_kind == "play" and not item.forced
    )
    assert "match_win_probability" in play.supported_metrics
    assert all("matchWinProbability" in item for item in play.alternatives)

    replay_store = TraceDecisionProjectionStoreV1(trace, bundle.rules)
    late_snapshot = next(
        item for item in replay_store.replay.decisions
        if item.decision.decision_kind == "play"
        and not item.decision.forced
        and sum(mask.bit_count() for mask in item.state.hands) <= 5
    )
    late_analysis = next(
        item for item in first.decisions
        if item.decision_id == late_snapshot.decision.decision_id
    )
    restored_late_analysis = type(late_analysis).from_dict(
        late_analysis.to_dict()
    )
    assert restored_late_analysis.confidence == late_analysis.confidence
    if (
        late_analysis.method.startswith("exhaustive_")
        and all(
            float(item.get("exactShare", 0.0)) == 1.0
            and item.get("searchCompletionStatus") == "exact"
            for item in late_analysis.alternatives
        )
    ):
        assert late_analysis.confidence == 1.0
    verifier = ActualDealOracleVerifierV1(
        bundle.rules,
        maximum_remaining_cards=5,
        maximum_nodes=100_000,
        maximum_seconds=5.0,
    )
    assert isinstance(verifier, TacticalVerifierPlugin)
    verified = verifier.verify_decision(late_snapshot, late_analysis)
    repeated = verifier.verify_decision(late_snapshot, late_analysis)
    assert verified.sha256 == repeated.sha256
    assert verified.status == "verified" and verified.oracle.complete
    assert verified.oracle.truth_mode.value == "actual_deal_oracle"
    assert verified.oracle.observer_recommendation == late_analysis.best_action
    recommended_id = Card.parse(verified.oracle.observer_recommendation).id
    recommended = next(
        item
        for item in verified.oracle.actions
        if Card.parse(dict(item.action)).id == recommended_id
    )
    assert verified.oracle.observer_recommendation_unchanged == (
        recommended.final_margin
        == max(item.final_margin for item in verified.oracle.actions)
    )
    assert all(item.completion_status == "exact" for item in verified.oracle.actions)
    opportunities = verifier.discover_opportunities(
        trace, maximum_decisions=2
    )
    repeated_opportunities = verifier.discover_opportunities(
        trace, maximum_decisions=2
    )
    assert [item.sha256 for item in opportunities] == [
        item.sha256 for item in repeated_opportunities
    ]
    assert opportunities
    assert all(item.remaining_cards <= 5 for item in opportunities)
    assert all(item.outcome_gap >= 0.0 for item in opportunities)
    viability = verifier.probe_verified_tactical_play(trace)
    repeated_viability = verifier.probe_verified_tactical_play(trace)
    assert viability.sha256 == repeated_viability.sha256
    assert viability.trace_hash == trace.sha256
    assert viability.status in {
        "viable",
        "no_verified_tactical_play_opportunity",
        "inconclusive",
    }
    assert viability.eligible_decision_count >= len(viability.scanned_decisions)
    assert viability.can_reject == (
        viability.scan_complete and not viability.viable
    )
    bounded_viability = verifier.probe_verified_tactical_play(
        trace, maximum_decisions=1
    )
    assert (
        bounded_viability.eligible_decision_count
        == viability.eligible_decision_count
    )
    if viability.eligible_decision_count > 1 and not bounded_viability.viable:
        assert bounded_viability.scan_complete is False
    assert set(viability.action_oracles_by_decision).issubset({
        item.decision_id for item in viability.scanned_decisions
    })


def test_cascaded_trace_analysis_preserves_all_decisions_and_confirms_a_bound(root):
    simulator = _simulator(root)
    trace = simulator.simulate(4).final_trace
    bundle = build_bundle(root)
    screening_profile = TraceAnalysisProfileV1.build(
        bid_samples=2,
        play_samples=2,
        exact_allocation_limit=20,
        exact_search_max_remaining=1,
        maximum_sampling_attempts=1_000,
        exact_search_max_nodes=1_000,
        exact_search_max_seconds=0.02,
    )
    confirmation_profile = TraceAnalysisProfileV1.build(
        bid_samples=4,
        play_samples=4,
        exact_allocation_limit=50,
        exact_search_max_remaining=2,
        maximum_sampling_attempts=2_000,
        exact_search_max_nodes=5_000,
        exact_search_max_seconds=0.05,
    )
    screening = StagedTraceAnalyzerV1(
        bundle.rules,
        bundle.convention,
        screening_profile,
        match_value_profile=bundle.component("match_value"),
        minimum_critical_confidence=0.0,
    )
    confirmation = StagedTraceAnalyzerV1(
        bundle.rules,
        bundle.convention,
        confirmation_profile,
        match_value_profile=bundle.component("match_value"),
    )
    profile = CascadedTraceAnalysisProfileV1.build(
        screening_profile,
        confirmation_profile,
        maximum_confirmation_decisions_per_trace=1,
        minimum_screening_score=0.0,
    )
    analyzer = CascadedTraceAnalyzerV1(screening, confirmation, profile)

    result = analyzer.analyze_trace(trace)

    assert len(result.decisions) == len(trace.decision_ledger.records)
    assert len(result.confirmed_decision_ids) == 1
    assert result.profile_hash == profile.sha256
    assert result.screening_profile_hash == screening_profile.sha256
    assert result.confirmation_profile_hash == confirmation_profile.sha256
    document = result.to_dict()
    assert document["schemaVersion"] == 2
    assert TraceAnalysisResultV1.from_dict(
        document,
        ledger=trace.decision_ledger,
    ).sha256 == result.sha256


def test_quality_diversity_selection_is_order_independent_and_globally_deduplicated():
    profile = QualityDiversityProfileV1.build({
        "schemaVersion": 1,
        "cellDimensions": ["decision_kind", "contract"],
        "topKPerCell": 2,
        "rejectedAuditModulus": 1,
    })

    def candidate(marker: str, score: float, fingerprint: str, contract: str = "sun"):
        selection = ContentSelectionScoreV1.build(
            score_family="decision_content_selection",
            profile_hash="a" * 64,
            evidence_hash=marker * 64,
            score=score,
            passed=True,
        )
        return QualityDiversityCandidateV1.build(
            trace_hash=marker * 64,
            decision_id=f"D-{marker}",
            observer_seat="south",
            truth_label=TruthLabel.ROBUST_BEST_KNOWN,
            confidence=0.9,
            structural_fingerprint=fingerprint * 64,
            dimensions={"decision_kind": "play", "contract": contract},
            selection_score=selection,
        )

    values = (
        candidate("1", 80, "a"),
        candidate("2", 90, "a"),
        candidate("3", 70, "b"),
        candidate("4", 60, "c"),
        candidate("5", 50, "d", contract="hokum"),
    )
    selector = QualityDiversitySelectorV1()
    forward = selector.select(values, profile)
    reverse = selector.select(reversed(values), profile)

    assert forward.sha256 == reverse.sha256
    assert {item.decision_id for item in forward.selected} == {"D-2", "D-3", "D-5"}
    assert {item["reason"] for item in forward.rejected} == {
        "duplicate_fingerprint", "cell_top_k"
    }
    assert set(forward.rejected_audit_sample) == {
        item["candidateHash"] for item in forward.rejected
    }

    failed = ContentSelectionScoreV1.build(
        score_family="decision_content_selection",
        profile_hash="a" * 64,
        evidence_hash="f" * 64,
        score=99,
        passed=False,
        rejection_reasons=("confidence",),
    )
    with pytest.raises(ValueError, match="failed truth"):
        QualityDiversityCandidateV1.build(
            trace_hash="f" * 64,
            decision_id="D-failed",
            observer_seat="south",
            truth_label=TruthLabel.ROBUST_BEST_KNOWN,
            confidence=0.9,
            structural_fingerprint="f" * 64,
            dimensions={"decision_kind": "play", "contract": "sun"},
            selection_score=failed,
        )


def test_organic_worker_claims_persists_and_fenced_commits_complete_shard(root, tmp_path):
    simulator = _simulator(root, maximum_deal_attempts=2)
    campaign = SimulationCampaignSpecV1.build(
        rules_hash=simulator.rules.sha256,
        bundle_hash=simulator.bundle_hash,
        engine_version=simulator.engine_version,
        analysis_profile_hash="3" * 64,
        perspective_profile_hash="4" * 64,
        selection_profile_hash="5" * 64,
        agent_release=simulator.agent_release,
        seat_agents=simulator.equal_seat_agents,
        lanes=("organic",),
        lane_profile_hashes={"organic": simulator.bundle_hash},
        seed_start=1,
        seed_count=2,
        seeds_per_shard=2,
    )
    control = SQLiteCampaignControlStore(tmp_path / "campaign.sqlite3", maximum_attempts=2)
    control.create_campaign(campaign)
    artifacts = LocalContentAddressedArtifactStore(tmp_path / "artifacts")
    worker = OrganicFoundryWorkerV1(
        control,
        artifacts,
        simulator,
        worker_id="worker-test",
        lease_seconds=60,
        heartbeat_seconds=10,
    )

    run = worker.run_once(campaign_hash=campaign.sha256)
    assert run.status == "completed" and run.trace_count >= 2
    assert run.artifact_count == run.trace_count
    campaign_row = control.get_campaign(campaign.sha256)
    assert campaign_row is not None and campaign_row["status"] == "completed"
    shard = control.list_shards(campaign.sha256)[0]
    references = shard["result"]["artifactReferences"]
    assert len(references) == run.artifact_count
    for reference in references:
        document = json.loads(artifacts.get(reference["uri"]))
        assert document["sha256"] == reference["ownerHash"]
        assert reference["ownerKind"] == "complete_trace_v1"
    assert worker.run_once(campaign_hash=campaign.sha256).status == "idle"


def test_complete_trace_artifact_round_trip_is_strict_and_hash_verified(root):
    trace = _simulator(root).simulate(2).final_trace
    document = trace.to_dict(trusted=True)
    loaded = CompleteGameTraceV1.from_dict(copy.deepcopy(document))
    assert loaded.sha256 == trace.sha256
    assert loaded.to_dict(trusted=True) == document

    tampered = copy.deepcopy(document)
    tampered["decisionLedger"]["decisions"][0]["actingSeat"] = "south"
    with pytest.raises(ValueError):
        CompleteGameTraceV1.from_dict(tampered)


def test_analysis_funnel_consumes_persisted_trace_and_is_order_independent(root, tmp_path):
    simulator = _simulator(root)
    trace = simulator.simulate(2).final_trace
    artifacts = LocalContentAddressedArtifactStore(tmp_path / "funnel-artifacts")
    encoded = json.dumps(
        trace.to_dict(trusted=True),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    stored = artifacts.put(
        encoded,
        media_type="application/vnd.balootx.complete-trace-v1+json",
        name=f"{trace.sha256}.complete-trace.json",
    )
    trace_reference = {
        "ownerKind": "complete_trace_v1",
        "ownerHash": trace.sha256,
        **stored,
    }
    bundle = build_bundle(root)
    analyzer = StagedTraceAnalyzerV1(
        bundle.rules,
        bundle.convention,
        TraceAnalysisProfileV1.build(
            bid_samples=2,
            play_samples=2,
            exact_allocation_limit=50,
            exact_search_max_remaining=2,
            maximum_sampling_attempts=2_000,
            exact_search_max_nodes=10_000,
            exact_search_max_seconds=0.5,
            exact_search_max_cache_entries=5_000,
        ),
        match_value_profile=bundle.component("match_value"),
    )
    selection_profile = SelectionProfileV1.build({
        "schemaVersion": 1,
        "id": "test_actor_selection",
        "revision": "1",
        "weights": {"outcomeImpact": 1.0, "robustness": 1.0},
        "allowedTruthLabels": ["exact_optimal", "robust_best_known", "debatable"],
        "minimumConfidence": 0.0,
        "minimumScore": 0.0,
    })
    diversity_profile = QualityDiversityProfileV1.build({
        "schemaVersion": 1,
        "cellDimensions": ["decision_kind", "contract"],
        "topKPerCell": 2,
        "rejectedAuditModulus": 10,
    })
    funnel = FoundryAnalysisFunnelV1(
        artifacts,
        analyzer,
        ActualDealOracleVerifierV1(
            bundle.rules,
            maximum_remaining_cards=5,
            maximum_nodes=100_000,
            maximum_seconds=5.0,
        ),
        selection_profile,
        diversity_profile,
        maximum_oracle_decisions_per_trace=1,
    )
    assert isinstance(funnel, AnalysisFunnelPlugin)

    first = funnel.process((trace_reference,))
    second = funnel.process(reversed((trace_reference,)))
    assert first.sha256 == second.sha256
    assert FoundryAnalysisFunnelResultV1.from_dict(first.to_dict()) == first
    assert first.trace_hashes == (trace.sha256,)
    assert len(first.analysis_hashes) == 1
    assert len(first.tactical_verification_hashes) == 1
    assert len(first.verified_laffa_evidence_hashes) == 1
    assert first.selection.sha256 == second.selection.sha256
    assert {
        item["ownerKind"] for item in first.artifact_references
    } == {
        "trace_analysis_v1",
        "tactical_verification_v1",
        "verified_laffa_evidence_v2",
        "surrounding_competence_evidence_v1",
        "quality_diversity_selection_v1",
    }
    for reference in first.artifact_references:
        assert artifacts.get(reference["uri"])
    verified_reference = next(
        item
        for item in first.artifact_references
        if item["ownerKind"] == "verified_laffa_evidence_v2"
    )
    verified_document = json.loads(artifacts.get(verified_reference["uri"]))
    assert verified_document["sha256"] in first.verified_laffa_evidence_hashes
    assert any(
        item["truthMode"] == "actual_deal_oracle"
        for item in verified_document["perspectives"]
    )
    assert "principalVariation" not in json.dumps(verified_document, sort_keys=True)
    assert any(
        edge["relation"] == "post_reveal_verification"
        for edge in verified_document["scenarioGraph"]["edges"]
    )

    class CompletedCampaign:
        status = "completed"

        def get_campaign(self, _campaign_hash):
            return {"status": self.status, "cancelRequested": False}

        def list_shards(self, _campaign_hash):
            return [{
                "status": self.status,
                "result": {
                    "candidateHashes": [trace.sha256],
                    "artifactReferences": [trace_reference],
                },
            }]

    campaign = CompletedCampaign()
    assert funnel.process_completed_campaign(campaign, "f" * 64).sha256 == first.sha256
    campaign.status = "running"
    with pytest.raises(ValueError, match="fully completed"):
        funnel.process_completed_campaign(campaign, "f" * 64)

    bad_reference = {**trace_reference, "ownerHash": "0" * 64}
    with pytest.raises(ValueError, match="owner identity"):
        funnel.process((bad_reference,))


def test_analysis_funnel_requires_verified_causal_evidence_for_play_only(root, tmp_path):
    trace = _simulator(root).simulate(2).final_trace
    bundle = build_bundle(root)
    analyzer = StagedTraceAnalyzerV1(
        bundle.rules,
        bundle.convention,
        TraceAnalysisProfileV1.build(
            bid_samples=2,
            play_samples=2,
            exact_allocation_limit=50,
            exact_search_max_remaining=2,
            maximum_sampling_attempts=2_000,
            exact_search_max_nodes=10_000,
            exact_search_max_seconds=0.5,
            exact_search_max_cache_entries=5_000,
        ),
        match_value_profile=bundle.component("match_value"),
    )
    analysis = analyzer.analyze_trace(trace)
    profile = SelectionProfileV1.build({
        "schemaVersion": 1,
        "id": "verified-tactical-play",
        "revision": "1",
        "weights": {"outcomeImpact": 1.0, "robustness": 1.0},
        "allowedTruthLabels": ["exact_optimal", "robust_best_known", "debatable"],
        "minimumConfidence": 0.0,
        "minimumScore": 0.0,
        "requireVerifiedTacticalPlay": True,
    })
    diversity = QualityDiversityProfileV1.build({
        "schemaVersion": 1,
        "cellDimensions": ["decision_kind", "contract"],
        "topKPerCell": 2,
        "rejectedAuditModulus": 10,
    })
    funnel = FoundryAnalysisFunnelV1(
        LocalContentAddressedArtifactStore(tmp_path / "tactical-gate"),
        analyzer,
        ActualDealOracleVerifierV1(bundle.rules),
        profile,
        diversity,
        maximum_oracle_decisions_per_trace=0,
    )
    records = {item.decision_id: item for item in trace.decision_ledger.records}
    actor = [
        item for item in analysis.laffa_evidence.perspectives
        if item.truth_mode.value == "actor_observer"
    ]
    play = replace(
        next(item for item in actor if records[item.decision_id].decision_kind == "play"),
        forced=False,
        truth_label=TruthLabel.ROBUST_BEST_KNOWN,
        confidence=1.0,
    )
    bid = replace(
        next(item for item in actor if records[item.decision_id].decision_kind == "bid"),
        forced=False,
        truth_label=TruthLabel.ROBUST_BEST_KNOWN,
        confidence=1.0,
    )
    store = TraceDecisionProjectionStoreV1(trace, bundle.rules)
    assert funnel._candidate(trace, store, play, {}, 4, "organic") is None
    assert funnel._candidate(trace, store, bid, {}, 4, "organic") is not None

    verified = SimpleNamespace(
        status="verified",
        oracle=SimpleNamespace(complete=True),
        evidence=(SimpleNamespace(kind=SimpleNamespace(value="only_winning_move")),),
    )
    candidate = funnel._candidate(
        trace,
        store,
        play,
        {play.decision_id: verified},
        4,
        "organic",
    )
    assert candidate is not None
    assert dict(candidate.dimensions)["tactical_detector"] == "only_winning_move"

    strict_profile = SelectionProfileV1.build({
        "schemaVersion": 1,
        "id": "finalist-required",
        "revision": "1",
        "weights": {"outcomeImpact": 1.0, "robustness": 1.0},
        "allowedTruthLabels": ["exact_optimal", "robust_best_known", "debatable"],
        "minimumConfidence": 0.0,
        "minimumScore": 0.0,
        "requireVerifiedTacticalPlay": True,
        "minimumSurroundingQuality": 75.0,
        "minimumSurroundingConfidence": 0.75,
        "minimumSurroundingMateriality": 25.0,
        "maximumSurroundingMaterialMistakes": 0,
        "maximumSurroundingUnverifiedDecisions": 0,
        "maximumSurroundingCatastrophicCrossings": 0,
    })
    qualification = SurroundingCompetenceEvidenceV1.build(
        selection_profile_hash=strict_profile.sha256,
        trace_analysis_hash=analysis.sha256,
        candidate_decision_id=play.decision_id,
        evaluated_decisions=2,
        preceding_decisions=1,
        following_decisions=1,
        minimum_quality=40.0,
        mean_quality=70.0,
        material_mistakes=1,
        unverified_decisions=1,
        catastrophic_crossings=0,
        passed=False,
        rejection_reasons=("material_mistakes", "unverified_decisions"),
    )
    nomination_funnel = FoundryAnalysisFunnelV1(
        LocalContentAddressedArtifactStore(tmp_path / "nomination-gate"),
        analyzer,
        ActualDealOracleVerifierV1(bundle.rules),
        strict_profile,
        diversity,
        maximum_oracle_decisions_per_trace=0,
        allow_pending_surrounding_nominations=True,
    )
    nomination = nomination_funnel._candidate(
        trace,
        store,
        play,
        {play.decision_id: verified},
        4,
        "guided",
        qualification,
    )
    assert nomination is not None
    assert dict(nomination.dimensions)["candidate_scope"] == "deal_replay_nomination"
    assert dict(nomination.dimensions)["surrounding_competence"] == "needs_strong_replay"
