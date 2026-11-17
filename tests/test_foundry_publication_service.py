from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from baloot_platform.analysis.bidding import BidChoice, BidChoiceEvaluation
from baloot_platform.analysis.decision_scores import (
    BID_DECISION_OBJECTIVE_V1,
    BID_DECISION_TIE_BREAK_V1,
    PLAY_DECISION_TIE_BREAK_V1,
    PRIMARY_DECISION_COHORT_SEED_POLICY_V1,
    bid_score_rows_v1,
    legal_action_set_hash_v1,
    play_score_rows_v1,
    primary_decision_score_confidence_v1,
    primary_decision_score_evidence_v1,
)
from baloot_platform.analysis.models import ActionEvaluation
from baloot_platform.challenge.analysis_pipeline import (
    StagedTraceAnalyzerV1,
    TraceAnalysisResultV1,
    TraceDecisionAnalysisV3,
    _decision_seed,
    _forced_legality_proof_v1,
)
from baloot_platform.challenge.evidence_v3 import LaffaEvidenceV2, canonical_sha256
from baloot_platform.challenge.finalist_competence import (
    canonical_surrounding_score_projection_v1,
)
from baloot_platform.challenge.locator import ChallengeLocatorServiceV1, SQLiteChallengeLocatorStore
from baloot_platform.challenge.perspective_pipeline import build_decision_perspective_evidence
from baloot_platform.challenge.publication import ChallengeAnswerPolicyV1
from baloot_platform.challenge.publication_service import ChallengePublicationServiceV1
from baloot_platform.challenge.replay import TraceDecisionProjectionStoreV1
from baloot_platform.challenge.simulation import NaturalLaffaSimulatorV1
from baloot_platform.config import build_bundle
from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Suit
from baloot_platform.storage.artifact_store import LocalContentAddressedArtifactStore


def test_exhaustive_population_confidence_uses_verified_search_coverage():
    alternatives = (
        {"exactShare": 26 / 27, "searchCompletionStatus": "exact"},
        {"exactShare": 1.0, "searchCompletionStatus": "exact"},
    )
    confidence = StagedTraceAnalyzerV1._confidence(
        27.0,
        kind="play",
        method="exhaustive_constrained_hidden_deals+hard_facts_only",
        alternatives=alternatives,
    )

    assert confidence == pytest.approx(26 / 27)
    assert confidence < 1.0


def test_exhaustive_population_confidence_separates_coverage_from_heuristic_truth():
    alternatives = (
        {"exactShare": 0.0, "searchCompletionStatus": "heuristic_complete"},
        {"exactShare": 0.0, "searchCompletionStatus": "heuristic_complete"},
    )

    confidence = StagedTraceAnalyzerV1._confidence(
        19.0,
        kind="play",
        method="exhaustive_constrained_hidden_deals+hard_facts_only",
        alternatives=alternatives,
    )

    assert confidence == 1.0


def _analysis_for_trace(trace, rules):
    projection_store = TraceDecisionProjectionStoreV1(trace, rules)
    snapshots = {
        item.decision.decision_id: item
        for item in projection_store.replay.decisions
    }
    analyses = []
    perspectives = []
    for record in trace.decision_ledger.records:
        evaluations = ()
        if not record.forced and record.decision_kind == "bid":
            evaluations = tuple(
                BidChoiceEvaluation(
                    BidChoice(
                        str(action["action"]),
                        (
                            Suit.parse(str(action["trumpSuit"]))
                            if action.get("trumpSuit") is not None
                            else None
                        ),
                    ),
                    12.0 - index * 4.0,
                    max(0.05, 0.75 - index * 0.08),
                    max(0.05, 0.70 - index * 0.07),
                    -float(index),
                    2.0 + index,
                    max(0.05, 0.8 - index * 0.08),
                    64,
                )
                for index, action in enumerate(record.legal_actions)
            )
            score_rows = bid_score_rows_v1(evaluations)
        elif not record.forced:
            evaluations = tuple(
                ActionEvaluation(
                    card=Card.parse(dict(action)),
                    expected_round_margin=12.0 - index * 4.0,
                    expected_team_awarded_points=26.0 - index * 2.0,
                    expected_opponent_awarded_points=14.0 + index * 2.0,
                    round_lead_probability=max(0.05, 0.75 - index * 0.08),
                    draw_probability=0.0,
                    confidence_low=0.55,
                    confidence_high=0.90,
                    best_share=max(0.05, 0.8 - index * 0.08),
                    samples=64,
                    effective_samples=64.0,
                    exact_share=0.0,
                    contract_success_probability=max(0.05, 0.70 - index * 0.07),
                    match_win_probability=max(0.05, 0.72 - index * 0.07),
                    match_confidence_low=0.50,
                    match_confidence_high=0.88,
                    deal_margins=(12.0 - index * 4.0,) * 64,
                    downside_cvar=-float(index),
                    volatility=2.0 + index,
                    search_completion_status="heuristic_complete",
                )
                for index, action in enumerate(record.legal_actions)
            )
            score_rows = play_score_rows_v1(evaluations)
        else:
            score_rows = ()
        alternatives = tuple(
            {
                "action": dict(action),
                "rank": index + 1,
                "expectedMargin": 12.0 - index * 4.0,
                "winProbability": 0.75 - index * 0.12,
                "contractSuccessProbability": 0.70 - index * 0.10,
                "downsideCvar10": -float(index),
                "volatility": 2.0 + index,
                "bestShare": 0.8 - index * 0.1,
            }
            for index, action in enumerate(record.legal_actions)
        )
        effective = 100.0 if record.forced else 64.0
        if record.forced:
            score_evidence = None
            forced_proof = _forced_legality_proof_v1(
                snapshots[record.decision_id],
                rules_hash=rules.sha256,
            )
            best_action = dict(alternatives[0]["action"])
            confidence = StagedTraceAnalyzerV1._confidence(
                effective,
                kind=record.decision_kind,
            )
            surrounding_projection = {
                "expectedRegret": 0.0,
                "probabilityDrop": 0.0,
                "catastrophicCrossing": False,
            }
        else:
            score_evidence = primary_decision_score_evidence_v1(
                decision_kind=record.decision_kind,
                actor=record.acting_seat,
                public_prefix_hash=record.public_prefix_hash,
                rules_hash=rules.sha256,
                analysis_identity_hash="c" * 64,
                trace_analysis_profile_hash="b" * 64,
                cohort_seed_policy=PRIMARY_DECISION_COHORT_SEED_POLICY_V1,
                cohort_index=0,
                cohort_seed=_decision_seed(
                    record,
                    "c" * 64,
                    seed_namespace="",
                    seed_offset=0,
                ),
                objective=(
                    BID_DECISION_OBJECTIVE_V1
                    if record.decision_kind == "bid"
                    else "match_win_probability"
                ),
                tie_break=(
                    BID_DECISION_TIE_BREAK_V1
                    if record.decision_kind == "bid"
                    else PLAY_DECISION_TIE_BREAK_V1
                ),
                backend="python_reference_exact_search_v1",
                method="paired_actor_observer_test_v1",
                hidden_deal_support=64,
                effective_sample_size=effective,
                recommended_action=dict(score_rows[0]["action"]),
                rows=score_rows,
                legal_action_set_hash=legal_action_set_hash_v1(
                    record.legal_actions
                ),
            )
            forced_proof = None
            alternatives = tuple(
                {
                    **dict(alternatives[index]),
                    "action": dict(row["action"]),
                    "rank": int(row["rank"]),
                    "expectedMargin": float(row["expectedRoundMargin"]),
                    "winProbability": float(row["roundLeadProbability"]),
                    "contractSuccessProbability": float(
                        row["contractSuccessProbability"]
                    ),
                    "downsideCvar10": float(row["downsideCvar10"]),
                    "volatility": float(row["volatility"]),
                    "bestShare": float(row["bestShare"]),
                }
                for index, row in enumerate(score_evidence.rows)
            )
            best_action = dict(score_evidence.recommended_action)
            confidence = primary_decision_score_confidence_v1(score_evidence)
            surrounding_projection = canonical_surrounding_score_projection_v1(
                score_evidence,
                actual_action=dict(record.chosen_action),
                best_action=best_action,
            )
        core = {
            "schemaVersion": 3,
            "decisionId": record.decision_id,
            "ordinal": record.ordinal,
            "actingSeat": record.acting_seat.name.lower(),
            "decisionKind": record.decision_kind,
            "forced": record.forced,
            "actualAction": dict(record.chosen_action),
            "bestAction": best_action,
            "alternatives": list(alternatives),
            "method": "paired_actor_observer_test_v1",
            "hiddenDeals": 64,
            "effectiveSampleSize": effective,
            "difficulty": 82.0,
            "debateScore": 0.0,
            "expectedRegret": float(surrounding_projection["expectedRegret"]),
            "probabilityDrop": float(surrounding_projection["probabilityDrop"]),
            "catastrophicCrossing": bool(
                surrounding_projection["catastrophicCrossing"]
            ),
            "supportedMetrics": [
                "best_deal_share", "contract_success_probability",
                "downside_cvar", "expected_round_margin",
                "round_lead_probability", "volatility",
            ],
            "unsupportedMetrics": ["match_win_probability"],
            "primaryDecisionScoreEvidenceHash": (
                score_evidence.sha256 if score_evidence is not None else None
            ),
            "forcedLegalityProofHash": (
                forced_proof["sha256"] if forced_proof is not None else None
            ),
        }
        analysis = TraceDecisionAnalysisV3(
            record.decision_id,
            record.ordinal,
            record.acting_seat.name.lower(),
            record.decision_kind,
            record.forced,
            dict(record.chosen_action),
            best_action,
            alternatives,
            core["method"],
            core["hiddenDeals"],
            effective,
            confidence,
            core["difficulty"],
            core["debateScore"],
            core["expectedRegret"],
            core["probabilityDrop"],
            core["catastrophicCrossing"],
            tuple(core["supportedMetrics"]),
            tuple(core["unsupportedMetrics"]),
            canonical_sha256(core),
            score_evidence,
            forced_proof,
        )
        analyses.append(analysis)
        perspectives.extend(build_decision_perspective_evidence(
            projection_store,
            {
                "decisionId": record.decision_id,
                "actor": record.acting_seat.name.lower(),
                "legalActions": [dict(item) for item in record.legal_actions],
            },
            analysis.audit_mapping(),
        ))
    laffa = LaffaEvidenceV2.build(
        trace.decision_ledger,
        perspectives,
        critical_threshold=70.0,
        minimum_critical_confidence=0.75,
        require_four_projections=True,
    )
    core = {
        "schemaVersion": 1,
        "traceHash": trace.sha256,
        "profileHash": "b" * 64,
        "analyzerId": "test_actor_analyzer",
        "analyzerVersion": "1.0.0",
        "decisionAnalysisHashes": [item.sha256 for item in analyses],
        "laffaEvidenceHash": laffa.sha256,
    }
    return TraceAnalysisResultV1(
        trace.sha256,
        core["profileHash"],
        core["analyzerId"],
        core["analyzerVersion"],
        tuple(analyses),
        laffa,
        canonical_sha256(core),
    )


def _policy():
    return ChallengeAnswerPolicyV1.build({
        "schemaVersion": 1,
        "allowedTruthLabels": ["exact_optimal", "robust_best_known", "debatable"],
        "minimumConfidence": 0.75,
        "acceptedMarginGap": 0.0,
        "acceptedProbabilityGap": 0.0,
        "publicMetrics": [
            "expectedRoundMargin", "roundLeadProbability",
            "contractSuccessProbability", "downsideCvar10",
            "volatility", "bestShare",
        ],
    })


def test_publication_bridge_loads_verified_analysis_and_hides_answer(root, tmp_path):
    bundle = build_bundle(root)
    trace = NaturalLaffaSimulatorV1.from_root(root).simulate(19).final_trace
    analysis = _analysis_for_trace(trace, bundle.rules)
    parsed = TraceAnalysisResultV1.from_dict(
        analysis.to_dict(),
        ledger=trace.decision_ledger,
    )
    assert parsed.sha256 == analysis.sha256
    artifact_store = LocalContentAddressedArtifactStore(tmp_path / "artifacts")
    reference = artifact_store.put(
        json.dumps(
            analysis.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8"),
        media_type="application/vnd.balootx.trace-analysis-v1+json",
        name=f"{analysis.sha256}.trace-analysis.json",
    )
    reference = {
        "ownerKind": "trace_analysis_v1",
        "ownerHash": analysis.sha256,
        **reference,
    }
    locator_service = ChallengeLocatorServiceV1(
        SQLiteChallengeLocatorStore(tmp_path / "locators.sqlite3"),
        rules=bundle.rules,
        secret=b"publication-bridge-secret-at-least-32-bytes",
        public_base_url="https://play.balootx.test",
        clock=lambda: 1_000,
    )
    player_names = dict(bundle.component("video")["identity"]["default_player_names"])
    publisher = ChallengePublicationServiceV1(
        artifact_store,
        locator_service,
        _policy(),
        player_names=player_names,
    )
    decision_id = analysis.laffa_evidence.critical_decision_ids[0]
    published = publisher.publish_actor_challenge(
        trace,
        analysis_reference=reference,
        decision_id=decision_id,
    )
    public = locator_service.resolve(published.locator.code)
    encoded = json.dumps(public, sort_keys=True)
    assert public["governedAnswerAvailable"] is True
    assert public["playerNames"] == player_names
    assert "recommendation" not in encoded
    assert "traceHash" not in encoded
    trusted = published.to_dict(trusted=True)
    assert trusted["provenance"]["analysisHash"] == analysis.sha256


def test_publication_bridge_rejects_tampered_or_noncritical_artifacts(root, tmp_path):
    bundle = build_bundle(root)
    trace = NaturalLaffaSimulatorV1.from_root(root).simulate(19).final_trace
    analysis = _analysis_for_trace(trace, bundle.rules)
    artifact_store = LocalContentAddressedArtifactStore(tmp_path / "artifacts")
    reference = artifact_store.put(
        json.dumps(analysis.to_dict(), sort_keys=True, separators=(",", ":")).encode(),
        media_type="application/vnd.balootx.trace-analysis-v1+json",
    )
    reference = {"ownerKind": "trace_analysis_v1", "ownerHash": analysis.sha256, **reference}
    locator_service = ChallengeLocatorServiceV1(
        SQLiteChallengeLocatorStore(tmp_path / "locators.sqlite3"),
        rules=bundle.rules,
        secret=b"publication-bridge-secret-at-least-32-bytes",
        public_base_url="https://play.balootx.test",
    )
    publisher = ChallengePublicationServiceV1(artifact_store, locator_service, _policy())
    with pytest.raises(ValueError, match="checksum"):
        publisher.publish_actor_challenge(
            trace,
            analysis_reference={**reference, "sha256": "0" * 64},
            decision_id=analysis.laffa_evidence.critical_decision_ids[0],
        )
    forced = next(
        item.decision_id for item in trace.decision_ledger.records if item.forced
    )
    with pytest.raises(ValueError, match="verified critical"):
        publisher.publish_actor_challenge(
            trace,
            analysis_reference=reference,
            decision_id=forced,
        )


@pytest.mark.parametrize(
    ("analysis", "message"),
    (
        (
            SimpleNamespace(
                analyzer_id="adaptive_finalist_trace_analyzer_v1",
                analyzer_version="1.2.0",
                unresolved_decision_ids=(),
            ),
            "historical adaptive finalist analysis is publication-ineligible",
        ),
        (
            SimpleNamespace(
                analyzer_id="test_actor_analyzer",
                analyzer_version="1.0.0",
                unresolved_decision_ids=("decision-unresolved",),
            ),
            "an unresolved decision cannot be published",
        ),
    ),
)
def test_publication_rejects_historical_adaptive_or_unresolved_analysis(
    root,
    tmp_path,
    monkeypatch,
    analysis,
    message,
):
    bundle = build_bundle(root)
    trace = NaturalLaffaSimulatorV1.from_root(root).simulate(19).final_trace
    publisher = ChallengePublicationServiceV1(
        LocalContentAddressedArtifactStore(tmp_path / "artifacts"),
        ChallengeLocatorServiceV1(
            SQLiteChallengeLocatorStore(tmp_path / "locators.sqlite3"),
            rules=bundle.rules,
            secret=b"historical-publication-test-secret",
            public_base_url="https://play.balootx.test",
        ),
        _policy(),
    )
    decision_id = (
        "decision-unresolved"
        if analysis.unresolved_decision_ids
        else trace.decision_ledger.records[0].decision_id
    )
    monkeypatch.setattr(
        publisher,
        "_load_analysis",
        lambda *_args, **_kwargs: analysis,
    )

    with pytest.raises(ValueError, match=message):
        publisher.publish_actor_challenge(
            trace,
            analysis_reference={},
            decision_id=decision_id,
        )
