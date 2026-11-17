from __future__ import annotations

import pytest

from baloot_platform.challenge.evidence_v3 import (
    DecisionPerspectiveEvidenceV3,
    LaffaEvidenceV2,
    TruthLabel,
    TruthMode,
    canonical_sha256,
)
from baloot_platform.challenge.profiles import ContentSelectionScoreV1
from baloot_platform.challenge.sakka_context import NaturalSakkaV1
from baloot_platform.challenge.sakka_evidence import (
    NaturalSakkaEvidenceV1,
    SakkaSelectionProfileV1,
)
from baloot_platform.challenge.simulation import NaturalLaffaSimulatorV1
from baloot_platform.core.enums import Seat


def _natural_match(root):
    simulator = NaturalLaffaSimulatorV1.from_root(root, maximum_deal_attempts=8)
    traces = []
    score = (0, 0)
    dealer = Seat.EAST
    round_number = 1
    seed = 70
    while max(score) < 152 and len(traces) < 20:
        trace = simulator.simulate(
            seed,
            dealer=dealer,
            match_score=score,
            round_number=round_number,
        ).final_trace
        traces.append(trace)
        score = tuple(trace.trusted_final_state["matchScore"])
        dealer = Seat.parse(trace.trusted_final_state["dealer"]).next()
        round_number = int(trace.trusted_final_state["roundNumber"]) + 1
        seed += 1
    return tuple(traces), NaturalSakkaV1.build(traces, target=152)


def _round_evidence(trace, index):
    seat = Seat.NORTH if index % 2 == 0 else Seat.EAST
    decision_id = f"critical-{index}"
    perspective = DecisionPerspectiveEvidenceV3(
        decision_id=decision_id,
        acting_seat=seat,
        observer_seat=seat,
        truth_mode=TruthMode.ACTOR_OBSERVER,
        projection_hash=canonical_sha256({"trace": trace.sha256, "seat": seat.name}),
        forced=False,
        truth_label=TruthLabel.ROBUST_BEST_KNOWN,
        confidence=0.90,
        evidence={"outcomeImpact": 80.0, "confidence": 90.0},
        recommended_action={"testAction": index},
        prompt_contract="choose_acting_action",
    )
    digest = canonical_sha256({"kind": "test-laffa-evidence", "trace": trace.sha256})
    return LaffaEvidenceV2(
        trace.game_id,
        trace.decision_ledger.sha256,
        (perspective,),
        (),
        (),
        (decision_id,),
        1,
        80.0,
        None,
        80.0,
        1.0,
        1,
        "single_perspective_masterpiece",
        0.75,
        None,
        None,
        digest,
    )


def test_natural_sakka_evidence_scores_complete_match_without_peak_only_bias(root):
    traces, sakka = _natural_match(root)
    assert len(traces) >= 2
    evidence = {
        trace.sha256: _round_evidence(trace, index)
        for index, trace in enumerate(traces)
    }
    scores = {}
    for index, trace in enumerate(traces):
        quality = 85.0 - min(index, 5)
        scores[trace.sha256] = ContentSelectionScoreV1.build(
            score_family="laffa_content_selection",
            profile_hash="b" * 64,
            evidence_hash=evidence[trace.sha256].sha256,
            score=quality,
            passed=True,
        )
    result = NaturalSakkaEvidenceV1.build(
        sakka,
        traces=traces,
        laffa_evidence=evidence,
        laffa_selection_scores=scores,
    )
    assert result.round_count == len(traces)
    assert result.minimum_laffa_quality == min(item.score for item in scores.values())
    assert result.peak_laffa_quality == max(item.score for item in scores.values())
    assert result.mean_laffa_quality < result.peak_laffa_quality
    assert result.team_opportunity_coverage == 100.0
    assert result.perspective_coverage == 50.0
    assert result.critical_decision_count == len(traces)

    profile = SakkaSelectionProfileV1.build({
        "schemaVersion": 1,
        "id": "natural_sakka_editorial_v1",
        "revision": "1",
        "weights": {
            "minimumLaffaQuality": 3.0,
            "meanLaffaQuality": 2.0,
            "matchTension": 1.0,
            "finalRoundConsequence": 1.0,
        },
        "minimumRounds": 2,
        "minimumLaffaQuality": 70.0,
        "minimumFinalRoundQuality": 75.0,
        "requireBothTeamOpportunities": True,
        "minimumScore": 60.0,
    })
    ranked = profile.score(result, final_round_quality=scores[traces[-1].sha256].score)
    assert ranked.score_family == "sakka_content_selection"
    assert ranked.passed

    strict = SakkaSelectionProfileV1.build({
        **{key: value for key, value in profile.to_dict().items() if key != "sha256"},
        "minimumLaffaQuality": 95.0,
    })
    rejected = strict.score(result, final_round_quality=100.0)
    assert not rejected.passed
    assert "filler_laffa" in rejected.rejection_reasons


def test_sakka_evidence_rejects_missing_or_mixed_round_authority(root):
    traces, sakka = _natural_match(root)
    evidence = {
        trace.sha256: _round_evidence(trace, index)
        for index, trace in enumerate(traces)
    }
    scores = {
        trace.sha256: ContentSelectionScoreV1.build(
            score_family="laffa_content_selection",
            profile_hash=("b" * 64 if index == 0 else "c" * 64),
            evidence_hash=evidence[trace.sha256].sha256,
            score=80.0,
            passed=True,
        )
        for index, trace in enumerate(traces)
    }
    if len(traces) > 1:
        with pytest.raises(ValueError, match="one selection profile"):
            NaturalSakkaEvidenceV1.build(
                sakka,
                traces=traces,
                laffa_evidence=evidence,
                laffa_selection_scores=scores,
            )
    missing = dict(evidence)
    missing.pop(traces[0].sha256)
    with pytest.raises(ValueError, match="one Laffa evidence"):
        NaturalSakkaEvidenceV1.build(
            sakka,
            traces=traces,
            laffa_evidence=missing,
            laffa_selection_scores={
                key: ContentSelectionScoreV1.build(
                    score_family="laffa_content_selection",
                    profile_hash="b" * 64,
                    evidence_hash=value.sha256,
                    score=80.0,
                    passed=True,
                )
                for key, value in evidence.items()
            },
        )
