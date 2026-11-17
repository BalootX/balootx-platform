from __future__ import annotations

import json

import pytest

from baloot_platform.challenge.analysis_pipeline import TraceDecisionAnalysisV3
from baloot_platform.challenge.evidence_v3 import (
    DecisionPerspectiveEvidenceV3,
    TruthLabel,
    TruthMode,
)
from baloot_platform.challenge.locator import (
    ChallengeLocatorServiceV1,
    SQLiteChallengeLocatorStore,
)
from baloot_platform.challenge.perspective_pipeline import build_observer_projection
from baloot_platform.challenge.publication import (
    ChallengeAnswerEvidenceV1,
    ChallengeAnswerPolicyV1,
)
from baloot_platform.challenge.replay import TraceDecisionProjectionStoreV1
from baloot_platform.challenge.simulation import NaturalLaffaSimulatorV1
from baloot_platform.config import build_bundle


def _fixture(root, tmp_path):
    bundle = build_bundle(root)
    trace = NaturalLaffaSimulatorV1.from_root(root).simulate(19).final_trace
    decision = next(
        item for item in trace.decision_ledger.records
        if item.decision_kind == "play" and len(item.legal_actions) >= 2
    )
    projection = build_observer_projection(
        TraceDecisionProjectionStoreV1(trace, bundle.rules),
        decision.decision_id,
        decision.acting_seat,
    )
    alternatives = tuple(
        {
            "action": dict(action),
            "rank": index + 1,
            "expectedMargin": 12.0 - index * 5.0,
            "winProbability": 0.72 - index * 0.14,
            "expectedTeamAwardedPoints": 18.0 - index,
            "expectedOpponentAwardedPoints": 6.0 + index,
            "matchWinProbability": 0.65 - index * 0.10,
            "bestShare": 0.80 - index * 0.10,
            "exactShare": 0.0,
        }
        for index, action in enumerate(decision.legal_actions)
    )
    analysis = TraceDecisionAnalysisV3(
        decision.decision_id,
        decision.ordinal,
        decision.acting_seat.name.lower(),
        decision.decision_kind,
        False,
        dict(decision.chosen_action),
        dict(alternatives[0]["action"]),
        alternatives,
        "paired_information_set_test_v1",
        128,
        128.0,
        0.92,
        84.0,
        20.0,
        0.0,
        0.0,
        False,
        (
            "expected_round_margin",
            "expected_team_awarded_points",
            "expected_opponent_awarded_points",
            "round_lead_probability",
            "match_win_probability",
            "best_deal_share",
        ),
        ("kabout_probability",),
        "a" * 64,
    )
    actor = DecisionPerspectiveEvidenceV3(
        decision_id=decision.decision_id,
        acting_seat=decision.acting_seat,
        observer_seat=decision.acting_seat,
        truth_mode=TruthMode.ACTOR_OBSERVER,
        projection_hash=projection.sha256,
        forced=False,
        truth_label=TruthLabel.ROBUST_BEST_KNOWN,
        confidence=0.92,
        evidence={"outcomeImpact": 70.0, "confidence": 92.0},
        recommended_action=dict(alternatives[0]["action"]),
        prompt_contract="choose_acting_action",
    )
    policy = ChallengeAnswerPolicyV1.build({
        "schemaVersion": 1,
        "allowedTruthLabels": ["exact_optimal", "robust_best_known", "debatable"],
        "minimumConfidence": 0.80,
        "acceptedMarginGap": 1.0,
        "acceptedProbabilityGap": 0.02,
        "publicMetrics": [
            "expectedRoundMargin",
            "expectedTeamAwardedPoints",
            "expectedOpponentAwardedPoints",
            "roundLeadProbability",
            "matchWinProbability",
            "bestShare",
        ],
    })
    answer = ChallengeAnswerEvidenceV1.build(
        trace_hash=trace.sha256,
        analysis=analysis,
        actor_evidence=actor,
        policy=policy,
    )
    service = ChallengeLocatorServiceV1(
        SQLiteChallengeLocatorStore(tmp_path / "answers.sqlite3"),
        rules=bundle.rules,
        secret=b"answer-evidence-test-secret-at-least-32-bytes",
        public_base_url="https://play.balootx.test",
        clock=lambda: 1_000,
    )
    return trace, decision, analysis, actor, policy, answer, service


def test_governed_answer_is_hidden_before_commit_and_typed_after(root, tmp_path):
    trace, decision, _analysis, _actor, _policy, answer, service = _fixture(root, tmp_path)
    locator = service.publish(
        trace,
        decision_id=decision.decision_id,
        observer_seat=decision.acting_seat,
        prompt_contract="choose_acting_action",
        answer_evidence=answer,
        answer_explanation="الحركة الوحيدة اللي تحفظ اللفة.",
    )
    public = service.resolve(locator.code)
    encoded = json.dumps(public, sort_keys=True)
    assert public["governedAnswerAvailable"] is True
    assert "recommendation" not in encoded
    assert "acceptedActions" not in encoded
    assert "analysisHash" not in encoded
    assert "الحركة الوحيدة" not in encoded

    alternative = dict(decision.legal_actions[-1])
    attempt = service.commit_attempt(
        locator.code,
        participant_id="publication-guest",
        chosen_action=alternative,
        response_seconds=15.0,
        confidence=0.9,
    )
    reveal = attempt["reveal"]
    assert reveal["status"] == "governed_answer"
    assert reveal["truthLabel"] == "robust_best_known"
    assert reveal["answerMode"] == "single_supported"
    assert reveal["recommendedAction"] == dict(decision.legal_actions[0])
    assert reveal["acceptedActions"] == [dict(decision.legal_actions[0])]
    assert reveal["chosenAccepted"] is (alternative == dict(decision.legal_actions[0]))
    assert reveal["explanation"] == "الحركة الوحيدة اللي تحفظ اللفة."
    assert all("kaboutProbability" not in item for item in reveal["actionEvidence"])
    stats = attempt["statistics"]
    assert stats["recommendedActionPercentage"] in {0.0, 100.0}
    assert stats["engineCognitiveDifficulty"] == 84.0
    assert stats["audienceDifficulty"]["status"] == "not_calibrated"
    assert "traceHash" not in json.dumps(attempt)
    assert "projectionHash" not in json.dumps(attempt)


def test_after_completion_withholds_answer_statistics_on_commit_and_restore(root, tmp_path):
    trace, decision, _analysis, _actor, _policy, answer, service = _fixture(root, tmp_path)
    locator = service.publish(
        trace,
        decision_id=decision.decision_id,
        observer_seat=decision.acting_seat,
        prompt_contract="choose_acting_action",
        reveal_policy="after_completion",
        answer_evidence=answer,
    )
    committed = service.commit_attempt(
        locator.code,
        participant_id="withheld-guest",
        chosen_action=dict(decision.legal_actions[0]),
    )
    restored = service.restore_attempt(locator.code, participant_id="withheld-guest")
    assert restored is not None
    for response in (committed, restored):
        assert response["reveal"]["status"] == "withheld_until_completion"
        stats = response["statistics"]
        assert stats["totalAttempts"] == 1
        assert stats["choices"][0]["count"] == 1
        assert all(not {"evidence", "rank"} & item.keys() for item in stats["choices"])
        assert stats["recommendedActionPercentage"] is None
        assert stats["engineCognitiveDifficulty"] is None
        assert stats["audienceDifficulty"]["status"] == "not_calibrated"


def test_audience_calibration_and_friend_choices_are_exact_and_idempotent(
    root, tmp_path
):
    trace, decision, _analysis, _actor, _policy, answer, base = _fixture(
        root, tmp_path
    )

    class Calibrator:
        def __init__(self):
            self.responses = []

        def record_attempt(self, **value):
            self.responses.append(dict(value))
            return self.recalibrate(
                value["decision_id"],
                engine_difficulty=value["engine_difficulty"],
            )

        def recalibrate(self, decision_id, *, engine_difficulty):
            players = {item["player_id"] for item in self.responses}
            public = len(self.responses) >= 2 and len(players) >= 2
            return {
                "challengeId": decision_id,
                "responseCount": len(self.responses),
                "distinctPlayers": len(players),
                "standardError": 2.5,
                "calibratedDifficulty": 78.0,
                "label": "expert",
                "discrimination": 1.25,
                "public": public,
            }

    calibrator = Calibrator()
    friends = {"player-a": ("player-b",), "player-b": ("player-a",)}
    service = ChallengeLocatorServiceV1(
        SQLiteChallengeLocatorStore(tmp_path / "calibrated-locators.sqlite3"),
        rules=base.rules,
        secret=b"calibrated-answer-secret-at-least-32-bytes",
        public_base_url="https://play.balootx.test",
        clock=lambda: 1_000,
        difficulty_calibrator=calibrator,
        friend_ids=lambda subject: friends.get(subject, ()),
    )
    locator = service.publish(
        trace,
        decision_id=decision.decision_id,
        observer_seat=decision.acting_seat,
        prompt_contract="choose_acting_action",
        answer_evidence=answer,
    )
    action = dict(decision.legal_actions[0])
    first = service.commit_attempt(
        locator.code,
        participant_id="player-a",
        chosen_action=action,
        response_seconds=8.0,
        confidence=0.9,
        include_friend_statistics=True,
        calibration_eligible=True,
    )
    assert first["statistics"]["audienceDifficulty"]["status"] == "not_calibrated"
    assert first["statistics"]["friends"]["totalAttempts"] == 0

    second = service.commit_attempt(
        locator.code,
        participant_id="player-b",
        chosen_action=action,
        response_seconds=10.0,
        confidence=0.8,
        include_friend_statistics=True,
        calibration_eligible=True,
    )
    audience = second["statistics"]["audienceDifficulty"]
    assert audience["status"] == "calibrated"
    assert audience["audienceCalibratedDifficulty"] == 78.0
    assert audience["label"] == "expert"
    assert second["statistics"]["friends"]["totalAttempts"] == 1

    retry = service.commit_attempt(
        locator.code,
        participant_id="player-a",
        chosen_action=action,
        response_seconds=8.0,
        confidence=0.9,
        include_friend_statistics=True,
        calibration_eligible=True,
    )
    assert retry["attemptId"] == first["attemptId"]
    assert len(calibrator.responses) == 2
    assert retry["statistics"]["friends"]["totalAttempts"] == 1


def test_answer_evidence_rejects_unresolved_mismatch_and_false_debate(root, tmp_path):
    trace, _decision, analysis, actor, policy, _answer, _service = _fixture(root, tmp_path)
    unresolved = DecisionPerspectiveEvidenceV3(
        decision_id=actor.decision_id,
        acting_seat=actor.acting_seat,
        observer_seat=actor.observer_seat,
        truth_mode=actor.truth_mode,
        projection_hash=actor.projection_hash,
        forced=False,
        truth_label=TruthLabel.UNRESOLVED,
        confidence=actor.confidence,
        evidence=actor.evidence,
        recommended_action=actor.recommended_action,
        prompt_contract=actor.prompt_contract,
    )
    with pytest.raises(ValueError, match="not publishable"):
        ChallengeAnswerEvidenceV1.build(
            trace_hash=trace.sha256,
            analysis=analysis,
            actor_evidence=unresolved,
            policy=policy,
        )

    debate = DecisionPerspectiveEvidenceV3(
        decision_id=actor.decision_id,
        acting_seat=actor.acting_seat,
        observer_seat=actor.observer_seat,
        truth_mode=actor.truth_mode,
        projection_hash=actor.projection_hash,
        forced=False,
        truth_label=TruthLabel.DEBATABLE,
        confidence=actor.confidence,
        evidence=actor.evidence,
        recommended_action=actor.recommended_action,
        prompt_contract=actor.prompt_contract,
    )
    with pytest.raises(ValueError, match="multiple supported"):
        ChallengeAnswerEvidenceV1.build(
            trace_hash=trace.sha256,
            analysis=analysis,
            actor_evidence=debate,
            policy=policy,
        )

    with pytest.raises(ValueError, match="unknown"):
        ChallengeAnswerPolicyV1.build({
            **policy.to_dict(),
            "sha256": policy.sha256,
        })


def test_locator_rejects_answer_bound_to_another_trace(root, tmp_path):
    trace, decision, _analysis, _actor, _policy, answer, service = _fixture(root, tmp_path)
    other = NaturalLaffaSimulatorV1.from_root(root).simulate(20).final_trace
    with pytest.raises(ValueError, match="trace identity"):
        service.publish(
            other,
            decision_id=other.decision_ledger.records[0].decision_id,
            observer_seat=other.decision_ledger.records[0].acting_seat,
            prompt_contract="choose_acting_action",
            answer_evidence=answer,
        )
    assert decision.decision_id == answer.decision_id
