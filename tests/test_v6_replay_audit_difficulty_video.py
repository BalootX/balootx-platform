from __future__ import annotations

from pathlib import Path

from baloot_platform.audit import OnlineDifficultyCalibrator, ReplayAuditService
from baloot_platform.config import build_bundle
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, Seat
from baloot_platform.generated import platform_pb2
from baloot_platform.learning.store import LearningStore


class _FakeLearning:
    def __init__(self, store: LearningStore, result: dict):
        self.store = store
        self.result = result
        self.calls = 0

    def reanalyze_decision(self, decision_id: str, **kwargs):
        self.calls += 1
        return self.store.add_analysis(
            decision_id,
            engine_version="6.0.0-test",
            bundle_hash="b" * 64,
            config={
                "samples": kwargs.get("samples", 96),
                "exactAllocationLimit": kwargs.get("exact_allocation_limit", 20000),
                "exactSearchRemaining": kwargs.get("exact_search_remaining", 10),
                "useConventions": kwargs.get("use_conventions", False),
            },
            result=self.result,
        )


def _play_decision(root: Path, store: LearningStore, *, legal_count: int = 2, source: str = "human"):
    bundle = build_bundle(root)
    engine = BalootEngine(bundle.rules)
    state = engine.create_round(seed=611, dealer=Seat.EAST)
    engine.shuffle_and_initial_deal(state)
    engine.bid_and_complete_deal(state, forced_contract=Contract.SUN, forced_buyer=Seat.NORTH)
    engine.prepare_projects(state)
    legal = list(engine.legal_moves(state))[:legal_count]
    assert legal
    selected = legal[0]
    decision = store.record_decision(
        game_id="GAME-V6",
        game_version=1,
        mode="full",
        state=state,
        actor=state.current_player,
        decision_kind="play",
        legal_actions=legal,
        selected_action=selected,
        decision_metadata={"source": source, "strategyId": "test-policy" if source == "bot" else None},
    )
    return decision, legal


def test_forced_decision_audit_skips_counterfactual_search(root: Path, tmp_path: Path):
    store = LearningStore(tmp_path / "learning.sqlite3")
    decision, legal = _play_decision(root, store, legal_count=1)
    learning = _FakeLearning(store, {"actions": []})
    audit = ReplayAuditService(root, store, learning).audit_decision(decision["decisionId"], force=True)
    assert audit["forced"] is True
    assert audit["qualityLabel"] == "forced"
    assert audit["engineMethod"] == "forced_legal_action"
    assert audit["bestAction"]["code"] == legal[0].code
    assert learning.calls == 0


def test_replay_audit_separates_effect_from_human_intent(root: Path, tmp_path: Path):
    store = LearningStore(tmp_path / "learning.sqlite3")
    decision, legal = _play_decision(root, store, legal_count=2)
    selected, best = legal[0], legal[1]
    result = {
        "method": "paired_counterfactual_test",
        "hiddenDeals": 128,
        "effectiveSampleSize": 112.0,
        "difficulty": 72.0,
        "debateScore": 28.0,
        "deductions": [{"kind": "void", "seat": "west", "suit": "hearts", "source": "trick-2"}],
        "signals": [{"kind": "low_then_high", "relationship": "partner", "forcedness": "meaningful_choice", "confidence": 0.55}],
        "actions": [
            {
                "card": selected.to_dict(), "expectedMatchMargin": -11.0, "winProbability": 0.31,
                "confidence95": [0.25, 0.37], "bestShare": 0.08, "exactShare": 1.0,
                "distribution": {"downsideCvar10": -35.0, "volatility": 17.0},
            },
            {
                "card": best.to_dict(), "expectedMatchMargin": 13.0, "winProbability": 0.73,
                "confidence95": [0.67, 0.79], "bestShare": 0.92, "exactShare": 1.0,
                "distribution": {"downsideCvar10": -4.0, "volatility": 8.0},
            },
        ],
    }
    learning = _FakeLearning(store, result)
    service = ReplayAuditService(root, store, learning)
    audit = service.audit_decision(decision["decisionId"], force=True)
    assert audit["bestAction"]["code"] == best.code
    assert audit["actualAction"]["code"] == selected.code
    assert audit["catastrophicCrossing"] is True
    assert audit["qualityLabel"] == "blunder"
    assert audit["actualIntent"]["status"] == "unknown"
    assert any(reason["code"] == "HUMAN_INTENT_UNKNOWN" for reason in audit["reasons"])
    arabic_reasons = " ".join(reason["detailAr"] for reason in audit["reasons"])
    assert "west" not in arabic_reasons
    assert "hearts" not in arabic_reasons
    assert "low_then_high" not in arabic_reasons
    assert "meaningful_choice" not in arabic_reasons
    assert selected.code not in arabic_reasons
    assert best.code not in arabic_reasons
    assert "غرب" in arabic_reasons
    assert "♥" in arabic_reasons

    store.add_feedback(
        decision["decisionId"],
        preferred_action=selected.to_dict(),
        engine_action=best.to_dict(),
        confidence=0.9,
        rationale="I expected my partner to regain the lead.",
        source="player",
        reviewer="player-1",
    )
    reviewed = service.audit_decision(decision["decisionId"], force=True)
    assert reviewed["actualIntent"]["status"] == "reviewed_statement"
    assert reviewed["actualIntent"]["source"] == "player"
    comparison = service.compare_decision_versions(decision["decisionId"])
    assert comparison["current"]["auditId"] == reviewed["auditId"]


def test_difficulty_is_replayable_and_adaptive_delivery_only_uses_published_challenges(root: Path, tmp_path: Path):
    store = LearningStore(tmp_path / "difficulty.sqlite3")
    profile = build_bundle(root).component("difficulty")
    calibrator = OnlineDifficultyCalibrator(store, profile)

    for challenge_id, score, status in (("CHL-PUBLISHED", 70.0, "published"), ("CHL-DRAFT", 30.0, "candidate")):
        store.put_challenge(
            challenge_id=challenge_id,
            decision_id=None,
            fingerprint=challenge_id.lower(),
            archetype="cliff",
            publishability=0.9,
            payload={"difficulty": score},
            episode=None,
            status=status,
        )
        store.upsert_difficulty_item({
            "challengeId": challenge_id,
            "engineDifficulty": score,
            "latentDifficulty": (score - 50.0) / 15.0,
            "discrimination": 1.0,
            "responseCount": 0,
            "distinctPlayers": 0,
            "standardError": 15.0,
            "calibratedScore": score,
            "label": "hard" if score >= 55 else "medium",
            "updatedAt": "2026-01-01T00:00:00+00:00",
        })

    for index in range(12):
        calibrator.record_attempt(
            decision_id="CHL-PUBLISHED",
            player_id=f"player-{index}",
            chosen_action={"kind": "scalar", "value": "A" if index < 4 else "B"},
            accepted_actions=[{"kind": "scalar", "value": "A"}],
            engine_difficulty=70.0,
            response_seconds=6.0 + index,
            confidence=0.7,
            reported_skill=50.0,
        )
    first = calibrator.recalibrate("CHL-PUBLISHED", engine_difficulty=70.0)
    second = calibrator.recalibrate("CHL-PUBLISHED", engine_difficulty=70.0)
    for key in ("engineDifficulty", "latentDifficulty", "discrimination", "calibratedScore", "responseCount", "observedSuccessRate"):
        assert first[key] == second[key]
    assert first["engineDifficulty"] == 70.0
    assert store.get_difficulty_player("player-1") is None  # external IDs are never stored directly
    next_item = calibrator.adaptive_next("player-1")
    assert next_item is not None and next_item["challengeId"] == "CHL-PUBLISHED"


def test_generated_grpc_contract_contains_replay_audit_difficulty_and_video_methods():
    service = platform_pb2.DESCRIPTOR.services_by_name["BalootPlatform"]
    names = {method.name for method in service.methods}
    assert {
        "GetReplay", "ReopenGame", "ForkDecision", "AuditGame", "ExplainDecision", "CompareDecision",
        "GetDecisionDifficulty", "RecordDecisionAttempt",
        "GetChallengeDifficulty", "RecordChallengeAttempt", "GetAdaptiveChallenge",
        "GetVideoRecord", "GetVideoTimeline", "GetVideoAuditMap", "RenderGameReplay",
        "CreateRenderBatch", "GetRenderBatch", "CancelRenderBatch",
    } <= names
    request = platform_pb2.AdaptiveChallengeRequest(player_id="p", recent_ids=["CHL-1"])
    assert list(request.recent_ids) == ["CHL-1"]
    render = platform_pb2.HistoricalReplayVideoRequest(
        identifier="BLT-TEST", decision_id="DEC-1", renderer_plugin="premium_pillow_v10",
        width=1080, height=1920, fps=30, audit_samples=96,
    )
    assert render.identifier == "BLT-TEST" and render.width == 1080
    fork = platform_pb2.ForkDecisionRequest(
        decision_id="DEC-1", action_json=b'{"kind":"card","code":"SA"}',
        continue_to_end=True, maximum_actions=256,
    )
    assert fork.continue_to_end and fork.maximum_actions == 256
    batch = platform_pb2.RenderBatchRequest(
        canonical_json=b'{"items":[]}', idempotency_key="batch-test"
    )
    assert batch.idempotency_key == "batch-test"
