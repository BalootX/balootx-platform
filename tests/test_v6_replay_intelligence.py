from __future__ import annotations

import json
from pathlib import Path

import yaml

from baloot_platform.agents.profiles import AgentProfileRegistry
from baloot_platform.audit.difficulty import OnlineDifficultyCalibrator
from baloot_platform.audit.service import ReplayAuditService
from baloot_platform.config import build_bundle
from baloot_platform.core.enums import Seat
from baloot_platform.learning.service import LearningService
from baloot_platform.learning.store import LearningStore
from baloot_platform.review import DecisionReviewService
from baloot_platform.service.game import GameManager
from baloot_platform.storage.game_store import SQLiteGameStore
from baloot_platform.video.remotion_renderer import RemotionProfessionalRenderer


def _system(root: Path, tmp_path: Path):
    store = LearningStore(tmp_path / "learning.sqlite3")
    registry = AgentProfileRegistry.from_yaml(root / "configs/agents/adaptive_lineup_v1.yaml")
    manager = GameManager(
        root,
        SQLiteGameStore(tmp_path / "games.sqlite3"),
        learning_store=store,
        agent_registry=registry,
    )
    learning = LearningService(root, tmp_path / "learning", store=store, game_manager=manager)
    manager.learning_service = learning
    audit = ReplayAuditService(root, store, learning)
    return store, manager, learning, audit


def _completed_beginner_game(root: Path, tmp_path: Path):
    store, manager, learning, audit = _system(root, tmp_path)
    lineup = {seat: "beginner" for seat in ("north", "west", "south", "east")}
    game = manager.create(seed=501, dealer=Seat.EAST, agent_profiles=lineup)
    game = manager.advance_bots(game.id, stop_before=None, maximum_actions=100)
    assert game.state.round_score is not None
    assert game.game_hash
    return store, manager, learning, audit, game


def test_bot_policy_trace_is_observer_safe_and_auditable(root: Path, tmp_path: Path):
    store, _manager, _learning, audit, game = _completed_beginner_game(root, tmp_path)
    decision = next(
        item for item in reversed(store.list_decisions(game.id))
        if item["decisionKind"] == "play" and len(item["legalActions"]) > 1
    )
    trace = decision["decisionMetadata"]
    assert trace["source"] == "bot"
    assert trace["observerSafe"] is True
    assert trace["profileId"] == "beginner"
    assert trace["rankedActions"]
    report = audit.audit_decision(
        decision["decisionId"], deep=False, force=True,
        overrides={
            "requested_samples": 4,
            "exact_allocation_limit": 100,
            "exact_search_remaining": 4,
            "maximum_sampling_attempts": 2000,
            "seed": 3,
        },
    )
    assert report["actualIntent"]["status"] == "recorded_policy_trace"
    assert report["actualIntent"]["profileId"] == "beginner"
    assert len(report["alternatives"]) >= 2
    assert report["reasons"]
    assert report["actualAction"]
    assert report["bestAction"]


def test_human_intent_stays_unknown_until_reviewed(root: Path, tmp_path: Path):
    store, manager, learning, audit = _system(root, tmp_path)
    game = manager.create(seed=409, dealer=Seat.EAST)
    actor = game.bidding.actor
    assert actor is not None
    manager.bid(game.id, seat=actor, action="pass", expected_version=game.version)
    decision = store.list_decisions(game.id)[0]
    first = audit.audit_decision(
        decision["decisionId"], deep=False, force=True,
        overrides={"requested_samples": 4, "seed": 4},
    )
    assert first["actualIntent"]["status"] == "unknown"
    learning.add_feedback(
        decision["decisionId"],
        preferred_action={"action": "pass", "trumpSuit": None},
        confidence=0.92,
        rationale="مررت لأن احتمال نجاح الشراء لا يبرر مخاطرة النتيجة.",
        source="expert",
        reviewer="expert-1",
    )
    reviewed = audit.audit_decision(
        decision["decisionId"], deep=False, force=True,
        overrides={"requested_samples": 4, "seed": 4},
    )
    assert reviewed["actualIntent"]["status"] == "reviewed_statement"
    assert "احتمال" in reviewed["actualIntent"]["rationale"]


def test_difficulty_calibration_is_replayable(root: Path, tmp_path: Path):
    store = LearningStore(tmp_path / "difficulty.sqlite3")
    calibrator = OnlineDifficultyCalibrator.from_yaml(store, root / "configs/difficulty/online_irt_v1.yaml")
    accepted = [{"kind": "card", "id": 1, "code": "S8", "suit": "spades", "rank": "eight"}]
    for index in range(12):
        calibrator.record_attempt(
            decision_id="DEC-DIFFICULTY",
            player_id=f"player-{index}",
            chosen_action=accepted[0] if index < 3 else {"kind": "card", "id": 2, "code": "S9", "suit": "spades", "rank": "nine"},
            accepted_actions=accepted,
            engine_difficulty=60.0,
            response_seconds=18.0 + index,
            confidence=0.7,
            reported_skill=55.0,
        )
    first = calibrator.recalibrate("DEC-DIFFICULTY", engine_difficulty=60.0)
    second = calibrator.recalibrate("DEC-DIFFICULTY", engine_difficulty=60.0)
    assert first["calibratedDifficulty"] == second["calibratedDifficulty"]
    assert first["responseCount"] == 12
    assert first["distinctPlayers"] == 12
    assert first["calibratedDifficulty"] > 50


def test_audit_enriched_dataset_preserves_preference_and_trace(root: Path, tmp_path: Path):
    store, _manager, learning, audit, game = _completed_beginner_game(root, tmp_path)
    decision = next(
        item for item in reversed(store.list_decisions(game.id))
        if item["decisionKind"] == "play" and len(item["legalActions"]) > 1
    )
    report = audit.audit_decision(
        decision["decisionId"], deep=False, force=True,
        overrides={
            "requested_samples": 4,
            "exact_allocation_limit": 100,
            "exact_search_remaining": 4,
            "maximum_sampling_attempts": 2000,
            "seed": 5,
        },
    )
    reviews = DecisionReviewService(store, audit)
    for reviewer, role in (("expert-2", "expert"), ("peer-2", "player")):
        reviews.submit(
            decision["decisionId"], reviewer_id=reviewer, reviewer_role=role,
            preferred_action=report["bestAction"], confidence=0.96,
            reason_codes=["card_counting"],
            rationale="تمت مراجعة البدائل ومقارنة الندم واحتمال الفوز.",
            view_mode="observer",
        )
    dataset = learning.build_dataset(minimum_confidence=0.5)
    row = json.loads(Path(dataset["examples"]).read_text(encoding="utf-8").splitlines()[0])
    assert row["schemaVersion"] == 3
    assert row["policyTrace"]["source"] == "bot"
    assert row["audit"]["auditId"]
    assert row["preferencePair"]["preferredAction"] == row["targetAction"]
    assert row["acceptedActions"]
    assert row["sampleWeight"] > 0
    metadata = dataset["manifestData"]["metadata"]
    assert metadata["auditEnrichedExamples"] == 1
    assert metadata["schemaVersion"] == 3


def test_professional_remotion_is_the_default_replay_renderer(root: Path):
    bundle = build_bundle(root)
    renderer = RemotionProfessionalRenderer.from_yaml(root / "configs/video/premium.yaml")
    assert bundle.video["renderer_plugin"] == "premium_video"
    assert renderer.descriptor.id == "premium_video"
    assert "historical_actual" in renderer.descriptor.capabilities
    assert "replay_audit" in renderer.descriptor.capabilities
    html = (root / "src/baloot_platform/service/static/replay_studio.html").read_text(encoding="utf-8")
    assert "مختبر مراجعة البلوت" in html
    assert "سبب اللعب" in html or "سبب" in html
    assert "advanced" in html
    assert 'href="/replay"' in html


def test_historical_replay_video_maps_every_archived_decision(root: Path, tmp_path: Path):
    store, _manager, learning, audit, game = _completed_beginner_game(root, tmp_path)
    decisions = store.list_decisions(game.game_hash or game.id, limit=10_000)
    critical = next(
        item for item in reversed(decisions)
        if item["decisionKind"] == "play" and len(item["legalActions"]) > 1
    )
    audit.audit_decision(
        critical["decisionId"], deep=False, force=True,
        overrides={
            "requested_samples": 4, "exact_allocation_limit": 100,
            "exact_search_remaining": 4, "maximum_sampling_attempts": 2000, "seed": 8,
        },
    )
    episode = learning.create_historical_replay_episode(
        game.game_hash or game.id, decision_id=critical["decisionId"]
    )
    assert episode.playback_mode == "historical_actual"
    assert len(episode.final_state.played_cards) == 32
    mapped_ids = {decision_id for _label, decision_id in episode.decision_index}
    assert {item["decisionId"] for item in decisions} <= mapped_ids

    profile = yaml.safe_load((root / "configs/video/premium.yaml").read_text(encoding="utf-8"))
    renderer = RemotionProfessionalRenderer(profile, repository_root=root)
    render_bundle = renderer.build_render_request(episode, export_preset="hooked_flagship")
    assert render_bundle.verify()
    mapped_ids = {item["decisionId"] for item in render_bundle.value["decisionIndex"]}
    assert {item["decisionId"] for item in decisions} <= mapped_ids
    assert render_bundle.value["timeline"]["totalFrames"] == 4320


def test_counterfactual_branch_can_complete_without_rewriting_source(root: Path, tmp_path: Path):
    store, _manager, learning, audit, game = _completed_beginner_game(root, tmp_path)
    source_hash = str(game.game_hash)
    before = store.verify_game(source_hash)
    assert before["valid"] is True
    decision = next(
        item for item in reversed(store.list_decisions(source_hash))
        if item["decisionKind"] == "play" and len(item["legalActions"]) > 1
    )
    report = audit.audit_decision(
        decision["decisionId"], deep=False, force=True,
        overrides={
            "requested_samples": 4, "exact_allocation_limit": 100,
            "exact_search_remaining": 4, "maximum_sampling_attempts": 2000, "seed": 9,
        },
    )
    branch = learning.fork_decision(
        decision["decisionId"], action=report["bestAction"],
        continue_to_end=True, maximum_actions=256,
    )
    assert branch["branchGameId"] != game.id
    assert branch["parentGameHash"] == source_hash
    assert branch["continuedToEnd"] is True
    assert branch["counterfactualComparison"]["completed"] is True
    assert branch["counterfactualComparison"]["branchRoundScore"] is not None
    after = store.verify_game(source_hash)
    assert after == before
