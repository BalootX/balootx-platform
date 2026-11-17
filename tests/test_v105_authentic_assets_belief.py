from __future__ import annotations

import json
from pathlib import Path

import yaml

from baloot_platform.analysis.models import InformationState
from baloot_platform.analysis.sequential_belief_v10 import SequentialParticleBelief
from baloot_platform.analysis.team_public_search_v10 import analyze_team_public_hybrid
from baloot_platform.config import build_bundle
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.episode_io import load_episode_spec
from baloot_platform.shared import verify_shared_environment
from baloot_platform.video.card_assets import RasterCardArt, create_card_art
from baloot_platform.video.cards import PremiumCardArt
from baloot_platform.video.remotion_renderer import RemotionProfessionalRenderer


def _profile(root: Path) -> dict:
    return yaml.safe_load((root / "configs/video/premium.yaml").read_text(encoding="utf-8"))


def _episode(root: Path):
    bundle = build_bundle(root)
    return load_episode_spec(root / "examples/v7-outcome-flip-source.json", bundle.rules)


def test_goodall_pack_is_complete_and_selected(root: Path):
    profile = _profile(root)
    assert profile["card_assets"]["pack_id"] == "goodall_modern_baloot_32"
    manifest_path = root / profile["card_assets"]["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["packId"] == "goodall_modern_baloot_32"
    assert len(manifest["cards"]) == 32
    assert set(manifest["backs"]) >= {"1B", "2B"}
    art = create_card_art(profile, 180, 252)
    assert isinstance(art, RasterCardArt)
    assert art.face(0).size == (180, 252)
    assert art.back().size == (180, 252)
    assert art.face(0).tobytes() != art.face(1).tobytes()


def test_procedural_card_back_copy_is_governed(monkeypatch):
    monkeypatch.setattr(
        "baloot_platform.video.cards.assert_arabic_ready", lambda: None
    )
    profile = {
        "card_assets": {"provider": "procedural_v1"},
        "localizations": {
            "video.card_back.label": "صكّة",
            "video.card_back.subtitle": "BALOOTX TEST",
        },
    }
    art = create_card_art(profile, 180, 252)
    assert isinstance(art, PremiumCardArt)
    assert art.back_label == "صكّة"
    assert art.back_subtitle == "BALOOTX TEST"


def test_authentic_renderer_has_slow_readable_public_timeline(root: Path):
    profile = _profile(root)
    assert 0.3 <= float(profile["timeline"]["between_cards_seconds"]) <= 0.4
    assert float(profile["timeline"]["winner_glow_seconds"] + profile["timeline"]["trick_collect_seconds"]) >= 1.2
    assert profile["presentation"]["large_text_panels_before_reveal"] is False
    assert profile["presentation"]["show_trick_winner_text"] is False

    renderer = RemotionProfessionalRenderer(profile, repository_root=root)
    bundle = renderer.build_render_request(_episode(root), export_preset="hooked_flagship")
    stages = [item["stage"] for item in bundle.value["timeline"]["segments"]]
    assert stages == ["hook", "rewind", "replay", "question", "reveal", "continuation", "outcome"]
    context = next(scene for scene in bundle.value["scenes"] if scene["stage"] == "continuation")
    lifecycle = next(layer["content"] for layer in context["layers"] if layer["id"] == "public-lifecycle")
    assert lifecycle["bidHistory"]
    assert len(lifecycle["completedTricks"]) == 8


def test_authentic_renderer_can_render_ordinary_non_shared_episode(root: Path):
    profile = _profile(root)
    episode = _episode(root)
    renderer = RemotionProfessionalRenderer(profile, repository_root=root)
    bundle = renderer.build_render_request(episode, export_preset="decision_clip")
    assert bundle.verify()
    assert bundle.value["render"]["width"] == 1080
    assert bundle.value["render"]["height"] == 1920
    assert bundle.value["render"]["fps"] == 30
    assert bundle.value["timeline"]["totalFrames"] == 600


def test_shared_environment_freezes_visual_contract(root: Path, tmp_path: Path):
    from baloot_platform.agents.profiles import AgentProfileRegistry
    from baloot_platform.learning.service import LearningService
    from baloot_platform.learning.store import LearningStore
    from baloot_platform.sakka import SakkaService, SQLiteSakkaStore
    from baloot_platform.service.game import GameManager
    from baloot_platform.storage.game_store import SQLiteGameStore

    learning_store = LearningStore(tmp_path / "learning.sqlite3")
    registry = AgentProfileRegistry.from_yaml(root / "configs/agents/adaptive_lineup_v1.yaml")
    manager = GameManager(
        root, SQLiteGameStore(tmp_path / "games.sqlite3"),
        learning_store=learning_store, agent_registry=registry,
    )
    learning = LearningService(root, tmp_path / "learning", store=learning_store, game_manager=manager)
    manager.learning_service = learning
    service = SakkaService(root, SQLiteSakkaStore(tmp_path / "shared.sqlite3"), game_manager=manager)
    board = service.create_board_from_new_game(
        created_by="USR-V105", seed=340991, dealer=__import__("baloot_platform.core.enums", fromlist=["Seat"]).Seat.EAST, observer_seat=__import__("baloot_platform.core.enums", fromlist=["Seat"]).Seat.SOUTH,
        scope="laffa", maximum_rounds=1,
    )
    stored = service.store.get_board(board.board_id)
    assert stored is not None
    from baloot_platform.storage.codec_v7 import decode
    session = decode(stored[1], root=root)
    environment = session.shared_environment
    assert environment is not None
    check = verify_shared_environment(environment)
    assert check["valid"], check
    visual = environment["visualContract"]
    assert visual["rendererPlugin"] == "premium_video"
    assert visual["cardAssets"]["pack_id"] == "goodall_modern_baloot_32"
    assert visual["cardAssets"]["freeze_in_shared_environment"] is True
    assert visual["pacingProfile"] in {"premium_motion_baloot_v19"}
    public = service.board_public(board.board_id) if hasattr(service, "board_public") else None
    if public is not None:
        assert public["sharedEnvironment"]["visualContractHash"] == environment["visualContractHash"]


def test_sequential_belief_filters_on_public_card(root: Path):
    bundle = build_bundle(root)
    episode = _episode(root)
    engine = BalootEngine(bundle.rules)
    state = episode.state_before_pause.clone()
    info = InformationState.from_complete(state, episode.viewer)
    tracker = SequentialParticleBelief.initialize(
        info, engine, particles=24, exact_allocation_limit=1000, seed=77,
    )
    actual = episode.final_state.played_cards[episode.pause_before_play_index]
    update = tracker.observe_card(engine, actor=actual.seat, card=actual.card, ekka=actual.ekka)
    assert update.surviving_particles > 0
    assert tracker.step == 1
    assert all(p.state.played_cards[-1].card.id == actual.card.id for p in tracker.particles)


def test_hybrid_team_public_search_returns_legal_action(root: Path):
    bundle = build_bundle(root)
    episode = _episode(root)
    engine = BalootEngine(bundle.rules)
    info = InformationState.from_complete(episode.state_before_pause.clone(), episode.viewer)
    result = analyze_team_public_hybrid(
        info, engine, particles=16, exact_allocation_limit=1000,
        exact_remaining_threshold=0, ismcts_iterations=64, seed=12,
    )
    legal_ids = {card.id for card in engine.legal_moves(episode.state_before_pause)}
    assert result.recommended_card.id in legal_ids
    assert result.method == "hybrid_information_set_team_search_v2"
