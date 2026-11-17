from __future__ import annotations

import json
from pathlib import Path

import yaml

from baloot_platform.config import build_bundle
from baloot_platform.core.episode_io import load_episode_spec
from baloot_platform.video.card_assets import RasterCardArt
from baloot_platform.video.card_registry import CardPackRegistry
from baloot_platform.video.remotion_renderer import RemotionProfessionalRenderer


def test_v16_profile_is_bundle_default_and_closed(root: Path):
    bundle = build_bundle(root)
    assert bundle.video["renderer_plugin"] == "premium_video"
    assert bundle.video["profile"]["version"] == "21.3.5"
    assert bundle.video["card_assets"]["pack_id"] == "goodall_modern_baloot_32"
    assert bundle.video["card_assets"]["freeze_in_shared_environment"] is True
    assert bundle.video["presentation"]["large_text_panels_before_reveal"] is False
    assert bundle.video["callouts"]["public_only_before_reveal"] is True


def test_goodall_pack_contains_exact_baloot_deck(root: Path):
    manifest_path = root / "assets/cards/packs/goodall_modern_baloot_32/manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["packId"] == "goodall_modern_baloot_32"
    assert len(data["cards"]) == 32
    assert set(data["backs"]) == {"1B", "2B"}
    art = RasterCardArt(180, 252, manifest_path="assets/cards/packs/goodall_modern_baloot_32/manifest.json", back_id="1B", root=root)
    assert art.face(0).size == (180, 252)
    assert art.face(31).size == (180, 252)
    assert art.back().size == (180, 252)


def test_authentic_renderer_timeline_has_natural_calls_and_privacy(root: Path):
    bundle = build_bundle(root)
    episode = load_episode_spec(root / "examples/v7-outcome-flip-source.json", bundle.rules)
    profile = yaml.safe_load((root / "configs/video/premium.yaml").read_text(encoding="utf-8"))
    renderer = RemotionProfessionalRenderer(profile, repository_root=root)
    render_bundle = renderer.build_render_request(episode, export_preset="hooked_flagship")
    assert renderer.descriptor.id == "premium_video"
    assert "historical_actual" in renderer.descriptor.capabilities
    assert "replay_audit" in renderer.descriptor.capabilities
    assert 0.3 <= profile["timeline"]["between_cards_seconds"] <= 0.4
    assert profile["timeline"]["winner_glow_seconds"] + profile["timeline"]["trick_collect_seconds"] >= 1.2
    assert profile["presentation"]["show_trick_winner_text"] is False
    assert profile["presentation"]["physical_trick_collection"] is True
    assert profile["render"]["fps"] == 60
    assert profile["timeline"]["pause_seconds"] >= 5.0
    assert profile["presentation"]["show_deep_analysis_only_after_reveal"] is True
    assert profile["audio"]["table_foley"] is True
    assert render_bundle.verify()
    assert all(scene["observer"] == "south" for scene in render_bundle.value["scenes"])


def test_studio_exposes_simple_authentic_controls(root: Path):
    html = (root / "src/baloot_platform/service/static/studio.html").read_text(encoding="utf-8")
    assert "premium_video" in html
    assert 'id="cardPack"' in html
    assert 'id="seatCallouts"' in html
    assert 'id="voiceCallouts"' in html
    assert 'id="tableFoley"' in html
    assert "طبيعي مثل جلسة البلوت" in html
    assert html.count('id="cardPack"') == 1


def test_card_pack_registry_is_dynamic_and_validated(root: Path):
    registry = CardPackRegistry.from_yaml("configs/assets/card_packs.yaml", root=root)
    options = registry.options()
    assert options[0]["id"] == "goodall_modern_baloot_32"
    assert registry.profile_fragment("goodall_modern_baloot_32", "2B")["back_id"] == "2B"
    assert registry.profile_fragment("procedural_v1", "1B")["provider"] == "procedural_v1"
    try:
        registry.profile_fragment("unknown-pack", "1B")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown card packs must fail closed")


def test_authentic_contact_sheet_does_not_use_project_dashboard(root: Path):
    source = (root / "src/baloot_platform/video/remotion_renderer.py").read_text(encoding="utf-8")
    assert "_contact_sheet" in source
    assert "dashboard" not in source.lower()
