from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from PIL import Image

from baloot_platform.core.cards import FULL_DECK
from baloot_platform.video.card_assets import create_card_art
from baloot_platform.video.visual_qa import LayoutAudit, balanced_grid, image_sha256


def _profile(root: Path) -> dict:
    return yaml.safe_load((root / "configs/video/premium.yaml").read_text(encoding="utf-8"))


def test_bundled_arabic_fonts_are_content_addressed(root: Path, fresh_professional_bundles) -> None:
    bundle = fresh_professional_bundles["hooked_flagship"]
    assert bundle.verify()
    font_assets = [item for item in bundle.value["assets"] if item["id"].startswith("noto-")]
    assert len(font_assets) == 3
    for item in font_assets:
        path = root.joinpath(*Path(item["path"]).parts)
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    source = (root / "src/baloot_platform/video/remotion/src/professional-prototype.tsx").read_text(encoding="utf-8")
    assert "@font-face" in source and "NotoSansArabic" in source and "NotoKufiArabic" in source


def test_all_32_baloot_cards_are_distinct_and_readable(root: Path) -> None:
    art = create_card_art(_profile(root), 180, 252)
    hashes = {image_sha256(art.face(card.id)) for card in FULL_DECK}
    assert len(hashes) == 32
    assert all(art.face(card.id).size == (180, 252) for card in FULL_DECK)


def test_remotion_waits_for_card_images_before_rendering(root: Path) -> None:
    source = (root / "src/baloot_platform/video/remotion/src/professional-prototype.tsx").read_text(encoding="utf-8")
    assert "<Img" in source
    assert "<img" not in source




def test_current_flagship_presentation_preserves_safe_areas(fresh_professional_bundles) -> None:
    bundle = fresh_professional_bundles["hooked_flagship"]
    assert bundle.verify()
    safe = bundle.value["presentation"]["safe_areas"]
    assert safe["profile"] == "vertical_social_v1"
    assert safe["top_px"] >= 180 and safe["bottom_px"] >= 300






def test_current_decision_bundle_is_observer_safe(fresh_professional_bundles) -> None:
    bundle = fresh_professional_bundles["decision_clip"]
    assert bundle.verify()
    assert all(scene["observer"] == "south" for scene in bundle.value["scenes"])
    assert bundle.value["timeline"]["totalFrames"] == round(
        bundle.value["presentation"]["exports"]["decision_clip"]["target_seconds"]
        * bundle.value["render"]["fps"]
    )


def test_balanced_contact_sheet_grid_uses_canvas() -> None:
    boxes, height = balanced_grid(
        10,
        canvas_width=1080,
        columns=4,
        top=62,
        side_margin=12,
        gap=12,
        label_height=28,
        bottom_margin=16,
    )
    assert len(boxes) == 10
    assert height == boxes[-1].bottom + 28 + 16
    audit = LayoutAudit(1080, height, safe_margin=0)
    for index, box in enumerate(boxes):
        audit.add(f"frame_{index}", box)
    assert audit.issues() == []
