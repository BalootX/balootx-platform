from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter

from baloot_platform.config import platform_root
from baloot_platform.core.cards import Card

from .cards import PremiumCardArt


class RasterCardArt:
    """Card-art provider backed by a versioned raster asset-pack manifest.

    The renderer receives only card IDs and never interprets game semantics.
    Missing or malformed assets fail fast, preventing a video from silently
    showing the wrong card face.
    """

    def __init__(
        self,
        width: int,
        height: int,
        *,
        manifest_path: str,
        back_id: str,
        root: str | Path | None = None,
    ) -> None:
        if width < 80 or height < 120:
            raise ValueError("card dimensions are too small")
        self.width = int(width)
        self.height = int(height)
        self.root = platform_root(root)
        self.manifest_path = str(manifest_path)
        manifest_file = self.root / self.manifest_path
        if not manifest_file.is_file():
            raise FileNotFoundError(f"card asset manifest not found: {manifest_file}")
        self.manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        self.pack_id = str(self.manifest["packId"])
        self.back_id = str(back_id or self.manifest.get("defaultBack", "1B"))
        cards = self.manifest.get("cards") or {}
        if len(cards) != 32:
            raise ValueError(f"card pack {self.pack_id} must contain exactly 32 Baloot cards")
        self._cards: dict[str, Path] = {}
        for key, value in cards.items():
            entry = value if isinstance(value, dict) else {"png": value}
            rel = entry.get("png") or entry.get("svg")
            if not rel:
                raise ValueError(f"card pack entry {key} has no renderable asset")
            path = self.root / str(rel)
            if not path.is_file():
                raise FileNotFoundError(f"card asset missing: {path}")
            self._cards[str(key)] = path
        backs = self.manifest.get("backs") or {}
        if self.back_id not in backs:
            raise KeyError(f"unknown card back {self.back_id} in pack {self.pack_id}")
        back_entry = backs[self.back_id] if isinstance(backs[self.back_id], dict) else {"png": backs[self.back_id]}
        back_rel = back_entry.get("png") or back_entry.get("svg")
        self._back_path = self.root / str(back_rel)
        if not self._back_path.is_file():
            raise FileNotFoundError(f"card back asset missing: {self._back_path}")

    @staticmethod
    def _key(card: Card) -> str:
        return f"{card.rank.short}_{card.suit.name.lower()}"

    @staticmethod
    def _load(path: Path) -> Image.Image:
        # Release packs contain pre-rasterized PNG masters so runtime rendering
        # does not depend on an SVG engine or browser.
        if path.suffix.lower() != ".png":
            raise ValueError(f"runtime card pack requires PNG assets, got {path}")
        return Image.open(path).convert("RGBA")

    @lru_cache(maxsize=64)
    def face(self, card_id: int) -> Image.Image:
        card = Card(card_id)
        path = self._cards[self._key(card)]
        image = self._load(path)
        return image.resize((self.width, self.height), Image.Resampling.LANCZOS)

    @lru_cache(maxsize=8)
    def back(self, label: str | None = None) -> Image.Image:  # noqa: ARG002 - API parity
        image = self._load(self._back_path)
        return image.resize((self.width, self.height), Image.Resampling.LANCZOS)

    @lru_cache(maxsize=128)
    def shadow(self, width: int, height: int, radius: int = 18) -> Image.Image:
        pad = max(10, width // 12)
        tile = Image.new("RGBA", (width + pad * 2, height + pad * 2), (0, 0, 0, 0))
        d = ImageDraw.Draw(tile)
        d.rounded_rectangle((pad, pad, pad + width, pad + height), radius=radius, fill=(0, 0, 0, 82))
        return tile.filter(ImageFilter.GaussianBlur(max(2, width // 45)))


def create_card_art(video_profile: dict[str, Any], width: int, height: int) -> PremiumCardArt | RasterCardArt:
    settings = dict(video_profile.get("card_assets") or {})
    localizations = dict(video_profile.get("localizations") or {})
    provider = str(settings.get("provider", "procedural_v1"))
    if provider == "procedural_v1":
        return PremiumCardArt(
            width,
            height,
            back_label=str(localizations["video.card_back.label"]),
            back_subtitle=str(localizations["video.card_back.subtitle"]),
        )
    if provider == "raster_pack_v1":
        return RasterCardArt(
            width,
            height,
            manifest_path=str(settings["manifest"]),
            back_id=str(settings.get("back_id", "1B")),
        )
    raise ValueError(f"unknown card-art provider: {provider}")
