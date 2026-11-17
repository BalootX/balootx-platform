"""Shared Arabic-first typography helpers for non-Remotion evidence surfaces."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PIL import ImageDraw, ImageFont

from .visual_qa import assert_text_supported


FONT_ROOT = Path(__file__).resolve().parents[1] / "resources/assets/fonts"
FONT_TITLE_REGULAR = str(FONT_ROOT / "NotoKufiArabic-Regular.ttf")
FONT_TITLE_BOLD = str(FONT_ROOT / "NotoKufiArabic-Bold.ttf")
FONT_BODY_REGULAR = str(FONT_ROOT / "NotoSansArabic-Regular.ttf")
FONT_BODY_BOLD = str(FONT_ROOT / "NotoSansArabic-Bold.ttf")
FONT_LATIN = str(FONT_ROOT / "DejaVuSans.ttf")
FONT_LATIN_BOLD = str(FONT_ROOT / "DejaVuSans-Bold.ttf")


def _existing(path: str, fallback: str) -> str:
    return path if Path(path).exists() else fallback


@lru_cache(maxsize=512)
def _font(
    size: int,
    *,
    arabic: bool = True,
    bold: bool = False,
    title: bool = False,
) -> ImageFont.FreeTypeFont:
    if arabic:
        if title:
            path = _existing(FONT_TITLE_BOLD, FONT_BODY_BOLD) if bold else _existing(
                FONT_TITLE_REGULAR, FONT_BODY_REGULAR
            )
        else:
            path = FONT_BODY_BOLD if bold else FONT_BODY_REGULAR
    else:
        path = FONT_LATIN_BOLD if bold else FONT_LATIN
    return ImageFont.truetype(path, int(size))


def _latin(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    size: int,
    fill: str,
    *,
    anchor: str = "mm",
    bold: bool = False,
) -> None:
    value = str(text)
    selected_font = _font(size, arabic=False, bold=bold)
    assert_text_supported(selected_font, value, label="Latin render text")
    draw.text(xy, value, font=selected_font, fill=fill, anchor=anchor)


def font_contract() -> dict[str, object]:
    return {
        "strategy": "bundled-noto-arabic-plus-dejavu-latin-vector-card-icons",
        "packages": ["fonts-noto-core", "fonts-noto-extra", "fonts-dejavu-core"],
        "paths": {
            "arabicBodyRegular": "resources/assets/fonts/NotoSansArabic-Regular.ttf",
            "arabicBodyBold": "resources/assets/fonts/NotoSansArabic-Bold.ttf",
            "arabicTitleRegular": "resources/assets/fonts/NotoKufiArabic-Regular.ttf",
            "arabicTitleBold": "resources/assets/fonts/NotoKufiArabic-Bold.ttf",
            "latinRegular": "resources/assets/fonts/DejaVuSans.ttf",
            "latinBold": "resources/assets/fonts/DejaVuSans-Bold.ttf",
        },
        "strictGlyphValidation": True,
        "mixedScriptPolicy": (
            "Arabic and Latin runs are drawn separately; card suits and ranks "
            "use card artwork or vectors."
        ),
    }
