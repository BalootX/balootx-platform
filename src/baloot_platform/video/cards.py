from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, features

from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Rank, Suit

FONT_ROOT = Path(__file__).resolve().parents[1] / "resources/assets/fonts"
LATIN_REGULAR_CANDIDATES = (
    str(FONT_ROOT / "DejaVuSans.ttf"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)
LATIN_BOLD_CANDIDATES = (
    str(FONT_ROOT / "DejaVuSans-Bold.ttf"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)
ARABIC_REGULAR_CANDIDATES = (
    str(FONT_ROOT / "NotoSansArabic-Regular.ttf"),
    str(FONT_ROOT / "NotoKufiArabic-Regular.ttf"),
    "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoKufiArabic-Regular.ttf",
)
ARABIC_BOLD_CANDIDATES = (
    str(FONT_ROOT / "NotoSansArabic-Bold.ttf"),
    str(FONT_ROOT / "NotoKufiArabic-Bold.ttf"),
    "/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoKufiArabic-Bold.ttf",
)


def _pick(candidates: tuple[str, ...]) -> str:
    for path in candidates:
        if Path(path).exists():
            return path
    raise RuntimeError(f"required font is unavailable: {candidates}")


LATIN_REGULAR = _pick(LATIN_REGULAR_CANDIDATES)
LATIN_BOLD = _pick(LATIN_BOLD_CANDIDATES)
ARABIC_REGULAR = _pick(ARABIC_REGULAR_CANDIDATES)
ARABIC_BOLD = _pick(ARABIC_BOLD_CANDIDATES)


@lru_cache(maxsize=128)
def font(size: int, bold: bool = False, arabic: bool = False) -> ImageFont.FreeTypeFont:
    if arabic:
        path = ARABIC_BOLD if bold else ARABIC_REGULAR
    else:
        path = LATIN_BOLD if bold else LATIN_REGULAR
    return ImageFont.truetype(path, size)


def assert_arabic_ready() -> None:
    if not features.check_feature("raqm"):
        raise RuntimeError("Pillow was built without RAQM; Arabic shaping and bidi text are required")
    # Loading both fonts here gives a clear readiness failure before rendering.
    font(18, arabic=True)
    font(18, bold=True, arabic=True)


def _round(value: float) -> int:
    return max(1, int(round(value)))


# Traditional, highly readable pip positions in normalized card coordinates.
PIP_LAYOUTS: dict[Rank, tuple[tuple[float, float, bool], ...]] = {
    Rank.SEVEN: (
        (.31, .25, False), (.69, .25, False),
        (.31, .43, False), (.69, .43, False),
        (.50, .52, False),
        (.31, .74, True), (.69, .74, True),
    ),
    Rank.EIGHT: (
        (.31, .23, False), (.69, .23, False),
        (.31, .41, False), (.69, .41, False),
        (.31, .59, True), (.69, .59, True),
        (.31, .77, True), (.69, .77, True),
    ),
    Rank.NINE: (
        (.31, .22, False), (.69, .22, False),
        (.31, .39, False), (.69, .39, False),
        (.50, .50, False),
        (.31, .61, True), (.69, .61, True),
        (.31, .78, True), (.69, .78, True),
    ),
    Rank.TEN: (
        (.31, .20, False), (.69, .20, False),
        (.50, .33, False),
        (.31, .39, False), (.69, .39, False),
        (.31, .61, True), (.69, .61, True),
        (.50, .67, True),
        (.31, .80, True), (.69, .80, True),
    ),
}


class PremiumCardArt:
    """Procedural, original Baloot card artwork.

    The class deliberately does not depend on third-party card image packs.  It
    renders the 32-card Baloot deck with large corner indices, familiar pip
    layouts, and two-way court illustrations suitable for vertical video.
    """

    def __init__(
        self,
        width: int,
        height: int,
        *,
        back_label: str,
        back_subtitle: str,
    ):
        if width < 80 or height < 120:
            raise ValueError("card dimensions are too small")
        self.width = int(width)
        self.height = int(height)
        self.scale = self.width / 168.0
        self.back_label = str(back_label)
        self.back_subtitle = str(back_subtitle)
        if not self.back_label or not self.back_subtitle:
            raise ValueError("card-back copy is required")
        assert_arabic_ready()

    def _text(self, draw: ImageDraw.ImageDraw, xy, text: str, size: int, fill: str, *, anchor: str = "mm", bold: bool = False) -> None:
        draw.text(xy, text, font=font(max(10, _round(size * self.scale)), bold=bold), fill=fill, anchor=anchor)

    def _index(self, layer: Image.Image, card: Card) -> None:
        d = ImageDraw.Draw(layer)
        red = card.suit in {Suit.HEARTS, Suit.DIAMONDS}
        color = "#B5162A" if red else "#111820"
        x = _round(15 * self.scale)
        y = _round(10 * self.scale)
        self._text(d, (x, y), card.rank.short, 31, color, anchor="la", bold=True)
        self._text(d, (x + _round(8 * self.scale), y + _round(38 * self.scale)), card.suit.symbol, 27, color)

    def _court(self, image: Image.Image, card: Card, color: str) -> None:
        """Draw an original two-way French-style court face.

        The illustration intentionally favours clarity at phone size: a crown or
        cap, a visible face, a coloured coat, and a centre suit medallion.  It is
        not copied from any commercial deck.
        """
        d = ImageDraw.Draw(image)
        s = self.scale
        left, top = _round(36*s), _round(54*s)
        right, bottom = self.width - _round(36*s), self.height - _round(54*s)
        d.rounded_rectangle(
            (left, top, right, bottom),
            radius=_round(12*s),
            fill="#F7EEDC",
            outline="#C49A43",
            width=_round(2*s),
        )

        half_h = max(1, self.height // 2 - _round(4*s))
        half = Image.new("RGBA", (self.width, half_h), (0, 0, 0, 0))
        hd = ImageDraw.Draw(half)
        cx = self.width // 2
        skin = "#D9A47E"
        hair = "#5B392B"
        coat = "#244E78" if card.suit in {Suit.SPADES, Suit.CLUBS} else "#9E3040"
        coat_alt = "#B33D4C" if card.suit in {Suit.SPADES, Suit.CLUBS} else "#2B5C8A"
        gold = "#D0A13B"
        head_y = _round(83*s)
        head_rx = _round(18*s)
        head_ry = _round(22*s)

        # Shoulder and coat block.
        hd.polygon(
            [
                (cx-_round(38*s), half_h-_round(9*s)),
                (cx-_round(29*s), head_y+_round(24*s)),
                (cx, head_y+_round(13*s)),
                (cx+_round(29*s), head_y+_round(24*s)),
                (cx+_round(38*s), half_h-_round(9*s)),
            ],
            fill=coat,
            outline="#4E3329",
        )
        hd.polygon(
            [
                (cx-_round(10*s), head_y+_round(16*s)),
                (cx, head_y+_round(34*s)),
                (cx+_round(10*s), head_y+_round(16*s)),
                (cx, head_y+_round(8*s)),
            ],
            fill="#F7EEDC",
            outline=gold,
        )
        hd.polygon(
            [
                (cx-_round(34*s), half_h-_round(9*s)),
                (cx-_round(4*s), head_y+_round(30*s)),
                (cx+_round(34*s), half_h-_round(9*s)),
            ],
            fill=coat_alt,
        )

        # Hair behind the face.
        hd.ellipse(
            (cx-head_rx-_round(5*s), head_y-head_ry-_round(5*s), cx+head_rx+_round(5*s), head_y+head_ry+_round(8*s)),
            fill=hair,
            outline="#4B2E23",
        )
        hd.ellipse((cx-head_rx, head_y-head_ry, cx+head_rx, head_y+head_ry), fill=skin, outline="#6D4933")
        # Eyes and nose remain visible even at 140 px cards.
        eye_y = head_y-_round(2*s)
        hd.line((cx-_round(9*s), eye_y, cx-_round(4*s), eye_y), fill="#32241F", width=max(1, _round(1.2*s)))
        hd.line((cx+_round(4*s), eye_y, cx+_round(9*s), eye_y), fill="#32241F", width=max(1, _round(1.2*s)))
        hd.line((cx, eye_y+_round(2*s), cx-_round(2*s), eye_y+_round(9*s)), fill="#8C5E43", width=max(1, _round(1.0*s)))
        hd.arc((cx-_round(8*s), eye_y+_round(7*s), cx+_round(8*s), eye_y+_round(17*s)), 15, 165, fill="#7A453A", width=max(1, _round(1.2*s)))

        if card.rank == Rank.KING:
            # Five-point crown and a clear beard.
            crown_y = head_y-head_ry-_round(5*s)
            hd.polygon(
                [
                    (cx-_round(23*s), crown_y+_round(10*s)),
                    (cx-_round(20*s), crown_y-_round(8*s)),
                    (cx-_round(10*s), crown_y+_round(2*s)),
                    (cx, crown_y-_round(12*s)),
                    (cx+_round(10*s), crown_y+_round(2*s)),
                    (cx+_round(20*s), crown_y-_round(8*s)),
                    (cx+_round(23*s), crown_y+_round(10*s)),
                ],
                fill=gold,
                outline="#77591C",
            )
            hd.polygon(
                [(cx-_round(15*s), head_y+_round(12*s)), (cx, head_y+_round(31*s)), (cx+_round(15*s), head_y+_round(12*s))],
                fill=hair,
                outline="#4B2E23",
            )
        elif card.rank == Rank.QUEEN:
            crown_y = head_y-head_ry-_round(4*s)
            hd.polygon(
                [
                    (cx-_round(22*s), crown_y+_round(10*s)),
                    (cx-_round(15*s), crown_y-_round(7*s)),
                    (cx, crown_y+_round(1*s)),
                    (cx+_round(15*s), crown_y-_round(7*s)),
                    (cx+_round(22*s), crown_y+_round(10*s)),
                ],
                fill=gold,
                outline="#77591C",
            )
            hd.ellipse((cx-_round(22*s), head_y-_round(5*s), cx-_round(13*s), head_y+_round(27*s)), fill=hair)
            hd.ellipse((cx+_round(13*s), head_y-_round(5*s), cx+_round(22*s), head_y+_round(27*s)), fill=hair)
        else:
            # Jack's soft cap and small feather distinguish it from Q/K.
            cap_y = head_y-head_ry-_round(3*s)
            hd.polygon(
                [(cx-_round(24*s), cap_y+_round(11*s)), (cx-_round(7*s), cap_y-_round(9*s)), (cx+_round(24*s), cap_y+_round(7*s)), (cx+_round(17*s), cap_y+_round(15*s))],
                fill=coat,
                outline="#4B2E23",
            )
            hd.line((cx+_round(13*s), cap_y-_round(4*s), cx+_round(25*s), cap_y-_round(18*s)), fill=gold, width=max(1, _round(2*s)))

        # Place the top figure and its 180-degree counterpart.
        image.alpha_composite(half, (0, _round(51*s)))
        image.alpha_composite(half.rotate(180), (0, self.height-half_h-_round(51*s)))
        d = ImageDraw.Draw(image)
        band_y = self.height // 2
        d.rounded_rectangle(
            (self.width//2-_round(29*s), band_y-_round(27*s), self.width//2+_round(29*s), band_y+_round(27*s)),
            radius=_round(15*s),
            fill="#FFF9ED",
            outline=gold,
            width=max(1, _round(2*s)),
        )
        self._text(d, (self.width//2, band_y), card.suit.symbol, 30, color, bold=True)

    @lru_cache(maxsize=64)
    def face(self, card_id: int) -> Image.Image:
        card = Card(card_id)
        s = self.scale
        image = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        d = ImageDraw.Draw(image)
        d.rounded_rectangle((2, 2, self.width-3, self.height-3), radius=_round(15*s), fill="#FFFDF8", outline="#D7D0C3", width=_round(2*s))
        d.rounded_rectangle((_round(7*s), _round(7*s), self.width-_round(8*s), self.height-_round(8*s)), radius=_round(11*s), outline="#EEE8DD", width=_round(1*s))
        red = card.suit in {Suit.HEARTS, Suit.DIAMONDS}
        color = "#B5162A" if red else "#111820"

        top_index = Image.new("RGBA", image.size, (0, 0, 0, 0))
        self._index(top_index, card)
        image.alpha_composite(top_index)
        image.alpha_composite(top_index.rotate(180))
        d = ImageDraw.Draw(image)

        if card.rank in {Rank.JACK, Rank.QUEEN, Rank.KING}:
            self._court(image, card, color)
        elif card.rank == Rank.ACE:
            self._text(d, (self.width//2, self.height//2), card.suit.symbol, 88, color)
            self._text(d, (self.width//2, self.height//2+_round(56*s)), "A", 23, "#8A8176", bold=True)
        else:
            for x, y, rotate in PIP_LAYOUTS[card.rank]:
                glyph = Image.new("RGBA", (_round(40*s), _round(42*s)), (0, 0, 0, 0))
                gd = ImageDraw.Draw(glyph)
                self._text(gd, (glyph.width//2, glyph.height//2), card.suit.symbol, 28, color)
                if rotate:
                    glyph = glyph.rotate(180)
                image.alpha_composite(glyph, (_round(self.width*x-glyph.width/2), _round(self.height*y-glyph.height/2)))
        return image

    @lru_cache(maxsize=16)
    def back(self, label: str | None = None) -> Image.Image:
        s = self.scale
        image = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        d = ImageDraw.Draw(image)
        d.rounded_rectangle((2, 2, self.width-3, self.height-3), radius=_round(15*s), fill="#FFF8E8", outline="#D7D0C3", width=_round(2*s))
        d.rounded_rectangle((_round(8*s), _round(8*s), self.width-_round(9*s), self.height-_round(9*s)), radius=_round(11*s), fill="#073E35", outline="#D4AA48", width=_round(3*s))
        # Fine geometric pattern inspired by tournament table branding.
        step = max(8, _round(14*s))
        for offset in range(-self.height, self.width+self.height, step):
            d.line((offset, 0, offset-self.height, self.height), fill="#0C5A4B", width=max(1, _round(1.4*s)))
        d.rounded_rectangle((_round(26*s), _round(43*s), self.width-_round(27*s), self.height-_round(44*s)), radius=_round(15*s), fill="#052D27", outline="#E2BD61", width=_round(2*s))
        d.ellipse((_round(43*s), _round(72*s), self.width-_round(44*s), self.height-_round(73*s)), outline="#A98238", width=_round(2*s))
        selected_label = self.back_label if label is None else str(label)
        d.text((self.width//2, self.height//2-_round(6*s)), selected_label, font=font(max(16, _round(25*s)), bold=True, arabic=True), fill="#F7D46A", anchor="mm", direction="rtl", language="ar")
        d.text((self.width//2, self.height//2+_round(36*s)), self.back_subtitle, font=font(max(8, _round(11*s)), bold=True), fill="#BEA15B", anchor="mm")
        return image

    @lru_cache(maxsize=128)
    def shadow(self, width: int, height: int, radius: int = 18) -> Image.Image:
        pad = max(10, width // 12)
        tile = Image.new("RGBA", (width + pad*2, height + pad*2), (0, 0, 0, 0))
        d = ImageDraw.Draw(tile)
        d.rounded_rectangle((pad, pad, pad+width, pad+height), radius=radius, fill=(0, 0, 0, 78))
        return tile.filter(ImageFilter.GaussianBlur(max(2, width//45)))
