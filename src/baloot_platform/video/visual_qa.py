from __future__ import annotations

import hashlib
import math
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont, features


@dataclass(frozen=True, slots=True)
class Rect:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    @property
    def area(self) -> int:
        return self.width * self.height

    def intersects(self, other: "Rect", *, padding: int = 0) -> bool:
        return not (
            self.right + padding <= other.left
            or other.right + padding <= self.left
            or self.bottom + padding <= other.top
            or other.bottom + padding <= self.top
        )

    def contains(self, other: "Rect", *, padding: int = 0) -> bool:
        return (
            other.left >= self.left + padding
            and other.top >= self.top + padding
            and other.right <= self.right - padding
            and other.bottom <= self.bottom - padding
        )

    def to_dict(self) -> dict[str, int]:
        return {"left": self.left, "top": self.top, "right": self.right, "bottom": self.bottom}


@dataclass(frozen=True, slots=True)
class LayoutItem:
    name: str
    rect: Rect
    layer: str = "overlay"
    may_overlap: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "rect": self.rect.to_dict(), "layer": self.layer, "mayOverlap": list(self.may_overlap)}


@dataclass(slots=True)
class LayoutAudit:
    width: int
    height: int
    safe_margin: int = 0
    items: list[LayoutItem] = field(default_factory=list)

    @property
    def safe_rect(self) -> Rect:
        return Rect(self.safe_margin, self.safe_margin, self.width - self.safe_margin, self.height - self.safe_margin)

    def add(self, name: str, rect: Rect, *, layer: str = "overlay", may_overlap: Sequence[str] = ()) -> Rect:
        self.items.append(LayoutItem(name=name, rect=rect, layer=layer, may_overlap=tuple(may_overlap)))
        return rect

    def issues(self) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        safe = self.safe_rect
        for item in self.items:
            if not safe.contains(item.rect):
                issues.append({"code": "out_of_safe_bounds", "item": item.name, "rect": item.rect.to_dict(), "safeRect": safe.to_dict()})
        for index, left in enumerate(self.items):
            for right in self.items[index + 1 :]:
                if left.layer != right.layer:
                    continue
                if right.name in left.may_overlap or left.name in right.may_overlap:
                    continue
                if left.rect.intersects(right.rect):
                    issues.append({
                        "code": "overlap",
                        "left": left.name,
                        "right": right.name,
                        "leftRect": left.rect.to_dict(),
                        "rightRect": right.rect.to_dict(),
                    })
        return issues

    def assert_valid(self) -> None:
        issues = self.issues()
        if issues:
            raise AssertionError(f"layout audit failed: {issues}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "safeMargin": self.safe_margin,
            "safeRect": self.safe_rect.to_dict(),
            "items": [item.to_dict() for item in self.items],
            "issues": self.issues(),
        }


# U+0378 is permanently unassigned and therefore gives a stable .notdef glyph.
_NOTDEF_CHARACTER = "\u0378"


def glyph_signature(font: ImageFont.FreeTypeFont, character: str) -> tuple[tuple[int, int], bytes]:
    mask = font.getmask(character)
    return mask.size, bytes(mask)


def unsupported_characters(font: ImageFont.FreeTypeFont, text: str) -> tuple[str, ...]:
    notdef = glyph_signature(font, _NOTDEF_CHARACTER)
    missing: list[str] = []
    for character in str(text):
        category = unicodedata.category(character)
        if character.isspace() or category in {"Cf", "Cc", "Mn"}:
            continue
        if glyph_signature(font, character) == notdef:
            missing.append(character)
    return tuple(dict.fromkeys(missing))


def assert_text_supported(font: ImageFont.FreeTypeFont, text: str, *, label: str = "text") -> None:
    missing = unsupported_characters(font, text)
    if missing:
        escaped = [f"U+{ord(value):04X} {value!r}" for value in missing]
        raise ValueError(f"{label} contains unsupported glyphs: {escaped}")


def text_bbox(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    *,
    direction: str | None = None,
    language: str | None = None,
) -> tuple[int, int, int, int]:
    kwargs: dict[str, Any] = {"font": font}
    value = str(text)
    if direction == "rtl" and not features.check_feature("raqm"):
        import arabic_reshaper
        from bidi.algorithm import get_display

        value = get_display(arabic_reshaper.reshape(value))
        direction = None
        language = None
    if direction:
        kwargs["direction"] = direction
    if language:
        kwargs["language"] = language
    return tuple(int(item) for item in draw.textbbox((0, 0), value, **kwargs))


def wrap_words_rtl(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    *,
    max_width: int,
    max_lines: int,
) -> list[str]:
    words = str(text).replace("\n", " ").split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        box = text_bbox(draw, candidate, font, direction="rtl", language="ar")
        if box[2] - box[0] <= max_width:
            current = candidate
            continue
        lines.append(current)
        current = word
        if len(lines) >= max_lines - 1:
            break
    remaining_start = sum(len(line.split()) for line in lines)
    if len(lines) < max_lines:
        tail_words = words[remaining_start:]
        current = " ".join(tail_words) if tail_words else current
        while current:
            box = text_bbox(draw, current, font, direction="rtl", language="ar")
            if box[2] - box[0] <= max_width:
                break
            pieces = current.split()
            if len(pieces) <= 1:
                current = current[:-1]
            else:
                pieces = pieces[:-1]
                current = " ".join(pieces) + "..."
        lines.append(current)
    return lines[:max_lines]


def draw_arabic_fit(
    draw: ImageDraw.ImageDraw,
    box: Rect,
    text: str,
    *,
    font_factory,
    fill: str,
    max_size: int,
    min_size: int = 11,
    max_lines: int = 2,
    bold: bool = False,
    title: bool = False,
    line_gap: int = 4,
) -> dict[str, Any]:
    for size in range(int(max_size), int(min_size) - 1, -1):
        font = font_factory(size, arabic=True, bold=bold, title=title)
        assert_text_supported(font, text, label="Arabic text")
        lines = wrap_words_rtl(draw, text, font, max_width=box.width, max_lines=max_lines)
        bboxes = [text_bbox(draw, line, font, direction="rtl", language="ar") for line in lines]
        line_heights = [max(1, value[3] - value[1]) for value in bboxes]
        total_height = sum(line_heights) + line_gap * max(0, len(lines) - 1)
        widths = [value[2] - value[0] for value in bboxes]
        if max(widths, default=0) <= box.width and total_height <= box.height:
            y = box.top + (box.height - total_height) / 2
            runs = []
            for line, height in zip(lines, line_heights):
                rendered_line = line
                text_kwargs: dict[str, Any] = {"direction": "rtl", "language": "ar"}
                if not features.check_feature("raqm"):
                    import arabic_reshaper
                    from bidi.algorithm import get_display

                    rendered_line = get_display(arabic_reshaper.reshape(line))
                    text_kwargs = {}
                draw.text(
                    (box.left + box.width / 2, y + height / 2),
                    rendered_line,
                    font=font,
                    fill=fill,
                    anchor="mm",
                    **text_kwargs,
                )
                runs.append({"text": line, "fontSize": size})
                y += height + line_gap
            return {"fontSize": size, "lines": lines, "runs": runs}
    raise ValueError(f"Arabic text does not fit box {box}: {text!r}")


def foreground_occupancy(image: Image.Image, box: Rect, *, background: tuple[int, int, int] | None = None) -> float:
    crop = np.asarray(image.crop((box.left, box.top, box.right, box.bottom)).convert("RGB"), dtype=np.int16)
    if crop.size == 0:
        return 0.0
    if background is None:
        border = np.concatenate((crop[0], crop[-1], crop[:, 0], crop[:, -1]), axis=0)
        background = tuple(int(value) for value in np.median(border, axis=0))
    delta = np.max(np.abs(crop - np.asarray(background, dtype=np.int16)), axis=2)
    return float(np.mean(delta > 24))


def image_sha256(image: Image.Image) -> str:
    return hashlib.sha256(image.convert("RGBA").tobytes()).hexdigest()


def image_file_sha256(path: str | Path) -> str:
    value = Path(path)
    return hashlib.sha256(value.read_bytes()).hexdigest()


def balanced_grid(
    count: int,
    *,
    canvas_width: int,
    columns: int,
    top: int,
    side_margin: int,
    gap: int,
    aspect_ratio: float = 9 / 16,
    label_height: int = 28,
    bottom_margin: int = 18,
) -> tuple[list[Rect], int]:
    if count <= 0 or columns <= 0:
        return [], top + bottom_margin
    tile_width = (canvas_width - side_margin * 2 - gap * (columns - 1)) // columns
    image_height = int(round(tile_width / aspect_ratio))
    row_height = image_height + label_height
    rows = int(math.ceil(count / columns))
    boxes: list[Rect] = []
    for row in range(rows):
        row_count = min(columns, count - row * columns)
        occupied = row_count * tile_width + max(0, row_count - 1) * gap
        row_left = (canvas_width - occupied) // 2
        for column in range(row_count):
            left = row_left + column * (tile_width + gap)
            y = top + row * (row_height + gap)
            boxes.append(Rect(left, y, left + tile_width, y + image_height))
    height = top + rows * row_height + max(0, rows - 1) * gap + bottom_margin
    return boxes, height


def cards_unique_hashes(card_art, card_ids: Iterable[int]) -> dict[int, str]:
    return {int(card_id): image_sha256(card_art.face(int(card_id))) for card_id in card_ids}
