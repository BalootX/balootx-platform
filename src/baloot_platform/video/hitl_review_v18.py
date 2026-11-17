from __future__ import annotations

import hashlib
import json
import math
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml
from PIL import Image, ImageDraw

from baloot_platform.config import platform_root
from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Seat

from .card_assets import create_card_art
from .media_validation import media_tool, run_media_command
from .typography import _font, _latin, font_contract
from .visual_qa import LayoutAudit, Rect, draw_arabic_fit


@dataclass(frozen=True, slots=True)
class HitlReviewClipArtifacts:
    video: Path
    poster: Path
    timeline: Path
    manifest: Path


class HitlReviewClipRendererV18:
    """Observer-safe, card-first review clip.

    The blind stage never displays the production recommendation, crowd
    consensus, hidden hands or challenger output.  A correct-action reveal is
    allowed only after the case explicitly declares that the reviewer submitted
    a choice.  This makes timing and disclosure policy machine-verifiable.
    """

    def __init__(self, profile: Mapping[str, Any]):
        self.profile = dict(profile)
        self._card_art_cache: dict[tuple[int, int], Any] = {}

    @staticmethod
    def _ffmpeg() -> str:
        return str(media_tool(platform_root(), "ffmpeg"))

    @staticmethod
    def _silent_wav(path: Path, seconds: float, sample_rate: int = 48000) -> None:
        frames = int(seconds * sample_rate)
        with wave.open(str(path), "wb") as value:
            value.setnchannels(2)
            value.setsampwidth(2)
            value.setframerate(sample_rate)
            value.writeframes(b"\0\0\0\0" * frames)

    def _card_art(self, width: int, height: int):
        key = (int(width), int(height))
        cached = self._card_art_cache.get(key)
        if cached is not None:
            return cached
        root = platform_root()
        video_profile = yaml.safe_load((root / "configs/video/premium.yaml").read_text(encoding="utf-8"))
        art = create_card_art(video_profile, key[0], key[1])
        self._card_art_cache[key] = art
        return art

    @staticmethod
    def _copy(cfg: Mapping[str, Any]) -> dict[str, str]:
        raw = cfg.get("copy")
        if not isinstance(raw, Mapping):
            raise ValueError("HITL review clip requires governed copy configuration")
        copy = {str(key): str(value) for key, value in raw.items()}
        if not copy or any(not value.strip() for value in copy.values()):
            raise ValueError("HITL review clip copy values must be non-empty strings")
        return copy

    @staticmethod
    def _actor_arabic(value: Any, copy: Mapping[str, str]) -> str:
        actor_key = value.name.lower() if isinstance(value, Seat) else str(value or "player").strip().lower()
        return copy.get(f"actor_{actor_key}", copy["actor_fallback"])

    @staticmethod
    def _card_name(card: Card, copy: Mapping[str, str]) -> str:
        rank_key = card.rank.short.lower()
        return f"{copy[f'rank_{rank_key}']} {copy[f'suit_{card.suit.name.lower()}']}"

    @staticmethod
    def _parse_actions(values: list[Any], *, maximum: int) -> list[Card]:
        cards: list[Card] = []
        for value in values[:maximum]:
            try:
                card = Card.parse(str(value))
            except Exception as exc:
                raise ValueError(f"HITL review action is not a valid Baloot card: {value!r}") from exc
            if card not in cards:
                cards.append(card)
        return cards

    @staticmethod
    def _layout(width: int, height: int, choice_count: int) -> tuple[LayoutAudit, dict[str, Rect], list[Rect]]:
        margin = max(24, width // 22)
        audit = LayoutAudit(width, height, safe_margin=margin)
        boxes = {
            "header": audit.add("header", Rect(margin + 10, margin + 10, width - margin - 10, int(height * 0.155))),
            "context": audit.add("context", Rect(margin * 2, int(height * 0.175), width - margin * 2, int(height * 0.265))),
            "prompt": audit.add("prompt", Rect(margin * 2, int(height * 0.285), width - margin * 5, int(height * 0.355))),
            "timer": audit.add("timer", Rect(width - margin * 4, int(height * 0.282), width - margin * 2, int(height * 0.357))),
            "reason": audit.add("reason", Rect(margin * 2, int(height * 0.71), width - margin * 2, int(height * 0.875))),
            "footer": audit.add("footer", Rect(margin * 2, int(height * 0.895), width - margin * 2, int(height * 0.945))),
        }
        columns = 2
        rows = max(1, math.ceil(max(1, choice_count) / columns))
        region = Rect(margin * 2, int(height * 0.375), width - margin * 2, int(height * 0.69))
        gap = max(14, width // 36)
        tile_width = (region.width - gap) // 2
        tile_height = (region.height - gap * (rows - 1)) // rows
        choice_boxes: list[Rect] = []
        for index in range(choice_count):
            row, column = divmod(index, columns)
            left = region.left + column * (tile_width + gap)
            top = region.top + row * (tile_height + gap)
            choice_boxes.append(audit.add(f"choice_{index}", Rect(left, top, left + tile_width, top + tile_height)))
        audit.assert_valid()
        return audit, boxes, choice_boxes

    @staticmethod
    def _timing(cfg: Mapping[str, Any], duration: float) -> dict[str, float]:
        intro = float(cfg.get("intro_seconds", 0.7))
        countdown = float(cfg.get("countdown_seconds", 5.0))
        selection = float(cfg.get("selection_seconds", 0.8))
        reveal = float(cfg.get("answer_reveal_seconds", 1.2))
        minimum = intro + countdown + selection + reveal + 0.6
        if duration < minimum:
            scale = max(0.01, duration / minimum)
            intro *= scale
            countdown *= scale
            selection *= scale
            reveal *= scale
        return {
            "introEnd": intro,
            "countdownEnd": intro + countdown,
            "selectionEnd": intro + countdown + selection,
            "revealEnd": intro + countdown + selection + reveal,
            "duration": duration,
            "countdownSeconds": float(cfg.get("countdown_seconds", 5.0)),
        }

    @staticmethod
    def _stage(now: float, timing: Mapping[str, float], *, reveal_allowed: bool) -> str:
        if now < timing["introEnd"]:
            return "intro"
        if now < timing["countdownEnd"]:
            return "blind_choice"
        if now < timing["selectionEnd"]:
            return "selection_confirmed"
        if reveal_allowed and now < timing["revealEnd"]:
            return "answer_reveal"
        return "reason_prompt"

    def _render_frame(
        self,
        *,
        case: Mapping[str, Any],
        width: int,
        height: int,
        time_seconds: float,
        duration: float,
        cfg: Mapping[str, Any],
    ) -> tuple[Image.Image, dict[str, Any]]:
        context = dict(case.get("publicContext") or {})
        maximum_choices = int(cfg.get("maximum_choices", 4))
        cards = self._parse_actions(list(case.get("legalActions") or []), maximum=maximum_choices)
        if len(cards) < 2:
            raise ValueError("HITL review clip requires at least two legal card choices")
        selected = Card.parse(str(case.get("selectedAction") or cards[0].code))
        review_submitted = bool(case.get("reviewSubmitted", False))
        correct_value = case.get("correctAction")
        correct = Card.parse(str(correct_value)) if review_submitted and correct_value else None
        reveal_allowed = bool(correct is not None and cfg.get("show_correct_after_submission", True))
        timing = self._timing(cfg, duration)
        copy = self._copy(cfg)
        stage = self._stage(time_seconds, timing, reveal_allowed=reveal_allowed)
        audit, boxes, choice_boxes = self._layout(width, height, len(cards))

        image = Image.new("RGB", (width, height), "#061714")
        draw = ImageDraw.Draw(image, "RGBA")
        outer = audit.safe_rect
        draw.rounded_rectangle((outer.left, outer.top, outer.right, outer.bottom), radius=32, fill=(7, 55, 46, 255), outline=(218, 178, 82, 255), width=3)

        header = boxes["header"]
        draw_arabic_fit(draw, Rect(header.left, header.top + 8, header.right, header.top + 62), copy["header_title"], font_factory=_font, fill="#F5C94C", max_size=max(22, width // 22), min_size=20, max_lines=1, bold=True, title=True)
        serial = str(case.get("caseId", "HITL"))
        if bool(cfg.get("show_case_locator", True)):
            maximum = int(cfg.get("case_locator_characters", 8))
            _latin(
                draw,
                (width / 2, header.top + 82),
                serial[:maximum],
                max(10, width // 60),
                "#8FA59D",
                bold=False,
            )

        phase_raw = str(context.get("phase") or "decision").strip()
        phase = copy.get(f"phase_{phase_raw.lower()}", phase_raw if any("\u0600" <= char <= "\u06ff" for char in phase_raw) else copy["phase_decision"])
        actor = self._actor_arabic(context.get("actor"), copy)
        context_box = boxes["context"]
        draw_arabic_fit(draw, Rect(context_box.left, context_box.top, context_box.right, context_box.top + context_box.height // 2), phase, font_factory=_font, fill="#FFF8E8", max_size=max(20, width // 28), min_size=16, max_lines=1, bold=True)
        draw_arabic_fit(draw, Rect(context_box.left, context_box.top + context_box.height // 2, context_box.right, context_box.bottom), copy["context_actor"].format(actor=actor), font_factory=_font, fill="#C5D4CE", max_size=max(15, width // 42), min_size=12, max_lines=1)

        prompt_text = copy[f"stage_{stage}"]
        prompt = boxes["prompt"]
        draw.rounded_rectangle((prompt.left, prompt.top, prompt.right, prompt.bottom), radius=20, fill=(3, 29, 24, 245), outline=(55, 112, 96, 255), width=2)
        draw_arabic_fit(draw, Rect(prompt.left + 12, prompt.top + 6, prompt.right - 12, prompt.bottom - 6), prompt_text, font_factory=_font, fill="#FFF8E8", max_size=max(18, width // 34), min_size=14, max_lines=2, bold=True)

        timer = boxes["timer"]
        draw.ellipse((timer.left, timer.top, timer.right, timer.bottom), fill=(8, 72, 59, 255), outline=(245, 201, 76, 255), width=3)
        if stage in {"intro", "blind_choice"}:
            elapsed = max(0.0, time_seconds - timing["introEnd"])
            remaining = max(1, int(math.ceil(timing["countdownSeconds"] - elapsed)))
            timer_text = str(remaining)
            _latin(draw, ((timer.left + timer.right) / 2, (timer.top + timer.bottom) / 2), timer_text, max(25, width // 24), "#FFF8E8", bold=True)
        elif stage in {"selection_confirmed", "answer_reveal"}:
            timer_text = "selected"
            cx = (timer.left + timer.right) / 2
            cy = (timer.top + timer.bottom) / 2
            scale = max(8, min(timer.width, timer.height) // 5)
            draw.line((cx - scale, cy, cx - scale // 3, cy + scale), fill="#FFF8E8", width=max(3, scale // 4))
            draw.line((cx - scale // 3, cy + scale, cx + scale, cy - scale), fill="#FFF8E8", width=max(3, scale // 4))
        else:
            timer_text = "info"
            _latin(draw, ((timer.left + timer.right) / 2, (timer.top + timer.bottom) / 2), "i", max(25, width // 24), "#FFF8E8", bold=True)

        # Card dimensions are constrained by the actual tile. Reserve only the
        # space enabled by the presentation profile.
        min_tile_width = min(box.width for box in choice_boxes)
        min_tile_height = min(box.height for box in choice_boxes)
        requested_width = int(cfg.get("choice_card_width", max(94, width // 6)))
        requested_height = int(cfg.get("choice_card_height", round(requested_width * 1.4)))
        available_width = max(80, min_tile_width - 20)
        show_codes = bool(cfg.get("show_developer_card_codes", False))
        code_space = 24 if show_codes else 0
        available_height = max(
            120,
            min_tile_height - 18 - 6 - 24 - code_space - 4,
        )
        ratio = requested_height / max(1, requested_width)
        card_width = max(80, min(requested_width, available_width, int(available_height / ratio)))
        card_height = max(120, min(requested_height, available_height, int(round(card_width * ratio))))
        art = self._card_art(card_width, card_height)
        choice_records: list[dict[str, Any]] = []
        for index, (card, box) in enumerate(zip(cards, choice_boxes)):
            is_selected = card.id == selected.id and stage in {"selection_confirmed", "answer_reveal", "reason_prompt"}
            is_correct = correct is not None and card.id == correct.id and stage in {"answer_reveal", "reason_prompt"}
            is_wrong_selected = is_selected and correct is not None and card.id != correct.id and stage in {"answer_reveal", "reason_prompt"}
            border = "#66DDB0" if is_correct else ("#F5C94C" if is_selected else "#376F62")
            if is_wrong_selected:
                border = "#F38A7C"
            fill = (7, 48, 40, 252) if is_selected or is_correct else (3, 29, 24, 244)
            draw.rounded_rectangle((box.left, box.top, box.right, box.bottom), radius=22, fill=fill, outline=border, width=4 if is_selected or is_correct else 2)
            face = art.face(card.id)
            face_x = box.left + (box.width - face.width) // 2
            face_y = box.top + 18
            image.paste(face, (face_x, face_y), face)
            name_top = face_y + face.height + 6
            name_bottom_margin = 30 if show_codes else 6
            name_box = Rect(
                box.left + 6,
                name_top,
                box.right - 6,
                min(box.bottom - name_bottom_margin, name_top + 30),
            )
            card_name = self._card_name(card, copy)
            draw_arabic_fit(draw, name_box, card_name, font_factory=_font, fill="#FFF8E8", max_size=max(12, width // 48), min_size=9, max_lines=1, bold=True)
            if show_codes:
                _latin(
                    draw,
                    ((box.left + box.right) / 2, box.bottom - 16),
                    card.code,
                    max(11, width // 56),
                    "#F5C94C",
                    bold=True,
                )
            choice_records.append({
                "code": card.code,
                "nameAr": card_name,
                "box": box.to_dict(),
                "selected": is_selected,
                "correct": is_correct,
            })

        reason = boxes["reason"]
        draw.rounded_rectangle((reason.left, reason.top, reason.right, reason.bottom), radius=22, fill=(3, 27, 23, 238), outline=(245, 201, 76, 235), width=3)
        if stage == "answer_reveal" and correct is not None:
            explanation = str(case.get("explanationAr") or copy["reveal_fallback"])
            draw_arabic_fit(draw, Rect(reason.left + 18, reason.top + 15, reason.right - 18, reason.bottom - 15), explanation, font_factory=_font, fill="#FFF8E8", max_size=max(15, width // 44), min_size=12, max_lines=3, bold=True)
        else:
            main = copy["reason_main"]
            secondary = copy["reason_secondary"]
            draw_arabic_fit(draw, Rect(reason.left + 18, reason.top + 15, reason.right - 18, reason.top + reason.height // 2), main, font_factory=_font, fill="#FFF8E8", max_size=max(15, width // 44), min_size=12, max_lines=2, bold=True)
            draw_arabic_fit(draw, Rect(reason.left + 18, reason.top + reason.height // 2, reason.right - 18, reason.bottom - 15), secondary, font_factory=_font, fill="#C5D4CE", max_size=max(13, width // 52), min_size=11, max_lines=2)

        footer = boxes["footer"]
        footer_text = copy["footer_submitted"] if review_submitted else copy["footer_blind"]
        draw_arabic_fit(draw, footer, footer_text, font_factory=_font, fill="#C5D4CE", max_size=13, min_size=10, max_lines=1)

        frame_meta = {
            "stage": stage,
            "timeSeconds": round(float(time_seconds), 6),
            "presentationMode": str(cfg.get("presentation_mode", "player")),
            "developerCodesVisible": show_codes,
            "caseLocatorCharacters": (
                int(cfg.get("case_locator_characters", 8))
                if bool(cfg.get("show_case_locator", True))
                else 0
            ),
            "countdown": timer_text if timer_text.isdigit() else None,
            "selectedAction": selected.code if stage in {"selection_confirmed", "answer_reveal", "reason_prompt"} else None,
            "correctAction": correct.code if stage in {"answer_reveal", "reason_prompt"} and correct is not None else None,
            "choices": choice_records,
            "layout": audit.to_dict(),
            "ownerAr": actor,
        }
        return image, frame_meta

    def render(
        self,
        *,
        case: Mapping[str, Any],
        output_dir: str | Path,
        basename: str | None = None,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
        duration: float | None = None,
    ) -> HitlReviewClipArtifacts:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        basename = basename or f"hitl-review-{case['caseId']}"
        cfg = dict(self.profile.get("review_clip") or self.profile)
        width = int(width or cfg.get("width", 720))
        height = int(height or cfg.get("height", 1280))
        fps = int(fps or cfg.get("fps", 30))
        duration = float(duration or cfg.get("duration_seconds", 9.0))
        timing = self._timing(cfg, duration)
        duration = float(timing["duration"])
        if (
            width < 320
            or height < 320
            or width * 16 != height * 9
            or fps not in {24, 25, 30, 50, 60}
        ):
            raise ValueError("invalid review clip dimensions or frame rate")
        video = output / f"{basename}.mp4"
        poster = output / f"{basename}.jpg"
        timeline_path = output / f"{basename}-timeline.json"
        manifest_path = output / f"{basename}-manifest.json"
        total_frames = max(1, int(math.ceil(duration * fps)))
        timeline: list[dict[str, Any]] = []
        stage_segments: list[dict[str, Any]] = []
        with tempfile.TemporaryDirectory(prefix=".hitl-render-", dir=output) as temp_name:
            temp = Path(temp_name)
            frame_list = temp / "frames.txt"
            audio = temp / "audio.wav"
            first_frame: Image.Image | None = None
            last_stage: str | None = None
            stage_start = 0.0
            with frame_list.open("w", encoding="utf-8") as listing:
                for index in range(total_frames):
                    now = index / fps
                    image, meta = self._render_frame(case=case, width=width, height=height, time_seconds=now, duration=duration, cfg=cfg)
                    if meta["stage"] != last_stage:
                        if last_stage is not None:
                            stage_segments.append({"stage": last_stage, "startSeconds": stage_start, "endSeconds": round(now, 6)})
                        last_stage = str(meta["stage"])
                        stage_start = round(now, 6)
                    frame = temp / f"frame-{index:06d}.png"
                    image.save(frame, compress_level=1)
                    if first_frame is None:
                        first_frame = image.copy()
                    listing.write(f"file '{frame.as_posix()}'\n")
                    listing.write(f"duration {1 / fps:.12f}\n")
                    timeline.append({"frame": index, **meta})
                listing.write(f"file '{frame.as_posix()}'\n")
            if last_stage is not None:
                stage_segments.append({"stage": last_stage, "startSeconds": stage_start, "endSeconds": duration})
            self._silent_wav(audio, duration + 1 / fps)
            command = [
                self._ffmpeg(), "-y", "-f", "concat", "-safe", "0", "-i", str(frame_list), "-i", str(audio),
                "-r", str(fps), "-fps_mode", "cfr", "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
                "-pix_fmt", "yuv420p", "-color_range", "tv", "-colorspace", "bt709",
                "-color_primaries", "bt709", "-color_trc", "bt709",
                "-x264-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709:range=limited:threads=1:lookahead_threads=1:sliced_threads=0",
                "-threads:v", "1", "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-threads:a", "1",
                "-fflags", "+bitexact", "-flags:v", "+bitexact", "-flags:a", "+bitexact", "-map_metadata", "-1",
                "-metadata", "creation_time=1980-01-01T00:00:00Z", "-metadata", "encoder=BalootPlatform",
                "-max_interleave_delta", "0", "-movflags", "+faststart", "-shortest", str(video),
            ]
            result = run_media_command(command, text=True, capture_output=True)
            if result.returncode:
                raise RuntimeError(result.stderr[-4000:])
            assert first_frame is not None
            first_frame.save(poster, quality=94, subsampling=0)

        timeline_path.write_text(
            json.dumps({"schemaVersion": 2, "segments": stage_segments, "frames": timeline}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        legal_codes = [Card.parse(str(value)).code for value in list(case.get("legalActions") or [])]
        review_submitted = bool(case.get("reviewSubmitted", False))
        correct_included = bool(review_submitted and case.get("correctAction"))
        manifest = {
            "schemaVersion": 20,
            "renderer": "hitl_review_v18",
            "caseId": str(case.get("caseId", "HITL")),
            "observerSafe": True,
            "hiddenHandsIncluded": False,
            "productionRecommendationIncluded": correct_included,
            "productionRecommendationBeforeSubmission": False,
            "answerRevealAfterSubmission": correct_included,
            "crowdConsensusIncluded": False,
            "width": width,
            "height": height,
            "fps": fps,
            "durationSeconds": duration,
            "countdownSeconds": float(timing["countdownSeconds"]),
            "choicePresentation": "card_tiles",
            "presentationMode": str(cfg.get("presentation_mode", "player")),
            "developerCodesVisible": bool(
                cfg.get("show_developer_card_codes", False)
            ),
            "caseLocatorCharacters": (
                int(cfg.get("case_locator_characters", 8))
                if bool(cfg.get("show_case_locator", True))
                else 0
            ),
            "legalChoices": legal_codes,
            "selectedAction": str(case.get("selectedAction") or legal_codes[0]),
            "correctAction": str(case.get("correctAction")) if correct_included else None,
            "layoutContract": self._layout(width, height, min(len(legal_codes), int(cfg.get("maximum_choices", 4))))[0].to_dict(),
            "fontContract": font_contract(),
            "videoSha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "posterSha256": hashlib.sha256(poster.read_bytes()).hexdigest(),
            "timelineSha256": hashlib.sha256(timeline_path.read_bytes()).hexdigest(),
        }
        manifest["manifestHash"] = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return HitlReviewClipArtifacts(video, poster, timeline_path, manifest_path)
