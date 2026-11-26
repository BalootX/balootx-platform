"""Governed Remotion video-renderer plugin.

The adapter is the only trusted-to-observer-safe boundary. The Node renderer
receives only a verified :class:`RenderBundleV1`; it never receives an engine
state, deck, seed, hidden hand, or ``EpisodeSpec``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import signal
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from threading import Event
from typing import Any, BinaryIO, Sequence

import yaml
from PIL import Image, ImageOps

from baloot_platform.config import platform_root, validate_profile_payload
from baloot_platform.challenge.evidence_v3 import (
    DecisionPerspectiveEvidenceV3,
    ObserverProjectionV1,
    RenderVariantV1,
)
from baloot_platform.core.scenario import EpisodeSpec
from baloot_platform.core.rules import RuleBook
from baloot_platform.filesystem import atomic_write_text
from baloot_platform.plugins.base import PluginDescriptor, RenderArtifacts

from .render_bundle import (
    RenderBundleV1,
    build_export_timeline,
    build_render_bundle,
    seeded_player_names,
)
from .audio_postprocess import finish_remotion_audio
from .media_validation import (
    APPROVED_LAYOUT_TABLE,
    APPROVED_LAYOUT_TABLE_SHA256,
    REMOTION_RENDERER_VERSION,
    compare_source_frame,
    media_tool,
    probe_media,
    validate_media_probe,
)
from .perspective_adapter import build_perspective_decision_scene_package
from .scenario_adapter import build_scenario_decision_scene_package
from .scene_adapter import build_legacy_episode_scene_package


_CARD_LABEL = re.compile(r"(?:10|[2-9JQKA])[♠♥♦♣]")
_LTR_ISOLATE = "\u2066"
_POP_DIRECTIONAL_ISOLATE = "\u2069"
_LAYOUT_PHASES = ("hook", "question", "reveal")
WORKER_RENDERER_ROOT = Path(sysconfig.get_path("purelib")) / "baloot_platform/video/remotion"


def _localized_template_value(value: object) -> str:
    """Keep compact card labels intact inside Arabic localization runs."""
    text = str(value)
    if _CARD_LABEL.fullmatch(text):
        return f"{_LTR_ISOLATE}{text}{_POP_DIRECTIONAL_ISOLATE}"
    return text


def _format_localized_template(template: str, values: Mapping[str, object]) -> str:
    try:
        return str(template).format_map(
            {
                str(key): _localized_template_value(value)
                for key, value in values.items()
            }
        )
    except KeyError as exc:
        raise ValueError(
            f"dynamic reveal reason is missing template value {exc.args[0]}"
        ) from exc


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _json_write(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _qualified_layout_evidence(
    work: Path, *, bundle_path: Path, bundle_sha256: str
) -> dict[str, Any]:
    evidence = json.loads((work / "layout-evidence.json").read_text(encoding="utf-8"))
    binding = evidence.get("binding", {})
    if binding.get("mode") != "qualification" or evidence.get("disposition") == "NEVER_POST":
        raise ValueError("layout measurement MP4 is permanently NEVER_POST")
    table_path = work / "layout-table.json"
    if (
        evidence.get("schemaVersion") != 1
        or evidence.get("status") != "passed"
        or binding.get("tableSha256") != APPROVED_LAYOUT_TABLE_SHA256
        or _sha256(table_path) != APPROVED_LAYOUT_TABLE_SHA256
        or binding.get("bundleFileSha256") != _sha256(bundle_path)
        or binding.get("bundleSha256") != bundle_sha256
        or evidence.get("videoSha256") != _sha256(work / "render.mp4")
    ):
        raise ValueError("qualification receipt must bind the exact layout table, bundle and rendered MP4")
    rows = evidence.get("items", [])
    if tuple(row.get("phase") for row in rows) != _LAYOUT_PHASES:
        raise ValueError("qualified layout checkpoint coverage is incomplete")
    for row in rows:
        source = row.get("source", "")
        validation = row.get("validation", {})
        if (
            not source or Path(source).name != source or not source.endswith(".png")
            or row.get("binding") != binding
            or validation.get("tableSha256") != APPROVED_LAYOUT_TABLE_SHA256
            or validation.get("status") != "passed"
            or _sha256(work / source) != row.get("pngSha256")
        ):
            raise ValueError("qualified layout checkpoint binding mismatch")
    return evidence


def _process_exists(pid: int) -> bool:
    """Report live processes, excluding exited Linux zombies awaiting reaping."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(
            process_query_limited_information,
            False,
            pid,
        )
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return ctypes.get_last_error() == 5  # Access denied still means it exists.
    try:
        # comm may contain spaces and parentheses; the state follows its last ')'.
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
        if stat.rsplit(")", 1)[1].split()[0] == "Z":
            return False
    except (OSError, IndexError):
        pass  # Non-Linux hosts and restricted procfs still use the signal probe.
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _terminate_process_tree(process: subprocess.Popen[Any]) -> None:
    """Stop one owned renderer process and all descendants."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                capture_output=True,
                timeout=30,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=10)


def _timestamp(frames: int, fps: int, *, srt: bool = False) -> str:
    milliseconds = round(frames * 1000 / fps)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    separator = "," if srt else "."
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{millis:03d}"


_COMPLETE_REPLAY_PHASES = ("hook", "setup", "replay", "question", "reveal", "hold")
# Replay clips that play on to the end of the round and its final score.
_ROUND_END_REPLAY_PHASES = (*_COMPLETE_REPLAY_PHASES[:-1], "continuation", "outcome")


def _complete_replay_checkpoints(
    segments: Mapping[str, Mapping[str, Any]], *, pause: int, fps: int, setup_actions: int,
) -> tuple[list[int], list[int]]:
    """Evidence frames and play counts for a replay decision clip (mirrors render-decision-clip.mjs)."""
    hook, setup, replay, question, reveal = (segments[stage] for stage in _COMPLETE_REPLAY_PHASES[:5])
    half = pause // 2
    sample = max(1, half - (half % 4 == 0))  # mid-trick, so no collected trick is sampled
    frames = [
        hook["startFrame"] + min(fps, hook["durationFrames"] - 1),
        setup["startFrame"] + setup["durationFrames"] // (2 * max(1, setup_actions)),
        replay["startFrame"] + -(-sample * replay["durationFrames"] // pause) - 1,
        question["startFrame"] + question["durationFrames"] // 2,
        reveal["startFrame"] + reveal["durationFrames"] // 2,
    ]
    counts = [pause, 0, sample, pause, pause]
    if "outcome" in segments:
        continuation, outcome = segments["continuation"], segments["outcome"]
        frames += [
            continuation["startFrame"] + min(continuation["durationFrames"] - 1,
                                             math.ceil(continuation["durationFrames"] / (32 - pause)) - 1),
            outcome["startFrame"] + outcome["durationFrames"] // 2,
        ]
        return frames, [*counts, pause + 1, 32]
    hold = segments["hold"]
    return [*frames, hold["startFrame"] + hold["durationFrames"] - 1], [*counts, pause + 1]


def _output_tail(stream: BinaryIO) -> str:
    # UTF-8 needs at most four bytes per character; keep diagnostics bounded
    # even when a failed renderer has emitted a large amount of progress output.
    stream.seek(0, os.SEEK_END)
    stream.seek(max(0, stream.tell() - 16000))
    return stream.read(16000).decode("utf-8", errors="replace").strip()[-4000:]


class RemotionProfessionalRenderer:
    """Process-isolated renderer for governed professional video exports."""

    descriptor = PluginDescriptor(
        id="premium_video",
        kind="video_renderer",
        version=REMOTION_RENDERER_VERSION,
        description="Observer-safe Remotion renderer for complete professional replays and decision clips",
        capabilities=(
            "full_baloot_lifecycle",
            "hooked_flagship",
            "decision_clip",
            "render_bundle_v1",
            "scene_document_v1",
            "scene_timeline_v1",
            "observer_safe",
            "arbitrary_observer_perspective",
            "typed_seat_experience",
            "historical_actual",
            "replay_audit",
            "project_timing",
            "physical_trick_collection",
            "shared_browser_render_session",
            "vetted_raster_card_faces",
            "arabic_rtl",
            "reduced_motion",
            "1080x1920",
            "60fps",
            "captions",
            "governed_exact_match_voice",
            "deterministic_original_foley",
            "governed_loudness_normalization",
            "contact_sheet",
            "decode_validation",
            "bt709_tagging",
            "qr_decode_validation",
            "provenance_manifest",
        ),
        resource_class="video-cpu-premium",
        deterministic=True,
        isolation="process",
        side_effects=("filesystem_output",),
        timeout_seconds=900,
        maximum_memory_mb=4096,
        compatible_profile_schemas=("1",),
    )

    def __init__(
        self,
        video_profile: Mapping[str, Any],
        *,
        repository_root: str | Path | None = None,
        timeout_seconds: int | None = None,
        use_worker_renderer: bool = False,
    ):
        if timeout_seconds is not None:
            if (
                not isinstance(timeout_seconds, int)
                or isinstance(timeout_seconds, bool)
                or timeout_seconds < 1
            ):
                raise ValueError("render timeout must be a positive integer")
            self.descriptor = replace(
                self.descriptor,
                timeout_seconds=timeout_seconds,
            )
        self.repository_root = platform_root(repository_root)
        self.profile = validate_profile_payload(
            "video",
            dict(video_profile),
            root=self.repository_root,
            source="<remotion-renderer-profile>",
        )
        source_renderer = self.repository_root / "src/baloot_platform/video/remotion"
        packaged_renderer = Path(__file__).resolve().parent / "remotion"
        self.use_worker_renderer = use_worker_renderer
        if use_worker_renderer:
            self.renderer_root = WORKER_RENDERER_ROOT
        else:
            self.renderer_root = source_renderer if source_renderer.is_dir() else packaged_renderer
        if not (self.renderer_root / "scripts/render-flagship.mjs").is_file():
            raise FileNotFoundError("Remotion renderer package is missing")
        self.ffprobe: Path | None = None
        self.ffmpeg: Path | None = None

    def _ffprobe(self) -> Path:
        if self.ffprobe is None:
            self.ffprobe = media_tool(self.repository_root, "ffprobe")
        return self.ffprobe

    def _ffmpeg(self) -> Path:
        if self.ffmpeg is None:
            self.ffmpeg = media_tool(self.repository_root, "ffmpeg")
        return self.ffmpeg

    @staticmethod
    def _playback_state_evidence(
        bundle: RenderBundleV1,
        checkpoints: Mapping[str, Any],
        *,
        require_rendered_dom: bool = False,
        clip_evidence: bool = False,
    ) -> list[dict[str, Any]]:
        context = next(
            (scene for scene in bundle.value["scenes"] if scene["stage"] == "continuation"),
            None,
        )
        if not isinstance(context, Mapping):
            raise ValueError("playback evidence requires a post-reveal continuation scene")
        layers = {str(layer["id"]): dict(layer["content"]) for layer in context["layers"]}
        lifecycle = layers.get("public-lifecycle")
        observer = layers.get("observer-lifecycle")
        if not isinstance(lifecycle, Mapping) or not isinstance(observer, Mapping):
            raise ValueError("playback evidence requires governed lifecycle layers")
        tricks = list(lifecycle.get("completedTricks") or [])
        plays = list(lifecycle.get("plays") or [])
        if not plays:
            plays = [play for trick in tricks for play in trick.get("plays", [])]
        initial_hand = list(observer.get("initialHandCards") or [])
        initial_five = list(observer.get("initialFiveCards") or [])
        has_setup = any(segment.get("stage") == "setup" for segment in bundle.value["timeline"]["segments"])
        if has_setup and (len(initial_five) != 5 or not {card["id"] for card in initial_five} <= {card["id"] for card in initial_hand}):
            raise ValueError("playback opening hand disagrees with the observer's deal")
        if len(plays) != 32 or len(tricks) != 8 or len(initial_hand) != 8:
            raise ValueError("playback evidence requires a complete eight-trick trace")
        observer_seat = str(bundle.value["observerSeat"])
        items = checkpoints.get("items")
        segments = {row["stage"]: row for row in bundle.value["timeline"]["segments"]}
        expected_counts = list(range(33))
        complete_replay = clip_evidence and "replay" in segments
        to_round_end = complete_replay and "outcome" in segments
        if complete_replay:
            pause = lifecycle.get("pauseBeforePlayIndex")
            if type(pause) is not int or not 1 <= pause < 32 or not has_setup:
                raise ValueError("decision-clip replay evidence requires recorded bidding and an interior play")
            expected_frames, expected_counts = _complete_replay_checkpoints(
                segments, pause=pause, fps=int(bundle.value["render"]["fps"]),
                setup_actions=len(lifecycle.get("bidHistory") or ()) + 1 + len(lifecycle.get("escalationHistory") or ()),
            )
            if (not isinstance(items, list)
                    or [row.get("phase") for row in items]
                    != list(_ROUND_END_REPLAY_PHASES if to_round_end else _COMPLETE_REPLAY_PHASES)
                    or [row.get("frame") for row in items] != expected_frames):
                raise ValueError(
                    "decision-clip replay evidence must cover hook, setup, replay, question, reveal "
                    "and hold (or continuation and outcome)"
                )
        elif clip_evidence:
            pause = lifecycle.get("pauseBeforePlayIndex")
            if type(pause) is not int or not 1 <= pause < 32:
                raise ValueError("decision-clip evidence requires an interior play decision")
            continuation = segments["continuation"]
            expected_frames = [segments[stage]["startFrame"] + segments[stage]["durationFrames"] // 2
                               for stage in ("context", "question", "reveal")]
            expected_frames[0] = segments["context"]["startFrame"] + segments["context"]["durationFrames"] - 1
            expected_frames.extend([continuation["startFrame"] + min(continuation["durationFrames"] - 1,
                math.ceil(continuation["durationFrames"] / (32 - pause)) - 1),
                continuation["startFrame"] + continuation["durationFrames"] - 1])
            expected_counts = [pause, pause, pause, pause + 1, 32]
            phases = ["context", "question", "reveal", "continuation", "continuation_end"]
            if "outcome" in segments:
                outcome = segments["outcome"]
                expected_frames.append(outcome["startFrame"] + outcome["durationFrames"] // 2)
                expected_counts.append(32)
                phases.append("outcome")
            if (not isinstance(items, list) or [row.get("phase") for row in items] != phases
                    or [row.get("frame") for row in items] != expected_frames):
                raise ValueError("decision-clip evidence must cover context, question, reveal and continuation")
        elif require_rendered_dom or (
            isinstance(items, list) and any(row.get("phase") == "outcome" for row in items)
        ):
            outcome = segments.get("outcome")
            if (
                outcome is None or not isinstance(items, list) or len(items) != 34
                or any(row.get("phase") == "outcome" for row in items[:-1])
                or items[-1].get("phase") != "outcome"
                or items[-1].get("frame") != outcome["startFrame"] + outcome["durationFrames"] // 2
                or items[-1].get("source") in {row.get("source") for row in items[:-1]}
            ):
                raise ValueError("full playback evidence must cover outcome at its midpoint with a unique source")
            expected_counts.append(32)
        if not isinstance(items, list) or [item.get("completedPlayCount") for item in items] != expected_counts:
            raise ValueError("playback evidence must cover every play boundary")
        evidence: list[dict[str, Any]] = []
        for item in items:
            count = int(item["completedPlayCount"])
            visible = plays[:count]
            remaining = [
                card
                for card in initial_hand
                if not any(
                    play["seat"] == observer_seat
                    and int(play["card"]["id"]) == int(card["id"])
                    for play in visible
                )
            ]
            hand_counts = {
                seat: 8 - sum(play["seat"] == seat for play in visible)
                for seat in ("north", "west", "south", "east")
            }
            position = count % 4
            current = [] if position == 0 else visible[-position:]
            if count > 0 and position == 0:
                table = list(tricks[count // 4 - 1]["plays"])
            else:
                table = current
            if clip_evidence and item["phase"] in {"hook", "question", "reveal"}:
                table = current
            # The final hold keeps the answered trick on the table, uncollected.
            if item.get("phase") == "hold":
                current = table
            if item.get("phase") == "outcome":
                table = []
            dom = item.get("renderedDom")
            if require_rendered_dom and dom is None:
                raise ValueError("playback DOM checkpoint is missing")
            if dom is not None:
                if not isinstance(dom, Mapping):
                    raise ValueError("playback DOM checkpoint identity is invalid")
                if item.get("phase") == "outcome":
                    expected_score = list(lifecycle["matchScore"])
                    if observer_seat in {"west", "east"}:
                        expected_score.reverse()
                    if dom.get("displayedScore") != expected_score:
                        raise ValueError("playback outcome HUD disagrees with terminal receipt score")
                if (
                    not isinstance(dom, Mapping)
                    or type(dom.get("frame")) is not int or dom["frame"] != item["frame"]
                    or type(dom.get("completedPlayCount")) is not int or dom["completedPlayCount"] != count
                    or not isinstance(dom.get("cards"), list)
                ):
                    raise ValueError("playback DOM checkpoint identity mismatch")
                cards = dom["cards"]
                if any(not isinstance(card, Mapping) for card in cards) or [
                    card.get("cardId") for card in cards
                ] != [int(play["card"]["id"]) for play in table]:
                    raise ValueError("playback DOM cards disagree with trace")
                for card, play in zip(cards, table, strict=True):
                    opacity = card.get("opacity")
                    if (type(card.get("cardId")) is not int
                        or card.get("cardCode") != play["card"]["code"]
                        or card.get("imageLoaded") is not True
                        or type(opacity) not in {int, float} or not 0 <= opacity <= 1
                        or type(card.get("visible")) is not bool
                        or (opacity == 0 and card["visible"])):
                        raise ValueError("playback DOM visibility evidence is invalid")
                if [card["cardId"] for card in cards if card["visible"]] != [
                    int(play["card"]["id"]) for play in current
                ]:
                    raise ValueError("playback DOM visible cards disagree with trace")
                hands = dom.get("hands")
                if (not isinstance(hands, list) or len(hands) != 4
                    or any(not isinstance(hand, Mapping) for hand in hands)
                    or {hand.get("seat") for hand in hands} != set(hand_counts)):
                    raise ValueError("playback DOM hand coverage is invalid")
                for hand in hands:
                    seat = hand["seat"]
                    faces = hand.get("faceCardIds")
                    opening = has_setup and count == 0
                    expected_faces = sorted(int(card["id"]) for card in (initial_five if opening else remaining)) if seat == observer_seat else []
                    expected_count = 5 if opening else hand_counts[seat]
                    if (type(hand.get("count")) is not int or hand["count"] != expected_count
                        or not isinstance(faces, list) or any(type(card) is not int for card in faces)
                        or sorted(faces) != expected_faces):
                        raise ValueError("playback DOM hand violates trace or observer safety")
            evidence.append(
                {
                    "completedPlayCount": count,
                    "frame": int(item["frame"]),
                    "source": str(item["source"]),
                    "playedCard": None if count == 0 else dict(plays[count - 1]),
                    "observerHandCardIds": sorted(int(card["id"]) for card in (initial_five if has_setup and count == 0 else remaining)),
                    "handCounts": {seat: 5 for seat in hand_counts} if has_setup and count == 0 else hand_counts,
                    "currentTrickCardIds": [int(play["card"]["id"]) for play in current],
                    "logicalTableCardIds": [int(play["card"]["id"]) for play in table],
                    "evidenceScope": "decision_clip_state_and_encoded_frame_parity" if clip_evidence else "trace_state_and_encoded_frame_parity",
                    **({"checkpointPhase": item["phase"]} if item.get("phase") is not None else {}),
                    "visibleCardStateVerified": False,
                    **({
                        "renderedDom": dict(dom),
                        "observedVisibleTableCardIds": [card["cardId"] for card in dom["cards"] if card["visible"]],
                    } if dom is not None else {}),
                    "completedTrickIndex": count // 4 - 1 if count > 0 and position == 0 else None,
                }
            )
        if (not complete_replay or to_round_end) and (evidence[-1]["observerHandCardIds"] or set(evidence[-1]["handCounts"].values()) != {0}):
            raise ValueError("playback evidence does not finish with empty hands")
        return evidence

    @classmethod
    def from_yaml(cls, path: str | Path) -> RemotionProfessionalRenderer:
        source = Path(path).resolve()
        return cls(
            yaml.safe_load(source.read_text(encoding="utf-8")),
            repository_root=source.parents[2],
        )

    def _profile(self, override: Mapping[str, Any] | None) -> dict[str, Any]:
        value = dict(override) if override is not None else dict(self.profile)
        return validate_profile_payload(
            "video",
            value,
            root=self.repository_root,
            source="<remotion-render-request-profile>",
        )

    def localized_reveal_reason(
        self,
        key: str,
        values: Mapping[str, object],
        *,
        video_profile: Mapping[str, Any] | None = None,
    ) -> str:
        profile = self._profile(video_profile)
        template = dict(profile["localizations"]).get(str(key))
        if not isinstance(template, str) or not template.strip():
            raise ValueError(
                "dynamic reveal reason requires a governed localization key"
            )
        return _format_localized_template(template, values)

    def build_render_request(
        self,
        request: EpisodeSpec,
        *,
        export_preset: str,
        video_profile: Mapping[str, Any] | None = None,
    ) -> RenderBundleV1:
        profile = self._profile(video_profile)
        package = build_legacy_episode_scene_package(request, profile)
        timeline = build_export_timeline(
            scenes=package.scenes,
            profile=profile,
            export_preset=export_preset,
        )
        question = package.scene("question")
        serial = str(request.game_serial or request.episode_id)
        identity_profile = profile["identity"]
        identity = {
            "gameSerial": serial,
            "sceneId": question.scene_id,
            "enrollmentUri": str(identity_profile["enrollment_uri_template"]).format(
                game_serial=serial
            ),
            "players": seeded_player_names(identity_profile, serial),
        }
        return build_render_bundle(
            scenes=package.scenes,
            timeline=timeline,
            profile=profile,
            export_preset=export_preset,
            localizations=profile["localizations"],
            identity=identity,
            repository_root=self.repository_root,
            decision_index=request.decision_index,
        )

    def build_perspective_render_request(
        self,
        *,
        projection: ObserverProjectionV1,
        evidence: DecisionPerspectiveEvidenceV3,
        variant: RenderVariantV1,
        effective_config_sha256: str,
        game_serial: str,
        legal_actions: tuple[Mapping[str, Any], ...] = (),
        reveal_reason_key: str | None = None,
        reveal_reason_values: Mapping[str, str] | None = None,
        player_names: Mapping[str, str] | None = None,
        video_profile: Mapping[str, Any] | None = None,
        export_preset: str = "decision_clip",
        complete_public_lifecycle: Mapping[str, Any] | None = None,
        complete_observer_lifecycle: Mapping[str, Any] | None = None,
        source_metadata: Mapping[str, Any] | None = None,
    ) -> RenderBundleV1:
        """Build one governed export for an explicitly isolated perspective."""

        if export_preset not in {"decision_clip", "hooked_flagship"}:
            raise ValueError(f"unsupported perspective export preset: {export_preset}")
        if export_preset == "hooked_flagship" and (
            complete_public_lifecycle is None or complete_observer_lifecycle is None
        ):
            raise ValueError(
                "hooked flagship requires a complete observer-safe Laffa lifecycle"
            )

        profile = self._profile(video_profile)
        package = build_perspective_decision_scene_package(
            projection=projection,
            evidence=evidence,
            variant=variant,
            profile=profile,
            effective_config_sha256=effective_config_sha256,
            legal_actions=legal_actions,
            reveal_reason_key=reveal_reason_key,
            complete_public_lifecycle=complete_public_lifecycle,
            complete_observer_lifecycle=complete_observer_lifecycle,
        )
        timeline = build_export_timeline(
            scenes=package.scenes,
            profile=profile,
            export_preset=export_preset,
        )
        question = package.scene("question")
        identity = {
            "gameSerial": str(game_serial),
            "sceneId": question.scene_id,
            "enrollmentUri": variant.challenge_locator,
            "players": dict(
                player_names
                or seeded_player_names(profile["identity"], str(game_serial))
            ),
        }
        localizations = dict(profile["localizations"])
        if reveal_reason_values is not None:
            if reveal_reason_key is None or reveal_reason_key not in localizations:
                raise ValueError(
                    "dynamic reveal reason requires a governed localization key"
                )
            localizations[reveal_reason_key] = _format_localized_template(
                str(localizations[reveal_reason_key]), reveal_reason_values
            )
        return build_render_bundle(
            scenes=package.scenes,
            timeline=timeline,
            profile=profile,
            export_preset=export_preset,
            localizations=localizations,
            identity=identity,
            repository_root=self.repository_root,
            decision_index=((variant.rendition_type, evidence.decision_id),),
            source_metadata=source_metadata,
        )

    def build_scenario_render_request(
        self,
        *,
        exact_state: Mapping[str, Any],
        proof_outcome: Mapping[str, Any],
        scenario_projection: Mapping[str, Any],
        effective_config_sha256: str,
        game_serial: str,
        observer_seat: str,
        enrollment_uri: str,
        export_preset: str = "decision_clip",
        source_metadata: Mapping[str, Any],
        rules: RuleBook,
        video_profile: Mapping[str, Any] | None = None,
        trusted_principal_variation: Sequence[str] | None = None,
        trusted_complete_trace: Mapping[str, Any] | None = None,
    ) -> RenderBundleV1:
        """Build a governed V4 scenario bundle from observer-safe evidence."""

        profile = self._profile(video_profile)
        package = build_scenario_decision_scene_package(
            exact_state=exact_state,
            proof_outcome=proof_outcome,
            scenario_projection=scenario_projection,
            profile=profile,
            effective_config_sha256=effective_config_sha256,
            observer_seat=observer_seat,
            rules=rules,
            trusted_principal_variation=trusted_principal_variation,
            trusted_complete_trace=trusted_complete_trace,
        )
        timeline = build_export_timeline(
            scenes=package.scenes,
            profile=profile,
            export_preset=export_preset,
        )
        question = package.scene("question")
        localizations = dict(profile["localizations"])
        reveal = package.scene("reveal")
        analysis = next(
            (layer["content"] for layer in reveal.value["layers"] if layer["id"] == "analysis"),
            {},
        )
        if "reasonValues" in analysis:
            key = analysis["reasonKey"]
            localizations[key] = _format_localized_template(
                localizations[key], analysis["reasonValues"]
            )
        if "proofScopeValues" in analysis:
            key = analysis["proofScopeKey"]
            localizations[key] = _format_localized_template(localizations[key], analysis["proofScopeValues"])
        return build_render_bundle(
            scenes=package.scenes,
            timeline=timeline,
            profile=profile,
            export_preset=export_preset,
            localizations=localizations,
            identity={
                "gameSerial": game_serial,
                "sceneId": question.scene_id,
                "enrollmentUri": enrollment_uri,
                "players": seeded_player_names(profile["identity"], game_serial),
            },
            repository_root=self.repository_root,
            decision_index=(
                (
                    "scenario_v4_" + str(scenario_projection["decisionKind"]),
                    f"{game_serial}:1",
                ),
            ),
            source_metadata=source_metadata,
        )

    @staticmethod
    def _reveal_reason(bundle: RenderBundleV1) -> str:
        if bundle.value["presentation"]["presentation"].get("show_justification_text") is False:
            return ""
        reveal = next(
            (scene for scene in bundle.value["scenes"] if scene["stage"] == "reveal"),
            None,
        )
        if reveal is None:
            raise ValueError("render bundle is missing its reveal scene")
        analysis = next(
            (
                layer["content"]
                for layer in reveal["layers"]
                if layer["id"] == "analysis"
            ),
            {},
        )
        reason_key = str(analysis.get("reasonKey") or "video.analysis.expert_reason")
        reason = bundle.value["localizations"].get(reason_key)
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("render reveal reason has no governed localization")
        return reason

    @classmethod
    def _caption_text(cls, bundle: RenderBundleV1) -> tuple[str, str]:
        """Caption copy and its stage; reasons appear only when explicitly enabled."""
        presentation = bundle.value["presentation"].get("presentation") or {}
        if presentation.get("show_justification_text") is False:
            return "", "reveal"
        return cls._reveal_reason(bundle), "reveal"

    def _command(self, script: Path) -> list[str]:
        if os.name != "nt":
            return ["node", str(script)]
        wrapper = Path(os.environ.get("BALOOT_HIDDEN_NODE_WRAPPER", ""))
        if not wrapper.is_file():
            return ["node", str(script)]
        relative_script = script.relative_to(self.renderer_root).as_posix()
        encoded = base64.b64encode(f"node {relative_script}".encode("utf-8")).decode(
            "ascii"
        )
        return [
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(wrapper),
            "-EncodedCommand",
            encoded,
            "-WorkingDirectory",
            str(self.renderer_root),
        ]

    def _run_renderer(
        self,
        *,
        script: Path,
        environment: Mapping[str, str],
        cancel_event: Event | None,
    ) -> None:
        process_environment = {**os.environ, "BALOOTX_PYTHON": sys.executable, **environment}
        if self.use_worker_renderer:
            process_environment["BALOOTX_DISABLE_BUNDLE_CACHE"] = "1"
        render_temp = os.environ.get("BALOOT_RENDER_TEMP", "").strip()
        temp_root = None
        if render_temp:
            temp_root = Path(render_temp).resolve()
            temp_root.mkdir(parents=True, exist_ok=True)
            process_environment["TEMP"] = str(temp_root)
            process_environment["TMP"] = str(temp_root)
        # Polling undrained PIPEs can deadlock a verbose child before it exits.
        # Spool to disk so cancellation stays responsive without retaining all
        # renderer output in memory.
        with (
            tempfile.TemporaryFile(dir=temp_root) as stdout,
            tempfile.TemporaryFile(dir=temp_root) as stderr,
        ):
            process = subprocess.Popen(
                self._command(script),
                cwd=self.renderer_root,
                env=process_environment,
                stdout=stdout,
                stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                start_new_session=os.name != "nt",
            )
            try:
                deadline = time.monotonic() + self.descriptor.timeout_seconds
                while process.poll() is None:
                    if cancel_event is not None and cancel_event.is_set():
                        raise RuntimeError("Remotion render cancelled")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Remotion render exceeded its governed timeout")
                    time.sleep(0.1)
                if process.returncode != 0:
                    detail = _output_tail(stderr) or _output_tail(stdout) or "Remotion render failed"
                    raise RuntimeError(detail)
            finally:
                if process.poll() is None:
                    _terminate_process_tree(process)

    @staticmethod
    def _contact_sheet(stills: list[Path], destination: Path) -> None:
        samples = []
        for path in stills[:6]:
            with Image.open(path) as source:
                samples.append(
                    ImageOps.fit(
                        source.convert("RGB"), (270, 480), Image.Resampling.LANCZOS
                    )
                )
        if not samples:
            raise RuntimeError("Remotion renderer produced no visual checkpoints")
        sheet = Image.new("RGB", (810, 960), "#041411")
        for index, sample in enumerate(samples):
            sheet.paste(sample, ((index % 3) * 270, (index // 3) * 480))
        sheet.save(
            destination, format="JPEG", quality=94, subsampling=0, optimize=False
        )

    @staticmethod
    def _qr_expectation(bundle: RenderBundleV1) -> tuple[str, str]:
        presentation = bundle.value["presentation"].get("presentation")
        if (
            not isinstance(presentation, Mapping)
            or presentation.get("show_enrollment_qr") is not True
        ):
            return "forbidden", "profile_disabled"
        source = bundle.value.get("source")
        if source is not None:
            exact_publishable = (
                source.get("auditOnly") is False
                and source.get("publicationEligible") is True
                and source.get("renderPurpose") == "content_candidate"
                and source.get("qualificationStatus") == "qualified"
                and source.get("qualificationFailures") == []
                and source.get("publicationBlockers") == []
            )
            if not exact_publishable:
                return "forbidden", "research_audit"
            if source.get("qrActionable") is not True:
                return "forbidden", "non_actionable_source"
        if (
            presentation.get("show_identity_pill") is not True
            or bundle.value["presentation"]["majlis"]["id_box"] != "qr"
        ):
            return "forbidden", "profile_hides_qr"
        return "required", "profile_enabled" if source is None else "actionable_exact_state"

    def execute(self, request: Any, **kwargs: Any) -> RenderArtifacts:
        export_preset = str(kwargs.get("export_preset", "hooked_flagship"))
        if export_preset not in {"hooked_flagship", "decision_clip"}:
            raise ValueError(f"unsupported Remotion export preset: {export_preset}")
        if isinstance(request, RenderBundleV1):
            bundle = request
            if bundle.value["exportPreset"] != export_preset or not bundle.verify(raise_voice_errors=True):
                raise ValueError(
                    "render bundle preset or integrity verification failed"
                )
        elif isinstance(request, EpisodeSpec):
            bundle = self.build_render_request(
                request,
                export_preset=export_preset,
                video_profile=kwargs.get("video_profile"),
            )
        else:
            raise TypeError("Remotion renderer requires EpisodeSpec or RenderBundleV1")

        output_dir = Path(kwargs.get("output_dir", "outputs")).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        basename = str(kwargs.get("basename") or bundle.value["identity"]["gameSerial"])
        if not basename or any(character in basename for character in '\\/:*?"<>|'):
            raise ValueError("render basename is invalid")
        script_name = (
            "render-flagship.mjs"
            if export_preset == "hooked_flagship"
            else "render-decision-clip.mjs"
        )
        playback_evidence = bool(kwargs.get("playback_evidence", False))
        qualify_layout = (
            export_preset == "decision_clip"
            and bundle.value["presentation"].get("majlis", {}).get("research_notice") is True
        )
        if qualify_layout:
            if "BALOOTX_LAYOUT_TABLE" in os.environ or "BALOOTX_LAYOUT_TABLE_SHA256" in os.environ:
                raise ValueError("layout table overrides require NEVER_POST measurement mode")
            if _sha256(self.renderer_root / APPROVED_LAYOUT_TABLE) != APPROVED_LAYOUT_TABLE_SHA256:
                raise ValueError("approved layout table SHA-256 mismatch")
        layout_files: dict[str, Path] = {}
        layout_geometry = None

        with tempfile.TemporaryDirectory(
            prefix=".baloot-remotion-", dir=output_dir
        ) as temporary:
            scratch = Path(temporary)
            work = scratch / "rendered"
            qr_validation_path = scratch / "qr-validation.json"
            bundle_path = scratch / "render-bundle.json"
            _json_write(bundle_path, bundle.to_dict())
            self._run_renderer(
                script=self.renderer_root / "scripts" / script_name,
                environment={
                    "BALOOTX_RENDER_BUNDLE": str(bundle_path),
                    "BALOOTX_RENDER_OUTPUT": str(work),
                    "BALOOTX_RENDER_BASENAME": "render",
                    "BALOOTX_ASSETS_ROOT": str(self.repository_root / "assets"),
                    "BALOOTX_BATCH_RENDER": "1",
                    "BALOOTX_PLAYBACK_EVIDENCE": "1" if playback_evidence else "0",
                },
                cancel_event=kwargs.get("cancel_event"),
            )
            layout_evidence = _qualified_layout_evidence(
                work, bundle_path=bundle_path, bundle_sha256=bundle.bundle_sha256,
            ) if qualify_layout else None
            audio_finishing = finish_remotion_audio(
                work / "render.mp4",
                bundle,
                repository_root=self.repository_root,
            )

            video = output_dir / f"{basename}.mp4"
            poster = output_dir / f"{basename}.jpg"
            captions = output_dir / f"{basename}.srt"
            captions_vtt = output_dir / f"{basename}.vtt"
            transcript = output_dir / f"{basename}-transcript.txt"
            contact = output_dir / f"{basename}-contact.jpg"
            episode_json = output_dir / f"{basename}.json"
            timeline_path = output_dir / f"{basename}-timeline.json"
            manifest = output_dir / f"{basename}-manifest.json"
            audio_evidence_path = output_dir / f"{basename}-audio.json"

            shutil.copyfile(work / "render.mp4", video)
            media_probe = probe_media(video, ffprobe=self._ffprobe())
            expected_frames = int(bundle.value["timeline"]["totalFrames"])
            render_profile = bundle.value["render"]
            validate_media_probe(
                media_probe,
                width=int(render_profile["width"]),
                height=int(render_profile["height"]),
                fps=int(render_profile["fps"]),
                minimum_frames=expected_frames,
                expected_frames=expected_frames,
                audio_rate=int(render_profile["audio_rate"]),
            )
            if layout_evidence is not None:
                for row in layout_evidence["items"]:
                    phase = row["phase"]
                    destination = output_dir / f"{basename}-layout-{phase}.png"
                    shutil.copyfile(work / row["source"], destination)
                    row["source"] = destination.name
                    layout_files[f"layout:{phase}"] = destination
                table_destination = output_dir / f"{basename}-layout-table.json"
                shutil.copyfile(work / "layout-table.json", table_destination)
                evidence_destination = output_dir / f"{basename}-layout.json"
                _json_write(evidence_destination, layout_evidence)
                layout_files.update(layoutTable=table_destination, layoutGeometry=evidence_destination)
                layout_geometry = {
                    "status": "passed",
                    "mode": "qualification",
                    "phases": {row["phase"]: row["validation"]["status"] for row in layout_evidence["items"]},
                    "bundleSha256": bundle.bundle_sha256,
                    "tableSha256": layout_evidence["binding"]["tableSha256"],
                    "rendererVideoSha256": layout_evidence["videoSha256"],
                    "videoSha256": _sha256(video),
                    "evidenceSha256": _sha256(evidence_destination),
                }
            playback_rows: list[dict[str, Any]] = []
            playback_files: dict[str, Path] = {}
            if playback_evidence:
                raw_playback = json.loads(
                    (work / "playback-evidence.json").read_text(encoding="utf-8")
                )
                playback_rows = self._playback_state_evidence(
                    bundle, raw_playback, require_rendered_dom=True,
                    clip_evidence=export_preset == "decision_clip",
                )
                for row in playback_rows:
                    count = int(row["completedPlayCount"])
                    source = work / str(row["source"])
                    checkpoint_id = str(row.get("checkpointPhase") or f"{count:02d}")
                    destination = output_dir / f"{basename}-playback-{checkpoint_id}.png"
                    shutil.copyfile(source, destination)
                    comparison = compare_source_frame(
                        video,
                        source,
                        frame=int(row["frame"]),
                        fps=int(render_profile["fps"]),
                        ffmpeg=self._ffmpeg(),
                    )
                    row["sourceSha256"] = _sha256(destination)
                    row["encodedFrameComparison"] = comparison
                    playback_files[f"playback:{checkpoint_id}"] = destination
            shutil.copyfile(work / "render.ar.vtt", captions_vtt)
            stills = sorted(work.glob("*.png"))
            poster_source = next(
                (
                    path
                    for path in stills
                    if "critical-pause" in path.name or "decision" in path.name
                ),
                stills[0],
            )
            with Image.open(poster_source) as source:
                source.convert("RGB").save(
                    poster, format="JPEG", quality=95, subsampling=0, optimize=False
                )
            self._contact_sheet(stills, contact)
            qr_policy, qr_reason = self._qr_expectation(bundle)
            self._run_renderer(
                script=self.renderer_root / "scripts" / "verify-qr.mjs",
                environment={
                    "BALOOTX_QR_IMAGE": str(poster),
                    "BALOOTX_QR_EXPECTED": (
                        str(bundle.value["identity"]["enrollmentUri"])
                        if qr_policy == "required"
                        else ""
                    ),
                    "BALOOTX_QR_RESULT": str(qr_validation_path),
                    "BALOOTX_QR_POLICY": qr_policy,
                    "BALOOTX_QR_REASON": qr_reason,
                },
                cancel_event=kwargs.get("cancel_event"),
            )
            qr_validation = json.loads(qr_validation_path.read_text(encoding="utf-8"))
            if (
                qr_validation.get("passed") is not True
                or qr_validation.get("policy") != qr_policy
                or qr_validation.get("reason") != qr_reason
                or (
                    qr_policy == "required"
                    and qr_validation.get("decodedValue")
                    != bundle.value["identity"]["enrollmentUri"]
                )
                or (
                    qr_policy == "forbidden"
                    and qr_validation.get("decodedValue") is not None
                )
            ):
                raise RuntimeError("rendered poster QR policy verification failed")

        _json_write(audio_evidence_path, audio_finishing)
        reason, caption_stage = self._caption_text(bundle)
        reveal = next(
            item
            for item in bundle.value["timeline"]["segments"]
            if item["stage"] == caption_stage
        )
        fps = int(bundle.value["render"]["fps"])
        caption_start = int(reveal["startFrame"])
        caption_end = min(
            caption_start + 240, caption_start + int(reveal["durationFrames"])
        )
        atomic_write_text(
            captions,
            f"1\n{_timestamp(caption_start, fps, srt=True)} --> {_timestamp(caption_end, fps, srt=True)}\n{reason}\n",
        )
        atomic_write_text(transcript, reason + "\n")
        _json_write(episode_json, bundle.to_dict())
        _json_write(timeline_path, bundle.value["timeline"])
        semantic_files: list[Path] = []
        for scene in bundle.value["scenes"]:
            path = output_dir / f"{basename}-scene-{scene['stage']}.json"
            _json_write(path, scene)
            semantic_files.append(path)

        files = {
            "video": video,
            "poster": poster,
            "bundle": episode_json,
            "captionsSrt": captions,
            "captionsVtt": captions_vtt,
            "transcript": transcript,
            "contactSheet": contact,
            "timeline": timeline_path,
            "audioEvidence": audio_evidence_path,
            **{
                f"scene:{path.stem.rsplit('-', 1)[-1]}": path for path in semantic_files
            },
            **playback_files,
            **layout_files,
        }
        manifest_value = {
            "schemaVersion": 1,
            "renderer": self.descriptor.to_dict(),
            "exportPreset": export_preset,
            "bundleSha256": bundle.bundle_sha256,
            "timelineSha256": bundle.value["timeline"]["timelineSha256"],
            "observerSafe": True,
            "deterministic": True,
            "identity": bundle.value["identity"],
            "decisionIndex": bundle.value["decisionIndex"],
            "source": bundle.value.get("source"),
            "sourceSha256": (
                _canonical_sha256(bundle.value["source"])
                if bundle.value.get("source") is not None
                else None
            ),
            "decisionReference": (
                bundle.value["source"].get("decisionReference")
                if bundle.value.get("source") is not None
                else None
            ),
            "publicProjectionHash": (
                bundle.value["source"].get("publicProjectionHash")
                if bundle.value.get("source") is not None
                else None
            ),
            "qrValidation": qr_validation,
            "mediaProbe": media_probe,
            "audioFinishing": audio_finishing,
            "playbackEvidence": playback_rows,
            **({"layoutGeometry": layout_geometry} if layout_geometry is not None else {}),
            "files": {
                key: {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for key, path in sorted(files.items())
            },
        }
        manifest_value["manifestSha256"] = hashlib.sha256(
            json.dumps(
                manifest_value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        _json_write(manifest, manifest_value)
        return RenderArtifacts(
            video=video,
            poster=poster,
            episode_json=episode_json,
            captions=captions,
            contact_sheet=contact,
            manifest=manifest,
            timeline=timeline_path,
            semantic_files=tuple(sorted(semantic_files)),
            captions_vtt=captions_vtt,
            transcript=transcript,
        )
