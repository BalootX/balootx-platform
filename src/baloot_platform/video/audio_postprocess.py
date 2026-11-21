"""Deterministic observer-safe audio finishing for every Remotion export."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
import wave
from array import array
from pathlib import Path
from typing import Any, Mapping

from baloot_platform.audit.models import content_hash

from ._voice_guard_foley_pin import VOICE_GUARD_FOLEY_SOURCE
from .media_validation import (
    MAXIMUM_LOUDNESS_DEVIATION_LUFS,
    MAXIMUM_TRUE_PEAK_DBTP,
    media_tool,
    probe_loudness,
    probe_media,
    run_media_command,
    validate_loudness,
    validate_media_probe,
)
from .render_bundle import RenderBundleV1, _RETIRED_VOICE_DEFAULTS

AUDIO_FINISHER_ID = "balootx_original_foley_v10"
_MAX_POST_ENCODE_CORRECTIONS = 3
_POST_ENCODE_MARGIN_DB = 0.10
_SEAT_PAN = {"north": 0.0, "west": -0.55, "south": 0.0, "east": 0.55}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _public_lifecycle(bundle: RenderBundleV1) -> Mapping[str, Any] | None:
    context = next(
        (scene for scene in bundle.value["scenes"] if scene.get("stage") == "continuation"),
        None,
    )
    if context is None:
        raise ValueError("audio finishing requires a post-reveal continuation scene")
    layer = next(
        (
            item
            for item in context.get("layers", ())
            if item.get("id") == "public-lifecycle"
        ),
        None,
    )
    if layer is None:
        source = bundle.value.get("source")
        if (
            isinstance(source, Mapping)
            and source.get("kind") == "research_scenario_result"
            and str(source.get("exactStateId") or "")
        ):
            # V4 decision clips intentionally carry only a compact observer-safe
            # exact-state projection. They have no full replay lifecycle to score;
            # timer/reveal/outcome cues below remain governed and deterministic.
            return None
        raise ValueError("audio finishing requires the public lifecycle layer")
    if not isinstance(layer.get("content"), Mapping):
        raise ValueError("audio finishing public lifecycle layer is invalid")
    return layer["content"]


def build_audio_cue_plan(bundle: RenderBundleV1) -> dict[str, Any]:
    """Derive sound events only from the verified public bundle timeline."""
    if not bundle.verify():
        raise ValueError("audio finishing requires a verified render bundle")
    value = bundle.value
    audio = dict(value.get("presentation", {}).get("audio") or {})
    if not audio.get("enabled", False):
        raise ValueError("governed video audio profile is disabled")
    if audio.get("ambience", False):
        raise ValueError("release audio does not permit continuous ambience or music")

    fps = int(bundle.value["render"]["fps"])
    segments = {item["stage"]: item for item in bundle.value["timeline"]["segments"]}
    lifecycle = _public_lifecycle(bundle)
    tricks = list(lifecycle.get("completedTricks") or ()) if lifecycle else []
    plays = [play for trick in tricks for play in list(trick.get("plays") or ())]
    pause = max(
        0,
        min(
            int(lifecycle.get("pauseBeforePlayIndex") or 0) if lifecycle else 0,
            len(plays),
        ),
    )
    context = segments.get("context")
    has_context_playback = context is not None and context["durationFrames"] > 0
    replay_target = max(0, pause - int(has_context_playback))
    events: list[dict[str, Any]] = []
    reveal_scene = next(
        (
            scene
            for scene in bundle.value["scenes"]
            if scene.get("stage") == "reveal"
        ),
        {},
    )
    analysis = next(
        (
            item.get("content") or {}
            for item in reveal_scene.get("layers", ())
            if item.get("id") == "analysis"
        ),
        {},
    )
    decision_kind = str(analysis.get("decisionKind") or "play")
    render_play_trace = not (
        bundle.value.get("exportPreset") == "decision_clip"
        and decision_kind in {"bid", "escalation", "project"}
    )

    def add(kind: str, frame: int, *, seat: str | None = None) -> None:
        bounded = max(
            0, min(int(frame), int(bundle.value["timeline"]["totalFrames"]) - 1)
        )
        events.append({"kind": kind, "frame": bounded, "seat": seat})

    def schedule_plays(stage: str, start_index: int, end_index: int) -> None:
        segment = segments.get(stage)
        count = max(0, end_index - start_index)
        if segment is None or segment["durationFrames"] <= 0 or count == 0:
            return
        interval = float(segment["durationFrames"]) / count
        for offset, play_index in enumerate(range(start_index, end_index)):
            frame = int(segment["startFrame"] + offset * interval)
            if audio.get("card_clicks", False) or audio.get("foley", False):
                add("card", frame, seat=str(plays[play_index].get("seat") or ""))
            if audio.get("table_foley", False) and (play_index + 1) % 4 == 0:
                add(
                    "collect",
                    frame + min(int(0.38 * fps), max(1, int(interval / 2))),
                    seat=str(tricks[play_index // 4].get("winner") or ""),
                )

    if render_play_trace:
        schedule_plays("replay", 0, replay_target)
        schedule_plays("context", replay_target, pause)
        schedule_plays("continuation", pause, len(plays))

    question = segments.get("question")
    if question and audio.get("countdown", False):
        end = int(question["startFrame"]) + int(question["durationFrames"])
        for seconds in (3, 2, 1):
            frame = end - seconds * fps
            if frame >= int(question["startFrame"]):
                add("timer", frame)
    reveal = segments.get("reveal")
    if reveal and audio.get("reveal_chime", False):
        add(
            "reveal",
            int(reveal["startFrame"]) + min(4, int(reveal["durationFrames"]) - 1),
        )
    outcome = segments.get("outcome")
    if outcome and audio.get("foley", False):
        add(
            "outcome",
            int(outcome["startFrame"]) + min(6, int(outcome["durationFrames"]) - 1),
        )

    events.sort(key=lambda item: (item["frame"], item["kind"], item.get("seat") or ""))
    normalization_true_peak = float(audio["normalization_true_peak_dbtp"])
    if not -6.0 <= normalization_true_peak <= -1.0:
        raise ValueError("audio normalization true peak must be between -6 and -1 dBTP")
    plan = {
        "schemaVersion": 1,
        "finisher": AUDIO_FINISHER_ID,
        "bundleSha256": bundle.bundle_sha256,
        "fps": fps,
        "sampleRate": int(bundle.value["render"].get("audio_rate", 48_000)),
        "totalFrames": int(bundle.value["timeline"]["totalFrames"]),
        "targetLufs": float(audio["target_lufs"]),
        "normalizationTruePeakDbtp": normalization_true_peak,
        "music": False,
        "profileSha256": content_hash(audio),
        "events": events,
    }
    previous_source = VOICE_GUARD_FOLEY_SOURCE.get(value.get("renderer", {}).get("sourceSha256"))
    if previous_source is not None and not value["audioCues"]:
        # Recreate only the legacy hash input. The render bundle and plan keep their current seals.
        value.pop("bundleSha256")
        retired_flags = value.pop("retiredVoiceFlags", None)
        if retired_flags is not None:
            audio.update(_RETIRED_VOICE_DEFAULTS, **retired_flags)
            value["presentation"]["audio"] = audio
        value["renderer"]["sourceSha256"] = previous_source
        previous_plan_sha = content_hash({
            **plan, "bundleSha256": content_hash(value), "profileSha256": content_hash(audio),
        })
        plan["noiseSeed"] = int(previous_plan_sha[:8], 16)
    return {**plan, "planSha256": content_hash(plan)}


def _add_tone(
    left: array,
    right: array,
    *,
    start: int,
    rate: int,
    duration: float,
    frequencies: tuple[float, ...],
    gain: float,
    pan: float,
    noise: bool = False,
    seed: int = 1,
) -> None:
    count = min(len(left) - start, max(0, int(duration * rate)))
    state = seed & 0x7FFFFFFF
    left_gain = math.sqrt((1.0 - pan) / 2.0)
    right_gain = math.sqrt((1.0 + pan) / 2.0)
    for index in range(count):
        progress = index / max(1, count)
        envelope = (1.0 - progress) ** 2
        signal = sum(
            math.sin(2.0 * math.pi * frequency * index / rate)
            for frequency in frequencies
        ) / max(1, len(frequencies))
        if noise:
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            signal = 0.55 * signal + 0.45 * ((state / 0x3FFFFFFF) - 1.0)
        sample = gain * envelope * signal
        position = start + index
        left[position] = max(
            -32767, min(32767, left[position] + int(32767 * sample * left_gain))
        )
        right[position] = max(
            -32767, min(32767, right[position] + int(32767 * sample * right_gain))
        )


def generate_original_foley(
    plan: Mapping[str, Any], destination: Path
) -> dict[str, Any]:
    rate = int(plan["sampleRate"])
    sample_count = round(int(plan["totalFrames"]) * rate / int(plan["fps"]))
    left = array("h", [0]) * sample_count
    right = array("h", [0]) * sample_count
    noise_seed = plan.get("noiseSeed")
    if noise_seed is None:
        noise_seed = int(plan["planSha256"][:8], 16)
    designs = {
        "card": (0.055, (1180.0, 1760.0), 0.22, True),
        "collect": (0.18, (190.0, 420.0), 0.20, True),
        "timer": (0.075, (720.0,), 0.16, False),
        "reveal": (0.34, (880.0, 1320.0), 0.18, False),
        "outcome": (0.52, (392.0, 523.25, 659.25), 0.16, False),
    }
    for ordinal, event in enumerate(plan["events"]):
        duration, frequencies, gain, noise = designs[str(event["kind"])]
        pan = _SEAT_PAN.get(str(event.get("seat") or ""), 0.0)
        _add_tone(
            left,
            right,
            start=round(int(event["frame"]) * rate / int(plan["fps"])),
            rate=rate,
            duration=duration,
            frequencies=frequencies,
            gain=gain,
            pan=pan,
            noise=noise,
            seed=noise_seed ^ ordinal,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    interleaved = array("h")
    for lvalue, rvalue in zip(left, right):
        interleaved.extend((lvalue, rvalue))
    with wave.open(str(destination), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(interleaved.tobytes())
    return {
        "path": destination.name,
        "bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
        "eventCount": len(plan["events"]),
        "eventCounts": {
            kind: sum(1 for event in plan["events"] if event["kind"] == kind)
            for kind in sorted({event["kind"] for event in plan["events"]})
        },
    }


def _video_stream_sha256(video: Path, *, ffmpeg: Path, directory: Path) -> str:
    elementary = directory / f"{video.stem}.h264"
    run_media_command(
        [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-map",
            "0:v:0",
            "-c",
            "copy",
            "-f",
            "h264",
            "-y",
            str(elementary),
        ],
        check=True,
        capture_output=True,
        timeout=180,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    return _sha256(elementary)


def _measured_loudnorm_filter(
    plan: Mapping[str, Any], measurement: Mapping[str, Any]
) -> str:
    values = {
        "measured_I": float(measurement["integratedLufs"]),
        "measured_TP": float(measurement["truePeakDbtp"]),
        "measured_LRA": float(measurement["loudnessRangeLu"]),
        "measured_thresh": float(measurement["thresholdLufs"]),
        "offset": float(measurement["targetOffset"]),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("audio normalization measurement must be finite")
    measured = ":".join(f"{key}={value}" for key, value in values.items())
    return (
        f"loudnorm=I={plan['targetLufs']}:TP={plan['normalizationTruePeakDbtp']}:"
        f"LRA=11:{measured}:linear=true:print_format=summary"
    )


def _dynamic_loudnorm_filter(plan: Mapping[str, Any]) -> str:
    return (
        f"loudnorm=I={plan['targetLufs']}:TP={plan['normalizationTruePeakDbtp']}:"
        "LRA=11:linear=false:print_format=summary"
    )


def _peak_controlled_mix(
    source: Path, destination: Path, *, plan: Mapping[str, Any], measurement: Mapping[str, Any]
) -> dict[str, Any] | None:
    required_gain = max(0.0, float(plan["targetLufs"]) - float(measurement["integratedLufs"]))
    allowed_peak = float(plan["normalizationTruePeakDbtp"]) - required_gain - _POST_ENCODE_MARGIN_DB
    if float(measurement["truePeakDbtp"]) <= allowed_peak:
        return None
    limit = 10 ** (allowed_peak / 20)
    with wave.open(str(source), "rb") as stream:
        if stream.getnchannels() != 2 or stream.getsampwidth() != 2:
            raise ValueError("audio peak control requires stereo PCM16")
        channels, rate, frames = stream.getnchannels(), stream.getframerate(), stream.getnframes()
        samples = array("h", stream.readframes(frames))
    for index, sample in enumerate(samples):
        amplitude = abs(sample) / 32768
        samples[index] = (-1 if sample < 0 else 1) * round(
            32767 * limit * math.tanh(amplitude / limit)
        )
    with wave.open(str(destination), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(samples.tobytes())
    return {"path": destination.name, "sha256": _sha256(destination), "allowedPeakDbtp": allowed_peak}


def _bounded_linear_gain_db(
    plan: Mapping[str, Any], measurement: Mapping[str, Any]
) -> float | None:
    target = float(plan["targetLufs"])
    integrated = float(measurement["integratedLufs"])
    true_peak = float(measurement["truePeakDbtp"])
    for margin in (_POST_ENCODE_MARGIN_DB, _POST_ENCODE_MARGIN_DB / 2):
        minimum = target - MAXIMUM_LOUDNESS_DEVIATION_LUFS + margin - integrated
        maximum = min(
            target + MAXIMUM_LOUDNESS_DEVIATION_LUFS - margin - integrated,
            MAXIMUM_TRUE_PEAK_DBTP - margin - true_peak,
        )
        if minimum <= maximum:
            return round(min(max(0.0, minimum), maximum), 2)
    return None


def _replace_filtered_audio(
    video: Path,
    output: Path,
    *,
    filter_value: str,
    plan: Mapping[str, Any],
    ffmpeg: Path,
) -> None:
    run_media_command(
        [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            str(plan["sampleRate"]),
            "-ac",
            "2",
            "-af",
            filter_value,
            "-map_metadata",
            "0",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-colorspace",
            "bt709",
            "-movflags",
            "+faststart",
            "-y",
            str(output),
        ],
        check=True,
        capture_output=True,
        timeout=300,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    os.replace(output, video)


def finish_remotion_audio(
    video: Path,
    bundle: RenderBundleV1,
    *,
    repository_root: Path,
    evidence_path: Path | None = None,
) -> dict[str, Any]:
    """Add governed original cues and normalize audio without re-encoding video."""
    video = video.resolve()
    root = repository_root.resolve()
    ffmpeg = media_tool(root, "ffmpeg")
    ffprobe = media_tool(root, "ffprobe")
    plan = build_audio_cue_plan(bundle)
    render = bundle.value["render"]
    expected_frames = int(bundle.value["timeline"]["totalFrames"])
    source_media = {"bytes": video.stat().st_size, "sha256": _sha256(video)}
    with tempfile.TemporaryDirectory(
        prefix="baloot-audio-finish-", dir=video.parent
    ) as temporary:
        work = Path(temporary)
        foley = work / "original-foley.wav"
        mixed = work / "mixed.wav"
        finished = work / "finished.mp4"
        foley_evidence = generate_original_foley(plan, foley)
        source_probe = probe_media(video, ffprobe=ffprobe)
        source_video_sha = _video_stream_sha256(video, ffmpeg=ffmpeg, directory=work)

        if source_probe.get("audio") is None:
            run_media_command(
                [
                    str(ffmpeg),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(foley),
                    "-ar",
                    str(plan["sampleRate"]),
                    "-ac",
                    "2",
                    "-c:a",
                    "pcm_s16le",
                    "-y",
                    str(mixed),
                ],
                check=True,
                capture_output=True,
                timeout=180,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        else:
            run_media_command(
                [
                    str(ffmpeg),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(video),
                    "-i",
                    str(foley),
                    "-filter_complex",
                    "[0:a:0][1:a:0]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[a]",
                    "-map",
                    "[a]",
                    "-ar",
                    str(plan["sampleRate"]),
                    "-ac",
                    "2",
                    "-c:a",
                    "pcm_s16le",
                    "-y",
                    str(mixed),
                ],
                check=True,
                capture_output=True,
                timeout=240,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )

        raw_mix_measurement = probe_loudness(
            mixed,
            ffmpeg=ffmpeg,
            target_lufs=float(plan["targetLufs"]),
            true_peak_dbtp=float(plan["normalizationTruePeakDbtp"]),
        )
        controlled = work / "peak-controlled.wav"
        peak_control = _peak_controlled_mix(
            mixed, controlled, plan=plan, measurement=raw_mix_measurement
        )
        if peak_control is not None:
            mixed = controlled
        mixed_audio = {"bytes": mixed.stat().st_size, "sha256": _sha256(mixed)}
        normalization_measurement = (
            probe_loudness(
                mixed,
                ffmpeg=ffmpeg,
                target_lufs=float(plan["targetLufs"]),
                true_peak_dbtp=float(plan["normalizationTruePeakDbtp"]),
            ) if peak_control is not None else raw_mix_measurement
        )
        loudnorm = _measured_loudnorm_filter(plan, normalization_measurement)
        run_media_command(
            [
                str(ffmpeg),
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video),
                "-i",
                str(mixed),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-ar",
                str(plan["sampleRate"]),
                "-ac",
                "2",
                "-af",
                loudnorm,
                "-shortest",
                "-map_metadata",
                "0",
                "-color_primaries",
                "bt709",
                "-color_trc",
                "bt709",
                "-colorspace",
                "bt709",
                "-movflags",
                "+faststart",
                "-y",
                str(finished),
            ],
            check=True,
            capture_output=True,
            timeout=300,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        post_encode_loudness_gain_db = 0.0
        post_encode_true_peak_trim_db = 0.0
        post_encode_corrections: list[dict[str, Any]] = []
        encoded_loudness = probe_loudness(
            finished,
            ffmpeg=ffmpeg,
            target_lufs=float(plan["targetLufs"]),
            true_peak_dbtp=float(plan["normalizationTruePeakDbtp"]),
        )
        for attempt in range(1, _MAX_POST_ENCODE_CORRECTIONS + 1):
            loudness_deviation = float(plan["targetLufs"]) - float(
                encoded_loudness["integratedLufs"]
            )
            if (
                abs(loudness_deviation) <= MAXIMUM_LOUDNESS_DEVIATION_LUFS
                and float(encoded_loudness["truePeakDbtp"])
                <= MAXIMUM_TRUE_PEAK_DBTP
            ):
                break
            linear_gain_db = _bounded_linear_gain_db(plan, encoded_loudness)
            post_encode_loudness_gain_db = round(
                loudness_deviation if linear_gain_db is None else linear_gain_db, 2
            )
            post_encode_true_peak_trim_db = round(
                min(
                    0.0,
                    MAXIMUM_TRUE_PEAK_DBTP
                    - float(encoded_loudness["truePeakDbtp"])
                    - 0.02,
                ),
                2,
            )
            corrected = work / f"post-encode-normalized-{attempt}.mp4"
            correction_plan = {
                **plan,
                "normalizationTruePeakDbtp": round(
                    float(plan["normalizationTruePeakDbtp"])
                    + post_encode_true_peak_trim_db,
                    2,
                ),
            }
            correction_kind = (
                "dynamic_loudnorm" if linear_gain_db is None else "bounded_linear_gain"
            )
            _replace_filtered_audio(
                finished,
                corrected,
                filter_value=(
                    _dynamic_loudnorm_filter(correction_plan)
                    if linear_gain_db is None
                    else f"volume={linear_gain_db}dB"
                ),
                plan=plan,
                ffmpeg=ffmpeg,
            )
            post_encode_corrections.append(
                {
                    "attempt": attempt,
                    "kind": correction_kind,
                    "input": encoded_loudness,
                    "loudnessGainDb": post_encode_loudness_gain_db,
                    "truePeakTrimDb": post_encode_true_peak_trim_db,
                    "normalizationTruePeakDbtp": correction_plan[
                        "normalizationTruePeakDbtp"
                    ],
                }
            )
            encoded_loudness = probe_loudness(
                finished,
                ffmpeg=ffmpeg,
                target_lufs=float(plan["targetLufs"]),
                true_peak_dbtp=float(plan["normalizationTruePeakDbtp"]),
            )
        output_video_sha = _video_stream_sha256(finished, ffmpeg=ffmpeg, directory=work)
        if output_video_sha != source_video_sha:
            raise RuntimeError("audio finishing changed the encoded video stream")
        final_probe = probe_media(finished, ffprobe=ffprobe)
        validate_media_probe(
            final_probe,
            width=int(render["width"]),
            height=int(render["height"]),
            fps=int(render["fps"]),
            minimum_frames=expected_frames,
            expected_frames=expected_frames,
            audio_rate=int(render.get("audio_rate", 48_000)),
        )
        final_loudness = encoded_loudness
        validate_loudness(final_loudness)
        evidence = {
            "schemaVersion": 1,
            "status": "passed",
            "finisher": AUDIO_FINISHER_ID,
            "bundleSha256": bundle.bundle_sha256,
            "audioProfileSha256": plan["profileSha256"],
            "cuePlanSha256": plan["planSha256"],
            "music": False,
            "normalizationTruePeakDbtp": plan["normalizationTruePeakDbtp"],
            "normalizationMeasurement": normalization_measurement,
            "rawMixMeasurement": raw_mix_measurement,
            "peakControl": peak_control,
            "postEncodeLoudnessGainDb": post_encode_loudness_gain_db,
            "postEncodeTruePeakTrimDb": post_encode_true_peak_trim_db,
            "postEncodeCorrections": post_encode_corrections,
            "sourceMedia": source_media,
            "foley": foley_evidence,
            "mixedAudio": mixed_audio,
            "sourceVideoStreamSha256": source_video_sha,
            "outputVideoStreamSha256": output_video_sha,
            "videoStreamPreserved": True,
            "loudness": final_loudness,
            "mediaProbe": final_probe,
            "output": {
                "bytes": finished.stat().st_size,
                "sha256": _sha256(finished),
            },
        }
        os.replace(finished, video)
    evidence["evidenceSha256"] = content_hash(evidence)
    if evidence_path is not None:
        evidence_path.write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return evidence
