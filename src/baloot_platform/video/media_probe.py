"""Shared stdlib-only media probe and validation boundary."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping


MEASUREMENT_MP4_MARKER = "balootx-layout-measurement-never-post"


def run_media_command(
    command: list[str],
    **kwargs: Any,
) -> subprocess.CompletedProcess[Any]:
    """Run FFmpeg-family tools through a Windows long-path-safe boundary."""
    arguments = list(command)
    if not arguments:
        raise ValueError("media command cannot be empty")
    executable = os.fspath(arguments[0])
    if os.environ.get("CI") == "true":
        tool = Path(executable).stem.lower()
        if tool == "ffmpeg" and "-i" in arguments:
            # FFmpeg decoder options are per input; encoder options are per output.
            inputs = []
            for argument in arguments[1:-1]:
                if argument == "-i":
                    inputs.extend(["-threads", "4"])
                inputs.append(argument)
            arguments = [
                executable, "-filter_threads", "4", "-filter_complex_threads", "4",
                *inputs, "-threads", "4", arguments[-1],
            ]
        elif tool == "ffprobe" and "-count_frames" in arguments:
            arguments[1:1] = ["-threads", "4"]
    if os.name == "nt" and os.path.isabs(executable):
        if executable.startswith("\\\\?\\"):
            extended = executable
        elif executable.startswith("\\\\"):
            extended = f"\\\\?\\UNC\\{executable[2:]}"
        else:
            extended = f"\\\\?\\{executable}"
        arguments[0] = Path(executable).name
        kwargs["executable"] = extended
    return subprocess.run(arguments, **kwargs)

def probe_media(path: Path, *, ffprobe: Path) -> dict[str, Any]:
    process = run_media_command(
        [
            str(ffprobe),
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=index,codec_type,codec_name,width,height,r_frame_rate,pix_fmt,color_space,color_primaries,color_transfer,sample_rate,channels,nb_read_frames:format=duration:format_tags=comment",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    raw = json.loads(process.stdout)
    if raw["format"].get("tags", {}).get("comment") == MEASUREMENT_MP4_MARKER:
        raise ValueError("layout measurement MP4 is permanently NEVER_POST")
    video = next(
        (item for item in raw.get("streams", ()) if item.get("codec_type") == "video"),
        None,
    )
    audio = next(
        (item for item in raw.get("streams", ()) if item.get("codec_type") == "audio"),
        None,
    )
    if video is None:
        raise ValueError("rendered media has no video stream")
    return {
        "durationSeconds": float(raw["format"]["duration"]),
        "video": {
            "codec": video.get("codec_name"),
            "width": int(video.get("width", 0)),
            "height": int(video.get("height", 0)),
            "frameRate": video.get("r_frame_rate"),
            "pixelFormat": video.get("pix_fmt"),
            "colorSpace": video.get("color_space"),
            "colorPrimaries": video.get("color_primaries"),
            "colorTransfer": video.get("color_transfer"),
            "decodedFrames": int(video.get("nb_read_frames") or 0),
        },
        "audio": None
        if audio is None
        else {
            "codec": audio.get("codec_name"),
            "sampleRate": int(audio.get("sample_rate") or 0),
            "channels": int(audio.get("channels") or 0),
        },
    }

def validate_media_probe(
    probe: Mapping[str, Any],
    *,
    width: int,
    height: int,
    fps: int,
    minimum_frames: int,
    audio_rate: int,
    expected_frames: int | None = None,
) -> None:
    video = probe["video"]
    audio = probe["audio"]
    if not (
        video["codec"] == "h264"
        and (video["width"], video["height"]) == (width, height)
        and video["frameRate"] == f"{fps}/1"
        and video["pixelFormat"] == "yuv420p"
        and video["colorSpace"] == "bt709"
        and video["colorPrimaries"] == "bt709"
        and video["colorTransfer"] == "bt709"
        and int(video["decodedFrames"]) >= minimum_frames
        and audio is not None
        and audio["codec"] == "aac"
        and audio["sampleRate"] == audio_rate
        and audio["channels"] == 2
    ):
        raise ValueError(f"governed media profile validation failed: {probe}")
    if expected_frames is not None and int(video["decodedFrames"]) != expected_frames:
        raise ValueError("decoded frame count does not match the governed timeline")
