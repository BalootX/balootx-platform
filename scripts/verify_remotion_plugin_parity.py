#!/usr/bin/env python3
"""Verify the governed Remotion plugin against the approved exports."""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import threading
from pathlib import Path

from PIL import Image, ImageChops


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from baloot_platform.filesystem import atomic_write_text  # noqa: E402
from baloot_platform.video.render_bundle import RenderBundleV1  # noqa: E402

OUTPUT = ROOT / "outputs/remotion-plugin-parity"
EXPORTS = {
    "hooked_flagship": {
        "approved": ROOT / "outputs/professional-video-flagship/balootx-complete-laffah-audit.mp4",
        "plugin": OUTPUT / "flagship-plugin-parity.mp4",
        "manifest": OUTPUT / "flagship-plugin-parity-manifest.json",
        "bundle": ROOT / "outputs/professional-video-prototypes/render-bundle-hooked_flagship.json",
    },
    "decision_clip": {
        "approved": ROOT / "outputs/professional-video-decision/balootx-decision-clip.mp4",
        "plugin": OUTPUT / "decision-plugin-parity.mp4",
        "manifest": OUTPUT / "decision-plugin-parity-manifest.json",
        "bundle": ROOT / "outputs/professional-video-prototypes/render-bundle-decision.json",
    },
}

VISUAL_PARITY_THRESHOLDS = {
    "meanAbsoluteChannelDelta": 0.05,
    "changedChannelFraction": 0.02,
    "maximumFrameMeanAbsoluteChannelDelta": 25.0,
    "maximumFrameChangedChannelFraction": 0.20,
    "materialFrameMeanAbsoluteChannelDelta": 2.0,
    "maximumMateriallyDifferentFrames": 2,
}
FFMPEG_OPERATION_TIMEOUT_SECONDS = 900


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            timeout=30,
            check=False,
        )
    else:
        process.kill()


def _watch_processes(
    processes: list[subprocess.Popen[bytes]], *, operation: str
) -> tuple[threading.Timer, threading.Event]:
    timed_out = threading.Event()

    def expire() -> None:
        timed_out.set()
        for process in processes:
            _terminate_process_tree(process)

    timer = threading.Timer(FFMPEG_OPERATION_TIMEOUT_SECONDS, expire)
    timer.daemon = True
    timer.name = f"balootx-{operation}-timeout"
    timer.start()
    return timer, timed_out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _probe(path: Path) -> dict[str, object]:
    executable = shutil.which("ffprobe")
    if executable is None:
        candidates = sorted(
            (ROOT / "src/baloot_platform/video/remotion/node_modules/@remotion").glob(
                "compositor-*/ffprobe*"
            )
        )
        if not candidates:
            raise ValueError("ffprobe is required to verify Remotion plugin parity")
        executable = str(candidates[0])
    result = subprocess.run(
        [
            executable, "-v", "error", "-show_entries",
            "format=duration:stream=codec_type,codec_name,width,height,pix_fmt,r_frame_rate,"
            "color_space,color_transfer,color_primaries,sample_rate,channels,nb_frames",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return json.loads(result.stdout)


def _ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable is not None:
        return executable
    candidates = sorted(
        (ROOT / "src/baloot_platform/video/remotion/node_modules/@remotion").glob(
            "compositor-*/ffmpeg*"
        )
    )
    if not candidates:
        raise ValueError("ffmpeg is required to verify decoded media parity")
    return str(candidates[0])


def _decoded_audio_sha256(path: Path) -> str:
    process = subprocess.Popen(
        [_ffmpeg(), "-v", "error", "-i", str(path), "-map", "0:a:0", "-vn",
         "-f", "wav", "-c:a", "pcm_s16le", "pipe:1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    timer, timed_out = _watch_processes(
        [process], operation="decoded audio parity"
    )
    assert process.stdout is not None and process.stderr is not None
    try:
        digest = hashlib.sha256()
        for block in iter(lambda: process.stdout.read(1024 * 1024), b""):
            digest.update(block)
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait(timeout=30)
    finally:
        timer.cancel()
        if process.poll() is None:
            _terminate_process_tree(process)
    if timed_out.is_set():
        raise TimeoutError(
            "decoded audio parity exceeded "
            f"{FFMPEG_OPERATION_TIMEOUT_SECONDS}s"
        )
    if return_code:
        raise ValueError(f"decoded audio parity failed: {stderr[-2000:]}")
    return digest.hexdigest()


def _read_png(stream: object) -> bytes | None:
    signature = stream.read(8)
    if not signature:
        return None
    if signature != b"\x89PNG\r\n\x1a\n":
        raise ValueError("decoded video stream did not contain a PNG frame")
    value = bytearray(signature)
    while True:
        header = stream.read(8)
        if len(header) != 8:
            raise ValueError("decoded PNG frame ended before its chunk header")
        length = struct.unpack(">I", header[:4])[0]
        body = stream.read(length + 4)
        if len(body) != length + 4:
            raise ValueError("decoded PNG frame ended before its chunk body")
        value.extend(header)
        value.extend(body)
        if header[4:8] == b"IEND":
            return bytes(value)


def _decoded_visual_parity(approved: Path, plugin: Path) -> tuple[dict[str, object], str, str]:
    def command(path: Path) -> list[str]:
        return [
            _ffmpeg(), "-v", "error", "-i", str(path), "-map", "0:v:0", "-an",
            "-f", "image2pipe", "-c:v", "png", "pipe:1",
        ]
    processes = [
        subprocess.Popen(command(path), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for path in (approved, plugin)
    ]
    timer, timed_out = _watch_processes(
        processes, operation="decoded visual parity"
    )
    for process in processes:
        assert process.stdout is not None and process.stderr is not None

    approved_digest = hashlib.sha256()
    plugin_digest = hashlib.sha256()
    frames = changed_frames = total_delta = changed_channels = total_channels = 0
    maximum_channel_delta = 0
    maximum_frame_mean = 0.0
    maximum_frame_changed_fraction = 0.0
    materially_different_frames: list[int] = []
    frame_channels: int | None = None
    try:
        while True:
            approved_png = _read_png(processes[0].stdout)
            plugin_png = _read_png(processes[1].stdout)
            if approved_png is None and plugin_png is None:
                break
            if approved_png is None or plugin_png is None:
                raise ValueError("approved and plugin videos have different frame counts")
            approved_digest.update(approved_png)
            plugin_digest.update(plugin_png)
            if frame_channels is None:
                with Image.open(io.BytesIO(approved_png)) as first_frame:
                    frame_channels = first_frame.width * first_frame.height * 3
            frame_delta = frame_changed = frame_maximum = 0
            if approved_png != plugin_png:
                with Image.open(io.BytesIO(approved_png)) as approved_image, Image.open(
                    io.BytesIO(plugin_png)
                ) as plugin_image:
                    difference = ImageChops.difference(
                        approved_image.convert("RGB"), plugin_image.convert("RGB")
                    )
                    histogram = difference.histogram()
                    frame_delta = sum(
                        (index % 256) * count for index, count in enumerate(histogram)
                    )
                    frame_changed = sum(
                        count for index, count in enumerate(histogram) if index % 256
                    )
                    frame_maximum = max(
                        maximum for _minimum, maximum in difference.getextrema()
                    )
            frame_mean = frame_delta / frame_channels
            frame_changed_fraction = frame_changed / frame_channels
            frames += 1
            changed_frames += int(frame_changed > 0)
            total_delta += frame_delta
            changed_channels += frame_changed
            total_channels += frame_channels
            maximum_channel_delta = max(maximum_channel_delta, frame_maximum)
            maximum_frame_mean = max(maximum_frame_mean, frame_mean)
            maximum_frame_changed_fraction = max(
                maximum_frame_changed_fraction, frame_changed_fraction
            )
            if frame_mean > VISUAL_PARITY_THRESHOLDS[
                "materialFrameMeanAbsoluteChannelDelta"
            ]:
                materially_different_frames.append(frames - 1)
    finally:
        timer.cancel()
        for process in processes:
            if process.stdout is not None:
                process.stdout.close()
            if process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    _terminate_process_tree(process)

    if timed_out.is_set():
        raise TimeoutError(
            "decoded visual parity exceeded "
            f"{FFMPEG_OPERATION_TIMEOUT_SECONDS}s"
        )

    errors: list[str] = []
    for process in processes:
        assert process.stderr is not None
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        if process.wait(timeout=30):
            errors.append(stderr[-2000:])
    if errors:
        raise ValueError(f"decoded visual parity failed: {'; '.join(errors)}")
    if not frames:
        raise ValueError("decoded video streams did not contain any frames")

    evidence: dict[str, object] = {
        "normalization": "decoded_rgb24_full_frame_v1",
        "frames": frames,
        "changedFrames": changed_frames,
        "maximumChannelDelta": maximum_channel_delta,
        "meanAbsoluteChannelDelta": total_delta / total_channels,
        "changedChannelFraction": changed_channels / total_channels,
        "maximumFrameMeanAbsoluteChannelDelta": maximum_frame_mean,
        "maximumFrameChangedChannelFraction": maximum_frame_changed_fraction,
        "materiallyDifferentFrames": materially_different_frames,
        "thresholds": VISUAL_PARITY_THRESHOLDS,
    }
    evidence["passed"] = (
        float(evidence["meanAbsoluteChannelDelta"])
        <= VISUAL_PARITY_THRESHOLDS["meanAbsoluteChannelDelta"]
        and float(evidence["changedChannelFraction"])
        <= VISUAL_PARITY_THRESHOLDS["changedChannelFraction"]
        and float(evidence["maximumFrameMeanAbsoluteChannelDelta"])
        <= VISUAL_PARITY_THRESHOLDS["maximumFrameMeanAbsoluteChannelDelta"]
        and float(evidence["maximumFrameChangedChannelFraction"])
        <= VISUAL_PARITY_THRESHOLDS["maximumFrameChangedChannelFraction"]
        and len(materially_different_frames)
        <= VISUAL_PARITY_THRESHOLDS["maximumMateriallyDifferentFrames"]
    )
    return evidence, approved_digest.hexdigest(), plugin_digest.hexdigest()


def main() -> None:
    evidence: dict[str, object] = {"schemaVersion": 1, "exports": {}}
    for preset, paths in EXPORTS.items():
        approved = paths["approved"]
        plugin = paths["plugin"]
        manifest_path = paths["manifest"]
        assert isinstance(approved, Path) and isinstance(plugin, Path)
        assert isinstance(manifest_path, Path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        bundle = RenderBundleV1(json.loads(paths["bundle"].read_text(encoding="utf-8")))
        if (not bundle.verify() or bundle.value["exportPreset"] != preset
                or manifest["bundleSha256"] != bundle.bundle_sha256
                or manifest["timelineSha256"] != bundle.value["timeline"]["timelineSha256"]):
            raise ValueError(f"{preset}: render bundle identity mismatch")
        fps = bundle.value["render"]["fps"]
        frames = bundle.value["timeline"]["totalFrames"]
        if manifest["renderer"]["id"] != "premium_video":
            raise ValueError(f"{preset}: wrong renderer identity")
        if manifest["exportPreset"] != preset:
            raise ValueError(f"{preset}: wrong export preset")
        if not manifest["observerSafe"] or not manifest["deterministic"]:
            raise ValueError(f"{preset}: observer safety or determinism is not declared")
        for file_record in manifest["files"].values():
            artifact = OUTPUT / file_record["name"]
            if artifact.stat().st_size != file_record["bytes"] or _sha256(artifact) != file_record["sha256"]:
                raise ValueError(f"{preset}: artifact hash mismatch for {artifact.name}")
        approved_hash = _sha256(approved)
        plugin_hash = _sha256(plugin)
        approved_probe = _probe(approved)
        probe = _probe(plugin)
        if approved_probe != probe:
            raise ValueError(f"{preset}: approved and plugin media probes differ")
        approved_audio_hash = _decoded_audio_sha256(approved)
        plugin_audio_hash = _decoded_audio_sha256(plugin)
        visual_parity, approved_video_hash, plugin_video_hash = _decoded_visual_parity(approved, plugin)
        if approved_audio_hash != plugin_audio_hash or not visual_parity["passed"]:
            raise ValueError(f"{preset}: normalized plugin media differs from the approved export")
        streams = {stream["codec_type"]: stream for stream in probe["streams"]}
        video = streams["video"]
        audio = streams["audio"]
        if (
            video["codec_name"] != "h264"
            or (video["width"], video["height"]) != (1080, 1920)
            or video["pix_fmt"] != "yuv420p"
            or video["r_frame_rate"] != f"{fps}/1"
            or int(video["nb_frames"]) != frames
            or visual_parity["frames"] != frames
            or video["color_space"] != "bt709"
            or video.get("color_transfer", "bt709") != "bt709"
            or video.get("color_primaries", "bt709") != "bt709"
            or audio["codec_name"] != "aac"
            or audio["sample_rate"] != "48000"
            or audio["channels"] != 2
        ):
            raise ValueError(f"{preset}: media contract mismatch")
        evidence["exports"][preset] = {
            "approvedVideo": approved.relative_to(ROOT).as_posix(),
            "pluginVideo": plugin.relative_to(ROOT).as_posix(),
            "containerByteIdentical": approved_hash == plugin_hash,
            "approvedContainerSha256": approved_hash,
            "pluginContainerSha256": plugin_hash,
            "decodedVideoByteIdentical": approved_video_hash == plugin_video_hash,
            "approvedDecodedVideoSha256": approved_video_hash,
            "pluginDecodedVideoSha256": plugin_video_hash,
            "normalizedMediaEquivalent": True,
            "visualParity": visual_parity,
            "decodedAudioSha256": plugin_audio_hash,
            "probe": probe,
            "bundleSha256": manifest["bundleSha256"],
            "timelineSha256": manifest["timelineSha256"],
        }
    destination = OUTPUT / "parity-evidence.json"
    atomic_write_text(
        destination,
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(destination)


if __name__ == "__main__":
    main()
