from pathlib import Path
import json
from types import SimpleNamespace

from PIL import Image
import pytest

from baloot_platform.video import media_probe, media_validation
from scripts import generate_media_probe


def test_video_reexports_the_shared_media_boundary():
    for name in ("run_media_command", "probe_media", "validate_media_probe"):
        assert getattr(media_validation, name) is getattr(media_probe, name)


def test_media_probe_refuses_renamed_measurement_mp4(monkeypatch):
    def fake_probe(command, **kwargs):
        assert "format_tags=comment" in command[command.index("-show_entries") + 1]
        return SimpleNamespace(stdout=json.dumps({
            "format": {"tags": {"comment": "balootx-layout-measurement-never-post"}},
            "streams": [],
        }))

    monkeypatch.setattr(media_probe, "run_media_command", fake_probe)
    with pytest.raises(ValueError, match="permanently NEVER_POST"):
        media_probe.probe_media(Path("renamed-release.mp4"), ffprobe=Path("ffprobe"))


@pytest.mark.parametrize("ci", [None, "false", "true"])
def test_configured_media_threads_cap_each_decoder_filters_and_encoder(tmp_path, monkeypatch, ci):
    profile = tmp_path / "ffmpeg-runtime.json"
    profile.write_text(json.dumps({"threads": 3}), encoding="utf-8")
    monkeypatch.setattr(media_probe, "_FFMPEG_RUNTIME_PATH", profile)
    if ci is None:
        monkeypatch.delenv("CI", raising=False)
    else:
        monkeypatch.setenv("CI", ci)
    commands = []
    monkeypatch.setattr(media_validation.subprocess, "run", lambda command, **kwargs: commands.append(command))
    command = ["ffmpeg", "-i", "video.mp4", "-i", "audio.wav", "-c:v", "copy", "finished.mp4"]
    media_validation.run_media_command(command)
    expected = [
        "ffmpeg", "-filter_threads", "3", "-filter_complex_threads", "3",
        "-threads", "3", "-i", "video.mp4", "-threads", "3", "-i", "audio.wav",
        "-c:v", "copy", "-threads", "3", "finished.mp4",
    ]
    assert commands == [expected]
    assert command == ["ffmpeg", "-i", "video.mp4", "-i", "audio.wav", "-c:v", "copy", "finished.mp4"]


def test_frame_count_decoder_is_capped_without_changing_version_commands(monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    threads = str(json.loads(media_probe._FFMPEG_RUNTIME_PATH.read_bytes())["threads"])
    commands = []
    monkeypatch.setattr(media_validation.subprocess, "run", lambda command, **kwargs: commands.append(command))
    media_validation.run_media_command(["ffprobe", "-count_frames", "video.mp4"])
    media_validation.run_media_command(["ffmpeg", "-version"])
    assert commands == [
        ["ffprobe", "-threads", threads, "-count_frames", "video.mp4"],
        ["ffmpeg", "-version"],
    ]


@pytest.mark.parametrize("profile", [{}, {"threads": 0}, {"threads": True}, {"threads": 1.5}])
def test_invalid_ffmpeg_thread_cap_fails_before_starting_media_tool(tmp_path, monkeypatch, profile):
    config = tmp_path / "ffmpeg-runtime.json"
    config.write_text(json.dumps(profile), encoding="utf-8")
    monkeypatch.setattr(media_probe, "_FFMPEG_RUNTIME_PATH", config)
    monkeypatch.setattr(media_probe.subprocess, "run", lambda *args, **kwargs: pytest.fail("started FFmpeg"))

    with pytest.raises(ValueError, match=r"ffmpeg-runtime\.json: threads must be a positive"):
        media_probe.run_media_command(["ffmpeg", "-i", "video.mp4", "output.mp4"])


def test_playback_checkpoint_seeks_to_exact_frame_time(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    Image.new("RGB", (4, 4), "red").save(source)
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        Image.new("RGB", (4, 4), "red").save(command[-1])

    monkeypatch.setattr(media_validation, "run_media_command", fake_run)
    result = media_validation.compare_source_frame(
        tmp_path / "video.mp4", source, frame=4079, fps=60,
        ffmpeg=Path("ffmpeg.exe"),
    )

    command = commands[0]
    assert command.index("-ss") < command.index("-i")
    assert command[command.index("-ss") + 1] == "67.983333333333"
    assert result["frame"] == 4079
    assert result["meanAbsoluteError"] == 0


@pytest.mark.parametrize(("fps", "manifest_target", "probe_target"), [
    (24, -19.4, -19.4), (60, -19.4, -19.4), (24, -19.3, -19.4), (24, -19.4, -19.3),
])
def test_verify_media_pins_loudness_and_compares_checkpoints(
    tmp_path, monkeypatch, fps, manifest_target, probe_target
):
    module = generate_media_probe
    monkeypatch.setattr(module, "ROOT", tmp_path)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"encoded media stand-in")
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text("{}", encoding="utf-8")
    value = {
        "render": {"width": 4, "height": 4, "fps": fps},
        "renderer": {},
        "timeline": {"totalFrames": 100, "segments": [{"stage": "reveal", "startFrame": 48}]},
    }
    monkeypatch.setattr(module, "RenderBundleV1", lambda _: SimpleNamespace(
        value=value, bundle_sha256="frozen-bundle", verify=lambda: True,
    ))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "bundleSha256": "frozen-bundle",
        "video": {"sha256": module.sha256(video), "bytes": video.stat().st_size},
        "audioFinishing": {"loudness": {"targetLufs": manifest_target}},
    }), encoding="utf-8")
    source = tmp_path / "reveal.png"
    Image.new("RGB", (4, 4), "red").save(source)
    probe = {"video": {
        "codec": "h264", "width": 4, "height": 4, "frameRate": f"{fps}/1",
        "pixelFormat": "yuv420p", "colorSpace": "bt709", "colorPrimaries": "bt709",
        "colorTransfer": "bt709", "decodedFrames": 100,
    }, "audio": {"codec": "aac", "sampleRate": 48000, "channels": 2}}
    monkeypatch.setattr(module, "probe_media", lambda *_args, **_kwargs: probe)
    monkeypatch.setattr(module, "probe_loudness", lambda *_args, **_kwargs: {
        "integratedLufs": -19.4, "targetLufs": probe_target, "truePeakDbtp": -2,
    })
    commands = []

    def fake_ffmpeg(command, **_kwargs):
        commands.append(command)
        Image.new("RGB", (4, 4), "red").save(command[-1])

    monkeypatch.setattr(media_validation, "run_media_command", fake_ffmpeg)
    item = {
        "video": video, "bundle": bundle_path, "manifest": manifest,
        "checkpoints": ((source.name, "reveal", 3),),
    }
    if manifest_target != -19.4 or probe_target != -19.4:
        with pytest.raises(ValueError, match="release loudness targetLufs"):
            module.verify_media(item, ffmpeg=Path("ffmpeg"), ffprobe=Path("ffprobe"))
        assert commands == []
        return

    result = module.verify_media(item, ffmpeg=Path("ffmpeg"), ffprobe=Path("ffprobe"))

    assert len(commands) == 1
    assert commands[0][commands[0].index("-ss") + 1] == f"{51 / fps:.12f}"
    assert result["sourceComparisons"][0]["frame"] == 51
    assert result["sourceComparisons"][0]["meanAbsoluteError"] == 0
