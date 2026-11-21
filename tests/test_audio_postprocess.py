from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from array import array
import wave

import pytest

from baloot_platform.video import audio_postprocess


@pytest.mark.parametrize("context_frames", [None, 0, 90])
def test_full_replay_schedules_every_card_and_collection_without_context(context_frames):
    replay_frames = 1980 if not context_frames else 1890
    segments = [{"stage": "replay", "startFrame": 900, "durationFrames": replay_frames},
        {"stage": "question", "startFrame": 2880, "durationFrames": 120},
        {"stage": "continuation", "startFrame": 3000, "durationFrames": 1320}]
    if context_frames is not None:
        segments.insert(1, {"stage": "context", "startFrame": 900 + replay_frames, "durationFrames": context_frames})
    tricks = [{"winner": "north", "plays": [{"seat": seat} for seat in ("north", "west", "south", "east")]} for _ in range(8)]
    candidate = SimpleNamespace(verify=lambda: True, bundle_sha256="test", value={
        "render": {"fps": 60}, "exportPreset": "hooked_flagship",
        "identity": {"gameSerial": "test-scenario"},
        "presentation": {"audio": {"enabled": True, "card_clicks": True, "table_foley": True,
            "target_lufs": -18, "normalization_true_peak_dbtp": -2.5}},
        "timeline": {"totalFrames": 4320, "segments": segments},
        "scenes": [{"stage": "continuation", "layers": [{"id": "public-lifecycle", "content": {
            "completedTricks": tricks, "pauseBeforePlayIndex": 24}}]}]})
    events = audio_postprocess.build_audio_cue_plan(candidate)["events"]
    assert len([row for row in events if row["kind"] == "card"]) == 32
    assert len([row for row in events if row["kind"] == "collect"]) == 8
    assert len([row for row in events if row["kind"] == "card" and row["frame"] < 2880]) == 24


@pytest.mark.parametrize("scenario_id", ["test-scenario", ""])
def test_foley_rejects_unversioned_scenario_plan(tmp_path: Path, scenario_id: str) -> None:
    plan = {
        "scenarioId": scenario_id,
        "sampleRate": 8000, "fps": 20, "totalFrames": 1,
        "planSha256": "12345678" + "0" * 56,
        "events": [],
    }
    dest = tmp_path / "foley.wav"
    with pytest.raises(ValueError, match="^MISSING_FOLEY_SEED_DERIVATION_VERSION"):
        audio_postprocess.generate_original_foley(plan, dest)
    assert not dest.exists()


def test_foley_seed_version_and_cue_locality(tmp_path: Path) -> None:
    legacy = {
        "sampleRate": 8000, "fps": 20, "totalFrames": 30,
        "planSha256": "12345678" + "0" * 56,
        "events": [
            {"kind": "card", "frame": 4, "seat": "west"},
            {"kind": "collect", "frame": 20, "seat": "east"},
        ],
    }
    dest = tmp_path / "foley.wav"
    # Captured with the unversioned generator before the seed change.
    assert audio_postprocess.generate_original_foley(legacy, dest)["sha256"] == (
        "ca400aa8db7cdadbd9542f6c46c6cc01bd809cdfe45248aa9d8a73b795d9aee4"
    )
    plan = {
        **legacy,
        "seedDerivationVersion": 2, "scenarioId": "test-scenario",
        "targetLufs": -18, "normalizationTruePeakDbtp": -2.5,
        "profileSha256": "a" * 64,
    }
    audio_postprocess.generate_original_foley(plan, dest)
    before = dest.read_bytes()
    audio_postprocess.generate_original_foley({**plan, "profileSha256": "b" * 64}, dest)
    assert dest.read_bytes() == before
    moved = {**plan, "events": [{**plan["events"][0], "frame": 5}, plan["events"][1]]}
    audio_postprocess.generate_original_foley(moved, dest)
    after = dest.read_bytes()
    # The moving card occupies frames 4 through 6 at the two positions.
    first_byte, last_byte = (44 + frame * 400 * 4 for frame in (4, 7))
    assert before[first_byte:last_byte] != after[first_byte:last_byte]
    assert before[:first_byte] == after[:first_byte]
    assert before[last_byte:] == after[last_byte:]
    moved_start = first_byte + 400 * 4
    assert before[first_byte:first_byte + 440 * 4] != after[moved_start:moved_start + 440 * 4]

    for changes in ({"scenarioId": "other-scenario"}, {"targetLufs": -19}):
        audio_postprocess.generate_original_foley({**plan, **changes}, dest)
        assert dest.read_bytes() != before
    for unknown_version in (3, None, True, 2.0):
        with pytest.raises(ValueError, match="unsupported Foley seed derivation version"):
            audio_postprocess.generate_original_foley({**plan, "seedDerivationVersion": unknown_version}, dest)


def _measurement(
    *, integrated_lufs: float = -27.4, true_peak_dbtp: float = -8.2, target_lufs: float = -18.0
) -> dict[str, float]:
    return {
        "integratedLufs": integrated_lufs,
        "truePeakDbtp": true_peak_dbtp,
        "loudnessRangeLu": 9.9,
        "thresholdLufs": -37.8,
        "targetLufs": target_lufs,
        "targetOffset": 0.6,
    }


def test_loudness_normalization_uses_the_measured_two_pass_filter() -> None:
    assert audio_postprocess.AUDIO_FINISHER_ID == "balootx_original_foley_v10"
    filter_value = audio_postprocess._measured_loudnorm_filter(
        {"targetLufs": -18.0, "normalizationTruePeakDbtp": -2.5},
        _measurement(),
    )

    assert filter_value.startswith("loudnorm=I=-18.0:TP=-2.5:LRA=11:")
    assert "measured_I=-27.4" in filter_value
    assert "measured_TP=-8.2" in filter_value
    assert "measured_LRA=9.9" in filter_value
    assert "measured_thresh=-37.8" in filter_value
    assert "offset=0.6" in filter_value
    assert ":linear=true:" in filter_value

    with pytest.raises(ValueError, match="must be finite"):
        audio_postprocess._measured_loudnorm_filter(
            {"targetLufs": -18.0, "normalizationTruePeakDbtp": -2.5},
            _measurement(integrated_lufs=float("-inf")),
        )


def test_bounded_linear_gain_is_used_only_when_both_limits_can_pass() -> None:
    plan = {"targetLufs": -18.0}

    assert audio_postprocess._bounded_linear_gain_db(
        plan, _measurement(integrated_lufs=-19.0, true_peak_dbtp=-1.44)
    ) == -0.16
    assert audio_postprocess._bounded_linear_gain_db(
        plan, _measurement(integrated_lufs=-20.47, true_peak_dbtp=0.75)
    ) is None
    assert audio_postprocess._bounded_linear_gain_db(
        plan, _measurement(integrated_lufs=-20.12, true_peak_dbtp=-2.09)
    ) is None
    assert audio_postprocess._bounded_linear_gain_db(
        plan, _measurement(integrated_lufs=-20.12, true_peak_dbtp=-2.5)
    ) == 0.72
    assert audio_postprocess._bounded_linear_gain_db(
        plan, _measurement(integrated_lufs=-19.3, true_peak_dbtp=-1.41)
    ) == -0.14


def test_peak_control_reduces_transients_using_frozen_audio_targets(tmp_path: Path) -> None:
    source, destination = tmp_path / "mix.wav", tmp_path / "controlled.wav"
    with wave.open(str(source), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(48_000)
        stream.writeframes(array("h", [2000, -2000, 30000, -30000]).tobytes())
    receipt = audio_postprocess._peak_controlled_mix(
        source, destination,
        plan={"targetLufs": -18.0, "normalizationTruePeakDbtp": -2.5},
        measurement=_measurement(integrated_lufs=-18.78, true_peak_dbtp=-0.07),
    )
    assert receipt is not None and receipt["allowedPeakDbtp"] < -2.5
    with wave.open(str(destination), "rb") as stream:
        controlled = array("h", stream.readframes(stream.getnframes()))
    assert abs(controlled[2]) < 30000
    assert abs(controlled[0] - 2000) < 10

@pytest.mark.parametrize(("integrated_lufs", "target_lufs"), [(-21.3, -19.4), (-19.4, -19.3)])
def test_failed_audio_validation_does_not_replace_the_source_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, integrated_lufs: float, target_lufs: float
) -> None:
    video = tmp_path / "source.mp4"
    evidence = tmp_path / "source.audio.json"
    video.write_bytes(b"approved-source-media")
    evidence.write_text("approved-evidence\n", encoding="utf-8")
    plan = {
        "seedDerivationVersion": 2,
        "targetLufs": -18.0,
        "normalizationTruePeakDbtp": -2.5,
        "sampleRate": 48_000,
        "profileSha256": "profile",
        "planSha256": "plan",
    }
    bundle = SimpleNamespace(
        value={
            "render": {"width": 1080, "height": 1920, "fps": 60},
            "timeline": {"totalFrames": 60},
        },
        bundle_sha256="bundle",
    )
    valid_probe = {
        "durationSeconds": 1.0,
        "video": {
            "codec": "h264",
            "width": 1080,
            "height": 1920,
            "frameRate": "60/1",
            "pixelFormat": "yuv420p",
            "colorSpace": "bt709",
            "colorPrimaries": "bt709",
            "colorTransfer": "bt709",
            "decodedFrames": 60,
        },
        "audio": {"codec": "aac", "sampleRate": 48_000, "channels": 2},
    }
    loudness_calls = 0

    def fake_foley(_plan, destination: Path):
        destination.write_bytes(b"foley")
        return {"sha256": "foley", "bytes": 5}

    def fake_media_command(command: list[str], **_kwargs):
        output = Path(command[-1])
        if output.suffix in {".wav", ".mp4"}:
            output.write_bytes(b"candidate-media")
        return SimpleNamespace(returncode=0)

    def fake_loudness(*_args, **_kwargs):
        nonlocal loudness_calls
        loudness_calls += 1
        if loudness_calls == 1:
            return _measurement()
        return _measurement(integrated_lufs=integrated_lufs, target_lufs=target_lufs)

    monkeypatch.setattr(audio_postprocess, "build_audio_cue_plan", lambda _bundle: plan)
    monkeypatch.setattr(audio_postprocess, "generate_original_foley", fake_foley)
    monkeypatch.setattr(audio_postprocess, "media_tool", lambda *_args: Path("tool"))
    monkeypatch.setattr(audio_postprocess, "probe_media", lambda *_args, **_kwargs: valid_probe)
    monkeypatch.setattr(audio_postprocess, "run_media_command", fake_media_command)
    monkeypatch.setattr(audio_postprocess, "probe_loudness", fake_loudness)
    monkeypatch.setattr(audio_postprocess, "_peak_controlled_mix", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(audio_postprocess, "_video_stream_sha256", lambda *_args, **_kwargs: "video-stream")

    with pytest.raises(ValueError, match="loudness"):
        audio_postprocess.finish_remotion_audio(
            video,
            bundle,
            repository_root=tmp_path,
            evidence_path=evidence,
        )

    assert video.read_bytes() == b"approved-source-media"
    assert evidence.read_text(encoding="utf-8") == "approved-evidence\n"


def test_failed_post_encode_audio_is_normalized_until_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "source.mp4"
    evidence_path = tmp_path / "source.audio.json"
    video.write_bytes(b"source-media")
    monkeypatch.setenv("BALOOTX_RELEASE_TARGET_LUFS", "-17.9")
    plan = {
        "seedDerivationVersion": 2,
        "targetLufs": -18.0,
        "normalizationTruePeakDbtp": -2.5,
        "sampleRate": 48_000,
        "profileSha256": "profile",
        "planSha256": "plan",
    }
    bundle = SimpleNamespace(
        value={
            "render": {"width": 1080, "height": 1920, "fps": 60},
            "timeline": {"totalFrames": 60},
        },
        bundle_sha256="bundle",
    )
    valid_probe = {
        "durationSeconds": 1.0,
        "video": {
            "codec": "h264",
            "width": 1080,
            "height": 1920,
            "frameRate": "60/1",
            "pixelFormat": "yuv420p",
            "colorSpace": "bt709",
            "colorPrimaries": "bt709",
            "colorTransfer": "bt709",
            "decodedFrames": 60,
        },
        "audio": {"codec": "aac", "sampleRate": 48_000, "channels": 2},
    }
    commands: list[list[str]] = []
    measurements = iter(
        (
            _measurement(),
            _measurement(integrated_lufs=-20.2, true_peak_dbtp=0.31, target_lufs=-19.4),
            _measurement(integrated_lufs=-20.47, true_peak_dbtp=0.75, target_lufs=-19.4),
            _measurement(integrated_lufs=-19.4, true_peak_dbtp=-1.72, target_lufs=-19.4),
        )
    )

    def fake_media_command(command: list[str], **_kwargs):
        commands.append(command)
        output = Path(command[-1])
        if output.suffix in {".wav", ".mp4"}:
            output.write_bytes(b"candidate-media")
        return SimpleNamespace(returncode=0)

    def fake_foley(_plan, destination: Path):
        destination.write_bytes(b"foley")
        return {"sha256": "foley", "bytes": 5}

    monkeypatch.setattr(audio_postprocess, "build_audio_cue_plan", lambda _bundle: plan)
    monkeypatch.setattr(audio_postprocess, "generate_original_foley", fake_foley)
    monkeypatch.setattr(audio_postprocess, "media_tool", lambda *_args: Path("tool"))
    monkeypatch.setattr(
        audio_postprocess, "probe_media", lambda *_args, **_kwargs: valid_probe
    )
    monkeypatch.setattr(audio_postprocess, "run_media_command", fake_media_command)
    monkeypatch.setattr(
        audio_postprocess,
        "probe_loudness",
        lambda *_args, **_kwargs: next(measurements),
    )
    monkeypatch.setattr(audio_postprocess, "_peak_controlled_mix", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        audio_postprocess,
        "_video_stream_sha256",
        lambda *_args, **_kwargs: "video-stream",
    )

    result = audio_postprocess.finish_remotion_audio(
        video,
        bundle,
        repository_root=tmp_path,
        evidence_path=evidence_path,
    )

    assert result["postEncodeLoudnessGainDb"] == 1.07
    assert result["seedDerivationVersion"] == 2
    assert result["postEncodeTruePeakTrimDb"] == -2.27
    assert len(result["postEncodeCorrections"]) == 2
    assert result["loudness"]["integratedLufs"] == -19.4
    assert result["loudness"]["truePeakDbtp"] == -1.72
    loudnorm_commands = [
        command
        for command in commands
        if any("loudnorm=I=" in argument for argument in command)
    ]
    assert len(loudnorm_commands) == 3
    filters = [
        argument
        for command in loudnorm_commands
        for argument in command
        if "loudnorm=I=" in argument
    ]
    assert "measured_I=-27.4" in filters[0]
    assert filters[1].startswith("loudnorm=I=-19.4:TP=-4.33:LRA=11:linear=false:")
    assert filters[2].startswith("loudnorm=I=-19.4:TP=-4.77:LRA=11:linear=false:")
    assert result["videoStreamPreserved"] is True
    assert video.read_bytes() == b"candidate-media"
    assert evidence_path.is_file()
