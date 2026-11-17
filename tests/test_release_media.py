from __future__ import annotations

import copy
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image

import baloot_platform.video.remotion_renderer as remotion_renderer_module
from baloot_platform.video.render_bundle import RenderBundleV1, resolve_motion_profile
from baloot_platform.video.remotion_renderer import _timestamp
from scripts import (
    generate_professional_video_prototypes,
    render_release,
    render_remotion_plugin_parity,
    verify_professional_decision_clip,
    verify_remotion_plugin_parity,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", params=["current", "historical"])
def decision_export_bundle(request, tmp_path_factory):
    destination = tmp_path_factory.mktemp(f"decision-{request.param}")
    with pytest.MonkeyPatch.context() as patch:
        if request.param == "historical":
            profile = yaml.safe_load(generate_professional_video_prototypes.PROFILE.read_text(encoding="utf-8"))
            historical = {"presentation": {"exports": {"decision_clip": json.loads((ROOT / "tests/fixtures/legacy-decision-export.json").read_text(encoding="utf-8"))}}}
            profile["exports"]["decision_clip"] = historical["presentation"]["exports"]["decision_clip"]
            profile["presentation"].pop("show_justification_text", None)
            profile_path = destination / "profile.yaml"
            profile_path.write_text(yaml.safe_dump(profile), encoding="utf-8")
            patch.setattr(generate_professional_video_prototypes, "PROFILE", profile_path)
        _, path = generate_professional_video_prototypes.generate(destination)
        value = json.loads(path.read_text(encoding="utf-8"))
    assert RenderBundleV1(value).verify()
    return value


@pytest.mark.parametrize("invalid", [None, "fps", "frames", "duration", "captions"])
def test_decision_export_verifier_uses_its_frozen_bundle(
    decision_export_bundle, tmp_path, monkeypatch, invalid,
):
    module = verify_professional_decision_clip
    bundle = decision_export_bundle
    for name, filename in (("BUNDLE", "bundle.json"), ("VIDEO", "video.mp4"),
                           ("CAPTIONS", "captions.vtt"), ("AUDIO_EVIDENCE", "audio.json")):
        monkeypatch.setattr(module, name, tmp_path / filename)
    monkeypatch.setattr(module, "OUTPUT", tmp_path)
    module.BUNDLE.write_text(json.dumps(bundle), encoding="utf-8")
    module.VIDEO.write_bytes(b"unit-test encoded-media stand-in")
    module.AUDIO_EVIDENCE.write_text(json.dumps({
        "status": "passed", "bundleSha256": bundle["bundleSha256"], "music": False,
        "videoStreamPreserved": True,
        "output": {"sha256": hashlib.sha256(module.VIDEO.read_bytes()).hexdigest()},
    }), encoding="utf-8")
    fps = bundle["render"]["fps"]
    frames = bundle["timeline"]["totalFrames"]
    reveal = next(item for item in bundle["timeline"]["segments"] if item["stage"] == "reveal")
    start = reveal["startFrame"]
    cue = f"{_timestamp(start, fps)} --> {_timestamp(start + min(240, reveal['durationFrames']), fps)}"
    no_text = bundle["presentation"]["presentation"].get("show_justification_text") is False
    if not no_text or invalid == "captions":
        module.CAPTIONS.write_text("WEBVTT\n\n" + ("wrong cue" if invalid == "captions" else cue), encoding="utf-8")
    for name in module.STILLS:
        Image.new("RGB", (1080, 1920)).save(tmp_path / name)
    probe = {"streams": [{
        "codec_name": "h264", "width": 1080, "height": 1920, "pix_fmt": "yuv420p",
        "color_space": "bt709", "color_primaries": "bt709", "color_transfer": "bt709",
        "r_frame_rate": f"{fps + (1 if invalid == 'fps' else 0)}/1",
        "nb_read_frames": str(frames + (1 if invalid == "frames" else 0)),
    }, {"codec_name": "aac", "sample_rate": "48000", "channels": 2}],
        "format": {"duration": str(frames / fps + (1 if invalid == "duration" else 0))}}
    monkeypatch.setattr(module.shutil, "which", lambda _name: "ffprobe")
    monkeypatch.setattr(module, "run_media_command", lambda *_args, **_kwargs: SimpleNamespace(stdout=json.dumps(probe)))
    if invalid is not None:
        with pytest.raises(ValueError, match="media validation|captions"):
            module.main()
    else:
        module.main()
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["story"]["durationFrames"] == frames
        assert manifest["bundleSha256"] == bundle["bundleSha256"]
        if no_text:
            assert manifest["captions"] == {"policy": "no_justification_text", "present": False}


def test_professional_identity_is_seeded_unique_and_bound_to_legal_source(tmp_path):
    module = generate_professional_video_prototypes
    identities = []
    for index, seed in enumerate((20261114, 20261114, 20261115)):
        output = tmp_path / str(index)
        _, path = module.generate(output, seed=seed)
        bundle = json.loads(path.read_text(encoding="utf-8"))
        manifest = json.loads((output / "source-manifest.json").read_text(encoding="utf-8"))
        players = bundle["identity"]["players"]
        assert len(set(players.values())) == 4
        assert set(players.values()) <= set(module.SAUDI_PLAYER_NAMES)
        assert manifest["players"] == players and manifest["identitySeed"] == seed
        assert manifest["legalReplay"] == "passed" and manifest["sourcePlays"] == 32
        assert manifest["sourceSha256"] == hashlib.sha256((output / "source.json").read_bytes()).hexdigest()
        assert manifest["bundles"][path.name] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert bundle["presentation"]["presentation"]["show_justification_text"] is False
        identities.append(players)
    assert identities[0] == identities[1]
    assert identities[0] != identities[2]


@pytest.mark.parametrize("invalid", [None, "cpu", "memory", "shm", "workspace", "font"])
def test_first_clip_preflight_refuses_unbounded_resources_or_changed_fonts(tmp_path, monkeypatch, invalid):
    from scripts import first_clip_job_metrics as module

    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "cpu.max").write_text("500000 100000" if invalid == "cpu" else "400000 100000")
    (cgroup / "memory.max").write_text(str((9 if invalid == "memory" else 8) * 1024**3))
    (cgroup / "cpu.stat").write_text("usage_usec 1000000\n")
    fonts = tmp_path / "assets/fonts"
    fonts.mkdir(parents=True)
    (fonts / "Noto.ttf").write_bytes(b"changed" if invalid == "font" else b"font")
    (fonts / "manifest.json").write_text(json.dumps({"fonts": {"Noto.ttf": {
        "sha256": hashlib.sha256(b"font").hexdigest(),
    }}}))
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "CGROUP", cgroup)
    def statvfs(path):
        size = (3 if invalid == "shm" else 2) if str(path) == "/dev/shm" else (30 if invalid == "workspace" else 20)
        return SimpleNamespace(f_blocks=size * 1024**3, f_frsize=1)
    monkeypatch.setattr(module.os, "statvfs", statvfs, raising=False)
    if invalid:
        with pytest.raises(ValueError):
            module.preflight(tmp_path)
        assert not (tmp_path / "pod-preflight.json").exists()
    else:
        module.preflight(tmp_path)
        assert json.loads((tmp_path / "pod-preflight.json").read_text())["cpuLimit"] == 4




def test_first_clip_run_exports_repo_root_for_direct_script_import(tmp_path, monkeypatch):
    from scripts import first_clip_job_metrics as module

    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "cpu.stat").write_text("usage_usec 1000000\n")
    (cgroup / "memory.peak").write_text(str(1024))
    monkeypatch.setattr(module, "CGROUP", cgroup)
    monkeypatch.setenv("PYTHONPATH", "src")
    output = tmp_path / "output"
    output.mkdir()
    (output / "pod-preflight.json").write_text(json.dumps({
        "cpuSecondsStart": 1.0,
        "startedEpoch": time.time(),
    }))

    captured = {}

    class CompletedProcess:
        pid = 123
        returncode = 0

        def poll(self):
            return self.returncode

    def popen(command, **kwargs):
        captured.update(kwargs)
        return CompletedProcess()

    monkeypatch.setattr(module.subprocess, "Popen", popen)
    monkeypatch.setattr(module.os, "statvfs", lambda _path: SimpleNamespace(
        f_blocks=1, f_bfree=0, f_frsize=1,
    ), raising=False)
    monkeypatch.setattr(module.os, "getpgid", lambda _pid: 0, raising=False)
    module.run(output, public=True, source_head="a" * 40, source_tree="b" * 40)

    entries = captured["env"]["PYTHONPATH"].split(os.pathsep)
    assert entries[:2] == [str(module.ROOT), str(module.ROOT / "src")]
    assert entries[2:] == ["src"]


@pytest.mark.parametrize("selector", [{"deal_id": "fixture"}, {"seed": "1"}])
def test_first_clip_selectors_require_public(tmp_path, selector):
    from scripts import first_clip_job_metrics as module

    with pytest.raises(ValueError, match="require --public"):
        module.run(tmp_path, **selector)


def test_frozen_curated_hand_is_legal_and_has_a_meaningful_late_choice():
    from baloot_platform.config import build_bundle
    from baloot_platform.core.episode_io import load_episode_spec

    path = ROOT / "examples/curated-decision-20261114.json"
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sourceSha256"]
    episode = load_episode_spec(path, build_bundle(ROOT).rules)
    assert len(episode.final_state.played_cards) == 32
    assert episode.pause_before_play_index == 22
    assert episode.final_state.seed == 20261117
    assert abs(episode.final_state.match_score[0] - episode.final_state.match_score[1]) == 2
    best, alternative = episode.analysis.actions[:2]
    assert best.expected_match_margin - alternative.expected_match_margin >= 3
    assert best.win_probability - alternative.win_probability >= 0.1


@pytest.mark.parametrize("invalid", [None, "fps", "frames", "bundle"])
def test_plugin_parity_verifier_uses_its_frozen_bundle(
    decision_export_bundle, tmp_path, monkeypatch, invalid,
):
    module = verify_remotion_plugin_parity
    bundle = decision_export_bundle
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    approved, plugin, manifest_path = [tmp_path / name for name in ("approved.mp4", "plugin.mp4", "manifest.json")]
    approved.write_bytes(b"approved media stand-in")
    plugin.write_bytes(b"plugin media stand-in")
    manifest_path.write_text(json.dumps({
        "renderer": {"id": "premium_video"}, "exportPreset": "decision_clip",
        "observerSafe": True, "deterministic": True, "files": {},
        "bundleSha256": "0" * 64 if invalid == "bundle" else bundle["bundleSha256"],
        "timelineSha256": bundle["timeline"]["timelineSha256"],
    }), encoding="utf-8")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "OUTPUT", tmp_path)
    monkeypatch.setattr(module, "EXPORTS", {"decision_clip": {
        "approved": approved, "plugin": plugin, "manifest": manifest_path, "bundle": bundle_path}})
    fps = bundle["render"]["fps"]
    frames = bundle["timeline"]["totalFrames"]
    probe = {"streams": [{
        "codec_type": "video", "codec_name": "h264", "width": 1080, "height": 1920,
        "pix_fmt": "yuv420p", "r_frame_rate": f"{fps + (1 if invalid == 'fps' else 0)}/1",
        "nb_frames": str(frames + (1 if invalid == "frames" else 0)),
        "color_space": "bt709", "color_transfer": "bt709", "color_primaries": "bt709",
    }, {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000", "channels": 2}]}
    monkeypatch.setattr(module, "_probe", lambda _path: probe)
    monkeypatch.setattr(module, "_decoded_sha256", lambda *_args, **_kwargs: "same-decoded-bytes")
    monkeypatch.setattr(module, "_decoded_visual_parity", lambda *_args: {"passed": True, "frames": frames})
    if invalid is not None:
        with pytest.raises(ValueError, match="bundle|media contract"):
            module.main()
    else:
        module.main()
        assert json.loads((tmp_path / "parity-evidence.json").read_text(encoding="utf-8"))["exports"]["decision_clip"]["bundleSha256"] == bundle["bundleSha256"]


def test_hitl_release_profile_is_exact_mobile_portrait() -> None:
    profile = yaml.safe_load(
        (ROOT / "configs/hitl/video.yaml").read_text(encoding="utf-8")
    )
    clip = profile["review_clip"]
    assert profile["profile"] == {
        "id": "premium_mobile_hitl",
        "version": "21.1.0",
        "status": "release_candidate",
    }
    assert (clip["width"], clip["height"]) == (1080, 1920)
    assert clip["observer_safe"] is True
    assert clip["show_developer_card_codes"] is False


def test_normal_and_reduced_motion_are_governed() -> None:
    profile = yaml.safe_load(
        (ROOT / "configs/video/premium.yaml").read_text(encoding="utf-8")
    )
    normal = resolve_motion_profile(profile)
    reduced_profile = copy.deepcopy(profile)
    reduced_profile["motion"]["mode"] = "reduced"
    reduced = resolve_motion_profile(reduced_profile)

    assert normal["mode"] == "normal"
    assert reduced["mode"] == "reduced"
    assert reduced["available_modes"] == ["normal", "reduced"]
    assert reduced["flight_steps"] < normal["flight_steps"]
    assert reduced["card_arc_height_px"] == 0
    assert reduced["winner_glow"] is False


def test_generated_schemas_enforce_vertical_release_and_motion_modes() -> None:
    hitl = json.loads(
        (ROOT / "schemas/profiles/hitl.schema.json").read_text(encoding="utf-8")
    )
    video = json.loads(
        (ROOT / "schemas/profiles/video.schema.json").read_text(encoding="utf-8")
    )
    release_gate = hitl["allOf"][0]["then"]["properties"]["review_clip"]
    assert release_gate["properties"]["width"]["const"] == 1080
    assert release_gate["properties"]["height"]["const"] == 1920
    assert video["properties"]["motion"]["properties"]["mode"]["enum"] == [
        "normal",
        "reduced",
    ]




def test_release_renderer_orchestrates_only_current_remotion_exports() -> None:
    source = (ROOT / "scripts/render_release.py").read_text(encoding="utf-8")
    assert "render-prototypes.mjs" in source
    assert "render-flagship.mjs" in source
    assert "render-decision-clip.mjs" in source
    assert 'environment={"BALOOTX_COLLECTION_ONLY": "1"}' in source
    assert "finish_release_audio()" in source
    assert "finish_remotion_audio(" in source
    assert "premium_v" not in source
    assert "render_v20" not in source


def test_plugin_parity_renderer_is_source_checkout_safe() -> None:
    environment = {**os.environ, "PYTHONPATH": ""}
    result = subprocess.run(
        [
            render_release.sys.executable,
            "-c",
            (
                "import runpy; "
                "runpy.run_path('scripts/render_remotion_plugin_parity.py', "
                "run_name='source_checkout_test')"
            ),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_plugin_parity_renderer_uses_the_release_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    def renderer(profile, *, repository_root, timeout_seconds):
        observed.update(
            profile=profile,
            repository_root=repository_root,
            timeout_seconds=timeout_seconds,
        )
        return object()

    monkeypatch.setattr(render_remotion_plugin_parity, "OUTPUT", tmp_path)
    monkeypatch.setattr(render_remotion_plugin_parity, "EXPORTS", ())
    monkeypatch.setattr(
        render_remotion_plugin_parity,
        "build_bundle",
        lambda _root: SimpleNamespace(video={"schema_version": 1}),
    )
    monkeypatch.setattr(
        render_remotion_plugin_parity,
        "RemotionProfessionalRenderer",
        renderer,
    )
    monkeypatch.setattr(
        render_remotion_plugin_parity,
        "render_timeout_seconds",
        lambda: 1800,
    )

    render_remotion_plugin_parity.main()

    assert observed == {
        "profile": {"schema_version": 1},
        "repository_root": render_remotion_plugin_parity.ROOT,
        "timeout_seconds": 1800,
    }


def test_release_renderer_runs_every_governed_media_stage_in_order(monkeypatch) -> None:
    events: list[tuple[object, ...]] = []
    monkeypatch.setattr(render_release, "release_render_lock", nullcontext)
    monkeypatch.setattr(
        render_release,
        "generate",
        lambda: events.append(("generate",)),
    )
    monkeypatch.setattr(
        render_release,
        "_node_command",
        lambda script: ["node", script],
    )

    def run(
        command: list[str],
        *,
        cwd: Path = render_release.ROOT,
        timeout: int | None = None,
        environment: dict[str, str] | None = None,
        **_evidence,
    ) -> None:
        events.append(("run", tuple(command), cwd, environment, timeout))

    monkeypatch.setattr(render_release, "_run", run)
    monkeypatch.setattr(
        render_release,
        "finish_release_audio",
        lambda: events.append(("audio",)),
    )
    monkeypatch.setattr(render_release, "render_timeout_seconds", lambda: 1200)

    render_release.main()

    assert events == [
        ("generate",),
        ("run", ("node", "render-prototypes.mjs"), render_release.REMOTION, None, None),
        ("run", ("node", "render-flagship.mjs"), render_release.REMOTION, None, None),
        (
            "run",
            ("node", "render-decision-clip.mjs"),
            render_release.REMOTION,
            None,
            None,
        ),
        (
            "run",
            ("node", "render-flagship.mjs"),
            render_release.REMOTION,
            {"BALOOTX_COLLECTION_ONLY": "1"},
            None,
        ),
        ("audio",),
        (
            "run",
            (
                render_release.sys.executable,
                "scripts/verify_professional_video_prototypes.py",
            ),
            render_release.ROOT,
            None,
            None,
        ),
        (
            "run",
            (render_release.sys.executable, "scripts/verify_professional_flagship.py"),
            render_release.ROOT,
            None,
            None,
        ),
        (
            "run",
            (
                render_release.sys.executable,
                "scripts/verify_professional_decision_clip.py",
            ),
            render_release.ROOT,
            None,
            None,
        ),
        (
            "run",
            (
                render_release.sys.executable,
                "scripts/render_remotion_plugin_parity.py",
            ),
            render_release.ROOT,
            None,
            1260,
        ),
        (
            "run",
            (
                render_release.sys.executable,
                "scripts/verify_remotion_plugin_parity.py",
            ),
            render_release.ROOT,
            None,
            None,
        ),
        (
            "run",
            (
                render_release.sys.executable,
                "scripts/generate_scene_preview.py",
                "--refresh",
            ),
            render_release.ROOT,
            None,
            None,
        ),
    ]


@pytest.mark.parametrize("failing_stage", ["prototypes", "collection"])
def test_release_renderer_stops_when_a_new_media_stage_fails(
    monkeypatch, failing_stage: str
) -> None:
    events: list[str] = []
    monkeypatch.setattr(render_release, "release_render_lock", nullcontext)
    monkeypatch.setattr(render_release, "generate", lambda: events.append("generate"))
    monkeypatch.setattr(
        render_release,
        "_node_command",
        lambda script: ["node", script],
    )

    def run(
        command: list[str],
        *,
        cwd: Path = render_release.ROOT,
        timeout: int | None = None,
        environment: dict[str, str] | None = None,
        **_evidence,
    ) -> None:
        del cwd, timeout
        stage = (
            "collection"
            if environment == {"BALOOTX_COLLECTION_ONLY": "1"}
            else "prototypes"
            if command[-1] == "render-prototypes.mjs"
            else str(command[-1])
        )
        events.append(stage)
        if stage == failing_stage:
            raise subprocess.TimeoutExpired(command, 1200)

    monkeypatch.setattr(render_release, "_run", run)
    monkeypatch.setattr(
        render_release,
        "finish_release_audio",
        lambda: events.append("audio"),
    )

    with pytest.raises(subprocess.TimeoutExpired):
        render_release.main()

    assert events == (
        ["generate", "prototypes"]
        if failing_stage == "prototypes"
        else [
            "generate",
            "prototypes",
            "render-flagship.mjs",
            "render-decision-clip.mjs",
            "collection",
        ]
    )


def test_release_renderer_requires_the_hidden_windows_node_wrapper(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        render_release,
        "os",
        SimpleNamespace(name="nt", environ={"BALOOT_HIDDEN_NODE_WRAPPER": str(tmp_path / "hidden-node-wrapper.ps1")}),
    )
    with pytest.raises(RuntimeError, match="hidden Node command wrapper"):
        render_release._node_command("render-prototypes.mjs")

    wrapper = tmp_path / "hidden-node-wrapper.ps1"
    wrapper.write_text("# test wrapper\n", encoding="utf-8")
    command = render_release._node_command("render-prototypes.mjs")
    assert command[0] == "powershell.exe"
    assert "-NonInteractive" in command
    assert command[command.index("-File") + 1] == str(wrapper)
    assert "-EncodedCommand" in command
    assert command[command.index("-WorkingDirectory") + 1] == str(
        render_release.REMOTION
    )


def test_release_run_preserves_failed_child_diagnostics(tmp_path: Path) -> None:
    evidence = tmp_path / "attempt"
    with pytest.raises(RuntimeError) as raised:
        render_release._run(
            [
                render_release.sys.executable,
                "-c",
                "import sys; print('child-out'); print('child-error', file=sys.stderr); sys.exit(7)",
            ],
            stage="diagnostic",
            evidence_dir=evidence,
        )

    assert "stage 'diagnostic' failed with exit code 7" in str(raised.value)
    assert "child-error" in str(raised.value)
    assert (evidence / "diagnostic.stdout.log").read_text(encoding="utf-8") == (
        "child-out\n"
    )
    assert (evidence / "diagnostic.stderr.log").read_text(encoding="utf-8") == (
        "child-error\n"
    )
    receipt = json.loads(
        (evidence / "diagnostic.receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["status"] == "failed"
    assert receipt["exitCode"] == 7
    assert receipt["stderr"]["bytes"] > 0


def test_release_renderer_help_never_starts_a_render() -> None:
    result = subprocess.run(
        [render_release.sys.executable, "scripts/render_release.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Render and verify" in result.stdout
    assert "outputs/professional-video" not in result.stdout


def test_release_render_timeout_is_configurable_and_bounded(monkeypatch) -> None:
    monkeypatch.delenv("BALOOT_RENDER_TIMEOUT_SECONDS", raising=False)
    assert render_release.render_timeout_seconds() == 1200

    for value in ("invalid", "299", "3601"):
        monkeypatch.setenv("BALOOT_RENDER_TIMEOUT_SECONDS", value)
        with pytest.raises(RuntimeError, match="must be an integer from 300 to 3600"):
            render_release.render_timeout_seconds()

    monkeypatch.setenv("BALOOT_RENDER_TIMEOUT_SECONDS", "3600")
    assert render_release.render_timeout_seconds() == 3600


@pytest.mark.parametrize("state,alive", [("Z", False), ("S", True), ("T", True)])
def test_process_liveness_distinguishes_linux_zombies(monkeypatch, state, alive):
    probes = []
    monkeypatch.setattr(remotion_renderer_module, "os", SimpleNamespace(
        name="posix", kill=lambda pid, signal: probes.append((pid, signal))))
    monkeypatch.setattr(Path, "read_text", lambda self, **kwargs: f"12345 (child (with) spaces) {state} 1 0 0")
    assert render_release._process_exists(12345) is alive
    assert probes == ([(12345, 0)] if alive else [])


def test_process_liveness_keeps_permission_denied_processes_alive(monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError("restricted process")

    monkeypatch.setattr(remotion_renderer_module, "os", SimpleNamespace(name="posix", kill=denied))
    monkeypatch.setattr(Path, "read_text", denied)
    assert render_release._process_exists(12345)


def test_release_run_terminates_an_owned_child_tree_on_timeout(tmp_path: Path) -> None:
    child_pid_path = tmp_path / "child.pid"
    parent = (
        "import pathlib, subprocess, sys, time; "
        "child=subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(30)']); "
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid)); "
        "time.sleep(30)"
    )

    with pytest.raises(subprocess.TimeoutExpired):
        render_release._run(
            [render_release.sys.executable, "-c", parent],
            timeout=1,
        )

    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 3
    while render_release._process_exists(child_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not render_release._process_exists(child_pid)


def test_release_render_lock_rejects_an_overlapping_writer(tmp_path: Path) -> None:
    lock = tmp_path / "render-release.lock"
    with render_release.release_render_lock(lock):
        with pytest.raises(RuntimeError, match="release render already active"):
            with render_release.release_render_lock(lock):
                raise AssertionError("overlapping writer was admitted")
    assert not lock.exists()


def test_release_render_lock_reclaims_a_stale_owner(tmp_path: Path) -> None:
    lock = tmp_path / "render-release.lock"
    lock.write_text(
        json.dumps({"pid": 2_147_483_647, "lockId": "stale"}),
        encoding="utf-8",
    )
    with render_release.release_render_lock(lock):
        assert json.loads(lock.read_text(encoding="utf-8"))["pid"] > 0
    assert not lock.exists()
