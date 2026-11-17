import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import wave

import pytest
import yaml

from scripts import generate_public_render_audio as speech


def test_public_speech_generation_binds_fresh_audio_to_public_text(tmp_path, monkeypatch, root):
    base = tmp_path / "assets/audio/voice_packs"
    for pack, filename in [("ar_sa_baloot_v1", "generated-manifest.json"), ("ar_sa_baloot_v2_smooth", "manifest.json")]:
        directory = base / pack
        directory.mkdir(parents=True)
        (directory / filename).write_bytes((root / "assets/audio/voice_packs" / pack / filename).read_bytes())
    engine = tmp_path / "engine"
    engine.write_bytes(b"synthetic executable")
    monkeypatch.setattr(speech.shutil, "which", lambda name: str(engine))
    invocations = []
    def run(command, **kwargs):
        invocations.append((command, kwargs))
        if command[-1] == "--version":
            return SimpleNamespace(stdout="eSpeak NG text-to-speech: 1.51 Data at: /synthetic/data")
        if "-w" not in command:
            with wave.open(command[-1], "wb") as stream:
                stream.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
                stream.writeframes(b"\x01\x00" * 480)
        return SimpleNamespace(stdout=b"")
    monkeypatch.setattr(speech, "_run", run)
    speech.generate(tmp_path)
    manifests = []
    for pack, filename in [("ar_sa_baloot_v1", "generated-manifest.json"), ("ar_sa_baloot_v2_smooth", "manifest.json")]:
        directory = base / pack
        manifest = json.loads((directory / filename).read_text(encoding="utf-8"))
        manifests.append(manifest)
        assert manifest["provider"] == "espeak-ng-ar-public"
        assert manifest["copiedReferenceAudio"] is False
        assert "Data at:" not in manifest["generatorVersion"]
        for clip in manifest["clips"].values():
            assert clip["sha256"] == hashlib.sha256((directory / clip["file"]).read_bytes()).hexdigest()
            assert clip["durationSeconds"] == 0.01
    assert [len(item["clips"]) for item in manifests] == [15, 16]
    synth = [row for row in invocations if "-w" in row[0]]
    assert len(synth) == 16
    assert all(row[0][1:5] == ["-v", "ar", "-s", "180"] and row[1]["input"] for row in synth)


def test_public_speech_fails_before_writes_without_qualified_tools(tmp_path, monkeypatch):
    monkeypatch.setattr(speech.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="install espeak-ng"):
        speech.generate(tmp_path)
    assert not list(tmp_path.iterdir())


def test_hosted_render_has_no_secret_or_host_dependency(root):
    text = (root / ".github/workflows/render.yml").read_text(encoding="utf-8")
    workflow = yaml.safe_load(text)
    assert workflow["permissions"] == {"contents": "read"}
    assert "pull_request_target" not in text and "secrets." not in text
    assert len(workflow["jobs"]["render"]["strategy"]["matrix"]["deal"]) == 2
    for job in workflow["jobs"].values():
        assert job["runs-on"] == "ubuntu-24.04"
        for step in job["steps"]:
            if "uses" in step:
                assert len(step["uses"].split("@")[1]) == 40
    assert "generate_public_render_audio.py" in text


def test_supported_card_back_matches_both_manifests(root):
    for prefix in [Path("assets"), Path("src/baloot_platform/resources/assets")]:
        directory = root / prefix / "video/professional"
        manifest = json.loads((directory / "manifest.json").read_text())
        item = manifest["assets"]["card-back.svg"]
        data = (directory / "card-back.svg").read_bytes()
        assert len(data) == item["bytes"]
        assert hashlib.sha256(data).hexdigest() == item["sha256"]
