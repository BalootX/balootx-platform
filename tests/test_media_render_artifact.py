from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

from scripts import media_render_artifact as media


@pytest.fixture
def package(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    for directory in media.OUTPUTS:
        path = root / directory / "media.bin"
        path.parent.mkdir(parents=True)
        path.write_bytes(directory.encode())
    inputs = {"source": "a" * 64, "toolchain": {"renderer": "4.0.503"}}
    identity = {"inputs": inputs, "identity": hashlib.sha256(media.canonical(inputs)).hexdigest()}
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setenv("GITHUB_REPOSITORY", "BalootX/BalootX")
    monkeypatch.setattr(media, "command", lambda *args: "a" * 40)
    destination = tmp_path / "artifact"
    media.seal(identity, destination, root)
    return root, identity, destination


def rewrite_manifest(package, change):
    path = package / "manifest.json"
    value = json.loads(path.read_bytes())
    change(value)
    media.write_json(path, value)
    (package / "manifest.sha256").write_text(media.sha(path))


def test_media_package_round_trip_replaces_only_outputs(package, tmp_path):
    source, identity, artifact = package
    root = tmp_path / "consumer"
    root.mkdir()
    (root / "keep.txt").write_text("source")
    stale = root / media.OUTPUTS[0] / "stale.mp4"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")
    media.restore(identity, artifact, "123", root)
    assert not stale.exists()
    assert (root / "keep.txt").read_text() == "source"
    assert {p.relative_to(root): p.read_bytes() for p in media.files(root)} == {
        p.relative_to(source): p.read_bytes() for p in media.files(source)
    }


@pytest.mark.parametrize("defect", ["manifest", "payload", "source", "toolchain", "identity", "producer", "files", "file_hash"])
def test_invalid_promotion_never_replaces_checked_out_media(package, tmp_path, defect):
    _, identity, artifact = package
    identity = copy.deepcopy(identity)
    if defect in {"source", "toolchain"}:
        identity["inputs"][defect] = "different"
    elif defect == "identity":
        identity["identity"] = "0" * 64
    elif defect == "manifest":
        (artifact / "manifest.json").write_bytes(b"{}")
    elif defect == "payload":
        with (artifact / "media.tar").open("ab") as stream:
            stream.write(b"tamper")
    elif defect == "producer":
        rewrite_manifest(artifact, lambda m: m["producer"].update(run="456"))
    elif defect == "files":
        rewrite_manifest(artifact, lambda m: m["files"].pop(next(iter(m["files"]))))
    else:
        rewrite_manifest(artifact, lambda m: next(iter(m["files"].values())).update(sha256="0" * 64))
    root = tmp_path / "consumer"
    marker = root / media.OUTPUTS[0] / "old.mp4"
    marker.parent.mkdir(parents=True)
    marker.write_bytes(b"old")
    with pytest.raises(ValueError):
        media.restore(identity, artifact, "123", root)
    assert marker.read_bytes() == b"old"
    assert list(root.rglob("*.bin")) == []


@pytest.mark.parametrize("name,kind", [("../escape", tarfile.REGTYPE),
    ("outputs/professional-video-flagship/../../escape", tarfile.REGTYPE),
    ("/tmp/escape", tarfile.REGTYPE), ("scripts/escape.py", tarfile.REGTYPE),
    ("outputs/professional-video-flagship/link", tarfile.SYMTYPE)])
def test_promotion_rejects_unsafe_archive_entries(package, tmp_path, name, kind):
    _, identity, artifact = package
    with tarfile.open(artifact / "media.tar", "w") as archive:
        member = tarfile.TarInfo(name)
        member.type = kind
        member.linkname = "../../escape" if kind == tarfile.SYMTYPE else ""
        archive.addfile(member, io.BytesIO())
    rewrite_manifest(artifact, lambda m: m.update(
        payloadSha256=media.sha(artifact / "media.tar"),
        files={name: {"bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}}))
    with pytest.raises(ValueError, match="unsafe"):
        media.restore(identity, artifact, "123", tmp_path / "consumer")


def test_source_identity_tracks_inputs_and_ignores_commit_and_docs(tmp_path):
    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)
    git("init")
    (tmp_path / "src").mkdir()
    (tmp_path / "src/input.py").write_text("first")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "input")
    first = media.source_identity(tmp_path)
    (tmp_path / "README.md").write_text("docs")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "docs")
    assert media.source_identity(tmp_path) == first
    (tmp_path / "src/input.py").write_text("second")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "input")
    assert media.source_identity(tmp_path) != first


def test_identity_failure_renders_without_publishing_an_unknown_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(media, "command", lambda *args: "")
    monkeypatch.setattr(media, "source_identity", lambda: "source")
    monkeypatch.setattr(media, "toolchain", lambda: (_ for _ in ()).throw(ValueError("unknown renderer")))
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    media.identify(tmp_path / "identity.json")
    assert json.loads((tmp_path / "identity.json").read_bytes()) == {}
    assert output.read_text() == "identity=\n"
    assert media.find_artifact("", "123", 1, lambda _: pytest.fail("unknown identity must not query")) == {}


@pytest.mark.parametrize("defect", [None, "expired", "current_run", "failed", "pending", "fork", "workflow", "event"])
def test_find_requires_successful_same_repository_ci_run(defect):
    artifact = {"id": 9, "expired": defect == "expired", "workflow_run": {"id": 123}}
    run = {"path": ".github/workflows/ci.yml", "status": "completed", "conclusion": "success",
           "event": "pull_request", "head_repository": {"id": 1}}
    for label, key, value in [("failed", "conclusion", "failure"), ("pending", "status", "in_progress"),
                              ("workflow", "path", ".github/workflows/publish.yml"),
                              ("event", "event", "workflow_dispatch"), ("fork", "head_repository", {"id": 2})]:
        if defect == label:
            run[key] = value
    def get(path):
        return {"artifacts": [artifact]} if path.startswith("actions/artifacts?") else run
    found = media.find_artifact("a" * 64, "123" if defect == "current_run" else "456", 1, get)
    assert found == ({"run": 123, "artifact": 9} if defect is None else {})


@pytest.mark.parametrize("operation", ["find", "restore"])
def test_unavailable_artifact_falls_back_to_render(package, tmp_path, monkeypatch, operation):
    _, identity, artifact = package
    identity_path = tmp_path / "identity.json"
    media.write_json(identity_path, identity)
    output = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("GITHUB_REPOSITORY_ID", "1")
    monkeypatch.setattr(media, "find_artifact", lambda *args: (_ for _ in ()).throw(OSError("API unavailable")))
    monkeypatch.setattr(media.sys, "argv", ["media_render_artifact.py", operation,
        "--identity", str(identity_path), "--package", str(artifact / "missing"), "--run", "123"])
    media.main()
    assert not output.exists()
