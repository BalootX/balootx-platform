#!/usr/bin/env python3
# ruff: noqa: E402
"""Render and verify both governed professional video exports."""

from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
import secrets
import subprocess
import tarfile
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baloot_platform.video.audio_postprocess import finish_remotion_audio
from baloot_platform.video.remotion_renderer import _process_exists, _terminate_process_tree
from baloot_platform.video.render_bundle import RenderBundleV1

try:
    from .generate_professional_video_prototypes import generate
except ImportError:
    from generate_professional_video_prototypes import generate


REMOTION = ROOT / "src/baloot_platform/video/remotion"
DEFAULT_RENDER_LOCK = ROOT / ".local-state/render-release.lock"


@contextmanager
def release_render_lock(path: Path = DEFAULT_RENDER_LOCK) -> Iterator[None]:
    """Prevent two release renders from writing the same governed outputs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = {
        "pid": os.getpid(),
        "lockId": secrets.token_hex(16),
        "createdUtc": datetime.now(timezone.utc).isoformat(),
    }
    encoded = (json.dumps(identity, sort_keys=True) + "\n").encode("utf-8")

    for _attempt in range(3):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                owner_pid = int(existing.get("pid", 0))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                owner_pid = 0
            if _process_exists(owner_pid):
                raise RuntimeError(
                    f"release render already active under process {owner_pid}"
                )
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            continue
        else:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
            break
    else:
        raise RuntimeError("could not acquire the release render lock")

    try:
        yield
    finally:
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
            if current.get("lockId") == identity["lockId"]:
                path.unlink()
        except FileNotFoundError:
            pass


def _node_command(script: str) -> list[str]:
    if os.name != "nt":
        return ["node", f"scripts/{script}"]
    wrapper = Path(os.environ.get("BALOOT_HIDDEN_NODE_WRAPPER", ""))
    if not wrapper.is_file():
        raise RuntimeError("the required hidden Node command wrapper is unavailable")
    command = base64.b64encode(f"node scripts/{script}".encode("utf-8")).decode("ascii")
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
        command,
        "-WorkingDirectory",
        str(REMOTION),
    ]


def render_timeout_seconds() -> int:
    raw = os.environ.get("BALOOT_RENDER_TIMEOUT_SECONDS", "1200")
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            "BALOOT_RENDER_TIMEOUT_SECONDS must be an integer from 300 to 3600"
        ) from exc
    if not 300 <= value <= 3600:
        raise RuntimeError(
            "BALOOT_RENDER_TIMEOUT_SECONDS must be an integer from 300 to 3600"
        )
    return value


def _run(
    command: list[str],
    *,
    cwd: Path = ROOT,
    timeout: int | None = None,
    environment: dict[str, str] | None = None,
    stage: str | None = None,
    evidence_dir: Path | None = None,
) -> None:
    resolved_timeout = render_timeout_seconds() if timeout is None else timeout
    stage_name = stage or Path(command[0]).stem
    if not stage_name or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789-_"
        for character in stage_name.lower()
    ):
        raise ValueError(f"invalid release render stage name: {stage_name!r}")
    if evidence_dir is None:
        evidence_dir = _new_render_evidence_dir()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = evidence_dir / f"{stage_name}.stdout.log"
    stderr_path = evidence_dir / f"{stage_name}.stderr.log"
    receipt_path = evidence_dir / f"{stage_name}.receipt.json"
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    with (
        stdout_path.open("w", encoding="utf-8", newline="\n") as stdout_file,
        stderr_path.open("w", encoding="utf-8", newline="\n") as stderr_file,
    ):
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=None if environment is None else {**os.environ, **environment},
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            encoding="utf-8",
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            start_new_session=os.name != "nt",
        )
        try:
            return_code = process.wait(timeout=resolved_timeout)
            status = "passed" if return_code == 0 else "failed"
        except subprocess.TimeoutExpired:
            _terminate_process_tree(process)
            return_code = process.poll()
            status = "timed_out"
    stdout = stdout_path.read_text(encoding="utf-8")
    stderr = stderr_path.read_text(encoding="utf-8")
    receipt = {
        "schemaVersion": 1,
        "stage": stage_name,
        "status": status,
        "command": command,
        "cwd": str(cwd.resolve()),
        "timeoutSeconds": resolved_timeout,
        "startedUtc": started_utc,
        "finishedUtc": datetime.now(timezone.utc).isoformat(),
        "durationSeconds": time.monotonic() - started,
        "exitCode": return_code,
        "stdout": {
            "path": str(stdout_path.resolve()),
            "bytes": stdout_path.stat().st_size,
            "sha256": hashlib.sha256(stdout_path.read_bytes()).hexdigest(),
        },
        "stderr": {
            "path": str(stderr_path.resolve()),
            "bytes": stderr_path.stat().st_size,
            "sha256": hashlib.sha256(stderr_path.read_bytes()).hexdigest(),
        },
    }
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if status == "timed_out":
        raise subprocess.TimeoutExpired(
            command,
            resolved_timeout,
            output=stdout[-4000:],
            stderr=stderr[-4000:],
        )
    if return_code:
        detail = (stderr or stdout or "child emitted no diagnostics").strip()[-4000:]
        raise RuntimeError(
            f"release render stage {stage_name!r} failed with exit code "
            f"{return_code}; stdout={stdout_path}; stderr={stderr_path}; "
            f"receipt={receipt_path}; tail={detail}"
        )


def _new_render_evidence_dir() -> Path:
    configured = os.environ.get("BALOOT_RENDER_EVIDENCE_DIR", "").strip()
    if configured:
        destination = Path(configured).resolve()
        destination.mkdir(parents=True, exist_ok=False)
        return destination
    token = secrets.token_hex(8)
    destination = Path(tempfile.gettempdir()) / f"balootx-render-release-{token}"
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def finish_release_audio(shard: str | None = None) -> None:
    exports = (
        (
            ROOT
            / "outputs/professional-video-flagship/balootx-complete-laffah-audit.mp4",
            ROOT
            / "outputs/professional-video-prototypes/render-bundle-hooked_flagship.json",
            ROOT
            / "outputs/professional-video-flagship/balootx-complete-laffah-audit.audio.json",
        ),
        (
            ROOT / "outputs/professional-video-decision/balootx-decision-clip.mp4",
            ROOT / "outputs/professional-video-prototypes/render-bundle-decision.json",
            ROOT
            / "outputs/professional-video-decision/balootx-decision-clip.audio.json",
        ),
    )
    for video, bundle_path, evidence_path in exports:
        if shard is not None and video.parent.name != f"professional-video-{shard}":
            continue
        bundle = RenderBundleV1(json.loads(bundle_path.read_text(encoding="utf-8")))
        finish_remotion_audio(
            video,
            bundle,
            repository_root=ROOT,
            evidence_path=evidence_path,
        )


CI_SHARD_OUTPUTS = {
    "flagship": ("professional-video-flagship",),
    "decision": ("professional-video-decision",),
    "companion": ("professional-video-prototypes", "professional-video-collection"),
    "parity": ("remotion-plugin-parity",),
}


def render_ci_shard(shard: str) -> None:
    """Render one independent lane from the supplied CI input bundle artifact."""
    evidence_dir = _new_render_evidence_dir()
    outputs = ROOT / "outputs"

    def run(stage: str, command: list[str], **kwargs) -> None:
        _run(command, stage=stage, evidence_dir=evidence_dir, **kwargs)

    if shard in ("flagship", "decision"):
        script = "render-flagship.mjs" if shard == "flagship" else "render-decision-clip.mjs"
        run(shard, _node_command(script), cwd=REMOTION)
        finish_release_audio(shard)
        verify = "verify_professional_flagship" if shard == "flagship" else "verify_professional_decision_clip"
        run(verify, [sys.executable, f"scripts/{verify}.py"])
    elif shard == "companion":
        run("prototypes", _node_command("render-prototypes.mjs"), cwd=REMOTION)
        run("collection", _node_command("render-flagship.mjs"), cwd=REMOTION,
            environment={"BALOOTX_COLLECTION_ONLY": "1",
                         "BALOOTX_RENDER_OUTPUT": str(outputs / "professional-video-collection")})
        run("verify-professional-prototypes", [sys.executable, "scripts/verify_professional_video_prototypes.py"])
    elif shard == "parity":
        run("render-plugin-parity", [sys.executable, "scripts/render_remotion_plugin_parity.py"],
            timeout=render_timeout_seconds() + 60)
    else:
        raise ValueError(f"unknown media CI shard: {shard}")

    epoch = int(os.environ["SOURCE_DATE_EPOCH"])

    def reproducible_member(member: tarfile.TarInfo) -> tarfile.TarInfo:
        return member.replace(
            mtime=epoch, uid=0, gid=0, uname="", gname="", deep=False,
            mode=0o755 if member.isdir() else 0o644,
        )

    with tarfile.open(Path(os.environ["RUNNER_TEMP"]) / "media-shard.tar", "w") as archive:
        for directory in CI_SHARD_OUTPUTS[shard]:
            archive.add(
                outputs / directory, arcname=f"outputs/{directory}", filter=reproducible_member,
            )


def main(*, research_notice: bool = False) -> None:
    evidence_dir = _new_render_evidence_dir()
    print(evidence_dir)

    def run(stage: str, command: list[str], **kwargs) -> None:
        _run(
            command,
            stage=stage,
            evidence_dir=evidence_dir,
            **kwargs,
        )

    with release_render_lock():
        if research_notice:
            command = [sys.executable, "-B", "-m", "scripts.render_research_notice"]
            if (ROOT / ".git").exists():
                command += [
                    "--source-head",
                    subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                    ).strip(),
                    "--source-tree",
                    subprocess.check_output(
                        ["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, text=True
                    ).strip(),
                ]
            run("research-notice-4", command)
            return
        generate()
        run("prototypes", _node_command("render-prototypes.mjs"), cwd=REMOTION)
        run("flagship", _node_command("render-flagship.mjs"), cwd=REMOTION)
        run("decision", _node_command("render-decision-clip.mjs"), cwd=REMOTION)
        run(
            "collection",
            _node_command("render-flagship.mjs"),
            cwd=REMOTION,
            environment={"BALOOTX_COLLECTION_ONLY": "1"},
        )
        finish_release_audio()
        for script in (
            "verify_professional_video_prototypes.py",
            "verify_professional_flagship.py",
            "verify_professional_decision_clip.py",
        ):
            run(Path(script).stem, [sys.executable, f"scripts/{script}"])
        run(
            "render-plugin-parity",
            [sys.executable, "scripts/render_remotion_plugin_parity.py"],
            timeout=render_timeout_seconds() + 60,
        )
        run(
            "verify-plugin-parity",
            [sys.executable, "scripts/verify_remotion_plugin_parity.py"],
        )
        run(
            "scene-preview",
            [sys.executable, "scripts/generate_scene_preview.py", "--refresh"],
        )
        print(ROOT / "outputs/professional-video-flagship/manifest.json")
        print(ROOT / "outputs/professional-video-decision/manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render and verify the governed professional media set."
    )
    parser.add_argument("--research-notice", action="store_true", help="Render only the first research notice and its critical-position thumbnail.")
    parser.add_argument("--ci-shard", choices=CI_SHARD_OUTPUTS)
    args = parser.parse_args()
    if args.ci_shard:
        render_ci_shard(args.ci_shard)
    else:
        main(research_notice=args.research_notice)
