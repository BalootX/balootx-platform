#!/usr/bin/env python3
"""Promote validated CI media only for the exact source and installed toolchain."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
# Conservative closure: Python imports/configuration can affect bundle generation.
INPUTS = ("src", "scripts", "assets", "configs", "examples", "generated", "plugins",
          "schemas", "native", "pyproject.toml", "requirements", ".gitattributes", ".github/workflows/ci.yml")
OUTPUTS = tuple(f"outputs/{name}" for name in (
    "professional-video-flagship", "professional-video-decision",
    "professional-video-prototypes", "remotion-plugin-parity", "scene-preview",
))
REMOTION = ROOT / "src/baloot_platform/video/remotion"
PREFIX = "media-render-v1-"


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def command(args: list[str], cwd: Path = ROOT) -> str:
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True,
                          text=True, encoding="utf-8", timeout=180).stdout.strip()


def source_identity(root: Path = ROOT, revision: str = "HEAD") -> str:
    tree = command(["git", "ls-tree", "-r", revision, "--", *INPUTS], root)
    if not tree:
        raise ValueError("media source tree is unknown")
    return hashlib.sha256(tree.encode()).hexdigest()


def toolchain() -> dict:
    if sys.platform != "linux":
        raise ValueError("promotion requires the Linux media toolchain")
    # Ensure the same default browser used by every render, without launching it.
    command(["node", "--input-type=module", "-e",
             "import {ensureBrowser} from '@remotion/renderer'; await ensureBrowser();"], REMOTION)
    browser = REMOTION / "node_modules/.remotion"
    browser_files = {p.relative_to(browser).as_posix(): sha(p)
                     for p in sorted(browser.rglob("*")) if p.is_file()}
    if not any(name.endswith("/chrome-headless-shell") for name in browser_files):
        raise ValueError("Remotion browser identity is unknown")
    binaries = {}
    for name in ("node", "ffmpeg", "ffprobe"):
        path = shutil.which(name)
        if not path:
            raise ValueError(f"{name} identity is unknown")
        binaries[name] = sha(Path(path))
    return {
        "python": sys.version,
        "packages": sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions()),
        "node": json.loads(command(["node", "-p", "JSON.stringify(process.versions)"])),
        "platform": [platform.machine(), platform.libc_ver(), Path("/etc/os-release").read_text()],
        "systemPackages": command(["dpkg-query", "-W", "-f=${binary:Package}=${Version}\n"]),
        "binaries": binaries,
        "browser": browser_files,
        "sourceDateEpoch": os.environ["SOURCE_DATE_EPOCH"],
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value) + b"\n")


def output(**values: object) -> None:
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
            for key, value in values.items():
                stream.write(f"{key}={value}\n")


def identify(destination: Path) -> None:
    value = {}
    try:
        command(["git", "diff", "--quiet", "HEAD", "--", *INPUTS])
        if command(["git", "ls-files", "--others", "--exclude-standard", "--", *INPUTS]):
            raise ValueError("untracked media inputs")
        inputs = {"source": source_identity(), "toolchain": toolchain()}
        value = {"inputs": inputs, "identity": hashlib.sha256(canonical(inputs)).hexdigest()}
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print(f"::notice::Unknown media identity; rendering required ({type(exc).__name__})")
    write_json(destination, value)
    output(identity=value.get("identity", ""))


def api(path: str) -> dict:
    request = urllib.request.Request(
        f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}/{path}",
        headers={"Authorization": f"Bearer {os.environ['GH_TOKEN']}",
                 "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def find_artifact(identity: str, current_run: str, repository_id: int, get=api) -> dict:
    if not identity:
        return {}
    artifacts = get("actions/artifacts?" + urllib.parse.urlencode({"name": PREFIX + identity, "per_page": 100}))
    # A bounded search miss costs a render, never a validation bypass.
    for artifact in artifacts["artifacts"][:20]:
        run_id = artifact["workflow_run"]["id"]
        if artifact["expired"] or str(run_id) == current_run:
            continue
        run = get(f"actions/runs/{run_id}")
        if (run["path"] == ".github/workflows/ci.yml" and run["status"] == "completed"
                and run["conclusion"] == "success" and run["event"] in {"pull_request", "merge_group", "push"}
                and run["head_repository"]["id"] == repository_id):
            return {"run": run_id, "artifact": artifact["id"]}
    return {}


def files(root: Path) -> list[Path]:
    result = []
    for relative in OUTPUTS:
        directory = root / relative
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError(f"missing media output: {relative}")
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError("media output contains a symlink")
            if path.is_file():
                result.append(path)
    if not result:
        raise ValueError("empty media output")
    return result


def seal(identity: dict, destination: Path, root: Path = ROOT) -> None:
    if not identity.get("identity"):
        raise ValueError("cannot publish unknown media identity")
    destination.mkdir(parents=True, exist_ok=True)
    records = {}
    with tarfile.open(destination / "media.tar", "w") as archive:
        for path in files(root):
            name = path.relative_to(root).as_posix()
            records[name] = {"sha256": sha(path), "bytes": path.stat().st_size}
            archive.add(path, arcname=name, recursive=False)
    manifest = {"schemaVersion": 1, **identity, "files": records,
                "payloadSha256": sha(destination / "media.tar"),
                "producer": {"run": os.environ["GITHUB_RUN_ID"],
                             "attempt": os.environ["GITHUB_RUN_ATTEMPT"],
                             "repository": os.environ["GITHUB_REPOSITORY"],
                             "commit": command(["git", "rev-parse", "HEAD"], root)}}
    write_json(destination / "manifest.json", manifest)
    (destination / "manifest.sha256").write_text(sha(destination / "manifest.json") + "\n", encoding="utf-8")


def restore(identity: dict, package: Path, run_id: str, root: Path = ROOT) -> None:
    manifest_path = package / "manifest.json"
    if sha(manifest_path) != (package / "manifest.sha256").read_text().strip():
        raise ValueError("media manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_bytes())
    if (not identity.get("identity") or manifest["schemaVersion"] != 1
            or any(manifest[key] != identity[key] for key in ("identity", "inputs"))
            or hashlib.sha256(canonical(identity["inputs"])).hexdigest() != identity["identity"]
            or manifest["producer"]["run"] != run_id
            or manifest["producer"]["repository"] != os.environ["GITHUB_REPOSITORY"]):
        raise ValueError("media input or producer identity mismatch")
    if sha(package / "media.tar") != manifest["payloadSha256"]:
        raise ValueError("media payload SHA256 mismatch")
    with tempfile.TemporaryDirectory(prefix="media-promote-") as temporary:
        staged = Path(temporary)
        with tarfile.open(package / "media.tar") as archive:
            members = archive.getmembers()
            names = [m.name for m in members]
            if not names or len(set(names)) != len(names) or set(names) != set(manifest["files"]):
                raise ValueError("media payload file set mismatch")
            for member in members:
                path = PurePosixPath(member.name)
                if (not member.isfile() or path.is_absolute() or ".." in path.parts
                        or "\\" in member.name or ":" in member.name
                        or path.as_posix() != member.name
                        or not any(member.name.startswith(prefix + "/") for prefix in OUTPUTS)):
                    raise ValueError("unsafe media payload path")
                record = manifest["files"][member.name]
                if member.size != record["bytes"]:
                    raise ValueError("media file size mismatch")
                target = staged / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("wb") as stream:
                    shutil.copyfileobj(source, stream)
                if sha(target) != record["sha256"]:
                    raise ValueError("media file SHA256 mismatch")
        # Validate the complete package before replacing any checked-out outputs.
        files(staged)
        for relative in OUTPUTS:
            target = root / relative
            if target.is_symlink() or (target.exists() and not target.is_dir()):
                raise ValueError("unsafe media output destination")
        for relative in OUTPUTS:
            target = root / relative
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(staged / relative, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("identity", "find", "seal", "restore"))
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--run", default="")
    args = parser.parse_args()
    if args.operation == "identity":
        identify(args.identity)
        return
    identity = json.loads(args.identity.read_bytes())
    if args.operation == "seal":
        seal(identity, args.package)
        return
    try:
        if args.operation == "find":
            found = find_artifact(identity.get("identity", ""), os.environ["GITHUB_RUN_ID"],
                                  int(os.environ["GITHUB_REPOSITORY_ID"]))
            output(**found)
        else:
            restore(identity, args.package, args.run)
            output(reused="true")
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError) as exc:
        print(f"::notice::Media promotion unavailable; rendering required ({type(exc).__name__})")


if __name__ == "__main__":
    main()
