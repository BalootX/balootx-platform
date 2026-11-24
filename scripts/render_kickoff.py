"""Verify the source and selected worker image before a manual render batch."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess


POLICY = Path(__file__).with_name("render-kickoff.json")
MAIN_REF = "refs/heads/main"
SOURCE_SHA = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")


def read_object(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def verify_worker_build(
    receipt_path: Path, source_head: str, source_tree: str, image_id: str
) -> str:
    if not all(SOURCE_SHA.fullmatch(value) for value in (source_head, source_tree)):
        raise ValueError("public render requires full source commit and tree IDs")
    if not IMAGE_ID.fullmatch(image_id):
        raise ValueError("public render requires the executing image ID")
    build = read_object(receipt_path, "public render build receipt")
    if build.get("status") != "PASSED" or build.get("target") != "video-worker":
        raise ValueError("public render requires a passed video-worker build receipt")
    if (build.get("sourceHead"), build.get("sourceTree")) != (source_head, source_tree):
        raise ValueError("public render build source does not match the selected source")
    if build.get("imageId") != image_id:
        raise ValueError("public render image does not match the selected build")
    lock_digest = build.get("packageLockSha256")
    if not isinstance(lock_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", lock_digest):
        raise ValueError("public render build receipt requires a package lock digest")
    return lock_digest


def git(checkout: Path, *args: str, timeout: int = 10) -> str:
    try:
        return subprocess.run(
            ["git", "--no-replace-objects", "--literal-pathspecs", *args],
            cwd=checkout,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        ).stdout.strip()
    except (OSError, UnicodeError, subprocess.SubprocessError) as exc:
        raise ValueError(f"cannot verify render source with git {args[0]}") from exc


def verify_kickoff(
    checkout: Path, source_head: str, source_tree: str, receipt_path: Path, image_id: str
) -> dict:
    verdict = {
        "schemaVersion": 1,
        "status": "failed",
        "sourceHead": source_head if SOURCE_SHA.fullmatch(source_head) else None,
        "sourceTree": source_tree if SOURCE_SHA.fullmatch(source_tree) else None,
        "imageId": image_id if IMAGE_ID.fullmatch(image_id) else None,
        "packageLockSha256": None,
        "mainHead": None,
        "missingRendererCommit": None,
        "reason": None,
    }
    try:
        verdict["packageLockSha256"] = verify_worker_build(
            receipt_path, source_head, source_tree, image_id
        )
        top, head, tree = git(
            checkout, "rev-parse", "--show-toplevel", "HEAD", "HEAD^{tree}"
        ).splitlines()
        if Path(top).resolve() != checkout.resolve() or (head, tree) != (source_head, source_tree):
            raise ValueError("render checkout does not match the pinned source commit and tree")
        if git(checkout, "rev-parse", "--is-shallow-repository") != "false":
            raise ValueError("render kickoff requires a complete Git history")
        if git(checkout, "status", "--porcelain", "--untracked-files=no"):
            raise ValueError("render kickoff requires clean tracked source inputs")

        policy = read_object(POLICY, "render kickoff policy")
        remote = policy.get("publicRemote")
        paths = policy.get("rendererPaths")
        if not isinstance(remote, str) or not remote or remote.startswith("-"):
            raise ValueError("render kickoff policy requires a public remote")
        if not isinstance(paths, list) or not paths or any(
            not isinstance(path, str)
            or not re.fullmatch(r"[A-Za-z0-9_./-]+", path)
            or PurePosixPath(path).as_posix() != path
            or PurePosixPath(path).is_absolute()
            or path == "."
            or ".." in PurePosixPath(path).parts
            for path in paths
        ):
            raise ValueError("render kickoff policy requires repository-relative renderer paths")

        main, separator, ref = git(
            checkout, "ls-remote", "--exit-code", remote, MAIN_REF, timeout=30
        ).partition("\t")
        if not separator or ref != MAIN_REF or not SOURCE_SHA.fullmatch(main):
            raise ValueError("public main did not resolve to one commit")
        verdict["mainHead"] = main
        if main != source_head:
            try:
                git(checkout, "cat-file", "-e", f"{main}^{{commit}}")
            except ValueError:
                git(
                    checkout, "fetch", "--no-tags", "--no-write-fetch-head", "--no-auto-maintenance",
                    "--no-recurse-submodules", remote, main, timeout=60,
                )
            git(checkout, "merge-base", source_head, main)
            missing = git(
                checkout, "log", "--full-history", "--diff-merges=first-parent",
                "--format=%H", "--max-count=1", f"{source_head}..{main}", "--", *paths,
            )
            if missing:
                verdict["missingRendererCommit"] = missing
                raise ValueError("render source is missing a merged renderer commit; repin and rebuild")
    except ValueError as exc:
        verdict["reason"] = str(exc)
    else:
        verdict["status"] = "passed"
    return verdict


def require_kickoff_receipt(
    path: Path, source_head: str, source_tree: str, image_id: str, lock_digest: str
) -> str:
    verdict = read_object(path, "render kickoff receipt")
    main = verdict.get("mainHead")
    if (
        verdict.get("schemaVersion") != 1
        or verdict.get("status") != "passed"
        or not isinstance(main, str)
        or not SOURCE_SHA.fullmatch(main)
        or verdict.get("missingRendererCommit") is not None
        or verdict.get("reason") is not None
    ):
        raise ValueError("public render requires a passed render kickoff receipt")
    keys = ("sourceHead", "sourceTree", "imageId", "packageLockSha256")
    if tuple(verdict.get(key) for key in keys) != (source_head, source_tree, image_id, lock_digest):
        raise ValueError("public render kickoff receipt does not match the selected source and image")
    return main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--build-receipt", type=Path, required=True)
    parser.add_argument("--image-id", required=True)
    args = parser.parse_args(argv)
    verdict = verify_kickoff(
        args.checkout, args.source_head, args.source_tree, args.build_receipt, args.image_id
    )
    print(json.dumps(verdict, sort_keys=True))
    return 0 if verdict["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
