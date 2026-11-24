"""Verify the source and selected worker image before a manual render batch."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from functools import partial
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
import traceback


TRUSTED_ROOT = Path(__file__).resolve().parents[1]
POLICY = TRUSTED_ROOT / "scripts/render-kickoff.json"
POLICY_SCHEMA = POLICY.with_suffix(".schema.json")
SOURCE_SHA = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
UTC_TIMESTAMP = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")


def utc_now() -> datetime:
    return datetime.now(UTC)


def read_object(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def load_policy() -> dict:
    from jsonschema import Draft202012Validator

    policy = read_object(POLICY, "render kickoff policy")
    schema = read_object(POLICY_SCHEMA, "render kickoff policy schema")
    error = next(Draft202012Validator(schema).iter_errors(policy), None)
    if error is not None:
        field = ".".join(map(str, error.path))
        if error.validator in {"additionalProperties", "required"}:
            field = error.message
        raise ValueError(f"render kickoff policy has an invalid field ({field})")
    return policy


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


class RenderContentMismatch(ValueError):
    pass


def git(checkout: Path, *args: str, timeout: int, outside_repository: bool = False) -> str:
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
    env.update(
        GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
        GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0",
    )
    if outside_repository:
        env["GIT_CEILING_DIRECTORIES"] = str(checkout.resolve().parent)
    try:
        return subprocess.run(
            ["git", "--no-replace-objects", "-c", "core.fsmonitor=false", *args],
            cwd=checkout,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        ).stdout.strip()
    except (OSError, UnicodeError, subprocess.SubprocessError) as exc:
        if args[:2] == ("diff", "--quiet") and isinstance(exc, subprocess.CalledProcessError) and exc.returncode == 1:
            raise RenderContentMismatch("render source differs from public main; repin and rebuild") from exc
        raise ValueError(f"cannot verify render source with git {args[0]}") from exc


def verify_checkout(checkout: Path, source_head: str, source_tree: str, *, timeout: int) -> None:
    top, head, tree = git(
        checkout, "rev-parse", "--show-toplevel", "HEAD", "HEAD^{tree}", timeout=timeout
    ).splitlines()
    if Path(top).resolve() != checkout.resolve() or (head, tree) != (source_head, source_tree):
        raise ValueError("render checkout does not match the pinned source commit and tree")
    if git(checkout, "status", "--porcelain", "--untracked-files=all", timeout=timeout):
        raise ValueError("render kickoff requires a clean checkout")


def new_verdict(source_head: str = "", source_tree: str = "", image_id: str = "") -> dict:
    return {
        "schemaVersion": 1,
        "status": "failed",
        "sourceHead": source_head if SOURCE_SHA.fullmatch(source_head) else None,
        "sourceTree": source_tree if SOURCE_SHA.fullmatch(source_tree) else None,
        "imageId": image_id if IMAGE_ID.fullmatch(image_id) else None,
        "packageLockSha256": None,
        "mainHead": None,
        "checkedAt": None,
        "missingRendererCommit": None,
        "reason": None,
    }


def verify_kickoff(
    checkout: Path, source_head: str, source_tree: str, receipt_path: Path, image_id: str
) -> dict:
    verdict = new_verdict(source_head, source_tree, image_id)
    try:
        verdict["packageLockSha256"] = verify_worker_build(
            receipt_path, source_head, source_tree, image_id
        )
        policy = load_policy()
        timeouts = policy["gitTimeoutSeconds"]
        paths = [".", *(f":(exclude,top,literal){path}" for path in policy["excludedPaths"])]
        remote = policy["publicRemote"]
        main_ref = policy["mainRef"]
        local_git = partial(git, checkout, timeout=timeouts["local"])
        verify_checkout(checkout, source_head, source_tree, timeout=timeouts["local"])
        if local_git("rev-parse", "--is-shallow-repository") != "false":
            raise ValueError("render kickoff requires a complete Git history")
        checked_at = utc_now().isoformat(timespec="seconds").replace("+00:00", "Z")
        with TemporaryDirectory() as directory:
            main, separator, ref = git(
                Path(directory), "ls-remote", "--exit-code", remote, main_ref,
                timeout=timeouts["discovery"], outside_repository=True,
            ).partition("\t")
        if not separator or ref != main_ref or not SOURCE_SHA.fullmatch(main):
            raise ValueError("public main did not resolve to one commit")
        verdict["mainHead"] = main
        verdict["checkedAt"] = checked_at
        try:
            local_git("cat-file", "-e", f"{main}^{{commit}}")
        except ValueError:
            git(
                checkout, "fetch", "--no-tags", "--no-write-fetch-head", "--no-auto-maintenance",
                "--no-recurse-submodules", remote, main, timeout=timeouts["fetch"],
            )

        script = TRUSTED_ROOT / "scripts/render_kickoff.py"
        for path in (script, POLICY, POLICY_SCHEMA):
            relative = path.relative_to(TRUSTED_ROOT).as_posix()
            try:
                expected = local_git("rev-parse", "--verify", f"{main}:{relative}")
            except ValueError as exc:
                raise ValueError(f"public main does not contain kickoff authority {relative}") from exc
            if local_git("hash-object", "--no-filters", "--", str(path)) != expected:
                raise ValueError(f"kickoff authority {relative} differs from public main; update the kickoff checkout")

        if main != source_head:
            try:
                local_git(
                    "diff", "--quiet", "--no-ext-diff", "--no-textconv", f"{main}..{source_head}", "--", *paths,
                )
            except RenderContentMismatch:
                missing = local_git(
                    "log", "--full-history", "--diff-merges=first-parent",
                    "--format=%H", "--max-count=1", f"{source_head}..{main}", "--", *paths,
                )
                if missing:
                    verdict["missingRendererCommit"] = missing
                    raise ValueError("render source is missing a merged renderer commit; repin and rebuild") from None
                raise
            local_git("merge-base", source_head, main)
    except ValueError as exc:
        verdict["reason"] = str(exc)
    except (Exception, KeyboardInterrupt):
        traceback.print_exc()
        verdict["reason"] = "unexpected render kickoff error"
    else:
        verdict["status"] = "passed"
    return verdict


def require_kickoff_receipt(
    path: Path, source_head: str, source_tree: str, image_id: str, lock_digest: str, *, checkout: Path
) -> tuple[str, str]:
    verdict = read_object(path, "render kickoff receipt")
    main = verdict.get("mainHead")
    if (
        verdict.keys() != new_verdict().keys()
        or type(verdict.get("schemaVersion")) is not int
        or verdict["schemaVersion"] != 1
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
    checked_at = verdict["checkedAt"]
    if not isinstance(checked_at, str) or not UTC_TIMESTAMP.fullmatch(checked_at):
        raise ValueError("public render kickoff receipt requires a UTC checkedAt timestamp")
    try:
        age = (utc_now() - datetime.fromisoformat(checked_at)).total_seconds()
    except ValueError as exc:
        raise ValueError("public render kickoff receipt requires a valid checkedAt timestamp") from exc
    if age < 0:
        raise ValueError("public render kickoff receipt is future-dated")
    policy = load_policy()
    if age > policy["maxReceiptAgeSeconds"]:
        raise ValueError("public render kickoff receipt has expired; rerun kickoff")
    verify_checkout(checkout, source_head, source_tree, timeout=policy["gitTimeoutSeconds"]["local"])
    return main, checked_at


class VerdictParser(argparse.ArgumentParser):
    def print_help(self, file=None):
        super().print_help(file or sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = VerdictParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--build-receipt", type=Path, required=True)
    parser.add_argument("--image-id", required=True)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        verdict = new_verdict()
        if exc.code == 0:
            verdict["status"] = "help"
        else:
            verdict["reason"] = "invalid CLI usage"
        print(json.dumps(verdict, sort_keys=True))
        return exc.code
    verdict = verify_kickoff(args.checkout, args.source_head, args.source_tree, args.build_receipt, args.image_id)
    print(json.dumps(verdict, sort_keys=True))
    return 0 if verdict["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
