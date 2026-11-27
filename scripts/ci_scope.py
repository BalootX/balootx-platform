"""Select CI work and require exact job and shard coverage for that selection."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ci.critical_review import pull_request_merge
from scripts.run_heavy_pytest import HEAVY_SHARDS, validate_union
from scripts.run_isolated_pytest import fast_files, node_sharded_files, validate_shards

HISTORY = Path(".github/python-test-durations.json")
VERSIONS = ("3.12", "3.13")
FULL = {
    "python-wheelhouse", "ffmpeg-packages", "native-build", "native-sanitizers", "python-fast-run",
    "python-backend-run", "python-heavy", "python-windows", "python-coverage", "media-inputs",
    "media-render", "media-release-smoke", "identity-postgres", "postgres-configuration",
}
SCOPED = {
    "sdks": "sdk", "foundry": "foundry", "remotion-checks": "remotion", "remotion-image-parity": "video_worker",
    "docker-constraints": "docker_constraints",
}
DOCKER_CONSTRAINT_INPUTS = {
    "Dockerfile", ".dockerignore", "Dockerfile.dockerignore", "pyproject.toml", "requirements-docker.txt",
    ".github/workflows/ci.yml", "scripts/ci_scope.py",
}
TEST_SETUP = {"ffmpeg-packages", "native-build"}
FOUNDRY = "src/baloot_platform/service/foundry_dashboard/src/"
FOUNDRY_TESTS = {
    "tests/test_configuration_draft_governance.py",
    "tests/test_foundry_dashboard.py",
    "tests/test_localization_usage_manifest.py",
}
SDK_TESTS = {
    "tests/test_foundry_catalog_control.py",
    "tests/test_research_operator_audit_sdk_contracts.py",
    "tests/test_v2_research_sdk_contracts.py",
    "tests/test_v18_hitl_governance.py",
    "tests/test_v20_protocol_ui_performance.py",
}
CI_TESTS = {
    "tests/test_heavy_pytest.py", "tests/test_isolated_pytest.py",
    "tests/test_repository_structure.py", "tests/test_media_render_artifact.py",
    "tests/test_remotion_image_parity.py",
    "tests/test_combined_finalist_policy_pilot.py", "tests/test_local_cpu_launcher.py",
    "tests/test_v18_hitl_governance.py", "tests/test_v20_asan_verifier_portability.py", "tests/test_verifier_preflight.py",
    "tests/test_v8_identity_sakka.py",
}


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, timeout=10).stdout


def changed_paths(base: str, head: str) -> set[str]:
    """Read the exact PR merge delta, including both sides of renames."""
    if not all(re.fullmatch(r"[0-9a-f]{40}", sha) for sha in (base, head)):
        raise ValueError("PR event commit identity is missing")
    checkout, merge_base = pull_request_merge(git, head)
    if merge_base != base:
        raise ValueError("checkout does not match both PR event parents")
    fields = git("diff", "--raw", "-z", "--no-renames", "--no-ext-diff", base, checkout, "--").split(b"\0")
    if fields[-1] != b"" or (len(fields) - 1) % 2:
        raise ValueError("malformed PR diff")
    paths = set()
    for header, path in zip(fields[:-1:2], fields[1:-1:2], strict=True):
        old_mode, new_mode, _, _, status = header.split()
        if (new_mode != b"100644" or (old_mode, status) not in {(b":000000", b"A"), (b":100644", b"M")}
                or not re.fullmatch(rb"[a-zA-Z0-9_./-]+", path)):
            raise ValueError("PR diff includes a removal, mode change or unsupported path")
        paths.add(path.decode("ascii"))
    return paths


def has_repository_reference(documents: set[str], changed_tests: set[str]) -> bool:
    # Basenames also catch split paths; test module names catch imports and node IDs.
    names = {Path(path).name for path in documents} | {Path(path).stem for path in changed_tests}
    if not names:
        return False
    args = [arg for name in sorted(names) for arg in ("-e", name)]
    excluded = [f":(exclude,literal){path}" for path in sorted(changed_tests)]
    try:
        git("grep", "--cached", "--quiet", "--text", "--fixed-strings", "--word-regexp",
            *args, "--", "tests", "src", "configs", *excluded)
    except subprocess.CalledProcessError as exc:
        if exc.returncode != 1 or exc.stderr:
            raise
        return False
    return True


def scoped_plan(paths: set[str]) -> dict | None:
    if not paths:
        return None
    sdk = foundry = False
    documents = set()
    changed_tests = set()
    tests = set()
    for path in paths:
        if path in {"README.md", "SECURITY.md"} or (path.startswith("docs/") and path.endswith(".md")):
            documents.add(path)
        elif path.startswith(FOUNDRY) and path.endswith((".ts", ".tsx", ".css")):
            foundry = True
            tests.update(FOUNDRY_TESTS)
        elif ((path.startswith("sdk/go/") and path.endswith(".go"))
              or (path.startswith("sdk/typescript/src/") and path.endswith(".ts"))):
            sdk = True
            tests.update(SDK_TESTS)
        elif (re.fullmatch(r"tests/test_[a-zA-Z0-9_]+\.py", path)
              and path not in CI_TESTS and not path.startswith("tests/test_ci_")):
            changed_tests.add(path)
        else:
            return None
    if has_repository_reference(documents, changed_tests):
        return None
    tests.update(changed_tests)
    if tests:
        history = json.loads((ROOT / HISTORY).read_bytes())
        seconds = history["shardSeconds"]
        if (not tests <= set(fast_files(history)) or not tests <= seconds.keys()
                or sum(seconds[path] for path in tests) > 600):
            return None
    # Foundry UI is the scoped change class included in the worker's COPY src.
    return {
        "full": False, "sdk": sdk, "foundry": foundry, "remotion": False, "video_worker": foundry,
        "docker_constraints": False, "tests": sorted(tests),
    }


def current_plan(event: str = "", base: str = "", head: str = "") -> dict:
    """Only pull requests can narrow work. A known diff scopes Docker constraint resolution independently."""
    full = {"full": True, **dict.fromkeys(SCOPED.values(), True), "tests": []}
    if event == "pull_request":
        try:
            paths = changed_paths(base, head)
            resolve_docker = not paths or bool(paths & DOCKER_CONSTRAINT_INPUTS)
            return {**(scoped_plan(paths) or full), "docker_constraints": resolve_docker}
        except (OSError, ValueError, subprocess.SubprocessError, KeyError, TypeError):
            print("CI scope could not be established; running the full profile.", file=sys.stderr)
    return full


def validate_results(event: str, results: dict) -> None:
    """Require success from selected producers and an explicit skip from every other producer."""
    plan = results.get("plan", {})
    selection = plan.get("outputs")
    if plan.get("result") != "success" or not isinstance(selection, dict):
        raise ValueError("CI plan did not succeed or its outputs are unavailable")
    for key in ("full", *SCOPED.values()):
        if selection.get(key) not in ("true", "false"):
            raise ValueError(f"invalid CI plan output: {key}")
    tests = json.loads(selection["tests"])
    if not isinstance(tests, list) or any(not isinstance(path, str) for path in tests):
        raise ValueError("invalid CI plan test list")
    full = selection["full"] == "true"
    expected = {
        "plan": "success",
        "preflight": "success",
        "commit-screen": "skipped" if event == "workflow_dispatch" else "success",
        "dependency-review": "success" if event == "pull_request" else "skipped",
        "python-version-smoke": "success" if event == "pull_request" else "skipped",
        "python-affected": "success" if tests else "skipped",
        **{job: "success" if selection[key] == "true" else "skipped" for job, key in SCOPED.items()},
        **{job: "success" if full or (tests and job in TEST_SETUP) else "skipped" for job in FULL},
    }
    if not full and event != "pull_request":
        raise ValueError("full CI plan is required for this event")
    full_keys = set(SCOPED.values())
    if event == "pull_request":
        full_keys.remove("docker_constraints")
    if full and (tests or any(selection[key] != "true" for key in full_keys)):
        raise ValueError("full CI plan must select all full producers")
    actual = {job: details["result"] for job, details in results.items()}
    if actual != expected:
        raise ValueError(f"CI producers differ from the plan: expected={expected}, actual={actual}")


def validate_coverage(receipts: Path, head: str, history: dict, event: str) -> dict:
    """Verify every shard for the event's Python versions."""
    files = fast_files(history)
    coverage = {}
    versions = ("3.13",) if event == "pull_request" else VERSIONS
    for pattern, filename in (("python-fast-run", "python-fast-status.json"), ("python-backend-run", "receipt.json"),
                              ("python-heavy", "receipt.json")):
        prefix = f"{pattern}-"
        if any(path.parent.name.removeprefix(prefix).split("-", 1)[0] not in versions
               for path in receipts.glob(f"{pattern}-*/{filename}")):
            raise ValueError("unexpected Python version in shard receipts")
    for version in versions:
        fast = [json.loads(path.read_bytes()) for path in receipts.glob(
            f"python-fast-run-{version}-*/python-fast-status.json")]
        validate_shards(fast, head, version, files, history)
        backend = [json.loads(path.read_bytes()) for path in
                   receipts.glob(f"python-backend-run-{version}-*/receipt.json")]
        coverage[f"backend-{version}"] = validate_union(
            backend, head, sorted(history["versionedNodeShardedFiles"]),
            python=version, seconds=history["nodeSeconds"][version],
            shards=history["versionedNodeShards"],
        )
        heavy = [json.loads(path.read_bytes()) for path in receipts.glob(f"python-heavy-{version}-*/receipt.json")]
        coverage[f"heavy-{version}"] = validate_union(
            heavy, head, sorted(node_sharded_files(history)), python=version,
            seconds=history["heavyNodeSeconds"], shards=HEAVY_SHARDS,
        )
    return coverage


def main() -> None:
    mode = sys.argv[1]
    if mode == "plan":
        event = os.environ["GITHUB_EVENT_NAME"]
        if event == "merge_group":
            group = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_bytes())["merge_group"]
            checked_out = git("rev-parse", "HEAD").decode("ascii").strip()
            if (group["head_ref"] != os.environ["GITHUB_REF"] or group["head_sha"] != os.environ["GITHUB_SHA"]
                    or group["head_sha"] != checked_out):
                raise ValueError("checkout or workflow context does not match the merge group head")
        outputs = current_plan(event, os.getenv("PR_BASE", ""), os.getenv("PR_HEAD", ""))
        with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as stream:
            for key, value in outputs.items():
                print(f"{key}={json.dumps(value, ensure_ascii=True, separators=(',', ':'))}", file=stream)
    elif mode == "gate":
        validate_results(os.environ["GITHUB_EVENT_NAME"], json.loads(os.environ["CI_RESULTS"]))
    elif mode == "coverage":
        coverage = validate_coverage(Path(sys.argv[2]), os.environ["GITHUB_SHA"],
                                     json.loads(HISTORY.read_bytes()), os.environ["GITHUB_EVENT_NAME"])
        print(json.dumps(coverage, indent=2))
    else:
        raise ValueError(f"unknown CI mode: {mode}")


if __name__ == "__main__":
    main()
