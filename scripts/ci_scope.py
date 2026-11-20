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
from scripts.run_heavy_pytest import validate_union
from scripts.run_isolated_pytest import fast_files, node_sharded_files, validate_shards

HISTORY = Path(".github/python-test-durations.json")
VERSIONS = ("3.12", "3.13")
FULL = {
    "python-wheelhouse", "ffmpeg-packages", "native-build", "native-sanitizers", "python-fast-run",
    "python-backend-run", "python-heavy", "python-coverage", "media-inputs",
    "media-render", "media-release-smoke",
}
SCOPED = {"sdks": "sdk", "foundry": "foundry", "remotion-checks": "remotion"}
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
}


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, timeout=10).stdout


def changed_paths(base: str, head: str) -> set[str]:
    """Read the exact PR merge delta, including both sides of renames."""
    if not all(re.fullmatch(r"[0-9a-f]{40}", sha) for sha in (base, head)):
        raise ValueError("PR event commit identity is missing")
    parents = git("rev-list", "--parents", "--max-count=1", "HEAD").decode("ascii").split()
    if len(parents) != 3 or parents[1:] != [base, head]:
        raise ValueError("checkout does not match both PR event parents")
    fields = git("diff", "--raw", "-z", "--no-renames", "--no-ext-diff", base, parents[0], "--").split(b"\0")
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


def scoped_plan(paths: set[str]) -> dict | None:
    if not paths:
        return None
    sdk = foundry = False
    changed_tests = set()
    tests = set()
    for path in paths:
        if path == "docs/ENGINE_GUIDE_AR.md":
            tests.add("tests/test_engine_guide.py")
        elif path in {"README.md", "SECURITY.md"} or (path.startswith("docs/") and path.endswith(".md")):
            continue
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
    if changed_tests:
        # Some test modules supply fixtures to other files, including node-sharded lanes.
        names = {Path(path).stem for path in changed_tests}
        names_pattern = "|".join(sorted(names)).encode("ascii")
        imports = re.compile(rb"\b(?:from|import)[ \t]+(?:\.?tests\.|\.)?(?:" + names_pattern + rb")\b")
        if any(imports.search(path.read_bytes()) for path in (ROOT / "tests").rglob("*.py")):
            return None
        tests.update(changed_tests)
    if tests:
        history = json.loads((ROOT / HISTORY).read_bytes())
        seconds = history["shardSeconds"]
        if (not tests <= set(fast_files(history)) or not tests <= seconds.keys()
                or sum(seconds[path] for path in tests) > 600):
            return None
    return {"full": False, "sdk": sdk, "foundry": foundry, "remotion": False, "tests": sorted(tests)}


def current_plan(event: str = "", base: str = "", head: str = "") -> dict:
    full = {"full": True, "sdk": True, "foundry": True, "remotion": True, "tests": []}
    if event == "pull_request":
        try:
            return scoped_plan(changed_paths(base, head)) or full
        except (OSError, ValueError, subprocess.SubprocessError, KeyError, TypeError):
            print("CI scope could not be established; running the full profile.", file=sys.stderr)
    return full


def validate_results(event: str, results: dict) -> None:
    """Require success from selected producers and an explicit skip from every other producer."""
    selection = results["plan"]["outputs"]
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
        "dependency-review": "success" if event == "pull_request" else "skipped",
        "python-version-smoke": "success" if event == "pull_request" and not full else "skipped",
        "python-affected": "success" if tests else "skipped",
        **{job: "success" if selection[key] == "true" else "skipped" for job, key in SCOPED.items()},
        **{job: "success" if full or (tests and job in TEST_SETUP) else "skipped" for job in FULL},
    }
    if not full and event != "pull_request":
        raise ValueError("full CI plan is required for this event")
    if full and (tests or any(selection[key] != "true" for key in SCOPED.values())):
        raise ValueError("full CI plan must select all full producers")
    actual = {job: details["result"] for job, details in results.items()}
    if actual != expected:
        raise ValueError(f"CI producers differ from the plan: expected={expected}, actual={actual}")


def validate_coverage(receipts: Path, head: str, history: dict) -> dict:
    """Verify both Python versions and the entire node-sharded population."""
    files = fast_files(history)
    coverage = {}
    for version in VERSIONS:
        fast = [json.loads(path.read_bytes()) for path in
                receipts.glob(f"python-fast-run-{version}-*/python-fast-status.json")]
        validate_shards(fast, head, version, files, history)
        backend = [json.loads(path.read_bytes()) for path in
                   receipts.glob(f"python-backend-run-{version}-*/receipt.json")]
        coverage[f"backend-{version}"] = validate_union(
            backend, head, sorted(history["versionedNodeShardedFiles"]),
            python=version, seconds=history["nodeSeconds"][version],
            shards=history["versionedNodeShards"],
        )
    heavy = [json.loads(path.read_bytes()) for path in receipts.glob("python-heavy-3.13-*/receipt.json")]
    coverage["heavy"] = validate_union(
        heavy, head, sorted(node_sharded_files(history)), seconds=history["heavyNodeSeconds"],
    )
    return coverage


def main() -> None:
    mode = sys.argv[1]
    if mode == "plan":
        outputs = current_plan(os.environ["GITHUB_EVENT_NAME"], os.getenv("PR_BASE", ""), os.getenv("PR_HEAD", ""))
        with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as stream:
            for key, value in outputs.items():
                print(f"{key}={json.dumps(value, ensure_ascii=True, separators=(',', ':'))}", file=stream)
    elif mode == "gate":
        validate_results(os.environ["GITHUB_EVENT_NAME"], json.loads(os.environ["CI_RESULTS"]))
    elif mode == "coverage":
        coverage = validate_coverage(Path(sys.argv[2]), os.environ["GITHUB_SHA"], json.loads(HISTORY.read_bytes()))
        print(json.dumps(coverage, indent=2))
    else:
        raise ValueError(f"unknown CI mode: {mode}")


if __name__ == "__main__":
    main()
