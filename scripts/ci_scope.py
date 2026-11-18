"""Select CI work and require exact job and shard coverage for that selection."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_heavy_pytest import validate_union
from scripts.run_isolated_pytest import fast_files, node_sharded_files, validate_shards

HISTORY = Path(".github/python-test-durations.json")
VERSIONS = ("3.12", "3.13")
FULL = {
    "python-wheelhouse", "native-build", "native-sanitizers", "python-fast-run",
    "python-backend-run", "python-heavy", "python-coverage", "media-inputs",
    "media-render", "media-release-smoke",
}
SCOPED = {"sdks": "sdk", "foundry": "foundry", "remotion-checks": "remotion"}


def current_plan() -> dict:
    return {"full": True, "sdk": True, "foundry": True, "remotion": True, "tests": []}


def validate_results(event: str, results: dict) -> None:
    """Require success from selected producers and an explicit skip from every other producer."""
    selection = results["plan"]["outputs"]
    for key in ("full", *SCOPED.values()):
        if selection.get(key) not in ("true", "false"):
            raise ValueError(f"invalid CI plan output: {key}")
    tests = json.loads(selection["tests"])
    if not isinstance(tests, list) or any(not isinstance(path, str) for path in tests):
        raise ValueError("invalid CI plan test list")
    expected = {
        "plan": "success",
        "preflight": "success",
        "dependency-review": "success" if event in {"pull_request", "merge_group"} else "skipped",
        "python-affected": "success" if tests else "skipped",
        **{job: "success" if selection[key] == "true" else "skipped" for job, key in SCOPED.items()},
        **{job: "success" if selection["full"] == "true" else "skipped" for job in FULL},
    }
    if selection["full"] != "true":
        raise ValueError("full CI plan is required for this event")
    if selection["full"] == "true" and (tests or any(selection[key] != "true" for key in SCOPED.values())):
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
        outputs = current_plan()
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
