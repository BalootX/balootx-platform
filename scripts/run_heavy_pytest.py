"""Deterministic pytest-node shards, with exact executed-coverage validation."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_isolated_pytest import clear_outputs, isolated_environment, node_sharded_files, resource_snapshot

_nodes = {"collected": [], "deselected": [], "collectionIssues": [], "reports": {}}


def pytest_collection_finish(session):
    _nodes["collected"] = [item.nodeid for item in session.items]


def pytest_deselected(items):
    _nodes["deselected"].extend(item.nodeid for item in items)


def pytest_collectreport(report):
    if report.outcome != "passed":
        _nodes["collectionIssues"].append({"nodeid": report.nodeid, "outcome": report.outcome})


def pytest_runtest_logreport(report):
    event = {"phase": report.when, "outcome": report.outcome,
             "xfail": hasattr(report, "wasxfail"), "seconds": report.duration}
    if report.outcome == "failed":
        event["longrepr"] = str(report.longrepr)
    _nodes["reports"].setdefault(report.nodeid, []).append(event)
    target = os.environ.get("BALOOT_CI_NODE_RECEIPT")
    if target:
        with Path(target).with_suffix(".jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps({"nodeid": report.nodeid, **event}) + "\n")


def pytest_sessionfinish(session, exitstatus):
    target = os.environ.get("BALOOT_CI_NODE_RECEIPT")
    if target:
        Path(target).write_text(json.dumps({**_nodes, "exitstatus": int(exitstatus)}, indent=2) + "\n", encoding="utf-8")


def split_nodes(nodes: list[str], shard: int, seconds: dict | None = None, shards: int = 4) -> list[str]:
    if type(shards) is not int or type(shard) is not int or not 1 <= shard <= shards or not nodes or len(set(nodes)) != len(nodes):
        raise ValueError("expected a unique nonempty collection and a valid shard")
    if seconds is None:
        return sorted(nodes)[shard - 1::shards]
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in seconds.values()):
        raise ValueError("node durations must be finite nonnegative seconds")
    groups, loads = [[] for _ in range(shards)], [0.0] * shards
    for node in sorted(nodes, key=lambda node: (-seconds.get(node, 1), node)):
        index = min(range(shards), key=lambda index: (loads[index], len(groups[index]), index))
        groups[index].append(node)
        loads[index] += seconds.get(node, 1)
    return sorted(groups[shard - 1])


def validate_execution(selected: list[str], execution: dict) -> None:
    if execution["exitstatus"] or execution["deselected"] or execution["collectionIssues"]:
        raise ValueError("pytest failed, deselected nodes, or skipped/failed collection")
    if sorted(execution["collected"]) != sorted(selected) or set(execution["reports"]) != set(selected):
        raise ValueError("selected nodes were missing or unexpected nodes ran")
    for node in selected:
        phases = execution["reports"][node]
        if ([event["phase"] for event in phases] != ["setup", "call", "teardown"] or
                any(event["outcome"] != "passed" or event["xfail"] for event in phases)):
            raise ValueError(f"node did not pass exactly once without skipping: {node}")


def _pytest(arguments: list[str], output: Path, name: str, seconds: int) -> tuple[int, dict | None]:
    target = output / f"{name}.json"
    clear_outputs(target, target.with_suffix(".jsonl"), output / f"{name}.log")
    with tempfile.TemporaryDirectory(prefix=f"baloot-heavy-{name}-", dir=os.environ.get("RUNNER_TEMP")) as runtime:
        env = isolated_environment(runtime)
        env.update(PYTEST_ADDOPTS="", PYTHONHASHSEED="0", TMPDIR=runtime,
                   BALOOT_CI_NODE_RECEIPT=str(target), BALOOT_PYTEST_FILE_BUDGET="0")
        with (output / f"{name}.log").open("w", encoding="utf-8") as log:
            code = subprocess.run(
                ["timeout", "--signal=TERM", "--kill-after=10s", str(seconds), sys.executable,
                 "-m", "pytest", "-q", "--override-ini=addopts=", "--color=no",
                 f"--override-ini=cache_dir={runtime}/pytest-cache", "--basetemp", f"{runtime}/tmp",
                 "-p", "scripts.run_heavy_pytest", *arguments],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False,
            ).returncode
    return code, json.loads(target.read_text(encoding="utf-8")) if target.exists() else None


def run_shard(shard: int, output: Path, versioned: bool = False) -> int:
    resources = {"beforeAffinity": resource_snapshot()}
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    clear_outputs(output / "receipt.json", *(output / f"{name}.{extension}"
                  for name in ("collection", "execution") for extension in ("json", "jsonl", "log")))
    history = json.loads((ROOT / ".github/python-test-durations.json").read_text(encoding="utf-8"))
    files = sorted(history["versionedNodeShardedFiles"] if versioned else node_sharded_files(history))
    seconds = history["nodeSeconds"][".".join(platform.python_version().split(".")[:2])] if versioned else None
    shards = history["versionedNodeShards"] if versioned else 4
    limit = 3600 if versioned else 5400
    started = time.monotonic()
    receipt = {"shard": shard, "head": os.environ.get("GITHUB_SHA"), "python": platform.python_version(),
               "files": files, "collected": [], "selected": [], "execution": None, "status": "failed",
               "resources": resources}
    resources["beforeTests"] = resource_snapshot()
    print("Resource scope: " + json.dumps(resources), flush=True)
    try:
        if not 1 <= shard <= shards:
            raise ValueError("node shard is outside the configured range")
        if not files:
            raise ValueError("no configured node-sharded files; refusing implicit full-suite collection")
        code, collection = _pytest(["--collect-only", *files], output, "collection", 300)
        if code or collection is None or collection["exitstatus"] or collection["deselected"] or collection["collectionIssues"]:
            raise ValueError("heavy collection failed, skipped, or was incomplete")
        nodes = sorted(collection["collected"])
        if any(node.split("::", 1)[0] not in files or "::" not in node for node in nodes):
            raise ValueError("collection escaped the configured heavy files")
        selected = split_nodes(nodes, shard, seconds, shards)
        receipt.update(collected=nodes, selected=selected)
        print(f"Heavy shard {shard}: {len(selected)}/{len(nodes)} nodes; execution limit {limit}s", flush=True)
        if selected:
            code, execution = _pytest(selected, output, "execution", limit)
        else:
            code, execution = 0, {"exitstatus": 0, "collected": [], "deselected": [], "collectionIssues": [], "reports": {}}
        receipt["execution"] = execution
        if code or execution is None:
            raise ValueError(f"heavy shard execution incomplete or failed (exit {code})")
        validate_execution(selected, execution)
        receipt["status"] = "passed"
        return 0
    except (OSError, ValueError) as exc:
        receipt["error"] = str(exc)
        print(f"::error::Heavy shard {shard}: {exc}", flush=True)
        return 1
    finally:
        resources["afterTests"] = resource_snapshot()
        receipt["seconds"] = round(time.monotonic() - started, 3)
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


def validate_union(receipts: list[dict], head: str | None, files: list[str], python: str = "3.13",
                   seconds: dict | None = None, shards: int = 4) -> dict:
    if type(shards) is not int or shards < 1 or len(receipts) != shards or {row["shard"] for row in receipts} != set(range(1, shards + 1)):
        raise ValueError("exactly the configured distinct node shard receipts are required")
    nodes = receipts[0]["collected"]
    if any(node.split("::", 1)[0] not in files or "::" not in node for node in nodes):
        raise ValueError("heavy manifest contains nodes outside the configured files")
    executed = []
    for row in receipts:
        if (row["status"] != "passed" or row["head"] != head or row["files"] != files or
                not row["python"].startswith(python + ".") or row["collected"] != nodes):
            raise ValueError("failed, stale, version-mismatched or inconsistent heavy shard")
        selected = split_nodes(nodes, row["shard"], seconds, shards)
        if row["selected"] != selected:
            raise ValueError("heavy shard assignment differs from the deterministic partition")
        validate_execution(selected, row["execution"])
        executed.extend(row["execution"]["reports"])
    if sorted(executed) != nodes:
        raise ValueError("heavy executed union differs from the complete collection")
    return {"status": "passed", "head": head, "files": files, "collected": len(nodes), "executed": len(executed)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "summary"))
    parser.add_argument("--shard", type=int)
    parser.add_argument("--versioned", action="store_true")
    parser.add_argument("--python", choices=("3.12", "3.13"), default="3.13")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipts", type=Path)
    args = parser.parse_args()
    if args.mode == "run":
        if args.shard is None:
            parser.error("run requires --shard")
        return run_shard(args.shard, args.output, args.versioned)
    if args.receipts is None:
        parser.error("summary requires --receipts")
    history = json.loads((ROOT / ".github/python-test-durations.json").read_text(encoding="utf-8"))
    receipts = [json.loads(path.read_text(encoding="utf-8")) for path in args.receipts.glob("*/receipt.json")]
    files = sorted(history["versionedNodeShardedFiles"] if args.versioned else node_sharded_files(history))
    seconds = history["nodeSeconds"][args.python] if args.versioned else None
    shards = history["versionedNodeShards"] if args.versioned else 4
    result = validate_union(receipts, os.environ.get("GITHUB_SHA"), files, args.python, seconds, shards)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
