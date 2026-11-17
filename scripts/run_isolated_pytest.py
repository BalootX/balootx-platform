"""Run isolated Python files on four CPUs, then timing-sensitive files serially."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
FILE_SAFETY_SECONDS = 840.0


def node_sharded_files(history: dict) -> set[str]:
    return set(history.get("nodeShardedFiles", [])) | {
        path for path, seconds in history["seconds"].items() if seconds > 3600
    }


def fast_files(history: dict) -> list[str]:
    inventory = {path.relative_to(ROOT).as_posix() for path in ROOT.glob("tests/test_*.py")}
    legacy, versioned = node_sharded_files(history), set(history.get("versionedNodeShardedFiles", []))
    if legacy & versioned:
        raise ValueError("legacy and per-version node profiles must be disjoint")
    excluded = legacy | versioned
    if not excluded <= inventory:
        raise ValueError("node-sharded files are missing from the test inventory")
    return sorted(inventory - excluded)


def isolated_environment(runtime: str) -> dict[str, str]:
    env = {**{name: value for name, value in os.environ.items() if not name.startswith("PYTEST_")},
           "BALOOT_DATA_DIR": runtime, "BALOOT_RUNTIME": runtime,
           "BALOOT_OUTPUTS": str(Path(runtime) / "outputs"),
           "BALOOT_NATIVE_LIBRARY": str(ROOT / "build/native-release/libbaloot_native.so"),
           "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "CMAKE_BUILD_PARALLEL_LEVEL"):
        env[name] = "1"
    return env


def clear_outputs(*paths: Path) -> None:
    for path in paths:
        path.unlink(missing_ok=True)
        if path.exists() or path.is_symlink():
            raise ValueError(f"stale pytest output could not be removed: {path}")


def resource_snapshot() -> dict:
    files = {}
    for name in ("/proc/self/cgroup", "/proc/loadavg", "/proc/pressure/cpu",
                 "/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu.stat",
                 "/sys/fs/cgroup/cpu.pressure", "/sys/fs/cgroup/cpuset.cpus.effective"):
        try:
            files[name] = Path(name).read_text(encoding="ascii").strip()
        except (OSError, UnicodeError):
            files[name] = None
    return {"monotonicSeconds": time.monotonic(),
            "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "files": files}


def validate_execution(execution: dict) -> dict:
    if (execution["exitstatus"] or execution["deselected"] or
            any(issue["outcome"] != "skipped" for issue in execution["collectionIssues"])):
        raise ValueError("pytest failed, deselected tests, or failed collection")
    nodes, reports = execution["collected"], execution["reports"]
    if not nodes or len(nodes) != len(set(nodes)) or set(nodes) != set(reports):
        raise ValueError("no tests executed or collected tests have no execution receipt")
    for node, phases in reports.items():
        if not phases:
            raise ValueError(f"incomplete test phases: {node}")
        expected = ["setup", "teardown"] if phases[0]["outcome"] == "skipped" else ["setup", "call", "teardown"]
        if ([event["phase"] for event in phases] != expected or
                any(event["outcome"] not in ("passed", "skipped") for event in phases)):
            raise ValueError(f"incomplete or failed test phases: {node}")
    calls = sum(event["phase"] == "call" for phases in reports.values() for event in phases)
    all_skipped = all(any(event["outcome"] == "skipped" for event in phases) for phases in reports.values())
    if not calls and not all_skipped:
        raise ValueError("no test call executed and tests were not all skipped")
    return {"callCount": calls, "allSkipped": all_skipped}


def all_skipped_allowlist(available: set[str]) -> dict[str, str]:
    allowed = {}
    for line in (ROOT / "ci/all-skipped-allowlist.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        path, separator, reason = line.partition(" # ")
        if not separator or not reason.strip() or path not in available or path in allowed:
            raise ValueError("all-skipped allowlist requires unique test files and a reason per file")
        allowed[path] = reason.strip()
    return allowed


def schedule(files: list[str], history: dict, light_timeout: int = 900) -> list[tuple[str, int]]:
    seconds = history["seconds"]
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in seconds.values()):
        raise ValueError("historical durations must be finite nonnegative seconds")
    heavy = set(history["incompleteFiles"]) | {path for path, value in seconds.items() if value >= 900}
    ordered = sorted(files, key=lambda path: (path not in heavy, -seconds.get(path, 0), path))
    return [(path, 3600 if path in heavy else light_timeout) for path in ordered]


def work_file_budget(path: str, timeout: int, history: dict) -> float:
    recorded = [history.get(field, {}).get(path, 0) for field in ("seconds", "shardSeconds")]
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in recorded):
        raise ValueError("historical durations must be finite nonnegative seconds")
    budget = min(FILE_SAFETY_SECONDS, timeout - 60)
    return budget if timeout >= 900 and max(recorded) < budget else 0.0


def split_files(files: list[str], history: dict, shards: int = 4) -> list[list[str]]:
    if not files or len(set(files)) != len(files) or not 1 <= shards <= len(files):
        raise ValueError("expected unique nonempty files and a valid shard count")
    history = {**history, "seconds": history.get("shardSeconds", history["seconds"])}
    ordered = [path for path, _ in schedule(files, history)]
    seconds = {path: max(1, history["seconds"].get(path, 1)) for path in files}
    serial = set(history.get("serialFiles", []))
    # Reserve one pool shard even when several serial files need dedicated VMs.
    dedicated = [path for path in ordered if path in serial and seconds[path] >= 900][:shards - 1]
    groups = [[path] for path in dedicated] + [[] for _ in range(shards - len(dedicated))]
    pooled = [[0.0] * 4 for _ in groups]
    serial_loads = [0.0] * len(groups)
    for path in sorted(set(files) - set(dedicated), key=lambda path: (-seconds[path], path)):
        def finish(index: int) -> float:
            if path in serial:
                return max(pooled[index]) + serial_loads[index] + seconds[path]
            return max(max(pooled[index]), min(pooled[index]) + seconds[path]) + serial_loads[index]

        index = min(range(len(dedicated), shards), key=lambda index: (finish(index), len(groups[index]), index))
        groups[index].append(path)
        if path in serial:
            serial_loads[index] += seconds[path]
        else:
            worker = min(range(4), key=lambda worker: (pooled[index][worker], worker))
            pooled[index][worker] += seconds[path]
    return [sorted(group) for group in groups]


def validate_shards(receipts: list[dict], head: str, python: str, files: list[str], history: dict) -> None:
    expected = split_files(files, history)
    if len(receipts) != 4 or {row["shard"] for row in receipts} != {1, 2, 3, 4}:
        raise ValueError("exactly four distinct fast shard receipts are required")
    for row in receipts:
        if (row != {"head": head, "python": python, "shard": row["shard"],
                    "result": "success", "files": expected[row["shard"] - 1]}):
            raise ValueError("fast shard failed or head, Python, or executed inventory differs")


def run(files: list[str], output: Path, workers: int = 4, timeout: int = 900) -> int:
    available = {path.relative_to(ROOT).as_posix() for path in ROOT.glob("tests/test_*.py")}
    if not files or len(set(files)) != len(files) or not set(files) <= available:
        raise ValueError("expected unique repository test files, with no omissions by default")
    if not 1 <= workers <= 4 or not 1 <= timeout <= 900:
        raise ValueError("workers must be 1..4 and timeout 1..900 seconds")
    history = json.loads((ROOT / ".github/python-test-durations.json").read_text(encoding="utf-8"))
    serial = history.get("serialFiles", [])
    if len(serial) != len(set(serial)) or not set(serial) <= available:
        raise ValueError("serial files must be unique repository test files")
    serial = set(serial)
    allowed = all_skipped_allowlist(available)
    planned = schedule(files, history, timeout)
    resources = {"beforeAffinity": resource_snapshot()}
    # Descendants inherit this affinity even if a native library ignores thread caps.
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:workers])
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    clear_outputs(output / "results.json", output / "results.jsonl")
    resources["beforeTests"] = resource_snapshot()
    (output / "resources.json").write_text(json.dumps(resources, indent=2) + "\n", encoding="utf-8")
    print("Resource scope: " + json.dumps(resources), flush=True)

    def execute(path: str, file_timeout: int) -> dict:
        print(f"START {path} (timeout {file_timeout}s)", flush=True)
        started = time.monotonic()
        log = output / (Path(path).stem + ".log")
        execution = output / (Path(path).stem + ".execution.json")
        clear_outputs(execution, execution.with_suffix(".jsonl"), log)
        evidence = {"callCount": 0, "allSkipped": False}
        error = None
        with tempfile.TemporaryDirectory(prefix="baloot-ci-", dir=os.environ.get("RUNNER_TEMP")) as runtime:
            env = isolated_environment(runtime)
            env["BALOOT_CI_NODE_RECEIPT"] = str(execution)
            env["BALOOT_PYTEST_FILE_BUDGET"] = "1" if work_file_budget(path, file_timeout, history) else "0"
            with log.open("w", encoding="utf-8") as capture:
                try:
                    result = subprocess.run(
                        ["timeout", "--signal=TERM", "--kill-after=10s", str(file_timeout),
                         sys.executable, "-m", "pytest", "-q", "--override-ini=addopts=",
                         f"--override-ini=cache_dir={runtime}/pytest-cache", "--basetemp", f"{runtime}/tmp",
                         "-p", "scripts.run_heavy_pytest", path],
                        cwd=ROOT, env=env, stdout=capture, stderr=subprocess.STDOUT, check=False,
                        start_new_session=True,
                    )
                    code = result.returncode
                    if code == 0:
                        try:
                            evidence = validate_execution(json.loads(execution.read_text(encoding="utf-8")))
                            if evidence["callCount"] == 0 and path not in allowed:
                                raise ValueError("zero call phases: file is not in the all-skipped allowlist")
                        except (OSError, ValueError, KeyError, TypeError) as exc:
                            code, error = 1, f"invalid execution receipt: {exc}"
                            capture.write("\n" + error + "\n")
                except OSError as exc:
                    capture.write(str(exc))
                    code = 127
        return {"file": path, "seconds": round(time.monotonic() - started, 3),
                "returncode": code, "timedOut": code == 124, "timeoutSeconds": file_timeout,
                "tier": "heavy" if file_timeout == 3600 else "light", **evidence, "error": error,
                "allSkippedAllowlistReason": allowed.get(path),
                "removableAllowlistEntry": path in allowed and evidence["callCount"] > 0}

    results = []

    def record(result: dict) -> None:
        results.append(result)
        with (output / "results.jsonl").open("a", encoding="utf-8") as receipt:
            receipt.write(json.dumps(result) + "\n")
        status = "PASS" if result["returncode"] == 0 else "FAIL"
        print(f"{status} {result['file']} ({result['seconds']}s, exit {result['returncode']})", flush=True)
        if result["allSkipped"]:
            print(f"ALL SKIPPED {result['file']}", flush=True)
        if result["removableAllowlistEntry"]:
            print(f"ALLOWLIST REMOVABLE {result['file']}: call phases executed", flush=True)
        if status == "FAIL":
            print(f"::error file={result['file']}::exit {result['returncode']}; timeout={result['timedOut']}", flush=True)
            with (output / (Path(result["file"]).stem + ".log")).open("rb") as capture:
                capture.seek(0, os.SEEK_END)
                capture.seek(max(0, capture.tell() - 16000))
                print(capture.read().decode("utf-8", errors="replace"), flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(execute, path, file_timeout) for path, file_timeout in planned if path not in serial]
        for future in as_completed(pending):
            record(future.result())
    # Timing assertions and native deadlines must not compete with pooled files.
    for path, file_timeout in planned:
        if path in serial:
            record(execute(path, file_timeout))
    (output / "results.json").write_text(json.dumps(sorted(results, key=lambda row: row["file"]), indent=2) + "\n", encoding="utf-8")
    failed = sum(row["returncode"] != 0 for row in results)
    resources["afterTests"] = resource_snapshot()
    (output / "resources.json").write_text(json.dumps(resources, indent=2) + "\n", encoding="utf-8")
    print("All-skipped files: " + ", ".join(row["file"] for row in results if row["allSkipped"]), flush=True)
    print(f"{failed} of {len(results)} Python test files failed", flush=True)
    return int(failed != 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=900, help="Light-file limit in seconds; historical heavy files receive 3600")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", type=int, choices=range(1, 5))
    args = parser.parse_args()
    history = json.loads((ROOT / ".github/python-test-durations.json").read_text(encoding="utf-8"))
    files = fast_files(history)
    if args.shard is not None:
        files = split_files(files, history)[args.shard - 1]
    excluded = node_sharded_files(history) | set(history.get("versionedNodeShardedFiles", []))
    print("Coverage handled by exact node-sharded jobs: " + ", ".join(sorted(excluded)), flush=True)
    return run(files, args.output, args.workers, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
