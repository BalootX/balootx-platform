#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import selectors
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "release/pytest-lanes-v20.json"
LOGS = ROOT / "release/test-lanes-v20"
TESTS = sorted((ROOT / "tests").glob("test_*.py"))
SUMMARY_RE = re.compile(
    r"(\d+) passed"
    r"(?:, \d+ (?:skipped|warnings?|xfailed|xpassed|deselected))*"
    r"(?:\x1b\[[0-9;]*m)* in "
)


def persist(results: list[dict], started_all: float, expected_files: int | None = None, report: Path = REPORT) -> dict:
    payload = {
        "schemaVersion": 1,
        "version": "20.0.0",
        "status": "passed" if len(results) == (expected_files if expected_files is not None else len(TESTS)) and all(x["status"] == "passed" for x in results) else "running_or_failed",
        "testFiles": len(results),
        "expectedTestFiles": expected_files if expected_files is not None else len(TESTS),
        "testsPassed": sum(int(x.get("testsPassed", 0)) for x in results),
        "durationSeconds": round(time.perf_counter() - started_all, 3),
        "results": results,
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def run_file(test_file: Path) -> dict:
    runtime = ROOT / ".release-runtime" / "v20" / test_file.stem
    shutil.rmtree(runtime, ignore_errors=True)
    runtime.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(ROOT / "src"),
        "BALOOT_ROOT": str(ROOT),
        "BALOOT_DATA_DIR": str(runtime),
        "BALOOT_RUNTIME": str(runtime),
        "BALOOT_OUTPUTS": str(runtime / "outputs"),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "SOURCE_DATE_EPOCH": "315532800",
        "NO_COLOR": "1",
        "PY_COLORS": "0",
        "FORCE_COLOR": "0",
        "TERM": "dumb",
    })
    command = [sys.executable, "-m", "pytest", "-vv", "--color=no", "--override-ini=addopts=", str(test_file)]
    started = time.perf_counter()
    capture_path = runtime / "pytest-output.log"
    timed_out = False
    returncode = None
    try:
        with capture_path.open("w", encoding="utf-8") as capture:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                text=True,
                stdout=capture,
                stderr=subprocess.STDOUT,
                timeout=1200,
                check=False,
            )
            returncode = completed.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        returncode = 124
    output = capture_path.read_text(encoding="utf-8", errors="replace") if capture_path.exists() else ""
    matches = re.findall(r"(\d+) passed", output)
    tests_passed = int(matches[-1]) if matches else 0
    summary_seen = bool(SUMMARY_RE.search(output))
    explicit_failure = (
        bool(re.search(r"(^|\n)(?:FAILED|ERROR)(?:\s|$)", output))
        or "short test summary info" in output
        or "ERROR collecting" in output
        or timed_out
    )
    passed = returncode == 0 and summary_seen and tests_passed > 0 and not explicit_failure
    return {
        "file": test_file.name,
        "status": "passed" if passed else "failed",
        "testsPassed": tests_passed,
        "processReturnCode": returncode,
        "timedOut": timed_out,
        "durationSeconds": round(time.perf_counter() - started, 3),
        "outputTail": output[-16000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="inclusive test-file index")
    parser.add_argument("--end", type=int, default=None, help="exclusive test-file index")
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()

    selected = TESTS[args.start:args.end]
    LOGS.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    started_all = time.perf_counter()
    for test_file in selected:
        result = run_file(test_file)
        results.append(result)
        (LOGS / f"{test_file.stem}.log").write_text(result["outputTail"], encoding="utf-8")
        persist(results, started_all, expected_files=len(selected), report=args.report)
        print(f"{test_file.name}: {result['status']} ({result['testsPassed']})", flush=True)
        if result["status"] != "passed":
            return 1
    payload = persist(results, started_all, expected_files=len(selected), report=args.report)
    payload["status"] = "passed"
    payload["selection"] = {"start": args.start, "end": args.end, "files": [x.name for x in selected]}
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("status", "testFiles", "testsPassed", "durationSeconds")}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
