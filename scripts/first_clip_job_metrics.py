#!/usr/bin/env python3
"""Fail closed on pod limits; measure the one owned media process group."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CGROUP = Path("/sys/fs/cgroup")


def cpu_seconds() -> float:
    values = dict(line.split() for line in (CGROUP / "cpu.stat").read_text().splitlines())
    return int(values["usage_usec"]) / 1_000_000


def preflight(output: Path) -> None:
    memory_limit = int((CGROUP / "memory.max").read_text())
    quota, period = map(int, (CGROUP / "cpu.max").read_text().split())
    if not (0 < quota / period <= 4 and memory_limit == 8 * 1024**3):
        raise ValueError("admitted pod must have <=4 CPU and exactly 8 GiB memory limit")
    shm = os.statvfs("/dev/shm")
    workspace = os.statvfs(ROOT)
    if shm.f_blocks * shm.f_frsize != 2 * 1024**3:
        raise ValueError("admitted /dev/shm must be 2 GiB within the pod memory limit")
    capacity = workspace.f_blocks * workspace.f_frsize
    if not 18 * 1024**3 <= capacity <= 20 * 1024**3:
        raise ValueError("workspace must be the admitted bounded 20 GiB filesystem")
    fonts = ROOT / "assets/fonts"
    for name, item in json.loads((fonts / "manifest.json").read_text())["fonts"].items():
        if hashlib.sha256((fonts / name).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"font hash mismatch: {name}")
    (output / "pod-preflight.json").write_text(json.dumps({
        "cpuLimit": quota / period, "memoryLimitBytes": memory_limit,
        "shmBytes": shm.f_blocks * shm.f_frsize, "workspaceCapacityBytes": capacity,
        "cpuSecondsStart": cpu_seconds(), "startedEpoch": time.time(),
        "hostAdmission": "SEPARATE_SERVERS_RECEIPT_REQUIRED",
    }, indent=2) + "\n")


def run(
    output: Path,
    *,
    public: bool = False,
    source_head: str | None = None,
    source_tree: str | None = None,
    deal_id: str | None = None,
    seed: str | int | None = None,
) -> None:
    if deal_id is not None and seed is not None:
        raise ValueError("deal-id and seed are mutually exclusive")
    if not public and (deal_id is not None or seed is not None):
        raise ValueError("deal-id and seed require --public")
    baseline = json.loads((output / "pod-preflight.json").read_text())
    command = [sys.executable, "-B"]
    if public:
        command += ["scripts/render_research_notice.py", "--public", "--output", str(output)]
        if deal_id is not None:
            command += ["--deal-id", deal_id]
        elif seed is not None:
            command += ["--seed", str(seed)]
        if source_head is None and (ROOT / ".git").exists():
            source_head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip()
        if source_tree is None and (ROOT / ".git").exists():
            source_tree = subprocess.check_output(
                ["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, text=True
            ).strip()
        if source_head is not None or source_tree is not None:
            command += ["--source-head", source_head or "", "--source-tree", source_tree or ""]
    else:
        command += ["scripts/render_first_decision.py", "--output", str(output)]
    # Direct script execution puts only ``scripts/`` on sys.path; retain the
    # workflow's PYTHONPATH while adding the repository package roots.
    environment = os.environ.copy()
    inherited_pythonpath = environment.get("PYTHONPATH")
    pythonpath = [str(ROOT), str(ROOT / "src")]
    if inherited_pythonpath:
        pythonpath.append(inherited_pythonpath)
    environment["PYTHONPATH"] = os.pathsep.join(pythonpath)
    process = subprocess.Popen(command, cwd=ROOT, env=environment, start_new_session=True)
    peak_rss = 0
    disk_peak = 0
    deadline = time.monotonic() + 2400
    try:
        while True:
            rss = 0
            for entry in Path("/proc").iterdir():
                if not entry.name.isdecimal():
                    continue
                try:
                    if os.getpgid(int(entry.name)) == process.pid:
                        for line in (entry / "status").read_text().splitlines():
                            if line.startswith("VmRSS:"):
                                rss += int(line.split()[1]) * 1024
                except (ProcessLookupError, FileNotFoundError, PermissionError):
                    continue
            peak_rss = max(peak_rss, rss)
            workspace = os.statvfs(ROOT)
            disk_peak = max(disk_peak, (workspace.f_blocks - workspace.f_bfree) * workspace.f_frsize)
            code = process.poll()
            if code is not None:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("one-clip process exceeded its 40 minute bound")
            time.sleep(5)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        output_bytes = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        (output / "runtime-metrics.json").write_text(json.dumps({
            "exitCode": process.returncode, "cpuSeconds": cpu_seconds() - baseline["cpuSecondsStart"],
            "elapsedSecondsIncludingPreparation": time.time() - baseline["startedEpoch"],
            "peakRssSampledProcessGroupBytes": peak_rss,
            "cgroupMemoryPeakBytes": int((CGROUP / "memory.peak").read_text()),
            "workspaceDiskPeakBytes": disk_peak, "outputBytes": output_bytes,
            "method": "cgroup v2 cpu.stat delta including tool preparation; memory.peak includes shm/cache; 5s sum of owned process-group /proc VmRSS (shared pages may count twice); 5s statvfs workspace allocated bytes",
        }, indent=2) + "\n")
    if code != 0:
        raise SystemExit(code)
    if output_bytes > 499_000_000:
        raise ValueError("artifact exceeds its 500 MB bound including receipts")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public", action="store_true", help="Run the approved-v4 public producer.")
    parser.add_argument("--source-head", help="Pinned source commit SHA for archive execution.")
    parser.add_argument("--source-tree", help="Pinned source tree SHA for archive execution.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--deal-id", "--deal-identifier", dest="deal_id",
        help="Select one governed V4 deal identifier.",
    )
    selection.add_argument("--seed", help="Deterministically select a governed V4 deal from the source catalog.")
    args = parser.parse_args()
    if not args.public and (args.deal_id is not None or args.seed is not None):
        parser.error("--deal-id/--seed require --public")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "preflight":
        preflight(args.output)
    else:
        run(
            args.output,
            public=args.public,
            source_head=args.source_head,
            source_tree=args.source_tree,
            deal_id=args.deal_id,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
