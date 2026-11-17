#!/usr/bin/env python3
"""Generate the canonical V20 pytest collection inventory."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from baloot_platform.filesystem import atomic_write_text  # noqa: E402

DEFAULT_OUTPUT = ROOT / "release/test-inventory-v20.json"


def utc_now() -> str:
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "315532800"))
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def collect_node_ids() -> list[str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(ROOT / "src"),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "SOURCE_DATE_EPOCH": "315532800",
            "NO_COLOR": "1",
            "PY_COLORS": "0",
            "FORCE_COLOR": "0",
            "TERM": "dumb",
        }
    )
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "--color=no",
            "--override-ini=addopts=",
            "tests",
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=1200,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            "pytest collection failed:\n"
            + (process.stdout + "\n" + process.stderr)[-20000:]
        )
    node_ids = sorted(
        line.strip().replace("\\", "/")
        for line in process.stdout.splitlines()
        if line.strip().startswith("tests/") and "::" in line
    )
    if not node_ids:
        raise RuntimeError("pytest collection returned no test node IDs")
    return node_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--version", default="20.0.0")
    arguments = parser.parse_args()
    node_ids = collect_node_ids()
    canonical = ("\n".join(node_ids) + "\n").encode("utf-8")
    files = sorted({node_id.split("::", 1)[0] for node_id in node_ids})
    payload = {
        "schemaVersion": 1,
        "project": "baloot-platform",
        "version": arguments.version,
        "generatedAt": utc_now(),
        "status": "passed",
        "testFiles": len(files),
        "testCases": len(node_ids),
        "inventorySha256": hashlib.sha256(canonical).hexdigest(),
        "files": files,
        "nodeIds": node_ids,
    }
    output = Path(arguments.output).resolve()
    atomic_write_text(
        output,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(output)


if __name__ == "__main__":
    main()
