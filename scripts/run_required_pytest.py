"""Run explicitly selected pytest nodes and require a passing phase receipt for each."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from scripts.run_heavy_pytest import validate_execution


def main() -> int:
    nodes = sys.argv[1:]
    if (not nodes or len(nodes) != len(set(nodes))
            or any(not node.startswith("tests/") or "::" not in node for node in nodes)):
        raise ValueError("expected unique, explicit repository pytest node IDs")
    with tempfile.TemporaryDirectory(prefix="baloot-required-") as runtime:
        receipt = Path(runtime) / "execution.json"
        env = {name: value for name, value in os.environ.items() if not name.startswith("PYTEST_")}
        env.update(BALOOT_CI_NODE_RECEIPT=str(receipt), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-o", "addopts=", "-p", "scripts.run_heavy_pytest", *nodes],
            env=env, check=False,
        )
        if result.returncode:
            return result.returncode
        validate_execution(nodes, json.loads(receipt.read_bytes()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
