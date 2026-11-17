from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from baloot_platform.filesystem import atomic_write_text
from baloot_platform.research.live_result_latency import (
    build_live_result_latency_receipt,
    validate_live_result_latency_receipt,
)

try:
    from scripts.generate_performance_evidence import git_identity
except ModuleNotFoundError:
    from generate_performance_evidence import git_identity


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "configs/bundle_ortools_research_native_v3_1.yaml"
DEFAULT_SAMPLES = ROOT / "output/playwright/research-live-result-latency-samples.json"
DEFAULT_OUTPUT = ROOT / "release/research-live-result-latency.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bind browser SSE live-result samples to the governed V3 bundle."
    )
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    measurement = json.loads(args.samples.read_text(encoding="utf-8"))
    receipt = build_live_result_latency_receipt(
        ROOT,
        bundle_path=args.bundle,
        measurement=measurement,
        source=git_identity(ROOT),
    )
    validate_live_result_latency_receipt(
        ROOT,
        bundle_path=args.bundle,
        receipt=receipt,
    )
    atomic_write_text(
        args.output,
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(args.output.resolve())
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
