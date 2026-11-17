from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from baloot_platform.core.rules import RuleBook
from baloot_platform.research.policy_proof_benchmark import (
    benchmark_reduced_policy_proof,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    root = args.root.resolve()
    rules = RuleBook.load(root / "configs/rules/saudi_common_v1.yaml")
    convention = yaml.safe_load(
        (root / "configs/conventions/conservative_v1.yaml").read_text(encoding="utf-8")
    )
    print(
        json.dumps(
            benchmark_reduced_policy_proof(
                rules,
                convention,
                iterations=args.iterations,
            ),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
