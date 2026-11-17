from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from baloot_platform.core.rules import RuleBook
from baloot_platform.research.policy_proof_campaign import (
    PolicyProofCampaignProfileV1,
    run_policy_proof_qualification_campaign,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--seeds-per-cell", type=int, default=2)
    args = parser.parse_args()
    root = args.root.resolve()
    rules = RuleBook.load(root / "configs/rules/saudi_common_v1.yaml")
    convention = yaml.safe_load(
        (root / "configs/conventions/conservative_v1.yaml").read_text(encoding="utf-8")
    )
    result = run_policy_proof_qualification_campaign(
        rules,
        convention,
        profile=PolicyProofCampaignProfileV1(seeds_per_cell=args.seeds_per_cell),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
