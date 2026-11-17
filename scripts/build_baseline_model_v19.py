#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from baloot_platform.ml.multitask_v7 import NumpyMultiTaskLinearModel, model_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="release/models/v19-baseline")
    parser.add_argument("--seed", type=int, default=17001)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    model = NumpyMultiTaskLinearModel(seed=args.seed)
    artifact = output / "numpy_multitask_linear_v19.npz"
    np.savez_compressed(
        artifact,
        policy=model.policy,
        belief=model.belief,
        values=model.values,
        intentions=model.intentions,
    )
    manifest = model_manifest(
        model_id="numpy_multitask_linear_v19_baseline",
        version="19.0.0",
        artifact=artifact,
        training_dataset_id="deterministic_initialization_no_training_data",
        metrics={
            "strengthStatus": "baseline_not_elite",
            "legalMaskEnforced": True,
            "hiddenInformationSafeContract": True,
            "promotionEligible": False,
            "seed": args.seed,
        },
    )
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
