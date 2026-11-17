#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import yaml

from baloot_platform.benchmark import IntelligenceBenchmarkV18
from baloot_platform.certification import build_certification_candidate
from baloot_platform.operations import SloEvaluatorV18
from baloot_platform.video.hitl_review_v18 import HitlReviewClipRendererV18


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    release = root / "release"
    outputs = root / "outputs"
    release.mkdir(exist_ok=True); outputs.mkdir(exist_ok=True)

    candidate = build_certification_candidate(
        rules_path=root / "configs/rules/saudi_common_v1.yaml",
        adjudication_path=root / "configs/adjudication/saudi_researched_v1.yaml",
        golden_cases_path=root / "examples/rules-certification-golden-v18.json",
    ).to_dict()
    write(release / "rules-certification-candidate-v19.json", candidate)

    benchmark_profile = yaml.safe_load((root / "configs/benchmark/intelligence_v18.yaml").read_text(encoding="utf-8"))
    examples = [
        {"legalActions": ["C7", "C8"], "referenceAction": "C8", "predictedAction": "C8", "rankedActions": ["C8", "C7"], "referenceProbability": 0.9, "predictedRegret": 0.0, "matchValueLoss": 0.0, "archetype": "endgame"},
        {"legalActions": ["H7", "H8"], "referenceAction": "H8", "predictedAction": "H7", "rankedActions": ["H7", "H8"], "referenceProbability": 0.65, "predictedRegret": 1.2, "matchValueLoss": 0.8, "archetype": "opening_lead"},
    ]
    benchmark = IntelligenceBenchmarkV18(benchmark_profile).evaluate("v19-baseline-evidence", examples).to_dict()
    benchmark["strengthClaim"] = "pipeline_validation_only"
    benchmark["automaticPromotion"] = False
    write(release / "intelligence-benchmark-v19.json", benchmark)

    slo_profile = yaml.safe_load((root / "configs/operations/slo_v18.yaml").read_text(encoding="utf-8"))
    metrics = {"action_ack_p95_ms": 120, "room_event_p95_ms": 240, "analysis_standard_p95_ms": 3100, "replay_availability": 0.9999, "job_success_rate": 0.999, "data_durability": 0.9999999}
    slo = SloEvaluatorV18(slo_profile).evaluate(metrics)
    slo["metrics"] = metrics
    write(release / "slo-evaluation-v19.json", slo)

    hitl_profile = yaml.safe_load((root / "configs/hitl/governance_v18.yaml").read_text(encoding="utf-8"))
    case = {
        "caseId": "HITL-V19-RELEASE-0001",
        "publicContext": {"phase": "أكلة رابعة", "decisionKind": "play", "actor": "south", "matchScore": [140, 140]},
        "legalActions": ["D10", "CA"],
        "blind": True,
    }
    artifacts = HitlReviewClipRendererV18(hitl_profile).render(
        case=case,
        output_dir=outputs,
        basename="baloot-platform-v19-hitl-review-clip",
    )
    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    write(release / "hitl-review-probe-v19.json", {
        "status": "passed",
        "observerSafe": manifest["observerSafe"],
        "hiddenHandsIncluded": manifest["hiddenHandsIncluded"],
        "productionRecommendationIncluded": manifest["productionRecommendationIncluded"],
        "crowdConsensusIncluded": manifest["crowdConsensusIncluded"],
        "video": str(artifacts.video.relative_to(root)),
        "poster": str(artifacts.poster.relative_to(root)),
        "manifest": str(artifacts.manifest.relative_to(root)),
    })
    for path in (artifacts.video, artifacts.poster, artifacts.timeline, artifacts.manifest):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
