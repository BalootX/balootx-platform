from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from baloot_platform.service.container import build_container


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="balootx-research-verify-") as raw:
        container = build_container(writable_root=Path(raw))
        try:
            research = container.research
            capabilities = research.capabilities()
            active = capabilities["research"]
            recipe = {
                "schemaVersion": 1,
                "titleAr": "اختبار مسار البحث الكامل",
                "output": {"kind": "laffa", "resultCount": 1},
                "profiles": {
                    "domainHash": capabilities["sha256"],
                    "bundleHash": capabilities["bundleHash"],
                    "rulesHash": capabilities["rulesHash"],
                    "conventionHash": capabilities["conventionHash"],
                    "agentRelease": active["defaultAgentRelease"],
                    "analysisProfileHash": active["analysisProfiles"]["screening"],
                    "scoreProfileHash": active["defaultScoreProfileHash"],
                },
                "situation": {
                    "dealer": "east",
                    "buyer": None,
                    "contracts": [],
                    "matchScore": [0, 0],
                    "requiredCards": {},
                    "forbiddenCards": {},
                    "projects": [],
                    "escalations": [],
                },
                "tactics": [],
                "qualification": {
                    "minimumTruth": "provisional",
                    "minimumConfidence": 0,
                    "minimumSurroundingQuality": 0,
                    "requireObserverSafety": True,
                    "requireAllDecisionAudit": True,
                    "maximumRegret": 999,
                },
                "resources": {
                    "seedStart": 940001,
                    "seedCount": 3,
                    "seedsPerShard": 1,
                    "candidatesPerSeed": 1,
                    "solverTimeoutSeconds": 5,
                    "maximumConflicts": 100000,
                    "analysisMode": "screening",
                },
                "status": "published",
            }
            created = research.create_recipe(recipe)
            run = research.submit(
                created["code"], idempotency_key="research-verification-v1"
            )
            current = run
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                current = research.get_run(run["runCode"])
                assert current is not None
                if current["status"] in {"completed", "failed", "cancelled"}:
                    break
                time.sleep(0.2)
            results = research.list_results(run_code=run["runCode"], limit=10)
            summary = {
                "schemaVersion": 1,
                "runCode": run["runCode"],
                "status": current["status"],
                "resultCount": current.get("resultCount", 0),
                "qualifiedCount": current.get("qualifiedCount", 0),
                "quarantineCount": current.get("quarantineCount", 0),
                "storedResults": results["total"],
                "error": current.get("error"),
            }
            if results["items"]:
                result = results["items"][0]
                decisions = research.get_decisions(result["resultCode"])
                summary.update(
                    {
                        "resultCode": result["resultCode"],
                        "disposition": result["disposition"],
                        "decisionCount": len(decisions),
                        "auditComplete": len(decisions) == result["decisionCount"],
                        "actualRecommendedSeparated": all(
                            "actualAction" in decision
                            and "recommendedAction" in decision
                            for decision in decisions
                        ),
                    }
                )
            print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
            return 0 if (
                summary["status"] == "completed"
                and summary["storedResults"] >= 1
                and summary.get("auditComplete") is True
                and summary.get("actualRecommendedSeparated") is True
            ) else 1
        finally:
            container.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
