#!/usr/bin/env python3
"""Benchmark the exact tactical finalist precheck on a frozen batch corpus.

The benchmark is intentionally local evidence.  It proves that the precheck
does not reject any finalist accepted by the prior full funnel, and reports how
many rejected finalists can safely avoid that expensive funnel in future runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.request import urlopen

from baloot_platform.challenge.evidence_v3 import canonical_sha256
from baloot_platform.challenge.trace import CompleteGameTraceV1
from baloot_platform.config import platform_root
from baloot_platform.service.container import build_container


DEFAULT_BATCH_ID = "382c130009054f1794cc4d4ac91c0982"
DEFAULT_OUTPUT = Path(
    ".local-state/evidence/finalist-tactical-precheck-corpus.json"
)


def _get_json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=30) as response:  # noqa: S310 - local operator API
        return json.loads(response.read())


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://127.0.0.1:8080")
    parser.add_argument("--batch-id", default=DEFAULT_BATCH_ID)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = platform_root()
    output = (args.output or root / DEFAULT_OUTPUT).resolve()
    api_base = str(args.api_base).rstrip("/")
    batch = _get_json(
        f"{api_base}/api/v2/video-production-batches/{args.batch_id}"
    )
    items = [
        item
        for item in dict(batch.get("result") or {}).get("validationItems", [])
        if item.get("status") in {"accepted", "rejected"}
        and item.get("finalistJobId")
    ]
    if not items:
        raise RuntimeError("batch has no terminal finalist validation corpus")

    container = build_container(root=root)
    verifier = container.finalist_validations.tactical_viability_probe
    if verifier is None:
        container.shutdown()
        raise RuntimeError("exact tactical viability precheck is not active")

    rows: list[dict[str, Any]] = []
    started = perf_counter()
    try:
        for item in items:
            finalist = _get_json(
                f"{api_base}/api/v2/finalist-validations/"
                f"{item['finalistJobId']}"
            )
            result_items = list(dict(finalist.get("result") or {}).get("items", []))
            if len(result_items) != 1:
                raise RuntimeError(
                    f"expected one finalist result item for {item['finalistJobId']}"
                )
            result_item = result_items[0]
            trace_reference = dict(result_item.get("traceArtifact") or {})
            trace = CompleteGameTraceV1.from_dict(
                json.loads(
                    container.learning.artifact_store.get(
                        str(trace_reference["uri"])
                    )
                )
            )
            probe = verifier.probe_verified_tactical_play(trace)
            rows.append(
                {
                    "ordinal": int(item["ordinal"]),
                    "candidateHash": str(item["candidateHash"]),
                    "baselineStatus": str(item["status"]),
                    "baselineRejectionCode": item.get("rejectionCode"),
                    "descendantTraceHash": trace.sha256,
                    "eligibleDecisionCount": probe.eligible_decision_count,
                    "scannedDecisionCount": len(probe.scanned_decisions),
                    "probeStatus": probe.status,
                    "viable": probe.viable,
                    "canReject": probe.can_reject,
                    "probeHash": probe.sha256,
                }
            )
    finally:
        container.shutdown()

    accepted = [row for row in rows if row["baselineStatus"] == "accepted"]
    rejected = [row for row in rows if row["baselineStatus"] == "rejected"]
    accepted_false_rejections = [row for row in accepted if row["canReject"]]
    if accepted_false_rejections:
        raise RuntimeError(
            "precheck would reject a baseline-accepted finalist: "
            + ", ".join(
                str(row["candidateHash"]) for row in accepted_false_rejections
            )
        )

    evidence: dict[str, Any] = {
        "schemaVersion": 1,
        "kind": "local_finalist_tactical_precheck_corpus",
        "sourceBatchId": str(args.batch_id),
        "sourceBatchStatus": str(batch.get("status")),
        "sourceBatchRequestHash": canonical_sha256(batch.get("request") or {}),
        "profileHash": container.finalist_validation_profile.sha256,
        "verifierPlugin": verifier.descriptor.to_dict(),
        "corpus": {
            "terminalFinalists": len(rows),
            "baselineAccepted": len(accepted),
            "baselineRejected": len(rejected),
        },
        "result": {
            "acceptedPreserved": len(accepted) - len(accepted_false_rejections),
            "acceptedFalseRejections": len(accepted_false_rejections),
            "rejectedSafelyShortCircuited": sum(
                1 for row in rejected if row["canReject"]
            ),
            "rejectedRequiringFullFunnel": sum(
                1 for row in rejected if not row["canReject"]
            ),
            "elapsedSeconds": round(perf_counter() - started, 6),
        },
        "rows": sorted(rows, key=lambda row: int(row["ordinal"])),
        "proofBoundary": (
            "Local frozen-corpus losslessness and rejection-yield evidence. "
            "It does not prove professional correctness, publication readiness, "
            "or production performance."
        ),
    }
    evidence["sha256"] = canonical_sha256(evidence)
    _write_json(output, evidence)
    print(
        json.dumps(
            {
                "output": output.as_posix(),
                "acceptedPreserved": evidence["result"]["acceptedPreserved"],
                "acceptedFalseRejections": evidence["result"][
                    "acceptedFalseRejections"
                ],
                "rejectedSafelyShortCircuited": evidence["result"][
                    "rejectedSafelyShortCircuited"
                ],
                "rejectedRequiringFullFunnel": evidence["result"][
                    "rejectedRequiringFullFunnel"
                ],
                "elapsedSeconds": evidence["result"]["elapsedSeconds"],
                "sha256": evidence["sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
