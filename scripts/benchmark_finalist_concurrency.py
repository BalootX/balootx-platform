#!/usr/bin/env python3
"""Compare one and two finalist workers on the same frozen candidates."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, Mapping
from urllib.request import urlopen

from baloot_platform.challenge.evidence_v3 import canonical_sha256
from baloot_platform.config import platform_root
from baloot_platform.service.container import build_container


DEFAULT_BATCH_ID = "382c130009054f1794cc4d4ac91c0982"
DEFAULT_OUTPUT = Path(
    ".local-state/evidence/finalist-concurrency-1-vs-2.json"
)


def _get_json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=30) as response:  # noqa: S310 - local operator API
        return json.loads(response.read())


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _semantic_signature(job: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(job.get("result") or {})
    items = list(result.get("items") or ())
    if len(items) != 1:
        raise RuntimeError("benchmark finalist job must contain exactly one item")
    item = dict(items[0])
    evidence = dict(item.get("evidence") or {})
    analysis = dict(item.get("analysisEvidence") or {})
    viability = dict(item.get("tacticalViabilityProbe") or {})
    return {
        "manifestHash": result.get("manifestHash"),
        "descendantTraceHash": item.get("descendantTraceHash"),
        "replayEvidenceHash": evidence.get("sha256"),
        "analysisEvidenceHash": analysis.get("sha256"),
        "tacticalViabilityHash": viability.get("sha256"),
        "blockers": sorted(str(value) for value in item.get("blockers") or ()),
        "localPreviewReady": bool(item.get("localPreviewReady")),
        "publicationReady": bool(item.get("publicationReady")),
    }


def _run(
    *,
    root: Path,
    candidates: tuple[dict[str, Any], ...],
    concurrency: int,
    run_label: str,
) -> dict[str, Any]:
    container = build_container(root=root)
    service = container.finalist_validations
    pending = list(candidates)
    active: dict[str, dict[str, Any]] = {}
    completed: dict[str, dict[str, Any]] = {}
    started = perf_counter()
    try:
        while pending or active:
            while pending and len(active) < concurrency:
                candidate = pending.pop(0)
                submitted = service.submit(
                    analysis_job_id=str(candidate["analysisJobId"]),
                    candidate_hashes=(str(candidate["candidateHash"]),),
                    idempotency_key=(
                        f"finalist-concurrency-{run_label}-{concurrency}-"
                        f"{candidate['candidateHash']}"
                    ),
                    required_decision_kind="play",
                )
                active[str(submitted["jobId"])] = candidate
            terminal: list[str] = []
            for job_id, candidate in active.items():
                job = service.get(job_id)
                if job is None:
                    raise RuntimeError(f"finalist job disappeared: {job_id}")
                if job["status"] in {"completed", "failed", "cancelled"}:
                    if job["status"] != "completed":
                        raise RuntimeError(
                            f"finalist job {job_id} ended as {job['status']}: "
                            f"{job.get('error')}"
                        )
                    completed[str(candidate["candidateHash"])] = {
                        "semantic": _semantic_signature(job),
                        "createdAt": str(job["createdAt"]),
                        "updatedAt": str(job["updatedAt"]),
                        "operationalMetrics": dict(
                            dict(job.get("result") or {}).get(
                                "operationalMetrics"
                            )
                            or {}
                        ),
                    }
                    terminal.append(job_id)
            for job_id in terminal:
                active.pop(job_id)
            if active:
                sleep(0.1)
    finally:
        container.shutdown()
    created = [
        datetime.fromisoformat(str(value["createdAt"]))
        for value in completed.values()
    ]
    updated = [
        datetime.fromisoformat(str(value["updatedAt"]))
        for value in completed.values()
    ]
    return {
        "concurrency": concurrency,
        "candidateCount": len(candidates),
        "wallSeconds": round((max(updated) - min(created)).total_seconds(), 6),
        "operatorWallSeconds": round(perf_counter() - started, 6),
        "results": completed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://127.0.0.1:8080")
    parser.add_argument("--batch-id", default=DEFAULT_BATCH_ID)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-label", required=True)
    args = parser.parse_args()

    root = platform_root()
    output = (args.output or root / DEFAULT_OUTPUT).resolve()
    api_base = str(args.api_base).rstrip("/")
    batch = _get_json(
        f"{api_base}/api/v2/video-production-batches/{args.batch_id}"
    )
    rows = [
        dict(item)
        for item in dict(batch.get("result") or {}).get("validationItems", [])
        if item.get("finalistJobId")
        and item.get("status") in {"accepted", "rejected"}
    ]
    accepted = sorted(
        (row for row in rows if row["status"] == "accepted"),
        key=lambda row: int(row["ordinal"]),
    )[:4]
    hard_rejects = sorted(
        (
            row
            for row in rows
            if row["status"] == "rejected"
            and row.get("rejectionCode")
            != "no_verified_tactical_play_opportunity"
        ),
        key=lambda row: int(row["ordinal"]),
    )[:3]
    cheap_rejects = sorted(
        (
            row
            for row in rows
            if row["status"] == "rejected"
            and row.get("rejectionCode")
            == "no_verified_tactical_play_opportunity"
        ),
        key=lambda row: int(row["ordinal"]),
    )
    selected_rows = list(accepted + hard_rejects + cheap_rejects[:1])
    selected_hashes = {str(row["candidateHash"]) for row in selected_rows}
    selected_rows.extend(
        row
        for row in sorted(rows, key=lambda value: int(value["ordinal"]))
        if str(row["candidateHash"]) not in selected_hashes
    )
    selected_rows = tuple(selected_rows[:8])
    if len(selected_rows) < 2:
        raise RuntimeError("benchmark corpus is too small")
    candidates = tuple(
        {
            "ordinal": int(row["ordinal"]),
            "candidateHash": str(row["candidateHash"]),
            "analysisJobId": str(row["analysisJobId"]),
            "baselineStatus": str(row["status"]),
            "baselineRejectionCode": row.get("rejectionCode"),
        }
        for row in selected_rows
    )

    sequential = _run(
        root=root,
        candidates=candidates,
        concurrency=1,
        run_label=str(args.run_label),
    )
    parallel = _run(
        root=root,
        candidates=tuple(reversed(candidates)),
        concurrency=2,
        run_label=str(args.run_label),
    )
    sequential_semantics = {
        key: value["semantic"] for key, value in sequential["results"].items()
    }
    parallel_semantics = {
        key: value["semantic"] for key, value in parallel["results"].items()
    }
    exact_parity = sequential_semantics == parallel_semantics
    semantic_differences = {
        candidate: {
            field: {
                "oneWorker": sequential_semantics[candidate][field],
                "twoWorkers": parallel_semantics[candidate][field],
            }
            for field in sorted(sequential_semantics[candidate])
            if sequential_semantics[candidate][field]
            != parallel_semantics[candidate][field]
        }
        for candidate in sorted(sequential_semantics)
        if sequential_semantics[candidate] != parallel_semantics[candidate]
    }
    speedup = (
        float(sequential["wallSeconds"]) / float(parallel["wallSeconds"])
        if float(parallel["wallSeconds"]) > 0
        else 0.0
    )
    evidence: dict[str, Any] = {
        "schemaVersion": 1,
        "kind": "local_finalist_concurrency_benchmark",
        "runLabel": str(args.run_label),
        "sourceBatchId": str(args.batch_id),
        "sourceBatchRequestHash": canonical_sha256(batch.get("request") or {}),
        "corpus": list(candidates),
        "sequential": sequential,
        "parallel": parallel,
        "result": {
            "exactSemanticParity": exact_parity,
            "semanticDifferences": semantic_differences,
            "speedup": round(speedup, 6),
            "parallelFaster": bool(
                float(parallel["wallSeconds"]) < float(sequential["wallSeconds"])
            ),
            "recommendedFinalistConcurrency": (
                2 if exact_parity and speedup > 1.0 else 1
            ),
        },
        "proofBoundary": (
            "Local measured finalist throughput and exact normalized-result "
            "parity on a bounded frozen corpus. It is not a production capacity "
            "claim or professional correctness attestation."
        ),
    }
    evidence["sha256"] = canonical_sha256(evidence)
    _write_json(output, evidence)
    print(json.dumps({
        "output": output.as_posix(),
        "candidateCount": len(candidates),
        "sequentialWallSeconds": sequential["wallSeconds"],
        "parallelWallSeconds": parallel["wallSeconds"],
        **evidence["result"],
        "sha256": evidence["sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
