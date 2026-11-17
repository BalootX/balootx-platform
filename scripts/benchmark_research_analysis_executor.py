from __future__ import annotations

import argparse
import json
import time

from baloot_platform.challenge.analysis_pipeline import (
    DecisionAnalysisExecutorProfileV1,
    StagedTraceAnalyzerV1,
)
from baloot_platform.challenge.trace import CompleteGameTraceV1
from baloot_platform.service.container import build_container


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark complete Research decision analysis on one retained Laffa."
    )
    parser.add_argument("result_code")
    parser.add_argument("--workers", nargs="+", type=int, default=(1, 2, 4))
    args = parser.parse_args()

    container = build_container()
    try:
        result = container.research.get_result(args.result_code)
        if result is None:
            raise SystemExit(f"unknown Research result: {args.result_code}")
        reference = result["artifacts"]["completeTrace"]["uri"]
        trace = CompleteGameTraceV1.from_dict(
            json.loads(
                container.research.artifact_store.get(reference).decode("utf-8")
            )
        )
        base = container.research.analyzers["screening"]
        measurements = []
        expected_hash = None
        for worker_count in args.workers:
            analyzer = StagedTraceAnalyzerV1(
                base.rules,
                base.convention,
                base.profile,
                match_value_profile=base.match_value_profile,
                critical_threshold=base.critical_threshold,
                minimum_critical_confidence=base.minimum_critical_confidence,
                exact_search_backend=base.exact_search_backend,
                executor_profile=DecisionAnalysisExecutorProfileV1.build(
                    worker_count=worker_count
                ),
            )
            started = time.perf_counter()
            analysis = analyzer.analyze_trace(trace)
            elapsed = time.perf_counter() - started
            if expected_hash is None:
                expected_hash = analysis.sha256
            elif analysis.sha256 != expected_hash:
                raise RuntimeError("worker-count analysis hashes are not identical")
            measurements.append(
                {
                    "workerCount": worker_count,
                    "seconds": elapsed,
                    "decisionCount": len(analysis.decisions),
                    "analysisHash": analysis.sha256,
                }
            )
        print(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "resultCode": args.result_code,
                    "traceHash": trace.sha256,
                    "measurements": measurements,
                },
                sort_keys=True,
            )
        )
    finally:
        container.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
