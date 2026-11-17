from __future__ import annotations

import copy
import ctypes
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
from collections import Counter
from typing import Any, Iterable, Mapping

from .sequential_candidate_search import SequentialCandidateSearchReceiptV1
from .fenced_occurrence import build_fenced_occurrence


_TIMING_FIELDS = {
    "construction": "constructionSeconds",
    "simulation": "simulationSeconds",
    "parity": "semanticParitySeconds",
    "analysis": "analysisSeconds",
    "tactical": "tacticalVerificationSeconds",
    "totalWall": "shardWallWorkSeconds",
}

_MILESTONE_FIELDS = {
    "firstFeasible": "firstFeasibleCandidateSeconds",
    "firstOptimal": "firstOptimalCandidateSeconds",
    "firstQualified": "firstQualifiedResultSeconds",
}

CALLING_THREAD_CPU_BOUNDARY = "calling_os_worker_thread_cpu_v1"
PROCESS_PEAK_RSS_BOUNDARY = "process_lifetime_peak_rss_native_inclusive_v1"

_SEQUENTIAL_CONTINUATION_STATUSES = ("optimal", "infeasible")
_SEQUENTIAL_STOP_REASONS = (
    "candidates_exhausted",
    "maximum_solutions_reached",
    "qualified",
    "incomplete",
    "cancelled",
    "fatal_quarantine",
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_V4_PERFORMANCE_RECEIPT_FIELDS = frozenset(
    {
        "schemaVersion",
        "measurementScope",
        "runCode",
        "executionKind",
        "generationLane",
        "attemptedSeeds",
        "attemptedCandidates",
        "structuralCandidates",
        "auctionFeasible",
        "solverOptimal",
        "solverIncomplete",
        "runtimeParityPassed",
        "semanticReplayPassed",
        "proofCompleted",
        "resultCount",
        "qualifiedCount",
        "rejectedCount",
        "quarantineCount",
        "resultCodes",
        "distinctQualifiedResultHashes",
        "qualifiedScenarioHashes",
        "funnel",
        "constructionSeconds",
        "simulationSeconds",
        "semanticParitySeconds",
        "analysisSeconds",
        "tacticalVerificationSeconds",
        "candidatePipelineSeconds",
        "stageTimingBoundary",
        "firstFeasibleCandidateSeconds",
        "firstOptimalCandidateSeconds",
        "firstQualifiedResultSeconds",
        "shardWallWorkSeconds",
        "cpuSeconds",
        "cpuMeasurementBoundary",
        "peakProcessRssBytes",
        "peakRssMeasurementBoundary",
        "environment",
        "cacheScopes",
        "proofResources",
        "cancellationObserved",
        "terminalStatus",
        "sha256",
    }
)
_V4_PERFORMANCE_FUNNEL_FIELDS = frozenset(
    {
        "attempted",
        "structurallyFeasible",
        "legalReplay",
        "semanticSatisfied",
        "proofCompleted",
        "qualified",
    }
)
_V4_PARTIAL_PERFORMANCE_RECEIPT_FIELDS = frozenset(
    {
        "schemaVersion",
        "measurementScope",
        "runCode",
        "executionKind",
        "generationLane",
        "terminalStatus",
        "terminalReason",
        "evidenceCompleteness",
        "attemptedSeeds",
        "attemptedCandidates",
        "structuralCandidates",
        "auctionFeasible",
        "solverOptimal",
        "solverIncomplete",
        "proofCompleted",
        "resultCount",
        "qualifiedCount",
        "rejectedCount",
        "quarantineCount",
        "resultCodes",
        "funnel",
        "constructionSeconds",
        "simulationSeconds",
        "semanticParitySeconds",
        "analysisSeconds",
        "tacticalVerificationSeconds",
        "candidatePipelineSeconds",
        "stageTimingBoundary",
        "firstFeasibleCandidateSeconds",
        "firstOptimalCandidateSeconds",
        "firstQualifiedResultSeconds",
        "shardWallWorkSeconds",
        "cpuSeconds",
        "cpuMeasurementBoundary",
        "peakProcessRssBytes",
        "peakRssMeasurementBoundary",
        "environment",
        "cacheScopes",
        "proofResources",
        "cancellationObserved",
        "sha256",
    }
)
_V4_PARTIAL_TERMINAL_REASONS = frozenset(
    {
        "cancelled_before_candidate",
        "cancelled_between_candidates",
        "cancelled_before_result_persistence",
        "cancelled_after_candidate",
        "cancelled_after_proof",
        "execution_failed",
        "resource_exhausted",
    }
)
_SEQUENTIAL_RECEIPT_ROW_FIELDS = (
    "schemaVersion",
    "occurrenceHash",
    "receiptHash",
    "runCode",
    "shardId",
    "attemptCount",
    "fencingToken",
    "seed",
    "admitted",
    "createdAt",
    "receiptBundle",
)
_SEQUENTIAL_RECEIPT_PAGE_FIELDS = (
    "items",
    "total",
    "offset",
    "limit",
    "nextOffset",
    "hasMore",
)


def _required_nonempty_text(value: Any, *, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"sequential search receipt row requires a {field}")
    return value


def _required_sha256(value: Any, *, field: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"sequential search receipt row requires a {field}")
    return value


def _required_nonnegative_integer(value: Any, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _required_positive_integer(value: Any, *, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _validated_sequential_search_receipt_row(
    row: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], tuple[str, int]]:
    if not isinstance(row, Mapping):
        raise ValueError("sequential search receipt row must be a mapping")
    document = copy.deepcopy(dict(row))
    if type(document.get("admitted")) is not bool or not document["admitted"]:
        raise ValueError("sequential search receipt row must be admitted")
    schema_version = document.get("schemaVersion")
    if type(schema_version) is not int or schema_version != 1:
        raise ValueError("sequential search receipt row schemaVersion must be 1")
    occurrence_hash = _required_sha256(
        document.get("occurrenceHash"), field="occurrenceHash"
    )
    receipt_hash = _required_sha256(document.get("receiptHash"), field="receiptHash")
    run_code = _required_nonempty_text(document.get("runCode"), field="runCode")
    shard_id = _required_nonempty_text(document.get("shardId"), field="shardId")
    attempt_count = _required_positive_integer(
        document.get("attemptCount"), field="sequential receipt attemptCount"
    )
    fencing_token = _required_positive_integer(
        document.get("fencingToken"), field="sequential receipt fencingToken"
    )
    seed = _required_nonnegative_integer(
        document.get("seed"), field="sequential receipt seed"
    )
    if seed > (1 << 63) - 1:
        raise ValueError("sequential receipt seed exceeds the database integer range")
    _required_nonempty_text(document.get("createdAt"), field="createdAt")
    raw_bundle = document.get("receiptBundle")
    if not isinstance(raw_bundle, Mapping):
        raise ValueError("sequential search receipt row requires a receiptBundle")
    verified = SequentialCandidateSearchReceiptV1.from_dict(
        copy.deepcopy(dict(raw_bundle))
    ).verify()
    bundle = verified.to_dict()
    if receipt_hash != verified.sha256:
        raise ValueError("sequential search receipt row hash linkage is invalid")
    if bundle["rootSeed"] != seed:
        raise ValueError("sequential search receipt row seed linkage is invalid")
    expected_occurrence = build_fenced_occurrence(
        kind="sequential_search_receipt",
        semantic_hash=receipt_hash,
        run_code=run_code,
        shard_id=shard_id,
        attempt_count=attempt_count,
        fencing_token=fencing_token,
        seed=seed,
    )
    if occurrence_hash != expected_occurrence["occurrenceHash"]:
        raise ValueError("sequential search receipt occurrence linkage is invalid")
    document["receiptBundle"] = bundle
    return document, bundle, (run_code, seed)


def _sequential_search_receipt_summary_from_bundle(
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    continuation_statuses = {status: 0 for status in _SEQUENTIAL_CONTINUATION_STATUSES}
    feedback_rows = list(bundle["rejectionFeedbackRows"])
    applied = 0
    unapplied = 0
    terminal_application_hashes: set[str] = set()
    for feedback in feedback_rows:
        is_applied = feedback["appliedToSubsequentSolve"]
        application = feedback["applicationReceipt"]
        if is_applied:
            applied += 1
            status = application["solverStatus"]
            if status not in continuation_statuses:
                raise ValueError("unknown sequential continuation solver status")
            continuation_statuses[status] += 1
            if application["solveEvidenceKind"] == (
                "post_rejection_continuation_receipt"
            ):
                terminal_application_hashes.add(application["solveEvidenceHash"])
        else:
            unapplied += 1

    terminal = bundle["terminalContinuationReceipt"]
    if terminal is not None and terminal["sha256"] not in terminal_application_hashes:
        status = terminal["solverStatus"]
        if status not in continuation_statuses:
            raise ValueError("unknown sequential continuation solver status")
        continuation_statuses[status] += 1

    stop_reason = bundle["stopReason"]
    if stop_reason not in _SEQUENTIAL_STOP_REASONS:
        raise ValueError("unknown sequential candidate stop reason")
    stop_reasons = {reason: 0 for reason in _SEQUENTIAL_STOP_REASONS}
    stop_reasons[stop_reason] = 1
    return {
        "admittedSeedCount": 1,
        "exactRejectionCount": len(feedback_rows),
        "noGoodAppliedCount": applied,
        "noGoodUnappliedCount": unapplied,
        "continuationSolveCount": sum(continuation_statuses.values()),
        "continuationStatusDistribution": continuation_statuses,
        "stopReasonDistribution": stop_reasons,
    }


def build_sequential_search_receipt_summary(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive one admitted seed's counters from its verified canonical receipt."""

    _, bundle, _ = _validated_sequential_search_receipt_row(row)
    return _sequential_search_receipt_summary_from_bundle(bundle)


def build_sequential_search_receipt_item(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a verified canonical receipt occurrence with its derived summary."""

    document, bundle, _ = _validated_sequential_search_receipt_row(row)
    return {
        **{key: document[key] for key in _SEQUENTIAL_RECEIPT_ROW_FIELDS},
        "summary": _sequential_search_receipt_summary_from_bundle(bundle),
    }


def build_sequential_search_receipt_page(
    page: Mapping[str, Any],
    *,
    run_code: str,
    offset: int,
    limit: int,
) -> dict[str, Any]:
    """Validate a complete public receipt page without coercing store metadata."""

    if not isinstance(page, Mapping) or set(page) != set(
        _SEQUENTIAL_RECEIPT_PAGE_FIELDS
    ):
        raise ValueError("sequential search receipt page shape is invalid")
    if type(run_code) is not str or not run_code.strip():
        raise ValueError(
            "sequential search receipt run code must be a non-empty string"
        )
    requested_offset = _required_nonnegative_integer(
        offset, field="sequential receipt requested offset"
    )
    requested_limit = _required_positive_integer(
        limit, field="sequential receipt requested limit"
    )
    items = page["items"]
    if type(items) is not list:
        raise ValueError("sequential search receipt page items must be a list")
    total = _required_nonnegative_integer(
        page["total"], field="sequential receipt page total"
    )
    page_offset = _required_nonnegative_integer(
        page["offset"], field="sequential receipt page offset"
    )
    page_limit = _required_positive_integer(
        page["limit"], field="sequential receipt page limit"
    )
    if page_offset != requested_offset or page_limit != requested_limit:
        raise ValueError("sequential search receipt page request metadata is invalid")
    expected_item_count = min(page_limit, max(total - page_offset, 0))
    if len(items) != expected_item_count:
        raise ValueError("sequential search receipt page item count is invalid")
    expected_has_more = page_offset + len(items) < total
    has_more = page["hasMore"]
    if type(has_more) is not bool or has_more is not expected_has_more:
        raise ValueError("sequential search receipt page hasMore is invalid")
    next_offset = page["nextOffset"]
    expected_next_offset = page_offset + len(items) if expected_has_more else None
    if next_offset is not None and type(next_offset) is not int:
        raise ValueError("sequential search receipt page nextOffset is invalid")
    if next_offset != expected_next_offset:
        raise ValueError("sequential search receipt page nextOffset is invalid")
    canonical_items = [build_sequential_search_receipt_item(item) for item in items]
    if any(item["runCode"] != run_code for item in canonical_items):
        raise ValueError("sequential search receipt page run linkage is invalid")
    return {
        "items": canonical_items,
        "total": total,
        "offset": page_offset,
        "limit": page_limit,
        "nextOffset": next_offset,
        "hasMore": has_more,
    }


def build_sequential_search_receipt_rollup(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate admitted per-seed receipts without trusting shard counters."""

    rollup = {
        "admittedSeedCount": 0,
        "exactRejectionCount": 0,
        "noGoodAppliedCount": 0,
        "noGoodUnappliedCount": 0,
        "continuationSolveCount": 0,
        "continuationStatusDistribution": {
            status: 0 for status in _SEQUENTIAL_CONTINUATION_STATUSES
        },
        "stopReasonDistribution": {reason: 0 for reason in _SEQUENTIAL_STOP_REASONS},
    }
    seen_run_seeds: set[tuple[str, int]] = set()
    for row in rows:
        _, bundle, run_seed = _validated_sequential_search_receipt_row(row)
        if run_seed in seen_run_seeds:
            raise ValueError("duplicate admitted sequential search run seed")
        seen_run_seeds.add(run_seed)
        summary = _sequential_search_receipt_summary_from_bundle(bundle)
        for field in (
            "admittedSeedCount",
            "exactRejectionCount",
            "noGoodAppliedCount",
            "noGoodUnappliedCount",
            "continuationSolveCount",
        ):
            rollup[field] += summary[field]
        for status in _SEQUENTIAL_CONTINUATION_STATUSES:
            rollup["continuationStatusDistribution"][status] += summary[
                "continuationStatusDistribution"
            ][status]
        for reason in _SEQUENTIAL_STOP_REASONS:
            rollup["stopReasonDistribution"][reason] += summary[
                "stopReasonDistribution"
            ][reason]
    return rollup


def _nonnegative_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        return None
    return number


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * float(percentile)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * fraction)


def _distribution(
    values: Iterable[float], *, sample_unit: str = "completed_shard"
) -> dict[str, Any]:
    samples = tuple(float(value) for value in values)
    return {
        "sampleCount": len(samples),
        "p50Seconds": _percentile(samples, 0.50),
        "p95Seconds": _percentile(samples, 0.95),
        "totalSeconds": sum(samples) if samples else None,
        "percentileMethod": "linear_interpolation_v1",
        "sampleUnit": sample_unit,
    }


def _availability(status: str, reason: str) -> dict[str, str]:
    return {"status": status, "reason": reason}


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def qualified_scenario_identity_hash_v1(result: Mapping[str, Any]) -> str:
    """Return the stable scenario identity used by V4 performance evidence."""

    if not isinstance(result, Mapping):
        raise ValueError("qualified Research scenario identity requires a result")
    projection = result.get("scenarioProjection")
    if not isinstance(projection, Mapping):
        raise ValueError(
            "qualified Research scenario identity requires a scenario projection"
        )
    scenario_template_id = projection.get("scenarioTemplateId")
    exact_state_id = projection.get("exactStateId")
    if (
        type(scenario_template_id) is not str
        or not scenario_template_id.strip()
        or type(exact_state_id) is not str
        or not exact_state_id.strip()
    ):
        raise ValueError(
            "qualified Research scenario identity requires template and exact-state IDs"
        )
    return _canonical_sha256(
        {
            "schemaVersion": 1,
            "kind": "qualified_research_scenario_identity_v1",
            "scenarioTemplateId": scenario_template_id,
            "exactStateId": exact_state_id,
        }
    )


def validate_v4_performance_receipt(
    value: Mapping[str, Any], *, run_code: str
) -> dict[str, Any]:
    """Validate one closed, hash-bound completed-run performance receipt."""

    if not isinstance(value, Mapping):
        raise ValueError("Research V4 performance receipt must be a mapping")
    receipt = copy.deepcopy(dict(value))
    if set(receipt) != _V4_PERFORMANCE_RECEIPT_FIELDS:
        raise ValueError("Research V4 performance receipt shape is invalid")
    if (
        type(receipt["schemaVersion"]) is not int
        or receipt["schemaVersion"] != 1
        or receipt["measurementScope"] != "research_v4_completed_run_v1"
        or receipt["runCode"] != run_code
    ):
        raise ValueError("Research V4 performance receipt authority is invalid")
    if receipt["executionKind"] not in {"exact_trace_prefix", "scenario_search"}:
        raise ValueError("Research V4 performance receipt execution kind is invalid")
    if type(receipt["generationLane"]) is not str or not receipt["generationLane"]:
        raise ValueError("Research V4 performance receipt generation lane is invalid")

    integer_fields = (
        "attemptedSeeds",
        "attemptedCandidates",
        "structuralCandidates",
        "auctionFeasible",
        "solverOptimal",
        "solverIncomplete",
        "runtimeParityPassed",
        "semanticReplayPassed",
        "proofCompleted",
        "resultCount",
        "qualifiedCount",
        "rejectedCount",
        "quarantineCount",
    )
    if any(
        type(receipt[field]) is not int or receipt[field] < 0
        for field in integer_fields
    ):
        raise ValueError("Research V4 performance receipt counters are invalid")
    if receipt["resultCount"] != sum(
        receipt[field]
        for field in ("qualifiedCount", "rejectedCount", "quarantineCount")
    ):
        raise ValueError("Research V4 performance receipt result counters disagree")

    result_codes = receipt["resultCodes"]
    if (
        type(result_codes) is not list
        or len(result_codes) != receipt["resultCount"]
        or any(type(code) is not str or not code for code in result_codes)
    ):
        raise ValueError(
            "Research V4 performance receipt result-code evidence is invalid"
        )
    qualified_result_hashes = receipt["distinctQualifiedResultHashes"]
    if (
        type(qualified_result_hashes) is not list
        or any(
            type(item) is not str or _SHA256_PATTERN.fullmatch(item) is None
            for item in qualified_result_hashes
        )
        or qualified_result_hashes != sorted(set(qualified_result_hashes))
        or len(qualified_result_hashes) > receipt["qualifiedCount"]
        or (receipt["qualifiedCount"] > 0 and not qualified_result_hashes)
    ):
        raise ValueError(
            "Research V4 performance receipt distinct-result evidence is invalid"
        )
    qualified_scenario_hashes = receipt["qualifiedScenarioHashes"]
    if (
        type(qualified_scenario_hashes) is not list
        or any(
            type(item) is not str or _SHA256_PATTERN.fullmatch(item) is None
            for item in qualified_scenario_hashes
        )
        or qualified_scenario_hashes != sorted(qualified_scenario_hashes)
        or len(qualified_scenario_hashes) != receipt["qualifiedCount"]
    ):
        raise ValueError(
            "Research V4 performance receipt qualified-scenario evidence is invalid"
        )

    funnel = receipt["funnel"]
    if not isinstance(funnel, Mapping) or set(funnel) != _V4_PERFORMANCE_FUNNEL_FIELDS:
        raise ValueError("Research V4 performance receipt funnel shape is invalid")
    if any(type(funnel[field]) is not int or funnel[field] < 0 for field in funnel):
        raise ValueError("Research V4 performance receipt funnel counters are invalid")
    if (
        funnel["attempted"] != receipt["attemptedCandidates"]
        or funnel["proofCompleted"] != receipt["proofCompleted"]
        or funnel["qualified"] != receipt["qualifiedCount"]
        or funnel["structurallyFeasible"] > funnel["attempted"]
        or funnel["legalReplay"] > funnel["attempted"]
        or funnel["semanticSatisfied"] > funnel["legalReplay"]
        or funnel["proofCompleted"] > funnel["legalReplay"]
        or funnel["qualified"]
        > min(funnel["semanticSatisfied"], funnel["proofCompleted"])
    ):
        raise ValueError("Research V4 performance receipt funnel counters disagree")
    if (
        receipt["runtimeParityPassed"] != funnel["legalReplay"]
        or receipt["semanticReplayPassed"] != funnel["legalReplay"]
        or funnel["structurallyFeasible"] > receipt["structuralCandidates"]
        or receipt["auctionFeasible"] > funnel["structurallyFeasible"]
        or receipt["solverOptimal"] + receipt["solverIncomplete"]
        > receipt["structuralCandidates"]
    ):
        raise ValueError("Research V4 performance receipt stage counters disagree")
    if receipt["executionKind"] == "exact_trace_prefix":
        if (
            receipt["generationLane"] != "exact_input"
            or receipt["attemptedSeeds"] != 0
            or receipt["attemptedCandidates"] != 1
            or receipt["structuralCandidates"] != 0
            or receipt["auctionFeasible"] != 0
        ):
            raise ValueError("Research V4 exact performance receipt counters disagree")
    elif (
        receipt["generationLane"] == "exact_input"
        or receipt["attemptedSeeds"] < 1
        or receipt["structuralCandidates"] != receipt["attemptedCandidates"]
        or receipt["auctionFeasible"] < funnel["legalReplay"]
    ):
        raise ValueError("Research V4 search performance receipt counters disagree")

    timing_fields = (
        "constructionSeconds",
        "simulationSeconds",
        "semanticParitySeconds",
        "analysisSeconds",
        "tacticalVerificationSeconds",
        "candidatePipelineSeconds",
        "firstFeasibleCandidateSeconds",
        "firstOptimalCandidateSeconds",
        "firstQualifiedResultSeconds",
        "shardWallWorkSeconds",
        "cpuSeconds",
    )
    if any(
        receipt[field] is not None and _nonnegative_number(receipt[field]) is None
        for field in timing_fields
    ):
        raise ValueError("Research V4 performance receipt timings are invalid")
    if receipt["cpuMeasurementBoundary"] != CALLING_THREAD_CPU_BOUNDARY:
        raise ValueError("Research V4 performance receipt CPU boundary is invalid")
    peak_rss = receipt["peakProcessRssBytes"]
    if peak_rss is not None and (type(peak_rss) is not int or peak_rss < 0):
        raise ValueError("Research V4 performance receipt peak RSS is invalid")
    expected_peak_boundary = PROCESS_PEAK_RSS_BOUNDARY if peak_rss is not None else None
    if receipt["peakRssMeasurementBoundary"] != expected_peak_boundary:
        raise ValueError("Research V4 performance receipt peak RSS boundary is invalid")
    if (
        not isinstance(receipt["environment"], Mapping)
        or not isinstance(receipt["cacheScopes"], Mapping)
        or not isinstance(receipt["stageTimingBoundary"], Mapping)
        or type(receipt["cancellationObserved"]) is not bool
        or type(receipt["terminalStatus"]) is not str
        or not receipt["terminalStatus"]
    ):
        raise ValueError("Research V4 performance receipt evidence is invalid")
    proof_resources = receipt["proofResources"]
    if not isinstance(proof_resources, Mapping) or set(proof_resources) != {
        "nodes",
        "cacheEntries",
        "cacheBytes",
        "hiddenWorlds",
        "aggregation",
    }:
        raise ValueError("Research V4 performance receipt proof resources are invalid")
    if (
        any(
            type(proof_resources[field]) is not int or proof_resources[field] < 0
            for field in ("nodes", "cacheEntries", "cacheBytes", "hiddenWorlds")
        )
        or type(proof_resources["aggregation"]) is not str
        or not proof_resources["aggregation"]
    ):
        raise ValueError("Research V4 performance receipt proof resources are invalid")

    declared_hash = receipt.pop("sha256")
    if type(declared_hash) is not str or declared_hash != _canonical_sha256(receipt):
        raise ValueError("Research V4 performance receipt hash is invalid")
    return {**receipt, "sha256": declared_hash}


def validate_v4_partial_performance_receipt(
    value: Mapping[str, Any], *, run_code: str
) -> dict[str, Any]:
    """Validate hash-bound V4 evidence from a cancelled or incomplete run."""

    if not isinstance(value, Mapping):
        raise ValueError("Research V4 partial performance receipt must be a mapping")
    receipt = copy.deepcopy(dict(value))
    if set(receipt) != _V4_PARTIAL_PERFORMANCE_RECEIPT_FIELDS:
        raise ValueError("Research V4 partial performance receipt shape is invalid")
    if (
        type(receipt["schemaVersion"]) is not int
        or receipt["schemaVersion"] != 1
        or receipt["measurementScope"]
        != "research_v4_cancelled_or_incomplete_run_v1"
        or receipt["runCode"] != run_code
    ):
        raise ValueError("Research V4 partial performance receipt authority is invalid")
    if receipt["executionKind"] not in {"exact_trace_prefix", "scenario_search"}:
        raise ValueError(
            "Research V4 partial performance receipt execution kind is invalid"
        )
    if type(receipt["generationLane"]) is not str or not receipt["generationLane"]:
        raise ValueError(
            "Research V4 partial performance receipt generation lane is invalid"
        )
    terminal_status = receipt["terminalStatus"]
    terminal_reason = receipt["terminalReason"]
    if (
        terminal_status not in {"cancelled", "incomplete"}
        or terminal_reason not in _V4_PARTIAL_TERMINAL_REASONS
        or (terminal_status == "cancelled")
        != str(terminal_reason).startswith("cancelled_")
    ):
        raise ValueError("Research V4 partial performance terminal evidence is invalid")

    integer_fields = (
        "attemptedSeeds",
        "attemptedCandidates",
        "structuralCandidates",
        "auctionFeasible",
        "solverOptimal",
        "solverIncomplete",
        "proofCompleted",
        "resultCount",
        "qualifiedCount",
        "rejectedCount",
        "quarantineCount",
    )
    if any(
        type(receipt[field]) is not int or receipt[field] < 0
        for field in integer_fields
    ):
        raise ValueError("Research V4 partial performance counters are invalid")
    if receipt["resultCount"] != sum(
        receipt[field]
        for field in ("qualifiedCount", "rejectedCount", "quarantineCount")
    ):
        raise ValueError("Research V4 partial performance result counters disagree")
    result_codes = receipt["resultCodes"]
    if (
        type(result_codes) is not list
        or len(result_codes) != receipt["resultCount"]
        or any(type(code) is not str or not code for code in result_codes)
    ):
        raise ValueError("Research V4 partial result-code evidence is invalid")

    funnel = receipt["funnel"]
    if not isinstance(funnel, Mapping) or set(funnel) != _V4_PERFORMANCE_FUNNEL_FIELDS:
        raise ValueError("Research V4 partial performance funnel shape is invalid")
    if any(type(funnel[field]) is not int or funnel[field] < 0 for field in funnel):
        raise ValueError("Research V4 partial performance funnel counters are invalid")
    if (
        funnel["attempted"] != receipt["attemptedCandidates"]
        or funnel["proofCompleted"] != receipt["proofCompleted"]
        or funnel["qualified"] != receipt["qualifiedCount"]
        or funnel["structurallyFeasible"] > funnel["attempted"]
        or funnel["legalReplay"] > funnel["attempted"]
        or funnel["semanticSatisfied"] > funnel["legalReplay"]
        or funnel["proofCompleted"] > funnel["legalReplay"]
        or funnel["qualified"]
        > min(funnel["semanticSatisfied"], funnel["proofCompleted"])
        or funnel["structurallyFeasible"] > receipt["structuralCandidates"]
        or receipt["auctionFeasible"] > funnel["structurallyFeasible"]
        or receipt["solverOptimal"] + receipt["solverIncomplete"]
        > receipt["structuralCandidates"]
    ):
        raise ValueError("Research V4 partial performance counters disagree")
    if receipt["executionKind"] == "exact_trace_prefix":
        if (
            receipt["generationLane"] != "exact_input"
            or receipt["attemptedSeeds"] != 0
            or receipt["attemptedCandidates"] > 1
            or receipt["structuralCandidates"] != 0
            or receipt["auctionFeasible"] != 0
            or receipt["solverOptimal"] != 0
            or receipt["solverIncomplete"] != 0
        ):
            raise ValueError("Research V4 partial exact counters disagree")
    elif (
        receipt["generationLane"] == "exact_input"
        or receipt["structuralCandidates"] != receipt["attemptedCandidates"]
        or receipt["auctionFeasible"] < funnel["legalReplay"]
    ):
        raise ValueError("Research V4 partial search counters disagree")

    timing_fields = (
        "constructionSeconds",
        "simulationSeconds",
        "semanticParitySeconds",
        "analysisSeconds",
        "tacticalVerificationSeconds",
        "candidatePipelineSeconds",
        "firstFeasibleCandidateSeconds",
        "firstOptimalCandidateSeconds",
        "firstQualifiedResultSeconds",
        "shardWallWorkSeconds",
        "cpuSeconds",
    )
    if any(
        receipt[field] is not None and _nonnegative_number(receipt[field]) is None
        for field in timing_fields
    ) or any(receipt[field] is None for field in ("shardWallWorkSeconds", "cpuSeconds")):
        raise ValueError("Research V4 partial performance timings are invalid")
    if receipt["cpuMeasurementBoundary"] != CALLING_THREAD_CPU_BOUNDARY:
        raise ValueError("Research V4 partial performance CPU boundary is invalid")
    peak_rss = receipt["peakProcessRssBytes"]
    if peak_rss is not None and (type(peak_rss) is not int or peak_rss < 0):
        raise ValueError("Research V4 partial performance peak RSS is invalid")
    expected_peak_boundary = PROCESS_PEAK_RSS_BOUNDARY if peak_rss is not None else None
    if receipt["peakRssMeasurementBoundary"] != expected_peak_boundary:
        raise ValueError("Research V4 partial performance peak RSS boundary is invalid")
    if (
        not isinstance(receipt["environment"], Mapping)
        or not isinstance(receipt["cacheScopes"], Mapping)
        or not isinstance(receipt["stageTimingBoundary"], Mapping)
        or type(receipt["cancellationObserved"]) is not bool
        or (terminal_status == "cancelled" and receipt["cancellationObserved"] is not True)
    ):
        raise ValueError("Research V4 partial performance evidence is invalid")

    proof_resources = receipt["proofResources"]
    proof_fields = {
        "nodes",
        "cacheEntries",
        "cacheBytes",
        "hiddenWorlds",
        "aggregation",
        "availability",
    }
    if not isinstance(proof_resources, Mapping) or set(proof_resources) != proof_fields:
        raise ValueError("Research V4 partial proof resources are invalid")
    availability = proof_resources["availability"]
    if availability not in {"not_started", "complete", "partial"}:
        raise ValueError("Research V4 partial proof resource availability is invalid")
    resource_values = [
        proof_resources[field]
        for field in ("nodes", "cacheEntries", "cacheBytes", "hiddenWorlds")
    ]
    if any(
        value is not None and (type(value) is not int or value < 0)
        for value in resource_values
    ):
        raise ValueError("Research V4 partial proof resources are invalid")
    if (
        type(proof_resources["aggregation"]) is not str
        or not proof_resources["aggregation"]
        or (availability == "not_started" and any(value is not None for value in resource_values))
        or (availability == "complete" and any(value is None for value in resource_values))
    ):
        raise ValueError("Research V4 partial proof resources disagree")

    completeness = receipt["evidenceCompleteness"]
    if not isinstance(completeness, Mapping) or set(completeness) != {
        "counters",
        "timings",
        "results",
        "proofResources",
    }:
        raise ValueError("Research V4 partial evidence completeness is invalid")
    if completeness != {
        "counters": "complete_through_terminal_observation_v1",
        "timings": "complete_through_terminal_observation_v1",
        "results": "retained_results_only_v1",
        "proofResources": availability,
    }:
        raise ValueError("Research V4 partial evidence completeness disagrees")

    declared_hash = receipt.pop("sha256")
    if type(declared_hash) is not str or declared_hash != _canonical_sha256(receipt):
        raise ValueError("Research V4 partial performance receipt hash is invalid")
    return {**receipt, "sha256": declared_hash}


def _plugin_id(value: Any, *, fallback: str) -> str:
    descriptor = getattr(value, "descriptor", None)
    plugin_id = str(getattr(descriptor, "id", "") or "").strip()
    return plugin_id or fallback


def build_research_environment_declaration(
    *,
    generation_lane: str,
    configured_shard_workers: int,
    effective_shard_workers: int,
    cp_sat_workers_per_solve: int | None,
    research_store: Any,
    artifact_store: Any,
) -> dict[str, Any]:
    """Return a sanitized, host-comparable declaration without host identity."""

    try:
        ortools_version = importlib.metadata.version("ortools")
    except importlib.metadata.PackageNotFoundError:
        ortools_version = None
    research_store_id = {
        "SQLiteResearchStoreV1": "sqlite_wal",
        "PostgresResearchStoreV1": "postgresql",
    }.get(type(research_store).__name__, "unavailable")
    artifact_store_id = _plugin_id(artifact_store, fallback="unavailable")
    common = {
        "schemaVersion": 1,
        "hardware": {
            "architecture": str(platform.machine() or "unavailable"),
            "processor": str(platform.processor() or "unavailable"),
            "logicalCpuCount": os.cpu_count(),
        },
        "operatingSystem": {
            "system": str(platform.system() or "unavailable"),
            "release": str(platform.release() or "unavailable"),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "ortools": {
            "available": ortools_version is not None,
            "version": ortools_version,
        },
        "workers": {
            "configuredShardWorkers": int(configured_shard_workers),
            "effectiveShardWorkers": int(effective_shard_workers),
        },
        "storage": {
            "researchControlPlane": research_store_id,
            "artifactStorePlugin": artifact_store_id,
        },
        "cpuMeasurementBoundary": CALLING_THREAD_CPU_BOUNDARY,
    }
    comparison_hash = _canonical_sha256(common)
    value = {
        **common,
        "generationLane": str(generation_lane),
        "workers": {
            **dict(common["workers"]),
            "cpSatWorkersPerSolve": (
                None
                if cp_sat_workers_per_solve is None
                else int(cp_sat_workers_per_solve)
            ),
        },
        "comparisonHash": comparison_hash,
    }
    value["sha256"] = _canonical_sha256(value)
    return value


def process_lifetime_peak_rss_bytes() -> int | None:
    """Best-effort native-inclusive process peak RSS using the standard library."""

    if os.name == "nt":
        try:
            from ctypes import wintypes

            class _ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(_ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            handle = kernel32.GetCurrentProcess()
            succeeded = psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            )
            return int(counters.PeakWorkingSetSize) if succeeded else None
        except (AttributeError, OSError, ValueError):
            return None
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if platform.system() == "Darwin" else value * 1024
    except (ImportError, OSError, ValueError):
        return None


def _cache_rollup(completed_results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    scopes: dict[str, list[dict[str, int]]] = {}
    for result in completed_results:
        raw_scopes = result.get("cacheScopes")
        if not isinstance(raw_scopes, Mapping):
            continue
        for scope, evidence in raw_scopes.items():
            if not isinstance(evidence, Mapping):
                continue
            row: dict[str, int] = {}
            valid = True
            for key in ("hits", "misses", "evictions", "entries", "capacity"):
                value = evidence.get(key)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    valid = False
                    break
                row[key] = int(value)
            if valid:
                scopes.setdefault(str(scope), []).append(row)
    return {
        scope: {
            "status": "measured",
            "measurementBoundary": "process_lifetime_cumulative_snapshot_v1",
            "aggregation": "maximum_observed_snapshot_not_additive_v1",
            "sampleCount": len(rows),
            **{
                key: max(row[key] for row in rows)
                for key in ("hits", "misses", "evictions", "entries", "capacity")
            },
            "note": (
                "Cumulative process snapshots are not summed across shards or processes."
            ),
        }
        for scope, rows in sorted(scopes.items())
    }


def build_research_performance_rollup(
    shards: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a deterministic rollup from persisted shard evidence only."""

    ordered = sorted(
        (dict(shard) for shard in shards),
        key=lambda shard: (
            str(shard.get("runCode") or ""),
            int(shard.get("shardIndex") or 0),
            str(shard.get("shardId") or ""),
        ),
    )
    completed_status_shards = [
        shard for shard in ordered if shard.get("status") == "completed"
    ]
    completed_shards = [
        shard
        for shard in completed_status_shards
        if isinstance(shard.get("result"), Mapping)
    ]
    completed_results = [dict(shard["result"]) for shard in completed_shards]

    partial_v4_receipt_count = 0
    for shard in ordered:
        partial_result = shard.get("result")
        if (
            not isinstance(partial_result, Mapping)
            or partial_result.get("measurementScope")
            != "research_v4_cancelled_or_incomplete_run_v1"
        ):
            continue
        validated_partial = validate_v4_partial_performance_receipt(
            partial_result,
            run_code=str(shard.get("runCode") or ""),
        )
        expected_shard_status = (
            "cancelled"
            if validated_partial["terminalStatus"] == "cancelled"
            else "failed"
        )
        if shard.get("status") != expected_shard_status:
            raise ValueError("Research V4 partial performance shard status disagrees")
        partial_v4_receipt_count += 1

    qualified_scenario_hashes: set[str] = set()
    v4_scenario_receipt_count = 0
    v4_scenario_evidence_complete = (
        bool(completed_status_shards)
        and len(completed_shards) == len(completed_status_shards)
        and partial_v4_receipt_count == 0
    )
    for shard, result in zip(completed_shards, completed_results, strict=True):
        if result.get("measurementScope") != "research_v4_completed_run_v1":
            v4_scenario_evidence_complete = False
            continue
        validated = validate_v4_performance_receipt(
            result,
            run_code=str(shard.get("runCode") or ""),
        )
        v4_scenario_receipt_count += 1
        qualified_scenario_hashes.update(validated["qualifiedScenarioHashes"])
    qualified_distinct_scenario_count = (
        len(qualified_scenario_hashes) if v4_scenario_evidence_complete else None
    )

    timing: dict[str, dict[str, Any]] = {}
    for name, field in _TIMING_FIELDS.items():
        values = [
            value
            for result in completed_results
            if (value := _nonnegative_number(result.get(field))) is not None
        ]
        timing[name] = _distribution(values, sample_unit="completed_shard_stage_total")

    milestone_timing: dict[str, dict[str, Any]] = {}
    for name, field in _MILESTONE_FIELDS.items():
        values = [
            value
            for result in completed_results
            if (value := _nonnegative_number(result.get(field))) is not None
        ]
        milestone_timing[name] = _distribution(
            values, sample_unit="completed_shard_elapsed_to_milestone"
        )

    cpu_values = [
        value
        for result in completed_results
        if (value := _nonnegative_number(result.get("cpuSeconds"))) is not None
    ]
    cpu_boundaries = sorted(
        {
            str(result.get("cpuMeasurementBoundary"))
            for result in completed_results
            if result.get("cpuMeasurementBoundary")
        }
    )
    cpu_complete = bool(cpu_values) and len(cpu_values) == len(completed_results)
    cpu_boundary = cpu_boundaries[0] if len(cpu_boundaries) == 1 else None
    cpu_measured = cpu_complete and cpu_boundary is not None
    cpu = {
        **_distribution(cpu_values, sample_unit="completed_shard_calling_thread"),
        "availability": (
            "measured" if cpu_measured else "mixed" if cpu_values else "unavailable"
        ),
        "measurementBoundary": cpu_boundary,
        "nativeWorkOnCallingThreadIncluded": True,
        "auxiliaryThreadsExcluded": True,
        "note": (
            "Calling OS worker-thread CPU includes native work executed on that thread; "
            "work performed by any auxiliary threads is excluded."
            if cpu_measured
            else "Complete CPU values and one declared boundary are unavailable."
        ),
    }

    environment_rows = [
        copy
        for result in completed_results
        if isinstance((copy := result.get("environment")), Mapping)
    ]
    environment_hashes = sorted(
        {str(item.get("sha256")) for item in environment_rows if item.get("sha256")}
    )
    comparison_hashes = sorted(
        {
            str(item.get("comparisonHash"))
            for item in environment_rows
            if item.get("comparisonHash")
        }
    )
    environment_complete = (
        bool(completed_results)
        and len(environment_rows) == len(completed_results)
        and len(comparison_hashes) == 1
    )
    common_declaration: dict[str, Any] | None = None
    if environment_complete:
        common_declaration = dict(environment_rows[0])
        common_declaration.pop("sha256", None)
        common_declaration.pop("comparisonHash", None)
        common_declaration.pop("generationLane", None)
        workers = dict(common_declaration.get("workers") or {})
        workers.pop("cpSatWorkersPerSolve", None)
        common_declaration["workers"] = workers
    environment = {
        "status": (
            "measured"
            if environment_complete
            else "mixed"
            if environment_rows
            else "unavailable"
        ),
        "declaration": (common_declaration),
        "environmentHashes": environment_hashes,
        "comparisonHashes": comparison_hashes,
        "sampleCount": len(environment_rows),
        "note": (
            "One sanitized declaration is shared by every completed shard."
            if environment_complete
            else "Historical, partial, or mixed environment evidence is not guessed."
        ),
    }

    peak_rows = [
        value
        for result in completed_results
        if (value := _nonnegative_number(result.get("peakProcessRssBytes"))) is not None
    ]
    peak_boundaries = sorted(
        {
            str(result.get("peakRssMeasurementBoundary"))
            for result in completed_results
            if result.get("peakRssMeasurementBoundary")
        }
    )
    peak_complete = (
        bool(completed_results)
        and len(peak_rows) == len(completed_results)
        and peak_boundaries == [PROCESS_PEAK_RSS_BOUNDARY]
    )
    peak_memory = {
        "availability": (
            "measured" if peak_complete else "mixed" if peak_rows else "unavailable"
        ),
        "peakProcessRssBytes": int(max(peak_rows)) if peak_rows else None,
        "sampleCount": len(peak_rows),
        "measurementBoundary": (
            PROCESS_PEAK_RSS_BOUNDARY
            if peak_boundaries == [PROCESS_PEAK_RSS_BOUNDARY]
            else None
        ),
        "aggregation": "maximum_observed_process_lifetime_peak_v1",
        "note": (
            "Native-inclusive process-lifetime peak; concurrent and prior process work is included, "
            "so this is not isolated run memory."
            if peak_complete
            else "Complete process peak-RSS evidence is unavailable."
        ),
    }
    cache_scopes = _cache_rollup(completed_results)

    attempted = sum(
        int(result.get("attemptedSeeds", 0)) for result in completed_results
    )
    result_count = sum(
        int(result.get("resultCount", 0)) for result in completed_results
    )
    qualified = sum(
        int(result.get("qualifiedCount", 0)) for result in completed_results
    )
    emitted_codes = [
        str(code)
        for result in completed_results
        for code in (result.get("resultCodes") or [])
    ]
    duplicate_evidence_complete = len(emitted_codes) == result_count
    duplicate_count = (
        result_count - len(set(emitted_codes)) if duplicate_evidence_complete else None
    )
    duplicate_rate = (
        duplicate_count / result_count
        if duplicate_count is not None and result_count > 0
        else (0.0 if duplicate_count == 0 else None)
    )

    statuses = Counter(str(shard.get("status") or "unknown") for shard in ordered)
    shard_executions = sum(int(shard.get("attemptCount", 0)) for shard in ordered)
    repeated_execution_count = sum(
        max(0, int(shard.get("attemptCount", 0)) - 1) for shard in ordered
    )

    construction_statuses = Counter()
    play_statuses = Counter()
    for result in completed_results:
        construction_statuses["OPTIMAL"] += int(result.get("solverOptimal", 0))
        construction_statuses["INCOMPLETE"] += int(result.get("solverIncomplete", 0))
        play_statuses["OPTIMAL"] += int(result.get("playCompilerOptimal", 0))
        play_statuses["FEASIBLE"] += int(result.get("playCompilerFeasible", 0))
        play_statuses["INCOMPLETE"] += int(result.get("playCompilerIncomplete", 0))

    operation_counters_complete = bool(ordered) and all(
        shard.get("retryCount") is not None and shard.get("reclaimCount") is not None
        for shard in ordered
    )
    retry_count = (
        sum(int(shard["retryCount"]) for shard in ordered)
        if operation_counters_complete
        else None
    )
    reclaim_count = (
        sum(int(shard["reclaimCount"]) for shard in ordered)
        if operation_counters_complete
        else None
    )

    wall_values = [
        value
        for result in completed_results
        if (value := _nonnegative_number(result.get("shardWallWorkSeconds")))
        is not None
    ]
    wall_complete = bool(completed_results) and len(wall_values) == len(
        completed_results
    )
    wall_total = timing["totalWall"]["totalSeconds"]
    cpu_total = cpu["totalSeconds"]
    qualified_distinct_per_wall_hour = (
        qualified_distinct_scenario_count * 3600.0 / float(wall_total)
        if qualified_distinct_scenario_count is not None
        and wall_complete
        and wall_total is not None
        and wall_total > 0
        else None
    )
    qualified_distinct_per_cpu_hour = (
        qualified_distinct_scenario_count * 3600.0 / float(cpu_total)
        if qualified_distinct_scenario_count is not None
        and cpu_measured
        and cpu_total is not None
        and cpu_total > 0
        else None
    )
    qualified_scenario_status = (
        "measured"
        if v4_scenario_evidence_complete
        else "mixed"
        if v4_scenario_receipt_count
        else "unavailable"
    )
    if qualified_distinct_per_wall_hour is not None:
        qualified_scenario_wall_status = "measured"
        qualified_scenario_wall_reason = (
            "complete_v4_identity_and_wall_duration_coverage_v1"
        )
    elif qualified_distinct_scenario_count is None:
        qualified_scenario_wall_status = qualified_scenario_status
        qualified_scenario_wall_reason = "scenario_identity_coverage_is_incomplete"
    elif not wall_complete:
        qualified_scenario_wall_status = "mixed"
        qualified_scenario_wall_reason = "wall_duration_coverage_is_incomplete"
    else:
        qualified_scenario_wall_status = "unavailable"
        qualified_scenario_wall_reason = "zero_wall_duration_denominator"

    if qualified_distinct_per_cpu_hour is not None:
        qualified_scenario_cpu_status = "measured"
        qualified_scenario_cpu_reason = (
            "complete_v4_identity_and_calling_thread_cpu_coverage_v1"
        )
    elif qualified_distinct_scenario_count is None:
        qualified_scenario_cpu_status = qualified_scenario_status
        qualified_scenario_cpu_reason = "scenario_identity_coverage_is_incomplete"
    elif not cpu_measured:
        qualified_scenario_cpu_status = "mixed" if cpu_values else "unavailable"
        qualified_scenario_cpu_reason = "cpu_measurement_coverage_is_incomplete"
    else:
        qualified_scenario_cpu_status = "unavailable"
        qualified_scenario_cpu_reason = "zero_cpu_duration_denominator"
    return {
        "schemaVersion": 1,
        "shardCount": len(ordered),
        "completedShardCount": int(statuses["completed"]),
        "attemptedSeeds": attempted,
        "resultCount": result_count,
        "qualifiedCount": qualified,
        "qualifiedDistinctScenarioCount": qualified_distinct_scenario_count,
        "qualifiedDistinctScenariosPerWallHour": qualified_distinct_per_wall_hour,
        "qualifiedDistinctScenariosPerCpuHour": qualified_distinct_per_cpu_hour,
        "qualifiedPer100Attempts": (
            qualified * 100.0 / attempted if attempted > 0 else 0.0
        ),
        "resultPer100Attempts": (
            result_count * 100.0 / attempted if attempted > 0 else 0.0
        ),
        "duplicateCount": duplicate_count,
        "duplicateRate": duplicate_rate,
        "duplicateEvidence": (
            "complete_result_code_evidence_v1"
            if duplicate_evidence_complete
            else "unavailable_or_partial"
        ),
        "timing": timing,
        "milestoneTiming": milestone_timing,
        "cpu": cpu,
        "environment": environment,
        "peakMemory": peak_memory,
        "cacheScopes": cache_scopes,
        "costProxy": {
            "wallSecondsPerQualified": (
                float(wall_total) / qualified
                if wall_total is not None and qualified > 0
                else None
            ),
            "callingOsThreadCpuSecondsPerQualified": (
                float(cpu_total) / qualified
                if cpu_measured and cpu_total is not None and qualified > 0
                else None
            ),
            "monetaryCost": None,
            "note": "Time-per-qualified proxies only; no hardware price evidence is recorded.",
        },
        "operations": {
            "currentQueuedShardCount": int(statuses["queued"]),
            "currentRunningShardCount": int(statuses["running"]),
            "currentCancelledShardCount": int(statuses["cancelled"]),
            "currentFailedShardCount": int(statuses["failed"]),
            "shardExecutionCount": shard_executions,
            "repeatedExecutionCount": repeated_execution_count,
            "retryCount": retry_count,
            "retryAvailability": (
                "measured" if operation_counters_complete else "unavailable_historical"
            ),
            "reclaimCount": reclaim_count,
            "reclaimAvailability": (
                "measured" if operation_counters_complete else "unavailable_historical"
            ),
        },
        "solverStatusDistribution": {
            "construction": {
                key: int(value)
                for key, value in sorted(construction_statuses.items())
                if value
            },
            "playCompiler": {
                key: int(value) for key, value in sorted(play_statuses.items()) if value
            },
            "coverage": "recorded_status_counters_only_v1",
        },
        "measurementAvailability": {
            "qualifiedDistinctScenarios": _availability(
                qualified_scenario_status,
                (
                    "complete_hash_bound_v4_scenario_identity_receipts_v1"
                    if v4_scenario_evidence_complete
                    else "cancelled_or_incomplete_v4_execution_receipts_present"
                    if partial_v4_receipt_count
                    else "legacy_or_partial_completed_result_coverage"
                    if v4_scenario_receipt_count
                    else "no_completed_hash_bound_v4_scenario_identity_receipts"
                ),
            ),
            "qualifiedDistinctScenarioWallRate": _availability(
                qualified_scenario_wall_status,
                qualified_scenario_wall_reason,
            ),
            "qualifiedDistinctScenarioCpuRate": _availability(
                qualified_scenario_cpu_status,
                qualified_scenario_cpu_reason,
            ),
            "hardware": _availability(
                environment["status"],
                "sanitized_environment_declaration_v1"
                if environment_complete
                else "historical_partial_or_mixed",
            ),
            "peakMemory": _availability(
                peak_memory["availability"],
                "process_lifetime_peak_not_isolated_run_memory"
                if peak_complete
                else "historical_or_partial",
            ),
            "cache": _availability(
                "measured" if cache_scopes else "unavailable",
                "named_process_lifetime_cumulative_scopes"
                if cache_scopes
                else "not_recorded_in_shard_evidence",
            ),
            "nativeChildCpu": _availability(
                "excluded",
                "calling-thread measurement excludes work on any auxiliary threads",
            ),
            "retries": _availability(
                "measured" if operation_counters_complete else "unavailable",
                "distinct_persisted_transition_counter"
                if operation_counters_complete
                else "historical_counter_unavailable",
            ),
            "reclaims": _availability(
                "measured" if operation_counters_complete else "unavailable",
                "distinct_persisted_transition_counter"
                if operation_counters_complete
                else "historical_counter_unavailable",
            ),
        },
    }


def build_research_performance_by_lane(
    shards: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    lanes: dict[str, list[dict[str, Any]]] = {}
    for source in shards:
        shard = dict(source)
        result = shard.get("result")
        lane = (
            str(result.get("generationLane"))
            if isinstance(result, Mapping) and result.get("generationLane")
            else "unavailable"
        )
        lanes.setdefault(lane, []).append(shard)
    return {
        lane: build_research_performance_rollup(items)
        for lane, items in sorted(lanes.items())
    }
