"""Governed evidence for browser-visible Research live-result latency."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from baloot_platform.config import build_bundle
from baloot_platform.research.scenario_contracts import canonical_sha256


EVIDENCE_KIND = "research_live_result_visibility_latency_v1"
MEASUREMENT_KIND = "browser_sse_live_result_visibility_v1"
BUDGET_FIELD = "research_live_result_visibility_p95_ms"
MINIMUM_SAMPLE_COUNT = 5
PERCENTILE_METHOD = "linear_interpolation_v1"
TRANSPORT = "server_sent_events_v1"
UI_PATH = "/foundry?section=research"
EVENT_PATH_TEMPLATE = "/api/v2/research-runs/{runCode}/events"
VISIBLE_SIGNAL = "campaign_progress.provisional_result"


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _validated_samples(measurement: Mapping[str, Any]) -> list[float]:
    expected = {
        "schemaVersion",
        "measurementKind",
        "capturedAt",
        "transport",
        "uiPath",
        "eventPathTemplate",
        "visibleSignal",
        "browser",
        "latenciesMilliseconds",
    }
    if set(measurement) != expected:
        raise ValueError("live-result latency measurement fields are invalid")
    if measurement["schemaVersion"] != 1:
        raise ValueError("live-result latency measurement schema is unsupported")
    expected_values = {
        "measurementKind": MEASUREMENT_KIND,
        "transport": TRANSPORT,
        "uiPath": UI_PATH,
        "eventPathTemplate": EVENT_PATH_TEMPLATE,
        "visibleSignal": VISIBLE_SIGNAL,
    }
    for field, expected_value in expected_values.items():
        if measurement[field] != expected_value:
            raise ValueError(f"live-result latency {field} is unsupported")
    if not isinstance(measurement["capturedAt"], str) or not measurement["capturedAt"]:
        raise ValueError("live-result latency capturedAt is required")
    browser = measurement["browser"]
    if (
        not isinstance(browser, Mapping)
        or set(browser) != {"name", "version"}
        or browser["name"] != "chrome"
        or not isinstance(browser["version"], str)
        or not browser["version"]
    ):
        raise ValueError("live-result latency browser identity is invalid")
    raw_samples = measurement["latenciesMilliseconds"]
    if not isinstance(raw_samples, list) or len(raw_samples) < MINIMUM_SAMPLE_COUNT:
        raise ValueError(
            f"live-result latency requires at least {MINIMUM_SAMPLE_COUNT} samples"
        )
    samples: list[float] = []
    for value in raw_samples:
        if type(value) not in {int, float}:
            raise ValueError("live-result latency samples must be numeric")
        sample = float(value)
        if not math.isfinite(sample) or sample < 0:
            raise ValueError("live-result latency samples must be finite and non-negative")
        samples.append(sample)
    return samples


def build_live_result_latency_receipt(
    root: Path,
    *,
    bundle_path: Path,
    measurement: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind browser samples to the selected bundle and its performance budget."""

    samples = _validated_samples(measurement)
    if not source:
        raise ValueError("live-result latency source identity is required")
    bundle = build_bundle(root, bundle_path=bundle_path)
    performance = bundle.components["performance"]
    budgets = dict(performance.data.get("budgets") or {})
    if BUDGET_FIELD not in budgets:
        raise ValueError(f"performance bundle is missing {BUDGET_FIELD}")
    maximum_p95_ms = float(budgets[BUDGET_FIELD])
    if not math.isfinite(maximum_p95_ms) or maximum_p95_ms <= 0:
        raise ValueError(f"performance bundle {BUDGET_FIELD} must be positive")
    p95_ms = round(_percentile(samples, 0.95), 6)
    core = {
        "schemaVersion": 1,
        "evidenceKind": EVIDENCE_KIND,
        "status": "passed" if p95_ms <= maximum_p95_ms else "failed",
        "source": copy.deepcopy(dict(source)),
        "configuration": {
            "bundleId": str(bundle.bundle_profile["profile"]["id"]),
            "bundleVersion": str(bundle.bundle_profile["profile"]["version"]),
            "bundleSha256": bundle.sha256,
            "performanceProfilePath": performance.path,
            "performanceProfileId": performance.id,
            "performanceProfileVersion": performance.version,
            "performanceProfileSha256": performance.sha256,
        },
        "budget": {
            "field": BUDGET_FIELD,
            "maximumP95Milliseconds": maximum_p95_ms,
        },
        "measurement": {
            **copy.deepcopy(dict(measurement)),
            "sampleCount": len(samples),
            "p50Milliseconds": round(_percentile(samples, 0.50), 6),
            "p95Milliseconds": p95_ms,
            "minimumMilliseconds": round(min(samples), 6),
            "maximumMilliseconds": round(max(samples), 6),
            "percentileMethod": PERCENTILE_METHOD,
        },
    }
    return {**core, "evidenceSha256": canonical_sha256(core)}


def validate_live_result_latency_receipt(
    root: Path,
    *,
    bundle_path: Path,
    receipt: Mapping[str, Any],
) -> None:
    """Fail closed when a retained receipt drifts from its samples or bundle."""

    expected_fields = {
        "schemaVersion",
        "evidenceKind",
        "status",
        "source",
        "configuration",
        "budget",
        "measurement",
        "evidenceSha256",
    }
    if set(receipt) != expected_fields:
        raise ValueError("live-result latency receipt fields are invalid")
    payload = {key: copy.deepcopy(value) for key, value in receipt.items() if key != "evidenceSha256"}
    if receipt["evidenceSha256"] != canonical_sha256(payload):
        raise ValueError("live-result latency receipt hash mismatch")
    measurement = dict(receipt["measurement"])
    derived_fields = {
        "sampleCount",
        "p50Milliseconds",
        "p95Milliseconds",
        "minimumMilliseconds",
        "maximumMilliseconds",
        "percentileMethod",
    }
    raw_measurement = {
        key: value for key, value in measurement.items() if key not in derived_fields
    }
    rebuilt = build_live_result_latency_receipt(
        root,
        bundle_path=bundle_path,
        measurement=raw_measurement,
        source=dict(receipt["source"]),
    )
    if dict(receipt) != rebuilt:
        raise ValueError("live-result latency receipt does not match its governed inputs")
