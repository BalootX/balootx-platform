from __future__ import annotations

import copy
from pathlib import Path

import pytest

from baloot_platform.research.live_result_latency import (
    BUDGET_FIELD,
    EVENT_PATH_TEMPLATE,
    MEASUREMENT_KIND,
    TRANSPORT,
    UI_PATH,
    VISIBLE_SIGNAL,
    build_live_result_latency_receipt,
    validate_live_result_latency_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "configs/bundle_ortools_research_native_v3_1.yaml"


def _measurement(latencies: list[float]) -> dict:
    return {
        "schemaVersion": 1,
        "measurementKind": MEASUREMENT_KIND,
        "capturedAt": "2026-09-02T00:00:00.000Z",
        "transport": TRANSPORT,
        "uiPath": UI_PATH,
        "eventPathTemplate": EVENT_PATH_TEMPLATE,
        "visibleSignal": VISIBLE_SIGNAL,
        "browser": {"name": "chrome", "version": "140.0.0.0"},
        "latenciesMilliseconds": latencies,
    }


def test_live_result_latency_receipt_binds_samples_to_v3_bundle_budget() -> None:
    receipt = build_live_result_latency_receipt(
        ROOT,
        bundle_path=BUNDLE,
        measurement=_measurement([10.0, 20.0, 30.0, 40.0, 50.0]),
        source={"binding": "test", "revision": "a" * 40},
    )

    assert receipt["status"] == "passed"
    assert receipt["budget"] == {
        "field": BUDGET_FIELD,
        "maximumP95Milliseconds": 1000.0,
    }
    assert receipt["measurement"]["sampleCount"] == 5
    assert receipt["measurement"]["p95Milliseconds"] == 48.0
    assert receipt["configuration"]["bundleId"] == (
        "baloot_ortools_research_native_v3"
    )
    validate_live_result_latency_receipt(
        ROOT,
        bundle_path=BUNDLE,
        receipt=receipt,
    )


def test_live_result_latency_receipt_fails_closed_for_budget_and_tampering() -> None:
    receipt = build_live_result_latency_receipt(
        ROOT,
        bundle_path=BUNDLE,
        measurement=_measurement([1001.0] * 5),
        source={"binding": "test", "revision": "b" * 40},
    )
    assert receipt["status"] == "failed"

    tampered = copy.deepcopy(receipt)
    tampered["measurement"]["p95Milliseconds"] = 1.0
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_live_result_latency_receipt(
            ROOT,
            bundle_path=BUNDLE,
            receipt=tampered,
        )


def test_live_result_latency_measurement_requires_five_browser_samples() -> None:
    with pytest.raises(ValueError, match="at least 5 samples"):
        build_live_result_latency_receipt(
            ROOT,
            bundle_path=BUNDLE,
            measurement=_measurement([10.0] * 4),
            source={"binding": "test", "revision": "c" * 40},
        )
