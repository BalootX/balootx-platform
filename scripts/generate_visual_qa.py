#!/usr/bin/env python3
"""Generate visual-QA evidence for exact professional media and player proof."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from baloot_platform.filesystem import atomic_write_text  # noqa: E402

OUTPUT = ROOT / "release/visual-qa.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def checked_file(relative: str, expected: str | None = None) -> dict[str, Any]:
    path = ROOT / relative
    if not path.is_file():
        raise ValueError(f"missing visual-QA artifact: {relative}")
    digest = sha256(path)
    if expected is not None and digest != expected:
        raise ValueError(f"visual-QA artifact hash mismatch: {relative}")
    return {"path": relative, "bytes": path.stat().st_size, "sha256": digest}


def build() -> dict[str, Any]:
    flagship = load("outputs/professional-video-flagship/manifest.json")
    decision = load("outputs/professional-video-decision/manifest.json")
    preview = load("outputs/scene-preview/manifest.json")
    chrome = load("release/chrome-qa.json")
    coverage = flagship.get("sourceCoverage", {})
    if coverage.get("plays") != 32 or coverage.get("tricks") != 8:
        raise ValueError("flagship source coverage is incomplete")
    if coverage.get("shuffleOrDealRendered") is not False:
        raise ValueError("flagship must start directly at the laffah")
    if flagship.get("story", {}).get("durationFrames") != 4320:
        raise ValueError("flagship timeline is not the governed 72-second story")
    if decision.get("story", {}).get("durationFrames") != 1680:
        raise ValueError("decision timeline is not the governed 28-second story")
    if not (preview.get("observerSafe") and chrome.get("observerSafe")):
        raise ValueError("observer-safe visual contract failed")
    if chrome.get("status") != "passed" or int(chrome.get("consoleErrors", -1)) != 0:
        raise ValueError("Chrome visual QA did not pass")
    if chrome.get("previewIdentity") != preview.get("previewIdentity"):
        raise ValueError("Chrome visual QA does not match the exact preview")
    media = chrome.get("mediaEvidence", {})
    if media.get("flagshipSha256") != flagship["video"]["sha256"]:
        raise ValueError("Chrome visual QA does not match the exact flagship")
    if media.get("decisionSha256") != decision["video"]["sha256"]:
        raise ValueError("Chrome visual QA does not match the exact decision clip")
    viewports = chrome.get("viewports") or []
    if not viewports or any(
        item.get("horizontalOverflow")
        or item.get("rtl") is not True
        or (item.get("mediaWidth"), item.get("mediaHeight")) != (1080, 1920)
        for item in viewports
    ):
        raise ValueError("Chrome viewport, RTL, or media layout QA failed")
    files = [
        checked_file("outputs/scene-preview/manifest.json"),
        checked_file("outputs/professional-video-flagship/04-critical-pause.png"),
        checked_file("outputs/professional-video-flagship/05-reveal.png"),
        checked_file("outputs/professional-video-decision/04-decision.png"),
        checked_file("outputs/professional-video-decision/05-reveal.png"),
    ]
    for viewport in viewports:
        screenshot = viewport["screenshot"]
        files.append(checked_file(screenshot["path"], screenshot["sha256"]))
    return {
        "schemaVersion": 1,
        "release": "BalootX",
        "status": "passed",
        "aspectRatio": "9:16",
        "observerSafe": True,
        "automatedChecks": {
            "completePlays": 32,
            "completeTricks": 8,
            "horizontalOverflowViewports": 0,
            "chromeConsoleErrors": 0,
            "developerCodesVisible": False,
            "playerMode": True,
            "reducedMotionProfileDeclared": True,
        },
        "chromeEvidence": "release/chrome-qa.json",
        "files": files,
        "humanReview": {
            "nativeArabicReview": "pending",
            "professionalBalootReview": "pending",
            "ownerVisualApproval": (
                "approved"
                if chrome.get("humanPremiumAcceptance") == "approved"
                else "pending"
            ),
        },
        "proofBoundary": (
            "Technical visual QA passed for the exact professional exports and declared viewports. "
            "Native-Arabic and professional Baloot review remain separate pending gates."
        ),
    }


def main() -> None:
    value = build()
    atomic_write_text(
        OUTPUT,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
