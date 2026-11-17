#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "release/verification-report.json"
OUTPUT = ROOT / "docs/VERIFICATION_REPORT.md"


def main() -> None:
    data = json.loads(REPORT.read_text(encoding="utf-8"))
    checks = data.get("checks", [])
    passed = sum(1 for check in checks if check.get("status") == "passed")
    lines = [
        "# Verification report",
        "",
        f"- Project: `{data.get('project')}`",
        f"- Version: `{data.get('version')}`",
        f"- Generated: `{data.get('generatedAt')}`",
        f"- Status: **{str(data.get('status', '')).upper()}**",
        f"- Checks passed: **{passed}/{len(checks)}**",
        "",
        "## Verification lanes",
        "",
        "| Check | Status | Duration |",
        "|---|---:|---:|",
    ]
    for check in checks:
        duration = check.get("durationSeconds")
        duration_text = f"{duration:.3f}s" if isinstance(duration, (int, float)) else "—"
        lines.append(f"| `{check.get('name')}` | {check.get('status')} | {duration_text} |")
    lines += ["", "## Release artifacts", "", "| Artifact | Bytes | SHA-256 |", "|---|---:|---|"]
    for path, metadata in sorted(data.get("artifacts", {}).items()):
        lines.append(f"| `{path}` | {metadata.get('bytes', 0):,} | `{metadata.get('sha256', '')}` |")
    lines += [
        "",
        "## Randomized lifecycle and native parity",
        "",
    ]
    property_path = ROOT / "release/property-check.json"
    if property_path.exists():
        prop = json.loads(property_path.read_text(encoding="utf-8"))
        lines += [
            f"- Complete rounds: **{prop.get('rounds')}**",
            f"- Sun / Hokum / Ashkal: **{prop.get('contracts', {}).get('sun')} / {prop.get('contracts', {}).get('hokum')} / {prop.get('contracts', {}).get('ashkal')}**",
            f"- Card plays validated: **{prop.get('plays')}**",
            f"- Native parity positions: **{prop.get('nativePositionsCompared')}**",
            f"- Project-bearing rounds: **{prop.get('projectRounds')}**",
            f"- Native accelerator enabled: **{prop.get('nativeEnabled')}**",
        ]
    lines += [
        "",
        "## Media contract",
        "",
        "The release verifier checks the final MP4 using FFprobe and a full FFmpeg decode. The expected contract is 1080×1920 H.264 video, constant 60 fps, `yuv420p`, BT.709 metadata, and AAC 48 kHz stereo audio.",
        "",
        "## Clean-room package verification",
        "",
        "The exact final source ZIP is extracted into a new directory, its source manifest is verified, the full source verifier is rerun, the wheel is rebuilt, and all release media is regenerated. The exact-package report is distributed beside the ZIP as `baloot-platform-20.0.0-final-package-verification.json`; it is intentionally external to avoid a recursive archive-hash dependency.",
        "",
        "## External validation boundary",
        "",
        "Docker/Compose definitions are statically validated by the release suite. The execution environment used for this release did not provide a Docker daemon, so OCI images could not be physically built or scanned here. Native binaries, the Python wheel, APIs, typed gRPC, SDKs, and video export were executed directly.",
        "",
    ]
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
