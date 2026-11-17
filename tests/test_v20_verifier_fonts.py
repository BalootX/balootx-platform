from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "scripts/verifier_preflight_v20.py"


def _load_preflight():
    spec = importlib.util.spec_from_file_location(
        "verifier_preflight_v20_tests", PREFLIGHT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_font_has_actionable_bounded_failure(tmp_path: Path) -> None:
    preflight = _load_preflight()

    result = preflight.font_check(
        "requiredArabic",
        "DefinitelyMissingArabic.ttf",
        preflight.ARABIC_SAMPLE,
        roots=(tmp_path,),
    )

    assert result["status"] == "failed"
    assert result["filename"] == "DefinitelyMissingArabic.ttf"
    assert "required font is unavailable" in result["error"]
    assert len(result["error"].encode("utf-8")) <= preflight.OUTPUT_LIMIT


def test_supported_verifier_fonts_have_required_glyphs() -> None:
    preflight = _load_preflight()
    missing_files = [
        filename
        for filename, _ in preflight.FONT_REQUIREMENTS.values()
        if preflight.resolve_font(filename) is None
    ]
    if missing_files:
        pytest.skip(
            "positive prerequisite proof runs in the supported verifier image; "
            f"host is missing {missing_files}"
        )

    results = [
        preflight.font_check(name, filename, sample)
        for name, (filename, sample) in preflight.FONT_REQUIREMENTS.items()
    ]

    assert all(result["status"] == "passed" for result in results), results
    assert all(result["missingGlyphs"] == [] for result in results)


def test_preflight_report_fails_before_generation_when_font_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preflight = _load_preflight()
    monkeypatch.setattr(preflight, "TOOLS", {})
    monkeypatch.setattr(preflight, "PYTHON_PACKAGES", ())
    monkeypatch.setattr(
        preflight,
        "FONT_REQUIREMENTS",
        {"missingArabic": ("MissingArabic.ttf", preflight.ARABIC_SAMPLE)},
    )
    monkeypatch.setenv("BALOOT_VERIFY_FONT_ROOT", str(tmp_path))

    report = preflight.build_report()

    assert report["status"] == "failed"
    assert report["failures"][0]["category"] == "font"
    assert report["failures"][0]["name"] == "missingArabic"
    assert report["policy"]["networkInstallationDuringVerification"] is False


def test_missing_tool_is_reported_without_running_a_shell() -> None:
    preflight = _load_preflight()

    result = preflight.command_check(
        "missing", ["balootx-definitely-missing-tool", "--version"]
    )

    assert result["status"] == "failed"
    assert "required executable is unavailable" in result["error"]


def test_command_timeout_is_bounded_and_configurable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BALOOT_VERIFY_COMMAND_TIMEOUT_SECONDS", "17")

    preflight = _load_preflight()

    assert preflight.COMMAND_TIMEOUT_SECONDS == 17
