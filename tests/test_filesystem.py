from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from baloot_platform import filesystem


GOVERNED_TEXT_WRITERS = (
    "scripts/benchmark_ortools_research_v4.py",
    "scripts/benchmark_research_scenario_v4.py",
    "scripts/benchmark_search.py",
    "scripts/benchmark_strength.py",
    "scripts/build_reproducible_wheel.py",
    "scripts/generate_media_probe.py",
    "scripts/generate_ortools_v4_five_video_evidence.py",
    "scripts/generate_performance_evidence.py",
    "scripts/generate_platform_evidence.py",
    "scripts/generate_professional_video_prototypes.py",
    "scripts/generate_rules_candidate.py",
    "scripts/generate_scene_preview.py",
    "scripts/generate_security_report.py",
    "scripts/generate_test_inventory_v20.py",
    "scripts/generate_visual_qa.py",
    "scripts/package_release.py",
    "scripts/verify_professional_decision_clip.py",
    "scripts/verify_professional_flagship.py",
    "scripts/verify_professional_video_prototypes.py",
    "scripts/verify_release.py",
    "scripts/verify_remotion_plugin_parity.py",
)


def test_atomic_write_text_replaces_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "evidence.json"
    target.write_text("old", encoding="utf-8")

    filesystem.atomic_write_text(target, "new\n")

    assert target.read_text(encoding="utf-8") == "new\n"
    assert not list(tmp_path.glob(".evidence.json.*.tmp"))


def test_atomic_write_text_raises_after_one_failed_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "evidence.json"
    target.write_text("old", encoding="utf-8")
    attempts = 0

    def replace(source: Path, destination: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts > 1:
            pytest.fail("atomic replacement attempted more than once")
        raise PermissionError(errno.EACCES, "sharing violation", source, 32, destination)

    monkeypatch.setattr(os, "replace", replace)

    with pytest.raises(PermissionError, match="Close any program using it") as error:
        filesystem.atomic_write_text(target, "complete\n")

    assert attempts == 1
    assert error.value.errno == errno.EACCES
    assert error.value.filename is None
    assert error.value.filename2 is None
    assert target.read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".evidence.json.*.tmp"))


def test_governed_release_text_writers_use_atomic_publication(root: Path) -> None:
    for relative in GOVERNED_TEXT_WRITERS:
        source = (root / relative).read_text(encoding="utf-8")
        assert "atomic_write_text" in source, relative
        assert ".write_text(" not in source, relative
