from __future__ import annotations

import importlib.util
import stat
import zipfile
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_archive.py"


@pytest.fixture
def verifier():
    spec = importlib.util.spec_from_file_location("current_archive_verifier", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_safe_extraction_accepts_regular_files_and_directories(
    tmp_path: Path, verifier
) -> None:
    archive_path = tmp_path / "safe.zip"
    destination = tmp_path / "extract"
    destination.mkdir()
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("repo/", b"")
        archive.writestr("repo/README.md", b"safe\n")
        archive.writestr("repo/src/module.py", b"VALUE = 1\n")

    with zipfile.ZipFile(archive_path) as archive:
        verifier.extract_safely(archive, destination)

    assert (destination / "repo/README.md").read_bytes() == b"safe\n"
    assert (destination / "repo/src/module.py").read_bytes() == b"VALUE = 1\n"


@pytest.mark.parametrize(
    "entry",
    (
        "../escape.txt",
        "repo/../../escape.txt",
        "/absolute.txt",
        "C:/drive.txt",
        "C:drive.txt",
        "repo\\windows-separator.txt",
    ),
)
def test_unsafe_paths_are_rejected_before_any_file_is_written(
    tmp_path: Path, verifier, entry: str
) -> None:
    archive_path = tmp_path / "unsafe.zip"
    destination = tmp_path / "extract"
    destination.mkdir()
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("repo/safe.txt", "must not be extracted")
        # Preserve the hostile spelling even on hosts where ZipInfo's
        # constructor normalizes Windows separators.
        unsafe = zipfile.ZipInfo("placeholder")
        unsafe.filename = entry
        archive.writestr(unsafe, "unsafe")

    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(verifier.ArchiveVerificationError):
            verifier.extract_safely(archive, destination)

    assert list(destination.rglob("*")) == []


@pytest.mark.parametrize("mode", (stat.S_IFLNK | 0o777, stat.S_IFIFO | 0o644))
def test_links_and_special_files_are_rejected_before_writing(
    tmp_path: Path, verifier, mode: int
) -> None:
    archive_path = tmp_path / "special.zip"
    destination = tmp_path / "extract"
    destination.mkdir()
    entry = zipfile.ZipInfo("repo/special")
    entry.create_system = 3
    entry.external_attr = mode << 16
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("repo/safe.txt", "must not be extracted")
        archive.writestr(entry, "../../outside")

    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(verifier.ArchiveVerificationError):
            verifier.extract_safely(archive, destination)

    assert list(destination.rglob("*")) == []


def test_case_collisions_are_rejected_before_writing(tmp_path: Path, verifier) -> None:
    archive_path = tmp_path / "collision.zip"
    destination = tmp_path / "extract"
    destination.mkdir()
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("repo/File.txt", "one")
        archive.writestr("repo/file.txt", "two")

    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(verifier.ArchiveVerificationError):
            verifier.extract_safely(archive, destination)

    assert list(destination.rglob("*")) == []


def test_expansion_budget_counts_all_entries_before_writing(
    tmp_path: Path, verifier, monkeypatch
) -> None:
    archive_path = tmp_path / "large.zip"
    destination = tmp_path / "extract"
    destination.mkdir()
    monkeypatch.setattr(verifier, "MAX_UNCOMPRESSED_BYTES", 4)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("repo/first.txt", b"123")
        archive.writestr("repo/second.txt", b"45")

    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(verifier.ArchiveVerificationError):
            verifier.extract_safely(archive, destination)

    assert list(destination.rglob("*")) == []
