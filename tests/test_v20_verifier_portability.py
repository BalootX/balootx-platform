from __future__ import annotations

import importlib.util
import stat
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "scripts/verify_archive_v20.py"
PACKAGE = ROOT / "scripts/package_release_v20.py"
TEST_LANES = ROOT / "scripts/run_test_lanes_v20.py"
VERIFY_RELEASE = ROOT / "scripts/verify_release_v20.py"
SYNC_RESOURCES = ROOT / "scripts/sync_package_resources.py"


def _load_verifier():
    spec = importlib.util.spec_from_file_location(
        "verify_archive_v20_portability", VERIFY
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_test_lanes():
    spec = importlib.util.spec_from_file_location(
        "run_test_lanes_v20_portability", TEST_LANES
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_resource_sync():
    spec = importlib.util.spec_from_file_location(
        "sync_package_resources_portability", SYNC_RESOURCES
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_workspace_cli_override_supports_spaces_and_wins_over_environment(
    tmp_path: Path,
) -> None:
    verifier = _load_verifier()
    cli_parent = tmp_path / "CLI workspace with spaces"
    environment_parent = tmp_path / "environment-workspace"

    resolved = verifier.resolve_workspace_parent(
        str(cli_parent),
        {verifier.WORKSPACE_ENV: str(environment_parent)},
    )

    assert resolved == cli_parent.resolve()
    assert resolved.is_dir()
    assert not environment_parent.exists()


def test_workspace_environment_override_is_used(tmp_path: Path) -> None:
    verifier = _load_verifier()
    environment_parent = tmp_path / "environment workspace"

    resolved = verifier.resolve_workspace_parent(
        None, {verifier.WORKSPACE_ENV: str(environment_parent)}
    )

    assert resolved == environment_parent.resolve()
    assert resolved.is_dir()


def test_workspace_rejects_file_and_symlink(tmp_path: Path) -> None:
    verifier = _load_verifier()
    regular_file = tmp_path / "not-a-directory"
    regular_file.write_text("x", encoding="utf-8")
    with pytest.raises(verifier.ArchiveVerificationError):
        verifier.resolve_workspace_parent(str(regular_file), {})

    link = tmp_path / "workspace-link"
    try:
        link.symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable for this test user")
    with pytest.raises(verifier.ArchiveVerificationError):
        verifier.resolve_workspace_parent(str(link), {})


def test_safe_extraction_accepts_regular_entries(tmp_path: Path) -> None:
    verifier = _load_verifier()
    archive_path = tmp_path / "safe.zip"
    destination = tmp_path / "extract"
    destination.mkdir()
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("repo/README.md", "safe")
        archive.writestr("repo/src/module.py", "VALUE = 1\n")

    with zipfile.ZipFile(archive_path) as archive:
        verifier.extract_archive_safely(archive, destination)

    assert (destination / "repo/README.md").read_text(encoding="utf-8") == "safe"
    assert (destination / "repo/src/module.py").is_file()


@pytest.mark.parametrize(
    "entry",
    (
        "../escape.txt",
        "/absolute.txt",
        "C:/drive.txt",
        "repo\\windows-separator.txt",
    ),
)
def test_safe_extraction_rejects_unsafe_paths_before_writing(
    tmp_path: Path, entry: str
) -> None:
    verifier = _load_verifier()
    archive_path = tmp_path / "unsafe.zip"
    destination = tmp_path / "extract"
    destination.mkdir()
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("repo/safe.txt", "must not be extracted")
        # ZipInfo's Windows constructor rewrites backslashes to forward
        # slashes. Assign after construction so the archive contains the exact
        # hostile name on every host and the verifier receives the same bytes.
        unsafe = zipfile.ZipInfo("placeholder")
        unsafe.filename = entry
        archive.writestr(unsafe, "unsafe")

    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(verifier.ArchiveVerificationError):
            verifier.extract_archive_safely(archive, destination)

    assert list(destination.rglob("*")) == []


def test_safe_extraction_rejects_symbolic_links(tmp_path: Path) -> None:
    verifier = _load_verifier()
    archive_path = tmp_path / "symlink.zip"
    destination = tmp_path / "extract"
    destination.mkdir()
    link = zipfile.ZipInfo("repo/link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(link, "../../outside")

    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(verifier.ArchiveVerificationError):
            verifier.extract_archive_safely(archive, destination)


def test_safe_extraction_rejects_case_collisions(tmp_path: Path) -> None:
    verifier = _load_verifier()
    archive_path = tmp_path / "collision.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("repo/File.txt", "one")
        archive.writestr("repo/file.txt", "two")

    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(verifier.ArchiveVerificationError):
            verifier.validate_archive(archive)


def test_archive_expansion_limit_is_enforced(tmp_path: Path) -> None:
    verifier = _load_verifier()
    archive_path = tmp_path / "large.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("repo/value.bin", b"12345")

    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(verifier.ArchiveVerificationError):
            verifier.validate_archive(archive, max_uncompressed_bytes=4)


def test_v20_packager_uses_canonical_posix_archive_paths() -> None:
    source = PACKAGE.read_text(encoding="utf-8")

    assert ".relative_to(ROOT).as_posix()" in source
    assert 'f"{PREFIX}/{relative_name(path)}"' in source


def test_exact_verifier_proves_imports_come_from_extracted_source() -> None:
    source = VERIFY_RELEASE.read_text(encoding="utf-8")

    assert "'source-import-origin'" in source
    assert "actual.is_relative_to(expected)" in source


def test_exact_verifier_preserves_requested_failed_workspace() -> None:
    source = VERIFY.read_text(encoding="utf-8")

    assert "preserve = bool(arguments.keep)" in source
    assert "if parent is not None and not preserve:" in source


def test_wheel_build_cleans_only_python_staging() -> None:
    source = VERIFY_RELEASE.read_text(encoding="utf-8")

    assert "ROOT/'build/lib'" in source
    assert "(ROOT/'build').glob('bdist.*')" in source
    assert "ROOT/'src/baloot_platform.egg-info'" in source
    assert "ROOT/'build/native-release'" not in source.split(
        "shutil.rmtree(ROOT/'dist'", 1
    )[0].split("for staging in", 1)[1]


def test_resource_marker_is_canonical_lf_on_every_host(tmp_path: Path) -> None:
    resource_sync = _load_resource_sync()
    marker = tmp_path / "marker.txt"

    resource_sync.write_marker(marker)

    assert marker.read_bytes() == resource_sync.MARKER_TEXT.encode("utf-8")
    assert b"\r\n" not in marker.read_bytes()


@pytest.mark.parametrize(
    "summary",
    (
        "3 passed in 1.00s",
        "3 passed, 1 warning in 1.00s",
        "3 passed, 2 skipped, 4 warnings in 1.00s",
    ),
)
def test_test_lane_parser_accepts_success_summaries_with_metadata(
    summary: str,
) -> None:
    lanes = _load_test_lanes()

    assert lanes.SUMMARY_RE.search(summary)
