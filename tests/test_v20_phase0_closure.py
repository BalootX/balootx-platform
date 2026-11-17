from __future__ import annotations

import importlib.util
import json
import subprocess
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "scripts/verifier_preflight_v20.py"
VERIFY_RELEASE = ROOT / "scripts/verify_release_v20.py"
VERIFY_ARCHIVE = ROOT / "scripts/verify_archive_v20.py"
PACKAGE_RELEASE = ROOT / "scripts/package_release_v20.py"
CLOSE_PHASE0 = ROOT / "scripts/close_phase0_v20.py"
DOCKERFILE = ROOT / "Dockerfile"
FONT_ROOT = ROOT / "assets/fonts"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bundled_font_manifest_is_complete_and_hash_valid() -> None:
    preflight = _load(PREFLIGHT, "phase0_preflight")
    result = preflight.font_manifest_check()

    assert result["status"] == "passed", result
    assert result["fonts"] == 6
    assert result["licenses"] == ["LICENSE-DEJAVU.txt", "LICENSE-NOTO.txt"]
    manifest = json.loads((FONT_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["fonts"]) == {
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
        "NotoKufiArabic-Bold.ttf",
        "NotoKufiArabic-Regular.ttf",
        "NotoSansArabic-Bold.ttf",
        "NotoSansArabic-Regular.ttf",
    }


def test_bundled_fonts_are_the_first_resolution_root() -> None:
    preflight = _load(PREFLIGHT, "phase0_font_resolution")

    assert preflight.FONT_ROOTS[0] == FONT_ROOT
    for filename, _ in preflight.FONT_REQUIREMENTS.values():
        assert preflight.resolve_font(filename) == (FONT_ROOT / filename).resolve()


def test_release_build_requires_two_clean_byte_identical_wheels() -> None:
    source = VERIFY_RELEASE.read_text(encoding="utf-8")

    assert "def build_reproducible_wheel_pair()" in source
    assert "for attempt in ('a','b')" in source
    assert "clean_python_staging()" in source
    assert "'two clean wheel builds are not byte-identical: '" in source
    assert "'byteIdentical':True" in source


def test_archive_verifier_requires_exact_test_inventory_parity() -> None:
    source = VERIFY_ARCHIVE.read_text(encoding="utf-8")

    assert '"release/test-inventory-v20.json"' in source
    assert '"source-package-test-inventory"' in source
    assert '"source and packaged pytest inventories do not match"' in source


def test_repository_package_includes_phase0_identity_reports() -> None:
    source = PACKAGE_RELEASE.read_text(encoding="utf-8")

    for name in (
        "test-inventory-v20.json",
        "wheel-reproducibility-v20.json",
        "verifier-preflight-v20.json",
    ):
        assert f'"{name}"' in source


def test_baseline_identity_records_git_or_an_explicit_package_boundary() -> None:
    close_phase0 = _load(CLOSE_PHASE0, "phase0_identity")
    identity = close_phase0.source_identity()

    git = ["git", "-c", f"safe.directory={ROOT}", "-C", str(ROOT)]
    commit_result = subprocess.run(
        [*git, "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    tree_result = subprocess.run(
        [*git, "rev-parse", "HEAD^{tree}"],
        text=True,
        capture_output=True,
        check=False,
    )
    if commit_result.returncode == 0 and tree_result.returncode == 0:
        assert identity["gitCommit"] == commit_result.stdout.strip()
        assert identity["gitTree"] == tree_result.stdout.strip()
        assert "reasonGitUnavailable" not in identity
    else:
        assert identity["gitCommit"] is None
        assert identity["gitTree"] is None
        assert identity["reasonGitUnavailable"]


def test_baseline_identity_reports_missing_git_metadata(monkeypatch) -> None:
    close_phase0 = _load(CLOSE_PHASE0, "phase0_identity_without_git")

    def unavailable(*args, **kwargs):
        raise subprocess.CalledProcessError(128, args[0])

    monkeypatch.setattr(close_phase0.subprocess, "run", unavailable)
    identity = close_phase0.source_identity()

    assert identity["gitCommit"] is None
    assert identity["gitTree"] is None
    assert identity["reasonGitUnavailable"]


@pytest.fixture
def immutable_git_tree(tmp_path: Path) -> tuple[Path, Path, str, str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    hooks = tmp_path / "empty-hooks"
    hooks.mkdir()

    def git(*arguments: str) -> str:
        return subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "-c",
                "core.autocrlf=false",
                "-c",
                f"core.hooksPath={hooks}",
                *arguments,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()

    git("-c", "init.templateDir=", "init")
    (repository / "README.md").write_bytes(b"immutable fixture\n")
    (repository / "src").mkdir()
    (repository / "src/rules.txt").write_bytes(b"fixture rules\n")
    git("add", "README.md", "src/rules.txt")
    git(
        "-c", "user.name=Baseline fixture",
        "-c", "user.email=baseline-fixture@example.invalid",
        "-c", "commit.gpgsign=false",
        "commit", "-m", "Immutable baseline fixture",
    )
    commit = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    archive = tmp_path / "baseline.zip"
    git("archive", "--format=zip", f"--output={archive}", commit)
    source = tmp_path / "source"
    source.mkdir()
    with zipfile.ZipFile(archive) as package:
        package.extractall(source)
    return repository, source, commit, tree


@pytest.mark.parametrize("mutation", ("tampered", "missing", "extra"))
def test_immutable_baseline_tree_is_exact_and_tamper_evident(
    immutable_git_tree: tuple[Path, Path, str, str], mutation: str
) -> None:
    close_phase0 = _load(CLOSE_PHASE0, "phase0_immutable_tree")
    repository, source, commit, tree = immutable_git_tree

    identity = close_phase0.validate_git_tree(source, repository, commit, tree)
    assert identity == {
        "gitCommit": commit,
        "gitTree": tree,
        "gitTreeFiles": 2,
        "gitTreeValidation": "passed",
    }

    if mutation == "tampered":
        (source / "README.md").write_bytes(b"tampered\n")
        expected_difference = {"mismatched": ["README.md"]}
    elif mutation == "missing":
        (source / "src/rules.txt").unlink()
        expected_difference = {"missing": ["src/rules.txt"]}
    else:
        (source / "extra.txt").write_bytes(b"extra\n")
        expected_difference = {"extra": ["extra.txt"]}
    with pytest.raises(RuntimeError, match="does not match declared Git tree") as error:
        close_phase0.validate_git_tree(source, repository, commit, tree)

    differences = json.loads(str(error.value))
    for key in ("missing", "extra", "mismatched"):
        assert differences[key] == expected_difference.get(key, [])


@pytest.mark.parametrize("mismatch", ("abbreviated_commit", "wrong_tree"))
def test_immutable_baseline_rejects_inexact_declared_identity(
    immutable_git_tree: tuple[Path, Path, str, str], mismatch: str
) -> None:
    close_phase0 = _load(CLOSE_PHASE0, "phase0_immutable_identity")
    repository, source, commit, tree = immutable_git_tree
    if mismatch == "abbreviated_commit":
        commit = commit[:12]
    else:
        tree = "0" * len(tree)

    with pytest.raises(RuntimeError, match="commit or tree does not resolve exactly"):
        close_phase0.validate_git_tree(source, repository, commit, tree)


def test_preflight_checks_the_declared_verification_workspace(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("BALOOT_VERIFY_WORKSPACE", str(tmp_path))
    preflight = _load(PREFLIGHT, "phase0_workspace")

    assert preflight.verification_workspace() == tmp_path.resolve()
    result = preflight.disk_check()
    assert result["path"] == str(tmp_path.resolve())


def test_supported_verifier_image_contains_bundled_font_assets() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    verifier = dockerfile.split("FROM python-base AS verifier", 1)[1]

    assert "COPY . ." in verifier
    assert "COPY dist ./dist" not in verifier
