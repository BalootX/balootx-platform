import json
import subprocess
import sys

import pytest

from scripts import remotion_image_parity


@pytest.fixture
def tree(tmp_path):
    node_modules = tmp_path / "node_modules"
    lockfile = tmp_path / "package-lock.json"
    versions = {"react": "19.3.1", "@scope/dependency": "4.0.544", "@scope/dependency/node_modules/react": "18.3.1"}
    packages = {
        "": {"name": "renderer"},
        **{f"node_modules/{path}": {"version": version} for path, version in versions.items()},
    }
    for path, version in versions.items():
        manifest = node_modules / path / "package.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({"version": version}), encoding="utf-8")
    lockfile.write_text(json.dumps({"lockfileVersion": 3, "packages": packages}), encoding="utf-8")
    return node_modules, lockfile


def test_matches_scoped_and_nested_packages(tree):
    assert remotion_image_parity.check_parity(*tree) == []


def test_cli_reports_installed_version_mismatch_missing_and_extra(root, tree):
    node_modules, lockfile = tree
    (node_modules / "react/package.json").write_text('{"version": "19.2.0"}', encoding="utf-8")
    (node_modules / "@scope/dependency/node_modules/react/package.json").unlink()
    extra = node_modules / "stale/package.json"
    extra.parent.mkdir()
    extra.write_text('{"version": "1.0.0"}', encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-P", str(root / "scripts/remotion_image_parity.py"), str(node_modules), str(lockfile)],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 1
    assert result.stderr.splitlines() == [
        "missing node_modules/@scope/dependency/node_modules/react: expected 18.3.1",
        "version mismatch node_modules/react: expected 19.3.1, found '19.2.0'",
        "extra node_modules/stale",
    ]


def test_requires_native_optional_packages_for_the_current_platform(tree, monkeypatch):
    monkeypatch.setattr(remotion_image_parity, "npm_platform", lambda: {"os": "linux", "cpu": "x64", "libc": "glibc"})
    node_modules, lockfile = tree
    lock = json.loads(lockfile.read_bytes())
    lock["packages"].update({
        "node_modules/typescript": {"version": "6.0.2", "dev": True},
        "node_modules/foreign-os": {"version": "1", "optional": True, "os": ["darwin"]},
        "node_modules/foreign-cpu": {"version": "1", "optional": True, "cpu": ["!x64"]},
        "node_modules/foreign-libc": {"version": "1", "optional": True, "libc": ["musl"]},
        "node_modules/native": {"version": "1", "optional": True, "os": ["linux"], "cpu": ["x64"], "libc": "!musl"},
        "node_modules/wasm-fallback": {"version": "1", "optional": True},
    })
    lockfile.write_text(json.dumps(lock), encoding="utf-8")
    assert remotion_image_parity.check_parity(*tree) == ["missing node_modules/native: expected 1"]

    native = node_modules / "native/package.json"
    native.parent.mkdir()
    native.write_text('{"version": "1"}', encoding="utf-8")
    assert remotion_image_parity.check_parity(*tree) == []

    native.write_text('{"version": "2"}', encoding="utf-8")
    for package in ("foreign-cpu", "foreign-libc"):
        foreign = node_modules / package / "package.json"
        foreign.parent.mkdir()
        foreign.write_text('{"version": "1"}', encoding="utf-8")
    assert remotion_image_parity.check_parity(*tree) == [
        "version mismatch node_modules/native: expected 1, found '2'",
        "extra node_modules/foreign-cpu",
        "extra node_modules/foreign-libc",
    ]


def test_rejects_unsupported_lockfile_version(tree):
    _, lockfile = tree
    lock = json.loads(lockfile.read_bytes())
    lock["lockfileVersion"] = 1
    lockfile.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(ValueError, match="version 2 or 3"):
        remotion_image_parity.check_parity(*tree)


def test_ignores_hidden_node_modules_folders(tree):
    node_modules, _ = tree
    hidden = node_modules / ".remotion/package.json"
    hidden.parent.mkdir()
    hidden.write_text('{"version": "1"}', encoding="utf-8")
    assert remotion_image_parity.check_parity(*tree) == []


@pytest.mark.parametrize(("machine", "npm_arch"), [("AMD64", "x64"), ("x86_64", "x64"), ("aarch64", "arm64")])
def test_maps_machine_names_to_npm_architectures(monkeypatch, machine, npm_arch):
    monkeypatch.setattr(remotion_image_parity.platform, "machine", lambda: machine)
    assert remotion_image_parity.npm_platform()["cpu"] == npm_arch
