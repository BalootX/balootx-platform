"""Check a production node_modules tree against an npm package lock."""
import argparse
import json
from pathlib import Path
import platform
import sys


def npm_platform():
    machine = platform.machine().lower()
    cpu = {
        "amd64": "x64", "x86_64": "x64", "x86": "ia32", "i386": "ia32", "i686": "ia32",
        "aarch64": "arm64", "armv7l": "arm", "loongarch64": "loong64",
    }.get(machine, machine)
    libc = platform.libc_ver()[0] if sys.platform == "linux" else ""
    if sys.platform == "linux" and not libc and any(Path("/lib").glob("ld-musl-*.so.1")):
        libc = "musl"
    return {"os": platform.system().lower().replace("windows", "win32"), "cpu": cpu, "libc": libc}


def matches_platform(rules, value):
    if isinstance(rules, str):
        rules = (rules,)
    return f"!{value}" not in rules and (
        value in rules or "any" in rules or all(rule.startswith("!") for rule in rules)
    )


def installed_manifests(node_modules):
    for entry in node_modules.iterdir():
        if entry.name.startswith(".") or not entry.is_dir():
            continue
        packages = entry.iterdir() if entry.name.startswith("@") and not entry.is_symlink() else (entry,)
        for package in packages:
            manifest = package / "package.json"
            if manifest.is_file():
                yield manifest
            nested = package / "node_modules"
            if not package.is_symlink() and not nested.is_symlink() and nested.is_dir():
                yield from installed_manifests(nested)


def check_parity(node_modules, lockfile):
    lock = json.loads(lockfile.read_bytes())
    if (not isinstance(lock, dict) or lock.get("lockfileVersion") not in (2, 3)
            or not isinstance(lock.get("packages"), dict)):
        raise ValueError("expected an npm lockfile with a version 2 or 3 packages map")
    installed = {
        "node_modules/" + manifest.parent.relative_to(node_modules).as_posix(): manifest
        for manifest in installed_manifests(node_modules)
    }
    expected = {}
    target = npm_platform()
    for path, row in lock["packages"].items():
        if not path:
            continue
        if (not path.startswith("node_modules/") or not isinstance(row, dict)
                or not isinstance(row.get("version"), str)):
            raise ValueError(f"unsupported lockfile package: {path}")
        if row.get("dev") or any(not matches_platform(row.get(key, ()), value) for key, value in target.items()):
            continue
        expected[path] = row

    problems = []
    for path, row in sorted(expected.items()):
        version = row["version"]
        if path not in installed:
            if not (row.get("optional") or row.get("devOptional")):
                problems.append(f"missing {path}: expected {version}")
            continue
        manifest = json.loads(installed[path].read_bytes())
        actual = manifest.get("version") if isinstance(manifest, dict) else None
        if actual != version:
            problems.append(f"version mismatch {path}: expected {version}, found {actual!r}")
    problems.extend(f"extra {path}" for path in sorted(installed.keys() - expected.keys()))
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("node_modules", type=Path)
    parser.add_argument("lockfile", type=Path)
    args = parser.parse_args()
    try:
        problems = check_parity(args.node_modules, args.lockfile)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Remotion image parity check failed: {exc}\n")
    if problems:
        parser.exit(1, "\n".join(problems) + "\n")
    print("Remotion image dependencies match the lockfile.")


if __name__ == "__main__":
    main()
