"""Seal or verify the package archives in the Foundry Playwright cache."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path


def archive_inventory(directory: Path) -> bytes:
    checksums = {}
    for path in directory.iterdir():
        if not path.name.endswith(".deb"):
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Expected a regular Playwright package archive: {path.name}")
        with path.open("rb") as archive:
            checksums[path.name] = file_digest(archive, "sha256").hexdigest()
    return (json.dumps(checksums, sort_keys=True) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("seal", "verify"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    manifest = args.directory / "inventory.json"
    if args.action == "seal":
        manifest.write_bytes(archive_inventory(args.directory))
        return
    if manifest.is_symlink() or not manifest.is_file():
        parser.exit(1, "Playwright package cache inventory is missing or invalid.\n")
    if manifest.read_bytes() != archive_inventory(args.directory):
        parser.exit(1, "Playwright package cache inventory does not match the restored archives.\n")


if __name__ == "__main__":
    main()
