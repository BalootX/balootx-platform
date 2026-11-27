"""Prepare an ordinary SQLite configuration image or restore its protected registry."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from .store import sqlite_oidc_configuration_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protected", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rollback", type=Path, help="Predecessor configuration image to restore into a new output")
    args = parser.parse_args()
    try:
        sqlite_oidc_configuration_image(args.protected, args.output, rollback=args.rollback)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    except (sqlite3.Error, OSError):
        raise SystemExit("Unable to verify or create the SQLite configuration image. Check database paths and custody.") from None


if __name__ == "__main__":
    main()
