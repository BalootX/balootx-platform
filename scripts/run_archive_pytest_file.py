#!/usr/bin/env python3
"""Run one archive test file with a pre-timeout Python stack receipt."""
from __future__ import annotations

import argparse
import faulthandler
import sys

import pytest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-after-seconds", required=True, type=float)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.diagnostic_after_seconds <= 0:
        parser.error("diagnostic delay must be positive")
    pytest_args = list(args.pytest_args)
    if pytest_args[:1] == ["--"]:
        pytest_args.pop(0)
    if not pytest_args:
        parser.error("at least one pytest argument is required")
    faulthandler.enable(file=sys.stderr, all_threads=True)
    faulthandler.dump_traceback_later(
        args.diagnostic_after_seconds,
        repeat=False,
        file=sys.stderr,
        exit=False,
    )
    try:
        return int(pytest.main(pytest_args))
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    raise SystemExit(main())
