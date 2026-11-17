#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    with TemporaryDirectory(prefix="balootx-openapi-") as temporary_root:
        os.environ["BALOOT_RUNTIME"] = str(Path(temporary_root) / "runtime")
        os.environ["BALOOT_OUTPUTS"] = str(Path(temporary_root) / "outputs")
        os.environ.pop("BALOOT_DATABASE_URL", None)
        from baloot_platform.service.api import CONTAINER, app

        try:
            output = ROOT / "api/openapi.json"
            rendered = (
                json.dumps(app.openapi(), ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            ).encode("utf-8")
            if args.check:
                if not output.exists() or output.read_bytes() != rendered:
                    raise SystemExit(f"stale OpenAPI contract: {output}")
                return 0
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(rendered)
            print(output)
            return 0
        finally:
            CONTAINER.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
