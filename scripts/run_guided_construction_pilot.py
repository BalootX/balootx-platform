from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from baloot_platform.research.guided_construction_pilot import (
    GuidedConstructionPilotProfileV1,
    run_guided_construction_pilot_v1,
)
from baloot_platform.service.container import build_container


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--attempts-per-lane-profile", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--disable-construction-pre-screen",
        action="store_true",
    )
    args = parser.parse_args()
    runtime = Path(tempfile.mkdtemp(prefix="baloot-guided-pilot-"))
    container = build_container(root=args.root.resolve(), writable_root=runtime)
    try:
        result = run_guided_construction_pilot_v1(
            container.research,
            profile=GuidedConstructionPilotProfileV1(
                attempts_per_lane_profile=args.attempts_per_lane_profile,
                seeds_per_shard=min(5, args.attempts_per_lane_profile),
                construction_pre_screen_enabled=(
                    not args.disable_construction_pre_screen
                ),
            ),
        )
        rendered = json.dumps(
            result, ensure_ascii=False, sort_keys=True, indent=2
        ) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8", newline="\n")
        print(rendered, end="")
        return 0
    finally:
        container.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
