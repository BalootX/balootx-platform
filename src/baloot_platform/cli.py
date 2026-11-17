from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


from .config import build_bundle, platform_root
from .core.enums import Contract, Suit
from .core.match import simulate_match
from .core.scenario import ScenarioRequest, generate_episode
from .video.remotion_renderer import RemotionProfessionalRenderer


def main() -> None:
    if sys.argv[1:2] == ['generate']:
        from .research.outcome_cli import main as generate
        generate(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(prog="baloot", epilog="Verified research batches: baloot generate --help")
    parser.add_argument("command", choices=["round", "match", "video"])
    parser.add_argument("--root")
    parser.add_argument("--seed", type=int, default=917)
    parser.add_argument(
        "--contract", choices=["sun", "hokum", "ashkal"], default="hokum"
    )
    parser.add_argument(
        "--trump", choices=["spades", "hearts", "diamonds", "clubs"], default="hearts"
    )
    parser.add_argument("--output", default="outputs/cli-baloot")
    args = parser.parse_args()
    default_root = platform_root()
    root = Path(args.root or default_root)
    bundle = build_bundle(root)
    contract = Contract.parse(args.contract)
    trump = Suit.parse(args.trump) if contract == Contract.HOKUM else None
    if args.command == "match":
        result = simulate_match(
            __import__(
                "baloot_platform.core.engine", fromlist=["BalootEngine"]
            ).BalootEngine(bundle.rules),
            seed=args.seed,
            forced_contract=contract,
            forced_trump=trump,
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return
    convention = bundle.convention
    episode = generate_episode(
        bundle.rules,
        convention,
        bundle_hash=bundle.sha256,
        request=ScenarioRequest.governed(
            seed=args.seed,
            contract=contract,
            trump_suit=trump,
            require_project=False,
        ),
        copy=bundle.video["localizations"],
    )
    if args.command == "round":
        print(json.dumps(episode.to_dict(), ensure_ascii=False, indent=2))
        return
    output = Path(args.output)
    renderer = RemotionProfessionalRenderer(bundle.video, repository_root=root)
    artifacts = renderer.execute(
        episode,
        export_preset="hooked_flagship",
        output_dir=output.parent,
        basename=output.name,
    )
    print(
        json.dumps(
            {k: str(getattr(artifacts, k)) for k in artifacts.__dataclass_fields__},
            ensure_ascii=False,
            indent=2,
        )
    )
