"""One command for verified outcome batches, without media dependencies."""
import argparse
import json
from pathlib import Path

from baloot_platform.config import platform_root
from baloot_platform.core.rules import RuleBook
from .outcome_generation import load_outcome_policy
from .outcome_rollout import generate_outcomes
from .outcome_spec import load_outcome_spec


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--seed', type=int, default=20261025)
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--seconds', type=float, default=270)
    parser.add_argument('--max-story-variants', type=int, default=3)
    parser.add_argument('--story-score-bucket', type=int, default=10)
    parser.add_argument('--story-swing-bucket', type=int, default=10)
    parser.add_argument('--story-opening-tricks', type=int, choices=range(4), default=3)
    parser.add_argument('--strategy', choices=['guided', 'rollout', 'first_deviation', 'forward_rollout'], default='guided')
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--rules', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    root = platform_root()  # Enforcement follows argument parsing; --help needs no host launcher.
    rules = RuleBook.load(args.rules or root/'configs/rules/saudi_federation_2023_v7.yaml')
    result = generate_outcomes(load_outcome_spec(args.spec, rules), rules,
        load_outcome_policy(args.profile or root/'configs/research/outcome_policy_v3.yaml'),
        count=args.count, seed=args.seed, workers=args.workers, maximum_seconds=args.seconds,
        strategy=args.strategy, output=args.out, maximum_story_variants=args.max_story_variants,
        story_settings=dict(scoreBucket=args.story_score_bucket, swingBucket=args.story_swing_bucket,
            openingTricks=args.story_opening_tricks))
    print(json.dumps({key: value for key, value in result.items()
        if key not in {'accepted', 'retained', 'plays', 'outcomeSpec', 'policyProfile'}}, ensure_ascii=False))
    if any(reason.startswith('unexpected_game_error:') and count for reason, count in result['rejectReasons'].items()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
