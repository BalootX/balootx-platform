"""One 4.5-minute local yield row; never invokes rendering or remote jobs."""
import argparse
import json
from pathlib import Path

from baloot_platform.core.rules import RuleBook
from baloot_platform.research.outcome_generation import load_outcome_policy
from baloot_platform.research.outcome_rollout import generate_outcomes
from baloot_platform.research.outcome_spec import compile_outcome_spec
from benchmark_outcome_construction import case_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True, choices=['sun-named', 'hokum-named', 'big-swing', 'kaboot'])
    parser.add_argument('--policy', required=True, choices=['informed', 'optimal', 'mixed'])
    parser.add_argument('--seed', type=int, default=20261025)
    parser.add_argument('--seconds', type=float, default=270)
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--strategy', choices=['guided', 'rollout'], default='guided')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rules = RuleBook.load('configs/rules/saudi_federation_2023_v7.yaml')
    policy = load_outcome_policy('configs/research/outcome_policy_v3.yaml')
    compiled = compile_outcome_spec(case_spec(args.case, args.policy, rules), rules)
    result = generate_outcomes(compiled, rules, policy, count=500, seed=args.seed,
        output=args.out, workers=args.workers, maximum_seconds=args.seconds, strategy=args.strategy)
    print(json.dumps(dict(case=args.case, policy=args.policy,
        **{key: value for key, value in result.items()
            if key not in {'accepted', 'retained', 'plays', 'outcomeSpec', 'policyProfile', 'workerReceipts'}})))
    if any(reason.startswith('unexpected_game_error:') and count for reason, count in result['rejectReasons'].items()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
