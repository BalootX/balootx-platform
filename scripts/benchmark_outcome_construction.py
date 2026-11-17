"""One bounded Stage-3b scenario measurement, without rendering or relabelling."""
import argparse
import json
from pathlib import Path

from baloot_platform.core.rules import RuleBook
from baloot_platform.research.outcome_generation import construct_outcome, load_outcome_policy
from baloot_platform.research.outcome_spec import compile_outcome_spec


def case_spec(case, policy, rules):
    raw = dict(schemaVersion=1, rulesProfile=rules.data['profile']['id'], mode='sun')
    if case in {'big-swing', 'kaboot'}:
        raw['category'] = 'big_swing' if case == 'big-swing' else 'kaboot'
    else:
        if case not in {'sun-named', 'hokum-named'}:
            raise ValueError('unknown outcome benchmark case')
        hokum = case == 'hokum-named'
        if hokum:
            raw.update(mode='hokum', trumpSuit='hearts')
        raw['outcome'] = dict(winner='north_south', decidingTrick=dict(index=8,
            winningTeam='north_south', winningCard='H10' if hokum else 'C7', beatenCard='CA' if hokum else 'HA'))
    if policy not in {'informed', 'optimal', 'mixed'}:
        raise ValueError('unknown outcome benchmark policy')
    raw['policies'] = (dict(default='informed', teams={'west_east': 'optimal'})
                       if policy == 'mixed' else dict(default=policy))
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True, choices=['sun-named', 'hokum-named', 'big-swing', 'kaboot'])
    parser.add_argument('--policy', required=True, choices=['informed', 'optimal', 'mixed'])
    parser.add_argument('--seed', type=int, default=20261025)
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--refinement', choices=['first_deviation', 'forward_rollout'], default='first_deviation')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.seconds <= 120:
        parser.error('each local measurement must fit 120 seconds')
    rules = RuleBook.load('configs/rules/saudi_federation_2023_v7.yaml')
    raw = case_spec(args.case, args.policy, rules)
    compiled = compile_outcome_spec(raw, rules)
    policy = load_outcome_policy('configs/research/outcome_policy_v2.yaml')
    result = construct_outcome(compiled, rules, policy, seed=args.seed,
        maximum_seconds=args.seconds, maximum_attempts=3, refinement=args.refinement)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(case=args.case, policy=args.policy, refinement=args.refinement, status=result['status'],
        attempts=len(result['attempts']), proposals=sum(row['proposals'] for row in result['attempts']),
        cuts=sum(len(row['cuts']) for row in result['attempts']), rejectReasons=result['rejectReasons'],
        auctionProposals=sum(row['auctionProposals'] for row in result['attempts']),
        auctionCuts=sum(len(row['auctionCuts']) for row in result['attempts']),
        forwardProbes=sum(row['forwardProbes'] for row in result['attempts']),
        wallSeconds=result['wallSeconds'], cpuSeconds=result['cpuSeconds'],
        acceptedPerSecond=result['acceptedPerSecond'], cpuSecondsPerAccepted=result['cpuSecondsPerAccepted'])))


if __name__ == '__main__':
    main()
