"""Bounded card-policy round measurement; fixed legal auction, no outcome search."""
import argparse
import csv
import json
import time
import math
from pathlib import Path

from baloot_platform.analysis.models import InformationState
from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.bots import BotProfile, HeuristicBot
from baloot_platform.core.enums import Seat
from baloot_platform.core.rules import RuleBook
from baloot_platform.research.outcome_generation import OutcomePolicyDriver, OutcomePolicyIncomplete, load_outcome_policy
from baloot_platform.research.outcome_spec import compile_outcome_spec
from baloot_platform.research.scenario_contracts import canonical_sha256


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['sun', 'hokum'], required=True)
    parser.add_argument('--policy', choices=['informed', 'optimal', 'mixed'], required=True)
    parser.add_argument('--seed', type=int, default=20261025)
    parser.add_argument('--start-index', type=int, default=0, choices=range(32))
    parser.add_argument('--decisions', type=int, default=32, choices=range(1, 33))
    parser.add_argument('--profile', type=Path, default=Path('configs/research/outcome_policy_v2.yaml'))
    parser.add_argument('--threads', type=int, choices=range(1, 5), default=None)
    parser.add_argument('--rounds', type=int, choices=range(1, 21), default=1)
    parser.add_argument('--out', type=Path, required=True)
    return parser.parse_args()


def measure_round(args):
    rules = RuleBook.load('configs/rules/saudi_federation_2023_v7.yaml')
    policy = load_outcome_policy(args.profile)
    compiled = compile_outcome_spec(dict(schemaVersion=1, rulesProfile=rules.data['profile']['id'],
        mode=args.mode, **(dict(trumpSuit='hearts') if args.mode == 'hokum' else {})), rules)
    driver = OutcomePolicyDriver(rules, compiled, policy, seed=args.seed, deadline=time.monotonic()+120,
        maximum_threads=args.threads)
    engine = driver.engine
    state = engine.create_round(seed=args.seed, dealer=Seat.EAST)
    engine.shuffle_and_initial_deal(state)
    auction = BiddingMachine(state.dealer, state.purchase_card.suit,
        AuctionDomainIRV1.from_rules(rules.data), state.purchase_card.rank)
    while not auction.finished:
        action = args.mode if not auction.records else 'pass'
        auction.submit(action, trump=state.purchase_card.suit if action == 'hokum' else None)
    assert engine.complete_deal_from_bid_result(state, auction.result)
    engine.prepare_projects(state)
    state._project_declaration_state = engine.project_domain.declaration_state(state.projects, dealer=state.dealer)
    def declare_for_turn():
        declaration = state._project_declaration_state
        if declaration is not None and declaration.current_actor == state.current_player:
            actions = engine.project_domain.legal_declaration_actions(declaration, trump_suit=state.trump_suit)
            assert len(actions) == 1
            engine.apply_project_declaration_action(state, actions[0])
    bot = HeuristicBot(engine, BotProfile(args.seed))
    while len(state.played_cards) < args.start_index:
        declare_for_turn()
        engine.play_card(state, bot.choose(state))
    started, cpu = time.monotonic(), time.process_time()
    rows, failure = [], None
    while state.current_player is not None and len(rows) < args.decisions:
        actor, legal = state.current_player, engine.legal_play_actions(state)
        kind = ('informed' if int(actor.team) == 0 else 'optimal') if args.policy == 'mixed' else args.policy
        tick, used = time.monotonic(), time.process_time()
        declare_for_turn()
        info = InformationState.from_complete(state, actor)
        scope_hash = canonical_sha256(info.to_dict() if kind == 'informed' else state.to_dict(reveal_hands=True))
        try:
            if len(legal) == 1:
                selected, evidence = legal[0], {'method': 'forced_by_rules'}
            else:
                selected, evidence, _, _ = driver.dd_card(state, info, kind)
        except (OutcomePolicyIncomplete, InterruptedError, RuntimeError) as error:
            failure = dict(reason=getattr(error, 'reason', str(error)),
                evidence=getattr(error, 'evidence', None))
            rows.append(dict(index=len(state.played_cards), policy=kind, status='incomplete',
                inputScopeHash=scope_hash,
                wallSeconds=time.monotonic()-tick, cpuSeconds=time.process_time()-used))
            break
        assert selected in legal
        rows.append(dict(index=len(state.played_cards), policy=kind, status='complete', action=selected.code,
            inputScopeHash=scope_hash,
            method=evidence['method'], wallSeconds=time.monotonic()-tick, cpuSeconds=time.process_time()-used,
            **(dict(requestedWorlds=evidence['requestedParticles'],
                solvedWorlds=evidence['sampledWorlds'], exhaustiveCohort=evidence['exhaustiveCohort'])
                if evidence.get('method') in {'sampled_pimc_dd_v1', 'sampled_pimc_dd_batch_v1'} else {}),
            **(dict(valueNorthSouth=next(row['value'] for row in evidence['rootActions']
                if row['action'] == selected.code)*(1 if int(actor.team) == 0 else -1))
                if evidence.get('method') == 'native_dd_mtdf_v1' else {})))
        engine.play_card(state, selected.card, ekka=selected.ekka)
    if args.policy == 'optimal' and state.round_score is not None:
        actual = state.round_score.awarded_points[0]-state.round_score.awarded_points[1]
        assert all(row.get('valueNorthSouth', actual) == actual for row in rows), 'DD/engine optimal-line value drift'
    driver.close()
    result = dict(mode=args.mode, policy=args.policy, seed=args.seed, startIndex=args.start_index,
        status='complete' if state.round_score else 'bounded_probe_complete' if failure is None else 'incomplete',
        profileHash=canonical_sha256(policy), configuredThreads=policy.get('cardPolicy', {}).get('maximumThreads', 1),
        effectiveThreads=min(args.threads or 4, policy.get('cardPolicy', {}).get('maximumThreads', 1)),
        requestedDecisions=args.decisions, fullRoundCompleted=state.round_score is not None,
        decisions=rows, failure=failure,
        wallSeconds=time.monotonic()-started, cpuSeconds=time.process_time()-cpu,
        score=state.round_score.to_dict() if state.round_score else None)
    return result


def main():
    args = arguments()
    rounds = []
    for _ in range(args.rounds):
        rounds.append(measure_round(args))
        args.seed += 1
        print(json.dumps({k: v for k, v in rounds[-1].items()
            if k not in {'decisions', 'score', 'failure'}}), flush=True)
    complete = [r for r in rounds if r['fullRoundCompleted']]
    def percentiles(key):
        values = sorted(r[key] for r in complete)
        return {f'p{p}': values[max(0, math.ceil(len(values)*p/100)-1)] if values else None for p in (50, 95)}
    result = rounds[0] if args.rounds == 1 else dict(rounds=rounds, requestedRounds=args.rounds,
        completeRounds=len(complete), incompleteRounds=len(rounds)-len(complete),
        completeRoundWall=percentiles('wallSeconds'), completeRoundCpu=percentiles('cpuSeconds'),
        allRoundP95Censored=len(complete) != len(rounds), threads=args.threads,
        wallSeconds=sum(r['wallSeconds'] for r in rounds), cpuSeconds=sum(r['cpuSeconds'] for r in rounds))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    fields = ['mode', 'policy', 'seed', 'status', 'fullRoundCompleted', 'effectiveThreads', 'wallSeconds', 'cpuSeconds']
    with args.out.with_suffix('.csv').open('w', encoding='utf-8', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rounds)
    print(json.dumps({key: value for key, value in result.items() if key not in {'decisions', 'score', 'failure', 'rounds'}},
        ensure_ascii=False))


if __name__ == '__main__':
    main()
