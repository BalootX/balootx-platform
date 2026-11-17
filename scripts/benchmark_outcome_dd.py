"""Capped, reproducible card-decision timings; unsupported is never a fast solve."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from baloot_platform.core.bots import BotProfile, HeuristicBot
from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, Seat
from baloot_platform.core.rules import RuleBook
from baloot_platform.native.accelerator import NativeAccelerator, NativeUnavailable


def summarize_rows(rows):
    groups = []
    for mode in ('sun', 'hokum'):
        for trick in (1, 2, 4, 6):
            group = [r for r in rows if r['mode'] == mode and r['trick'] == trick]
            if not group:
                continue
            exact = sorted(r['wallSeconds'] for r in group if r['status'] == 'exact')
            censored = [r['wallSeconds'] for r in group if r['status'] != 'exact']
            def quantile(percent):
                rank = (percent*len(group)+99)//100-1
                # Do not hide failed solves by reporting a successes-only p95.
                return exact[rank] if rank < len(exact) and (
                    not censored or exact[rank] <= min(censored)) else None
            groups.append(dict(mode=mode, trick=trick, counts=dict(Counter(r['status'] for r in group)),
                p50Seconds=quantile(50), p95Seconds=quantile(95),
                incompleteReasons=dict(Counter(r.get('stopReason', r.get('reason', 'unknown'))
                    for r in group if r['status'] != 'exact'))))
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['legacy', 'dd'], default='legacy')
    parser.add_argument('--library', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--deals', type=int, default=20)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--seconds', type=float, default=1)
    args = parser.parse_args()
    if not 1 <= args.deals <= 20 or not 0 <= args.offset <= 10000 or not 0 < args.seconds <= 1:
        parser.error('at most 20 deals and one second/decision per local measurement')
    rules = RuleBook.load('configs/rules/saudi_federation_2023_v7.yaml')
    engine = BalootEngine(rules)
    start, cpu = time.monotonic(), time.process_time()
    rows, initialization_error = [], None
    try:
        if args.backend == 'legacy':
            backend = NativeAccelerator(rules, args.library)
        else:
            from baloot_platform.analysis.double_dummy import DoubleDummySolver
            backend = DoubleDummySolver(rules, args.library)
    except NativeUnavailable as error:
        initialization_error = str(error)
    for mode in (Contract.SUN, Contract.HOKUM):
        for seed in range(20261025+args.offset, 20261025+args.offset+args.deals):
            state = engine.create_round(seed=seed, dealer=Seat.EAST)
            engine.shuffle_and_initial_deal(state)
            auction = BiddingMachine(state.dealer,state.purchase_card.suit,
                AuctionDomainIRV1.from_rules(rules.data),state.purchase_card.rank)
            while not auction.finished:
                action = mode.value if not auction.records else 'pass'
                auction.submit(action,trump=state.purchase_card.suit if action == 'hokum' else None)
            assert engine.complete_deal_from_bid_result(state,auction.result)
            engine.prepare_projects(state)
            bot = HeuristicBot(engine, BotProfile(seed))
            while len(state.played_cards) <= 20:
                index = len(state.played_cards)
                if index in (0, 4, 12, 20):
                    tick, used = time.monotonic(), time.process_time()
                    row = dict(mode=mode.value, seed=seed, trick=index//4+1)
                    if initialization_error:
                        row.update(status='unsupported', reason=initialization_error)
                    else:
                        try:
                            if args.backend == 'legacy':
                                result = backend.exact_team_value(state, engine, state.current_player.team,
                                    max_nodes=10000000, max_seconds=args.seconds, max_cache_entries=1000000)
                                row.update(status='exact' if result.exact else 'incomplete', nodes=result.nodes)
                            else:
                                result = backend.solve(state, maximum_seconds=args.seconds)
                                row.update(status=result['status'], nodes=result['nodes'], stopReason=result['stopReason'])
                                row['nativeCache'] = result.get('nativeCache', {})
                        except (NativeUnavailable, ValueError) as error:
                            row.update(status='unsupported', reason=str(error))
                    row.update(wallSeconds=time.monotonic()-tick, cpuSeconds=time.process_time()-used)
                    rows.append(row)
                engine.play_card(state, bot.choose(state))
    groups = summarize_rows(rows)
    if args.backend == 'dd' and initialization_error is None:
        backend.close()
    result = dict(backend=args.backend, rulesHash=rules.sha256,
        libraryHash=hashlib.sha256(Path(args.library).read_bytes()).hexdigest(), rows=rows, groups=groups,
        wallSeconds=time.monotonic()-start, cpuSeconds=time.process_time()-cpu)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key != 'rows'}))


if __name__ == '__main__':
    main()
