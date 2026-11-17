"""Compare DD root values with the existing native CP-SAT exact authority."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import yaml

from baloot_platform.analysis.double_dummy import DoubleDummySolver
from baloot_platform.core.bots import BotProfile, HeuristicBot
from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, Seat
from baloot_platform.core.models import LegalPlayActionV1
from baloot_platform.core.rules import RuleBook
from baloot_platform.research.ortools_exact_state import ExactNormalizationStopped, OrToolsExactStateNormalizerV1
from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
from baloot_platform.research.ortools_research_executor import _complete_state_quality_artifact_v1
from baloot_platform.research.outcome_generation import OutcomePolicyIncomplete, _prefix
from baloot_platform.research.scenario_contracts import ExactTracePrefixV1, ScenarioProofRequestV1, ScenarioProofBudgetV1, ScenarioQualityExecutionBudgetV1


def cp_sat_oracle(rules, normalizer, prefix, actor, policy):
    deadline = time.monotonic()+5
    def cancelled():
        return time.monotonic() >= deadline
    normalized = normalizer.normalize(prefix,decision_point={'kind':'play'},maximum_seconds=5,cancel_check=cancelled)
    request = ScenarioProofRequestV1.build(dict(schemaVersion=1,mode='exact',visibility='complete_state',
        claim='winning_root_action_count',targetTeam=actor.team.name.lower(),utility='awarded_point_margin',
        successThreshold=0,budgetProfile=policy['id']))
    settings = policy['optimal']
    budget = ScenarioProofBudgetV1.build(policy['id'],settings['proofBudget'])
    artifact = _complete_state_quality_artifact_v1(rules,normalized,prefix,{'kind':'play'},request,
        maximum_nodes=budget.maximum_nodes,maximum_depth=budget.maximum_depth,
        maximum_seconds=max(1e-6,deadline-time.monotonic()),cancel_check=cancelled,budget=budget,
        execution_budget=ScenarioQualityExecutionBudgetV1.build(settings['executionBudget']),
        maximum_conflicts=settings['maximumConflicts'])
    if cancelled() or artifact['proof']['completionStatus'] != 'exact_complete':
        raise OutcomePolicyIncomplete('oracle_incomplete')
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--count', type=int, default=50)
    parser.add_argument('--library', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.offset < 200 or not 1 <= args.count <= 50 or args.offset+args.count > 200:
        parser.error('validation chunks must be 1..50 positions, offset 0..199')
    rules = RuleBook.load('configs/rules/saudi_federation_2023_v7.yaml')
    policy = yaml.safe_load(Path('configs/research/outcome_policy_v1.yaml').read_text(encoding='utf-8'))
    engine, dd = BalootEngine(rules), DoubleDummySolver(rules,args.library)
    start, cpu = time.monotonic(), time.process_time()
    normalizer = OrToolsExactStateNormalizerV1(rules,native_cp_sat=True)
    rows, ordinal = [], 0
    for mode in (Contract.SUN, Contract.HOKUM):
        collected = 0
        for seed in range(20261025, 20263025):
            state = engine.create_round(seed=seed,dealer=Seat.EAST)
            engine.shuffle_and_initial_deal(state)
            auction = BiddingMachine(state.dealer,state.purchase_card.suit,
                AuctionDomainIRV1.from_rules(rules.data),state.purchase_card.rank)
            while not auction.finished:
                action = mode.value if not auction.records else 'pass'
                auction.submit(action,trump=state.purchase_card.suit if action == 'hokum' else None)
            assert engine.complete_deal_from_bid_result(state,auction.result)
            bids = [dict(actor=r.seat.name.lower(),action=r.action,
                **({'trumpSuit':r.trump.name.lower()} if r.trump is not None else {})) for r in state.bid_history]
            engine.prepare_projects(state)
            bot = HeuristicBot(engine,BotProfile(seed))
            plays = []
            while state.current_player is not None:
                if len(plays) >= 24 and len(engine.legal_play_actions(state)) > 1 and collected < 100:
                    current = ordinal
                    ordinal += 1
                    collected += 1
                    if args.offset <= current < args.offset+args.count:
                        declarations = [{key:event.data[key] for key in ('owner','kind','value','highRank','declarationOrder')}
                            for event in state.events if event.type == 'project_declared']
                        prefix = _prefix(state.shuffle.deck_order,'east',bids,plays=plays).to_dict()
                        prefix.pop('sha256')
                        prefix['projectDeclarations'] = declarations
                        prefix = ExactTracePrefixV1.build(prefix)
                        row = dict(seed=seed,mode=mode.value,index=len(plays),ordinal=current)
                        actual = dd.solve(state,maximum_seconds=1,all_root_values=True)
                        if actual['status'] != 'exact':
                            row.update(status='incomplete_dd',dd=actual)
                        else:
                            try:
                                artifact = cp_sat_oracle(rules,normalizer,prefix,state.current_player,policy)
                            except (ExactNormalizationStopped,InformationSetCompilationStopped,OutcomePolicyIncomplete) as error:
                                row.update(status='incomplete_oracle',reason=str(error))
                            else:
                                expected = {r['action']:r['value'] for r in artifact['proof']['rootActions']}
                                observed = {r['action']:r['value'] for r in actual['rootActions']}
                                best = dd.solve(state,maximum_seconds=1)
                                expected_best = min((action for action,value in expected.items() if value == max(expected.values())),
                                    key=lambda code: LegalPlayActionV1.from_code(code).action_id)
                                row.update(status='match' if observed == expected and best['status'] == 'exact'
                                    and best['selectedAction'] == expected_best else 'DISAGREEMENT',
                                    expected=expected,observed=observed,dd=actual,bestOnly=best)
                        rows.append(row)
                        print(json.dumps({key:value for key,value in row.items() if key not in {'dd','bestOnly','expected','observed'}}),flush=True)
                action = bot.choose(state)
                played = engine.play_card(state,action)
                plays.append(dict(actor=played.seat.name.lower(),card=played.card.code,ekka=played.ekka))
            if collected >= 100:
                break
            if time.monotonic()-start >= 235:
                break
        if len(rows) >= args.count or time.monotonic()-start >= 235:
            break
    result = dict(rows=rows,matches=sum(r['status']=='match' for r in rows),
        disagreements=sum(r['status']=='DISAGREEMENT' for r in rows),wallSeconds=time.monotonic()-start,
        cpuSeconds=time.process_time()-cpu,rulesHash=rules.sha256,libraryHash=dd.library_hash)
    result['status'] = 'complete' if result['matches'] == args.count else 'incomplete'
    args.out.parent.mkdir(exist_ok=True,parents=True)
    args.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({key:value for key,value in result.items() if key != 'rows'}))
    if result['matches'] != args.count:
        raise SystemExit(1 if result['disagreements'] else 2)


if __name__ == '__main__':
    main()
