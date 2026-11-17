"""Bounded, diverse CP-SAT proposals followed by declared-policy engine replay."""
from __future__ import annotations

import json
import multiprocessing
import os
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from itertools import permutations

from ortools.sat.python import cp_model

from baloot_platform.core.enums import Team, Suit
from baloot_platform.core.cards import Card
from baloot_platform.core.rules import RuleBook
from baloot_platform.native.accelerator import NativeAccelerator, NativeUnavailable
from .campaign_coordinator import decision_state_card_vector_v1
from .exact_state import ExactStateNormalizerV1
from .ortools_research_executor import GenericCompleteTraceCompilerV1
from .outcome_generation import (
    OutcomePolicyDriver, OutcomePolicyIncomplete, _construction_scenario, _prefix,
    construct_outcome, validate_outcome_policy,
)
from .outcome_spec import compile_outcome_spec
from .scenario_contracts import canonical_sha256
from .scenario_evidence import ScenarioRuntimeVerifierV1
from .scenario_predicate_lowering_v2 import (
    CpSatActionVariablesV1, CpSatPredicateModelContextV1, CpSatTrickVariablesV1,
    ScenarioPredicateLoweringRegistryV2,
)
from .ortools_scenario_model import reify_scenario_expression
from .outcome_search import OutcomeSearch, SEARCH_PROFILE, deck_distance
from .outcome_catalogue import catalogue_rows, game_facts, story_facts, story_granularity, write_exports, write_neighbourhoods


def _merge_distinct(candidates, count, story_cap):
    """Admission depends on the finished seed streams, never their arrival order."""
    accepted, seen, decks, stories, rejects = [], set(), [], Counter(), Counter()
    for row in sorted(candidates, key=lambda row: (row['diversityKey'], row['id'])):
        if len(accepted) == count:
            break
        key, deck, story = row['diversityKey'], row['deckOrder'], row['storyHash']
        reason = ('duplicate_batch_symmetry' if key in seen else
            'duplicate_batch_deck_distance' if any(deck_distance(deck, d) < 4 for d in decks) else
            'story_variant_cap' if stories[story] >= story_cap else None)
        if reason:
            rejects[reason] += 1
            continue
        accepted.append(row)
        seen.add(key)
        decks.append(deck)
        stories[story] += 1
    return accepted, rejects


class DealProposalStream:
    """Reuse one existing compiler model/solver; distance cuts enumerate it."""
    def __init__(self, compiled, rules, policy, seed, deadline, partition=(1, 0, 1), distance=4):
        self.compiled, self.rules, self.policy = compiled, rules, policy
        self.seed, self.deadline, self.partition, self.distance = seed, deadline, partition, distance
        self.model, self.solver, self.last, self.ordinal = None, None, None, 0

    def _capture(self, model, attempt, solver, remaining, maximum_conflicts):
        self.model, self.solver = model, solver or cp_model.CpSolver()
        self.last = attempt
        return attempt

    def next(self, maximum_seconds):
        remaining = min(maximum_seconds, self.deadline-time.monotonic())
        if remaining <= 0:
            raise OutcomePolicyIncomplete('campaign_time_budget')
        if self.model is None:
            result = GenericCompleteTraceCompilerV1(rules=self.rules, native_cp_sat_scoring=True,
                structural_only=True, canonicalize_deal_packets=True,
                minimum_deck_hamming_distance=self.distance, candidate_partition=self.partition,
                auction_model_refiner=self._capture)(
                    _construction_scenario(self.compiled, self.policy), (), seed=self.seed,
                    maximum_seconds=remaining)
            if result.candidate is None or self.last is None:
                raise OutcomePolicyIncomplete('proposal:'+str(result.reason or result.status))
        else:
            # Enumeration exclusion, not a claim that the previous policy or
            # budget UNKNOWN proved that deal impossible.
            if self.last is not None:
                self.model.add_deck_distance_cut(self.last.deck_order)
                self.model.model.clear_hints()
                self.last = None
            attempt = self.model.solve(seed=self.seed+self.ordinal, solver=self.solver,
                maximum_seconds=remaining, maximum_conflicts=250000, search_branching='automatic')
            if attempt.status != 'satisfied':
                raise OutcomePolicyIncomplete('proposal:'+attempt.status)
            self.last = attempt
        self.ordinal += 1
        return self.last


def _satisfaction(compiled, normalized):
    """Apply existing predicate authorities to constants from engine replay."""
    model, solver = cp_model.CpModel(), cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_deterministic_time = 1
    tricks = {}
    for trick in normalized.round_state.completed_tricks:
        actions = tuple(CpSatActionVariablesV1(f'{trick.index}:{index}',
            model.new_constant(play.card.id), model.new_constant(int(play.seat)),
            model.new_constant(int(play.seat)), model.new_constant(int(play.ekka)), model.new_constant(1))
            for index, play in enumerate(trick.plays))
        winner = next(action for action, play in zip(actions, trick.plays, strict=True) if play.seat == trick.winner)
        tricks[trick.index] = CpSatTrickVariablesV1(trick.index, actions, winner)
    context = CpSatPredicateModelContextV1(model, normalized.exact_state_id, tricks, (), model.new_constant(1))
    runtime, lowerer = ScenarioRuntimeVerifierV1(), ScenarioPredicateLoweringRegistryV2()
    literals, extractions, rows = {}, {}, {}
    for predicate in compiled.scenario.predicates():
        if predicate.kind in {'play.action_matches', 'play.trick_card_count', 'play.trick_winner_matches'}:
            lowering = lowerer.lower_predicate(predicate, context)
            literals[predicate.predicate_id] = lowering.literal
            extractions[predicate.predicate_id] = lowering
        else:
            runtime.registry.require(predicate)
            status, observed, failure = runtime._evaluate_predicate(predicate, normalized,
                proof_outcome=None, tactical_evidence=None)
            literals[predicate.predicate_id] = model.new_constant(int(status == 'satisfied'))
            rows[predicate.predicate_id] = dict(predicateId=predicate.predicate_id, kind=predicate.kind,
                status=status, observed=observed, failureCode=failure)
    constraints = reify_scenario_expression(model, compiled.scenario.constraints, literals)
    outcome = reify_scenario_expression(model, compiled.scenario.outcome, literals)
    if solver.solve(model) not in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
        raise OutcomePolicyIncomplete('verification_budget')
    for key, lowering in extractions.items():
        value = lowering.extract(solver)
        rows[key] = dict(predicateId=key, kind=value['kind'],
            status='satisfied' if value['satisfied'] else 'failed', observed=value['observed'], evidence=value)
    constraints_ok, outcome_ok = solver.boolean_value(constraints), solver.boolean_value(outcome)
    core = dict(scenarioHash=compiled.scenario.sha256, exactStateId=normalized.exact_state_id,
        constraintsSatisfied=constraints_ok, outcomeSatisfied=outcome_ok,
        hardPredicatesSatisfied=all(row['status'] == 'satisfied' for row in rows.values()),
        predicates=[rows[key] for key in sorted(rows)],
        status='satisfied' if constraints_ok and outcome_ok else 'semantic_postcondition_miss')
    return {**core, 'sha256': canonical_sha256(core)}


def verify_policy_line(compiled, driver, context, plays):
    receipts = driver.receipts
    prefix = _prefix(context['deckOrder'], context['dealer'],
        [r['action'] for r in receipts if r['decisionKind'] == 'bid'],
        escalations=[dict(actor=r['actor'], action=r['action'].split(':', 1)[0],
            **({'requestedMode': r['action'].split(':', 1)[1]} if ':' in r['action'] else {}))
            for r in receipts if r['decisionKind'] == 'escalation' and r['action'] != driver.engine.escalation_domain.closure_action],
        plays=plays, project_actions=[r['action'] for r in receipts if r['decisionKind'] == 'project'])
    normalized = ExactStateNormalizerV1(driver.rules).normalize(prefix)
    if normalized.round_state.round_score is None or len(normalized.round_state.played_cards) != 32:
        raise AssertionError('policy rollout did not complete an engine round')
    if normalized.round_state.round_score.to_dict() != driver.final_state.round_score.to_dict():
        raise AssertionError('independent engine replay differs from policy rollout')
    satisfaction = _satisfaction(compiled, normalized)
    # Nested count/OR expressions govern satisfaction, not the status of an
    # unselected alternative leaf (e.g. exactly seven of eight trick wins).
    missed = [row['predicateId'] for row in satisfaction['predicates'] if row['status'] != 'satisfied']
    prefixes = [(0, 0)]
    shell = normalized.round_state.clone()
    for count in range(1, 9):
        shell.completed_tricks = normalized.round_state.completed_tricks[:count]
        prefixes.append(tuple(normalized.round_state.round_score.raw_card_points) if count == 8
            else NativeAccelerator._raw_points(shell))
    trajectory = []
    for row in compiled.trajectory:
        side = int(Team[row.team.upper()])
        start, stop = prefixes[row.start], prefixes[row.stop]
        value = stop[side]-stop[1-side]-start[side]+start[1-side]
        trajectory.append(dict(start=row.start, stop=row.stop, team=row.team, value=value,
            minimum=row.minimum, maximum=row.maximum, satisfied=row.minimum <= value <= row.maximum))
    okay = satisfaction['constraintsSatisfied'] and satisfaction['outcomeSatisfied'] and all(row['satisfied'] for row in trajectory)
    return okay, dict(exactTracePrefix=prefix.to_dict(), satisfaction=satisfaction,
        trajectory=trajectory, failedPredicates=missed, score=normalized.round_state.round_score.to_dict(),
        facts=game_facts(normalized.round_state, driver.engine.scoring_domain.last_trick_bonus_raw,
            driver.engine.scoring_domain.buyer_tie_outcome))


def _diversity_key(compiled, state, auction):
    """Joint initial-deal/public-auction symmetry, fixing only spec-named suits."""
    if state.played_cards:
        raise ValueError('outcome diversity requires the initial deal')
    source = state.to_dict(reveal_hands=True)
    statuses = decision_state_card_vector_v1(source)
    named = compiled.source.get('outcome', {}).get('decidingTrick', {})
    suits = {int(Card.parse(named[key]).suit) for key in ('winningCard', 'beatenCard') if key in named}
    if compiled.source.get('trumpSuit'):
        suits.add(int(Suit.parse(compiled.source['trumpSuit'])))
    candidates = []
    for mapping in permutations(range(4)):
        if any(mapping[s] != s for s in suits):
            continue
        vector = [8]*32
        for card, status in enumerate(statuses):
            vector[mapping[card//8]*8+card % 8] = status
        bids = [{**row, **(dict(trumpSuit=Suit(mapping[int(Suit.parse(row['trumpSuit']))]).name.lower())
            if row.get('trumpSuit') is not None else {})} for row in auction]
        candidates.append(canonical_sha256(dict(policy='outcome_joint_suit_symmetry_v2',
            hands=vector, actor=int(state.current_player), buyer=int(state.buyer), dealer=int(state.dealer),
            purchase=mapping[state.purchase_card.id//8]*8+state.purchase_card.id % 8,
            trump=mapping[int(state.trump_suit)] if state.trump_suit is not None else None,
            auction=bids, mode=state.contract.value)))
    return min(candidates)


def _rollout_worker(source, rules_data, policy, seed, count, deadline, shard, workers, output, strategy,
                    story_cap=3, story_settings=None):
    started, cpu = time.monotonic(), time.process_time()
    rules = RuleBook(rules_data)
    compiled = compile_outcome_spec(source, rules)
    stream = DealProposalStream(compiled, rules, policy, seed, deadline, (workers, shard, 1))
    driver = OutcomePolicyDriver(rules, compiled, policy, seed=seed, deadline=deadline, maximum_threads=1)
    stages, rejects, attempted, accepted, seen = Counter(), Counter(), [], [], set()
    seen_proposals = set()
    accepted_decks = []
    stories = Counter()
    search = OutcomeSearch(compiled, seed) if strategy == 'guided' and driver.spec_fixed_contract else None
    if time.monotonic() >= deadline:
        rejects['worker_startup_budget'] += 1

    def fresh():
        stages['cpSatAttempts'] += 1
        proposal = stream.next(15)
        if not driver.spec_fixed_contract:
            proposal = driver.refine_auction(stream.model, proposal, stream.solver,
                driver.remaining, 250000)
            if proposal.status != 'satisfied':
                raise OutcomePolicyIncomplete('open_auction:'+proposal.status)
        stages['cpSatProposed'] += 1
        context = dict(deckOrder=list(proposal.deck_order), dealer=stream.model.dealer.name.lower(),
            auction=proposal.auction.to_dict(), escalation=dict(observation=dict(actions=[])),
            rootCpSolutionHash=proposal.solution_hash, rootCpModelHash=proposal.model_hash)
        bids = [{key: a[key] for key in ('actor', 'action', 'trumpSuit') if a.get(key) is not None}
            for a in context['auction']['completion']['actions']]
        initial = ExactStateNormalizerV1(rules).normalize(_prefix(context['deckOrder'], context['dealer'], bids)).round_state
        context['cardOwners'] = [next(seat for seat, mask in enumerate(initial.hands) if mask & (1 << c)) for c in range(32)]
        context['purchaseCard'] = initial.purchase_card.id
        return context
    try:
        while len(accepted) < count and time.monotonic() < deadline:
            tick, used = time.monotonic(), time.process_time()
            row = dict(ordinal=len(attempted), stage='proposal')
            driver.receipts, driver.events, driver.final_state = [], [], None
            try:
                stages['proposalAttempts'] += 1
                if strategy not in {'rollout', 'guided'}:
                    result = construct_outcome(compiled, rules, policy, seed=seed+len(attempted)*workers,
                        maximum_seconds=min(60, deadline-time.monotonic()), maximum_attempts=1, refinement=strategy)
                    stages['proposed'] += sum(r['proposals'] > 0 for r in result['attempts'])
                    if result['accepted'] is None:
                        raise OutcomePolicyIncomplete(next(iter(result['rejectReasons']), 'unknown'))
                    record = result['accepted']
                    prefix = record['exactTracePrefix']
                    initial = ExactStateNormalizerV1(rules).normalize(_prefix(prefix['deckOrder'],
                        prefix['dealer'], prefix['auctionActions']))
                    key = _diversity_key(compiled, initial.round_state, prefix['auctionActions'])
                    final = ExactStateNormalizerV1(rules).normalize(prefix).round_state
                    record['facts'] = game_facts(final, driver.engine.scoring_domain.last_trick_bonus_raw,
                        driver.engine.scoring_domain.buyer_tie_outcome)
                else:
                    context, provenance = search.next(fresh) if search is not None else (fresh(), dict(kind='cp_sat'))
                    stages['proposed'] += 1
                    if provenance['kind'] != 'cp_sat':
                        stages[provenance['kind']+'Proposed'] += 1
                    row['search'] = provenance
                    driver.seed = seed+len(attempted)*workers
                    driver.deadline = deadline
                    bids = [{key: action[key] for key in ('actor', 'action', 'trumpSuit') if action.get(key) is not None}
                        for action in context['auction']['completion']['actions']]
                    initial = ExactStateNormalizerV1(rules).normalize(_prefix(context['deckOrder'], context['dealer'], bids))
                    key = _diversity_key(compiled, initial.round_state, bids)
                    if key in seen_proposals:
                        raise OutcomePolicyIncomplete('duplicate_proposal_symmetry')
                    seen_proposals.add(key)
                    if any(deck_distance(context['deckOrder'], deck) < 4 for deck in accepted_decks):
                        raise OutcomePolicyIncomplete('duplicate_deck_distance')
                    row['stage'] = 'policy'
                    stages['policyStarted'] += 1
                    plays = driver.construct_line(context)
                    stages['policyCompleted'] += 1
                    row['stage'] = 'verification'
                    okay, record = verify_policy_line(compiled, driver, context, plays)
                    stages['verified'] += 1
                    if search is not None:
                        search.observe(context, driver.final_state, record, okay, provenance)
                    if not okay:
                        row['failedPredicates'] = record['failedPredicates']
                        row['failedPredicateKinds'] = dict(Counter(p['kind']
                            for p in record['satisfaction']['predicates'] if p['status'] != 'satisfied'))
                        row['trajectoryMisses'] = sum(not item['satisfied'] for item in record['trajectory'])
                        raise OutcomePolicyIncomplete('outcome_miss')
                    record.update(seed=driver.seed, specHash=compiled.sha256, rulesHash=rules.sha256,
                        policyProfileHash=canonical_sha256(policy), seatPolicies=dict(compiled.seat_policies),
                        decisionReceipts=driver.receipts, proposalHash=canonical_sha256(context['deckOrder']),
                        rootCpSolutionHash=context['rootCpSolutionHash'], rootCpModelHash=context['rootCpModelHash'],
                        proposalModelHash=context['rootCpModelHash'] if provenance['kind'] == 'cp_sat' else None,
                        search=provenance, diversityKey=key,
                        qualification='outcome_verified_under_declared_policies')
                if time.monotonic() >= deadline:
                    raise OutcomePolicyIncomplete('campaign_time_budget')
                row['stage'] = 'admission'
                stages['accepted'] += 1
                if key in seen:
                    raise OutcomePolicyIncomplete('duplicate_symmetry')
                deck = record['exactTracePrefix']['deckOrder']
                if any(deck_distance(deck, previous) < 4 for previous in accepted_decks):
                    raise OutcomePolicyIncomplete('duplicate_deck_distance')
                story = story_facts(record['facts'], compiled, story_settings)
                if stories[story['storyHash']] >= story_cap:
                    raise OutcomePolicyIncomplete('story_variant_cap')
                seen.add(key)
                accepted_decks.append(deck)
                stories[story['storyHash']] += 1
                record.update(story)
                stages['distinctAccepted'] += 1
                document = {**record, 'sha256': canonical_sha256(record)}
                path = Path(output)/f'worker-{shard}-scenario-{len(accepted):06d}.json'
                path.write_text(json.dumps(document, ensure_ascii=False), encoding='utf-8')
                identity = canonical_sha256(dict(spec=compiled.sha256,
                    policy=canonical_sha256(policy), prefix=record['exactTracePrefix']))
                accepted.append(dict(id='scenario-'+identity, path=str(path), sha256=document['sha256'],
                    diversityKey=key, seed=record['seed'], score=record.get('score'),
                    deckOrder=deck, storyHash=story['storyHash'], parent=record.get('search', {}).get('parent')))
                if search is not None:
                    search.admit(context, driver.final_state, path.stem, provenance, seed=driver.seed)
                row['reason'] = 'accepted'
            except (OutcomePolicyIncomplete, InterruptedError) as error:
                row['reason'] = getattr(error, 'reason', 'cancelled')
                evidence = getattr(error, 'evidence', None) or {}
                if 'solvedWorlds' in evidence:
                    row['solvedWorldsBeforeStop'] = evidence['solvedWorlds']
                    row['incompleteWorldReasons'] = dict(Counter(world.get('stopReason', 'unknown')
                        for world in evidence.get('cohort', {}).get('worlds', []) if world['status'] != 'exact'))
                rejects[row['reason']] += 1
                if row['stage'] == 'proposal' and row['reason'].endswith('infeasible'):
                    attempted.append(row)
                    break
            except Exception as error:
                if isinstance(error, (OSError, MemoryError, NativeUnavailable, ImportError)) or row['stage'] == 'admission':
                    raise
                if isinstance(error, RuntimeError) and str(error) in {'no legal hidden deals could reproduce the public history',
                        'public project history leaves no feasible hidden allocation',
                        'public facts leave no feasible hidden allocation'}:
                    row['reason'] = 'informed_sampling_incomplete'
                else:
                    row.update(reason='unexpected_game_error:'+type(error).__name__,
                        error=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
                rejects[row['reason']] += 1
            row.update(wallSeconds=time.monotonic()-tick, cpuSeconds=time.process_time()-used)
            if strategy in {'rollout', 'guided'}:
                row['completedCardDecisions'] = sum(r['decisionKind'] == 'play' for r in driver.receipts)
            attempted.append(row)
            # Preserve partial output without retrying a possibly deterministic defect.
            if row['reason'].startswith('unexpected_game_error:'):
                break
            if len(attempted) % 5 == 0:
                print(f'worker={shard} proposals={stages["proposed"]} accepted={len(accepted)}', flush=True)
    finally:
        driver.close()
    result = dict(worker=shard, stages=dict(stages), rejectReasons=dict(rejects), attempts=attempted,
        accepted=accepted, wallSeconds=time.monotonic()-started,
        cpuSeconds=time.process_time()-cpu, processId=os.getpid(), processCpuSeconds=time.process_time(),
        neighbourhoods=search.neighbourhoods if search is not None else {},
        quotaCompleted=len(accepted) == count,
        budgetCensored=(len(accepted) < count or any('budget' in reason or 'incomplete' in reason
            or reason.startswith('unexpected_game_error:') or reason == 'cancelled' for reason in rejects)))
    (Path(output)/f'worker-{shard}.json').write_text(json.dumps(result), encoding='utf-8')
    return result


def generate_outcomes(compiled, rules, policy, *, count, seed, output, workers=4,
                      maximum_seconds=270, strategy='guided', maximum_story_variants=3, story_settings=None):
    """Default volume path; returns an honest partial and its complete funnel."""
    compiled.require_integrity()
    policy = validate_outcome_policy(policy)
    if compiled.rules_hash != rules.sha256 or 'cardPolicy' not in policy:
        raise ValueError('rollout generation requires a rules-bound DD profile')
    if type(workers) is not int or not 1 <= workers <= 4 or type(count) is not int or count < 1:
        raise ValueError('positive count and one to four workers are required')
    if type(maximum_seconds) not in {int, float} or not 0 < maximum_seconds <= 270:
        raise ValueError('local volume runs must fit 4.5 minutes')
    if strategy not in {'guided', 'rollout', 'first_deviation', 'forward_rollout'}:
        raise ValueError('unsupported construction strategy')
    if type(maximum_story_variants) is not int or maximum_story_variants < 1:
        raise ValueError('maximum_story_variants must be a positive integer')
    story_settings = story_granularity(story_settings)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started, parent_cpu = time.monotonic(), time.process_time()
    # Reserve final merge/cleanup time inside the requested run envelope.
    deadline = started+max(.01, maximum_seconds-3)
    args = (dict(compiled.source), rules.data, policy, seed, count, deadline)
    if workers == 1:
        results = [_rollout_worker(*args, 0, 1, str(output), strategy, maximum_story_variants, story_settings)]
    else:
        spawn = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=workers, mp_context=spawn) as pool:
            pending = [pool.submit(_rollout_worker, *args[:3], seed+i, count, deadline,
                i, workers, str(output), strategy, maximum_story_variants, story_settings) for i in range(workers)]
            results = [job.result() for job in pending]
    stages, rejects = Counter(), Counter()
    for result in results:
        stages.update(result['stages'])
        rejects.update(result['rejectReasons'])
    accepted, merge_rejects = _merge_distinct([r for result in results for r in result['accepted']],
        count, maximum_story_variants)
    rejects.update(merge_rejects)
    stages['shardDistinctAccepted'] = stages['distinctAccepted']
    stages['fixedCountCut'] = stages['shardDistinctAccepted']-len(accepted)-sum(merge_rejects.values())
    stages['distinctAccepted'] = len(accepted)
    plays = catalogue_rows(accepted, compiled, story_settings)
    stages['storyDistinctAccepted'] = len({row['storyHash'] for row in plays})
    neighbourhoods = write_neighbourhoods(output, results, accepted)
    wall = time.monotonic()-started
    if workers == 1:
        cpu = sum(result['cpuSeconds'] for result in results)
    else:
        process_cpu = {}
        for result in results:
            pid = result['processId']
            process_cpu[pid] = max(process_cpu.get(pid, 0), result['processCpuSeconds'])
        cpu = sum(process_cpu.values())+time.process_time()-parent_cpu
    core = dict(schemaVersion=1, status='complete' if len(accepted) >= count else 'partial',
        requestedCount=count, accepted=accepted, distinctAccepted=len(accepted),
        storyDistinctAccepted=stages['storyDistinctAccepted'],
        maximumStoryVariants=maximum_story_variants, storyCapScope='merged_batch',
        storyGranularity=story_settings, mergeOrder='diversityKey,id; fixed count cut',
        workerSeedStreams=[dict(worker=i, initialSeed=seed+i, stride=workers, quota=count) for i in range(workers)],
        reproducible=all(not r.get('budgetCensored', True) for r in results),
        reproducibilityScope='fixed shard quotas; wall/proof-budget-censored runs are not reproducible',
        retained=accepted[:count], strategy=strategy, workers=workers,
        plays=plays, searchProfile=SEARCH_PROFILE if strategy == 'guided' else None,
        neighbourhoodTotals=neighbourhoods,
        specHash=compiled.sha256, policyProfileHash=canonical_sha256(policy),
        outcomeSpec=compiled.to_dict(), policyProfile=policy, seed=seed, rulesHash=rules.sha256,
        proposalScope='CP-SAT deal/auction/project projection; full outcome checked after policy rollout',
        nativeThreadsPerWorker=1,
        minimumDeckHammingDistance=4, deckDistanceScope='merged_batch', canonicalizeDealPackets=True,
        wallSeconds=wall, cpuSeconds=cpu, stages=dict(stages), rejectReasons=dict(rejects),
        proposalSuccessFraction=stages['proposed']/max(1, stages['proposalAttempts']),
        stageFractions={key: stages[key]/max(1, stages['proposed'])
            for key in ('policyStarted', 'policyCompleted', 'verified', 'accepted', 'distinctAccepted')},
        acceptedPerHour=len(accepted)*3600/wall, cpuSecondsPerAccepted=cpu/len(accepted) if accepted else None,
        workerReceipts=[str(output/f'worker-{i}.json') for i in range(workers)])
    result = {**core, 'sha256': canonical_sha256(core)}
    (output/'catalogue.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    write_exports(output, plays)
    return result
