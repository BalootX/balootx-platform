"""Private equal-share family-stream experiment; no publication or novelty credit.

Use run_local_capped.ps1 on Windows, or the existing resource/freeze-controlled
research launcher on Linux. --authority is a retained configuration snapshot,
not the process-active bundle. Its schema-4 strict v6 gate is required unchanged.
The output contains private hands/proof strategies and must remain private.
--families selects catalog IDs for the round-flip and guard lanes respectively;
their tag compatibility and native construction support are validated before launch.
Saved openings visit each configured decision before rotating their starting seed.
--proof-workers reserves independent singleton CP-SAT jobs plus one producer;
the existing launcher enforces the aggregate CPU/memory and priority ceiling.
--tail-first privately compares backward construction with the existing producer;
it requires an explicitly frozen execution without familyOpeningMode, not a bypass.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import math
import time
from collections import Counter
from pathlib import Path

from baloot_platform.core.rules import RuleBook
from baloot_platform.core.cards import FULL_DECK as FULL_DECK
from baloot_platform.core.rule_source_catalog import load_exact_rule_source_catalog
from baloot_platform.config import profile_bundle_from_resolved
from baloot_platform.filesystem import atomic_write_text
from baloot_platform.research.family_catalog import ResearchFamilyCatalogV1, TagQueryV1
from baloot_platform.research.family_proposal_stream import (
    build_family_target_v1, family_proposal_stream_v1, validate_family_target_scenario_v1,
    tail_first_proposal_stream_v1 as tail_first_proposal_stream,
)
from baloot_platform.research.lesson_tags import derive_trace_lesson_tags_v1
from baloot_platform.research.ortools_exact_state import ExactNormalizationStopped, OrToolsExactStateNormalizerV1
from baloot_platform.research.ortools_research_executor import (
    OrToolsUnifiedResearchExecutorV1, _BidQualityCacheV1, _campaign_variant_from_execution_v1,
    rules_preserving_symmetry_key_v1,
)
from baloot_platform.research.scenario_backend import ScenarioResearchAuthorityV1
from baloot_platform.research.scenario_contracts import (
    ExactTracePrefixV1, ResearchScenarioV1, ScenarioPlayingQualityPolicyV1,
    ScenarioQualityExecutionBudgetV1, canonical_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
LANES = (('round-flip', 'negative-project', {'impact': ['round-flip']}),
         ('guard/protect', 'stopper_guard', {'technique': ['guard/protect']}))


def retain(path, document):
    """Never overwrite a previous attempt, including failed attempts."""
    if path.exists():
        raise ValueError(f'evidence already exists: {path.name}')
    if path.suffix == '.gz':
        with path.open('xb') as destination, gzip.GzipFile(fileobj=destination, mode='wb', mtime=0) as stream:
            stream.write((json.dumps(document, sort_keys=True, ensure_ascii=False) + '\n').encode('utf-8'))
    else:
        atomic_write_text(path, json.dumps(document, sort_keys=True, ensure_ascii=False, indent=2) + '\n')


def strict_authority(challenge):
    authority = ScenarioResearchAuthorityV1.from_challenge_profile(challenge)
    execution = authority.quality_execution
    interest = execution.video_interestingness if execution is not None else None
    if (authority.backend != 'ortools_cp_sat_v3' or interest is None
            or interest['schemaVersion'] != 4
            or interest['baselinePolicyId'] != 'observer_safe_public_two_trick_control_v6'
            or interest['allowTieRuleOnlyLessons']
            or not execution.featured_play_native_proof or not execution.exact_tail_proof):
        raise ValueError('experiment requires an explicit frozen strict v6 authority; no fallback to bundled defaults')
    return authority


def tail_quality_policy(document):
    policy = ScenarioPlayingQualityPolicyV1.build(document)
    if policy.visibility != 'actor_information_set' or policy.mode != 'budgeted':
        raise ValueError('experiment requires an actor-visible bounded prefix with the declared exact tail')
    return policy


def freeze_inputs(document, execution_document, rules):
    """Rebuild all identities through the existing draft bundle authority."""
    snapshot = document['snapshot']
    if snapshot.get('sha256') != canonical_sha256({key: value for key, value in snapshot.items() if key != 'sha256'}):
        raise ValueError('source configuration snapshot checksum mismatch')
    if any(component['sha256'] != canonical_sha256(component['payload']) for component in snapshot['components'].values()):
        raise ValueError('source configuration component checksum mismatch')
    if 'sha256' not in execution_document:
        raise ValueError('experiment requires an immutable checksummed qualityExecution document')
    profiles = {key: copy.deepcopy(value['payload']) for key, value in document['snapshot']['components'].items()}
    if profiles['challenge'].get('scenarioResearchV4', {}).get('backend') != 'ortools_cp_sat_v3':
        raise ValueError('experiment requires a retained native V3 source configuration')
    execution = ScenarioQualityExecutionBudgetV1.build(execution_document)
    profiles['rules'] = copy.deepcopy(rules.data)
    # Profile payloads are raw configuration; the checked receipt hash is derived.
    profiles['challenge']['scenarioResearchV4']['qualityExecution'] = {
        key: value for key, value in execution.to_dict().items() if key != 'sha256'}
    source_catalog = load_exact_rule_source_catalog(rules, ROOT)
    profiles['certification']['candidate_profile_identity'] = dict(id=rules.id, version=rules.version, sha256=rules.sha256)
    profiles['certification']['rule_source_catalog'] = dict(path=f'rules/source_catalog_{rules.id}.yaml',
        id=source_catalog.data['catalog']['id'], version=source_catalog.data['catalog']['version'], sha256=source_catalog.sha256)
    bundle = profile_bundle_from_resolved(dict(profiles=profiles, bundle=dict(bundleId='private_tag_experiment',
        version=rules.version, status='draft', mode=document['snapshot']['mode'])), root=ROOT)
    authority = strict_authority(bundle.component('challenge'))
    return bundle, authority, source_catalog


def auction_seed_hints(directory, rules_hash):
    """Seed hints only: the current native provider must revalidate each auction."""
    hints = []
    for path in sorted(directory.glob('qualified-*.json.gz')):
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            row = json.load(stream)
        receipt, seed = row['receipt'], row['seed']
        if (type(seed) is not int or seed < 0 or receipt['rulesHash'] != rules_hash
                or receipt['status'] != 'complete' or receipt['operation'] != 'generate'
                or row['prefix'] != receipt['finalPrefix']):
            raise ValueError('auction seed hints require same-rules generated auction records')
        outcome = receipt['auctionOutcome']
        if outcome['contract'] == 'sun' and not outcome['allPassed']:
            hints.append(dict(seed=seed, dealer=row['prefix']['dealer'], buyer=outcome['buyer'],
                              receiptHash=receipt['sha256']))
    if not hints or len({row['seed'] for row in hints}) != len(hints):
        raise ValueError('auction seed hints require unique compatible Sun openings')
    return hints


def _opening_and_context_ordinal(hints, pair, *, decision_count=1):
    rotation, decision = divmod(pair, decision_count)
    if not hints:
        seats = ('north', 'west', 'south', 'east')
        opening = {'dealer': seats[rotation % 4], 'buyer': seats[(rotation // 4) % 4]}
        return opening, rotation // 16 * decision_count + decision
    openings = {}
    for hint in hints:
        openings.setdefault((hint['dealer'], hint['buyer']), hint)
    contexts = tuple(openings.values())
    return contexts[rotation % len(contexts)], rotation // len(contexts) * decision_count + decision


def _binding_and_auction_seeds(hints, opening, ordinal, *, decision_count=1):
    if not hints:
        return ordinal, None
    seeds = [row['seed'] for row in hints
             if row['dealer'] == opening['dealer'] and row['buyer'] == opening['buyer']]
    if not seeds:
        raise ValueError('auction seed hints require a matching opening')
    rotation, decision = divmod(ordinal, decision_count)
    group, offset = divmod(rotation, len(seeds))
    binding = group * decision_count + decision
    # Scan positions before advancing the starting seed; rotate before truncation.
    return binding, seeds[offset:] + seeds[:offset]


def target_and_scenario(catalog, lane, ordinal, *, opening=None, lanes=None):
    """Enable only the selected templates in a detached, retained draft."""
    lanes = LANES if lanes is None else lanes
    raw = copy.deepcopy(catalog.to_dict())
    raw.pop('sha256')
    for row in raw['families']:
        row['enabled'] = row['id'] in {entry[1] for entry in lanes}
    draft = ResearchFamilyCatalogV1.build(raw)
    family = next((row for row in draft.families if row.family_id == lane[1]), None)
    if family is None:
        raise ValueError(f'experiment family is not in the selected catalog: {lane[1]}')
    choices = family.parameters.choices(('sun',))
    # Cover cheaper late roots immediately, then cycle every configured band.
    choices['playIndex'] = sorted(choices['playIndex'], reverse=True)
    bindings = {}
    cursor = ordinal
    for key in ('playIndex', 'seat', 'suit', 'rank', 'projectType', 'contract'):
        bindings[key] = choices[key][cursor % len(choices[key])]
        cursor //= len(choices[key])
    target = build_family_target_v1(dict(schemaVersion=1, catalog=draft.to_dict(), familyId=family.family_id,
        tagQuery={**lane[2], 'mode': ['sun'], 'trickBand': [f"trick-{bindings['playIndex'] // 4 + 1}"]},
        bindings=bindings, dealer=opening['dealer'] if opening else ('north', 'west', 'south', 'east')[ordinal % 4],
        buyer=opening['buyer'] if opening else ('north', 'west', 'south', 'east')[(ordinal // 4) % 4], trump=None))
    index = bindings['playIndex']
    scenario = ResearchScenarioV1.build(dict(schemaVersion=1, origin={'kind': 'search'},
        constraints=family.instantiate(bindings).to_dict(),
        decisionPoint={'kind': 'play', 'selector': {'remainingCards': 8 - index // 4,
            'trickIndex': index // 4, 'position': index % 4, 'actingSeat': bindings['seat']}},
        outcome={'predicateId': 'PRED-EXPERIMENT-REPORT', 'kind': 'auction.contract', 'version': 1,
            'hardness': 'report', 'arguments': {'contract': 'sun'}}))
    validate_family_target_scenario_v1(target, scenario)
    return target, scenario


def accepted_packet(execution, *, rules, scenario, target, maximum_seconds):
    """Only consume canonical accepted executor objects, not proposal documents."""
    if execution.status != 'qualified' or execution.candidate is None:
        raise ValueError('a proposal or rejected execution is not an accepted witness')
    trace = execution.candidate.trace
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        ExactTracePrefixV1.build(trace['exactTracePrefix']), decision_point=scenario.decision_point,
        maximum_seconds=maximum_seconds)
    tags = derive_trace_lesson_tags_v1(trace, normalized=normalized, rules=rules)
    if tags is None or tags != trace.get('lessonTags') or not TagQueryV1.build(target['tagQuery']).matches(tags['tags']):
        raise ValueError('accepted witness does not carry its independently derived native target tags')
    receipt = trace['playingQualityEvidence']['receipt']
    group = receipt['postAuction']['projectPlay']
    artifact_hash = tags['sources']['proofArtifactHash']
    artifact = group['artifacts'][artifact_hash]
    baseline = group.get('baselineEvidence', {}).get(artifact_hash)
    tie = group.get('tieRuleEvidence', {}).get(artifact_hash)
    baseline_action = tags['sources'].get('baselineAction')
    paired = {'status': 'missing_baseline_replay_line', 'baselineAction': baseline_action}
    if 'rootWitnesses' in artifact:
        selected_action = artifact['proof']['selectedAction']
        roots = {row['action']: row['value'] for row in artifact['proof']['rootActions']}
        lines = artifact['rootWitnesses']
        if not isinstance(lines, list) or len({row['action'] for row in lines}) != len(lines):
            raise ValueError('accepted root replay evidence is malformed')
        witnesses = {row['action']: row for row in lines}
        for action in (selected_action, baseline_action):
            row = witnesses.get(action)
            if (action not in roots or not row or not row.get('worldId') or not row.get('principalVariation')
                    or row['principalVariation'][0] != action
                    or row.get('worstCaseValue') != roots.get(action)):
                raise ValueError('accepted root replay does not bind its action and value')
        paired = {'status': 'exact_root_worst_cases', 'baselineAction': baseline_action,
            'selected': witnesses[selected_action], 'baseline': witnesses[baseline_action],
            'scope': 'independent_worst_case_worlds; not_actual_deal_or_inference_ablation'}
    # Native bounds are not concrete root-specific replay lines or an ablation.
    packet = dict(schemaVersion=1, executionHash=execution.sha256, semanticHash=execution.semantic_hash,
        lessonTags=tags, decisionPoint=scenario.decision_point, exactTracePrefix=trace['exactTracePrefix'],
        exactNormalization=normalized.to_dict(), classifier=tags.get('classifier'),
        sourceArtifact=artifact, baselineArtifact=baseline, tieRuleEvidence=tie,
        chosenContinuation=receipt.get('finalPrefix'), chosenContinuationScope='recorded_world; not a paired adversarial witness',
        nativeContinuation=trace.get('playPolicyContinuation'),
        pairedContinuations=paired,
        inferenceAblation={'status': 'not_run'}, audit={'status': 'pending', 'newLesson': None},
        qualificationBoundary='Canonical executor accepted; native tags are editorial, novelty/causal audit pending.')
    return {**packet, 'sha256': canonical_sha256(packet)}




def _percentiles(values):
    values = sorted(values)
    return {f'p{p}': values[max(0, math.ceil(len(values) * p / 100) - 1)] if values else None for p in (50, 95, 99)}


def run(args):
    tail_first = getattr(args, 'tail_first', False)
    if type(tail_first) is not bool:
        raise ValueError('tail_first must be a boolean')
    if type(args.proof_workers) is not int or not 1 <= args.proof_workers <= 10:
        raise ValueError('proof_workers must be an integer in 1..10 within the existing launcher grant')
    start_pair = getattr(args, 'start_pair', 0)
    if type(start_pair) is not int or start_pair < 0:
        raise ValueError('start_pair must be a nonnegative integer')
    for key in ('seconds', 'slice_seconds'):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if args.seconds > 3600 or args.slice_seconds * 2 > args.seconds or args.seed < 0:
        raise ValueError('bounded experiment requires 0 < paired slices <= generation <= one hour and nonnegative seed')
    selected_families = getattr(args, 'families', None)
    if selected_families is not None and (not isinstance(selected_families, (list, tuple))
            or len(selected_families) != len(LANES)
            or any(type(value) is not str or not value.strip() for value in selected_families)):
        raise ValueError('families requires two catalog family IDs: round-flip, guard/protect')
    lanes = tuple((label, selected_families[index] if selected_families is not None else family_id, query)
                  for index, (label, family_id, query) in enumerate(LANES))
    supplied = json.loads(args.authority.read_text(encoding='utf-8'))
    rules = RuleBook.load(args.rules)
    catalog = ResearchFamilyCatalogV1.load(args.catalog)
    execution_document = json.loads(args.execution.read_text(encoding='utf-8'))
    bundle, authority, source_catalog = freeze_inputs(supplied, execution_document, rules)
    policy = tail_quality_policy(json.loads(args.policy.read_text(encoding='utf-8')))
    hint_directory = getattr(args, 'auction_library', None)
    if tail_first and (authority.quality_execution.family_opening_mode is not None or hint_directory is not None):
        raise ValueError('tail-first requires explicitly frozen plain proposals without an auction seed library')
    hints = auction_seed_hints(hint_directory, rules.sha256) if hint_directory else None
    profile = policy.decisions['play']['budgetProfile']
    proof, search = authority.proof_budgets['profiles'][profile], authority.search_budgets['profiles'][profile]
    # Fail on invalid targets/gates before creating output or dispatching a solver.
    decision_counts = {}
    for lane in lanes:
        family = next((family for family in catalog.families if family.family_id == lane[1]), None)
        if family is None:
            raise ValueError(f'experiment family is not in the selected catalog: {lane[1]}')
        decision_counts[lane[0]] = len(family.parameters.choices(('sun',))['playIndex'])
        opening, context_ordinal = _opening_and_context_ordinal(
            hints, start_pair, decision_count=decision_counts[lane[0]])
        binding_ordinal, _ = _binding_and_auction_seeds(
            hints, opening, context_ordinal, decision_count=decision_counts[lane[0]])
        target_and_scenario(catalog, lane, binding_ordinal, opening=opening, lanes=lanes)
    # Archived sources have no .git; the existing sync carrier owns custody.
    revision, tree = args.source_commit, args.source_tree
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    frozen = dict(schemaVersion=1, sourceCommit=revision, sourceTree=tree, rules=rules.data, rulesHash=rules.sha256,
        sourceIdentityAuthority='launcher_provenance',
        sourceSnapshot=supplied['snapshot'], sourceSnapshotHash=canonical_sha256(supplied['snapshot']),
        configurationSnapshot=bundle.configuration_snapshot(), draftBundleHash=bundle.sha256,
        authority=authority.to_dict(), familyCatalog=catalog.to_dict(), playingQuality=policy.to_dict(),
        operationalBudget={'generationSeconds': args.seconds, 'sliceSeconds': args.slice_seconds,
            'seedStart': args.seed, 'proofWorkers': args.proof_workers, 'startPair': start_pair,
            'construction': 'native_tail_first_proposal_v1' if tail_first else 'qualified_history_v1',
            'targetSchedule': 'opening_decision_seed_binding_cartesian_v3' if hints else 'opening_decision_binding_cartesian_v2',
            'allocationShares': {lane[0]: .5 for lane in lanes},
            'laneFamilies': {lane[0]: lane[1] for lane in lanes}},
        acceptedLessonLibraryStatus='not_supplied; audited novelty cannot be calculated', privateDraftOnly=True)
    if hints is not None:
        frozen['auctionSeedHints'] = {'sha256': canonical_sha256(hints), 'openings': hints,
                                     'scope': 'source seeds only; current native revalidation required'}
    frozen_hash = canonical_sha256(frozen)
    retain(args.output / 'authority.json', {**frozen, 'sha256': frozen_hash})
    started = time.perf_counter()
    deadline = started + args.seconds
    seen, packets, summaries, latency = set(), [], [], []
    funnels = {lane[0]: Counter() for lane in lanes}
    rejection_reasons = {lane[0]: Counter() for lane in lanes}
    cpu_seconds = 0.0
    bid_cache = _BidQualityCacheV1(authority.cache_limits['maximumEntries'], authority.cache_limits['maximumBytes'])
    try:
        spent = {lane[0]: 0.0 for lane in lanes}
        pair = start_pair
        while time.perf_counter() < deadline and any(seconds < args.seconds / 2 for seconds in spent.values()):
            for lane in lanes:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                slice_seconds = min(args.slice_seconds, args.seconds / 2 - spent[lane[0]], remaining)
                if slice_seconds <= 0:
                    continue
                ordinal = pair * 2 + lanes.index(lane)
                opening, context_ordinal = _opening_and_context_ordinal(
                    hints, pair, decision_count=decision_counts[lane[0]])
                binding_ordinal, source_seeds = _binding_and_auction_seeds(
                    hints, opening, context_ordinal, decision_count=decision_counts[lane[0]])
                target, scenario = target_and_scenario(catalog, lane, binding_ordinal, opening=opening, lanes=lanes)
                raw = authority.quality_execution.to_dict()
                raw.pop('sha256', None)
                raw.pop('familyCatalog', None)
                raw.update(candidateSource='runtime_proposal_stream_v1', proposalSeedCount=search['maximumCandidates'],
                    auditConstructedTrace=True, familyTarget=target, proposalPlayIndices=[target['bindings']['playIndex']],
                    exactPlayFromTrick=target['bindings']['playIndex'] // 4 + 1)
                execution_budget = ScenarioQualityExecutionBudgetV1.build(raw)
                if source_seeds is not None:
                    source_seeds = source_seeds[:execution_budget.proposal_seed_count]
                slice_bundle, slice_authority, _ = freeze_inputs(
                    {'snapshot': bundle.configuration_snapshot()}, execution_budget.to_dict(), rules)
                executor = OrToolsUnifiedResearchExecutorV1(rules, bundle_revision_id='private_tag_experiment',
                    bundle_hash=slice_bundle.sha256, backend_id='ortools_cp_sat_v3',
                    rule_source_catalog={'data': source_catalog.data, 'sha256': source_catalog.sha256},
                    quality_policy=policy,
                    quality_execution=execution_budget, quality_proof_profiles=authority.proof_budgets['profiles'],
                    quality_search_profiles=authority.search_budgets['profiles'], bid_quality_cache=bid_cache)
                slice_started = time.perf_counter()
                parameters = dict(maximumCampaignSeconds=slice_seconds,
                    maximumSeconds=min(search['maximumWallSeconds'], execution_budget.maximum_wall_seconds),
                    maximumCandidates=search['maximumCandidates'], maximumConflicts=search['maximumConflicts'],
                    maximumLineCountLimbs=authority.refusal_thresholds['maximumLineCountLimbs'],
                    seed=args.seed + ordinal * search['maximumCandidates'])
                progress, native_progress = [], []
                def report_native(payload):
                    native_progress.append(dict(payload))
                    if payload.get('proposalDisposition'):
                        print(json.dumps({'slice': ordinal, 'nativeProgress': payload}), flush=True)
                request = policy.request_for('play', target['bindings']['seat'],
                    play_index=target['bindings']['playIndex'], execution_budget=execution_budget)
                if tail_first:
                    proposals = tail_first_proposal_stream(rules, execution_budget, parameters,
                        scenario=scenario, request=request, proof=proof, search=search, playing_quality=policy,
                        campaign_started=slice_started, cancel_check=lambda: time.perf_counter() >= deadline,
                        progress_callback=progress.append)
                else:
                    proposals = family_proposal_stream_v1(rules, execution_budget, parameters,
                        campaign_started=slice_started, cancel_check=lambda: time.perf_counter() >= deadline,
                        progress_callback=progress.append, auction_provider=executor.construct_qualified_auction,
                        source_seeds=source_seeds, playing_quality=policy)
                campaign = executor.execute_proposal_stream(scenario, request, proposals=proposals, proof_worker_count=args.proof_workers,
                    maximum_campaign_seconds=slice_seconds, maximum_candidates=search['maximumCandidates'],
                    result_count=search['maximumCandidates'], symmetry_policy_id='rules_preserving_v1',
                    allow_symmetry_only_variants=False,
                    minimum_decision_state_distance=authority.diversity_defaults['minimumDecisionStateHammingDistance'],
                    maximum_proof_nodes=proof['maximumNodes'], maximum_proof_depth=proof['maximumDepth'],
                    maximum_hidden_worlds=proof['maximumHiddenWorlds'], maximum_conflicts=search['maximumConflicts'],
                    policy_search=proof.get('policySearch'),
                    maximum_prefix_lift_attempts=search['maximumPrefixLiftAttempts'], maximum_seconds=parameters['maximumSeconds'],
                    cancel_check=lambda: time.perf_counter() >= deadline,
                    progress_callback=report_native)
                cpu_seconds += campaign.cpu_seconds
                stem = f'slice-{ordinal:03}'
                dispositions = [row for row in (*progress, *native_progress) if row.get('proposalDisposition')]
                retain(args.output / f'{stem}.json.gz', dict(target=target, scenario=scenario.to_dict(),
                    configurationSnapshot=slice_bundle.configuration_snapshot(), authorityHash=slice_authority.sha256,
                    executionBudget=execution_budget.to_dict(), campaign=campaign.to_dict(),
                    executions=[{'execution': item.to_dict(), 'semanticArtifacts': item.semantic_artifact_documents()
                        if getattr(item, 'artifact_preimages', None) is not None else None}
                        for item in campaign.executions], constructionProgress=progress,
                    nativeProgress=native_progress, proposalDispositions=dispositions))
                count = funnels[lane[0]]
                count.update(screened=len(progress), constructed=sum(row.get('proposalCount', 0) for row in progress),
                    bidQualified=sum(row.get('auctionQualification', {}).get('status') == 'complete'
                        and row['auctionQualification'].get('allPassed') is False for row in progress),
                    proofAttempts=len({row['proposalOrdinal'] for row in native_progress if row.get('nativeAttemptStarted')}),
                    completedExecutions=len(campaign.executions), canonicalAccepted=len(campaign.accepted_results),
                    incompleteAttempts=sum(row['proposalDisposition'] == 'incomplete' for row in dispositions),
                    rejectedBeforeDispatch=sum(row['proposalDisposition'] == 'rejected_before_dispatch' for row in dispositions))
                for row in dispositions:
                    if row.get('dispositionReason'):
                        rejection_reasons[lane[0]][row['dispositionReason']] += 1
                for item in campaign.accepted_results:
                    key = rules_preserving_symmetry_key_v1(_campaign_variant_from_execution_v1(item))
                    if key in seen:
                        count['symmetryDuplicates'] += 1
                        continue
                    packet_started = time.perf_counter()
                    try:
                        packet = accepted_packet(item, rules=rules, scenario=scenario, target=target,
                            maximum_seconds=min(parameters['maximumSeconds'], max(.001, deadline - packet_started)))
                    except ExactNormalizationStopped as error:
                        count['packetIncomplete'] += 1
                        retain(args.output / f'{stem}-packet-incomplete-{count["packetIncomplete"]:04}.json',
                            {'executionHash': item.sha256, 'partialReceipt': error.partial_receipt})
                        continue
                    latency.append(time.perf_counter() - packet_started)
                    seen.add(key)
                    packets.append(packet)
                    retain(args.output / f'accepted-{len(packets):04}.json.gz', packet)
                    count['nativeTaggedDistinct'] += 1
                summary = dict(lane=lane[0], ordinal=ordinal, status=campaign.status,
                    allocatedSeconds=slice_seconds, wallSeconds=time.perf_counter() - slice_started,
                    cpuSeconds=campaign.cpu_seconds, counts=dict(count))
                summaries.append(summary)
                spent[lane[0]] += summary['wallSeconds']
                print(json.dumps(summary), flush=True)
            pair += 1
    finally:
        counts = {lane[0]: sum(TagQueryV1.build(lane[2]).matches(packet['lessonTags']['tags']) for packet in packets) for lane in lanes}
        summary = dict(schemaVersion=1, sourceCommit=revision, sourceTree=tree, frozenAuthorityHash=frozen_hash,
            proofWorkers=args.proof_workers, startPair=start_pair,
            wallSeconds=time.perf_counter() - started, cpuSeconds=cpu_seconds, cpuTimeCoverage='sampled_lower_bound',
            activeSeconds=None, activeTimeCoverage='not_measured; wall time includes any controller freeze',
            allocationShares={lane[0]: .5 for lane in lanes}, slices=summaries,
            funnels={key: dict(value) for key, value in funnels.items()}, rejections={key: dict(value) for key, value in rejection_reasons.items()},
            nativeGateRateByLane={key: value['canonicalAccepted'] / value['proofAttempts'] if value['proofAttempts'] else None
                for key, value in funnels.items()},
            nativeTaggedDistinctByTag=counts, unionDistinct=len(packets),
            intersectionDistinct=sum(all(TagQueryV1.build(lane[2]).matches(packet['lessonTags']['tags']) for lane in lanes) for packet in packets),
            packetLatencySeconds=_percentiles(latency), sliceLatencySeconds=_percentiles([row['wallSeconds'] for row in summaries]),
            nativeWorldCountDistribution=dict(Counter(str(packet['sourceArtifact'].get('proof', {}).get('hiddenWorlds', {}).get('count', 'unknown')) for packet in packets)),
            trickBandCounts=dict(Counter(band for packet in packets for band in packet['lessonTags']['tags']['trickBand'])),
            auditedNewByTag=None, auditStatus='pending',
            auditQueue=[{'file': f'accepted-{index:04}.json.gz', 'packetHash': packet['sha256']}
                for index, packet in enumerate(packets, start=1)],
            acceptanceStatus='inconclusive_until_independent_audits_and_negative_regressions',
            missingEvidence=['baseline paired replay lines', 'inference ablations', '20 soft-reject audits/tag',
                '25 audit8 negative revalidations', '2 audit8 positive revalidations', 'accepted lesson library'],
            rendered=False, published=False)
        retain(args.output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authority', type=Path, required=True)
    parser.add_argument('--rules', type=Path, required=True)
    parser.add_argument('--execution', type=Path, required=True, help='immutable strict qualityExecution JSON; no implicit gate overrides')
    parser.add_argument('--policy', type=Path, required=True, help='immutable canonical playingQuality JSON')
    parser.add_argument('--catalog', type=Path, default=ROOT / 'configs/research/families.yaml')
    parser.add_argument('--families', nargs=2, metavar=('ROUND_FLIP', 'GUARD'),
        help='catalog family IDs for the existing equal-share tag lanes; omitted uses the original families')
    parser.add_argument('--auction-library', type=Path,
        help='retained bid-only records used as seed hints; every opening is revalidated natively')
    parser.add_argument('--tail-first', action='store_true',
        help='private backward-construction comparison; requires explicit plain proposal authority, never reuses its proof')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--start-pair', type=int, default=0,
        help='first pair in this source revision target schedule; use a new output directory')
    parser.add_argument('--source-commit', required=True, help='exact source commit supplied by the existing sync carrier')
    parser.add_argument('--source-tree', help='source tree supplied by the existing sync carrier')
    parser.add_argument('--seconds', type=float, default=2400)
    parser.add_argument('--slice-seconds', type=float, default=60)
    parser.add_argument('--proof-workers', type=int, default=8,
        help='1..10 independent single-worker CP-SAT jobs; reserve producer CPU within the launcher grant')
    print(json.dumps(run(parser.parse_args())), flush=True)


if __name__ == '__main__':
    main()
