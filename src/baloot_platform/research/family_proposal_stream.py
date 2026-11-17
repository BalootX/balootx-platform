"""Frozen structural families feeding native independent qualification."""

import time
from itertools import chain
from .family_catalog import (
    ResearchFamilyCatalogV1, family_catalog_schema_v1 as family_catalog_schema_v1,
    tag_query_schema_v1 as tag_query_schema_v1, family_target_schema_v1 as family_target_schema_v1,
    build_family_target_v1, _build_family_target_cached_v1 as _build_family_target_cached_v1,
    family_target_authoring_v1 as family_target_authoring_v1,
    validate_family_target_scenario_v1 as validate_family_target_scenario_v1,
    validate_family_target_generation_v1,
    build_family_generation_request_v1, matching_generation_families_v1,
)
from .family_construction import (
    construct_family_initial_prefix_v1, construct_history_family_proposal_v1, family_proposal_witness_v1,
)
from .scenario_contracts import NATIVE_FAMILY_OPENING_MODE, canonical_sha256 as canonical_sha256
from .ortools_play_tail import TailPrefixActionV1
from .ortools_tail_first_candidate import (
    family_tail_first_plan_v1 as family_tail_first_plan_v1,
    tail_first_proposal_stream_v1 as tail_first_proposal_stream_v1,
)




def family_proposal_stream_v1(rules, execution, parameters, *, campaign_started,
                              cancel_check, progress_callback, worker_cpu_callback=None,
                              auction_provider=None, source_seeds=None, family_target=None, scenario=None,
                              playing_quality=None):
    target = build_family_target_v1(family_target if family_target is not None else execution.family_target)
    validate_family_target_generation_v1(target, rules=rules, execution=execution, playing_quality=playing_quality)
    if family_target is not None and (execution.family_catalog is None
            or target["catalog"]["sha256"] != execution.family_catalog["sha256"]):
        raise ValueError("family generation target must use the frozen catalog")
    catalog = ResearchFamilyCatalogV1.build(target["catalog"])
    family = next(row for row in catalog.families if row.family_id == target["familyId"])
    yield from _history_family_proposals_v1(rules, execution, parameters,
        family=family, bindings=target["bindings"], dealer=target["dealer"],
        buyer=target["buyer"], trump=target["trump"], family_target_hash=target["sha256"],
        campaign_started=campaign_started, cancel_check=cancel_check,
        progress_callback=progress_callback, auction_provider=auction_provider, source_seeds=source_seeds,
        scenario=scenario, playing_quality=playing_quality)


def _history_family_proposals_v1(rules, execution, parameters, *, family, bindings,
                                dealer, buyer, trump, campaign_started, cancel_check,
                                progress_callback, auction_provider, source_seeds=None,
                                family_target_hash=None, scenario=None, playing_quality=None,
                                _select_query_suit=False, _allocated_decks=None):
    deadline = campaign_started + parameters["maximumCampaignSeconds"]
    native_opening = getattr(execution, 'family_opening_mode', None) == NATIVE_FAMILY_OPENING_MODE
    if native_opening and auction_provider is None:
        raise ValueError('native family opening requires a qualified auction provider')
    emitted, excluded = 0, []
    allocated_decks = [] if _allocated_decks is None else _allocated_decks
    policy_prefix_steps = (min(4 * (bindings['playIndex'] // 4),
                              4 * (execution.exact_play_from_trick - 1))
        if playing_quality is not None and playing_quality.mode == 'budgeted'
        and playing_quality.visibility == 'actor_information_set' and execution.exact_play_from_trick is not None else 0)
    if source_seeds is None:
        source_seeds = range(parameters["seed"], parameters["seed"] + execution.proposal_seed_count)
    elif (not isinstance(source_seeds, (list, tuple))
            or len(source_seeds) > execution.proposal_seed_count
            or any(type(seed) is not int or seed < 0 for seed in source_seeds)
            or len(set(source_seeds)) != len(source_seeds)):
        raise ValueError('source seeds must be unique nonnegative integers within proposalSeedCount')
    for seed in source_seeds:
        construction_remaining = min(parameters['maximumSeconds'],
            execution.proposal_maximum_seconds or parameters['maximumSeconds'])
        excluded_prefixes = []
        allocation = None
        while construction_remaining > 0:
            remaining = deadline - time.perf_counter()
            if remaining <= 0 or cancel_check():
                return
            attempt_started = time.perf_counter()
            if native_opening and allocation is None:
                allocation = construct_family_initial_prefix_v1(family=family, bindings=bindings,
                    rules=rules, dealer=dealer, buyer=buyer, trump=trump, seed=seed,
                    maximum_seconds=min(remaining, construction_remaining),
                    maximum_conflicts=parameters['maximumConflicts'], excluded_deck_orders=tuple(allocated_decks),
                    cancel_check=lambda: cancel_check() or time.perf_counter() >= deadline,
                    **({'scenario': scenario} if scenario is not None else {}))
                elapsed = time.perf_counter() - attempt_started
                construction_remaining -= elapsed
                progress_callback({'currentStage': 'family_allocation', 'screenedSeed': seed,
                    'screenedDealer': dealer, 'proposalOnly': True, 'proposalCount': 0,
                    'constructionStatus': allocation['status'], 'constructionElapsedSeconds': elapsed,
                    **({'constructionReason': allocation['reason']} if allocation.get('reason') else {}),
                    **({'allocationModelHash': allocation['allocation']['modelHash']}
                       if 'allocation' in allocation else {})})
                if allocation['status'] != 'constructed':
                    break
                allocated_decks.append(tuple(allocation['initialPrefix']['deckOrder']))
                remaining = min(deadline - time.perf_counter(), construction_remaining)
                if remaining <= 0 or cancel_check():
                    return
                attempt_started = time.perf_counter()
            solved_buyer = allocation['allocation']['auction']['completion']['buyer'] if native_opening else buyer
            proposal = construct_history_family_proposal_v1(family=family, bindings=bindings,
                rules=rules, dealer=dealer, buyer=solved_buyer, trump=trump,
                maximum_seconds=min(remaining, construction_remaining),
                maximum_conflicts=parameters["maximumConflicts"], seed=seed,
                rollout_policy_id=execution.early_play_policy_id,
                excluded_deck_orders=tuple(excluded),
                cancel_check=lambda: cancel_check() or time.perf_counter() >= deadline,
                **({'excluded_play_prefixes': tuple(excluded_prefixes)} if excluded_prefixes else {}),
                **({'scenario': scenario} if scenario is not None else {}),
                **({'policy_prefix_steps': policy_prefix_steps} if policy_prefix_steps else {}),
                **({'_select_query_suit': True} if _select_query_suit else {}),
                **({'initial_prefix': allocation['initialPrefix']} if native_opening else {}),
                **({'auction_provider': auction_provider} if auction_provider is not None else {}))
            elapsed = time.perf_counter() - attempt_started
            construction_remaining -= elapsed
            stopped = cancel_check() or time.perf_counter() >= deadline
            progress_callback({"currentStage": "family_proposal", "screenedSeed": seed, "screenedDealer": dealer,
                **({"familyTargetHash": family_target_hash} if family_target_hash is not None else {}),
                "proposalOnly": True,
                "proposalCount": int(proposal["status"] == "proposal"),
                "constructionStatus": proposal["status"],
                "constructionElapsedSeconds": elapsed,
                **({"proposalOrdinal": parameters["seed"] + emitted} if proposal["status"] == "proposal" else {}),
                **({"nativeAttemptStarted": False, "proposalDisposition": "rejected_before_dispatch",
                    "dispositionReason": "construction_budget_exhausted" if time.perf_counter() >= deadline else "cancelled"}
                   if stopped and proposal["status"] == "proposal" else {}),
                **({'auctionQualification': proposal['auctionQualification']} if proposal.get('auctionQualification') else {}),
                **({'constructionInfeasibleCore': proposal['constructionInfeasibleCore']}
                   if 'constructionInfeasibleCore' in proposal else {}),
                **({"constructionReason": proposal["reason"]} if proposal.get("reason") else {})})
            # A different random seed cannot repair an exhaustively infeasible model.
            if proposal["status"] == "infeasible":
                return
            if proposal['status'] != 'proposal':
                break
            if stopped:
                return
            if native_opening:
                proposal['familyOpening'] = {**allocation,
                    'auctionQualification': proposal.pop('nativeOpeningAuction')}
            witness = family_proposal_witness_v1(proposal, rules=rules,
                interest_policy=execution.video_interestingness)
            if excluded_prefixes:
                current = tuple(TailPrefixActionV1.build(row['actor'], row['card'], ekka=row.get('ekka', False))
                                for row in proposal['decisionPrefix']['playActions'])
                if current in excluded_prefixes:
                    raise ValueError('family constructor repeated an excluded decision prefix')
            yield parameters["seed"] + emitted, witness
            emitted += 1
            if emitted >= parameters["maximumCandidates"]:
                return
            if auction_provider is None:
                excluded.append(tuple(proposal['deckOrder']))
                break
            excluded_prefixes.append(tuple(TailPrefixActionV1.build(row['actor'], row['card'], ekka=row.get('ekka', False))
                                           for row in proposal['decisionPrefix']['playActions']))


def family_query_proposal_stream_v1(rules, execution, parameters, *, generation, scenario,
                                    campaign_started, cancel_check, progress_callback,
                                    worker_cpu_callback=None, auction_provider=None, playing_quality=None):
    """Distribute the existing bounded producer across requested enabled families."""
    request = build_family_generation_request_v1(generation,
        maximum_auction_seeds=execution.proposal_seed_count)
    catalog = ResearchFamilyCatalogV1.build(execution.family_catalog)
    families = matching_generation_families_v1(request, catalog, scenario,
        play_indices=execution.proposal_play_indices, rules=rules, execution=execution, playing_quality=playing_quality)
    shares = catalog.allocation_shares(request["tagQuery"],
        eligible_family_ids={family.family_id for family in families})
    counts = {family.family_id: 0 for family in families}
    selector = scenario.decision_point["selector"]
    index = selector["trickIndex"] * 4 + selector["position"]
    emitted = 0
    allocated_decks = []
    limit = execution.proposal_seed_count
    seats = ("north", "west", "south", "east")
    openings = chain(request.get("auctionSeeds", ()), (
        {"seed": parameters["seed"] + ordinal, "dealer": seats[(parameters["seed"] + ordinal) % 4]}
        for ordinal in range(limit)))
    seen = set()
    for opening in openings:
        pair = (opening["seed"], opening["dealer"])
        if pair in seen:
            continue
        if len(seen) >= limit:
            return
        seen.add(pair)
        untried = list(families)
        while untried:
            if cancel_check() or time.perf_counter() >= campaign_started + parameters["maximumCampaignSeconds"]:
                return
            attempt = sum(counts.values()) + 1
            family = min(untried, key=lambda row: (
                counts[row.family_id] - attempt * shares[row.family_id], row.family_id))
            untried.remove(family)
            counts[family.family_id] += 1
            seed = opening["seed"]
            choices = family.parameters.choices(("sun",))
            cursor, bindings = seed, {}
            for key in ("suit", "rank", "projectType"):
                bindings[key] = choices[key][cursor % len(choices[key])]
                cursor //= len(choices[key])
            bindings.update(seat=selector["actingSeat"], playIndex=index, contract="sun")
            family_miss = False

            def report(value):
                nonlocal family_miss
                family_miss = value.get("constructionStatus") == "seed_miss" and (
                    execution.family_opening_mode == NATIVE_FAMILY_OPENING_MODE
                    or value.get("constructionReason") in {
                    "negative_project_mismatch", "observer_suit_count_mismatch",
                    "history_suit_count_mismatch", "history_expression_infeasible", "history_prefixes_exhausted",
                    "fixed_deal_projects_infeasible"})
                progress_callback({**value, "familyId": family.family_id,
                    "generationRequestHash": request["sha256"],
                    **({"proposalOrdinal": parameters["seed"] + emitted}
                       if value.get("proposalCount") else {})})

            rows = _history_family_proposals_v1(rules, execution, {**parameters, "seed": seed,
                # Multiple families retain their existing weighted one-proposal turns.
                "maximumCandidates": parameters['maximumCandidates'] - emitted
                    if auction_provider is not None and len(families) == 1 else 1},
                family=family, bindings=bindings, dealer=opening["dealer"], buyer=None, trump=None,
                campaign_started=campaign_started, cancel_check=cancel_check, progress_callback=report,
                auction_provider=auction_provider, source_seeds=[seed], scenario=scenario, playing_quality=playing_quality,
                **({'_allocated_decks': allocated_decks} if execution.family_opening_mode is not None else {}),
                _select_query_suit=True)
            try:
                for _, witness in rows:
                    yield parameters["seed"] + emitted, witness
                    emitted += 1
            finally:
                rows.close()
            if emitted >= parameters["maximumCandidates"]:
                return
            # Native families allocate different deals; only seeded opening misses are shared.
            if not family_miss:
                break
