"""Native family proposals: existential legal witnesses, never strategic admission."""

import math
import time

from ortools.sat.python import cp_model

from baloot_platform.core.cards import Card, FULL_DECK
from baloot_platform.core.dealing import DealLayoutV1
from baloot_platform.core.enums import Contract, Seat, Suit, Team
from baloot_platform.core.models import LegalPlayActionV1
from .ortools_play_tail import PlayTailSolveStopped, TailPrefixActionV1
from .ortools_infeasible_core import sound_infeasible_assumption_indexes
from .ortools_scenario_model import OrToolsScenarioModelV1, reify_scenario_expression
from .scenario_contracts import ExactTracePrefixV1, NATIVE_FAMILY_OPENING_MODE, canonical_sha256
from .scenario_deal_predicates_v2 import DealPredicateModelContextV2
from .scenario_predicate_auction_escalation_v2 import CpSatAuctionPredicateContextV1
from .scenario_adjudication_predicates_v3 import shared_scenario_predicate_catalog_lowering_manifest_v3
from .scenario_project_outcome_lowering_v2 import CpSatProjectPredicateContextV1
from .family_catalog import (
    validate_family_construction_expression_v1, _mandatory_family_predicates_v1, history_family_absences_v1,
    family_construction_expression_v1,
)
from .construction_trace import construction_trace_prefix_v1


def construct_family_initial_prefix_v1(*, family, bindings, rules, dealer, buyer, trump,
        maximum_seconds, maximum_conflicts, seed, excluded_deck_orders=(), cancel_check=None, scenario=None):
    """Native allocation narrowing; fresh auction/play/proof admission remains mandatory."""
    from .ortools_project_fragment import attach_project_lifecycle_fragment_v1

    if (type(maximum_seconds) not in (int, float) or not math.isfinite(maximum_seconds)
            or maximum_seconds <= 0 or type(maximum_conflicts) is not int or maximum_conflicts <= 0
            or type(seed) is not int or seed < 0):
        raise ValueError('native family allocation requires a positive finite budget and nonnegative seed')
    if not rules.data['dealing']['allow_no_cut']:
        raise ValueError('native family allocation requires post-cut constructed deck support')
    contract = Contract.parse(bindings['contract'])
    if (contract == Contract.HOKUM) != (trump is not None) or contract == Contract.ASHKAL:
        raise ValueError('native family allocation supports Sun without trump or Hokum with trump')
    expression = family_construction_expression_v1(family.instantiate(bindings), scenario)
    predicates = tuple(p for p in _mandatory_family_predicates_v1(expression)
                       if p.hardness == 'hard' and p.kind.startswith('project.'))
    declaration_only = bool(predicates) and all(p.kind != 'project.baloot'
        and p.arguments.get('state', 'eligible') in {'eligible', 'declared'} for p in predicates)
    structural = tuple(p for p in _mandatory_family_predicates_v1(expression)
        if p.hardness == 'hard' and (p.kind.startswith('auction.')
            or (p.kind.startswith('deal.') and not p.kind.startswith('deal.remaining_')
                and p.arguments.get('handStage') != 'remaining')))
    deadline = time.monotonic() + maximum_seconds
    def stopped():
        return time.monotonic() >= deadline or bool(cancel_check and cancel_check())
    base = {'qualified': False, 'mode': NATIVE_FAMILY_OPENING_MODE}
    if stopped():
        return {**base, 'status': 'incomplete', 'reason': 'native_allocation_cancelled_or_budget'}
    model = OrToolsScenarioModelV1(rules=rules, dealer=dealer, required_buyer=buyer,
        excluded_deck_orders=excluded_deck_orders, canonicalize_deal_packets=True)
    auction = model.add_native_auction_fragment(allowed_contracts=(contract,), require_optimal_actions=True)
    if trump is not None:
        model.model.add(auction.trump == int(Suit.parse(trump)))
    anchors = [None] * 32
    # Positive project anchors narrow construction, never prove family-wide impossibility.
    for predicate in predicates:
        if predicate.kind != 'project.exists' and not (
                predicate.kind == 'project.count' and predicate.arguments['minimum'] > 0):
            continue
        args = predicate.arguments
        seats = (Seat.parse(args['owner']),) if args.get('owner') else tuple(
            seat for seat in Seat if seat.team == Team[args['team'].upper()])
        for seat in seats:
            patterns = rules.project_domain.candidates(seat, FULL_DECK, contract)
            pattern = next((p for p in patterns
                if p.kind == args.get('kind', bindings['projectType'])
                and ('value' not in args or p.value == args['value'])
                and all(anchors[card.id] in (None, seat) for card in p.cards)), None)
            if pattern is not None:
                for card in pattern.cards:
                    anchors[card.id] = seat
                    model.model.add(model.owner_of_card[card.id] == int(seat))
                break
    try:
        # A remaining card must have belonged to that seat after purchase.
        for predicate in _mandatory_family_predicates_v1(expression):
            if predicate.hardness == 'hard' and predicate.kind == 'deal.remaining_card_owner':
                model.model.add(model.owner_of_card[predicate.arguments['card']]
                                == int(Seat.parse(predicate.arguments['seat'])))
        if structural:
            contexts = {}
            if any(p.kind.startswith('deal.') for p in structural):
                contexts['deal'] = DealPredicateModelContextV2(model)
            if any(p.kind.startswith('auction.') for p in structural):
                contexts['auction'] = CpSatAuctionPredicateContextV1(model.model, auction, Seat.parse(dealer))
            manifest = shared_scenario_predicate_catalog_lowering_manifest_v3()
            for predicate in structural:
                context = contexts[predicate.kind.split('.')[0]]
                model.model.add(manifest.lower_predicate(predicate, context).literal == 1)
        if predicates:
            projects = attach_project_lifecycle_fragment_v1(model.model, owner_of_card=model.owner_of_card,
                fixed_card_owners=anchors, contract=bindings['contract'], trump=trump, dealer=dealer,
                domain=rules.project_domain, include_declaration_actions=not declaration_only,
                declaration_only=declaration_only, cancel_check=stopped,
                second_trick_leader=None if declaration_only else model.model.new_int_var(0, 3, 'unknown_reveal_leader'))
            context = CpSatProjectPredicateContextV1(model.model, 'opening_projects', projects)
            manifest = shared_scenario_predicate_catalog_lowering_manifest_v3()
            for predicate in predicates:
                model.model.add(manifest.lower_predicate(predicate, context).literal == 1)
        if stopped():
            raise InterruptedError('native_allocation_cancelled_or_budget')
        solution = model.solve(seed=seed, maximum_seconds=deadline - time.monotonic(),
            maximum_conflicts=maximum_conflicts, search_branching='automatic', cancel_check=stopped)
    except InterruptedError:
        return {**base, 'status': 'incomplete', 'reason': 'native_allocation_cancelled_or_budget'}
    if solution.solver_status == 'model_invalid':
        raise ValueError('native family allocation model is invalid: ' + model.model.validate())
    if solution.status != 'satisfied':
        return {**base, 'status': 'seed_miss' if solution.status == 'infeasible' else 'incomplete',
            'reason': 'native_allocation_' + solution.status, 'allocation': solution.to_dict()}
    initial = ExactTracePrefixV1.build(dict(schemaVersion=1, dealer=Seat.parse(dealer).name.lower(),
        deckOrder=list(solution.deck_order), roundNumber=1, matchScore=[0, 0], auctionActions=[],
        escalationActions=[], projectActions=[], projectDeclarations=[], projectAssertions=[],
        playActions=[], stateAssertions={}))
    return {**base, 'status': 'constructed', 'initialPrefix': initial.to_dict(),
        'allocation': solution.to_dict(), 'rulesHash': rules.sha256, 'seed': seed,
        'familyHash': canonical_sha256(family.to_dict()), 'expressionHash': expression.sha256,
        'constructionNarrowing': {'visibleHandAuctionPolicy': True, 'unknownRevealLeader': not declaration_only,
            **({'projectScope': 'project_declarations'} if declaration_only else {}),
            'projectCardOwners': {str(card): owner.name.lower()
                                 for card, owner in enumerate(anchors) if owner is not None}}}


def _family_initial_deal_v1(rules, seed, dealer, *, initial_prefix=None):
    from baloot_platform.core.engine import BalootEngine

    engine = BalootEngine(rules)
    if initial_prefix is not None:
        initial = ExactTracePrefixV1.build(initial_prefix.to_dict()
            if isinstance(initial_prefix, ExactTracePrefixV1) else initial_prefix)
        if (initial.dealer != Seat.parse(dealer).name.lower() or initial.round_number != 1
                or initial.auction_actions or initial.escalation_actions or initial.project_actions
                or initial.project_declarations or initial.project_assertions or initial.play_actions
                or initial.adjudication_events or initial.state_assertions):
            raise ValueError('family initial prefix requires the requested dealer, round one and no actions or assertions')
        state = engine.create_round(seed=seed, dealer=Seat.parse(dealer), match_score=initial.match_score)
        engine.initial_deal_from_preordered(state, initial.deck_order)
        return engine, state, initial
    state = engine.create_round(seed=seed, dealer=Seat.parse(dealer))
    engine.shuffle_and_initial_deal(state, cut_position=(seed * 17 % 31) + 1)
    initial = ExactTracePrefixV1.build({
        'schemaVersion': 1, 'deckOrder': list(state.shuffle.deck_order),
        'dealer': state.dealer.name.lower(), 'roundNumber': state.round_number,
        'matchScore': list(state.match_score), 'auctionActions': [],
        'escalationActions': [], 'projectActions': [], 'projectDeclarations': [],
        'projectAssertions': [], 'playActions': [], 'stateAssertions': {},
    })
    return engine, state, initial


def _qualified_family_opening_v1(rules, seed, dealer, auction_provider, remaining, cancel_check,
                                 *, initial_prefix=None):
    from baloot_platform.core.auction_ir import AuctionDomainIRV1
    from baloot_platform.core.bidding_machine import BiddingMachine

    engine, state, initial = _family_initial_deal_v1(rules, seed, dealer,
        **({'initial_prefix': initial_prefix} if initial_prefix is not None else {}))
    receipt = auction_provider(initial, maximum_seconds=remaining(), cancel_check=cancel_check)
    if receipt['status'] != 'complete':
        return engine, state, None, receipt
    auction_domain = AuctionDomainIRV1.from_rules(rules.data)
    if (receipt.get('operation') != 'generate' or receipt.get('partialReceipt') is not None
            or receipt.get('qualificationAuthority') != 'cp_sat'
            or receipt.get('rulesHash') != rules.sha256 or receipt.get('coveredFromBidIndex') != 0
            or receipt.get('sourcePrefixHash') != initial.sha256
            or (not receipt.get('decisions') and not auction_domain.require_preselected)
            or any(row.get('actualActionOptimal') is not True for row in receipt['decisions'])):
        raise ValueError('qualified auction requires complete native generated bid evidence')
    prefix = ExactTracePrefixV1.build(receipt['finalPrefix'])
    if (prefix.deck_order != initial.deck_order or prefix.dealer != initial.dealer
            or prefix.match_score != initial.match_score or prefix.round_number != initial.round_number
            or prefix.play_actions or prefix.project_actions or prefix.escalation_actions
            or prefix.project_declarations or prefix.project_assertions or prefix.adjudication_events
            or prefix.state_assertions):
        raise ValueError('qualified auction changed its deal or contains post-auction actions')
    machine = BiddingMachine(state.dealer, state.purchase_card.suit, auction_domain, state.purchase_card.rank)
    for index, action in enumerate(prefix.auction_actions):
        if index < len(machine.records):
            record = machine.records[index]
            if (record.seat != Seat.parse(action['actor']) or record.action != action['action']
                    or record.trump != (Suit.parse(action['trumpSuit']) if action.get('trumpSuit') else None)):
                raise ValueError('qualified auction disagrees with preselected bidding')
            continue
        if machine.actor != Seat.parse(action['actor']):
            raise ValueError('qualified auction has the wrong actor')
        machine.submit(action['action'], trump=Suit.parse(action['trumpSuit']) if action.get('trumpSuit') else None)
    if machine.result is None:
        raise ValueError('qualified auction did not reach its terminal state')
    if not engine.complete_deal_from_bid_result(state, machine.result):
        return engine, state, None, receipt
    engine.prepare_projects(state)
    return engine, state, prefix.to_dict(), receipt


def _history_query_suit_bindings_v1(family, bindings, scenario, hand, preceding_plays):
    def parameterized_ids(node):
        if node.get('op') == 'all':
            return {identity for child in node['children'] for identity in parameterized_ids(child)}
        return ({node['predicateId']} if node.get('op') is None
            and node.get('kind') == 'deal.remaining_suit_count'
            and node['arguments'].get('suit') == {'$parameter': 'suit'} else set())

    variable_ids = parameterized_ids(family.to_dict()['template']['expression'])
    if not any(p.predicate_id in variable_ids and p.arguments['seat'] == bindings['seat']
               for p in _mandatory_family_predicates_v1(family.instantiate(bindings))
               if p.kind == 'deal.remaining_suit_count'):
        return (dict(bindings),)
    authored = {p.predicate_id: p.sha256 for section in (scenario.constraints, scenario.outcome)
                for p in section.predicates()} if scenario is not None else {}
    suits = family.parameters.suits
    offset = suits.index(bindings['suit'])
    alternatives = []
    for suit in suits[offset:] + suits[:offset]:
        selected = {**bindings, 'suit': suit}
        expression = family.instantiate(selected)
        if any(p.predicate_id in authored and p.sha256 != authored[p.predicate_id]
               for p in expression.predicates()):
            continue  # An authored predicate identity freezes its arguments.
        expression = family_construction_expression_v1(expression, scenario)
        for predicate in _mandatory_family_predicates_v1(expression):
            if predicate.kind != 'deal.remaining_suit_count' or predicate.arguments['seat'] != bindings['seat']:
                continue
            count = sum(card.suit == Suit.parse(predicate.arguments['suit']) for card in hand)
            # Each intervening trick removes one of this seat's cards, never adds one.
            if max(predicate.arguments['minimum'], count - preceding_plays, 0) > min(predicate.arguments['maximum'], count):
                break
        else:
            alternatives.append(selected)
    return tuple(alternatives)


def _fixed_deal_project_status_v1(expression, *, rules, opening, buyer, contract, trump, deadline,
                                 maximum_conflicts, seed, cancel_check):
    predicates = [p for p in _mandatory_family_predicates_v1(expression)
                  if p.hardness == 'hard' and p.kind.startswith('project.')]
    if not predicates:
        return None
    from .ortools_project_fragment import attach_project_lifecycle_fragment_v1
    from .ortools_information_set_proof import _solve_cp_sat_with_cancel_v1

    def stopped():
        return time.monotonic() >= deadline or bool(cancel_check and cancel_check())

    model = cp_model.CpModel()
    owners = [None] * 32
    layout = DealLayoutV1.build(rules.structural_domain, dealer=opening['dealer'], buyer=buyer)
    for seat in Seat:
        for position in layout.positions_for(seat, final=True):
            owners[opening['deckOrder'][position]] = seat
    variables = tuple(model.new_int_var(int(owner), int(owner), f'project_owner_{card}')
                      for card, owner in enumerate(owners))
    try:
        projects = attach_project_lifecycle_fragment_v1(model, owner_of_card=variables,
            fixed_card_owners=owners, contract=contract, dealer=opening['dealer'], trump=trump,
            domain=rules.project_domain, compact_fixed_states=True, tighten_value_bounds=True,
            include_declaration_actions=True, cancel_check=stopped,
            # Overapproximate the unknown reveal leader: only UNSAT rejects this fixed deal.
            second_trick_leader=model.new_int_var(0, 3, 'project_second_leader'))
    except InterruptedError:
        return cp_model.UNKNOWN
    context = CpSatProjectPredicateContextV1(model, 'fixed_deal_projects', projects)
    manifest = shared_scenario_predicate_catalog_lowering_manifest_v3()
    for predicate in predicates:
        model.add(manifest.lower_predicate(predicate, context).literal == 1)
    seconds = deadline - time.monotonic()
    if seconds <= 0 or stopped():
        return cp_model.UNKNOWN
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = seed % (2**31)
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.max_number_of_conflicts = maximum_conflicts
    status = _solve_cp_sat_with_cancel_v1(solver, model, stopped)
    if status == cp_model.MODEL_INVALID:
        raise ValueError('fixed-deal project model is invalid: ' + model.validate())
    return status


def construct_history_family_proposal_v1(*, family, bindings, rules, dealer, buyer,
        trump, maximum_seconds, maximum_conflicts, seed, rollout_policy_id,
        excluded_deck_orders=(), cancel_check=None, auction_provider=None, scenario=None,
        policy_prefix_steps=0, _select_query_suit=False, excluded_play_prefixes=(), initial_prefix=None):
    """Construct a fresh native tail around a seeded, policy-generated prefix.

    Qualified auctions retain the original deal; legacy inputs may reallocate.
    Both routes still require the executor's independent full-prefix quality audit.
    """
    from .construction_pre_screen import (
        _proposal_opening, _proposal_policy_score, _ProposalScreenStopped,
    )
    from .negative_project_prefix import build_negative_project_prefix_v1
    from baloot_platform.core.project_visibility import (
        project_rank_is_public, project_rank_visibility_policy,
    )

    if buyer is None and auction_provider is None:
        raise ValueError("an unconstrained buyer requires a qualified auction provider")
    if initial_prefix is not None and auction_provider is None:
        raise ValueError('a constructed initial prefix requires a qualified auction provider')
    expression = family_construction_expression_v1(family.instantiate(bindings), scenario)
    absence = history_family_absences_v1(expression, contract=bindings['contract'], trump=trump)
    contract = Contract.parse(bindings['contract'])
    required_trump = Suit.parse(trump) if trump is not None else None
    if contract == Contract.HOKUM and auction_provider is None:
        raise ValueError('Hokum history requires a qualified auction provider')
    if rules.project_domain.declaration_mode != "mandatory":
        raise ValueError("history family producer requires mandatory project declarations")
    if type(maximum_seconds) not in (int, float) or not math.isfinite(maximum_seconds) or maximum_seconds <= 0:
        raise ValueError("maximum_seconds must be finite and positive")
    if type(policy_prefix_steps) is not int or not 0 <= policy_prefix_steps <= bindings['playIndex']:
        raise ValueError('policy prefix must stop before the requested decision')
    deadline = time.monotonic() + maximum_seconds
    base = {"qualified": False, "constructionScope": "history_bound_hidden_reallocation"}
    if auction_provider is not None:
        base['constructionScope'] = 'history_bound_qualified_auction'
    def check():
        if time.monotonic() >= deadline or (cancel_check and cancel_check()):
            raise _ProposalScreenStopped("cancelled_or_budget")
    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0 or (cancel_check and cancel_check()):
            raise _ProposalScreenStopped('cancelled_or_budget')
        return seconds
    try:
        check()
        if auction_provider is not None and buyer is not None and any(
                p.hardness == 'hard' and p.kind.startswith('project.')
                for p in _mandatory_family_predicates_v1(expression)):
            # Sun/Hokum fix the purchase recipient; discard impossible deals before proving their auction.
            _, _, initial = _family_initial_deal_v1(rules, seed, dealer,
                **({'initial_prefix': initial_prefix} if initial_prefix is not None else {}))
            project_status = _fixed_deal_project_status_v1(expression, rules=rules,
                opening=initial.to_dict(), buyer=buyer, contract=bindings['contract'], trump=trump,
                deadline=deadline, maximum_conflicts=maximum_conflicts,
                seed=seed, cancel_check=cancel_check)
            if project_status == cp_model.INFEASIBLE:
                return {**base, 'status': 'seed_miss', 'reason': 'fixed_deal_projects_infeasible'}
            if project_status == cp_model.UNKNOWN:
                return {**base, 'status': 'incomplete', 'reason': 'fixed_deal_projects_incomplete'}
            check()
        if auction_provider is None:
            engine, state, opening = _proposal_opening(rules, seed, dealer=dealer)
        else:
            engine, state, opening, auction_receipt = _qualified_family_opening_v1(
                rules, seed, dealer, auction_provider, remaining, cancel_check,
                **({'initial_prefix': initial_prefix} if initial_prefix is not None else {}))
            base['auctionQualification'] = {'status': auction_receipt['status'],
                'receiptHash': auction_receipt['sha256'],
                'elapsedSeconds': auction_receipt['executionReceipt']['elapsedSeconds'],
                'sourcePrefixHash': auction_receipt['sourcePrefixHash'],
                'allPassed': auction_receipt.get('auctionOutcome', {}).get('allPassed'),
                'cacheHit': auction_receipt['executionReceipt'].get('bidCacheHit', False),
                **({key: auction_receipt['auctionOutcome'][key] for key in ('contract', 'buyer')
                    if key in auction_receipt.get('auctionOutcome', {})}
                   if auction_receipt['status'] == 'complete' else {})}
            if auction_receipt['status'] != 'complete':
                return {**base, 'status': 'incomplete', 'reason': 'auction_quality_incomplete'}
        check()
        if (opening is None or state.contract != contract
                or getattr(state, 'trump_suit', None) != required_trump
                or state.dealer != Seat.parse(dealer)
                or (buyer is not None and state.buyer != Seat.parse(buyer))):
            return {**base, "status": "seed_miss", "reason": "opening_mismatch"}
        if buyer is None:
            buyer = state.buyer.name.lower()
        if auction_provider is not None:
            project_status = _fixed_deal_project_status_v1(expression, rules=rules, opening=opening,
                buyer=buyer, contract=bindings['contract'], trump=trump,
                deadline=deadline, maximum_conflicts=maximum_conflicts,
                seed=seed, cancel_check=cancel_check)
            if project_status == cp_model.INFEASIBLE:
                return {**base, 'status': 'seed_miss', 'reason': 'fixed_deal_projects_infeasible'}
            if project_status == cp_model.UNKNOWN:
                return {**base, 'status': 'incomplete', 'reason': 'fixed_deal_projects_incomplete'}
        prefix = []
        # Leave free choices to the solver, but don't rebuild the declared policy prefix.
        prefix_length = (max(4 * (bindings['playIndex'] // 4 - 1), policy_prefix_steps)
                         if auction_provider is not None else bindings['playIndex'])
        for _ in range(prefix_length):
            check()
            legal = engine.legal_play_actions(state)
            selected = legal[0] if len(legal) == 1 else LegalPlayActionV1.from_code(
                _proposal_policy_score(rules, engine, state, rollout_policy_id, deadline, check)[0])
            prefix.append(TailPrefixActionV1(state.current_player, selected.card, selected.ekka))
            engine.play_card(state, selected.card, ekka=selected.ekka)
            # Mandatory declarations settle after the first trick, not at eligibility detection.
            if len(prefix) == 4 and any(project.declared
                    for predicate in absence for project in state.projects
                    if project.owner.team.name.lower() == predicate.arguments["team"]):
                return {**base, "status": "seed_miss", "reason": "negative_project_mismatch"}
        check()
        if auction_provider is not None and len(prefix) // 4 == bindings['playIndex'] // 4:
            actor = state.current_player
            # Turn order is fixed within this trick; never predict its next winner.
            for _ in range(bindings['playIndex'] - len(prefix)):
                actor = actor.next()
            if actor != Seat.parse(bindings['seat']):
                return {**base, 'status': 'seed_miss', 'reason': 'decision_actor_mismatch'}
        binding_options = (bindings,)
        if _select_query_suit and auction_provider is not None:
            binding_options = _history_query_suit_bindings_v1(family, bindings, scenario,
                state.hand(Seat.parse(bindings['seat'])), bindings['playIndex'] // 4 - len(prefix) // 4)
            if not binding_options:
                return {**base, 'status': 'seed_miss', 'reason': 'history_suit_count_mismatch'}
        remaining_owners = {}
        if auction_provider is None:
            if state.current_player != Seat.parse(bindings["seat"]):
                return {**base, "status": "seed_miss", "reason": "decision_actor_mismatch"}
            observer_hand = state.hand(state.current_player)
            # Hidden reallocation preserves the observer's hand, so it cannot repair this miss.
            for predicate in _mandatory_family_predicates_v1(expression):
                if (predicate.kind == "deal.remaining_suit_count"
                        and predicate.arguments["seat"] == bindings["seat"]):
                    count = sum(card.suit == Suit.parse(predicate.arguments["suit"]) for card in observer_hand)
                    if not predicate.arguments["minimum"] <= count <= predicate.arguments["maximum"]:
                        return {**base, "status": "seed_miss", "reason": "observer_suit_count_mismatch"}
            original_owners = [next(int(seat) for seat in Seat
                if state.initial_hands[int(seat)] & (1 << card.id)) for card in FULL_DECK]
            projects = [p.to_dict() for p in state.projects]
            rank_policy = project_rank_visibility_policy(rules)
            public = {seat: [{"kind": p.kind, **({"high_rank": p.high_rank}
                if project_rank_is_public(p.to_dict(), projects, '', rank_policy,
                                          comparison_complete=bool(state.completed_tricks)) else {})}
                for p in state.projects if p.owner == seat and p.declared] for seat in Seat}
            check()
            sample = build_negative_project_prefix_v1(rules=rules, dealer=dealer, buyer=buyer,
                contract="sun", trump=None, purchase_card=state.purchase_card,
                prefix=prefix, observer=state.current_player,
                observer_hand=observer_hand, public_projects=public,
                revealed_projects=[(p.owner, p.cards) for p in state.projects if p.revealed],
                required_remaining_owners={}, auction_actions=opening["auctionActions"],
                excluded_ownerships=(original_owners,), maximum_seconds=remaining(),
                maximum_conflicts=maximum_conflicts, random_seed=seed,
                cancel_check=lambda: bool(cancel_check and cancel_check()) or time.monotonic() >= deadline)
            played = {play.card.id for play in prefix}
            remaining_owners = {card: sample['ownership'][card.id] for card in FULL_DECK
                                if card.id not in played}
        for selected in binding_options:
            check()
            result = construct_family_proposal_v1(family=family, bindings=selected, rules=rules,
                dealer=dealer, buyer=buyer, trump=trump, prefix=prefix,
                required_remaining_owners=remaining_owners,
                auction_actions=opening["auctionActions"], purchase_card=state.purchase_card,
                maximum_seconds=remaining(), maximum_conflicts=maximum_conflicts,
                seed=seed, excluded_deck_orders=excluded_deck_orders, cancel_check=cancel_check,
                excluded_play_prefixes=excluded_play_prefixes,
                **({'scenario': scenario} if scenario is not None else {}),
                **({'policy_prefix_steps': policy_prefix_steps, 'policy_id': rollout_policy_id}
                   if auction_provider is not None and policy_prefix_steps else {}),
                **({'match_score': tuple(state.match_score)} if initial_prefix is not None else {}),
                **({'required_deck_order': opening['deckOrder']} if auction_provider is not None else {}))
            if result['status'] != 'infeasible':
                break
        # Infeasibility applies to this sampled history, not the whole family.
        if result["status"] == "infeasible":
            result["status"] = "seed_miss"
            result["reason"] = "history_prefixes_exhausted" if excluded_play_prefixes else "history_expression_infeasible"
        result["historySourceSeed"] = seed
        result["historyPolicyId"] = rollout_policy_id
        result["reallocatedHistoryPolicyAudited"] = False
        if auction_provider is not None:
            result['auctionQualification'] = base['auctionQualification']
            if initial_prefix is not None:
                # Auction receipt hashes exclude machine timings throughout the native backend.
                result['nativeOpeningAuction'] = {key: value for key, value in auction_receipt.items()
                                                  if key != 'executionReceipt'}
        return result
    except (PlayTailSolveStopped, _ProposalScreenStopped) as stopped:
        status = getattr(stopped, "solver_status", "unknown")
        return {**base, "status": "seed_miss" if status == "infeasible" else "incomplete",
                "solverStatus": status, "reason": str(stopped)}


def family_proposal_witness_v1(proposal, *, rules, interest_policy):
    """Hand a native witness to the existing proposal-stream qualification path.

    Construction evidence is provenance, never proof or interest acceptance.
    The executor independently normalizes this prefix and proves the decision.
    """
    if (proposal.get("status") != "proposal" or proposal.get("qualified") is not False
            or proposal.get("rulesHash") != rules.sha256):
        raise ValueError("family witness requires an unqualified proposal under the selected rules")
    complete = construction_trace_prefix_v1(proposal["trace"])
    index = proposal["bindings"]["playIndex"]
    if type(index) is not int or not 16 <= index <= 27:
        raise ValueError("family witness decision index is outside the catalog range")
    full = complete.to_dict()
    decision = {key: value for key, value in full.items() if key != "sha256"}
    decision["playActions"] = full["playActions"][:index]
    prefix = ExactTracePrefixV1.build(decision)
    if (complete.to_dict() != proposal["completePrefix"]
            or prefix.to_dict() != proposal["decisionPrefix"]):
        raise ValueError("family witness prefix disagrees with its construction trace")
    focal = full["playActions"][index]
    if focal["actor"] != proposal["bindings"]["seat"]:
        raise ValueError("family witness actor disagrees with the configured decision")
    seed = proposal["seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("family witness seed must be a nonnegative integer")
    source = {key: proposal[key] for key in (
        'familyHash', 'expressionHash', 'rulesHash', 'bindings', 'seed',
        'constructionModelHash', 'trace', 'predicateEvidence')}
    if 'familyOpening' in proposal:
        source['familyOpening'] = proposal['familyOpening']
    row = {"prefix": prefix.to_dict(), "prefixHash": prefix.sha256,
        "rulesHash": rules.sha256, "policyHash": canonical_sha256(interest_policy),
        "playIndex": index, "actor": focal["actor"],
        "action": LegalPlayActionV1(Card.parse(focal["card"]), focal.get("ekka") is True).code,
        "sourceHash": canonical_sha256(source), "sourceSeed": seed}
    if 'familyOpening' in proposal:
        opening = proposal['familyOpening']
        row['familyOpening'] = {**opening, 'sha256': canonical_sha256(opening)}
    return {**row, "sha256": canonical_sha256(row)}


def construct_family_proposal_v1(
    *, family, bindings, rules, dealer, buyer, trump, maximum_seconds,
    maximum_conflicts, seed, match_score=(0, 0), prefix=(),
    required_remaining_owners=None, excluded_deck_orders=(), cancel_check=None,
    auction_actions=(), purchase_card=None, required_deck_order=None, scenario=None,
    policy_prefix_steps=0, policy_id="observer_safe_trick_control_v1", excluded_play_prefixes=(),
):
    """Solve the whole configured expression in one native full-round model.

    This developer-level constructor also accepts disabled authoring drafts;
    scheduling/publication remains gated separately. Prefixes and remaining-card
    anchors narrow construction without supplying hidden cards to a player.
    Unsupported strategic/state predicates fail before any solve, including when
    nested under ANY/NOT. A feasible witness still needs native strategic proof
    and the canonical interest gate before admission.
    """
    if type(maximum_seconds) not in (int, float) or not math.isfinite(maximum_seconds) or maximum_seconds <= 0:
        raise ValueError("maximum_seconds must be finite and positive")
    if type(maximum_conflicts) is not int or maximum_conflicts <= 0 or type(seed) is not int:
        raise ValueError("maximum_conflicts must be positive and seed must be an integer")
    expression = family_construction_expression_v1(family.instantiate(bindings), scenario)
    manifest = shared_scenario_predicate_catalog_lowering_manifest_v3()
    validate_family_construction_expression_v1(expression)
    predicates = tuple({predicate.predicate_id: predicate for predicate in expression.predicates()}.values())
    contract, dealer, buyer = Contract.parse(bindings["contract"]), Seat.parse(dealer), Seat.parse(buyer)
    trump = Suit.parse(trump) if trump is not None else None
    if (contract == Contract.HOKUM) != (trump is not None):
        raise ValueError("Hokum requires trump; Sun forbids trump")
    index, actor = bindings["playIndex"], Seat.parse(bindings["seat"])
    if type(policy_prefix_steps) is not int or not 0 <= policy_prefix_steps <= index:
        raise ValueError('policy prefix must stop before the requested decision')
    prefix = tuple(prefix)
    if len(prefix) > index:
        raise ValueError("fixed prefix extends beyond the requested decision")
    excluded_play_prefixes = tuple(tuple(row) for row in excluded_play_prefixes)
    for row in excluded_play_prefixes:
        if len(row) != index or any(not isinstance(play, TailPrefixActionV1) for play in row):
            raise ValueError('excluded play prefix must end exactly at the configured decision')
        if len({play.card.id for play in row}) != index:
            raise ValueError('excluded play prefix contains duplicate cards')
        for play in row:
            TailPrefixActionV1.build(play.actor, play.card, ekka=play.ekka)
    played_cards = {play.card.id for play in prefix}
    if len(played_cards) != len(prefix):
        raise ValueError("fixed prefix contains duplicate cards")
    anchors = {Card.parse(card).id: Seat.parse(owner)
               for card, owner in (required_remaining_owners or {}).items()}
    if played_cards.intersection(anchors):
        raise ValueError("remaining motif anchor was already played")
    fixed_owners = [None] * 32
    for play in prefix:
        fixed_owners[play.card.id] = play.actor
    for card_id, owner in anchors.items():
        fixed_owners[card_id] = owner
    started = time.monotonic()
    deadline = started + maximum_seconds

    def stopped():
        return time.monotonic() >= deadline or (cancel_check is not None and cancel_check())

    provenance = {"schemaVersion": 1, "familyHash": canonical_sha256(family.to_dict()),
        "expressionHash": expression.sha256, "rulesHash": rules.sha256,
        "bindings": dict(bindings), "seed": seed, "qualified": False}
    if stopped():
        return {**provenance, "status": "incomplete", "reason": "cancelled_or_budget"}
    scenario = OrToolsScenarioModelV1(rules=rules, dealer=dealer, required_buyer=buyer,
        excluded_deck_orders=excluded_deck_orders, canonicalize_deal_packets=required_deck_order is None)
    round_owners = tuple(fixed_owners) if prefix or anchors else None
    if required_deck_order is not None:
        if (not isinstance(required_deck_order, (tuple, list)) or len(required_deck_order) != 32
                or any(type(card) is not int for card in required_deck_order)
                or set(required_deck_order) != set(range(32))):
            raise ValueError('required deck must contain all 32 unique integer cards')
        for variable, card in zip(scenario.card_at_position, required_deck_order, strict=True):
            scenario.model.add(variable == card)
        if contract in {Contract.SUN, Contract.HOKUM}:
            # Only Sun/Hokum guarantee that the buyer takes the purchase card.
            layout = DealLayoutV1.build(rules.structural_domain, dealer=dealer, buyer=buyer)
            round_owners = [None] * 32
            for seat in Seat:
                for position in layout.positions_for(seat, final=True):
                    round_owners[required_deck_order[position]] = seat
            round_owners = tuple(round_owners)
    auction = scenario.add_native_auction_fragment(allowed_contracts=(contract,),
        required_actions=auction_actions, required_actions_complete=bool(auction_actions),
        required_purchase_card=purchase_card)
    if trump is not None:
        scenario.model.add(auction.trump == int(trump))
    for card_id, owner in enumerate(fixed_owners):
        if owner is not None:
            scenario.model.add(scenario.owner_of_card[card_id] == int(owner))
    solver, assumptions = None, {}
    try:
        branch = scenario.add_native_round_continuation(contract=contract, trump=trump,
            first_leader=rules.play_domain.resolve_first_leader(dealer=dealer, buyer=buyer,
                structural_domain=rules.structural_domain), locked=False,
            match_score=match_score, prefix=prefix, cancel_check=stopped,
            decision_prefix_length=index,
            **({'policy_prefix_steps': policy_prefix_steps, 'policy_id': policy_id} if policy_prefix_steps else {}),
            fixed_card_owners=round_owners)
        model, slots = scenario.model, branch.play.play_state_context.play_slots
        if excluded_play_prefixes:
            # Exclude the decision history, never the fixed deal or its future suffix.
            model.add_forbidden_assignments(
                [variable for slot in slots[:index] for variable in (slot.actor, slot.card, slot.ekka)],
                [[value for play in row for value in (int(play.actor), play.card.id, int(play.ekka))]
                 for row in excluded_play_prefixes])
        actor_assumption = model.new_bool_var('assume_family_decision_actor')
        model.add(slots[index].actor == int(actor)).only_enforce_if(actor_assumption)
        model.add_assumption(actor_assumption)
        assumptions[actor_assumption.index] = 'decision_actor'
        remaining = []
        for card_id, owner in enumerate(scenario.owner_of_card):
            if card_id in played_cards:
                remaining.append(model.new_constant(4))
                continue
            if len(prefix) == index:
                remaining.append(owner)
                continue
            occurrences = []
            for step in range(len(prefix), index):
                equal = model.new_bool_var(f"family_played_{card_id}_{step}")
                model.add(slots[step].card == card_id).only_enforce_if(equal)
                model.add(slots[step].card != card_id).only_enforce_if(equal.Not())
                occurrences.append(equal)
            played = model.new_bool_var(f"family_played_{card_id}")
            model.add_max_equality(played, occurrences)
            remaining_owner = model.new_int_var(0, 4, f"family_remaining_owner_{card_id}")
            model.add(remaining_owner == 4).only_enforce_if(played)
            model.add(remaining_owner == owner).only_enforce_if(played.Not())
            remaining.append(remaining_owner)
        for card_id, owner in anchors.items():
            model.add(remaining[card_id] == int(owner))
        contexts = {
            "deal": DealPredicateModelContextV2(scenario, tuple(remaining)),
            "auction": CpSatAuctionPredicateContextV1(model, auction, dealer),
            "project": CpSatProjectPredicateContextV1(model, "family_projects", branch.projects),
            "outcome": branch.scoring.outcome_context,
            "play": branch.play.predicate_context,
            "state": branch.play.play_state_context,
        }
        lowered = {p.predicate_id: manifest.lower_predicate(p, contexts[p.kind.split(".")[0]])
                   for p in predicates}
        expression_assumption = reify_scenario_expression(model, expression,
            {key: value.literal for key, value in lowered.items()})
        model.add_assumption(expression_assumption)
        assumptions[expression_assumption.index] = 'scenario_expression'
        if stopped():
            raise PlayTailSolveStopped("unknown")
        solver = cp_model.CpSolver()
        play = branch.play.solve(maximum_time_seconds=deadline - time.monotonic(),
            maximum_conflicts=maximum_conflicts, random_seed=seed, solver=solver, cancel_check=stopped)
    except PlayTailSolveStopped as exc:
        # A sufficient core describes this fixed history, never the whole family.
        core = ({'constructionInfeasibleCore': sorted({assumptions[index]
                    for index in sound_infeasible_assumption_indexes(solver, assumptions)})}
                if exc.solver_status == 'infeasible' and solver is not None else {})
        return {**provenance, "status": "infeasible" if exc.solver_status == "infeasible" else "incomplete",
                "solverStatus": exc.solver_status, "elapsedSeconds": time.monotonic() - started, **core}
    deck = [solver.value(card) for card in scenario.card_at_position]
    auction_evidence = auction.extract(solver).to_dict()
    escalation = branch.escalation.extract_stage_result(solver,
        model_hash=branch.play.model_hash, solver_status=play.solver_status,
        branches=int(solver.num_branches), conflicts=int(solver.num_conflicts)).to_dict()
    trace = {"deckOrder": deck, "dealer": dealer.name.lower(), "roundNumber": 1,
        "matchScore": list(match_score), "auction": auction_evidence,
        "escalation": escalation, "projects": branch.projects.extract(solver).to_dict(),
        "play": play.to_dict()}
    complete_prefix = construction_trace_prefix_v1(trace)
    decision_document = complete_prefix.to_dict()
    decision_document.pop("sha256", None)
    decision_document["playActions"] = decision_document["playActions"][:index]
    decision_prefix = ExactTracePrefixV1.build(decision_document)
    return {**provenance, "status": "proposal", "constructionScope": "configured_native_round",
        "deckOrder": deck, "auction": auction_evidence, "play": play.at_prefix(index),
        "trace": trace,
        "completePrefix": complete_prefix.to_dict(), "decisionPrefix": decision_prefix.to_dict(),
        "predicateEvidence": [value.extract(solver) for value in lowered.values()],
        "constructionModelHash": branch.play.model_hash,
        "elapsedSeconds": time.monotonic() - started}
