"""Native hidden-card construction and public-project ownership probes."""

import copy
import math
import time

from ortools.sat.python import cp_model

from baloot_platform.core.cards import Card, FULL_DECK
from baloot_platform.core.enums import Contract, Seat
from .ortools_play_tail import OrToolsPlayTailStageV1, PlayTailSolveStopped
from .ortools_project_fragment import attach_project_lifecycle_fragment_v1
from .ortools_scenario_model import OrToolsScenarioModelV1
from .scenario_contracts import canonical_sha256


def _bind_revealed_projects(model, projects, revealed, domain):
    available = {(p.owner, frozenset(c.id for c in p.cards)) for p in projects}
    if not revealed <= available:
        raise ValueError("revealed project is not a governed project pattern")
    for project in projects:
        visible = project.comparison_winner if domain.reveal_scope == "comparison_winner" else project.awarded
        model.add(visible == int((project.owner, frozenset(c.id for c in project.cards)) in revealed))


def build_negative_project_prefix_v1(
    *, rules, dealer, buyer, contract, trump, purchase_card, prefix,
    observer, observer_hand, public_projects, revealed_projects,
    required_remaining_owners, auction_actions=(), excluded_ownerships=(),
    maximum_seconds=30, random_seed=0, maximum_conflicts=250_000, cancel_check=None,
    _observer_inference_probe=False,
):
    """Reallocate hidden cards around a caller-authored remaining-card motif.

    ``public_projects`` covers all seats with their publicly declared project
    descriptors (kind, optionally high_rank if actually public); an empty list
    means no declaration. ``revealed_projects`` is the complete publicly revealed
    list of (seat, cards), including scoring teammates under scoring-team rules.
    Never supply a losing project's private cards.
    ``required_remaining_owners`` binds only motif anchors (e.g. stopper, loser,
    return), not a complete hidden fixture. Exclusions are 32-card owner vectors,
    so a different packet ordering does not masquerade as a new allocation.

    shortcut: preserves public play history; new-history and strategic-policy
    reachability must be separately established before accepting a lesson.
    """
    if type(maximum_seconds) not in (int, float) or not math.isfinite(maximum_seconds) or maximum_seconds <= 0:
        raise ValueError("maximum_seconds must be finite and positive")
    if type(maximum_conflicts) is not int or maximum_conflicts < 1:
        raise ValueError("maximum_conflicts must be positive")
    if type(_observer_inference_probe) is not bool:
        raise TypeError("observer inference probe must be boolean")
    if _observer_inference_probe and not auction_actions:
        raise ValueError("observer inference requires a complete auction")
    deadline = time.monotonic() + maximum_seconds

    def stopped():
        return time.monotonic() >= deadline or bool(cancel_check and cancel_check())

    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0 or bool(cancel_check and cancel_check()):
            raise PlayTailSolveStopped('unknown')
        return seconds

    remaining()
    if rules.project_domain.declaration_mode != "mandatory":
        raise ValueError("negative-project inference requires mandatory declarations")
    observer = Seat.parse(observer)
    contract = Contract.parse(contract)
    prefix = tuple(prefix)
    played = {play.card.id for play in prefix}
    hand = tuple(Card.parse(card) for card in observer_hand)
    if len(played) != len(prefix) or len({c.id for c in hand}) != len(hand):
        raise ValueError("duplicate public or observer cards")
    if played.intersection(c.id for c in hand):
        raise ValueError("observer remaining hand contains a played card")
    expected = {Seat.parse(seat): tuple(items) for seat, items in public_projects.items()}
    if set(expected) != set(Seat):
        raise ValueError("public project declarations must cover all seats")
    for items in expected.values():
        for item in items:
            if set(item) - {"kind", "high_rank"} or "kind" not in item:
                raise ValueError("project descriptors accept kind and public high_rank only")
    anchors = {Card.parse(card): Seat.parse(seat) for card, seat in required_remaining_owners.items()}
    if any(card.id in played for card in anchors):
        raise ValueError("remaining motif anchor was already played")
    if len(anchors) >= 32 - len(played) - len(hand):
        raise ValueError("motif must leave hidden ownership free to solve")
    if _observer_inference_probe and (excluded_ownerships or len(anchors) != 1
            or any(card in hand or owner == observer for card, owner in anchors.items())):
        raise ValueError("observer inference permits one unseen-card ownership assumption and no exclusions")
    fixed_owners = {play.card: play.actor for play in prefix}
    fixed_owners.update({card: observer for card in hand})
    for card, seat in anchors.items():
        if card in fixed_owners and fixed_owners[card] != seat:
            raise ValueError("motif anchor disagrees with observer hand")
        fixed_owners[card] = seat
    remaining()
    scenario = OrToolsScenarioModelV1(
        rules=rules, dealer=dealer, required_buyer=buyer,
        canonicalize_deal_packets=not _observer_inference_probe,
    )
    auction_arguments = dict(
        required_actions=auction_actions, required_purchase_card=purchase_card,
        required_actions_complete=_observer_inference_probe,
        allowed_contracts=(contract,),
    )
    if _observer_inference_probe:
        # Inference evidence must constrain auction legality natively too.
        auction = scenario.add_native_auction_fragment(**auction_arguments)
    else:
        auction = scenario.add_auction_fragment(**auction_arguments,
            canonical_completion_only=not auction_actions)
    if trump is not None:
        from baloot_platform.core.enums import Suit
        scenario.model.add(auction.trump == int(Suit.parse(trump)))
    for card in FULL_DECK:
        if card.id not in played:
            if card in hand:
                scenario.model.add(scenario.owner_of_card[card.id] == int(observer))
            else:
                scenario.model.add(scenario.owner_of_card[card.id] != int(observer))
    for card, seat in fixed_owners.items():
        scenario.model.add(scenario.owner_of_card[card.id] == int(seat))
    for owners in excluded_ownerships:
        if len(owners) != 32:
            raise ValueError("ownership exclusion requires 32 seats")
        scenario.model.add_forbidden_assignments(
            scenario.owner_of_card, [tuple(int(Seat.parse(seat)) for seat in owners)],
        )
    second_trick_leader = (
        scenario.model.new_int_var(0, 3, "negative_project_second_trick_leader")
        if rules.project_domain.tie_break_order == "second_trick_reveal" else None
    )
    projects = attach_project_lifecycle_fragment_v1(
        scenario.model, owner_of_card=scenario.owner_of_card, contract=contract,
        dealer=dealer, trump=trump, domain=rules.project_domain,
        second_trick_leader=second_trick_leader,
        fixed_card_owners=tuple(fixed_owners.get(card) for card in FULL_DECK),
        compact_fixed_states=True, tighten_value_bounds=True,
    )
    for seat, descriptors in expected.items():
        patterns = [p for p in projects.projects if p.owner == seat]
        scenario.model.add(sum(p.declared for p in patterns) == len(descriptors))
        if not descriptors and not _observer_inference_probe:
            # Construction restriction, not a claim that silence proves this
            # when a declaration could have been suppressed by an earlier one.
            for project in patterns:
                scenario.model.add(project.eligible == 0)
        # Public descriptors may repeat; count their multiplicity, not identities.
        keys = [(d["kind"], d.get("high_rank")) for d in descriptors]
        for kind, high_rank in set(keys):
            scenario.model.add(sum(p.declared for p in patterns if p.kind == kind
                and (high_rank is None or p.high_rank == high_rank)) == keys.count((kind, high_rank)))
    revealed = {(Seat.parse(seat), frozenset(Card.parse(c).id for c in cards))
                for seat, cards in revealed_projects}
    _bind_revealed_projects(scenario.model, projects.projects, revealed, rules.project_domain)
    remaining()
    solver = cp_model.CpSolver()
    tie_binding = {}
    if second_trick_leader is not None:
        def bind_reveal_order(build):
            scenario.model.add(second_trick_leader == build._extract.winner_seats[0])

        tie_binding = {
            "construction_plan_hash": canonical_sha256({
                "kind": "negative_project_second_trick_reveal_v1", "rulesHash": rules.sha256}),
            "model_configurator": bind_reveal_order,
        }
    def model_evidence(status):
        return {"solverStatus": status, "constructionModelHash": canonical_sha256(str(scenario.model.proto)),
                "modelSize": {"variables": len(scenario.model.proto.variables),
                              "constraints": len(scenario.model.proto.constraints)}}

    try:
        play = OrToolsPlayTailStageV1(native_conversion=True).solve(
            cards=FULL_DECK, ownership=fixed_owners, contract=contract, trump=trump,
            first_leader=rules.play_domain.resolve_first_leader(
                dealer=Seat.parse(dealer), buyer=Seat.parse(buyer),
                structural_domain=rules.structural_domain),
            play_domain=rules.play_domain, structural_domain=rules.structural_domain,
            prefix=prefix, decision_prefix_length=len(prefix), decision_actor=observer,
            shared_model=scenario.model, shared_owner_of_card=scenario.owner_of_card,
            solver=solver, maximum_time_seconds=remaining(), random_seed=random_seed,
            maximum_conflicts=maximum_conflicts, cancel_check=stopped,
            **tie_binding,
        )
    except PlayTailSolveStopped as stop:
        if not _observer_inference_probe:
            raise
        return {"status": "infeasible" if stop.solver_status == "infeasible" else "incomplete",
                "qualified": False, "rulesHash": rules.sha256,
                **model_evidence(stop.solver_status)}
    return {
        "status": "proposal", "constructionScope": "history_bound_hidden_reallocation",
        "qualified": False, "rulesHash": rules.sha256,
        "deckOrder": [solver.value(card) for card in scenario.card_at_position],
        "ownership": [solver.value(owner) for owner in scenario.owner_of_card],
        "play": play,
        **({**model_evidence(play.solver_status),
            "declaredProjects": [{"owner": project.owner.name.lower(), "projectId": project.project_id}
                for project in projects.projects if solver.boolean_value(project.declared)]}
           if _observer_inference_probe else {}),
    }


def probe_public_project_rank_ownership_v1(
    *, rules, source_prefix, normalized, project_owner, card, expected_owner,
    maximum_seconds=30, maximum_conflicts=250_000, cancel_check=None,
):
    """Test one public rank's ownership consequence, not strategic optimality.

    shortcut: plain Sun after all project reveals; extend the native observation
    model before supporting escalation, Ashkal roles or pre-reveal inference.
    """
    from baloot_platform.core.exact_state_view import build_canonical_observer_information_v1
    from .ortools_exact_state import (
        OrToolsExactStateNormalizerV1, OrToolsNormalizedExactStateV1,
        ExactNormalizationStopped, _effective_prefix_hash,
    )
    from .ortools_play_tail import TailPrefixActionV1
    from .scenario_contracts import ExactTracePrefixV1

    if not isinstance(source_prefix, ExactTracePrefixV1) or not isinstance(normalized, OrToolsNormalizedExactStateV1):
        raise TypeError("project inference requires an exact prefix and native normalization")
    normalized.require_integrity()
    if (normalized.prefix_hash != _effective_prefix_hash(source_prefix)
            or normalized.rules_hash != rules.sha256
            or normalized.normalization_receipt.payload.get('backendId') != 'ortools_cp_sat_v3'):
        raise ValueError("project inference source authority mismatch")
    if type(maximum_seconds) not in (int, float) or not math.isfinite(maximum_seconds) or maximum_seconds <= 0:
        raise ValueError("maximum_seconds must be finite and positive")
    if type(maximum_conflicts) is not int or maximum_conflicts < 1:
        raise ValueError("maximum_conflicts must be positive")
    deadline = time.monotonic() + maximum_seconds

    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0 or (cancel_check and cancel_check()):
            raise PlayTailSolveStopped('unknown')
        return seconds

    state = normalized.state
    observer = Seat.parse(normalized.next_actor)
    project_owner, expected_owner, card = Seat.parse(project_owner), Seat.parse(expected_owner), Card.parse(card)
    if (rules.project_domain.declaration_mode != 'mandatory' or state['contract'] != 'sun'
            or normalized.phase != 'playing' or len(source_prefix.play_actions) < 8
            or source_prefix.escalation_actions or not source_prefix.auction_actions
            or source_prefix.adjudication_events or len(source_prefix.project_actions) != 4
            or state['escalation'] != 'normal'
            or state['winningBidder'] != state['buyer']
            or state['buyer'] != state['purchaseRecipient']):
        raise ValueError("project inference supports mandatory post-reveal plain Sun only")
    information = build_canonical_observer_information_v1(complete_state=state, acting_seat=observer.name.lower())
    public = information['publicState']
    clues = [p for p in public['projects'] if p['owner'] == project_owner.name.lower()
             and p['highRank'] is not None and not p['revealed'] and p['declarationOrder'] < 999]
    if project_owner == observer or len(clues) != 1:
        raise ValueError("project inference requires one publicly announced, unrevealed rank")
    clue = clues[0]
    prefix = tuple(TailPrefixActionV1.build(row['actor'], row['card'], ekka=row.get('ekka', False))
                   for row in source_prefix.play_actions)
    if expected_owner == observer or any(p.card == card for p in prefix) or any(
            Card.parse(c) == card for c in public['hands'][observer.name.lower()]):
        raise ValueError("project inference requires an unseen opposing or partner card")
    descriptors = {seat: [] for seat in Seat}
    for project in public['projects']:
        if project['declarationOrder'] < 999:
            descriptors[Seat.parse(project['owner'])].append({'kind': project['kind'],
                **({'high_rank': project['highRank']} if project['highRank'] is not None else {})})

    def without_clue(value):
        if isinstance(value, dict):
            value = dict(value)
            if (value.get('owner') == clue['owner'] and value.get('kind') == clue['kind']
                    and value.get('declarationOrder') == clue['declarationOrder']):
                value['highRank'] = None
            return {key: without_clue(item) for key, item in value.items()}
        if isinstance(value, list):
            return [without_clue(item) for item in value]
        return value

    common = dict(rules=rules, dealer=source_prefix.dealer, buyer=public['buyer'], contract='sun', trump=None,
        purchase_card=public['purchaseCard'], prefix=prefix, observer=observer,
        observer_hand=public['hands'][observer.name.lower()],
        revealed_projects=[(p['owner'], p['cards']) for p in public['projects'] if p['revealed']],
        auction_actions=source_prefix.auction_actions, maximum_conflicts=maximum_conflicts,
        cancel_check=lambda: time.monotonic() >= deadline or bool(cancel_check and cancel_check()),
        _observer_inference_probe=True)
    probes = []
    result = {'schemaVersion': 1, 'kind': 'public_project_rank_ownership_v1', 'qualified': False,
        'rulesHash': rules.sha256, 'sourcePrefixHash': source_prefix.sha256,
        'normalizationPrefixHash': normalized.prefix_hash,
        'exactStateId': normalized.exact_state_id, 'observerInformationHash': information['observerInformationHash'],
        'clue': {key: clue[key] for key in ('owner', 'kind', 'highRank', 'declarationOrder')},
        'ownership': {'card': card.code, 'owner': expected_owner.name.lower()}, 'probes': probes}

    def probe(owner, rank_included):
        projects = copy.deepcopy(descriptors)
        if not rank_included:
            projects[project_owner][0].pop('high_rank')
        solved = build_negative_project_prefix_v1(**common, public_projects=projects,
            required_remaining_owners={card: owner}, maximum_seconds=remaining())
        probes.append({'rankIncluded': rank_included, 'assumedOwner': owner.name.lower(),
            **{key: solved[key] for key in ('status', 'solverStatus', 'constructionModelHash', 'modelSize')}})
        return solved

    try:
        # Contrary UNSAT is meaningful only when the same full observation is satisfiable.
        original = probe(expected_owner, True)
        if original['status'] != 'proposal':
            result['status'] = 'incomplete' if original['status'] == 'incomplete' else 'source_model_disagreement'
        else:
            others = [seat for seat in Seat if seat not in (observer, expected_owner)]
            for owner in others:
                contrary = probe(owner, True)
                if contrary['status'] != 'infeasible':
                    result['status'] = 'not_entailed' if contrary['status'] == 'proposal' else 'incomplete'
                    break
            else:
                result['status'] = 'independent_of_rank'
                for owner in others:
                    contrary = probe(owner, False)
                    if contrary['status'] == 'incomplete':
                        result['status'] = 'incomplete'
                        break
                    if contrary['status'] != 'proposal':
                        continue
                    payload = source_prefix.to_dict()
                    payload.pop('sha256', None)
                    payload['deckOrder'] = contrary['deckOrder']
                    declared = {p['owner']: p['projectId'] for p in contrary['declaredProjects']}
                    payload['projectActions'] = [({'actor': row['actor'], 'action': 'declare',
                        'projectId': declared[row['actor']]} if row['actor'] in declared else
                        {'actor': row['actor'], 'action': 'pass'}) for row in source_prefix.project_actions]
                    payload['stateAssertions'], payload['projectAssertions'], payload['projectDeclarations'] = {}, [], []
                    witness = ExactTracePrefixV1.build(payload)
                    derived = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(
                        witness, maximum_seconds=remaining(), cancel_check=common['cancel_check'])
                    witness_public = build_canonical_observer_information_v1(
                        complete_state=derived.state, acting_seat=observer.name.lower())['publicState']
                    if without_clue(witness_public) != without_clue(public):
                        result['status'] = 'witness_observation_mismatch'
                        break
                    result.update(status='clue_dependent', contraryWitness=witness.to_dict(),
                        contraryExactStateId=derived.exact_state_id)
                    break
        remaining()
    except (PlayTailSolveStopped, ExactNormalizationStopped):
        result['status'] = 'incomplete'
    return {**result, 'sha256': canonical_sha256(result)}
