from dataclasses import replace
from types import SimpleNamespace
import json

import pytest

from baloot_platform.core.rules import RuleBook
from baloot_platform.research.family_catalog import (
    ResearchFamilyCatalogV1, validate_family_construction_expression_v1, family_construction_expression_v1,
)
from baloot_platform.research.family_construction import (
    construct_family_proposal_v1, construct_history_family_proposal_v1, family_proposal_witness_v1,
)
from baloot_platform.research.scenario_contracts import ExactTracePrefixV1, ScenarioExpressionV1, canonical_sha256
from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
from baloot_platform.research.ortools_play_tail import TailPrefixActionV1, PlayTailSolveStopped
from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Seat, Team
from baloot_platform.research.scenario_predicate_lowering_v2 import (
    CpSatActionVariablesV1, CpSatPredicateModelContextV1, CpSatTrickVariablesV1,
)


def arguments(root):
    family = ResearchFamilyCatalogV1.load(root / 'configs/research/families.yaml').families[0]
    return dict(family=family, bindings=dict(suit='spades', seat='north', rank='seven',
        projectType='siri', contract='sun', playIndex=24),
        rules=RuleBook.load(root / 'configs/rules/saudi_federation_2023_v7.yaml'),
        dealer='east', buyer='north', trump=None, maximum_seconds=30,
        maximum_conflicts=10000, seed=4)


def hokum_arguments(root):
    raw = ResearchFamilyCatalogV1.load(root / 'configs/research/families.yaml').to_dict()
    raw.pop('sha256')
    family = next(row for row in raw['families'] if row['id'] == 'stopper_guard')
    family.update(enabled=True, contracts=['hokum'])
    family['capabilityTags']['mode'] = ['hokum']
    family['template']['expression'] = {'op': 'all', 'children': [
        {'predicateId': 'PRED-HOKUM', 'kind': 'auction.contract', 'version': 1, 'hardness': 'hard',
         'arguments': {'contract': {'$parameter': 'contract'}}},
        {'predicateId': 'PRED-CHOICES', 'kind': 'state.legal_action_count', 'version': 1,
         'hardness': 'hard', 'arguments': {'minimum': 1, 'maximum': 4}},
        {'predicateId': 'PRED-PROJECTS', 'kind': 'project.count', 'version': 1, 'hardness': 'hard',
         'arguments': {'team': 'north_south', 'minimum': 0, 'maximum': 16, 'state': 'declared'}},
    ]}
    raw['families'] = [family]
    args = arguments(root)
    return {**args, 'family': ResearchFamilyCatalogV1.build(raw).families[0],
        'bindings': {**args['bindings'], 'contract': 'hokum'},
        'rules': RuleBook.load(root / 'configs/rules/saudi_federation_2023_v8.yaml'),
        'dealer': 'south', 'buyer': 'north', 'trump': 'spades', 'seed': 209941006}


def constructed_initial_prefix():
    # Native allocation witness; neither its legal auction nor its projects prove interest.
    return ExactTracePrefixV1.build(dict(schemaVersion=1, dealer='west', roundNumber=1,
        matchScore=[10, 20], deckOrder=[16, 17, 21, 8, 9, 11, 6, 7, 14, 4, 5, 18, 27, 31, 26, 28,
            15, 30, 20, 22, 1, 10, 12, 13, 19, 25, 29, 0, 2, 3, 23, 24],
        auctionActions=[], escalationActions=[], projectActions=[], projectDeclarations=[],
        projectAssertions=[], playActions=[], stateAssertions={}))


def test_native_initial_allocation_cancels_before_build_and_rejects_invalid_limits(root, monkeypatch):
    from baloot_platform.research import family_construction as module
    monkeypatch.setattr(module, 'OrToolsScenarioModelV1', lambda **kw: pytest.fail('cancelled allocation built a model'))
    args = arguments(root)
    result = module.construct_family_initial_prefix_v1(**args, cancel_check=lambda: True)
    assert result['status'] == 'incomplete' and result['qualified'] is False
    for limit in (0, True, float('inf')):
        with pytest.raises(ValueError, match='budget'):
            module.construct_family_initial_prefix_v1(**{**args, 'maximum_seconds': limit})


def test_constructed_initial_deal_preserves_cards_purchase_and_score_without_shuffle(root, monkeypatch):
    from baloot_platform.core.engine import BalootEngine
    from baloot_platform.core.dealing import DealLayoutV1
    from baloot_platform.research.family_construction import _family_initial_deal_v1

    rules = RuleBook.load(root / 'configs/rules/saudi_federation_2023_v8.yaml')
    prefix = constructed_initial_prefix()
    monkeypatch.setattr(BalootEngine, 'shuffle_and_initial_deal',
        lambda *a, **kw: pytest.fail('constructed opening reshuffled its deck'))
    _, state, initial = _family_initial_deal_v1(rules, 72, 'west', initial_prefix=prefix)
    layout = DealLayoutV1.build(rules.structural_domain, dealer=Seat.WEST, buyer=Seat.NORTH)
    assert initial.to_dict() == prefix.to_dict() and initial.project_actions_present
    assert tuple(state.shuffle.deck_order) == prefix.deck_order
    assert state.purchase_card.id == prefix.deck_order[layout.purchase_position]
    assert state.match_score == [10, 20] and state.round_number == 1
    for seat in Seat:
        assert {card.id for card in state.hand(seat)} == {
            prefix.deck_order[position] for position in layout.positions_for(seat, final=False)}


@pytest.mark.parametrize('change', ['dealer', 'round', 'auction', 'play', 'checksum', 'duplicate'])
def test_constructed_initial_deal_rejects_noninitial_or_corrupt_inputs(root, monkeypatch, change):
    from baloot_platform.core.engine import BalootEngine
    from baloot_platform.research.family_construction import _family_initial_deal_v1

    rules = RuleBook.load(root / 'configs/rules/saudi_federation_2023_v8.yaml')
    prefix = constructed_initial_prefix().to_dict()
    prefix.pop('sha256')
    if change == 'dealer':
        prefix['dealer'] = 'east'
    elif change == 'round':
        prefix['roundNumber'] = 2
    elif change == 'auction':
        prefix['auctionActions'] = [{'actor': 'north', 'action': 'sun'}]
    elif change == 'play':
        prefix['playActions'] = [{'actor': 'north', 'card': 'S7'}]
    elif change == 'checksum':
        prefix['sha256'] = '0' * 64
    else:
        prefix['deckOrder'][0] = prefix['deckOrder'][1]
    monkeypatch.setattr(BalootEngine, 'initial_deal_from_preordered',
        lambda *a, **kw: pytest.fail('invalid prefix reached dealing'))
    with pytest.raises(ValueError):
        _family_initial_deal_v1(rules, 72, 'west', initial_prefix=prefix)


def test_constructed_initial_prefix_cannot_bypass_native_auction_or_cut_rules(root):
    from baloot_platform.research.family_construction import _family_initial_deal_v1

    args = {**arguments(root), 'dealer': 'west'}
    prefix = constructed_initial_prefix()
    with pytest.raises(ValueError, match='qualified auction provider'):
        construct_history_family_proposal_v1(**args,
            initial_prefix=prefix, rollout_policy_id='observer_safe_trick_control_v1')
    data = args['rules'].data
    data['dealing']['allow_no_cut'] = False
    rules = RuleBook(data)
    with pytest.raises(ValueError, match='post-cut constructed deck'):
        _family_initial_deal_v1(rules, 72, 'west', initial_prefix=prefix)


@pytest.mark.parametrize('field,value', [
    ('roundNumber', 2), ('matchScore', [11, 20]),
    ('playActions', [{'actor': 'north', 'card': 'S7'}]),
    ('projectActions', [{'actor': 'north', 'action': 'pass'}]),
    ('projectAssertions', [{'owner': 'south', 'kind': 'siri', 'state': 'eligible'}]),
])
def test_constructed_opening_rejects_changed_metadata_or_postauction_evidence(root, field, value):
    from baloot_platform.research.family_construction import _qualified_family_opening_v1

    rules = RuleBook.load(root / 'configs/rules/saudi_federation_2023_v8.yaml')
    def auction(initial, **kwargs):
        raw = initial.to_dict()
        raw.pop('sha256')
        raw.update(auctionActions=[{'actor': 'north', 'action': 'sun'}], **{field: value})
        return dict(status='complete', operation='generate', qualificationAuthority='cp_sat',
            rulesHash=rules.sha256, coveredFromBidIndex=0, partialReceipt=None,
            sourcePrefixHash=initial.sha256, decisions=[{'actualActionOptimal': True}],
            finalPrefix=ExactTracePrefixV1.build(raw).to_dict())
    with pytest.raises(ValueError, match='changed its deal or contains post-auction actions'):
        _qualified_family_opening_v1(rules, 72, 'west', auction, lambda: 30, None,
            initial_prefix=constructed_initial_prefix())


@pytest.fixture
def native_seat_relations(monkeypatch):
    from baloot_platform.research.scenario_deal_predicates_v2 import DealPredicateLoweringRegistryV2

    original = DealPredicateLoweringRegistryV2._lower

    def lower(self, kind, *args):
        assert kind != 'deal.seat_relation', 'native construction used Python seat-relation truth'
        return original(self, kind, *args)

    monkeypatch.setattr(DealPredicateLoweringRegistryV2, '_lower', lower)


def test_native_initial_allocation_applies_balanced_hands_and_purchase_constraints(root, native_seat_relations):
    from baloot_platform.core.dealing import DealLayoutV1
    from baloot_platform.core.enums import Rank
    from baloot_platform.research.family_construction import construct_family_initial_prefix_v1

    args = {**arguments(root), 'dealer': 'west', 'seed': 1792287200,
        'maximum_conflicts': 1000000,
        'rules': RuleBook.load(root / 'configs/rules/saudi_federation_2023_v8.yaml')}
    raw = args['family'].to_dict()
    raw['template']['expression'] = {'op': 'all', 'children': [
        {'predicateId': 'PRED-PARTNER', 'kind': 'deal.seat_relation', 'version': 1, 'hardness': 'hard',
         'arguments': {'seat': 'north', 'otherSeat': 'south', 'relation': 'partner'}},
        {'predicateId': 'PRED-PURCHASE', 'kind': 'deal.purchase_card', 'version': 1, 'hardness': 'hard',
         'arguments': {'card': Card.parse('H7').id}},
        *[{'predicateId': 'PRED-HIGH-' + seat.name, 'kind': 'deal.high_card_count', 'version': 1,
           'hardness': 'hard', 'arguments': {'seat': seat.name.lower(), 'handStage': 'post_purchase',
                                          'minimum': 3, 'maximum': 3}} for seat in Seat],
    ]}
    args['family'] = replace(args['family'], _json=json.dumps(raw))
    result = construct_family_initial_prefix_v1(**args)
    assert result['status'] == 'constructed', result.get('reason')
    assert result['qualified'] is False
    completion = result['allocation']['auction']['completion']
    assert completion['contract'] == 'sun' and completion['buyer'] == 'north'
    deck = result['initialPrefix']['deckOrder']
    layout = DealLayoutV1.build(args['rules'].structural_domain, dealer=Seat.WEST, buyer=Seat.NORTH)
    assert deck[layout.purchase_position] == Card.parse('H7').id
    assert [sum(Card(deck[position]).rank in {Rank.ACE, Rank.TEN, Rank.KING}
                for position in layout.positions_for(seat, final=True)) for seat in Seat] == [3] * 4


@pytest.mark.parametrize('conflict', ['purchase_owner', 'auction_buyer', 'seat_relation', 'remaining_owner'])
def test_native_initial_allocation_rejects_conflicting_structural_constraints(root, conflict):
    from baloot_platform.research.family_construction import construct_family_initial_prefix_v1

    args = {**arguments(root), 'dealer': 'west', 'seed': 1792287200,
        'maximum_conflicts': 1000000,
        'rules': RuleBook.load(root / 'configs/rules/saudi_federation_2023_v8.yaml')}
    raw = args['family'].to_dict()
    predicates = ([
        {'predicateId': 'PRED-PURCHASE', 'kind': 'deal.purchase_card', 'version': 1, 'hardness': 'hard',
         'arguments': {'card': Card.parse('H7').id}},
        {'predicateId': 'PRED-OWNER', 'kind': 'deal.card_owner', 'version': 1, 'hardness': 'hard',
         'arguments': {'card': Card.parse('H7').id, 'seat': 'west', 'handStage': 'post_purchase'}},
    ] if conflict == 'purchase_owner' else [
        {'predicateId': 'PRED-BUYER', 'kind': 'auction.buyer', 'version': 1, 'hardness': 'hard',
         'arguments': {'seat': 'west'}},
    ])
    if conflict == 'seat_relation':
        predicates = [{'predicateId': 'PRED-PARTNER', 'kind': 'deal.seat_relation', 'version': 1,
            'hardness': 'hard', 'arguments': {'seat': 'north', 'otherSeat': 'west', 'relation': 'partner'}}]
    if conflict == 'remaining_owner':
        predicates = [
            {'predicateId': 'PRED-OWNER', 'kind': 'deal.card_owner', 'version': 1, 'hardness': 'hard',
             'arguments': {'card': Card.parse('H7').id, 'seat': 'north', 'handStage': 'post_purchase'}},
            {'predicateId': 'PRED-REMAINING', 'kind': 'deal.remaining_card_owner', 'version': 1,
             'hardness': 'hard', 'arguments': {'card': Card.parse('H7').id, 'seat': 'west'}},
        ]
    raw['template']['expression'] = {'op': 'all', 'children': predicates}
    args['family'] = replace(args['family'], _json=json.dumps(raw))
    result = construct_family_initial_prefix_v1(**args)
    assert result['status'] == 'seed_miss' and result['reason'] == 'native_allocation_infeasible'
    assert result['qualified'] is False and 'initialPrefix' not in result


def test_native_initial_allocation_defers_remaining_and_disjunctive_constraints(root):
    from baloot_platform.research.family_construction import construct_family_initial_prefix_v1

    args = {**arguments(root), 'dealer': 'west', 'seed': 1792287200,
        'maximum_conflicts': 1000000,
        'rules': RuleBook.load(root / 'configs/rules/saudi_federation_2023_v8.yaml')}
    raw = args['family'].to_dict()
    raw['template']['expression'] = {'op': 'all', 'children': [
        {'predicateId': 'PRED-REMAINING', 'kind': 'deal.remaining_suit_count', 'version': 1,
         'hardness': 'hard', 'arguments': {'seat': 'north', 'suit': 'spades', 'minimum': 0, 'maximum': 0}},
        {'op': 'any', 'children': [
            {'predicateId': 'PRED-BUYER-' + seat.upper(), 'kind': 'auction.buyer', 'version': 1,
             'hardness': 'hard', 'arguments': {'seat': seat}} for seat in ('north', 'west')]},
    ]}
    args['family'] = replace(args['family'], _json=json.dumps(raw))
    result = construct_family_initial_prefix_v1(**args)
    assert result['status'] == 'constructed' and result['qualified'] is False
    assert result['expressionHash'] == args['family'].instantiate(args['bindings']).sha256


@pytest.mark.parametrize('scope', ['mandatory', 'any', 'not', 'count'])
def test_native_initial_allocation_lifts_only_necessary_remaining_ownership(root, scope):
    from baloot_platform.core.dealing import DealLayoutV1
    from baloot_platform.research.family_construction import construct_family_initial_prefix_v1

    args = {**arguments(root), 'dealer': 'west', 'seed': 1792287200,
        'maximum_conflicts': 1000000,
        'rules': RuleBook.load(root / 'configs/rules/saudi_federation_2023_v8.yaml')}
    card = Card.parse('H7').id
    remaining = {'predicateId': 'PRED-REMAINING', 'kind': 'deal.remaining_card_owner', 'version': 1,
        'hardness': 'hard', 'arguments': {'card': card, 'seat': 'west'}}
    north = {'predicateId': 'PRED-NORTH', 'kind': 'deal.card_owner', 'version': 1,
        'hardness': 'hard', 'arguments': {'card': card, 'seat': 'north', 'handStage': 'post_purchase'}}
    conditional = ({'op': 'any', 'children': [remaining, {**north, 'predicateId': 'PRED-NORTH-ALTERNATIVE'}]}
        if scope == 'any' else
        {'op': 'count', 'minimum': 0, 'maximum': 1, 'children': [remaining]} if scope == 'count' else
        {'op': 'not', 'children': [remaining]})
    raw = args['family'].to_dict()
    raw['template']['expression'] = ({'op': 'all', 'children': [remaining]} if scope == 'mandatory' else
        {'op': 'all', 'children': [north, conditional]})
    args['family'] = replace(args['family'], _json=json.dumps(raw))
    result = construct_family_initial_prefix_v1(**args)
    assert result['status'] == 'constructed' and result['qualified'] is False
    layout = DealLayoutV1.build(args['rules'].structural_domain, dealer=Seat.WEST, buyer=Seat.NORTH)
    seat = Seat.WEST if scope == 'mandatory' else Seat.NORTH
    assert card in [result['initialPrefix']['deckOrder'][position]
                    for position in layout.positions_for(seat, final=True)]


@pytest.mark.parametrize('state', ['declared', 'comparison_winner', 'baloot'])
def test_native_allocation_uses_only_the_requested_project_lifecycle(root, monkeypatch, state):
    from baloot_platform.research import ortools_project_fragment as projects
    from baloot_platform.research.family_construction import construct_family_initial_prefix_v1

    args = {**arguments(root), 'rules': RuleBook.load(root / 'configs/rules/saudi_federation_2023_v8.yaml')}
    raw = args['family'].to_dict()
    raw['template']['expression'] = ({'predicateId': 'PRED-BALOOT', 'kind': 'project.baloot',
        'version': 1, 'hardness': 'hard', 'arguments': {'team': 'north_south', 'achieved': False}}
        if state == 'baloot' else {'predicateId': 'PRED-COUNT', 'kind': 'project.count',
            'version': 1, 'hardness': 'hard',
            'arguments': {'team': 'north_south', 'minimum': 1, 'maximum': 16, 'state': state}})
    args['family'] = replace(args['family'], _json=json.dumps(raw))
    original = projects.attach_project_lifecycle_fragment_v1

    def attach(*a, **kwargs):
        assert kwargs.get('declaration_only', False) is (state == 'declared')
        assert kwargs['include_declaration_actions'] is (state != 'declared')
        assert (kwargs['second_trick_leader'] is None) is (state == 'declared')
        if state != 'declared':
            raise RuntimeError('full lifecycle retained')
        return original(*a, **kwargs)

    monkeypatch.setattr(projects, 'attach_project_lifecycle_fragment_v1', attach)
    if state != 'declared':
        with pytest.raises(RuntimeError, match='full lifecycle retained'):
            construct_family_initial_prefix_v1(**args)
    else:
        # Allocation took 30s under the shared CPU cap; this checks projection, not latency.
        result = construct_family_initial_prefix_v1(**{**args, 'maximum_seconds': 90})
        assert result['status'] == 'constructed' and result['qualified'] is False, {
            'reason': result.get('reason'), **{key: result.get('allocation', {}).get(key)
                for key in ('status', 'solverStatus', 'branches', 'conflicts')}}
        assert result['constructionNarrowing']['projectScope'] == 'project_declarations'
        assert result['constructionNarrowing']['unknownRevealLeader'] is False


@pytest.mark.parametrize('native_allocation', [False, 'full', 'declarations'])
def test_constructed_opening_passes_fresh_native_auction_quality_without_changing_deal(root, native_allocation, deterministic_work):
    import time
    import yaml
    from baloot_platform.core.models import LegalPlayActionV1
    from baloot_platform.research.construction_pre_screen import _proposal_policy_score
    from baloot_platform.research.family_construction import (
        _qualified_family_opening_v1, construct_family_initial_prefix_v1,
    )
    from baloot_platform.research.ortools_research_executor import OrToolsUnifiedResearchExecutorV1
    from baloot_platform.research.scenario_contracts import (
        ScenarioPlayingQualityPolicyV1, ScenarioQualityExecutionBudgetV1,
    )

    rules = RuleBook.load(root / 'configs/rules/saudi_federation_2023_v8.yaml')
    section = yaml.safe_load((root / 'configs/challenge/intelligent_research.yaml').read_text(
        encoding='utf-8'))['scenarioResearchV4']
    quality = ScenarioPlayingQualityPolicyV1.build(dict(schemaVersion=2, mode='budgeted',
        visibility='actor_information_set', decisions={kind: dict(utility='awarded_point_margin',
            budgetProfile='bounded_v1', horizon={'unit': 'plies', 'value': 1}, boundaryEvaluator=boundary)
            for kind, boundary in {'bid': 'auction_hand_margin_v1',
                'escalation': 'current_raw_point_margin_v1', 'project': 'current_raw_point_margin_v1',
                'play': 'current_raw_point_margin_v1'}.items()}))
    executor = OrToolsUnifiedResearchExecutorV1(rules, backend_id='ortools_cp_sat_v3',
        bundle_revision_id='CBR-CONSTRUCTED-FAMILY-TEST', bundle_hash='a' * 64,
        quality_policy=quality,
        quality_execution=ScenarioQualityExecutionBudgetV1.build(section['qualityExecution']),
        quality_proof_profiles=section['proofBudgets']['profiles'],
        quality_search_profiles=section['searchBudgets']['profiles'])
    args = {**arguments(root), 'rules': rules, 'dealer': 'west', 'seed': 1792287200,
        'maximum_conflicts': 250000}
    raw = args['family'].to_dict()
    raw['template']['expression'] = {'op': 'all', 'children': [
        {'predicateId': 'PRED-PROJECT-' + team.upper(), 'kind': 'project.count',
         'version': 1, 'hardness': 'hard',
         'arguments': {'team': team, 'state': 'declared', 'minimum': 1, 'maximum': 16}}
        for team in ('north_south', 'west_east')]}
    if native_allocation == 'full':
        raw['template']['expression']['children'].append({'predicateId': 'PRED-AWARDED',
            'kind': 'project.count', 'version': 1, 'hardness': 'hard',
            'arguments': {'team': 'north_south', 'state': 'awarded', 'minimum': 0, 'maximum': 16}})
    args['family'] = replace(args['family'], _json=json.dumps(raw))
    # The full allocation alone took ~30s; later history keeps its original budget.
    allocation = construct_family_initial_prefix_v1(**{**args, 'maximum_seconds': 90}) if native_allocation else None
    if native_allocation:
        assert allocation['status'] == 'constructed', {
            'reason': allocation.get('reason'), **{key: allocation.get('allocation', {}).get(key)
                for key in ('status', 'solverStatus', 'branches', 'conflicts')}}
        initial = ExactTracePrefixV1.build(allocation['initialPrefix'])
        assert allocation['allocation']['auction']['completion']['buyer'] == 'north'
    else:
        initial = constructed_initial_prefix()
    engine, state, opening, receipt = _qualified_family_opening_v1(rules, 72, 'west',
        executor.construct_qualified_auction, lambda: 30, None, initial_prefix=initial)
    assert receipt['status'] == 'complete' and receipt['qualificationAuthority'] == 'cp_sat'
    bid_count = len(opening['auctionActions'])
    assert receipt['coveredFromBidIndex'] == 0 and receipt['coveredToBidIndex'] == bid_count
    assert [row['bidIndex'] for row in receipt['decisions']] == list(range(bid_count))
    assert bid_count > 0 and all(row['actualActionOptimal'] for row in receipt['decisions'])
    assert receipt['sourcePrefixHash'] == initial.sha256
    assert state.buyer == Seat.NORTH and state.contract.value == 'sun'
    assert opening['deckOrder'] == list(initial.deck_order) and tuple(opening['matchScore']) == initial.match_score
    assert tuple(state.match_score) == initial.match_score and all(p.declared is False for p in state.projects)
    for _ in range(24):
        legal = engine.legal_play_actions(state)
        action = legal[0] if len(legal) == 1 else LegalPlayActionV1.from_code(_proposal_policy_score(
            rules, engine, state, 'observer_safe_trick_control_v1', deterministic_work.read("monotonic") + 10, lambda: None)[0])
        engine.play_card(state, action.card, ekka=action.ekka)
    args['bindings']['seat'] = state.current_player.name.lower()
    proposal = construct_history_family_proposal_v1(**args, initial_prefix=initial, policy_prefix_steps=24,
        rollout_policy_id='observer_safe_trick_control_v1', auction_provider=executor.construct_qualified_auction)
    assert proposal['status'] == 'proposal', {key: proposal.get(key) for key in ('status', 'reason', 'solverStatus')}
    assert proposal['qualified'] is False and all(row['satisfied'] for row in proposal['predicateEvidence'])
    complete = ExactTracePrefixV1.build(proposal['completePrefix'])
    assert len(complete.play_actions) == 32 and len(proposal['decisionPrefix']['playActions']) == 24
    assert complete.deck_order == initial.deck_order and complete.match_score == initial.match_score
    assert list(complete.auction_actions) == opening['auctionActions']
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(
        complete, maximum_seconds=30)
    normalized.require_integrity()
    if native_allocation:
        from baloot_platform.research.scenario_contracts import build_family_opening_evidence_v1
        from baloot_platform.research.family_catalog import build_family_target_v1
        from baloot_platform.research.result_authority import validate_trace_quality_evidence_v1
        proposal['familyOpening'] = {**allocation, 'auctionQualification': proposal.pop('nativeOpeningAuction')}
        policy = json.loads((root / 'tests/fixtures/actor_interest_calibration.json').read_text())['cases']['7189']['policy']
        witness = family_proposal_witness_v1(proposal, rules=rules, interest_policy=policy)
        evidence = witness['familyOpening']
        if native_allocation == 'full':
            assert set(evidence['constructionNarrowing']) == {
                'visibleHandAuctionPolicy', 'unknownRevealLeader', 'projectCardOwners'}
            assert evidence['constructionNarrowing']['unknownRevealLeader'] is True
        else:
            assert evidence['constructionNarrowing']['projectScope'] == 'project_declarations'
            assert evidence['constructionNarrowing']['unknownRevealLeader'] is False
        kwargs = dict(prefix=ExactTracePrefixV1.build(witness['prefix']), rules_hash=rules.sha256,
                      source_seed=args['seed'])
        assert build_family_opening_evidence_v1(evidence, **kwargs) == evidence
        assert 'executionReceipt' not in evidence['auctionQualification']
        validate_trace_quality_evidence_v1({'familyOpening': evidence, 'exactTracePrefix': complete.to_dict()},
                                          rules_hash=rules.sha256)
        catalog = ResearchFamilyCatalogV1.load(root / 'configs/research/families.yaml').to_dict()
        catalog.pop('sha256')
        catalog['families'][0] = {**args['family'].to_dict(), 'enabled': True}
        target = build_family_target_v1(dict(schemaVersion=1, catalog=catalog, familyId=args['family'].family_id,
            bindings=args['bindings'], dealer='west', buyer='north', trump=None, tagQuery={'mode': ['sun']}))
        execution = ScenarioQualityExecutionBudgetV1.build({**section['qualityExecution'],
            'candidateSource': 'runtime_proposal_stream_v1', 'familyOpeningMode': 'native_project_auction_v1',
            'familyTarget': target, 'proposalSeedCount': 1, 'auditConstructedTrace': True,
            'exactPlayFromTrick': 7, 'earlyPlayPolicyId': 'observer_safe_trick_control_v1',
            'videoInterestingness': policy})
        native_executor = OrToolsUnifiedResearchExecutorV1(rules, backend_id='ortools_cp_sat_v3',
            bundle_revision_id='CBR-CONSTRUCTED-FAMILY-TEST', bundle_hash='a' * 64,
            quality_policy=quality, quality_execution=execution,
            quality_proof_profiles=section['proofBudgets']['profiles'],
            quality_search_profiles=section['searchBudgets']['profiles'], runtime_proposals={0: witness})
        assert native_executor.runtime_proposals[0]['familyOpening'] == evidence
        from baloot_platform.research.ortools_research_executor import OrToolsResearchPreflightV1
        preflight = OrToolsResearchPreflightV1('search', 'playPrefix', ('deal', 'playPrefix'), (), (), (), 'a' * 64)
        candidate = executor._candidate_from_normalized_exact(normalized, complete, origin='search')
        retained = executor._result('semantic_postcondition_miss', preflight, candidate, None, None, (),
            time.perf_counter(), time.process_time(), 1, scenario_hash='c' * 64, _family_opening=evidence)
        assert retained.candidate.trace['familyOpening'] == evidence
        assert retained.research_authority['traceHash'] == canonical_sha256(retained.candidate.trace)
        assert retained.artifact_preimages.to_dict()['trace']['familyOpening'] == evidence
        for change in ({'visibleHandAuctionPolicy': False}, {'projectScope': 'unsupported'},
                {'unknownRevealLeader': not evidence['constructionNarrowing']['unknownRevealLeader']},
                {'unexpected': True}):
            altered = json.loads(json.dumps(evidence))
            altered['constructionNarrowing'].update(change)
            altered['sha256'] = canonical_sha256({key: value for key, value in altered.items() if key != 'sha256'})
            with pytest.raises(ValueError, match='narrowing mismatch'):
                build_family_opening_evidence_v1(altered, **kwargs)
        missing = {key: value for key, value in witness.items() if key not in {'sha256', 'familyOpening'}}
        missing['sha256'] = canonical_sha256(missing)
        with pytest.raises(ValueError, match='proposal data identity'):
            OrToolsUnifiedResearchExecutorV1(rules, backend_id='ortools_cp_sat_v3',
                bundle_revision_id='CBR-CONSTRUCTED-FAMILY-TEST', bundle_hash='a' * 64,
                quality_policy=quality, quality_execution=execution,
                quality_proof_profiles=section['proofBudgets']['profiles'],
                quality_search_profiles=section['searchBudgets']['profiles'], runtime_proposals={0: missing})
        altered = json.loads(json.dumps(evidence))
        altered['allocation']['solutionHash'] = '0' * 64
        altered['sha256'] = canonical_sha256({key: value for key, value in altered.items() if key != 'sha256'})
        with pytest.raises(ValueError, match='solution checksum'):
            build_family_opening_evidence_v1(altered, **kwargs)
        altered = json.loads(json.dumps(evidence))
        altered['allocation']['auction']['completion']['trumpSuit'] = 'clubs'
        altered['sha256'] = canonical_sha256({key: value for key, value in altered.items() if key != 'sha256'})
        with pytest.raises(ValueError, match='ownership mismatch'):
            build_family_opening_evidence_v1(altered, **kwargs)


def test_constructed_history_preserves_initial_state_in_precheck_auction_and_native_tail(root, monkeypatch):
    from baloot_platform.core.auction_ir import AuctionDomainIRV1
    from baloot_platform.core.bidding_machine import BiddingMachine
    from baloot_platform.core.dealing import DealLayoutV1
    from baloot_platform.core.engine import BalootEngine
    from baloot_platform.research import family_construction as module, construction_pre_screen

    args = {**arguments(root), 'dealer': 'west',
        'rules': RuleBook.load(root / 'configs/rules/saudi_federation_2023_v8.yaml')}
    raw = args['family'].to_dict()
    raw['template']['expression'] = {'predicateId': 'PRED-PROJECTS', 'kind': 'project.count',
        'version': 1, 'hardness': 'hard',
        'arguments': {'team': 'north_south', 'state': 'declared', 'minimum': 0, 'maximum': 16}}
    args['family'] = replace(args['family'], _json=json.dumps(raw))
    initial = constructed_initial_prefix()
    prechecks, auctions, native_calls, auction_actions = [], [], [], []
    def precheck(*a, **kwargs):
        prechecks.append(kwargs['opening'])
        return None
    def auction(prefix, **kwargs):
        auctions.append(prefix)
        raw = prefix.to_dict()
        raw.pop('sha256')
        layout = DealLayoutV1.build(args['rules'].structural_domain, dealer=Seat.WEST, buyer=Seat.NORTH)
        card = Card(prefix.deck_order[layout.purchase_position])
        machine = BiddingMachine(Seat.WEST, card.suit, AuctionDomainIRV1.from_rules(args['rules'].data), card.rank)
        while not machine.finished:
            action = 'sun' if machine.actor == Seat.NORTH and not machine.in_recall else 'pass'
            auction_actions.append({'actor': machine.actor.name.lower(), 'action': action})
            machine.submit(action)
        raw['auctionActions'] = auction_actions
        return dict(status='complete', operation='generate', qualificationAuthority='cp_sat',
            rulesHash=args['rules'].sha256, coveredFromBidIndex=0, partialReceipt=None,
            sourcePrefixHash=prefix.sha256, decisions=[{'actualActionOptimal': True}],
            finalPrefix=ExactTracePrefixV1.build(raw).to_dict(), sha256='b' * 64,
            executionReceipt={'elapsedSeconds': 0})
    def construct(**kwargs):
        native_calls.append(kwargs)
        return {'status': 'proposal', 'qualified': False}
    monkeypatch.setattr(BalootEngine, 'shuffle_and_initial_deal',
        lambda *a, **kw: pytest.fail('constructed history reshuffled its deck'))
    monkeypatch.setattr(module, '_fixed_deal_project_status_v1', precheck)
    monkeypatch.setattr(module, 'construct_family_proposal_v1', construct)
    monkeypatch.setattr(construction_pre_screen, '_proposal_policy_score',
        lambda rules, engine, state, *a: (engine.legal_play_actions(state)[0].code, {}))
    result = construct_history_family_proposal_v1(**args, initial_prefix=initial,
        rollout_policy_id='observer_safe_trick_control_v1', auction_provider=auction)
    assert result['status'] == 'proposal' and result['qualified'] is False
    assert len(prechecks) == 2 and prechecks[0] == initial.to_dict()
    assert [row.to_dict() for row in auctions] == [initial.to_dict()]
    assert all(row['deckOrder'] == list(initial.deck_order) for row in prechecks)
    assert len(native_calls) == 1 and len(native_calls[0]['prefix']) == 20
    assert native_calls[0]['required_deck_order'] == list(initial.deck_order)
    assert native_calls[0]['match_score'] == initial.match_score
    assert native_calls[0]['auction_actions'] == auction_actions
    assert result['auctionQualification']['sourcePrefixHash'] == initial.sha256


@pytest.mark.parametrize('visibility,index', [('complete_state', 24), ('actor_information_set', 27)])
def test_hokum_history_uses_native_qualified_auction_and_proves_native_tail(root, monkeypatch, visibility, index):
    import time
    import yaml
    from baloot_platform.core.models import LegalPlayActionV1
    from baloot_platform.core.scoring_ir import ScoringDomainIRV1
    from baloot_platform.research import ortools_project_fragment
    from baloot_platform.research.construction_pre_screen import _proposal_policy_score
    from baloot_platform.research.family_construction import _qualified_family_opening_v1
    from baloot_platform.research.ortools_complete_state_graph import OrToolsCompleteStateGraphCompilerV1
    from baloot_platform.research.ortools_information_set_compiler import (
        OrToolsActorInformationSetCompilerV1, prove_actor_information_set_v1,
    )
    from baloot_platform.research.ortools_proof import OrToolsScenarioGraphProofV1
    from baloot_platform.research.ortools_research_executor import OrToolsUnifiedResearchExecutorV1
    from baloot_platform.research.scenario_contracts import (
        ScenarioPlayingQualityPolicyV1, ScenarioProofRequestV1, ScenarioQualityExecutionBudgetV1,
    )

    args = hokum_arguments(root)
    args['bindings']['playIndex'] = index
    section = yaml.safe_load((root / 'configs/challenge/intelligent_research.yaml').read_text(
        encoding='utf-8'))['scenarioResearchV4']
    quality = ScenarioPlayingQualityPolicyV1.build({'schemaVersion': 2, 'mode': 'budgeted',
        'visibility': 'actor_information_set', 'decisions': {
            kind: {'utility': 'awarded_point_margin', 'budgetProfile': 'bounded_v1',
                   'horizon': {'unit': 'plies', 'value': 1}, 'boundaryEvaluator': boundary}
            for kind, boundary in {'bid': 'auction_hand_margin_v1',
                'escalation': 'current_raw_point_margin_v1', 'project': 'current_raw_point_margin_v1',
                'play': 'current_raw_point_margin_v1'}.items()}})
    executor = OrToolsUnifiedResearchExecutorV1(args['rules'], backend_id='ortools_cp_sat_v3',
        bundle_revision_id='CBR-HOKUM-FAMILY-TEST', bundle_hash='a' * 64,
        quality_policy=quality,
        quality_execution=ScenarioQualityExecutionBudgetV1.build(section['qualityExecution']),
        quality_proof_profiles=section['proofBudgets']['profiles'],
        quality_search_profiles=section['searchBudgets']['profiles'])
    engine, state, opening, receipt = _qualified_family_opening_v1(args['rules'], args['seed'],
        args['dealer'], executor.construct_qualified_auction, lambda: 30, None)
    assert receipt['status'] == 'complete' and receipt['qualificationAuthority'] == 'cp_sat'
    assert len(receipt['decisions']) == 8 and all(row['actualActionOptimal'] for row in receipt['decisions'])
    assert state.contract.value == 'hokum' and state.trump_suit.name.lower() == 'spades'
    assert state.buyer == Seat.NORTH
    for _ in range(index):
        legal = engine.legal_play_actions(state)
        action = legal[0] if len(legal) == 1 else LegalPlayActionV1.from_code(_proposal_policy_score(
            args['rules'], engine, state, 'observer_safe_trick_control_v1', time.monotonic() + 10, lambda: None)[0])
        engine.play_card(state, action.card, ekka=action.ekka)
    args['bindings']['seat'] = state.current_player.name.lower()
    project_inputs = []
    attach = ortools_project_fragment.attach_project_lifecycle_fragment_v1

    def projects(*a, **kwargs):
        project_inputs.append((kwargs.get('contract'), kwargs.get('trump')))
        return attach(*a, **kwargs)

    monkeypatch.setattr(ortools_project_fragment, 'attach_project_lifecycle_fragment_v1', projects)
    proposal = construct_history_family_proposal_v1(**args, policy_prefix_steps=index,
        rollout_policy_id='observer_safe_trick_control_v1', auction_provider=executor.construct_qualified_auction)
    assert proposal['status'] == 'proposal', proposal
    assert proposal['qualified'] is False and all(row['satisfied'] for row in proposal['predicateEvidence'])
    assert ('hokum', 'spades') in project_inputs
    assert proposal['completePrefix']['deckOrder'] == opening['deckOrder']
    witness = family_proposal_witness_v1(proposal, rules=args['rules'], interest_policy={})
    normalizer = OrToolsExactStateNormalizerV1(args['rules'], native_cp_sat=True)
    normalized = normalizer.normalize(witness['prefix'], decision_point={'kind': 'play'}, maximum_seconds=30)
    normalized.require_integrity()
    assert normalized.state['contract'] == 'hokum' and normalized.state['trumpSuit'] == 'spades'
    assert normalized.next_actor == args['bindings']['seat']
    assert len(normalized.state['completedTricks']) == 6
    normalizer.normalize_play_or_project(ExactTracePrefixV1.build(proposal['completePrefix']), maximum_seconds=30)
    request = ScenarioProofRequestV1.build({'schemaVersion': 1, 'mode': 'exact', 'visibility': visibility,
        **({'strategyClass': 'pure_deterministic'} if visibility == 'actor_information_set' else {}),
        'targetTeam': Seat.parse(normalized.next_actor).team.name.lower(), 'utility': 'awarded_point_margin',
        'claim': 'winning_root_action_count', 'successThreshold': 0, 'budgetProfile': 'late_game_exact_v1'})
    if visibility == 'complete_state':
        compilation = OrToolsCompleteStateGraphCompilerV1(play_domain=args['rules'].play_domain,
            scoring_domain=ScoringDomainIRV1.from_rules(args['rules'].data),
            structural_domain=args['rules'].structural_domain, project_domain=args['rules'].project_domain,
            native_conversion=True).compile(
            normalized, request, expected_rules_hash=args['rules'].sha256, maximum_nodes=2000,
            maximum_depth=8, maximum_seconds=30)
        assert compilation.complete and compilation.graph is not None
        proof = OrToolsScenarioGraphProofV1().prove(compilation.graph, request, maximum_seconds=30)
    else:
        compilation = OrToolsActorInformationSetCompilerV1(args['rules'], native_cp_sat=True).compile(
            normalized, request, maximum_nodes=2000, maximum_depth=8, maximum_worlds=64, maximum_seconds=30)
        assert compilation.graph.hidden_worlds['complete'] is True
        proof = prove_actor_information_set_v1(compilation, request, maximum_nodes=2000,
            maximum_worlds=64, maximum_seconds=30, native_action_quality=True)
    assert proof.completion_status in {'exact_complete', 'proof_failed'}
    assert {row.action_code for row in proof.root_actions} == {row['code'] for row in normalized.legal_actions}
    assert all(type(row.value) is int for row in proof.root_actions)


def test_hokum_history_requires_qualified_auction_before_opening(root, monkeypatch):
    from baloot_platform.research import construction_pre_screen
    monkeypatch.setattr(construction_pre_screen, '_proposal_opening',
        lambda *a, **kw: pytest.fail('Hokum reached the Sun-only legacy opening'))
    with pytest.raises(ValueError, match='Hokum history requires a qualified auction'):
        construct_history_family_proposal_v1(**hokum_arguments(root), rollout_policy_id='observer_safe_trick_control_v1')


@pytest.mark.parametrize('contract,trump,dealer,buyer', [
    ('sun', None, 'south', 'north'), ('hokum', 'hearts', 'south', 'north'),
    ('hokum', None, 'south', 'north'), ('hokum', 'spades', 'east', 'north'),
    ('hokum', 'spades', 'south', 'west'),
])
def test_hokum_history_rejects_mismatched_qualified_opening(root, monkeypatch, contract, trump, dealer, buyer):
    from baloot_platform.core.enums import Contract, Suit
    from baloot_platform.research import family_construction as module
    state = SimpleNamespace(contract=Contract.parse(contract), trump_suit=Suit.parse(trump) if trump else None,
                            dealer=Seat.parse(dealer), buyer=Seat.parse(buyer))
    receipt = dict(status='complete', sha256='a' * 64, sourcePrefixHash='b' * 64,
                   executionReceipt={'elapsedSeconds': 0})
    monkeypatch.setattr(module, '_qualified_family_opening_v1', lambda *a, **kw: (None, state, {}, receipt))
    checks = []
    def precheck(*args, **kwargs):
        checks.append(kwargs)
        return None
    monkeypatch.setattr(module, '_fixed_deal_project_status_v1', precheck)
    result = construct_history_family_proposal_v1(**hokum_arguments(root),
        rollout_policy_id='observer_safe_trick_control_v1', auction_provider=object())
    assert result['status'] == 'seed_miss' and result['reason'] == 'opening_mismatch' and not result['qualified']
    assert len(checks) == 1 and checks[0]['contract'] == 'hokum' and checks[0]['trump'] == 'spades'
    assert not checks[0]['opening']['auctionActions']


@pytest.mark.real_clock
def test_configured_project_rank_family_constructs_a_fresh_native_round(root, native_seat_relations):
    import time
    from baloot_platform.research.construction_pre_screen import _proposal_opening, _proposal_policy_score
    from baloot_platform.core.models import LegalPlayActionV1

    args = arguments(root)
    args['family'], = ResearchFamilyCatalogV1.load(
        root / 'configs/research/public_project_rank_inference.yaml').families
    raw = args['family'].to_dict()
    raw['template']['expression'] = {'op': 'all', 'children': [raw['template']['expression'],
        {'predicateId': 'PRED-PARTNER', 'kind': 'deal.seat_relation', 'version': 1, 'hardness': 'hard',
         'arguments': {'seat': 'north', 'otherSeat': 'south', 'relation': 'partner'}}]}
    args['family'] = replace(args['family'], _json=json.dumps(raw))
    args['rules'] = RuleBook.load(root / 'configs/rules/saudi_federation_2023_v8.yaml')
    for seed in range(200):
        engine, state, opening = _proposal_opening(args['rules'], seed)
        if opening is None or state.contract.value != 'sun':
            continue
        projects = state.projects
        if (len(projects) == 2 and all(p.kind == 'siri' and p.value == 4 for p in projects)
                and len({p.team for p in projects}) == 2):
            break
    else:
        pytest.fail('no freshly seeded same-category Sun project comparison')
    prefix = []
    for _ in range(24):
        legal = engine.legal_play_actions(state)
        action = legal[0] if len(legal) == 1 else LegalPlayActionV1.from_code(
            _proposal_policy_score(args['rules'], engine, state,
                'observer_safe_trick_control_v1', time.monotonic() + 10, lambda: None)[0])
        prefix.append(TailPrefixActionV1.build(state.current_player, action.card, ekka=action.ekka))
        engine.play_card(state, action.card, ekka=action.ekka)
    args['bindings']['seat'] = state.current_player.name.lower()
    args.update(seed=seed, dealer=state.dealer.name.lower(), buyer=state.buyer.name.lower(),
        required_deck_order=opening['deckOrder'], auction_actions=opening['auctionActions'], prefix=tuple(prefix))
    from baloot_platform.research.family_construction import _fixed_deal_project_status_v1
    from ortools.sat.python import cp_model
    expression = args['family'].instantiate(args['bindings'])
    check = dict(rules=args['rules'], opening=opening, buyer=args['buyer'], contract='sun', trump=None,
        deadline=time.monotonic() + 30, maximum_conflicts=10000, seed=seed, cancel_check=None)
    assert _fixed_deal_project_status_v1(expression, **check) == cp_model.OPTIMAL
    wrong_count = ScenarioExpressionV1.build(dict(predicateId='PRED-NO-NS-PROJECT', kind='project.count',
        version=1, hardness='hard', arguments={'team': 'north_south', 'minimum': 0, 'maximum': 0,
                                             'state': 'declared'}))
    assert _fixed_deal_project_status_v1(wrong_count, **check) == cp_model.INFEASIBLE
    assert _fixed_deal_project_status_v1(expression, **{**check, 'cancel_check': lambda: True}) == cp_model.UNKNOWN
    assert _fixed_deal_project_status_v1(expression, **{**check, 'deadline': time.monotonic() - 1}) == cp_model.UNKNOWN
    result = construct_family_proposal_v1(**args)
    assert result['status'] == 'proposal', result
    assert result['qualified'] is False
    assert result['decisionPrefix']['playActions'] == [play.to_dict() for play in prefix]
    assert len(result['completePrefix']['playActions']) == 32
    assert len(result['predicateEvidence']) == 7
    assert all(row['satisfied'] is True for row in result['predicateEvidence'])
    normalized = OrToolsExactStateNormalizerV1(args['rules'], native_cp_sat=True).normalize(
        result['decisionPrefix'], decision_point={'kind': 'play'}, maximum_seconds=30)
    normalized.require_integrity()
    assert normalized.next_actor == args['bindings']['seat']
    assert len(normalized.state['completedTricks']) == 6
    assert len(normalized.legal_actions) >= 2
    choices, = (row for row in result['predicateEvidence']
                if row['predicateId'] == 'PRED-PROJECT-CHOICES')
    assert choices['satisfied'] is True


@pytest.mark.parametrize('op', ['all', 'any', 'not', 'count'])
@pytest.mark.parametrize('other', ['proof', 'prefer', 'report'])
def test_scenario_construction_keeps_only_complete_mandatory_native_clauses(root, op, other):
    args = arguments(root)
    original = args['family'].instantiate(args['bindings'])
    ace = dict(predicateId='PRED-USER-ACE', kind='play.action_matches', version=1, hardness='hard',
        arguments={'trickIndex': 6, 'position': 0, 'selector': {'ranks': ['ace']}})
    skipped = (dict(predicateId='PRED-USER-PROOF', kind='proof.winning_root_action_count', version=1,
        hardness='hard', arguments={'minimum': 1, 'maximum': 1}) if other == 'proof' else
        {**ace, 'predicateId': 'PRED-USER-SOFT', 'hardness': other})
    raw = {'op': op, 'children': [skipped] if op == 'not' else [ace, skipped]}
    if op == 'count':
        raw.update(minimum=1, maximum=1)
    scenario = SimpleNamespace(constraints=ScenarioExpressionV1.build(raw), outcome=original)
    combined = family_construction_expression_v1(original, scenario)
    ids = {predicate.predicate_id for predicate in combined.predicates()}
    assert ('PRED-USER-ACE' in ids) == (op == 'all')
    assert not ids.intersection({'PRED-USER-PROOF', 'PRED-USER-SOFT'})
    if op != 'all':
        assert combined.sha256 == original.sha256


@pytest.mark.parametrize('op', ['any', 'not', 'count'])
def test_complete_native_boolean_subtree_and_identical_ids_keep_their_semantics(root, op):
    args = arguments(root)
    original = args['family'].instantiate(args['bindings'])
    contract = next(predicate for predicate in original.predicates() if predicate.kind == 'auction.contract')
    raw = {'op': op, 'children': [contract.to_dict()]}
    if op == 'count':
        raw.update(minimum=0, maximum=0)
    extra = ScenarioExpressionV1.build(raw)
    combined = family_construction_expression_v1(original, SimpleNamespace(constraints=extra, outcome=original))
    assert [child.sha256 for child in combined.children] == [original.sha256, extra.sha256]
    changed = {key: value for key, value in contract.to_dict().items() if key != 'sha256'}
    conflict = ScenarioExpressionV1.build({**changed, 'arguments': {'contract': 'hokum'}})
    with pytest.raises(ValueError, match='conflicting family/scenario predicateId'):
        family_construction_expression_v1(original, SimpleNamespace(constraints=conflict, outcome=original))
    assert family_construction_expression_v1(original).sha256 == original.sha256


def test_cancelled_family_is_not_a_qualified_proposal(root):
    result = construct_family_proposal_v1(**arguments(root), cancel_check=lambda: True)
    assert result['status'] == 'incomplete'
    assert result['qualified'] is False
    assert 'deckOrder' not in result


@pytest.mark.parametrize('stage', ['auction', 'project'])
def test_family_cancellation_during_native_build_is_incomplete_not_an_exception(root, monkeypatch, stage):
    from baloot_platform.research import ortools_project_fragment, ortools_scenario_model

    cancelled = False
    original_auction = ortools_scenario_model.OrToolsScenarioModelV1.add_native_auction_fragment
    original_projects = ortools_project_fragment.attach_project_lifecycle_fragment_v1

    def auction(*args, **kwargs):
        nonlocal cancelled
        result = original_auction(*args, **kwargs)
        cancelled = stage == 'auction'
        return result

    def projects(*args, **kwargs):
        nonlocal cancelled
        cancelled = True
        return original_projects(*args, **kwargs)

    monkeypatch.setattr(ortools_scenario_model.OrToolsScenarioModelV1, 'add_native_auction_fragment', auction)
    monkeypatch.setattr(ortools_project_fragment, 'attach_project_lifecycle_fragment_v1', projects)
    result = construct_family_proposal_v1(**arguments(root), cancel_check=lambda: cancelled)
    assert result['status'] == 'incomplete' and result['solverStatus'] == 'cancelled'
    assert result['qualified'] is False
    assert not {'trace', 'deckOrder', 'constructionModelHash'}.intersection(result)


def test_history_producer_cancellation_precedes_opening(root, monkeypatch):
    from baloot_platform.research import construction_pre_screen
    monkeypatch.setattr(construction_pre_screen, '_proposal_opening',
                        lambda *args, **kwargs: pytest.fail('opening after cancellation'))
    result = construct_history_family_proposal_v1(**arguments(root),
        rollout_policy_id='observer_safe_trick_control_v1', cancel_check=lambda: True)
    assert result['status'] == 'incomplete' and not result['qualified']


def test_qualified_opening_replays_native_bids_without_heuristic_bidding(root, monkeypatch):
    from baloot_platform.core.engine import BalootEngine
    from baloot_platform.research.family_construction import _qualified_family_opening_v1
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1

    rules = arguments(root)['rules']
    seen = []
    def auction(initial, **kwargs):
        seen.append(initial)
        raw = initial.to_dict()
        raw.pop('sha256')
        raw['auctionActions'] = [{'actor': 'north', 'action': 'sun'}]
        final = ExactTracePrefixV1.build(raw)
        return {'status': 'complete', 'operation': 'generate', 'qualificationAuthority': 'cp_sat',
            'rulesHash': rules.sha256, 'coveredFromBidIndex': 0, 'partialReceipt': None,
            'sourcePrefixHash': initial.sha256, 'decisions': [{'actualActionOptimal': True}],
            'finalPrefix': final.to_dict()}
    monkeypatch.setattr(BalootEngine, 'bid_and_complete_deal',
        lambda *a, **kw: pytest.fail('native opening reran heuristic bidding'))
    _, state, prefix, _ = _qualified_family_opening_v1(rules, 4, 'east', auction, lambda: 10, None)
    assert state.buyer == Seat.NORTH and prefix['auctionActions'] == [{'actor': 'north', 'action': 'sun'}]
    assert tuple(prefix['deckOrder']) == seen[0].deck_order


@pytest.mark.parametrize('contract', ['sun', 'hokum'])
def test_incomplete_auction_stops_before_play_or_reallocation(root, monkeypatch, contract):
    from baloot_platform.research import family_construction as module, negative_project_prefix
    receipt = {'status': 'incomplete', 'sha256': 'b' * 64, 'sourcePrefixHash': 'c' * 64,
        'executionReceipt': {'elapsedSeconds': 2},
        'auctionOutcome': {'contract': 'sun', 'buyer': 'west'}}
    monkeypatch.setattr(module, '_qualified_family_opening_v1',
        lambda *a, **kw: (None, None, None, receipt))
    monkeypatch.setattr(negative_project_prefix, 'build_negative_project_prefix_v1',
        lambda **kw: pytest.fail('incomplete auction reached play construction'))
    args = arguments(root) if contract == 'sun' else hokum_arguments(root)
    result = construct_history_family_proposal_v1(**args,
        rollout_policy_id='observer_safe_trick_control_v1', auction_provider=lambda *a: None)
    assert result['status'] == 'incomplete' and result['reason'] == 'auction_quality_incomplete'
    assert result['auctionQualification']['status'] == 'incomplete' and not result['qualified']
    assert 'contract' not in result['auctionQualification']
    assert 'buyer' not in result['auctionQualification']


def test_unconstrained_buyer_requires_native_qualified_auction(root, monkeypatch):
    from baloot_platform.research import construction_pre_screen
    monkeypatch.setattr(construction_pre_screen, '_proposal_opening',
        lambda *a, **kw: pytest.fail('unconstrained buyer reached heuristic auction'))
    args = {**arguments(root), 'buyer': None}
    with pytest.raises(ValueError, match='qualified auction provider'):
        construct_history_family_proposal_v1(**args, rollout_policy_id='observer_safe_trick_control_v1')


@pytest.mark.parametrize('buyer,accepted', [(None, True), ('west', True), ('north', False)])
def test_history_follows_qualified_buyer_only_when_not_explicitly_constrained(root, monkeypatch, buyer, accepted):
    from baloot_platform.core.enums import Contract
    from baloot_platform.research import family_construction as module

    args = {**arguments(root), 'buyer': buyer}
    raw = args['family'].to_dict()
    raw['template']['expression'] = {'predicateId': 'PRED-SUN', 'kind': 'auction.contract',
        'version': 1, 'hardness': 'hard', 'arguments': {'contract': 'sun'}}
    args['family'] = replace(args['family'], _json=json.dumps(raw))
    requested = SimpleNamespace(constraints=args['family'].instantiate(args['bindings']),
        outcome=ScenarioExpressionV1.build(dict(predicateId='PRED-USER-ACE', kind='play.action_matches',
            version=1, hardness='hard', arguments={'trickIndex': 6, 'position': 0, 'selector': {'ranks': ['ace']}})))
    state = SimpleNamespace(contract=Contract.SUN, dealer=Seat.EAST, buyer=Seat.WEST,
        current_player=Seat.NORTH, initial_hands=[255 << (8 * seat) for seat in range(4)],
        projects=[], completed_tricks=[], purchase_card=Card.parse('DA'), hand=lambda actor: [])
    action = SimpleNamespace(card=Card.parse('S7'), ekka=False)
    engine = SimpleNamespace(legal_play_actions=lambda state: [action], play_card=lambda *a, **kw: None)
    opening = {'auctionActions': [], 'deckOrder': list(range(32))}
    receipt = {'status': 'complete', 'sha256': 'b' * 64, 'sourcePrefixHash': 'c' * 64,
        'executionReceipt': {'elapsedSeconds': 0},
        'auctionOutcome': {'allPassed': False, 'contract': 'sun', 'buyer': 'west',
                           'privateAuditDetail': 'not public progress'}}
    monkeypatch.setattr(module, '_qualified_family_opening_v1',
        lambda *a, **kw: (engine, state, opening, receipt))
    calls = []
    def construct(**kwargs):
        calls.append(kwargs)
        return {'status': 'proposal', 'qualified': False}
    monkeypatch.setattr(module, 'construct_family_proposal_v1', construct)
    result = construct_history_family_proposal_v1(**args,
        rollout_policy_id='observer_safe_trick_control_v1', auction_provider=lambda *a: None, scenario=requested)
    assert result['qualified'] is False
    assert result['auctionQualification']['contract'] == 'sun'
    assert result['auctionQualification']['buyer'] == 'west'
    assert result['auctionQualification']['allPassed'] is False
    assert 'privateAuditDetail' not in result['auctionQualification']
    if accepted:
        assert result['status'] == 'proposal'
        assert calls[0]['buyer'] == 'west'
        assert calls[0]['required_deck_order'] == opening['deckOrder']
        assert len(calls[0]['prefix']) == 20
        assert calls[0]['required_remaining_owners'] == {}
        assert calls[0]['scenario'] is requested
        assert result['auctionQualification']['receiptHash'] == receipt['sha256']
    else:
        assert result['status'] == 'seed_miss' and result['reason'] == 'opening_mismatch'
        assert not calls


@pytest.mark.parametrize('position', range(4))
@pytest.mark.parametrize('qualified', [False, True])
@pytest.mark.parametrize('policy_steps', [0, 24])
def test_qualified_history_preserves_free_tail_and_reuses_policy_prefix(root, monkeypatch, position, qualified, policy_steps):
    from baloot_platform.core.enums import Contract
    from baloot_platform.core.models import LegalPlayActionV1
    from baloot_platform.research import construction_pre_screen, family_construction as module

    args = arguments(root)
    args['bindings'].update(seat='east', playIndex=24 + position)
    state = SimpleNamespace(contract=Contract.SUN, dealer=Seat.EAST, buyer=Seat.NORTH,
        current_player=Seat((int(Seat.EAST) - position) % 4) if qualified and policy_steps else Seat.NORTH,
        initial_hands=[255 << (8 * seat) for seat in range(4)],
        projects=[], purchase_card=Card.parse('DA'))
    actions = [LegalPlayActionV1.from_code(code) for code in ('S7', 'H7')]
    engine = SimpleNamespace(legal_play_actions=lambda state: actions, play_card=lambda *a, **kw: None)
    opening = {'auctionActions': [], 'deckOrder': list(range(32))}
    receipt = {'status': 'complete', 'sha256': 'b' * 64, 'sourcePrefixHash': 'c' * 64,
        'executionReceipt': {'elapsedSeconds': 0}}
    monkeypatch.setattr(construction_pre_screen, '_proposal_opening', lambda *a, **kw: (engine, state, opening))
    monkeypatch.setattr(module, '_qualified_family_opening_v1', lambda *a, **kw: (engine, state, opening, receipt))
    monkeypatch.setattr(module, '_fixed_deal_project_status_v1', lambda *a, **kw: None)
    preferences = []
    def score(*unused, **kwargs):
        preferences.append(kwargs.get('preferred_completed_trick_winner'))
        return 'S7', {'S7': 1, 'H7': 1}
    monkeypatch.setattr(construction_pre_screen, '_proposal_policy_score', score)
    native_calls = []
    def construct(**kwargs):
        native_calls.append(kwargs)
        return {'status': 'proposal', 'qualified': False}
    monkeypatch.setattr(module, 'construct_family_proposal_v1', construct)

    result = construct_history_family_proposal_v1(**args, rollout_policy_id='observer_safe_trick_control_v1',
        auction_provider=(lambda *a: None) if qualified else None, policy_prefix_steps=policy_steps)
    length = (24 if policy_steps else 20) if qualified else args['bindings']['playIndex']
    assert preferences == [None] * length
    if qualified:
        assert result['status'] == 'proposal'
        assert len(native_calls) == 1 and len(native_calls[0]['prefix']) == length
        assert native_calls[0]['required_deck_order'] == opening['deckOrder']
        assert native_calls[0]['auction_actions'] == opening['auctionActions']
        assert native_calls[0]['required_remaining_owners'] == {}
        assert native_calls[0]['bindings']['seat'] == 'east'
        assert native_calls[0]['bindings']['playIndex'] == 24 + position
        assert native_calls[0].get('policy_prefix_steps', 0) == policy_steps
        if policy_steps:
            assert native_calls[0]['policy_id'] == 'observer_safe_trick_control_v1'
            native_calls.clear()
            state.current_player = state.current_player.next()
            rejected = construct_history_family_proposal_v1(**args, rollout_policy_id='observer_safe_trick_control_v1',
                auction_provider=lambda *a: None, policy_prefix_steps=policy_steps)
            assert rejected['status'] == 'seed_miss' and rejected['reason'] == 'decision_actor_mismatch'
            assert not native_calls
            if position == 0:
                preferences.clear()
                cancelled = [False]
                def cancel_on_last_play(*unused, **kwargs):
                    cancelled[0] = len(preferences) == 24
                engine.play_card = cancel_on_last_play
                stopped = construct_history_family_proposal_v1(**args, rollout_policy_id='observer_safe_trick_control_v1',
                    auction_provider=lambda *a: None, policy_prefix_steps=policy_steps,
                    cancel_check=lambda: cancelled[0])
                assert stopped['status'] == 'incomplete' and stopped['reason'] == 'cancelled_or_budget'
                assert not native_calls
    else:
        assert not native_calls
        assert result['status'] == 'seed_miss' and result['reason'] == 'decision_actor_mismatch'
    assert result['qualified'] is False


@pytest.mark.real_clock
@pytest.mark.parametrize('operator', ['any', 'not', 'count', 'prefer', 'report'])
def test_project_precheck_never_enforces_optional_or_soft_clauses(root, monkeypatch, operator):
    from baloot_platform.research import family_construction as module, ortools_project_fragment
    leaf = dict(predicateId='PRED-OPTIONAL-PROJECT', kind='project.count', version=1,
        hardness=operator if operator in {'prefer', 'report'} else 'hard',
        arguments={'minimum': 1, 'maximum': 1, 'state': 'declared'})
    raw = leaf if operator in {'prefer', 'report'} else dict(op=operator, children=[leaf],
        **({'minimum': 0, 'maximum': 1} if operator == 'count' else {}))
    expression = ScenarioExpressionV1.build(dict(op='all', children=[raw]))
    monkeypatch.setattr(ortools_project_fragment, 'attach_project_lifecycle_fragment_v1',
                        lambda *a, **kw: pytest.fail('optional project reached native model'))
    assert module._fixed_deal_project_status_v1(expression, rules=None, opening=None,
        buyer=None, contract='sun', trump=None, deadline=0, maximum_conflicts=100, seed=1, cancel_check=None) is None


@pytest.mark.parametrize('status,expected', [('INFEASIBLE', 'seed_miss'), ('UNKNOWN', 'incomplete')])
@pytest.mark.parametrize('buyer', ['north', None])
def test_fixed_deal_project_failure_stops_before_rollout_or_full_model(root, monkeypatch, status, expected, buyer):
    from ortools.sat.python import cp_model
    from baloot_platform.research import family_construction as module
    from baloot_platform.core.enums import Contract
    args = {**arguments(root), 'buyer': buyer}
    state = SimpleNamespace(contract=Contract.SUN, dealer=Seat.EAST, buyer=Seat.NORTH)
    receipt = dict(status='complete', sha256='a' * 64, sourcePrefixHash='b' * 64,
                   executionReceipt={'elapsedSeconds': 0})
    auctions, checks = [], []
    def opening(*args, **kwargs):
        auctions.append(True)
        return None, state, {'deckOrder': list(range(32))}, receipt
    def precheck(*args, **kwargs):
        checks.append(kwargs)
        return getattr(cp_model, status)
    monkeypatch.setattr(module, '_qualified_family_opening_v1', opening)
    monkeypatch.setattr(module, '_fixed_deal_project_status_v1', precheck)
    monkeypatch.setattr(module, 'construct_family_proposal_v1', lambda **kw: pytest.fail('rejected project reached full model'))
    result = construct_history_family_proposal_v1(**args, rollout_policy_id='observer_safe_trick_control_v1',
                                                auction_provider=object())
    assert result['status'] == expected and result['qualified'] is False
    assert result['reason'] == 'fixed_deal_projects_' + ('infeasible' if status == 'INFEASIBLE' else 'incomplete')
    assert len(checks) == 1 and checks[0]['buyer'] == 'north'
    if buyer is None:
        assert auctions == [True] and result['auctionQualification']['receiptHash'] == receipt['sha256']
    else:
        assert not auctions and 'auctionQualification' not in result


def test_native_fixed_project_miss_avoids_auction_proof(root, monkeypatch):
    import time
    from baloot_platform.research import family_construction as module

    args = arguments(root)
    args['family'], = ResearchFamilyCatalogV1.load(
        root / 'configs/research/public_project_rank_inference.yaml').families
    args.update(rules=RuleBook.load(root / 'configs/rules/saudi_federation_2023_v8.yaml'),
                seed=209941437, dealer='west', buyer='north')
    monkeypatch.setattr(module, '_qualified_family_opening_v1',
        lambda *a, **kw: pytest.fail('native-infeasible deal reached auction proof'))
    monkeypatch.setattr(module, 'construct_family_proposal_v1',
        lambda **kw: pytest.fail('native-infeasible deal reached full model'))
    started = time.perf_counter()
    result = construct_history_family_proposal_v1(**args, rollout_policy_id='observer_safe_trick_control_v1',
                                                auction_provider=object())
    assert result['status'] == 'seed_miss' and result['reason'] == 'fixed_deal_projects_infeasible'
    assert result['qualified'] is False and 'auctionQualification' not in result
    print(f'native project precheck before auction: {time.perf_counter() - started:.6f}s')


@pytest.mark.parametrize('case,expected', [
    ('select', 'hearts'), ('already', 'spades'), ('no_suit', None), ('configured', None),
    ('explicit_target', 'spades'), ('literal', 'spades'), ('authored', None), ('extra', 'hearts'),
    ('any', 'spades'), ('not', 'spades'), ('count', 'spades'),
])
def test_query_suit_selection_preserves_authored_constraints_and_native_ownership(root, monkeypatch, case, expected):
    from baloot_platform.core.enums import Contract
    from baloot_platform.research import family_construction as module

    args = arguments(root)
    args['family'] = next(row for row in ResearchFamilyCatalogV1.load(root / 'configs/research/families.yaml').families
                          if row.family_id == 'stopper_guard')
    raw = args['family'].to_dict()
    leaf = raw['template']['expression']['children'][1]
    if case == 'literal':
        leaf['arguments']['suit'] = 'spades'
    elif case in {'any', 'not', 'count'}:
        raw['template']['expression']['children'][1] = {'op': case, 'children': [leaf],
            **({'minimum': 0, 'maximum': 1} if case == 'count' else {})}
    if case == 'configured':
        raw['parameters']['suits'] = ['spades', 'clubs']
        args['family'] = replace(args['family'], parameters=replace(args['family'].parameters, suits=('spades', 'clubs')))
    args['family'] = replace(args['family'], _json=json.dumps(raw))
    original = dict(args['bindings'])
    requested = None
    if case == 'authored':
        frozen = args['family'].instantiate(original)
        requested = SimpleNamespace(constraints=frozen, outcome=frozen)
    elif case == 'extra':
        requested = SimpleNamespace(constraints=ScenarioExpressionV1.build({
            'predicateId': 'PRED-USER-HEARTS', 'kind': 'deal.remaining_suit_count', 'version': 1,
            'hardness': 'hard', 'arguments': {'seat': 'north', 'suit': 'hearts', 'minimum': 2, 'maximum': 2}}),
            outcome=args['family'].instantiate(original).children[0])
    authored_hashes = (requested.constraints.sha256, requested.outcome.sha256) if requested else None
    cards = ('S7', 'S8', 'DK') if case == 'already' else ('S7', 'H8', 'DK') if case == 'no_suit' else ('H7', 'H8', 'DK')
    state = SimpleNamespace(contract=Contract.SUN, dealer=Seat.EAST, buyer=Seat.NORTH,
        current_player=Seat.NORTH, projects=[], purchase_card=Card.parse('DA'),
        hand=lambda seat: [Card.parse(card) for card in cards])
    action = SimpleNamespace(card=Card.parse('S7'), ekka=False)
    engine = SimpleNamespace(legal_play_actions=lambda state: [action], play_card=lambda *a, **kw: None)
    opening = {'auctionActions': [], 'deckOrder': list(range(32))}
    receipt = {'status': 'complete', 'sha256': 'b' * 64, 'sourcePrefixHash': 'c' * 64,
        'executionReceipt': {'elapsedSeconds': 0}}
    monkeypatch.setattr(module, '_qualified_family_opening_v1', lambda *a, **kw: (engine, state, opening, receipt))
    calls = []
    def construct(**kwargs):
        calls.append(kwargs)
        assert kwargs['scenario'] is requested if requested else 'scenario' not in kwargs
        expression = family_construction_expression_v1(kwargs['family'].instantiate(kwargs['bindings']), requested)
        if requested:
            assert {p.predicate_id for p in requested.constraints.predicates()} <= {
                p.predicate_id for p in expression.predicates()}
        return {'status': 'proposal', 'qualified': False, 'bindings': dict(kwargs['bindings']),
            'expressionHash': expression.sha256}
    monkeypatch.setattr(module, 'construct_family_proposal_v1', construct)
    result = construct_history_family_proposal_v1(**args, rollout_policy_id='observer_safe_trick_control_v1',
        auction_provider=lambda *a: None, scenario=requested, _select_query_suit=case != 'explicit_target')
    assert args['bindings'] == original
    if requested:
        assert (requested.constraints.sha256, requested.outcome.sha256) == authored_hashes
    assert result['qualified'] is False
    if expected is None:
        assert result['status'] == 'seed_miss' and result['reason'] == 'history_suit_count_mismatch'
        assert not calls  # No native construction model for a provably impossible seeded history.
    else:
        assert len(calls) == 1 and result['status'] == 'proposal'
        assert result['bindings'] == {**original, 'suit': expected}
        assert calls[0]['required_deck_order'] == opening['deckOrder']
        assert len(calls[0]['prefix']) == 20


@pytest.mark.real_clock
@pytest.mark.parametrize('case,count', [
    ('proposal', 2), ('exhausted', 3), ('incomplete', 1), ('deadline', 1),
    ('explicit', 1), ('authored', 1),
])
def test_query_retries_distinct_native_infeasible_bindings_within_one_opening_budget(root, monkeypatch, case, count):
    from baloot_platform.core.enums import Contract
    from baloot_platform.research import family_construction as module

    args = arguments(root)
    raw = args['family'].to_dict()
    raw['template']['expression'] = dict(predicateId='PRED-SINGLETON', kind='deal.remaining_suit_count',
        version=1, hardness='hard', arguments={'seat': {'$parameter': 'seat'},
            'suit': {'$parameter': 'suit'}, 'minimum': 1, 'maximum': 1})
    args['family'] = replace(args['family'], _json=json.dumps(raw))
    requested = None
    if case == 'authored':
        expression = args['family'].instantiate(args['bindings'])
        requested = SimpleNamespace(constraints=expression, outcome=expression)
    state = SimpleNamespace(contract=Contract.SUN, dealer=Seat.EAST, buyer=Seat.NORTH,
        current_player=Seat.NORTH, projects=[], purchase_card=Card.parse('DA'),
        hand=lambda seat: [Card.parse(card) for card in ('S7', 'H8', 'D9')])
    action = SimpleNamespace(card=Card.parse('S7'), ekka=False)
    engine = SimpleNamespace(legal_play_actions=lambda state: [action], play_card=lambda *a, **kw: None)
    opening = {'auctionActions': [], 'deckOrder': list(range(32))}
    receipt = {'status': 'complete', 'sha256': 'b' * 64, 'sourcePrefixHash': 'c' * 64,
        'executionReceipt': {'elapsedSeconds': 0}}
    openings = []
    def qualified(*unused, **kwargs):
        openings.append(True)
        return engine, state, opening, receipt
    monkeypatch.setattr(module, '_qualified_family_opening_v1', qualified)
    now = [0.0]
    monkeypatch.setattr(module.time, 'monotonic', lambda: now[0])
    calls = []
    def construct(**kwargs):
        calls.append(kwargs)
        now[0] += args['maximum_seconds'] if case == 'deadline' else 2
        status = ('proposal' if case == 'proposal' and len(calls) == 2 else
                  'incomplete' if case == 'incomplete' else 'infeasible')
        return {'status': status, 'qualified': False, 'bindings': dict(kwargs['bindings'])}
    monkeypatch.setattr(module, 'construct_family_proposal_v1', construct)
    result = construct_history_family_proposal_v1(**args, rollout_policy_id='observer_safe_trick_control_v1',
        auction_provider=lambda *a: None, scenario=requested, _select_query_suit=case != 'explicit')
    assert len(openings) == 1 and len(calls) == count
    assert [call['bindings']['suit'] for call in calls] == ['spades', 'hearts', 'diamonds'][:count]
    assert [call['maximum_seconds'] for call in calls] == [args['maximum_seconds'] - 2 * index for index in range(count)]
    assert all(call['prefix'] == calls[0]['prefix'] and len(call['prefix']) == 20
        and call['required_deck_order'] == opening['deckOrder']
        and call['auction_actions'] == opening['auctionActions']
        and call['required_remaining_owners'] == {} for call in calls)
    assert result['qualified'] is False
    if case == 'proposal':
        assert result['status'] == 'proposal' and result['bindings']['suit'] == 'hearts'
    elif case in {'incomplete', 'deadline'}:
        assert result['status'] == 'incomplete'
    else:
        assert result['status'] == 'seed_miss' and result['reason'] == 'history_expression_infeasible'


def test_automatic_preselected_auction_does_not_invent_player_decisions(root):
    from baloot_platform.research.family_construction import _qualified_family_opening_v1
    data = arguments(root)['rules'].data
    data['contracts'].update(allowed=['sun'], default='sun', require_preselected=True)
    data['bidding']['first_round'].update(allowed=['pass', 'sun'], hokum_uses_purchase_suit=False)
    data['bidding']['second_round'].update(allowed=['pass', 'sun'], hokum_must_differ_from_purchase_suit=False)
    rules = RuleBook(data)
    def auction(initial, **kwargs):
        return {'status': 'complete', 'operation': 'generate', 'qualificationAuthority': 'cp_sat',
            'rulesHash': rules.sha256, 'coveredFromBidIndex': 0, 'partialReceipt': None,
            'sourcePrefixHash': initial.sha256, 'decisions': [], 'finalPrefix': initial.to_dict()}
    _, state, prefix, _ = _qualified_family_opening_v1(rules, 4, 'east', auction, lambda: 10, None)
    assert state.buyer == Seat.NORTH and prefix['auctionActions'] == []


@pytest.mark.real_clock
def test_history_deadline_expiring_while_preparing_child_arguments_is_incomplete(root, monkeypatch):
    from baloot_platform.research import family_construction as module, construction_pre_screen
    from baloot_platform.research import negative_project_prefix
    from baloot_platform.core.enums import Contract
    now = [0]
    monkeypatch.setattr(module.time, 'monotonic', lambda: now[0])
    args = arguments(root)
    def hand(actor):
        now[0] = args['maximum_seconds']
        return [Card.parse('S7')]
    # Isolate the child budget handoff; rule legality is covered by the real seeded test.
    state = SimpleNamespace(contract=Contract.SUN, dealer=Seat.EAST, buyer=Seat.NORTH,
        current_player=Seat.NORTH, initial_hands=[255 << (8 * seat) for seat in range(4)],
        projects=[], completed_tricks=[], purchase_card=Card.parse('DA'), hand=hand)
    action = SimpleNamespace(card=Card.parse('S7'), ekka=False)
    engine = SimpleNamespace(legal_play_actions=lambda state: [action], play_card=lambda *a, **kw: None)
    monkeypatch.setattr(construction_pre_screen, '_proposal_opening',
        lambda *a, **kw: (engine, state, {'auctionActions': []}))
    monkeypatch.setattr(negative_project_prefix, 'build_negative_project_prefix_v1',
        lambda **kw: pytest.fail('expired argument preparation launched native construction'))
    result = construct_history_family_proposal_v1(**args, rollout_policy_id='observer_safe_trick_control_v1')
    assert result['status'] == 'incomplete' and result['solverStatus'] == 'unknown'


def test_history_uses_requested_dealer_and_stops_declared_absence_miss_after_first_trick(root, monkeypatch):
    from baloot_platform.research import construction_pre_screen, negative_project_prefix
    from baloot_platform.core.enums import Contract

    args = arguments(root)
    project = SimpleNamespace(owner=Seat.WEST, declared=False)
    state = SimpleNamespace(contract=Contract.SUN, dealer=Seat.EAST, buyer=Seat.NORTH,
        current_player=Seat.NORTH, initial_hands=[255 << (8 * seat) for seat in range(4)],
        projects=[project])
    action = SimpleNamespace(card=Card.parse('S7'), ekka=False)
    plays, selected_dealers = [], []
    def play(*unused, **kwargs):
        plays.append(action)
        project.declared = len(plays) >= 4
    engine = SimpleNamespace(legal_play_actions=lambda state: [action], play_card=play)
    def opening(*unused, dealer):
        selected_dealers.append(dealer)
        return engine, state, {'auctionActions': []}
    monkeypatch.setattr(construction_pre_screen, '_proposal_opening', opening)
    monkeypatch.setattr(negative_project_prefix, 'build_negative_project_prefix_v1',
        lambda **kw: pytest.fail('declared absence miss launched a native model'))

    result = construct_history_family_proposal_v1(**args,
        rollout_policy_id='observer_safe_trick_control_v1')
    assert selected_dealers == [args['dealer']]
    assert len(plays) == 4
    assert result['status'] == 'seed_miss' and result['reason'] == 'negative_project_mismatch'
    assert not result['qualified']


@pytest.mark.parametrize('optional', [False, True])
def test_fixed_observer_suit_miss_skips_native_models_only_for_mandatory_predicate(root, monkeypatch, optional):
    from baloot_platform.research import construction_pre_screen, negative_project_prefix
    from baloot_platform.core.enums import Contract

    args = arguments(root)
    args['family'] = next(family for family in
        ResearchFamilyCatalogV1.load(root / 'configs/research/families.yaml').families
        if family.family_id == 'stopper_guard')
    if optional:
        raw = args['family'].to_dict()
        raw['template']['expression']['children'][1] = {'op': 'count', 'minimum': 0, 'maximum': 1,
            'children': [raw['template']['expression']['children'][1]]}
        args['family'] = replace(args['family'], _json=json.dumps(raw))
    state = SimpleNamespace(contract=Contract.SUN, dealer=Seat.EAST, buyer=Seat.NORTH,
        current_player=Seat.NORTH, initial_hands=[255 << (8 * seat) for seat in range(4)],
        projects=[], completed_tricks=[], purchase_card=Card.parse('DA'), hand=lambda actor: [Card.parse('H7')])
    action = SimpleNamespace(card=Card.parse('S7'), ekka=False)
    engine = SimpleNamespace(legal_play_actions=lambda state: [action], play_card=lambda *a, **kw: None)
    monkeypatch.setattr(construction_pre_screen, '_proposal_opening',
        lambda *a, **kw: (engine, state, {'auctionActions': []}))
    def sample(**kwargs):
        raise RuntimeError('native reallocation reached')
    monkeypatch.setattr(negative_project_prefix, 'build_negative_project_prefix_v1', sample)
    if optional:
        with pytest.raises(RuntimeError, match='native reallocation reached'):
            construct_history_family_proposal_v1(**args, rollout_policy_id='observer_safe_trick_control_v1')
    else:
        result = construct_history_family_proposal_v1(**args, rollout_policy_id='observer_safe_trick_control_v1')
        assert result['status'] == 'seed_miss' and result['reason'] == 'observer_suit_count_mismatch'
        assert not result['qualified']


@pytest.mark.parametrize('optional', ['any', 'not', 'count'])
def test_only_mandatory_absence_is_used_as_a_cheap_seed_filter(root, optional):
    args = arguments(root)
    raw = args['family'].to_dict()
    child = raw['template']['expression']['children'][1]
    raw['template']['expression']['children'][1] = {
        'op': optional, 'children': [child],
        **({'minimum': 0, 'maximum': 1} if optional == 'count' else {})}
    args['family'] = replace(args['family'], _json=json.dumps(raw))
    from baloot_platform.research.family_construction import history_family_absences_v1
    assert history_family_absences_v1(args['family'].instantiate(args['bindings']),
        contract='sun', trump=None) == ()


def test_cross_suit_guard_draft_includes_audited_shapes_not_same_suit_choices(root):
    family, = ResearchFamilyCatalogV1.load(root / 'configs/research/cross_suit_guard.yaml').families
    examples = json.loads((root / 'tests/fixtures/audit5_public_baseline.json').read_text())
    expected = {728948: True, 729156: True, 731186: True, 731276: False, 731707: False}
    for row in examples:
        if row['seed'] not in expected:
            continue
        bindings = {key: values[0] for key, values in family.parameters.choices(family.contracts).items()}
        hand = [Card.parse(code) for code in row['hand']]
        bindings.update(seat=row['actor'], playIndex=row['index'], suit=hand[0].suit.name.lower())
        expression = family.instantiate(bindings)
        validate_family_construction_expression_v1(expression)
        count = next(p.arguments for p in expression.predicates() if p.kind == 'deal.remaining_suit_count')
        matching = sum(card.suit == hand[0].suit for card in hand)
        assert (count['minimum'] <= matching <= count['maximum']) == expected[row['seed']]


@pytest.mark.parametrize('family_id', ['negative-project', 'stopper_guard', 'cross_suit_guard'])
def test_seeded_history_native_construction_without_fixture(root, family_id, deterministic_work):
    import time
    from baloot_platform.research.construction_pre_screen import _proposal_opening, _proposal_policy_score
    from baloot_platform.core.models import LegalPlayActionV1
    args = arguments(root)
    if family_id == 'cross_suit_guard':
        args['family'], = ResearchFamilyCatalogV1.load(root / 'configs/research/cross_suit_guard.yaml').families
    else:
        args['family'] = next(family for family in
            ResearchFamilyCatalogV1.load(root / 'configs/research/families.yaml').families
            if family.family_id == family_id)
    # Discover a matching seeded public history, not a saved golden deal.
    args['rules'] = RuleBook.load(root / 'configs/rules/saudi_federation_2023_v8.yaml')
    for seed in range(100):
        engine, state, opening = _proposal_opening(args['rules'], seed)
        if opening is None or state.contract.value != 'sun':
            continue
        deadline = deterministic_work.read("monotonic") + 10
        for _ in range(args['bindings']['playIndex']):
            legal = engine.legal_play_actions(state)
            action = legal[0] if len(legal) == 1 else LegalPlayActionV1.from_code(
                _proposal_policy_score(args['rules'], engine, state,
                    'observer_safe_trick_control_v1', deadline, lambda: None)[0])
            engine.play_card(state, action.card, ekka=action.ekka)
        if family_id == 'negative-project':
            if any(p.declared and p.owner.team.name.lower() == 'west_east' for p in state.projects):
                continue
        else:
            hand = state.hand(state.current_player)
            if len({card.suit for card in hand}) != (2 if family_id == 'cross_suit_guard' else 1):
                continue
            args['bindings']['suit'] = hand[0].suit.name.lower()
        args.update(seed=seed, dealer=state.dealer.name.lower(), buyer=state.buyer.name.lower())
        args['bindings']['seat'] = state.current_player.name.lower()
        break
    else:
        pytest.fail('no seeded Sun negative-project public history')
    result = construct_history_family_proposal_v1(**args,
        rollout_policy_id='observer_safe_trick_control_v1')
    assert result['status'] == 'proposal', result
    assert result['reallocatedHistoryPolicyAudited'] is False
    assert not result['qualified']
    witness = family_proposal_witness_v1(result, rules=args['rules'], interest_policy={})
    normalized = OrToolsExactStateNormalizerV1(args['rules'], native_cp_sat=True).normalize(
        witness['prefix'], decision_point={'kind': 'play'}, maximum_seconds=30)
    normalized.require_integrity()
    assert normalized.next_actor == args['bindings']['seat']
    if family_id == 'stopper_guard':
        assert all(Card.parse(card).suit.name.lower() == args['bindings']['suit']
                   for card in normalized.state['hands'][normalized.next_actor])
    elif family_id == 'cross_suit_guard':
        assert len({Card.parse(card).suit for card in normalized.state['hands'][normalized.next_actor]}) == 2
        assert len(normalized.legal_actions) >= 2
        margin, = (row for row in result['predicateEvidence'] if row['kind'] == 'outcome.round_margin')
        assert margin['satisfied'] is True
        assert -10 <= margin['observed']['value'] <= 10


def test_unsupported_nested_family_predicate_fails_before_solve(root):
    args = arguments(root)
    raw = args['family'].to_dict()
    raw['template']['expression']['children'].append({
        'predicateId': 'PRED-COUNT', 'kind': 'proof.winning_root_action_count',
        'version': 1, 'hardness': 'hard', 'arguments': {'minimum': 1, 'maximum': 1}})
    args['family'] = replace(args['family'], _json=json.dumps(raw))
    with pytest.raises(ValueError):
        construct_family_proposal_v1(**args)


@pytest.mark.parametrize('steps', [-1, True, 25])
def test_family_policy_cannot_cover_the_featured_decision(root, monkeypatch, steps):
    from baloot_platform.research import family_construction as module
    monkeypatch.setattr(module, 'OrToolsScenarioModelV1', lambda **kw: pytest.fail('invalid policy launched a model'))
    with pytest.raises(ValueError, match='policy prefix'):
        construct_family_proposal_v1(**arguments(root), policy_prefix_steps=steps)
    with pytest.raises(ValueError, match='policy prefix'):
        construct_history_family_proposal_v1(**arguments(root), policy_prefix_steps=steps,
            rollout_policy_id='observer_safe_trick_control_v1')


@pytest.mark.real_clock
@pytest.mark.parametrize('prefix_length', [20, 24])
def test_native_round_family_constrains_new_lead_in_to_the_same_visible_policy(root, prefix_length):
    import time
    from baloot_platform.research.construction_pre_screen import _proposal_opening, _proposal_policy_score
    from baloot_platform.core.models import LegalPlayActionV1

    args = arguments(root)
    args['rules'] = RuleBook.load(root / 'configs/rules/saudi_federation_2023_v8.yaml')
    for seed in range(100):
        engine, state, opening = _proposal_opening(args['rules'], seed, dealer='east')
        if opening is not None and state.contract.value == 'sun':
            break
    else:
        pytest.fail('no seeded Sun opening')
    prefix = []
    for _ in range(24):
        legal = engine.legal_play_actions(state)
        selected = legal[0] if len(legal) == 1 else LegalPlayActionV1.from_code(_proposal_policy_score(
            args['rules'], engine, state, 'observer_safe_trick_control_v1', time.monotonic() + 10, lambda: None)[0])
        prefix.append(TailPrefixActionV1(state.current_player, selected.card, selected.ekka))
        engine.play_card(state, selected.card, ekka=selected.ekka)
    raw = args['family'].to_dict()
    raw['template']['expression'] = dict(predicateId='PRED-SUN', kind='auction.contract',
        version=1, hardness='hard', arguments={'contract': 'sun'})
    args['family'] = replace(args['family'], _json=json.dumps(raw))
    args['bindings'].update(seat=state.current_player.name.lower(), playIndex=24)
    preferred, _ = _proposal_policy_score(args['rules'], engine, state,
        'observer_safe_trick_control_v1', time.monotonic() + 10, lambda: None)
    different = next(action.card for action in engine.legal_play_actions(state) if action.code != preferred)
    args['scenario'] = SimpleNamespace(constraints=ScenarioExpressionV1.build(dict(
        predicateId='PRED-EXACT-TAIL', kind='play.action_matches', version=1, hardness='hard',
        arguments={'trickIndex': 6, 'position': 0, 'selector': {'cards': [different.id]}})),
        outcome=args['family'].instantiate(args['bindings']))
    args.update(seed=seed, buyer=state.buyer.name.lower(), prefix=prefix[:prefix_length],
        required_deck_order=opening['deckOrder'], auction_actions=opening['auctionActions'],
        purchase_card=state.purchase_card, policy_prefix_steps=24, policy_id='observer_safe_trick_control_v1')
    result = construct_family_proposal_v1(**args)
    assert result['status'] == 'proposal', result
    assert result['qualified'] is False
    assert result['deckOrder'] == opening['deckOrder']
    assert result['decisionPrefix']['playActions'] == [play.to_dict() for play in prefix]
    assert len(result['completePrefix']['playActions']) == 32
    assert result['completePrefix']['playActions'][24]['card'] == different.code


@pytest.mark.parametrize('kind', ['play.action_matches', 'play.trick_winner_matches'])
def test_unsafe_native_family_complement_fails_before_model_creation(root, monkeypatch, kind):
    from baloot_platform.research import family_construction as module

    args = arguments(root)
    raw = args['family'].to_dict()
    leaf = {'predicateId': 'PRED-UNSAFE', 'kind': kind, 'version': 1, 'hardness': 'hard',
            'arguments': {'trickIndex': 6, 'selector': {'ranks': ['ace']},
                          **({'position': 0} if kind == 'play.action_matches' else {})}}
    raw['template']['expression'] = {'op': 'any', 'children': [
        {'op': 'not', 'children': [{'op': 'count', 'minimum': 0, 'maximum': 1, 'children': [leaf]}]}]}
    args['family'] = replace(args['family'], _json=json.dumps(raw))
    monkeypatch.setattr(module, 'OrToolsScenarioModelV1',
                        lambda **kwargs: pytest.fail('unsafe complement reached model construction'))
    with pytest.raises(ValueError, match='safe complement'):
        construct_family_proposal_v1(**args)


def test_native_family_nested_expression_matches_complete_finite_model_truth_table(root):
    from ortools.sat.python import cp_model
    from baloot_platform.research.ortools_scenario_model import reify_scenario_expression
    from baloot_platform.research.scenario_predicate_manifest_v2 import shared_scenario_predicate_catalog_lowering_manifest_v2

    def predicate(predicate_id, kind, **arguments):
        return {'predicateId': predicate_id, 'kind': kind, 'version': 1,
                'hardness': 'hard', 'arguments': {'trickIndex': 6, **arguments}}

    suit, rank, seat = {'$parameter': 'suit'}, {'$parameter': 'rank'}, {'$parameter': 'seat'}
    raw = ResearchFamilyCatalogV1.load(root / 'configs/research/families.yaml').to_dict()
    raw.pop('sha256')
    raw['families'] = raw['families'][:1]
    raw['families'][0]['enabled'] = True
    raw['families'][0]['template']['expression'] = {'op': 'all', 'children': [
        {'op': 'count', 'minimum': 1, 'maximum': 1, 'children': [
            predicate('PRED-ACTION', 'play.action_matches', position=0, actingSeat=seat,
                selector={'suits': [suit], 'ranks': [rank], 'ekka': 'required', 'owner': {'relation': 'actor'}}),
            predicate('PRED-WINNER', 'play.trick_winner_matches', team='north_south',
                selector={'ranks': [rank], 'owner': {'relation': 'seat', 'seat': seat}}),
        ]},
        {'op': 'any', 'children': [
            {'op': 'not', 'children': [predicate('PRED-ACE-COUNT', 'play.trick_card_count',
                minimum=1, maximum=1, selector={'ranks': [rank], 'suits': [suit], 'ekka': 'forbidden',
                                               'owner': {'relation': 'team', 'team': 'north_south'}})]},
            predicate('PRED-SUIT-COUNT', 'play.trick_card_count', minimum=1, maximum=1,
                      selector={'suits': [suit]}),
        ]},
    ]}
    family = ResearchFamilyCatalogV1.build(raw).families[0]
    selected = {key: values[0] for key, values in family.parameters.choices(family.contracts).items()}
    expression = family.instantiate({**selected, 'suit': 'hearts', 'rank': 'ace', 'seat': 'north'})
    validate_family_construction_expression_v1(expression)

    model = cp_model.CpModel()
    card = model.new_int_var_from_domain(cp_model.Domain.from_values([Card.parse('SA').id, Card.parse('HA').id]), 'card')
    actor = model.new_int_var(0, 1, 'actor')
    owner = model.new_int_var(0, 1, 'owner')
    ekka, present = model.new_bool_var('ekka'), model.new_bool_var('present')
    first = CpSatActionVariablesV1('T6-FIRST', card, actor, owner, ekka, present)
    actions = [first]
    for code, acting_seat in [('HK', Seat.WEST), ('CA', Seat.SOUTH), ('D7', Seat.EAST)]:
        actions.append(CpSatActionVariablesV1(f'T6-{code}', model.new_constant(Card.parse(code).id),
            model.new_constant(int(acting_seat)), model.new_constant(int(acting_seat)),
            model.new_constant(0), model.new_constant(1)))
    context = CpSatPredicateModelContextV1(model, 'family-finite-parity',
        {6: CpSatTrickVariablesV1(6, tuple(actions), first)}, (), model.new_constant(1))
    manifest = shared_scenario_predicate_catalog_lowering_manifest_v2()
    lowerings = {p.predicate_id: manifest.lower_predicate(p, context) for p in expression.predicates()}
    target = reify_scenario_expression(model, expression, {key: row.literal for key, row in lowerings.items()})

    class TruthTable(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()
            self.rows = {}

        def on_solution_callback(self):
            values = tuple(self.value(variable) for variable in (card, actor, owner, ekka, present))
            card_id, actor_id, owner_id, is_ekka, is_present = values
            hearts = card_id == Card.parse('HA').id
            expected = {
                'PRED-ACTION': bool(is_present and hearts and actor_id == 0 and owner_id == actor_id and is_ekka),
                'PRED-WINNER': bool(is_present and actor_id == 0 and owner_id == 0),
                'PRED-ACE-COUNT': bool(is_present and hearts and not is_ekka and owner_id == 0),
                'PRED-SUIT-COUNT': not bool(is_present and hearts),  # The fixed HK always matches.
            }
            assert {key: self.boolean_value(row.literal) for key, row in lowerings.items()} == expected
            oracle = (int(expected['PRED-ACTION']) + int(expected['PRED-WINNER']) == 1
                      and (not expected['PRED-ACE-COUNT'] or expected['PRED-SUIT-COUNT']))
            assert self.boolean_value(target) == oracle
            self.rows[values] = oracle

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.max_time_in_seconds = 10
    truth = TruthTable()
    assert solver.solve(model, truth) == cp_model.OPTIMAL
    assert len(truth.rows) == 32
    assert set(truth.rows.values()) == {False, True}
    assert all(row.extract(solver)['kind'].startswith('play.') for row in lowerings.values())


@pytest.mark.parametrize('contract', ['sun', 'hokum'])
@pytest.mark.parametrize('conflict', [None, 'prefix', 'anchor'])
def test_fixed_deck_specializes_all_original_owners_without_erasing_conflicts(root, monkeypatch, contract, conflict):
    from ortools.sat.python import cp_model
    from baloot_platform.core.dealing import DealLayoutV1
    from baloot_platform.research import family_construction as module

    args = arguments(root)
    raw = args['family'].to_dict()
    raw['contracts'] = [contract]
    raw['template']['expression'] = dict(predicateId='PRED-CONTRACT', kind='auction.contract',
        version=1, hardness='hard', arguments={'contract': {'$parameter': 'contract'}})
    args['family'] = replace(args['family'], contracts=(contract,), _json=json.dumps(raw))
    args['bindings']['contract'] = contract
    args.update(dealer='east' if contract == 'sun' else 'south', buyer='north',
        trump=None if contract == 'sun' else 'spades', required_deck_order=list(reversed(range(32))))
    layout = DealLayoutV1.build(args['rules'].structural_domain,
        dealer=Seat.parse(args['dealer']), buyer=Seat.NORTH)
    expected = [None] * 32
    for seat in Seat:
        for position in layout.positions_for(seat, final=True):
            expected[args['required_deck_order'][position]] = seat
    assert expected[args['required_deck_order'][layout.purchase_position]] == Seat.NORTH
    wrong_owner = expected[0].next()
    if conflict == 'prefix':
        args['prefix'] = (TailPrefixActionV1.build(wrong_owner, 'S7'),)
    elif conflict == 'anchor':
        args['required_remaining_owners'] = {'S7': wrong_owner}
    model = cp_model.CpModel()
    owners = tuple(model.new_int_var(0, 3, f'owner_{card}') for card in range(32))
    positions = tuple(model.new_int_var(0, 31, f'position_{index}') for index in range(32))

    def continuation(**kwargs):
        assert kwargs['fixed_card_owners'] == tuple(expected)
        assert kwargs['prefix'] == args.get('prefix', ())
        for card, owner in enumerate(kwargs['fixed_card_owners']):
            model.add(owners[card] == int(owner))
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        status = solver.solve(model)
        assert status == (cp_model.INFEASIBLE if conflict else cp_model.OPTIMAL)
        if not conflict:
            assert [solver.value(variable) for variable in positions] == args['required_deck_order']
        raise PlayTailSolveStopped('infeasible' if conflict else 'unknown')

    scenario = SimpleNamespace(model=model, owner_of_card=owners, card_at_position=positions,
        add_native_auction_fragment=lambda **kwargs: SimpleNamespace(trump=model.new_int_var(0, 3, 'trump')),
        add_native_round_continuation=continuation)
    def scenario_model(**kwargs):
        assert kwargs['canonicalize_deal_packets'] is False
        return scenario
    monkeypatch.setattr(module, 'OrToolsScenarioModelV1', scenario_model)
    result = construct_family_proposal_v1(**args)
    assert result['status'] == ('infeasible' if conflict else 'incomplete')
    assert not result['qualified']


def test_prefix_and_motif_specialize_ownership_before_round_lowering(root, monkeypatch):
    from ortools.sat.python import cp_model
    from baloot_platform.research import family_construction as module

    model = cp_model.CpModel()
    owners = tuple(model.new_int_var(0, 3, f'owner_{card}') for card in range(32))
    prefix = (TailPrefixActionV1.build('north', 'S7'),)

    def continuation(**kwargs):
        fixed = kwargs['fixed_card_owners']
        assert fixed[Card.parse('S7').id] == Seat.NORTH
        assert fixed[Card.parse('DA').id] == Seat.EAST
        assert sum(owner is not None for owner in fixed) == 2
        assert kwargs['prefix'] == prefix
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        assert solver.solve(model) == cp_model.OPTIMAL
        assert solver.value(owners[Card.parse('S7').id]) == int(Seat.NORTH)
        assert solver.value(owners[Card.parse('DA').id]) == int(Seat.EAST)
        raise PlayTailSolveStopped('unknown')

    scenario = SimpleNamespace(model=model, owner_of_card=owners,
        add_native_auction_fragment=lambda **kwargs: None,
        add_native_round_continuation=continuation)
    monkeypatch.setattr(module, 'OrToolsScenarioModelV1', lambda **kwargs: scenario)
    result = construct_family_proposal_v1(**arguments(root), prefix=prefix,
        required_remaining_owners={'DA': 'east'})
    assert result['status'] == 'incomplete'
    assert result['solverStatus'] == 'unknown'
    assert not result['qualified']


@pytest.mark.parametrize('duplicate', [False, True])
def test_rejects_inconsistent_prefix_motif_before_lowering(root, monkeypatch, duplicate):
    from baloot_platform.research import family_construction as module

    def unexpected(**kwargs):
        pytest.fail('invalid prefix reached model construction')

    monkeypatch.setattr(module, 'OrToolsScenarioModelV1', unexpected)
    play = TailPrefixActionV1.build('north', 'S7')
    with pytest.raises(ValueError, match='duplicate cards' if duplicate else 'already played'):
        construct_family_proposal_v1(**arguments(root),
            prefix=(play, play) if duplicate else (play,),
            required_remaining_owners={} if duplicate else {'S7': 'north'})


@pytest.mark.parametrize('prefix_length,requested_card', [
    (length, card) for card in (None, 'root', 'already_played') for length in (20, 27)
] + [(27, 'wrong_actor'), (20, 'legal_two'), (27, 'legal_two'), (27, 'legal_one'),
     (20, 'exclude_decision'), (27, 'exclude_decision'),
     (20, 'raw_points'), (27, 'raw_points'), (27, 'raw_old_prefix')])
def test_native_family_preserves_fixed_deck_and_prefix_under_finite_budget(root, prefix_length, requested_card):
    from baloot_platform.core.dealing import DealLayoutV1

    fixture = json.loads((root / 'tests/fixtures/research-native-qualified-round-v3.json').read_text())
    recorded = fixture['exactTracePrefix']
    args = arguments(root)
    # A known legal prefix makes this a semantic check, not a free-search speed gate.
    args.update(rules=RuleBook.load(root / 'configs/rules/saudi_federation_2023_v2.yaml'),
        family=next(family for family in ResearchFamilyCatalogV1.load(
            root / 'configs/research/families.yaml').families if family.family_id == 'positive-project'),
        buyer='west', required_deck_order=recorded['deckOrder'], auction_actions=recorded['auctionActions'])
    assert args['rules'].sha256 == fixture['rulesHash']
    args['bindings'].update(seat='east', playIndex=27)
    args['prefix'] = tuple(TailPrefixActionV1.build(row['actor'], Card.parse(row['card']),
        ekka=row['ekka']) for row in recorded['playActions'][:prefix_length])
    layout = DealLayoutV1.build(args['rules'].structural_domain, dealer=Seat.EAST, buyer=Seat.WEST)
    played = {Card.parse(row['card']).id for row in recorded['playActions'][:27]}
    args['required_remaining_owners'] = {} if prefix_length == 20 else {Card.parse(recorded['deckOrder'][position]).code: seat.name.lower()
        for seat in Seat for position in layout.positions_for(seat, final=True)
        if recorded['deckOrder'][position] not in played}
    if requested_card == 'wrong_actor':
        raw = args['family'].to_dict()
        raw['template']['expression'] = dict(predicateId='PRED-CONTRACT', kind='auction.contract',
            version=1, hardness='hard', arguments={'contract': 'sun'})
        args['family'] = replace(args['family'], _json=json.dumps(raw))
        args['bindings']['seat'] = 'north'
    elif requested_card in {'legal_two', 'legal_one'}:
        count = 2 if requested_card == 'legal_two' else 1
        args['scenario'] = SimpleNamespace(constraints=ScenarioExpressionV1.build(dict(
            predicateId='PRED-USER-CHOICES', kind='state.legal_action_count', version=1, hardness='hard',
            arguments={'minimum': count, 'maximum': count})),
            outcome=args['family'].instantiate(args['bindings']))
    elif requested_card in {'raw_points', 'raw_old_prefix'}:
        reference = OrToolsExactStateNormalizerV1(args['rules'], native_cp_sat=True).normalize(
            ExactTracePrefixV1.build({**recorded, 'playActions': recorded['playActions'][:27]}),
            decision_point={'kind': 'play'}, maximum_seconds=30)
        tricks = reference.state['completedTricks']
        raw = sum(t['rawPoints'] for t in tricks if Seat.parse(t['winner']).team == Team.NORTH_SOUTH)
        previous_raw = sum(t['rawPoints'] for t in tricks[:5] if Seat.parse(t['winner']).team == Team.NORTH_SOUTH)
        assert raw != previous_raw
        expected_raw = raw if requested_card == 'raw_points' else previous_raw
        args['scenario'] = SimpleNamespace(constraints=ScenarioExpressionV1.build(dict(
            predicateId='PRED-USER-RAW', kind='state.current_raw_points', version=1, hardness='hard',
            arguments={'team': 'north_south', 'minimum': expected_raw, 'maximum': expected_raw})),
            outcome=args['family'].instantiate(args['bindings']))
    elif requested_card not in {None, 'exclude_decision'}:
        card = Card.parse(recorded['playActions'][27 if requested_card == 'root' else 0]['card']).id
        args['scenario'] = SimpleNamespace(constraints=ScenarioExpressionV1.build(dict(
            predicateId='PRED-USER-ROOT', kind='play.action_matches', version=1, hardness='hard',
            arguments={'trickIndex': 6, 'position': 3, 'selector': {'cards': [card]}})),
            outcome=args['family'].instantiate(args['bindings']))
    result = construct_family_proposal_v1(**args)
    if requested_card in {'already_played', 'wrong_actor', 'legal_one', 'raw_old_prefix'}:
        assert result['status'] == 'infeasible' and result['qualified'] is False
        assert 'trace' not in result
        assert ('decision_actor' if requested_card == 'wrong_actor' else 'scenario_expression') in result['constructionInfeasibleCore']
        assert set(result['constructionInfeasibleCore']) <= {'decision_actor', 'scenario_expression'}
        return
    assert result['status'] == 'proposal', result
    if requested_card == 'root':
        assert Card.parse(result['completePrefix']['playActions'][27]['card']).id == card
        evidence = next(row for row in result['predicateEvidence'] if row['predicateId'] == 'PRED-USER-ROOT')
        assert evidence['satisfied'] is True
    assert result['qualified'] is False
    assert sorted(result['deckOrder']) == list(range(32))
    assert result['constructionModelHash']
    assert len(result['trace']['play']['plays']) == 32
    assert result['trace']['projects']['projects'] is not None
    assert result['trace']['escalation']['observation'] is not None
    assert len(result['completePrefix']['playActions']) == 32
    assert result['deckOrder'] == recorded['deckOrder']
    assert len(result['decisionPrefix']['playActions']) == 27
    assert result['decisionPrefix']['playActions'][:prefix_length] == [row.to_dict() for row in args['prefix']]
    assert result['decisionPrefix']['playActions'] == result['completePrefix']['playActions'][:27]
    assert result['decisionPrefix']['sha256'] != result['completePrefix']['sha256']
    normalized = OrToolsExactStateNormalizerV1(args['rules'], native_cp_sat=True).normalize(
        result['decisionPrefix'], decision_point={'kind': 'play'}, maximum_seconds=30)
    normalized.require_integrity()
    assert normalized.next_actor == args['bindings']['seat']
    assert len(normalized.state['completedTricks']) == 6
    if requested_card == 'legal_two':
        assert len(normalized.legal_actions) == 2
        evidence = next(row for row in result['predicateEvidence'] if row['predicateId'] == 'PRED-USER-CHOICES')
        assert evidence['satisfied'] is True
    if requested_card == 'raw_points':
        assert sum(t['rawPoints'] for t in normalized.state['completedTricks']
                   if Seat.parse(t['winner']).team == Team.NORTH_SOUTH) == expected_raw
        evidence = next(row for row in result['predicateEvidence'] if row['predicateId'] == 'PRED-USER-RAW')
        assert evidence['satisfied'] is True
    if requested_card == 'exclude_decision':
        excluded = tuple(TailPrefixActionV1.build(row['actor'], row['card'], ekka=row['ekka'])
                         for row in result['decisionPrefix']['playActions'])
        with pytest.raises(ValueError, match='end exactly'):
            construct_family_proposal_v1(**args, excluded_play_prefixes=(excluded + (excluded[0],),))
        alternate = construct_family_proposal_v1(**args, excluded_play_prefixes=(excluded,))
        if prefix_length == 27:
            # Varying the root or future suffix cannot escape an excluded decision.
            assert alternate['status'] == 'infeasible'
        else:
            assert alternate['status'] == 'proposal', alternate
            assert alternate['deckOrder'] == result['deckOrder']
            assert alternate['decisionPrefix']['playActions'] != result['decisionPrefix']['playActions']
            assert alternate['decisionPrefix']['playActions'][:20] == result['decisionPrefix']['playActions'][:20]
            assert alternate['qualified'] is False
    witness = family_proposal_witness_v1(result, rules=args['rules'], interest_policy={})
    assert witness['prefixHash'] == result['decisionPrefix']['sha256']
    assert witness['actor'] == normalized.next_actor
    assert witness['action'] in {row['code'] for row in normalized.legal_actions}
    assert witness['sourceHash'] == family_proposal_witness_v1(
        {**result, 'elapsedSeconds': 999}, rules=args['rules'], interest_policy={})['sourceHash']
    assert witness['policyHash'] == canonical_sha256({})
    assert 'qualified' not in witness
    with pytest.raises(ValueError, match='prefix disagrees'):
        family_proposal_witness_v1({**result, 'decisionPrefix': result['completePrefix']},
            rules=args['rules'], interest_policy={})
    with pytest.raises(ValueError, match='unqualified proposal'):
        family_proposal_witness_v1({**result, 'qualified': True},
            rules=args['rules'], interest_policy={})
