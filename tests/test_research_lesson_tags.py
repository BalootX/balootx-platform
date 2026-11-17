import copy
from unittest.mock import Mock

import pytest

from baloot_platform.research.lesson_tags import derive_lesson_tags_v1
from baloot_platform.research.contracts import canonical_sha256


def test_automatic_tag_support_uses_live_classifier_families_not_future_declarations():
    from baloot_platform.research.lesson_tags import live_lesson_tag_values_v1
    from baloot_platform.research.lesson_prefilter import CLASSIFIED_LESSON_FAMILIES_V1, LESSON_FAMILIES

    support = live_lesson_tag_values_v1()
    assert support == {'impact': ['round-flip', 'points-only'],
                       'technique': ['guard/protect', 'unblock', 'partner-entry']}
    assert 'negative_project_inference' in LESSON_FAMILIES - CLASSIFIED_LESSON_FAMILIES_V1
    support['technique'].append('project-inference')
    assert 'project-inference' not in live_lesson_tag_values_v1()['technique']


def trace_inputs(*, canonical=False):
    group = canonical_evidence() if canonical else evidence()
    trace = {'exactTracePrefix': {'playActions': [{}] * 26},
             'playingQualityEvidence': {'receipt': {'postAuction': {'projectPlay': group}}}}
    normalized = Mock(decision_kind='play', exact_state_id='STATE-1', next_actor='north',
        rules_hash='rules', sha256='normalization', state={'contract': 'sun', 'trumpSuit': None,
        'hands': {'north': ['HK', 'S10'], 'east': ['CA']}, 'completedTricks': [], 'currentTrick': None})
    return trace, normalized


def test_trace_tags_use_owned_focal_proof_and_only_observer_cards(monkeypatch):
    from baloot_platform.research import lesson_tags as module
    trace, normalized = trace_inputs()
    calls = []
    rules = Mock(sha256='rules')
    def classify(**kwargs):
        calls.append(kwargs)
        return {'family': 'stopper_forced_return', 'deductions': ['public_control'],
                'qualificationEvidence': False}
    monkeypatch.setattr(module, 'classify_public_lesson_v1', classify)
    result = module.derive_trace_lesson_tags_v1(trace, normalized=normalized, rules=rules)
    assert result['tags']['impact'] == ['round-flip']
    assert result['tags']['technique'] == ['guard/protect']
    assert result['tags']['level'] == []
    assert calls[0]['own_cards'] == ['HK', 'S10']
    assert calls[0]['current_trick'] == {'plays': []}
    normalized.state['hands']['east'] = ['DA']
    assert module.derive_trace_lesson_tags_v1(trace, normalized=normalized, rules=rules) == result
    assert calls[0] == calls[1]
    assert result['sha256'] == canonical_sha256({k: v for k, v in result.items() if k != 'sha256'})


@pytest.mark.parametrize('missing', ['baselineEvidence', 'tieRuleEvidence'])
def test_trace_tags_never_copy_requested_impacts(monkeypatch, missing):
    from baloot_platform.research import lesson_tags as module
    trace, normalized = trace_inputs()
    trace['requestedTags'] = {'impact': ['round-flip'], 'level': ['advanced']}
    trace['playingQualityEvidence']['receipt']['postAuction']['projectPlay'].pop(missing)
    monkeypatch.setattr(module, 'classify_public_lesson_v1', lambda **kw: {'family': None})
    result = module.derive_trace_lesson_tags_v1(trace, normalized=normalized, rules=Mock(sha256='rules'))
    assert result['tags']['impact'] == result['tags']['level'] == []


def test_trace_tags_reject_rules_substitution_and_missing_focal_proof():
    from baloot_platform.research.lesson_tags import derive_trace_lesson_tags_v1
    trace, normalized = trace_inputs()
    with pytest.raises(ValueError, match='different rules'):
        derive_trace_lesson_tags_v1(trace, normalized=normalized, rules=Mock(sha256='other'))
    trace['playingQualityEvidence']['receipt']['postAuction']['projectPlay']['decisions'] = []
    assert derive_trace_lesson_tags_v1(trace, normalized=normalized, rules=Mock(sha256='rules')) is None


def test_api_lesson_document_preserves_trace_shape_and_rejects_corrupt_tags(root):
    from baloot_platform.core.rules import RuleBook
    from baloot_platform.research.lesson_tags import derive_trace_lesson_tags_v1
    from baloot_platform.service.api import ResearchLessonTagsV1Response
    from baloot_platform.research.scenario_result import public_scenario_projection_v1
    trace, normalized = trace_inputs()
    rules = RuleBook.load(root / 'configs/rules/saudi_federation_2023_v8.yaml')
    normalized.rules_hash = rules.sha256
    document = derive_trace_lesson_tags_v1(trace, normalized=normalized, rules=rules)
    projection = public_scenario_projection_v1(
        {'publicProjectionHash': 'a' * 64, 'lessonTags': document},
        result_code='LAF-0123456789', predicate_evidence_count=0)
    public = projection['lessonTags']
    assert ResearchLessonTagsV1Response.model_validate(public).model_dump() == public
    assert set(public) == {'schemaVersion', 'tags', 'playIndex', 'qualificationEvidence', 'sha256'}
    changed = copy.deepcopy(public)
    changed['tags']['impact'] = ['invented']
    changed['sha256'] = canonical_sha256({k: v for k, v in changed.items() if k != 'sha256'})
    with pytest.raises(ValueError):
        ResearchLessonTagsV1Response.model_validate(changed)


def test_tag_query_api_transports_repeated_values_and_catalog_validation(monkeypatch):
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    from baloot_platform.service import api
    calls = []
    def page(**filters):
        calls.append(filters)
        return dict(items=[], total=0, offset=0, limit=100, nextOffset=None,
                    hasMore=False, groupBy=None, groups=[])
    monkeypatch.setattr(api, 'RESEARCH', SimpleNamespace(list_results=page))
    monkeypatch.setattr(api, '_require_trusted', lambda request: None)
    client = TestClient(api.app)
    response = client.get('/api/v2/research-results', params=[
        ('lessonImpact', 'points-only'), ('lessonImpact', 'round-flip'),
        ('lessonTechnique', 'guard/protect'), ('lessonMode', 'sun')])
    assert response.status_code == 200, response.text
    assert calls[0]['tag_query'] == {'impact': ['round-flip', 'points-only'],
                                  'technique': ['guard/protect'], 'mode': ['sun']}
    assert client.get('/api/v2/research-results', params={'lessonImpact': 'invented'}).status_code == 422
    assert client.get('/api/v2/research-results',
                      params=[('lessonMode', 'sun')] * 3).status_code == 422
    assert len(calls) == 1
    from baloot_platform.research.family_catalog import TAG_AXES
    parameters = {row['name']: row['schema'] for row in
                  api.app.openapi()['paths']['/api/v2/research-results']['get']['parameters']}
    for axis in TAG_AXES:
        array = parameters[f'lesson{axis[0].upper()}{axis[1:]}']['anyOf'][0]
        assert array['items']['enum'] == list(TAG_AXES[axis])
        assert array['minItems'] == 1 and array['maxItems'] == len(TAG_AXES[axis])


def test_result_authority_rederives_tags_from_owned_normalization(monkeypatch):
    from baloot_platform.research import ortools_exact_state, lesson_tags
    from baloot_platform.research.result_authority import validated_trace_lesson_tags_v1
    trace, normalized = trace_inputs()
    raw = {key: key for key in ('exactStateId', 'prefixHash', 'rulesHash', 'stateHash',
        'phase', 'decisionKind', 'nextActor', 'projectDeclarationsVerified',
        'projectAssertionsVerified', 'eventChainHash', 'sha256')}
    raw.update(legalActions=[], state=normalized.state)
    trace.update(exactNormalization=raw, normalizationReceipt={}, lessonTags={'derived': True})
    seen = []
    def reconstruct(**fields):
        seen.append(fields)
        return normalized
    monkeypatch.setattr(ortools_exact_state, 'OrToolsNormalizedExactStateV1', reconstruct)
    monkeypatch.setattr(lesson_tags, 'derive_trace_lesson_tags_v1',
                        lambda actual, **kw: {'derived': True})
    assert validated_trace_lesson_tags_v1(trace, rules=Mock()) == {'derived': True}
    assert seen[0]['state'] is raw['state']
    trace['lessonTags'] = {'requested': True}
    with pytest.raises(ValueError, match='disagree'):
        validated_trace_lesson_tags_v1(trace, rules=Mock())
    assert validated_trace_lesson_tags_v1({}, rules=None) is None


@pytest.mark.parametrize('canonical', [False, True])
@pytest.mark.parametrize('by_query', [False, True])
@pytest.mark.parametrize('query,status', [
    ({'impact': ['round-flip']}, 'qualified'),
    ({'impact': ['project-swing']}, 'semantic_postcondition_miss'),
    ({'level': ['advanced']}, 'semantic_postcondition_miss'),
])
def test_result_tag_query_is_checked_after_quality_validation_and_before_hashing(root, monkeypatch, query, status, by_query, canonical):
    from types import SimpleNamespace
    from baloot_platform.core.rules import RuleBook
    from baloot_platform.research import ortools_research_executor as executor_module
    from baloot_platform.research import lesson_tags as module
    trace, normalized = trace_inputs(canonical=canonical)
    rules = RuleBook.load(root / 'configs/rules/saudi_federation_2023_v8.yaml')
    normalized.rules_hash = rules.sha256
    validated = []
    # This test isolates admission ordering, not strategic proof validation.
    monkeypatch.setattr(executor_module, 'validate_trace_quality_evidence_v1',
                        lambda *a, **kw: validated.append(True))
    monkeypatch.setattr(executor_module, '_validate_play_policy_continuation_binding_v1', lambda *a, **kw: None)
    def classify(**kwargs):
        assert validated == [True]
        return {'family': 'stopper_forced_return', 'qualificationEvidence': False}
    monkeypatch.setattr(module, 'classify_public_lesson_v1', classify)
    executor = executor_module.OrToolsUnifiedResearchExecutorV1(rules, backend_id='ortools_cp_sat_v3',
        bundle_revision_id='CBR-TAG-TEST', bundle_hash='a' * 64)
    executor.quality_execution = SimpleNamespace(family_target=None if by_query else {'tagQuery': query})
    if by_query:
        from baloot_platform.research.family_catalog import build_family_generation_request_v1
        executor.generation = build_family_generation_request_v1(dict(schemaVersion=1,
            catalogHash='a' * 64, tagQuery=query))
    stage = executor_module.OrToolsResearchStageReceiptV1('playPrefix', *(['a' * 64] * 3), 'optimal')
    candidate = executor_module.OrToolsNeutralCandidateV1('search', trace, {}, (),
        {'playPrefix': stage}, exact_state_id='STATE-1')
    preflight = executor_module.OrToolsResearchPreflightV1('search', 'playPrefix', ('playPrefix',),
        (), (), (), 'b' * 64)
    result = executor._result('qualified', preflight, candidate, None, None, (), 0, 0, 1,
        scenario_hash='c' * 64, _lesson_normalized=normalized)
    assert result.status == status
    assert result.candidate.trace['lessonTags']['tags']['impact'] == ['round-flip']
    assert result.candidate.sha256 != candidate.sha256
    assert result.research_authority['traceHash'] == canonical_sha256(result.candidate.trace)
    if by_query:
        assert result.candidate.trace['generationRequest'] == executor.generation
    if status != 'qualified':
        assert result.execution_receipt['candidateAttemptReason'] == (
            'generation_result_tags_mismatch' if by_query else 'family_target_result_tags_mismatch')


def seal_tie(group):
    tie = group['tieRuleEvidence']['proof']
    tie['sha256'] = canonical_sha256({key: value for key, value in tie.items() if key != 'sha256'})


def evidence():
    artifact = {'exactStateId': 'STATE-1', 'request': {
        'mode': 'exact', 'visibility': 'actor_information_set',
        'utility': 'awarded_point_margin', 'targetTeam': 'north_south'},
        'proof': {'completionStatus': 'exact_complete', 'hiddenWorlds': {'complete': True},
                  'rootActions': [{'action': 'HK', 'value': 0}, {'action': 'S10', 'value': -26}]}}
    winner = copy.deepcopy(artifact)
    winner['request']['utility'] = 'round_win'
    winner['proof']['rootActions'] = [{'action': 'HK', 'value': 1}, {'action': 'S10', 'value': -1}]
    comparison = {'margin': artifact, 'roundWin': winner, 'exactStateId': 'STATE-1',
                  'allocationUniverseHash': 'same-worlds'}
    group = {'artifacts': {'proof': artifact}, 'roundWinEvidence': {'proof': winner},
            'baselineEvidence': {'proof': {'exactStateId': 'STATE-1',
                'proof': {'selectedAction': 'S10'}, 'proofModel': {'actionQuality': {
                    'selectedPolicyAction': 'S10', 'scope': 'observer_safe_policy_conditioned',
                    'solverStatus': 'optimal', 'actionScores': [
                        {'action': 'HK', 'value': 1}, {'action': 'S10', 'value': 2}]}}}},
            'tieRuleEvidence': {'proof': {'sourceArtifactHash': 'proof', 'baselineAction': 'S10',
                'buyer_makes': copy.deepcopy(comparison), 'buyer_loses': copy.deepcopy(comparison)}},
            'decisions': [{'proofArtifactHash': 'proof', 'exactStateId': 'STATE-1',
                           'action': 'HK', 'playIndex': 26}]}
    seal_tie(group)
    return group


def canonical_evidence():
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1
    group = evidence()
    prefix = ExactTracePrefixV1.build(dict(schemaVersion=1, deckOrder=list(range(32)),
        dealer='east', auctionActions=[], escalationActions=[], projectDeclarations=[], playActions=[]))
    group['decisions'][0]['sourcePrefixHash'] = prefix.sha256
    baseline = group['baselineEvidence']['proof']
    baseline.pop('exactStateId')
    baseline['allocationModel'] = {'kind': 'observer_safe_public_projection_v1'}
    baseline['sourcePrefix'] = prefix.to_dict()
    tie = group['tieRuleEvidence']['proof']
    tie['sourcePrefixHash'] = prefix.sha256
    for rule in ('buyer_makes', 'buyer_loses'):
        comparison = tie[rule]
        comparison['exactStateId'] = 'STATE-1' if rule == 'buyer_makes' else 'STATE-TIE-VARIANT'
        for key in ('margin', 'roundWin'):
            comparison[key].pop('exactStateId')
            comparison[key]['allocationModel'] = {'exactStateId': comparison['exactStateId'],
                'observerScopeBinding': {'exactStateId': comparison['exactStateId']}}
    artifact = group['artifacts']['proof']
    artifact.pop('exactStateId')
    artifact['allocationModel'] = {'exactStateId': 'STATE-1',
        'observerScopeBinding': {'exactStateId': 'STATE-1'}}
    seal_tie(group)
    return group


def test_canonical_allocation_and_baseline_prefix_produce_tags_without_flat_state_ids(monkeypatch):
    from baloot_platform.research import lesson_tags as module
    trace, normalized = trace_inputs(canonical=True)
    normalized.prefix_hash = 'effective-prefix-excludes-assertions'
    monkeypatch.setattr(module, 'classify_public_lesson_v1',
        lambda **kw: {'family': 'stopper_forced_return', 'qualificationEvidence': False})
    tags = module.derive_trace_lesson_tags_v1(trace, normalized=normalized, rules=Mock(sha256='rules'))
    assert tags['tags']['impact'] == ['round-flip']
    assert tags['tags']['technique'] == ['guard/protect']
    assert tags['sources']['exactStateId'] == 'STATE-1'


@pytest.mark.parametrize('fault', ['flat', 'allocation', 'observer', 'missing', 'duplicate-row'])
def test_canonical_proof_rejects_missing_conflicting_or_ambiguous_ownership(fault):
    group = canonical_evidence()
    artifact = group['artifacts']['proof']
    if fault == 'flat':
        artifact['exactStateId'] = 'OTHER'
    elif fault == 'allocation':
        artifact['allocationModel']['exactStateId'] = 'OTHER'
    elif fault == 'observer':
        artifact['allocationModel']['observerScopeBinding']['exactStateId'] = 'OTHER'
    elif fault == 'missing':
        artifact['allocationModel'] = {}
    else:
        group['decisions'].append(copy.deepcopy(group['decisions'][0]))
    with pytest.raises(ValueError, match='normalization does not belong|position does not match'):
        derive(group)


@pytest.mark.parametrize('fault', ['missing', 'corrupt-prefix', 'other-prefix', 'flat-state', 'observer-state', 'tie-prefix', 'tie-state'])
def test_canonical_baseline_and_tie_substitution_cannot_invent_impact(fault):
    from baloot_platform.research.scenario_contracts import ExactTracePrefixV1
    group = canonical_evidence()
    baseline = group['baselineEvidence']['proof']
    tie = group['tieRuleEvidence']['proof']
    if fault == 'missing':
        baseline.pop('sourcePrefix')
    elif fault in {'corrupt-prefix', 'other-prefix'}:
        baseline['sourcePrefix']['dealer'] = 'north'
        if fault == 'other-prefix':
            baseline['sourcePrefix'].pop('sha256')
            baseline['sourcePrefix'] = ExactTracePrefixV1.build(baseline['sourcePrefix']).to_dict()
    elif fault == 'flat-state':
        baseline['exactStateId'] = 'OTHER'
    elif fault == 'observer-state':
        baseline['allocationModel']['observerScopeBinding'] = {'exactStateId': 'OTHER'}
    elif fault == 'tie-prefix':
        tie['sourcePrefixHash'] = 'OTHER'
    else:
        tie['buyer_loses']['roundWin']['allocationModel']['observerScopeBinding']['exactStateId'] = 'OTHER'
    seal_tie(group)
    assert derive(group)['tags']['impact'] == []


def test_canonical_tags_rederive_on_persisted_readback_and_reject_substitution(monkeypatch):
    from baloot_platform.research import lesson_tags, ortools_exact_state
    from baloot_platform.research.result_authority import validated_trace_lesson_tags_v1
    trace, normalized = trace_inputs(canonical=True)
    rules = Mock(sha256='rules')
    monkeypatch.setattr(lesson_tags, 'classify_public_lesson_v1', lambda **kw: {'family': 'stopper_forced_return'})
    trace['lessonTags'] = lesson_tags.derive_trace_lesson_tags_v1(trace, normalized=normalized, rules=rules)
    trace['exactNormalization'] = {key: key for key in ('exactStateId', 'prefixHash', 'rulesHash', 'stateHash',
        'phase', 'decisionKind', 'nextActor', 'projectDeclarationsVerified', 'projectAssertionsVerified',
        'eventChainHash', 'sha256')}
    trace['exactNormalization'].update(legalActions=[], state=normalized.state)
    trace['normalizationReceipt'] = {}
    monkeypatch.setattr(ortools_exact_state, 'OrToolsNormalizedExactStateV1', lambda **kw: normalized)
    assert validated_trace_lesson_tags_v1(trace, rules=rules) == trace['lessonTags']
    group = trace['playingQualityEvidence']['receipt']['postAuction']['projectPlay']
    group['baselineEvidence']['proof']['sourcePrefix']['dealer'] = 'north'
    with pytest.raises(ValueError, match='disagree'):
        validated_trace_lesson_tags_v1(trace, rules=rules)


def derive(group, **kwargs):
    normalized = kwargs.pop('normalized', Mock(
        exact_state_id='STATE-1', state={'contract': 'sun'}, sha256='normalization'))
    return derive_lesson_tags_v1(group, proof_artifact_hash='proof', action='HK',
                               play_index=26, normalized=normalized, **kwargs)


def test_round_flip_uses_winner_proof_not_zero_margin():
    group = evidence()
    tags = derive(group)
    assert tags['tags']['impact'] == ['round-flip']
    assert tags['tags']['mode'] == ['sun']
    assert tags['tags']['trickBand'] == ['trick-7']
    assert tags['tags']['level'] == []
    assert tags['qualificationEvidence'] is False
    for rule in ('buyer_makes', 'buyer_loses'):
        group['tieRuleEvidence']['proof'][rule]['roundWin']['proof']['rootActions'][0]['value'] = -1
    seal_tie(group)
    assert derive(group)['tags']['impact'] == ['points-only']


def test_missing_or_wrong_winner_evidence_does_not_invent_impact():
    group = evidence()
    group['tieRuleEvidence']['proof']['buyer_loses']['roundWin']['exactStateId'] = 'OTHER'
    seal_tie(group)
    assert derive(group)['tags']['impact'] == []
    group.pop('tieRuleEvidence')
    assert derive(group)['tags']['impact'] == []


def test_losing_third_action_cannot_turn_a_winning_baseline_into_round_flip():
    group = evidence()
    group['artifacts']['proof']['proof']['rootActions'].append({'action': 'C7', 'value': -40})
    for rule in ('buyer_makes', 'buyer_loses'):
        comparison = group['tieRuleEvidence']['proof'][rule]
        comparison['margin']['proof']['rootActions'].append({'action': 'C7', 'value': -40})
        comparison['roundWin']['proof']['rootActions'][1]['value'] = 1
        comparison['roundWin']['proof']['rootActions'].append({'action': 'C7', 'value': -1})
    seal_tie(group)
    assert derive(group)['tags']['impact'] == ['points-only']


@pytest.mark.parametrize('fault', ['tie-only', 'baseline-tie', 'baseline-owner', 'worlds', 'incomplete', 'baseline-binding'])
def test_impact_requires_strict_owned_baseline_and_both_complete_tie_companions(fault):
    group = evidence()
    baseline = group['baselineEvidence']['proof']
    tie = group['tieRuleEvidence']['proof']
    if fault == 'tie-only':
        tie['buyer_loses']['roundWin']['proof']['rootActions'][1]['value'] = 1
    elif fault == 'baseline-tie':
        baseline['proofModel']['actionQuality']['actionScores'][0]['value'] = 2
    elif fault == 'baseline-owner':
        baseline['exactStateId'] = 'OTHER'
    elif fault == 'worlds':
        tie['buyer_loses']['allocationUniverseHash'] = 'different'
    elif fault == 'incomplete':
        tie['buyer_loses']['roundWin']['proof']['completionStatus'] = 'incomplete'
    else:
        tie['baselineAction'] = 'HK'
    seal_tie(group)
    assert derive(group)['tags']['impact'] == []


def test_trick_band_cannot_be_relabelled_from_a_different_position():
    group = evidence()
    group['decisions'][0]['playIndex'] = 23
    with pytest.raises(ValueError, match='position does not match'):
        derive(group)


def test_classifier_and_owner_audit_are_bound_and_level_is_not_guessed():
    binding = {'proofArtifactHash': 'proof', 'exactStateId': 'STATE-1', 'action': 'HK'}
    tags = derive(evidence(), classifier={**binding, 'family': 'stopper_forced_return'},
                  audit={**binding, 'verdict': 'accepted', 'level': 'intermediate'})
    assert tags['tags']['technique'] == ['guard/protect']
    assert tags['tags']['level'] == ['intermediate']
    assert derive(evidence(), audit={**binding, 'verdict': 'rejected', 'level': 'advanced'})['tags']['level'] == []
    with pytest.raises(ValueError, match='does not belong'):
        derive(evidence(), audit={**binding, 'proofArtifactHash': 'substitute'})
    with pytest.raises(ValueError, match='does not belong'):
        derive(evidence(), classifier={**binding, 'action': 'S10', 'family': 'partner_entry'})


def test_mode_requires_matching_intact_normalization():
    normalized = Mock(exact_state_id='OTHER', state={'contract': 'hokum'}, sha256='other')
    with pytest.raises(ValueError, match='normalization does not belong'):
        derive(evidence(), normalized=normalized)
    normalized.require_integrity.assert_called_once_with()
    normalized.exact_state_id = 'STATE-1'
    normalized.require_integrity.side_effect = ValueError('corrupted normalization')
    with pytest.raises(ValueError, match='corrupted normalization'):
        derive(evidence(), normalized=normalized)


def test_selected_action_is_bound_into_tag_identity():
    result = derive(evidence())
    assert result['sources']['action'] == 'HK'
    assert result['sources']['normalizationHash'] == 'normalization'


@pytest.mark.parametrize('mutation', ['incomplete', 'worlds', 'nonoptimal'])
def test_tags_do_not_promote_unproved_or_nonoptimal_choices(mutation):
    group = evidence()
    proof = group['artifacts']['proof']['proof']
    if mutation == 'incomplete':
        proof['completionStatus'] = 'incomplete'
    elif mutation == 'worlds':
        proof['hiddenWorlds']['complete'] = False
    else:
        proof['rootActions'][0]['value'] = -40
    with pytest.raises(ValueError, match='exact optimal'):
        derive(group)
