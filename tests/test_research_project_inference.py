"""Native late-suffix inference and stored-reader coverage, not full-round certification."""
import copy
import json
from types import SimpleNamespace

import pytest
import yaml

from baloot_platform.core.rules import RuleBook
from baloot_platform.research.scenario_contracts import (
    ExactTracePrefixV1, ScenarioProofRequestV1, ScenarioQualityExecutionBudgetV1, canonical_sha256,
)


def _forbid_python_auction_semantics(monkeypatch):
    from baloot_platform.core.auction_ir import AuctionDomainIRV1

    def forbidden(*args, **kwargs):
        pytest.fail('native inference called Python auction semantics')

    for method in ('legal_actions', 'resolve_action'):
        monkeypatch.setattr(AuctionDomainIRV1, method, forbidden)


@pytest.fixture(scope='module')
def native_inference_round(root):
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    from baloot_platform.research.ortools_research_executor import (
        PUBLIC_TWO_TRICK_BASELINE_ID, _resolve_quality_continuation_v1,
    )

    rules = RuleBook.load(root / 'configs/rules/saudi_federation_2023_v8.yaml')
    prefix = ExactTracePrefixV1.build(json.loads(
        (root / 'tests/fixtures/project_rank_ownership_prefix.json').read_text(encoding='utf-8')))
    configuration = yaml.safe_load((root / 'configs/challenge/progressive_mining_ortools_v3_1.yaml').read_text(
        encoding='utf-8'))['scenarioResearchV4']
    request = ScenarioProofRequestV1.build({'schemaVersion': 1, 'mode': 'exact',
        'visibility': 'actor_information_set', 'strategyClass': 'pure_deterministic',
        'claim': 'winning_root_action_count', 'targetTeam': 'north_south',
        'utility': 'awarded_point_margin', 'successThreshold': 0, 'budgetProfile': 'actor_information_set_exact_v1'})
    execution = ScenarioQualityExecutionBudgetV1.build({'schemaVersion': 1, 'maximumWallSeconds': 60,
        'maximumEvidenceBytes': 268435456, 'maximumEscalationDecisions': 16,
        'videoInterestingness': {'schemaVersion': 4, 'maximumFinalMargin': 40,
            'minimumPointSwing': 20, 'maximumFeaturedTrick': 7, 'minimumMeaningfulDecisions': 1,
            'minimumConsistentWorlds': 2, 'minimumBaselineRegret': 20, 'minimumNextBestRegret': 20,
            'baselinePolicyId': PUBLIC_TWO_TRICK_BASELINE_ID, 'allowTieRuleOnlyLessons': False}})
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(
        prefix, maximum_seconds=10)
    with pytest.MonkeyPatch.context() as monkeypatch:
        _forbid_python_auction_semantics(monkeypatch)
        group = _resolve_quality_continuation_v1(rules, prefix, request, maximum_nodes=5000,
            maximum_depth=8, maximum_worlds=64, maximum_seconds=60, maximum_decision_seconds=20,
            maximum_evidence_bytes=execution.maximum_evidence_bytes,
            proof_budget_profiles=configuration['proofBudgets']['profiles'],
            search_budget_profiles=configuration['searchBudgets']['profiles'], execution_budget=execution,
            featured_play_index=25, featured_decision_point={'kind': 'play',
                'selector': {'actingSeat': 'south', 'trickIndex': 6, 'position': 1}})
    assert group['status'] == 'complete', group.get('partial')
    assert len(group.get('projectInferenceEvidence', {})) == 1
    return SimpleNamespace(rules=rules, prefix=prefix, request=request, execution=execution,
        normalized=normalized, group=group, configuration=configuration)


def test_native_suffix_produces_inference_and_passes_stored_readback(native_inference_round, monkeypatch):
    from baloot_platform.research.result_authority import validate_native_decision_quality_evidence_v1
    from baloot_platform.research.lesson_tags import derive_trace_lesson_tags_v1

    fixture = native_inference_round
    _forbid_python_auction_semantics(monkeypatch)
    validate_native_decision_quality_evidence_v1(fixture.group, source_prefix=fixture.prefix,
        rules_hash=fixture.rules.sha256, request=fixture.request, rules=fixture.rules,
        execution_budget=fixture.execution,
        search_budget_profiles=fixture.configuration['searchBudgets']['profiles'], maximum_seconds=60)
    evidence, = fixture.group['projectInferenceEvidence'].values()
    assert evidence['scope'] == 'public_rank_excludes_no_salvage_world'
    assert evidence['ownershipProbe']['ownership'] == {'card': 'C9', 'owner': 'west'}
    assert {row['value'] for row in evidence['contraryRoundWin']['proof']['rootActions']} == {-1}
    assert evidence['contraryRoundWin']['proof']['completionStatus'] == 'exact_complete'
    trace = {'exactTracePrefix': fixture.prefix.to_dict(),
             'playingQualityEvidence': {'receipt': {'postAuction': {'projectPlay': fixture.group}}}}
    tags = derive_trace_lesson_tags_v1(trace, normalized=fixture.normalized, rules=fixture.rules)
    assert tags['tags']['impact'] == ['round-flip']
    assert 'project-inference' in tags['tags']['technique']
    assert tags['sources']['projectInferenceEvidenceHash'] == evidence['sha256']
    assert tags['qualificationEvidence'] is False


@pytest.mark.parametrize('fault', ['action', 'prefix', 'state', 'rules', 'ownership', 'winning-world'])
def test_project_companion_rejects_rehashed_substitution(native_inference_round, fault):
    from baloot_platform.research.result_authority import _validate_project_inference_quality_companions_v1

    fixture = native_inference_round
    group = copy.deepcopy(fixture.group)
    evidence, = group['projectInferenceEvidence'].values()
    if fault == 'action':
        evidence['action'] = 'C10'
    elif fault == 'prefix':
        evidence['sourcePrefixHash'] = 'a' * 64
    elif fault == 'state':
        evidence['exactStateId'] = 'STATE-OTHER'
    elif fault == 'rules':
        evidence['rulesHash'] = 'a' * 64
    elif fault == 'ownership':
        evidence['ownershipProbe']['ownership']['owner'] = 'north'
    else:
        evidence['contraryRoundWin']['proof']['rootActions'][0]['value'] = 1
    evidence['sha256'] = canonical_sha256({k: v for k, v in evidence.items() if k != 'sha256'})
    with pytest.raises(ValueError, match='project inference companion'):
        _validate_project_inference_quality_companions_v1(group, rules=fixture.rules,
            execution_budget=fixture.execution,
            search_budget_profiles=fixture.configuration['searchBudgets']['profiles'],
            remaining=lambda: 20, cancel_check=None)


def test_project_inference_does_not_enable_unfinished_generated_families(native_inference_round):
    from baloot_platform.research.lesson_tags import live_lesson_tag_values_v1

    assert 'project-inference' not in live_lesson_tag_values_v1()['technique']


@pytest.mark.parametrize('fault', ['complete-state', 'incomplete-worlds', 'losing-source'])
def test_inference_requires_the_source_win_over_all_hidden_worlds(native_inference_round, monkeypatch, fault):
    from baloot_platform.research import ortools_research_executor as executor
    from baloot_platform.research.scenario_contracts import ScenarioProofBudgetV1

    fixture = native_inference_round
    digest, = fixture.group['projectInferenceEvidence']
    source = fixture.group['artifacts'][digest]
    winner = copy.deepcopy(fixture.group['roundWinEvidence'][digest])
    if fault == 'complete-state':
        winner['request']['visibility'] = 'complete_state'
    elif fault == 'incomplete-worlds':
        winner['proof']['hiddenWorlds']['complete'] = False
    else:
        next(row for row in winner['proof']['rootActions'] if row['action'] == source['proof']['selectedAction'])['value'] = -1
    monkeypatch.setattr(executor, '_complete_state_quality_artifact_v1', lambda *a, **kw: pytest.fail('proved unsupported witness'))
    budget = ScenarioProofBudgetV1.build(fixture.request.budget_profile,
        fixture.configuration['proofBudgets']['profiles'][fixture.request.budget_profile])
    assert executor._project_inference_quality_evidence_v1(fixture.rules, fixture.normalized,
        fixture.prefix, source, winner, budget=budget, maximum_nodes=5000, maximum_depth=8,
        maximum_conflicts=250000, remaining_seconds=lambda: 10, cancel_check=None) is None


def test_legacy_groups_need_no_new_evidence_or_budgets():
    from baloot_platform.research.result_authority import _validate_project_inference_quality_companions_v1

    _validate_project_inference_quality_companions_v1({}, rules=None, execution_budget=None,
        search_budget_profiles=None, remaining=lambda: pytest.fail('spent legacy budget'), cancel_check=None)


@pytest.mark.parametrize('stage', ['ownership', 'contrary'])
def test_incomplete_reproof_is_not_reported_as_corrupt_evidence(native_inference_round, monkeypatch, stage):
    from baloot_platform.research import ortools_research_executor as executor, negative_project_prefix
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
    from baloot_platform.research.result_authority import _validate_project_inference_quality_companions_v1

    if stage == 'ownership':
        monkeypatch.setattr(negative_project_prefix, 'probe_public_project_rank_ownership_v1',
                            lambda **kwargs: {'status': 'incomplete'})
    else:
        monkeypatch.setattr(executor, '_complete_state_quality_artifact_v1',
                            lambda *args, **kwargs: {'proof': {'completionStatus': 'incomplete'}})
    fixture = native_inference_round
    with pytest.raises(InformationSetCompilationStopped, match='project_inference_.*incomplete'):
        _validate_project_inference_quality_companions_v1(fixture.group, rules=fixture.rules,
            execution_budget=fixture.execution,
            search_budget_profiles=fixture.configuration['searchBudgets']['profiles'],
            remaining=lambda: 20, cancel_check=None)
