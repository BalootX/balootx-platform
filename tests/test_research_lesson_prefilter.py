from types import SimpleNamespace
import json
from pathlib import Path
import pytest
from baloot_platform.research.lesson_prefilter import classify_public_lesson_v1, lesson_is_repeated_v1


def test_proposal_selection_config_is_closed_canonical_and_hash_bound(root):
    import yaml
    from baloot_platform.research.scenario_contracts import ScenarioQualityExecutionBudgetV1 as Budget
    profile = yaml.safe_load((root / 'configs/challenge/intelligent_research.yaml').read_text(encoding='utf-8'))
    policy = json.loads((root / 'tests/fixtures/actor_interest_calibration.json').read_text())['cases']['7189']['policy']
    raw = {**profile['scenarioResearchV4']['qualityExecution'],
           'candidateSource': 'runtime_proposal_stream_v1', 'proposalSeedCount': 20,
           'auditConstructedTrace': True, 'earlyPlayPolicyId': 'observer_safe_trick_control_v1',
           'exactPlayFromTrick': 7, 'videoInterestingness': policy}
    old = Budget.build(raw)
    assert 'proposalPlayIndices' not in old.to_dict()
    assert Budget.build(old.to_dict()).sha256 == old.sha256
    selected = Budget.build({**raw, 'proposalPlayIndices': [23, 16],
                             'acceptedLessonLibrary': {'overtake_unblock': 2}})
    assert selected.proposal_play_indices == (16, 23)
    assert Budget.build(selected.to_dict()).sha256 == selected.sha256 != old.sha256
    causal = Budget.build({**raw, 'acceptedLessonLibrary': {'overtake_unblock': ['a' * 64]}})
    assert Budget.build(causal.to_dict()).sha256 == causal.sha256 != selected.sha256
    assert causal.to_dict()['acceptedLessonLibrary'] == {'overtake_unblock': ['a' * 64]}
    from jsonschema import Draft202012Validator
    from scripts.generate_profile_schemas import apply_release_constraints, infer
    schema = infer(profile)
    apply_release_constraints('challenge', schema, profile)
    library_schema = schema['properties']['scenarioResearchV4']['properties']['qualityExecution']['properties']['acceptedLessonLibrary']
    validator = Draft202012Validator(library_schema)
    for valid in ({'overtake_unblock': 2}, {'overtake_unblock': ['a' * 64]}, {'overtake_unblock': []}):
        validator.validate(valid)
    for invalid in ({'overtake_unblock': ['A' * 64]}, {'overtake_unblock': ['a' * 64] * 2}, {'unknown': []}):
        assert list(validator.iter_errors(invalid))
    for bad in ([], [True], [32], [16, 16], None):
        with pytest.raises(ValueError):
            Budget.build({**raw, 'proposalPlayIndices': bad})
    for bad in ({'unknown': 2}, {'overtake_unblock': True}, None):
        with pytest.raises(ValueError):
            Budget.build({**raw, 'acceptedLessonLibrary': bad})


def test_known_family_needs_longer_explicit_chain():
    row = dict(family='overtake_unblock', deductions=['a', 'b'])
    assert lesson_is_repeated_v1(row, {'overtake_unblock': 2})
    assert not lesson_is_repeated_v1(row, {'overtake_unblock': 1})
    assert not lesson_is_repeated_v1(dict(family=None, deductions=[]), {'overtake_unblock': 2})
    with pytest.raises(ValueError):
        lesson_is_repeated_v1(row, {'unregistered': 1})


def test_ace_against_outstanding_ten_is_a_hypothesis_not_proof():
    ranks = {'7': 0, '8': 1, '9': 2, 'J': 3, 'Q': 4, 'K': 5, '10': 6, 'A': 7}
    domain = SimpleNamespace(strength=lambda card, *args: ranks[card.code[1:]], points=lambda *args: 0)
    row = classify_public_lesson_v1(rules=SimpleNamespace(play_domain=domain), actor='south',
        own_cards=['DA', 'DK'], completed_tricks=[],
        current_trick={'plays': [{'actor': 'east', 'card': 'D9'}]}, contract='sun', trump=None,
        action='DA', baseline='DK')
    assert row['family'] == 'ace_vs_ten'
    assert row['qualificationEvidence'] is False
    assert lesson_is_repeated_v1(row, {'ace_vs_ten': 2})


@pytest.mark.parametrize('seed,family', [(819579,'overtake_unblock'), (825599,'overtake_unblock'),
    (821953,'overtake_unblock'), (826181,'cash_into_partner'), (827721,'ace_vs_ten'),
    (829779,'stranded_master')])
def test_audited_elementary_and_repeated_families(seed, family):
    from baloot_platform.core.rules import RuleBook
    root = Path(__file__).resolve().parents[1]
    rules = RuleBook.load(root/'configs/rules/saudi_federation_2023_v4.yaml')
    rows = json.loads((root/'tests/fixtures/audit7_public_lessons.json').read_text())
    case = next(dict(row) for row in rows if row['seed'] == seed)
    case.pop('seed')
    result = classify_public_lesson_v1(rules=rules, **case)
    assert result['family'] == family
    assert lesson_is_repeated_v1(result, {family: 2})


AUDIT8_UNKNOWN = {873575, 873648, 876835, 877939, 880939, 872963}


@pytest.mark.parametrize('case', json.loads((Path(__file__).parent / 'fixtures/audit8_public_lessons.json').read_text()),
                         ids=lambda case: f"{case['seed']}-{case['playIndex']}")
def test_audit8_public_only_lesson_regressions(case):
    from baloot_platform.core.rules import RuleBook
    from baloot_platform.research.lesson_prefilter import LESSON_FAMILIES
    row = dict(case)
    seed = row.pop('seed')
    row.pop('playIndex')
    rules = RuleBook.load(Path(__file__).resolve().parents[1] / 'configs/rules/saudi_federation_2023_v8.yaml')
    result = classify_public_lesson_v1(rules=rules, **row)
    assert result['qualificationEvidence'] is False
    repeated = lesson_is_repeated_v1(result, dict.fromkeys(LESSON_FAMILIES, 2))
    if seed == 879034:
        assert result['family'] == 'overtake_unblock'
        assert len(result['deductions']) == 4
        assert not repeated
    elif seed in AUDIT8_UNKNOWN:
        assert result['classification'] == 'unknown'
        assert not repeated
    else:
        assert repeated


def test_audit8_fixture_contains_no_private_or_terminal_evidence():
    rows = json.loads((Path(__file__).parent / 'fixtures/audit8_public_lessons.json').read_text())
    assert len(rows) == 27
    assert len(AUDIT8_UNKNOWN - {872963}) == 5  # 20/25 reviewed rejects caught; both keeps survive.
    for row in rows:
        assert set(row) == {'seed', 'playIndex', 'actor', 'action', 'baseline', 'own_cards',
                            'completed_tricks', 'current_trick', 'contract', 'trump'}
        for trick in [*row['completed_tricks'], row['current_trick']]:
            assert set(trick) == {'plays'}
            assert all(set(play) == {'actor', 'card'} for play in trick['plays'])


def test_side_suit_before_trump_requires_actual_two_card_tail():
    from baloot_platform.core.rules import RuleBook
    rules = RuleBook.load(Path(__file__).resolve().parents[1] / 'configs/rules/saudi_federation_2023_v4.yaml')
    result = classify_public_lesson_v1(rules=rules, actor='south', own_cards=['D7', 'CA', 'C9'],
        completed_tricks=[], current_trick={'plays': []}, contract='hokum', trump='clubs',
        action='D7', baseline='CA')
    assert result['classification'] == 'unknown'


def test_secure_partner_banking_does_not_infer_ruff_safety():
    from baloot_platform.core.rules import RuleBook
    rules = RuleBook.load(Path(__file__).resolve().parents[1] / 'configs/rules/saudi_federation_2023_v4.yaml')
    result = classify_public_lesson_v1(rules=rules, actor='south', own_cards=['CA', 'C7'],
        completed_tricks=[], current_trick={'plays': [{'actor': 'north', 'card': 'SA'}]},
        contract='hokum', trump='hearts', action='CA', baseline='C7')
    assert result['classification'] == 'unknown'


def test_causal_fingerprint_ignores_suit_renaming_but_distinguishes_plan():
    from baloot_platform.core.rules import RuleBook
    rules = RuleBook.load(Path(__file__).resolve().parents[1] / 'configs/rules/saudi_federation_2023_v6.yaml')
    def classify(suit):
        return classify_public_lesson_v1(rules=rules, actor='south',
            own_cards=[suit+'A', suit+'K'], completed_tricks=[],
            current_trick={'plays': [{'actor': 'east', 'card': suit+'9'}]},
            contract='sun', trump=None, action=suit+'A', baseline=suit+'K')
    first, renamed = classify('D'), classify('H')
    assert first['causalFingerprint'] == renamed['causalFingerprint']
    library = {first['family']: [first['causalFingerprint']]}
    assert lesson_is_repeated_v1(renamed, library)
    assert not lesson_is_repeated_v1({**first, 'causalFingerprint': '0'*64}, library)
    assert not lesson_is_repeated_v1({'family': None, 'deductions': [], 'causalFingerprint': None}, library)
    def bank(history):
        return classify_public_lesson_v1(rules=rules, actor='south', own_cards=['CA', 'C7'],
            completed_tricks=history, current_trick={'plays': [{'actor': 'north', 'card': 'SA'}]},
            contract='sun', trump=None, action='CA', baseline='C7')
    secure = bank([])
    void = bank([{'plays': [{'actor': 'east', 'card': 'C8'}, {'actor': 'north', 'card': 'D7'}]}])
    assert secure['family'] == void['family'] == 'cash_into_partner'
    assert len(secure['deductions']) == len(void['deductions'])
    assert secure['causalFingerprint'] != void['causalFingerprint']
    assert not lesson_is_repeated_v1(void, {secure['family']: [secure['causalFingerprint']]})
    for bad in ([{}], ['bad'], [first['causalFingerprint']]*2, [True]):
        with pytest.raises(ValueError):
            lesson_is_repeated_v1(first, {first['family']: bad})
