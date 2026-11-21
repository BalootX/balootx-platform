"""Editorial tags derived after native receipt validation, never proof admission."""

from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Contract, Suit
from .contracts import canonical_sha256
from .scenario_contracts import ExactTracePrefixV1
from .family_catalog import TAG_AXES
from .decision_selection import baseline_strictly_prefers_alternative_v1
from baloot_platform.core.models import LegalPlayActionV1
from .lesson_prefilter import CLASSIFIED_LESSON_FAMILIES_V1, classify_public_lesson_v1


def classify_public_mechanism_v1(*, rules, current_trick, contract, trump, action):
    """Label the selected play using the public table and contract."""
    chosen = LegalPlayActionV1.from_code(action).card
    cards = [Card.parse(play["card"]) for play in current_trick["plays"]]
    if not cards:
        return "LEAD"
    contract = Contract.parse(contract)
    trump = Suit.parse(trump) if contract == Contract.HOKUM and trump is not None else None
    led = cards[0].suit
    if chosen.suit != led:
        return "TRUMP_RUFF" if chosen.suit == trump else "VOID_OFF_SUIT_DISCARD"
    if trump != led and any(card.suit == trump for card in cards):
        return "FOLLOW_SUIT_UNDERPLAY"
    strength = rules.play_domain.strength
    if all(strength(chosen, contract, trump) > strength(card, contract, trump)
           for card in cards if card.suit == led):
        return "FOLLOW_SUIT_CAPTURE"
    return "FOLLOW_SUIT_UNDERPLAY"


def derive_trace_lesson_tags_v1(trace, *, normalized, rules):
    """Derive focal tags only after the caller validates canonical trace quality."""
    if normalized is None or normalized.decision_kind != 'play':
        return None
    normalized.require_integrity()
    if normalized.rules_hash != rules.sha256:
        raise ValueError('lesson normalization uses different rules')
    receipt = trace.get('playingQualityEvidence', {}).get('receipt', {})
    group = (receipt.get('postAuction') or {}).get('projectPlay') or {}
    index = len(trace.get('exactTracePrefix', {}).get('playActions', ()))
    if not 16 <= index <= 27:
        return None
    rows = [row for row in group.get('decisions', ())
            if row.get('decisionKind', 'play') == 'play'
            and row.get('exactStateId') == normalized.exact_state_id
            and row.get('playIndex') == index]
    if len(rows) != 1:
        return None
    row = rows[0]
    artifact_hash, action = row.get('proofArtifactHash'), row.get('action')
    artifact = group.get('artifacts', {}).get(artifact_hash, {})
    values = _exact_values(artifact, action)
    if values is None or values[action] != max(values.values()):
        return None
    classifier = None
    baseline = group.get('baselineEvidence', {}).get(artifact_hash, {})
    quality = baseline.get('proofModel', {}).get('actionQuality', {})
    baseline_action = quality.get('selectedPolicyAction')
    if (_baseline_matches_decision(baseline, row)
            and quality.get('scope') == 'observer_safe_policy_conditioned'
            and quality.get('solverStatus') == 'optimal'
            and baseline.get('proof', {}).get('selectedAction') == baseline_action
            and baseline_action in values
            and baseline_strictly_prefers_alternative_v1(quality, action)):
        state = normalized.state
        classifier = classify_public_lesson_v1(rules=rules, actor=normalized.next_actor,
            own_cards=state['hands'][normalized.next_actor],
            completed_tricks=state['completedTricks'], current_trick=state.get('currentTrick') or {'plays': []},
            contract=state['contract'], trump=state.get('trumpSuit'),
            action=LegalPlayActionV1.from_code(action).card.code,
            baseline=LegalPlayActionV1.from_code(baseline_action).card.code)
        classifier = {**classifier, 'proofArtifactHash': artifact_hash,
                      'exactStateId': normalized.exact_state_id, 'action': action}
    tags = derive_lesson_tags_v1(group, proof_artifact_hash=artifact_hash, action=action,
        play_index=index, normalized=normalized, classifier=classifier)
    # Retain the actual heuristic preimage, not a requested family label.
    core = {**tags, 'classifier': classifier, 'qualityEvidenceHash': canonical_sha256(receipt)}
    core.pop('sha256')
    return {**core, 'sha256': canonical_sha256(core)}


_TECHNIQUES = {
    'stopper_forced_return': 'guard/protect', 'stranded_master': 'guard/protect',
    'overtake_unblock': 'unblock', 'partner_entry': 'partner-entry',
    'negative_project_inference': 'project-inference',
    'positive_project_inference': 'project-inference', 'void_count': 'count/void-inference',
    'endplay': 'endplay', 'trump_management': 'trump-control',
    'baloot_timing': 'baloot-timing', 'ekka_timing': 'ace-timing',
}


def live_lesson_tag_values_v1(*, project_inference_supported=False):
    """Search-ready producers only; historical and audit-owned tags stay readable."""
    techniques = {_TECHNIQUES[family] for family in CLASSIFIED_LESSON_FAMILIES_V1 if family in _TECHNIQUES}
    if project_inference_supported:
        techniques.add('project-inference')
    return {'impact': ['round-flip', 'points-only'],
            'technique': [value for value in TAG_AXES['technique'] if value in techniques]}


def _exact_values(artifact, action):
    request, proof = artifact.get('request', {}), artifact.get('proof', {})
    if request.get('mode') != 'exact' or proof.get('completionStatus') != 'exact_complete':
        return None
    if request.get('visibility') == 'actor_information_set' and (
        proof.get('hiddenWorlds', {}).get('complete') is not True
    ):
        return None
    rows = proof.get('rootActions', [])
    values = {row.get('action'): row.get('value') for row in rows}
    if len(values) != len(rows) or action not in values or len(values) < 2:
        return None
    if any(type(value) is not int for value in values.values()):
        return None
    return values


def _artifact_matches_state(artifact, state_id, *, required=True):
    allocation = artifact.get('allocationModel', {})
    identities = [document['exactStateId'] for document in (
        artifact, allocation, allocation.get('observerScopeBinding', {}))
        if 'exactStateId' in document]
    return bool(identities or not required) and all(
        type(value) is str and value == state_id for value in identities)


def _baseline_matches_decision(baseline, row):
    if not _artifact_matches_state(baseline, row['exactStateId'], required=False):
        return False
    if 'sourcePrefix' in baseline or 'sourcePrefixHash' in row:
        # Policy artifacts own the raw prefix; normalization excludes assertions.
        try:
            prefix = ExactTracePrefixV1.build(baseline['sourcePrefix'])
        except (KeyError, TypeError, ValueError):
            return False
        return prefix.sha256 == row.get('sourcePrefixHash')
    return _artifact_matches_state(baseline, row['exactStateId'])


def derive_lesson_tags_v1(group, *, proof_artifact_hash, action, play_index,
                          normalized, classifier=None, audit=None):
    """Consume an already ownership/hash-validated quality group.

    Missing causal evidence leaves an axis empty; family targets are not evidence.
    Audit and classifier inputs must identify this exact decision and artifact.
    The caller remains responsible for authenticating the audit source.
    """
    if type(play_index) is not int or not 16 <= play_index <= 27:
        raise ValueError('lesson tags require play index 16..27')
    normalized.require_integrity()
    contract = normalized.state.get('contract')
    if contract not in TAG_AXES['mode']:
        raise ValueError('unsupported lesson mode')
    artifact = group.get('artifacts', {}).get(proof_artifact_hash, {})
    state_id = normalized.exact_state_id
    if not _artifact_matches_state(artifact, state_id):
        raise ValueError('lesson normalization does not belong to this decision')
    values = _exact_values(artifact, action)
    if not state_id or values is None or values[action] != max(values.values()):
        raise ValueError('lesson tags require an exact optimal decision')
    rows = [row for row in group.get('decisions', ())
            if row.get('proofArtifactHash') == proof_artifact_hash
            and row.get('exactStateId') == state_id and row.get('action') == action
            and row.get('playIndex') == play_index]
    if len(rows) != 1:
        raise ValueError('lesson position does not match the owned decision')
    row = rows[0]
    tags = {axis: [] for axis in TAG_AXES}
    tags['mode'] = [contract]
    tags['trickBand'] = [f'trick-{play_index // 4 + 1}']
    sources = {'proofArtifactHash': proof_artifact_hash, 'exactStateId': state_id,
               'action': action, 'normalizationHash': normalized.sha256}
    request = artifact.get('request', {})
    baseline = group.get('baselineEvidence', {}).get(proof_artifact_hash, {})
    quality = baseline.get('proofModel', {}).get('actionQuality', {})
    baseline_action = quality.get('selectedPolicyAction')
    tie = group.get('tieRuleEvidence', {}).get(proof_artifact_hash, {})
    pair_impacts = []
    if (_baseline_matches_decision(baseline, row)
            and quality.get('scope') == 'observer_safe_policy_conditioned'
            and quality.get('solverStatus') == 'optimal'
            and baseline.get('proof', {}).get('selectedAction') == baseline_action
            and baseline_action in values
            and baseline_strictly_prefers_alternative_v1(quality, action)
            and tie.get('sourceArtifactHash') == proof_artifact_hash
            and tie.get('sourcePrefixHash') == row.get('sourcePrefixHash')
            and tie.get('baselineAction') == baseline_action
            and tie.get('sha256') == canonical_sha256({key: value for key, value in tie.items() if key != 'sha256'})):
        universe = None
        for rule in ('buyer_makes', 'buyer_loses'):
            comparison = tie.get(rule, {})
            margin, winner = comparison.get('margin', {}), comparison.get('roundWin', {})
            margins, wins = _exact_values(margin, action), _exact_values(winner, action)
            current_universe = comparison.get('allocationUniverseHash')
            if (margins is None or wins is None or set(margins) != set(values) or set(wins) != set(values)
                    or not current_universe or universe is not None and current_universe != universe
                    or set(wins.values()) - {-1, 1}
                    or any(not _artifact_matches_state(document, comparison.get('exactStateId'))
                           for document in (margin, winner))
                    or any(document.get('request', {}).get(key) != request.get(key)
                           for document in (margin, winner) for key in ('targetTeam', 'visibility'))
                    or margin.get('request', {}).get('utility') != 'awarded_point_margin'
                    or winner.get('request', {}).get('utility') != 'round_win'
                    or margins[action] != max(margins.values()) or margins[action] <= margins[baseline_action]):
                break
            universe = current_universe
            pair_impacts.append('round-flip' if wins[action] == 1 and wins[baseline_action] == -1
                                else 'points-only' if wins[action] == wins[baseline_action] else None)
        if len(pair_impacts) == 2 and pair_impacts[0] is not None and pair_impacts[0] == pair_impacts[1]:
            tags['impact'] = [pair_impacts[0]]
            sources.update(baselineAction=baseline_action, baselineEvidenceHash=canonical_sha256(baseline),
                           tieRuleEvidenceHash=tie['sha256'], impactScope='worst_case_baseline_pair_both_tie_rules')
    for label, evidence in (('classifier', classifier), ('audit', audit)):
        if evidence is None:
            continue
        if (evidence.get('proofArtifactHash') != proof_artifact_hash
                or evidence.get('exactStateId') != state_id
                or evidence.get('action') != action):
            raise ValueError(f'{label} does not belong to this decision')
        sources[f'{label}Hash'] = canonical_sha256(evidence)
    if classifier is not None:
        technique = _TECHNIQUES.get(classifier.get('family'))
        if technique:
            tags['technique'] = [technique]
    inference = group.get('projectInferenceEvidence', {}).get(proof_artifact_hash)
    if inference is not None:
        if (inference.get('sha256') != canonical_sha256({k: v for k, v in inference.items() if k != 'sha256'})
                or inference.get('kind') != 'native_public_project_inference_v1'
                or inference.get('scope') != 'public_rank_excludes_no_salvage_world'
                or inference.get('sourceArtifactHash') != proof_artifact_hash
                or inference.get('sourcePrefixHash') != row.get('sourcePrefixHash')
                or inference.get('exactStateId') != state_id or inference.get('action') != action
                or inference.get('rulesHash') != normalized.rules_hash
                or inference.get('sourceRoundWinHash') != canonical_sha256(
                    group.get('roundWinEvidence', {}).get(proof_artifact_hash, {}))):
            raise ValueError('project inference does not belong to this decision')
        if tags['impact']:
            techniques = {*tags['technique'], 'project-inference'}
            tags['technique'] = [value for value in TAG_AXES['technique'] if value in techniques]
            sources.update(projectInferenceEvidenceHash=inference['sha256'],
                           projectInferenceScope=inference['scope'])
    if audit is not None and audit.get('verdict') == 'accepted':
        if audit.get('level') not in TAG_AXES['level']:
            raise ValueError('accepted lesson audit requires a supported level')
        tags['level'] = [audit['level']]
    core = {'schemaVersion': 1, 'tags': tags, 'sources': sources,
            'playIndex': play_index, 'qualificationEvidence': False}
    return {**core, 'sha256': canonical_sha256(core)}
