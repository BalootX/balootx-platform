from baloot_platform.agents.buying_policy_v4 import (
    BidAction, BidCandidate, BuyContext, BuyPolicyConfig, decide_buy,
)
from baloot_platform.agents.project_disclosure_v4 import (
    ProjectContext, ProjectPolicyConfig, decide_project,
)
from baloot_platform.challenge.interestingness_v4 import (
    ActionOutcome, ChallengeArchetype, ContextFeatures, score_interestingness,
)
from baloot_platform.history.provenance_v4 import ProvenanceLink, canonical_hash, game_serial


def test_interestingness_scores_cliff_and_buy_pass() -> None:
    score = score_interestingness(
        [
            ActionOutcome('buy_hokum', 0.72, 0.78, 28, downside_cvar=-0.05, hidden_deal_values=(0.8, 0.7, 0.6)),
            ActionOutcome('pass', -0.34, 0.22, -19, downside_cvar=-0.65, hidden_deal_values=(-0.4, -0.2, -0.3), is_obvious_heuristic_choice=True),
        ],
        ContextFeatures(decision_kind='bid', opponent_bought=False, score_pressure=0.7),
    )
    assert score.best_action_id == 'buy_hokum'
    assert score.cliff_score > 0.5
    assert score.archetype_scores[ChallengeArchetype.BUY_OR_PASS.value] > 0.4


def test_buy_policy_only_selects_legal_candidate() -> None:
    decision = decide_buy(
        BuyContext(
            actor='south', hand=('AS', '10S'), purchase_card='7H',
            legal_candidates=(
                BidCandidate(BidAction.PASS),
                BidCandidate(BidAction.HOKUM, 'hearts', 0.72, 18.0, -8.0),
            ),
            team_score=120, opponent_score=135,
        ),
        BuyPolicyConfig(deterministic_seed=7),
    )
    assert decision.action in {BidAction.PASS, BidAction.HOKUM}


def test_mandatory_project_cannot_be_concealed() -> None:
    decision = decide_project(
        ProjectContext('north', 'seq_100', 100, 0.8, 0.9, True, True),
        ProjectPolicyConfig(deception_probability=1.0, mistake_probability=1.0),
    )
    assert decision.declare is True
    assert 'mandatory_declaration' in decision.reason_codes


def test_provenance_is_stable_and_tamper_evident() -> None:
    payload = {'game': 'x', 'events': [1, 2, 3]}
    digest = canonical_hash(payload)
    assert digest == canonical_hash(payload)
    serial = game_serial(digest)
    assert serial.startswith('BLT-')
    link = ProvenanceLink(serial, digest, 'a' * 64, 'b' * 64, '4.1.0', challenge_id='CHL-1')
    original = link.artifact_identity
    changed = ProvenanceLink(serial, digest, 'c' * 64, 'b' * 64, '4.1.0', challenge_id='CHL-1')
    assert original != changed.artifact_identity
