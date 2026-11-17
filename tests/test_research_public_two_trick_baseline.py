import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from baloot_platform.core.enums import Seat
from baloot_platform.research.decision_selection import baseline_strictly_prefers_alternative_v1
from baloot_platform.research.ortools_research_executor import (
    PUBLIC_EXHAUSTED_SUIT_BASELINE_ID,
    PUBLIC_TWO_TRICK_BASELINE_ID,
    _observer_safe_play_policy_artifact_v1,
)
from baloot_platform.research.scenario_contracts import (
    ScenarioProofRequestV1, build_scenario_video_interestingness_v1,
)


def test_two_trick_baseline_is_an_explicit_versioned_selection():
    policy = {"schemaVersion": 4, "maximumFinalMargin": 40, "minimumPointSwing": 20,
        "maximumFeaturedTrick": 7, "minimumMeaningfulDecisions": 1, "minimumConsistentWorlds": 2,
        "minimumBaselineRegret": 20, "minimumNextBestRegret": 20,
        "baselinePolicyId": PUBLIC_TWO_TRICK_BASELINE_ID, "allowTieRuleOnlyLessons": False}
    assert build_scenario_video_interestingness_v1(policy)["baselinePolicyId"] == PUBLIC_TWO_TRICK_BASELINE_ID


@pytest.mark.parametrize("seed", [826181, 827721, 829779])
def test_audit7_elementary_moves_are_baseline_obvious_without_hidden_hands(rules, seed):
    cases = json.loads((Path(__file__).parent / "fixtures/audit7_public_lessons.json").read_text())
    case = next(row for row in cases if row["seed"] == seed)
    state = dict(contract=case["contract"], trumpSuit=case["trump"],
        completedTricks=case["completed_tricks"], currentTrick={**case["current_trick"], "index": 6},
        hands={case["actor"]: case["own_cards"]}, projects=[])
    request = ScenarioProofRequestV1.build({"schemaVersion": 1, "mode": "bounded",
        "visibility": "actor_information_set", "strategyClass": "pure_deterministic",
        "claim": "winning_root_action_count", "targetTeam": Seat.parse(case["actor"]).team.name.lower(),
        "utility": "awarded_point_margin", "successThreshold": 0, "budgetProfile": "test",
        "horizon": {"unit": "tricks", "value": 1}, "boundaryEvaluator": "current_raw_point_margin_v1"})
    normalized = SimpleNamespace(decision_kind="play", next_actor=case["actor"], state=state,
        legal_actions=tuple({"code": card} for card in case["own_cards"]))
    results = []
    for hidden in (["SA"], ["C8"]):
        for seat in Seat:
            if seat.name.lower() != case["actor"]:
                state["hands"][seat.name.lower()] = hidden
        old, _ = _observer_safe_play_policy_artifact_v1(rules, normalized, request,
            actual_action=None, policy_id=PUBLIC_EXHAUSTED_SUIT_BASELINE_ID,
            maximum_seconds=2, cancel_check=None)
        assert old == case["baseline"]
        selected, artifact = _observer_safe_play_policy_artifact_v1(rules, normalized, request,
            actual_action=None, policy_id=PUBLIC_TWO_TRICK_BASELINE_ID,
            maximum_seconds=2, cancel_check=None)
        assert selected == case["action"]
        assert not baseline_strictly_prefers_alternative_v1(artifact["proofModel"]["actionQuality"], case["action"])
        results.append(copy.deepcopy(artifact))
    assert results[0] == results[1]
