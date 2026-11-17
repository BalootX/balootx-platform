from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from baloot_platform.research import construction_pre_screen as screen
from baloot_platform.research.scenario_contracts import ExactTracePrefixV1


@pytest.fixture
def policy():
    return {"schemaVersion": 4, "maximumFinalMargin": 40, "minimumPointSwing": 20,
            "maximumFeaturedTrick": 7, "minimumMeaningfulDecisions": 1,
            "minimumConsistentWorlds": 2, "minimumBaselineRegret": 20,
            "minimumNextBestRegret": 20, "baselinePolicyId": "observer_safe_trick_control_v1",
            "allowTieRuleOnlyLessons": False}


@pytest.fixture
def completed_seed(rules):
    for seed in range(16):
        _engine, _state, prefix = screen._proposal_opening(rules, seed)
        if prefix is not None:
            return seed
    pytest.fail("existing runtime fixture seeds all passed out")


def test_targeted_opening_keeps_legacy_seed_dealer_default(rules):
    from baloot_platform.core.enums import Seat

    _, legacy, legacy_prefix = screen._proposal_opening(rules, 1)
    _, explicit, explicit_prefix = screen._proposal_opening(rules, 1, dealer=Seat(1).name.lower())
    assert legacy.dealer == explicit.dealer == Seat(1)
    assert legacy_prefix == explicit_prefix
    _, targeted, targeted_prefix = screen._proposal_opening(rules, 1, dealer='north')
    assert targeted.dealer == Seat.NORTH
    if targeted_prefix is not None:
        assert targeted_prefix['dealer'] == 'north'


@pytest.mark.real_clock
@pytest.mark.parametrize('preferred', [None, 'east'])
def test_proposal_policy_forwards_optional_winner_preference(rules, completed_seed, monkeypatch, preferred):
    import time
    from baloot_platform.core.enums import Seat
    from baloot_platform.research import ortools_research_executor

    engine, state, _ = screen._proposal_opening(rules, completed_seed)
    actions = engine.legal_play_actions(state)
    scores = {action.code: int(index == 0) for index, action in enumerate(actions)}
    seen = []
    def artifact(*args, **kwargs):
        seen.append(kwargs['preferred_completed_trick_winner'])
        return actions[0].code, {'proofModel': {'actionQuality': {
            'actionScores': [{'action': action, 'value': value} for action, value in scores.items()]}}}
    monkeypatch.setattr(ortools_research_executor, '_observer_safe_play_policy_artifact_v1', artifact)
    preference = Seat.parse(preferred) if preferred is not None else None
    selected, actual_scores = screen._proposal_policy_score(
        rules, engine, state, 'observer_safe_trick_control_v1', time.monotonic() + 10,
        lambda: None, preferred_completed_trick_winner=preference)
    assert seen == [preference]
    assert selected == actions[0].code and actual_scores == scores


@pytest.fixture
def heuristic(monkeypatch):
    # Controlled estimates test plumbing only; they are never native truth.
    def score(rules, engine, state, policy_id, deadline, check):
        check()
        actions = engine.legal_play_actions(state)
        return actions[0].code, {action.code: int(index == 0) for index, action in enumerate(actions)}

    def values(engine, state, maximum_nodes, check, **kwargs):
        check()
        actions = engine.legal_play_actions(state)
        result = {action.code: 100 if index == len(actions) - 1 else 0
                  for index, action in enumerate(actions)}
        if kwargs.get("root_action") is not None:
            result = {kwargs["root_action"]: result[kwargs["root_action"]]}
        return result, 1

    monkeypatch.setattr(screen, "_proposal_policy_score", score)
    monkeypatch.setattr(screen, "_proposal_root_values", values)


def run(rules, policy, seed, **kwargs):
    return screen.screen_runtime_proposals_v1(
        rules, seed=seed, rollout_policy_id="observer_safe_trick_control_v1",
        selection_policy=policy, maximum_nodes=100, maximum_seconds=10, **kwargs)


def test_screen_scans_all_indices_and_keeps_estimates_unqualified(
    rules, policy, completed_seed, heuristic,
):
    before = copy.deepcopy(policy)
    result = run(rules, policy, completed_seed)
    assert result["status"] == "complete"
    assert result["projectedFinalMargin"] >= 0
    assert [row["playIndex"] for row in result["decisions"]] == list(range(32))
    assert result["proposals"]
    indices = [row["playIndex"] for row in result["proposals"]]
    assert indices == sorted(indices, reverse=True)
    assert policy == before
    for row in result["proposals"]:
        prefix = ExactTracePrefixV1.build(row["prefix"])
        assert prefix.sha256 == row["prefixHash"]
        assert len(prefix.play_actions) == row["playIndex"]
        assert row["baselineScores"][row["baseline"]] > row["baselineScores"][row["action"]]
        assert row["rulesHash"] == rules.sha256
        assert row["proposalOnly"] is True and row["qualified"] is False


def test_limited_search_budget_retains_tail_progress(
    rules, policy, completed_seed, heuristic, monkeypatch,
):
    full = run(rules, policy, completed_seed)
    clock = [0.0]
    monkeypatch.setattr(screen.time, "monotonic", lambda: clock[0])
    values = screen._proposal_root_values
    searched = []

    def bounded(engine, state, maximum_nodes, check, **kwargs):
        searched.append(len(state.completed_tricks) * 4 + len(state.current_trick.plays))
        clock[0] += 6
        return values(engine, state, maximum_nodes, check, **kwargs)

    monkeypatch.setattr(screen, "_proposal_root_values", bounded)
    result = run(rules, policy, completed_seed)
    assert result["status"] == "time_budget"
    assert [row["playIndex"] for row in result["decisions"]] == list(range(32))
    assert result["projectedFinalMargin"] == full["projectedFinalMargin"]
    assert len(searched) == 2 and searched[0] > searched[1]
    assert result["proposals"] == full["proposals"][:1]
    assert result["decisions"][searched[1]]["status"] == "time_budget"
    assert result["decisions"][0]["status"] == "pending"
    assert result["qualified"] is False


def test_frozen_gaps_and_equal_baseline_scores_do_not_create_proposals(
    rules, policy, completed_seed, heuristic, monkeypatch,
):
    policy["minimumNextBestRegret"] = 101
    assert run(rules, policy, completed_seed)["proposals"] == []
    policy["minimumNextBestRegret"] = 20
    def tied(rules, engine, state, policy_id, deadline, check):
        actions = engine.legal_play_actions(state)
        return actions[0].code, {action.code: 0 for action in actions}
    monkeypatch.setattr(screen, "_proposal_policy_score", tied)
    assert run(rules, policy, completed_seed)["proposals"] == []


@pytest.mark.real_clock
def test_cancel_and_expired_budget_are_explicit(rules, policy):
    assert run(rules, policy, 0, cancel_check=lambda: True)["status"] == "cancelled"
    result = screen.screen_runtime_proposals_v1(
        rules, seed=0, rollout_policy_id="observer_safe_trick_control_v1",
        selection_policy=policy, maximum_nodes=1, maximum_seconds=1e-12)
    assert result["status"] == "time_budget"
    assert result["qualified"] is False


def test_node_stop_restores_real_runtime_state(rules, completed_seed):
    engine, state, _prefix = screen._proposal_opening(rules, completed_seed)
    before = engine.state_hash(state)
    with pytest.raises(screen._ProposalScreenStopped, match="node_budget"):
        screen._proposal_root_values(engine, state, 1, lambda: None)
    assert engine.state_hash(state) == before


def test_real_policy_screen_retains_explicit_node_budget(rules, policy, completed_seed):
    result = screen.screen_runtime_proposals_v1(
        rules, seed=completed_seed, rollout_policy_id=policy["baselinePolicyId"],
        selection_policy=policy, maximum_nodes=1, maximum_seconds=10)
    assert result["status"] == "complete"
    assert [row["playIndex"] for row in result["decisions"]] == list(range(32))
    assert any(row["status"] == "node_budget" for row in result["decisions"])
    assert result["proposals"] == [] and result["qualified"] is False


@pytest.mark.parametrize("optional_projects", [False, True])
@pytest.mark.parametrize("play_index", range(4))
def test_refinement_preserves_prefix_and_never_qualifies(
    rules, root, policy, completed_seed, heuristic, monkeypatch, optional_projects, play_index,
):
    if optional_projects:
        from baloot_platform.core.rules import RuleBook
        rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    row = next(row for seed in range(16) for row in run(rules, policy, seed)["proposals"]
               if row["playIndex"] == play_index)
    assert len(row["prefix"]["playActions"]) == play_index
    assert len(row["prefix"].get("projectActions", [])) == (play_index + 1 if optional_projects else 0)
    assert ("projectActions" in row["prefix"]) is optional_projects
    before = copy.deepcopy(row)
    def worlds(info, engine, **kwargs):
        kwargs["cancel_check"]()
        return SimpleNamespace(states=[info._test_state] * 2, method="test_only", coverage_complete=True)
    # Exercise the real seed/prefix restoration; only world/value estimates are controlled.
    from baloot_platform.analysis.models import InformationState
    monkeypatch.setattr(InformationState, "from_complete", staticmethod(
        lambda state, observer: SimpleNamespace(_test_state=state)))
    monkeypatch.setattr("baloot_platform.analysis.hidden_deals.generate_hidden_deals", worlds)
    result = screen.refine_runtime_proposal_v1(
        rules, row, selection_policy=policy, maximum_nodes=100,
        maximum_seconds=10, maximum_worlds=2)
    assert result["status"] == "priority_candidate"
    assert result["qualified"] is False and result["proposalOnly"] is True
    assert result["prefixHash"] == row["prefixHash"] and row == before
    changed = copy.deepcopy(policy)
    changed["minimumNextBestRegret"] += 1
    with pytest.raises(ValueError, match="frozen rules"):
        screen.refine_runtime_proposal_v1(rules, row, selection_policy=changed,
            maximum_nodes=100, maximum_seconds=10, maximum_worlds=2)


def test_real_world_refinement_keeps_partial_coverage_explicit(
    rules, policy, completed_seed, heuristic,
):
    row = run(rules, policy, completed_seed)["proposals"][0]
    result = screen.refine_runtime_proposal_v1(
        rules, row, selection_policy=policy, maximum_nodes=100,
        maximum_seconds=10, maximum_worlds=1)
    assert result["status"] in {"worlds_incomplete", "single_world"}
    assert result["qualified"] is False


def test_policy_solver_stop_cannot_become_a_proposal(rules, policy, completed_seed, monkeypatch):
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
    def stopped(*args, **kwargs):
        raise InformationSetCompilationStopped("policy_action_not_optimal")
    monkeypatch.setattr("baloot_platform.research.ortools_research_executor._observer_safe_play_policy_artifact_v1", stopped)
    result = run(rules, policy, completed_seed)
    assert result["status"] == "policy_incomplete:policy_action_not_optimal"
    assert result["proposals"] == [] and result["qualified"] is False


@pytest.mark.parametrize("optional_projects", [False, True])
def test_proposed_project_history_normalizes_natively(root, policy, heuristic, optional_projects):
    from baloot_platform.core.rules import RuleBook
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
    profile = "saudi_project_optional_unratified_v1" if optional_projects else "saudi_federation_2023_v2"
    rules = RuleBook.load(root / f"configs/rules/{profile}.yaml")
    row = next(row for row in run(rules, policy, 722974)["proposals"] if row["playIndex"] >= 4)
    if optional_projects:
        assert len(row["prefix"]["projectActions"]) == 4
    else:
        assert row["prefix"]["projectDeclarations"]
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        ExactTracePrefixV1.build(row["prefix"]), decision_point={"kind": "play", "selector": {}},
        maximum_seconds=10)
    assert normalized.next_actor == row["actor"]
    assert row["action"] in {action["code"] for action in normalized.legal_actions}


def test_winner_preview_binds_baseline_both_ties_and_unknown(rules, monkeypatch):
    calls = []
    def roots(engine, world, nodes, check, **kwargs):
        calls.append((kwargs['root_action'], kwargs['utility']))
        return {kwargs['root_action']: 1 if kwargs['root_action'] == 'DA+ekka' else -1}, 1
    monkeypatch.setattr(screen, '_proposal_root_values', roots)
    result = screen._proposal_winner_preview(rules, [object()], 'DA+ekka', 'DK', 10, lambda: None)
    assert result['roundFlip'] is True
    assert calls == [('DA+ekka', 'round_win'), ('DK', 'round_win')]*2
    assert {row['tieRule'] for row in result['tieVariants']} == {'buyer_makes', 'buyer_loses'}
    assert result['qualified'] is False
    def tie_sensitive(engine, world, nodes, check, **kwargs):
        value = 1 if kwargs['root_action'] == 'DA' or engine.scoring_domain.buyer_tie_outcome == 'buyer_makes' else -1
        return {kwargs['root_action']: value}, 1
    monkeypatch.setattr(screen, '_proposal_root_values', tie_sensitive)
    result = screen._proposal_winner_preview(rules, [object()], 'DA', 'DK', 10, lambda: None)
    assert result['status'] == 'complete' and result['roundFlip'] is False
    def stopped(*args, **kwargs):
        raise screen._ProposalScreenStopped('node_budget')
    monkeypatch.setattr(screen, '_proposal_root_values', stopped)
    result = screen._proposal_winner_preview(rules, [object()], 'DA', 'DK', 1, lambda: None)
    assert result['status'] == 'unknown' and result['roundFlip'] is None


def test_round_win_preview_uses_winner_not_zero_margin():
    from baloot_platform.core.enums import Seat, Team
    state = SimpleNamespace(current_player=Seat.SOUTH,
        round_score=SimpleNamespace(awarded_points=(0, 0), winner=Team.WEST_EAST))
    class Engine:
        def legal_play_actions(self, state):
            return [SimpleNamespace(code='D7', card='D7', ekka=False)]
        def state_hash(self, state):
            return str(state.current_player)
        def apply(self, state, *args, **kwargs):
            previous = state.current_player
            state.current_player = None
            return previous
        def undo(self, state, token):
            state.current_player = token
    values, _ = screen._proposal_root_values(Engine(), state, 5, lambda: None, utility='round_win')
    assert values == {'D7': -1}
    assert state.current_player == Seat.SOUTH
