from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json

import pytest

from baloot_platform.analysis.search import (
    SearchResult,
    SearchTimeBudgetExceeded,
    exact_team_value,
    heuristic_rollout_result,
    terminal_margin,
)
from baloot_platform.analysis.search_backend import PythonReferenceSearchBackend
from baloot_platform.core.bots import BotProfile, HeuristicBot
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, Seat, Suit
from baloot_platform.core.models import RoundState
from baloot_platform.core.scenario import ScenarioRequest
from baloot_platform.native.accelerator import NativeAccelerator, NativeUnavailable


def _endgame(rules, *, remaining: int = 5):
    engine = BalootEngine(rules)
    state = engine.create_round(seed=24001, dealer=Seat.EAST)
    engine.shuffle_and_initial_deal(state, cut_position=9)
    engine.bid_and_complete_deal(
        state,
        forced_contract=Contract.HOKUM,
        forced_buyer=Seat.SOUTH,
        forced_trump=Suit.CLUBS,
    )
    engine.prepare_projects(state)
    bot = HeuristicBot(engine, BotProfile(deterministic_seed=24001))
    while sum(mask.bit_count() for mask in state.hands) > remaining:
        engine.play_card(state, bot.choose(state))
    return engine, state


def _clone_oracle(state, engine, team):
    if state.current_player is None:
        return terminal_margin(state, team)
    values = []
    for card in engine.legal_actions(state):
        child = state.clone()
        engine.play_card(child, card)
        values.append(_clone_oracle(child, engine, team))
    if state.current_player.team == team:
        return max(values)
    return min(values)


def test_v21_exact_search_matches_clone_oracle_and_restores_state(rules):
    engine, state = _endgame(rules)
    team = Seat.SOUTH.team
    expected = _clone_oracle(state, engine, team)
    before = engine.canonical_state(state)
    first = exact_team_value(
        state,
        engine,
        team,
        max_nodes=100_000,
        max_seconds=10.0,
    )
    second = exact_team_value(
        state,
        engine,
        team,
        max_nodes=100_000,
        max_seconds=10.0,
    )
    assert first == second
    assert first.value == expected
    assert first.exact is True
    assert first.completion_status == "exact"
    assert len(first.principal_variation) == 5
    assert engine.canonical_state(state) == before


def test_v21_exact_search_uses_reversible_kernel_without_round_clones(
    rules,
    monkeypatch,
):
    engine, state = _endgame(rules, remaining=4)

    def reject_clone(_state):
        raise AssertionError("exact search must not clone complete rounds")

    monkeypatch.setattr(RoundState, "clone", reject_clone)
    result = exact_team_value(
        state,
        engine,
        Seat.SOUTH.team,
        max_nodes=100_000,
        max_seconds=10.0,
        max_cache_entries=0,
    )
    assert result.completion_status == "exact"


@pytest.mark.real_clock
def test_v21_exact_search_never_labels_partial_results_exact(rules, monkeypatch):
    engine, state = _endgame(rules)
    before = engine.canonical_state(state)
    node_limited = exact_team_value(
        state,
        engine,
        Seat.SOUTH.team,
        max_nodes=1,
        max_seconds=10.0,
    )
    depth_limited = exact_team_value(
        state,
        engine,
        Seat.SOUTH.team,
        max_nodes=100_000,
        max_seconds=10.0,
        max_depth=1,
    )
    cancelled = exact_team_value(
        state,
        engine,
        Seat.SOUTH.team,
        max_nodes=100_000,
        max_seconds=10.0,
        cancel_check=lambda: True,
    )
    clock = iter((100.0, 101.0))
    with monkeypatch.context() as controlled:
        controlled.setattr(
            "baloot_platform.analysis.search.monotonic",
            lambda: next(clock),
        )
        with pytest.raises(
            SearchTimeBudgetExceeded,
            match="without canonical output",
        ):
            exact_team_value(
                state,
                engine,
                Seat.SOUTH.team,
                max_nodes=100_000,
                max_seconds=0.5,
            )
    assert node_limited.exact is False
    assert node_limited.completion_status == "budget_exhausted"
    assert depth_limited.exact is False
    assert depth_limited.completion_status == "budget_exhausted"
    assert cancelled.exact is False
    assert cancelled.completion_status == "cancelled"
    assert engine.canonical_state(state) == before


@pytest.mark.real_clock
def test_python_reference_backend_propagates_wall_clock_exhaustion(rules, monkeypatch):
    engine, state = _endgame(rules)
    before = engine.canonical_state(state)
    clock = iter((200.0, 201.0))
    monkeypatch.setattr(
        "baloot_platform.analysis.search.monotonic",
        lambda: next(clock),
    )

    with pytest.raises(SearchTimeBudgetExceeded):
        PythonReferenceSearchBackend().search(
            state,
            engine,
            Seat.SOUTH.team,
            max_nodes=100_000,
            max_seconds=0.5,
            max_depth=None,
            max_cache_entries=100_000,
            cancel_check=None,
        )

    assert "fail_closed_wall_clock" in (
        PythonReferenceSearchBackend.descriptor.capabilities
    )
    assert engine.canonical_state(state) == before


def test_heuristic_complete_is_supported_but_can_never_be_exact(rules):
    engine, state = _endgame(rules, remaining=4)
    result = heuristic_rollout_result(state, engine, Seat.SOUTH.team, 31)
    assert result.completion_status == "heuristic_complete"
    assert result.exact is False
    with pytest.raises(ValueError, match="exact flag"):
        SearchResult(
            value=0.0,
            own_points=0.0,
            opponent_points=0.0,
            exact=True,
            nodes=0,
            completion_status="heuristic_complete",
        )


def test_v21_exact_search_fails_safely_on_memory_exhaustion(
    rules,
    monkeypatch,
):
    engine, state = _endgame(rules, remaining=4)
    before = engine.canonical_state(state)

    def exhaust_memory(_bot, _state):
        raise MemoryError

    monkeypatch.setattr(HeuristicBot, "choose", exhaust_memory)
    result = exact_team_value(
        state,
        engine,
        Seat.SOUTH.team,
        max_nodes=100_000,
        max_seconds=10.0,
    )
    assert result.completion_status == "failed"
    assert result.exact is False
    assert result.principal_variation == ()
    assert engine.canonical_state(state) == before


def test_v21_exact_search_is_stable_under_concurrent_invocation(rules):
    engine, state = _endgame(rules, remaining=4)
    before = engine.canonical_state(state)

    def solve_once(_index):
        private_state = state.clone()
        return exact_team_value(
            private_state,
            engine,
            Seat.SOUTH.team,
            max_nodes=100_000,
            max_seconds=10.0,
        )

    expected = solve_once(0)
    for workers in (1, 2, 4):
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(solve_once, range(4)))
        assert results == [expected] * 4
    assert engine.canonical_state(state) == before


def test_v21_exact_search_resource_budgets_are_governed():
    request = ScenarioRequest.governed()
    assert request.exact_search_max_depth == 32
    assert request.exact_search_max_cache_entries == 100_000


def test_v21_exact_search_matches_frozen_independent_oracles(root, rules):
    fixture = json.loads(
        (root / "tests/fixtures/synthetic-exact-endgames.json").read_text(
            encoding="utf-8"
        )
    )
    path = root / "scripts/generate_exact_endgames.py"
    spec = importlib.util.spec_from_file_location(
        "generate_exact_endgames",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    engine = BalootEngine(rules)
    assert len(fixture["cases"]) == 12
    for case in fixture["cases"]:
        state = module.prepare(engine, case)
        assert engine.state_hash(state) == case["stateSha256"]
        result = exact_team_value(
            state,
            engine,
            Seat.SOUTH.team,
            max_nodes=100_000,
            max_seconds=10.0,
            max_depth=case["remainingCards"],
            max_cache_entries=100_000,
        )
        assert result.completion_status == "exact"
        assert result.value == case["expectedMargin"]
        assert engine.state_hash(state) == case["stateSha256"]


def test_native_v3_exact_search_matches_python_on_frozen_endgames(root, rules):
    try:
        native = NativeAccelerator(rules)
    except (NativeUnavailable, OSError):
        pytest.skip("native V3 library is not built in this test lane")
    if not native.supports_v3:
        pytest.skip("native library does not expose exact-search ABI V3")
    fixture = json.loads(
        (root / "tests/fixtures/synthetic-exact-endgames.json").read_text(encoding="utf-8")
    )
    path = root / "scripts/generate_exact_endgames.py"
    spec = importlib.util.spec_from_file_location("native_exact_endgames", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    engine = BalootEngine(rules)
    for case in fixture["cases"]:
        state = module.prepare(engine, case)
        expected = exact_team_value(
            state,
            engine,
            Seat.SOUTH.team,
            max_nodes=100_000,
            max_seconds=10.0,
            max_depth=case["remainingCards"],
            max_cache_entries=100_000,
        )
        actual = native.exact_team_value(
            state,
            engine,
            Seat.SOUTH.team,
            max_nodes=100_000,
            max_seconds=10.0,
            max_depth=case["remainingCards"],
            max_cache_entries=100_000,
        )
        assert actual.completion_status == "exact"
        assert actual.value == expected.value
        assert actual.own_points == expected.own_points
        assert actual.opponent_points == expected.opponent_points
        assert actual.principal_variation == expected.principal_variation
        assert actual.contract_succeeded == expected.contract_succeeded
        assert actual.score_after_round == expected.score_after_round
        assert actual.match_winner == expected.match_winner


def test_native_v3_exact_search_honors_delayed_cancellation(rules):
    try:
        native = NativeAccelerator(rules)
    except (NativeUnavailable, OSError):
        pytest.skip("native V3 library is not built in this test lane")
    if not native.supports_v3:
        pytest.skip("native library does not expose exact-search ABI V3")

    engine, state = _endgame(rules, remaining=24)
    checks = 0

    def cancel_after_native_call_starts() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 2

    result = native.exact_team_value(
        state,
        engine,
        Seat.SOUTH.team,
        max_nodes=100_000_000,
        max_seconds=5.0,
        max_depth=24,
        max_cache_entries=100_000,
        cancel_check=cancel_after_native_call_starts,
    )

    assert checks == 2
    assert result.completion_status == "cancelled"
    assert result.exact is False


def test_native_v3_batched_rollouts_match_reference_with_unresolved_projects(rules):
    try:
        native = NativeAccelerator(rules)
    except (NativeUnavailable, OSError):
        pytest.skip("native V3 library is not built in this test lane")
    if not native.supports_v3:
        pytest.skip("native library does not expose rollout ABI V3")

    engine = BalootEngine(rules)
    state = engine.create_round(seed=24001, dealer=Seat.EAST)
    engine.shuffle_and_initial_deal(state, cut_position=9)
    engine.bid_and_complete_deal(
        state,
        forced_contract=Contract.HOKUM,
        forced_buyer=Seat.SOUTH,
        forced_trump=Suit.CLUBS,
    )
    engine.prepare_projects(state)
    assert state.projects and not any(project.scoring for project in state.projects)
    before = engine.canonical_state(state)
    cards = tuple(engine.legal_moves(state)[:3])
    actual = native.heuristic_rollout_batch(
        tuple(state for _ in cards),
        cards,
        state.current_player.team,
    )
    expected = []
    for card in cards:
        child = state.clone()
        child.events = []
        engine.play_card(child, card, ekka=False, emit_events=False)
        expected.append(
            heuristic_rollout_result(
                child,
                engine,
                state.current_player.team,
                24001 + card.id,
            )
        )
    for accelerated, reference in zip(actual, expected, strict=True):
        assert accelerated.completion_status == "heuristic_complete"
        assert reference.completion_status == "heuristic_complete"
        assert not accelerated.exact and not reference.exact
        assert accelerated.value == reference.value
        assert accelerated.own_points == reference.own_points
        assert accelerated.opponent_points == reference.opponent_points
        assert accelerated.principal_variation == reference.principal_variation
        assert accelerated.contract_succeeded == reference.contract_succeeded
        assert accelerated.score_after_round == reference.score_after_round
        assert accelerated.match_winner == reference.match_winner
    assert engine.canonical_state(state) == before


def test_v21_exact_endgame_fixture_generator_is_current(root):
    path = root / "scripts/generate_exact_endgames.py"
    spec = importlib.util.spec_from_file_location(
        "generate_exact_endgames_check",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = json.dumps(
        module.generate(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    assert (
        root / "tests/fixtures/synthetic-exact-endgames.json"
    ).read_text(encoding="utf-8") == expected
