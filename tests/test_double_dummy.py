import copy
import ctypes
import threading
from types import SimpleNamespace

import pytest

from baloot_platform.analysis.double_dummy import DoubleDummySolver
from baloot_platform.core.bots import BotProfile, HeuristicBot
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, Seat
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.scoring_ir import ScoringDomainIRV1
from baloot_platform.native.accelerator import NativeAccelerator


@pytest.mark.parametrize("tie", ["buyer_makes", "buyer_loses"])
def test_dd_payoff_table_comes_from_the_engine_for_project_and_baloot_rounds(root, tie):
    data = copy.deepcopy(RuleBook.load(root / "configs/rules/saudi_federation_2023_v7.yaml").data)
    data["scoring"]["contract"]["buyer_tie_outcome"] = tie
    rules = RuleBook(data)
    solver = DoubleDummySolver.__new__(DoubleDummySolver)
    solver.rules, solver.scoring, solver._payoff_cache = rules, ScoringDomainIRV1.from_rules(data), {}
    engine = BalootEngine(rules)
    seen_project, seen_baloot = False, False
    for mode in (Contract.SUN, Contract.HOKUM):
        for seed in range(20):
            state = engine.create_round(seed=seed, dealer=Seat.EAST)
            engine.shuffle_and_initial_deal(state)
            assert engine.bid_and_complete_deal(state, forced_contract=mode, forced_buyer=Seat.NORTH)
            engine.prepare_projects(state)
            table, total, digest = solver._payoffs(state, 'awarded_point_margin')
            raw_table, _, _ = solver._payoffs(state, 'raw_point_margin')
            assert len(digest) == 64
            bot = HeuristicBot(engine, BotProfile(seed))
            while state.current_player is not None:
                engine.play_card(state, bot.choose(state))
            first = int(state.completed_tricks[0].winner)
            teams = {trick.winner.team.value for trick in state.completed_tricks}
            sweep = 0 if len(teams) == 2 else 1+next(iter(teams))
            raw = state.round_score.raw_card_points
            index = (first*3+sweep)*(total+1)+raw[0]
            assert table[index] == state.round_score.awarded_points[0]-state.round_score.awarded_points[1]
            assert raw_table[index] == raw[0]-raw[1]
            seen_project |= bool(sum(state.round_score.project_points))
            seen_baloot |= bool(sum(state.round_score.baloot_points))
    assert seen_project and seen_baloot


def test_dd_adapter_never_exposes_partial_root_values(root):
    from baloot_platform.analysis.double_dummy import _DDResult
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v7.yaml")
    engine = BalootEngine(rules)
    state = engine.create_round(seed=92, dealer=Seat.EAST)
    engine.shuffle_and_initial_deal(state)
    engine.bid_and_complete_deal(state, forced_contract=Contract.SUN, forced_buyer=Seat.NORTH)
    engine.prepare_projects(state)
    solver = DoubleDummySolver.__new__(DoubleDummySolver)
    solver.rules, solver.scoring, solver._payoff_cache = rules, ScoringDomainIRV1.from_rules(rules.data), {}
    solver.native_rules = NativeAccelerator._build_transition_rules_v2(rules, rules.project_domain, solver.scoring)
    solver.library_hash = "0"*64
    def incomplete(*args):
        result = ctypes.cast(args[-1], ctypes.POINTER(_DDResult)).contents
        result.status, result.action_count, result.best_action = 1, 1, 7
        result.actions[7], result.values[7] = 1, 10000
        return 0
    solver.lib = SimpleNamespace(baloot_double_dummy_v1=incomplete)
    result = solver.solve(state)
    assert result["status"] == "incomplete"
    assert result["rootActions"] == [] and result["selectedAction"] is None


def test_dd_cache_is_bounded_resized_and_closed_without_double_free():
    from baloot_platform.analysis.double_dummy import _DDResult
    from baloot_platform.native.accelerator import _RulesV2, _StateV2, _SearchLimitsV3
    solver = DoubleDummySolver.__new__(DoubleDummySolver)
    solver.native_rules, solver._cache_api = _RulesV2(), True
    solver._cache_lock = threading.RLock()
    solver._cache_handle, solver._cache_capacity, solver._cache_finalizer = None, 0, None
    created, freed, counters = [], [], {}
    def create(capacity):
        handle = len(created)+1
        created.append(capacity)
        counters[handle] = [0, 0, 0, 0]
        return handle
    def stats(handle, output):
        for index, value in enumerate(counters[handle]):
            output[index] = value
        return 0
    def solve(handle, *args):
        counters[handle] = [1, counters[handle][1]+1, counters[handle][2]+2, 0]
        return 0
    solver.lib = SimpleNamespace(baloot_dd_cache_create_v1=create, baloot_dd_cache_free_v1=freed.append,
        baloot_dd_cache_stats_v1=stats, baloot_double_dummy_cached_v1=solve,
        baloot_double_dummy_v1=lambda *args: 0)
    limits, result = _SearchLimitsV3(), _DDResult()
    limits.maximum_cache_entries = 100
    args = (_StateV2(), (ctypes.c_int32*12)(), 0, 4, 0, True, limits, ctypes.c_uint32(0), result)
    _, first = solver._native_solve(*args)
    _, second = solver._native_solve(*args)
    assert first['reused'] is False and second['reused'] is True
    assert second['entriesBefore'] == second['entriesAfter'] == 1 and second['hits'] == 1
    assert created == [100] and freed == []
    limits.maximum_cache_entries = 50
    solver._native_solve(*args)
    assert created == [100, 50] and freed == [1]
    solver.close()
    solver.close()
    assert freed == [1, 2] and solver._cache_handle is None
    limits.maximum_cache_entries = 0
    assert solver._native_solve(*args) == (0, {'enabled': False})
