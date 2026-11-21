"""Perfect-information DD with engine-compiled terminal payoffs; no fallback."""
from __future__ import annotations

import ctypes
import hashlib
import math
import time
import threading
import weakref

from baloot_platform.core.cards import Card, iter_mask
from baloot_platform.core.enums import Contract, Seat, Team
from baloot_platform.core.models import LegalPlayActionV1, RoundState, Trick
from baloot_platform.core.scoring import score_round
from baloot_platform.core.scoring_ir import ScoringDomainIRV1
from baloot_platform.native.accelerator import (
    NativeAccelerator, NativeUnavailable, _PlayV2, _RulesV2, _SearchLimitsV3, _StateV2,
)


class _DDResult(ctypes.Structure):
    _fields_ = [('nodes', ctypes.c_uint64), ('values', ctypes.c_int32*64),
        ('status', ctypes.c_uint8), ('best_action', ctypes.c_uint8),
        ('action_count', ctypes.c_uint8), ('reserved', ctypes.c_uint8), ('actions', ctypes.c_uint8*64)]


class _DDTask(ctypes.Structure):
    _fields_ = [('state', _StateV2), ('payoffs', ctypes.POINTER(ctypes.c_int32)),
        ('payoff_count', ctypes.c_uint32), ('raw_total', ctypes.c_uint16),
        ('first_winner', ctypes.c_uint8), ('won_mask', ctypes.c_uint8)]


class DoubleDummySolver:
    def __init__(self, rules, library=None):
        if rules.project_domain.declaration_mode != 'mandatory':
            raise ValueError('DD requires the mandatory-declaration ruleset')
        self.rules, self.scoring = rules, ScoringDomainIRV1.from_rules(rules.data)
        self.path = NativeAccelerator._locate_library(library)
        self.library_hash = hashlib.sha256(self.path.read_bytes()).hexdigest()
        try:
            self.lib = ctypes.CDLL(str(self.path))
        except OSError as error:
            raise NativeUnavailable(f'DD native library could not load: {self.path}') from error
        if not hasattr(self.lib, 'baloot_double_dummy_v1'):
            raise NativeUnavailable('DD V1 is not built in this native library')
        version = getattr(self.lib, 'baloot_double_dummy_abi_version', None)
        if version is None:
            raise NativeUnavailable('DD library predates the frozen V1 calling convention')
        version.restype = ctypes.c_uint32
        if version() != 1:
            raise NativeUnavailable('DD calling convention version mismatch')
        for name, structure in [('baloot_sizeof_rules_v2', _RulesV2), ('baloot_sizeof_state_v2', _StateV2),
                ('baloot_sizeof_search_limits_v3', _SearchLimitsV3), ('baloot_sizeof_dd_result_v1', _DDResult)]:
            function = getattr(self.lib, name)
            function.restype = ctypes.c_uint32
            if function() != ctypes.sizeof(structure):
                raise NativeUnavailable(f'DD V1 ABI mismatch: {name}')
        self.lib.baloot_double_dummy_v1.argtypes = [ctypes.POINTER(_RulesV2), ctypes.POINTER(_StateV2),
            ctypes.POINTER(ctypes.c_int32), ctypes.c_uint32, ctypes.c_uint16, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8,
            ctypes.POINTER(_SearchLimitsV3), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(_DDResult)]
        self.lib.baloot_double_dummy_v1.restype = ctypes.c_int32
        self.native_rules = NativeAccelerator._build_transition_rules_v2(rules, rules.project_domain, self.scoring)
        self._payoff_cache = {}
        self._cache_lock = threading.RLock()
        self._cache_handle, self._cache_capacity, self._cache_finalizer = None, 0, None
        self._cache_api = all(hasattr(self.lib, name) for name in ('baloot_dd_cache_create_v1',
            'baloot_dd_cache_free_v1', 'baloot_dd_cache_stats_v1', 'baloot_double_dummy_cached_v1'))
        if self._cache_api:
            self.lib.baloot_dd_cache_create_v1.argtypes = [ctypes.c_uint32]
            self.lib.baloot_dd_cache_create_v1.restype = ctypes.c_void_p
            self.lib.baloot_dd_cache_free_v1.argtypes = [ctypes.c_void_p]
            self.lib.baloot_dd_cache_free_v1.restype = None
            self.lib.baloot_dd_cache_stats_v1.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
            self.lib.baloot_dd_cache_stats_v1.restype = ctypes.c_int32
            self.lib.baloot_double_dummy_cached_v1.argtypes = [ctypes.c_void_p]+self.lib.baloot_double_dummy_v1.argtypes
            self.lib.baloot_double_dummy_cached_v1.restype = ctypes.c_int32
        self._batch_api = self._cache_api and all(hasattr(self.lib, name) for name in (
            'baloot_sizeof_dd_task_v1', 'baloot_double_dummy_batch_v1'))
        if self._batch_api:
            self.lib.baloot_sizeof_dd_task_v1.restype = ctypes.c_uint32
            if self.lib.baloot_sizeof_dd_task_v1() != ctypes.sizeof(_DDTask):
                raise NativeUnavailable('DD batch task ABI mismatch')
            self.lib.baloot_double_dummy_batch_v1.argtypes = [ctypes.c_void_p, ctypes.POINTER(_RulesV2),
                ctypes.POINTER(_DDTask), ctypes.c_uint32, ctypes.c_uint8, ctypes.POINTER(_SearchLimitsV3),
                ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(_DDResult)]
            self.lib.baloot_double_dummy_batch_v1.restype = ctypes.c_int32

    def close(self):
        with self._cache_lock:
            if self._cache_finalizer is not None:
                self._cache_finalizer()
            self._cache_handle, self._cache_capacity, self._cache_finalizer = None, 0, None

    def _native_solve(self, native, table, total, first, mask, all_roots, limits, cancelled, result):
        arguments = (ctypes.byref(self.native_rules), ctypes.byref(native), table, len(table), total,
            first, mask, int(all_roots), ctypes.byref(limits), ctypes.byref(cancelled), ctypes.byref(result))
        if not getattr(self, '_cache_api', False):
            return self.lib.baloot_double_dummy_v1(*arguments), dict(enabled=False)
        with self._cache_lock:
            capacity = limits.maximum_cache_entries
            if not capacity:
                self.close()
                return self.lib.baloot_double_dummy_v1(*arguments), dict(enabled=False)
            reused = self._workspace(capacity)
            before, after = (ctypes.c_uint64*4)(), (ctypes.c_uint64*4)()
            if self.lib.baloot_dd_cache_stats_v1(self._cache_handle, before) != 0:
                raise RuntimeError('DD cache telemetry failed')
            code = self.lib.baloot_double_dummy_cached_v1(self._cache_handle, *arguments)
            if self.lib.baloot_dd_cache_stats_v1(self._cache_handle, after) != 0:
                raise RuntimeError('DD cache telemetry failed')
            return code, dict(enabled=True, reused=reused, entriesBefore=before[0], entriesAfter=after[0],
                hits=after[1]-before[1], lookups=after[2]-before[2], resets=after[3]-before[3])

    def _workspace(self, capacity):
        if capacity != self._cache_capacity:
            self.close()
        reused = self._cache_handle is not None
        if not reused:
            self._cache_handle = self.lib.baloot_dd_cache_create_v1(capacity)
            if not self._cache_handle:
                raise MemoryError('DD cache allocation failed')
            self._cache_capacity = capacity
            self._cache_finalizer = weakref.finalize(self, self.lib.baloot_dd_cache_free_v1, self._cache_handle)
        return reused

    def _payoffs(self, state, objective):
        domain = self.rules.project_domain
        hands = {seat: tuple(iter_mask(state.initial_hands[int(seat)])) for seat in Seat}
        detected = domain.detect(hands, state.contract, state.dealer)
        outcomes = []
        for first in Seat:
            projects, _ = domain.resolve_declarations(detected, dealer=state.dealer,
                trump_suit=state.trump_suit, second_trick_leader=first)
            baloot = domain.complete_baloot_teams(hands, projects, state.contract, state.trump_suit)
            outcomes.append((projects, baloot))
        key = (state.contract.value, int(state.declarer), state.escalation,
            tuple(tuple(sorted(row.items())) for row in state.escalation_history), objective,
            tuple((domain.project_points(projects), tuple(sorted(int(t) for t in baloot))) for projects, baloot in outcomes))
        if key in self._payoff_cache:
            return self._payoff_cache[key]
        total = self.scoring.raw_card_totals[state.contract.effective.value]
        shell = RoundState(self.rules.sha256, 0, 1, state.dealer)
        shell.contract, shell.declarer = state.contract, state.declarer
        shell.escalation, shell.escalation_history = state.escalation, state.escalation_history
        values = []
        for projects, baloot in outcomes:
            shell.projects, shell.baloot_teams = projects, set(baloot)
            for sweep in range(3):
                for raw in range(total+1):
                    raw = total if sweep == 1 else 0 if sweep == 2 else raw
                    if objective == 'raw_point_margin':
                        values.append(raw-(total-raw))
                        continue
                    # score_round is the one scoring authority. Only the exact
                    # native game tree can select a reachable table entry.
                    last = 0 if raw >= self.scoring.last_trick_bonus_raw else 1
                    winners = ([0]*8 if sweep == 1 else [1]*8 if sweep == 2 else
                               [0, 1, 0, 1, 0, 1, 1-last, last])
                    if sweep:
                        last = winners[-1]
                    raw_cards = [raw, total-raw]
                    raw_cards[last] -= self.scoring.last_trick_bonus_raw
                    tricks = [Trick(i, Seat(winners[i]), winner=Seat(winners[i])) for i in range(8)]
                    for side in (0, 1):
                        recipient = next((t for t in tricks if int(t.winner.team) == side), None)
                        if recipient is not None:
                            recipient.raw_points = raw_cards[side]
                    shell.completed_tricks = tricks
                    score = score_round(shell, self.scoring, domain)
                    values.append(score.awarded_points[0]-score.awarded_points[1])
        table = (ctypes.c_int32*len(values))(*values)
        if len(self._payoff_cache) >= 256:
            self._payoff_cache.pop(next(iter(self._payoff_cache)))
        result = table, total, hashlib.sha256(bytes(table)).hexdigest()
        self._payoff_cache[key] = result
        return result

    def _prepare(self, state, objective):
        if objective not in {'awarded_point_margin', 'raw_point_margin'}:
            raise ValueError('unsupported DD objective')
        if (state.rules_hash != self.rules.sha256 or state.current_player is None
                or state.current_trick is None or state.declarer is None or len(state.completed_tricks) >= 8):
            raise ValueError('DD requires an active rules-bound engine state')
        state.validate_conservation()
        table, total, table_hash = self._payoffs(state, objective)
        native = _StateV2()
        native.abi_version = 2
        native.contract = 1 if state.contract == Contract.HOKUM else 0
        native.trump_suit = int(state.trump_suit) if state.trump_suit is not None else -1
        native.current_player, native.trick_size = int(state.current_player), len(state.current_trick.plays)
        native.locked, native.trump_broken = int(state.locked), int(state.trump_broken)
        native.completed_tricks = len(state.completed_tricks)
        native.first_leader = int(state.first_leader)
        for index in range(4):
            native.hands[index] = state.hands[index]
        raw = NativeAccelerator._raw_points(state)
        native.raw_points[0], native.raw_points[1] = raw
        for index, play in enumerate(state.current_trick.plays):
            native.trick[index] = _PlayV2(int(play.seat), play.card.id, int(play.ekka), 0)
        first = int(state.completed_tricks[0].winner) if state.completed_tricks else 4
        mask = 0
        for trick in state.completed_tricks:
            mask |= 1 << int(trick.winner.team)
        return native, table, total, table_hash, first, mask

    @staticmethod
    def _roots(state, result):
        return [dict(action=LegalPlayActionV1(Card(i%32), i>=32).code,
            value=int(result.values[i])*(1 if state.current_player.team == Team.NORTH_SOUTH else -1))
            for i in range(64) if result.actions[i]]

    def solve(self, state, *, objective='awarded_point_margin', maximum_seconds=5,
              maximum_nodes=10000000, maximum_cache_entries=1000000, cancel_check=None, all_root_values=False):
        """Use only the node budget when maximum_seconds is None."""
        started, cpu = time.monotonic(), time.process_time()
        if ((maximum_seconds is not None and (not math.isfinite(maximum_seconds) or not 0 < maximum_seconds <= 5))
                or type(maximum_nodes) is not int or not 1 <= maximum_nodes < 2**64
                or type(maximum_cache_entries) is not int or not 0 <= maximum_cache_entries <= 1000000):
            raise ValueError('DD requires finite single-worker budgets')
        native, table, total, table_hash, first, mask = self._prepare(state, objective)
        remaining = None if maximum_seconds is None else maximum_seconds-(time.monotonic()-started)
        result = _DDResult()
        cache_info = dict(enabled=False)
        if (remaining is not None and remaining <= 0) or (cancel_check is not None and cancel_check()):
            status = 'incomplete'
        else:
            limits = _SearchLimitsV3()
            limits.maximum_nodes, limits.maximum_cache_entries = maximum_nodes, maximum_cache_entries
            limits.maximum_milliseconds = 0 if remaining is None else max(1, int(remaining*1000))
            cancelled, stop, poller = NativeAccelerator._start_cancellation_poller(cancel_check)
            try:
                code, cache_info = self._native_solve(native, table, total, first, mask,
                    all_root_values, limits, cancelled, result)
            finally:
                NativeAccelerator._stop_cancellation_poller(stop, poller)
            if code != 0:
                raise ValueError(f'DD rejected its engine input: {code}')
            if result.status == 3:
                raise RuntimeError('DD internal model error')
            status = 'exact' if (result.status == 0 and (maximum_seconds is None or time.monotonic()-started <= maximum_seconds)
                and not (cancel_check is not None and cancel_check())) else 'incomplete'
        roots = []
        if status == 'exact':
            roots = self._roots(state, result)
        stop_reason = (None if status == 'exact' else 'cancelled' if result.status == 2
            or cancel_check is not None and cancel_check() else
            'maximum_nodes' if result.nodes > maximum_nodes else 'maximum_seconds')
        return dict(status=status, nodes=int(result.nodes), rootActions=roots,
            nativeCache=cache_info,
            stopReason=stop_reason,
            limits=dict(maximumSeconds=maximum_seconds, maximumNodes=maximum_nodes,
                maximumCacheEntries=maximum_cache_entries, allRootValues=all_root_values),
            nativeInputHash=hashlib.sha256(bytes(self.native_rules)+bytes(native)+bytes(table)).hexdigest(),
            rootCoverage='all_values_exact' if all_root_values else 'optimal_card_proved_by_full_tree_bounds',
            selectedAction=next((r['action'] for r in roots if r['value'] == max(x['value'] for x in roots)), None),
            objective=objective, method='native_dd_mtdf_v1', scoringTableHash=table_hash,
            libraryHash=self.library_hash, rulesHash=self.rules.sha256,
            wallSeconds=time.monotonic()-started, cpuSeconds=time.process_time()-cpu)

    def solve_many(self, states, *, objective='awarded_point_margin', maximum_seconds=5,
                   maximum_nodes=10000000, maximum_cache_entries=1000000, maximum_threads=4, cancel_check=None):
        """One native cohort call; no Python workers or per-alternative calls."""
        started, cpu = time.monotonic(), time.process_time()
        if not self._batch_api:
            raise NativeUnavailable('DD cohort API is not built; no serial-policy fallback')
        states = tuple(states)
        if (not 1 <= len(states) <= 256 or type(maximum_threads) is not int or not 1 <= maximum_threads <= 4
                or not math.isfinite(maximum_seconds) or not 0 < maximum_seconds <= 60
                or type(maximum_nodes) is not int or not 1 <= maximum_nodes < 2**64
                or type(maximum_cache_entries) is not int or not 1 <= maximum_cache_entries <= 1000000):
            raise ValueError('DD cohort requires bounded worlds, threads, time, nodes and shared cache')
        # Keep every table alive even if preparing later worlds evicts a payoff
        # cache entry. No true-deal state is consulted to prepare a sampled one.
        prepared = []
        for state in states:
            if time.monotonic()-started >= maximum_seconds or (cancel_check is not None and cancel_check()):
                return dict(status='incomplete', worlds=[], stopReason='preparation_budget',
                    wallSeconds=time.monotonic()-started, cpuSeconds=time.process_time()-cpu)
            prepared.append(self._prepare(state, objective))
        tasks = (_DDTask*len(states))()
        for task, (native, table, total, _, first, mask) in zip(tasks, prepared, strict=True):
            task.state, task.payoffs, task.payoff_count = native, table, len(table)
            task.raw_total, task.first_winner, task.won_mask = total, first, mask
        results = (_DDResult*len(states))()
        remaining = maximum_seconds-(time.monotonic()-started)
        if remaining <= 0 or (cancel_check is not None and cancel_check()):
            return dict(status='incomplete', worlds=[], stopReason='preparation_budget',
                wallSeconds=time.monotonic()-started, cpuSeconds=time.process_time()-cpu)
        limits = _SearchLimitsV3()
        limits.maximum_nodes, limits.maximum_cache_entries = maximum_nodes, maximum_cache_entries
        limits.maximum_milliseconds = max(1, int(remaining*1000))
        cancelled, stop, poller = NativeAccelerator._start_cancellation_poller(cancel_check)
        try:
            with self._cache_lock:
                reused = self._workspace(maximum_cache_entries)
                before, after = (ctypes.c_uint64*4)(), (ctypes.c_uint64*4)()
                if self.lib.baloot_dd_cache_stats_v1(self._cache_handle, before) != 0:
                    raise RuntimeError('DD cache telemetry failed')
                code = self.lib.baloot_double_dummy_batch_v1(self._cache_handle, ctypes.byref(self.native_rules),
                    tasks, len(states), maximum_threads, ctypes.byref(limits), ctypes.byref(cancelled), results)
                if self.lib.baloot_dd_cache_stats_v1(self._cache_handle, after) != 0:
                    raise RuntimeError('DD cache telemetry failed')
        finally:
            NativeAccelerator._stop_cancellation_poller(stop, poller)
        if code != len(states):
            raise ValueError(f'DD rejected its cohort input: {code}')
        if any(result.status == 3 for result in results):
            raise RuntimeError('DD cohort internal model error')
        worlds = []
        for state, data, result in zip(states, prepared, results, strict=True):
            native, table, _, table_hash, _, _ = data
            worlds.append(dict(status='exact' if result.status == 0 else 'incomplete', nodes=int(result.nodes),
                rootActions=self._roots(state, result) if result.status == 0 else [],
                stopReason=None if result.status == 0 else 'maximum_nodes' if result.nodes > maximum_nodes else 'cohort_budget',
                nativeInputHash=hashlib.sha256(bytes(self.native_rules)+bytes(native)+bytes(table)).hexdigest(),
                scoringTableHash=table_hash))
        complete = (all(world['status'] == 'exact' for world in worlds)
            and time.monotonic()-started <= maximum_seconds and not (cancel_check is not None and cancel_check()))
        return dict(status='exact' if complete else 'incomplete', worlds=worlds,
            method='native_dd_cohort_v1', objective=objective, maximumThreads=maximum_threads,
            limits=dict(maximumSeconds=maximum_seconds, maximumNodesPerWorld=maximum_nodes,
                maximumCacheEntries=maximum_cache_entries),
            nativeCache=dict(enabled=True, reused=reused, entriesBefore=before[0], entriesAfter=after[0],
                hits=after[1]-before[1], lookups=after[2]-before[2], resets=after[3]-before[3]),
            libraryHash=self.library_hash, rulesHash=self.rules.sha256,
            wallSeconds=time.monotonic()-started, cpuSeconds=time.process_time()-cpu)
