from __future__ import annotations

import ast
import copy
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
import inspect
import itertools
import json
import multiprocessing
import multiprocessing.connection
import os
from threading import Barrier, Event, Thread, enumerate as active_threads
from pathlib import Path
import textwrap
import time

import pytest

from deterministic_work import set_monotonic

pytest.importorskip("ortools")

from baloot_platform.core.cards import Card, FULL_DECK, cards_to_mask
from baloot_platform.core.dealing import DealLayoutV1
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, GamePhase, Seat, Suit, Team
from baloot_platform.core.exact_state_view import (
    build_canonical_observer_information_v1,
    canonical_observer_information_hash_v1,
)
from baloot_platform.core.models import RoundState, Trick
from baloot_platform.research.ortools_information_set_proof import (
    FiniteInformationNodeV1,
    FiniteInformationSetProblemV1,
    FiniteInformationWorldV1,
    InformationSetProofBudgetV1,
    OrToolsActorInformationSetProofV1,
    _solve_cp_sat_with_cancel_v1,
)
from baloot_platform.research.ortools_information_set_compiler import (
    ActorInformationGraphWorldV1,
    ActorInformationSetCompilationV1,
    ActorInformationSetGraphV1,
    InformationSetCompilationStopped,
    OrToolsActorInformationSetCompilerV1,
    _integral_success_threshold,
    hidden_allocation_upper_bound_v1,
    prove_actor_information_set_v1,
)
from baloot_platform.research.ortools_exact_state import (
    ExactNormalizationStopped,
    OrToolsExactNormalizationReceiptV1,
    OrToolsExactStateNormalizerV1,
    OrToolsNormalizedExactStateV1,
)
from baloot_platform.research.scenario_contracts import (
    ExactTracePrefixV1,
    ScenarioProofRequestV1,
    canonical_sha256,
)
from baloot_platform.research.ortools_proof import (
    ProofGraphEdgeV1,
    ProofGraphNodeV1,
    ProofGraphV1,
)


def _world_parent_death_probe(connection):
    connection.send(os.getpid())
    connection.recv()
    os._exit(0)


@pytest.mark.real_clock
@pytest.mark.parametrize("admissible", [None, ("SA",)])
def test_native_interleaved_opening_prefers_symbolic_selection(root, record_property, monkeypatch, admissible):
    import yaml
    from baloot_platform.core.rules import RuleBook
    from baloot_platform.research.scenario_contracts import NativeObservedProofLimitsV1

    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    prefix = ExactTracePrefixV1.build({"schemaVersion": 1, "dealer": "east", "deckOrder": [
        7, 19, 21, 3, 6, 17, 8, 9, 11, 5, 10, 12, 22, 30, 23, 31,
        14, 15, 24, 26, 20, 18, 29, 0, 16, 28, 1, 2, 4, 13, 25, 27],
        "auctionActions": [*({"actor": seat, "action": "pass"}
            for seat in ("north", "west", "south", "east")), {"actor": "north", "action": "sun"}],
        "escalationActions": [], "projectDeclarations": [], "playActions": [],
        "projectActions": [{"actor": "north", "action": "declare", "projectId": "PROJECT-1460484D8F1BE0CD"}]})
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(prefix)
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    request = ScenarioProofRequestV1.build({"schemaVersion": 1, "mode": "bounded",
        "visibility": "actor_information_set", "strategyClass": "pure_deterministic",
        "claim": "outcome_reachable", "targetTeam": "north_south", "utility": "awarded_point_margin",
        "successThreshold": 0, "budgetProfile": "bounded_v1", "horizon": {"unit": "tricks", "value": 1},
        "boundaryEvaluator": "current_raw_point_margin_v1"})
    profile = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(
        encoding="utf-8"))["scenarioResearchV4"]["proofBudgets"]["profiles"]["bounded_v1"]
    measured_seconds = 900.0  # Sulaiman p99 682.612s; capped at the 900s file lane.
    profile["maximumWallSeconds"] = max(float(profile["maximumWallSeconds"]), measured_seconds)
    profile["maximumCpuSeconds"] = max(float(profile["maximumCpuSeconds"]), measured_seconds)
    assert profile["maximumWallSeconds"] >= measured_seconds
    assert profile["maximumCpuSeconds"] >= measured_seconds
    limits = NativeObservedProofLimitsV1.build(profile["nativeObserved"])
    started = time.monotonic()
    deadline = started + profile["maximumWallSeconds"]
    contexts = {}
    worlds, preparation = compiler._prepare_observed_play_worlds(exact, request,
        maximum_worlds=8, maximum_attempts=100_000, maximum_model_bytes=100_000_000,
        maximum_seconds=deadline - time.monotonic(), project_prefix=prefix, project_contexts=contexts)
    record_property("preparationSeconds", time.monotonic() - started)
    record_property("modelCount", len(preparation["models"]))
    assert contexts and len(preparation["models"]) == 8
    from baloot_platform.research.ortools_complete_state_graph import _CpSatPathOracle

    def forbidden(*args, **kwargs):
        pytest.fail("latent project enumeration would starve the native policy proof")

    monkeypatch.setattr(_CpSatPathOracle, "enumerate_paths", forbidden)
    selected = compiler._solve_observed_selected_action(worlds, Team.NORTH_SOUTH,
        native_limits=limits, maximum_seconds=deadline - time.monotonic(),
        project_contexts=contexts, admissible_root_actions=admissible)
    record_property("totalSeconds", time.monotonic() - started)
    record_property("selectedAction", selected["selectedAction"])
    record_property("usage", json.dumps(selected["usage"], sort_keys=True))
    assert selected["status"] == "complete"
    assert selected["selectedAction"] == "SA"
    assert selected["proofs"][selected["bestProofHash"]]["value"] == 13
    assert selected.get("admissibleRootActions") == (list(admissible) if admissible is not None else None)


def _world_parent_death_candidate(connection, stop):
    from baloot_platform.research.ortools_information_set_compiler import _initialize_world_compile_worker

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=1, mp_context=context, initializer=_initialize_world_compile_worker,
        initargs=(stop, context.Array("d", 1), context.Value("i", 0)),
    ) as pool:
        pool.submit(_world_parent_death_probe, connection)
        stop.wait()


@pytest.mark.real_clock
def test_world_compile_worker_exits_when_candidate_is_force_terminated():
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe()
    stop = context.Event()
    candidate = context.Process(target=_world_parent_death_candidate, args=(writer, stop))
    candidate.start()
    writer.close()
    try:
        # Spawn+import latency is machine-dependent; wait for the start message or the
        # candidate's death before terminating it.
        ready = multiprocessing.connection.wait([reader, candidate.sentinel])
        assert reader in ready, "world worker did not start"
        assert type(reader.recv()) is int
        assert candidate.is_alive()
        candidate.terminate()
        candidate.join()
        with pytest.raises(EOFError):
            reader.recv()
    finally:
        try:
            reader.send(None)
        except (BrokenPipeError, EOFError, OSError):
            pass
        if candidate.is_alive():
            candidate.terminate()
        candidate.join(10)
        reader.close()


def test_native_action_dominance_matches_exhaustive_interval_oracle():
    """Dominance needs separated bounds, not exact action values or a win."""
    from ortools.sat.python import cp_model
    from baloot_platform.research.ortools_information_set_proof import build_native_bid_dominance_model_v1

    intervals = [(low, high) for low in range(-2, 3) for high in range(low, 3)]
    for selected_bounds, alternative_bounds in itertools.product(intervals, repeat=2):
        lower, upper = selected_bounds
        other_lower, other_upper = alternative_bounds
        alternatives = [{"action": "alternative", "upperBound": other_upper}]
        model, _, claim, _, regret, _ = build_native_bid_dominance_model_v1(lower, alternatives)
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        assert solver.solve(model) == cp_model.OPTIMAL
        oracle = all(selected >= other for selected in range(lower, upper + 1)
                     for other in range(other_lower, other_upper + 1))
        assert solver.boolean_value(claim) == oracle
        if oracle:
            assert solver.value(regret) == 0
        # Solver scalar inputs are certified bounds, not the unknown true score.
        assert solver.value(regret) == max(0, other_upper - lower)


def test_matching_complete_state_strategy_guarantees_are_optimal_on_joint_path():
    """Independent oracle for reuse; no production certificate is admitted here.

    Exhaust every depth-three binary zero-sum game and every pure strategy pair.
    Both guarantees quantify over ALL opposing strategies, not one replay. The
    conclusion covers the joint reached path only, not off-path subgames or
    actor-information-set sequential optimality.
    """
    maximizing = tuple(itertools.product((0, 1), repeat=5))
    minimizing = tuple(itertools.product((0, 1), repeat=2))
    checked = 0
    for leaves in itertools.product((-1, 0, 1), repeat=8):
        last = tuple(max(leaves[2 * index:2 * index + 2]) for index in range(4))
        middle = (min(last[:2]), min(last[2:]))
        root = max(middle)

        def payoff(target, opponent):
            first = target[0]
            second = opponent[first]
            third = target[1 + 2 * first + second]
            return leaves[4 * first + 2 * second + third]

        floors = {target: min(payoff(target, opponent) for opponent in minimizing)
                  for target in maximizing}
        ceilings = {opponent: max(payoff(target, opponent) for target in maximizing)
                    for opponent in minimizing}
        assert max(floors.values()) == min(ceilings.values()) == root
        for target, opponent in itertools.product(maximizing, minimizing):
            if floors[target] != ceilings[opponent]:
                continue
            first, second = target[0], opponent[target[0]]
            third = target[1 + 2 * first + second]
            assert middle[first] == root
            assert last[2 * first + second] == middle[first]
            assert leaves[4 * first + 2 * second + third] == last[2 * first + second]
            checked += 1
    assert checked > 0

    # Matching root guarantees do not make either policy subgame-perfect at an
    # unreached history. Never reuse this argument for every possible branch.
    leaves = (0, 0, 0, 2, 1, 1, 1, 1)
    target, opponent = (1, 0, 0, 0, 0), (0, 0)
    assert min(payoff(target, other) for other in minimizing) == 1
    assert max(payoff(other, opponent) for other in maximizing) == 1
    assert leaves[2 + target[2]] < max(leaves[2:4])

    # A successful paired replay alone supplies neither unilateral guarantee.
    leaves = (0, 0, 2, 2, 1, 1, 1, 1)
    target, opponent = (0, 0, 0, 0, 0), (1, 0)
    assert payoff(target, opponent) == 2
    assert min(payoff(target, other) for other in minimizing) == 0
    assert max(payoff(other, opponent) for other in maximizing) == 2


@pytest.mark.parametrize("world_count", [1, 2])
@pytest.mark.parametrize("losing_choice", [False, True])
def test_adversary_preserves_shared_dag_under_unique_node_budget(world_count, losing_choice):
    leaf = FiniteInformationNodeV1.terminal(1)
    children = {"a": leaf, "b": leaf}
    if losing_choice:
        children["c"] = FiniteInformationNodeV1.terminal(-1)
    branch = FiniteInformationNodeV1.decision_observed("opponent", observation_id="choice",
        children=children)
    problem = FiniteInformationSetProblemV1.build(problem_id="shared-adversary",
        worlds=[FiniteInformationWorldV1.build(str(index), {"start": branch})
                for index in range(world_count)], success_threshold=1)
    result = OrToolsActorInformationSetProofV1().prove(problem,
        budget=InformationSetProofBudgetV1(maximum_nodes=2 + int(losing_choice), maximum_worlds=2))
    assert result.completion_status == "exact_complete"
    assert result.winning_root_action_count == int(not losing_choice)


@pytest.mark.real_clock
def test_master_fixed_action_is_enforced_and_hash_bound():
    from baloot_platform.research.ortools_information_set_proof import _Catalog, _CounterexampleCut
    service = OrToolsActorInformationSetProofV1()
    catalog = _Catalog({("target", "root"): ("a", "b")}, 1, (-1, 1))
    cuts = [_CounterexampleCut(-1, (("root", "a"),))]
    def solve(fixed=None):
        return service._solve_master("0" * 64, "root", catalog, cuts,
            time.monotonic() + 5, None, fixed_actions=fixed)
    free, losing, winning = solve(), solve({"root": "a"}), solve({"root": "b"})
    assert free.value == winning.value == 1 and losing.value == -1
    assert losing.policy == (("root", "a"),)
    assert len({free.model_hash, losing.model_hash, winning.model_hash}) == 3
    assert solve({}).model_hash == free.model_hash
    for invalid in ({"root": "invalid"}, {"unknown": "a"}):
        with pytest.raises(ValueError, match="target observation domain"):
            solve(invalid)


@pytest.mark.real_clock
def test_master_reified_cuts_match_exhaustive_fixed_policy_values():
    from itertools import product
    from baloot_platform.research.ortools_information_set_proof import _Catalog, _CounterexampleCut

    catalog = _Catalog({("target", name): ("a", "b") for name in ("x", "y")}, 2, (-10, 10))
    cuts = [_CounterexampleCut(8, ()), _CounterexampleCut(-3, (("x", "a"),)),
            _CounterexampleCut(-7, (("x", "b"), ("y", "b"))),
            _CounterexampleCut(9, ()), _CounterexampleCut(4, (("x", "a"),)),
            _CounterexampleCut(-3, (("x", "a"),))]
    for choices in product(("a", "b"), repeat=2):
        policy = dict(zip(("x", "y"), choices, strict=True))
        expected = min(cut.utility for cut in cuts
                       if all(policy[key] == action for key, action in cut.target_actions))
        result = OrToolsActorInformationSetProofV1()._solve_master(
            "0" * 64, "root", catalog, cuts, time.monotonic() + 5, None, fixed_actions=policy)
        assert result.status == "optimal" and result.value == expected


@pytest.mark.real_clock
def test_master_preserves_choices_without_weakening_value_or_fixed_actions():
    from baloot_platform.research.ortools_information_set_proof import _Catalog, _CounterexampleCut

    service = OrToolsActorInformationSetProofV1()
    names = tuple(f"observation-{index:02}" for index in range(16))
    catalog = _Catalog({("target", name): ("a", "b") for name in names}, 16, (-1, 1))
    cuts = [_CounterexampleCut(-1, tuple((name, "a") for name in names))]
    preferred = {name: "a" for name in names}
    def solve(preferences=None, fixed=None):
        return service._solve_master("0" * 64, "root", catalog, cuts,
            time.monotonic() + 5, None, canonicalize_policy=False,
            preferred_actions=preferences, fixed_actions=fixed)
    result = solve(preferred)
    assert result.status == "optimal" and result.value == 1
    assert sum(action != preferred[key] for key, action in result.policy) == 1
    assert solve(dict(reversed(list(preferred.items())))) == result
    assert solve().model_hash == solve({}).model_hash != result.model_hash
    fixed = solve(preferred, preferred)
    assert fixed.status == "optimal" and fixed.value == -1 and dict(fixed.policy) == preferred
    for invalid in ({"missing": "a"}, {names[0]: "missing"}, {names[0]: True}, []):
        with pytest.raises(ValueError, match="target observation domain"):
            solve(invalid)


@pytest.mark.real_clock
@pytest.mark.parametrize("count", [1, 20, 40])
def test_master_canonical_blocks_preserve_lexicographic_policy(count, monkeypatch):
    from baloot_platform.research.ortools_information_set_proof import _Catalog, _CounterexampleCut
    service = OrToolsActorInformationSetProofV1()
    names = tuple(f"observation-{index:02}" for index in range(count))
    catalog = _Catalog({("target", name): ("a", "b", "c") for name in names}, count, (0, 1))
    cuts = [_CounterexampleCut(0, tuple((name, "a") for name in names)),
            _CounterexampleCut(0, ((names[-1], "b"),))]
    calls = []
    original = service._solve_optimal
    def solve(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(service, "_solve_optimal", solve)
    result = service._solve_master("0" * 64, "root", catalog, cuts, time.monotonic() + 10, None)
    assert result.status == "optimal" and result.value == 1
    assert result.policy == tuple((name, "c" if name == names[-1] else "a") for name in names)
    assert len(calls) == 1 + (count + 18) // 19


@pytest.mark.real_clock
@pytest.mark.parametrize("count", [1, 30, 31, 64])
def test_adversary_canonical_blocks_preserve_inactive_and_forced_choices(count, monkeypatch):
    service = OrToolsActorInformationSetProofV1()
    names = tuple(f"observation-{index:03}" for index in range(count))
    worlds = [FiniteInformationWorldV1.build(f"world-{index:03}", {
        "root": FiniteInformationNodeV1.decision_observed("opponent", observation_id=name,
            children={"a": FiniteInformationNodeV1.terminal(0),
                      "b": FiniteInformationNodeV1.terminal(-1)})})
        for index, name in enumerate(names)]
    worlds.append(FiniteInformationWorldV1.build("world-forced", {
        "root": FiniteInformationNodeV1.decision_observed("opponent", observation_id="forced",
            children={"only": FiniteInformationNodeV1.terminal(0)})}))
    problem = FiniteInformationSetProblemV1.build(problem_id="canonical-adversary",
        worlds=worlds, success_threshold=1)
    deadline = time.monotonic() + 10
    catalog = service._catalog(problem, 1000, "root", deadline=deadline)
    calls = []
    original = service._solve_optimal
    def solve(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(service, "_solve_optimal", solve)
    result = service._solve_adversary(problem, "root", catalog, (), 1000, deadline, None)
    expected_policy = (("forced", "only"), *(
        (name, "b" if index == 0 else "a") for index, name in enumerate(names)))
    assert result.status == "optimal" and result.value == -1
    assert result.world_id == "world-000" and result.path == ("root", "b")
    assert result.target_actions == () and result.opponent_policy_hash == canonical_sha256(expected_policy)
    assert len(calls) == 2 + (count + 29) // 30


def test_master_canonical_policy_stays_stable_across_successive_long_cuts():
    from baloot_platform.research.ortools_information_set_proof import _Catalog, _CounterexampleCut

    service = OrToolsActorInformationSetProofV1()
    names = tuple(f"observation-{index:02}" for index in range(99))
    catalog = _Catalog({("target", name): ("a", "b", "c") for name in names}, 99, (-200, -110))
    first = _CounterexampleCut(-112, tuple((name, "a") for name in names[:19]))
    second = _CounterexampleCut(-112, tuple((name, "b" if index == 18 else "a")
        for index, name in enumerate(names[:19])))
    churn = {}
    for canonical in (False, True):
        policies = []
        started = time.monotonic()
        for cuts in ([], [first], [first, second]):
            result = service._solve_master("0" * 64, "root", catalog, cuts,
                started + 10, None, canonicalize_policy=canonical)
            assert result.status == "optimal" and result.value == -110
            policies.append(dict(result.policy))
        churn[canonical] = sum(before[name] != after[name]
            for before, after in zip(policies, policies[1:]) for name in names)
        if canonical:
            assert churn[canonical] == 2
            assert policies[-1] == {name: "c" if index == 18 else "a"
                for index, name in enumerate(names)}
        print(f"canonical={canonical} changed_choices={churn[canonical]} seconds={time.monotonic()-started:.6f}")


@pytest.mark.real_clock
def test_master_applies_explicit_conflict_budget(monkeypatch):
    from baloot_platform.research.ortools_information_set_proof import _Catalog

    service = OrToolsActorInformationSetProofV1()
    catalog = _Catalog({("target", "root"): ("a", "b")}, 1, (0, 1))
    original = service._solve_optimal
    observed = []
    def solve(solver, *args, **kwargs):
        observed.append(solver.parameters.max_number_of_conflicts)
        return original(solver, *args, **kwargs)
    monkeypatch.setattr(service, "_solve_optimal", solve)
    result = service._solve_master("0" * 64, "root", catalog, [], time.monotonic() + 10, None,
        maximum_conflicts=7)
    assert result.status == "optimal" and observed and set(observed) == {7}
    for value in (True, 0, -1, 1.5):
        with pytest.raises(ValueError, match="conflict limit"):
            service._solve_master("0" * 64, "root", catalog, [], time.monotonic() + 10, None,
                maximum_conflicts=value)


@pytest.mark.real_clock
def test_intermediate_master_skips_canonical_resolves(monkeypatch):
    from baloot_platform.research.ortools_information_set_proof import _Catalog, _CounterexampleCut
    service = OrToolsActorInformationSetProofV1()
    catalog = _Catalog({("target", f"observation-{index:02}"): ("a", "b", "c")
                        for index in range(40)}, 40, (0, 1))
    cuts = [_CounterexampleCut(0, (("observation-39", "a"),))]
    calls = []
    original = service._solve_optimal
    monkeypatch.setattr(service, "_solve_optimal",
        lambda *args, **kwargs: calls.append(1) or original(*args, **kwargs))
    fast = service._solve_master("0" * 64, "root", catalog, cuts,
        time.monotonic() + 10, None, canonicalize_policy=False)
    assert fast.status == "optimal" and fast.value == 1 and len(calls) == 1
    assert fast == service._solve_master("0" * 64, "root", catalog, cuts,
        time.monotonic() + 10, None, canonicalize_policy=False)
    canonical = service._solve_master("0" * 64, "root", catalog, cuts,
        time.monotonic() + 10, None)
    assert fast.model_hash != canonical.model_hash


@pytest.mark.real_clock
def test_root_qualification_rejects_partial_values_before_solving(monkeypatch):
    problem = FiniteInformationSetProblemV1.build(problem_id="qualification-boundary",
        worlds=[FiniteInformationWorldV1.build("world", {"a": FiniteInformationNodeV1.terminal(1)})],
        success_threshold=1)
    service = OrToolsActorInformationSetProofV1()
    proven = service.prove(problem)
    assert proven.completion_status == "exact_complete"
    monkeypatch.setattr(service, "_solve_optimal", lambda *args, **kwargs: pytest.fail("partial root reached solver"))
    for roots in ((), (replace(proven.roots[0], completion_status="incomplete"),)):
        with pytest.raises(ValueError, match="completed exact root proofs"):
            service._certify_root_thresholds(roots, success_threshold=1, problem_hash=problem.sha256,
                claim="winning_root_action_count", selected_action=None,
                deadline=time.monotonic() + 5, cancel_check=None)


def test_compact_forced_information_prefix_preserves_domains():
    leaf = FiniteInformationNodeV1.terminal(1)
    compact = leaf.with_forced_prefix((("target", "visible", "pass"),))
    expanded = FiniteInformationNodeV1.decision_observed("target", observation_id="visible",
        children={"pass": leaf})
    def problem(node, other=None):
        worlds = [FiniteInformationWorldV1.build("one", {"start": node})]
        if other is not None:
            worlds.append(FiniteInformationWorldV1.build("two", {"start": other}))
        return FiniteInformationSetProblemV1.build(problem_id="forced", worlds=worlds, success_threshold=1)
    service = OrToolsActorInformationSetProofV1()
    assert service.prove(problem(compact)).winning_root_action_count == 1
    assert service.prove(problem(compact)).roots[0].principal_variation == (
        service.prove(problem(expanded)).roots[0].principal_variation) == ("start", "pass")
    assert service._catalog(problem(compact), 10).domains == service._catalog(problem(expanded), 10).domains
    assert compact.to_dict()["forcedPrefix"][0]["action"] == "pass"
    assert leaf.to_dict() == {"utility": 1}
    checks = []
    def cancel_during_prefix():
        checks.append(True)
        return len(checks) == 3
    with pytest.raises(RuntimeError, match="cancelled_or_wall_time"):
        service._catalog(problem(compact.with_forced_prefix((("target", "earlier", "wait"),))),
                         10, cancel_check=cancel_during_prefix)
    assert len(checks) == 3
    checks.clear()
    stopped = service.prove(problem(compact.with_forced_prefix((("target", "earlier", "wait"),))),
                            cancel_check=cancel_during_prefix)
    assert stopped.completion_status == "proof_incomplete"
    assert stopped.compiled_nodes == 1
    early = service.prove(problem(compact), cancel_check=lambda: True)
    assert early.compiled_nodes == 0
    with pytest.raises(ValueError, match="observation_action_domain_mismatch"):
        service._catalog(problem(compact, leaf.with_forced_prefix((("target", "visible", "declare"),))), 10)
    with pytest.raises(ValueError, match="both teams"):
        service._catalog(problem(compact, leaf.with_forced_prefix((("opponent", "visible", "pass"),))), 10)


def test_compact_witness_keeps_forced_actions_before_decision_and_terminal():
    leaf = FiniteInformationNodeV1.terminal(1).with_forced_prefix(
        (("opponent", "end", "finish"),))
    node = FiniteInformationNodeV1.decision_observed("target", observation_id="choice",
        children={"choose": leaf}).with_forced_prefix((("target", "start", "pass"),))
    target = {("target", "start"): "pass", ("target", "choice"): "choose"}
    opponent = {("opponent", "end"): "finish"}
    assert OrToolsActorInformationSetProofV1._play(node, target, opponent) == (
        1, ("pass", "choose", "finish"), (("start", "pass"), ("choice", "choose")))
    with pytest.raises(ValueError, match="forced.*policy"):
        OrToolsActorInformationSetProofV1._play(node, target,
            {("opponent", "end"): "wrong"})


def _observer_state(rules, marker: int) -> dict[str, object]:
    state = RoundState(rules.sha256, 91, marker, Seat.EAST)
    state.phase = GamePhase.PLAYING
    state.purchase_card = Card(0)
    state.initial_hands = [
        cards_to_mask(FULL_DECK[int(seat) * 8 : (int(seat) + 1) * 8]) for seat in Seat
    ]
    state.hands = list(state.initial_hands)
    state.contract = Contract.SUN
    state.buyer = Seat.NORTH
    state.declarer = Seat.NORTH
    state.first_leader = Seat.SOUTH
    state.current_player = Seat.SOUTH
    state.current_trick = Trick(0, Seat.SOUTH)
    return state.to_dict(reveal_hands=True)


def _swap_hidden_cards(state: dict[str, object]) -> None:
    for key in ("initialHands", "hands"):
        hands = state[key]
        assert isinstance(hands, dict)
        north = hands["north"]
        west = hands["west"]
        assert isinstance(north, list) and isinstance(west, list)
        north[1], west[1] = west[1], north[1]


def _target(
    complete_state: dict[str, object],
    x: int,
    y: int,
) -> FiniteInformationNodeV1:
    return FiniteInformationNodeV1.decision(
        "target",
        complete_state=complete_state,
        acting_seat="south",
        children={
            "x": FiniteInformationNodeV1.terminal(x),
            "y": FiniteInformationNodeV1.terminal(y),
        },
    )


def _clairvoyance_problem(rules) -> FiniteInformationSetProblemV1:
    safe_a = _observer_state(rules, 1)
    safe_b = copy.deepcopy(safe_a)
    _swap_hidden_cards(safe_b)
    win_observation = _observer_state(rules, 2)
    return FiniteInformationSetProblemV1.build(
        problem_id="two-world-clairvoyance",
        success_threshold=5,
        worlds=(
            FiniteInformationWorldV1.build(
                "world-a",
                {
                    "safe": _target(safe_a, 10, 0),
                    "win": _target(win_observation, 7, 1),
                },
            ),
            FiniteInformationWorldV1.build(
                "world-b",
                {
                    "safe": _target(safe_b, 0, 10),
                    "win": _target(win_observation, 7, 1),
                },
            ),
        ),
    )


def _exhaustive_root_value(
    problem: FiniteInformationSetProblemV1,
    root_action: str,
) -> int:
    """Independent small-fixture oracle; production qualification stays CP-SAT."""
    domains: dict[tuple[str, str], tuple[str, ...]] = {}

    def collect(node: FiniteInformationNodeV1) -> None:
        if node.utility is not None:
            return
        assert node.actor is not None and node.observation_id is not None
        key = (node.actor, node.observation_id)
        actions = tuple(action for action, _child in node.children)
        assert domains.setdefault(key, actions) == actions
        for _action, child in node.children:
            collect(child)

    def play(
        node: FiniteInformationNodeV1,
        target: dict[tuple[str, str], str],
        opponent: dict[tuple[str, str], str],
    ) -> int:
        current = node
        while current.utility is None:
            assert current.actor is not None and current.observation_id is not None
            policy = target if current.actor == "target" else opponent
            current = dict(current.children)[
                policy[(current.actor, current.observation_id)]
            ]
        return current.utility

    for world in problem.worlds:
        collect(dict(world.roots)[root_action])
    target_keys = tuple(sorted(key for key in domains if key[0] == "target"))
    opponent_keys = tuple(sorted(key for key in domains if key[0] == "opponent"))
    target_rows = itertools.product(*(domains[key] for key in target_keys))
    best: int | None = None
    for target_actions in target_rows:
        target = dict(zip(target_keys, target_actions, strict=True))
        worst: int | None = None
        opponent_rows = itertools.product(*(domains[key] for key in opponent_keys))
        for opponent_actions in opponent_rows:
            opponent = dict(zip(opponent_keys, opponent_actions, strict=True))
            for world in problem.worlds:
                value = play(dict(world.roots)[root_action], target, opponent)
                worst = value if worst is None else min(worst, value)
        assert worst is not None
        best = worst if best is None else max(best, worst)
    assert best is not None
    return best


@pytest.mark.parametrize("positive,leak,expected", [(True, False, 1), (False, False, 0), (False, True, 1)])
def test_unobserved_opportunity_orders_share_actor_policies(positive, leak, expected):
    """Abstract proof-model check, not Baloot escalation reachability evidence."""
    worlds = []
    for hidden in (((0, 1), (1, 0), (1, 1)) if positive else (0, 1)):
        for order in ((0, 1), (1, 0)):
            def node(index):
                if index == 2:
                    return FiniteInformationNodeV1.terminal(0)
                actor = order[index]
                own = hidden[actor] if positive else 0
                observation = canonical_sha256({
                    "seat": actor, "own": own,
                    **({"deliberatelyLeakedWorld": hidden} if leak else {}),
                })
                utility = (1 if own else -1) if positive else (1 if hidden == actor else -1)
                return FiniteInformationNodeV1.decision_observed(
                    "target", observation_id=observation,
                    children={"announce": FiniteInformationNodeV1.terminal(utility), "decline": node(index + 1)},
                )

            worlds.append(FiniteInformationWorldV1.build(f"{hidden}-{order}", {"continue": node(0)}))
    problem = FiniteInformationSetProblemV1.build(
        problem_id=f"joint-opportunity-{positive}-{leak}", worlds=worlds, success_threshold=1,
    )
    proof = OrToolsActorInformationSetProofV1().prove(
        problem, budget=InformationSetProofBudgetV1(maximum_nodes=1000, maximum_worlds=8, maximum_wall_seconds=5),
    )
    assert proof.completion_status == "exact_complete"
    assert proof.roots[0].robust_value == _exhaustive_root_value(problem, "continue") == expected
    assert proof.winning_root_action_count == expected


@pytest.mark.parametrize("positive,leak,expected", [(True, False, 1), (False, False, 0), (False, True, 1)])
def test_native_escalation_orders_share_policies_through_graph_compiler(rules, positive, leak, expected):
    """Native legal escalation edges; synthetic private signals and terminal utility."""
    from baloot_platform.core.escalation_ir import EscalationDomainIRV1
    from baloot_platform.research.ortools_nonplay_graphs import (
        NonPlayGraphBudgetV1, compile_escalation_decision_graph_v1,
        enumerate_escalation_opportunity_orders_v1,
    )

    worlds = []
    seats = (Seat.WEST, Seat.EAST)
    coverage = enumerate_escalation_opportunity_orders_v1(
        EscalationDomainIRV1.from_rules(rules.data), declarer=Seat.SOUTH,
        contract=Contract.SUN, match_score=(101, 50), maximum_worlds=6 if positive else 4,
        hidden_world_count=3 if positive else 2,
        maximum_seconds=5, budget=NonPlayGraphBudgetV1(100, 8),
    )
    assert coverage["complete"] and coverage["orderWorldCount"] == 2
    assert coverage["combinedWorldCount"] == (6 if positive else 4)
    for hidden in (((0, 1), (1, 0), (1, 1)) if positive else (0, 1)):
        for order in coverage["worlds"]:
            observations = {}

            def observer(value):
                actor = seats.index(value.actor)
                identity = canonical_sha256({
                    "actor": value.actor.name.lower(), "public": value.public_state,
                    "ownSignal": hidden[actor] if positive else 0,
                    **({"deliberatelyLeakedWorld": hidden} if leak else {}),
                })
                observations[value.node_id] = identity
                return identity

            def utility(terminal):
                if not terminal.actions:
                    return 0
                actor = seats.index(terminal.actions[-1].actor)
                return (1 if hidden[actor] else -1) if positive else (1 if hidden == actor else -1)

            native = compile_escalation_decision_graph_v1(
                EscalationDomainIRV1.from_rules(rules.data), declarer=Seat.SOUTH,
                contract=Contract.SUN, match_score=(101, 50), native_cp_sat=True,
                opportunity_orders=order["opportunityOrders"],
                decision_observer=observer, terminal_utility=utility,
            )
            # Shared continuation root isolates uncertainty AFTER a known decision.
            root = ProofGraphNodeV1.decision("fixture-root", "west_east",
                (ProofGraphEdgeV1.build(0, "continue", native.root_node_id),), actor_seat="west")
            graph = ProofGraphV1.build(scope="exact", root_node_id=root.node_id,
                nodes=(root, *native.nodes))
            observations[root.node_id] = "fixture-visible-root"
            worlds.append(ActorInformationGraphWorldV1.build(
                world_id=f"{hidden}-{order['worldId']}", decision_kind="escalation", graph=graph,
                observation_ids=observations,
            ))
    request = ScenarioProofRequestV1.build({
        "schemaVersion": 1, "mode": "exact", "visibility": "actor_information_set",
        "strategyClass": "pure_deterministic",
        "claim": "outcome_reachable", "targetTeam": "west_east", "utility": "awarded_point_margin",
        "successThreshold": 1, "budgetProfile": "test",
    })
    compilation = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True).compile_graph_worlds(
        worlds, request, maximum_nodes=1000, maximum_depth=10, maximum_worlds=8, maximum_seconds=5,
    )
    problem = compilation.graph.problem
    proof = OrToolsActorInformationSetProofV1().prove(problem,
        budget=InformationSetProofBudgetV1(maximum_nodes=1000, maximum_worlds=8, maximum_wall_seconds=5))
    assert proof.completion_status == "exact_complete"
    assert proof.roots[0].robust_value == _exhaustive_root_value(problem, "continue") == expected
    assert proof.winning_root_action_count == expected


def test_cp_sat_binds_one_action_across_indistinguishable_worlds(rules) -> None:
    problem = _clairvoyance_problem(rules)
    proof = OrToolsActorInformationSetProofV1().prove(problem)
    rows = {row.root_action: row for row in proof.roots}

    assert proof.completion_status == "exact_complete"
    assert proof.qualification_authority == "cp_sat"
    assert proof.winning_root_action_count == 1
    assert (
        rows["safe"].robust_value == 0
    )  # A clairvoyant per-world choice would report 10.
    assert rows["safe"].satisfies is False
    assert rows["win"].robust_value == 7
    assert rows["win"].satisfies is True
    win_hash = dict(problem.worlds[0].roots)["win"].observation_id
    assert rows["win"].target_policy == ((win_hash, "x"),)
    assert all(row.solver_status == "optimal_converged" for row in proof.roots)
    assert rows["safe"].iterations == 3
    assert len(rows["safe"].cut_hashes) == 2
    assert rows["win"].iterations == 1
    assert rows["win"].cut_hashes == ()
    # No single-world perfect-information upper can certify the joint value:
    # each world admits a different best action hidden from the same observer.
    for world in problem.worlds:
        single = FiniteInformationSetProblemV1.build(problem_id="one-world-" + world.world_id,
            success_threshold=problem.success_threshold, worlds=(world,))
        result = OrToolsActorInformationSetProofV1().prove(single)
        assert result.completion_status == "exact_complete"
        assert next(row.robust_value for row in result.roots if row.root_action == "safe") == 10


def test_hidden_world_nodes_use_real_canonical_observer_documents(rules) -> None:
    first_state = _observer_state(rules, 9)
    second_state = copy.deepcopy(first_state)
    _swap_hidden_cards(second_state)

    first_projection = build_canonical_observer_information_v1(
        complete_state=first_state,
        acting_seat="south",
    )
    second_projection = build_canonical_observer_information_v1(
        complete_state=second_state,
        acting_seat="south",
    )
    first_node = _target(first_state, 1, 0)
    second_node = _target(second_state, 0, 1)

    assert first_projection == second_projection
    assert first_node.observation_id == second_node.observation_id
    assert first_node.observation_id == canonical_observer_information_hash_v1(
        complete_state=first_state,
        acting_seat="south",
    )
    assert (
        first_node.to_dict()["observationId"]
        == first_projection["observerInformationHash"]
    )
    assert "initialHands" not in first_node.to_dict()
    assert "hands" not in first_node.to_dict()


def test_canonical_observation_action_mismatch_enters_model_quarantine(rules) -> None:
    observation = _observer_state(rules, 10)
    first = FiniteInformationNodeV1.decision(
        "target",
        complete_state=observation,
        acting_seat="south",
        children={
            "x": FiniteInformationNodeV1.terminal(1),
            "y": FiniteInformationNodeV1.terminal(0),
        },
    )
    mismatched = FiniteInformationNodeV1.decision(
        "target",
        complete_state=observation,
        acting_seat="south",
        children={
            "x": FiniteInformationNodeV1.terminal(1),
            "z": FiniteInformationNodeV1.terminal(0),
        },
    )
    problem = FiniteInformationSetProblemV1.build(
        problem_id="canonical-domain-mismatch",
        success_threshold=0,
        worlds=(
            FiniteInformationWorldV1.build("one", {"H7": first}),
            FiniteInformationWorldV1.build("two", {"H7": mismatched}),
        ),
    )

    receipt = OrToolsActorInformationSetProofV1().prove(problem)
    document = receipt.to_dict()
    assert receipt.completion_status == "model_error"
    assert receipt.model_status == "quarantined"
    assert receipt.quarantine_reason == "observation_action_domain_mismatch"
    assert receipt.winning_root_action_count is None
    assert document["completionStatus"] != "proof_failed"
    assert document["modelStatus"] == "quarantined"
    assert all(root.completion_status == "model_error" for root in receipt.roots)

    outcome = prove_actor_information_set_v1(
        ActorInformationSetCompilationV1(
            ActorInformationSetGraphV1(
                problem,
                {"count": 2, "complete": True},
                {"kind": "test_allocation_model"},
            ),
            node_count=4,
        ),
        _actor_request(),
        maximum_nodes=100,
        maximum_worlds=2,
        maximum_seconds=2.0,
    )
    assert outcome.completion_status == "model_error"
    assert outcome.solver_status == "model_error"
    assert outcome.claim_satisfied is False


def test_opponent_observation_is_also_nonanticipative(rules) -> None:
    observation = _observer_state(rules, 3)

    def opponent(first: int, second: int) -> FiniteInformationNodeV1:
        return FiniteInformationNodeV1.decision(
            "opponent",
            complete_state=observation,
            acting_seat="west",
            children={
                "a": FiniteInformationNodeV1.terminal(first),
                "b": FiniteInformationNodeV1.terminal(second),
            },
        )

    problem = FiniteInformationSetProblemV1.build(
        problem_id="opponent-policy",
        success_threshold=0,
        worlds=(
            FiniteInformationWorldV1.build("one", {"root": opponent(3, 8)}),
            FiniteInformationWorldV1.build("two", {"root": opponent(-2, 9)}),
        ),
    )
    row = OrToolsActorInformationSetProofV1().prove(problem).roots[0]
    assert row.robust_value == -2
    assert row.opponent_policy_count == 2


def test_cp_sat_matches_brute_force_two_world_oracle(rules) -> None:
    service = OrToolsActorInformationSetProofV1()
    observation = _observer_state(rules, 4)
    for values in itertools.product((-2, 3), repeat=4):
        problem = FiniteInformationSetProblemV1.build(
            problem_id="brute-force-parity-" + "-".join(map(str, values)),
            success_threshold=0,
            worlds=(
                FiniteInformationWorldV1.build(
                    "one",
                    {"root": _target(observation, values[0], values[1])},
                ),
                FiniteInformationWorldV1.build(
                    "two",
                    {"root": _target(observation, values[2], values[3])},
                ),
            ),
        )
        expected = _exhaustive_root_value(problem, "root")
        assert service.prove(problem).roots[0].robust_value == expected


def test_cegis_matches_exhaustive_oracle_with_both_actors(rules) -> None:
    target_observation = _observer_state(rules, 5)
    opponent_observation = _observer_state(rules, 6)

    def opponent(a: int, b: int) -> FiniteInformationNodeV1:
        return FiniteInformationNodeV1.decision(
            "opponent",
            complete_state=opponent_observation,
            acting_seat="west",
            children={
                "a": FiniteInformationNodeV1.terminal(a),
                "b": FiniteInformationNodeV1.terminal(b),
            },
        )

    problem = FiniteInformationSetProblemV1.build(
        problem_id="mixed-two-world-oracle",
        success_threshold=0,
        worlds=(
            FiniteInformationWorldV1.build(
                "one",
                {
                    "root": FiniteInformationNodeV1.decision(
                        "target",
                        complete_state=target_observation,
                        acting_seat="south",
                        children={
                            "x": opponent(8, -4),
                            "y": opponent(3, 2),
                        },
                    )
                },
            ),
            FiniteInformationWorldV1.build(
                "two",
                {
                    "root": FiniteInformationNodeV1.decision(
                        "target",
                        complete_state=target_observation,
                        acting_seat="south",
                        children={
                            "x": opponent(6, -2),
                            "y": opponent(1, 4),
                        },
                    )
                },
            ),
        ),
    )

    row = OrToolsActorInformationSetProofV1().prove(problem).roots[0]
    assert row.completion_status == "exact"
    assert row.robust_value == _exhaustive_root_value(problem, "root")


@pytest.mark.parametrize("allowance", [0, -1, True, float("nan"), float("inf"), 11])
def test_runtime_allowance_cannot_exceed_declared_proof_budget(rules, allowance):
    with pytest.raises(ValueError, match="runtime wall allowance"):
        OrToolsActorInformationSetProofV1().prove(
            _clairvoyance_problem(rules), runtime_maximum_seconds=allowance)


@pytest.mark.parametrize("seconds", [0, -1, True, float("nan"), float("inf")])
def test_declared_proof_wall_budget_must_be_finite(seconds):
    with pytest.raises(ValueError, match="positive and finite"):
        InformationSetProofBudgetV1(maximum_wall_seconds=seconds)


def test_runtime_deadline_preserves_declared_budget_identity(rules, monkeypatch):
    from baloot_platform.research import ortools_information_set_proof as module

    set_monotonic(monkeypatch, module, 0, 2)
    budget = InformationSetProofBudgetV1(maximum_wall_seconds=10)
    proof = OrToolsActorInformationSetProofV1().prove(
        _clairvoyance_problem(rules), budget=budget, runtime_maximum_seconds=1)
    assert proof.budget_hash == budget.sha256
    assert proof.completion_status == "proof_incomplete"
    assert all(root.exhaustion_reason == "cancelled_or_wall_time" for root in proof.roots)


def test_budget_exhaustion_is_incomplete_with_honest_root_bounds(rules) -> None:
    proof = OrToolsActorInformationSetProofV1().prove(
        _clairvoyance_problem(rules),
        budget=InformationSetProofBudgetV1(maximum_policies=1),
    )

    assert proof.completion_status == "proof_incomplete"
    assert proof.winning_root_action_count is None
    assert proof.winning_root_action_lower_bound == 0
    assert proof.winning_root_action_upper_bound == 2
    assert all(
        root.exhaustion_reason == "maximum_policies_target" for root in proof.roots
    )


def test_time_budgeted_symbolic_policy_search_preserves_exact_oracle_and_limits(rules, monkeypatch):
    from baloot_platform.research import ortools_information_set_proof as module

    problem = _clairvoyance_problem(rules)
    legacy = InformationSetProofBudgetV1(maximum_policies=1)
    assert "policySearch" not in legacy.to_dict()
    budget = InformationSetProofBudgetV1(maximum_policies=1, policy_search="time_budgeted_symbolic_v1")
    assert budget.sha256 != legacy.sha256
    proof = OrToolsActorInformationSetProofV1().prove(problem, budget=budget)
    assert all(row.completion_status == "exact" for row in proof.roots)
    assert all(row.robust_value == _exhaustive_root_value(problem, row.root_action) for row in proof.roots)
    for limits in ({"cancel_check": lambda: True}, {"runtime_maximum_seconds": 1}):
        with monkeypatch.context() as patch:
            if "runtime_maximum_seconds" in limits:
                set_monotonic(patch, module, 0, 2)
            partial = OrToolsActorInformationSetProofV1().prove(problem, budget=budget, **limits)
        assert partial.completion_status == "proof_incomplete"
        assert all(row.exhaustion_reason == "cancelled_or_wall_time" for row in partial.roots)
    for invalid in (True, "other", []):
        with pytest.raises(ValueError, match="policy search"):
            InformationSetProofBudgetV1(policy_search=invalid)


@pytest.mark.real_clock
@pytest.mark.parametrize("reason", ["cancelled", "maximum_seconds"])
@pytest.mark.parametrize("phase", ["before", "during", "after"])
def test_play_world_model_build_obeys_compilation_stop(rules, monkeypatch, reason, phase):
    from types import SimpleNamespace
    import baloot_platform.research.ortools_information_set_compiler as module

    @dataclass
    class Exact:
        state: dict
        state_hash: str

    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    world = {"currentPlayer": "north"}
    clock = [2.0 if phase == "before" and reason == "maximum_seconds" else 0.0]
    cancelled = [reason == "cancelled" and phase == "before"]
    parsed = []
    built = []
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    def parse(exact, **kwargs):
        parsed.append(exact)
        return SimpleNamespace()
    def build(start, **kwargs):
        built.append(start)
        if reason == "cancelled":
            cancelled[0] = True
        else:
            clock[0] = 2.0
        assert callable(kwargs.get("cancel_check")), "model building must receive the compilation stop check"
        if phase == "after":
            return SimpleNamespace(enumerate_paths=lambda **kw: pytest.fail("stopped model must not be solved"))
        kwargs["cancel_check"]()
        pytest.fail("stopped graph construction continued")
    monkeypatch.setattr(compiler.graph_compiler, "_parse_start", parse)
    monkeypatch.setattr(compiler.graph_compiler, "_path_oracle", build)
    with pytest.raises(InformationSetCompilationStopped) as stopped:
        compiler._compile_play_worlds(Exact(world, "source"), _actor_request(), (world,),
            maximum_nodes=100, maximum_depth=8, deadline=1.0,
            cancel_check=lambda: cancelled[0], requested_recall={}, histories=None)
    assert stopped.value.reason == reason
    assert stopped.value.partial_receipt["stage"] == "information_graph_compilation"
    assert stopped.value.partial_receipt["evidence"] == {
        "compiledWorlds": 0, "compatibleWorlds": 1, "nodeCount": 0,
        "maximumNodes": 100, "maximumDepth": 8,
    }
    assert len(parsed) == len(built) == int(phase != "before")


def test_world_iteration_cancellation_and_unknown_are_incomplete(
    monkeypatch: pytest.MonkeyPatch,
    rules,
) -> None:
    service = OrToolsActorInformationSetProofV1()
    problem = _clairvoyance_problem(rules)

    world_limited = service.prove(
        problem,
        budget=InformationSetProofBudgetV1(maximum_worlds=1),
    )
    assert world_limited.completion_status == "proof_incomplete"
    assert all(
        root.exhaustion_reason == "maximum_worlds" for root in world_limited.roots
    )

    iteration_limited = service.prove(
        problem,
        budget=InformationSetProofBudgetV1(maximum_iterations=1),
    )
    assert iteration_limited.completion_status == "proof_incomplete"
    assert iteration_limited.winning_root_action_count is None
    iteration_rows = {root.root_action: root for root in iteration_limited.roots}
    assert iteration_rows["safe"].exhaustion_reason == "maximum_iterations"
    assert iteration_rows["win"].completion_status == "exact"
    assert iteration_limited.winning_root_action_lower_bound == 1
    assert iteration_limited.winning_root_action_upper_bound == 2

    cancelled = service.prove(problem, cancel_check=lambda: True)
    assert cancelled.completion_status == "proof_incomplete"
    assert all(
        root.exhaustion_reason == "cancelled_or_wall_time" for root in cancelled.roots
    )

    from ortools.sat.python import cp_model

    def unknown(_solver, _model, _deadline, statuses, _cancel_check=None):
        statuses.append(cp_model.UNKNOWN)
        return False

    monkeypatch.setattr(
        OrToolsActorInformationSetProofV1,
        "_solve_optimal",
        staticmethod(unknown),
    )
    unknown_receipt = service.prove(problem)
    assert unknown_receipt.completion_status == "proof_incomplete"
    assert all(
        root.exhaustion_reason == "master_not_optimal" for root in unknown_receipt.roots
    )
    assert all(root.solver_status == "unknown" for root in unknown_receipt.roots)
    outcome = prove_actor_information_set_v1(
        ActorInformationSetCompilationV1(
            ActorInformationSetGraphV1(
                problem,
                {"count": 2, "complete": True},
                {"kind": "test_allocation_model"},
                decision_kind="bid",
                root_action_ids={"safe": 1, "win": 2},
            ),
            node_count=2,
        ),
        _actor_request(threshold=5, claim="winning_root_action_count"),
        maximum_nodes=100,
        maximum_worlds=2,
        maximum_seconds=2,
    )
    assert outcome.winning_root_action_count is None
    assert outcome.winning_root_action_lower_bound == 0
    assert outcome.winning_root_action_upper_bound == 2


def test_determinism_and_closed_visible_root_domain(rules) -> None:
    service = OrToolsActorInformationSetProofV1()
    problem = _clairvoyance_problem(rules)
    first = service.prove(problem).to_dict()
    second = service.prove(problem).to_dict()
    assert first["semanticSha256"] == second["semanticSha256"]
    assert [root["cutHashes"] for root in first["roots"]] == [
        root["cutHashes"] for root in second["roots"]
    ]
    assert [root["modelHash"] for root in first["roots"]] == [
        root["modelHash"] for root in second["roots"]
    ]
    first.pop("elapsedWallSeconds")
    second.pop("elapsedWallSeconds")
    first.pop("sha256")
    second.pop("sha256")
    assert first == second

    with pytest.raises(ValueError, match="same visible root actions"):
        FiniteInformationSetProblemV1.build(
            problem_id="bad-roots",
            success_threshold=0,
            worlds=(
                FiniteInformationWorldV1.build(
                    "one", {"a": _target(_observer_state(rules, 7), 1, 0)}
                ),
                FiniteInformationWorldV1.build(
                    "two", {"b": _target(_observer_state(rules, 8), 1, 0)}
                ),
            ),
        )


def test_continuation_count_rejected_and_module_has_no_runtime_imports(rules) -> None:
    with pytest.raises(ValueError, match="continuation counts are unsupported"):
        OrToolsActorInformationSetProofV1().prove(
            _clairvoyance_problem(rules),
            claim="winning_continuation_line_count",
        )

    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "baloot_platform"
        / "research"
        / "ortools_information_set_proof.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        str(node.module or "")
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    forbidden = {"engine", "bots", "hidden_deals", "proof_service", "policy_proof"}
    assert not any(
        any(part == item for part in name.split("."))
        for name in imported
        for item in forbidden
    )


def _actor_request(
    *,
    threshold: float = 0.0,
    claim: str = "outcome_reachable",
    selected_action: str | None = None,
) -> ScenarioProofRequestV1:
    return ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "exact",
            "visibility": "actor_information_set",
            "strategyClass": "pure_deterministic",
            "claim": claim,
            "targetTeam": "north_south",
            "utility": "awarded_point_margin",
            "successThreshold": threshold,
            "budgetProfile": "late_game_exact",
            **({"selectedAction": selected_action} if selected_action is not None else {}),
        }
    )


def _bounded_actor_request(**kwargs) -> ScenarioProofRequestV1:
    value = _actor_request(**kwargs).to_dict()
    value.pop("sha256")
    value.update(
        mode="bounded",
        horizon={"unit": "round_end", "value": 1},
        boundaryEvaluator="current_raw_point_margin_v1",
    )
    return ScenarioProofRequestV1.build(value)


def _bounded_compilation(problem, request, *, complete=False):
    scope = {
        "mode": "bounded",
        "requestHash": request.sha256,
        "horizon": dict(request.horizon),
        "boundaryEvaluator": request.boundary_evaluator,
        "boundaryEvaluated": False,
        "boundaryNodeCount": 0,
        "continuationCoverage": "round_end",
        "worldCoverageComplete": complete,
        "worldScopeHash": canonical_sha256([world.world_id for world in problem.worlds]),
        "worldLimit": len(problem.worlds),
        "probabilityModel": None,
    }
    return ActorInformationSetCompilationV1(
        ActorInformationSetGraphV1(
            problem, {"count": len(problem.worlds), "complete": complete},
            {"kind": "test_allocation_model"},
            root_action_ids={action: index for index, (action, _) in enumerate(problem.worlds[0].roots)},
            analysis_scope=scope,
        ),
        node_count=10,
    )


@pytest.mark.parametrize("complete", [False, True])
def test_budgeted_quality_is_nonanticipative_and_never_globally_exact(rules, complete):
    problem = _clairvoyance_problem(rules)
    request = _bounded_actor_request(threshold=5, claim="winning_root_action_count")
    compilation = _bounded_compilation(problem, request, complete=complete)
    outcome = prove_actor_information_set_v1(
        compilation, request, maximum_nodes=1000, maximum_worlds=2,
        maximum_seconds=5, native_action_quality=True,
    )
    assert outcome.completion_status == "bounded_complete"
    assert outcome.solver_status == "optimal"
    assert outcome.selected_action == "win"
    document = outcome.model_document()
    assert document["rootActionCountScope"] == "declared_worlds_only"
    assert document["analysisScope"]["worldCoverageComplete"] is complete
    quality = document["actionQuality"]
    assert quality["analysisScope"] == document["analysisScope"]
    assert quality["bestValue"] == 7
    assert quality["regret"] == 0
    assert {row.action_code: row.value for row in outcome.root_actions} == {
        "safe": 0, "win": 7,
    }
    quality_core = {key: value for key, value in quality.items() if key != "sha256"}
    assert quality["sha256"] == canonical_sha256({**quality_core, "modelProto": document["modelProto"]})


@pytest.mark.parametrize("mutation", ["exact", "missing", "request", "worlds", "coverage", "threshold", "probability"])
def test_budgeted_scope_cannot_be_substituted_or_promoted(rules, mutation):
    problem = _clairvoyance_problem(rules)
    request = _bounded_actor_request(threshold=5, claim="winning_root_action_count")
    compilation = _bounded_compilation(problem, request)
    scope = dict(compilation.graph.analysis_scope)
    if mutation == "exact":
        request = _actor_request(threshold=5, claim="winning_root_action_count")
    elif mutation == "missing":
        scope = None
    elif mutation == "request":
        scope["requestHash"] = "0" * 64
    elif mutation == "worlds":
        scope["worldScopeHash"] = "0" * 64
    elif mutation == "threshold":
        request = _bounded_actor_request(threshold=11, claim="winning_root_action_count")
        scope["requestHash"] = request.sha256
    elif mutation == "probability":
        scope["probabilityModel"] = "uniform"
    else:
        scope["worldCoverageComplete"] = True
    compilation = replace(compilation, graph=replace(compilation.graph, analysis_scope=scope))
    with pytest.raises(ValueError, match="coverage|scope evidence|threshold"):
        prove_actor_information_set_v1(
            compilation, request, maximum_nodes=1000, maximum_worlds=2,
            maximum_seconds=5, native_action_quality=True,
        )


def test_budgeted_action_analysis_cancellation_does_not_certify_quality(rules):
    problem = _clairvoyance_problem(rules)
    request = _bounded_actor_request(threshold=5, claim="winning_root_action_count")
    outcome = prove_actor_information_set_v1(
        _bounded_compilation(problem, request), request,
        maximum_nodes=1000, maximum_worlds=2, maximum_seconds=5,
        native_action_quality=True, cancel_check=lambda: True,
    )
    assert outcome.completion_status == "proof_incomplete"
    assert "actionQuality" not in outcome.model_document()
    assert outcome.winning_root_action_count is None


def test_budgeted_false_claim_remains_false_within_declared_scope(rules):
    problem = replace(_clairvoyance_problem(rules), success_threshold=11)
    request = _bounded_actor_request(threshold=11, claim="outcome_reachable")
    outcome = prove_actor_information_set_v1(
        _bounded_compilation(problem, request), request,
        maximum_nodes=1000, maximum_worlds=2, maximum_seconds=5,
        native_action_quality=True,
    )
    assert outcome.completion_status == "proof_failed"
    assert outcome.claim_satisfied is False
    assert outcome.model_document()["analysisScope"]["worldCoverageComplete"] is False
    assert outcome.model_document()["actionQuality"]["bestValue"] == 7


def test_fractional_threshold_preserves_integer_utility_comparison() -> None:
    assert _integral_success_threshold(0.5) == 1
    assert _integral_success_threshold(-0.5) == 0
    assert _integral_success_threshold(2.0) == 2


@pytest.mark.parametrize(
    "relation,selected,expected_action,expected_value,expected_regret",
    [
        ("target", None, "S7", -2, 0),
        ("target", "SA", "SA", -11, 9),
        ("opponent", None, "SA", -11, 0),
        ("opponent", "S7", "S7", -2, 9),
    ],
)
def test_native_action_quality_distinguishes_bad_moves_even_when_every_move_loses(
    relation, selected, expected_action, expected_value, expected_regret,
):
    problem = FiniteInformationSetProblemV1.build(
        problem_id="quality-all-losing",
        success_threshold=0,
        worlds=(FiniteInformationWorldV1.build("one", {
            "SA": FiniteInformationNodeV1.terminal(-11),
            "S7": FiniteInformationNodeV1.terminal(-2),
        }),),
    )
    compilation = ActorInformationSetCompilationV1(
        ActorInformationSetGraphV1(
            problem, {"count": 1, "complete": True}, {"kind": "test_allocation_model"},
            root_actor_relation=relation,
        ), 2,
    )
    outcome = prove_actor_information_set_v1(
        compilation,
        _actor_request(claim="winning_root_action_count", selected_action=selected),
        maximum_nodes=100, maximum_worlds=1, maximum_seconds=3,
        native_action_quality=True,
    )
    document = outcome.model_document()
    quality = document["actionQuality"]
    assert outcome.completion_status == "exact_complete"
    assert outcome.winning_root_action_count == 0
    assert outcome.selected_action == expected_action
    assert quality["actualAction"] == expected_action
    assert quality["actualValue"] == expected_value
    assert quality["regret"] == expected_regret
    assert quality["actualActionOptimal"] is (expected_regret == 0)
    assert quality["scope"] == "compiled_actor_information_set"
    quality_core = {key: value for key, value in quality.items() if key != "sha256"}
    quality_core["modelProto"] = document["modelProto"]
    assert canonical_sha256(quality_core) == quality["sha256"]
    # Independently solve the exported model; reported quality must be bound
    # to real variables, not only appended metadata.
    from ortools.sat.python import cp_model

    model = cp_model.CpModel()
    assert model.proto.parse_text_format(document["modelProto"])
    by_name = {var.name: index for index, var in enumerate(model.proto.variables)}
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    assert solver.solve(model) == cp_model.OPTIMAL
    for field in ("regret", "bestValue", "actualValue", "bestActionId"):
        variable = model.get_int_var_from_proto_index(by_name[quality["variableBindings"][field]])
        assert solver.value(variable) == quality[field]


@pytest.mark.parametrize("selected", [None, "SA"])
def test_native_action_quality_records_only_executed_tie_breaks(selected):
    problem = FiniteInformationSetProblemV1.build(
        problem_id="quality-tied-losing", success_threshold=0,
        worlds=(FiniteInformationWorldV1.build("one", {
            "SA": FiniteInformationNodeV1.terminal(-2),
            "S7": FiniteInformationNodeV1.terminal(-2),
        }),),
    )
    compilation = ActorInformationSetCompilationV1(
        ActorInformationSetGraphV1(problem, {"count": 1, "complete": True},
            {"kind": "test_allocation_model"}, root_actor_relation="target"), 2)
    outcome = prove_actor_information_set_v1(compilation,
        _actor_request(claim="winning_root_action_count", selected_action=selected),
        maximum_nodes=100, maximum_worlds=1, maximum_seconds=3, native_action_quality=True)
    quality = outcome.model_document()["actionQuality"]
    assert outcome.winning_root_action_count == 0
    assert set(quality["optimalActions"]) == {"SA", "S7"}
    assert quality["actualAction"] == (selected or "S7")
    assert quality.get("selectionTieBreak") == ("stable_action_id" if selected is None else None)


def test_native_action_quality_preserves_nonanticipativity(rules):
    problem = _clairvoyance_problem(rules)
    compilation = ActorInformationSetCompilationV1(
        ActorInformationSetGraphV1(
            problem, {"count": 2, "complete": True}, {"kind": "test_allocation_model"},
            decision_kind="bid", root_action_ids={"safe": 0, "win": 1},
        ), 10,
    )
    outcomes = [prove_actor_information_set_v1(
        compilation, _actor_request(threshold=5, claim="winning_root_action_count"),
        maximum_nodes=100, maximum_worlds=2, maximum_seconds=allowance,
        native_action_quality=True, declared_maximum_seconds=3,
    ) for allowance in (2, 3)]
    assert outcomes[0].selected_action == "win"
    quality = outcomes[0].model_document()["actionQuality"]
    assert quality["bestValue"] == 7
    assert quality["optimalActions"] == ["win"]
    assert quality == outcomes[1].model_document()["actionQuality"]
    assert outcomes[0].model_document() == outcomes[1].model_document()


@pytest.mark.parametrize("policy_search", [None, "time_budgeted_symbolic_v1"])
def test_exact_root_replays_preserve_historical_proof_and_model_identities(policy_search):
    def tail(value):
        return FiniteInformationNodeV1.decision_observed('opponent', observation_id='reply', children={
            'H7': FiniteInformationNodeV1.terminal(value), 'H8': FiniteInformationNodeV1.terminal(value + 3)})

    problem = FiniteInformationSetProblemV1.build(problem_id="independent-root-worst-worlds", success_threshold=0,
        worlds=(FiniteInformationWorldV1.build("one", {
            "SA": tail(20), "S7": tail(-30)}),
            FiniteInformationWorldV1.build("two", {
                "SA": tail(10), "S7": tail(-10)})))
    compilation = ActorInformationSetCompilationV1(ActorInformationSetGraphV1(problem,
        {"count": 2, "complete": True}, {"kind": "test_allocation_model"}), 4)
    request = _actor_request(claim="winning_root_action_count")
    options = dict(maximum_nodes=100, maximum_worlds=2, maximum_seconds=3,
        native_action_quality=True, policy_search=policy_search)
    historical = prove_actor_information_set_v1(compilation, request, **options)
    lines = []
    current = prove_actor_information_set_v1(compilation, request, **options, _root_witnesses=lines)
    assert current.to_dict() == historical.to_dict()
    assert current.model_document() == historical.model_document()
    assert {(row['action'], row['worldId'], row['worstCaseValue']) for row in lines} == {
        ('SA', 'two', 10), ('S7', 'one', -30)}
    for row in lines:
        world = next(world for world in problem.worlds if world.world_id == row['worldId'])
        node = dict(world.roots)[row['action']]
        assert row['principalVariation'][0] == row['action']
        for action in row['principalVariation'][1:]:
            node = dict(node.children)[action]
        assert node.utility == row['worstCaseValue']
        assert len(row['principalVariation']) == 2
    incomplete_lines = []
    incomplete = prove_actor_information_set_v1(compilation, request,
        **{**options, 'maximum_nodes': 1}, _root_witnesses=incomplete_lines)
    assert incomplete.completion_status == 'proof_incomplete' and incomplete_lines == []
    raw = {key: value for key, value in request.to_dict().items() if key != 'sha256'}
    raw.update(mode='bounded', claim='outcome_reachable', horizon={'unit': 'round_end', 'value': 1},
        boundaryEvaluator='current_raw_point_margin_v1')
    bounded_request = ScenarioProofRequestV1.build(raw)
    bounded_lines = []
    bounded = prove_actor_information_set_v1(_bounded_compilation(problem, bounded_request, complete=True),
        bounded_request, **options, _root_witnesses=bounded_lines)
    assert bounded.completion_status == 'bounded_complete' and bounded_lines == []
    for invalid in ({}, ['stale']):
        with pytest.raises(ValueError, match='empty list'):
            prove_actor_information_set_v1(compilation, request, **options, _root_witnesses=invalid)


@pytest.mark.real_clock
@pytest.mark.parametrize("cancelled", [True, False])
def test_native_action_quality_exhaustion_never_returns_success(cancelled):
    from baloot_platform.research.ortools_information_set_compiler import (
        _certify_actor_action_quality_v1,
    )

    problem = FiniteInformationSetProblemV1.build(
        problem_id="quality-stop", success_threshold=0,
        worlds=(FiniteInformationWorldV1.build("one", {
            "S7": FiniteInformationNodeV1.terminal(1),
        }),),
    )
    receipt = OrToolsActorInformationSetProofV1().prove(problem)
    with pytest.raises(InformationSetCompilationStopped) as error:
        _certify_actor_action_quality_v1(
            receipt, action_ids={"S7": 0}, actor_relation="target",
            selected_action=None,
            deadline=time.monotonic() + (3 if cancelled else -1),
            cancel_check=(lambda: True) if cancelled else None,
        )
    assert error.value.partial_receipt["status"] == "incomplete"
    assert error.value.partial_receipt["stage"] == "action_quality"


@pytest.mark.parametrize("status_name", ["INFEASIBLE", "MODEL_INVALID"])
def test_native_quality_model_disagreement_is_quarantined_not_a_false_claim(
    monkeypatch, status_name,
):
    from ortools.sat.python import cp_model

    problem = FiniteInformationSetProblemV1.build(
        problem_id="quality-model-fault", success_threshold=0,
        worlds=(FiniteInformationWorldV1.build("one", {
            "S7": FiniteInformationNodeV1.terminal(7),
        }),),
    )
    monkeypatch.setattr(
        "baloot_platform.research.ortools_information_set_compiler._solve_cp_sat_with_cancel_v1",
        lambda *_args: getattr(cp_model, status_name),
    )
    outcome = prove_actor_information_set_v1(
        ActorInformationSetCompilationV1(ActorInformationSetGraphV1(
            problem, {"count": 1, "complete": True}, {"kind": "test_allocation_model"},
        ), 1),
        _actor_request(claim="winning_root_action_count"),
        maximum_nodes=100, maximum_worlds=1, maximum_seconds=3,
        native_action_quality=True,
    )
    assert outcome.completion_status == "model_error"
    assert outcome.claim_satisfied is False
    assert outcome.model_document()["modelStatus"] == "quarantined"
    assert outcome.model_document()["modelProto"]
    assert outcome.selected_action is None


def test_default_principal_variation_keeps_zero_above_negative() -> None:
    problem = FiniteInformationSetProblemV1.build(
        problem_id="zero-is-a-real-robust-value",
        success_threshold=1,
        worlds=(
            FiniteInformationWorldV1.build(
                "one",
                {
                    "H7": FiniteInformationNodeV1.terminal(-1),
                    "S7": FiniteInformationNodeV1.terminal(0),
                },
            ),
        ),
    )
    graph = ActorInformationSetGraphV1(
        problem,
        {"count": 1, "complete": True},
        {"kind": "test_allocation_model"},
    )
    outcome = prove_actor_information_set_v1(
        ActorInformationSetCompilationV1(graph, node_count=2),
        _actor_request(threshold=1.0),
        maximum_nodes=100,
        maximum_worlds=2,
        maximum_seconds=2.0,
    )

    assert outcome.selected_action == "S7"
    assert outcome.principal_variation == ("S7",)
    assert outcome.completion_status == "proof_failed"
    assert outcome.solver_status == "optimal"
    assert outcome.claim_satisfied is False
    assert outcome.model_document()["receiptCompletionStatus"] == "exact_complete"


def test_opponent_root_selects_the_minimum_target_utility() -> None:
    problem = FiniteInformationSetProblemV1.build(
        problem_id="opponent-root-minimizes",
        success_threshold=0,
        worlds=(
            FiniteInformationWorldV1.build(
                "one",
                {
                    "H7": FiniteInformationNodeV1.terminal(-3),
                    "S7": FiniteInformationNodeV1.terminal(4),
                },
            ),
        ),
    )
    graph = ActorInformationSetGraphV1(
        problem,
        {"count": 1, "complete": True},
        {"kind": "test_allocation_model"},
        root_actor="west",
        root_actor_team="east_west",
        root_actor_relation="opponent",
    )

    outcome = prove_actor_information_set_v1(
        ActorInformationSetCompilationV1(graph, node_count=2),
        _actor_request(claim="winning_root_action_count"),
        maximum_nodes=100,
        maximum_worlds=2,
        maximum_seconds=2.0,
    )

    assert outcome.completion_status == "exact_complete"
    assert outcome.selected_action == "H7"
    assert outcome.principal_variation == ("H7",)
    assert outcome.model_document()["rootActor"] == {
        "seat": "west",
        "team": "east_west",
        "relation": "opponent",
    }


@pytest.mark.parametrize("cancel_after_leaf", [False, True])
def test_graph_world_adapter_preserves_shared_leaf(rules, monkeypatch, cancel_after_leaf):
    graph = ProofGraphV1.build(scope="exact", root_node_id="root", nodes=(
        ProofGraphNodeV1.decision("root", "north_south", (
            ProofGraphEdgeV1.build(0, "a", "leaf"),
            ProofGraphEdgeV1.build(1, "b", "leaf")), actor_seat="north"),
        ProofGraphNodeV1.leaf("leaf", 1)))
    world = ActorInformationGraphWorldV1.build(world_id="shared", decision_kind="project",
        graph=graph, observation_ids={"root": "root-observation"})
    request = _actor_request(threshold=1, claim="winning_root_action_count")
    cancelled = [False]
    original = FiniteInformationNodeV1.terminal
    def terminal(cls, utility):
        result = original(utility)
        cancelled[0] = cancel_after_leaf
        return result
    monkeypatch.setattr(FiniteInformationNodeV1, "terminal", classmethod(terminal))
    def compile_world():
        return OrToolsActorInformationSetCompilerV1(rules).compile_graph_worlds(
            [world], request, maximum_nodes=2, maximum_depth=2, maximum_worlds=1, maximum_seconds=2,
            cancel_check=lambda: cancelled[0])
    if cancel_after_leaf:
        with pytest.raises(InformationSetCompilationStopped) as stopped:
            compile_world()
        assert stopped.value.reason == "cancelled"
        return
    compiled = compile_world()
    assert compiled.node_count == 2
    outcome = prove_actor_information_set_v1(compiled, request,
        maximum_nodes=2, maximum_worlds=1, maximum_seconds=2)
    assert outcome.winning_root_action_count == 2


@pytest.mark.parametrize("same_history", [False, True])
def test_compact_graph_shares_identical_forced_endpoints(rules, same_history):
    graph = ProofGraphV1.build(scope="exact", root_node_id="root", nodes=(
        ProofGraphNodeV1.decision("root", "north_south", (
            ProofGraphEdgeV1.build(0, "a", "left"),
            ProofGraphEdgeV1.build(1, "b", "right")), actor_seat="north"),
        *(ProofGraphNodeV1.decision(name, "west_east", (
            ProofGraphEdgeV1.build(0, "pass", "leaf"),), actor_seat="west")
          for name in ("left", "right")),
        ProofGraphNodeV1.leaf("leaf", 1)))
    world = ActorInformationGraphWorldV1.build(world_id="shared-forced", decision_kind="project",
        graph=graph, observation_ids={"root": "root", "left": "same",
                                     "right": "same" if same_history else "different"})
    request = _actor_request(threshold=1, claim="winning_root_action_count")
    limit = 2 if same_history else 3
    if not same_history:
        with pytest.raises(InformationSetCompilationStopped) as stopped:
            OrToolsActorInformationSetCompilerV1(rules).compile_graph_worlds(
                [world], request, maximum_nodes=2, maximum_depth=3, maximum_worlds=1,
                maximum_seconds=2, _compact_forced=True)
        assert stopped.value.reason == "maximum_nodes"
    compiled = OrToolsActorInformationSetCompilerV1(rules).compile_graph_worlds(
        [world], request, maximum_nodes=limit, maximum_depth=3, maximum_worlds=1,
        maximum_seconds=2, _compact_forced=True)
    assert compiled.node_count == limit
    proof = prove_actor_information_set_v1(compiled, request,
        maximum_nodes=limit, maximum_worlds=1, maximum_seconds=2)
    assert proof.winning_root_action_count == 2


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("cancel_observation", ["detour", "shared"])
def test_shared_graph_node_at_longer_path_still_checks_depth(rules, packed, compact, cancel_observation):
    graph = ProofGraphV1.build(scope="exact", root_node_id="root", nodes=(
        ProofGraphNodeV1.decision("root", "north_south", (
            ProofGraphEdgeV1.build(0, "short", "shared"),
            ProofGraphEdgeV1.build(1, "long", "detour")), actor_seat="north"),
        ProofGraphNodeV1.decision("detour", "north_south", (
            ProofGraphEdgeV1.build(0, "continue", "shared"),), actor_seat="north"),
        ProofGraphNodeV1.decision("shared", "north_south", (
            ProofGraphEdgeV1.build(0, "finish", "leaf"),), actor_seat="north"),
        ProofGraphNodeV1.leaf("leaf", 1)))
    world = ActorInformationGraphWorldV1.build(world_id="depth", decision_kind="project",
        graph=graph, observation_ids={key: key for key in ("root", "detour", "shared")})
    if packed:
        from baloot_platform.research.ortools_proof import ProofGraphSequencePackV1
        transport = ProofGraphSequencePackV1.build(graph, observation_ids=world.observation_ids)
        assert len(transport.graph.nodes) < len(graph.nodes)
        expanded, observations = transport.expand()
        restored = ActorInformationGraphWorldV1.build(world_id=world.world_id,
            decision_kind=world.decision_kind, graph=expanded, observation_ids=observations)
        assert restored.sha256 == world.sha256
        world = restored
    with pytest.raises(InformationSetCompilationStopped) as stopped:
        OrToolsActorInformationSetCompilerV1(rules).compile_graph_worlds(
            [world], _actor_request(threshold=1, claim="winning_root_action_count"),
            maximum_nodes=10, maximum_depth=2, maximum_worlds=1, maximum_seconds=2,
            _compact_forced=compact)
    assert stopped.value.reason == "maximum_depth"
    if compact:
        request = _actor_request(threshold=1, claim="winning_root_action_count")
        compiled = OrToolsActorInformationSetCompilerV1(rules).compile_graph_worlds(
            [world], request, maximum_nodes=3, maximum_depth=3,
            maximum_worlds=1, maximum_seconds=2, _compact_forced=True)
        assert compiled.node_count == 3
        proof = prove_actor_information_set_v1(compiled, request,
            maximum_nodes=3, maximum_worlds=1, maximum_seconds=2)
        assert proof.winning_root_action_count == 2
        cancelled = [False]
        class Observations(dict):
            def __getitem__(self, key):
                value = super().__getitem__(key)
                if key == cancel_observation:
                    cancelled[0] = True
                return value
        interrupted_world = replace(world, observation_ids=Observations(world.observation_ids))
        with pytest.raises(InformationSetCompilationStopped) as stopped:
            OrToolsActorInformationSetCompilerV1(rules).compile_graph_worlds(
                [interrupted_world], request, maximum_nodes=3, maximum_depth=3,
                maximum_worlds=1, maximum_seconds=2, _compact_forced=True,
                cancel_check=lambda: cancelled[0])
        assert stopped.value.reason == "cancelled"


def _nonplay_graph_world(
    decision_kind: str,
    world_id: str,
    *,
    reverse_hidden_payoff: bool,
    hidden_observation: str = "shared-hidden-observation",
    root_actor: str = "north",
    root_team: str = "north_south",
    bounded: bool = False,
) -> ActorInformationGraphWorldV1:
    root_id = f"{world_id}-root"
    choice_id = f"{world_id}-choice"
    x_id = f"{world_id}-x"
    y_id = f"{world_id}-y"
    force_id = f"{world_id}-force"
    x_value, y_value = (0, 1) if reverse_hidden_payoff else (1, 0)
    graph = ProofGraphV1.build(
        scope="bounded" if bounded else "exact",
        boundary_evaluator="current_raw_point_margin_v1" if bounded else None,
        root_node_id=root_id,
        nodes=(
            ProofGraphNodeV1.decision(
                root_id,
                root_team,
                (
                    ProofGraphEdgeV1.build(4, "hidden-choice", choice_id),
                    ProofGraphEdgeV1.build(9, "forced-win", force_id),
                ),
                actor_seat=root_actor,
            ),
            ProofGraphNodeV1.decision(
                choice_id,
                "north_south",
                (
                    ProofGraphEdgeV1.build(0, "x", x_id),
                    ProofGraphEdgeV1.build(1, "y", y_id),
                ),
                actor_seat="south",
            ),
            ProofGraphNodeV1.leaf(x_id, x_value, kind="boundary" if bounded else "terminal"),
            ProofGraphNodeV1.leaf(y_id, y_value, kind="boundary" if bounded else "terminal"),
            ProofGraphNodeV1.leaf(force_id, 1, kind="boundary" if bounded else "terminal"),
        ),
    )
    return ActorInformationGraphWorldV1.build(
        world_id=world_id,
        decision_kind=decision_kind,
        graph=graph,
        observation_ids={
            root_id: "shared-root-observation",
            choice_id: hidden_observation,
        },
    )


@pytest.mark.parametrize("decision_kind", ("bid", "escalation", "project"))
@pytest.mark.parametrize("bounded", (False, True))
def test_nonplay_hidden_worlds_cannot_choose_clairvoyantly(
    rules,
    decision_kind: str,
    bounded: bool,
) -> None:
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=bounded)
    worlds = (
        _nonplay_graph_world(
            decision_kind,
            f"{decision_kind}-world-a",
            reverse_hidden_payoff=False,
            bounded=bounded,
        ),
        _nonplay_graph_world(
            decision_kind,
            f"{decision_kind}-world-b",
            reverse_hidden_payoff=True,
            bounded=bounded,
        ),
    )
    request = (_bounded_actor_request if bounded else _actor_request)(
        threshold=1, claim="winning_root_action_count",
    )

    compilation = compiler.compile_graph_worlds(
        worlds,
        request,
        maximum_nodes=100,
        maximum_depth=8,
        maximum_worlds=2,
        maximum_seconds=2,
    )
    outcome = prove_actor_information_set_v1(
        compilation,
        request,
        maximum_nodes=100,
        maximum_worlds=2,
        maximum_seconds=2,
        native_action_quality=bounded,
    )

    roots = {row.action_code: row for row in outcome.root_actions}
    assert compilation.graph.decision_kind == decision_kind
    assert compilation.graph.root_action_ids == {
        "hidden-choice": 4,
        "forced-win": 9,
    }
    assert roots["hidden-choice"].value == 0
    assert roots["hidden-choice"].satisfies is False
    assert roots["forced-win"].value == 1
    assert outcome.winning_root_action_count == 1
    assert outcome.selected_action == "forced-win"
    assert outcome.completion_status == ("bounded_complete" if bounded else "exact_complete")
    assert compilation.graph.hidden_worlds["complete"] is not bounded
    assert outcome.model_document()["decisionKind"] == decision_kind
    assert compilation.graph.hidden_worlds["coverage"]["method"] == (
        "bounded_nonplay_observation_graphs_v1" if bounded else "exact_nonplay_observation_graphs_v1"
    )


@pytest.mark.parametrize("decision_kind", ("bid", "escalation", "project"))
@pytest.mark.parametrize("threshold,claim,winning_count", (
    (1, "winning_root_action_count", 1), (2, "outcome_reachable", 0),
))
def test_native_nonplay_actor_root_count_uses_certified_vectors(rules, decision_kind, threshold, claim, winning_count):
    """Exercise real finite-world CEGIS and qualification, not full auction search."""
    from baloot_platform.research.ortools_research_executor import (
        OrToolsNeutralCandidateV1,
        OrToolsResearchCompilerV1,
        OrToolsResearchStageReceiptV1,
        _proof_stage_bindings_v1,
    )
    from baloot_platform.research.scenario_contracts import ResearchScenarioV1

    request = _actor_request(threshold=threshold, claim=claim)
    compilation = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True).compile_graph_worlds(
        tuple(_nonplay_graph_world(decision_kind, f"world-{index}", reverse_hidden_payoff=bool(index))
              for index in range(2)), request,
        maximum_nodes=100, maximum_depth=8, maximum_worlds=2, maximum_seconds=2,
    )
    outcome = prove_actor_information_set_v1(compilation, request,
        maximum_nodes=100, maximum_worlds=2, maximum_seconds=2, native_action_quality=True)
    assert outcome.completion_status == ("proof_failed" if winning_count == 0 else "exact_complete")
    assert outcome.winning_root_action_count == winning_count
    assert "nativeQualificationVariables" not in outcome.model_document()

    def candidate_for(proof):
        return OrToolsNeutralCandidateV1("search", {},
            {"currentPlayer": "north", "decisionPoint": {"kind": decision_kind}}, (),
            {"proof": OrToolsResearchStageReceiptV1(
                "proof", proof.model_hash, proof.sha256, proof.sha256, proof.solver_status)})

    def scenario_for(expected):
        leaf = {"predicateId": "PRED-NONPLAY-COUNT", "kind": "proof.winning_root_action_count",
            "version": 1, "hardness": "hard", "arguments": {"minimum": expected, "maximum": expected}}
        return ResearchScenarioV1.build({"schemaVersion": 1, "origin": {"kind": "search"},
            "constraints": {**leaf, "predicateId": "PRED-NONPLAY-COUNT-REPORT", "hardness": "report"},
            "decisionPoint": {"kind": decision_kind}, "outcome": leaf})

    candidate = candidate_for(outcome)
    compiler = OrToolsResearchCompilerV1(native_cp_sat_authority=True)
    for expected, status in ((winning_count, "satisfied"), (1 - winning_count, "target_miss")):
        bindings = _proof_stage_bindings_v1(candidate, outcome, rules, require_native=True)
        context = bindings["proof"].context
        assert context.root_actions == ()
        assert len(context.root_winning_variables) == len(outcome.root_actions) == 2
        evaluation = compiler.evaluate(scenario_for(expected), candidate, bindings, maximum_seconds=2)
        assert evaluation.status == status
        if status == "satisfied":
            assert all(row["observed"]["winningCount"] == winning_count for row in evaluation.evidence)

    for attribute in ("root_winning_variables", "root_value_variables"):
        bindings = _proof_stage_bindings_v1(candidate, outcome, rules, require_native=True)
        context = bindings["proof"].context
        certified = getattr(context, attribute)
        for wrong in (certified[:-1], certified[::-1], (None, certified[1])):
            setattr(context, attribute, wrong)
            assert compiler.evaluate(scenario_for(1), candidate, bindings, maximum_seconds=2).status == "incomplete"

    incomplete = prove_actor_information_set_v1(compilation, request,
        maximum_nodes=1, maximum_worlds=2, maximum_seconds=2, native_action_quality=True)
    assert incomplete.completion_status == "proof_incomplete"
    with pytest.raises(ValueError, match="native proof model is missing root-action variable bindings"):
        _proof_stage_bindings_v1(candidate_for(incomplete), incomplete, rules, require_native=True)

    bounded_request = _bounded_actor_request(threshold=threshold,
        claim="outcome_forced" if winning_count else "outcome_reachable")
    bounded = prove_actor_information_set_v1(
        _bounded_compilation(compilation.graph.problem, bounded_request, complete=True), bounded_request,
        maximum_nodes=100, maximum_worlds=2, maximum_seconds=2, native_action_quality=True)
    assert bounded.completion_status == "proof_failed"
    assert bounded.hidden_worlds["complete"] is True
    assert bounded.model_document()["analysisScope"]["mode"] == "bounded"
    with pytest.raises(ValueError, match="do not own exhaustive proof evidence"):
        _proof_stage_bindings_v1(candidate_for(bounded), bounded, rules, require_native=True)


@pytest.mark.parametrize(
    "native,bounded_request,bounded_world,error",
    ((False, True, True, "require native CP-SAT"),
     (True, False, True, "scope/evaluator disagrees"),
     (True, True, False, "scope/evaluator disagrees")),
)
def test_nonplay_information_graph_scope_cannot_be_promoted(
    rules, native, bounded_request, bounded_world, error,
) -> None:
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=native)
    world = _nonplay_graph_world(
        "bid", "scope-world", reverse_hidden_payoff=False, bounded=bounded_world,
    )
    request = (_bounded_actor_request if bounded_request else _actor_request)(
        threshold=1, claim="winning_root_action_count",
    )
    with pytest.raises(ValueError, match=error):
        compiler.compile_graph_worlds(
            (world,), request, maximum_nodes=100, maximum_depth=8,
            maximum_worlds=1, maximum_seconds=2,
        )


def test_nonplay_distinguishable_worlds_may_choose_distinct_actions(rules) -> None:
    compiler = OrToolsActorInformationSetCompilerV1(rules)
    request = _actor_request(threshold=1, claim="winning_root_action_count")
    compilation = compiler.compile_graph_worlds(
        (
            _nonplay_graph_world(
                "bid",
                "visible-world-a",
                reverse_hidden_payoff=False,
                hidden_observation="visible-a",
            ),
            _nonplay_graph_world(
                "bid",
                "visible-world-b",
                reverse_hidden_payoff=True,
                hidden_observation="visible-b",
            ),
        ),
        request,
        maximum_nodes=100,
        maximum_depth=8,
        maximum_worlds=2,
        maximum_seconds=2,
    )
    outcome = prove_actor_information_set_v1(
        compilation,
        request,
        maximum_nodes=100,
        maximum_worlds=2,
        maximum_seconds=2,
    )

    assert outcome.winning_root_action_count == 2
    assert {row.value for row in outcome.root_actions} == {1}


def test_nonplay_opponent_root_uses_certified_action_ids_and_minimizes(rules) -> None:
    compiler = OrToolsActorInformationSetCompilerV1(rules)
    request = _actor_request(threshold=1, claim="winning_root_action_count")
    compilation = compiler.compile_graph_worlds(
        (
            _nonplay_graph_world(
                "escalation",
                "opponent-root",
                reverse_hidden_payoff=False,
                root_actor="west",
                root_team="west_east",
            ),
        ),
        request,
        maximum_nodes=100,
        maximum_depth=8,
        maximum_worlds=2,
        maximum_seconds=2,
    )
    outcome = prove_actor_information_set_v1(
        compilation,
        request,
        maximum_nodes=100,
        maximum_worlds=2,
        maximum_seconds=2,
    )

    assert compilation.graph.root_actor_relation == "opponent"
    assert outcome.selected_action == "hidden-choice"
    assert {row.action_code: row.action_id for row in outcome.root_actions} == {
        "hidden-choice": 4,
        "forced-win": 9,
    }


def test_nonplay_preflight_uses_measured_world_and_node_limits(rules) -> None:
    compiler = OrToolsActorInformationSetCompilerV1(rules)
    request = _actor_request(threshold=1, claim="winning_root_action_count")
    worlds = (
        _nonplay_graph_world("project", "budget-world-a", reverse_hidden_payoff=False),
        _nonplay_graph_world("project", "budget-world-b", reverse_hidden_payoff=True),
    )

    with pytest.raises(InformationSetCompilationStopped) as world_error:
        compiler.compile_graph_worlds(
            worlds,
            request,
            maximum_nodes=100,
            maximum_depth=8,
            maximum_worlds=1,
            maximum_seconds=2,
        )
    assert world_error.value.reason == "maximum_hidden_worlds"
    assert world_error.value.partial_receipt["evidence"] == {
        "estimatedHiddenWorlds": 2,
        "maximumHiddenWorlds": 1,
    }

    with pytest.raises(InformationSetCompilationStopped) as node_error:
        compiler.compile_graph_worlds(
            worlds[:1],
            request,
            maximum_nodes=1,
            maximum_depth=8,
            maximum_worlds=1,
            maximum_seconds=2,
        )
    assert node_error.value.reason == "maximum_nodes"


def test_hidden_allocation_upper_bound_is_exact_and_strict() -> None:
    assert hidden_allocation_upper_bound_v1((1, 1, 2)) == 12
    assert hidden_allocation_upper_bound_v1((8, 8, 8)) == 9_465_511_770
    with pytest.raises(ValueError, match="non-negative integers"):
        hidden_allocation_upper_bound_v1((1, True))


def test_fixed_observer_declaration_does_not_constrain_hidden_allocation(root) -> None:
    from baloot_platform.core.rules import RuleBook

    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    raw = json.loads((root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json")
                     .read_text(encoding="utf-8"))["exactTracePrefix"]
    raw.update(playActions=[], projectActions=[{"actor": "north", "action": "declare",
        "projectId": "PROJECT-63C3808618DAA0BB"}], projectDeclarations=[],
        projectAssertions=[], stateAssertions={})
    prefix = ExactTracePrefixV1.build(raw)
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(prefix)
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    fixed = compiler._public_fixed_ownership(exact)
    assert compiler._observer_project_facts_fixed(exact, fixed, ())

    state = copy.deepcopy(exact.state)
    state["events"].append({"type": "project_passed", "owner": "west"})
    assert not compiler._observer_project_facts_fixed(replace(exact, state=state), fixed, ())
    state = copy.deepcopy(exact.state)
    state["projectWinningTeam"] = "north_south"
    assert not compiler._observer_project_facts_fixed(replace(exact, state=state), fixed, ())
    state = copy.deepcopy(exact.state)
    state["projects"][0]["revealed"] = True
    assert not compiler._observer_project_facts_fixed(replace(exact, state=state), fixed, ())
    state = copy.deepcopy(exact.state)
    state["projects"][0]["awarded"] = False
    assert not compiler._observer_project_facts_fixed(replace(exact, state=state), fixed, ())
    state = copy.deepcopy(exact.state)
    state["projects"][0]["owner"] = "west"
    assert not compiler._observer_project_facts_fixed(replace(exact, state=state), fixed, ())
    state = copy.deepcopy(exact.state)
    state["events"].append({"type": "projects_compared", "winningTeam": None})
    assert not compiler._observer_project_facts_fixed(replace(exact, state=state), fixed, ())
    state = copy.deepcopy(exact.state)
    state["projects"][1]["awarded"] = True
    assert not compiler._observer_project_facts_fixed(replace(exact, state=state), fixed, ())
    missing = dict(fixed)
    missing.pop(next(card for card, owner in fixed.items() if owner == Seat.NORTH))
    assert not compiler._observer_project_facts_fixed(exact, missing, ())
    assert not compiler._observer_project_facts_fixed(exact, fixed, ("played",))

    worlds = compiler.enumerate_graph_world_inputs(exact, prefix, {"kind": "play"},
        maximum_worlds=8, maximum_seconds=10, bounded_world_scope=True)
    assert len(worlds) == 8
    assert len({item[0].state_hash for item in worlds}) == 8
    world, world_prefix = worlds[0]
    world.require_integrity()
    assert world.state["initialHands"]["north"] == exact.state["initialHands"]["north"]
    assert canonical_observer_information_hash_v1(complete_state=world.state, acting_seat="north") == (
        canonical_observer_information_hash_v1(complete_state=exact.state, acting_seat="north"))
    assert world_prefix.project_actions[0]["projectId"] == prefix.project_actions[0]["projectId"]
    invalid = world_prefix.to_dict()
    invalid.pop("sha256")
    invalid["projectActions"][0]["projectId"] = "PROJECT-0000000000000000"
    invalid_prefix = ExactTracePrefixV1.build(invalid)
    with pytest.raises(ValueError):
        OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(
            invalid_prefix)


@pytest.mark.real_clock
@pytest.mark.parametrize("pattern", (("pass", "pass"), ("declare", "pass"), ("pass", "declare")))
@pytest.mark.parametrize("contract,trump", (("sun", None), ("hokum", "spades")))
def test_native_hidden_project_rebinding_matches_reference_without_python_semantics(
    root, monkeypatch, pattern, contract, trump,
) -> None:
    from baloot_platform.core.rules import RuleBook
    from baloot_platform.core.project_ir import ProjectDomainIRV1

    rules = RuleBook.load(root / "configs/rules/saudi_project_optional_unratified_v1.yaml")
    raw = json.loads((root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(encoding="utf-8"))["exactTracePrefix"]
    seats = rules.project_domain.declaration_seats(Seat.parse(raw["dealer"]))
    raw["projectActions"] = [
        {"actor": seat.name.lower(), "action": action,
         **({"projectId": "PROJECT-0000000000000000"} if action == "declare" else {})}
        for seat, action in zip(seats, pattern)
    ]
    prefix = ExactTracePrefixV1.build(raw)
    hands = {seat: tuple(FULL_DECK[int(seat) * 8:(int(seat) + 1) * 8]) for seat in Seat}
    state = {"contract": contract, "trumpSuit": trump}
    expected = OrToolsActorInformationSetCompilerV1(rules)._rebind_project_actions(
        prefix, hands=hands, state=state, deadline=time.monotonic() + 10,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("native rebinding used Python project semantics")

    for method in ("detect", "declaration_state", "legal_declaration_actions", "apply_declaration_action"):
        monkeypatch.setattr(ProjectDomainIRV1, method, forbidden)
    actual = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)._rebind_project_actions(
        prefix, hands=hands, state=state, deadline=time.monotonic() + 10,
    )
    assert actual == expected


@pytest.mark.parametrize("dealer", list(Seat))
@pytest.mark.parametrize("actor", list(Seat))
def test_native_bid_world_symmetry_preserves_every_purchase_recipient(rules, dealer, actor):
    layout = DealLayoutV1.build(rules.structural_domain, dealer=dealer)
    native = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    legacy = OrToolsActorInformationSetCompilerV1(rules)
    groups = native._bid_hidden_group_positions(layout, actor)
    legacy_groups = legacy._bid_hidden_group_positions(layout, actor)
    assert sorted(position for group in groups for position in group) == sorted(
        position for group in legacy_groups for position in group
    )
    assert hidden_allocation_upper_bound_v1(tuple(map(len, groups))) * 16 == (
        hidden_allocation_upper_bound_v1(tuple(map(len, legacy_groups)))
    )
    deck = list(range(32))
    for buyer in Seat:
        completed = DealLayoutV1.build(rules.structural_domain, dealer=dealer, buyer=buyer)
        for group in groups:
            # Each transposition generates the complete within-group symmetry.
            for left, right in itertools.combinations(group, 2):
                swapped = deck.copy()
                swapped[left], swapped[right] = swapped[right], swapped[left]
                for seat in Seat:
                    for final in (False, True):
                        positions = completed.positions_for(seat, final=final)
                        assert {deck[index] for index in positions} == {swapped[index] for index in positions}
                assert swapped[layout.purchase_position] == deck[layout.purchase_position]


@pytest.mark.parametrize("stage", ("bid", "play"))
def test_native_bid_world_symmetry_preserves_replayed_observations(root, rules, stage):
    raw = json.loads((root / "examples" / "research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
        encoding="utf-8",
    ))["exactTracePrefix"]
    if stage == "bid":
        for key in ("auctionActions", "escalationActions", "projectDeclarations", "projectAssertions", "playActions"):
            raw[key] = []
        raw["stateAssertions"] = {}
    decision = {"kind": stage}
    prefix = ExactTracePrefixV1.build(raw)
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    original = normalizer.normalize(prefix, decision_point=decision)
    layout = DealLayoutV1.build(rules.structural_domain, dealer=Seat.parse(prefix.dealer))
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    altered = copy.deepcopy(raw)
    # Change every canonical group, including the other initial hands, while
    # retaining each possible purchase branch's ownership.
    for group in compiler._bid_hidden_group_positions(layout, Seat.NORTH):
        for destination, source in zip(group, reversed(group), strict=True):
            altered["deckOrder"][destination] = raw["deckOrder"][source]
    replayed = normalizer.normalize(ExactTracePrefixV1.build(altered), decision_point=decision)
    assert original.next_actor == replayed.next_actor
    assert original.legal_actions == replayed.legal_actions
    for seat in Seat:
        assert build_canonical_observer_information_v1(
            complete_state=original.state, acting_seat=seat.name.lower(),
        ) == build_canonical_observer_information_v1(
            complete_state=replayed.state, acting_seat=seat.name.lower(),
        )


@pytest.mark.real_clock
def test_native_bid_allocation_does_not_read_opponents_private_cards(root, rules, monkeypatch):
    import baloot_platform.research.ortools_information_set_compiler as module

    raw = json.loads((root / "examples" / "research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
        encoding="utf-8",
    ))["exactTracePrefix"]
    for key in ("auctionActions", "escalationActions", "projectDeclarations", "projectAssertions", "playActions"):
        raw[key] = []
    raw["stateAssertions"] = {}
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    original_prefix = ExactTracePrefixV1.build(raw)
    original = normalizer.normalize(original_prefix, decision_point={"kind": "bid"})
    actor = Seat.parse(original.next_actor)
    layout = DealLayoutV1.build(rules.structural_domain, dealer=Seat.parse(original_prefix.dealer))
    others = [seat for seat in Seat if seat != actor]
    left, right = (layout.positions_for(seat, final=False)[0] for seat in others[:2])
    raw["deckOrder"][left], raw["deckOrder"][right] = raw["deckOrder"][right], raw["deckOrder"][left]
    alternate_prefix = ExactTracePrefixV1.build(raw)
    alternate = normalizer.normalize(alternate_prefix, decision_point={"kind": "bid"})
    assert original.state_hash != alternate.state_hash
    assert canonical_observer_information_hash_v1(complete_state=original.state, acting_seat=actor.name.lower()) == (
        canonical_observer_information_hash_v1(complete_state=alternate.state, acting_seat=actor.name.lower())
    )
    models = []
    class Captured(Exception):
        pass
    def capture(_solver, model, _cancel):
        models.append(str(model.proto))
        raise Captured
    monkeypatch.setattr(module, "_solve_cp_sat_with_cancel_v1", capture)
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    known = {compiler._card(card) for card in original.state["hands"][actor.name.lower()]}
    known.add(compiler._card(original.state["purchaseCard"]))
    arguments = dict(
        pool=tuple(card for card in FULL_DECK if card not in known),
        group_sizes=tuple(map(len, compiler._bid_hidden_group_positions(layout, actor))),
        maximum_worlds=2, cancel_check=None,
    )
    for exact, prefix in ((original, original_prefix), (alternate, alternate_prefix)):
        with pytest.raises(Captured):
            compiler._enumerate_hidden_groups(**arguments, deadline=time.monotonic() + 10,
                exact=exact, bid_prefix=prefix)
    assert models[0] == models[1]
    with pytest.raises(ValueError, match="context differs"):
        compiler._enumerate_hidden_groups(**arguments, deadline=time.monotonic() + 10,
            exact=original, bid_prefix=alternate_prefix)
    assert len(models) == 2

    observed = [compiler._native_bid_root_model(exact, prefix, visibility="actor_information_set")
                for exact, prefix in ((original, original_prefix), (alternate, alternate_prefix))]
    assert str(observed[0].model.proto) == str(observed[1].model.proto)
    complete = [compiler._native_bid_root_model(exact, prefix, visibility="complete_state")
                for exact, prefix in ((original, original_prefix), (alternate, alternate_prefix))]
    assert str(complete[0].model.proto) != str(complete[1].model.proto)
    with pytest.raises(ValueError, match="differs"):
        compiler._native_bid_root_model(original, alternate_prefix, visibility="complete_state")
    with pytest.raises(ValueError, match="explicit visibility"):
        compiler._native_bid_root_model(original, original_prefix, visibility="unknown")


@pytest.mark.parametrize("history_steps", (0, 1, 2, 8))
def test_native_post_auction_project_root_preserves_exact_deal(rules, history_steps):
    from ortools.sat.python import cp_model
    from baloot_platform.research.ortools_play_tail import TailPrefixActionV1
    prefix = ExactTracePrefixV1.build({"schemaVersion": 1, "deckOrder": list(range(32)),
        "dealer": "east", "auctionActions": [{"actor": "north", "action": "sun"}],
        "escalationActions": [], "projectDeclarations": [], "projectActions": [], "playActions": []})
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    exact = normalizer.normalize_play_or_project(prefix)
    for _ in range(history_steps):
        prefix = normalizer.append_normalized_action(prefix, exact, sorted(row["code"] for row in exact.legal_actions)[0])
        exact = normalizer.normalize_play_or_project(prefix)
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    model = compiler._native_post_auction_root_model(exact, prefix,
        opportunity_order=("west", "east", "north", "south"), escalation_events=[])
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 10
    assert solver.solve(model.model) == cp_model.OPTIMAL
    assert [solver.value(card) for card in model.card_at_position] == list(prefix.deck_order)
    assert model._complete_information_deck == prefix.deck_order
    assert solver.value(model._native_round_fragment.escalation.native["length"]) == 0
    actor = Seat.parse(exact.next_actor)
    branch = model._native_round_fragment
    if exact.decision_kind == "project":
        legal = {code for code, literal in branch.projects.declaration_actions[int(actor)].items() if solver.value(literal)}
    else:
        play = branch.play._extract(solver, cp_model.OPTIMAL, model._model_hash(search_branching="automatic"))
        legal = {action.card.code + ("+ekka" if action.ekka else "")
                 for action in play.legal_play_actions_by_step[len(prefix.play_actions)]}
        assert [row.to_dict() for row in play.plays[:len(prefix.play_actions)]] == [
            TailPrefixActionV1.build(row["actor"], row["card"], ekka=row.get("ekka", False)).to_dict()
            for row in prefix.play_actions]
    assert legal == {row["code"] for row in exact.legal_actions}
    with pytest.raises(InformationSetCompilationStopped, match="cancelled"):
        compiler._native_post_auction_root_model(exact, prefix, opportunity_order=tuple(Seat),
            escalation_events=[], cancel_check=lambda: True)


@pytest.mark.parametrize("limit,expected", (("cancel", "cancelled"),
    ("bytes", "maximum_model_bytes"), ("queries", "maximum_queries"), ("compile", "cancelled")))
def test_native_post_auction_bounds_preserve_resource_exhaustion(rules, monkeypatch, limit, expected):
    prefix = ExactTracePrefixV1.build({"schemaVersion": 1, "deckOrder": list(range(32)),
        "dealer": "east", "auctionActions": [{"actor": "north", "action": "sun"}],
        "escalationActions": [], "projectDeclarations": [], "projectActions": [], "playActions": []})
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(prefix)
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    if limit == "compile":
        from baloot_platform.research.ortools_play_tail import PlayTailSolveStopped
        def stopped(*args, **kwargs):
            raise PlayTailSolveStopped("cancelled", evidence={"constructionStarted": True})
        monkeypatch.setattr(compiler, "_native_post_auction_root_model", stopped)
    with pytest.raises(InformationSetCompilationStopped) as caught:
        compiler._solve_native_post_auction_policy_bounds(exact, prefix, Seat.parse(exact.next_actor).team,
            selected_root_action=exact.legal_actions[0]["code"],
            opportunity_order=("west", "east", "north", "south"), escalation_events=[],
            maximum_iterations=10, maximum_queries=1 if limit == "queries" else 10,
            maximum_observations=100, maximum_model_bytes=1 if limit == "bytes" else 100_000_000,
            maximum_conflicts=250_000, maximum_depth=64, maximum_seconds=30,
            cancel_check=lambda: limit == "cancel")
    assert caught.value.reason == expected
    receipt = caught.value.partial_receipt["evidence"]
    assert receipt["status"] == "incomplete" and receipt["strategicCoverage"] is False
    assert receipt["queriesUsed"] == (1 if limit == "queries" else 0)
    assert receipt["strategy"] is None


@pytest.mark.parametrize("visibility", ("complete_state", "actor_information_set"))
@pytest.mark.parametrize("branch_count", (1, 2))
def test_native_bid_strategy_compacts_only_complete_state_ownership(rules, monkeypatch, visibility, branch_count):
    prefix = ExactTracePrefixV1.build({"schemaVersion": 1, "deckOrder": list(range(32)),
        "dealer": "east", "auctionActions": [], "escalationActions": [],
        "projectDeclarations": [], "playActions": []})
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(prefix, decision_point={"kind": "bid"})
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    def prepare(*args, **kwargs):
        assert kwargs["compact_known_initial_ownership"] is (visibility == "complete_state")
        return (None,) * branch_count, {"constructionComplete": True}
    def solve(*args, **kwargs):
        assert kwargs["tighten_utility_ceiling"] is (branch_count == 1)
        assert kwargs["shorten_counterexamples"] is (visibility == "complete_state" and branch_count > 1)
        key = "opponent_ceiling_policy" if branch_count == 1 else "_unavailable_opponent_ceiling_policy"
        assert kwargs[key] == ceiling_policy
        assert ("_unavailable_opponent_ceiling_policy" in kwargs) is (branch_count > 1)
        assert kwargs["fixed_root"] is None
        raise InformationSetCompilationStopped("cancelled", stage="test_strategy")
    monkeypatch.setattr(compiler, "_prepare_native_bid_branches", prepare)
    monkeypatch.setattr(compiler, "_solve_symbolic_round_policy_bounds", solve)
    ceiling_policy = {"auction_action": "pass", "escalation_action": "decline",
        "native_play_policy_id": "observer_safe_trick_control_v1"}
    with pytest.raises(InformationSetCompilationStopped, match="cancelled"):
        compiler._solve_native_bid_policy_bounds(exact, prefix, Seat.parse(exact.next_actor).team,
            visibility=visibility, maximum_iterations=10, maximum_queries=20,
            maximum_observations=100, maximum_branches=13, maximum_model_bytes=100_000_000,
            maximum_conflicts=100_000, maximum_seconds=10, opponent_ceiling_policy=ceiling_policy)


@pytest.mark.real_clock
@pytest.mark.parametrize("visibility", ("complete_state", "actor_information_set"))
@pytest.mark.parametrize("fallback", (None, "observer_safe_trick_control_v1"))
def test_native_single_branch_bid_threshold_shortens_counterexamples(rules, monkeypatch, visibility, fallback):
    from types import SimpleNamespace
    prefix = ExactTracePrefixV1.build({"schemaVersion": 1, "deckOrder": list(range(32)), "dealer": "east"})
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "bid"})
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    model = SimpleNamespace(_auction_observation_hash=lambda *args: "root-observation")
    monkeypatch.setattr(compiler, "_prepare_native_bid_branches",
        lambda *args, **kwargs: ((model,), {"constructionComplete": True}))

    def solve(*args, **kwargs):
        assert kwargs["required_threshold"] == 44
        assert kwargs["shorten_counterexamples"] is (fallback is None)
        assert kwargs["tighten_utility_ceiling"] is False
        assert kwargs["fixed_root"][2] == "sun"
        raise InformationSetCompilationStopped("cancelled", stage="test_strategy")

    monkeypatch.setattr(compiler, "_solve_symbolic_round_policy_bounds", solve)
    with pytest.raises(InformationSetCompilationStopped, match="cancelled"):
        compiler._solve_native_bid_policy_bounds(exact, prefix, Seat.parse(exact.next_actor).team,
            visibility=visibility, selected_root_action="sun", required_threshold=44,
            strategy_play_fallback_id=fallback, maximum_iterations=10, maximum_queries=20,
            maximum_observations=100, maximum_branches=13, maximum_model_bytes=100_000_000,
            maximum_conflicts=100_000, maximum_seconds=10)


def test_native_bid_reuses_owned_construction_but_repeats_proof_queries(rules, monkeypatch):
    prefix = ExactTracePrefixV1.build({"schemaVersion": 1, "deckOrder": list(range(32)),
        "dealer": "east", "auctionActions": [], "escalationActions": [],
        "projectDeclarations": [], "playActions": []})
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "bid"})
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    built = []
    original = compiler._build_native_bid_branch

    def build(*args, **kwargs):
        built.append(kwargs["branch"])
        return original(*args, **kwargs)

    monkeypatch.setattr(compiler, "_build_native_bid_branch", build)
    arguments = dict(visibility="complete_state", selected_root_action="sun",
        compact_purchase_invariant_ownership=True, maximum_branches=13,
        maximum_model_bytes=100_000_000, maximum_conflicts=250_000,
        maximum_queries=100, maximum_depth=64, maximum_seconds=30)
    models, first = compiler._prepare_native_bid_branches(exact, prefix, **arguments)
    assert models and built and first["constructionComplete"]
    built.clear()
    reused, second = compiler._prepare_native_bid_branches(exact, prefix, **arguments)
    assert not built and all(a is b for a, b in zip(models, reused, strict=True))
    assert second["constructionComplete"] and second["queriesUsed"] == first["queriesUsed"] > 0
    assert [row.get("sourceModelHash") for row in first["branches"]] == [
        row.get("sourceModelHash") for row in second["branches"]]
    # A different configured opportunity order requires fresh construction.
    arguments["opportunity_order"] = ("north", "west", "south", "east")
    models, changed = compiler._prepare_native_bid_branches(exact, prefix, **arguments)
    assert built and changed["scope"]["escalationOpportunityOrder"] == list(arguments["opportunity_order"])
    # A caller accidentally mutating an owned model fails closed, not as exact proof.
    models[0].model.add(0 == 1)
    with pytest.raises(ValueError, match="construction was mutated"):
        compiler._prepare_native_bid_branches(exact, prefix, **arguments)
    assert compiler._native_bid_models is None


@pytest.mark.parametrize("visibility", ("complete_state", "actor_information_set"))
def test_native_bid_root_branch_covers_redeal_without_candidate_constraints(rules, visibility):
    prefix = ExactTracePrefixV1.build({"schemaVersion": 1, "deckOrder": list(range(32)),
        "dealer": "east", "auctionActions": [], "escalationActions": [],
        "projectDeclarations": [], "playActions": []})
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "bid"})
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    specs = compiler._native_bid_branch_specs(Seat.EAST)
    assert specs[0] is None and len(specs) == len(set(specs)) == 13
    assert {(contract, trump) for contract, trump, _, _ in specs[1:]} == {
        (Contract.SUN, None), (Contract.ASHKAL, None), *((Contract.HOKUM, suit) for suit in Suit)}
    assert {locked for _, _, _, locked in specs[1:]} == {False, True}
    model = compiler._build_native_bid_branch(exact, prefix, visibility=visibility, branch=None)
    result = model.find_native_round_margin_below(target=Team.NORTH_SOUTH, threshold=1,
        maximum_seconds=5, maximum_conflicts=100_000)
    assert result["status"] == "witness_found"
    assert result["candidate"]["allPassed"] and len(result["candidate"]["auctionActions"]) == 8
    if visibility == "complete_state":
        assert result["candidate"]["deckOrder"] == list(prefix.deck_order)
    else:
        layout = DealLayoutV1.build(rules.structural_domain, dealer=Seat.EAST)
        positions = layout.positions_for(Seat.NORTH, final=False)
        assert {result["candidate"]["deckOrder"][index] for index in positions} == {
            prefix.deck_order[index] for index in positions}
    with pytest.raises(InformationSetCompilationStopped, match="cancelled"):
        compiler._build_native_bid_branch(exact, prefix, visibility=visibility,
            branch=None, cancel_check=lambda: True)
    with pytest.raises(ValueError, match="unknown native bid branch"):
        compiler._build_native_bid_branch(exact, prefix, visibility=visibility, branch=("invalid",))


@pytest.mark.parametrize("contract", tuple(Contract))
def test_native_bid_purchased_branch_compiles_all_decision_stages(rules, contract):
    prefix = ExactTracePrefixV1.build({"schemaVersion": 1, "deckOrder": list(range(32)),
        "dealer": "east", "auctionActions": [], "escalationActions": [],
        "projectDeclarations": [], "playActions": []})
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "bid"})
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    spec = next(row for row in compiler._native_bid_branch_specs(Seat.EAST)
                if row is not None and row[0] == contract and row[3] is False)
    scenario = compiler._build_native_bid_branch(exact, prefix, visibility="complete_state", branch=spec)
    assert scenario.model.validate() == ""
    assert scenario.auction_fragment.prefix_length == 0
    branch = scenario._native_round_fragment
    assert branch.scope["contract"] == contract.value
    assert len(branch.play.play_state_context.play_slots) == 32
    assert branch.escalation_opportunities and branch.projects is not None
    assert len(branch.scoring.awarded_points) == 2
    assert not any((scenario._observed_auction_bindings, scenario._observed_escalation_bindings,
                    scenario._observed_project_bindings, scenario._observed_play_bindings))
    assert not branch.scope["strategicCoverage"]


def test_observed_bid_projects_compact_only_visible_ownership(rules, monkeypatch):
    from baloot_platform.research import ortools_project_fragment
    from baloot_platform.research.ortools_play_tail import OrToolsPlayTailStageV1

    prefix = ExactTracePrefixV1.build({"schemaVersion": 1, "deckOrder": list(range(32)),
        "dealer": "east", "auctionActions": [], "escalationActions": [],
        "projectDeclarations": [], "playActions": []})
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "bid"})
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    spec = next(row for row in compiler._native_bid_branch_specs(Seat.EAST)
                if row is not None and row[0] == Contract.SUN and not row[3])
    recorded = []
    play_ownership = []
    attach = ortools_project_fragment.attach_project_lifecycle_fragment_v1
    build_play = OrToolsPlayTailStageV1.build_model

    def capture(*args, **kwargs):
        recorded.append(kwargs)
        return attach(*args, **kwargs)

    monkeypatch.setattr(ortools_project_fragment, "attach_project_lifecycle_fragment_v1", capture)
    def capture_play(self, **kwargs):
        play_ownership.append(kwargs["ownership"])
        return build_play(self, **kwargs)
    monkeypatch.setattr(OrToolsPlayTailStageV1, "build_model", capture_play)
    compact = compiler._build_native_bid_branch(exact, prefix,
        visibility="actor_information_set", branch=spec)
    baseline = compiler._native_bid_root_model(exact, prefix, visibility="actor_information_set")
    baseline.add_native_round_continuation(contract=spec[0], trump=spec[1],
        first_leader=spec[2], locked=spec[3], match_score=prefix.match_score)
    actor = Seat.parse(exact.next_actor)
    visible = {compiler._card(card).id for card in exact.state["hands"][actor.name.lower()]}
    assert recorded[0]["fixed_card_owners"] == tuple(
        actor if card_id in visible else None for card_id in range(32))
    assert recorded[0]["compact_fixed_states"] is True
    assert recorded[1]["fixed_card_owners"] is None
    # The same five visible owners reach play; no hidden card is specialized.
    assert play_ownership == [{Card(card_id): actor for card_id in visible}, None]
    assert compact.model.validate() == baseline.model.validate() == ""
    assert len(compact.model.proto.variables) < len(baseline.model.proto.variables)
    assert len(compact.model.proto.constraints) < len(baseline.model.proto.constraints)
    assert all(max(tight.proto.domain) < max(loose.proto.domain) for tight, loose in zip(
        compact._native_round_fragment.scoring.awarded_points,
        baseline._native_round_fragment.scoring.awarded_points, strict=True))


@pytest.mark.real_clock
@pytest.mark.parametrize("visibility", ("complete_state", "actor_information_set"))
@pytest.mark.parametrize("enumerate_actions", (False, True))
def test_native_bid_policy_owned_pipeline_preserves_incomplete_evidence(root, rules, visibility, enumerate_actions):
    from baloot_platform.config import load_profile_payload

    prefix = ExactTracePrefixV1.build({"schemaVersion": 1, "deckOrder": list(range(32)),
        "dealer": "east", "auctionActions": [], "escalationActions": [],
        "projectDeclarations": [], "playActions": []})
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "bid"})
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    profile = load_profile_payload("challenge", root / "configs/challenge/intelligent_research.yaml", root=root)
    tier = profile["scenarioResearchV4"]["proofBudgets"]["profiles"]["offline_heavyweight_v1"]
    arguments = dict(visibility=visibility, maximum_iterations=1, maximum_queries=1,
        maximum_observations=1000, maximum_branches=13, maximum_model_bytes=tier["maximumCacheBytes"],
        maximum_conflicts=100_000, maximum_seconds=60)
    with pytest.raises(InformationSetCompilationStopped) as caught:
        compiler._solve_native_bid_policy_bounds(exact, prefix, Team.NORTH_SOUTH,
            **({"enumerate_root_actions": True} if enumerate_actions else {"selected_root_action": "pass"}), **arguments)
    receipt = caught.value.partial_receipt
    assert receipt["stage"] == "native_bid_policy_bounds"
    evidence = receipt["evidence"]
    assert evidence["status"] == "incomplete" and not evidence["strategicCoverage"]
    assert evidence["sha256"] == canonical_sha256({key: value for key, value in evidence.items() if key != "sha256"})
    assert evidence["preparation"] is None
    assert evidence["cause"]["stage"] == "native_bid_preparation"
    assert caught.value.reason == "maximum_queries"
    assert evidence["activeRootAction" if enumerate_actions else "selectedRootAction"] == (None if enumerate_actions else "pass")
    if enumerate_actions:
        assert evidence["requestedRootActions"] == sorted(row["code"] for row in exact.legal_actions)
        assert evidence["rootActionValues"] == []
    preparation = evidence["cause"]["evidence"]
    assert not preparation["constructionComplete"]
    assert preparation["queriesUsed"] == preparation["limits"]["queries"] == 1
    assert evidence["queriesUsed"] == 1
    assert preparation["limits"]["seconds"] < arguments["maximum_seconds"]
    for overrides, reason in (({"maximum_branches": 1}, "maximum_branches"),
                              ({"cancel_check": lambda: True}, "cancelled")):
        with pytest.raises(InformationSetCompilationStopped) as stopped:
            compiler._solve_native_bid_policy_bounds(exact, prefix, Team.NORTH_SOUTH,
                **{**arguments, **overrides})
        assert stopped.value.reason == reason
        assert stopped.value.partial_receipt["evidence"]["preparation"] is None
    with pytest.raises(ValueError):
        compiler._solve_native_bid_policy_bounds(exact, prefix, Team.NORTH_SOUTH,
            **{**arguments, "maximum_queries": True})
    for action in (True, "invalid"):
        with pytest.raises(ValueError, match="fixed bid action"):
            compiler._solve_native_bid_policy_bounds(exact, prefix, Team.NORTH_SOUTH,
                selected_root_action=action, **arguments)


@pytest.mark.parametrize("prior_pass", (False, True))
def test_native_bid_action_enumeration_reuses_preparation_and_shares_budgets(rules, monkeypatch, prior_pass):
    # Orchestration check only; native strategy correctness has separate fixtures.
    prefix = ExactTracePrefixV1.build({"schemaVersion": 1, "deckOrder": list(range(32)),
        "dealer": "east", "auctionActions": [{"actor": "north", "action": "pass"}] if prior_pass else [], "escalationActions": [],
        "projectDeclarations": [], "playActions": []})
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(prefix, decision_point={"kind": "bid"})
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    model = compiler._native_bid_root_model(exact, prefix, visibility="complete_state")
    actor = Seat.parse(exact.next_actor)
    target_team = actor.team
    expected_observation = model._auction_observation_hash(actor, exact.state["hands"][actor.name.lower()],
        exact.state["purchaseCard"], ["pass"] if prior_pass else [])
    original = str(model.model.proto)
    prepared, calls = [], []
    def prepare(*args, **kwargs):
        prepared.append(kwargs)
        return (model,), {"constructionComplete": True}
    def solve(models, target, **kwargs):
        assert len(models) == 1 and models[0] is not model
        assert kwargs["tighten_utility_ceiling"] is True
        assert str(model.model.proto) == original
        calls.append(kwargs)
        observation, domain, action = kwargs["fixed_root"]
        assert observation == expected_observation
        from ortools.sat.python import cp_model
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.max_time_in_seconds = 5
        assert solver.solve(models[0].model) == cp_model.OPTIMAL
        selected = model.auction_fragment.choices[len(prefix.auction_actions)][action]
        assert solver.boolean_value(selected)
        opposite = models[0].model.clone()
        opposite.add(selected == 0)
        assert solver.solve(opposite) == cp_model.INFEASIBLE
        return {"status": "scoped_bound_converged", "strategicCoverage": False,
            "queries": [{}], "masters": [{}], "domains": {observation: domain}, "policy": {observation: action}}
    monkeypatch.setattr(compiler, "_prepare_native_bid_branches", prepare)
    monkeypatch.setattr(compiler, "_solve_symbolic_round_policy_bounds", solve)
    actions = sorted(row["code"] for row in exact.legal_actions)
    assert len(actions) > 1
    limits = dict(visibility="complete_state", enumerate_root_actions=True, maximum_iterations=len(actions),
        maximum_queries=len(actions), maximum_observations=len(actions), maximum_branches=13,
        maximum_model_bytes=1_000_000, maximum_conflicts=1000, maximum_seconds=10)
    result = compiler._solve_native_bid_policy_bounds(exact, prefix, target_team, **limits)
    assert len(prepared) == 1 and len(calls) == len(actions)
    assert [row["action"] for row in result["rootActionValues"]] == actions
    assert result["activeRootAction"] is None and not result["strategicCoverage"]
    assert result["status"] == "scoped_action_values_complete"
    assert [row["maximum_queries"] for row in calls] == list(range(len(actions), 0, -1))
    assert result["sha256"] == canonical_sha256({key: value for key, value in result.items() if key != "sha256"})
    with pytest.raises(InformationSetCompilationStopped) as caught:
        compiler._solve_native_bid_policy_bounds(exact, prefix, target_team, **{**limits, "maximum_queries": 1})
    partial = caught.value.partial_receipt["evidence"]
    assert caught.value.reason == "maximum_queries"
    assert len(partial["rootActionValues"]) == 1 and partial["activeRootAction"] == actions[1]
    assert partial["status"] == "incomplete" and not partial["strategicCoverage"]
    for changes in ({"enumerate_root_actions": 1}, {"selected_root_action": "pass"}):
        with pytest.raises(ValueError, match="choose one"):
            compiler._solve_native_bid_policy_bounds(exact, prefix, target_team, **{**limits, **changes})


@pytest.mark.parametrize("case", ("complete", "native_ceiling", "interval", "interval_allowance_exceeded", "interval_preparation_incomplete", "witness", "unknown", "empty", "budget", "cancelled",
    "qualification_unknown", "qualification_budget", "qualification_cancelled", "model_mismatch",
    "partial_partition", "selected_gap", "selected_partition", "missing_partition", "duplicate_partition",
    "unknown_omission", "unproved_policy_omission", "missing_model", "wrong_root", "wrong_policy",
    "wrong_rules", "wrong_action", "wrong_partition", "selected_model", "prepared_model",
    "selected_usage_missing", "selected_usage_mismatch", "selected_usage_exceeded"))
@pytest.mark.parametrize("order", (None, ("west", "south", "east", "north")))
def test_native_bid_dominance_preserves_unknown_ties_and_partial_bounds(rules, monkeypatch, case, order,
        visibility="complete_state", alternative_first=False):
    # Coordinator-only test; native strategy/query semantics have separate fixtures.
    from types import SimpleNamespace
    interval = case.startswith("interval")
    success = case in {"complete", "native_ceiling", "interval"}
    prefix = ExactTracePrefixV1.build({"schemaVersion": 1, "deckOrder": list(range(32)),
        "dealer": "east", "auctionActions": [], "escalationActions": [],
        "projectDeclarations": [], "playActions": []})
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(prefix, decision_point={"kind": "bid"})
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    target = Seat.parse(exact.next_actor).team
    requests = []
    partitions = [{"partitionIndex": index, "retained": index == 0,
        "sourceModelHash": "owned-test-model",
        "reachability": {"solverStatus": "infeasible"}} for index in
        range(len(compiler._native_bid_branch_specs(Seat.parse(prefix.dealer))))]
    scope = {"rulesHash": rules.sha256, "rootHash": exact.state_hash, "visibility": "complete_state",
        "selectedRootAction": "sun", "matchScore": list(prefix.match_score),
        "continuationBoundary": "current_deal_completion", "nextDealAnalyzed": False,
        "partition": [None if row is None else {"contract": row[0].value,
            "trump": row[1].name.lower() if row[1] is not None else None,
            "firstLeader": row[2].name.lower(), "locked": row[3]}
            for row in compiler._native_bid_branch_specs(Seat.parse(prefix.dealer))]}
    if order is not None:
        scope["escalationOpportunityOrder"] = list(order)
    selected = {"status": "scoped_bound_converged", "queriesUsed": 1,
        "preparation": {"queriesUsed": 0, "constructionComplete": case != "selected_partition", "branches": partitions, "scope": scope},
        "strategy": {"queriesUsed": 1, "status": "scoped_bound_converged", "upperBound": 92,
            "scope": {"branches": [{"modelHash": "substituted" if case == "selected_model" else "owned-test-model"}]},
            "lowerBound": 91 if case == "selected_gap" else 92, "queries": [{}]}}
    if visibility == "actor_information_set":
        scope.update(visibility=visibility, rootHash=canonical_observer_information_hash_v1(
            complete_state=exact.state, acting_seat=exact.next_actor))
        info = {"kind": "native_bid_information_bindings_v1", "strategicCoverage": False,
                "cpModelProtoHash": "a" * 64}
        partitions[0].update(cpModelProtoHash="a" * 64,
            informationBindings={**info, "sha256": canonical_sha256(info)})
        root_model = compiler._native_bid_root_model(exact, prefix, visibility=visibility)
        actor = Seat.parse(exact.next_actor)
        selected["strategy"]["scope"]["fixedRoot"] = {
            "observationHash": root_model._auction_observation_hash(actor,
                [compiler._card(card) for card in exact.state["hands"][actor.name.lower()]],
                compiler._card(exact.state["purchaseCard"]), []),
            "legalActions": sorted(row["code"] for row in exact.legal_actions),
            "selectedAction": "sun"}
        if case == "missing_information":
            partitions[0].pop("informationBindings")
        elif case == "wrong_observed_action":
            selected["strategy"]["scope"]["fixedRoot"]["selectedAction"] = "pass"
        elif case == "wrong_observed_domain":
            selected["strategy"]["scope"]["fixedRoot"]["legalActions"] = ["sun"]
    def selected_bounds(*a, **kw):
        assert kw["visibility"] == visibility
        assert kw.get("opportunity_order") == order
        if interval:
            assert kw["incumbent_only"] is True and kw["incumbent_play_policies"] == {"test": "policy"}
            assert 0 < kw["maximum_seconds"] <= 2 and kw["maximum_queries"] == 10
        if case == "native_ceiling":
            assert kw["opponent_ceiling_policy"]["native_play_policy_id"] == limits["opponent_play_policy_id"]
        if alternative_first:
            assert len(requests) == len(exact.legal_actions) - 1
            assert kw["incumbent_threshold"] == 92
        return selected
    if interval:
        selected["status"] = selected["strategy"]["status"] = (
            "scoped_incumbent_threshold_complete" if alternative_first else "scoped_incumbent_bound_complete")
        selected["strategy"]["upperBound"] = 100
        if case == "interval_allowance_exceeded":
            selected["queriesUsed"] = selected["strategy"]["queriesUsed"] = 11
    monkeypatch.setattr(compiler, "_solve_native_bid_policy_bounds", selected_bounds)
    if case == "selected_usage_missing":
        selected.pop("queriesUsed")
    elif case == "selected_usage_mismatch":
        selected["queriesUsed"] = 0
    elif case == "selected_usage_exceeded":
        selected["queriesUsed"] = selected["strategy"]["queriesUsed"] = 10**9

    def query(**kw):
        requests.append(kw)
        assert kw["target"] == kw["policy_team"] == target.other
        assert kw["threshold"] == -92
        assert kw["auction_action"] == "pass" and kw["escalation_action"] == "decline"
        return {"status": "incomplete" if case == "unknown" else "queried",
            "sourceModelHash": "substituted-model" if case == "model_mismatch" else "owned-test-model",
            "bindings": [{"status": "bound"}, {"status": "bound"}],
            "query": {"status": "witness_found" if case == "witness" else "no_matching_outcome",
                "solverStatus": "optimal" if case == "witness" else "unknown" if case == "unknown" else "infeasible"}}

    def prepare(*a, **kw):
        assert kw["visibility"] == ("complete_state" if alternative_first else visibility)
        assert kw.get("opportunity_order") == order
        assert kw["selected_root_action"] != "sun"
        alternatives = copy.deepcopy(partitions)
        if case == "unknown_omission":
            alternatives[-1]["reachability"]["solverStatus"] = "unknown"
        elif case == "unproved_policy_omission":
            alternatives[-1]["reachability"]["policyBindings"] = [{"status": "incomplete"}]
        elif case == "missing_model":
            alternatives[-1]["retained"] = True
        elif case == "prepared_model":
            alternatives[0]["sourceModelHash"] = "substituted"
        alternative_scope = {**scope, "selectedRootAction": kw["selected_root_action"],
            "nonplayPolicy": {"team": target.other.name.lower(), **kw["nonplay_policy"]}}
        field = {"wrong_root": "rootHash", "wrong_policy": "nonplayPolicy", "wrong_rules": "rulesHash",
            "wrong_action": "selectedRootAction", "wrong_partition": "partition"}.get(case)
        if field is not None:
            alternative_scope[field] = "substituted"
        if "alternative_upper_bound" in kw or kw.get("optimize_alternative_upper"):
            assert kw["visibility"] == "complete_state" and (alternative_first or kw["alternative_upper_bound"] == 92)
            if not interval:
                assert "opponent_play_policy_id" not in kw
            if case == "interval_preparation_incomplete" and requests:
                raise InformationSetCompilationStopped("maximum_seconds", stage="native_bid_preparation",
                    evidence={"scope": alternative_scope, "queriesUsed": 2, "constructionComplete": False})
            if case == "empty":
                alternatives[0]["retained"] = False
            else:
                alternatives[0]["alternativeQuery"] = query(target=target.other, policy_team=target.other,
                    threshold=-92, auction_action="pass", escalation_action="decline")
            return (), {"kind": "native_bid_streamed_world_upper_v1" if alternative_first else "native_bid_streamed_alternative_preparation_v1",
                **({"upperBound": 92} if alternative_first else {}),
                "queriesUsed": 3, "proofQueriesUsed": 3, "constructionComplete": case != "partial_partition",
                "scope": alternative_scope, "branches": partitions[:-1] if case == "missing_partition" else
                    [partitions[0], *partitions[:-1]] if case == "duplicate_partition" else alternatives}
        return (() if case == "empty" else (SimpleNamespace(find_native_round_with_constant_policy_below=query,
            _model_hash=lambda **kw: "owned-test-model"),)), {
                "constructionComplete": case != "partial_partition",
                "scope": alternative_scope,
                "branches": partitions[:-1] if case == "missing_partition" else
                    [partitions[0], *partitions[:-1]] if case == "duplicate_partition" else alternatives}

    monkeypatch.setattr(compiler, "_prepare_native_bid_branches", prepare)
    qualification_cancelled = False
    limits = dict(selected_root_action="sun", opponent_strategy={"auction_action": "pass", "escalation_action": "decline"},
        maximum_iterations=10, maximum_queries=1 if case == "budget" else 100,
        maximum_observations=100, maximum_branches=13, maximum_model_bytes=1000000,
        maximum_conflicts=1000, maximum_seconds=10, cancel_check=lambda: case == "cancelled" or qualification_cancelled)
    if case == "qualification_budget":
        limits["maximum_queries"] = 1 + 3 * (len(exact.legal_actions) - 1)
    limits["opportunity_order"] = order
    if visibility != "complete_state":
        limits["visibility"] = visibility
    if interval:
        limits.update(incumbent_play_policies={"test": "policy"}, incumbent_maximum_seconds=2,
            incumbent_maximum_queries=10)
    if case == "native_ceiling":
        from baloot_platform.research.ortools_play_tail import NATIVE_OBSERVER_SAFE_PLAY_POLICY_IDS
        limits["opponent_play_policy_id"] = sorted(NATIVE_OBSERVER_SAFE_PLAY_POLICY_IDS)[0]
    if alternative_first:
        import baloot_platform.research.result_authority as authority
        # Only orchestration is substituted; native reader fixtures cover ownership.
        limits["maximum_depth"] = 64
        def validate_world_bound(receipt, **kw):
            assert kw["maximum_depth"] == limits["maximum_depth"]
            return receipt["upperBound"]
        monkeypatch.setattr(authority, "validate_native_bid_world_upper_v1", validate_world_bound)
        monkeypatch.setattr(authority, "validate_native_incumbent_threshold_floor_v1", lambda receipt, **kw: receipt["lowerBound"])
        limits["alternative_first"] = True
    if case in {"qualification_unknown", "qualification_cancelled"}:
        from ortools.sat.python import cp_model
        import baloot_platform.research.ortools_information_set_compiler as module
        original_solve = module._solve_cp_sat_with_cancel_v1

        def interrupted(*args):
            nonlocal qualification_cancelled
            if case == "qualification_unknown":
                return cp_model.UNKNOWN
            status = original_solve(*args)
            qualification_cancelled = True
            return status

        monkeypatch.setattr(module, "_solve_cp_sat_with_cancel_v1", interrupted)
    if case == "model_mismatch" or case == "prepared_model" and visibility == "complete_state":
        with pytest.raises(ValueError, match="alternative model ownership"):
            compiler._prove_native_bid_selected_action(exact, prefix, target, **limits)
        assert len(requests) == 1
        return
    result = compiler._prove_native_bid_selected_action(exact, prefix, target, **limits)
    assert result.get("escalationOpportunityOrder") == (None if order is None else list(order))
    assert result["visibility"] == visibility
    if visibility == "actor_information_set":
        assert result["observerRootHash"] == scope["rootHash"]
        assert result["observerRootHash"] != exact.state_hash
    assert result["rulesHash"] == rules.sha256 and result["sourcePrefixHash"] == prefix.sha256
    assert result["exactStateId"] == exact.exact_state_id and result["stateHash"] == exact.state_hash
    assert result["legalActions"] == sorted(row["code"] for row in exact.legal_actions)
    assert result["targetTeam"] == target.name.lower()
    assert result["completionScope"] == "selected_bid_current_deal"
    assert result["utility"] == "awarded_point_margin"
    assert result["limits"] == {"maximumIterations": limits["maximum_iterations"],
        "maximumQueries": limits["maximum_queries"], "maximumObservations": limits["maximum_observations"],
        "maximumBranchesPerPreparation": limits["maximum_branches"],
        "maximumModelBytesPerPreparation": limits["maximum_model_bytes"],
        "maximumConflictsPerSolve": limits["maximum_conflicts"],
        "maximumSeconds": limits["maximum_seconds"], "workersPerSolve": 1,
        **({"maximumDepth": limits["maximum_depth"]} if "maximum_depth" in limits else {}),
        **({"incumbentMaximumSeconds": 2, "incumbentMaximumQueries": 10} if interval else {})}
    assert result["status"] == ("selected_action_optimal" if success else "incomplete")
    assert result["actualActionOptimal"] is (True if success else None)
    assert result["optimalActionCount"] is None
    assert result["optimalActionCountLowerBound"] == (1 if success else 0)
    assert result["optimalActionCountUpperBound"] == len(exact.legal_actions)
    assert result["strategicCoverage"] is False
    assert result["sha256"] == canonical_sha256({k: v for k, v in result.items() if k != "sha256"})
    if case == "interval_preparation_incomplete":
        assert result["activePreparation"] == result["cause"]["evidence"]
        assert result["activePreparation"]["scope"]["selectedRootAction"] == result["activeAction"]
        assert result["queriesUsed"] == (0 if alternative_first else 1) + 3 * len(requests) + 2
    if success:
        assert len(requests) == len(exact.legal_actions) - 1
        assert all(row["upperBound"] == 92 for row in result["alternatives"])
        assert all(row["sourceModelHashes"] == [item["sourceModelHash"] for item in row["queries"]]
            for row in result["alternatives"])
        assert result["queriesUsed"] == 2 + 3 * len(requests)
        from ortools.sat.python import cp_model
        qualification = result["qualificationModel"]
        quality = result["actionQuality"]
        assert quality["scope"] == "compiled_" + visibility
        assert quality["actualActionOptimal"] is True
        assert quality["actualValue"] == quality["bestValue"] == (None if interval else 92)
        if interval:
            assert result["kind"] == ("native_bid_selected_action_dominance_v3" if alternative_first else "native_bid_selected_action_dominance_v2")
            assert quality["selectedValueBounds"] == {"lower": 92, "upper": 100}
            assert result["incumbentPlayPolicies"] == limits["incumbent_play_policies"]
        assert quality["regret"] == 0 and quality["optimalActionsComplete"] is False
        assert quality["optimalActions"] == ["sun"] and quality["optimalActionCount"] is None
        assert quality["optimalActionCountLowerBound"] == 1
        assert quality["optimalActionCountUpperBound"] == len(exact.legal_actions)
        assert quality["qualificationModelHash"] == qualification["sha256"]
        assert quality["sha256"] == canonical_sha256({**{k: v for k, v in quality.items() if k != "sha256"},
            "modelProto": qualification["modelProto"]})
        assert qualification["solverStatus"] == "optimal"
        assert qualification["selectedProofHash"] == canonical_sha256(selected)
        assert qualification["sha256"] == canonical_sha256({k: v for k, v in qualification.items() if k != "sha256"})
        assert [row["sourceProofHash"] for row in qualification["alternatives"]] == [canonical_sha256(row) for row in result["alternatives"]]
        model = cp_model.CpModel()
        assert model.proto.parse_text_format(qualification["modelProto"])
        model.add(model.get_bool_var_from_proto_index(qualification["claimLiteralIndex"]) == 0)
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        assert solver.solve(model) == cp_model.INFEASIBLE
    if case in {"budget", "cancelled", "empty"}:
        assert requests == []
    if not success:
        assert result["actionQuality"] is None
    if case.startswith("qualification_"):
        assert len(requests) == len(exact.legal_actions) - 1
        assert result["qualificationModel"] is None if case == "qualification_budget" else result["qualificationModel"] is not None
    for strategy in ({}, {"auction_action": True, "escalation_action": None},
                     {"auction_action": None, "escalation_action": None},
                     {"auction_action": "pass", "escalation_action": "decline", "hiddenHands": []}):
        with pytest.raises(ValueError):
            compiler._prove_native_bid_selected_action(exact, prefix, target, **{**limits, "opponent_strategy": strategy})
    with pytest.raises(ValueError, match="ownership"):
        compiler._prove_native_bid_selected_action(exact, prefix, target.other, **limits)
    with pytest.raises(ValueError, match="selected action"):
        compiler._prove_native_bid_selected_action(exact, prefix, target, **{**limits, "selected_root_action": "invalid"})
    if case == "complete":
        calls_before = len(requests)
        for field in ("maximum_iterations", "maximum_queries", "maximum_observations",
                "maximum_branches", "maximum_model_bytes", "maximum_conflicts"):
            for invalid in (0, -1, True, 1.5, None):
                with pytest.raises(ValueError, match="limits must be positive integers"):
                    compiler._prove_native_bid_selected_action(exact, prefix, target, **{**limits, field: invalid})
        assert len(requests) == calls_before


@pytest.mark.parametrize("case", ("complete", "interval", "selected_gap", "unknown_omission",
    "unproved_policy_omission", "wrong_root", "missing_information", "wrong_observed_action", "wrong_observed_domain"))
def test_observed_bid_dominance_requires_information_coverage(rules, monkeypatch, case):
    test_native_bid_dominance_preserves_unknown_ties_and_partial_bounds(
        rules, monkeypatch, case, None, visibility="actor_information_set")


@pytest.mark.parametrize("visibility", ("complete_state", "actor_information_set"))
@pytest.mark.parametrize("case", ("interval", "interval_allowance_exceeded", "interval_preparation_incomplete"))
def test_alternative_first_dominance_reserves_budget_and_waits_for_all_bounds(rules, monkeypatch, visibility, case):
    test_native_bid_dominance_preserves_unknown_ties_and_partial_bounds(
        rules, monkeypatch, case, None, visibility=visibility, alternative_first=True)


@pytest.mark.parametrize("visibility", ("complete_state", "actor_information_set"))
@pytest.mark.parametrize("prior_pass", (False, True))
def test_native_fixed_bid_prunes_before_full_round_construction(rules, visibility, prior_pass, monkeypatch):
    from ortools.sat.python import cp_model

    prefix = ExactTracePrefixV1.build({"schemaVersion": 1, "deckOrder": list(range(32)),
        "dealer": "east", "auctionActions": [{"actor": "north", "action": "pass"}] if prior_pass else [],
        "escalationActions": [], "projectDeclarations": [], "playActions": []})
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(prefix, decision_point={"kind": "bid"})
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    constructed = []
    original = compiler._build_native_bid_branch

    def build(*args, **kwargs):
        constructed.append(kwargs["branch"])
        return original(*args, **kwargs)

    monkeypatch.setattr(compiler, "_build_native_bid_branch", build)
    limits = dict(visibility=visibility, maximum_branches=13, maximum_model_bytes=100_000_000,
        maximum_conflicts=250_000, maximum_seconds=60, selected_root_action="sun")
    models, receipt = compiler._prepare_native_bid_branches(exact, prefix, **limits)
    assert len(models) == len(constructed) == 1
    assert constructed[0][0].value == "sun"
    assert receipt["scope"]["selectedRootAction"] == "sun"
    assert receipt["constructionComplete"] and not receipt["strategicCoverage"]
    assert len(receipt["branches"]) == 13
    assert receipt["queriesUsed"] == 13
    with pytest.raises(InformationSetCompilationStopped) as exhausted:
        compiler._prepare_native_bid_branches(exact, prefix, **limits, maximum_queries=1)
    assert exhausted.value.reason == "maximum_queries"
    assert sum(row["retained"] for row in receipt["branches"]) == 1
    assert all(row["reachability"]["solverStatus"] == "infeasible"
        for row in receipt["branches"] if not row["retained"])
    model = models[0].model.clone()
    model.add(models[0].auction_fragment.choices[len(prefix.auction_actions)]["sun"] == 0)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 10
    assert solver.solve(model) == cp_model.INFEASIBLE
    assert receipt["sha256"] == canonical_sha256({k: v for k, v in receipt.items() if k != "sha256"})
    for action in (True, "invalid"):
        with pytest.raises(ValueError, match="selected root action"):
            compiler._prepare_native_bid_branches(exact, prefix, **{**limits, "selected_root_action": action})


@pytest.mark.parametrize("failure", (None, "proof", "counterexample", "no_outcome", "queries", "tight_queries", "memory"))
@pytest.mark.parametrize("optimize", (False, True))
def test_native_streamed_bid_preparation_releases_models_and_owns_receipts(rules, monkeypatch, failure, optimize):
    # Coordinator check: real auction reachability, small substitute round models.
    import weakref
    from types import SimpleNamespace
    from ortools.sat.python import cp_model
    prefix = ExactTracePrefixV1.build({"schemaVersion": 1, "deckOrder": list(range(32)),
        "dealer": "east", "auctionActions": [], "escalationActions": [],
        "projectDeclarations": [], "playActions": []})
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(prefix, decision_point={"kind": "bid"})
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    references, calls = [], []

    class Branch:
        def __init__(self):
            self.model = cp_model.CpModel()
            self.auction_fragment = SimpleNamespace(choices=[{"pass": self.model.new_bool_var("pass")}])
            self.identity = str(len(references))
            self.points = (self.model.new_int_var(0, 100, "points0"), self.model.new_int_var(0, 100, "points1"))

        def _native_round_query_context(self):
            return {}, self.points

        def _model_hash(self, **kwargs):
            return self.identity

        def constrain_native_auction_root_choice(self, action, *, actor):
            assert action == "pass" and actor == Seat.NORTH
            self.model.add(self.auction_fragment.choices[0][action] == 1)

        def find_native_round_with_constant_policy_below(self, **kwargs):
            calls.append(kwargs)
            assert kwargs["record_model_bytes"] is True and kwargs["threshold"] == (101 if optimize else -92)
            assert kwargs["optimize_margin"] is optimize
            source_bytes = len(str(self.model.proto).encode("utf-8"))
            core = {"sourceModelHash": self.identity, "status": "queried",
                "bindings": [{"status": "bound"}],
                "query": None if failure == "no_outcome" else {"status": "incomplete" if failure == "proof"
                    else "witness_found" if optimize or failure == "counterexample" else "no_matching_outcome",
                    "solverStatus": "unknown" if failure == "proof" else "feasible" if failure == "counterexample"
                        else "optimal" if optimize else "infeasible",
                    **({"candidate": {"utility": -93}} if failure == "counterexample"
                        else {"candidate": {"utility": -len(references)}} if optimize else {})},
                "modelBytes": {"retainedUpperBound": 10**9 if failure == "memory" else 3 * source_bytes}}
            return {**core, "sha256": canonical_sha256(core)}

    def build(*args, **kwargs):
        assert all(reference() is None for reference in references), "previous branch still retained"
        assert kwargs["compact_known_initial_ownership"] is True
        branch = Branch()
        references.append(weakref.ref(branch))
        return branch

    monkeypatch.setattr(compiler, "_build_native_bid_branch", build)
    limits = dict(visibility="complete_state", maximum_branches=13, maximum_model_bytes=100_000_000,
        maximum_conflicts=250_000, maximum_seconds=30,
        maximum_queries=1 if failure == "queries" else 4 if failure == "tight_queries" else 100,
        selected_root_action="pass", policy_team=Seat.parse(exact.next_actor).team.other,
        nonplay_policy={"auction_action": "pass", "escalation_action": "decline" if failure == "tight_queries" else None},
        **({"optimize_alternative_upper": True} if optimize else {"alternative_upper_bound": 92}))
    if failure is None:
        models, receipt = compiler._prepare_native_bid_branches(exact, prefix, **limits)
        assert models == () and len(references) > 1
        assert all(reference() is None for reference in references)
    else:
        with pytest.raises(InformationSetCompilationStopped) as stopped:
            compiler._prepare_native_bid_branches(exact, prefix, **limits)
        assert stopped.value.reason == {"proof": "alternative_upper_bound_not_certified",
            "counterexample": "alternative_upper_bound_not_certified" if optimize else "alternative_upper_bound_refuted",
            "no_outcome": "alternative_upper_bound_not_certified",
            "queries": "maximum_queries", "tight_queries": "maximum_queries", "memory": "maximum_model_bytes"}[failure]
        receipt = stopped.value.partial_receipt["evidence"]
    assert receipt["constructionComplete"] is (failure is None)
    assert receipt["kind"] == ("native_bid_streamed_world_upper_v1" if optimize else "native_bid_streamed_alternative_preparation_v1")
    if optimize:
        assert receipt["upperBound"] == (len(references) if failure is None else None)
        assert receipt["worldStateHash"] == exact.state_hash
        assert receipt["worldPrefixHash"] == prefix.sha256
        assert receipt["worldExactStateId"] == exact.exact_state_id
        assert receipt["worldEffectivePrefixHash"] == exact.prefix_hash
        assert receipt["targetTeam"] == Seat.parse(exact.next_actor).team.name.lower()
        assert "alternativeUpperBound" not in receipt
        assert receipt["targetDecisionScope"] == "complete_information_cooperative_relaxation"
        with pytest.raises(ValueError, match="opponent policy team"):
            compiler._prepare_native_bid_branches(exact, prefix,
                **{**limits, "policy_team": Seat.parse(exact.next_actor).team})
    retained = [row for row in receipt["branches"] if "alternativeQuery" in row]
    assert len(retained) == len(calls)
    assert receipt["proofQueriesUsed"] == (1 if failure == "no_outcome" else 2) * len(calls)
    assert receipt["queriesUsed"] >= receipt["proofQueriesUsed"]
    if failure == "tight_queries":
        assert len(calls) == 1 and receipt["queriesUsed"] == 4
    for row in retained:
        assert row["alternativeQuery"]["sourceModelHash"] == row["sourceModelHash"]
    assert receipt["sha256"] == canonical_sha256({k: v for k, v in receipt.items() if k != "sha256"})
    for invalid in (True, 1.5, "92"):
        with pytest.raises(ValueError, match="bound"):
            compiler._prepare_native_bid_branches(exact, prefix, **{**limits, "alternative_upper_bound": invalid})
    with pytest.raises(ValueError, match="query cap"):
        compiler._prepare_native_bid_branches(exact, prefix, **{**limits, "maximum_queries": None})

    if optimize and failure == "tight_queries":
        # An interrupted preparation can reuse its completed cells, not the
        # unfinished cell. Caller mutation cannot change the owned checkpoint.
        completed = copy.deepcopy(receipt["branches"][0])
        owned_checkpoint = copy.deepcopy(compiler._native_bid_partial_upper)
        receipt["branches"][0]["alternativeUpperBound"] = 999
        failure = None
        prior_calls = len(calls)
        _, resumed = compiler._prepare_native_bid_branches(exact, prefix,
            **{**limits, "maximum_queries": 100})
        assert resumed["constructionComplete"] is True
        assert resumed["branches"][0] == completed
        assert len(calls) - prior_calls == sum(row["retained"] for row in resumed["branches"]) - 1
        assert resumed["proofQueriesUsed"] == 2 * sum(row["retained"] for row in resumed["branches"])
        assert resumed["queriesUsed"] >= receipt["queriesUsed"]
        assert compiler._native_bid_partial_upper is None
        # A changed policy must rebuild, even at the same cards and bid root.
        compiler._native_bid_partial_upper = owned_checkpoint
        prior_calls = len(calls)
        _, changed = compiler._prepare_native_bid_branches(exact, prefix,
            **{**limits, "maximum_queries": 100,
               "nonplay_policy": {"auction_action": "pass", "escalation_action": None}})
        assert len(calls) - prior_calls == sum(row["retained"] for row in changed["branches"])


@pytest.mark.parametrize("depth_status", ("depth_bound_proven", "over_depth_witness", "incomplete"))
@pytest.mark.parametrize("order", (None, ("west", "south", "east", "north")))
def test_native_bid_preparation_retains_and_charges_depth_query(rules, monkeypatch, depth_status, order):
    from baloot_platform.research.ortools_scenario_model import OrToolsScenarioModelV1
    prefix = ExactTracePrefixV1.build({"schemaVersion": 1, "deckOrder": list(range(32)),
        "dealer": "east", "auctionActions": [], "escalationActions": [],
        "projectDeclarations": [], "playActions": []})
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(prefix, decision_point={"kind": "bid"})
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    calls = []
    def query(self, **kwargs):
        calls.append(kwargs)
        assert kwargs["maximum_depth"] == 40
        assert self._native_round_fragment.scope["escalationOpportunityOrder"] == (
            "symbolic_per_window" if order is None else list(order))
        return {"status": depth_status}
    monkeypatch.setattr(OrToolsScenarioModelV1, "find_native_continuation_depth_above", query)
    limits = dict(visibility="complete_state", maximum_branches=13, maximum_model_bytes=100_000_000,
        maximum_conflicts=250_000, maximum_seconds=30, selected_root_action="sun", maximum_depth=40,
        maximum_queries=20, opportunity_order=order)
    if depth_status == "depth_bound_proven":
        _, receipt = compiler._prepare_native_bid_branches(exact, prefix, **limits)
        assert receipt["constructionComplete"] is True and receipt["queriesUsed"] == 14
    else:
        with pytest.raises(InformationSetCompilationStopped) as failure:
            compiler._prepare_native_bid_branches(exact, prefix, **limits)
        assert failure.value.reason == ("maximum_depth" if depth_status == "over_depth_witness" else "depth_bound_incomplete")
        receipt = failure.value.partial_receipt["evidence"]
        assert receipt["constructionComplete"] is False and receipt["queriesUsed"] >= 3
    assert len(calls) == 1
    assert receipt["limits"]["maximumDepth"] == 40
    assert receipt["scope"].get("escalationOpportunityOrder") == (None if order is None else list(order))
    assert [row["depthQuery"]["status"] for row in receipt["branches"] if "depthQuery" in row] == [depth_status]


@pytest.mark.real_clock
@pytest.mark.parametrize("visibility", ("complete_state", "actor_information_set"))
def test_native_bid_preparation_covers_partition_and_preserves_budget_receipts(root, rules, visibility, record_property, monkeypatch):
    from baloot_platform.config import load_profile_payload
    import baloot_platform.research.ortools_information_set_compiler as module

    prefix = ExactTracePrefixV1.build({"schemaVersion": 1, "deckOrder": list(range(32)),
        "dealer": "east", "auctionActions": [], "escalationActions": [],
        "projectDeclarations": [], "playActions": []})
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "bid"})
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    # Full symbolic deal models use the existing offline tier, not an invented
    # interactive allowance. This test does not assert a latency/yield gate.
    profile = load_profile_payload("challenge", root / "configs/challenge/intelligent_research.yaml", root=root)
    tier = profile["scenarioResearchV4"]["proofBudgets"]["profiles"]["offline_heavyweight_v1"]
    arguments = dict(visibility=visibility, maximum_branches=len(compiler._native_bid_branch_specs(Seat.EAST)),
        maximum_model_bytes=tier["maximumCacheBytes"], maximum_conflicts=tier["maximumNodes"],
        maximum_seconds=tier["maximumWallSeconds"])
    models, receipt = compiler._prepare_native_bid_branches(exact, prefix, **arguments)
    assert receipt["constructionComplete"] and not receipt["strategicCoverage"]
    assert len(receipt["branches"]) == len(receipt["scope"]["partition"]) == 13
    assert len(models) == sum(row["retained"] for row in receipt["branches"]) < 13
    assert all(row["reachability"]["solverStatus"] == "infeasible"
               for row in receipt["branches"] if not row["retained"])
    assert all(not row["retained"] for row in receipt["branches"]
               if (spec := receipt["scope"]["partition"][row["partitionIndex"]]) is not None
               and spec["contract"] in {"sun", "ashkal"} and spec["locked"])
    assert [row["partitionIndex"] for row in receipt["branches"]] == list(range(13))
    assert receipt["modelBytes"] == sum(row["modelBytes"] for row in receipt["branches"])
    assert all(model.model.validate() == "" for model in models)
    for model, branch in zip(models, (row for row in receipt["branches"] if row["retained"]), strict=True):
        if visibility == "complete_state":
            assert "informationBindings" not in branch
            continue
        information = branch["informationBindings"]
        compiler._validate_native_bid_information_bindings(model, branch)
        for mutation in ("recipient", "order", "missing_slot", "reordered_slots"):
            changed = copy.deepcopy(branch)
            altered = changed["informationBindings"]
            if mutation == "recipient":
                altered["purchaseRecipient"]["index"] = True
            elif mutation == "order":
                altered["opportunityOrder"] = [seat.name.lower() for seat in Seat]
            elif len(altered["escalationOpportunities"]) < 2:
                continue
            elif mutation == "missing_slot":
                altered["escalationOpportunities"].pop()
            else:
                altered["escalationOpportunities"][:2] = reversed(altered["escalationOpportunities"][:2])
            altered["sha256"] = canonical_sha256({k: v for k, v in altered.items() if k != "sha256"})
            with pytest.raises(ValueError, match="native bid information"):
                compiler._validate_native_bid_information_bindings(model, changed)
        assert information["sha256"] == canonical_sha256({k: v for k, v in information.items() if k != "sha256"})
        assert information["cpModelProtoHash"] == branch["cpModelProtoHash"]
        assert information["opportunityOrder"] is None and information["strategicCoverage"] is False
        variables = [information["purchaseRecipient"]]
        fragment = model._native_round_fragment
        slots = () if fragment is None else fragment.escalation_opportunities
        assert len(information["escalationOpportunities"]) == len(slots)
        for row, slot in zip(information["escalationOpportunities"], slots, strict=True):
            assert (row["window"], row["ordinal"]) == (slot["window"], slot["ordinal"])
            for key in ("actor", "active", "announced", "declined"):
                assert row[key]["index"] == slot[key].index
                variables.append(row[key])
        assert information["purchaseRecipient"]["index"] == model.auction_fragment.states[-1]["recipient"].index
        for variable in variables:
            native = model.model.proto.variables[variable["index"]]
            assert variable["name"] == native.name
            assert variable["domain"] == list(native.domain)
    assert receipt["scope"]["visibility"] == visibility and not receipt["scope"]["nextDealAnalyzed"]
    assert receipt["sha256"] == canonical_sha256({key: value for key, value in receipt.items() if key != "sha256"})
    record_property("model_bytes", receipt["modelBytes"])
    record_property("branch_model_bytes", [row["modelBytes"] for row in receipt["branches"]])
    del models
    for field, value, reason, completed in (
            ("maximum_branches", 1, "maximum_branches", 0),
            ("maximum_model_bytes", 1, "maximum_model_bytes", 1),
            ("maximum_seconds", 1, "maximum_seconds", 0),
            ("cancel_check", lambda: True, "cancelled", 0)):
        with monkeypatch.context() as patch, pytest.raises(InformationSetCompilationStopped) as caught:
            if field == "maximum_seconds":
                set_monotonic(patch, module, 0, 2)
            compiler._prepare_native_bid_branches(exact, prefix, **{**arguments, field: value})
        assert caught.value.reason == reason
        partial = caught.value.partial_receipt["evidence"]
        assert not partial["constructionComplete"] and not partial["strategicCoverage"]
        assert len(partial["branches"]) == completed
    for field in ("maximum_branches", "maximum_model_bytes", "maximum_conflicts", "maximum_seconds"):
        with pytest.raises(ValueError, match="budgets"):
            compiler._prepare_native_bid_branches(exact, prefix, **{**arguments, field: True})
    def exhausted(solver, model, cancel_check):
        solver.parameters.max_time_in_seconds = 0
        return solver.solve(model)
    with monkeypatch.context() as patch:
        patch.setattr(module, "_solve_cp_sat_with_cancel_v1", exhausted)
        with pytest.raises(InformationSetCompilationStopped) as caught:
            compiler._prepare_native_bid_branches(exact, prefix, **arguments)
    assert caught.value.reason == "reachability_incomplete"
    partial = caught.value.partial_receipt["evidence"]
    assert not partial["constructionComplete"] and not partial["strategicCoverage"]
    assert partial["branches"][-1]["reachability"]["solverStatus"] == "unknown"
    assert partial["branches"][0]["retained"]


@pytest.mark.parametrize("visibility", ("complete_state", "actor_information_set"))
def test_native_policy_prunes_locked_branches_before_play_models(root, rules, visibility, monkeypatch):
    from ortools.sat.python import cp_model
    from baloot_platform.config import load_profile_payload
    import baloot_platform.research.ortools_scenario_model as model_module
    prefix = ExactTracePrefixV1.build({"schemaVersion": 1, "deckOrder": list(range(32)),
        "dealer": "east", "auctionActions": [], "escalationActions": [], "projectDeclarations": [], "playActions": []})
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(prefix, decision_point={"kind": "bid"})
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    constructed = []
    original = compiler._build_native_bid_branch

    def build(*args, **kwargs):
        constructed.append(kwargs["branch"])
        return original(*args, **kwargs)

    monkeypatch.setattr(compiler, "_build_native_bid_branch", build)
    profile = load_profile_payload("challenge", root / "configs/challenge/intelligent_research.yaml", root=root)
    tier = profile["scenarioResearchV4"]["proofBudgets"]["profiles"]["offline_heavyweight_v1"]
    args = dict(visibility=visibility, selected_root_action="pass", policy_team=Seat.parse(exact.next_actor).team.other,
        nonplay_policy={"auction_action": "pass", "escalation_action": "decline"},
        maximum_branches=13, maximum_model_bytes=tier["maximumCacheBytes"], maximum_conflicts=250_000, maximum_seconds=60)
    models, receipt = compiler._prepare_native_bid_branches(exact, prefix, **args)
    assert models and receipt["constructionComplete"] and not receipt["strategicCoverage"]
    assert len(receipt["branches"]) == 13
    assert not any(spec is not None and spec[3] for spec in constructed)
    for row in receipt["branches"]:
        reachability = row["reachability"]
        assert all(binding["status"] == "bound" for binding in reachability["policyBindings"])
        if not row["retained"]:
            assert reachability["solverStatus"] == "infeasible"
    assert receipt["scope"]["nonplayPolicy"]["team"] == args["policy_team"].name.lower()
    constructed.clear()
    with monkeypatch.context() as patch:
        patch.setattr(model_module, "_solve_cp_sat_with_cancel_v1", lambda *a: cp_model.UNKNOWN)
        with pytest.raises(InformationSetCompilationStopped) as caught:
            compiler._prepare_native_bid_branches(exact, prefix, **args)
    assert caught.value.reason == "policy_legality_not_proven"
    partial = caught.value.partial_receipt["evidence"]
    assert not partial["constructionComplete"] and constructed == []
    assert partial["branches"][0]["policyBindings"][0]["status"] == "incomplete"


@pytest.mark.real_clock
@pytest.mark.parametrize("typed_stop", (False, True))
def test_world_normalization_retains_only_typed_private_stop_receipt(
    monkeypatch: pytest.MonkeyPatch, root: Path, rules, typed_stop: bool,
) -> None:
    raw = json.loads((root / "examples" / "research-scenario-v4-ace-sacrifice-two-aces.json")
                     .read_text(encoding="utf-8"))["exactTracePrefix"]
    raw.update(auctionActions=[], escalationActions=[], projectDeclarations=[],
               projectAssertions=[], playActions=[], stateAssertions={})
    prefix = ExactTracePrefixV1.build(raw)
    decision = {"kind": "bid", "selector": {"actionIndex": 0}}
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point=decision)
    failure = (ExactNormalizationStopped("unknown", evidence={"sourceStage": "auction"})
               if typed_stop else RuntimeError("private error must not be persisted"))
    if not typed_stop:
        failure.partial_receipt = {"private": "not a trusted normalization receipt"}

    def stop(*args, **kwargs):
        assert 0 < kwargs["maximum_seconds"] <= 10
        raise failure

    monkeypatch.setattr(OrToolsExactStateNormalizerV1, "normalize", stop)
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    with pytest.raises(InformationSetCompilationStopped) as caught:
        compiler.enumerate_graph_world_inputs(
            exact, prefix, decision, maximum_worlds=2, maximum_seconds=10,
            bounded_world_scope=True)
    receipt = caught.value.partial_receipt
    assert receipt["status"] == "incomplete"
    assert receipt["exhaustionReason"] == "hidden_world_normalization_incomplete"
    assert receipt["stage"] == "hidden_world_nonplay_normalization"
    assert receipt["evidence"] == {
        "normalizedWorlds": 0,
        **({"causeReceipt": failure.partial_receipt} if typed_stop else {}),
    }
    assert receipt["sha256"] == canonical_sha256({
        key: value for key, value in receipt.items() if key != "sha256"})
    from baloot_platform.research.service import _operator_incomplete_compilation_v1
    public = _operator_incomplete_compilation_v1(receipt)
    assert "evidence" not in public and "causeReceipt" not in public
    assert "private error" not in json.dumps(receipt)


@pytest.mark.real_clock
def test_nonplay_bid_refusal_uses_measured_hidden_deck_space(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    rules,
) -> None:
    raw = json.loads(
        (
            root / "examples" / "research-scenario-v4-ace-sacrifice-two-aces.json"
        ).read_text(encoding="utf-8")
    )["exactTracePrefix"]
    raw.update(
        {
            "auctionActions": [],
            "escalationActions": [],
            "projectDeclarations": [],
            "projectAssertions": [],
            "playActions": [],
            "stateAssertions": {},
        }
    )
    prefix = ExactTracePrefixV1.build(raw)
    decision = {"kind": "bid", "selector": {"actionIndex": 0}}
    exact = OrToolsExactStateNormalizerV1(
        rules,
        native_cp_sat=True,
    ).normalize(prefix, decision_point=decision)
    compiler = OrToolsActorInformationSetCompilerV1(rules)

    measurement = compiler.measure_nonplay_hidden_worlds(exact, prefix, "bid")
    assert measurement["hiddenCards"] == 26
    assert measurement["groupSizes"] == [5, 5, 5, *([1] * 11)]
    assert measurement["hiddenWorldUpperBound"] == (
        hidden_allocation_upper_bound_v1((5, 5, 5, *([1] * 11)))
    )
    with pytest.raises(InformationSetCompilationStopped) as stopped:
        compiler.enumerate_graph_world_inputs(
            exact,
            prefix,
            decision,
            maximum_worlds=1_000_000,
            maximum_seconds=2,
        )
    assert stopped.value.reason == "maximum_hidden_worlds"
    assert stopped.value.partial_receipt["stage"] == ("hidden_world_nonplay_preflight")
    assert stopped.value.partial_receipt["evidence"]["decisionKind"] == "bid"
    native = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    native_measurement = native.measure_nonplay_hidden_worlds(exact, prefix, "bid")
    assert native_measurement["hiddenWorldUpperBound"] * 16 == measurement["hiddenWorldUpperBound"]
    assert native_measurement["method"] == "initial_hands_and_purchase_invariant_undealt_groups_v1"
    with pytest.raises(InformationSetCompilationStopped) as native_stopped:
        native.enumerate_graph_world_inputs(
            exact, prefix, decision, maximum_worlds=1_000_000, maximum_seconds=2,
        )
    assert native_stopped.value.partial_receipt["evidence"]["hiddenWorldUpperBound"] == (
        native_measurement["hiddenWorldUpperBound"]
    )
    import baloot_platform.research.ortools_information_set_compiler as compiler_module
    import baloot_platform.research.ortools_scenario_model as scenario_module
    original_solve = compiler_module._solve_cp_sat_with_cancel_v1
    shared_queries = []
    def inspect_shared_query(solver, model, cancel):
        names = {variable.name for variable in model.proto.variables}
        if "card_at_position_0" in names:
            assert any(name.startswith("hidden_0_") for name in names)
            assert "purchase_recipient" in names
            shared_queries.append(True)
        return original_solve(solver, model, cancel)
    def forbidden(*args, **kwargs):
        raise AssertionError("native bid worlds called the legacy auction enumerator")
    monkeypatch.setattr(compiler_module, "_solve_cp_sat_with_cancel_v1", inspect_shared_query)
    monkeypatch.setattr(scenario_module, "_enumerate_auction", forbidden)
    limited = native.enumerate_graph_world_inputs(
        exact, prefix, decision, maximum_worlds=2, maximum_seconds=10,
        bounded_world_scope=True,
    )
    repeated = native.enumerate_graph_world_inputs(
        exact, prefix, decision, maximum_worlds=2, maximum_seconds=10,
        bounded_world_scope=True,
    )
    assert len(limited) == 2
    assert not shared_queries  # Bounded sampling retains its smaller allocation model.
    layout = DealLayoutV1.build(rules.structural_domain, dealer=Seat.parse(prefix.dealer))
    actor = Seat.parse(exact.next_actor)
    known = {native._card(card) for card in exact.state["hands"][actor.name.lower()]}
    known.add(native._card(exact.state["purchaseCard"]))
    with pytest.raises(InformationSetCompilationStopped) as exhaustive_stopped:
        native._enumerate_hidden_groups(
            tuple(card for card in FULL_DECK if card not in known),
            tuple(map(len, native._bid_hidden_group_positions(layout, actor))),
            deadline=time.monotonic() + 10, maximum_worlds=2, cancel_check=None,
            exact=exact, bid_prefix=prefix,
        )
    assert exhaustive_stopped.value.reason == "maximum_hidden_worlds"
    assert shared_queries
    assert [item[0].sha256 for item in limited] == [item[0].sha256 for item in repeated]
    assert all(
        canonical_observer_information_hash_v1(
            complete_state=item.state, acting_seat=exact.next_actor,
        ) == canonical_observer_information_hash_v1(
            complete_state=exact.state, acting_seat=exact.next_actor,
        )
        for item, _ in limited
    )
    monkeypatch.setattr(native, "_prefix_with_deck", lambda source, *_args, **_kwargs: source)
    deduplicated = native.enumerate_graph_world_inputs(
        exact, prefix, decision, maximum_worlds=2, maximum_seconds=10,
        bounded_world_scope=True,
    )
    assert len(deduplicated) == 1


@pytest.mark.parametrize("bounded", (False, True))
@pytest.mark.parametrize("multiplier", (1, 2))
def test_compile_nonplay_uses_the_shared_world_callback(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    rules,
    bounded,
    multiplier,
) -> None:
    raw = json.loads(
        (
            root / "examples" / "research-scenario-v4-ace-sacrifice-two-aces.json"
        ).read_text(encoding="utf-8")
    )["exactTracePrefix"]
    raw.update(
        {
            "auctionActions": [],
            "escalationActions": [],
            "projectDeclarations": [],
            "projectAssertions": [],
            "playActions": [],
            "stateAssertions": {},
        }
    )
    prefix = ExactTracePrefixV1.build(raw)
    decision = {"kind": "bid", "selector": {"actionIndex": 0}}
    exact = OrToolsExactStateNormalizerV1(
        rules,
        native_cp_sat=True,
    ).normalize(prefix, decision_point=decision)
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=bounded)
    allocation_limits = []
    def enumerate_inputs(*args, **kwargs):
        allocation_limits.append(kwargs["maximum_worlds"])
        return ((exact, prefix), (exact, prefix))
    monkeypatch.setattr(
        compiler,
        "enumerate_graph_world_inputs",
        enumerate_inputs,
    )
    calls = 0

    def graph_callback(_exact, _prefix, _request):
        nonlocal calls
        calls += 1
        variants = tuple(_nonplay_graph_world(
            "bid", f"callback-world-{calls}-{index}", reverse_hidden_payoff=calls == 2,
            bounded=bounded) for index in range(multiplier))
        return variants[0] if multiplier == 1 else variants

    request = (_bounded_actor_request if bounded else _actor_request)(
        threshold=1, claim="winning_root_action_count",
    )
    compilation = compiler.compile_nonplay(
        exact,
        prefix,
        decision,
        request,
        graph_compiler=graph_callback,
        maximum_nodes=100,
        maximum_depth=8,
        maximum_worlds=2 * multiplier,
        maximum_seconds=2,
        graph_world_multiplier=multiplier,
    )
    outcome = prove_actor_information_set_v1(
        compilation,
        request,
        maximum_nodes=100,
        maximum_worlds=2 * multiplier,
        maximum_seconds=2,
        native_action_quality=bounded,
    )

    assert calls == 2
    assert allocation_limits == [2]
    assert len(compilation.graph.problem.worlds) == 2 * multiplier
    if multiplier > 1:
        assert compilation.graph.model_document["worldExpansion"]["combinedWorldCount"] == 4
    else:
        assert "worldExpansion" not in compilation.graph.model_document
    assert outcome.completion_status == ("bounded_complete" if bounded else "exact_complete"), outcome.to_dict()
    assert compilation.graph.decision_kind == "bid"
    assert outcome.winning_root_action_count == 1


def test_native_interleaved_play_scope_counts_allocations_not_history_expansions(rules, monkeypatch):
    from types import SimpleNamespace
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    first = _observer_state(rules, 1)
    second = copy.deepcopy(first)
    _swap_hidden_cards(second)
    exact = SimpleNamespace(state=first, next_actor="south", state_hash=canonical_sha256(first),
        exact_state_id="STATE-INTERLEAVED", sha256="1" * 64)
    other = SimpleNamespace(state=second, sha256="2" * 64)
    prefix = SimpleNamespace(sha256="3" * 64)
    calls = []
    def inputs(*_args, **_kwargs):
        calls.append(True)
        return ((exact, prefix), (other, prefix))
    monkeypatch.setattr(compiler, "enumerate_graph_world_inputs", inputs)
    def graphs(normalized, *_args):
        return tuple(_nonplay_graph_world("play", normalized.sha256 + str(index),
            root_actor="south", reverse_hidden_payoff=False) for index in range(2))
    compilation = compiler.compile_nonplay(exact, prefix, {"kind": "play"}, _actor_request(),
        graph_compiler=graphs, maximum_nodes=100, maximum_depth=8, maximum_worlds=4,
        maximum_seconds=2, graph_world_multiplier=2)
    assert calls == [True]
    scope = compilation.graph.observer_scope
    assert scope.hidden_world_count == 2
    binding = compilation.graph.model_document["observerScopeBinding"]
    assert binding["graphWorldCount"] == compilation.graph.hidden_worlds["count"] == 4
    assert binding["sourceStateHash"] == exact.state_hash
    assert binding["scope"] == scope.to_safe_dict()


@pytest.mark.parametrize("case", ["too_many_allocations", "short_callback", "long_callback", "preflight", "coercion"])
def test_nonplay_graph_expansion_fails_closed(rules, monkeypatch, case):
    from types import SimpleNamespace
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    exact = SimpleNamespace(sha256="exact")
    prefix = SimpleNamespace(sha256="prefix")
    calls = []
    def inputs(*args, **kwargs):
        calls.append("allocations")
        return ((exact, prefix),) * (3 if case == "too_many_allocations" else 1)
    def callback(*args):
        calls.append("callback")
        return tuple(_nonplay_graph_world("bid", str(index), reverse_hidden_payoff=False)
                     for index in range(3 if case == "long_callback" else 1))
    monkeypatch.setattr(compiler, "enumerate_graph_world_inputs", inputs)
    expected = InformationSetCompilationStopped if case in {"preflight", "too_many_allocations"} else ValueError
    with pytest.raises(expected) as exc:
        compiler.compile_nonplay(exact, prefix, {"kind": "bid"}, _actor_request(),
            graph_compiler=callback, maximum_nodes=100, maximum_depth=8,
            maximum_worlds=1 if case == "preflight" else 4, maximum_seconds=2,
            graph_world_multiplier=True if case == "coercion" else 2)
    if expected is InformationSetCompilationStopped:
        assert exc.value.reason == "maximum_combined_worlds"
        assert "callback" not in calls
    if case in {"preflight", "coercion"}:
        assert not calls


@pytest.mark.parametrize("winning_count", [0, 1, 3])
def test_actor_root_count_claim_reports_exact_zero_one_and_many(
    winning_count: int,
    rules,
) -> None:
    values = (1,) * winning_count + (0,) * (3 - winning_count)
    roots = {
        code: FiniteInformationNodeV1.terminal(value)
        for code, value in zip(("H7", "S7", "D7"), values, strict=True)
    }
    problem = FiniteInformationSetProblemV1.build(
        problem_id=f"root-count-{winning_count}",
        success_threshold=1,
        worlds=(FiniteInformationWorldV1.build("one", roots),),
    )
    graph = ActorInformationSetGraphV1(
        problem,
        {"count": 1, "complete": True},
        {"kind": "test_allocation_model"},
    )

    outcome = prove_actor_information_set_v1(
        ActorInformationSetCompilationV1(graph, node_count=3),
        _actor_request(threshold=1.0, claim="winning_root_action_count"),
        maximum_nodes=100,
        maximum_worlds=2,
        maximum_seconds=2.0,
    )

    assert outcome.completion_status == "exact_complete"
    assert outcome.claim_satisfied is True
    assert outcome.winning_root_action_count == winning_count
    model_document = outcome.model_document()
    assert model_document["qualificationSource"] == (
        "cp_sat_cegis_root_certificates_v2"
    )
    assert len(model_document["rootActionVariables"]) == 3
    assert model_document["winningRootActionCountVariable"] == (
        "winning_root_action_count"
    )
    assert model_document["claimEvidenceBinding"] == {
        "kind": "winning_root_action_count",
        "variable": "proof_claim_satisfied",
        "literalIndex": model_document["claimEvidenceBinding"]["literalIndex"],
        "winningCountVariable": "winning_root_action_count",
    }
    from ortools.sat.python import cp_model

    bindings = cp_model.CpModel()
    assert bindings.proto.parse_text_format(model_document["modelProto"])
    assert {variable.name for variable in bindings.proto.variables} >= {
        "root_value_0",
        "root_value_1",
        "root_value_2",
        "root_wins_0",
        "root_wins_1",
        "root_wins_2",
        "winning_root_action_count",
        "proof_claim_satisfied",
    }
    claim_index = model_document["claimEvidenceBinding"]["literalIndex"]
    assert bindings.proto.variables[claim_index].name == "proof_claim_satisfied"
    assert sum(row["satisfies"] is True for row in model_document["roots"]) == (
        winning_count
    )
    assert all(row["robustValue"] in {0, 1} for row in model_document["roots"])

    from baloot_platform.research.ortools_research_executor import (
        OrToolsNeutralCandidateV1,
        OrToolsResearchCompilerV1,
        OrToolsResearchStageReceiptV1,
        _proof_stage_bindings_v1,
    )
    from baloot_platform.research.scenario_contracts import ResearchScenarioV1

    candidate = OrToolsNeutralCandidateV1(
        "search", {"play": {"decisionState": {"prefixLength": 24}}},
        {"currentPlayer": "east"}, (),
        {"proof": OrToolsResearchStageReceiptV1(
            "proof", outcome.model_hash, outcome.sha256, outcome.sha256, "optimal"
        )},
    )
    compiler = OrToolsResearchCompilerV1(native_cp_sat_authority=True)
    for expected, status in ((winning_count, "satisfied"), ((winning_count + 1) % 4, "target_miss")):
        leaf = {
            "predicateId": "PRED-ACTOR-ROOT-COUNT", "kind": "proof.winning_root_action_count",
            "version": 1, "hardness": "hard", "arguments": {"minimum": expected, "maximum": expected},
        }
        scenario = ResearchScenarioV1.build({
            "schemaVersion": 1, "origin": {"kind": "search"},
            "constraints": {**leaf, "predicateId": "PRED-ACTOR-COUNT-REPORT", "hardness": "report"},
            "decisionPoint": {"kind": "play"}, "outcome": leaf,
        })
        contexts = _proof_stage_bindings_v1(candidate, outcome, rules, require_native=True)
        original_actions = contexts["proof"].context.root_actions
        contexts["proof"].context.root_actions = (
            replace(original_actions[0], winning=contexts["proof"].context.model.new_bool_var("wrong_win")),
            *original_actions[1:],
        )
        assert compiler.evaluate(scenario, candidate, contexts, maximum_seconds=2).status == "incomplete"
        contexts["proof"].context.root_actions = original_actions
        evaluation = compiler.evaluate(scenario, candidate, contexts, maximum_seconds=2)
        assert evaluation.status == status
        if status == "satisfied":
            assert evaluation.evidence[0]["observed"]["winningCount"] == winning_count
        substituted = replace(candidate, stage_receipts={"proof": replace(
            candidate.stage_receipts["proof"], model_hash="0" * 64
        )})
        assert compiler.evaluate(scenario, substituted, contexts, maximum_seconds=2).status == "incomplete"
        # Missing/contradictory native bindings must never use the Python count.
        contexts["proof"].context.model.proto.variables[0].name = "missing_root_value"
        stopped = compiler.evaluate(scenario, candidate, contexts, maximum_seconds=2)
        assert stopped.status == "incomplete"


def test_native_actor_contract_rescue_uses_owned_request_utility(rules):
    """Finite terminal utilities certify the tactical interface, not a full round."""
    from baloot_platform.research.ortools_research_executor import (
        DefaultStageContextCompilerV1,
        OrToolsNeutralCandidateV1,
        OrToolsResearchCompilerV1,
        OrToolsResearchStageReceiptV1,
        _proof_stage_bindings_v1,
    )
    from baloot_platform.research.scenario_contracts import ResearchScenarioV1

    raw_request = _actor_request(threshold=1, claim="winning_root_action_count").to_dict()
    raw_request.pop("sha256")
    raw_request.update(utility="contract_success", utilityEncoding="signed_win_loss_v1")
    request = ScenarioProofRequestV1.build(raw_request)
    problem = FiniteInformationSetProblemV1.build(problem_id="contract-rescue-interface", success_threshold=1,
        worlds=tuple(FiniteInformationWorldV1.build(f"world-{index}", {
            "H7": FiniteInformationNodeV1.terminal(1), "S7": FiniteInformationNodeV1.terminal(-1),
        }) for index in range(2)))
    compilation = ActorInformationSetCompilationV1(ActorInformationSetGraphV1(
        problem, {"count": 2, "complete": True}, {"kind": "test_allocation_model"},
        root_actor="north", root_actor_team="north_south"), node_count=4)
    outcome = prove_actor_information_set_v1(compilation, request, maximum_nodes=100,
        maximum_worlds=2, maximum_seconds=2, native_action_quality=True)
    original = outcome.to_dict(), outcome.model_document()
    assert outcome.completion_status == "exact_complete" and outcome.winning_root_action_count == 1
    assert "utility" not in outcome.model_document()

    def candidate_for(proof):
        return OrToolsNeutralCandidateV1("search", {"play": {"decisionState": {"prefixLength": 24}}},
            {"currentPlayer": "north", "decisionPoint": {"kind": "play"}}, (),
            {stage: OrToolsResearchStageReceiptV1(stage, proof.model_hash, proof.sha256,
                proof.sha256, proof.solver_status) for stage in ("proof", "tactics")})

    leaf = {"predicateId": "PRED-CONTRACT-RESCUE", "kind": "tactic.verified", "version": 1,
        "hardness": "hard", "arguments": {"kind": "contract_rescue", "minimumScore": 100,
            "minimumEvidence": 1, "causalRequired": True}}
    scenario = ResearchScenarioV1.build({"schemaVersion": 1, "origin": {"kind": "search"},
        "constraints": {"predicateId": "PRED-CONTRACT-COUNT", "kind": "proof.winning_root_action_count",
            "version": 1, "hardness": "report", "arguments": {"minimum": 1, "maximum": 1}},
        "decisionPoint": {"kind": "play"}, "outcome": leaf})
    candidate = candidate_for(outcome)
    compiler = OrToolsResearchCompilerV1(native_cp_sat_authority=True)
    default = _proof_stage_bindings_v1(candidate, outcome, rules, require_native=True)
    assert default["proof"].context.proof_utility is None
    assert compiler.evaluate(scenario, candidate, default, maximum_seconds=2).status == "incomplete"
    bindings = _proof_stage_bindings_v1(candidate, outcome, rules, require_native=True, proof_request=request)
    assert bindings["proof"].context.proof_utility == "contract_success"
    evaluation = compiler.evaluate(scenario, candidate, bindings, maximum_seconds=2)
    assert evaluation.status == "satisfied"
    rescue = next(row for row in evaluation.evidence if row["kind"] == "tactic.verified")
    assert rescue["observed"]["verified"] is True and rescue["observed"]["causalProven"] is True
    assert rescue["observed"]["winningVariables"] == [row["winningVariable"]
        for row in outcome.model_document()["rootActionVariables"]]
    assert (outcome.to_dict(), outcome.model_document()) == original

    wrong_request = ScenarioProofRequestV1.build({**raw_request, "successThreshold": 2})
    with pytest.raises(ValueError, match="request or utility binding"):
        _proof_stage_bindings_v1(candidate, outcome, rules, require_native=True, proof_request=wrong_request)
    with pytest.raises(ValueError, match="checksum mismatch"):
        _proof_stage_bindings_v1(candidate, outcome, rules, require_native=True,
            proof_request=replace(request, utility="round_win", utility_encoding=None))
    with pytest.raises(TypeError, match="ScenarioProofRequestV1"):
        _proof_stage_bindings_v1(candidate, outcome, rules, require_native=True, proof_request=request.to_dict())
    with pytest.raises(TypeError, match="ScenarioProofRequestV1"):
        DefaultStageContextCompilerV1(rules, native_cp_sat_authority=True, proof_request=request.to_dict())
    with pytest.raises(ValueError, match="native CP-SAT authority"):
        _proof_stage_bindings_v1(candidate, outcome, rules, proof_request=request)
    with pytest.raises(ValueError, match="native CP-SAT authority"):
        DefaultStageContextCompilerV1(rules, proof_request=request)
    document = {**outcome.model_document(), "utility": "round_win"}
    changed = replace(outcome, model_hash=canonical_sha256(document), _model_document_json=json.dumps(document))
    changed = replace(changed, sha256=canonical_sha256({key: value for key, value in changed.to_dict().items()
        if key != "sha256"}))
    with pytest.raises(ValueError, match="request or utility binding"):
        _proof_stage_bindings_v1(candidate_for(changed), changed, rules, require_native=True, proof_request=request)
    incomplete = prove_actor_information_set_v1(compilation, request, maximum_nodes=1,
        maximum_worlds=2, maximum_seconds=2, native_action_quality=True)
    assert incomplete.completion_status == "proof_incomplete"
    with pytest.raises(ValueError, match="native proof model is missing"):
        _proof_stage_bindings_v1(candidate_for(incomplete), incomplete, rules, require_native=True, proof_request=request)
    bounded_request = ScenarioProofRequestV1.build({**raw_request, "mode": "bounded", "claim": "outcome_forced",
        "horizon": {"unit": "round_end", "value": 1}, "boundaryEvaluator": "current_raw_point_margin_v1"})
    bounded = prove_actor_information_set_v1(_bounded_compilation(problem, bounded_request, complete=True),
        bounded_request, maximum_nodes=100, maximum_worlds=2, maximum_seconds=2, native_action_quality=True)
    assert bounded.completion_status == "proof_failed" and bounded.hidden_worlds["complete"] is True
    with pytest.raises(ValueError, match="do not own exhaustive proof evidence"):
        _proof_stage_bindings_v1(candidate_for(bounded), bounded, rules, require_native=True, proof_request=bounded_request)


@pytest.mark.real_clock
@pytest.mark.parametrize("tactic,prefix_length", (("kabout_creation", 0), ("kabout_preservation", 4),
    ("kabout_prevention", 0)))
def test_native_actor_kabout_uses_owned_public_winner_sums(root, rules, tactic, prefix_length, monkeypatch):
    """Real finite CEGIS certifies the interface, not a full-round Kabout search."""
    from ortools.sat.python import cp_model
    from baloot_platform.research.ortools_research_executor import (
        OrToolsNeutralCandidateV1, OrToolsResearchCompilerV1, OrToolsResearchStageReceiptV1,
        _proof_stage_bindings_v1,
    )
    from baloot_platform.research.scenario_contracts import ResearchScenarioV1

    raw = json.loads((root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
        encoding="utf-8"))["exactTracePrefix"]
    raw["playActions"] = raw["playActions"][:prefix_length]
    raw["stateAssertions"], raw["projectAssertions"] = {}, []
    if prefix_length == 0:
        raw["projectDeclarations"] = []
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        ExactTracePrefixV1.build(raw), decision_point={"kind": "play"})
    actor = Seat.parse(exact.next_actor)
    codes = [row["code"] for row in exact.legal_actions[:2]]
    assert len(codes) == 2
    raw_request = _actor_request(threshold=1, claim="winning_root_action_count").to_dict()
    raw_request.pop("sha256")
    prevention = tactic == "kabout_prevention"
    raw_request.update(utility="kabout", targetTeam=(actor.team.other if prevention else actor.team).name.lower())
    request = ScenarioProofRequestV1.build(raw_request)
    problem = FiniteInformationSetProblemV1.build(problem_id=f"{tactic}-interface", success_threshold=1,
        worlds=(FiniteInformationWorldV1.build("one", {
            codes[0]: FiniteInformationNodeV1.terminal(-1 if prevention else 1).with_forced_prefix(
                (("opponent", "forced-opponent", "hold"),)),
            codes[1]: FiniteInformationNodeV1.terminal(1 if prevention else -1).with_forced_prefix(
                (("target", "forced-target", "continue"),))}),))
    compilation = ActorInformationSetCompilationV1(ActorInformationSetGraphV1(problem,
        {"count": 1, "complete": True}, {"kind": "test_allocation_model"},
        root_actor=exact.next_actor, root_actor_team=actor.team.name.lower(),
        root_actor_relation="opponent" if prevention else "target"), node_count=2)
    compilation = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)._bind_observer_scope(
        compilation, exact, request, (exact.state,))
    graph = compilation.graph
    calls = []
    original_prove = OrToolsActorInformationSetProofV1.prove
    def record_prove(self, *args, **kwargs):
        calls.append((kwargs["budget"].sha256, kwargs.get("runtime_maximum_seconds")))
        return original_prove(self, *args, **kwargs)
    monkeypatch.setattr(OrToolsActorInformationSetProofV1, "prove", record_prove)
    outcome = prove_actor_information_set_v1(compilation, request, maximum_nodes=100,
        maximum_worlds=2, maximum_seconds=2, declared_maximum_seconds=2,
        native_action_quality=True, native_opponent_stop=prevention)
    assert outcome.completion_status == "exact_complete" and outcome.selected_action == codes[0]
    if prevention:
        assert calls[0][0] == calls[1][0] and 0 < calls[1][1] < calls[0][1] == 2
        companion = outcome.model_document()["opponentStopCertificate"]
        assert {row["rootAction"]: row["robustValue"] for row in companion["roots"]} == {codes[0]: 1, codes[1]: -1}
        stop_root = next(row for row in companion["roots"] if row["rootAction"] == codes[0])
        assert stop_root["targetPolicy"] == [{"observationId": "forced-opponent", "action": "hold"}]
        baseline = prove_actor_information_set_v1(compilation, request, maximum_nodes=100,
            maximum_worlds=2, maximum_seconds=2, declared_maximum_seconds=2, native_action_quality=True)
        explicit_default = prove_actor_information_set_v1(compilation, request, maximum_nodes=100,
            maximum_worlds=2, maximum_seconds=2, declared_maximum_seconds=2,
            native_action_quality=True, native_opponent_stop=False)
        assert baseline.to_dict() == explicit_default.to_dict()
        assert baseline.model_document() == explicit_default.model_document()
        assert baseline.selected_action == outcome.selected_action and baseline.root_actions == outcome.root_actions
        assert baseline.model_document()["actionQuality"] == outcome.model_document()["actionQuality"]
    original = outcome.to_dict(), outcome.model_document()

    def candidate_for(proof=outcome, state=exact.state):
        # Contradictory legacy trace totals must not supply native base counts.
        return OrToolsNeutralCandidateV1("search", {"play": {"decisionState": {"prefixLength": prefix_length},
            "tricks": [{"winner": Seat((int(actor) + 1) % 4).name.lower()}]}}, state, (),
            {stage: OrToolsResearchStageReceiptV1(stage, proof.model_hash, proof.sha256,
                proof.sha256, proof.solver_status) for stage in ("proof", "tactics")}, exact.exact_state_id)

    candidate = candidate_for()
    leaf = {"predicateId": "PRED-KABOUT", "kind": "tactic.verified", "version": 1, "hardness": "hard",
        "arguments": {"kind": tactic, "minimumScore": 100, "minimumEvidence": 1, "causalRequired": True}}
    raw_scenario = {"schemaVersion": 1, "origin": {"kind": "search"},
        "decisionPoint": {"kind": "play"}, "constraints": {"predicateId": "PRED-KABOUT-COUNT",
            "kind": "proof.winning_root_action_count", "version": 1, "hardness": "report",
            "arguments": {"minimum": 1, "maximum": 1}}, "outcome": leaf}
    scenario = ResearchScenarioV1.build(raw_scenario)
    compiler = OrToolsResearchCompilerV1(native_cp_sat_authority=True)

    def bindings(candidate=candidate, proof=outcome, information_graph=graph):
        return _proof_stage_bindings_v1(candidate, proof, rules, require_native=True,
            proof_request=request, information_graph=information_graph)

    bound = bindings()
    assert bound["tactics"].context.base_tricks is None
    evaluated = compiler.evaluate(scenario, candidate, bound, maximum_seconds=2)
    assert evaluated.status == "satisfied"
    observed = next(row["observed"] for row in evaluated.evidence if row["kind"] == "tactic.verified")
    assert observed["verified"] is True and observed["causalProven"] is True
    expected = [0, 0]
    expected[int(actor.team)] = prefix_length // 4
    assert [observed["baseNorthSouthTricks"], observed["baseWestEastTricks"]] == expected
    assert observed["winningVariables"] == [row["winningVariable"]
        for row in outcome.model_document()["rootActionVariables"]]
    assert (outcome.to_dict(), outcome.model_document()) == original
    if prevention:
        assert observed["selectedGuaranteeScope"] == "uniform_opponent_no_target_kabout_policy"
        assert observed["stoppingVariables"] == [row["winningVariable"] for row in companion["rootActionVariables"]]
    for hardness in ("hard", "report", "prefer"):
        check = ResearchScenarioV1.build({**raw_scenario, "outcome": {**leaf, "hardness": hardness}})
        bound = bindings()
        context = bound["tactics"].context
        context.base_trick_variables = tuple(context.model.new_constant(value) for value in expected)
        assert compiler.evaluate(check, candidate, bound, maximum_seconds=2).status == "incomplete"
        if prevention:
            bound = bindings()
            context = bound["tactics"].context
            context.root_stopping_variables = tuple(context.model.new_constant(1) for _ in codes)
            assert compiler.evaluate(check, candidate, bound, maximum_seconds=2).status == "incomplete"
        if not prevention:
            without_stop = ResearchScenarioV1.build({**raw_scenario, "outcome": {**leaf, "hardness": hardness,
                "arguments": {**leaf["arguments"], "kind": "kabout_prevention"}}})
            assert compiler.evaluate(without_stop, candidate, bindings(), maximum_seconds=2).status == "incomplete"
    bound = bindings()
    context = bound["proof"].context
    context.model.add(sum(context.base_trick_variables) != prefix_length // 4)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 2
    assert solver.solve(context.model) == cp_model.INFEASIBLE
    bound = bindings()
    context = bound["proof"].context
    base_indices = {variable.index for variable in context.base_trick_variables}
    changed_count = 0
    source_model = cp_model.CpModel()
    assert source_model.proto.parse_text_format(outcome.model_document()["modelProto"])
    for constraint in tuple(context.model.proto.constraints)[len(source_model.proto.constraints):]:
        if (constraint.has_linear() and len(base_indices.intersection(constraint.linear.vars)) == 1):
            count_index = next(index for index in constraint.linear.vars if index in base_indices)
            constraint.linear.vars.clear()
            constraint.linear.vars.append(count_index)
            constraint.linear.coeffs.clear()
            constraint.linear.coeffs.append(1)
            constraint.linear.domain.clear()
            constraint.linear.domain.extend([1, 1] if prefix_length == 0 else [0, 0])
            changed_count += 1
    assert changed_count == 2
    assert compiler.evaluate(scenario, candidate, bound, maximum_seconds=2).status == "incomplete"

    # Historical/default calls ignore the optional graph and keep their model/hash.
    legacy = _proof_stage_bindings_v1(candidate, outcome, rules)
    legacy_graph = _proof_stage_bindings_v1(candidate, outcome, rules, information_graph=graph)
    assert legacy["proof"].context.sha256 == legacy_graph["proof"].context.sha256
    assert str(legacy["proof"].context.model.proto) == str(legacy_graph["proof"].context.model.proto)
    # Even legacy model utility metadata cannot qualify the newly supported path.
    document = {**outcome.model_document(), "utility": "kabout"}
    missing_request = replace(outcome, model_hash=canonical_sha256(document), _model_document_json=json.dumps(document))
    missing_request = replace(missing_request, sha256=canonical_sha256({key: value
        for key, value in missing_request.to_dict().items() if key != "sha256"}))
    missing_candidate = candidate_for(missing_request)
    unowned = _proof_stage_bindings_v1(missing_candidate, missing_request, rules, require_native=True)
    assert unowned["proof"].context.proof_utility == "kabout"
    for hardness in ("hard", "report", "prefer"):
        check = ResearchScenarioV1.build({**raw_scenario, "outcome": {**leaf, "hardness": hardness}})
        assert compiler.evaluate(check, missing_candidate, unowned,
            maximum_seconds=2).status == "incomplete"
    for changed_candidate, changed_proof, changed_graph in (
        (candidate_for(state={**exact.state, "roundNumber": 99}), outcome, graph),
        (replace(candidate, exact_state_id="STATE-OTHER"), outcome, graph),
        (candidate, outcome, None),
        (candidate, replace(outcome, completion_status="proof_incomplete"), graph),
        (candidate, outcome, replace(graph, analysis_scope={"mode": "bounded"})),
    ):
        with pytest.raises(ValueError):
            bindings(changed_candidate, changed_proof, changed_graph)
    wrong_request = ScenarioProofRequestV1.build({**raw_request, "successThreshold": 2})
    with pytest.raises(ValueError, match="request or utility binding"):
        _proof_stage_bindings_v1(candidate, outcome, rules, require_native=True,
            proof_request=wrong_request, information_graph=graph)
    if prevention:
        for field, value in (("requestHash", "0" * 64), ("worldIds", ["other-world"]),
                ("roots", list(reversed(companion["scope"]["roots"]))), ("graphHash", "0" * 64),
                ("modelHash", "0" * 64), ("budgetHash", "0" * 64), ("sourcePrefix", [])):
            document = outcome.model_document()
            changed = document["opponentStopCertificate"]
            if field in {"modelHash", "budgetHash", "sourcePrefix"}:
                changed[field] = value
            else:
                changed["scope"][field] = value
            changed["sha256"] = canonical_sha256({key: item for key, item in changed.items() if key != "sha256"})
            changed_proof = replace(outcome, model_hash=canonical_sha256(document), _model_document_json=json.dumps(document))
            changed_proof = replace(changed_proof, sha256=canonical_sha256({key: item
                for key, item in changed_proof.to_dict().items() if key != "sha256"}))
            with pytest.raises(ValueError, match="stopping"):
                bindings(candidate_for(changed_proof), changed_proof, graph)
        # Even a threshold that counts losses as wins cannot invent a causal Kabout alternative.
        loss_request = ScenarioProofRequestV1.build({**raw_request, "successThreshold": -1})
        losses = FiniteInformationSetProblemV1.build(problem_id="prevention-no-kabout-alternative", success_threshold=-1,
            worlds=(FiniteInformationWorldV1.build("one", {code: FiniteInformationNodeV1.terminal(-1) for code in codes}),))
        loss_compilation = replace(compilation, graph=replace(graph, problem=losses))
        loss_proof = prove_actor_information_set_v1(loss_compilation, loss_request, maximum_nodes=100,
            maximum_worlds=2, maximum_seconds=2, native_action_quality=True, native_opponent_stop=True)
        loss_candidate = candidate_for(loss_proof)
        loss_binding = _proof_stage_bindings_v1(loss_candidate, loss_proof, rules, require_native=True,
            proof_request=loss_request, information_graph=loss_compilation.graph)
        assert loss_proof.winning_root_action_count == 2
        assert compiler.evaluate(scenario, loss_candidate, loss_binding, maximum_seconds=2).status == "target_miss"
        cancelled = False
        original_dual = OrToolsActorInformationSetProofV1._opponent_stop_problem
        def cancel_dual(*args, **kwargs):
            nonlocal cancelled
            cancelled = True
            return original_dual(*args, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(OrToolsActorInformationSetProofV1, "_opponent_stop_problem", staticmethod(cancel_dual))
            interrupted = prove_actor_information_set_v1(compilation, request, maximum_nodes=100,
                maximum_worlds=2, maximum_seconds=2, native_action_quality=True,
                native_opponent_stop=True, cancel_check=lambda: cancelled)
        assert interrupted.completion_status == "exact_complete"
        assert interrupted.model_document()["opponentStopCertificate"]["completionStatus"] == "proof_incomplete"
        with pytest.raises(ValueError):
            bindings(candidate_for(interrupted), interrupted, graph)
        return  # The shared malformed public-history guards are covered by the other two cases.
    # Rehashed adversarial interface receipts exercise primitive validation after ownership.
    for state in (
        {key: value for key, value in exact.state.items() if key != "completedTricks"},
        {**exact.state, "currentTrick": {**exact.state["currentTrick"], "index": 8}},
        {**exact.state, "completedTricks": [{"index": 0, "winner": True, "plays": [{}, {}, {}, {}]}],
            "currentTrick": {"index": 1}},
        {**exact.state, "completedTricks": [{"index": 0, "plays": [{}, {}, {}, {}]}], "currentTrick": {"index": 1}},
    ):
        changed_graph = replace(graph, model_document={**graph.model_document, "observerScopeBinding": {
            **graph.model_document["observerScopeBinding"], "sourceStateHash": canonical_sha256(state)}})
        changed_proof = replace(outcome, graph_hash=changed_graph.to_dict()["sha256"])
        changed_proof = replace(changed_proof, sha256=canonical_sha256({key: value
            for key, value in changed_proof.to_dict().items() if key != "sha256"}))
        with pytest.raises(ValueError, match="native public trick"):
            bindings(candidate_for(changed_proof, state), changed_proof, changed_graph)


def test_actor_kabout_failure_is_not_a_uniform_opponent_prevention_certificate(root, rules):
    """A different hidden-world witness defeats each proposed stopping policy."""
    raw = _actor_request(threshold=1, claim="winning_root_action_count").to_dict()
    raw.pop("sha256")
    request = ScenarioProofRequestV1.build({**raw, "utility": "kabout", "targetTeam": "west_east"})
    prefix = json.loads((root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
        encoding="utf-8"))["exactTracePrefix"]
    prefix.update(playActions=[], projectDeclarations=[], projectAssertions=[], stateAssertions={})
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        ExactTracePrefixV1.build(prefix), decision_point={"kind": "play"})
    other = copy.deepcopy(exact.state)
    other["hands"]["west"][0], other["hands"]["east"][0] = other["hands"]["east"][0], other["hands"]["west"][0]
    for seat in ("west", "east"):
        other["initialHands"][seat] = copy.deepcopy(other["hands"][seat])
    states = (exact.state, other)
    utilities = ((-1, 1), (1, -1))
    problem = FiniteInformationSetProblemV1.build(problem_id="kabout-prevention-quantifiers", success_threshold=1,
        worlds=tuple(FiniteInformationWorldV1.build(canonical_sha256(state), {
            "H7": FiniteInformationNodeV1.decision_observed("opponent", observation_id="same-hidden-choice",
                children={action: FiniteInformationNodeV1.terminal(value)
                    for action, value in zip(("L", "R"), values)}),
            "D7": FiniteInformationNodeV1.terminal(-1),
            "S7": FiniteInformationNodeV1.terminal(1)}) for state, values in zip(states, utilities)))
    graph = ActorInformationSetGraphV1(problem, {"count": 2, "complete": True},
        {"kind": "test_allocation_model"}, root_actor="north", root_actor_team="north_south",
        root_actor_relation="opponent")
    compilation = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)._bind_observer_scope(
        ActorInformationSetCompilationV1(graph, node_count=10), exact, request, states)
    graph = compilation.graph
    proof = prove_actor_information_set_v1(compilation, request,
        maximum_nodes=100, maximum_worlds=2, maximum_seconds=2, native_action_quality=True, native_opponent_stop=True)
    assert proof.completion_status == "exact_complete"
    assert {row.action_code: row.value for row in proof.root_actions} == {"H7": -1, "D7": -1, "S7": 1}
    assert proof.selected_action == "H7"
    assert all(any(world[action] == 1 for world in utilities) for action in range(2))
    from baloot_platform.research.scenario_predicate_capabilities_v3 import NATIVE_V3_ACTOR_ROOT_TACTIC_IDS
    assert "kabout_prevention" in NATIVE_V3_ACTOR_ROOT_TACTIC_IDS
    assert {row["rootAction"]: row["robustValue"]
        for row in proof.model_document()["opponentStopCertificate"]["roots"]} == {"H7": -1, "D7": 1, "S7": -1}
    from baloot_platform.research.ortools_research_executor import (
        OrToolsNeutralCandidateV1, OrToolsResearchCompilerV1, OrToolsResearchStageReceiptV1,
        _proof_stage_bindings_v1,
    )
    from baloot_platform.research.scenario_contracts import ResearchScenarioV1
    candidate = OrToolsNeutralCandidateV1("search", {"play": {"decisionState": {"prefixLength": 0}}},
        exact.state, (),
        {stage: OrToolsResearchStageReceiptV1(stage, proof.model_hash, proof.sha256,
            proof.sha256, proof.solver_status) for stage in ("proof", "tactics")}, exact.exact_state_id)
    for hardness in ("hard", "report", "prefer"):
        scenario = ResearchScenarioV1.build({"schemaVersion": 1, "origin": {"kind": "search"},
            "decisionPoint": {"kind": "play"}, "constraints": {"predicateId": "PRED-PREVENTION-COUNT",
                "kind": "proof.winning_root_action_count", "version": 1, "hardness": "report",
                "arguments": {"minimum": 1, "maximum": 1}}, "outcome": {
                    "predicateId": "PRED-PREVENTION", "kind": "tactic.verified", "version": 1,
                    "hardness": hardness, "arguments": {"kind": "kabout_prevention",
                        "minimumScore": 0, "minimumEvidence": 1, "causalRequired": True}}})
        binding = _proof_stage_bindings_v1(candidate, proof, rules, require_native=True,
            proof_request=request, information_graph=graph)
        result = OrToolsResearchCompilerV1(native_cp_sat_authority=True).evaluate(
            scenario, candidate, binding, maximum_seconds=2)
        assert result.status == ("target_miss" if hardness == "hard" else "satisfied")
        if result.evidence:
            evidence = next(row for row in result.evidence if row["kind"] == "tactic.verified")
            assert evidence["observed"]["verified"] is False


def test_success_threshold_mutates_native_root_win_certification() -> None:
    roots = {
        "H7": FiniteInformationNodeV1.terminal(1),
        "S7": FiniteInformationNodeV1.terminal(2),
        "D7": FiniteInformationNodeV1.terminal(3),
    }

    def prove(threshold: int):
        problem = FiniteInformationSetProblemV1.build(
            problem_id=f"threshold-native-{threshold}",
            success_threshold=threshold,
            worlds=(FiniteInformationWorldV1.build("one", roots),),
        )
        graph = ActorInformationSetGraphV1(
            problem,
            {"count": 1, "complete": True},
            {"kind": "test_allocation_model"},
        )
        return prove_actor_information_set_v1(
            ActorInformationSetCompilationV1(graph, node_count=3),
            _actor_request(
                threshold=float(threshold),
                claim="winning_root_action_count",
            ),
            maximum_nodes=100,
            maximum_worlds=2,
            maximum_seconds=2.0,
        )

    threshold_two = prove(2)
    threshold_three = prove(3)
    assert threshold_two.winning_root_action_count == 2
    assert threshold_three.winning_root_action_count == 1
    first_document = threshold_two.model_document()
    second_document = threshold_three.model_document()
    assert (
        first_document["qualificationModelHash"]
        != second_document["qualificationModelHash"]
    )
    assert {row["action"]: row["satisfies"] for row in first_document["roots"]} == {
        "H7": False,
        "S7": True,
        "D7": True,
    }
    assert {row["action"]: row["satisfies"] for row in second_document["roots"]} == {
        "H7": False,
        "S7": False,
        "D7": True,
    }


@pytest.mark.parametrize("solver_count", [1, 2])
def test_shared_cancellation_watcher_stops_active_solves(solver_count) -> None:
    from ortools.sat.python import cp_model

    solving = Event()
    together = Barrier(solver_count, action=solving.set)

    class BlockingSolver:
        def __init__(self):
            self.released = Event()
            self.stopped = False
            self.polls = 0

        def solve(self, _model):
            together.wait()
            self.released.wait()
            return cp_model.UNKNOWN

        def cancellation_requested(self):
            if not solving.is_set():
                return False
            self.polls += 1
            if self.polls == 2:
                # Let a missing stop_search fail by assertion on the next watcher tick.
                self.released.set()
            return True

        def stop_search(self):
            if self.polls == 1:
                self.stopped = True
            self.released.set()

    solvers = [BlockingSolver() for _ in range(solver_count)]
    statuses = []
    workers = [Thread(target=lambda solver=solver: statuses.append(
        _solve_cp_sat_with_cancel_v1(solver, object(), solver.cancellation_requested)))
        for solver in solvers]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert statuses == [cp_model.UNKNOWN] * solver_count
    assert all(solver.stopped for solver in solvers)
    assert sum(thread.name == "actor-info-cancel-watch" for thread in active_threads()) == 1


def _play(seat: Seat, card: str) -> dict[str, object]:
    return {"seat": seat.name.lower(), "card": Card.parse(card).to_dict()}


def _declared(project: dict[str, object]) -> bool:
    return int(project["declarationOrder"]) < 999


def _project_hands() -> dict[Seat, tuple[Card, ...]]:
    return {
        Seat.NORTH: tuple(
            map(Card.parse, ("H7", "H8", "H9", "SA", "D10", "CJ", "DQ", "C7"))
        ),
        Seat.WEST: tuple(
            map(Card.parse, ("D7", "D8", "D9", "S10", "HA", "C10", "SQ", "C8"))
        ),
        Seat.SOUTH: tuple(
            map(Card.parse, ("S7", "S8", "SJ", "HQ", "HK", "DA", "C9", "CA"))
        ),
        Seat.EAST: tuple(
            map(Card.parse, ("S9", "DJ", "SK", "H10", "HJ", "CK", "DK", "CQ"))
        ),
    }


def _public_siri(owner: Seat, order: int) -> dict[str, object]:
    return {
        "owner": owner.name.lower(),
        "kind": "siri",
        "value": 4,
        "highRank": 2,
        "declarationOrder": order,
    }


@pytest.mark.parametrize("profile", ["saudi_common_v1", "saudi_federation_2023_v8"])
def test_native_hidden_project_rank_keeps_both_consistent_worlds(root, profile):
    from baloot_platform.core.rules import RuleBook
    from baloot_platform.research.ortools_information_set_compiler import _IncompatibleProjectWorld

    rules = RuleBook.load(root / "configs/rules" / f"{profile}.yaml")
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    hands = _project_hands()
    # North's four-card sequence wins by category; West's top rank is not needed.
    hands[Seat.NORTH] = tuple(Card.parse("H10") if card == Card.parse("C7") else card for card in hands[Seat.NORTH])
    hands[Seat.EAST] = tuple(Card.parse("C7") if card == Card.parse("H10") else card for card in hands[Seat.EAST])
    other = dict(hands)
    other[Seat.NORTH] = tuple(Card.parse("D7") if card == Card.parse("D10") else card for card in hands[Seat.NORTH])
    other[Seat.WEST] = tuple(Card.parse("D10") if card == Card.parse("D7") else card for card in hands[Seat.WEST])
    state = {
        "contract": "sun", "dealer": "north", "currentPlayer": "south",
        "completedTricks": [{"winner": "north", "plays": [_play(Seat.WEST, "S10"), _play(Seat.SOUTH, "S7"),
                                       _play(Seat.EAST, "S9"), _play(Seat.NORTH, "SA")]}],
        "currentTrick": None,
        "projects": [{"owner": "north", "kind": "fifty", "value": 10,
                      "highRank": 3, "declarationOrder": 2}, _public_siri(Seat.WEST, 1)],
        "projectRankVisibility": {"mode": "comparison_only", "categoryPriorities": dict(rules.project_domain.category_priorities)},
    }
    rows, winner, receipt = compiler._projects_with_receipt(state, hands)
    changed_rows, changed_winner, changed_receipt = compiler._projects_with_receipt(state, other)
    assert winner == changed_winner == Team.NORTH_SOUTH
    assert next(row["highRank"] for row in rows if row["owner"] == "west") == 2
    assert next(row["highRank"] for row in changed_rows if row["owner"] == "west") == 3
    from baloot_platform.core.exact_state_view import _safe_projects
    assert _safe_projects({**state, "projects": rows}, "south") == _safe_projects(
        {**state, "projects": changed_rows}, "south")
    assert compiler._public_project_key({**state["projects"][1], "highRank": None}, state)[3] is None
    assert receipt["historyProjectionHash"] == changed_receipt["historyProjectionHash"]
    legacy = {key: value for key, value in state.items() if key != "projectRankVisibility"}
    with pytest.raises(_IncompatibleProjectWorld):
        compiler._projects_with_receipt(legacy, other)
    # During project decisions next_actor need not equal the pending play actor.
    from ortools.sat.python import cp_model
    from baloot_platform.research.ortools_project_fragment import attach_project_lifecycle_fragment_v1
    ownership = {card.id: seat for seat, hand in other.items() for card in hand}
    for observer, expected in (("south", cp_model.OPTIMAL), ("west", cp_model.INFEASIBLE)):
        model = cp_model.CpModel()
        fragment = attach_project_lifecycle_fragment_v1(model,
            owner_of_card=tuple(model.new_int_var(0, 3, f"owner_{card}") for card in range(32)),
            fixed_card_owners=tuple(ownership[card] for card in range(32)),
            contract=Contract.SUN, dealer=Seat.NORTH, trump=None, domain=rules.project_domain,
            second_trick_leader=Seat.NORTH)
        compiler._bind_public_project_choices(model, fragment, {**state, "currentPlayer": "west"}, observer=observer)
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        assert solver.solve(model) == expected


@pytest.mark.parametrize("native", [False, True])
def test_hidden_worlds_require_recorded_auction_and_escalation_reachability(
    rules, monkeypatch, native,
) -> None:
    engine = BalootEngine(rules)
    round_state = engine.create_round(
        seed=37,
        dealer=Seat.EAST,
        match_score=(0, 0),
    )
    engine.shuffle_and_initial_deal(round_state)
    engine.bid_and_complete_deal(
        round_state,
        forced_contract=Contract.HOKUM,
        forced_buyer=Seat.NORTH,
        forced_trump=Suit.HEARTS,
    )
    state = round_state.to_dict(reveal_hands=True)
    hands = {
        seat: tuple(
            Card.parse(card) for card in state["initialHands"][seat.name.lower()]
        )
        for seat in Seat
    }
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=native)
    if native:
        from baloot_platform.core.auction_ir import AuctionDomainIRV1
        from baloot_platform.core.bidding_machine import BiddingMachine
        purchase = Card.parse(state["purchaseCard"])
        machine = BiddingMachine(Seat.EAST, purchase.suit, AuctionDomainIRV1.from_rules(rules.data), purchase.rank)
        for row in state["biddingHistory"]:
            machine.submit(row["action"], trump=Suit.parse(row["trumpSuit"]) if row.get("trumpSuit") else None)
        while not machine.finished:
            machine.submit("pass")
        state["biddingHistory"] = [row.to_dict() for row in machine.records]
        import baloot_platform.research.ortools_scenario_model as auction_module
        def forbidden(*args, **kwargs):
            raise AssertionError("native hidden-world reachability used interpreted auction truth")
        for name in ("_enumerate_auction", "_advance_auction", "_auction_legal_choices"):
            monkeypatch.setattr(auction_module, name, forbidden)
        native_models = []
        add_native_auction = auction_module.OrToolsScenarioModelV1.add_native_auction_fragment

        def record_native_auction(scenario, **kwargs):
            native_models.append(scenario)
            return add_native_auction(scenario, **kwargs)

        monkeypatch.setattr(auction_module.OrToolsScenarioModelV1,
                            "add_native_auction_fragment", record_native_auction)

    assert compiler._recorded_nonplay_reachable(
        state,
        hands,
        maximum_seconds=2.0,
        cancel_check=None,
    )
    if native:
        template_hash = canonical_sha256(str(native_models[0].model.proto))
        recipient = Seat.parse(state.get("purchaseRecipient") or state["buyer"])
        first, second = [seat for seat in Seat if seat != recipient][:2]
        alternate = dict(hands)
        alternate[first] = (hands[second][0], *hands[first][1:])
        alternate[second] = (hands[first][0], *hands[second][1:])
        assert compiler._recorded_nonplay_reachable(
            state, alternate, maximum_seconds=2.0, cancel_check=None)
        wrong_owner = dict(hands)
        exchanged = hands[first][0]
        wrong_owner[recipient] = tuple(exchanged if card == purchase else card for card in hands[recipient])
        wrong_owner[first] = (purchase, *hands[first][1:])
        assert not compiler._recorded_nonplay_reachable(
            state, wrong_owner, maximum_seconds=2.0, cancel_check=None)
        assert len(native_models) == 1
        assert canonical_sha256(str(native_models[0].model.proto)) == template_hash
        from ortools.sat.python import cp_model
        with monkeypatch.context() as patch:
            patch.setattr("baloot_platform.research.ortools_information_set_compiler._solve_cp_sat_with_cancel_v1",
                          lambda *args: cp_model.UNKNOWN)
            with pytest.raises(InformationSetCompilationStopped, match="auction_reachability_incomplete"):
                compiler._recorded_nonplay_reachable(
                    state, hands, maximum_seconds=2.0, cancel_check=None)

    bad_auction = copy.deepcopy(state)
    bad_auction["biddingHistory"][0]["seat"] = "west"
    assert not compiler._recorded_nonplay_reachable(
        bad_auction,
        hands,
        maximum_seconds=2.0,
        cancel_check=None,
    )
    if native:
        assert len(native_models) == 2

    reached_escalation = copy.deepcopy(state)
    reached_escalation["escalationHistory"] = [
        {"seat": "west", "action": "double", "mode": "open"},
        {"seat": "north", "action": "triple", "mode": "open"},
    ]
    reached_escalation["escalation"] = "triple"
    assert compiler._recorded_nonplay_reachable(
        reached_escalation,
        hands,
        maximum_seconds=2.0,
        cancel_check=None,
    )

    bad_escalation = copy.deepcopy(state)
    bad_escalation["escalationHistory"] = [
        {"seat": "north", "action": "double", "mode": "open"}
    ]
    bad_escalation["escalation"] = "double"
    assert not compiler._recorded_nonplay_reachable(
        bad_escalation,
        hands,
        maximum_seconds=2.0,
        cancel_check=None,
    )
    if native:
        for change in ("declarer", "truncated_history"):
            invalid = copy.deepcopy(state)
            if change == "declarer":
                invalid["declarer"] = "west"
            else:
                invalid["biddingHistory"] = invalid["biddingHistory"][:1]
            assert not compiler._recorded_nonplay_reachable(
                invalid, hands, maximum_seconds=2.0, cancel_check=None)


def test_project_worlds_replay_only_reached_lifecycle(rules) -> None:
    compiler = OrToolsActorInformationSetCompilerV1(rules)
    hands = _project_hands()
    base: dict[str, object] = {
        "contract": "sun",
        "dealer": "east",
        "completedTricks": [],
        "currentTrick": None,
    }

    initial, initial_winner = compiler._projects(base, hands)
    assert initial_winner is None
    assert not any(
        _declared(row) or row["revealed"] or row["scoring"] for row in initial
    )

    first_two = copy.deepcopy(base)
    first_two["currentTrick"] = {
        "plays": [_play(Seat.NORTH, "SA"), _play(Seat.WEST, "S10")]
    }
    announced, announced_winner = compiler._projects(first_two, hands)
    assert announced_winner is None
    assert {row["owner"] for row in announced if _declared(row)} == {"north", "west"}
    assert not any(row["scoring"] or row["revealed"] for row in announced)

    compared = copy.deepcopy(base)
    compared["completedTricks"] = [
        {
            "plays": [
                _play(Seat.NORTH, "SA"),
                _play(Seat.WEST, "S10"),
                _play(Seat.SOUTH, "S7"),
                _play(Seat.EAST, "S9"),
            ]
        }
    ]
    compared_rows, winner = compiler._projects(compared, hands)
    assert winner == Team.NORTH_SOUTH
    assert any(row["owner"] == "north" and row["scoring"] for row in compared_rows)
    assert not any(row["revealed"] for row in compared_rows)

    revealing = copy.deepcopy(compared)
    revealing["currentTrick"] = {"plays": [_play(Seat.NORTH, "D10")]}
    revealed_rows, revealed_winner = compiler._projects(revealing, hands)
    assert revealed_winner == Team.NORTH_SOUTH
    assert any(row["owner"] == "north" and row["revealed"] for row in revealed_rows)
    assert not any(row["owner"] == "west" and row["revealed"] for row in revealed_rows)


def test_native_project_worlds_bind_reached_lifecycle_from_one_model(rules) -> None:
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    hands = _project_hands()
    declarations = [_public_siri(Seat.NORTH, 1), _public_siri(Seat.WEST, 2)]
    base: dict[str, object] = {
        "contract": "sun",
        "dealer": "east",
        "completedTricks": [],
        "currentTrick": None,
        "projects": [],
    }

    initial, initial_winner, initial_receipt = compiler._projects_with_receipt(
        base, hands
    )
    assert initial_winner is None
    assert not any(
        _declared(row) or row["revealed"] or row["scoring"] for row in initial
    )

    first_two = copy.deepcopy(base)
    first_two["currentTrick"] = {
        "plays": [_play(Seat.NORTH, "SA"), _play(Seat.WEST, "S10")]
    }
    first_two["projects"] = declarations
    announced, announced_winner, announced_receipt = compiler._projects_with_receipt(
        first_two, hands
    )
    assert announced_winner is None
    assert {row["owner"] for row in announced if _declared(row)} == {
        "north",
        "west",
    }
    assert not any(row["scoring"] or row["revealed"] for row in announced)

    compared = copy.deepcopy(base)
    compared["completedTricks"] = [
        {
            "plays": [
                _play(Seat.NORTH, "SA"),
                _play(Seat.WEST, "S10"),
                _play(Seat.SOUTH, "S7"),
                _play(Seat.EAST, "S9"),
            ]
        }
    ]
    compared["projects"] = declarations
    compared["currentTrick"] = {"index": 1, "plays": []}
    compared_rows, winner, compared_receipt = compiler._projects_with_receipt(
        compared, hands
    )
    assert winner == Team.NORTH_SOUTH
    assert any(row["owner"] == "north" and row["scoring"] for row in compared_rows)
    assert not any(row["revealed"] for row in compared_rows)
    assert compared_receipt["selectedProjects"]
    assert all(row["cards"] for row in compared_receipt["selectedProjects"])
    assert all(row["cards"] is None for row in compared_rows)

    revealing = copy.deepcopy(compared)
    revealing["currentTrick"] = {"plays": [_play(Seat.NORTH, "D10")]}
    revealed_rows, revealed_winner, revealed_receipt = compiler._projects_with_receipt(
        revealing, hands
    )
    assert revealed_winner == Team.NORTH_SOUTH
    assert any(row["owner"] == "north" and row["revealed"] for row in revealed_rows)
    assert not any(row["owner"] == "west" and row["revealed"] for row in revealed_rows)
    receipts = (
        initial_receipt,
        announced_receipt,
        compared_receipt,
        revealed_receipt,
    )
    assert all(receipt is not None for receipt in receipts)
    assert len({receipt["modelHash"] for receipt in receipts if receipt}) == 4
    assert all(receipt["authority"] == "cp_sat" for receipt in receipts if receipt)
    assert all(receipt["variableBindings"] for receipt in receipts if receipt)
    for receipt in receipts:
        bindings = receipt["variableBindings"]
        assert all(all(row.values()) for row in bindings)
        compact = [row for row in bindings if row["declared"] == "information_project_ineligible"]
        assert compact
        assert all(row["reachedOrder"] == "information_unreached_order" for row in compact)


def test_native_project_world_binds_explicit_actions_before_play(rules) -> None:
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    state = {
        "contract": "sun",
        "dealer": "east",
        "completedTricks": [],
        "currentTrick": None,
        "projects": [_public_siri(Seat.NORTH, 1)],
        "events": [{"type": "project_declared", "owner": "north"}],
    }
    rows, winner, receipt = compiler._projects_with_receipt(state, _project_hands())
    assert winner is None
    assert {row["owner"] for row in rows if _declared(row)} == {"north"}
    assert not any(row["scoring"] or row["revealed"] for row in rows)
    assert receipt["authority"] == "cp_sat"
    state["events"] = []
    with pytest.raises(InformationSetCompilationStopped):
        compiler._projects_with_receipt(state, _project_hands())


@pytest.mark.real_clock
def test_native_allocation_before_declarations_needs_no_project_fragment(rules, monkeypatch):
    from types import SimpleNamespace
    from baloot_platform.research import ortools_information_set_compiler as module
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    exact = SimpleNamespace(next_actor="north", state={
        "initialHands": {seat.name.lower(): [card.code for card in hand]
                         for seat, hand in _project_hands().items()},
        "completedTricks": [], "currentTrick": None, "projects": [], "events": [],
    })
    fixed = compiler._public_fixed_ownership(exact)
    def forbidden(*args, **kwargs):
        pytest.fail("no public project evidence must not build a project fragment")
    monkeypatch.setattr(module, "attach_project_lifecycle_fragment_v1", forbidden)
    real_solve = module._solve_cp_sat_with_cancel_v1
    exhaustive_flags = []
    def tracked_solve(solver, *args, **kwargs):
        exhaustive_flags.append(solver.parameters.enumerate_all_solutions)
        return real_solve(solver, *args, **kwargs)
    monkeypatch.setattr(module, "_solve_cp_sat_with_cancel_v1", tracked_solve)
    worlds = compiler._enumerate_hidden_groups(
        tuple(card for card in FULL_DECK if card not in fixed), (0, 8, 8, 8),
        deadline=time.monotonic() + 5, maximum_worlds=2,
        cancel_check=None, bounded_world_scope=True, exact=exact,
    )
    assert len(worlds) == 2 and worlds[0] != worlds[1]
    assert exhaustive_flags == [False, False]
    for world in worlds:
        assert tuple(map(len, world)) == (0, 8, 8, 8)
        assert {card for hand in world for card in hand} == set(FULL_DECK) - set(fixed)


@pytest.mark.real_clock
def test_native_batched_group_allocations_preserve_the_finite_universe(rules, monkeypatch):
    from baloot_platform.research import ortools_information_set_compiler as module
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    calls = []
    solve = module._solve_cp_sat_with_cancel_v1

    def measured(solver, *args):
        calls.append(solver.parameters.enumerate_all_solutions)
        return solve(solver, *args)

    monkeypatch.setattr(module, "_solve_cp_sat_with_cancel_v1", measured)
    arguments = dict(pool=FULL_DECK[:4], group_sizes=(2, 2), maximum_worlds=10,
        cancel_check=None)
    serial = compiler._enumerate_hidden_groups(**arguments, deadline=time.monotonic() + 5)
    assert len(serial) == 6
    calls.clear()
    batch = compiler._enumerate_hidden_groups(**arguments, deadline=time.monotonic() + 5,
        bounded_world_scope=True, _batch_allocations=True)
    assert batch == serial and calls == [True]
    arguments["maximum_worlds"] = 2
    limited = compiler._enumerate_hidden_groups(**arguments, deadline=time.monotonic() + 5,
        bounded_world_scope=True, _batch_allocations=True)
    repeated = compiler._enumerate_hidden_groups(**arguments, deadline=time.monotonic() + 5,
        bounded_world_scope=True, _batch_allocations=True)
    assert limited == repeated and len(set(limited)) == 2
    assert set(limited) <= set(serial)
    with pytest.raises(ValueError, match="explicitly bounded"):
        compiler._enumerate_hidden_groups(**arguments, deadline=time.monotonic() + 5,
            _batch_allocations=True)
    with pytest.raises(InformationSetCompilationStopped, match="cancelled"):
        compiler._enumerate_hidden_groups(**{**arguments, "cancel_check": lambda: True},
            deadline=time.monotonic() + 5, bounded_world_scope=True, _batch_allocations=True)


@pytest.mark.real_clock
@pytest.mark.parametrize("status_name", ("UNKNOWN", "MODEL_INVALID", "FEASIBLE"))
def test_native_batched_group_allocations_fail_closed_on_unfinished_search(rules, monkeypatch, status_name):
    from ortools.sat.python import cp_model
    from baloot_platform.research import ortools_information_set_compiler as module
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    solve = module._solve_cp_sat_with_cancel_v1

    def incomplete(solver, model, cancel, callback):
        assert solve(solver, model, cancel, callback) in {cp_model.OPTIMAL, cp_model.FEASIBLE}
        return getattr(cp_model, status_name)

    monkeypatch.setattr(module, "_solve_cp_sat_with_cancel_v1", incomplete)
    with pytest.raises(InformationSetCompilationStopped, match="hidden_world_solver_incomplete"):
        compiler._enumerate_hidden_groups(FULL_DECK[:4], (2, 2),
            deadline=time.monotonic() + 5, maximum_worlds=10 if status_name == "FEASIBLE" else 2,
            cancel_check=None, bounded_world_scope=True, _batch_allocations=True)


@pytest.mark.parametrize("batched", (False, True))
def test_native_interleaved_preparation_uses_the_configured_allocation_method(root, rules, monkeypatch, batched):
    raw = json.loads((root / "examples/research-scenario-bounded-quality-prefix.json").read_text(encoding="utf-8"))
    raw["playActions"] = raw["playActions"][:1]
    raw["projectActions"] = raw["projectActions"][:2]
    prefix = ExactTracePrefixV1.build(raw)
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "play"}, maximum_seconds=5)
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    calls = []
    enumerate_inputs = compiler.enumerate_graph_world_inputs

    def inputs(*args, **kwargs):
        calls.append(kwargs["_batch_allocations"])
        return enumerate_inputs(*args, **kwargs)

    monkeypatch.setattr(compiler, "enumerate_graph_world_inputs", inputs)
    request = ScenarioProofRequestV1.build({
        "schemaVersion": 1, "mode": "bounded", "visibility": "actor_information_set",
        "strategyClass": "pure_deterministic", "claim": "outcome_reachable",
        "targetTeam": "west_east", "utility": "awarded_point_margin", "successThreshold": 0,
        "budgetProfile": "bounded_v1", "horizon": {"unit": "tricks", "value": 1},
        "boundaryEvaluator": "current_raw_point_margin_v1",
    })
    method = "deterministic_cp_sat_batched_prefix_v1" if batched else "deterministic_cp_sat_prefix_v1"
    worlds, preparation = compiler._prepare_observed_play_worlds(exact, request,
        maximum_worlds=2, maximum_attempts=100, maximum_model_bytes=10_000_000,
        maximum_seconds=15, project_prefix=prefix, project_contexts={}, play_allocation_method=method)
    assert calls == [batched] and worlds
    allocation = preparation["allocationModel"]
    assert allocation["coverage"] == "supplied_worlds_only"
    assert allocation["worldCount"] == 2
    assert len({row["worldStateHash"] for row in preparation["models"]}) == 2
    if batched:
        assert allocation["worldScope"] == {"method": method, "complete": False,
            "maximumWorlds": 2, "probabilityModel": None}
    else:
        assert "worldScope" not in allocation


def test_native_empty_play_history_needs_no_future_play_solve(rules, monkeypatch):
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)

    def forbidden(*args, **kwargs):
        pytest.fail("empty history must not construct future play")

    monkeypatch.setattr(type(compiler.graph_compiler.play_stage), "solve", forbidden)
    assert compiler._legal_history(
        {"completedTricks": [], "currentTrick": None}, _project_hands(),
        maximum_seconds=1, cancel_check=None,
    )
    with pytest.raises(InformationSetCompilationStopped, match="cancelled"):
        compiler._legal_history(
            {"completedTricks": [], "currentTrick": None}, _project_hands(),
            maximum_seconds=1, cancel_check=lambda: True,
        )


@pytest.mark.real_clock
@pytest.mark.parametrize("batched", (False, True))
@pytest.mark.parametrize("profile", ("saudi_common_v1", "saudi_federation_2023_v8"))
def test_native_play_allocations_depend_only_on_the_actor_observation(root, profile, batched):
    from baloot_platform.core.rules import RuleBook

    rules = RuleBook.load(root / "configs/rules" / f"{profile}.yaml")
    hands = {
        Seat.NORTH: tuple(map(Card.parse, ("S7", "SJ", "H8", "HQ", "D9", "DK", "C10", "CA"))),
        Seat.WEST: tuple(map(Card.parse, ("S8", "SQ", "H9", "HK", "D10", "DA", "C7", "CJ"))),
        Seat.SOUTH: tuple(map(Card.parse, ("S9", "SK", "H10", "HA", "D7", "DJ", "C8", "CQ"))),
        Seat.EAST: tuple(map(Card.parse, ("S10", "SA", "H7", "HJ", "D8", "DQ", "C9", "CK"))),
    }
    changed = dict(hands)
    changed[Seat.WEST] = tuple(Card.parse({"CJ": "C8", "D10": "C9"}.get(card.code, card.code))
                               for card in hands[Seat.WEST])
    changed[Seat.SOUTH] = tuple(Card.parse("CJ") if card.code == "C8" else card for card in hands[Seat.SOUTH])
    changed[Seat.EAST] = tuple(Card.parse("D10") if card.code == "C9" else card for card in hands[Seat.EAST])
    layout = DealLayoutV1.build(rules.structural_domain, dealer=Seat.EAST, buyer=Seat.NORTH)
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    exacts = []
    for allocation in (hands, changed):
        deck = [None] * 32
        for seat in Seat:
            positions = layout.positions_for(seat, final=True)
            cards = list(allocation[seat])
            if seat == Seat.NORTH:
                deck[layout.purchase_position] = cards.pop(0).id
                positions = tuple(position for position in positions if position != layout.purchase_position)
            for position, card in zip(positions, cards, strict=True):
                deck[position] = card.id
        prefix = ExactTracePrefixV1.build({"schemaVersion": 1, "dealer": "east", "deckOrder": deck,
            "auctionActions": [{"actor": "north", "action": "sun"}], "escalationActions": [],
            "projectDeclarations": [], "playActions": []})
        exacts.append(normalizer.normalize(prefix, decision_point={"kind": "play"}, maximum_seconds=5))
    assert exacts[0].state_hash != exacts[1].state_hash
    assert not exacts[0].state["projects"] and exacts[1].state["projects"]
    assert all(project["declarationOrder"] == 999 for project in exacts[1].state["projects"])
    assert canonical_observer_information_hash_v1(complete_state=exacts[0].state, acting_seat="north") == (
        canonical_observer_information_hash_v1(complete_state=exacts[1].state, acting_seat="north"))
    samples = []
    for exact in exacts:
        compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
        worlds, _receipt = compiler._enumerate_worlds(exact, maximum_worlds=2, maximum_attempts=100,
            deadline=time.monotonic() + 15, cancel_check=None, bounded_world_scope=True,
            _batch_allocations=batched)
        samples.append({canonical_sha256(world["hands"]) for world in worlds})
    assert samples[0] == samples[1]


@pytest.mark.real_clock
@pytest.mark.parametrize("after_reveal", (False, True))
def test_short_native_prefix_does_not_reveal_a_future_actor_project(rules, after_reveal, monkeypatch):
    prefix = ExactTracePrefixV1.build({
        "schemaVersion": 1, "dealer": "east", "roundNumber": 1, "matchScore": [0, 0],
        "deckOrder": [19, 21, 22, 2, 5, 6, 0, 1, 4, 8, 10, 12, 23, 30, 16, 17,
                      9, 29, 25, 26, 24, 18, 20, 15, 27, 31, 3, 7, 11, 13, 14, 28],
        "auctionActions": [{"actor": "north", "action": "hokum", "trumpSuit": "clubs"},
                           *({"actor": seat, "action": "pass"} for seat in ("west", "south", "east"))],
        "escalationActions": [], "projectDeclarations": [], "projectAssertions": [],
        "projectActions": [{"actor": "north", "action": "declare", "projectId": "PROJECT-2580A924C75447BB"},
                           *({"actor": seat, "action": "pass"} for seat in ("west", "south", "east"))],
        "playActions": [{"actor": seat, "card": card} for seat, card in
                        (("north", 18), ("west", 16), ("south", 29), ("east", 28), ("east", 8))
                        + ((("north", 24), ("west", 15), ("south", 11), ("north", 19)) if after_reveal else ())],
        "stateAssertions": {},
    })
    exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize_play_or_project(prefix)
    assert exact.next_actor == ("west" if after_reveal else "north")
    assert any(project["revealed"] for project in exact.state["projects"]) == after_reveal
    hands = {Seat.parse(seat): tuple(Card.parse(card) for card in cards)
             for seat, cards in exact.state["initialHands"].items()}
    projects, _winner = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)._projects(exact.state, hands)
    assert projects == exact.state["projects"]
    if after_reveal:
        compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
        monkeypatch.setattr(
            compiler,
            "_legal_history",
            lambda *args, **kwargs: pytest.fail(
                "native allocation model already certified the play history"
            ),
        )
        worlds, receipt = compiler._enumerate_worlds(
            exact, maximum_worlds=2, maximum_attempts=100,
            deadline=time.monotonic() + 15, cancel_check=None, bounded_world_scope=True,
        )
        assert len(worlds) == 2
        assert receipt["attempts"] == 2
        from ortools.sat.python import cp_model
        from baloot_platform.research import ortools_information_set_compiler as compiler_module
        real_solve = compiler_module._solve_cp_sat_with_cancel_v1
        solve_calls = 0

        batched, batch_receipt = compiler._enumerate_worlds(
            exact, maximum_worlds=2, maximum_attempts=100,
            deadline=time.monotonic() + 15, cancel_check=None,
            bounded_world_scope=True, _batch_allocations=True,
        )
        repeated, repeated_receipt = compiler._enumerate_worlds(
            exact, maximum_worlds=2, maximum_attempts=100,
            deadline=time.monotonic() + 15, cancel_check=None,
            bounded_world_scope=True, _batch_allocations=True,
        )
        assert len(batched) == len({canonical_sha256(world) for world in batched}) == 2
        assert batched == repeated and batch_receipt == repeated_receipt
        assert batch_receipt['attempts'] == 2
        assert batch_receipt['worldScope']['method'] == 'deterministic_cp_sat_batched_prefix_v1'
        assert receipt['worldScope']['method'] == 'deterministic_cp_sat_prefix_v1'
        real_world_state = compiler._world_state
        visited = []

        def reject_first_world(*args, **kwargs):
            visited.append(canonical_sha256({seat.name: [card.id for card in cards]
                for seat, cards in args[1].items()}))
            return None if len(visited) == 1 else real_world_state(*args, **kwargs)

        with monkeypatch.context() as batch_patch:
            batch_patch.setattr(compiler, '_world_state', reject_first_world)
            refilled, refill_receipt = compiler._enumerate_worlds(
                exact, maximum_worlds=2, maximum_attempts=100,
                deadline=time.monotonic() + 15, cancel_check=None,
                bounded_world_scope=True, _batch_allocations=True,
            )
        assert len(refilled) == 2 and refill_receipt['attempts'] == 3
        assert len(visited) == len(set(visited)) == 3
        with pytest.raises(InformationSetCompilationStopped) as cancelled:
            compiler._enumerate_worlds(
                exact, maximum_worlds=2, maximum_attempts=100,
                deadline=time.monotonic() + 15, cancel_check=lambda: True,
                bounded_world_scope=True, _batch_allocations=True,
            )
        assert cancelled.value.reason == 'cancelled'
        with pytest.raises(ValueError, match='bounded world scope'):
            compiler._enumerate_worlds(exact, maximum_worlds=2, maximum_attempts=100,
                deadline=time.monotonic() + 15, cancel_check=None, _batch_allocations=True)

        def one_world_then_unknown(*args, **kwargs):
            nonlocal solve_calls
            is_allocation = any(variable.name.startswith("owns_") for variable in args[1].proto.variables)
            if is_allocation:
                solve_calls += 1
                if solve_calls == 2:
                    return cp_model.UNKNOWN
            return real_solve(*args, **kwargs)

        with monkeypatch.context() as bounded_patch:
            bounded_patch.setattr(compiler_module, "_solve_cp_sat_with_cancel_v1", one_world_then_unknown)
            partial_worlds, partial_receipt = compiler._enumerate_worlds(
                exact, maximum_worlds=8, maximum_attempts=100,
                deadline=time.monotonic() + 15, cancel_check=None, bounded_world_scope=True,
            )
        assert len(partial_worlds) == 1
        assert partial_receipt["boundedTerminationReason"] == "hidden_world_solver_incomplete"
        from baloot_platform.research.ortools_play_tail import PlayTailSolveStopped
        fixed = compiler._public_fixed_ownership(exact)
        def cancel_history(**kwargs):
            assert kwargs["prefix"] == compiler._history(exact.state)
            assert kwargs["shared_model"] is not None
            assert len(kwargs["shared_owner_of_card"]) == 32
            raise PlayTailSolveStopped("cancelled")
        monkeypatch.setattr(compiler.graph_compiler.play_stage, "build_model", cancel_history)
        with pytest.raises(InformationSetCompilationStopped) as stopped:
            compiler._enumerate_hidden_groups(
                tuple(card for card in FULL_DECK if card not in fixed),
                tuple(8 - sum(owner == seat for owner in fixed.values()) for seat in Seat),
                deadline=time.monotonic() + 15, maximum_worlds=2,
                cancel_check=None, bounded_world_scope=True, exact=exact,
            )
        assert stopped.value.reason == "cancelled"
        assert stopped.value.partial_receipt["stage"] == "hidden_world_nonplay_allocation"


def test_native_project_world_rejects_hidden_hand_that_cannot_match_declaration(
    rules,
) -> None:
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    hands = _project_hands()
    state: dict[str, object] = {
        "contract": "sun",
        "dealer": "east",
        "completedTricks": [],
        "currentTrick": {"plays": [_play(Seat.NORTH, "SA"), _play(Seat.WEST, "S10")]},
        "projects": [_public_siri(Seat.NORTH, 1), _public_siri(Seat.WEST, 2)],
    }
    rows, winner = compiler._projects(state, hands)
    assert winner is None
    assert {row["owner"] for row in rows if _declared(row)} == {"north", "west"}

    incompatible = dict(hands)
    north = list(incompatible[Seat.NORTH])
    east = list(incompatible[Seat.EAST])
    north_index = north.index(Card.parse("H9"))
    east_index = east.index(Card.parse("SK"))
    north[north_index], east[east_index] = east[east_index], north[north_index]
    incompatible[Seat.NORTH] = tuple(north)
    incompatible[Seat.EAST] = tuple(east)
    with pytest.raises(ValueError, match="public project history is incompatible"):
        compiler._projects(state, incompatible)


def test_native_project_compiler_has_no_python_truth_constant_projection() -> None:
    source = textwrap.dedent(
        inspect.getsource(OrToolsActorInformationSetCompilerV1._projects_v3)
    )
    tree = ast.parse(source)
    assert "attach_project_lifecycle_fragment_v1" in source
    assert "OrToolsProjectLifecycleModelV1" not in source
    # An unreached comparison may bind literal zero, never Python-derived truth.
    constants = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "new_constant"]
    assert all(len(node.args) == 1 and not node.keywords
               and isinstance(node.args[0], ast.Constant)
               and type(node.args[0].value) is int and node.args[0].value == 0
               for node in constants)


@pytest.mark.real_clock
@pytest.mark.parametrize("mode,maximum_worlds,horizon,stop_at", [
    ("exact", 64, None, None), ("bounded", 64, "round_end", None), ("bounded", 1, "round_end", None),
    ("bounded", 1, "plies", None), ("bounded", 1, "tricks", None),
    ("bounded", 1, "plies", 1), ("bounded", 1, "plies", 2),
])
def test_actor_hokum_compiles_and_proves_exactly(root: Path, rules, mode, maximum_worlds, horizon, stop_at, monkeypatch) -> None:
    case = next(
        item
        for item in json.loads(
            (root / "tests" / "fixtures" / "synthetic-exact-endgames.json").read_text(encoding="utf-8")
        )["cases"]
        if item["contract"] == "hokum" and len(item["rootActions"]) == 1
    )
    engine = BalootEngine(rules)
    state = engine.create_round(
        seed=case["seed"],
        dealer=Seat.parse(case["dealer"]),
        match_score=(0, 0),
    )
    engine.shuffle_and_initial_deal(state, cut_position=case["cutPosition"])
    engine.bid_and_complete_deal(
        state,
        forced_contract=Contract.HOKUM,
        forced_buyer=Seat.parse(case["buyer"]),
        forced_trump=Suit.parse(case["trumpSuit"]),
    )
    engine.prepare_projects(state)
    for action_id in case["actionPrefix"]:
        engine.play_card(state, int(action_id))
    document = state.to_dict(reveal_hands=True)
    state_hash = canonical_sha256(document)
    identity = canonical_sha256({"stateHash": state_hash, "rulesHash": rules.sha256})
    exact = OrToolsNormalizedExactStateV1(
        f"STATE-{identity[:20].upper()}",
        "0" * 64,
        rules.sha256,
        state_hash,
        "playing",
        "play",
        state.current_player.name.lower(),
        tuple(
            {
                "actionId": card.id,
                "code": card.code,
                "card": card.to_dict(),
                "ekka": False,
            }
            for card in engine.legal_actions(state)
        ),
        document,
        0,
        0,
        "1" * 64,
        identity,
        OrToolsExactNormalizationReceiptV1({"sha256": "2" * 64}),
    )
    request_builder = _bounded_actor_request if mode == "bounded" else _actor_request
    request = request_builder(threshold=case["expectedMargin"], claim="winning_root_action_count")
    if horizon is not None:
        request_value = request.to_dict()
        request_value.pop("sha256")
        request_value["horizon"] = {"unit": horizon, "value": 1}
        request = ScenarioProofRequestV1.build(request_value)
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=mode == "bounded")
    if stop_at is not None:
        from baloot_platform.research.ortools_complete_state_graph import _CpSatPathOracle
        from baloot_platform.research.ortools_play_tail import PlayTailSolveStopped
        original = _CpSatPathOracle.solve
        calls = 0
        def stopped_solve(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == stop_at:
                raise PlayTailSolveStopped("cancelled")
            return original(self, *args, **kwargs)
        monkeypatch.setattr(_CpSatPathOracle, "solve", stopped_solve)
        with pytest.raises(InformationSetCompilationStopped) as stopped:
            compiler.compile(exact, request, maximum_nodes=100_000,
                maximum_depth=8, maximum_worlds=maximum_worlds, maximum_seconds=30)
        assert calls == stop_at
        assert stopped.value.reason == "cancelled"
        assert stopped.value.partial_receipt["stage"] == "information_graph_compilation"
        return
    compilation = compiler.compile(
        exact,
        request,
        maximum_nodes=100_000,
        maximum_depth=8,
        maximum_worlds=maximum_worlds,
        maximum_seconds=30,
    )
    if mode == "bounded" and maximum_worlds == 64:
        # Exercise real spawned native workers without changing allocation coverage.
        limits = dict(maximum_nodes=100_000, maximum_depth=8,
            cancel_check=None, requested_recall={}, histories=None)
        serial = compiler._compile_play_worlds(exact, request, [document] * 4,
            deadline=time.monotonic() + 30, **limits)
        cpu_samples = []
        parallel = compiler._compile_play_worlds_parallel(
            exact, request, [document] * 4, graph_workers=4,
            worker_cpu_callback=cpu_samples.append, deadline=time.monotonic() + 30, **limits)
        assert parallel == serial
        assert cpu_samples and all(sample > 0 for sample in cpu_samples)
        with pytest.raises(InformationSetCompilationStopped, match="maximum_nodes"):
            compiler._compile_play_worlds_parallel(exact, request, [document] * 4,
                graph_workers=4, deadline=time.monotonic() + 30,
                **{**limits, "maximum_nodes": serial[1] - 1})
        cpu_samples.clear()
        with pytest.raises(InformationSetCompilationStopped, match="cancelled"):
            compiler._compile_play_worlds_parallel(exact, request, [document] * 4,
                graph_workers=4, worker_cpu_callback=cpu_samples.append, deadline=time.monotonic() + 30,
                **{**limits, "cancel_check": lambda: sum(cpu_samples) > 0.001})
        assert sum(cpu_samples) > 0.001
    proof = prove_actor_information_set_v1(
        compilation,
        request,
        maximum_nodes=100_000,
        maximum_worlds=maximum_worlds,
        maximum_seconds=30,
        native_action_quality=mode == "bounded",
    )

    assert compilation.graph.hidden_worlds["complete"] is (maximum_worlds > 1)
    assert compilation.graph.hidden_worlds["count"] == 1
    actor = Seat.parse(document["currentPlayer"])
    assert compilation.graph.root_actor == actor.name.lower()
    assert compilation.graph.root_actor_team == actor.team.name.lower()
    assert compilation.graph.root_actor_relation == (
        "target" if actor.team == Team.NORTH_SOUTH else "opponent"
    )
    assert proof.completion_status == f"{mode}_complete"
    assert len(case["rootActions"]) == 1
    assert [card.code for card in engine.legal_actions(state)] == case["rootActions"]
    assert proof.selected_action == case["rootActions"][0]
    if horizon in {"plies", "tricks"}:
        scope = proof.model_document()["analysisScope"]
        assert scope["boundaryEvaluated"] is True
        assert scope["continuationCoverage"] == "requested_horizon"
        assert scope["transitionModelsHash"] == canonical_sha256(
            compilation.graph.model_document["transitionModels"]
        )
        engine.play_card(state, Card.parse(case["rootActions"][0]).id)
        points = sum(
            trick.raw_points * (1 if trick.winner.team == Team.NORTH_SOUTH else -1)
            for trick in state.completed_tricks
        )
        assert proof.root_actions[0].value == points
    else:
        assert proof.winning_root_action_count == 1


@pytest.mark.parametrize("contract,trump,locked", [
    ("sun", None, False), ("hokum", "spades", False), ("hokum", "spades", True),
])
def test_native_history_checks_only_recorded_tricks(rules, monkeypatch, contract, trump, locked):
    from baloot_platform.research.ortools_play_tail import OrToolsPlayTailStageV1

    initial = {seat: tuple(FULL_DECK[int(seat) * 8:(int(seat) + 1) * 8]) for seat in Seat}
    trace = OrToolsPlayTailStageV1(native_conversion=True).solve(
        cards=FULL_DECK, ownership={card: seat for seat in Seat for card in initial[seat]},
        contract=contract, trump=trump, first_leader=Seat.SOUTH,
        play_domain=rules.play_domain, structural_domain=rules.structural_domain,
        locked=locked, maximum_time_seconds=30,
    )
    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    legacy = OrToolsActorInformationSetCompilerV1(rules)
    solve = compiler.graph_compiler.play_stage.solve
    steps = []

    def tracked_solve(**kwargs):
        steps.append(kwargs["play_step_count"])
        return solve(**kwargs)

    monkeypatch.setattr(compiler.graph_compiler.play_stage, "solve", tracked_solve)
    rows = [{"seat": action.actor.name.lower(), "card": action.card.to_dict(), "ekka": action.ekka}
            for action in trace.plays]
    for length in (0, 1, 4, 5, 31, 32):
        completed = length // 4
        state = {
            "contract": contract, "trumpSuit": trump, "firstLeader": "south", "locked": locked,
            "completedTricks": [
                {"plays": rows[index * 4:(index + 1) * 4], "winner": trace.trick_winners[index].name.lower()}
                for index in range(completed)
            ],
            "currentTrick": {"plays": rows[completed * 4:length]},
        }
        assert compiler._legal_history(state, initial, maximum_seconds=30, cancel_check=None)
        if length:
            assert steps[-1] == ((length + 3) // 4) * 4
        else:
            assert not steps
        assert legacy._legal_history(state, initial, maximum_seconds=30, cancel_check=None)

        if completed:
            invalid = copy.deepcopy(state)
            invalid["completedTricks"][0]["winner"] = rules.structural_domain.offset(trace.trick_winners[0], 1).name.lower()
        elif length:
            invalid = copy.deepcopy(state)
            invalid["currentTrick"]["plays"][0]["seat"] = "north"
        else:
            continue
        assert not compiler._legal_history(invalid, initial, maximum_seconds=30, cancel_check=None)
        assert not legacy._legal_history(invalid, initial, maximum_seconds=30, cancel_check=None)


@pytest.mark.parametrize("status", ["unknown", "cancelled", "model_invalid"])
def test_native_unknown_world_reachability_is_not_an_excluded_world(rules, monkeypatch, status):
    from baloot_platform.research.ortools_play_tail import PlayTailSolveStopped

    compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
    monkeypatch.setattr(compiler.graph_compiler.play_stage, "solve", lambda **kwargs: (_ for _ in ()).throw(PlayTailSolveStopped(status)))
    with pytest.raises(InformationSetCompilationStopped) as stopped:
        compiler._legal_history(
            {"contract": "sun", "firstLeader": "south", "completedTricks": [],
             "currentTrick": {"plays": [{"seat": "south", "card": FULL_DECK[16].to_dict()}]}},
            {seat: tuple(FULL_DECK[int(seat) * 8:(int(seat) + 1) * 8]) for seat in Seat},
            maximum_seconds=1, cancel_check=None,
        )
    assert stopped.value.reason == status
    assert stopped.value.partial_receipt["stage"] == "hidden_world_play_reachability"


def test_actor_hokum_requires_trump_and_exhaustion_is_receipted(rules) -> None:
    compiler = OrToolsActorInformationSetCompilerV1(rules)
    with pytest.raises(ValueError, match="requires trumpSuit"):
        compiler._projects(
            {"contract": "hokum", "dealer": "east"},
            {seat: () for seat in Seat},
        )

    stopped = InformationSetCompilationStopped(
        "maximum_hidden_worlds",
        stage="hidden_world_enumeration",
        evidence={"attempts": 4, "acceptedWorlds": 3},
    )
    assert stopped.partial_receipt["stage"] == "hidden_world_enumeration"
    assert stopped.partial_receipt["evidence"] == {"attempts": 4, "acceptedWorlds": 3}


def test_incomplete_receipt_json_snapshot_preserves_hash_and_owns_evidence():
    from baloot_platform.research.scenario_contracts import canonical_json, canonical_sha256
    evidence = {"queries": ({"actions": ["pass", "صن"], "value": -3},), "complete": False}
    expected = {"schemaVersion": 1, "status": "incomplete", "stage": "fixture",
        "exhaustionReason": "maximum_seconds", "evidence": copy.deepcopy(evidence)}
    stopped = InformationSetCompilationStopped("maximum_seconds", stage="fixture", evidence=evidence)
    assert stopped.partial_receipt == {**json.loads(canonical_json(expected)), "sha256": canonical_sha256(expected)}
    evidence["queries"][0]["actions"].append("mutated")
    assert stopped.partial_receipt["evidence"]["queries"][0]["actions"] == ["pass", "صن"]
    stopped.partial_receipt["evidence"]["queries"][0]["actions"].append("receipt-only")
    assert evidence["queries"][0]["actions"] == ["pass", "صن", "mutated"]
