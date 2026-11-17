"""Bounded actor-safe pre-screen for constructed Research candidates."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import math
import time
import multiprocessing
from multiprocessing.util import Finalize
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeout
from collections import deque
from types import SimpleNamespace
from typing import Any, Mapping


_proposal_cancel = _proposal_cpu = _proposal_slot = None


def _sample_proposal_cpu():
    with _proposal_cpu.get_lock():
        _proposal_cpu[_proposal_slot] = time.process_time()


def _initialize_proposal_worker(cancelled, cpu_slots, next_slot):
    global _proposal_cancel, _proposal_cpu, _proposal_slot
    _proposal_cancel, _proposal_cpu = cancelled, cpu_slots
    with next_slot.get_lock():
        _proposal_slot = next_slot.value
        next_slot.value += 1
    _sample_proposal_cpu()
    Finalize(None, _sample_proposal_cpu, exitpriority=0)


def _screen_refine_proposal_seed(payload):
    from baloot_platform.core.rules import RuleBook

    rules_data, seed, policy_id, policy, parameters, deadline, attempt_seconds = payload
    rules = RuleBook(rules_data)

    def cancelled():
        _sample_proposal_cpu()
        return _proposal_cancel.is_set() or time.perf_counter() >= deadline

    rows = []
    try:
        if cancelled():
            return seed, 0, rows
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return seed, 0, rows
        screened = screen_runtime_proposals_v1(rules, seed=seed, rollout_policy_id=policy_id,
            selection_policy=policy, maximum_nodes=parameters["maximumProofNodes"],
            maximum_seconds=min(attempt_seconds, remaining), cancel_check=cancelled,
            proposal_play_indices=parameters.get("proposalPlayIndices"),
            accepted_lesson_library=parameters.get("acceptedLessonLibrary"))
        for proposal in screened["proposals"]:
            if cancelled():
                break
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            refined = refine_runtime_proposal_v1(rules, proposal, selection_policy=policy,
                maximum_nodes=parameters["maximumProofNodes"],
                maximum_seconds=min(attempt_seconds, remaining),
                maximum_worlds=parameters["maximumHiddenWorlds"], cancel_check=cancelled)
            if refined["status"] == "priority_candidate":
                rows.append(refined)
        return seed, len(screened["proposals"]), rows
    finally:
        _sample_proposal_cpu()


def parallel_runtime_proposal_seeds(rules, execution, parameters, *, deadline,
                                    cancel_check, worker_cpu_callback):
    """Bound pending work and preserve seed order independently of completion order."""
    context = multiprocessing.get_context("spawn")
    parameters = dict(parameters)
    parameters["proposalPlayIndices"] = getattr(execution, "proposal_play_indices", None)
    parameters["acceptedLessonLibrary"] = getattr(execution, "accepted_lesson_library", None)
    workers = min(parameters["requestedShardWorkers"], execution.proposal_seed_count)
    cancelled = context.Event()
    cpu_slots, next_slot = context.Array("d", workers), context.Value("i", 0)
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=context,
        initializer=_initialize_proposal_worker, initargs=(cancelled, cpu_slots, next_slot))
    seeds = iter(range(parameters["seed"], parameters["seed"] + execution.proposal_seed_count))
    pending = deque()
    attempt_seconds = execution.proposal_maximum_seconds or parameters["maximumSeconds"]

    def submit():
        seed = next(seeds, None)
        if seed is not None:
            pending.append(pool.submit(_screen_refine_proposal_seed, (
                rules.data, seed, execution.early_play_policy_id, execution.video_interestingness,
                parameters, deadline, attempt_seconds)))

    try:
        if cancel_check() or time.perf_counter() >= deadline:
            return
        for _ in range(workers):
            submit()
        while pending:
            if cancel_check() or time.perf_counter() >= deadline:
                break
            try:
                row = pending[0].result(timeout=parameters.get("cancellationPollMilliseconds", 50) / 1000)
            except FutureTimeout:
                continue
            pending.popleft()
            if cancel_check() or time.perf_counter() >= deadline:
                break
            yield row
            submit()
    finally:
        cancelled.set()
        pool.shutdown(wait=True, cancel_futures=True)
        if worker_cpu_callback is not None:
            with cpu_slots.get_lock():
                worker_cpu_callback(sum(cpu_slots[:]))

from baloot_platform.challenge.trace import CompleteGameTraceV1
from baloot_platform.challenge.oracle import (
    ActualDealOracleVerifierV1,
    TacticalViabilityProbeV1,
)

from .contracts import (
    ConstructionPreScreenProfileV1,
    TacticalViabilityPreScreenProfileV1,
    canonical_sha256,
)


class _ProposalScreenStopped(Exception):
    pass


def _proposal_limits(maximum_nodes, maximum_seconds, cancel_check):
    if type(maximum_nodes) is not int or maximum_nodes < 1:
        raise ValueError("proposal maximum_nodes must be a positive integer")
    if type(maximum_seconds) not in (int, float) or not math.isfinite(maximum_seconds) or maximum_seconds <= 0:
        raise ValueError("proposal maximum_seconds must be finite and positive")
    deadline = time.monotonic() + maximum_seconds

    def check():
        if cancel_check is not None and cancel_check():
            raise _ProposalScreenStopped("cancelled")
        if time.monotonic() >= deadline:
            raise _ProposalScreenStopped("time_budget")
    return deadline, check


def _proposal_policy_score(rules, engine, state, policy_id, deadline, check, *,
                           preferred_completed_trick_winner=None):
    from baloot_platform.core.enums import Seat
    from .ortools_research_executor import _observer_safe_play_policy_artifact_v1
    from .ortools_information_set_compiler import InformationSetCompilationStopped
    from .scenario_contracts import ScenarioProofRequestV1

    check()
    legal = engine.legal_play_actions(state)
    normalized = SimpleNamespace(
        decision_kind="play", next_actor=state.current_player.name.lower(),
        state={"contract": state.contract.value,
               "trumpSuit": state.trump_suit.name.lower() if state.trump_suit is not None else None,
               "currentTrick": state.current_trick.to_dict(),
               "completedTricks": [trick.to_dict() for trick in state.completed_tricks],
               "projects": [project.to_dict() for project in state.projects],
               "hands": {seat.name.lower(): [card.to_dict() for card in state.hand(seat)] for seat in Seat}},
        legal_actions=[action.to_dict() for action in legal],
    )
    request = ScenarioProofRequestV1.build({
        "schemaVersion": 1, "mode": "bounded", "visibility": "actor_information_set",
        "strategyClass": "pure_deterministic", "claim": "winning_root_action_count",
        "targetTeam": state.current_player.team.name.lower(), "utility": "awarded_point_margin",
        "successThreshold": 0, "budgetProfile": "benchmark",
        "horizon": {"unit": "tricks", "value": 1},
        "boundaryEvaluator": "current_raw_point_margin_v1",
    })
    try:
        action, artifact = _observer_safe_play_policy_artifact_v1(
            rules, normalized, request, actual_action=None, policy_id=policy_id,
            preferred_completed_trick_winner=preferred_completed_trick_winner,
            maximum_seconds=min(1.0, max(0.001, deadline - time.monotonic())),
            cancel_check=lambda: (check() or False),
        )
    except InformationSetCompilationStopped as stopped:
        check()
        raise _ProposalScreenStopped("policy_incomplete:" + stopped.reason) from stopped
    check()
    scores = {row["action"]: row["value"]
              for row in artifact["proofModel"]["actionQuality"]["actionScores"]}
    if action not in scores or scores[action] != max(scores.values()):
        raise ValueError("proposal policy selection disagrees with its scores")
    return action, scores


def _proposal_root_values(engine, state, maximum_nodes, check, *, policy_score=None, root_action=None,
                          utility="awarded_point_margin"):
    """Perfect-information priority signal, never actor-visible proof."""
    if utility not in {"awarded_point_margin", "round_win"}:
        raise ValueError("unsupported proposal utility")
    target = int(state.current_player.team)
    nodes = 0

    def search(alpha, beta):
        nonlocal nodes
        check()
        nodes += 1
        if nodes > maximum_nodes:
            raise _ProposalScreenStopped("node_budget")
        if state.current_player is None:
            if utility == "round_win":
                return 1 if int(state.round_score.winner) == target else -1
            points = state.round_score.awarded_points
            return points[target] - points[1 - target]
        maximize = int(state.current_player.team) == target
        actions = engine.legal_play_actions(state)
        if maximize and policy_score is not None and len(actions) > 1:
            selected, _scores = policy_score(state)
            actions = [action for action in actions if action.code == selected]
            maximize = False
        value = -math.inf if maximize else math.inf
        for action in actions:
            token = engine.apply(state, action.card, ekka=action.ekka,
                                 verify_identity=False, emit_events=False)
            try:
                result = search(alpha, beta)
            finally:
                engine.undo(state, token)
            value = max(value, result) if maximize else min(value, result)
            if maximize:
                alpha = max(alpha, value)
            else:
                beta = min(beta, value)
            if alpha >= beta:
                break
        return value

    before = engine.state_hash(state)
    values = {}
    try:
        for action in engine.legal_play_actions(state):
            if root_action is not None and action.code != root_action:
                continue
            token = engine.apply(state, action.card, ekka=action.ekka,
                                 verify_identity=False, emit_events=False)
            try:
                values[action.code] = search(-math.inf, math.inf)
            finally:
                engine.undo(state, token)
    finally:
        if engine.state_hash(state) != before:
            raise RuntimeError("proposal search did not restore its input state")
    if not values:
        raise ValueError("proposal root action is not legal")
    return values, nodes


def _proposal_winner_preview(rules, worlds, action, baseline, maximum_nodes, check):
    """DD worst-case priority only: native information-set proof remains mandatory."""
    from baloot_platform.core.engine import BalootEngine
    from .result_authority import native_tie_rule_profile_v1

    result = {"status": "unknown", "action": action, "baseline": baseline,
              "utility": "round_win", "tieVariants": [], "roundFlip": None,
              "proposalOnly": True, "qualified": False}
    try:
        for tie in ("buyer_makes", "buyer_loses"):
            companion = native_tie_rule_profile_v1(rules, tie)
            engine = BalootEngine(companion)
            values = {action: 1, baseline: 1}
            for world in worlds:
                for selected in values:
                    roots, _ = _proposal_root_values(engine, world, maximum_nodes, check,
                        root_action=selected, utility="round_win")
                    values[selected] = min(values[selected], roots[selected])
            result["tieVariants"].append({"tieRule": tie, "rulesHash": companion.sha256,
                                          "rootValues": values})
        result.update(status="complete", roundFlip=all(
            row["rootValues"][action] == 1 and row["rootValues"][baseline] == -1
            for row in result["tieVariants"]))
    except _ProposalScreenStopped as stopped:
        result["reason"] = str(stopped)
    return result


def _proposal_opening(rules, seed, *, dealer=None):
    from baloot_platform.core.engine import BalootEngine
    from baloot_platform.core.enums import Seat
    engine = BalootEngine(rules)
    state = engine.create_round(seed=seed, dealer=Seat(seed % 4) if dealer is None else Seat.parse(dealer))
    engine.shuffle_and_initial_deal(state, cut_position=(seed * 17 % 31) + 1)
    if not engine.bid_and_complete_deal(state):
        return engine, state, None
    engine.prepare_projects(state)
    prefix = {"schemaVersion": 1, "deckOrder": list(state.shuffle.deck_order),
              "dealer": state.dealer.name.lower(), "matchScore": list(state.match_score),
              "roundNumber": state.round_number,
              "auctionActions": [{"actor": bid.seat.name.lower(), "action": bid.action,
                  **({"trumpSuit": bid.trump.name.lower()} if bid.trump is not None else {})}
                  for bid in state.bid_history],
              "escalationActions": [], "projectActions": [], "projectDeclarations": [],
              "playActions": [], "stateAssertions": {}, "projectAssertions": []}
    if rules.project_domain.declaration_mode != "optional":
        prefix.pop("projectActions")
    return engine, state, prefix


def screen_runtime_proposals_v1(rules, *, seed, rollout_policy_id, selection_policy,
                                maximum_nodes, maximum_seconds, cancel_check=None,
                                accepted_lesson_library=None, proposal_play_indices=None):
    """Screen one seeded round; the caller owns batching, storage and native admission."""
    from baloot_platform.core.models import LegalPlayActionV1
    from .scenario_contracts import ExactTracePrefixV1, build_scenario_video_interestingness_v1
    if type(seed) is not int or seed < 0 or not isinstance(rollout_policy_id, str) or not rollout_policy_id:
        raise ValueError("proposal seed and rollout policy are required")
    policy = build_scenario_video_interestingness_v1(copy.deepcopy(dict(selection_policy)))
    if proposal_play_indices is not None and (
            not isinstance(proposal_play_indices, (tuple, list)) or not proposal_play_indices
            or any(type(index) is not int or not 0 <= index <= 31 for index in proposal_play_indices)
            or len(set(proposal_play_indices)) != len(proposal_play_indices)):
        raise ValueError("proposal play indices must be unique integers in 0..31")
    if accepted_lesson_library is not None:
        from .lesson_prefilter import lesson_is_repeated_v1
        lesson_is_repeated_v1({"family": None, "deductions": []}, accepted_lesson_library)
    if policy["schemaVersion"] != 4:
        raise ValueError("runtime proposals require schema-4 selection policy")
    deadline, check = _proposal_limits(maximum_nodes, maximum_seconds, cancel_check)
    result = {"schemaVersion": 1, "stage": "runtime_proposal", "seed": seed,
              "rulesHash": rules.sha256, "policyHash": canonical_sha256(policy),
              "rolloutPolicyId": rollout_policy_id, "proposalOnly": True,
              "qualified": False, "status": "complete", "decisions": [], "proposals": []}
    try:
        check()
        engine, state, prefix = _proposal_opening(rules, seed)
        check()
        if prefix is None:
            result["status"] = "all_pass"
            result["sha256"] = canonical_sha256(result)
            return result
        snapshots = []
        for index in range(32):
            check()
            declaration = state._project_declaration_state
            if declaration is not None and not declaration.complete and declaration.current_actor == state.current_player:
                action = rules.project_domain.legal_declaration_actions(declaration, trump_suit=state.trump_suit)[0]
                engine.apply_project_declaration_action(state, action)
                prefix["projectActions"].append(action.to_dict())
            legal = engine.legal_play_actions(state)
            prefix["projectDeclarations"] = [
                {key: event.to_dict()[key] for key in ("owner", "kind", "value", "highRank", "declarationOrder")}
                for event in state.events if event.to_dict().get("type") == "project_declared"
            ]
            selected = (legal[0].code if len(legal) == 1 else
                        _proposal_policy_score(rules, engine, state, rollout_policy_id, deadline, check)[0])
            row = {"playIndex": index, "status": "forced" if len(legal) == 1 else "pending"}
            result["decisions"].append(row)
            if len(legal) > 1:
                snapshots.append((row, copy.deepcopy(state), copy.deepcopy(prefix)))
            action = LegalPlayActionV1.from_code(selected)
            actor = state.current_player
            engine.play_card(state, action.card, ekka=action.ekka)
            prefix["playActions"].append({"actor": actor.name.lower(), "card": action.card.code, "ekka": action.ekka})
        result["projectedFinalMargin"] = abs(state.round_score.awarded_points[0] - state.round_score.awarded_points[1])
        # Finish the policy line before spending search budget, starting with the cheapest tails.
        for row, state, prefix in reversed(snapshots):
            check()
            index = row["playIndex"]
            if proposal_play_indices is not None and index not in proposal_play_indices:
                row["status"] = "outside_requested_indices"
                continue
            baseline, scores = _proposal_policy_score(
                rules, engine, state, policy["baselinePolicyId"], deadline, check)
            row["status"] = "baseline_all_tied"
            if len(set(scores.values())) > 1:
                try:
                    values, nodes = _proposal_root_values(engine, state, maximum_nodes, check)
                except _ProposalScreenStopped as stopped:
                    row["status"] = str(stopped)
                    if str(stopped) != "node_budget":
                        raise
                else:
                    best = max(values.values())
                    optimal = [action for action, value in values.items() if value == best]
                    row.update(status="optimal_tied", nodes=nodes)
                    if len(optimal) == 1:
                        action = optimal[0]
                        gap = best - max(value for other, value in values.items() if other != action)
                        regret = best - values[baseline]
                        row["status"] = "priority_miss"
                        if (scores[baseline] > scores[action] and gap >= policy["minimumNextBestRegret"]
                                and regret >= policy["minimumBaselineRegret"]):
                            if accepted_lesson_library is not None:
                                from .lesson_prefilter import classify_public_lesson_v1, lesson_is_repeated_v1
                                hypothesis = classify_public_lesson_v1(rules=rules,
                                    actor=state.current_player.name.lower(),
                                    own_cards=[card.code for card in state.hand(state.current_player)],
                                    completed_tricks=[trick.to_dict() for trick in state.completed_tricks],
                                    current_trick=state.current_trick.to_dict(),
                                    contract=state.contract.value,
                                    trump=state.trump_suit.name.lower() if state.trump_suit is not None else None,
                                    action=LegalPlayActionV1.from_code(action).card.code,
                                    baseline=LegalPlayActionV1.from_code(baseline).card.code)
                                row["lessonHypothesis"] = hypothesis
                                if lesson_is_repeated_v1(hypothesis, accepted_lesson_library):
                                    row["status"] = "known_lesson_family"
                                    continue
                            typed = ExactTracePrefixV1.build(copy.deepcopy(prefix))
                            proposal = {"seed": seed, "playIndex": index, "actor": state.current_player.name.lower(),
                                "action": action, "baseline": baseline, "baselineScores": scores,
                                "ddValues": values, "ddGap": gap, "ddRegret": regret,
                                "prefix": typed.to_dict(), "prefixHash": typed.sha256,
                                "stateHash": engine.state_hash(state), "rulesHash": rules.sha256,
                                "policyHash": result["policyHash"], "rolloutPolicyId": rollout_policy_id,
                                "stage": "runtime_proposal", "status": "proposed",
                                "proposalOnly": True, "qualified": False}
                            if "lessonHypothesis" in row:
                                proposal["lessonHypothesis"] = row["lessonHypothesis"]
                            proposal["sha256"] = canonical_sha256(proposal)
                            result["proposals"].append(proposal)
                            row["status"] = "proposed"
    except _ProposalScreenStopped as stopped:
        result["status"] = str(stopped)
    result["sha256"] = canonical_sha256(result)
    return result


def preflight_play_world_count_v1(rules, exact, *, maximum_worlds,
                                  maximum_attempts, maximum_seconds, cancel_check=None):
    """Count native-compatible worlds without constructing continuation graphs.

    Only cap+1 compatible worlds establish over-cap; a budget stop is unknown.
    This cost screen is not tactical qualification or impossibility evidence.
    """
    from .ortools_information_set_compiler import (
        InformationSetCompilationStopped, OrToolsActorInformationSetCompilerV1,
    )
    if type(maximum_worlds) is not int or maximum_worlds < 1:
        raise ValueError("maximum_worlds must be a positive integer")
    if type(maximum_attempts) is not int or maximum_attempts < 1:
        raise ValueError("maximum_attempts must be a positive integer")
    deadline, check = _proposal_limits(maximum_attempts, maximum_seconds, cancel_check)
    if exact.decision_kind != "play":
        raise ValueError("world count preflight requires a normalized play state")
    result = {"schemaVersion": 1, "stage": "play_world_count_preflight",
              "rulesHash": rules.sha256, "exactStateId": exact.exact_state_id,
              "exactHash": exact.sha256, "maximumWorlds": maximum_worlds,
              "maximumAttempts": maximum_attempts, "maximumSeconds": maximum_seconds,
              "status": "incomplete", "worldCount": None,
              "worldCountLowerBound": 0, "coverageComplete": False,
              "proposalOnly": True, "qualified": False}
    try:
        check()
        compiler = OrToolsActorInformationSetCompilerV1(rules, native_cp_sat=True)
        worlds, receipt = compiler._enumerate_worlds(
            exact, maximum_worlds=maximum_worlds, maximum_attempts=maximum_attempts,
            deadline=deadline, cancel_check=cancel_check)
        result.update(status="within_cap", worldCount=len(worlds),
                      worldCountLowerBound=len(worlds), coverageComplete=True, receipt=receipt)
    except InformationSetCompilationStopped as stopped:
        receipt = stopped.partial_receipt
        count = receipt.get("evidence", {}).get("acceptedWorlds", 0)
        result.update(reason=stopped.reason, receipt=receipt, worldCountLowerBound=count)
        if stopped.reason == "maximum_hidden_worlds" and count > maximum_worlds:
            result["status"] = "over_cap"
    except _ProposalScreenStopped as stopped:
        result["reason"] = str(stopped)
    result["sha256"] = canonical_sha256(result)
    return result


def refine_runtime_proposal_v1(rules, proposal, *, selection_policy, maximum_nodes,
                               maximum_seconds, maximum_worlds, cancel_check=None):
    """Rank hidden-world heuristic bounds; every returned row still requires native proof."""
    from baloot_platform.analysis.hidden_deals import generate_hidden_deals
    from baloot_platform.analysis.models import InformationState
    from baloot_platform.core.cards import Card
    from .scenario_contracts import ExactTracePrefixV1, build_scenario_video_interestingness_v1
    row = copy.deepcopy(dict(proposal))
    digest = row.pop("sha256", None)
    if digest != canonical_sha256(row):
        raise ValueError("proposal hash mismatch")
    policy = build_scenario_video_interestingness_v1(copy.deepcopy(dict(selection_policy)))
    if policy["schemaVersion"] != 4 or row["rulesHash"] != rules.sha256 or row["policyHash"] != canonical_sha256(policy):
        raise ValueError("proposal does not own the frozen rules and selection policy")
    if row.get("proposalOnly") is not True or row.get("qualified") is not False:
        raise ValueError("refinement accepts proposal-only evidence")
    if type(maximum_worlds) is not int or not 1 <= maximum_worlds <= 10000:
        raise ValueError("proposal maximum_worlds must be an integer in 1..10000")
    typed = ExactTracePrefixV1.build(row["prefix"])
    if typed.sha256 != row["prefixHash"] or len(typed.play_actions) != row["playIndex"] or not 0 <= row["playIndex"] <= 31:
        raise ValueError("proposal prefix identity mismatch")
    deadline, check = _proposal_limits(maximum_nodes, maximum_seconds, cancel_check)
    result = {**row, "stage": "runtime_refine", "status": "incomplete"}
    try:
        check()
        engine, state, opening = _proposal_opening(rules, row["seed"])
        if opening is None:
            raise ValueError("proposal opening passed out")
        if typed.project_actions_present:
            opening["projectActions"] = list(typed.project_actions)
        opening["playActions"] = list(typed.play_actions)
        opening["projectDeclarations"] = list(typed.project_declarations)
        if ExactTracePrefixV1.build(opening).sha256 != typed.sha256:
            raise ValueError("proposal opening does not match its seeded frozen rules")
        project_index = 0
        # The focal declaration is recorded before the focal play, which is not in the prefix.
        for action in (*typed.play_actions, None):
            check()
            declaration = state._project_declaration_state
            if declaration is not None and not declaration.complete and declaration.current_actor == state.current_player:
                if project_index >= len(typed.project_actions):
                    raise ValueError("proposal project action missing")
                options = rules.project_domain.legal_declaration_actions(declaration, trump_suit=state.trump_suit)
                chosen = next((option for option in options if option.to_dict() == typed.project_actions[project_index]), None)
                if chosen is None:
                    raise ValueError("proposal project action is not legal")
                engine.apply_project_declaration_action(state, chosen)
                project_index += 1
            if action is None:
                break
            if action["actor"] != state.current_player.name.lower():
                raise ValueError("proposal play actor mismatch")
            engine.play_card(state, Card.parse(action["card"]), ekka=action.get("ekka", False))
        if project_index != len(typed.project_actions) or engine.state_hash(state) != row["stateHash"]:
            raise ValueError("proposal restored state mismatch")
        worlds = generate_hidden_deals(
            InformationState.from_complete(state, state.current_player), engine,
            requested_samples=maximum_worlds, exact_allocation_limit=maximum_worlds,
            maximum_attempts=maximum_worlds, seed=row["seed"], cancel_check=lambda: (check() or False))
        check()
        result.update(worlds=len(worlds.states), worldMethod=worlds.method,
                      worldCoverageComplete=worlds.coverage_complete)
        if not worlds.coverage_complete:
            result["status"] = "worlds_incomplete"
        elif len(worlds.states) < policy["minimumConsistentWorlds"]:
            result["status"] = "single_world"
        else:
            uppers = {action: math.inf for action in row["ddValues"]}
            for world in worlds.states:
                values, _nodes = _proposal_root_values(engine, world, maximum_nodes, check)
                if set(values) != set(uppers):
                    raise ValueError("proposal hidden worlds disagree on visible legal actions")
                for action in uppers:
                    uppers[action] = min(uppers[action], values[action])
            selected = row["action"]
            gap = uppers[selected] - max(value for action, value in uppers.items() if action != selected)
            result.update(worldMinDdUpper=uppers, worldMinDdGap=gap, status="priority_gap_lost")
            if gap >= policy["minimumNextBestRegret"]:
                def score(world):
                    return _proposal_policy_score(rules, engine, world, policy["baselinePolicyId"], deadline, check)
                floor = min(_proposal_root_values(engine, world, maximum_nodes, check,
                            policy_score=score, root_action=selected)[0][selected] for world in worlds.states)
                sandwich = floor - max(value for action, value in uppers.items() if action != selected)
                baseline_regret = floor - uppers[row["baseline"]]
                result.update(fixedVisiblePolicyFloor=floor, sandwichGap=sandwich,
                              robustBaselineRegret=baseline_regret,
                              status="priority_candidate" if sandwich >= policy["minimumNextBestRegret"]
                              and baseline_regret >= policy["minimumBaselineRegret"] else "deferred")
                if result["status"] == "priority_candidate":
                    result["winnerPreview"] = _proposal_winner_preview(
                        rules, worlds.states, selected, row["baseline"], maximum_nodes, check)
    except _ProposalScreenStopped as stopped:
        result["status"] = str(stopped)
    result["sha256"] = canonical_sha256(result)
    return result


@dataclass(frozen=True, slots=True)
class ConstructionPreScreenEvidenceV1:
    profile_hash: str
    trace_hash: str
    analyzed_decisions: tuple[Mapping[str, Any], ...]
    rejected: bool
    rejection_reasons: tuple[str, ...]
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "kind": "construction_actor_safe_pre_screen",
            "proofBoundary": (
                "cheap rejection evidence only; passing is not tactical or "
                "qualification proof"
            ),
            "profileHash": self.profile_hash,
            "traceHash": self.trace_hash,
            "analyzedDecisions": [dict(item) for item in self.analyzed_decisions],
            "rejected": self.rejected,
            "rejectionReasons": list(self.rejection_reasons),
            "sha256": self.sha256,
        }


class ConstructionPreScreenV1:
    def __init__(self, profile: ConstructionPreScreenProfileV1) -> None:
        self.profile = profile

    def _decision_ids(self, trace: CompleteGameTraceV1) -> tuple[str, ...]:
        bids = [
            item for item in trace.decision_ledger.records
            if item.decision_kind == "bid" and not item.forced
        ]
        plays = [
            item for item in trace.decision_ledger.records
            if item.decision_kind == "play" and not item.forced
        ]
        first_by_seat = {}
        for item in plays:
            first_by_seat.setdefault(item.acting_seat, item)
        selected = list(first_by_seat.values())
        remaining = sorted(
            (item for item in plays if item not in selected),
            key=lambda item: (-len(item.legal_actions), item.ordinal),
        )
        selected.extend(
            remaining[: max(0, self.profile.maximum_play_decisions - len(selected))]
        )
        selected = sorted(selected[: self.profile.maximum_play_decisions], key=lambda item: item.ordinal)
        return tuple(item.decision_id for item in [*bids, *selected])

    def evaluate(
        self,
        trace: CompleteGameTraceV1,
        analyzer: Any,
        *,
        cancel_check=None,
    ) -> ConstructionPreScreenEvidenceV1:
        if not self.profile.enabled:
            raise ValueError("construction pre-screen is disabled")
        analyses = analyzer.analyze_decisions(
            trace,
            self._decision_ids(trace),
            seed_namespace=f"construction_pre_screen:{self.profile.sha256}",
            cancel_check=cancel_check,
        )
        rows = tuple(item.audit_mapping() for item in analyses)
        reasons: set[str] = set()
        for item in analyses:
            if item.confidence < self.profile.minimum_confidence:
                continue
            maximum_regret = (
                self.profile.maximum_bid_regret
                if item.decision_kind == "bid"
                else self.profile.maximum_play_regret
            )
            if item.expected_regret > maximum_regret:
                reasons.add(f"obvious_{item.decision_kind}_regret")
            if item.probability_drop > self.profile.maximum_probability_drop:
                reasons.add(f"obvious_{item.decision_kind}_probability_drop")
            if (
                self.profile.reject_catastrophic_crossing
                and item.catastrophic_crossing
            ):
                reasons.add(f"{item.decision_kind}_catastrophic_crossing")
        core = {
            "schemaVersion": 1,
            "profileHash": self.profile.sha256,
            "traceHash": trace.sha256,
            "analysisHashes": [item.sha256 for item in analyses],
            "rejected": bool(reasons),
            "rejectionReasons": sorted(reasons),
        }
        return ConstructionPreScreenEvidenceV1(
            self.profile.sha256,
            trace.sha256,
            rows,
            bool(reasons),
            tuple(sorted(reasons)),
            canonical_sha256(core),
        )


@dataclass(frozen=True, slots=True)
class TacticalViabilityPreScreenEvidenceV1:
    profile_hash: str
    trace_hash: str
    required_tactics: tuple[str, ...]
    found_tactics: tuple[str, ...]
    probe: TacticalViabilityProbeV1
    rejected: bool
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "kind": "exact_tactical_viability_pre_screen",
            "proofBoundary": (
                "actual-deal rejection evidence only; passing does not replace "
                "actor-safe analysis or qualification"
            ),
            "profileHash": self.profile_hash,
            "traceHash": self.trace_hash,
            "requiredTactics": list(self.required_tactics),
            "foundTactics": list(self.found_tactics),
            "probe": self.probe.to_dict(),
            "rejected": self.rejected,
            "sha256": self.sha256,
        }


class TacticalViabilityPreScreenV1:
    def __init__(
        self,
        profile: TacticalViabilityPreScreenProfileV1,
        verifier: ActualDealOracleVerifierV1,
    ) -> None:
        self.profile = profile
        self.verifier = verifier

    def evaluate(
        self,
        trace: CompleteGameTraceV1,
        *,
        required_tactics: tuple[str, ...],
        cancel_check=None,
    ) -> TacticalViabilityPreScreenEvidenceV1:
        if not self.profile.enabled:
            raise ValueError("tactical viability pre-screen is disabled")
        required = tuple(sorted(set(required_tactics)))
        probe = self.verifier.probe_verified_tactical_play(
            trace,
            maximum_decisions=self.profile.maximum_decisions,
            required_tactics=required,
            cancel_check=cancel_check,
        )
        found = tuple(sorted({
            evidence.kind.value
            for decision in probe.scanned_decisions
            for evidence in decision.evidence
        }))
        missing = bool(required) and not set(required).intersection(found)
        if not self.profile.required_tactics_only:
            missing = not found
        rejected = bool(
            self.profile.reject_when_exact_absent
            and probe.scan_complete
            and missing
        )
        core = {
            "schemaVersion": 1,
            "profileHash": self.profile.sha256,
            "traceHash": trace.sha256,
            "requiredTactics": list(required),
            "foundTactics": list(found),
            "probeHash": probe.sha256,
            "rejected": rejected,
        }
        return TacticalViabilityPreScreenEvidenceV1(
            self.profile.sha256,
            trace.sha256,
            required,
            found,
            probe,
            rejected,
            canonical_sha256(core),
        )
