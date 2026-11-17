"""Policy-coupled native construction; no replay is admitted on a target miss."""
from __future__ import annotations

import copy
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import yaml

from baloot_platform.analysis.bidding import analyze_bid, legal_bid_choices
from baloot_platform.analysis.double_dummy import DoubleDummySolver
from baloot_platform.analysis.hidden_deals import generate_hidden_deals
from baloot_platform.analysis.ismcts_v8 import analyze_information_set_mcts
from baloot_platform.analysis.models import InformationState
from baloot_platform.analysis.search import heuristic_rollout_result
from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.cards import Card, cards_to_mask
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import GamePhase, Seat, Rank, Suit
from baloot_platform.core.models import LegalPlayActionV1, RoundState, Trick
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.shuffle import SplitMix64

from .construction_trace import construction_trace_prefix_v1
from .ortools_escalation_stage import OrToolsEscalationStageV1
from .ortools_exact_state import OrToolsExactStateNormalizerV1, ExactNormalizationStopped
from .ortools_information_set_compiler import InformationSetCompilationStopped
from .ortools_play_tail import PlayTailSolveStopped
from .ortools_research_executor import GenericCompleteTraceCompilerV1, _complete_state_quality_artifact_v1
from .outcome_spec import CompiledOutcomeSpec, _integer, _object
from .scenario_contracts import ExactTracePrefixV1, ResearchScenarioV1, ScenarioProofBudgetV1, ScenarioProofRequestV1, ScenarioQualityExecutionBudgetV1, canonical_sha256


class OutcomePolicyIncomplete(RuntimeError):
    def __init__(self, reason, evidence=None):
        self.reason, self.evidence = reason, evidence
        super().__init__(reason)


class OutcomePolicyDeviation(Exception):
    def __init__(self, index, action, scope_hash):
        self.index, self.action, self.scope_hash = index, action, scope_hash


def load_outcome_policy(path: str | Path) -> dict[str, Any]:
    return validate_outcome_policy(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def validate_outcome_policy(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = _object(value, {"schemaVersion", "id", "informed", "optimal", "construction", "opportunityOrder", "metadata", "cardPolicy"}, "outcome policy")
    if raw.get("schemaVersion") != 1 or type(raw["schemaVersion"]) is not int or not raw.get("id"):
        raise ValueError("outcome policy requires version 1 and an identity")
    informed = _object(raw.get("informed"), {"particles", "iterations", "maximumSamplingAttempts", "bidSamples", "deliberateMistakes", "bidTieBreak"}, "informed policy")
    if informed.get("deliberateMistakes") is not False:
        raise ValueError("informed policy must disable deliberate mistakes")
    for key in ("particles", "iterations", "maximumSamplingAttempts", "bidSamples"):
        _integer(informed.get(key), 2, 80000, key)
    if (type(informed.get("bidTieBreak")) is not list or len(informed["bidTieBreak"]) != 4
            or set(informed["bidTieBreak"]) != {"sun", "hokum", "ashkal", "pass"}):
        raise ValueError("bidTieBreak must list every action once")
    construction = _object(raw.get("construction"), {"sunOpeningAces", "hokumOpeningHonors"}, "construction")
    _integer(construction.get("sunOpeningAces"), 0, 4, "sunOpeningAces")
    if construction.get("hokumOpeningHonors") != ["jack", "nine"]:
        raise ValueError("this policy revision uses Jack/Nine trump construction hints")
    optimal = _object(raw.get("optimal"), {"proofBudget", "executionBudget", "maximumConflicts", "onIncomplete"}, "optimal policy")
    if optimal.get("onIncomplete") != "reject":
        raise ValueError("optimal policy cannot fall back or relabel incomplete proofs")
    budget = ScenarioProofBudgetV1.build(raw["id"], optimal.get("proofBudget", {}))
    if (budget.maximum_wall_seconds > 5 or budget.maximum_cpu_seconds > 5
            or budget.maximum_nodes > 50000 or budget.maximum_depth > 64
            or budget.maximum_workers != 1 or budget.maximum_hidden_worlds != 1
            or budget.native_observed is None or budget.maximum_cache_bytes > 8*1024**3):
        raise ValueError("optimal proof budget exceeds the frozen single-worker decision limits")
    _integer(optimal.get("maximumConflicts"), 1, 250000, "maximumConflicts")
    execution = _object(optimal.get("executionBudget"), {"schemaVersion", "maximumWallSeconds",
        "maximumEvidenceBytes", "maximumEscalationDecisions"}, "native execution budget")
    execution = ScenarioQualityExecutionBudgetV1.build(execution)
    if execution.maximum_wall_seconds > budget.maximum_wall_seconds:
        raise ValueError("native execution must fit the decision budget")
    if (type(raw.get("opportunityOrder")) is not list or len(raw["opportunityOrder"]) != 4
            or set(raw["opportunityOrder"]) != {s.name.lower() for s in Seat}):
        raise ValueError("opportunityOrder must name all four seats once")
    if "cardPolicy" in raw:
        card = _object(raw["cardPolicy"], {"method", "particles", "maximumSamplingAttempts", "maximumWallSeconds",
            "maximumCpuSeconds", "maximumNodes", "maximumCacheEntries", "samplingSeed", "tieBreak", "maximumThreads"}, "DD card policy")
        batch = card.get("method") == "native_dd_pimc_batch_v1"
        if card.get("method") not in {"native_dd_pimc_v1", "native_dd_pimc_batch_v1"} or card.get("tieBreak") != "stable_action_id":
            raise ValueError("DD card policy requires its frozen method and action order")
        if batch:
            _integer(card.get("maximumThreads"), 1, 4, "maximumThreads")
        elif "maximumThreads" in card:
            raise ValueError("serial DD profiles do not admit a thread-count override")
        for key, lower, upper in (("particles", 2, 256), ("maximumSamplingAttempts", 2, 80000),
                ("maximumNodes", 1, 100000000 if batch else 10000000),
                ("maximumCacheEntries", 1 if batch else 0, 1000000), ("samplingSeed", 0, 2**32-1)):
            _integer(card.get(key), lower, upper, key)
        for key in ("maximumWallSeconds", "maximumCpuSeconds"):
            ceiling = (60 if key == "maximumWallSeconds" else 240) if batch else 5
            if type(card.get(key)) not in {float, int} or not 0 < card[key] <= ceiling:
                raise ValueError("DD card policy exceeds its method's decision cap")
        if batch and card["maximumCpuSeconds"] > card["maximumWallSeconds"]*card["maximumThreads"]:
            raise ValueError("cohort CPU budget exceeds its thread/wall envelope")
        if card["maximumSamplingAttempts"] < card["particles"]:
            raise ValueError("DD sampler attempts must cover the requested cohort")
    canonical_sha256(raw)
    return raw


def _prefix(deck, dealer, bids, escalations=(), plays=(), project_actions=None):
    return ExactTracePrefixV1.build(dict(schemaVersion=1, deckOrder=list(deck), dealer=dealer,
        roundNumber=1, matchScore=[0, 0], auctionActions=list(bids), escalationActions=list(escalations),
        **({"projectActions": list(project_actions)} if project_actions is not None else {}),
        projectDeclarations=[], playActions=list(plays), stateAssertions={}))


def _bid_observation(state, machine, actor):
    return dict(rulesHash=state.rules_hash, observer=actor.name.lower(),
        observerHand=[card.code for card in state.hand(actor)], dealer=state.dealer.name.lower(),
        purchaseCard=state.purchase_card.code, matchScore=list(state.match_score),
        bids=[row.to_dict() for row in machine.records])


def _informed_bid(observation, machine, rules, settings, seed, stopped):
    """The analyzer receives an own-hand shell, never the constructed hidden deal."""
    actor = Seat.parse(observation["observer"])
    shell = RoundState(rules.sha256, seed, 1, Seat.parse(observation["dealer"]),
                       match_score=list(observation["matchScore"]))
    shell.hands[int(actor)] = cards_to_mask(Card.parse(card) for card in observation["observerHand"])
    shell.initial_five = list(shell.hands)
    shell.purchase_card = Card.parse(observation["purchaseCard"])
    shell.phase = GamePhase.INITIAL_DEALT
    analysis = analyze_bid(shell, copy.deepcopy(machine), actor, rules,
        requested_samples=settings["bidSamples"], seed=seed, cancel_check=stopped)
    best = max((row.expected_match_margin, row.team_win_probability, row.best_share) for row in analysis.choices)
    tied = [row.choice for row in analysis.choices
            if (row.expected_match_margin, row.team_win_probability, row.best_share) == best]
    order = settings["bidTieBreak"]
    choice = min(tied, key=lambda item: (order.index(item.action), item.id))
    return choice, analysis.to_dict()


def _preplay_information(state, actor):
    shell = state.clone()
    shell.projects = []  # No project categories/ranks are public before trick 1.
    shell.phase = GamePhase.PLAYING
    shell.current_player = shell.first_leader
    shell.current_trick = Trick(0, shell.first_leader)
    return InformationState.from_complete(shell, actor)


class OutcomePolicyDriver:
    def __init__(self, rules, compiled, policy, *, seed, deadline, refinement="first_deviation", maximum_threads=None):
        self.rules, self.compiled, self.policy = rules, compiled, policy
        self.engine = BalootEngine(rules)
        self.seed, self.deadline = seed, deadline
        self.receipts, self.events = [], []
        self.normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
        self.deck = None
        self.final_state = None
        self.dd = None
        self.informed_dd = {}
        self.card_cache = {}
        self.proposals, self.cuts = 0, []
        self.auction_proposals, self.auction_cuts, self.bid_cache = 0, [], {}
        if maximum_threads is not None:
            _integer(maximum_threads, 1, 4, "maximum_threads")
        self.maximum_threads = maximum_threads
        if refinement not in ("first_deviation", "forward_rollout"):
            raise ValueError("unsupported outcome refinement strategy")
        self.refinement, self.forward_probes, self.first_difference = refinement, 0, None

    @property
    def spec_fixed_contract(self):
        # An explicit team alone delegates a legal auction to construction, not informed bid quality.
        return bool(set(self.compiled.source) & {"mode", "trumpSuit", "buyer", "team", "auction"})

    def stopped(self):
        return time.monotonic() >= self.deadline

    def close(self):
        for solver in [self.dd, *self.informed_dd.values()]:
            if solver is not None and (close := getattr(solver, "close", None)) is not None:
                close()

    def remaining(self):
        if self.stopped():
            raise OutcomePolicyIncomplete("candidate_time_budget")
        return self.deadline - time.monotonic()

    def record(self, actor, kind, scope, action, evidence, *, guarantee):
        bid_policy = "spec_fixed" if self.spec_fixed_contract else "informed"
        core = dict(actor=actor.name.lower(), decisionKind=kind,
            policy=(bid_policy if kind in {"bid", "escalation"} else dict(self.compiled.seat_policies)[actor.name.lower()]),
            policyProfileHash=canonical_sha256(self.policy), limits=copy.deepcopy(self.policy),
            inputScopeHash=canonical_sha256(scope), action=action, guarantee=guarantee,
            evidence=evidence)
        if kind in {"bid", "escalation"}:
            core["bid_policy"] = bid_policy
        self.receipts.append({**core, "sha256": canonical_sha256(core)})

    def exact(self, prefix, kind, actor):
        limits = self.policy["optimal"]
        budget = ScenarioProofBudgetV1.build(self.policy["id"], limits["proofBudget"])
        deadline = min(self.deadline, time.monotonic()+budget.maximum_wall_seconds)
        cpu_deadline = time.process_time()+budget.maximum_cpu_seconds
        decision = {"kind": kind}
        if kind == "escalation":
            decision["selector"] = {"actingSeat": actor.name.lower()}
        def stopped():
            return time.monotonic() >= deadline or time.process_time() >= cpu_deadline
        normalized = self.normalizer.normalize(prefix, decision_point=decision,
            maximum_seconds=max(1e-6, deadline-time.monotonic()), cancel_check=stopped,
            **(dict(recorded_escalation_opportunities=self.events, opportunity_order=self.policy["opportunityOrder"])
               if kind == "escalation" else {}))
        if normalized.next_actor != actor.name.lower():
            raise AssertionError("native policy actor differs from the runtime actor")
        request = ScenarioProofRequestV1.build(dict(schemaVersion=1, mode="exact", visibility="complete_state",
            claim="winning_root_action_count", targetTeam=actor.team.name.lower(), utility="awarded_point_margin",
            successThreshold=0, budgetProfile=self.policy["id"]))
        artifact = _complete_state_quality_artifact_v1(self.rules, normalized, prefix, decision, request,
            maximum_nodes=budget.maximum_nodes, maximum_depth=budget.maximum_depth,
            maximum_seconds=max(1e-6, deadline-time.monotonic()), cancel_check=stopped,
            budget=budget, maximum_conflicts=limits["maximumConflicts"],
            execution_budget=ScenarioQualityExecutionBudgetV1.build(limits["executionBudget"]),
            opportunity_order=self.policy["opportunityOrder"] if kind in {"bid", "escalation"} else None,
            escalation_events=self.events if kind == "escalation" else None)
        proof = artifact["proof"]
        if stopped():
            raise OutcomePolicyIncomplete("optimal_decision_budget", proof)
        if proof["completionStatus"] != "exact_complete" or proof["solverStatus"] != "optimal":
            raise OutcomePolicyIncomplete("optimal_proof_incomplete", proof)
        return proof["selectedAction"], artifact

    def _informed_escalation(self, state, actor, choices):
        info = _preplay_information(state, actor)
        seed = int(canonical_sha256(dict(information=info.to_dict(),
            profile=canonical_sha256(self.policy),
            seed=self.policy.get("cardPolicy", {}).get("samplingSeed", 0)))[:16], 16)
        settings = self.policy["informed"]
        cohort = generate_hidden_deals(info, self.engine, requested_samples=settings["particles"],
            exact_allocation_limit=1, maximum_attempts=settings["maximumSamplingAttempts"], seed=seed,
            cancel_check=self.stopped)
        if not cohort.states:
            raise OutcomePolicyIncomplete("informed_sampling_incomplete")
        values = {code: [] for code in choices}
        for sample in cohort.states:
            for code in choices:
                self.remaining()
                branch = sample.clone()
                branch.phase = GamePhase.FINAL_DEALT
                branch.projects = []
                branch.current_trick, branch.current_player = None, None
                if code != self.engine.escalation_domain.closure_action:
                    action, _, mode = code.partition(":")
                    self.engine.apply_escalation(branch, seat=actor, action=action, mode=mode or None)
                self.engine.prepare_projects(branch)
                # Same cohort and rollout seed for every alternative; no hidden actual cards.
                outcome = heuristic_rollout_result(branch, self.engine, actor.team, seed)
                values[code].append(outcome.own_points-outcome.opponent_points)
        means = {code: sum(rows)/len(rows) for code, rows in values.items()}
        return min(choices, key=lambda code: (-means[code], code)), info.to_dict(), dict(
            method="paired_observer_safe_escalation_rollout_v1", samplingSeed=seed, values=values, means=means)

    def choose_bid(self, state, machine, expected):
        actor, choices = machine.actor, legal_bid_choices(machine)
        observation = _bid_observation(state, machine, actor)
        if self.spec_fixed_contract:
            choice = next((item for item in choices if item.action == expected.get("action")
                and (item.trump_suit.name.lower() if item.trump_suit is not None else None) == expected.get("trumpSuit")
                and expected.get("actor", actor.name.lower()) == actor.name.lower()), None)
            if choice is None:
                raise OutcomePolicyIncomplete("spec_fixed_auction_illegal")
            return choice, observation, {"specHash": self.compiled.sha256, "legalityOnly": True}, "spec_fixed_legal"
        key = canonical_sha256(dict(observation=observation, profile=canonical_sha256(self.policy)))
        if key not in self.bid_cache:
            if len(choices) == 1:
                self.bid_cache[key] = choices[0], {"forcedLegalAction": True}, "forced_by_rules"
            else:
                choice, evidence = _informed_bid(observation, machine, self.rules,
                    self.policy["informed"], int(key[:16], 16), self.stopped)
                self.bid_cache[key] = choice, {**evidence, "samplingSeed": int(key[:16], 16)}, "sampled_policy_not_exact"
            if len(self.bid_cache) > 256:
                self.bid_cache.pop(next(iter(self.bid_cache)))
        choice, evidence, guarantee = self.bid_cache[key]
        return choice, observation, evidence, guarantee

    def refine_auction(self, model, attempt, solver, remaining, maximum_conflicts):
        """Bind the first informed-bid disagreement in the existing deal model."""
        while attempt.status == "satisfied":
            self.remaining()
            self.auction_proposals += 1
            self.receipts = []
            state = self.engine.create_round(seed=self.seed, dealer=model.dealer)
            self.engine.initial_deal_from_preordered(state, attempt.deck_order)
            machine = BiddingMachine(state.dealer, state.purchase_card.suit,
                AuctionDomainIRV1.from_rules(self.rules.data), state.purchase_card.rank)
            public = []
            for expected in attempt.auction.to_dict()["completion"]["actions"]:
                choice, scope, evidence, guarantee = self.choose_bid(state, machine, expected)
                action = dict(actor=machine.actor.name.lower(), action=choice.action,
                    **(dict(trumpSuit=choice.trump_suit.name.lower()) if choice.trump_suit is not None else {}))
                self.record(machine.actor, "bid", scope, action, evidence, guarantee=guarantee)
                code = choice.action+(":"+choice.trump_suit.name.lower() if choice.trump_suit is not None else "")
                if any(action.get(key) != expected.get(key) for key in ("actor", "action", "trumpSuit")):
                    bound = model.bind_native_auction_observed_choice(actor=machine.actor,
                        visible_cards=scope["observerHand"], purchase_card=scope["purchaseCard"], action=code,
                        public_choices=public, maximum_seconds=min(self.remaining(), remaining()),
                        maximum_conflicts=maximum_conflicts, cancel_check=self.stopped, _prefix_legality=True)
                    if bound["status"] != "bound":
                        raise OutcomePolicyIncomplete("informed_bid_legality_incomplete", bound)
                    self.auction_cuts.append(dict(inputScopeHash=canonical_sha256(scope), action=code, binding=bound))
                    attempt = model.solve(seed=self.seed, solver=solver, search_branching="automatic",
                        maximum_seconds=min(self.remaining(), remaining()), maximum_conflicts=maximum_conflicts,
                        cancel_check=self.stopped)
                    break
                public.append(code)
                machine.submit(choice.action, trump=choice.trump_suit)
            else:
                return attempt
        return attempt

    def dd_card(self, state, info, policy):
        if info.observer != info.current_player:
            raise ValueError("card policy requires the acting seat's information")
        settings = self.policy["cardPolicy"]
        wall = min(5, settings["maximumWallSeconds"]) if policy == "optimal" else settings["maximumWallSeconds"]
        cpu = min(5, settings["maximumCpuSeconds"]) if policy == "optimal" else settings["maximumCpuSeconds"]
        deadline = min(self.deadline, time.monotonic()+wall)
        cpu_deadline = time.process_time()+cpu
        objective = self.compiled.source.get("objective", "awarded_point_margin")
        scope = info.to_dict() if policy == "informed" else state.to_dict(reveal_hands=True)
        key = canonical_sha256(dict(scope=scope, policy=policy, objective=objective,
                                    profile=canonical_sha256(self.policy)))
        if key in self.card_cache:
            selected, evidence, guarantee, observation = self.card_cache[key]
            return selected, {**evidence, "cacheHit": True}, guarantee, observation
        if policy == "optimal":
            if self.dd is None:
                self.dd = DoubleDummySolver(self.rules)
            solver = self.dd
        else:
            # No private true-deal/other-seat search cache enters this actor's
            # informed evaluation, including through budget/cache-hit behavior.
            if info.observer not in self.informed_dd:
                self.informed_dd[info.observer] = DoubleDummySolver(self.rules)
            solver = self.informed_dd[info.observer]
        def stopped():
            return time.monotonic() >= deadline or time.process_time() >= cpu_deadline
        limits = dict(maximum_nodes=settings["maximumNodes"], maximum_cache_entries=settings["maximumCacheEntries"],
                      cancel_check=stopped)
        if policy == "optimal":
            limits["maximum_nodes"] = min(10000000, limits["maximum_nodes"])
            result = solver.solve(state, objective=objective, maximum_seconds=max(1e-6, deadline-time.monotonic()), **limits)
            if result["status"] != "exact" or stopped():
                raise OutcomePolicyIncomplete("dd_optimal_incomplete", result)
            answer = LegalPlayActionV1.from_code(result["selectedAction"]), result, "exact_complete_state", scope
        else:
            # Sampling is a function of this observer's information, never the
            # true hidden deal, construction seed or private opportunity history.
            sample_seed = int(canonical_sha256(dict(information=scope, seed=settings["samplingSeed"]))[:16], 16)
            cohort = generate_hidden_deals(info, self.engine, requested_samples=settings["particles"],
                exact_allocation_limit=(settings["maximumSamplingAttempts"]
                    if settings["method"] == "native_dd_pimc_batch_v1" else settings["particles"]),
                maximum_attempts=settings["maximumSamplingAttempts"],
                seed=sample_seed, cancel_check=stopped)
            if not cohort.coverage_complete and len(cohort.states) != settings["particles"]:
                raise OutcomePolicyIncomplete("pimc_sampling_incomplete")
            samples, exhaustive = cohort.states, cohort.coverage_complete
            if settings["method"] == "native_dd_pimc_batch_v1" and len(samples) > settings["particles"]:
                # A bounded exhaustive allocator may discover more than K legal
                # worlds. Sample K uniformly, never send an oversized cohort.
                indices, rng = list(range(len(samples))), SplitMix64(sample_seed)
                for index in range(settings["particles"]):
                    chosen_index = index+rng.randbelow(len(indices)-index)
                    indices[index], indices[chosen_index] = indices[chosen_index], indices[index]
                samples = tuple(samples[index] for index in indices[:settings["particles"]])
                exhaustive = False
            worlds, values, native_cohort = [], None, None
            if settings["method"] == "native_dd_pimc_batch_v1":
                native_cohort = solver.solve_many(samples, objective=objective,
                    maximum_threads=min(settings["maximumThreads"], self.maximum_threads or settings["maximumThreads"]),
                    maximum_seconds=max(1e-6, deadline-time.monotonic()), **limits)
                worlds = native_cohort["worlds"]
                if (native_cohort["status"] != "exact" or len(worlds) != len(samples)
                        or any(world["status"] != "exact" for world in worlds) or stopped()):
                    raise OutcomePolicyIncomplete("pimc_world_incomplete", dict(
                        solvedWorlds=sum(world["status"] == "exact" for world in worlds), cohort=native_cohort))
            else:
                for world in samples:
                    if stopped():
                        raise OutcomePolicyIncomplete("pimc_decision_budget", {"solvedWorlds": len(worlds)})
                    result = solver.solve(world, objective=objective, all_root_values=True,
                        maximum_seconds=max(1e-6, deadline-time.monotonic()), **limits)
                    if result["status"] != "exact" or stopped():
                        raise OutcomePolicyIncomplete("pimc_world_incomplete", dict(solvedWorlds=len(worlds), world=result))
                    worlds.append(result)
            for result in worlds:
                row = {r["action"]: r["value"] for r in result["rootActions"]}
                if values is None:
                    values = {action: [] for action in row}
                if set(row) != set(values):
                    raise AssertionError("PIMC worlds disagree on observer-legal actions")
                for action in values:
                    values[action].append(row[action])
            if not values:
                raise OutcomePolicyIncomplete("pimc_sampling_incomplete")
            def action_id(code):
                action = LegalPlayActionV1.from_code(code)
                return action.card.id+32*action.ekka
            chosen = min(values, key=lambda code: (-sum(values[code]), action_id(code)))
            evidence = dict(method="sampled_pimc_dd_batch_v1" if native_cohort is not None else "sampled_pimc_dd_v1",
                objective=objective, sampleSeed=sample_seed,
                requestedParticles=settings["particles"], sampledWorlds=len(worlds),
                sampler=cohort.method, enumeratedWorlds=len(cohort.states), samplingAttempts=cohort.attempted,
                exhaustiveCohort=exhaustive,
                values=values, meanValues={code: sum(row)/len(row) for code, row in values.items()}, worlds=worlds)
            if native_cohort is not None:
                evidence["cohort"] = {key: value for key, value in native_cohort.items() if key != "worlds"}
            answer = LegalPlayActionV1.from_code(chosen), evidence, "sampled_policy_not_exact", scope
        if len(self.card_cache) >= 256:
            self.card_cache.pop(next(iter(self.card_cache)))
        self.card_cache[key] = answer
        return answer

    def construct_line(self, context, proposed=None, *, continue_after_deviation=False):
        self.receipts, self.events, self.final_state = [], [], None
        self.first_difference = None
        engine, policies = self.engine, dict(self.compiled.seat_policies)
        state = engine.create_round(seed=self.seed, dealer=Seat.parse(context["dealer"]))
        engine.initial_deal_from_preordered(state, context["deckOrder"])
        self.deck = tuple(context["deckOrder"])
        machine = BiddingMachine(state.dealer, state.purchase_card.suit, AuctionDomainIRV1.from_rules(self.rules.data), state.purchase_card.rank)
        bids, escalations, plays, project_actions = [], [], [], []
        expected_bids = self.compiled.source.get("auction", context["auction"]["completion"]["actions"])
        while not machine.finished:
            self.remaining()
            actor = machine.actor
            expected = expected_bids[len(bids)] if len(bids) < len(expected_bids) else {}
            choice, observation, evidence, guarantee = self.choose_bid(state, machine, expected)
            action = dict(actor=actor.name.lower(), action=choice.action)
            if choice.trump_suit is not None:
                action["trumpSuit"] = choice.trump_suit.name.lower()
            self.record(actor, "bid", observation, action, evidence, guarantee=guarantee)
            expected = expected_bids[len(bids)] if len(bids) < len(expected_bids) else {}
            if any(action.get(key) != expected.get(key) for key in ("actor", "action", "trumpSuit")):
                raise AssertionError("auction refinement did not reach its recorded informed policy")
            bids.append(action)
            machine.submit(choice.action, trump=choice.trump_suit)
        if not engine.complete_deal_from_bid_result(state, machine.result):
            raise OutcomePolicyIncomplete("all_pass_under_requested_policy")
        source = self.compiled.source
        if (("mode" in source and state.contract.value != source["mode"])
                or ("trumpSuit" in source and state.trump_suit != Suit.parse(source["trumpSuit"]))
                or ("buyer" in source and state.buyer != Seat.parse(source["buyer"]))):
            raise AssertionError("constructed legal auction differs from the spec-fixed contract")
        stage = OrToolsEscalationStageV1(engine.escalation_domain, declarer=state.declarer,
            contract=state.contract, match_score=tuple(state.match_score), native_cp_sat=True)
        expected_escalation = context["escalation"]["observation"]["actions"]
        for _ in range(20):
            replay = stage.replay_opportunities(required_actions=(), events=self.events,
                opportunity_order=self.policy["opportunityOrder"], maximum_seconds=self.remaining(), cancel_check=self.stopped)
            if replay["status"] != "satisfied":
                raise OutcomePolicyIncomplete("escalation_normalization_incomplete", replay)
            actor_name = replay["nextActor"]
            if actor_name is None:
                break
            actor = Seat.parse(actor_name)
            choices = [stage.domain.closure_action, *(row["code"].split(":",1)[1]
                for row in replay["finalState"]["opportunityState"]["remainingChoices"] if row["actor"] == actor_name)]
            if self.spec_fixed_contract:
                expected = expected_escalation[len(escalations)] if len(escalations) < len(expected_escalation) else {}
                code = (expected["action"]+(":"+expected["requestedMode"] if expected.get("requestedMode") else "")
                        if expected.get("actor") == actor_name else stage.domain.closure_action)
                observation = _preplay_information(state, actor).to_dict()
                evidence, guarantee = {"specHash": self.compiled.sha256, "legalityOnly": True}, "spec_fixed_legal"
            else:
                code, observation, evidence = self._informed_escalation(state, actor, choices)
                guarantee = "sampled_policy_not_exact"
            if code not in choices:
                raise AssertionError("policy escalation is not legal")
            self.record(actor, "escalation", observation, code, evidence, guarantee=guarantee)
            self.events.append(dict(actor=actor_name, actionCode=code))
            if code != stage.domain.closure_action:
                action, _, mode = code.partition(":")
                engine.apply_escalation(state, seat=actor, action=action, mode=mode or None)
                escalations.append(dict(actor=actor_name, action=action, **({"requestedMode": mode} if mode else {})))
        else:
            raise OutcomePolicyIncomplete("escalation_decision_budget")
        actual = [(r["actor"], r["action"], r.get("requestedMode")) for r in escalations]
        if actual != [(r["actor"], r["action"], r.get("requestedMode")) for r in expected_escalation]:
            raise OutcomePolicyIncomplete("escalation_policy_incompatible")
        engine.prepare_projects(state)
        state._project_declaration_state = engine.project_domain.declaration_state(state.projects, dealer=state.dealer)
        while state.current_player is not None:
            self.remaining()
            actor = state.current_player
            if not state.completed_tricks:
                before = InformationState.from_complete(state, actor).to_dict()
                choices = engine.project_domain.legal_declaration_actions(
                    state._project_declaration_state, trump_suit=state.trump_suit)
                if len(choices) != 1:
                    raise AssertionError("mandatory project declaration is not forced")
                action = choices[0].to_dict()
                engine.apply_project_declaration_action(state, action)
                project_actions.append(action)
                self.record(actor, "project", before, action,
                    {"mandatoryDeclaration": True, "rulesHash": self.rules.sha256}, guarantee="forced_by_rules")
            info = InformationState.from_complete(state, actor)
            legal = engine.legal_play_actions(state)
            if len(legal) == 1:
                selected, evidence, guarantee = legal[0], {"forcedLegalAction": True}, "forced_by_rules"
                observation = info.to_dict()
            elif "cardPolicy" in self.policy:
                selected, evidence, guarantee, observation = self.dd_card(state, info, policies[actor.name.lower()])
            elif policies[actor.name.lower()] == "informed":
                settings = self.policy["informed"]
                result = analyze_information_set_mcts(info, engine, particles=settings["particles"],
                    exact_allocation_limit=1, maximum_sampling_attempts=settings["maximumSamplingAttempts"],
                    iterations=settings["iterations"], include_ekka=True,
                    seed=self.seed+len(self.receipts), cancel_check=self.stopped)
                selected = LegalPlayActionV1(result.recommended_card, result.recommended_ekka)
                evidence, guarantee, observation = result.to_dict(), "sampled_policy_not_exact", info.to_dict()
            else:
                prefix = _prefix(self.deck, context["dealer"], bids, escalations, plays, project_actions)
                code, evidence = self.exact(prefix, "play", actor)
                selected, guarantee = LegalPlayActionV1.from_code(code), "exact_complete_state"
                observation = dict(completePrefix=prefix.to_dict())
            if selected not in legal:
                raise AssertionError("selected policy card is not engine-legal")
            self.record(actor, "play", observation, selected.to_dict(), evidence, guarantee=guarantee)
            if proposed is not None:
                expected = proposed[len(plays)]
                if (expected.actor, expected.card, expected.ekka) != (actor, selected.card, selected.ekka):
                    if self.first_difference is None:
                        self.first_difference = OutcomePolicyDeviation(len(plays), selected, canonical_sha256(observation))
                    if not continue_after_deviation:
                        raise self.first_difference
            engine.play_card(state, selected.card, ekka=selected.ekka)
            plays.append(dict(actor=actor.name.lower(), card=selected.card.code, ekka=selected.ekka))
        self.final_state = state
        return plays

    def bind(self, build, context):
        self.receipts, self.events, self.final_state = [], [], None
        self.compiled.constrain_play_model(build)
        if "cardPolicy" in self.policy:
            return
        plays = self.construct_line(context)
        for index, play in enumerate(plays):
            slot = build.predicate_context.tricks[index//4].actions[index%4]
            build.model.add(slot.card == Card.parse(play["card"]).id)
            build.model.add(slot.actor == int(Seat.parse(play["actor"])))
            build.model.add(slot.ekka == int(play["ekka"]))

    def refine(self, build, context, proposed, solver, remaining, maximum_conflicts):
        """Add the first observer-state implication, then solve this same model.

        CP-SAT's Python API re-solves the augmented model; it does not retain
        learned SAT clauses across calls. The deal/model/solver remain shared.
        """
        while True:
            self.remaining()
            self.receipts, self.events, self.final_state = [], [], None
            self.proposals += 1
            try:
                if self.refinement == "forward_rollout":
                    actual = self.construct_line(context, proposed.plays, continue_after_deviation=True)
                    if self.first_difference is not None:
                        self.forward_probes += 1
                        active = build.model.new_bool_var(f"forward_line_{self.forward_probes}")
                        for index, action in enumerate(actual):
                            slot = build.predicate_context.tricks[index//4].actions[index%4]
                            for variable, value in ((slot.card, Card.parse(action["card"]).id),
                                    (slot.actor, int(Seat.parse(action["actor"]))), (slot.ekka, int(action["ekka"]))):
                                build.model.add(variable == value).only_enforce_if(active)
                        previous_assumptions = tuple(build.model.proto.assumptions)
                        build.model.add_assumption(active)
                        verified = False
                        try:
                            # Same model/solver; native predicates check the
                            # entire policy line, without a second scorer.
                            witness = build.solve(solver=solver, maximum_time_seconds=min(self.remaining(), remaining()),
                                maximum_conflicts=maximum_conflicts, random_seed=self.seed, cancel_check=self.stopped)
                            verified = True
                            return witness
                        except PlayTailSolveStopped as stopped:
                            if stopped.solver_status != "infeasible":
                                raise
                        finally:
                            if not verified:
                                build.model.clear_assumptions()
                                build.model.proto.assumptions.extend(previous_assumptions)
                                build.model.add(active == 0)
                        self.final_state = None
                        raise self.first_difference
                else:
                    self.construct_line(context, proposed.plays)
                return proposed
            except OutcomePolicyDeviation as deviation:
                # The model fixes the deal, auction and escalation. Exact public
                # prefix equality therefore also fixes this seat's information.
                guards = []
                for index, action in enumerate(proposed.plays[:deviation.index]):
                    slot = build.predicate_context.tricks[index//4].actions[index%4]
                    for name, variable, value in (("card", slot.card, action.card.id),
                            ("actor", slot.actor, int(action.actor)), ("ekka", slot.ekka, int(action.ekka))):
                        equal = build.model.new_bool_var(f"policy_cut_{len(self.cuts)}_{index}_{name}")
                        build.model.add(variable == value).only_enforce_if(equal)
                        build.model.add(variable != value).only_enforce_if(equal.Not())
                        guards.append(equal)
                slot = build.predicate_context.tricks[deviation.index//4].actions[deviation.index%4]
                build.model.add(slot.card == deviation.action.card.id).only_enforce_if(guards)
                build.model.add(slot.ekka == int(deviation.action.ekka)).only_enforce_if(guards)
                self.cuts.append(dict(index=deviation.index, action=deviation.action.code,
                    inputScopeHash=deviation.scope_hash, publicPrefixHash=canonical_sha256([
                        row.to_dict() for row in proposed.plays[:deviation.index]]),
                    ownHandFixedByModel=True))
                proposed = build.solve(solver=solver, maximum_time_seconds=min(self.remaining(), remaining()),
                    maximum_conflicts=maximum_conflicts, random_seed=self.seed, cancel_check=self.stopped)


def _construction_scenario(compiled, policy):
    # Target a plausible opening hand before paying for a whole policy line.
    # These proposal hints never replace bid-quality selection/admission.
    raw = compiled.scenario.to_dict()
    raw.pop("sha256")
    raw["constraints"].pop("sha256", None)
    hints = policy["construction"]
    mode = compiled.source.get("mode", "hokum" if "trumpSuit" in compiled.source else None)
    if mode in {"sun", "ashkal"}:
        leaves = [("deal.rank_count", dict(seat="north", handStage="initial_five", rank="ace",
                    minimum=hints["sunOpeningAces"], maximum=4))]
    elif mode == "hokum":
        leaves = [("deal.card_owner", dict(card=Card.of(Suit.parse(compiled.source["trumpSuit"]), Rank.parse(rank)).id,
                    seat="north", handStage="initial_five")) for rank in hints["hokumOpeningHonors"]]
    else:
        leaves = []
    # A named winner/victim already implies ownership; lower it before fixing
    # a deal, rather than discovering the wrong owner at the play solve.
    named = compiled.source.get("outcome", {}).get("decidingTrick", {})
    winning_team = named.get("winningTeam", compiled.source.get("team", compiled.source.get("outcome", {}).get("winner", "north_south")))
    for key, side in (("winningCard", winning_team), ("beatenCard", named.get("beatenCardTeam",
            "west_east" if winning_team == "north_south" else "north_south"))):
        if key in named:
            leaves.append(("deal.card_allowed", dict(card=Card.parse(named[key]).id, handStage="post_purchase",
                seats=[seat.name.lower() for seat in Seat if seat.team.name.lower() == side])))
    for index, (kind, arguments) in enumerate(leaves):
        raw["constraints"]["children"].append(dict(predicateId=f"PRED-POLICY-SEED-{index:03d}",
            kind=kind, version=1, hardness="hard", arguments=arguments))
    return ResearchScenarioV1.build(raw)


def construct_outcome(compiled: CompiledOutcomeSpec, rules: RuleBook, policy: Mapping[str, Any], *,
                      seed: int, maximum_seconds: float = 120, maximum_attempts: int = 3,
                      refinement: str = "first_deviation") -> dict[str, Any]:
    """Legacy single-result cut/refinement path; batch generation uses rollouts."""
    compiled.require_integrity()
    if compiled.rules_hash != rules.sha256:
        raise ValueError("outcome spec and construction rules differ")
    if compiled.source.get("objective", "awarded_point_margin") != "awarded_point_margin" and "cardPolicy" not in policy:
        raise ValueError("raw_point_margin requires the DD card-policy profile; it is not admitted yet")
    if type(maximum_seconds) not in {int, float} or not 0 < maximum_seconds < 300:
        raise ValueError("local construction must fit a run shorter than five minutes")
    if type(maximum_attempts) is not int or maximum_attempts < 1:
        raise ValueError("maximum_attempts must be positive")
    policy = validate_outcome_policy(policy)
    if refinement not in ("first_deviation", "forward_rollout") or (
            refinement == "forward_rollout" and "cardPolicy" not in policy):
        raise ValueError("requested refinement requires the DD policy profile")
    construction_scenario = _construction_scenario(compiled, policy)
    start, cpu = time.monotonic(), time.process_time()
    deadline = start+maximum_seconds
    rejects, attempts, excluded, decisions, accepted = Counter(), [], [], [], None
    driver = None
    for number in range(maximum_attempts):
        if time.monotonic() >= deadline:
            rejects["campaign_time_budget"] += 1
            break
        driver = OutcomePolicyDriver(rules, compiled, policy, seed=seed+number, deadline=deadline, refinement=refinement)
        attempt_start = time.monotonic()
        partial, result = None, None
        try:
            result = GenericCompleteTraceCompilerV1(rules=rules, native_cp_sat_scoring=True,
                play_model_configurator=driver.bind,
                **(dict(auction_model_refiner=driver.refine_auction) if not driver.spec_fixed_contract else {}),
                **(dict(play_model_refiner=driver.refine) if "cardPolicy" in policy else {}))(
                    construction_scenario, tuple(excluded), seed=seed+number,
                    maximum_seconds=deadline-time.monotonic())
            if result.candidate is None:
                reason = ("policy_line_native_verification:" if driver.final_state is not None else "construction:") + (result.reason or result.status)
                partial = result.partial_receipt
            elif driver.final_state is None:
                raise AssertionError("native construction bypassed the policy callback")
            elif time.monotonic() >= deadline:
                reason = "native_verification_budget"
            else:
                candidate = result.candidate
                prefix = construction_trace_prefix_v1(candidate.trace)
                terminal = driver.final_state.round_score
                for name, points in (("rawPoints", terminal.raw_card_points), ("cardGamePoints", terminal.card_game_points),
                        ("projectPoints", terminal.project_points), ("balootPoints", terminal.baloot_points),
                        ("totalBeforeAward", terminal.total_before_award), ("gamePoints", terminal.awarded_points)):
                    if candidate.trace["score"][name] != list(points):
                        raise AssertionError(f"native/engine scoring disagrees on {name}")
                if (candidate.trace["score"]["roundWinner"] != terminal.winner.name.lower()
                        or candidate.trace["score"]["kabout"] != terminal.kabout):
                    raise AssertionError("native/engine outcome class disagrees")
                accepted = dict(seed=seed+number, specHash=compiled.sha256, rulesHash=rules.sha256,
                    policyProfile=policy, policyProfileHash=canonical_sha256(policy), seatPolicies=dict(compiled.seat_policies),
                    constructionScenario=construction_scenario.to_dict(),
                    trace=candidate.trace, exactTracePrefix=prefix.to_dict(), decisionReceipts=driver.receipts,
                    policyRefinement=dict(proposals=driver.proposals, cuts=driver.cuts,
                        auctionProposals=driver.auction_proposals, auctionCuts=driver.auction_cuts,
                        strategy=refinement, forwardProbes=driver.forward_probes),
                    score=driver.final_state.round_score.to_dict(), qualification="outcome_verified_under_declared_policies")
                attempts.append(dict(seed=seed+number, reason="accepted", wallSeconds=time.monotonic()-attempt_start,
                                     completedPolicyDecisions=len(driver.receipts), proposals=driver.proposals, cuts=driver.cuts,
                                     auctionProposals=driver.auction_proposals, auctionCuts=driver.auction_cuts,
                                     forwardProbes=driver.forward_probes))
                break
        except OutcomePolicyIncomplete as exc:
            reason = exc.reason
            partial = exc.evidence
        except (InformationSetCompilationStopped, ExactNormalizationStopped, InterruptedError) as exc:
            reason = f"policy_incomplete:{getattr(exc, 'reason', type(exc).__name__)}"
            partial = getattr(exc, "partial_receipt", None)
        except RuntimeError as exc:
            if str(exc) != "no legal hidden deals could reproduce the public history":
                raise
            reason = "informed_sampling_incomplete"
        rejects[reason] += 1
        decisions.extend(driver.receipts)
        attempts.append(dict(seed=seed+number, reason=reason, wallSeconds=time.monotonic()-attempt_start,
                             completedPolicyDecisions=len(driver.receipts), partialEvidence=partial,
                             proposals=driver.proposals, cuts=driver.cuts,
                             auctionProposals=driver.auction_proposals, auctionCuts=driver.auction_cuts,
                             forwardProbes=driver.forward_probes))
        proved_unsat = result is not None and result.reason == "fixed_deal_play_infeasible" and (
            result.partial_receipt or {}).get("exhaustionReason") == "infeasible"
        if proved_unsat and driver.deck is not None:
            excluded.append(driver.deck)
        # UNKNOWN is not a no-good, and must not prevent trying another deal.
        driver.close()
    if driver is not None:
        driver.close()
    wall, used_cpu = time.monotonic()-start, time.process_time()-cpu
    core = dict(schemaVersion=1, status="complete" if accepted else "incomplete", accepted=accepted,
        refinementStrategy=refinement,
        specHash=compiled.sha256, policyProfile=policy, policyProfileHash=canonical_sha256(policy),
        constructionScenario=construction_scenario.to_dict(), attempts=attempts,
        rejectReasons=dict(rejects), partialDecisionReceipts=decisions, wallSeconds=wall, cpuSeconds=used_cpu,
        acceptedPerSecond=(1/wall if accepted else 0), cpuSecondsPerAccepted=(used_cpu if accepted else None))
    return {**core, "sha256": canonical_sha256(core)}
