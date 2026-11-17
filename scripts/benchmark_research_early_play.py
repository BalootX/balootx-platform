"""Exploratory matched-play check for Research playing policies.

This is diagnostic evidence, not a Research qualification or strength claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import monotonic
from typing import Callable, Mapping

from baloot_platform.analysis.belief_v7 import BeliefPolicy
from baloot_platform.analysis.evaluator import AnalysisPolicy, analyze
from baloot_platform.analysis.models import InformationState
from baloot_platform.config import build_bundle
from baloot_platform.core.cards import Card
from baloot_platform.core.bots import BotProfile, HeuristicBot
from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, Seat, Team
from baloot_platform.core.models import LegalPlayActionV1
from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
from baloot_platform.research.ortools_research_executor import (
    _observer_safe_play_policy_artifact_v1,
    _select_quality_play_action_v1,
)
from baloot_platform.research.scenario_contracts import (
    ExactTracePrefixV1,
    ScenarioPlayingQualityPolicyV1,
    ScenarioProofBudgetV1,
    ScenarioProofRequestV1,
    ScenarioQualityExecutionBudgetV1,
)


class BenchmarkBudgetExceeded(RuntimeError):
    """A diagnostic arm stopped without a complete scored round."""


def _prefix(initial, state, plays, *, project_actions=None):
    declarations = [project.public_declaration() for project in state.projects if project.declared]
    declarations.sort(key=lambda row: row["declarationOrder"])
    return ExactTracePrefixV1.build({
        "schemaVersion": 1,
        "deckOrder": list(initial.shuffle.deck_order),
        "dealer": initial.dealer.name.lower(),
        "roundNumber": initial.round_number,
        "matchScore": list(initial.match_score),
        "auctionActions": [
            {"actor": bid.seat.name.lower(), "action": bid.action,
             **({"trumpSuit": bid.trump.name.lower()} if bid.trump is not None else {})}
            for bid in initial.bid_history
        ],
        "escalationActions": [],
        **({"projectActions": project_actions, "projectDeclarations": []}
           if project_actions is not None else {"projectDeclarations": declarations}),
        "playActions": plays,
    })


def _first_divergence_proof(initial, rules, arms, policy, execution, budget):
    left, right = (arm["playActions"] for arm in arms)
    index = next((i for i, (a, b) in enumerate(zip(left, right, strict=True))
                  if a != b), None)
    if index is None:
        return {"status": "same_trace"}
    if left[:index] != right[:index]:
        raise AssertionError("first divergence must share the same legal prefix")
    actor = Seat.parse(left[index]["actor"])
    if right[index]["actor"] != actor.name.lower():
        raise AssertionError("divergent arms changed acting seat")
    declaration = rules.project_domain.declaration_state(initial.projects,
                                                          dealer=initial.dealer)
    project_actions = []
    for _ in range(min(index + 1, 4)):
        options = rules.project_domain.legal_declaration_actions(
            declaration, trump_suit=initial.trump_suit)
        if len(options) != 1:
            raise AssertionError("first-divergence proof requires forced project actions")
        declaration, _ = rules.project_domain.apply_declaration_action(
            declaration, options[0], trump_suit=initial.trump_suit)
        project_actions.append(options[0].to_dict())
    prefix = _prefix(initial, initial, left[:index], project_actions=project_actions)
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "play"}, maximum_seconds=20)
    if normalized.next_actor != actor.name.lower():
        raise AssertionError("first-divergence actor differs from normalized state")
    request_value = policy.request_for("play", actor.name.lower()).to_dict()
    request_value.pop("sha256")
    request_value["targetTeam"] = actor.team.name.lower()
    request = ScenarioProofRequestV1.build(request_value)

    def code(row):
        return LegalPlayActionV1(Card.parse(row["card"]), row["ekka"]).code

    candidate, baseline = code(left[index]), code(right[index])
    try:
        _, proof, native, _ = _select_quality_play_action_v1(
            rules, prefix, normalized, request, quality_policy=policy,
            execution_budget=execution, budget=budget, featured_now=True,
            actual_action=None, remaining_seconds=lambda: budget.maximum_wall_seconds,
            cancel_check=None, private_recall=None,
            maximum_worlds=min(budget.maximum_hidden_worlds,
                               execution.maximum_hidden_worlds_per_decision),
            maximum_depth=budget.maximum_depth)
    except InformationSetCompilationStopped as exc:
        receipt = exc.partial_receipt
        stages = []
        while isinstance(receipt, dict):
            stages.append({"stage": receipt["stage"],
                           "reason": receipt["exhaustionReason"],
                           "evidenceKeys": sorted(receipt["evidence"])})
            receipt = receipt["evidence"].get("causeReceipt")
        return {"status": "incomplete", "playIndex": index, "actor": actor.name.lower(),
                "candidateAction": candidate, "baselineAction": baseline,
                "reason": exc.reason, "receiptStages": stages}
    if proof is None or native is None:
        raise AssertionError("first-divergence proof did not use the native observed model")
    roots = {row["action"]: row["value"] for row in proof.to_dict()["rootActions"]}
    if candidate not in roots or baseline not in roots:
        raise AssertionError("first-divergence action is missing from proof roots")
    return {"status": proof.completion_status, "playIndex": index,
            "actor": actor.name.lower(), "exactStateId": normalized.exact_state_id,
            "candidateAction": candidate, "baselineAction": baseline,
            "candidateValue": roots[candidate], "baselineValue": roots[baseline],
            "hiddenWorlds": native["rootArtifact"].get("hiddenWorlds")}


def run_pair(*, seed: int, contract: Contract, rules, early_tricks: int = 6,
             candidate_selector: Callable | None = None,
             candidate_policy: str = "observer_safe_trick_control_v1",
             current_quality_profile: Mapping | None = None,
             diagnose_first_divergence: bool = False,
             opening_action: str | None = None) -> dict:
    """Swap teams, or screen one forced opening on a frozen legal round."""
    if opening_action is not None and current_quality_profile is None:
        raise ValueError("opening action screen requires the current quality profile")
    engine = BalootEngine(rules)
    initial = engine.create_round(seed=seed, dealer=Seat(seed % 4))
    engine.shuffle_and_initial_deal(initial, cut_position=(seed * 17 % 31) + 1)
    buyer = initial.dealer.next()
    trump = None if contract == Contract.SUN else initial.purchase_card.suit
    auction = BiddingMachine(initial.dealer, initial.purchase_card.suit,
                             AuctionDomainIRV1.from_rules(rules.data))
    if auction.actor != buyer:
        raise AssertionError("benchmark buyer is not the first bidder")
    result = auction.submit(contract.value, trump=trump)
    while result is None:
        result = auction.submit("pass")
    if not engine.complete_deal_from_bid_result(initial, result):
        raise AssertionError("benchmark auction did not purchase")
    engine.prepare_projects(initial)
    execution = None
    proof_budget = None
    policy = None
    if current_quality_profile is not None:
        execution = ScenarioQualityExecutionBudgetV1.build(
            current_quality_profile["qualityExecution"])
        proof_budget = ScenarioProofBudgetV1.build(
            "bounded_v1", current_quality_profile["proofBudgets"]["profiles"]["bounded_v1"])
        boundaries = current_quality_profile["authoringDefaults"]["playingQuality"][
            "boundaryEvaluatorByDecision"]
        policy = ScenarioPlayingQualityPolicyV1.build({
            "schemaVersion": 1, "mode": "budgeted",
            "decisions": {kind: {
                "utility": "awarded_point_margin", "budgetProfile": "bounded_v1",
                "horizon": {"unit": "tricks" if kind == "play" else "plies", "value": 1},
                "boundaryEvaluator": boundaries[kind],
            } for kind in ("bid", "escalation", "project", "play")},
        })
    source = _prefix(initial, initial, [], project_actions=[] if policy else None)
    arms = []
    for candidate_team in ((Team.NORTH_SOUTH,) if opening_action else
                           (Team.NORTH_SOUTH, Team.WEST_EAST)):
        request = ScenarioProofRequestV1.build({
            "schemaVersion": 1, "mode": "bounded", "visibility": "actor_information_set",
            "strategyClass": "pure_deterministic", "claim": "winning_root_action_count",
            "targetTeam": candidate_team.name.lower(), "utility": "awarded_point_margin",
            "successThreshold": 0, "budgetProfile": "benchmark",
            "horizon": {"unit": "tricks", "value": 1},
            "boundaryEvaluator": "current_raw_point_margin_v1",
        })
        state = initial.clone()
        normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
        bot = HeuristicBot(engine, BotProfile(deterministic_seed=seed))
        plays = []
        project_actions = [] if policy else None
        declaration = (rules.project_domain.declaration_state(initial.projects,
                       dealer=initial.dealer) if policy else None)
        policy_plays = 0
        started = monotonic()

        def remaining_quality_seconds():
            seconds = min(proof_budget.maximum_wall_seconds,
                          started + execution.maximum_wall_seconds - monotonic())
            if seconds <= 0:
                raise BenchmarkBudgetExceeded(
                    f"arm={candidate_team.name.lower()} play={len(plays)} "
                    "exceeded frozen qualityExecution.maximumWallSeconds")
            return seconds

        while state.current_player is not None:
            actor = state.current_player
            if declaration is not None and len(plays) < 4:
                options = rules.project_domain.legal_declaration_actions(
                    declaration, trump_suit=state.trump_suit)
                if len(options) != 1 or options[0].actor != actor:
                    raise AssertionError("matched benchmark requires a forced project action")
                declaration, _ = rules.project_domain.apply_declaration_action(
                    declaration, options[0], trump_suit=state.trump_suit)
                project_actions.append(options[0].to_dict())
            prefix = _prefix(initial, state, plays, project_actions=project_actions)
            legal = {action.code for action in engine.legal_play_actions(state)}
            if actor.team == candidate_team and (policy is not None or
                                                  state.current_trick.index < early_tricks):
                try:
                    normalized = normalizer.normalize(prefix, decision_point={"kind": "play"},
                                                      maximum_seconds=20)
                except Exception as exc:
                    raise RuntimeError(
                        f"native prefix failed at seed={seed} team={candidate_team.name.lower()} "
                        f"play={len(plays)} prefix={prefix.sha256} actions={plays} "
                        f"declarations={list(prefix.project_declarations)}"
                    ) from exc
                native_legal = {str(row["code"]) for row in normalized.legal_actions}
                if normalized.next_actor != actor.name.lower() or native_legal != legal:
                    raise AssertionError("native/runtime play decision disagrees")
                if opening_action is not None and not plays:
                    if opening_action not in native_legal:
                        raise ValueError("forced opening action is not legal")
                    action = LegalPlayActionV1.from_code(opening_action)
                elif policy is not None:
                    request_value = policy.request_for("play", actor.name.lower()).to_dict()
                    request_value.pop("sha256")
                    request_value["targetTeam"] = actor.team.name.lower()
                    actor_request = ScenarioProofRequestV1.build(request_value)
                    artifact, proof, _, _ = _select_quality_play_action_v1(
                        rules, prefix, normalized, actor_request,
                        quality_policy=policy, execution_budget=execution,
                        budget=proof_budget, featured_now=False, actual_action=None,
                        remaining_seconds=remaining_quality_seconds,
                        cancel_check=None, private_recall=None,
                        maximum_worlds=min(proof_budget.maximum_hidden_worlds,
                                           execution.maximum_hidden_worlds_per_decision),
                        maximum_depth=proof_budget.maximum_depth,
                    )
                    proof_value = artifact["proof"] if artifact is not None else proof.to_dict()
                    quality = (artifact["proofModel"] if artifact is not None
                               else proof.model_document())["actionQuality"]
                    if (proof_value["completionStatus"] != "bounded_complete"
                            or quality["actualActionOptimal"] is not True
                            or quality["actualAction"] != proof_value["selectedAction"]):
                        raise RuntimeError("current Research play selection is incomplete")
                    action = LegalPlayActionV1.from_code(proof_value["selectedAction"])
                elif candidate_selector is None:
                    code, artifact = _observer_safe_play_policy_artifact_v1(
                        rules, normalized, request, actual_action=None,
                        maximum_seconds=2, cancel_check=None)
                    if artifact["request"]["targetTeam"] != actor.team.name.lower():
                        raise AssertionError("Research policy request targets the wrong team")
                    action = LegalPlayActionV1.from_code(code)
                else:
                    action = candidate_selector(state, len(plays))
                policy_plays += 1
            else:
                action = LegalPlayActionV1(bot.choose(state))
            if action.code not in legal:
                raise AssertionError("benchmark policy chose an illegal action")
            plays.append({"actor": actor.name.lower(), "card": action.card.code,
                          "ekka": action.ekka})
            engine.play_card(state, action.card, ekka=action.ekka)
        final = normalizer.normalize(_prefix(initial, state, plays,
                                             project_actions=project_actions),
                                     decision_point={"kind": "play"}, maximum_seconds=20)
        if state.round_score is None or final.state["roundScore"] is None:
            raise AssertionError("benchmark round did not score")
        runtime_score = state.round_score.to_dict()
        native_score = final.state["roundScore"]
        if native_score != runtime_score:
            raise AssertionError(f"native/runtime scoring differs: {native_score} != {runtime_score}")
        if len(plays) != 32:
            raise AssertionError("benchmark round is incomplete")
        margin = (runtime_score["awardedPoints"][int(candidate_team)]
                  - runtime_score["awardedPoints"][int(candidate_team.other)])
        arms.append({"candidateTeam": candidate_team.name.lower(), "margin": margin,
                     "playActions": plays, "finalPrefixHash": _prefix(initial, state, plays,
                         project_actions=project_actions).sha256,
                     "policyPlayCount": policy_plays, "roundScore": runtime_score,
                     "seconds": round(monotonic() - started, 3)})
    return {"seed": seed, "contract": contract.value, "rulesHash": rules.sha256,
            "sourcePrefixHash": source.sha256, "candidatePolicy": candidate_policy,
            "baselinePolicy": "transparent_heuristic",
            **({"exactPlayFromTrick": execution.exact_play_from_trick,
                "projectScope": "mandatory_forced_declarations_only"} if policy else
               {"earlyTricks": early_tricks}),
            **({"qualityExecutionHash": execution.sha256,
                "proofBudgetHash": proof_budget.sha256,
                "policyHash": policy.sha256,
                "profileScope": "source_candidate_ordinary_nonfeatured",
                **({"candidateAdvantage": sum(arm["margin"] for arm in arms)}
                   if opening_action is None else {})} if policy else {}),
            **({"firstDivergence": _first_divergence_proof(
                initial, rules, arms, policy, execution, proof_budget)}
               if diagnose_first_divergence and policy else {}),
            **({"openingAction": opening_action, "scope": "single_deal_causal_screen"}
               if opening_action else {}),
            "arms": arms,
            **({"pairedMargin": sum(arm["margin"] for arm in arms) / 2}
               if opening_action is None else {})}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--contract", choices=["sun", "hokum"], required=True)
    parser.add_argument("--candidate", choices=["v1", "analyzer16", "current_source"], default="v1")
    parser.add_argument("--first-divergence-proof", action="store_true")
    parser.add_argument("--opening-action")
    args = parser.parse_args()
    if args.first_divergence_proof and args.candidate != "current_source":
        parser.error("first-divergence proof requires --candidate current_source")
    if args.opening_action and args.candidate != "current_source":
        parser.error("opening action screen requires --candidate current_source")
    root = Path(__file__).resolve().parents[1]
    bundle = build_bundle(root, **({"bundle_path": "configs/bundle_intelligent_research.yaml"}
                                   if args.candidate == "current_source" else {}))
    selector = None
    policy_id = "observer_safe_trick_control_v1"
    if args.candidate == "current_source":
        policy_id = "source_candidate_ordinary_nonfeatured"
    elif args.candidate == "analyzer16":
        belief = BeliefPolicy.from_profile(bundle.component("belief"))
        policy_id = "benchmark_analyzer_16_v1"
        policy = AnalysisPolicy.from_profile({
            "profile": {"id": policy_id, "version": "1.0.0"},
            "requested_samples": 16,
            "exact_allocation_limit": 100,
            "exact_search_max_remaining": 0,
            "maximum_sampling_attempts": 2000,
            "objective": "round_margin",
        }, belief_policy=belief)

        def selector(state, play_index):
            info = InformationState.from_complete(state, state.current_player)
            result = analyze(
                info, bundle.rules, bundle.convention,
                analysis_policy=policy, belief_policy=belief,
                seed=args.seed * 100 + play_index,
            )
            if result.hidden_deals < 1 or any(
                row.search_completion_status not in {"exact", "heuristic_complete"}
                for row in result.actions
            ):
                raise RuntimeError("analyzer16 did not complete its governed sample")
            # The analyzer ranks cards, not Ekka declarations. Screen its card
            # choice with the ordinary action; Ekka quality remains out of scope.
            return LegalPlayActionV1(result.recommended_card, False)

    try:
        result = run_pair(seed=args.seed, contract=Contract(args.contract),
                          rules=bundle.rules, candidate_selector=selector,
                          candidate_policy=policy_id,
                          current_quality_profile=(bundle.component("challenge")["scenarioResearchV4"]
                              if args.candidate == "current_source" else None),
                          diagnose_first_divergence=args.first_divergence_proof,
                          opening_action=args.opening_action)
    except BenchmarkBudgetExceeded as exc:
        result = {"status": "incomplete", "reason": str(exc), "seed": args.seed,
                  "contract": args.contract, "candidatePolicy": policy_id,
                  "rulesHash": bundle.rules.sha256}
    if args.candidate == "current_source":
        result["bundleSourceHash"] = bundle.sha256
        result["bundleProfileVersion"] = bundle.bundle_profile["profile"]["version"]
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
