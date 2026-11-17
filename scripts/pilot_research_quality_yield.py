"""Native quality-yield diagnostics, not qualification or strength claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import monotonic, process_time
from typing import Mapping

import yaml

from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.cards import Card
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, Seat
from baloot_platform.core.models import LegalPlayActionV1
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.scoring_ir import ScoringDomainIRV1
from baloot_platform.research.ortools_complete_state_graph import OrToolsCompleteStateGraphCompilerV1
from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
from baloot_platform.research.ortools_proof import OrToolsScenarioGraphProofV1
from baloot_platform.research.ortools_research_executor import (
    resolve_quality_post_auction_continuation_v1,
    resolve_quality_round_continuation_v1,
)
from baloot_platform.research.runtime_compatibility import check_runtime_compatibility_v1
from baloot_platform.research.scenario_contracts import (
    ExactTracePrefixV1,
    ScenarioPlayingQualityPolicyV1,
    ScenarioProofRequestV1,
    ScenarioQualityExecutionBudgetV1,
    canonical_sha256,
)


def _stop_chain(receipt: Mapping | None) -> list[dict]:
    chain = []
    while isinstance(receipt, Mapping):
        chain.append({key: receipt[key] for key in ("stage", "exhaustionReason", "status") if key in receipt})
        evidence = receipt.get("evidence")
        receipt = (evidence.get("causeReceipt") if isinstance(evidence, Mapping) else None)
    return chain


def run_full_round(input_path: Path, rules_path: Path, *, seconds: int,
                   exact_play_from_trick: int) -> dict:
    """Retain one native full-round attempt; never retry or claim qualification."""
    from baloot_platform.research.result_authority import validate_native_round_quality_evidence_v1

    root = Path(__file__).resolve().parents[1]
    settings = yaml.safe_load((root / "configs/challenge/intelligent_research.yaml").read_text(
        encoding="utf-8"))["scenarioResearchV4"]
    execution_raw = dict(settings["qualityExecution"])
    if not 0 < seconds <= execution_raw["maximumWallSeconds"]:
        raise ValueError("attempt exceeds the authored execution ceiling")
    execution_raw.update(exactPlayFromTrick=exact_play_from_trick, exactTailProof=True,
                         earlyPlayPolicyId="observer_safe_trick_control_v1")
    execution = ScenarioQualityExecutionBudgetV1.build(execution_raw)
    supplied = json.loads(input_path.read_text(encoding="utf-8"))
    raw = dict(supplied.get("exactTracePrefix", supplied))
    for key in ("sha256",):
        raw.pop(key, None)
    # The supplied deal is fixed; no historic action or project claim is injected.
    raw.update(auctionActions=[], escalationActions=[], projectActions=[],
               projectDeclarations=[], projectAssertions=[], playActions=[], stateAssertions={})
    initial = ExactTracePrefixV1.build(raw)
    rules = RuleBook.load(rules_path)
    authoring = settings["authoringDefaults"]["playingQuality"]
    profile_id = authoring["budgetProfile"]
    budget = settings["proofBudgets"]["profiles"][profile_id]
    policy = ScenarioPlayingQualityPolicyV1.build({
        "schemaVersion": 2, "mode": "budgeted", "visibility": "actor_information_set",
        "decisions": {kind: {
            "utility": "awarded_point_margin", "budgetProfile": profile_id,
            "horizon": {"unit": "tricks" if kind == "play" else "plies", "value": 1},
            "boundaryEvaluator": authoring["boundaryEvaluatorByDecision"][kind],
        } for kind in ("bid", "escalation", "project", "play")},
    })
    decision = {"kind": "play", "selector": {}}
    started, started_cpu = monotonic(), process_time()

    def progress(row):
        if row.get("kind") in {"quality_decision_checkpoint_v1", "quality_stage_checkpoint_v1"}:
            print(json.dumps({key: row[key] for key in (
                "kind", "stage", "playIndex", "actor", "action", "elapsedSeconds"
            ) if key in row}), flush=True)

    terminal = []
    receipt = resolve_quality_round_continuation_v1(
        rules, initial, decision, policy,
        opportunity_order=execution.escalation_opportunity_order,
        maximum_nodes=budget["maximumNodes"], maximum_depth=budget["maximumDepth"],
        maximum_worlds=min(budget["maximumHiddenWorlds"], execution.maximum_hidden_worlds_per_decision),
        maximum_seconds=execution.maximum_wall_seconds, runtime_maximum_seconds=seconds,
        maximum_decision_seconds=budget["maximumWallSeconds"],
        maximum_escalation_decisions=execution.maximum_escalation_decisions,
        maximum_evidence_bytes=execution.maximum_evidence_bytes,
        proof_budget_profiles=settings["proofBudgets"]["profiles"],
        search_budget_profiles=settings["searchBudgets"]["profiles"],
        execution_budget=execution, checkpoint_callback=progress,
        final_normalized_callback=terminal.append,
    )
    # Independent readback covers partial receipts too; incomplete remains incomplete.
    readback_error = None
    try:
        validate_native_round_quality_evidence_v1(receipt, source_prefix=initial,
            rules_hash=rules.sha256, policy=policy, decision_point=decision,
            rules=rules, search_budget_profiles=settings["searchBudgets"]["profiles"],
            maximum_seconds=max(0.001, seconds - (monotonic() - started)))
    except (ValueError, RuntimeError, TimeoutError) as error:
        # Preserve the solver receipt while failing closed on readback disagreement.
        readback_error = f"{type(error).__name__}: {error}"
    post = receipt.get("postAuction") or {}
    play = post.get("projectPlay") or {}
    rows, artifacts = play.get("decisions") or [], play.get("artifacts") or {}
    tail = [row for row in rows if row["decisionKind"] == "play"
            and row["playIndex"] >= 4 * (exact_play_from_trick - 1)]
    tail_exact = readback_error is None and len(tail) == 4 * (9 - exact_play_from_trick) and all(
        artifacts[row["proofArtifactHash"]]["request"]["mode"] == "exact"
        and artifacts[row["proofArtifactHash"]]["proof"]["completionStatus"] == "exact_complete"
        and row["actualActionOptimal"] is True
        and artifacts[row["proofArtifactHash"]]["proofModel"]["kind"] in {
            "actor_information_set_cegis_model_v1", "forced_action_optimality_model_v1"}
        and (artifacts[row["proofArtifactHash"]]["proofModel"]["kind"] == "forced_action_optimality_model_v1"
             or artifacts[row["proofArtifactHash"]]["proof"]["hiddenWorlds"]["complete"] is True)
        for row in tail)
    configuration_hash = canonical_sha256({"rulesHash": rules.sha256, "settings": settings,
        "qualityExecution": execution.to_dict()})
    compatibility = None
    if readback_error is None and receipt["status"] == "complete" and len(receipt["finalPrefix"]["playActions"]) == 32:
        from baloot_platform.version import __version__
        assert len(terminal) == 1, "complete round must retain its terminal normalization"
        compatibility = check_runtime_compatibility_v1(
            exact_state=terminal[0].to_dict(), semantic_result_hash=receipt["sha256"],
            bundle_hash=configuration_hash,
            trace_hash=ExactTracePrefixV1.build(receipt["finalPrefix"]).sha256,
            rules=rules, engine_version=__version__, match_score_before=initial.match_score,
            project_actions=receipt["finalPrefix"]["projectActions"],
        ).receipt.to_dict()
    return {"summary": {"status": "verification_failed" if readback_error else receipt["status"], "scope": receipt["scope"],
        "exactPlayFromTrick": exact_play_from_trick, "tailCertified": tail_exact,
        "exactTailNonForcedDecisions": sum(
            len(artifacts[row["proofArtifactHash"]]["proof"]["rootActions"]) > 1
            for row in tail),
        "playActions": len(receipt["finalPrefix"]["playActions"]),
        "auditedDecisions": len(receipt["auction"]["decisions"]) + len(rows)
            + len((post.get("escalation") or {}).get("decisions") or []),
        "elapsedSeconds": round(monotonic() - started, 3),
        "cpuSeconds": round(process_time() - started_cpu, 3),
        "runtimeCompatibility": compatibility["status"] if compatibility else "not_checked",
        "stopChain": _stop_chain(receipt.get("partialReceipt")),
        "boundary": "Authored-config diagnostic, not a published-bundle qualified result; early decisions bounded/policy-conditioned."},
        "sourcePrefix": initial.to_dict(), "rulesHash": rules.sha256, "readbackError": readback_error,
        "authoredConfigurationHash": configuration_hash, "runtimeCompatibility": compatibility,
        "playingQuality": policy.to_dict(), "receipt": receipt}


def _late_probes(rules: RuleBook, complete_prefix: Mapping, decisions: list[dict],
                 artifacts: Mapping, *, start: int, limit: int) -> list[dict]:
    probes = []
    normalizer = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True)
    compiler = OrToolsCompleteStateGraphCompilerV1(
        play_domain=rules.play_domain,
        scoring_domain=ScoringDomainIRV1.from_rules(rules.data),
        native_conversion=True,
        structural_domain=rules.structural_domain,
        project_domain=rules.project_domain,
    )
    for count in range(start, 32):
        if len(probes) == limit:
            break
        raw = {key: value for key, value in complete_prefix.items() if key != "sha256"}
        raw["playActions"] = list(complete_prefix["playActions"][:count])
        raw["stateAssertions"] = {}
        raw["projectAssertions"] = []
        prefix = ExactTracePrefixV1.build(raw)
        started = monotonic()
        started_cpu = process_time()
        exact = normalizer.normalize(prefix, decision_point={"kind": "play"}, maximum_seconds=10)
        if len(exact.legal_actions) < 2:
            continue
        request = ScenarioProofRequestV1.build({
            "schemaVersion": 1, "mode": "exact", "visibility": "complete_state",
            "claim": "winning_root_action_count", "targetTeam": Seat.parse(exact.next_actor).team.name.lower(),
            "utility": "awarded_point_margin", "successThreshold": 0,
            "budgetProfile": "late_game_exact_v1",
        })
        compiled = compiler.compile(exact, request, expected_rules_hash=rules.sha256,
                                    maximum_nodes=10_000, maximum_depth=32-count,
                                    maximum_seconds=20)
        recorded = next((row for row in decisions
                         if row.get("decisionKind") == "play" and row["playIndex"] == count), None)
        played = complete_prefix["playActions"][count]
        recorded_action = LegalPlayActionV1(Card.parse(played["card"]), played.get("ekka", False)).code
        row = {"playIndex": count, "actor": exact.next_actor, "exactStateId": exact.exact_state_id,
               "playedCard": Card.parse(played["card"]).code, "recordedAction": recorded_action,
               "legalActionCount": len(exact.legal_actions), "graphComplete": compiled.complete}
        if recorded is not None:
            quality_artifact = artifacts[recorded["proofArtifactHash"]]
            quality = quality_artifact["proofModel"].get("actionQuality", {})
            hidden = quality_artifact["proof"].get("hiddenWorlds") or {}
            row["recordedQuality"] = {
                "action": recorded["action"], "selectionReason": recorded["selectionReason"],
                "actualActionOptimal": recorded["actualActionOptimal"], "regret": recorded["regret"],
                "scope": quality.get("scope"), "actionScores": quality.get("actionScores"),
                "horizon": quality_artifact["request"].get("horizon"),
                "hiddenWorldCount": hidden.get("count"), "hiddenWorldsComplete": hidden.get("complete"),
            }
        if compiled.complete:
            proof = OrToolsScenarioGraphProofV1().prove(compiled.graph, request, maximum_seconds=10)
            row.update(proofCompletion=proof.completion_status,
                       rootValues=[item.to_dict() for item in proof.root_actions],
                       outcomeChanging=len({item.satisfies for item in proof.root_actions}) > 1,
                       uniquelyWinning=proof.winning_root_action_count == 1,
                       recordedIsUniqueWinner=(proof.winning_root_action_count == 1
                                               and any(item.action_code == recorded_action and item.satisfies
                                                       for item in proof.root_actions)))
        row["elapsedSeconds"] = round(monotonic() - started, 3)
        row["cpuSeconds"] = round(process_time() - started_cpu, 3)
        probes.append(row)
    return probes


def run(seed: int, contract: Contract, seconds: int, decision_seconds: int, worlds: int,
        published_budget: bool, late_probes: bool, probe_start: int, probe_limit: int,
        exact_play_from_trick: int = 7) -> dict:
    rules = RuleBook.load(Path("configs/rules/saudi_project_optional_unratified_v1.yaml"))
    engine = BalootEngine(rules)
    state = engine.create_round(seed=seed, dealer=Seat.EAST)
    engine.shuffle_and_initial_deal(state, cut_position=(seed * 17 % 31) + 1)
    auction = BiddingMachine(state.dealer, state.purchase_card.suit, AuctionDomainIRV1.from_rules(rules.data))
    result = auction.submit(contract.value, trump=None if contract == Contract.SUN else state.purchase_card.suit)
    while result is None:
        result = auction.submit("pass")
    assert engine.complete_deal_from_bid_result(state, result)
    prefix = ExactTracePrefixV1.build({
        "schemaVersion": 1, "deckOrder": list(state.shuffle.deck_order),
        "dealer": state.dealer.name.lower(), "roundNumber": state.round_number,
        "matchScore": list(state.match_score),
        "auctionActions": [{"actor": bid.seat.name.lower(), "action": bid.action,
                            **({"trumpSuit": bid.trump.name.lower()} if bid.trump else {})}
                           for bid in state.bid_history],
        "escalationActions": [], "projectActions": [], "playActions": [],
    })
    profile = yaml.safe_load(Path("configs/challenge/intelligent_research.yaml").read_text(encoding="utf-8"))[
        "scenarioResearchV4"] if published_budget else None
    policy = ScenarioPlayingQualityPolicyV1.build({
        "schemaVersion": 1, "mode": "budgeted",
        "decisions": {kind: {"utility": "awarded_point_margin",
                             "budgetProfile": "bounded_v1" if published_budget else "test",
                             "horizon": {"unit": "tricks" if kind == "play" else "plies", "value": 1},
                             "boundaryEvaluator": (profile["authoringDefaults"]["playingQuality"][
                                 "boundaryEvaluatorByDecision"][kind] if profile else "current_raw_point_margin_v1")}
                      for kind in ("bid", "escalation", "project", "play")},
    })
    execution = dict(profile["qualityExecution"]) if profile else {
        "schemaVersion": 1, "maximumWallSeconds": seconds, "maximumEvidenceBytes": 30_000_000,
        "maximumEscalationDecisions": 20,
        "earlyPlayPolicyId": "observer_safe_trick_control_v1", "exactPlayFromTrick": 7,
    }
    execution["exactPlayFromTrick"] = exact_play_from_trick
    budget = ScenarioQualityExecutionBudgetV1.build(execution)
    terminal = []
    started = monotonic()
    started_cpu = process_time()
    attempt = resolve_quality_post_auction_continuation_v1(
        rules, prefix, {"kind": "escalation", "selector": {"actingSeat": "west"}}, policy,
        opportunity_order=("west", "east", "north", "south"),
        maximum_nodes=250_000 if profile else 1000, maximum_depth=40 if profile else 8,
        maximum_worlds=worlds,
        maximum_seconds=budget.maximum_wall_seconds,
        runtime_maximum_seconds=seconds if profile else None,
        maximum_decision_seconds=20 if profile else decision_seconds,
        maximum_escalation_decisions=budget.maximum_escalation_decisions,
        maximum_evidence_bytes=budget.maximum_evidence_bytes,
        proof_budget_profiles={"bounded_v1": profile["proofBudgets"]["profiles"]["bounded_v1"]}
        if profile else None,
        execution_budget=budget, final_normalized_callback=terminal.append,
    )
    compatibility = "not_checked"
    if attempt["status"] == "complete":
        assert len(attempt["finalPrefix"]["playActions"]) == 32 and len(terminal) == 1
        check = check_runtime_compatibility_v1(
            exact_state=terminal[0].to_dict(), semantic_result_hash=attempt["sha256"],
            bundle_hash="b" * 64, trace_hash=prefix.sha256, rules=rules,
            engine_version="quality-yield-pilot",
            match_score_before=prefix.match_score,
            project_actions=attempt["finalPrefix"]["projectActions"],
        )
        compatibility = check.receipt.to_dict()["status"]
    probes = (_late_probes(rules, attempt["finalPrefix"], attempt["projectPlay"]["decisions"],
                           attempt["projectPlay"]["artifacts"], start=probe_start, limit=probe_limit)
              if late_probes and compatibility == "passed" else [])
    partial = attempt["partialReceipt"] or {}
    artifact = (attempt["projectPlay"] or {}).get("artifacts", {}).get(partial.get("proofArtifactHash"), {})
    proof = artifact.get("proof", {})
    return {"seed": seed, "contract": contract.value, "scope": attempt["scope"],
            "exactPlayFromTrick": exact_play_from_trick,
            "budget": "configured_candidate_bounded_v1_pilot_60s" if profile else "synthetic",
            "status": attempt["status"], "runtimeCompatibility": compatibility,
            "playActions": len(attempt["finalPrefix"]["playActions"]),
            "auditedDecisions": len((attempt["projectPlay"] or {}).get("decisions", [])),
            "elapsedSeconds": round(monotonic() - started, 3),
            "cpuSeconds": round(process_time() - started_cpu, 3),
            "trickSixActions": attempt["finalPrefix"]["playActions"][20:24],
            "awardedPoints": (terminal[0].state["roundScore"]["awardedPoints"]
                              if compatibility == "passed" else None),
            "stopChain": _stop_chain(attempt["partialReceipt"]),
            "stopProof": {key: proof[key] for key in ("completionStatus", "exhaustionReason", "stopReason", "solverStatus")
                          if key in proof},
            "stopProofKeys": sorted(proof) if partial.get("proofArtifactHash") else [],
            "lateProbes": probes,
            "_completeTracePrefix": attempt["finalPrefix"] if attempt["status"] == "complete" else None,
            "compatibilityFailures": check.receipt.to_dict()["failureCodes"] if attempt["status"] == "complete" else []}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--contract", choices=("sun", "hokum"))
    parser.add_argument("--full-round-input", type=Path)
    parser.add_argument("--rules", type=Path)
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--decision-seconds", type=int, default=2)
    parser.add_argument("--worlds", type=int, default=1)
    parser.add_argument("--published-budget", action="store_true")
    parser.add_argument("--late-probes", action="store_true")
    parser.add_argument("--probe-start", type=int, choices=range(20, 32), default=24)
    parser.add_argument("--probe-limit", type=int, choices=range(1, 13), default=2)
    parser.add_argument("--exact-play-from-trick", type=int, choices=range(1, 9), default=7)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.full_round_input is not None:
        if args.rules is None or args.output_dir is None or args.exact_play_from_trick < 2:
            parser.error("full-round input requires rules, output directory, and an exact tail after trick one")
        destination = args.output_dir / "actor-visible-full-round.json"
        if destination.exists():
            parser.error("output already exists; preserve the earlier attempt")
        document = run_full_round(args.full_round_input, args.rules, seconds=args.seconds,
                                  exact_play_from_trick=args.exact_play_from_trick)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as output:
            json.dump(document, output, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        print(json.dumps({**document["summary"], "retainedPath": str(destination)}, sort_keys=True))
        raise SystemExit(0)
    if args.seed is None or args.contract is None:
        parser.error("post-auction pilot requires seed and contract")
    summary = run(args.seed, Contract(args.contract), args.seconds, args.decision_seconds,
                  args.worlds, args.published_budget, args.late_probes,
                  args.probe_start, args.probe_limit, args.exact_play_from_trick)
    trace = summary.pop("_completeTracePrefix")
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        path = args.output_dir / f"post-auction-{args.contract}-{args.seed}.json"
        with path.open("x", encoding="utf-8") as output:
            json.dump({"summary": summary, "completeExactTracePrefix": trace}, output,
                      ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        summary["retainedPath"] = str(path)
    print(json.dumps(summary, sort_keys=True))
