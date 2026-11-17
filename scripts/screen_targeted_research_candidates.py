"""Native V3 diagnostic sieve; neither structural witnesses nor screens qualify rounds.

Run through run_local_capped.ps1. Full quality/admission and the unchanged video
interestingness filter are still required for every retained candidate.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
import urllib.request
from pathlib import Path

from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Rank, Seat, Suit
from baloot_platform.filesystem import atomic_write_text
from baloot_platform.research.scenario_contracts import (
    ExactTracePrefixV1, ResearchScenarioV1, ScenarioProofBudgetV1,
    ScenarioProofRequestV1, canonical_sha256,
)

ROOT = Path(__file__).resolve().parents[1]


def retain(path, value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"immutable evidence already exists: {path.name}")
    else:
        atomic_write_text(path, text)


def plans(spec):
    required = {"schemaVersion", "bundleRevisionId", "candidateCount", "seedStart",
        "constructionBudgetProfile", "bidBudgetProfile", "campaignBudgetProfile",
        "contracts", "dealers", "trumpSuits", "initialHighCardRange",
        "postPurchaseHighCardRange", "postPurchaseAceTenRange", "trumpCountRange",
        "project", "splitTrumpRanks", "selectedValueBand", "maximumAlternativeGap", "maximumSurvivors"}
    if set(spec) - {"maximumBidScreenSeconds", "screeningRoot"} != required or type(spec["schemaVersion"]) is not int or spec["schemaVersion"] != 1:
        raise ValueError("closed diagnostic batch schema mismatch")
    if (type(spec.get("screeningRoot", "initial_bid")) is not str
            or spec.get("screeningRoot", "initial_bid") not in {"initial_bid", "construction_purchase"}):
        raise ValueError("unsupported diagnostic screeningRoot")
    if "maximumBidScreenSeconds" in spec and (type(spec["maximumBidScreenSeconds"]) is not int
            or spec["maximumBidScreenSeconds"] < 1):
        raise ValueError("positive integer required: maximumBidScreenSeconds")
    for name in ("candidateCount", "seedStart", "maximumAlternativeGap", "maximumSurvivors"):
        if type(spec[name]) is not int or spec[name] < 1:
            raise ValueError(f"positive integer required: {name}")
    if spec["maximumSurvivors"] > spec["candidateCount"]:
        raise ValueError("survivor count exceeds candidate count")
    for name in ("initialHighCardRange", "postPurchaseHighCardRange", "postPurchaseAceTenRange",
                 "trumpCountRange", "selectedValueBand"):
        pair = spec[name]
        if (not isinstance(pair, list) or len(pair) != 2
                or any(type(x) is not int for x in pair) or pair[0] > pair[1]):
            raise ValueError(f"ordered integer range required: {name}")
    for name, allowed in (("contracts", {"sun", "hokum"}),
            ("dealers", {"north", "west", "south", "east"}),
            ("trumpSuits", {"diamonds", "clubs", "hearts", "spades"})):
        values = spec[name]
        if (not isinstance(values, list) or not values or any(type(x) is not str for x in values)
                or len(set(values)) != len(values) or not set(values) <= allowed):
            raise ValueError(f"invalid diagnostic dimension: {name}")
    if (set(spec["project"]) != {"kind", "gameValuesByContract", "state"}
            or spec["project"]["state"] != "eligible"
            or set(spec["project"]["gameValuesByContract"]) != set(spec["contracts"])
            or any(type(v) is not int or v < 0 for v in spec["project"]["gameValuesByContract"].values())):
        raise ValueError("both teams are targeted for eligibility, not simultaneous awards")
    if (not isinstance(spec["splitTrumpRanks"], list) or len(spec["splitTrumpRanks"]) != 2
            or any(type(rank) is not str for rank in spec["splitTrumpRanks"])
            or len(set(spec["splitTrumpRanks"])) != 2):
        raise ValueError("two distinct trump ranks required")
    for rank in spec["splitTrumpRanks"]:
        Rank.parse(rank)
    result = []
    seats = ("north", "west", "south", "east")
    for index in range(spec["candidateCount"]):
        contract = spec["contracts"][index % len(spec["contracts"])]
        dealer = spec["dealers"][(index // len(spec["contracts"])) % len(spec["dealers"])]
        trump = spec["trumpSuits"][(index // len(spec["contracts"])) % len(spec["trumpSuits"])]
        predicates = []

        def add(predicate_kind, **arguments):
            predicates.append(dict(predicateId=f"PRED-TARGET-{index:03}-{len(predicates):02}",
                kind=predicate_kind, version=1, hardness="hard", arguments=arguments))

        add("auction.contract", contract=contract)
        add("auction.dealer", seat=dealer)
        if contract == "hokum":
            add("auction.trump", suit=trump)
            for rank, team_seats in zip(spec["splitTrumpRanks"], (("north", "south"), ("west", "east"))):
                add("deal.card_allowed", card=Card.of(Suit.parse(trump), Rank.parse(rank)).id,
                    seats=list(team_seats), handStage="post_purchase")
        for seat in seats:
            for stage, bounds in (("initial_five", spec["initialHighCardRange"]),
                    ("post_purchase", spec["postPurchaseHighCardRange"])):
                add("deal.high_card_count", seat=seat, handStage=stage,
                    minimum=bounds[0], maximum=bounds[1])
            for rank in ("ace", "ten"):
                bounds = spec["postPurchaseAceTenRange"]
                add("deal.rank_count", seat=seat, handStage="post_purchase", rank=rank,
                    minimum=bounds[0], maximum=bounds[1])
            if contract == "hokum":
                bounds = spec["trumpCountRange"]
                add("deal.suit_count", seat=seat, handStage="post_purchase", suit=trump,
                    minimum=bounds[0], maximum=bounds[1])
        for team in ("north_south", "west_east"):
            add("project.exists", team=team, kind=spec["project"]["kind"], state="eligible",
                value=spec["project"]["gameValuesByContract"][contract])
        scenario = ResearchScenarioV1.build(dict(schemaVersion=1, origin={"kind": "search"},
            constraints={"op": "all", "children": predicates},
            decisionPoint={"kind": "play", "selector": {"trickIndex": 0, "position": 0}},
            outcome={"predicateId": f"PRED-REPORT-{index:03}", "kind": "state.legal_action_count",
                "version": 1, "hardness": "report", "arguments": {"minimum": 0, "maximum": 32}}))
        result.append(dict(index=index, seed=spec["seedStart"] + index,
            dealer=dealer, scenario=scenario.to_dict()))
    return result


def numeric_survivor(artifact, spec):
    """Editorial sieve only; accept no unfinished or pass-action producer."""
    if artifact.get("proof", {}).get("completionStatus") != "exact_complete":
        return False
    graph = artifact.get("graph", {})
    if not isinstance(graph.get("selectedAction"), str) or graph["selectedAction"] == "pass":
        return False
    strategy = graph.get("selectedValueProof", {}).get("strategy", {})
    lower, upper = strategy.get("lowerBound"), strategy.get("upperBound")
    alternatives = graph.get("alternatives", [])
    if (type(lower) is not int or type(upper) is not int or lower > upper or not alternatives
            or any(type(row.get("upperBound")) is not int for row in alternatives)):
        return False
    gap = lower - max(row["upperBound"] for row in alternatives)
    band = spec["selectedValueBand"]
    return lower <= band[1] and upper >= band[0] and 0 <= gap <= spec["maximumAlternativeGap"]


def require_screening_allocation(seconds, incumbent):
    if (incumbent is not None and incumbent["schemaVersion"] == 2
            and incumbent.get("completeStateAlternativeFirst", True)
            and seconds <= incumbent["maximumSeconds"]):
        raise ValueError(
            f"incompatible frozen screening allocation: {seconds}s decision allowance "
            f"cannot reserve {incumbent['maximumSeconds']}s for the incumbent and leave "
            "positive alternative time; select a compatible immutable allocation before search")


def construction_selected_bid(trace):
    """Screen the actual final purchase; earlier constructed bids remain unassessed."""
    actions = trace["auction"]["completion"]["actions"]
    purchases = [index for index, row in enumerate(actions) if row["action"] != "pass"]
    if not purchases:
        raise ValueError("construction has no purchase action")
    index = purchases[-1]
    row = actions[index]
    action = row["action"]
    if action == "hokum":
        if not row.get("trumpSuit"):
            raise ValueError("constructed Hokum purchase lacks trump")
        action += ":" + row["trumpSuit"]
    return index, action, [
        {"actor": prior["actor"], "action": prior["action"],
         **({"trumpSuit": prior["trumpSuit"]} if prior.get("trumpSuit") else {})}
        for prior in actions[:index]]


def wait_for_render(url, job):
    while True:
        with urllib.request.urlopen(f"{url}/api/v1/jobs/{job}", timeout=10) as response:
            row = json.load(response)
        if row["status"] in {"queued", "running"}:
            time.sleep(15)
            continue
        if row["status"] != "completed":
            raise RuntimeError(f"owned render ended as {row['status']}; no solver batch launched")
        return


def run(spec, batch, output, data):
    from baloot_platform.service.container import build_container
    from baloot_platform.research.ortools_research_executor import (
        GenericCompleteTraceCompilerV1, OrToolsUnifiedResearchExecutorV1,
        _select_observed_bid_quality_artifact_v1, _complete_state_quality_artifact_v1,
    )
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1, ExactNormalizationStopped
    from baloot_platform.research.ortools_information_set_compiler import InformationSetCompilationStopped
    from baloot_platform.research.scenario_backend import ScenarioResearchAuthorityV1
    from baloot_platform.research.result_authority import reverify_native_bid_dominance_quality_v1
    if (output / "summary.json").exists() or next(output.glob("candidate-*.json"), None) is not None:
        raise ValueError("batch already executed; preserve receipts and choose a new explicit output")
    container = build_container(root=ROOT, writable_root=data)
    try:
        authority, context = container.research.execution_context_factory.freeze_selector(
            bundle_revision_id=spec["bundleRevisionId"], analysis_mode="cascaded")
        frozen = ScenarioResearchAuthorityV1.from_challenge_profile(context.bundle.component("challenge"))
        if (frozen.backend != "ortools_cp_sat_v3" or frozen.quality_execution is None
                or frozen.quality_execution.video_interestingness is None
                or frozen.quality_execution.exact_play_from_trick != 1):
            raise ValueError("native V3 frozen quality authority required")
        search = frozen.search_budgets["profiles"][spec["constructionBudgetProfile"]]
        campaign = frozen.search_budgets["profiles"][spec["campaignBudgetProfile"]]
        budget = ScenarioProofBudgetV1.build(spec["bidBudgetProfile"],
            frozen.proof_budgets["profiles"][spec["bidBudgetProfile"]])
        if len(batch) > campaign["maximumCandidates"] or budget.native_observed is None:
            raise ValueError("diagnostic batch exceeds frozen authority")
        screen_seconds = spec.get("maximumBidScreenSeconds", min(budget.maximum_wall_seconds,
            budget.maximum_cpu_seconds))
        if screen_seconds > min(budget.maximum_wall_seconds, budget.maximum_cpu_seconds):
            raise ValueError("diagnostic screen exceeds its frozen proof tier")
        require_screening_allocation(min(screen_seconds, campaign["maximumWallSeconds"]),
            frozen.quality_execution.bid_incumbent_proof)
        executor = OrToolsUnifiedResearchExecutorV1(context.bundle.rules,
            bundle_revision_id=context.bundle_revision_id, bundle_hash=context.bundle.sha256,
            backend_id="ortools_cp_sat_v3")
        retain(output / "authority.json", dict(configuration=authority.to_dict(),
            snapshot=context.bundle.configuration_snapshot(), spec=spec,
            specHash=canonical_sha256(spec), loweringManifest=executor.lowering_manifest.manifest_hash))
        started, cpu_started = time.monotonic(), time.process_time()
        deadline = started + campaign["maximumWallSeconds"]
        excluded, outcomes, survivors = [], [], []
        normalizer = OrToolsExactStateNormalizerV1(context.bundle.rules, native_cp_sat=True)
        for plan in batch:
            if time.monotonic() >= deadline:
                break
            scenario = ResearchScenarioV1.build(plan["scenario"])
            compiler = GenericCompleteTraceCompilerV1(rules=context.bundle.rules, dealer=plan["dealer"],
                native_cp_sat_scoring=True, structural_only=True, manifest=executor.lowering_manifest)
            attempt = compiler(scenario, tuple(excluded), seed=plan["seed"],
                maximum_seconds=min(search["maximumWallSeconds"], deadline-time.monotonic()),
                maximum_conflicts=search["maximumConflicts"])
            row = dict(index=plan["index"], construction={"status": attempt.status, "reason": attempt.reason,
                "candidate": attempt.candidate.to_dict() if attempt.candidate is not None else None,
                "partialReceipt": attempt.partial_receipt}, freshReadback=False,
                screenPassed=False, qualified=False)
            if attempt.candidate is not None:
                deck = tuple(attempt.candidate.trace["deckOrder"])
                excluded.append(deck)
                selected_action, auction_actions = None, []
                if spec.get("screeningRoot") == "construction_purchase":
                    action_index, selected_action, auction_actions = construction_selected_bid(attempt.candidate.trace)
                    row["screenScope"] = dict(kind="construction_purchase", actionIndex=action_index,
                        selectedAction=selected_action, earlierActionsCertified=False)
                prefix = ExactTracePrefixV1.build(dict(schemaVersion=1,
                    deckOrder=list(deck), dealer=plan["dealer"],
                    auctionActions=auction_actions, escalationActions=[], projectActions=[],
                    projectDeclarations=[], playActions=[], matchScore=[0, 0]))
                decision = {"kind": "bid", "selector": {"actionIndex": len(auction_actions)}}
                request = None
                root_deadline = min(deadline, time.monotonic() + screen_seconds)

                def root_remaining():
                    seconds = root_deadline - time.monotonic()
                    if seconds <= 0:
                        raise InformationSetCompilationStopped("maximum_seconds", stage="targeted_bid_sieve")
                    return seconds

                try:
                    exact = normalizer.normalize(prefix, decision_point=decision,
                        maximum_seconds=root_remaining())
                    if selected_action is not None and selected_action not in {row["code"] for row in exact.legal_actions}:
                        raise ValueError("constructed purchase is not legal at its audited prefix")
                    request = ScenarioProofRequestV1.build(dict(schemaVersion=1, mode="exact",
                        visibility="complete_state", strategyClass="pure_deterministic",
                        claim="winning_root_action_count", targetTeam=Seat.parse(exact.next_actor).team.name.lower(),
                        utility="awarded_point_margin", successThreshold=0, budgetProfile=budget.profile_id,
                        **({"selectedAction": selected_action} if selected_action is not None else {})))
                    proof_options = dict(
                        budget=budget, execution_budget=frozen.quality_execution,
                        maximum_conflicts=search["maximumConflicts"], maximum_depth=budget.maximum_depth,
                        maximum_nodes=budget.maximum_nodes,
                        opportunity_order=frozen.quality_execution.escalation_opportunity_order,
                        maximum_seconds=root_remaining(),
                        _search_generated=True)
                    if selected_action is None:
                        artifact = _select_observed_bid_quality_artifact_v1(
                            context.bundle.rules, exact, prefix, request, decision=decision, **proof_options)
                    else:
                        artifact = _complete_state_quality_artifact_v1(
                            context.bundle.rules, exact, prefix, decision, request,
                            _defer_bid_fallback=True, cancel_check=None, **proof_options)
                    row["bidArtifact"] = artifact
                    if numeric_survivor(artifact, spec):
                        reverify_native_bid_dominance_quality_v1(artifact, rules=context.bundle.rules,
                            request=request, budget=budget, maximum_conflicts=search["maximumConflicts"],
                            execution_budget=frozen.quality_execution,
                            opportunity_order=frozen.quality_execution.escalation_opportunity_order,
                            maximum_seconds=root_remaining())
                        row.update(freshReadback=True, screenPassed=True)
                        survivors.append(plan["index"])
                except (InformationSetCompilationStopped, ExactNormalizationStopped) as stopped:
                    row["incomplete"] = stopped.partial_receipt
            outcomes.append(row)
            retain(output / f"candidate-{plan['index']:03}.json", row)
            print(json.dumps(dict(index=plan["index"], status=attempt.status,
                screenPassed=row["screenPassed"], qualified=False)), flush=True)
            if len(survivors) >= spec["maximumSurvivors"]:
                break
        summary = dict(schemaVersion=1, requested=len(batch), attempted=len(outcomes),
            freshVerifiedScreenSurvivors=survivors, qualifiedRounds=0,
            wallSeconds=time.monotonic()-started, cpuSeconds=time.process_time()-cpu_started,
            stoppedForBudget=time.monotonic() >= deadline,
            boundary="Bid-root screen only; constructed earlier bids are unassessed. Full native V3 all-decision admission and unchanged interestingness gate required.")
        retain(output / "summary.json", summary)
        print(json.dumps(summary), flush=True)
    finally:
        container.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--wait-render-job")
    parser.add_argument("--server", default="http://127.0.0.1:8174")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    batch = plans(spec)
    # Self-check closed types and unfinished-proof rejection without a solver.
    for field in ("candidateCount", "maximumBidScreenSeconds"):
        wrong = copy.deepcopy(spec)
        wrong[field] = True
        try:
            plans(wrong)
        except ValueError:
            pass
        else:
            raise AssertionError(f"coercive {field} accepted")
    assert not numeric_survivor({"proof": {"completionStatus": "incomplete"}}, spec)
    purchase_index, action, prior = construction_selected_bid({"auction": {"completion": {"actions": [
        {"actor": "west", "action": "pass"},
        {"actor": "south", "action": "hokum", "trumpSuit": "hearts"},
        {"actor": "north", "action": "sun"},
        {"actor": "south", "action": "pass"}]}}})
    assert purchase_index == 2 and action == "sun" and len(prior) == 2
    assert prior[1]["trumpSuit"] == "hearts"
    _, action, _ = construction_selected_bid({"auction": {"completion": {"actions": [
        {"actor": "south", "action": "hokum", "trumpSuit": "hearts"}]}}})
    assert action == "hokum:hearts"
    for invalid_root in (False, [], "unregistered"):
        wrong = copy.deepcopy(spec)
        wrong["screeningRoot"] = invalid_root
        try:
            plans(wrong)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid screeningRoot accepted")
    sample = {"proof": {"completionStatus": "exact_complete"}, "graph": {
        "selectedAction": "sun", "selectedValueProof": {"strategy": {"lowerBound": 1, "upperBound": 1}},
        "alternatives": [{"upperBound": 0}]}}
    assert numeric_survivor(sample, spec) == (spec["selectedValueBand"][0] <= 1 <= spec["selectedValueBand"][1])
    sample["graph"]["selectedAction"] = "pass"
    assert not numeric_survivor(sample, spec)
    sample["graph"]["selectedAction"] = "sun"
    sample["graph"]["alternatives"][0]["upperBound"] = 2
    assert not numeric_survivor(sample, spec)
    for seconds, incumbent_seconds, valid in ((20, 20, False), (20, 5, True)):
        try:
            require_screening_allocation(seconds,
                {"schemaVersion": 2, "maximumSeconds": incumbent_seconds})
        except ValueError:
            assert not valid
        else:
            assert valid
    retain(args.output / "plans.json", dict(schemaVersion=1, spec=spec, plans=batch))
    if args.prepare_only:
        print(json.dumps(dict(prepared=len(batch), solverLaunched=False)), flush=True)
        return
    if args.wait_render_job:
        wait_for_render(args.server.rstrip("/"), args.wait_render_job)
    run(spec, batch, args.output, args.data)


if __name__ == "__main__":
    main()
