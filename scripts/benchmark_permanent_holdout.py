#!/usr/bin/env python3
# ruff: noqa: E402
"""Evaluate a frozen agent policy on the complete adjudicated holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import tracemalloc
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY_ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from baloot_platform.agents.profiles import (
    AdaptiveBiddingStrategy,
    AdaptivePlayStrategy,
    AgentProfileRegistry,
)
from baloot_platform.analysis.search_errors import require_uncensored_play_trace
from baloot_platform.benchmark import (
    IntelligenceBenchmarkV18,
    canonical_sha256,
    load_adjudicated_holdout_jsonl,
    load_candidate_jsonl,
    reconstruct_state,
    validate_trusted_sidecar,
)
from baloot_platform.config import build_bundle, platform_root
from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Seat


DEFAULT_HOLDOUT = Path("release/permanent-holdout.jsonl")
DEFAULT_CANDIDATES = Path("private-inputs/strength-candidates.jsonl")
DEFAULT_TRUSTED = Path(".release-runtime/strength-holdout/trusted-replay-recipes.json")
DEFAULT_EXCLUSIONS = Path("release/strength-holdout-excluded-groups.json")
DEFAULT_OUTPUT = Path("release/agent-qualification/permanent-holdout-evaluation.json")


def _git_identity(root: Path) -> tuple[str, str]:
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("permanent-holdout benchmark requires a clean source worktree")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    return revision, tree


def _bid_state(engine: BalootEngine, candidate: dict[str, Any], recipe: dict[str, Any]):
    if str(recipe.get("stage")) != "initial_bidding":
        raise ValueError("trusted recipe is not an initial-bidding state")
    state = engine.create_round(
        seed=int(recipe["seed"]), dealer=Seat.parse(recipe["dealer"])
    )
    engine.shuffle_and_initial_deal(state, cut_position=int(recipe["cutPosition"]))
    if state.purchase_card is None:
        raise ValueError("bidding candidate has no purchase card")
    machine = BiddingMachine(
        state.dealer,
        state.purchase_card.suit,
        AuctionDomainIRV1.from_rules(engine.rules.data),
    )
    if engine.state_hash(state) != str(candidate["stateSha256"]):
        raise ValueError("trusted bidding replay did not reproduce candidate state")
    if machine.actor is None or machine.actor.name.lower() != str(candidate["observer"]):
        raise ValueError("bidding candidate observer is not the actor")
    if list(machine.legal_actions()) != list(candidate["legalActions"]):
        raise ValueError("trusted bidding replay legal actions mismatch")
    return state, machine


def _card_code(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("code") or value.get("action") or "")
    return str(value)


def _completion(trace: dict[str, Any]) -> str:
    require_uncensored_play_trace(trace)
    source = str(trace.get("decisionSource") or "")
    solver = str(trace.get("analysisSolverCompletion") or "")
    if solver == "exact":
        return "exact"
    if source in {"transparent_heuristic", "only_legal_action", "paired_bid_simulation"}:
        return "heuristic_complete"
    if source in {"hidden_information_search", "production_model"}:
        return "budget_exhausted"
    return "failed"


def build_predictions(
    *, root: Path, candidates: dict[str, Any], trusted: dict[str, Any],
    holdout: dict[str, Any], policy_id: str,
) -> list[dict[str, Any]]:
    validate_trusted_sidecar(candidates, trusted)
    candidate_by_id = {
        str(item["candidateId"]): dict(item) for item in candidates["positions"]
    }
    recipes = dict(trusted.get("recipes") or {})
    bundle = build_bundle(root)
    registry = AgentProfileRegistry.from_bundle(root)
    profile = registry.get(policy_id)
    engine = BalootEngine(bundle.rules)
    play = AdaptivePlayStrategy(profile, bundle.convention)
    bidding = AdaptiveBiddingStrategy(profile)
    predictions: list[dict[str, Any]] = []
    for row in holdout["positions"]:
        position_id = str(row["positionId"])
        candidate = candidate_by_id.get(position_id)
        recipe = dict(recipes.get(position_id) or {})
        if candidate is None or not recipe:
            raise ValueError(f"holdout position lacks trusted reconstruction: {position_id}")
        seed = int(canonical_sha256({
            "positionId": position_id, "policyId": policy_id,
            "datasetSha256": holdout["datasetSha256"],
        })[:16], 16)
        tracemalloc.start()
        started = time.perf_counter()
        if recipe.get("stage") == "initial_bidding":
            state, machine = _bid_state(engine, candidate, recipe)
            action, _trump, trace = bidding.choose_with_trace(
                state, machine, bundle.rules, seed=seed
            )
            predicted = str(action)
            ranked = [
                str(item.get("action") or "") for item in trace.get("rankedActions") or ()
            ]
        else:
            state = reconstruct_state(engine, candidate, recipe)
            card, trace = play.choose_with_trace(engine, state, seed=seed)
            predicted = card.code
            ranked = [
                _card_code(item.get("action"))
                for item in trace.get("rankedActions") or ()
            ]
        latency = (time.perf_counter() - started) * 1000.0
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        legal = [str(item) for item in row["legalActions"]]
        ranked = list(dict.fromkeys(item for item in [predicted, *ranked, *legal] if item in legal))
        predictions.append({
            "positionId": position_id,
            "predictedAction": predicted,
            "rankedActions": ranked,
            "predictedConfidence": float(trace.get("confidence", 0.0)),
            "searchCompletion": _completion(trace),
            "latencyMs": latency,
            "memoryBytes": float(peak),
            "nodes": float(trace.get("analysisHiddenDeals") or 0.0),
        })
    return predictions


def evaluate(
    *, root: Path, candidates_path: Path, trusted_path: Path,
    holdout_path: Path, exclusions_path: Path, policy_id: str,
    source_revision: str, source_tree: str,
) -> dict[str, Any]:
    candidates = load_candidate_jsonl(candidates_path)
    trusted = json.loads(trusted_path.read_text(encoding="utf-8"))
    holdout = load_adjudicated_holdout_jsonl(holdout_path)
    excluded = json.loads(exclusions_path.read_text(encoding="utf-8"))
    required = {"dealHash", "episodeId", "sourceGroup", "seedGroup"}
    if set(excluded) != required or any(not isinstance(excluded[key], list) for key in required):
        raise ValueError("excluded groups must explicitly declare all four lineage groups")
    predictions = build_predictions(
        root=root, candidates=candidates, trusted=trusted,
        holdout=holdout, policy_id=policy_id,
    )
    result = IntelligenceBenchmarkV18(
        build_bundle(root).component("benchmark")
    ).evaluate_permanent_holdout(
        policy_id,
        holdout["positions"],
        predictions,
        dataset_id=str(holdout["datasetId"]),
        excluded_groups=excluded,
    ).to_dict()
    result.update({
        "sourceRevision": source_revision,
        "sourceTree": source_tree,
        "policyProfileHash": canonical_sha256(
            AgentProfileRegistry.from_bundle(root).get(policy_id).to_dict()
        ),
        "holdoutManifestSha256": holdout["manifestSha256"],
        "excludedGroupsSha256": canonical_sha256(excluded),
    })
    result["resultHash"] = canonical_sha256({
        key: value for key, value in result.items() if key != "resultHash"
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--trusted", type=Path, default=DEFAULT_TRUSTED)
    parser.add_argument("--excluded-groups", type=Path, default=DEFAULT_EXCLUSIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--policy")
    args = parser.parse_args()
    root = platform_root()
    benchmark = build_bundle(root).component("benchmark")
    governed_candidate = str(benchmark["paired_comparison"]["candidate_profile_id"])
    governed_baseline = str(benchmark["paired_comparison"]["baseline_profile_id"])
    policy_id = str(args.policy or governed_candidate)
    if policy_id not in {governed_candidate, governed_baseline}:
        parser.error("policy must be the governed candidate or baseline")
    revision, tree = _git_identity(root)
    value = evaluate(
        root=root,
        candidates_path=(root / args.candidates).resolve(),
        trusted_path=(root / args.trusted).resolve(),
        holdout_path=(root / args.holdout).resolve(),
        exclusions_path=(root / args.excluded_groups).resolve(),
        policy_id=policy_id,
        source_revision=revision,
        source_tree=tree,
    )
    output = (root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(output)


if __name__ == "__main__":
    main()
