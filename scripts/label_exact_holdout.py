#!/usr/bin/env python3
# ruff: noqa: E402
"""Attach deterministic clone-minimax labels to tractable positions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY_ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from baloot_platform.analysis.belief_v7 import BeliefPolicy
from baloot_platform.analysis.hidden_deals import generate_hidden_deals
from baloot_platform.analysis.models import InformationState
from baloot_platform.analysis.search import exact_team_value
from baloot_platform.benchmark import (
    benchmark_configuration_sha256,
    canonical_sha256,
    load_candidate_jsonl,
    reconstruct_state,
    validate_trusted_sidecar,
)
from baloot_platform.config import build_bundle, platform_root
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.scenario import ScenarioRequest


DEFAULT_CANDIDATES = Path(
    "private-inputs/strength-candidates.jsonl"
)
DEFAULT_TRUSTED = Path(
    ".release-runtime/strength-holdout/trusted-replay-recipes.json"
)
DEFAULT_OUTPUT = Path(
    ".release-runtime/strength-holdout/exact-labels.json"
)
DEFAULT_ARCHETYPES = ("endgame", "outcome_flip")
ORACLE_ID = "observer-safe-exhaustive-exact-search-v1"


def information_set_action_values(
    state,
    engine: BalootEngine,
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if state.current_player is None:
        raise ValueError("exact information-set label requires an actor")
    observer = state.current_player
    info = InformationState.from_complete(state, observer)
    belief_policy = BeliefPolicy.from_profile()
    request = ScenarioRequest.governed()
    deals = generate_hidden_deals(
        info,
        engine,
        requested_samples=1,
        exact_allocation_limit=belief_policy.exact_allocation_limit,
        maximum_attempts=request.maximum_sampling_attempts,
        seed=seed,
    )
    if deals.method not in {
        "exhaustive_constrained_hidden_deals",
        "exhaustive_project_conditioned_hidden_deals",
    }:
        raise ValueError(
            "exact label requires exhaustive compatible hidden deals"
        )
    legal = tuple(engine.legal_actions(deals.states[0]))
    legal_ids = tuple(card.id for card in legal)
    margins: dict[int, list[float]] = {
        card.id: [] for card in legal
    }
    nodes = 0
    for compatible in deals.states:
        if tuple(
            card.id for card in engine.legal_actions(compatible)
        ) != legal_ids:
            raise AssertionError(
                "root legal actions differ across compatible deals"
            )
        for card in legal:
            token = engine.apply(
                compatible,
                card,
                verify_identity=False,
            )
            try:
                result = exact_team_value(
                    compatible,
                    engine,
                    observer.team,
                    max_nodes=request.exact_search_max_nodes,
                    max_seconds=request.exact_search_max_seconds,
                    max_depth=request.exact_search_max_depth,
                    max_cache_entries=(
                        request.exact_search_max_cache_entries
                    ),
                )
            finally:
                engine.undo(compatible, token)
            if result.completion_status != "exact":
                raise RuntimeError(
                    "information-set continuation was "
                    f"{result.completion_status}"
                )
            margins[card.id].append(float(result.value))
            nodes += int(result.nodes)
    values = [
        {
            "action": card.code,
            "expectedTeamMargin": (
                sum(margins[card.id]) / len(margins[card.id])
            ),
        }
        for card in legal
    ]
    return values, {
        "method": deals.method,
        "compatibleDealCount": len(deals.states),
        "attemptedAllocations": deals.attempted,
        "nodes": nodes,
        "exactAllocationLimit": belief_policy.exact_allocation_limit,
        "searchBudget": {
            "maxNodes": request.exact_search_max_nodes,
            "maxSecondsSafetyCancellation": (
                request.exact_search_max_seconds
            ),
            "maxDepth": request.exact_search_max_depth,
            "maxCacheEntries": request.exact_search_max_cache_entries,
        },
    }


def label_exact_positions(
    candidates: dict[str, Any],
    trusted: dict[str, Any],
    *,
    root: Path,
    archetypes: tuple[str, ...] = DEFAULT_ARCHETYPES,
    candidate_ids: set[str] | None = None,
) -> dict[str, Any]:
    validate_trusted_sidecar(candidates, trusted)
    bundle = build_bundle(root)
    configuration_sha256 = benchmark_configuration_sha256(bundle)
    if bundle.rules.sha256 != str(candidates["rulesSha256"]):
        raise ValueError("candidate rules identity is stale")
    if configuration_sha256 != str(
        candidates["configurationSha256"]
    ):
        raise ValueError("candidate configuration identity is stale")
    engine = BalootEngine(bundle.rules)
    recipes = dict(trusted["recipes"])
    selected = [
        dict(item)
        for item in candidates["positions"]
        if str(item.get("archetype", "")) in set(archetypes)
        and (
            candidate_ids is None
            or str(item.get("candidateId", "")) in candidate_ids
        )
    ]
    if not selected:
        raise ValueError("no tractable exact-label candidates selected")
    authority_sha256 = canonical_sha256(
        {
            "oracleId": ORACLE_ID,
            "rulesSha256": bundle.rules.sha256,
            "configurationSha256": configuration_sha256,
            "algorithm": (
                "enumerate every hard-deduction-compatible hidden deal from "
                "the observer information state; evaluate every legal action "
                "on the identical cohort with exact team continuation"
            ),
        }
    )
    labels = []
    for candidate in sorted(
        selected,
        key=lambda item: str(item["candidateId"]),
    ):
        candidate_id = str(candidate["candidateId"])
        recipe = dict(recipes.get(candidate_id) or {})
        if not recipe:
            raise ValueError("trusted replay recipe is missing")
        state = reconstruct_state(engine, candidate, recipe)
        values, information_set = information_set_action_values(
            state,
            engine,
            seed=21_000_000 + len(labels),
        )
        best = max(
            float(item["expectedTeamMargin"]) for item in values
        )
        reference_actions = sorted(
            str(item["action"])
            for item in values
            if float(item["expectedTeamMargin"]) == best
        )
        labels.append(
            {
                "candidateId": candidate_id,
                "stateSha256": candidate["stateSha256"],
                "observer": candidate["observer"],
                "legalActions": list(candidate["legalActions"]),
                "referenceActions": reference_actions,
                "actionValues": sorted(
                    values,
                    key=lambda item: str(item["action"]),
                ),
                "labelSource": "exact",
                "labelAuthorityId": ORACLE_ID,
                "labelAuthoritySha256": authority_sha256,
                "searchCompletion": "exact",
                "informationSet": information_set,
                "nodes": information_set["nodes"],
            }
        )
    value = {
        "schemaVersion": 1,
        "candidateDatasetId": candidates["datasetId"],
        "candidateDatasetSha256": candidates["datasetSha256"],
        "rulesSha256": bundle.rules.sha256,
        "configurationSha256": configuration_sha256,
        "oracleId": ORACLE_ID,
        "oracleAuthoritySha256": authority_sha256,
        "exactLabelCount": len(labels),
        "tieCount": sum(
            len(item["referenceActions"]) > 1 for item in labels
        ),
        "positions": labels,
        "proofBoundary": (
            "Deterministic local software-oracle evidence under the declared "
            "uniform exhaustive compatible-deal prior and authoritative exact "
            "continuation. It does not reveal or optimize for the actual "
            "hidden deal and is not external Baloot certification, human "
            "adjudication, equilibrium proof, or a strength claim."
        ),
    }
    value["evidenceSha256"] = canonical_sha256(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--trusted", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--archetype",
        action="append",
        choices=DEFAULT_ARCHETYPES,
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = platform_root()
    candidate_path = args.candidates or root / DEFAULT_CANDIDATES
    trusted_path = args.trusted or root / DEFAULT_TRUSTED
    output = args.output or root / DEFAULT_OUTPUT
    candidates = load_candidate_jsonl(candidate_path)
    trusted = json.loads(trusted_path.read_text(encoding="utf-8"))
    value = label_exact_positions(
        candidates,
        trusted,
        root=root,
        archetypes=tuple(args.archetype or DEFAULT_ARCHETYPES),
    )
    text = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    if args.check:
        if not output.exists() or output.read_text(encoding="utf-8") != text:
            raise SystemExit("exact holdout labels are stale")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8", newline="\n")
    print(output)


if __name__ == "__main__":
    main()
