#!/usr/bin/env python3
"""Compare current observer-safe policies on protected exact labels."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
import math
from pathlib import Path
import platform
import re
import statistics
import subprocess
import time
from typing import Any

import yaml

from baloot_platform.agents.profiles import (
    AdaptivePlayStrategy,
    AgentProfileRegistry,
)
from baloot_platform.analysis.search_errors import require_uncensored_play_trace
from baloot_platform.benchmark import (
    benchmark_configuration_sha256,
    canonical_sha256,
    load_candidate_jsonl,
    reconstruct_state,
    validate_trusted_sidecar,
)
from baloot_platform.config import build_bundle, platform_root
from baloot_platform.core.bots import BotProfile, HeuristicBot
from baloot_platform.core.engine import BalootEngine
from baloot_platform.filesystem import atomic_write_text


DEFAULT_CANDIDATES = Path(
    "private-inputs/strength-candidates.jsonl"
)
DEFAULT_TRUSTED = Path(
    ".release-runtime/strength-holdout/trusted-replay-recipes.json"
)
DEFAULT_EXACT = Path(
    ".release-runtime/strength-holdout/exact-labels.json"
)
DEFAULT_OUTPUT = Path(
    "release/intelligence-benchmark.json"
)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    point = (len(ordered) - 1) * quantile
    lower = int(math.floor(point))
    upper = int(math.ceil(point))
    if lower == upper:
        return ordered[lower]
    fraction = point - lower
    return (
        ordered[lower] * (1.0 - fraction)
        + ordered[upper] * fraction
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("policy preanalysis requires decisions")
    hits = sum(bool(item["hit"]) for item in rows)
    latencies = [float(item["latencyMs"]) for item in rows]
    regrets = [float(item["regret"]) for item in rows]
    sources = Counter(str(item["decisionSource"]) for item in rows)
    archetypes: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        archetypes.setdefault(str(row["archetype"]), []).append(row)
    return {
        "positions": len(rows),
        "top1Accuracy": hits / len(rows),
        "expectedRegret": statistics.fmean(regrets),
        "worstRegret": max(regrets),
        "illegalActionRate": (
            sum(bool(item["illegal"]) for item in rows) / len(rows)
        ),
        "variationAppliedCount": sum(
            bool(item.get("variationApplied", False)) for item in rows
        ),
        "latencyMs": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
        },
        "decisionSources": dict(sorted(sources.items())),
        "archetypes": {
            key: {
                "positions": len(items),
                "top1Accuracy": (
                    sum(bool(item["hit"]) for item in items) / len(items)
                ),
                "expectedRegret": statistics.fmean(
                    float(item["regret"]) for item in items
                ),
            }
            for key, items in sorted(archetypes.items())
        },
    }


def _code(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("code", ""))
    return str(value)


def run_preanalysis(
    candidates: dict[str, Any],
    trusted: dict[str, Any],
    exact: dict[str, Any],
    *,
    root: Path,
    policy_ids: tuple[str, ...],
    source_revision: str | None = None,
    source_tree: str | None = None,
    workspace_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    validate_trusted_sidecar(candidates, trusted)
    if (
        str(exact.get("candidateDatasetId", ""))
        != str(candidates["datasetId"])
        or str(exact.get("candidateDatasetSha256", ""))
        != str(candidates["datasetSha256"])
    ):
        raise ValueError("exact label dataset identity mismatch")
    evidence_hash = str(exact.get("evidenceSha256", ""))
    exact_payload = {
        key: value
        for key, value in exact.items()
        if key != "evidenceSha256"
    }
    if evidence_hash != canonical_sha256(exact_payload):
        raise ValueError("exact label evidence hash mismatch")
    candidate_by_id = {
        str(item["candidateId"]): dict(item)
        for item in candidates["positions"]
    }
    recipes = dict(trusted["recipes"])
    labels = [dict(item) for item in exact.get("positions") or ()]
    if int(exact.get("exactLabelCount", -1)) != len(labels):
        raise ValueError("exact label count mismatch")
    bundle = build_bundle(root)
    engine = BalootEngine(bundle.rules)
    registry = AgentProfileRegistry.from_bundle(root)
    convention = yaml.safe_load(
        (
            root / "configs/conventions/conservative_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    reports: dict[str, Any] = {}
    for policy_id in policy_ids:
        if policy_id == "heuristic":
            policy_snapshot = {
                "id": "transparent_heuristic",
                "version": "1",
            }
            strategy = None
        else:
            source_profile = (
                "solver_fast"
                if policy_id in {
                    "solver_16_sampled",
                    "solver_8",
                    "solver_4",
                }
                else policy_id
            )
            profile = registry.get(source_profile)
            if policy_id in {
                "solver_16_sampled",
                "solver_8",
                "solver_4",
            }:
                profile = replace(
                    profile,
                    id=policy_id,
                    play_search_samples=(
                        16
                        if policy_id == "solver_16_sampled"
                        else 8
                        if policy_id == "solver_8"
                        else 4
                    ),
                    play_exact_allocation_limit=1,
                )
            policy_snapshot = profile.to_dict()
            strategy = AdaptivePlayStrategy(profile, convention)
        decision_rows = []
        for label in labels:
            candidate_id = str(label["candidateId"])
            candidate = candidate_by_id[candidate_id]
            state = reconstruct_state(
                engine,
                candidate,
                dict(recipes[candidate_id]),
            )
            seed = int(
                canonical_sha256(
                    {
                        "candidateId": candidate_id,
                        "policyId": policy_id,
                    }
                )[:16],
                16,
            )
            started = time.perf_counter()
            if strategy is None:
                card = HeuristicBot(
                    engine,
                    BotProfile(deterministic_seed=seed),
                ).choose(state)
                ranked = [card.code]
                source = "transparent_heuristic"
                variation_applied = False
            else:
                card, trace = strategy.choose_with_trace(
                    engine,
                    state,
                    seed=seed,
                )
                require_uncensored_play_trace(trace)
                ranked = [
                    _code(item.get("action"))
                    for item in trace["rankedActions"]
                ]
                source = str(trace["decisionSource"])
                variation_applied = bool(
                    trace["mistakeApplied"]
                    or trace["deceptionApplied"]
                )
            latency_ms = (time.perf_counter() - started) * 1000.0
            legal = set(str(item) for item in candidate["legalActions"])
            values = {
                str(item["action"]): float(
                    item["expectedTeamMargin"]
                )
                for item in label["actionValues"]
            }
            best = max(values.values())
            predicted = card.code
            regret = (
                best - values[predicted]
                if predicted in values
                else best - min(values.values()) + 1.0
            )
            decision_rows.append(
                {
                    "archetype": candidate["archetype"],
                    "hit": predicted in set(label["referenceActions"]),
                    "regret": regret,
                    "illegal": predicted not in legal,
                    "latencyMs": latency_ms,
                    "decisionSource": source,
                    "variationApplied": variation_applied,
                    "rankedLegal": (
                        bool(ranked)
                        and all(item in legal for item in ranked)
                    ),
                }
            )
        report = summarize(decision_rows)
        report.update(
            {
                "policyId": policy_id,
                "policySnapshotSha256": canonical_sha256(
                    policy_snapshot
                ),
                "allRankedActionsLegal": all(
                    bool(item["rankedLegal"]) for item in decision_rows
                ),
            }
        )
        reports[policy_id] = report
    result = {
        "schemaVersion": 2,
        "release": "BalootX",
        "status": "passed",
        "kind": "observer_safe_exact_subset_preanalysis",
        "source": {
            "revision": source_revision,
            "tree": source_tree,
            "workspaceManifestSha256": workspace_manifest_sha256,
            "binding": (
                "exact_git_head"
                if source_revision and source_tree
                else "local_workspace_manifest"
                if workspace_manifest_sha256
                else "unbound_test_invocation"
            ),
        },
        "candidateDatasetId": candidates["datasetId"],
        "candidateDatasetSha256": candidates["datasetSha256"],
        "exactEvidenceSha256": evidence_hash,
        "rulesSha256": bundle.rules.sha256,
        "configurationSha256": benchmark_configuration_sha256(bundle),
        "policies": dict(sorted(reports.items())),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "automaticPromotion": False,
        "promotionDecision": {
            "decision": "not_eligible",
            "reason": (
                "The independently adjudicated permanent holdout and paired "
                "comparison remain required for promotion."
            ),
        },
        "proofBoundary": (
            "Protected 120-position tractable endgame/outcome-flip "
            "preanalysis under exhaustive compatible-deal exact labels. "
            "It is not the 500-position permanent holdout, paired match "
            "evidence, cross-hardware performance proof, human validation, "
            "or a release promotion decision."
        ),
    }
    result["semanticResultSha256"] = canonical_sha256(
        {
            "candidateDatasetSha256": result[
                "candidateDatasetSha256"
            ],
            "exactEvidenceSha256": evidence_hash,
            "policies": {
                key: {
                    item_key: item_value
                    for item_key, item_value in value.items()
                    if item_key != "latencyMs"
                }
                for key, value in result["policies"].items()
            },
        }
    )
    return result


def git_identity(root: Path, output: Path) -> tuple[str, str]:
    relative_output = output.resolve().relative_to(root.resolve()).as_posix()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    unexpected = [
        line
        for line in status
        if line[3:].replace("\\", "/") != relative_output
    ]
    if unexpected:
        raise RuntimeError(
            "policy preanalysis requires a clean source worktree; "
            f"unexpected changes: {unexpected}"
        )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return revision, tree


def source_manifest_identity(path: Path) -> tuple[str, str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    source = dict(manifest.get("source") or {})
    snapshot = dict(source.get("worktreeSnapshot") or {})
    revision = str(source.get("revision") or "")
    if (
        manifest.get("schemaVersion") != 1
        or source.get("binding") != "git_head_plus_worktree_snapshot"
        or re.fullmatch(r"[0-9a-f]{40}", revision) is None
        or re.fullmatch(r"[0-9a-f]{40}", str(source.get("tree") or "")) is None
        or re.fullmatch(r"[0-9a-f]{64}", str(snapshot.get("sha256") or ""))
        is None
    ):
        raise ValueError("source manifest identity is invalid")
    return revision, str(snapshot["sha256"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--trusted", type=Path)
    parser.add_argument("--exact-labels", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument(
        "--policy",
        action="append",
        choices=(
            "heuristic",
            "strong",
            "expert",
            "expert_deterministic",
            "finalist_candidate",
            "solver_fast",
            "solver_16_sampled",
            "solver_8",
            "solver_4",
        ),
    )
    args = parser.parse_args()
    root = platform_root()
    output = (args.output or root / DEFAULT_OUTPUT).resolve()
    if args.source_manifest:
        source_revision, workspace_manifest_sha256 = source_manifest_identity(
            args.source_manifest.resolve()
        )
        source_tree = None
    else:
        source_revision, source_tree = git_identity(root, output)
        workspace_manifest_sha256 = None
    result = run_preanalysis(
        load_candidate_jsonl(
            args.candidates or root / DEFAULT_CANDIDATES
        ),
        json.loads(
            (args.trusted or root / DEFAULT_TRUSTED).read_text(
                encoding="utf-8"
            )
        ),
        json.loads(
            (args.exact_labels or root / DEFAULT_EXACT).read_text(
                encoding="utf-8"
            )
        ),
        root=root,
        policy_ids=tuple(
            args.policy or ("heuristic", "strong", "expert")
        ),
        source_revision=source_revision,
        source_tree=source_tree,
        workspace_manifest_sha256=workspace_manifest_sha256,
    )
    atomic_write_text(
        output,
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    print(output)


if __name__ == "__main__":
    main()
