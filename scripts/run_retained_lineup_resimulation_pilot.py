from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from baloot_platform.agents.profiles import AgentProfileRegistry
from baloot_platform.challenge.simulation import NaturalLaffaSimulatorV1
from baloot_platform.challenge.trace import CompleteGameTraceV1
from baloot_platform.core.rules import RuleBook
from baloot_platform.research.contracts import ResearchRecipeV1, canonical_sha256
from baloot_platform.service.container import build_container


PROFILE_IDS = (
    "foundry_balanced",
    "expert_deterministic",
    "finalist_candidate",
)


def _cas_path(root: Path, digest: str) -> Path:
    return root / digest[:2] / digest[2:4] / digest


def _canonical_action(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _action_drift(parent: CompleteGameTraceV1, child: CompleteGameTraceV1) -> dict[str, Any]:
    left = parent.decision_ledger.records
    right = child.decision_ledger.records
    comparable = 0
    changed = 0
    first_divergence: int | None = None
    common_prefix = 0
    for before, after in zip(left, right, strict=False):
        identity_matches = (
            before.decision_kind == after.decision_kind
            and before.acting_seat == after.acting_seat
        )
        action_matches = _canonical_action(before.chosen_action) == _canonical_action(
            after.chosen_action
        )
        if first_divergence is None and identity_matches and action_matches:
            common_prefix += 1
        elif first_divergence is None:
            first_divergence = min(before.ordinal, after.ordinal)
        if identity_matches:
            comparable += 1
            changed += int(not action_matches)
    if first_divergence is None and len(left) != len(right):
        first_divergence = min(len(left), len(right)) + 1
    return {
        "parentDecisionCount": len(left),
        "childDecisionCount": len(right),
        "commonPrefixDecisions": common_prefix,
        "firstDivergenceOrdinal": first_divergence,
        "alignedDecisionCount": comparable,
        "alignedActionChanges": changed,
    }


def _select_rows(connection: sqlite3.Connection, *, per_profile: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_code, target in (
        ("RUN-2827A2AA10F0", "project_400"),
        ("RUN-FA2206765BD4", "kabout_opportunity"),
    ):
        candidates = []
        for row in connection.execute(
            """
            SELECT result_code, payload_json
            FROM research_results
            WHERE run_code = ?
            ORDER BY result_code
            """,
            (run_code,),
        ):
            payload = json.loads(row["payload_json"])
            quality = float(payload["qualification"]["minimumSurroundingQuality"])
            candidates.append((quality, str(row["result_code"]), payload))
        candidates.sort(key=lambda item: (item[0], item[1]))
        if len(candidates) <= per_profile:
            selected = candidates
        elif per_profile == 1:
            selected = [candidates[len(candidates) // 2]]
        else:
            indices = sorted(
                {
                    round(index * (len(candidates) - 1) / (per_profile - 1))
                    for index in range(per_profile)
                }
            )
            selected = [candidates[index] for index in indices]
        for quality, result_code, payload in selected:
            run = connection.execute(
                "SELECT recipe_code FROM research_runs WHERE run_code = ?",
                (run_code,),
            ).fetchone()
            recipe_row = connection.execute(
                "SELECT payload_json FROM research_recipes WHERE recipe_code = ?",
                (run["recipe_code"],),
            ).fetchone()
            rows.append(
                {
                    "target": target,
                    "runCode": run_code,
                    "resultCode": result_code,
                    "originalQuality": quality,
                    "result": payload,
                    "recipe": json.loads(recipe_row["payload_json"]),
                }
            )
    return rows


def _qualify(research: Any, trace: CompleteGameTraceV1, recipe: ResearchRecipeV1) -> tuple[dict[str, Any], Any]:
    analysis = research.analyzers["screening"].analyze_trace(trace)
    tactical = research._verify_tactics(  # noqa: SLF001 - qualification pilot uses the production gate
        trace,
        analysis,
        recipe=recipe,
        cancel_check=None,
    )
    audit = research._decision_audit(trace, analysis, tactical)  # noqa: SLF001
    qualification = research._qualification(recipe, audit, analysis)  # noqa: SLF001
    score = research.score_profile.score(analysis.laffa_evidence)
    failures = list(qualification["failures"])
    if not score.passed:
        failures.extend(
            f"score_profile_gate:{reason}" for reason in score.rejection_reasons
        )
    qualification = {
        **qualification,
        "qualified": not failures,
        "truthLabel": qualification["truthLabel"] if not failures else "rejected",
        "failures": failures,
        "governedScore": score.to_dict(),
    }
    return qualification, audit


def _rollup(rows: list[dict[str, Any]], profile_id: str) -> dict[str, Any]:
    selected = [row for row in rows if row["profileId"] == profile_id]
    walls = [float(row["wallSeconds"]) for row in selected]
    cpus = [float(row["cpuSeconds"]) for row in selected]
    qualities = [float(row["qualification"]["minimumSurroundingQuality"]) for row in selected]
    failures = Counter(
        failure
        for row in selected
        for failure in row["qualification"]["failures"]
    )
    worst = sorted(
        (
            {
                "resultCode": row["resultCode"],
                "target": row["target"],
                **decision,
            }
            for row in selected
            for decision in row["worstDecisions"]
        ),
        key=lambda item: (
            -float(item["expectedRegret"]),
            str(item["resultCode"]),
            int(item["ordinal"]),
        ),
    )[:5]
    return {
        "profileId": profile_id,
        "caseCount": len(selected),
        "qualifiedCount": sum(bool(row["qualification"]["qualified"]) for row in selected),
        "tacticalTargetCaseCount": sum(bool(row["requiredTactics"]) for row in selected),
        "tacticalProofCount": sum(
            row["targetTacticalProof"] is True for row in selected
        ),
        "surroundingQuality": {
            "minimum": min(qualities, default=None),
            "mean": statistics.fmean(qualities) if qualities else None,
            "maximum": max(qualities, default=None),
        },
        "wallSeconds": sum(walls),
        "cpuSeconds": sum(cpus),
        "medianCaseWallSeconds": statistics.median(walls) if walls else None,
        "actionDriftCaseCount": sum(
            row["actionDrift"]["firstDivergenceOrdinal"] is not None for row in selected
        ),
        "failureCounts": dict(sorted(failures.items())),
        "worstDecisions": worst,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--cas-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("src/baloot_platform/resources"))
    parser.add_argument("--per-target", type=int, default=4, choices=range(1, 11))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    connection = sqlite3.connect(args.database)
    connection.row_factory = sqlite3.Row
    cases = _select_rows(connection, per_profile=args.per_target)
    if not cases:
        raise SystemExit("no retained candidate rows found")

    runtime = Path(tempfile.mkdtemp(prefix="baloot-lineup-pilot-"))
    container = build_container(root=args.root.resolve(), writable_root=runtime)
    exact_search_backend = next(
        plugin
        for plugin in container.registry.all()
        if str(plugin.descriptor.kind) == "search"
    )
    rows: list[dict[str, Any]] = []
    try:
        for case in cases:
            result = case["result"]
            digest = str(result["artifacts"]["completeTrace"]["sha256"])
            parent = CompleteGameTraceV1.from_dict(
                json.loads(_cas_path(args.cas_root, digest).read_text(encoding="utf-8"))
            )
            snapshot = dict(parent.trusted_final_state["configurationSnapshot"])
            components = dict(snapshot["components"])
            rules = RuleBook(dict(components["rules"]["payload"]))
            registry = AgentProfileRegistry.from_mapping(components["agents"]["payload"])
            recipe = ResearchRecipeV1.build(case["recipe"])
            for profile_id in PROFILE_IDS:
                simulator = NaturalLaffaSimulatorV1(
                    rules,
                    components["conventions"]["payload"],
                    bundle_hash=parent.bundle_hash,
                    configuration_snapshot=snapshot,
                    agent_profile=registry.get(profile_id),
                    generation_tier="balanced_discovery",
                    exact_search_backend=exact_search_backend,
                )
                cpu_started = time.process_time()
                wall_started = time.perf_counter()
                child = simulator.resimulate_constructed_play_as_counterfactual_descendant(
                    parent
                )
                qualification, audit = _qualify(container.research, child, recipe)
                wall_seconds = time.perf_counter() - wall_started
                cpu_seconds = time.process_time() - cpu_started
                regrets = [
                    float(item["expectedRegret"])
                    for item in audit["decisions"]
                    if not bool(item["forced"])
                ]
                worst_decisions = sorted(
                    (
                        {
                            "ordinal": int(item["ordinal"]),
                            "decisionKind": str(item["decisionKind"]),
                            "actingSeat": str(item["actingSeat"]),
                            "actualAction": dict(item["actualAction"]),
                            "recommendedAction": dict(item["recommendedAction"]),
                            "expectedRegret": float(item["expectedRegret"]),
                            "probabilityDrop": float(item["probabilityDrop"]),
                            "confidence": float(item["confidence"]),
                            "truthLabel": str(item["truthLabel"]),
                        }
                        for item in audit["decisions"]
                        if not bool(item["forced"])
                    ),
                    key=lambda item: (-item["expectedRegret"], item["ordinal"]),
                )[:3]
                required_tactics = [
                    str(item["kind"])
                    for item in recipe.tactics
                    if item["mode"] in {"hard", "qualification_gate"}
                ]
                tactical_gate_failed = any(
                    str(failure).startswith("tactical_gate_not_proven:")
                    for failure in qualification["failures"]
                )
                row = {
                    "target": case["target"],
                    "resultCode": case["resultCode"],
                    "originalConstructedQualification": dict(
                        result["qualification"]
                    ),
                    "parentTraceHash": parent.sha256,
                    "childTraceHash": child.sha256,
                    "deckCommitment": child.seed_provenance["deckCommitment"],
                    "profileId": profile_id,
                    "profileHash": simulator.profile_hash,
                    "agentRelease": simulator.agent_release,
                    "wallSeconds": wall_seconds,
                    "cpuSeconds": cpu_seconds,
                    "maximumDecisionRegret": max(regrets, default=0.0),
                    "meanDecisionRegret": statistics.fmean(regrets) if regrets else 0.0,
                    "worstDecisions": worst_decisions,
                    "requiredTactics": required_tactics,
                    "targetTacticalProof": (
                        not tactical_gate_failed if required_tactics else None
                    ),
                    "qualification": qualification,
                    "actionDrift": _action_drift(parent, child),
                }
                rows.append(row)
                print(
                    json.dumps(
                        {
                            "case": case["resultCode"],
                            "target": case["target"],
                            "profile": profile_id,
                            "quality": qualification["minimumSurroundingQuality"],
                            "qualified": qualification["qualified"],
                            "wallSeconds": round(wall_seconds, 3),
                            "firstDrift": row["actionDrift"]["firstDivergenceOrdinal"],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        container.shutdown()
        connection.close()

    payload: dict[str, Any] = {
        "schemaVersion": 1,
        "pilotId": "retained_constructed_deal_lineup_resimulation",
        "pilotRevision": 1,
        "sourceDatabase": args.database.name,
        "sourceCasDirectory": args.cas_root.name,
        "caseSelection": "deterministic_quality_quantiles",
        "casesPerTarget": args.per_target,
        "profileIds": list(PROFILE_IDS),
        "constructionChanged": False,
        "qualificationAuthority": "production_actor_safe_screening_and_tactical_gates",
        "host": {
            "platform": os.name,
            "processorCount": os.cpu_count(),
        },
        "rows": rows,
        "rollups": [_rollup(rows, profile_id) for profile_id in PROFILE_IDS],
    }
    payload["sha256"] = canonical_sha256(payload)
    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
