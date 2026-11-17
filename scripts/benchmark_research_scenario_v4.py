from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from baloot_platform.config import build_bundle
from baloot_platform.filesystem import atomic_write_text
from baloot_platform.research.exact_state import ExactStateNormalizerV1
from baloot_platform.research.performance_stats import (
    qualified_scenario_identity_hash_v1,
)
from baloot_platform.research.proof_service import UnifiedScenarioProofServiceV1
from baloot_platform.research.recipe_v4 import ResearchRecipeV4
from baloot_platform.research.scenario_evidence import ScenarioRuntimeVerifierV1
from baloot_platform.research.scenario_result import (
    SCENARIO_ARTIFACT_MEDIA_TYPES,
    build_constraint_evidence_v1,
    build_scenario_result_v4,
    canonical_json_bytes,
)
from baloot_platform.research.scenario_contracts import (
    ResearchScenarioV1,
    ScenarioProofRequestV1,
    canonical_sha256,
)
from baloot_platform.research.scenario_search import ScenarioSearchExecutorV1
from baloot_platform.service.container import build_container

try:
    from scripts.generate_performance_evidence import git_identity
except ModuleNotFoundError:  # direct ``python scripts/...`` execution
    from generate_performance_evidence import git_identity


ROOT = Path(__file__).resolve().parents[1]


def _processor_affinity() -> list[int] | None:
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    if os.name != "nt":
        return None
    process_mask = ctypes.c_size_t()
    system_mask = ctypes.c_size_t()
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.GetProcessAffinityMask.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    )
    kernel32.GetProcessAffinityMask.restype = ctypes.c_int
    if not kernel32.GetProcessAffinityMask(
        kernel32.GetCurrentProcess(),
        ctypes.byref(process_mask),
        ctypes.byref(system_mask),
    ):
        raise OSError(ctypes.get_last_error(), "GetProcessAffinityMask failed")
    return [
        index
        for index in range(process_mask.value.bit_length())
        if process_mask.value & (1 << index)
    ]


def _predicate(predicate_id: str, kind: str, arguments: dict[str, Any]):
    return {
        "predicateId": predicate_id,
        "kind": kind,
        "version": 1,
        "hardness": "hard",
        "arguments": arguments,
    }


def _measure(operation: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    result = operation()
    return {
        "result": result,
        "wallSeconds": time.perf_counter() - wall_started,
        "processCpuSeconds": time.process_time() - cpu_started,
    }


def _measure_reproducible(
    operation: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    samples = [_measure(operation), _measure(operation)]
    semantic_hashes = [
        str(sample["result"]["semanticReceiptSha256"]) for sample in samples
    ]
    if len(set(semantic_hashes)) != 1:
        raise RuntimeError("Research V4 benchmark semantic replay was not reproducible")
    primary = samples[0]["result"]
    return {
        **primary,
        "wallSeconds": samples[0]["wallSeconds"],
        "cpuSeconds": samples[0]["processCpuSeconds"],
        "runtime": {
            "sampleCount": len(samples),
            "wallSeconds": [sample["wallSeconds"] for sample in samples],
            "processCpuSeconds": [sample["processCpuSeconds"] for sample in samples],
            "cpuMeasurementBoundary": "whole_process_cpu_v1",
            "auxiliaryThreadsIncluded": True,
            "childProcessesExcluded": True,
        },
        "reproducibility": {
            "repeatCount": len(samples),
            "scope": "semantic_outcome_excluding_resource_timings_v1",
            "matched": True,
            "semanticReceiptSha256": semantic_hashes[0],
            "fullReceiptSha256": [
                str(sample["result"]["receiptSha256"]) for sample in samples
            ],
            "fullReceiptByteIdentityExpected": False,
        },
    }


def _freeze_configuration_authority(
    root: Path,
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    with tempfile.TemporaryDirectory(
        prefix="research-v4-benchmark-configuration-",
        ignore_cleanup_errors=True,
    ) as temporary:
        container = build_container(root=root, writable_root=Path(temporary))
        try:
            selected = dict(container.configuration.resolve()["bundle"])
            authority, context = (
                container.research.execution_context_factory.freeze_selector(
                    bundle_revision_id=str(selected["revisionId"]),
                    analysis_mode="cascaded",
                )
            )
            authority_value = authority.to_dict()
            reference = dict(authority.snapshot_artifact)
            snapshot_bytes = container.learning.artifact_store.get(
                str(reference["uri"])
            )
            if hashlib.sha256(snapshot_bytes).hexdigest() != reference["sha256"]:
                raise RuntimeError("Research V4 configuration snapshot hash mismatch")
            snapshot = json.loads(snapshot_bytes.decode("utf-8"))
            if canonical_json_bytes(snapshot) != snapshot_bytes:
                raise RuntimeError(
                    "embedded Research V4 configuration snapshot is not canonical"
                )
            return authority_value, context.bundle, snapshot
        finally:
            container.shutdown()


def _recipe(
    *,
    code: str,
    title_ar: str,
    scenario: ResearchScenarioV1,
    proof: ScenarioProofRequestV1,
    configuration: dict[str, Any],
    resources: dict[str, Any],
) -> ResearchRecipeV4:
    qualification = {
        "minimumTruth": "robust_best_known",
        "minimumConfidence": 0.75,
        "minimumSurroundingQuality": 70.0,
        "requireObserverSafety": True,
        "requireAllDecisionAudit": True,
        "maximumRegret": 0.0,
    }
    diversity = {
        "schemaVersion": 1,
        "cellDimensions": ["contract", "project", "decision_kind", "proof_mode"],
        "topKPerCell": 3,
        "rejectedAuditModulus": 100,
    }
    render_policy = {"mode": "on_demand"}
    compatibility_fields = {
        "analysisMode",
        "candidatesPerSeed",
        "maximumConflicts",
    }
    legacy_resources = compatibility_fields.intersection(resources)
    authoring_resources = dict(resources)
    if legacy_resources:
        for field in compatibility_fields:
            authoring_resources.pop(field, None)
        authoring_resources["generationPolicy"] = "ortools_cp_sat_v3"
    recipe = ResearchRecipeV4.build(
        {
            "schemaVersion": 4,
            "code": code,
            "titleAr": title_ar,
            "configuration": configuration,
            "scenario": scenario.to_dict(),
            "proof": proof.to_dict(),
            "qualification": qualification,
            "resources": authoring_resources,
            "diversity": diversity,
            "renderPolicy": render_policy,
            "status": "published",
        }
    )
    if legacy_resources:
        frozen = recipe.to_dict()
        frozen["resources"] = dict(resources)
        frozen.pop("sha256")
        frozen["sha256"] = canonical_sha256(frozen)
        recipe = ResearchRecipeV4.build(frozen)
    if not recipe.capability_plan.valid:
        raise RuntimeError("frozen Research V4 benchmark recipe is invalid")
    return recipe


def _artifact_references(documents: dict[str, dict[str, Any]]) -> dict[str, Any]:
    references = {}
    for key, document in documents.items():
        payload = canonical_json_bytes(document)
        digest = canonical_sha256(document)
        references[key] = {
            "uri": f"cas://sha256/{digest}",
            "sha256": digest,
            "bytes": len(payload),
            "mediaType": SCENARIO_ARTIFACT_MEDIA_TYPES[key],
            "name": f"{digest}.json",
        }
    return references


def _qualified_result_receipt(
    *,
    recipe: ResearchRecipeV4,
    normalized: Any,
    proof: Any,
    satisfaction: Any,
    bundle_hash: str,
    candidate_hash: str,
    construction_receipt: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    constraint = build_constraint_evidence_v1(
        recipe,
        normalized,
        satisfaction,
        construction_receipt=construction_receipt,
    )
    documents = {
        "exactState": normalized.to_dict(),
        "scenarioSatisfaction": satisfaction.to_dict(),
        "constraintEvidence": constraint,
        "proofOutcome": proof.to_dict(),
    }
    artifacts = _artifact_references(documents)
    run_hash = canonical_sha256(
        {
            "kind": "research_v4_benchmark_run_v1",
            "recipeHash": recipe.sha256,
            "candidateHash": candidate_hash,
            "bundleHash": bundle_hash,
        }
    )
    result = build_scenario_result_v4(
        run={"runCode": f"RUN-{run_hash[:12].upper()}", "runHash": run_hash},
        recipe=recipe,
        normalized=normalized,
        satisfaction=satisfaction,
        proof=proof,
        constraint_evidence=constraint,
        artifacts=artifacts,
        created_at="1980-01-01T00:00:00+00:00",
    )
    receipt_core = {
        "schemaVersion": 1,
        "candidateHash": candidate_hash,
        "constructionReceiptHash": (
            str(construction_receipt["sha256"])
            if construction_receipt is not None
            else None
        ),
        "exactStateHash": normalized.sha256,
        "scenarioSatisfactionHash": satisfaction.sha256,
        "constraintEvidenceHash": constraint["sha256"],
        "proofOutcomeHash": proof.sha256,
        "artifactHashes": {
            key: value["sha256"] for key, value in sorted(artifacts.items())
        },
        "resultCode": result["resultCode"],
        "resultHash": result["resultHash"],
        "disposition": result["disposition"],
        "bundleHash": bundle_hash,
    }
    return result, {**receipt_core, "sha256": canonical_sha256(receipt_core)}


def _semantic_receipt(
    *,
    origin: str,
    normalized: Any,
    proof: Any,
    result: dict[str, Any],
    funnel: dict[str, int] | None,
    bundle_hash: str,
) -> str:
    payload = proof.payload
    return canonical_sha256(
        {
            "schemaVersion": 1,
            "origin": origin,
            "bundleHash": bundle_hash,
            "exactStateId": normalized.exact_state_id,
            "stateHash": normalized.state_hash,
            "rootActions": payload["rootActions"],
            "selectedAction": payload["selectedAction"],
            "winningRootActionCount": payload["winningRootActionCount"],
            "continuationLines": payload["continuationLines"],
            "completionStatus": payload["completionStatus"],
            "claimSatisfied": payload["claimSatisfied"],
            "principalVariation": payload["principalVariation"],
            "disposition": result["disposition"],
            "funnel": funnel,
        }
    )


def _exact_workload(
    *,
    fixture_path: Path,
    recipe_code: str,
    bundle: Any,
    configuration: dict[str, Any],
    service: UnifiedScenarioProofServiceV1,
) -> dict[str, Any]:
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    proof_request = ScenarioProofRequestV1.build(fixture["proof"])
    expected = dict(fixture["expected"])
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {
                "kind": "exact_trace_prefix",
                "exactTracePrefix": fixture["exactTracePrefix"],
            },
            "constraints": {
                "op": "all",
                "children": [
                    _predicate(
                        "PRED-BENCH-BUYER",
                        "auction.buyer",
                        {"seat": expected["buyer"]},
                    ),
                    _predicate(
                        "PRED-BENCH-CONTRACT",
                        "auction.contract",
                        {"contract": expected["contract"]},
                    ),
                ],
            },
            "decisionPoint": {"kind": "play", "selector": {}},
            "outcome": _predicate(
                "PRED-BENCH-ROOT-COUNT",
                "proof.winning_root_action_count",
                {
                    "minimum": int(expected["winningRootActionCount"]),
                    "maximum": int(expected["winningRootActionCount"]),
                },
            ),
        }
    )
    recipe = _recipe(
        code=recipe_code,
        title_ar=str(fixture["titleAr"]),
        scenario=scenario,
        proof=proof_request,
        configuration=configuration,
        resources={
            "generationLane": "ortools",
            "generationPolicy": "natural_policy",
            "seedStart": 0,
            "seedCount": 1,
            "seedsPerShard": 1,
            "candidatesPerSeed": 1,
            "solverTimeoutSeconds": 10.0,
            "maximumConflicts": 250000,
            "analysisMode": "cascaded",
            "proofBudgetProfile": proof_request.budget_profile,
        },
    )
    normalized = ExactStateNormalizerV1(bundle.rules).normalize(
        fixture["exactTracePrefix"],
        decision_point=recipe.scenario.decision_point,
    )
    proof = service.prove(
        normalized,
        proof_request,
        bundle_hash=bundle.sha256,
    )
    satisfaction = ScenarioRuntimeVerifierV1().verify(
        recipe.scenario,
        normalized,
        proof_outcome=proof,
    )
    candidate_hash = canonical_sha256(
        {
            "recipeHash": recipe.sha256,
            "exactStateHash": normalized.sha256,
            "proofOutcomeHash": proof.sha256,
            "scenarioSatisfactionHash": satisfaction.sha256,
        }
    )
    result, receipt = _qualified_result_receipt(
        recipe=recipe,
        normalized=normalized,
        proof=proof,
        satisfaction=satisfaction,
        bundle_hash=bundle.sha256,
        candidate_hash=candidate_hash,
    )
    payload = proof.payload
    budget = service.budgets.get(proof_request.budget_profile)
    return {
        "origin": "exact_input",
        "status": payload["completionStatus"],
        "scenarioTemplateId": result["scenarioProjection"]["scenarioTemplateId"],
        "exactStateId": normalized.exact_state_id,
        "legalRootActions": len(payload["rootActions"]),
        "winningRootActions": payload["winningRootActionCount"],
        "nodes": int(payload["resources"]["nodes"]),
        "cacheEntries": int(payload["resources"]["cacheEntries"]),
        "hiddenWorlds": int(payload["hiddenWorlds"]["count"]),
        "proofHash": proof.sha256,
        "proofBudget": {
            "profileId": budget.profile_id,
            "profileSha256": budget.sha256,
            "registrySha256": service.budgets.sha256,
        },
        "result": {
            "resultCode": result["resultCode"],
            "resultHash": result["resultHash"],
            "disposition": result["disposition"],
            "qualification": result["qualification"],
        },
        "scenarioIdentityHash": qualified_scenario_identity_hash_v1(result),
        "receipt": receipt,
        "receiptSha256": receipt["sha256"],
        "semanticReceiptSha256": _semantic_receipt(
            origin="exact_input",
            normalized=normalized,
            proof=proof,
            result=result,
            funnel=None,
            bundle_hash=bundle.sha256,
        ),
    }


def _generated_workload(
    bundle: Any,
    configuration: dict[str, Any],
    service: UnifiedScenarioProofServiceV1,
) -> dict[str, Any]:
    proof = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "exact",
            "visibility": "complete_state",
            "claim": "target_team_has_no_salvage",
            "targetTeam": "north_south",
            "utility": "contract_success",
            "successThreshold": 1,
            "lineUniverse": "all_legal",
            "budgetProfile": "late_game_exact_v1",
        }
    )
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": {
                "op": "all",
                "children": [
                    _predicate("PRED-BUYER", "auction.buyer", {"seat": "south"}),
                    _predicate(
                        "PRED-CONTRACT", "auction.contract", {"contract": "sun"}
                    ),
                    _predicate(
                        "PRED-PROJECT",
                        "project.exists",
                        {
                            "team": "north_south",
                            "owner": "south",
                            "kind": "hundred_sequence",
                            "value": 20,
                            "state": "awarded",
                        },
                    ),
                ],
            },
            "decisionPoint": {"kind": "play", "selector": {"remainingCards": 8}},
            "outcome": _predicate(
                "PRED-ROOT",
                "proof.winning_root_action_count",
                {"minimum": 0, "maximum": 0},
            ),
        }
    )
    recipe = _recipe(
        code="RCP-V4SUN100GEN",
        title_ar="سيناريو مولد لمشروع مية في الصن بلا إنقاذ",
        scenario=scenario,
        proof=proof,
        configuration=configuration,
        resources={
            "generationLane": "ortools",
            "generationPolicy": "scenario_guided_actor_safe",
            "seedStart": 12004,
            "seedCount": 1,
            "seedsPerShard": 1,
            "candidatesPerSeed": 1,
            "solverTimeoutSeconds": 10.0,
            "maximumConflicts": 250000,
            "analysisMode": "cascaded",
            "proofBudgetProfile": "late_game_exact_v1",
        },
    )
    candidate = ScenarioSearchExecutorV1(proof=service).build_candidate(
        recipe,
        rules=bundle.rules,
        bundle_hash=bundle.sha256,
        seed=12004,
    )
    if (
        candidate.normalized is None
        or candidate.proof is None
        or candidate.satisfaction is None
    ):
        raise RuntimeError("frozen generated V4 acceptance workload did not qualify")
    result, receipt = _qualified_result_receipt(
        recipe=recipe,
        normalized=candidate.normalized,
        proof=candidate.proof,
        satisfaction=candidate.satisfaction,
        bundle_hash=bundle.sha256,
        candidate_hash=candidate.sha256,
        construction_receipt=dict(candidate.construction_receipt),
    )
    structural = dict(candidate.construction_receipt.get("structural") or {})
    completion = str(candidate.proof.payload["completionStatus"])
    funnel = {
        "attempted": 1,
        "structurallyFeasible": int(structural.get("status") == "satisfied"),
        "legalReplay": int(candidate.normalized is not None),
        "semanticSatisfied": int(candidate.satisfaction.status == "satisfied"),
        "proofCompleted": int(completion != "proof_incomplete"),
        "qualified": int(result["disposition"] == "qualified"),
    }
    if result["disposition"] != "qualified":
        raise RuntimeError("frozen generated V4 acceptance workload did not qualify")
    budget = service.budgets.get(proof.budget_profile)
    return {
        "origin": "generated_search",
        "status": candidate.status,
        "seed": 12004,
        "scenarioTemplateId": result["scenarioProjection"]["scenarioTemplateId"],
        "exactStateId": candidate.normalized.exact_state_id,
        "winningRootActions": candidate.proof.payload["winningRootActionCount"],
        "nodes": int(candidate.proof.payload["resources"]["nodes"]),
        "cacheEntries": int(candidate.proof.payload["resources"]["cacheEntries"]),
        "hiddenWorlds": int(candidate.proof.payload["hiddenWorlds"]["count"]),
        "proofBudget": {
            "profileId": budget.profile_id,
            "profileSha256": budget.sha256,
            "registrySha256": service.budgets.sha256,
        },
        "structuralSolverStatus": candidate.construction_receipt["structural"][
            "solverStatus"
        ],
        "observerSafe": candidate.construction_receipt["policy"]["observerSafe"],
        "funnel": funnel,
        "result": {
            "resultCode": result["resultCode"],
            "resultHash": result["resultHash"],
            "disposition": result["disposition"],
            "qualification": result["qualification"],
        },
        "scenarioIdentityHash": qualified_scenario_identity_hash_v1(result),
        "receipt": receipt,
        "receiptSha256": receipt["sha256"],
        "semanticReceiptSha256": _semantic_receipt(
            origin="generated_search",
            normalized=candidate.normalized,
            proof=candidate.proof,
            result=result,
            funnel=funnel,
            bundle_hash=bundle.sha256,
        ),
    }


def run_benchmark(
    root: Path = ROOT,
    *,
    source_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    configuration, bundle, configuration_snapshot = _freeze_configuration_authority(
        root
    )
    source_bundle = build_bundle(root)
    source_components = {
        family: component.sha256
        for family, component in source_bundle.components.items()
    }
    published_components = {
        family: component.sha256
        for family, component in bundle.components.items()
    }
    if (
        source_bundle.rules.sha256 != bundle.rules.sha256
        or source_components != published_components
    ):
        mismatches = {
            family: {
                "source": source_components.get(family),
                "published": published_components.get(family),
            }
            for family in sorted(set(source_components) | set(published_components))
            if source_components.get(family) != published_components.get(family)
        }
        raise RuntimeError(
            "published Research configuration differs from the source bundle: "
            + json.dumps(
                {
                    "rules": {
                        "source": source_bundle.rules.sha256,
                        "published": bundle.rules.sha256,
                    },
                    "components": mismatches,
                },
                sort_keys=True,
            )
        )
    proof_service = UnifiedScenarioProofServiceV1.from_repository(root)
    workloads = {
        "exactNoSalvage": _measure_reproducible(
            lambda: _exact_workload(
                fixture_path=root / "examples/research-scenario-v4-golden.json",
                recipe_code="RCP-V4BENCHNS",
                bundle=bundle,
                configuration=configuration,
                service=proof_service,
            )
        ),
        "exactUniqueSavingMove": _measure_reproducible(
            lambda: _exact_workload(
                fixture_path=root / "examples/research-scenario-v4-unique-saving.json",
                recipe_code="RCP-V4BENCHUS",
                bundle=bundle,
                configuration=configuration,
                service=proof_service,
            )
        ),
        "generatedSunHundredNoSalvage": _measure_reproducible(
            lambda: _generated_workload(
                bundle,
                configuration,
                proof_service,
            )
        ),
    }
    total_cpu = sum(float(item["cpuSeconds"]) for item in workloads.values())
    total_wall = sum(float(item["wallSeconds"]) for item in workloads.values())
    generated = [
        item for item in workloads.values() if item["origin"] == "generated_search"
    ]
    qualified_result_hashes = {
        str(item["result"]["resultHash"])
        for item in generated
        if item["result"]["disposition"] == "qualified"
    }
    qualified_scenario_hashes = {
        str(item["scenarioIdentityHash"])
        for item in generated
        if item["result"]["disposition"] == "qualified"
    }
    generated_cpu = sum(float(item["cpuSeconds"]) for item in generated)
    funnel_keys = (
        "attempted",
        "structurallyFeasible",
        "legalReplay",
        "semanticSatisfied",
        "proofCompleted",
        "qualified",
    )
    generated_funnel = {
        key: sum(int(item["funnel"][key]) for item in generated) for key in funnel_keys
    }
    profile = dict(bundle.bundle_profile["profile"])
    proof_budget_profiles = {
        str(item["proofBudget"]["profileId"]): str(item["proofBudget"]["profileSha256"])
        for item in workloads.values()
    }
    evidence = {
        "schemaVersion": 2,
        "kind": "research_scenario_v4_performance",
        "status": "passed",
        "source": (git_identity(root) if source_identity is None else source_identity),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "logicalCpuCount": os.cpu_count(),
            "processorAffinity": _processor_affinity(),
            "nativeThreadLimits": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "BLIS_NUM_THREADS",
                    "ORTOOLS_NUM_THREADS",
                )
            },
            "cpuMeasurementBoundary": "whole_process_cpu_v1",
            "cpuMeasurementNote": (
                "time.process_time charges CPU consumed by this process, including "
                "its auxiliary threads; child-process CPU is excluded."
            ),
            "measurementExclusions": [
                "published_configuration_bootstrap",
                "configuration_snapshot_freeze",
                "proof_service_initialization",
            ],
        },
        "bundle": {
            "revisionId": configuration["bundleRevisionId"],
            "id": str(profile["id"]),
            "version": str(profile["version"]),
            "sha256": source_bundle.sha256,
            "semanticRuntimeSha256": bundle.sha256,
            "rulesSha256": bundle.rules.sha256,
            "componentSha256": {
                family: component.sha256
                for family, component in sorted(bundle.components.items())
            },
        },
        "configurationAuthority": configuration,
        "configurationSnapshot": configuration_snapshot,
        "proofBudgetProfileSha256": hashlib.sha256(
            (root / "configs/research/proof_budgets_v1.yaml").read_bytes()
        ).hexdigest(),
        "proofBudgetAuthority": {
            "registrySha256": proof_service.budgets.sha256,
            "profiles": dict(sorted(proof_budget_profiles.items())),
        },
        "workloads": workloads,
        "summary": {
            "workloadCount": len(workloads),
            "generatedFunnel": generated_funnel,
            "qualifiedGeneratedResultCount": generated_funnel["qualified"],
            "qualifiedDistinctGeneratedScenarioCount": len(qualified_scenario_hashes),
            "qualifiedDistinctGeneratedScenarioHashes": sorted(
                qualified_scenario_hashes
            ),
            "qualifiedDistinctGeneratedResultHashes": sorted(qualified_result_hashes),
            "totalWallSeconds": total_wall,
            "totalCpuSeconds": total_cpu,
            "measurementSampleScope": "first_sample_per_workload_v1",
            "generatedCpuSeconds": generated_cpu,
            "qualifiedDistinctGeneratedScenariosPerCpuHour": (
                len(qualified_scenario_hashes) * 3600.0 / generated_cpu
                if generated_cpu > 0
                else None
            ),
        },
        "proofBoundary": (
            "Two repeated local samples prove deterministic semantic outcomes for "
            "the frozen workloads. Resource timings and their hash-bound full receipts "
            "are intentionally sample-specific. The yield metric counts only distinct, "
            "fully qualified generated scenario identities and is not a cross-hardware claim."
        ),
    }
    evidence["evidenceSha256"] = canonical_sha256(evidence)
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Research V4 proof acceptance."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "release/research-scenario-v4-performance.json",
    )
    args = parser.parse_args()
    evidence = run_benchmark(ROOT)
    atomic_write_text(
        args.output,
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
