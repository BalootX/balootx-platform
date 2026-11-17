from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import sys
import threading
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from baloot_platform.config import build_bundle
from baloot_platform.filesystem import atomic_write_text
from baloot_platform import __version__
from baloot_platform.research.campaign_coordinator import (
    CampaignBudgetV1,
    CampaignShardOutcomeV1,
    DeterministicCpSatCampaignCoordinatorV1,
)
from baloot_platform.research.ortools_exact_state import (
    OrToolsExactStateNormalizerV1,
)
from baloot_platform.research.ortools_research_executor import (
    OrToolsUnifiedResearchExecutorV1,
)
from baloot_platform.research.result_authority import ExecutionReceiptV1
from baloot_platform.research.recipe_v4 import ResearchRecipeV4
from baloot_platform.research.runtime_compatibility import (
    check_runtime_compatibility_v1,
    require_runtime_compatibility_passed_v1,
)
from baloot_platform.research.scenario_contracts import (
    ResearchScenarioV1,
    ScenarioProofRequestV1,
    canonical_sha256,
)
from baloot_platform.research.sharding import partition_research_shards
from baloot_platform.research.performance_stats import (
    process_lifetime_peak_rss_bytes,
)
from baloot_platform.service.container import build_container

try:
    from scripts.generate_performance_evidence import git_identity
except ModuleNotFoundError:  # direct ``python scripts/...`` execution
    from generate_performance_evidence import git_identity


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/research/ortools_performance_benchmark_v2.yaml"
OUTPUT = ROOT / "release/ortools-research-v3-performance.json"
_PROFILE_BUNDLE_BY_VERSION = {
    "1.1.0": {
        "id": "baloot_ortools_research_native_v3",
        "source": "configs/bundle_ortools_research_native_v3.yaml",
        "backend": "ortools_cp_sat_v3",
    },
    "2.0.0": {
        "id": "baloot_ortools_research_native_v3",
        "source": "configs/bundle_intelligent_research.yaml",
        "backend": "ortools_cp_sat_v3",
    },
}
_OPERATIONS = (
    "validationEstimate",
    "originalFlagshipContradiction",
    "exactNormalization",
    "fixedAlternateProof",
    "alternateFirstQualifiedResult",
    "alternateFiveVariants",
    "cancellationAcknowledgement",
)
_BUDGET_FIELDS = {
    "validationEstimate": "validationEstimateP95Ms",
    "originalFlagshipContradiction": "originalFlagshipContradictionP95Ms",
    "exactNormalization": "exactNormalizationP95Ms",
    "fixedAlternateProof": "fixedAlternateProofP95Ms",
    "alternateFirstQualifiedResult": "alternateFirstQualifiedResultP95Ms",
    "alternateFiveVariants": "alternateFiveVariantsP95Ms",
    "cancellationAcknowledgement": "cancellationAcknowledgementP95Ms",
}
_REQUIRED_STAGE_TIMINGS = {
    "validationEstimate": frozenset({"validation", "estimate", "total"}),
    "originalFlagshipContradiction": frozenset(
        {"validation", "estimate", "preflight", "total"}
    ),
    "exactNormalization": frozenset({"exactNormalization", "total"}),
    "fixedAlternateProof": frozenset({"graph", "proof", "total"}),
    "alternateFirstQualifiedResult": frozenset(
        {
            "validation",
            "estimate",
            "construction",
            "tailConstruction",
            "proofBoundary",
            "prefixLift",
            "proof",
            "qualification",
            "compatibility",
            "total",
        }
    ),
    "alternateFiveVariants": frozenset(
        {
            "validation",
            "estimate",
            "construction",
            "tailConstruction",
            "proofBoundary",
            "prefixLift",
            "proof",
            "qualification",
            "compatibility",
            "total",
        }
    ),
    "cancellationAcknowledgement": frozenset(
        {"cancellationAcknowledgement", "total"}
    ),
}
_REQUIRED_COMPLETED_VARIANTS = {
    "validationEstimate": 0,
    "originalFlagshipContradiction": 0,
    "exactNormalization": 1,
    "fixedAlternateProof": 1,
    "alternateFirstQualifiedResult": 1,
    "alternateFiveVariants": 5,
    "cancellationAcknowledgement": 0,
}


@dataclass(frozen=True, slots=True)
class BenchmarkOperationV1:
    factory: Callable[[], Callable[[], Mapping[str, Any]]]


@dataclass(frozen=True, slots=True)
class BenchmarkRecipeSourceV1:
    raw: Mapping[str, Any]
    scenario: ResearchScenarioV1
    proof: ScenarioProofRequestV1
    identity: Mapping[str, Any]


def _load_profile(path: Path = PROFILE) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "schemaVersion",
        "profile",
        "bundle",
        "originalFlagshipRequest",
        "alternateVariantRecipe",
        "measurement",
        "budgets",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("OR-Tools benchmark profile shape is invalid")
    profile = copy.deepcopy(dict(value))
    if profile["schemaVersion"] != 1:
        raise ValueError("OR-Tools benchmark profile schemaVersion must be 1")
    identity = profile["profile"]
    if (
        not isinstance(identity, Mapping)
        or set(identity) != {"id", "version", "status"}
        or identity["id"] != "ortools_research_v4_reference_benchmark"
        or identity["status"] != "governed_candidate"
        or identity["version"] not in _PROFILE_BUNDLE_BY_VERSION
    ):
        raise ValueError("OR-Tools benchmark profile identity is unsupported")
    bundle = profile["bundle"]
    if not isinstance(bundle, Mapping) or set(bundle) != {"id", "source", "backend"}:
        raise ValueError("OR-Tools benchmark bundle selection is invalid")
    if dict(bundle) != _PROFILE_BUNDLE_BY_VERSION[str(identity["version"])]:
        raise ValueError("OR-Tools benchmark must select the published V4 bundle")
    original = profile["originalFlagshipRequest"]
    if not isinstance(original, Mapping) or dict(original) != {
        "source": "examples/research-scenario-v4-flagship-exact-infeasible.json",
        "expectedStatus": "exact_structural_contradiction",
        "theoremId": "FLAGSHIP-TWO-ACES-TRICK7-LEAD-V1",
    }:
        raise ValueError("OR-Tools benchmark original flagship request is invalid")
    alternate = profile["alternateVariantRecipe"]
    if not isinstance(alternate, Mapping) or dict(alternate) != {
        "source": (
            "examples/research-scenario-v4-flagship-"
            "alternate-target-wins-six-search-five.json"
        ),
        "relationToOriginal": "target_team_wins_trick_six",
    }:
        raise ValueError("OR-Tools benchmark alternate recipe is invalid")
    measurement = profile["measurement"]
    expected_measurement = {
        "coldSamples",
        "warmSamples",
        "warmupIterations",
        "percentileMethod",
        "coldDefinition",
        "warmDefinition",
    }
    if not isinstance(measurement, Mapping) or set(measurement) != expected_measurement:
        raise ValueError("OR-Tools benchmark measurement policy is invalid")
    if measurement["percentileMethod"] != "linear_interpolation_v1":
        raise ValueError("OR-Tools benchmark percentile method is unsupported")
    if measurement["coldDefinition"] != "fresh_operation_instance_same_process_v1":
        raise ValueError("OR-Tools benchmark cold definition is unsupported")
    if measurement["warmDefinition"] != "reused_operation_instance_after_warmup_v1":
        raise ValueError("OR-Tools benchmark warm definition is unsupported")
    for field in ("coldSamples", "warmSamples"):
        if type(measurement[field]) is not int or measurement[field] < 2:
            raise ValueError(f"OR-Tools benchmark {field} must be at least two")
    if type(measurement["warmupIterations"]) is not int or measurement[
        "warmupIterations"
    ] < 0:
        raise ValueError("OR-Tools benchmark warmupIterations must be non-negative")
    budgets = profile["budgets"]
    if not isinstance(budgets, Mapping) or set(budgets) != set(_BUDGET_FIELDS.values()):
        raise ValueError("OR-Tools benchmark budgets are incomplete")
    if any(type(value) not in {int, float} or value <= 0 for value in budgets.values()):
        raise ValueError("OR-Tools benchmark budgets must be positive")
    return profile


def _load_benchmark_recipe_source(
    root: Path,
    source_name: str,
    *,
    alternate: bool,
) -> BenchmarkRecipeSourceV1:
    source = root / source_name
    if not source.is_file():
        raise FileNotFoundError(
            "canonical flagship benchmark recipe is missing: " + source.as_posix()
        )
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise TypeError("recipe must be an object")
        scenario = ResearchScenarioV1.build(dict(raw.get("scenario") or {}))
        proof = ScenarioProofRequestV1.build(dict(raw.get("proof") or {}))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("canonical flagship benchmark recipe is invalid") from exc
    expected_output = (
        {"kind": "laffa_collection", "resultCount": 5}
        if alternate
        else {"kind": "laffa", "resultCount": 1}
    )
    expected_render = (
        {"mode": "on_demand", "videoCount": 5}
        if alternate
        else {"mode": "disabled", "videoCount": 0}
    )
    if (
        raw.get("schemaVersion") != 4
        or raw.get("status") != "published"
        or raw.get("output") != expected_output
        or raw.get("renderPolicy") != expected_render
        or scenario.origin.kind != "search"
        or proof.mode != "exact"
        or proof.visibility != "complete_state"
        or proof.claim != "winning_root_action_count"
        or proof.target_team != "west_east"
        or proof.utility != "round_win"
        or proof.success_threshold != 1
    ):
        raise ValueError("canonical flagship benchmark contract is unsupported")
    selector = dict(scenario.decision_point.get("selector") or {})
    if scenario.decision_point.get("kind") != "play" or selector != {
        "trickIndex": 5,
        "position": 3,
        "remainingCards": 3,
        "actingSeat": "west",
    }:
        raise ValueError("canonical flagship benchmark decision is unsupported")
    predicates = tuple(
        predicate
        for predicate in scenario.predicates()
        if predicate.hardness == "hard"
    )

    def has(kind: str, expected: Mapping[str, Any]) -> bool:
        return any(
            predicate.kind == kind
            and all(
                predicate.arguments.get(key) == value
                for key, value in expected.items()
            )
            for predicate in predicates
        )

    low_selector = {"ranks": ["seven", "eight", "nine"]}
    actor_ace = {"ranks": ["ace"], "owner": {"relation": "actor"}}
    common = (
        has("auction.contract", {"contract": "sun"}),
        has("auction.buyer", {"seat": "north"}),
        has(
            "project.exists",
            {
                "team": "north_south",
                "owner": "north",
                "kind": "hundred_sequence",
                "value": 20,
                "state": "awarded",
            },
        ),
        has(
            "play.action_matches",
            {
                "trickIndex": 5,
                "position": 3,
                "actingSeat": "west",
                "selector": actor_ace,
            },
        ),
        (
            has(
                "play.trick_winner_matches",
                {
                    "trickIndex": 5,
                    "seat": "east",
                    "selector": low_selector,
                },
            )
            if alternate
            else has(
                "play.trick_winner_matches",
                {
                    "trickIndex": 5,
                    "team": "north_south",
                    "selector": low_selector,
                },
            )
        ),
        all(
            has(
                "play.trick_winner_matches",
                {
                    "trickIndex": trick,
                    "team": "west_east",
                    "selector": low_selector,
                },
            )
            for trick in (6, 7)
        ),
        has(
            "play.trick_card_count",
            {
                "trickIndex": 7,
                "selector": {
                    "ranks": ["ace"],
                    "owner": {"relation": "team", "team": "north_south"},
                },
                "minimum": 2,
                "maximum": 2,
            },
        ),
        has("proof.winning_root_action_matches", {"selector": actor_ace}),
        has(
            "proof.winning_root_action_count",
            {"minimum": 1, "maximum": 1},
        ),
        has("outcome.team_round", {"team": "west_east", "status": "won"}),
    )
    if not all(common):
        raise ValueError("canonical flagship benchmark semantics are incomplete")
    identity = {
        "source": source_name,
        "sourceSha256": canonical_sha256(raw),
        "recipeCode": str(raw.get("code") or ""),
        "scenarioHash": scenario.sha256,
        "proofHash": proof.sha256,
        "classification": (
            "alternate_target_team_wins_trick_six"
            if alternate
            else "original_exact_structural_contradiction"
        ),
    }
    return BenchmarkRecipeSourceV1(
        copy.deepcopy(dict(raw)), scenario, proof, identity
    )


def _validate_benchmark_recipe_identities(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    identities = copy.deepcopy(dict(value))
    if set(identities) != {"original", "alternate"}:
        raise ValueError("flagship benchmark recipe identities are incomplete")
    expected = {
        "original": (
            "examples/research-scenario-v4-flagship-exact-infeasible.json",
            "original_exact_structural_contradiction",
        ),
        "alternate": (
            "examples/research-scenario-v4-flagship-"
            "alternate-target-wins-six-search-five.json",
            "alternate_target_team_wins_trick_six",
        ),
    }
    for name, (source, classification) in expected.items():
        identity = dict(identities[name])
        if set(identity) != {
            "source",
            "sourceSha256",
            "recipeCode",
            "scenarioHash",
            "proofHash",
            "classification",
        }:
            raise ValueError("flagship benchmark recipe identity shape is invalid")
        if identity["source"] != source or identity["classification"] != classification:
            raise ValueError("flagship benchmark recipe identity is mislabeled")
        if not str(identity["recipeCode"]).startswith("RCP-"):
            raise ValueError("flagship benchmark recipe code is invalid")
        for field in ("sourceSha256", "scenarioHash", "proofHash"):
            digest = identity[field]
            if type(digest) is not str or len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"flagship benchmark {field} is invalid")
    return identities


def _governed_resource_limits(
    bundle: Any, profile_id: str = "late_game_exact_v1"
) -> dict[str, Any]:
    profile = bundle.component("challenge")["scenarioResearchV4"]
    search = profile["searchBudgets"]["profiles"].get(profile_id)
    if not isinstance(search, Mapping):
        raise ValueError("OR-Tools benchmark resource profile is unavailable")
    cache = profile["cacheLimits"]
    return {
        "profileId": profile_id,
        "maximumConflicts": int(search["maximumConflicts"]),
        "maximumMemoryBytes": int(search["maximumMemoryBytes"]),
        "maximumCacheBytes": int(cache["maximumBytes"]),
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _distribution(values: list[float]) -> dict[str, Any]:
    return {
        "sampleCount": len(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "minimum": min(values),
        "maximum": max(values),
        "percentileMethod": "linear_interpolation_v1",
    }


def _measure_once(
    name: str,
    operation: Callable[[], Mapping[str, Any]],
) -> dict[str, Any]:
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    result = copy.deepcopy(dict(operation()))
    wall_seconds = max(0.0, time.perf_counter() - wall_started)
    cpu_seconds = max(0.0, time.process_time() - cpu_started)
    semantic_hash = str(result.get("semanticHash") or "")
    context_hash = str(result.get("contextHash") or "")
    if len(semantic_hash) != 64 or len(context_hash) != 64:
        raise ValueError(f"{name} must return semantic and context hashes")
    resources = copy.deepcopy(dict(result.get("resources") or {}))
    normalization_authority: dict[str, str] = {}
    if name == "exactNormalization":
        normalization_authority = {
            field: str(resources.get(field) or "")
            for field in (
                "normalizationBackendId",
                "normalizationReceiptHash",
                "normalizationModelHash",
            )
        }
        if (
            not normalization_authority["normalizationBackendId"]
            or any(
                len(normalization_authority[field]) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in normalization_authority[field]
                )
                for field in ("normalizationReceiptHash", "normalizationModelHash")
            )
        ):
            raise ValueError("exactNormalization authority evidence is invalid")
    for field in (
        "conflicts",
        "branches",
        "cacheBytes",
        "completedCandidates",
        "completedVariants",
    ):
        value = resources.get(field, 0)
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} resource {field} must be non-negative")
    timings = copy.deepcopy(dict(result.get("stageTimings") or {}))
    if set(timings) != set(_REQUIRED_STAGE_TIMINGS[name]):
        raise ValueError(f"{name} stage timing set is incomplete")
    for stage, duration in timings.items():
        if (
            type(duration) not in {int, float}
            or not math.isfinite(float(duration))
            or duration < 0
        ):
            raise ValueError(f"{name} stage timing {stage} is invalid")
    peak = process_lifetime_peak_rss_bytes() or 0
    measurement_ms = float(result.get("measurementMs", timings["total"] * 1000.0))
    if not math.isfinite(measurement_ms) or measurement_ms < 0:
        raise ValueError(f"{name} returned an invalid measured duration")
    receipt = ExecutionReceiptV1.build(
        semantic_result_hash=semantic_hash,
        stage_timings=timings,
        cpu_seconds=cpu_seconds,
        peak_memory_bytes=peak,
        conflicts=int(resources.get("conflicts", 0)),
        branches=int(resources.get("branches", 0)),
        cache_bytes=int(resources.get("cacheBytes", 0)),
        completed_variants=int(resources.get("completedVariants", 0)),
        machine={
            "system": platform.system(),
            "machine": platform.machine(),
            "contextHash": context_hash,
            **normalization_authority,
        },
    ).to_dict()
    return {
        "measurementMs": measurement_ms,
        "wallMs": wall_seconds * 1000.0,
        "cpuMs": cpu_seconds * 1000.0,
        "semanticHash": semantic_hash,
        "contextHash": context_hash,
        "resources": resources,
        "executionReceipt": receipt,
    }


def _measure_operation(
    name: str,
    operation: BenchmarkOperationV1,
    *,
    cold_samples: int,
    warm_samples: int,
    warmups: int,
    budget_ms: float,
    resource_limits: Mapping[str, int],
) -> dict[str, Any]:
    cold = [_measure_once(name, operation.factory()) for _ in range(cold_samples)]
    warm_operation = operation.factory()
    for _ in range(warmups):
        warm_operation()
    warm = [_measure_once(name, warm_operation) for _ in range(warm_samples)]
    all_samples = [*cold, *warm]
    semantic_hashes = sorted({item["semanticHash"] for item in all_samples})
    context_hashes = sorted({item["contextHash"] for item in all_samples})
    if len(semantic_hashes) != 1 or len(context_hashes) != 1:
        diagnostic = []
        for phase, samples in (("cold", cold), ("warm", warm)):
            for index, item in enumerate(samples):
                campaign = dict(item["resources"].get("campaignReceipt") or {})
                coordinator = dict(campaign.get("coordinatorReceipt") or {})
                diagnostic.append(
                    {
                        "phase": phase,
                        "index": index,
                        "semanticHash": item["semanticHash"],
                        "contextHash": item["contextHash"],
                        "acceptedSemanticHashes": campaign.get(
                            "acceptedSemanticHashes"
                        ),
                        "executionSemanticHashes": campaign.get(
                            "executionSemanticHashes"
                        ),
                        "noGoodHashes": [
                            canonical_sha256(value)
                            for value in campaign.get("noGoodVectors") or ()
                        ],
                        "coordinatorAttempts": [
                            {
                                "shardIndex": attempt.get("shardIndex"),
                                "status": attempt.get("status"),
                                "errorCode": attempt.get("errorCode"),
                                "variantHashes": attempt.get("variantHashes"),
                                "partialEvidence": {
                                    key: partial[key]
                                    for key in (
                                        "currentStage",
                                        "errorCode",
                                        "campaignStatus",
                                        "candidateBudgetExhausted",
                                    )
                                    if key in partial
                                },
                            }
                            for attempt in coordinator.get("attempts") or ()
                            for partial in (
                                dict(attempt.get("partialEvidence") or {}),
                            )
                        ],
                    }
                )
        raise RuntimeError(
            f"{name} semantic or context identity drifted: "
            + json.dumps(diagnostic, sort_keys=True, separators=(",", ":"))
        )
    cold_ms = [float(item["measurementMs"]) for item in cold]
    warm_ms = [float(item["measurementMs"]) for item in warm]
    p95_ms = max(_percentile(cold_ms, 0.95), _percentile(warm_ms, 0.95))
    resource_rows = [dict(item["resources"]) for item in all_samples]
    resources = {
        field: max(int(row.get(field, 0)) for row in resource_rows)
        for field in (
            "conflicts",
            "branches",
            "cacheBytes",
            "completedCandidates",
            "completedVariants",
        )
    }
    resources["peakMemoryBytes"] = max(
        int(item["executionReceipt"]["peakMemoryBytes"]) for item in all_samples
    )
    if name == "cancellationAcknowledgement":
        statuses = {str(row.get("cancellationStatus") or "") for row in resource_rows}
        if statuses != {"cancelled"}:
            raise RuntimeError("cancellation acknowledgement outcome drifted")
        resources["cancellationStatus"] = "cancelled"
    required_completed = _REQUIRED_COMPLETED_VARIANTS[name]
    resource_gate = {
        "limits": {
            "profileId": str(resource_limits["profileId"]),
            "maximumConflicts": int(resource_limits["maximumConflicts"]),
            "maximumMemoryBytes": int(resource_limits["maximumMemoryBytes"]),
            "maximumCacheBytes": int(resource_limits["maximumCacheBytes"]),
            "requiredCompletedVariants": required_completed,
        },
        "observed": {
            "conflicts": resources["conflicts"],
            "branches": resources["branches"],
            "peakMemoryBytes": resources["peakMemoryBytes"],
            "cacheBytes": resources["cacheBytes"],
            "completedCandidates": resources["completedCandidates"],
            "completedVariants": resources["completedVariants"],
        },
    }
    resource_gate["passed"] = (
        resources["conflicts"] <= int(resource_limits["maximumConflicts"])
        and resources["peakMemoryBytes"]
        <= int(resource_limits["maximumMemoryBytes"])
        and resources["cacheBytes"] <= int(resource_limits["maximumCacheBytes"])
        and resources["completedVariants"] == required_completed
    )
    core = {
        "schemaVersion": 1,
        "semanticHash": semantic_hashes[0],
        "contextHash": context_hashes[0],
        "budgetMs": budget_ms,
        "passed": p95_ms <= budget_ms and resource_gate["passed"],
        "cold": {
            "latencyMs": _distribution(cold_ms),
            "wallMs": _distribution([float(item["wallMs"]) for item in cold]),
            "cpuMs": _distribution([float(item["cpuMs"]) for item in cold]),
            "stageTimingsSeconds": {
                stage: _distribution(
                    [
                        float(item["executionReceipt"]["stageTimings"][stage])
                        for item in cold
                    ]
                )
                for stage in sorted(_REQUIRED_STAGE_TIMINGS[name])
            },
            "executionReceiptHashes": [
                item["executionReceipt"]["sha256"] for item in cold
            ],
            "samples": cold,
        },
        "warm": {
            "latencyMs": _distribution(warm_ms),
            "wallMs": _distribution([float(item["wallMs"]) for item in warm]),
            "cpuMs": _distribution([float(item["cpuMs"]) for item in warm]),
            "stageTimingsSeconds": {
                stage: _distribution(
                    [
                        float(item["executionReceipt"]["stageTimings"][stage])
                        for item in warm
                    ]
                )
                for stage in sorted(_REQUIRED_STAGE_TIMINGS[name])
            },
            "executionReceiptHashes": [
                item["executionReceipt"]["sha256"] for item in warm
            ],
            "samples": warm,
        },
        "resources": resources,
        "resourceGate": resource_gate,
    }
    return {**core, "sha256": canonical_sha256(core)}


def _predicate(
    predicate_id: str,
    kind: str,
    arguments: Mapping[str, Any],
    *,
    hardness: str = "hard",
) -> dict[str, Any]:
    return {
        "predicateId": predicate_id,
        "kind": kind,
        "version": 1,
        "hardness": hardness,
        "arguments": copy.deepcopy(dict(arguments)),
    }


def _publish_explicit_bundle(container: Any, root: Path, profile: Mapping[str, Any]):
    source = str(profile["bundle"]["source"])
    packaged = build_bundle(root, bundle_path=source)
    resolved = container.configuration.bootstrap_packaged_bundle(
        actor_id="ORTOOLS-V4-PERFORMANCE",
        bundle_path=source,
        scope="environment",
        environment="ortools-v4-performance",
    )
    authority, context = container.research.execution_context_factory.freeze_selector(
        bundle_revision_id=str(resolved["bundle"]["revisionId"]),
        analysis_mode="cascaded",
    )
    expected_profile = dict(packaged.bundle_profile.get("profile") or {})
    frozen_profile = dict(context.bundle.bundle_profile.get("profile") or {})
    expected_components = {
        family: component.sha256
        for family, component in packaged.components.items()
    }
    frozen_components = {
        family: component.sha256
        for family, component in context.bundle.components.items()
    }
    if (
        frozen_profile.get("id") != expected_profile.get("id")
        or frozen_profile.get("version") != expected_profile.get("version")
        or context.bundle.mode != packaged.mode
        or context.bundle.rules.sha256 != packaged.rules.sha256
        or frozen_components != expected_components
    ):
        raise RuntimeError("published performance bundle identity mismatch")
    challenge = context.bundle.component("challenge")["scenarioResearchV4"]
    if challenge["backend"] != profile["bundle"]["backend"]:
        raise RuntimeError("published performance bundle backend mismatch")
    return authority, context


class _RealOperationsV1:
    def __init__(
        self,
        root: Path,
        profile: Mapping[str, Any],
        original_request: BenchmarkRecipeSourceV1,
        alternate_recipe: BenchmarkRecipeSourceV1,
    ) -> None:
        import tempfile

        self._temporary = tempfile.TemporaryDirectory(
            prefix="balootx-ortools-v4-performance-",
            ignore_cleanup_errors=True,
        )
        self.container = build_container(
            root=root,
            writable_root=Path(self._temporary.name),
        )
        self.authority, self.context = _publish_explicit_bundle(
            self.container, root, profile
        )
        self.configuration = self.authority.to_dict()
        self.context_hash = canonical_sha256(
            {
                "configurationAuthorityHash": self.authority.sha256,
                "researchContextHash": self.context.context_hash,
                "bundleRevisionId": self.context.bundle_revision_id,
                "bundleHash": self.context.bundle.sha256,
                "backend": str(profile["bundle"]["backend"]),
            }
        )
        research_profile = self.context.bundle.component("challenge")[
            "scenarioResearchV4"
        ]
        alternate_resources = dict(alternate_recipe.raw["resources"])
        self.search_profile_id = str(alternate_resources["proofBudgetProfile"])
        self.proof_budget = dict(
            research_profile["proofBudgets"]["profiles"][self.search_profile_id]
        )
        self.search_budget = dict(
            research_profile["searchBudgets"]["profiles"][self.search_profile_id]
        )
        self.exact_proof_budget = dict(
            research_profile["proofBudgets"]["profiles"]["late_game_exact_v1"]
        )
        self.exact_search_budget = dict(
            research_profile["searchBudgets"]["profiles"]["late_game_exact_v1"]
        )
        self.maximum_line_count_limbs = int(
            research_profile["refusalThresholds"]["maximumLineCountLimbs"]
        )
        self.minimum_decision_state_distance = int(
            research_profile["diversityDefaults"][
                "minimumDecisionStateHammingDistance"
            ]
        )
        for source in (original_request, alternate_recipe):
            resources = dict(source.raw.get("resources") or {})
            profile_id = resources.get("proofBudgetProfile")
            search_budget = research_profile["searchBudgets"]["profiles"].get(
                profile_id
            )
            if (
                resources.get("generationLane") != "ortools"
                or resources.get("generationPolicy") != "ortools_cp_sat_v3"
                or search_budget is None
                or float(resources.get("solverTimeoutSeconds", 0))
                > float((search_budget or {})["maximumWallSeconds"])
                or int(resources.get("maximumConflicts", 0))
                > int((search_budget or {})["maximumConflicts"])
            ):
                raise ValueError(
                    "flagship benchmark recipe exceeds the governed exact tier"
                )
        self.original_payload = self._bound_recipe(original_request)
        self.search_one = self._bound_recipe(alternate_recipe, result_count=1)
        self.search_five = self._bound_recipe(alternate_recipe, result_count=5)
        fixture = json.loads(
            (
                root
                / "examples/research-scenario-v4-ace-sacrifice-two-aces.json"
            ).read_text(encoding="utf-8")
        )
        self.exact_payload = {
            "schemaVersion": 4,
            "titleAr": str(fixture["titleAr"]),
            "configuration": copy.deepcopy(self.configuration),
            "scenario": {
                "schemaVersion": 1,
                "origin": {
                    "kind": "exact_trace_prefix",
                    "exactTracePrefix": copy.deepcopy(fixture["exactTracePrefix"]),
                },
                "constraints": _predicate(
                    "PRED-BENCH-EXACT-PHASE",
                    "state.phase",
                    {"state": "playing"},
                ),
                "decisionPoint": {
                    "kind": "play",
                    "selector": {"trickIndex": 6, "actingSeat": "west"},
                },
                "outcome": _predicate(
                    "PRED-BENCH-EXACT-ACTOR",
                    "state.next_actor",
                    {"seat": "west"},
                ),
            },
            "proof": copy.deepcopy(fixture["proof"]),
            "resources": {
                "generationLane": "ortools",
                "generationPolicy": "ortools_cp_sat_v3",
                "seedStart": 1,
                "seedCount": 1,
                "seedsPerShard": 1,
                "solverTimeoutSeconds": float(
                    self.exact_search_budget["maximumWallSeconds"]
                ),
                "proofBudgetProfile": "late_game_exact_v1",
            },
            "status": "published",
        }
        # Recipe validation is measured independently. Exact normalization begins
        # with the already-validated typed scenario used by the service boundary.
        self.exact_scenario = ResearchScenarioV1.build(self.exact_payload["scenario"])

    def _bound_recipe(
        self,
        source: BenchmarkRecipeSourceV1,
        *,
        result_count: int | None = None,
    ) -> dict[str, Any]:
        payload = copy.deepcopy(dict(source.raw))
        payload.pop("sha256", None)
        payload.pop("capabilityPlan", None)
        payload["configuration"] = copy.deepcopy(self.configuration)
        if result_count is not None:
            payload["output"] = {
                "kind": "laffa" if result_count == 1 else "laffa_collection",
                "resultCount": result_count,
            }
            payload["renderPolicy"] = {
                "mode": "on_demand",
                "videoCount": result_count,
            }
        return ResearchRecipeV4.build(payload).to_dict()

    def close(self) -> None:
        self.container.shutdown()
        self._temporary.cleanup()

    def _executor(self) -> OrToolsUnifiedResearchExecutorV1:
        return OrToolsUnifiedResearchExecutorV1(
            self.context.bundle.rules,
            bundle_revision_id=self.context.bundle_revision_id,
            bundle_hash=self.context.bundle.sha256,
            backend_id="ortools_cp_sat_v3",
        )

    def validation_estimate(self) -> Mapping[str, Any]:
        payload = copy.deepcopy(self.search_one)
        total_started = time.perf_counter()
        stage_started = time.perf_counter()
        validated = self.container.research.validate_recipe(payload)
        validation_seconds = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        estimate = self.container.research.estimate(payload)
        estimate_seconds = time.perf_counter() - stage_started
        if not validated["valid"]:
            raise RuntimeError("alternate benchmark recipe did not validate")
        semantic = canonical_sha256(
            {"validated": validated, "estimate": estimate}
        )
        return {
            "semanticHash": semantic,
            "contextHash": self.context_hash,
            "stageTimings": {
                "validation": validation_seconds,
                "estimate": estimate_seconds,
                "total": time.perf_counter() - total_started,
            },
            "resources": {"completedVariants": 0},
        }

    def original_flagship_contradiction(
        self,
        executor: OrToolsUnifiedResearchExecutorV1 | None = None,
    ) -> Mapping[str, Any]:
        started = time.perf_counter()
        stage_started = time.perf_counter()
        validation = self.container.research.validate_recipe(self.original_payload)
        validation_seconds = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        estimate = self.container.research.estimate(self.original_payload)
        estimate_seconds = time.perf_counter() - stage_started
        structural_issue = next(
            (
                item
                for item in validation.get("issues", ())
                if item.get("code") == "exact_structural_infeasibility"
            ),
            None,
        )
        validation_theorem = (
            (
                ((structural_issue or {}).get("details") or {}).get(
                    "structuralInfeasibility"
                )
                or {}
            ).get("theorem")
            or {}
        )
        if (
            validation.get("valid") is not False
            or structural_issue is None
            or structural_issue.get("class") != "structurally_infeasible"
            or validation_theorem.get("theoremId")
            != "FLAGSHIP-TWO-ACES-TRICK7-LEAD-V1"
            or estimate.get("valid") is not False
            or estimate.get("launchAdmissible") is not False
            or estimate.get("refusalReason")
            != "exact_structural_infeasibility"
            or estimate.get("exhaustiveSearchInfeasibilityEstablishable") is not True
        ):
            raise RuntimeError(
                "original flagship validation/estimate did not fail closed"
            )
        scenario = ResearchScenarioV1.build(self.original_payload["scenario"])
        proof = ScenarioProofRequestV1.build(self.original_payload["proof"])
        result = (executor or self._executor()).execute(
            scenario,
            proof,
            maximum_proof_nodes=int(self.exact_proof_budget["maximumNodes"]),
            maximum_proof_depth=int(self.exact_proof_budget["maximumDepth"]),
            maximum_hidden_worlds=int(
                self.exact_proof_budget["maximumHiddenWorlds"]
            ),
            maximum_line_count_limbs=self.maximum_line_count_limbs,
            maximum_candidates=1,
            maximum_seconds=float(
                self.original_payload["resources"]["solverTimeoutSeconds"]
            ),
        )
        receipt = dict(result.execution_receipt)
        contradiction = dict(receipt.get("structuralInfeasibility") or {})
        theorem = dict(contradiction.get("theorem") or {})
        if (
            result.status != "infeasible"
            or result.candidate is not None
            or result.proof is not None
            or contradiction.get("status") != "exact_structural_infeasibility"
            or contradiction.get("stage") != "preflight"
            or theorem.get("status") != "exact_structural_contradiction"
            or theorem.get("theoremId")
            != "FLAGSHIP-TWO-ACES-TRICK7-LEAD-V1"
            or receipt.get("completedCandidates") != 0
        ):
            raise RuntimeError("original flagship contradiction was not proven exactly")
        preflight_seconds = float(receipt["stageTimings"]["total"])
        return {
            "semanticHash": result.semantic_hash,
            "contextHash": self.context_hash,
            "stageTimings": {
                "validation": validation_seconds,
                "estimate": estimate_seconds,
                "preflight": preflight_seconds,
                "total": time.perf_counter() - started,
            },
            "resources": {
                "completedVariants": 0,
                "conflicts": 0,
                "branches": 0,
                "cacheBytes": 0,
                "executorReceiptHash": receipt["sha256"],
                "contradictionReceiptHash": contradiction["sha256"],
                "theoremReceiptHash": theorem["sha256"],
                "validationHash": canonical_sha256(validation),
                "estimateHash": canonical_sha256(estimate),
            },
        }

    def exact_normalization(
        self, normalizer: OrToolsExactStateNormalizerV1 | None = None
    ) -> Mapping[str, Any]:
        started = time.perf_counter()
        normalized = (normalizer or OrToolsExactStateNormalizerV1(
            self.context.bundle.rules, native_cp_sat=True
        )).normalize(
            self.exact_scenario.origin.exact_trace_prefix,
            decision_point=self.exact_scenario.decision_point,
            maximum_seconds=60.0,
        )
        receipt = normalized.normalization_receipt.to_dict()
        backend_id = str(
            self.context.bundle.component("challenge")["scenarioResearchV4"][
                "backend"
            ]
        )
        if receipt.get("backendId") != backend_id:
            raise RuntimeError("exact normalization backend does not match the bundle")
        return {
            "semanticHash": normalized.sha256,
            "contextHash": self.context_hash,
            "stageTimings": {
                "exactNormalization": time.perf_counter() - started,
                "total": time.perf_counter() - started,
            },
            "resources": {
                "completedVariants": 1,
                "normalizationBackendId": backend_id,
                "normalizationReceiptHash": receipt["sha256"],
                "normalizationModelHash": receipt["modelHash"],
            },
        }

    def fixed_alternate_proof(
        self, executor: OrToolsUnifiedResearchExecutorV1 | None = None
    ) -> Mapping[str, Any]:
        total_started = time.perf_counter()
        proof = ScenarioProofRequestV1.build(self.exact_payload["proof"])
        result = (executor or self._executor()).execute(
            self.exact_scenario,
            proof,
            maximum_proof_nodes=int(self.exact_proof_budget["maximumNodes"]),
            maximum_proof_depth=int(self.exact_proof_budget["maximumDepth"]),
            maximum_hidden_worlds=int(
                self.exact_proof_budget["maximumHiddenWorlds"]
            ),
            maximum_line_count_limbs=self.maximum_line_count_limbs,
            maximum_candidates=1,
            maximum_seconds=float(self.exact_search_budget["maximumWallSeconds"]),
        )
        if (
            result.status != "qualified"
            or result.proof is None
            or result.proof.get("completionStatus") != "exact_complete"
            or result.proof.get("claimSatisfied") is not True
            or result.proof.get("winningRootActionCount") != 1
        ):
            raise RuntimeError("fixed alternate proof did not qualify exactly")
        mapped = self._executor_result(result, completed_variants=1)
        receipt_timings = dict(result.execution_receipt["stageTimings"])
        proof_seconds = float(receipt_timings["graph"]) + float(
            receipt_timings["proof"]
        )
        mapped["measurementMs"] = proof_seconds * 1000.0
        mapped["stageTimings"] = {
            "graph": float(receipt_timings["graph"]),
            "proof": float(receipt_timings["proof"]),
            "total": proof_seconds,
        }
        mapped["resources"]["endToEndWallSeconds"] = (
            time.perf_counter() - total_started
        )
        return mapped

    def _search(
        self,
        *,
        count: int,
        executor: OrToolsUnifiedResearchExecutorV1 | None = None,
    ) -> Mapping[str, Any]:
        payload = self.search_one if count == 1 else self.search_five
        total_started = time.perf_counter()
        stage_started = time.perf_counter()
        validated = self.container.research.validate_recipe(payload)
        validation_seconds = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        estimate = self.container.research.estimate(payload)
        estimate_seconds = time.perf_counter() - stage_started
        if not validated["valid"] or not estimate["launchAdmissible"]:
            raise RuntimeError("alternate benchmark recipe was not launch-admissible")
        scenario = ResearchScenarioV1.build(payload["scenario"])
        proof = ScenarioProofRequestV1.build(payload["proof"])
        executor = executor or self._executor()
        if count == 1:
            resources = dict(payload["resources"])
            result = executor.execute(
                scenario,
                proof,
                maximum_proof_nodes=int(self.proof_budget["maximumNodes"]),
                maximum_proof_depth=int(self.proof_budget["maximumDepth"]),
                maximum_hidden_worlds=int(
                    self.proof_budget["maximumHiddenWorlds"]
                ),
                maximum_line_count_limbs=self.maximum_line_count_limbs,
                maximum_candidates=int(resources["seedCount"]),
                maximum_prefix_lift_attempts=int(
                    self.search_budget["maximumPrefixLiftAttempts"]
                ),
                maximum_seconds=float(resources["solverTimeoutSeconds"]),
            )
            if not self._is_qualified_alternate(result):
                raise RuntimeError(
                    f"alternate first-result benchmark ended as {result.status}"
                )
            compatibility_seconds = self._compatibility_seconds(result)
            mapped = self._executor_result(result, completed_variants=1)
            mapped["stageTimings"] = self._search_stage_timings(
                (result,),
                validation_seconds=validation_seconds,
                estimate_seconds=estimate_seconds,
                compatibility_seconds=compatibility_seconds,
                total_seconds=time.perf_counter() - total_started,
            )
            return mapped
        resources = dict(payload["resources"])
        shards = partition_research_shards(
            run_code="RUN-PERF-FIVE",
            run_hash=canonical_sha256(
                {"kind": "ortools_v4_performance", "scenario": scenario.sha256}
            ),
            recipe_hash=canonical_sha256(payload),
            seed_start=int(resources["seedStart"]),
            seed_count=int(resources["seedCount"]),
            seeds_per_shard=int(resources["seedsPerShard"]),
            maximum_attempts=1,
        )
        campaign = executor.execute_campaign(
            scenario,
            proof,
            result_count=5,
            symmetry_policy_id="rules_preserving_v1",
            allow_symmetry_only_variants=False,
            minimum_decision_state_distance=self.minimum_decision_state_distance,
            maximum_proof_nodes=int(self.proof_budget["maximumNodes"]),
            maximum_proof_depth=int(self.proof_budget["maximumDepth"]),
            maximum_hidden_worlds=int(self.proof_budget["maximumHiddenWorlds"]),
            maximum_line_count_limbs=self.maximum_line_count_limbs,
            maximum_candidates=int(resources["seedCount"]),
            maximum_prefix_lift_attempts=int(
                self.search_budget["maximumPrefixLiftAttempts"]
            ),
            maximum_seconds=float(resources["solverTimeoutSeconds"]),
            shards=shards,
            shard_worker_count=min(
                len(shards), int(self.search_budget["maximumShardWorkers"])
            ),
            cancellation_poll_milliseconds=int(
                self.search_budget["cancellationCheckIntervalMilliseconds"]
            ),
        )
        if (
            campaign.status != "completed"
            or len(campaign.accepted_results) != 5
            or not all(
                self._is_qualified_alternate(item)
                for item in campaign.accepted_results
            )
        ):
            raise RuntimeError(
                f"alternate five-variant benchmark ended as {campaign.status}"
            )
        if (
            len({item.semantic_hash for item in campaign.accepted_results}) != 5
            or len(
                {
                    item.candidate.exact_state_id
                    for item in campaign.accepted_results
                    if item.candidate is not None
                }
            )
            != 5
        ):
            raise RuntimeError(
                "alternate five-variant benchmark returned duplicate variants"
            )
        executions = [dict(item.execution_receipt) for item in campaign.executions]
        compatibility_seconds = sum(
            self._compatibility_seconds(item) for item in campaign.accepted_results
        )
        resources = self._aggregate_executor_resources(campaign.executions)
        return {
            "semanticHash": campaign.semantic_hash,
            "contextHash": self.context_hash,
            "stageTimings": self._search_stage_timings(
                campaign.executions,
                validation_seconds=validation_seconds,
                estimate_seconds=estimate_seconds,
                compatibility_seconds=compatibility_seconds,
                total_seconds=time.perf_counter() - total_started,
            ),
            "resources": {
                "completedVariants": len(campaign.accepted_results),
                "completedCandidates": sum(
                    int(item.execution_receipt["completedCandidates"])
                    for item in campaign.executions
                ),
                **resources,
                "campaignReceiptHash": campaign.sha256,
                "campaignReceipt": campaign.to_dict(),
                "executionReceiptHashes": [item["sha256"] for item in executions],
                "executionReceipts": executions,
            },
        }

    def _executor_result(
        self,
        result: Any,
        *,
        completed_variants: int,
    ) -> dict[str, Any]:
        receipt = dict(result.execution_receipt)
        resources = self._aggregate_executor_resources((result,))
        return {
            "semanticHash": result.semantic_hash,
            "contextHash": self.context_hash,
            "resources": {
                "completedVariants": completed_variants,
                "completedCandidates": int(receipt["completedCandidates"]),
                **resources,
                "executorReceiptHash": receipt["sha256"],
                "executorReceipt": receipt,
            },
        }

    @staticmethod
    def _aggregate_executor_resources(executions: Any) -> dict[str, int]:
        conflicts = branches = cache_bytes = 0
        for execution in executions:
            cache_bytes += int(execution.execution_receipt.get("cacheBytes", 0))
            candidate = execution.candidate
            if candidate is None:
                continue
            for receipt in candidate.stage_receipts.values():
                conflicts += int(receipt.conflicts)
                branches += int(receipt.branches)
        return {
            "conflicts": conflicts,
            "branches": branches,
            "cacheBytes": cache_bytes,
        }

    @staticmethod
    def _is_qualified_alternate(result: Any) -> bool:
        proof = result.proof or {}
        return bool(
            result.status == "qualified"
            and result.candidate is not None
            and result.predicate_evaluation is not None
            and result.predicate_evaluation.satisfied
            and proof.get("completionStatus") == "exact_complete"
            and proof.get("claimSatisfied") is True
            and proof.get("winningRootActionCount") == 1
        )

    @staticmethod
    def _search_stage_timings(
        executions: Any,
        *,
        validation_seconds: float,
        estimate_seconds: float,
        compatibility_seconds: float,
        total_seconds: float,
    ) -> dict[str, float]:
        executor_timings = {
            stage: sum(
                float(execution.execution_receipt["stageTimings"].get(stage, 0.0))
                for execution in executions
            )
            for stage in (
                "construction",
                "tailConstruction",
                "proofBoundary",
                "prefixLift",
                "proof",
                "qualification",
            )
        }
        return {
            "validation": validation_seconds,
            "estimate": estimate_seconds,
            **executor_timings,
            "compatibility": compatibility_seconds,
            "total": total_seconds,
        }

    def _compatibility_seconds(self, result: Any) -> float:
        if result.candidate is None or result.research_authority is None:
            raise RuntimeError("qualified flagship result lacks compatibility inputs")
        trace = result.candidate.trace
        exact_state = trace.get("completeExactNormalization") or trace.get(
            "exactNormalization"
        )
        if not isinstance(exact_state, Mapping):
            raise RuntimeError("qualified flagship result lacks exact normalization")
        started = time.perf_counter()
        check = check_runtime_compatibility_v1(
            exact_state=exact_state,
            semantic_result_hash=result.semantic_hash,
            bundle_hash=self.context.bundle.sha256,
            trace_hash=str(result.research_authority["traceHash"]),
            rules=self.context.bundle.rules,
            engine_version=__version__,
            match_score_before=(
                dict(trace["completeExactTracePrefix"]).get("matchScore")
                if isinstance(trace.get("completeExactTracePrefix"), Mapping)
                else None
            ),
        )
        require_runtime_compatibility_passed_v1(check.receipt)
        return time.perf_counter() - started

    def cancellation(self) -> Mapping[str, Any]:
        total_started = time.perf_counter()
        cancellation = threading.Event()
        started = threading.Event()
        cancelled_at: list[float] = []
        shard = partition_research_shards(
            run_code="RUN-PERFCANCEL",
            run_hash=canonical_sha256("performance-cancellation-run"),
            recipe_hash=canonical_sha256("performance-cancellation-recipe"),
            seed_start=1,
            seed_count=1,
            seeds_per_shard=1,
        )

        def solver(request: Any) -> CampaignShardOutcomeV1:
            started.set()
            request.cancellation.wait(1.0)
            return CampaignShardOutcomeV1(())

        def cancel() -> None:
            if not started.wait(1.0):
                return
            cancelled_at.append(time.perf_counter())
            cancellation.set()

        trigger = threading.Thread(target=cancel)
        trigger.start()
        receipt = DeterministicCpSatCampaignCoordinatorV1().run(
            shards=shard,
            solver=solver,
            budget=CampaignBudgetV1(
                1,
                1,
                min(2.0, float(self.search_budget["maximumWallSeconds"])),
                int(self.search_budget["cancellationCheckIntervalMilliseconds"]),
            ),
            cancellation_event=cancellation,
        )
        acknowledged = time.perf_counter()
        trigger.join(timeout=1.0)
        if receipt.status != "cancelled" or not cancelled_at:
            raise RuntimeError("cancellation benchmark did not cancel")
        acknowledgement_ms = (acknowledged - cancelled_at[0]) * 1000.0
        semantic = canonical_sha256(
            {
                "schemaVersion": 1,
                "status": "cancelled",
                "bundleRevisionId": self.context.bundle_revision_id,
                "cancellationPollMilliseconds": int(
                    self.search_budget["cancellationCheckIntervalMilliseconds"]
                ),
            }
        )
        return {
            "semanticHash": semantic,
            "contextHash": self.context_hash,
            "measurementMs": acknowledgement_ms,
            "stageTimings": {
                "cancellationAcknowledgement": acknowledgement_ms / 1000.0,
                "total": time.perf_counter() - total_started,
            },
            "resources": {
                "completedVariants": 0,
                "cancellationStatus": "cancelled",
            },
        }

    def operations(self) -> dict[str, BenchmarkOperationV1]:
        return {
            "validationEstimate": BenchmarkOperationV1(
                lambda: self.validation_estimate
            ),
            "originalFlagshipContradiction": BenchmarkOperationV1(
                lambda: (
                    lambda executor=self._executor(): (
                        self.original_flagship_contradiction(executor)
                    )
                )
            ),
            "exactNormalization": BenchmarkOperationV1(
                lambda: (
                    lambda normalizer=OrToolsExactStateNormalizerV1(
                        self.context.bundle.rules,
                        native_cp_sat=True,
                    ): self.exact_normalization(normalizer)
                )
            ),
            "fixedAlternateProof": BenchmarkOperationV1(
                lambda: (
                    lambda executor=self._executor(): self.fixed_alternate_proof(
                        executor
                    )
                )
            ),
            "alternateFirstQualifiedResult": BenchmarkOperationV1(
                lambda: (
                    lambda executor=self._executor(): self._search(
                        count=1, executor=executor
                    )
                )
            ),
            "alternateFiveVariants": BenchmarkOperationV1(
                lambda: (
                    lambda executor=self._executor(): self._search(
                        count=5, executor=executor
                    )
                )
            ),
            "cancellationAcknowledgement": BenchmarkOperationV1(
                lambda: self.cancellation
            ),
        }


def run_benchmark(
    root: Path = ROOT,
    *,
    operations: Mapping[str, BenchmarkOperationV1] | None = None,
    source_identity: Mapping[str, Any] | None = None,
    profile_path: Path | None = None,
    benchmark_recipe_identities: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    selected_profile_path = profile_path or (
        root / "configs/research/ortools_performance_benchmark_v2.yaml"
    )
    profile = _load_profile(selected_profile_path)
    if operations is None and profile["originalFlagshipRequest"]["expectedStatus"] == "exact_structural_contradiction":
        raise ValueError(
            "historical benchmark profile assumes an unsound V3 flagship "
            "contradiction; no current performance receipt may use it"
        )
    profile_hash = canonical_sha256(profile)
    explicit_bundle = build_bundle(
        root,
        bundle_path=str(profile["bundle"]["source"]),
    )
    original_request: BenchmarkRecipeSourceV1 | None = None
    alternate_recipe: BenchmarkRecipeSourceV1 | None = None
    if benchmark_recipe_identities is None:
        original_request = _load_benchmark_recipe_source(
            root,
            str(profile["originalFlagshipRequest"]["source"]),
            alternate=False,
        )
        alternate_recipe = _load_benchmark_recipe_source(
            root,
            str(profile["alternateVariantRecipe"]["source"]),
            alternate=True,
        )
        recipe_identities = {
            "original": copy.deepcopy(dict(original_request.identity)),
            "alternate": copy.deepcopy(dict(alternate_recipe.identity)),
        }
    elif operations is None:
        raise ValueError("real benchmark cannot inject recipe identities")
    else:
        recipe_identities = _validate_benchmark_recipe_identities(
            benchmark_recipe_identities
        )
    owner: _RealOperationsV1 | None = None
    if operations is None:
        assert original_request is not None and alternate_recipe is not None
        owner = _RealOperationsV1(
            root,
            profile,
            original_request,
            alternate_recipe,
        )
        operations = owner.operations()
    if set(operations) != set(_OPERATIONS):
        raise ValueError("OR-Tools benchmark operation set is incomplete")
    measurement = profile["measurement"]
    resource_limits = _governed_resource_limits(
        explicit_bundle,
        owner.search_profile_id if owner is not None else "late_game_exact_v1",
    )
    results: dict[str, Any] = {}
    try:
        for name in _OPERATIONS:
            results[name] = _measure_operation(
                name,
                operations[name],
                cold_samples=int(measurement["coldSamples"]),
                warm_samples=int(measurement["warmSamples"]),
                warmups=int(measurement["warmupIterations"]),
                budget_ms=float(profile["budgets"][_BUDGET_FIELDS[name]]),
                resource_limits=resource_limits,
            )
        measured_backends = {
            str(sample["resources"]["normalizationBackendId"])
            for phase in ("cold", "warm")
            for sample in results["exactNormalization"][phase]["samples"]
        }
        if measured_backends != {str(profile["bundle"]["backend"])}:
            raise RuntimeError("exact normalization benchmark authority drifted")
    finally:
        if owner is not None:
            owner.close()
    core = {
        "schemaVersion": 1,
        "kind": "ortools_research_v4_performance_evidence",
        "status": (
            "passed" if all(item["passed"] for item in results.values()) else "failed"
        ),
        "source": copy.deepcopy(
            dict(source_identity) if source_identity is not None else git_identity(root)
        ),
        "benchmarkProfile": {
            **copy.deepcopy(profile),
            "sha256": profile_hash,
        },
        "scenarioRequests": recipe_identities,
        "researchAuthority": {
            "backendId": str(profile["bundle"]["backend"]),
            "qualificationAuthority": "cp_sat",
            "bundleId": profile["bundle"]["id"],
            "sourceBundleHash": explicit_bundle.sha256,
            "runtimeBundleHash": (
                owner.context.bundle.sha256 if owner is not None else None
            ),
            "rulesHash": explicit_bundle.rules.sha256,
        },
        "environment": {
            "system": platform.system(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "pythonImplementation": platform.python_implementation(),
            "logicalCpuCount": os.cpu_count(),
            "cpSatWorkersPerSolve": 1,
            "solverVersion": version("ortools"),
        },
        "operations": results,
        "resourceMeasurementBoundary": {
            "conflictsAndBranches": "candidate_stage_receipts_only",
            "tailConstructionCounters": "unavailable",
            "maximumConflicts": "observed_gate_not_executor_enforcement_proof",
            "peakMemory": "process_lifetime_peak_rss",
            "processLifetimeMemoryProfileId": resource_limits["profileId"],
            "cache": "executor_receipt_cache_bytes",
            "cancellation": "cooperative_acknowledgement",
        },
        "proofBoundary": (
            "The literal original request is measured only as the named exact "
            "structural contradiction FLAGSHIP-TWO-ACES-TRICK7-LEAD-V1. "
            "First-result, five-variant, and fixed-proof timings belong only to the "
            "explicit alternate where the target team wins trick six; they are not "
            "evidence that the original request has a solution. "
            "Cold means a fresh operation instance in the same process; warm means "
            "a reused instance after the governed warmup. P50/P95 values are "
            "host-bound execution evidence. Fixed-proof latency is graph compilation "
            "plus CP-SAT proof; its end-to-end wall time, including separately measured "
            "exact normalization and qualification, remains resource evidence. "
            "Semantic and context hashes must remain identical across every sample; "
            "timings and machine resources are excluded from semantic identity."
        ),
    }
    return {**core, "evidenceSha256": canonical_sha256(core)}


def profile_quality_fixture(root: Path, *, complete_round: bool = False) -> dict[str, Any]:
    """Diagnostic stage timings, not a release latency or strength benchmark."""
    from collections import defaultdict
    from contextlib import ExitStack
    from functools import wraps
    from unittest.mock import patch

    import pytest
    from ortools.sat.python import cp_model
    from baloot_platform.research import ortools_research_executor as executor
    from baloot_platform.research.ortools_play_tail import OrToolsPlayTailStageV1

    rows = defaultdict(lambda: {"calls": 0, "inclusiveSeconds": 0.0, "exclusiveSeconds": 0.0})
    solve_sites = defaultdict(lambda: {"calls": 0, "seconds": 0.0})
    normalization_sites = defaultdict(lambda: {"calls": 0, "seconds": 0.0})
    normalization_inputs = defaultdict(int)
    normalization_instance_inputs = defaultdict(int)
    normalization_labels = {}
    normalization_sources = defaultdict(set)
    local, lock, threads = threading.local(), threading.Lock(), set()

    def timed(name, operation):
        @wraps(operation)
        def invoke(*args, **kwargs):
            site = None
            normalization_key = None
            normalization_site = None
            if name == "normalization" and len(args) > 1:
                prefix = args[1]
                normalization_key = canonical_sha256({
                    "prefix": getattr(prefix, "sha256", None),
                    "decision": kwargs.get("decision_point") or {"kind": "bid"},
                    "opportunities": kwargs.get("recorded_escalation_opportunities"),
                    "order": kwargs.get("opportunity_order"),
                    "template": (kwargs.get("_play_template") or (None,))[0],
                })
                frame = sys._getframe(1)
                path = Path(frame.f_code.co_filename)
                label = path.relative_to(root) if path.is_relative_to(root) else path.name
                normalization_site = f"{label}:{frame.f_code.co_firstlineno}:{frame.f_code.co_name}"
                del frame
            if name == "cpSatSolve":
                frame = sys._getframe(1)
                callers = []
                for _ in range(6):
                    if frame is None:
                        break
                    path = Path(frame.f_code.co_filename)
                    label = path.relative_to(root) if path.is_relative_to(root) else path.name
                    callers.append(f"{label}:{frame.f_code.co_firstlineno}:{frame.f_code.co_name}")
                    frame = frame.f_back
                site = " <- ".join(callers)
                del frame
            stack = getattr(local, "stack", None)
            if stack is None:
                local.stack = stack = []
            children = [0.0]
            stack.append(children)
            started = time.perf_counter()
            try:
                return operation(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - started
                stack.pop()
                if stack:
                    stack[-1][0] += elapsed
                with lock:
                    threads.add(threading.get_ident())
                    row = rows[name]
                    row["calls"] += 1
                    row["inclusiveSeconds"] += elapsed
                    row["exclusiveSeconds"] += elapsed - children[0]
                    if normalization_key is not None:
                        normalization_inputs[normalization_key] += 1
                        normalization_instance_inputs[(id(args[0]), normalization_key)] += 1
                        normalization_labels[normalization_key] = (
                            len(prefix.play_actions), len(prefix.project_actions),
                            str((kwargs.get("decision_point") or {}).get("kind", "bid")),
                        )
                        normalization_sources[normalization_key].add(normalization_site)
                        normalization_sites[normalization_site]["calls"] += 1
                        normalization_sites[normalization_site]["seconds"] += elapsed
                    if site is not None:
                        solve_sites[site]["calls"] += 1
                        solve_sites[site]["seconds"] += elapsed
        return invoke

    targets = (
        (OrToolsExactStateNormalizerV1, "normalize", "normalization"),
        (OrToolsPlayTailStageV1, "build_model", "playModelConstruction"),
        (executor, "compile_observed_decision_v1", "observedGraphCompilation"),
        (executor, "prove_actor_information_set_v1", "informationSetProof"),
        (cp_model.CpSolver, "solve", "cpSatSolve"),
    )
    arguments = [str(root / "tests/test_research_ortools_unified_executor.py"),
                 "-q", "-k", "test_native_quality_generates_all_first_trick_decisions and "
                 + ("True" if complete_round else "False"), "--tb=short"]
    started = time.perf_counter()
    with ExitStack() as patches:
        for owner, attribute, name in targets:
            patches.enter_context(patch.object(owner, attribute, timed(name, getattr(owner, attribute))))
        exit_code = int(pytest.main(arguments))
    elapsed = time.perf_counter() - started
    valid = all(0 <= row["exclusiveSeconds"] <= row["inclusiveSeconds"] for row in rows.values())
    repeated_by_position = defaultdict(lambda: {"calls": 0, "distinctInputs": 0})
    repeated_by_source = defaultdict(lambda: {"calls": 0, "distinctInputs": 0})
    for key, count in normalization_inputs.items():
        row = repeated_by_position[str(normalization_labels[key])]
        row["calls"] += count
        row["distinctInputs"] += 1
        if count > 1:
            source_row = repeated_by_source[str(sorted(normalization_sources[key]))]
            source_row["calls"] += count
            source_row["distinctInputs"] += 1
    core = {"schemaVersion": 1, "kind": "native_quality_stage_timing_diagnostic_v1",
        "status": "passed" if exit_code == 0 and rows and valid else "failed",
        "testExitCode": exit_code, "elapsedSeconds": elapsed, "stages": dict(rows),
        "normalizationCallSites": dict(normalization_sites),
        "normalizationInputCounts": {
            "calls": sum(normalization_inputs.values()),
            "distinctPrefixDecisions": len(normalization_inputs),
            "repeatedCalls": sum(count - 1 for count in normalization_inputs.values()),
            "sameInstanceRepeatedCalls": sum(count - 1 for count in normalization_instance_inputs.values()),
        },
        "normalizationByPosition": dict(repeated_by_position),
        "normalizationRepeatedBySource": dict(repeated_by_source),
        "solverCallSites": dict(solve_sites),
        "timingAccountingValid": valid, "observedThreadCount": len(threads),
        "source": git_identity(root), "python": platform.python_version(), "solverVersion": version("ortools"),
        "boundary": "One instrumented " + ("complete-round" if complete_round else "first-trick")
        + " fixture, not P95 or strength evidence. Inclusive times overlap; exclusive times subtract timed children on the same thread only."}
    return {**core, "evidenceSha256": canonical_sha256(core)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark the published OR-Tools-only Research V4 path."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quality-diagnostic", action="store_true")
    parser.add_argument("--quality-diagnostic-full-round", action="store_true")
    args = parser.parse_args()
    if args.quality_diagnostic_full_round and not args.quality_diagnostic:
        parser.error("--quality-diagnostic-full-round requires --quality-diagnostic")
    if args.quality_diagnostic and args.output is None:
        parser.error("--quality-diagnostic requires an explicit --output outside release evidence")
    output = args.output or OUTPUT
    if args.quality_diagnostic and output.resolve().is_relative_to((ROOT / "release").resolve()):
        parser.error("quality diagnostics must not overwrite release evidence")
    if args.quality_diagnostic and output.exists():
        parser.error("quality diagnostics require an unused output path to preserve prior evidence")
    evidence = profile_quality_fixture(ROOT, complete_round=args.quality_diagnostic_full_round) if args.quality_diagnostic else run_benchmark(ROOT)
    atomic_write_text(
        output,
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(output.resolve())
    return 0 if evidence["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
