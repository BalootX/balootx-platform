"""Matched qualification pilot for OR-Tools guided deal construction.

This module deliberately treats CP-SAT objectives as construction guidance.
Only the normal Research replay, actor-safe analyzer, tactical verifier, score
profile, and qualification gates may produce a qualified finalist.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from typing import Any, Mapping

from .contracts import canonical_sha256


@dataclass(frozen=True, slots=True)
class GuidedConstructionPilotProfileV1:
    profile_id: str = "guided_structural_tactical_pilot"
    revision: int = 2
    attempts_per_lane_profile: int = 20
    seed_start: int = 910_000
    seeds_per_shard: int = 5
    candidates_per_seed: int = 1
    solver_timeout_seconds: float = 2.0
    maximum_conflicts: int = 50_000
    minimum_yield_multiplier: float = 2.0
    minimum_cost_reduction: float = 0.40
    maximum_wait_seconds: float = 900.0
    poll_seconds: float = 0.20
    construction_pre_screen_enabled: bool = True
    construction_pre_screen_maximum_play_decisions: int = 4
    construction_pre_screen_maximum_bid_regret: float = 5.0
    construction_pre_screen_maximum_play_regret: float = 8.0
    construction_pre_screen_maximum_probability_drop: float = 0.25
    construction_pre_screen_minimum_confidence: float = 0.20

    def __post_init__(self) -> None:
        if not self.profile_id or self.revision < 1:
            raise ValueError("guided pilot identity is invalid")
        if not 1 <= self.attempts_per_lane_profile <= 100:
            raise ValueError("guided pilot attempts must be in [1, 100]")
        if self.seed_start < 0:
            raise ValueError("guided pilot seed start must be non-negative")
        if not 1 <= self.seeds_per_shard <= self.attempts_per_lane_profile:
            raise ValueError("guided pilot shard size is invalid")
        if not 1 <= self.candidates_per_seed <= 10:
            raise ValueError("guided pilot candidates per seed is invalid")
        if not 0.1 <= self.solver_timeout_seconds <= 60:
            raise ValueError("guided pilot solver timeout is invalid")
        if not 1 <= self.maximum_conflicts <= 10_000_000:
            raise ValueError("guided pilot conflict budget is invalid")
        if self.minimum_yield_multiplier <= 0:
            raise ValueError("guided pilot yield threshold is invalid")
        if not 0 <= self.minimum_cost_reduction < 1:
            raise ValueError("guided pilot cost threshold is invalid")
        if self.maximum_wait_seconds <= 0 or not 0.01 <= self.poll_seconds <= 5:
            raise ValueError("guided pilot wait budget is invalid")
        if not 1 <= self.construction_pre_screen_maximum_play_decisions <= 16:
            raise ValueError("guided pilot pre-screen decision bound is invalid")
        if min(
            self.construction_pre_screen_maximum_bid_regret,
            self.construction_pre_screen_maximum_play_regret,
        ) < 0:
            raise ValueError("guided pilot pre-screen regret bound is invalid")
        if not 0 <= self.construction_pre_screen_maximum_probability_drop <= 1:
            raise ValueError("guided pilot pre-screen probability bound is invalid")
        if not 0 <= self.construction_pre_screen_minimum_confidence <= 1:
            raise ValueError("guided pilot pre-screen confidence is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "profileId": self.profile_id,
            "revision": self.revision,
            "attemptsPerLaneProfile": self.attempts_per_lane_profile,
            "seedStart": self.seed_start,
            "seedsPerShard": self.seeds_per_shard,
            "candidatesPerSeed": self.candidates_per_seed,
            "solverTimeoutSeconds": self.solver_timeout_seconds,
            "maximumConflicts": self.maximum_conflicts,
            "minimumYieldMultiplier": self.minimum_yield_multiplier,
            "minimumCostReduction": self.minimum_cost_reduction,
            "maximumWaitSeconds": self.maximum_wait_seconds,
            "pollSeconds": self.poll_seconds,
            "constructionPreScreen": {
                "enabled": self.construction_pre_screen_enabled,
                "maximumPlayDecisions": (
                    self.construction_pre_screen_maximum_play_decisions
                ),
                "maximumBidRegret": (
                    self.construction_pre_screen_maximum_bid_regret
                ),
                "maximumPlayRegret": (
                    self.construction_pre_screen_maximum_play_regret
                ),
                "maximumProbabilityDrop": (
                    self.construction_pre_screen_maximum_probability_drop
                ),
                "minimumConfidence": (
                    self.construction_pre_screen_minimum_confidence
                ),
            },
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def _profile_definitions() -> tuple[dict[str, Any], ...]:
    return (
        {
            "id": "project_400",
            "titleAr": "مشروع 400",
            "seedOffset": 0,
            "situation": {
                "dealer": "east",
                "buyer": "south",
                "purchaseRound": None,
                "trumpSuit": None,
                "contracts": ["sun"],
                "matchScore": [0, 0],
                "requiredCards": {},
                "forbiddenCards": {},
                "projects": [
                    {
                        "seat": "south",
                        "contract": "sun",
                        "kind": "four_hundred",
                        "value": 400,
                        "highRank": "ace",
                        "mode": "hard",
                    }
                ],
                "escalations": [],
            },
            "tactics": [],
            "targetTactic": None,
            "deepVerificationPolicy": "off",
            "constructionIntent": "hard_domain_project_pattern",
        },
        {
            "id": "kabout_opportunity",
            "titleAr": "فرصة كبوت",
            "seedOffset": 100_000,
            "situation": {
                "dealer": "east",
                "buyer": "south",
                "purchaseRound": None,
                "trumpSuit": None,
                "contracts": ["sun"],
                "matchScore": [0, 0],
                "requiredCards": {},
                "forbiddenCards": {},
                "projects": [],
                "escalations": [],
            },
            "tactics": [
                {
                    "kind": "kabout_creation",
                    "mode": "qualification_gate",
                    "minimumScore": 1,
                    "weight": 1,
                    "perspectives": ["south"],
                }
            ],
            "targetTactic": "kabout_creation",
            "deepVerificationPolicy": "required_gates",
            "constructionIntent": "domain_tactical_preference_plus_causal_gate",
        },
        {
            "id": "small_card_delayed_gain_precursor",
            "titleAr": "ورقة صغيرة تفتح مكسب",
            "seedOffset": 200_000,
            "situation": {
                "dealer": "east",
                "buyer": "south",
                "purchaseRound": None,
                "trumpSuit": "hearts",
                "contracts": ["hokum"],
                "matchScore": [0, 0],
                "requiredCards": {},
                "forbiddenCards": {},
                "projects": [],
                "escalations": [],
            },
            "tactics": [
                {
                    "kind": "small_card_big_gain",
                    "mode": "qualification_gate",
                    "minimumScore": 1,
                    "weight": 1,
                    "perspectives": ["south"],
                }
            ],
            "targetTactic": "small_card_big_gain",
            "deepVerificationPolicy": "required_gates",
            "constructionIntent": "domain_tactical_preference_plus_causal_gate",
        },
    )


def build_guided_pilot_recipe_v1(
    research: Any,
    definition: Mapping[str, Any],
    profile: GuidedConstructionPilotProfileV1,
) -> dict[str, Any]:
    capabilities = research.capabilities()
    active = dict(capabilities["research"])
    seed_start = profile.seed_start + int(definition["seedOffset"])
    return {
        "schemaVersion": 1,
        "titleAr": str(definition["titleAr"]),
        "output": {
            "kind": "laffa",
            "resultCount": profile.attempts_per_lane_profile,
        },
        "profiles": {
            "domainHash": capabilities["sha256"],
            "bundleHash": capabilities["bundleHash"],
            "rulesHash": capabilities["rulesHash"],
            "conventionHash": capabilities["conventionHash"],
            "agentRelease": active["defaultAgentRelease"],
            "analysisProfileHash": active["analysisProfiles"]["screening"],
            "scoreProfileHash": active["defaultScoreProfileHash"],
        },
        "situation": dict(definition["situation"]),
        "tactics": [dict(item) for item in definition["tactics"]],
        "qualification": {
            "minimumTruth": "provisional",
            "minimumConfidence": 0,
            "minimumSurroundingQuality": 70,
            "requireObserverSafety": True,
            "requireAllDecisionAudit": True,
            "maximumRegret": 0,
        },
        "resources": {
            "generationLane": "ortools",
            "seedStart": seed_start,
            "seedCount": profile.attempts_per_lane_profile,
            "seedsPerShard": profile.seeds_per_shard,
            "candidatesPerSeed": profile.candidates_per_seed,
            "solverTimeoutSeconds": profile.solver_timeout_seconds,
            "maximumConflicts": profile.maximum_conflicts,
            "analysisMode": "screening",
            "playConstructionMode": "actor_safe_agent_verified",
            "deepVerificationPolicy": definition["deepVerificationPolicy"],
            "constructionPreScreen": {
                "schemaVersion": 1,
                "id": "guided_construction_shallow_pre_screen",
                "revision": "1",
                "enabled": profile.construction_pre_screen_enabled,
                "maximumPlayDecisions": (
                    profile.construction_pre_screen_maximum_play_decisions
                ),
                "maximumBidRegret": (
                    profile.construction_pre_screen_maximum_bid_regret
                ),
                "maximumPlayRegret": (
                    profile.construction_pre_screen_maximum_play_regret
                ),
                "maximumProbabilityDrop": (
                    profile.construction_pre_screen_maximum_probability_drop
                ),
                "minimumConfidence": (
                    profile.construction_pre_screen_minimum_confidence
                ),
                "rejectCatastrophicCrossing": True,
            },
        },
        "status": "published",
    }


def _page_items(research: Any, run_code: str) -> tuple[dict[str, Any], ...]:
    page = research.list_results(run_code=run_code, sort="result_code_asc", limit=500)
    return tuple(dict(item) for item in page.get("items") or ())


def _lane_metrics(
    research: Any,
    run: Mapping[str, Any],
    *,
    target_tactic: str | None,
) -> dict[str, Any]:
    items = _page_items(research, str(run["runCode"]))
    qualified = [item for item in items if item.get("disposition") == "qualified"]
    result_fingerprints = [
        str(item.get("gameHash") or item["resultHash"]) for item in items
    ]
    distinct_results = len(set(result_fingerprints))
    fingerprints = [str(item.get("gameHash") or item["resultHash"]) for item in qualified]
    distinct = len(set(fingerprints))
    duplicate_count = len(fingerprints) - distinct
    cpu_seconds = run.get("cpuSeconds")
    qualified_per_cpu_hour = (
        distinct * 3600.0 / float(cpu_seconds)
        if cpu_seconds is not None and float(cpu_seconds) > 0
        else None
    )
    cpu_cost = (
        float(cpu_seconds) / distinct
        if cpu_seconds is not None and distinct > 0
        else None
    )
    tactic_proven = sum(
        target_tactic is not None
        and target_tactic in set(item.get("verifiedTacticalTags") or ())
        for item in items
    )
    surrounding = [
        float(dict(item.get("qualification") or {}).get("minimumSurroundingQuality", 0))
        for item in items
    ]
    return {
        "runCode": str(run["runCode"]),
        "status": str(run["status"]),
        "attemptedSeeds": int(run.get("attemptedSeeds", 0)),
        "generatedCandidates": int(run.get("generatedCandidates", 0)),
        "resultCount": len(items),
        "distinctResultCount": distinct_results,
        "duplicateResultCount": len(result_fingerprints) - distinct_results,
        "duplicateResultRate": (
            (len(result_fingerprints) - distinct_results) / len(result_fingerprints)
            if result_fingerprints
            else 0.0
        ),
        "qualifiedCount": len(qualified),
        "distinctQualifiedFinalists": distinct,
        "duplicateQualifiedCount": duplicate_count,
        "duplicateQualifiedRate": duplicate_count / len(qualified) if qualified else 0.0,
        "targetTacticalProofCount": tactic_proven,
        "minimumSurroundingQuality": min(surrounding) if surrounding else None,
        "meanSurroundingQuality": sum(surrounding) / len(surrounding) if surrounding else None,
        "semanticDefectCount": int(run.get("semanticDefectCount", 0)),
        "quarantineCount": int(run.get("quarantineCount", 0)),
        "constructionPreScreenCandidates": int(
            run.get("constructionPreScreenCandidates", 0)
        ),
        "constructionPreScreenRejected": int(
            run.get("constructionPreScreenRejected", 0)
        ),
        "constructionPreScreenSeconds": float(
            run.get("constructionPreScreenSeconds", 0.0)
        ),
        "analyzedCandidateCount": int(run.get("analyzedCandidateCount", 0)),
        "elapsedWallSeconds": float(run.get("elapsedWallSeconds", 0)),
        "cpuSeconds": cpu_seconds,
        "distinctQualifiedPerCpuHour": qualified_per_cpu_hour,
        "cpuSecondsPerDistinctQualified": cpu_cost,
        "stageSeconds": dict(run.get("stageSeconds") or {}),
    }


def _comparison_decision(
    ortools: Mapping[str, Any],
    organic: Mapping[str, Any],
    profile: GuidedConstructionPilotProfileV1,
) -> dict[str, Any]:
    reasons: list[str] = []
    if ortools["status"] != "completed" or organic["status"] != "completed":
        reasons.append("lane_incomplete")
    if int(ortools["semanticDefectCount"]) or int(organic["semanticDefectCount"]):
        reasons.append("semantic_defect")
    if int(ortools["distinctQualifiedFinalists"]) < 1:
        reasons.append("no_distinct_ortools_finalist")
    ort_yield = ortools.get("distinctQualifiedPerCpuHour")
    org_yield = organic.get("distinctQualifiedPerCpuHour")
    yield_multiplier = (
        float(ort_yield) / float(org_yield)
        if ort_yield is not None and org_yield is not None and float(org_yield) > 0
        else None
    )
    ort_cost = ortools.get("cpuSecondsPerDistinctQualified")
    org_cost = organic.get("cpuSecondsPerDistinctQualified")
    cost_reduction = (
        1.0 - float(ort_cost) / float(org_cost)
        if ort_cost is not None and org_cost is not None and float(org_cost) > 0
        else None
    )
    yield_passed = (
        yield_multiplier is not None
        and yield_multiplier >= profile.minimum_yield_multiplier
    ) or (
        int(organic["distinctQualifiedFinalists"]) == 0
        and int(ortools["distinctQualifiedFinalists"]) > 0
    )
    cost_passed = (
        cost_reduction is not None
        and cost_reduction >= profile.minimum_cost_reduction
    )
    if not (yield_passed or cost_passed):
        reasons.append("end_to_end_threshold_not_met")
    retain = not reasons
    return {
        "retainGuidedObjectiveOrConstraint": retain,
        "recommendation": "retain" if retain else "do_not_promote",
        "yieldMultiplier": yield_multiplier,
        "costReduction": cost_reduction,
        "yieldGatePassed": yield_passed,
        "costGatePassed": cost_passed,
        "reasons": reasons,
    }


def run_guided_construction_pilot_v1(
    research: Any,
    *,
    profile: GuidedConstructionPilotProfileV1 | None = None,
) -> dict[str, Any]:
    cfg = profile or GuidedConstructionPilotProfileV1()
    capabilities = research.capabilities()
    compiler = dict(
        dict(capabilities.get("research") or {}).get("compilerQualification") or {}
    )
    if not compiler.get("executableBindingComplete"):
        raise ValueError("guided pilot requires complete executable Domain IR binding")
    campaigns: list[tuple[dict[str, Any], str]] = []
    for definition in _profile_definitions():
        recipe = research.create_recipe(
            build_guided_pilot_recipe_v1(research, definition, cfg)
        )
        campaign = research.create_comparison_campaign(
            recipe["code"],
            idempotency_key=(
                f"guided-pilot:{cfg.sha256}:{definition['id']}"
            ),
        )
        campaigns.append((definition, str(campaign["campaignCode"])))

    deadline = time.monotonic() + cfg.maximum_wait_seconds
    completed: dict[str, dict[str, Any]] = {}
    while len(completed) < len(campaigns):
        for definition, code in campaigns:
            if str(definition["id"]) in completed:
                continue
            campaign = research.get_comparison_campaign(code)
            if campaign is None:
                raise RuntimeError(f"comparison campaign disappeared: {code}")
            if campaign["status"] in {"completed", "failed", "cancelled"}:
                completed[str(definition["id"])] = campaign
        if len(completed) == len(campaigns):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("guided construction pilot exceeded its wait budget")
        time.sleep(cfg.poll_seconds)

    profiles: list[dict[str, Any]] = []
    for definition, code in campaigns:
        campaign = completed[str(definition["id"])]
        lanes = dict(campaign["runs"])
        ortools = _lane_metrics(
            research,
            lanes["ortools"],
            target_tactic=definition["targetTactic"],
        )
        organic = _lane_metrics(
            research,
            lanes["organic_runtime"],
            target_tactic=definition["targetTactic"],
        )
        decision = _comparison_decision(ortools, organic, cfg)
        profiles.append(
            {
                "profileId": definition["id"],
                "campaignCode": code,
                "constructionIntent": definition["constructionIntent"],
                "targetTactic": definition["targetTactic"],
                "constructionSolverTruth": "structural_objective_only",
                "qualificationAuthority": "actor_safe_research_analyzer",
                "lanes": {"ortools": ortools, "organic_runtime": organic},
                "decision": decision,
            }
        )
    stable_core = {
        "schemaVersion": 1,
        "pilotProfileHash": cfg.sha256,
        "domainHash": capabilities["sha256"],
        "bundleHash": capabilities["bundleHash"],
        "profiles": profiles,
    }
    return {
        **stable_core,
        "pilot": "guided_structural_tactical_construction",
        "pilotProfile": cfg.to_dict(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "retainedProfiles": [
            item["profileId"]
            for item in profiles
            if item["decision"]["retainGuidedObjectiveOrConstraint"]
        ],
        "evidenceHash": canonical_sha256(stable_core),
    }
