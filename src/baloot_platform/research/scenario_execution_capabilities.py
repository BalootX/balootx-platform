"""Executable Foundry capability projection for the OR-Tools Research backend."""

from __future__ import annotations

import copy
from typing import Any, Mapping

from baloot_platform.adjudication.research import AdjudicationResearchAuthorityV1

from .scenario_backend import (
    ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3,
    PHASE4_FULL_SERVICE_PROVEN_COMBINATIONS_V1,
    PHASE4_REQUIRED_COMBINATIONS_V1,
    OrToolsScenarioResearchBackendV2,
    materialize_default_policy_profiles_v1,
)
from .scenario_contracts import (
    LINE_UNIVERSES,
    PROOF_CLAIMS,
    PROOF_BOUNDARY_EVALUATOR_DEFINITIONS,
    PROOF_MODES,
    PROOF_HORIZON_UNITS,
    PLAYING_QUALITY_HORIZON_UNITS_BY_DECISION_V1,
    PLAYING_QUALITY_MODES,
    PROOF_VISIBILITIES,
    SCENARIO_DECISION_KINDS,
    ScenarioPlayingQualityPolicyV1,
    canonical_sha256,
)
from .scenario_predicate_manifest_v2 import (
    shared_scenario_predicate_catalog_lowering_manifest_v2,
)
from .scenario_predicate_capabilities_v3 import (
    NATIVE_V3_TACTIC_IDS,
    SCENARIO_PREDICATE_CAPABILITY_BACKEND_V3,
)
from .recipe_v4 import research_recipe_v4_authoring_catalog


_ORIGINS = ("search", "exact_trace_prefix")
_DECISION_KINDS = ("bid", "escalation", "project", "play")
_SEATS = ("north", "west", "south", "east")


def _phase4_acceptance_gaps(
    backend: OrToolsScenarioResearchBackendV2,
) -> list[dict[str, Any]]:
    if backend.authority.backend != ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3:
        return []
    return [
        {
            "acceptanceId": "phase4_strategic_proof_matrix_v1",
            "origin": origin,
            "decisionKind": decision_kind,
            "mode": mode,
            "visibility": visibility,
            "reason": "non_vacuous_full_service_fixture_missing",
            "requiredEvidence": [
                "full_service_execution",
                "non_vacuous_outcome_threshold",
                "differentiating_root_outcomes",
                "persisted_proof_receipt",
            ],
        }
        for origin, decision_kind, mode, visibility in sorted(
            PHASE4_REQUIRED_COMBINATIONS_V1
            - PHASE4_FULL_SERVICE_PROVEN_COMBINATIONS_V1
        )
    ]


def _policy_profiles(
    backend: OrToolsScenarioResearchBackendV2,
) -> dict[str, dict[str, Any]]:
    return materialize_default_policy_profiles_v1(backend.authority.policy_catalog)


def _proof_budget_profile(
    backend: OrToolsScenarioResearchBackendV2,
    *,
    mode: str,
    visibility: str,
) -> str:
    proof_profiles = dict(backend.authority.proof_budgets.get("profiles", {}))
    search_profiles = dict(backend.authority.search_budgets.get("profiles", {}))
    available = set(proof_profiles) & set(search_profiles)
    preferred = (
        "actor_information_set_exact_v1"
        if visibility == "actor_information_set"
        else "bounded_v1"
        if mode == "bounded"
        else "late_game_exact_v1"
    )
    if preferred in available:
        return preferred
    if not available:
        raise ValueError("scenario Research has no shared proof/search budget profile")
    return sorted(available)[0]


def _sample_recipe(
    backend: OrToolsScenarioResearchBackendV2,
    *,
    origin: str,
    decision_kind: str,
    mode: str,
    visibility: str,
    claim: str,
    line_universe: str | None,
) -> dict[str, Any]:
    profile_id = _proof_budget_profile(
        backend,
        mode=mode,
        visibility=visibility,
    )
    selector: dict[str, Any] = {}
    if decision_kind == "play":
        selector = {
            "trickIndex": 0,
            "position": 0,
            "remainingCards": 8,
            "actingSeat": "south",
        }
    elif decision_kind == "bid":
        selector = {"actionIndex": 0}
    elif decision_kind == "escalation":
        selector = {"actionIndex": 0, "actingSeat": "south"}
    elif decision_kind == "project":
        selector = {"owner": "south", "projectIndex": 0}
    proof: dict[str, Any] = {
        "mode": mode,
        "visibility": visibility,
        "claim": claim,
        "targetTeam": "north_south",
        "utility": "round_win",
        "successThreshold": 1,
        "budgetProfile": profile_id,
    }
    if visibility == "actor_information_set":
        proof["strategyClass"] = "pure_deterministic"
    if mode == "bounded":
        proof.update(
            {
                "horizon": {
                    "unit": "tricks" if decision_kind == "play" else "plies",
                    "value": 1,
                },
                "boundaryEvaluator": "current_raw_point_margin_v1",
            }
        )
    elif mode == "policy_conditioned":
        proof.update(
            {
                "strategyClass": "frozen_policy",
                "policies": _policy_profiles(backend),
            }
        )
    if line_universe is not None:
        proof["lineUniverse"] = line_universe
    if claim == "selected_action_is_unique_winner":
        proof["selectedAction"] = "D7"
    return {
        "scenario": {
            "origin": {"kind": origin},
            "decisionPoint": {"kind": decision_kind, "selector": selector},
        },
        "proof": proof,
        "resources": {"proofBudgetProfile": profile_id},
    }


def _profile_predicate_capability_manifest_v3(
    backend: OrToolsScenarioResearchBackendV2,
    combinations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Project possible V3 contexts through the selected bundle execution gate."""

    allowed = {
        (
            str(item["origin"]),
            str(item["decisionKind"]),
            str(item["mode"]),
            str(item["visibility"]),
        ): frozenset(str(value) for value in item["lineUniverses"])
        for item in combinations
    }
    manifest = backend.lowering_manifest.capability_manifest.catalog()
    capabilities: list[dict[str, Any]] = []
    for source in manifest["capabilities"]:
        contexts = []
        for raw_context in source["executionContexts"]:
            context = copy.deepcopy(dict(raw_context))
            key = (
                str(context["origin"]),
                str(context["decisionKind"]),
                str(context["proofMode"]),
                str(context["visibility"]),
            )
            supported_lines = allowed.get(key)
            requested_lines = frozenset(
                str(value) for value in context["lineUniverses"]
            )
            if supported_lines is not None and (
                not requested_lines or requested_lines <= supported_lines
            ):
                contexts.append(context)
        if not contexts:
            continue
        retained_decisions = {str(item["decisionKind"]) for item in contexts}
        argument_contexts = []
        for raw_argument_context in source.get("argumentContexts", ()):
            argument_context = copy.deepcopy(dict(raw_argument_context))
            decisions = sorted(
                retained_decisions
                & {str(value) for value in argument_context["decisionKinds"]}
            )
            if decisions:
                argument_contexts.append(
                    {**argument_context, "decisionKinds": decisions}
                )
        row = {
            key: copy.deepcopy(value)
            for key, value in source.items()
            if key
            not in {
                "sha256",
                "origins",
                "decisionKinds",
                "proofModes",
                "visibilityClasses",
                "executionContexts",
                "argumentContexts",
            }
        }
        row.update(
            {
                "origins": sorted({item["origin"] for item in contexts}),
                "decisionKinds": sorted(
                    {item["decisionKind"] for item in contexts}
                ),
                "proofModes": sorted({item["proofMode"] for item in contexts}),
                "visibilityClasses": sorted(
                    {item["visibility"] for item in contexts}
                ),
                "executionContexts": contexts,
            }
        )
        if argument_contexts:
            row["argumentContexts"] = argument_contexts
        capabilities.append({**row, "sha256": canonical_sha256(row)})
    core = {
        key: copy.deepcopy(value)
        for key, value in manifest.items()
        if key not in {"sha256", "capabilities"}
    }
    core["capabilities"] = capabilities
    return {**core, "sha256": canonical_sha256(core)}


def build_scenario_execution_capabilities_v1(
    backend: OrToolsScenarioResearchBackendV2,
    *,
    adjudication_authority: AdjudicationResearchAuthorityV1 | None = None,
    expose_shard_worker_cap: bool = True,
) -> dict[str, Any]:
    """Build controls only from selections accepted by ``execution_support``."""

    combinations: list[dict[str, Any]] = []
    for origin in _ORIGINS:
        for decision_kind in _DECISION_KINDS:
            for mode in PROOF_MODES:
                for visibility in PROOF_VISIBILITIES:
                    claims: list[str] = []
                    line_universes: list[str] = []
                    selection_constraints: dict[str, Any] | None = None
                    for claim in PROOF_CLAIMS:
                        universes = (
                            LINE_UNIVERSES
                            if claim == "winning_continuation_line_count"
                            else (None,)
                        )
                        supported_universes: list[str] = []
                        for line_universe in universes:
                            recipe = _sample_recipe(
                                backend,
                                origin=origin,
                                decision_kind=decision_kind,
                                mode=mode,
                                visibility=visibility,
                                claim=claim,
                                line_universe=line_universe,
                            )
                            support = backend.execution_support(recipe)
                            if support["executionReady"]:
                                support_constraints = dict(
                                    support.get("selectionConstraints", {})
                                )
                                if selection_constraints is None:
                                    selection_constraints = support_constraints
                                elif selection_constraints != support_constraints:
                                    raise ValueError(
                                        "scenario execution constraints differ inside "
                                        "one executable combination"
                                    )
                                if line_universe is not None:
                                    supported_universes.append(line_universe)
                                elif claim not in claims:
                                    claims.append(claim)
                        if supported_universes:
                            claims.append(claim)
                            line_universes.extend(supported_universes)
                    if not claims:
                        continue
                    combinations.append(
                        {
                            "origin": origin,
                            "decisionKind": decision_kind,
                            "mode": mode,
                            "visibility": visibility,
                            "claims": claims,
                            "lineUniverses": list(dict.fromkeys(line_universes)),
                            "strategyClasses": [
                                "frozen_policy"
                                if mode == "policy_conditioned"
                                else "pure_deterministic"
                            ],
                            "contractValues": (
                                list(selection_constraints["contractValues"])
                                if selection_constraints
                                and "contractValues" in selection_constraints
                                else None
                            ),
                        }
                    )
    unavailable_reasons: dict[str, str] = {}
    for decision_kind in _DECISION_KINDS:
        if any(item["decisionKind"] == decision_kind for item in combinations):
            continue
        sample = _sample_recipe(
            backend,
            origin="exact_trace_prefix",
            decision_kind=decision_kind,
            mode="exact",
            visibility="complete_state",
            claim="outcome_reachable",
            line_universe=None,
        )
        reasons = tuple(backend.execution_support(sample).get("reasons", ()))
        unavailable_reasons[decision_kind] = next(
            (
                str(reason)
                for reason in reasons
                if str(reason).startswith(f"{decision_kind}_")
                or reason == "phase4_substantive_full_service_evidence_missing"
            ),
            "decision_point_proof_unavailable",
        )
    supported_decisions = {item["decisionKind"] for item in combinations}
    selector_schemas = {
        "bid": {
            "type": "object",
            "additionalProperties": False,
            "required": ["actionIndex"],
            "properties": {
                "actionIndex": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 10_000,
                    "default": 0,
                }
            },
        },
        "escalation": {
            "type": "object",
            "additionalProperties": False,
            "required": ["actionIndex", "actingSeat"],
            "properties": {
                "actionIndex": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 10_000,
                    "default": 0,
                },
                "actingSeat": {
                    "type": "string",
                    "enum": list(_SEATS),
                    "default": "south",
                },
            },
        },
        "project": {
            "type": "object",
            "additionalProperties": False,
            "required": ["owner", "projectIndex"],
            "properties": {
                "owner": {"type": "string", "enum": list(_SEATS), "default": "south"},
                "projectIndex": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 0,
                    "default": 0,
                },
            },
        },
        "play": {
            "type": "object",
            "additionalProperties": False,
            "required": ["trickIndex", "remainingCards"],
            "properties": {
                "trickIndex": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 7,
                    "default": 0,
                },
                "position": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 3,
                },
                "remainingCards": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 8,
                    "default": 8,
                },
                "actingSeat": {
                    "type": "string",
                    "enum": list(_SEATS),
                    "default": "south",
                },
            },
        },
    }
    manifest = shared_scenario_predicate_catalog_lowering_manifest_v2()
    authoring_catalog = research_recipe_v4_authoring_catalog(
        execution_capabilities={
            "schemaVersion": 1,
            "combinations": combinations,
            "decisionUnavailableReasons": unavailable_reasons,
            "decisionSelectorSchemas": {
                kind: copy.deepcopy(selector_schemas[kind])
                for kind in _DECISION_KINDS
                if kind in supported_decisions
            },
        },
        policy_catalog=backend.authority.policy_catalog,
        default_policy_profiles=_policy_profiles(backend),
        proof_budgets=backend.authority.proof_budgets,
        search_budgets=backend.authority.search_budgets,
        expose_shard_worker_cap=expose_shard_worker_cap,
        authoring_defaults=backend.authority.authoring_defaults,
        predicate_kinds=tuple(
            definition.kind for definition in manifest.exposed_definitions
        ),
        adjudication_catalog=(
            adjudication_authority.authoring_catalog()
            if adjudication_authority is not None
            else None
        ),
    )
    if backend.authority.backend == SCENARIO_PREDICATE_CAPABILITY_BACKEND_V3:
        authoring_core = {
            key: copy.deepcopy(value)
            for key, value in authoring_catalog.items()
            if key != "sha256"
        }
        authoring_core["predicateCapabilityManifest"] = (
            _profile_predicate_capability_manifest_v3(backend, combinations)
        )
        authoring_core["tacticKinds"] = sorted(NATIVE_V3_TACTIC_IDS)
        execution = backend.authority.quality_execution
        from .family_catalog import project_inference_generation_supported_v1
        project_inference_supported = project_inference_generation_supported_v1(backend._rules, execution)
        if execution is not None and execution.family_target is not None:
            from .family_catalog import family_target_authoring_v1, validate_family_target_generation_v1
            try:
                # This is an available authoring combination, not the operator's chosen policy.
                exact_quality = (ScenarioPlayingQualityPolicyV1.build({
                    'schemaVersion': 2, 'mode': 'exact', 'visibility': 'actor_information_set',
                    'decisions': {kind: {'utility': 'awarded_point_margin',
                        'budgetProfile': _proof_budget_profile(backend, mode='exact', visibility='actor_information_set')}
                        for kind in SCENARIO_DECISION_KINDS}}) if project_inference_supported else None)
                validate_family_target_generation_v1(execution.family_target, rules=backend._rules,
                    execution=execution, playing_quality=exact_quality)
            except ValueError:
                pass  # Historical targets remain editable, not executable controls.
            else:
                authoring_core["familyGeneration"] = family_target_authoring_v1(execution.family_target)
        if (execution is not None and execution.family_catalog is not None
                and backend._rules is not None and backend._rules.project_domain.declaration_mode == "mandatory"):
            from .family_catalog import (ResearchFamilyCatalogV1, family_search_request_schema_v1,
                                         family_generation_techniques_v1)
            catalog = ResearchFamilyCatalogV1.build(execution.family_catalog)
            live_techniques = {family.family_id: family_generation_techniques_v1(family) for family in catalog.families}
            request_schema = family_search_request_schema_v1(maximum_auction_seeds=execution.proposal_seed_count,
                project_inference_supported=project_inference_supported and any(
                    family.enabled and 'sun' in family.contracts and 'project-inference' in live_techniques[family.family_id]
                    for family in catalog.families))
            authoring_core["familySearch"] = {"schemaVersion": 1,
                "catalog": copy.deepcopy(dict(execution.family_catalog)),
                "playIndices": list(execution.proposal_play_indices
                                    if execution.proposal_play_indices is not None else range(32)),
                "liveTechniquesByFamily": live_techniques,
                "requestSchema": request_schema,
                "supportedAxes": list(request_schema['properties']['tagQuery']['properties'])}
        if execution is not None and execution.escalation_opportunity_order is not None:
            quality_defaults = dict(
                backend.authority.authoring_defaults.get("playingQuality") or {}
            )
            quality_budget_profile = str(
                quality_defaults.get("budgetProfile")
                or authoring_core["fields"]["resources.proofBudgetProfile"]["default"]
            )
            if quality_budget_profile not in authoring_core["proof"]["budgetProfiles"]:
                raise ValueError("playingQuality authoring budget profile is unpublished")
            quality_boundary_defaults = dict(
                quality_defaults.get("boundaryEvaluatorByDecision") or {}
            )
            quality_exact_profiles = dict(quality_defaults.get("exactBudgetProfileByVisibility") or {
                visibility: _proof_budget_profile(backend, mode="exact", visibility=visibility)
                for visibility in PROOF_VISIBILITIES
            })
            if (set(quality_exact_profiles) != set(PROOF_VISIBILITIES)
                    or any(value not in authoring_core["proof"]["budgetProfiles"]
                           for value in quality_exact_profiles.values())):
                raise ValueError("playingQuality exact budget defaults are unpublished")
            if set(quality_boundary_defaults) != set(SCENARIO_DECISION_KINDS):
                raise ValueError("playingQuality boundary defaults must cover every decision kind")
            for kind, evaluator in quality_boundary_defaults.items():
                definition = PROOF_BOUNDARY_EVALUATOR_DEFINITIONS.get(evaluator)
                if definition is None or kind not in definition["decisionKinds"]:
                    raise ValueError("playingQuality boundary default is unsupported")
            authoring_core["playingQuality"] = {
                "schemaVersion": 1,
                "policySchemaVersion": 2,
                "visibilities": list(PROOF_VISIBILITIES),
                "legacyVisibility": "actor_information_set",
                "modes": list(PLAYING_QUALITY_MODES),
                "modesByOrigin": {"search": list(PLAYING_QUALITY_MODES),
                                  "exact_trace_prefix": list(PLAYING_QUALITY_MODES) if execution.allow_exact_input_quality else ["budgeted"]},
                "decisionKinds": list(SCENARIO_DECISION_KINDS),
                "defaultBudgetProfile": quality_budget_profile,
                "defaultBudgetProfileByMode": {
                    "budgeted": quality_budget_profile,
                    "exact": quality_exact_profiles["actor_information_set"],
                },
                "defaultBoundaryEvaluatorByDecision": quality_boundary_defaults,
                "defaultBudgetProfileByModeAndVisibility": {
                    mode: {visibility: quality_budget_profile if mode == "budgeted"
                           else quality_exact_profiles[visibility]
                           for visibility in PROOF_VISIBILITIES}
                    for mode in PLAYING_QUALITY_MODES
                },
                "requiredDecisionFields": ["utility", "budgetProfile", "horizon", "boundaryEvaluator"],
                "fieldSources": {
                    "visibility": "proof.visibilities",
                    "utility": "proof.utilities", "budgetProfile": "proof.budgetProfiles",
                    "horizon.unit": "proof.horizonUnits", "boundaryEvaluator": "playingQuality.boundaryEvaluators",
                },
                "execution": execution.to_dict(),
                "boundaryEvaluators": [name for name, definition in PROOF_BOUNDARY_EVALUATOR_DEFINITIONS.items()
                                       if definition["qualityOperatorVisible"]],
                "boundaryEvaluatorDefinitions": {
                    name: {key: list(value) if isinstance(value, tuple) else value
                           for key, value in definition.items()}
                    for name, definition in PROOF_BOUNDARY_EVALUATOR_DEFINITIONS.items()
                    if definition["qualityOperatorVisible"]
                },
                "horizonBounds": {
                    name: {unit: {"minimum": 1, "step": 1,
                        **({"maximum": min(profile["maximumDepth"], profile["maximumHorizonActions"])
                            // (len(_SEATS) if unit == "tricks" else 1)} if unit != "round_end" else {})}
                        for unit in PROOF_HORIZON_UNITS}
                    for name, profile in authoring_core["proof"]["budgetProfiles"].items()
                },
                "horizonUnitsByDecision": {
                    kind: list(units)
                    for kind, units in PLAYING_QUALITY_HORIZON_UNITS_BY_DECISION_V1.items()
                },
                "recordedEscalationOpportunities": {
                    "type": "array", "maxItems": execution.maximum_escalation_decisions,
                    "items": {"type": "object", "additionalProperties": False,
                        "required": ["actor", "actionCode"], "properties": {
                            "actor": {"type": "string", "enum": list(_SEATS)},
                            "actionCode": {"type": "string", "minLength": 1},
                        }},
                    "actionSource": "exact_trace_legal_action_preview",
                },
            }
        authoring_catalog = {
            **authoring_core,
            "sha256": canonical_sha256(authoring_core),
        }
    core = {
        "schemaVersion": 1,
        "combinations": combinations,
        "phase4AcceptanceGaps": _phase4_acceptance_gaps(backend),
        "decisionUnavailableReasons": unavailable_reasons,
        "decisionSelectorSchemas": {
            kind: copy.deepcopy(selector_schemas[kind])
            for kind in _DECISION_KINDS
            if kind in supported_decisions
        },
    }
    core["authoringCatalog"] = authoring_catalog
    return {**core, "sha256": canonical_sha256(core)}


def project_scenario_configuration_execution_v1(
    configuration: Mapping[str, Any],
    execution_capabilities: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the shared scenario controls to executable combinations only."""

    value = copy.deepcopy(dict(configuration))
    combinations = list(execution_capabilities["combinations"])
    unavailable_reasons = dict(
        execution_capabilities.get("decisionUnavailableReasons", {})
    )
    origins = list(dict.fromkeys(item["origin"] for item in combinations))
    modes = list(dict.fromkeys(item["mode"] for item in combinations))
    visibilities = list(dict.fromkeys(item["visibility"] for item in combinations))
    claims = list(
        dict.fromkeys(claim for item in combinations for claim in item["claims"])
    )
    line_universes = list(
        dict.fromkeys(
            universe for item in combinations for universe in item["lineUniverses"]
        )
    )
    strategy_classes = list(
        dict.fromkeys(
            strategy for item in combinations for strategy in item["strategyClasses"]
        )
    )
    decision_capabilities = []
    for kind in _DECISION_KINDS:
        selected = [item for item in combinations if item["decisionKind"] == kind]
        decision_capabilities.append(
            {
                "kind": kind,
                "authoringSupported": bool(selected),
                "exactTraceNormalizationSupported": any(
                    item["origin"] == "exact_trace_prefix" for item in selected
                ),
                "searchGenerationSupported": any(
                    item["origin"] == "search" for item in selected
                ),
                "proofModes": list(dict.fromkeys(item["mode"] for item in selected)),
                "renderPreparationSupported": bool(selected),
                "unavailableReason": (
                    None
                    if selected
                    else unavailable_reasons.get(
                        kind, "decision_point_proof_unavailable"
                    )
                ),
            }
        )
    value.update(
        {
            "origins": origins,
            "decisionCapabilities": decision_capabilities,
            "proofModes": modes,
            "proofStrategyClasses": strategy_classes,
            "proofVisibilities": visibilities,
            "proofClaims": claims,
            "proofLineUniverses": line_universes,
        }
    )
    value.pop("sha256", None)
    return {**value, "sha256": canonical_sha256(value)}


__all__ = [
    "build_scenario_execution_capabilities_v1",
    "project_scenario_configuration_execution_v1",
]
