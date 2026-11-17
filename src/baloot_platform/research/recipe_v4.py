"""Canonical Research Recipe V4 contract."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from baloot_platform.challenge.quality_diversity import QualityDiversityProfileV1
from baloot_platform.core.models import LegalPlayActionV1
from baloot_platform.tactics import TacticalKind, TACTICAL_LOCALIZATION_KEYS

from .contracts import ResearchConfigurationAuthorityV1
from .scenario_contracts import (
    LINE_UNIVERSES,
    OPERATOR_PROOF_BOUNDARY_EVALUATORS,
    validate_proof_boundary_scope_v1,
    PROOF_CLAIMS,
    PROOF_HORIZON_UNITS,
    PROOF_MODES,
    PROOF_STRATEGY_CLASSES,
    PROOF_UTILITIES,
    PROOF_UTILITY_ENCODINGS,
    PROOF_VISIBILITIES,
    ResearchScenarioV1,
    ScenarioPlayingQualityPolicyV1,
    SCENARIO_EXPRESSION_MAXIMUM_CHILDREN,
    SCENARIO_EXPRESSION_MAXIMUM_COUNT_RANGE,
    SCENARIO_EXPRESSION_MAXIMUM_DEPTH,
    SCENARIO_EXPRESSION_MAXIMUM_NODES,
    SCENARIO_EXPRESSION_MAXIMUM_SERIALIZED_BYTES,
    SCENARIO_OPERATORS,
    SCENARIO_SECTIONS,
    ScenarioProofRequestV1,
    canonical_sha256,
)
from .scenario_registry import ScenarioCapabilityPlanV1, ScenarioCompilerV1
from .family_catalog import build_family_generation_request_v1, lesson_tag_catalog_v1


_CODE = re.compile(r"^RCP-[A-Z0-9]{6,20}$")

_DEFAULT_QUALIFICATION = {
    "minimumTruth": "robust_best_known",
    "minimumConfidence": 0.75,
    "minimumSurroundingQuality": 70.0,
    "requireObserverSafety": True,
    "requireAllDecisionAudit": True,
    "maximumRegret": 0.0,
}
_DEFAULT_RESOURCES = {
    "generationLane": "ortools",
    "generationPolicy": "ortools_cp_sat_v3",
    "seedStart": 1,
    "seedCount": 4,
    "seedsPerShard": 4,
    "solverTimeoutSeconds": 15.0,
    "proofBudgetProfile": "late_game_exact_v1",
}
_LEGACY_RESOURCE_DEFAULTS = {
    **_DEFAULT_RESOURCES,
    "generationPolicy": "natural_policy",
    "candidatesPerSeed": 4,
    "maximumConflicts": 250000,
    "analysisMode": "cascaded",
}
_COMPATIBILITY_ONLY_RESOURCE_FIELDS = frozenset(
    {"candidatesPerSeed", "maximumConflicts", "analysisMode"}
)
_DEFAULT_DIVERSITY = {
    "schemaVersion": 1,
    "cellDimensions": ["contract", "project", "decision_kind", "proof_mode"],
    "topKPerCell": 3,
    "rejectedAuditModulus": 100,
}
_DEFAULT_OUTPUT = {"kind": "laffa", "resultCount": 1}
_DEFAULT_RENDER_POLICY = {"mode": "on_demand", "videoCount": 1}
_BID_ACTION = re.compile(
    r"[a-z][a-z0-9_]*(?::(?:diamonds|clubs|hearts|spades))?"
)
_ESCALATION_ACTION = re.compile(r"[a-z][a-z0-9_]*(?::(?:open|locked))?")
_PROJECT_ACTION = re.compile(
    r"(?:pass|declare:[a-z][a-z0-9_]*:[0-9]+:[0-9]+|"
    r"declare:PROJECT-[0-9A-F]{16})"
)


_V4_FIELD_MESSAGE_KEYS = {
    "titleAr": "research_scenario_recipe_title",
    "qualification.minimumTruth": "research_label_truth",
    "qualification.minimumConfidence": "research_label_confidence",
    "qualification.minimumSurroundingQuality": "research_minimum_surrounding_quality",
    "qualification.maximumRegret": "research_maximum_regret",
    "qualification.requireObserverSafety": "research_observer_safety_required",
    "qualification.requireAllDecisionAudit": "research_all_decisions_required",
    "resources.seedStart": "research_seed_start",
    "resources.seedCount": "research_label_seeds",
    "resources.seedsPerShard": "research_seeds_per_shard",
    "resources.shardWorkerCount": "research_shard_worker_count",
    "resources.solverTimeoutSeconds": "research_estimate_solver",
    "resources.proofBudgetProfile": "research_scenario_budget",
    "diversity.cellDimensions": "research_diversity_dimensions",
    "diversity.topKPerCell": "research_diversity_top_k",
    "diversity.rejectedAuditModulus": "research_diversity_audit",
    "output.kind": "research_label_output",
    "output.resultCount": "research_label_result_count",
    "renderPolicy.mode": "research_scenario_render_policy",
    "renderPolicy.videoCount": "research_scenario_desired_videos",
    "status": "research_scenario_lifecycle_status",
}

_V4_FIELD_HELP_KEYS = {
    "titleAr": "research_scenario_help_title",
    "qualification.minimumTruth": "research_scenario_help_minimum_truth",
    "qualification.minimumConfidence": "research_scenario_help_minimum_confidence",
    "qualification.minimumSurroundingQuality": "research_scenario_help_minimum_surrounding_quality",
    "qualification.maximumRegret": "research_scenario_help_maximum_regret",
    "qualification.requireObserverSafety": "research_scenario_help_require_observer_safety",
    "qualification.requireAllDecisionAudit": "research_scenario_help_require_all_decision_audit",
    "resources.seedStart": "research_scenario_help_seed_start",
    "resources.seedCount": "research_scenario_help_seed_count",
    "resources.seedsPerShard": "research_scenario_help_seeds_per_shard",
    "resources.shardWorkerCount": "research_scenario_help_shard_worker_count",
    "resources.solverTimeoutSeconds": "research_scenario_help_solver_timeout_seconds",
    "resources.proofBudgetProfile": "research_scenario_help_proof_budget_profile",
    "diversity.cellDimensions": "research_scenario_help_diversity_dimensions",
    "diversity.topKPerCell": "research_scenario_help_diversity_top_k",
    "diversity.rejectedAuditModulus": "research_scenario_help_rejected_audit_modulus",
    "output.kind": "research_scenario_help_output_kind",
    "output.resultCount": "research_scenario_help_result_count",
    "renderPolicy.mode": "research_scenario_help_render_mode",
    "renderPolicy.videoCount": "research_scenario_video_count_help",
    "status": "research_scenario_help_recipe_status",
}

# Catalog-driven Scenario Studio copy that is not attached to one recipe field.
_V4_INTERFACE_COPY_KEYS = (
    "research_scenario_argument_card_selector_help",
    "research_scenario_authority_help",
    "research_scenario_authority_title",
    "research_scenario_backend",
    "research_scenario_backend_help",
    "research_scenario_backend_ortools_cp_sat_v2",
    "research_scenario_backend_ortools_cp_sat_v3",
    "research_scenario_deterministic_execution",
    "research_scenario_deterministic_execution_help",
    "research_scenario_exact_state_available",
    "research_scenario_exact_state_unavailable",
    "research_scenario_planner",
    "research_scenario_planner_help",
    "research_scenario_planner_tail_first_v1",
    "research_scenario_proof_bounded_help",
    "research_scenario_proof_budgets",
    "research_scenario_proof_budgets_help",
    "research_scenario_proof_exact_help",
    "research_scenario_proof_policy_conditioned_help",
    "research_scenario_runtime_compatibility",
    "research_scenario_runtime_compatibility_failed",
    "research_scenario_runtime_compatibility_help",
    "research_scenario_runtime_compatibility_not_checked",
    "research_scenario_runtime_compatibility_passed",
    "research_scenario_search_budgets",
    "research_scenario_search_budgets_help",
    "research_scenario_timing_compatibility",
    "research_scenario_timing_construction",
    "research_scenario_timing_first_result",
    "research_scenario_timing_help",
    "research_scenario_timing_playing_quality",
    "research_scenario_timing_quality_auction",
    "research_scenario_timing_quality_project_play",
    "research_scenario_timing_prefix_lift",
    "research_scenario_timing_proof",
    "research_scenario_timing_qualification",
    "research_scenario_timing_tail_construction",
    "research_scenario_timing_title",
    "research_scenario_timing_total",
    "research_scenario_timing_validation",
    "research_scenario_timing_variant",
)

_V4_PREDICATE_HELP_KEYS = {
    "play.action_matches": "research_scenario_predicate_play_action_matches_help",
    "play.trick_winner_matches": "research_scenario_predicate_play_trick_winner_matches_help",
    "play.trick_card_count": "research_scenario_predicate_play_trick_card_count_help",
    "proof.winning_root_action_matches": "research_scenario_predicate_proof_winning_root_action_matches_help",
}

_V4_VALUE_MESSAGE_KEYS = {
    "all": "research_scenario_operator_all",
    "any": "research_scenario_operator_any",
    "not": "research_scenario_operator_not",
    "count": "research_scenario_operator_count",
    "hard": "research_scenario_hardness_hard",
    "prefer": "research_scenario_hardness_prefer",
    "report": "research_scenario_hardness_report",
    "search": "research_scenario_origin_search",
    "exact_trace_prefix": "research_scenario_origin_exact_trace_prefix",
    "bid": "research_scenario_decision_bid",
    "escalation": "research_scenario_decision_escalation",
    "project": "research_scenario_decision_project",
    "play": "research_scenario_decision_play",
    "exact": "research_scenario_proof_exact",
    "bounded": "research_scenario_proof_bounded",
    "budgeted": "research_scenario_proof_policy_conditioned",
    "policy_conditioned": "research_scenario_proof_policy_conditioned",
    "complete_state": "research_scenario_visibility_complete_state",
    "actor_information_set": "research_scenario_visibility_actor_information_set",
    "outcome_reachable": "research_scenario_claim_outcome_reachable",
    "outcome_forced": "research_scenario_claim_outcome_forced",
    "winning_root_action_count": "research_scenario_claim_winning_root_action_count",
    "winning_continuation_line_count": "research_scenario_claim_winning_continuation_line_count",
    "selected_action_is_unique_winner": "research_scenario_claim_selected_action_is_unique_winner",
    "target_team_has_no_salvage": "research_scenario_claim_target_team_has_no_salvage",
    "pure_deterministic": "research_scenario_strategy_pure_deterministic",
    "frozen_policy": "research_scenario_strategy_frozen_policy",
    "laffa": "research_output_laffa",
    "laffa_collection": "research_output_collection",
    "on_demand": "research_render_on_demand",
    "disabled": "research_render_disabled",
    "draft": "research_scenario_status_draft",
    "published": "research_scenario_status_published",
    "retired": "research_scenario_status_retired",
    "complete": "research_status_completed",
    "incomplete": "research_scenario_support_incomplete",
    "derived": "research_scenario_support_derived",
    "bounds": "research_scenario_support_bounds",
    "unsupported": "research_scenario_capability_unsupported",
    "construction": "research_scenario_support_construction",
    "derived_state": "research_scenario_support_derived_state",
    "proof": "research_scenario_stage_proof",
    "report_only": "research_scenario_support_report_only",
    "public": "research_scenario_support_public",
    "observation": "research_scenario_support_observation",
    "reference_world_only": "research_scenario_support_reference_world_only",
    "nonanticipative": "research_scenario_support_nonanticipative",
    "constant": "research_scenario_cost_constant",
    "linear": "research_scenario_cost_linear",
    "solver": "research_scenario_cost_solver",
    "tree_search": "research_scenario_cost_tree_search",
    "fixed_action_map_v1": "research_scenario_policy_fixed_action_map_v1",
    "seeded_legal_action_order_v1": "research_scenario_policy_seeded_legal_action_order_v1",
    "created": "research_scenario_option_phase_created",
    "shuffled": "research_scenario_option_phase_shuffled",
    "initial_dealt": "research_scenario_option_phase_initial_dealt",
    "bidding": "research_scenario_option_phase_bidding",
    "final_dealt": "research_scenario_option_phase_final_dealt",
    "projects": "research_scenario_option_phase_projects",
    "playing": "research_scenario_option_phase_playing",
    "scored": "research_scenario_option_phase_scored",
    "initial_five": "research_scenario_option_initial_five",
    "post_purchase": "research_scenario_option_post_purchase",
    "seven": "research_scenario_option_rank_seven",
    "eight": "research_scenario_option_rank_eight",
    "nine": "research_scenario_option_rank_nine",
    "ten": "research_scenario_option_rank_ten",
    "jack": "research_scenario_option_rank_jack",
    "queen": "research_scenario_option_rank_queen",
    "king": "research_scenario_option_rank_king",
    "ace": "research_scenario_option_rank_ace",
    "siri": "research_scenario_option_project_siri",
    "fifty": "research_scenario_option_project_fifty",
    "hundred_sequence": "research_scenario_option_project_hundred_sequence",
    "hundred_four_kind": "research_scenario_option_project_hundred_four_kind",
    "four_hundred": "research_scenario_option_project_four_hundred",
    "4": "research_scenario_option_project_value_4",
    "10": "research_scenario_option_project_value_10",
    "20": "research_scenario_option_project_value_20",
    "40": "research_scenario_option_project_value_40",
    "eligible": "research_scenario_option_project_eligible",
    "declared": "research_scenario_option_project_declared",
    "comparison_winner": "research_scenario_option_project_comparison_winner",
    "awarded": "research_scenario_option_project_awarded",
    "public_fact": "research_scenario_option_information_public_fact",
    "legal_deduction": "research_scenario_option_information_legal_deduction",
    "probabilistic_inference": "research_scenario_option_information_probabilistic_inference",
    "policy_assumption": "research_scenario_option_information_policy_assumption",
    "won": "research_scenario_option_outcome_won",
    "lost": "research_scenario_option_outcome_lost",
    "tied": "research_scenario_option_outcome_tied",
    "all_legal": "research_scenario_option_lines_all_legal",
    "optimal_strategy": "research_scenario_option_lines_optimal_strategy",
    "awarded_point_margin": "research_scenario_option_utility_awarded_point_margin",
    "contract_success": "research_scenario_option_utility_contract_success",
    "round_win": "research_scenario_option_utility_round_win",
    "match_win": "research_scenario_option_utility_match_win",
    "kabout": "research_scenario_option_utility_kabout",
    "current_raw_point_margin_v1": "research_scenario_option_boundary_current_raw_point_margin_v1",
    "auction_hand_margin_v1": "research_scenario_option_boundary_auction_hand_margin_v1",
    "plies": "research_scenario_option_horizon_plies",
    "decisions": "research_scenario_option_horizon_decisions",
    "tricks": "research_scenario_option_horizon_tricks",
    "round_end": "research_scenario_option_horizon_round_end",
    "interactive_validation_v1": "research_scenario_option_budget_interactive_validation_v1",
    "bounded_v1": "research_scenario_option_budget_bounded_v1",
    "late_game_exact_v1": "research_scenario_option_budget_late_game_exact_v1",
    "actor_information_set_exact_v1": "research_scenario_option_budget_actor_information_set_exact_v1",
    "offline_heavyweight_v1": "research_scenario_option_budget_offline_heavyweight_v1",
    "open": "research_scenario_option_mode_open",
    "locked": "research_scenario_option_mode_locked",
    "north": "research_scenario_option_seat_north",
    "west": "research_scenario_option_seat_west",
    "south": "research_scenario_option_seat_south",
    "east": "research_scenario_option_seat_east",
    "north_south": "research_scenario_option_team_north_south",
    "west_east": "research_scenario_option_team_west_east",
    "sun": "research_scenario_option_contract_sun",
    "hokum": "research_scenario_option_contract_hokum",
    "ashkal": "research_scenario_option_contract_ashkal",
    "spades": "research_scenario_option_suit_spades",
    "hearts": "research_scenario_option_suit_hearts",
    "diamonds": "research_scenario_option_suit_diamonds",
    "clubs": "research_scenario_option_suit_clubs",
    "pass": "research_scenario_option_action_pass",
    "declare": "research_project_declared",
    "decline": "research_scenario_option_action_decline",
    "double": "research_escalation_double",
    "triple": "research_escalation_triple",
    "four": "research_escalation_four",
    "gahwa": "research_escalation_gahwa",
    **TACTICAL_LOCALIZATION_KEYS,
    "exact_optimal": "research_truth_exact",
    "robust_best_known": "research_truth_robust",
    "debatable": "research_truth_debatable",
    "provisional": "research_truth_provisional",
    "contract": "research_label_contract",
    "decision_kind": "research_scenario_decision",
    "proof_mode": "research_scenario_proof_mode",
    "deal": "research_scenario_stage_deal",
    "auction": "research_scenario_stage_auction",
    "playPrefix": "research_scenario_stage_playPrefix",
    "state": "research_scenario_stage_state",
    "outcome": "research_scenario_stage_outcome",
    "tactics": "research_scenario_stage_tactics",
    "none": "research_scenario_estimate_unavailable",
}


def research_recipe_v4_authoring_catalog(
    *,
    execution_capabilities: Mapping[str, Any],
    policy_catalog: Mapping[str, Any],
    default_policy_profiles: Mapping[str, Any],
    proof_budgets: Mapping[str, Any],
    authoring_defaults: Mapping[str, Any],
    predicate_kinds: tuple[str, ...],
    search_budgets: Mapping[str, Any] | None = None,
    expose_shard_worker_cap: bool = True,
    adjudication_catalog: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the closed authoring authority for new V4 recipes.

    Legacy V1-V3 catalog rows are intentionally not reused here.  This view is
    generated from the V4 contract constants and the selected bundle's frozen
    policy and budget catalogs.
    """

    combinations = copy.deepcopy(list(execution_capabilities["combinations"]))
    selectors = copy.deepcopy(dict(execution_capabilities["decisionSelectorSchemas"]))
    budget_profiles = copy.deepcopy(dict(proof_budgets.get("profiles", {})))
    shard_worker_limits = {
        name: int(profile["maximumShardWorkers"])
        for name, profile in dict((search_budgets or {}).get("profiles", {})).items()
    }
    default_budget = (
        "late_game_exact_v1"
        if "late_game_exact_v1" in budget_profiles
        else sorted(budget_profiles)[0]
    )
    resource_defaults = dict(authoring_defaults.get("resources") or {})
    seeds_per_shard_default = int(
        resource_defaults.get("seedsPerShard", _DEFAULT_RESOURCES["seedsPerShard"])
    )
    fields: dict[str, dict[str, Any]] = {
        "titleAr": {
            "control": "text",
            "minimumLength": 3,
            "maximumLength": 160,
            "required": True,
            "defaultKey": "research_scenario_default_title",
        },
        "qualification.minimumTruth": {
            "control": "enum", "values": ["exact_optimal", "robust_best_known", "debatable", "provisional"], "default": _DEFAULT_QUALIFICATION["minimumTruth"],
        },
        "qualification.minimumConfidence": {"control": "number", "minimum": 0, "maximum": 1, "default": _DEFAULT_QUALIFICATION["minimumConfidence"]},
        "qualification.minimumSurroundingQuality": {"control": "number", "minimum": 0, "maximum": 100, "default": _DEFAULT_QUALIFICATION["minimumSurroundingQuality"]},
        "qualification.maximumRegret": {"control": "number", "minimum": 0, "default": _DEFAULT_QUALIFICATION["maximumRegret"]},
        "qualification.requireObserverSafety": {"control": "boolean", "value": True, "default": True, "mode": "invariant"},
        "qualification.requireAllDecisionAudit": {"control": "boolean", "value": True, "default": True, "mode": "invariant"},
        "resources.seedStart": {"control": "integer", "minimum": 0, "default": _DEFAULT_RESOURCES["seedStart"]},
        "resources.seedCount": {"control": "integer", "minimum": 1, "maximum": 1_000_000_000, "default": _DEFAULT_RESOURCES["seedCount"]},
        "resources.seedsPerShard": {"control": "integer", "minimum": 1, "maximum": 1_000_000, "default": seeds_per_shard_default},
        "resources.solverTimeoutSeconds": {"control": "number", "minimum": 0.001, "maximum": 900, "default": _DEFAULT_RESOURCES["solverTimeoutSeconds"]},
        "resources.proofBudgetProfile": {"control": "enum", "values": sorted(budget_profiles), "default": default_budget},
        "diversity.cellDimensions": {"control": "multi_enum", "values": ["contract", "project", "decision_kind", "proof_mode"], "minimumItems": 1, "uniqueItems": True, "default": copy.deepcopy(_DEFAULT_DIVERSITY["cellDimensions"])},
        "diversity.topKPerCell": {"control": "integer", "minimum": 1, "maximum": 1_000, "default": _DEFAULT_DIVERSITY["topKPerCell"]},
        "diversity.rejectedAuditModulus": {"control": "integer", "minimum": 1, "maximum": 1_000_000, "default": _DEFAULT_DIVERSITY["rejectedAuditModulus"]},
        "output.kind": {"control": "enum", "values": ["laffa", "laffa_collection"], "default": _DEFAULT_OUTPUT["kind"]},
        "output.resultCount": {"control": "integer", "minimum": 1, "maximum": 1_000, "default": _DEFAULT_OUTPUT["resultCount"]},
        "renderPolicy.mode": {"control": "enum", "values": ["on_demand", "disabled"], "default": _DEFAULT_RENDER_POLICY["mode"]},
        "renderPolicy.videoCount": {"control": "integer", "minimum": 0, "maximum": 100, "default": _DEFAULT_RENDER_POLICY["videoCount"]},
        "status": {"control": "enum", "values": ["draft", "published", "retired"], "default": "published"},
    }
    if expose_shard_worker_cap:
        fields["resources.shardWorkerCount"] = {"control": "integer", "minimum": 1, "maximum": max(shard_worker_limits.values(), default=1), "maximumByBudgetProfile": shard_worker_limits, "default": None, "required": False}
    for path, key in _V4_FIELD_MESSAGE_KEYS.items():
        if path not in fields:
            continue
        fields[path]["labelKey"] = key
        fields[path]["helpKey"] = _V4_FIELD_HELP_KEYS[path]
    exact_trace = {
        "schemaVersion": 1,
        "roundNumber": {"type": "integer", "minimum": 1, "maximum": 10_000, "default": 1},
        "matchScore": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0, "maximum": 10_000}, "default": [0, 0]},
        "projectActions": {
            "type": "array",
            "maxItems": 4,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["actor", "action"],
                "properties": {
                    "actor": {
                        "type": "string",
                        "enum": ["north", "west", "south", "east"],
                    },
                    "action": {
                        "type": "string",
                        "enum": ["pass", "declare"],
                    },
                    "projectId": {
                        "type": "string",
                        "pattern": r"^PROJECT-[0-9A-F]{16}$",
                    },
                },
                "allOf": [
                    {
                        "if": {
                            "properties": {"action": {"const": "declare"}},
                            "required": ["action"],
                        },
                        "then": {"required": ["projectId"]},
                    },
                    {
                        "if": {
                            "properties": {"action": {"const": "pass"}},
                            "required": ["action"],
                        },
                        "then": {"not": {"required": ["projectId"]}},
                    },
                ],
            },
        },
        "stateAssertions": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "phase": {
                    "type": "string",
                    "enum": ["created", "shuffled", "initial_dealt", "bidding", "final_dealt", "projects", "playing", "scored", "complete"],
                    "x-messageArKey": "research_scenario_exact_state_phase",
                },
                "buyer": {
                    "type": "string",
                    "enum": ["north", "west", "south", "east"],
                    "x-messageArKey": "research_label_buyer",
                },
                "contract": {
                    "type": "string",
                    "enum": ["sun", "hokum", "ashkal"],
                    "x-messageArKey": "research_label_contract",
                },
                "trumpSuit": {
                    "type": "string",
                    "enum": ["spades", "hearts", "diamonds", "clubs"],
                    "x-messageArKey": "research_scenario_argument_suit",
                },
                "currentPlayer": {
                    "type": "string",
                    "enum": ["north", "west", "south", "east"],
                    "x-messageArKey": "research_scenario_next_actor",
                },
                "projectWinningTeam": {
                    "type": "string",
                    "enum": ["north_south", "west_east"],
                    "x-messageArKey": "research_scenario_target_team",
                },
            },
        },
    }
    if adjudication_catalog is not None:
        adjudication = copy.deepcopy(dict(adjudication_catalog))
        required_adjudication = {
            "schemaVersion",
            "authorityProfileId",
            "authoritySha256",
            "fieldPath",
            "qualificationSource",
            "exactInputItemSchema",
            "events",
            "sha256",
        }
        if set(adjudication) != required_adjudication:
            raise ValueError("adjudication authoring catalog shape is invalid")
        if (
            adjudication["schemaVersion"] != 1
            or adjudication["fieldPath"]
            != "scenario.origin.exactTracePrefix.adjudicationEvents"
            or adjudication["qualificationSource"] != "report_only"
        ):
            raise ValueError("adjudication authoring catalog identity is invalid")
        exact_trace["adjudicationEvents"] = {
            "type": "array",
            "default": [],
            "uniqueItems": True,
            "items": adjudication["exactInputItemSchema"],
            "x-messageArKey": "research_adjudication_events",
            "x-messageEnKey": "research_adjudication_events_en",
            "x-authorityProfileId": adjudication["authorityProfileId"],
            "x-authoritySha256": adjudication["authoritySha256"],
            "x-catalogSha256": adjudication["sha256"],
            "x-capabilities": adjudication["events"],
        }
    relationships = [
        {"code": "single_laffa_requires_one_result", "if": {"output.kind": "laffa"}, "requires": {"output.resultCount": 1}},
        {"code": "exact_origin_requires_one_result", "if": {"scenario.origin.kind": "exact_trace_prefix"}, "requires": {"output.resultCount": 1}},
        {"code": "video_count_fits_results", "requiresRelation": {"left": "renderPolicy.videoCount", "operator": "<=", "right": "output.resultCount"}},
        {"code": "disabled_render_requires_zero", "if": {"renderPolicy.mode": "disabled"}, "requires": {"renderPolicy.videoCount": 0}},
        {"code": "shards_divide_seed_count", "requiresRelation": {"left": "resources.seedCount", "operator": "divisible_by", "right": "resources.seedsPerShard"}},
        {"code": "proof_resource_budget_identity", "requiresRelation": {"left": "proof.budgetProfile", "operator": "equals", "right": "resources.proofBudgetProfile"}},
    ]
    core = {
        "schemaVersion": 1,
        "recipeSchemaVersion": 4,
        "sections": list(SCENARIO_SECTIONS),
        "logicalOperators": sorted(SCENARIO_OPERATORS),
        "tacticKinds": sorted(item.value for item in TacticalKind),
        "lessonTags": lesson_tag_catalog_v1(),
        "proof": {
            "modes": list(PROOF_MODES), "visibilities": list(PROOF_VISIBILITIES), "strategyClasses": list(PROOF_STRATEGY_CLASSES),
            "claims": list(PROOF_CLAIMS), "utilities": list(PROOF_UTILITIES), "lineUniverses": list(LINE_UNIVERSES),
            "utilityEncodings": {key: list(values) for key, values in PROOF_UTILITY_ENCODINGS.items()},
            "horizonUnits": list(PROOF_HORIZON_UNITS), "boundaryEvaluators": list(OPERATOR_PROOF_BOUNDARY_EVALUATORS),
            "budgetProfiles": budget_profiles,
        },
        "fields": fields,
        "relationships": relationships,
        "expressionLimits": {
            "maximumDepth": SCENARIO_EXPRESSION_MAXIMUM_DEPTH,
            "maximumNodes": SCENARIO_EXPRESSION_MAXIMUM_NODES,
            "maximumChildren": SCENARIO_EXPRESSION_MAXIMUM_CHILDREN,
            "maximumSerializedBytes": SCENARIO_EXPRESSION_MAXIMUM_SERIALIZED_BYTES,
            "maximumCountRange": SCENARIO_EXPRESSION_MAXIMUM_COUNT_RANGE,
        },
        "exactTrace": exact_trace,
        "executionCombinations": combinations,
        "decisionSelectorSchemas": selectors,
        "policyCatalog": {
            **copy.deepcopy(dict(policy_catalog)),
            "seed": {"type": "integer", "minimum": 0, "maximum": 2_147_483_647},
            "defaultProfiles": copy.deepcopy(dict(default_policy_profiles)),
        },
        "localizationKeys": {
            "interface": list(_V4_INTERFACE_COPY_KEYS),
            "values": copy.deepcopy(_V4_VALUE_MESSAGE_KEYS),
            "fields": copy.deepcopy(_V4_FIELD_MESSAGE_KEYS),
            "exactArguments": {
                "action": "research_scenario_argument_actions", "actor": "research_scenario_argument_actor", "card": "research_scenario_argument_card",
                "declarationOrder": "research_scenario_argument_declaration_order", "ekka": "research_scenario_argument_ekka", "mode": "research_scenario_argument_mode",
                "owner": "research_scenario_argument_owner", "projectId": "research_scenario_argument_project_kind", "state": "research_scenario_argument_state", "trumpSuit": "research_scenario_argument_suit",
            },
            "predicateHelp": {
                kind: _V4_PREDICATE_HELP_KEYS[kind]
                for kind in predicate_kinds
                if kind in _V4_PREDICATE_HELP_KEYS
            },
        },
    }
    return {**core, "sha256": canonical_sha256(core)}


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return copy.deepcopy(dict(value))


def _validate_selected_action(kind: str, value: str | None) -> None:
    if value is None:
        return
    if kind == "play":
        try:
            canonical = LegalPlayActionV1.from_code(value).code
        except (TypeError, ValueError) as exc:
            raise ValueError("proof selectedAction is invalid for play") from exc
        if canonical != value:
            raise ValueError("proof selectedAction is not canonical for play")
        return
    pattern = {
        "bid": _BID_ACTION,
        "escalation": _ESCALATION_ACTION,
        "project": _PROJECT_ACTION,
    }[kind]
    if pattern.fullmatch(value) is None:
        raise ValueError(f"proof selectedAction is invalid for {kind}")


def _closed(raw: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {', '.join(unknown)}")


def _strict_int(value: Any, label: str, minimum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _strict_number(value: Any, label: str, minimum: float) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{label} must be a number")
    result = float(value)
    if result < minimum or result != result or result in {float("inf"), float("-inf")}:
        raise ValueError(f"{label} is invalid")
    return result


@dataclass(frozen=True, slots=True)
class ResearchRecipeV4:
    code: str
    title_ar: str
    configuration: Mapping[str, Any]
    scenario: ResearchScenarioV1
    proof: ScenarioProofRequestV1
    qualification: Mapping[str, Any]
    resources: Mapping[str, Any]
    diversity: Mapping[str, Any]
    render_policy: Mapping[str, Any]
    status: str
    capability_plan: ScenarioCapabilityPlanV1
    sha256: str
    output: Mapping[str, Any] = field(
        default_factory=lambda: copy.deepcopy(_DEFAULT_OUTPUT)
    )
    legacy_contract: bool = False
    playing_quality: ScenarioPlayingQualityPolicyV1 | None = None
    generation: Mapping[str, Any] | None = None

    @classmethod
    def build(
        cls,
        value: Mapping[str, Any],
        *,
        compiler: ScenarioCompilerV1 | None = None,
    ) -> "ResearchRecipeV4":
        raw = _object(value, "research recipe v4")
        _closed(
            raw,
            {
                "schemaVersion",
                "code",
                "titleAr",
                "output",
                "configuration",
                "scenario",
                "proof",
                "playingQuality",
                "generation",
                "qualification",
                "resources",
                "diversity",
                "renderPolicy",
                "status",
                "capabilityPlan",
                "sha256",
            },
            "research recipe v4",
        )
        if raw.get("schemaVersion") != 4 or type(raw.get("schemaVersion")) is not int:
            raise ValueError("research recipe v4 schemaVersion must be integer 4")
        title_ar = str(raw.get("titleAr") or "").strip()
        if not 3 <= len(title_ar) <= 160:
            raise ValueError("research recipe v4 titleAr length is invalid")
        authority = ResearchConfigurationAuthorityV1.build(
            _object(raw.get("configuration"), "research recipe v4 configuration")
        )
        scenario = ResearchScenarioV1.build(
            _object(raw.get("scenario"), "research recipe v4 scenario")
        )
        proof = ScenarioProofRequestV1.build(
            _object(raw.get("proof"), "research recipe v4 proof")
        )
        validate_proof_boundary_scope_v1(proof, str(scenario.decision_point["kind"]))
        playing_quality = (
            ScenarioPlayingQualityPolicyV1.build(raw["playingQuality"])
            if "playingQuality" in raw else None
        )
        generation = (build_family_generation_request_v1(raw["generation"])
                      if "generation" in raw else None)
        if generation is not None and (scenario.origin.kind != "search"
                or scenario.decision_point["kind"] != "play" or playing_quality is None):
            raise ValueError("family generation requires search-play with playingQuality")
        _validate_selected_action(
            str(scenario.decision_point["kind"]), proof.selected_action
        )
        qualification = {**_DEFAULT_QUALIFICATION, **_object(raw.get("qualification", {}), "research recipe v4 qualification")}
        _closed(qualification, set(_DEFAULT_QUALIFICATION), "research recipe v4 qualification")
        if type(qualification["requireObserverSafety"]) is not bool or qualification["requireObserverSafety"] is not True:
            raise ValueError("Research V4 requires observer safety")
        if type(qualification["requireAllDecisionAudit"]) is not bool or qualification["requireAllDecisionAudit"] is not True:
            raise ValueError("Research V4 requires complete decision auditing")
        _strict_number(qualification["minimumConfidence"], "minimumConfidence", 0.0)
        if float(qualification["minimumConfidence"]) > 1.0:
            raise ValueError("minimumConfidence cannot exceed 1")
        _strict_number(qualification["minimumSurroundingQuality"], "minimumSurroundingQuality", 0.0)
        if float(qualification["minimumSurroundingQuality"]) > 100.0:
            raise ValueError("minimumSurroundingQuality cannot exceed 100")
        _strict_number(qualification["maximumRegret"], "maximumRegret", 0.0)

        resource_input = _object(
            raw.get("resources", {}), "research recipe v4 resources"
        )
        declared = str(raw.get("sha256") or "")
        legacy_resources = bool(
            declared
            and _COMPATIBILITY_ONLY_RESOURCE_FIELDS.intersection(resource_input)
        )
        historical_v2_resources = bool(
            declared and resource_input.get("generationPolicy") == "ortools_cp_sat_v2"
        )
        resource_defaults = copy.deepcopy(
            _LEGACY_RESOURCE_DEFAULTS if legacy_resources else _DEFAULT_RESOURCES
        )
        resources = {**resource_defaults, **resource_input}
        _closed(resources, set(resource_defaults) | {"shardWorkerCount"}, "research recipe v4 resources")
        if legacy_resources:
            if resources["generationLane"] not in {"ortools", "organic_runtime"}:
                raise ValueError("research recipe v4 generationLane is unsupported")
            if resources["generationPolicy"] not in {
                "natural_policy",
                "scenario_guided_actor_safe",
                "ortools_cp_sat_v2",
                "ortools_cp_sat_v3",
            }:
                raise ValueError("research recipe v4 generationPolicy is unsupported")
        elif historical_v2_resources:
            if resources["generationLane"] != "ortools":
                raise ValueError("historical OR-Tools V4 requires generationLane ortools")
        elif resources["generationLane"] != "ortools":
            raise ValueError("new Research V4 authoring requires generationLane ortools")
        elif resources["generationPolicy"] != "ortools_cp_sat_v3":
            raise ValueError(
                "new Research V4 authoring requires generationPolicy ortools_cp_sat_v3"
            )
        for key in ("seedStart", "seedCount", "seedsPerShard"):
            resources[key] = _strict_int(resources[key], key, 0 if key == "seedStart" else 1)
        if "shardWorkerCount" in resources:
            if legacy_resources or historical_v2_resources:
                raise ValueError("historical Research V4 cannot override shard workers")
            resources["shardWorkerCount"] = _strict_int(resources["shardWorkerCount"], "shardWorkerCount", 1)
        if legacy_resources:
            for key in ("candidatesPerSeed", "maximumConflicts"):
                resources[key] = _strict_int(resources[key], key, 1)
        resources["solverTimeoutSeconds"] = _strict_number(resources["solverTimeoutSeconds"], "solverTimeoutSeconds", 0.001)
        if resources["seedCount"] % resources["seedsPerShard"] != 0:
            raise ValueError("seedCount must be divisible by seedsPerShard")
        if (
            legacy_resources
            and scenario.origin.kind == "exact_trace_prefix"
            and resources["generationPolicy"]
            not in {"natural_policy", "ortools_cp_sat_v2", "ortools_cp_sat_v3"}
        ):
            raise ValueError("exact trace prefix cannot use scenario-guided generation")

        output = {
            **_DEFAULT_OUTPUT,
            **_object(raw.get("output", {}), "research recipe v4 output"),
        }
        _closed(output, set(_DEFAULT_OUTPUT), "research recipe v4 output")
        if type(output["kind"]) is not str or output["kind"] not in {
            "laffa",
            "laffa_collection",
        }:
            raise ValueError("research recipe v4 output kind is unsupported")
        output["resultCount"] = _strict_int(
            output["resultCount"], "output.resultCount", 1
        )
        if output["resultCount"] > 1_000:
            raise ValueError("output.resultCount cannot exceed 1000")
        if output["kind"] == "laffa" and output["resultCount"] != 1:
            raise ValueError("single laffa output requires exactly one result")
        if (
            scenario.origin.kind == "exact_trace_prefix"
            and output["resultCount"] != 1
        ):
            raise ValueError("exact trace prefix can produce exactly one result")

        diversity = {**_DEFAULT_DIVERSITY, **_object(raw.get("diversity", {}), "research recipe v4 diversity")}
        _closed(diversity, set(_DEFAULT_DIVERSITY), "research recipe v4 diversity")
        if diversity["schemaVersion"] != 1 or type(diversity["schemaVersion"]) is not int:
            raise ValueError("research recipe v4 diversity schemaVersion must be integer 1")
        if not isinstance(diversity["cellDimensions"], list) or not diversity["cellDimensions"]:
            raise ValueError("research recipe v4 diversity cellDimensions must be non-empty")
        diversity["topKPerCell"] = _strict_int(diversity["topKPerCell"], "topKPerCell", 1)
        diversity["rejectedAuditModulus"] = _strict_int(diversity["rejectedAuditModulus"], "rejectedAuditModulus", 1)
        QualityDiversityProfileV1.build(diversity)

        render_policy_input = _object(
            raw.get("renderPolicy", {}), "research recipe v4 renderPolicy"
        )
        render_policy = {**_DEFAULT_RENDER_POLICY, **render_policy_input}
        if (
            render_policy.get("mode") == "disabled"
            and "videoCount" not in render_policy_input
        ):
            render_policy["videoCount"] = 0
        _closed(render_policy, set(_DEFAULT_RENDER_POLICY), "research recipe v4 renderPolicy")
        if type(render_policy["mode"]) is not str or render_policy["mode"] not in {
            "on_demand",
            "disabled",
        }:
            raise ValueError("research recipe v4 renderPolicy mode is unsupported")
        render_policy["videoCount"] = _strict_int(
            render_policy["videoCount"], "renderPolicy.videoCount", 0
        )
        if render_policy["videoCount"] > 100:
            raise ValueError("renderPolicy.videoCount cannot exceed 100")
        if render_policy["videoCount"] > output["resultCount"]:
            raise ValueError("renderPolicy.videoCount cannot exceed output.resultCount")
        if render_policy["mode"] == "disabled" and render_policy["videoCount"] != 0:
            raise ValueError("disabled rendering requires videoCount zero")
        status = str(raw.get("status") or "published")
        if status not in {"draft", "published", "retired"}:
            raise ValueError("research recipe v4 status is unsupported")
        capability_plan = (compiler or ScenarioCompilerV1()).compile(scenario, proof)
        # Validation endpoints must be able to normalize an invalid draft or
        # publication request and return the compiler's typed diagnostics.
        # Publication remains fail-closed in ResearchFoundryServiceV1.
        supplied_code = str(raw.get("code") or "").strip().upper()
        if supplied_code and _CODE.fullmatch(supplied_code) is None:
            raise ValueError("research recipe v4 code is invalid")
        core = {
            "schemaVersion": 4,
            "code": supplied_code,
            "titleAr": title_ar,
            "output": output,
            "configuration": authority.to_dict(),
            "scenario": scenario.to_dict(),
            "proof": proof.to_dict(),
            "qualification": qualification,
            "resources": resources,
            "diversity": diversity,
            "renderPolicy": render_policy,
            "status": status,
            "capabilityPlan": capability_plan.to_dict(),
        }
        if playing_quality is not None:
            core["playingQuality"] = playing_quality.to_dict()
        if generation is not None:
            core["generation"] = generation
        digest = canonical_sha256(core)
        code = supplied_code or f"RCP-{digest[:10].upper()}"
        if not supplied_code:
            core["code"] = code
            digest = canonical_sha256(core)
        legacy_contract = False
        if declared and declared != digest:
            legacy_core = {
                key: copy.deepcopy(item)
                for key, item in core.items()
                if key != "output"
            }
            legacy_core["renderPolicy"] = {"mode": render_policy["mode"]}
            legacy_digest = canonical_sha256(legacy_core)
            legacy_contract = (
                "output" not in raw
                and "videoCount" not in render_policy_input
                and declared == legacy_digest
            )
            if not legacy_contract:
                raise ValueError("research recipe v4 checksum mismatch")
            digest = legacy_digest
        declared_plan = raw.get("capabilityPlan")
        if declared_plan is not None and dict(declared_plan) != capability_plan.to_dict():
            raise ValueError("research recipe v4 capability plan is stale")
        return cls(
            code,
            title_ar,
            authority.to_dict(),
            scenario,
            proof,
            qualification,
            resources,
            diversity,
            render_policy,
            status,
            capability_plan,
            digest,
            output=output,
            legacy_contract=legacy_contract,
            playing_quality=playing_quality,
            generation=generation,
        )

    @property
    def profiles(self) -> Mapping[str, Any]:
        authority = ResearchConfigurationAuthorityV1.build(self.configuration)
        identities = authority.identities
        return {
            "domainHash": identities["domain"]["sha256"],
            "bundleHash": authority.semantic_bundle_hash,
            "rulesHash": identities["rules"]["sha256"],
            "conventionHash": identities["convention"]["sha256"],
            "agentRelease": identities["agent"]["id"],
            "analysisProfileHash": identities["analyzer"]["sha256"],
            "scoreProfileHash": identities["score"]["sha256"],
        }

    @property
    def situation(self) -> Mapping[str, Any]:
        """Empty legacy projection; V4 execution must consume the scenario plan."""

        return {"projects": [], "contracts": [], "escalations": []}

    @property
    def tactics(self) -> tuple[Mapping[str, Any], ...]:
        return ()

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schemaVersion": 4,
            "code": self.code,
            "titleAr": self.title_ar,
            "output": copy.deepcopy(dict(self.output)),
            "configuration": copy.deepcopy(dict(self.configuration)),
            "scenario": self.scenario.to_dict(),
            "proof": self.proof.to_dict(),
            "qualification": copy.deepcopy(dict(self.qualification)),
            "resources": copy.deepcopy(dict(self.resources)),
            "diversity": copy.deepcopy(dict(self.diversity)),
            "renderPolicy": copy.deepcopy(dict(self.render_policy)),
            "status": self.status,
            "capabilityPlan": self.capability_plan.to_dict(),
            "sha256": self.sha256,
        }
        if self.legacy_contract:
            value.pop("output")
            value["renderPolicy"] = {"mode": self.render_policy["mode"]}
        if self.playing_quality is not None:
            value["playingQuality"] = self.playing_quality.to_dict()
        if self.generation is not None:
            value["generation"] = copy.deepcopy(dict(self.generation))
        return value


__all__ = ["ResearchRecipeV4"]
