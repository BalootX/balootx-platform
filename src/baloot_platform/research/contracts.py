from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Mapping
from urllib.parse import urlparse

if TYPE_CHECKING:
    from baloot_platform.configuration import EffectiveRuntimeConfigV2


SEMANTIC_DISCREPANCY_CATEGORIES = frozenset({
    "compiler_parity_defect",
    "runtime_transition_defect",
    "hard_constraint_violation",
    "observer_information_defect",
})
HISTORICAL_RECORD_REQUIRES_READJUDICATION = (
    "historical_record_requires_readjudication"
)


def effective_disagreement_category(
    category: str, schema_version: int = 1
) -> tuple[str, bool]:
    """Classify immutable historical records under the current taxonomy."""
    value = str(category)
    if value in SEMANTIC_DISCREPANCY_CATEGORIES:
        return value, True
    return HISTORICAL_RECORD_REQUIRES_READJUDICATION, False


_OUTPUT_KINDS = {"laffa", "laffa_collection", "contextualized_laffa", "sakka"}
_TRUTH_LEVELS = {
    "exact_optimal": 4,
    "robust_best_known": 3,
    "debatable": 2,
    "provisional": 1,
    "rejected": 0,
}
_TACTICS = {
    "only_winning_move",
    "small_card_big_gain",
    "large_card_sacrifice",
    "kabout_creation",
    "kabout_preservation",
    "kabout_prevention",
    "contract_rescue",
    "entry_unblocking",
    "trump_control",
    "partner_enablement",
    "project_bidding",
    "debate",
}
_TACTICAL_DEAL_BALANCE_GATED_TACTICS = frozenset(
    {"only_winning_move", "small_card_big_gain"}
)
_CONTRACTS = {"sun", "hokum", "ashkal"}
_SEATS = {"north", "west", "south", "east"}
_PROJECT_KINDS = {
    "siri", "fifty", "hundred_sequence", "hundred_four_kind", "four_hundred"
}
_ESCALATION_LEVELS = {"normal", "double", "triple", "four", "gahwa"}
_MODES = {"hard", "preference", "qualification_gate"}
_V3_TACTIC_MODES = {"preference", "qualification_gate"}
_DEEP_VERIFICATION_POLICIES = {"off", "required_gates", "all_candidates"}
_PLAY_CONSTRUCTION_MODES = {
    "actor_safe_agent_verified",
    "cp_sat_actor_safe_late_game_v1",
    "cp_sat_feasibility_line",
}
_AUCTION_CONSTRUCTION_MODES = {
    "actor_safe_agent_verified",
    "cp_sat_actor_safe_decision_v1",
    "cp_sat_feasibility_line",
}
_CONSTRUCTION_BUYER_RELATIONS = {"automatic", "objective", "partner"}
_SUITS = {"spades", "hearts", "diamonds", "clubs"}
_CARD_CODE = re.compile(r"^(?:S|H|D|C)(?:7|8|9|10|J|Q|K|A)$")
_RESEARCH_CONFIGURATION_IDENTITIES = {
    "domain",
    "rules",
    "convention",
    "belief",
    "agent",
    "analyzer",
    "score",
    "decisionScore",
}
_RESEARCH_IDENTITY_SOURCE_FAMILIES = {
    "domain": frozenset({"rules", "conventions", "product"}),
    "rules": frozenset({"rules"}),
    "convention": frozenset({"conventions"}),
    "belief": frozenset({"belief"}),
    "agent": frozenset({"agents", "challenge"}),
    "analyzer": frozenset({
        "challenge",
        "analysis",
        "belief",
        "match_value",
        "performance",
        "rules",
        "conventions",
    }),
    "score": frozenset({"challenge"}),
    "decisionScore": frozenset({"challenge"}),
}
_EFFECTIVE_RUNTIME_CONFIG_MEDIA_TYPE = (
    "application/vnd.balootx.effective-runtime-config-v2+json"
)

# V3 carries this field only because the historical strategic-auction profile
# contract requires it.  The selected published bundle remains the semantic
# owner at challenge.confirmation.bid_samples; the current published bundle is
# frozen at 192.  V1/V2 records are intentionally not migrated to this value.
RESEARCH_STRATEGIC_AUCTION_DEEP_BID_SAMPLES_V3 = 192
BID_DECISION_OBJECTIVE_V1 = "expected_round_margin"
RESEARCH_STRATEGIC_AUCTION_TIE_BREAK_V1 = (
    "round_lead_probability_then_best_share_then_legal_action_order_v1"
)
_LEGACY_RESEARCH_STRATEGIC_AUCTION_TIE_BREAK_V1 = (
    "win_probability_then_best_share_then_action_id"
)


_RESEARCH_RECIPE_FIELD_DEFAULTS: dict[str, Any] = {
    "output.kind": "laffa",
    "output.resultCount": 1,
    "situation.dealer": "east",
    "situation.buyer": None,
    "situation.objectiveSeat": None,
    "situation.constructionBuyerRelation": "automatic",
    "situation.purchaseRound": None,
    "situation.purchaseSuits": [],
    "situation.trumpSuit": None,
    "situation.contracts": ["sun"],
    "situation.matchScore": [0, 0],
    "situation.requiredCards": {},
    "situation.forbiddenCards": {},
    "situation.finalSuitCountRanges": {},
    "situation.projects": [],
    "situation.projects[].mode": "hard",
    "situation.escalations": [],
    "tactics": [{
        "kind": "small_card_big_gain",
        "mode": "qualification_gate",
        "minimumScore": 70.0,
        "weight": 1.0,
        "perspectives": [],
    }],
    "tactics[].mode": "preference",
    "tactics[].minimumScore": 0.0,
    "tactics[].weight": 1.0,
    "tactics[].perspectives": [],
    "qualification.minimumTruth": "robust_best_known",
    "qualification.minimumConfidence": 0.75,
    "qualification.minimumSurroundingQuality": 70.0,
    "qualification.maximumRegret": 0.0,
    "resources.generationLane": "ortools",
    "resources.seedStart": 1,
    "resources.seedCount": 10,
    "resources.seedsPerShard": 10,
    "resources.candidatesPerSeed": 1,
    "resources.solverTimeoutSeconds": 15.0,
    "resources.maximumConflicts": 250_000,
    "resources.analysisMode": "cascaded",
    "resources.deepVerificationPolicy": "required_gates",
    "resources.tacticalConstruction": None,
    "resources.tacticalConstruction.enabled": False,
    "resources.tacticalConstruction.maximumPayoffTricks": 4,
    "resources.tacticalConstruction.minimumLaterRawPoints": 10,
    "resources.tacticalConstruction.requirePrecursorForGatedTactics": True,
    "resources.constructionPreScreen": None,
    "resources.constructionPreScreen.enabled": False,
    "resources.constructionPreScreen.maximumPlayDecisions": 4,
    "resources.constructionPreScreen.maximumBidRegret": 0.0,
    "resources.constructionPreScreen.maximumPlayRegret": 0.0,
    "resources.constructionPreScreen.maximumProbabilityDrop": 0.0,
    "resources.constructionPreScreen.minimumConfidence": 0.0,
    "resources.constructionPreScreen.rejectCatastrophicCrossing": True,
    "resources.tacticalViabilityPreScreen": None,
    "resources.tacticalViabilityPreScreen.enabled": False,
    "resources.tacticalViabilityPreScreen.revision": "1",
    "resources.tacticalViabilityPreScreen.maximumDecisions": 8,
    "resources.tacticalViabilityPreScreen.requiredTacticsOnly": True,
    "resources.tacticalViabilityPreScreen.rejectWhenExactAbsent": True,
    "resources.tacticalDealBalance": None,
    "resources.tacticalDealBalance.enabled": False,
    "resources.tacticalDealBalance.maximumTeamHighCardDifference": 2,
    "resources.tacticalDealBalance.minimumHighCardsPerSeat": 2,
    "resources.tacticalDealBalance.maximumHighCardsPerSeat": 4,
    "resources.tacticalDealBalance.minimumTargetLowCards": 1,
    "resources.auctionConstructionMode": "actor_safe_agent_verified",
    "resources.strategicAuction": None,
    "resources.strategicAuction.requestedSamples": (
        RESEARCH_STRATEGIC_AUCTION_DEEP_BID_SAMPLES_V3
    ),
    "resources.strategicAuction.integerScale": 1_000,
    "resources.strategicAuction.maximumConflicts": 50_000,
    "resources.strategicAuction.maximumSelectionSeconds": 5.0,
    "resources.strategicAuction.requireOptimalForQualification": True,
    "resources.playConstructionMode": "actor_safe_agent_verified",
    "resources.strategicPlay": None,
    "resources.strategicPlay.minimumPlayedCards": 24,
    "resources.strategicPlay.requestedSamples": 32,
    "resources.strategicPlay.exactAllocationLimit": 500,
    "resources.strategicPlay.maximumSamplingAttempts": 500,
    "resources.strategicPlay.maximumWorlds": 500,
    "resources.strategicPlay.maximumContinuationSteps": 20_000,
    "resources.strategicPlay.maximumPlies": 8,
    "resources.strategicPlay.maximumSelectionSeconds": 30.0,
    "resources.strategicPlay.requireExactForQualification": True,
    "resources.strategicPlay.outcomeBoundary": 1,
    "diversity": {
        "schemaVersion": 1,
        "cellDimensions": [
            "contract",
            "tactical_detector",
            "acting_seat",
            "source",
        ],
        "topKPerCell": 3,
        "rejectedAuditModulus": 100,
    },
    "diversity.cellDimensions": [
        "contract",
        "tactical_detector",
        "acting_seat",
        "source",
    ],
    "diversity.topKPerCell": 3,
    "diversity.rejectedAuditModulus": 100,
    "status": "published",
}

_RESEARCH_RECIPE_REQUIRED_FIELDS = frozenset({
    "titleAr",
    "configuration.bundleRevisionId",
    "situation.projects[].seat",
    "situation.projects[].contract",
    "situation.projects[].kind",
    "situation.projects[].value",
    "situation.projects[].highRank",
    "tactics[].kind",
})

_RESEARCH_RECIPE_FIELD_EXAMPLES: dict[str, Any] = {
    "titleAr": "لفة صن فيها لعب وحيد",
    "configuration.bundleRevisionId": "BLR-PUBLISHED-001",
    "situation.requiredCards": {"south": ["SA", "H7"]},
    "situation.forbiddenCards": {"south": ["C7"]},
    "situation.finalSuitCountRanges": {
        "south": {"spades": {"minimum": 2, "maximum": 4}}
    },
    "situation.projects": [{
        "seat": "south",
        "contract": "sun",
        "kind": "four_hundred",
        "value": 400,
        "highRank": "ace",
        "mode": "hard",
    }],
    "situation.projects[].seat": "south",
    "situation.projects[].contract": "sun",
    "situation.projects[].kind": "four_hundred",
    "situation.projects[].value": 400,
    "situation.projects[].highRank": "ace",
    "tactics": [{
        "kind": "only_winning_move",
        "mode": "qualification_gate",
        "minimumScore": 70.0,
        "weight": 1.0,
        "perspectives": ["south"],
    }],
    "tactics[].kind": "only_winning_move",
}

_RESEARCH_PROFILE_ACTIVATION_DEFAULTS: dict[str, dict[str, Any]] = {
    "resources.constructionPreScreen": {
        "schemaVersion": 1,
        "id": "actor_safe_pre_screen_v1",
        "revision": "1",
        "enabled": False,
        "maximumPlayDecisions": 4,
        "maximumBidRegret": 0.0,
        "maximumPlayRegret": 0.0,
        "maximumProbabilityDrop": 0.0,
        "minimumConfidence": 0.0,
        "rejectCatastrophicCrossing": True,
    },
    "resources.tacticalViabilityPreScreen": {
        "schemaVersion": 1,
        "id": "exact_tactical_viability_screen_v1",
        "revision": "1",
        "enabled": False,
        "maximumDecisions": 8,
        "requiredTacticsOnly": True,
        "rejectWhenExactAbsent": True,
    },
    "resources.tacticalConstruction": {
        "schemaVersion": 1,
        "id": "tactical_play_precursor_v1",
        "revision": 1,
        "enabled": False,
        "maximumPayoffTricks": 4,
        "minimumLaterRawPoints": 10,
        "requirePrecursorForGatedTactics": True,
    },
    "resources.tacticalDealBalance": {
        "schemaVersion": 1,
        "id": "balanced_tactical_deal_v1",
        "revision": 1,
        "enabled": False,
        "maximumTeamHighCardDifference": 2,
        "minimumHighCardsPerSeat": 2,
        "maximumHighCardsPerSeat": 4,
        "minimumTargetLowCards": 1,
    },
    "resources.strategicAuction": {
        "schemaVersion": 1,
        "requestedSamples": RESEARCH_STRATEGIC_AUCTION_DEEP_BID_SAMPLES_V3,
        "integerScale": 1_000,
        "maximumConflicts": 50_000,
        "maximumSelectionSeconds": 5.0,
        "objective": "expected_round_margin",
        "tieBreak": RESEARCH_STRATEGIC_AUCTION_TIE_BREAK_V1,
        "requireOptimalForQualification": True,
    },
    "resources.strategicPlay": {
        "schemaVersion": 1,
        "minimumPlayedCards": 24,
        "requestedSamples": 32,
        "exactAllocationLimit": 500,
        "maximumSamplingAttempts": 500,
        "maximumWorlds": 500,
        "maximumContinuationSteps": 20_000,
        "maximumPlies": 8,
        "maximumSelectionSeconds": 30.0,
        "requireExactForQualification": True,
        "continuationPolicy": "observer_safe_heuristic_v1",
        "policyProfileId": "maximin_margin",
        "policyRevision": 1,
        "outcomeBoundary": 1,
        "utility": "actor_team_awarded_point_margin",
        "tieBreak": "maximin_then_total_then_card_id",
    },
}

_RESEARCH_RECIPE_RELATION_COPY: dict[str, tuple[str, str]] = {
    "explicit_buyer_excludes_relation": (
        "research_relation_badge_buyer",
        "research_relation_badge_buyer_help",
    ),
    "construction_relation_requires_objective_seat": (
        "research_label_objective_seat",
        "research_objective_help",
    ),
    "trump_requires_hokum": (
        "research_relation_badge_trump",
        "research_relation_badge_trump_help",
    ),
    "four_hundred_requires_sun": (
        "research_relation_badge_project",
        "research_relation_badge_project_help",
    ),
    "high_escalation_requires_hokum": (
        "research_relation_badge_escalation",
        "research_relation_badge_escalation_help",
    ),
    "strategic_auction_requires_ortools": (
        "research_relation_badge_auction",
        "research_relation_badge_auction_help",
    ),
    "strategic_auction_profile_requires_mode": (
        "research_relation_auction_profile",
        "research_relation_auction_profile_help",
    ),
    "strategic_play_requires_ortools": (
        "research_relation_badge_play",
        "research_relation_badge_play_help",
    ),
    "strategic_play_profile_requires_mode": (
        "research_relation_play_profile",
        "research_relation_play_profile_help",
    ),
    "actor_safe_auction_requires_ortools": (
        "research_relation_badge_auction",
        "research_relation_badge_auction_help",
    ),
    "actor_safe_auction_requires_actor_safe_play": (
        "research_relation_actor_safe_play",
        "research_relation_actor_safe_play_help",
    ),
    "organic_forbids_ortools_solver_controls": (
        "research_lane_organic",
        "research_lane_organic_help",
    ),
    "organic_forbids_ortools_structural_constraints": (
        "research_lane_organic",
        "research_lane_organic_help",
    ),
    "actor_safe_auction_forbids_escalation": (
        "research_relation_actor_safe_escalation",
        "research_relation_actor_safe_escalation_help",
    ),
    "strategic_play_samples_fit_worlds": (
        "research_relation_samples_worlds",
        "research_relation_samples_worlds_help",
    ),
    "strategic_play_exact_limit_fits_worlds": (
        "research_relation_exact_worlds",
        "research_relation_exact_worlds_help",
    ),
    "strategic_play_attempts_cover_samples": (
        "research_relation_attempts_samples",
        "research_relation_attempts_samples_help",
    ),
    "strategic_play_plies_cover_tail": (
        "research_relation_plies_tail",
        "research_relation_plies_tail_help",
    ),
    "hard_tactics_require_verification": (
        "research_relation_badge_proof",
        "research_relation_badge_proof_help",
    ),
    "tactical_precursor_is_construction_only": (
        "research_relation_precursor",
        "research_relation_precursor_help",
    ),
    "tactical_precursor_requires_small_card_gate": (
        "research_relation_precursor_tactic",
        "research_relation_precursor_tactic_help",
    ),
    "tactical_precursor_supports_only_small_card_gate": (
        "research_relation_precursor_tactic",
        "research_relation_precursor_tactic_help",
    ),
    "tactical_viability_screen_requires_proof": (
        "research_relation_badge_tactical_screen",
        "research_relation_badge_tactical_screen_help",
    ),
    "tactical_viability_required_only_requires_gate": (
        "research_relation_required_tactic",
        "research_relation_required_tactic_help",
    ),
    "tactical_deal_balance_requires_supported_tactic": (
        "research_relation_badge_deal_balance",
        "research_relation_badge_deal_balance_help",
    ),
    "tactical_deal_balance_requires_ortools": (
        "research_relation_badge_deal_balance",
        "research_relation_badge_deal_balance_help",
    ),
    "tactical_deal_balance_bounds_ordered": (
        "research_relation_balance_bounds",
        "research_relation_balance_bounds_help",
    ),
    "project_contract_matches_requested_contract": (
        "research_relation_project_contract",
        "research_relation_project_contract_help",
    ),
    "at_most_one_hard_project": (
        "research_relation_badge_project",
        "research_relation_badge_project_help",
    ),
    "required_cards_unique_across_seats": (
        "research_relation_required_unique",
        "research_relation_required_unique_help",
    ),
    "required_and_forbidden_cards_are_disjoint": (
        "research_relation_cards_disjoint",
        "research_relation_cards_disjoint_help",
    ),
    "suit_ranges_form_eight_card_hand": (
        "research_relation_suit_ranges",
        "research_relation_suit_ranges_help",
    ),
    "tactical_kinds_are_unique": (
        "research_relation_tactics_unique",
        "research_relation_tactics_unique_help",
    ),
    "single_laffa_requires_one_result": (
        "research_label_output",
        "research_step_output_help",
    ),
    "required_cards_fit_suit_ranges": (
        "research_relation_suit_ranges",
        "research_relation_suit_ranges_help",
    ),
    "tactics_require_exactly_one_contract": (
        "research_step_situation",
        "research_step_situation_help",
    ),
}


def _research_field_help_key(path: str) -> str:
    if path == "titleAr":
        return "research_review_help"
    if path.startswith("configuration.") or path.startswith("profiles."):
        return "research_bundle_help"
    if path.startswith("output."):
        return "research_step_output_help"
    if path in {"situation.requiredCards", "situation.forbiddenCards"}:
        return "research_card_picker_help"
    if path == "situation.finalSuitCountRanges":
        return "research_suit_ranges_help"
    if path.startswith("situation.projects"):
        return "research_relation_badge_project_help"
    if path == "situation.objectiveSeat":
        return "research_objective_help"
    if path == "situation.constructionBuyerRelation":
        return "research_relation_help"
    if path == "situation.purchaseSuits":
        return "research_purchase_suits_help"
    if path == "situation.escalations":
        return "research_escalation_help"
    if path.startswith("situation."):
        return "research_step_situation_help"
    if path.startswith("tactics"):
        return "research_add_tactic_help"
    if path.startswith("qualification."):
        return "research_step_proof_help"
    if path.startswith("resources.constructionPreScreen"):
        return "research_prescreen_help"
    if path.startswith("resources.tacticalViabilityPreScreen"):
        return "research_tactical_prescreen_help"
    if path.startswith("resources.tacticalDealBalance"):
        return "research_deal_balance_help"
    if path.startswith("resources.tacticalConstruction"):
        return "research_tactical_construction_help"
    if path.startswith("resources.strategicAuction"):
        return "research_auction_strategic_scope"
    if path.startswith("resources.strategicPlay"):
        return "research_play_construction_strategic_help"
    if path == "resources.auctionConstructionMode":
        return "research_auction_construction_agent_help"
    if path == "resources.playConstructionMode":
        return "research_play_construction_agent_help"
    if path == "resources.generationLane":
        return "research_lane_ortools_help"
    if path == "resources.deepVerificationPolicy":
        return "research_deep_verification_help"
    if path.startswith("resources."):
        return "research_step_budget_help"
    if path.startswith("diversity"):
        return "research_diversity_dimensions_help"
    if path.startswith("renderPolicy"):
        return "research_render_on_demand_help"
    return "research_review_help"


def _research_field_value_type(control: str) -> str:
    return {
        "boolean": "boolean",
        "enum": "string",
        "optional_enum": "string_or_null",
        "integer": "integer",
        "number": "number",
        "integer_pair": "array",
        "multi_enum": "array",
        "seat_card_picker": "object",
        "seat_suit_ranges": "object",
        "project_constraint": "array",
        "repeatable_tactic": "array",
        "profile": "object_or_null",
        "published_revision": "string",
        "text": "string",
        "lifecycle": "string",
        "checksum": "string_or_null",
        "identity": "string_or_null",
    }[control]


def _research_field_applicability(path: str) -> Mapping[str, Any] | None:
    ortools_only = {
        "situation.objectiveSeat",
        "situation.constructionBuyerRelation",
        "situation.purchaseSuits",
        "situation.requiredCards",
        "situation.forbiddenCards",
        "situation.finalSuitCountRanges",
        "resources.candidatesPerSeed",
        "resources.solverTimeoutSeconds",
        "resources.maximumConflicts",
        "resources.auctionConstructionMode",
        "resources.playConstructionMode",
        "resources.tacticalConstruction",
        "resources.tacticalDealBalance",
        "resources.strategicAuction",
        "resources.strategicPlay",
    }
    if path in ortools_only:
        return {
            "path": "resources.generationLane",
            "operator": "equals",
            "value": "ortools",
        }
    for profile_path in (
        "resources.constructionPreScreen",
        "resources.tacticalViabilityPreScreen",
        "resources.tacticalConstruction",
        "resources.tacticalDealBalance",
        "resources.strategicAuction",
        "resources.strategicPlay",
    ):
        if path.startswith(profile_path + "."):
            return {
                "path": profile_path,
                "operator": "present",
            }
    return None


def _complete_research_field_catalog(
    fields: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    for path, raw_definition in fields.items():
        definition = copy.deepcopy(dict(raw_definition))
        definition.setdefault(
            "valueType",
            _research_field_value_type(str(definition["control"])),
        )
        mode = str(definition["mode"])
        if path in _RESEARCH_PROFILE_ACTIVATION_DEFAULTS:
            definition["default"] = copy.deepcopy(
                _RESEARCH_PROFILE_ACTIVATION_DEFAULTS[path]
            )
            definition["omitWhenDisabled"] = True
        elif path in _RESEARCH_RECIPE_FIELD_DEFAULTS:
            definition["default"] = copy.deepcopy(
                _RESEARCH_RECIPE_FIELD_DEFAULTS[path]
            )
        elif "default" in definition:
            definition["default"] = copy.deepcopy(definition["default"])
        elif path in _RESEARCH_RECIPE_REQUIRED_FIELDS:
            definition["required"] = True
            if "[]" in path:
                definition["requiredWhenParentPresent"] = True
        elif "value" in definition:
            definition["default"] = copy.deepcopy(definition["value"])
        elif mode == "invariant" and len(definition.get("values") or ()) == 1:
            definition["default"] = copy.deepcopy(definition["values"][0])
        elif mode in {"frozen", "plugin_identity"}:
            definition["required"] = True
        else:
            raise RuntimeError(
                f"research operator field lacks a governed default or required marker: {path}"
            )
        definition["helpKey"] = _research_field_help_key(path)
        applicability = _research_field_applicability(path)
        if applicability is not None:
            definition["applicableWhen"] = copy.deepcopy(dict(applicability))
            definition["nonApplicableBehavior"] = "omitted"
        if path == "titleAr":
            definition["defaultKey"] = "research_preset_small_gain"
        if path in _RESEARCH_RECIPE_FIELD_EXAMPLES:
            example = _RESEARCH_RECIPE_FIELD_EXAMPLES[path]
        elif definition.get("default") not in (None, [], {}):
            example = definition["default"]
        elif definition.get("values"):
            example = definition["values"][0]
        elif definition["control"] == "boolean":
            example = True
        elif definition["valueType"] in {"integer", "number"}:
            example = definition.get("minimum", 1)
        elif definition["valueType"] == "array":
            example = []
        elif definition["valueType"].startswith("object"):
            example = {}
        else:
            example = "configured-value"
        definition["example"] = copy.deepcopy(example)
        completed[str(path)] = definition
    return completed


def _complete_research_relationship_catalog(
    relationships: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    completed: list[dict[str, Any]] = []
    for raw_relationship in relationships:
        relationship = copy.deepcopy(dict(raw_relationship))
        code = str(relationship["code"])
        try:
            label_key, help_key = _RESEARCH_RECIPE_RELATION_COPY[code]
        except KeyError as exc:
            raise RuntimeError(
                f"research relationship lacks governed presentation metadata: {code}"
            ) from exc
        relationship["labelKey"] = label_key
        relationship["helpKey"] = help_key
        relationship["exampleKey"] = help_key
        assertion = next(
            (
                {key: copy.deepcopy(relationship[key])}
                for key in ("requires", "forbids", "requiresRelation", "constraint")
                if key in relationship
            ),
            {},
        )
        relationship["example"] = {
            "valid": {
                "condition": copy.deepcopy(relationship.get("if")),
                "satisfies": copy.deepcopy(assertion),
            },
            "invalid": {
                "condition": copy.deepcopy(relationship.get("if")),
                "violates": sorted(assertion),
            },
        }
        completed.append(relationship)
    if set(_RESEARCH_RECIPE_RELATION_COPY) != {
        str(item["code"]) for item in completed
    }:
        raise RuntimeError("research relationship presentation inventory has drifted")
    return completed


def _research_recipe_presets_v1() -> list[dict[str, Any]]:
    base = {
        "qualification": {
            "minimumSurroundingQuality": 75.0,
            "maximumRegret": 0.0,
        },
        "resources": {"deepVerificationPolicy": "required_gates"},
    }
    definitions = (
        (
            "project_400",
            "research_preset_400",
            {
                "situation": {
                    "dealer": "west",
                    "buyer": "south",
                    "contracts": ["sun"],
                    "projects": [{
                        "seat": "south",
                        "contract": "sun",
                        "kind": "four_hundred",
                        "value": 400,
                        "highRank": "ace",
                        "mode": "hard",
                    }],
                },
                "tactics": [{
                    "kind": "kabout_creation",
                    "mode": "qualification_gate",
                    "minimumScore": 70.0,
                    "weight": 1.0,
                    "perspectives": [],
                }],
            },
        ),
        (
            "small_card_gain",
            "research_preset_small_gain",
            {
                "tactics": [{
                    "kind": "small_card_big_gain",
                    "mode": "qualification_gate",
                    "minimumScore": 70.0,
                    "weight": 1.0,
                    "perspectives": [],
                }],
                "resources": {
                    "tacticalDealBalance": {
                        "schemaVersion": 1,
                        "id": "balanced_tactical_deal_v1",
                        "revision": 1,
                        "enabled": True,
                        "maximumTeamHighCardDifference": 2,
                        "minimumHighCardsPerSeat": 2,
                        "maximumHighCardsPerSeat": 4,
                        "minimumTargetLowCards": 1,
                    }
                },
            },
        ),
        (
            "only_winning_move",
            "research_preset_only_move",
            {
                "situation": {"contracts": ["sun"]},
                "tactics": [{
                    "kind": "only_winning_move",
                    "mode": "qualification_gate",
                    "minimumScore": 70.0,
                    "weight": 1.0,
                    "perspectives": [],
                }],
                "resources": {
                    "tacticalDealBalance": {
                        "schemaVersion": 1,
                        "id": "balanced_tactical_deal_v1",
                        "revision": 1,
                        "enabled": True,
                        "maximumTeamHighCardDifference": 2,
                        "minimumHighCardsPerSeat": 2,
                        "maximumHighCardsPerSeat": 4,
                        "minimumTargetLowCards": 1,
                    }
                },
            },
        ),
        (
            "large_card_sacrifice",
            "research_preset_sacrifice",
            {
                "tactics": [{
                    "kind": "large_card_sacrifice",
                    "mode": "qualification_gate",
                    "minimumScore": 70.0,
                    "weight": 1.0,
                    "perspectives": [],
                }],
            },
        ),
        (
            "critical_score",
            "research_preset_critical",
            {
                "situation": {"matchScore": [140, 145]},
                "tactics": [{
                    "kind": "contract_rescue",
                    "mode": "qualification_gate",
                    "minimumScore": 70.0,
                    "weight": 1.0,
                    "perspectives": [],
                }],
            },
        ),
    )
    presets = []
    for code, label_key, patch in definitions:
        merged = copy.deepcopy(base)
        for key, value in patch.items():
            if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
                merged[key] = {**dict(merged[key]), **copy.deepcopy(dict(value))}
            else:
                merged[key] = copy.deepcopy(value)
        presets.append({
            "code": code,
            "labelKey": label_key,
            "helpKey": "research_review_help",
            "patch": merged,
        })
    return presets


def _research_resource_presets_v1() -> list[dict[str, Any]]:
    values = (
        ("quick_exploration", "research_budget_preset_quick", 20, 1, 10, 2.0, 50_000),
        ("standard_research", "research_budget_preset_standard", 100, 2, 10, 15.0, 250_000),
        ("thorough_campaign", "research_budget_preset_thorough", 1_000, 4, 25, 60.0, 1_000_000),
    )
    return [
        {
            "code": code,
            "labelKey": label_key,
            "helpKey": "research_step_budget_help",
            "patch": {
                "resources": {
                    "seedCount": seeds,
                    "candidatesPerSeed": candidates,
                    "seedsPerShard": shard_size,
                    "solverTimeoutSeconds": timeout,
                    "maximumConflicts": conflicts,
                }
            },
        }
        for code, label_key, seeds, candidates, shard_size, timeout, conflicts in values
    ]


def _card_constraints(
    value: Mapping[str, Any] | None, *, globally_unique: bool
) -> dict[str, list[str]]:
    source = dict(value or {})
    if set(source) - _SEATS:
        raise ValueError("research card constraints contain an invalid seat")
    result: dict[str, list[str]] = {}
    claimed: set[str] = set()
    for seat, values in sorted(source.items()):
        if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple, set)):
            raise ValueError("research card constraints must be card lists")
        cards = sorted({str(item).strip().upper() for item in values or ()})
        if any(not _CARD_CODE.fullmatch(card) for card in cards):
            raise ValueError("research card constraint contains an invalid card")
        if globally_unique and claimed.intersection(cards):
            raise ValueError("one required card cannot belong to multiple seats")
        if len(cards) > 8:
            raise ValueError("one seat cannot contain more than eight cards")
        if cards:
            result[str(seat)] = cards
            claimed.update(cards)
    return result


def _suit_count_ranges(
    value: Mapping[str, Any] | None,
) -> dict[str, dict[str, dict[str, int]]]:
    source = dict(value or {})
    if set(source) - _SEATS:
        raise ValueError("research suit ranges contain an invalid seat")
    result: dict[str, dict[str, dict[str, int]]] = {}
    for seat, raw_ranges in sorted(source.items()):
        ranges = dict(raw_ranges or {})
        if set(ranges) - _SUITS:
            raise ValueError("research suit ranges contain an invalid suit")
        normalized: dict[str, dict[str, int]] = {}
        minimum_total = 0
        maximum_total = 0
        for suit in sorted(_SUITS):
            raw_bounds = ranges.get(suit, {"minimum": 0, "maximum": 8})
            if not isinstance(raw_bounds, Mapping):
                raise ValueError("research suit range must be an object")
            if set(raw_bounds) - {"minimum", "maximum"}:
                raise ValueError("research suit range contains an unknown bound")
            lower = int(raw_bounds.get("minimum", 0))
            upper = int(raw_bounds.get("maximum", 8))
            if not 0 <= lower <= upper <= 8:
                raise ValueError("research suit range must stay between zero and eight")
            minimum_total += lower
            maximum_total += upper
            if suit in ranges:
                normalized[suit] = {"minimum": lower, "maximum": upper}
        if minimum_total > 8 or maximum_total < 8:
            raise ValueError("research suit ranges cannot form an eight-card hand")
        if normalized:
            result[str(seat)] = normalized
    return result


def research_recipe_configuration_v1() -> dict[str, Any]:
    """Machine-readable authority for the current Research recipe controls.

    The catalog distinguishes operator choices, frozen profile identities,
    supported invariants, and behavior that requires a governed plugin.  It is
    intentionally data-only so the API, Foundry, documentation, and contract
    tests can consume the same field and relationship inventory.
    """

    fields: dict[str, dict[str, Any]] = {
        "schemaVersion": {"control": "integer", "value": 3, "mode": "invariant"},
        "code": {"control": "identity", "pattern": "^RCP-[A-Z0-9]{6,20}$", "default": None, "mode": "derived_identity"},
        "titleAr": {"control": "text", "minimumLength": 3, "maximumLength": 160, "mode": "operator"},
        "sha256": {"control": "checksum", "pattern": "^[0-9a-f]{64}$", "default": None, "mode": "derived_identity"},
        "configuration.schemaVersion": {"control": "integer", "value": 1, "mode": "invariant"},
        "configuration.bundleRevisionId": {"control": "published_revision", "mode": "operator"},
        "output.kind": {"control": "enum", "values": ["laffa", "laffa_collection"], "mode": "operator"},
        "output.resultCount": {"control": "integer", "minimum": 1, "maximum": 1000, "mode": "operator"},
        "profiles.*": {"control": "published_revision", "mode": "frozen"},
        "situation.dealer": {"control": "enum", "values": sorted(_SEATS), "mode": "operator"},
        "situation.buyer": {"control": "optional_enum", "values": sorted(_SEATS), "mode": "operator"},
        "situation.objectiveSeat": {"control": "optional_enum", "values": sorted(_SEATS), "mode": "operator"},
        "situation.constructionBuyerRelation": {"control": "enum", "values": sorted(_CONSTRUCTION_BUYER_RELATIONS), "mode": "operator"},
        "situation.purchaseRound": {"control": "optional_enum", "valueType": "integer_or_null", "values": [1, 2], "mode": "operator"},
        "situation.purchaseSuits": {"control": "multi_enum", "values": sorted(_SUITS), "mode": "operator"},
        "situation.trumpSuit": {"control": "optional_enum", "values": sorted(_SUITS), "mode": "operator"},
        "situation.contracts": {"control": "multi_enum", "values": sorted(_CONTRACTS), "mode": "operator"},
        "situation.matchScore": {"control": "integer_pair", "itemValueType": "integer", "exactLength": 2, "minimum": 0, "maximum": 10_000, "mode": "operator"},
        "situation.requiredCards": {"control": "seat_card_picker", "cardListValueType": "array", "cardValueType": "string", "maximumPerSeat": 8, "mode": "operator"},
        "situation.forbiddenCards": {"control": "seat_card_picker", "cardListValueType": "array", "cardValueType": "string", "maximumPerSeat": 8, "mode": "operator"},
        "situation.finalSuitCountRanges": {"control": "seat_suit_ranges", "boundValueType": "integer", "minimum": 0, "maximum": 8, "mode": "operator"},
        "situation.projects": {"control": "project_constraint", "values": sorted(_PROJECT_KINDS), "maximumHard": 1, "modes": ["hard"], "mode": "operator"},
        "situation.projects[].seat": {"control": "enum", "values": sorted(_SEATS), "mode": "operator"},
        "situation.projects[].contract": {"control": "enum", "values": sorted(_CONTRACTS), "mode": "operator"},
        "situation.projects[].kind": {"control": "enum", "values": sorted(_PROJECT_KINDS), "mode": "operator"},
        "situation.projects[].value": {"control": "integer", "minimum": 0, "maximum": 400, "mode": "operator"},
        "situation.projects[].highRank": {"control": "enum", "values": ["seven", "eight", "nine", "ten", "jack", "queen", "king", "ace"], "mode": "operator"},
        "situation.projects[].mode": {"control": "enum", "values": ["hard"], "mode": "operator"},
        "situation.escalations": {
            "control": "multi_enum",
            "values": sorted(_ESCALATION_LEVELS),
            "maximum": 1,
            "emptyValueAlias": {
                "value": "normal",
                "canonical": [],
                "labelKey": "research_escalation_normal",
            },
            "mode": "operator",
        },
        "tactics": {
            "control": "repeatable_tactic",
            "values": sorted(_TACTICS),
            "modes": sorted(_V3_TACTIC_MODES),
            "modeSemantics": {
                "preference": {
                    "effect": "selection_score",
                    "requiresEvidenceAtOrAboveMinimumScore": True,
                    "filtersEvidenceByPerspective": True,
                    "qualificationGate": False,
                },
                "qualification_gate": {
                    "effect": "post_replay_qualification",
                    "requiresEvidenceAtOrAboveMinimumScore": True,
                    "filtersEvidenceByPerspective": True,
                    "qualificationGate": True,
                },
            },
            "maximum": len(_TACTICS),
            "mode": "operator",
        },
        "tactics[].kind": {"control": "enum", "values": sorted(_TACTICS), "mode": "operator"},
        "tactics[].mode": {
            "control": "enum",
            "values": sorted(_V3_TACTIC_MODES),
            "modeSemanticsSource": "tactics.modeSemantics",
            "mode": "operator",
        },
        "tactics[].minimumScore": {"control": "number", "minimum": 0, "maximum": 100, "mode": "operator"},
        "tactics[].weight": {"control": "number", "exclusiveMinimum": 0, "maximum": 100, "mode": "operator"},
        "tactics[].perspectives": {"control": "multi_enum", "values": sorted(_SEATS), "mode": "operator"},
        "qualification.minimumTruth": {"control": "enum", "values": ["exact_optimal", "robust_best_known", "debatable", "provisional"], "mode": "operator"},
        "qualification.minimumConfidence": {"control": "number", "minimum": 0, "maximum": 1, "mode": "operator"},
        "qualification.minimumSurroundingQuality": {"control": "number", "minimum": 0, "maximum": 100, "default": 70, "mode": "operator"},
        "qualification.maximumRegret": {"control": "number", "minimum": 0, "mode": "operator"},
        "qualification.requireObserverSafety": {"control": "boolean", "value": True, "mode": "invariant"},
        "qualification.requireAllDecisionAudit": {"control": "boolean", "value": True, "mode": "invariant"},
        "resources.generationLane": {"control": "enum", "values": ["ortools", "organic_runtime"], "mode": "operator"},
        "resources.seedStart": {"control": "integer", "minimum": 0, "mode": "operator"},
        "resources.seedCount": {"control": "integer", "minimum": 1, "maximum": 1_000_000_000, "default": 10, "mode": "operator"},
        "resources.seedsPerShard": {"control": "integer", "minimum": 1, "maximum": 1_000_000, "default": 10, "mode": "operator"},
        "resources.candidatesPerSeed": {"control": "integer", "minimum": 1, "maximum": 100, "mode": "operator"},
        "resources.solverTimeoutSeconds": {"control": "number", "minimum": 0.1, "maximum": 900, "mode": "operator"},
        "resources.maximumConflicts": {"control": "integer", "minimum": 1, "maximum": 10_000_000, "mode": "operator"},
        "resources.analysisMode": {"control": "enum", "values": ["screening", "cascaded", "deep"], "mode": "operator"},
        "resources.deepVerificationPolicy": {
            "control": "enum",
            "values": sorted(_DEEP_VERIFICATION_POLICIES),
            "policySemantics": {
                "required_gates": (
                    "bounded verification for every configured tactic that "
                    "affects qualification or ranking"
                ),
                "all_candidates": (
                    "bounded verification for every candidate, including "
                    "recipes without configured tactics"
                ),
                "off": (
                    "no bounded tactical verification; unavailable when a "
                    "qualification gate requires tactical proof"
                ),
            },
            "mode": "operator",
        },
        "resources.tacticalConstruction": {"control": "profile", "schema": "TacticalPlayConstructionProfileV1", "mode": "operator"},
        "resources.tacticalConstruction.schemaVersion": {"control": "integer", "value": 1, "mode": "invariant"},
        "resources.tacticalConstruction.id": {"control": "enum", "values": ["tactical_play_precursor_v1"], "value": "tactical_play_precursor_v1", "mode": "invariant"},
        "resources.tacticalConstruction.revision": {"control": "integer", "value": 1, "mode": "invariant"},
        "resources.tacticalConstruction.sha256": {"control": "checksum", "pattern": "^[0-9a-f]{64}$", "default": None, "mode": "derived_identity"},
        "resources.tacticalConstruction.enabled": {"control": "boolean", "mode": "operator"},
        "resources.tacticalConstruction.maximumPayoffTricks": {"control": "integer", "minimum": 1, "maximum": 4, "mode": "operator"},
        "resources.tacticalConstruction.minimumLaterRawPoints": {"control": "integer", "minimum": 1, "maximum": 40, "mode": "operator"},
        "resources.tacticalConstruction.requirePrecursorForGatedTactics": {"control": "boolean", "mode": "operator"},
        "resources.constructionPreScreen": {"control": "profile", "schema": "ConstructionPreScreenProfileV1", "mode": "operator"},
        "resources.constructionPreScreen.schemaVersion": {"control": "integer", "value": 1, "mode": "invariant"},
        "resources.constructionPreScreen.id": {"control": "enum", "values": ["actor_safe_pre_screen_v1"], "value": "actor_safe_pre_screen_v1", "mode": "invariant"},
        "resources.constructionPreScreen.revision": {"control": "enum", "values": ["1"], "value": "1", "mode": "invariant"},
        "resources.constructionPreScreen.sha256": {"control": "checksum", "pattern": "^[0-9a-f]{64}$", "default": None, "mode": "derived_identity"},
        "resources.constructionPreScreen.enabled": {"control": "boolean", "default": False, "mode": "operator"},
        "resources.constructionPreScreen.maximumPlayDecisions": {"control": "integer", "minimum": 1, "maximum": 16, "mode": "operator"},
        "resources.constructionPreScreen.maximumBidRegret": {"control": "number", "minimum": 0, "mode": "operator"},
        "resources.constructionPreScreen.maximumPlayRegret": {"control": "number", "minimum": 0, "mode": "operator"},
        "resources.constructionPreScreen.maximumProbabilityDrop": {"control": "number", "minimum": 0, "maximum": 1, "mode": "operator"},
        "resources.constructionPreScreen.minimumConfidence": {"control": "number", "minimum": 0, "maximum": 1, "default": 0, "mode": "operator"},
        "resources.constructionPreScreen.rejectCatastrophicCrossing": {"control": "boolean", "mode": "operator"},
        "resources.tacticalViabilityPreScreen": {"control": "profile", "schema": "TacticalViabilityPreScreenProfileV1", "mode": "operator"},
        "resources.tacticalViabilityPreScreen.schemaVersion": {"control": "integer", "value": 1, "mode": "invariant"},
        "resources.tacticalViabilityPreScreen.id": {"control": "enum", "values": ["exact_tactical_viability_screen_v1"], "value": "exact_tactical_viability_screen_v1", "mode": "invariant"},
        "resources.tacticalViabilityPreScreen.revision": {"control": "enum", "values": ["1"], "value": "1", "mode": "invariant"},
        "resources.tacticalViabilityPreScreen.sha256": {"control": "checksum", "pattern": "^[0-9a-f]{64}$", "default": None, "mode": "derived_identity"},
        "resources.tacticalViabilityPreScreen.enabled": {"control": "boolean", "default": False, "mode": "operator"},
        "resources.tacticalViabilityPreScreen.maximumDecisions": {"control": "integer", "minimum": 1, "maximum": 16, "mode": "operator"},
        "resources.tacticalViabilityPreScreen.requiredTacticsOnly": {"control": "boolean", "mode": "operator"},
        "resources.tacticalViabilityPreScreen.rejectWhenExactAbsent": {"control": "boolean", "mode": "operator"},
        "resources.tacticalDealBalance": {"control": "profile", "schema": "TacticalDealBalanceProfileV1", "mode": "operator"},
        "resources.tacticalDealBalance.schemaVersion": {"control": "integer", "value": 1, "mode": "invariant"},
        "resources.tacticalDealBalance.id": {"control": "enum", "values": ["balanced_tactical_deal_v1"], "value": "balanced_tactical_deal_v1", "mode": "invariant"},
        "resources.tacticalDealBalance.revision": {"control": "integer", "value": 1, "mode": "invariant"},
        "resources.tacticalDealBalance.sha256": {"control": "checksum", "pattern": "^[0-9a-f]{64}$", "default": None, "mode": "derived_identity"},
        "resources.tacticalDealBalance.enabled": {"control": "boolean", "mode": "operator"},
        "resources.tacticalDealBalance.maximumTeamHighCardDifference": {"control": "integer", "minimum": 0, "maximum": 12, "mode": "operator"},
        "resources.tacticalDealBalance.minimumHighCardsPerSeat": {"control": "integer", "minimum": 0, "maximum": 3, "mode": "operator"},
        "resources.tacticalDealBalance.maximumHighCardsPerSeat": {"control": "integer", "minimum": 3, "maximum": 8, "mode": "operator"},
        "resources.tacticalDealBalance.minimumTargetLowCards": {"control": "integer", "minimum": 0, "maximum": 3, "mode": "operator"},
        "resources.auctionConstructionMode": {"control": "enum", "values": sorted(_AUCTION_CONSTRUCTION_MODES), "mode": "operator"},
        "resources.strategicAuction": {"control": "profile", "schema": "ActorSafeCpSatAuctionProfileV1", "mode": "operator"},
        "resources.strategicAuction.schemaVersion": {"control": "integer", "value": 1, "mode": "invariant"},
        "resources.strategicAuction.solverBackend": {"control": "enum", "values": ["ortools_cp_sat"], "mode": "invariant"},
        "resources.strategicAuction.sha256": {"control": "checksum", "pattern": "^[0-9a-f]{64}$", "default": None, "mode": "derived_identity"},
        "resources.strategicAuction.requestedSamples": {
            "control": "integer",
            "minimum": 2,
            "maximum": 1_000,
            "value": RESEARCH_STRATEGIC_AUCTION_DEEP_BID_SAMPLES_V3,
            "mode": "derived",
            "derivedFrom": "configuration.challenge.confirmation.bid_samples",
            "semantics": "compatibility_mirror",
        },
        "resources.strategicAuction.integerScale": {"control": "integer", "minimum": 100, "maximum": 100_000, "mode": "operator"},
        "resources.strategicAuction.maximumConflicts": {"control": "integer", "minimum": 1, "maximum": 10_000_000, "mode": "operator"},
        "resources.strategicAuction.maximumSelectionSeconds": {"control": "number", "minimum": 0.01, "maximum": 30, "mode": "operator"},
        "resources.strategicAuction.objective": {"control": "enum", "values": ["expected_round_margin"], "mode": "invariant"},
        "resources.strategicAuction.tieBreak": {
            "control": "enum",
            "values": [RESEARCH_STRATEGIC_AUCTION_TIE_BREAK_V1],
            "mode": "invariant",
        },
        "resources.strategicAuction.requireOptimalForQualification": {"control": "boolean", "mode": "operator"},
        "resources.playConstructionMode": {"control": "enum", "values": sorted(_PLAY_CONSTRUCTION_MODES), "mode": "operator"},
        "resources.strategicPlay": {"control": "profile", "schema": "ActorSafeCpSatPlayProfileV1", "mode": "operator"},
        "resources.strategicPlay.schemaVersion": {"control": "integer", "value": 1, "mode": "invariant"},
        "resources.strategicPlay.solverBackend": {"control": "enum", "values": ["batched_cp_sat"], "mode": "invariant"},
        "resources.strategicPlay.sha256": {"control": "checksum", "pattern": "^[0-9a-f]{64}$", "default": None, "mode": "derived_identity"},
        "resources.strategicPlay.minimumPlayedCards": {"control": "integer", "minimum": 20, "maximum": 28, "mode": "operator"},
        "resources.strategicPlay.requestedSamples": {"control": "integer", "minimum": 1, "maximum": 5_000, "mode": "operator"},
        "resources.strategicPlay.exactAllocationLimit": {"control": "integer", "minimum": 1, "maximum": 5_000, "mode": "operator"},
        "resources.strategicPlay.maximumSamplingAttempts": {"control": "integer", "minimum": 1, "maximum": 1_000_000, "mode": "operator"},
        "resources.strategicPlay.maximumWorlds": {"control": "integer", "minimum": 1, "maximum": 5_000, "mode": "operator"},
        "resources.strategicPlay.maximumContinuationSteps": {"control": "integer", "minimum": 1, "maximum": 2_000_000, "mode": "operator"},
        "resources.strategicPlay.maximumPlies": {"control": "integer", "minimum": 4, "maximum": 12, "mode": "operator", "relationship": "minimum_is_32_minus_minimumPlayedCards"},
        "resources.strategicPlay.maximumSelectionSeconds": {"control": "number", "minimum": 0.01, "maximum": 30, "mode": "operator"},
        "resources.strategicPlay.requireExactForQualification": {"control": "boolean", "mode": "operator"},
        "resources.strategicPlay.continuationPolicy": {"control": "enum", "values": ["observer_safe_heuristic_v1"], "mode": "invariant"},
        "resources.strategicPlay.policyProfileId": {"control": "enum", "values": ["maximin_margin"], "mode": "invariant"},
        "resources.strategicPlay.policyRevision": {"control": "integer", "value": 1, "mode": "invariant"},
        "resources.strategicPlay.outcomeBoundary": {"control": "integer", "minimum": -10_000, "maximum": 10_000, "mode": "operator"},
        "resources.strategicPlay.utility": {"control": "enum", "values": ["actor_team_awarded_point_margin"], "mode": "invariant"},
        "resources.strategicPlay.tieBreak": {"control": "enum", "values": ["maximin_then_total_then_card_id"], "mode": "invariant"},
        "diversity": {"control": "profile", "schema": "QualityDiversityProfileV1", "mode": "operator"},
        "diversity.schemaVersion": {"control": "integer", "value": 1, "mode": "invariant"},
        "diversity.sha256": {"control": "checksum", "pattern": "^[0-9a-f]{64}$", "default": None, "mode": "derived_identity"},
        "diversity.cellDimensions": {"control": "multi_enum", "values": ["contract", "purchase_round", "trump_suit", "project", "escalation", "tactical_detector", "acting_seat", "score_pressure", "perspective_breadth", "source", "candidate_scope"], "mode": "operator"},
        "diversity.topKPerCell": {"control": "integer", "minimum": 1, "maximum": 1_000, "mode": "operator_safety_bound"},
        "diversity.rejectedAuditModulus": {"control": "integer", "minimum": 1, "maximum": 1_000_000, "mode": "operator_safety_bound"},
        "renderPolicy.mode": {
            "control": "enum",
            "values": ["on_demand"],
            "value": "on_demand",
            "mode": "invariant",
        },
        "status": {"control": "lifecycle", "values": ["draft", "published", "retired"], "mode": "operator"},
    }
    relationships = [
        {"code": "explicit_buyer_excludes_relation", "if": "situation.buyer", "requires": {"situation.constructionBuyerRelation": "automatic"}},
        {"code": "construction_relation_requires_objective_seat", "if": {"situation.constructionBuyerRelation": ["objective", "partner"]}, "requires": {"situation.objectiveSeat": "present"}},
        {"code": "trump_requires_hokum", "if": "situation.trumpSuit", "requires": {"situation.contracts": ["hokum"]}},
        {"code": "four_hundred_requires_sun", "if": {"situation.projects[].kind": "four_hundred"}, "requires": {"situation.projects[].contract": "sun"}},
        {"code": "high_escalation_requires_hokum", "if": {"situation.escalations": ["triple", "four", "gahwa"]}, "requires": {"situation.contracts": ["hokum"]}},
        {"code": "strategic_auction_requires_ortools", "if": {"resources.auctionConstructionMode": "cp_sat_actor_safe_decision_v1"}, "requires": {"resources.generationLane": "ortools", "resources.strategicAuction": "present"}},
        {"code": "strategic_auction_profile_requires_mode", "if": "resources.strategicAuction", "requires": {"resources.auctionConstructionMode": "cp_sat_actor_safe_decision_v1"}},
        {"code": "strategic_play_requires_ortools", "if": {"resources.playConstructionMode": "cp_sat_actor_safe_late_game_v1"}, "requires": {"resources.generationLane": "ortools", "resources.strategicPlay": "present"}},
        {"code": "strategic_play_profile_requires_mode", "if": "resources.strategicPlay", "requires": {"resources.playConstructionMode": "cp_sat_actor_safe_late_game_v1"}},
        {"code": "actor_safe_auction_requires_ortools", "if": {"resources.auctionConstructionMode": ["actor_safe_agent_verified", "cp_sat_actor_safe_decision_v1"]}, "requires": {"resources.generationLane": "ortools"}},
        {"code": "actor_safe_auction_requires_actor_safe_play", "if": {"resources.auctionConstructionMode": ["actor_safe_agent_verified", "cp_sat_actor_safe_decision_v1"]}, "requires": {"resources.playConstructionMode": ["actor_safe_agent_verified", "cp_sat_actor_safe_late_game_v1"]}},
        {"code": "organic_forbids_ortools_solver_controls", "if": {"resources.generationLane": "organic_runtime"}, "forbids": {"resources.candidatesPerSeed": "present", "resources.solverTimeoutSeconds": "present", "resources.maximumConflicts": "present"}},
        {"code": "organic_forbids_ortools_structural_constraints", "if": {"resources.generationLane": "organic_runtime"}, "forbids": {"situation.objectiveSeat": "present", "situation.constructionBuyerRelation": ["objective", "partner"], "situation.purchaseSuits": "present", "situation.requiredCards": "present", "situation.forbiddenCards": "present", "situation.finalSuitCountRanges": "present"}},
        {"code": "actor_safe_auction_forbids_escalation", "if": {"resources.auctionConstructionMode": ["actor_safe_agent_verified", "cp_sat_actor_safe_decision_v1"]}, "forbids": {"situation.escalations": "present"}},
        {"code": "strategic_play_samples_fit_worlds", "if": {"resources.playConstructionMode": "cp_sat_actor_safe_late_game_v1"}, "requiresRelation": {"left": "resources.strategicPlay.requestedSamples", "operator": "<=", "right": "resources.strategicPlay.maximumWorlds"}},
        {"code": "strategic_play_exact_limit_fits_worlds", "if": {"resources.playConstructionMode": "cp_sat_actor_safe_late_game_v1"}, "requiresRelation": {"left": "resources.strategicPlay.exactAllocationLimit", "operator": "<=", "right": "resources.strategicPlay.maximumWorlds"}},
        {"code": "strategic_play_attempts_cover_samples", "if": {"resources.playConstructionMode": "cp_sat_actor_safe_late_game_v1"}, "requiresRelation": {"left": "resources.strategicPlay.maximumSamplingAttempts", "operator": ">=", "right": "resources.strategicPlay.requestedSamples"}},
        {"code": "strategic_play_plies_cover_tail", "if": {"resources.playConstructionMode": "cp_sat_actor_safe_late_game_v1"}, "requiresRelation": {"left": "resources.strategicPlay.maximumPlies", "operator": ">=", "expression": "32 - resources.strategicPlay.minimumPlayedCards"}},
        {"code": "hard_tactics_require_verification", "if": {"tactics[].mode": ["qualification_gate"]}, "forbids": {"resources.deepVerificationPolicy": "off"}},
        {"code": "tactical_precursor_is_construction_only", "if": {"resources.tacticalConstruction.enabled": True}, "requires": {"resources.generationLane": "ortools", "resources.playConstructionMode": "cp_sat_feasibility_line"}},
        {"code": "tactical_precursor_requires_small_card_gate", "if": {"resources.tacticalConstruction.enabled": True}, "constraint": {"kind": "tactic_row_matches", "tacticKind": "small_card_big_gain", "modes": ["qualification_gate"]}},
        {"code": "tactical_precursor_supports_only_small_card_gate", "if": {"resources.tacticalConstruction.enabled": True, "resources.tacticalConstruction.requirePrecursorForGatedTactics": True}, "constraint": {"kind": "gated_tactic_kinds_subset", "allowed": ["small_card_big_gain"]}},
        {"code": "tactical_viability_screen_requires_proof", "if": {"resources.tacticalViabilityPreScreen.enabled": True}, "forbids": {"resources.deepVerificationPolicy": "off"}},
        {"code": "tactical_viability_required_only_requires_gate", "if": {"resources.tacticalViabilityPreScreen.enabled": True, "resources.tacticalViabilityPreScreen.requiredTacticsOnly": True}, "requires": {"tactics[].mode": ["qualification_gate"]}},
        {"code": "tactical_deal_balance_requires_ortools", "if": {"resources.tacticalDealBalance.enabled": True}, "requires": {"resources.generationLane": "ortools"}},
        {"code": "tactical_deal_balance_requires_supported_tactic", "if": {"resources.tacticalDealBalance.enabled": True}, "constraint": {"kind": "tactic_row_matches_any", "tacticKinds": sorted(_TACTICAL_DEAL_BALANCE_GATED_TACTICS), "modes": ["qualification_gate"]}},
        {"code": "tactical_deal_balance_bounds_ordered", "if": "resources.tacticalDealBalance", "requiresRelation": {"left": "resources.tacticalDealBalance.minimumHighCardsPerSeat", "operator": "<=", "right": "resources.tacticalDealBalance.maximumHighCardsPerSeat"}},
        {"code": "project_contract_matches_requested_contract", "if": {"situation.projects": "present"}, "requiresRelation": {"left": "situation.projects[].contract", "operator": "member_of", "right": "situation.contracts"}},
        {"code": "at_most_one_hard_project", "if": "situation.projects", "constraint": {"kind": "maximum_items_by_value", "path": "situation.projects[].mode", "value": "hard", "maximum": 1}},
        {"code": "required_cards_unique_across_seats", "if": "situation.requiredCards", "constraint": {"kind": "unique_cards", "maximumPerSeat": 8}},
        {"code": "required_and_forbidden_cards_are_disjoint", "if": "situation.requiredCards", "constraint": {"kind": "disjoint_per_seat", "other": "situation.forbiddenCards"}},
        {"code": "suit_ranges_form_eight_card_hand", "if": "situation.finalSuitCountRanges", "constraint": {"kind": "bounded_sum", "minimum": 0, "maximum": 8, "exactTotal": 8}},
        {"code": "required_cards_fit_suit_ranges", "if": {"situation.requiredCards": "present", "situation.finalSuitCountRanges": "present"}, "constraint": {"kind": "required_cards_fit_suit_ranges"}},
        {"code": "tactical_kinds_are_unique", "if": "tactics", "constraint": {"kind": "unique", "path": "tactics[].kind"}},
        {"code": "single_laffa_requires_one_result", "if": {"output.kind": "laffa"}, "requires": {"output.resultCount": 1}},
        {"code": "tactics_require_exactly_one_contract", "if": "tactics", "constraint": {"kind": "exact_items", "path": "situation.contracts", "count": 1}},
    ]
    fields = _complete_research_field_catalog(fields)
    relationships = _complete_research_relationship_catalog(relationships)
    core = {
        "schemaVersion": 1,
        "recipeSchemaVersion": 3,
        "supportedRecipeSchemaVersions": [1, 2, 3],
        "legacyReadSchemaVersions": [1, 2],
        "fields": fields,
        "relationships": relationships,
        "recipePresets": _research_recipe_presets_v1(),
        "resourcePresets": _research_resource_presets_v1(),
        "unsupportedVariants": [
            {
                "code": "direct_contextualized_laffa",
                "field": "output.kind",
                "value": "contextualized_laffa",
                "requires": "governed_context_search_descendant",
            },
            {
                "code": "direct_sakka_monolith",
                "field": "output.kind",
                "value": "sakka",
                "requires": "verified_contextualized_laffa_composer",
            },
            {
                "code": "actor_safe_auction_escalation",
                "field": "situation.escalations",
                "requires": "governed_actor_safe_escalation_policy_plugin",
            },
            {
                "code": "soft_project_targeting",
                "field": "situation.projects[].mode",
                "requires": "governed_project_objective_plugin",
            },
            {
                "code": "multiple_hard_projects",
                "field": "situation.projects",
                "requires": "governed_multi_project_construction_plugin",
            },
            {
                "code": "tactical_precursor_other_than_small_card_gain",
                "field": "resources.tacticalConstruction",
                "requires": "tactical_construction_plugin_for_requested_detector",
            },
            {
                "code": "full_round_global_optimal_play",
                "field": "resources.strategicPlay.minimumPlayedCards",
                "requires": "governed_full_information_set_policy_plugin",
            },
            {
                "code": "alternate_strategic_auction_policy",
                "field": "resources.strategicAuction",
                "requires": "compatible_research_auction_selector_plugin",
            },
            {
                "code": "alternate_strategic_play_policy",
                "field": "resources.strategicPlay",
                "requires": "compatible_research_play_selector_plugin",
            },
            {
                "code": "placeholder_diversity_dimensions",
                "field": "diversity.cellDimensions",
                "values": [
                    "decision_kind",
                    "trick_number",
                    "buyer_relationship",
                    "difficulty_band",
                    "video_duration",
                    "surrounding_competence",
                ],
                "requires": "governed_dynamic_dimension_plugin",
            },
            {
                "code": "automatic_research_render",
                "field": "renderPolicy.mode",
                "requires": "explicit_operator_request",
            },
            {
                "code": "uncompiled_hard_tactical_mode",
                "field": "tactics[].mode",
                "value": "hard",
                "requires": "governed_cp_sat_hard_tactical_lowering_plugin",
            },
        ],
        "unsupportedBehaviorRequiresPlugin": True,
    }
    return {**core, "sha256": canonical_sha256(core)}


def _research_path_values(value: Mapping[str, Any], path: str) -> list[Any]:
    current: list[Any] = [value]
    for segment in str(path).split("."):
        repeated = segment.endswith("[]")
        key = segment[:-2] if repeated else segment
        following: list[Any] = []
        for parent in current:
            if not isinstance(parent, Mapping) or key not in parent:
                continue
            child = parent[key]
            if repeated:
                if isinstance(child, (str, bytes)) or not isinstance(
                    child, (list, tuple)
                ):
                    continue
                following.extend(child)
            else:
                following.append(child)
        current = following
    return current


def _research_present(values: Iterable[Any]) -> bool:
    return any(value not in (None, "", [], {}, ()) for value in values)


def _research_matches(values: list[Any], expected: Any) -> bool:
    if expected == "present":
        return _research_present(values)
    if not values:
        return False
    if isinstance(expected, list):
        for actual in values:
            if isinstance(actual, (list, tuple, set)):
                if list(actual) == expected:
                    return True
            elif actual in expected:
                return True
        return False
    return any(actual == expected for actual in values)


def _research_condition_matches(
    payload: Mapping[str, Any], condition: Any
) -> bool:
    if isinstance(condition, str):
        return _research_present(_research_path_values(payload, condition))
    if not isinstance(condition, Mapping):
        raise RuntimeError("research relationship condition is invalid")
    for path, expected in condition.items():
        values = _research_path_values(payload, str(path))
        if expected == "present":
            matches = _research_present(values)
        elif isinstance(expected, list):
            matches = any(
                bool(set(actual).intersection(expected))
                if isinstance(actual, (list, tuple, set))
                else actual in expected
                for actual in values
            )
        else:
            matches = any(actual == expected for actual in values)
        if not matches:
            return False
    return True


def _validate_research_relation_assertion(
    payload: Mapping[str, Any], relationship: Mapping[str, Any]
) -> bool:
    if "requires" in relationship:
        return all(
            _research_matches(_research_path_values(payload, str(path)), expected)
            for path, expected in dict(relationship["requires"]).items()
        )
    if "forbids" in relationship:
        return all(
            not _research_matches(
                _research_path_values(payload, str(path)), expected
            )
            for path, expected in dict(relationship["forbids"]).items()
        )
    if "requiresRelation" in relationship:
        relation = dict(relationship["requiresRelation"])
        left_values = _research_path_values(payload, str(relation["left"]))
        operator = str(relation["operator"])
        if "right" in relation:
            right_values = _research_path_values(payload, str(relation["right"]))
        else:
            expression = str(relation.get("expression") or "")
            match = re.fullmatch(r"32 - ([A-Za-z0-9_.\[\]]+)", expression)
            if match is None:
                raise RuntimeError("unsupported research relationship expression")
            source = _research_path_values(payload, match.group(1))
            right_values = [] if not source else [32 - int(source[0])]
        if not left_values or not right_values:
            return False
        if operator == "member_of":
            allowed = right_values[0]
            return isinstance(allowed, (list, tuple, set)) and all(
                value in allowed for value in left_values
            )
        left = left_values[0]
        right = right_values[0]
        if operator == "<=":
            return left <= right
        if operator == ">=":
            return left >= right
        raise RuntimeError(f"unsupported research relationship operator: {operator}")
    constraint = dict(relationship.get("constraint") or {})
    kind = str(constraint.get("kind") or "")
    if kind == "unique_cards":
        seat_cards = _research_path_values(payload, "situation.requiredCards")
        if not seat_cards or not isinstance(seat_cards[0], Mapping):
            return True
        maximum = int(constraint["maximumPerSeat"])
        flattened = [
            str(card)
            for cards in seat_cards[0].values()
            for card in cards
        ]
        return (
            len(flattened) == len(set(flattened))
            and all(len(cards) <= maximum for cards in seat_cards[0].values())
        )
    if kind == "disjoint_per_seat":
        required_values = _research_path_values(payload, "situation.requiredCards")
        other_values = _research_path_values(payload, str(constraint["other"]))
        if not required_values or not other_values:
            return True
        required = dict(required_values[0])
        other = dict(other_values[0])
        return all(
            not set(cards).intersection(other.get(seat, ()))
            for seat, cards in required.items()
        )
    if kind == "bounded_sum":
        ranges_values = _research_path_values(
            payload, "situation.finalSuitCountRanges"
        )
        if not ranges_values:
            return True
        exact_total = int(constraint["exactTotal"])
        for ranges in dict(ranges_values[0]).values():
            minimum_total = sum(
                int(dict(bounds).get("minimum", 0))
                for bounds in dict(ranges).values()
            )
            maximum_total = sum(
                int(dict(bounds).get("maximum", exact_total))
                for bounds in dict(ranges).values()
            ) + (4 - len(dict(ranges))) * exact_total
            if minimum_total > exact_total or maximum_total < exact_total:
                return False
        return True
    if kind == "unique":
        values = _research_path_values(payload, str(constraint["path"]))
        return len(values) == len(set(values))
    if kind == "maximum_items_by_value":
        values = _research_path_values(payload, str(constraint["path"]))
        return sum(value == constraint["value"] for value in values) <= int(
            constraint["maximum"]
        )
    if kind == "gated_tactic_kinds_subset":
        tactics = _research_path_values(payload, "tactics")
        items = tactics[0] if tactics else []
        gated = {
            str(item["kind"])
            for item in items
            if str(item.get("mode")) in {"hard", "qualification_gate"}
        }
        return gated.issubset(set(constraint["allowed"]))
    if kind == "tactic_row_matches":
        tactics = _research_path_values(payload, "tactics")
        items = tactics[0] if tactics else []
        return any(
            str(item.get("kind")) == str(constraint["tacticKind"])
            and str(item.get("mode")) in set(constraint["modes"])
            for item in items
            if isinstance(item, Mapping)
        )
    if kind == "tactic_row_matches_any":
        tactics = _research_path_values(payload, "tactics")
        items = tactics[0] if tactics else []
        tactic_kinds = {str(item) for item in constraint["tacticKinds"]}
        modes = {str(item) for item in constraint["modes"]}
        return any(
            str(item.get("kind")) in tactic_kinds
            and str(item.get("mode")) in modes
            for item in items
            if isinstance(item, Mapping)
        )
    if kind == "required_cards_fit_suit_ranges":
        required_values = _research_path_values(
            payload, "situation.requiredCards"
        )
        range_values = _research_path_values(
            payload, "situation.finalSuitCountRanges"
        )
        if not required_values or not range_values:
            return True
        required = dict(required_values[0])
        ranges = dict(range_values[0])
        suit_names = {
            "S": "spades",
            "H": "hearts",
            "D": "diamonds",
            "C": "clubs",
        }
        for seat, cards in required.items():
            seat_ranges = dict(ranges.get(seat) or {})
            for suit_code, suit_name in suit_names.items():
                required_count = sum(
                    str(card).startswith(suit_code) for card in cards
                )
                maximum = int(
                    dict(seat_ranges.get(suit_name) or {}).get("maximum", 8)
                )
                if required_count > maximum:
                    return False
        return True
    if kind == "exact_items":
        values = _research_path_values(payload, str(constraint["path"]))
        return bool(values) and len(values[0]) == int(constraint["count"])
    raise RuntimeError(f"unsupported research relationship constraint: {kind}")


def _validate_declared_research_relationships(
    payload: Mapping[str, Any],
) -> None:
    relationships = research_recipe_configuration_v1()["relationships"]
    for relationship in relationships:
        if not _research_condition_matches(payload, relationship.get("if")):
            continue
        if not _validate_research_relation_assertion(payload, relationship):
            raise ValueError(
                f"research relationship violation: {relationship['code']}"
            )


def _validate_v3_research_field_types(payload: Mapping[str, Any]) -> None:
    fields = research_recipe_configuration_v1()["fields"]
    for path, definition in fields.items():
        if ".*" in path:
            continue
        values = _research_path_values(payload, str(path))
        if definition.get("required") is True:
            if "[]." in path:
                parent_path, child_path = path.split("[].", 1)
                parent_values = _research_path_values(payload, parent_path)
                for rows in parent_values:
                    if not isinstance(rows, list):
                        continue
                    for index, row in enumerate(rows):
                        if not isinstance(row, Mapping) or not _research_present(
                            _research_path_values(row, child_path)
                        ):
                            raise ValueError(
                                "research required field is missing: "
                                f"{parent_path}[{index}].{child_path}"
                            )
            elif not _research_present(values):
                raise ValueError(f"research required field is missing: {path}")
        if not values:
            continue
        value_type = str(definition["valueType"])
        for value in values:
            valid = True
            if value_type == "integer":
                valid = isinstance(value, int) and not isinstance(value, bool)
            elif value_type == "number":
                valid = isinstance(value, (int, float)) and not isinstance(
                    value, bool
                )
                if valid and not math.isfinite(float(value)):
                    raise ValueError(f"research field must be finite: {path}")
            elif value_type == "boolean":
                valid = isinstance(value, bool)
            elif value_type == "string":
                valid = isinstance(value, str)
            elif value_type == "integer_or_null":
                valid = value is None or (
                    isinstance(value, int) and not isinstance(value, bool)
                )
            elif value_type == "string_or_null":
                valid = value is None or isinstance(value, str)
            elif value_type == "array":
                valid = isinstance(value, list)
            elif value_type == "object":
                valid = isinstance(value, Mapping)
            elif value_type == "object_or_null":
                valid = value is None or isinstance(value, Mapping)
            else:
                raise RuntimeError(
                    f"unsupported research catalog value type: {value_type}"
                )
            if not valid:
                raise ValueError(
                    f"research field type mismatch: {path} requires {value_type}"
                )
            if value is None:
                continue
            if (
                value_type == "array"
                and definition.get("control") == "multi_enum"
                and definition.get("values")
            ):
                allowed_values = set(definition["values"])
                if any(item not in allowed_values for item in value):
                    raise ValueError(
                        f"research field value is unsupported: {path}"
                    )
            if "value" in definition and value != definition["value"]:
                raise ValueError(
                    f"research invariant field mismatch: {path}"
                )
            if definition.get("values") and value_type not in {
                "array",
                "object",
                "object_or_null",
            }:
                if value not in definition["values"]:
                    raise ValueError(f"research field value is unsupported: {path}")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise ValueError(f"research field must be finite: {path}")
                if "minimum" in definition and numeric < float(
                    definition["minimum"]
                ):
                    raise ValueError(f"research field is below minimum: {path}")
                if "exclusiveMinimum" in definition and numeric <= float(
                    definition["exclusiveMinimum"]
                ):
                    raise ValueError(
                        f"research field must exceed its minimum: {path}"
                    )
                if "maximum" in definition and numeric > float(
                    definition["maximum"]
                ):
                    raise ValueError(f"research field exceeds maximum: {path}")
            if isinstance(value, str):
                if "minimumLength" in definition and len(value) < int(
                    definition["minimumLength"]
                ):
                    raise ValueError(f"research field is too short: {path}")
                if "maximumLength" in definition and len(value) > int(
                    definition["maximumLength"]
                ):
                    raise ValueError(f"research field is too long: {path}")
                if definition.get("pattern") and re.fullmatch(
                    str(definition["pattern"]), value
                ) is None:
                    raise ValueError(f"research field pattern mismatch: {path}")
            if isinstance(value, list):
                control = str(definition["control"])
                if control in {"multi_enum", "integer_pair"}:
                    allowed = definition.get("values")
                    if allowed is not None and any(
                        item not in allowed for item in value
                    ):
                        raise ValueError(
                            f"research field contains an unsupported value: {path}"
                        )
                if "exactLength" in definition and len(value) != int(
                    definition["exactLength"]
                ):
                    raise ValueError(
                        f"research field length is invalid: {path}"
                    )
                if control in {"multi_enum", "repeatable_tactic"} and (
                    "maximum" in definition
                    and len(value) > int(definition["maximum"])
                ):
                    raise ValueError(f"research field has too many items: {path}")

    match_score = _research_path_values(payload, "situation.matchScore")
    if match_score:
        score = match_score[0]
        if not isinstance(score, list) or len(score) != 2 or any(
            not isinstance(item, int) or isinstance(item, bool)
            for item in score
        ):
            raise ValueError(
                "research field type mismatch: situation.matchScore requires "
                "exactly two integers"
            )
    for path in ("situation.requiredCards", "situation.forbiddenCards"):
        card_maps = _research_path_values(payload, path)
        if card_maps and any(
            not isinstance(cards, list)
            or any(not isinstance(card, str) for card in cards)
            for cards in dict(card_maps[0]).values()
        ):
            raise ValueError(
                f"research field type mismatch: {path} requires card arrays"
            )
    suit_ranges = _research_path_values(
        payload, "situation.finalSuitCountRanges"
    )
    if suit_ranges:
        for seat_ranges in dict(suit_ranges[0]).values():
            if not isinstance(seat_ranges, Mapping):
                raise ValueError(
                    "research field type mismatch: "
                    "situation.finalSuitCountRanges requires suit objects"
                )
            for bounds in seat_ranges.values():
                if not isinstance(bounds, Mapping) or any(
                    not isinstance(bound, int) or isinstance(bound, bool)
                    for bound in bounds.values()
                ):
                    raise ValueError(
                        "research field type mismatch: "
                        "situation.finalSuitCountRanges requires integer bounds"
                    )


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ActorSafeCpSatAuctionProfileV1:
    requested_samples: int
    integer_scale: int
    maximum_conflicts: int
    maximum_selection_seconds: float
    objective: str
    tie_break: str
    require_optimal_for_qualification: bool
    sha256: str

    @classmethod
    def build(cls, payload: Mapping[str, Any]) -> "ActorSafeCpSatAuctionProfileV1":
        raw = dict(payload)
        authored = {
            "schemaVersion",
            "requestedSamples",
            "integerScale",
            "maximumConflicts",
            "maximumSelectionSeconds",
            "objective",
            "tieBreak",
            "requireOptimalForQualification",
        }
        derived = {"solverBackend", "sha256"}
        if (
            not authored.issubset(raw)
            or set(raw) - authored - derived
            or int(raw.get("schemaVersion", 0)) != 1
        ):
            raise ValueError(
                "actor-safe CP-SAT auction profile is incomplete or contains unknown keys"
            )
        samples = int(raw["requestedSamples"])
        scale = int(raw["integerScale"])
        conflicts = int(raw["maximumConflicts"])
        seconds = float(raw["maximumSelectionSeconds"])
        objective = str(raw["objective"])
        tie_break = str(raw["tieBreak"])
        require_optimal = bool(raw["requireOptimalForQualification"])
        if not 2 <= samples <= 1_000:
            raise ValueError("actor-safe auction sample count must be in [2, 1000]")
        if not 100 <= scale <= 100_000:
            raise ValueError("actor-safe auction integer scale is invalid")
        if not 1 <= conflicts <= 10_000_000:
            raise ValueError("actor-safe auction conflict budget is invalid")
        if not 0.01 <= seconds <= 30.0:
            raise ValueError("actor-safe auction time budget is invalid")
        if objective != BID_DECISION_OBJECTIVE_V1:
            raise ValueError("unsupported actor-safe auction objective")
        if tie_break != RESEARCH_STRATEGIC_AUCTION_TIE_BREAK_V1:
            raise ValueError("unsupported actor-safe auction tie-break")
        core = {
            "schemaVersion": 1,
            "requestedSamples": samples,
            "integerScale": scale,
            "maximumConflicts": conflicts,
            "maximumSelectionSeconds": seconds,
            "objective": objective,
            "tieBreak": tie_break,
            "requireOptimalForQualification": require_optimal,
            "solverBackend": "ortools_cp_sat",
        }
        digest = canonical_sha256(core)
        if raw.get("solverBackend") not in {None, "ortools_cp_sat"}:
            raise ValueError("unsupported actor-safe auction solver backend")
        if raw.get("sha256") not in {None, digest}:
            raise ValueError("actor-safe auction profile checksum mismatch")
        return cls(
            samples,
            scale,
            conflicts,
            seconds,
            objective,
            tie_break,
            require_optimal,
            digest,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "requestedSamples": self.requested_samples,
            "integerScale": self.integer_scale,
            "maximumConflicts": self.maximum_conflicts,
            "maximumSelectionSeconds": self.maximum_selection_seconds,
            "objective": self.objective,
            "tieBreak": self.tie_break,
            "requireOptimalForQualification": self.require_optimal_for_qualification,
            "solverBackend": "ortools_cp_sat",
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ActorSafeCpSatPlayProfileV1:
    minimum_played_cards: int
    requested_samples: int
    exact_allocation_limit: int
    maximum_sampling_attempts: int
    maximum_worlds: int
    maximum_continuation_steps: int
    maximum_plies: int
    maximum_selection_seconds: float
    require_exact_for_qualification: bool
    continuation_policy: str
    policy_profile_id: str
    policy_revision: int
    outcome_boundary: int
    utility: str
    tie_break: str
    sha256: str

    @classmethod
    def build(cls, payload: Mapping[str, Any]) -> "ActorSafeCpSatPlayProfileV1":
        raw = dict(payload)
        authored = {
            "schemaVersion",
            "minimumPlayedCards",
            "requestedSamples",
            "exactAllocationLimit",
            "maximumSamplingAttempts",
            "maximumWorlds",
            "maximumContinuationSteps",
            "maximumPlies",
            "maximumSelectionSeconds",
            "requireExactForQualification",
            "continuationPolicy",
            "policyProfileId",
            "policyRevision",
            "outcomeBoundary",
            "utility",
            "tieBreak",
        }
        derived = {"solverBackend", "sha256"}
        if (
            not authored.issubset(raw)
            or set(raw) - authored - derived
            or int(raw.get("schemaVersion", 0)) != 1
        ):
            raise ValueError(
                "actor-safe CP-SAT play profile is incomplete or contains unknown keys"
            )
        minimum_played = int(raw["minimumPlayedCards"])
        requested = int(raw["requestedSamples"])
        exact_limit = int(raw["exactAllocationLimit"])
        attempts = int(raw["maximumSamplingAttempts"])
        worlds = int(raw["maximumWorlds"])
        continuation_steps = int(raw["maximumContinuationSteps"])
        plies = int(raw["maximumPlies"])
        seconds = float(raw["maximumSelectionSeconds"])
        require_exact = bool(raw["requireExactForQualification"])
        continuation_policy = str(raw["continuationPolicy"])
        policy_profile_id = str(raw["policyProfileId"])
        policy_revision = int(raw["policyRevision"])
        outcome_boundary = int(raw["outcomeBoundary"])
        utility = str(raw["utility"])
        tie_break = str(raw["tieBreak"])
        if not 20 <= minimum_played <= 28:
            raise ValueError("strategic play must start between cards 20 and 28")
        if not 1 <= requested <= worlds <= 5_000:
            raise ValueError("strategic hidden-world bounds are invalid")
        if not requested <= attempts <= 1_000_000:
            raise ValueError("strategic hidden-world attempt bound is invalid")
        if not 1 <= exact_limit <= worlds:
            raise ValueError("strategic exact-allocation limit exceeds the world bound")
        if not 1 <= continuation_steps <= 2_000_000:
            raise ValueError("strategic continuation-step bound is invalid")
        if not 32 - minimum_played <= plies <= 12:
            raise ValueError("strategic ply bound cannot complete the configured tail")
        if not 0.01 <= seconds <= 30.0:
            raise ValueError("strategic solver time bound is invalid")
        if continuation_policy != "observer_safe_heuristic_v1":
            raise ValueError("unsupported strategic continuation policy")
        if policy_profile_id != "maximin_margin" or policy_revision != 1:
            raise ValueError("unsupported strategic policy profile")
        policy_profile = ReducedPolicyProfileV1(
            profile_id=policy_profile_id,
            revision=policy_revision,
            outcome_boundary=outcome_boundary,
            utility=utility,
            tie_break=tie_break,
        )
        core = {
            "schemaVersion": 1,
            "minimumPlayedCards": minimum_played,
            "requestedSamples": requested,
            "exactAllocationLimit": exact_limit,
            "maximumSamplingAttempts": attempts,
            "maximumWorlds": worlds,
            "maximumContinuationSteps": continuation_steps,
            "maximumPlies": plies,
            "maximumSelectionSeconds": seconds,
            "requireExactForQualification": require_exact,
            "continuationPolicy": continuation_policy,
            "policyProfile": policy_profile.to_dict(),
            "solverBackend": "batched_cp_sat",
        }
        digest = canonical_sha256(core)
        if raw.get("solverBackend") not in {None, "batched_cp_sat"}:
            raise ValueError("unsupported strategic solver backend")
        if raw.get("sha256") not in {None, digest}:
            raise ValueError("strategic play profile checksum mismatch")
        return cls(
            minimum_played,
            requested,
            exact_limit,
            attempts,
            worlds,
            continuation_steps,
            plies,
            seconds,
            require_exact,
            continuation_policy,
            policy_profile_id,
            policy_revision,
            outcome_boundary,
            utility,
            tie_break,
            digest,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "minimumPlayedCards": self.minimum_played_cards,
            "requestedSamples": self.requested_samples,
            "exactAllocationLimit": self.exact_allocation_limit,
            "maximumSamplingAttempts": self.maximum_sampling_attempts,
            "maximumWorlds": self.maximum_worlds,
            "maximumContinuationSteps": self.maximum_continuation_steps,
            "maximumPlies": self.maximum_plies,
            "maximumSelectionSeconds": self.maximum_selection_seconds,
            "requireExactForQualification": self.require_exact_for_qualification,
            "continuationPolicy": self.continuation_policy,
            "policyProfileId": self.policy_profile_id,
            "policyRevision": self.policy_revision,
            "outcomeBoundary": self.outcome_boundary,
            "utility": self.utility,
            "tieBreak": self.tie_break,
            "solverBackend": "batched_cp_sat",
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ConstructionPreScreenProfileV1:
    profile_id: str
    revision: str
    enabled: bool
    maximum_play_decisions: int
    maximum_bid_regret: float
    maximum_play_regret: float
    maximum_probability_drop: float
    minimum_confidence: float
    reject_catastrophic_crossing: bool
    sha256: str

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "ConstructionPreScreenProfileV1":
        allowed = {
            "schemaVersion", "id", "revision", "enabled",
            "maximumPlayDecisions", "maximumBidRegret", "maximumPlayRegret",
            "maximumProbabilityDrop", "minimumConfidence",
            "rejectCatastrophicCrossing", "sha256",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                f"unknown construction pre-screen keys: {sorted(unknown)}"
            )
        core = {
            "schemaVersion": int(value.get("schemaVersion", 0)),
            "id": str(value.get("id") or "").strip(),
            "revision": str(value.get("revision") or "").strip(),
            "enabled": bool(value.get("enabled", False)),
            "maximumPlayDecisions": int(value.get("maximumPlayDecisions", 0)),
            "maximumBidRegret": float(value.get("maximumBidRegret", 0.0)),
            "maximumPlayRegret": float(value.get("maximumPlayRegret", 0.0)),
            "maximumProbabilityDrop": float(
                value.get("maximumProbabilityDrop", 0.0)
            ),
            "minimumConfidence": float(value.get("minimumConfidence", 0.0)),
            "rejectCatastrophicCrossing": bool(
                value.get("rejectCatastrophicCrossing", True)
            ),
        }
        if core["schemaVersion"] != 1 or not core["id"] or not core["revision"]:
            raise ValueError("construction pre-screen identity is invalid")
        if not 1 <= core["maximumPlayDecisions"] <= 16:
            raise ValueError("construction pre-screen play bound must be in [1, 16]")
        if min(core["maximumBidRegret"], core["maximumPlayRegret"]) < 0:
            raise ValueError("construction pre-screen regret bounds cannot be negative")
        if not 0 <= core["maximumProbabilityDrop"] <= 1:
            raise ValueError("construction pre-screen probability bound is invalid")
        if not 0 <= core["minimumConfidence"] <= 1:
            raise ValueError("construction pre-screen confidence is invalid")
        digest = canonical_sha256(core)
        declared = str(value.get("sha256") or "")
        if declared and declared != digest:
            raise ValueError("construction pre-screen checksum mismatch")
        return cls(
            core["id"],
            core["revision"],
            core["enabled"],
            core["maximumPlayDecisions"],
            core["maximumBidRegret"],
            core["maximumPlayRegret"],
            core["maximumProbabilityDrop"],
            core["minimumConfidence"],
            core["rejectCatastrophicCrossing"],
            digest,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "id": self.profile_id,
            "revision": self.revision,
            "enabled": self.enabled,
            "maximumPlayDecisions": self.maximum_play_decisions,
            "maximumBidRegret": self.maximum_bid_regret,
            "maximumPlayRegret": self.maximum_play_regret,
            "maximumProbabilityDrop": self.maximum_probability_drop,
            "minimumConfidence": self.minimum_confidence,
            "rejectCatastrophicCrossing": self.reject_catastrophic_crossing,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class TacticalViabilityPreScreenProfileV1:
    """Governed exact-line rejection screen before full actor-safe analysis."""

    profile_id: str
    revision: str
    enabled: bool
    maximum_decisions: int
    required_tactics_only: bool
    reject_when_exact_absent: bool
    sha256: str

    @classmethod
    def build(
        cls, value: Mapping[str, Any]
    ) -> "TacticalViabilityPreScreenProfileV1":
        allowed = {
            "schemaVersion",
            "id",
            "revision",
            "enabled",
            "maximumDecisions",
            "requiredTacticsOnly",
            "rejectWhenExactAbsent",
            "sha256",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                f"unknown tactical viability pre-screen keys: {sorted(unknown)}"
            )
        core = {
            "schemaVersion": int(value.get("schemaVersion", 0)),
            "id": str(value.get("id") or "").strip(),
            "revision": str(value.get("revision") or "").strip(),
            "enabled": bool(value.get("enabled", False)),
            "maximumDecisions": int(value.get("maximumDecisions", 0)),
            "requiredTacticsOnly": bool(
                value.get("requiredTacticsOnly", True)
            ),
            "rejectWhenExactAbsent": bool(
                value.get("rejectWhenExactAbsent", True)
            ),
        }
        if (
            core["schemaVersion"] != 1
            or core["id"] != "exact_tactical_viability_screen_v1"
            or re.fullmatch(r"[1-9][0-9]*", core["revision"]) is None
        ):
            raise ValueError("tactical viability pre-screen identity is invalid")
        if not 1 <= core["maximumDecisions"] <= 16:
            raise ValueError(
                "tactical viability pre-screen decision bound must be in [1, 16]"
            )
        digest = canonical_sha256(core)
        declared = str(value.get("sha256") or "")
        if declared and declared != digest:
            raise ValueError("tactical viability pre-screen checksum mismatch")
        return cls(
            core["id"],
            core["revision"],
            core["enabled"],
            core["maximumDecisions"],
            core["requiredTacticsOnly"],
            core["rejectWhenExactAbsent"],
            digest,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "id": self.profile_id,
            "revision": self.revision,
            "enabled": self.enabled,
            "maximumDecisions": self.maximum_decisions,
            "requiredTacticsOnly": self.required_tactics_only,
            "rejectWhenExactAbsent": self.reject_when_exact_absent,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ReducedPolicyProfileV1:
    """Versioned semantics for one bounded policy proof."""

    profile_id: str = "maximin_margin"
    revision: int = 1
    outcome_boundary: int = 1
    utility: str = "actor_team_awarded_point_margin"
    tie_break: str = "maximin_then_total_then_card_id"

    def __post_init__(self) -> None:
        if not self.profile_id.strip() or self.revision < 1:
            raise ValueError("policy profile requires a stable id and positive revision")
        if self.utility != "actor_team_awarded_point_margin":
            raise ValueError(f"unsupported reduced-policy utility: {self.utility}")
        if self.tie_break != "maximin_then_total_then_card_id":
            raise ValueError(f"unsupported reduced-policy tie break: {self.tie_break}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "profileId": self.profile_id,
            "revision": self.revision,
            "outcomeBoundary": self.outcome_boundary,
            "utility": self.utility,
            "tieBreak": self.tie_break,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def _research_strategic_auction_profile_for_legacy_recipe(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Hydrate the historical tie-break without admitting it to V3.

    ResearchRecipeV1 is also the common-contract validator used by V2.  Those
    immutable records may contain the former tie-break and its exact profile
    hash.  V3 validates its catalog invariants before reaching this adapter, so
    current authoring remains restricted to the canonical tie-break.
    """

    raw = copy.deepcopy(dict(value))
    if raw.get("tieBreak") != _LEGACY_RESEARCH_STRATEGIC_AUCTION_TIE_BREAK_V1:
        return ActorSafeCpSatAuctionProfileV1.build(raw).to_dict()

    declared_hash = str(raw.pop("sha256", ""))
    raw["tieBreak"] = RESEARCH_STRATEGIC_AUCTION_TIE_BREAK_V1
    current = ActorSafeCpSatAuctionProfileV1.build(raw).to_dict()
    current.pop("sha256", None)
    current["tieBreak"] = _LEGACY_RESEARCH_STRATEGIC_AUCTION_TIE_BREAK_V1
    legacy_hash = canonical_sha256(current)
    if declared_hash and declared_hash != legacy_hash:
        raise ValueError("actor-safe auction profile checksum mismatch")
    return {**current, "sha256": legacy_hash}


def _closed(value: Mapping[str, Any], allowed: Iterable[str], where: str) -> dict[str, Any]:
    raw = copy.deepcopy(dict(value))
    unknown = sorted(set(raw) - set(allowed))
    if unknown:
        raise ValueError(f"unknown {where} keys: {unknown}")
    return raw


def _hash(value: str, name: str) -> str:
    text = str(value or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise ValueError(f"{name} must be a SHA-256 identity")
    return text


def _rule_leaves(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    if isinstance(value, Mapping):
        for key, child in sorted(value.items(), key=lambda item: str(item[0])):
            yield from _rule_leaves(child, (*path, str(key)))
        return
    yield path, copy.deepcopy(value)


def _compiler_stage(path: tuple[str, ...]) -> str:
    section = path[0]
    if section in {"deck", "seating", "dealing"}:
        return "cp_sat_native_constraint"
    if section == "projects":
        return "cp_sat_pattern_plus_engine_gate"
    if section in {"contracts", "card_rules", "bidding", "escalation", "play", "scoring"}:
        return "engine_semantic_replay_gate"
    if section in {"policies", "analysis_defaults"}:
        return "agent_or_analysis_policy_gate"
    if section == "disputed_options":
        return "published_variant_gate"
    return "uncovered"


_RULE_EVIDENCE: dict[str, dict[str, tuple[str, ...]]] = {
    "deck": {
        "runtime": ("src/baloot_platform/core/cards.py", "src/baloot_platform/core/rules.py"),
        "solver": ("src/baloot_platform/challenge/constraint_deal_search.py",),
        "tests": ("tests/test_constraint_deal_search.py", "tests/test_rules_runtime_authority.py"),
    },
    "seating": {
        "runtime": ("src/baloot_platform/core/dealing.py",),
        "solver": ("src/baloot_platform/challenge/constraint_deal_search.py",),
        "tests": ("tests/test_constraint_deal_search.py",),
    },
    "dealing": {
        "runtime": ("src/baloot_platform/core/dealing.py", "src/baloot_platform/core/shuffle.py"),
        "solver": ("src/baloot_platform/challenge/constraint_deal_search.py",),
        "tests": ("tests/test_constraint_deal_search.py",),
    },
    "card_rules": {
        "runtime": ("src/baloot_platform/core/cards.py", "src/baloot_platform/core/rules.py"),
        "solver": ("src/baloot_platform/challenge/replay.py",),
        "tests": ("tests/test_play_obligations.py", "tests/test_rules_runtime_authority.py"),
    },
    "contracts": {
        "runtime": ("src/baloot_platform/core/bidding.py",),
        "solver": ("src/baloot_platform/challenge/replay.py",),
        "tests": ("tests/test_bidding_escalation.py", "tests/test_rules_runtime_authority.py"),
    },
    "bidding": {
        "runtime": ("src/baloot_platform/core/bidding.py",),
        "solver": ("src/baloot_platform/challenge/replay.py",),
        "tests": ("tests/test_bidding_escalation.py",),
    },
    "escalation": {
        "runtime": ("src/baloot_platform/core/bidding.py",),
        "solver": ("src/baloot_platform/challenge/replay.py",),
        "tests": ("tests/test_bidding_escalation.py",),
    },
    "projects": {
        "runtime": ("src/baloot_platform/core/projects.py",),
        "solver": ("src/baloot_platform/challenge/constraint_deal_search.py", "src/baloot_platform/challenge/replay.py"),
        "tests": ("tests/test_projects_scoring_extended.py", "tests/test_hidden_deals_projects.py"),
    },
    "play": {
        "runtime": ("src/baloot_platform/core/engine.py",),
        "solver": ("src/baloot_platform/challenge/replay.py",),
        "tests": ("tests/test_play_obligations.py",),
    },
    "scoring": {
        "runtime": ("src/baloot_platform/core/scoring.py",),
        "solver": ("src/baloot_platform/challenge/replay.py",),
        "tests": ("tests/test_projects_scoring_extended.py", "tests/test_rules_runtime_authority.py"),
    },
    "policies": {
        "runtime": ("src/baloot_platform/agents/profiles.py",),
        "solver": ("src/baloot_platform/challenge/analysis_pipeline.py",),
        "tests": ("tests/test_agent_configuration.py",),
    },
    "analysis_defaults": {
        "runtime": ("src/baloot_platform/analysis/evaluator.py",),
        "solver": ("src/baloot_platform/challenge/analysis_pipeline.py",),
        "tests": ("tests/test_analysis.py",),
    },
    "disputed_options": {
        "runtime": ("src/baloot_platform/core/rules.py",),
        "solver": ("src/baloot_platform/challenge/replay.py",),
        "tests": ("tests/test_rules_candidate.py", "tests/test_rules_runtime_authority.py"),
    },
}


def _coverage_evidence(section: str, stage: str) -> dict[str, Any]:
    evidence = _RULE_EVIDENCE.get(section, {})
    documentation = (
        "docs/ENGINE_GUIDE_AR.md",
        "docs/OR_TOOLS_RESEARCH_FOUNDRY.md",
    )
    return {
        "runtime": list(evidence.get("runtime", ())),
        "solver": list(evidence.get("solver", ())),
        "documentation": list(documentation),
        "tests": list(evidence.get("tests", ())),
        "proofBoundary": (
            "direct_cp_sat_constraint"
            if stage.startswith("cp_sat_")
            else "deterministic_post_solve_semantic_gate"
        ),
    }


@dataclass(frozen=True, slots=True)
class BalootDomainSpecV2:
    """Frozen view compiled from existing published Baloot authorities."""

    rules: Mapping[str, Any]
    conventions: Mapping[str, Any]
    localization: Mapping[str, Any]
    rule_primitives: tuple[Mapping[str, Any], ...]
    tactical_patterns: tuple[str, ...]
    plugin_capabilities: tuple[Mapping[str, Any], ...]
    rules_hash: str
    convention_hash: str
    localization_hash: str
    bundle_hash: str
    sha256: str

    @classmethod
    def build(
        cls,
        *,
        rules: Mapping[str, Any],
        conventions: Mapping[str, Any],
        localization: Mapping[str, Any],
        bundle_hash: str,
        plugins: Iterable[Mapping[str, Any]] = (),
    ) -> "BalootDomainSpecV2":
        frozen_rules = copy.deepcopy(dict(rules))
        frozen_conventions = copy.deepcopy(dict(conventions))
        frozen_localization = copy.deepcopy(dict(localization))
        primitives: list[dict[str, Any]] = []
        for section, payload in sorted(frozen_rules.items()):
            if section in {"schema_version", "profile"}:
                continue
            for path, leaf in _rule_leaves(payload, (str(section),)):
                stage = _compiler_stage(path)
                evidence = _coverage_evidence(str(section), stage)
                primitives.append({
                    "id": "rules." + ".".join(path),
                    "kind": "built_in_rule_primitive",
                    "enabled": True,
                    "runtimeCoverage": stage != "uncovered",
                    "solverCoverage": stage != "uncovered",
                    "compilerStage": stage,
                    "cpSatNative": stage.startswith("cp_sat_"),
                    "documentationCoverage": stage != "uncovered",
                    "testCoverage": stage != "uncovered",
                    "coverageEvidence": evidence,
                    "payloadHash": canonical_sha256(leaf),
                })
        plugin_rows = tuple(copy.deepcopy(dict(item)) for item in plugins)
        rules_hash = canonical_sha256(frozen_rules)
        convention_hash = canonical_sha256(frozen_conventions)
        localization_hash = canonical_sha256(frozen_localization)
        core = {
            "schemaVersion": 2,
            "rulesHash": rules_hash,
            "conventionHash": convention_hash,
            "localizationHash": localization_hash,
            "bundleHash": _hash(bundle_hash, "bundleHash"),
            "rulePrimitives": primitives,
            "tacticalPatterns": sorted(_TACTICS),
            "pluginCapabilities": list(plugin_rows),
        }
        return cls(
            frozen_rules,
            frozen_conventions,
            frozen_localization,
            tuple(primitives),
            tuple(sorted(_TACTICS)),
            plugin_rows,
            rules_hash,
            convention_hash,
            localization_hash,
            core["bundleHash"],
            canonical_sha256(core),
        )

    def to_dict(self, *, include_payloads: bool = False) -> dict[str, Any]:
        value = {
            "schemaVersion": 2,
            "rulesHash": self.rules_hash,
            "conventionHash": self.convention_hash,
            "localizationHash": self.localization_hash,
            "bundleHash": self.bundle_hash,
            "rulePrimitives": [dict(item) for item in self.rule_primitives],
            "tacticalPatterns": list(self.tactical_patterns),
            "pluginCapabilities": [dict(item) for item in self.plugin_capabilities],
            "sha256": self.sha256,
        }
        if include_payloads:
            value.update({
                "rules": copy.deepcopy(dict(self.rules)),
                "conventions": copy.deepcopy(dict(self.conventions)),
                "localization": copy.deepcopy(dict(self.localization)),
            })
        return value


@dataclass(frozen=True, slots=True)
class ResearchRecipeV1:
    code: str
    title_ar: str
    output: Mapping[str, Any]
    profiles: Mapping[str, Any]
    situation: Mapping[str, Any]
    tactics: tuple[Mapping[str, Any], ...]
    qualification: Mapping[str, Any]
    resources: Mapping[str, Any]
    status: str
    sha256: str

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "ResearchRecipeV1":
        raw = _closed(
            value,
            {"schemaVersion", "code", "titleAr", "output", "profiles", "situation", "tactics", "qualification", "resources", "status", "sha256"},
            "research recipe",
        )
        if int(raw.get("schemaVersion", 1)) != 1:
            raise ValueError("unsupported research recipe schema version")
        code = str(raw.get("code") or "").strip().upper()
        if code and not re.fullmatch(r"RCP-[A-Z0-9]{6,20}", code):
            raise ValueError("research recipe code is invalid")
        title_ar = str(raw.get("titleAr") or "").strip()
        if not 3 <= len(title_ar) <= 160:
            raise ValueError("Saudi Arabic recipe title must contain 3 to 160 characters")

        output = _closed(raw.get("output") or {}, {"kind", "resultCount"}, "research output")
        output_kind = str(output.get("kind") or "laffa")
        result_count = int(output.get("resultCount", 1))
        if output_kind not in _OUTPUT_KINDS or not 1 <= result_count <= 1000:
            raise ValueError("research output kind or result count is invalid")
        output = {"kind": output_kind, "resultCount": result_count}

        profiles = _closed(
            raw.get("profiles") or {},
            {"domainHash", "bundleHash", "rulesHash", "conventionHash", "agentRelease", "analysisProfileHash", "scoreProfileHash"},
            "research profiles",
        )
        for key in ("domainHash", "bundleHash", "rulesHash", "conventionHash", "analysisProfileHash", "scoreProfileHash"):
            profiles[key] = _hash(str(profiles.get(key) or ""), key)
        profiles["agentRelease"] = str(profiles.get("agentRelease") or "").strip()
        if not profiles["agentRelease"]:
            raise ValueError("research recipe requires a frozen agent release")

        situation = _closed(
            raw.get("situation") or {},
            {
                "dealer", "buyer", "objectiveSeat", "constructionBuyerRelation",
                "purchaseRound", "purchaseSuits", "trumpSuit", "contracts",
                "matchScore", "requiredCards", "forbiddenCards",
                "finalSuitCountRanges", "projects", "escalations",
            },
            "research situation",
        )
        dealer = str(situation.get("dealer") or "east")
        buyer = situation.get("buyer")
        objective_seat = situation.get("objectiveSeat")
        buyer_relation = str(
            situation.get("constructionBuyerRelation") or "automatic"
        )
        if (
            dealer not in _SEATS
            or (buyer is not None and str(buyer) not in _SEATS)
            or (objective_seat is not None and str(objective_seat) not in _SEATS)
        ):
            raise ValueError("research dealer, buyer, or objective seat is invalid")
        if buyer_relation not in _CONSTRUCTION_BUYER_RELATIONS:
            raise ValueError("research construction buyer relationship is invalid")
        if buyer is not None and buyer_relation != "automatic":
            raise ValueError(
                "an explicit buyer cannot also use a construction buyer relationship"
            )
        contracts = sorted({str(item) for item in situation.get("contracts") or ()})
        if set(contracts) - _CONTRACTS:
            raise ValueError("research contains an unsupported contract")
        purchase_round = situation.get("purchaseRound")
        if purchase_round is not None:
            purchase_round = int(purchase_round)
            if purchase_round not in {1, 2}:
                raise ValueError("research purchase round must be one, two, or omitted")
        purchase_suits = sorted(
            {str(item) for item in situation.get("purchaseSuits") or ()}
        )
        if set(purchase_suits) - _SUITS:
            raise ValueError("research purchase suit is invalid")
        trump_suit = situation.get("trumpSuit")
        if trump_suit is not None:
            trump_suit = str(trump_suit)
            if trump_suit not in {"spades", "hearts", "diamonds", "clubs"}:
                raise ValueError("research trump suit is invalid")
            if contracts != ["hokum"]:
                raise ValueError("a requested trump suit requires only the Hokum contract")
        score = [int(item) for item in situation.get("matchScore") or (0, 0)]
        if len(score) != 2 or any(item < 0 or item > 10_000 for item in score):
            raise ValueError("research match score must contain two non-negative values")
        required_cards = _card_constraints(
            situation.get("requiredCards"), globally_unique=True
        )
        forbidden_cards = _card_constraints(
            situation.get("forbiddenCards"), globally_unique=False
        )
        if any(
            card in set(forbidden_cards.get(seat, ()))
            for seat, cards in required_cards.items()
            for card in cards
        ):
            raise ValueError("one card cannot be both required and forbidden for a seat")
        suit_ranges = _suit_count_ranges(situation.get("finalSuitCountRanges"))
        projects = []
        for item in situation.get("projects") or ():
            project = _closed(item, {"seat", "contract", "kind", "value", "highRank", "mode"}, "project constraint")
            if str(project.get("seat")) not in _SEATS:
                raise ValueError("project seat is invalid")
            if str(project.get("contract")) not in _CONTRACTS:
                raise ValueError("project contract is invalid")
            if str(project.get("kind")) not in _PROJECT_KINDS:
                raise ValueError("project kind is invalid")
            if str(project.get("mode", "hard")) not in _MODES:
                raise ValueError("project mode is invalid")
            project_contract = str(project.get("contract"))
            project_kind = str(project.get("kind"))
            if contracts and project_contract not in contracts:
                raise ValueError("project contract must match the requested contract")
            if project_kind == "four_hundred":
                if project_contract != "sun":
                    raise ValueError("four_hundred project requires sun contract")
                if int(project.get("value", 0)) != 400 or str(project.get("highRank")) != "ace":
                    raise ValueError("four_hundred project requires four aces worth 400")
            projects.append(project)
        escalations = sorted(
            {str(item).strip().lower() for item in situation.get("escalations") or ()}
        )
        if set(escalations) - _ESCALATION_LEVELS:
            raise ValueError("research contains an unsupported escalation")
        if len(escalations) > 1:
            raise ValueError("research supports one required escalation level")
        if escalations and escalations[0] in {"triple", "four", "gahwa"}:
            if contracts != ["hokum"]:
                raise ValueError(
                    "Triple, Four, and Gahwa research require the Hokum contract"
                )
        situation = {
            "dealer": dealer,
            "buyer": str(buyer) if buyer is not None else None,
            "objectiveSeat": (
                str(objective_seat) if objective_seat is not None else None
            ),
            "constructionBuyerRelation": buyer_relation,
            "purchaseRound": purchase_round,
            "purchaseSuits": purchase_suits,
            "trumpSuit": trump_suit,
            "contracts": contracts,
            "matchScore": score,
            "requiredCards": required_cards,
            "forbiddenCards": forbidden_cards,
            "finalSuitCountRanges": suit_ranges,
            "projects": projects,
            "escalations": escalations,
        }

        tactics = []
        for item in raw.get("tactics") or ():
            tactic = _closed(item, {"kind", "mode", "minimumScore", "weight", "perspectives"}, "tactical constraint")
            kind = str(tactic.get("kind") or "")
            mode = str(tactic.get("mode") or "preference")
            score_value = float(tactic.get("minimumScore", 0.0))
            weight = float(tactic.get("weight", 1.0))
            if kind not in _TACTICS or mode not in _MODES:
                raise ValueError("research tactical kind or mode is invalid")
            if not 0.0 <= score_value <= 100.0 or not 0 < weight <= 100:
                raise ValueError("research tactical score or weight is invalid")
            perspectives = sorted({str(item) for item in tactic.get("perspectives") or ()})
            if set(perspectives) - _SEATS:
                raise ValueError("research tactical perspective is invalid")
            tactics.append({"kind": kind, "mode": mode, "minimumScore": score_value, "weight": weight, "perspectives": perspectives})
        tactic_kinds = [item["kind"] for item in tactics]
        if len(tactic_kinds) != len(set(tactic_kinds)):
            raise ValueError("research tactical constraints must use unique kinds")

        qualification = _closed(
            raw.get("qualification") or {},
            {"minimumTruth", "minimumConfidence", "minimumSurroundingQuality", "requireObserverSafety", "requireAllDecisionAudit", "maximumRegret"},
            "research qualification",
        )
        minimum_truth = str(qualification.get("minimumTruth") or "robust_best_known")
        if minimum_truth not in _TRUTH_LEVELS:
            raise ValueError("research minimum truth level is invalid")
        qualification = {
            "minimumTruth": minimum_truth,
            "minimumConfidence": float(qualification.get("minimumConfidence", 0.75)),
            "minimumSurroundingQuality": float(qualification.get("minimumSurroundingQuality", 70.0)),
            "requireObserverSafety": bool(qualification.get("requireObserverSafety", True)),
            "requireAllDecisionAudit": bool(qualification.get("requireAllDecisionAudit", True)),
            "maximumRegret": float(qualification.get("maximumRegret", 0.0)),
        }
        if not 0 <= qualification["minimumConfidence"] <= 1:
            raise ValueError("research minimum confidence is invalid")
        if not 0 <= qualification["minimumSurroundingQuality"] <= 100:
            raise ValueError("research surrounding quality is invalid")
        if qualification["maximumRegret"] < 0:
            raise ValueError("research maximum regret is invalid")

        resources = _closed(
            raw.get("resources") or {},
            {
                "generationLane", "seedStart", "seedCount", "seedsPerShard",
                "candidatesPerSeed", "solverTimeoutSeconds", "maximumConflicts",
                "analysisMode", "deepVerificationPolicy",
                "tacticalConstruction",
                "constructionPreScreen", "tacticalViabilityPreScreen",
                "tacticalDealBalance",
                "playConstructionMode",
                "strategicPlay", "auctionConstructionMode", "strategicAuction",
            },
            "research resources",
        )
        resources = {
            "generationLane": str(resources.get("generationLane") or "ortools"),
            "seedStart": int(resources.get("seedStart", 1)),
            "seedCount": int(resources.get("seedCount", 10)),
            "seedsPerShard": int(resources.get("seedsPerShard", 10)),
            "candidatesPerSeed": int(resources.get("candidatesPerSeed", 1)),
            "solverTimeoutSeconds": float(resources.get("solverTimeoutSeconds", 15.0)),
            "maximumConflicts": int(resources.get("maximumConflicts", 250_000)),
            "analysisMode": str(resources.get("analysisMode") or "cascaded"),
            "deepVerificationPolicy": str(
                resources.get("deepVerificationPolicy") or "required_gates"
            ),
        }
        raw_pre_screen = dict(
            (raw.get("resources") or {}).get("constructionPreScreen") or {}
        )
        if raw_pre_screen:
            resources["constructionPreScreen"] = (
                ConstructionPreScreenProfileV1.build(raw_pre_screen).to_dict()
            )
        raw_tactical_pre_screen = dict(
            (raw.get("resources") or {}).get("tacticalViabilityPreScreen") or {}
        )
        if raw_tactical_pre_screen:
            resources["tacticalViabilityPreScreen"] = (
                TacticalViabilityPreScreenProfileV1.build(
                    raw_tactical_pre_screen
                ).to_dict()
            )
        raw_tactical_deal_balance = dict(
            (raw.get("resources") or {}).get("tacticalDealBalance") or {}
        )
        if raw_tactical_deal_balance:
            from .tactical_construction import TacticalDealBalanceProfileV1

            resources["tacticalDealBalance"] = (
                TacticalDealBalanceProfileV1.build(
                    raw_tactical_deal_balance
                ).to_dict()
            )
        raw_tactical_construction = dict(
            (raw.get("resources") or {}).get("tacticalConstruction") or {}
        )
        if raw_tactical_construction:
            from .tactical_construction import TacticalPlayConstructionProfileV1

            resources["tacticalConstruction"] = (
                TacticalPlayConstructionProfileV1.build(
                    raw_tactical_construction
                ).to_dict()
            )
        raw_play_construction_mode = (raw.get("resources") or {}).get(
            "playConstructionMode"
        )
        if raw_play_construction_mode is not None:
            play_construction_mode = str(raw_play_construction_mode)
            if play_construction_mode not in _PLAY_CONSTRUCTION_MODES:
                raise ValueError("research play-construction mode is invalid")
            resources["playConstructionMode"] = play_construction_mode
        raw_auction_construction_mode = (raw.get("resources") or {}).get(
            "auctionConstructionMode"
        )
        if raw_auction_construction_mode is not None:
            auction_construction_mode = str(raw_auction_construction_mode)
            if auction_construction_mode not in _AUCTION_CONSTRUCTION_MODES:
                raise ValueError("research auction-construction mode is invalid")
            resources["auctionConstructionMode"] = auction_construction_mode
        raw_strategic_auction = dict(
            (raw.get("resources") or {}).get("strategicAuction") or {}
        )
        if raw_strategic_auction:
            resources["strategicAuction"] = (
                _research_strategic_auction_profile_for_legacy_recipe(
                    raw_strategic_auction
                )
            )
        if (
            resources.get("auctionConstructionMode")
            == "cp_sat_actor_safe_decision_v1"
            and "strategicAuction" not in resources
        ):
            raise ValueError(
                "CP-SAT actor-safe auction mode requires a strategic auction profile"
            )
        if (
            "strategicAuction" in resources
            and resources.get("auctionConstructionMode")
            != "cp_sat_actor_safe_decision_v1"
        ):
            raise ValueError(
                "strategic auction profile requires the CP-SAT actor-safe auction mode"
            )
        raw_strategic_play = dict(
            (raw.get("resources") or {}).get("strategicPlay") or {}
        )
        if raw_strategic_play:
            outcome_boundary = int(raw_strategic_play.get("outcomeBoundary", 0))
            if not -10_000 <= outcome_boundary <= 10_000:
                raise ValueError(
                    "strategic play outcome boundary must be in [-10000, 10000]"
                )
            resources["strategicPlay"] = (
                ActorSafeCpSatPlayProfileV1.build(raw_strategic_play).to_dict()
            )
        if (
            resources.get("playConstructionMode")
            == "cp_sat_actor_safe_late_game_v1"
            and "strategicPlay" not in resources
        ):
            raise ValueError(
                "CP-SAT actor-safe late-game mode requires a strategic play profile"
            )
        if (
            "strategicPlay" in resources
            and resources.get("playConstructionMode")
            != "cp_sat_actor_safe_late_game_v1"
        ):
            raise ValueError(
                "strategic play profile requires the CP-SAT actor-safe late-game mode"
            )
        if "tacticalConstruction" in resources:
            tactical_construction = dict(resources["tacticalConstruction"])
            if tactical_construction["enabled"]:
                if (
                    resources["generationLane"] != "ortools"
                    or resources.get("playConstructionMode")
                    != "cp_sat_feasibility_line"
                ):
                    raise ValueError(
                        "tactical play construction requires the OR-Tools CP-SAT feasibility line"
                    )
                gated_kinds = {
                    str(item["kind"])
                    for item in tactics
                    if item["mode"] in {"hard", "qualification_gate"}
                }
                if "small_card_big_gain" not in gated_kinds:
                    raise ValueError(
                        "tactical play construction requires a gated small-card-big-gain tactic"
                    )
                if (
                    tactical_construction["requirePrecursorForGatedTactics"]
                    and gated_kinds != {"small_card_big_gain"}
                ):
                    raise ValueError(
                        "the active tactical play constructor cannot compile every gated tactic"
                    )
        if "tacticalViabilityPreScreen" in resources:
            tactical_pre_screen = dict(resources["tacticalViabilityPreScreen"])
            if tactical_pre_screen["enabled"]:
                gated_tactics = [
                    item
                    for item in tactics
                    if item["mode"] in {"hard", "qualification_gate"}
                ]
                if resources["deepVerificationPolicy"] == "off":
                    raise ValueError(
                        "tactical viability pre-screen requires deep verification"
                    )
                if tactical_pre_screen["requiredTacticsOnly"] and not gated_tactics:
                    raise ValueError(
                        "required-tactic viability screening requires a hard or qualification-gate tactic"
                    )
        if "tacticalDealBalance" in resources:
            tactical_deal_balance = dict(resources["tacticalDealBalance"])
            if tactical_deal_balance["enabled"]:
                gated_kinds = {
                    str(item["kind"])
                    for item in tactics
                    if item["mode"] in {"hard", "qualification_gate"}
                }
                if resources["generationLane"] != "ortools":
                    raise ValueError(
                        "tactical deal balance requires the OR-Tools generation lane"
                    )
                if not gated_kinds.intersection(
                    _TACTICAL_DEAL_BALANCE_GATED_TACTICS
                ):
                    raise ValueError(
                        "tactical deal balance requires an only-winning-move "
                        "or small-card-big-gain gate"
                    )
        if resources["generationLane"] not in {"ortools", "organic_runtime"}:
            raise ValueError("research generation lane is invalid")
        if (
            resources.get("auctionConstructionMode")
            in {"actor_safe_agent_verified", "cp_sat_actor_safe_decision_v1"}
            and resources["generationLane"] != "ortools"
        ):
            raise ValueError(
                "actor-safe auction construction requires the OR-Tools lane"
            )
        if (
            resources.get("auctionConstructionMode")
            in {"actor_safe_agent_verified", "cp_sat_actor_safe_decision_v1"}
            and resources.get("playConstructionMode", "cp_sat_feasibility_line")
            == "cp_sat_feasibility_line"
        ):
            raise ValueError(
                "actor-safe auction construction requires an actor-safe play line"
            )
        if (
            resources.get("auctionConstructionMode")
            in {"actor_safe_agent_verified", "cp_sat_actor_safe_decision_v1"}
            and situation["escalations"]
        ):
            raise ValueError(
                "actor-safe auction construction does not yet support required escalation"
            )
        if resources["seedStart"] < 0 or not 1 <= resources["seedCount"] <= 1_000_000_000:
            raise ValueError("research seed range is invalid")
        if not 1 <= resources["seedsPerShard"] <= 1_000_000:
            raise ValueError("research shard size is invalid")
        if not 1 <= resources["candidatesPerSeed"] <= 100:
            raise ValueError("research candidate count is invalid")
        if not 0.1 <= resources["solverTimeoutSeconds"] <= 900:
            raise ValueError("research solver timeout is invalid")
        if not 1 <= resources["maximumConflicts"] <= 10_000_000:
            raise ValueError("research conflict budget is invalid")
        if resources["analysisMode"] not in {"screening", "cascaded", "deep"}:
            raise ValueError("research analysis mode is invalid")
        if resources["deepVerificationPolicy"] not in _DEEP_VERIFICATION_POLICIES:
            raise ValueError("research deep-verification policy is invalid")
        if (
            resources["deepVerificationPolicy"] == "off"
            and any(item["mode"] in {"hard", "qualification_gate"} for item in tactics)
        ):
            raise ValueError(
                "hard tactical requirements cannot disable deep verification"
            )
        status = str(raw.get("status") or "draft")
        if status not in {"draft", "published", "retired"}:
            raise ValueError("research recipe status is invalid")
        core = {
            "schemaVersion": 1,
            "code": code,
            "titleAr": title_ar,
            "output": output,
            "profiles": profiles,
            "situation": situation,
            "tactics": tactics,
            "qualification": qualification,
            "resources": resources,
            "status": status,
        }
        digest = canonical_sha256(core)
        if not code:
            code = f"RCP-{digest[:10].upper()}"
            core["code"] = code
            digest = canonical_sha256(core)
        declared = str(raw.get("sha256") or "")
        if declared and declared != digest:
            raise ValueError("research recipe checksum mismatch")
        return cls(code, title_ar, output, profiles, situation, tuple(tactics), qualification, resources, status, digest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "code": self.code,
            "titleAr": self.title_ar,
            "output": copy.deepcopy(dict(self.output)),
            "profiles": copy.deepcopy(dict(self.profiles)),
            "situation": copy.deepcopy(dict(self.situation)),
            "tactics": [copy.deepcopy(dict(item)) for item in self.tactics],
            "qualification": copy.deepcopy(dict(self.qualification)),
            "resources": copy.deepcopy(dict(self.resources)),
            "status": self.status,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ResearchRecipeV2:
    """Current recipe contract with explicit diversity and render policies.

    V1 remains readable through :func:`build_research_recipe`; new callers can
    freeze editorial diversity and the on-demand render boundary without
    hiding either policy inside worker defaults.
    """

    code: str
    title_ar: str
    output: Mapping[str, Any]
    profiles: Mapping[str, Any]
    situation: Mapping[str, Any]
    tactics: tuple[Mapping[str, Any], ...]
    qualification: Mapping[str, Any]
    resources: Mapping[str, Any]
    diversity: Mapping[str, Any]
    render_policy: Mapping[str, Any]
    status: str
    sha256: str

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "ResearchRecipeV2":
        raw = _closed(
            value,
            {
                "schemaVersion", "code", "titleAr", "output", "profiles",
                "situation", "tactics", "qualification", "resources",
                "diversity", "renderPolicy", "status", "sha256",
            },
            "research recipe v2",
        )
        if int(raw.get("schemaVersion", 0)) != 2:
            raise ValueError("unsupported research recipe schema version")
        supplied_code = str(raw.get("code") or "").strip().upper()
        common_payload = {
            key: copy.deepcopy(raw[key])
            for key in (
                "code", "titleAr", "output", "profiles", "situation",
                "tactics", "qualification", "resources", "status",
            )
            if key in raw
        }
        # V1 is used only to validate the common contract.  A generated V2
        # code must include the V2-only policies in its identity, so prevent
        # the V1 validator from creating the final code prematurely.
        if not supplied_code:
            common_payload["code"] = "RCP-VALIDATE"
        common_payload["schemaVersion"] = 1
        common = ResearchRecipeV1.build(common_payload)
        if any(item.get("mode") != "hard" for item in common.situation["projects"]):
            raise ValueError(
                "ResearchRecipeV2 project constraints are hard constraints; "
                "soft project targeting requires a governed tactical plugin"
            )

        diversity_raw = _closed(
            raw.get("diversity") or {},
            {
                "schemaVersion", "cellDimensions", "topKPerCell",
                "rejectedAuditModulus", "sha256",
            },
            "research diversity policy",
        )
        from baloot_platform.challenge.quality_diversity import (
            QualityDiversityProfileV1,
        )

        declared_diversity_hash = str(diversity_raw.pop("sha256", ""))
        cell_dimensions = (
            diversity_raw["cellDimensions"]
            if "cellDimensions" in diversity_raw
            else (
                "contract", "tactical_detector", "acting_seat", "source",
            )
        )
        diversity_profile = QualityDiversityProfileV1.build({
            "schemaVersion": int(diversity_raw.get("schemaVersion", 1)),
            "cellDimensions": cell_dimensions,
            "topKPerCell": int(diversity_raw.get("topKPerCell", 3)),
            "rejectedAuditModulus": int(
                diversity_raw.get("rejectedAuditModulus", 100)
            ),
        })
        if declared_diversity_hash and declared_diversity_hash != diversity_profile.sha256:
            raise ValueError("research diversity profile checksum mismatch")
        diversity = diversity_profile.to_dict()

        render_raw = _closed(
            raw.get("renderPolicy") or {},
            {"mode"},
            "research render policy",
        )
        render_policy = {"mode": str(render_raw.get("mode") or "on_demand")}
        if render_policy["mode"] != "on_demand":
            raise ValueError("research rendering is currently on-demand only")

        core = {
            "schemaVersion": 2,
            "code": supplied_code,
            "titleAr": common.title_ar,
            "output": copy.deepcopy(dict(common.output)),
            "profiles": copy.deepcopy(dict(common.profiles)),
            "situation": copy.deepcopy(dict(common.situation)),
            "tactics": [copy.deepcopy(dict(item)) for item in common.tactics],
            "qualification": copy.deepcopy(dict(common.qualification)),
            "resources": copy.deepcopy(dict(common.resources)),
            "diversity": diversity,
            "renderPolicy": render_policy,
            "status": common.status,
        }
        digest = canonical_sha256(core)
        code = supplied_code
        if not code:
            code = f"RCP-{digest[:10].upper()}"
            core["code"] = code
            digest = canonical_sha256(core)
        declared = str(raw.get("sha256") or "")
        if declared and declared != digest:
            raise ValueError("research recipe checksum mismatch")
        return cls(
            code,
            common.title_ar,
            common.output,
            common.profiles,
            common.situation,
            common.tactics,
            common.qualification,
            common.resources,
            diversity,
            render_policy,
            common.status,
            digest,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 2,
            "code": self.code,
            "titleAr": self.title_ar,
            "output": copy.deepcopy(dict(self.output)),
            "profiles": copy.deepcopy(dict(self.profiles)),
            "situation": copy.deepcopy(dict(self.situation)),
            "tactics": [copy.deepcopy(dict(item)) for item in self.tactics],
            "qualification": copy.deepcopy(dict(self.qualification)),
            "resources": copy.deepcopy(dict(self.resources)),
            "diversity": copy.deepcopy(dict(self.diversity)),
            "renderPolicy": copy.deepcopy(dict(self.render_policy)),
            "status": self.status,
            "sha256": self.sha256,
        }


def build_research_configuration_selector_v1(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("research configuration selector must be an object")
    value = _closed(value, {"schemaVersion", "bundleRevisionId"}, "research configuration selector")
    if set(value) != {"schemaVersion", "bundleRevisionId"}:
        raise ValueError("research configuration selector fields are invalid")
    if type(value["schemaVersion"]) is not int or value["schemaVersion"] != 1:
        raise ValueError("research configuration selector schemaVersion must be integer 1")
    revision_id = value["bundleRevisionId"]
    if type(revision_id) is not str or not 1 <= len(revision_id) <= 200:
        raise ValueError("research configuration selector bundleRevisionId is invalid")
    return dict(value)


@dataclass(frozen=True, slots=True)
class ResearchConfigurationAuthorityV1:
    """Frozen Research projection of one EffectiveRuntimeConfigV2 artifact.

    The artifact is the configuration snapshot authority.  The remaining
    fields are closed, hash-bound query projections which Phase 2 can verify
    without allowing a client to assemble an ungoverned profile bundle.
    """

    bundle_revision_id: str
    bundle_id: str
    bundle_version: str
    semantic_bundle_hash: str
    effective_config_hash: str
    effective_runtime_config_hash: str
    snapshot_contract: str
    snapshot_artifact: Mapping[str, Any]
    component_revisions: Mapping[str, Mapping[str, Any]]
    identities: Mapping[str, Mapping[str, Any]]
    sha256: str

    @staticmethod
    def _component_revisions(
        value: Mapping[str, Any],
    ) -> dict[str, dict[str, str]]:
        if not isinstance(value, Mapping) or not value:
            raise ValueError(
                "research configuration requires component revision identities"
            )
        revisions: dict[str, dict[str, str]] = {}
        for family, raw_value in sorted(value.items()):
            family_name = str(family).strip()
            if not family_name or not isinstance(raw_value, Mapping):
                raise ValueError("research component revision identity is invalid")
            revision = _closed(
                raw_value,
                {"revisionId", "profileId", "version", "payloadSha256"},
                f"research component revision {family_name}",
            )
            normalized = {
                "revisionId": str(revision.get("revisionId") or "").strip(),
                "profileId": str(revision.get("profileId") or "").strip(),
                "version": str(revision.get("version") or "").strip(),
                "payloadSha256": _hash(
                    str(revision.get("payloadSha256") or ""),
                    f"componentRevisions.{family_name}.payloadSha256",
                ),
            }
            if not all(
                normalized[key] for key in ("revisionId", "profileId", "version")
            ):
                raise ValueError("research component revision identity is incomplete")
            revisions[family_name] = normalized
        required_families = set().union(*_RESEARCH_IDENTITY_SOURCE_FAMILIES.values())
        missing = required_families - set(revisions)
        if missing:
            raise ValueError(
                "research configuration is missing component revisions: "
                + ", ".join(sorted(missing))
            )
        return revisions

    @staticmethod
    def _snapshot_artifact(value: Mapping[str, Any]) -> dict[str, Any]:
        artifact = _closed(
            value,
            {"uri", "sha256", "bytes", "mediaType", "name"},
            "effective runtime configuration snapshot artifact",
        )
        artifact_hash = _hash(
            str(artifact.get("sha256") or ""), "snapshotArtifact.sha256"
        )
        uri = str(artifact.get("uri") or "").strip()
        parsed = urlparse(uri)
        is_local_cas = (
            parsed.scheme == "cas"
            and parsed.netloc == "sha256"
            and parsed.path == f"/{artifact_hash}"
        )
        is_s3_cas = (
            parsed.scheme == "s3"
            and bool(parsed.netloc)
            and parsed.path not in {"", "/"}
            and parsed.path.count("/") >= 2
            and parsed.path.rsplit("/", 1)[-1] == artifact_hash
        )
        if (
            parsed.params
            or parsed.query
            or parsed.fragment
            or not (is_local_cas or is_s3_cas)
        ):
            raise ValueError(
                "effective runtime configuration snapshot reference does not match its hash"
            )
        byte_count = int(artifact.get("bytes", 0))
        if byte_count < 1:
            raise ValueError(
                "effective runtime configuration snapshot must contain bytes"
            )
        media_type = str(artifact.get("mediaType") or "").strip()
        if media_type != _EFFECTIVE_RUNTIME_CONFIG_MEDIA_TYPE:
            raise ValueError(
                "effective runtime configuration snapshot media type is invalid"
            )
        name = artifact.get("name")
        if name is not None and not str(name).strip():
            raise ValueError(
                "effective runtime configuration snapshot name cannot be empty"
            )
        return {
            "uri": uri,
            "sha256": artifact_hash,
            "bytes": byte_count,
            "mediaType": media_type,
            "name": str(name).strip() if name is not None else None,
        }

    @staticmethod
    def _identities(
        value: Mapping[str, Any],
        component_revisions: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        if not isinstance(value, Mapping):
            raise ValueError("research configuration identities must be an object")
        unknown = set(value) - _RESEARCH_CONFIGURATION_IDENTITIES
        missing = _RESEARCH_CONFIGURATION_IDENTITIES - set(value)
        if unknown or missing:
            details = []
            if unknown:
                details.append("unknown=" + ",".join(sorted(unknown)))
            if missing:
                details.append("missing=" + ",".join(sorted(missing)))
            raise ValueError(
                "research configuration identity set is invalid: " + "; ".join(details)
            )
        identities: dict[str, dict[str, Any]] = {}
        for role in sorted(_RESEARCH_CONFIGURATION_IDENTITIES):
            raw_identity = value[role]
            if not isinstance(raw_identity, Mapping):
                raise ValueError(f"research {role} identity must be an object")
            identity = _closed(
                raw_identity,
                {
                    "kind",
                    "id",
                    "version",
                    "sha256",
                    "sourceRevisions",
                },
                f"research {role} identity",
            )
            kind = str(identity.get("kind") or "").strip()
            identity_id = str(identity.get("id") or "").strip()
            version = str(identity.get("version") or "").strip()
            if kind != role:
                raise ValueError(f"research {role} identity kind mismatch")
            if not identity_id or not version:
                raise ValueError(f"research {role} identity is incomplete")
            normalized: dict[str, Any] = {
                "kind": role,
                "id": identity_id,
                "version": version,
                "sha256": _hash(
                    str(identity.get("sha256") or ""),
                    f"identities.{role}.sha256",
                ),
            }
            raw_sources = identity.get("sourceRevisions")
            if not isinstance(raw_sources, Mapping):
                raise ValueError(
                    f"research {role} identity requires source revisions"
                )
            expected_families = _RESEARCH_IDENTITY_SOURCE_FAMILIES[role]
            if set(raw_sources) != expected_families:
                raise ValueError(
                    f"research {role} identity source revisions are incomplete"
                )
            source_revisions: dict[str, str] = {}
            for family in sorted(expected_families):
                expected_revision = str(
                    component_revisions[family]["revisionId"]
                )
                if str(raw_sources.get(family) or "") != expected_revision:
                    raise ValueError(
                        f"research {role} identity source revision mismatch"
                    )
                source_revisions[family] = expected_revision
            normalized["sourceRevisions"] = source_revisions
            identities[role] = normalized
        return identities

    @classmethod
    def build(
        cls, value: Mapping[str, Any]
    ) -> "ResearchConfigurationAuthorityV1":
        raw = _closed(
            value,
            {
                "schemaVersion",
                "bundleRevisionId",
                "bundleId",
                "bundleVersion",
                "semanticBundleHash",
                "effectiveConfigHash",
                "effectiveRuntimeConfigHash",
                "snapshotContract",
                "snapshotArtifact",
                "componentRevisions",
                "identities",
                "sha256",
            },
            "research configuration authority",
        )
        if int(raw.get("schemaVersion", 0)) != 1:
            raise ValueError("unsupported research configuration authority version")
        bundle_revision_id = str(raw.get("bundleRevisionId") or "").strip()
        bundle_id = str(raw.get("bundleId") or "").strip()
        bundle_version = str(raw.get("bundleVersion") or "").strip()
        if not bundle_revision_id or not bundle_id or not bundle_version:
            raise ValueError("research configuration bundle identity is incomplete")
        if not re.fullmatch(
            r"[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?",
            bundle_version,
        ):
            raise ValueError("research configuration bundle version is invalid")
        semantic_bundle_hash = _hash(
            str(raw.get("semanticBundleHash") or ""), "semanticBundleHash"
        )
        effective_config_hash = _hash(
            str(raw.get("effectiveConfigHash") or ""), "effectiveConfigHash"
        )
        effective_runtime_config_hash = _hash(
            str(raw.get("effectiveRuntimeConfigHash") or ""),
            "effectiveRuntimeConfigHash",
        )
        snapshot_contract = str(raw.get("snapshotContract") or "").strip()
        if snapshot_contract != "EffectiveRuntimeConfigV2":
            raise ValueError(
                "research configuration snapshot contract must be EffectiveRuntimeConfigV2"
            )
        snapshot_artifact = cls._snapshot_artifact(
            raw.get("snapshotArtifact") or {}
        )
        component_revisions = cls._component_revisions(
            raw.get("componentRevisions") or {}
        )
        identities = cls._identities(
            raw.get("identities") or {}, component_revisions
        )
        core = {
            "schemaVersion": 1,
            "bundleRevisionId": bundle_revision_id,
            "bundleId": bundle_id,
            "bundleVersion": bundle_version,
            "semanticBundleHash": semantic_bundle_hash,
            "effectiveConfigHash": effective_config_hash,
            "effectiveRuntimeConfigHash": effective_runtime_config_hash,
            "snapshotContract": snapshot_contract,
            "snapshotArtifact": snapshot_artifact,
            "componentRevisions": component_revisions,
            "identities": identities,
        }
        digest = canonical_sha256(core)
        declared = str(raw.get("sha256") or "")
        if declared and declared != digest:
            raise ValueError("research configuration authority checksum mismatch")
        return cls(
            bundle_revision_id,
            bundle_id,
            bundle_version,
            semantic_bundle_hash,
            effective_config_hash,
            effective_runtime_config_hash,
            snapshot_contract,
            snapshot_artifact,
            component_revisions,
            identities,
            digest,
        )

    @classmethod
    def freeze(
        cls,
        *,
        effective_runtime_config: "EffectiveRuntimeConfigV2",
        snapshot_artifact: Mapping[str, Any],
        semantic_bundle_hash: str,
        identities: Mapping[str, Any],
    ) -> "ResearchConfigurationAuthorityV1":
        """Derive the canonical authority from the existing V2 snapshot."""
        if not effective_runtime_config.verify():
            raise ValueError("EffectiveRuntimeConfigV2 checksum verification failed")
        snapshot = effective_runtime_config.to_dict()
        if (
            int(snapshot.get("schemaVersion", 0)) != 2
            or snapshot.get("contractVersion") != "2.0.0"
        ):
            raise ValueError("research requires EffectiveRuntimeConfigV2")
        snapshot_bytes = canonical_json(snapshot).encode("utf-8")
        expected_artifact_hash = hashlib.sha256(snapshot_bytes).hexdigest()
        supplied_artifact = dict(snapshot_artifact)
        if (
            supplied_artifact.get("sha256") != expected_artifact_hash
            or int(supplied_artifact.get("bytes", -1)) != len(snapshot_bytes)
        ):
            raise ValueError(
                "EffectiveRuntimeConfigV2 snapshot artifact does not match canonical bytes"
            )
        bundle = dict(snapshot["bundle"])
        return cls.build(
            {
                "schemaVersion": 1,
                "bundleRevisionId": bundle["revisionId"],
                "bundleId": bundle["bundleId"],
                "bundleVersion": bundle["version"],
                "semanticBundleHash": semantic_bundle_hash,
                "effectiveConfigHash": snapshot["sourceEffectiveHash"],
                "effectiveRuntimeConfigHash": snapshot["contentSha256"],
                "snapshotContract": "EffectiveRuntimeConfigV2",
                "snapshotArtifact": supplied_artifact,
                "componentRevisions": snapshot["componentRevisions"],
                "identities": identities,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "bundleRevisionId": self.bundle_revision_id,
            "bundleId": self.bundle_id,
            "bundleVersion": self.bundle_version,
            "semanticBundleHash": self.semantic_bundle_hash,
            "effectiveConfigHash": self.effective_config_hash,
            "effectiveRuntimeConfigHash": self.effective_runtime_config_hash,
            "snapshotContract": self.snapshot_contract,
            "snapshotArtifact": copy.deepcopy(dict(self.snapshot_artifact)),
            "componentRevisions": {
                family: copy.deepcopy(dict(identity))
                for family, identity in self.component_revisions.items()
            },
            "identities": {
                role: copy.deepcopy(dict(identity))
                for role, identity in self.identities.items()
            },
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ResearchRecipeV3(ResearchRecipeV2):
    """Research recipe with one server-frozen configuration authority."""

    configuration: Mapping[str, Any]

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "ResearchRecipeV3":
        raw = _closed(
            value,
            {
                "schemaVersion",
                "code",
                "titleAr",
                "output",
                "profiles",
                "situation",
                "tactics",
                "qualification",
                "resources",
                "diversity",
                "renderPolicy",
                "configuration",
                "status",
                "sha256",
            },
            "research recipe v3",
        )
        if int(raw.get("schemaVersion", 0)) != 3:
            raise ValueError("unsupported research recipe schema version")
        _validate_v3_research_field_types(raw)
        supplied_code = str(raw.get("code") or "").strip().upper()
        common_payload = {
            key: copy.deepcopy(raw[key])
            for key in (
                "code",
                "titleAr",
                "output",
                "profiles",
                "situation",
                "tactics",
                "qualification",
                "resources",
                "diversity",
                "renderPolicy",
                "status",
            )
            if key in raw
        }
        if not supplied_code:
            common_payload["code"] = "RCP-VALIDATE"
        if "tactics" not in common_payload:
            common_payload["tactics"] = copy.deepcopy(
                _RESEARCH_RECIPE_FIELD_DEFAULTS["tactics"]
            )
        situation_payload = copy.deepcopy(
            dict(common_payload.get("situation") or {})
        )
        situation_payload.setdefault(
            "contracts",
            copy.deepcopy(
                _RESEARCH_RECIPE_FIELD_DEFAULTS["situation.contracts"]
            ),
        )
        if situation_payload.get("escalations") == ["normal"]:
            situation_payload["escalations"] = []
        projects_payload = []
        for project in situation_payload.get("projects") or ():
            normalized_project = copy.deepcopy(dict(project))
            normalized_project.setdefault(
                "mode",
                _RESEARCH_RECIPE_FIELD_DEFAULTS["situation.projects[].mode"],
            )
            projects_payload.append(normalized_project)
        if "projects" in situation_payload:
            situation_payload["projects"] = projects_payload
        common_payload["situation"] = situation_payload

        resource_payload = copy.deepcopy(
            dict(common_payload.get("resources") or {})
        )
        generation_lane = str(
            resource_payload.get("generationLane")
            or _RESEARCH_RECIPE_FIELD_DEFAULTS["resources.generationLane"]
        )
        if generation_lane == "ortools":
            resource_payload.setdefault(
                "auctionConstructionMode",
                _RESEARCH_RECIPE_FIELD_DEFAULTS[
                    "resources.auctionConstructionMode"
                ],
            )
            resource_payload.setdefault(
                "playConstructionMode",
                _RESEARCH_RECIPE_FIELD_DEFAULTS[
                    "resources.playConstructionMode"
                ],
            )
        else:
            ortools_only = {
                "candidatesPerSeed",
                "solverTimeoutSeconds",
                "maximumConflicts",
            }
            if ortools_only.intersection(resource_payload):
                raise ValueError(
                    "research relationship violation: "
                    "organic_forbids_ortools_solver_controls"
                )
        common_payload["resources"] = resource_payload
        common_payload.setdefault(
            "status", _RESEARCH_RECIPE_FIELD_DEFAULTS["status"]
        )
        common_payload["schemaVersion"] = 2
        common = ResearchRecipeV2.build(common_payload)
        if (
            common.qualification["requireObserverSafety"] is not True
            or common.qualification["requireAllDecisionAudit"] is not True
        ):
            raise ValueError(
                "V3 Research requires observer safety and complete decision auditing"
            )
        normalized_situation = copy.deepcopy(dict(common.situation))
        normalized_situation["projects"] = [
            {
                "seat": str(project["seat"]),
                "contract": str(project["contract"]),
                "kind": str(project["kind"]),
                "value": int(project["value"]),
                "highRank": str(project["highRank"]),
                "mode": str(project.get("mode", "hard")),
            }
            for project in common.situation["projects"]
        ]
        normalized_resources = copy.deepcopy(dict(common.resources))
        if normalized_resources["generationLane"] == "organic_runtime":
            for key in (
                "candidatesPerSeed",
                "solverTimeoutSeconds",
                "maximumConflicts",
            ):
                normalized_resources.pop(key, None)
        if common.output["kind"] not in {"laffa", "laffa_collection"}:
            raise ValueError(
                "V3 Research directly supports only laffa and laffa_collection"
            )
        _validate_declared_research_relationships(
            {
                "schemaVersion": 3,
                "output": copy.deepcopy(dict(common.output)),
                "situation": normalized_situation,
                "tactics": [
                    copy.deepcopy(dict(item)) for item in common.tactics
                ],
                "qualification": copy.deepcopy(dict(common.qualification)),
                "resources": copy.deepcopy(dict(normalized_resources)),
                "diversity": copy.deepcopy(dict(common.diversity)),
                "renderPolicy": copy.deepcopy(dict(common.render_policy)),
                "status": common.status,
            }
        )
        authority = ResearchConfigurationAuthorityV1.build(
            raw.get("configuration") or {}
        )
        profiles = dict(common.profiles)
        identities = authority.identities
        expected_profiles = {
            "domainHash": identities["domain"]["sha256"],
            "bundleHash": authority.semantic_bundle_hash,
            "rulesHash": identities["rules"]["sha256"],
            "conventionHash": identities["convention"]["sha256"],
            "agentRelease": identities["agent"]["id"],
            "analysisProfileHash": identities["analyzer"]["sha256"],
            "scoreProfileHash": identities["score"]["sha256"],
        }
        mismatched = sorted(
            key for key, expected in expected_profiles.items()
            if profiles.get(key) != expected
        )
        if mismatched:
            raise ValueError(
                "research recipe profiles mismatch frozen configuration identities: "
                + ", ".join(mismatched)
            )
        core = {
            "schemaVersion": 3,
            "code": supplied_code,
            "titleAr": common.title_ar,
            "output": copy.deepcopy(dict(common.output)),
            "profiles": copy.deepcopy(dict(common.profiles)),
            "situation": normalized_situation,
            "tactics": [copy.deepcopy(dict(item)) for item in common.tactics],
            "qualification": copy.deepcopy(dict(common.qualification)),
            "resources": normalized_resources,
            "diversity": copy.deepcopy(dict(common.diversity)),
            "renderPolicy": copy.deepcopy(dict(common.render_policy)),
            "configuration": authority.to_dict(),
            "status": common.status,
        }
        digest = canonical_sha256(core)
        code = supplied_code
        if not code:
            code = f"RCP-{digest[:10].upper()}"
            core["code"] = code
            digest = canonical_sha256(core)
        declared = str(raw.get("sha256") or "")
        if declared and declared != digest:
            raise ValueError("research recipe checksum mismatch")
        return cls(
            code,
            common.title_ar,
            common.output,
            common.profiles,
            normalized_situation,
            common.tactics,
            common.qualification,
            normalized_resources,
            common.diversity,
            common.render_policy,
            common.status,
            digest,
            authority.to_dict(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 3,
            "code": self.code,
            "titleAr": self.title_ar,
            "output": copy.deepcopy(dict(self.output)),
            "profiles": copy.deepcopy(dict(self.profiles)),
            "situation": copy.deepcopy(dict(self.situation)),
            "tactics": [copy.deepcopy(dict(item)) for item in self.tactics],
            "qualification": copy.deepcopy(dict(self.qualification)),
            "resources": copy.deepcopy(dict(self.resources)),
            "diversity": copy.deepcopy(dict(self.diversity)),
            "renderPolicy": copy.deepcopy(dict(self.render_policy)),
            "configuration": copy.deepcopy(dict(self.configuration)),
            "status": self.status,
            "sha256": self.sha256,
        }


def build_research_recipe(
    value: Mapping[str, Any],
) -> ResearchRecipeV1 | ResearchRecipeV2 | ResearchRecipeV3 | Any:
    version = int(dict(value).get("schemaVersion", 1))
    if version == 1:
        return ResearchRecipeV1.build(value)
    if version == 2:
        return ResearchRecipeV2.build(value)
    if version == 3:
        return ResearchRecipeV3.build(value)
    if version == 4:
        # Local import avoids making the V1-V3 contract module depend on the
        # V4 capability compiler during legacy read-only hydration.
        from .recipe_v4 import ResearchRecipeV4

        return ResearchRecipeV4.build(value)
    raise ValueError("unsupported research recipe schema version")


def build_legacy_research_recipe_for_read(
    value: Mapping[str, Any],
) -> ResearchRecipeV1 | ResearchRecipeV2:
    """Hydrate immutable V1/V2 records without applying current V3 gates.

    This adapter is intentionally read-only.  New validation, estimation, and
    execution must use a V3 recipe bound to one frozen bundle authority.
    """

    version = int(dict(value).get("schemaVersion", 1))
    if version == 1:
        return ResearchRecipeV1.build(value)
    if version == 2:
        return ResearchRecipeV2.build(value)
    raise ValueError("legacy Research read adapter supports only V1 and V2")


@dataclass(frozen=True, slots=True)
class ResearchEstimateV1:
    seed_count: int
    candidate_count: int
    shard_count: int
    maximum_solver_seconds: float
    analysis_decision_upper_bound: int
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "seedCount": self.seed_count,
            "candidateCount": self.candidate_count,
            "shardCount": self.shard_count,
            "maximumSolverSeconds": self.maximum_solver_seconds,
            "analysisDecisionUpperBound": self.analysis_decision_upper_bound,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class SemanticDisagreementV1:
    disagreement_id: str
    run_code: str
    candidate_hash: str
    category: str
    solver_evidence: Mapping[str, Any]
    engine_evidence: Mapping[str, Any]
    status: str
    sha256: str

    @classmethod
    def build(
        cls,
        *,
        run_code: str,
        candidate_hash: str,
        category: str,
        solver_evidence: Mapping[str, Any],
        engine_evidence: Mapping[str, Any],
    ) -> "SemanticDisagreementV1":
        if category not in {"constraint_mismatch", "legality_mismatch", "score_mismatch", "optimality_mismatch", "provenance_mismatch"}:
            raise ValueError("unsupported semantic disagreement category")
        core = {
            "schemaVersion": 1,
            "runCode": str(run_code),
            "candidateHash": _hash(candidate_hash, "candidateHash"),
            "category": category,
            "solverEvidence": copy.deepcopy(dict(solver_evidence)),
            "engineEvidence": copy.deepcopy(dict(engine_evidence)),
            "status": "open",
        }
        digest = canonical_sha256(core)
        return cls(f"DSG-{digest[:12].upper()}", str(run_code), core["candidateHash"], category, core["solverEvidence"], core["engineEvidence"], "open", digest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "disagreementId": self.disagreement_id,
            "runCode": self.run_code,
            "candidateHash": self.candidate_hash,
            "category": self.category,
            "solverEvidence": copy.deepcopy(dict(self.solver_evidence)),
            "engineEvidence": copy.deepcopy(dict(self.engine_evidence)),
            "status": self.status,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class SemanticDisagreementV2:
    disagreement_id: str
    run_code: str
    candidate_hash: str
    category: str
    solver_evidence: Mapping[str, Any]
    runtime_evidence: Mapping[str, Any]
    status: str
    sha256: str

    @classmethod
    def build(
        cls,
        *,
        run_code: str,
        candidate_hash: str,
        category: str,
        solver_evidence: Mapping[str, Any],
        runtime_evidence: Mapping[str, Any],
    ) -> "SemanticDisagreementV2":
        semantic_categories = {
            "compiler_parity_defect",
            "runtime_transition_defect",
            "hard_constraint_violation",
            "observer_information_defect",
        }
        if category not in semantic_categories:
            raise ValueError("unsupported V2 semantic discrepancy category")
        core = {
            "schemaVersion": 2,
            "runCode": str(run_code),
            "candidateHash": _hash(candidate_hash, "candidateHash"),
            "category": category,
            "semanticDefect": True,
            "solverEvidence": copy.deepcopy(dict(solver_evidence)),
            "runtimeEvidence": copy.deepcopy(dict(runtime_evidence)),
            "status": "open",
        }
        digest = canonical_sha256(core)
        return cls(
            f"DSG-{digest[:12].upper()}", str(run_code), core["candidateHash"],
            category, core["solverEvidence"], core["runtimeEvidence"], "open", digest,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 2,
            "disagreementId": self.disagreement_id,
            "runCode": self.run_code,
            "candidateHash": self.candidate_hash,
            "category": self.category,
            "semanticDefect": True,
            "solverEvidence": copy.deepcopy(dict(self.solver_evidence)),
            "runtimeEvidence": copy.deepcopy(dict(self.runtime_evidence)),
            "status": self.status,
            "sha256": self.sha256,
        }


def truth_rank(value: str) -> int:
    return _TRUTH_LEVELS[str(value)]
