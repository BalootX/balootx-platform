"""Scenario predicate capability registry and deterministic V4 compiler."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

from baloot_platform.tactics import TacticalKind, TACTICAL_LOCALIZATION_KEYS

from .scenario_contracts import (
    CONSTRUCTION_CAPABILITIES,
    PREDICATE_HARDNESS,
    PROOF_MODES,
    SCENARIO_SECTIONS,
    ResearchScenarioV1,
    ScenarioDiagnosticV1,
    ScenarioExpressionV1,
    ScenarioPredicateV1,
    ScenarioProofRequestV1,
    canonical_sha256,
)


SCENARIO_RULE_APPLICABILITY_CODES = (
    "contract_not_enabled_by_rules",
    "ashkal_not_enabled_by_rules",
    "trump_not_applicable_to_contract",
    "auction_action_not_enabled_by_rules",
    "escalation_level_not_enabled_by_rules",
    "escalation_action_not_enabled_by_rules",
    "escalation_not_enabled_by_rules",
    "projects_not_enabled_by_rules",
    "project_kind_value_not_enabled_by_rules",
    "baloot_not_enabled_for_contract",
    "kabout_not_enabled_by_rules",
    "gahwa_not_enabled_for_contract",
    "tactic_detector_not_enabled_by_bundle",
    "ekka_not_enabled_by_rules",
)


@dataclass(frozen=True, slots=True)
class ScenarioPredicateDefinitionV1:
    kind: str
    version: int
    category: str
    stage: str
    argument_schema: Mapping[str, Any]
    message_ar_key: str
    message_en_key: str
    required_domain_primitives: tuple[str, ...]
    allowed_hardness: tuple[str, ...]
    construction_capability: str
    runtime_verifier: str
    supported_proof_modes: tuple[str, ...]
    hidden_information: str
    cost_class: str
    safe_complement: bool
    examples: tuple[Mapping[str, Any], ...]
    counterexamples: tuple[Mapping[str, Any], ...]
    documentation: tuple[str, ...]
    tests: tuple[str, ...]
    lifecycle_status: str
    sha256: str

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "ScenarioPredicateDefinitionV1":
        raw = copy.deepcopy(dict(value))
        required = {
            "kind",
            "version",
            "category",
            "stage",
            "argumentSchema",
            "messageArKey",
            "messageEnKey",
            "requiredDomainPrimitives",
            "allowedHardness",
            "constructionCapability",
            "runtimeVerifier",
            "supportedProofModes",
            "hiddenInformation",
            "costClass",
            "safeComplement",
            "examples",
            "counterexamples",
            "documentation",
            "tests",
            "lifecycleStatus",
        }
        unknown = sorted(set(raw) - {*required, "sha256"})
        missing = sorted(required - set(raw))
        if unknown or missing:
            raise ValueError(
                "scenario predicate definition shape mismatch; "
                f"missing={missing}, unknown={unknown}"
            )
        if type(raw["version"]) is not int or raw["version"] < 1:
            raise ValueError("scenario predicate definition version is invalid")
        if str(raw["constructionCapability"]) not in CONSTRUCTION_CAPABILITIES:
            raise ValueError("scenario predicate construction capability is invalid")
        hardness = tuple(str(item) for item in raw["allowedHardness"])
        if not hardness or not set(hardness).issubset(PREDICATE_HARDNESS):
            raise ValueError("scenario predicate allowed hardness is invalid")
        proof_modes = tuple(str(item) for item in raw["supportedProofModes"])
        if not set(proof_modes).issubset(PROOF_MODES):
            raise ValueError("scenario predicate proof modes are invalid")
        if str(raw["hiddenInformation"]) not in {
            "public",
            "actor_visible",
            "complete_state_only",
            "information_set_safe",
        }:
            raise ValueError("scenario predicate hidden-information class is invalid")
        if str(raw["costClass"]) not in {
            "constant",
            "linear",
            "solver",
            "tree_search",
            "hidden_world_search",
        }:
            raise ValueError("scenario predicate cost class is invalid")
        if type(raw["safeComplement"]) is not bool:
            raise TypeError("scenario predicate safeComplement must be boolean")
        for field in ("examples", "counterexamples"):
            if not isinstance(raw[field], list) or not raw[field]:
                raise ValueError(f"scenario predicate {field} must be non-empty")
        for field in ("documentation", "tests"):
            if not isinstance(raw[field], list) or not raw[field]:
                raise ValueError(f"scenario predicate {field} must be non-empty")
        if str(raw["lifecycleStatus"]) not in {"active", "deprecated", "retired"}:
            raise ValueError("scenario predicate lifecycle status is invalid")
        core = {key: raw[key] for key in sorted(required)}
        digest = canonical_sha256(core)
        declared = str(raw.get("sha256") or "")
        if declared and declared != digest:
            raise ValueError("scenario predicate definition checksum mismatch")
        return cls(
            str(raw["kind"]),
            int(raw["version"]),
            str(raw["category"]),
            str(raw["stage"]),
            copy.deepcopy(dict(raw["argumentSchema"])),
            str(raw["messageArKey"]),
            str(raw["messageEnKey"]),
            tuple(str(item) for item in raw["requiredDomainPrimitives"]),
            hardness,
            str(raw["constructionCapability"]),
            str(raw["runtimeVerifier"]),
            proof_modes,
            str(raw["hiddenInformation"]),
            str(raw["costClass"]),
            raw["safeComplement"],
            tuple(copy.deepcopy(dict(item)) for item in raw["examples"]),
            tuple(copy.deepcopy(dict(item)) for item in raw["counterexamples"]),
            tuple(str(item) for item in raw["documentation"]),
            tuple(str(item) for item in raw["tests"]),
            str(raw["lifecycleStatus"]),
            digest,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "kind": self.kind,
            "version": self.version,
            "category": self.category,
            "stage": self.stage,
            "argumentSchema": copy.deepcopy(dict(self.argument_schema)),
            "messageArKey": self.message_ar_key,
            "messageEnKey": self.message_en_key,
            "requiredDomainPrimitives": list(self.required_domain_primitives),
            "allowedHardness": list(self.allowed_hardness),
            "constructionCapability": self.construction_capability,
            "runtimeVerifier": self.runtime_verifier,
            "supportedProofModes": list(self.supported_proof_modes),
            "hiddenInformation": self.hidden_information,
            "costClass": self.cost_class,
            "safeComplement": self.safe_complement,
            "examples": [copy.deepcopy(dict(item)) for item in self.examples],
            "counterexamples": [
                copy.deepcopy(dict(item)) for item in self.counterexamples
            ],
            "documentation": list(self.documentation),
            "tests": list(self.tests),
            "lifecycleStatus": self.lifecycle_status,
            "sha256": self.sha256,
        }


def _schema(
    *, required: tuple[str, ...], properties: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(required),
        "properties": copy.deepcopy(dict(properties)),
    }


def _schema_example(schema: Mapping[str, Any]) -> Any:
    if "enum" in schema:
        return copy.deepcopy(schema["enum"][0])
    expected = schema.get("type")
    if expected == "object":
        properties = dict(schema.get("properties") or {})
        return {
            name: _schema_example(properties[name])
            for name in schema.get("required") or ()
        }
    if expected == "array":
        return [
            _schema_example(dict(schema.get("items") or {}))
            for _ in range(max(1, int(schema.get("minItems", 0))))
        ]
    if expected == "integer":
        return int(schema.get("minimum", 0))
    if expected == "number":
        return float(schema.get("minimum", 0.0))
    if expected == "boolean":
        return False
    if expected == "string":
        return "0" if schema.get("pattern") == "^[0-9]+$" else "example"
    raise ValueError("scenario predicate schema cannot produce a catalog example")


_SEAT = {"type": "string", "enum": ["north", "west", "south", "east"]}
_TEAM = {"type": "string", "enum": ["north_south", "west_east"]}
_CONTRACT = {"type": "string", "enum": ["sun", "hokum", "ashkal"]}
_SUIT = {"type": "string", "enum": ["spades", "hearts", "diamonds", "clubs"]}
_CARD = {"type": "integer", "minimum": 0, "maximum": 31}
_PLAY_ACTION = {
    "type": "object",
    "additionalProperties": False,
    "required": ["card", "ekka"],
    "properties": {"card": _CARD, "ekka": {"type": "boolean"}},
}
_NON_NEGATIVE = {"type": "integer", "minimum": 0}
_HAND_STAGE = {"type": "string", "enum": ["initial_five", "post_purchase"]}
_ARGUMENT_LABEL_KEYS = {
    "action": "research_scenario_argument_action",
    "achieved": "research_scenario_argument_achieved",
    "actions": "research_scenario_argument_actions",
    "actingSeat": "research_scenario_argument_actor",
    "actor": "research_scenario_argument_actor",
    "applicable": "research_scenario_argument_applicable",
    "card": "research_scenario_argument_card",
    "cards": "research_scenario_argument_cards",
    "causalRequired": "research_scenario_argument_causal_required",
    "classification": "research_scenario_argument_classification",
    "complete": "research_scenario_argument_complete",
    "contract": "research_scenario_argument_contract",
    "eligible": "research_scenario_argument_eligible",
    "ekka": "research_scenario_argument_ekka",
    "handStage": "research_scenario_argument_hand_stage",
    "kind": "research_scenario_argument_kind",
    "level": "research_scenario_argument_level",
    "lineUniverse": "research_scenario_argument_line_universe",
    "locked": "research_scenario_argument_locked",
    "maximum": "research_scenario_argument_maximum",
    "maximumDifference": "research_scenario_argument_maximum_difference",
    "minimum": "research_scenario_argument_minimum",
    "minimumEvidence": "research_scenario_argument_minimum_evidence",
    "minimumScore": "research_scenario_argument_minimum_score",
    "mode": "research_scenario_argument_mode",
    "observer": "research_scenario_argument_observer",
    "otherSeat": "research_scenario_argument_other_seat",
    "owner": "research_scenario_argument_owner",
    "position": "research_scenario_argument_position",
    "rank": "research_scenario_argument_rank",
    "relation": "research_scenario_argument_relation",
    "round": "research_scenario_argument_round",
    "seat": "research_scenario_argument_seat",
    "seats": "research_scenario_argument_seats",
    "state": "research_scenario_argument_state",
    "status": "research_scenario_argument_status",
    "suit": "research_scenario_argument_suit",
    "targetTeam": "research_scenario_argument_target_team",
    "targetSeat": "research_scenario_argument_target_seat",
    "team": "research_scenario_argument_team",
    "terminal": "research_scenario_argument_terminal",
    "trickIndex": "research_scenario_argument_trick_index",
    "trumpSuit": "research_scenario_argument_suit",
    "value": "research_scenario_argument_value",
    "visibility": "research_scenario_argument_visibility",
}
_ARGUMENT_ENUM_LABEL_KEYS = {
    **TACTICAL_LOCALIZATION_KEYS,
    "4": "research_scenario_option_project_value_4",
    "10": "research_scenario_option_project_value_10",
    "20": "research_scenario_option_project_value_20",
    "40": "research_scenario_option_project_value_40",
    "initial_five": "research_scenario_option_initial_five",
    "post_purchase": "research_scenario_option_post_purchase",
    "seven": "research_scenario_option_rank_seven",
    "eight": "research_scenario_option_rank_eight",
    "nine": "research_scenario_option_rank_nine",
    "jack": "research_scenario_option_rank_jack",
    "queen": "research_scenario_option_rank_queen",
    "king": "research_scenario_option_rank_king",
    "ten": "research_scenario_option_rank_ten",
    "ace": "research_scenario_option_rank_ace",
    "siri": "research_scenario_option_project_siri",
    "fifty": "research_scenario_option_project_fifty",
    "hundred_sequence": "research_scenario_option_project_hundred_sequence",
    "hundred_four_kind": "research_scenario_option_project_hundred_four_kind",
    "four_hundred": "research_scenario_option_project_four_hundred",
    "normal": "research_escalation_normal",
    "double": "research_escalation_double",
    "triple": "research_escalation_triple",
    "four": "research_escalation_four",
    "gahwa": "research_escalation_gahwa",
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
    "policy_conditioned": "research_scenario_option_lines_policy_conditioned",
    "same": "research_scenario_option_relation_same",
    "partner": "research_scenario_option_relation_partner",
    "next": "research_scenario_option_relation_next",
    "previous": "research_scenario_option_relation_previous",
    "opponent": "research_scenario_option_relation_opponent",
    "bidder": "research_scenario_option_relation_bidder",
    "bidder_partner": "research_scenario_option_relation_bidder_partner",
    "visible": "research_scenario_option_visibility_visible",
    "hidden": "research_scenario_option_visibility_hidden",
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
    "decline": "research_scenario_option_action_decline",
    "open": "research_scenario_option_mode_open",
    "locked": "research_scenario_option_mode_locked",
    "created": "research_scenario_option_phase_created",
    "shuffled": "research_scenario_option_phase_shuffled",
    "initial_dealt": "research_scenario_option_phase_initial_dealt",
    "bidding": "research_scenario_option_phase_bidding",
    "final_dealt": "research_scenario_option_phase_final_dealt",
    "projects": "research_scenario_option_phase_projects",
    "playing": "research_scenario_option_phase_playing",
    "scored": "research_scenario_option_phase_scored",
    "complete": "research_scenario_option_phase_complete",
}
_STAGE_PRIMITIVES = {
    "deal": ("rules.deck.cards_per_player",),
    "auction": ("rules.bidding.rounds",),
    "escalation": ("rules.escalation.enabled",),
    "projects": ("rules.projects.enabled",),
    "playPrefix": ("rules.play.must_follow_suit",),
    "outcome": ("rules.scoring.contract.buyer_team_must_win",),
}


def _definition(
    kind: str,
    stage: str,
    schema: Mapping[str, Any],
    capability: str,
    *,
    proof_modes: tuple[str, ...] = (),
    hidden: str = "public",
    cost: str = "constant",
    safe_complement: bool = True,
    hardness: tuple[str, ...] = ("hard", "prefer", "report"),
    required_primitives: tuple[str, ...] = (),
    message_key: str | None = None,
    lifecycle_status: str = "active",
) -> ScenarioPredicateDefinitionV1:
    slug = kind.replace(".", "_")

    def decorate(
        raw_schema: Mapping[str, Any],
        *,
        property_name: str | None = None,
    ) -> dict[str, Any]:
        property_schema = copy.deepcopy(dict(raw_schema))
        if property_name is not None:
            name = property_name
            key = _ARGUMENT_LABEL_KEYS[name]
            if name == "kind" and kind.startswith("tactic."):
                key = "research_scenario_argument_tactic_kind"
            elif name == "kind" and kind.startswith("project."):
                key = "research_scenario_argument_project_kind"
            property_schema["x-messageArKey"] = key
            property_schema["x-messageEnKey"] = f"{key}_en"
            enum_values = property_schema.get("enum")
            if (
                enum_values is None
                and property_schema.get("type") == "array"
                and isinstance(property_schema.get("items"), Mapping)
            ):
                enum_values = property_schema["items"].get("enum")
            if kind == "project.exists" and name == "kind":
                enum_values = [
                    "siri",
                    "fifty",
                    "hundred_sequence",
                    "hundred_four_kind",
                    "four_hundred",
                ]
            elif kind == "project.exists" and name == "value":
                enum_values = [4, 10, 20, 40]
            if isinstance(enum_values, list):
                enum_keys = {
                    str(value): _ARGUMENT_ENUM_LABEL_KEYS[str(value)]
                    for value in enum_values
                    if str(value) in _ARGUMENT_ENUM_LABEL_KEYS
                }
                if enum_keys:
                    property_schema["x-enumMessageArKeys"] = enum_keys
                    property_schema["x-enumMessageEnKeys"] = {
                        value: f"{label}_en" for value, label in enum_keys.items()
                    }
        if property_schema.get("type") == "object":
            property_schema["properties"] = {
                name: decorate(value, property_name=name)
                for name, value in dict(property_schema.get("properties") or {}).items()
            }
        elif property_schema.get("type") == "array" and isinstance(
            property_schema.get("items"), Mapping
        ):
            property_schema["items"] = decorate(property_schema["items"])
        return property_schema

    argument_schema = decorate(schema)
    example_arguments = _schema_example(argument_schema)
    if kind.startswith("tactic."):
        example_arguments["kind"] = "only_winning_move"
    return ScenarioPredicateDefinitionV1.build(
        {
            "kind": kind,
            "version": 1,
            "category": stage,
            "stage": stage,
            "argumentSchema": argument_schema,
            "messageArKey": message_key or f"research_scenario_{slug}",
            "messageEnKey": f"{message_key}_en"
            if message_key
            else f"research_scenario_{slug}_en",
            "requiredDomainPrimitives": list(
                required_primitives or _STAGE_PRIMITIVES.get(stage, ())
            ),
            "allowedHardness": list(hardness),
            "constructionCapability": capability,
            "runtimeVerifier": f"verify_{slug}_v1",
            "supportedProofModes": list(proof_modes),
            "hiddenInformation": hidden,
            "costClass": cost,
            "safeComplement": safe_complement,
            "examples": [
                {
                    "kind": kind,
                    "version": 1,
                    "hardness": hardness[0],
                    "arguments": example_arguments,
                }
            ],
            "counterexamples": [
                {
                    "kind": kind,
                    "version": 1,
                    "hardness": hardness[0],
                    "arguments": {"unsupportedField": True},
                }
            ],
            "documentation": [
                "docs/RESEARCH_SCENARIO_V4.md",
                "docs/ENGINE_GUIDE_AR.md",
            ],
            "tests": [
                "tests/test_research_scenario_v4.py",
                "tests/test_research_scenario_predicate_inventory_v1.py",
            ],
            "lifecycleStatus": lifecycle_status,
        }
    )


def built_in_scenario_predicates_v1() -> tuple[ScenarioPredicateDefinitionV1, ...]:
    def range_schema(field: str) -> Mapping[str, Any]:
        return _schema(
            required=(field, "minimum", "maximum", "handStage"),
            properties={
                field: _SEAT,
                "minimum": _NON_NEGATIVE,
                "maximum": _NON_NEGATIVE,
                "handStage": _HAND_STAGE,
            },
        )

    definitions = (
        _definition(
            "deal.card_owner",
            "deal",
            _schema(
                required=("card", "seat", "handStage"),
                properties={"card": _CARD, "seat": _SEAT, "handStage": _HAND_STAGE},
            ),
            "cp_sat_exact",
            hidden="complete_state_only",
            cost="solver",
        ),
        _definition(
            "deal.card_allowed",
            "deal",
            _schema(
                required=("card", "seats", "handStage"),
                properties={
                    "card": _CARD,
                    "seats": {
                        "type": "array",
                        "items": _SEAT,
                        "minItems": 1,
                        "maxItems": 4,
                        "uniqueItems": True,
                    },
                    "handStage": _HAND_STAGE,
                },
            ),
            "cp_sat_exact",
            hidden="complete_state_only",
            cost="solver",
        ),
        _definition(
            "deal.card_forbidden",
            "deal",
            _schema(
                required=("card", "seat", "handStage"),
                properties={"card": _CARD, "seat": _SEAT, "handStage": _HAND_STAGE},
            ),
            "cp_sat_exact",
            hidden="complete_state_only",
            cost="solver",
        ),
        _definition(
            "deal.suit_count",
            "deal",
            _schema(
                required=("seat", "suit", "minimum", "maximum", "handStage"),
                properties={
                    "seat": _SEAT,
                    "suit": _SUIT,
                    "minimum": _NON_NEGATIVE,
                    "maximum": _NON_NEGATIVE,
                    "handStage": _HAND_STAGE,
                },
            ),
            "cp_sat_exact",
            hidden="complete_state_only",
            cost="solver",
        ),
        _definition(
            "deal.rank_count",
            "deal",
            _schema(
                required=("seat", "rank", "minimum", "maximum", "handStage"),
                properties={
                    "seat": _SEAT,
                    "rank": {
                        "type": "string",
                        "enum": [
                            "seven",
                            "eight",
                            "nine",
                            "jack",
                            "queen",
                            "king",
                            "ten",
                            "ace",
                        ],
                    },
                    "minimum": _NON_NEGATIVE,
                    "maximum": _NON_NEGATIVE,
                    "handStage": _HAND_STAGE,
                },
            ),
            "cp_sat_exact",
            hidden="complete_state_only",
            cost="solver",
        ),
        _definition(
            "deal.high_card_count",
            "deal",
            range_schema("seat"),
            "cp_sat_exact",
            hidden="complete_state_only",
            cost="solver",
        ),
        _definition(
            "deal.low_card_count",
            "deal",
            range_schema("seat"),
            "cp_sat_exact",
            hidden="complete_state_only",
            cost="solver",
        ),
        _definition(
            "deal.purchase_card",
            "deal",
            _schema(required=("card",), properties={"card": _CARD}),
            "cp_sat_exact",
            hidden="public",
            cost="solver",
        ),
        _definition(
            "deal.purchase_suit",
            "deal",
            _schema(required=("suit",), properties={"suit": _SUIT}),
            "cp_sat_exact",
            hidden="public",
            cost="solver",
        ),
        _definition(
            "deal.remaining_card_owner",
            "deal",
            _schema(
                required=("card", "seat"), properties={"card": _CARD, "seat": _SEAT}
            ),
            "incremental_actor_safe",
            hidden="complete_state_only",
        ),
        _definition(
            "deal.remaining_suit_count",
            "deal",
            _schema(
                required=("seat", "suit", "minimum", "maximum"),
                properties={
                    "seat": _SEAT,
                    "suit": _SUIT,
                    "minimum": _NON_NEGATIVE,
                    "maximum": _NON_NEGATIVE,
                },
            ),
            "postcondition_only",
            hidden="complete_state_only",
            cost="linear",
        ),
        _definition(
            "deal.remaining_rank_count",
            "deal",
            _schema(
                required=("seat", "rank", "minimum", "maximum"),
                properties={
                    "seat": _SEAT,
                    "rank": {
                        "type": "string",
                        "enum": [
                            "seven",
                            "eight",
                            "nine",
                            "jack",
                            "queen",
                            "king",
                            "ten",
                            "ace",
                        ],
                    },
                    "minimum": _NON_NEGATIVE,
                    "maximum": _NON_NEGATIVE,
                },
            ),
            "postcondition_only",
            hidden="complete_state_only",
            cost="linear",
        ),
        _definition(
            "deal.team_high_card_balance",
            "deal",
            _schema(
                required=("team", "maximumDifference", "handStage"),
                properties={
                    "team": _TEAM,
                    "maximumDifference": _NON_NEGATIVE,
                    "handStage": _HAND_STAGE,
                },
            ),
            "postcondition_only",
            hidden="complete_state_only",
            cost="linear",
        ),
        _definition(
            "deal.seat_relation",
            "deal",
            _schema(
                required=("seat", "otherSeat", "relation"),
                properties={
                    "seat": _SEAT,
                    "otherSeat": _SEAT,
                    "relation": {
                        "type": "string",
                        "enum": ["same", "partner", "next", "previous", "opponent"],
                    },
                },
            ),
            "postcondition_only",
            safe_complement=False,
        ),
        _definition(
            "auction.dealer",
            "auction",
            _schema(required=("seat",), properties={"seat": _SEAT}),
            "cp_sat_exact",
        ),
        _definition(
            "auction.buyer",
            "auction",
            _schema(required=("seat",), properties={"seat": _SEAT}),
            "guided_search",
        ),
        _definition(
            "auction.ashkal_buyer_relation",
            "auction",
            _schema(
                required=("relation",),
                properties={
                    "relation": {
                        "type": "string",
                        "enum": ["bidder", "bidder_partner"],
                    }
                },
            ),
            "postcondition_only",
            safe_complement=False,
        ),
        _definition(
            "auction.contract",
            "auction",
            _schema(required=("contract",), properties={"contract": _CONTRACT}),
            "guided_search",
        ),
        _definition(
            "auction.trump",
            "auction",
            _schema(required=("suit",), properties={"suit": _SUIT}),
            "guided_search",
        ),
        _definition(
            "auction.purchase_round",
            "auction",
            _schema(
                required=("round",),
                properties={"round": {"type": "integer", "minimum": 1, "maximum": 2}},
            ),
            "guided_search",
        ),
        _definition(
            "auction.round",
            "auction",
            _schema(
                required=("round",),
                properties={"round": {"type": "integer", "minimum": 1, "maximum": 2}},
            ),
            "postcondition_only",
        ),
        _definition(
            "auction.action_sequence",
            "auction",
            _schema(
                required=("actions", "complete"),
                properties={
                    "actions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["action"],
                            "properties": {
                                "actor": _SEAT,
                                "action": {
                                    "type": "string",
                                    "enum": ["pass", "sun", "hokum", "ashkal"],
                                },
                                "trumpSuit": _SUIT,
                            },
                        },
                        "minItems": 1,
                    },
                    "complete": {"type": "boolean"},
                },
            ),
            "cp_sat_exact",
            hidden="public",
            cost="solver",
            safe_complement=False,
        ),
        _definition(
            "auction.legal_actions",
            "auction",
            _schema(
                required=("actions", "complete"),
                properties={
                    "actions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["action"],
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "enum": ["pass", "sun", "hokum", "ashkal"],
                                },
                                "trumpSuit": _SUIT,
                            },
                        },
                        "minItems": 1,
                        "maxItems": 7,
                        "uniqueItems": True,
                    },
                    "complete": {"type": "boolean"},
                },
            ),
            "postcondition_only",
            hidden="actor_visible",
            safe_complement=False,
        ),
        _definition(
            "auction.pass_count",
            "auction",
            _schema(
                required=("minimum", "maximum"),
                properties={"minimum": _NON_NEGATIVE, "maximum": _NON_NEGATIVE},
            ),
            "postcondition_only",
        ),
        _definition(
            "auction.terminal",
            "auction",
            _schema(
                required=("terminal",), properties={"terminal": {"type": "boolean"}}
            ),
            "postcondition_only",
        ),
        _definition(
            "escalation.level",
            "escalation",
            _schema(
                required=("level",),
                properties={
                    "level": {
                        "type": "string",
                        "enum": ["normal", "double", "triple", "four", "gahwa"],
                    }
                },
            ),
            "guided_search",
        ),
        _definition(
            "escalation.action_sequence",
            "escalation",
            _schema(
                required=("actions",),
                properties={
                    "actions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["actor", "action"],
                            "properties": {
                                "actor": _SEAT,
                                "action": {
                                    "type": "string",
                                    "enum": [
                                        "decline",
                                        "double",
                                        "triple",
                                        "four",
                                        "gahwa",
                                    ],
                                },
                                "mode": {
                                    "type": "string",
                                    "enum": ["open", "locked"],
                                },
                            },
                        },
                        "minItems": 1,
                    },
                    "complete": {"type": "boolean"},
                },
            ),
            "cp_sat_exact",
            cost="solver",
            safe_complement=False,
        ),
        _definition(
            "escalation.locked",
            "escalation",
            _schema(required=("locked",), properties={"locked": {"type": "boolean"}}),
            "postcondition_only",
        ),
        _definition(
            "escalation.eligible_actor",
            "escalation",
            _schema(
                required=("team", "eligible"),
                properties={
                    "team": _TEAM,
                    "seat": _SEAT,
                    "eligible": {"type": "boolean"},
                },
            ),
            "postcondition_only",
            hidden="actor_visible",
        ),
        _definition(
            "escalation.legal_actions",
            "escalation",
            _schema(
                required=("actions", "complete"),
                properties={
                    "actions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["action"],
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "enum": [
                                        "decline",
                                        "double",
                                        "triple",
                                        "four",
                                        "gahwa",
                                    ],
                                },
                                "mode": {
                                    "type": "string",
                                    "enum": ["open", "locked"],
                                },
                            },
                        },
                        "minItems": 1,
                        "maxItems": 9,
                        "uniqueItems": True,
                    },
                    "complete": {"type": "boolean"},
                },
            ),
            "postcondition_only",
            hidden="actor_visible",
            safe_complement=False,
        ),
        _definition(
            "escalation.terminal",
            "escalation",
            _schema(
                required=("terminal",),
                properties={"terminal": {"type": "boolean"}},
            ),
            "postcondition_only",
        ),
        _definition(
            "project.exists",
            "projects",
            _schema(
                required=("team", "kind", "value", "state"),
                properties={
                    "team": _TEAM,
                    "owner": _SEAT,
                    "kind": {"type": "string", "minLength": 1, "maxLength": 80},
                    "value": _NON_NEGATIVE,
                    "state": {
                        "type": "string",
                        "enum": [
                            "eligible",
                            "declared",
                            "comparison_winner",
                            "awarded",
                        ],
                    },
                },
            ),
            "guided_search",
            hidden="complete_state_only",
            cost="solver",
        ),
        _definition(
            "project.count",
            "projects",
            _schema(
                required=("team", "minimum", "maximum"),
                properties={
                    "team": _TEAM,
                    "minimum": _NON_NEGATIVE,
                    "maximum": _NON_NEGATIVE,
                    "state": {
                        "type": "string",
                        "enum": [
                            "eligible",
                            "declared",
                            "comparison_winner",
                            "awarded",
                        ],
                    },
                },
            ),
            "postcondition_only",
            hidden="complete_state_only",
            cost="linear",
        ),
        _definition(
            "project.combined_value",
            "projects",
            _schema(
                required=("team", "minimum", "maximum", "state"),
                properties={
                    "team": _TEAM,
                    "minimum": _NON_NEGATIVE,
                    "maximum": _NON_NEGATIVE,
                    "state": {
                        "type": "string",
                        "enum": [
                            "eligible",
                            "declared",
                            "comparison_winner",
                            "awarded",
                        ],
                    },
                },
            ),
            "postcondition_only",
            hidden="complete_state_only",
            cost="linear",
        ),
        _definition(
            "project.baloot",
            "projects",
            _schema(
                required=("team", "achieved"),
                properties={"team": _TEAM, "achieved": {"type": "boolean"}},
            ),
            "postcondition_only",
        ),
        _definition(
            "project.overlap_count",
            "projects",
            _schema(
                required=("owner", "minimum", "maximum"),
                properties={
                    "owner": _SEAT,
                    "minimum": _NON_NEGATIVE,
                    "maximum": _NON_NEGATIVE,
                },
            ),
            "postcondition_only",
            hidden="complete_state_only",
            cost="linear",
        ),
        _definition(
            "play.trick_index",
            "playPrefix",
            _schema(
                required=("minimum", "maximum"),
                properties={
                    "minimum": {"type": "integer", "minimum": 0, "maximum": 7},
                    "maximum": {"type": "integer", "minimum": 0, "maximum": 7},
                },
            ),
            "guided_search",
        ),
        _definition(
            "play.current_player",
            "playPrefix",
            _schema(required=("seat",), properties={"seat": _SEAT}),
            "postcondition_only",
        ),
        _definition(
            "play.remaining_hand_size",
            "playPrefix",
            _schema(
                required=("seat", "minimum", "maximum"),
                properties={
                    "seat": _SEAT,
                    "minimum": {"type": "integer", "minimum": 0, "maximum": 8},
                    "maximum": {"type": "integer", "minimum": 0, "maximum": 8},
                },
            ),
            "postcondition_only",
        ),
        _definition(
            "play.card_prefix",
            "playPrefix",
            _schema(
                required=("cards",),
                properties={
                    "cards": {
                        "type": "array",
                        "items": _CARD,
                        "minItems": 1,
                        "maxItems": 32,
                        "uniqueItems": True,
                    },
                    "complete": {"type": "boolean"},
                },
            ),
            "incremental_actor_safe",
            hidden="public",
            cost="linear",
            safe_complement=False,
        ),
        _definition(
            "play.action_prefix",
            "playPrefix",
            _schema(
                required=("actions",),
                properties={
                    "actions": {
                        "type": "array",
                        "items": _PLAY_ACTION,
                        "minItems": 1,
                        "maxItems": 32,
                        "uniqueItems": True,
                    },
                    "complete": {"type": "boolean"},
                },
            ),
            "incremental_actor_safe",
            hidden="public",
            cost="linear",
            safe_complement=False,
            required_primitives=(
                "rules.play.must_follow_suit",
                "rules.play.ekka_enabled",
            ),
        ),
        _definition(
            "play.current_trick_leader",
            "playPrefix",
            _schema(required=("seat",), properties={"seat": _SEAT}),
            "postcondition_only",
        ),
        _definition(
            "play.current_trick_suit_led",
            "playPrefix",
            _schema(required=("suit",), properties={"suit": _SUIT}),
            "incremental_actor_safe",
        ),
        _definition(
            "play.trump_use_count",
            "playPrefix",
            _schema(
                required=("minimum", "maximum"),
                properties={
                    "trickIndex": {"type": "integer", "minimum": 0, "maximum": 7},
                    "minimum": {"type": "integer", "minimum": 0, "maximum": 4},
                    "maximum": {"type": "integer", "minimum": 0, "maximum": 32},
                },
            ),
            "postcondition_only",
            cost="linear",
        ),
        _definition(
            "play.current_trick_card",
            "playPrefix",
            _schema(
                required=("position", "seat", "card"),
                properties={
                    "position": {"type": "integer", "minimum": 0, "maximum": 3},
                    "seat": _SEAT,
                    "card": _CARD,
                    "ekka": {"type": "boolean"},
                },
            ),
            "incremental_actor_safe",
        ),
        _definition(
            "play.completed_trick_winner",
            "playPrefix",
            _schema(
                required=("trickIndex", "seat"),
                properties={
                    "trickIndex": {"type": "integer", "minimum": 0, "maximum": 7},
                    "seat": _SEAT,
                },
            ),
            "postcondition_only",
        ),
        _definition(
            "state.legal_action_count",
            "state",
            _schema(
                required=("minimum", "maximum"),
                properties={"minimum": _NON_NEGATIVE, "maximum": _NON_NEGATIVE},
            ),
            "postcondition_only",
        ),
        _definition(
            "state.match_score",
            "state",
            _schema(
                required=("team", "minimum", "maximum"),
                properties={
                    "team": _TEAM,
                    "minimum": _NON_NEGATIVE,
                    "maximum": _NON_NEGATIVE,
                },
            ),
            "postcondition_only",
        ),
        _definition(
            "state.phase",
            "state",
            _schema(
                required=("state",),
                properties={
                    "state": {
                        "type": "string",
                        "enum": [
                            "created",
                            "shuffled",
                            "initial_dealt",
                            "bidding",
                            "final_dealt",
                            "projects",
                            "playing",
                            "scored",
                            "complete",
                        ],
                    }
                },
            ),
            "postcondition_only",
        ),
        _definition(
            "state.next_actor",
            "state",
            _schema(required=("seat",), properties={"seat": _SEAT}),
            "postcondition_only",
            hidden="actor_visible",
        ),
        _definition(
            "state.observer_visibility",
            "state",
            _schema(
                required=("observer", "classification"),
                properties={
                    "observer": _SEAT,
                    "classification": {
                        "type": "string",
                        "enum": [
                            "public_fact",
                            "legal_deduction",
                            "probabilistic_inference",
                            "policy_assumption",
                        ],
                    },
                },
            ),
            "postcondition_only",
            hidden="information_set_safe",
            hardness=("report",),
            safe_complement=False,
        ),
        _definition(
            "state.observer_card_visibility",
            "state",
            _schema(
                required=("observer", "card", "visibility"),
                properties={
                    "observer": _SEAT,
                    "card": _CARD,
                    "visibility": {
                        "type": "string",
                        "enum": ["visible", "hidden"],
                    },
                },
            ),
            "postcondition_only",
            hidden="information_set_safe",
        ),
        _definition(
            "state.current_raw_points",
            "state",
            _schema(
                required=("team", "minimum", "maximum"),
                properties={
                    "team": _TEAM,
                    "minimum": _NON_NEGATIVE,
                    "maximum": _NON_NEGATIVE,
                },
            ),
            "postcondition_only",
            cost="linear",
        ),
        _definition(
            "state.current_project_points",
            "state",
            _schema(
                required=("team", "minimum", "maximum"),
                properties={
                    "team": _TEAM,
                    "minimum": _NON_NEGATIVE,
                    "maximum": _NON_NEGATIVE,
                },
            ),
            "postcondition_only",
            cost="linear",
        ),
        _definition(
            "state.current_game_points",
            "state",
            _schema(
                required=("team", "minimum", "maximum"),
                properties={
                    "team": _TEAM,
                    "minimum": _NON_NEGATIVE,
                    "maximum": _NON_NEGATIVE,
                },
            ),
            "postcondition_only",
            cost="linear",
        ),
        _definition(
            "state.legal_cards_include",
            "state",
            _schema(
                required=("cards",),
                properties={
                    "cards": {
                        "type": "array",
                        "items": _CARD,
                        "minItems": 1,
                        "maxItems": 8,
                        "uniqueItems": True,
                    },
                    "complete": {"type": "boolean"},
                },
            ),
            "postcondition_only",
            hidden="actor_visible",
        ),
        _definition(
            "state.legal_play_actions_include",
            "state",
            _schema(
                required=("actions",),
                properties={
                    "actions": {
                        "type": "array",
                        "items": _PLAY_ACTION,
                        "minItems": 1,
                        "maxItems": 16,
                        "uniqueItems": True,
                    },
                    "complete": {"type": "boolean"},
                },
            ),
            "postcondition_only",
            hidden="actor_visible",
            required_primitives=(
                "rules.play.must_follow_suit",
                "rules.play.ekka_enabled",
            ),
        ),
        _definition(
            "outcome.buyer_contract",
            "outcome",
            _schema(
                required=("status",),
                properties={"status": {"type": "string", "enum": ["won", "lost"]}},
            ),
            "postcondition_only",
            proof_modes=("exact", "bounded", "policy_conditioned"),
            cost="tree_search",
        ),
        _definition(
            "outcome.team_round",
            "outcome",
            _schema(
                required=("team", "status"),
                properties={
                    "team": _TEAM,
                    "status": {"type": "string", "enum": ["won", "lost", "tied"]},
                },
            ),
            "postcondition_only",
            proof_modes=("exact", "bounded", "policy_conditioned"),
            cost="tree_search",
        ),
        _definition(
            "outcome.team_match",
            "outcome",
            _schema(
                required=("team", "status"),
                properties={
                    "team": _TEAM,
                    "status": {"type": "string", "enum": ["won", "lost", "tied"]},
                },
            ),
            "postcondition_only",
            proof_modes=("exact", "bounded", "policy_conditioned"),
            cost="tree_search",
        ),
        _definition(
            "outcome.raw_points",
            "outcome",
            _schema(
                required=("team", "minimum", "maximum"),
                properties={
                    "team": _TEAM,
                    "minimum": _NON_NEGATIVE,
                    "maximum": _NON_NEGATIVE,
                },
            ),
            "postcondition_only",
            proof_modes=("exact", "bounded", "policy_conditioned"),
            cost="tree_search",
        ),
        _definition(
            "outcome.game_points",
            "outcome",
            _schema(
                required=("team", "minimum", "maximum"),
                properties={
                    "team": _TEAM,
                    "minimum": _NON_NEGATIVE,
                    "maximum": _NON_NEGATIVE,
                },
            ),
            "postcondition_only",
            proof_modes=("exact", "bounded", "policy_conditioned"),
            cost="tree_search",
        ),
        _definition(
            "outcome.project_points",
            "outcome",
            _schema(
                required=("team", "minimum", "maximum"),
                properties={
                    "team": _TEAM,
                    "minimum": _NON_NEGATIVE,
                    "maximum": _NON_NEGATIVE,
                },
            ),
            "postcondition_only",
            proof_modes=("exact", "bounded", "policy_conditioned"),
            cost="tree_search",
        ),
        _definition(
            "outcome.round_margin",
            "outcome",
            _schema(
                required=("team", "minimum", "maximum"),
                properties={
                    "team": _TEAM,
                    "minimum": {"type": "number"},
                    "maximum": {"type": "number"},
                },
            ),
            "postcondition_only",
            proof_modes=("exact", "bounded", "policy_conditioned"),
            cost="tree_search",
        ),
        _definition(
            "outcome.kabout",
            "outcome",
            _schema(
                required=("team", "achieved"),
                properties={"team": _TEAM, "achieved": {"type": "boolean"}},
            ),
            "postcondition_only",
            proof_modes=("exact", "bounded", "policy_conditioned"),
            cost="tree_search",
        ),
        _definition(
            "outcome.gahwa",
            "outcome",
            _schema(
                required=("team", "achieved"),
                properties={"team": _TEAM, "achieved": {"type": "boolean"}},
            ),
            "postcondition_only",
            proof_modes=("exact", "bounded", "policy_conditioned"),
            cost="tree_search",
        ),
        _definition(
            "outcome.match_target_crossing",
            "outcome",
            _schema(
                required=("team", "achieved"),
                properties={"team": _TEAM, "achieved": {"type": "boolean"}},
            ),
            "postcondition_only",
            proof_modes=("exact", "bounded", "policy_conditioned"),
            cost="tree_search",
        ),
        _definition(
            "tactic.verified",
            "tactics",
            _schema(
                required=("kind", "minimumScore", "minimumEvidence"),
                properties={
                    "kind": {
                        "type": "string",
                        "enum": sorted(item.value for item in TacticalKind),
                    },
                    "actingSeat": _SEAT,
                    "observer": _SEAT,
                    "targetSeat": _SEAT,
                    "targetTeam": _TEAM,
                    "causalRequired": {"type": "boolean"},
                    "minimumScore": {"type": "number", "minimum": 0, "maximum": 100},
                    "minimumEvidence": _NON_NEGATIVE,
                },
            ),
            "postcondition_only",
            hidden="information_set_safe",
            cost="linear",
        ),
        _definition(
            "tactic.applicable",
            "tactics",
            _schema(
                required=("kind", "applicable"),
                properties={
                    "kind": {
                        "type": "string",
                        "enum": sorted(item.value for item in TacticalKind),
                    },
                    "actingSeat": _SEAT,
                    "observer": _SEAT,
                    "targetSeat": _SEAT,
                    "targetTeam": _TEAM,
                    "applicable": {"type": "boolean"},
                },
            ),
            "postcondition_only",
            hidden="information_set_safe",
            cost="linear",
        ),
        _definition(
            "proof.winning_root_action_count",
            "proof",
            _schema(
                required=("minimum", "maximum"),
                properties={"minimum": _NON_NEGATIVE, "maximum": _NON_NEGATIVE},
            ),
            "postcondition_only",
            proof_modes=("exact", "bounded", "policy_conditioned"),
            hidden="information_set_safe",
            cost="tree_search",
            hardness=("hard", "report"),
        ),
        _definition(
            "proof.winning_root_card_rank_count",
            "proof",
            _schema(
                required=("rank", "minimum", "maximum"),
                properties={
                    "rank": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {
                            "type": "string",
                            "enum": [
                                "seven",
                                "eight",
                                "nine",
                                "ten",
                                "jack",
                                "queen",
                                "king",
                                "ace",
                            ],
                        },
                    },
                    "minimum": {"type": "integer", "minimum": 0, "maximum": 8},
                    "maximum": {"type": "integer", "minimum": 0, "maximum": 8},
                },
            ),
            "postcondition_only",
            proof_modes=("exact",),
            hidden="complete_state_only",
            cost="linear",
            hardness=("hard", "report"),
        ),
        _definition(
            "proof.winning_continuation_line_count",
            "proof",
            _schema(
                required=("lineUniverse",),
                properties={
                    "lineUniverse": {
                        "type": "string",
                        "enum": ["all_legal", "optimal_strategy", "policy_conditioned"],
                    },
                    "minimum": {"type": "string", "pattern": "^[0-9]+$"},
                    "maximum": {"type": "string", "pattern": "^[0-9]+$"},
                },
            ),
            "postcondition_only",
            proof_modes=("exact", "bounded", "policy_conditioned"),
            hidden="information_set_safe",
            cost="tree_search",
            hardness=("hard", "report"),
        ),
        _definition(
            "proof.principal_variation_trick_winner",
            "proof",
            _schema(
                required=("trickIndex", "team", "rank"),
                properties={
                    "trickIndex": {"type": "integer", "minimum": 0, "maximum": 7},
                    "team": _TEAM,
                    "rank": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {
                            "type": "string",
                            "enum": [
                                "seven",
                                "eight",
                                "nine",
                                "ten",
                                "jack",
                                "queen",
                                "king",
                                "ace",
                            ],
                        },
                    },
                },
            ),
            "postcondition_only",
            proof_modes=("exact",),
            hidden="complete_state_only",
            cost="linear",
            hardness=("hard", "report"),
            message_key="research_scenario_play_completed_trick_winner",
        ),
        _definition(
            "proof.principal_variation_trick_team_rank_count",
            "proof",
            _schema(
                required=("trickIndex", "team", "rank", "minimum", "maximum"),
                properties={
                    "trickIndex": {"type": "integer", "minimum": 0, "maximum": 7},
                    "team": _TEAM,
                    "rank": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {
                            "type": "string",
                            "enum": [
                                "seven",
                                "eight",
                                "nine",
                                "ten",
                                "jack",
                                "queen",
                                "king",
                                "ace",
                            ],
                        },
                    },
                    "minimum": {"type": "integer", "minimum": 0, "maximum": 2},
                    "maximum": {"type": "integer", "minimum": 0, "maximum": 2},
                },
            ),
            "postcondition_only",
            proof_modes=("exact",),
            hidden="complete_state_only",
            cost="linear",
            hardness=("hard", "report"),
        ),
    )
    kinds = [item.kind for item in definitions]
    if len(kinds) != len(set(kinds)):
        raise AssertionError("built-in scenario predicate kinds must be unique")
    return definitions


def _validate_schema_value(value: Any, schema: Mapping[str, Any], path: str) -> None:
    expected = schema.get("type")
    if expected == "object":
        if not isinstance(value, Mapping):
            raise TypeError(f"{path} must be an object")
        raw = dict(value)
        properties = dict(schema.get("properties") or {})
        required = set(schema.get("required") or ())
        missing = sorted(required - set(raw))
        unknown = (
            sorted(set(raw) - set(properties))
            if schema.get("additionalProperties") is False
            else []
        )
        if missing or unknown:
            raise ValueError(
                f"{path} shape mismatch; missing={missing}, unknown={unknown}"
            )
        for key, item in raw.items():
            if key in properties:
                _validate_schema_value(item, properties[key], f"{path}.{key}")
        return
    if expected == "array":
        if not isinstance(value, list):
            raise TypeError(f"{path} must be an array")
        if len(value) < int(schema.get("minItems", 0)) or len(value) > int(
            schema.get("maxItems", 1 << 30)
        ):
            raise ValueError(f"{path} array length is invalid")
        if schema.get("uniqueItems") and len(
            {canonical_sha256(item) for item in value}
        ) != len(value):
            raise ValueError(f"{path} values must be unique")
        for index, item in enumerate(value):
            _validate_schema_value(
                item, dict(schema.get("items") or {}), f"{path}[{index}]"
            )
        return
    if expected == "integer" and type(value) is not int:
        raise TypeError(f"{path} must be an integer")
    if expected == "number" and type(value) not in {int, float}:
        raise TypeError(f"{path} must be a number")
    if expected == "string" and not isinstance(value, str):
        raise TypeError(f"{path} must be a string")
    if expected == "boolean" and type(value) is not bool:
        raise TypeError(f"{path} must be a boolean")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} is not an allowed value")
    if type(value) in {int, float}:
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{path} is below its minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{path} exceeds its maximum")
    if isinstance(value, str):
        if len(value) < int(schema.get("minLength", 0)) or len(value) > int(
            schema.get("maxLength", 1 << 30)
        ):
            raise ValueError(f"{path} string length is invalid")
        pattern = schema.get("pattern")
        if pattern is not None:
            import re

            if re.fullmatch(str(pattern), value) is None:
                raise ValueError(f"{path} does not match its pattern")


def _validate_schema_localization(schema: Mapping[str, Any], path: str) -> None:
    if schema.get("type") == "object":
        for name, value in dict(schema.get("properties") or {}).items():
            item = dict(value)
            if not item.get("x-messageArKey") or not item.get("x-messageEnKey"):
                raise ValueError(
                    f"scenario predicate argument {path}.{name} lacks localization"
                )
            enum_values = item.get("enum")
            if (
                enum_values is None
                and item.get("type") == "array"
                and isinstance(item.get("items"), Mapping)
            ):
                enum_values = item["items"].get("enum")
            if isinstance(enum_values, list):
                expected = {str(value) for value in enum_values}
                ar_keys = set(dict(item.get("x-enumMessageArKeys") or {}))
                en_keys = set(dict(item.get("x-enumMessageEnKeys") or {}))
                if ar_keys != expected or en_keys != expected:
                    raise ValueError(
                        f"scenario predicate argument {path}.{name} enum localization is incomplete"
                    )
            _validate_schema_localization(item, f"{path}.{name}")
    elif schema.get("type") == "array" and isinstance(schema.get("items"), Mapping):
        _validate_schema_localization(schema["items"], f"{path}[]")


class ScenarioPredicateRegistryV1:
    def __init__(
        self, definitions: tuple[ScenarioPredicateDefinitionV1, ...] | None = None
    ):
        rows = definitions or built_in_scenario_predicates_v1()
        self._definitions = {item.kind: item for item in rows}
        if len(self._definitions) != len(rows):
            raise ValueError("scenario predicate registry contains duplicate kinds")
        for definition in rows:
            if (
                definition.stage not in SCENARIO_SECTIONS
                or definition.category != definition.stage
            ):
                raise ValueError(
                    f"scenario predicate {definition.kind} has an invalid stage"
                )
            if (
                "hard" in definition.allowed_hardness
                and not definition.runtime_verifier
            ):
                raise ValueError(
                    f"hard scenario predicate {definition.kind} lacks a runtime verifier"
                )
            if (
                definition.construction_capability == "unsupported"
                and "hard" in definition.allowed_hardness
            ):
                raise ValueError(
                    f"unsupported scenario predicate {definition.kind} allows hard qualification"
                )
            expected_verifier = f"verify_{definition.kind.replace('.', '_')}_v1"
            if definition.runtime_verifier != expected_verifier:
                raise ValueError(
                    f"scenario predicate {definition.kind} runtime verifier identity drifted"
                )
            _validate_schema_localization(
                definition.argument_schema,
                f"{definition.kind}.arguments",
            )
            for index, example in enumerate(definition.examples):
                if (
                    example.get("kind") != definition.kind
                    or example.get("version") != definition.version
                    or example.get("hardness") not in definition.allowed_hardness
                ):
                    raise ValueError(
                        f"scenario predicate {definition.kind} example {index} identity drifted"
                    )
                _validate_schema_value(
                    example.get("arguments"),
                    definition.argument_schema,
                    f"{definition.kind}.examples[{index}].arguments",
                )
            for index, counterexample in enumerate(definition.counterexamples):
                try:
                    _validate_schema_value(
                        counterexample.get("arguments"),
                        definition.argument_schema,
                        f"{definition.kind}.counterexamples[{index}].arguments",
                    )
                except (TypeError, ValueError):
                    continue
                raise ValueError(
                    f"scenario predicate {definition.kind} counterexample {index} is valid"
                )
        active_stages = {
            item.stage
            for item in rows
            if item.lifecycle_status == "active"
            and item.construction_capability != "unsupported"
        }
        if active_stages != set(SCENARIO_SECTIONS):
            raise ValueError(
                "scenario predicate registry stage coverage is incomplete: "
                + ", ".join(sorted(set(SCENARIO_SECTIONS) - active_stages))
            )

    def definition(self, kind: str) -> ScenarioPredicateDefinitionV1 | None:
        return self._definitions.get(str(kind))

    def require(self, predicate: ScenarioPredicateV1) -> ScenarioPredicateDefinitionV1:
        definition = self.definition(predicate.kind)
        if definition is None or definition.version != predicate.version:
            raise ValueError(
                f"unsupported scenario predicate: {predicate.kind}@{predicate.version}"
            )
        _validate_schema_value(
            predicate.arguments, definition.argument_schema, predicate.predicate_id
        )
        if predicate.hardness not in definition.allowed_hardness:
            raise ValueError(
                f"{predicate.kind} does not allow hardness {predicate.hardness}"
            )
        return definition

    def catalog(self) -> dict[str, Any]:
        rows = [
            self._definitions[key].to_dict()
            for key in sorted(self._definitions)
            if self._definitions[key].lifecycle_status == "active"
            and self._definitions[key].construction_capability != "unsupported"
        ]
        core = {
            "schemaVersion": 1,
            "registryVersion": "scenario_predicate_registry_v1",
            "definitions": rows,
        }
        return {**core, "sha256": canonical_sha256(core)}


@dataclass(frozen=True, slots=True)
class ScenarioCapabilityPlanV1:
    scenario_hash: str
    proof_hash: str
    rows: tuple[Mapping[str, Any], ...]
    diagnostics: tuple[ScenarioDiagnosticV1, ...]
    valid: bool
    construction_hash: str
    runtime_verification_hash: str
    proof_plan_hash: str
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "scenarioHash": self.scenario_hash,
            "proofHash": self.proof_hash,
            "rows": [copy.deepcopy(dict(item)) for item in self.rows],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "valid": self.valid,
            "constructionHash": self.construction_hash,
            "runtimeVerificationHash": self.runtime_verification_hash,
            "proofPlanHash": self.proof_plan_hash,
            "sha256": self.sha256,
        }


class ScenarioCompilerV1:
    def __init__(self, registry: ScenarioPredicateRegistryV1 | None = None):
        self.registry = registry or ScenarioPredicateRegistryV1()

    @staticmethod
    def _diagnostic(
        *,
        index: int,
        diagnostic_class: str,
        code: str,
        predicate_ids: tuple[str, ...],
        field_paths: tuple[str, ...] | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> ScenarioDiagnosticV1:
        return ScenarioDiagnosticV1.build(
            {
                "issueId": f"ISS-{index:04d}",
                "class": diagnostic_class,
                "severity": "error",
                "code": code,
                "predicateIds": list(predicate_ids),
                "fieldPaths": list(field_paths)
                if field_paths is not None
                else [f"scenario.predicates.{item}" for item in predicate_ids],
                "messageArKey": f"research_scenario_issue_{code}",
                "messageEnKey": f"research_scenario_issue_{code}_en",
                "details": dict(details or {}),
            }
        )

    def compile(
        self,
        scenario: ResearchScenarioV1,
        proof: ScenarioProofRequestV1,
        *,
        active_domain_primitives: frozenset[str] | None = None,
    ) -> ScenarioCapabilityPlanV1:
        diagnostics: list[ScenarioDiagnosticV1] = []
        rows: list[dict[str, Any]] = []
        owner_by_card: dict[tuple[str, int], tuple[str, str]] = {}
        singleton_values: dict[str, tuple[str, str]] = {}
        decision_kind = str(scenario.decision_point["kind"])
        non_play_policy_supported = (
            decision_kind in {"bid", "escalation", "project"}
            and proof.mode == "policy_conditioned"
            and proof.visibility == "complete_state"
        )
        if decision_kind != "play" and not non_play_policy_supported:
            diagnostics.append(
                self._diagnostic(
                    index=1,
                    diagnostic_class="unsupported_predicate",
                    code="decision_point_proof_unavailable",
                    predicate_ids=(),
                    field_paths=("scenario.decisionPoint.kind",),
                    details={
                        "decisionKind": decision_kind,
                        "supportedDecisionKinds": [
                            "play",
                            "bid:policy_conditioned:complete_state",
                            "escalation:policy_conditioned:complete_state",
                            "project:policy_conditioned:complete_state",
                        ],
                    },
                )
            )
        for expression in (scenario.constraints, scenario.outcome):
            self._validate_complements(expression, diagnostics)
        for predicate in scenario.predicates():
            try:
                definition = self.registry.require(predicate)
            except (TypeError, ValueError) as exc:
                diagnostics.append(
                    self._diagnostic(
                        index=len(diagnostics) + 1,
                        diagnostic_class="unsupported_predicate",
                        code="unsupported_or_invalid_predicate",
                        predicate_ids=(predicate.predicate_id,),
                        details={"message": str(exc)},
                    )
                )
                continue
            missing_primitives = tuple(
                sorted(
                    set(definition.required_domain_primitives)
                    - set(
                        active_domain_primitives
                        or definition.required_domain_primitives
                    )
                )
            )
            if missing_primitives:
                diagnostics.append(
                    self._diagnostic(
                        index=len(diagnostics) + 1,
                        diagnostic_class="rules_inapplicable",
                        code="missing_domain_primitives",
                        predicate_ids=(predicate.predicate_id,),
                        details={"primitiveIds": list(missing_primitives)},
                    )
                )
            if (
                predicate.hardness == "hard"
                and definition.construction_capability == "unsupported"
            ):
                diagnostics.append(
                    self._diagnostic(
                        index=len(diagnostics) + 1,
                        diagnostic_class="unsupported_predicate",
                        code="hard_predicate_is_unsupported",
                        predicate_ids=(predicate.predicate_id,),
                    )
                )
            if (
                definition.supported_proof_modes
                and proof.mode not in definition.supported_proof_modes
            ):
                diagnostics.append(
                    self._diagnostic(
                        index=len(diagnostics) + 1,
                        diagnostic_class="unsupported_predicate",
                        code="predicate_proof_mode_mismatch",
                        predicate_ids=(predicate.predicate_id,),
                        details={"proofMode": proof.mode},
                    )
                )
            if predicate.kind == "deal.card_owner":
                card = int(predicate.arguments["card"])
                seat = str(predicate.arguments["seat"])
                hand_stage = str(predicate.arguments["handStage"])
                owner_key = (hand_stage, card)
                prior = owner_by_card.get(owner_key)
                if (
                    prior is not None
                    and prior[0] != seat
                    and predicate.hardness == "hard"
                ):
                    diagnostics.append(
                        self._diagnostic(
                            index=len(diagnostics) + 1,
                            diagnostic_class="contradictory_constraints",
                            code="card_required_by_two_seats",
                            predicate_ids=(prior[1], predicate.predicate_id),
                            details={
                                "card": card,
                                "handStage": hand_stage,
                                "seats": [prior[0], seat],
                            },
                        )
                    )
                elif predicate.hardness == "hard":
                    owner_by_card[owner_key] = (seat, predicate.predicate_id)
            singleton_field = {
                "auction.dealer": "seat",
                "auction.buyer": "seat",
                "auction.contract": "contract",
                "auction.trump": "suit",
                "auction.purchase_round": "round",
            }.get(predicate.kind)
            if singleton_field is not None and predicate.hardness == "hard":
                selected = str(predicate.arguments[singleton_field])
                prior = singleton_values.get(predicate.kind)
                if prior is not None and prior[0] != selected:
                    diagnostics.append(
                        self._diagnostic(
                            index=len(diagnostics) + 1,
                            diagnostic_class="contradictory_constraints",
                            code="conflicting_singleton_predicates",
                            predicate_ids=(prior[1], predicate.predicate_id),
                            details={
                                "kind": predicate.kind,
                                "values": [prior[0], selected],
                            },
                        )
                    )
                elif prior is None:
                    singleton_values[predicate.kind] = (
                        selected,
                        predicate.predicate_id,
                    )
            if (
                predicate.kind == "project.exists"
                and predicate.hardness == "hard"
                and predicate.arguments.get("owner") is not None
            ):
                owner = str(predicate.arguments["owner"])
                owner_team = (
                    "north_south" if owner in {"north", "south"} else "west_east"
                )
                if owner_team != str(predicate.arguments["team"]):
                    diagnostics.append(
                        self._diagnostic(
                            index=len(diagnostics) + 1,
                            diagnostic_class="contradictory_constraints",
                            code="project_owner_team_mismatch",
                            predicate_ids=(predicate.predicate_id,),
                            details={
                                "owner": owner,
                                "ownerTeam": owner_team,
                                "requestedTeam": predicate.arguments["team"],
                            },
                        )
                    )
            if (
                "minimum" in predicate.arguments
                and "maximum" in predicate.arguments
                and float(predicate.arguments["minimum"])
                > float(predicate.arguments["maximum"])
            ):
                diagnostics.append(
                    self._diagnostic(
                        index=len(diagnostics) + 1,
                        diagnostic_class="contradictory_constraints",
                        code="invalid_predicate_range",
                        predicate_ids=(predicate.predicate_id,),
                        details={
                            "minimum": predicate.arguments["minimum"],
                            "maximum": predicate.arguments["maximum"],
                        },
                    )
                )
            rows.append(
                {
                    "predicateId": predicate.predicate_id,
                    "predicateHash": predicate.sha256,
                    "kind": predicate.kind,
                    "hardness": predicate.hardness,
                    "constructionCapability": definition.construction_capability,
                    "runtimeVerifier": definition.runtime_verifier,
                    "proofModes": list(definition.supported_proof_modes),
                    "hiddenInformation": definition.hidden_information,
                    "costClass": definition.cost_class,
                    "missingDomainPrimitives": list(missing_primitives),
                }
            )
            if (
                scenario.origin.kind == "search"
                and definition.construction_capability == "exact_input_only"
                and predicate.hardness == "hard"
            ):
                diagnostics.append(
                    self._diagnostic(
                        index=len(diagnostics) + 1,
                        diagnostic_class="unsupported_predicate",
                        code="predicate_requires_exact_input",
                        predicate_ids=(predicate.predicate_id,),
                        details={"kind": predicate.kind},
                    )
                )

        if scenario.origin.kind == "search" and decision_kind == "play":
            selector = scenario.decision_point["selector"]
            decision_trick = selector.get("trickIndex")
            if decision_trick is not None:
                future_completed = [
                    predicate
                    for predicate in scenario.predicates()
                    if predicate.hardness == "hard"
                    and predicate.kind == "play.completed_trick_winner"
                    and int(predicate.arguments["trickIndex"])
                    >= int(decision_trick)
                ]
                for predicate in future_completed:
                    diagnostics.append(
                        self._diagnostic(
                            index=len(diagnostics) + 1,
                            diagnostic_class="contradictory_constraints",
                            code="completed_trick_is_not_before_play_decision",
                            predicate_ids=(predicate.predicate_id,),
                            field_paths=(
                                "scenario.decisionPoint.selector.trickIndex",
                                f"scenario.predicates.{predicate.predicate_id}.arguments.trickIndex",
                            ),
                            details={
                                "decisionTrickIndex": int(decision_trick),
                                "completedTrickIndex": int(
                                    predicate.arguments["trickIndex"]
                                ),
                            },
                        )
                    )
            current_player_rows = [
                predicate
                for predicate in scenario.predicates()
                if predicate.hardness == "hard"
                and predicate.kind == "play.current_player"
            ]
            if (
                len(current_player_rows) == 1
                and selector.get("actingSeat") is not None
                and current_player_rows[0].arguments["seat"] != selector["actingSeat"]
            ):
                diagnostics.append(
                    self._diagnostic(
                        index=len(diagnostics) + 1,
                        diagnostic_class="contradictory_constraints",
                        code="play_decision_actor_conflicts_with_current_player",
                        predicate_ids=(current_player_rows[0].predicate_id,),
                        field_paths=(
                            "scenario.decisionPoint.selector.actingSeat",
                            f"scenario.predicates.{current_player_rows[0].predicate_id}",
                        ),
                        details={
                            "actingSeat": selector["actingSeat"],
                            "currentPlayer": current_player_rows[0].arguments["seat"],
                        },
                    )
                )
            exact_trick_rows = [
                predicate
                for predicate in scenario.predicates()
                if predicate.hardness == "hard"
                and predicate.kind == "play.trick_index"
                and predicate.arguments["minimum"] == predicate.arguments["maximum"]
            ]
            if len(exact_trick_rows) == 1:
                trick_index = int(exact_trick_rows[0].arguments["minimum"])
                remaining_cards = int(selector.get("remainingCards", 8))
                maximum_remaining = 32 - (4 * trick_index)
                minimum_remaining = maximum_remaining - 3
                if not minimum_remaining <= remaining_cards <= maximum_remaining:
                    diagnostics.append(
                        self._diagnostic(
                            index=len(diagnostics) + 1,
                            diagnostic_class="contradictory_constraints",
                            code="play_decision_remaining_cards_conflicts_with_trick_index",
                            predicate_ids=(exact_trick_rows[0].predicate_id,),
                            field_paths=(
                                "scenario.decisionPoint.selector.remainingCards",
                                f"scenario.predicates.{exact_trick_rows[0].predicate_id}",
                            ),
                            details={
                                "trickIndex": trick_index,
                                "remainingCards": remaining_cards,
                                "allowedRemainingCards": [
                                    minimum_remaining,
                                    maximum_remaining,
                                ],
                            },
                        )
                    )

        construction_rows = [
            item
            for item in rows
            if item["constructionCapability"]
            in {"cp_sat_exact", "incremental_actor_safe", "guided_search"}
        ]
        runtime_rows = [
            {
                "predicateId": item["predicateId"],
                "runtimeVerifier": item["runtimeVerifier"],
            }
            for item in rows
        ]
        proof_rows = [
            {"predicateId": item["predicateId"], "proofModes": item["proofModes"]}
            for item in rows
            if item["proofModes"]
        ]
        construction_hash = canonical_sha256(
            {"schemaVersion": 1, "rows": construction_rows}
        )
        runtime_hash = canonical_sha256({"schemaVersion": 1, "rows": runtime_rows})
        proof_hash = canonical_sha256(
            {"schemaVersion": 1, "request": proof.to_dict(), "rows": proof_rows}
        )
        core = {
            "schemaVersion": 1,
            "scenarioHash": scenario.sha256,
            "proofHash": proof.sha256,
            "rows": rows,
            "diagnosticHashes": [item.sha256 for item in diagnostics],
            "valid": not any(item.severity == "error" for item in diagnostics),
            "constructionHash": construction_hash,
            "runtimeVerificationHash": runtime_hash,
            "proofPlanHash": proof_hash,
        }
        digest = canonical_sha256(core)
        return ScenarioCapabilityPlanV1(
            scenario.sha256,
            proof.sha256,
            tuple(rows),
            tuple(diagnostics),
            bool(core["valid"]),
            construction_hash,
            runtime_hash,
            proof_hash,
            digest,
        )

    def _validate_complements(
        self,
        expression: ScenarioExpressionV1,
        diagnostics: list[ScenarioDiagnosticV1],
        *,
        under_not: bool = False,
    ) -> None:
        if expression.predicate is not None:
            definition = self.registry.definition(expression.predicate.kind)
            if under_not and (definition is None or not definition.safe_complement):
                diagnostics.append(
                    self._diagnostic(
                        index=len(diagnostics) + 1,
                        diagnostic_class="unsupported_predicate",
                        code="unsafe_logical_complement",
                        predicate_ids=(expression.predicate.predicate_id,),
                    )
                )
            return
        for child in expression.children:
            self._validate_complements(
                child,
                diagnostics,
                under_not=under_not or expression.op == "not",
            )


__all__ = [
    "ScenarioCapabilityPlanV1",
    "ScenarioCompilerV1",
    "ScenarioPredicateDefinitionV1",
    "ScenarioPredicateRegistryV1",
    "built_in_scenario_predicates_v1",
]
