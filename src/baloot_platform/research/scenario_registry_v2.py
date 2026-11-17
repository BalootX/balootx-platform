"""Fail-closed Scenario Research V2 predicate catalog contracts.

This module is deliberately isolated from the active V1 registry.  It provides
the stricter catalog records needed by the OR-Tools-only research backend while
keeping legacy recipes and catalogs reproducible.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping

from .scenario_contracts import (
    PREDICATE_HARDNESS,
    SCENARIO_SECTIONS,
    canonical_sha256,
)
from .scenario_registry import (
    ScenarioPredicateDefinitionV1,
    _validate_schema_localization,
    _validate_schema_value,
    built_in_scenario_predicates_v1,
)


CP_SAT_SEARCH_STATUSES = frozenset({"exact", "derived", "unsupported"})
CP_SAT_EXACT_INPUT_STATUSES = frozenset({"exact", "unsupported"})
CP_SAT_QUALIFICATION_SOURCES = frozenset(
    {"construction", "derived_state", "proof", "report_only"}
)
CP_SAT_ACTOR_INFORMATION_CLASSES = frozenset(
    {"public", "observation", "reference_world_only", "nonanticipative"}
)
CP_SAT_EXACT_PROOF_STATUSES = frozenset({"exact", "unsupported"})
CP_SAT_BOUNDED_PROOF_STATUSES = frozenset({"exact", "bounds", "unsupported"})
CP_SAT_POLICY_PROOF_STATUSES = frozenset({"exact", "unsupported"})

_RANKS = ("seven", "eight", "nine", "ten", "jack", "queen", "king", "ace")
_SUITS = ("spades", "hearts", "diamonds", "clubs")
_SEATS = ("north", "west", "south", "east")
_TEAMS = ("north_south", "west_east")
_OWNER_RELATIONS = ("actor", "partner", "opponent_team", "seat", "team")
_REMAINING_DEAL_KINDS = frozenset(
    {
        "deal.remaining_card_owner",
        "deal.remaining_suit_count",
        "deal.remaining_rank_count",
    }
)


def _strict_string(value: Any, path: str, *, allow_empty: bool = False) -> str:
    if type(value) is not str:
        raise TypeError(f"{path} must be a string")
    if not allow_empty and not value:
        raise ValueError(f"{path} must be non-empty")
    return value


def _strict_string_list(
    value: Any,
    path: str,
    *,
    allowed: frozenset[str] | set[str] | None = None,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if type(value) is not list:
        raise TypeError(f"{path} must be an array")
    if not allow_empty and not value:
        raise ValueError(f"{path} must be non-empty")
    items = tuple(_strict_string(item, f"{path}[]") for item in value)
    if len(set(items)) != len(items):
        raise ValueError(f"{path} values must be unique")
    if allowed is not None and not set(items).issubset(allowed):
        raise ValueError(f"{path} contains an unsupported value")
    return items


@dataclass(frozen=True, slots=True)
class CpSatSupportV1:
    """Truthful CP-SAT capability metadata for one predicate kind."""

    search_status: str
    search_stages: tuple[str, ...]
    exact_input_status: str
    exact_input_stages: tuple[str, ...]
    qualification_source: str
    actor_information_set: str
    exact_proof: str
    bounded_proof: str
    policy_conditioned_proof: str
    evidence_extractor: str

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "CpSatSupportV1":
        if not isinstance(value, Mapping):
            raise TypeError("cpSatSupport must be an object")
        raw = copy.deepcopy(dict(value))
        required = {
            "schemaVersion",
            "search",
            "exactInput",
            "qualificationSource",
            "actorInformationSet",
            "proof",
            "evidenceExtractor",
        }
        missing = sorted(required - set(raw))
        unknown = sorted(set(raw) - required)
        if missing or unknown:
            raise ValueError(
                "cpSatSupport shape mismatch; "
                f"missing={missing}, unknown={unknown}"
            )
        if type(raw["schemaVersion"]) is not int or raw["schemaVersion"] != 1:
            raise ValueError("cpSatSupport schemaVersion must equal 1")

        search_status, search_stages = cls._build_stage_support(
            raw["search"], "cpSatSupport.search", CP_SAT_SEARCH_STATUSES
        )
        exact_input_status, exact_input_stages = cls._build_stage_support(
            raw["exactInput"],
            "cpSatSupport.exactInput",
            CP_SAT_EXACT_INPUT_STATUSES,
        )
        qualification_source = _strict_string(
            raw["qualificationSource"], "cpSatSupport.qualificationSource"
        )
        if qualification_source not in CP_SAT_QUALIFICATION_SOURCES:
            raise ValueError("cpSatSupport qualificationSource is invalid")
        actor_information_set = _strict_string(
            raw["actorInformationSet"], "cpSatSupport.actorInformationSet"
        )
        if actor_information_set not in CP_SAT_ACTOR_INFORMATION_CLASSES:
            raise ValueError("cpSatSupport actorInformationSet is invalid")

        proof = raw["proof"]
        if not isinstance(proof, Mapping):
            raise TypeError("cpSatSupport.proof must be an object")
        proof_raw = dict(proof)
        proof_required = {"exact", "bounded", "policyConditioned"}
        proof_missing = sorted(proof_required - set(proof_raw))
        proof_unknown = sorted(set(proof_raw) - proof_required)
        if proof_missing or proof_unknown:
            raise ValueError(
                "cpSatSupport.proof shape mismatch; "
                f"missing={proof_missing}, unknown={proof_unknown}"
            )
        exact_proof = _strict_string(
            proof_raw["exact"], "cpSatSupport.proof.exact"
        )
        bounded_proof = _strict_string(
            proof_raw["bounded"], "cpSatSupport.proof.bounded"
        )
        policy_proof = _strict_string(
            proof_raw["policyConditioned"],
            "cpSatSupport.proof.policyConditioned",
        )
        if exact_proof not in CP_SAT_EXACT_PROOF_STATUSES:
            raise ValueError("cpSatSupport exact proof status is invalid")
        if bounded_proof not in CP_SAT_BOUNDED_PROOF_STATUSES:
            raise ValueError("cpSatSupport bounded proof status is invalid")
        if policy_proof not in CP_SAT_POLICY_PROOF_STATUSES:
            raise ValueError("cpSatSupport policy-conditioned proof status is invalid")

        evidence_extractor = _strict_string(
            raw["evidenceExtractor"],
            "cpSatSupport.evidenceExtractor",
            allow_empty=True,
        )
        if (
            search_status != "unsupported"
            or exact_input_status != "unsupported"
            or qualification_source != "report_only"
        ) and not evidence_extractor:
            raise ValueError(
                "supported cpSatSupport requires a stable evidence extractor"
            )
        if qualification_source == "construction" and search_status != "exact":
            raise ValueError(
                "construction qualification requires exact search support"
            )
        if qualification_source in {"derived_state", "proof"} and search_status != "derived":
            raise ValueError(
                f"{qualification_source} qualification requires derived search support"
            )
        if search_status == "unsupported" and any(
            status != "unsupported"
            for status in (exact_proof, bounded_proof, policy_proof)
        ):
            raise ValueError(
                "proof support requires exact or derived search support"
            )
        return cls(
            search_status,
            search_stages,
            exact_input_status,
            exact_input_stages,
            qualification_source,
            actor_information_set,
            exact_proof,
            bounded_proof,
            policy_proof,
            evidence_extractor,
        )

    @staticmethod
    def _build_stage_support(
        value: Any,
        path: str,
        statuses: frozenset[str],
    ) -> tuple[str, tuple[str, ...]]:
        if not isinstance(value, Mapping):
            raise TypeError(f"{path} must be an object")
        raw = dict(value)
        required = {"status", "stages"}
        missing = sorted(required - set(raw))
        unknown = sorted(set(raw) - required)
        if missing or unknown:
            raise ValueError(
                f"{path} shape mismatch; missing={missing}, unknown={unknown}"
            )
        status = _strict_string(raw["status"], f"{path}.status")
        if status not in statuses:
            raise ValueError(f"{path}.status is invalid")
        stages = _strict_string_list(
            raw["stages"],
            f"{path}.stages",
            allowed=set(SCENARIO_SECTIONS),
            allow_empty=True,
        )
        if status == "unsupported" and stages:
            raise ValueError(f"{path}.stages must be empty when unsupported")
        if status != "unsupported" and not stages:
            raise ValueError(f"{path}.stages must be non-empty when supported")
        return status, stages

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "search": {
                "status": self.search_status,
                "stages": list(self.search_stages),
            },
            "exactInput": {
                "status": self.exact_input_status,
                "stages": list(self.exact_input_stages),
            },
            "qualificationSource": self.qualification_source,
            "actorInformationSet": self.actor_information_set,
            "proof": {
                "exact": self.exact_proof,
                "bounded": self.bounded_proof,
                "policyConditioned": self.policy_conditioned_proof,
            },
            "evidenceExtractor": self.evidence_extractor,
        }


@dataclass(frozen=True, slots=True)
class CardSelectorV1:
    """Closed, canonical selector shared by action and trick predicates."""

    cards: tuple[int, ...]
    ranks: tuple[str, ...]
    suits: tuple[str, ...]
    ekka: str
    owner_relation: str | None
    owner_seat: str | None
    owner_team: str | None
    sha256: str

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "CardSelectorV1":
        if not isinstance(value, Mapping):
            raise TypeError("card selector must be an object")
        raw = copy.deepcopy(dict(value))
        allowed = {"cards", "ranks", "suits", "ekka", "owner"}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"card selector contains unknown fields: {unknown}")

        cards = cls._card_ids(raw.get("cards"))
        ranks = cls._enum_items(raw.get("ranks"), "ranks", _RANKS)
        suits = cls._enum_items(raw.get("suits"), "suits", _SUITS)
        ekka = raw.get("ekka", "any")
        if type(ekka) is not str:
            raise TypeError("card selector ekka must be a string")
        if ekka not in {"any", "required", "forbidden"}:
            raise ValueError("card selector ekka is invalid")

        owner_relation: str | None = None
        owner_seat: str | None = None
        owner_team: str | None = None
        if "owner" in raw:
            owner = raw["owner"]
            if not isinstance(owner, Mapping):
                raise TypeError("card selector owner must be an object")
            owner_raw = dict(owner)
            owner_allowed = {"relation", "seat", "team"}
            owner_unknown = sorted(set(owner_raw) - owner_allowed)
            owner_missing = [] if "relation" in owner_raw else ["relation"]
            if owner_missing or owner_unknown:
                raise ValueError(
                    "card selector owner shape mismatch; "
                    f"missing={owner_missing}, unknown={owner_unknown}"
                )
            owner_relation = _strict_string(
                owner_raw["relation"], "card selector owner.relation"
            )
            if owner_relation not in _OWNER_RELATIONS:
                raise ValueError("card selector owner relation is invalid")
            if owner_relation == "seat":
                if set(owner_raw) != {"relation", "seat"}:
                    raise ValueError("seat owner relation requires only seat")
                owner_seat = _strict_string(
                    owner_raw["seat"], "card selector owner.seat"
                )
                if owner_seat not in _SEATS:
                    raise ValueError("card selector owner seat is invalid")
            elif owner_relation == "team":
                if set(owner_raw) != {"relation", "team"}:
                    raise ValueError("team owner relation requires only team")
                owner_team = _strict_string(
                    owner_raw["team"], "card selector owner.team"
                )
                if owner_team not in _TEAMS:
                    raise ValueError("card selector owner team is invalid")
            elif set(owner_raw) != {"relation"}:
                raise ValueError(
                    f"{owner_relation} owner relation does not accept seat or team"
                )

        if not cards and not ranks and not suits and ekka == "any" and not owner_relation:
            raise ValueError("card selector must contain at least one effective filter")
        core: dict[str, Any] = {
            "schemaVersion": 1,
            "cards": list(cards),
            "ranks": list(ranks),
            "suits": list(suits),
            "ekka": ekka,
            "owner": (
                None
                if owner_relation is None
                else {
                    "relation": owner_relation,
                    **({"seat": owner_seat} if owner_seat is not None else {}),
                    **({"team": owner_team} if owner_team is not None else {}),
                }
            ),
        }
        return cls(
            cards,
            ranks,
            suits,
            ekka,
            owner_relation,
            owner_seat,
            owner_team,
            canonical_sha256(core),
        )

    @staticmethod
    def _card_ids(value: Any) -> tuple[int, ...]:
        if value is None:
            return ()
        if type(value) is not list:
            raise TypeError("card selector cards must be an array")
        if not value:
            raise ValueError("card selector cards must be non-empty when supplied")
        if any(type(item) is not int for item in value):
            raise TypeError("card selector card IDs must be integers")
        cards = tuple(value)
        if any(item < 0 or item > 31 for item in cards):
            raise ValueError("card selector card ID is outside the Baloot deck")
        if len(set(cards)) != len(cards):
            raise ValueError("card selector card IDs must be unique")
        return cards

    @staticmethod
    def _enum_items(
        value: Any,
        name: str,
        allowed: tuple[str, ...],
    ) -> tuple[str, ...]:
        if value is None:
            return ()
        if type(value) is not list:
            raise TypeError(f"card selector {name} must be an array")
        if not value:
            raise ValueError(f"card selector {name} must be non-empty when supplied")
        if any(type(item) is not str for item in value):
            raise TypeError(f"card selector {name} values must be strings")
        items = tuple(value)
        if len(set(items)) != len(items):
            raise ValueError(f"card selector {name} values must be unique")
        if not set(items).issubset(allowed):
            raise ValueError(f"card selector {name} contains an invalid value")
        return items

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"ekka": self.ekka}
        if self.cards:
            value["cards"] = list(self.cards)
        if self.ranks:
            value["ranks"] = list(self.ranks)
        if self.suits:
            value["suits"] = list(self.suits)
        if self.owner_relation is not None:
            value["owner"] = {
                "relation": self.owner_relation,
                **({"seat": self.owner_seat} if self.owner_seat is not None else {}),
                **({"team": self.owner_team} if self.owner_team is not None else {}),
            }
        return value

    @classmethod
    def argument_schema(cls) -> dict[str, Any]:
        def localized(schema: Mapping[str, Any], key: str) -> dict[str, Any]:
            return {
                **copy.deepcopy(dict(schema)),
                "x-messageArKey": key,
                "x-messageEnKey": f"{key}_en",
            }

        def enum_schema(values: tuple[str, ...], key: str) -> dict[str, Any]:
            return localized(
                {
                    "type": "string",
                    "enum": list(values),
                    "x-enumMessageArKeys": {
                        item: f"research_scenario_option_{item}" for item in values
                    },
                    "x-enumMessageEnKeys": {
                        item: f"research_scenario_option_{item}_en" for item in values
                    },
                },
                key,
            )

        rank_item = enum_schema(_RANKS, "research_scenario_argument_rank")
        suit_item = enum_schema(_SUITS, "research_scenario_argument_suit")
        owner_relation = enum_schema(
            _OWNER_RELATIONS, "research_scenario_argument_relation"
        )
        seat = enum_schema(_SEATS, "research_scenario_argument_seat")
        team = enum_schema(_TEAMS, "research_scenario_argument_team")
        ekka = enum_schema(
            ("any", "required", "forbidden"), "research_scenario_argument_ekka"
        )
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "cards": localized(
                    {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0, "maximum": 31},
                        "minItems": 1,
                        "maxItems": 32,
                        "uniqueItems": True,
                    },
                    "research_scenario_argument_cards",
                ),
                "ranks": localized(
                    {
                        "type": "array",
                        "items": rank_item,
                        "minItems": 1,
                        "maxItems": len(_RANKS),
                        "uniqueItems": True,
                        "x-enumMessageArKeys": {
                            item: f"research_scenario_option_{item}"
                            for item in _RANKS
                        },
                        "x-enumMessageEnKeys": {
                            item: f"research_scenario_option_{item}_en"
                            for item in _RANKS
                        },
                    },
                    "research_scenario_argument_rank",
                ),
                "suits": localized(
                    {
                        "type": "array",
                        "items": suit_item,
                        "minItems": 1,
                        "maxItems": len(_SUITS),
                        "uniqueItems": True,
                        "x-enumMessageArKeys": {
                            item: f"research_scenario_option_{item}"
                            for item in _SUITS
                        },
                        "x-enumMessageEnKeys": {
                            item: f"research_scenario_option_{item}_en"
                            for item in _SUITS
                        },
                    },
                    "research_scenario_argument_suit",
                ),
                "ekka": ekka,
                "owner": localized(
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["relation"],
                        "properties": {
                            "relation": owner_relation,
                            "seat": seat,
                            "team": team,
                        },
                    },
                    "research_scenario_argument_owner",
                ),
            },
        }


@dataclass(frozen=True, slots=True)
class ScenarioPredicateDefinitionV2:
    kind: str
    version: int
    category: str
    stage: str
    argument_schema: Mapping[str, Any]
    message_ar_key: str
    message_en_key: str
    required_domain_primitives: tuple[str, ...]
    allowed_hardness: tuple[str, ...]
    cp_sat_support: CpSatSupportV1
    cost_class: str
    safe_complement: bool
    examples: tuple[Mapping[str, Any], ...]
    counterexamples: tuple[Mapping[str, Any], ...]
    documentation: tuple[str, ...]
    tests: tuple[str, ...]
    lifecycle_status: str
    sha256: str

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "ScenarioPredicateDefinitionV2":
        if not isinstance(value, Mapping):
            raise TypeError("scenario predicate definition V2 must be an object")
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
            "cpSatSupport",
            "costClass",
            "safeComplement",
            "examples",
            "counterexamples",
            "documentation",
            "tests",
            "lifecycleStatus",
        }
        missing = sorted(required - set(raw))
        unknown = sorted(set(raw) - {*required, "sha256"})
        if missing or unknown:
            raise ValueError(
                "scenario predicate definition V2 shape mismatch; "
                f"missing={missing}, unknown={unknown}"
            )
        kind = _strict_string(raw["kind"], "scenario predicate kind")
        if type(raw["version"]) is not int or raw["version"] < 1:
            raise ValueError("scenario predicate definition V2 version is invalid")
        category = _strict_string(raw["category"], "scenario predicate category")
        stage = _strict_string(raw["stage"], "scenario predicate stage")
        if stage not in SCENARIO_SECTIONS or category != stage:
            raise ValueError("scenario predicate definition V2 stage is invalid")
        if not isinstance(raw["argumentSchema"], Mapping):
            raise TypeError("scenario predicate argumentSchema must be an object")
        argument_schema = copy.deepcopy(dict(raw["argumentSchema"]))
        if (
            argument_schema.get("type") != "object"
            or argument_schema.get("additionalProperties") is not False
        ):
            raise ValueError("scenario predicate argumentSchema must be closed")
        _validate_schema_localization(argument_schema, f"{kind}.arguments")
        message_ar_key = _strict_string(
            raw["messageArKey"], "scenario predicate messageArKey"
        )
        message_en_key = _strict_string(
            raw["messageEnKey"], "scenario predicate messageEnKey"
        )
        required_primitives = _strict_string_list(
            raw["requiredDomainPrimitives"],
            "scenario predicate requiredDomainPrimitives",
            allow_empty=True,
        )
        allowed_hardness = _strict_string_list(
            raw["allowedHardness"],
            "scenario predicate allowedHardness",
            allowed=set(PREDICATE_HARDNESS),
        )
        cp_sat_support = CpSatSupportV1.build(raw["cpSatSupport"])
        cost_class = _strict_string(raw["costClass"], "scenario predicate costClass")
        if cost_class not in {
            "constant",
            "linear",
            "solver",
            "tree_search",
            "hidden_world_search",
        }:
            raise ValueError("scenario predicate costClass is invalid")
        if type(raw["safeComplement"]) is not bool:
            raise TypeError("scenario predicate safeComplement must be boolean")
        lifecycle = _strict_string(
            raw["lifecycleStatus"], "scenario predicate lifecycleStatus"
        )
        if lifecycle not in {"active", "deprecated", "retired"}:
            raise ValueError("scenario predicate lifecycleStatus is invalid")

        examples = cls._example_rows(raw["examples"], "examples")
        counterexamples = cls._example_rows(raw["counterexamples"], "counterexamples")
        documentation = _strict_string_list(
            raw["documentation"], "scenario predicate documentation"
        )
        tests = _strict_string_list(raw["tests"], "scenario predicate tests")
        for index, example in enumerate(examples):
            if (
                example.get("kind") != kind
                or example.get("version") != raw["version"]
                or example.get("hardness") not in allowed_hardness
            ):
                raise ValueError(
                    f"scenario predicate {kind} example {index} identity drifted"
                )
            cls._validate_arguments(kind, example.get("arguments"), argument_schema)
        for index, counterexample in enumerate(counterexamples):
            try:
                cls._validate_arguments(
                    kind, counterexample.get("arguments"), argument_schema
                )
            except (TypeError, ValueError):
                continue
            raise ValueError(
                f"scenario predicate {kind} counterexample {index} is valid"
            )

        core = {
            "kind": kind,
            "version": raw["version"],
            "category": category,
            "stage": stage,
            "argumentSchema": argument_schema,
            "messageArKey": message_ar_key,
            "messageEnKey": message_en_key,
            "requiredDomainPrimitives": list(required_primitives),
            "allowedHardness": list(allowed_hardness),
            "cpSatSupport": cp_sat_support.to_dict(),
            "costClass": cost_class,
            "safeComplement": raw["safeComplement"],
            "examples": [copy.deepcopy(dict(item)) for item in examples],
            "counterexamples": [
                copy.deepcopy(dict(item)) for item in counterexamples
            ],
            "documentation": list(documentation),
            "tests": list(tests),
            "lifecycleStatus": lifecycle,
        }
        digest = canonical_sha256(core)
        declared = raw.get("sha256")
        if declared is not None:
            declared = _strict_string(declared, "scenario predicate sha256")
            if declared != digest:
                raise ValueError("scenario predicate definition V2 checksum mismatch")
        return cls(
            kind,
            raw["version"],
            category,
            stage,
            argument_schema,
            message_ar_key,
            message_en_key,
            required_primitives,
            allowed_hardness,
            cp_sat_support,
            cost_class,
            raw["safeComplement"],
            examples,
            counterexamples,
            documentation,
            tests,
            lifecycle,
            digest,
        )

    @staticmethod
    def _example_rows(value: Any, field: str) -> tuple[Mapping[str, Any], ...]:
        if type(value) is not list or not value:
            raise ValueError(f"scenario predicate {field} must be a non-empty array")
        if any(not isinstance(item, Mapping) for item in value):
            raise TypeError(f"scenario predicate {field} entries must be objects")
        return tuple(copy.deepcopy(dict(item)) for item in value)

    @staticmethod
    def _validate_arguments(
        kind: str,
        value: Any,
        schema: Mapping[str, Any],
    ) -> None:
        _validate_schema_value(value, schema, f"{kind}.arguments")
        arguments = dict(value)
        if kind in {
            "play.action_matches",
            "play.trick_winner_matches",
            "play.trick_card_count",
            "proof.winning_root_action_matches",
        }:
            CardSelectorV1.build(arguments["selector"])
        if kind == "play.trick_winner_matches" and {
            "seat",
            "team",
        } <= set(arguments):
            raise ValueError(
                "play.trick_winner_matches accepts either seat or team, not both"
            )
        if kind in {
            "play.trick_card_count",
            "proof.winning_root_action_count",
        } and int(arguments["minimum"]) > int(arguments["maximum"]):
            raise ValueError(f"{kind} minimum cannot exceed maximum")

    def publication_issues(self) -> tuple[str, ...]:
        issues: list[str] = []
        support = self.cp_sat_support
        if self.lifecycle_status != "active":
            issues.append("lifecycle_not_active")
        if support.search_status == "unsupported":
            issues.append("search_not_exactly_supported")
        elif self.stage not in support.search_stages:
            issues.append("search_stage_missing")
        if support.exact_input_status != "exact":
            issues.append("exact_input_not_supported")
        elif self.stage not in support.exact_input_stages:
            issues.append("exact_input_stage_missing")
        if not support.evidence_extractor:
            issues.append("evidence_extractor_missing")
        if "hard" in self.allowed_hardness and support.qualification_source in {
            "report_only"
        }:
            issues.append("hard_qualification_source_invalid")
        if "hard" in self.allowed_hardness and support.search_status not in {
            "exact",
            "derived",
        }:
            issues.append("hard_search_support_invalid")
        if "hard" in self.allowed_hardness and all(
            status == "unsupported"
            for status in (
                support.exact_proof,
                support.bounded_proof,
                support.policy_conditioned_proof,
            )
        ):
            issues.append("hard_proof_compatibility_missing")
        return tuple(issues)

    def require_publishable(self) -> None:
        issues = self.publication_issues()
        if issues:
            raise ValueError(
                f"scenario predicate {self.kind} is not publishable: "
                + ", ".join(issues)
            )

    def validate_arguments(self, value: Mapping[str, Any]) -> None:
        self._validate_arguments(self.kind, value, self.argument_schema)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 2,
            "kind": self.kind,
            "version": self.version,
            "category": self.category,
            "stage": self.stage,
            "argumentSchema": copy.deepcopy(dict(self.argument_schema)),
            "messageArKey": self.message_ar_key,
            "messageEnKey": self.message_en_key,
            "requiredDomainPrimitives": list(self.required_domain_primitives),
            "allowedHardness": list(self.allowed_hardness),
            "cpSatSupport": self.cp_sat_support.to_dict(),
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


# These are the only V1 kinds whose complete argument domains are directly and
# exactly lowered by ScenarioStructuralCompilerV1 today.  In particular,
# project.exists is excluded because non-eligible lifecycle states currently
# lower only to a necessary card pattern.
EXACT_V1_COMPATIBILITY_KINDS = frozenset(
    {
        "deal.card_owner",
        "deal.card_allowed",
        "deal.card_forbidden",
        "deal.suit_count",
        "deal.rank_count",
        "deal.high_card_count",
        "deal.low_card_count",
        "deal.purchase_card",
        "deal.purchase_suit",
        "auction.dealer",
    }
)

_INFORMATION_CLASS_ADAPTER = {
    "public": "public",
    "actor_visible": "observation",
    "complete_state_only": "reference_world_only",
    "information_set_safe": "nonanticipative",
}


def _cp_sat_support_payload(
    definition: ScenarioPredicateDefinitionV1,
    *,
    exact: bool,
) -> dict[str, Any]:
    if exact:
        return {
            "schemaVersion": 1,
            "search": {"status": "exact", "stages": [definition.stage]},
            "exactInput": {"status": "exact", "stages": [definition.stage]},
            "qualificationSource": "construction",
            "actorInformationSet": _INFORMATION_CLASS_ADAPTER[
                definition.hidden_information
            ],
            "proof": {
                "exact": "exact",
                "bounded": "exact",
                "policyConditioned": "exact",
            },
            "evidenceExtractor": "scenario_structural_compiler_v1.predicate_evidence",
        }
    return {
        "schemaVersion": 1,
        "search": {"status": "unsupported", "stages": []},
        "exactInput": {"status": "unsupported", "stages": []},
        "qualificationSource": "report_only",
        "actorInformationSet": _INFORMATION_CLASS_ADAPTER[
            definition.hidden_information
        ],
        "proof": {
            "exact": "unsupported",
            "bounded": "unsupported",
            "policyConditioned": "unsupported",
        },
        "evidenceExtractor": "",
    }


def adapt_scenario_predicate_definition_v1(
    definition: ScenarioPredicateDefinitionV1,
) -> ScenarioPredicateDefinitionV2:
    """Adapt one immutable V1 definition without overstating CP-SAT support."""

    exact = definition.kind in EXACT_V1_COMPATIBILITY_KINDS
    hardness = list(definition.allowed_hardness)
    if not exact:
        hardness = [item for item in hardness if item != "hard"]
    if not hardness:
        hardness = ["report"]
    example_hardness = hardness[0]

    def adapt_rows(rows: tuple[Mapping[str, Any], ...]) -> list[dict[str, Any]]:
        result = [copy.deepcopy(dict(item)) for item in rows]
        for item in result:
            if item.get("hardness") not in hardness:
                item["hardness"] = example_hardness
        return result

    return ScenarioPredicateDefinitionV2.build(
        {
            "kind": definition.kind,
            "version": definition.version,
            "category": definition.category,
            "stage": definition.stage,
            "argumentSchema": copy.deepcopy(dict(definition.argument_schema)),
            "messageArKey": definition.message_ar_key,
            "messageEnKey": definition.message_en_key,
            "requiredDomainPrimitives": list(definition.required_domain_primitives),
            "allowedHardness": hardness,
            "cpSatSupport": _cp_sat_support_payload(definition, exact=exact),
            "costClass": definition.cost_class,
            "safeComplement": definition.safe_complement,
            "examples": adapt_rows(definition.examples),
            "counterexamples": adapt_rows(definition.counterexamples),
            "documentation": list(definition.documentation),
            "tests": list(definition.tests),
            "lifecycleStatus": definition.lifecycle_status,
        }
    )


@lru_cache(maxsize=1)
def compatible_scenario_predicates_v2() -> tuple[ScenarioPredicateDefinitionV2, ...]:
    """Return all V1 definitions adapted conservatively to the V2 contract."""

    return tuple(
        adapt_scenario_predicate_definition_v1(definition)
        for definition in built_in_scenario_predicates_v1()
    )


def validate_predicate_publication_v2(
    definitions: tuple[ScenarioPredicateDefinitionV2, ...],
) -> str:
    """Validate a complete selected publication set and return its stable hash."""

    if not definitions:
        raise ValueError("scenario predicate V2 publication cannot be empty")
    kinds = [definition.kind for definition in definitions]
    if len(kinds) != len(set(kinds)):
        raise ValueError("scenario predicate V2 publication contains duplicate kinds")
    failures: dict[str, tuple[str, ...]] = {}
    for definition in definitions:
        issues = definition.publication_issues()
        if issues:
            failures[definition.kind] = issues
    if failures:
        detail = "; ".join(
            f"{kind}={','.join(failures[kind])}" for kind in sorted(failures)
        )
        raise ValueError(f"scenario predicate V2 publication is incomplete: {detail}")
    core = {
        "schemaVersion": 2,
        "registryVersion": "scenario_predicate_registry_v2",
        "definitions": [
            definition.to_dict() for definition in sorted(definitions, key=lambda row: row.kind)
        ],
    }
    return canonical_sha256(core)


def play_action_matches_definition_v2() -> ScenarioPredicateDefinitionV2:
    """Define the reusable action/card-selector predicate before its lowering."""

    def localized_integer(key: str, maximum: int) -> dict[str, Any]:
        return {
            "type": "integer",
            "minimum": 0,
            "maximum": maximum,
            "x-messageArKey": key,
            "x-messageEnKey": f"{key}_en",
        }

    selector_schema = CardSelectorV1.argument_schema()
    selector_schema.update(
        {
            "x-messageArKey": "research_scenario_argument_card_selector",
            "x-messageEnKey": "research_scenario_argument_card_selector_en",
        }
    )
    seat_schema = {
        "type": "string",
        "enum": list(_SEATS),
        "x-messageArKey": "research_scenario_argument_actor",
        "x-messageEnKey": "research_scenario_argument_actor_en",
        "x-enumMessageArKeys": {
            seat: f"research_scenario_option_seat_{seat}" for seat in _SEATS
        },
        "x-enumMessageEnKeys": {
            seat: f"research_scenario_option_seat_{seat}_en" for seat in _SEATS
        },
    }
    argument_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["trickIndex", "position", "selector"],
        "properties": {
            "trickIndex": localized_integer(
                "research_scenario_argument_trick_index", 7
            ),
            "position": localized_integer("research_scenario_argument_position", 3),
            "actingSeat": seat_schema,
            "selector": selector_schema,
        },
    }
    return ScenarioPredicateDefinitionV2.build(
        {
            "kind": "play.action_matches",
            "version": 1,
            "category": "playPrefix",
            "stage": "playPrefix",
            "argumentSchema": argument_schema,
            "messageArKey": "research_scenario_predicate_play_action_matches",
            "messageEnKey": "research_scenario_predicate_play_action_matches_en",
            "requiredDomainPrimitives": ["rules.play.must_follow_suit"],
            "allowedHardness": ["report"],
            "cpSatSupport": {
                "schemaVersion": 1,
                "search": {"status": "unsupported", "stages": []},
                "exactInput": {"status": "unsupported", "stages": []},
                "qualificationSource": "report_only",
                "actorInformationSet": "reference_world_only",
                "proof": {
                    "exact": "unsupported",
                    "bounded": "unsupported",
                    "policyConditioned": "unsupported",
                },
                "evidenceExtractor": "",
            },
            "costClass": "linear",
            "safeComplement": False,
            "examples": [
                {
                    "kind": "play.action_matches",
                    "version": 1,
                    "hardness": "report",
                    "arguments": {
                        "trickIndex": 5,
                        "position": 3,
                        "actingSeat": "west",
                        "selector": {
                            "ranks": ["ace"],
                            "owner": {"relation": "actor"},
                        },
                    },
                }
            ],
            "counterexamples": [
                {
                    "kind": "play.action_matches",
                    "version": 1,
                    "hardness": "report",
                    "arguments": {
                        "trickIndex": 5,
                        "position": 3,
                        "selector": {},
                    },
                }
            ],
            "documentation": ["docs/RESEARCH_SCENARIO_V4.md"],
            "tests": ["tests/test_research_scenario_registry_v2.py"],
            "lifecycleStatus": "active",
        }
    )


def _localized_enum_schema(
    values: tuple[str, ...],
    argument_key: str,
    option_prefix: str,
) -> dict[str, Any]:
    return {
        "type": "string",
        "enum": list(values),
        "x-messageArKey": argument_key,
        "x-messageEnKey": f"{argument_key}_en",
        "x-enumMessageArKeys": {
            value: f"{option_prefix}{value}" for value in values
        },
        "x-enumMessageEnKeys": {
            value: f"{option_prefix}{value}_en" for value in values
        },
    }


def _localized_integer_schema(
    argument_key: str,
    *,
    maximum: int,
) -> dict[str, Any]:
    return {
        "type": "integer",
        "minimum": 0,
        "maximum": maximum,
        "x-messageArKey": argument_key,
        "x-messageEnKey": f"{argument_key}_en",
    }


def _selector_argument_schema() -> dict[str, Any]:
    schema = CardSelectorV1.argument_schema()
    schema.update(
        {
            "x-messageArKey": "research_scenario_argument_card_selector",
            "x-messageEnKey": "research_scenario_argument_card_selector_en",
        }
    )
    return schema


def play_trick_winner_matches_definition_v2() -> ScenarioPredicateDefinitionV2:
    return ScenarioPredicateDefinitionV2.build(
        {
            "kind": "play.trick_winner_matches",
            "version": 1,
            "category": "playPrefix",
            "stage": "playPrefix",
            "argumentSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["trickIndex", "selector"],
                "properties": {
                    "trickIndex": _localized_integer_schema(
                        "research_scenario_argument_trick_index", maximum=7
                    ),
                    "seat": _localized_enum_schema(
                        _SEATS,
                        "research_scenario_argument_seat",
                        "research_scenario_option_seat_",
                    ),
                    "team": _localized_enum_schema(
                        _TEAMS,
                        "research_scenario_argument_team",
                        "research_scenario_option_team_",
                    ),
                    "selector": _selector_argument_schema(),
                },
            },
            "messageArKey": "research_scenario_predicate_play_trick_winner_matches",
            "messageEnKey": "research_scenario_predicate_play_trick_winner_matches_en",
            "requiredDomainPrimitives": ["rules.play.trick_winner"],
            "allowedHardness": ["report"],
            "cpSatSupport": {
                "schemaVersion": 1,
                "search": {"status": "unsupported", "stages": []},
                "exactInput": {"status": "unsupported", "stages": []},
                "qualificationSource": "report_only",
                "actorInformationSet": "reference_world_only",
                "proof": {
                    "exact": "unsupported",
                    "bounded": "unsupported",
                    "policyConditioned": "unsupported",
                },
                "evidenceExtractor": "",
            },
            "costClass": "linear",
            "safeComplement": False,
            "examples": [
                {
                    "kind": "play.trick_winner_matches",
                    "version": 1,
                    "hardness": "report",
                    "arguments": {
                        "trickIndex": 6,
                        "team": "west_east",
                        "selector": {"ranks": ["seven", "eight", "nine"]},
                    },
                }
            ],
            "counterexamples": [
                {
                    "kind": "play.trick_winner_matches",
                    "version": 1,
                    "hardness": "report",
                    "arguments": {
                        "trickIndex": 6,
                        "seat": "west",
                        "team": "west_east",
                        "selector": {"ranks": ["seven"]},
                    },
                }
            ],
            "documentation": ["docs/RESEARCH_SCENARIO_V4.md"],
            "tests": ["tests/test_research_scenario_predicate_lowering_v2.py"],
            "lifecycleStatus": "active",
        }
    )


def play_trick_card_count_definition_v2() -> ScenarioPredicateDefinitionV2:
    return ScenarioPredicateDefinitionV2.build(
        {
            "kind": "play.trick_card_count",
            "version": 1,
            "category": "playPrefix",
            "stage": "playPrefix",
            "argumentSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["trickIndex", "selector", "minimum", "maximum"],
                "properties": {
                    "trickIndex": _localized_integer_schema(
                        "research_scenario_argument_trick_index", maximum=7
                    ),
                    "selector": _selector_argument_schema(),
                    "minimum": _localized_integer_schema(
                        "research_scenario_argument_minimum", maximum=4
                    ),
                    "maximum": _localized_integer_schema(
                        "research_scenario_argument_maximum", maximum=4
                    ),
                },
            },
            "messageArKey": "research_scenario_predicate_play_trick_card_count",
            "messageEnKey": "research_scenario_predicate_play_trick_card_count_en",
            "requiredDomainPrimitives": ["rules.play.trick_cards"],
            "allowedHardness": ["report"],
            "cpSatSupport": {
                "schemaVersion": 1,
                "search": {"status": "unsupported", "stages": []},
                "exactInput": {"status": "unsupported", "stages": []},
                "qualificationSource": "report_only",
                "actorInformationSet": "reference_world_only",
                "proof": {
                    "exact": "unsupported",
                    "bounded": "unsupported",
                    "policyConditioned": "unsupported",
                },
                "evidenceExtractor": "",
            },
            "costClass": "linear",
            "safeComplement": True,
            "examples": [
                {
                    "kind": "play.trick_card_count",
                    "version": 1,
                    "hardness": "report",
                    "arguments": {
                        "trickIndex": 7,
                        "selector": {
                            "ranks": ["ace"],
                            "owner": {
                                "relation": "team",
                                "team": "north_south",
                            },
                        },
                        "minimum": 2,
                        "maximum": 2,
                    },
                }
            ],
            "counterexamples": [
                {
                    "kind": "play.trick_card_count",
                    "version": 1,
                    "hardness": "report",
                    "arguments": {
                        "trickIndex": 7,
                        "selector": {"ranks": ["ace"]},
                        "minimum": 3,
                        "maximum": 2,
                    },
                }
            ],
            "documentation": ["docs/RESEARCH_SCENARIO_V4.md"],
            "tests": ["tests/test_research_scenario_predicate_lowering_v2.py"],
            "lifecycleStatus": "active",
        }
    )


def proof_winning_root_action_matches_definition_v2() -> ScenarioPredicateDefinitionV2:
    return ScenarioPredicateDefinitionV2.build(
        {
            "kind": "proof.winning_root_action_matches",
            "version": 1,
            "category": "proof",
            "stage": "proof",
            "argumentSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["selector"],
                "properties": {"selector": _selector_argument_schema()},
            },
            "messageArKey": "research_scenario_predicate_proof_winning_root_action_matches",
            "messageEnKey": "research_scenario_predicate_proof_winning_root_action_matches_en",
            "requiredDomainPrimitives": ["rules.proof.root_actions"],
            "allowedHardness": ["report"],
            "cpSatSupport": {
                "schemaVersion": 1,
                "search": {"status": "unsupported", "stages": []},
                "exactInput": {"status": "unsupported", "stages": []},
                "qualificationSource": "report_only",
                "actorInformationSet": "reference_world_only",
                "proof": {
                    "exact": "unsupported",
                    "bounded": "unsupported",
                    "policyConditioned": "unsupported",
                },
                "evidenceExtractor": "",
            },
            "costClass": "tree_search",
            "safeComplement": True,
            "examples": [
                {
                    "kind": "proof.winning_root_action_matches",
                    "version": 1,
                    "hardness": "report",
                    "arguments": {"selector": {"ranks": ["ace"]}},
                }
            ],
            "counterexamples": [
                {
                    "kind": "proof.winning_root_action_matches",
                    "version": 1,
                    "hardness": "report",
                    "arguments": {"selector": {}},
                }
            ],
            "documentation": ["docs/RESEARCH_SCENARIO_V4.md"],
            "tests": ["tests/test_research_scenario_predicate_lowering_v2.py"],
            "lifecycleStatus": "active",
        }
    )


def proof_winning_root_action_count_definition_v2() -> ScenarioPredicateDefinitionV2:
    source = next(
        definition
        for definition in compatible_scenario_predicates_v2()
        if definition.kind == "proof.winning_root_action_count"
    )
    return source


FLAGSHIP_PREDICATE_KINDS_V2 = frozenset(
    {
        "play.action_matches",
        "play.trick_winner_matches",
        "play.trick_card_count",
        "proof.winning_root_action_matches",
        "proof.winning_root_action_count",
    }
)
GENERAL_PREDICATE_KINDS_V2 = FLAGSHIP_PREDICATE_KINDS_V2 - {
    "proof.winning_root_action_count"
}


@lru_cache(maxsize=1)
def flagship_predicate_definitions_v2() -> tuple[ScenarioPredicateDefinitionV2, ...]:
    """Return the fail-closed catalog view before lowerings are registered."""

    return (
        play_action_matches_definition_v2(),
        play_trick_winner_matches_definition_v2(),
        play_trick_card_count_definition_v2(),
        proof_winning_root_action_matches_definition_v2(),
        proof_winning_root_action_count_definition_v2(),
    )


@lru_cache(maxsize=1)
def scenario_predicate_definitions_v2() -> tuple[ScenarioPredicateDefinitionV2, ...]:
    """Merge legacy adapters and fail-closed flagship definitions without duplicates."""

    rows = {
        definition.kind: definition
        for definition in compatible_scenario_predicates_v2()
    }
    rows.update(
        {definition.kind: definition for definition in flagship_predicate_definitions_v2()}
    )
    return tuple(rows[kind] for kind in sorted(rows))


def registered_scenario_predicate_definitions_v2(
    evidence_extractors: Mapping[str, str],
    owning_tests: Mapping[str, str],
) -> tuple[ScenarioPredicateDefinitionV2, ...]:
    """Bind exact support only to complete lowering/extractor/test registrations."""

    definitions = {
        definition.kind: definition for definition in scenario_predicate_definitions_v2()
    }
    extractor_kinds = set(evidence_extractors)
    test_kinds = set(owning_tests)
    if extractor_kinds != test_kinds:
        raise ValueError(
            "predicate lowering registrations require matching extractor and test sets"
        )
    unknown = sorted(extractor_kinds - set(definitions))
    if unknown:
        raise ValueError(f"unknown scenario predicate registrations: {unknown}")
    legacy_hardness = {
        definition.kind: definition.allowed_hardness
        for definition in built_in_scenario_predicates_v1()
    }
    rows: list[ScenarioPredicateDefinitionV2] = []
    for definition in definitions.values():
        extractor = evidence_extractors.get(definition.kind)
        if extractor is None:
            rows.append(definition)
            continue
        extractor = _strict_string(
            extractor, f"{definition.kind} evidence extractor"
        )
        test_path = _strict_string(
            owning_tests[definition.kind], f"{definition.kind} owning test"
        )
        payload = definition.to_dict()
        payload.pop("schemaVersion")
        payload.pop("sha256")
        payload["allowedHardness"] = list(
            ("hard", "report")
            if definition.kind == "state.observer_visibility"
            else legacy_hardness.get(definition.kind, ("hard", "report"))
        )
        if "hard" not in payload["allowedHardness"]:
            raise ValueError(
                f"registered predicate {definition.kind} is not hard-capable"
            )
        tests = list(payload["tests"])
        if test_path not in tests:
            tests.append(test_path)
        payload["tests"] = tests
        is_derived = definition.stage in {
            "state",
            "outcome",
            "tactics",
            "proof",
        } or definition.kind in _REMAINING_DEAL_KINDS
        if definition.stage in {"tactics", "proof"}:
            qualification_source = "proof"
        elif is_derived:
            qualification_source = "derived_state"
        else:
            qualification_source = "construction"
        payload["cpSatSupport"] = {
            "schemaVersion": 1,
            "search": {
                "status": "derived" if is_derived else "exact",
                "stages": (
                    [definition.stage, "playPrefix"]
                    if definition.kind in _REMAINING_DEAL_KINDS
                    else [definition.stage]
                ),
            },
            "exactInput": {
                "status": "exact",
                "stages": (
                    [definition.stage, "playPrefix"]
                    if definition.kind in _REMAINING_DEAL_KINDS
                    else [definition.stage]
                ),
            },
            "qualificationSource": qualification_source,
            "actorInformationSet": definition.cp_sat_support.actor_information_set,
            "proof": {
                "exact": "exact",
                "bounded": "exact",
                "policyConditioned": "exact",
            },
            "evidenceExtractor": extractor,
        }
        rows.append(ScenarioPredicateDefinitionV2.build(payload))
    return tuple(sorted(rows, key=lambda row: row.kind))


def _registered_flagship_predicate_definitions_v2(
    evidence_extractors: Mapping[str, str],
) -> tuple[ScenarioPredicateDefinitionV2, ...]:
    """Upgrade only kinds possessing a concrete lowering/extractor registration."""

    unknown = sorted(set(evidence_extractors) - GENERAL_PREDICATE_KINDS_V2)
    if unknown:
        raise ValueError(f"unknown flagship predicate registrations: {unknown}")
    tests = {
        kind: "tests/test_research_scenario_predicate_lowering_v2.py"
        for kind in evidence_extractors
    }
    registered = {
        definition.kind: definition
        for definition in registered_scenario_predicate_definitions_v2(
            evidence_extractors,
            tests,
        )
    }
    return tuple(
        registered[kind] for kind in sorted(GENERAL_PREDICATE_KINDS_V2)
    )
