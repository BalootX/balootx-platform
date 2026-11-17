"""V3-only native adjudication predicates and merged dispatch authority.

The historical V2 catalog is never modified. This module adds finite
attempted-action events with native CP-SAT primitives and merges them with V2
at the service seam. False Ekka is exposed only by its unratified variant.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Mapping

from ortools.sat.python import cp_model

from baloot_platform.adjudication.research import AuthoredAdjudicationEventV1
from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Seat

from .ortools_adjudication_attempts import (
    AdjudicationAttemptKindV1,
    AdjudicationRemedyV1,
    PurchaseAttemptTokenV1,
    add_adjudication_attempt_branch_v1,
)
from .scenario_contracts import ScenarioPredicateV1, canonical_sha256
from .scenario_deal_predicates_v2 import (
    DealPredicateLoweringRegistryV2,
    NATIVE_SEAT_RELATION_IMPLEMENTATION,
)
from .scenario_predicate_capabilities_v3 import (
    ScenarioPredicateCapabilityExtensionV3,
    ScenarioPredicateCapabilityManifestV3,
    ScenarioPredicateCapabilityV3,
    ScenarioPredicateExecutionContextV3,
    native_scenario_predicate_definition_v3,
)
from .scenario_predicate_lowering_v2 import CpSatPredicateLoweringV1
from .scenario_predicate_manifest_v2 import (
    ScenarioPredicateCatalogLoweringManifestV2,
    ScenarioPredicateLoweringRegistrationV2,
    shared_scenario_predicate_catalog_lowering_manifest_v2,
)
from .scenario_registry_v2 import ScenarioPredicateDefinitionV2


ADJUDICATION_AUTHORITY_ID_V3 = "gea_baloot_2023_adjudication_research_v1"
ADJUDICATION_AUTHORITY_SHA256_V3 = (
    "685a24c8c51e0b099d5dda5fd080cbdb4bd45608c8b05382c5ed97c64fdc0b21"
)
OUT_OF_TURN_PREDICATE_V3 = "adjudication.out_of_turn_attempt"
PURCHASE_BEFORE_TURN_PREDICATE_V3 = (
    "adjudication.purchase_before_turn_attempt"
)
FALSE_EKKA_PREDICATE_V3 = "adjudication.false_ekka_attempt"

_SEATS = tuple(seat.name.lower() for seat in Seat)
_TOKENS = tuple(token.name.lower() for token in PurchaseAttemptTokenV1)
_OUT_OF_TURN_REMEDIES = ("continue_play", "opponent_register_with_projects")
_PURCHASE_REMEDIES = ("continue_play", "opponent_takweesh")
_FALSE_EKKA_REMEDIES = _OUT_OF_TURN_REMEDIES


def _enum_schema(values: tuple[str, ...], key: str) -> dict[str, Any]:
    return {
        "type": "string",
        "enum": list(values),
        "x-messageArKey": key,
        "x-messageEnKey": f"{key}_en",
        "x-enumMessageArKeys": {value: f"{key}_{value}" for value in values},
        "x-enumMessageEnKeys": {
            value: f"{key}_{value}_en" for value in values
        },
    }


def _definition(
    *,
    kind: str,
    category: str,
    action_name: str,
    action_schema: Mapping[str, Any],
    remedies: tuple[str, ...],
    primitive: str,
    extractor: str,
) -> ScenarioPredicateDefinitionV2:
    properties = {
        "attemptedActor": _enum_schema(
            _SEATS, "research_scenario_argument_attempted_actor"
        ),
        action_name: copy.deepcopy(dict(action_schema)),
        "requiredRemedy": _enum_schema(
            remedies, "research_scenario_argument_required_remedy"
        ),
    }
    arguments = {
        "attemptedActor": "west",
        action_name: 0 if action_name == "attemptedCardId" else "sun",
    }
    return ScenarioPredicateDefinitionV2.build(
        {
            "kind": kind,
            "version": 1,
            "category": category,
            "stage": category,
            "argumentSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["attemptedActor", action_name],
                "properties": properties,
            },
            "messageArKey": f"research_scenario_{kind.replace('.', '_')}",
            "messageEnKey": f"research_scenario_{kind.replace('.', '_')}_en",
            "requiredDomainPrimitives": [primitive],
            "allowedHardness": ["hard"],
            "cpSatSupport": {
                "schemaVersion": 1,
                "search": {"status": "exact", "stages": [category]},
                "exactInput": {"status": "exact", "stages": [category]},
                "qualificationSource": "construction",
                "actorInformationSet": "public",
                "proof": {
                    "exact": "exact",
                    "bounded": "bounds",
                    "policyConditioned": "exact",
                },
                "evidenceExtractor": extractor,
            },
            "costClass": "solver",
            "safeComplement": True,
            "examples": [
                {
                    "kind": kind,
                    "version": 1,
                    "hardness": "hard",
                    "arguments": arguments,
                }
            ],
            "counterexamples": [
                {
                    "kind": kind,
                    "version": 1,
                    "hardness": "hard",
                    "arguments": {"attemptedActor": "invalid"},
                }
            ],
            "documentation": ["docs/RESEARCH_SCENARIO_V4.md"],
            "tests": ["tests/test_research_scenario_adjudication_predicates_v3.py"],
            "lifecycleStatus": "active",
        }
    )


@lru_cache(maxsize=2)
def adjudication_predicate_definitions_v3(
    include_false_ekka: bool = False,
) -> tuple[ScenarioPredicateDefinitionV2, ...]:
    if type(include_false_ekka) is not bool:
        raise TypeError("include_false_ekka must be a boolean")
    definitions = [
        _definition(
            kind=OUT_OF_TURN_PREDICATE_V3,
            category="playPrefix",
            action_name="attemptedCardId",
            action_schema={
                "type": "integer",
                "minimum": 0,
                "maximum": 31,
                "x-messageArKey": "research_scenario_argument_attempted_card",
                "x-messageEnKey": "research_scenario_argument_attempted_card_en",
            },
            remedies=_OUT_OF_TURN_REMEDIES,
            primitive="rules.play.current_actor",
            extractor="cp_sat.adjudication_out_of_turn_attempt_evidence_v1",
        ),
        _definition(
            kind=PURCHASE_BEFORE_TURN_PREDICATE_V3,
            category="auction",
            action_name="purchaseToken",
            action_schema=_enum_schema(
                _TOKENS, "research_scenario_argument_purchase_token"
            ),
            remedies=_PURCHASE_REMEDIES,
            primitive="rules.auction.legal_actions",
            extractor=(
                "cp_sat.adjudication_purchase_before_turn_attempt_evidence_v1"
            ),
        ),
    ]
    if include_false_ekka:
        definitions.append(
            _definition(
                kind=FALSE_EKKA_PREDICATE_V3,
                category="playPrefix",
                action_name="attemptedCardId",
                action_schema={
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 31,
                    "x-messageArKey": "research_scenario_argument_attempted_card",
                    "x-messageEnKey": "research_scenario_argument_attempted_card_en",
                },
                remedies=_FALSE_EKKA_REMEDIES,
                primitive="rules.play.legal_actions",
                extractor="cp_sat.adjudication_false_ekka_attempt_evidence_v1",
            )
        )
    return tuple(definitions)


@dataclass(frozen=True, slots=True)
class AdjudicationPredicateModelContextV3:
    model: cp_model.CpModel
    context_id: str
    origin: str
    expected_actor: cp_model.IntVar
    attempted_actor: cp_model.IntVar
    attempted_action: cp_model.IntVar
    attempted_action_available: cp_model.IntVar
    exact_events: tuple[AuthoredAdjudicationEventV1, ...] = ()
    action_sequence: int | cp_model.IntVar | None = None
    authority_id: str = ADJUDICATION_AUTHORITY_ID_V3
    authority_sha256: str = ADJUDICATION_AUTHORITY_SHA256_V3

    def __post_init__(self) -> None:
        if not isinstance(self.model, cp_model.CpModel):
            raise TypeError("adjudication predicate context requires a CpModel")
        if self.origin not in {"search", "exact_trace_prefix"}:
            raise ValueError("adjudication predicate origin is unsupported")
        if self.origin == "search" and self.exact_events:
            raise ValueError("search context cannot contain exact adjudication events")
        if isinstance(self.action_sequence, cp_model.IntVar):
            if (self.origin != "search" or self.action_sequence.model_proto is not self.model.proto
                    or self.action_sequence.proto.domain[0] < 0):
                raise ValueError("adjudication search boundary must be nonnegative and belong to its CP-SAT model")
        elif self.action_sequence is not None and (
            type(self.action_sequence) is not int or self.action_sequence < 0
        ):
            raise ValueError("adjudication action_sequence must be non-negative")


@dataclass(frozen=True, slots=True)
class ScenarioPredicateStageContextV3:
    """One stage model shared by legacy and V3-only predicate lowerings."""

    base_context: Any
    adjudication_contexts: Mapping[str, AdjudicationPredicateModelContextV3]

    def __post_init__(self) -> None:
        model = getattr(self.base_context, "model", None)
        if not isinstance(model, cp_model.CpModel):
            raise TypeError("V3 predicate stage context requires a CP-SAT base model")
        contexts = dict(self.adjudication_contexts)
        if any(
            type(predicate_id) is not str
            or not predicate_id
            or context.model is not model
            for predicate_id, context in contexts.items()
        ):
            raise ValueError(
                "V3 adjudication contexts must be named and share the stage model"
            )
        object.__setattr__(self, "adjudication_contexts", MappingProxyType(contexts))

    @property
    def model(self) -> cp_model.CpModel:
        return self.base_context.model

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_context, name)


def _matching_event(
    predicate: ScenarioPredicateV1,
    context: AdjudicationPredicateModelContextV3,
) -> AuthoredAdjudicationEventV1 | None:
    args = predicate.arguments
    expected_kind, expected_reason = {
        OUT_OF_TURN_PREDICATE_V3: ("out_of_turn", "play_before_turn"),
        PURCHASE_BEFORE_TURN_PREDICATE_V3: ("invalid_bid", "purchase_before_turn"),
        FALSE_EKKA_PREDICATE_V3: (
            "false_ekka_attempt",
            "illegal_ekka_declaration",
        ),
    }[predicate.kind]
    for event in context.exact_events:
        if (
            context.action_sequence is not None
            and event.action_sequence != context.action_sequence
        ):
            continue
        action = dict(event.attempted_action)
        requested_action = (
            args["attemptedCardId"]
            if predicate.kind
            in {OUT_OF_TURN_PREDICATE_V3, FALSE_EKKA_PREDICATE_V3}
            else args["purchaseToken"]
        )
        observed_action = action.get("cardId", action.get("token"))
        if (
            predicate.kind in {OUT_OF_TURN_PREDICATE_V3, FALSE_EKKA_PREDICATE_V3}
            and observed_action is None
            and "card" in action
        ):
            try:
                observed_action = Card.parse(action["card"]).id
            except ValueError:
                observed_action = None
        if (
            event.kind.value == expected_kind
            and event.reason_code == expected_reason
            and event.actor_seat == args["attemptedActor"]
            and observed_action == requested_action
            and (
                predicate.kind != FALSE_EKKA_PREDICATE_V3
                or action.get("ekka") is True
            )
            and (
                args.get("requiredRemedy") is None
                or event.selected_remedy == args["requiredRemedy"]
            )
        ):
            return event
    return None


def lower_adjudication_predicate_v3(
    predicate: ScenarioPredicateV1,
    context: AdjudicationPredicateModelContextV3,
) -> CpSatPredicateLoweringV1:
    definitions = {
        item.kind: item for item in adjudication_predicate_definitions_v3(True)
    }
    definition = definitions.get(predicate.kind)
    if definition is None:
        raise ValueError(f"unsupported V3 adjudication predicate: {predicate.kind}")
    definition._validate_arguments(predicate.kind, predicate.arguments, definition.argument_schema)
    args = predicate.arguments
    context.model.add(
        context.attempted_actor == int(Seat.parse(args["attemptedActor"]))
    )
    if predicate.kind == OUT_OF_TURN_PREDICATE_V3:
        kind = AdjudicationAttemptKindV1.OUT_OF_TURN
        action_value = int(args["attemptedCardId"])
    elif predicate.kind == FALSE_EKKA_PREDICATE_V3:
        kind = AdjudicationAttemptKindV1.FALSE_EKKA
        action_value = int(args["attemptedCardId"])
    else:
        kind = AdjudicationAttemptKindV1.PURCHASE_BEFORE_TURN
        action_value = int(PurchaseAttemptTokenV1[args["purchaseToken"].upper()])
    context.model.add(context.attempted_action == action_value)
    remedy = args.get("requiredRemedy")
    branch = add_adjudication_attempt_branch_v1(
        context.model,
        kind=kind,
        expected_actor=context.expected_actor,
        attempted_actor=context.attempted_actor,
        attempted_action=context.attempted_action,
        authority_id=context.authority_id,
        authority_hash=context.authority_sha256,
    )
    if remedy:
        # A remedy constrains an existing event, not the truth of an inactive leaf.
        context.model.add(
            branch.admissible_remedy == int(AdjudicationRemedyV1[remedy.upper()])
        ).only_enforce_if(branch.event_present)
    literal = context.model.new_bool_var(f"{predicate.predicate_id}_attempt_satisfied")
    context.model.add(literal <= branch.event_present)
    context.model.add(literal <= context.attempted_action_available)
    context.model.add(
        literal >= branch.event_present + context.attempted_action_available - 1
    )
    event = None
    if context.origin == "exact_trace_prefix":
        event = _matching_event(predicate, context)
        sidecar_present = context.model.new_bool_var(f"{predicate.predicate_id}_sidecar_present")
        context.model.add(sidecar_present == int(event is not None))
        expected_actor_matches = context.model.new_bool_var(
            f"{predicate.predicate_id}_expected_actor_matches_sidecar"
        )
        if event is None or event.expected_actor_seat is None:
            context.model.add(expected_actor_matches == 0)
        else:
            sidecar_expected = int(Seat.parse(event.expected_actor_seat))
            context.model.add(context.expected_actor == sidecar_expected).only_enforce_if(
                expected_actor_matches
            )
            context.model.add(context.expected_actor != sidecar_expected).only_enforce_if(
                expected_actor_matches.Not()
            )
        literal = context.model.new_bool_var(f"{predicate.predicate_id}_satisfied")
        context.model.add(literal <= branch.event_present)
        context.model.add(literal <= context.attempted_action_available)
        context.model.add(literal <= sidecar_present)
        context.model.add(literal <= expected_actor_matches)
        context.model.add(
            literal >= branch.event_present
            + context.attempted_action_available
            + sidecar_present
            + expected_actor_matches
            - 3
        )
    lowering_core = {
        "schemaVersion": 1,
        "predicateSha256": predicate.sha256,
        "contextId": context.context_id,
        "origin": context.origin,
        "authorityId": context.authority_id,
        "authoritySha256": context.authority_sha256,
    }

    def extract(solver: cp_model.CpSolver) -> Mapping[str, Any]:
        event_present = bool(solver.boolean_value(branch.event_present))
        result = {
            "schemaVersion": 1,
            "backendId": "ortools_cp_sat_v3",
            "kind": kind.value,
            "eventPresent": event_present,
            "attemptedActionAvailable": bool(
                solver.boolean_value(context.attempted_action_available)
            ),
            "expectedActor": Seat(solver.value(branch.expected_actor)).name.lower(),
            "attemptedActor": Seat(solver.value(branch.attempted_actor)).name.lower(),
            "attemptedAction": (
                {"type": "card", "cardId": solver.value(branch.attempted_action)}
                if kind
                in {AdjudicationAttemptKindV1.OUT_OF_TURN, AdjudicationAttemptKindV1.FALSE_EKKA}
                else {
                    "type": "auction_token",
                    "token": PurchaseAttemptTokenV1(
                        solver.value(branch.attempted_action)
                    ).name.lower(),
                }
            ),
            "allowedRemedies": list(
                _OUT_OF_TURN_REMEDIES
                if kind
                in {AdjudicationAttemptKindV1.OUT_OF_TURN, AdjudicationAttemptKindV1.FALSE_EKKA}
                else _PURCHASE_REMEDIES
            ),
            "ordinaryActionHistoryMutated": False,
            "requiresTableReferee": True,
            "authorityId": context.authority_id,
            "authorityHash": context.authority_sha256,
        }
        if remedy and solver.boolean_value(literal):
            result["selectedRemedy"] = remedy
        if context.origin == "exact_trace_prefix":
            result["exactSidecarSha256"] = event.sha256 if event is not None else None
        if context.action_sequence is not None:
            result["actionSequence"] = (
                solver.value(context.action_sequence)
                if isinstance(context.action_sequence, cp_model.IntVar)
                else context.action_sequence
            )
        return result

    return CpSatPredicateLoweringV1(
        predicate.predicate_id,
        predicate.kind,
        literal,
        definition.cp_sat_support.evidence_extractor,
        canonical_sha256(lowering_core),
        extract,
    )


def _registrations(
    include_false_ekka: bool = False,
) -> tuple[ScenarioPredicateLoweringRegistrationV2, ...]:
    rows = []
    for definition in adjudication_predicate_definitions_v3(include_false_ekka):
        rows.append(
            ScenarioPredicateLoweringRegistrationV2(
                definition.kind,
                definition.category,
                "scenario_adjudication_predicates_v3.lower_adjudication_predicate_v3",
                definition.cp_sat_support.evidence_extractor,
                "tests/test_research_scenario_adjudication_predicates_v3.py",
            )
        )
    return tuple(rows)


def _extensions(
    authority_id: str,
    authority_sha256: str,
    include_false_ekka: bool = False,
) -> tuple[ScenarioPredicateCapabilityExtensionV3, ...]:
    definitions = {
        item.kind: item
        for item in adjudication_predicate_definitions_v3(include_false_ekka)
    }
    extensions = []
    for registration in _registrations(include_false_ekka):
        definition = definitions[registration.kind]
        decisions = (
            ("play",)
            if definition.category == "playPrefix"
            else ("bid", "play")
        )
        contexts = tuple(
            ScenarioPredicateExecutionContextV3(origin, decision, proof, "complete_state")
            for origin in ("exact_trace_prefix", "search")
            for decision in decisions
            for proof in ("bounded", "exact", "policy_conditioned")
        )
        capability = ScenarioPredicateCapabilityV3.build(
            kind=definition.kind,
            definition_sha256=definition.sha256,
            registration_sha256=canonical_sha256(registration.to_dict()),
            lowerer_owner=registration.lowerer_owner,
            evidence_extractor=registration.evidence_extractor_id,
            owning_test=registration.owning_test,
            qualification_binding="cp_sat_literal",
            required_rule_primitives=definition.required_domain_primitives,
            execution_contexts=contexts,
        )
        extensions.append(
            ScenarioPredicateCapabilityExtensionV3.build(
                capability,
                definition_owner="scenario_adjudication_predicates_v3",
                authority_id=authority_id,
                authority_sha256=authority_sha256,
            )
        )
    return tuple(extensions)


class ScenarioPredicateCatalogLoweringManifestV3:
    """Merged V2 plus V3-only definitions with one fail-closed dispatcher."""

    def __init__(
        self,
        base: ScenarioPredicateCatalogLoweringManifestV2 | None = None,
        *,
        adjudication_authority_id: str = ADJUDICATION_AUTHORITY_ID_V3,
        adjudication_authority_sha256: str = ADJUDICATION_AUTHORITY_SHA256_V3,
        include_false_ekka: bool = False,
        native_delayed_payoff: bool = False,
    ) -> None:
        self._base = base or shared_scenario_predicate_catalog_lowering_manifest_v2()
        self._native_deal_lowerer = DealPredicateLoweringRegistryV2()
        self.adjudication_authority_id = str(adjudication_authority_id)
        self.adjudication_authority_sha256 = str(
            adjudication_authority_sha256
        )
        if type(include_false_ekka) is not bool:
            raise TypeError("include_false_ekka must be a boolean")
        self.include_false_ekka = include_false_ekka
        self._extensions = _extensions(
            self.adjudication_authority_id,
            self.adjudication_authority_sha256,
            include_false_ekka,
        )
        self._definitions = {item.kind: native_scenario_predicate_definition_v3(item) for item in self._base.exposed_definitions}
        self._definitions.update(
            {
                item.kind: item
                for item in adjudication_predicate_definitions_v3(include_false_ekka)
            }
        )
        self._registrations = (
            *self._base.registrations,
            *_registrations(include_false_ekka),
        )
        self._capabilities = ScenarioPredicateCapabilityManifestV3(
            base=self._base, extensions=self._extensions, native_delayed_payoff=native_delayed_payoff,
        )
        self.manifest_hash = canonical_sha256(self._catalog_core())

    @property
    def registrations(self):
        return self._registrations

    @property
    def exposed_definitions(self):
        return tuple(self._definitions[kind] for kind in sorted(self._definitions))

    @property
    def hidden_definitions(self):
        return self._base.hidden_definitions

    @property
    def capability_manifest(self) -> ScenarioPredicateCapabilityManifestV3:
        return self._capabilities

    def definition(self, kind: str) -> ScenarioPredicateDefinitionV2 | None:
        return self._definitions.get(str(kind))

    def _catalog_core(self) -> dict[str, Any]:
        return {
            "schemaVersion": 3,
            "baseManifestHash": self._base.manifest_hash,
            "nativeLoweringImplementations": {
                "deal.seat_relation": NATIVE_SEAT_RELATION_IMPLEMENTATION,
            },
            "capabilityManifestHash": self._capabilities.manifest_hash,
            "registrations": [item.to_dict() for item in self._registrations],
            "exposedDefinitions": [item.to_dict() for item in self.exposed_definitions],
            "hiddenDefinitions": [item.to_dict() for item in self.hidden_definitions],
        }

    def catalog(self) -> dict[str, Any]:
        return {**self._catalog_core(), "sha256": self.manifest_hash}

    def parse_predicate(self, value: Mapping[str, Any]) -> ScenarioPredicateV1:
        predicate = ScenarioPredicateV1.build(value)
        definition = self.definition(predicate.kind)
        if definition is None:
            raise ValueError(f"predicate is hidden or unsupported: {predicate.kind}")
        if predicate.version != definition.version:
            raise ValueError(f"predicate version is unsupported: {predicate.kind}")
        if predicate.hardness not in definition.allowed_hardness:
            raise ValueError(f"predicate hardness is unsupported: {predicate.kind}")
        definition._validate_arguments(predicate.kind, predicate.arguments, definition.argument_schema)
        return predicate

    def lower_predicate(self, predicate: ScenarioPredicateV1, context: Any) -> Any:
        if predicate.kind == "deal.seat_relation":
            return self._native_deal_lowerer.lower_predicate(
                predicate,
                context.base_context if isinstance(context, ScenarioPredicateStageContextV3) else context,
                native_seat_relation=True,
            )
        if predicate.kind in {
            OUT_OF_TURN_PREDICATE_V3,
            PURCHASE_BEFORE_TURN_PREDICATE_V3,
            FALSE_EKKA_PREDICATE_V3,
        }:
            adjudication_context = (
                context.adjudication_contexts.get(predicate.predicate_id)
                if isinstance(context, ScenarioPredicateStageContextV3)
                else context
            )
            if not isinstance(adjudication_context, AdjudicationPredicateModelContextV3):
                raise TypeError("V3 adjudication predicate requires its native model context")
            return lower_adjudication_predicate_v3(predicate, adjudication_context)
        return self._base.lower_predicate(
            predicate,
            context.base_context
            if isinstance(context, ScenarioPredicateStageContextV3)
            else context,
        )


@lru_cache(maxsize=16)
def shared_scenario_predicate_catalog_lowering_manifest_v3(
    adjudication_authority_id: str = ADJUDICATION_AUTHORITY_ID_V3,
    adjudication_authority_sha256: str = ADJUDICATION_AUTHORITY_SHA256_V3,
    include_false_ekka: bool = False,
    native_delayed_payoff: bool = False,
) -> ScenarioPredicateCatalogLoweringManifestV3:
    return ScenarioPredicateCatalogLoweringManifestV3(
        adjudication_authority_id=adjudication_authority_id,
        adjudication_authority_sha256=adjudication_authority_sha256,
        include_false_ekka=include_false_ekka,
        native_delayed_payoff=native_delayed_payoff,
    )


__all__ = [
    "ADJUDICATION_AUTHORITY_ID_V3",
    "ADJUDICATION_AUTHORITY_SHA256_V3",
    "AdjudicationPredicateModelContextV3",
    "FALSE_EKKA_PREDICATE_V3",
    "OUT_OF_TURN_PREDICATE_V3",
    "PURCHASE_BEFORE_TURN_PREDICATE_V3",
    "ScenarioPredicateCatalogLoweringManifestV3",
    "ScenarioPredicateStageContextV3",
    "adjudication_predicate_definitions_v3",
    "lower_adjudication_predicate_v3",
    "shared_scenario_predicate_catalog_lowering_manifest_v3",
]
