"""Truthful V3-only capability overlay for scenario predicates.

The V2 predicate definitions and lowering manifest are immutable historical
contracts.  This module joins those definitions to their real lowering and
evidence registrations, then publishes only execution combinations supported
by the native V3 Research lane.  It deliberately does not dispatch a lowering
or import the Research backend.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable

from baloot_platform.tactics import DELAYED_PAYOFF_TACTIC_IDS

from .scenario_contracts import canonical_sha256, hard_tactic_causal_request_issue_v1
from .scenario_predicate_manifest_v2 import (
    ScenarioPredicateCatalogLoweringManifestV2,
    ScenarioPredicateLoweringRegistrationV2,
    shared_scenario_predicate_catalog_lowering_manifest_v2,
)
from .scenario_proof_tactic_predicates_v2 import (
    TACTICAL_DETECTOR_PREREQUISITES_V1,
)
from .scenario_registry_v2 import ScenarioPredicateDefinitionV2


SCENARIO_PREDICATE_CAPABILITY_BACKEND_V3 = "ortools_cp_sat_v3"
SCENARIO_PREDICATE_CAPABILITY_REGISTRY_V3 = (
    "scenario_predicate_capability_registry_v3"
)

_ORIGINS = ("exact_trace_prefix", "search")
_DECISION_KINDS = ("bid", "escalation", "project", "play")
_PROOF_MODES = ("bounded", "exact", "policy_conditioned")
_VISIBILITIES = ("actor_information_set", "complete_state")
_ACTOR_INFORMATION_SAFE = frozenset({"public", "observation", "nonanticipative"})
_HASH = re.compile(r"^[0-9a-f]{64}$")

# These restrictions are keyed by the real evidence registrations rather than
# by a second list of predicate kinds.  Their production extractors decode play
# cards and therefore cannot qualify bid or escalation root actions.
_PLAY_ONLY_EVIDENCE_EXTRACTORS = frozenset(
    {
        "cp_sat.proof_principal_trick_team_rank_count_evidence_v1",
        "cp_sat.proof_principal_trick_winner_evidence_v1",
        "cp_sat.proof_root_card_rank_count_evidence_v1",
        "cp_sat.winning_root_match_evidence_v1",
    }
)
_CONTINUATION_COUNT_EVIDENCE_EXTRACTOR = (
    "cp_sat.proof_continuation_line_count_evidence_v1"
)
# These producers currently bind complete-state variables, not actor-world evidence.
_ACTOR_INFORMATION_UNAVAILABLE_EVIDENCE_EXTRACTORS = frozenset({
    "cp_sat.proof_principal_trick_team_rank_count_evidence_v1",
    "cp_sat.proof_principal_trick_winner_evidence_v1",
})
_NATIVE_ACTOR_CARD_ROOT_EVIDENCE_EXTRACTORS = frozenset({
    "cp_sat.proof_root_card_rank_count_evidence_v1",
    "cp_sat.winning_root_match_evidence_v1",
})
_TACTIC_EVIDENCE_EXTRACTORS = frozenset({
    "cp_sat.registered_tactic_applicable_evidence_v1",
    "cp_sat.registered_tactic_verified_evidence_v1",
})
NATIVE_V3_TACTIC_IDS = frozenset(TACTICAL_DETECTOR_PREREQUISITES_V1)
NATIVE_V3_ACTOR_ROOT_TACTIC_IDS = frozenset({
    "only_winning_move", "debate", "contract_rescue",
    "kabout_creation", "kabout_preservation", "kabout_prevention",
    "small_card_big_gain",
})
NATIVE_V3_NONPLAY_STATE_KINDS = frozenset({
    "state.phase", "state.next_actor", "state.legal_action_count",
    "state.match_score", "state.current_raw_points",
    "state.current_project_points", "state.current_game_points",
})


def native_v3_state_decision_supported(kind: str, decision_kind: str) -> bool:
    if decision_kind == "play":
        return True
    if kind == "state.observer_card_visibility":
        return decision_kind in {"bid", "escalation", "project"}
    if decision_kind in {"bid", "escalation", "project"}:
        return kind in NATIVE_V3_NONPLAY_STATE_KINDS
    return False


def _strict_text(value: Any, path: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{path} is required")
    return value


def _strict_hash(value: Any, path: str) -> str:
    text = _strict_text(value, path)
    if _HASH.fullmatch(text) is None:
        raise ValueError(f"{path} must be a lowercase sha256")
    return text


def _normalized_texts(
    values: Iterable[str],
    path: str,
    *,
    allowed: frozenset[str] | set[str] | None = None,
) -> tuple[str, ...]:
    rows = tuple(_strict_text(value, f"{path}[]") for value in values)
    if len(rows) != len(set(rows)):
        raise ValueError(f"{path} must not contain duplicates")
    if allowed is not None and not set(rows).issubset(allowed):
        raise ValueError(f"{path} contains an unsupported value")
    return tuple(sorted(rows))


@dataclass(frozen=True, slots=True, order=True)
class ScenarioPredicateExecutionContextV3:
    """One supported origin/decision/proof/visibility combination."""

    origin: str
    decision_kind: str
    proof_mode: str
    visibility: str
    line_universes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.origin not in _ORIGINS:
            raise ValueError("predicate capability origin is unsupported")
        if self.decision_kind not in _DECISION_KINDS:
            raise ValueError("predicate capability decision kind is unsupported")
        if self.proof_mode not in _PROOF_MODES:
            raise ValueError("predicate capability proof mode is unsupported")
        if self.visibility not in _VISIBILITIES:
            raise ValueError("predicate capability visibility is unsupported")
        normalized = _normalized_texts(
            self.line_universes,
            "predicate capability lineUniverses",
            allowed={"all_legal", "optimal_strategy", "policy_conditioned"},
        )
        object.__setattr__(self, "line_universes", normalized)
        if self.visibility == "actor_information_set" and (
            self.proof_mode != "exact" or normalized
        ):
            raise ValueError(
                "actor-information predicate capability is exact and uncounted only"
            )
        if self.proof_mode == "policy_conditioned" and normalized not in {
            (),
            ("policy_conditioned",),
        }:
            raise ValueError(
                "policy-conditioned capability uses only its policy line universe"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin": self.origin,
            "decisionKind": self.decision_kind,
            "proofMode": self.proof_mode,
            "visibility": self.visibility,
            "lineUniverses": list(self.line_universes),
        }


@dataclass(frozen=True, slots=True, order=True)
class ScenarioPredicateArgumentContextV3:
    """Argument-specific restrictions within one predicate capability."""

    argument_name: str
    argument_value: str
    decision_kinds: tuple[str, ...]
    required_utility: str | None = None
    required_proof_bindings: tuple[tuple[str, int], ...] = ()
    execution_contexts: tuple[ScenarioPredicateExecutionContextV3, ...] = ()
    causal_verification_available: bool | None = None

    def __post_init__(self) -> None:
        if self.causal_verification_available is not None and type(self.causal_verification_available) is not bool:
            raise TypeError("predicate argument causal verification availability must be a boolean")
        _strict_text(self.argument_name, "predicate argument context name")
        _strict_text(self.argument_value, "predicate argument context value")
        decisions = _normalized_texts(
            self.decision_kinds,
            "predicate argument context decisionKinds",
            allowed=set(_DECISION_KINDS),
        )
        if not decisions:
            raise ValueError("predicate argument context requires decision kinds")
        object.__setattr__(self, "decision_kinds", decisions)
        if self.required_utility is not None:
            _strict_text(
                self.required_utility,
                "predicate argument context requiredUtility",
            )
        bindings = tuple(sorted(self.required_proof_bindings))
        names = [name for name, _minimum in bindings]
        if len(names) != len(set(names)) or any(
            not name or type(minimum) is not int or minimum < 1
            for name, minimum in bindings
        ):
            raise ValueError("predicate argument context proof bindings are invalid")
        object.__setattr__(self, "required_proof_bindings", bindings)
        contexts = tuple(sorted(self.execution_contexts))
        if len(contexts) != len(set(contexts)) or any(
            context.decision_kind not in decisions for context in contexts
        ):
            raise ValueError("predicate argument execution contexts are invalid")
        object.__setattr__(self, "execution_contexts", contexts)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "argumentName": self.argument_name,
            "argumentValue": self.argument_value,
            "decisionKinds": list(self.decision_kinds),
            "requiredUtility": self.required_utility,
            "requiredProofBindings": [
                {"binding": binding, "minimumCount": minimum}
                for binding, minimum in self.required_proof_bindings
            ],
        }
        if self.execution_contexts:
            result["executionContexts"] = [
                context.to_dict() for context in self.execution_contexts
            ]
        if self.causal_verification_available is not None:
            result["causalVerificationAvailable"] = self.causal_verification_available
        return result


@dataclass(frozen=True, slots=True)
class ScenarioPredicateCapabilityV3:
    """V3 execution truth bound to one frozen V2 predicate registration."""

    kind: str
    definition_sha256: str
    registration_sha256: str
    lowerer_owner: str
    evidence_extractor: str
    owning_test: str
    qualification_binding: str
    required_rule_primitives: tuple[str, ...]
    execution_contexts: tuple[ScenarioPredicateExecutionContextV3, ...]
    argument_contexts: tuple[ScenarioPredicateArgumentContextV3, ...]
    native_status: str
    sha256: str

    @classmethod
    def build(
        cls,
        *,
        kind: str,
        definition_sha256: str,
        registration_sha256: str,
        lowerer_owner: str,
        evidence_extractor: str,
        owning_test: str,
        qualification_binding: str,
        required_rule_primitives: Iterable[str],
        execution_contexts: Iterable[ScenarioPredicateExecutionContextV3],
        argument_contexts: Iterable[ScenarioPredicateArgumentContextV3] = (),
        native_status: str = "native_exact",
    ) -> "ScenarioPredicateCapabilityV3":
        kind = _strict_text(kind, "predicate capability kind")
        definition_sha256 = _strict_hash(
            definition_sha256, "predicate capability definitionSha256"
        )
        registration_sha256 = _strict_hash(
            registration_sha256, "predicate capability registrationSha256"
        )
        lowerer_owner = _strict_text(
            lowerer_owner, "predicate capability lowererOwner"
        )
        evidence_extractor = _strict_text(
            evidence_extractor, "predicate capability evidenceExtractor"
        )
        owning_test = _strict_text(owning_test, "predicate capability owningTest")
        if qualification_binding not in {"cp_sat_literal", "proof_variable"}:
            raise ValueError("predicate capability qualificationBinding is unsupported")
        if native_status not in {"native_exact", "unsupported"}:
            raise ValueError("predicate capability nativeStatus is unsupported")
        primitives = _normalized_texts(
            required_rule_primitives,
            "predicate capability requiredRulePrimitives",
        )
        contexts = tuple(sorted(execution_contexts))
        if len(contexts) != len(set(contexts)):
            raise ValueError(
                "predicate capability executionContexts contain duplicates"
            )
        if native_status == "native_exact" and not contexts:
            raise ValueError("native predicate capability requires execution contexts")
        argument_rows = tuple(sorted(argument_contexts))
        argument_keys = [
            (item.argument_name, item.argument_value) for item in argument_rows
        ]
        if len(argument_keys) != len(set(argument_keys)):
            raise ValueError("predicate capability argumentContexts contain duplicates")
        supported_decisions = {item.decision_kind for item in contexts}
        if any(
            not set(item.decision_kinds).issubset(supported_decisions)
            or not set(item.execution_contexts).issubset(contexts)
            for item in argument_rows
        ):
            raise ValueError(
                "predicate capability argument context exceeds execution contexts"
            )
        core = {
            "schemaVersion": 3,
            "kind": kind,
            "definitionSha256": definition_sha256,
            "registrationSha256": registration_sha256,
            "lowererOwner": lowerer_owner,
            "evidenceExtractor": evidence_extractor,
            "owningTest": owning_test,
            "qualificationBinding": qualification_binding,
            "requiredRulePrimitives": list(primitives),
            "nativeStatus": native_status,
            "origins": sorted({item.origin for item in contexts}),
            "decisionKinds": sorted({item.decision_kind for item in contexts}),
            "proofModes": sorted({item.proof_mode for item in contexts}),
            "visibilityClasses": sorted({item.visibility for item in contexts}),
            "executionContexts": [item.to_dict() for item in contexts],
        }
        if argument_rows:
            core["argumentContexts"] = [item.to_dict() for item in argument_rows]
        return cls(
            kind,
            definition_sha256,
            registration_sha256,
            lowerer_owner,
            evidence_extractor,
            owning_test,
            qualification_binding,
            primitives,
            contexts,
            argument_rows,
            native_status,
            canonical_sha256(core),
        )

    @property
    def origins(self) -> tuple[str, ...]:
        return tuple(sorted({item.origin for item in self.execution_contexts}))

    @property
    def decision_kinds(self) -> tuple[str, ...]:
        return tuple(
            sorted({item.decision_kind for item in self.execution_contexts})
        )

    @property
    def proof_modes(self) -> tuple[str, ...]:
        return tuple(sorted({item.proof_mode for item in self.execution_contexts}))

    @property
    def visibility_classes(self) -> tuple[str, ...]:
        return tuple(sorted({item.visibility for item in self.execution_contexts}))

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schemaVersion": 3,
            "kind": self.kind,
            "definitionSha256": self.definition_sha256,
            "registrationSha256": self.registration_sha256,
            "lowererOwner": self.lowerer_owner,
            "evidenceExtractor": self.evidence_extractor,
            "owningTest": self.owning_test,
            "qualificationBinding": self.qualification_binding,
            "requiredRulePrimitives": list(self.required_rule_primitives),
            "nativeStatus": self.native_status,
            "origins": list(self.origins),
            "decisionKinds": list(self.decision_kinds),
            "proofModes": list(self.proof_modes),
            "visibilityClasses": list(self.visibility_classes),
            "executionContexts": [
                item.to_dict() for item in self.execution_contexts
            ],
            "sha256": self.sha256,
        }
        if self.argument_contexts:
            result["argumentContexts"] = [
                item.to_dict() for item in self.argument_contexts
            ]
        return result


@dataclass(frozen=True, slots=True)
class ScenarioPredicateCapabilityExtensionV3:
    """Authority binding for a native predicate introduced only in V3.

    Extension definitions remain outside the frozen V2 catalog.  Their own
    definition and registration hashes are carried by ``capability`` and the
    selected bundle authority is bound here.  Report-only referee evidence is
    not a native hard-predicate extension and must stay in its sidecar catalog.
    """

    capability: ScenarioPredicateCapabilityV3
    definition_owner: str
    authority_id: str
    authority_sha256: str
    sha256: str

    @classmethod
    def build(
        cls,
        capability: ScenarioPredicateCapabilityV3,
        *,
        definition_owner: str,
        authority_id: str,
        authority_sha256: str,
    ) -> "ScenarioPredicateCapabilityExtensionV3":
        if not isinstance(capability, ScenarioPredicateCapabilityV3):
            raise TypeError("V3 predicate extension capability is invalid")
        definition_owner = _strict_text(
            definition_owner, "V3 predicate extension definitionOwner"
        )
        authority_id = _strict_text(
            authority_id, "V3 predicate extension authorityId"
        )
        authority_sha256 = _strict_hash(
            authority_sha256, "V3 predicate extension authoritySha256"
        )
        core = {
            "schemaVersion": 1,
            "kind": capability.kind,
            "definitionOwner": definition_owner,
            "definitionSha256": capability.definition_sha256,
            "registrationOwner": capability.lowerer_owner,
            "registrationSha256": capability.registration_sha256,
            "authorityId": authority_id,
            "authoritySha256": authority_sha256,
            "capabilitySha256": capability.sha256,
        }
        return cls(
            capability,
            definition_owner,
            authority_id,
            authority_sha256,
            canonical_sha256(core),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "kind": self.capability.kind,
            "definitionOwner": self.definition_owner,
            "definitionSha256": self.capability.definition_sha256,
            "registrationOwner": self.capability.lowerer_owner,
            "registrationSha256": self.capability.registration_sha256,
            "authorityId": self.authority_id,
            "authoritySha256": self.authority_sha256,
            "capabilitySha256": self.capability.sha256,
            "sha256": self.sha256,
        }


def native_scenario_predicate_definition_v3(definition: ScenarioPredicateDefinitionV2) -> ScenarioPredicateDefinitionV2:
    """Expose certified native roots without rewriting historical V2 metadata."""
    if definition.cp_sat_support.evidence_extractor not in _NATIVE_ACTOR_CARD_ROOT_EVIDENCE_EXTRACTORS:
        return definition
    payload = definition.to_dict()
    payload.pop("schemaVersion")
    payload.pop("sha256")
    payload["cpSatSupport"]["actorInformationSet"] = "nonanticipative"
    payload["tests"] = sorted({*payload["tests"], "tests/test_research_ortools_unified_executor.py"})
    return ScenarioPredicateDefinitionV2.build(payload)


def _supported_origins(definition: ScenarioPredicateDefinitionV2) -> tuple[str, ...]:
    result: list[str] = []
    if definition.cp_sat_support.exact_input_status == "exact":
        result.append("exact_trace_prefix")
    if definition.cp_sat_support.search_status in {"exact", "derived"}:
        result.append("search")
    return tuple(result)


def _supported_modes(definition: ScenarioPredicateDefinitionV2) -> tuple[str, ...]:
    result: list[str] = []
    if definition.cp_sat_support.bounded_proof != "unsupported":
        result.append("bounded")
    if definition.cp_sat_support.exact_proof != "unsupported":
        result.append("exact")
    if definition.cp_sat_support.policy_conditioned_proof != "unsupported":
        result.append("policy_conditioned")
    return tuple(result)


def _line_universes(
    registration: ScenarioPredicateLoweringRegistrationV2,
    proof_mode: str,
) -> tuple[str, ...]:
    if registration.evidence_extractor_id != _CONTINUATION_COUNT_EVIDENCE_EXTRACTOR:
        return ()
    if proof_mode == "policy_conditioned":
        return ("policy_conditioned",)
    return ("all_legal", "optimal_strategy")


def _derive_contexts(
    definition: ScenarioPredicateDefinitionV2,
    registration: ScenarioPredicateLoweringRegistrationV2,
) -> tuple[ScenarioPredicateExecutionContextV3, ...]:
    if definition.stage == "state":
        decision_kinds = tuple(
            kind for kind in _DECISION_KINDS
            if native_v3_state_decision_supported(definition.kind, kind)
        )
    elif registration.evidence_extractor_id in _PLAY_ONLY_EVIDENCE_EXTRACTORS:
        decision_kinds = ("play",)
    else:
        decision_kinds = _DECISION_KINDS
    contexts: list[ScenarioPredicateExecutionContextV3] = []
    for origin in _supported_origins(definition):
        for decision_kind in decision_kinds:
            for proof_mode in _supported_modes(definition):
                contexts.append(
                    ScenarioPredicateExecutionContextV3(
                        origin,
                        decision_kind,
                        proof_mode,
                        "complete_state",
                        _line_universes(registration, proof_mode),
                    )
                )
                if (
                    proof_mode == "exact"
                    and definition.cp_sat_support.actor_information_set
                    in _ACTOR_INFORMATION_SAFE
                    and registration.evidence_extractor_id
                    != _CONTINUATION_COUNT_EVIDENCE_EXTRACTOR
                    and registration.evidence_extractor_id
                    not in _ACTOR_INFORMATION_UNAVAILABLE_EVIDENCE_EXTRACTORS
                    and (registration.evidence_extractor_id not in _TACTIC_EVIDENCE_EXTRACTORS
                         or decision_kind == "play")
                ):
                    contexts.append(
                        ScenarioPredicateExecutionContextV3(
                            origin,
                            decision_kind,
                            proof_mode,
                            "actor_information_set",
                        )
                    )
    return tuple(contexts)


def _derive_argument_contexts(
    predicate_kind: str,
    *, native_delayed_payoff: bool = False,
) -> tuple[ScenarioPredicateArgumentContextV3, ...]:
    if predicate_kind not in {"tactic.applicable", "tactic.verified"}:
        return ()
    return tuple(
        ScenarioPredicateArgumentContextV3(
            argument_name="kind",
            argument_value=prerequisite.tactic_id,
            causal_verification_available=(
                hard_tactic_causal_request_issue_v1(
                    predicate_kind, "hard", {"kind": prerequisite.tactic_id, "causalRequired": True},
                    native_delayed_payoff=native_delayed_payoff,
                ) is None
            ) if predicate_kind == "tactic.verified" else None,
            decision_kinds=("project",) if prerequisite.tactic_id == "project_bidding"
            else prerequisite.decision_kinds,
            required_utility="awarded_point_margin" if prerequisite.tactic_id in {"project_bidding", "trump_control", "partner_enablement", "entry_unblocking"}
            or native_delayed_payoff and prerequisite.tactic_id in DELAYED_PAYOFF_TACTIC_IDS
            else prerequisite.required_utility,
            required_proof_bindings=(
                *prerequisite.required_proof_bindings,
                ("root_project_consequence_variables", 2),
                ("tactical_detector_profile", 1),
            ) if prerequisite.tactic_id == "project_bidding"
            else (*prerequisite.required_proof_bindings, ("trump_control_suffix_actions", 1),
                  ("tactical_detector_profile", 1), ("causal_gain_scale", 1),
                  ("causal_gain_terminal_awarded_scope", 1))
            if prerequisite.tactic_id == "trump_control"
            else (*prerequisite.required_proof_bindings, ("root_raw_payoff_variables", 2),
                  ("tactical_detector_profile", 1), ("causal_gain_scale", 1),
                  ("causal_gain_terminal_awarded_scope", 1))
            if prerequisite.tactic_id in {"partner_enablement", "entry_unblocking"}
            or native_delayed_payoff and prerequisite.tactic_id in DELAYED_PAYOFF_TACTIC_IDS
            else (*prerequisite.required_proof_bindings, ("tactical_detector_profile", 1))
            if prerequisite.tactic_id in DELAYED_PAYOFF_TACTIC_IDS
            else prerequisite.required_proof_bindings,
            execution_contexts=tuple(
                ScenarioPredicateExecutionContextV3(
                    origin, "project" if prerequisite.tactic_id == "project_bidding" else "play", mode, "complete_state",
                )
                for origin in _ORIGINS for mode in ("exact",)
            ) if prerequisite.tactic_id in {"project_bidding", "trump_control", "partner_enablement", "entry_unblocking"}
            or native_delayed_payoff and prerequisite.tactic_id in DELAYED_PAYOFF_TACTIC_IDS else tuple(
                ScenarioPredicateExecutionContextV3(origin, decision, mode, visibility)
                for origin in _ORIGINS
                for decision in prerequisite.decision_kinds
                for mode in _PROOF_MODES
                for visibility in _VISIBILITIES
                if visibility == "complete_state" or (
                    prerequisite.tactic_id in NATIVE_V3_ACTOR_ROOT_TACTIC_IDS
                    and decision == "play" and mode == "exact"
                )
            ),
        )
        for prerequisite in TACTICAL_DETECTOR_PREREQUISITES_V1.values()
        if prerequisite.tactic_id in NATIVE_V3_TACTIC_IDS
    )


def derive_scenario_predicate_capabilities_v3(
    base: ScenarioPredicateCatalogLoweringManifestV2 | None = None,
    *, native_delayed_payoff: bool = False,
) -> tuple[ScenarioPredicateCapabilityV3, ...]:
    """Join every exposed V2 definition to its concrete native registration."""

    base = base or shared_scenario_predicate_catalog_lowering_manifest_v2()
    definitions = {item.kind: native_scenario_predicate_definition_v3(item) for item in base.exposed_definitions}
    registrations = {item.kind: item for item in base.registrations}
    if len(definitions) != len(base.exposed_definitions):
        raise ValueError("V3 capability base contains duplicate definitions")
    if len(registrations) != len(base.registrations):
        raise ValueError("V3 capability base contains duplicate registrations")
    if set(definitions) != set(registrations):
        raise ValueError(
            "V3 capability base definition/registration coverage is incomplete; "
            f"missing={sorted(set(definitions) - set(registrations))}, "
            f"extra={sorted(set(registrations) - set(definitions))}"
        )
    available_extractors = {
        item.evidence_extractor_id for item in base.registrations
    }
    missing_restriction_owners = sorted(
        (_PLAY_ONLY_EVIDENCE_EXTRACTORS | {_CONTINUATION_COUNT_EVIDENCE_EXTRACTOR}
         | _ACTOR_INFORMATION_UNAVAILABLE_EVIDENCE_EXTRACTORS)
        - available_extractors
    )
    if missing_restriction_owners:
        raise ValueError(
            "V3 capability restriction owner disappeared: "
            + ", ".join(missing_restriction_owners)
        )

    result: list[ScenarioPredicateCapabilityV3] = []
    for kind in sorted(definitions):
        definition = definitions[kind]
        registration = registrations[kind]
        if registration.evidence_extractor_id != (
            definition.cp_sat_support.evidence_extractor
        ):
            raise ValueError(
                f"V3 capability evidence registration drifted for {kind}"
            )
        contexts = _derive_contexts(definition, registration)
        result.append(
            ScenarioPredicateCapabilityV3.build(
                kind=kind,
                definition_sha256=definition.sha256,
                registration_sha256=canonical_sha256(registration.to_dict()),
                lowerer_owner=registration.lowerer_owner,
                evidence_extractor=registration.evidence_extractor_id,
                owning_test=registration.owning_test,
                qualification_binding=(
                    "proof_variable"
                    if definition.cp_sat_support.qualification_source == "proof"
                    else "cp_sat_literal"
                ),
                required_rule_primitives=definition.required_domain_primitives,
                execution_contexts=contexts,
                argument_contexts=_derive_argument_contexts(kind, native_delayed_payoff=native_delayed_payoff),
            )
        )
    return tuple(result)


class ScenarioPredicateCapabilityManifestV3:
    """Immutable, dynamically complete native V3 predicate capabilities."""

    def __init__(
        self,
        capabilities: Iterable[ScenarioPredicateCapabilityV3] | None = None,
        *,
        base: ScenarioPredicateCatalogLoweringManifestV2 | None = None,
        extensions: Iterable[ScenarioPredicateCapabilityExtensionV3] = (),
        require_publishable: bool = True,
        native_delayed_payoff: bool = False,
    ) -> None:
        if type(native_delayed_payoff) is not bool:
            raise TypeError("native delayed payoff capability must be a boolean")
        self._native_delayed_payoff = native_delayed_payoff
        self._base = base or shared_scenario_predicate_catalog_lowering_manifest_v2()
        extension_rows = tuple(
            sorted(extensions, key=lambda item: item.capability.kind)
        )
        extension_kinds = [item.capability.kind for item in extension_rows]
        extension_duplicates = sorted(
            kind for kind in set(extension_kinds) if extension_kinds.count(kind) > 1
        )
        if extension_duplicates:
            raise ValueError(
                "V3 predicate extensions contain duplicates: "
                f"{extension_duplicates}"
            )
        rows = tuple(
            sorted(
                capabilities
                if capabilities is not None
                else (
                    *derive_scenario_predicate_capabilities_v3(self._base, native_delayed_payoff=native_delayed_payoff),
                    *(item.capability for item in extension_rows),
                ),
                key=lambda item: item.kind,
            )
        )
        self._validate(
            rows,
            extensions=extension_rows,
            require_publishable=require_publishable,
        )
        self._capabilities = rows
        self._extensions = extension_rows
        core = {
            "schemaVersion": 3,
            "registryVersion": SCENARIO_PREDICATE_CAPABILITY_REGISTRY_V3,
            "backendId": SCENARIO_PREDICATE_CAPABILITY_BACKEND_V3,
            "baseManifestHash": self._base.manifest_hash,
            "capabilities": [item.to_dict() for item in rows],
        }
        if extension_rows:
            core["extensions"] = [item.to_dict() for item in extension_rows]
        self._catalog = {**core, "sha256": canonical_sha256(core)}

    def _validate(
        self,
        rows: tuple[ScenarioPredicateCapabilityV3, ...],
        *,
        extensions: tuple[ScenarioPredicateCapabilityExtensionV3, ...],
        require_publishable: bool,
    ) -> None:
        definitions = {item.kind: native_scenario_predicate_definition_v3(item) for item in self._base.exposed_definitions}
        registrations = {item.kind: item for item in self._base.registrations}
        extension_by_kind = {
            item.capability.kind: item for item in extensions
        }
        overlap = sorted(set(definitions) & set(extension_by_kind))
        if overlap:
            raise ValueError(
                "V3 predicate extensions cannot replace frozen V2 kinds: "
                f"{overlap}"
            )
        for extension in extensions:
            extension_document = extension.to_dict()
            declared_extension_hash = extension_document.pop("sha256")
            if declared_extension_hash != canonical_sha256(extension_document):
                raise ValueError(
                    "V3 predicate extension checksum drifted for "
                    f"{extension.capability.kind}"
                )
        kinds = [item.kind for item in rows]
        duplicates = sorted(kind for kind in set(kinds) if kinds.count(kind) > 1)
        if duplicates:
            raise ValueError(
                f"V3 predicate capabilities contain duplicates: {duplicates}"
            )
        expected_kinds = set(definitions) | set(extension_by_kind)
        missing = sorted(expected_kinds - set(kinds))
        extra = sorted(set(kinds) - expected_kinds)
        if missing or extra:
            raise ValueError(
                "V3 predicate capability coverage is incomplete; "
                f"missing={missing}, extra={extra}"
            )
        for row in rows:
            row_document = row.to_dict()
            declared_row_hash = row_document.pop("sha256")
            if declared_row_hash != canonical_sha256(row_document):
                raise ValueError(
                    f"V3 predicate capability checksum drifted for {row.kind}"
                )
            if row.kind in definitions:
                definition = definitions[row.kind]
                registration = registrations[row.kind]
                expected_registration_hash = canonical_sha256(
                    registration.to_dict()
                )
                if row.definition_sha256 != definition.sha256:
                    raise ValueError(
                        "V3 predicate capability definition hash drifted for "
                        f"{row.kind}"
                    )
                if row.registration_sha256 != expected_registration_hash:
                    raise ValueError(
                        "V3 predicate capability registration hash drifted for "
                        f"{row.kind}"
                    )
                if (
                    row.lowerer_owner != registration.lowerer_owner
                    or row.evidence_extractor != registration.evidence_extractor_id
                    or row.owning_test != registration.owning_test
                ):
                    raise ValueError(
                        "V3 predicate capability registration binding drifted for "
                        f"{row.kind}"
                    )
                expected_qualification_binding = (
                    "proof_variable"
                    if definition.cp_sat_support.qualification_source == "proof"
                    else "cp_sat_literal"
                )
                if row.qualification_binding != expected_qualification_binding:
                    raise ValueError(
                        "V3 predicate capability qualification binding drifted for "
                        f"{row.kind}"
                    )
                if row.required_rule_primitives != tuple(
                    sorted(definition.required_domain_primitives)
                ):
                    raise ValueError(
                        "V3 predicate capability rule primitives drifted for "
                        f"{row.kind}"
                    )
                if row.execution_contexts != tuple(
                    sorted(_derive_contexts(definition, registration))
                ):
                    raise ValueError(
                        "V3 predicate capability execution contexts drifted for "
                        f"{row.kind}"
                    )
                if row.argument_contexts != tuple(
                    sorted(_derive_argument_contexts(row.kind, native_delayed_payoff=self._native_delayed_payoff))
                ):
                    raise ValueError(
                        "V3 predicate capability argument contexts drifted for "
                        f"{row.kind}"
                    )
            else:
                extension = extension_by_kind[row.kind]
                if row != extension.capability:
                    raise ValueError(
                        "V3 predicate extension capability binding drifted for "
                        f"{row.kind}"
                    )
                if not row.required_rule_primitives:
                    raise ValueError(
                        "V3 native predicate extension requires rule primitives: "
                        f"{row.kind}"
                    )
            if row.evidence_extractor in _PLAY_ONLY_EVIDENCE_EXTRACTORS and (
                row.decision_kinds != ("play",)
            ):
                raise ValueError(
                    f"V3 card-root capability is not play-only: {row.kind}"
                )
            if row.evidence_extractor == _CONTINUATION_COUNT_EVIDENCE_EXTRACTOR and (
                "actor_information_set" in row.visibility_classes
            ):
                raise ValueError(
                    "V3 actor-information continuation counting is unsupported"
                )
            if row.evidence_extractor in _ACTOR_INFORMATION_UNAVAILABLE_EVIDENCE_EXTRACTORS and (
                "actor_information_set" in row.visibility_classes
            ):
                raise ValueError("V3 actor-information evidence producer is unavailable: " + row.kind)
            if any(
                context.visibility == "actor_information_set"
                and context.proof_mode != "exact"
                for context in row.execution_contexts
            ):
                raise ValueError(
                    "V3 predicate advertises unsupported actor-information context: "
                    f"{row.kind}"
                )
            if require_publishable and row.native_status != "native_exact":
                raise ValueError(
                    f"V3 published predicate is not native exact: {row.kind}"
                )

    @property
    def capabilities(self) -> tuple[ScenarioPredicateCapabilityV3, ...]:
        return self._capabilities

    @property
    def extensions(self) -> tuple[ScenarioPredicateCapabilityExtensionV3, ...]:
        return self._extensions

    @property
    def manifest_hash(self) -> str:
        return str(self._catalog["sha256"])

    def capability(self, kind: str) -> ScenarioPredicateCapabilityV3 | None:
        return next((item for item in self._capabilities if item.kind == kind), None)

    def require_publishable(self) -> None:
        for row in self._capabilities:
            if row.native_status != "native_exact":
                raise ValueError(
                    f"V3 published predicate is not native exact: {row.kind}"
                )

    def catalog(self) -> dict[str, Any]:
        return copy.deepcopy(self._catalog)


@lru_cache(maxsize=1)
def shared_scenario_predicate_capability_manifest_v3() -> (
    ScenarioPredicateCapabilityManifestV3
):
    return ScenarioPredicateCapabilityManifestV3()


__all__ = [
    "SCENARIO_PREDICATE_CAPABILITY_BACKEND_V3",
    "SCENARIO_PREDICATE_CAPABILITY_REGISTRY_V3",
    "ScenarioPredicateCapabilityManifestV3",
    "ScenarioPredicateCapabilityExtensionV3",
    "ScenarioPredicateCapabilityV3",
    "ScenarioPredicateExecutionContextV3",
    "derive_scenario_predicate_capabilities_v3",
    "native_scenario_predicate_definition_v3",
    "shared_scenario_predicate_capability_manifest_v3",
]
