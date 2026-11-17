"""Deterministic merged catalog and lowering manifest for Research V2."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Any

from .ortools_play_state_predicates import (
    PLAY_STATE_PREDICATE_REGISTRATIONS_V1,
    OrToolsPlayStatePredicateLoweringV1,
)
from .scenario_contracts import ScenarioPredicateV1, canonical_sha256
from .scenario_deal_predicates_v2 import (
    DEAL_PREDICATE_REGISTRATIONS_V2,
    DealPredicateLoweringRegistryV2,
)
from .scenario_predicate_auction_escalation_v2 import (
    AUCTION_ESCALATION_PREDICATE_REGISTRATIONS_V1,
    AuctionEscalationPredicateLoweringV1,
)
from .scenario_predicate_lowering_v2 import ScenarioPredicateLoweringRegistryV2
from .scenario_project_outcome_lowering_v2 import (
    ProjectOutcomePredicateLoweringV2,
    project_outcome_predicate_registrations_v1,
)
from .scenario_proof_tactic_predicates_v2 import (
    PROOF_TACTIC_PREDICATE_REGISTRATIONS_V2,
    ProofTacticPredicateLoweringV2,
)
from .scenario_registry_v2 import (
    ScenarioPredicateDefinitionV2,
    registered_scenario_predicate_definitions_v2,
    scenario_predicate_definitions_v2,
)


@dataclass(frozen=True, slots=True)
class ScenarioPredicateLoweringRegistrationV2:
    kind: str
    category: str
    lowerer_owner: str
    evidence_extractor_id: str
    owning_test: str

    def __post_init__(self) -> None:
        for field_name in (
            "kind",
            "category",
            "lowerer_owner",
            "evidence_extractor_id",
            "owning_test",
        ):
            value = getattr(self, field_name)
            if type(value) is not str or not value:
                raise ValueError(f"predicate registration {field_name} is required")

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "category": self.category,
            "lowererOwner": self.lowerer_owner,
            "evidenceExtractor": self.evidence_extractor_id,
            "owningTest": self.owning_test,
        }


def _merged_registrations_v2() -> tuple[ScenarioPredicateLoweringRegistrationV2, ...]:
    rows: list[ScenarioPredicateLoweringRegistrationV2] = []
    rows.extend(
        ScenarioPredicateLoweringRegistrationV2(
            item.kind,
            "deal",
            "scenario_deal_predicates_v2.DealPredicateLoweringRegistryV2",
            item.evidence_extractor_id,
            "tests/test_research_scenario_deal_predicates_v2.py",
        )
        for item in DEAL_PREDICATE_REGISTRATIONS_V2
    )
    rows.extend(
        ScenarioPredicateLoweringRegistrationV2(
            item.kind,
            "auction_escalation",
            "scenario_predicate_auction_escalation_v2."
            "AuctionEscalationPredicateLoweringV1",
            item.evidence_extractor_id,
            "tests/test_research_auction_escalation_predicates_v2.py",
        )
        for item in AUCTION_ESCALATION_PREDICATE_REGISTRATIONS_V1
    )
    rows.extend(
        ScenarioPredicateLoweringRegistrationV2(
            kind,
            "play_state",
            "ortools_play_state_predicates.OrToolsPlayStatePredicateLoweringV1",
            extractor,
            "tests/test_research_ortools_play_state_predicates.py",
        )
        for kind, extractor in PLAY_STATE_PREDICATE_REGISTRATIONS_V1.items()
    )
    rows.extend(
        ScenarioPredicateLoweringRegistrationV2(
            item.kind,
            "project_outcome",
            "scenario_project_outcome_lowering_v2.ProjectOutcomePredicateLoweringV2",
            item.evidence_extractor_id,
            "tests/test_research_project_outcome_predicate_lowering_v2.py",
        )
        for item in project_outcome_predicate_registrations_v1()
    )
    rows.extend(
        ScenarioPredicateLoweringRegistrationV2(
            kind,
            "proof_tactic",
            "scenario_proof_tactic_predicates_v2.ProofTacticPredicateLoweringV2",
            extractor,
            "tests/test_research_scenario_proof_tactic_predicates_v2.py",
        )
        for kind, extractor in PROOF_TACTIC_PREDICATE_REGISTRATIONS_V2.items()
    )
    general = ScenarioPredicateLoweringRegistryV2()
    rows.extend(
        ScenarioPredicateLoweringRegistrationV2(
            definition.kind,
            "general",
            "scenario_predicate_lowering_v2.ScenarioPredicateLoweringRegistryV2",
            definition.cp_sat_support.evidence_extractor,
            "tests/test_research_scenario_predicate_lowering_v2.py",
        )
        for definition in general.definitions
    )
    return tuple(sorted(rows, key=lambda row: row.kind))


# Backwards-compatible public projection, derived from the lowering owners rather
# than a count that must be updated whenever the authoritative catalog grows.
_CATEGORY_ORDER_V2 = (
    "deal",
    "auction_escalation",
    "play_state",
    "project_outcome",
    "proof_tactic",
    "general",
)
_category_counts_v2 = Counter(row.category for row in _merged_registrations_v2())
EXPECTED_CATEGORY_COUNTS_V2 = MappingProxyType(
    {category: _category_counts_v2[category] for category in _CATEGORY_ORDER_V2}
)


class ScenarioPredicateCatalogLoweringManifestV2:
    """One exact authority over every currently registered hard predicate."""

    def __init__(self) -> None:
        registrations = _merged_registrations_v2()
        kinds = [item.kind for item in registrations]
        if len(kinds) != len(set(kinds)):
            duplicates = sorted(kind for kind in set(kinds) if kinds.count(kind) > 1)
            raise ValueError(
                f"merged predicate registrations contain duplicates: {duplicates}"
            )
        actual_counts = Counter(item.category for item in registrations)
        counts = {
            category: actual_counts[category]
            for category in EXPECTED_CATEGORY_COUNTS_V2
        }
        if set(actual_counts) != set(EXPECTED_CATEGORY_COUNTS_V2) or counts != dict(
            EXPECTED_CATEGORY_COUNTS_V2
        ):
            raise ValueError(
                f"merged predicate category coverage drifted: {dict(actual_counts)}"
            )

        expected_registered = {
            definition.kind for definition in scenario_predicate_definitions_v2()
        }
        if set(kinds) != expected_registered:
            missing = sorted(expected_registered - set(kinds))
            extra = sorted(set(kinds) - expected_registered)
            raise ValueError(
                "merged predicate registration coverage is incomplete; "
                f"missing={missing}, extra={extra}"
            )

        extractors = {item.kind: item.evidence_extractor_id for item in registrations}
        tests = {item.kind: item.owning_test for item in registrations}
        full_definitions = registered_scenario_predicate_definitions_v2(
            extractors,
            tests,
        )
        definitions = {definition.kind: definition for definition in full_definitions}
        if set(definitions) != expected_registered:
            missing = sorted(expected_registered - set(definitions))
            extra = sorted(set(definitions) - expected_registered)
            raise AssertionError(
                "merged V2 catalog does not match authoritative definitions; "
                f"missing={missing}, extra={extra}"
            )
        exposed = tuple(definitions[kind] for kind in sorted(expected_registered))
        for definition in exposed:
            if "hard" not in definition.allowed_hardness:
                raise ValueError(
                    f"exposed predicate {definition.kind} is not hard-capable"
                )
            definition.require_publishable()
            if tests[definition.kind] not in definition.tests:
                raise ValueError(
                    f"exposed predicate {definition.kind} lacks its lowering test"
                )
        self._registrations = registrations
        self._registration_by_kind = MappingProxyType(
            {item.kind: item for item in registrations}
        )
        self._definitions = MappingProxyType(definitions)
        self._exposed = exposed
        self._hidden: tuple[ScenarioPredicateDefinitionV2, ...] = ()
        deal_lowerer = DealPredicateLoweringRegistryV2()
        auction_escalation_lowerer = AuctionEscalationPredicateLoweringV1()
        play_state_lowerer = OrToolsPlayStatePredicateLoweringV1()
        project_outcome_lowerer = ProjectOutcomePredicateLoweringV2()
        proof_tactic_lowerer = ProofTacticPredicateLoweringV2()
        general_lowerer = ScenarioPredicateLoweringRegistryV2()
        self._dispatchers = MappingProxyType(
            {
                "deal": deal_lowerer.lower_predicate,
                "auction_escalation": auction_escalation_lowerer.lower_predicate,
                "play_state": play_state_lowerer.lower_predicate,
                "project_outcome": project_outcome_lowerer.lower_predicate,
                "proof_tactic": proof_tactic_lowerer.lower_predicate,
                "general": general_lowerer.lower_predicate,
            }
        )
        component_rows = {
            category: [
                item.to_dict() for item in registrations if item.category == category
            ]
            for category in EXPECTED_CATEGORY_COUNTS_V2
        }
        core = {
            "schemaVersion": 2,
            "registryVersion": "scenario_predicate_catalog_lowering_manifest_v2",
            "categoryCounts": counts,
            "componentHashes": {
                category: canonical_sha256(component_rows[category])
                for category in sorted(component_rows)
            },
            "registrations": [item.to_dict() for item in registrations],
            "exposedDefinitions": [
                {"kind": item.kind, "sha256": item.sha256} for item in exposed
            ],
            "hiddenDefinitions": [],
        }
        self._catalog = {**core, "sha256": canonical_sha256(core)}

    @property
    def registrations(self) -> tuple[ScenarioPredicateLoweringRegistrationV2, ...]:
        return self._registrations

    @property
    def exposed_definitions(self) -> tuple[ScenarioPredicateDefinitionV2, ...]:
        return self._exposed

    @property
    def hidden_definitions(self) -> tuple[ScenarioPredicateDefinitionV2, ...]:
        return self._hidden

    @property
    def manifest_hash(self) -> str:
        return str(self._catalog["sha256"])

    def definition(self, kind: str) -> ScenarioPredicateDefinitionV2 | None:
        return self._definitions.get(str(kind))

    def catalog(self) -> dict[str, Any]:
        return {
            **self._catalog,
            "categoryCounts": dict(self._catalog["categoryCounts"]),
            "componentHashes": dict(self._catalog["componentHashes"]),
            "registrations": [dict(item) for item in self._catalog["registrations"]],
            "exposedDefinitions": [
                dict(item) for item in self._catalog["exposedDefinitions"]
            ],
            "hiddenDefinitions": [
                dict(item) for item in self._catalog["hiddenDefinitions"]
            ],
        }

    def lower_predicate(self, predicate: ScenarioPredicateV1, context: Any) -> Any:
        registration = self._registration_by_kind.get(predicate.kind)
        if registration is None:
            raise ValueError(
                f"predicate is hidden or has no exact lowering: {predicate.kind}"
            )
        return self._dispatchers[registration.category](
            predicate,
            context,
        )


@lru_cache(maxsize=1)
def shared_scenario_predicate_catalog_lowering_manifest_v2() -> (
    ScenarioPredicateCatalogLoweringManifestV2
):
    """Return the immutable process-wide predicate lowering manifest."""

    return ScenarioPredicateCatalogLoweringManifestV2()


__all__ = [
    "EXPECTED_CATEGORY_COUNTS_V2",
    "ScenarioPredicateCatalogLoweringManifestV2",
    "ScenarioPredicateLoweringRegistrationV2",
    "shared_scenario_predicate_catalog_lowering_manifest_v2",
]
