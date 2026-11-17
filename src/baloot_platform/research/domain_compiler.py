from __future__ import annotations

import copy
import importlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Iterable, Mapping

from jsonschema import Draft202012Validator

from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.escalation_ir import EscalationDomainIRV1
from baloot_platform.core.project_ir import ProjectDomainIRV1
from baloot_platform.core.play_ir import PlayDomainIRV1
from baloot_platform.core.scoring_ir import ScoringDomainIRV1
from baloot_platform.core.structural_ir import StructuralDomainIRV1

from .contracts import canonical_sha256
from .convention_proofs import (
    convention_behavioral_proofs,
    validate_convention_profile,
)


_HARD_SECTIONS = {
    "contracts",
    "seating",
    "deck",
    "card_rules",
    "dealing",
    "bidding",
    "escalation",
    "play",
    "projects",
    "scoring",
    "disputed_options",
}

_DIRECT_CP_SAT_SECTIONS = {"deck", "seating", "dealing"}

_RUNTIME_OWNERS = {
    "contracts": "runtime.bidding_v1",
    "seating": "runtime.dealing_v1",
    "deck": "runtime.cards_v1",
    "card_rules": "runtime.card_order_v1",
    "dealing": "runtime.dealing_v1",
    "bidding": "runtime.bidding_v1",
    "escalation": "runtime.bidding_v1",
    "play": "runtime.play_legality_v1",
    "projects": "runtime.projects_v1",
    "scoring": "runtime.scoring_v1",
    "disputed_options": "runtime.published_variant_gate_v1",
}

_REFERENCE_OWNERS = {key: f"reference.{key}_interpreter_v1" for key in _HARD_SECTIONS}

_CP_SAT_OWNERS = {
    "contracts": "cp_sat.complete_auction_table_v1",
    "deck": "cp_sat.unique_card_allocation_v1",
    "seating": "cp_sat.seat_allocation_v1",
    "dealing": "cp_sat.deal_position_v1",
    "bidding": "cp_sat.complete_auction_table_v1",
    "escalation": "cp_sat.complete_escalation_table_v1",
    "projects": "cp_sat.complete_projects_v1",
    "card_rules": "cp_sat.complete_play_scoring_v1",
    "play": "cp_sat.complete_play_scoring_v1",
    "scoring": "cp_sat.complete_play_scoring_v1",
}

# Executable bindings are deliberately separate from the human-readable
# lowering names above.  A declared owner is not coverage: the Python symbol
# or registered plugin capability must be present in the effective runtime.
_PYTHON_LOWERING_BINDINGS = {
    "runtime.bidding_v1": "baloot_platform.core.bidding_machine:BiddingMachine",
    "runtime.dealing_v1": "baloot_platform.core.structural_ir:StructuralDomainIRV1",
    "runtime.cards_v1": "baloot_platform.core.cards:Card",
    "runtime.card_order_v1": "baloot_platform.core.play_ir:PlayDomainIRV1",
    "runtime.play_legality_v1": "baloot_platform.core.play_ir:PlayDomainIRV1",
    "runtime.projects_v1": "baloot_platform.core.projects:detect_projects",
    "runtime.scoring_v1": "baloot_platform.core.scoring:score_round",
    "runtime.published_variant_gate_v1": "baloot_platform.core.rules:RuleBook",
    "policy.runtime_selection_v1": "baloot_platform.agents.profiles:AgentProfileRegistry",
    "policy.analysis_defaults_v1": "baloot_platform.analysis.evaluator:AnalysisPolicy",
    "belief.conservative_conventions_v1": "baloot_platform.analysis.hidden_deals:generate_hidden_deals",
    "presentation.governed_localization_v1": "baloot_platform.guide:build_engine_guide",
    "policy.bidding_heuristics_v1": "baloot_platform.core.bidding:run_bidding",
    "policy.published_variant_metadata_v1": "baloot_platform.core.rules:RuleBook",
}

_REFERENCE_BINDINGS = {
    "reference.contracts_interpreter_v1": "baloot_platform.research.reference_auction:ReferenceAuctionInterpreterV1",
    "reference.seating_interpreter_v1": "baloot_platform.research.reference_structural:ReferenceStructuralInterpreterV1",
    "reference.deck_interpreter_v1": "baloot_platform.research.reference_structural:ReferenceStructuralInterpreterV1",
    "reference.card_rules_interpreter_v1": "baloot_platform.research.reference_play:ReferencePlayInterpreterV1",
    "reference.dealing_interpreter_v1": "baloot_platform.research.reference_structural:ReferenceStructuralInterpreterV1",
    "reference.bidding_interpreter_v1": "baloot_platform.research.reference_auction:ReferenceAuctionInterpreterV1",
    "reference.escalation_interpreter_v1": "baloot_platform.research.reference_escalation:ReferenceEscalationInterpreterV1",
    "reference.play_interpreter_v1": "baloot_platform.research.reference_play:ReferencePlayInterpreterV1",
    "reference.projects_interpreter_v1": "baloot_platform.research.reference_project:ReferenceProjectInterpreterV1",
    "reference.scoring_interpreter_v1": "baloot_platform.research.reference_scoring:ReferenceScoringInterpreterV1",
    "reference.disputed_options_interpreter_v1": "baloot_platform.core.rules:RuleBook",
}

_CP_SAT_BINDINGS = {
    "cp_sat.complete_auction_table_v1": (
        "ortools_cp_sat_auction_compiler_v1",
        "domain_lowering",
        "complete_auction",
    ),
    "cp_sat.unique_card_allocation_v1": (
        "ortools_cp_sat_deal_generator_v1",
        "generator",
        "complete_deck_permutation",
    ),
    "cp_sat.seat_allocation_v1": (
        "ortools_cp_sat_deal_generator_v1",
        "generator",
        "seat_suit_shape_constraints",
    ),
    "cp_sat.deal_position_v1": (
        "ortools_cp_sat_deal_generator_v1",
        "generator",
        "complete_deck_permutation",
    ),
    "cp_sat.complete_escalation_table_v1": (
        "ortools_cp_sat_escalation_compiler_v1",
        "domain_lowering",
        "complete_escalation",
    ),
    "cp_sat.complete_projects_v1": (
        "ortools_cp_sat_project_compiler_v1",
        "domain_lowering",
        "project_comparison",
    ),
    "cp_sat.complete_play_scoring_v1": (
        "ortools_cp_sat_play_compiler_v1",
        "domain_lowering",
        "complete_play",
    ),
}

_TEST_OWNERS = {
    "contracts": (
        "tests/test_bidding_escalation.py",
        "tests/test_rules_runtime_authority.py",
    ),
    "seating": (
        "tests/test_reference_structural_v1.py",
        "tests/test_structural_compiler_v1.py",
        "tests/test_constraint_deal_search.py",
    ),
    "deck": (
        "tests/test_reference_structural_v1.py",
        "tests/test_structural_compiler_v1.py",
        "tests/test_constraint_deal_search.py",
        "tests/test_rules_runtime_authority.py",
    ),
    "card_rules": (
        "tests/test_play_obligations.py",
        "tests/test_rules_runtime_authority.py",
    ),
    "dealing": (
        "tests/test_reference_structural_v1.py",
        "tests/test_structural_compiler_v1.py",
        "tests/test_constraint_deal_search.py",
    ),
    "bidding": ("tests/test_bidding_escalation.py",),
    "escalation": ("tests/test_bidding_escalation.py",),
    "play": ("tests/test_play_obligations.py",),
    "projects": (
        "tests/test_projects_scoring_extended.py",
        "tests/test_hidden_deals_projects.py",
    ),
    "scoring": (
        "tests/test_projects_scoring_extended.py",
        "tests/test_ortools_scoring_stage_v3.py",
        "tests/test_rules_runtime_authority.py",
    ),
    "disputed_options": (
        "tests/test_rules_candidate.py",
        "tests/test_rules_runtime_authority.py",
    ),
}

_DOCS = ("docs/ENGINE_GUIDE_AR.md", "docs/OR_TOOLS_RESEARCH_FOUNDRY.md")

_EXECUTABLE_TARGETS = frozenset({"runtime", "reference", "cpSat", "policy"})

_BEHAVIORAL_PROOF_CACHE: dict[str, dict[str, dict[str, Any]]] = {}
_BEHAVIORAL_PROOF_CACHE_MAXIMUM = 8
_PRESENTATION_PROOF_CACHE: dict[str, dict[str, dict[str, Any]]] = {}

_FIXED_INVARIANT_PROBES: dict[str, Any] = {
    "rules.seating.play_order": ["north", "east", "south", "west"],
    "rules.seating.dealer_rotation": "left",
    "rules.seating.partnerships.north": "east",
    "rules.seating.partnerships.west": "south",
    "rules.seating.partnerships.south": "west",
    "rules.seating.partnerships.east": "north",
    "rules.deck.suits": ["clubs", "diamonds", "hearts", "spades"],
    "rules.deck.ranks": [
        "ace",
        "king",
        "queen",
        "jack",
        "ten",
        "nine",
        "eight",
        "seven",
    ],
    "rules.deck.players": 5,
    "rules.deck.cards_per_player": 9,
    "rules.dealing.shuffle_algorithm": "unsupported_shuffle_v1",
    "rules.dealing.cutter": "dealer_right",
    "rules.dealing.exposed_purchase_cards": 2,
    "rules.dealing.final_cards_for_buyer": 3,
    "rules.dealing.final_cards_for_others": 2,
    "rules.dealing.first_recipient": "dealer_left",
    "rules.dealing.all_pass_redeal": False,
    "rules.dealing.annulling.names": ["kawesh"],
    "rules.dealing.annulling.declaration_window": "before_bidding",
    "rules.bidding.rounds": 3,
    "rules.bidding.pass_allowed": False,
    "rules.bidding.first_round.allowed": ["pass", "sun", "hokum"],
    "rules.bidding.first_round.hokum_uses_purchase_suit": False,
    "rules.bidding.second_round.allowed": ["pass", "sun"],
    "rules.bidding.second_round.hokum_must_differ_from_purchase_suit": False,
    "rules.bidding.ashkal.enabled": False,
    "rules.bidding.ashkal.eligible_priority_positions": [2, 3],
    "rules.bidding.ashkal.taker": "bidder",
    "rules.bidding.ashkal.resolves_to": "hokum",
    "rules.escalation.default_level": "double",
    "rules.escalation.opening_actor_policy": "declarer_next",
    "rules.escalation.closure_policy": "actor_pass_sequence",
    "rules.play.must_follow_suit": False,
    "rules.play.must_trump_when_void": False,
    "rules.play.must_overtrump_when_possible": False,
    "rules.play.partner_winning_exception": "none",
    "rules.play.locked_hokum_blocks_trump_lead": False,
    "rules.play.ekka_enabled": False,
    "rules.projects.declaration_timing": "before_play",
    "rules.projects.comparison_timing": "before_play",
    "rules.projects.tie_break_order": "unsupported_order",
    "rules.projects.reveal_scope": "unsupported_scope",
    "rules.projects.top_rank_visibility": "unsupported_visibility",
    "rules.projects.reveal_timing": "after_round",
    "rules.projects.only_winning_team_scores": False,
    "rules.projects.baloot.only_hokum": False,
    "rules.projects.baloot.declare_on_second_card": False,
    "rules.projects.baloot.siri_or_fifty_inclusion": "blocked",
    "rules.projects.baloot.hundred_inclusion": "automatic",
    "rules.projects.baloot.single_card_in_other_project": "automatic",
    "rules.scoring.raw_card_totals.sun": 129,
    "rules.scoring.raw_card_totals.hokum": 161,
    "rules.scoring.conversion.sun": "truncate",
    "rules.scoring.conversion.hokum": "truncate",
    "rules.scoring.contract.buyer_team_must_win": False,
    "rules.scoring.contract.loss_transfers_all": False,
    "rules.scoring.escalation.winner_takes_all_when_multiplied": False,
    "rules.scoring.kabout.winner": "project_total",
    "rules.scoring.match.tie_above_target_continues": False,
    "rules.scoring.match.dealer_rotates_right": False,
    "rules.disputed_options.first_leader.alternatives": [],
    "rules.disputed_options.double_unspecified_mode.alternatives": [],
    "rules.disputed_options.equal_hokum_sequence_trump_priority.alternatives": [],
    "rules.disputed_options.hints_have_universal_meaning.alternatives": [True],
    "rules.disputed_options.kawesh_saneen.alternatives": [],
}

_PATH_DEFINITION_PREFIXES = {
    ("bidding", "heuristics"): "bidding_heuristics",
    ("card_rules", "sequence_strength"): "project_sequence_strength",
    (
        "disputed_options",
        "equal_hokum_sequence_trump_priority",
        "selected",
    ): "disputed_project_priority",
}


def _leaves(
    value: Any, path: tuple[str, ...] = ()
) -> Iterable[tuple[tuple[str, ...], Any]]:
    if isinstance(value, Mapping):
        for key, child in sorted(value.items(), key=lambda item: str(item[0])):
            yield from _leaves(child, (*path, str(key)))
        return
    if isinstance(value, (list, tuple)):
        # Lists are one governed value. Treating every index as a configuration
        # key would make order changes appear to add and remove rule leaves.
        yield path, copy.deepcopy(list(value))
        return
    yield path, copy.deepcopy(value)


def _replace_path(root: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    if parts and parts[0] == "rules":
        parts = parts[1:]
    cursor: dict[str, Any] = root
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            raise KeyError(dotted_path)
        cursor = child
    cursor[parts[-1]] = copy.deepcopy(value)


@dataclass(frozen=True, slots=True)
class DomainPrimitiveDefinitionV1:
    kind: str
    version: str
    category: str
    required_targets: tuple[str, ...]
    runtime_lowering: str | None
    reference_lowering: str | None
    cp_sat_lowering: str | None
    policy_lowering: str | None
    documentation: tuple[str, ...]
    tests: tuple[str, ...]
    influence_proof: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "kind": self.kind,
            "version": self.version,
            "category": self.category,
            "requiredTargets": list(self.required_targets),
            "lowerings": {
                "runtime": self.runtime_lowering,
                "reference": self.reference_lowering,
                "cpSat": self.cp_sat_lowering,
                "policy": self.policy_lowering,
            },
            "documentation": list(self.documentation),
            "tests": list(self.tests),
            "influenceProof": self.influence_proof,
        }


@dataclass(frozen=True, slots=True)
class DomainPrimitiveInstanceV1:
    primitive_id: str
    definition_kind: str
    category: str
    value: Any
    value_hash: str
    required_targets: tuple[str, ...]
    covered_targets: tuple[str, ...]
    missing_targets: tuple[str, ...]
    coverage_status: str
    executable_binding_status: str
    behavioral_or_invariant_status: str
    declared_behavioral_evidence: str | None
    verified_behavioral_evidence: str | None
    qualification_eligible: bool

    def to_dict(self, *, include_value: bool = False) -> dict[str, Any]:
        value = {
            "schemaVersion": 1,
            "primitiveId": self.primitive_id,
            "definitionKind": self.definition_kind,
            "category": self.category,
            "valueHash": self.value_hash,
            "requiredTargets": list(self.required_targets),
            "coveredTargets": list(self.covered_targets),
            "missingTargets": list(self.missing_targets),
            "coverageStatus": self.coverage_status,
            "executableBindingStatus": self.executable_binding_status,
            "behavioralOrInvariantStatus": self.behavioral_or_invariant_status,
            "declaredBehavioralEvidence": self.declared_behavioral_evidence,
            "verifiedBehavioralEvidence": self.verified_behavioral_evidence,
            "qualificationEligible": self.qualification_eligible,
        }
        if include_value:
            value["value"] = copy.deepcopy(self.value)
        return value


@dataclass(frozen=True, slots=True)
class CompilerCoverageV1:
    rows: tuple[DomainPrimitiveInstanceV1, ...]
    covered: int
    total: int
    percentage: float
    complete: bool
    executable_binding_covered: int
    executable_binding_total: int
    executable_binding_percentage: float
    executable_binding_complete: bool
    behavioral_or_invariant_covered: int
    behavioral_or_invariant_total: int
    behavioral_or_invariant_percentage: float
    behavioral_or_invariant_complete: bool
    sha256: str

    def to_dict(self, *, include_values: bool = False) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "covered": self.covered,
            "total": self.total,
            "percentage": self.percentage,
            "complete": self.complete,
            "executableBindingCovered": self.executable_binding_covered,
            "executableBindingTotal": self.executable_binding_total,
            "executableBindingPercentage": self.executable_binding_percentage,
            "executableBindingComplete": self.executable_binding_complete,
            "behavioralOrInvariantCovered": self.behavioral_or_invariant_covered,
            "behavioralOrInvariantTotal": self.behavioral_or_invariant_total,
            "behavioralOrInvariantPercentage": self.behavioral_or_invariant_percentage,
            "behavioralOrInvariantComplete": self.behavioral_or_invariant_complete,
            "rows": [row.to_dict(include_value=include_values) for row in self.rows],
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class DomainIRV1:
    rules: Mapping[str, Any]
    conventions: Mapping[str, Any]
    localization: Mapping[str, Any]
    primitive_instances: tuple[DomainPrimitiveInstanceV1, ...]
    sha256: str

    def to_dict(self, *, include_payloads: bool = False) -> dict[str, Any]:
        value = {
            "schemaVersion": 1,
            "primitiveInstances": [
                item.to_dict(include_value=include_payloads)
                for item in self.primitive_instances
            ],
            "sha256": self.sha256,
        }
        if include_payloads:
            value.update(
                {
                    "rules": copy.deepcopy(dict(self.rules)),
                    "conventions": copy.deepcopy(dict(self.conventions)),
                    "localization": copy.deepcopy(dict(self.localization)),
                }
            )
        return value


@dataclass(frozen=True, slots=True)
class CompiledDomainBundleV1:
    domain_ir: DomainIRV1
    coverage: CompilerCoverageV1
    definitions: tuple[DomainPrimitiveDefinitionV1, ...]
    runtime_plan_hash: str
    reference_plan_hash: str
    cp_sat_blueprint_hash: str
    policy_plan_hash: str
    ui_schema_hash: str
    documentation_model_hash: str
    target_payloads: Mapping[str, Any]
    publishable: bool
    sha256: str

    def to_dict(self, *, include_payloads: bool = False) -> dict[str, Any]:
        value = {
            "schemaVersion": 1,
            "compilerVersion": "domain_compiler_v1",
            "domainIr": self.domain_ir.to_dict(include_payloads=include_payloads),
            "coverage": self.coverage.to_dict(include_values=include_payloads),
            "definitions": [item.to_dict() for item in self.definitions],
            "targets": {
                "runtimePlanHash": self.runtime_plan_hash,
                "referencePlanHash": self.reference_plan_hash,
                "cpSatBlueprintHash": self.cp_sat_blueprint_hash,
                "policyPlanHash": self.policy_plan_hash,
                "uiSchemaHash": self.ui_schema_hash,
                "documentationModelHash": self.documentation_model_hash,
            },
            "publishable": self.publishable,
            "sha256": self.sha256,
        }
        if include_payloads:
            value["targetPayloads"] = copy.deepcopy(dict(self.target_payloads))
        return value

    def require_publishable(self) -> None:
        if not self.publishable:
            missing = [
                row.primitive_id
                for row in self.coverage.rows
                if not row.qualification_eligible
            ]
            raise ValueError(
                "domain compiler qualification coverage is incomplete; "
                "missing executable binding or verified behavioral/invariant proof for: "
                + ", ".join(missing[:20])
            )


@dataclass(frozen=True, slots=True)
class BalootDomainSpecV3:
    bundle_hash: str
    rules_hash: str
    convention_hash: str
    localization_hash: str
    compiled: CompiledDomainBundleV1
    plugin_capabilities: tuple[Mapping[str, Any], ...]
    sha256: str

    def to_dict(self, *, include_payloads: bool = False) -> dict[str, Any]:
        return {
            "schemaVersion": 3,
            "bundleHash": self.bundle_hash,
            "rulesHash": self.rules_hash,
            "conventionHash": self.convention_hash,
            "localizationHash": self.localization_hash,
            "compiledDomain": self.compiled.to_dict(include_payloads=include_payloads),
            "pluginCapabilities": [
                copy.deepcopy(dict(item)) for item in self.plugin_capabilities
            ],
            "sha256": self.sha256,
        }


class BalootDomainCompilerV1:
    """Truthful compiler inventory for the V3 migration.

    Executable bindings prove only that a declared lowering can be resolved.
    They do not prove that a configured value changes behavior. Publication
    additionally requires verified behavioral or invariant evidence for every
    leaf. Post-solve runtime replay is deliberately not a CP-SAT lowering.
    """

    def __init__(self, *, cache_size: int = 16) -> None:
        if not 1 <= int(cache_size) <= 256:
            raise ValueError("compiled-domain cache size must be in [1, 256]")
        self._definitions = self._build_definitions()
        self._compiled_cache_capacity = int(cache_size)
        self._compiled_cache: OrderedDict[str, BalootDomainSpecV3] = OrderedDict()
        self._compiled_cache_hits = 0
        self._compiled_cache_misses = 0
        self._compiled_cache_evictions = 0
        self._compiled_cache_lock = threading.RLock()

    def cache_metrics_snapshot(self) -> dict[str, str | int]:
        """Return aggregate metrics without exposing cached domain identities."""

        with self._compiled_cache_lock:
            return {
                "scope": "compiled_domain",
                "hits": self._compiled_cache_hits,
                "misses": self._compiled_cache_misses,
                "evictions": self._compiled_cache_evictions,
                "entries": len(self._compiled_cache),
                "capacity": self._compiled_cache_capacity,
            }

    @staticmethod
    def _validate_rules_payload(value: Mapping[str, Any]) -> None:
        """Fail closed before assigning primitive ownership to rule leaves.

        New nested keys are not implicitly owned by their parent section. A
        new primitive first needs a governed schema/plugin/compiler update.
        """
        schema_path = files("baloot_platform.resources").joinpath(
            "schemas/profiles/rules.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        errors = sorted(
            Draft202012Validator(schema).iter_errors(dict(value)),
            key=lambda item: tuple(str(part) for part in item.absolute_path),
        )
        if not errors:
            return
        first = errors[0]
        location = ".".join(str(part) for part in first.absolute_path) or "<root>"
        raise ValueError(
            f"Baloot domain rules are outside the governed primitive schema at {location}: "
            f"{first.message}"
        )

    @staticmethod
    def _build_definitions() -> dict[str, DomainPrimitiveDefinitionV1]:
        result: dict[str, DomainPrimitiveDefinitionV1] = {}
        for section in sorted(_HARD_SECTIONS):
            required = (
                "runtime",
                "reference",
                "cpSat",
                "documentation",
                "tests",
                "influence",
            )
            result[section] = DomainPrimitiveDefinitionV1(
                kind=f"rules.{section}.v1",
                version="1.0.0",
                category="semantic_rule",
                required_targets=required,
                runtime_lowering=_RUNTIME_OWNERS.get(section),
                reference_lowering=_REFERENCE_OWNERS.get(section),
                cp_sat_lowering=_CP_SAT_OWNERS.get(section),
                policy_lowering=None,
                documentation=_DOCS,
                tests=_TEST_OWNERS.get(section, ()),
                influence_proof="tests/test_domain_configuration_influence.py",
            )
        result["conventions"] = DomainPrimitiveDefinitionV1(
            kind="conventions.observer_belief.v1",
            version="1.0.0",
            category="convention",
            required_targets=("policy", "documentation", "tests", "influence"),
            runtime_lowering=None,
            reference_lowering=None,
            cp_sat_lowering=None,
            policy_lowering="belief.conservative_conventions_v1",
            documentation=_DOCS,
            tests=(
                "tests/test_convention_profile_compiler_v1.py",
                "tests/test_v105_authentic_assets_belief.py",
            ),
            influence_proof="tests/test_domain_configuration_influence.py",
        )
        result["localization"] = DomainPrimitiveDefinitionV1(
            kind="presentation.localization.v1",
            version="1.0.0",
            category="presentation",
            required_targets=("policy", "documentation", "tests", "influence"),
            runtime_lowering=None,
            reference_lowering=None,
            cp_sat_lowering=None,
            policy_lowering="presentation.governed_localization_v1",
            documentation=_DOCS,
            tests=("tests/test_foundry_dashboard.py", "tests/test_engine_guide.py"),
            influence_proof="tests/test_domain_configuration_influence.py",
        )
        result["bidding_heuristics"] = DomainPrimitiveDefinitionV1(
            kind="rules.bidding.heuristics.v1",
            version="1.0.0",
            category="policy",
            required_targets=("policy", "documentation", "tests", "influence"),
            runtime_lowering=None,
            reference_lowering=None,
            cp_sat_lowering=None,
            policy_lowering="policy.bidding_heuristics_v1",
            documentation=_DOCS,
            tests=(
                "tests/test_agent_configuration.py",
                "tests/test_bidding_escalation.py",
            ),
            influence_proof="tests/test_domain_configuration_influence.py",
        )
        result["project_sequence_strength"] = DomainPrimitiveDefinitionV1(
            kind="rules.projects.sequence_strength.v1",
            version="1.0.0",
            category="semantic_rule",
            required_targets=(
                "runtime",
                "reference",
                "cpSat",
                "documentation",
                "tests",
                "influence",
            ),
            runtime_lowering="runtime.projects_v1",
            reference_lowering="reference.projects_interpreter_v1",
            cp_sat_lowering="cp_sat.complete_projects_v1",
            policy_lowering=None,
            documentation=_DOCS,
            tests=(
                "tests/test_reference_project_v1.py",
                "tests/test_project_compiler_v1.py",
            ),
            influence_proof="tests/test_domain_configuration_influence.py",
        )
        # Disputed-option alternatives are governed documentation/policy data,
        # not active mechanics. The selected Hokum project tie-break is the one
        # exception currently consumed by runtime and CP-SAT project lowering.
        result["disputed_options"] = DomainPrimitiveDefinitionV1(
            kind="rules.disputed_options.metadata.v1",
            version="1.0.0",
            category="policy",
            required_targets=("policy", "documentation", "tests", "influence"),
            runtime_lowering=None,
            reference_lowering=None,
            cp_sat_lowering=None,
            policy_lowering="policy.published_variant_metadata_v1",
            documentation=_DOCS,
            tests=_TEST_OWNERS["disputed_options"],
            influence_proof="tests/test_domain_configuration_influence.py",
        )
        result["disputed_project_priority"] = DomainPrimitiveDefinitionV1(
            kind="rules.disputed_options.project_priority.v1",
            version="1.0.0",
            category="semantic_rule",
            required_targets=(
                "runtime",
                "reference",
                "cpSat",
                "documentation",
                "tests",
                "influence",
            ),
            runtime_lowering="runtime.projects_v1",
            reference_lowering="reference.projects_interpreter_v1",
            cp_sat_lowering="cp_sat.complete_projects_v1",
            policy_lowering=None,
            documentation=_DOCS,
            tests=(
                "tests/test_projects_scoring_extended.py",
                "tests/test_project_compiler_v1.py",
            ),
            influence_proof="tests/test_domain_configuration_influence.py",
        )
        return result

    def _definition_for_path(
        self,
        *,
        section: str,
        path: tuple[str, ...],
    ) -> DomainPrimitiveDefinitionV1:
        relative = path[1:] if path and path[0] == "rules" else path
        for prefix, definition_key in sorted(
            _PATH_DEFINITION_PREFIXES.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            if relative[: len(prefix)] == prefix:
                return self._definitions[definition_key]
        return self._definitions[section]

    @staticmethod
    def _python_receipt(*, target: str, owner: str | None) -> dict[str, Any]:
        binding_map = (
            _REFERENCE_BINDINGS if target == "reference" else _PYTHON_LOWERING_BINDINGS
        )
        symbol = binding_map.get(str(owner or ""))
        receipt = {
            "target": target,
            "owner": owner,
            "bindingType": "python_symbol",
            "symbol": symbol,
            "verified": False,
        }
        if not owner:
            receipt["failure"] = "lowering_owner_missing"
            return receipt
        if not symbol:
            receipt["failure"] = "executable_binding_missing"
            return receipt
        module_name, separator, attribute_name = symbol.partition(":")
        if not separator or not module_name or not attribute_name:
            receipt["failure"] = "executable_binding_invalid"
            return receipt
        try:
            module = importlib.import_module(module_name)
            value = getattr(module, attribute_name)
        except (ImportError, AttributeError) as exc:
            receipt["failure"] = f"executable_binding_unavailable:{type(exc).__name__}"
            return receipt
        if not callable(value):
            receipt["failure"] = "executable_binding_not_callable"
            return receipt
        receipt["verified"] = True
        receipt["proofHash"] = canonical_sha256(
            {
                "target": target,
                "owner": owner,
                "symbol": symbol,
                "module": str(getattr(module, "__name__", module_name)),
                "callable": str(getattr(value, "__qualname__", attribute_name)),
            }
        )
        return receipt

    @staticmethod
    def _cp_sat_receipt(
        *, owner: str | None, plugins: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, Any]:
        binding = _CP_SAT_BINDINGS.get(str(owner or ""))
        receipt = {
            "target": "cpSat",
            "owner": owner,
            "bindingType": "registered_plugin_capability",
            "verified": False,
        }
        if not owner:
            receipt["failure"] = "lowering_owner_missing"
            return receipt
        if binding is None:
            receipt["failure"] = "plugin_binding_missing"
            return receipt
        plugin_id, expected_kind, required_capability = binding
        receipt.update(
            {
                "pluginId": plugin_id,
                "expectedKind": expected_kind,
                "requiredCapability": required_capability,
            }
        )
        plugin = plugins.get(plugin_id)
        if plugin is None:
            receipt["failure"] = "required_plugin_missing"
            return receipt
        if str(plugin.get("kind") or "") != expected_kind:
            receipt["failure"] = "required_plugin_kind_mismatch"
            return receipt
        capabilities = {str(item) for item in plugin.get("capabilities") or ()}
        if required_capability not in capabilities:
            receipt["failure"] = "required_plugin_capability_missing"
            return receipt
        if plugin.get("deterministic") is not True:
            receipt["failure"] = "required_plugin_not_deterministic"
            return receipt
        receipt.update(
            {
                "verified": True,
                "pluginVersion": str(plugin.get("version") or ""),
                "proofHash": canonical_sha256(
                    {
                        "owner": owner,
                        "pluginId": plugin_id,
                        "pluginVersion": str(plugin.get("version") or ""),
                        "kind": expected_kind,
                        "capability": required_capability,
                        "deterministic": True,
                    }
                ),
            }
        )
        return receipt

    @classmethod
    def _executable_receipts(
        cls,
        definition: DomainPrimitiveDefinitionV1,
        *,
        plugins: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        receipts: dict[str, dict[str, Any]] = {}
        if "runtime" in definition.required_targets:
            receipts["runtime"] = cls._python_receipt(
                target="runtime", owner=definition.runtime_lowering
            )
        if "reference" in definition.required_targets:
            receipts["reference"] = cls._python_receipt(
                target="reference", owner=definition.reference_lowering
            )
        if "policy" in definition.required_targets:
            receipts["policy"] = cls._python_receipt(
                target="policy", owner=definition.policy_lowering
            )
        if "cpSat" in definition.required_targets:
            receipts["cpSat"] = cls._cp_sat_receipt(
                owner=definition.cp_sat_lowering, plugins=plugins
            )
        return receipts

    @classmethod
    def _covered_targets(
        cls,
        definition: DomainPrimitiveDefinitionV1,
        *,
        plugins: Mapping[str, Mapping[str, Any]],
    ) -> tuple[str, ...]:
        values = {
            "runtime": definition.runtime_lowering,
            "reference": definition.reference_lowering,
            "cpSat": definition.cp_sat_lowering,
            "policy": definition.policy_lowering,
            "documentation": definition.documentation,
            "tests": definition.tests,
            # A test path is a declaration, not proof that the leaf governs
            # behavior. Verified behavioral/invariant evidence is recorded
            # separately and is intentionally absent in this migration slice.
            "influence": None,
        }
        executable = cls._executable_receipts(definition, plugins=plugins)
        return tuple(
            key
            for key in definition.required_targets
            if values.get(key)
            and (key not in executable or executable[key].get("verified") is True)
        )

    @classmethod
    def _behavioral_proofs(
        cls,
        *,
        rules: Mapping[str, Any],
        plugins: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Execute bounded proofs for the domain leaves supported today.

        The receipt is produced from observed runtime/CP-SAT-lowering behavior
        or from a concrete RuleBook rejection. Merely naming a test file or
        propagating a value hash is never enough.
        """
        lowering_plugin_ids = {binding[0] for binding in _CP_SAT_BINDINGS.values()}
        cache_key = canonical_sha256(
            {
                "schemaVersion": 1,
                "rules": rules,
                "plugins": [
                    {
                        "id": str(plugin_id),
                        "kind": str(plugin.get("kind") or ""),
                        "version": str(plugin.get("version") or ""),
                        "deterministic": plugin.get("deterministic") is True,
                        "capabilities": sorted(
                            str(item) for item in plugin.get("capabilities") or ()
                        ),
                    }
                    for plugin_id, plugin in sorted(plugins.items())
                    if plugin_id in lowering_plugin_ids
                ],
            }
        )
        cached = _BEHAVIORAL_PROOF_CACHE.get(cache_key)
        if cached is not None:
            return copy.deepcopy(cached)
        from baloot_platform.core.bidding import (
            BidResult,
            _choose_action,
            _hokum_score,
            _sun_score,
        )
        from baloot_platform.core.bidding_machine import BiddingMachine
        from baloot_platform.core.cards import FULL_DECK, Card, cards_to_mask
        from baloot_platform.core.dealing import DealLayoutV1
        from baloot_platform.core.engine import BalootEngine
        from baloot_platform.core.enums import Contract, Rank, Seat, Suit, Team
        from baloot_platform.core.enums import GamePhase
        from baloot_platform.core.models import PlayedCard, Project, RoundState, Trick
        from baloot_platform.core.projects import (
            complete_baloot_teams,
            detect_projects,
        )
        from baloot_platform.core.rules import RuleBook
        from baloot_platform.core.scoring import score_round

        from .auction_compiler import OrToolsCpSatAuctionCompilerV1
        from .escalation_compiler import OrToolsCpSatEscalationCompilerV1
        from .play_compiler import OrToolsCpSatPlayCompilerV1
        from .ortools_project_model import OrToolsProjectLifecycleModelV1
        from .ortools_scoring_stage import OrToolsScoringStageV1
        from .reference_play import ReferencePlayInterpreterV1, ReferencePlayV1
        from .reference_escalation import ReferenceEscalationInterpreterV1
        from .reference_project import ReferenceProjectInterpreterV1
        from .reference_scoring import ReferenceScoringInterpreterV1
        from .reference_structural import ReferenceStructuralInterpreterV1
        from baloot_platform.challenge.constraint_deal_search import (
            OrToolsCpSatDealGeneratorV1,
        )

        cp_sat_binding = cls._cp_sat_receipt(
            owner="cp_sat.complete_auction_table_v1", plugins=plugins
        )
        if cp_sat_binding.get("verified") is not True:
            return {}

        baseline_payload = copy.deepcopy(dict(rules))
        baseline = RuleBook(baseline_payload)
        proofs: dict[str, dict[str, Any]] = {}

        class _PolicyProbeMachine:
            def __init__(
                self,
                *,
                actor: Seat,
                round_index: int,
                legal_actions: tuple[str, ...],
            ) -> None:
                self.actor = actor
                self.round_index = round_index
                self._legal_actions = legal_actions

            def legal_actions(self) -> tuple[str, ...]:
                return self._legal_actions

        def bidding_policy_observation(rulebook: RuleBook) -> dict[str, Any]:
            """Observe the actual deterministic bidding policy, not value hashes."""

            actor = Seat.NORTH
            purchase_card = Card.of(Suit.HEARTS, Rank.SEVEN)
            rank_cards = tuple(Card.of(Suit.SPADES, rank) for rank in Rank)
            sun_shape = tuple(
                Card.of(Suit.SPADES, rank)
                for rank in (Rank.ACE, Rank.TEN, Rank.KING, Rank.QUEEN, Rank.JACK)
            )
            hokum_shape = tuple(
                Card.of(Suit.HEARTS, rank)
                for rank in (Rank.JACK, Rank.NINE, Rank.ACE, Rank.TEN, Rank.KING)
            )
            mixed_hokum = (
                Card.of(Suit.HEARTS, Rank.JACK),
                Card.of(Suit.HEARTS, Rank.NINE),
                Card.of(Suit.SPADES, Rank.ACE),
                Card.of(Suit.CLUBS, Rank.TEN),
                Card.of(Suit.DIAMONDS, Rank.KING),
            )

            def choose(
                hand: tuple[Card, ...],
                *,
                round_index: int,
                actions: tuple[str, ...],
            ) -> dict[str, str | None]:
                action, trump = _choose_action(
                    _PolicyProbeMachine(
                        actor=actor,
                        round_index=round_index,
                        legal_actions=actions,
                    ),
                    {actor: hand},
                    purchase_card,
                    rulebook,
                )
                return {
                    "action": action,
                    "trump": trump.name.lower() if trump is not None else None,
                }

            heuristics = rulebook.data["bidding"]["heuristics"]
            sun_score = _sun_score(sun_shape, rulebook)
            hokum_score = _hokum_score(hokum_shape, Suit.HEARTS, rulebook)
            ashkal_score = _sun_score((*sun_shape, purchase_card), rulebook)
            return {
                "sunRankScores": [
                    round(_sun_score((card,), rulebook), 6) for card in rank_cards
                ],
                "sunShapeScore": round(sun_score, 6),
                "hokumRankScores": [
                    round(
                        _hokum_score(
                            (Card.of(Suit.HEARTS, rank),),
                            Suit.HEARTS,
                            rulebook,
                        ),
                        6,
                    )
                    for rank in Rank
                ],
                "hokumShapeScore": round(hokum_score, 6),
                "hokumOffTrumpScore": round(
                    _hokum_score(mixed_hokum, Suit.HEARTS, rulebook), 6
                ),
                "margins": {
                    "sunFirst": round(
                        sun_score - float(heuristics["sun_threshold"]), 6
                    ),
                    "hokumFirst": round(
                        hokum_score - float(heuristics["hokum_threshold"]), 6
                    ),
                    "ashkalFirst": round(
                        ashkal_score - float(heuristics["ashkal_partner_threshold"]),
                        6,
                    ),
                    "sunSecond": round(
                        sun_score - float(heuristics["second_sun_threshold"]),
                        6,
                    ),
                    "hokumSecond": round(
                        hokum_score - float(heuristics["second_hokum_threshold"]),
                        6,
                    ),
                },
                "choices": {
                    "sunFirst": choose(
                        sun_shape, round_index=1, actions=("pass", "sun")
                    ),
                    "hokumFirst": choose(
                        hokum_shape, round_index=1, actions=("pass", "hokum")
                    ),
                    "ashkalFirst": choose(
                        sun_shape, round_index=1, actions=("pass", "ashkal")
                    ),
                    "sunSecond": choose(
                        sun_shape, round_index=2, actions=("pass", "sun")
                    ),
                    "hokumSecond": choose(
                        hokum_shape, round_index=2, actions=("pass", "hokum")
                    ),
                },
            }

        heuristic_counterfactuals: dict[str, Any] = {
            "rules.bidding.heuristics.sun_rank_weights": [
                float(value) + (0.75 if index == 0 else 0.0)
                for index, value in enumerate(
                    baseline_payload["bidding"]["heuristics"]["sun_rank_weights"]
                )
            ],
            "rules.bidding.heuristics.sun_extra_suit_card_weight": float(
                baseline_payload["bidding"]["heuristics"]["sun_extra_suit_card_weight"]
            )
            + 0.5,
            "rules.bidding.heuristics.sun_extra_suit_card_after": int(
                baseline_payload["bidding"]["heuristics"]["sun_extra_suit_card_after"]
            )
            + 1,
            "rules.bidding.heuristics.hokum_trump_rank_weights": [
                float(value) + (0.75 if index == 0 else 0.0)
                for index, value in enumerate(
                    baseline_payload["bidding"]["heuristics"][
                        "hokum_trump_rank_weights"
                    ]
                )
            ],
            "rules.bidding.heuristics.hokum_extra_trump_card_weight": float(
                baseline_payload["bidding"]["heuristics"][
                    "hokum_extra_trump_card_weight"
                ]
            )
            + 0.5,
            "rules.bidding.heuristics.hokum_extra_trump_card_after": int(
                baseline_payload["bidding"]["heuristics"][
                    "hokum_extra_trump_card_after"
                ]
            )
            + 1,
            "rules.bidding.heuristics.hokum_off_trump_ace_ten_weight": float(
                baseline_payload["bidding"]["heuristics"][
                    "hokum_off_trump_ace_ten_weight"
                ]
            )
            + 0.5,
            "rules.bidding.heuristics.sun_threshold": float(
                baseline_payload["bidding"]["heuristics"]["sun_threshold"]
            )
            + 100.0,
            "rules.bidding.heuristics.hokum_threshold": float(
                baseline_payload["bidding"]["heuristics"]["hokum_threshold"]
            )
            + 100.0,
            "rules.bidding.heuristics.ashkal_partner_threshold": float(
                baseline_payload["bidding"]["heuristics"]["ashkal_partner_threshold"]
            )
            + 100.0,
            "rules.bidding.heuristics.second_sun_threshold": float(
                baseline_payload["bidding"]["heuristics"]["second_sun_threshold"]
            )
            + 100.0,
            "rules.bidding.heuristics.second_hokum_threshold": float(
                baseline_payload["bidding"]["heuristics"]["second_hokum_threshold"]
            )
            + 100.0,
        }
        baseline_policy_observation = bidding_policy_observation(baseline)
        for primitive_id, counterfactual_value in sorted(
            heuristic_counterfactuals.items()
        ):
            counterfactual_payload = copy.deepcopy(baseline_payload)
            _replace_path(counterfactual_payload, primitive_id, counterfactual_value)
            counterfactual_rules = RuleBook(counterfactual_payload)
            counterfactual_observation = bidding_policy_observation(
                counterfactual_rules
            )
            if baseline_policy_observation == counterfactual_observation:
                raise ValueError(
                    f"bidding heuristic counterfactual did not influence {primitive_id}"
                )
            core = {
                "schemaVersion": 1,
                "primitiveId": primitive_id,
                "status": "proven_behavioral",
                "proofKind": "bidding_policy_counterfactual",
                "rulesHash": baseline.sha256,
                "counterfactualRulesHash": counterfactual_rules.sha256,
                "observationHash": canonical_sha256(
                    [baseline_policy_observation, counterfactual_observation]
                ),
                "targetsObserved": ["policy"],
                "policyOwner": "baloot_platform.core.bidding._choose_action",
                "proofBoundary": "deterministic_policy_scores_margins_and_actions",
            }
            proofs[primitive_id] = {
                **core,
                "proofHash": canonical_sha256(core),
            }

        rule_leaf_ids = {".".join(path) for path, _ in _leaves(baseline_payload, ("rules",))}
        for primitive_id, counterexample in sorted(_FIXED_INVARIANT_PROBES.items()):
            if primitive_id not in rule_leaf_ids:
                # Rejecting an invented leaf is not evidence for the selected profile.
                continue
            if primitive_id.startswith(
                ("rules.deck.", "rules.seating.", "rules.dealing.")
            ):
                # Structural invariants receive target-executed receipts only
                # after the shared IR has passed runtime/reference/CP-SAT
                # parity below. A generic RuleBook rejection is insufficient.
                continue
            mutated = copy.deepcopy(baseline_payload)
            _replace_path(mutated, primitive_id, counterexample)
            try:
                RuleBook(mutated)
            except ValueError as exc:
                core = {
                    "schemaVersion": 1,
                    "primitiveId": primitive_id,
                    "status": "proven_invariant",
                    "proofKind": "runtime_rulebook_counterexample_rejection",
                    "rulesHash": baseline.sha256,
                    "counterexampleHash": canonical_sha256(counterexample),
                    "rejectionType": type(exc).__name__,
                    "targetsObserved": ["runtime", "reference", "cpSat"],
                    "proofBoundary": "shared_rulebook_admission_gate",
                }
                proofs[primitive_id] = {**core, "proofHash": canonical_sha256(core)}

        structural_bindings = (
            cls._cp_sat_receipt(
                owner="cp_sat.unique_card_allocation_v1", plugins=plugins
            ),
            cls._cp_sat_receipt(owner="cp_sat.seat_allocation_v1", plugins=plugins),
            cls._cp_sat_receipt(owner="cp_sat.deal_position_v1", plugins=plugins),
        )
        structural_plugin = plugins.get("ortools_cp_sat_deal_generator_v1") or {}
        structural_capabilities = {
            str(item) for item in structural_plugin.get("capabilities") or ()
        }
        structural_ready = (
            all(item.get("verified") is True for item in structural_bindings)
            and "domain_structural_semantics" in structural_capabilities
            and callable(
                getattr(OrToolsCpSatDealGeneratorV1, "compile_structural_domain", None)
            )
        )
        if structural_ready:
            structural_hand_ranks = (
                "seven",
                "eight",
                "nine",
                "seven",
                "eight",
            )

            def runtime_structure(
                rulebook: RuleBook,
                *,
                skip_cut: bool = False,
                hand_ranks: tuple[str, ...] = structural_hand_ranks,
            ) -> dict[str, Any]:
                engine = BalootEngine(rulebook)
                state = engine.create_round(seed=1905, dealer=Seat.EAST)
                try:
                    engine.shuffle_and_initial_deal(
                        state,
                        cut_position=11,
                        skip_cut=skip_cut,
                    )
                except ValueError as exc:
                    if skip_cut and "requires a cut" in str(exc):
                        return {"accepted": False}
                    raise
                layout = DealLayoutV1.build(
                    engine.structural_domain,
                    dealer=Seat.EAST,
                    buyer=Seat.SOUTH,
                )
                rank_values = tuple(Rank.parse(value) for value in hand_ranks)
                cards = tuple(
                    Card.of(suit, rank)
                    for suit, rank in zip(
                        (
                            Suit.SPADES,
                            Suit.HEARTS,
                            Suit.DIAMONDS,
                            Suit.CLUBS,
                            Suit.SPADES,
                        ),
                        rank_values,
                        strict=True,
                    )
                )
                annul_state = engine.create_round(seed=1, dealer=Seat.EAST)
                annul_state.phase = GamePhase.INITIAL_DEALT
                annul_state.initial_five[int(Seat.NORTH)] = cards_to_mask(cards)
                annul_state.hands[int(Seat.NORTH)] = cards_to_mask(cards)
                annul_eligible = engine.can_annul_initial_deal(annul_state, Seat.NORTH)
                dealer_advances = None
                if annul_eligible:
                    engine.annul_initial_deal(
                        annul_state,
                        Seat.NORTH,
                        reason=engine.structural_domain.annulling_names[0],
                    )
                    dealer_advances = bool(
                        annul_state.events[-1].data["dealerAdvances"]
                    )
                assert state.shuffle is not None
                all_passed = not engine.complete_deal_from_bid_result(
                    state, BidResult((), True)
                )
                all_pass_event = state.events[-1]
                return {
                    **engine.structural_domain.semantic_observation(dealer=Seat.EAST),
                    "accepted": True,
                    "cutApplied": state.shuffle.cut_position is not None,
                    "cutPosition": state.shuffle.cut_position,
                    "cutter": (
                        state.shuffle.cutter.name.lower()
                        if state.shuffle.cutter is not None
                        else None
                    ),
                    "deckOrder": list(state.shuffle.deck_order),
                    "initialPackets": [
                        {
                            "packet": packet.packet,
                            "seat": packet.seat.name.lower(),
                            "positions": list(packet.positions),
                        }
                        for packet in layout.initial_packets
                    ],
                    "finalPackets": [
                        {
                            "seat": packet.seat.name.lower(),
                            "positions": list(packet.positions),
                        }
                        for packet in layout.final_packets
                    ],
                    "purchasePosition": layout.purchase_position,
                    "undealtPositions": list(layout.undealt_positions),
                    "initialPositions": [
                        list(value) for value in layout.initial_positions
                    ],
                    "finalPositions": [
                        list(value) for value in layout.final_hand_positions or ()
                    ],
                    "annullingEligible": annul_eligible,
                    "annullingDealerAdvances": dealer_advances,
                    "allPassRedeal": {
                        "redeal": all_passed,
                        "nextDealer": all_pass_event.data.get("nextDealer"),
                    },
                }

            def reference_structure(
                rulebook: RuleBook,
                *,
                skip_cut: bool = False,
                hand_ranks: tuple[str, ...] = structural_hand_ranks,
            ) -> dict[str, Any]:
                reference = ReferenceStructuralInterpreterV1(rulebook.structural_domain)
                shuffle = reference.shuffle_observation(
                    seed=1905,
                    dealer=Seat.EAST,
                    cut_position=11,
                    skip_cut=skip_cut,
                )
                if not shuffle["accepted"]:
                    return shuffle
                layout = reference.layout(dealer=Seat.EAST, buyer=Seat.SOUTH).to_dict()
                annul = reference.annulling_observation(hand_ranks)
                return {
                    **reference.semantic_observation(dealer=Seat.EAST),
                    **shuffle,
                    "initialPackets": layout["initialPackets"],
                    "finalPackets": layout["finalPackets"],
                    "purchasePosition": layout["purchasePosition"],
                    "undealtPositions": layout["undealtPositions"],
                    "initialPositions": layout["initialPositions"],
                    "finalPositions": layout["finalPositions"],
                    "annullingEligible": annul["eligible"],
                    "annullingDealerAdvances": annul["dealerAdvances"],
                    "allPassRedeal": reference.all_pass_observation(dealer=Seat.EAST),
                }

            def cp_sat_structure(
                rulebook: RuleBook,
                *,
                skip_cut: bool = False,
                hand_ranks: tuple[str, ...] = structural_hand_ranks,
            ) -> dict[str, Any]:
                plan = OrToolsCpSatDealGeneratorV1.compile_structural_domain(
                    rules=rulebook.structural_domain,
                    dealer=Seat.EAST,
                    buyer=Seat.SOUTH,
                    seed=1905,
                    cut_position=11,
                    skip_cut=skip_cut,
                    annulling_hand_ranks=hand_ranks,
                )
                return plan.semantic_observation()

            def behavioral_projection(
                observation: Mapping[str, Any],
            ) -> dict[str, Any]:
                return {
                    key: copy.deepcopy(value)
                    for key, value in observation.items()
                    if key
                    not in {
                        "structuralDomainIrHash",
                        "deck",
                        "seating",
                        "dealing",
                    }
                }

            baseline_structural_observations = [
                reference_structure(baseline),
                runtime_structure(baseline),
                cp_sat_structure(baseline),
            ]
            if not all(
                observation == baseline_structural_observations[0]
                for observation in baseline_structural_observations[1:]
            ):
                raise ValueError("structural baseline compiler parity failure")
            baseline_structural_hash = canonical_sha256(
                baseline_structural_observations[0]
            )
            dynamic_structural_ids = {
                "rules.dealing.allow_no_cut",
                "rules.dealing.annulling.dealer_advances",
                "rules.dealing.annulling.eligible_ranks",
                "rules.dealing.annulling.enabled",
                "rules.dealing.cut_enabled",
                "rules.dealing.initial_packets",
            }
            all_structural_ids = {
                ".".join(path)
                for section in ("deck", "seating", "dealing")
                for path, _value in _leaves(
                    baseline_payload[section], ("rules", section)
                )
            }
            fixed_structural_ids = set(_FIXED_INVARIANT_PROBES) & (
                all_structural_ids - dynamic_structural_ids
            )
            if len(all_structural_ids) != 25:
                raise ValueError(
                    "structural DomainIR must own exactly 25 governed leaves"
                )
            if fixed_structural_ids != (all_structural_ids - dynamic_structural_ids):
                raise ValueError(
                    "structural invariant probes do not cover every fixed leaf"
                )
            for primitive_id in sorted(fixed_structural_ids):
                counterexample = _FIXED_INVARIANT_PROBES[primitive_id]
                mutated = copy.deepcopy(baseline_payload)
                _replace_path(mutated, primitive_id, counterexample)
                try:
                    StructuralDomainIRV1.from_rules(mutated)
                except ValueError as exc:
                    try:
                        RuleBook(mutated)
                    except ValueError as runtime_exc:
                        core = {
                            "schemaVersion": 1,
                            "primitiveId": primitive_id,
                            "status": "proven_invariant",
                            "proofKind": (
                                "structural_domain_ir_counterexample_rejection"
                            ),
                            "rulesHash": baseline.sha256,
                            "counterexampleHash": canonical_sha256(counterexample),
                            "rejectionType": type(exc).__name__,
                            "runtimeRejectionType": type(runtime_exc).__name__,
                            "baselineObservationHash": (baseline_structural_hash),
                            "targetsObserved": [
                                "reference",
                                "runtime",
                                "cpSat",
                            ],
                            "referenceIndependent": True,
                            "proofBoundary": ("shared_structural_ir_admission_gate"),
                        }
                        proofs[primitive_id] = {
                            **core,
                            "proofHash": canonical_sha256(core),
                        }
                    else:
                        raise ValueError(
                            f"runtime accepted invalid structural leaf {primitive_id}"
                        )
                else:
                    raise ValueError(
                        f"StructuralDomainIR accepted invalid leaf {primitive_id}"
                    )

            def prove_structural_counterfactual(
                *,
                primitive_id: str,
                baseline_rules: RuleBook,
                counterfactual_rules: RuleBook,
                skip_cut: bool = False,
                activation_rules_hash: str | None = None,
                governing_rules_hash: str | None = None,
            ) -> None:
                baseline_observations = [
                    reference_structure(baseline_rules, skip_cut=skip_cut),
                    runtime_structure(baseline_rules, skip_cut=skip_cut),
                    cp_sat_structure(baseline_rules, skip_cut=skip_cut),
                ]
                counterfactual_observations = [
                    reference_structure(counterfactual_rules, skip_cut=skip_cut),
                    runtime_structure(counterfactual_rules, skip_cut=skip_cut),
                    cp_sat_structure(counterfactual_rules, skip_cut=skip_cut),
                ]
                if not all(
                    item == baseline_observations[0]
                    for item in baseline_observations[1:]
                ):
                    raise ValueError(
                        f"structural baseline compiler parity failure for {primitive_id}"
                    )
                if not all(
                    item == counterfactual_observations[0]
                    for item in counterfactual_observations[1:]
                ):
                    raise ValueError(
                        f"structural counterfactual compiler parity failure for {primitive_id}"
                    )
                baseline_behavior = behavioral_projection(baseline_observations[0])
                counterfactual_behavior = behavioral_projection(
                    counterfactual_observations[0]
                )
                if baseline_behavior == counterfactual_behavior:
                    raise ValueError(
                        f"structural counterfactual did not influence {primitive_id}"
                    )
                core = {
                    "schemaVersion": 1,
                    "primitiveId": primitive_id,
                    "status": "proven_behavioral",
                    "proofKind": "structural_reference_runtime_cp_sat_counterfactual",
                    "rulesHash": governing_rules_hash or baseline_rules.sha256,
                    "counterfactualRulesHash": counterfactual_rules.sha256,
                    "observationHash": canonical_sha256(
                        [baseline_behavior, counterfactual_behavior]
                    ),
                    "targetsObserved": ["reference", "runtime", "cpSat"],
                    "referenceIndependent": True,
                    "proofBoundary": "deterministic_deck_layout_and_annulling_probe",
                }
                if activation_rules_hash is not None:
                    core["activationRulesHash"] = activation_rules_hash
                proofs[primitive_id] = {
                    **core,
                    "proofHash": canonical_sha256(core),
                }

            cut_baseline_payload = copy.deepcopy(baseline_payload)
            cut_baseline_payload["dealing"]["allow_no_cut"] = True
            cut_baseline = RuleBook(cut_baseline_payload)
            cut_payload = copy.deepcopy(cut_baseline_payload)
            cut_payload["dealing"]["cut_enabled"] = not bool(
                cut_baseline_payload["dealing"]["cut_enabled"]
            )
            prove_structural_counterfactual(
                primitive_id="rules.dealing.cut_enabled",
                baseline_rules=cut_baseline,
                counterfactual_rules=RuleBook(cut_payload),
                activation_rules_hash=(
                    cut_baseline.sha256
                    if cut_baseline.sha256 != baseline.sha256
                    else None
                ),
                governing_rules_hash=baseline.sha256,
            )

            no_cut_baseline_payload = copy.deepcopy(baseline_payload)
            no_cut_baseline_payload["dealing"]["cut_enabled"] = True
            no_cut_baseline = RuleBook(no_cut_baseline_payload)
            no_cut_payload = copy.deepcopy(no_cut_baseline_payload)
            no_cut_payload["dealing"]["allow_no_cut"] = not bool(
                no_cut_baseline_payload["dealing"]["allow_no_cut"]
            )
            prove_structural_counterfactual(
                primitive_id="rules.dealing.allow_no_cut",
                baseline_rules=no_cut_baseline,
                counterfactual_rules=RuleBook(no_cut_payload),
                skip_cut=True,
                activation_rules_hash=(
                    no_cut_baseline.sha256
                    if no_cut_baseline.sha256 != baseline.sha256
                    else None
                ),
                governing_rules_hash=baseline.sha256,
            )

            packets_payload = copy.deepcopy(baseline_payload)
            packets_payload["dealing"]["initial_packets"] = (
                [2, 3]
                if list(packets_payload["dealing"]["initial_packets"]) != [2, 3]
                else [3, 2]
            )
            prove_structural_counterfactual(
                primitive_id="rules.dealing.initial_packets",
                baseline_rules=baseline,
                counterfactual_rules=RuleBook(packets_payload),
            )

            annul_enabled_payload = copy.deepcopy(baseline_payload)
            annul_enabled_payload["dealing"]["annulling"]["enabled"] = not bool(
                baseline_payload["dealing"]["annulling"]["enabled"]
            )
            prove_structural_counterfactual(
                primitive_id="rules.dealing.annulling.enabled",
                baseline_rules=baseline,
                counterfactual_rules=RuleBook(annul_enabled_payload),
            )

            annul_active_payload = copy.deepcopy(baseline_payload)
            annul_active_payload["dealing"]["annulling"]["enabled"] = True
            annul_active = RuleBook(annul_active_payload)
            annul_ranks_payload = copy.deepcopy(annul_active_payload)
            annul_ranks_payload["dealing"]["annulling"]["eligible_ranks"] = (
                ["seven", "eight"]
                if list(annul_active_payload["dealing"]["annulling"]["eligible_ranks"])
                != ["seven", "eight"]
                else ["nine", "eight", "seven"]
            )
            prove_structural_counterfactual(
                primitive_id="rules.dealing.annulling.eligible_ranks",
                baseline_rules=annul_active,
                counterfactual_rules=RuleBook(annul_ranks_payload),
                activation_rules_hash=(
                    annul_active.sha256
                    if annul_active.sha256 != baseline.sha256
                    else None
                ),
                governing_rules_hash=baseline.sha256,
            )

            annul_dealer_payload = copy.deepcopy(annul_active_payload)
            annul_dealer_payload["dealing"]["annulling"]["dealer_advances"] = not bool(
                annul_active_payload["dealing"]["annulling"]["dealer_advances"]
            )
            prove_structural_counterfactual(
                primitive_id="rules.dealing.annulling.dealer_advances",
                baseline_rules=annul_active,
                counterfactual_rules=RuleBook(annul_dealer_payload),
                activation_rules_hash=(
                    annul_active.sha256
                    if annul_active.sha256 != baseline.sha256
                    else None
                ),
                governing_rules_hash=baseline.sha256,
            )

        allowed = tuple(str(item) for item in baseline_payload["contracts"]["allowed"])
        auction_compiler = OrToolsCpSatAuctionCompilerV1()

        def prove_auction_counterfactual(
            *,
            primitive_id: str,
            mutate: Any,
            required_actions: tuple[str, ...],
            required_actions_complete: bool,
            purchase_rank: Rank | None,
            proof_boundary: str,
        ) -> None:
            counterfactual_payload = copy.deepcopy(baseline_payload)
            mutate(counterfactual_payload)
            counterfactual = RuleBook(counterfactual_payload)

            def observe(rulebook: RuleBook) -> dict[str, Any]:
                parity = auction_compiler.reference_runtime_parity(
                    dealer=Seat.EAST,
                    purchase_suit=Suit.HEARTS,
                    purchase_rank=purchase_rank,
                    rules=rulebook,
                )
                plan = auction_compiler.compile(
                    dealer=Seat.EAST,
                    purchase_suit=Suit.HEARTS,
                    purchase_rank=purchase_rank,
                    rules=rulebook,
                    required_actions=(
                        {"action": action} for action in required_actions
                    ),
                    required_actions_complete=required_actions_complete,
                    timeout_seconds=5.0,
                    maximum_conflicts=100_000,
                )
                return {
                    "referenceRuntimeParityHash": parity["proofHash"],
                    "traceCount": parity["traceCount"],
                    "traceSetHash": parity["traceSetHash"],
                    "cpSatPlanHash": plan.sha256 if plan is not None else None,
                    "result": (
                        None
                        if plan is None
                        else {
                            "buyer": plan.buyer.name.lower(),
                            "purchaseRecipient": (
                                plan.purchase_recipient.name.lower()
                                if plan.purchase_recipient is not None
                                else None
                            ),
                            "contract": plan.contract.value,
                            "actions": [item.action for item in plan.actions],
                        }
                    ),
                }

            baseline_observation = observe(baseline)
            counterfactual_observation = observe(counterfactual)
            if baseline_observation == counterfactual_observation:
                raise ValueError(
                    f"auction counterfactual did not influence {primitive_id}"
                )
            core = {
                "schemaVersion": 1,
                "primitiveId": primitive_id,
                "status": "proven_behavioral",
                "proofKind": "auction_shared_ir_reference_runtime_cp_sat_counterfactual",
                "rulesHash": baseline.sha256,
                "counterfactualRulesHash": counterfactual.sha256,
                "observationHash": canonical_sha256(
                    [baseline_observation, counterfactual_observation]
                ),
                "referenceRuntimeParityHashes": [
                    baseline_observation["referenceRuntimeParityHash"],
                    counterfactual_observation["referenceRuntimeParityHash"],
                ],
                "targetsObserved": ["reference", "runtime", "cpSat"],
                "referenceIndependent": False,
                "proofBoundary": proof_boundary,
            }
            proofs[primitive_id] = {**core, "proofHash": canonical_sha256(core)}

        ashkal = baseline_payload["bidding"]["ashkal"]
        if (
            bool(ashkal.get("enabled"))
            and "ashkal" in allowed
            and "contract_owner" in ashkal
        ):
            contract_owner = str(ashkal.get("contract_owner", "taker"))
            counterfactual_contract_owner = (
                "taker" if contract_owner == "bidder" else "bidder"
            )
            prove_auction_counterfactual(
                primitive_id="rules.bidding.ashkal.contract_owner",
                mutate=lambda payload: payload["bidding"]["ashkal"].__setitem__(
                    "contract_owner", counterfactual_contract_owner
                ),
                required_actions=("pass", "pass", "ashkal", "pass", "pass"),
                required_actions_complete=True,
                purchase_rank=Rank.SEVEN,
                proof_boundary=(
                    "ashkal_caller_contract_owner_distinct_from_purchase_card_taker"
                ),
            )

        exposed_ace_mode = baseline_payload["bidding"]["first_round"].get(
            "exposed_ace_hokum_sun_overcall"
        )
        if exposed_ace_mode in {"dealer_right_only", "priority_recall"}:
            counterfactual_mode = (
                "priority_recall"
                if exposed_ace_mode == "dealer_right_only"
                else "dealer_right_only"
            )
            prove_auction_counterfactual(
                primitive_id=(
                    "rules.bidding.first_round.exposed_ace_hokum_sun_overcall"
                ),
                mutate=lambda payload: payload["bidding"]["first_round"].__setitem__(
                    "exposed_ace_hokum_sun_overcall", counterfactual_mode
                ),
                required_actions=("pass", "pass", "hokum", "pass", "sun"),
                required_actions_complete=False,
                purchase_rank=Rank.ACE,
                proof_boundary=(
                    "exposed_ace_rank_aware_hokum_to_sun_recall_restriction"
                ),
            )

        if "ashkal" in allowed and not bool(
            baseline_payload["contracts"].get("require_preselected")
        ):
            counterfactual_payload = copy.deepcopy(baseline_payload)
            counterfactual_allowed = [item for item in allowed if item != "ashkal"]
            counterfactual_payload["contracts"]["allowed"] = counterfactual_allowed
            counterfactual_payload["bidding"]["first_round"]["allowed"] = [
                "pass",
                *counterfactual_allowed,
            ]
            counterfactual_payload["bidding"]["second_round"]["allowed"] = [
                "pass",
                *[item for item in counterfactual_allowed if item != "ashkal"],
            ]
            counterfactual = RuleBook(counterfactual_payload)

            def observed_actions(rulebook: RuleBook) -> tuple[str, ...]:
                machine = BiddingMachine(
                    Seat.EAST,
                    Suit.HEARTS,
                    AuctionDomainIRV1.from_rules(rulebook.data),
                )
                machine.submit("pass")
                machine.submit("pass")
                return machine.legal_actions()

            compiler = OrToolsCpSatAuctionCompilerV1()
            baseline_actions = observed_actions(baseline)
            counterfactual_actions = observed_actions(counterfactual)
            baseline_parity = compiler.reference_runtime_parity(
                dealer=Seat.EAST, purchase_suit=Suit.HEARTS, rules=baseline
            )
            counterfactual_parity = compiler.reference_runtime_parity(
                dealer=Seat.EAST,
                purchase_suit=Suit.HEARTS,
                rules=counterfactual,
            )
            baseline_plans = compiler._legal_auctions(
                dealer=Seat.EAST, purchase_suit=Suit.HEARTS, rules=baseline
            )
            counterfactual_plans = compiler._legal_auctions(
                dealer=Seat.EAST, purchase_suit=Suit.HEARTS, rules=counterfactual
            )
            baseline_ashkal = sum(
                item.contract == Contract.ASHKAL for item in baseline_plans
            )
            counterfactual_ashkal = sum(
                item.contract == Contract.ASHKAL for item in counterfactual_plans
            )
            if (
                baseline_actions != counterfactual_actions
                and baseline_ashkal > 0
                and counterfactual_ashkal == 0
            ):
                core = {
                    "schemaVersion": 1,
                    "primitiveId": "rules.contracts.allowed",
                    "status": "proven_behavioral",
                    "proofKind": "auction_reference_runtime_and_cp_sat_table_counterfactual",
                    "rulesHash": baseline.sha256,
                    "counterfactualRulesHash": counterfactual.sha256,
                    "runtimeObservationHash": canonical_sha256(
                        {
                            "baseline": list(baseline_actions),
                            "counterfactual": list(counterfactual_actions),
                        }
                    ),
                    "cpSatObservationHash": canonical_sha256(
                        {
                            "baselinePlanCount": len(baseline_plans),
                            "counterfactualPlanCount": len(counterfactual_plans),
                            "baselineAshkalCount": baseline_ashkal,
                            "counterfactualAshkalCount": counterfactual_ashkal,
                        }
                    ),
                    "referenceRuntimeParityHashes": [
                        baseline_parity["proofHash"],
                        counterfactual_parity["proofHash"],
                    ],
                    "targetsObserved": ["reference", "runtime", "cpSat"],
                    "referenceIndependent": False,
                    "proofBoundary": "exhaustive_auction_trace_set_for_counterfactual_profiles",
                }
                proofs["rules.contracts.allowed"] = {
                    **core,
                    "proofHash": canonical_sha256(core),
                }

        if not bool(baseline_payload["contracts"].get("require_preselected")):
            preselected_payload = copy.deepcopy(baseline_payload)
            preselected_payload["contracts"].update(
                {
                    "allowed": ["sun"],
                    "default": "sun",
                    "require_preselected": True,
                }
            )
            preselected_payload["bidding"]["first_round"].update(
                {
                    "allowed": ["pass", "sun"],
                    "hokum_uses_purchase_suit": False,
                }
            )
            preselected_payload["bidding"]["second_round"].update(
                {
                    "allowed": ["pass", "sun"],
                    "hokum_must_differ_from_purchase_suit": False,
                }
            )
            preselected = RuleBook(preselected_payload)
            compiler = OrToolsCpSatAuctionCompilerV1()
            parity = compiler.reference_runtime_parity(
                dealer=Seat.EAST,
                purchase_suit=Suit.HEARTS,
                rules=preselected,
            )
            plan = compiler.compile(
                dealer=Seat.EAST,
                purchase_suit=Suit.HEARTS,
                rules=preselected,
            )
            if (
                plan is not None
                and plan.contract == Contract.SUN
                and plan.purchase_round == 1
                and parity["traceCount"] == 1
            ):
                shared = {
                    "schemaVersion": 1,
                    "status": "proven_behavioral",
                    "proofKind": "preselected_contract_joint_counterfactual",
                    "rulesHash": baseline.sha256,
                    "counterfactualRulesHash": preselected.sha256,
                    "counterfactualAuctionDomainIrHash": parity["auctionDomainIrHash"],
                    "referenceRuntimeParityHash": parity["proofHash"],
                    "cpSatPlanHash": plan.sha256,
                    "targetsObserved": ["reference", "runtime", "cpSat"],
                    "referenceIndependent": True,
                    "proofBoundary": "minimal_valid_cross_field_preselected_profile",
                    "jointPrimitiveIds": [
                        "rules.contracts.default",
                        "rules.contracts.require_preselected",
                    ],
                }
                for primitive_id in shared["jointPrimitiveIds"]:
                    core = {**shared, "primitiveId": primitive_id}
                    proofs[primitive_id] = {
                        **core,
                        "proofHash": canonical_sha256(core),
                    }

        escalation_binding = cls._cp_sat_receipt(
            owner="cp_sat.complete_escalation_table_v1", plugins=plugins
        )
        if escalation_binding.get("verified") is True:
            escalation_compiler = OrToolsCpSatEscalationCompilerV1()

            def escalation_observation(
                rulebook: RuleBook,
                *,
                contract: Contract,
                match_score: tuple[int, int],
                actions: tuple[tuple[Seat, str, str | None], ...],
            ) -> dict[str, Any] | None:
                declarer = Seat.SOUTH
                runtime_state = escalation_compiler._root_state(
                    rules=rulebook,
                    dealer=Seat.EAST,
                    declarer=declarer,
                    contract=contract,
                    match_score=match_score,
                    domain=EscalationDomainIRV1.from_rules(rulebook.data),
                )
                try:
                    for actor, action, mode in actions:
                        BalootEngine(rulebook).apply_escalation(
                            runtime_state,
                            seat=actor,
                            action=action,
                            mode=mode,
                        )
                except ValueError:
                    runtime_value = None
                else:
                    runtime_value = {
                        "level": runtime_state.escalation,
                        "mode": "locked" if runtime_state.locked else "open",
                        "doubleSeat": (
                            runtime_state.double_seat.name.lower()
                            if runtime_state.double_seat is not None
                            else None
                        ),
                        "actions": [
                            [
                                str(item["seat"]),
                                str(item["action"]),
                                str(item["mode"]),
                            ]
                            for item in runtime_state.escalation_history
                        ],
                    }

                reference = ReferenceEscalationInterpreterV1.from_rules(rulebook.data)
                try:
                    reference_plan = reference.apply_sequence(
                        declarer=declarer,
                        contract=contract,
                        match_score=match_score,
                        actions=actions,
                    )
                except ValueError:
                    reference_value = None
                else:
                    reference_value = {
                        "level": reference_plan.state.level,
                        "mode": ("locked" if reference_plan.state.locked else "open"),
                        "doubleSeat": (
                            reference_plan.state.double_seat.name.lower()
                            if reference_plan.state.double_seat is not None
                            else None
                        ),
                        "actions": [
                            [
                                item.actor.name.lower(),
                                item.action,
                                item.resolved_mode,
                            ]
                            for item in reference_plan.actions
                        ],
                    }

                cp_sat_plan = escalation_compiler.compile(
                    dealer=Seat.EAST,
                    declarer=declarer,
                    contract=contract,
                    match_score=match_score,
                    rules=rulebook,
                    required_actions=actions,
                    timeout_seconds=5.0,
                    maximum_conflicts=100_000,
                )
                cp_sat_value = (
                    None
                    if cp_sat_plan is None
                    else {
                        "level": cp_sat_plan.level,
                        "mode": "locked" if cp_sat_plan.locked else "open",
                        "doubleSeat": (
                            cp_sat_plan.double_seat.name.lower()
                            if cp_sat_plan.double_seat is not None
                            else None
                        ),
                        "actions": [
                            [
                                item.actor.name.lower(),
                                item.action,
                                item.resolved_mode,
                            ]
                            for item in cp_sat_plan.actions
                        ],
                    }
                )
                if not runtime_value == reference_value == cp_sat_value:
                    raise ValueError("escalation compiler parity failure")
                return runtime_value

            def prove_escalation_counterfactual(
                *,
                primitive_id: str,
                mutate: Any,
                contract: Contract,
                match_score: tuple[int, int],
                actions: tuple[tuple[Seat, str, str | None], ...],
                proof_boundary: str,
                joint_dependencies: tuple[str, ...] = (),
            ) -> None:
                counterfactual_payload = copy.deepcopy(baseline_payload)
                mutate(counterfactual_payload)
                counterfactual = RuleBook(counterfactual_payload)
                baseline_value = escalation_observation(
                    baseline,
                    contract=contract,
                    match_score=match_score,
                    actions=actions,
                )
                counterfactual_value = escalation_observation(
                    counterfactual,
                    contract=contract,
                    match_score=match_score,
                    actions=actions,
                )
                if baseline_value == counterfactual_value:
                    return
                core = {
                    "schemaVersion": 1,
                    "primitiveId": primitive_id,
                    "status": "proven_behavioral",
                    "proofKind": "escalation_reference_runtime_cp_sat_counterfactual",
                    "rulesHash": baseline.sha256,
                    "counterfactualRulesHash": counterfactual.sha256,
                    "observationHash": canonical_sha256(
                        [baseline_value, counterfactual_value]
                    ),
                    "targetsObserved": ["reference", "runtime", "cpSat"],
                    "referenceIndependent": False,
                    "referenceDomainIrHashes": [
                        ReferenceEscalationInterpreterV1.from_rules(
                            baseline.data
                        ).domain.sha256,
                        ReferenceEscalationInterpreterV1.from_rules(
                            counterfactual.data
                        ).domain.sha256,
                    ],
                    "proofBoundary": proof_boundary,
                    "jointDependencies": list(joint_dependencies),
                }
                proofs[primitive_id] = {
                    **core,
                    "proofHash": canonical_sha256(core),
                }

            one_double = ((Seat.WEST, "double", None),)
            prove_escalation_counterfactual(
                primitive_id="rules.escalation.enabled",
                mutate=lambda payload: payload["escalation"].__setitem__(
                    "enabled", False
                ),
                contract=Contract.HOKUM,
                match_score=(0, 0),
                actions=one_double,
                proof_boundary="single_legal_hokum_double_enabled_vs_disabled",
            )
            prove_escalation_counterfactual(
                primitive_id="rules.escalation.hokum_chain",
                mutate=lambda payload: payload["escalation"].__setitem__(
                    "hokum_chain", ["double", "triple"]
                ),
                contract=Contract.HOKUM,
                match_score=(0, 0),
                actions=(
                    (Seat.WEST, "double", None),
                    (Seat.SOUTH, "triple", None),
                    (Seat.WEST, "four", None),
                ),
                proof_boundary="hokum_four_present_vs_truncated_chain",
            )
            prove_escalation_counterfactual(
                primitive_id="rules.escalation.sun_chain",
                mutate=lambda payload: payload["escalation"].__setitem__(
                    "sun_chain", []
                ),
                contract=Contract.SUN,
                match_score=(101, 50),
                actions=one_double,
                proof_boundary="sun_double_present_vs_empty_chain",
            )

            def mutate_default_mode(payload: dict[str, Any]) -> None:
                payload["escalation"]["default_mode"] = "locked"

            prove_escalation_counterfactual(
                primitive_id="rules.escalation.default_mode",
                mutate=mutate_default_mode,
                contract=Contract.HOKUM,
                match_score=(0, 0),
                actions=one_double,
                proof_boundary="unspecified_hokum_double_resolves_open_vs_locked",
            )
            prove_escalation_counterfactual(
                primitive_id="rules.escalation.response_actor_policy",
                mutate=lambda payload: payload["escalation"].__setitem__(
                    "response_actor_policy", "team"
                ),
                contract=Contract.HOKUM,
                match_score=(0, 0),
                actions=(
                    (Seat.WEST, "double", None),
                    (Seat.NORTH, "triple", None),
                ),
                proof_boundary="declarer_only_vs_partner_team_triple",
            )
            prove_escalation_counterfactual(
                primitive_id="rules.escalation.sun_score_threshold",
                mutate=lambda payload: payload["escalation"].__setitem__(
                    "sun_score_threshold", 110
                ),
                contract=Contract.SUN,
                match_score=(101, 50),
                actions=one_double,
                proof_boundary="same_score_crosses_100_but_not_110",
            )
            prove_escalation_counterfactual(
                primitive_id="rules.escalation.sun_double_requires_split_score",
                mutate=lambda payload: payload["escalation"].__setitem__(
                    "sun_double_requires_split_score", False
                ),
                contract=Contract.SUN,
                match_score=(80, 60),
                actions=one_double,
                proof_boundary="non_split_score_rejected_vs_allowed",
            )
            if "sun_lower_score_inclusive" in baseline_payload["escalation"]:
                prove_escalation_counterfactual(
                    primitive_id="rules.escalation.sun_lower_score_inclusive",
                    mutate=lambda payload: payload["escalation"].__setitem__(
                        "sun_lower_score_inclusive",
                        not bool(
                            baseline_payload["escalation"][
                                "sun_lower_score_inclusive"
                            ]
                        ),
                    ),
                    contract=Contract.SUN,
                    match_score=(101, 100),
                    actions=one_double,
                    proof_boundary="lower_score_equal_threshold_allowed_vs_rejected",
                    joint_dependencies=(
                        "rules.escalation.sun_score_threshold",
                        "rules.escalation.sun_double_requires_split_score",
                    ),
                )

        play_binding = cls._cp_sat_receipt(
            owner="cp_sat.complete_play_scoring_v1", plugins=plugins
        )
        if play_binding.get("verified") is True:
            play_compiler = OrToolsCpSatPlayCompilerV1()

            def recalculate_raw_totals(payload: dict[str, Any]) -> None:
                cards = payload["card_rules"]
                bonus = int(payload["scoring"]["last_trick_bonus_raw"])
                payload["scoring"]["raw_card_totals"]["sun"] = (
                    4 * sum(int(value) for value in cards["sun_points"]) + bonus
                )
                payload["scoring"]["raw_card_totals"]["hokum"] = (
                    sum(int(value) for value in cards["hokum_trump_points"])
                    + 3 * sum(int(value) for value in cards["hokum_non_trump_points"])
                    + bonus
                )

            table_contexts = {
                "sun_strength": (Contract.SUN, None, "strength"),
                "hokum_non_trump_strength": (
                    Contract.HOKUM,
                    Suit.HEARTS,
                    "strength",
                ),
                "hokum_trump_strength": (
                    Contract.HOKUM,
                    Suit.SPADES,
                    "strength",
                ),
                "sun_points": (Contract.SUN, None, "points"),
                "hokum_non_trump_points": (
                    Contract.HOKUM,
                    Suit.HEARTS,
                    "points",
                ),
                "hokum_trump_points": (
                    Contract.HOKUM,
                    Suit.SPADES,
                    "points",
                ),
            }
            for key, (contract, trump, method_name) in table_contexts.items():
                mutated_payload = copy.deepcopy(baseline_payload)
                mutated_payload["card_rules"][key] = list(
                    reversed(mutated_payload["card_rules"][key])
                )
                if key.endswith("_points"):
                    recalculate_raw_totals(mutated_payload)
                mutated = RuleBook(mutated_payload)
                baseline_reference = ReferencePlayInterpreterV1.from_rules(
                    baseline.data
                )
                mutated_reference = ReferencePlayInterpreterV1.from_rules(mutated.data)
                cards = tuple(Card.of(Suit.SPADES, rank) for rank in range(8))
                baseline_runtime = [
                    int(getattr(baseline, method_name)(card, contract, trump))
                    for card in cards
                ]
                mutated_runtime = [
                    int(getattr(mutated, method_name)(card, contract, trump))
                    for card in cards
                ]
                baseline_reference_values = [
                    int(getattr(baseline_reference, method_name)(card, contract, trump))
                    for card in cards
                ]
                mutated_reference_values = [
                    int(getattr(mutated_reference, method_name)(card, contract, trump))
                    for card in cards
                ]
                baseline_cp_sat = play_compiler.configuration_probe(baseline)
                mutated_cp_sat = play_compiler.configuration_probe(mutated)
                baseline_cp_sat_values = baseline_cp_sat["tables"][key]
                mutated_cp_sat_values = mutated_cp_sat["tables"][key]
                if (
                    baseline_runtime
                    == baseline_reference_values
                    == baseline_cp_sat_values
                    and mutated_runtime
                    == mutated_reference_values
                    == mutated_cp_sat_values
                    and baseline_runtime != mutated_runtime
                ):
                    primitive_id = f"rules.card_rules.{key}"
                    core = {
                        "schemaVersion": 1,
                        "primitiveId": primitive_id,
                        "status": "proven_behavioral",
                        "proofKind": "play_table_reference_runtime_cp_sat_counterfactual",
                        "rulesHash": baseline.sha256,
                        "counterfactualRulesHash": mutated.sha256,
                        "runtimeObservationHash": canonical_sha256(
                            [baseline_runtime, mutated_runtime]
                        ),
                        "referenceObservationHash": canonical_sha256(
                            [baseline_reference_values, mutated_reference_values]
                        ),
                        "cpSatObservationHash": canonical_sha256(
                            [baseline_cp_sat_values, mutated_cp_sat_values]
                        ),
                        "targetsObserved": ["reference", "runtime", "cpSat"],
                        "referenceIndependent": True,
                        "proofBoundary": "all_eight_rank_outputs_in_selected_contract_context",
                    }
                    proofs[primitive_id] = {
                        **core,
                        "proofHash": canonical_sha256(core),
                    }

            discard_payload = copy.deepcopy(baseline_payload)
            discard_payload["play"]["allow_discard_when_cannot_overtrump"] = not bool(
                baseline_payload["play"]["allow_discard_when_cannot_overtrump"]
            )
            discard_rules = RuleBook(discard_payload)
            plays = (
                ReferencePlayV1(Seat.NORTH, Card.parse("S7")),
                ReferencePlayV1(Seat.WEST, Card.parse("H9")),
            )
            hand = (Card.parse("H7"), Card.parse("C7"))

            def runtime_discard_observation(rulebook: RuleBook) -> list[str]:
                state = RoundState(rulebook.sha256, 1, 1, Seat.EAST)
                state.contract = Contract.HOKUM
                state.buyer = Seat.SOUTH
                state.declarer = Seat.SOUTH
                state.trump_suit = Suit.HEARTS
                state.first_leader = Seat.NORTH
                state.phase = GamePhase.PLAYING
                state.hands = [0, 0, cards_to_mask(hand), 0]
                state.initial_hands = list(state.hands)
                trick = Trick(0, Seat.NORTH)
                for ordinal, play in enumerate(plays):
                    trick.plays.append(
                        PlayedCard(play.seat, play.card, 0, ordinal, (), play.ekka)
                    )
                    state.initial_hands[int(play.seat)] |= 1 << play.card.id
                state.current_trick = trick
                state.current_player = Seat.SOUTH
                return [card.code for card in BalootEngine(rulebook).legal_moves(state)]

            def reference_discard_observation(rulebook: RuleBook) -> list[str]:
                reference = ReferencePlayInterpreterV1.from_rules(rulebook.data)
                return [
                    card.code
                    for card in reference.legal_moves(
                        hand,
                        plays,
                        actor=Seat.SOUTH,
                        contract=Contract.HOKUM,
                        trump=Suit.HEARTS,
                    )
                ]

            baseline_runtime = runtime_discard_observation(baseline)
            mutated_runtime = runtime_discard_observation(discard_rules)
            baseline_reference = reference_discard_observation(baseline)
            mutated_reference = reference_discard_observation(discard_rules)
            baseline_cp_sat = play_compiler.configuration_probe(baseline)[
                "unableToOvertrumpActions"
            ]
            mutated_cp_sat = play_compiler.configuration_probe(discard_rules)[
                "unableToOvertrumpActions"
            ]
            if (
                baseline_runtime == baseline_reference
                and mutated_runtime == mutated_reference
                and baseline_runtime != mutated_runtime
                and baseline_cp_sat != mutated_cp_sat
            ):
                primitive_id = "rules.play.allow_discard_when_cannot_overtrump"
                core = {
                    "schemaVersion": 1,
                    "primitiveId": primitive_id,
                    "status": "proven_behavioral",
                    "proofKind": "unable_to_overtrump_reference_runtime_cp_sat_counterfactual",
                    "rulesHash": baseline.sha256,
                    "counterfactualRulesHash": discard_rules.sha256,
                    "runtimeObservationHash": canonical_sha256(
                        [baseline_runtime, mutated_runtime]
                    ),
                    "referenceObservationHash": canonical_sha256(
                        [baseline_reference, mutated_reference]
                    ),
                    "cpSatObservationHash": canonical_sha256(
                        [baseline_cp_sat, mutated_cp_sat]
                    ),
                    "targetsObserved": ["reference", "runtime", "cpSat"],
                    "referenceIndependent": True,
                    "proofBoundary": "void_in_led_suit_with_lower_trump_and_discard_available",
                }
                proofs[primitive_id] = {
                    **core,
                    "proofHash": canonical_sha256(core),
                }

            def first_leader_observation(rulebook: RuleBook) -> dict[str, str]:
                dealer = Seat.EAST
                buyer = Seat.SOUTH
                reference_value = ReferencePlayInterpreterV1.from_rules(
                    rulebook.data
                ).resolve_first_leader(dealer=dealer, buyer=buyer)
                runtime_value = BalootEngine(rulebook).resolve_first_leader(
                    dealer=dealer, buyer=buyer
                )
                cp_sat_value = play_compiler.resolve_first_leader(
                    rules=rulebook, dealer=dealer, buyer=buyer
                )
                if reference_value != runtime_value or runtime_value != cp_sat_value:
                    raise ValueError("first-leader compiler parity failure")
                return {
                    "reference": reference_value.name.lower(),
                    "runtime": runtime_value.name.lower(),
                    "cpSat": cp_sat_value.name.lower(),
                }

            leader_payload = copy.deepcopy(baseline_payload)
            leader_payload["play"]["first_leader"] = "buyer"
            leader_rules = RuleBook(leader_payload)
            baseline_leader = first_leader_observation(baseline)
            buyer_leader = first_leader_observation(leader_rules)
            if baseline_leader != buyer_leader:
                primitive_id = "rules.play.first_leader"
                core = {
                    "schemaVersion": 1,
                    "primitiveId": primitive_id,
                    "status": "proven_behavioral",
                    "proofKind": "first_leader_reference_runtime_cp_sat_counterfactual",
                    "rulesHash": baseline.sha256,
                    "counterfactualRulesHash": leader_rules.sha256,
                    "observationHash": canonical_sha256(
                        [baseline_leader, buyer_leader]
                    ),
                    "targetsObserved": ["reference", "runtime", "cpSat"],
                    "referenceIndependent": True,
                    "proofBoundary": "dealer_right_versus_buyer_policy",
                }
                proofs[primitive_id] = {
                    **core,
                    "proofHash": canonical_sha256(core),
                }

            configured_north_payload = copy.deepcopy(baseline_payload)
            configured_north_payload["play"]["first_leader"] = "configured"
            configured_north_payload["play"]["configured_first_leader"] = "north"
            configured_west_payload = copy.deepcopy(configured_north_payload)
            configured_west_payload["play"]["configured_first_leader"] = "west"
            configured_north = RuleBook(configured_north_payload)
            configured_west = RuleBook(configured_west_payload)
            north_observation = first_leader_observation(configured_north)
            west_observation = first_leader_observation(configured_west)
            if north_observation != west_observation:
                primitive_id = "rules.play.configured_first_leader"
                core = {
                    "schemaVersion": 1,
                    "primitiveId": primitive_id,
                    "status": "proven_behavioral",
                    "proofKind": "configured_first_leader_reference_runtime_cp_sat_counterfactual",
                    "rulesHash": baseline.sha256,
                    "activationRulesHash": configured_north.sha256,
                    "counterfactualRulesHash": configured_west.sha256,
                    "observationHash": canonical_sha256(
                        [north_observation, west_observation]
                    ),
                    "targetsObserved": ["reference", "runtime", "cpSat"],
                    "referenceIndependent": True,
                    "proofBoundary": "configured_policy_north_versus_west",
                }
                proofs[primitive_id] = {
                    **core,
                    "proofHash": canonical_sha256(core),
                }

            def distribute(total: int, count: int) -> list[int]:
                quotient, remainder = divmod(int(total), int(count))
                return [quotient + int(index < remainder) for index in range(count)]

            def ordinary_scoring_fixture(
                rulebook: RuleBook, contract: Contract
            ) -> dict[str, Any]:
                effective = Contract.parse(contract).effective.value
                available = int(
                    rulebook.data["scoring"]["raw_card_totals"][effective]
                ) - int(rulebook.data["scoring"]["last_trick_bonus_raw"])
                west_east_raw = max(4, available // 3)
                north_south_raw = available - west_east_raw
                return {
                    "contract": contract,
                    "trick_raw_points": [
                        *distribute(north_south_raw, 4),
                        *distribute(west_east_raw, 4),
                    ],
                    "trick_winner_teams": [
                        *([Team.NORTH_SOUTH] * 4),
                        *([Team.WEST_EAST] * 4),
                    ],
                }

            def kabout_scoring_fixture(
                rulebook: RuleBook, contract: Contract
            ) -> dict[str, Any]:
                effective = Contract.parse(contract).effective.value
                available = int(
                    rulebook.data["scoring"]["raw_card_totals"][effective]
                ) - int(rulebook.data["scoring"]["last_trick_bonus_raw"])
                return {
                    "contract": contract,
                    "trick_raw_points": distribute(available, 8),
                    "trick_winner_teams": [Team.NORTH_SOUTH] * 8,
                }

            def scoring_observation(
                rulebook: RuleBook,
                *,
                contract: Contract,
                trick_raw_points: Iterable[int],
                trick_winner_teams: Iterable[Team],
                escalation: str = "normal",
                project_points: tuple[int, int] = (0, 0),
                baloot_points: tuple[int, int] = (0, 0),
                starting_match_score: tuple[int, int] = (0, 0),
            ) -> dict[str, Any]:
                raw_rows = tuple(int(value) for value in trick_raw_points)
                winner_teams = tuple(Team(int(value)) for value in trick_winner_teams)
                winner_seats = tuple(
                    Seat.NORTH if team == Team.NORTH_SOUTH else Seat.WEST
                    for team in winner_teams
                )
                state = RoundState(
                    rulebook.sha256,
                    1,
                    1,
                    Seat.EAST,
                    match_score=[int(value) for value in starting_match_score],
                )
                state.contract = Contract.parse(contract)
                state.buyer = Seat.NORTH
                state.declarer = Seat.NORTH
                state.escalation = str(escalation)
                state.completed_tricks = [
                    Trick(
                        index=index,
                        leader=winner,
                        winner=winner,
                        raw_points=raw_rows[index],
                    )
                    for index, winner in enumerate(winner_seats)
                ]
                state.projects = [
                    Project(
                        owner=Seat.NORTH if team_index == 0 else Seat.WEST,
                        kind="scoring_proof",
                        cards=(),
                        value=int(value),
                        high_rank=0,
                        category_priority=0,
                        scoring=True,
                    )
                    for team_index, value in enumerate(project_points)
                    if int(value) > 0
                ]
                baloot_value = int(
                    rulebook.data["projects"]["baloot"]["value_game_points"]
                )
                state.baloot_teams = {
                    Team(team_index)
                    for team_index, value in enumerate(baloot_points)
                    if int(value) == baloot_value and int(value) > 0
                }
                if any(int(value) not in {0, baloot_value} for value in baloot_points):
                    raise ValueError(
                        "scoring proof Baloot points must match the governed value"
                    )

                scoring_domain = ScoringDomainIRV1.from_rules(rulebook.data)
                project_domain = ProjectDomainIRV1.from_rules(rulebook.data)
                runtime_score = score_round(state, scoring_domain, project_domain)
                runtime_final = list(starting_match_score)
                if runtime_score.gahwa:
                    runtime_winner_index = int(runtime_score.winner)
                    runtime_final[runtime_winner_index] = max(
                        scoring_domain.match_target,
                        runtime_final[1 - runtime_winner_index] + 1,
                    )
                else:
                    runtime_final[0] += runtime_score.awarded_points[0]
                    runtime_final[1] += runtime_score.awarded_points[1]
                state.match_score = runtime_final
                runtime_match_winner = (
                    runtime_score.match_winner
                    if runtime_score.match_winner is not None
                    else BalootEngine(rulebook)._determine_match_winner(state)
                )

                reference = ReferenceScoringInterpreterV1(scoring_domain)
                reference_score = reference.score(
                    contract=contract,
                    declarer=Seat.NORTH,
                    escalation=escalation,
                    trick_raw_points=raw_rows,
                    trick_winners=winner_seats,
                    project_points=project_points,
                    baloot_points=baloot_points,
                )
                reference_match = reference.apply_match_award(
                    starting_score=starting_match_score,
                    awarded_points=reference_score.awarded_points,
                    round_winner=reference_score.winner,
                    gahwa=reference_score.gahwa,
                )
                cp_sat = play_compiler.scoring_probe(
                    domain=scoring_domain,
                    contract=contract,
                    declarer=Seat.NORTH,
                    escalation=escalation,
                    trick_raw_points=raw_rows,
                    trick_winner_teams=winner_teams,
                    project_points=project_points,
                    baloot_points=baloot_points,
                    starting_match_score=starting_match_score,
                )
                fields = (
                    "rawCardPoints",
                    "cardGamePoints",
                    "projectPoints",
                    "balootPoints",
                    "totalBeforeAward",
                    "winner",
                    "awardedPoints",
                    "countingTeam",
                    "multiplier",
                    "kabout",
                    "gahwa",
                    "matchWinner",
                    "finalMatchScore",
                )
                runtime_payload = {
                    **runtime_score.to_dict(),
                    "matchWinner": (
                        runtime_match_winner.name.lower()
                        if runtime_match_winner is not None
                        else None
                    ),
                    "finalMatchScore": runtime_final,
                }
                reference_payload = {
                    **reference_score.to_dict(),
                    "matchWinner": (
                        reference_match.winner.name.lower()
                        if reference_match.winner is not None
                        else None
                    ),
                    "finalMatchScore": list(reference_match.final_score),
                }
                normalized_runtime = {
                    key: copy.deepcopy(runtime_payload[key]) for key in fields
                }
                normalized_reference = {
                    key: copy.deepcopy(reference_payload[key]) for key in fields
                }
                normalized_cp_sat = {key: copy.deepcopy(cp_sat[key]) for key in fields}
                if not (
                    normalized_runtime == normalized_reference == normalized_cp_sat
                ):
                    raise ValueError("scoring compiler parity failure")
                return normalized_runtime

            def prove_scoring_counterfactual(
                *,
                primitive_id: str,
                mutate: Any,
                scenario: Any,
                proof_boundary: str,
                joint_dependencies: tuple[str, ...] = (),
            ) -> None:
                counterfactual_payload = copy.deepcopy(baseline_payload)
                mutate(counterfactual_payload)
                counterfactual = RuleBook(counterfactual_payload)
                baseline_observation = scoring_observation(
                    baseline, **scenario(baseline)
                )
                counterfactual_observation = scoring_observation(
                    counterfactual, **scenario(counterfactual)
                )
                if baseline_observation == counterfactual_observation:
                    return
                core = {
                    "schemaVersion": 1,
                    "primitiveId": primitive_id,
                    "status": "proven_behavioral",
                    "proofKind": "scoring_reference_runtime_cp_sat_counterfactual",
                    "rulesHash": baseline.sha256,
                    "counterfactualRulesHash": counterfactual.sha256,
                    "observationHash": canonical_sha256(
                        [baseline_observation, counterfactual_observation]
                    ),
                    "targetsObserved": ["reference", "runtime", "cpSat"],
                    "referenceIndependent": True,
                    "proofBoundary": proof_boundary,
                    "jointDependencies": list(joint_dependencies),
                }
                proofs[primitive_id] = {
                    **core,
                    "proofHash": canonical_sha256(core),
                }

            def mutate_last_trick_bonus(payload: dict[str, Any]) -> None:
                payload["scoring"]["last_trick_bonus_raw"] = (
                    int(payload["scoring"]["last_trick_bonus_raw"]) + 1
                )
                payload["scoring"]["raw_card_totals"]["sun"] += 1
                payload["scoring"]["raw_card_totals"]["hokum"] += 1

            if "buyer_tie_outcome" in baseline_payload["scoring"]["contract"]:
                def flip_buyer_tie(payload: dict[str, Any]) -> None:
                    contract = payload["scoring"]["contract"]
                    contract["buyer_tie_outcome"] = (
                        "buyer_loses" if contract["buyer_tie_outcome"] == "buyer_makes" else "buyer_makes"
                    )

                prove_scoring_counterfactual(
                    primitive_id="rules.scoring.contract.buyer_tie_outcome",
                    mutate=flip_buyer_tie,
                    scenario=lambda rulebook: {
                        "contract": Contract.SUN,
                        "trick_raw_points": [
                            *distribute(int(rulebook.data["scoring"]["raw_card_totals"]["sun"]) - 65, 4),
                            *distribute(65 - int(rulebook.data["scoring"]["last_trick_bonus_raw"]), 4),
                        ],
                        "trick_winner_teams": [*([Team.NORTH_SOUTH] * 4), *([Team.WEST_EAST] * 4)],
                        "project_points": (5, 5),
                    },
                    proof_boundary="sun_18_18_buyer_makes_versus_buyer_loses",
                )

            def fixed_sun_bonus_scenario(_rules: RuleBook) -> dict[str, Any]:
                return {
                    "contract": Contract.SUN,
                    "trick_raw_points": [30, 30, 25, 20, 4, 4, 4, 3],
                    "trick_winner_teams": [
                        *([Team.NORTH_SOUTH] * 4),
                        *([Team.WEST_EAST] * 4),
                    ],
                }

            prove_scoring_counterfactual(
                primitive_id="rules.scoring.last_trick_bonus_raw",
                mutate=mutate_last_trick_bonus,
                scenario=fixed_sun_bonus_scenario,
                proof_boundary="sun_counting_raw_25_to_26_rounding_boundary",
                joint_dependencies=(
                    "rules.scoring.raw_card_totals.sun",
                    "rules.scoring.raw_card_totals.hokum",
                ),
            )

            for contract_name, contract in (
                ("sun", Contract.SUN),
                ("hokum", Contract.HOKUM),
            ):

                def mutate_card_total(
                    payload: dict[str, Any], *, contract_name: str = contract_name
                ) -> None:
                    payload["scoring"]["card_game_point_totals"][contract_name] += 3

                prove_scoring_counterfactual(
                    primitive_id=(
                        "rules.scoring.card_game_point_totals." + contract_name
                    ),
                    mutate=mutate_card_total,
                    scenario=lambda rulebook,
                    contract=contract: ordinary_scoring_fixture(rulebook, contract),
                    proof_boundary=f"{contract_name}_card_game_total_plus_three",
                )

            prove_scoring_counterfactual(
                primitive_id="rules.scoring.escalation.project_multiplier_cap",
                mutate=lambda payload: payload["scoring"]["escalation"].__setitem__(
                    "project_multiplier_cap", 3
                ),
                scenario=lambda rulebook: {
                    **ordinary_scoring_fixture(rulebook, Contract.SUN),
                    "escalation": "four",
                    "project_points": (10, 0),
                },
                proof_boundary="four_level_escalation_with_ten_project_points",
            )
            prove_scoring_counterfactual(
                primitive_id="rules.scoring.escalation.baloot_multiplier_cap",
                mutate=lambda payload: payload["scoring"]["escalation"].__setitem__(
                    "baloot_multiplier_cap", 2
                ),
                scenario=lambda rulebook: {
                    **ordinary_scoring_fixture(rulebook, Contract.HOKUM),
                    "escalation": "four",
                    "baloot_points": (2, 0),
                },
                proof_boundary="four_level_escalation_with_baloot_points",
            )
            prove_scoring_counterfactual(
                primitive_id="rules.scoring.kabout.enabled",
                mutate=lambda payload: payload["scoring"]["kabout"].__setitem__(
                    "enabled", False
                ),
                scenario=lambda rulebook: kabout_scoring_fixture(
                    rulebook, Contract.SUN
                ),
                proof_boundary="all_eight_tricks_won_by_north_south",
            )
            for contract_name, contract in (
                ("sun", Contract.SUN),
                ("hokum", Contract.HOKUM),
            ):

                def mutate_kabout_base(
                    payload: dict[str, Any], *, contract_name: str = contract_name
                ) -> None:
                    payload["scoring"]["kabout"]["base_game_points"][contract_name] += 7

                prove_scoring_counterfactual(
                    primitive_id=(
                        "rules.scoring.kabout.base_game_points." + contract_name
                    ),
                    mutate=mutate_kabout_base,
                    scenario=lambda rulebook, contract=contract: kabout_scoring_fixture(
                        rulebook, contract
                    ),
                    proof_boundary=f"{contract_name}_kabout_base_plus_seven",
                )

            def kabout_multiplier_observation(
                rulebook: RuleBook,
            ) -> dict[str, Any]:
                scoring_domain = ScoringDomainIRV1.from_rules(rulebook.data)
                raw_rows = tuple(
                    distribute(
                        scoring_domain.raw_card_totals["sun"]
                        - scoring_domain.last_trick_bonus_raw,
                        8,
                    )
                )
                state = RoundState(rulebook.sha256, 1, 1, Seat.EAST)
                state.contract = Contract.SUN
                state.buyer = Seat.SOUTH
                state.declarer = Seat.SOUTH
                state.escalation = "double"
                state.escalation_history = [{"team": Team.WEST_EAST}]
                state.completed_tricks = [
                    Trick(
                        index=index,
                        leader=Seat.SOUTH,
                        winner=Seat.SOUTH,
                        raw_points=raw,
                    )
                    for index, raw in enumerate(raw_rows)
                ]
                state.projects = [
                    Project(
                        owner=Seat.SOUTH,
                        kind="scoring_proof",
                        cards=(),
                        value=100,
                        high_rank=0,
                        category_priority=0,
                        scoring=True,
                    )
                ]
                state.baloot_teams = {Team.NORTH_SOUTH}
                runtime = score_round(
                    state,
                    scoring_domain,
                    ProjectDomainIRV1.from_rules(rulebook.data),
                )
                reference = ReferenceScoringInterpreterV1(scoring_domain).score(
                    contract=Contract.SUN,
                    declarer=Seat.SOUTH,
                    escalation="double",
                    trick_raw_points=raw_rows,
                    trick_winners=(Seat.SOUTH,) * 8,
                    project_points=(100, 0),
                    baloot_points=(2, 0),
                )
                native = OrToolsScoringStageV1(
                    scoring_domain
                ).solve(
                    contract=Contract.SUN,
                    buyer=Seat.SOUTH,
                    escalation="double",
                    last_escalating_team=Team.WEST_EAST,
                    raw_points=runtime.raw_card_points,
                    project_points=(100, 0),
                    baloot_points=(2, 0),
                    trick_winner_teams=(Team.NORTH_SOUTH,) * 8,
                    starting_match_score=(0, 0),
                )
                if native.status != "satisfied" or native.score is None:
                    raise ValueError("native scoring mutation probe did not solve")
                awarded = (
                    tuple(runtime.awarded_points),
                    tuple(reference.awarded_points),
                    tuple(native.score["gamePoints"]),
                )
                if len(set(awarded)) != 1:
                    raise ValueError("native scoring mutation probe parity failure")
                return {
                    "runtimeAwarded": list(awarded[0]),
                    "referenceAwarded": list(awarded[1]),
                    "nativeAwarded": list(awarded[2]),
                    "nativeModelHash": native.model_hash,
                    "nativeEvidenceHash": native.evidence_hash,
                    "nativeVariableCount": native.variable_count,
                    "nativeConstraintCount": native.constraint_count,
                }

            if "multiplier_cap" in baseline_payload["scoring"]["kabout"]:
                kabout_multiplier_payload = copy.deepcopy(baseline_payload)
                current_kabout_cap = int(
                    baseline_payload["scoring"]["kabout"]["multiplier_cap"]
                )
                kabout_multiplier_payload["scoring"]["kabout"][
                    "multiplier_cap"
                ] = 2 if current_kabout_cap == 1 else 1
                kabout_multiplier_rules = RuleBook(kabout_multiplier_payload)
                baseline_kabout_multiplier = kabout_multiplier_observation(baseline)
                counterfactual_kabout_multiplier = kabout_multiplier_observation(
                    kabout_multiplier_rules
                )
                if baseline_kabout_multiplier == counterfactual_kabout_multiplier:
                    raise ValueError("Kabout multiplier cap mutation had no effect")
                primitive_id = "rules.scoring.kabout.multiplier_cap"
                core = {
                    "schemaVersion": 1,
                    "primitiveId": primitive_id,
                    "status": "proven_behavioral",
                    "proofKind": (
                        "scoring_reference_runtime_native_cp_sat_counterfactual"
                    ),
                    "rulesHash": baseline.sha256,
                    "counterfactualRulesHash": kabout_multiplier_rules.sha256,
                    "observationHash": canonical_sha256(
                        [
                            baseline_kabout_multiplier,
                            counterfactual_kabout_multiplier,
                        ]
                    ),
                    "targetsObserved": ["reference", "runtime", "cpSat"],
                    "referenceIndependent": True,
                    "proofBoundary": (
                        "complete_round_scoring_sun_double_kabout_with_project_and_baloot"
                    ),
                    "nativeLowering": "native_cp_sat_scoring_v2",
                }
                proofs[primitive_id] = {
                    **core,
                    "proofHash": canonical_sha256(core),
                }

            baseline_match_target = int(baseline_payload["scoring"]["match"]["target"])
            prove_scoring_counterfactual(
                primitive_id="rules.scoring.match.target",
                mutate=lambda payload: payload["scoring"]["match"].__setitem__(
                    "target", baseline_match_target + 100
                ),
                scenario=lambda rulebook: {
                    **kabout_scoring_fixture(rulebook, Contract.SUN),
                    "starting_match_score": (baseline_match_target - 1, 0),
                },
                proof_boundary="same_round_crosses_152_but_not_252",
            )

        project_binding = cls._cp_sat_receipt(
            owner="cp_sat.complete_projects_v1", plugins=plugins
        )
        if project_binding.get("verified") is True:

            def project_fingerprint(projects: Iterable[Any]) -> list[list[Any]]:
                return sorted(
                    [
                        [
                            project.owner.name.lower(),
                            str(project.kind),
                            [card.id for card in project.cards],
                            int(project.value),
                            int(project.high_rank),
                            int(project.category_priority),
                        ]
                        for project in projects
                    ],
                    key=canonical_sha256,
                )

            def complete_hands(primary: Iterable[str]) -> dict[Seat, tuple[Card, ...]]:
                north = tuple(Card.parse(value) for value in primary)
                if len(north) != 8 or len(set(north)) != 8:
                    raise ValueError(
                        "project proof hand must contain eight unique cards"
                    )
                remaining = [card for card in FULL_DECK if card not in north]
                return {
                    Seat.NORTH: north,
                    Seat.WEST: tuple(remaining[:8]),
                    Seat.SOUTH: tuple(remaining[8:16]),
                    Seat.EAST: tuple(remaining[16:]),
                }

            # These hand-only influence fixtures fix a synthetic second-trick leader.
            project_fixture_second_leader = Seat.SOUTH

            def complete_declaration_actions(
                domain: ProjectDomainIRV1,
                projects: Iterable[Any],
                *,
                dealer: Seat,
                trump_suit: Suit | None,
                second_trick_leader: Seat = project_fixture_second_leader,
            ) -> tuple[Any, list[Any], Team | None]:
                state = domain.declaration_state(tuple(projects), dealer=dealer)
                while not state.complete:
                    legal = domain.legal_declaration_actions(
                        state,
                        trump_suit=trump_suit,
                    )
                    chosen = next(
                        (action for action in legal if action.action == "declare"),
                        legal[0],
                    )
                    state, _announced = domain.apply_declaration_action(
                        state,
                        chosen,
                        trump_suit=trump_suit,
                    )
                scored, winner = domain.score_team(
                    state.projects,
                    trump_suit=trump_suit,
                    second_trick_leader=second_trick_leader,
                )
                return state, scored, winner

            priority_hands = {
                Seat.NORTH: tuple(
                    Card.parse(value)
                    for value in ("H7", "H8", "H9", "HA", "D9", "HQ", "D8", "C7")
                ),
                Seat.WEST: tuple(
                    Card.parse(value)
                    for value in ("S7", "S8", "S9", "CK", "SJ", "DK", "D10", "DQ")
                ),
                Seat.SOUTH: tuple(
                    Card.parse(value)
                    for value in ("S10", "HJ", "CQ", "H10", "CJ", "SQ", "SK", "C9")
                ),
                Seat.EAST: tuple(
                    Card.parse(value)
                    for value in ("DA", "SA", "DJ", "HK", "C8", "CA", "D7", "C10")
                ),
            }

            def project_declaration_mode_observation(
                rulebook: RuleBook,
            ) -> dict[str, Any]:
                domain = ProjectDomainIRV1.from_rules(rulebook.data)
                runtime_projects = detect_projects(
                    priority_hands,
                    Contract.SUN,
                    rulebook,
                    Seat.EAST,
                )
                runtime_state = domain.declaration_state(
                    runtime_projects,
                    dealer=Seat.EAST,
                )
                runtime_legal = domain.legal_declaration_actions(
                    runtime_state,
                    trump_suit=None,
                )
                runtime_action = runtime_legal[0]
                runtime_next, _runtime_announced = domain.apply_declaration_action(
                    runtime_state,
                    runtime_action,
                    trump_suit=None,
                )

                reference = ReferenceProjectInterpreterV1.from_rules(rulebook.data)
                reference_projects = [
                    project
                    for seat in Seat
                    for project in reference.select(
                        reference.candidates(
                            seat,
                            priority_hands[seat],
                            Contract.SUN,
                        )
                    )
                ]
                reference_state = reference.domain.declaration_state(
                    reference_projects,
                    dealer=Seat.EAST,
                )
                reference_legal = reference.domain.legal_declaration_actions(
                    reference_state,
                    trump_suit=None,
                )
                reference_next, _reference_announced = (
                    reference.domain.apply_declaration_action(
                        reference_state,
                        reference_legal[0],
                        trump_suit=None,
                    )
                )
                runtime_value = {
                    "mode": domain.declaration_mode,
                    "legalActions": [action.to_dict() for action in runtime_legal],
                    "selectedAction": runtime_action.to_dict(),
                    "declaredAfterAction": sum(
                        bool(project.declared) for project in runtime_next.projects
                    ),
                }
                reference_value = {
                    "mode": reference.domain.declaration_mode,
                    "legalActions": [action.to_dict() for action in reference_legal],
                    "selectedAction": reference_legal[0].to_dict(),
                    "declaredAfterAction": sum(
                        bool(project.declared) for project in reference_next.projects
                    ),
                }
                if runtime_value != reference_value:
                    raise ValueError("project declaration mode reference parity failure")

                native = OrToolsProjectLifecycleModelV1().solve(
                    hands=priority_hands,
                    contract=Contract.SUN,
                    dealer=Seat.EAST,
                    trump=None,
                    domain=domain,
                    second_trick_leader=project_fixture_second_leader,
                    required_declaration_actions=(runtime_action,),
                )
                if (
                    not native.declaration_actions
                    or native.declaration_actions[0] != runtime_action
                ):
                    raise ValueError("project declaration mode CP-SAT parity failure")
                return {
                    **runtime_value,
                    "referenceDomainIrHash": reference.domain.sha256,
                    "cpSatFirstAction": native.declaration_actions[0].to_dict(),
                    "cpSatModelHash": native.model_hash,
                    "cpSatEvidenceHash": native.evidence_hash,
                }

            if "declaration_mode" in baseline_payload["projects"]:
                declaration_mode_payload = copy.deepcopy(baseline_payload)
                current_mode = str(
                    declaration_mode_payload["projects"]["declaration_mode"]
                )
                declaration_mode_payload["projects"]["declaration_mode"] = (
                    "optional" if current_mode == "mandatory" else "mandatory"
                )
                declaration_mode_rules = RuleBook(declaration_mode_payload)
                baseline_declaration_mode = project_declaration_mode_observation(
                    baseline
                )
                counterfactual_declaration_mode = (
                    project_declaration_mode_observation(declaration_mode_rules)
                )
                if baseline_declaration_mode == counterfactual_declaration_mode:
                    raise ValueError("project declaration mode mutation had no effect")
                primitive_id = "rules.projects.declaration_mode"
                core = {
                    "schemaVersion": 1,
                    "primitiveId": primitive_id,
                    "status": "proven_behavioral",
                    "proofKind": (
                        "project_declaration_mode_reference_runtime_native_cp_sat_counterfactual"
                    ),
                    "rulesHash": baseline.sha256,
                    "counterfactualRulesHash": declaration_mode_rules.sha256,
                    "observationHash": canonical_sha256(
                        [
                            baseline_declaration_mode,
                            counterfactual_declaration_mode,
                        ]
                    ),
                    "targetsObserved": ["reference", "runtime", "cpSat"],
                    "referenceIndependent": False,
                    "proofBoundary": (
                        "first_project_actor_legal_actions_and_native_action_binding"
                    ),
                }
                proofs[primitive_id] = {
                    **core,
                    "proofHash": canonical_sha256(core),
                }

            def project_priority_observation(rulebook: RuleBook) -> dict[str, Any]:
                def declared_value(projects: Iterable[Any]) -> list[list[str]]:
                    return sorted(
                        [
                            [project.owner.name.lower(), project.kind]
                            for project in projects
                            if project.declared
                        ],
                        key=canonical_sha256,
                    )

                project_domain = ProjectDomainIRV1.from_rules(rulebook.data)
                runtime_projects = detect_projects(
                    priority_hands, Contract.HOKUM, rulebook, Seat.NORTH
                )
                runtime_state, runtime_projects, runtime_winner = (
                    complete_declaration_actions(
                        project_domain,
                        runtime_projects,
                        dealer=Seat.NORTH,
                        trump_suit=Suit.HEARTS,
                        second_trick_leader=Seat.WEST,
                    )
                )
                runtime_value = {
                    "winner": (
                        runtime_winner.name.lower()
                        if runtime_winner is not None
                        else None
                    ),
                    "declared": declared_value(runtime_projects),
                }

                reference = ReferenceProjectInterpreterV1.from_rules(rulebook.data)
                reference_projects = [
                    project
                    for seat in Seat
                    for project in reference.select(
                        reference.candidates(seat, priority_hands[seat], Contract.HOKUM)
                    )
                ]
                reference_state, reference_projects, reference_winner = (
                    complete_declaration_actions(
                        reference.domain,
                        reference_projects,
                        dealer=Seat.NORTH,
                        trump_suit=Suit.HEARTS,
                        second_trick_leader=Seat.WEST,
                    )
                )
                reference_value = {
                    "winner": (
                        reference_winner.name.lower()
                        if reference_winner is not None
                        else None
                    ),
                    "declared": declared_value(reference_projects),
                }
                cp_sat = OrToolsProjectLifecycleModelV1().solve(
                    hands=priority_hands,
                    contract=Contract.HOKUM,
                    dealer=Seat.NORTH,
                    trump=Suit.HEARTS,
                    domain=project_domain,
                    second_trick_leader=Seat.WEST,
                    timeout_seconds=5.0,
                    maximum_conflicts=100_000,
                    required_declaration_actions=runtime_state.actions,
                    required_declaration_actions_complete=True,
                )
                cp_sat_value = {
                    "winner": (
                        cp_sat.winning_team.name.lower()
                        if cp_sat.winning_team is not None
                        else None
                    ),
                    "declared": declared_value(cp_sat.projects),
                }
                if not runtime_value == reference_value == cp_sat_value:
                    raise ValueError("project priority compiler parity failure")
                if reference_state.actions != runtime_state.actions:
                    raise ValueError("project priority action parity failure")
                return runtime_value

            priority_payload = copy.deepcopy(baseline_payload)
            priority_payload["disputed_options"][
                "equal_hokum_sequence_trump_priority"
            ] = {"selected": True, "alternatives": [False]}
            priority_rules = RuleBook(priority_payload)
            baseline_priority = project_priority_observation(baseline)
            counterfactual_priority = project_priority_observation(priority_rules)
            if baseline_priority != counterfactual_priority:
                primitive_id = (
                    "rules.disputed_options."
                    "equal_hokum_sequence_trump_priority.selected"
                )
                core = {
                    "schemaVersion": 1,
                    "primitiveId": primitive_id,
                    "status": "proven_behavioral",
                    "proofKind": "project_priority_reference_runtime_cp_sat_counterfactual",
                    "rulesHash": baseline.sha256,
                    "counterfactualRulesHash": priority_rules.sha256,
                    "observationHash": canonical_sha256(
                        [baseline_priority, counterfactual_priority]
                    ),
                    "targetsObserved": ["reference", "runtime", "cpSat"],
                    "referenceIndependent": False,
                    "proofBoundary": "equal_hokum_sequences_non_trump_first_trump_last",
                    "jointDependencies": [
                        "rules.disputed_options."
                        "equal_hokum_sequence_trump_priority.alternatives"
                    ],
                }
                proofs[primitive_id] = {
                    **core,
                    "proofHash": canonical_sha256(core),
                }

            project_observation_cache: dict[tuple[str, str, str], dict[str, Any]] = {}

            def project_observation(
                rulebook: RuleBook,
                hands: Mapping[Seat, tuple[Card, ...]],
                contract: Contract,
            ) -> dict[str, Any]:
                hand_hash = canonical_sha256(
                    {
                        seat.name.lower(): [card.id for card in hands[seat]]
                        for seat in Seat
                    }
                )
                cache_key = (rulebook.sha256, hand_hash, contract.value)
                cached_observation = project_observation_cache.get(cache_key)
                if cached_observation is not None:
                    return copy.deepcopy(cached_observation)
                runtime_projects = detect_projects(
                    dict(hands), contract, rulebook, Seat.EAST
                )
                domain = ProjectDomainIRV1.from_rules(rulebook.data)
                reference = ReferenceProjectInterpreterV1.from_rules(rulebook.data)
                reference_projects = [
                    project
                    for seat in Seat
                    for project in reference.select(
                        reference.candidates(seat, hands[seat], contract)
                    )
                ]
                cp_sat = OrToolsProjectLifecycleModelV1().solve(
                    hands=hands,
                    contract=contract,
                    dealer=Seat.EAST,
                    trump=Suit.HEARTS if contract == Contract.HOKUM else None,
                    domain=domain,
                    second_trick_leader=project_fixture_second_leader,
                    timeout_seconds=5.0,
                    maximum_conflicts=100_000,
                )
                runtime_fingerprint = project_fingerprint(runtime_projects)
                reference_fingerprint = project_fingerprint(reference_projects)
                cp_sat_fingerprint = project_fingerprint(
                    project for project in cp_sat.projects if project.selected
                )
                if not (
                    runtime_fingerprint == reference_fingerprint == cp_sat_fingerprint
                ):
                    raise ValueError("project compiler parity failure")
                observation = {
                    "projects": runtime_fingerprint,
                    "referenceDomainIrHash": reference.domain.sha256,
                    "cpSatModelHash": cp_sat.model_hash,
                    "cpSatEvidenceHash": cp_sat.evidence_hash,
                }
                project_observation_cache[cache_key] = copy.deepcopy(observation)
                return observation

            def prove_project_counterfactual(
                *,
                primitive_id: str,
                hands: Mapping[Seat, tuple[Card, ...]],
                contract: Contract,
                mutate: Any,
                proof_boundary: str,
            ) -> None:
                counterfactual_payload = copy.deepcopy(baseline_payload)
                mutate(counterfactual_payload)
                counterfactual = RuleBook(counterfactual_payload)
                baseline_observation = project_observation(baseline, hands, contract)
                counterfactual_observation = project_observation(
                    counterfactual, hands, contract
                )
                if (
                    baseline_observation["projects"]
                    == counterfactual_observation["projects"]
                ):
                    return
                core = {
                    "schemaVersion": 1,
                    "primitiveId": primitive_id,
                    "status": "proven_behavioral",
                    "proofKind": "project_reference_runtime_cp_sat_counterfactual",
                    "rulesHash": baseline.sha256,
                    "counterfactualRulesHash": counterfactual.sha256,
                    "observationHash": canonical_sha256(
                        [baseline_observation, counterfactual_observation]
                    ),
                    "targetsObserved": ["reference", "runtime", "cpSat"],
                    "referenceIndependent": False,
                    "proofBoundary": proof_boundary,
                }
                proofs[primitive_id] = {
                    **core,
                    "proofHash": canonical_sha256(core),
                }

            category_priority_observation_cache: dict[
                tuple[str, str, str, str], dict[str, Any]
            ] = {}

            def project_category_priority_observation(
                rulebook: RuleBook,
                hands: Mapping[Seat, tuple[Card, ...]],
                contract: Contract,
                trump_suit: Suit | None,
            ) -> dict[str, Any]:
                hand_hash = canonical_sha256(
                    {
                        seat.name.lower(): [card.id for card in hands[seat]]
                        for seat in Seat
                    }
                )
                cache_key = (
                    rulebook.sha256,
                    hand_hash,
                    contract.value,
                    trump_suit.name.lower() if trump_suit is not None else "none",
                )
                cached = category_priority_observation_cache.get(cache_key)
                if cached is not None:
                    return copy.deepcopy(cached)

                domain = ProjectDomainIRV1.from_rules(rulebook.data)
                runtime_projects = detect_projects(
                    dict(hands), contract, rulebook, Seat.EAST
                )
                runtime_state, runtime_projects, runtime_winner = (
                    complete_declaration_actions(
                        domain,
                        runtime_projects,
                        dealer=Seat.EAST,
                        trump_suit=trump_suit,
                    )
                )
                runtime_comparison_winner = domain.comparison_winner(
                    runtime_projects,
                    trump_suit=trump_suit,
                    second_trick_leader=project_fixture_second_leader,
                )

                reference = ReferenceProjectInterpreterV1.from_rules(rulebook.data)
                reference_projects = [
                    project
                    for seat in Seat
                    for project in reference.select(
                        reference.candidates(seat, hands[seat], contract)
                    )
                ]
                reference_state, reference_projects, reference_winner = (
                    complete_declaration_actions(
                        reference.domain,
                        reference_projects,
                        dealer=Seat.EAST,
                        trump_suit=trump_suit,
                    )
                )
                reference_comparison_winner = reference.domain.comparison_winner(
                    reference_projects,
                    trump_suit=trump_suit,
                    second_trick_leader=project_fixture_second_leader,
                )

                native = OrToolsProjectLifecycleModelV1().solve(
                    hands=hands,
                    contract=contract,
                    dealer=Seat.EAST,
                    trump=trump_suit,
                    domain=domain,
                    second_trick_leader=project_fixture_second_leader,
                    timeout_seconds=5.0,
                    maximum_conflicts=100_000,
                    required_declaration_actions=runtime_state.actions,
                    required_declaration_actions_complete=True,
                )
                if reference_state.actions != runtime_state.actions:
                    raise ValueError("native project action parity failure")

                def resolved_fingerprint(
                    projects: Iterable[Any], comparison_winner: Any
                ) -> list[list[Any]]:
                    return sorted(
                        [
                            [
                                project.owner.name.lower(),
                                str(project.kind),
                                [card.id for card in project.cards],
                                int(project.value),
                                int(project.high_rank),
                                int(project.category_priority),
                                bool(project.declared),
                                project == comparison_winner,
                                bool(project.scoring),
                            ]
                            for project in projects
                        ],
                        key=canonical_sha256,
                    )

                runtime_fingerprint = resolved_fingerprint(
                    runtime_projects, runtime_comparison_winner
                )
                reference_fingerprint = resolved_fingerprint(
                    reference_projects, reference_comparison_winner
                )
                native_fingerprint = sorted(
                    [
                        [
                            project.owner.name.lower(),
                            project.kind,
                            [card.id for card in project.cards],
                            project.value,
                            project.high_rank,
                            project.category_priority,
                            project.declared,
                            project.comparison_winner,
                            project.awarded,
                        ]
                        for project in native.projects
                        if project.selected
                    ],
                    key=canonical_sha256,
                )
                if not (
                    runtime_fingerprint == reference_fingerprint == native_fingerprint
                    and runtime_winner == reference_winner == native.winning_team
                ):
                    raise ValueError("native project priority compiler parity failure")
                observation = {
                    "projects": runtime_fingerprint,
                    "winningTeam": (
                        runtime_winner.name.lower()
                        if runtime_winner is not None
                        else None
                    ),
                    "referenceDomainIrHash": reference.domain.sha256,
                    "cpSatModelHash": native.model_hash,
                    "cpSatEvidenceHash": native.evidence_hash,
                }
                category_priority_observation_cache[cache_key] = copy.deepcopy(
                    observation
                )
                return observation

            def prove_project_category_priority_counterfactual(
                *,
                category: str,
                counterfactual_value: int,
                hands: Mapping[Seat, tuple[Card, ...]],
                contract: Contract,
                trump_suit: Suit | None,
                proof_boundary: str,
            ) -> None:
                primitive_id = (
                    f"rules.projects.comparison.category_priorities.{category}"
                )
                counterfactual_payload = copy.deepcopy(baseline_payload)
                counterfactual_payload["projects"]["comparison"][
                    "category_priorities"
                ][category] = counterfactual_value
                counterfactual = RuleBook(counterfactual_payload)
                baseline_observation = project_category_priority_observation(
                    baseline, hands, contract, trump_suit
                )
                counterfactual_observation = project_category_priority_observation(
                    counterfactual, hands, contract, trump_suit
                )
                if baseline_observation == counterfactual_observation:
                    raise ValueError(
                        f"project category priority did not influence {primitive_id}"
                    )
                core = {
                    "schemaVersion": 1,
                    "primitiveId": primitive_id,
                    "status": "proven_behavioral",
                    "proofKind": (
                        "project_category_priority_reference_runtime_native_cp_sat_counterfactual"
                    ),
                    "rulesHash": baseline.sha256,
                    "counterfactualRulesHash": counterfactual.sha256,
                    "observationHash": canonical_sha256(
                        [baseline_observation, counterfactual_observation]
                    ),
                    "targetsObserved": ["reference", "runtime", "cpSat"],
                    "referenceIndependent": False,
                    "proofBoundary": proof_boundary,
                }
                proofs[primitive_id] = {
                    **core,
                    "proofHash": canonical_sha256(core),
                }

            sequence_three_hands = complete_hands(
                ("S7", "S8", "S9", "H7", "D9", "C10", "HQ", "DK")
            )
            sequence_four_hands = complete_hands(
                ("S7", "S8", "S9", "S10", "H7", "D9", "CQ", "HK")
            )
            sequence_five_hands = complete_hands(
                ("S7", "S8", "S9", "S10", "SJ", "H7", "DQ", "CK")
            )

            if "comparison" in baseline_payload["projects"]:
                priority_values = baseline_payload["projects"]["comparison"][
                    "category_priorities"
                ]
                for category, fixture_hands in (
                    ("siri", sequence_three_hands),
                    ("fifty", sequence_four_hands),
                ):
                    prove_project_category_priority_counterfactual(
                        category=category,
                        counterfactual_value=int(priority_values[category]) + 7,
                        hands=fixture_hands,
                        contract=Contract.SUN,
                        trump_suit=None,
                        proof_boundary=(
                            f"fixed_{category}_candidate_rank_and_native_solver_evidence"
                        ),
                    )

                hundred_priority_hands = {
                    Seat.NORTH: tuple(
                        Card.parse(value)
                        for value in (
                            "H7", "H8", "H9", "H10", "HJ", "CK", "HQ", "DJ"
                        )
                    ),
                    Seat.WEST: tuple(
                        Card.parse(value)
                        for value in (
                            "SA", "HA", "DA", "CA", "C10", "C7", "S10", "SJ"
                        )
                    ),
                    Seat.SOUTH: tuple(
                        Card.parse(value)
                        for value in (
                            "C9", "D10", "DK", "CJ", "D8", "DQ", "S7", "C8"
                        )
                    ),
                    Seat.EAST: tuple(
                        Card.parse(value)
                        for value in (
                            "SK", "HK", "D7", "SQ", "D9", "CQ", "S9", "S8"
                        )
                    ),
                }
                prove_project_category_priority_counterfactual(
                    category="hundred_sequence",
                    counterfactual_value=0,
                    hands=hundred_priority_hands,
                    contract=Contract.HOKUM,
                    trump_suit=Suit.DIAMONDS,
                    proof_boundary=(
                        "equal_hundred_sequence_priority_mutation_flips_native_winner"
                    ),
                )
                prove_project_category_priority_counterfactual(
                    category="hundred_four_kind",
                    counterfactual_value=3,
                    hands=hundred_priority_hands,
                    contract=Contract.HOKUM,
                    trump_suit=Suit.DIAMONDS,
                    proof_boundary=(
                        "equal_hundred_quartet_priority_mutation_flips_native_winner"
                    ),
                )

            prove_project_counterfactual(
                primitive_id="rules.projects.enabled",
                hands=sequence_three_hands,
                contract=Contract.SUN,
                mutate=lambda payload: payload["projects"].__setitem__(
                    "enabled", False
                ),
                proof_boundary="fixed_complete_deal_with_project_generation_disabled",
            )
            prove_project_counterfactual(
                primitive_id="rules.projects.sequences.enabled",
                hands=sequence_three_hands,
                contract=Contract.SUN,
                mutate=lambda payload: payload["projects"]["sequences"].__setitem__(
                    "enabled", False
                ),
                proof_boundary="fixed_complete_deal_with_exact_three_card_sequence",
            )
            prove_project_counterfactual(
                primitive_id="rules.projects.sequences.minimum_length",
                hands=sequence_three_hands,
                contract=Contract.SUN,
                mutate=lambda payload: payload["projects"]["sequences"].__setitem__(
                    "minimum_length", 4
                ),
                proof_boundary="three_card_sequence_rejected_by_four_card_minimum",
            )

            sequence_fixtures = {
                "three": sequence_three_hands,
                "four": sequence_four_hands,
                "five_plus": sequence_five_hands,
            }
            for contract_name, contract in (
                ("sun", Contract.SUN),
                ("hokum", Contract.HOKUM),
            ):
                for length_name, fixture_hands in sequence_fixtures.items():
                    primitive_id = (
                        "rules.projects.sequences.values_game_points."
                        f"{contract_name}.{length_name}"
                    )

                    def mutate_sequence_value(
                        payload: dict[str, Any],
                        *,
                        contract_name: str = contract_name,
                        length_name: str = length_name,
                    ) -> None:
                        values = payload["projects"]["sequences"]["values_game_points"]
                        values[contract_name][length_name] = (
                            int(values[contract_name][length_name]) + 17
                        )

                    prove_project_counterfactual(
                        primitive_id=primitive_id,
                        hands=fixture_hands,
                        contract=contract,
                        mutate=mutate_sequence_value,
                        proof_boundary=f"fixed_{contract_name}_{length_name}_sequence_value",
                    )

            def four_kind_hands(rank_code: str) -> dict[Seat, tuple[Card, ...]]:
                group = tuple(
                    Card.parse(f"{suit}{rank_code}") for suit in ("S", "H", "D", "C")
                )
                fillers = tuple(card for card in FULL_DECK if card not in group)[:4]
                return complete_hands(tuple(card.code for card in (*group, *fillers)))

            if "comparison" in baseline_payload["projects"]:
                four_hundred_priority = baseline_payload["projects"]["comparison"][
                    "category_priorities"
                ]["four_hundred"]
                prove_project_category_priority_counterfactual(
                    category="four_hundred",
                    counterfactual_value=int(four_hundred_priority) + 7,
                    hands=four_kind_hands("A"),
                    contract=Contract.SUN,
                    trump_suit=None,
                    proof_boundary=(
                        "fixed_four_hundred_candidate_rank_and_native_solver_evidence"
                    ),
                )

            ace_four_kind_hands = four_kind_hands("A")
            prove_project_counterfactual(
                primitive_id="rules.projects.four_of_kind.enabled",
                hands=ace_four_kind_hands,
                contract=Contract.SUN,
                mutate=lambda payload: payload["projects"]["four_of_kind"].__setitem__(
                    "enabled", False
                ),
                proof_boundary="fixed_complete_deal_with_four_aces",
            )
            prove_project_counterfactual(
                primitive_id="rules.projects.four_of_kind.eligible_ranks",
                hands=ace_four_kind_hands,
                contract=Contract.SUN,
                mutate=lambda payload: payload["projects"]["four_of_kind"].__setitem__(
                    "eligible_ranks",
                    [
                        rank
                        for rank in payload["projects"]["four_of_kind"][
                            "eligible_ranks"
                        ]
                        if str(rank) != "ace"
                    ],
                ),
                proof_boundary="four_aces_removed_by_eligible_rank_counterfactual",
            )

            rank_codes = {
                "ten": "10",
                "jack": "J",
                "queen": "Q",
                "king": "K",
                "ace": "A",
            }
            for contract_name, contract in (
                ("sun", Contract.SUN),
                ("hokum", Contract.HOKUM),
            ):
                for rank_name, rank_code in rank_codes.items():
                    primitive_id = (
                        "rules.projects.four_of_kind.values_game_points."
                        f"{contract_name}.{rank_name}"
                    )

                    def mutate_four_kind_value(
                        payload: dict[str, Any],
                        *,
                        contract_name: str = contract_name,
                        rank_name: str = rank_name,
                    ) -> None:
                        values = payload["projects"]["four_of_kind"][
                            "values_game_points"
                        ]
                        values[contract_name][rank_name] = (
                            int(values[contract_name][rank_name]) + 17
                        )

                    prove_project_counterfactual(
                        primitive_id=primitive_id,
                        hands=four_kind_hands(rank_code),
                        contract=contract,
                        mutate=mutate_four_kind_value,
                        proof_boundary=f"fixed_{contract_name}_four_{rank_name}_value",
                    )

            maximum_hands = complete_hands(
                ("S7", "S8", "S9", "SA", "HA", "DA", "CA", "C10")
            )
            prove_project_counterfactual(
                primitive_id="rules.projects.max_scoring_projects_per_player",
                hands=maximum_hands,
                contract=Contract.SUN,
                mutate=lambda payload: payload["projects"].__setitem__(
                    "max_scoring_projects_per_player", 1
                ),
                proof_boundary="one_seat_with_non_overlapping_sequence_and_four_kind",
            )
            prove_project_counterfactual(
                primitive_id="rules.projects.cards_may_overlap",
                hands=sequence_five_hands,
                contract=Contract.SUN,
                mutate=lambda payload: payload["projects"].__setitem__(
                    "cards_may_overlap", True
                ),
                proof_boundary="one_seat_with_overlapping_three_four_and_five_card_sequences",
            )

            baloot_hands = complete_hands(
                ("HK", "HQ", "S7", "D9", "C10", "SQ", "DK", "CA")
            )

            def baloot_observation(
                rulebook: RuleBook,
                hands: dict[Seat, tuple[Card, ...]] = baloot_hands,
            ) -> dict[str, Any]:
                domain = ProjectDomainIRV1.from_rules(rulebook.data)
                runtime_projects = detect_projects(
                    hands, Contract.HOKUM, rulebook, Seat.EAST
                )
                runtime_state, runtime_projects, _runtime_winner = (
                    complete_declaration_actions(
                        domain,
                        runtime_projects,
                        dealer=Seat.EAST,
                        trump_suit=Suit.HEARTS,
                    )
                )
                runtime_teams = complete_baloot_teams(
                    hands,
                    runtime_projects,
                    Contract.HOKUM,
                    Suit.HEARTS,
                    rulebook,
                )
                reference = ReferenceProjectInterpreterV1.from_rules(rulebook.data)
                reference_projects = [
                    project
                    for seat in Seat
                    for project in reference.select(
                        reference.candidates(seat, hands[seat], Contract.HOKUM)
                    )
                ]
                reference_state, reference_projects, _reference_winner = (
                    complete_declaration_actions(
                        reference.domain,
                        reference_projects,
                        dealer=Seat.EAST,
                        trump_suit=Suit.HEARTS,
                    )
                )
                reference_teams = reference.complete_baloot_teams(
                    hands,
                    reference_projects,
                    Contract.HOKUM,
                    Suit.HEARTS,
                )
                plan = OrToolsProjectLifecycleModelV1().solve(
                    hands=hands,
                    contract=Contract.HOKUM,
                    dealer=Seat.EAST,
                    trump=Suit.HEARTS,
                    domain=domain,
                    second_trick_leader=project_fixture_second_leader,
                    timeout_seconds=5.0,
                    maximum_conflicts=100_000,
                    required_declaration_actions=runtime_state.actions,
                    required_declaration_actions_complete=True,
                )
                if (
                    runtime_teams != reference_teams
                    or reference_teams != plan.baloot_teams
                    or reference_state.actions != runtime_state.actions
                ):
                    raise ValueError("Baloot project compiler parity failure")
                value = int(rulebook.data["projects"]["baloot"]["value_game_points"])
                runtime_points = [0, 0]
                for team in runtime_teams:
                    runtime_points[int(team)] += value
                if runtime_points != list(plan.baloot_points):
                    raise ValueError("Baloot value compiler parity failure")
                return {
                    "teams": [
                        team.name.lower() for team in sorted(runtime_teams, key=int)
                    ],
                    "points": runtime_points,
                    "referenceDomainIrHash": reference.domain.sha256,
                    "cpSatModelHash": plan.model_hash,
                    "cpSatEvidenceHash": plan.evidence_hash,
                }

            for primitive_id, mutate, boundary in (
                (
                    "rules.projects.baloot.enabled",
                    lambda payload: payload["projects"]["baloot"].__setitem__(
                        "enabled", False
                    ),
                    "fixed_hokum_hand_with_trump_king_and_queen",
                ),
                (
                    "rules.projects.baloot.value_game_points",
                    lambda payload: payload["projects"]["baloot"].__setitem__(
                        "value_game_points",
                        int(payload["projects"]["baloot"]["value_game_points"]) + 1,
                    ),
                    "fixed_hokum_baloot_award_value",
                ),
            ):
                counterfactual_payload = copy.deepcopy(baseline_payload)
                mutate(counterfactual_payload)
                counterfactual = RuleBook(counterfactual_payload)
                baseline_observation = baloot_observation(baseline)
                counterfactual_observation = baloot_observation(counterfactual)
                if baseline_observation != counterfactual_observation:
                    core = {
                        "schemaVersion": 1,
                        "primitiveId": primitive_id,
                        "status": "proven_behavioral",
                        "proofKind": "baloot_reference_runtime_cp_sat_counterfactual",
                        "rulesHash": baseline.sha256,
                        "counterfactualRulesHash": counterfactual.sha256,
                        "observationHash": canonical_sha256(
                            [baseline_observation, counterfactual_observation]
                        ),
                        "targetsObserved": ["reference", "runtime", "cpSat"],
                        "referenceIndependent": True,
                        "proofBoundary": boundary,
                    }
                    proofs[primitive_id] = {
                        **core,
                        "proofHash": canonical_sha256(core),
                    }

            hundred_baloot_hands = complete_hands(
                ("H9", "H10", "HJ", "HQ", "HK", "S7", "D9", "C10")
            )
            hundred_payload = copy.deepcopy(baseline_payload)
            inclusion = str(hundred_payload["projects"]["baloot"]["hundred_inclusion"])
            hundred_payload["projects"]["baloot"]["hundred_inclusion"] = (
                "automatic" if inclusion == "blocked" else "blocked"
            )
            hundred_counterfactual = RuleBook(hundred_payload)
            baseline_hundred = baloot_observation(baseline, hundred_baloot_hands)
            counterfactual_hundred = baloot_observation(
                hundred_counterfactual, hundred_baloot_hands
            )
            if baseline_hundred != counterfactual_hundred:
                primitive_id = "rules.projects.baloot.hundred_inclusion"
                core = {
                    "schemaVersion": 1,
                    "primitiveId": primitive_id,
                    "status": "proven_behavioral",
                    "proofKind": "baloot_hundred_reference_runtime_cp_sat_counterfactual",
                    "rulesHash": baseline.sha256,
                    "counterfactualRulesHash": hundred_counterfactual.sha256,
                    "observationHash": canonical_sha256(
                        [baseline_hundred, counterfactual_hundred]
                    ),
                    "targetsObserved": ["reference", "runtime", "cpSat"],
                    "referenceIndependent": True,
                    "proofBoundary": "fixed_hokum_hundred_with_trump_king_and_queen",
                }
                proofs[primitive_id] = {
                    **core,
                    "proofHash": canonical_sha256(core),
                }

            sequence_payload = copy.deepcopy(baseline_payload)
            sequence_payload["card_rules"]["sequence_strength"] = [
                0,
                2,
                4,
                1,
                3,
                5,
                6,
                7,
            ]
            sequence_rules = RuleBook(sequence_payload)

            baseline_projects = project_observation(
                baseline, sequence_three_hands, Contract.SUN
            )
            counterfactual_projects = project_observation(
                sequence_rules, sequence_three_hands, Contract.SUN
            )
            if baseline_projects["projects"] != counterfactual_projects["projects"]:
                primitive_id = "rules.card_rules.sequence_strength"
                core = {
                    "schemaVersion": 1,
                    "primitiveId": primitive_id,
                    "status": "proven_behavioral",
                    "proofKind": "project_sequence_reference_runtime_cp_sat_counterfactual",
                    "rulesHash": baseline.sha256,
                    "counterfactualRulesHash": sequence_rules.sha256,
                    "observationHash": canonical_sha256(
                        [baseline_projects, counterfactual_projects]
                    ),
                    "targetsObserved": ["reference", "runtime", "cpSat"],
                    "referenceIndependent": True,
                    "proofBoundary": "fixed_complete_deal_with_sequence_order_counterfactual",
                }
                proofs[primitive_id] = {
                    **core,
                    "proofHash": canonical_sha256(core),
                }
        if len(_BEHAVIORAL_PROOF_CACHE) >= _BEHAVIORAL_PROOF_CACHE_MAXIMUM:
            _BEHAVIORAL_PROOF_CACHE.pop(next(iter(_BEHAVIORAL_PROOF_CACHE)))
        _BEHAVIORAL_PROOF_CACHE[cache_key] = copy.deepcopy(proofs)
        return proofs

    @classmethod
    def _presentation_proofs(
        cls,
        *,
        localization: Mapping[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """Load the one exact profile receipt without runtime source scans."""

        product_hash = canonical_sha256(localization)
        cached = _PRESENTATION_PROOF_CACHE.get(product_hash)
        if cached is not None:
            return copy.deepcopy(cached)
        payloads: list[dict[str, Any]] = []
        for resource_name in (
            "localization-usage-v1.json",
            "localization-usage-ortools-v4.json",
            f"localization-history/{product_hash}.json",
        ):
            try:
                resource = files("baloot_platform").joinpath(
                    "resources", "generated", resource_name
                )
                payload = json.loads(resource.read_text(encoding="utf-8"))
            except (FileNotFoundError, ModuleNotFoundError, json.JSONDecodeError):
                continue
            if payload.get("productHash") == product_hash:
                if not resource_name.startswith("localization-history/") or payload not in payloads:
                    payloads.append(payload)
        if len(payloads) != 1:
            return {}
        payload = payloads[0]
        if (
            payload.get("schemaVersion") != 1
            or payload.get("kind") != "LocalizationUsageManifestV1"
        ):
            return {}
        expected_hash = str(payload.get("sha256") or "")
        core = dict(payload)
        core.pop("sha256", None)
        if expected_hash != canonical_sha256(core):
            return {}
        proofs: dict[str, dict[str, Any]] = {}
        for item in payload.get("receipts") or ():
            receipt = dict(item)
            primitive_id = str(receipt.get("primitiveId") or "")
            status = str(receipt.get("status") or "")
            proof_hash = str(receipt.get("proofHash") or "")
            proof_core = dict(receipt)
            proof_core.pop("proofHash", None)
            if (
                not primitive_id.startswith("localization.")
                or primitive_id in proofs
                or status not in {"proven_behavioral", "proven_invariant"}
                or proof_hash != canonical_sha256(proof_core)
            ):
                return {}
            proofs[primitive_id] = receipt
        _PRESENTATION_PROOF_CACHE[product_hash] = copy.deepcopy(proofs)
        return copy.deepcopy(proofs)

    @staticmethod
    def _validated_behavioral_proof(
        *,
        primitive_id: str,
        proof: Mapping[str, Any] | None,
        rules_hash: str | None = None,
        profile_hash: str | None = None,
    ) -> dict[str, Any]:
        """Return one canonical proof receipt or fail closed to no evidence."""

        if not isinstance(proof, Mapping):
            return {}
        receipt = copy.deepcopy(dict(proof))
        if (
            receipt.get("primitiveId") != primitive_id
            or receipt.get("status")
            not in {"proven_behavioral", "proven_invariant"}
        ):
            return {}
        proof_hash = receipt.get("proofHash")
        core = dict(receipt)
        core.pop("proofHash", None)
        if not isinstance(proof_hash, str) or proof_hash != canonical_sha256(core):
            return {}
        if rules_hash is not None and receipt.get("rulesHash") != rules_hash:
            return {}
        if profile_hash is not None and receipt.get("profileHash") != profile_hash:
            return {}
        if primitive_id.startswith(("rules.", "conventions.")):
            if (
                type(receipt.get("schemaVersion")) is not int
                or receipt["schemaVersion"] != 1
                or not isinstance(receipt.get("proofKind"), str)
                or not receipt["proofKind"]
                or not isinstance(receipt.get("proofBoundary"), str)
                or not receipt["proofBoundary"]
                or not isinstance(receipt.get("targetsObserved"), list)
                or not receipt["targetsObserved"]
                or any(
                    not isinstance(target, str) or not target
                    for target in receipt["targetsObserved"]
                )
            ):
                return {}
        return receipt

    @classmethod
    def _instance(
        cls,
        *,
        primitive_id: str,
        definition: DomainPrimitiveDefinitionV1,
        value: Any,
        plugins: Mapping[str, Mapping[str, Any]],
        behavioral_proof: Mapping[str, Any] | None = None,
    ) -> DomainPrimitiveInstanceV1:
        covered = cls._covered_targets(definition, plugins=plugins)
        missing = tuple(
            item for item in definition.required_targets if item not in covered
        )
        executable_required = tuple(
            item for item in definition.required_targets if item in _EXECUTABLE_TARGETS
        )
        executable_complete = all(item in covered for item in executable_required)

        proof = cls._validated_behavioral_proof(
            primitive_id=primitive_id,
            proof=behavioral_proof,
        )
        behavioral_status = str(proof.get("status") or "unproven")
        qualification_eligible = executable_complete and behavioral_status in {
            "proven_behavioral",
            "proven_invariant",
        }
        return DomainPrimitiveInstanceV1(
            primitive_id=primitive_id,
            definition_kind=definition.kind,
            category=definition.category,
            value=value,
            value_hash=canonical_sha256(value),
            required_targets=definition.required_targets,
            covered_targets=covered,
            missing_targets=missing,
            coverage_status="covered" if qualification_eligible else "unproven",
            executable_binding_status=(
                "covered" if executable_complete else "incomplete"
            ),
            behavioral_or_invariant_status=behavioral_status,
            declared_behavioral_evidence=definition.influence_proof,
            verified_behavioral_evidence=(
                str(proof["proofHash"]) if proof.get("proofHash") else None
            ),
            qualification_eligible=qualification_eligible,
        )

    def compile(
        self,
        *,
        rules: Mapping[str, Any],
        conventions: Mapping[str, Any],
        localization: Mapping[str, Any],
        bundle_hash: str,
        plugins: Iterable[Mapping[str, Any]] = (),
    ) -> BalootDomainSpecV3:
        """Compile or reuse one complete, immutable domain result."""

        frozen_rules = copy.deepcopy(dict(rules))
        frozen_conventions = copy.deepcopy(dict(conventions))
        frozen_localization = copy.deepcopy(dict(localization))
        plugin_rows = tuple(copy.deepcopy(dict(item)) for item in plugins)
        cache_key = canonical_sha256(
            {
                "schemaVersion": 1,
                "rules": frozen_rules,
                "conventions": frozen_conventions,
                "localization": frozen_localization,
                "bundleHash": str(bundle_hash),
                "plugins": list(plugin_rows),
            }
        )
        with self._compiled_cache_lock:
            cached = self._compiled_cache.get(cache_key)
            if cached is not None:
                self._compiled_cache_hits += 1
                self._compiled_cache.move_to_end(cache_key)
                return copy.deepcopy(cached)
            self._compiled_cache_misses += 1

        compiled = self._compile_uncached(
            rules=frozen_rules,
            conventions=frozen_conventions,
            localization=frozen_localization,
            bundle_hash=str(bundle_hash),
            plugins=plugin_rows,
        )
        with self._compiled_cache_lock:
            cached = self._compiled_cache.get(cache_key)
            if cached is not None:
                self._compiled_cache.move_to_end(cache_key)
                return copy.deepcopy(cached)
            self._compiled_cache[cache_key] = copy.deepcopy(compiled)
            self._compiled_cache.move_to_end(cache_key)
            while len(self._compiled_cache) > self._compiled_cache_capacity:
                self._compiled_cache.popitem(last=False)
                self._compiled_cache_evictions += 1
        return compiled

    def _compile_uncached(
        self,
        *,
        rules: Mapping[str, Any],
        conventions: Mapping[str, Any],
        localization: Mapping[str, Any],
        bundle_hash: str,
        plugins: Iterable[Mapping[str, Any]] = (),
    ) -> BalootDomainSpecV3:
        frozen_rules = copy.deepcopy(dict(rules))
        frozen_conventions = copy.deepcopy(dict(conventions))
        frozen_localization = copy.deepcopy(dict(localization))
        self._validate_rules_payload(frozen_rules)
        validate_convention_profile(frozen_conventions)
        plugin_rows = tuple(copy.deepcopy(dict(item)) for item in plugins)
        plugin_index = {
            str(item.get("id") or ""): item for item in plugin_rows if item.get("id")
        }
        raw_behavioral_proofs = self._behavioral_proofs(
            rules=frozen_rules,
            plugins=plugin_index,
        )
        raw_behavioral_proofs.update(
            self._presentation_proofs(localization=frozen_localization)
        )
        raw_behavioral_proofs.update(
            convention_behavioral_proofs(
                conventions=frozen_conventions,
                rules=frozen_rules,
            )
        )
        rules_hash = canonical_sha256(frozen_rules)
        convention_hash = canonical_sha256(frozen_conventions)
        behavioral_proofs: dict[str, dict[str, Any]] = {}
        for primitive_id, proof in raw_behavioral_proofs.items():
            verified = self._validated_behavioral_proof(
                primitive_id=primitive_id,
                proof=proof,
                rules_hash=(
                    rules_hash
                    if primitive_id.startswith(("rules.", "conventions."))
                    else None
                ),
                profile_hash=(
                    convention_hash if primitive_id.startswith("conventions.") else None
                ),
            )
            if verified:
                behavioral_proofs[primitive_id] = verified
        unknown = sorted(
            set(frozen_rules) - {"schema_version", "profile"} - set(self._definitions)
        )
        if unknown:
            raise ValueError(f"unknown Baloot domain sections: {unknown}")

        instances: list[DomainPrimitiveInstanceV1] = []
        for section, payload in sorted(frozen_rules.items()):
            if section in {"schema_version", "profile"}:
                continue
            for path, leaf in _leaves(payload, ("rules", section)):
                definition = self._definition_for_path(section=section, path=path)
                instances.append(
                    self._instance(
                        primitive_id=".".join(path),
                        definition=definition,
                        value=leaf,
                        plugins=plugin_index,
                        behavioral_proof=behavioral_proofs.get(".".join(path)),
                    )
                )

        for family, payload in (
            ("conventions", frozen_conventions),
            ("localization", frozen_localization),
        ):
            definition = self._definitions[family]
            for path, leaf in _leaves(payload, (family,)):
                instances.append(
                    self._instance(
                        primitive_id=".".join(path),
                        definition=definition,
                        value=leaf,
                        plugins=plugin_index,
                        behavioral_proof=behavioral_proofs.get(".".join(path)),
                    )
                )

        instances.sort(key=lambda item: item.primitive_id)
        ir_core = {
            "schemaVersion": 1,
            "rulesHash": canonical_sha256(frozen_rules),
            "conventionHash": canonical_sha256(frozen_conventions),
            "localizationHash": canonical_sha256(frozen_localization),
            "primitiveInstances": [
                item.to_dict(include_value=True) for item in instances
            ],
        }
        domain_ir = DomainIRV1(
            frozen_rules,
            frozen_conventions,
            frozen_localization,
            tuple(instances),
            canonical_sha256(ir_core),
        )
        covered_count = sum(item.qualification_eligible for item in instances)
        executable_binding_covered = sum(
            item.executable_binding_status == "covered" for item in instances
        )
        behavioral_or_invariant_covered = sum(
            item.behavioral_or_invariant_status
            in {"proven_behavioral", "proven_invariant"}
            for item in instances
        )
        coverage_core = {
            "schemaVersion": 1,
            "rows": [item.to_dict() for item in instances],
            "covered": covered_count,
            "total": len(instances),
            "executableBindingCovered": executable_binding_covered,
            "behavioralOrInvariantCovered": behavioral_or_invariant_covered,
        }
        coverage = CompilerCoverageV1(
            rows=tuple(instances),
            covered=covered_count,
            total=len(instances),
            percentage=(
                100.0
                if not instances
                else round(covered_count * 100.0 / len(instances), 6)
            ),
            complete=covered_count == len(instances),
            executable_binding_covered=executable_binding_covered,
            executable_binding_total=len(instances),
            executable_binding_percentage=(
                100.0
                if not instances
                else round(executable_binding_covered * 100.0 / len(instances), 6)
            ),
            executable_binding_complete=executable_binding_covered == len(instances),
            behavioral_or_invariant_covered=behavioral_or_invariant_covered,
            behavioral_or_invariant_total=len(instances),
            behavioral_or_invariant_percentage=(
                100.0
                if not instances
                else round(behavioral_or_invariant_covered * 100.0 / len(instances), 6)
            ),
            behavioral_or_invariant_complete=(
                behavioral_or_invariant_covered == len(instances)
            ),
            sha256=canonical_sha256(coverage_core),
        )
        definition_rows = tuple(
            self._definitions[key] for key in sorted(self._definitions)
        )
        definitions_by_kind = {item.kind: item for item in definition_rows}
        lowering_receipts = {
            item.kind: self._executable_receipts(item, plugins=plugin_index)
            for item in definition_rows
        }
        auction_ir_payload = AuctionDomainIRV1.from_rules(frozen_rules).to_dict()
        structural_ir_payload = StructuralDomainIRV1.from_rules(frozen_rules).to_dict()
        escalation_ir_payload = EscalationDomainIRV1.from_rules(frozen_rules).to_dict()
        project_ir_payload = ProjectDomainIRV1.from_rules(frozen_rules).to_dict()
        play_ir_payload = PlayDomainIRV1.from_rules(frozen_rules).to_dict()
        scoring_ir_payload = ScoringDomainIRV1.from_rules(frozen_rules).to_dict()
        runtime_plan = {
            "schemaVersion": 1,
            "compiledStructural": structural_ir_payload,
            "compiledAuction": auction_ir_payload,
            "compiledEscalation": escalation_ir_payload,
            "compiledProjects": project_ir_payload,
            "compiledPlay": play_ir_payload,
            "compiledScoring": scoring_ir_payload,
            "entries": [
                {
                    "primitiveId": item.primitive_id,
                    "lowering": definitions_by_kind[
                        item.definition_kind
                    ].runtime_lowering,
                    "executableReceipt": lowering_receipts[item.definition_kind].get(
                        "runtime"
                    ),
                    "valueHash": item.value_hash,
                }
                for item in instances
                if definitions_by_kind[item.definition_kind].runtime_lowering
                is not None
            ],
        }
        reference_plan = {
            "schemaVersion": 1,
            "compiledStructural": structural_ir_payload,
            "compiledAuction": auction_ir_payload,
            "compiledEscalation": escalation_ir_payload,
            "compiledProjects": project_ir_payload,
            "compiledPlay": play_ir_payload,
            "compiledScoring": scoring_ir_payload,
            "entries": [
                {
                    "primitiveId": item.primitive_id,
                    "lowering": definitions_by_kind[
                        item.definition_kind
                    ].reference_lowering,
                    "executableReceipt": lowering_receipts[item.definition_kind].get(
                        "reference"
                    ),
                    "valueHash": item.value_hash,
                }
                for item in instances
                if definitions_by_kind[item.definition_kind].reference_lowering
                is not None
            ],
        }
        cp_sat_blueprint = {
            "schemaVersion": 1,
            "compiledStructural": structural_ir_payload,
            "compiledAuction": auction_ir_payload,
            "compiledEscalation": escalation_ir_payload,
            "compiledProjects": project_ir_payload,
            "compiledPlay": play_ir_payload,
            "compiledScoring": scoring_ir_payload,
            "entries": [
                {
                    "primitiveId": item.primitive_id,
                    "lowering": definitions_by_kind[
                        item.definition_kind
                    ].cp_sat_lowering,
                    "executableReceipt": lowering_receipts[item.definition_kind].get(
                        "cpSat"
                    ),
                    "valueHash": item.value_hash,
                }
                for item in instances
                if definitions_by_kind[item.definition_kind].cp_sat_lowering is not None
            ],
        }
        policy_plan = {
            "schemaVersion": 1,
            "entries": [
                {
                    "primitiveId": item.primitive_id,
                    "lowering": definitions_by_kind[
                        item.definition_kind
                    ].policy_lowering,
                    "executableReceipt": lowering_receipts[item.definition_kind].get(
                        "policy"
                    ),
                    "valueHash": item.value_hash,
                }
                for item in instances
                if definitions_by_kind[item.definition_kind].policy_lowering is not None
            ],
        }
        ui_schema = {
            "schemaVersion": 1,
            "fields": [
                {
                    "id": item.primitive_id,
                    "category": item.category,
                    "valueType": (
                        "boolean"
                        if isinstance(item.value, bool)
                        else "integer"
                        if isinstance(item.value, int)
                        else "number"
                        if isinstance(item.value, float)
                        else "array"
                        if isinstance(item.value, list)
                        else "string"
                    ),
                    "valueHash": item.value_hash,
                }
                for item in instances
            ],
        }
        documentation_model = {
            "schemaVersion": 1,
            "items": [
                {
                    "primitiveId": item.primitive_id,
                    "category": item.category,
                    "definitionKind": item.definition_kind,
                    "valueHash": item.value_hash,
                    "sources": list(
                        definitions_by_kind[item.definition_kind].documentation
                    ),
                }
                for item in instances
            ],
        }
        target_payloads = {
            "runtimePlan": runtime_plan,
            "referencePlan": reference_plan,
            "cpSatBlueprint": cp_sat_blueprint,
            "policyPlan": policy_plan,
            "uiSchema": ui_schema,
            "documentationModel": documentation_model,
            "executableLowerings": lowering_receipts,
            "behavioralProofs": behavioral_proofs,
        }
        runtime_hash = canonical_sha256(runtime_plan)
        reference_hash = canonical_sha256(reference_plan)
        cp_sat_hash = canonical_sha256(cp_sat_blueprint)
        policy_hash = canonical_sha256(policy_plan)
        ui_hash = canonical_sha256(ui_schema)
        docs_hash = canonical_sha256(documentation_model)
        compiled_core = {
            "schemaVersion": 1,
            "domainIrHash": domain_ir.sha256,
            "coverageHash": coverage.sha256,
            "targets": [
                runtime_hash,
                reference_hash,
                cp_sat_hash,
                policy_hash,
                ui_hash,
                docs_hash,
            ],
        }
        compiled = CompiledDomainBundleV1(
            domain_ir,
            coverage,
            definition_rows,
            runtime_hash,
            reference_hash,
            cp_sat_hash,
            policy_hash,
            ui_hash,
            docs_hash,
            target_payloads,
            coverage.complete,
            canonical_sha256(compiled_core),
        )
        spec_core = {
            "schemaVersion": 3,
            "bundleHash": str(bundle_hash),
            "rulesHash": ir_core["rulesHash"],
            "conventionHash": ir_core["conventionHash"],
            "localizationHash": ir_core["localizationHash"],
            "compiledHash": compiled.sha256,
            "pluginCapabilities": list(plugin_rows),
        }
        return BalootDomainSpecV3(
            str(bundle_hash),
            ir_core["rulesHash"],
            ir_core["conventionHash"],
            ir_core["localizationHash"],
            compiled,
            plugin_rows,
            canonical_sha256(spec_core),
        )
