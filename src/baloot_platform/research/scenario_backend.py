"""Bundle-bound authority for configurable scenario Research V4."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as package_version
import re
from typing import Any, Mapping

from baloot_platform.adjudication.research import AdjudicationResearchAuthorityV1
from baloot_platform.core.enums import Seat
from baloot_platform.tactics import TacticalDetectorProfileV1

from .scenario_contracts import (
    PLAYING_QUALITY_HORIZON_UNITS_BY_DECISION_V1,
    ResearchScenarioV1,
    ScenarioOriginV1,
    ScenarioProofRequestV1,
    ScenarioPlayingQualityPolicyV1,
    ScenarioQualityExecutionBudgetV1,
    canonical_sha256,
    hard_continuation_count_request_issue_v1,
    hard_tactic_causal_request_issue_v1,
)
from .ortools_flagship_research import OrToolsFlagshipResearchV1
from .ortools_research_executor import (
    OrToolsUnifiedResearchExecutorV1,
    _BidQualityCacheV1,
    _V3_BACKEND_VERSION,
    _reject_v3_semantic_callback_injection,
    _quality_budget_snapshots,
    resolve_quality_round_continuation_v1,
)
from .scenario_adjudication_predicates_v3 import (
    shared_scenario_predicate_catalog_lowering_manifest_v3,
)
from .scenario_predicate_manifest_v2 import (
    shared_scenario_predicate_catalog_lowering_manifest_v2,
)
from .scenario_predicate_capabilities_v3 import (
    NATIVE_V3_TACTIC_IDS,
    native_v3_state_decision_supported,
)
from .scenario_registry_v2 import ScenarioPredicateDefinitionV2
from .result_authority import validate_native_round_quality_evidence_v1, validate_trace_quality_evidence_v1


LEGACY_SCENARIO_RESEARCH_BACKEND = "legacy_hybrid_v1"
ORTOOLS_SCENARIO_RESEARCH_BACKEND = "ortools_cp_sat_v2"
ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3 = "ortools_cp_sat_v3"
ORTOOLS_SCENARIO_RESEARCH_BACKENDS = frozenset(
    {ORTOOLS_SCENARIO_RESEARCH_BACKEND, ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3}
)
SCENARIO_RESEARCH_BACKENDS = frozenset(
    {LEGACY_SCENARIO_RESEARCH_BACKEND, *ORTOOLS_SCENARIO_RESEARCH_BACKENDS}
)
SCENARIO_RESEARCH_PLANNERS = frozenset({"tail_first_v1"})
ORTOOLS_SCENARIO_RESEARCH_BACKEND_VERSION = "2.0.0"
ORTOOLS_SCENARIO_GENERATION_POLICY = "ortools_cp_sat_v2"
ORTOOLS_SCENARIO_EXECUTOR_ID = "ortools_unified_research_executor_v1"
ORTOOLS_SCENARIO_EXECUTOR_VERSION = "1.0.0"
LEGACY_SCENARIO_GENERATION_POLICIES = frozenset(
    {"natural_policy", "scenario_guided_actor_safe"}
)
_ORIGINS = frozenset({"search", "exact_trace_prefix"})
_PROOF_MODES = frozenset({"exact", "bounded", "policy_conditioned"})
_VISIBILITIES = frozenset({"complete_state", "actor_information_set"})
_PROOF_VISIBILITY_COMBINATIONS = frozenset(
    {
        ("exact", "complete_state"),
        ("bounded", "complete_state"),
        ("policy_conditioned", "complete_state"),
        ("exact", "actor_information_set"),
    }
)
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_DEFAULT_EXECUTION_CLAIMS = frozenset(
    {
        "outcome_reachable",
        "outcome_forced",
        "selected_action_is_unique_winner",
        "target_team_has_no_salvage",
        "winning_continuation_line_count",
    }
)
_WINNING_ROOT_ACTION_COUNT_CLAIM = "winning_root_action_count"
_COMPLETE_STATE_PLAY_CLAIMS = _DEFAULT_EXECUTION_CLAIMS | {
    _WINNING_ROOT_ACTION_COUNT_CLAIM
}
_DEFAULT_EXECUTION_UTILITIES = frozenset(
    {"awarded_point_margin", "contract_success", "round_win", "match_win", "kabout"}
)
_DEFAULT_EXECUTION_LINE_UNIVERSES = frozenset({None, "all_legal", "optimal_strategy"})
_POLICY_SEATS = frozenset({"north", "west", "south", "east"})
_NATIVE_V3_TACTIC_IDS = NATIVE_V3_TACTIC_IDS


def materialize_default_policy_profiles_v1(
    policy_catalog: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Materialize the frozen catalog default once for all operator paths."""

    policy_id = str(policy_catalog["defaultPolicyId"])
    definition = dict(dict(policy_catalog["profiles"])[policy_id])
    profiles: dict[str, dict[str, Any]] = {}
    for seed, seat in enumerate(("north", "west", "south", "east"), start=1):
        core = {
            "schemaVersion": 1,
            "seat": seat,
            "policyId": policy_id,
            "version": str(definition["version"]),
            "seed": seed,
            "choices": [],
            "materialization": str(definition["materialization"]),
        }
        profiles[seat] = {
            **{key: value for key, value in core.items() if key != "seat"},
            "profileHash": canonical_sha256(core),
        }
    return profiles

PHASE4_REQUIRED_COMBINATIONS_V1 = frozenset(
    (origin, decision_kind, mode, visibility)
    for origin in _ORIGINS
    for decision_kind in ("bid", "escalation", "project", "play")
    for mode, visibility in (
        ("exact", "complete_state"),
        ("bounded", "complete_state"),
        ("policy_conditioned", "complete_state"),
        ("exact", "actor_information_set"),
    )
)
# Exhaustive early-round bid, escalation, and project proofs remain hidden until
# their full-service graphs complete under a published tier.
PHASE4_FULL_SERVICE_PROVEN_COMBINATIONS_V1 = frozenset(
    combination
    for combination in PHASE4_REQUIRED_COMBINATIONS_V1
    if not (
        combination[1] in {"bid", "escalation", "project"}
        and combination[2] == "exact"
    )
)


@dataclass(frozen=True, slots=True)
class ScenarioResearchAuthorityV1:
    """Immutable Research authority selected by one frozen challenge profile."""

    backend: str
    planner: str | None
    deterministic_execution: bool
    policy_catalog: Mapping[str, Any]
    proof_budgets: Mapping[str, Any]
    search_budgets: Mapping[str, Any]
    diversity_defaults: Mapping[str, Any]
    cache_limits: Mapping[str, Any]
    refusal_thresholds: Mapping[str, Any]
    authoring_defaults: Mapping[str, Any]
    sha256: str
    quality_execution: ScenarioQualityExecutionBudgetV1 | None = None

    @classmethod
    def from_challenge_profile(
        cls, challenge_profile: Mapping[str, Any]
    ) -> "ScenarioResearchAuthorityV1":
        raw = challenge_profile.get("scenarioResearchV4")
        if raw is None:
            core = {
                "schemaVersion": 1,
                "backend": LEGACY_SCENARIO_RESEARCH_BACKEND,
            }
            return cls(
                LEGACY_SCENARIO_RESEARCH_BACKEND,
                None,
                True,
                {},
                {},
                {},
                {},
                {},
                {},
                {},
                canonical_sha256(core),
            )
        if not isinstance(raw, Mapping):
            raise TypeError("challenge scenarioResearchV4 must be an object")
        value = copy.deepcopy(dict(raw))
        required = {
            "schemaVersion",
            "backend",
            "planner",
            "deterministicExecution",
            "policyCatalog",
            "proofBudgets",
            "searchBudgets",
            "diversityDefaults",
            "cacheLimits",
            "refusalThresholds",
        }
        optional = {"authoringDefaults", "qualityExecution"}
        unknown = sorted(set(value) - required - optional)
        missing = sorted(required - set(value))
        if unknown or missing:
            raise ValueError(
                "challenge scenarioResearchV4 shape mismatch; "
                f"missing={missing}, unknown={unknown}"
            )
        if type(value["schemaVersion"]) is not int or value["schemaVersion"] != 1:
            raise ValueError("challenge scenarioResearchV4 schemaVersion must be 1")
        backend = str(value["backend"])
        if backend not in SCENARIO_RESEARCH_BACKENDS:
            raise ValueError("challenge scenarioResearchV4 backend is unsupported")
        if backend not in ORTOOLS_SCENARIO_RESEARCH_BACKENDS:
            raise ValueError(
                "an explicit scenarioResearchV4 section must select a published "
                "OR-Tools backend"
            )
        planner = str(value["planner"])
        if planner not in SCENARIO_RESEARCH_PLANNERS:
            raise ValueError("challenge scenarioResearchV4 planner is unsupported")
        if type(value["deterministicExecution"]) is not bool:
            raise TypeError(
                "challenge scenarioResearchV4 deterministicExecution must be boolean"
            )
        if value["deterministicExecution"] is not True:
            raise ValueError(
                "challenge scenarioResearchV4 must enable deterministic execution"
            )
        sections: dict[str, dict[str, Any]] = {}
        for name in (
            "policyCatalog",
            "proofBudgets",
            "searchBudgets",
            "diversityDefaults",
            "cacheLimits",
            "refusalThresholds",
        ):
            section = value[name]
            if not isinstance(section, Mapping):
                raise TypeError(
                    f"challenge scenarioResearchV4 {name} must be an object"
                )
            sections[name] = copy.deepcopy(dict(section))
        authoring_defaults = value.get("authoringDefaults", {})
        if not isinstance(authoring_defaults, Mapping):
            raise TypeError(
                "challenge scenarioResearchV4 authoringDefaults must be an object"
            )
        authoring_defaults = copy.deepcopy(dict(authoring_defaults))
        policy_catalog = sections["policyCatalog"]
        if set(policy_catalog) != {"schemaVersion", "defaultPolicyId", "profiles"}:
            raise ValueError(
                "challenge scenarioResearchV4 policyCatalog shape is invalid"
            )
        profiles = policy_catalog.get("profiles")
        if (
            policy_catalog.get("schemaVersion") != 1
            or policy_catalog.get("defaultPolicyId") != "fixed_action_map_v1"
            or not isinstance(profiles, Mapping)
            or set(profiles) != {"fixed_action_map_v1"}
            or dict(profiles["fixed_action_map_v1"])
            != {
                "version": "1.0.0",
                "materialization": "seeded_legal_action_order_v1",
            }
        ):
            raise ValueError(
                "challenge scenarioResearchV4 policyCatalog is unsupported"
            )
        core = {key: value[key] for key in sorted(required)}
        if authoring_defaults:
            core["authoringDefaults"] = authoring_defaults
        quality_execution = None
        if "qualityExecution" in value:
            if backend != ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3:
                raise ValueError("qualityExecution requires the native V3 backend")
            quality_execution = ScenarioQualityExecutionBudgetV1.build(value["qualityExecution"])
            if (quality_execution.exact_tail_budget_profile is not None
                    and quality_execution.exact_tail_budget_profile not in sections["proofBudgets"]["profiles"]):
                raise ValueError("quality exact tail proof budget profile is unavailable")
            core["qualityExecution"] = quality_execution.to_dict()
        return cls(
            backend,
            planner,
            True,
            sections["policyCatalog"],
            sections["proofBudgets"],
            sections["searchBudgets"],
            sections["diversityDefaults"],
            sections["cacheLimits"],
            sections["refusalThresholds"],
            authoring_defaults,
            canonical_sha256(core),
            quality_execution,
        )

    @property
    def is_ortools(self) -> bool:
        return self.backend in ORTOOLS_SCENARIO_RESEARCH_BACKENDS

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schemaVersion": 1,
            "backend": self.backend,
            "sha256": self.sha256,
        }
        if self.is_ortools:
            if self.quality_execution is not None:
                value["qualityExecution"] = self.quality_execution.to_dict()
            value.update(
                {
                    "planner": self.planner,
                    "deterministicExecution": self.deterministic_execution,
                    "policyCatalog": copy.deepcopy(dict(self.policy_catalog)),
                    "proofBudgets": copy.deepcopy(dict(self.proof_budgets)),
                    "searchBudgets": copy.deepcopy(dict(self.search_budgets)),
                    "diversityDefaults": copy.deepcopy(dict(self.diversity_defaults)),
                    "cacheLimits": copy.deepcopy(dict(self.cache_limits)),
                    "refusalThresholds": copy.deepcopy(dict(self.refusal_thresholds)),
                }
            )
            if self.authoring_defaults:
                value["authoringDefaults"] = copy.deepcopy(
                    dict(self.authoring_defaults)
                )
        return value


@dataclass(frozen=True, slots=True)
class OrToolsScenarioBackendIdentityV1:
    """Content-bound identity of one preflight-only backend instance."""

    backend_id: str
    backend_version: str
    solver_version: str
    planner: str
    bundle_revision_id: str
    bundle_identity_hash: str
    authority_hash: str
    predicate_catalog_hash: str
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "backendId": self.backend_id,
            "backendVersion": self.backend_version,
            "qualificationAuthority": "cp_sat",
            "solverVersion": self.solver_version,
            "planner": self.planner,
            "deterministicExecution": True,
            "bundleRevisionId": self.bundle_revision_id,
            "bundleIdentityHash": self.bundle_identity_hash,
            "authorityHash": self.authority_hash,
            "predicateCatalogHash": self.predicate_catalog_hash,
            "sha256": self.sha256,
        }


class OrToolsScenarioResearchBackendV2:
    """Engine-free, bundle-bound catalog, preflight, and execution boundary."""

    def __init__(
        self,
        *,
        challenge_profile: Mapping[str, Any],
        bundle_identity: Mapping[str, Any] | Any,
        rules: Any | None = None,
        adjudication_authority: Mapping[str, Any] | None = None,
        rule_source_catalog: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(challenge_profile, Mapping):
            raise TypeError("challenge_profile must be an already-resolved object")
        authority = ScenarioResearchAuthorityV1.from_challenge_profile(
            copy.deepcopy(dict(challenge_profile))
        )
        if not authority.is_ortools:
            raise ValueError(
                "OrToolsScenarioResearchBackendV2 requires an explicit "
                "ortools_cp_sat_v2 or ortools_cp_sat_v3 scenarioResearchV4 section"
            )
        raw_tactical = challenge_profile.get("tactical_detectors")
        if (authority.backend != ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3
                and isinstance(raw_tactical, Mapping) and "nativeDelayedPayoff" in raw_tactical):
            raise ValueError("nativeDelayedPayoff requires the native V3 Research backend")
        raw_bundle = (
            bundle_identity.to_dict()
            if hasattr(bundle_identity, "to_dict")
            else bundle_identity
        )
        if not isinstance(raw_bundle, Mapping):
            raise TypeError("bundle_identity must be an already-resolved object")
        bundle = copy.deepcopy(dict(raw_bundle))
        required_bundle_fields = {
            "bundleRevisionId",
            "semanticBundleHash",
            "effectiveRuntimeConfigHash",
        }
        missing = sorted(required_bundle_fields - set(bundle))
        if missing:
            raise ValueError(f"bundle_identity is incomplete: {missing}")
        revision_id = bundle["bundleRevisionId"]
        if type(revision_id) is not str or not revision_id.strip():
            raise ValueError("bundle_identity bundleRevisionId is invalid")
        for name in ("semanticBundleHash", "effectiveRuntimeConfigHash"):
            value = bundle[name]
            if type(value) is not str or _HASH_PATTERN.fullmatch(value) is None:
                raise ValueError(f"bundle_identity {name} is invalid")

        selected_adjudication_authority = (
            AdjudicationResearchAuthorityV1.from_mapping(
                copy.deepcopy(dict(adjudication_authority))
            )
            if adjudication_authority is not None
            else None
        )
        if (
            selected_adjudication_authority is not None
            and authority.backend != ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3
        ):
            raise ValueError("adjudication authority is supported only by V3")
        tactical_detector_profile = (
            TacticalDetectorProfileV1.build(copy.deepcopy(raw_tactical))
            if authority.backend == ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3
            and "tactical_detectors" in challenge_profile else None
        )
        native_delayed_payoff = (tactical_detector_profile is not None
            and tactical_detector_profile.native_delayed_gain_metric is not None)
        manifest = (
            shared_scenario_predicate_catalog_lowering_manifest_v3(
                selected_adjudication_authority.profile_id,
                selected_adjudication_authority.sha256,
                "false_ekka_attempt"
                in selected_adjudication_authority.capabilities,
                native_delayed_payoff=native_delayed_payoff,
            )
            if authority.backend == ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3
            and selected_adjudication_authority is not None
            else shared_scenario_predicate_catalog_lowering_manifest_v3(native_delayed_payoff=native_delayed_payoff)
            if authority.backend == ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3
            else shared_scenario_predicate_catalog_lowering_manifest_v2()
        )
        definitions = (*manifest.exposed_definitions, *manifest.hidden_definitions)
        if not definitions:
            raise ValueError("predicate catalog cannot be empty")
        by_kind = {definition.kind: definition for definition in definitions}
        if len(by_kind) != len(definitions):
            raise ValueError("predicate catalog contains duplicate kinds")

        self._authority = authority
        self._bid_quality_cache = (
            _BidQualityCacheV1(authority.cache_limits['maximumEntries'], authority.cache_limits['maximumBytes'])
            if authority.backend == ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3 and authority.quality_execution is not None
            else None
        )
        self._bundle = bundle
        self._tactical_detector_profile = tactical_detector_profile
        self._rules = rules
        self._rule_source_catalog = copy.deepcopy(rule_source_catalog)
        self._lowering_manifest = manifest
        self._definitions = tuple(sorted(definitions, key=lambda item: item.kind))
        self._by_kind = by_kind
        self._predicate_capabilities_v3 = (
            manifest.capability_manifest
            if authority.backend == ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3
            else None
        )
        bundle_hash = canonical_sha256(bundle)
        catalog_hash = manifest.manifest_hash
        try:
            solver_version = package_version("ortools")
        except PackageNotFoundError:
            solver_version = "not_installed"
        backend_version = (
            ORTOOLS_SCENARIO_RESEARCH_BACKEND_VERSION
            if authority.backend == ORTOOLS_SCENARIO_RESEARCH_BACKEND
            else _V3_BACKEND_VERSION
        )
        identity_core = {
            "schemaVersion": 1,
            "backendId": authority.backend,
            "backendVersion": backend_version,
            "qualificationAuthority": "cp_sat",
            "solverVersion": solver_version,
            "planner": authority.planner,
            "deterministicExecution": True,
            "bundleRevisionId": revision_id,
            "bundleIdentityHash": bundle_hash,
            "authorityHash": authority.sha256,
            "predicateCatalogHash": catalog_hash,
        }
        self._identity = OrToolsScenarioBackendIdentityV1(
            authority.backend,
            backend_version,
            solver_version,
            str(authority.planner),
            revision_id,
            bundle_hash,
            authority.sha256,
            catalog_hash,
            canonical_sha256(identity_core),
        )

    @property
    def identity(self) -> OrToolsScenarioBackendIdentityV1:
        return self._identity

    @property
    def authority(self) -> ScenarioResearchAuthorityV1:
        return self._authority

    @property
    def lowering_manifest(self) -> Any:
        """Return the frozen backend-selected predicate lowering authority."""

        return self._lowering_manifest

    def native_capability_plan(self, recipe: Mapping[str, Any] | Any) -> dict[str, Any]:
        """Bind requested predicates to the selected V3 catalog, not V1 metadata."""

        if self.authority.backend != ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3:
            raise ValueError("native capability plans require the V3 backend")
        raw = self._recipe_mapping(recipe)
        scenario = ResearchScenarioV1.build(raw["scenario"])
        proof = ScenarioProofRequestV1.build(raw["proof"])
        catalog = self._predicate_capabilities_v3
        assert catalog is not None
        rows = []
        for predicate in sorted(scenario.predicates(), key=lambda item: item.predicate_id):
            capability = catalog.capability(predicate.kind)
            rows.append({
                "predicateId": predicate.predicate_id,
                "predicateHash": predicate.sha256,
                "kind": predicate.kind,
                "hardness": predicate.hardness,
                "nativeStatus": (
                    capability.native_status if capability is not None else "unsupported"
                ),
                "capabilityHash": capability.sha256 if capability is not None else None,
                "qualificationBinding": (
                    capability.qualification_binding if capability is not None else None
                ),
            })
        core = {
            "schemaVersion": 3,
            "backendIdentityHash": self.identity.sha256,
            "catalogHash": catalog.manifest_hash,
            "scenarioHash": scenario.sha256,
            "proofHash": proof.sha256,
            "rows": rows,
        }
        return {**core, "sha256": canonical_sha256(core)}

    @property
    def executor_identity(self) -> dict[str, Any]:
        """Return the immutable execution identity for newly authored V4 runs."""

        is_v3 = self.authority.backend == ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3
        core = {
            "schemaVersion": 1,
            "executorId": (
                "ortools_unified_research_executor_v3"
                if is_v3
                else ORTOOLS_SCENARIO_EXECUTOR_ID
            ),
            "executorVersion": _V3_BACKEND_VERSION if is_v3 else ORTOOLS_SCENARIO_EXECUTOR_VERSION,
            "backendIdentityHash": self.identity.sha256,
            "loweringManifestHash": self.identity.predicate_catalog_hash,
            "bundleRevisionId": self.identity.bundle_revision_id,
            "bundleIdentityHash": self.identity.bundle_identity_hash,
        }
        return {**core, "sha256": canonical_sha256(core)}

    def execution_support(self, recipe: Mapping[str, Any] | Any) -> dict[str, Any]:
        """Describe the default execution slices that are currently supported.

        Construction and predicate lowering cover a wider catalog. The default
        executor supports scored play continuations plus search or exact-input
        bid and escalation roots under frozen declarative policies. Optional
        project declarations are strategic only at a fully authored final
        first-trick declaration point; mandatory declarations remain forced
        transitions rather than decisions.
        Search additionally requires a deterministic decision selector so its
        trace can be frozen before proof.
        """

        raw = self._recipe_mapping(recipe)
        scenario = (
            raw.get("scenario") if isinstance(raw.get("scenario"), Mapping) else {}
        )
        proof = raw.get("proof") if isinstance(raw.get("proof"), Mapping) else {}
        resources = (
            raw.get("resources") if isinstance(raw.get("resources"), Mapping) else {}
        )
        origin = (
            scenario.get("origin")
            if isinstance(scenario.get("origin"), Mapping)
            else {}
        )
        decision = (
            scenario.get("decisionPoint")
            if isinstance(scenario.get("decisionPoint"), Mapping)
            else {}
        )
        selector = (
            decision.get("selector")
            if isinstance(decision.get("selector"), Mapping)
            else {}
        )
        decision_kind = decision.get("kind")
        project_mode = None
        rules_data = getattr(self._rules, "data", None)
        if isinstance(rules_data, Mapping):
            projects = rules_data.get("projects")
            if isinstance(projects, Mapping):
                project_mode = projects.get("declaration_mode")
        project_decision_reason = "project_declaration_mode_missing_or_mandatory"
        trick_index = selector.get("trickIndex")
        remaining_cards = selector.get("remainingCards")
        search_play_selector_ready = (
            decision_kind == "play"
            and type(trick_index) is int
            and 0 <= trick_index <= 7
            and (
                remaining_cards is None
                or (
                    type(remaining_cards) is int
                    and 0 <= remaining_cards <= 8
                )
            )
        )
        origin_kind = origin.get("kind")
        project_selector_ready = (
            selector.get("owner") is not None
            and selector.get("projectIndex", 0) == 0
        )
        project_policy_ready = (
            project_mode == "optional"
            and origin_kind in {"search", "exact_trace_prefix"}
            and decision_kind == "project"
            and project_selector_ready
            and proof.get("mode") == "policy_conditioned"
            and proof.get("visibility") == "complete_state"
            and proof.get("strategyClass") == "frozen_policy"
        )
        project_complete_ready = (
            project_mode == "optional"
            and origin_kind in {"search", "exact_trace_prefix"}
            and decision_kind == "project"
            and project_selector_ready
            and proof.get("mode") in {"exact", "bounded"}
            and proof.get("visibility") == "complete_state"
            and proof.get("strategyClass") in {None, "pure_deterministic"}
        )
        nonplay_actor_information_ready = (
            origin_kind in {"search", "exact_trace_prefix"}
            and proof.get("mode") == "exact"
            and proof.get("visibility") == "actor_information_set"
            and proof.get("strategyClass") in {None, "pure_deterministic"}
            and proof.get("claim") != "winning_continuation_line_count"
            and (
                (
                    decision_kind == "bid"
                    and type(selector.get("actionIndex")) is int
                    and selector["actionIndex"] >= 0
                )
                or (
                    decision_kind == "escalation"
                    and selector.get("actingSeat") is not None
                )
                or (
                    decision_kind == "project"
                    and project_mode == "optional"
                    and project_selector_ready
                )
            )
        )
        search_nonplay_selector_ready = (
            (
                decision_kind == "bid"
                and type(selector.get("actionIndex")) is int
                and selector["actionIndex"] >= 0
            )
            or (
                decision_kind == "escalation"
                and type(selector.get("actionIndex")) is int
                and selector["actionIndex"] >= 0
                and selector.get("actingSeat") is not None
            )
        )
        nonplay_policy_ready = (
            project_policy_ready
            or (
                origin_kind in {"search", "exact_trace_prefix"}
                and decision_kind in {"bid", "escalation"}
                and proof.get("mode") == "policy_conditioned"
                and proof.get("visibility") == "complete_state"
                and proof.get("strategyClass") == "frozen_policy"
                and (
                    (
                        type(selector.get("actionIndex")) is int
                        and selector["actionIndex"] >= 0
                    )
                    if decision_kind == "bid"
                    else selector.get("actingSeat") is not None
                )
            )
        )
        nonplay_complete_ready = (
            project_complete_ready
            or (
                origin_kind in {"search", "exact_trace_prefix"}
                and decision_kind in {"bid", "escalation"}
                and proof.get("mode") in {"exact", "bounded"}
                and proof.get("visibility") == "complete_state"
                and proof.get("strategyClass") in {None, "pure_deterministic"}
                and (
                    type(selector.get("actionIndex")) is int
                    and selector["actionIndex"] >= 0
                    if decision_kind == "bid"
                    else selector.get("actingSeat") is not None
                )
            )
        )
        phase4_full_service_ready = (
            self.authority.backend != ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3
            or (
                origin_kind,
                decision_kind,
                proof.get("mode"),
                proof.get("visibility"),
            )
            in PHASE4_FULL_SERVICE_PROVEN_COMBINATIONS_V1
        )
        origin_ready = origin_kind == "exact_trace_prefix" or (
            origin_kind == "search"
            and (
                search_play_selector_ready
                or search_nonplay_selector_ready
                or project_policy_ready
                or project_complete_ready
                or nonplay_actor_information_ready
            )
        )
        origin_reason = (
            "search_decision_selector_required"
            if origin_kind == "search"
            else "exact_origin_required"
        )
        checks = (
            (
                getattr(self.authority.quality_execution, 'family_proposal_mode', None) is None
                or (origin_kind == 'search' and decision_kind == 'play'
                    and proof.get('mode') == 'exact' and proof.get('visibility') == 'actor_information_set'),
                'family_target_inapplicable',
            ),
            (origin_ready, origin_reason),
            (
                (
                    decision_kind != "project"
                    or project_policy_ready
                    or project_complete_ready
                    or nonplay_actor_information_ready
                ),
                project_decision_reason,
            ),
            (
                phase4_full_service_ready,
                "phase4_substantive_full_service_evidence_missing",
            ),
            (
                proof.get("visibility") != "actor_information_set"
                or proof.get("mode") == "exact",
                "actor_information_set_requires_exact_mode",
            ),
            (
                proof.get("visibility") != "actor_information_set"
                or proof.get("claim") != "winning_continuation_line_count",
                "actor_information_set_continuation_count_unsupported",
            ),
            (
                (
                    decision_kind == "play"
                    or nonplay_policy_ready
                    or nonplay_complete_ready
                    or nonplay_actor_information_ready
                ),
                "nonplay_terminal_continuation_unavailable",
            ),
            (
                proof.get("mode") in {"exact", "bounded", "policy_conditioned"},
                "default_execution_proof_mode_unsupported",
            ),
            (
                proof.get("visibility") == "complete_state"
                or (
                    proof.get("visibility") == "actor_information_set"
                    and proof.get("mode") == "exact"
                    and (
                        decision.get("kind") == "play"
                        or nonplay_actor_information_ready
                    )
                    and origin_kind in {"search", "exact_trace_prefix"}
                ),
                "default_execution_requires_complete_state",
            ),
            (
                proof.get("claim") in _DEFAULT_EXECUTION_CLAIMS
                or (
                    (decision_kind in {"play", "bid"} or nonplay_complete_ready)
                    and proof.get("visibility") == "complete_state"
                    and proof.get("claim") == _WINNING_ROOT_ACTION_COUNT_CLAIM
                )
                or (
                    proof.get("visibility") == "actor_information_set"
                    and proof.get("claim") == _WINNING_ROOT_ACTION_COUNT_CLAIM
                ),
                # Actor-information proof already certifies this exact count.
                "proof_claim_not_executable",
            ),
            (
                proof.get("utility") in _DEFAULT_EXECUTION_UTILITIES,
                "proof_utility_not_executable",
            ),
            (
                proof.get("utility") != "kabout"
                or proof.get("mode") != "bounded"
                or dict(proof.get("horizon") or {}).get("unit") == "round_end",
                "bounded_kabout_requires_round_end",
            ),
            (
                (
                    proof.get("lineUniverse") is None
                    if proof.get("visibility") == "actor_information_set"
                    else proof.get("lineUniverse")
                    in (
                        {None, "policy_conditioned"}
                        if proof.get("mode") == "policy_conditioned"
                        else _DEFAULT_EXECUTION_LINE_UNIVERSES
                    )
                ),
                "line_universe_not_executable",
            ),
            (
                proof.get("mode") != "bounded"
                or decision_kind == "play"
                or (
                    self.authority.backend == ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3
                    and proof.get("visibility") == "complete_state"
                    and dict(proof.get("horizon") or {}).get("unit")
                    in {"tricks", "round_end"}
                )
                or dict(proof.get("horizon") or {}).get("unit")
                in {"plies", "decisions"},
                "nonplay_bounded_horizon_unit_unsupported",
            ),
            (
                proof.get("mode") != "policy_conditioned"
                or not self._policy_issues(proof),
                "policy_profile_unsupported",
            ),
            (
                proof.get("mode") != "policy_conditioned"
                or (
                    nonplay_policy_ready
                    or (
                        decision_kind == "play"
                        and origin_kind in {"search", "exact_trace_prefix"}
                        and proof.get("visibility") == "complete_state"
                        and proof.get("strategyClass") == "frozen_policy"
                        and proof.get("claim") in _COMPLETE_STATE_PLAY_CLAIMS
                        and (
                            proof.get("claim") != "winning_continuation_line_count"
                            or proof.get("lineUniverse") == "policy_conditioned"
                        )
                    )
                ),
                "policy_execution_combination_unproven",
            ),
            (
                proof.get("budgetProfile") == resources.get("proofBudgetProfile"),
                "proof_budget_binding_mismatch",
            ),
        )
        reasons = [reason for supported, reason in checks if not supported]
        selection_constraints = (
            {"contractValues": ["sun", "hokum"]}
            if phase4_full_service_ready
            and proof.get("visibility") == "actor_information_set"
            and proof.get("mode") == "exact"
            and decision.get("kind") == "play"
            and origin_kind in {"search", "exact_trace_prefix"}
            else {"projectDecisionOrdinals": [3], "projectIndexes": [0]}
            if phase4_full_service_ready
            and decision_kind == "project"
            and (project_policy_ready or project_complete_ready)
            else {}
        )
        core = {
            "schemaVersion": 1,
            "backendIdentityHash": self.identity.sha256,
            "executorIdentity": self.executor_identity,
            "executionReady": not reasons,
            "reasons": reasons,
            "selectionConstraints": selection_constraints,
        }
        return {**core, "sha256": canonical_sha256(core)}

    def catalog(
        self,
        *,
        origin: str,
        proof_mode: str,
        visibility: str,
        hardness: str = "hard",
    ) -> dict[str, Any]:
        self._require_combination(origin, proof_mode, visibility)
        if hardness not in {"hard", "prefer", "report"}:
            raise ValueError("predicate hardness is unsupported")
        selected = [
            definition
            for definition in self._definitions
            if hardness in definition.allowed_hardness
            and self._supports(definition, origin, proof_mode, visibility)
        ]
        definitions = [definition.to_dict() for definition in selected]
        if self.authority.backend == ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3:
            for definition in definitions:
                if definition["kind"] not in {"tactic.applicable", "tactic.verified"}:
                    continue
                capability = self._predicate_capabilities_v3.capability(definition["kind"])
                supported_tactics = {
                    item.argument_value for item in capability.argument_contexts
                    if not item.execution_contexts or any(
                        context.origin == origin and context.proof_mode == proof_mode
                        and context.visibility == visibility
                        for context in item.execution_contexts
                    )
                }
                kind_schema = definition["argumentSchema"]["properties"]["kind"]
                kind_schema["enum"] = sorted(supported_tactics)
                for field in ("x-enumMessageArKeys", "x-enumMessageEnKeys"):
                    kind_schema[field] = {
                        key: value
                        for key, value in kind_schema[field].items()
                        if key in supported_tactics
                    }
        core = {
            "schemaVersion": 2,
            "backendIdentityHash": self.identity.sha256,
            "origin": origin,
            "proofMode": proof_mode,
            "visibility": visibility,
            "hardness": hardness,
            "definitions": definitions,
        }
        return {**core, "sha256": canonical_sha256(core)}

    def validate(self, recipe: Mapping[str, Any] | Any) -> dict[str, Any]:
        raw = self._recipe_mapping(recipe)
        issues: list[dict[str, Any]] = []
        qualification = raw.get("qualification")
        raw_quality = raw.get("playingQuality")
        raw_scenario = raw.get("scenario")
        if "generation" in raw:
            try:
                self._validate_generation_request(raw["generation"], ResearchScenarioV1.build(raw_scenario), raw_quality)
            except (TypeError, ValueError) as error:
                issues.append(self._issue("family_generation_inapplicable", "generation", str(error)))
        family_target = getattr(self.authority.quality_execution, "family_target", None)
        if family_target is not None:
            from .family_catalog import validate_family_target_scenario_v1
            try:
                validate_family_target_scenario_v1(family_target, ResearchScenarioV1.build(raw_scenario),
                    rules=self._rules, execution=self.authority.quality_execution, playing_quality=raw_quality)
                if not isinstance(raw_quality, Mapping):
                    raise ValueError("familyTarget requires playingQuality")
                if self.authority.quality_execution.family_proposal_mode is not None:
                    from .ortools_tail_first_candidate import family_tail_first_plan_v1
                    if 'generation' in raw:
                        raise ValueError('familyProposalMode requires the explicit familyTarget, not generation')
                    family_tail_first_plan_v1(self._rules, self.authority.quality_execution,
                        scenario=ResearchScenarioV1.build(raw_scenario),
                        request=ScenarioProofRequestV1.build(raw.get('proof')),
                        playing_quality=ScenarioPlayingQualityPolicyV1.build(raw_quality))
            except (TypeError, ValueError) as error:
                issues.append(self._issue("family_target_inapplicable", "scenario", str(error)))
        raw_origin = raw_scenario.get("origin") if isinstance(raw_scenario, Mapping) else None
        exact_quality_available = False
        exact_quality_input_issue = None
        if isinstance(raw_origin, Mapping) and raw_origin.get("kind") == "exact_trace_prefix":
            try:
                exact_origin = ScenarioOriginV1.build(raw_origin)
                prefix = exact_origin.exact_trace_prefix
                history = exact_origin.recorded_escalation_opportunities
                execution = self.authority.quality_execution
                if (self.authority.backend == ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3
                        and isinstance(raw_quality, Mapping) and execution is not None
                        and execution.escalation_opportunity_order is not None
                        and history is not None and prefix is not None):
                    if not prefix.project_actions_present:
                        exact_quality_input_issue = self._issue(
                            "exact_input_project_actions_required",
                            "scenario.origin.exactTracePrefix.projectActions",
                            "Record native legal project actions interleaved with first-trick plays; "
                            "legacy declarations alone cannot certify all-decision quality.",
                        )
                    elif prefix.project_declarations:
                        exact_quality_input_issue = self._issue(
                            "exact_input_project_declarations_legacy",
                            "scenario.origin.exactTracePrefix.projectDeclarations",
                            "Use the native project-action timeline instead of legacy declarations; "
                            "do not discard a real declaration without recording its legal action.",
                        )
                    if exact_quality_input_issue is not None:
                        issues.append(exact_quality_input_issue)
                exact_quality_available = (
                    self.authority.backend == ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3
                    and isinstance(raw_quality, Mapping)
                    and (raw_quality.get("mode") == "budgeted" or
                         (raw_quality.get("mode") == "exact" and execution is not None
                          and execution.allow_exact_input_quality))
                    and execution is not None and execution.escalation_opportunity_order is not None
                    and history is not None and len(history) <= execution.maximum_escalation_decisions
                    and prefix is not None and prefix.project_actions_present and not prefix.project_declarations
                    and (bool(prefix.project_actions or prefix.play_actions)
                         or not (prefix.escalation_actions or history)
                         or (isinstance(raw_scenario.get("decisionPoint"), Mapping)
                             and raw_scenario["decisionPoint"].get("kind") == "escalation"
                             and isinstance(raw.get("proof"), Mapping)
                             and raw["proof"].get("mode") in {"exact", "bounded"}
                             and not (prefix.state_assertions or prefix.project_assertions)))
                )
            except (TypeError, ValueError):
                pass  # Canonical scenario validation reports malformed exact input below.
        if isinstance(raw_origin, Mapping) and "recordedEscalationOpportunities" in raw_origin:
            history_path = "scenario.origin.recordedEscalationOpportunities"
            history = raw_origin["recordedEscalationOpportunities"]
            execution = self.authority.quality_execution
            if not isinstance(history, list):
                issues.append(self._issue("recorded_escalation_history_invalid", history_path))
            elif execution is None:
                issues.append(self._issue("frozen_quality_execution_unavailable", history_path))
            elif len(history) > execution.maximum_escalation_decisions:
                issues.append(self._issue("recorded_escalation_history_exceeds_budget", history_path))
            if not isinstance(raw_quality, Mapping):
                issues.append(self._issue("recorded_escalation_history_requires_quality_policy", history_path))
            if not exact_quality_available and exact_quality_input_issue is None:
                issues.append(self._issue("recorded_escalation_history_audit_unavailable", history_path))
        quality_available = exact_quality_available or (
            self.authority.backend == ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3
            and isinstance(raw_origin, Mapping) and raw_origin.get("kind") == "search"
            and isinstance(raw_quality, Mapping) and raw_quality.get("mode") in {"budgeted", "exact"}
            and self.authority.quality_execution is not None
            and self.authority.quality_execution.escalation_opportunity_order is not None
        )
        quality_unavailable_detail = "selected_decision_proof_only"
        if isinstance(raw_quality, Mapping):
            if isinstance(raw_origin, Mapping) and raw_origin.get("kind") == "exact_trace_prefix":
                quality_unavailable_detail = "exact_input_recorded_quality_dispatch_unavailable"
            elif raw_quality.get("mode") == "exact":
                quality_unavailable_detail = "exact_full_round_quality_coverage_unavailable"
            elif raw_quality.get("mode") == "budgeted":
                quality_unavailable_detail = "frozen_quality_execution_unavailable"
        if (
            self.authority.backend == ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3
            and isinstance(qualification, Mapping)
            and qualification.get("requireAllDecisionAudit") is True
            and not quality_available
        ):
            issues.append({
                "code": "all_decision_quality_evidence_unavailable",
                "path": "qualification.requireAllDecisionAudit",
                "detail": quality_unavailable_detail,
            })
        scenario = raw.get("scenario")
        proof = raw.get("proof")
        resources = raw.get("resources", {})
        if not isinstance(scenario, Mapping):
            issues.append(self._issue("scenario_required", "scenario"))
            scenario = {}
        if not isinstance(proof, Mapping):
            issues.append(self._issue("proof_required", "proof"))
            proof = {}
        if not isinstance(resources, Mapping):
            issues.append(self._issue("resources_invalid", "resources"))
            resources = {}

        origin_value = scenario.get("origin", {})
        origin = origin_value.get("kind") if isinstance(origin_value, Mapping) else None
        mode = proof.get("mode")
        visibility = proof.get("visibility")
        try:
            self._require_combination(origin, mode, visibility)
        except ValueError as exc:
            issues.append(
                self._issue("unsupported_research_combination", "proof", str(exc))
            )

        lane = resources.get("generationLane", "ortools")
        if lane != "ortools":
            issues.append(
                self._issue(
                    "legacy_generation_lane_forbidden",
                    "resources.generationLane",
                )
            )
        policy = resources.get("generationPolicy")
        if policy in LEGACY_SCENARIO_GENERATION_POLICIES:
            issues.append(
                self._issue(
                    "legacy_generation_policy_forbidden",
                    "resources.generationPolicy",
                )
            )
        elif policy not in {None, self._authority.backend}:
            issues.append(
                self._issue(
                    "generation_policy_unsupported",
                    "resources.generationPolicy",
                )
            )

        budget_profile = proof.get("budgetProfile")
        proof_profiles = self._profiles(self._authority.proof_budgets)
        search_profiles = self._profiles(self._authority.search_budgets)
        if "playingQuality" in raw:
            try:
                quality_policy = ScenarioPlayingQualityPolicyV1.build(raw["playingQuality"])
            except (TypeError, ValueError) as exc:
                issues.append(self._issue("playing_quality_invalid", "playingQuality", str(exc)))
            else:
                if self.authority.backend != ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3:
                    issues.append(self._issue("playing_quality_backend_unsupported", "playingQuality"))
                elif (self.authority.backend == ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3
                        and not quality_available
                        and not any(row["code"] == "all_decision_quality_evidence_unavailable" for row in issues)):
                    issues.append({
                        "code": "all_decision_quality_evidence_unavailable",
                        "path": "playingQuality",
                        "detail": quality_unavailable_detail,
                    })
                for kind, analysis in quality_policy.decisions.items():
                    if analysis["budgetProfile"] not in proof_profiles:
                        issues.append(self._issue(
                            "budget_profile_unsupported", f"playingQuality.decisions.{kind}.budgetProfile",
                        ))
                        continue
                    horizon = analysis.get("horizon")
                    if (
                        horizon is not None
                        and horizon["unit"]
                        not in PLAYING_QUALITY_HORIZON_UNITS_BY_DECISION_V1[kind]
                    ):
                        issues.append(
                            self._issue(
                                "playing_quality_horizon_unit_unsupported",
                                f"playingQuality.decisions.{kind}.horizon",
                                f"unit={horizon['unit']}",
                            )
                        )
                    if horizon is not None and horizon["unit"] != "round_end":
                        requested_actions = horizon["value"] * (len(Seat) if horizon["unit"] == "tricks" else 1)
                        budget = proof_profiles[analysis["budgetProfile"]]
                        for limit, code in (
                            ("maximumHorizonActions", "bounded_horizon_exceeds_proof_budget"),
                            ("maximumDepth", "bounded_depth_exceeds_proof_budget"),
                        ):
                            if requested_actions > budget[limit]:
                                issues.append(self._issue(
                                    code, f"playingQuality.decisions.{kind}.horizon",
                                    f"requestedActions={requested_actions};{limit}={budget[limit]}",
                                ))
        if (
            type(budget_profile) is not str
            or budget_profile not in proof_profiles
            or budget_profile not in search_profiles
        ):
            issues.append(
                self._issue("budget_profile_unsupported", "proof.budgetProfile")
            )
        decision = scenario.get("decisionPoint")
        if isinstance(decision, Mapping) and proof.get("mode") == "policy_conditioned":
            for code, path, detail in self._policy_issues(proof):
                issues.append(self._issue(code, path, detail))

        compiled_stages: set[str] = set()
        for path, predicate in self._predicates(scenario):
            kind = predicate.get("kind")
            version = predicate.get("version")
            hardness = predicate.get("hardness")
            definition = self._by_kind.get(kind) if type(kind) is str else None
            if definition is None or version != definition.version:
                issues.append(self._issue("predicate_unsupported", path))
                continue
            if hardness not in definition.allowed_hardness:
                issues.append(self._issue("predicate_hardness_unsupported", path))
                continue
            try:
                arguments = predicate.get("arguments")
                definition.validate_arguments(arguments)
            except (TypeError, ValueError) as exc:
                issues.append(
                    self._issue("predicate_arguments_invalid", path, str(exc))
                )
                continue
            if self.authority.backend == ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3:
                causal_issue = hard_tactic_causal_request_issue_v1(
                    str(kind), str(hardness), arguments,
                    native_delayed_payoff=(self._tactical_detector_profile is not None
                        and self._tactical_detector_profile.native_delayed_gain_metric is not None),
                )
                if causal_issue is not None:
                    issues.append(self._issue(
                        causal_issue, f"{path}.arguments.causalRequired", str(arguments["kind"]),
                    ))
                    continue
                count_issue = hard_continuation_count_request_issue_v1(
                    str(kind), str(hardness), arguments,
                    proof.get("selectedAction"), proof.get("lineUniverse"),
                )
                if count_issue is not None:
                    issues.append(self._issue(
                        count_issue,
                        "proof.selectedAction" if proof.get("selectedAction") is None
                        else "proof.lineUniverse",
                        path,
                    ))
                    continue
            if (
                self.authority.backend == ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3
                and definition.kind in {"tactic.applicable", "tactic.verified"}
                and str(arguments["kind"]) not in _NATIVE_V3_TACTIC_IDS
            ):
                issues.append(
                    self._issue(
                        "tactic_native_producer_unavailable",
                        path,
                        str(arguments["kind"]),
                    )
                )
                continue
            if (
                self.authority.backend == ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3
                and definition.stage == "state"
                and isinstance(decision, Mapping)
                and not native_v3_state_decision_supported(
                    definition.kind, str(decision.get("kind")),
                )
            ):
                issues.append(self._issue(
                    "predicate_native_nonplay_state_unavailable", path,
                    str(decision.get("kind")),
                ))
                continue
            if self._predicate_capabilities_v3 is not None:
                capability = self._predicate_capabilities_v3.capability(definition.kind)
                decision_kind = (
                    decision.get("kind") if isinstance(decision, Mapping) else None
                )
                if (
                    capability is not None
                    and capability.argument_contexts
                    and definition.kind in {"tactic.applicable", "tactic.verified"}
                ):
                    argument_context = next(
                        (
                            item
                            for item in capability.argument_contexts
                            if item.argument_name == "kind"
                            and item.argument_value == str(arguments["kind"])
                        ),
                        None,
                    )
                    if argument_context is None:
                        issues.append(
                            self._issue(
                                "tactic_native_producer_unavailable",
                                path,
                                str(arguments["kind"]),
                            )
                        )
                        continue
                    if not argument_context.required_proof_bindings:
                        issues.append(
                            self._issue(
                                "tactic_proof_bindings_unavailable",
                                path,
                                str(arguments["kind"]),
                            )
                        )
                        continue
                    if decision_kind not in argument_context.decision_kinds:
                        issues.append(
                            self._issue(
                                "tactic_decision_kind_unsupported",
                                path,
                                str(decision_kind),
                            )
                        )
                        continue
                    if (
                        argument_context.required_utility is not None
                        and proof.get("utility")
                        != argument_context.required_utility
                    ):
                        issues.append(
                            self._issue(
                                "tactic_proof_utility_unsupported",
                                path,
                                str(argument_context.required_utility),
                            )
                        )
                        continue
                    if argument_context.execution_contexts and not any(
                        context.origin == origin
                        and context.decision_kind == decision_kind
                        and context.proof_mode == mode
                        and context.visibility == visibility
                        and (not context.line_universes
                             or proof.get("lineUniverse") in context.line_universes)
                        for context in argument_context.execution_contexts
                    ):
                        issues.append(self._issue(
                            "predicate_v3_context_unsupported", path,
                            str(decision_kind),
                        ))
                        continue
                line_universe = proof.get("lineUniverse")
                if capability is None or not any(
                    context.origin == origin
                    and context.decision_kind == decision_kind
                    and context.proof_mode == mode
                    and context.visibility == visibility
                    and (
                        not context.line_universes
                        or line_universe in context.line_universes
                    )
                    for context in capability.execution_contexts
                ):
                    issues.append(
                        self._issue(
                            "predicate_v3_context_unsupported",
                            path,
                            str(decision_kind),
                        )
                    )
                    continue
            if (
                hardness == "hard"
                and visibility == "actor_information_set"
                and definition.cp_sat_support.actor_information_set
                == "reference_world_only"
            ):
                issues.append(
                    self._issue(
                        "predicate_reference_world_only",
                        path,
                        "hard actor-information-set predicates must be observer-safe",
                    )
                )
                continue
            if hardness == "hard" and mode in _PROOF_MODES:
                proof_status = {
                    "exact": definition.cp_sat_support.exact_proof,
                    "bounded": definition.cp_sat_support.bounded_proof,
                    "policy_conditioned": (
                        definition.cp_sat_support.policy_conditioned_proof
                    ),
                }[str(mode)]
                if proof_status == "unsupported":
                    issues.append(self._issue("predicate_proof_mode_unsupported", path))
                    continue
            if hardness == "hard" and (
                origin not in _ORIGINS
                or mode not in _PROOF_MODES
                or visibility not in _VISIBILITIES
                or not self._supports(definition, origin, mode, visibility)
            ):
                issues.append(self._issue("predicate_cp_sat_unsupported", path))
                continue
            if hardness == "hard":
                compiled_stages.update(self._stages(definition, str(origin)))

        # V3 play predicates describe a witnessed line; exact round-win proof
        # does not require that line in every continuation. The legacy theorem
        # only proves that the opponent can avoid one trick-seven continuation.
        if not issues and self.authority.backend != ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3:
            contradiction = self._structural_contradiction_issue(scenario, proof)
            if contradiction is not None:
                issues.append(contradiction)

        if self.authority.backend == ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3:
            for issue in issues:
                predicate_ids = [
                    predicate["predicateId"]
                    for path, predicate in self._predicates(scenario)
                    if isinstance(predicate.get("predicateId"), str)
                    and (issue["path"] == path or issue["path"].startswith(f"{path}."))
                ]
                if predicate_ids:
                    issue["predicateIds"] = predicate_ids

        execution = self.execution_support(raw)
        core = {
            "schemaVersion": 1,
            "backendIdentity": self.identity.to_dict(),
            "valid": not issues,
            "executionReady": not issues and bool(execution["executionReady"]),
            "compiledStages": sorted(compiled_stages),
            "issues": issues,
        }
        return {**core, "sha256": canonical_sha256(core)}

    def preflight(self, recipe: Mapping[str, Any] | Any) -> dict[str, Any]:
        return self.validate(recipe)

    def estimate(self, recipe: Mapping[str, Any] | Any) -> dict[str, Any]:
        raw = self._recipe_mapping(recipe)
        validation = self.validate(raw)
        proof = raw.get("proof") if isinstance(raw.get("proof"), Mapping) else {}
        profile_id = proof.get("budgetProfile")
        proof_profile = self._profiles(self._authority.proof_budgets).get(profile_id)
        search_profile = self._profiles(self._authority.search_budgets).get(profile_id)
        execution = self.execution_support(raw)
        launch_admissible = bool(validation["valid"] and execution["executionReady"])
        refusal_reason = None
        estimated_hidden_worlds = None
        if not validation["valid"]:
            refusal_reason = validation["issues"][0]["code"]
        elif not execution["executionReady"]:
            refusal_reason = execution["reasons"][0]
        if (
            launch_admissible and self.authority.backend == "ortools_cp_sat_v3"
            and isinstance(raw.get("playingQuality"), Mapping)
            and raw.get("scenario", {}).get("origin", {}).get("kind") in {"search", "exact_trace_prefix"}
        ):
            quality = ScenarioPlayingQualityPolicyV1.build(raw["playingQuality"])
            quality_budgets = self._profiles(self.authority.proof_budgets)
            if quality.mode == "exact" and self._rules is None:
                launch_admissible = False
                refusal_reason = "playing_quality_rule_authority_unavailable"
            elif quality.mode == "exact" and any(
                int(quality_budgets[analysis["budgetProfile"]]["maximumDepth"])
                < self._rules.structural_domain.card_count
                for analysis in quality.decisions.values()
            ):
                # Every kind includes a decision before the first play. Even
                # ignoring other decisions, its exact continuation has 32 plays.
                launch_admissible = False
                refusal_reason = "playing_quality_full_round_depth_exceeds_budget"
            elif quality.mode == "exact" and quality.visibility == "actor_information_set":
                from baloot_platform.core.dealing import DealLayoutV1
                from .ortools_information_set_compiler import (
                    OrToolsActorInformationSetCompilerV1, hidden_allocation_upper_bound_v1,
                )

                compiler = OrToolsActorInformationSetCompilerV1(self._rules, native_cp_sat=True)
                # The explicit graph lane enumerates these allocations. Research
                # search instead constructs an auction and audits its bids; the
                # native margin audit covers worlds symbolically, not by listing
                # them. Do not apply the enumeration limit to that proof path.
                required_worlds = min(
                    hidden_allocation_upper_bound_v1(tuple(map(len,
                        compiler._bid_hidden_group_positions(
                            DealLayoutV1.build(self._rules.structural_domain, dealer=dealer), actor))))
                    for dealer in Seat for actor in Seat
                )
                estimated_hidden_worlds = required_worlds
                bid_budget = quality_budgets[
                    quality.decisions["bid"]["budgetProfile"]
                ]
                quality_execution = self.authority.quality_execution
                symbolic_bid_audit = (
                    quality.decisions["bid"]["utility"] == "awarded_point_margin"
                    and bid_budget.get("nativeObserved") is not None
                    and quality_execution is not None
                    and quality_execution.escalation_opportunity_order is not None
                    and (raw["scenario"]["origin"]["kind"] == "search"
                         or raw["scenario"]["decisionPoint"]["kind"] != "bid")
                )
                if not symbolic_bid_audit and required_worlds > int(bid_budget["maximumHiddenWorlds"]):
                    launch_admissible = False
                    refusal_reason = "playing_quality_first_bid_hidden_worlds_exceed_budget"
        exact_structural_infeasibility = any(
            item["code"] == "exact_structural_infeasibility"
            for item in validation["issues"]
        )
        core = {
            "schemaVersion": 2,
            "backendIdentity": self.identity.to_dict(),
            "validationHash": validation["sha256"],
            "estimateStatus": "admissible" if launch_admissible else "refused",
            "launchAdmissible": launch_admissible,
            "refusalReason": refusal_reason,
            "compiledStages": validation["compiledStages"],
            "budgetProfile": profile_id,
            "proofBudget": copy.deepcopy(proof_profile),
            "searchBudget": copy.deepcopy(search_profile),
            "diversityDefaults": copy.deepcopy(
                dict(self._authority.diversity_defaults)
            ),
            "cacheLimits": copy.deepcopy(dict(self._authority.cache_limits)),
            "refusalThresholds": copy.deepcopy(
                dict(self._authority.refusal_thresholds)
            ),
            "estimatedModelVariables": None,
            "estimatedModelConstraints": None,
            "estimatedGraphNodes": None,
            "estimatedGraphEdges": None,
            "estimatedHiddenWorlds": estimated_hidden_worlds,
            "estimatedCountLimbs": None,
            "resourceTier": profile_id,
            "firstResultCostClass": "not_estimable",
            "requestedResultsCostClass": "not_estimable",
            "alternativeProofModes": [],
            "requestedCombinationSupported": bool(
                validation["valid"] and execution["executionReady"]
            ),
            "exhaustiveSearchInfeasibilityEstablishable": (
                exact_structural_infeasibility
            ),
        }
        return {**core, "sha256": canonical_sha256(core)}

    def _structural_contradiction_issue(
        self,
        scenario: Mapping[str, Any],
        proof: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Return an exact named contradiction from the shared recognizer."""

        if self._rules is None:
            return None
        try:
            typed_scenario = ResearchScenarioV1.build(scenario)
            typed_proof = ScenarioProofRequestV1.build(proof)
        except (KeyError, TypeError, ValueError):
            return None
        research = OrToolsFlagshipResearchV1(rules=self._rules)
        named_core = research.contradiction_core_for_scenario(
            typed_scenario, typed_proof
        )
        if named_core is None:
            return None
        theorem = research.contradiction_receipt().to_dict()
        evidence_core = {
            "schemaVersion": 1,
            "status": "exact_structural_infeasibility",
            "stage": "preflight",
            "theorem": theorem,
            "namedCorePredicateIds": list(named_core),
            "scenarioHash": typed_scenario.sha256,
            "proofRequestHash": typed_proof.sha256,
        }
        evidence = {**evidence_core, "sha256": canonical_sha256(evidence_core)}
        return {
            "code": "exact_structural_infeasibility",
            "path": "scenario.constraints",
            "detail": theorem["theoremId"],
            "structuralInfeasibility": evidence,
        }

    def execute_quality_round(
        self, rules: Any, prefix: Any, decision_point: Mapping[str, Any],
        policy: ScenarioPlayingQualityPolicyV1, *, opportunity_order: Any = None,
        maximum_seconds: float | None = None, recorded_events: Any = None,
        recorded_trace: Any = None, cancel_check: Any = None,
    ) -> dict[str, Any]:
        """Run internal quality composition from frozen limits; no admission grant."""
        execution = self.authority.quality_execution
        if self.authority.backend != ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3 or execution is None:
            raise ValueError("round quality requires a frozen native qualityExecution allowance")
        frozen_order = execution.escalation_opportunity_order
        if frozen_order is not None:
            if opportunity_order is not None and tuple(opportunity_order) != frozen_order:
                raise ValueError("round quality opportunity order disagrees with frozen configuration")
            opportunity_order = frozen_order
        elif opportunity_order is None:
            raise ValueError("round quality requires an explicit opportunity order")
        if not isinstance(policy, ScenarioPlayingQualityPolicyV1):
            raise TypeError("round quality requires a typed policy")
        policy = ScenarioPlayingQualityPolicyV1.build(policy.to_dict())
        budgets, snapshots = _quality_budget_snapshots(
            {entry["budgetProfile"] for entry in policy.decisions.values()},
            self._profiles(self.authority.proof_budgets),
            execution_budget=execution,
        )
        result = resolve_quality_round_continuation_v1(
            rules, prefix, decision_point, policy, opportunity_order=opportunity_order,
            maximum_nodes=max(item.maximum_nodes for item in budgets.values()),
            maximum_depth=max(item.maximum_depth for item in budgets.values()),
            maximum_worlds=max(item.maximum_hidden_worlds for item in budgets.values()),
            maximum_seconds=execution.maximum_wall_seconds if maximum_seconds is None else maximum_seconds,
            maximum_decision_seconds=max(item.maximum_wall_seconds for item in budgets.values()),
            maximum_escalation_decisions=execution.maximum_escalation_decisions,
            maximum_evidence_bytes=execution.maximum_evidence_bytes,
            proof_budget_profiles=snapshots, execution_budget=execution,
            search_budget_profiles=self._profiles(self.authority.search_budgets),
            recorded_events=recorded_events, recorded_trace=recorded_trace, cancel_check=cancel_check,
        )
        core = {key: value for key, value in result.items() if key not in {"sha256", "executionReceipt"}}
        if result.get("sha256") != canonical_sha256(core):
            raise ValueError("round quality producer checksum mismatch")
        validate_native_round_quality_evidence_v1(result, source_prefix=prefix,
            rules_hash=rules.sha256, policy=policy, decision_point=decision_point, rules=rules,
            search_budget_profiles=self._profiles(self.authority.search_budgets))
        core.update(sourceQualityHash=result["sha256"], backendIdentity=self.identity.to_dict(),
                    configurationAuthority=copy.deepcopy(dict(self._bundle)))
        return {**core, "sha256": canonical_sha256(core),
                **({"executionReceipt": result["executionReceipt"]} if "executionReceipt" in result else {})}

    def audit_quality_trace(
        self, rules: Any, trace: Mapping[str, Any], source_prefix: Any,
        decision_point: Mapping[str, Any], policy: ScenarioPlayingQualityPolicyV1, *,
        recorded_events: Any, maximum_seconds: float | None = None, cancel_check: Any = None,
    ) -> dict[str, Any]:
        """Attach a frozen audit without altering recorded actions or qualifying it."""
        from .scenario_contracts import ExactTracePrefixV1

        if "playingQualityEvidence" in trace:
            raise ValueError("quality trace already has immutable audit evidence")
        if recorded_events is None:
            raise ValueError("quality trace requires recorded escalation opportunity events")
        recorded = ExactTracePrefixV1.build(trace["completeExactTracePrefix"])
        result = self.execute_quality_round(rules, source_prefix, decision_point, policy,
            recorded_events=recorded_events, recorded_trace=recorded,
            maximum_seconds=maximum_seconds, cancel_check=cancel_check)
        return self._quality_trace_result(rules, trace, source_prefix, decision_point, result)

    def generate_quality_trace(
        self, rules: Any, source_prefix: Any, decision_point: Mapping[str, Any],
        policy: ScenarioPlayingQualityPolicyV1, *, maximum_seconds: float | None = None,
        cancel_check: Any = None,
    ) -> dict[str, Any]:
        """Generate a native quality-controlled line, retaining partial coverage."""
        result = self.execute_quality_round(rules, source_prefix, decision_point, policy,
            maximum_seconds=maximum_seconds, cancel_check=cancel_check)
        trace = {"completeExactTracePrefix": copy.deepcopy(result["finalPrefix"])}
        return self._quality_trace_result(rules, trace, source_prefix, decision_point, result)

    def _quality_trace_result(self, rules, trace, source_prefix, decision_point, result):
        receipt = {k: copy.deepcopy(v) for k, v in result.items()
                   if k not in {"executionReceipt", "backendIdentity", "configurationAuthority", "sourceQualityHash"}}
        receipt["sha256"] = result["sourceQualityHash"]
        attached = {**copy.deepcopy(dict(trace)), "playingQualityEvidence": {
            "sourcePrefix": source_prefix.to_dict(), "decisionPoint": copy.deepcopy(dict(decision_point)),
            "receipt": receipt}}
        validate_trace_quality_evidence_v1(attached, rules_hash=rules.sha256, rules=rules,
            required_search_profiles=self._profiles(self.authority.search_budgets))
        return {"trace": attached, "quality": result}

    def execute_campaign(
        self,
        rules: Any,
        scenario: Any,
        proof_request: Any,
        playing_quality: ScenarioPlayingQualityPolicyV1 | None = None,
        **parameters: Any,
    ) -> Any:
        """Delegate one frozen campaign to the unified CP-SAT executor."""

        generation = parameters.pop("generation", None)
        if generation is not None:
            self._validate_generation_request(generation, scenario, playing_quality)
            if parameters.get("runtime_proposal_stream") is None:
                raise ValueError("family generation requires its bounded proposal stream")
        target = getattr(self.authority.quality_execution, "family_target", None)
        if target is not None:
            from .family_catalog import validate_family_target_scenario_v1
            validate_family_target_scenario_v1(target, scenario, rules=rules,
                execution=self.authority.quality_execution, playing_quality=playing_quality)
            if parameters.get("runtime_proposal_stream") is None:
                raise ValueError("familyTarget requires its bounded proposal stream")
            if self.authority.quality_execution.family_proposal_mode is not None:
                from .ortools_tail_first_candidate import family_tail_first_plan_v1
                if generation is not None:
                    raise ValueError('familyProposalMode requires the explicit familyTarget, not generation')
                family_tail_first_plan_v1(rules, self.authority.quality_execution,
                    scenario=scenario, request=proof_request, playing_quality=playing_quality)

        _reject_v3_semantic_callback_injection(
            self.authority.backend,
            **{
                name: parameters.get(name)
                for name in (
                    "candidate_compiler",
                    "context_compiler",
                    "graph_compiler",
                )
            },
        )
        adjudication_authority = parameters.pop("adjudication_authority", None)
        runtime_proposals = parameters.pop("runtime_proposals", None)
        proposal_stream = parameters.pop("runtime_proposal_stream", None)
        if proposal_stream is not None and runtime_proposals is not None:
            raise ValueError("choose a proposal stream or a materialized queue, not both")
        executor = self._new_executor(rules, playing_quality,
            runtime_proposals=runtime_proposals, adjudication_authority=adjudication_authority,
            generation=generation)
        if proposal_stream is not None:
            parameters.pop("shards", None)
            parameters.pop("seed", None)
            parameters["proof_worker_count"] = parameters.pop("shard_worker_count")
            return executor.execute_proposal_stream(
                scenario, proof_request, proposals=proposal_stream, **parameters)
        return executor.execute_campaign(scenario, proof_request, **parameters)

    def for_run(self):
        """Share immutable authority, not proof-cache contents, with a new run."""
        backend = copy.copy(self)
        if self._bid_quality_cache is not None:
            backend._bid_quality_cache = _BidQualityCacheV1(
                self._bid_quality_cache.maximum_entries, self._bid_quality_cache.maximum_bytes)
        return backend

    def qualified_auction_provider(self, rules: Any, playing_quality: ScenarioPlayingQualityPolicyV1) -> Any:
        """Construct bids with the same frozen executor settings as the campaign."""
        if playing_quality is None:
            raise ValueError("qualified auction requires a typed playing quality policy")
        return self._new_executor(rules, playing_quality).construct_qualified_auction

    def _validate_generation_request(self, generation, scenario, playing_quality):
        from .family_catalog import ResearchFamilyCatalogV1, matching_generation_families_v1
        execution = self.authority.quality_execution
        if (self.authority.backend != ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3 or playing_quality is None
                or execution is None or execution.family_catalog is None):
            raise ValueError("generation requires a published native familyCatalog and playingQuality")
        if self._rules is None or self._rules.project_domain.declaration_mode != "mandatory":
            raise ValueError("family generation requires mandatory project declarations")
        return matching_generation_families_v1(generation, ResearchFamilyCatalogV1.build(execution.family_catalog),
                                               scenario, play_indices=execution.proposal_play_indices,
                                               maximum_auction_seeds=execution.proposal_seed_count,
                                               rules=self._rules, execution=execution, playing_quality=playing_quality)

    def _new_executor(self, rules, playing_quality, *, runtime_proposals=None, adjudication_authority=None,
                      generation=None):
        executor_parameters = {
            "bundle_revision_id": self.identity.bundle_revision_id,
            "bundle_hash": str(self._bundle["semanticBundleHash"]),
        }
        if playing_quality is not None:
            if self.authority.backend != ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3 or self.authority.quality_execution is None:
                raise ValueError("campaign quality requires frozen native qualityExecution")
            if not isinstance(playing_quality, ScenarioPlayingQualityPolicyV1):
                raise TypeError("campaign quality requires a typed policy")
            executor_parameters.update(
                quality_policy=ScenarioPlayingQualityPolicyV1.build(playing_quality.to_dict()),
                quality_execution=self.authority.quality_execution,
                quality_proof_profiles=self._profiles(self.authority.proof_budgets),
                quality_search_profiles=self._profiles(self.authority.search_budgets),
                bid_quality_cache=self._bid_quality_cache,
            )
        if self.authority.backend == ORTOOLS_SCENARIO_RESEARCH_BACKEND_V3:
            if generation is not None:
                executor_parameters["generation"] = generation
            executor_parameters["backend_id"] = self.authority.backend
            executor_parameters["tactical_detector_profile"] = self._tactical_detector_profile
            executor_parameters["rule_source_catalog"] = self._rule_source_catalog
            executor_parameters["witness_policies"] = (
                materialize_default_policy_profiles_v1(
                    self.authority.policy_catalog
                )
            )
            executor_parameters["adjudication_authority"] = (
                copy.deepcopy(dict(adjudication_authority))
                if isinstance(adjudication_authority, Mapping)
                else None
            )
        return OrToolsUnifiedResearchExecutorV1(
            rules,
            runtime_proposals=runtime_proposals,
            **executor_parameters,
        )

    def _policy_issues(
        self,
        proof: Mapping[str, Any],
    ) -> list[tuple[str, str, str | None]]:
        if proof.get("mode") != "policy_conditioned":
            return []
        raw = proof.get("policies")
        if not isinstance(raw, Mapping) or set(raw) != _POLICY_SEATS:
            return [
                (
                    "policy_profiles_required",
                    "proof.policies",
                    "one declarative policy is required for every seat",
                )
            ]
        catalog = self._authority.policy_catalog
        policy_id = str(catalog["defaultPolicyId"])
        definition = dict(dict(catalog["profiles"])[policy_id])
        version = str(definition["version"])
        materialization = str(definition["materialization"])
        issues: list[tuple[str, str, str | None]] = []
        fields = {
            "schemaVersion",
            "policyId",
            "version",
            "seed",
            "choices",
            "materialization",
            "profileHash",
        }
        for seat in sorted(_POLICY_SEATS):
            path = f"proof.policies.{seat}"
            value = raw.get(seat)
            if not isinstance(value, Mapping) or set(value) != fields:
                issues.append(("policy_profile_invalid", path, None))
                continue
            choices = value.get("choices")
            seed = value.get("seed")
            if (
                value.get("schemaVersion") != 1
                or value.get("policyId") != policy_id
                or value.get("version") != version
                or value.get("materialization") != materialization
                or type(seed) is not int
                or seed < 0
                or choices != []
            ):
                issues.append(("policy_identity_unsupported", path, None))
                continue
            core = {
                "schemaVersion": 1,
                "seat": seat,
                "policyId": policy_id,
                "version": version,
                "seed": seed,
                "choices": [],
                "materialization": materialization,
            }
            if value.get("profileHash") != canonical_sha256(core):
                issues.append(("policy_profile_hash_mismatch", path, None))
        return issues

    @staticmethod
    def _recipe_mapping(recipe: Mapping[str, Any] | Any) -> dict[str, Any]:
        value = recipe.to_dict() if hasattr(recipe, "to_dict") else recipe
        if not isinstance(value, Mapping):
            raise TypeError("recipe must be a mapping or expose to_dict()")
        return copy.deepcopy(dict(value))

    @staticmethod
    def _issue(code: str, path: str, detail: str | None = None) -> dict[str, Any]:
        issue = {"code": code, "path": path}
        if detail:
            issue["detail"] = detail
        return issue

    @staticmethod
    def _profiles(section: Mapping[str, Any]) -> dict[str, Any]:
        profiles = section.get("profiles") if isinstance(section, Mapping) else None
        return copy.deepcopy(dict(profiles)) if isinstance(profiles, Mapping) else {}

    @classmethod
    def _predicates(
        cls, scenario: Mapping[str, Any]
    ) -> tuple[tuple[str, Mapping[str, Any]], ...]:
        found: list[tuple[str, Mapping[str, Any]]] = []

        def visit(value: Any, path: str) -> None:
            if not isinstance(value, Mapping):
                return
            if {"kind", "version", "hardness", "arguments"}.issubset(value):
                found.append((path, value))
                return
            children = value.get("children")
            if isinstance(children, list):
                for index, child in enumerate(children):
                    visit(child, f"{path}.children[{index}]")

        for name in ("constraints", "outcome"):
            visit(scenario.get(name), f"scenario.{name}")
        return tuple(found)

    @staticmethod
    def _require_combination(origin: Any, mode: Any, visibility: Any) -> None:
        if origin not in _ORIGINS:
            raise ValueError("scenario origin is unsupported")
        if mode not in _PROOF_MODES:
            raise ValueError("proof mode is unsupported")
        if visibility not in _VISIBILITIES:
            raise ValueError("proof visibility is unsupported")
        if (mode, visibility) not in _PROOF_VISIBILITY_COMBINATIONS:
            raise ValueError("proof mode and visibility combination is unsupported")

    @staticmethod
    def _stages(
        definition: ScenarioPredicateDefinitionV2, origin: str
    ) -> tuple[str, ...]:
        support = definition.cp_sat_support
        return (
            support.search_stages if origin == "search" else support.exact_input_stages
        )

    def _supports(
        self,
        definition: ScenarioPredicateDefinitionV2,
        origin: str,
        proof_mode: str,
        visibility: str,
    ) -> bool:
        if self._predicate_capabilities_v3 is not None:
            capability = self._predicate_capabilities_v3.capability(definition.kind)
            if capability is None or not any(
                context.origin == origin and context.proof_mode == proof_mode
                and context.visibility == visibility
                for context in capability.execution_contexts
            ):
                return False
        support = definition.cp_sat_support
        origin_supported = (
            support.search_status in {"exact", "derived"}
            if origin == "search"
            else support.exact_input_status == "exact"
        )
        proof_status = {
            "exact": support.exact_proof,
            "bounded": support.bounded_proof,
            "policy_conditioned": support.policy_conditioned_proof,
        }[proof_mode]
        visibility_supported = (
            visibility == "complete_state"
            or support.actor_information_set
            in {"public", "observation", "nonanticipative"}
        )
        return (
            origin_supported and proof_status != "unsupported" and visibility_supported
        )
