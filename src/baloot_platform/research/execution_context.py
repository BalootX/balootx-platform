"""Immutable, bundle-specific execution contexts for Research.

Only recipe creation resolves mutable configuration rows.  Once frozen, a
Research recipe is reconstructed from its verified EffectiveRuntimeConfigV2
artifact so retries, reclaim, and process restarts cannot drift to the active
bundle.
"""

from __future__ import annotations

import copy
import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from baloot_platform.agents.profiles import AgentProfileRegistry
from baloot_platform.agents.qualification import (
    FoundryAgentReleaseV1,
    validate_finalist_analysis_compatibility,
)
from baloot_platform.analysis.belief_v7 import BeliefPolicy
from baloot_platform.analysis.evaluator import AnalysisPolicy
from baloot_platform.analysis.search_backend import (
    NativeExactSearchBackendV3,
    PythonReferenceSearchBackend,
)
from baloot_platform.challenge.analysis_pipeline import (
    CascadedTraceAnalysisProfileV1,
    CascadedTraceAnalyzerV1,
    DecisionAnalysisExecutorProfileV1,
    StagedTraceAnalyzerV1,
    TraceAnalysisProfileV1,
    build_finalist_reanalysis_trace_profile_v1,
    trace_analysis_composite_identity_hash_v1,
)
from baloot_platform.challenge.deep_analysis import DeepAnalysisProfileV1
from baloot_platform.challenge.finalist_agent_agreement import (
    FinalistAgentAgreementV1,
)
from baloot_platform.challenge.finalist_validation import (
    FinalistValidationProfileV1,
)
from baloot_platform.challenge.oracle import ActualDealOracleVerifierV1
from baloot_platform.challenge.profiles import SelectionProfileV1, ThrillProfileV1
from baloot_platform.challenge.simulation import NaturalLaffaSimulatorV1
from baloot_platform.challenge.tactical_evidence import (
    CausalTacticalDetectorV1,
    TacticalDetectorProfileV1,
)
from baloot_platform.config import ProfileBundle, profile_bundle_from_resolved
from baloot_platform.configuration import EffectiveRuntimeConfigV2
from baloot_platform.native.accelerator import NativeAccelerator, NativeUnavailable

from .contracts import (
    BalootDomainSpecV2,
    ResearchConfigurationAuthorityV1,
    ResearchRecipeV1,
    ResearchRecipeV2,
    ResearchRecipeV3,
    canonical_json,
    canonical_sha256,
)
from .domain_compiler import (
    BalootDomainCompilerV1,
    BalootDomainSpecV3,
    CompiledDomainBundleV1,
)
from .domain_artifact import DomainCompilationArtifactRepositoryV1
from .recipe_v4 import ResearchRecipeV4


_EFFECTIVE_MEDIA_TYPE = "application/vnd.balootx.effective-runtime-config-v2+json"


def _freeze_profile_value(value: Any) -> Any:
    """Recursively freeze a retained governed profile without changing it."""

    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("governed profile keys must be strings")
            frozen[key] = _freeze_profile_value(item)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_profile_value(item) for item in value)
    return value


def _descriptor_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "id": str(value["id"]),
        "version": str(value["version"]),
        "kind": str(value.get("kind") or "plugin"),
    }
    artifact = value.get("artifactSha256")
    if artifact is not None:
        identity["artifactSha256"] = str(artifact)
    return identity


def _component_identity(bundle: ProfileBundle, family: str) -> tuple[str, str, str]:
    if family == "rules":
        return bundle.rules.id, bundle.rules.version, bundle.rules.sha256
    artifact = bundle.components[family]
    return artifact.id, artifact.version, artifact.sha256


def research_analysis_identity_hash_v1(
    *,
    mode: str,
    descriptor: Mapping[str, Any],
    profile_hash: str,
    analysis_policy: AnalysisPolicy,
    belief_policy: BeliefPolicy,
    match_value_profile_hash: str,
    rules_hash: str,
    convention_hash: str,
    challenge_hash: str,
    performance_hash: str,
    search_backend_descriptor: Mapping[str, Any],
) -> str:
    """Return the frozen composite identity for one Research analyzer.

    Keep this projection byte-for-byte compatible with the V3 identity that
    previously lived inline in ``_build_context``.  It is deliberately pure so
    the identity can be frozen before an analyzer exists and then supplied as
    that analyzer's actor-observer cohort authority.
    """

    return trace_analysis_composite_identity_hash_v1(
        mode=mode,
        descriptor=descriptor,
        profile_hash=profile_hash,
        analysis_policy=analysis_policy,
        belief_policy=belief_policy,
        match_value_profile_hash=match_value_profile_hash,
        rules_hash=rules_hash,
        convention_hash=convention_hash,
        challenge_hash=challenge_hash,
        performance_hash=performance_hash,
        search_backend_descriptor=search_backend_descriptor,
    )


@dataclass(frozen=True, slots=True)
class ResearchHostSafetyCeilingV1:
    """Active host resource ceilings; selected product behavior stays frozen."""

    allowed_search_backends: frozenset[str]
    maximum_deal_attempts: int
    decision_executor_limits: Mapping[str, Mapping[str, float]]
    maximum_oracle_decisions: int
    maximum_oracle_remaining_cards: int
    maximum_search_nodes: int
    maximum_search_seconds: float
    maximum_search_depth: int
    maximum_search_cache_entries: int
    maximum_reanalysis_search_nodes: int
    maximum_reanalysis_search_seconds: float
    maximum_reanalysis_samples: int
    maximum_reanalysis_sampling_attempts: int
    maximum_reanalysis_allocation_limit: int
    maximum_reanalysis_search_remaining: int

    @classmethod
    def from_bundle(cls, bundle: ProfileBundle) -> "ResearchHostSafetyCeilingV1":
        performance = dict(bundle.component("performance"))
        search = dict(performance.get("search") or {})
        allowed = frozenset(
            str(search.get(key) or "").strip()
            for key in ("preferred_backend", "fallback_backend")
            if str(search.get(key) or "").strip()
        )
        challenge = dict(bundle.component("challenge"))
        executors = dict(challenge["decision_analysis_executor"])
        limits: dict[str, Mapping[str, float]] = {}
        for name in ("screening", "confirmation", "finalistReanalysis"):
            value = dict(executors[name])
            limits[name] = MappingProxyType(
                {
                    "workerCount": float(value["workerCount"]),
                    "maximumPendingPerWorker": float(value["maximumPendingPerWorker"]),
                    "maximumSeconds": float(value["maximumSeconds"]),
                }
            )
        deep = DeepAnalysisProfileV1.build(challenge["deep_analysis"])
        confirmation = dict(challenge["confirmation"])
        stability = dict(challenge["stability"])
        return cls(
            allowed,
            int(
                bundle.component("shared")["schedule"][
                    "deal_attempts_per_maximum_round"
                ]
            ),
            MappingProxyType(limits),
            deep.maximum_oracle_decisions_per_trace,
            deep.maximum_oracle_remaining_cards,
            int(confirmation["exact_search_max_nodes"]),
            float(confirmation["exact_search_max_seconds"]),
            int(confirmation["exact_search_max_depth"]),
            int(confirmation["exact_search_max_cache_entries"]),
            int(stability["exact_search_max_nodes"]),
            float(stability["exact_search_max_seconds"]),
            int(stability["reanalysis_samples"]),
            int(stability["maximum_sampling_attempts"]),
            int(stability["exact_allocation_limit"]),
            int(stability["exact_search_remaining"]),
        )

    def validate_bundle(self, bundle: ProfileBundle) -> None:
        challenge = dict(bundle.component("challenge"))
        selected_executors = dict(challenge["decision_analysis_executor"])
        for name, ceiling in self.decision_executor_limits.items():
            selected = dict(selected_executors[name])
            for field in (
                "workerCount",
                "maximumPendingPerWorker",
                "maximumSeconds",
            ):
                if float(selected[field]) > float(ceiling[field]):
                    raise ValueError(
                        f"selected Research {name}.{field} exceeds the active host safety ceiling"
                    )
        selected_deals = int(
            bundle.component("shared")["schedule"]["deal_attempts_per_maximum_round"]
        )
        if selected_deals > self.maximum_deal_attempts:
            raise ValueError(
                "selected Research deal attempts exceed the active host safety ceiling"
            )
        deep = DeepAnalysisProfileV1.build(challenge["deep_analysis"])
        if deep.maximum_oracle_decisions_per_trace > self.maximum_oracle_decisions:
            raise ValueError(
                "selected Research oracle decision count exceeds the active host safety ceiling"
            )
        if deep.maximum_oracle_remaining_cards > self.maximum_oracle_remaining_cards:
            raise ValueError(
                "selected Research oracle window exceeds the active host safety ceiling"
            )
        confirmation = dict(challenge["confirmation"])
        limits = (
            ("exact_search_max_nodes", int, self.maximum_search_nodes),
            ("exact_search_max_seconds", float, self.maximum_search_seconds),
            ("exact_search_max_depth", int, self.maximum_search_depth),
            (
                "exact_search_max_cache_entries",
                int,
                self.maximum_search_cache_entries,
            ),
        )
        for name, convert, ceiling in limits:
            if convert(confirmation[name]) > ceiling:
                raise ValueError(
                    f"selected Research {name} exceeds the active host safety ceiling"
                )
        stability = dict(challenge["stability"])
        for name, convert, ceiling in (
            (
                "exact_search_max_nodes",
                int,
                self.maximum_reanalysis_search_nodes,
            ),
            (
                "exact_search_max_seconds",
                float,
                self.maximum_reanalysis_search_seconds,
            ),
            (
                "reanalysis_samples",
                int,
                self.maximum_reanalysis_samples,
            ),
            (
                "maximum_sampling_attempts",
                int,
                self.maximum_reanalysis_sampling_attempts,
            ),
            (
                "exact_allocation_limit",
                int,
                self.maximum_reanalysis_allocation_limit,
            ),
            (
                "exact_search_remaining",
                int,
                self.maximum_reanalysis_search_remaining,
            ),
        ):
            if convert(stability[name]) > ceiling:
                raise ValueError(
                    f"selected Research stability.{name} exceeds the active host safety ceiling"
                )


@dataclass(frozen=True, slots=True)
class ResearchExecutionContextV1:
    effective_runtime_config_hash: str
    source_effective_hash: str
    bundle_revision_id: str
    bundle: ProfileBundle
    legacy_domain_v2: BalootDomainSpecV2
    compiled_domain: BalootDomainSpecV3
    belief_policy: BeliefPolicy
    analysis_policy: AnalysisPolicy
    match_value_profile: Mapping[str, Any]
    simulator: NaturalLaffaSimulatorV1
    analyzers: Mapping[str, Any]
    score_profile: ThrillProfileV1
    decision_score_profile: SelectionProfileV1
    tactical_verifier: ActualDealOracleVerifierV1
    maximum_tactical_decisions: int
    comparison_profile: Mapping[str, Any]
    comparison_profile_hash: str
    workflow_copy: Mapping[str, str]
    exact_search_backend: Any
    plugin_identities: tuple[Mapping[str, Any], ...]
    model_identities: tuple[Mapping[str, Any], ...]
    analysis_identity_hashes: Mapping[str, str]
    agent_identity_hash: str
    context_hash: str

    @property
    def score_profile_hash(self) -> str:
        return self.score_profile.sha256

    @property
    def decision_score_profile_hash(self) -> str:
        return self.decision_score_profile.sha256

    def authority_identities(
        self,
        *,
        component_revisions: Mapping[str, Mapping[str, Any]],
        analysis_mode: str,
    ) -> dict[str, dict[str, Any]]:
        revisions = {
            family: str(component_revisions[family]["revisionId"])
            for family in component_revisions
        }

        def identity(
            role: str,
            identity_id: str,
            version: str,
            sha256: str,
            families: tuple[str, ...],
        ) -> dict[str, Any]:
            return {
                "kind": role,
                "id": identity_id,
                "version": version,
                "sha256": sha256,
                "sourceRevisions": {
                    family: revisions[family] for family in sorted(families)
                },
            }

        rules_id, rules_version, rules_hash = _component_identity(self.bundle, "rules")
        convention_id, convention_version, convention_hash = _component_identity(
            self.bundle, "conventions"
        )
        belief_id, belief_version, belief_hash = _component_identity(
            self.bundle, "belief"
        )
        return {
            "domain": identity(
                "domain",
                "baloot_domain_spec_v3",
                "3.0.0",
                self.compiled_domain.sha256,
                ("rules", "conventions", "product"),
            ),
            "rules": identity("rules", rules_id, rules_version, rules_hash, ("rules",)),
            "convention": identity(
                "convention",
                convention_id,
                convention_version,
                convention_hash,
                ("conventions",),
            ),
            "belief": identity(
                "belief",
                belief_id,
                belief_version,
                belief_hash,
                ("belief",),
            ),
            "agent": identity(
                "agent",
                self.simulator.agent_release,
                self.bundle.components["agents"].version,
                self.agent_identity_hash,
                ("agents", "challenge"),
            ),
            "analyzer": identity(
                "analyzer",
                str(analysis_mode),
                str(self.analyzers[analysis_mode].descriptor.version),
                self.analysis_identity_hashes[analysis_mode],
                (
                    "challenge",
                    "analysis",
                    "belief",
                    "match_value",
                    "performance",
                    "rules",
                    "conventions",
                ),
            ),
            "score": identity(
                "score",
                self.score_profile.profile_id,
                self.score_profile.revision,
                self.score_profile.sha256,
                ("challenge",),
            ),
            "decisionScore": identity(
                "decisionScore",
                self.decision_score_profile.profile_id,
                self.decision_score_profile.revision,
                self.decision_score_profile.sha256,
                ("challenge",),
            ),
        }


class ResearchExecutionContextFactoryV1:
    """Build and cache immutable Research behavior from exact snapshots."""

    def __init__(
        self,
        *,
        configuration_service: Any,
        artifact_store: Any,
        research_store: Any | None = None,
        root: str | Path,
        plugin_capabilities: tuple[Mapping[str, Any], ...],
        shared_plugin_descriptors: tuple[Mapping[str, Any], ...],
        active_bundle: ProfileBundle,
        active_exact_search_backend: Any,
        cache_size: int = 8,
    ):
        if not 1 <= int(cache_size) <= 64:
            raise ValueError("Research execution context cache size must be in [1, 64]")
        self.configuration_service = configuration_service
        self.artifact_store = artifact_store
        self.domain_compilation_artifacts = (
            None
            if research_store is None
            else DomainCompilationArtifactRepositoryV1(
                store=research_store,
                artifact_store=artifact_store,
                cache_size=cache_size,
            )
        )
        self.root = Path(root).resolve()
        self.plugin_capabilities = tuple(
            copy.deepcopy(dict(item)) for item in plugin_capabilities
        )
        self.shared_plugin_descriptors = tuple(
            copy.deepcopy(dict(item)) for item in shared_plugin_descriptors
        )
        self.host_ceiling = ResearchHostSafetyCeilingV1.from_bundle(active_bundle)
        self.active_exact_search_backend = active_exact_search_backend
        self.cache_size = int(cache_size)
        self._cache: OrderedDict[str, ResearchExecutionContextV1] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0
        self._lock = threading.RLock()
        self.domain_compiler = BalootDomainCompilerV1()

    @staticmethod
    def _resolved_from_effective(
        effective: EffectiveRuntimeConfigV2,
    ) -> dict[str, Any]:
        snapshot = effective.to_dict()
        bundle = copy.deepcopy(dict(snapshot["bundle"]))
        bundle["status"] = "published"
        return {
            "bundle": bundle,
            "profiles": copy.deepcopy(dict(snapshot["normalizedValues"])),
            "profileRevisions": copy.deepcopy(dict(snapshot["componentRevisions"])),
            "effectiveHash": str(snapshot["sourceEffectiveHash"]),
            "resolutionTrace": copy.deepcopy(list(snapshot["resolutionTrace"])),
            "invalidationEpoch": int(snapshot.get("invalidationEpoch", 0)),
        }

    def _search_backend(self, bundle: ProfileBundle) -> Any:
        search = dict(bundle.component("performance").get("search") or {})
        preferred = str(search.get("preferred_backend") or "").strip()
        fallback = str(search.get("fallback_backend") or "").strip()
        required = bool(search.get("require_preferred", False))
        if preferred not in self.host_ceiling.allowed_search_backends:
            raise ValueError(
                "selected Research search backend is outside the active host allowlist"
            )
        if fallback not in self.host_ceiling.allowed_search_backends:
            raise ValueError(
                "selected Research fallback backend is outside the active host allowlist"
            )
        python_id = PythonReferenceSearchBackend.descriptor.id
        native_id = NativeExactSearchBackendV3.descriptor.id
        if preferred == python_id:
            return PythonReferenceSearchBackend()
        if preferred != native_id or fallback != python_id:
            raise ValueError("unsupported selected Research search backend policy")
        try:
            return NativeExactSearchBackendV3(NativeAccelerator(bundle.rules))
        except (NativeUnavailable, OSError):
            if required:
                raise
            return PythonReferenceSearchBackend()

    @staticmethod
    def _executor_profile(
        value: Mapping[str, Any],
    ) -> DecisionAnalysisExecutorProfileV1:
        return DecisionAnalysisExecutorProfileV1.build(
            worker_count=int(value["workerCount"]),
            maximum_pending_per_worker=int(value["maximumPendingPerWorker"]),
            maximum_seconds=float(value["maximumSeconds"]),
            poll_seconds=float(value["pollSeconds"]),
        )

    def _build_context(
        self,
        resolved: Mapping[str, Any],
        *,
        effective_runtime_config_hash: str | None,
    ) -> ResearchExecutionContextV1:
        bundle = profile_bundle_from_resolved(resolved, root=self.root)
        self.host_ceiling.validate_bundle(bundle)
        belief_policy = BeliefPolicy.from_profile(bundle.component("belief"))
        analysis_policy = AnalysisPolicy.from_profile(
            bundle.component("analysis"), belief_policy=belief_policy
        )
        exact_backend = self._search_backend(bundle)
        challenge = copy.deepcopy(dict(bundle.component("challenge")))
        decision_score_profile = SelectionProfileV1.build(
            challenge["selection_policy"]
        )
        confirmation = dict(challenge["confirmation"])
        screening = dict(challenge["screening"])
        executor_config = dict(challenge["decision_analysis_executor"])
        deep_profile = DeepAnalysisProfileV1.build(challenge["deep_analysis"])
        match_value_profile = copy.deepcopy(dict(bundle.component("match_value")))

        confirmation_profile = TraceAnalysisProfileV1.build(
            bid_samples=int(confirmation["bid_samples"]),
            play_samples=int(confirmation["deep_samples"]),
            exact_allocation_limit=int(confirmation["exact_allocation_limit"]),
            exact_search_max_remaining=int(confirmation["exact_search_remaining"]),
            maximum_sampling_attempts=int(confirmation["maximum_sampling_attempts"]),
            exact_search_max_nodes=int(confirmation["exact_search_max_nodes"]),
            exact_search_max_seconds=float(confirmation["exact_search_max_seconds"]),
            exact_search_max_depth=int(confirmation["exact_search_max_depth"]),
            exact_search_max_cache_entries=int(
                confirmation["exact_search_max_cache_entries"]
            ),
            search_backend_id=exact_backend.descriptor.id,
            objective=str(confirmation["objective"]),
            use_conventions=False,
        )
        finalist_reanalysis_profile = build_finalist_reanalysis_trace_profile_v1(
            challenge["stability"],
            confirmation,
            search_backend_id=exact_backend.descriptor.id,
            belief_policy=belief_policy,
        )
        screening_profile = TraceAnalysisProfileV1.build(
            bid_samples=int(screening["shallow_samples"]),
            play_samples=int(screening["shallow_samples"]),
            exact_allocation_limit=int(screening["shallow_exact_allocation_limit"]),
            exact_search_max_remaining=int(screening["shallow_exact_search_remaining"]),
            maximum_sampling_attempts=int(screening["maximum_sampling_attempts"]),
            exact_search_max_nodes=int(screening["exact_search_max_nodes"]),
            exact_search_max_seconds=float(screening["exact_search_max_seconds"]),
            exact_search_max_depth=int(confirmation["exact_search_max_depth"]),
            exact_search_max_cache_entries=int(
                confirmation["exact_search_max_cache_entries"]
            ),
            search_backend_id=exact_backend.descriptor.id,
            objective=str(confirmation["objective"]),
            use_conventions=False,
        )
        cascaded_profile = CascadedTraceAnalysisProfileV1.build(
            screening_profile,
            confirmation_profile,
            maximum_confirmation_decisions_per_trace=(
                deep_profile.maximum_confirmation_decisions_per_trace
            ),
            minimum_screening_score=deep_profile.minimum_screening_score,
        )
        identity_inputs = {
            "analysis_policy": analysis_policy,
            "belief_policy": belief_policy,
            "match_value_profile_hash": bundle.components["match_value"].sha256,
            "rules_hash": bundle.rules.sha256,
            "convention_hash": bundle.components["conventions"].sha256,
            "challenge_hash": bundle.components["challenge"].sha256,
            "performance_hash": bundle.components["performance"].sha256,
            "search_backend_descriptor": exact_backend.descriptor.to_dict(),
        }
        analysis_identity_hashes = {
            "screening": research_analysis_identity_hash_v1(
                mode="screening",
                descriptor=StagedTraceAnalyzerV1.descriptor.to_dict(),
                profile_hash=screening_profile.sha256,
                **identity_inputs,
            ),
            "cascaded": research_analysis_identity_hash_v1(
                mode="cascaded",
                descriptor=CascadedTraceAnalyzerV1.descriptor.to_dict(),
                profile_hash=cascaded_profile.sha256,
                **identity_inputs,
            ),
            "deep": research_analysis_identity_hash_v1(
                mode="deep",
                descriptor=StagedTraceAnalyzerV1.descriptor.to_dict(),
                profile_hash=confirmation_profile.sha256,
                **identity_inputs,
            ),
            "finalist_reanalysis": research_analysis_identity_hash_v1(
                mode="finalist_reanalysis",
                descriptor=StagedTraceAnalyzerV1.descriptor.to_dict(),
                profile_hash=finalist_reanalysis_profile.sha256,
                **identity_inputs,
            ),
        }

        critical_threshold = (
            float(challenge["publication"]["minimum_publishability"]) * 100.0
        )
        deep_analyzer = StagedTraceAnalyzerV1(
            bundle.rules,
            bundle.convention,
            confirmation_profile,
            match_value_profile=match_value_profile,
            critical_threshold=critical_threshold,
            minimum_critical_confidence=float(
                confirmation["minimum_confirmation_confidence"]
            ),
            analysis_policy=analysis_policy,
            belief_policy=belief_policy,
            exact_search_backend=exact_backend,
            executor_profile=self._executor_profile(executor_config["confirmation"]),
            cohort_authority_hash=analysis_identity_hashes["deep"],
        )
        screening_analyzer = StagedTraceAnalyzerV1(
            bundle.rules,
            bundle.convention,
            screening_profile,
            match_value_profile=match_value_profile,
            critical_threshold=critical_threshold,
            minimum_critical_confidence=0.0,
            analysis_policy=analysis_policy,
            belief_policy=belief_policy,
            exact_search_backend=exact_backend,
            executor_profile=self._executor_profile(executor_config["screening"]),
            cohort_authority_hash=analysis_identity_hashes["screening"],
        )
        cascaded_analyzer = CascadedTraceAnalyzerV1(
            screening_analyzer, deep_analyzer, cascaded_profile
        )
        analyzers = MappingProxyType(
            {
                "screening": screening_analyzer,
                "cascaded": cascaded_analyzer,
                "deep": deep_analyzer,
            }
        )

        agent_registry = AgentProfileRegistry.from_mapping(bundle.component("agents"))
        finalist_validation = FinalistValidationProfileV1.build(
            challenge["finalist_validation"]
        )
        finalist_profile = agent_registry.get(finalist_validation.agent_profile_id)
        if finalist_profile.id != "finalist_candidate":
            raise ValueError(
                "Research V3 primary policy must be the governed finalist_candidate"
            )
        validate_finalist_analysis_compatibility(finalist_profile, confirmation)
        finalist_lineup = {
            seat: finalist_profile.id for seat in ("north", "west", "south", "east")
        }
        finalist_release = FoundryAgentReleaseV1.build(
            release_id=f"{finalist_profile.id}-configured-candidate",
            profile=finalist_profile,
            lineup_snapshot=agent_registry.snapshot_lineup(finalist_lineup),
            benchmark_profile=bundle.component("benchmark"),
        )
        finalist_agreement = FinalistAgentAgreementV1(
            finalist_profile,
            challenge["stability"],
            maximum_decisions_per_trace=(
                finalist_validation.maximum_confirmation_decisions_per_trace
            ),
            minimum_confidence=decision_score_profile.minimum_surrounding_confidence,
            convention=bundle.convention,
            exact_search_backend=exact_backend,
            analysis_policy=analysis_policy,
            belief_policy=belief_policy,
            match_value_profile=match_value_profile,
            primary_trace_analysis_profile=confirmation_profile,
            primary_analysis_identity_hash=analysis_identity_hashes["deep"],
            reanalysis_trace_analysis_profile=finalist_reanalysis_profile,
            reanalysis_analysis_identity_hash=(
                analysis_identity_hashes["finalist_reanalysis"]
            ),
        )
        simulator = NaturalLaffaSimulatorV1(
            bundle.rules,
            bundle.convention,
            bundle_hash=bundle.sha256,
            configuration_snapshot=bundle.configuration_snapshot(),
            structural_domain=bundle.rules.structural_domain,
            play_domain=bundle.rules.play_domain,
            agent_profile=finalist_profile,
            agent_release=finalist_release,
            generation_tier="throughput_reference",
            maximum_deal_attempts=int(
                bundle.component("shared")["schedule"][
                    "deal_attempts_per_maximum_round"
                ]
            ),
            all_pass_dealer_policy=str(
                bundle.component("game")["all_pass"]["dealer_policy"]
            ),
            exact_search_backend=exact_backend,
            decision_agreement=finalist_agreement,
            analysis_policy=analysis_policy,
            belief_policy=belief_policy,
            match_value_profile=match_value_profile,
            trace_analysis_profile=confirmation_profile,
            analysis_identity_hash=analysis_identity_hashes["deep"],
        )
        if (
            simulator.agent_profile.use_production_model
            or simulator.agent_profile.model_weight != 0.0
        ):
            raise ValueError(
                "Research agents requiring a production model need a frozen "
                "model identity and runtime plugin"
            )

        detector = CausalTacticalDetectorV1(
            TacticalDetectorProfileV1.build(challenge["tactical_detectors"])
        )
        tactical_verifier = ActualDealOracleVerifierV1(
            bundle.rules,
            maximum_remaining_cards=deep_profile.maximum_oracle_remaining_cards,
            maximum_nodes=int(confirmation["exact_search_max_nodes"]),
            maximum_seconds=float(confirmation["exact_search_max_seconds"]),
            maximum_depth=int(confirmation["exact_search_max_depth"]),
            maximum_cache_entries=int(confirmation["exact_search_max_cache_entries"]),
            detector=detector,
        )
        score_profile = ThrillProfileV1.build(challenge["thrill_profile"])
        comparison = copy.deepcopy(dict(challenge["research_comparison"]))
        comparison_hash = canonical_sha256(comparison)
        workflow = {
            str(key): str(value)
            for key, value in dict(
                dict(bundle.component("product").get("foundry_copy") or {}).get(
                    "groups"
                )
                or {}
            )
            .get("workflow", {})
            .items()
        }

        component_plugins = [
            *self.shared_plugin_descriptors,
            simulator.descriptor.to_dict(),
            finalist_agreement.descriptor.to_dict(),
            screening_analyzer.descriptor.to_dict(),
            cascaded_analyzer.descriptor.to_dict(),
            exact_backend.descriptor.to_dict(),
            tactical_verifier.descriptor.to_dict(),
            detector.descriptor.to_dict(),
        ]
        unique_plugins: dict[tuple[str, str, str], dict[str, Any]] = {}
        for item in component_plugins:
            key = (str(item.get("kind")), str(item.get("id")), str(item.get("version")))
            unique_plugins[key] = _descriptor_identity(item)
        plugin_identities = tuple(
            sorted(
                unique_plugins.values(),
                key=lambda value: (
                    str(value["id"]),
                    str(value["version"]),
                    canonical_json(value),
                ),
            )
        )
        model_identities: tuple[Mapping[str, Any], ...] = ()
        if effective_runtime_config_hash is None:
            effective_runtime_config_hash = EffectiveRuntimeConfigV2.from_resolved(
                resolved, plugin_identities=plugin_identities,
                model_identities=model_identities,
            ).content_sha256

        # The compiled domain depends only on its declared lowering plugins.
        # The process registry may contain unrelated plugins and is therefore
        # availability inventory, never part of a frozen domain identity.
        domain_plugins = [
            copy.deepcopy(dict(item)) for item in self.shared_plugin_descriptors
        ]
        legacy_domain_v2 = BalootDomainSpecV2.build(
            rules=bundle.rules.data,
            conventions=bundle.convention,
            localization=bundle.component("product"),
            bundle_hash=bundle.sha256,
            plugins=domain_plugins,
        )
        compiled_domain = self.domain_compiler.compile(
            rules=bundle.rules.data,
            conventions=bundle.convention,
            localization=bundle.component("product"),
            bundle_hash=bundle.sha256,
            plugins=domain_plugins,
        )

        agent_identity_hash = canonical_sha256(
            {
                "schemaVersion": 3,
                "authority": "research_primary_policy_v4",
                "descriptor": simulator.descriptor.to_dict(),
                "agentRelease": simulator.agent_release,
                "agentProfile": simulator.agent_profile.to_dict(),
                "generationTier": simulator.generation_tier,
                "finalistValidationProfile": finalist_validation.to_dict(),
                "decisionAgreement": {
                    "descriptor": finalist_agreement.descriptor.to_dict(),
                    "profile": finalist_agreement.profile.to_dict(),
                    "profileHash": finalist_agreement.profile.sha256,
                    "exactPrimaryPolicy": finalist_agreement.exact_primary_policy,
                },
                "deepAnalysis": {
                    "profile": confirmation_profile.to_dict(),
                    "profileHash": confirmation_profile.sha256,
                    "analysisIdentityHash": analysis_identity_hashes["deep"],
                    "playBudget": dict(finalist_agreement.primary_analysis_budget),
                },
                "finalistReanalysis": {
                    "profile": finalist_reanalysis_profile.to_dict(),
                    "profileHash": finalist_reanalysis_profile.sha256,
                    "analysisIdentityHash": analysis_identity_hashes[
                        "finalist_reanalysis"
                    ],
                    "playBudget": dict(
                        finalist_agreement.independent_analysis_budget
                    ),
                },
            }
        )
        context_core = {
            "schemaVersion": 3,
            "authority": "research_primary_policy_v4",
            "effectiveRuntimeConfigHash": effective_runtime_config_hash,
            "semanticBundleHash": bundle.sha256,
            "domainHash": compiled_domain.sha256,
            "legacyDomainV2Hash": legacy_domain_v2.sha256,
            "compiledDomainHash": compiled_domain.sha256,
            "agentIdentityHash": agent_identity_hash,
            "analysisIdentityHashes": analysis_identity_hashes,
            "scoreProfileHash": score_profile.sha256,
            "decisionScoreProfileHash": decision_score_profile.sha256,
            "comparisonProfileHash": comparison_hash,
            "plugins": list(plugin_identities),
        }
        return ResearchExecutionContextV1(
            str(effective_runtime_config_hash),
            str(resolved["effectiveHash"]),
            str(resolved["bundle"]["revisionId"]),
            bundle,
            legacy_domain_v2,
            compiled_domain,
            belief_policy,
            analysis_policy,
            _freeze_profile_value(match_value_profile),
            simulator,
            analyzers,
            score_profile,
            decision_score_profile,
            tactical_verifier,
            deep_profile.maximum_oracle_decisions_per_trace,
            MappingProxyType(comparison),
            comparison_hash,
            MappingProxyType(workflow),
            exact_backend,
            plugin_identities,
            model_identities,
            MappingProxyType(analysis_identity_hashes),
            agent_identity_hash,
            canonical_sha256(context_core),
        )

    def _cached_context(
        self,
        resolved: Mapping[str, Any],
        effective: EffectiveRuntimeConfigV2,
        *,
        built: ResearchExecutionContextV1 | None = None,
    ) -> ResearchExecutionContextV1:
        key = effective.content_sha256
        if built is not None and (
            built.effective_runtime_config_hash != key
            or built.source_effective_hash != str(resolved["effectiveHash"])
            or built.bundle_revision_id != str(resolved["bundle"]["revisionId"])
        ):
            raise ValueError("Research prebuilt context identity mismatch")
        with self._lock:
            existing = self._cache.get(key)
            if existing is not None:
                self._cache_hits += 1
                self._cache.move_to_end(key)
                return existing
            self._cache_misses += 1
        if built is None:
            built = self._build_context(
                resolved, effective_runtime_config_hash=effective.content_sha256
            )
        domain_compilation_artifacts = getattr(
            self, "domain_compilation_artifacts", None
        )
        if domain_compilation_artifacts is not None:
            domain_compilation_artifacts.register(
                built.compiled_domain.compiled,
                bundle_hash=built.bundle.sha256,
                effective_runtime_config_hash=built.effective_runtime_config_hash,
                context_hash=built.context_hash,
                domain_hash=built.compiled_domain.sha256,
            )
        with self._lock:
            existing = self._cache.get(key)
            if existing is not None:
                self._cache.move_to_end(key)
                return existing
            self._cache[key] = built
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
                self._cache_evictions += 1
            return built

    def resolve_bundle_revision(
        self, bundle_revision_id: str, *, require_published: bool = True
    ) -> tuple[EffectiveRuntimeConfigV2, ResearchExecutionContextV1]:
        resolved = self.configuration_service.resolve(
            revision_id=str(bundle_revision_id), allow_draft=False,
            allow_retired=not require_published,
        )
        if str(resolved["bundle"]["revisionId"]) != str(bundle_revision_id):
            raise KeyError("exact Research configuration bundle revision not found")
        return self.context_from_resolved(resolved, require_published=require_published)

    def context_from_resolved(
        self, resolved: Mapping[str, Any], *, require_published: bool = True
    ) -> tuple[EffectiveRuntimeConfigV2, ResearchExecutionContextV1]:
        """Build a context from a bundle already verified by configuration resolution."""
        bundle_revision_id = str(resolved["bundle"]["revisionId"])
        if require_published and str(resolved["bundle"].get("status")) != "published":
            raise ValueError("new Research recipes require a published bundle revision")
        with self._lock:
            candidates = tuple(reversed(self._cache.values()))
        context = None
        for candidate in candidates:
            if (
                candidate.bundle_revision_id != str(bundle_revision_id)
                or candidate.source_effective_hash != str(resolved["effectiveHash"])
            ):
                continue
            effective = EffectiveRuntimeConfigV2.from_resolved(
                resolved, plugin_identities=candidate.plugin_identities,
                model_identities=candidate.model_identities,
            )
            if effective.content_sha256 != candidate.effective_runtime_config_hash:
                continue
            # Source authority and native availability can change outside the rows.
            bundle = profile_bundle_from_resolved(resolved, root=self.root)
            self.host_ceiling.validate_bundle(bundle)
            backend = self._search_backend(bundle)
            if (
                bundle.sha256 == candidate.bundle.sha256
                and _descriptor_identity(backend.descriptor.to_dict())
                == _descriptor_identity(candidate.exact_search_backend.descriptor.to_dict())
            ):
                context = candidate
                break
        if context is None:
            context = self._build_context(resolved, effective_runtime_config_hash=None)
        effective = EffectiveRuntimeConfigV2.from_resolved(
            resolved,
            plugin_identities=context.plugin_identities,
            model_identities=context.model_identities,
        )
        context = self._cached_context(resolved, effective, built=context)
        if tuple(effective.to_dict()["pluginIdentities"]) != context.plugin_identities:
            raise ValueError("Research plugin identities changed during context freeze")
        return effective, context

    def freeze_selector(
        self,
        *,
        bundle_revision_id: str,
        analysis_mode: str,
    ) -> tuple[ResearchConfigurationAuthorityV1, ResearchExecutionContextV1]:
        effective, context = self.resolve_bundle_revision(bundle_revision_id)
        if analysis_mode not in context.analyzers:
            raise ValueError("selected Research analysis mode is unsupported")
        snapshot_bytes = canonical_json(effective.to_dict()).encode("utf-8")
        artifact = self.artifact_store.put(
            snapshot_bytes,
            media_type=_EFFECTIVE_MEDIA_TYPE,
            name="effective-runtime-config-v2.json",
        )
        authority = ResearchConfigurationAuthorityV1.freeze(
            effective_runtime_config=effective,
            snapshot_artifact=artifact,
            semantic_bundle_hash=context.bundle.sha256,
            identities=context.authority_identities(
                component_revisions=effective.to_dict()["componentRevisions"],
                analysis_mode=analysis_mode,
            ),
        )
        return authority, context

    def resolve_authority(
        self,
        authority_value: Mapping[str, Any],
        *,
        analysis_mode: str | None = None,
    ) -> ResearchExecutionContextV1:
        authority = ResearchConfigurationAuthorityV1.build(authority_value)
        reference = dict(authority.snapshot_artifact)
        data = self.artifact_store.get(str(reference["uri"]))
        if len(data) != int(reference["bytes"]):
            raise ValueError("Research configuration snapshot byte count mismatch")
        if hashlib.sha256(data).hexdigest() != str(reference["sha256"]):
            raise ValueError(
                "Research configuration snapshot artifact checksum mismatch"
            )
        effective = EffectiveRuntimeConfigV2.from_bytes(data)
        if effective.content_sha256 != authority.effective_runtime_config_hash:
            raise ValueError("Research effective configuration identity mismatch")
        resolved = self._resolved_from_effective(effective)
        context = self._cached_context(resolved, effective)
        if context.bundle_revision_id != authority.bundle_revision_id:
            raise ValueError("Research configuration bundle revision mismatch")
        if context.bundle.sha256 != authority.semantic_bundle_hash:
            raise ValueError("Research semantic bundle identity mismatch")
        if context.source_effective_hash != authority.effective_config_hash:
            raise ValueError("Research source configuration identity mismatch")
        snapshot_value = effective.to_dict()
        if dict(snapshot_value["componentRevisions"]) != {
            family: dict(value)
            for family, value in authority.component_revisions.items()
        }:
            raise ValueError("Research component revision provenance mismatch")
        frozen_analysis_mode = str(authority.identities["analyzer"]["id"])
        if analysis_mode is None:
            analysis_mode = frozen_analysis_mode
        elif analysis_mode != frozen_analysis_mode:
            raise ValueError("Research analyzer selection differs from frozen authority")
        if analysis_mode not in context.analyzers:
            raise ValueError("Research frozen analyzer identity is unavailable")
        expected_identities = context.authority_identities(
            component_revisions=snapshot_value["componentRevisions"],
            analysis_mode=analysis_mode,
        )
        if expected_identities != {
            role: dict(value) for role, value in authority.identities.items()
        }:
            raise ValueError("Research execution identity mismatch")
        if tuple(snapshot_value["pluginIdentities"]) != context.plugin_identities:
            raise ValueError("Research plugin availability or identity mismatch")
        if tuple(snapshot_value["modelIdentities"]) != context.model_identities:
            raise ValueError("Research model availability or identity mismatch")
        return context

    def resolve_recipe(
        self,
        recipe: ResearchRecipeV1 | ResearchRecipeV2 | ResearchRecipeV3 | ResearchRecipeV4,
    ) -> ResearchExecutionContextV1:
        if isinstance(recipe, ResearchRecipeV4):
            return self.resolve_authority(recipe.configuration)
        analysis_mode = str(recipe.resources["analysisMode"])
        if isinstance(recipe, ResearchRecipeV3):
            return self.resolve_authority(
                recipe.configuration,
                analysis_mode=analysis_mode,
            )
        expected = dict(recipe.profiles)
        matches: list[ResearchExecutionContextV1] = []
        rows = self.configuration_service.store.list_bundles(status=None, limit=10_000)
        for row in rows:
            if str(row.status) not in {"published", "superseded"}:
                continue
            try:
                effective, context = self.resolve_bundle_revision(
                    str(row.revision_id), require_published=False
                )
            except (KeyError, PermissionError, RuntimeError, ValueError):
                continue
            projection = {
                "domainHash": context.legacy_domain_v2.sha256,
                "bundleHash": context.bundle.sha256,
                "rulesHash": context.legacy_domain_v2.rules_hash,
                "conventionHash": context.legacy_domain_v2.convention_hash,
                "agentRelease": context.simulator.agent_release,
                "analysisProfileHash": context.analyzers[analysis_mode].profile.sha256,
                "scoreProfileHash": context.score_profile.sha256,
            }
            if projection == expected:
                matches.append(context)
        unique = {item.effective_runtime_config_hash: item for item in matches}
        if len(unique) != 1:
            raise ValueError("historical_record_requires_readjudication")
        return next(iter(unique.values()))

    @property
    def cached_context_count(self) -> int:
        with self._lock:
            return len(self._cache)

    def cache_metrics_snapshot(self) -> dict[str, str | int]:
        """Return aggregate metrics without exposing configuration identities."""

        with self._lock:
            return {
                "scope": "research_execution_context",
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "evictions": self._cache_evictions,
                "entries": len(self._cache),
                "capacity": self.cache_size,
            }

    def cache_metrics_snapshots(self) -> tuple[dict[str, str | int], ...]:
        """Return the two independently measured immutable-cache scopes."""

        return (
            self.cache_metrics_snapshot(),
            self.domain_compiler.cache_metrics_snapshot(),
        )

    def compiled_domain_for_hash(
        self, compilation_hash: str
    ) -> CompiledDomainBundleV1 | None:
        expected = str(compilation_hash)
        with self._lock:
            for context in self._cache.values():
                if context.compiled_domain.compiled.sha256 == expected:
                    return context.compiled_domain.compiled
        if self.domain_compilation_artifacts is None:
            return None
        return self.domain_compilation_artifacts.get(expected)
