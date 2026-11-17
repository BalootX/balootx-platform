"""Independent, bounded adjudication of finalist surrounding decisions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from baloot_platform.analysis.decision_scores import (
    PRIMARY_DECISION_COHORT_SEED_POLICY_V1,
)
from baloot_platform.plugins.base import PluginDescriptor

from .analysis_pipeline import (
    CascadedTraceAnalysisProfileV1,
    CascadedTraceAnalyzerV1,
    StagedTraceAnalyzerV1,
    TraceAnalysisProfileV1,
    TraceAnalysisResultV1,
    TraceDecisionAnalysisV3,
    _decision_seed,
    build_finalist_reanalysis_trace_profile_v1,
)
from .evidence_v3 import canonical_sha256
from .finalist_competence import (
    build_finalist_adjudication_profile_v2,
    canonical_finalist_cohort_v1,
    canonical_score_support_classification_v1,
    evaluate_primary_competence_v1,
    finalist_adjudication_target_required_v2,
    normalize_finalist_adjudication_profile_v2,
    project_finalist_adjudication_v2,
)
from .foundry import MultiPerspectiveFoundryV1
from .profiles import SelectionProfileV1
from .replay import TraceDecisionProjectionStoreV1
from .trace import CompleteGameTraceV1


def _action_hash(value: Mapping[str, Any]) -> str:
    return canonical_sha256(dict(value))


def _frozen_component_payload_v1(
    trace: CompleteGameTraceV1,
    component_name: str,
) -> dict[str, Any]:
    state = dict(trace.trusted_final_state)
    generation = state.get("generation")
    snapshot_value = state.get("configurationSnapshot")
    if not isinstance(generation, Mapping) or not isinstance(
        snapshot_value, Mapping
    ):
        raise ValueError(
            "adaptive finalist authority requires a frozen configuration snapshot"
        )
    snapshot = dict(snapshot_value)
    declared_snapshot_hash = str(snapshot.pop("sha256", ""))
    if (
        snapshot.get("schemaVersion") != 1
        or isinstance(snapshot.get("schemaVersion"), bool)
        or str(snapshot.get("bundleHash") or "") != trace.bundle_hash
        or str(generation.get("configurationSnapshotHash") or "")
        != declared_snapshot_hash
        or canonical_sha256(snapshot) != declared_snapshot_hash
    ):
        raise ValueError("adaptive finalist frozen configuration snapshot is invalid")
    components = snapshot.get("components")
    if not isinstance(components, Mapping):
        raise ValueError("adaptive finalist configuration components are missing")
    component_value = components.get(component_name)
    if not isinstance(component_value, Mapping):
        raise ValueError(
            f"adaptive finalist {component_name} configuration is missing"
        )
    component = dict(component_value)
    payload = component.get("payload")
    if (
        not isinstance(payload, Mapping)
        or canonical_sha256(payload) != str(component.get("sha256") or "")
    ):
        raise ValueError(
            f"adaptive finalist {component_name} configuration is invalid"
        )
    return dict(payload)


def _frozen_trace_analysis_profile_v1(
    challenge: Mapping[str, Any],
    *,
    stage: str,
    search_backend_id: str,
) -> TraceAnalysisProfileV1:
    confirmation = dict(challenge.get("confirmation") or {})
    screening = dict(challenge.get("screening") or {})
    if stage == "confirmation":
        bid_samples = int(confirmation.get("bid_samples", 0))
        play_samples = int(confirmation.get("deep_samples", 0))
        exact_allocation_limit = int(
            confirmation.get("exact_allocation_limit", 0)
        )
        exact_search_max_remaining = int(
            confirmation.get("exact_search_remaining", -1)
        )
        maximum_sampling_attempts = int(
            confirmation.get("maximum_sampling_attempts", 0)
        )
        exact_search_max_nodes = int(
            confirmation.get("exact_search_max_nodes", 0)
        )
        exact_search_max_seconds = float(
            confirmation.get("exact_search_max_seconds", 0.0)
        )
    elif stage == "screening":
        bid_samples = int(screening.get("shallow_samples", 0))
        play_samples = int(screening.get("shallow_samples", 0))
        exact_allocation_limit = int(
            screening.get("shallow_exact_allocation_limit", 0)
        )
        exact_search_max_remaining = int(
            screening.get("shallow_exact_search_remaining", -1)
        )
        maximum_sampling_attempts = int(
            screening.get("maximum_sampling_attempts", 0)
        )
        exact_search_max_nodes = int(screening.get("exact_search_max_nodes", 0))
        exact_search_max_seconds = float(
            screening.get("exact_search_max_seconds", 0.0)
        )
    else:
        raise ValueError("unsupported frozen trace-analysis stage")
    return TraceAnalysisProfileV1.build(
        bid_samples=bid_samples,
        play_samples=play_samples,
        exact_allocation_limit=exact_allocation_limit,
        exact_search_max_remaining=exact_search_max_remaining,
        maximum_sampling_attempts=maximum_sampling_attempts,
        exact_search_max_nodes=exact_search_max_nodes,
        exact_search_max_seconds=exact_search_max_seconds,
        exact_search_max_depth=int(
            confirmation.get("exact_search_max_depth", 0)
        ),
        exact_search_max_cache_entries=int(
            confirmation.get("exact_search_max_cache_entries", -1)
        ),
        search_backend_id=search_backend_id,
        objective=str(confirmation.get("objective") or ""),
        use_conventions=False,
    )


def _support_classification(item: TraceDecisionAnalysisV3) -> dict[str, Any]:
    evidence = item.primary_decision_score_evidence
    if evidence is None:
        return {
            "exhaustiveExactSupport": False,
            "populationCoverageComplete": False,
            "solverCompletion": "missing",
            "truthClass": "unresolved",
            "budgetExhausted": False,
            "supportClassificationValid": False,
        }
    result = canonical_score_support_classification_v1(evidence)
    valid = bool(
        item.method == evidence.method
        and item.hidden_deals == evidence.hidden_deal_support
        and float(item.effective_sample_size)
        == float(evidence.effective_sample_size)
    )
    return {**result, "supportClassificationValid": valid}


def _has_exhaustive_exact_support(item: TraceDecisionAnalysisV3) -> bool:
    return bool(_support_classification(item)["exhaustiveExactSupport"])


def _has_complete_population_heuristic_support(
    item: TraceDecisionAnalysisV3,
) -> bool:
    return bool(
        _support_classification(item)["populationCoverageComplete"]
        and not _support_classification(item)["exhaustiveExactSupport"]
    )


def _solver_completion(item: TraceDecisionAnalysisV3) -> str:
    return str(_support_classification(item)["solverCompletion"])


def _truth_class(item: TraceDecisionAnalysisV3) -> str:
    return str(_support_classification(item)["truthClass"])


@dataclass(frozen=True, slots=True)
class FinalistSurroundingAdjudicationProfileV1:
    primary_profile_hash: str
    reanalysis_profile_hash: str
    maximum_decisions_per_trace: int
    seed_offset: int
    minimum_hidden_deals: int
    minimum_confidence: float
    minimum_quality: float
    minimum_best_share: float
    maximum_competence_expected_margin_regret: float
    maximum_competence_match_win_drop: float
    maximum_expected_margin_drift: float
    maximum_cohorts: int
    minimum_agreement: int
    third_cohort_on_disagreement: bool
    require_same_recommended_action: bool
    sha256: str

    @classmethod
    def build(
        cls,
        stability: Mapping[str, Any],
        *,
        primary_profile_hash: str,
        reanalysis_profile_hash: str,
        maximum_decisions_per_trace: int,
        minimum_confidence: float,
        minimum_quality: float,
    ) -> "FinalistSurroundingAdjudicationProfileV1":
        value = build_finalist_adjudication_profile_v2(
            stability,
            primary_profile_hash=primary_profile_hash,
            reanalysis_profile_hash=reanalysis_profile_hash,
            maximum_decisions_per_trace=maximum_decisions_per_trace,
            minimum_confidence=minimum_confidence,
            minimum_quality=minimum_quality,
        )
        return cls.from_mapping(value)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "FinalistSurroundingAdjudicationProfileV1":
        core = normalize_finalist_adjudication_profile_v2(value)
        return cls(
            core["primaryProfileHash"],
            core["reanalysisProfileHash"],
            core["maximumDecisionsPerTrace"],
            core["seedOffset"],
            core["minimumHiddenDeals"],
            core["minimumConfidence"],
            core["minimumQuality"],
            core["minimumBestShare"],
            core["maximumCompetenceExpectedMarginRegret"],
            core["maximumCompetenceMatchWinDrop"],
            core["maximumExpectedMarginDrift"],
            core["maximumCohorts"],
            core["minimumAgreement"],
            core["thirdCohortOnDisagreement"],
            True,
            core["sha256"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 2,
            "primaryProfileHash": self.primary_profile_hash,
            "reanalysisProfileHash": self.reanalysis_profile_hash,
            "maximumDecisionsPerTrace": self.maximum_decisions_per_trace,
            "seedOffset": self.seed_offset,
            "minimumHiddenDeals": self.minimum_hidden_deals,
            "minimumConfidence": self.minimum_confidence,
            "minimumQuality": self.minimum_quality,
            "minimumBestShare": self.minimum_best_share,
            "maximumCompetenceExpectedMarginRegret": (
                self.maximum_competence_expected_margin_regret
            ),
            "maximumCompetenceMatchWinDrop": (
                self.maximum_competence_match_win_drop
            ),
            "maximumExpectedMarginDrift": self.maximum_expected_margin_drift,
            "maximumCohorts": self.maximum_cohorts,
            "minimumAgreement": self.minimum_agreement,
            "thirdCohortOnDisagreement": self.third_cohort_on_disagreement,
            "requireSameRecommendedAction": self.require_same_recommended_action,
            "sha256": self.sha256,
        }


def adaptive_finalist_effective_profile_hash_v1(
    *,
    primary_profile_hash: str,
    adjudication_profile_hash: str,
    selection_profile_hash: str,
) -> str:
    """Hash the exact three authorities that govern adaptive finalist analysis."""

    return canonical_sha256({
        "schemaVersion": 2,
        "primaryProfileHash": str(primary_profile_hash),
        "adjudicationProfileHash": str(adjudication_profile_hash),
        "selectionProfileHash": str(selection_profile_hash),
    })


def frozen_adaptive_finalist_profile_hash_v1(
    trace: CompleteGameTraceV1,
) -> str:
    """Rebuild the adaptive finalist authority from one immutable trace.

    The trace's generation-time deep profile selects the exact governed search
    backend.  The remaining profiles are rebuilt only from the trace's frozen
    configuration snapshot, never from the process-active bundle or host.
    """

    from .deep_analysis import DeepAnalysisProfileV1
    from .finalist_validation import FinalistValidationProfileV1

    challenge = _frozen_component_payload_v1(trace, "challenge")
    performance = _frozen_component_payload_v1(trace, "performance")
    generation = dict(trace.trusted_final_state.get("generation") or {})
    trusted_deep_hash = str(generation.get("traceAnalysisProfileHash") or "")
    search = dict(performance.get("search") or {})
    backend_ids = tuple(dict.fromkeys(
        str(search.get(key) or "").strip()
        for key in ("preferred_backend", "fallback_backend")
        if str(search.get(key) or "").strip()
    ))
    matches = tuple(
        (backend_id, profile)
        for backend_id in backend_ids
        for profile in (
            _frozen_trace_analysis_profile_v1(
                challenge,
                stage="confirmation",
                search_backend_id=backend_id,
            ),
        )
        if profile.sha256 == trusted_deep_hash
    )
    if len(matches) != 1:
        raise ValueError(
            "adaptive finalist generation deep profile is not uniquely governed"
        )
    backend_id, confirmation_profile = matches[0]
    screening_profile = _frozen_trace_analysis_profile_v1(
        challenge,
        stage="screening",
        search_backend_id=backend_id,
    )
    reanalysis_profile = build_finalist_reanalysis_trace_profile_v1(
        dict(challenge.get("stability") or {}),
        dict(challenge.get("confirmation") or {}),
        search_backend_id=backend_id,
    )
    deep_profile = DeepAnalysisProfileV1.build(
        dict(challenge.get("deep_analysis") or {})
    )
    finalist_profile = FinalistValidationProfileV1.build(
        dict(challenge.get("finalist_validation") or {})
    )
    selection_profile = SelectionProfileV1.build(
        dict(challenge.get("selection_policy") or {})
    )
    primary_profile = CascadedTraceAnalysisProfileV1.build(
        screening_profile,
        confirmation_profile,
        maximum_confirmation_decisions_per_trace=(
            finalist_profile.maximum_confirmation_decisions_per_trace
        ),
        minimum_screening_score=deep_profile.minimum_screening_score,
    )
    adjudication_profile = FinalistSurroundingAdjudicationProfileV1.build(
        dict(challenge.get("stability") or {}),
        primary_profile_hash=primary_profile.sha256,
        reanalysis_profile_hash=reanalysis_profile.sha256,
        maximum_decisions_per_trace=(
            finalist_profile.maximum_confirmation_decisions_per_trace
        ),
        minimum_confidence=selection_profile.minimum_surrounding_confidence,
        minimum_quality=selection_profile.minimum_surrounding_quality,
    )
    return adaptive_finalist_effective_profile_hash_v1(
        primary_profile_hash=primary_profile.sha256,
        adjudication_profile_hash=adjudication_profile.sha256,
        selection_profile_hash=selection_profile.sha256,
    )


@dataclass(frozen=True, slots=True)
class _EffectiveAdjudicatedProfile:
    sha256: str


class AdaptiveFinalistTraceAnalyzerV1:
    """Escalate only decisions that could invalidate a complete finalist trace."""

    descriptor = PluginDescriptor(
        id="adaptive_finalist_trace_analyzer_v1",
        kind="trace_analyzer",
        version="1.3.0",
        description=(
            "Cascaded all-seat analysis with independent bounded surrounding-decision "
            "stability adjudication"
        ),
        capabilities=(
            "all_decisions",
            "all_seats",
            "four_observer_projections",
            "independent_deterministic_cohort",
            "bounded_surrounding_adjudication",
            "conservative_surrounding_competence_band",
            "explicit_unresolved_evidence",
        ),
        resource_class="cpu",
        deterministic=True,
        timeout_seconds=900,
        maximum_memory_mb=4096,
    )

    def __init__(
        self,
        primary: CascadedTraceAnalyzerV1,
        reanalysis: StagedTraceAnalyzerV1,
        selection_profile: SelectionProfileV1,
        profile: FinalistSurroundingAdjudicationProfileV1,
    ):
        if primary.rules.sha256 != reanalysis.rules.sha256:
            raise ValueError("finalist adjudication analyzers must use identical rules")
        if primary.profile.sha256 != profile.primary_profile_hash:
            raise ValueError("finalist adjudication primary profile identity mismatch")
        if reanalysis.profile.sha256 != profile.reanalysis_profile_hash:
            raise ValueError("finalist adjudication reanalysis profile identity mismatch")
        if (
            selection_profile.minimum_surrounding_confidence
            != profile.minimum_confidence
            or selection_profile.minimum_surrounding_quality
            != profile.minimum_quality
        ):
            raise ValueError(
                "finalist adjudication selection thresholds must match the "
                "adjudication profile"
            )
        self.primary = primary
        self.reanalysis = reanalysis
        self.selection_profile = selection_profile
        self.adjudication_profile = profile
        self.rules = primary.rules
        self.critical_threshold = primary.critical_threshold
        self.minimum_critical_confidence = primary.minimum_critical_confidence
        self.profile = _EffectiveAdjudicatedProfile(
            adaptive_finalist_effective_profile_hash_v1(
                primary_profile_hash=primary.profile.sha256,
                adjudication_profile_hash=profile.sha256,
                selection_profile_hash=selection_profile.sha256,
            )
        )

    def _requires_adjudication(self, item: TraceDecisionAnalysisV3) -> bool:
        return finalist_adjudication_target_required_v2(
            item,
            adjudication_profile=self.adjudication_profile.to_dict(),
            selection_profile=self.selection_profile.to_dict(),
        )

    def _competence_cohort(
        self,
        item: TraceDecisionAnalysisV3,
        *,
        cohort_index: int,
        primary_action: Mapping[str, Any],
    ) -> dict[str, Any]:
        del primary_action  # The replay-bound action is authoritative per cohort.
        return canonical_finalist_cohort_v1(
            item.primary_decision_score_evidence,
            cohort_index=cohort_index,
            evidence_hash=item.sha256,
            actual_action=item.actual_action,
            best_action=item.best_action,
        )

    def _competence_evidence(
        self,
        cohorts: Sequence[TraceDecisionAnalysisV3],
    ) -> dict[str, Any]:
        primary_action = dict(cohorts[0].actual_action)
        return evaluate_primary_competence_v1(
            primary_action,
            tuple(
                self._competence_cohort(
                    item,
                    cohort_index=index,
                    primary_action=primary_action,
                )
                for index, item in enumerate(cohorts)
            ),
            minimum_hidden_deals=self.adjudication_profile.minimum_hidden_deals,
            minimum_confidence=self.adjudication_profile.minimum_confidence,
            minimum_best_share=self.adjudication_profile.minimum_best_share,
            maximum_expected_margin_regret=(
                self.adjudication_profile.maximum_competence_expected_margin_regret
            ),
            maximum_match_win_drop=(
                self.adjudication_profile.maximum_competence_match_win_drop
            ),
            maximum_expected_margin_drift=(
                self.adjudication_profile.maximum_expected_margin_drift
            ),
            minimum_agreement=self.adjudication_profile.minimum_agreement,
        )

    def _evaluate_cohorts(
        self,
        cohorts: Sequence[TraceDecisionAnalysisV3],
        *,
        initial_reasons: Sequence[str] = (),
    ) -> tuple[TraceDecisionAnalysisV3, dict[str, Any]]:
        if len(cohorts) < 2 or len(cohorts) > self.adjudication_profile.maximum_cohorts:
            raise ValueError("finalist adjudication cohort count is outside policy")
        primary = cohorts[0]
        if any(
            item.decision_id != primary.decision_id
            or item.ordinal != primary.ordinal
            or item.acting_seat != primary.acting_seat
            or item.decision_kind != primary.decision_kind
            or item.actual_action != primary.actual_action
            for item in cohorts[1:]
        ):
            raise ValueError("finalist adjudication decision identity mismatch")

        normalized = tuple(
            self._competence_cohort(
                item,
                cohort_index=index,
                primary_action=primary.actual_action,
            )
            for index, item in enumerate(cohorts)
        )
        score_evidence = tuple(
            item.primary_decision_score_evidence for item in cohorts
        )
        if all(item is not None for item in score_evidence):
            primary_score = score_evidence[0]
            assert primary_score is not None
            if (
                primary_score.cohort_index != 0
                or primary_score.cohort_seed_policy
                != PRIMARY_DECISION_COHORT_SEED_POLICY_V1
            ):
                raise ValueError("finalist adjudication primary cohort role mismatch")
            reanalysis_scores = tuple(score_evidence[1:])
            if any(
                item is None
                or item.cohort_index != 1
                or item.cohort_seed_policy != primary_score.cohort_seed_policy
                or item.trace_analysis_profile_hash
                != self.adjudication_profile.reanalysis_profile_hash
                for item in reanalysis_scores
            ):
                raise ValueError("finalist adjudication reanalysis cohort role mismatch")
            if any(
                item is None
                or item.public_prefix_hash != primary_score.public_prefix_hash
                or item.legal_action_set_hash != primary_score.legal_action_set_hash
                or item.rules_hash != primary_score.rules_hash
                or item.objective != primary_score.objective
                or item.tie_break != primary_score.tie_break
                or item.backend != primary_score.backend
                for item in reanalysis_scores
            ):
                raise ValueError("finalist adjudication cohort authority drift")
            reanalysis_identities = {
                item.analysis_identity_hash
                for item in reanalysis_scores
                if item is not None
            }
            if len(reanalysis_identities) != 1 or (
                primary_score.analysis_identity_hash in reanalysis_identities
            ):
                raise ValueError("finalist adjudication lane identities are invalid")
            seeds = [item.cohort_seed for item in score_evidence if item is not None]
            if len(seeds) != len(set(seeds)) or len({item.sha256 for item in cohorts}) != len(
                cohorts
            ):
                raise ValueError("finalist adjudication cohort evidence is not distinct")
        selected = primary
        projection = project_finalist_adjudication_v2(
            normalized,
            primary_best_action=primary.best_action,
            primary_actual_action=primary.actual_action,
            adjudication_profile=self.adjudication_profile.to_dict(),
            initial_reasons=initial_reasons,
        )
        serialized = [item.to_dict() for item in cohorts]
        evidence = {
            "decisionId": primary.decision_id,
            "ordinal": primary.ordinal,
            **projection,
            "cohortCount": len(cohorts),
            "cohortAnalysisHashes": [item.sha256 for item in cohorts],
            "cohortAnalyses": serialized,
            "cohortBestActionHashes": [
                _action_hash(item.best_action) for item in cohorts
            ],
            "cohortHiddenDeals": [item.hidden_deals for item in cohorts],
            "cohortConfidence": [item["confidence"] for item in normalized],
            "cohortExhaustiveExactSupport": [
                _has_exhaustive_exact_support(item) for item in cohorts
            ],
            "cohortPopulationCoverageComplete": [
                bool(
                    _has_exhaustive_exact_support(item)
                    or _has_complete_population_heuristic_support(item)
                )
                for item in cohorts
            ],
            "cohortSolverCompletion": [
                _solver_completion(item) for item in cohorts
            ],
            "cohortTruthClass": [_truth_class(item) for item in cohorts],
            # Preserve the existing two-cohort audit fields for consumers.
            "primaryAnalysisHash": cohorts[0].sha256,
            "reanalysisHash": cohorts[1].sha256,
            "primaryAnalysis": cohorts[0].to_dict(),
            "reanalysis": cohorts[1].to_dict(),
            "primaryBestActionHash": _action_hash(cohorts[0].best_action),
            "reanalysisBestActionHash": _action_hash(cohorts[1].best_action),
            "selectedAnalysisHash": selected.sha256,
            "retainedPrimaryAnalysisHash": primary.sha256,
            "primaryAnalysisRetained": selected is primary,
            "primaryHiddenDeals": cohorts[0].hidden_deals,
            "reanalysisHiddenDeals": cohorts[1].hidden_deals,
            "primaryConfidence": normalized[0]["confidence"],
            "reanalysisConfidence": normalized[1]["confidence"],
            "primaryExhaustiveExactSupport": _has_exhaustive_exact_support(cohorts[0]),
            "reanalysisExhaustiveExactSupport": _has_exhaustive_exact_support(cohorts[1]),
        }
        return selected, evidence

    def _evaluate_pair(
        self,
        primary: TraceDecisionAnalysisV3,
        reanalysis: TraceDecisionAnalysisV3,
    ) -> tuple[TraceDecisionAnalysisV3, dict[str, Any]]:
        return self._evaluate_cohorts((primary, reanalysis))

    @staticmethod
    def _validate_governed_cohort_seeds(
        trace: CompleteGameTraceV1,
        analyses: Sequence[TraceDecisionAnalysisV3],
        *,
        cohort_index: int,
        seed_namespace: str,
        seed_offset: int,
    ) -> None:
        records = {
            item.decision_id: item for item in trace.decision_ledger.records
        }
        for analysis in analyses:
            score = analysis.primary_decision_score_evidence
            if score is None:
                if not analysis.forced:
                    raise ValueError(
                        "finalist adjudication cohort is missing canonical score evidence"
                    )
                continue
            record = records.get(analysis.decision_id)
            if record is None:
                raise ValueError(
                    "finalist adjudication cohort decision is absent from the ledger"
                )
            expected_seed = _decision_seed(
                record,
                score.analysis_identity_hash,
                seed_namespace=seed_namespace,
                seed_offset=seed_offset,
            )
            if (
                score.cohort_seed_policy
                != PRIMARY_DECISION_COHORT_SEED_POLICY_V1
                or score.cohort_index != cohort_index
                or score.cohort_seed != expected_seed
            ):
                raise ValueError(
                    "finalist adjudication cohort seed is not governed by the ledger"
                )

    def _adjudicate_decisions(
        self,
        decisions: tuple[TraceDecisionAnalysisV3, ...],
        confirmations: tuple[TraceDecisionAnalysisV3, ...],
        *,
        third_confirmations: tuple[TraceDecisionAnalysisV3, ...] = (),
        overflow_ids: tuple[str, ...] = (),
        target_ids: tuple[str, ...] | None = None,
        bounded_ids: tuple[str, ...] | None = None,
        third_target_ids: tuple[str, ...] | None = None,
    ) -> tuple[tuple[TraceDecisionAnalysisV3, ...], dict[str, Any]]:
        primary_by_id = {item.decision_id: item for item in decisions}
        confirmation_by_id = {item.decision_id: item for item in confirmations}
        third_by_id = {item.decision_id: item for item in third_confirmations}
        if len(primary_by_id) != len(decisions):
            raise ValueError("finalist adjudication primary decisions are not unique")
        if len(confirmation_by_id) != len(confirmations):
            raise ValueError("finalist adjudication confirmations are not unique")
        if len(third_by_id) != len(third_confirmations):
            raise ValueError("finalist adjudication third confirmations are not unique")

        def ordered(values: Sequence[str]) -> tuple[str, ...]:
            frozen = tuple(str(item) for item in values)
            if len(frozen) != len(set(frozen)) or not set(frozen).issubset(
                primary_by_id
            ):
                raise ValueError("finalist adjudication target identity is invalid")
            return tuple(
                sorted(frozen, key=lambda item: (primary_by_id[item].ordinal, item))
            )

        bounded_order = ordered(
            tuple(confirmation_by_id) if bounded_ids is None else bounded_ids
        )
        overflow_order = ordered(overflow_ids)
        target_order = ordered(
            (*bounded_order, *overflow_order) if target_ids is None else target_ids
        )
        expected_bounded = target_order[
            : self.adjudication_profile.maximum_decisions_per_trace
        ]
        expected_overflow = target_order[
            self.adjudication_profile.maximum_decisions_per_trace :
        ]
        if bounded_order != expected_bounded or overflow_order != expected_overflow:
            raise ValueError("finalist adjudication target bound is inconsistent")
        if set(confirmation_by_id) != set(bounded_order):
            raise ValueError("finalist adjudication confirmation coverage is incomplete")
        pair_evidence_by_id = {
            decision_id: self._evaluate_cohorts((
                primary_by_id[decision_id],
                confirmation_by_id[decision_id],
            ))[1]
            for decision_id in bounded_order
        }
        third_order = ordered(
            tuple(third_by_id) if third_target_ids is None else third_target_ids
        )
        expected_third_order = (
            tuple(
                decision_id
                for decision_id in bounded_order
                if pair_evidence_by_id[decision_id]["strictReasons"]
            )
            if (
                self.adjudication_profile.maximum_cohorts >= 3
                and self.adjudication_profile.third_cohort_on_disagreement
            )
            else ()
        )
        if (
            third_order != expected_third_order
            or set(third_by_id) != set(third_order)
        ):
            raise ValueError("finalist adjudication third-cohort coverage is incomplete")
        evaluated: list[dict[str, Any]] = []
        unresolved = set(overflow_order)
        competent: set[str] = set()
        for decision_id in bounded_order:
            pair = (primary_by_id[decision_id], confirmation_by_id[decision_id])
            pair_evidence = pair_evidence_by_id[decision_id]
            cohorts = (
                (*pair, third_by_id[decision_id])
                if decision_id in third_by_id else pair
            )
            _retained_primary, evidence = self._evaluate_cohorts(
                cohorts,
                initial_reasons=pair_evidence["strictReasons"],
            )
            evaluated.append(evidence)
            if evidence["status"] == "unresolved":
                unresolved.add(decision_id)
            elif evidence["status"] == "primary_robust_competent":
                competent.add(decision_id)
        for decision_id in overflow_order:
            item = primary_by_id[decision_id]
            classification = _support_classification(item)
            normalized_primary = self._competence_cohort(
                item,
                cohort_index=0,
                primary_action=item.actual_action,
            )
            evaluated.append({
                "decisionId": decision_id,
                "ordinal": item.ordinal,
                "status": "unresolved",
                "strictRecommendationStatus": None,
                "strictPrimaryActionConfirmed": False,
                "reasons": ["adjudication_bound_exceeded"],
                "strictReasons": ["adjudication_bound_exceeded"],
                "competenceEvidence": None,
                "initialReasons": [],
                "cohortCount": 1,
                "agreementCount": 0,
                "supporterCohortIndices": [],
                "majorityActionHash": None,
                "cohortAnalysisHashes": [item.sha256],
                "cohortAnalyses": [item.to_dict()],
                "cohortBestActionHashes": [_action_hash(item.best_action)],
                "cohortHiddenDeals": [item.hidden_deals],
                "cohortConfidence": [normalized_primary["confidence"]],
                "cohortExhaustiveExactSupport": [
                    bool(classification["exhaustiveExactSupport"])
                ],
                "cohortPopulationCoverageComplete": [
                    bool(classification["populationCoverageComplete"])
                ],
                "cohortSolverCompletion": [
                    str(classification["solverCompletion"])
                ],
                "cohortTruthClass": [str(classification["truthClass"])],
                "primaryAnalysisHash": item.sha256,
                "reanalysisHash": None,
                "primaryAnalysis": item.to_dict(),
                "reanalysis": None,
                "primaryBestActionHash": _action_hash(item.best_action),
                "reanalysisBestActionHash": None,
                "selectedAnalysisHash": item.sha256,
                "retainedPrimaryAnalysisHash": item.sha256,
                "primaryAnalysisRetained": True,
                "featuredTruthEligible": False,
                "surroundingVerified": False,
                "primaryHiddenDeals": item.hidden_deals,
                "reanalysisHiddenDeals": None,
                "primaryConfidence": normalized_primary["confidence"],
                "reanalysisConfidence": None,
                "primaryExhaustiveExactSupport": bool(
                    classification["exhaustiveExactSupport"]
                ),
                "reanalysisExhaustiveExactSupport": None,
                "expectedMarginDrift": None,
            })
        # The primary replay-bound analyses are immutable truth. Reanalysis is
        # verdict evidence only and must never be substituted into the result.
        adjudicated = decisions
        if [item["decisionId"] for item in evaluated] != list(target_order):
            raise ValueError("finalist adjudication evaluation coverage is incomplete")
        unresolved_ids = tuple(sorted(unresolved))
        core = {
            "schemaVersion": 2,
            "verdictSchemaVersion": 2,
            "profileHash": self.adjudication_profile.sha256,
            "adjudicationProfile": self.adjudication_profile.to_dict(),
            "selectionProfile": self.selection_profile.to_dict(),
            "primaryProfile": self.primary.profile.to_dict(),
            "primaryProfileHash": self.primary.profile.sha256,
            "reanalysisProfileHash": self.reanalysis.profile.sha256,
            "seedNamespace": "finalist_surrounding_adjudication_v2",
            "seedOffset": self.adjudication_profile.seed_offset,
            "thirdSeedNamespace": "finalist_surrounding_adjudication_v2:cohort3",
            "thirdSeedOffset": self.adjudication_profile.seed_offset * 2,
            "targetDecisionIds": list(target_order),
            "boundedDecisionIds": list(bounded_order),
            "overflowDecisionIds": list(overflow_order),
            "thirdCohortDecisionIds": list(third_order),
            "evaluations": evaluated,
            "unresolvedDecisionIds": list(unresolved_ids),
            "competentDecisionIds": sorted(competent),
        }
        return adjudicated, {**core, "sha256": canonical_sha256(core)}

    def analyze_trace(
        self,
        trace: CompleteGameTraceV1,
        *,
        cancel_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> TraceAnalysisResultV1:
        primary = self.primary.analyze_trace(
            trace,
            cancel_check=cancel_check,
            progress_callback=(
                None
                if progress_callback is None
                else lambda fraction: progress_callback(0.70 * fraction)
            ),
        )
        self._validate_governed_cohort_seeds(
            trace,
            primary.decisions,
            cohort_index=0,
            seed_namespace="",
            seed_offset=0,
        )
        targets = tuple(
            item.decision_id
            for item in sorted(primary.decisions, key=lambda item: (item.ordinal, item.decision_id))
            if self._requires_adjudication(item)
        )
        bounded = targets[: self.adjudication_profile.maximum_decisions_per_trace]
        overflow = targets[self.adjudication_profile.maximum_decisions_per_trace :]
        confirmations = self.reanalysis.analyze_decisions(
            trace,
            bounded,
            seed_namespace="finalist_surrounding_adjudication_v2",
            seed_offset=self.adjudication_profile.seed_offset,
            cancel_check=cancel_check,
            progress_callback=(
                None
                if progress_callback is None
                else lambda fraction: progress_callback(0.70 + 0.15 * fraction)
            ),
        )
        self._validate_governed_cohort_seeds(
            trace,
            confirmations,
            cohort_index=1,
            seed_namespace="finalist_surrounding_adjudication_v2",
            seed_offset=self.adjudication_profile.seed_offset,
        )
        primary_by_id = {item.decision_id: item for item in primary.decisions}
        third_targets = (
            tuple(
                item.decision_id
                for item in confirmations
                if self._evaluate_cohorts(
                    (primary_by_id[item.decision_id], item)
                )[1]["strictReasons"]
            )
            if (
                self.adjudication_profile.maximum_cohorts >= 3
                and self.adjudication_profile.third_cohort_on_disagreement
            )
            else ()
        )
        third_confirmations: tuple[TraceDecisionAnalysisV3, ...] = ()
        if (
            third_targets
            and self.adjudication_profile.maximum_cohorts >= 3
            and self.adjudication_profile.third_cohort_on_disagreement
        ):
            third_confirmations = self.reanalysis.analyze_decisions(
                trace,
                third_targets,
                seed_namespace="finalist_surrounding_adjudication_v2:cohort3",
                seed_offset=self.adjudication_profile.seed_offset * 2,
                cancel_check=cancel_check,
                progress_callback=(
                    None
                    if progress_callback is None
                    else lambda fraction: progress_callback(0.85 + 0.15 * fraction)
                ),
            )
            self._validate_governed_cohort_seeds(
                trace,
                third_confirmations,
                cohort_index=1,
                seed_namespace="finalist_surrounding_adjudication_v2:cohort3",
                seed_offset=self.adjudication_profile.seed_offset * 2,
            )
        elif progress_callback is not None:
            progress_callback(1.0)
        decisions, evidence = self._adjudicate_decisions(
            primary.decisions,
            confirmations,
            third_confirmations=third_confirmations,
            overflow_ids=overflow,
            target_ids=targets,
            bounded_ids=bounded,
            third_target_ids=third_targets,
        )
        store = TraceDecisionProjectionStoreV1(trace, self.rules)
        laffa = MultiPerspectiveFoundryV1(
            store,
            critical_threshold=self.critical_threshold,
            minimum_critical_confidence=self.minimum_critical_confidence,
        ).analyze_laffa(
            trace.game_hash,
            {"decisions": [item.audit_mapping() for item in decisions]},
        )
        unresolved = tuple(str(item) for item in evidence["unresolvedDecisionIds"])
        core = {
            "schemaVersion": 3,
            "traceHash": trace.sha256,
            "profileHash": self.profile.sha256,
            "analyzerId": self.descriptor.id,
            "analyzerVersion": self.descriptor.version,
            "decisionAnalysisHashes": [item.sha256 for item in decisions],
            "laffaEvidenceHash": laffa.sha256,
            "confirmedDecisionIds": list(primary.confirmed_decision_ids),
            "screeningProfileHash": primary.screening_profile_hash,
            "confirmationProfileHash": primary.confirmation_profile_hash,
            "adjudicationEvidenceHash": evidence["sha256"],
            "unresolvedDecisionIds": list(unresolved),
        }
        return TraceAnalysisResultV1(
            trace.sha256,
            self.profile.sha256,
            self.descriptor.id,
            self.descriptor.version,
            decisions,
            laffa,
            canonical_sha256(core),
            primary.confirmed_decision_ids,
            primary.screening_profile_hash,
            primary.confirmation_profile_hash,
            evidence,
            unresolved,
        )
