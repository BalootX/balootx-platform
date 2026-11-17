"""Observer-safe multi-cohort agreement for finalist agent decisions."""
from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from baloot_platform.agents.profiles import (
    AdaptiveBiddingStrategy,
    AdaptivePlayStrategy,
    AgentProfile,
    primary_bid_decision_score_authority_v1,
    primary_play_decision_score_authority_v1,
    validate_primary_bid_decision_score_trace_v1,
    validate_primary_play_decision_score_trace_v1,
)
from baloot_platform.analysis.belief_v7 import BeliefPolicy
from baloot_platform.analysis.bidding import BidChoice
from baloot_platform.analysis.decision_scores import canonical_json_bytes_v1
from baloot_platform.analysis.evaluator import AnalysisPolicy
from baloot_platform.analysis.search_errors import require_uncensored_play_trace
from baloot_platform.analysis.search_backend import (
    ExactSearchBackend,
    PythonReferenceSearchBackend,
)
from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.cards import Card
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.models import RoundState
from baloot_platform.plugins.base import PluginDescriptor

from .decision_cohort import (
    PRIMARY_DECISION_COHORT_SALT_V1,
    derive_actor_observer_cohort_seed_v1,
)
from .evidence_v3 import canonical_sha256
from .finalist_competence import (
    canonical_score_confidence_v1,
    canonical_score_metrics_by_action_v1,
    canonical_score_support_classification_v1,
    evaluate_primary_competence_v1,
    evaluate_strict_cohort_agreement_v1,
)


def _canonical_sha256_hex(value: str, *, field: str) -> str:
    digest = str(value)
    if (
        len(digest) != 64
        or digest != digest.lower()
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{field} must be a canonical lowercase SHA-256 digest")
    return digest


def validate_trace_analysis_authority_v1(
    trace_analysis_profile: Any | None,
    analysis_identity_hash: str | None,
    *,
    exact_search_backend: ExactSearchBackend | None,
) -> tuple[Any | None, str | None, str | None]:
    """Validate one optional frozen trace-analysis profile/identity pair.

    The profile is intentionally duck-typed here. Importing the challenge
    analysis pipeline would create a runtime cycle through replay/simulation.
    """

    if (trace_analysis_profile is None) != (analysis_identity_hash is None):
        raise ValueError(
            "trace analysis profile and analysis identity hash must be supplied together"
        )
    if trace_analysis_profile is None:
        return None, None, None
    profile_hash = _canonical_sha256_hex(
        getattr(trace_analysis_profile, "sha256", ""),
        field="trace analysis profile hash",
    )
    identity_hash = _canonical_sha256_hex(
        str(analysis_identity_hash or ""),
        field="analysis identity hash",
    )
    to_dict = getattr(trace_analysis_profile, "to_dict", None)
    if not callable(to_dict):
        raise ValueError("trace analysis profile must expose canonical to_dict evidence")
    serialized = dict(to_dict())
    declared_hash = str(serialized.pop("sha256", ""))
    if declared_hash != profile_hash or canonical_sha256(serialized) != profile_hash:
        raise ValueError("trace analysis profile checksum mismatch")
    required = (
        "bid_samples",
        "play_samples",
        "exact_allocation_limit",
        "exact_search_max_remaining",
        "maximum_sampling_attempts",
        "exact_search_max_nodes",
        "exact_search_max_seconds",
        "exact_search_max_depth",
        "exact_search_max_cache_entries",
        "search_backend_id",
        "objective",
        "use_conventions",
    )
    missing = tuple(name for name in required if not hasattr(trace_analysis_profile, name))
    if missing:
        raise ValueError(f"trace analysis profile is incomplete: {list(missing)}")
    backend_id = (
        PythonReferenceSearchBackend.descriptor.id
        if exact_search_backend is None
        else str(exact_search_backend.descriptor.id)
    )
    if str(trace_analysis_profile.search_backend_id) != backend_id:
        raise ValueError("trace analysis profile does not match the exact-search backend")
    if not isinstance(trace_analysis_profile.use_conventions, bool):
        raise ValueError("trace analysis convention policy must be boolean")
    return trace_analysis_profile, profile_hash, identity_hash


def validate_finalist_primary_profile_v1(
    agent_profile: AgentProfile,
    trace_analysis_profile: Any,
) -> None:
    """Fail closed unless the agent can execute the full frozen deep policy."""

    if agent_profile.play_search_mode != "paired_information_set":
        raise ValueError("frozen finalist generation requires paired information-set play")
    if agent_profile.play_search_max_remaining != 32:
        raise ValueError("frozen finalist generation requires full-line play analysis")
    if agent_profile.play_search_objective != str(trace_analysis_profile.objective):
        raise ValueError("finalist play objective does not match trace analysis")
    if agent_profile.bid_samples != int(trace_analysis_profile.bid_samples):
        raise ValueError("finalist bid samples do not match trace analysis")
    if agent_profile.skill != 1.0:
        raise ValueError("frozen finalist generation requires full configured skill")
    if agent_profile.use_production_model:
        raise ValueError("frozen finalist generation forbids an unfrozen production model")
    if agent_profile.mistake_probability != 0.0:
        raise ValueError("frozen finalist generation requires zero configured mistakes")
    if agent_profile.deception_probability != 0.0:
        raise ValueError("frozen finalist generation requires zero configured deception")


def trace_analysis_play_budget_v1(trace_analysis_profile: Any) -> dict[str, Any]:
    """Project the exact deep play budget accepted by AdaptivePlayStrategy."""

    return {
        "requested_samples": int(trace_analysis_profile.play_samples),
        "exact_allocation_limit": int(trace_analysis_profile.exact_allocation_limit),
        "exact_search_max_remaining": int(
            trace_analysis_profile.exact_search_max_remaining
        ),
        "maximum_sampling_attempts": int(
            trace_analysis_profile.maximum_sampling_attempts
        ),
        "exact_search_max_nodes": int(trace_analysis_profile.exact_search_max_nodes),
        "exact_search_max_seconds": float(
            trace_analysis_profile.exact_search_max_seconds
        ),
        "exact_search_max_depth": (
            int(trace_analysis_profile.exact_search_max_depth)
            if trace_analysis_profile.exact_search_max_depth is not None
            else None
        ),
        "exact_search_max_cache_entries": int(
            trace_analysis_profile.exact_search_max_cache_entries
        ),
    }


def _action_hash(value: Mapping[str, Any]) -> str:
    return canonical_sha256(dict(value))


@dataclass(frozen=True, slots=True)
class FinalistAgentAgreementProfileV1:
    base_agent_profile_hash: str
    independent_agent_profile_hash: str
    stability_profile_hash: str
    primary_trace_analysis_profile_hash: str | None
    primary_analysis_identity_hash: str | None
    reanalysis_trace_analysis_profile_hash: str | None
    reanalysis_analysis_identity_hash: str | None
    maximum_decisions_per_trace: int
    seed_offset: int
    minimum_hidden_deals: int
    minimum_confidence: float
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
        base_profile: AgentProfile,
        maximum_decisions_per_trace: int,
        minimum_confidence: float,
        independent_profile: AgentProfile | None = None,
        primary_trace_analysis_profile_hash: str | None = None,
        primary_analysis_identity_hash: str | None = None,
        reanalysis_trace_analysis_profile_hash: str | None = None,
        reanalysis_analysis_identity_hash: str | None = None,
    ) -> tuple["FinalistAgentAgreementProfileV1", AgentProfile]:
        if base_profile.mistake_probability != 0.0:
            raise ValueError("finalist agreement requires zero configured mistakes")
        if base_profile.deception_probability != 0.0:
            raise ValueError("finalist agreement requires zero configured deception")
        maximum = int(maximum_decisions_per_trace)
        minimum_hidden_deals = int(stability.get("minimum_hidden_deals", 0))
        confidence = float(minimum_confidence)
        minimum_best_share = float(stability.get("minimum_best_share", -1.0))
        competence_regret = float(
            stability.get("maximum_competence_expected_margin_regret", -1.0)
        )
        competence_win_drop = float(
            stability.get("maximum_competence_match_win_drop", -1.0)
        )
        maximum_margin_drift = float(
            stability.get("maximum_expected_margin_drift", -1.0)
        )
        maximum_cohorts = int(stability.get("maximum_cohorts", 0))
        minimum_agreement = int(stability.get("minimum_agreement", 0))
        third_on_disagreement = stability.get("third_cohort_on_disagreement")
        if not 1 <= maximum <= 64:
            raise ValueError("finalist agent agreement decision bound must be in [1, 64]")
        if minimum_hidden_deals < 2:
            raise ValueError("finalist agent agreement requires at least two hidden deals")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("finalist agent agreement confidence is out of range")
        if not 0.0 <= minimum_best_share <= 1.0:
            raise ValueError("finalist agent agreement best-share threshold is invalid")
        if competence_regret < 0.0:
            raise ValueError("finalist agent competence regret is invalid")
        if not 0.0 <= competence_win_drop <= 1.0:
            raise ValueError("finalist agent competence match-win drop is invalid")
        if maximum_margin_drift < 0.0:
            raise ValueError("finalist agent agreement margin drift is invalid")
        if not 2 <= maximum_cohorts <= 3:
            raise ValueError("finalist agent agreement cohort bound must be in [2, 3]")
        if not 2 <= minimum_agreement <= maximum_cohorts:
            raise ValueError("finalist agent agreement threshold is invalid")
        if not isinstance(third_on_disagreement, bool):
            raise ValueError("finalist agent third-cohort policy must be boolean")
        if maximum_cohorts == 3 and not third_on_disagreement:
            raise ValueError(
                "three-cohort finalist agreement requires adaptive third analysis"
            )
        if stability.get("require_same_recommended_action") is not True:
            raise ValueError("finalist agent agreement requires cohort recommendation agreement")
        independent = independent_profile or replace(
            base_profile,
            id=f"{base_profile.id}__stability_cohort",
            bid_samples=int(stability["reanalysis_samples"]),
            play_search_budget=int(stability["exact_search_max_nodes"]),
            play_search_samples=int(stability["reanalysis_samples"]),
            play_exact_allocation_limit=int(stability["exact_allocation_limit"]),
            play_exact_search_remaining=int(stability["exact_search_remaining"]),
            play_maximum_sampling_attempts=int(stability["maximum_sampling_attempts"]),
        )
        base_hash = canonical_sha256(base_profile.to_dict())
        independent_hash = canonical_sha256(independent.to_dict())
        stability_snapshot = {
            key: stability[key]
            for key in (
                "reanalysis_samples",
                "maximum_sampling_attempts",
                "exact_allocation_limit",
                "exact_search_remaining",
                "exact_search_max_nodes",
                "exact_search_max_seconds",
                "seed_offset",
                "maximum_cohorts",
                "minimum_agreement",
                "third_cohort_on_disagreement",
                "minimum_best_share",
                "maximum_competence_expected_margin_regret",
                "maximum_competence_match_win_drop",
                "maximum_expected_margin_drift",
                "minimum_hidden_deals",
                "require_same_recommended_action",
            )
        }
        stability_hash = canonical_sha256(stability_snapshot)
        authority_values = (
            primary_trace_analysis_profile_hash,
            primary_analysis_identity_hash,
            reanalysis_trace_analysis_profile_hash,
            reanalysis_analysis_identity_hash,
        )
        if any(item is not None for item in authority_values) and not all(
            item is not None for item in authority_values
        ):
            raise ValueError(
                "finalist primary and reanalysis authority pairs must be complete"
            )
        for field, value in zip(
            (
                "primary trace-analysis profile hash",
                "primary analysis identity hash",
                "reanalysis trace-analysis profile hash",
                "reanalysis analysis identity hash",
            ),
            authority_values,
        ):
            if value is not None:
                _canonical_sha256_hex(value, field=field)
        core = {
            "schemaVersion": 2,
            "baseAgentProfileHash": base_hash,
            "independentAgentProfileHash": independent_hash,
            "stabilityProfileHash": stability_hash,
            "primaryTraceAnalysisProfileHash": primary_trace_analysis_profile_hash,
            "primaryAnalysisIdentityHash": primary_analysis_identity_hash,
            "reanalysisTraceAnalysisProfileHash": (
                reanalysis_trace_analysis_profile_hash
            ),
            "reanalysisAnalysisIdentityHash": reanalysis_analysis_identity_hash,
            "maximumDecisionsPerTrace": maximum,
            "seedOffset": int(stability["seed_offset"]),
            "minimumHiddenDeals": minimum_hidden_deals,
            "minimumConfidence": confidence,
            "minimumBestShare": minimum_best_share,
            "maximumCompetenceExpectedMarginRegret": competence_regret,
            "maximumCompetenceMatchWinDrop": competence_win_drop,
            "maximumExpectedMarginDrift": maximum_margin_drift,
            "maximumCohorts": maximum_cohorts,
            "minimumAgreement": minimum_agreement,
            "thirdCohortOnDisagreement": third_on_disagreement,
            "requireSameRecommendedAction": True,
        }
        return (
            cls(
                base_hash,
                independent_hash,
                stability_hash,
                primary_trace_analysis_profile_hash,
                primary_analysis_identity_hash,
                reanalysis_trace_analysis_profile_hash,
                reanalysis_analysis_identity_hash,
                maximum,
                core["seedOffset"],
                minimum_hidden_deals,
                confidence,
                minimum_best_share,
                competence_regret,
                competence_win_drop,
                maximum_margin_drift,
                maximum_cohorts,
                minimum_agreement,
                bool(third_on_disagreement),
                True,
                canonical_sha256(core),
            ),
            independent,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 2,
            "baseAgentProfileHash": self.base_agent_profile_hash,
            "independentAgentProfileHash": self.independent_agent_profile_hash,
            "stabilityProfileHash": self.stability_profile_hash,
            "primaryTraceAnalysisProfileHash": (
                self.primary_trace_analysis_profile_hash
            ),
            "primaryAnalysisIdentityHash": self.primary_analysis_identity_hash,
            "reanalysisTraceAnalysisProfileHash": (
                self.reanalysis_trace_analysis_profile_hash
            ),
            "reanalysisAnalysisIdentityHash": (
                self.reanalysis_analysis_identity_hash
            ),
            "maximumDecisionsPerTrace": self.maximum_decisions_per_trace,
            "seedOffset": self.seed_offset,
            "minimumHiddenDeals": self.minimum_hidden_deals,
            "minimumConfidence": self.minimum_confidence,
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


class FinalistAgentAgreementV1:
    """Require independent observer-safe agreement before a finalist action."""

    descriptor = PluginDescriptor(
        id="finalist_agent_agreement_v1",
        kind="agent_decision_policy",
        version="1.3.0",
        description=(
            "Bounded independent-cohort agreement with explicit deterministic "
            "primary-action retention and competence provenance"
        ),
        capabilities=(
            "observer_safe",
            "equal_seat_policy",
            "independent_deterministic_cohort",
            "bounded_runtime",
            "cancellation",
            "conservative_surrounding_competence_band",
            "explicit_unresolved_primary_retention",
        ),
        resource_class="cpu",
        deterministic=True,
        timeout_seconds=900,
        maximum_memory_mb=4096,
    )

    def __init__(
        self,
        base_profile: AgentProfile,
        stability: Mapping[str, Any],
        *,
        maximum_decisions_per_trace: int,
        minimum_confidence: float,
        convention: Mapping[str, Any],
        exact_search_backend: ExactSearchBackend | None = None,
        analysis_policy: AnalysisPolicy | None = None,
        belief_policy: BeliefPolicy | None = None,
        match_value_profile: Mapping[str, Any] | None = None,
        trace_analysis_profile: Any | None = None,
        analysis_identity_hash: str | None = None,
        primary_trace_analysis_profile: Any | None = None,
        primary_analysis_identity_hash: str | None = None,
        reanalysis_trace_analysis_profile: Any | None = None,
        reanalysis_analysis_identity_hash: str | None = None,
    ):
        if primary_trace_analysis_profile is None:
            primary_trace_analysis_profile = trace_analysis_profile
        elif trace_analysis_profile is not None:
            raise ValueError("duplicate finalist primary trace-analysis profile")
        if primary_analysis_identity_hash is None:
            primary_analysis_identity_hash = analysis_identity_hash
        elif analysis_identity_hash is not None:
            raise ValueError("duplicate finalist primary analysis identity")
        (
            frozen_primary_profile,
            primary_trace_analysis_profile_hash,
            frozen_primary_identity_hash,
        ) = validate_trace_analysis_authority_v1(
            primary_trace_analysis_profile,
            primary_analysis_identity_hash,
            exact_search_backend=exact_search_backend,
        )
        (
            frozen_reanalysis_profile,
            reanalysis_trace_analysis_profile_hash,
            frozen_reanalysis_identity_hash,
        ) = validate_trace_analysis_authority_v1(
            reanalysis_trace_analysis_profile,
            reanalysis_analysis_identity_hash,
            exact_search_backend=exact_search_backend,
        )
        if (frozen_primary_profile is None) != (frozen_reanalysis_profile is None):
            raise ValueError(
                "finalist primary and reanalysis authority pairs must be complete"
            )
        exact_primary_policy = frozen_primary_profile is not None
        if frozen_primary_profile is not None:
            assert frozen_reanalysis_profile is not None
            if (
                primary_trace_analysis_profile_hash
                == reanalysis_trace_analysis_profile_hash
            ):
                raise ValueError(
                    "finalist primary and reanalysis profiles must be distinct"
                )
            if frozen_primary_identity_hash == frozen_reanalysis_identity_hash:
                raise ValueError(
                    "finalist primary and reanalysis identities must be distinct"
                )
            validate_finalist_primary_profile_v1(
                base_profile,
                frozen_primary_profile,
            )
            expected_reanalysis = {
                "bid_samples": int(stability["reanalysis_samples"]),
                "play_samples": int(stability["reanalysis_samples"]),
                "maximum_sampling_attempts": int(
                    stability["maximum_sampling_attempts"]
                ),
                "exact_allocation_limit": int(stability["exact_allocation_limit"]),
                "exact_search_max_remaining": int(
                    stability["exact_search_remaining"]
                ),
                "exact_search_max_nodes": int(stability["exact_search_max_nodes"]),
                "exact_search_max_seconds": float(
                    stability["exact_search_max_seconds"]
                ),
                "exact_search_max_depth": frozen_primary_profile.exact_search_max_depth,
                "exact_search_max_cache_entries": (
                    frozen_primary_profile.exact_search_max_cache_entries
                ),
                "search_backend_id": frozen_primary_profile.search_backend_id,
                "objective": frozen_primary_profile.objective,
                "use_conventions": frozen_primary_profile.use_conventions,
            }
            if any(
                getattr(frozen_reanalysis_profile, field) != expected
                for field, expected in expected_reanalysis.items()
            ):
                raise ValueError(
                    "finalist reanalysis profile does not match governed stability"
                )
            independent_override = replace(
                base_profile,
                id=f"{base_profile.id}__stability_cohort",
                bid_samples=int(frozen_reanalysis_profile.bid_samples),
                play_search_budget=int(frozen_reanalysis_profile.exact_search_max_nodes),
                play_search_samples=int(frozen_reanalysis_profile.play_samples),
                play_exact_allocation_limit=int(
                    frozen_reanalysis_profile.exact_allocation_limit
                ),
                play_exact_search_remaining=int(
                    frozen_reanalysis_profile.exact_search_max_remaining
                ),
                play_maximum_sampling_attempts=int(
                    frozen_reanalysis_profile.maximum_sampling_attempts
                ),
            )
        else:
            independent_override = None
        profile, independent = FinalistAgentAgreementProfileV1.build(
            stability,
            base_profile=base_profile,
            maximum_decisions_per_trace=maximum_decisions_per_trace,
            minimum_confidence=minimum_confidence,
            independent_profile=independent_override,
            primary_trace_analysis_profile_hash=(
                primary_trace_analysis_profile_hash
            ),
            primary_analysis_identity_hash=frozen_primary_identity_hash,
            reanalysis_trace_analysis_profile_hash=(
                reanalysis_trace_analysis_profile_hash
            ),
            reanalysis_analysis_identity_hash=frozen_reanalysis_identity_hash,
        )
        self.profile = profile
        self.independent_agent_profile = independent
        self.primary_trace_analysis_profile = frozen_primary_profile
        self.primary_trace_analysis_profile_hash = primary_trace_analysis_profile_hash
        self.primary_analysis_identity_hash = frozen_primary_identity_hash
        self.reanalysis_trace_analysis_profile = frozen_reanalysis_profile
        self.reanalysis_trace_analysis_profile_hash = (
            reanalysis_trace_analysis_profile_hash
        )
        self.reanalysis_analysis_identity_hash = frozen_reanalysis_identity_hash
        # Primary aliases remain the simulator contract and direct-deep parity authority.
        self.trace_analysis_profile = frozen_primary_profile
        self.trace_analysis_profile_hash = primary_trace_analysis_profile_hash
        self.analysis_identity_hash = frozen_primary_identity_hash
        self.exact_primary_policy = exact_primary_policy
        primary_budget = (
            trace_analysis_play_budget_v1(frozen_primary_profile)
            if frozen_primary_profile is not None
            else {
                "requested_samples": int(stability["reanalysis_samples"]),
                "maximum_sampling_attempts": int(stability["maximum_sampling_attempts"]),
                "exact_allocation_limit": int(stability["exact_allocation_limit"]),
                "exact_search_max_remaining": int(stability["exact_search_remaining"]),
                "exact_search_max_nodes": int(stability["exact_search_max_nodes"]),
                "exact_search_max_seconds": float(stability["exact_search_max_seconds"]),
            }
        )
        reanalysis_budget = (
            trace_analysis_play_budget_v1(frozen_reanalysis_profile)
            if frozen_reanalysis_profile is not None
            else primary_budget
        )
        self.primary_analysis_budget = MappingProxyType(dict(primary_budget))
        self.independent_analysis_budget = MappingProxyType(
            dict(reanalysis_budget)
        )
        analysis_convention = (
            dict(convention)
            if frozen_reanalysis_profile is None
            or bool(frozen_reanalysis_profile.use_conventions)
            else {}
        )
        self.independent_bidding = AdaptiveBiddingStrategy(
            independent,
            exact_search_backend=exact_search_backend,
        )
        self.independent_play = AdaptivePlayStrategy(
            independent,
            analysis_convention,
            exact_search_backend=exact_search_backend,
            analysis_policy=analysis_policy,
            belief_policy=belief_policy,
            match_value_profile=match_value_profile,
        )
        self.third_bidding = AdaptiveBiddingStrategy(
            independent,
            exact_search_backend=exact_search_backend,
        )
        self.third_play = AdaptivePlayStrategy(
            independent,
            analysis_convention,
            exact_search_backend=exact_search_backend,
            analysis_policy=analysis_policy,
            belief_policy=belief_policy,
            match_value_profile=match_value_profile,
        )

    def independent_seed(
        self,
        primary_seed: int,
        decision_kind: str,
        cohort_index: int = 2,
    ) -> int:
        if not 2 <= int(cohort_index) <= self.profile.maximum_cohorts:
            raise ValueError("finalist cohort index is outside the governed bound")
        return int(canonical_sha256({
            "policy": "finalist_independent_cohort_seed_v1",
            "agreementProfileHash": self.profile.sha256,
            "decisionKind": str(decision_kind),
            "primarySeed": int(primary_seed),
            "cohortIndex": int(cohort_index),
            "seedOffset": self.profile.seed_offset,
        })[:16], 16) + self.profile.seed_offset * int(cohort_index)

    def actor_observer_seed(
        self,
        *,
        public_prefix_hash: str,
        decision_kind: str,
        acting_seat: Any,
        cohort_index: int = 0,
    ) -> int:
        """Derive an injected-policy seed from actor-visible state only."""

        if not self.exact_primary_policy or self.primary_analysis_identity_hash is None:
            raise ValueError("actor-observer cohort seeds require frozen analysis authority")
        if not 0 <= int(cohort_index) < self.profile.maximum_cohorts:
            raise ValueError("finalist actor-observer cohort index is outside policy")
        authority_hash = (
            self.primary_analysis_identity_hash
            if int(cohort_index) == 0
            else self.reanalysis_analysis_identity_hash
        )
        assert authority_hash is not None
        return derive_actor_observer_cohort_seed_v1(
            public_prefix_hash=public_prefix_hash,
            decision_kind=decision_kind,
            acting_seat=acting_seat,
            cohort_authority_hash=authority_hash,
            cohort_index=int(cohort_index),
            governed_salt=PRIMARY_DECISION_COHORT_SALT_V1,
        )

    def _required_primary_seed(
        self,
        supplied_seed: int,
        *,
        public_prefix_hash: str | None,
        decision_kind: str,
        acting_seat: Any,
    ) -> int:
        if not self.exact_primary_policy:
            return int(supplied_seed)
        if public_prefix_hash is None:
            raise ValueError("frozen finalist agreement requires a public-prefix hash")
        expected = self.actor_observer_seed(
            public_prefix_hash=public_prefix_hash,
            decision_kind=decision_kind,
            acting_seat=acting_seat,
            cohort_index=0,
        )
        if int(supplied_seed) != expected:
            raise ValueError("primary decision seed does not match actor-observer authority")
        return expected

    def _cohort_seed(
        self,
        primary_seed: int,
        decision_kind: str,
        legacy_cohort_index: int,
        *,
        public_prefix_hash: str | None,
        acting_seat: Any,
    ) -> int:
        if not self.exact_primary_policy:
            return self.independent_seed(
                primary_seed,
                decision_kind,
                legacy_cohort_index,
            )
        assert public_prefix_hash is not None
        return self.actor_observer_seed(
            public_prefix_hash=public_prefix_hash,
            decision_kind=decision_kind,
            acting_seat=acting_seat,
            cohort_index=legacy_cohort_index - 1,
        )

    def _trace_with_analysis_identity(
        self,
        trace: Mapping[str, Any],
        *,
        cohort_index: int,
    ) -> dict[str, Any]:
        require_uncensored_play_trace(trace)
        result = dict(trace)
        if not self.exact_primary_policy:
            return result
        primary_role = int(cohort_index) == 0
        profile_hash = (
            self.primary_trace_analysis_profile_hash
            if primary_role
            else self.reanalysis_trace_analysis_profile_hash
        )
        identity_hash = (
            self.primary_analysis_identity_hash
            if primary_role
            else self.reanalysis_analysis_identity_hash
        )
        budget = (
            self.primary_analysis_budget
            if primary_role
            else self.independent_analysis_budget
        )
        assert profile_hash is not None
        assert identity_hash is not None
        for key, expected in (
            ("traceAnalysisProfileHash", profile_hash),
            ("analysisIdentityHash", identity_hash),
        ):
            actual = result.get(key)
            if actual is not None and str(actual) != expected:
                raise ValueError(f"finalist policy trace contains drifted {key}")
            result[key] = expected
        actual_budget = result.get("analysisBudget")
        if actual_budget is not None and dict(actual_budget) != dict(budget):
            raise ValueError("finalist policy trace contains drifted analysisBudget")
        result["analysisBudget"] = dict(budget)
        result["analysisAuthorityRole"] = (
            "deep_confirmation" if primary_role else "finalist_reanalysis"
        )
        return result

    def _analysis_authority_for_cohort(
        self,
        cohort_index: int,
    ) -> tuple[Any, str, str]:
        if not self.exact_primary_policy:
            raise ValueError("cohort analysis authority requires exact-primary policy")
        primary_role = int(cohort_index) == 0
        profile = (
            self.primary_trace_analysis_profile
            if primary_role
            else self.reanalysis_trace_analysis_profile
        )
        profile_hash = (
            self.primary_trace_analysis_profile_hash
            if primary_role
            else self.reanalysis_trace_analysis_profile_hash
        )
        identity_hash = (
            self.primary_analysis_identity_hash
            if primary_role
            else self.reanalysis_analysis_identity_hash
        )
        assert profile is not None
        assert profile_hash is not None
        assert identity_hash is not None
        return profile, profile_hash, identity_hash

    def _play_score_authority(
        self,
        *,
        engine: BalootEngine,
        state: RoundState,
        legal_cards: tuple[Card, ...],
        public_prefix_hash: str | None,
        cohort_index: int,
        cohort_seed: int,
    ) -> dict[str, Any] | None:
        if not self.exact_primary_policy:
            return None
        if public_prefix_hash is None or state.current_player is None:
            raise ValueError(
                "frozen finalist play evidence requires an actor-visible public prefix"
            )
        profile, profile_hash, identity_hash = self._analysis_authority_for_cohort(
            cohort_index
        )
        return primary_play_decision_score_authority_v1(
            actor=state.current_player,
            public_prefix_hash=public_prefix_hash,
            legal_cards=legal_cards,
            rules_hash=engine.rules.sha256,
            analysis_identity_hash=identity_hash,
            trace_analysis_profile_hash=profile_hash,
            cohort_index=cohort_index,
            cohort_seed=cohort_seed,
            objective=str(profile.objective),
            backend=str(profile.search_backend_id),
        )

    def _bid_score_authority(
        self,
        *,
        state: RoundState,
        machine: BiddingMachine,
        rules: Any,
        legal_choices: tuple[BidChoice, ...],
        public_prefix_hash: str | None,
        cohort_index: int,
        cohort_seed: int,
    ) -> dict[str, Any] | None:
        if not self.exact_primary_policy:
            return None
        if public_prefix_hash is None or machine.actor is None:
            raise ValueError(
                "frozen finalist bid evidence requires an actor-visible public prefix"
            )
        profile, profile_hash, identity_hash = self._analysis_authority_for_cohort(
            cohort_index
        )
        return primary_bid_decision_score_authority_v1(
            actor=machine.actor,
            public_prefix_hash=public_prefix_hash,
            legal_choices=legal_choices,
            rules_hash=str(rules.sha256),
            analysis_identity_hash=identity_hash,
            trace_analysis_profile_hash=profile_hash,
            cohort_index=cohort_index,
            cohort_seed=cohort_seed,
            backend=str(profile.search_backend_id),
        )

    @staticmethod
    def _support_classification(trace: Mapping[str, Any]) -> dict[str, Any]:
        score_evidence = trace.get("primaryDecisionScoreEvidence")
        if not isinstance(score_evidence, Mapping):
            return {
                "exhaustiveExactSupport": False,
                "populationCoverageComplete": False,
                "solverCompletion": "missing",
                "truthClass": "unresolved",
                "budgetExhausted": False,
                "supportClassificationValid": False,
            }
        classification = canonical_score_support_classification_v1(
            score_evidence
        )
        declared = {
            "analysisExhaustiveSupport": "exhaustiveExactSupport",
            "analysisPopulationCoverageComplete": "populationCoverageComplete",
            "analysisSolverCompletion": "solverCompletion",
            "analysisTruthClass": "truthClass",
            "analysisCompletionStatuses": "completionStatuses",
        }
        valid = True
        for trace_key, class_key in declared.items():
            if trace_key not in trace:
                continue
            actual = trace.get(trace_key)
            expected = classification[class_key]
            if class_key == "completionStatuses":
                actual = sorted(str(item) for item in (actual or ()))
            if actual != expected:
                valid = False
        if trace.get("analysisHiddenDeals") != score_evidence.get(
            "hiddenDealSupport"
        ):
            valid = False
        return {
            **classification,
            "supportClassificationValid": valid,
        }

    @staticmethod
    def _ranked_metrics(
        trace: Mapping[str, Any],
    ) -> dict[str, dict[str, Any]]:
        score_evidence = trace.get("primaryDecisionScoreEvidence")
        if isinstance(score_evidence, Mapping):
            if (
                trace.get("primaryDecisionScoreEvidenceHash")
                != score_evidence.get("sha256")
            ):
                raise ValueError("finalist trace score-evidence hash drifted")
            if _action_hash(dict(score_evidence.get("recommendedAction") or {})) != (
                _action_hash(dict(trace.get("intendedBestAction") or {}))
            ):
                raise ValueError(
                    "finalist trace recommendation drifts from canonical score rows"
                )
            return canonical_score_metrics_by_action_v1(score_evidence)
        # Legacy traces remain readable for provenance, but duplicated ranking
        # rows are not a current finalist score authority and cannot qualify.
        return {}

    def _resolve_cohorts(
        self,
        traces: Sequence[Mapping[str, Any]],
    ) -> tuple[Mapping[str, Any] | None, list[str], int, str | None]:
        primary_action = dict(traces[0]["selectedAction"])
        normalized = []
        for index, trace in enumerate(traces):
            cohort = self._competence_cohort(
                trace,
                cohort_index=index,
                primary_action=primary_action,
            )
            cohort["canonicalScoreEvidencePresent"] = isinstance(
                trace.get("primaryDecisionScoreEvidence"), Mapping
            )
            normalized.append(cohort)
        evidence = evaluate_strict_cohort_agreement_v1(
            normalized,
            minimum_hidden_deals=self.profile.minimum_hidden_deals,
            minimum_confidence=self.profile.minimum_confidence,
            minimum_best_share=self.profile.minimum_best_share,
            maximum_expected_margin_drift=(
                self.profile.maximum_expected_margin_drift
            ),
            minimum_agreement=self.profile.minimum_agreement,
        )
        selected = evidence["selectedAction"]
        return (
            dict(selected) if isinstance(selected, Mapping) else None,
            list(evidence["strictReasons"]),
            int(evidence["agreementCount"]),
            evidence["majorityActionHash"],
        )

    @staticmethod
    def _primary_retention_strict_reasons(
        reasons: Sequence[str],
        *,
        selected: Mapping[str, Any] | None,
        primary_best_action: Mapping[str, Any],
    ) -> list[str]:
        projected = set(str(item) for item in reasons)
        if selected is not None and _action_hash(selected) != _action_hash(
            primary_best_action
        ):
            projected.add("primary_recommendation_not_confirmed")
        return sorted(projected)

    def _competence_cohort(
        self,
        trace: Mapping[str, Any],
        *,
        cohort_index: int,
        primary_action: Mapping[str, Any],
    ) -> dict[str, Any]:
        metrics = self._ranked_metrics(trace)
        best_action = dict(trace["intendedBestAction"])
        best = dict(metrics.get(_action_hash(best_action)) or {})
        primary = dict(metrics.get(_action_hash(primary_action)) or {})
        value_interval = primary.pop("confidence95", None)
        best.pop("confidence95", None)
        classification = self._support_classification(trace)
        score_evidence = trace.get("primaryDecisionScoreEvidence")
        confidence = 0.0
        if isinstance(score_evidence, Mapping):
            confidence = canonical_score_confidence_v1(score_evidence)
            if canonical_json_bytes_v1(trace.get("confidence")) != (
                canonical_json_bytes_v1(confidence)
            ):
                raise ValueError(
                    "finalist trace confidence drifts from canonical score rows"
                )
        return {
            "cohortIndex": int(cohort_index),
            "evidenceHash": trace.get("primaryDecisionScoreEvidenceHash")
            or trace.get("decisionScoreRowsHash"),
            "traceSeed": trace.get("seed"),
            "bestAction": best_action,
            "primaryAction": dict(primary_action),
            "bestMetrics": best,
            "primaryMetrics": primary,
            "valueConfidence95": value_interval,
            "hiddenDeals": (
                score_evidence.get("hiddenDealSupport")
                if isinstance(score_evidence, Mapping)
                else None
            ),
            "confidence": confidence,
            "bestShare": best.get("bestShare", 0.0),
            "canonicalScoreEvidencePresent": isinstance(
                score_evidence, Mapping
            ),
            **classification,
            "catastrophicCrossing": False,
        }

    def _resolve_primary_competence(
        self,
        primary_action: Mapping[str, Any],
        traces: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        return evaluate_primary_competence_v1(
            primary_action,
            tuple(
                self._competence_cohort(
                    trace,
                    cohort_index=index,
                    primary_action=primary_action,
                )
                for index, trace in enumerate(traces)
            ),
            minimum_hidden_deals=self.profile.minimum_hidden_deals,
            minimum_confidence=self.profile.minimum_confidence,
            minimum_best_share=self.profile.minimum_best_share,
            maximum_expected_margin_regret=(
                self.profile.maximum_competence_expected_margin_regret
            ),
            maximum_match_win_drop=self.profile.maximum_competence_match_win_drop,
            maximum_expected_margin_drift=self.profile.maximum_expected_margin_drift,
            minimum_agreement=self.profile.minimum_agreement,
        )

    @staticmethod
    def _competence_blocking_reasons(
        evidence: Mapping[str, Any],
    ) -> list[str]:
        reasons = {
            str(item) for item in evidence.get("resolutionReasons") or ()
        }
        for evaluation in evidence.get("cohortEvaluations") or ():
            reasons.update(str(item) for item in evaluation.get("reasons") or ())
        return sorted(reasons)

    def _combined_trace(
        self,
        *,
        decision_kind: str,
        primary_seed: int,
        independent_seed: int | None,
        third_seed: int | None,
        primary: Mapping[str, Any],
        independent: Mapping[str, Any] | None,
        third: Mapping[str, Any] | None,
        selected: Mapping[str, Any],
        reasons: Sequence[str],
        agreement_count: int = 0,
        majority_action_hash: str | None = None,
        initial_reasons: Sequence[str] = (),
        strict_reasons: Sequence[str] = (),
        competence_evidence: Mapping[str, Any] | None = None,
        bound_exceeded: bool = False,
    ) -> dict[str, Any]:
        unresolved = bool(reasons)
        cohort_traces = tuple(
            item for item in (primary, independent, third) if item is not None
        )
        competence_resolved = bool(
            competence_evidence is not None
            and competence_evidence.get("status") == "primary_robust_competent"
        )
        status = (
            "unresolved"
            if unresolved
            else (
                "primary_robust_competent"
                if competence_resolved
                else "strict_confirmed"
            )
        )
        result = {
            "schemaVersion": 2,
            "verdictSchemaVersion": 2,
            "source": "finalist_multi_cohort_agent",
            "decisionKind": decision_kind,
            "agreementProfile": self.profile.to_dict(),
            "primarySeed": int(primary_seed),
            "independentSeed": independent_seed,
            "thirdSeed": third_seed,
            "primaryTrace": dict(primary),
            "independentTrace": dict(independent) if independent is not None else None,
            "thirdTrace": dict(third) if third is not None else None,
            "cohorts": [dict(item) for item in cohort_traces],
            "cohortCount": len(cohort_traces),
            "populationCoverageCompleteByCohort": [
                bool(
                    self._support_classification(item)[
                        "populationCoverageComplete"
                    ]
                )
                for item in cohort_traces
            ],
            "solverCompletionByCohort": [
                str(self._support_classification(item)["solverCompletion"])
                for item in cohort_traces
            ],
            "truthClassByCohort": [
                str(self._support_classification(item)["truthClass"])
                for item in cohort_traces
            ],
            "agreementCount": int(agreement_count),
            "majorityActionHash": majority_action_hash,
            "status": status,
            "reasons": sorted(set(str(item) for item in reasons)),
            "initialReasons": sorted(set(str(item) for item in initial_reasons)),
            "strictReasons": sorted(set(str(item) for item in strict_reasons)),
            "competenceEvidence": (
                dict(competence_evidence)
                if competence_evidence is not None else None
            ),
            "selectedAction": dict(selected),
            "fallbackPolicy": None,
            "agreementBoundExceeded": bool(bound_exceeded),
            "qualificationBlocked": status == "unresolved",
            "surroundingVerified": status != "unresolved",
            "featuredTruthEligible": status == "strict_confirmed",
            "observerSafe": True,
        }
        if self.exact_primary_policy:
            assert self.trace_analysis_profile_hash is not None
            assert self.analysis_identity_hash is not None
            result.update({
                "selectionMode": "primary_policy_only",
                "primaryActionPreserved": True,
                "traceAnalysisProfileHash": self.trace_analysis_profile_hash,
                "analysisIdentityHash": self.analysis_identity_hash,
                "analysisBudget": dict(self.primary_analysis_budget),
                "analysisAuthorityRole": "deep_confirmation",
                "primaryActionHash": _action_hash(dict(selected)),
                "analysisAuthorities": {
                    "primary": {
                        "role": "deep_confirmation",
                        "traceAnalysisProfileHash": (
                            self.primary_trace_analysis_profile_hash
                        ),
                        "analysisIdentityHash": self.primary_analysis_identity_hash,
                        "analysisBudget": dict(self.primary_analysis_budget),
                    },
                    "reanalysis": {
                        "role": "finalist_reanalysis",
                        "traceAnalysisProfileHash": (
                            self.reanalysis_trace_analysis_profile_hash
                        ),
                        "analysisIdentityHash": self.reanalysis_analysis_identity_hash,
                        "analysisBudget": dict(self.independent_analysis_budget),
                    },
                },
            })
            if decision_kind in {"bid", "play"}:
                evidence = primary.get("primaryDecisionScoreEvidence")
                if not isinstance(evidence, Mapping):
                    raise ValueError(
                        "combined finalist trace lost primary score evidence"
                    )
                result.update({
                    "primaryDecisionScoreEvidence": dict(evidence),
                    "primaryDecisionScoreEvidenceHash": primary.get(
                        "primaryDecisionScoreEvidenceHash"
                    ),
                    "decisionScoreRowsHash": primary.get("decisionScoreRowsHash"),
                    "legalActionSetHash": primary.get("legalActionSetHash"),
                })
        return result

    def choose_bid(
        self,
        primary_strategy: AdaptiveBiddingStrategy,
        state: RoundState,
        machine: BiddingMachine,
        rules: Any,
        legal_choices: Sequence[BidChoice],
        *,
        seed: int,
        agreement_ordinal: int,
        cancel_check: Callable[[], bool] | None = None,
        public_prefix_hash: str | None = None,
    ) -> tuple[str, Any | None, dict[str, Any]]:
        acting_seat = getattr(machine, "actor", None)
        primary_seed = self._required_primary_seed(
            seed,
            public_prefix_hash=public_prefix_hash,
            decision_kind="bid",
            acting_seat=acting_seat,
        )
        legal = tuple(legal_choices)
        primary_authority = self._bid_score_authority(
            state=state,
            machine=machine,
            rules=rules,
            legal_choices=legal,
            public_prefix_hash=public_prefix_hash,
            cohort_index=0,
            cohort_seed=primary_seed,
        )
        action, trump, primary = primary_strategy.choose_with_trace(
            state,
            machine,
            rules,
            seed=primary_seed,
            cancel_check=cancel_check,
            **(
                {"primary_decision_score_authority": primary_authority}
                if primary_authority is not None
                else {}
            ),
        )
        primary = self._trace_with_analysis_identity(primary, cohort_index=0)
        primary_selected = {
            "action": action,
            "trumpSuit": trump.name.lower() if trump is not None else None,
        }
        if primary_authority is not None:
            validate_primary_bid_decision_score_trace_v1(
                primary,
                primary_authority,
                selected_action=primary_selected,
                forced=len(legal) == 1,
            )
        if len(legal) <= 1:
            return action, trump, primary
        if cancel_check is not None and cancel_check():
            raise InterruptedError("finalist decision agreement cancelled")
        if agreement_ordinal > self.profile.maximum_decisions_per_trace:
            selected = primary_selected
            choice = next(item for item in legal_choices if item.to_dict() == selected)
            return choice.action, choice.trump_suit, self._combined_trace(
                decision_kind="bid", primary_seed=primary_seed, independent_seed=None,
                third_seed=None, primary=primary, independent=None, third=None,
                selected=selected,
                reasons=("agreement_bound_exceeded",), bound_exceeded=True,
                strict_reasons=("agreement_bound_exceeded",),
            )
        independent_seed = self._cohort_seed(
            primary_seed,
            "bid",
            2,
            public_prefix_hash=public_prefix_hash,
            acting_seat=acting_seat,
        )
        _second_action, _second_trump, independent = self.independent_bidding.choose_with_trace(
            state, machine, rules, seed=independent_seed,
            cancel_check=cancel_check,
            **(
                {
                    "primary_decision_score_authority": self._bid_score_authority(
                        state=state,
                        machine=machine,
                        rules=rules,
                        legal_choices=legal,
                        public_prefix_hash=public_prefix_hash,
                        cohort_index=1,
                        cohort_seed=independent_seed,
                    )
                }
                if self.exact_primary_policy
                else {}
            ),
        )
        independent = self._trace_with_analysis_identity(independent, cohort_index=1)
        if self.exact_primary_policy:
            independent_authority = self._bid_score_authority(
                state=state,
                machine=machine,
                rules=rules,
                legal_choices=legal,
                public_prefix_hash=public_prefix_hash,
                cohort_index=1,
                cohort_seed=independent_seed,
            )
            assert independent_authority is not None
            validate_primary_bid_decision_score_trace_v1(
                independent,
                independent_authority,
                selected_action=dict(independent["selectedAction"]),
                forced=False,
            )
        if cancel_check is not None and cancel_check():
            raise InterruptedError("finalist decision agreement cancelled")
        traces: list[Mapping[str, Any]] = [primary, independent]
        selected, initial_reasons, agreement_count, majority_hash = (
            self._resolve_cohorts(traces)
        )
        third_seed: int | None = None
        third: Mapping[str, Any] | None = None
        competence_evidence: Mapping[str, Any] | None = None
        strict_reasons: list[str] = []
        reasons = list(initial_reasons)
        if (
            selected is None
            and self.profile.maximum_cohorts >= 3
            and self.profile.third_cohort_on_disagreement
        ):
            third_seed = self._cohort_seed(
                primary_seed,
                "bid",
                3,
                public_prefix_hash=public_prefix_hash,
                acting_seat=acting_seat,
            )
            _third_action, _third_trump, third_trace = (
                self.third_bidding.choose_with_trace(
                    state,
                    machine,
                    rules,
                    seed=third_seed,
                    cancel_check=cancel_check,
                    **(
                        {
                            "primary_decision_score_authority": (
                                self._bid_score_authority(
                                    state=state,
                                    machine=machine,
                                    rules=rules,
                                    legal_choices=legal,
                                    public_prefix_hash=public_prefix_hash,
                                    cohort_index=2,
                                    cohort_seed=third_seed,
                                )
                            )
                        }
                        if self.exact_primary_policy
                        else {}
                    ),
                )
            )
            if cancel_check is not None and cancel_check():
                raise InterruptedError("finalist decision agreement cancelled")
            third = self._trace_with_analysis_identity(third_trace, cohort_index=2)
            if self.exact_primary_policy:
                third_authority = self._bid_score_authority(
                    state=state,
                    machine=machine,
                    rules=rules,
                    legal_choices=legal,
                    public_prefix_hash=public_prefix_hash,
                    cohort_index=2,
                    cohort_seed=third_seed,
                )
                assert third_authority is not None
                validate_primary_bid_decision_score_trace_v1(
                    third,
                    third_authority,
                    selected_action=dict(third["selectedAction"]),
                    forced=False,
                )
            traces.append(third)
            selected, reasons, agreement_count, majority_hash = (
                self._resolve_cohorts(traces)
            )
        if self.exact_primary_policy:
            strict_reasons = self._primary_retention_strict_reasons(
                reasons,
                selected=selected,
                primary_best_action=primary["intendedBestAction"],
            )
            selected = primary_selected
            if strict_reasons:
                competence_evidence = self._resolve_primary_competence(
                    primary_selected,
                    traces,
                )
                competence_evidence = {
                    **competence_evidence,
                    "strictReasons": strict_reasons,
                }
                if (
                    competence_evidence.get("status")
                    == "primary_robust_competent"
                ):
                    reasons = []
                else:
                    reasons = sorted({
                        *strict_reasons,
                        *self._competence_blocking_reasons(competence_evidence),
                    })
        elif selected is None or _action_hash(selected) != _action_hash(
            primary_selected
        ):
            strict_reasons = self._primary_retention_strict_reasons(
                reasons,
                selected=selected,
                primary_best_action=primary["intendedBestAction"],
            )
            competence = self._resolve_primary_competence(primary_selected, traces)
            competence_evidence = {
                **competence,
                "strictReasons": strict_reasons,
            }
            if competence.get("status") == "primary_robust_competent":
                selected = primary_selected
                competence_evidence = {
                    **competence,
                    "strictReasons": strict_reasons,
                }
                reasons = []
            else:
                selected = primary_selected
                reasons = sorted({
                    *strict_reasons,
                    *self._competence_blocking_reasons(competence),
                })
        choice = next(item for item in legal_choices if item.to_dict() == selected)
        action, trump = choice.action, choice.trump_suit
        return action, trump, self._combined_trace(
            decision_kind="bid", primary_seed=primary_seed,
            independent_seed=independent_seed, primary=primary,
            third_seed=third_seed, independent=independent, third=third,
            selected=selected, reasons=reasons,
            agreement_count=agreement_count,
            majority_action_hash=majority_hash,
            initial_reasons=initial_reasons,
            strict_reasons=strict_reasons,
            competence_evidence=competence_evidence,
        )

    def choose_play(
        self,
        primary_strategy: AdaptivePlayStrategy,
        engine: BalootEngine,
        state: RoundState,
        legal_cards: Sequence[Card],
        *,
        seed: int,
        agreement_ordinal: int,
        cancel_check: Callable[[], bool] | None = None,
        public_prefix_hash: str | None = None,
    ) -> tuple[Card, dict[str, Any]]:
        legal = tuple(legal_cards)
        acting_seat = getattr(state, "current_player", None)
        primary_seed = self._required_primary_seed(
            seed,
            public_prefix_hash=public_prefix_hash,
            decision_kind="play",
            acting_seat=acting_seat,
        )
        primary_kwargs: dict[str, Any] = {
            "seed": primary_seed,
            "cancel_check": cancel_check,
        }
        primary_authority = self._play_score_authority(
            engine=engine,
            state=state,
            legal_cards=legal,
            public_prefix_hash=public_prefix_hash,
            cohort_index=0,
            cohort_seed=primary_seed,
        )
        if self.exact_primary_policy:
            primary_kwargs["analysis_budget"] = self.primary_analysis_budget
            primary_kwargs["primary_decision_score_authority"] = primary_authority
        card, primary = primary_strategy.choose_with_trace(
            engine,
            state,
            legal,
            **primary_kwargs,
        )
        primary = self._trace_with_analysis_identity(primary, cohort_index=0)
        primary_selected = card.to_dict()
        if primary_authority is not None:
            validate_primary_play_decision_score_trace_v1(
                primary,
                primary_authority,
                selected_action=primary_selected,
                forced=len(legal) == 1,
            )
        if len(legal) <= 1:
            return card, primary
        if cancel_check is not None and cancel_check():
            raise InterruptedError("finalist decision agreement cancelled")
        if agreement_ordinal > self.profile.maximum_decisions_per_trace:
            selected = primary_selected
            card = next(item for item in legal if item.to_dict() == selected)
            return card, self._combined_trace(
                decision_kind="play", primary_seed=primary_seed, independent_seed=None,
                third_seed=None, primary=primary, independent=None, third=None,
                selected=selected,
                reasons=("agreement_bound_exceeded",), bound_exceeded=True,
                strict_reasons=("agreement_bound_exceeded",),
            )
        independent_seed = self._cohort_seed(
            primary_seed,
            "play",
            2,
            public_prefix_hash=public_prefix_hash,
            acting_seat=acting_seat,
        )
        _second, independent = self.independent_play.choose_with_trace(
            engine,
            state,
            legal,
            seed=independent_seed,
            cancel_check=cancel_check,
            analysis_budget=self.independent_analysis_budget,
            **(
                {
                    "primary_decision_score_authority": self._play_score_authority(
                        engine=engine,
                        state=state,
                        legal_cards=legal,
                        public_prefix_hash=public_prefix_hash,
                        cohort_index=1,
                        cohort_seed=independent_seed,
                    )
                }
                if self.exact_primary_policy
                else {}
            ),
        )
        independent = self._trace_with_analysis_identity(independent, cohort_index=1)
        if self.exact_primary_policy:
            independent_authority = self._play_score_authority(
                engine=engine,
                state=state,
                legal_cards=legal,
                public_prefix_hash=public_prefix_hash,
                cohort_index=1,
                cohort_seed=independent_seed,
            )
            assert independent_authority is not None
            validate_primary_play_decision_score_trace_v1(
                independent,
                independent_authority,
                selected_action=dict(independent["selectedAction"]),
                forced=False,
            )
        traces: list[Mapping[str, Any]] = [primary, independent]
        selected, initial_reasons, agreement_count, majority_hash = (
            self._resolve_cohorts(traces)
        )
        third_seed: int | None = None
        third: Mapping[str, Any] | None = None
        competence_evidence: Mapping[str, Any] | None = None
        strict_reasons: list[str] = []
        reasons = list(initial_reasons)
        if (
            selected is None
            and self.profile.maximum_cohorts >= 3
            and self.profile.third_cohort_on_disagreement
        ):
            third_seed = self._cohort_seed(
                primary_seed,
                "play",
                3,
                public_prefix_hash=public_prefix_hash,
                acting_seat=acting_seat,
            )
            _third, third_trace = self.third_play.choose_with_trace(
                engine,
                state,
                legal,
                seed=third_seed,
                cancel_check=cancel_check,
                analysis_budget=self.independent_analysis_budget,
                **(
                    {
                        "primary_decision_score_authority": (
                            self._play_score_authority(
                                engine=engine,
                                state=state,
                                legal_cards=legal,
                                public_prefix_hash=public_prefix_hash,
                                cohort_index=2,
                                cohort_seed=third_seed,
                            )
                        )
                    }
                    if self.exact_primary_policy
                    else {}
                ),
            )
            if cancel_check is not None and cancel_check():
                raise InterruptedError("finalist decision agreement cancelled")
            third = self._trace_with_analysis_identity(third_trace, cohort_index=2)
            if self.exact_primary_policy:
                third_authority = self._play_score_authority(
                    engine=engine,
                    state=state,
                    legal_cards=legal,
                    public_prefix_hash=public_prefix_hash,
                    cohort_index=2,
                    cohort_seed=third_seed,
                )
                assert third_authority is not None
                validate_primary_play_decision_score_trace_v1(
                    third,
                    third_authority,
                    selected_action=dict(third["selectedAction"]),
                    forced=False,
                )
            traces.append(third)
            selected, reasons, agreement_count, majority_hash = (
                self._resolve_cohorts(traces)
            )
        if self.exact_primary_policy:
            strict_reasons = self._primary_retention_strict_reasons(
                reasons,
                selected=selected,
                primary_best_action=primary["intendedBestAction"],
            )
            selected = primary_selected
            if strict_reasons:
                competence_evidence = self._resolve_primary_competence(
                    primary_selected,
                    traces,
                )
                competence_evidence = {
                    **competence_evidence,
                    "strictReasons": strict_reasons,
                }
                if (
                    competence_evidence.get("status")
                    == "primary_robust_competent"
                ):
                    reasons = []
                else:
                    reasons = sorted({
                        *strict_reasons,
                        *self._competence_blocking_reasons(competence_evidence),
                    })
        elif selected is None or _action_hash(selected) != _action_hash(
            primary_selected
        ):
            strict_reasons = self._primary_retention_strict_reasons(
                reasons,
                selected=selected,
                primary_best_action=primary["intendedBestAction"],
            )
            competence = self._resolve_primary_competence(primary_selected, traces)
            competence_evidence = {
                **competence,
                "strictReasons": strict_reasons,
            }
            if competence.get("status") == "primary_robust_competent":
                selected = primary_selected
                competence_evidence = {
                    **competence,
                    "strictReasons": strict_reasons,
                }
                reasons = []
            else:
                selected = primary_selected
                reasons = sorted({
                    *strict_reasons,
                    *self._competence_blocking_reasons(competence),
                })
        card = next(item for item in legal if item.to_dict() == selected)
        return card, self._combined_trace(
            decision_kind="play", primary_seed=primary_seed,
            independent_seed=independent_seed, primary=primary,
            third_seed=third_seed, independent=independent, third=third,
            selected=selected, reasons=reasons,
            agreement_count=agreement_count,
            majority_action_hash=majority_hash,
            initial_reasons=initial_reasons,
            strict_reasons=strict_reasons,
            competence_evidence=competence_evidence,
        )
