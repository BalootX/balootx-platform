from dataclasses import replace
from types import SimpleNamespace

import pytest

from baloot_platform.agents.profiles import AgentProfile, AgentProfileRegistry
from baloot_platform.analysis.bidding import BidChoice
from baloot_platform.analysis.decision_scores import (
    canonical_json_bytes_v1,
    canonical_sha256_v1,
    primary_decision_score_confidence_v1,
    primary_decision_score_evidence_v1,
)
from baloot_platform.challenge.analysis_pipeline import TraceAnalysisProfileV1
from baloot_platform.challenge.evidence_v3 import canonical_sha256
from baloot_platform.challenge.finalist_agent_agreement import (
    FinalistAgentAgreementProfileV1,
    FinalistAgentAgreementV1,
)
from baloot_platform.challenge.finalist_competence import (
    build_finalist_adjudication_profile_v2,
    canonical_finalist_cohort_v1,
    project_finalist_adjudication_v2,
)
from baloot_platform.challenge.simulation import NaturalLaffaSimulatorV1
from baloot_platform.config import build_bundle
from baloot_platform.core.cards import Card
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, Seat, Suit


def _agent() -> AgentProfile:
    return AgentProfile.from_mapping("finalist_candidate", {
        "skill": 1,
        "buying_strength": 0.8,
        "play_search_budget": 8000,
        "play_search_mode": "paired_information_set",
        "play_search_objective": "match_win_probability",
        "play_search_max_remaining": 32,
        "play_search_samples": 192,
        "play_exact_allocation_limit": 128,
        "play_exact_search_remaining": 9,
        "play_maximum_sampling_attempts": 6000,
        "mistake_probability": 0,
        "deception_probability": 0,
        "use_production_model": False,
        "bid_samples": 192,
    })


def _stability(**overrides) -> dict:
    stability = {
        "reanalysis_samples": 256,
        "maximum_sampling_attempts": 6000,
        "exact_allocation_limit": 256,
        "exact_search_remaining": 9,
        "exact_search_max_nodes": 120000,
        "exact_search_max_seconds": 0.8,
        "seed_offset": 104729,
        "minimum_hidden_deals": 8,
        "minimum_best_share": 0.5,
        "maximum_competence_expected_margin_regret": 1.0,
        "maximum_competence_match_win_drop": 0.01,
        "maximum_expected_margin_drift": 5.0,
        "maximum_cohorts": 3,
        "minimum_agreement": 2,
        "third_cohort_on_disagreement": True,
        "require_same_recommended_action": True,
    }
    stability.update(overrides)
    return stability


def _agreement(
    maximum: int = 64,
    **stability_overrides,
) -> FinalistAgentAgreementV1:
    return FinalistAgentAgreementV1(
        _agent(),
        _stability(**stability_overrides),
        maximum_decisions_per_trace=maximum,
        minimum_confidence=0.75,
        convention={},
    )


def _deep_trace_profile() -> TraceAnalysisProfileV1:
    return TraceAnalysisProfileV1.build(
        bid_samples=192,
        play_samples=192,
        exact_allocation_limit=128,
        exact_search_max_remaining=9,
        maximum_sampling_attempts=4000,
        exact_search_max_nodes=80000,
        exact_search_max_seconds=0.4,
        exact_search_max_depth=32,
        exact_search_max_cache_entries=100000,
        objective="match_win_probability",
        use_conventions=False,
    )


def _reanalysis_trace_profile() -> TraceAnalysisProfileV1:
    primary = _deep_trace_profile()
    return TraceAnalysisProfileV1.build(
        bid_samples=256,
        play_samples=256,
        exact_allocation_limit=256,
        exact_search_max_remaining=9,
        maximum_sampling_attempts=6000,
        exact_search_max_nodes=120000,
        exact_search_max_seconds=0.8,
        exact_search_max_depth=primary.exact_search_max_depth,
        exact_search_max_cache_entries=primary.exact_search_max_cache_entries,
        search_backend_id=primary.search_backend_id,
        objective=primary.objective,
        use_conventions=False,
    )


def _analysis_identity(role: str, profile: TraceAnalysisProfileV1) -> str:
    return canonical_sha256({
        "schemaVersion": 1,
        "authorityRole": role,
        "traceAnalysisProfileHash": profile.sha256,
        "analysisPolicy": "test_actor_observer_primary_v1",
    })


def _exact_authorities() -> tuple[
    TraceAnalysisProfileV1, str, TraceAnalysisProfileV1, str
]:
    primary = _deep_trace_profile()
    reanalysis = _reanalysis_trace_profile()
    return (
        primary,
        _analysis_identity("deep_confirmation", primary),
        reanalysis,
        _analysis_identity("finalist_reanalysis", reanalysis),
    )


def _exact_agreement(
    maximum: int = 64,
    **stability_overrides,
) -> FinalistAgentAgreementV1:
    primary, primary_identity, reanalysis, reanalysis_identity = (
        _exact_authorities()
    )
    return FinalistAgentAgreementV1(
        _agent(),
        _stability(**stability_overrides),
        maximum_decisions_per_trace=maximum,
        minimum_confidence=0.75,
        convention={"signals": {"enabled": True}},
        primary_trace_analysis_profile=primary,
        primary_analysis_identity_hash=primary_identity,
        reanalysis_trace_analysis_profile=reanalysis,
        reanalysis_analysis_identity_hash=reanalysis_identity,
    )


def _play_trace(
    best: Card,
    other: Card,
    *,
    hidden_deals: int = 192,
    confidence: float = 1.0,
    completion: str = "exact",
    exhaustive: bool = True,
    best_share: float = 0.7,
    best_margin: float = 5.0,
    other_margin: float = 2.0,
    best_win: float = 0.60,
    other_win: float = 0.50,
    population_complete: bool = False,
    best_interval: tuple[float, float] = (0.0, 1.0),
    other_interval: tuple[float, float] = (0.0, 1.0),
) -> dict:
    return {
        "intendedBestAction": best.to_dict(),
        "selectedAction": best.to_dict(),
        "confidence": confidence,
        "analysisHiddenDeals": hidden_deals,
        "analysisCompletionStatuses": [completion],
        "analysisExhaustiveSupport": exhaustive,
        "analysisPopulationCoverageComplete": population_complete or exhaustive,
        "analysisSolverCompletion": (
            "exact" if exhaustive else completion
        ),
        "analysisTruthClass": (
            "exact_optimal"
            if exhaustive else (
                "full_population_best_known"
                if population_complete else "unresolved"
            )
        ),
        "rankedActions": [
            {
                "action": best.to_dict(), "rank": 1,
                "expectedMargin": best_margin, "bestShare": best_share,
                "winProbability": best_win,
                "matchWinProbability": best_win,
                "confidence95": list(best_interval),
            },
            {
                "action": other.to_dict(), "rank": 2,
                "expectedMargin": other_margin, "bestShare": 1.0 - best_share,
                "winProbability": other_win,
                "matchWinProbability": other_win,
                "confidence95": list(other_interval),
            },
        ],
    }


class _PlayStrategy:
    def __init__(
        self,
        selected: Card,
        trace: dict,
        *,
        cancelled=None,
        emit_score_evidence: bool = True,
        canonicalize_confidence: bool = True,
    ):
        self.selected = selected
        self.trace = trace
        self.calls = 0
        self.call_kwargs = []
        self.cancelled = cancelled
        self.emit_score_evidence = emit_score_evidence
        self.canonicalize_confidence = canonicalize_confidence

    def choose_with_trace(self, *_args, **_kwargs):
        self.calls += 1
        self.call_kwargs.append(dict(_kwargs))
        if self.cancelled is not None:
            self.cancelled["called"] = True
        trace = dict(self.trace)
        authority = _kwargs.get("primary_decision_score_authority")
        if self.emit_score_evidence:
            ranked = tuple(trace.get("rankedActions") or ())
            rows = []
            hidden = int(trace.get("analysisHiddenDeals") or 2)
            completion = str(
                (trace.get("analysisCompletionStatuses") or ["sampled"])[0]
            )
            exhaustive = bool(trace.get("analysisExhaustiveSupport"))
            population_complete = bool(
                trace.get("analysisPopulationCoverageComplete")
            )
            resolved_authority = authority or {
                "actor": Seat.NORTH,
                "publicPrefixHash": "0" * 64,
                "rulesHash": "1" * 64,
                "analysisIdentityHash": "2" * 64,
                "traceAnalysisProfileHash": "3" * 64,
                "cohortSeedPolicy": "test_actor_observer_seed_v1",
                "cohortIndex": 0,
                "cohortSeed": int(_kwargs.get("seed", 0)),
                "objective": "match_win_probability",
                "tieBreak": "test_card_order_v1",
                "backend": "test_reference_v1",
            }
            for rank, item in enumerate(ranked, start=1):
                margin = float(item.get("expectedMargin", 0.0))
                probability = float(item.get("winProbability", 0.5))
                action = dict(item["action"])
                interval = [float(value) for value in item["confidence95"]]
                rows.append({
                    "rank": rank,
                    "actionId": f"card:{Card.parse(action).id}",
                    "action": action,
                    "expectedRoundMargin": margin,
                    "expectedTeamAwardedPoints": 100.0 + margin,
                    "expectedOpponentAwardedPoints": 100.0,
                    "roundLeadProbability": probability,
                    "drawProbability": 0.0,
                    "roundLeadConfidence95": interval,
                    "bestShare": float(item.get("bestShare", 0.0)),
                    "samples": hidden,
                    "effectiveSamples": float(hidden),
                    "exactShare": 1.0 if exhaustive else 0.0,
                    "contractSuccessProbability": probability,
                    "matchWinProbability": float(
                        item.get("matchWinProbability", probability)
                    ),
                    "matchWinConfidence95": interval,
                    "matchValueModel": "test_match_value_v1",
                    "supportMarginsHash": canonical_sha256_v1([margin]),
                    "supportMarginCount": 1,
                    "downsideCvar10": margin,
                    "volatility": 0.0,
                    "principalVariationHash": canonical_sha256_v1([]),
                    "principalVariationLength": 0,
                    "searchNodes": 1,
                    "searchCompletionStatus": (
                        "exact" if exhaustive else completion
                    ),
                })
            evidence = primary_decision_score_evidence_v1(
                decision_kind="play",
                actor=resolved_authority["actor"],
                public_prefix_hash=resolved_authority["publicPrefixHash"],
                rules_hash=resolved_authority["rulesHash"],
                analysis_identity_hash=resolved_authority["analysisIdentityHash"],
                trace_analysis_profile_hash=resolved_authority[
                    "traceAnalysisProfileHash"
                ],
                cohort_seed_policy=resolved_authority["cohortSeedPolicy"],
                cohort_index=resolved_authority["cohortIndex"],
                cohort_seed=resolved_authority["cohortSeed"],
                objective=resolved_authority["objective"],
                tie_break=resolved_authority["tieBreak"],
                backend=resolved_authority["backend"],
                method=(
                    "exhaustive_test_hidden_information_search"
                    if exhaustive or population_complete
                    else "test_hidden_information_search"
                ),
                hidden_deal_support=hidden,
                effective_sample_size=float(hidden),
                recommended_action=self.selected.to_dict(),
                rows=rows,
                legal_action_set_hash=(
                    resolved_authority.get("legalActionSetHash")
                ),
            )
            trace.update({
                "analysisMethod": evidence.method,
                "analysisHiddenDeals": evidence.hidden_deal_support,
                "analysisEffectiveSampleSize": evidence.effective_sample_size,
                "publicPrefixHash": evidence.public_prefix_hash,
                "rulesHash": evidence.rules_hash,
                "analysisIdentityHash": evidence.analysis_identity_hash,
                "traceAnalysisProfileHash": evidence.trace_analysis_profile_hash,
                "cohortSeedPolicy": evidence.cohort_seed_policy,
                "cohortIndex": evidence.cohort_index,
                "cohortSeed": evidence.cohort_seed,
                "analysisObjective": evidence.objective,
                "analysisTieBreak": evidence.tie_break,
                "analysisBackend": evidence.backend,
                "legalActionSetHash": evidence.legal_action_set_hash,
                "primaryDecisionScoreEvidence": evidence.to_dict(),
                "primaryDecisionScoreEvidenceHash": evidence.sha256,
                "decisionScoreRowsHash": evidence.rows_hash,
                "forcedLegalityProof": None,
            })
            if self.canonicalize_confidence:
                trace["confidence"] = primary_decision_score_confidence_v1(
                    evidence
                )
        return self.selected, trace


class _BidStrategy:
    def __init__(
        self,
        selected: BidChoice,
        trace: dict,
        *,
        canonicalize_confidence: bool = True,
    ):
        self.selected = selected
        self.trace = trace
        self.calls = 0
        self.canonicalize_confidence = canonicalize_confidence

    def choose_with_trace(self, *_args, **_kwargs):
        self.calls += 1
        trace = dict(self.trace)
        authority = _kwargs.get("primary_decision_score_authority") or {
            "actor": Seat.NORTH,
            "publicPrefixHash": "0" * 64,
            "rulesHash": "1" * 64,
            "analysisIdentityHash": "2" * 64,
            "traceAnalysisProfileHash": "3" * 64,
            "cohortSeedPolicy": "test_actor_observer_seed_v1",
            "cohortIndex": 0,
            "cohortSeed": int(_kwargs.get("seed", 0)),
            "objective": "expected_round_margin",
            "tieBreak": "test_bid_order_v1",
            "backend": "test_reference_v1",
        }
        hidden = int(trace.get("analysisHiddenDeals") or 2)
        rows = []
        for rank, item in enumerate(trace.get("rankedActions") or (), start=1):
            action = {
                "action": item["action"],
                "trumpSuit": item.get("trumpSuit"),
            }
            probability = float(item.get("winProbability", 0.5))
            rows.append({
                "rank": rank,
                "actionId": BidChoice(
                    action["action"],
                    Suit.parse(action["trumpSuit"])
                    if action["trumpSuit"] is not None else None,
                ).id,
                "action": action,
                "expectedRoundMargin": float(item["expectedMargin"]),
                "roundLeadProbability": probability,
                "contractSuccessProbability": probability,
                "downsideCvar10": float(item["expectedMargin"]),
                "volatility": 0.0,
                "bestShare": float(item["bestShare"]),
                "samples": hidden,
            })
        evidence = primary_decision_score_evidence_v1(
            decision_kind="bid",
            actor=authority["actor"],
            public_prefix_hash=authority["publicPrefixHash"],
            rules_hash=authority["rulesHash"],
            analysis_identity_hash=authority["analysisIdentityHash"],
            trace_analysis_profile_hash=authority["traceAnalysisProfileHash"],
            cohort_seed_policy=authority["cohortSeedPolicy"],
            cohort_index=authority["cohortIndex"],
            cohort_seed=authority["cohortSeed"],
            objective=authority["objective"],
            tie_break=authority["tieBreak"],
            backend=authority["backend"],
            method="test_hidden_information_bid_analysis",
            hidden_deal_support=hidden,
            effective_sample_size=float(hidden),
            recommended_action=self.selected.to_dict(),
            rows=rows,
            legal_action_set_hash=authority.get("legalActionSetHash"),
        )
        trace.update({
            "analysisMethod": evidence.method,
            "analysisHiddenDeals": evidence.hidden_deal_support,
            "analysisEffectiveSampleSize": evidence.effective_sample_size,
            "publicPrefixHash": evidence.public_prefix_hash,
            "rulesHash": evidence.rules_hash,
            "analysisIdentityHash": evidence.analysis_identity_hash,
            "traceAnalysisProfileHash": evidence.trace_analysis_profile_hash,
            "cohortSeedPolicy": evidence.cohort_seed_policy,
            "cohortIndex": evidence.cohort_index,
            "cohortSeed": evidence.cohort_seed,
            "analysisObjective": evidence.objective,
            "analysisTieBreak": evidence.tie_break,
            "analysisBackend": evidence.backend,
            "legalActionSetHash": evidence.legal_action_set_hash,
            "primaryDecisionScoreEvidence": evidence.to_dict(),
            "primaryDecisionScoreEvidenceHash": evidence.sha256,
            "decisionScoreRowsHash": evidence.rows_hash,
            "forcedLegalityProof": None,
        })
        if self.canonicalize_confidence:
            trace["confidence"] = primary_decision_score_confidence_v1(evidence)
        return self.selected.action, self.selected.trump_suit, trace


class _InterruptingBidStrategy:
    def choose_with_trace(self, *_args, **_kwargs):
        raise InterruptedError("bid analysis cancelled")


def _bid_trace(
    best: BidChoice,
    other: BidChoice,
    *,
    hidden_deals: int = 192,
    confidence: float = 1.0,
) -> dict:
    return {
        "intendedBestAction": best.to_dict(),
        "selectedAction": best.to_dict(),
        "confidence": confidence,
        "analysisHiddenDeals": hidden_deals,
        "analysisCompletionStatuses": ["sampled"],
        "analysisExhaustiveSupport": False,
        "analysisPopulationCoverageComplete": False,
        "analysisSolverCompletion": "sampled",
        "analysisTruthClass": "unresolved",
        "rankedActions": [
            {
                **best.to_dict(), "rank": 1, "expectedMargin": 4.0,
                "bestShare": 0.7, "winProbability": 0.60,
            },
            {
                **other.to_dict(), "rank": 2, "expectedMargin": 1.0,
                "bestShare": 0.3, "winProbability": 0.50,
            },
        ],
    }


def _canonical_replay_verdict(
    agreement: FinalistAgentAgreementV1,
    trace: dict,
) -> dict:
    primary_action = dict(trace["primaryTrace"]["selectedAction"])
    cohorts = tuple(
        canonical_finalist_cohort_v1(
            item["primaryDecisionScoreEvidence"],
            cohort_index=index,
            evidence_hash=item["primaryDecisionScoreEvidenceHash"],
            actual_action=primary_action,
            best_action=item["intendedBestAction"],
        )
        for index, item in enumerate(trace["cohorts"])
    )
    profile = agreement.profile
    adjudication_profile = build_finalist_adjudication_profile_v2(
        {
            "seed_offset": profile.seed_offset,
            "minimum_hidden_deals": profile.minimum_hidden_deals,
            "minimum_best_share": profile.minimum_best_share,
            "maximum_competence_expected_margin_regret": (
                profile.maximum_competence_expected_margin_regret
            ),
            "maximum_competence_match_win_drop": (
                profile.maximum_competence_match_win_drop
            ),
            "maximum_expected_margin_drift": profile.maximum_expected_margin_drift,
            "maximum_cohorts": profile.maximum_cohorts,
            "minimum_agreement": profile.minimum_agreement,
            "third_cohort_on_disagreement": profile.third_cohort_on_disagreement,
            "require_same_recommended_action": (
                profile.require_same_recommended_action
            ),
        },
        primary_profile_hash=(
            profile.primary_trace_analysis_profile_hash or "1" * 64
        ),
        reanalysis_profile_hash=(
            profile.reanalysis_trace_analysis_profile_hash or "2" * 64
        ),
        maximum_decisions_per_trace=profile.maximum_decisions_per_trace,
        minimum_confidence=profile.minimum_confidence,
        minimum_quality=0.0,
    )
    return project_finalist_adjudication_v2(
        cohorts,
        primary_best_action=trace["primaryTrace"]["intendedBestAction"],
        primary_actual_action=primary_action,
        adjudication_profile=adjudication_profile,
        initial_reasons=trace["initialReasons"],
    )


def test_independent_seed_is_deterministic_separate_and_kind_specific():
    agreement = _agreement()

    first = agreement.independent_seed(17, "play")

    assert first == agreement.independent_seed(17, "play")
    assert first != 17
    assert first != agreement.independent_seed(17, "bid")
    assert first != agreement.independent_seed(17, "play", 3)


def test_frozen_analysis_authority_is_all_or_none_and_profile_compatible():
    profile, identity, reanalysis, reanalysis_identity = _exact_authorities()
    stability = _stability()
    kwargs = {
        "maximum_decisions_per_trace": 64,
        "minimum_confidence": 0.75,
        "convention": {},
    }

    with pytest.raises(ValueError, match="supplied together"):
        FinalistAgentAgreementV1(
            _agent(),
            stability,
            trace_analysis_profile=profile,
            **kwargs,
        )
    with pytest.raises(ValueError, match="supplied together"):
        FinalistAgentAgreementV1(
            _agent(),
            stability,
            analysis_identity_hash="a" * 64,
            **kwargs,
        )
    with pytest.raises(ValueError, match="objective"):
        FinalistAgentAgreementV1(
            replace(_agent(), play_search_objective="match_point_margin"),
            stability,
            primary_trace_analysis_profile=profile,
            primary_analysis_identity_hash=identity,
            reanalysis_trace_analysis_profile=reanalysis,
            reanalysis_analysis_identity_hash=reanalysis_identity,
            **kwargs,
        )
    with pytest.raises(ValueError, match="full-line"):
        FinalistAgentAgreementV1(
            replace(_agent(), play_search_max_remaining=31),
            stability,
            primary_trace_analysis_profile=profile,
            primary_analysis_identity_hash=identity,
            reanalysis_trace_analysis_profile=reanalysis,
            reanalysis_analysis_identity_hash=reanalysis_identity,
            **kwargs,
        )

    with pytest.raises(ValueError, match="profiles must be distinct"):
        FinalistAgentAgreementV1(
            _agent(),
            stability,
            primary_trace_analysis_profile=profile,
            primary_analysis_identity_hash=identity,
            reanalysis_trace_analysis_profile=profile,
            reanalysis_analysis_identity_hash=reanalysis_identity,
            **kwargs,
        )
    with pytest.raises(ValueError, match="identities must be distinct"):
        FinalistAgentAgreementV1(
            _agent(),
            stability,
            primary_trace_analysis_profile=profile,
            primary_analysis_identity_hash=identity,
            reanalysis_trace_analysis_profile=reanalysis,
            reanalysis_analysis_identity_hash=identity,
            **kwargs,
        )


def test_frozen_primary_seed_uses_only_actor_visible_state_and_analysis_identity():
    agreement = _exact_agreement()
    public_hash = canonical_sha256({"observer": "north", "legal": ["S7", "S8"]})

    first = agreement.actor_observer_seed(
        public_prefix_hash=public_hash,
        decision_kind="play",
        acting_seat=Seat.NORTH,
    )

    # Root seed, trace identity, and selected action are deliberately absent
    # from the authority API and therefore cannot perturb this seed.
    for _untrusted_context in (
        {"rootSeed": 1, "traceHash": "b" * 64, "selectedAction": "S7"},
        {"rootSeed": 999, "traceHash": "c" * 64, "selectedAction": "S8"},
    ):
        assert agreement.actor_observer_seed(
            public_prefix_hash=public_hash,
            decision_kind="play",
            acting_seat=Seat.NORTH,
        ) == first
    assert agreement.actor_observer_seed(
        public_prefix_hash=canonical_sha256({"observer": "east"}),
        decision_kind="play",
        acting_seat=Seat.EAST,
    ) != first


def test_injected_stability_never_replaces_primary_and_uses_distinct_budgets():
    agreement = _exact_agreement()
    assert agreement.independent_play.convention == {}
    assert agreement.third_play.convention == {}
    first, second = Card.parse("D10"), Card.parse("C7")
    primary = _PlayStrategy(first, _play_trace(first, second))
    independent = _PlayStrategy(second, _play_trace(second, first))
    third = _PlayStrategy(second, _play_trace(second, first))
    agreement.independent_play = independent
    agreement.third_play = third
    public_hash = canonical_sha256({"observer": "north", "state": "same"})
    primary_seed = agreement.actor_observer_seed(
        public_prefix_hash=public_hash,
        decision_kind="play",
        acting_seat=Seat.NORTH,
    )

    card, trace = agreement.choose_play(
        primary,
        SimpleNamespace(rules=SimpleNamespace(sha256="f" * 64)),
        SimpleNamespace(current_player=Seat.NORTH),
        (first, second),
        seed=primary_seed,
        agreement_ordinal=1,
        public_prefix_hash=public_hash,
    )

    primary_budget = {
        "requested_samples": 192,
        "exact_allocation_limit": 128,
        "exact_search_max_remaining": 9,
        "maximum_sampling_attempts": 4000,
        "exact_search_max_nodes": 80000,
        "exact_search_max_seconds": 0.4,
        "exact_search_max_depth": 32,
        "exact_search_max_cache_entries": 100000,
    }
    reanalysis_budget = {
        "requested_samples": 256,
        "exact_allocation_limit": 256,
        "exact_search_max_remaining": 9,
        "maximum_sampling_attempts": 6000,
        "exact_search_max_nodes": 120000,
        "exact_search_max_seconds": 0.8,
        "exact_search_max_depth": 32,
        "exact_search_max_cache_entries": 100000,
    }
    primary_profile, primary_identity, reanalysis_profile, reanalysis_identity = (
        _exact_authorities()
    )
    assert card == first
    assert trace["selectedAction"] == first.to_dict()
    assert trace["primaryActionPreserved"] is True
    assert trace["selectionMode"] == "primary_policy_only"
    assert trace["status"] == "unresolved"
    assert trace["qualificationBlocked"] is True
    assert trace["fallbackPolicy"] is None
    assert "primary_recommendation_not_confirmed" in trace["reasons"]
    assert trace["traceAnalysisProfileHash"] == primary_profile.sha256
    assert trace["analysisIdentityHash"] == primary_identity
    assert trace["analysisBudget"] == primary_budget
    assert trace["analysisAuthorities"] == {
        "primary": {
            "role": "deep_confirmation",
            "traceAnalysisProfileHash": primary_profile.sha256,
            "analysisIdentityHash": primary_identity,
            "analysisBudget": primary_budget,
        },
        "reanalysis": {
            "role": "finalist_reanalysis",
            "traceAnalysisProfileHash": reanalysis_profile.sha256,
            "analysisIdentityHash": reanalysis_identity,
            "analysisBudget": reanalysis_budget,
        },
    }
    primary_evidence = trace["primaryTrace"]["primaryDecisionScoreEvidence"]
    assert trace["primaryDecisionScoreEvidence"] == primary_evidence
    assert trace["primaryDecisionScoreEvidenceHash"] == primary_evidence["sha256"]
    assert trace["decisionScoreRowsHash"] == primary_evidence["rowsHash"]
    assert primary_evidence["recommendedAction"] == first.to_dict()
    assert [
        cohort["primaryDecisionScoreEvidence"]["cohortIndex"]
        for cohort in trace["cohorts"]
    ] == [0, 1, 2]
    assert [cohort["analysisAuthorityRole"] for cohort in trace["cohorts"]] == [
        "deep_confirmation", "finalist_reanalysis", "finalist_reanalysis",
    ]
    assert [cohort["traceAnalysisProfileHash"] for cohort in trace["cohorts"]] == [
        primary_profile.sha256,
        reanalysis_profile.sha256,
        reanalysis_profile.sha256,
    ]
    assert [cohort["analysisIdentityHash"] for cohort in trace["cohorts"]] == [
        primary_identity,
        reanalysis_identity,
        reanalysis_identity,
    ]
    assert [cohort["analysisBudget"] for cohort in trace["cohorts"]] == [
        primary_budget, reanalysis_budget, reanalysis_budget,
    ]
    assert [
        dict(strategy.call_kwargs[0]["analysis_budget"])
        for strategy in (primary, independent, third)
    ] == [primary_budget, reanalysis_budget, reanalysis_budget]
    assert primary.call_kwargs[0]["seed"] == primary_seed
    assert independent.call_kwargs[0]["seed"] == agreement.actor_observer_seed(
        public_prefix_hash=public_hash,
        decision_kind="play",
        acting_seat=Seat.NORTH,
        cohort_index=1,
    )
    assert third.call_kwargs[0]["seed"] == agreement.actor_observer_seed(
        public_prefix_hash=public_hash,
        decision_kind="play",
        acting_seat=Seat.NORTH,
        cohort_index=2,
    )
    assert len({
        primary.call_kwargs[0]["seed"],
        independent.call_kwargs[0]["seed"],
        third.call_kwargs[0]["seed"],
    }) == 3


def test_injected_agreement_bound_returns_original_primary_without_fallback():
    agreement = _exact_agreement(maximum=1)
    first, second = Card.parse("D10"), Card.parse("C7")
    primary = _PlayStrategy(first, _play_trace(first, second))
    independent = _PlayStrategy(second, _play_trace(second, first))
    agreement.independent_play = independent
    public_hash = canonical_sha256({"observer": "north", "state": "bounded"})
    primary_seed = agreement.actor_observer_seed(
        public_prefix_hash=public_hash,
        decision_kind="play",
        acting_seat=Seat.NORTH,
    )

    card, trace = agreement.choose_play(
        primary,
        SimpleNamespace(rules=SimpleNamespace(sha256="f" * 64)),
        SimpleNamespace(current_player=Seat.NORTH),
        (first, second),
        seed=primary_seed,
        agreement_ordinal=2,
        public_prefix_hash=public_hash,
    )

    assert card == first
    assert independent.calls == 0
    assert trace["selectedAction"] == first.to_dict()
    assert trace["qualificationBlocked"] is True
    assert trace["fallbackPolicy"] is None


@pytest.mark.parametrize("hidden_deals", [2, 5, 10])
def test_frozen_primary_accepts_complete_population_best_known_support(
    hidden_deals,
):
    agreement = _exact_agreement()
    first, second = Card.parse("D10"), Card.parse("C7")
    population = _play_trace(
        first,
        second,
        hidden_deals=hidden_deals,
        confidence=0.25,
        completion="heuristic_complete",
        exhaustive=False,
        population_complete=True,
        best_share=1.0,
    )
    primary = _PlayStrategy(first, population)
    independent = _PlayStrategy(first, population)
    third = _PlayStrategy(first, population)
    agreement.independent_play = independent
    agreement.third_play = third
    public_hash = canonical_sha256({"observer": "north", "state": "finite"})
    primary_seed = agreement.actor_observer_seed(
        public_prefix_hash=public_hash,
        decision_kind="play",
        acting_seat=Seat.NORTH,
    )

    card, trace = agreement.choose_play(
        primary,
        SimpleNamespace(rules=SimpleNamespace(sha256="f" * 64)),
        SimpleNamespace(current_player=Seat.NORTH),
        (first, second),
        seed=primary_seed,
        agreement_ordinal=1,
        public_prefix_hash=public_hash,
    )

    assert card == first
    assert trace["status"] == "primary_robust_competent"
    assert trace["qualificationBlocked"] is False
    assert trace["reasons"] == []
    assert trace["featuredTruthEligible"] is False
    assert "population_solver_best_known_only" in trace["strictReasons"]
    assert trace["truthClassByCohort"] == [
        "full_population_best_known",
        "full_population_best_known",
        "full_population_best_known",
    ]
    assert trace["populationCoverageCompleteByCohort"] == [True, True, True]
    assert trace["solverCompletionByCohort"] == [
        "heuristic_complete",
        "heuristic_complete",
        "heuristic_complete",
    ]
    assert third.calls == 1


def test_frozen_primary_keeps_best_share_gate_for_complete_population() -> None:
    agreement = _exact_agreement()
    first, second = Card.parse("D10"), Card.parse("C7")
    low_share = _play_trace(
        first,
        second,
        hidden_deals=2,
        confidence=1.0,
        completion="heuristic_complete",
        exhaustive=False,
        population_complete=True,
        best_share=0.49,
    )
    primary = _PlayStrategy(first, low_share)
    agreement.independent_play = _PlayStrategy(first, low_share)
    agreement.third_play = _PlayStrategy(first, low_share)
    public_hash = canonical_sha256({"observer": "north", "state": "low-share"})

    _card, trace = agreement.choose_play(
        primary,
        SimpleNamespace(rules=SimpleNamespace(sha256="f" * 64)),
        SimpleNamespace(current_player=Seat.NORTH),
        (first, second),
        seed=agreement.actor_observer_seed(
            public_prefix_hash=public_hash,
            decision_kind="play",
            acting_seat=Seat.NORTH,
        ),
        agreement_ordinal=1,
        public_prefix_hash=public_hash,
    )

    assert trace["status"] == "primary_robust_competent"
    assert trace["qualificationBlocked"] is False
    assert trace["reasons"] == []
    assert "low_best_share" in trace["strictReasons"]
    assert "population_solver_best_known_only" in trace["strictReasons"]


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"maximum_cohorts": 4}, "cohort bound"),
        ({"minimum_agreement": 3, "maximum_cohorts": 2}, "threshold"),
        ({"third_cohort_on_disagreement": False}, "adaptive third"),
        ({"maximum_competence_expected_margin_regret": -0.1}, "competence regret"),
        ({"maximum_competence_match_win_drop": 1.1}, "match-win drop"),
    ],
)
def test_invalid_cohort_configuration_is_rejected(override, message):
    stability = {
        "reanalysis_samples": 256,
        "maximum_sampling_attempts": 6000,
        "exact_allocation_limit": 256,
        "exact_search_remaining": 9,
        "exact_search_max_nodes": 120000,
        "exact_search_max_seconds": 0.8,
        "seed_offset": 104729,
        "minimum_hidden_deals": 8,
        "minimum_best_share": 0.5,
        "maximum_competence_expected_margin_regret": 1.0,
        "maximum_competence_match_win_drop": 0.01,
        "maximum_expected_margin_drift": 5.0,
        "maximum_cohorts": 3,
        "minimum_agreement": 2,
        "third_cohort_on_disagreement": True,
        "require_same_recommended_action": True,
    }
    stability.update(override)

    with pytest.raises(ValueError, match=message):
        FinalistAgentAgreementProfileV1.build(
            stability,
            base_profile=_agent(),
            maximum_decisions_per_trace=64,
            minimum_confidence=0.75,
        )


def test_all_seats_receive_identical_stable_agreement_policy():
    agreement = _agreement()
    first, second = Card.parse("D10"), Card.parse("C7")
    primary = _PlayStrategy(first, _play_trace(first, second))
    independent = _PlayStrategy(first, _play_trace(first, second))
    agreement.independent_play = independent

    outputs = []
    for seat in Seat:
        card, trace = agreement.choose_play(
            primary,
            SimpleNamespace(),
            SimpleNamespace(current_player=seat),
            (first, second),
            seed=100 + int(seat),
            agreement_ordinal=1,
        )
        outputs.append((card, trace["status"], trace["qualificationBlocked"]))

    assert outputs == [(first, "strict_confirmed", False)] * 4
    assert primary.calls == 4
    assert independent.calls == 4


def test_play_confidence_uses_canonical_ess8_and_matches_replay_verdict():
    first, second = Card.parse("D10"), Card.parse("C7")
    display_high = _play_trace(
        first,
        second,
        hidden_deals=8,
        confidence=1.0,
        completion="budget_exhausted",
        exhaustive=False,
    )
    drifted = _agreement()
    drifted.independent_play = _PlayStrategy(first, display_high)
    drifted.third_play = _PlayStrategy(first, display_high)
    with pytest.raises(ValueError, match="confidence drifts from canonical score rows"):
        drifted.choose_play(
            _PlayStrategy(
                first,
                display_high,
                canonicalize_confidence=False,
            ),
            SimpleNamespace(),
            SimpleNamespace(),
            (first, second),
            seed=18,
            agreement_ordinal=1,
        )

    agreement = _agreement()
    agreement.independent_play = _PlayStrategy(first, display_high)
    agreement.third_play = _PlayStrategy(first, display_high)
    _card, trace = agreement.choose_play(
        _PlayStrategy(first, display_high),
        SimpleNamespace(),
        SimpleNamespace(),
        (first, second),
        seed=18,
        agreement_ordinal=1,
    )

    expected_confidence = (8.0**0.5) / 10.0
    assert all(
        item["confidence"] == expected_confidence for item in trace["cohorts"]
    )
    assert trace["status"] == "unresolved"
    assert trace["featuredTruthEligible"] is False
    replay = _canonical_replay_verdict(agreement, trace)
    common = (
        "status",
        "reasons",
        "strictReasons",
        "initialReasons",
        "agreementCount",
        "majorityActionHash",
        "featuredTruthEligible",
        "surroundingVerified",
    )
    assert canonical_json_bytes_v1({key: trace[key] for key in common}) == (
        canonical_json_bytes_v1({key: replay[key] for key in common})
    )


def test_bid_confidence_uses_canonical_ess36_and_matches_replay_verdict():
    passing, sun = BidChoice("pass"), BidChoice("sun")
    display_low = _bid_trace(
        sun,
        passing,
        hidden_deals=36,
        confidence=0.25,
    )
    drifted = _agreement()
    drifted.independent_bidding = _BidStrategy(sun, display_low)
    with pytest.raises(ValueError, match="confidence drifts from canonical score rows"):
        drifted.choose_bid(
            _BidStrategy(
                sun,
                display_low,
                canonicalize_confidence=False,
            ),
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
            (passing, sun),
            seed=36,
            agreement_ordinal=1,
        )

    agreement = _agreement()
    agreement.independent_bidding = _BidStrategy(sun, display_low)
    _action, _trump, trace = agreement.choose_bid(
        _BidStrategy(sun, display_low),
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        (passing, sun),
        seed=36,
        agreement_ordinal=1,
    )

    assert all(item["confidence"] == 0.75 for item in trace["cohorts"])
    assert trace["status"] == "strict_confirmed"
    assert trace["featuredTruthEligible"] is True
    replay = _canonical_replay_verdict(agreement, trace)
    common = (
        "status",
        "reasons",
        "strictReasons",
        "initialReasons",
        "agreementCount",
        "majorityActionHash",
        "featuredTruthEligible",
        "surroundingVerified",
    )
    assert canonical_json_bytes_v1({key: trace[key] for key in common}) == (
        canonical_json_bytes_v1({key: replay[key] for key in common})
    )


def test_reversal_retains_primary_and_blocks_qualification_without_fallback():
    agreement = _agreement()
    passing = BidChoice("pass")
    sun = BidChoice("sun")
    hokum = BidChoice("hokum", Suit.SPADES)
    primary = _BidStrategy(sun, _bid_trace(sun, hokum))
    agreement.independent_bidding = _BidStrategy(hokum, _bid_trace(hokum, sun))
    agreement.third_bidding = _BidStrategy(passing, _bid_trace(passing, sun))

    action, trump, trace = agreement.choose_bid(
        primary,
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        (passing, sun, hokum),
        seed=31,
        agreement_ordinal=1,
    )

    assert (action, trump) == ("sun", None)
    assert trace["selectedAction"] == sun.to_dict()
    assert trace["status"] == "unresolved"
    assert "cohort_reversal" in trace["reasons"]
    assert "insufficient_qualified_agreement" in trace["reasons"]
    assert trace["qualificationBlocked"] is True
    assert trace["fallbackPolicy"] is None
    assert trace["cohortCount"] == 3


def test_third_cohort_resolves_two_of_three_without_weakening_support_gates():
    agreement = _agreement()
    passing = BidChoice("pass")
    sun = BidChoice("sun")
    hokum = BidChoice("hokum", Suit.SPADES)
    primary = _BidStrategy(sun, _bid_trace(sun, hokum))
    agreement.independent_bidding = _BidStrategy(hokum, _bid_trace(hokum, sun))
    agreement.third_bidding = _BidStrategy(sun, _bid_trace(sun, passing))

    action, trump, trace = agreement.choose_bid(
        primary,
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        (passing, sun, hokum),
        seed=37,
        agreement_ordinal=1,
    )

    assert (action, trump) == ("sun", None)
    assert trace["status"] == "strict_confirmed"
    assert trace["agreementCount"] == 2
    assert trace["qualificationBlocked"] is False
    assert trace["initialReasons"] == [
        "cohort_reversal",
        "insufficient_qualified_agreement",
    ]


def test_low_support_budget_exhaustion_is_unresolved():
    agreement = _agreement()
    first, second = Card.parse("D10"), Card.parse("C7")
    low = _play_trace(
        first, second, hidden_deals=6, confidence=0.24,
        completion="budget_exhausted",
        exhaustive=False,
    )
    primary = _PlayStrategy(first, low)
    agreement.independent_play = _PlayStrategy(first, low)
    agreement.third_play = _PlayStrategy(first, low)

    _card, trace = agreement.choose_play(
        primary,
        SimpleNamespace(),
        SimpleNamespace(),
        (first, second),
        seed=41,
        agreement_ordinal=1,
    )

    assert trace["qualificationBlocked"] is True
    assert {
        "analysis_not_converged",
        "budget_exhausted_low_support",
        "insufficient_qualified_agreement",
        "low_confidence",
        "low_hidden_deal_support",
    } <= set(trace["reasons"])


def test_budget_exhaustion_requires_stable_independent_competence():
    agreement = _agreement()
    first, second = Card.parse("D10"), Card.parse("C7")
    unconverged = _play_trace(
        first,
        second,
        hidden_deals=192,
        confidence=1.0,
        completion="budget_exhausted",
        exhaustive=False,
    )
    primary = _PlayStrategy(first, unconverged)
    agreement.independent_play = _PlayStrategy(first, unconverged)
    agreement.third_play = _PlayStrategy(first, unconverged)

    _card, trace = agreement.choose_play(
        primary,
        SimpleNamespace(),
        SimpleNamespace(),
        (first, second),
        seed=42,
        agreement_ordinal=1,
    )

    assert trace["qualificationBlocked"] is False
    assert trace["status"] == "primary_robust_competent"
    assert trace["competenceEvidence"]["agreementCount"] == 3
    assert {
        "analysis_not_converged",
        "insufficient_qualified_agreement",
    } <= set(trace["competenceEvidence"]["strictReasons"])


def test_value_equivalent_low_best_share_is_robust_competent():
    agreement = _agreement()
    first, second = Card.parse("D10"), Card.parse("C7")
    first_line = _play_trace(
        first,
        second,
        best_share=0.4,
        completion="budget_exhausted",
        exhaustive=False,
        best_margin=5.0,
        other_margin=4.7,
        best_win=0.600,
        other_win=0.595,
    )
    second_line = _play_trace(
        second,
        first,
        best_share=0.4,
        completion="budget_exhausted",
        exhaustive=False,
        best_margin=5.0,
        other_margin=4.7,
        best_win=0.600,
        other_win=0.595,
    )
    primary = _PlayStrategy(first, first_line)
    agreement.independent_play = _PlayStrategy(second, second_line)
    agreement.third_play = _PlayStrategy(first, first_line)

    card, trace = agreement.choose_play(
        primary,
        SimpleNamespace(),
        SimpleNamespace(),
        (first, second),
        seed=44,
        agreement_ordinal=1,
    )

    assert card == first
    assert trace["status"] == "primary_robust_competent"
    assert trace["qualificationBlocked"] is False
    assert trace["competenceEvidence"]["agreementCount"] == 3


@pytest.mark.parametrize(
    ("agreement_overrides", "secondary_overrides", "expected_reason"),
    [
        (
            {},
            {
                "best_margin": 5.0,
                "other_margin": 3.5,
                "best_win": 0.60,
                "other_win": 0.60,
            },
            "competence_regret",
        ),
        (
            {},
            {
                "best_margin": 5.0,
                "other_margin": 4.5,
                "best_win": 0.60,
                "other_win": 0.58,
            },
            "competence_match_win_drop",
        ),
        (
            {"maximum_competence_match_win_drop": 1.0},
            {
                "best_margin": 0.5,
                "other_margin": -0.1,
                "best_win": 0.60,
                "other_win": 0.49,
            },
            "failure_boundary_crossing",
        ),
        (
            {"maximum_expected_margin_drift": 0.1},
            {
                "best_margin": 5.0,
                "other_margin": 4.5,
                "best_win": 0.60,
                "other_win": 0.60,
            },
            "competence_expected_margin_drift",
        ),
    ],
)
def test_competence_boundaries_block_without_replacing_primary(
    agreement_overrides,
    secondary_overrides,
    expected_reason,
):
    agreement = _agreement(**agreement_overrides)
    primary_card, alternate = Card.parse("D10"), Card.parse("C7")
    primary = _PlayStrategy(
        primary_card,
        _play_trace(primary_card, alternate),
    )
    alternate_trace = _play_trace(
        alternate,
        primary_card,
        **secondary_overrides,
    )
    agreement.independent_play = _PlayStrategy(alternate, alternate_trace)
    agreement.third_play = _PlayStrategy(alternate, alternate_trace)

    card, trace = agreement.choose_play(
        primary,
        SimpleNamespace(),
        SimpleNamespace(),
        (primary_card, alternate),
        seed=4701,
        agreement_ordinal=1,
    )

    assert card == primary_card
    assert trace["selectedAction"] == primary_card.to_dict()
    assert trace["fallbackPolicy"] is None
    assert trace["status"] == "unresolved"
    assert trace["qualificationBlocked"] is True
    assert expected_reason in trace["reasons"]
    assert trace["competenceEvidence"]["primaryActionHash"] == canonical_sha256(
        primary_card.to_dict()
    )


def test_disjoint_primary_value_intervals_are_unresolved_without_replacement():
    agreement = _agreement()
    primary_card, alternate = Card.parse("D10"), Card.parse("C7")
    primary = _PlayStrategy(
        primary_card,
        _play_trace(
            primary_card,
            alternate,
            best_interval=(0.0, 0.4),
        ),
    )
    alternate_trace = _play_trace(
        alternate,
        primary_card,
        best_margin=5.0,
        other_margin=4.7,
        best_win=0.600,
        other_win=0.595,
        other_interval=(0.6, 1.0),
    )
    agreement.independent_play = _PlayStrategy(alternate, alternate_trace)
    agreement.third_play = _PlayStrategy(alternate, alternate_trace)

    card, trace = agreement.choose_play(
        primary,
        SimpleNamespace(),
        SimpleNamespace(),
        (primary_card, alternate),
        seed=4702,
        agreement_ordinal=1,
    )

    assert card == primary_card
    assert trace["selectedAction"] == primary_card.to_dict()
    assert trace["status"] == "unresolved"
    assert "competence_value_interval_instability" in trace["reasons"]
    assert trace["competenceEvidence"]["valueIntervalsStable"] is False


def test_missing_canonical_score_evidence_is_unresolved_and_retains_primary():
    agreement = _agreement()
    primary_card, alternate = Card.parse("D10"), Card.parse("C7")
    trace_without_scores = _play_trace(primary_card, alternate)
    primary = _PlayStrategy(
        primary_card,
        trace_without_scores,
        emit_score_evidence=False,
    )
    agreement.independent_play = _PlayStrategy(
        primary_card,
        trace_without_scores,
        emit_score_evidence=False,
    )
    agreement.third_play = _PlayStrategy(
        primary_card,
        trace_without_scores,
        emit_score_evidence=False,
    )

    card, trace = agreement.choose_play(
        primary,
        SimpleNamespace(),
        SimpleNamespace(),
        (primary_card, alternate),
        seed=4703,
        agreement_ordinal=1,
    )

    assert card == primary_card
    assert trace["selectedAction"] == primary_card.to_dict()
    assert trace["status"] == "unresolved"
    assert trace["fallbackPolicy"] is None
    assert "missing_primary_decision_score_evidence" in trace["reasons"]


def test_loose_support_mirror_drift_cannot_override_canonical_score_evidence():
    agreement = _agreement()
    primary_card, alternate = Card.parse("D10"), Card.parse("C7")
    drifted = _play_trace(primary_card, alternate)
    drifted["analysisTruthClass"] = "unresolved"
    primary = _PlayStrategy(primary_card, drifted)
    agreement.independent_play = _PlayStrategy(primary_card, drifted)
    agreement.third_play = _PlayStrategy(primary_card, drifted)

    card, trace = agreement.choose_play(
        primary,
        SimpleNamespace(),
        SimpleNamespace(),
        (primary_card, alternate),
        seed=4704,
        agreement_ordinal=1,
    )

    assert card == primary_card
    assert trace["selectedAction"] == primary_card.to_dict()
    assert trace["status"] == "unresolved"
    assert "invalid_support_truth_classification" in trace["reasons"]


@pytest.mark.parametrize("hidden_deals", [1, 6])
def test_exhaustive_exact_population_is_supported_even_when_small(hidden_deals):
    agreement = _agreement()
    first, second = Card.parse("D10"), Card.parse("C7")
    exact = _play_trace(
        first,
        second,
        hidden_deals=hidden_deals,
        confidence=0.1,
        completion="exact",
        exhaustive=True,
    )
    primary = _PlayStrategy(first, exact)
    independent = _PlayStrategy(first, exact)
    third = _PlayStrategy(second, _play_trace(second, first))
    agreement.independent_play = independent
    agreement.third_play = third

    card, trace = agreement.choose_play(
        primary,
        SimpleNamespace(),
        SimpleNamespace(),
        (first, second),
        seed=43,
        agreement_ordinal=1,
    )

    assert card == first
    assert trace["status"] == "strict_confirmed"
    assert trace["cohortCount"] == 2
    assert trace["qualificationBlocked"] is False
    assert third.calls == 0


@pytest.mark.parametrize("hidden_deals", [2, 10, 33])
def test_complete_population_heuristic_is_robust_agreement_but_not_exact_truth(
    hidden_deals,
):
    agreement = _agreement()
    first, second = Card.parse("D10"), Card.parse("C7")
    population = _play_trace(
        first,
        second,
        hidden_deals=hidden_deals,
        confidence=0.1,
        completion="heuristic_complete",
        exhaustive=False,
        population_complete=True,
        best_margin=5.0,
        other_margin=4.7,
        best_win=0.600,
        other_win=0.595,
    )
    primary = _PlayStrategy(first, population)
    agreement.independent_play = _PlayStrategy(first, population)
    agreement.third_play = _PlayStrategy(first, population)

    card, trace = agreement.choose_play(
        primary,
        SimpleNamespace(),
        SimpleNamespace(),
        (first, second),
        seed=45,
        agreement_ordinal=1,
    )

    assert card == first
    assert trace["status"] == "primary_robust_competent"
    assert trace["qualificationBlocked"] is False
    assert trace["featuredTruthEligible"] is False
    assert "population_solver_best_known_only" in trace["strictReasons"]
    assert trace["truthClassByCohort"] == [
        "full_population_best_known",
    ] * 3
    assert trace["populationCoverageCompleteByCohort"] == [True, True, True]
    assert trace["solverCompletionByCohort"] == [
        "heuristic_complete",
        "heuristic_complete",
        "heuristic_complete",
    ]


def test_sampled_low_confidence_cannot_use_population_competence():
    agreement = _agreement()
    first, second = Card.parse("D10"), Card.parse("C7")
    sampled = _play_trace(
        first,
        second,
        hidden_deals=33,
        confidence=0.1,
        completion="budget_exhausted",
        exhaustive=False,
    )
    primary = _PlayStrategy(first, sampled)
    agreement.independent_play = _PlayStrategy(first, sampled)
    agreement.third_play = _PlayStrategy(first, sampled)

    _card, trace = agreement.choose_play(
        primary,
        SimpleNamespace(),
        SimpleNamespace(),
        (first, second),
        seed=46,
        agreement_ordinal=1,
    )

    assert trace["status"] == "unresolved"
    assert trace["qualificationBlocked"] is True
    assert "low_confidence" in trace["reasons"]


def test_cancellation_stops_before_independent_cohort():
    agreement = _agreement()
    first, second = Card.parse("D10"), Card.parse("C7")
    primary = _PlayStrategy(first, _play_trace(first, second))
    independent = _PlayStrategy(first, _play_trace(first, second))
    agreement.independent_play = independent

    with pytest.raises(InterruptedError, match="agreement cancelled"):
        agreement.choose_play(
            primary,
            SimpleNamespace(),
            SimpleNamespace(),
            (first, second),
            seed=51,
            agreement_ordinal=1,
            cancel_check=lambda: True,
        )

    assert independent.calls == 0


def test_cancellation_inside_independent_bid_analysis_propagates():
    agreement = _agreement()
    passing = BidChoice("pass")
    sun = BidChoice("sun")
    primary = _BidStrategy(sun, _bid_trace(sun, passing))
    agreement.independent_bidding = _InterruptingBidStrategy()

    with pytest.raises(InterruptedError, match="bid analysis cancelled"):
        agreement.choose_bid(
            primary,
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
            (passing, sun),
            seed=55,
            agreement_ordinal=1,
        )


def test_cancellation_inside_adaptive_third_bid_analysis_propagates():
    agreement = _agreement()
    passing = BidChoice("pass")
    sun = BidChoice("sun")
    hokum = BidChoice("hokum", Suit.SPADES)
    primary = _BidStrategy(sun, _bid_trace(sun, hokum))
    agreement.independent_bidding = _BidStrategy(
        hokum, _bid_trace(hokum, sun)
    )
    agreement.third_bidding = _InterruptingBidStrategy()

    with pytest.raises(InterruptedError, match="bid analysis cancelled"):
        agreement.choose_bid(
            primary,
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
            (passing, sun, hokum),
            seed=57,
            agreement_ordinal=1,
        )


def test_runtime_bound_skips_independent_work_and_records_unresolved_fallback():
    agreement = _agreement(maximum=1)
    first, second = Card.parse("D10"), Card.parse("C7")
    primary = _PlayStrategy(first, _play_trace(first, second))
    independent = _PlayStrategy(first, _play_trace(first, second))
    agreement.independent_play = independent

    _card, trace = agreement.choose_play(
        primary,
        SimpleNamespace(),
        SimpleNamespace(),
        (first, second),
        seed=61,
        agreement_ordinal=2,
    )

    assert independent.calls == 0
    assert trace["agreementBoundExceeded"] is True
    assert trace["reasons"] == ["agreement_bound_exceeded"]
    assert trace["qualificationBlocked"] is True


def test_simulator_freezes_agreement_profile_and_unresolved_status_in_trace(root):
    bundle = build_bundle(root)
    base = replace(
        AgentProfileRegistry.from_bundle(root).get("finalist_candidate"),
        play_search_mode="disabled",
        play_search_budget=1,
        bid_samples=2,
    )
    stability = {
        "reanalysis_samples": 2,
        "maximum_sampling_attempts": 8,
        "exact_allocation_limit": 2,
        "exact_search_remaining": 1,
        "exact_search_max_nodes": 1,
        "exact_search_max_seconds": 0.01,
        "seed_offset": 104729,
        "minimum_hidden_deals": 2,
        "minimum_best_share": 0.5,
        "maximum_competence_expected_margin_regret": 1.0,
        "maximum_competence_match_win_drop": 0.01,
        "maximum_expected_margin_drift": 5.0,
        "maximum_cohorts": 3,
        "minimum_agreement": 2,
        "third_cohort_on_disagreement": True,
        "require_same_recommended_action": True,
    }
    agreement = FinalistAgentAgreementV1(
        base,
        stability,
        maximum_decisions_per_trace=64,
        minimum_confidence=0.75,
        convention=bundle.convention,
    )
    simulator = NaturalLaffaSimulatorV1(
        bundle.rules,
        bundle.convention,
        bundle_hash=bundle.sha256,
        configuration_snapshot=bundle.configuration_snapshot(),
        agent_profile=base,
        maximum_deal_attempts=1,
        decision_agreement=agreement,
    )

    first = simulator.simulate(73)
    second = simulator.simulate(73)
    trace = first.final_trace
    generation = dict(trace.trusted_final_state["generation"])
    agreement_state = dict(trace.trusted_final_state["decisionAgreement"])

    assert first.sha256 == second.sha256
    assert generation["decisionAgreementProfileHash"] == agreement.profile.sha256
    assert generation["decisionAgreementDecisionCount"] >= 1
    assert generation["decisionAgreementUnresolvedCount"] >= 1
    assert agreement_state["profile"]["sha256"] == agreement.profile.sha256
    assert agreement_state["qualifiedAtGeneration"] is False
    policies = {
        int(item["ordinal"]): dict(item["trace"])
        for item in trace.trusted_final_state["decisionPolicyTraces"]
    }
    assert any(item.get("qualificationBlocked") for item in policies.values())
    assert all(
        dict(policies[item.ordinal]["selectedAction"]) == dict(item.chosen_action)
        for item in trace.decision_ledger.records
    )


def test_simulator_retains_frozen_analysis_hashes_and_actor_state_seeds(root):
    bundle = build_bundle(root)
    base = replace(
        AgentProfileRegistry.from_bundle(root).get("finalist_candidate"),
        play_search_budget=1,
        play_search_samples=2,
        play_exact_allocation_limit=2,
        play_exact_search_remaining=1,
        play_maximum_sampling_attempts=8,
        bid_samples=2,
    )
    trace_profile = TraceAnalysisProfileV1.build(
        bid_samples=2,
        play_samples=2,
        exact_allocation_limit=2,
        exact_search_max_remaining=1,
        maximum_sampling_attempts=8,
        exact_search_max_nodes=1,
        exact_search_max_seconds=0.01,
        exact_search_max_depth=1,
        exact_search_max_cache_entries=0,
        objective="match_win_probability",
        use_conventions=False,
    )
    stability = {
        "reanalysis_samples": 3,
        "maximum_sampling_attempts": 8,
        "exact_allocation_limit": 2,
        "exact_search_remaining": 1,
        "exact_search_max_nodes": 1,
        "exact_search_max_seconds": 0.01,
        "seed_offset": 104729,
        "minimum_hidden_deals": 2,
        "minimum_best_share": 0.5,
        "maximum_competence_expected_margin_regret": 1.0,
        "maximum_competence_match_win_drop": 0.01,
        "maximum_expected_margin_drift": 5.0,
        "maximum_cohorts": 3,
        "minimum_agreement": 2,
        "third_cohort_on_disagreement": True,
        "require_same_recommended_action": True,
    }
    reanalysis_profile = TraceAnalysisProfileV1.build(
        bid_samples=3,
        play_samples=3,
        exact_allocation_limit=2,
        exact_search_max_remaining=1,
        maximum_sampling_attempts=8,
        exact_search_max_nodes=1,
        exact_search_max_seconds=0.01,
        exact_search_max_depth=1,
        exact_search_max_cache_entries=0,
        search_backend_id=trace_profile.search_backend_id,
        objective=trace_profile.objective,
        use_conventions=False,
    )
    identity_hash = _analysis_identity("deep_confirmation", trace_profile)
    reanalysis_identity_hash = _analysis_identity(
        "finalist_reanalysis", reanalysis_profile
    )
    agreement = FinalistAgentAgreementV1(
        base,
        stability,
        maximum_decisions_per_trace=64,
        minimum_confidence=0.75,
        convention=bundle.convention,
        primary_trace_analysis_profile=trace_profile,
        primary_analysis_identity_hash=identity_hash,
        reanalysis_trace_analysis_profile=reanalysis_profile,
        reanalysis_analysis_identity_hash=reanalysis_identity_hash,
    )
    with pytest.raises(ValueError, match="supplied together"):
        NaturalLaffaSimulatorV1(
            bundle.rules,
            bundle.convention,
            bundle_hash=bundle.sha256,
            configuration_snapshot=bundle.configuration_snapshot(),
            agent_profile=base,
            maximum_deal_attempts=1,
            trace_analysis_profile=trace_profile,
        )
    simulator = NaturalLaffaSimulatorV1(
        bundle.rules,
        bundle.convention,
        bundle_hash=bundle.sha256,
        configuration_snapshot=bundle.configuration_snapshot(),
        agent_profile=base,
        maximum_deal_attempts=1,
        decision_agreement=agreement,
        trace_analysis_profile=trace_profile,
        analysis_identity_hash=identity_hash,
    )

    trace = simulator.simulate(73).final_trace
    generation = dict(trace.trusted_final_state["generation"])
    policies = {
        int(item["ordinal"]): dict(item["trace"])
        for item in trace.trusted_final_state["decisionPolicyTraces"]
    }

    assert generation["traceAnalysisProfileHash"] == trace_profile.sha256
    assert generation["analysisIdentityHash"] == identity_hash
    assert trace.trusted_final_state["decisionAgreement"][
        "traceAnalysisProfileHash"
    ] == trace_profile.sha256
    assert trace.trusted_final_state["decisionAgreement"][
        "analysisIdentityHash"
    ] == identity_hash
    assert agreement.profile.primary_trace_analysis_profile_hash == (
        trace_profile.sha256
    )
    assert agreement.profile.reanalysis_trace_analysis_profile_hash == (
        reanalysis_profile.sha256
    )
    assert agreement.profile.primary_analysis_identity_hash == identity_hash
    assert agreement.profile.reanalysis_analysis_identity_hash == (
        reanalysis_identity_hash
    )
    assert policies
    for record in trace.decision_ledger.records:
        policy = policies[record.ordinal]
        assert policy["traceAnalysisProfileHash"] == trace_profile.sha256
        assert policy["analysisIdentityHash"] == identity_hash
        expected_seed = simulator.primary_decision_seed(
            public_prefix_hash=record.public_prefix_hash,
            decision_kind=record.decision_kind,
            acting_seat=record.acting_seat,
        )
        assert int(policy.get("primarySeed", policy.get("seed"))) == expected_seed


def test_independent_play_uses_exact_governed_stability_budget(
    root, monkeypatch,
):
    bundle = build_bundle(root)
    engine = BalootEngine(bundle.rules)
    state = engine.create_round(seed=812, dealer=Seat.EAST)
    engine.shuffle_and_initial_deal(state)
    engine.bid_and_complete_deal(
        state,
        forced_contract=Contract.SUN,
        forced_buyer=Seat.SOUTH,
    )
    engine.prepare_projects(state)
    legal = engine.legal_moves(state)
    captured = {}

    def fake_analyze(_info, _rules, _convention, **kwargs):
        captured.update(kwargs)
        action = SimpleNamespace(
            card=legal[0],
            expected_match_margin=1.0,
            win_probability=0.5,
            downside_cvar=0.0,
            best_share=1.0,
            exact_share=0.0,
            search_completion_status="heuristic_complete",
        )
        return SimpleNamespace(
            actions=(action,),
            method="test_search",
            effective_sample_size=256.0,
            hidden_deals=256,
        )

    monkeypatch.setattr("baloot_platform.agents.profiles.analyze", fake_analyze)
    agreement = FinalistAgentAgreementV1(
        _agent(),
        {
            "reanalysis_samples": 256,
            "maximum_sampling_attempts": 6000,
            "exact_allocation_limit": 256,
            "exact_search_remaining": 9,
            "exact_search_max_nodes": 120000,
            "exact_search_max_seconds": 0.8,
            "seed_offset": 104729,
            "minimum_hidden_deals": 8,
            "minimum_best_share": 0.5,
            "maximum_competence_expected_margin_regret": 1.0,
            "maximum_competence_match_win_drop": 0.01,
            "maximum_expected_margin_drift": 5.0,
            "maximum_cohorts": 3,
            "minimum_agreement": 2,
            "third_cohort_on_disagreement": True,
            "require_same_recommended_action": True,
        },
        maximum_decisions_per_trace=64,
        minimum_confidence=0.75,
        convention=bundle.convention,
    )

    agreement.independent_play.choose_with_trace(
        engine,
        state,
        legal,
        seed=91,
        analysis_budget=agreement.independent_analysis_budget,
    )

    assert captured["requested_samples"] == 256
    assert captured["maximum_sampling_attempts"] == 6000
    assert captured["exact_allocation_limit"] == 256
    assert captured["exact_search_max_remaining"] == 9
    assert captured["exact_search_max_nodes"] == 120000
    assert captured["exact_search_max_seconds"] == 0.8
