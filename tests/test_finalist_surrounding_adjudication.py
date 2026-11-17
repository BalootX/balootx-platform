import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from baloot_platform.agents.profiles import AgentProfile
from baloot_platform.analysis.decision_scores import (
    canonical_sha256_v1,
    primary_decision_score_evidence_v1,
)
from baloot_platform.challenge.analysis_pipeline import (
    CascadedTraceAnalysisProfileV1,
    TraceDecisionAnalysisV3,
)
from baloot_platform.challenge.evidence_v3 import canonical_sha256
from baloot_platform.challenge.finalist_agent_agreement import (
    FinalistAgentAgreementV1,
)
from baloot_platform.challenge.finalist_adjudication import (
    AdaptiveFinalistTraceAnalyzerV1,
    FinalistSurroundingAdjudicationProfileV1,
    frozen_adaptive_finalist_profile_hash_v1,
)
from baloot_platform.challenge.funnel import FoundryAnalysisFunnelV1
from baloot_platform.challenge.profiles import SelectionProfileV1
from baloot_platform.core.cards import Card
from baloot_platform.service.container import build_container


def test_frozen_trace_rebuilds_exact_adaptive_finalist_profile(root, tmp_path):
    container = build_container(root=root, writable_root=tmp_path / "runtime")
    try:
        trace = container.finalist_replay.simulator.simulate(19).final_trace
        expected = container.finalist_validations.funnel.analyzer.profile.sha256

        assert frozen_adaptive_finalist_profile_hash_v1(trace) == expected

        trusted_state = copy.deepcopy(trace.trusted_final_state)
        trusted_state["generation"]["traceAnalysisProfileHash"] = "0" * 64
        substituted = replace(trace, trusted_final_state=trusted_state)
        with pytest.raises(ValueError, match="not uniquely governed"):
            frozen_adaptive_finalist_profile_hash_v1(substituted)
    finally:
        container.shutdown()


def _selection() -> SelectionProfileV1:
    return SelectionProfileV1.build({
        "schemaVersion": 1,
        "id": "strict-finalist",
        "revision": "1",
        "weights": {"outcomeImpact": 1, "robustness": 1},
        "allowedTruthLabels": ["exact_optimal", "robust_best_known"],
        "minimumConfidence": 0.75,
        "minimumScore": 50,
        "minimumSurroundingQuality": 75,
        "minimumSurroundingConfidence": 0.75,
        "minimumSurroundingMateriality": 25,
        "maximumSurroundingMaterialMistakes": 0,
        "maximumSurroundingUnverifiedDecisions": 0,
        "maximumSurroundingCatastrophicCrossings": 0,
    })


def _analysis(
    decision_id: str,
    *,
    best: str,
    actual: str = "D10",
    hidden_deals: int = 192,
    confidence: float = 1.0,
    regret: float = 0.0,
    drop: float = 0.0,
    catastrophic: bool = False,
    best_margin: float = 5.0,
    completion: str = "exact",
    exact_share: float = 1.0,
    method: str = "exhaustive_constrained_hidden_deals+hard_facts_only",
    best_share: float = 0.7,
    confidence_interval: tuple[float, float] | None = None,
    ordinal: int = 1,
    cohort_index: int = 0,
    cohort_seed: int | None = None,
    alternative_margin_shift: float = 0.0,
    include_score_evidence: bool = True,
) -> TraceDecisionAnalysisV3:
    best_action = Card.parse(best).to_dict()
    actual_action = Card.parse(actual).to_dict()
    other_action = (
        actual_action
        if actual_action != best_action
        else Card.parse("C7" if best != "C7" else "D10").to_dict()
    )
    best_probability = 0.6
    other_probability = best_probability - max(0.0, drop)
    interval = list(confidence_interval or (0.55, 0.65))
    rows = (
        {
            "rank": 1,
            "actionId": f"card:{Card.parse(best_action).id}",
            "action": best_action,
            "expectedRoundMargin": best_margin,
            "expectedTeamAwardedPoints": 100.0 + best_margin,
            "expectedOpponentAwardedPoints": 100.0,
            "roundLeadProbability": best_probability,
            "drawProbability": 0.0,
            "roundLeadConfidence95": interval,
            "bestShare": best_share,
            "samples": hidden_deals,
            "effectiveSamples": float(hidden_deals),
            "exactShare": exact_share,
            "contractSuccessProbability": best_probability,
            "matchWinProbability": best_probability,
            "matchWinConfidence95": interval,
            "matchValueModel": "test_match_value_v1",
            "supportMarginsHash": canonical_sha256_v1([best_margin]),
            "supportMarginCount": 1,
            "downsideCvar10": best_margin,
            "volatility": 0.0,
            "principalVariationHash": canonical_sha256_v1([]),
            "principalVariationLength": 0,
            "searchNodes": 1,
            "searchCompletionStatus": completion,
        },
        {
            "rank": 2,
            "actionId": f"card:{Card.parse(other_action).id}",
            "action": other_action,
            "expectedRoundMargin": best_margin - max(0.0, regret),
            "expectedTeamAwardedPoints": (
                100.0 + best_margin - max(0.0, regret)
            ),
            "expectedOpponentAwardedPoints": 100.0,
            "roundLeadProbability": other_probability,
            "drawProbability": 0.0,
            "roundLeadConfidence95": interval,
            "bestShare": 1.0 - best_share,
            "samples": hidden_deals,
            "effectiveSamples": float(hidden_deals),
            "exactShare": exact_share,
            "contractSuccessProbability": other_probability,
            "matchWinProbability": other_probability,
            "matchWinConfidence95": interval,
            "matchValueModel": "test_match_value_v1",
            "supportMarginsHash": canonical_sha256_v1(
                [best_margin - max(0.0, regret)]
            ),
            "supportMarginCount": 1,
            "downsideCvar10": best_margin - max(0.0, regret),
            "volatility": 0.0,
            "principalVariationHash": canonical_sha256_v1([]),
            "principalVariationLength": 0,
            "searchNodes": 1,
            "searchCompletionStatus": completion,
        },
    )
    score_evidence = (
        primary_decision_score_evidence_v1(
            decision_kind="play",
            actor="east",
            public_prefix_hash="4" * 64,
            rules_hash="3" * 64,
            analysis_identity_hash=("a" if cohort_index == 0 else "b") * 64,
            trace_analysis_profile_hash=(
                ("1" if cohort_index == 0 else "2") * 64
            ),
            cohort_seed_policy="actor_observer_cohort_seed_v1",
            cohort_index=cohort_index,
            cohort_seed=(1000 + cohort_index if cohort_seed is None else cohort_seed),
            objective="match_win_probability",
            tie_break="test_tie_break_v1",
            backend="test_reference_backend_v1",
            method=method,
            hidden_deal_support=hidden_deals,
            effective_sample_size=float(hidden_deals),
            recommended_action=best_action,
            rows=rows,
        )
        if include_score_evidence
        else None
    )
    alternatives = (
        {
            "action": best_action,
            "rank": 1,
            "expectedMargin": best_margin + alternative_margin_shift,
            "winProbability": 0.6,
            "matchWinProbability": 0.6,
            "bestShare": best_share,
            "searchCompletionStatus": completion,
            "exactShare": exact_share,
            "confidence95": confidence_interval,
        },
        {
            "action": other_action,
            "rank": 2,
            "expectedMargin": (
                best_margin - max(0.0, regret) - alternative_margin_shift
            ),
            "winProbability": 0.6 - max(0.0, drop),
            "matchWinProbability": 0.6 - max(0.0, drop),
            "bestShare": 1.0 - best_share,
            "searchCompletionStatus": completion,
            "exactShare": exact_share,
            "confidence95": confidence_interval,
        },
    )
    core = {
        "schemaVersion": 3,
        "decisionId": decision_id,
        "ordinal": ordinal,
        "actingSeat": "east",
        "decisionKind": "play",
        "forced": False,
        "actualAction": actual_action,
        "bestAction": best_action,
        "alternatives": [dict(item) for item in alternatives],
        "method": method,
        "hiddenDeals": hidden_deals,
        "effectiveSampleSize": float(hidden_deals),
        "difficulty": 80.0,
        "debateScore": 30.0,
        "expectedRegret": regret,
        "probabilityDrop": drop,
        "catastrophicCrossing": catastrophic,
        "supportedMetrics": ["expected_round_margin"],
        "unsupportedMetrics": [],
        "primaryDecisionScoreEvidenceHash": (
            score_evidence.sha256 if score_evidence is not None else None
        ),
        "forcedLegalityProofHash": None,
    }
    return TraceDecisionAnalysisV3(
        decision_id,
        ordinal,
        "east",
        "play",
        False,
        core["actualAction"],
        core["bestAction"],
        alternatives,
        core["method"],
        hidden_deals,
        float(hidden_deals),
        confidence,
        80.0,
        30.0,
        regret,
        drop,
        catastrophic,
        ("expected_round_margin",),
        (),
        canonical_sha256(core),
        score_evidence,
        None,
    )


def _online_agreement() -> FinalistAgentAgreementV1:
    agent = AgentProfile.from_mapping("finalist_test", {
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
    return FinalistAgentAgreementV1(
        agent,
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
        convention={},
    )


def _online_trace(item: TraceDecisionAnalysisV3) -> dict:
    evidence = item.primary_decision_score_evidence
    assert evidence is not None
    return {
        "intendedBestAction": dict(item.best_action),
        "selectedAction": dict(item.actual_action),
        "confidence": item.confidence,
        "analysisHiddenDeals": evidence.hidden_deal_support,
        "primaryDecisionScoreEvidence": evidence.to_dict(),
        "primaryDecisionScoreEvidenceHash": evidence.sha256,
        "decisionScoreRowsHash": evidence.rows_hash,
        "seed": evidence.cohort_seed,
    }


def _adjudicator(
    *,
    maximum_decisions_per_trace: int = 64,
    maximum_cohorts: int = 3,
    minimum_agreement: int = 2,
) -> AdaptiveFinalistTraceAnalyzerV1:
    primary_profile_core = {
        "schemaVersion": 1,
        "screeningProfileHash": "1" * 64,
        "confirmationProfileHash": "1" * 64,
        "maximumConfirmationDecisionsPerTrace": 64,
        "minimumScreeningScore": 0.0,
    }
    primary_profile = CascadedTraceAnalysisProfileV1.from_mapping({
        **primary_profile_core,
        "sha256": canonical_sha256(primary_profile_core),
    })
    primary_hash = primary_profile.sha256
    reanalysis_hash = "2" * 64
    rules = SimpleNamespace(sha256="3" * 64)
    primary = SimpleNamespace(
        rules=rules,
        profile=primary_profile,
        critical_threshold=70.0,
        minimum_critical_confidence=0.75,
    )
    reanalysis = SimpleNamespace(
        rules=rules,
        profile=SimpleNamespace(sha256=reanalysis_hash),
    )
    profile = FinalistSurroundingAdjudicationProfileV1.build(
        {
            "seed_offset": 104729,
            "minimum_hidden_deals": 8,
            "minimum_best_share": 0.5,
            "maximum_competence_expected_margin_regret": 1.0,
            "maximum_competence_match_win_drop": 0.01,
            "maximum_expected_margin_drift": 5.0,
            "maximum_cohorts": maximum_cohorts,
            "minimum_agreement": minimum_agreement,
            "third_cohort_on_disagreement": maximum_cohorts >= 3,
            "require_same_recommended_action": True,
        },
        primary_profile_hash=primary_hash,
        reanalysis_profile_hash=reanalysis_hash,
        maximum_decisions_per_trace=maximum_decisions_per_trace,
        minimum_confidence=0.75,
        minimum_quality=75.0,
    )
    return AdaptiveFinalistTraceAnalyzerV1(
        primary, reanalysis, _selection(), profile
    )


def test_adjudicator_rejects_selection_threshold_authority_drift():
    adjudicator = _adjudicator()
    selection_value = _selection().to_dict()
    selection_value.pop("sha256")
    selection_value["minimumSurroundingConfidence"] = 0.5
    drifted_selection = SelectionProfileV1.build(selection_value)

    with pytest.raises(ValueError, match="selection thresholds must match"):
        AdaptiveFinalistTraceAnalyzerV1(
            adjudicator.primary,
            adjudicator.reanalysis,
            drifted_selection,
            adjudicator.adjudication_profile,
        )


def test_independent_cohort_reversal_is_explicitly_unresolved_and_deterministic():
    adjudicator = _adjudicator(maximum_cohorts=2)
    primary = _analysis("D1", best="D10")
    independent = _analysis(
        "D1", best="C7", regret=5.0, drop=0.1, cohort_index=1
    )
    decisions, evidence = adjudicator._adjudicate_decisions(
        (primary,), (independent,)
    )
    repeated_decisions, repeated = adjudicator._adjudicate_decisions(
        (primary,), (independent,)
    )

    assert evidence["unresolvedDecisionIds"] == ["D1"]
    evaluation = evidence["evaluations"][0]
    assert evaluation["strictReasons"] == [
        "cohort_reversal",
        "insufficient_qualified_agreement",
    ]
    assert evaluation["status"] == "unresolved"
    assert "insufficient_competence_agreement" in evaluation["reasons"]
    assert "competence_regret" in evaluation["reasons"]
    assert evaluation["primaryAnalysisRetained"] is True
    assert decisions == (primary,)
    assert decisions == repeated_decisions
    assert evidence == repeated


def test_third_cohort_resolves_two_of_three_with_frozen_provenance():
    adjudicator = _adjudicator()
    primary = _analysis("D-majority", best="D10", best_margin=5.0)
    independent = _analysis(
        "D-majority", best="C7", best_margin=4.0, cohort_index=1
    )
    third = _analysis(
        "D-majority", best="D10", best_margin=6.0,
        cohort_index=1, cohort_seed=1002,
    )

    decisions, evidence = adjudicator._adjudicate_decisions(
        (primary,),
        (independent,),
        third_confirmations=(third,),
    )

    evaluation = evidence["evaluations"][0]
    assert evidence["unresolvedDecisionIds"] == []
    assert evaluation["status"] == "strict_confirmed"
    assert evaluation["strictRecommendationStatus"] == "confirmed_primary"
    assert evaluation["cohortCount"] == 3
    assert evaluation["agreementCount"] == 2
    assert evaluation["initialReasons"] == [
        "cohort_reversal",
        "insufficient_qualified_agreement",
    ]
    assert evaluation["cohortAnalysisHashes"] == [
        primary.sha256,
        independent.sha256,
        third.sha256,
    ]
    assert decisions[0].best_action == primary.best_action
    assert decisions[0] is primary
    assert evaluation["primaryAnalysisRetained"] is True


def test_three_cohort_quorum_can_require_all_three_without_pair_failure():
    adjudicator = _adjudicator(minimum_agreement=3)
    cohorts = tuple(
        _analysis(
            "D-unanimous-three",
            best="D10",
            cohort_index=0 if index == 0 else 1,
            cohort_seed=1000 + index,
        )
        for index in range(3)
    )

    _, evidence = adjudicator._adjudicate_decisions(
        (cohorts[0],),
        (cohorts[1],),
        third_confirmations=(cohorts[2],),
    )

    evaluation = evidence["evaluations"][0]
    assert evidence["thirdCohortDecisionIds"] == ["D-unanimous-three"]
    assert evaluation["status"] == "strict_confirmed"
    assert evaluation["agreementCount"] == 3
    assert evaluation["supporterCohortIndices"] == [0, 1, 2]


def test_value_equivalent_actual_action_is_robust_surrounding_competence():
    adjudicator = _adjudicator()
    cohorts = tuple(
        _analysis(
            "D-competent",
            best="C7",
            actual="D10",
            regret=0.3,
            drop=0.005,
            best_share=0.4,
            completion="budget_exhausted",
            exact_share=0.0,
            method="paired_information_set",
            cohort_index=0 if cohort_index == 0 else 1,
            cohort_seed=1000 + cohort_index,
        )
        for cohort_index in range(3)
    )

    decisions, evidence = adjudicator._adjudicate_decisions(
        (cohorts[0],),
        (cohorts[1],),
        third_confirmations=(cohorts[2],),
    )

    evaluation = evidence["evaluations"][0]
    assert evidence["unresolvedDecisionIds"] == []
    assert evidence["competentDecisionIds"] == ["D-competent"]
    assert evaluation["status"] == "primary_robust_competent"
    assert evaluation["competenceEvidence"]["agreementCount"] == 3
    assert decisions[0].actual_action == Card.parse("D10").to_dict()
    assert decisions[0] is cohorts[0]
    assert evaluation["featuredTruthEligible"] is False
    assert evaluation["primaryAnalysisRetained"] is True


def test_budget_exhaustion_needs_stable_independent_competence():
    adjudicator = _adjudicator()
    cohorts = tuple(
        _analysis(
            "D-budget",
            best="C7",
            actual="D10",
            regret=0.2,
            drop=0.004,
            completion="budget_exhausted",
            cohort_index=0 if cohort_index == 0 else 1,
            cohort_seed=1000 + cohort_index,
        )
        for cohort_index in range(3)
    )

    _, evidence = adjudicator._adjudicate_decisions(
        (cohorts[0],),
        (cohorts[1],),
        third_confirmations=(cohorts[2],),
    )

    evaluation = evidence["evaluations"][0]
    assert evaluation["status"] == "primary_robust_competent"
    assert evaluation["strictReasons"] == [
        "analysis_not_converged",
        "insufficient_qualified_agreement",
    ]
    assert all(
        item["budgetExhausted"]
        for item in evaluation["competenceEvidence"]["cohortEvaluations"]
    )


def test_unstable_value_intervals_cannot_be_called_competent():
    adjudicator = _adjudicator()
    cohorts = (
        _analysis(
            "D-interval",
            best="C7",
            actual="D10",
            regret=0.2,
            drop=0.004,
            best_share=0.4,
            completion="budget_exhausted",
            exact_share=0.0,
            method="paired_information_set",
            confidence_interval=(0.05, 0.2),
            cohort_index=0,
        ),
        _analysis(
            "D-interval",
            best="C7",
            actual="D10",
            regret=0.2,
            drop=0.004,
            best_share=0.4,
            completion="budget_exhausted",
            exact_share=0.0,
            method="paired_information_set",
            confidence_interval=(0.35, 0.5),
            cohort_index=1,
        ),
        _analysis(
            "D-interval",
            best="C7",
            actual="D10",
            regret=0.2,
            drop=0.004,
            best_share=0.4,
            completion="budget_exhausted",
            exact_share=0.0,
            method="paired_information_set",
            confidence_interval=(0.7, 0.85),
            cohort_index=1,
            cohort_seed=1002,
        ),
    )

    _, evidence = adjudicator._adjudicate_decisions(
        (cohorts[0],),
        (cohorts[1],),
        third_confirmations=(cohorts[2],),
    )

    evaluation = evidence["evaluations"][0]
    assert evaluation["status"] == "unresolved"
    assert evaluation["competenceEvidence"]["valueIntervalsStable"] is False


def test_materially_bad_unanimous_actual_action_is_not_competent():
    adjudicator = _adjudicator()
    primary = _analysis(
        "D-bad", best="C7", actual="D10", regret=5.0, drop=0.1
    )
    independent = _analysis(
        "D-bad", best="C7", actual="D10", regret=5.0, drop=0.1,
        cohort_index=1,
    )

    decisions, evidence = adjudicator._adjudicate_decisions(
        (primary,), (independent,)
    )
    qualification = _selection().qualify_surrounding(
        (*decisions, _analysis("featured", best="D10", ordinal=2)),
        candidate_decision_id="featured",
        trace_analysis_hash="b" * 64,
    )

    evaluation = evidence["evaluations"][0]
    assert evaluation["status"] == "strict_confirmed"
    assert evaluation["strictRecommendationStatus"] == "confirmed_mistake"
    assert evaluation["primaryAnalysisRetained"] is True
    assert decisions == (primary,)
    assert evidence["competentDecisionIds"] == []
    assert qualification.passed is False
    assert qualification.material_mistakes == 1


def test_online_and_posttrace_competence_evaluators_have_semantic_parity():
    cohorts = tuple(
        _analysis(
            "D-parity",
            best="C7",
            actual="D10",
            regret=0.4,
            drop=0.006,
            best_share=0.4,
            completion="budget_exhausted",
            exact_share=0.0,
            method="paired_information_set",
            cohort_index=0 if index == 0 else 1,
            cohort_seed=1000 + index,
        )
        for index in range(3)
    )
    posttrace = _adjudicator()._competence_evidence(cohorts)
    online = _online_agreement()._resolve_primary_competence(
        cohorts[0].actual_action,
        tuple(_online_trace(item) for item in cohorts),
    )

    for key in (
        "status",
        "primaryActionHash",
        "agreementCount",
        "minimumAgreement",
        "expectedMarginRegretDrift",
        "maximumExpectedMarginDrift",
        "valueIntervalsStable",
        "resolutionReasons",
    ):
        assert online[key] == posttrace[key]
    for online_cohort, posttrace_cohort in zip(
        online["cohortEvaluations"],
        posttrace["cohortEvaluations"],
        strict=True,
    ):
        for key in (
            "cohortIndex",
            "bestActionHash",
            "primaryActionHash",
            "expectedMarginRegret",
            "matchWinDrop",
            "matchWinMetricSupported",
            "valueConfidence95",
            "budgetExhausted",
            "populationCoverageComplete",
            "solverCompletion",
            "truthClass",
            "exhaustiveExactSupport",
            "strictReasons",
            "reasons",
            "qualified",
        ):
            assert online_cohort[key] == posttrace_cohort[key]


def test_missing_canonical_score_evidence_fails_closed_and_retains_primary():
    primary = _analysis(
        "D-missing-score", best="C7", actual="D10",
        include_score_evidence=False,
    )
    independent = _analysis(
        "D-missing-score", best="C7", actual="D10", cohort_index=1,
        include_score_evidence=False,
    )
    third = _analysis(
        "D-missing-score", best="C7", actual="D10", cohort_index=1,
        cohort_seed=1002, alternative_margin_shift=1.0,
        include_score_evidence=False,
    )

    decisions, evidence = _adjudicator()._adjudicate_decisions(
        (primary,), (independent,), third_confirmations=(third,)
    )

    evaluation = evidence["evaluations"][0]
    assert evaluation["status"] == "unresolved"
    assert evidence["unresolvedDecisionIds"] == ["D-missing-score"]
    assert evidence["competentDecisionIds"] == []
    assert "missing_primary_decision_score_evidence" in evaluation["strictReasons"]
    assert "missing_competence_utility" in evaluation["reasons"]
    assert decisions == (primary,)
    assert evaluation["primaryAnalysisRetained"] is True


def test_display_alternative_tamper_cannot_change_canonical_verdict():
    common = {
        "decision_id": "D-canonical-only",
        "best": "C7",
        "actual": "D10",
        "regret": 0.4,
        "drop": 0.006,
        "best_share": 0.4,
        "completion": "budget_exhausted",
        "exact_share": 0.0,
        "method": "paired_information_set",
    }
    baseline = tuple(
        _analysis(
            cohort_index=0 if index == 0 else 1,
            cohort_seed=1000 + index,
            **common,
        )
        for index in range(3)
    )
    tampered = tuple(
        _analysis(
            cohort_index=0 if index == 0 else 1,
            cohort_seed=1000 + index,
            alternative_margin_shift=1000.0,
            **common,
        )
        for index in range(3)
    )

    _, baseline_evidence = _adjudicator()._adjudicate_decisions(
        (baseline[0],),
        (baseline[1],),
        third_confirmations=(baseline[2],),
    )
    _, tampered_evidence = _adjudicator()._adjudicate_decisions(
        (tampered[0],),
        (tampered[1],),
        third_confirmations=(tampered[2],),
    )

    baseline_evaluation = baseline_evidence["evaluations"][0]
    tampered_evaluation = tampered_evidence["evaluations"][0]
    assert baseline[0].sha256 != tampered[0].sha256
    assert (
        baseline[0].primary_decision_score_evidence.sha256
        == tampered[0].primary_decision_score_evidence.sha256
    )
    assert baseline_evaluation["status"] == "primary_robust_competent"
    assert tampered_evaluation["status"] == baseline_evaluation["status"]
    assert (
        tampered_evaluation["competenceEvidence"]["agreementCount"]
        == baseline_evaluation["competenceEvidence"]["agreementCount"]
    )
    assert [
        item["expectedMarginRegret"]
        for item in tampered_evaluation["competenceEvidence"][
            "cohortEvaluations"
        ]
    ] == [
        item["expectedMarginRegret"]
        for item in baseline_evaluation["competenceEvidence"][
            "cohortEvaluations"
        ]
    ]


def test_overflow_evaluation_uses_the_same_closed_shape_and_retains_primary():
    bounded = _analysis("D-bounded", best="D10", ordinal=1)
    confirmation = _analysis(
        "D-bounded", best="D10", ordinal=1, cohort_index=1
    )
    primary = _analysis("D-overflow", best="D10", ordinal=2)
    adjudicator = _adjudicator(maximum_decisions_per_trace=1)
    regular = adjudicator._evaluate_pair(bounded, confirmation)[1]

    decisions, evidence = adjudicator._adjudicate_decisions(
        (bounded, primary),
        (confirmation,),
        overflow_ids=("D-overflow",),
    )

    overflow = evidence["evaluations"][1]
    assert set(overflow) == set(regular)
    assert overflow["status"] == "unresolved"
    assert overflow["reasons"] == ["adjudication_bound_exceeded"]
    assert overflow["cohortCount"] == 1
    assert overflow["reanalysis"] is None
    assert overflow["reanalysisHash"] is None
    assert overflow["cohortAnalysisHashes"] == [primary.sha256]
    assert overflow["selectedAnalysisHash"] == primary.sha256
    assert overflow["retainedPrimaryAnalysisHash"] == primary.sha256
    assert overflow["primaryAnalysisRetained"] is True
    assert evidence["targetDecisionIds"] == ["D-bounded", "D-overflow"]
    assert evidence["boundedDecisionIds"] == ["D-bounded"]
    assert evidence["overflowDecisionIds"] == ["D-overflow"]
    assert evidence["thirdCohortDecisionIds"] == []
    assert evidence["adjudicationProfile"]["sha256"] == evidence["profileHash"]
    assert decisions == (bounded, primary)


def test_competence_only_decision_remains_blocked_as_featured_truth():
    analysis = SimpleNamespace(
        unresolved_decision_ids=("unresolved",),
        adjudication_evidence={"competentDecisionIds": ["competent"]},
    )

    assert FoundryAnalysisFunnelV1._featured_truth_blocked_ids(analysis) == (
        "competent",
        "unresolved",
    )


def test_low_support_budget_exhaustion_stays_unresolved_even_when_actions_agree():
    adjudicator = _adjudicator()
    primary = _analysis(
        "D2", best="D10", hidden_deals=6, confidence=0.24,
        completion="budget_exhausted",
    )
    independent = _analysis(
        "D2", best="D10", hidden_deals=6, confidence=0.24,
        completion="budget_exhausted", cohort_index=1,
    )
    third = _analysis(
        "D2", best="D10", hidden_deals=6, confidence=0.24,
        completion="budget_exhausted", cohort_index=1, cohort_seed=1002,
    )

    _, evidence = adjudicator._adjudicate_decisions(
        (primary,), (independent,), third_confirmations=(third,)
    )

    evaluation = evidence["evaluations"][0]
    reasons = evaluation["reasons"]
    assert evidence["unresolvedDecisionIds"] == ["D2"]
    assert evaluation["strictReasons"] == [
        "analysis_not_converged",
        "budget_exhausted_low_support",
        "insufficient_qualified_agreement",
        "low_confidence",
        "low_hidden_deal_support",
    ]
    assert "insufficient_competence_agreement" in reasons
    assert "low_confidence" in reasons
    assert "low_hidden_deal_support" in reasons


@pytest.mark.parametrize("hidden_deals", [1, 6])
def test_complete_exact_small_population_is_supported(hidden_deals):
    adjudicator = _adjudicator()
    exact = {
        "best": "D10",
        "hidden_deals": hidden_deals,
        "confidence": 0.1,
        "completion": "exact",
        "exact_share": 1.0,
        "method": "exhaustive_constrained_hidden_deals+hard_facts_only",
    }
    primary = _analysis("D-exact", **exact)
    independent = _analysis("D-exact", cohort_index=1, **exact)

    _, evidence = adjudicator._adjudicate_decisions((primary,), (independent,))

    evaluation = evidence["evaluations"][0]
    assert evidence["unresolvedDecisionIds"] == []
    assert evaluation["status"] == "strict_confirmed"
    assert evaluation["primaryExhaustiveExactSupport"] is True
    assert evaluation["reanalysisExhaustiveExactSupport"] is True


@pytest.mark.parametrize("hidden_deals", [10, 15, 33])
def test_complete_population_heuristic_is_surrounding_competence_only(
    hidden_deals,
):
    adjudicator = _adjudicator()
    values = {
        "best": "C7",
        "actual": "D10",
        "hidden_deals": hidden_deals,
        "confidence": 0.1,
        "regret": 0.3,
        "drop": 0.005,
        "completion": "heuristic_complete",
        "exact_share": 0.0,
        "method": "exhaustive_constrained_hidden_deals+hard_facts_only",
    }
    cohorts = tuple(
        _analysis(
            "D-population",
            cohort_index=0 if index == 0 else 1,
            cohort_seed=1000 + index,
            **values,
        )
        for index in range(3)
    )

    _, evidence = adjudicator._adjudicate_decisions(
        (cohorts[0],),
        (cohorts[1],),
        third_confirmations=(cohorts[2],),
    )

    evaluation = evidence["evaluations"][0]
    assert evaluation["status"] == "primary_robust_competent"
    assert evidence["competentDecisionIds"] == ["D-population"]
    assert evidence["unresolvedDecisionIds"] == []
    assert evaluation["cohortPopulationCoverageComplete"] == [True, True, True]
    assert evaluation["cohortSolverCompletion"] == [
        "heuristic_complete",
    ] * 3
    assert evaluation["cohortTruthClass"] == [
        "full_population_best_known",
    ] * 3
    assert "population_solver_best_known_only" in evaluation["strictReasons"]
    assert evaluation["featuredTruthEligible"] is False
    assert evaluation["surroundingVerified"] is True


@pytest.mark.parametrize(
    "method,completion",
    [
        ("paired_information_set", "budget_exhausted"),
        (
            "exhaustive_constrained_hidden_deals+hard_facts_only",
            "budget_exhausted",
        ),
    ],
)
def test_low_confidence_without_completed_population_remains_unresolved(
    method,
    completion,
):
    adjudicator = _adjudicator()
    values = {
        "best": "C7",
        "actual": "D10",
        "hidden_deals": 33,
        "confidence": 0.1,
        "regret": 0.3,
        "drop": 0.005,
        "completion": completion,
        "method": method,
    }
    cohorts = tuple(
        _analysis(
            "D-not-complete",
            cohort_index=0 if index == 0 else 1,
            cohort_seed=1000 + index,
            **values,
        )
        for index in range(3)
    )

    _, evidence = adjudicator._adjudicate_decisions(
        (cohorts[0],),
        (cohorts[1],),
        third_confirmations=(cohorts[2],),
    )

    evaluation = evidence["evaluations"][0]
    assert evaluation["status"] == "unresolved"
    assert evidence["unresolvedDecisionIds"] == ["D-not-complete"]
    assert evaluation["cohortPopulationCoverageComplete"] == [False] * 3


def test_stable_independent_cohorts_retain_conservative_complete_evidence():
    adjudicator = _adjudicator()
    primary = _analysis("D3", best="C7", regret=2.0, drop=0.03, best_margin=6.0)
    independent = _analysis(
        "D3", best="C7", regret=3.0, drop=0.04, best_margin=5.0,
        cohort_index=1,
    )

    decisions, evidence = adjudicator._adjudicate_decisions(
        (primary,), (independent,)
    )

    assert evidence["unresolvedDecisionIds"] == []
    evaluation = evidence["evaluations"][0]
    assert evaluation["status"] == "strict_confirmed"
    assert evaluation["strictRecommendationStatus"] == "confirmed_mistake"
    assert evaluation["primaryAnalysisRetained"] is True
    assert decisions == (primary,)


def test_apparently_good_material_decision_still_gets_independent_adjudication():
    adjudicator = _adjudicator()
    material = _analysis("D4", best="D10", regret=5.0)

    assert adjudicator._requires_adjudication(material) is True


def test_display_only_materiality_cannot_change_the_adjudication_target_set():
    adjudicator = _adjudicator()
    baseline = _analysis("D-display-target", best="D10")
    tampered = _analysis(
        "D-display-target",
        best="D10",
        alternative_margin_shift=1_000.0,
    )

    assert baseline.sha256 != tampered.sha256
    assert (
        baseline.primary_decision_score_evidence.sha256
        == tampered.primary_decision_score_evidence.sha256
    )
    assert adjudicator._requires_adjudication(baseline) is False
    assert adjudicator._requires_adjudication(tampered) is False


def test_unresolved_adjudication_counts_as_unverified_without_relaxing_gate():
    profile = _selection()
    candidate = _analysis("candidate", best="D10", ordinal=1)
    otherwise_good = _analysis("surrounding", best="D10", ordinal=2)

    result = profile.qualify_surrounding(
        (candidate, otherwise_good),
        candidate_decision_id="candidate",
        trace_analysis_hash="a" * 64,
        unresolved_decision_ids=("surrounding",),
    )

    assert result.passed is False
    assert result.material_mistakes == 0
    assert result.unverified_decisions == 1
    assert result.rejection_reasons == ("unverified_decisions",)


def test_robust_competence_counts_as_verified_surrounding_evidence():
    profile = _selection()
    candidate = _analysis("candidate", best="D10", ordinal=1)
    low_confidence_but_competent = _analysis(
        "competent",
        best="D10",
        ordinal=2,
        confidence=0.2,
    )

    result = profile.qualify_surrounding(
        (candidate, low_confidence_but_competent),
        candidate_decision_id="candidate",
        trace_analysis_hash="a" * 64,
        competence_verified_decision_ids=("competent",),
    )

    assert result.passed is True
    assert result.unverified_decisions == 0
    assert result.rejection_reasons == ()


def test_unresolved_status_overrides_competence_only_surrounding_evidence():
    profile = _selection()
    candidate = _analysis("candidate", best="D10", ordinal=1)
    disputed = _analysis("disputed", best="D10", ordinal=2, confidence=0.2)

    result = profile.qualify_surrounding(
        (candidate, disputed),
        candidate_decision_id="candidate",
        trace_analysis_hash="a" * 64,
        competence_verified_decision_ids=("disputed",),
        unresolved_decision_ids=("disputed",),
    )

    assert result.passed is False
    assert result.unverified_decisions == 1
    assert result.rejection_reasons == ("unverified_decisions",)


def test_surrounding_qualification_ignores_fully_rehashed_display_metric_drift():
    profile = _selection()
    candidate = _analysis("candidate-display-ssot", best="D10", ordinal=1)
    surrounding = _analysis("surrounding-display-ssot", best="D10", ordinal=2)
    tampered_alternatives = tuple(
        {
            **dict(item),
            "expectedMargin": float(item["expectedMargin"]) + 10_000.0,
            "winProbability": 0.0,
        }
        for item in surrounding.alternatives
    )
    tampered = replace(
        surrounding,
        alternatives=tampered_alternatives,
        confidence=0.0,
        difficulty=0.0,
        debate_score=0.0,
        expected_regret=10_000.0,
        probability_drop=1.0,
        catastrophic_crossing=True,
        sha256="f" * 64,
    )

    baseline = profile.qualify_surrounding(
        (candidate, surrounding),
        candidate_decision_id=candidate.decision_id,
        trace_analysis_hash="a" * 64,
        competence_verified_decision_ids=(surrounding.decision_id,),
    )
    drifted = profile.qualify_surrounding(
        (candidate, tampered),
        candidate_decision_id=candidate.decision_id,
        trace_analysis_hash="a" * 64,
        competence_verified_decision_ids=(tampered.decision_id,),
    )

    assert drifted.to_dict() == baseline.to_dict()
