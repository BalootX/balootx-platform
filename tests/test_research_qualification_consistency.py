from __future__ import annotations

from types import SimpleNamespace

import pytest

import baloot_platform.research.service as research_service_module
from baloot_platform.research.decision_audit import (
    DECISION_CONSISTENCY_EVIDENCE_UNAVAILABLE,
    DECISION_MATERIAL_COUNTEREXAMPLE,
    DECISION_RECOMMENDATION_MISMATCH,
    HISTORICAL_RECORD_REQUIRES_READJUDICATION,
    InformationSetEvidenceV1,
    SequentialResponseEvidenceV1,
    decision_consistency_qualification_failures,
)
from baloot_platform.research.service import (
    ResearchFoundryServiceV1,
    _decision_truth_label_v1,
    _preference_selection_score,
    _tactical_verification_provenance,
)


def _actions() -> tuple[dict[str, str], ...]:
    return ({"action": "pass"}, {"action": "sun"})


def _response(
    *,
    actual: dict[str, str],
    recommended: dict[str, str],
    forced: bool = False,
) -> dict[str, object]:
    actions = (actual,) if forced else _actions()
    alternatives = tuple(
        {
            "action": action,
            "searchCompletionStatus": "heuristic_complete",
            "exactShare": 0.0,
        }
        for action in actions
    )
    information_set = InformationSetEvidenceV1.build(
        observer="south",
        information_state={"schemaVersion": 1, "observer": "south"},
        legal_actions=actions,
        alternatives=alternatives,
        forced=forced,
        analysis_hash="a" * 64,
        method="constructive_uniform_hidden_deals+hard_facts_only",
        hidden_deal_support=2,
        effective_sample_size=2.0,
    )
    return SequentialResponseEvidenceV1.build(
        actual_action=actual,
        recommended_action=recommended,
        legal_actions=actions,
        alternatives=alternatives,
        information_set=information_set,
        expected_regret=0.0,
        forced=forced,
    ).to_dict()


def _decision(
    *,
    kind: str = "play",
    forced: bool = False,
    actual: dict[str, str] | None = None,
    recommended: dict[str, str] | None = None,
    acting_seat: str = "south",
    tactical_evidence: tuple[dict[str, object], ...] = (),
) -> dict[str, object]:
    actual_action = actual or {"action": "pass"}
    recommended_action = recommended or actual_action
    response = _response(
        actual=actual_action,
        recommended=recommended_action,
        forced=forced,
    )
    return {
        "decisionId": f"{kind.upper()}-{acting_seat.upper()}",
        "decisionKind": kind,
        "actingSeat": acting_seat,
        "forced": forced,
        "actualAction": actual_action,
        "recommendedAction": recommended_action,
        "actualMatchesRecommendation": actual_action == recommended_action,
        "sequentialResponseEvidence": response,
        "truthLabel": "exact_optimal" if forced else "robust_best_known",
        "confidence": 1.0,
        "expectedRegret": 0.0,
        "probabilityDrop": 0.0,
        "tacticalEvidence": list(tactical_evidence),
    }


def _audit(*decisions: dict[str, object], schema_version: int = 2) -> dict[str, object]:
    return {
        "schemaVersion": schema_version,
        "complete": True,
        "informationSetComplete": True,
        "sequentialResponseComplete": True,
        "decisions": list(decisions),
    }


def _qualification(
    audit: dict[str, object],
    *,
    tactics: tuple[dict[str, object], ...] = (),
    minimum_truth: str = "provisional",
    critical_decision_ids: tuple[str, ...] = ("CRITICAL-1",),
    competent_decision_ids: tuple[str, ...] = (),
    unresolved_decision_ids: tuple[str, ...] = (),
) -> dict[str, object]:
    service = object.__new__(ResearchFoundryServiceV1)
    coverage = SimpleNamespace(
        executable_binding_complete=True,
        behavioral_or_invariant_complete=True,
    )
    service.compiled_domain = SimpleNamespace(
        compiled=SimpleNamespace(coverage=coverage)
    )
    recipe = SimpleNamespace(
        qualification={
            "minimumTruth": minimum_truth,
            "minimumConfidence": 0.0,
            "minimumSurroundingQuality": 0.0,
            "requireObserverSafety": True,
            "requireAllDecisionAudit": True,
            "maximumRegret": 0.0,
        },
        tactics=tactics,
    )
    analysis = SimpleNamespace(
        laffa_evidence=SimpleNamespace(
            critical_decision_ids=critical_decision_ids
        ),
        adjudication_evidence={
            "competentDecisionIds": list(competent_decision_ids),
            "unresolvedDecisionIds": list(unresolved_decision_ids),
        },
        unresolved_decision_ids=unresolved_decision_ids,
    )
    return service._qualification(recipe, audit, analysis)


@pytest.mark.parametrize(
    (
        "sequential_response_status",
        "confidence",
        "debate_score",
        "expected",
    ),
    [
        ("exact_sequential_response", 0.0, 100.0, "exact_optimal"),
        (
            "robust_policy_conditioned_response",
            0.75,
            100.0,
            "robust_best_known",
        ),
        ("incomplete", 1.0, 70.0, "debatable"),
        ("incomplete", 1.0, 69.999, "provisional"),
    ],
)
def test_decision_truth_uses_proof_tier_before_debate_interest(
    sequential_response_status,
    confidence,
    debate_score,
    expected,
):
    assert _decision_truth_label_v1(
        sequential_response_status=sequential_response_status,
        confidence=confidence,
        debate_score=debate_score,
    ) == expected


def test_robust_debatable_decision_passes_robust_recipe_truth_gate() -> None:
    decision = _decision()
    decision["truthLabel"] = _decision_truth_label_v1(
        sequential_response_status="robust_policy_conditioned_response",
        confidence=1.0,
        debate_score=100.0,
    )

    result = _qualification(
        _audit(decision),
        minimum_truth="robust_best_known",
    )

    assert result["qualified"] is True
    assert result["truthLabel"] == "robust_best_known"
    assert "truth_below_required_tier" not in result["failures"]


def _tactic(
    *,
    mode: str = "preference",
    minimum_score: float = 70.0,
    perspectives: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "kind": "small_card_big_gain",
        "mode": mode,
        "minimumScore": minimum_score,
        "weight": 1.0,
        "perspectives": list(perspectives),
    }


def _tactical_evidence(magnitude: float) -> tuple[dict[str, object], ...]:
    return ({
        "kind": "small_card_big_gain",
        "magnitude": magnitude,
        "proofHash": "b" * 64,
    },)


@pytest.mark.parametrize("decision_kind", ["bid", "play"])
def test_zero_margin_regret_cannot_hide_a_recommendation_counterexample(
    decision_kind: str,
) -> None:
    decision = _decision(
        kind=decision_kind,
        actual={"action": "pass"},
        recommended={"action": "sun"},
    )
    assert decision["expectedRegret"] == 0.0
    assert decision["actualMatchesRecommendation"] is False
    assert (
        decision["sequentialResponseEvidence"]["materialCounterexampleFound"]
        is True
    )

    result = _qualification(_audit(decision))

    assert result["qualified"] is False
    assert result["truthLabel"] == "rejected"
    assert DECISION_MATERIAL_COUNTEREXAMPLE in result["failures"]
    assert DECISION_RECOMMENDATION_MISMATCH in result["failures"]
    assert "surrounding_agent_regret" not in result["failures"]


def test_matching_forced_decision_remains_eligible_but_inconsistent_forced_evidence_fails() -> None:
    forced = _decision(kind="play", forced=True)
    assert decision_consistency_qualification_failures(_audit(forced)) == ()
    assert _qualification(_audit(forced))["qualified"] is True

    inconsistent = _decision(
        kind="play",
        forced=True,
        actual={"action": "pass"},
        recommended={"action": "sun"},
    )
    failures = decision_consistency_qualification_failures(_audit(inconsistent))
    assert failures == (DECISION_RECOMMENDATION_MISMATCH,)
    assert _qualification(_audit(inconsistent))["qualified"] is False


def test_material_counterexample_is_an_independent_gate() -> None:
    decision = _decision(kind="bid")
    decision["sequentialResponseEvidence"][
        "materialCounterexampleFound"
    ] = True

    failures = decision_consistency_qualification_failures(_audit(decision))

    assert failures == (DECISION_MATERIAL_COUNTEREXAMPLE,)
    assert _qualification(_audit(decision))["qualified"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda decision: decision.pop("actualMatchesRecommendation"),
        lambda decision: decision.pop("actualAction"),
        lambda decision: decision.pop("sequentialResponseEvidence"),
        lambda decision: decision.update({"recommendationAvailable": False}),
        lambda decision: decision["sequentialResponseEvidence"].pop(
            "materialCounterexampleFound"
        ),
    ],
)
def test_current_audit_with_unavailable_decision_evidence_fails_closed(
    mutation,
) -> None:
    decision = _decision(kind="bid")
    mutation(decision)

    result = _qualification(_audit(decision))

    assert result["qualified"] is False
    assert DECISION_CONSISTENCY_EVIDENCE_UNAVAILABLE in result["failures"]


def test_legacy_audit_is_readjudication_not_a_new_semantic_claim() -> None:
    decision = _decision(
        kind="play",
        actual={"action": "pass"},
        recommended={"action": "sun"},
    )
    legacy = _audit(decision, schema_version=1)

    failures = decision_consistency_qualification_failures(legacy)
    result = _qualification(legacy)

    assert failures == (HISTORICAL_RECORD_REQUIRES_READJUDICATION,)
    assert result["qualified"] is False
    assert HISTORICAL_RECORD_REQUIRES_READJUDICATION in result["failures"]
    assert DECISION_MATERIAL_COUNTEREXAMPLE not in result["failures"]
    assert DECISION_RECOMMENDATION_MISMATCH not in result["failures"]


def test_unknown_future_audit_schema_fails_closed_without_legacy_label() -> None:
    result = _qualification(_audit(_decision(), schema_version=3))

    assert result["qualified"] is False
    assert DECISION_CONSISTENCY_EVIDENCE_UNAVAILABLE in result["failures"]
    assert HISTORICAL_RECORD_REQUIRES_READJUDICATION not in result["failures"]


def test_current_empty_audit_has_no_implicit_consistency_proof() -> None:
    audit = _audit()

    assert decision_consistency_qualification_failures(audit) == (
        DECISION_CONSISTENCY_EVIDENCE_UNAVAILABLE,
    )
    assert _qualification(audit)["qualified"] is False


def test_preference_threshold_and_perspective_filter_selection_score() -> None:
    south = _decision(
        tactical_evidence=_tactical_evidence(80.0),
        acting_seat="south",
    )
    included = _qualification(
        _audit(south),
        tactics=(_tactic(perspectives=("south",)),),
    )
    evaluation = included["tacticEvaluations"][0]

    assert included["qualified"] is True
    assert evaluation == {
        "kind": "small_card_big_gain",
        "mode": "preference",
        "minimumScore": 70.0,
        "weight": 1.0,
        "perspectives": ["south"],
        "score": 80.0,
        "applicableEvidenceCount": 1,
        "matchingEvidenceCount": 1,
        "matchingEvidence": [{
            "decisionId": "PLAY-SOUTH",
            "actingSeat": "south",
            "magnitude": 80.0,
            "proofHash": "b" * 64,
        }],
        "eligible": True,
    }
    assert _preference_selection_score(60.0, [evaluation]) == 70.0

    above_observed = _qualification(
        _audit(south),
        tactics=(_tactic(minimum_score=80.000001),),
    )["tacticEvaluations"][0]
    wrong_seat = _qualification(
        _audit(south),
        tactics=(_tactic(perspectives=("east",)),),
    )["tacticEvaluations"][0]

    assert above_observed["eligible"] is False
    assert above_observed["score"] == 80.0
    assert wrong_seat["eligible"] is False
    assert wrong_seat["score"] == 0.0
    assert above_observed["matchingEvidence"] == []
    assert wrong_seat["matchingEvidence"] == []
    assert _preference_selection_score(60.0, [above_observed]) == 30.0
    assert _preference_selection_score(60.0, [wrong_seat]) == 30.0


def test_preference_uses_only_the_requested_acting_seat() -> None:
    south = _decision(
        tactical_evidence=_tactical_evidence(60.0),
        acting_seat="south",
    )
    east = _decision(
        tactical_evidence=_tactical_evidence(90.0),
        acting_seat="east",
    )
    audit = _audit(south, east)

    any_seat = _qualification(
        audit,
        tactics=(_tactic(minimum_score=0.0),),
    )["tacticEvaluations"][0]
    south_only = _qualification(
        audit,
        tactics=(_tactic(minimum_score=0.0, perspectives=("south",)),),
    )["tacticEvaluations"][0]
    east_only = _qualification(
        audit,
        tactics=(_tactic(minimum_score=0.0, perspectives=("east",)),),
    )["tacticEvaluations"][0]

    assert any_seat["score"] == 90.0
    assert south_only["score"] == 60.0
    assert east_only["score"] == 90.0


def test_missing_preference_evidence_never_passes_even_at_zero_threshold() -> None:
    result = _qualification(
        _audit(_decision()),
        tactics=(_tactic(minimum_score=0.0),),
    )
    evaluation = result["tacticEvaluations"][0]

    assert result["qualified"] is True
    assert evaluation["eligible"] is False
    assert evaluation["matchingEvidenceCount"] == 0
    assert _preference_selection_score(60.0, [evaluation]) == 30.0


def test_preference_miss_does_not_reject_but_qualification_gate_miss_does() -> None:
    decision = _decision(tactical_evidence=_tactical_evidence(69.0))
    preference = _qualification(
        _audit(decision),
        tactics=(_tactic(mode="preference", minimum_score=70.0),),
    )
    gate = _qualification(
        _audit(decision),
        tactics=(_tactic(mode="qualification_gate", minimum_score=70.0),),
    )

    assert preference["qualified"] is True
    assert preference["tacticEvaluations"][0]["eligible"] is False
    assert gate["qualified"] is False
    assert gate["failures"][-1] == (
        "tactical_gate_not_proven:small_card_big_gain"
    )


@pytest.mark.parametrize(
    ("competent_decision_ids", "unresolved_decision_ids"),
    [
        (("PLAY-SOUTH",), ()),
        ((), ("PLAY-SOUTH",)),
    ],
)
def test_blocked_tactical_evidence_cannot_satisfy_retention_or_critical_truth(
    competent_decision_ids: tuple[str, ...],
    unresolved_decision_ids: tuple[str, ...],
) -> None:
    decision = _decision(tactical_evidence=_tactical_evidence(90.0))

    result = _qualification(
        _audit(decision),
        tactics=(_tactic(mode="qualification_gate", minimum_score=70.0),),
        critical_decision_ids=("PLAY-SOUTH",),
        competent_decision_ids=competent_decision_ids,
        unresolved_decision_ids=unresolved_decision_ids,
    )

    assert result["qualified"] is False
    assert "no_verified_critical_point" in result["failures"]
    assert (
        "tactical_gate_not_proven:small_card_big_gain" in result["failures"]
    )
    assert result["tacticalEvidenceCount"] == 0
    assert result["tacticEvaluations"][0]["eligible"] is False
    assert result["analysisCriticalDecisionIds"] == ["PLAY-SOUTH"]
    assert result["featuredTruthEligibleCriticalDecisionIds"] == []
    assert result["featuredTruthBlockedDecisionIds"] == ["PLAY-SOUTH"]


def test_matching_preference_always_outranks_missing_preference() -> None:
    matched = _qualification(
        _audit(_decision(tactical_evidence=_tactical_evidence(70.0))),
        tactics=(_tactic(minimum_score=70.0),),
    )["tacticEvaluations"][0]
    missing = _qualification(
        _audit(_decision()),
        tactics=(_tactic(minimum_score=70.0),),
    )["tacticEvaluations"][0]

    assert _preference_selection_score(90.0, [matched]) == 80.0
    assert _preference_selection_score(90.0, [missing]) == 45.0


def test_bounded_preference_verification_prioritizes_requested_acting_seat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def snapshot(decision_id: str, seat: str, ordinal: int):
        return SimpleNamespace(
            decision=SimpleNamespace(
                decision_id=decision_id,
                decision_kind="play",
                forced=False,
                legal_actions=({"card": "S7"}, {"card": "S8"}),
                ordinal=ordinal,
                acting_seat=SimpleNamespace(name=seat.upper()),
            ),
            state=SimpleNamespace(hands=(1, 1, 1, 1)),
        )

    east = snapshot("PLAY-EAST", "east", 31)
    south = snapshot("PLAY-SOUTH", "south", 1)
    monkeypatch.setattr(
        research_service_module,
        "TraceDecisionProjectionStoreV1",
        lambda trace, rules: SimpleNamespace(
            replay=SimpleNamespace(decisions=(east, south))
        ),
    )
    verified: list[str] = []
    verifier = SimpleNamespace(
        maximum_remaining_cards=32,
        verify_decision=lambda item, analysis, cancel_check: (
            verified.append(item.decision.decision_id)
            or item.decision.decision_id
        ),
    )
    service = object.__new__(ResearchFoundryServiceV1)
    service.tactical_verifier = verifier
    service.maximum_tactical_decisions = 1
    service.simulator = SimpleNamespace(rules=object())
    recipe = SimpleNamespace(
        resources={"deepVerificationPolicy": "required_gates"},
        tactics=(_tactic(perspectives=("south",)),),
    )
    analysis = SimpleNamespace(
        decisions=(
            SimpleNamespace(decision_id="PLAY-EAST"),
            SimpleNamespace(decision_id="PLAY-SOUTH"),
        ),
        laffa_evidence=SimpleNamespace(
            critical_decision_ids=("PLAY-EAST",)
        ),
    )

    results = service._verify_tactics(
        object(),
        analysis,
        recipe=recipe,
        cancel_check=None,
    )

    assert results == ("PLAY-SOUTH",)
    assert verified == ["PLAY-SOUTH"]


def test_tactical_quota_uses_the_post_trace_blocked_union_before_ranking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def snapshot(decision_id: str, ordinal: int):
        return SimpleNamespace(
            decision=SimpleNamespace(
                decision_id=decision_id,
                decision_kind="play",
                forced=False,
                legal_actions=({"card": "S7"}, {"card": "S8"}),
                ordinal=ordinal,
                acting_seat=SimpleNamespace(name="SOUTH"),
            ),
            state=SimpleNamespace(hands=(1, 1, 1, 1)),
        )

    snapshots = (
        snapshot("PLAY-UNRESOLVED", 30),
        snapshot("PLAY-COMPETENT", 29),
        snapshot("PLAY-ELIGIBLE", 1),
    )
    monkeypatch.setattr(
        research_service_module,
        "TraceDecisionProjectionStoreV1",
        lambda trace, rules: SimpleNamespace(
            replay=SimpleNamespace(decisions=snapshots)
        ),
    )
    verified: list[str] = []
    service = object.__new__(ResearchFoundryServiceV1)
    service.tactical_verifier = SimpleNamespace(
        maximum_remaining_cards=32,
        verify_decision=lambda item, analysis, cancel_check: (
            verified.append(item.decision.decision_id)
            or item.decision.decision_id
        ),
    )
    service.maximum_tactical_decisions = 1
    service.simulator = SimpleNamespace(rules=object())
    recipe = SimpleNamespace(
        resources={"deepVerificationPolicy": "required_gates"},
        tactics=(_tactic(),),
    )
    analysis = SimpleNamespace(
        decisions=tuple(
            SimpleNamespace(decision_id=item.decision.decision_id)
            for item in snapshots
        ),
        laffa_evidence=SimpleNamespace(
            critical_decision_ids=("PLAY-UNRESOLVED",)
        ),
        adjudication_evidence={
            "competentDecisionIds": ["PLAY-COMPETENT"],
            "unresolvedDecisionIds": ["PLAY-UNRESOLVED"],
        },
        unresolved_decision_ids=("PLAY-UNRESOLVED",),
    )

    results = service._verify_tactics(
        object(),
        analysis,
        recipe=recipe,
        cancel_check=None,
    )

    assert results == ("PLAY-ELIGIBLE",)
    assert verified == ["PLAY-ELIGIBLE"]


def test_tactical_verification_provenance_records_policy_cost_and_influence() -> None:
    preference = SimpleNamespace(
        resources={"deepVerificationPolicy": "required_gates"},
        tactics=(_tactic(),),
    )
    evidence = _tactical_verification_provenance(preference, 3)

    assert evidence == {
        "deepVerificationPolicy": "required_gates",
        "tacticalVerificationRequired": True,
        "tacticalVerificationCount": 3,
    }
