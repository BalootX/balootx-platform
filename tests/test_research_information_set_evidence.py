from __future__ import annotations

from baloot_platform.research import (
    InformationSetEvidenceV1,
    SequentialResponseEvidenceV1,
)
from baloot_platform.core.cards import Card
from baloot_platform.core.dealing import DealLayoutV1
from baloot_platform.core.enums import Contract, Seat
from baloot_platform.service.container import build_container


def _info():
    return {
        "schemaVersion": 1,
        "observer": "south",
        "hand": [{"id": 1}, {"id": 2}],
        "publicHistory": [],
    }


def _actions():
    return ({"id": 1}, {"id": 2})


def _alternatives(status: str = "heuristic_complete"):
    return (
        {
            "action": {"id": 1},
            "searchCompletionStatus": status,
            "exactShare": 1.0 if status == "exact" else 0.0,
        },
        {
            "action": {"id": 2},
            "searchCompletionStatus": status,
            "exactShare": 1.0 if status == "exact" else 0.0,
        },
    )


def test_paired_information_set_proof_is_complete_and_deterministic():
    arguments = dict(
        observer="south",
        information_state=_info(),
        legal_actions=_actions(),
        alternatives=_alternatives(),
        forced=False,
        analysis_hash="a" * 64,
        method="constructive_uniform_hidden_deals+hard_facts_only",
        hidden_deal_support=24,
        effective_sample_size=24.0,
    )
    first = InformationSetEvidenceV1.build(**arguments)
    second = InformationSetEvidenceV1.build(**arguments)
    assert first == second
    assert first.status == "paired_sampled_information_set"
    assert first.observer_safe is True
    assert first.action_coverage_complete is True
    assert first.actor_choice_bound_once is True
    assert first.to_dict()["hiddenStateIdentifiersExposed"] is False


def test_exhaustive_exact_information_set_is_labeled_exact():
    proof = InformationSetEvidenceV1.build(
        observer="south",
        information_state=_info(),
        legal_actions=_actions(),
        alternatives=_alternatives("exact"),
        forced=False,
        analysis_hash="b" * 64,
        method="exhaustive_constrained_hidden_deals+hard_facts_only",
        hidden_deal_support=8,
        effective_sample_size=8.0,
    )
    assert proof.status == "exact_information_set"


def test_missing_action_or_hidden_state_method_fails_closed():
    missing = InformationSetEvidenceV1.build(
        observer="south",
        information_state=_info(),
        legal_actions=_actions(),
        alternatives=(_alternatives()[0],),
        forced=False,
        analysis_hash="c" * 64,
        method="constructive_uniform_hidden_deals+hard_facts_only",
        hidden_deal_support=24,
        effective_sample_size=24.0,
    )
    assert missing.status == "incomplete"
    assert missing.action_coverage_complete is False

    leaked = InformationSetEvidenceV1.build(
        observer="south",
        information_state=_info(),
        legal_actions=_actions(),
        alternatives=_alternatives(),
        forced=False,
        analysis_hash="d" * 64,
        method="actual_deal_oracle",
        hidden_deal_support=1,
        effective_sample_size=1.0,
    )
    assert leaked.status == "incomplete"
    assert leaked.observer_safe is False


def test_sequential_response_distinguishes_exact_sampled_and_counterexample():
    sampled_information = InformationSetEvidenceV1.build(
        observer="south",
        information_state=_info(),
        legal_actions=_actions(),
        alternatives=_alternatives(),
        forced=False,
        analysis_hash="e" * 64,
        method="constructive_uniform_hidden_deals+hard_facts_only",
        hidden_deal_support=24,
        effective_sample_size=24.0,
    )
    sampled = SequentialResponseEvidenceV1.build(
        actual_action={"id": 1},
        recommended_action={"id": 1},
        legal_actions=_actions(),
        alternatives=_alternatives(),
        information_set=sampled_information,
        expected_regret=0.0,
        forced=False,
    )
    assert sampled.status == "robust_policy_conditioned_response"
    assert sampled.exact_continuation_coverage is False
    assert sampled.material_counterexample_found is False

    exact_information = InformationSetEvidenceV1.build(
        observer="south",
        information_state=_info(),
        legal_actions=_actions(),
        alternatives=_alternatives("exact"),
        forced=False,
        analysis_hash="f" * 64,
        method="exhaustive_constrained_hidden_deals+hard_facts_only",
        hidden_deal_support=8,
        effective_sample_size=8.0,
    )
    exact = SequentialResponseEvidenceV1.build(
        actual_action={"id": 2},
        recommended_action={"id": 1},
        legal_actions=_actions(),
        alternatives=_alternatives("exact"),
        information_set=exact_information,
        expected_regret=12.0,
        forced=False,
    )
    assert exact.status == "exact_sequential_response"
    assert exact.exact_continuation_coverage is True
    assert exact.material_counterexample_found is True


def test_real_complete_laffa_has_actor_safe_proof_for_every_choice():
    service = build_container().research
    trace = service.simulator.simulate(17).final_trace
    analysis = service.analyzers["screening"].analyze_trace(trace)
    audit = service._decision_audit(trace, analysis)
    choice_ids = {item.decision_id for item in trace.decision_ledger.records}
    decisions = [
        item for item in audit["decisions"] if item["decisionId"] in choice_ids
    ]
    assert audit["schemaVersion"] == 2
    assert audit["complete"] is True
    assert audit["informationSetComplete"] is True
    assert audit["sequentialResponseComplete"] is True
    assert len(decisions) == len(choice_ids)
    assert all(item["informationSetEvidence"]["observerSafe"] for item in decisions)
    assert all(
        item["informationSetEvidence"]["actionCoverageComplete"]
        for item in decisions
    )
    assert all(
        item["informationSetEvidence"]["actorChoiceBoundOnce"]
        for item in decisions
    )
    assert all(
        item["sequentialResponseEvidence"]["allAlternativesEvaluated"]
        for item in decisions
    )
    assert all(
        item["sequentialResponseEvidence"]["pairedActorSafeCohort"]
        for item in decisions
    )


def test_escalation_audit_has_actor_safe_legal_choices_and_remains_provisional():
    service = build_container().research
    deck_order = tuple(range(32))
    dealer = Seat.EAST
    buyer = Seat.NORTH
    layout = DealLayoutV1.build(
        service.simulator.structural_domain, dealer=dealer, buyer=buyer
    )
    purchase_suit = Card(deck_order[layout.purchase_position]).suit
    auction = service.auction_compiler.compile(
        dealer=dealer,
        purchase_suit=purchase_suit,
        rules=service.simulator.rules,
        required_buyer=buyer,
        required_contracts=(Contract.HOKUM,),
        required_purchase_round=1,
        required_trump=purchase_suit,
    )
    assert auction is not None
    escalation = service.escalation_compiler.compile(
        dealer=dealer,
        declarer=auction.declarer,
        contract=auction.contract,
        match_score=(0, 0),
        rules=service.simulator.rules,
        required_level="double",
        required_mode="locked",
    )
    assert escalation is not None and escalation.actions
    simulation = service.simulator.simulate_preordered_deck(
        7,
        deck_order,
        deal_generator_provenance={
            "pluginId": "test_escalation_audit",
            "profileHash": "1" * 64,
        },
        dealer=dealer,
        scripted_bids=tuple(item.to_dict() for item in auction.actions),
        scripted_escalations=tuple(item.to_dict() for item in escalation.actions),
    )
    trace = simulation.final_trace
    analysis = service.analyzers["screening"].analyze_trace(trace)
    audit = service._decision_audit(trace, analysis)
    decisions = [
        item for item in audit["decisions"] if item["decisionKind"] == "escalation"
    ]

    assert audit["complete"] is True
    assert audit["incompleteGovernedActionCount"] == 0
    assert len(decisions) == len(escalation.actions)
    first = decisions[0]
    assert first["forced"] is False
    assert first["recommendationAvailable"] is False
    assert first["truthLabel"] == "provisional"
    assert first["confidence"] == 0.0
    assert first["actualAction"] in first["legalActions"]
    assert {item["action"] for item in first["legalActions"]} >= {
        "decline",
        "double",
    }
    assert first["informationSetEvidence"]["observerSafe"] is True
    assert first["informationSetEvidence"]["actionCoverageComplete"] is True
    assert first["informationSetEvidence"]["status"] == "incomplete"
    assert first["sequentialResponseEvidence"]["status"] == "incomplete"
    assert first["selectionExplanation"]["available"] is True
    assert first["selectionExplanation"]["source"] == (
        "recorded_compiled_escalation_plan"
    )
    assert first["analyticalExplanation"]["status"] == "unresolved"
    assert first["convergence"] == "exact_legal_actions_strategy_unresolved"
    assert first["provenance"]["legalityAuthority"] == (
        "reference_escalation_interpreter_v1"
    )
    assert first["provenance"]["analysisHash"]
    assert first["informationSetEvidence"]["hiddenStateIdentifiersExposed"] is False

    from baloot_platform.service.api import ResearchDecisionAuditResponse

    assert ResearchDecisionAuditResponse.model_validate(first).decisionKind == "escalation"
