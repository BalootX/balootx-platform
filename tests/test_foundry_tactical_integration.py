from __future__ import annotations

import json

import pytest

from baloot_platform.challenge.evidence_v3 import (
    DecisionLedgerV1,
    DecisionPerspectiveEvidenceV3,
    LaffaEvidenceV2,
    ScenarioGraphV1,
    ScenarioNodeV1,
    TruthLabel,
    TruthMode,
)
from baloot_platform.challenge.oracle import (
    ActualDealActionOracleV1,
    ActualDealDecisionOracleV1,
    TacticalVerificationResultV1,
)
from baloot_platform.challenge.tactical_evidence import TacticalEvidenceV1, TacticalKind
from baloot_platform.challenge.tactical_integration import (
    integrate_verified_tactical_evidence,
)
from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Seat


def _ledger() -> DecisionLedgerV1:
    actions = (Card.parse("H7").to_dict(), Card.parse("HA").to_dict())
    return DecisionLedgerV1.build(
        "game-1",
        ({
            "decisionId": "decision-1",
            "ordinal": 12,
            "actor": "south",
            "decisionKind": "play",
            "phase": "play",
            "publicPrefixHash": "1" * 64,
            "completeStateHash": "2" * 64,
            "legalActions": actions,
            "chosenAction": actions[0],
            "resultingEventHash": "3" * 64,
        },),
    )


def _base(ledger: DecisionLedgerV1) -> LaffaEvidenceV2:
    values = [
        DecisionPerspectiveEvidenceV3(
            "decision-1",
            Seat.SOUTH,
            Seat.SOUTH,
            TruthMode.ACTOR_OBSERVER,
            "4" * 64,
            False,
            TruthLabel.ROBUST_BEST_KNOWN,
            0.9,
            {"outcomeImpact": 80.0, "confidence": 90.0},
            Card.parse("H7").to_dict(),
            "choose_acting_action",
        )
    ]
    for seat in (Seat.NORTH, Seat.EAST, Seat.WEST):
        values.append(
            DecisionPerspectiveEvidenceV3(
                "decision-1",
                Seat.SOUTH,
                seat,
                TruthMode.SEAT_EXPERIENCE,
                str(int(seat) + 5) * 64,
                False,
                TruthLabel.UNRESOLVED,
                0.0,
                {"threatVisibility": 25.0},
                None,
                "identify_threat",
            )
        )
    graph = ScenarioGraphV1.build(
        (ScenarioNodeV1("decision:12", 12, "decision", "decision-1", "9" * 64),),
        (),
    )
    return LaffaEvidenceV2.build(
        ledger,
        values,
        critical_threshold=70.0,
        minimum_critical_confidence=0.75,
        scenario_graph=graph,
    )


def _verification(*, status: str = "verified") -> TacticalVerificationResultV1:
    selected = ActualDealActionOracleV1(
        Card.parse("H7").to_dict(),
        10.0,
        -5.0,
        20.0,
        True,
        False,
        False,
        100,
        ("H7", "HJ", "H8"),
        (),
        "exact",
        "a" * 64,
    )
    alternative = ActualDealActionOracleV1(
        Card.parse("HA").to_dict(),
        100.0,
        5.0,
        0.0,
        False,
        False,
        False,
        100,
        ("HA", "HJ", "SA"),
        (),
        "exact",
        "b" * 64,
    )
    oracle = ActualDealDecisionOracleV1(
        "decision-1",
        TruthMode.ACTUAL_DEAL_ORACLE,
        Card.parse("H7").to_dict(),
        (selected, alternative),
        True,
        "c" * 64,
    )
    evidence = (
        TacticalEvidenceV1(
            TacticalKind.ONLY_WINNING_MOVE,
            "H7",
            ("HA",),
            100.0,
            2,
            ("entry",),
            (("outcomeBoundaryCrossed", 1.0),),
            "d" * 64,
        ),
        TacticalEvidenceV1(
            TacticalKind.SMALL_CARD_BIG_GAIN,
            "H7",
            ("HA",),
            90.0,
            2,
            ("entry", "promotion"),
            (("minimumDelayedGain", 8.0),),
            "e" * 64,
        ),
    )
    return TacticalVerificationResultV1(oracle, evidence, status, "f" * 64)


def test_verified_tactics_remain_a_separate_truth_mode_and_build_causal_graph():
    ledger = _ledger()
    base = _base(ledger)
    enriched = integrate_verified_tactical_evidence(
        base,
        ledger,
        (_verification(),),
        critical_threshold=70.0,
    )

    assert [item.to_dict() for item in enriched.perspectives[:4]] == [
        item.to_dict() for item in base.perspectives
    ]
    actor = next(
        item for item in enriched.perspectives
        if item.truth_mode == TruthMode.ACTOR_OBSERVER
    )
    oracle = next(
        item for item in enriched.perspectives
        if item.truth_mode == TruthMode.ACTUAL_DEAL_ORACLE
    )
    assert actor.recommended_action == Card.parse("H7").to_dict()
    assert actor.evidence == base.perspectives[0].evidence
    assert oracle.recommended_action is None
    assert oracle.prompt_contract == "post_reveal_analysis"
    assert oracle.truth_label == TruthLabel.EXACT_OPTIMAL
    assert oracle.evidence["uniqueness"] == 100.0
    assert oracle.evidence["tacticalDepth"] == 90.0
    assert enriched.multi_angle_yield == base.multi_angle_yield == 0

    graph = enriched.scenario_graph
    assert graph is not None
    assert {item.kind for item in graph.nodes} == {
        "decision",
        "actual_deal_oracle_verification",
        "counterfactual_tactical_payoff",
    }
    assert {item.relation for item in graph.edges} == {
        "post_reveal_verification",
        "only_winning_move",
        "small_card_big_gain",
    }
    assert all(
        item.ordinal >= 12
        for item in graph.nodes
        if item.kind == "counterfactual_tactical_payoff"
    )

    serialized = json.dumps(enriched.to_dict(), sort_keys=True)
    assert "principalVariation" not in serialized
    assert '"rank": "ace"' not in serialized
    loaded = LaffaEvidenceV2.from_dict(enriched.to_dict(), ledger=ledger)
    assert loaded.sha256 == enriched.sha256


def test_tactical_integration_skips_unresolved_and_fails_closed_on_bad_inputs():
    ledger = _ledger()
    base = _base(ledger)
    unresolved = integrate_verified_tactical_evidence(
        base,
        ledger,
        (_verification(status="unresolved"),),
        critical_threshold=70.0,
    )
    assert unresolved.scenario_graph is not None
    assert unresolved.scenario_graph.sha256 == base.scenario_graph.sha256
    assert all(
        item.truth_mode != TruthMode.ACTUAL_DEAL_ORACLE
        for item in unresolved.perspectives
    )

    with pytest.raises(ValueError, match="duplicate"):
        integrate_verified_tactical_evidence(
            base,
            ledger,
            (_verification(), _verification()),
            critical_threshold=70.0,
        )

    with pytest.raises(ValueError, match="post-reveal"):
        DecisionPerspectiveEvidenceV3(
            "decision-1",
            Seat.SOUTH,
            Seat.SOUTH,
            TruthMode.ACTUAL_DEAL_ORACLE,
            "0" * 64,
            False,
            TruthLabel.UNRESOLVED,
            1.0,
            {"confidence": 100.0},
            None,
            "choose_acting_action",
        )


def test_project_bidding_tactic_maps_to_outcome_without_changing_actor_truth():
    ledger = _ledger()
    base = _base(ledger)
    source = _verification()
    project = TacticalEvidenceV1(
        TacticalKind.PROJECT_BIDDING,
        "H7",
        ("HA",),
        75.0,
        0,
        (),
        (("projectConsequenceGain", 15.0),),
        "1" * 64,
    )
    verification = TacticalVerificationResultV1(
        source.oracle,
        (project,),
        "verified",
        "2" * 64,
    )
    enriched = integrate_verified_tactical_evidence(
        base,
        ledger,
        (verification,),
        critical_threshold=70.0,
    )
    actor = next(
        item for item in enriched.perspectives
        if item.truth_mode == TruthMode.ACTOR_OBSERVER
    )
    oracle = next(
        item for item in enriched.perspectives
        if item.truth_mode == TruthMode.ACTUAL_DEAL_ORACLE
    )
    assert actor.evidence == base.perspectives[0].evidence
    assert oracle.evidence["outcomeImpact"] == 75.0
    assert "project_bidding" in {
        edge.relation for edge in enriched.scenario_graph.edges
    }
