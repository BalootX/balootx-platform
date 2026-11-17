from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from baloot_platform.challenge.evidence_v3 import (
    DecisionLedgerV1,
    DecisionPerspectiveEvidenceV3,
    LaffaEvidenceV2,
    ObserverProjectionV1,
    ScenarioEdgeV1,
    ScenarioGraphV1,
    ScenarioNodeV1,
    TruthLabel,
    TruthMode,
    relative_team_value,
)
from baloot_platform.challenge.perspective_pipeline import (
    build_all_observer_projections,
    build_decision_perspective_evidence,
    derive_seat_experience_evidence,
)
from baloot_platform.challenge.tactical_evidence import (
    CausalActionLineV1,
    CausalTacticalDetectorV1,
    TacticalKind,
    detect_tactical_evidence,
)
from baloot_platform.challenge.foundry import MultiPerspectiveFoundryV1
from baloot_platform.challenge.profiles import (
    PerspectivePolicyV1,
    SelectionProfileV1,
    ThrillProfileV1,
)
from baloot_platform.challenge.scoring import ProfileChallengeScorerV1
from baloot_platform.challenge.symmetry import (
    SeatRotationV1,
    SuitRelabelingV1,
    audit_rotated_perspective_evidence,
)
from baloot_platform.plugins.base import (
    ChallengeScorerPlugin,
    PerspectiveAnalyzerPlugin,
    TacticalDetectorPlugin,
)
from baloot_platform.core.cards import Card, mask_to_cards
from baloot_platform.core.enums import Contract, Suit
from baloot_platform.core.scenario import ScenarioRequest, generate_episode
from baloot_platform.core.enums import Seat, Team
from baloot_platform.learning.store import LearningStore
from baloot_platform.audit.models import AuditedAlternative, DecisionAudit, GameAuditReport


def _decision(decision_id: str, ordinal: int, actor: Seat, *, forced: bool = False) -> dict:
    legal = [{"code": "S7"}] if forced else [{"code": "S7"}, {"code": "S8"}]
    return {
        "decisionId": decision_id,
        "ordinal": ordinal,
        "actor": actor.name.lower(),
        "decisionKind": "play",
        "phase": "playing",
        "publicState": {"observer": actor.name.lower(), "ordinal": ordinal},
        "legalActions": legal,
        "selectedAction": legal[0],
    }


def _four_views(decision_id: str, actor: Seat, peak: float, *, forced: bool = False):
    values = [
        DecisionPerspectiveEvidenceV3(
            decision_id=decision_id,
            acting_seat=actor,
            observer_seat=actor,
            truth_mode=TruthMode.ACTOR_OBSERVER,
            projection_hash=f"{decision_id}-{actor.name}",
            forced=forced,
            truth_label=TruthLabel.ROBUST_BEST_KNOWN,
            confidence=0.9,
            evidence={"outcomeImpact": peak, "robustness": 80},
            recommended_action=None if forced else {"code": "S7"},
            prompt_contract="choose_acting_action",
        )
    ]
    for observer in Seat:
        if observer == actor:
            continue
        values.append(
            DecisionPerspectiveEvidenceV3(
                decision_id=decision_id,
                acting_seat=actor,
                observer_seat=observer,
                truth_mode=TruthMode.SEAT_EXPERIENCE,
                projection_hash=f"{decision_id}-{observer.name}",
                forced=forced,
                truth_label=TruthLabel.ROBUST_BEST_KNOWN,
                confidence=0.75,
                evidence={"threatVisibility": peak / 2},
                prompt_contract="identify_threat",
            )
        )
    return values


def test_laffa_requires_every_decision_and_all_four_isolated_views():
    ledger = DecisionLedgerV1.build(
        "GAME-1",
        [_decision("D1", 1, Seat.NORTH), _decision("D2", 2, Seat.EAST)],
    )
    complete = _four_views("D1", Seat.NORTH, 91) + _four_views("D2", Seat.EAST, 82)
    graph = ScenarioGraphV1.build(
        (
            ScenarioNodeV1("setup", 1, "decision", "D1", "a" * 64),
            ScenarioNodeV1("payoff", 2, "decision", "D2", "b" * 64),
        ),
        (ScenarioEdgeV1("setup", "payoff", "enables", 88),),
    )
    report = LaffaEvidenceV2.build(ledger, complete, scenario_graph=graph)
    assert report.perspective_breadth == 2
    assert report.classification == "two_seat_duel"
    assert {item.seat for item in report.seat_journeys} == set(Seat)
    assert next(item for item in report.seat_journeys if item.seat == Seat.SOUTH).status == "not_applicable"
    assert report.overall_peak == 91
    assert report.second_peak == 82
    assert report.top_k_thrill_mass == 173
    assert report.causal_chain_strength == 88
    assert report.outcome_importance == 91
    assert report.narrative_arc == 90
    assert report.scenario_graph_sha256 == graph.sha256
    assert len(report.team_journeys) == 2

    with pytest.raises(ValueError, match="four isolated"):
        LaffaEvidenceV2.build(ledger, complete[:-1])
    with pytest.raises(ValueError, match="cover every ledger"):
        LaffaEvidenceV2.build(ledger, complete[:4])


def test_non_actor_cannot_receive_an_actor_action_recommendation():
    with pytest.raises(ValueError, match="cannot receive"):
        DecisionPerspectiveEvidenceV3(
            decision_id="D1",
            acting_seat=Seat.NORTH,
            observer_seat=Seat.SOUTH,
            truth_mode=TruthMode.SEAT_EXPERIENCE,
            projection_hash="safe",
            forced=False,
            truth_label=TruthLabel.ROBUST_BEST_KNOWN,
            confidence=0.8,
            evidence={"surprise": 70},
            recommended_action={"code": "S7"},
            prompt_contract="predict_opponent_play",
        )


def test_scenario_graph_rejects_backward_causality():
    with pytest.raises(ValueError, match="backwards"):
        ScenarioGraphV1.build(
            (
                ScenarioNodeV1("later", 5, "payoff", "D2", "a" * 64),
                ScenarioNodeV1("earlier", 1, "setup", "D1", "b" * 64),
            ),
            (ScenarioEdgeV1("later", "earlier", "causes", 50),),
        )


def test_selection_profiles_do_not_rewrite_raw_evidence():
    item = _four_views("D1", Seat.WEST, 90)[0]
    assert item.selection_score({"outcomeImpact": 3, "robustness": 1}) == 87.5
    assert item.evidence == {"outcomeImpact": 90, "robustness": 80}


def test_decision_score_excludes_non_applicable_dimensions_and_is_auditable():
    item = _four_views("D1", Seat.WEST, 90)[0]
    profile = SelectionProfileV1.build({
        "schemaVersion": 1,
        "id": "balanced-decision",
        "revision": "1",
        "weights": {
            "outcomeImpact": 18,
            "robustness": 8,
            "kaboutImpact": 8,
            "delayedGain": 6,
        },
        "allowedTruthLabels": ["exact_optimal", "robust_best_known"],
        "minimumConfidence": 0,
        "minimumScore": 0,
    })
    score = profile.score(item)
    breakdown = profile.breakdown(item)

    assert score.passed is True
    assert score.score == pytest.approx((90 * 18 + 80 * 8) / 26)
    assert dict(breakdown.dimensions) == {
        "outcomeImpact": 90.0,
        "robustness": 80.0,
    }
    assert set(dict(breakdown.normalized_weights)) == {
        "outcomeImpact", "robustness",
    }
    assert breakdown.sha256 == profile.breakdown(item).sha256


def test_selection_profile_hashes_and_validates_the_tactical_play_gate():
    base = {
        "schemaVersion": 1,
        "id": "professional-decision",
        "revision": "3",
        "weights": {"outcomeImpact": 1, "robustness": 1},
        "allowedTruthLabels": ["exact_optimal", "robust_best_known"],
        "minimumConfidence": 0.8,
        "minimumScore": 70,
    }
    open_profile = SelectionProfileV1.build(base)
    tactical_profile = SelectionProfileV1.build({
        **base,
        "requireVerifiedTacticalPlay": True,
    })
    assert open_profile.require_verified_tactical_play is False
    assert tactical_profile.require_verified_tactical_play is True
    assert tactical_profile.sha256 != open_profile.sha256
    assert tactical_profile.to_dict()["requireVerifiedTacticalPlay"] is True
    with pytest.raises(ValueError, match="must be boolean"):
        SelectionProfileV1.build({
            **base,
            "requireVerifiedTacticalPlay": "true",
        })


def test_selection_profile_qualifies_candidate_specific_surrounding_play():
    profile = SelectionProfileV1.build({
        "schemaVersion": 1,
        "id": "professional-complete-laffa",
        "revision": "1",
        "weights": {"outcomeImpact": 1, "robustness": 1},
        "allowedTruthLabels": ["exact_optimal", "robust_best_known"],
        "minimumConfidence": 0.8,
        "minimumScore": 70,
        "minimumSurroundingQuality": 75,
        "minimumSurroundingConfidence": 0.75,
        "minimumSurroundingMateriality": 25,
        "maximumSurroundingMaterialMistakes": 0,
        "maximumSurroundingUnverifiedDecisions": 0,
        "maximumSurroundingCatastrophicCrossings": 0,
    })
    candidate = SimpleNamespace(
        decision_id="target", ordinal=2, forced=False, expected_regret=20,
        probability_drop=1, confidence=1, catastrophic_crossing=True,
        alternatives=({"expectedMargin": 20, "winProbability": 1}, {"expectedMargin": 0, "winProbability": 0}),
        difficulty=100, debate_score=0,
    )
    strong = SimpleNamespace(
        decision_id="strong", ordinal=1, forced=False, expected_regret=1,
        probability_drop=0.05, confidence=1, catastrophic_crossing=False,
        alternatives=({"expectedMargin": 5, "winProbability": 0.6}, {"expectedMargin": 4, "winProbability": 0.5}),
        difficulty=30, debate_score=0,
    )
    weak = SimpleNamespace(
        decision_id="weak", ordinal=3, forced=False, expected_regret=5,
        probability_drop=0.2, confidence=0.5, catastrophic_crossing=True,
        alternatives=({"expectedMargin": 10, "winProbability": 0.8}, {"expectedMargin": 0, "winProbability": 0.2}),
        difficulty=90, debate_score=0,
    )
    failed = profile.qualify_surrounding(
        (candidate, strong, weak),
        candidate_decision_id="target",
        trace_analysis_hash="a" * 64,
    )
    assert failed.passed is False
    assert failed.material_mistakes == 1
    assert failed.unverified_decisions == 1
    assert failed.catastrophic_crossings == 1
    assert failed.preceding_decisions == 1
    assert failed.following_decisions == 1

    passed = profile.qualify_surrounding(
        (candidate, strong),
        candidate_decision_id="target",
        trace_analysis_hash="a" * 64,
    )
    assert passed.passed is True
    assert passed.minimum_quality == 90
    assert passed.sha256 == profile.qualify_surrounding(
        (candidate, strong),
        candidate_decision_id="target",
        trace_analysis_hash="a" * 64,
    ).sha256


def test_named_selection_and_thrill_profiles_apply_hard_truth_gates():
    actor = _four_views("D1", Seat.WEST, 90)[0]
    selection = SelectionProfileV1.build({
        "schemaVersion": 1,
        "id": "professional-decision",
        "revision": "1",
        "weights": {"outcomeImpact": 3, "robustness": 1},
        "allowedTruthLabels": ["exact_optimal", "robust_best_known"],
        "minimumConfidence": 0.8,
        "minimumScore": 80,
    })
    result = selection.score(actor)
    assert result.passed and result.score == 87.5
    assert result.score_family == "decision_content_selection"
    assert selection.score(_four_views("D1", Seat.WEST, 90)[1]).passed is False

    ledger = DecisionLedgerV1.build(
        "GAME-1", [_decision("D1", 1, Seat.NORTH), _decision("D2", 2, Seat.EAST)]
    )
    laffa = LaffaEvidenceV2.build(
        ledger, _four_views("D1", Seat.NORTH, 91) + _four_views("D2", Seat.EAST, 82)
    )
    thrill = ThrillProfileV1.build({
        "schemaVersion": 1,
        "id": "two-peak-laffa",
        "revision": "1",
        "weights": {"overallPeak": 2, "secondPeak": 1, "criticalDensity": 1},
        "minimumCriticalPoints": 2,
        "minimumScore": 80,
    })
    laffa_score = thrill.score(laffa)
    assert laffa_score.passed
    assert laffa_score.score_family == "laffa_content_selection"

    perspective = PerspectivePolicyV1.build({
        "schemaVersion": 1,
        "id": "all-four",
        "revision": "1",
        "actorPrompt": "choose_acting_action",
        "seatExperiencePrompts": ["identify_threat", "read_partner_signal"],
        "requireFourProjections": True,
    })
    assert perspective.require_four_projections is True
    with pytest.raises(ValueError, match="non-actor"):
        PerspectivePolicyV1.build({
            "id": "bad", "revision": "1",
            "actorPrompt": "choose_acting_action",
            "seatExperiencePrompts": ["choose_acting_action"],
        })


def test_confidence_alone_never_turns_an_ordinary_decision_into_a_critical_point():
    item = DecisionPerspectiveEvidenceV3(
        decision_id="D1",
        acting_seat=Seat.NORTH,
        observer_seat=Seat.NORTH,
        truth_mode=TruthMode.ACTOR_OBSERVER,
        projection_hash="safe",
        forced=False,
        truth_label=TruthLabel.ROBUST_BEST_KNOWN,
        confidence=1.0,
        evidence={"outcomeImpact": 10, "robustness": 100, "confidence": 100},
        recommended_action={"code": "S7"},
        prompt_contract="choose_acting_action",
    )
    views = [item, *_four_views("D1", Seat.NORTH, 10)[1:]]
    ledger = DecisionLedgerV1.build("GAME-1", [_decision("D1", 1, Seat.NORTH)])
    laffa = LaffaEvidenceV2.build(ledger, views)
    assert item.peak == 10
    assert laffa.critical_decision_ids == ()


def test_low_confidence_peak_cannot_inflate_complete_laffa_score():
    low = _four_views("D1", Seat.NORTH, 100)
    low[0] = replace(low[0], confidence=0.2)
    high = _four_views("D2", Seat.EAST, 80)
    ledger = DecisionLedgerV1.build(
        "GAME-CONFIDENCE",
        [_decision("D1", 1, Seat.NORTH), _decision("D2", 2, Seat.EAST)],
    )
    laffa = LaffaEvidenceV2.build(ledger, [*low, *high])
    assert laffa.overall_peak == 80
    assert laffa.critical_decision_ids == ("D2",)


def test_laffa_thrill_score_is_plugin_driven_and_rotation_invariant():
    ledger = DecisionLedgerV1.build(
        "GAME-ROTATION",
        [_decision("D1", 1, Seat.NORTH), _decision("D2", 2, Seat.EAST)],
    )
    graph = ScenarioGraphV1.build(
        (
            ScenarioNodeV1("D1", 1, "decision", "D1", "1" * 64),
            ScenarioNodeV1("D2", 2, "decision", "D2", "2" * 64),
        ),
        (ScenarioEdgeV1("D1", "D2", "partner_enablement", 88),),
    )
    original = _four_views("D1", Seat.NORTH, 91) + _four_views(
        "D2", Seat.EAST, 82
    )
    rotation = SeatRotationV1(1)
    rotated = [
        DecisionPerspectiveEvidenceV3(
            decision_id=item.decision_id,
            acting_seat=rotation.map_seat(item.acting_seat),
            observer_seat=rotation.map_seat(item.observer_seat),
            truth_mode=item.truth_mode,
            projection_hash=f"rotated-{item.projection_hash}",
            forced=item.forced,
            truth_label=item.truth_label,
            confidence=item.confidence,
            evidence=dict(item.evidence),
            recommended_action=item.recommended_action,
            prompt_contract=item.prompt_contract,
        )
        for item in original
    ]
    rotated_ledger = DecisionLedgerV1.build(
        "GAME-ROTATION-MAPPED",
        [
            _decision(
                item.decision_id,
                item.ordinal,
                rotation.map_seat(item.acting_seat),
            )
            for item in ledger.records
        ],
    )
    first = LaffaEvidenceV2.build(ledger, original, scenario_graph=graph)
    second = LaffaEvidenceV2.build(
        rotated_ledger,
        rotated,
        scenario_graph=graph,
    )
    profile = ThrillProfileV1.build({
        "schemaVersion": 1,
        "id": "rotation-neutral-thrill",
        "revision": "1",
        "weights": {
            "overallPeak": 2,
            "secondPeak": 1,
            "perspectiveBreadth": 1,
            "causalChainStrength": 1,
            "outcomeImportance": 1,
            "narrativeArc": 1,
        },
        "minimumCriticalPoints": 1,
        "minimumScore": 0,
    })
    scorer = ProfileChallengeScorerV1()
    assert isinstance(scorer, ChallengeScorerPlugin)
    first_score = scorer.score(first, profile)
    second_score = scorer.score(second, profile)
    assert first_score.score == second_score.score
    assert first_score.passed == second_score.passed


def test_team_relative_normalization_changes_sign_under_team_swap():
    assert relative_team_value((80, 120), Team.WEST_EAST) == 40
    assert relative_team_value((80, 120), Team.NORTH_SOUTH) == -40
    assert relative_team_value((80, 120), Seat.EAST) == 40


def test_seat_team_and_suit_symmetry_contracts_are_bijective():
    values = (80.0, 120.0)
    for turns in range(4):
        transform = SeatRotationV1(turns)
        for seat in Seat:
            assert transform.inverse_seat(transform.map_seat(seat)) == seat
            mapped_values = transform.map_team_values(values)
            assert relative_team_value(values, seat) == relative_team_value(
                mapped_values, transform.map_seat(seat)
            )
            assert SeatRotationV1.observer_offset(seat, seat.partner) == 2

    relabel = SuitRelabelingV1.from_mapping({
        Suit.SPADES: Suit.HEARTS,
        Suit.HEARTS: Suit.CLUBS,
        Suit.CLUBS: Suit.DIAMONDS,
        Suit.DIAMONDS: Suit.SPADES,
    })
    card = Card.parse("SJ")
    mapped = relabel.map_card(card)
    assert mapped.suit == Suit.HEARTS and mapped.rank == card.rank
    with pytest.raises(ValueError, match="bijection"):
        SuitRelabelingV1((Suit.SPADES,) * 4)


def test_rotation_audit_compares_independent_outputs_after_inverse_mapping():
    original = tuple(_four_views("D1", Seat.NORTH, 90))
    rotation = SeatRotationV1(1)
    rotated = tuple(
        DecisionPerspectiveEvidenceV3(
            decision_id=item.decision_id,
            acting_seat=rotation.map_seat(item.acting_seat),
            observer_seat=rotation.map_seat(item.observer_seat),
            truth_mode=item.truth_mode,
            projection_hash=f"rotated-{item.projection_hash}",
            forced=item.forced,
            truth_label=item.truth_label,
            confidence=item.confidence,
            evidence=dict(item.evidence),
            recommended_action=item.recommended_action,
            prompt_contract=item.prompt_contract,
        )
        for item in original
    )
    passed = audit_rotated_perspective_evidence(original, rotated, quarter_turns=1)
    assert passed.passed and passed.compared_records == 4

    broken = list(rotated)
    item = broken[0]
    broken[0] = DecisionPerspectiveEvidenceV3(
        decision_id=item.decision_id,
        acting_seat=item.acting_seat,
        observer_seat=item.observer_seat,
        truth_mode=item.truth_mode,
        projection_hash=item.projection_hash,
        forced=item.forced,
        truth_label=item.truth_label,
        confidence=item.confidence,
        evidence={**item.evidence, "outcomeImpact": 1},
        recommended_action=item.recommended_action,
        prompt_contract=item.prompt_contract,
    )
    failed = audit_rotated_perspective_evidence(original, broken, quarter_turns=1)
    assert not failed.passed and failed.maximum_delta == 89


def test_learning_store_builds_ledger_from_every_recorded_decision(tmp_path):
    store = LearningStore(tmp_path / "learning.sqlite3")
    # The ledger method's storage integration is exercised with its canonical
    # read methods so it cannot silently reintroduce a configured viewer.
    records = [_decision("D1", 1, Seat.NORTH), _decision("D2", 2, Seat.EAST)]
    class TrustedState:
        def to_dict(self, *, reveal_hands=True):
            return {"rulesHash": "r", "revealHands": reveal_hands}

    store.get_game = lambda _identifier: {"gameId": "GAME-1", "gameHash": "HASH-1"}  # type: ignore[method-assign]
    store.list_decisions = lambda _identifier, limit=1000: records  # type: ignore[method-assign]
    store.load_decision_payload = lambda _identifier: {"state": TrustedState()}  # type: ignore[method-assign]
    ledger = store.decision_ledger("GAME-1")
    assert [item.acting_seat for item in ledger.records] == [Seat.NORTH, Seat.EAST]
    assert ledger.game_id == "HASH-1"


def test_internal_four_projection_builder_keeps_each_view_isolated(rules, convention):
    episode = generate_episode(
        rules,
        convention,
        bundle_hash=rules.sha256,
        request=ScenarioRequest.governed(
            seed=3,
            contract=Contract.HOKUM,
            buyer=Seat.SOUTH,
            trump_suit=Suit.HEARTS,
            requested_samples=2,
            exact_allocation_limit=50,
            exact_search_max_remaining=2,
            max_seed_attempts=20,
        ),
    )

    class Store:
        play_domain = rules.play_domain

        def get_decision(self, _decision_id):
            return {"decisionId": "D1", "decisionKind": "play"}

        def load_decision_payload(self, _decision_id):
            return {"state": episode.state_before_pause}

    projections = build_all_observer_projections(Store(), "D1")
    assert set(projections) == set(Seat)
    assert len({item.sha256 for item in projections.values()}) == 4
    for seat, item in projections.items():
        assert item.information_state["observer"] == seat.name.lower()
        expected = [card.to_dict() for card in mask_to_cards(episode.state_before_pause.hands[int(seat)])]
        assert item.information_state["observerHand"] == expected
        assert "publicDeductions" in item.information_state
        assert "signals" in item.information_state

    decision = {
        "decisionId": "D1", "decisionKind": "play", "actor": "south",
        "legalActions": [{"code": "S7"}, {"code": "S8"}],
    }
    audit = {
        "forced": False,
        "bestAction": {"kind": "card", "code": "S7"},
        "expectedRegret": 12,
        "winProbabilityDrop": 0.3,
        "catastrophicCrossing": True,
        "predictedDifficulty": 88,
        "qualityScore": 45,
        "auditConfidence": 0.9,
        "debateScore": 10,
        "reasons": [{"code": "entry"}],
        "alternatives": [{"exactShare": 0.5}, {"exactShare": 0.5}],
    }
    evidence = build_decision_perspective_evidence(Store(), decision, audit)
    assert len(evidence) == 4
    actor_truth = next(item for item in evidence if item.truth_mode == TruthMode.ACTOR_OBSERVER)
    assert actor_truth.observer_seat == actor_truth.acting_seat == Seat.SOUTH
    assert actor_truth.recommended_action == {"kind": "card", "code": "S7"}
    assert actor_truth.truth_label == TruthLabel.ROBUST_BEST_KNOWN
    assert all(
        item.recommended_action is None
        for item in evidence
        if item.truth_mode == TruthMode.SEAT_EXPERIENCE
    )

    ledger_record = {
        **decision,
        "ordinal": 1,
        "phase": "playing",
        "publicState": {"observer": "south"},
        "selectedAction": {"code": "S8"},
    }

    class FoundryStore(Store):
        def decision_ledger(self, _identifier):
            return DecisionLedgerV1.build("GAME-1", (ledger_record,))

        def list_decisions(self, _identifier, limit=1000):
            return [ledger_record]

    foundry = MultiPerspectiveFoundryV1(FoundryStore(), critical_threshold=70)
    assert isinstance(foundry, PerspectiveAnalyzerPlugin)
    laffa = foundry.analyze_laffa("GAME-1", {"decisions": [audit | {"decisionId": "D1"}]})
    assert len(laffa.perspectives) == 4
    assert laffa.scenario_graph_sha256 is not None


def test_observer_experience_uses_only_isolated_deductions_and_qualified_signals():
    base = {
        "observer": "north",
        "observerHand": [{"id": index} for index in range(8)],
        "knownUnplayed": {
            "north": [{"id": index} for index in range(8)],
            "west": [],
            "south": [],
            "east": [],
        },
        "handSizes": [8, 8, 8, 8],
        "projects": [],
        "escalation": "normal",
        "signals": [],
    }
    exact = ObserverProjectionV1.from_public_mapping(
        "D1",
        Seat.NORTH,
        base | {
            "publicDeductions": [{
                "kind": "void_suit",
                "seat": "south",
                "suit": "hearts",
                "certainty": 1.0,
                "source": "trick_1_failed_to_follow",
            }],
        },
    )
    label, confidence, evidence, prompt = derive_seat_experience_evidence(
        exact,
        acting_seat=Seat.SOUTH,
    )
    assert label == TruthLabel.ROBUST_BEST_KNOWN
    assert confidence == 1.0
    assert prompt == "infer_distribution"
    assert evidence["inferenceAbility"] == 25.0

    signal = ObserverProjectionV1.from_public_mapping(
        "D2",
        Seat.NORTH,
        base | {
            "publicDeductions": [],
            "signals": [{
                "kind": "low_then_high",
                "seat": "south",
                "relationship": "partner",
                "confidence": 0.58,
            }],
        },
    )
    label, confidence, evidence, prompt = derive_seat_experience_evidence(
        signal,
        acting_seat=Seat.SOUTH,
    )
    assert label == TruthLabel.DEBATABLE
    assert confidence == 0.58
    assert prompt == "read_partner_signal"
    assert evidence["partnerSignalSignificance"] == pytest.approx(58.0)

    _, _, _, fallback = derive_seat_experience_evidence(
        signal,
        acting_seat=Seat.SOUTH,
        allowed_prompts=("post_reveal_analysis",),
    )
    assert fallback == "post_reveal_analysis"
    with pytest.raises(ValueError, match="unsupported"):
        derive_seat_experience_evidence(
            signal,
            acting_seat=Seat.SOUTH,
            allowed_prompts=("choose_acting_action",),
        )


def test_exact_optimal_requires_exhaustive_exact_supported_truth(rules, convention):
    episode = generate_episode(
        rules,
        convention,
        bundle_hash=rules.sha256,
        request=ScenarioRequest.governed(
            seed=7,
            contract=Contract.SUN,
            buyer=Seat.NORTH,
            requested_samples=2,
            exact_allocation_limit=50,
            exact_search_max_remaining=2,
            max_seed_attempts=20,
        ),
    )

    class Store:
        play_domain = rules.play_domain

        def get_decision(self, _decision_id):
            return {"decisionId": "D1", "decisionKind": "play"}

        def load_decision_payload(self, _decision_id):
            return {"state": episode.state_before_pause}

    decision = {
        "decisionId": "D1", "decisionKind": "play", "actor": "north",
        "legalActions": [{"code": "S7"}, {"code": "S8"}],
    }
    base = {
        "forced": False,
        "bestAction": {"kind": "card", "code": "S7"},
        "auditConfidence": 1.0,
        "hiddenDeals": 4,
        "effectiveSampleSize": 4,
        "alternatives": [
            {"exactShare": 1.0, "searchCompletionStatus": "exact"},
            {"exactShare": 1.0, "searchCompletionStatus": "exact"},
        ],
    }
    sampled = build_decision_perspective_evidence(
        Store(), decision, base | {"engineMethod": "constructive_uniform_hidden_deals+hard_facts_only"}
    )
    assert sampled[0].truth_label == TruthLabel.ROBUST_BEST_KNOWN
    exhaustive = build_decision_perspective_evidence(
        Store(), decision, base | {"engineMethod": "exhaustive_constrained_hidden_deals+hard_facts_only"}
    )
    assert exhaustive[0].truth_label == TruthLabel.EXACT_OPTIMAL
    heuristic = build_decision_perspective_evidence(
        Store(),
        decision,
        base
        | {
            "engineMethod": "exhaustive_constrained_hidden_deals+hard_facts_only",
            "alternatives": [
                {
                    "exactShare": 1.0,
                    "searchCompletionStatus": "heuristic_complete",
                },
                {
                    "exactShare": 1.0,
                    "searchCompletionStatus": "heuristic_complete",
                },
            ],
        },
    )
    assert heuristic[0].truth_label == TruthLabel.ROBUST_BEST_KNOWN

    with pytest.raises(ValueError, match="best action"):
        build_decision_perspective_evidence(Store(), decision, base | {"bestAction": None})


def test_tactical_detectors_require_causal_delayed_payoff():
    plugin = CausalTacticalDetectorV1()
    assert isinstance(plugin, TacticalDetectorPlugin)
    selected = CausalActionLineV1(
        "S7", 15, -2, 18, True, 0.65, 0.05, 3,
        ("promotion", "partner_enablement"), ("S7", "H8", "S8"),
        0.80, 10, 0.10,
        causal_source_hash="c" * 64,
    )
    alternative = CausalActionLineV1(
        "SA", 95, 4, 2, False, 0.20, 0.35, 0,
        (), ("SA",), 0.40, 0, 0.05,
    )
    evidence = plugin.detect(selected, (alternative,))
    kinds = {item.kind for item in evidence}
    assert TacticalKind.ONLY_WINNING_MOVE in kinds
    assert TacticalKind.SMALL_CARD_BIG_GAIN in kinds
    assert TacticalKind.KABOUT_CREATION in kinds
    assert TacticalKind.KABOUT_PREVENTION in kinds
    assert TacticalKind.ENTRY_UNBLOCKING in kinds
    assert TacticalKind.PARTNER_ENABLEMENT in kinds
    assert TacticalKind.CONTRACT_RESCUE in kinds
    assert TacticalKind.PROJECT_BIDDING in kinds

    no_cause = CausalActionLineV1(
        "S7", 15, -2, 18, True, 0.65, 0.05, 0, (), ("S7",),
    )
    false_positive_kinds = {item.kind for item in detect_tactical_evidence(no_cause, (alternative,))}
    assert TacticalKind.SMALL_CARD_BIG_GAIN not in false_positive_kinds
    assert TacticalKind.ENTRY_UNBLOCKING not in false_positive_kinds

    debate = CausalActionLineV1(
        "H8", 25, 1, 3, True, 0.1, 0.1, 1, ("entry",), ("H8",),
        hidden_deal_disagreement=0.4,
    )
    close = CausalActionLineV1(
        "H9", 40, 1, 2, True, 0.1, 0.1, 1, (), ("H9",),
        hidden_deal_disagreement=0.35,
    )
    assert TacticalKind.DEBATE in {item.kind for item in detect_tactical_evidence(debate, (close,))}


def test_game_audit_summary_cannot_hide_seat_or_team_journeys():
    def audit(decision_id: str, actor: Seat, regret: float, quality: float) -> DecisionAudit:
        return DecisionAudit(
            decision_id, "h" * 64, "G", None, int(decision_id[1:]), "play",
            actor.name.lower(), "playing", {"kind": "card"}, {"kind": "card"},
            1, "good", quality, False, regret, 0.1, False,
            (AuditedAlternative({"kind": "card"}, 1, 1, 0.5, 0.4, 0.6, 0, 0, 1, 0, 0, 0),),
            (), (), (), {}, "test", "1", "b", None, None, False, 4, 4, 50, 20, 0.8,
        )

    report = GameAuditReport.build(
        game_id="G", game_hash=None, game_serial=None, mode="test",
        bundle_hash="b", engine_version="1", config={},
        decisions=(audit("D1", Seat.NORTH, 1, 90), audit("D2", Seat.EAST, 8, 40)),
    )
    assert report.summary["bySeat"]["east"]["maximumExpectedRegret"] == 8
    assert report.summary["bySeat"]["south"]["status"] == "not_applicable"
    assert report.summary["byTeam"]["west_east"]["minimumActionQuality"] == 40
