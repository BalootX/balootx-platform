from __future__ import annotations

import copy
import json
import time
from dataclasses import replace

import pytest

from baloot_platform.challenge.constraint_deal_search import (
    DealConstraintSpecV1,
    OrToolsCpSatDealGeneratorV1,
)
from baloot_platform.challenge.evidence_v3 import canonical_sha256
from baloot_platform.challenge.guided_search import GuidedLegalSearchV1
from baloot_platform.challenge.replay import CompleteTraceReplayerV1
from baloot_platform.challenge.simulation import NaturalLaffaSimulatorV1
from baloot_platform.config import build_bundle
from baloot_platform.core.cards import Card
from baloot_platform.core.dealing import DealLayoutV1
from baloot_platform.core.enums import Contract, Rank, Seat, Suit
from baloot_platform.core.projects import project_card_patterns
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.structural_ir import StructuralDomainIRV1


def _spec(**overrides):
    value = {
        "schemaVersion": 1,
        "id": "cp_sat_test_v1",
        "revision": "1",
        "enabled": True,
        "backend": "ortools_cp_sat_v1",
        "assumedBuyer": "south",
        "purchaseSuits": ["spades"],
        "initialRequiredCards": {"south": ["SJ", "S9"]},
        "finalRequiredCards": {},
        "finalForbiddenCards": {},
        "initialSuitCountRanges": {
            "south": {"spades": {"minimum": 2, "maximum": 5}},
        },
        "finalSuitCountRanges": {
            "south": {"spades": {"minimum": 3, "maximum": 8}},
        },
        "projectPattern": None,
        "maximumSolutions": 1,
        "maximumConflicts": 1_000_000,
        "safetyTimeoutSeconds": 30.0,
        "solverWorkers": 1,
        "seedOffset": 0,
        "tags": ["engine_validation_required", "guided_structural"],
    }
    value.update(overrides)
    return DealConstraintSpecV1.build(value)


@pytest.mark.real_clock
def test_cp_sat_deadline_budget_preserves_default_and_clamps_remaining_time():
    spec = _spec(safetyTimeoutSeconds=30.0)
    generator = OrToolsCpSatDealGeneratorV1()

    assert generator._remaining_solver_seconds(spec, None) == 30.0
    deadline = time.monotonic() + 0.5
    remaining = generator._remaining_solver_seconds(spec, deadline)
    assert 0 < remaining <= 0.5
    with pytest.raises(InterruptedError, match="generation deadline exceeded"):
        generator._remaining_solver_seconds(spec, time.monotonic() - 1.0)


@pytest.mark.real_clock
def test_cp_sat_generation_rejects_an_expired_deadline_before_solver():
    with pytest.raises(InterruptedError, match="generation deadline exceeded"):
        OrToolsCpSatDealGeneratorV1().generate(
            _spec(),
            root_seed=1,
            dealer=Seat.EAST,
            rules=object(),
            deadline_monotonic=time.monotonic() - 1.0,
        )


def test_unconstrained_single_candidate_is_seeded_and_fixed_for_verification(root):
    rules = build_bundle(root).rules
    spec = _spec(
        assumedBuyer=None,
        purchaseSuits=[],
        initialRequiredCards={},
        finalRequiredCards={},
        finalForbiddenCards={},
        initialSuitCountRanges={},
        finalSuitCountRanges={},
        maximumSolutions=1,
        safetyTimeoutSeconds=5.0,
    )
    generator = OrToolsCpSatDealGeneratorV1()

    first = generator.generate(
        spec,
        root_seed=202608080019,
        dealer=Seat.EAST,
        rules=rules,
    )[0]
    repeated = generator.generate(
        spec,
        root_seed=202608080019,
        dealer=Seat.EAST,
        rules=rules,
    )[0]
    different_seed = generator.generate(
        spec,
        root_seed=202608080020,
        dealer=Seat.EAST,
        rules=rules,
    )[0]

    assert first == repeated
    assert first.deck_order != different_seed.deck_order
    assert first.solver_status == "optimal"
    assert sorted(first.deck_order) == list(range(32))
    continuation = generator.probe_continuation(
        spec,
        root_seed=202608080019,
        dealer=Seat.EAST,
        rules=rules,
        rejected_candidates=(first,),
    )
    assert continuation.solution_ordinal == 1
    assert not continuation.exhausted
    assert continuation.witness_deck_commitment != first.deck_commitment


@pytest.mark.real_clock
def test_cp_sat_deadline_is_checked_between_compact_solver_calls(
    root,
    monkeypatch: pytest.MonkeyPatch,
):
    bundle = build_bundle(root)
    spec = _spec(
        purchaseSuits=[],
        initialRequiredCards={},
        initialSuitCountRanges={},
        finalSuitCountRanges={},
        tacticalObjectiveWeights={"only_winning_move": 100},
        objectiveTargetSeat="west",
        tacticalContract="sun",
        maximumSolutions=1,
    )
    calls = 0

    def bounded_solver_seconds(active_spec, deadline_monotonic):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise InterruptedError("CP-SAT deal generation deadline exceeded")
        return active_spec.safety_timeout_seconds

    monkeypatch.setattr(
        OrToolsCpSatDealGeneratorV1,
        "_remaining_solver_seconds",
        staticmethod(bounded_solver_seconds),
    )

    with pytest.raises(InterruptedError, match="generation deadline exceeded"):
        OrToolsCpSatDealGeneratorV1().generate(
            spec,
            root_seed=20260809,
            dealer=Seat.EAST,
            rules=bundle.rules,
            deadline_monotonic=time.monotonic() + 120.0,
        )
    assert calls == 2


def test_structural_domain_snapshot_round_trips_through_closed_core_parser(root):
    structural = build_bundle(root).rules.structural_domain
    snapshot = structural.to_dict()

    assert StructuralDomainIRV1.from_dict(snapshot) == structural

    unknown = {**snapshot, "foreign": True}
    with pytest.raises(ValueError, match="unknown or missing"):
        StructuralDomainIRV1.from_dict(unknown)

    aliased = {**snapshot, "schemaVersion": True}
    aliased["sha256"] = canonical_sha256(
        {key: value for key, value in aliased.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="identity"):
        StructuralDomainIRV1.from_dict(aliased)


def test_legacy_constraint_benchmark_uses_rulebook_structural_authority(
    root, monkeypatch: pytest.MonkeyPatch
):
    from scripts.benchmark_constraint_deal_search import _benchmark_layout

    rules = build_bundle(root).rules
    expected = object()
    observed = {}

    def capture(
        cls,
        dealing: StructuralDomainIRV1,
        *,
        dealer: Seat,
        buyer: Seat | None = None,
    ):
        observed.update(dealing=dealing, dealer=dealer, buyer=buyer)
        return expected

    monkeypatch.setattr(DealLayoutV1, "build", classmethod(capture))

    assert _benchmark_layout(rules) is expected
    assert observed == {
        "dealing": rules.structural_domain,
        "dealer": Seat.EAST,
        "buyer": Seat.SOUTH,
    }
    monkeypatch.undo()
    layout = _benchmark_layout(rules)
    assert layout.dealer is Seat.EAST
    assert layout.buyer is Seat.SOUTH
    assert layout.purchase_position == (
        sum(rules.structural_domain.initial_packets)
        * rules.structural_domain.players
    )


@pytest.mark.parametrize("kind", [
    "only_winning_move",
    "small_card_big_gain",
    "large_card_sacrifice",
    "kabout_creation",
    "kabout_preservation",
    "kabout_prevention",
    "entry_unblocking",
    "trump_control",
    "contract_rescue",
    "partner_enablement",
    "project_bidding",
    "debate",
])
def test_every_governed_tactic_has_deterministic_structural_guidance(root, kind):
    rules = build_bundle(root).rules
    spec = _spec(
        tacticalObjectiveWeights={kind: 100},
        objectiveTargetSeat="south",
        tacticalContract="sun",
    )
    first = OrToolsCpSatDealGeneratorV1._tactical_coefficients(spec, rules=rules)
    second = OrToolsCpSatDealGeneratorV1._tactical_coefficients(spec, rules=rules)
    assert first
    assert first == second


def test_unknown_tactical_construction_objective_fails_closed(root):
    spec = _spec(
        tacticalObjectiveWeights={"unsupported_tactic": 100},
        objectiveTargetSeat="south",
        tacticalContract="sun",
    )
    with pytest.raises(ValueError, match="unsupported tactical construction"):
        OrToolsCpSatDealGeneratorV1._tactical_coefficients(
            spec,
            rules=build_bundle(root).rules,
        )


def test_contract_sensitive_tactical_construction_requires_exact_context():
    with pytest.raises(ValueError, match="requires an exact contract"):
        _spec(
            tacticalObjectiveWeights={"only_winning_move": 100},
            objectiveTargetSeat="south",
        )
    with pytest.raises(ValueError, match="requires an exact trump suit"):
        _spec(
            tacticalObjectiveWeights={"only_winning_move": 100},
            objectiveTargetSeat="south",
            tacticalContract="hokum",
        )
    with pytest.raises(ValueError, match="cannot declare trump"):
        _spec(
            tacticalObjectiveWeights={"only_winning_move": 100},
            objectiveTargetSeat="south",
            tacticalContract="sun",
            tacticalTrump="spades",
        )


def test_tactical_contract_context_is_frozen_in_profile_identity():
    sun = _spec(tacticalContract="sun")
    ashkal = _spec(tacticalContract="ashkal")
    assert sun.sha256 != ashkal.sha256
    sun_payload = sun.to_dict()
    ashkal_payload = ashkal.to_dict()
    sun_payload.pop("sha256")
    ashkal_payload.pop("sha256")
    assert DealConstraintSpecV1.build(sun_payload) == sun
    assert DealConstraintSpecV1.build(ashkal_payload) == ashkal


def test_sun_and_ashkal_use_the_same_governed_strength_bands(root):
    rules = build_bundle(root).rules
    sun = _spec(tacticalContract="sun")
    ashkal = _spec(tacticalContract="ashkal")
    assert OrToolsCpSatDealGeneratorV1._tactical_card_bands(
        sun,
        rules=rules,
    ) == OrToolsCpSatDealGeneratorV1._tactical_card_bands(
        ashkal,
        rules=rules,
    )
    high, low = OrToolsCpSatDealGeneratorV1._tactical_card_bands(
        sun,
        rules=rules,
    )
    assert {Card(card).rank for card in high} == {Rank.TEN, Rank.KING, Rank.ACE}
    assert {Card(card).rank for card in low} == {Rank.SEVEN, Rank.EIGHT, Rank.NINE}


@pytest.mark.parametrize("trump", list(Suit))
def test_hokum_strength_bands_follow_the_selected_trump_symmetrically(root, trump):
    rules = build_bundle(root).rules
    spec = _spec(
        tacticalContract="hokum",
        tacticalTrump=trump.name.lower(),
    )
    high_ids, low_ids = OrToolsCpSatDealGeneratorV1._tactical_card_bands(
        spec,
        rules=rules,
    )
    high = {Card(card_id) for card_id in high_ids}
    low = {Card(card_id) for card_id in low_ids}
    assert {
        card.rank for card in high if card.suit == trump
    } == {Rank.NINE, Rank.JACK, Rank.ACE}
    assert {
        card.rank for card in low if card.suit == trump
    } == {Rank.SEVEN, Rank.EIGHT, Rank.QUEEN}
    for suit in set(Suit) - {trump}:
        assert {
            card.rank for card in high if card.suit == suit
        } == {Rank.TEN, Rank.KING, Rank.ACE}
        assert {
            card.rank for card in low if card.suit == suit
        } == {Rank.SEVEN, Rank.EIGHT, Rank.NINE}


def test_tactical_bands_follow_a_governed_strength_table_change(root):
    baseline = build_bundle(root).rules
    changed_payload = copy.deepcopy(baseline.data)
    trump_strength = changed_payload["card_rules"]["hokum_trump_strength"]
    trump_strength[int(Rank.NINE)], trump_strength[int(Rank.QUEEN)] = (
        trump_strength[int(Rank.QUEEN)],
        trump_strength[int(Rank.NINE)],
    )
    changed = RuleBook(changed_payload)
    spec = _spec(tacticalContract="hokum", tacticalTrump="hearts")

    baseline_high, baseline_low = OrToolsCpSatDealGeneratorV1._tactical_card_bands(
        spec,
        rules=baseline,
    )
    changed_high, changed_low = OrToolsCpSatDealGeneratorV1._tactical_card_bands(
        spec,
        rules=changed,
    )
    heart_nine = Card.parse("H9").id
    heart_queen = Card.parse("HQ").id
    assert heart_nine in baseline_high and heart_queen in baseline_low
    assert heart_nine in changed_low and heart_queen in changed_high


@pytest.mark.parametrize("kind", ["only_winning_move", "small_card_big_gain"])
def test_tactical_deal_balance_prevents_dominant_and_empty_high_card_hands(
    root,
    kind,
):
    bundle = build_bundle(root)
    spec = _spec(
        purchaseSuits=[],
        initialRequiredCards={},
        initialSuitCountRanges={},
        finalSuitCountRanges={},
        tacticalObjectiveWeights={kind: 100},
        objectiveTargetSeat="west",
        tacticalContract="sun",
        tacticalDealBalance={
            "maximumTeamHighCardDifference": 2,
            "minimumHighCardsPerSeat": 2,
            "maximumHighCardsPerSeat": 4,
            "minimumTargetLowCards": 1,
        },
    )
    candidate = OrToolsCpSatDealGeneratorV1().generate(
        spec, root_seed=20260809, dealer=Seat.EAST, rules=bundle.rules
    )[0]
    layout = DealLayoutV1.build(
        StructuralDomainIRV1.from_rules(bundle.rules.data),
        dealer=Seat.EAST,
        buyer=Seat.SOUTH,
    )
    hands = {
        seat: tuple(
            Card(candidate.deck_order[position])
            for position in layout.positions_for(seat, final=True)
        )
        for seat in Seat
    }
    high_ids, low_ids = OrToolsCpSatDealGeneratorV1._tactical_card_bands(
        spec,
        rules=bundle.rules,
    )
    high_counts = {
        seat: sum(
            card.id in high_ids
            for card in cards
        )
        for seat, cards in hands.items()
    }
    assert all(2 <= value <= 4 for value in high_counts.values())
    assert abs(
        high_counts[Seat.WEST] + high_counts[Seat.EAST]
        - high_counts[Seat.NORTH] - high_counts[Seat.SOUTH]
    ) <= 2
    assert sum(
        card.id in low_ids
        for card in hands[Seat.WEST]
    ) >= 1


def test_contract_aware_balance_can_run_without_a_tactical_score_objective(root):
    bundle = build_bundle(root)
    spec = _spec(
        purchaseSuits=[],
        initialRequiredCards={},
        initialSuitCountRanges={},
        finalSuitCountRanges={},
        objectiveTargetSeat="west",
        tacticalContract="hokum",
        tacticalTrump="hearts",
        tacticalDealBalance={
            "maximumTeamHighCardDifference": 12,
            "minimumHighCardsPerSeat": 0,
            "maximumHighCardsPerSeat": 8,
            "minimumTargetLowCards": 1,
        },
    )
    candidate = OrToolsCpSatDealGeneratorV1().generate(
        spec,
        root_seed=20260810,
        dealer=Seat.EAST,
        rules=bundle.rules,
    )[0]
    assert candidate.structural_objective_score == 0
    assert candidate.solver_status == "optimal"


def test_deal_layout_is_the_single_complete_slot_map_for_every_rotation(root):
    dealing = StructuralDomainIRV1.from_rules(build_bundle(root).rules.data)
    for dealer in Seat:
        initial = DealLayoutV1.build(dealing, dealer=dealer)
        assert initial.purchase_position == 20
        assert sorted(
            position for values in initial.initial_positions for position in values
        ) == list(range(20))
        assert all(len(values) == 5 for values in initial.initial_positions)
        for buyer in Seat:
            complete = DealLayoutV1.build(dealing, dealer=dealer, buyer=buyer)
            assert complete.final_hand_positions is not None
            assert all(len(values) == 8 for values in complete.final_hand_positions)
            assert sorted(
                position
                for values in complete.final_hand_positions
                for position in values
            ) == list(range(32))


def test_cp_sat_deck_is_deterministic_structural_and_engine_replayable(root, deterministic_work):
    bundle = build_bundle(root)
    generator = OrToolsCpSatDealGeneratorV1()
    spec = _spec()
    first = generator.generate(
        spec, root_seed=20260806, dealer=Seat.EAST, rules=bundle.rules
    )
    second = generator.generate(
        spec, root_seed=20260806, dealer=Seat.EAST, rules=bundle.rules
    )
    bounded = generator.generate(
        spec,
        root_seed=20260806,
        dealer=Seat.EAST,
        rules=bundle.rules,
        deadline_monotonic=deterministic_work.read("monotonic") + 120.0,
    )
    assert first == second == bounded
    assert len(first) == 1
    candidate = first[0]
    assert sorted(candidate.deck_order) == list(range(32))
    layout = DealLayoutV1.build(
        StructuralDomainIRV1.from_rules(bundle.rules.data),
        dealer=Seat.EAST,
        buyer=Seat.SOUTH,
    )
    assert candidate.deck_order[layout.purchase_position] // 8 == 0
    assert {Card.parse("SJ").id, Card.parse("S9").id}.issubset(
        {
            candidate.deck_order[position]
            for position in layout.positions_for(Seat.SOUTH, final=False)
        }
    )
    initial_spades = sum(
        candidate.deck_order[position] // 8 == 0
        for position in layout.positions_for(Seat.SOUTH, final=False)
    )
    final_spades = sum(
        candidate.deck_order[position] // 8 == 0
        for position in layout.positions_for(Seat.SOUTH, final=True)
    )
    assert 2 <= initial_spades <= 5
    assert 3 <= final_spades <= 8

    simulator = NaturalLaffaSimulatorV1.from_root(
        root,
        profile_id="foundry_throughput",
        maximum_deal_attempts=1,
    )
    simulation = simulator.simulate_preordered_deck(
        20260806,
        candidate.deck_order,
        deal_generator_provenance=candidate.provenance_dict(),
    )
    trace = simulation.final_trace
    assert trace.seed_provenance["deckCommitment"] == candidate.deck_commitment
    assert candidate.provenance_dict()["pluginVersion"] == (
        OrToolsCpSatDealGeneratorV1.descriptor.version
    )
    assert trace.trusted_final_state["shuffle"]["algorithm"] == "governed_preordered_v1"
    assert (
        trace.trusted_final_state["generation"]["dealGenerator"]["candidateHash"]
        == candidate.candidate_hash
    )
    assert "deckOrder" not in json.dumps(trace.to_dict(trusted=False), sort_keys=True)
    for decision in trace.decision_ledger.records:
        assert dict(decision.chosen_action) in [
            dict(item) for item in decision.legal_actions
        ]
    canonical_replay = CompleteTraceReplayerV1(bundle.rules).replay(trace)
    assert len(canonical_replay.decisions) == len(trace.decision_ledger.records)

    tampered_state = copy.deepcopy(trace.trusted_final_state)
    tampered_state["generation"]["dealGenerator"]["deckCommitment"] = "0" * 64
    with pytest.raises(ValueError, match="generator commitment mismatch"):
        CompleteTraceReplayerV1(bundle.rules).replay(
            replace(trace, trusted_final_state=tampered_state)
        )

    missing_profile_state = copy.deepcopy(trace.trusted_final_state)
    missing_profile_state["generation"]["dealGenerator"]["profileHash"] = ""
    with pytest.raises(ValueError, match="generator profile identity"):
        CompleteTraceReplayerV1(bundle.rules).replay(
            replace(trace, trusted_final_state=missing_profile_state)
        )
    replay = simulator.resimulate_exact_deal(
        trace,
        match_score=(0, 0),
        round_number=1,
    )
    assert replay.seed_provenance["deckCommitment"] == candidate.deck_commitment


def test_cp_sat_project_proxy_comes_from_authoritative_rule_patterns(root):
    bundle = build_bundle(root)
    spec = _spec(
        purchaseSuits=[],
        initialSuitCountRanges={},
        finalSuitCountRanges={},
        projectPattern={
            "seat": "south",
            "contract": "sun",
            "kind": "siri",
            "value": 4,
            "highRank": "king",
        },
    )
    candidate = OrToolsCpSatDealGeneratorV1().generate(
        spec,
        root_seed=90210,
        dealer=Seat.EAST,
        rules=bundle.rules,
    )[0]
    layout = DealLayoutV1.build(
        StructuralDomainIRV1.from_rules(bundle.rules.data),
        dealer=Seat.EAST,
        buyer=Seat.SOUTH,
    )
    hand_mask = sum(
        1 << candidate.deck_order[position]
        for position in layout.positions_for(Seat.SOUTH, final=True)
    )
    patterns = project_card_patterns(
        kind="siri",
        value=4,
        high_rank=int(Rank.KING),
        contract=Contract.SUN,
        rules=bundle.rules,
    )
    assert patterns
    assert any(hand_mask & pattern == pattern for pattern in patterns)


def test_cp_sat_applies_required_and_forbidden_cards_to_final_hands(root):
    bundle = build_bundle(root)
    spec = _spec(
        purchaseSuits=[],
        initialRequiredCards={},
        initialSuitCountRanges={},
        finalSuitCountRanges={},
        finalRequiredCards={"south": ["SA"], "north": ["H7"]},
        finalForbiddenCards={"south": ["H7"], "north": ["SA"]},
    )
    candidate = OrToolsCpSatDealGeneratorV1().generate(
        spec, root_seed=1905, dealer=Seat.EAST, rules=bundle.rules
    )[0]
    layout = DealLayoutV1.build(
        StructuralDomainIRV1.from_rules(bundle.rules.data),
        dealer=Seat.EAST,
        buyer=Seat.SOUTH,
    )
    south = {
        candidate.deck_order[position]
        for position in layout.positions_for(Seat.SOUTH, final=True)
    }
    north = {
        candidate.deck_order[position]
        for position in layout.positions_for(Seat.NORTH, final=True)
    }
    assert Card.parse("SA").id in south
    assert Card.parse("H7").id not in south
    assert Card.parse("H7").id in north
    assert Card.parse("SA").id not in north


def test_cp_sat_no_good_cut_returns_distinct_stable_complete_assignments(root):
    bundle = build_bundle(root)
    spec = _spec(maximumSolutions=2)
    generator = OrToolsCpSatDealGeneratorV1()

    first = generator.generate(
        spec, root_seed=7719, dealer=Seat.EAST, rules=bundle.rules
    )
    second = generator.generate(
        spec, root_seed=7719, dealer=Seat.EAST, rules=bundle.rules
    )

    assert first == second
    assert len(first) == 2
    assert first[0].deck_order != first[1].deck_order
    assert first[0].provenance_dict()["precedingNoGoodCutCount"] == 0
    assert first[1].provenance_dict()["precedingNoGoodCutCount"] == 1
    assert first[1].provenance_dict()["noGoodCutType"] == "complete_deck_assignment_v1"


@pytest.mark.parametrize(
    ("compact", "expected_cut_type"),
    [
        (False, "complete_deck_assignment_v1"),
        (True, "final_hand_allocation_v1"),
    ],
    ids=("complete-deck", "compact-allocation"),
)
def test_cp_sat_generate_next_matches_batch_order_and_exact_cut_provenance(
    root,
    compact,
    expected_cut_type,
):
    bundle = build_bundle(root)
    overrides = {"maximumSolutions": 2}
    if compact:
        overrides.update({
            "purchaseSuits": [],
            "initialRequiredCards": {},
            "initialSuitCountRanges": {},
            "finalSuitCountRanges": {},
            "tacticalObjectiveWeights": {"only_winning_move": 100},
            "objectiveTargetSeat": "west",
            "tacticalContract": "sun",
        })
    spec = _spec(**overrides)
    generator = OrToolsCpSatDealGeneratorV1()
    kwargs = {
        "root_seed": 7719,
        "dealer": Seat.EAST,
        "rules": bundle.rules,
    }

    batch = generator.generate(spec, **kwargs)
    first = generator.generate_next(spec, **kwargs)
    assert first is not None
    second = generator.generate_next(
        spec,
        rejected_candidates=(first,),
        **kwargs,
    )
    rerun = generator.generate_next(
        spec,
        rejected_candidates=(first,),
        **kwargs,
    )

    assert (first, second) == batch
    assert second == rerun
    assert first.no_good_cut_type == expected_cut_type
    assert second is not None
    assert second.no_good_cut_type == expected_cut_type
    assert first.deck_order != second.deck_order
    assert (
        first.structural_objective_score
        == second.structural_objective_score
    )
    if compact:
        layout = DealLayoutV1.build(
            bundle.rules.structural_domain,
            dealer=Seat.EAST,
            buyer=Seat.SOUTH,
        )
        assert generator._hands_from_deck(
            first.deck_order,
            layout=layout,
        ) != generator._hands_from_deck(
            second.deck_order,
            layout=layout,
        )
    first_provenance = first.provenance_dict()
    second_provenance = second.provenance_dict()
    assert first_provenance["precedingNoGoodCutCount"] == 0
    assert first_provenance["precedingNoGoodCutHashes"] == []
    assert second_provenance["precedingNoGoodCutCount"] == 1
    assert len(second_provenance["precedingNoGoodCutHashes"]) == 1
    assert second_provenance["searchIdentityHash"] == (
        first_provenance["searchIdentityHash"]
    )
    assert generator.generate_next(
        spec,
        rejected_candidates=batch,
        **kwargs,
    ) is None


def test_cp_sat_generate_next_rejects_foreign_or_unordered_candidates(root):
    bundle = build_bundle(root)
    spec = _spec(maximumSolutions=2)
    generator = OrToolsCpSatDealGeneratorV1()
    batch = generator.generate(
        spec,
        root_seed=991,
        dealer=Seat.EAST,
        rules=bundle.rules,
    )

    with pytest.raises(ValueError, match="different governed search identity"):
        generator.generate_next(
            spec,
            root_seed=992,
            dealer=Seat.EAST,
            rules=bundle.rules,
            rejected_candidates=(batch[0],),
        )
    with pytest.raises(ValueError, match="exact ordered ordinal prefix"):
        generator.generate_next(
            spec,
            root_seed=991,
            dealer=Seat.EAST,
            rules=bundle.rules,
            rejected_candidates=(batch[1],),
        )
    reordered = replace(
        batch[1],
        preceding_no_good_cut_hashes=("0" * 64,),
        candidate_hash="",
    )
    reordered = replace(
        reordered,
        candidate_hash=canonical_sha256(reordered.integrity_core()),
    )
    with pytest.raises(ValueError, match="no-good provenance"):
        generator.generate_next(
            spec,
            root_seed=991,
            dealer=Seat.EAST,
            rules=bundle.rules,
            rejected_candidates=(
                batch[0],
                reordered,
            ),
        )
    wrong_cut = replace(
        batch[0],
        no_good_cut_type="final_hand_allocation_v1",
        candidate_hash="",
    )
    wrong_cut = replace(
        wrong_cut,
        candidate_hash=canonical_sha256(wrong_cut.integrity_core()),
    )
    with pytest.raises(ValueError, match="no-good cut"):
        generator.generate_next(
            spec,
            root_seed=991,
            dealer=Seat.EAST,
            rules=bundle.rules,
            rejected_candidates=(wrong_cut,),
        )
    with pytest.raises(ValueError, match="different governed profile"):
        generator.generate_next(
            _spec(maximumSolutions=2, seedOffset=1),
            root_seed=991,
            dealer=Seat.EAST,
            rules=bundle.rules,
            rejected_candidates=(batch[0],),
        )


@pytest.mark.real_clock
def test_cp_sat_generate_next_honors_deadline_with_rejection_prefix(root):
    bundle = build_bundle(root)
    spec = _spec(maximumSolutions=2)
    generator = OrToolsCpSatDealGeneratorV1()
    first = generator.generate_next(
        spec,
        root_seed=17,
        dealer=Seat.EAST,
        rules=bundle.rules,
    )
    assert first is not None

    with pytest.raises(InterruptedError, match="generation deadline exceeded"):
        generator.generate_next(
            spec,
            root_seed=17,
            dealer=Seat.EAST,
            rules=bundle.rules,
            rejected_candidates=(first,),
            deadline_monotonic=time.monotonic() - 1.0,
        )


@pytest.mark.real_clock
def test_cp_sat_continuation_receipt_is_deterministic_and_fail_closed(
    root,
    monkeypatch,
):
    bundle = build_bundle(root)
    spec = _spec(maximumSolutions=2)
    generator = OrToolsCpSatDealGeneratorV1()
    kwargs = {
        "root_seed": 1771,
        "dealer": Seat.EAST,
        "rules": bundle.rules,
    }
    batch = generator.generate(spec, **kwargs)

    initial = generator.probe_continuation(
        spec,
        rejected_candidates=(),
        **kwargs,
    )
    assert initial.solution_ordinal == 0
    assert initial.solver_status == "optimal"
    assert initial.witness_candidate_hash == batch[0].candidate_hash
    assert initial.preceding_no_good_cut_hashes == ()
    assert initial.integrity_core()["solverInvokedAfterFinalCut"] is False

    receipt = generator.probe_continuation(
        spec,
        rejected_candidates=(batch[0],),
        **kwargs,
    )
    rerun = generator.probe_continuation(
        spec,
        rejected_candidates=(batch[0],),
        **kwargs,
    )

    assert receipt == rerun
    assert receipt.solver_status == "optimal"
    assert receipt.solution_ordinal == 1
    assert receipt.witness_candidate_hash == batch[1].candidate_hash
    assert receipt.witness_deck_commitment == batch[1].deck_commitment
    assert receipt.witness_deck_commitment != batch[0].deck_commitment
    assert canonical_sha256(receipt.integrity_core()) == receipt.receipt_hash
    assert "deckOrder" not in json.dumps(receipt.to_dict(), sort_keys=True)
    assert (
        "initial_and_post_rejection_continuation_receipts"
        in generator.descriptor.capabilities
    )
    assert generator.descriptor.version == "1.5.0"

    with pytest.raises(InterruptedError, match="generation deadline exceeded"):
        generator.probe_continuation(
            spec,
            rejected_candidates=(batch[0],),
            deadline_monotonic=time.monotonic() - 1.0,
            **kwargs,
        )
    with pytest.raises(ValueError, match="different governed profile"):
        generator.probe_continuation(
            _spec(maximumSolutions=2, seedOffset=1),
            rejected_candidates=(batch[0],),
            **kwargs,
        )
    with pytest.raises(ValueError, match="integrity hash"):
        generator.probe_continuation(
            spec,
            rejected_candidates=(replace(
                batch[0],
                candidate_hash="0" * 64,
            ),),
            **kwargs,
        )

    from ortools.sat.python import cp_model

    class UnknownSolver:
        @staticmethod
        def solve(_model):
            return cp_model.UNKNOWN

    monkeypatch.setattr(
        generator,
        "_full_solver",
        lambda *_args, **_kwargs: UnknownSolver(),
    )
    with pytest.raises(RuntimeError, match="did not prove an optimal"):
        generator.probe_continuation(
            spec,
            rejected_candidates=(batch[0],),
            **kwargs,
        )


def test_cp_sat_tactical_structure_objective_is_deterministic_and_auditable(root):
    bundle = build_bundle(root)
    spec = _spec(
        purchaseSuits=[],
        initialRequiredCards={},
        initialSuitCountRanges={},
        finalSuitCountRanges={},
        tacticalObjectiveWeights={"kabout_creation": 100},
        objectiveTargetSeat="south",
        tacticalContract="sun",
    )
    generator = OrToolsCpSatDealGeneratorV1()

    first = generator.generate(
        spec, root_seed=501, dealer=Seat.EAST, rules=bundle.rules
    )[0]
    second = generator.generate(
        spec, root_seed=501, dealer=Seat.EAST, rules=bundle.rules
    )[0]

    assert first == second
    assert first.structural_objective_score > 0
    assert first.provenance_dict()["structuralObjectiveScore"] == first.structural_objective_score
    assert spec.to_dict()["tacticalObjectiveWeights"] == {"kabout_creation": 100}


def test_compact_project_400_allocation_replays_and_preserves_solver_hands(root):
    bundle = build_bundle(root)
    spec = _spec(
        purchaseSuits=[],
        initialRequiredCards={},
        initialSuitCountRanges={},
        finalSuitCountRanges={},
        projectPattern={
            "seat": "south",
            "contract": "sun",
            "kind": "four_hundred",
            "value": 40,
            "highRank": "ace",
        },
        tacticalObjectiveWeights={"kabout_creation": 100},
        objectiveTargetSeat="south",
        tacticalContract="sun",
    )
    candidate = OrToolsCpSatDealGeneratorV1().generate(
        spec, root_seed=10, dealer=Seat.WEST, rules=bundle.rules
    )[0]
    simulation = NaturalLaffaSimulatorV1.from_root(
        root,
        profile_id="foundry_balanced",
        maximum_deal_attempts=1,
    ).simulate_preordered_deck(
        10,
        candidate.deck_order,
        deal_generator_provenance=candidate.provenance_dict(),
        dealer=Seat.WEST,
    )

    layout = DealLayoutV1.build(
        StructuralDomainIRV1.from_rules(bundle.rules.data),
        dealer=Seat.WEST,
        buyer=Seat.SOUTH,
    )
    south_cards = {
        Card(candidate.deck_order[position]).code
        for position in layout.positions_for(Seat.SOUTH, final=True)
    }
    assert {"SA", "HA", "DA", "CA"}.issubset(south_cards)
    assert candidate.structural_objective_score > 0
    assert candidate.provenance_dict()["noGoodCutType"] == "final_hand_allocation_v1"
    assert simulation.publishable_laffa is True


def test_active_constraint_profile_fails_closed_without_plugin():
    simulator = object.__new__(NaturalLaffaSimulatorV1)
    simulator.agent_profile = type("Agent", (), {"id": "strong"})()
    profile = type(
        "Profile",
        (),
        {
            "agent_profile_id": "strong",
            "require_actual_deal_opportunity": False,
            "deal_constraint_spec": _spec(),
        },
    )()
    with pytest.raises(ValueError, match="require a generator plugin"):
        GuidedLegalSearchV1(simulator, profile)


def test_assumed_buyer_is_a_rejection_gate_and_never_a_forced_action():
    spec = _spec(assumedBuyer="south", finalSuitCountRanges={})
    simulation = type(
        "Simulation",
        (),
        {
            "publishable_laffa": True,
            "final_trace": type(
                "Trace", (), {"trusted_final_state": {"buyer": "east"}}
            )(),
        },
    )()
    assert not GuidedLegalSearchV1._constraint_buyer_matches(spec, simulation)


def test_constraint_profile_rejects_nondeterministic_workers_and_unknown_keys():
    with pytest.raises(ValueError, match="requires one worker"):
        _spec(solverWorkers=2)
    with pytest.raises(ValueError, match="unknown deal constraint"):
        _spec(invented=True)
