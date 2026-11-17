from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from ortools.sat.python import cp_model

from baloot_platform.core.cards import Card
from baloot_platform.core.dealing import DealLayoutV1
from baloot_platform.core.enums import Contract, Rank, Seat, Suit, Team
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.scoring_ir import ScoringDomainIRV1
from baloot_platform.research.ortools_play_tail import (
    NATIVE_OBSERVER_SAFE_PLAY_POLICY_IDS,
    NATIVE_PARTNER_FEED_POLICY_IDS,
    NATIVE_ESTABLISHED_CONTROL_POLICY_ID,
    NATIVE_SECURE_CONTROL_POLICY_ID,
    NATIVE_GUARD_PROTECTION_POLICY_ID,
    NATIVE_CLOSED_TRICK_FEED_POLICY_ID,
    OrToolsPlayTailStageV1,
    PlayTailSolveStopped,
    TailActionConstraintV1,
    TailPrefixActionV1,
    TailTrickCardCountConstraintV1,
    TailTrickWinnerConstraintV1,
    _observer_safe_play_score_v1,
    _established_control_bonus_v1,
    _guard_discard_cards_v1,
    _guard_discard_bonus_v1,
)
from baloot_platform.research.ortools_research_executor import _observer_safe_play_policy_artifact_v1
from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
from baloot_platform.research.scenario_contracts import ExactTracePrefixV1, ScenarioProofRequestV1
from baloot_platform.research.campaign_coordinator import (
    CampaignVariantV1,
    build_rules_preserving_diversity_document_v1,
    decision_state_card_hamming_distance_v1,
)
from baloot_platform.research.ortools_scenario_model import OrToolsScenarioModelV1
from baloot_platform.research.play_compiler import OrToolsCpSatPlayCompilerV1
from baloot_platform.research.scenario_contracts import ScenarioPredicateV1
from baloot_platform.research.scenario_predicate_manifest_v2 import (
    shared_scenario_predicate_catalog_lowering_manifest_v2,
)
from baloot_platform.research.reference_play import (
    ReferencePlayInterpreterV1,
    ReferencePlayV1,
)


def _fixed_ownership(hands: dict[Seat, tuple[str, ...]]) -> dict[str, Seat]:
    return {card: seat for seat, cards in hands.items() for card in cards}


def _solve(
    rules, hands, *, contract="sun", trump=None, leader="north", prefix=(),
    native_conversion=False, **kwargs
):
    cards = [card for seat in Seat for card in hands[seat]]
    return OrToolsPlayTailStageV1(native_conversion=native_conversion).solve(
        cards=cards,
        ownership=_fixed_ownership(hands),
        contract=contract,
        trump=trump,
        first_leader=leader,
        play_domain=rules.play_domain,
        structural_domain=rules.structural_domain,
        prefix=prefix,
        **kwargs,
    )


@pytest.mark.real_clock
def test_play_tail_uses_recorded_seed_without_multiworker_search(rules) -> None:
    hands = {
        Seat.NORTH: ("SA",),
        Seat.WEST: ("H7",),
        Seat.SOUTH: ("D7",),
        Seat.EAST: ("C7",),
    }
    solver = cp_model.CpSolver()
    telemetry = []
    first = _solve(rules, hands, random_seed=17, solver=solver, telemetry_callback=telemetry.append)
    second = _solve(rules, hands, random_seed=17)
    assert solver.parameters.random_seed == 17
    assert solver.parameters.num_search_workers == 1
    assert first.evidence_hash == second.evidence_hash
    assert telemetry[0]["solverStatus"] == "running"
    assert telemetry[-1]["solverStatus"] == "optimal"
    assert telemetry[-1]["modelBuildSeconds"] >= 0
    assert telemetry[-1]["modelConfigureSeconds"] >= 0
    assert telemetry[-1]["wallTimeSeconds"] >= 0


@pytest.mark.parametrize("contract,trump", [("sun", None), ("hokum", "hearts")])
def test_native_authored_field_domains_keep_legality_and_reference_parity(rules, contract, trump):
    hands = {Seat.NORTH: ("SA", "D7"), Seat.WEST: ("S10", "H7"),
             Seat.SOUTH: ("SK", "D8"), Seat.EAST: ("S7", "H8")}
    prefix = tuple(TailPrefixActionV1(seat, Card.parse(card)) for seat, card in
                   ((Seat.NORTH, "SA"), (Seat.WEST, "S10"),
                    (Seat.SOUTH, "SK"), (Seat.EAST, "S7")))
    kwargs = dict(cards=tuple(_fixed_ownership(hands)), ownership=_fixed_ownership(hands),
                  contract=contract, trump=trump, first_leader="north",
                  play_domain=rules.play_domain, structural_domain=rules.structural_domain,
                  prefix=prefix)
    native = OrToolsPlayTailStageV1(native_conversion=True).build_model(**kwargs)
    domains = {var.name: list(var.domain) for var in native.model.proto.variables}
    assert domains["card_1"] == [Card.parse("S10").id] * 2
    assert domains["actor_1"] == [int(Seat.WEST)] * 2
    assert domains["suit_1"] == [int(Suit.SPADES)] * 2
    result = native.solve(maximum_time_seconds=10)
    reference = OrToolsPlayTailStageV1().build_model(**kwargs).solve(maximum_time_seconds=10)
    assert result.plays == reference.plays
    assert result.legal_actions_by_step == reference.legal_actions_by_step
    assert result.trick_winners == reference.trick_winners
    for invalid in (TailPrefixActionV1(Seat.EAST, Card.parse("S10")),
                    TailPrefixActionV1(Seat.WEST, Card.parse("H7"))):
        with pytest.raises(PlayTailSolveStopped) as error:
            OrToolsPlayTailStageV1(native_conversion=True).build_model(
                **{**kwargs, "prefix": (prefix[0], invalid, *prefix[2:])}
            ).solve(maximum_time_seconds=10)
        assert error.value.solver_status == "infeasible"


def test_native_authored_point_domain_never_widens_existing_bound(rules):
    import copy

    raw = copy.deepcopy(rules.data)
    raw["card_rules"]["sun_points"] = [0] * 8
    raw["card_rules"]["sun_points"][int(Rank.ACE)] = 40
    domain = type(rules.play_domain).from_rules(raw, structural_domain=rules.structural_domain)
    cards = ("SA", "H7", "D7", "C7")
    prefix = tuple(TailPrefixActionV1(seat, Card.parse(card)) for seat, card in zip(Seat, cards))
    for native in (False, True):
        build = OrToolsPlayTailStageV1(native_conversion=native).build_model(
            cards=cards, ownership=dict(zip(cards, Seat)), contract="sun", trump=None,
            first_leader="north", play_domain=domain,
            structural_domain=rules.structural_domain, prefix=prefix)
        fields = {var.name: list(var.domain) for var in build.model.proto.variables}
        assert fields["points_0"] == [0, 32]
        with pytest.raises(PlayTailSolveStopped) as error:
            build.solve(maximum_time_seconds=10)
        assert error.value.solver_status == "infeasible"


@pytest.mark.parametrize("contract,trump", [(Contract.SUN, None), (Contract.HOKUM, Suit.HEARTS)])
def test_observer_safe_play_zero_policy_matches_actor_visible_choice(rules, contract, trump):
    cards = tuple(Card(card_id) for card_id in range(32))
    ownership = {card.code: Seat(card.id % 4) for card in cards}
    actor = Seat.NORTH
    hand = tuple(card for card in cards if ownership[card.code] == actor)
    request = ScenarioProofRequestV1.build({
        "schemaVersion": 1, "mode": "bounded", "visibility": "actor_information_set",
        "strategyClass": "pure_deterministic", "claim": "winning_root_action_count",
        "targetTeam": actor.team.name.lower(), "utility": "awarded_point_margin",
        "successThreshold": 0, "budgetProfile": "test",
        "horizon": {"unit": "tricks", "value": 1},
        "boundaryEvaluator": "current_raw_point_margin_v1",
    })
    normalized = SimpleNamespace(
        decision_kind="play", next_actor=actor.name.lower(),
        state={
            "contract": contract.value,
            "trumpSuit": trump.name.lower() if trump is not None else None,
            "currentTrick": {"index": 0, "leader": actor.name.lower(), "plays": []},
            "hands": {actor.name.lower(): [card.code for card in hand]},
        },
        legal_actions=tuple({"code": card.code} for card in hand),
    )
    expected, _ = _observer_safe_play_policy_artifact_v1(
        rules, normalized, request, actual_action=None,
        maximum_seconds=2, cancel_check=None,
    )
    build = OrToolsPlayTailStageV1().build_model(
        cards=cards, ownership=ownership, contract=contract, trump=trump,
        first_leader=actor, play_domain=rules.play_domain,
        structural_domain=rules.structural_domain, play_step_count=4,
        policy_prefix_steps=4,
    )
    result = build.solve(maximum_time_seconds=10)
    assert result.plays[0].card.code == expected
    assert not result.plays[0].ekka


def test_observer_safe_play_zero_policy_matches_native_exact_state(root, rules):
    fixture = json.loads((root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(encoding="utf-8"))
    exact = dict(fixture["exactTracePrefix"])
    exact["playActions"] = []
    exact["projectDeclarations"] = []
    exact["projectAssertions"] = []
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        ExactTracePrefixV1.build(exact), decision_point={"kind": "play"}, maximum_seconds=10,
    )
    actor = Seat.parse(normalized.next_actor)
    state = normalized.state
    ownership = {Card.parse(card): Seat.parse(seat) for seat, hand in state["hands"].items() for card in hand}
    request = ScenarioProofRequestV1.build({
        "schemaVersion": 1, "mode": "bounded", "visibility": "actor_information_set",
        "strategyClass": "pure_deterministic", "claim": "winning_root_action_count",
        "targetTeam": actor.team.name.lower(), "utility": "awarded_point_margin",
        "successThreshold": 0, "budgetProfile": "test",
        "horizon": {"unit": "tricks", "value": 1},
        "boundaryEvaluator": "current_raw_point_margin_v1",
    })
    expected, _ = _observer_safe_play_policy_artifact_v1(
        rules, normalized, request, actual_action=None,
        maximum_seconds=2, cancel_check=None,
    )
    build = OrToolsPlayTailStageV1(native_conversion=True).build_model(
        cards=tuple(Card(card_id) for card_id in range(32)),
        ownership=ownership, contract=state["contract"], trump=state.get("trumpSuit"),
        first_leader=actor, play_domain=rules.play_domain,
        structural_domain=rules.structural_domain, play_step_count=4,
        locked=bool(state.get("locked", False)), policy_prefix_steps=4,
    )
    assert build.solve(maximum_time_seconds=10).plays[0].card.code == expected


@pytest.mark.parametrize("policy_steps,reachable", [(12, True), (16, False)])
def test_golden_ace_tail_reachability_depends_on_opening_policy(
    root, rules, policy_steps, reachable
):
    fixture = json.loads((root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(encoding="utf-8"))
    exact = dict(fixture["exactTracePrefix"])
    plays = exact["playActions"]
    exact.update(playActions=[], projectDeclarations=[], projectAssertions=[], stateAssertions={})
    state = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        ExactTracePrefixV1.build(exact), decision_point={"kind": "play"}, maximum_seconds=10,
    ).state
    early_cards = {Card.parse(play["card"]).id for play in plays[:20]}
    tail_owners = {
        card.id: Seat.parse(seat)
        for seat, hand in state["hands"].items()
        for card in map(Card.parse, hand)
        if card.id not in early_cards
    }
    assert len(tail_owners) == 12
    build = OrToolsPlayTailStageV1().build_model(
        cards=tuple(Card(card_id) for card_id in range(32)),
        ownership={Card(card_id): seat for card_id, seat in tail_owners.items()},
        contract="sun", trump=None, first_leader=Seat.parse(plays[0]["actor"]),
        play_domain=rules.play_domain, structural_domain=rules.structural_domain,
        play_step_count=24, policy_prefix_steps=policy_steps,
        policy_id="observer_safe_tactical_category_v1",
        action_constraints=tuple(
            TailActionConstraintV1.build(step=20 + index, actor=play["actor"], cards=(play["card"],))
            for index, play in enumerate(plays[20:23])
        ),
    )
    for selected in build.selected_play_indexes[:20]:
        for card_id in tail_owners:
            build.model.add(selected != card_id)
    if reachable:
        assert len(build.solve(maximum_time_seconds=30).plays) == 24
    else:
        with pytest.raises(PlayTailSolveStopped) as error:
            build.solve(maximum_time_seconds=30)
        assert error.value.solver_status == "infeasible"


@pytest.mark.parametrize("contract,trump,locked,steps,policy_steps,seed,policy_id", [
    (Contract.SUN, None, False, 8, 8, None, "observer_safe_trick_control_v1"),
    (Contract.HOKUM, Suit.HEARTS, False, 8, 8, None, "observer_safe_trick_control_v1"),
    (Contract.HOKUM, Suit.HEARTS, True, 8, 8, None, "observer_safe_trick_control_v1"),
    (Contract.SUN, None, False, 24, 23, 43, "observer_safe_trick_control_v1"),
    (Contract.HOKUM, Suit.HEARTS, False, 24, 24, 51, "observer_safe_trick_control_v1"),
    (Contract.HOKUM, Suit.HEARTS, True, 24, 24, 51, "observer_safe_trick_control_v1"),
    (Contract.SUN, None, False, 8, 8, None, "observer_safe_tactical_category_v1"),
    (Contract.SUN, None, False, 8, 8, None, "observer_safe_partner_feed_v1"),
    (Contract.SUN, None, False, 8, 8, None, "observer_safe_partner_feed_compact_v1"),
    (Contract.HOKUM, Suit.HEARTS, False, 8, 8, None, "observer_safe_partner_feed_compact_v1"),
    (Contract.SUN, None, False, 8, 8, None, "observer_safe_partner_feed_established_control_v1"),
    (Contract.HOKUM, Suit.HEARTS, False, 8, 8, None, "observer_safe_partner_feed_established_control_v1"),
    (Contract.SUN, None, False, 24, 24, 43, NATIVE_SECURE_CONTROL_POLICY_ID),
    (Contract.HOKUM, Suit.HEARTS, False, 8, 8, None, NATIVE_SECURE_CONTROL_POLICY_ID),
    (Contract.SUN, None, False, 8, 8, None, NATIVE_GUARD_PROTECTION_POLICY_ID),
    (Contract.HOKUM, Suit.HEARTS, False, 8, 8, None, NATIVE_GUARD_PROTECTION_POLICY_ID),
    (Contract.SUN, None, False, 8, 8, None, NATIVE_CLOSED_TRICK_FEED_POLICY_ID),
    (Contract.HOKUM, Suit.HEARTS, False, 8, 8, None, NATIVE_CLOSED_TRICK_FEED_POLICY_ID),
])
def test_observer_safe_policy_matches_every_action_in_prefix(
    rules, contract, trump, locked, steps, policy_steps, seed, policy_id
):
    cards = tuple(Card(card_id) for card_id in range(32))
    shuffled = list(cards)
    if seed is not None:
        random.Random(seed).shuffle(shuffled)
    ownership = {card.code: Seat(index % 4) for index, card in enumerate(shuffled)}
    result = OrToolsPlayTailStageV1(native_conversion=True).solve(
        cards=cards, ownership=ownership, contract=contract, trump=trump,
        first_leader=Seat.NORTH, play_domain=rules.play_domain,
        structural_domain=rules.structural_domain, play_step_count=steps,
        policy_prefix_steps=policy_steps, policy_id=policy_id,
        locked=locked, maximum_time_seconds=30,
    )
    reference = ReferencePlayInterpreterV1(rules.play_domain, rules.structural_domain)
    remaining = {seat: {card for card in cards if ownership[card.code] == seat} for seat in Seat}
    current: list[ReferencePlayV1] = []
    for play in result.plays[:policy_steps]:
        legal = reference.legal_moves(
            tuple(remaining[play.actor]), current, actor=play.actor,
            contract=contract, trump=trump, locked=locked,
        )
        winner = (
            next(row for row in current if row.seat == reference.trick_winner(
                current, contract=contract, trump=trump,
            )) if current else None
        )

        def value(card: Card) -> int:
            strength = reference.strength(card, contract, trump)
            points = reference.points(card, contract, trump)
            if winner is None:
                score = _observer_safe_play_score_v1(strength, points)
                if policy_id == NATIVE_ESTABLISHED_CONTROL_POLICY_ID:
                    unknown = set().union(*remaining.values()) - remaining[play.actor]
                    if not any(
                            other.suit == card.suit and reference.strength(other, contract, trump) > strength
                            or contract == Contract.HOKUM and card.suit != trump and other.suit == trump
                            for other in unknown):
                        score += _established_control_bonus_v1(rules.play_domain, contract, trump)
            else:
                challenger = [*current, ReferencePlayV1(play.actor, card)]
                takes_control = reference.trick_winner(
                    challenger, contract=contract, trump=trump,
                ) == play.actor
                score = _observer_safe_play_score_v1(
                    strength, points,
                    partner_winning=winner.seat.team == play.actor.team,
                    takes_control=takes_control,
                    feed_partner=policy_id in NATIVE_PARTNER_FEED_POLICY_IDS and (
                        policy_id != NATIVE_CLOSED_TRICK_FEED_POLICY_ID or len(current) == 3),
                    secure_control=policy_id == NATIVE_SECURE_CONTROL_POLICY_ID and len(current) in (1, 2),
                )
                if (policy_id == NATIVE_GUARD_PROTECTION_POLICY_ID
                        and winner.seat.team != play.actor.team and card.suit != current[0].card.suit
                        and (contract.effective != Contract.HOKUM or card.suit != trump)
                        and card in _guard_discard_cards_v1(own_cards=remaining[play.actor],
                            public_played=set(cards) - set().union(*remaining.values()), pool=cards,
                            domain=rules.play_domain, contract=contract, trump=trump)):
                    score -= _guard_discard_bonus_v1(rules.play_domain, contract, trump)
            if policy_id == "observer_safe_tactical_category_v1":
                return 64 * ((score + 9_999) // 10_000) - card.id
            return score * 100 - card.id

        assert play.card == max(legal, key=value)
        assert not play.ekka
        remaining[play.actor].remove(play.card)
        current.append(ReferencePlayV1(play.actor, play.card, play.ekka))
        if len(current) == 4:
            current.clear()


@pytest.mark.parametrize("contract,trump", [
    (Contract.SUN, None), (Contract.HOKUM, Suit.HEARTS),
])
def test_policy_follow_table_control_matches_reference(rules, contract, trump):
    reference = ReferencePlayInterpreterV1(rules.play_domain, rules.structural_domain)
    for led in Suit:
        for incumbent in (Card(card_id) for card_id in range(32)):
            if incumbent.suit != led and incumbent.suit != trump:
                continue
            lead_card = next(
                Card(card_id) for card_id in range(32)
                if Card(card_id).suit == led and Card(card_id) != incumbent
            )
            prior = (
                (ReferencePlayV1(Seat.NORTH, incumbent),)
                if incumbent.suit == led else
                (ReferencePlayV1(Seat.NORTH, lead_card), ReferencePlayV1(Seat.EAST, incumbent))
            )
            incumbent_strength = reference.strength(incumbent, contract, trump)
            for card in (Card(card_id) for card_id in range(32)):
                if card == incumbent or (incumbent.suit != led and card == lead_card):
                    continue
                table_control = (
                    (card.suit == incumbent.suit and reference.strength(card, contract, trump) > incumbent_strength)
                    or (contract.effective == Contract.HOKUM and card.suit == trump and incumbent.suit != trump)
                )
                actual = reference.trick_winner(
                    (*prior, ReferencePlayV1(Seat.SOUTH, card)),
                    contract=contract, trump=trump,
                ) == Seat.SOUTH
                assert table_control == actual


@pytest.mark.parametrize(
    "policy_id",
    ("observer_safe_trick_control_v1", "observer_safe_tactical_category_v1"),
)
def test_policy_prefix_rejects_fixed_golden_opening_that_is_not_policy_play(
    root, rules, policy_id
):
    exact = json.loads((root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(encoding="utf-8"))["exactTracePrefix"]
    deck = tuple(Card(card_id) for card_id in exact["deckOrder"])
    layout = DealLayoutV1.build(rules.structural_domain, dealer=Seat.EAST, buyer=Seat.NORTH)
    ownership = {
        deck[position].code: seat
        for seat in Seat for position in layout.positions_for(seat, final=True)
    }
    prefix = tuple(
        TailPrefixActionV1.build(row["actor"], row["card"])
        for row in exact["playActions"][:1]
    )
    build = OrToolsPlayTailStageV1(native_conversion=True).build_model(
        cards=tuple(Card(card_id) for card_id in range(32)),
        ownership=ownership, contract="sun", trump=None, first_leader="north",
        play_domain=rules.play_domain, structural_domain=rules.structural_domain,
        prefix=prefix, play_step_count=4, policy_prefix_steps=1,
        policy_id=policy_id,
    )
    with pytest.raises(PlayTailSolveStopped) as stopped:
        build.solve(maximum_time_seconds=10)
    assert stopped.value.solver_status == "infeasible"


@pytest.mark.parametrize("policy_id", ["observer_safe_tactical_category_v1", NATIVE_ESTABLISHED_CONTROL_POLICY_ID])
def test_selective_tail_policy_uses_remaining_cards_and_breaks_category_ties(rules, policy_id):
    cards = tuple(Card(card_id) for card_id in range(32))
    ownership = {card: Seat(card.id % 4) for card in cards}

    def build(prefix=()):
        model = OrToolsPlayTailStageV1().build_model(
            cards=cards, ownership=ownership, contract="sun", trump=None,
            first_leader="north", play_domain=rules.play_domain,
            structural_domain=rules.structural_domain, selected_play_count=12,
            prefix=prefix, policy_prefix_steps=3,
            policy_id=policy_id,
        )
        for card, selected in zip(cards, model.selected_card_literals, strict=True):
            model.model.add(selected == int(4 <= card.id < 16))
        return model

    result = build().solve(maximum_time_seconds=10)
    assert result.plays[0].card == Card(4)
    if policy_id == NATIVE_ESTABLISHED_CONTROL_POLICY_ID:
        # Unselected SA/S10/SK are not public discards. Forcing SJ to be an
        # established winner must fail even though they are absent from the tail.
        changed = build().model
        index = next(index for index, var in enumerate(changed.proto.variables)
                     if var.name == "established_prefix_0_master_4")
        changed.add(changed.get_bool_var_from_proto_index(index) == 1)
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.max_time_in_seconds = 10
        assert solver.solve(changed) == cp_model.INFEASIBLE
    with pytest.raises(PlayTailSolveStopped) as stopped:
        build((TailPrefixActionV1.build("north", Card(12)),)).solve(
            maximum_time_seconds=10
        )
    assert stopped.value.solver_status == "infeasible"


@pytest.mark.parametrize("size", [4, 5, 6, 7])
@pytest.mark.parametrize("contract,trump", [(Contract.SUN, None), (Contract.HOKUM, Suit.HEARTS)])
def test_midround_hand_sizes_match_reference_legality(rules, size, contract, trump):
    hands = {seat: tuple(Card(int(seat) + 4 * index).code for index in range(size)) for seat in Seat}
    reference = ReferencePlayInterpreterV1(rules.play_domain, rules.structural_domain)
    prefix = ()
    for _ in range(2):
        result = _solve(rules, hands, contract=contract, trump=trump, prefix=prefix)
        actor = rules.structural_domain.offset(Seat.NORTH, len(prefix))
        expected = reference.legal_moves(
            tuple(Card.parse(card) for card in hands[actor]),
            tuple(ReferencePlayV1(item.actor, item.card, item.ekka) for item in prefix),
            actor=actor, contract=contract, trump=trump,
        )
        assert result.solver_status == "optimal"
        assert result.next_actor == actor
        assert {card.id for card in result.legal_actions} == {card.id for card in expected}
        prefix = (*prefix, TailPrefixActionV1(actor, result.legal_actions[0]))


def test_model_construction_honors_cancellation(rules) -> None:
    checks = 0

    def cancelled() -> bool:
        nonlocal checks
        checks += 1
        return checks > 1

    with pytest.raises(PlayTailSolveStopped, match="unknown"):
        OrToolsPlayTailStageV1().build_model(
            cards=tuple(Card(card_id) for card_id in range(32)),
            ownership=None,
            contract="sun",
            trump=None,
            first_leader="north",
            play_domain=rules.play_domain,
            structural_domain=rules.structural_domain,
            play_step_count=24,
            cancel_check=cancelled,
        )

    assert checks == 2


def test_pre_solve_cancellation_retains_zero_solver_counters(rules) -> None:
    cards = tuple(Card.parse(code) for code in ("S7", "H7", "D7", "C7"))
    build = OrToolsPlayTailStageV1().build_model(
        cards=cards,
        ownership={card: seat for card, seat in zip(cards, Seat, strict=True)},
        contract="sun",
        trump=None,
        first_leader="north",
        play_domain=rules.play_domain,
        structural_domain=rules.structural_domain,
    )
    with pytest.raises(PlayTailSolveStopped) as stopped:
        build.solve(cancel_check=lambda: True)
    assert stopped.value.solver_status == "unknown"
    assert stopped.value.partial_receipt["evidence"]["branches"] == 0
    assert stopped.value.partial_receipt["evidence"]["conflicts"] == 0


@pytest.mark.real_clock
def test_play_tail_deadline_includes_model_construction(monkeypatch, rules) -> None:
    from baloot_platform.research import ortools_play_tail

    clock = [0.0]
    monkeypatch.setattr(ortools_play_tail.time, "monotonic", lambda: clock[0])

    def build(_self, **kwargs):
        clock[0] = 2.0
        assert kwargs["cancel_check"]() is True
        raise PlayTailSolveStopped("unknown", evidence={"cancelled": True})

    monkeypatch.setattr(OrToolsPlayTailStageV1, "build_model", build)
    with pytest.raises(PlayTailSolveStopped) as stopped:
        OrToolsPlayTailStageV1().solve(
            cards=(), ownership=None, contract="sun", trump=None,
            first_leader="north", play_domain=rules.play_domain,
            structural_domain=rules.structural_domain,
            maximum_time_seconds=1.0,
        )
    evidence = stopped.value.partial_receipt["evidence"]
    assert evidence["budgetExceeded"] is True
    assert evidence["modelBuildSeconds"] >= 0


def _full_round_fixture(root: Path, rules):
    fixture = json.loads(
        (root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
            encoding="utf-8"
        )
    )
    exact = fixture["exactTracePrefix"]
    deck = tuple(Card(card_id) for card_id in exact["deckOrder"])
    layout = DealLayoutV1.build(
        rules.structural_domain, dealer=Seat.EAST, buyer=Seat.NORTH
    )
    hands = {
        seat: tuple(
            deck[position] for position in layout.positions_for(seat, final=True)
        )
        for seat in Seat
    }
    continuation_actors = (
        "west",
        "east",
        "north",
        "west",
        "south",
        "east",
        "north",
        "west",
        "south",
    )
    continuation_cards = (
        "CA",
        "C8",
        "C7",
        "D7",
        "S7",
        "C9",
        "DA",
        "D8",
        "SA",
    )
    actions = tuple(
        TailPrefixActionV1.build(
            action["actor"], action["card"], ekka=action.get("ekka", False)
        )
        for action in exact["playActions"]
    ) + tuple(
        TailPrefixActionV1.build(actor, card)
        for actor, card in zip(continuation_actors, continuation_cards, strict=True)
    )
    assert len(actions) == 32
    return deck, hands, actions


@pytest.mark.parametrize("native_conversion", (False, True))
def test_exhaustive_twelve_card_current_trick_matches_reference(rules, native_conversion) -> None:
    hands = {
        Seat.NORTH: ("S7", "HA", "C7"),
        Seat.WEST: ("S8", "H7", "C8"),
        Seat.SOUTH: ("S9", "HJ", "C9"),
        Seat.EAST: ("SA", "H8", "D7"),
    }
    parsed = {
        seat: tuple(Card.parse(card) for card in cards) for seat, cards in hands.items()
    }
    reference = ReferencePlayInterpreterV1(rules.play_domain, rules.structural_domain)
    visited = 0

    def walk(prefix: tuple[TailPrefixActionV1, ...]) -> None:
        nonlocal visited
        visited += 1
        result = _solve(
            rules,
            hands,
            contract=Contract.HOKUM,
            trump=Suit.HEARTS,
            prefix=prefix,
            native_conversion=native_conversion,
        )
        actor = rules.structural_domain.offset(Seat.NORTH, len(prefix))
        remaining = tuple(
            card
            for card in parsed[actor]
            if card.id not in {play.card.id for play in prefix}
        )
        reference_plays = tuple(
            ReferencePlayV1(play.actor, play.card, play.ekka) for play in prefix
        )
        expected = reference.legal_moves(
            remaining,
            reference_plays,
            actor=actor,
            contract=Contract.HOKUM,
            trump=Suit.HEARTS,
        )
        assert result.next_actor == actor
        assert {card.code for card in result.legal_actions} == {
            card.code for card in expected
        }
        assert result.current_trick == prefix
        if len(prefix) == 3:
            return
        for card in expected:
            walk((*prefix, TailPrefixActionV1(actor, card)))

    walk(())
    assert visited == 10


@pytest.mark.parametrize("contract,trump", [(Contract.SUN, None), (Contract.HOKUM, Suit.HEARTS)])
def test_native_play_history_carries_legality_across_tricks(rules, contract, trump):
    hands = {
        Seat.NORTH: ("S7", "HA", "C7"), Seat.WEST: ("S8", "H7", "C8"),
        Seat.SOUTH: ("S9", "HJ", "C9"), Seat.EAST: ("SA", "H8", "D7"),
    }
    path = _solve(rules, hands, contract=contract, trump=trump, native_conversion=True).plays
    for count in (0, 5, 9):
        prefix = tuple(TailPrefixActionV1(row.actor, row.card, row.ekka) for row in path[:count])
        build = OrToolsPlayTailStageV1(native_conversion=True).build_model(
            cards=tuple(_fixed_ownership(hands)), ownership=_fixed_ownership(hands),
            contract=contract, trump=trump, first_leader=Seat.NORTH,
            play_domain=rules.play_domain, structural_domain=rules.structural_domain, prefix=prefix)
        pool = sorted(Card.parse(card) for card in _fixed_ownership(hands))
        for step, action in enumerate(prefix):
            index = pool.index(action.card)
            assert list(build.selected_play_indexes[step].proto.domain) == [index, index]
        assert sum(variable.name.startswith("available_owner_")
                   for variable in build.model.proto.variables) <= 4 * len(path)
        assert not any(variable.name.startswith("available_follow_")
                       for variable in build.model.proto.variables)
        if contract == Contract.SUN:
            # Disabled Hokum obligations fold away; legality still matches the
            # independent reference for every recorded action below.
            assert not any(variable.name.startswith(("trump_ok_", "higher_ok_", "force_trump_"))
                           for variable in build.model.proto.variables)
        solver = cp_model.CpSolver()
        actual = build.solve(solver=solver, maximum_time_seconds=10)
        reference = replace(build._extract, native_ekka_flags_by_step=None)(
            solver, cp_model.OPTIMAL, build.model_hash)
        assert actual.legal_play_actions_by_step == reference.legal_play_actions_by_step
        assert actual.plays[:count] == path[:count]
        # A card already played by its owner cannot return on a later trick.
        repeated = build.model.clone()
        repeated.add(build._extract.card_ids[4] == build._extract.card_ids[0])
        assert solver.solve(repeated) == cp_model.INFEASIBLE


def test_complete_tail_winner_points_leader_and_hash_parity(rules) -> None:
    hands = {
        Seat.NORTH: ("S7", "S8", "SA"),
        Seat.WEST: ("H7", "H9", "HA"),
        Seat.SOUTH: ("C7", "C8", "CA"),
        Seat.EAST: ("D7", "D8", "DA"),
    }
    first = _solve(rules, hands)
    second = _solve(rules, hands)
    reference = ReferencePlayInterpreterV1(rules.play_domain, rules.structural_domain)
    assert first.to_dict() == second.to_dict()
    for trick in range(3):
        plays = tuple(
            ReferencePlayV1(play.actor, play.card)
            for play in first.plays[trick * 4 : trick * 4 + 4]
        )
        assert first.trick_winners[trick] == reference.trick_winner(
            plays, contract=Contract.SUN, trump=None
        )
        assert first.trick_raw_points[trick] == reference.trick_raw_points(
            plays, contract=Contract.SUN, trump=None
        )
        if trick < 2:
            assert first.plays[(trick + 1) * 4].actor == first.trick_winners[trick]
    assert len(first.model_hash) == len(first.evidence_hash) == len(first.sha256) == 64


def test_complete_tail_projects_any_decision_prefix_without_a_second_solve(
    rules,
) -> None:
    hands = {
        Seat.NORTH: ("S7", "S8", "SA"),
        Seat.WEST: ("H7", "H9", "HA"),
        Seat.SOUTH: ("C7", "C8", "CA"),
        Seat.EAST: ("D7", "D8", "DA"),
    }
    complete = _solve(rules, hands)

    for prefix_length in range(len(complete.plays) + 1):
        projected = complete.at_prefix(prefix_length)
        fixed = _solve(
            rules,
            hands,
            prefix=complete.plays[:prefix_length],
            tail=complete.plays[prefix_length:],
        )
        assert projected.plays == fixed.plays
        assert projected.remaining_hands == fixed.remaining_hands
        assert projected.current_trick == fixed.current_trick
        assert projected.next_actor == fixed.next_actor
        assert projected.legal_actions == fixed.legal_actions
        assert projected.legal_play_actions == fixed.legal_play_actions
        assert projected.at_prefix(prefix_length).to_dict() == projected.to_dict()

    with pytest.raises(ValueError, match="prefix_length"):
        complete.at_prefix(-1)


def test_hokum_obligations_and_partner_exception_match_reference(rules) -> None:
    partner_hands = {
        Seat.NORTH: ("SA", "SK", "S7"),
        Seat.WEST: ("S8", "C8", "D8"),
        Seat.SOUTH: ("HJ", "H7", "C7"),
        Seat.EAST: ("HA", "D7", "D9"),
    }
    overtrump_hands = {
        Seat.NORTH: ("S7", "SA", "SK"),
        Seat.WEST: ("H9", "C8", "D8"),
        Seat.SOUTH: ("HJ", "H7", "C7"),
        Seat.EAST: ("HA", "D7", "D9"),
    }
    led_trump_hands = {
        Seat.NORTH: ("H9", "S7", "S8"),
        Seat.WEST: ("HJ", "HA", "C8"),
        Seat.SOUTH: ("H7", "C7", "C9"),
        Seat.EAST: ("H8", "D7", "D8"),
    }
    fourth_hands = {
        Seat.NORTH: ("S7", "SA", "SK"),
        Seat.WEST: ("C8", "D8", "D9"),
        Seat.SOUTH: ("HJ", "C7", "C9"),
        Seat.EAST: ("HA", "H7", "D7"),
    }
    cases = (
        (partner_hands, (("north", "SA"), ("west", "S8"))),
        (partner_hands, (("north", "SK"), ("west", "S8"))),
        (overtrump_hands, (("north", "S7"), ("west", "H9"))),
        (led_trump_hands, (("north", "H9"),)),
        (fourth_hands, (("north", "S7"), ("west", "C8"), ("south", "HJ"))),
    )
    reference = ReferencePlayInterpreterV1(rules.play_domain, rules.structural_domain)
    for hands, prefix_codes in cases:
        prefix = tuple(
            TailPrefixActionV1.build(actor, card) for actor, card in prefix_codes
        )
        result = _solve(
            rules,
            hands,
            contract=Contract.HOKUM,
            trump=Suit.HEARTS,
            prefix=prefix,
        )
        actor = rules.structural_domain.offset(Seat.NORTH, len(prefix))
        remaining = tuple(
            Card.parse(card)
            for card in hands[actor]
            if card not in {play.card.code for play in prefix}
        )
        expected = reference.legal_moves(
            remaining,
            tuple(ReferencePlayV1(play.actor, play.card) for play in prefix),
            actor=actor,
            contract=Contract.HOKUM,
            trump=Suit.HEARTS,
        )
        assert {card.code for card in result.legal_actions} == {
            card.code for card in expected
        }


def test_locked_hokum_lead_uses_non_trump_when_available(rules) -> None:
    hands = {
        Seat.NORTH: ("H7", "S7", "C7"),
        Seat.WEST: ("H8", "S8", "C8"),
        Seat.SOUTH: ("H9", "S9", "C9"),
        Seat.EAST: ("HA", "SA", "CA"),
    }
    result = _solve(
        rules,
        hands,
        contract=Contract.HOKUM,
        trump=Suit.HEARTS,
        locked=True,
    )
    assert {card.code for card in result.legal_actions} == {"S7", "C7"}


def test_ekka_is_a_distinct_legal_lead_action(rules) -> None:
    highest = {
        Seat.NORTH: ("SA",),
        Seat.WEST: ("H7",),
        Seat.SOUTH: ("D7",),
        Seat.EAST: ("C7",),
    }
    result = _solve(
        rules,
        highest,
        contract=Contract.HOKUM,
        trump=Suit.HEARTS,
    )
    assert [
        (action.card.code, action.ekka) for action in result.legal_play_actions
    ] == [
        ("SA", False),
        ("SA", True),
    ]
    fixed = _solve(
        rules,
        highest,
        contract=Contract.HOKUM,
        trump=Suit.HEARTS,
        prefix=(TailPrefixActionV1.build("north", "SA", ekka=True),),
        ekka_leads=(True,),
    )
    assert fixed.plays[0].ekka is True


@pytest.mark.parametrize("play_steps", [4, 8])
@pytest.mark.parametrize("contract,trump", [(Contract.SUN, None), (Contract.HOKUM, Suit.HEARTS)])
@pytest.mark.parametrize("leader", list(Seat))
def test_native_action_extraction_reads_solver_flags(rules, play_steps, contract, trump, leader):
    from ortools.sat.python import cp_model
    from baloot_platform.core.models import LegalPlayActionV1

    hands = {
        Seat.NORTH: ("SK", "C7"), Seat.WEST: ("SA", "S7"),
        Seat.SOUTH: ("D7", "D8"), Seat.EAST: ("H7", "H8"),
    }
    build = OrToolsPlayTailStageV1(native_conversion=True).build_model(
        cards=tuple(_fixed_ownership(hands)), ownership=_fixed_ownership(hands),
        contract=contract, trump=trump, first_leader=leader,
        play_domain=rules.play_domain, structural_domain=rules.structural_domain,
        play_step_count=play_steps,
    )
    solver = cp_model.CpSolver()
    if contract == Contract.SUN:
        assert len({flag.index for flag in build._extract.ekka_leads}) == 1
        assert list(build._extract.ekka_leads[0].proto.domain) == [0, 0]
        flags = build._extract.native_ekka_flags_by_step
        assert flags is not None
        assert len({id(flag) for row in flags for flag in row}) == 1
    result = build.solve(solver=solver)
    reference = replace(build._extract, native_ekka_flags_by_step=None)(
        solver, cp_model.OPTIMAL, build.model_hash,
    )
    native = replace(build._extract, domain=SimpleNamespace())(
        solver, cp_model.OPTIMAL, build.model_hash,
    )
    assert native.evidence_hash == reference.evidence_hash == result.evidence_hash
    assert native.legal_play_actions_by_step == reference.legal_play_actions_by_step
    before = str(build.model.proto)
    for step, actions in enumerate(reference.legal_play_actions_by_step):
        assert {code for code, literal in build.native_legal_action_literals(step).items()
                if solver.boolean_value(literal)} == {
                    LegalPlayActionV1(action.card, action.ekka).code for action in actions}
    for step in (True, -1, play_steps, "0"):
        with pytest.raises(ValueError, match="outside the compiled horizon"):
            build.native_legal_action_literals(step)
    legacy = replace(build, _extract=replace(build._extract, native_ekka_flags_by_step=None))
    with pytest.raises(ValueError, match="requires native action variables"):
        legacy.native_legal_action_literals(0)
    assert str(build.model.proto) == before
    if leader == Seat.NORTH:
        assert {(action.card.code, action.ekka) for action in native.legal_play_actions_by_step[0]} == (
            {("SK", False), ("C7", False), ("C7", True)}
            if contract == Contract.HOKUM else {("SK", False), ("C7", False)}
        )


@pytest.mark.parametrize("contract,trump", [(Contract.SUN, None), *[(Contract.HOKUM, suit) for suit in Suit]])
@pytest.mark.parametrize("policy_id,secure_control", [
    *[(policy, False) for policy in sorted(NATIVE_OBSERVER_SAFE_PLAY_POLICY_IDS)],
    (NATIVE_SECURE_CONTROL_POLICY_ID, True),
    (NATIVE_CLOSED_TRICK_FEED_POLICY_ID, True),
])
def test_compact_visible_score_lowering_equals_frozen_table_every_context(rules, contract, trump, policy_id, secure_control):
    from baloot_platform.research.ortools_play_tail import _compact_observed_card_score_v1

    model = cp_model.CpModel()
    suit = model.new_int_var(0, 3, "incumbent_suit")
    strength = model.new_int_var(0, 7, "incumbent_strength")
    partner = model.new_bool_var("partner_winning")
    comparisons, differs = {}, []
    for card in map(Card, range(32)):
        card_strength = rules.play_domain.strength(card, contract, trump)
        points = rules.play_domain.points(card, contract, trump)
        def priority(has_partner, takes_control):
            score = _observer_safe_play_score_v1(card_strength, points,
                partner_winning=has_partner, takes_control=takes_control,
                feed_partner=policy_id in NATIVE_PARTNER_FEED_POLICY_IDS and (
                    policy_id != NATIVE_CLOSED_TRICK_FEED_POLICY_ID or secure_control),
                secure_control=policy_id == NATIVE_SECURE_CONTROL_POLICY_ID and secure_control)
            return (score + 9_999) // 10_000 if policy_id == "observer_safe_tactical_category_v1" else score
        compact, _ = _compact_observed_card_score_v1(model, (suit, strength, partner), card,
            card_strength, points, contract, trump, "compact", comparisons, priority=priority)
        rows = []
        for incumbent_suit in range(4):
            for incumbent_strength in range(8):
                control = (int(card.suit) == incumbent_suit and card_strength > incumbent_strength or
                    contract == Contract.HOKUM and card.suit == trump and incumbent_suit != int(trump))
                for has_partner in (0, 1):
                    rows.append((incumbent_suit, incumbent_strength, has_partner,
                        priority(bool(has_partner), control)))
        legacy = model.new_int_var(min(row[-1] for row in rows), max(row[-1] for row in rows), f"legacy_{card.id}")
        model.add_allowed_assignments([suit, strength, partner, legacy], rows)
        mismatch = model.new_bool_var(f"differs_{card.id}")
        model.add(compact != legacy).only_enforce_if(mismatch)
        differs.append(mismatch)
    model.add_bool_or(differs)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 5
    assert solver.solve(model) == cp_model.INFEASIBLE


@pytest.mark.parametrize("policy_id", sorted(NATIVE_OBSERVER_SAFE_PLAY_POLICY_IDS))
@pytest.mark.parametrize("contract,trump", [(Contract.SUN, None), (Contract.HOKUM, Suit.HEARTS)])
@pytest.mark.parametrize("team", list(Team))
@pytest.mark.parametrize("full_deck", [False, True])
def test_native_visible_policy_matches_public_trick_reference(rules, policy_id, contract, trump, team, full_deck):
    from baloot_platform.core.models import LegalPlayActionV1
    hands = {Seat.NORTH: ("SK", "C7"), Seat.WEST: ("SA", "S7"),
             Seat.SOUTH: ("S10", "D8"), Seat.EAST: ("H7", "H8")}
    if full_deck:
        cards = list(range(32))
        random.Random(42).shuffle(cards)
        hands = {seat: tuple(Card(card).code for card in cards[int(seat)*8:int(seat)*8+8]) for seat in Seat}
    build = OrToolsPlayTailStageV1(native_conversion=True).build_model(
        cards=tuple(_fixed_ownership(hands)), ownership=_fixed_ownership(hands),
        contract=contract, trump=trump, first_leader=Seat.NORTH,
        play_domain=rules.play_domain, structural_domain=rules.structural_domain)
    original = str(build.model.proto)
    for invalid in (True, -1, sum(map(len, hands.values()))):
        with pytest.raises(ValueError):
            build.constrain_native_observer_safe_policy(team=team, policy_id=policy_id, start_step=invalid)
    assert original == str(build.model.proto)
    build.constrain_native_observer_safe_policy(team=team, policy_id=policy_id)
    # Functional full-round parity uses the existing full-prefix test ceiling;
    # this is not a production latency qualification.
    result = build.solve(maximum_time_seconds=30 if full_deck else 10)
    reference = ReferencePlayInterpreterV1.from_rules(rules.data)
    for step, play in enumerate(result.plays):
        if play.actor.team != team:
            continue
        trick = [ReferencePlayV1(row.actor, row.card) for row in result.plays[step // 4 * 4:step]]
        def rank(action):
            strength = reference.strength(action.card, contract, trump)
            points = reference.points(action.card, contract, trump)
            kwargs = {}
            if trick:
                incumbent = reference.trick_winner(trick, contract=contract, trump=trump)
                after = reference.trick_winner([*trick, ReferencePlayV1(play.actor, action.card)], contract=contract, trump=trump)
                kwargs = dict(partner_winning=incumbent.team == play.actor.team, takes_control=after == play.actor)
            score = _observer_safe_play_score_v1(strength, points,
                feed_partner=policy_id in NATIVE_PARTNER_FEED_POLICY_IDS and (
                    policy_id != NATIVE_CLOSED_TRICK_FEED_POLICY_ID or len(trick) == 3),
                secure_control=policy_id == NATIVE_SECURE_CONTROL_POLICY_ID and len(trick) in (1, 2), **kwargs)
            if not trick and policy_id == NATIVE_ESTABLISHED_CONTROL_POLICY_ID:
                known = {Card.parse(card) for card in hands[play.actor]}
                known.update(row.card for row in result.plays[:step])
                unknown = {Card.parse(card) for hand in hands.values() for card in hand} - known
                if not any(
                        other.suit == action.card.suit and reference.strength(other, contract, trump) > strength
                        or contract == Contract.HOKUM and action.card.suit != trump and other.suit == trump
                        for other in unknown):
                    score += _established_control_bonus_v1(rules.play_domain, contract, trump)
            if (trick and policy_id == NATIVE_GUARD_PROTECTION_POLICY_ID
                    and incumbent.team != play.actor.team and action.card.suit != trick[0].card.suit
                    and (contract.effective != Contract.HOKUM or action.card.suit != trump)):
                public = {row.card for row in result.plays[:step]}
                own = {Card.parse(card) for card in hands[play.actor]} - public
                if action.card in _guard_discard_cards_v1(own_cards=own, public_played=public,
                        pool=tuple(map(Card.parse, _fixed_ownership(hands))), domain=rules.play_domain,
                        contract=contract, trump=trump):
                    score -= _guard_discard_bonus_v1(rules.play_domain, contract, trump)
            if policy_id == "observer_safe_tactical_category_v1":
                score = (score + 9_999) // 10_000
            return score, -LegalPlayActionV1(action.card, action.ekka).action_id
        expected = max(result.legal_play_actions_by_step[step], key=rank)
        assert (play.card, play.ekka) == (expected.card, expected.ekka)
    if policy_id == "observer_safe_partner_feed_compact_v1" and team == Seat.NORTH.team:
        # The direct lead chain must exclude every non-best root, not merely
        # happen to generate one best witness. Both card and Ekka are checked.
        changed = build.model.clone()
        different = changed.new_bool_var("different_lead")
        changed.add(build._extract.card_ids[0] != result.plays[0].card.id).only_enforce_if(different)
        changed.add(build._extract.card_ids[0] == result.plays[0].card.id).only_enforce_if(different.Not())
        changed.add_bool_or([different, build._extract.ekka_leads[0]])
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.max_time_in_seconds = 10
        assert solver.solve(changed) == cp_model.INFEASIBLE


@pytest.mark.parametrize("contract,trump", [("sun", None), ("hokum", "clubs")])
@pytest.mark.parametrize("position,expected", [(1, "HA"), (3, "HK")])
def test_secure_control_uses_ace_before_later_opponents_not_on_last_play(rules, contract, trump, position, expected):
    actor = Seat.WEST if position == 1 else Seat.EAST
    hands = {Seat.NORTH: ("H9", "D7"), Seat.SOUTH: ("H10", "H7"),
        actor: ("HA", "HK"), actor.partner: ("HQ", "H8")}
    prefix = [TailPrefixActionV1.build("north", "H9")]
    if position == 3:
        prefix += [TailPrefixActionV1.build("west", "HQ"), TailPrefixActionV1.build("south", "H7")]
    policy_id = "observer_safe_partner_feed_secure_control_v1"
    build = OrToolsPlayTailStageV1(native_conversion=True).build_model(
        cards=tuple(_fixed_ownership(hands)), ownership=_fixed_ownership(hands),
        contract=contract, trump=trump, first_leader="north", prefix=prefix,
        play_domain=rules.play_domain, structural_domain=rules.structural_domain)
    build.constrain_native_observer_safe_policy(team=actor.team, policy_id=policy_id, start_step=position)
    result = build.solve(maximum_time_seconds=10)
    assert result.plays[position].card.code == expected
    # Exclude the unwanted move across every continuation, not just one witness.
    changed = build.model.clone()
    changed.add(build._extract.card_ids[position] != Card.parse(expected).id)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 5
    assert solver.solve(changed) == cp_model.INFEASIBLE
    request = ScenarioProofRequestV1.build({"schemaVersion": 1, "mode": "bounded",
        "visibility": "actor_information_set", "strategyClass": "pure_deterministic",
        "claim": "winning_root_action_count", "targetTeam": actor.team.name.lower(),
        "utility": "awarded_point_margin", "successThreshold": 0, "budgetProfile": "test",
        "horizon": {"unit": "tricks", "value": 1},
        "boundaryEvaluator": "current_raw_point_margin_v1"})
    normalized = SimpleNamespace(decision_kind="play", next_actor=actor.name.lower(),
        state={"contract": contract, "trumpSuit": trump, "completedTricks": [],
            "currentTrick": {"index": 0, "leader": "north", "plays": [row.to_dict() for row in prefix]},
            "hands": {seat.name.lower(): list(hand) for seat, hand in hands.items()}},
        legal_actions=({"code": "HA"}, {"code": "HK"}))
    selected, artifact = _observer_safe_play_policy_artifact_v1(rules, normalized, request,
        actual_action=None, maximum_seconds=2, cancel_check=None, policy_id=policy_id)
    assert selected == expected
    normalized.state["hands"]["north"], normalized.state["hands"]["south"] = (
        normalized.state["hands"]["south"], normalized.state["hands"]["north"])
    assert _observer_safe_play_policy_artifact_v1(rules, normalized, request,
        actual_action=None, maximum_seconds=2, cancel_check=None, policy_id=policy_id) == (selected, artifact)
    assert artifact["proofModel"]["actionQuality"]["scope"] == "observer_safe_policy_conditioned"


@pytest.mark.parametrize("policy_id", NATIVE_OBSERVER_SAFE_PLAY_POLICY_IDS)
def test_native_policy_guard_replaces_only_the_matching_step(rules, policy_id):
    hands = {Seat.NORTH: ("H9", "D8"), Seat.WEST: ("HA", "HK"),
             Seat.SOUTH: ("H7", "S7"), Seat.EAST: ("HQ", "C7")}
    build = OrToolsPlayTailStageV1(native_conversion=True).build_model(
        cards=tuple(_fixed_ownership(hands)), ownership=_fixed_ownership(hands),
        contract="sun", trump=None, first_leader="north",
        prefix=[TailPrefixActionV1.build("north", "H9")],
        play_domain=rules.play_domain, structural_domain=rules.structural_domain)
    baseline = replace(build, model=build.model.clone())
    baseline.constrain_native_observer_safe_policy(team=Team.WEST_EAST, policy_id=policy_id, start_step=1)
    expected = baseline.solve(maximum_time_seconds=5).plays[1].card
    replacement = Card.parse("HK" if expected.code == "HA" else "HA")
    guard = build.model.new_bool_var("owned_information_matches")
    build.model.add(build._extract.card_ids[1] == replacement.id).only_enforce_if(guard)
    build.constrain_native_observer_safe_policy(team=Team.WEST_EAST, policy_id=policy_id,
        start_step=1, _override_matches={1: [guard]})
    for matching, allowed in ((False, expected), (True, replacement)):
        owned = replace(build, model=build.model.clone())
        owned.model.add(guard == int(matching))
        result = owned.solve(maximum_time_seconds=5)
        assert result.plays[1].card == allowed
        # No unconstrained alternative survives either side of the guard.
        owned.model.add(build._extract.card_ids[1] != allowed.id)
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.max_time_in_seconds = 5
        assert solver.solve(owned.model) == cp_model.INFEASIBLE


@pytest.mark.parametrize("own,public,led,partner,contract,trump,expected", [
    (("C7", "C10", "HK"), (), "diamonds", False, "sun", None, {"C7"}),
    (("C7", "C10", "HK"), ("CA",), "diamonds", False, "sun", None, set()),
    (("C7", "C9", "C10", "HK"), (), "diamonds", False, "sun", None, set()),
    (("C7", "C10", "HK"), (), "clubs", False, "sun", None, set()),
    (("C7", "C10", "HK"), (), "diamonds", True, "sun", None, set()),
    (("C7", "C9", "HK"), (), "diamonds", False, "sun", None, set()),
    (("C7", "C10", "HK"), (), "diamonds", False, "hokum", "clubs", set()),
    (("C7", "C10", "HK"), (), "diamonds", False, "hokum", "hearts", {"C7"}),
])
def test_native_guard_discard_is_owned_public_and_contract_safe(
        rules, own, public, led, partner, contract, trump, expected):
    from baloot_platform.research.ortools_play_tail import _native_guard_discard_flags_v1
    contract = Contract.parse(contract)
    trump = Suit.parse(trump) if trump else None
    own_cards, public_cards = set(map(Card.parse, own)), set(map(Card.parse, public))
    pool = tuple(map(Card, range(32)))
    for hidden_owner in (Seat.NORTH, Seat.SOUTH):
        model = cp_model.CpModel()
        owners = [model.new_constant(int(Seat.EAST if card in own_cards else hidden_owner)) for card in pool]
        flags = _native_guard_discard_flags_v1(model, pool=pool, owners=owners,
            actor=model.new_constant(int(Seat.EAST)),
            played=[model.new_constant(int(card in public_cards)) for card in pool],
            led_suit=model.new_constant(int(Suit.parse(led))), partner_winning=model.new_constant(int(partner)),
            domain=rules.play_domain, contract=contract, trump=trump, name="guard")
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.max_time_in_seconds = 2
        assert solver.solve(model) == cp_model.OPTIMAL
        assert {card.code for card, flag in zip(pool, flags, strict=True) if solver.boolean_value(flag)} == expected


@pytest.mark.parametrize("prefix_policy,position,expected", [(False, 2, "S9"), (True, 2, "S9"), (False, 3, "S10")])
def test_closed_trick_feed_preserves_stopper_before_last_actor(rules, prefix_policy, position, expected):
    hands = {Seat.NORTH: ("DA", "D10", "HA", "H10", "SA", "CA", "SQ", "CK"),
        Seat.WEST: ("D9", "DJ", "DQ", "DK", "C7", "C8", "C9", "CJ"),
        Seat.SOUTH: ("D7", "D8", "H7", "H8", "H9", "HJ", "S7", "S8"),
        Seat.EAST: ("HQ", "HK", "SJ", "S9", "S10", "SK", "C10", "CQ")}
    played = (("west", "DK"), ("south", "D7"))
    if position == 3:
        hands[Seat.NORTH] = ("D7", *hands[Seat.NORTH][1:])
        hands[Seat.SOUTH] = ("DA", *hands[Seat.SOUTH][1:])
        played = (("north", "D7"), ("west", "DK"), ("south", "D8"))
    prefix = tuple(TailPrefixActionV1.build(seat, card) for seat, card in played)
    build = OrToolsPlayTailStageV1(native_conversion=True).build_model(
        cards=tuple(_fixed_ownership(hands)), ownership=_fixed_ownership(hands),
        contract="sun", trump=None, first_leader=played[0][0], prefix=prefix, play_step_count=4,
        play_domain=rules.play_domain, structural_domain=rules.structural_domain,
        **(dict(policy_prefix_steps=position+1, policy_id=NATIVE_CLOSED_TRICK_FEED_POLICY_ID) if prefix_policy else {}))
    if not prefix_policy:
        build.constrain_native_observer_safe_policy(team=Team.WEST_EAST,
            policy_id=NATIVE_CLOSED_TRICK_FEED_POLICY_ID, start_step=position)
    result = build.solve(maximum_time_seconds=10)
    assert result.plays[position].card.code == expected
    if position == 2:
        assert result.plays[3].card.code in {"DA", "D10"}
        assert result.trick_winners[0] == Seat.NORTH
    request = ScenarioProofRequestV1.build({"schemaVersion": 1, "mode": "bounded",
        "visibility": "actor_information_set", "strategyClass": "pure_deterministic",
        "claim": "winning_root_action_count", "targetTeam": "west_east",
        "utility": "awarded_point_margin", "successThreshold": 0, "budgetProfile": "test",
        "horizon": {"unit": "tricks", "value": 1}, "boundaryEvaluator": "current_raw_point_margin_v1"})
    state = {"contract": "sun", "trumpSuit": None, "completedTricks": [],
        "hands": {seat.name.lower(): list(hand) for seat, hand in hands.items()},
        "currentTrick": {"index": 0, "leader": played[0][0], "plays": [
            {"actor": row.actor.name.lower(), "card": row.card.code} for row in prefix]}}
    choices = []
    for swapped in (False, True):
        if swapped:
            state["hands"]["north"], state["hands"]["south"] = state["hands"]["south"], state["hands"]["north"]
        choices.append(_observer_safe_play_policy_artifact_v1(rules,
            SimpleNamespace(decision_kind="play", next_actor="east", state=state,
                legal_actions=tuple({"code": card} for card in hands[Seat.EAST])),
            request, actual_action=None, policy_id=NATIVE_CLOSED_TRICK_FEED_POLICY_ID,
            maximum_seconds=2, cancel_check=None)[0])
    assert choices == [expected, expected]


@pytest.mark.parametrize("prefix_policy", [False, True])
def test_guard_protection_retains_seven_for_ten_in_both_native_paths_and_runtime(rules, prefix_policy):
    hands = {Seat.NORTH: ("D10", "CA", "CQ"), Seat.WEST: ("DJ", "CJ", "H8"),
             Seat.SOUTH: ("D7", "C8", "S7"), Seat.EAST: ("C7", "C10", "HK")}
    played = (("north", "D10"), ("west", "DJ"), ("south", "D7"))
    expected = "HK"
    if prefix_policy:
        hands = {Seat.NORTH: ("DA", "CA", "CQ", "C8", "C9", "CK", "H7", "H8"),
                 Seat.WEST: ("D7", "D10", "DJ", "DQ", "DK", "S7", "S8", "S9"),
                 Seat.SOUTH: ("D8", "D9", "HA", "H10", "CJ", "HQ", "SA", "SJ"),
                 Seat.EAST: ("C7", "C10", "HK", "H9", "SQ", "S10", "SK", "HJ")}
        played = (("north", "DA"), ("west", "D7"), ("south", "D8"))
        expected = "H9"
    prefix = tuple(TailPrefixActionV1.build(seat, card) for seat, card in played)
    kwargs = dict(cards=tuple(_fixed_ownership(hands)), ownership=_fixed_ownership(hands),
        contract="sun", trump=None, first_leader="north", prefix=prefix,
        play_domain=rules.play_domain, structural_domain=rules.structural_domain)
    build = OrToolsPlayTailStageV1(native_conversion=True).build_model(**kwargs,
        **(dict(policy_prefix_steps=4, policy_id=NATIVE_GUARD_PROTECTION_POLICY_ID) if prefix_policy else {}))
    if not prefix_policy:
        build.constrain_native_observer_safe_policy(team=Team.WEST_EAST,
            policy_id=NATIVE_GUARD_PROTECTION_POLICY_ID, start_step=3)
    result = build.solve(maximum_time_seconds=10)
    assert result.plays[3].card.code == expected
    reference = OrToolsPlayTailStageV1(native_conversion=True).build_model(**kwargs)
    reference.constrain_native_observer_safe_policy(team=Team.WEST_EAST,
        policy_id="observer_safe_trick_control_v1", start_step=3)
    assert reference.solve(maximum_time_seconds=10).plays[3].card.code == "C7"
    request = ScenarioProofRequestV1.build({"schemaVersion": 1, "mode": "bounded",
        "visibility": "actor_information_set", "strategyClass": "pure_deterministic",
        "claim": "winning_root_action_count", "targetTeam": "west_east",
        "utility": "awarded_point_margin", "successThreshold": 0, "budgetProfile": "test",
        "horizon": {"unit": "tricks", "value": 1}, "boundaryEvaluator": "current_raw_point_margin_v1"})
    state = {"contract": "sun", "trumpSuit": None, "completedTricks": [],
        "hands": {seat.name.lower(): list(hand) for seat, hand in hands.items()},
        "currentTrick": {"index": 5, "leader": "north", "plays": [
            {"actor": row.actor.name.lower(), "card": row.card.code} for row in prefix]}}
    actions = []
    for hidden in (False, True):
        if hidden:
            state["hands"]["north"], state["hands"]["south"] = state["hands"]["south"], state["hands"]["north"]
        actions.append(_observer_safe_play_policy_artifact_v1(rules,
            SimpleNamespace(decision_kind="play", next_actor="east", state=state,
                legal_actions=tuple({"code": card} for card in hands[Seat.EAST])),
            request, actual_action=None, policy_id=NATIVE_GUARD_PROTECTION_POLICY_ID,
            maximum_seconds=2, cancel_check=None)[0])
    assert actions == [expected, expected]


@pytest.mark.parametrize("seed,actor,hand,partner,project,remaining,expected", [
    (722162, "west", ["H9", "SK"], "east", ["HJ", "HQ", "HK"],
     ["H9", "SK", "HJ", "HQ", "SA", "S9", "D7", "C7"], "H9"),
    (723502, "south", ["D7", "CJ"], "north", ["D10", "DJ", "DQ"],
     ["D7", "CJ", "D10", "DA", "C10", "CK", "H7", "S7"], "D7"),
])
def test_public_partner_entry_rejects_audited_obvious_leads(rules, seed, actor, hand, partner, project, remaining, expected):
    from baloot_platform.research.ortools_research_executor import PUBLIC_PARTNER_ENTRY_BASELINE_ID
    from baloot_platform.research.decision_selection import baseline_strictly_prefers_alternative_v1
    public = [card.code for card in map(Card, range(32)) if card.code not in remaining]
    state = {"contract": "sun", "trumpSuit": None,
        "currentTrick": {"index": 6, "plays": []},
        "completedTricks": [{"plays": [{"actor": actor, "card": code}]} for code in public],
        "hands": {actor: hand, partner: ["SA"]},
        "projects": [{"owner": partner, "cards": project, "revealed": True}]}
    if seed == 723502:
        state["completedTricks"] = [trick for trick in state["completedTricks"]
            if trick["plays"][0]["card"] not in {"C7", "DQ"}]
        state["completedTricks"].append({"plays": [
            {"actor": "west", "card": "C7"}, {"actor": partner, "card": "DQ"}]})
    request = ScenarioProofRequestV1.build({"schemaVersion": 1, "mode": "bounded",
        "visibility": "actor_information_set", "strategyClass": "pure_deterministic",
        "claim": "winning_root_action_count", "targetTeam": Seat.parse(actor).team.name.lower(),
        "utility": "awarded_point_margin", "successThreshold": 0, "budgetProfile": "test",
        "horizon": {"unit": "tricks", "value": 1}, "boundaryEvaluator": "current_raw_point_margin_v1"})
    normalized = SimpleNamespace(decision_kind="play", next_actor=actor, state=state,
                                 legal_actions=tuple({"code": card} for card in hand))
    artifacts = []
    for hidden in (["SA"], ["C8"]):
        state["hands"][partner] = hidden
        selected, artifact = _observer_safe_play_policy_artifact_v1(rules, normalized, request,
            actual_action=None, policy_id=PUBLIC_PARTNER_ENTRY_BASELINE_ID,
            maximum_seconds=2, cancel_check=None)
        assert selected == expected
        assert not baseline_strictly_prefers_alternative_v1(artifact["proofModel"]["actionQuality"], expected)
        artifacts.append(artifact)
    assert artifacts[0] == artifacts[1]


@pytest.mark.parametrize("baseline_id", ["observer_safe_public_partner_entry_v2", "observer_safe_public_stranded_master_v3", "observer_safe_public_master_retention_v4", "observer_safe_public_exhausted_suit_v5"])
def test_public_partner_entry_ignores_unrevealed_projects_and_uses_strict_schema(rules, baseline_id):
    from baloot_platform.research.ortools_research_executor import _public_partner_entry_cards_v2
    from baloot_platform.research.scenario_contracts import build_scenario_video_interestingness_v1
    state = {"completedTricks": [], "currentTrick": {"plays": []},
        "hands": {"west": ["H9", "SK"]},
        "projects": [{"owner": "east", "cards": ["HA", "H10", "HK"], "revealed": False}]}
    before = _public_partner_entry_cards_v2(state, Seat.WEST, rules.play_domain, Contract.SUN, None)
    state["projects"][0]["cards"] = ["DA", "D10", "DK"]
    assert _public_partner_entry_cards_v2(state, Seat.WEST, rules.play_domain, Contract.SUN, None) == before
    policy = {"schemaVersion": 4, "maximumFinalMargin": 40, "minimumPointSwing": 20,
        "maximumFeaturedTrick": 7, "minimumMeaningfulDecisions": 1, "minimumConsistentWorlds": 2,
        "minimumBaselineRegret": 20, "minimumNextBestRegret": 20,
        "baselinePolicyId": baseline_id, "allowTieRuleOnlyLessons": False}
    assert build_scenario_video_interestingness_v1(policy)["baselinePolicyId"] == policy["baselinePolicyId"]


@pytest.mark.parametrize("case,hand,plays,unseen,expected", [
    ("722109", ["SQ", "DK"], [("north", "HK"), ("west", "S10")], ["DJ", "CQ", "CK", "C10"], "SQ"),
    ("722974", ["DK", "C10"], [("north", "SQ"), ("west", "D8")], ["DJ", "CQ", "CK", "C7"], "C10"),
    ("partner_not_secure", ["SQ", "DK"], [("north", "HK"), ("west", "S10")], ["HA", "DJ", "CQ", "CK"], "DK"),
    ("retained_not_master", ["SQ", "DK"], [("north", "HK"), ("west", "S10")], ["DA", "DJ", "CQ", "CK"], "DK"),
    ("suit_not_exhausted", ["SQ", "DK"], [("north", "HK"), ("west", "S10")], ["S7", "DJ", "CQ", "CK"], "DK"),
])
def test_public_stranded_master_baseline_is_versioned_and_observer_safe(rules, case, hand, plays, unseen, expected):
    from baloot_platform.research.ortools_research_executor import (
        PUBLIC_PARTNER_ENTRY_BASELINE_ID, PUBLIC_STRANDED_MASTER_BASELINE_ID)
    from baloot_platform.research.decision_selection import baseline_strictly_prefers_alternative_v1
    remaining = {*hand, *unseen, *(card for _, card in plays)}
    public = [card.code for card in map(Card, range(32)) if card.code not in remaining]
    state = {"contract": "sun", "trumpSuit": None,
        "currentTrick": {"index": 6, "plays": [{"actor": seat, "card": card} for seat, card in plays]},
        "completedTricks": [{"plays": [{"actor": "south", "card": code}]} for code in public],
        "hands": {"south": hand, "north": unseen[:1], "east": unseen[1:3], "west": unseen[3:]}}
    request = ScenarioProofRequestV1.build({"schemaVersion": 1, "mode": "bounded",
        "visibility": "actor_information_set", "strategyClass": "pure_deterministic",
        "claim": "winning_root_action_count", "targetTeam": "north_south",
        "utility": "awarded_point_margin", "successThreshold": 0, "budgetProfile": "test",
        "horizon": {"unit": "tricks", "value": 1}, "boundaryEvaluator": "current_raw_point_margin_v1"})
    normalized = SimpleNamespace(decision_kind="play", next_actor="south", state=state,
        legal_actions=tuple({"code": card} for card in hand))
    artifacts = []
    for hidden in (unseen, list(reversed(unseen))):
        state["hands"].update(north=hidden[:1], east=hidden[1:3], west=hidden[3:])
        old, old_artifact = _observer_safe_play_policy_artifact_v1(rules, normalized, request,
            actual_action=None, policy_id=PUBLIC_PARTNER_ENTRY_BASELINE_ID, maximum_seconds=2, cancel_check=None)
        assert old == ("C10" if case == "722974" else "DK")
        selected, artifact = _observer_safe_play_policy_artifact_v1(rules, normalized, request,
            actual_action=None, policy_id=PUBLIC_STRANDED_MASTER_BASELINE_ID, maximum_seconds=2, cancel_check=None)
        assert selected == expected
        if case == "722109":
            assert baseline_strictly_prefers_alternative_v1(old_artifact["proofModel"]["actionQuality"], "SQ")
            assert not baseline_strictly_prefers_alternative_v1(artifact["proofModel"]["actionQuality"], "SQ")
        if case == "722974":
            assert baseline_strictly_prefers_alternative_v1(artifact["proofModel"]["actionQuality"], "DK")
        artifacts.append(artifact)
    assert artifacts[0] == artifacts[1]


@pytest.mark.parametrize("case", json.loads((Path(__file__).parent / "fixtures/audit5_public_baseline.json").read_text()))
@pytest.mark.parametrize("baseline_id", ["observer_safe_public_master_retention_v4", "observer_safe_public_exhausted_suit_v5"])
def test_audit5_master_retention_baseline(rules, case, baseline_id):
    from baloot_platform.research.ortools_research_executor import PUBLIC_STRANDED_MASTER_BASELINE_ID
    from baloot_platform.research.decision_selection import baseline_strictly_prefers_alternative_v1
    state = {"contract": "sun", "trumpSuit": None,
        "currentTrick": {"index": 6, "plays": case["current"]},
        "completedTricks": [{"plays": rows} for rows in case["history"]],
        "hands": {case["actor"]: case["hand"]}, "projects": case["projects"]}
    request = ScenarioProofRequestV1.build({"schemaVersion": 1, "mode": "bounded",
        "visibility": "actor_information_set", "strategyClass": "pure_deterministic",
        "claim": "winning_root_action_count", "targetTeam": Seat.parse(case["actor"]).team.name.lower(),
        "utility": "awarded_point_margin", "successThreshold": 0, "budgetProfile": "test",
        "horizon": {"unit": "tricks", "value": 1}, "boundaryEvaluator": "current_raw_point_margin_v1"})
    normalized = SimpleNamespace(decision_kind="play", next_actor=case["actor"], state=state,
        legal_actions=tuple({"code": card} for card in case["hand"]))
    artifacts = []
    for other_hand in (["SA"], ["C8"]):
        for seat in Seat:
            if seat.name.lower() != case["actor"]:
                state["hands"][seat.name.lower()] = other_hand
        _, old = _observer_safe_play_policy_artifact_v1(rules, normalized, request,
            actual_action=None, policy_id=PUBLIC_STRANDED_MASTER_BASELINE_ID, maximum_seconds=2, cancel_check=None)
        assert baseline_strictly_prefers_alternative_v1(old["proofModel"]["actionQuality"], case["chosen"])
        selected, artifact = _observer_safe_play_policy_artifact_v1(rules, normalized, request,
            actual_action=None, policy_id=baseline_id, maximum_seconds=2, cancel_check=None)
        disagreement = baseline_strictly_prefers_alternative_v1(artifact["proofModel"]["actionQuality"], case["chosen"])
        assert disagreement == (case["seed"] in {729156, 728948, 731186}), (selected, case)
        artifacts.append(artifact)
    assert artifacts[0] == artifacts[1]


@pytest.mark.parametrize("situation,expected", [
    ("insecure_partner", "DK"), ("opponent", "DK"),
    ("diamond_still_hidden", "C10"), ("leading", "C10"),
])
def test_audit6_exhausted_suit_baseline_ignores_insecure_partner_and_hidden_hands(rules, situation, expected):
    from baloot_platform.research.ortools_research_executor import (
        PUBLIC_MASTER_RETENTION_BASELINE_ID, PUBLIC_EXHAUSTED_SUIT_BASELINE_ID)
    from baloot_platform.research.decision_selection import baseline_strictly_prefers_alternative_v1
    case = json.loads((Path(__file__).parent / "fixtures/audit6_public_baseline.json").read_text())
    state = {"contract": "sun", "trumpSuit": None,
        "currentTrick": {"index": 6, "plays": case["current"]},
        "completedTricks": [{"plays": rows} for rows in case["history"]],
        "hands": {"west": case["hand"]}, "projects": case["projects"]}
    if situation == "opponent":
        state["currentTrick"]["plays"][0]["actor"] = "south"
    elif situation == "diamond_still_hidden":
        state["completedTricks"][5]["plays"][1]["card"] = "CJ"
    elif situation == "leading":
        state["completedTricks"].append({"plays": state["currentTrick"]["plays"]})
        state["currentTrick"]["plays"] = []
    request = ScenarioProofRequestV1.build({"schemaVersion": 1, "mode": "bounded",
        "visibility": "actor_information_set", "strategyClass": "pure_deterministic",
        "claim": "winning_root_action_count", "targetTeam": Seat.WEST.team.name.lower(),
        "utility": "awarded_point_margin", "successThreshold": 0, "budgetProfile": "test",
        "horizon": {"unit": "tricks", "value": 1}, "boundaryEvaluator": "current_raw_point_margin_v1"})
    normalized = SimpleNamespace(decision_kind="play", next_actor="west", state=state,
        legal_actions=tuple({"code": card} for card in case["hand"]))
    artifacts = []
    for hidden in (["HK"], ["C7"]):
        state["hands"].update(north=hidden, south=hidden, east=hidden)
        old, _ = _observer_safe_play_policy_artifact_v1(rules, normalized, request,
            actual_action=None, policy_id=PUBLIC_MASTER_RETENTION_BASELINE_ID,
            maximum_seconds=2, cancel_check=None)
        assert old == ("DK" if situation == "opponent" else "C10")
        selected, artifact = _observer_safe_play_policy_artifact_v1(rules, normalized, request,
            actual_action=None, policy_id=PUBLIC_EXHAUSTED_SUIT_BASELINE_ID,
            maximum_seconds=2, cancel_check=None)
        assert selected == expected
        assert baseline_strictly_prefers_alternative_v1(artifact["proofModel"]["actionQuality"], "DK") == (expected != "DK")
        artifacts.append(artifact)
    assert artifacts[0] == artifacts[1]


def test_established_policy_preserves_winners_and_ignores_hidden_seat_assignment(rules):
    hands = {Seat.NORTH: ("SK", "H10"), Seat.WEST: ("HQ", "D8"),
             Seat.SOUTH: ("SJ", "D7"), Seat.EAST: ("HA", "H7")}
    # SA/S10 are public completed cards; SK is an established Sun winner.
    request = ScenarioProofRequestV1.build({"schemaVersion": 1, "mode": "bounded",
        "visibility": "actor_information_set", "strategyClass": "pure_deterministic",
        "claim": "winning_root_action_count", "targetTeam": "north_south",
        "utility": "awarded_point_margin", "successThreshold": 0, "budgetProfile": "test",
        "horizon": {"unit": "tricks", "value": 1},
        "boundaryEvaluator": "current_raw_point_margin_v1"})
    public = [card for card in map(Card, range(32)) if card.code not in _fixed_ownership(hands)]
    normalized = SimpleNamespace(decision_kind="play", next_actor="north",
        state={"contract": "sun", "trumpSuit": None,
            "currentTrick": {"index": 6, "leader": "north", "plays": []},
            "completedTricks": [{"plays": [{"card": card.code} for card in public[start:start+4]]}
                                for start in range(0, len(public), 4)],
            "hands": {seat.name.lower(): list(hand) for seat, hand in hands.items()}},
        legal_actions=({"code": "SK"}, {"code": "H10"}))
    artifacts = []
    for swapped in (False, True):
        ownership = _fixed_ownership(hands)
        if swapped:
            ownership["HA"], ownership["HQ"] = ownership["HQ"], ownership["HA"]
            normalized.state["hands"]["west"], normalized.state["hands"]["east"] = (
                normalized.state["hands"]["east"], normalized.state["hands"]["west"])
        build = OrToolsPlayTailStageV1(native_conversion=True).build_model(
            cards=tuple(ownership), ownership=ownership, contract="sun", trump=None, first_leader="north",
            play_domain=rules.play_domain, structural_domain=rules.structural_domain)
        build.constrain_native_observer_safe_policy(team=Team.NORTH_SOUTH,
            policy_id=NATIVE_ESTABLISHED_CONTROL_POLICY_ID)
        result = build.solve(maximum_time_seconds=10)
        assert result.plays[0].card.code == "SK"
        master_index = next(index for index, variable in enumerate(build.model.proto.variables)
            if variable.name == f"visible_policy_{int(Team.NORTH_SOUTH)}_0_selected_master")
        no_master = build.model.clone()
        no_master.add(no_master.get_bool_var_from_proto_index(master_index) == 0)
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.max_time_in_seconds = 10
        assert solver.solve(no_master) == cp_model.INFEASIBLE
        wrong_winner = build.model.clone()
        wrong_winner.add(build._extract.winner_seats[0] != int(Seat.NORTH))
        assert solver.solve(wrong_winner) == cp_model.INFEASIBLE
        # Reject the off-suit Ten, not merely select a favorable witness.
        changed = build.model.clone()
        changed.add(build._extract.card_ids[0] == Card.parse("H10").id)
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        assert solver.solve(changed) == cp_model.INFEASIBLE
        selected, artifact = _observer_safe_play_policy_artifact_v1(rules, normalized, request,
            actual_action=None, maximum_seconds=2, cancel_check=None,
            policy_id=NATIVE_ESTABLISHED_CONTROL_POLICY_ID)
        assert selected == "SK"
        artifacts.append(artifact)
    assert artifacts[0] == artifacts[1]


@pytest.mark.parametrize("contract,trump", [("sun", None), ("hokum", "hearts")])
def test_established_master_winner_implication_is_redundant(rules, contract, trump):
    build = OrToolsPlayTailStageV1(native_conversion=True).build_model(
        cards=("SA", "SK", "HA", "H9", "DA", "D8", "CA", "C7"),
        ownership=None, contract=contract, trump=trump, first_leader="north",
        play_domain=rules.play_domain, structural_domain=rules.structural_domain)
    build.constrain_native_observer_safe_policy(team=Team.NORTH_SOUTH,
        policy_id=NATIVE_ESTABLISHED_CONTROL_POLICY_ID)
    names = {var.name: index for index, var in enumerate(build.model.proto.variables)}
    active = names[f"visible_policy_{int(Team.NORTH_SOUTH)}_0_active"]
    master = names[f"visible_policy_{int(Team.NORTH_SOUTH)}_0_selected_master"]
    original = build.model.clone()
    implications = [constraint for constraint in original.proto.constraints
        if list(constraint.enforcement_literal) == [active, master]]
    assert len(implications) == 1
    # Disable the new implication, then ask the original native rules for a
    # counterexample across every symbolic allocation, not one fixed hand.
    implications[0].enforcement_literal.append(original.new_constant(0).index)
    original.add(original.get_bool_var_from_proto_index(master) == 1)
    original.add(build._extract.winner_seats[0] != build._extract.actors[0])
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 10
    assert solver.solve(original) == cp_model.INFEASIBLE


@pytest.mark.parametrize("contract,trump", [("sun", None), ("hokum", "hearts")])
@pytest.mark.parametrize("play_steps", [4, 8])
def test_native_unique_winner_rank_is_redundant(rules, contract, trump, play_steps):
    build = OrToolsPlayTailStageV1(native_conversion=True).build_model(
        cards=("S7", "SA", "H7", "HJ", "D7", "DA", "C7", "CA"),
        ownership=None, contract=contract, trump=trump, first_leader="north",
        play_domain=rules.play_domain, structural_domain=rules.structural_domain,
        play_step_count=play_steps)
    original = build.model.clone()
    constraints = [row for row in original.proto.constraints if row.name.startswith("winner_unique_rank_")]
    assert constraints
    violations = []
    for row in constraints:
        literals = [original.get_bool_var_from_proto_index(value) if value >= 0
                    else original.get_bool_var_from_proto_index(-value - 1).Not()
                    for value in row.enforcement_literal]
        expression = sum(coefficient * original.get_int_var_from_proto_index(index)
                         for index, coefficient in zip(row.linear.vars, row.linear.coeffs, strict=True))
        # Disable the added implication. The original winner/uniqueness model
        # must independently forbid any eligible non-winner tying its maximum.
        row.enforcement_literal.append(original.new_constant(0).index)
        violation = original.new_bool_var(f"violates_{len(violations)}")
        original.add_bool_and(literals).only_enforce_if(violation)
        original.add(expression >= 0).only_enforce_if(violation)
        violations.append(violation)
    original.add_bool_or(violations)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 10
    assert solver.solve(original) == cp_model.INFEASIBLE


@pytest.mark.parametrize("play_steps", [4, 8])
def test_ekka_rejects_stronger_unplayed_card_beyond_horizon(rules, play_steps):
    hands = {
        Seat.NORTH: ("SK", "C7"),
        Seat.WEST: ("SA", "S7"),
        Seat.SOUTH: ("D7", "D8"),
        Seat.EAST: ("H7", "H8"),
    }
    with pytest.raises(PlayTailSolveStopped) as stopped:
        _solve(
            rules, hands, contract=Contract.HOKUM, trump=Suit.HEARTS,
            prefix=(TailPrefixActionV1.build("north", "SK", ekka=True),),
            play_step_count=play_steps,
        )
    assert stopped.value.solver_status == "infeasible"


def test_generated_search_can_choose_ekka_as_a_native_action(rules) -> None:
    build = OrToolsPlayTailStageV1().build_model(
        cards=("SA", "H7", "D7", "C7"),
        ownership={
            "SA": Seat.NORTH,
            "H7": Seat.WEST,
            "D7": Seat.SOUTH,
            "C7": Seat.EAST,
        },
        contract=Contract.HOKUM,
        trump=Suit.HEARTS,
        first_leader=Seat.NORTH,
        play_domain=rules.play_domain,
        structural_domain=rules.structural_domain,
    )
    build.model.add(build.play_state_context.play_slots[28].ekka == 1)

    result = build.solve(maximum_time_seconds=5)

    assert result.plays[0].card.code == "SA"
    assert result.plays[0].ekka is True


def test_variable_ownership_is_solved_inside_declared_domains(rules) -> None:
    cards = ("S7", "S8", "S9", "SA", "H7", "H8", "H9", "HA")
    allowed = {
        card: (Seat.NORTH, Seat.SOUTH)
        if card.startswith("S")
        else (Seat.WEST, Seat.EAST)
        for card in cards
    }
    result = OrToolsPlayTailStageV1().solve(
        cards=cards,
        ownership=allowed,
        contract=Contract.SUN,
        trump=None,
        first_leader=Seat.NORTH,
        play_domain=rules.play_domain,
        structural_domain=rules.structural_domain,
    )
    assert all(len(hand) == 2 for hand in result.initial_hands)
    for seat in Seat:
        assert all(
            seat in allowed[card.code] for card in result.initial_hands[int(seat)]
        )


def test_generic_tail_predicates_encode_two_ace_low_card_pattern(rules) -> None:
    hands = {
        Seat.NORTH: ("H7", "D8", "DA"),
        Seat.WEST: ("CA", "C7", "C8"),
        Seat.SOUTH: ("HA", "S7", "SA"),
        Seat.EAST: ("D7", "S8", "C9"),
    }
    prefix = (
        TailPrefixActionV1.build("south", "HA"),
        TailPrefixActionV1.build("east", "D7"),
        TailPrefixActionV1.build("north", "H7"),
    )
    result = _solve(
        rules,
        hands,
        leader=Seat.SOUTH,
        prefix=prefix,
        action_constraints=(
            TailActionConstraintV1.build(
                step=3,
                team=Team.WEST_EAST,
                ranks=(Rank.ACE,),
                actor_team_wins_trick=False,
            ),
        ),
        winner_constraints=(
            TailTrickWinnerConstraintV1.build(
                trick=1, team=Team.WEST_EAST, ranks=(Rank.SEVEN, Rank.EIGHT, Rank.NINE)
            ),
            TailTrickWinnerConstraintV1.build(
                trick=2, team=Team.WEST_EAST, ranks=(Rank.SEVEN, Rank.EIGHT, Rank.NINE)
            ),
        ),
        card_count_constraints=(
            TailTrickCardCountConstraintV1.build(
                trick=2,
                owner_team=Team.NORTH_SOUTH,
                ranks=(Rank.ACE,),
                minimum=2,
                maximum=2,
            ),
        ),
    )
    assert result.next_actor == Seat.WEST
    assert [card.code for card in result.legal_actions] == ["C7", "C8", "CA"]
    assert result.plays[3].card == Card.parse("CA")
    assert result.trick_winners[0].team == Team.NORTH_SOUTH
    assert (
        result.trick_winners[1].team == result.trick_winners[2].team == Team.WEST_EAST
    )
    assert result.trick_winning_cards[1].rank in {Rank.SEVEN, Rank.EIGHT, Rank.NINE}
    assert result.trick_winning_cards[2].rank in {Rank.SEVEN, Rank.EIGHT, Rank.NINE}
    final = result.plays[8:12]
    assert (
        sum(
            play.actor.team == Team.NORTH_SOUTH and play.card.rank == Rank.ACE
            for play in final
        )
        == 2
    )


def test_complete_round_can_escape_kabout_after_losing_five_tricks(rules) -> None:
    result = OrToolsPlayTailStageV1().solve(
        cards=tuple(Card(card_id) for card_id in range(32)),
        ownership=None,
        contract=Contract.SUN,
        trump=None,
        first_leader=Seat.NORTH,
        play_domain=rules.play_domain,
        structural_domain=rules.structural_domain,
        scoring_domain=ScoringDomainIRV1.from_rules(rules.data),
        play_step_count=32,
        decision_prefix_length=21,
        decision_actor=Seat.SOUTH,
        action_constraints=(
            TailActionConstraintV1.build(step=21, actor=Seat.SOUTH, ranks=(Rank.ACE,)),
        ),
        winner_constraints=(
            *(TailTrickWinnerConstraintV1.build(trick=index, team=Team.WEST_EAST) for index in range(5)),
            TailTrickWinnerConstraintV1.build(trick=5, seat=Seat.SOUTH),
            TailTrickWinnerConstraintV1.build(trick=6, seat=Seat.NORTH),
            TailTrickWinnerConstraintV1.build(trick=7, seat=Seat.SOUTH),
        ),
        maximum_time_seconds=20,
    )

    assert [winner.team for winner in result.trick_winners[:5]] == [Team.WEST_EAST] * 5
    assert result.plays[21].actor == Seat.SOUTH
    assert result.plays[21].card.rank == Rank.ACE
    assert result.trick_winners[5:] == (Seat.SOUTH, Seat.NORTH, Seat.SOUTH)


def test_play_predicate_lowering_can_escape_kabout_after_losing_five_tricks(
    rules,
) -> None:
    manifest = shared_scenario_predicate_catalog_lowering_manifest_v2()
    all_ranks = {
        "ranks": [
            "seven",
            "eight",
            "nine",
            "ten",
            "jack",
            "queen",
            "king",
            "ace",
        ]
    }
    predicates = [
        *(
            ScenarioPredicateV1.build(
                {
                    "predicateId": f"PRED-WE-{index}",
                    "kind": "play.trick_winner_matches",
                    "version": 1,
                    "hardness": "hard",
                    "arguments": {
                        "trickIndex": index,
                        "team": "west_east",
                        "selector": all_ranks,
                    },
                }
            )
            for index in range(5)
        ),
        ScenarioPredicateV1.build(
            {
                "predicateId": "PRED-ACE",
                "kind": "play.action_matches",
                "version": 1,
                "hardness": "hard",
                "arguments": {
                    "trickIndex": 5,
                    "position": 1,
                    "actingSeat": "south",
                    "selector": {"ranks": ["ace"]},
                },
            }
        ),
        *(
            ScenarioPredicateV1.build(
                {
                    "predicateId": f"PRED-WINNER-{index}",
                    "kind": "play.trick_winner_matches",
                    "version": 1,
                    "hardness": "hard",
                    "arguments": {
                        "trickIndex": index,
                        "seat": seat,
                        "selector": all_ranks,
                    },
                }
            )
            for index, seat in ((5, "south"), (6, "north"), (7, "south"))
        ),
    ]

    def configure(build) -> None:
        for predicate in predicates:
            lowering = manifest.lower_predicate(predicate, build.predicate_context)
            build.model.add(lowering.literal == 1)

    result = OrToolsPlayTailStageV1().solve(
        cards=tuple(Card(card_id) for card_id in range(32)),
        ownership=None,
        contract=Contract.SUN,
        trump=None,
        first_leader=Seat.NORTH,
        play_domain=rules.play_domain,
        structural_domain=rules.structural_domain,
        scoring_domain=ScoringDomainIRV1.from_rules(rules.data),
        play_step_count=32,
        decision_prefix_length=21,
        decision_actor=Seat.SOUTH,
        construction_plan_hash="a" * 64,
        model_configurator=configure,
        maximum_time_seconds=20,
    )

    assert [winner.team for winner in result.trick_winners[:5]] == [
        Team.WEST_EAST
    ] * 5
    assert result.plays[21].actor == Seat.SOUTH
    assert result.plays[21].card.rank == Rank.ACE
    assert result.trick_winners[5:] == (Seat.SOUTH, Seat.NORTH, Seat.SOUTH)


def test_complete_round_can_start_last_two_tricks_with_low_winning_lead(
    rules,
) -> None:
    result = OrToolsPlayTailStageV1().solve(
        cards=tuple(Card(card_id) for card_id in range(32)),
        ownership=None,
        contract=Contract.SUN,
        trump=None,
        first_leader=Seat.NORTH,
        play_domain=rules.play_domain,
        structural_domain=rules.structural_domain,
        scoring_domain=ScoringDomainIRV1.from_rules(rules.data),
        play_step_count=32,
        decision_prefix_length=24,
        decision_actor=Seat.SOUTH,
        action_constraints=(
            TailActionConstraintV1.build(
                step=24,
                actor=Seat.SOUTH,
                ranks=(Rank.SEVEN, Rank.EIGHT, Rank.NINE),
            ),
        ),
        winner_constraints=(
            TailTrickWinnerConstraintV1.build(trick=6, seat=Seat.SOUTH),
            TailTrickWinnerConstraintV1.build(trick=7, seat=Seat.SOUTH),
        ),
        maximum_time_seconds=20,
    )

    assert result.plays[24].actor == Seat.SOUTH
    assert result.plays[24].card.rank in {Rank.SEVEN, Rank.EIGHT, Rank.NINE}
    assert result.trick_winners[6:] == (Seat.SOUTH, Seat.SOUTH)


def test_complete_round_matches_existing_cp_sat_authority(root: Path, rules) -> None:
    _, hands, actions = _full_round_fixture(root, rules)
    scoring = ScoringDomainIRV1.from_rules(rules.data)
    ownership = {card: seat for seat, cards in hands.items() for card in cards}
    result = OrToolsPlayTailStageV1().solve(
        cards=tuple(Card(card_id) for card_id in range(32)),
        ownership=ownership,
        contract=Contract.SUN,
        trump=None,
        first_leader=Seat.NORTH,
        play_domain=rules.play_domain,
        structural_domain=rules.structural_domain,
        scoring_domain=scoring,
        prefix=actions,
        maximum_time_seconds=30,
    )
    reference = OrToolsCpSatPlayCompilerV1().compile(
        hands=hands,
        contract=Contract.SUN,
        trump=None,
        declarer=Seat.NORTH,
        rules=rules,
        dealer=Seat.EAST,
        buyer=Seat.NORTH,
        required_cards=tuple(play.card for play in actions),
        timeout_seconds=30,
    )
    assert result.plays == actions
    assert result.next_actor is None
    assert result.legal_actions == ()
    assert all(not hand for hand in result.remaining_hands)
    assert result.raw_team_points == reference.raw_card_points == (14, 116)
    assert result.card_game_points == reference.card_game_points == (2, 24)
    assert result.scoring_domain_hash == scoring.sha256
    assert result.trick_winners == tuple(trick.winner for trick in reference.tricks)
    assert result.trick_raw_points == tuple(
        trick.raw_points for trick in reference.tricks
    )
    interpreter = ReferencePlayInterpreterV1(rules.play_domain, rules.structural_domain)
    remaining = {seat: list(cards) for seat, cards in hands.items()}
    current: list[ReferencePlayV1] = []
    for step, action in enumerate(actions):
        expected = interpreter.legal_moves(
            remaining[action.actor],
            current,
            actor=action.actor,
            contract=Contract.SUN,
            trump=None,
        )
        assert result.legal_actions_by_step[step] == expected
        remaining[action.actor].remove(action.card)
        current.append(ReferencePlayV1(action.actor, action.card, action.ekka))
        if len(current) == 4:
            current = []


def test_terminal_hokum_predicate_context_accepts_raw_total_above_sun_maximum(
    root: Path,
) -> None:
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    plays = tuple(
        TailPrefixActionV1(Seat.parse(actor), Card(card_id))
        for actor, card_id in (
            ("north", 15), ("west", 12), ("south", 9), ("east", 8),
            ("north", 31), ("west", 30), ("south", 28), ("east", 24),
            ("north", 23), ("west", 21), ("south", 17), ("east", 19),
            ("north", 14), ("west", 11), ("south", 10), ("east", 25),
            ("west", 3), ("south", 18), ("east", 5), ("north", 0),
            ("south", 20), ("east", 26), ("north", 16), ("west", 22),
            ("south", 27), ("east", 6), ("north", 4), ("west", 1),
            ("south", 29), ("east", 7), ("north", 2), ("west", 13),
        )
    )
    ownership = {play.card: play.actor for play in plays}
    build = OrToolsPlayTailStageV1(native_conversion=True).build_model(
        cards=tuple(Card(card_id) for card_id in range(32)),
        ownership=ownership,
        contract=Contract.HOKUM,
        trump=Suit.DIAMONDS,
        first_leader=Seat.NORTH,
        play_domain=rules.play_domain,
        structural_domain=rules.structural_domain,
        scoring_domain=ScoringDomainIRV1.from_rules(rules.data),
        prefix=plays,
        decision_prefix_length=32,
        play_step_count=32,
    )

    result = build.solve(maximum_time_seconds=10)

    assert result.raw_team_points == (148, 14)


def test_full_round_fixed_prefix_and_tail_bind_without_oracle_repair(
    root: Path, rules
) -> None:
    _, hands, actions = _full_round_fixture(root, rules)
    result = OrToolsPlayTailStageV1().solve(
        cards=tuple(Card(card_id) for card_id in range(32)),
        ownership={card: seat for seat, cards in hands.items() for card in cards},
        contract=Contract.SUN,
        trump=None,
        first_leader=Seat.NORTH,
        play_domain=rules.play_domain,
        structural_domain=rules.structural_domain,
        scoring_domain=ScoringDomainIRV1.from_rules(rules.data),
        prefix=actions[:16],
        tail=actions[20:],
        maximum_time_seconds=30,
    )
    assert result.plays[:16] == actions[:16]
    assert result.plays[20:] == actions[20:]
    assert result.prefix_length == 16
    assert result.next_actor == actions[16].actor
    assert actions[16].card in result.legal_actions


def test_full_round_decision_cut_enforces_exact_rules_preserving_distance(
    root: Path, rules
) -> None:
    deck, hands, _ = _full_round_fixture(root, rules)
    ownership = {card: seat for seat, cards in hands.items() for card in cards}
    common = {
        "cards": tuple(Card(card_id) for card_id in range(32)),
        "ownership": ownership,
        "contract": Contract.SUN,
        "trump": None,
        "first_leader": Seat.NORTH,
        "play_domain": rules.play_domain,
        "structural_domain": rules.structural_domain,
        "scoring_domain": ScoringDomainIRV1.from_rules(rules.data),
        "maximum_time_seconds": 30,
    }
    decision_step = 24
    purchase = deck[20]
    first = OrToolsPlayTailStageV1().solve(**common).at_prefix(decision_step)

    def diversity(result, purchase_card):
        state = {
            "nextActor": result.next_actor.name.lower(),
            "hands": {
                seat.name.lower(): [
                    card.code for card in result.remaining_hands[int(seat)]
                ]
                for seat in Seat
            },
            "purchaseCard": purchase_card.code,
            "currentTrick": {
                "leader": (
                    result.current_trick[0].actor.name.lower()
                    if result.current_trick
                    else None
                ),
                "plays": [
                    {
                        "seat": play.actor.name.lower(),
                        "card": play.card.code,
                    }
                    for play in result.current_trick
                ],
            },
            "contract": "sun",
            "trumpSuit": None,
            "buyer": "north",
            "dealer": "east",
        }
        return build_rules_preserving_diversity_document_v1(state)

    first_diversity = diversity(first, purchase)
    purchase_blind_vector = tuple(
        8 if status == 9 else status
        for status in first_diversity["decisionStateCardVector"]
    )
    with pytest.raises(PlayTailSolveStopped, match="infeasible"):
        OrToolsPlayTailStageV1().solve(
            **common,
            prefix=first.plays[:decision_step],
            tail=first.plays[decision_step:],
            decision_prefix_length=decision_step,
            decision_actor=first.next_actor,
            excluded_decision_status_vectors=(purchase_blind_vector,),
            minimum_decision_status_hamming_distance=1,
        )
    with pytest.raises(PlayTailSolveStopped, match="infeasible"):
        OrToolsPlayTailStageV1().solve(
            **common,
            prefix=first.plays[:decision_step],
            tail=first.plays[decision_step:],
            decision_prefix_length=decision_step,
            decision_actor=first.next_actor,
            decision_purchase_card=purchase,
            excluded_decision_status_vectors=(
                first_diversity["decisionStateCardVector"],
            ),
            minimum_decision_status_hamming_distance=4,
        )

    second = OrToolsPlayTailStageV1().solve(
        **common,
        decision_prefix_length=decision_step,
        decision_actor=first.next_actor,
        decision_purchase_card=purchase,
        excluded_decision_status_vectors=(
            first_diversity["decisionStateCardVector"],
        ),
        minimum_decision_status_hamming_distance=4,
    ).at_prefix(decision_step)
    first_variant = CampaignVariantV1.build(
        {"name": "first"}, symmetry_document=first_diversity
    )
    second_variant = CampaignVariantV1.build(
        {"name": "second"}, symmetry_document=diversity(second, purchase)
    )
    assert decision_state_card_hamming_distance_v1(
        first_variant, second_variant
    ) >= 4

    partial_step = decision_step + 1
    partial = first.at_prefix(partial_step)
    current_purchase = partial.current_trick[0].card
    partial_diversity = diversity(partial, current_purchase)
    with pytest.raises(PlayTailSolveStopped, match="infeasible"):
        OrToolsPlayTailStageV1().solve(
            **common,
            prefix=partial.plays[:partial_step],
            tail=partial.plays[partial_step:],
            decision_prefix_length=partial_step,
            decision_actor=partial.next_actor,
            decision_purchase_card=current_purchase,
            excluded_decision_status_vectors=(
                partial_diversity["decisionStateCardVector"],
            ),
            minimum_decision_status_hamming_distance=1,
        )


def test_play_prefix_lift_uses_shared_owner_of_card_variables(
    root: Path, rules
) -> None:
    from ortools.sat.python import cp_model

    deck, hands, actions = _full_round_fixture(root, rules)
    construction = OrToolsScenarioModelV1(
        rules=rules, dealer=Seat.EAST, required_buyer=Seat.NORTH
    )
    for position, card in enumerate(deck):
        construction.model.add(construction.card_at_position[position] == card.id)
    shared_solver = cp_model.CpSolver()
    result = OrToolsPlayTailStageV1().solve(
        cards=tuple(Card(card_id) for card_id in range(32)),
        ownership={card: seat for seat, cards in hands.items() for card in cards},
        contract=Contract.SUN,
        trump=None,
        first_leader=Seat.NORTH,
        play_domain=rules.play_domain,
        structural_domain=rules.structural_domain,
        scoring_domain=ScoringDomainIRV1.from_rules(rules.data),
        prefix=actions,
        shared_model=construction.model,
        shared_owner_of_card=construction.owner_of_card,
        solver=shared_solver,
        maximum_time_seconds=30,
    )
    assert result.plays == actions
    assert tuple(
        Seat(shared_solver.value(owner)) for owner in construction.owner_of_card
    ) == tuple(
        next(seat for seat, cards in hands.items() if Card(card_id) in cards)
        for card_id in range(32)
    )


@pytest.mark.parametrize("winner", list(Seat))
def test_shared_symbolic_ownership_searches_adversarial_hidden_world(rules, winner) -> None:
    from ortools.sat.python import cp_model

    cards = tuple(Card.parse(code) for code in ("S7", "S8", "S9", "SA"))
    model = cp_model.CpModel()
    owners = tuple(model.new_int_var(0, 3, f"hidden_owner_{index}") for index in range(32))
    # Only North's card is known. No opponent world is supplied or enumerated.
    model.add(owners[cards[0].id] == int(Seat.NORTH))
    build = OrToolsPlayTailStageV1().build_model(
        cards=cards, ownership=None, contract=Contract.SUN, trump=None,
        first_leader=Seat.NORTH, play_domain=rules.play_domain,
        structural_domain=rules.structural_domain,
        shared_model=model, shared_owner_of_card=owners,
        winner_constraints=(TailTrickWinnerConstraintV1.build(trick=0, seat=winner),),
    )
    assert build.model is model
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 5
    status = solver.solve(model)
    if winner == Seat.NORTH:
        assert status == cp_model.INFEASIBLE
    else:
        assert status == cp_model.OPTIMAL
        assert solver.value(owners[cards[0].id]) == int(Seat.NORTH)
        assert solver.value(owners[cards[-1].id]) == int(winner)
        assert sorted(solver.value(owners[card.id]) for card in cards) == [0, 1, 2, 3]


def test_tail_scope_and_infeasibility_fail_closed(rules) -> None:
    with pytest.raises(ValueError, match="one through eight"):
        OrToolsPlayTailStageV1().solve(
            cards=tuple(range(15)),
            ownership=None,
            contract="sun",
            trump=None,
            first_leader="north",
            play_domain=rules.play_domain,
            structural_domain=rules.structural_domain,
        )
    hands = {
        Seat.NORTH: ("S7",),
        Seat.WEST: ("S8",),
        Seat.SOUTH: ("S9",),
        Seat.EAST: ("SA",),
    }
    with pytest.raises(
        PlayTailSolveStopped, match="did not produce a legal witness"
    ) as stopped:
        _solve(
            rules,
            hands,
            action_constraints=(TailActionConstraintV1.build(step=0, actor=Seat.WEST),),
        )
    evidence = stopped.value.partial_receipt["evidence"]
    assert len(evidence["modelHash"]) == 64
    assert evidence["variableCount"] > 0
    assert evidence["constraintCount"] > 0
    assert evidence["solverStatus"] == "infeasible"
    assert evidence["branches"] >= 0
    assert evidence["conflicts"] >= 0


def test_new_tail_stage_has_no_runtime_or_engine_dependency(root: Path) -> None:
    source = (root / "src/baloot_platform/research/ortools_play_tail.py").read_text(
        encoding="utf-8"
    )
    assert "BalootEngine" not in source
    assert "core.engine" not in source
    assert "reference_play" not in source
    assert "HeuristicBot" not in source


@pytest.mark.parametrize("cards_per_hand,start_step", [(2, 0), (3, 0), (3, 1), (8, 0)])
@pytest.mark.parametrize("contract,trump", [(Contract.SUN, None), (Contract.HOKUM, Suit.HEARTS)])
def test_ranked_policy_uses_local_tail_slots(rules, cards_per_hand, start_step, contract, trump):
    from baloot_platform.core.models import LegalPlayActionV1
    from baloot_platform.research.ortools_proof import (
        DeclarativeActionPolicyV1, ProofGraphEdgeV1, select_declarative_policy_action_v1)
    from baloot_platform.research.scenario_contracts import canonical_sha256

    hands = {seat: tuple(card.code for card in map(Card, range(32))
        if int(card.suit) == int(seat))[:cards_per_hand] for seat in Seat}
    ownership = _fixed_ownership(hands)
    prefix = (TailPrefixActionV1.build(Seat.NORTH, hands[Seat.NORTH][0]),) if start_step else ()
    build = OrToolsPlayTailStageV1(native_conversion=True).build_model(
        cards=tuple(ownership), ownership=ownership, contract=contract, trump=trump,
        first_leader=Seat.NORTH, play_domain=rules.play_domain,
        structural_domain=rules.structural_domain, prefix=prefix)
    parsed = {}
    for team in Team:
        policies = {}
        for seat in Seat:
            if seat.team != team:
                continue
            core = dict(schemaVersion=1, seat=seat.name.lower(), policyId="fixed_action_map_v1",
                version="1.0.0", seed=0, choices=[], materialization="seeded_legal_action_order_v1")
            raw = {key: value for key, value in core.items() if key != "seat"}
            raw["profileHash"] = canonical_sha256(core)
            policies[seat.name.lower()] = raw
            parsed[seat] = DeclarativeActionPolicyV1.build(seat.name.lower(), raw)
        assert build.constrain_declarative_play_policies(team=team, policies=policies, start_step=start_step)
    result = build.solve(maximum_time_seconds=5)
    assert len(result.plays) == cards_per_hand * 4
    assert result.plays[:start_step] == prefix
    for step, actual in enumerate(result.plays[start_step:], start_step):
        actions = [LegalPlayActionV1(row.card, row.ekka) for row in result.legal_play_actions_by_step[step]]
        expected = select_declarative_policy_action_v1(parsed[actual.actor], [
            ProofGraphEdgeV1.build(action.action_id, action.code, "reference") for action in actions])
        assert LegalPlayActionV1(actual.card, actual.ekka).code == expected.action_code
