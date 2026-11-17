from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from ortools.sat.python import cp_model

import baloot_platform.research.ortools_tail_first_lift as tail_lift_module
import baloot_platform.research.ortools_play_tail as play_tail_module
import baloot_platform.research.ortools_research_executor as research_executor

from baloot_platform.core.cards import FULL_DECK, Card
from baloot_platform.core.dealing import DealLayoutV1
from baloot_platform.core.enums import Contract, Seat, Suit
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.scoring_ir import ScoringDomainIRV1
from baloot_platform.research.ortools_complete_state_graph import (
    CompleteStateGraphSeedV1,
    OrToolsCompleteStateGraphCompilerV1,
)
from baloot_platform.research.ortools_play_tail import (
    OrToolsPlayTailStageV1,
    PlayZeroObservationActionNoGoodV1,
    PlayTailSolveStopped,
    TailPrefixActionV1,
    apply_play_zero_observation_action_no_good_v1,
)
from baloot_platform.research.ortools_project_model import (
    OrToolsProjectLifecycleModelV1,
)
from baloot_platform.research.ortools_tail_first_candidate import (
    _bind_reachable_project_truth_v1,
)
from baloot_platform.research.scenario_predicate_manifest_v2 import (
    shared_scenario_predicate_catalog_lowering_manifest_v2,
)
from baloot_platform.research.ortools_tail_first_lift import (
    OrToolsTailFirstBackwardLiftV1,
    _bind_proof_seed_prefix,
    _preserve_tail_cards_before_decision,
    _seed_clone,
)
from baloot_platform.research.ortools_research_executor import (
    GenericCompleteTraceCompilerV1,
    OrToolsUnifiedResearchExecutorV1,
)
from baloot_platform.research.ortools_tail_proof_boundary import (
    OrToolsTailProofBoundaryV1,
)
from baloot_platform.research.scenario_contracts import (
    ResearchScenarioV1,
    ScenarioProofRequestV1,
)
from baloot_platform.research.tail_first_stage_planner import (
    TailFirstStagePlannerV1,
)


def _leaf(predicate_id: str, kind: str, arguments: dict, hardness: str = "hard"):
    return {
        "predicateId": predicate_id,
        "kind": kind,
        "version": 1,
        "hardness": hardness,
        "arguments": arguments,
    }


def test_native_boundary_uses_legal_prefix_for_authored_hands(rules, monkeypatch):
    suits = ("spades", "hearts", "diamonds", "clubs")
    hands = {seat: (Card(8 * int(seat)), Card(8 * int(seat) + 1)) for seat in Seat}
    tail = OrToolsPlayTailStageV1(native_conversion=True).solve(
        cards=tuple(card for hand in hands.values() for card in hand),
        ownership={card: seat for seat, hand in hands.items() for card in hand},
        contract="sun", trump=None, first_leader="north",
        play_domain=rules.play_domain, structural_domain=rules.structural_domain,
        decision_prefix_length=0, decision_actor="north", maximum_time_seconds=5,
    )
    scenario = ResearchScenarioV1.build({
        "schemaVersion": 1, "origin": {"kind": "search"},
        "constraints": {"op": "all", "children": [
            _leaf("PRED-CONTRACT", "auction.contract", {"contract": "sun"}),
            _leaf("PRED-BUYER", "auction.buyer", {"seat": "north"}),
            _leaf("PRED-CHOICES", "state.legal_action_count", {"minimum": 2, "maximum": 2}),
            *(_leaf(f"PRED-HAND-{int(seat)}", "deal.suit_count", {
                "seat": seat.name.lower(), "suit": suits[int(seat)],
                "minimum": 8, "maximum": 8, "handStage": "post_purchase",
            }) for seat in Seat),
        ]},
        "decisionPoint": {"kind": "play", "selector": {
            "trickIndex": 6, "position": 0, "remainingCards": 2,
            "actingSeat": "north",
        }},
        "outcome": _leaf("PRED-REPORT", "outcome.buyer_contract", {"status": "won"}, "report"),
    })
    plan = TailFirstStagePlannerV1().plan(scenario)
    boundary = OrToolsTailProofBoundaryV1(rules=rules, native_cp_sat=True)
    reached = boundary.solve(scenario, plan, tail, maximum_seconds=15)
    assert reached.status == "feasible", reached.reason
    assert reached.seed.base_raw == (120, 0)
    assert reached.seed.base_tricks == (6, 0)
    # The old aggregate relaxation admitted this impossible opening history.
    impossible = boundary.solve(scenario, plan, tail, required_base_raw_ns=0,
                                maximum_seconds=15)
    assert impossible.status == "infeasible", impossible.reason
    # A failed history must also activate this same native path without authored hands.
    raw = scenario.to_dict()
    raw.pop("sha256")
    raw["constraints"].pop("sha256")
    raw["constraints"]["children"] = [item for item in raw["constraints"]["children"]
        if not item["kind"].startswith("deal.")]
    relaxed = ResearchScenarioV1.build(raw)
    def observe_prefix(*args, **kwargs):
        assert len(kwargs["cards"]) == 32
        assert kwargs["shared_model"] is not None
        raise RuntimeError("native reachability activated")
    monkeypatch.setattr(OrToolsPlayTailStageV1, "build_model", observe_prefix)
    with pytest.raises(RuntimeError, match="native reachability activated"):
        boundary.solve(relaxed, TailFirstStagePlannerV1().plan(relaxed), tail,
                       excluded=(reached.no_good,), maximum_seconds=15)


def test_native_trick_five_tail_lifts_without_widening_proof_depth(rules):
    hands = {seat: tuple(Card(8 * int(seat) + rank) for rank in range(4, 8)) for seat in Seat}
    tail = OrToolsPlayTailStageV1(native_conversion=True).solve(
        cards=tuple(card for hand in hands.values() for card in hand),
        ownership={card: seat for seat, hand in hands.items() for card in hand},
        contract="sun", trump=None, first_leader="north",
        play_domain=rules.play_domain, structural_domain=rules.structural_domain,
        decision_prefix_length=0, decision_actor="north", maximum_time_seconds=10,
    )
    scenario = ResearchScenarioV1.build({
        "schemaVersion": 1, "origin": {"kind": "search"},
        "constraints": {"op": "all", "children": [
            _leaf("PRED-CONTRACT", "auction.contract", {"contract": "sun"}),
            _leaf("PRED-BUYER", "auction.buyer", {"seat": "north"}),
            _leaf("PRED-CHOICES", "state.legal_action_count", {"minimum": 4, "maximum": 4}),
            *(_leaf(f"PRED-HAND-{int(seat)}", "deal.suit_count", {
                "seat": seat.name.lower(), "suit": ("spades", "hearts", "diamonds", "clubs")[int(seat)],
                "minimum": 8, "maximum": 8, "handStage": "post_purchase",
            }) for seat in Seat),
        ]},
        "decisionPoint": {"kind": "play", "selector": {
            "trickIndex": 4, "position": 0, "remainingCards": 4, "actingSeat": "north",
        }},
        "outcome": _leaf("PRED-REPORT", "outcome.buyer_contract", {"status": "won"}, "report"),
    })
    plan = TailFirstStagePlannerV1(native_cp_sat=True).plan(scenario)
    reached = OrToolsTailProofBoundaryV1(rules=rules, native_cp_sat=True).solve(
        scenario, plan, tail, maximum_seconds=30,
    )
    assert reached.status == "feasible", reached.reason
    assert sum(reached.seed.base_tricks) == 4
    request = ScenarioProofRequestV1.build({
        "schemaVersion": 1, "mode": "exact", "visibility": "complete_state",
        "claim": "winning_root_action_count", "targetTeam": "north_south",
        "utility": "awarded_point_margin", "successThreshold": 0, "budgetProfile": "late_game_exact_v1",
    })
    compiler = OrToolsCompleteStateGraphCompilerV1(
        play_domain=rules.play_domain, scoring_domain=ScoringDomainIRV1.from_rules(rules.data),
        structural_domain=rules.structural_domain, project_domain=rules.project_domain, native_conversion=True,
    )
    for depth, reason in ((12, "maximum_depth"), (16, "maximum_nodes")):
        result = compiler.compile(reached.seed, request, expected_rules_hash=rules.sha256,
            maximum_nodes=1, maximum_depth=depth, maximum_seconds=30)
        assert not result.complete and result.receipt.exhaustion_reason == reason
    lifted = OrToolsTailFirstBackwardLiftV1(rules=rules, native_cp_sat_scoring=True).lift(
        scenario, plan, tail, reached.seed, maximum_seconds=60,
    )
    assert lifted.status == "feasible", lifted.reason
    plays = lifted.candidate.trace["play"]["plays"]
    assert len(plays) == 32
    assert len({item["card"] for item in plays}) == 32
    assert [(item["actor"], item["card"]) for item in plays[16:]] == [
        (item.actor.name.lower(), item.card.code) for item in tail.plays
    ]


@pytest.mark.parametrize("second_context", ["non_sweep", "sweep", "hokum", "cache_miss"])
def test_cached_raw_tie_skips_only_owned_non_sweep_sun_proof(
    rules, tail_lift_fixture, monkeypatch, second_context,
):
    import baloot_platform.research.ortools_tail_first_candidate as candidate_module

    _, tail, seed = tail_lift_fixture
    target = seed.next_actor.team
    seed = _seed_clone(seed, base_tricks=(3, 3))
    scenario = ResearchScenarioV1.build({
        "schemaVersion": 1, "origin": {"kind": "search"},
        "constraints": {"op": "all", "children": [
            _leaf("PRED-SUN", "auction.contract", {"contract": "sun"}),
            _leaf("PRED-BUYER", "auction.buyer", {"seat": "north"}),
            _leaf("PRED-ACTION", "play.action_matches", {
                "trickIndex": 6, "position": 0, "actingSeat": seed.next_actor.name.lower(),
                "selector": {"cards": [tail.plays[0].card.id]},
            }),
            _leaf("PRED-UNIQUE", "proof.winning_root_action_count", {"minimum": 1, "maximum": 1}),
        ]},
        "decisionPoint": {"kind": "play", "selector": {
            "trickIndex": 6, "position": 0, "remainingCards": 2,
            "actingSeat": seed.next_actor.name.lower(),
        }},
        "outcome": _leaf("PRED-REPORT", "outcome.team_round", {
            "team": target.name.lower(), "status": "won",
        }, "report"),
    })
    request = ScenarioProofRequestV1.build({
        "schemaVersion": 1, "mode": "exact", "visibility": "complete_state",
        "claim": "winning_root_action_count", "targetTeam": target.name.lower(),
        "utility": "awarded_point_margin", "successThreshold": 1, "budgetProfile": "tail_first_test",
    })
    second = _seed_clone(seed, awarded_project_points=(
        seed.awarded_project_points[0] + 20, seed.awarded_project_points[1],
    ))
    if second_context == "sweep":
        second = _seed_clone(second, base_tricks=(6, 0))
    elif second_context == "hokum":
        second = _seed_clone(second, contract=Contract.HOKUM, trump=Suit.CLUBS)
    elif second_context == "cache_miss":
        second = _seed_clone(second, buyer=seed.buyer.partner)
    utilities, boundaries = [], []
    class Screened(Exception):
        pass
    def boundary(*args, **kwargs):
        index = len(boundaries)
        if index == 2:
            expected = ["awarded_point_margin", candidate_module.INTERNAL_RAW_POINT_MARGIN_UTILITY_V1]
            if second_context != "non_sweep":
                expected.append("awarded_point_margin")
                if second_context == "cache_miss":
                    expected.append(candidate_module.INTERNAL_RAW_POINT_MARGIN_UTILITY_V1)
            assert utilities == expected
            if second_context == "non_sweep":
                exclusions = kwargs["excluded_proof_states"]
                assert boundaries[1].proof_state_no_good in exclusions
                same_project = [item for item in exclusions if item.awarded_project_points == second.awarded_project_points]
                assert {item.base_tricks_ns for item in same_project} == {1, 2, 3, 4, 5}
                assert {item.base_raw_ns for item in same_project} == set(range(sum(seed.base_raw) + 1))
            raise Screened
        current = seed if index == 0 else second
        no_good = SimpleNamespace(decision_state_hash="0" * 64,
            boundary_combination=("east", "sun", "north", None),
            base_raw_ns=current.base_raw[0], base_tricks_ns=current.base_tricks[0],
            awarded_project_points=current.awarded_project_points, baloot_teams=(False, False),
            escalation_semantics=("normal", False, None),
            proof_state_projection_hash=current.proof_state_projection_hash)
        value = SimpleNamespace(status="feasible", seed=current, no_good=no_good,
            proof_state_no_good=no_good, model_build_seconds=0, solver_seconds=0,
            solve_attempts=1, reason=None)
        boundaries.append(value)
        return value
    def compile_graph(_compiler, _seed, proof_request, **kwargs):
        utilities.append(proof_request.utility)
        return SimpleNamespace(complete=True, graph=SimpleNamespace(sha256="mock-graph"),
            receipt=SimpleNamespace(to_dict=lambda: {"status": "complete"}))
    # Counterproof fixtures exercise control flow, not production certification.
    def tied_proof(*args, **kwargs):
        return SimpleNamespace(completion_status="exact_complete", claim_satisfied=False,
            winning_root_action_count=2, root_actions=(
                SimpleNamespace(action_code="CA", value=0), SimpleNamespace(action_code="CK", value=0)))
    original_solve = OrToolsPlayTailStageV1.solve
    def solve_attached(stage, **kwargs):
        original_solve(stage, **kwargs)
        return tail
    monkeypatch.setattr(OrToolsPlayTailStageV1, "solve", solve_attached)
    monkeypatch.setattr(OrToolsTailProofBoundaryV1, "solve", boundary)
    monkeypatch.setattr(candidate_module.OrToolsCompleteStateGraphCompilerV1, "compile", compile_graph)
    monkeypatch.setattr(candidate_module.OrToolsScenarioGraphProofV1, "prove", tied_proof)
    compiler = candidate_module.OrToolsTailFirstCandidateCompilerV1(
        rules=rules, plan=TailFirstStagePlannerV1(native_cp_sat=True).plan(scenario), proof_request=request,
        maximum_proof_nodes=1000, maximum_proof_depth=8, maximum_line_count_limbs=64,
        maximum_conflicts=250000, maximum_prefix_lift_attempts=1, native_cp_sat_scoring=True,
    )
    with pytest.raises(Screened):
        compiler(scenario, (), seed=1, maximum_seconds=30, maximum_conflicts=250000)


@pytest.fixture(scope="session")
def tail_lift_fixture(root: Path, rules):
    exact = json.loads(
        (root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(
            encoding="utf-8"
        )
    )["exactTracePrefix"]
    deck = tuple(Card(card_id) for card_id in exact["deckOrder"])
    layout = DealLayoutV1.build(
        rules.structural_domain,
        dealer=Seat.EAST,
        buyer=Seat.NORTH,
    )
    hands = {
        seat: tuple(
            deck[position] for position in layout.positions_for(seat, final=True)
        )
        for seat in Seat
    }
    continuation = tuple(
        TailPrefixActionV1.build(actor, card)
        for actor, card in zip(
            (
                "west",
                "east",
                "north",
                "west",
                "south",
                "east",
                "north",
                "west",
                "south",
            ),
            ("CA", "C8", "C7", "D7", "S7", "C9", "DA", "D8", "SA"),
            strict=True,
        )
    )
    plays = (
        tuple(
            TailPrefixActionV1.build(
                action["actor"], action["card"], ekka=action.get("ekka", False)
            )
            for action in exact["playActions"]
        )
        + continuation
    )
    assert len(plays) == 32
    ownership = {card: seat for seat, hand in hands.items() for card in hand}
    full = OrToolsPlayTailStageV1().solve(
        cards=FULL_DECK,
        ownership=ownership,
        contract=Contract.SUN,
        trump=None,
        first_leader=Seat.NORTH,
        play_domain=rules.play_domain,
        structural_domain=rules.structural_domain,
        scoring_domain=ScoringDomainIRV1.from_rules(rules.data),
        tail=plays[24:],
        maximum_time_seconds=10,
    )
    projected = full.at_prefix(24)
    tail = OrToolsPlayTailStageV1().solve(
        cards=tuple(card for hand in projected.remaining_hands for card in hand),
        ownership={
            card: seat for seat in Seat for card in projected.remaining_hands[int(seat)]
        },
        contract=Contract.SUN,
        trump=None,
        first_leader=projected.next_actor,
        play_domain=rules.play_domain,
        structural_domain=rules.structural_domain,
        tail=full.plays[24:],
        maximum_time_seconds=10,
    )
    projects = OrToolsProjectLifecycleModelV1().solve(
        hands=hands,
        contract=Contract.SUN,
        dealer=Seat.EAST,
        trump=None,
        domain=rules.project_domain,
        timeout_seconds=10,
    )
    raw = [0, 0]
    tricks = [0, 0]
    for winner, points in zip(
        full.trick_winners[:6], full.trick_raw_points[:6], strict=True
    ):
        raw[int(winner.team)] += points
        tricks[int(winner.team)] += 1
    boundary = CompleteStateGraphSeedV1.build(
        exact_state_id="STATE-TAIL-FIRST-LIFT",
        rules_hash=rules.sha256,
        remaining_hands=projected.remaining_hands,
        current_trick=(),
        first_leader=projected.next_actor,
        next_actor=projected.next_actor,
        legal_actions=projected.legal_play_actions,
        contract=Contract.SUN,
        trump=None,
        buyer=Seat.NORTH,
        locked=False,
        base_raw=(raw[0], raw[1]),
        base_tricks=(tricks[0], tricks[1]),
        awarded_project_points=projects.combined_awarded_value,
        baloot_points=projects.baloot_points,
        escalation="normal",
        last_escalating_team=None,
        starting_match_score=(0, 0),
    )
    return hands, tail, boundary


@pytest.mark.parametrize("match_score,width,composed_dealer,require_double", [
    ((0, 0), 0, False, False), ((101, 30), 0, False, False), ((101, 30), 2, False, False),
    ((200, 200), 0, False, False), ((101, 30), 2, True, False), ((100, 30), 1, False, True),
    (None, 0, False, True),
])
def test_native_tail_candidate_preserves_required_starting_score(
    rules, tail_lift_fixture, monkeypatch, match_score, width, composed_dealer, require_double,
):
    from baloot_platform.research.ortools_tail_first_candidate import OrToolsTailFirstCandidateCompilerV1

    hands, _tail, boundary = tail_lift_fixture
    initial_scores = () if match_score is None else tuple(
        _leaf(f"PRED-INITIAL-SCORE-{team.upper().replace('_', '-')}", "state.match_score", {
            "team": team, "minimum": score, "maximum": score + width,
        }) for team, score in zip(("north_south", "west_east"), match_score, strict=True)
    )
    scenario = _scenario(hands, boundary, *initial_scores)
    if require_double:
        raw = scenario.to_dict()
        raw.pop("sha256")
        raw["constraints"] = {"op": "all", "children": [raw["constraints"],
            _leaf("PRED-REQUIRED-DOUBLE", "escalation.level", {"level": "double"})]}
        scenario = ResearchScenarioV1.build(raw)
    if composed_dealer:
        raw = scenario.to_dict()
        raw.pop("sha256")
        raw["constraints"] = {"op": "all", "children": [raw["constraints"], {"op": "any", "children": [
            _leaf(f"PRED-DEALER-{seat.upper()}", "auction.dealer", {"seat": seat})
            for seat in ("north", "east")]}]}
        scenario = ResearchScenarioV1.build(raw)
    plan = TailFirstStagePlannerV1(native_cp_sat=True).plan(scenario)
    assert plan.planner == "tail_first"
    original_boundary = OrToolsTailProofBoundaryV1.solve
    selected_scores = []

    def inspect_boundary(stage, *args, **kwargs):
        if composed_dealer:
            assert kwargs["boundary_combination"][0] == "east"
            assert stage.starting_match_score[0] == 101
        if match_score is not None:
            assert all(lower <= score <= lower + width
                for lower, score in zip(match_score, stage.starting_match_score, strict=True))
        selected_scores.append(stage.starting_match_score)
        result = original_boundary(stage, *args, **kwargs)
        if require_double and stage.starting_match_score[0] == 100:
            assert result.status == "infeasible"
            return result
        assert result.status == "feasible", result.reason
        assert result.seed.starting_match_score == stage.starting_match_score
        return result

    monkeypatch.setattr(OrToolsTailProofBoundaryV1, "solve", inspect_boundary)
    request = ScenarioProofRequestV1.build({
        "schemaVersion": 1, "mode": "exact", "visibility": "complete_state",
        "claim": "outcome_reachable", "targetTeam": "north_south",
        "utility": "awarded_point_margin", "successThreshold": -100000,
        "budgetProfile": "late_game_exact_v1",
    })
    compiler = OrToolsTailFirstCandidateCompilerV1(rules=rules, plan=plan,
        proof_request=request, maximum_proof_nodes=100_000, maximum_proof_depth=8,
        maximum_line_count_limbs=64, maximum_conflicts=250_000,
        maximum_prefix_lift_attempts=1, native_cp_sat_scoring=True)
    attempt = compiler(scenario, (), seed=0, maximum_seconds=30, maximum_conflicts=250_000)
    assert attempt.status == "feasible", attempt.reason
    assert attempt.candidate.trace["matchScore"] == list(selected_scores[-1])
    assert len(attempt.candidate.trace["play"]["plays"]) == 32
    if composed_dealer:
        assert attempt.candidate.trace["dealer"] == "east"
    if require_double:
        if match_score is None:
            assert sorted(attempt.candidate.trace["matchScore"]) == [0, 101]
        else:
            assert attempt.candidate.trace["matchScore"][0] == 101
        assert attempt.candidate.trace["escalation"]["observation"]["level"] == "double"
        assert attempt.proof_outcome.completion_status == "exact_complete"
        assert attempt.proof_outcome.claim_satisfied is True


@pytest.fixture(scope="session")
def ratified_tie_lift_fixture(root: Path):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v8.yaml")
    hands = {
        seat: tuple(Card.parse(card) for card in cards.split())
        for seat, cards in (
            (Seat.NORTH, "S7 S8 S9 D7 D9 DJ DK C7"),
            (Seat.WEST, "H7 H8 H9 CA C9 D8 D10 DQ"),
            (Seat.SOUTH, "S10 SQ SA H10 HQ HA C10 CQ"),
            (Seat.EAST, "SJ SK HJ HK DA C8 CJ CK"),
        )
    }
    plays = tuple(
        TailPrefixActionV1.build(actor, card)
        for trick in (
            "north:C7 west:CA south:C10 east:C8",
            "west:H7 south:HA east:HJ north:D7",
            "south:SA east:SK north:S7 west:D8",
            "south:SQ east:SJ north:S8 west:D10",
            "south:S10 east:CK north:S9 west:DQ",
            "south:HQ east:HK north:DJ west:H8",
            "east:DA north:DK west:C9 south:CQ",
            "east:CJ north:D9 west:H9 south:H10",
        )
        for actor, card in (action.split(":") for action in trick.split())
    )
    full = OrToolsPlayTailStageV1(native_conversion=True).solve(
        cards=FULL_DECK,
        ownership={card: seat for seat, hand in hands.items() for card in hand},
        contract=Contract.SUN, trump=None, first_leader=Seat.NORTH,
        play_domain=rules.play_domain, structural_domain=rules.structural_domain,
        scoring_domain=ScoringDomainIRV1.from_rules(rules.data),
        prefix=plays, decision_prefix_length=24, maximum_time_seconds=10,
    )
    assert full.trick_winners[0] == Seat.WEST
    assert full.plays[0].actor == Seat.NORTH
    projects = OrToolsProjectLifecycleModelV1().solve(
        hands=hands, contract=Contract.SUN, dealer=Seat.EAST, trump=None,
        domain=rules.project_domain, second_trick_leader=full.trick_winners[0],
        timeout_seconds=10,
    )
    declared = tuple(project for project in projects.projects if project.declared)
    assert len(declared) == 2
    assert len({(project.kind, project.high_rank) for project in declared}) == 1
    assert projects.winning_team == Seat.WEST.team
    projected = full.at_prefix(24)
    tail = OrToolsPlayTailStageV1(native_conversion=True).solve(
        cards=tuple(card for hand in projected.remaining_hands for card in hand),
        ownership={card: seat for seat in Seat
                   for card in projected.remaining_hands[int(seat)]},
        contract=Contract.SUN, trump=None, first_leader=projected.next_actor,
        play_domain=rules.play_domain, structural_domain=rules.structural_domain,
        tail=full.plays[24:], maximum_time_seconds=10,
    )
    raw = [0, 0]
    tricks = [0, 0]
    for winner, points in zip(full.trick_winners[:6], full.trick_raw_points[:6], strict=True):
        raw[int(winner.team)] += points
        tricks[int(winner.team)] += 1
    boundary = CompleteStateGraphSeedV1.build(
        exact_state_id="STATE-RATIFIED-TIE-LIFT", rules_hash=rules.sha256,
        remaining_hands=projected.remaining_hands, current_trick=(),
        first_leader=projected.next_actor, next_actor=projected.next_actor,
        legal_actions=projected.legal_play_actions, contract=Contract.SUN, trump=None,
        buyer=Seat.NORTH, locked=False, base_raw=tuple(raw), base_tricks=tuple(tricks),
        awarded_project_points=projects.combined_awarded_value,
        baloot_points=projects.baloot_points, escalation="normal",
        last_escalating_team=None, starting_match_score=(0, 0),
    )
    return rules, hands, full, tail, boundary, projects


def _scenario(hands, boundary, *extra):
    deal = tuple(
        _leaf(
            f"PRED-OWNER-{card.id:02d}",
            "deal.card_owner",
            {
                "card": card.id,
                "seat": seat.name.lower(),
                "handStage": "post_purchase",
            },
        )
        for seat, hand in hands.items()
        for card in hand
    )
    constraints = {
        "op": "all",
        "children": [
            _leaf("PRED-CONTRACT", "auction.contract", {"contract": "sun"}),
            _leaf("PRED-BUYER", "auction.buyer", {"seat": "north"}),
            _leaf(
                "PRED-TRICK",
                "play.trick_index",
                {"minimum": 6, "maximum": 6},
            ),
            *deal,
            *extra,
        ],
    }
    return ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": constraints,
            "decisionPoint": {
                "kind": "play",
                "selector": {
                "trickIndex": 6,
                "remainingCards": 2,
                "position": 0,
                "actingSeat": boundary.next_actor.name.lower(),
                },
            },
            "outcome": _leaf(
                "PRED-REPORT-ROUND",
                "outcome.team_round",
                {"team": "north_south", "status": "won"},
                "report",
            ),
        }
    )


@pytest.mark.parametrize("operator", ["all", "not", "any", "count", "contradiction"])
def test_native_tail_ownership_shortcut_respects_expression(
    ratified_tie_lift_fixture, monkeypatch, operator,
):
    from baloot_platform.research.ortools_tail_first_candidate import OrToolsTailFirstCandidateCompilerV1

    rules, hands, full, tail, boundary, _projects = ratified_tie_lift_fixture
    payload = _scenario(hands, boundary).to_dict()
    payload.pop("sha256")
    children = payload["constraints"]["children"]
    payload["constraints"].pop("sha256")
    index = next(index for index, row in enumerate(children)
                 if row.get("kind") == "deal.card_owner")
    positive = children[index]
    wrong = _leaf("PRED-OPTIONAL-WRONG-OWNER", "deal.card_owner", {
        **positive["arguments"], "seat": Seat.parse(positive["arguments"]["seat"]).partner.name.lower(),
    })
    if operator != "all":
        children[index] = {
            "op": "all" if operator == "contradiction" else operator,
            "children": [wrong] if operator == "not" else [positive, wrong],
            **({"minimum": 1, "maximum": 1} if operator == "count" else {}),
        }
    scenario = ResearchScenarioV1.build(payload)
    original_solve = OrToolsPlayTailStageV1.solve

    def fixed_witness(stage, **kwargs):
        # Fix play symmetry, not ownership or the expression under test.
        if kwargs.get("play_step_count") == 32:
            kwargs["prefix"] = full.plays
        else:
            kwargs["tail"] = tail.plays
        return original_solve(stage, **kwargs)

    class ReachedBoundary(Exception):
        pass

    def reached_boundary(*_args, **_kwargs):
        raise ReachedBoundary

    monkeypatch.setattr(OrToolsPlayTailStageV1, "solve", fixed_witness)
    monkeypatch.setattr(OrToolsTailProofBoundaryV1, "solve", reached_boundary)
    request = ScenarioProofRequestV1.build({
        "schemaVersion": 1, "mode": "exact", "visibility": "complete_state",
        "claim": "winning_root_action_count", "targetTeam": "north_south",
        "utility": "round_win", "successThreshold": 1, "budgetProfile": "tail_first_test",
    })
    compiler = OrToolsTailFirstCandidateCompilerV1(
        rules=rules, plan=TailFirstStagePlannerV1(native_cp_sat=True).plan(scenario),
        proof_request=request, maximum_proof_nodes=1000, maximum_proof_depth=8,
        maximum_line_count_limbs=64, maximum_conflicts=250_000,
        maximum_prefix_lift_attempts=1, native_cp_sat_scoring=True,
    )
    if operator == "contradiction":
        attempt = compiler(scenario, (), seed=0, maximum_seconds=15, maximum_conflicts=250_000)
        # Initial-input selection is partial, so no witness is not global impossibility.
        assert attempt.status == "not_found" and attempt.reason == "initial_input_choice_no_witness"
        assert attempt.candidate is None
    else:
        with pytest.raises(ReachedBoundary):
            compiler(scenario, (), seed=0, maximum_seconds=15, maximum_conflicts=250_000)


def _pin_generic_native_opening(monkeypatch, opening, *, rules, hands, scenario, completion=(), pin_deck=True):
    # Fix deck symmetry to test opening semantics, not broad-search throughput.
    original_build = OrToolsPlayTailStageV1.build_model
    original_solve = OrToolsPlayTailStageV1.solve
    original_deal = research_executor.OrToolsScenarioModelV1.solve
    openings, completions = [], []
    buyers = {Seat.parse(predicate.arguments["seat"]) for predicate in scenario.predicates()
              if predicate.hardness == "hard" and predicate.kind == "auction.buyer"}
    assert len(buyers) == 1
    layout = DealLayoutV1.build(rules.structural_domain, dealer=Seat.EAST, buyer=buyers.pop())
    deck = [None] * len(FULL_DECK)
    for seat in Seat:
        for position, card in zip(layout.positions_for(seat, final=True),
                                  sorted(hands[seat], key=lambda card: card.id), strict=True):
            deck[position] = card.id
    assert len(deck) == len(set(deck)) == 32 and sorted(deck) == list(range(32))
    deck = tuple(deck)

    def solve_fixed_deck(model, **kwargs):
        assert kwargs["solver"].parameters.cp_model_probing_level == 0
        if pin_deck:
            for position, card_id in enumerate(deck):
                model.model.add(model.card_at_position[position] == card_id)
        return original_deal(model, **kwargs)

    def build_pinned(stage, **kwargs):
        build = original_build(stage, **kwargs)
        if kwargs.get("shared_model") is not None:
            assert tuple(kwargs["cards"]) == FULL_DECK
            assert len(build.selected_play_indexes) == 4
            openings.append(kwargs)
            actions, offset = opening, 0
        else:
            actions, offset = completion[4:], 4
        for index, action in enumerate(actions, start=offset):
            build.model.add(build.selected_play_indexes[index] == action.card.id)
            if index % 4 == 0:
                build.model.add(build.ekka_lead_variables[index // 4] == int(action.ekka))
        return build

    def solve_captured(stage, **kwargs):
        if tuple(kwargs["cards"]) == FULL_DECK:
            assert tuple(kwargs.get("prefix", ())) == tuple(opening)
            completions.append(kwargs)
        return original_solve(stage, **kwargs)

    monkeypatch.setattr(OrToolsPlayTailStageV1, "build_model", build_pinned)
    monkeypatch.setattr(OrToolsPlayTailStageV1, "solve", solve_captured)
    monkeypatch.setattr(research_executor.OrToolsScenarioModelV1, "solve", solve_fixed_deck)
    return openings, completions, deck


@pytest.mark.parametrize("contradictory_awards", [False, True])
@pytest.mark.parametrize("pin_deck", [False, True], ids=["unpinned-deck", "fixed-deck"])
def test_ratified_generic_structural_awards_follow_its_native_opening(
    ratified_tie_lift_fixture, monkeypatch, contradictory_awards, pin_deck,
):
    rules, hands, full, _tail, boundary, projects = ratified_tie_lift_fixture
    owner = Seat.NORTH if contradictory_awards else Seat.WEST
    project = next(row for row in projects.projects if row.owner == owner and row.declared)
    scenario = _scenario(
        hands, boundary,
        _leaf("PRED-NORMAL", "escalation.level", {"level": "normal"}),
        _leaf("PRED-AWARD", "project.exists", {
            "owner": owner.name.lower(), "team": owner.team.name.lower(),
            "kind": project.kind, "value": project.value, "state": "awarded",
        }),
    )
    openings, completions, deck = _pin_generic_native_opening(
        monkeypatch, full.plays[:4], rules=rules, hands=hands, scenario=scenario,
        pin_deck=pin_deck,
    )
    attempt = GenericCompleteTraceCompilerV1(
        rules=rules, structural_only=True, native_cp_sat_scoring=True,
        manifest=research_executor.shared_scenario_predicate_catalog_lowering_manifest_v3(),
    )(scenario, (), seed=0, maximum_seconds=15)

    assert len(openings) == 1 and not completions
    assert attempt.status == ("infeasible" if contradictory_awards else "feasible"), attempt.reason
    assert (attempt.candidate is None) is contradictory_awards
    if pin_deck and attempt.candidate is not None:
        assert attempt.candidate.trace["deckOrder"] == list(deck)


@pytest.mark.parametrize("require_project", [False, True])
def test_ratified_generic_full_trace_retains_live_opening_awards_and_solver(
    ratified_tie_lift_fixture, monkeypatch, require_project,
):
    rules, hands, full, _tail, boundary, projects = ratified_tie_lift_fixture
    project = next(row for row in projects.projects if row.owner == Seat.WEST and row.awarded)
    predicates = [_leaf("PRED-NORMAL", "escalation.level", {"level": "normal"})]
    if require_project:
        predicates.append(_leaf("PRED-AWARD", "project.exists", {
            "owner": "west", "team": "west_east", "kind": project.kind,
            "value": project.value, "state": "awarded",
        }))
    scenario = _scenario(hands, boundary, *predicates)
    openings, completions, deck = _pin_generic_native_opening(
        monkeypatch, full.plays[:4], rules=rules, hands=hands, scenario=scenario,
        completion=full.plays,
    )
    solvers, construction = {}, {}
    original_deal = research_executor.OrToolsScenarioModelV1.solve
    original_opening = play_tail_module._PlayTailExtractionV1.__call__
    original_projects = research_executor.OrToolsProjectLifecycleFragmentV1.extract
    original_escalation = research_executor.CpSatEscalationPredicateContextV1.extract_stage_result

    def solve_deal(model, **kwargs):
        solvers["deal"] = kwargs.get("solver")
        assert isinstance(solvers["deal"], cp_model.CpSolver)
        result = original_deal(model, **kwargs)
        construction["deal"] = result
        return result

    def extract_opening(extract, solver, *args):
        if len(extract.card_ids) == 4:
            solvers["opening"] = solver
        return original_opening(extract, solver, *args)

    def extract_projects(fragment, solver):
        if fragment.combined_awarded_value[0].name.startswith("construction_projects_"):
            solvers["projects"] = solver
        return original_projects(fragment, solver)

    def extract_escalation(context, solver, **kwargs):
        if context.context_id == "construction_escalation":
            solvers["escalation"] = solver
        return original_escalation(context, solver, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("ratified native complete trace used standalone project fallback")

    monkeypatch.setattr(research_executor.OrToolsScenarioModelV1, "solve", solve_deal)
    monkeypatch.setattr(play_tail_module._PlayTailExtractionV1, "__call__", extract_opening)
    monkeypatch.setattr(research_executor.OrToolsProjectLifecycleFragmentV1, "extract", extract_projects)
    monkeypatch.setattr(research_executor.CpSatEscalationPredicateContextV1, "extract_stage_result", extract_escalation)
    monkeypatch.setattr(OrToolsProjectLifecycleModelV1, "solve", forbidden)
    attempt = GenericCompleteTraceCompilerV1(
        rules=rules, native_cp_sat_scoring=True,
    )(scenario, (), seed=0, maximum_seconds=30)

    assert attempt.status == "feasible", attempt.reason
    assert len(openings) == len(completions) == 1
    assert set(solvers) == {"deal", "opening", "projects", "escalation"}
    assert all(solver is solvers["deal"] for solver in solvers.values())
    trace = attempt.candidate.trace
    assert trace["deckOrder"] == list(deck)
    assert trace["play"]["plays"][:4] == [action.to_dict() for action in full.plays[:4]]
    assert trace["play"]["tricks"][0]["winner"] == "west"
    assert trace["projects"]["winningTeam"] == "west_east"
    assert trace["projects"]["combinedAwardedValue"] == list(projects.combined_awarded_value)
    assert trace["score"]["projectPoints"] == trace["projects"]["combinedAwardedValue"]
    assert trace["projects"]["modelHash"] == construction["deal"].model_hash
    assert trace["projects"]["modelHash"] == attempt.candidate.stage_receipts["deal"].model_hash
    assert trace["projects"]["variableCount"] == construction["deal"].variable_count
    assert trace["projects"]["constraintCount"] == construction["deal"].constraint_count
    declarations = trace["projects"]["declarationActions"]
    assert [row["actor"] for row in declarations] == [
        seat.name.lower() for seat in rules.project_domain.declaration_seats(Seat.EAST)
    ]
    for action in declarations:
        declared_ids = {row["projectId"] for row in trace["projects"]["projects"]
                        if row["owner"] == action["actor"] and row["declared"]}
        if declared_ids:
            assert action["action"] == "declare" and action["projectId"] in declared_ids
        else:
            assert action == {"actor": action["actor"], "action": "pass"}
    project_payload = scenario.to_dict()
    project_payload.pop("sha256")
    project_payload["decisionPoint"] = {
        "kind": "project", "selector": {"owner": "west", "projectIndex": 0},
    }
    prefix = research_executor._search_decision_prefix(
        attempt.candidate, ResearchScenarioV1.build(project_payload), native_cp_sat=True,
    ).to_dict()
    assert prefix["projectActions"] == declarations[:1]
    assert prefix["playActions"] == [full.plays[0].to_dict()]
    assert trace["escalation"]["observation"]["terminal"] is True
    assert completions[0]["locked"] is False
    if require_project:
        from baloot_platform.core.exact_state_view import build_canonical_observer_information_v1
        from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1

        monkeypatch.undo()
        exact = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
            prefix, decision_point=project_payload["decisionPoint"], maximum_seconds=15,
        )
        exact.require_integrity()
        assert exact.decision_kind == "project" and exact.next_actor == "west"
        assert [row["code"] for row in exact.legal_actions] == [f"declare:{project.project_id}"]
        public = build_canonical_observer_information_v1(
            complete_state=exact.state, acting_seat="west",
        )["publicState"]
        assert public["currentPlayer"] == "west"
        assert public["completedTricks"] == [] and public["projectWinningTeam"] is None
        assert public["roundScore"] is None
        assert not any(row["scoring"] or row["revealed"] for row in public["projects"])
        assert len(public["hands"]["west"]) == 8
        for seat, count in (("north", 7), ("south", 8), ("east", 8)):
            assert public["hands"][seat] == {"count": count}
        north_project = next(row for row in public["projects"] if row["owner"] == "north")
        assert north_project["cards"] is None and north_project["highRank"] is None


def test_ratified_generic_eligible_only_structural_has_no_play_model(
    ratified_tie_lift_fixture, monkeypatch,
):
    rules, hands, _full, _tail, boundary, projects = ratified_tie_lift_fixture
    project = next(row for row in projects.projects if row.owner == Seat.NORTH and row.declared)
    scenario = _scenario(hands, boundary, _leaf("PRED-ELIGIBLE", "project.exists", {
        "owner": "north", "team": "north_south", "kind": project.kind,
        "value": project.value, "state": "eligible",
    }))

    def forbidden(*args, **kwargs):
        pytest.fail("eligibility-only structural construction built a play model")

    monkeypatch.setattr(OrToolsPlayTailStageV1, "build_model", forbidden)
    attempt = GenericCompleteTraceCompilerV1(
        rules=rules, structural_only=True, native_cp_sat_scoring=True,
        manifest=research_executor.shared_scenario_predicate_catalog_lowering_manifest_v3(),
    )(scenario, (), seed=0, maximum_seconds=15)
    assert attempt.status == "feasible", attempt.reason
    assert "play" not in attempt.candidate.trace
    assert set(attempt.candidate.stage_receipts) == {"structural"}


def test_ratified_generic_buyer_first_uses_the_actual_non_dealer_right_buyer(
    ratified_tie_lift_fixture, monkeypatch,
):
    rules, hands, full, _tail, boundary, _projects = ratified_tie_lift_fixture
    data = copy.deepcopy(rules.data)
    data["play"]["first_leader"] = "buyer"
    rules = RuleBook(data)
    payload = _scenario(hands, boundary).to_dict()
    payload.pop("sha256")
    payload["constraints"].pop("sha256")
    buyer_predicate = next(row for row in payload["constraints"]["children"]
                           if row.get("kind") == "auction.buyer")
    buyer_predicate["arguments"] = {"seat": "west"}
    buyer_predicate.pop("sha256")
    payload["constraints"]["children"] = [row for row in payload["constraints"]["children"]
                                           if row.get("kind") != "play.trick_index"]
    payload["decisionPoint"]["selector"] = {"trickIndex": 0, "position": 0, "actingSeat": "west"}
    scenario = ResearchScenarioV1.build(payload)
    opening = (*full.plays[1:4], full.plays[0])
    openings, completions, deck = _pin_generic_native_opening(
        monkeypatch, opening, rules=rules, hands=hands, scenario=scenario,
    )
    attempt = GenericCompleteTraceCompilerV1(
        rules=rules, native_cp_sat_scoring=True,
    )(scenario, (), seed=0, maximum_seconds=30)

    assert attempt.status == "feasible", attempt.reason
    assert any(row["first_leader"] == Seat.WEST for row in openings)
    assert len(completions) == 1 and completions[0]["first_leader"] == Seat.WEST
    trace = attempt.candidate.trace
    assert trace["deckOrder"] == list(deck)
    assert trace["buyer"] == "west"
    assert rules.structural_domain.dealer_right(Seat.EAST) == Seat.NORTH
    assert trace["play"]["plays"][:4] == [action.to_dict() for action in opening]


@pytest.mark.parametrize(("locked", "trump_lead"), [
    pytest.param(False, True, id="open-trump-lead"),
    pytest.param(True, True, id="locked-illegal-trump-lead"),
    pytest.param(True, False, id="locked-legal-nontrump-lead"),
])
def test_ratified_generic_hokum_opening_uses_the_actual_escalation_lock(
    ratified_tie_lift_fixture, monkeypatch, locked, trump_lead,
):
    rules, hands, full, _tail, boundary, _projects = ratified_tie_lift_fixture
    opening = full.plays[:4] if trump_lead else tuple(
        TailPrefixActionV1.build(actor, card)
        for actor, card in (("north", "S7"), ("west", "CA"), ("south", "S10"), ("east", "SJ"))
    )
    assert (opening[0].card.suit == Suit.CLUBS) is trump_lead
    assert any(card.suit != Suit.CLUBS for card in hands[Seat.NORTH])
    payload = _scenario(
        hands, boundary,
        _leaf("PRED-TRUMP", "auction.trump", {"suit": "clubs"}),
        _leaf("PRED-DOUBLE", "escalation.level", {"level": "double"}),
        _leaf("PRED-LOCKED", "escalation.locked", {"locked": locked}),
    ).to_dict()
    payload.pop("sha256")
    payload["constraints"].pop("sha256")
    contract_predicate = next(row for row in payload["constraints"]["children"]
                              if row.get("kind") == "auction.contract")
    contract_predicate["arguments"] = {"contract": "hokum"}
    contract_predicate.pop("sha256")
    payload["constraints"]["children"] = [row for row in payload["constraints"]["children"]
                                           if row.get("kind") != "play.trick_index"]
    payload["decisionPoint"]["selector"] = {"trickIndex": 0, "position": 0, "actingSeat": "north"}
    scenario = ResearchScenarioV1.build(payload)
    openings, completions, deck = _pin_generic_native_opening(
        monkeypatch, opening, rules=rules, hands=hands, scenario=scenario,
    )
    attempt = GenericCompleteTraceCompilerV1(
        rules=rules, native_cp_sat_scoring=True,
    )(scenario, (), seed=0, maximum_seconds=30)

    assert openings
    illegal_opening = locked and trump_lead
    assert attempt.status == ("infeasible" if illegal_opening else "feasible"), attempt.reason
    if illegal_opening:
        assert attempt.candidate is None and not completions
    else:
        trace = attempt.candidate.trace
        assert trace["deckOrder"] == list(deck)
        assert trace["escalation"]["observation"]["level"] == "double"
        assert trace["escalation"]["observation"]["mode"] == ("locked" if locked else "open")
        assert trace["play"]["plays"][:4] == [action.to_dict() for action in opening]
        assert len(completions) == 1 and completions[0]["locked"] is locked


@pytest.mark.parametrize("reuse_hint", [False, True])
@pytest.mark.parametrize("contradictory_awards", [False, True])
def test_ratified_lift_binds_awards_to_the_actual_first_trick(
    ratified_tie_lift_fixture, reuse_hint, contradictory_awards,
):
    rules, hands, full, tail, boundary, projects = ratified_tie_lift_fixture
    award_owner = Seat.NORTH if contradictory_awards else Seat.WEST
    project = next(row for row in projects.projects if row.owner == award_owner and row.declared)
    if contradictory_awards:
        boundary = _seed_clone(boundary, awarded_project_points=tuple(
            reversed(boundary.awarded_project_points)
        ))
    scenario = _scenario(
        hands, boundary,
        _leaf("PRED-DEALER", "auction.dealer", {"seat": "east"}),
        _leaf("PRED-AWARD", "project.exists", {
            "owner": award_owner.name.lower(), "team": award_owner.team.name.lower(),
            "kind": project.kind, "value": project.value, "state": "awarded",
        }),
        *(_leaf(f"PRED-OPENING-{index:02d}", "play.action_matches", {
            "trickIndex": index // 4, "position": index % 4,
            "actingSeat": action.actor.name.lower(), "selector": {"cards": [action.card.id]},
        }) for index, action in enumerate(full.plays[:24])),
    )
    attempt = OrToolsTailFirstBackwardLiftV1(rules=rules, native_cp_sat_scoring=True).lift(
        scenario, TailFirstStagePlannerV1(native_cp_sat=True).plan(scenario), tail, boundary,
        play_hint=full if reuse_hint else None, maximum_seconds=60,
    )
    if contradictory_awards:
        assert attempt.status == "infeasible", attempt.reason
        assert attempt.candidate is None
    else:
        assert attempt.status == "feasible", attempt.reason
        trace = attempt.candidate.trace
        assert trace["play"]["tricks"][0]["winner"] == "west"
        assert trace["projects"]["winningTeam"] == "west_east"
        assert trace["projects"]["combinedAwardedValue"] == list(projects.combined_awarded_value)
        assert {row["owner"] for row in trace["projects"]["projects"]
                if row["comparisonWinner"]} == {"west"}


def test_ratified_lift_recomputes_tied_awards_without_project_predicates(
    ratified_tie_lift_fixture,
):
    rules, hands, full, tail, boundary, projects = ratified_tie_lift_fixture
    scenario = _scenario(
        hands, boundary, _leaf("PRED-DEALER", "auction.dealer", {"seat": "east"}),
    )
    attempt = OrToolsTailFirstBackwardLiftV1(rules=rules, native_cp_sat_scoring=True).lift(
        scenario, TailFirstStagePlannerV1(native_cp_sat=True).plan(scenario), tail, boundary,
        play_hint=full, certified_projects=replace(projects, projects=()), maximum_seconds=60,
    )
    assert attempt.status == "feasible", attempt.reason
    trace = attempt.candidate.trace
    assert trace["projects"]["projects"]
    assert trace["projects"]["winningTeam"] == "west_east"
    assert trace["projects"]["combinedAwardedValue"] == list(projects.combined_awarded_value)


@pytest.mark.parametrize("contradictory_awards", [False, True])
def test_ratified_prefix_reachability_uses_the_first_trick_winner(
    ratified_tie_lift_fixture, contradictory_awards,
):
    rules, hands, full, tail, boundary, projects = ratified_tie_lift_fixture
    award_owner = Seat.NORTH if contradictory_awards else Seat.WEST
    project = next(row for row in projects.projects if row.owner == award_owner and row.declared)
    if contradictory_awards:
        boundary = _seed_clone(boundary, awarded_project_points=tuple(
            reversed(boundary.awarded_project_points)
        ))
    scenario = _scenario(hands, boundary, _leaf("PRED-AWARD", "project.exists", {
        "owner": award_owner.name.lower(), "team": award_owner.team.name.lower(),
        "kind": project.kind, "value": project.value, "state": "awarded",
    }))

    def configure(build):
        _bind_proof_seed_prefix(build, boundary, 6)
        _bind_reachable_project_truth_v1(
            build, scenario=scenario,
            manifest=shared_scenario_predicate_catalog_lowering_manifest_v2(),
            project_predicates=tuple(predicate for predicate in scenario.predicates()
                                     if predicate.kind == "project.exists"),
            seed=boundary, tail=tail, rules=rules, contract=Contract.SUN,
            dealer=Seat.EAST, trump=None,
        )

    options = dict(
        cards=FULL_DECK,
        ownership={card: seat for seat, hand in hands.items() for card in hand},
        contract=Contract.SUN, trump=None, first_leader=Seat.NORTH,
        play_domain=rules.play_domain, structural_domain=rules.structural_domain,
        scoring_domain=ScoringDomainIRV1.from_rules(rules.data),
        prefix=full.plays[:28], decision_prefix_length=24, play_step_count=28,
        construction_plan_hash="d" * 64, model_configurator=configure,
        maximum_time_seconds=15,
    )
    if contradictory_awards:
        with pytest.raises(PlayTailSolveStopped) as exc:
            OrToolsPlayTailStageV1(native_conversion=True).solve(**options)
        assert exc.value.solver_status == "infeasible"
    else:
        reached = OrToolsPlayTailStageV1(native_conversion=True).solve(**options)
        assert reached.trick_winners[0] == Seat.WEST


def _pin_ratified_boundary_opening(monkeypatch, full):
    original = OrToolsPlayTailStageV1.build_model
    horizons = []

    def build_pinned(stage, **kwargs):
        assert tuple(kwargs["cards"]) == FULL_DECK
        assert kwargs["shared_model"] is not None
        build = original(stage, **kwargs)
        horizon = len(build.selected_play_indexes)
        assert horizon in {4, 28}
        for index, action in enumerate(full.plays[:min(24, horizon)]):
            build.model.add(build.selected_play_indexes[index] == action.card.id)
        horizons.append(horizon)
        return build

    monkeypatch.setattr(OrToolsPlayTailStageV1, "build_model", build_pinned)
    return horizons


@pytest.mark.parametrize("operator", ["all", "any", "not", "count", "contradiction"])
def test_native_boundary_winner_bounds_respect_expression(
    ratified_tie_lift_fixture, monkeypatch, operator,
):
    rules, hands, full, tail, boundary, _projects = ratified_tie_lift_fixture
    selector = {"ranks": ["seven", "eight", "nine", "jack", "queen", "king", "ten", "ace"]}
    winner = _leaf("PRED-PREFIX-WINNER", "play.trick_winner_matches", {
        "trickIndex": 0, "team": "west_east", "selector": selector,
    })
    other = _leaf("PRED-PREFIX-OTHER-WINNER", "play.trick_winner_matches", {
        "trickIndex": 0, "team": "north_south", "selector": selector,
    })
    expression = winner if operator == "all" else {
        "op": "all" if operator == "contradiction" else operator,
        "children": [other] if operator == "not" else [winner, other],
        **({"minimum": 1, "maximum": 1} if operator == "count" else {}),
    }
    scenario = _scenario(hands, boundary, expression)
    ownership = {card: seat for seat, hand in hands.items() for card in hand}
    _pin_ratified_boundary_opening(monkeypatch, full)
    result = OrToolsTailProofBoundaryV1(rules=rules, native_cp_sat=True).solve(
        scenario, TailFirstStagePlannerV1(native_cp_sat=True).plan(scenario), tail,
        fixed_card_owners=tuple(ownership[card] for card in FULL_DECK),
        boundary_combination=("east", "sun", "north", None),
        required_base_raw_ns=boundary.base_raw[0],
        required_base_tricks_ns=boundary.base_tricks[0], maximum_seconds=15,
    )
    if operator == "contradiction":
        assert result.status == "infeasible", result.reason
        assert result.seed is None
    else:
        assert result.status == "feasible", result.reason
        assert result.seed.base_tricks == boundary.base_tricks


@pytest.mark.parametrize("path", ["aggregate", "full_prefix"])
@pytest.mark.parametrize("contradictory_awards", [False, True])
def test_ratified_boundary_binds_awards_to_its_native_first_trick(
    ratified_tie_lift_fixture, monkeypatch, path, contradictory_awards,
):
    rules, hands, full, tail, boundary, projects = ratified_tie_lift_fixture
    award_owner = Seat.NORTH if contradictory_awards else Seat.WEST
    project = next(row for row in projects.projects
                   if row.owner == award_owner and row.declared)
    scenario = _scenario(hands, boundary, _leaf("PRED-AWARD", "project.exists", {
        "owner": award_owner.name.lower(), "team": award_owner.team.name.lower(),
        "kind": project.kind, "value": project.value, "state": "awarded",
    }))
    ownership = {card: seat for seat, hand in hands.items() for card in hand}
    horizons = _pin_ratified_boundary_opening(monkeypatch, full)
    result = OrToolsTailProofBoundaryV1(rules=rules, native_cp_sat=True).solve(
        scenario, TailFirstStagePlannerV1(native_cp_sat=True).plan(scenario), tail,
        fixed_card_owners=(tuple(ownership[card] for card in FULL_DECK)
                           if path == "aggregate" else None),
        boundary_combination=("east", "sun", "north", None),
        required_base_raw_ns=boundary.base_raw[0],
        required_base_tricks_ns=boundary.base_tricks[0], maximum_seconds=30,
    )
    assert horizons == [4 if path == "aggregate" else 28]
    if contradictory_awards:
        assert result.status == "infeasible", result.reason
        assert result.seed is None
    else:
        assert result.status == "feasible", result.reason
        assert result.seed.awarded_project_points == projects.combined_awarded_value
        assert result.project_result.winning_team == Seat.WEST.team
        assert {row.owner for row in result.project_result.projects
                if row.comparison_winner} == {Seat.WEST}


def test_ratified_aggregate_boundary_has_live_awards_without_project_predicates(
    ratified_tie_lift_fixture, monkeypatch,
):
    rules, hands, full, tail, boundary, projects = ratified_tie_lift_fixture
    scenario = _scenario(hands, boundary)
    ownership = {card: seat for seat, hand in hands.items() for card in hand}
    horizons = _pin_ratified_boundary_opening(monkeypatch, full)

    def forbidden(*args, **kwargs):
        pytest.fail("ratified native boundary used standalone project fallback")

    monkeypatch.setattr(OrToolsProjectLifecycleModelV1, "solve", forbidden)
    result = OrToolsTailProofBoundaryV1(rules=rules, native_cp_sat=True).solve(
        scenario, TailFirstStagePlannerV1(native_cp_sat=True).plan(scenario), tail,
        fixed_card_owners=tuple(ownership[card] for card in FULL_DECK),
        boundary_combination=("east", "sun", "north", None),
        required_base_raw_ns=boundary.base_raw[0],
        required_base_tricks_ns=boundary.base_tricks[0], maximum_seconds=15,
    )
    assert horizons == [4]
    assert result.status == "feasible", result.reason
    assert result.seed.awarded_project_points == projects.combined_awarded_value
    assert result.project_result.winning_team == Seat.WEST.team


def test_ratified_boundary_opening_cannot_spend_a_reserved_tail_card(
    ratified_tie_lift_fixture, monkeypatch,
):
    rules, hands, _full, tail, boundary, _projects = ratified_tie_lift_fixture
    reserved = Card.parse("DK")
    assert any(play.card == reserved for play in tail.plays)
    ownership = {card: seat for seat, hand in hands.items() for card in hand}
    control = OrToolsPlayTailStageV1(native_conversion=True).solve(
        cards=FULL_DECK, ownership=ownership, contract=Contract.SUN, trump=None,
        first_leader=Seat.NORTH, play_domain=rules.play_domain,
        structural_domain=rules.structural_domain,
        scoring_domain=ScoringDomainIRV1.from_rules(rules.data),
        prefix=(TailPrefixActionV1.build("north", reserved),),
        play_step_count=4, decision_prefix_length=0, maximum_time_seconds=10,
    )
    assert len(control.plays) == 4 and control.plays[0].card == reserved
    original = OrToolsPlayTailStageV1.build_model
    horizons = []

    def spend_reserved(stage, **kwargs):
        assert kwargs["shared_model"] is not None
        build = original(stage, **kwargs)
        horizons.append(len(build.selected_play_indexes))
        build.model.add(build.selected_play_indexes[0] == reserved.id)
        return build

    monkeypatch.setattr(OrToolsPlayTailStageV1, "build_model", spend_reserved)
    scenario = _scenario(hands, boundary)
    result = OrToolsTailProofBoundaryV1(rules=rules, native_cp_sat=True).solve(
        scenario, TailFirstStagePlannerV1(native_cp_sat=True).plan(scenario), tail,
        fixed_card_owners=tuple(ownership[card] for card in FULL_DECK),
        boundary_combination=("east", "sun", "north", None),
        required_base_raw_ns=boundary.base_raw[0],
        required_base_tricks_ns=boundary.base_tricks[0], maximum_seconds=15,
    )
    assert horizons == [4]
    assert result.status == "infeasible", result.reason


def test_ratified_boundary_history_cut_preserves_the_opposite_project_award(
    ratified_tie_lift_fixture,
):
    rules, hands, _full, tail, boundary, _projects = ratified_tie_lift_fixture
    scenario = _scenario(
        hands, boundary,
        _leaf("PRED-NORMAL", "escalation.level", {"level": "normal"}),
        _leaf("PRED-OPEN", "escalation.locked", {"locked": False}),
    )
    ownership = {card: seat for seat, hand in hands.items() for card in hand}
    options = dict(
        fixed_card_owners=tuple(ownership[card] for card in FULL_DECK),
        boundary_combination=("east", "sun", "north", None),
        required_base_raw_ns=boundary.base_raw[0],
        required_base_tricks_ns=boundary.base_tricks[0], maximum_seconds=15,
    )
    model = OrToolsTailProofBoundaryV1(rules=rules, native_cp_sat=True)
    plan = TailFirstStagePlannerV1(native_cp_sat=True).plan(scenario)
    first = model.solve(scenario, plan, tail, **options)
    assert first.status == "feasible", first.reason
    assert first.no_good is not None and first.proof_state_no_good is not None
    second = model.solve(
        scenario, plan, tail, **options, excluded=(first.no_good,),
        excluded_proof_states=(first.proof_state_no_good,),
    )
    assert second.status == "feasible", second.reason
    assert second.no_good is not None
    assert second.no_good.vector[:35] == first.no_good.vector[:35]
    assert second.no_good.vector != first.no_good.vector
    assert second.seed.base_raw == first.seed.base_raw == boundary.base_raw
    assert second.seed.base_tricks == first.seed.base_tricks == boundary.base_tricks
    assert second.seed.escalation == first.seed.escalation == "normal"
    assert second.seed.locked is first.seed.locked is False
    assert second.seed.awarded_project_points == tuple(
        reversed(first.seed.awarded_project_points)
    )
    assert second.seed.proof_state_projection_hash != first.seed.proof_state_projection_hash
    assert {first.project_result.winning_team, second.project_result.winning_team} == {
        Seat.NORTH.team, Seat.WEST.team,
    }


@pytest.mark.parametrize("native", [False, True])
def test_tail_first_lift_builds_complete_trace_and_preserves_proof_boundary(
    rules, tail_lift_fixture, monkeypatch, native
) -> None:
    import baloot_platform.research.ortools_tail_first_lift as lift_module

    scenario_models = []
    original_scenario_model = lift_module.OrToolsScenarioModelV1

    def scenario_model(**kwargs):
        scenario_models.append(kwargs)
        return original_scenario_model(**kwargs)

    monkeypatch.setattr(lift_module, "OrToolsScenarioModelV1", scenario_model)
    if native:
        import baloot_platform.research.ortools_scenario_model as auction_model
        from baloot_platform.core.escalation_ir import EscalationDomainIRV1
        from baloot_platform.core.scoring_ir import ScoringDomainIRV1
        from baloot_platform.research.ortools_escalation_stage import OrToolsEscalationStageV1

        def forbidden(*args, **kwargs):
            pytest.fail("native tail lift used legacy auction/escalation evaluation")

        monkeypatch.setattr(auction_model, "_enumerate_auction", forbidden)
        monkeypatch.setattr(OrToolsEscalationStageV1, "_candidates", forbidden)
        monkeypatch.setattr(EscalationDomainIRV1, "transition", forbidden)
        monkeypatch.setattr(EscalationDomainIRV1, "legal_decisions", forbidden)
        monkeypatch.setattr(ScoringDomainIRV1, "convert_raw", forbidden)
    hands, tail, boundary = tail_lift_fixture
    scenario = _relaxed_late_play_scenario(tail_lift_fixture)
    plan = TailFirstStagePlannerV1().plan(scenario)

    attempt = OrToolsTailFirstBackwardLiftV1(rules=rules, native_cp_sat_scoring=native).lift(
        scenario,
        plan,
        tail,
        boundary,
        maximum_seconds=60,
    )

    assert attempt.status == "feasible", attempt.reason
    assert attempt.candidate is not None
    assert attempt.candidate.exact_state_id is None
    assert attempt.candidate.trace["dealer"] == "east"
    assert attempt.candidate.trace["buyer"] == "north"
    assert attempt.candidate.trace["contract"] == "sun"
    assert attempt.candidate.trace["trumpSuit"] is None
    assert len(attempt.candidate.trace["deckOrder"]) == 32
    assert len(attempt.candidate.trace["play"]["plays"]) == 32
    assert attempt.candidate.state["proofStateProjectionHash"] == (
        boundary.proof_state_projection_hash
    )
    assert attempt.candidate.stage_receipts["state"].evidence_hash == (
        boundary.proof_state_projection_hash
    )
    assert scenario_models
    assert all(
        options["canonicalize_deal_packets"] is True
        for options in scenario_models
    )


def test_tail_first_lift_binds_remaining_card_ownership(
    rules, tail_lift_fixture
) -> None:
    hands, tail, boundary = tail_lift_fixture
    remaining = boundary.remaining_hands[int(boundary.next_actor)][0]
    scenario = _scenario(
        hands,
        boundary,
        _leaf(
            "PRED-REMAINING-OWNER",
            "deal.remaining_card_owner",
            {"card": remaining.id, "seat": boundary.next_actor.name.lower()},
        ),
    )

    attempt = OrToolsTailFirstBackwardLiftV1(rules=rules).lift(
        scenario,
        TailFirstStagePlannerV1().plan(scenario),
        tail,
        boundary,
        maximum_seconds=60,
    )

    assert attempt.status == "feasible", attempt.reason


def test_tail_first_lift_excludes_one_play_zero_observation_action(
    rules, tail_lift_fixture
) -> None:
    hands, tail, boundary = tail_lift_fixture
    scenario = _scenario(hands, boundary)
    plan = TailFirstStagePlannerV1().plan(scenario)
    first = OrToolsTailFirstBackwardLiftV1(rules=rules).lift(
        scenario, plan, tail, boundary, maximum_seconds=60,
    )
    assert first.status == "feasible", first.reason
    assert first.candidate is not None
    first_play = first.candidate.trace["play"]["plays"][0]
    actor = Seat.parse(first_play["actor"])
    rejected = PlayZeroObservationActionNoGoodV1.build(
        actor=actor,
        actor_cards=first.candidate.trace["play"]["initialHands"][
            actor.name.lower()
        ],
        selected_card=first_play["card"],
        ekka=bool(first_play.get("ekka", False)),
    )

    second = OrToolsTailFirstBackwardLiftV1(rules=rules).lift(
        scenario,
        plan,
        tail,
        boundary,
        excluded_play_zero_actions=(rejected,),
        maximum_seconds=60,
    )

    assert second.status in {"feasible", "infeasible"}, second.reason
    if second.candidate is not None:
        second_play = second.candidate.trace["play"]["plays"][0]
        second_hand = {
            Card.parse(card).id
            for card in second.candidate.trace["play"]["initialHands"][
                actor.name.lower()
            ]
        }
        assert (
            second_hand != set(rejected.actor_card_ids)
            or Card.parse(second_play["card"]).id != rejected.selected_card_id
            or bool(second_play.get("ekka", False)) != rejected.ekka
        )


@pytest.mark.parametrize(
    ("actor_cards", "selected_card", "ekka", "feasible"),
    [
        (tuple(range(8)), 0, False, False),
        (tuple(range(8)), 1, False, True),
        ((*range(7), 8), 0, False, True),
        (tuple(range(8)), 0, True, True),
    ],
)
def test_play_zero_observation_action_no_good_truth_table(
    actor_cards, selected_card, ekka, feasible
) -> None:
    model = cp_model.CpModel()
    owners = tuple(model.new_int_var(0, 3, f"owner_{card}") for card in range(32))
    selected = model.new_int_var(0, 31, "selected")
    ekka_lead = model.new_bool_var("ekka")
    for card, owner in enumerate(owners):
        model.add(owner == (0 if card in actor_cards else 1))
    model.add(selected == selected_card)
    model.add(ekka_lead == int(ekka))
    apply_play_zero_observation_action_no_good_v1(
        model,
        owner_of_card=owners,
        selected_card=selected,
        ekka_lead=ekka_lead,
        no_good=PlayZeroObservationActionNoGoodV1.build(
            actor=Seat.NORTH,
            actor_cards=range(8),
            selected_card=0,
        ),
        name="test",
    )

    status = cp_model.CpSolver().solve(model)

    assert (status in {cp_model.FEASIBLE, cp_model.OPTIMAL}) is feasible




def test_tail_first_lift_rejects_partial_play_hint(rules, tail_lift_fixture) -> None:
    hands, tail, boundary = tail_lift_fixture
    with pytest.raises(ValueError, match="play hint must be one complete round"):
        OrToolsTailFirstBackwardLiftV1(rules=rules).lift(
            _scenario(hands, boundary),
            TailFirstStagePlannerV1().plan(_scenario(hands, boundary)),
            tail,
            boundary,
            play_hint=tail,
        )


@pytest.mark.real_clock
def test_tail_first_lift_deadline_before_play_solve_is_incomplete(
    rules, tail_lift_fixture
) -> None:
    hands, tail, boundary = tail_lift_fixture
    scenario = _scenario(hands, boundary)
    attempt = OrToolsTailFirstBackwardLiftV1(rules=rules).lift(
        scenario,
        TailFirstStagePlannerV1().plan(scenario),
        tail,
        boundary,
        maximum_seconds=0.001,
    )
    assert attempt.status == "incomplete"
    assert attempt.reason == "tail_prefix_lift_budget"


def test_tail_first_lift_rejects_auction_trump_different_from_proof_tail(
    rules, tail_lift_fixture
) -> None:
    hands, tail, boundary = tail_lift_fixture
    lifter = OrToolsTailFirstBackwardLiftV1(rules=rules, native_cp_sat_scoring=True)
    with pytest.raises(AssertionError, match="auction and tactical tail disagree on trump"):
        lifter._candidate(
            _scenario(hands, boundary),
            None,
            type("Auction", (), {"completion_buyer": boundary.buyer,
                                 "completion_contract": boundary.contract,
                                 "completion_trump": Suit.HEARTS})(),
            None, None, tail, tail, boundary, None, {},
        )


def test_tail_first_lift_identifies_optimal_auction_incompatible_tail(
    rules, tail_lift_fixture, monkeypatch
) -> None:
    hands, tail, boundary = tail_lift_fixture
    scenario = _scenario(hands, boundary)
    monkeypatch.setattr(
        tail_lift_module,
        "_solve_cp_sat_with_cancel_v1",
        lambda *_args, **_kwargs: cp_model.INFEASIBLE,
    )

    attempt = OrToolsTailFirstBackwardLiftV1(rules=rules).lift(
        scenario,
        TailFirstStagePlannerV1().plan(scenario),
        tail,
        boundary,
        require_optimal_auction_actions=True,
        maximum_seconds=5,
    )

    assert attempt.status == "infeasible"
    assert attempt.reason == "tail_optimal_auction_infeasible"


def _relaxed_late_play_scenario(tail_lift_fixture, *extra):
    hands, tail, expected_boundary = tail_lift_fixture
    action_constraints = tuple(
        _leaf(
            f"PRED-TAIL-ACTION-{index:02d}",
            "play.action_matches",
            {
                "trickIndex": 6 + index // 4,
                "position": index % 4,
                "actingSeat": action.actor.name.lower(),
                "selector": {"cards": [action.card.id]},
            },
        )
        for index, action in enumerate(tail.plays)
    )
    scenario = _scenario(
        hands,
        expected_boundary,
        _leaf("PRED-BUYER-EXACT", "auction.buyer", {"seat": "north"}),
        _leaf(
            "PRED-RAW-NS",
            "state.current_raw_points",
            {
                "team": "north_south",
                "minimum": expected_boundary.base_raw[0],
                "maximum": expected_boundary.base_raw[0],
            },
        ),
        _leaf(
            "PRED-RAW-WE",
            "state.current_raw_points",
            {
                "team": "west_east",
                "minimum": expected_boundary.base_raw[1],
                "maximum": expected_boundary.base_raw[1],
            },
        ),
        *extra,
        *action_constraints,
    )
    return scenario


def _relaxed_exact_proof() -> ScenarioProofRequestV1:
    return ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "exact",
            "visibility": "complete_state",
            "claim": "outcome_reachable",
            "targetTeam": "north_south",
            "utility": "round_win",
            "successThreshold": -100_000,
            "budgetProfile": "tail_first_test",
        }
    )


def test_tail_boundary_constructs_relaxed_late_play(rules, tail_lift_fixture, monkeypatch) -> None:
    from baloot_platform.core.escalation_ir import EscalationDomainIRV1
    from baloot_platform.research.ortools_escalation_stage import OrToolsEscalationStageV1

    def forbidden(*args, **kwargs):
        pytest.fail("native boundary used reference escalation evaluation")

    monkeypatch.setattr(OrToolsEscalationStageV1, "_candidates", forbidden)
    monkeypatch.setattr(EscalationDomainIRV1, "transition", forbidden)
    monkeypatch.setattr(EscalationDomainIRV1, "legal_decisions", forbidden)
    _hands, tail, expected_boundary = tail_lift_fixture
    scenario = _relaxed_late_play_scenario(tail_lift_fixture)
    plan = TailFirstStagePlannerV1().plan(scenario)
    constructed = OrToolsTailProofBoundaryV1(rules=rules, native_cp_sat=True).solve(
        scenario,
        plan,
        tail,
        boundary_combination=("east", "sun", "north", None),
        maximum_seconds=15,
    )
    assert constructed.status == "feasible", constructed
    assert constructed.seed is not None
    assert constructed.seed.base_raw == expected_boundary.base_raw


def test_prefix_reachability_binds_the_actual_proof_state(rules, tail_lift_fixture) -> None:
    hands, tail, boundary = tail_lift_fixture
    impossible = _seed_clone(
        boundary,
        base_raw=(sum(boundary.base_raw), 0),
        base_tricks=(0, 6),
    )
    options = dict(
        cards=FULL_DECK,
        ownership={card: seat for seat, hand in hands.items() for card in hand},
        contract=Contract.SUN,
        trump=None,
        first_leader=Seat.NORTH,
        play_domain=rules.play_domain,
        structural_domain=rules.structural_domain,
        scoring_domain=ScoringDomainIRV1.from_rules(rules.data),
        tail=tail.plays,
        decision_prefix_length=24,
        play_step_count=32,
        construction_plan_hash="a" * 64,
        maximum_time_seconds=10,
    )
    reached = OrToolsPlayTailStageV1().solve(
        **options,
        model_configurator=lambda build: _bind_proof_seed_prefix(build, boundary, 6),
    )
    assert reached.at_prefix(24).next_actor == boundary.next_actor
    with pytest.raises(PlayTailSolveStopped) as exc:
        OrToolsPlayTailStageV1().solve(
            **options,
            model_configurator=lambda build: _bind_proof_seed_prefix(build, impossible, 6),
        )
    assert exc.value.solver_status == "infeasible"


def test_full_deal_partial_round_keeps_eight_card_hand_sizes(
    rules, tail_lift_fixture
) -> None:
    hands, tail, _boundary = tail_lift_fixture
    builds = []
    solver = cp_model.CpSolver()
    result = OrToolsPlayTailStageV1().solve(
        cards=FULL_DECK,
        ownership={card: seat for seat, hand in hands.items() for card in hand},
        contract=Contract.SUN,
        trump=None,
        first_leader=Seat.NORTH,
        play_domain=rules.play_domain,
        structural_domain=rules.structural_domain,
        scoring_domain=ScoringDomainIRV1.from_rules(rules.data),
        tail=tail.plays[:4],
        decision_prefix_length=24,
        play_step_count=28,
        construction_plan_hash="c" * 64,
        model_configurator=builds.append,
        maximum_time_seconds=15,
        solver=solver,
    )
    assert len(result.plays) == 28
    assert len(builds) == 1
    assert tuple(
        solver.value(builds[0].play_state_context.remaining_hand_sizes[seat])
        for seat in Seat
    ) == (2, 2, 2, 2)
    assert tuple(
        len(hand) for hand in result.at_prefix(24).remaining_hands
    ) == (2, 2, 2, 2)


def test_prefix_reachability_binds_project_awards_to_the_same_deal(
    rules, tail_lift_fixture
) -> None:
    hands, tail, boundary = tail_lift_fixture
    scenario = _relaxed_late_play_scenario(
        tail_lift_fixture,
        _leaf(
            "PRED-AWARDED-HUNDRED",
            "project.exists",
            {
                "team": "north_south", "owner": "north",
                "kind": "hundred_sequence", "value": 20, "state": "awarded",
            },
        ),
    )
    manifest = shared_scenario_predicate_catalog_lowering_manifest_v2()
    options = dict(
        cards=FULL_DECK,
        ownership={card: seat for seat, hand in hands.items() for card in hand},
        contract=Contract.SUN,
        trump=None,
        first_leader=Seat.NORTH,
        play_domain=rules.play_domain,
        structural_domain=rules.structural_domain,
        scoring_domain=ScoringDomainIRV1.from_rules(rules.data),
        tail=tail.plays[:4],
        decision_prefix_length=24,
        play_step_count=28,
        construction_plan_hash="b" * 64,
        maximum_time_seconds=15,
    )

    def bind(seed):
        def configure(build):
            _bind_proof_seed_prefix(build, seed, 6)
            _bind_reachable_project_truth_v1(
                build,
                scenario=scenario,
                manifest=manifest,
                project_predicates=tuple(
                    predicate for predicate in scenario.predicates()
                    if predicate.kind == "project.exists"
                ),
                seed=seed,
                tail=tail,
                rules=rules,
                contract=Contract.SUN,
                dealer=Seat.EAST,
                trump=None,
            )
        return configure

    assert boundary.awarded_project_points == (20, 0)
    OrToolsPlayTailStageV1().solve(
        **options, model_configurator=bind(boundary),
    )
    wrong_award = _seed_clone(boundary, awarded_project_points=(0, 44))
    with pytest.raises(PlayTailSolveStopped) as exc:
        OrToolsPlayTailStageV1().solve(
            **options, model_configurator=bind(wrong_award),
        )
    assert exc.value.solver_status == "infeasible"


def test_prefix_probe_cannot_spend_a_card_reserved_for_later_tail(
    rules, tail_lift_fixture,
) -> None:
    hands, tail, _boundary = tail_lift_fixture
    later_card = next(
        play.card for play in tail.plays[4:] if play.actor == Seat.NORTH
    )
    options = dict(
        cards=FULL_DECK,
        ownership={card: seat for seat, hand in hands.items() for card in hand},
        contract=Contract.SUN,
        trump=None,
        first_leader=Seat.NORTH,
        play_domain=rules.play_domain,
        structural_domain=rules.structural_domain,
        scoring_domain=ScoringDomainIRV1.from_rules(rules.data),
        tail=tail.plays[:4],
        play_step_count=24,
        construction_plan_hash="c" * 64,
        maximum_time_seconds=10,
    )

    def spend_later_card(build):
        build.model.add(build.selected_play_indexes[0] == later_card.id)

    # The weaker prefix model can legally consume a card that the full tail needs.
    OrToolsPlayTailStageV1().solve(**options, model_configurator=spend_later_card)

    def preserve_tail(build):
        spend_later_card(build)
        assert _preserve_tail_cards_before_decision(build, tail, 20)

    with pytest.raises(PlayTailSolveStopped) as exc:
        OrToolsPlayTailStageV1().solve(**options, model_configurator=preserve_tail)
    assert exc.value.solver_status == "infeasible"


@pytest.mark.real_clock
def test_tail_boundary_deadline_is_incomplete_not_infeasible(rules, tail_lift_fixture) -> None:
    _hands, tail, _expected_boundary = tail_lift_fixture
    scenario = _relaxed_late_play_scenario(tail_lift_fixture)
    result = OrToolsTailProofBoundaryV1(rules=rules).solve(
        scenario, TailFirstStagePlannerV1().plan(scenario), tail,
        maximum_seconds=0.000000001,
    )
    assert result.status == "incomplete"
    assert result.reason == "tail_boundary_budget"


@pytest.mark.parametrize(
    ("raw_points", "won_tricks"),
    [(1, None), (22, 0)],
)
def test_tail_boundary_rejects_uncollectible_sun_prefix_points(
    rules, tail_lift_fixture, raw_points, won_tricks
) -> None:
    _hands, tail, _expected_boundary = tail_lift_fixture
    raw = _relaxed_late_play_scenario(tail_lift_fixture).to_dict()
    raw.pop("sha256")
    raw["constraints"].pop("sha256")
    raw["constraints"]["children"] = [
        predicate for predicate in raw["constraints"]["children"]
        if predicate["kind"] != "state.current_raw_points"
    ]
    scenario = ResearchScenarioV1.build(raw)
    result = OrToolsTailProofBoundaryV1(rules=rules, native_cp_sat=True).solve(
        scenario, TailFirstStagePlannerV1().plan(scenario), tail,
        boundary_combination=("east", "sun", "north", None),
        required_base_raw_ns=raw_points,
        required_base_tricks_ns=won_tricks,
        maximum_seconds=8,
    )
    assert result.status == "infeasible", result


def test_hard_awarded_hundred_boundary_uses_project_witness(
    rules, tail_lift_fixture
) -> None:
    _hands, tail, expected_boundary = tail_lift_fixture
    scenario = _relaxed_late_play_scenario(
        tail_lift_fixture,
        _leaf(
            "PRED-AWARDED-HUNDRED",
            "project.exists",
            {
                "team": "north_south",
                "owner": "north",
                "kind": "hundred_sequence",
                "value": 20,
                "state": "awarded",
            },
        ),
    )
    plan = TailFirstStagePlannerV1().plan(scenario)

    constructed = OrToolsTailProofBoundaryV1(rules=rules).solve(
        scenario,
        plan,
        tail,
        boundary_combination=("east", "sun", "north", None),
        maximum_seconds=8,
    )

    assert constructed.status == "feasible", constructed
    assert constructed.seed is not None
    assert constructed.project_result is not None
    assert constructed.project_result.evidence_hash
    assert constructed.seed.awarded_project_points[0] >= 20
    binding = next(
        item
        for item in constructed.evidence_bindings
        if item["predicateId"] == "PRED-AWARDED-HUNDRED"
    )
    assert binding["literalName"] == (
        "scenario_assumption_PRED-AWARDED-HUNDRED"
    )
    assert binding["literalIndex"] >= 0
    evidence = constructed.predicate_evidence["PRED-AWARDED-HUNDRED"]
    assert evidence["satisfied"] is True
    assert binding["evidenceHash"] == evidence["sha256"]


def test_lift_recomputes_certified_projects_for_its_final_deal(
    rules, tail_lift_fixture
) -> None:
    hands, tail, boundary = tail_lift_fixture
    certified = OrToolsProjectLifecycleModelV1().solve(
        hands=hands, contract=Contract.SUN, dealer=Seat.EAST, trump=None,
        domain=rules.project_domain,
    )
    assert certified.projects
    assert certified.combined_awarded_value == boundary.awarded_project_points
    awarded = next(row for row in certified.projects if row.awarded)
    scenario = _relaxed_late_play_scenario(
        tail_lift_fixture,
        _leaf("PRED-CERTIFIED-PROJECT", "project.exists", {
            "team": awarded.team.name.lower(),
            "owner": awarded.owner.name.lower(),
            "kind": awarded.kind,
            "value": awarded.value,
            "state": "awarded",
        }),
    )
    plan = TailFirstStagePlannerV1().plan(scenario)
    stale = replace(certified, projects=())

    attempt = OrToolsTailFirstBackwardLiftV1(
        rules=rules, native_cp_sat_scoring=True,
    ).lift(
        scenario, plan, tail, boundary,
        certified_projects=stale, maximum_seconds=60,
    )

    assert attempt.status == "feasible", attempt.reason
    assert attempt.candidate is not None
    projects = attempt.candidate.trace["projects"]
    assert projects["projects"]
    assert projects["combinedAwardedValue"] == list(boundary.awarded_project_points)


def test_proof_state_cut_collapses_project_neutral_history_permutations(
    rules, tail_lift_fixture
) -> None:
    _hands, tail, _expected_boundary = tail_lift_fixture
    raw = _relaxed_late_play_scenario(tail_lift_fixture).to_dict()
    raw.pop("sha256")
    raw["constraints"].pop("sha256")
    raw["constraints"]["children"] = [
        predicate
        for predicate in raw["constraints"]["children"]
        if not (
            predicate["kind"] == "deal.card_owner"
            and predicate["arguments"]["card"]
            in {Card.parse("CK").id, Card.parse("S9").id}
        )
    ]
    raw["constraints"]["children"].extend(
        (
            _leaf("PRED-ESCALATION-NORMAL", "escalation.level", {"level": "normal"}),
            _leaf("PRED-ESCALATION-OPEN", "escalation.locked", {"locked": False}),
        )
    )
    scenario = ResearchScenarioV1.build(raw)
    plan = TailFirstStagePlannerV1().plan(scenario)
    no_project_data = copy.deepcopy(rules.data)
    no_project_data["projects"]["enabled"] = False
    boundary = OrToolsTailProofBoundaryV1(rules=RuleBook(no_project_data))

    first = boundary.solve(
        scenario,
        plan,
        tail,
        boundary_combination=("east", "sun", "north", None),
        required_base_tricks_ns=2,
        maximum_seconds=15,
    )
    assert first.status == "feasible" and first.seed is not None
    assert first.no_good is not None and first.proof_state_no_good is not None
    assert (
        first.proof_state_no_good.proof_state_projection_hash
        == first.seed.proof_state_projection_hash
    )

    second_history = boundary.solve(
        scenario,
        plan,
        tail,
        excluded=(first.no_good,),
        boundary_combination=("east", "sun", "north", None),
        required_base_tricks_ns=2,
        maximum_seconds=15,
    )
    assert second_history.status == "feasible" and second_history.seed is not None
    assert (
        second_history.seed.proof_state_projection_hash
        == first.seed.proof_state_projection_hash
    ), (first.seed.to_dict(), second_history.seed.to_dict())

    required = boundary.solve(
        scenario,
        plan,
        tail,
        required_history=first.no_good,
        required_proof_state=first.proof_state_no_good,
        boundary_combination=("east", "sun", "north", None),
        required_base_tricks_ns=2,
        maximum_seconds=15,
    )
    assert required.status == "feasible" and required.seed is not None
    assert (
        required.seed.proof_state_projection_hash
        == first.seed.proof_state_projection_hash
    )

    collapsed = boundary.solve(
        scenario,
        plan,
        tail,
        excluded_proof_states=(first.proof_state_no_good,),
        boundary_combination=("east", "sun", "north", None),
        required_base_tricks_ns=2,
        maximum_seconds=15,
    )
    assert collapsed.status == "infeasible"


@pytest.mark.real_clock
def test_default_executor_finds_relaxed_late_play(rules, tail_lift_fixture) -> None:
    scenario = _relaxed_late_play_scenario(tail_lift_fixture,
        _leaf("PRED-AUTO-TACTIC", "tactic.applicable", {
            "kind": "only_winning_move", "applicable": True}))
    result = OrToolsUnifiedResearchExecutorV1(
        rules,
        bundle_revision_id="CBR-TAIL-FIRST-TEST",
        bundle_hash="a" * 64,
    ).execute(
        scenario,
        _relaxed_exact_proof(),
        maximum_proof_nodes=10_000,
        maximum_proof_depth=12,
        maximum_prefix_lift_attempts=32,
        maximum_seconds=30,
    )
    assert result.status == "qualified", json.dumps(result.execution_receipt, sort_keys=True)
    assert result.proof is not None and result.proof["claimSatisfied"] is True
    evidence = result.proof["nativeQualification"]["PRED-AUTO-TACTIC"]
    assert evidence["satisfied"] is True
    assert evidence["observed"]["selectedRootAction"] == result.proof["selectedAction"]
    assert result.candidate is not None
    assert result.candidate.trace["contract"] == "sun"
    assert result.execution_receipt["stageTimings"]["tailConstruction"] > 0
    assert result.execution_receipt["stageTimings"]["proofBoundaryModelBuild"] > 0
    assert result.execution_receipt["stageTimings"]["proofBoundarySolver"] > 0
    assert result.execution_receipt["stageTimings"]["prefixLift"] > 0
    tail_receipt = result.candidate.stage_receipts["tailConstruction"]
    assert tail_receipt.variable_count > 0
    assert tail_receipt.constraint_count > 0
    assert tail_receipt.branches > 0


@pytest.mark.parametrize('hardness', ['report', 'prefer'])
@pytest.mark.parametrize('selective_tail', [False, True])
@pytest.mark.parametrize('tactic', ['trump_control', 'small_card_big_gain', 'large_card_sacrifice', 'partner_enablement', 'entry_unblocking'])
def test_native_tail_proof_records_non_qualifying_tactic_reports(rules, tail_lift_fixture, monkeypatch, hardness, selective_tail, tactic):
    from baloot_platform.tactics import TacticalDetectorProfileV1
    from baloot_platform.research.ortools_tail_first_candidate import OrToolsTailFirstCandidateCompilerV1
    from baloot_platform.research.ortools_proof import OrToolsScenarioGraphProofV1

    scenario = _relaxed_late_play_scenario(tail_lift_fixture,
        _leaf('PRED-AUTO-TACTIC', 'tactic.applicable', {'kind': 'only_winning_move', 'applicable': True}),
        _leaf('PRED-TACTIC-REPORT', 'tactic.verified', {'kind': tactic,
            'minimumScore': 0, 'minimumEvidence': 0, 'causalRequired': True}, hardness))
    if selective_tail:
        payload = scenario.to_dict()
        payload.pop('sha256')
        payload['constraints'].pop('sha256')
        payload['constraints']['children'] = [child for child in payload['constraints']['children']
            if child.get('kind') != 'play.trick_index']
        scenario = ResearchScenarioV1.build(payload)
    expected_raw = tail_lift_fixture[2].base_raw
    solve_boundary = OrToolsTailProofBoundaryV1.solve
    def observe_boundary(self, *args, **kwargs):
        assert kwargs['required_base_raw_ns'] == (None if selective_tail else expected_raw[0])
        boundary = solve_boundary(self, *args, **kwargs)
        if boundary.status == 'feasible':
            assert boundary.seed.base_raw == expected_raw
        return boundary
    monkeypatch.setattr(OrToolsTailProofBoundaryV1, 'solve', observe_boundary)
    prove = OrToolsScenarioGraphProofV1.prove
    class ReachedNativeProof(RuntimeError):
        pass
    def observe_proof(self, *args, **kwargs):
        proof = prove(self, *args, **kwargs)
        if kwargs.get('tactic_configure') is not None:
            reported = proof.native_qualification['PRED-TACTIC-REPORT']
            if tactic in {'partner_enablement', 'entry_unblocking'}:
                assert reported['observed']['mechanismScope'] == 'paired_canonical_continuations'
                assert reported['observed']['pairedWinnerVariables'] == [
                    {key: value for key, value in row['variables'].items() if key.endswith('WinnerSeat')}
                    for row in proof.model_document()['pairedRawPayoffVariables']]
            else:
                assert reported['satisfied'] is False
                assert reported['observed']['causalProven'] is False
            if tactic == 'trump_control':
                assert reported['observed']['trumpExtraction'] is False
                assert reported['observed']['causalGainScale'] == args[0].utility_scale
                assert reported['observed']['detectorProfileHash'] == TacticalDetectorProfileV1.defaults().sha256
            elif tactic not in {'partner_enablement', 'entry_unblocking'}:
                assert reported['observed']['causalProofStatus'] == 'missing_native_delayed_payoff_witness'
            assert reported['observed']['detectorVersion'] == '3.0.1'
            raise ReachedNativeProof
        return proof
    monkeypatch.setattr(OrToolsScenarioGraphProofV1, 'prove', observe_proof)
    # Isolate the actual native proof component, not source-profile/service admission.
    stages = []
    request = _relaxed_exact_proof()
    if tactic in {'trump_control', 'partner_enablement', 'entry_unblocking'}:
        raw_request = request.to_dict()
        raw_request.pop('sha256')
        request = ScenarioProofRequestV1.build({**raw_request, 'utility': 'awarded_point_margin'})
    compiler = OrToolsTailFirstCandidateCompilerV1(rules=rules,
        plan=TailFirstStagePlannerV1(native_cp_sat=True).plan(scenario), proof_request=request,
        maximum_proof_nodes=10000, maximum_proof_depth=12, maximum_line_count_limbs=4,
        maximum_conflicts=100000, maximum_prefix_lift_attempts=32, native_cp_sat_scoring=True,
        tactical_detector_profile=TacticalDetectorProfileV1.defaults(),
        progress_callback=stages.append)
    with pytest.raises(ReachedNativeProof):
        attempt = compiler(scenario, (), seed=0, maximum_seconds=30, maximum_conflicts=100000)
        pytest.fail(json.dumps({'reason': attempt.reason, 'stages': stages}, sort_keys=True))


@pytest.mark.real_clock
def test_default_executor_lifts_unowned_kabout_escape_without_project_retries(
    rules,
) -> None:
    any_rank = {
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
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": {
                "op": "all",
                "children": [
                    _leaf("PRED-CONTRACT", "auction.contract", {"contract": "sun"}),
                    *(
                        _leaf(
                            f"PRED-WE-{trick}",
                            "play.trick_winner_matches",
                            {
                                "trickIndex": trick,
                                "team": "west_east",
                                "selector": any_rank,
                            },
                        )
                        for trick in range(5)
                    ),
                    _leaf(
                        "PRED-ACE",
                        "play.action_matches",
                        {
                            "trickIndex": 5,
                            "position": 1,
                            "actingSeat": "south",
                            "selector": {"ranks": ["ace"]},
                        },
                    ),
                    *(
                        _leaf(
                            f"PRED-WINNER-{trick}",
                            "play.trick_winner_matches",
                            {
                                "trickIndex": trick,
                                "seat": seat,
                                "selector": any_rank,
                            },
                        )
                        for trick, seat in (
                            (5, "south"),
                            (6, "north"),
                            (7, "south"),
                        )
                    ),
                ],
            },
            "decisionPoint": {
                "kind": "play",
                "selector": {
                    "trickIndex": 5,
                    "position": 1,
                    "remainingCards": 3,
                    "actingSeat": "south",
                },
            },
            "outcome": _leaf(
                "PRED-NO-KABOUT",
                "outcome.kabout",
                {"team": "north_south", "achieved": False},
            ),
        }
    )
    proof = ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "exact",
            "visibility": "complete_state",
            "claim": "outcome_reachable",
            "targetTeam": "north_south",
            "utility": "round_win",
            "successThreshold": -1_000,
            "budgetProfile": "tail_first_test",
        }
    )

    result = OrToolsUnifiedResearchExecutorV1(
        rules,
        bundle_revision_id="CBR-TAIL-FIRST-TEST",
        bundle_hash="a" * 64,
    ).execute(
        scenario,
        proof,
        maximum_proof_nodes=10_000,
        maximum_proof_depth=12,
        maximum_prefix_lift_attempts=32,
        maximum_seconds=30,
    )

    # The lift succeeds once; the independent proof PV need not retain this
    # construction witness's Ace/winner pattern, so qualification rejects it.
    assert result.status == "semantic_postcondition_miss"
    assert result.candidate is not None
    assert result.candidate.trace["tailFirstSearchCounters"]["prefixLiftAttempts"] == 1
    assert result.predicate_evaluation is not None
    assert result.predicate_evaluation.status == "target_miss"
    assert result.execution_receipt["stageTimings"]["prefixLift"] < 15


def test_tail_boundary_binds_remaining_card_ownership(rules) -> None:
    hands = {
        Seat.NORTH: ("H7", "H8"),
        Seat.WEST: ("D7", "D8"),
        Seat.SOUTH: ("C7", "SA"),
        Seat.EAST: ("S7", "S8"),
    }
    plays = tuple(
        TailPrefixActionV1.build(actor, card)
        for actor, card in (
            ("south", "C7"),
            ("east", "S7"),
            ("north", "H7"),
            ("west", "D7"),
            ("south", "SA"),
            ("east", "S8"),
            ("north", "H8"),
            ("west", "D8"),
        )
    )
    tail = OrToolsPlayTailStageV1().solve(
        cards=tuple(card for hand in hands.values() for card in hand),
        ownership={card: seat for seat, hand in hands.items() for card in hand},
        contract="sun",
        trump=None,
        first_leader="south",
        play_domain=rules.play_domain,
        structural_domain=rules.structural_domain,
        tail=plays,
        decision_prefix_length=0,
        decision_actor="south",
        maximum_time_seconds=5,
    )
    scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": {
                "op": "all",
                "children": [
                    _leaf("PRED-CONTRACT", "auction.contract", {"contract": "sun"}),
                    _leaf(
                        "PRED-SOUTH-ACE",
                        "deal.remaining_card_owner",
                        {"card": 7, "seat": "south"},
                    ),
                    _leaf(
                        "PRED-TWO-ACTIONS",
                        "state.legal_action_count",
                        {"minimum": 2, "maximum": 2},
                    ),
                ],
            },
            "decisionPoint": {
                "kind": "play",
                "selector": {
                    "trickIndex": 6,
                    "position": 0,
                    "remainingCards": 2,
                    "actingSeat": "south",
                },
            },
            "outcome": _leaf(
                "PRED-ROUND-WIN",
                "outcome.team_round",
                {"team": "north_south", "status": "won"},
                "report",
            ),
        }
    )
    result = OrToolsTailProofBoundaryV1(rules=rules).solve(
        scenario,
        TailFirstStagePlannerV1().plan(scenario),
        tail,
        maximum_seconds=5,
    )

    assert result.status == "feasible", result
    assert result.seed is not None
    assert Card.parse("SA") in result.seed.remaining_hands[int(Seat.SOUTH)]


def test_default_tail_first_executor_requires_governed_lift_attempt_budget(
    rules, tail_lift_fixture
) -> None:
    with pytest.raises(
        ValueError,
        match="tail-first Research requires maximum_prefix_lift_attempts",
    ):
        OrToolsUnifiedResearchExecutorV1(
            rules,
            bundle_revision_id="CBR-TAIL-FIRST-TEST",
            bundle_hash="a" * 64,
        ).execute(
            _relaxed_late_play_scenario(tail_lift_fixture),
            _relaxed_exact_proof(),
            maximum_proof_nodes=10_000,
            maximum_proof_depth=12,
            maximum_seconds=30,
        )


@pytest.mark.real_clock
def test_default_executor_finds_sun_awarded_hundred_late_play(
    rules, tail_lift_fixture
) -> None:
    scenario = _relaxed_late_play_scenario(
        tail_lift_fixture,
        _leaf(
            "PRED-AWARDED-HUNDRED",
            "project.exists",
            {
                "team": "north_south",
                "owner": "north",
                "kind": "hundred_sequence",
                "value": 20,
                "state": "awarded",
            },
        ),
    )
    result = OrToolsUnifiedResearchExecutorV1(
        rules,
        bundle_revision_id="CBR-TAIL-FIRST-TEST",
        bundle_hash="a" * 64,
    ).execute(
        scenario,
        _relaxed_exact_proof(),
        maximum_proof_nodes=10_000,
        maximum_proof_depth=12,
        maximum_prefix_lift_attempts=32,
        maximum_seconds=30,
    )

    assert result.status == "qualified"
    assert result.proof is not None and result.proof["claimSatisfied"] is True
    assert result.candidate is not None
    assert result.candidate.trace["contract"] == "sun"
    assert result.candidate.trace["projects"]["combinedAwardedValue"][0] >= 20
    assert result.execution_receipt["stageTimings"]["prefixLift"] < 15
    assert result.execution_receipt["stageTimings"]["tailModelBuild"] > 0
    assert result.execution_receipt["stageTimings"]["tailSolver"] > 0


@pytest.mark.parametrize(
    "contradiction",
    [
        _leaf(
            "PRED-AUCTION-CONTRADICTION",
            "auction.action_sequence",
            {
                "actions": [{"actor": "east", "action": "sun"}],
                "complete": True,
            },
        ),
        _leaf(
            "PRED-PROJECT-CONTRADICTION",
            "project.baloot",
            {"team": "north_south", "achieved": True},
        ),
    ],
)
def test_auction_and_project_contradictions_are_exactly_infeasible(
    rules, tail_lift_fixture, contradiction
) -> None:
    hands, tail, boundary = tail_lift_fixture
    scenario = _scenario(hands, boundary, contradiction)
    plan = TailFirstStagePlannerV1().plan(scenario)

    attempt = OrToolsTailFirstBackwardLiftV1(rules=rules).lift(
        scenario,
        plan,
        tail,
        boundary,
        maximum_seconds=30,
    )

    assert attempt.status == "infeasible"
    assert attempt.reason == "tail_prefix_lift_infeasible"


def test_tail_plan_and_proof_checksums_fail_closed(rules, tail_lift_fixture) -> None:
    hands, tail, boundary = tail_lift_fixture
    scenario = _scenario(hands, boundary)
    plan = TailFirstStagePlannerV1().plan(scenario)
    lift = OrToolsTailFirstBackwardLiftV1(rules=rules)

    with pytest.raises(ValueError, match="plan checksum mismatch"):
        lift.lift(scenario, replace(plan, sha256="0" * 64), tail, boundary)
    with pytest.raises(ValueError, match="projection checksum mismatch"):
        lift.lift(
            scenario,
            plan,
            tail,
            replace(boundary, proof_state_projection_hash="0" * 64),
        )
