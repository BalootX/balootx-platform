from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Contract, Seat, Suit
from baloot_platform.core.scoring_ir import ScoringDomainIRV1
from baloot_platform.challenge.evidence_v3 import canonical_sha256
from baloot_platform.research import OrToolsCpSatPlayCompilerV1
from baloot_platform.service.container import build_container


def _trace(seed: int):
    return build_container().research.simulator.simulate(seed).final_trace.trusted_final_state


@pytest.mark.parametrize("seed,contract", [(1, "sun"), (5, "hokum")])
def test_complete_fixed_trace_matches_play_tricks_and_score(seed: int, contract: str):
    container = build_container()
    state = _trace(seed)
    assert state["contract"] == contract
    plan = OrToolsCpSatPlayCompilerV1().verify_trace(
        state, rules=container.research.simulator.rules
    )
    assert plan.solver_status == "optimal"
    assert [trick.winner.name.lower() for trick in plan.tricks] == [
        trick["winner"] for trick in state["completedTricks"]
    ]
    assert list(plan.awarded_points) == state["roundScore"]["awardedPoints"]


def test_fixed_trace_compilation_is_deterministic():
    container = build_container()
    state = _trace(5)
    compiler = OrToolsCpSatPlayCompilerV1()
    first = compiler.verify_trace(state, rules=container.research.simulator.rules)
    second = compiler.verify_trace(state, rules=container.research.simulator.rules)
    assert first == second


def test_distinct_comparison_outputs_are_independently_verified_not_cross_compared():
    container = build_container()
    try:
        # A matched Research comparison freezes inputs, not outputs. Each lane
        # may retain a different deal and play line. These neutral fixtures
        # prove the parity scope, not either lane's generation implementation.
        comparison_outputs = {
            "lane_a": container.research.simulator.simulate(1),
            "lane_b": container.research.simulator.simulate(5),
        }
        states = {
            lane: game.final_trace.trusted_final_state
            for lane, game in comparison_outputs.items()
        }
        assert states["lane_a"]["gameId"] != states["lane_b"]["gameId"]
        assert states["lane_a"]["initialHands"] != states["lane_b"][
            "initialHands"
        ]
        assert [
            play["card"]
            for trick in states["lane_a"]["completedTricks"]
            for play in trick["plays"]
        ] != [
            play["card"]
            for trick in states["lane_b"]["completedTricks"]
            for play in trick["plays"]
        ]

        compiler = OrToolsCpSatPlayCompilerV1()
        plans = {
            lane: compiler.verify_trace(
                state,
                rules=container.research.simulator.rules,
            )
            for lane, state in states.items()
        }
        project_plans = {
            lane: container.research.project_compiler.verify_trace(
                state,
                rules=container.research.simulator.rules,
            )
            for lane, state in states.items()
        }
        assert plans["lane_a"].sha256 != plans["lane_b"].sha256
        for lane, game in comparison_outputs.items():
            state = states[lane]
            assert container.research._compiled_play_mismatches(
                plans[lane],
                game,
                project_plan=project_plans[lane],
                project_domain=container.research.simulator.rules.project_domain,
                scoring_domain=ScoringDomainIRV1.from_rules(
                    container.research.simulator.rules.data
                ),
                structural_domain=container.research.simulator.structural_domain,
                expected_dealer=state["generation"]["dealer"],
                expected_round_number=state["generation"]["roundNumber"],
                expected_next_dealer=(
                    container.research.simulator.structural_domain.next_dealer(
                        Seat.parse(state["dealer"])
                    )
                ),
                starting_match_score=tuple(state["generation"]["matchScore"]),
            ) == ()
    finally:
        container.shutdown()


def test_same_frozen_game_requires_exact_trick_round_match_and_lifecycle_parity(
    monkeypatch: pytest.MonkeyPatch,
):
    container = build_container()
    try:
        scoring_domain = ScoringDomainIRV1.from_rules(
            container.research.simulator.rules.data
        )
        starting_match_score = (
            scoring_domain.match_target - 1,
            scoring_domain.match_target - 1,
        )
        simulation = container.research.simulator.simulate(
            1,
            dealer=Seat.EAST,
            match_score=starting_match_score,
        )
        state = simulation.final_trace.trusted_final_state
        assert state["matchWinner"] is not None
        assert state["roundScore"]["matchWinner"] == state["matchWinner"]
        plan = OrToolsCpSatPlayCompilerV1().verify_trace(
            state,
            rules=container.research.simulator.rules,
        )
        project_plan = container.research.project_compiler.verify_trace(
            state,
            rules=container.research.simulator.rules,
        )
        parity_arguments = {
            "project_plan": project_plan,
            "project_domain": container.research.simulator.rules.project_domain,
            "scoring_domain": scoring_domain,
            "structural_domain": container.research.simulator.structural_domain,
            "expected_dealer": Seat.EAST,
            "expected_round_number": state["generation"]["roundNumber"],
            "expected_next_dealer": (
                container.research.simulator.structural_domain.next_dealer(
                    Seat.EAST
                )
            ),
            "starting_match_score": starting_match_score,
        }
        assert container.research._compiled_play_mismatches(
            plan,
            simulation,
            **parity_arguments,
        ) == ()

        deck_order = tuple(state["shuffle"]["deckOrder"])
        deck_commitment = state["shuffle"]["deckCommitment"]
        candidate_hash = "c" * 64
        constructed_state = copy.deepcopy(state)
        constructed_state["shuffle"]["algorithm"] = "governed_preordered_v1"
        constructed_state["generation"]["dealGenerator"] = {
            "deckCommitment": deck_commitment,
            "candidateHash": candidate_hash,
        }
        constructed_simulation = SimpleNamespace(
            final_trace=SimpleNamespace(trusted_final_state=constructed_state)
        )
        constructed_arguments = {
            **parity_arguments,
            "expected_deck_order": deck_order,
            "expected_deck_commitment": deck_commitment,
            "expected_candidate_hash": candidate_hash,
        }
        assert container.research._compiled_play_mismatches(
            plan,
            constructed_simulation,
            **constructed_arguments,
        ) == ()

        deck_variants = {}
        wrong_commitment = copy.deepcopy(constructed_state)
        wrong_commitment["shuffle"]["deckCommitment"] = "0" * 64
        deck_variants["deck_commitment"] = wrong_commitment

        wrong_candidate = copy.deepcopy(constructed_state)
        wrong_candidate["generation"]["dealGenerator"]["candidateHash"] = (
            "d" * 64
        )
        deck_variants["candidate_hash"] = wrong_candidate

        wrong_hand = copy.deepcopy(constructed_state)
        north_card = wrong_hand["initialHands"]["north"][0]
        west_card = wrong_hand["initialHands"]["west"][0]
        wrong_hand["initialHands"]["north"][0] = west_card
        wrong_hand["initialHands"]["west"][0] = north_card
        deck_variants["initial_hands"] = wrong_hand
        for label, tampered_state in deck_variants.items():
            tampered = SimpleNamespace(
                final_trace=SimpleNamespace(trusted_final_state=tampered_state)
            )
            assert container.research._compiled_play_mismatches(
                plan,
                tampered,
                **constructed_arguments,
            ) == ("compiler_parity_defect",), label

        wrong_match_winner = (
            "west_east"
            if state["matchWinner"] != "west_east"
            else "north_south"
        )
        variants = {}

        trick_score = copy.deepcopy(state)
        trick_score["completedTricks"][0]["rawPoints"] += 1
        variants["trick_raw_points"] = trick_score

        ekka_lead = copy.deepcopy(state)
        ekka_lead["completedTricks"][0]["plays"][0]["ekka"] = not bool(
            ekka_lead["completedTricks"][0]["plays"][0].get("ekka", False)
        )
        variants["trick_ekka_lead"] = ekka_lead

        project_score = copy.deepcopy(state)
        project_score["roundScore"]["projectPoints"][0] += 1
        variants["project_points"] = project_score

        project_team = copy.deepcopy(state)
        project_team["projects"][0]["team"] = (
            "west_east"
            if project_team["projects"][0]["team"] == "north_south"
            else "north_south"
        )
        variants["project_team"] = project_team

        project_kind = copy.deepcopy(state)
        project_kind["projects"][0]["kind"] += "_tampered"
        variants["project_kind"] = project_kind

        revealed_project_index = next(
            index
            for index, project in enumerate(state["projects"])
            if project["revealed"]
        )
        project_card = copy.deepcopy(state)
        project_card["projects"][revealed_project_index]["cards"][0] = {
            "id": (
                project_card["projects"][revealed_project_index]["cards"][0][
                    "id"
                ]
                + 1
            )
            % 32
        }
        variants["project_card"] = project_card

        project_declaration = copy.deepcopy(state)
        project_declaration["projects"][0]["declarationOrder"] += 1
        variants["project_declaration"] = project_declaration

        project_scoring = copy.deepcopy(state)
        project_scoring["projects"][0]["scoring"] = not bool(
            project_scoring["projects"][0]["scoring"]
        )
        variants["project_scoring"] = project_scoring

        project_winner = copy.deepcopy(state)
        project_winner["projectWinningTeam"] = (
            "west_east"
            if project_winner["projectWinningTeam"] == "north_south"
            else "north_south"
        )
        variants["project_winner"] = project_winner

        baloot_teams = copy.deepcopy(state)
        baloot_teams["balootTeams"] = (
            [] if baloot_teams["balootTeams"] else ["north_south"]
        )
        variants["baloot_teams"] = baloot_teams

        round_score = copy.deepcopy(state)
        round_score["roundScore"]["awardedPoints"][0] += 1
        variants["round_awarded_points"] = round_score

        final_match_score = copy.deepcopy(state)
        final_match_score["matchScore"][0] += 1
        variants["final_match_score"] = final_match_score

        round_match_winner = copy.deepcopy(state)
        round_match_winner["roundScore"]["matchWinner"] = wrong_match_winner
        variants["round_match_winner"] = round_match_winner

        final_match_winner = copy.deepcopy(state)
        final_match_winner["matchWinner"] = wrong_match_winner
        variants["final_match_winner"] = final_match_winner

        dealer = copy.deepcopy(state)
        dealer["dealer"] = "west"
        variants["dealer"] = dealer

        phase = copy.deepcopy(state)
        phase["phase"] = "playing"
        variants["terminal_phase"] = phase

        current_player = copy.deepcopy(state)
        current_player["currentPlayer"] = "north"
        variants["terminal_current_player"] = current_player

        round_number = copy.deepcopy(state)
        round_number["roundNumber"] += 1
        variants["round_number"] = round_number

        play_actor = copy.deepcopy(state)
        play_actor["completedTricks"][0]["plays"][0]["seat"] = "west"
        variants["play_actor"] = play_actor

        legal_cards = copy.deepcopy(state)
        legal_cards["completedTricks"][0]["plays"][0]["legalCards"] = []
        variants["play_legal_cards"] = legal_cards

        round_created = copy.deepcopy(state)
        round_created["events"][0]["matchScore"][0] += 1
        variants["round_created"] = round_created

        round_scored = copy.deepcopy(state)
        round_scored["events"][-1]["matchScore"][0] += 1
        variants["round_scored"] = round_scored

        explanation = copy.deepcopy(state)
        explanation["roundScore"]["explanation"][0] += "_tampered"
        variants["round_explanation"] = explanation

        reveal_event = copy.deepcopy(state)
        reveal = next(
            event
            for event in reveal_event["events"]
            if event["type"] == "projects_revealed"
        )
        reveal["owner"] = "west" if reveal["owner"] != "west" else "north"
        variants["project_reveal_event"] = reveal_event

        verifier = OrToolsCpSatPlayCompilerV1()
        monkeypatch.setattr(verifier, "compile", lambda **_kwargs: plan)
        with pytest.raises(AssertionError, match="trick identity"):
            verifier.verify_trace(
                variants["trick_raw_points"],
                rules=container.research.simulator.rules,
            )
        with pytest.raises(AssertionError, match="final match score"):
            verifier.verify_trace(
                variants["final_match_score"],
                rules=container.research.simulator.rules,
            )

        for label, tampered_state in variants.items():
            tampered = SimpleNamespace(
                final_trace=SimpleNamespace(trusted_final_state=tampered_state)
            )
            assert container.research._compiled_play_mismatches(
                plan,
                tampered,
                **parity_arguments,
            ) == ("compiler_parity_defect",), label

        changed_project = replace(
            project_plan.projects[0],
            category_priority=project_plan.projects[0].category_priority + 1,
        )
        changed_plan = replace(
            project_plan,
            projects=(changed_project, *project_plan.projects[1:]),
        )
        changed_payload = changed_plan.to_dict()
        changed_payload.pop("sha256")
        changed_plan = replace(
            changed_plan,
            sha256=canonical_sha256(changed_payload),
        )
        assert container.research._compiled_play_mismatches(
            plan,
            simulation,
            **{**parity_arguments, "project_plan": changed_plan},
        ) == ("compiler_parity_defect",)
    finally:
        container.shutdown()


def test_cross_seat_card_tamper_is_semantically_infeasible():
    container = build_container()
    state = copy.deepcopy(_trace(5))
    first = state["completedTricks"][0]["plays"][0]["card"]
    second = state["completedTricks"][0]["plays"][1]["card"]
    state["completedTricks"][0]["plays"][0]["card"] = second
    state["completedTricks"][0]["plays"][1]["card"] = first
    with pytest.raises(ValueError, match="semantically infeasible"):
        OrToolsCpSatPlayCompilerV1().verify_trace(
            state, rules=container.research.simulator.rules
        )


def test_compiler_constructs_a_complete_legal_line_without_runtime_playback():
    container = build_container()
    state = _trace(8)
    hands = {
        Seat.parse(seat): tuple(Card.parse(card) for card in cards)
        for seat, cards in state["initialHands"].items()
    }
    plan = OrToolsCpSatPlayCompilerV1().compile(
        hands=hands,
        contract=Contract.parse(state["contract"]),
        trump=Suit.parse(state["trumpSuit"]),
        first_leader=Seat.parse(state["firstLeader"]),
        declarer=Seat.parse(state["declarer"]),
        rules=container.research.simulator.rules,
        timeout_seconds=5,
        maximum_conflicts=250_000,
    )
    repeated = OrToolsCpSatPlayCompilerV1().compile(
        hands=hands,
        contract=Contract.parse(state["contract"]),
        trump=Suit.parse(state["trumpSuit"]),
        first_leader=Seat.parse(state["firstLeader"]),
        declarer=Seat.parse(state["declarer"]),
        rules=container.research.simulator.rules,
        timeout_seconds=5,
        maximum_conflicts=250_000,
    )
    assert plan.solver_status == "optimal"
    assert repeated == plan
    assert len(plan.tricks) == 8
    assert sorted(card.id for trick in plan.tricks for card in trick.cards) == list(range(32))
    assert sum(plan.raw_card_points) == int(
        container.research.simulator.rules.data["scoring"]["raw_card_totals"]["hokum"]
    )


@pytest.mark.parametrize("include_round_context", [False, True])
def test_authoritative_legal_line_is_hard_bound_for_cp_sat_verification(
    monkeypatch: pytest.MonkeyPatch,
    include_round_context: bool,
):
    from ortools.sat.python import cp_model

    container = build_container()
    try:
        state = container.research.simulator.simulate(
            5
        ).final_trace.trusted_final_state
        hands = {
            Seat.parse(seat): tuple(Card.parse(card) for card in cards)
            for seat, cards in state["initialHands"].items()
        }
        models = []
        solver_type = cp_model.CpSolver

        class CapturingSolver(solver_type):
            def solve(self, model, *args, **kwargs):
                models.append(model)
                return super().solve(model, *args, **kwargs)

        monkeypatch.setattr(cp_model, "CpSolver", CapturingSolver)
        context = (
            {
                "dealer": Seat.parse(state["dealer"]),
                "buyer": Seat.parse(state["buyer"]),
            }
            if include_round_context
            else {"first_leader": Seat.parse(state["firstLeader"])}
        )
        plan = OrToolsCpSatPlayCompilerV1().compile(
            hands=hands,
            contract=Contract.parse(state["contract"]),
            trump=Suit.parse(state["trumpSuit"]),
            declarer=Seat.parse(state["declarer"]),
            rules=container.research.simulator.rules,
            **context,
            timeout_seconds=5.0,
            maximum_conflicts=250_000,
        )

        complete_play_models = [
            model
            for model in models
            if any(
                variable.name.startswith("ordered_card_t")
                for variable in model.proto.variables
            )
        ]
        assert len(complete_play_models) == 1
        model = complete_play_models[0]
        ordered_card_indexes = {
            index
            for index, variable in enumerate(model.proto.variables)
            if variable.name.startswith("ordered_card_t")
        }
        fixed_indexes = {
            constraint.linear.vars[0]
            for constraint in model.proto.constraints
            if len(constraint.linear.vars) == 1
            and list(constraint.linear.coeffs) == [1]
            and len(constraint.linear.domain) == 2
            and constraint.linear.domain[0] == constraint.linear.domain[1]
        }
        assert len(ordered_card_indexes) == 32
        assert ordered_card_indexes <= fixed_indexes
        assert plan.solver_status == "optimal"
        assert len(plan.tricks) == 8
        assert sorted(
            card.id for trick in plan.tricks for card in trick.cards
        ) == list(range(32))
    finally:
        container.shutdown()


def test_compiler_constructs_a_deterministic_small_card_tactical_precursor():
    container = build_container()
    state = _trace(8)
    hands = {
        Seat.parse(seat): tuple(Card.parse(card) for card in cards)
        for seat, cards in state["initialHands"].items()
    }
    precursor = {
        "kind": "small_card_big_gain",
        "targetSeat": state["firstLeader"],
        "maximumPayoffTricks": 4,
        "minimumLaterRawPoints": 10,
    }
    plans = [
        OrToolsCpSatPlayCompilerV1().compile(
            hands=hands,
            contract=Contract.parse(state["contract"]),
            trump=Suit.parse(state["trumpSuit"]),
            first_leader=Seat.parse(state["firstLeader"]),
            declarer=Seat.parse(state["declarer"]),
            rules=container.research.simulator.rules,
            tactical_precursor=precursor,
            timeout_seconds=30,
            maximum_conflicts=1_000_000,
        )
        for _ in range(2)
    ]
    assert plans[0] == plans[1]
    assert plans[0].solver_status == "optimal"
    evidence = plans[0].tactical_precursor
    assert evidence is not None
    assert evidence["kind"] == "small_card_big_gain"
    assert evidence["structuralPrecursorOnly"] is True
    assert evidence["actorSafe"] is False
    assert evidence["qualificationBypass"] is False
    assert evidence["witnesses"]
    assert evidence["contract"] == state["contract"]
    assert evidence["trump"] == state["trumpSuit"]
    assert evidence["strengthBand"]["kind"] == "governed_bottom_three_v1"
    assert evidence["strengthBand"]["playDomainIrHash"] == (
        container.research.simulator.rules.play_domain.sha256
    )
    for witness in evidence["witnesses"]:
        setup_card = Card.parse(witness["setupCard"])
        governed_strength = container.research.simulator.rules.play_domain.strength(
            setup_card,
            Contract.parse(state["contract"]),
            Suit.parse(state["trumpSuit"]),
        )
        assert witness["setupCardStrength"] == governed_strength
        assert governed_strength <= evidence["strengthBand"]["maximum"]
        if setup_card.suit == Suit.parse(state["trumpSuit"]):
            assert setup_card.rank.name.lower() not in {"nine", "jack"}
