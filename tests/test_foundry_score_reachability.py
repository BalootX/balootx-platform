from __future__ import annotations

from dataclasses import replace

import pytest

from baloot_platform.challenge.replay import CompleteTraceReplayerV1
from baloot_platform.challenge.score_reachability import (
    AuthoritativeRoundOutcomeCatalogV1,
    AuthoritativeScoreReachabilityV1,
    ScoreReachabilityCatalogFactoryV1,
    ScoreReachabilityProofV1,
    ScoreReachabilityProfileV1,
    VerifiedEnumeratedContextV1,
)
from baloot_platform.challenge.simulation import NaturalLaffaSimulatorV1
from baloot_platform.core.enums import Seat


def _profile():
    return ScoreReachabilityProfileV1.build({
        "schemaVersion": 1,
        "id": "authoritative_score_reachability_test_v1",
        "revision": "1",
        "maximumWitnessTraces": 100,
        "witnessesPerOutcome": 4,
        "maximumRounds": 20,
        "maximumStates": 100000,
    })


def _played(simulator, seed, dealer):
    result = simulator.simulate(
        seed,
        dealer=dealer,
        match_score=(0, 0),
        round_number=1,
    )
    assert result.publishable_laffa
    return result.final_trace


def test_normal_round_action_line_replays_under_another_pre_target_score(root):
    simulator = NaturalLaffaSimulatorV1.from_root(root)
    trace = _played(simulator, 1, Seat.EAST)
    original = tuple(trace.trusted_final_state["roundScore"]["awardedPoints"])
    proof = CompleteTraceReplayerV1(simulator.rules).replay_in_context(
        trace,
        match_score=(10, 20),
        round_number=4,
    )
    assert proof.awarded_points == original
    assert proof.ending_score == (10 + original[0], 20 + original[1])
    assert proof.dealer == Seat.EAST
    assert proof.next_dealer == Seat.NORTH
    assert proof.round_number == 4


def test_reachability_builds_exact_dealer_and_score_path_from_engine_witnesses(root):
    simulator = NaturalLaffaSimulatorV1.from_root(root)
    east = _played(simulator, 1, Seat.EAST)
    north = _played(simulator, 2, Seat.NORTH)
    east_points = tuple(east.trusted_final_state["roundScore"]["awardedPoints"])
    north_points = tuple(north.trusted_final_state["roundScore"]["awardedPoints"])
    goal = (east_points[0] + north_points[0], east_points[1] + north_points[1])
    assert max(goal) < 152
    catalog = AuthoritativeRoundOutcomeCatalogV1.build(
        (east, north),
        rules=simulator.rules,
        profile=_profile(),
    )
    engine = AuthoritativeScoreReachabilityV1(catalog, simulator.rules, _profile())
    factory_engine = ScoreReachabilityCatalogFactoryV1(
        simulator.rules, _profile()
    ).build((east, north))
    assert factory_engine.catalog.sha256 == catalog.sha256
    first = engine.prove(
        goal,
        initial_dealer=Seat.EAST,
        target_dealer=Seat.WEST,
    )
    second = engine.prove(
        goal,
        initial_dealer=Seat.EAST,
        target_dealer=Seat.WEST,
    )
    assert first.sha256 == second.sha256
    assert len(first.steps) == 2
    assert len({item.deal_commitment for item in first.steps}) == 2
    assert first.steps[0].dealer == Seat.EAST
    assert first.steps[1].dealer == Seat.NORTH
    assert first.steps[-1].ending_score == goal
    assert first.round_number == 3
    verified = VerifiedEnumeratedContextV1.build(first)
    assert verified.context.starting_score == goal
    assert verified.context.dealer == Seat.WEST
    assert verified.context.reachability_mode == "verified_enumeration"
    assert verified.context.reachability_proof_hash == first.sha256
    verified.verify()
    restored = ScoreReachabilityProofV1.from_dict(first.to_dict())
    assert restored == first

    with pytest.raises(ValueError, match="does not reach"):
        VerifiedEnumeratedContextV1.build(
            replace(first, target_score=(goal[0] + 1, goal[1]))
        )
    with pytest.raises(ValueError, match="proof hash"):
        VerifiedEnumeratedContextV1.build(replace(first, sha256="0" * 64))


def test_reachability_fails_closed_when_catalog_cannot_supply_dealer_path(root):
    simulator = NaturalLaffaSimulatorV1.from_root(root)
    east = _played(simulator, 1, Seat.EAST)
    points = tuple(east.trusted_final_state["roundScore"]["awardedPoints"])
    catalog = AuthoritativeRoundOutcomeCatalogV1.build(
        (east,),
        rules=simulator.rules,
        profile=_profile(),
    )
    engine = AuthoritativeScoreReachabilityV1(catalog, simulator.rules, _profile())
    with pytest.raises(LookupError, match="no verified score path"):
        engine.prove(
            (points[0] * 2, points[1] * 2),
            initial_dealer=Seat.EAST,
            target_dealer=Seat.WEST,
        )


def test_reachability_rejects_post_target_and_unbounded_profiles(root):
    simulator = NaturalLaffaSimulatorV1.from_root(root)
    east = _played(simulator, 1, Seat.EAST)
    catalog = AuthoritativeRoundOutcomeCatalogV1.build(
        (east,), rules=simulator.rules, profile=_profile()
    )
    engine = AuthoritativeScoreReachabilityV1(catalog, simulator.rules, _profile())
    with pytest.raises(ValueError, match="pre-match score"):
        engine.prove((152, 0), target_dealer=Seat.EAST)
    with pytest.raises(ValueError, match="state bound"):
        ScoreReachabilityProfileV1.build({
            "schemaVersion": 1,
            "id": "bad",
            "revision": "1",
            "maximumWitnessTraces": 1,
            "witnessesPerOutcome": 1,
            "maximumRounds": 1,
            "maximumStates": 0,
        })
