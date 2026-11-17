from dataclasses import replace

import pytest

from baloot_platform.analysis.hidden_deals import (
    _build_start,
    _build_start_reference,
    _should_stop_hidden_deal_generation,
    generate_hidden_deals,
)
from baloot_platform.analysis.models import InformationState
from baloot_platform.core.bots import BotProfile, HeuristicBot
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, Seat


def _unrevealed_project_state(rules):
    engine = BalootEngine(rules)
    state = engine.create_round(seed=1)
    engine.shuffle_and_initial_deal(state)
    engine.bid_and_complete_deal(
        state,
        forced_contract=Contract.SUN,
        forced_buyer=Seat.SOUTH,
    )
    engine.prepare_projects(state)
    bot = HeuristicBot(engine, BotProfile(state.seed))
    while len(state.played_cards) < 4:
        engine.play_card(state, bot.choose(state))
    info = InformationState.from_complete(state, state.current_player)
    assert any(project.declared and not project.revealed for project in info.projects)
    return engine, info


def test_hidden_deals_condition_on_unrevealed_public_project(rules):
    engine, info = _unrevealed_project_state(rules)

    result = generate_hidden_deals(
        info,
        engine,
        requested_samples=4,
        exact_allocation_limit=100,
        maximum_attempts=20,
        seed=123,
    )

    assert result.method == "constructive_uniform_project_conditioned_hidden_deals"
    assert len(result.states) == 4
    expected = tuple(project.to_dict() for project in info.projects)
    for state in result.states:
        projection = InformationState.from_complete(state, info.observer)
        assert tuple(project.to_dict() for project in projection.projects) == expected
        assert projection.observer_hand == info.observer_hand


def test_project_conditioning_is_deterministic(rules):
    engine, info = _unrevealed_project_state(rules)
    kwargs = dict(
        requested_samples=4,
        exact_allocation_limit=100,
        maximum_attempts=20,
        seed=9876,
    )

    first = generate_hidden_deals(info, engine, **kwargs)
    second = generate_hidden_deals(info, engine, **kwargs)

    assert [state.initial_hands for state in first.states] == [
        state.initial_hands for state in second.states
    ]
    assert first.attempted == second.attempted


def test_exhaustive_project_conditioned_cohort_is_not_truncated():
    assert not _should_stop_hidden_deal_generation(
        "exhaustive_project_conditioned_hidden_deals",
        accepted_count=1,
        requested_samples=1,
    )
    assert not _should_stop_hidden_deal_generation(
        "exhaustive_constrained_hidden_deals",
        accepted_count=1,
        requested_samples=1,
    )
    assert _should_stop_hidden_deal_generation(
        "constructive_uniform_project_conditioned_hidden_deals",
        accepted_count=1,
        requested_samples=1,
    )


def test_impossible_public_project_fails_closed(rules):
    engine, info = _unrevealed_project_state(rules)
    impossible = replace(
        info,
        projects=(replace(info.projects[0], kind="unsupported_project"),),
    )

    with pytest.raises(
        RuntimeError,
        match="public project history leaves no feasible hidden allocation",
    ):
        generate_hidden_deals(
            impossible,
            engine,
            requested_samples=1,
            exact_allocation_limit=100,
            maximum_attempts=20,
            seed=123,
        )


def test_bounded_hidden_state_reconstruction_matches_full_replay(rules):
    engine = BalootEngine(rules)
    state = engine.create_round(seed=24021, dealer=Seat.EAST)
    engine.shuffle_and_initial_deal(state, cut_position=7)
    engine.bid_and_complete_deal(
        state,
        forced_contract=Contract.HOKUM,
        forced_buyer=Seat.SOUTH,
    )
    engine.prepare_projects(state)
    bot = HeuristicBot(engine, BotProfile(state.seed))
    while len(state.played_cards) < 25:
        engine.play_card(state, bot.choose(state))

    info = InformationState.from_complete(state, state.current_player)
    hands = {seat: state.hand(seat) for seat in Seat}
    reference = _build_start_reference(info, hands, engine)
    accelerated = _build_start(info, hands, engine)

    assert accelerated.hands == reference.hands
    assert accelerated.initial_hands == reference.initial_hands
    assert [trick.public_dict() for trick in accelerated.completed_tricks] == [
        trick.public_dict() for trick in reference.completed_tricks
    ]
    assert accelerated.current_trick.public_dict() == reference.current_trick.public_dict()
    assert [project.to_dict() for project in accelerated.projects] == [
        project.to_dict() for project in reference.projects
    ]
    assert accelerated.baloot_teams == reference.baloot_teams
    assert accelerated.trump_broken == reference.trump_broken
    assert engine.legal_moves(accelerated) == engine.legal_moves(reference)

    accelerated_line = []
    reference_line = []
    accelerated_bot = HeuristicBot(engine, BotProfile(1))
    reference_bot = HeuristicBot(engine, BotProfile(1))
    while accelerated.current_player is not None:
        card = accelerated_bot.choose(accelerated)
        accelerated_line.append(card.code)
        engine.play_card(accelerated, card, emit_events=False)
    while reference.current_player is not None:
        card = reference_bot.choose(reference)
        reference_line.append(card.code)
        engine.play_card(reference, card, emit_events=False)
    assert accelerated_line == reference_line
    assert accelerated.round_score.to_dict() == reference.round_score.to_dict()
    assert accelerated.match_score == reference.match_score
