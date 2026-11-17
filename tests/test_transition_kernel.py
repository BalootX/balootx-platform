from __future__ import annotations

import re

from baloot_platform.core.bots import BotProfile, HeuristicBot
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, Seat, Suit
from baloot_platform.core.models import RoundState


def _active(rules):
    engine = BalootEngine(rules)
    state = engine.create_round(seed=21001, dealer=Seat.EAST)
    engine.shuffle_and_initial_deal(state, cut_position=7)
    engine.bid_and_complete_deal(
        state,
        forced_contract=Contract.HOKUM,
        forced_buyer=Seat.SOUTH,
        forced_trump=Suit.HEARTS,
    )
    engine.prepare_projects(state)
    return engine, state


def test_reference_apply_undo_restores_byte_identical_canonical_state(rules):
    engine, state = _active(rules)
    before = engine.canonical_state(state)
    before_hash = engine.state_hash(state)
    action = engine.legal_actions(state)[0]
    token = engine.apply(state, action)
    assert engine.state_hash(state) != before_hash
    engine.undo(state, token)
    assert engine.canonical_state(state) == before
    assert engine.state_hash(state) == before_hash


def test_reference_apply_undo_does_not_clone_full_round(
    rules,
    monkeypatch,
):
    engine, state = _active(rules)

    def reject_clone(_state):
        raise AssertionError("full RoundState.clone is forbidden in apply/undo")

    monkeypatch.setattr(RoundState, "clone", reject_clone)
    token = engine.apply(state, engine.legal_actions(state)[0])
    engine.undo(state, token)


def test_search_transition_can_suppress_only_transient_events(rules):
    engine, state = _active(rules)
    recorded = state.clone()
    silent = state.clone()
    action = engine.legal_actions(recorded)[0]

    recorded_token = engine.apply(
        recorded,
        action,
        verify_identity=False,
    )
    silent_token = engine.apply(
        silent,
        action,
        verify_identity=False,
        emit_events=False,
    )
    assert len(recorded.events) > len(state.events)
    assert len(silent.events) == len(state.events)

    recorded_value = recorded.to_dict(reveal_hands=True)
    silent_value = silent.to_dict(reveal_hands=True)
    recorded_value.pop("events")
    silent_value.pop("events")
    assert recorded_value == silent_value
    engine.undo(recorded, recorded_token)
    engine.undo(silent, silent_token)
    assert engine.canonical_state(recorded) == engine.canonical_state(state)
    assert engine.canonical_state(silent) == engine.canonical_state(state)


def test_heuristic_ranking_does_not_build_hypothetical_events(
    rules,
    monkeypatch,
):
    engine, state = _active(rules)
    first = engine.legal_actions(state)[0]
    engine.play_card(state, first)
    bot = HeuristicBot(engine, BotProfile(deterministic_seed=state.seed))

    def reject_emit(_state, _event_type, **_data):
        raise AssertionError("hypothetical ranking emitted an event")

    monkeypatch.setattr(RoundState, "emit", reject_emit)
    assert bot.choose(state) in engine.legal_actions(state)


def test_reference_apply_undo_restores_trick_and_terminal_boundaries(rules):
    engine, state = _active(rules)
    bot = HeuristicBot(engine, BotProfile(deterministic_seed=state.seed))
    for _ in range(3):
        engine.play_card(state, bot.choose(state))
    before_trick = engine.canonical_state(state)
    token = engine.apply(state, bot.choose(state))
    engine.undo(state, token)
    assert engine.canonical_state(state) == before_trick

    while sum(mask.bit_count() for mask in state.hands) > 1:
        engine.play_card(state, bot.choose(state))
    before_terminal = engine.canonical_state(state)
    token = engine.apply(state, bot.choose(state))
    assert engine.terminal_result(state) is not None
    engine.undo(state, token)
    assert engine.canonical_state(state) == before_terminal
    assert engine.terminal_result(state) is None


def test_reference_transition_rejects_illegal_action(rules):
    engine, state = _active(rules)
    legal = {card.id for card in engine.legal_actions(state)}
    illegal = next(card for card in range(32) if card not in legal)
    before = engine.canonical_state(state)
    try:
        engine.apply(state, illegal)
    except ValueError:
        pass
    else:
        raise AssertionError("illegal action was accepted")
    assert engine.canonical_state(state) == before


def test_reference_apply_is_atomic_when_play_raises_after_mutation(
    rules,
    monkeypatch,
):
    engine, state = _active(rules)
    action = engine.legal_actions(state)[0]
    before = engine.canonical_state(state)

    def broken_play(target, card, **_kwargs):
        target.hands[int(target.current_player)] ^= 1 << card.id
        target.current_player = target.current_player.next()
        raise RuntimeError("injected post-mutation failure")

    monkeypatch.setattr(engine, "play_card", broken_play)
    try:
        engine.apply(state, action)
    except RuntimeError as exc:
        assert "post-mutation" in str(exc)
    else:
        raise AssertionError("injected transition failure was swallowed")
    assert engine.canonical_state(state) == before


def test_reference_transition_classifies_malformed_actions(rules):
    engine, state = _active(rules)
    before = engine.canonical_state(state)
    for malformed in ("", "invalid-card", {}, {"suit": "x", "rank": "ace"}):
        try:
            engine.apply(state, malformed)
        except ValueError as exc:
            assert "invalid card" in str(exc)
        else:
            raise AssertionError(f"malformed action was accepted: {malformed!r}")
        assert engine.canonical_state(state) == before


def test_invalid_ekka_rejection_does_not_mutate_state(rules):
    engine, state = _active(rules)
    action = next(
        card
        for card in engine.legal_actions(state)
        if not engine._can_declare_ekka(state, state.current_player, card)
    )
    before = engine.canonical_state(state)
    try:
        engine.apply(state, action, ekka=True)
    except ValueError as exc:
        assert "Ekka may only" in str(exc)
    else:
        raise AssertionError("invalid Ekka was accepted")
    assert engine.canonical_state(state) == before


def test_undo_token_is_bound_to_exact_post_action_state(rules):
    engine, state = _active(rules)
    first = engine.legal_actions(state)[0]
    token = engine.apply(state, first)
    second = engine.legal_actions(state)[0]
    engine.play_card(state, second)
    try:
        engine.undo(state, token)
    except ValueError as exc:
        assert "does not match current state" in str(exc)
    else:
        raise AssertionError("stale undo token was accepted")


def test_observer_view_discloses_only_observer_hand(rules):
    engine, state = _active(rules)
    view = engine.observer_view(state, Seat.SOUTH)
    assert len(view["hands"]["south"]) == 8
    for seat in ("north", "west", "east"):
        assert view["hands"][seat] == {"count": 8}
    encoded = str(view)
    hidden_codes = {
        card.code
        for seat in (Seat.NORTH, Seat.WEST, Seat.EAST)
        for card in state.hand(seat)
    }
    assert not any(
        re.search(rf"'code': '{re.escape(code)}'", encoded)
        for code in hidden_codes
    )


def test_observer_view_hides_undeclared_opponent_projects(rules):
    engine, state = _active(rules)
    hidden = [
        project
        for project in state.projects
        if project.owner != Seat.SOUTH
        and not project.declared
        and not project.revealed
    ]
    if not hidden:
        state = engine.create_round(seed=22_019, dealer=Seat.EAST)
        engine.shuffle_and_initial_deal(state, cut_position=17)
        engine.bid_and_complete_deal(
            state,
            forced_contract=Contract.SUN,
            forced_buyer=Seat.SOUTH,
        )
        engine.prepare_projects(state)
        hidden = [
            project
            for project in state.projects
            if project.owner != Seat.SOUTH
            and not project.declared
            and not project.revealed
        ]
    assert hidden

    view = engine.observer_view(state, Seat.SOUTH)
    visible = {
        (project["owner"], project["kind"], project["highRank"])
        for project in view["projects"]
    }
    for project in hidden:
        assert (
            project.owner.name.lower(),
            project.kind,
            project.high_rank,
        ) not in visible
    assert view["projectWinningTeam"] is None


def test_observer_view_hides_historical_legal_choices(rules):
    engine, state = _active(rules)
    for _ in range(6):
        legal = engine.legal_actions(state)
        engine.play_card(state, legal[-1])

    view = engine.observer_view(state, state.current_player)
    assert "seed" not in view
    assert "initialHands" not in view
    assert "seed" not in view["shuffle"]
    assert "legalCards" not in str(view["currentTrick"])
    assert "legalCards" not in str(view["completedTricks"])
    assert "legalCards" not in str(view["events"])
    assert not any("seed" in event for event in view["events"])
    assert not any(
        event["type"] == "project_holdings_detected"
        for event in view["events"]
    )


def test_terminal_result_is_none_until_round_completion(rules):
    engine, state = _active(rules)
    assert engine.terminal_result(state) is None
    bot = HeuristicBot(engine, BotProfile(deterministic_seed=state.seed))
    while state.current_player is not None:
        engine.play_card(state, bot.choose(state))
    result = engine.terminal_result(state)
    assert result is not None
    assert result["roundScore"]
    assert len(result["stateSha256"]) == 64


def test_100_seeded_rounds_replay_identically_and_conserve_cards(rules):
    engine = BalootEngine(rules)
    for seed in range(22000, 22100):
        left = engine.create_round(seed=seed, dealer=Seat(seed % 4))
        right = engine.create_round(seed=seed, dealer=Seat(seed % 4))
        for state in (left, right):
            engine.shuffle_and_initial_deal(
                state, cut_position=(seed % 31) + 1
            )
            engine.bid_and_complete_deal(
                state,
                forced_contract=Contract.HOKUM,
                forced_buyer=Seat.SOUTH,
                forced_trump=Suit(seed % 4),
            )
            engine.prepare_projects(state)
        bots = (
            HeuristicBot(engine, BotProfile(deterministic_seed=seed)),
            HeuristicBot(engine, BotProfile(deterministic_seed=seed)),
        )
        while left.current_player is not None:
            left_action = bots[0].choose(left)
            right_action = bots[1].choose(right)
            assert left_action == right_action
            engine.play_card(left, left_action)
            engine.play_card(right, right_action)
            left.validate_conservation()
            right.validate_conservation()
            assert engine.state_hash(left) == engine.state_hash(right)
        assert engine.terminal_result(left) == engine.terminal_result(right)
