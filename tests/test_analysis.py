from baloot_platform.analysis import evaluator
from baloot_platform.analysis.evaluator import analyze
from baloot_platform.analysis.models import InformationState, SignalObservation
from baloot_platform.core.cards import mask_to_cards
from baloot_platform.core.bots import BotProfile, HeuristicBot
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, Seat, Suit


def _partial(rules):
    e = BalootEngine(rules)
    s = e.create_round(seed=3)
    e.shuffle_and_initial_deal(s)
    e.bid_and_complete_deal(
        s,
        forced_contract=Contract.HOKUM,
        forced_buyer=Seat.SOUTH,
        forced_trump=Suit.HEARTS,
    )
    e.prepare_projects(s)
    bot = HeuristicBot(e, BotProfile(s.seed))
    while len(s.played_cards) < 24:
        e.play_card(s, bot.choose(s))
    return e, s


def test_information_state_hides_other_hands(rules):
    _, s = _partial(rules)
    info = InformationState.from_complete(s, Seat.SOUTH)
    data = info.to_dict()
    assert "hands" not in data and "initialHands" not in data
    assert len(data["observerHand"]) == s.hands[int(Seat.SOUTH)].bit_count()


def test_analysis_returns_legal_actions(rules, convention):
    e, s = _partial(rules)
    info = InformationState.from_complete(s, Seat.SOUTH)
    if s.current_player != Seat.SOUTH:
        while s.current_player != Seat.SOUTH:
            e.play_card(s, HeuristicBot(e, BotProfile(s.seed)).choose(s))
        info = InformationState.from_complete(s, Seat.SOUTH)
    result = analyze(
        info,
        rules,
        convention,
        requested_samples=4,
        exact_allocation_limit=200,
        exact_search_max_remaining=4,
        maximum_sampling_attempts=5000,
        seed=5,
    )
    legal = set(e.legal_moves(s))
    assert result.recommended_card in legal
    assert {a.card for a in result.actions} == legal
    assert result.hidden_deals > 0
    assert {action.search_completion_status for action in result.actions} <= {
        "exact",
        "heuristic_complete",
        "budget_exhausted",
        "cancelled",
        "failed",
    }
    assert all(
        action.to_dict()["searchCompletionStatus"] == action.search_completion_status
        for action in result.actions
    )


def test_hard_facts_only_does_not_feed_display_signals_into_analysis(
    rules, convention, monkeypatch
):
    engine, state = _partial(rules)
    while state.current_player != Seat.SOUTH:
        engine.play_card(state, HeuristicBot(engine, BotProfile(state.seed)).choose(state))
    info = InformationState.from_complete(state, Seat.SOUTH)
    signal = SignalObservation(
        "minimum_winner",
        Seat.SOUTH,
        "self",
        (mask_to_cards(info.observer_hand)[0],),
        0.58,
        "meaningful_choice",
        ("soft display evidence",),
        "Not a hard fact.",
    )
    monkeypatch.setattr(evaluator, "detect_signals", lambda *_args: (signal,))
    common = {
        "requested_samples": 2,
        "exact_allocation_limit": 50,
        "exact_search_max_remaining": 4,
        "maximum_sampling_attempts": 5000,
        "seed": 91,
    }
    shown = analyze(
        info,
        rules,
        {**convention, "display_observations": True},
        **common,
    )
    hidden = analyze(
        info,
        rules,
        {**convention, "display_observations": False},
        **common,
    )
    assert shown.signals == (signal,)
    assert hidden.signals == ()
    assert shown.difficulty == min(100, hidden.difficulty + 8)
    assert shown.recommended_card == hidden.recommended_card
