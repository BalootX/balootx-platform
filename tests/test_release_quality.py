from baloot_platform.core.enums import Contract, Seat, Suit
from baloot_platform.core.scenario import ScenarioRequest, generate_episode


def test_publishable_scenario_has_meaningful_exact_decision(rules, convention):
    episode = generate_episode(
        rules,
        convention,
        bundle_hash=rules.sha256,
        request=ScenarioRequest.governed(
            seed=27,
            dealer=Seat.EAST,
            contract=Contract.HOKUM,
            buyer=Seat.SOUTH,
            trump_suit=Suit.HEARTS,
            require_project=False,
            pause_trick_min=5,
            pause_trick_max=5,
            requested_samples=80,
            exact_allocation_limit=20_000,
            exact_search_max_remaining=10,
            max_seed_attempts=1,
            max_candidates_per_seed=4,
            minimum_hidden_deals=20,
            minimum_expected_margin_gap=3.0,
            minimum_win_probability_gap=0.10,
            require_exact_analysis=True,
            target_debate_score=55,
        ),
    )
    actions = episode.analysis.actions
    assert len(actions) >= 2
    assert episode.analysis.hidden_deals >= 20
    assert min(item.exact_share for item in actions[:2]) == 1.0
    assert actions[0].expected_match_margin - actions[1].expected_match_margin >= 3.0
    assert actions[0].win_probability - actions[1].win_probability >= 0.10
    assert len(episode.final_state.completed_tricks) == 8
    assert len(episode.final_state.played_cards) == 32


def test_studio_guided_mode_has_no_cut_pacing_and_orientation(root):
    html = (root / "src/baloot_platform/service/static/studio.html").read_text(encoding="utf-8")
    assert '<option value="none">تمرير دون قص</option>' in html
    assert 'id="pacing"' in html
    assert 'id="cardOrientation"' in html
    assert 'premium_video' in html
