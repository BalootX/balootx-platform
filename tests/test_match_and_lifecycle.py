from baloot_platform.core.enums import Contract,Seat,Suit
from baloot_platform.core.match_scenario import MatchScenarioRequest,generate_match_episode
from baloot_platform.core.scenario import ScenarioRequest,generate_episode


def test_episode_lifecycle_event_order(rules,convention):
    ep=generate_episode(rules,convention,bundle_hash=rules.sha256,request=ScenarioRequest.governed(seed=31,contract=Contract.HOKUM,buyer=Seat.SOUTH,trump_suit=Suit.HEARTS,require_project=False,requested_samples=2,exact_allocation_limit=100,exact_search_max_remaining=2,max_seed_attempts=10))
    names=[e.type for e in ep.final_state.events]
    required=['round_created','deck_shuffled','purchase_card_exposed','initial_deal_complete','contract_purchased','final_deal_complete','play_started','round_scored']
    indices=[names.index(name) for name in required]
    assert indices==sorted(indices)
    assert sum(e.type=='card_played' for e in ep.final_state.events)==32
    assert sum(e.type=='trick_completed' for e in ep.final_state.events)==8
    assert ep.final_state.deal_trace[0].stage=='initial'


def test_match_episode_progresses_score_and_dealer(rules,convention):
    # Lower target through a cloned rule profile would make a short match, but
    # this test only needs deterministic multi-round continuation.
    match=generate_match_episode(rules,convention,bundle_hash=rules.sha256,request=MatchScenarioRequest.governed(seed=44,maximum_rounds=2,contract=Contract.SUN,require_project=False,requested_samples=2,exact_search_max_remaining=2))
    assert 1<=len(match.rounds)<=2
    if len(match.rounds)==2:
        assert match.rounds[1].final_state.dealer==match.rounds[0].final_state.dealer.next()
        assert tuple(match.rounds[1].state_before_pause.match_score)==tuple(match.rounds[0].final_state.match_score)
    assert match.final_score==tuple(match.rounds[-1].final_state.match_score)
