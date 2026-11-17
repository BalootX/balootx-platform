import pytest

from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.bots import BotProfile, HeuristicBot
from baloot_platform.core.cards import Card, FULL_DECK, cards_to_mask
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, GamePhase, Rank, Seat, Suit, Team
from baloot_platform.core.models import Project, RoundState, Trick
from baloot_platform.core.project_ir import ProjectDomainIRV1
from baloot_platform.core.projects import detect_projects, score_project_team
from baloot_platform.core.scoring import score_round
from baloot_platform.core.scoring_ir import ScoringDomainIRV1
from baloot_platform.core.shuffle import shuffle_and_cut
from baloot_platform.core.validation import validate_completed_round


def _score_round(state, rules):
    return score_round(
        state,
        ScoringDomainIRV1.from_rules(rules.data),
        ProjectDomainIRV1.from_rules(rules.data),
    )


def test_card_deck_identity():
    assert len(FULL_DECK)==32
    assert len({c.code for c in FULL_DECK})==32
    assert Card.parse('H10') == Card.of(Suit.HEARTS, Rank.TEN)
    for card in FULL_DECK:
        expected = f"{Suit(card.id // 8).name[0]}{Rank(card.id % 8).short}"
        assert card.code == card.to_dict()["code"] == expected
        assert Card.parse(expected) == card
    # Preserve legacy constructor behavior outside canonical integer identities.
    assert Card(1.0).code == Card(True).code == Card(1).code


def test_shuffle_reproducible_and_committed():
    a=shuffle_and_cut(44,11,dealer=Seat.EAST)
    b=shuffle_and_cut(44,11,dealer=Seat.EAST)
    assert a.cards==b.cards
    assert a.record.deck_commitment==b.record.deck_commitment
    assert len(set(a.cards))==32
    assert a.record.cutter==Seat.SOUTH


def test_initial_and_final_deal(rules):
    e=BalootEngine(rules);s=e.create_round(seed=8,dealer=Seat.EAST);e.shuffle_and_initial_deal(s,cut_position=9)
    assert [m.bit_count() for m in s.initial_five]==[5]*4
    assert s.purchase_card is not None and len(s.undealt)==11
    assert e.bid_and_complete_deal(s,forced_contract=Contract.HOKUM,forced_buyer=Seat.SOUTH,forced_trump=Suit.HEARTS)
    assert [m.bit_count() for m in s.hands]==[8]*4
    assert s.buyer==Seat.SOUTH and s.first_leader==Seat.NORTH
    assert [r.count if hasattr(r,'count') else len(r.cards) for r in s.deal_trace]


def test_ashkal_partner_buys(rules):
    e=BalootEngine(rules);s=e.create_round(seed=9);e.shuffle_and_initial_deal(s)
    m=BiddingMachine(s.dealer,s.purchase_card.suit,AuctionDomainIRV1.from_rules(rules.data))
    assert m.submit('pass') is None
    assert m.submit('pass') is None
    assert m.submit('ashkal') is None
    # Earlier players receive their published Sun recall opportunity.
    assert m.actor == m.order[1]
    assert m.submit('pass') is None
    result=m.submit('pass')
    assert result.contract==Contract.ASHKAL
    assert result.buyer==m.order[2].partner
    e.complete_deal_from_bid_result(s,result)
    assert s.buyer==result.buyer


def test_second_round_hokum_must_change_suit(rules):
    e=BalootEngine(rules);s=e.create_round(seed=10);e.shuffle_and_initial_deal(s)
    m=BiddingMachine(s.dealer,s.purchase_card.suit,AuctionDomainIRV1.from_rules(rules.data))
    for _ in range(4): m.submit('pass')
    with pytest.raises(ValueError): m.submit('hokum',trump=s.purchase_card.suit)
    different=next(x for x in Suit if x!=s.purchase_card.suit)
    result=m.submit('hokum',trump=different)
    # Hokum candidate stands after the remaining seats complete their turns.
    while result is None:
        result=m.submit('pass')
    assert result.contract==Contract.HOKUM and result.trump==different


def _playing_state(rules, contract=Contract.SUN, trump=None):
    state=RoundState(rules.sha256,1,1,Seat.EAST)
    state.contract=contract;state.buyer=Seat.SOUTH;state.declarer=Seat.SOUTH;state.trump_suit=trump
    state.first_leader=Seat.NORTH;state.current_player=Seat.NORTH;state.phase=GamePhase.PLAYING;state.current_trick=Trick(0,Seat.NORTH)
    return state


def test_sun_must_follow_suit(rules):
    e=BalootEngine(rules);s=_playing_state(rules)
    hands={Seat.NORTH:[Card.parse('S7')],Seat.WEST:[Card.parse('S8'),Card.parse('H7')],Seat.SOUTH:[Card.parse('D7')],Seat.EAST:[Card.parse('C7')]}
    s.hands=[cards_to_mask(hands[x]) for x in Seat]
    e.play_card(s,Card.parse('S7'))
    assert e.legal_moves(s)==(Card.parse('S8'),)


def test_hokum_must_cut_opponent(rules):
    e=BalootEngine(rules);s=_playing_state(rules,Contract.HOKUM,Suit.HEARTS)
    hands={Seat.NORTH:[Card.parse('S7')],Seat.WEST:[Card.parse('H7'),Card.parse('C7')],Seat.SOUTH:[Card.parse('D7')],Seat.EAST:[Card.parse('C8')]}
    s.hands=[cards_to_mask(hands[x]) for x in Seat]
    e.play_card(s,Card.parse('S7'))
    assert e.legal_moves(s)==(Card.parse('H7'),)


def test_locked_hokum_cannot_lead_trump(rules):
    e=BalootEngine(rules);s=_playing_state(rules,Contract.HOKUM,Suit.HEARTS);s.locked=True
    hands={Seat.NORTH:[Card.parse('H7'),Card.parse('S7')],Seat.WEST:[Card.parse('H8')],Seat.SOUTH:[Card.parse('D7')],Seat.EAST:[Card.parse('C7')]}
    s.initial_hands=[cards_to_mask(hands[x]) for x in Seat];s.hands=list(s.initial_hands)
    assert e.legal_moves(s)==(Card.parse('S7'),)


def test_project_detection_sequence_and_four_kind(rules):
    hand=(Card.parse('SA'),Card.parse('SK'),Card.parse('SQ'),Card.parse('SJ'),Card.parse('HA'),Card.parse('DA'),Card.parse('CA'),Card.parse('H7'))
    projects=detect_projects({Seat.NORTH:hand,Seat.WEST:(),Seat.SOUTH:(),Seat.EAST:()},Contract.SUN,rules,Seat.EAST)
    assert any(p.kind=='four_hundred' for p in projects)
    # Four Aces consumes S-A, leaving the non-overlapping S-K/S-Q/S-J Siri.
    assert any(p.kind=='siri' for p in projects)
    assert not any(a.card_mask & b.card_mask for i,a in enumerate(projects) for b in projects[i+1:] if a.owner==b.owner)


def test_project_team_comparison(rules):
    a=Project(Seat.NORTH,'hundred_sequence',(),20,7,1,True,1)
    b=Project(Seat.WEST,'hundred_four_kind',(),20,5,2,True,2)
    updated,winner=score_project_team(
        [a,b], domain=ProjectDomainIRV1.from_rules(rules.data)
    )
    assert winner==Team.WEST_EAST
    assert next(p for p in updated if p.owner==Seat.WEST).scoring


def _score_state(rules, contract, declarer, raw_counting, counting_team, project_points=(0,0), baloot=()):
    s=RoundState(rules.sha256,1,1,Seat.EAST);s.contract=contract;s.declarer=declarer;s.buyer=declarer
    # The published raw total includes the ten-point last-trick bonus.
    total=162 if contract.effective==Contract.HOKUM else 130
    raw_other=total-raw_counting
    counting_winner=Seat.NORTH if counting_team==Team.NORTH_SOUTH else Seat.WEST
    other_winner=Seat.WEST if counting_team==Team.NORTH_SOUTH else Seat.NORTH
    s.completed_tricks=[]

    def split(value, count):
        pieces=[value//count]*count
        pieces[-1]+=value-sum(pieces)
        return pieces

    # Four tricks to the non-counting team, three normal tricks to the
    # counting team, then a zero-card-point final trick to the counting team.
    # score_round() adds the ten-point last-trick bonus itself.
    for i,v in enumerate(split(raw_other,4)):
        s.completed_tricks.append(Trick(i,other_winner,winner=other_winner,raw_points=v))
    for i,v in enumerate(split(raw_counting-10,3),4):
        s.completed_tricks.append(Trick(i,counting_winner,winner=counting_winner,raw_points=v))
    s.completed_tricks.append(Trick(7,counting_winner,winner=counting_winner,raw_points=0))

    s.projects=[]
    for team,value in enumerate(project_points):
        if value: s.projects.append(Project(Seat.NORTH if team==0 else Seat.WEST,'test',(),value,0,0,scoring=True))
    s.baloot_teams=set(baloot);return s


def test_hokum_rounding_example(rules):
    # Declarer E-W, counting N-S takes 77 including last bonus -> 8:8,
    # counting side rounded up and therefore loses the tie.
    s=_score_state(rules,Contract.HOKUM,Seat.EAST,77,Team.NORTH_SOUTH)
    score=_score_round(s,rules)
    assert score.card_game_points==[8,8]
    assert score.winner==Team.WEST_EAST
    assert score.awarded_points==[8,8]


def test_sun_project_example(rules):
    # Declarer N-S; counting E-W 66 -> 14, declarer 12 + 10 project = 22.
    s=_score_state(rules,Contract.SUN,Seat.SOUTH,66,Team.WEST_EAST,project_points=(10,0))
    score=_score_round(s,rules)
    assert score.card_game_points==[12,14]
    assert score.winner==Team.NORTH_SOUTH
    assert score.awarded_points==[22,14]


def test_complete_round_and_events(rules):
    e=BalootEngine(rules);s=e.create_round(seed=1);e.shuffle_and_initial_deal(s);e.bid_and_complete_deal(s,forced_contract=Contract.HOKUM,forced_buyer=Seat.SOUTH,forced_trump=Suit.HEARTS);e.prepare_projects(s);bot=HeuristicBot(e,BotProfile(s.seed))
    while s.current_player is not None: e.play_card(s,bot.choose(s))
    assert validate_completed_round(s)==[]
    assert len(s.completed_tricks)==8 and len(s.played_cards)==32


def test_many_complete_rounds(rules):
    e=BalootEngine(rules)
    for seed in range(20):
        s=e.create_round(seed=seed);e.shuffle_and_initial_deal(s);e.bid_and_complete_deal(s,forced_contract=Contract.SUN,forced_buyer=Seat.NORTH);e.prepare_projects(s);bot=HeuristicBot(e,BotProfile(s.seed))
        while s.current_player is not None: e.play_card(s,bot.choose(s))
        assert not validate_completed_round(s)
