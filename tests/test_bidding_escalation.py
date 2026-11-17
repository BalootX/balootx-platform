from dataclasses import replace
import copy
import pytest

from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, GamePhase, Seat, Suit
from baloot_platform.core.rules import RuleBook


def test_published_first_round_hokum_sun_recall_priority(rules):
    # Dealer South: E,N,W pass; S Hokum. Recall proceeds W,N,E. W Sun,
    # then N Sun supersedes W; E passes, so N is buyer/declarer.
    m=BiddingMachine(Seat.SOUTH,Suit.HEARTS,AuctionDomainIRV1.from_rules(rules.data))
    assert m.order == [Seat.EAST,Seat.NORTH,Seat.WEST,Seat.SOUTH]
    for _ in range(3): assert m.submit('pass') is None
    assert m.submit('hokum') is None
    assert m.actor==Seat.WEST
    assert m.submit('sun') is None
    assert m.actor==Seat.NORTH
    assert m.submit('sun') is None
    result=m.submit('pass')
    assert result.contract==Contract.SUN and result.buyer==Seat.NORTH


def test_first_round_later_sun_recalled_by_earlier_player(rules):
    m=BiddingMachine(Seat.EAST,Suit.SPADES,AuctionDomainIRV1.from_rules(rules.data))
    assert m.submit('pass') is None       # North
    assert m.submit('pass') is None       # West
    assert m.submit('sun') is None        # South
    assert m.actor==Seat.WEST
    assert m.submit('pass') is None
    result=m.submit('sun')                 # North supersedes later South
    assert result.contract==Contract.SUN and result.buyer==Seat.NORTH


def test_second_round_passers_cannot_reenter(rules):
    m=BiddingMachine(Seat.EAST,Suit.SPADES,AuctionDomainIRV1.from_rules(rules.data))
    for _ in range(4): m.submit('pass')
    assert m.round_index==2 and m.actor==Seat.NORTH
    m.submit('pass')
    m.submit('hokum',trump=Suit.HEARTS)
    # only later players continue; North is never recalled
    assert m.actor==Seat.SOUTH
    result=m.submit('sun')
    assert result.contract==Contract.SUN and result.buyer==Seat.SOUTH


def test_all_pass_redeal(rules):
    m=BiddingMachine(Seat.EAST,Suit.SPADES,AuctionDomainIRV1.from_rules(rules.data))
    result=None
    for _ in range(8): result=m.submit('pass')
    assert result and result.all_passed and result.contract is None


def _dealt(rules,contract=Contract.HOKUM,score=(0,0)):
    e=BalootEngine(rules);s=e.create_round(seed=222,dealer=Seat.EAST,match_score=score)
    e.shuffle_and_initial_deal(s);e.bid_and_complete_deal(s,forced_contract=contract,forced_buyer=Seat.SOUTH,forced_trump=Suit.HEARTS if contract==Contract.HOKUM else None)
    return e,s


def test_hokum_escalation_chain_and_actor_rules(rules):
    e,s=_dealt(rules)
    e.apply_escalation(s,seat=Seat.WEST,action='double',mode='locked')
    assert s.escalation=='double' and s.locked
    with pytest.raises(ValueError):e.apply_escalation(s,seat=Seat.NORTH,action='triple')
    e.apply_escalation(s,seat=Seat.SOUTH,action='triple')
    assert not s.locked
    with pytest.raises(ValueError):e.apply_escalation(s,seat=Seat.EAST,action='four')
    e.apply_escalation(s,seat=Seat.WEST,action='four',mode='open')
    e.apply_escalation(s,seat=Seat.SOUTH,action='gahwa')
    assert s.escalation=='gahwa' and not s.locked


@pytest.mark.parametrize("seat", [Seat.WEST, Seat.EAST])
def test_either_opponent_seat_may_open_escalation(rules, seat):
    e, s = _dealt(rules)
    e.apply_escalation(s, seat=seat, action="double")
    assert s.double_seat == seat


@pytest.mark.parametrize("seat", [Seat.SOUTH, Seat.NORTH])
def test_declarer_team_may_not_open_escalation(rules, seat):
    e, s = _dealt(rules)
    with pytest.raises(ValueError, match="opponent"):
        e.apply_escalation(s, seat=seat, action="double")


def test_no_escalation_closes_by_advancing_to_projects_without_announcement(rules):
    e, s = _dealt(rules)
    e.prepare_projects(s)
    assert s.phase == GamePhase.PLAYING
    assert s.escalation == "normal"
    assert s.escalation_history == []


def test_sun_double_threshold_and_no_triple(rules):
    e,s=_dealt(rules,Contract.SUN,(101,50))
    e.apply_escalation(s,seat=Seat.WEST,action='double')
    assert s.escalation=='double'
    with pytest.raises(ValueError):e.apply_escalation(s,seat=Seat.SOUTH,action='triple')
    e2,s2=_dealt(rules,Contract.SUN,(80,60))
    with pytest.raises(ValueError):e2.apply_escalation(s2,seat=Seat.WEST,action='double')


def test_first_leader_policy_is_configurable(rules):
    data=copy.deepcopy(rules.data);data['play']['first_leader']='buyer';rb=RuleBook(data)
    e,s=_dealt(rb)
    assert s.first_leader==s.buyer==Seat.SOUTH


def test_annulling_variant_is_configurable(rules):
    data=copy.deepcopy(rules.data);data['dealing']['annulling']['enabled']=True;rb=RuleBook(data)
    e=BalootEngine(rb);s=e.create_round(seed=1);e.shuffle_and_initial_deal(s)
    # Directly construct eligible five-card hand; this tests rule behavior, not seed search.
    from baloot_platform.core.cards import Card,cards_to_mask
    eligible=(Card.parse('S7'),Card.parse('H7'),Card.parse('D8'),Card.parse('C8'),Card.parse('S9'))
    s.initial_five[int(Seat.NORTH)]=cards_to_mask(eligible);s.hands[int(Seat.NORTH)]=cards_to_mask(eligible)
    assert e.can_annul_initial_deal(s,Seat.NORTH)
    e.annul_initial_deal(s,Seat.NORTH)
    assert s.annulled_by==Seat.NORTH and s.phase==GamePhase.COMPLETE
