import pytest
from baloot_platform.core.cards import Card,cards_to_mask
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract,GamePhase,Seat,Suit
from baloot_platform.core.models import PlayedCard,RoundState,Trick


def state(rules, hands, *, trick_cards=(), leader=Seat.NORTH, current=None, locked=False):
    s=RoundState(rules.sha256,1,1,Seat.EAST);s.contract=Contract.HOKUM;s.buyer=Seat.SOUTH;s.declarer=Seat.SOUTH;s.trump_suit=Suit.HEARTS;s.first_leader=leader;s.phase=GamePhase.PLAYING;s.locked=locked
    s.hands=[cards_to_mask(hands.get(seat,())) for seat in Seat];s.initial_hands=list(s.hands)
    t=Trick(0,leader)
    seat=leader
    for card,ekka in trick_cards:
        parsed=Card.parse(card);t.plays.append(PlayedCard(seat,parsed,0,len(t.plays),(),ekka));s.initial_hands[int(seat)]|=1<<parsed.id;seat=seat.next()
    s.current_trick=t;s.current_player=current if current is not None else seat
    return s


def codes(cards):return {c.code for c in cards}


def test_second_player_trump_lead_must_overtrump(rules):
    e=BalootEngine(rules);s=state(rules,{Seat.WEST:(Card.parse('H9'),Card.parse('H7'))},trick_cards=(('H8',False),))
    assert codes(e.legal_moves(s))=={'H9'}


def test_second_player_void_must_cut(rules):
    e=BalootEngine(rules);s=state(rules,{Seat.WEST:(Card.parse('H7'),Card.parse('C7'))},trick_cards=(('S7',False),))
    assert codes(e.legal_moves(s))=={'H7'}


def test_third_player_partner_winning_ace_may_discard(rules):
    e=BalootEngine(rules);s=state(rules,{Seat.SOUTH:(Card.parse('H7'),Card.parse('C7'))},trick_cards=(('SA',False),('S7',False)))
    assert codes(e.legal_moves(s))=={'H7','C7'}


def test_third_player_partner_non_ace_without_ekka_must_cut(rules):
    e=BalootEngine(rules);s=state(rules,{Seat.SOUTH:(Card.parse('H7'),Card.parse('C7'))},trick_cards=(('SK',False),('S7',False)))
    assert codes(e.legal_moves(s))=={'H7'}


def test_third_player_partner_ekka_may_discard(rules):
    e=BalootEngine(rules);s=state(rules,{Seat.SOUTH:(Card.parse('H7'),Card.parse('C7'))},trick_cards=(('SK',True),('S7',False)))
    assert codes(e.legal_moves(s))=={'H7','C7'}


def test_third_player_opponent_trump_winning_must_overtrump_or_discard(rules):
    e=BalootEngine(rules)
    s=state(rules,{Seat.SOUTH:(Card.parse('HJ'),Card.parse('C7'))},trick_cards=(('S7',False),('H9',False)))
    assert codes(e.legal_moves(s))=={'HJ'}
    s2=state(rules,{Seat.SOUTH:(Card.parse('H7'),Card.parse('C7'))},trick_cards=(('S7',False),('H9',False)))
    assert codes(e.legal_moves(s2))=={'H7','C7'}


def test_fourth_player_partner_second_winning_may_discard(rules):
    e=BalootEngine(rules);s=state(rules,{Seat.EAST:(Card.parse('H7'),Card.parse('C7'))},trick_cards=(('S7',False),('H9',False),('C8',False)))
    assert codes(e.legal_moves(s))=={'H7','C7'}


def test_fourth_player_opponent_winning_must_win_with_trump(rules):
    e=BalootEngine(rules);s=state(rules,{Seat.EAST:(Card.parse('HJ'),Card.parse('C7'))},trick_cards=(('S7',False),('H9',False),('C8',False)))
    # West (partner of East) is currently winning here, so discard is allowed.
    assert codes(e.legal_moves(s))=={'HJ','C7'}
    # Put opponent South in the winning seat with a trump.
    s2=state(rules,{Seat.EAST:(Card.parse('HJ'),Card.parse('C7'))},trick_cards=(('S7',False),('C8',False),('H9',False)))
    assert codes(e.legal_moves(s2))=={'HJ'}


def test_locked_hokum_all_trumps_may_lead_trump(rules):
    e=BalootEngine(rules);s=state(rules,{Seat.NORTH:(Card.parse('H7'),Card.parse('H8'))},locked=True,current=Seat.NORTH)
    assert codes(e.legal_moves(s))=={'H7','H8'}


def test_illegal_ekka_rejected(rules):
    e=BalootEngine(rules);s=state(rules,{Seat.NORTH:(Card.parse('SK'),),Seat.WEST:(Card.parse('SA'),),Seat.SOUTH:(),Seat.EAST:()},current=Seat.NORTH)
    with pytest.raises(ValueError):e.play_card(s,Card.parse('SK'),ekka=True)
