import copy
from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Contract,Rank,Seat,Suit,Team
from baloot_platform.core.models import Project
from baloot_platform.core.project_ir import ProjectDomainIRV1
from baloot_platform.core.projects import detect_projects,score_project_team
from baloot_platform.core.scoring import _hokum_game_points,_sun_game_points,score_round
from baloot_platform.core.scoring_ir import ScoringDomainIRV1
from test_core import _score_state


def _score_round(state, rules):
    return score_round(
        state,
        ScoringDomainIRV1.from_rules(rules.data),
        ProjectDomainIRV1.from_rules(rules.data),
    )


def test_conversion_boundaries():
    assert [_hokum_game_points(v) for v in (55,56,65,66,75,76)]==[5,6,6,7,7,8]
    assert [_sun_game_points(v) for v in (55,56,64,65,66,74,75)]==[11,12,12,13,14,14,15]


def test_all_project_values(rules):
    hands={seat:() for seat in Seat}
    hands[Seat.NORTH]=(Card.parse('SA'),Card.parse('SK'),Card.parse('SQ'),Card.parse('SJ'),Card.parse('S10'),Card.parse('H7'),Card.parse('D7'),Card.parse('C7'))
    sun=detect_projects(hands,Contract.SUN,rules,Seat.EAST)
    assert any(p.kind=='hundred_sequence' and p.value==20 for p in sun)
    hokum=detect_projects(hands,Contract.HOKUM,rules,Seat.EAST)
    assert any(p.kind=='hundred_sequence' and p.value==10 for p in hokum)


def test_four_kind_eligibility_and_values(rules):
    for rank,code,value_sun,value_hokum in [(Rank.ACE,'A',40,10),(Rank.KING,'K',20,10),(Rank.QUEEN,'Q',20,10),(Rank.JACK,'J',20,10),(Rank.TEN,'10',20,10)]:
        cards=tuple(Card.of(suit,rank) for suit in Suit)+(Card.parse('S7'),Card.parse('H8'),Card.parse('D9'),Card.parse('C7'))
        hands={seat:() for seat in Seat};hands[Seat.NORTH]=cards[:8]
        assert any(p.value==value_sun for p in detect_projects(hands,Contract.SUN,rules,Seat.EAST))
        assert any(p.value==value_hokum for p in detect_projects(hands,Contract.HOKUM,rules,Seat.EAST))


def test_project_tie_earlier_declaration_wins(rules):
    a=Project(Seat.NORTH,'siri',(),2,6,0,True,1)
    b=Project(Seat.WEST,'siri',(),2,6,0,True,2)
    _,winner=score_project_team(
        [a,b], domain=ProjectDomainIRV1.from_rules(rules.data)
    )
    assert winner==Team.NORTH_SOUTH


def test_four_kind_beats_equal_value_five_sequence(rules):
    sequence=Project(Seat.NORTH,'hundred_sequence',(),20,7,1,True,1)
    group=Project(Seat.WEST,'hundred_four_kind',(),20,3,2,True,2)
    _,winner=score_project_team(
        [sequence,group], domain=ProjectDomainIRV1.from_rules(rules.data)
    )
    assert winner==Team.WEST_EAST


def test_optional_trump_sequence_priority(rules):
    plain=Project(Seat.NORTH,'siri',(Card.parse('CK'),Card.parse('CQ'),Card.parse('CJ')),2,6,0,True,1)
    trump=Project(Seat.WEST,'siri',(Card.parse('HK'),Card.parse('HQ'),Card.parse('HJ')),2,6,0,True,2)
    domain=ProjectDomainIRV1.from_rules(rules.data)
    variant_data=copy.deepcopy(rules.data)
    variant_data['disputed_options']['equal_hokum_sequence_trump_priority']={
        'selected': True,
        'alternatives': [False],
    }
    variant_domain=ProjectDomainIRV1.from_rules(variant_data)
    _,normal=score_project_team(
        [plain,trump], trump_suit=Suit.HEARTS, domain=domain
    )
    _,variant=score_project_team(
        [plain,trump], trump_suit=Suit.HEARTS, domain=variant_domain
    )
    assert normal==Team.NORTH_SOUTH and variant==Team.WEST_EAST


def test_pagat_hokum_baloot_tie_and_loss(rules):
    s=_score_state(rules,Contract.HOKUM,Seat.NORTH,91,Team.WEST_EAST,baloot=(Team.NORTH_SOUTH,))
    score=_score_round(s,rules)
    assert score.card_game_points==[7,9] and score.total_before_award==[9,9]
    assert score.winner==Team.NORTH_SOUTH and score.awarded_points==[9,9]
    s2=_score_state(rules,Contract.HOKUM,Seat.NORTH,92,Team.WEST_EAST,baloot=(Team.NORTH_SOUTH,))
    score2=_score_round(s2,rules)
    assert score2.winner==Team.WEST_EAST and score2.awarded_points==[0,18]


def test_doubled_tripled_and_quadrupled_awards(rules):
    for level,multiplier in [('double',2),('triple',3),('four',4)]:
        s=_score_state(rules,Contract.HOKUM,Seat.NORTH,90,Team.WEST_EAST,project_points=(10,0),baloot=(Team.NORTH_SOUTH,))
        s.escalation=level
        # Make counting team match escalation alternation; result winner may vary,
        # but total winner-take-all formula is deterministic.
        s.escalation_history=[{'team':'west_east'}] if level in {'double','four'} else [{'team':'north_south'}]
        score=_score_round(s,rules)
        assert sum(score.awarded_points)==16*multiplier+20+2


def test_kabout_uses_only_winning_team_projects(rules):
    s=_score_state(rules,Contract.HOKUM,Seat.NORTH,162,Team.NORTH_SOUTH,project_points=(10,10),baloot=(Team.NORTH_SOUTH,Team.WEST_EAST))
    # Reassign all tricks to N-S to force Kabout.
    for trick in s.completed_tricks: trick.winner=Seat.NORTH
    score=_score_round(s,rules)
    assert score.kabout and score.awarded_points==[37,0]  # 25 + 10 + 2


def test_gahwa_sets_match_winner(rules):
    s=_score_state(rules,Contract.HOKUM,Seat.NORTH,100,Team.WEST_EAST);s.escalation='gahwa';s.escalation_history=[{'team':'north_south'}]
    score=_score_round(s,rules)
    assert score.gahwa and score.match_winner==score.winner
