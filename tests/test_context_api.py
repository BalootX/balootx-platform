import time
from fastapi.testclient import TestClient

from baloot_platform.core.bots import BotProfile, HeuristicBot
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, Seat, Suit
from baloot_platform.core.serialization import continue_trusted_context, replay_trusted_context
from baloot_platform.service.api import app


def test_trusted_context_replay_and_continue(rules):
    e=BalootEngine(rules);s=e.create_round(seed=12);e.shuffle_and_initial_deal(s);e.bid_and_complete_deal(s,forced_contract=Contract.SUN,forced_buyer=Seat.NORTH);e.prepare_projects(s);bot=HeuristicBot(e,BotProfile(s.seed))
    while len(s.played_cards)<13:e.play_card(s,bot.choose(s))
    data=s.to_dict(reveal_hands=True)
    rebuilt=replay_trusted_context(data,rules)
    assert rebuilt.current_player==s.current_player and rebuilt.hands==s.hands
    final=continue_trusted_context(data,rules)
    assert final.round_score is not None and len(final.completed_tricks)==8


def test_api_health_options_and_game():
    c=TestClient(app)
    assert c.get('/healthz').status_code==200
    opts=c.get('/api/v1/options').json();assert any(p['id']=='premium_video' for p in opts['plugins'])
    game=c.post('/api/v1/games',json={'seed':2,'dealer':'east'}).json();gid=game['gameId']
    for seat,action in [('north','pass'),('west','hokum'),('south','pass'),('east','pass'),('north','pass')]:
        response=c.post(f'/api/v1/games/{gid}/bids',json={'seat':seat,'action':action});assert response.status_code==200
    state=c.get(f'/api/v1/games/{gid}').json();assert state['contract']=='hokum' and state['buyer']=='west'


def test_trusted_hidden_information_is_denied_in_production_without_key(monkeypatch):
    monkeypatch.setenv('BALOOT_ENV','production')
    monkeypatch.delenv('BALOOT_ALLOW_TRUSTED_CONTEXT',raising=False)
    monkeypatch.setenv('BALOOT_TRUSTED_API_KEY','top-secret')
    c=TestClient(app)
    game=c.post('/api/v1/games',json={'seed':23,'dealer':'east'}).json()
    denied=c.get(f"/api/v1/games/{game['gameId']}?revealHands=true")
    assert denied.status_code==403
    allowed=c.get(
        f"/api/v1/games/{game['gameId']}?revealHands=true",
        headers={'x-baloot-trusted-key':'top-secret'},
    )
    assert allowed.status_code==200
    assert isinstance(allowed.json()['hands']['south'],list)
