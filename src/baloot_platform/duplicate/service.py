from __future__ import annotations

import hashlib
import json
import secrets
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from baloot_platform.config import platform_root
from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Seat
from baloot_platform.duplicate.models import BoardDefinition, DuplicateRoom, DuplicateRun, digest, now
from baloot_platform.duplicate.store import SQLiteDuplicateStore
from baloot_platform.service.game import GameManager, GameSession
from baloot_platform.storage.codec_v7 import decode, encode


class DuplicateBalootService:
    """Same-start competitive Baloot with immutable boards and comparable runs."""
    def __init__(self,root:str|Path|None,store:SQLiteDuplicateStore):
        self.root=platform_root(root); self.store=store

    @staticmethod
    def _legal_actions(session: GameSession) -> list[dict[str, Any]]:
        if session.bidding is not None and not session.bidding.finished:
            actions: list[dict[str, Any]] = []
            for action in session.bidding.legal_actions():
                if action == "hokum" and session.bidding.round_index == 2:
                    from baloot_platform.core.enums import Suit
                    actions.extend({"kind": "bid", "action": action, "trumpSuit": suit.name.lower()} for suit in Suit if suit != session.bidding.purchase_suit)
                else:
                    actions.append({"kind": "bid", "action": action})
            return actions
        if session.state.current_player is not None:
            return [{"kind": "play", "card": card.code} for card in session.engine.legal_moves(session.state)]
        return []

    @staticmethod
    def _state_hash(session:GameSession)->str:
        return hashlib.sha256(json.dumps(session.to_dict(reveal_hands=True),ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()

    def create_board(self,session:GameSession,*,observer_seat:Seat=Seat.SOUTH,label:str|None=None,bot_policy_hashes:Mapping[str,str]|None=None,scoring_profile:str='raw_round_margin_v1',time_control:Mapping[str,Any]|None=None,reveal_policy:str='after_completion')->BoardDefinition:
        payload=encode(session); payload_hash=hashlib.sha256(payload).hexdigest()
        public_prefix=hashlib.sha256(json.dumps([e.to_dict() for e in session.state.events],ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        identity={"schema":1,"payloadHash":payload_hash,"bundleHash":session.bundle_hash,"observer":observer_seat.name.lower(),"eventPrefix":public_prefix,"bots":dict(sorted((bot_policy_hashes or session.agent_profiles).items())),"score":list(session.state.match_score),"scoring":scoring_profile,"time":dict(time_control or {"decisionSeconds":30}),"reveal":reveal_policy}
        board_hash=digest(identity); board_id=f"BRD-{board_hash[:20].upper()}"; serial=f"DBL-{board_hash[:12].upper()}"
        board=BoardDefinition(board_id,1,board_hash,serial,session.bundle_hash,observer_seat,tuple(session.state.match_score),public_prefix,dict(bot_policy_hashes or session.agent_profiles),scoring_profile,dict(time_control or {"decisionSeconds":30}),reveal_policy,payload_hash,label=label)
        self.store.put_board(board,payload); return board

    def open_room(self,board_id:str,*,mode:str='friend',settings:Mapping[str,Any]|None=None)->DuplicateRoom:
        item=self.store.get_board(board_id)
        if item is None:raise KeyError(board_id)
        room=DuplicateRoom(f"ROOM-{uuid.uuid4().hex[:16].upper()}",item[0].board_id,secrets.token_hex(3).upper(),mode,'open',dict(settings or {}))
        self.store.put_room(room); return room

    def start_run(self,room_id:str,subject_id:str)->dict[str,Any]:
        room=self.store.get_room(room_id)
        if room is None:raise KeyError(room_id)
        item=self.store.get_board(room.board_id)
        if item is None:raise KeyError(room.board_id)
        board,payload=item; self.store.join(room.room_id,subject_id)
        session=decode(payload,root=self.root); session.id=f"DUP-{uuid.uuid4().hex[:16]}"; session.version=1
        manager=GameManager(self.root); manager.import_session(session); session=manager.advance_bots(session.id,stop_before=board.observer_seat)
        state_hash=self._state_hash(session); run=DuplicateRun(f"RUN-{uuid.uuid4().hex[:20].upper()}",room.room_id,board.board_id,subject_id,board.comparison_key,'active',session.version,state_hash)
        self.store.put_run(run,encode(session)); return {"board":board.to_dict(),"room":room.to_dict(),"run":run.to_dict(),"state":session.to_dict(reveal_hands=False),"legalActions":self._legal_actions(session)}

    def apply_action(self,run_id:str,action:Mapping[str,Any])->dict[str,Any]:
        loaded=self.store.get_run(run_id)
        if loaded is None:raise KeyError(run_id)
        run,payload=loaded
        if run.status!='active':raise ValueError('duplicate run is not active')
        board_item=self.store.get_board(run.board_id); assert board_item is not None
        board=board_item[0]; session=decode(payload,root=self.root)
        manager=GameManager(self.root); manager.import_session(session)
        observer=board.observer_seat; kind=str(action.get('kind','play'))
        if kind=='bid':
            from baloot_platform.core.enums import Suit
            trump=Suit.parse(action['trumpSuit']) if action.get('trumpSuit') else None
            session=manager.bid(session.id,seat=observer,action=str(action['action']),trump=trump,expected_version=session.version)
        elif kind=='play':
            session=manager.play(session.id,seat=observer,card=Card.parse(action['card']),ekka=bool(action.get('ekka',False)),expected_version=session.version)
        else:raise ValueError('action kind must be bid or play')
        session=manager.advance_bots(session.id,stop_before=observer)
        state_hash=self._state_hash(session); sequence=len(self.store.actions(run_id)); self.store.add_action(run_id,sequence,dict(action),state_hash)
        completed=session.state.current_player is None and (session.bidding is None or session.bidding.finished)
        result=None; status='active'; completed_at=None
        if completed and session.state.round_score is not None:
            team=observer.team; score=session.state.round_score
            result={"awardedPoints":score.awarded_points[int(team)],"opponentAwardedPoints":score.awarded_points[int(team.other)],"roundMargin":score.awarded_points[int(team)]-score.awarded_points[int(team.other)],"contractSucceeded":score.winner==session.state.buyer.team,"matchScore":list(session.state.match_score),"matchWinner":session.state.match_winner.name.lower() if session.state.match_winner else None,"decisionRegret":0.0}
            status='completed'; completed_at=now()
        updated=replace(run,status=status,version=session.version,state_hash=state_hash,result=result,completed_at=completed_at)
        self.store.put_run(updated,encode(session))
        return {"run":updated.to_dict(),"state":session.to_dict(reveal_hands=completed or board.reveal_policy=='immediate'),"legalActions":self._legal_actions(session),"actions":self.store.actions(run_id)}

    def reopen_run(self,run_id:str)->dict[str,Any]:
        loaded=self.store.get_run(run_id)
        if loaded is None:raise KeyError(run_id)
        run,payload=loaded; session=decode(payload,root=self.root)
        board=self.store.get_board(run.board_id)[0]
        reveal=run.status=='completed' or board.reveal_policy=='immediate'
        return {"board":board.to_dict(),"run":run.to_dict(),"state":session.to_dict(reveal_hands=reveal),"legalActions":self._legal_actions(session),"actions":self.store.actions(run_id),"leaderboard":self.store.leaderboard(board.board_id,board.comparison_key)}
