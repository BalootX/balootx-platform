from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from baloot_platform.analysis.bidding import BidAnalysisResult, BidChoice, analyze_bid
from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.bidding import _choose_action
from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.bots import BotProfile, HeuristicBot
from baloot_platform.core.cards import mask_to_cards
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.episode_copy import governed_episode_text
from baloot_platform.core.enums import Seat
from baloot_platform.core.models import BidRecord, RoundState
from baloot_platform.core.rules import RuleBook
from baloot_platform.history.provenance_v4 import game_serial


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class BidEpisodeSpec:
    schema_version: int
    episode_id: str
    bundle_hash: str
    viewer: Seat
    state_at_bid: RoundState
    final_state: RoundState
    bidding_history_before: tuple[BidRecord, ...]
    bid_pause_index: int
    analysis: BidAnalysisResult
    title_ar: str
    question_ar: str
    game_serial: str | None = None
    game_record_hash: str | None = None
    challenge_id: str | None = None
    analysis_id: str | None = None
    model_id: str | None = None
    decision_id: str | None = None

    @property
    def state_before_pause(self) -> RoundState:
        return self.state_at_bid

    def to_dict(self) -> dict[str, Any]:
        final = self.final_state
        return {
            "schemaVersion": self.schema_version,
            "episodeId": self.episode_id,
            "decisionKind": "bid",
            "bundleHash": self.bundle_hash,
            "viewer": self.viewer.name.lower(),
            "titleAr": self.title_ar,
            "questionAr": self.question_ar,
            "provenance": {
                "gameSerial": self.game_serial,
                "gameRecordHash": self.game_record_hash,
                "challengeId": self.challenge_id,
                "analysisId": self.analysis_id,
                "modelId": self.model_id,
                "decisionId": self.decision_id,
            },
            "shuffle": final.shuffle.to_dict(reveal_order=True) if final.shuffle else None,
            "dealTrace": [record.to_dict(reveal_cards=True) for record in final.deal_trace],
            "roundNumber": final.round_number,
            "dealer": final.dealer.name.lower(),
            "purchaseCard": final.purchase_card.to_dict() if final.purchase_card else None,
            "initialFive": {
                seat.name.lower(): [card.to_dict() for card in mask_to_cards(final.initial_five[int(seat)])]
                for seat in Seat
            },
            "viewerInitialFive": [card.to_dict() for card in mask_to_cards(self.state_at_bid.initial_five[int(self.viewer)])],
            "biddingHistoryBefore": [item.to_dict() for item in self.bidding_history_before],
            "bidPauseIndex": self.bid_pause_index,
            "analysis": self.analysis.to_dict(),
            "biddingHistory": [item.to_dict() for item in final.bid_history],
            "buyer": final.buyer.name.lower() if final.buyer else None,
            "declarer": final.declarer.name.lower() if final.declarer else None,
            "contract": final.contract.value if final.contract else None,
            "trumpSuit": final.trump_suit.name.lower() if final.trump_suit else None,
            "firstLeader": final.first_leader.name.lower() if final.first_leader else None,
            "projects": [project.to_dict() for project in final.projects],
            "completedTricks": [trick.to_dict() for trick in final.completed_tricks],
            "roundScore": final.round_score.to_dict() if final.round_score else None,
            "events": [event.to_dict() for event in final.events],
        }


def _finish_from_choice(
    state: RoundState,
    machine: BiddingMachine,
    choice: BidChoice,
    engine: BalootEngine,
    *,
    seed: int,
) -> RoundState | None:
    final = state.clone()
    auction = copy.deepcopy(machine)
    result = auction.submit(choice.action, trump=choice.trump_suit)
    five = {seat: mask_to_cards(final.initial_five[int(seat)]) for seat in Seat}
    while result is None:
        action, trump = _choose_action(auction, five, final.purchase_card, engine.rules)
        result = auction.submit(action, trump=trump)
    if result.all_passed:
        return None
    engine.complete_deal_from_bid_result(final, result)
    engine.prepare_projects(final)
    bot = HeuristicBot(engine, BotProfile(seed + 313))
    while final.current_player is not None:
        engine.play_card(final, bot.choose(final))
    return final


def generate_bid_episode(
    rules: RuleBook,
    *,
    bundle_hash: str,
    copy: Mapping[str, Any] | None = None,
    seed: int = 1,
    viewer: Seat = Seat.SOUTH,
    dealer: Seat = Seat.EAST,
    match_score: tuple[int, int] = (0, 0),
    shallow_samples: int = 12,
    deep_samples: int = 64,
    maximum_seed_attempts: int = 200,
    minimum_margin_swing: float = 6.0,
    minimum_win_swing: float = 0.08,
) -> BidEpisodeSpec:
    engine = BalootEngine(rules)
    last_error: Exception | None = None
    for offset in range(maximum_seed_attempts):
        current_seed = seed + offset
        try:
            state = engine.create_round(seed=current_seed, dealer=dealer, match_score=match_score)
            engine.shuffle_and_initial_deal(state)
            machine = BiddingMachine(
                state.dealer,
                state.purchase_card.suit,
                AuctionDomainIRV1.from_rules(rules.data),
                state.purchase_card.rank,
            )
            best_snapshot: tuple[float, RoundState, BiddingMachine, BidAnalysisResult] | None = None
            while machine.result is None:
                actor = machine.actor
                if actor == viewer and len(machine.legal_actions()) >= 2:
                    shallow = analyze_bid(
                        state,
                        copy.deepcopy(machine),
                        viewer,
                        rules,
                        requested_samples=shallow_samples,
                        seed=current_seed * 1009 + len(machine.records),
                    )
                    ordered = list(shallow.choices)
                    margin_swing = ordered[0].expected_match_margin - ordered[-1].expected_match_margin
                    win_swing = ordered[0].team_win_probability - ordered[-1].team_win_probability
                    quality = margin_swing + 40.0 * win_swing + shallow.difficulty / 10.0
                    if best_snapshot is None or quality > best_snapshot[0]:
                        best_snapshot = (quality, state.clone(), copy.deepcopy(machine), shallow)
                five = {seat: mask_to_cards(state.initial_five[int(seat)]) for seat in Seat}
                action, trump = _choose_action(machine, five, state.purchase_card, rules)
                result = machine.submit(action, trump=trump)
                if result is not None:
                    break
            if best_snapshot is None:
                continue
            _quality, bid_state, bid_machine, shallow = best_snapshot
            deep = analyze_bid(
                bid_state,
                copy.deepcopy(bid_machine),
                viewer,
                rules,
                requested_samples=deep_samples,
                seed=current_seed * 65537 + len(bid_machine.records),
            )
            margin_swing = deep.choices[0].expected_match_margin - deep.choices[-1].expected_match_margin
            win_swing = deep.choices[0].team_win_probability - deep.choices[-1].team_win_probability
            if margin_swing < minimum_margin_swing and win_swing < minimum_win_swing:
                continue
            final = _finish_from_choice(
                bid_state,
                bid_machine,
                deep.recommended,
                engine,
                seed=current_seed,
            )
            if final is None or final.round_score is None:
                continue
            record_hash = hashlib.sha256(
                _canonical(
                    {
                        "bundleHash": bundle_hash,
                        "seed": current_seed,
                        "decisionKind": "bid",
                        "events": [event.to_dict() for event in final.events],
                        "score": final.round_score.to_dict(),
                    }
                ).encode("utf-8")
            ).hexdigest()
            episode_id = f"baloot-buy-{current_seed}-{len(bid_machine.records)}-{rules.sha256[:8]}"
            return BidEpisodeSpec(
                1,
                episode_id,
                bundle_hash,
                viewer,
                bid_state,
                final,
                tuple(copy.deepcopy(bid_machine.records)),
                len(bid_machine.records),
                deep,
                governed_episode_text(
                    "video.bid.challenge_title",
                    copy=copy,
                    number=str(current_seed),
                ),
                governed_episode_text("video.bid.challenge_question", copy=copy),
                game_serial(record_hash),
                record_hash,
            )
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"could not generate a publishable bid challenge: {last_error}")
