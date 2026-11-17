"""Deterministic replay and observer-projection adapter for Foundry traces."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

from baloot_platform.analysis.bidding import legal_bid_choices
from baloot_platform.analysis.models import InformationState
from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.cards import Card
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import GamePhase, Seat, Suit
from baloot_platform.core.models import RoundState
from baloot_platform.core.rules import RuleBook

from .evidence_v3 import DecisionLedgerRecordV1, canonical_sha256
from .simulation import (
    canonical_bid_complete_state_v1,
    observer_safe_bid_state_v1,
)
from .trace import CompleteGameTraceV1, TraceEventV1


@dataclass(frozen=True, slots=True)
class ReplayDecisionStateV1:
    decision: DecisionLedgerRecordV1
    state: RoundState
    bidding: BiddingMachine | None


def actor_observer_public_projection_v1(
    snapshot: ReplayDecisionStateV1,
) -> dict[str, Any]:
    """Reconstruct one decision's canonical actor-visible public state."""

    record = snapshot.decision
    if record.decision_kind == "bid":
        if snapshot.bidding is None:
            raise ValueError("bid actor-observer projection requires bidding state")
        return observer_safe_bid_state_v1(
            snapshot.state,
            snapshot.bidding,
            record.acting_seat,
        )
    if record.decision_kind == "play":
        if snapshot.bidding is not None:
            raise ValueError("play actor-observer projection cannot contain bidding state")
        return InformationState.from_complete(
            snapshot.state,
            record.acting_seat,
        ).to_dict()
    raise ValueError(
        f"unsupported actor-observer decision kind: {record.decision_kind}"
    )


def verify_actor_observer_public_prefix_v1(
    snapshot: ReplayDecisionStateV1,
) -> str:
    """Fail closed when a ledger prefix is not bound to replayed public state."""

    calculated = canonical_sha256(actor_observer_public_projection_v1(snapshot))
    if calculated != snapshot.decision.public_prefix_hash:
        raise ValueError(
            "trace "
            f"{snapshot.decision.decision_kind} actor-observer public-prefix hash "
            "mismatch"
        )
    return calculated


@dataclass(frozen=True, slots=True)
class VerifiedTraceReplayV1:
    trace_hash: str
    decisions: tuple[ReplayDecisionStateV1, ...]
    final_state_hash: str
    event_chain_hash: str


@dataclass(frozen=True, slots=True)
class ContextReplayProofV1:
    trace_hash: str
    starting_score: tuple[int, int]
    ending_score: tuple[int, int]
    awarded_points: tuple[int, int]
    dealer: Seat
    next_dealer: Seat
    round_number: int
    final_state_hash: str
    event_chain_hash: str
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "traceHash": self.trace_hash,
            "startingScore": list(self.starting_score),
            "endingScore": list(self.ending_score),
            "awardedPoints": list(self.awarded_points),
            "dealer": self.dealer.name.lower(),
            "nextDealer": self.next_dealer.name.lower(),
            "roundNumber": self.round_number,
            "finalStateHash": self.final_state_hash,
            "eventChainHash": self.event_chain_hash,
            "sha256": self.sha256,
        }


class CompleteTraceReplayerV1:
    """Replay a trusted trace only through authoritative legal transitions."""

    def __init__(self, rules: RuleBook):
        self.rules = rules

    @staticmethod
    def _mapping_key(value: Mapping[str, Any]) -> str:
        if "action" in value:
            trump = value.get("trumpSuit")
            return f"{value['action']}:{trump or ''}"
        return f"card:{Card.parse(dict(value)).id}"

    def replay(self, trace: CompleteGameTraceV1) -> VerifiedTraceReplayV1:
        if trace.rules_hash != self.rules.sha256:
            raise ValueError("trace rules hash does not match the replay engine")
        trusted = dict(trace.trusted_final_state)
        generation = dict(trusted.get("generation") or {})
        seed = int(generation.get("attemptSeed", trace.seed_provenance["seed"]))
        dealer = Seat.parse(
            str(generation.get("dealer", trace.seed_provenance["dealer"]))
        )
        round_number = int(generation.get("roundNumber", trusted.get("roundNumber", 1)))
        match_score_values = generation.get(
            "matchScore", trusted.get("matchScore", (0, 0))
        )
        match_score = (int(match_score_values[0]), int(match_score_values[1]))

        engine = BalootEngine(self.rules)
        state = engine.create_round(
            seed=seed,
            dealer=dealer,
            round_number=round_number,
            match_score=match_score,
        )
        deal_generator = dict(generation.get("dealGenerator") or {})
        if deal_generator:
            if not str(deal_generator.get("profileHash") or ""):
                raise ValueError(
                    "constructed trace lacks its generator profile identity"
                )
            trusted_shuffle = dict(trusted.get("shuffle") or {})
            if trusted_shuffle.get("algorithm") != "governed_preordered_v1":
                raise ValueError("constructed trace has the wrong shuffle algorithm")
            deck_order = tuple(
                int(item) for item in trusted_shuffle.get("deckOrder") or ()
            )
            if len(deck_order) != 32 or sorted(deck_order) != list(range(32)):
                raise ValueError(
                    "constructed trace lacks its complete trusted deck order"
                )
            expected_commitment = str(deal_generator.get("deckCommitment") or "")
            if expected_commitment and expected_commitment != str(
                trusted_shuffle.get("deckCommitment") or ""
            ):
                raise ValueError("constructed trace generator commitment mismatch")
            engine.initial_deal_from_preordered(state, deck_order)
            assert state.shuffle is not None
            if state.shuffle.to_dict(reveal_order=True) != trusted_shuffle:
                raise ValueError("constructed trace shuffle evidence mismatch")
        else:
            cut = trace.seed_provenance.get("cutPosition")
            engine.shuffle_and_initial_deal(
                state,
                cut_position=int(cut) if cut is not None else None,
            )
        assert state.purchase_card is not None
        state.phase = GamePhase.BIDDING
        machine = BiddingMachine(
            state.dealer,
            state.purchase_card.suit,
            AuctionDomainIRV1.from_rules(self.rules.data),
            state.purchase_card.rank,
        )
        snapshots: list[ReplayDecisionStateV1] = []
        records = iter(trace.decision_ledger.records)
        record = next(records, None)

        while record is not None and record.decision_kind == "bid":
            if machine.actor != record.acting_seat:
                raise ValueError(
                    "trace bidding actor does not match authoritative turn order"
                )
            legal = tuple(item.to_dict() for item in legal_bid_choices(machine))
            if {self._mapping_key(item) for item in legal} != {
                self._mapping_key(item) for item in record.legal_actions
            }:
                raise ValueError("trace bidding legal actions do not match the engine")
            complete_hash = canonical_sha256(
                canonical_bid_complete_state_v1(state, machine)
            )
            if complete_hash != record.complete_state_hash:
                raise ValueError("trace bidding complete-state hash mismatch")
            snapshot = ReplayDecisionStateV1(
                record,
                state.clone(),
                copy.deepcopy(machine),
            )
            verify_actor_observer_public_prefix_v1(snapshot)
            snapshots.append(snapshot)
            selected = dict(record.chosen_action)
            trump = selected.get("trumpSuit")
            machine.submit(
                str(selected["action"]),
                trump=Suit.parse(str(trump)) if trump is not None else None,
            )
            record = next(records, None)

        if machine.result is None:
            raise ValueError("trace auction ended before the bidding machine completed")
        purchased = engine.complete_deal_from_bid_result(state, machine.result)
        if purchased:
            for scripted in generation.get("scriptedEscalations") or ():
                action = dict(scripted)
                engine.apply_escalation(
                    state,
                    seat=Seat.parse(str(action["actor"])),
                    action=str(action["action"]),
                    mode=(
                        str(action["requestedMode"])
                        if action.get("requestedMode") is not None
                        else None
                    ),
                )
            engine.prepare_projects(state)
        elif record is not None:
            raise ValueError("an all-pass trace cannot contain play decisions")

        while record is not None:
            if record.decision_kind != "play":
                raise ValueError("unsupported decision kind in natural trace replay")
            if state.current_player != record.acting_seat:
                raise ValueError(
                    "trace play actor does not match authoritative turn order"
                )
            legal_cards = engine.legal_moves(state, record.acting_seat)
            if {item.id for item in legal_cards} != {
                Card.parse(dict(item)).id for item in record.legal_actions
            }:
                raise ValueError("trace play legal actions do not match the engine")
            if (
                canonical_sha256(state.to_dict(reveal_hands=True))
                != record.complete_state_hash
            ):
                raise ValueError("trace play complete-state hash mismatch")
            card = Card.parse(dict(record.chosen_action))
            if card not in legal_cards:
                raise ValueError("trace selected card is no longer legal")
            snapshot = ReplayDecisionStateV1(record, state.clone(), None)
            verify_actor_observer_public_prefix_v1(snapshot)
            snapshots.append(snapshot)
            engine.play_card(state, card)
            record = next(records, None)

        rebuilt_events = TraceEventV1.chain(item.to_dict() for item in state.events)
        if tuple(item.to_dict() for item in rebuilt_events) != tuple(
            item.to_dict() for item in trace.events
        ):
            raise ValueError("trace event chain does not reproduce exactly")
        known_event_hashes = {item.event_hash for item in rebuilt_events}
        if any(
            item.decision.resulting_event_hash not in known_event_hashes
            for item in snapshots
        ):
            raise ValueError("a replayed decision is not linked to its canonical event")
        replayed = state.to_dict(reveal_hands=True)
        for key, value in replayed.items():
            if trusted.get(key) != value:
                raise ValueError(f"trace final state does not reproduce field {key}")
        event_chain_hash = (
            rebuilt_events[-1].event_hash if rebuilt_events else canonical_sha256([])
        )
        return VerifiedTraceReplayV1(
            trace_hash=trace.sha256,
            decisions=tuple(snapshots),
            final_state_hash=canonical_sha256(replayed),
            event_chain_hash=event_chain_hash,
        )

    def replay_in_context(
        self,
        trace: CompleteGameTraceV1,
        *,
        match_score: tuple[int, int],
        round_number: int,
    ) -> ContextReplayProofV1:
        """Prove a normal scored action line under another pre-round score.

        Score-dependent escalation is intentionally excluded. The engine still
        revalidates every bid and card against the newly created state.
        """
        if trace.rules_hash != self.rules.sha256:
            raise ValueError(
                "trace rules hash does not match the context replay engine"
            )
        trusted = dict(trace.trusted_final_state)
        if str(trusted.get("phase")) != GamePhase.SCORED.value:
            raise ValueError("context reachability requires a complete scored trace")
        if str(trusted.get("escalation") or "normal") != "normal":
            raise ValueError("score-context transfer excludes escalated rounds")
        original_score = trusted.get("roundScore")
        if not isinstance(original_score, Mapping) or bool(original_score.get("gahwa")):
            raise ValueError("score-context transfer excludes missing or Gahwa scoring")
        start = tuple(int(item) for item in match_score)
        if len(start) != 2 or min(start) < 0:
            raise ValueError("context replay requires two nonnegative match scores")
        number = int(round_number)
        if number < 1:
            raise ValueError("context replay round number must be positive")
        generation = dict(trusted.get("generation") or {})
        dealer = Seat.parse(
            str(generation.get("dealer", trace.seed_provenance["dealer"]))
        )
        seed = int(generation.get("attemptSeed", trace.seed_provenance["seed"]))
        engine = BalootEngine(self.rules)
        state = engine.create_round(
            seed=seed,
            dealer=dealer,
            round_number=number,
            match_score=start,
        )
        cut = trace.seed_provenance.get("cutPosition")
        engine.shuffle_and_initial_deal(
            state,
            cut_position=int(cut) if cut is not None else None,
        )
        assert state.purchase_card is not None
        state.phase = GamePhase.BIDDING
        machine = BiddingMachine(
            dealer,
            state.purchase_card.suit,
            AuctionDomainIRV1.from_rules(self.rules.data),
            state.purchase_card.rank,
        )
        records = iter(trace.decision_ledger.records)
        record = next(records, None)
        while record is not None and record.decision_kind == "bid":
            if machine.actor != record.acting_seat:
                raise ValueError("context replay bidding actor mismatch")
            legal = tuple(item.to_dict() for item in legal_bid_choices(machine))
            if {self._mapping_key(item) for item in legal} != {
                self._mapping_key(item) for item in record.legal_actions
            }:
                raise ValueError("context replay bidding options changed")
            selected = dict(record.chosen_action)
            trump = selected.get("trumpSuit")
            machine.submit(
                str(selected["action"]),
                trump=Suit.parse(str(trump)) if trump is not None else None,
            )
            record = next(records, None)
        if machine.result is None or not engine.complete_deal_from_bid_result(
            state, machine.result
        ):
            raise ValueError(
                "context replay action line does not produce a purchased round"
            )
        engine.prepare_projects(state)
        while record is not None:
            if (
                record.decision_kind != "play"
                or state.current_player != record.acting_seat
            ):
                raise ValueError("context replay play sequence changed")
            legal_cards = engine.legal_moves(state, record.acting_seat)
            card = Card.parse(dict(record.chosen_action))
            if card not in legal_cards:
                raise ValueError("context replay selected card is not legal")
            engine.play_card(state, card)
            record = next(records, None)
        if state.phase != GamePhase.SCORED or state.round_score is None:
            raise ValueError("context replay did not reach authoritative scoring")
        awarded = tuple(int(item) for item in state.round_score.awarded_points)
        original_awarded = tuple(
            int(item) for item in original_score.get("awardedPoints") or ()
        )
        if awarded != original_awarded:
            raise ValueError(
                "normal round score changed under transferred match context"
            )
        ending = tuple(int(item) for item in state.match_score)
        events = TraceEventV1.chain(item.to_dict() for item in state.events)
        event_hash = events[-1].event_hash if events else canonical_sha256([])
        final_hash = canonical_sha256(state.to_dict(reveal_hands=True))
        core = {
            "schemaVersion": 1,
            "traceHash": trace.sha256,
            "startingScore": list(start),
            "endingScore": list(ending),
            "awardedPoints": list(awarded),
            "dealer": dealer.name.lower(),
            "nextDealer": engine.next_dealer(dealer).name.lower(),
            "roundNumber": number,
            "finalStateHash": final_hash,
            "eventChainHash": event_hash,
        }
        return ContextReplayProofV1(
            trace.sha256,
            start,
            ending,
            awarded,
            dealer,
            engine.next_dealer(dealer),
            number,
            final_hash,
            event_hash,
            canonical_sha256(core),
        )


class TraceDecisionProjectionStoreV1:
    """Read-only adapter exposing replayed trace decisions to perspective tools."""

    def __init__(self, trace: CompleteGameTraceV1, rules: RuleBook):
        self.trace = trace
        self.play_domain = rules.play_domain
        self.replay = CompleteTraceReplayerV1(rules).replay(trace)
        self._states = {
            item.decision.decision_id: item for item in self.replay.decisions
        }

    def get_decision(self, decision_id: str) -> dict[str, Any] | None:
        item = self._states.get(str(decision_id))
        if item is None:
            return None
        record = item.decision
        return {
            "decisionId": record.decision_id,
            "gameId": self.trace.game_id,
            "gameHash": self.trace.game_hash,
            "ordinal": record.ordinal,
            "decisionKind": record.decision_kind,
            "actor": record.acting_seat.name.lower(),
            "phase": record.phase,
            "legalActions": [dict(value) for value in record.legal_actions],
            "selectedAction": dict(record.chosen_action),
            "publicPrefixHash": record.public_prefix_hash,
            "completeStateHash": record.complete_state_hash,
            "resultingEventHash": record.resulting_event_hash,
        }

    def load_decision_payload(self, decision_id: str) -> dict[str, Any]:
        try:
            item = self._states[str(decision_id)]
        except KeyError as exc:
            raise KeyError(decision_id) from exc
        return {
            "state": item.state.clone(),
            "bidding": copy.deepcopy(item.bidding),
            "decisionKind": item.decision.decision_kind,
            "legalActions": tuple(dict(value) for value in item.decision.legal_actions),
            "selectedAction": dict(item.decision.chosen_action),
        }

    def list_decisions(
        self, identifier: str, *, limit: int = 1000
    ) -> list[dict[str, Any]]:
        if str(identifier) not in {
            self.trace.game_id,
            self.trace.game_hash,
            self.trace.sha256,
        }:
            return []
        return [
            item
            for decision_id in tuple(self._states)[: max(1, min(10000, int(limit)))]
            if (item := self.get_decision(decision_id)) is not None
        ]

    def decision_ledger(self, identifier: str):
        if str(identifier) not in {
            self.trace.game_id,
            self.trace.game_hash,
            self.trace.sha256,
        }:
            raise KeyError(identifier)
        return self.trace.decision_ledger
