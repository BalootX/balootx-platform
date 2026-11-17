from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .bidding import run_bidding
from .cards import Card, cards_to_mask, iter_mask, mask_to_cards
from .dealing import DealLayoutV1
from .escalation_ir import EscalationDomainIRV1, EscalationStateV1
from .enums import Contract, GamePhase, Rank, Seat, Suit, Team
from .models import DealRecord, LegalPlayActionV1, PlayedCard, RoundState, Trick
from .play_ir import PlayDomainIRV1
from .projects import (
    baloot_project_state,
    announce_project_for_turn,
    score_project_team,
)
from .rules import RuleBook
from .project_visibility import project_rank_visibility_policy, redact_project_ranks
from .scoring import score_round
from .scoring_ir import ScoringDomainIRV1
from .shuffle import ShuffledDeck, governed_preordered_deck, shuffle_and_cut
from .structural_ir import StructuralDomainIRV1


@dataclass(frozen=True, slots=True)
class _PlayUndoSnapshot:
    hands: tuple[int, int, int, int]
    current_player: Seat | None
    current_trick: Trick | None
    completed_tricks_length: int
    projects: tuple
    project_winning_team: Team | None
    baloot_teams: frozenset[Team]
    trump_broken: bool
    round_score: Any
    match_score: tuple[int, int]
    match_winner: Team | None
    phase: GamePhase
    events_length: int


@dataclass(frozen=True, slots=True)
class EngineUndoToken:
    state_identity: int
    rules_hash: str
    play: _PlayUndoSnapshot
    before_sha256: str | None
    after_sha256: str | None
    action_kind: str


def _copy_trick(trick: Trick | None) -> Trick | None:
    if trick is None:
        return None
    return Trick(
        index=trick.index,
        leader=trick.leader,
        plays=list(trick.plays),
        winner=trick.winner,
        raw_points=trick.raw_points,
    )


class BalootEngine:
    """Authoritative deterministic Baloot round state machine.

    All product, API, probability, ML, and video layers consume this state
    machine. They are not allowed to reproduce legal-play or scoring logic.
    """

    def __init__(
        self,
        rules: RuleBook,
        *,
        structural_domain: StructuralDomainIRV1 | None = None,
        play_domain: PlayDomainIRV1 | None = None,
    ):
        self.rules = rules
        compiled_structural = rules.structural_domain.require_integrity()
        if structural_domain is not None:
            structural_domain.require_integrity()
        if (
            structural_domain is not None
            and structural_domain.sha256 != compiled_structural.sha256
        ):
            raise ValueError("structural domain does not match the active rules")
        self.structural_domain = structural_domain or compiled_structural
        compiled_play = rules.play_domain.require_integrity(
            self.structural_domain
        )
        if play_domain is not None:
            play_domain.require_integrity(self.structural_domain)
        if play_domain is not None and play_domain != compiled_play:
            raise ValueError("play domain does not match the active rules")
        self.play_domain = play_domain or compiled_play
        self.escalation_domain = EscalationDomainIRV1.from_rules(rules.data)
        self.project_domain = rules.project_domain
        self.scoring_domain = ScoringDomainIRV1.from_rules(rules.data)

    def legal_actions(self, state: RoundState) -> tuple[Card, ...]:
        """Return legacy card-only legal plays in stable card order."""
        return tuple(sorted(self.legal_moves(state), key=lambda card: card.id))

    def legal_play_actions(
        self,
        state: RoundState,
    ) -> tuple[LegalPlayActionV1, ...]:
        """Return every semantic play action in stable action-space order."""

        cards = self.legal_actions(state)
        actions = [LegalPlayActionV1(card) for card in cards]
        if (
            state.current_player is not None
            and state.current_trick is not None
            and not state.current_trick.plays
        ):
            actions.extend(
                LegalPlayActionV1(card, True)
                for card in cards
                if self._can_declare_ekka(state, state.current_player, card)
            )
        return tuple(sorted(actions, key=lambda action: action.action_id))

    def apply(
        self,
        state: RoundState,
        action: Card | int | str,
        *,
        ekka: bool = False,
        verify_identity: bool = True,
        emit_events: bool = True,
    ) -> EngineUndoToken:
        """Apply one legal play and return a bounded reversible delta."""
        if type(ekka) is not bool:
            raise TypeError("ekka must be a boolean")
        before_sha256 = self.state_hash(state) if verify_identity else None
        snapshot = _PlayUndoSnapshot(
            hands=tuple(state.hands),
            current_player=state.current_player,
            current_trick=_copy_trick(state.current_trick),
            completed_tricks_length=len(state.completed_tricks),
            projects=tuple(state.projects),
            project_winning_team=state.project_winning_team,
            baloot_teams=frozenset(state.baloot_teams),
            trump_broken=state.trump_broken,
            round_score=state.round_score,
            match_score=(int(state.match_score[0]), int(state.match_score[1])),
            match_winner=state.match_winner,
            phase=state.phase,
            events_length=len(state.events),
        )
        try:
            self.play_card(
                state,
                Card.parse(action),
                ekka=ekka,
                emit_events=emit_events,
            )
        except Exception:
            self._restore_play_snapshot(state, snapshot)
            raise
        return EngineUndoToken(
            state_identity=id(state),
            rules_hash=state.rules_hash,
            play=snapshot,
            before_sha256=before_sha256,
            after_sha256=(
                self.state_hash(state) if verify_identity else None
            ),
            action_kind="play",
        )

    def undo(self, state: RoundState, token: EngineUndoToken) -> None:
        """Restore a state in place and verify byte-identical canonical identity."""
        if token.action_kind != "play":
            raise ValueError(f"unsupported undo token kind: {token.action_kind}")
        if id(state) != token.state_identity:
            raise ValueError("undo token belongs to another state object")
        if state.rules_hash != token.rules_hash:
            raise ValueError("undo token rules hash mismatch")
        if (
            token.after_sha256 is not None
            and self.state_hash(state) != token.after_sha256
        ):
            raise ValueError("undo token does not match current state")
        snapshot = token.play
        self._restore_play_snapshot(state, snapshot)
        if (
            token.before_sha256 is not None
            and self.state_hash(state) != token.before_sha256
        ):
            raise AssertionError("undo failed to restore canonical state")

    @staticmethod
    def _restore_play_snapshot(
        state: RoundState,
        snapshot: _PlayUndoSnapshot,
    ) -> None:
        state.hands = list(snapshot.hands)
        state.current_player = snapshot.current_player
        state.current_trick = _copy_trick(snapshot.current_trick)
        del state.completed_tricks[snapshot.completed_tricks_length :]
        state.projects = list(snapshot.projects)
        state.project_winning_team = snapshot.project_winning_team
        state.baloot_teams = set(snapshot.baloot_teams)
        state.trump_broken = snapshot.trump_broken
        state.round_score = snapshot.round_score
        state.match_score = list(snapshot.match_score)
        state.match_winner = snapshot.match_winner
        state.phase = snapshot.phase
        del state.events[snapshot.events_length :]

    def terminal_result(self, state: RoundState) -> dict[str, Any] | None:
        if state.current_player is not None or state.round_score is None:
            return None
        return {
            "roundScore": state.round_score.to_dict(),
            "matchScore": list(state.match_score),
            "matchWinner": (
                state.match_winner.name.lower()
                if state.match_winner is not None
                else None
            ),
            "stateSha256": self.state_hash(state),
        }

    @staticmethod
    def observer_view(state: RoundState, seat: Seat) -> dict[str, Any]:
        """Return a public projection plus only the named observer's own hand."""
        view = state.to_dict(reveal_hands=False)
        view.pop("seed", None)
        view.pop("initialHands", None)
        if isinstance(view.get("shuffle"), dict):
            view["shuffle"].pop("seed", None)
        view["observer"] = seat.name.lower()
        view["hands"][seat.name.lower()] = [
            card.to_dict() for card in state.hand(seat)
        ]
        view["currentTrick"] = (
            state.current_trick.public_dict()
            if state.current_trick is not None
            else None
        )
        view["completedTricks"] = [
            trick.public_dict() for trick in state.completed_tricks
        ]
        view["projects"] = [
            project.to_dict()
            for project in state.projects
            if (
                project.owner == seat
                or project.declared
                or project.revealed
            )
        ]
        redact_project_ranks(view, seat.name.lower(), view["projects"])
        if not any(
            project.declared or project.revealed
            for project in state.projects
        ):
            view["projectWinningTeam"] = None
        events = []
        for event in state.events:
            if event.type == "project_holdings_detected":
                continue
            public = event.to_dict()
            if event.type in {"round_created", "deck_shuffled"}:
                public.pop("seed", None)
            if event.type == "card_played":
                public.pop("legalCards", None)
            events.append(public)
        view["events"] = events
        if state.project_rank_visibility is not None:
            from .exact_state_view import _public_events
            view["events"] = _public_events(
                view, view["projectWinningTeam"], view["projects"],
                validated_plays=[play.public_dict() for play in state.played_cards],
            )
        return view

    @staticmethod
    def canonical_state(state: RoundState) -> bytes:
        return json.dumps(
            state.to_dict(reveal_hands=True),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def state_hash(self, state: RoundState) -> str:
        return hashlib.sha256(self.canonical_state(state)).hexdigest()

    def dealer_right(self, dealer: Seat) -> Seat:
        # With N→W→S→E anticlockwise, next is the dealer's right-hand neighbour.
        return self.structural_domain.dealer_right(dealer)

    def dealer_left(self, dealer: Seat) -> Seat:
        return self.structural_domain.dealer_left(dealer)

    def next_dealer(self, dealer: Seat) -> Seat:
        return self.structural_domain.next_dealer(dealer)

    def partner(self, seat: Seat) -> Seat:
        return self.structural_domain.partner(seat)

    def next_round_dealer(self, state: RoundState) -> Seat:
        if state.annulled_by is not None:
            return self.structural_domain.next_dealer_after_annulment(state.dealer)
        return self.next_dealer(state.dealer)

    def resolve_first_leader(self, *, dealer: Seat, buyer: Seat) -> Seat:
        return self.play_domain.resolve_first_leader(
            dealer=dealer,
            buyer=buyer,
            structural_domain=self.structural_domain,
        )

    def create_round(
        self,
        *,
        seed: int,
        dealer: Seat = Seat.EAST,
        round_number: int = 1,
        match_score: tuple[int, int] = (0, 0),
    ) -> RoundState:
        state = RoundState(
            self.rules.sha256,
            int(seed),
            int(round_number),
            dealer,
            match_score=[int(match_score[0]), int(match_score[1])],
            project_rank_visibility=project_rank_visibility_policy(self.rules),
        )
        state.emit(
            "round_created",
            dealer=dealer.name.lower(),
            seed=int(seed),
            roundNumber=int(round_number),
            matchScore=list(state.match_score),
        )
        return state

    def shuffle_and_initial_deal(
        self,
        state: RoundState,
        *,
        cut_position: int | None = None,
        skip_cut: bool = False,
    ) -> None:
        if state.phase != GamePhase.CREATED:
            raise ValueError("round must be in created phase")
        if skip_cut and not self.structural_domain.allow_no_cut:
            raise ValueError("the selected rules profile requires a cut")
        shuffled = shuffle_and_cut(
            state.seed,
            cut_position,
            dealer=state.dealer,
            structural_domain=self.structural_domain,
            skip_cut=skip_cut,
        )
        self._initial_deal_from_shuffled(state, shuffled)

    def initial_deal_from_preordered(
        self,
        state: RoundState,
        deck_order: tuple[int, ...] | list[int],
    ) -> None:
        """Deal a validated constructed deck through the normal engine path."""
        if state.phase != GamePhase.CREATED:
            raise ValueError("round must be in created phase")
        if not self.structural_domain.allow_no_cut:
            raise ValueError("the selected rules profile does not allow a post-cut constructed deck")
        self._initial_deal_from_shuffled(
            state,
            governed_preordered_deck(
                deck_order,
                seed=state.seed,
                dealer=state.dealer,
                structural_domain=self.structural_domain,
            ),
        )

    def _initial_deal_from_shuffled(
        self,
        state: RoundState,
        shuffled: ShuffledDeck,
    ) -> None:
        """Single authoritative initial-deal implementation for every source."""
        if state.phase != GamePhase.CREATED:
            raise ValueError("round must be in created phase")
        deck_ids = self.structural_domain.validate_deck_order(
            shuffled.record.deck_order
        )
        if tuple(card.id for card in shuffled.cards) != deck_ids:
            raise ValueError("shuffled card payload does not match its deck order")
        if shuffled.record.algorithm not in {
            self.structural_domain.shuffle_algorithm,
            "governed_preordered_v1",
        }:
            raise ValueError("shuffle algorithm does not match the structural domain")
        layout = DealLayoutV1.build(self.structural_domain, dealer=state.dealer)
        state.shuffle = shuffled.record
        state.phase = GamePhase.SHUFFLED
        state.emit("deck_shuffled", **shuffled.record.to_dict(reveal_order=False))
        deck = list(shuffled.cards)
        hands: dict[Seat, list[Card]] = {seat: [] for seat in Seat}
        start = layout.initial_packets[0].seat
        for packet in layout.initial_packets:
            cards = tuple(deck[position] for position in packet.positions)
            hands[packet.seat].extend(cards)
            record = DealRecord(
                sequence=len(state.deal_trace),
                stage=packet.stage,
                packet=packet.packet,
                seat=packet.seat,
                cards=cards,
            )
            state.deal_trace.append(record)
            state.emit(
                "cards_dealt",
                stage=packet.stage,
                packet=packet.packet,
                seat=packet.seat.name.lower(),
                count=len(cards),
            )
        state.purchase_card = deck[layout.purchase_position]
        state.deal_trace.append(
            DealRecord(
                sequence=len(state.deal_trace),
                stage="purchase",
                packet=0,
                seat=None,
                cards=(state.purchase_card,),
                face_up=True,
            )
        )
        state.emit("purchase_card_exposed", card=state.purchase_card.to_dict())
        state.initial_five = [
            cards_to_mask(hands[Seat(i)])
            for i in range(self.structural_domain.players)
        ]
        state.hands = list(state.initial_five)
        state.phase = GamePhase.INITIAL_DEALT
        state.emit(
            "initial_deal_complete",
            firstBidder=start.name.lower(),
            cutter=(
                shuffled.record.cutter.name.lower()
                if shuffled.record.cutter is not None
                else None
            ),
        )
        state.undealt = tuple(deck[position] for position in layout.undealt_positions)

    def can_annul_initial_deal(self, state: RoundState, seat: Seat) -> bool:
        """Return whether *seat* may declare the configurable Kawesh/Saneen variant."""
        seat = Seat.parse(seat)
        if (
            not self.structural_domain.annulling_enabled
            or state.phase != GamePhase.INITIAL_DEALT
            or not self.structural_domain.can_declare_annulment(
                dealer=state.dealer,
                seat=seat,
            )
        ):
            return False
        cards = state.hand(seat)
        return self.structural_domain.can_annul_ranks(
            tuple(card.rank for card in cards)
        )

    def annul_initial_deal(self, state: RoundState, seat: Seat, *, reason: str = "kawesh") -> None:
        if not self.can_annul_initial_deal(state, seat):
            raise ValueError("seat is not eligible to annul this initial deal")
        if not self.structural_domain.accepts_annulment_reason(reason):
            raise ValueError(
                f"unsupported annulment reason: {reason!r}; allowed reasons are "
                + ", ".join(self.structural_domain.annulling_names)
            )
        state.annulled_by = seat
        state.annul_reason = str(reason)
        state.current_player = None
        state.phase = GamePhase.COMPLETE
        state.emit(
            "deal_annulled",
            seat=seat.name.lower(),
            reason=str(reason),
            dealerAdvances=self.structural_domain.annulling_dealer_advances,
            nextDealer=self.structural_domain.next_dealer_after_annulment(
                state.dealer
            ).name.lower(),
        )

    def bid_and_complete_deal(
        self,
        state: RoundState,
        *,
        forced_contract: Contract | None = None,
        forced_buyer: Seat | None = None,
        forced_trump: Suit | None = None,
    ) -> bool:
        if state.phase != GamePhase.INITIAL_DEALT or state.purchase_card is None:
            raise ValueError("initial deal required")
        state.phase = GamePhase.BIDDING
        five = {seat: mask_to_cards(state.initial_five[int(seat)]) for seat in Seat}
        result = run_bidding(
            five,
            state.purchase_card,
            state.dealer,
            self.rules,
            forced_contract=forced_contract,
            forced_buyer=forced_buyer,
            forced_trump=forced_trump,
        )
        return self.complete_deal_from_bid_result(state, result)

    def complete_deal_from_bid_result(self, state: RoundState, result) -> bool:
        if state.phase not in {GamePhase.INITIAL_DEALT, GamePhase.BIDDING} or state.purchase_card is None:
            raise ValueError("initial deal required")
        state.phase = GamePhase.BIDDING
        state.bid_history.extend(result.history)
        for bid in result.history:
            state.emit("bid_made", **bid.to_dict())
        if result.all_passed:
            if not self.structural_domain.all_pass_redeal:
                raise ValueError("the active structural domain rejects all-pass redeal")
            state.emit(
                "all_pass_redeal",
                nextDealer=self.next_dealer(state.dealer).name.lower(),
            )
            state.phase = GamePhase.COMPLETE
            return False
        assert (
            result.contract is not None
            and result.caller is not None
            and result.buyer is not None
            and result.declarer is not None
        )
        state.contract = result.contract
        state.buyer = result.buyer
        state.card_taker = (
            result.purchase_recipient
            if result.purchase_recipient is not None
            else result.buyer
        )
        state.declarer = result.declarer
        state.trump_suit = result.trump
        if state.winning_bidder != result.caller:
            raise AssertionError("auction caller differs from the canonical bid trace")
        state.emit(
            "contract_purchased",
            winningBidder=(
                state.winning_bidder.name.lower()
                if state.winning_bidder is not None
                else None
            ),
            buyer=state.buyer.name.lower(),
            contractOwner=state.contract_owner.name.lower(),
            purchaseRecipient=state.purchase_recipient.name.lower(),
            declarer=state.declarer.name.lower(),
            contract=state.contract.value,
            trumpSuit=(
                state.trump_suit.name.lower() if state.trump_suit is not None else None
            ),
            purchaseCard=state.purchase_card.to_dict(),
        )
        five = {seat: mask_to_cards(state.initial_five[int(seat)]) for seat in Seat}
        undealt = list(state.undealt)
        layout = DealLayoutV1.build(
            self.structural_domain,
            dealer=state.dealer,
            buyer=state.purchase_recipient,
        )
        undealt_by_position = dict(zip(layout.undealt_positions, undealt, strict=True))
        complete: dict[Seat, list[Card]] = {
            seat: list(five[seat]) for seat in Seat
        }
        complete[state.purchase_recipient].append(state.purchase_card)
        consumed = 0
        for packet in layout.final_packets:
            cards = tuple(undealt_by_position[position] for position in packet.positions)
            consumed += len(cards)
            complete[packet.seat].extend(cards)
            state.deal_trace.append(
                DealRecord(
                    sequence=len(state.deal_trace),
                    stage=packet.stage,
                    packet=packet.packet,
                    seat=packet.seat,
                    cards=cards,
                )
            )
            state.emit(
                "cards_dealt",
                stage=packet.stage,
                packet=packet.packet,
                seat=packet.seat.name.lower(),
                count=len(cards),
            )
        if consumed != len(undealt):
            raise AssertionError("final deal did not consume the remaining deck")
        if any(
            len(complete[seat]) != self.structural_domain.cards_per_player
            for seat in Seat
        ):
            raise AssertionError("every player must finish with eight cards")
        state.initial_hands = [
            cards_to_mask(complete[Seat(i)])
            for i in range(self.structural_domain.players)
        ]
        state.hands = list(state.initial_hands)
        state.phase = GamePhase.FINAL_DEALT
        state.first_leader = self.resolve_first_leader(
            dealer=state.dealer,
            buyer=state.buyer,
        )
        state.current_player = state.first_leader
        state.emit(
            "final_deal_complete",
            buyer=state.buyer.name.lower(),
            contractOwner=state.contract_owner.name.lower(),
            purchaseRecipient=state.purchase_recipient.name.lower(),
            firstLeader=state.first_leader.name.lower(),
        )
        state.validate_conservation()
        return True

    def apply_escalation(
        self,
        state: RoundState,
        *,
        seat: Seat,
        action: str,
        mode: str | None = None,
    ) -> None:
        if state.phase not in {GamePhase.FINAL_DEALT, GamePhase.PROJECTS}:
            raise ValueError("escalation must happen after the final deal and before play")
        if state.contract is None or state.declarer is None:
            raise ValueError("a contract is required")
        normalized_action = str(action).strip().lower()
        transition = self.escalation_domain.transition(
            EscalationStateV1(
                state.escalation,
                state.locked,
                state.double_seat,
            ),
            declarer=state.declarer,
            contract=state.contract,
            match_score=(int(state.match_score[0]), int(state.match_score[1])),
            actor=seat,
            action=normalized_action,
            mode=mode,
        )
        state.escalation = transition.state.level
        state.locked = transition.state.locked
        state.double_seat = transition.state.double_seat
        record = {
            "seat": seat.name.lower(),
            "team": seat.team.name.lower(),
            "action": normalized_action,
            "mode": transition.resolved_mode,
        }
        state.escalation_history.append(record)
        state.emit("escalation_announced", **record)

    def prepare_projects(self, state: RoundState) -> None:
        if state.phase != GamePhase.FINAL_DEALT or state.contract is None:
            raise ValueError("final deal required")
        if not self.escalation_domain.can_close(
            EscalationStateV1(
                state.escalation,
                state.locked,
                state.double_seat,
            )
        ):
            raise ValueError("escalation window cannot advance to projects")
        hands = {seat: state.hand(seat) for seat in Seat}
        state.projects = self.project_domain.detect(
            hands, state.contract, state.dealer
        )
        state._project_declaration_state = (
            self.project_domain.declaration_state(
                state.projects,
                dealer=state.dealer,
            )
            if self.project_domain.declaration_mode == "optional"
            else None
        )
        state.project_winning_team = None
        state.phase = GamePhase.PROJECTS
        state.emit("project_holdings_detected", count=len(state.projects))
        state.phase = GamePhase.PLAYING
        assert state.first_leader is not None
        state.current_trick = Trick(0, state.first_leader)
        state.current_player = state.first_leader
        state.emit("play_started", firstLeader=state.first_leader.name.lower())

    def apply_project_declaration_action(
        self,
        state: RoundState,
        action: Any,
        *,
        emit_events: bool = True,
    ) -> None:
        declaration = state._project_declaration_state
        if (
            state.phase != GamePhase.PLAYING
            or state.current_trick is None
            or state.current_trick.index != 0
            or declaration is None
            or declaration.complete
        ):
            raise ValueError("no active project declaration action")
        if declaration.current_actor != state.current_player:
            raise ValueError("project declaration actor must be the current player")
        updated, announced = self.project_domain.apply_declaration_action(
            declaration,
            action,
            trump_suit=state.trump_suit,
        )
        state._project_declaration_state = updated
        state.projects = list(updated.projects)
        applied = updated.actions[-1]
        if emit_events:
            state.emit("project_declaration_action", **applied.to_dict())
            if announced is not None:
                state.emit(
                    "project_declared",
                    owner=announced.owner.name.lower(),
                    kind=announced.kind,
                    value=announced.value,
                    highRank=announced.high_rank,
                    declarationOrder=announced.declaration_order,
                )

    def _announce_project_for_turn(
        self,
        state: RoundState,
        seat: Seat,
        *,
        emit_events: bool = True,
    ) -> None:
        """Announce one qualifying project when the player acts in trick one."""
        if (
            state.current_trick is None
            or not self.project_domain.should_announce(state.current_trick.index)
        ):
            return
        declaration = state._project_declaration_state
        if declaration is not None:
            if declaration.current_actor == seat:
                raise ValueError(
                    "project declaration action is required before playing"
                )
            return
        updated, announced = announce_project_for_turn(
            state.projects,
            seat,
            trump_suit=state.trump_suit,
            domain=self.project_domain,
        )
        state.projects = updated
        if announced is not None and emit_events:
            state.emit(
                "project_declared",
                owner=announced.owner.name.lower(),
                kind=announced.kind,
                value=announced.value,
                highRank=announced.high_rank,
                declarationOrder=announced.declaration_order,
            )

    def _reveal_projects_for_turn(
        self,
        state: RoundState,
        seat: Seat,
        *,
        emit_events: bool = True,
    ) -> None:
        """Reveal scoring projects at the owner's turn in trick two.

        The winning team's undeclared partner projects may also be shown, as long
        as they were selected as non-overlapping projects from the original hand.
        Revealing all projects before trick two would leak private cards to players
        who act earlier in that trick, so this transition is deliberately per-seat.
        """
        if (
            state.current_trick is None
            or not self.project_domain.should_reveal(state.current_trick.index)
        ):
            return
        state.projects, revealed = self.project_domain.reveal_for_turn(
            state.projects, seat, trump_suit=state.trump_suit,
            second_trick_leader=(state.completed_tricks[0].winner if state.completed_tricks else None),
        )
        if not revealed:
            return
        # A scoring Siri or Fifty that visibly contains both trump King and
        # Queen includes Baloot automatically. This is emitted at the owner's
        # reveal turn—not after trick one—so hidden project cards are not leaked
        # before they are legally shown.
        if (
            self.project_domain.baloot_enabled
            and state.contract == Contract.HOKUM
            and state.trump_suit is not None
        ):
            included, _blocked = baloot_project_state(
                state.projects,
                owner=seat,
                trump_suit=state.trump_suit,
                domain=self.project_domain,
            )
            if included:
                if seat.team not in state.baloot_teams:
                    state.baloot_teams.add(seat.team)
                    if emit_events:
                        state.emit(
                            "baloot_included_in_project",
                            owner=seat.name.lower(),
                            team=seat.team.name.lower(),
                        )
        if emit_events:
            state.emit(
                "projects_revealed",
                owner=seat.name.lower(),
                projects=[
                    project.to_dict() for project in revealed
                ],
            )

    def _trick_winner(self, trick: Trick, state: RoundState) -> Seat:
        if not trick.plays or state.contract is None:
            raise ValueError("cannot resolve empty trick")
        led = trick.plays[0].card.suit
        trump = state.trump_suit if state.contract == Contract.HOKUM else None
        eligible = trick.plays
        if trump is not None and any(p.card.suit == trump for p in eligible):
            eligible = [p for p in eligible if p.card.suit == trump]
        else:
            eligible = [p for p in eligible if p.card.suit == led]
        return max(
            eligible,
            key=lambda p: self.rules.strength(
                p.card, state.contract, state.trump_suit
            ),
        ).seat

    def _current_winner(self, trick: Trick, state: RoundState) -> Seat:
        return self._trick_winner(trick, state)

    def _can_declare_ekka(
        self, state: RoundState, seat: Seat, card: Card
    ) -> bool:
        if (
            not self.play_domain.ekka_enabled
            or
            state.contract != Contract.HOKUM
            or state.trump_suit is None
            or card.suit == state.trump_suit
        ):
            return False
        remaining = 0
        for mask in state.hands:
            remaining |= mask
        higher = [
            other
            for other in (Card.of(card.suit, rank) for rank in Rank)
            if self.rules.strength(other, Contract.SUN, None)
            > self.rules.strength(card, Contract.SUN, None)
        ]
        return not any(remaining & (1 << other.id) for other in higher)

    def legal_moves(
        self, state: RoundState, seat: Seat | None = None
    ) -> tuple[Card, ...]:
        if (
            state.phase != GamePhase.PLAYING
            or state.current_trick is None
            or state.contract is None
        ):
            return ()
        seat = state.current_player if seat is None else seat
        if seat is None or seat != state.current_player:
            return ()
        hand = tuple(iter_mask(state.hands[int(seat)]))
        trick = state.current_trick
        if not trick.plays:
            if (
                self.play_domain.locked_hokum_blocks_trump_lead
                and
                state.contract == Contract.HOKUM
                and state.locked
                and state.trump_suit is not None
            ):
                non_trump = tuple(c for c in hand if c.suit != state.trump_suit)
                if non_trump:
                    return non_trump
            return hand
        led = trick.led_suit
        assert led is not None
        followers = tuple(c for c in hand if c.suit == led)
        trump = state.trump_suit if state.contract == Contract.HOKUM else None
        if followers and self.play_domain.must_follow_suit:
            if self.play_domain.must_overtrump_when_possible and trump is not None and led == trump:
                current_winner = self._current_winner(trick, state)
                if current_winner.team != seat.team:
                    winning_card = next(
                        p.card for p in trick.plays if p.seat == current_winner
                    )
                    higher = tuple(
                        c
                        for c in followers
                        if self.rules.strength(c, state.contract, trump)
                        > self.rules.strength(
                            winning_card, state.contract, trump
                        )
                    )
                    if higher:
                        return higher
            return followers
        if state.contract in (Contract.SUN, Contract.ASHKAL) or trump is None:
            return hand
        trumps = tuple(c for c in hand if c.suit == trump)
        if not self.play_domain.must_trump_when_void or not trumps:
            return hand
        current_winner = self._current_winner(trick, state)
        winning_play = next(p for p in trick.plays if p.seat == current_winner)
        if (
            current_winner.team == seat.team
            and self.play_domain.partner_winning_exception
            == "position_aware_ace_or_ekka"
        ):
            # The only partner-winning case that still forces a cut is third hand
            # when partner led a non-Ace and did not announce Ekka.
            position = len(trick.plays)
            partner_led = trick.plays[0].seat == self.partner(seat)
            partner_ekka = trick.plays[0].ekka
            partner_ace = trick.plays[0].card.rank == Rank.ACE
            if position == 2 and partner_led and not (partner_ekka or partner_ace):
                return trumps
            return hand
        if (
            winning_play.card.suit == trump
            and self.play_domain.must_overtrump_when_possible
        ):
            higher = tuple(
                c
                for c in trumps
                if self.rules.strength(c, state.contract, trump)
                > self.rules.strength(winning_play.card, state.contract, trump)
            )
            if higher:
                return higher
            return (
                hand
                if self.play_domain.allow_discard_when_cannot_overtrump
                else trumps
            )
        return trumps

    def _determine_match_winner(self, state: RoundState) -> Team | None:
        round_winner = (
            state.round_score.winner
            if state.round_score is not None
            else Team.NORTH_SOUTH
        )
        _, winner = self.scoring_domain.apply_match_award(
            starting_score=state.match_score,
            awarded_points=(0, 0),
            round_winner=round_winner,
        )
        return winner

    def play_card(
        self,
        state: RoundState,
        card: Card | int,
        *,
        ekka: bool | None = None,
        emit_events: bool = True,
    ) -> PlayedCard:
        if ekka is not None and type(ekka) is not bool:
            raise TypeError("ekka must be a boolean or null")
        if state.current_player is None or state.current_trick is None:
            raise ValueError("no active player")
        card = Card.parse(card)
        seat = state.current_player
        legal = self.legal_moves(state, seat)
        if card not in legal:
            raise ValueError(
                f"illegal card {card.code}; legal: {[c.code for c in legal]}"
            )
        if not state.hands[int(seat)] & (1 << card.id):
            raise ValueError("card is not in acting player's hand")
        can_ekka = not state.current_trick.plays and self._can_declare_ekka(
            state, seat, card
        )
        if ekka is True and not can_ekka:
            raise ValueError(
                "Ekka may only be declared on the highest remaining "
                "non-trump lead"
            )
        # All invalid input is rejected before project/event/state mutation.
        self._announce_project_for_turn(
            state,
            seat,
            emit_events=emit_events,
        )
        self._reveal_projects_for_turn(
            state,
            seat,
            emit_events=emit_events,
        )
        state.hands[int(seat)] ^= 1 << card.id
        declare_ekka = bool(ekka) if ekka is not None else False
        baloot = False
        if (
            self.project_domain.baloot_enabled
            and state.contract == Contract.HOKUM
            and state.trump_suit is not None
        ):
            if card.suit == state.trump_suit and card.rank in {
                Rank.KING,
                Rank.QUEEN,
            }:
                other = Rank.QUEEN if card.rank == Rank.KING else Rank.KING
                other_card = Card.of(state.trump_suit, other)
                originally = bool(
                    state.initial_hands[int(seat)] & (1 << other_card.id)
                )
                already_played = any(
                    p.seat == seat and p.card == other_card
                    for t in state.completed_tricks
                    for p in t.plays
                )
                already_played = already_played or any(
                    p.seat == seat and p.card == other_card
                    for p in state.current_trick.plays
                )
                _included, blocked = baloot_project_state(
                    state.projects,
                    owner=seat,
                    trump_suit=state.trump_suit,
                    domain=self.project_domain,
                )
                baloot = originally and already_played and not blocked and seat.team not in state.baloot_teams
                if baloot:
                    state.baloot_teams.add(seat.team)
                    if emit_events:
                        state.emit(
                            "baloot_declared",
                            seat=seat.name.lower(),
                            team=seat.team.name.lower(),
                        )
        played = PlayedCard(
            seat,
            card,
            state.current_trick.index,
            len(state.current_trick.plays),
            legal,
            declare_ekka,
            baloot,
        )
        state.current_trick.plays.append(played)
        if emit_events:
            state.emit("card_played", **played.to_dict())
        if (
            state.trump_suit is not None
            and card.suit == state.trump_suit
            and state.current_trick.led_suit != state.trump_suit
        ):
            state.trump_broken = True
        if state.current_trick.complete:
            winner = self._trick_winner(state.current_trick, state)
            state.current_trick.winner = winner
            state.current_trick.raw_points = sum(
                self.rules.points(p.card, state.contract, state.trump_suit)
                for p in state.current_trick.plays
            )
            completed = state.current_trick
            state.completed_tricks.append(completed)
            if emit_events:
                state.emit(
                    "trick_completed",
                    trickIndex=completed.index,
                    winner=winner.name.lower(),
                    rawPoints=completed.raw_points,
                )
            if self.project_domain.should_compare(completed.index) and state.projects:
                state.projects, state.project_winning_team = score_project_team(
                    state.projects,
                    trump_suit=state.trump_suit,
                    domain=self.project_domain,
                    second_trick_leader=state.completed_tricks[0].winner,
                )
                if emit_events:
                    state.emit(
                        "projects_compared",
                        winningTeam=(
                            state.project_winning_team.name.lower()
                            if state.project_winning_team is not None
                            else None
                        ),
                        declarations=[
                            p.public_declaration()
                            for p in state.projects
                            if p.declared
                        ],
                    )
                # Cards are intentionally not revealed here. Each owner shows
                # scoring projects when their turn arrives in trick two.
            if completed.index == 1 and state.projects:
                if emit_events:
                    state.emit(
                        "projects_collected",
                        owners=sorted(
                            {
                                p.owner.name.lower()
                                for p in state.projects
                                if p.revealed
                            }
                        ),
                    )
            if len(state.completed_tricks) == 8:
                state.current_trick = None
                state.current_player = None
                state.round_score = score_round(
                    state, self.scoring_domain, self.project_domain
                )
                final_score, state.match_winner = self.scoring_domain.apply_match_award(
                    starting_score=state.match_score,
                    awarded_points=state.round_score.awarded_points,
                    round_winner=state.round_score.winner,
                    gahwa=state.round_score.gahwa,
                )
                state.match_score[:] = final_score
                state.round_score.match_winner = state.match_winner
                state.phase = GamePhase.SCORED
                if emit_events:
                    state.emit(
                        "round_scored",
                        score=state.round_score.to_dict(),
                        matchScore=list(state.match_score),
                        matchWinner=(
                            state.match_winner.name.lower()
                            if state.match_winner is not None
                            else None
                        ),
                    )
            else:
                state.current_trick = Trick(len(state.completed_tricks), winner)
                state.current_player = winner
        else:
            state.current_player = self.structural_domain.offset(seat, 1)
        state.validate_conservation()
        return played
