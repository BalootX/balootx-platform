"""Minimal observer-safe compatibility projection for the legacy renderer.

The scene reducer is the semantic authority. Until every lifecycle pixel
is read directly from scene documents, this immutable projection prevents the
legacy renderer from receiving a trusted ``EpisodeSpec`` or ``RoundState``.
It contains only public round facts, the selected observer's own cards, and
approved post-reveal analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping

from baloot_platform.analysis.models import AnalysisResult
from baloot_platform.core.cards import Card, cards_to_mask, mask_to_cards
from baloot_platform.core.enums import Contract, GamePhase, Seat, Suit, Team
from baloot_platform.core.models import (
    BidRecord,
    PlayedCard,
    Project,
    RoundScore,
    Trick,
)
from baloot_platform.core.scenario import EpisodeSpec

from .scene import SceneDocumentV1


@dataclass(frozen=True, slots=True)
class PublicShuffleProjectionV1:
    algorithm: str
    dealer: Seat
    cutter: Seat | None
    cut_position: int | None
    deck_commitment: str


@dataclass(frozen=True, slots=True)
class ObserverDealRecordV1:
    sequence: int
    stage: str
    packet: int
    seat: Seat | None
    cards: tuple[Card | None, ...]
    face_up: bool


@dataclass(frozen=True, slots=True)
class PauseStateProjectionV1:
    match_score: tuple[int, int]


@dataclass(frozen=True, slots=True)
class RenderStateProjectionV1:
    round_number: int
    dealer: Seat
    phase: GamePhase
    match_score: tuple[int, int]
    shuffle: PublicShuffleProjectionV1 | None
    deal_trace: tuple[ObserverDealRecordV1, ...]
    purchase_card: Card | None
    initial_five: tuple[int, int, int, int]
    initial_hands: tuple[int, int, int, int]
    bid_history: tuple[BidRecord, ...]
    contract: Contract | None
    buyer: Seat | None
    declarer: Seat | None
    trump_suit: Suit | None
    escalation: str
    escalation_history: tuple[Mapping[str, Any], ...]
    locked: bool
    first_leader: Seat | None
    completed_tricks: tuple[Trick, ...]
    projects: tuple[Project, ...]
    project_winning_team: Team | None
    baloot_teams: frozenset[Team]
    round_score: RoundScore | None
    match_winner: Team | None

    @property
    def played_cards(self) -> tuple[PlayedCard, ...]:
        return tuple(
            play
            for trick in self.completed_tricks
            for play in trick.plays
        )

    @property
    def winning_bid(self) -> BidRecord | None:
        return next(
            (
                bid
                for bid in reversed(self.bid_history)
                if bid.action != "pass"
            ),
            None,
        )

    @property
    def winning_bidder(self) -> Seat | None:
        bid = self.winning_bid
        if bid is None:
            return None
        return bid.bidder if bid.bidder is not None else bid.seat

    @property
    def contract_owner(self) -> Seat | None:
        return self.buyer

    @property
    def purchase_recipient(self) -> Seat | None:
        return self.buyer


@dataclass(frozen=True, slots=True)
class RenderEpisodeProjectionV1:
    schema_version: int
    episode_id: str
    bundle_hash: str
    viewer: Seat
    state_before_pause: PauseStateProjectionV1
    final_state: RenderStateProjectionV1
    pause_before_play_index: int
    analysis: AnalysisResult
    title_ar: str
    question_ar: str
    game_serial: str | None
    game_record_hash: str | None
    challenge_id: str | None
    analysis_id: str | None
    model_id: str | None
    audit_id: str | None
    decision_id: str | None
    decision_index: tuple[tuple[str, str], ...]
    audit_summary: Mapping[str, Any]
    playback_mode: str
    shared_board: bool


def _public_trick(trick: Trick) -> Trick:
    return Trick(
        index=trick.index,
        leader=trick.leader,
        plays=[
            replace(play, legal_cards=())
            for play in trick.plays
        ],
        winner=trick.winner,
        raw_points=trick.raw_points,
    )


def _observer_project(project: Project, observer: Seat) -> Project | None:
    if not (
        project.owner == observer
        or project.declared
        or project.revealed
    ):
        return None
    cards = (
        project.cards
        if project.owner == observer or project.revealed or project.scoring
        else ()
    )
    return replace(project, cards=cards)


def _safe_audit_summary(
    episode: EpisodeSpec,
    reveal_scene: SceneDocumentV1,
) -> Mapping[str, Any]:
    source = episode.audit_summary or {}
    allowed = {
        key: source.get(key)
        for key in (
            "actualAction",
            "bestAction",
            "expectedRegret",
            "winProbabilityDrop",
            "qualityLabel",
            "qualityScore",
            "auditConfidence",
        )
        if key in source
    }
    actual_intent = source.get("actualIntent")
    if isinstance(actual_intent, Mapping):
        allowed["actualIntent"] = {
            "status": actual_intent.get("status"),
        }
    analysis_layer = next(
        layer
        for layer in reveal_scene.value["layers"]
        if layer["id"] == "analysis"
    )
    shared = analysis_layer["content"].get("sharedComparison")
    if isinstance(shared, Mapping):
        allowed["sharedComparison"] = dict(shared)
    return MappingProxyType(allowed)


def build_observer_safe_render_episode(
    episode: EpisodeSpec,
    reveal_scene: SceneDocumentV1,
) -> RenderEpisodeProjectionV1:
    """Remove all trusted-only state before invoking legacy lifecycle pixels."""

    state = episode.final_state
    observer = episode.viewer
    viewer_index = int(observer)
    initial_five = [0, 0, 0, 0]
    initial_hands = [0, 0, 0, 0]
    initial_five[viewer_index] = state.initial_five[viewer_index]
    initial_hands[viewer_index] = state.initial_hands[viewer_index]

    deal_trace: list[ObserverDealRecordV1] = []
    for record in state.deal_trace:
        visible = record.seat == observer or record.face_up
        cards: tuple[Card | None, ...]
        if visible:
            cards = tuple(record.cards)
        else:
            cards = tuple(None for _ in record.cards)
        deal_trace.append(
            ObserverDealRecordV1(
                sequence=record.sequence,
                stage=record.stage,
                packet=record.packet,
                seat=record.seat,
                cards=cards,
                face_up=record.face_up,
            )
        )

    projects = tuple(
        safe
        for project in state.projects
        if (safe := _observer_project(project, observer)) is not None
    )
    shuffle = (
        PublicShuffleProjectionV1(
            algorithm=state.shuffle.algorithm,
            dealer=state.shuffle.dealer,
            cutter=state.shuffle.cutter,
            cut_position=state.shuffle.cut_position,
            deck_commitment=state.shuffle.deck_commitment,
        )
        if state.shuffle is not None
        else None
    )
    render_state = RenderStateProjectionV1(
        round_number=state.round_number,
        dealer=state.dealer,
        phase=state.phase,
        match_score=tuple(state.match_score),
        shuffle=shuffle,
        deal_trace=tuple(deal_trace),
        purchase_card=state.purchase_card,
        initial_five=tuple(initial_five),
        initial_hands=tuple(initial_hands),
        bid_history=tuple(state.bid_history),
        contract=state.contract,
        buyer=state.buyer,
        declarer=state.declarer,
        trump_suit=state.trump_suit,
        escalation=state.escalation,
        escalation_history=tuple(
            MappingProxyType(dict(item))
            for item in state.escalation_history
        ),
        locked=state.locked,
        first_leader=state.first_leader,
        completed_tricks=tuple(
            _public_trick(trick)
            for trick in state.completed_tricks
        ),
        projects=projects,
        project_winning_team=state.project_winning_team,
        baloot_teams=frozenset(state.baloot_teams),
        round_score=state.round_score,
        match_winner=state.match_winner,
    )
    shared_comparison = (episode.audit_summary or {}).get(
        "sharedComparison"
    )
    return RenderEpisodeProjectionV1(
        schema_version=21,
        episode_id=episode.episode_id,
        bundle_hash=episode.bundle_hash,
        viewer=observer,
        state_before_pause=PauseStateProjectionV1(
            tuple(episode.state_before_pause.match_score)
        ),
        final_state=render_state,
        pause_before_play_index=episode.pause_before_play_index,
        analysis=episode.analysis,
        title_ar=episode.title_ar,
        question_ar=episode.question_ar,
        game_serial=episode.game_serial,
        game_record_hash=episode.game_record_hash,
        challenge_id=episode.challenge_id,
        analysis_id=episode.analysis_id,
        model_id=episode.model_id,
        audit_id=episode.audit_id,
        decision_id=episode.decision_id,
        decision_index=tuple(episode.decision_index),
        audit_summary=_safe_audit_summary(episode, reveal_scene),
        playback_mode=episode.playback_mode,
        shared_board=bool(
            isinstance(shared_comparison, Mapping)
            and shared_comparison.get("sameStartVerified", False)
        ),
    )


def lifecycle_scene_facts(
    projection: RenderEpisodeProjectionV1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Serialize the renderer's safe lifecycle state into canonical scene facts."""

    state = projection.final_state
    public_deal = []
    observer_deal = []
    for record in state.deal_trace:
        public_deal.append(
            {
                "sequence": record.sequence,
                "stage": record.stage,
                "packet": record.packet,
                "seat": (
                    record.seat.name.lower()
                    if record.seat is not None
                    else None
                ),
                "count": len(record.cards),
                "faceUp": record.face_up,
                "cards": [
                    card.to_dict()
                    for card in record.cards
                    if card is not None
                ]
                if record.face_up
                else None,
            }
        )
        if record.seat == projection.viewer:
            observer_deal.append(
                {
                    "sequence": record.sequence,
                    "cards": [
                        card.to_dict()
                        for card in record.cards
                        if card is not None
                    ],
                }
            )
    public = {
        "roundNumber": state.round_number,
        "pauseBeforePlayIndex": projection.pause_before_play_index,
        "dealer": state.dealer.name.lower(),
        "phase": state.phase.value,
        "matchScore": list(state.match_score),
        "preDecisionMatchScore": list(
            projection.state_before_pause.match_score
        ),
        "shuffle": (
            {
                "algorithm": state.shuffle.algorithm,
                "dealer": state.shuffle.dealer.name.lower(),
                "cutter": (
                    state.shuffle.cutter.name.lower()
                    if state.shuffle.cutter is not None
                    else None
                ),
                "cutPosition": state.shuffle.cut_position,
                "deckCommitment": state.shuffle.deck_commitment,
            }
            if state.shuffle is not None
            else None
        ),
        "dealSequence": public_deal,
        "purchaseCard": (
            state.purchase_card.to_dict()
            if state.purchase_card is not None
            else None
        ),
        "bidHistory": [item.to_dict() for item in state.bid_history],
        "contract": state.contract.value if state.contract else None,
        "buyer": state.buyer.name.lower() if state.buyer is not None else None,
        "declarer": (
            state.declarer.name.lower()
            if state.declarer is not None
            else None
        ),
        "trumpSuit": (
            state.trump_suit.name.lower()
            if state.trump_suit is not None
            else None
        ),
        "escalation": state.escalation,
        "escalationHistory": [dict(item) for item in state.escalation_history],
        "locked": state.locked,
        "firstLeader": (
            state.first_leader.name.lower()
            if state.first_leader is not None
            else None
        ),
        "completedTricks": [
            item.public_dict() for item in state.completed_tricks
        ],
        "projects": [
            {
                **item.to_dict(),
                "declared": item.declared,
                "categoryPriority": item.category_priority,
            }
            for item in state.projects
            if item.declared or item.revealed
        ],
        "projectWinningTeam": (
            state.project_winning_team.name.lower()
            if state.project_winning_team is not None
            else None
        ),
        "balootTeams": sorted(
            item.name.lower() for item in state.baloot_teams
        ),
        "roundScore": (
            state.round_score.to_dict()
            if state.round_score is not None
            else None
        ),
        "matchWinner": (
            state.match_winner.name.lower()
            if state.match_winner is not None
            else None
        ),
    }
    observer = {
        "seat": projection.viewer.name.lower(),
        "initialFiveCards": [
            card.to_dict()
            for card in mask_to_cards(
                state.initial_five[int(projection.viewer)]
            )
        ],
        "initialHandCards": [
            card.to_dict()
            for card in mask_to_cards(
                state.initial_hands[int(projection.viewer)]
            )
        ],
        "dealCards": observer_deal,
        "privateProjects": [
            {
                **item.to_dict(),
                "cards": [card.to_dict() for card in item.cards],
                "declared": item.declared,
                "categoryPriority": item.category_priority,
            }
            for item in state.projects
            if item.owner == projection.viewer
        ],
    }
    return public, observer


def _layer_content(scene: SceneDocumentV1, layer_id: str) -> Mapping[str, Any]:
    layer = next(
        (
            item
            for item in scene.value["layers"]
            if item["id"] == layer_id
        ),
        None,
    )
    if layer is None or not isinstance(layer.get("content"), Mapping):
        raise ValueError(f"context scene is missing {layer_id!r}")
    return layer["content"]


def _optional_seat(value: Any) -> Seat | None:
    return None if value is None else Seat.parse(value)


def _optional_team(value: Any) -> Team | None:
    return None if value is None else Team[str(value).upper()]


def rebuild_render_episode_from_continuation_scene(
    envelope: RenderEpisodeProjectionV1,
    context: SceneDocumentV1,
) -> RenderEpisodeProjectionV1:
    """Rebuild completed replay state only from its post-reveal scene."""

    if context.value["stage"] != "continuation" or not context.verify():
        raise ValueError("renderer replay must be a valid continuation scene")
    public = _layer_content(context, "public-lifecycle")
    observer = _layer_content(context, "observer-lifecycle")
    viewer = Seat.parse(observer["seat"])
    if viewer != envelope.viewer:
        raise ValueError("context observer does not match render envelope")

    observer_deal = {
        int(item["sequence"]): tuple(
            Card.parse(card) for card in item["cards"]
        )
        for item in observer["dealCards"]
    }
    deal_trace = []
    for item in public["dealSequence"]:
        sequence = int(item["sequence"])
        public_cards = item.get("cards") or ()
        cards: tuple[Card | None, ...]
        if sequence in observer_deal:
            cards = observer_deal[sequence]
        elif public_cards:
            cards = tuple(Card.parse(card) for card in public_cards)
        else:
            cards = tuple(None for _ in range(int(item["count"])))
        deal_trace.append(
            ObserverDealRecordV1(
                sequence=sequence,
                stage=str(item["stage"]),
                packet=int(item["packet"]),
                seat=_optional_seat(item.get("seat")),
                cards=cards,
                face_up=bool(item["faceUp"]),
            )
        )

    bid_history = tuple(
        BidRecord(
            round_index=int(item["round"]),
            turn_index=int(item["turn"]),
            seat=Seat.parse(item["seat"]),
            action=str(item["action"]),
            contract=(
                Contract.parse(item["contract"])
                if item.get("contract") is not None
                else None
            ),
            trump=(
                Suit.parse(item["trumpSuit"])
                if item.get("trumpSuit") is not None
                else None
            ),
            bidder=_optional_seat(item.get("bidder")),
            buyer=_optional_seat(item.get("buyer")),
        )
        for item in public["bidHistory"]
    )
    completed_tricks = []
    for item in public["completedTricks"]:
        plays = [
            PlayedCard(
                seat=Seat.parse(play["seat"]),
                card=Card.parse(play["card"]),
                trick_index=int(play["trickIndex"]),
                position=int(play["position"]),
                ekka=bool(play.get("ekka", False)),
                baloot=bool(play.get("baloot", False)),
            )
            for play in item["plays"]
        ]
        completed_tricks.append(
            Trick(
                index=int(item["index"]),
                leader=Seat.parse(item["leader"]),
                plays=plays,
                winner=_optional_seat(item.get("winner")),
                raw_points=int(item["rawPoints"]),
            )
        )
    project_values_by_id = {
        (
            str(item["owner"]),
            str(item["kind"]),
            int(item["declarationOrder"]),
        ): item
        for item in public["projects"]
    }
    for item in observer.get("privateProjects") or ():
        project_values_by_id[
            (
                str(item["owner"]),
                str(item["kind"]),
                int(item["declarationOrder"]),
            )
        ] = item
    project_values = list(project_values_by_id.values())
    projects = tuple(
        Project(
            owner=Seat.parse(item["owner"]),
            kind=str(item["kind"]),
            cards=tuple(
                Card.parse(card) for card in (item.get("cards") or ())
            ),
            value=int(item["value"]),
            high_rank=int(item["highRank"]),
            category_priority=int(item["categoryPriority"]),
            declared=bool(item.get("declared", True)),
            declaration_order=int(item["declarationOrder"]),
            scoring=bool(item.get("scoring", False)),
            revealed=bool(item.get("revealed", False)),
        )
        for item in project_values
    )
    score_value = public.get("roundScore")
    score = (
        RoundScore(
            raw_card_points=list(score_value["rawCardPoints"]),
            card_game_points=list(score_value["cardGamePoints"]),
            project_points=list(score_value["projectPoints"]),
            baloot_points=list(score_value["balootPoints"]),
            total_before_award=list(score_value["totalBeforeAward"]),
            winner=Team[str(score_value["winner"]).upper()],
            awarded_points=list(score_value["awardedPoints"]),
            counting_team=Team[str(score_value["countingTeam"]).upper()],
            multiplier=int(score_value["multiplier"]),
            kabout=bool(score_value["kabout"]),
            gahwa=bool(score_value["gahwa"]),
            match_winner=_optional_team(score_value.get("matchWinner")),
            explanation=list(score_value.get("explanation") or ()),
        )
        if score_value is not None
        else None
    )
    shuffle_value = public.get("shuffle")
    shuffle = (
        PublicShuffleProjectionV1(
            algorithm=str(shuffle_value["algorithm"]),
            dealer=Seat.parse(shuffle_value["dealer"]),
            cutter=_optional_seat(shuffle_value.get("cutter")),
            cut_position=shuffle_value.get("cutPosition"),
            deck_commitment=str(shuffle_value["deckCommitment"]),
        )
        if shuffle_value is not None
        else None
    )
    initial_five = [0, 0, 0, 0]
    initial_hands = [0, 0, 0, 0]
    initial_five[int(viewer)] = cards_to_mask(
        Card.parse(card) for card in observer["initialFiveCards"]
    )
    initial_hands[int(viewer)] = cards_to_mask(
        Card.parse(card) for card in observer["initialHandCards"]
    )
    state = RenderStateProjectionV1(
        round_number=int(public["roundNumber"]),
        dealer=Seat.parse(public["dealer"]),
        phase=GamePhase(str(public["phase"])),
        match_score=tuple(public["matchScore"]),
        shuffle=shuffle,
        deal_trace=tuple(deal_trace),
        purchase_card=(
            Card.parse(public["purchaseCard"])
            if public.get("purchaseCard") is not None
            else None
        ),
        initial_five=tuple(initial_five),
        initial_hands=tuple(initial_hands),
        bid_history=bid_history,
        contract=(
            Contract.parse(public["contract"])
            if public.get("contract") is not None
            else None
        ),
        buyer=_optional_seat(public.get("buyer")),
        declarer=_optional_seat(public.get("declarer")),
        trump_suit=(
            Suit.parse(public["trumpSuit"])
            if public.get("trumpSuit") is not None
            else None
        ),
        escalation=str(public["escalation"]),
        escalation_history=tuple(
            MappingProxyType(dict(item))
            for item in public["escalationHistory"]
        ),
        locked=bool(public["locked"]),
        first_leader=_optional_seat(public.get("firstLeader")),
        completed_tricks=tuple(completed_tricks),
        projects=projects,
        project_winning_team=_optional_team(
            public.get("projectWinningTeam")
        ),
        baloot_teams=frozenset(
            Team[str(item).upper()] for item in public["balootTeams"]
        ),
        round_score=score,
        match_winner=_optional_team(public.get("matchWinner")),
    )
    return replace(
        envelope,
        state_before_pause=PauseStateProjectionV1(
            tuple(public["preDecisionMatchScore"])
        ),
        final_state=state,
    )
