from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from baloot_platform.core.cards import Card, cards_to_mask
from baloot_platform.core.enums import Contract, Seat, Suit, Team
from baloot_platform.core.models import PlayedCard, RoundState
from baloot_platform.core.project_visibility import redact_project_ranks


@dataclass(frozen=True, slots=True)
class BidChoice:
    action: str
    trump_suit: Suit | None = None

    @property
    def id(self) -> str:
        return self.action if self.trump_suit is None else f"{self.action}:{self.trump_suit.name.lower()}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "trumpSuit": self.trump_suit.name.lower() if self.trump_suit is not None else None,
        }


@dataclass(frozen=True, slots=True)
class BidChoiceEvaluation:
    choice: BidChoice
    expected_match_margin: float
    team_win_probability: float
    contract_success_probability: float
    downside_cvar10: float
    volatility: float
    best_share: float
    samples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.choice.to_dict(),
            "expectedRoundMargin": self.expected_match_margin,
            "roundLeadProbability": self.team_win_probability,
            "contractSuccessProbability": self.contract_success_probability,
            "downsideCvar10": self.downside_cvar10,
            "volatility": self.volatility,
            "bestShare": self.best_share,
            "samples": self.samples,
            "deprecatedAliases": {
                "expectedMatchMargin": self.expected_match_margin,
                "teamWinProbability": self.team_win_probability,
            },
        }


@dataclass(frozen=True, slots=True)
class PublicProject:
    owner: Seat
    kind: str
    value: int
    high_rank: int | None
    declared: bool
    scoring: bool
    revealed: bool
    cards: tuple[Card, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "owner": self.owner.name.lower(),
            "kind": self.kind,
            "value": self.value,
            "highRank": self.high_rank,
            "declared": self.declared,
            "scoring": self.scoring,
            "revealed": self.revealed,
            "cards": [card.to_dict() for card in self.cards] if self.revealed else None,
        }


def public_projects_from_complete(state: RoundState, observer: Seat) -> tuple[PublicProject, ...]:
    """Shared rank projection without building the rest of an information state."""
    rows = [project.to_dict() for project in state.projects]
    redact_project_ranks(
        {"projects": rows, "projectRankVisibility": state.project_rank_visibility,
         "completedTricks": state.completed_tricks}, observer.name.lower(), rows,
    )
    return tuple(
        PublicProject(project.owner, project.kind, project.value, public["highRank"],
                      project.declared, project.scoring, project.revealed,
                      project.cards if project.revealed else ())
        for project, public in zip(state.projects, rows, strict=True)
        if project.declared or project.revealed
    )


@dataclass(frozen=True, slots=True)
class InformationState:
    """Observer-safe public state used by probability and ML subsystems.

    `known_unplayed` contains only cards that the observer is legitimately
    entitled to locate: their own current hand, the exposed purchase card while
    it remains with the buyer, and cards revealed by scoring projects. Opponent
    legal alternatives are stripped from public plays.
    """

    rules_hash: str
    observer: Seat
    observer_hand: int
    known_unplayed: tuple[int, int, int, int]
    hand_sizes: tuple[int, int, int, int]
    dealer: Seat
    buyer: Seat
    declarer: Seat
    purchase_card: Card | None
    contract: Contract
    trump_suit: Suit | None
    first_leader: Seat
    current_player: Seat
    locked: bool
    escalation: str
    escalation_history: tuple[dict[str, Any], ...]
    match_score: tuple[int, int]
    plays: tuple[PlayedCard, ...]
    projects: tuple[PublicProject, ...]
    baloot_teams: tuple[Team, ...]
    trump_broken: bool
    bid_history: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_complete(cls, state: RoundState, observer: Seat) -> "InformationState":
        if state.contract is None or state.buyer is None or state.declarer is None or state.first_leader is None or state.current_player is None:
            raise ValueError("an active, contracted play state is required")

        # Never expose an opponent's actual legal alternatives. The observer's
        # own previous alternatives are information they had at the time and may
        # be retained for personal forcedness analysis.
        public_plays = tuple(
            replace(play, legal_cards=play.legal_cards if play.seat == observer else ())
            for play in state.played_cards
        )

        known = [0, 0, 0, 0]
        known[int(observer)] = state.hands[int(observer)]
        played_mask = cards_to_mask(play.card for play in public_plays)
        if state.purchase_card is not None and not (played_mask & (1 << state.purchase_card.id)):
            if state.hands[int(state.buyer)] & (1 << state.purchase_card.id):
                known[int(state.buyer)] |= 1 << state.purchase_card.id

        public_projects = public_projects_from_complete(state, observer)
        for project in public_projects:
            if project.revealed:
                for card in project.cards:
                    if state.hands[int(project.owner)] & (1 << card.id):
                        known[int(project.owner)] |= 1 << card.id

        # Known unplayed cards must be disjoint across seats and fit the public
        # hand sizes. This catches accidental hidden-state leakage immediately.
        union = 0
        for seat in Seat:
            mask = known[int(seat)]
            if union & mask:
                raise AssertionError("known public cards assigned to multiple seats")
            if mask.bit_count() > state.hands[int(seat)].bit_count():
                raise AssertionError("known public cards exceed a player's hand size")
            union |= mask

        return cls(
            state.rules_hash,
            observer,
            state.hands[int(observer)],
            tuple(known),
            tuple(mask.bit_count() for mask in state.hands),
            state.dealer,
            state.buyer,
            state.declarer,
            state.purchase_card,
            state.contract,
            state.trump_suit,
            state.first_leader,
            state.current_player,
            state.locked,
            state.escalation,
            tuple(dict(item) for item in state.escalation_history),
            tuple(state.match_score),
            public_plays,
            public_projects,
            tuple(sorted(state.baloot_teams, key=int)),
            state.trump_broken,
            tuple(item.to_dict() for item in state.bid_history),
        )

    @property
    def known_cards_mask(self) -> int:
        result = cards_to_mask(play.card for play in self.plays)
        for mask in self.known_unplayed:
            result |= mask
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "rulesHash": self.rules_hash,
            "observer": self.observer.name.lower(),
            "observerHand": [c.to_dict() for c in CardMask(self.observer_hand)],
            "knownUnplayed": {
                seat.name.lower(): [card.to_dict() for card in CardMask(self.known_unplayed[int(seat)])]
                for seat in Seat
            },
            "handSizes": list(self.hand_sizes),
            "dealer": self.dealer.name.lower(),
            "buyer": self.buyer.name.lower(),
            "declarer": self.declarer.name.lower(),
            "purchaseCard": self.purchase_card.to_dict() if self.purchase_card else None,
            "contract": self.contract.value,
            "trumpSuit": self.trump_suit.name.lower() if self.trump_suit is not None else None,
            "firstLeader": self.first_leader.name.lower(),
            "currentPlayer": self.current_player.name.lower(),
            "locked": self.locked,
            "escalation": self.escalation,
            "escalationHistory": [dict(item) for item in self.escalation_history],
            "matchScore": list(self.match_score),
            "plays": [p.to_dict() for p in self.plays],
            "projects": [project.to_dict() for project in self.projects],
            "balootTeams": [t.name.lower() for t in self.baloot_teams],
            "trumpBroken": self.trump_broken,
            "biddingHistory": [dict(item) for item in self.bid_history],
        }


class CardMask:
    def __init__(self, mask: int):
        self.mask = mask
    def __iter__(self):
        from baloot_platform.core.cards import iter_mask
        return iter_mask(self.mask)


@dataclass(frozen=True, slots=True)
class HardDeduction:
    kind: str
    seat: Seat
    suit: Suit | None
    certainty: float
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "seat": self.seat.name.lower(), "suit": self.suit.name.lower() if self.suit is not None else None, "certainty": self.certainty, "source": self.source}


@dataclass(frozen=True, slots=True)
class SignalObservation:
    kind: str
    seat: Seat
    relationship: str
    cards: tuple[Card, ...]
    confidence: float
    forcedness: str
    interpretations: tuple[str, ...]
    caveat: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "seat": self.seat.name.lower(),
            "relationship": self.relationship,
            "cards": [c.to_dict() for c in self.cards],
            "confidence": self.confidence,
            "forcedness": self.forcedness,
            "interpretations": list(self.interpretations),
            "caveat": self.caveat,
        }


SEARCH_COMPLETION_STATUSES = frozenset({
    "exact",
    "heuristic_complete",
    "budget_exhausted",
    "cancelled",
    "failed",
})


@dataclass(frozen=True, slots=True)
class ActionEvaluation:
    card: Card
    expected_round_margin: float
    expected_team_awarded_points: float
    expected_opponent_awarded_points: float
    round_lead_probability: float
    draw_probability: float
    confidence_low: float
    confidence_high: float
    best_share: float
    samples: int
    effective_samples: float
    exact_share: float
    contract_success_probability: float = 0.0
    match_win_probability: float = 0.0
    match_confidence_low: float = 0.0
    match_confidence_high: float = 0.0
    match_value_model: str = "policy_conditioned_exact_dynamic_program_v2"
    deal_margins: tuple[float, ...] = ()
    downside_cvar: float = 0.0
    volatility: float = 0.0
    principal_variation: tuple[str, ...] = ()
    search_nodes: int = 0
    search_completion_status: str = "failed"

    def __post_init__(self) -> None:
        if self.search_completion_status not in SEARCH_COMPLETION_STATUSES:
            raise ValueError(
                "unsupported search completion status: "
                f"{self.search_completion_status}"
            )

    # Source-compatible aliases. They are retained for older SDK clients but
    # are explicitly marked deprecated in serialized output.
    @property
    def expected_match_margin(self) -> float:
        return self.expected_round_margin

    @property
    def expected_team_points(self) -> float:
        return self.expected_team_awarded_points

    @property
    def expected_opponent_points(self) -> float:
        return self.expected_opponent_awarded_points

    @property
    def win_probability(self) -> float:
        return self.round_lead_probability

    def to_dict(self) -> dict[str, Any]:
        return {
            "card": self.card.to_dict(),
            "expectedRoundMargin": self.expected_round_margin,
            "expectedTeamAwardedPoints": self.expected_team_awarded_points,
            "expectedOpponentAwardedPoints": self.expected_opponent_awarded_points,
            "roundLeadProbability": self.round_lead_probability,
            "contractSuccessProbability": self.contract_success_probability,
            "matchWinProbability": self.match_win_probability,
            "matchWinConfidence95": [self.match_confidence_low, self.match_confidence_high],
            "matchValueModel": self.match_value_model,
            "drawProbability": self.draw_probability,
            "roundLeadConfidence95": [self.confidence_low, self.confidence_high],
            "bestShare": self.best_share,
            "samples": self.samples,
            "effectiveSamples": self.effective_samples,
            "exactShare": self.exact_share,
            "distribution": {
                "downsideCvar10": self.downside_cvar,
                "volatility": self.volatility,
                "dealCount": len(self.deal_margins),
            },
            "principalVariation": list(self.principal_variation),
            "searchNodes": self.search_nodes,
            "searchCompletionStatus": self.search_completion_status,
            "deprecatedAliases": {
                "expectedMatchMargin": self.expected_round_margin,
                "expectedTeamPoints": self.expected_team_awarded_points,
                "expectedOpponentPoints": self.expected_opponent_awarded_points,
                "winProbability": self.round_lead_probability,
            },
            # Retained at top level for old SDKs during the V7 compatibility window.
            "expectedMatchMargin": self.expected_round_margin,
            "expectedTeamPoints": self.expected_team_awarded_points,
            "expectedOpponentPoints": self.expected_opponent_awarded_points,
            "winProbability": self.round_lead_probability,
            "confidence95": [self.confidence_low, self.confidence_high],
        }


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    observer: Seat
    recommended_card: Card
    actions: tuple[ActionEvaluation, ...]
    deductions: tuple[HardDeduction, ...]
    signals: tuple[SignalObservation, ...]
    method: str
    hidden_deals: int
    effective_sample_size: float
    difficulty: int
    debate_score: int
    conventions_applied: bool
    explanation: tuple[str, ...] = field(default_factory=tuple)
    objective: str = "match_win_probability"
    analysis_profile_id: str = ""
    analysis_profile_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "observer": self.observer.name.lower(),
            "recommendedCard": self.recommended_card.to_dict(),
            "actions": [a.to_dict() for a in self.actions],
            "deductions": [d.to_dict() for d in self.deductions],
            "signals": [s.to_dict() for s in self.signals],
            "method": self.method,
            "hiddenDeals": self.hidden_deals,
            "effectiveSampleSize": self.effective_sample_size,
            "difficulty": self.difficulty,
            "debateScore": self.debate_score,
            "conventionsApplied": self.conventions_applied,
            "explanation": list(self.explanation),
            "objective": self.objective,
            "analysisProfile": {
                "id": self.analysis_profile_id,
                "version": self.analysis_profile_version,
            },
        }
