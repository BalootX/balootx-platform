"""Post-reveal actual-deal oracle used only for causal verification."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from baloot_platform.analysis.search import exact_team_value
from baloot_platform.core.cards import Card, iter_mask
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Seat, Suit, Team
from baloot_platform.core.models import RoundState
from baloot_platform.core.rules import RuleBook
from baloot_platform.plugins.base import PluginDescriptor

from .analysis_pipeline import TraceDecisionAnalysisV3
from .evidence_v3 import TruthMode, canonical_sha256
from .replay import ReplayDecisionStateV1, TraceDecisionProjectionStoreV1
from .trace import CompleteGameTraceV1
from .tactical_evidence import (
    CausalActionLineV1,
    CausalTacticalDetectorV1,
    TacticalEvidenceV1,
    TacticalKind,
)


@dataclass(frozen=True, slots=True)
class OracleTrickFactV1:
    ordinal: int
    leader: Seat
    winner: Seat
    lead_card: str
    winning_card: str
    winning_card_strength: float
    plays: tuple[tuple[str, str], ...]
    relative_points: float
    actor_team_trumps_remaining: int
    opponent_trumps_remaining: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "leader": self.leader.name.lower(),
            "winner": self.winner.name.lower(),
            "leadCard": self.lead_card,
            "winningCard": self.winning_card,
            "winningCardStrength": self.winning_card_strength,
            "plays": [{"seat": seat, "card": card} for seat, card in self.plays],
            "relativePoints": self.relative_points,
            "actorTeamTrumpsRemaining": self.actor_team_trumps_remaining,
            "opponentTrumpsRemaining": self.opponent_trumps_remaining,
        }


@dataclass(frozen=True, slots=True)
class ActualDealActionOracleV1:
    action: Mapping[str, Any]
    card_strength: float
    immediate_margin: float
    final_margin: float | None
    contract_success: bool | None
    creates_kabout: bool | None
    allows_opponent_kabout: bool | None
    search_nodes: int
    principal_variation: tuple[str, ...]
    future_tricks: tuple[OracleTrickFactV1, ...]
    completion_status: str
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "action": dict(self.action),
            "cardStrength": self.card_strength,
            "immediateMargin": self.immediate_margin,
            "finalMargin": self.final_margin,
            "contractSuccess": self.contract_success,
            "createsKabout": self.creates_kabout,
            "allowsOpponentKabout": self.allows_opponent_kabout,
            "searchNodes": self.search_nodes,
            "principalVariation": list(self.principal_variation),
            "futureTricks": [item.to_dict() for item in self.future_tricks],
            "completionStatus": self.completion_status,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ActualDealDecisionOracleV1:
    decision_id: str
    truth_mode: TruthMode
    observer_recommendation: Mapping[str, Any]
    actions: tuple[ActualDealActionOracleV1, ...]
    complete: bool
    sha256: str
    observer_recommendation_unchanged: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "decisionId": self.decision_id,
            "truthMode": self.truth_mode.value,
            "observerRecommendation": dict(self.observer_recommendation),
            "observerRecommendationUnchanged": (
                self.observer_recommendation_unchanged
            ),
            "actions": [item.to_dict() for item in self.actions],
            "complete": self.complete,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class TacticalVerificationResultV1:
    oracle: ActualDealDecisionOracleV1
    evidence: tuple[TacticalEvidenceV1, ...]
    status: str
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "oracle": self.oracle.to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
            "status": self.status,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ExactLineCausalityV1:
    selected_action_hash: str
    alternative_action_hash: str
    mechanisms: tuple[str, ...]
    payoff_depth: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "selectedActionHash": self.selected_action_hash,
            "alternativeActionHash": self.alternative_action_hash,
            "mechanisms": list(self.mechanisms),
            "payoffDepth": self.payoff_depth,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ActualDealOpportunityV1:
    decision_id: str
    acting_seat: Seat
    remaining_cards: int
    best_action: Mapping[str, Any]
    alternative_action: Mapping[str, Any]
    best_margin: float | None
    alternative_margin: float | None
    outcome_gap: float
    complete: bool
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "decisionId": self.decision_id,
            "actingSeat": self.acting_seat.name.lower(),
            "remainingCards": self.remaining_cards,
            "bestAction": dict(self.best_action),
            "alternativeAction": dict(self.alternative_action),
            "bestMargin": self.best_margin,
            "alternativeMargin": self.alternative_margin,
            "outcomeGap": self.outcome_gap,
            "complete": self.complete,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class TacticalViabilityDecisionV1:
    """Exact actual-deal evidence used only to prove tactical impossibility."""

    decision_id: str
    acting_seat: Seat
    remaining_cards: int
    actions: tuple[ActualDealActionOracleV1, ...]
    evidence: tuple[TacticalEvidenceV1, ...]
    complete: bool
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "decisionId": self.decision_id,
            "actingSeat": self.acting_seat.name.lower(),
            "remainingCards": self.remaining_cards,
            "actions": [item.to_dict() for item in self.actions],
            "evidence": [item.to_dict() for item in self.evidence],
            "complete": self.complete,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class TacticalViabilityProbeV1:
    """Lossless preflight for a play-only, verified-tactical finalist request.

    A negative result is authoritative only when every eligible late play was
    solved exactly. A positive result may stop at the first viable decision;
    the normal observer-safe finalist analysis still decides what is shown.
    """

    trace_hash: str
    required_tactics: tuple[str, ...]
    eligible_decision_count: int
    scanned_decisions: tuple[TacticalViabilityDecisionV1, ...]
    scan_complete: bool
    viable: bool
    status: str
    sha256: str

    @property
    def can_reject(self) -> bool:
        return self.scan_complete and not self.viable

    @property
    def action_oracles_by_decision(
        self,
    ) -> Mapping[str, tuple[ActualDealActionOracleV1, ...]]:
        return {
            item.decision_id: item.actions
            for item in self.scanned_decisions
            if item.complete
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "traceHash": self.trace_hash,
            "requiredTactics": list(self.required_tactics),
            "eligibleDecisionCount": self.eligible_decision_count,
            "scannedDecisions": [item.to_dict() for item in self.scanned_decisions],
            "scanComplete": self.scan_complete,
            "viable": self.viable,
            "status": self.status,
            "sha256": self.sha256,
        }


def derive_exact_line_causality(
    selected: ActualDealActionOracleV1,
    alternative: ActualDealActionOracleV1,
    *,
    acting_seat: Seat,
    trump_suit: Suit | None,
    low_card_max_strength: float = 35.0,
) -> ExactLineCausalityV1:
    """Derive only mechanisms supported by paired exact continuation facts."""
    if selected.final_margin is None or alternative.final_margin is None:
        raise ValueError("exact line causality requires completed action oracles")
    selected_card = Card.parse(dict(selected.action))
    selected_facts = selected.future_tricks
    alternative_facts = alternative.future_tricks
    final_gain = selected.final_margin - alternative.final_margin
    mechanisms: set[str] = set()
    payoff_depth = 0
    selected_cumulative = 0.0
    alternative_cumulative = 0.0
    for index, fact in enumerate(selected_facts):
        selected_cumulative += fact.relative_points
        if index < len(alternative_facts):
            alternative_cumulative += alternative_facts[index].relative_points
        if payoff_depth == 0 and selected_cumulative > alternative_cumulative:
            payoff_depth = index + 1
    if final_gain > 0 and payoff_depth == 0:
        payoff_depth = max(1, len(selected_facts))

    if final_gain > 0:
        for index, fact in enumerate(selected_facts):
            other = alternative_facts[index] if index < len(alternative_facts) else None
            if (
                index + 1 >= payoff_depth
                and fact.winner.team == acting_seat.team
                and (other is None or other.winner.team != acting_seat.team)
            ):
                mechanisms.add("entry")
            if (
                fact.winner == acting_seat.partner
                and (other is None or other.winner != acting_seat.partner)
            ):
                mechanisms.add("partner_enablement")
            winning = Card.parse(fact.winning_card)
            other_winning = Card.parse(other.winning_card) if other is not None else None
            if (
                selected.card_strength <= low_card_max_strength
                and index > 0
                and fact.winner.team == acting_seat.team
                and winning.suit == selected_card.suit
                and fact.winning_card_strength <= low_card_max_strength
                and (
                    other is None
                    or other.winner.team != acting_seat.team
                    or other_winning is None
                    or other_winning.suit != selected_card.suit
                )
            ):
                mechanisms.add("promotion")
            if (
                index > 0
                and fact.leader.team == acting_seat.team.other
                and Card.parse(fact.lead_card).suit == selected_card.suit
                and fact.winner.team == acting_seat.team
                and (
                    other is None
                    or Card.parse(other.lead_card).suit != selected_card.suit
                    or other.winner.team != acting_seat.team
                )
            ):
                mechanisms.add("forced_return")
            if other is not None:
                if (
                    fact.opponent_trumps_remaining == 0
                    and other.opponent_trumps_remaining > 0
                ):
                    mechanisms.add("trump_exhaustion")
                if (
                    fact.actor_team_trumps_remaining
                    > other.actor_team_trumps_remaining
                ):
                    mechanisms.add("trump_preservation")
        if trump_suit is not None and selected_card.suit == trump_suit and selected_facts:
            first = selected_facts[0]
            if any(
                Seat.parse(seat).team == acting_seat.team.other
                and Card.parse(card).suit == trump_suit
                for seat, card in first.plays
            ):
                mechanisms.add("trump_extraction")
    core = {
        "schemaVersion": 1,
        "selectedActionHash": selected.sha256,
        "alternativeActionHash": alternative.sha256,
        "actingSeat": acting_seat.name.lower(),
        "trumpSuit": trump_suit.name.lower() if trump_suit is not None else None,
        "lowCardMaxStrength": float(low_card_max_strength),
        "mechanisms": sorted(mechanisms),
        "payoffDepth": payoff_depth,
    }
    return ExactLineCausalityV1(
        selected.sha256,
        alternative.sha256,
        tuple(core["mechanisms"]),
        payoff_depth,
        canonical_sha256(core),
    )


class ActualDealOracleVerifierV1:
    """Verify late-game causal claims without changing actor-safe truth."""

    descriptor = PluginDescriptor(
        id="actual_deal_oracle_verifier_v1",
        kind="tactical_verifier",
        version="1.3.0",
        description="Exact late-game actual-deal counterfactual verifier kept separate from actor truth",
        capabilities=(
            "actual_deal_oracle",
            "exact_late_game",
            "only_winning_move",
            "kabout_creation_prevention",
            "contract_rescue",
            "paired_trick_causality",
            "small_card_big_gain",
            "large_card_sacrifice",
            "entry_unblocking",
            "partner_enablement",
            "trump_control",
            "observer_recommendation_isolation",
            "hindsight_sensitivity_rejection",
            "discovery_only_opportunity_scan",
            "lossless_tactical_viability_preflight",
        ),
        deterministic=True,
        timeout_seconds=300,
        maximum_memory_mb=2048,
    )

    def __init__(
        self,
        rules: RuleBook,
        *,
        maximum_remaining_cards: int = 12,
        maximum_nodes: int = 250_000,
        maximum_seconds: float = 5.0,
        maximum_depth: int | None = None,
        maximum_cache_entries: int = 100_000,
        outcome_boundary: float = 0.0,
        detector: CausalTacticalDetectorV1 | None = None,
    ):
        if not 1 <= int(maximum_remaining_cards) <= 32:
            raise ValueError("oracle remaining-card limit must be in [1, 32]")
        if int(maximum_nodes) < 1 or float(maximum_seconds) <= 0:
            raise ValueError("oracle search budgets must be positive")
        if maximum_depth is not None and int(maximum_depth) < 1:
            raise ValueError("oracle maximum depth must be positive")
        if int(maximum_cache_entries) < 0:
            raise ValueError("oracle cache bound cannot be negative")
        self.rules = rules
        self.maximum_remaining_cards = int(maximum_remaining_cards)
        self.maximum_nodes = int(maximum_nodes)
        self.maximum_seconds = float(maximum_seconds)
        self.maximum_depth = int(maximum_depth) if maximum_depth is not None else None
        self.maximum_cache_entries = int(maximum_cache_entries)
        self.outcome_boundary = float(outcome_boundary)
        self.detector = detector or CausalTacticalDetectorV1()

    def _detect_from_actions(
        self,
        snapshot: ReplayDecisionStateV1,
        actions: tuple[ActualDealActionOracleV1, ...],
        selected_action: ActualDealActionOracleV1,
    ) -> tuple[TacticalEvidenceV1, ...]:
        """Apply the one governed causal detector to paired exact lines."""

        if (
            len(actions) < 2
            or selected_action.final_margin is None
            or any(item.final_margin is None for item in actions)
        ):
            return ()
        best_margin = max(float(item.final_margin) for item in actions)
        other_actions = tuple(item for item in actions if item is not selected_action)
        if not other_actions:
            return ()
        best_other_action = max(
            other_actions,
            key=lambda item: float(item.final_margin),
        )
        causality = derive_exact_line_causality(
            selected_action,
            best_other_action,
            acting_seat=snapshot.decision.acting_seat,
            trump_suit=snapshot.state.trump_suit,
            low_card_max_strength=self.detector.profile.low_card_max_strength,
        )
        lines: list[CausalActionLineV1] = []
        for item in actions:
            assert item.final_margin is not None
            preserves = (
                item.final_margin > self.outcome_boundary
                if best_margin > self.outcome_boundary
                else item.final_margin == best_margin
            )
            lines.append(CausalActionLineV1(
                action_id=str(item.action.get("code") or item.action.get("id")),
                card_strength=item.card_strength,
                immediate_margin=item.immediate_margin,
                final_margin=item.final_margin,
                preserves_outcome=preserves,
                kabout_probability=1.0 if item.creates_kabout else 0.0,
                opponent_kabout_probability=(
                    1.0 if item.allows_opponent_kabout else 0.0
                ),
                payoff_depth=(
                    causality.payoff_depth if item is selected_action else 0
                ),
                mechanisms=(
                    causality.mechanisms if item is selected_action else ()
                ),
                principal_variation=item.principal_variation,
                contract_success_probability=(
                    1.0 if item.contract_success else 0.0
                ),
                causal_source_hash=(
                    causality.sha256 if item is selected_action else None
                ),
            ))
        selected_index = actions.index(selected_action)
        selected = lines[selected_index]
        alternatives = tuple(
            item for index, item in enumerate(lines) if index != selected_index
        )
        return self.detector.detect(selected, alternatives)

    def _card_strength(self, state: RoundState, card: Card) -> float:
        assert state.contract is not None
        values = [
            self.rules.strength(Card.of(card.suit, rank), state.contract, state.trump_suit)
            for rank in card.rank.__class__
        ]
        low, high = min(values), max(values)
        value = self.rules.strength(card, state.contract, state.trump_suit)
        return 100.0 * (value - low) / max(1, high - low)

    @staticmethod
    def _immediate_margin(before: RoundState, after: RoundState, actor_team) -> float:
        if len(after.completed_tricks) == len(before.completed_tricks):
            return 0.0
        trick = after.completed_tricks[-1]
        return float(trick.raw_points if trick.winner.team == actor_team else -trick.raw_points)

    def _trick_fact(
        self,
        state: RoundState,
        *,
        baseline: int,
        actor_team: Team,
    ) -> OracleTrickFactV1:
        trick = state.completed_tricks[-1]
        assert trick.winner is not None and state.contract is not None
        winning_play = next(item for item in trick.plays if item.seat == trick.winner)
        trump = state.trump_suit if state.contract.effective.value == "hokum" else None

        def remaining(team: Team) -> int:
            if trump is None:
                return 0
            return sum(
                1
                for seat in Seat
                if seat.team == team
                for card in iter_mask(state.hands[int(seat)])
                if card.suit == trump
            )

        return OracleTrickFactV1(
            len(state.completed_tricks) - baseline,
            trick.leader,
            trick.winner,
            trick.plays[0].card.code,
            winning_play.card.code,
            self._card_strength(state, winning_play.card),
            tuple((item.seat.name.lower(), item.card.code) for item in trick.plays),
            float(trick.raw_points if trick.winner.team == actor_team else -trick.raw_points),
            remaining(actor_team),
            remaining(actor_team.other),
        )

    def _action(
        self,
        snapshot: ReplayDecisionStateV1,
        action: Mapping[str, Any],
        *,
        cancel_check: Callable[[], bool] | None,
    ) -> ActualDealActionOracleV1:
        engine = BalootEngine(self.rules)
        card = Card.parse(dict(action))
        child = snapshot.state.clone()
        baseline = len(child.completed_tricks)
        engine.play_card(child, card, emit_events=False)
        child.events = []
        immediate = self._immediate_margin(snapshot.state, child, snapshot.decision.acting_seat.team)
        facts: list[OracleTrickFactV1] = []
        if len(child.completed_tricks) > baseline:
            facts.append(self._trick_fact(
                child,
                baseline=baseline,
                actor_team=snapshot.decision.acting_seat.team,
            ))
        search = exact_team_value(
            child,
            engine,
            snapshot.decision.acting_seat.team,
            max_nodes=self.maximum_nodes,
            max_seconds=self.maximum_seconds,
            max_depth=self.maximum_depth,
            max_cache_entries=self.maximum_cache_entries,
            cancel_check=cancel_check,
        )
        final_margin: float | None = None
        contract_success: bool | None = None
        creates_kabout: bool | None = None
        allows_opponent_kabout: bool | None = None
        line = (card.code, *search.principal_variation)
        status = search.completion_status
        if search.exact:
            terminal = child.clone()
            try:
                for code in search.principal_variation:
                    completed = len(terminal.completed_tricks)
                    engine.play_card(terminal, Card.parse(code), emit_events=False)
                    if len(terminal.completed_tricks) > completed:
                        facts.append(self._trick_fact(
                            terminal,
                            baseline=baseline,
                            actor_team=snapshot.decision.acting_seat.team,
                        ))
                if terminal.current_player is not None or terminal.round_score is None:
                    raise ValueError("exact oracle principal variation is not terminal")
                final_margin = float(search.value)
                contract_success = bool(search.contract_succeeded)
                kabout = bool(terminal.round_score.kabout)
                winner = terminal.round_score.winner
                actor_team = snapshot.decision.acting_seat.team
                creates_kabout = kabout and winner == actor_team
                allows_opponent_kabout = kabout and winner == actor_team.other
            except (ValueError, AssertionError):
                status = "failed"
        core = {
            "schemaVersion": 1,
            "action": dict(action),
            "cardStrength": self._card_strength(snapshot.state, card),
            "immediateMargin": immediate,
            "finalMargin": final_margin,
            "contractSuccess": contract_success,
            "createsKabout": creates_kabout,
            "allowsOpponentKabout": allows_opponent_kabout,
            "searchNodes": search.nodes,
            "principalVariation": list(line),
            "futureTricks": [item.to_dict() for item in facts],
            "completionStatus": status,
        }
        return ActualDealActionOracleV1(
            dict(action),
            float(core["cardStrength"]),
            immediate,
            final_margin,
            contract_success,
            creates_kabout,
            allows_opponent_kabout,
            search.nodes,
            tuple(line),
            tuple(facts),
            status,
            canonical_sha256(core),
        )

    def verify_decision(
        self,
        snapshot: ReplayDecisionStateV1,
        actor_analysis: TraceDecisionAnalysisV3,
        *,
        cancel_check: Callable[[], bool] | None = None,
        precomputed_actions: tuple[ActualDealActionOracleV1, ...] | None = None,
    ) -> TacticalVerificationResultV1:
        if snapshot.decision.decision_id != actor_analysis.decision_id:
            raise ValueError("oracle snapshot and actor analysis refer to different decisions")
        if snapshot.decision.decision_kind != "play":
            raise ValueError("actual-deal tactical verification currently supports card play only")
        remaining = sum(mask.bit_count() for mask in snapshot.state.hands)
        if remaining > self.maximum_remaining_cards:
            raise ValueError("decision is outside the governed exact-oracle endgame window")
        actions = (
            tuple(precomputed_actions)
            if precomputed_actions is not None
            else tuple(
                self._action(snapshot, action, cancel_check=cancel_check)
                for action in snapshot.decision.legal_actions
            )
        )
        expected_action_ids = {
            Card.parse(dict(item)).id for item in snapshot.decision.legal_actions
        }
        actual_action_ids = {Card.parse(dict(item.action)).id for item in actions}
        if (
            len(actions) != len(snapshot.decision.legal_actions)
            or actual_action_ids != expected_action_ids
        ):
            raise ValueError(
                "precomputed actual-deal actions do not match the legal action set"
            )
        complete = bool(actions) and all(item.completion_status == "exact" for item in actions)
        oracle_core = {
            "schemaVersion": 1,
            "decisionId": snapshot.decision.decision_id,
            "truthMode": TruthMode.ACTUAL_DEAL_ORACLE.value,
            "observerRecommendation": dict(actor_analysis.best_action),
            "actionHashes": [item.sha256 for item in actions],
            "complete": complete,
        }
        recommendation_unchanged = False
        if complete:
            margins = [
                float(item.final_margin)
                for item in actions
                if item.final_margin is not None
            ]
            best_margin = max(margins)
            recommended_key = Card.parse(dict(actor_analysis.best_action)).id
            recommended_action = next(
                item
                for item in actions
                if Card.parse(dict(item.action)).id == recommended_key
            )
            recommendation_unchanged = (
                recommended_action.final_margin is not None
                and abs(float(recommended_action.final_margin) - best_margin)
                <= 1e-9
            )
        oracle_core["observerRecommendationUnchanged"] = recommendation_unchanged
        oracle = ActualDealDecisionOracleV1(
            snapshot.decision.decision_id,
            TruthMode.ACTUAL_DEAL_ORACLE,
            dict(actor_analysis.best_action),
            actions,
            complete,
            canonical_sha256(oracle_core),
            recommendation_unchanged,
        )
        evidence: tuple[TacticalEvidenceV1, ...] = ()
        status = "unresolved"
        if complete and recommendation_unchanged:
            recommended_key = Card.parse(dict(actor_analysis.best_action)).id
            selected_action = next(
                item for item in actions
                if Card.parse(dict(item.action)).id == recommended_key
            )
            evidence = self._detect_from_actions(
                snapshot,
                actions,
                selected_action,
            )
            status = "verified"
        elif complete:
            status = "hindsight_sensitive"
        core = {
            "schemaVersion": 1,
            "oracleHash": oracle.sha256,
            "evidenceHashes": [item.causal_proof_hash for item in evidence],
            "status": status,
        }
        return TacticalVerificationResultV1(
            oracle,
            evidence,
            status,
            canonical_sha256(core),
        )

    def probe_verified_tactical_play(
        self,
        trace: CompleteGameTraceV1,
        *,
        maximum_decisions: int | None = None,
        required_tactics: Iterable[str | TacticalKind] = (),
        cancel_check: Callable[[], bool] | None = None,
    ) -> TacticalViabilityProbeV1:
        """Prove that an exact-window tactical play exists or cannot exist.

        This probe never recommends a public action. It only short-circuits a
        play-only finalist when every eligible actual-deal line is exact and
        the same governed detector finds no requested tactical evidence for
        any tied best action. Unrequested evidence never ends the scan early.
        """

        required = tuple(sorted({
            (
                item.value
                if isinstance(item, TacticalKind)
                else TacticalKind(str(item)).value
            )
            for item in required_tactics
        }))
        store = TraceDecisionProjectionStoreV1(trace, self.rules)
        all_eligible = tuple(
            snapshot
            for snapshot in reversed(store.replay.decisions)
            if snapshot.decision.decision_kind == "play"
            and not snapshot.decision.forced
            and len(snapshot.decision.legal_actions) >= 2
            and sum(mask.bit_count() for mask in snapshot.state.hands)
            <= self.maximum_remaining_cards
        )
        if maximum_decisions is not None and not 1 <= int(maximum_decisions) <= 128:
            raise ValueError("tactical viability decision bound is out of range")
        eligible = (
            all_eligible[: int(maximum_decisions)]
            if maximum_decisions is not None
            else all_eligible
        )
        scanned: list[TacticalViabilityDecisionV1] = []
        viable = False
        inconclusive = False
        for snapshot in eligible:
            if cancel_check is not None and cancel_check():
                raise InterruptedError("tactical viability probe cancelled")
            actions = tuple(
                self._action(snapshot, action, cancel_check=cancel_check)
                for action in snapshot.decision.legal_actions
            )
            complete = bool(actions) and all(
                item.completion_status == "exact"
                and item.final_margin is not None
                for item in actions
            )
            evidence: list[TacticalEvidenceV1] = []
            if complete:
                best_margin = max(float(item.final_margin) for item in actions)
                tied_best = tuple(
                    item for item in actions
                    if abs(float(item.final_margin) - best_margin) <= 1e-9
                )
                for selected in sorted(tied_best, key=lambda item: item.sha256):
                    evidence.extend(
                        self._detect_from_actions(snapshot, actions, selected)
                    )
            else:
                inconclusive = True
            evidence_by_hash = {
                item.causal_proof_hash: item for item in evidence
            }
            unique_evidence = tuple(
                evidence_by_hash[key] for key in sorted(evidence_by_hash)
            )
            matching_evidence = tuple(
                item
                for item in unique_evidence
                if not required or item.kind.value in required
            )
            remaining = sum(mask.bit_count() for mask in snapshot.state.hands)
            core = {
                "schemaVersion": 1,
                "decisionId": snapshot.decision.decision_id,
                "actingSeat": snapshot.decision.acting_seat.name.lower(),
                "remainingCards": remaining,
                "actionHashes": [item.sha256 for item in actions],
                "evidenceHashes": [
                    item.causal_proof_hash for item in unique_evidence
                ],
                "complete": complete,
            }
            scanned.append(TacticalViabilityDecisionV1(
                snapshot.decision.decision_id,
                snapshot.decision.acting_seat,
                remaining,
                actions,
                unique_evidence,
                complete,
                canonical_sha256(core),
            ))
            if matching_evidence:
                viable = True
                break
        scan_complete = bool(
            not viable
            and not inconclusive
            and len(scanned) == len(all_eligible)
        )
        status = (
            "viable"
            if viable
            else (
                "no_verified_tactical_play_opportunity"
                if scan_complete
                else "inconclusive"
            )
        )
        core = {
            "schemaVersion": 1,
            "traceHash": trace.sha256,
            "requiredTactics": list(required),
            "eligibleDecisionCount": len(all_eligible),
            "scannedDecisionHashes": [item.sha256 for item in scanned],
            "scanComplete": scan_complete,
            "viable": viable,
            "status": status,
        }
        return TacticalViabilityProbeV1(
            trace.sha256,
            required,
            len(all_eligible),
            tuple(scanned),
            scan_complete,
            viable,
            status,
            canonical_sha256(core),
        )

    def discover_opportunities(
        self,
        trace: CompleteGameTraceV1,
        *,
        maximum_decisions: int,
        cancel_check: Callable[[], bool] | None = None,
    ) -> tuple[ActualDealOpportunityV1, ...]:
        """Use exact hidden-state lines only to discover finalist traces.

        This never produces the public recommendation. The acting player's
        observer-safe analysis must independently agree during verification.
        """
        if not 1 <= int(maximum_decisions) <= 128:
            raise ValueError("opportunity scan decision bound is out of range")
        store = TraceDecisionProjectionStoreV1(trace, self.rules)
        eligible = [
            snapshot
            for snapshot in reversed(store.replay.decisions)
            if snapshot.decision.decision_kind == "play"
            and not snapshot.decision.forced
            and len(snapshot.decision.legal_actions) >= 2
            and sum(mask.bit_count() for mask in snapshot.state.hands)
            <= self.maximum_remaining_cards
        ][: int(maximum_decisions)]
        opportunities: list[ActualDealOpportunityV1] = []
        for snapshot in eligible:
            if cancel_check is not None and cancel_check():
                raise InterruptedError("actual-deal opportunity scan cancelled")
            actions = tuple(
                self._action(snapshot, action, cancel_check=cancel_check)
                for action in snapshot.decision.legal_actions
            )
            complete = bool(actions) and all(
                item.completion_status == "exact"
                and item.final_margin is not None
                for item in actions
            )
            ranked = sorted(
                actions,
                key=lambda item: (
                    -float(item.final_margin)
                    if item.final_margin is not None
                    else float("inf"),
                    canonical_sha256(dict(item.action)),
                ),
            )
            best, alternative = ranked[:2]
            gap = (
                max(0.0, float(best.final_margin) - float(alternative.final_margin))
                if complete
                else 0.0
            )
            core = {
                "schemaVersion": 1,
                "decisionId": snapshot.decision.decision_id,
                "actingSeat": snapshot.decision.acting_seat.name.lower(),
                "remainingCards": sum(
                    mask.bit_count() for mask in snapshot.state.hands
                ),
                "bestActionHash": best.sha256,
                "alternativeActionHash": alternative.sha256,
                "bestMargin": best.final_margin,
                "alternativeMargin": alternative.final_margin,
                "outcomeGap": gap,
                "complete": complete,
            }
            opportunities.append(ActualDealOpportunityV1(
                snapshot.decision.decision_id,
                snapshot.decision.acting_seat,
                int(core["remainingCards"]),
                dict(best.action),
                dict(alternative.action),
                best.final_margin,
                alternative.final_margin,
                gap,
                complete,
                canonical_sha256(core),
            ))
        return tuple(opportunities)
