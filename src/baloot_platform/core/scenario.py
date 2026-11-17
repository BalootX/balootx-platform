from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import yaml

from baloot_platform.config import platform_root
from baloot_platform.analysis.evaluator import analyze
from baloot_platform.analysis.models import AnalysisResult, InformationState

from .bots import BotProfile, HeuristicBot
from .episode_copy import governed_episode_text
from .engine import BalootEngine
from .enums import Contract, Seat, Suit
from .models import RoundState
from .rules import RuleBook


@dataclass(frozen=True, slots=True)
class EscalationRequest:
    action: str
    seat: Seat
    mode: str | None = None


@dataclass(frozen=True, slots=True)
class ScenarioRequest:
    seed: int
    viewer: Seat
    dealer: Seat
    round_number: int
    cut_position: int | None
    contract: Contract | None
    buyer: Seat | None
    trump_suit: Suit | None
    match_score: tuple[int, int]
    escalation: tuple[EscalationRequest, ...]
    escalation_level: str
    escalation_mode: str
    require_project: bool
    pause_trick_min: int
    pause_trick_max: int
    requested_samples: int
    exact_allocation_limit: int
    exact_search_max_remaining: int
    maximum_sampling_attempts: int
    exact_search_max_nodes: int
    exact_search_max_seconds: float
    exact_search_max_depth: int
    exact_search_max_cache_entries: int
    max_seed_attempts: int
    max_candidates_per_seed: int
    minimum_hidden_deals: int
    minimum_expected_margin_gap: float
    minimum_win_probability_gap: float
    require_exact_analysis: bool
    target_debate_score: int
    agent_profiles: tuple[tuple[str, str], ...]

    @classmethod
    def governed(
        cls,
        profile: dict[str, Any] | None = None,
        **overrides: Any,
    ) -> "ScenarioRequest":
        if profile is None:
            profile = yaml.safe_load(
                (
                    platform_root()
                    / "configs/challenge/progressive_mining_v1.yaml"
                ).read_text(encoding="utf-8")
            )
        raw = dict(profile["scenario_defaults"])
        unknown = set(overrides) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(
                f"unknown scenario request keys: {sorted(unknown)}"
            )
        raw.update(overrides)
        raw["viewer"] = _enum_value(Seat, raw["viewer"])
        raw["dealer"] = _enum_value(Seat, raw["dealer"])
        raw["contract"] = _optional_enum_value(Contract, raw["contract"])
        raw["buyer"] = _optional_enum_value(Seat, raw["buyer"])
        raw["trump_suit"] = _optional_enum_value(Suit, raw["trump_suit"])
        raw["match_score"] = tuple(int(item) for item in raw["match_score"])
        raw["escalation"] = tuple(
            item
            if isinstance(item, EscalationRequest)
            else EscalationRequest(
                action=str(item["action"]),
                seat=_enum_value(Seat, item["seat"]),
                mode=item.get("mode"),
            )
            for item in raw["escalation"]
        )
        raw["agent_profiles"] = tuple(
            tuple(str(part) for part in item) for item in raw["agent_profiles"]
        )
        if int(raw["pause_trick_min"]) > int(raw["pause_trick_max"]):
            raise ValueError("pause trick minimum cannot exceed maximum")
        positive_fields = (
            "round_number", "requested_samples", "exact_allocation_limit",
            "maximum_sampling_attempts", "exact_search_max_nodes",
            "exact_search_max_seconds", "exact_search_max_depth",
            "exact_search_max_cache_entries", "max_seed_attempts",
            "max_candidates_per_seed", "minimum_hidden_deals",
        )
        if any(float(raw[key]) <= 0 for key in positive_fields):
            raise ValueError("scenario resource values must be positive")
        if not 0.0 <= float(raw["minimum_win_probability_gap"]) <= 1.0:
            raise ValueError("minimum win-probability gap must be in [0, 1]")
        return cls(**raw)


def _enum_value(enum_type: Any, value: Any) -> Any:
    if isinstance(value, enum_type):
        return value
    return enum_type[str(value).upper()]


def _optional_enum_value(enum_type: Any, value: Any) -> Any:
    return None if value is None else _enum_value(enum_type, value)


@dataclass(frozen=True, slots=True)
class EpisodeSpec:
    schema_version: int
    episode_id: str
    bundle_hash: str
    viewer: Seat
    state_before_pause: RoundState
    final_state: RoundState
    pause_before_play_index: int
    analysis: AnalysisResult
    title_ar: str
    question_ar: str
    game_serial: str | None = None
    game_record_hash: str | None = None
    challenge_id: str | None = None
    analysis_id: str | None = None
    model_id: str | None = None
    audit_id: str | None = None
    decision_id: str | None = None
    decision_index: tuple[tuple[str, str], ...] = ()
    audit_summary: dict[str, Any] | None = None
    playback_mode: str = "challenge_branch"

    def to_dict(self) -> dict[str, Any]:
        final = self.final_state
        return {
            "schemaVersion": self.schema_version,
            "episodeId": self.episode_id,
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
                "auditId": self.audit_id,
                "decisionId": self.decision_id,
            },
            "decisionIndex": [
                {"label": label, "decisionId": decision_id}
                for label, decision_id in self.decision_index
            ],
            "auditSummary": self.audit_summary,
            "playbackMode": self.playback_mode,
            "shuffle": final.shuffle.to_dict(reveal_order=True) if final.shuffle else None,
            "dealTrace": [r.to_dict(reveal_cards=True) for r in final.deal_trace],
            "roundNumber": final.round_number,
            "dealer": final.dealer.name.lower(),
            "cutter": (
                final.shuffle.cutter.name.lower()
                if final.shuffle is not None and final.shuffle.cutter is not None
                else None
            ),
            "buyer": final.buyer.name.lower() if final.buyer is not None else None,
            "declarer": final.declarer.name.lower() if final.declarer is not None else None,
            "contract": final.contract.value if final.contract is not None else None,
            "trumpSuit": final.trump_suit.name.lower() if final.trump_suit is not None else None,
            "firstLeader": final.first_leader.name.lower() if final.first_leader is not None else None,
            "purchaseCard": final.purchase_card.to_dict() if final.purchase_card else None,
            "initialFive": {
                seat.name.lower(): [c.to_dict() for c in self._mask_cards(final.initial_five[int(seat)])]
                for seat in Seat
            },
            "initialHands": {
                seat.name.lower(): [c.to_dict() for c in self._mask_cards(final.initial_hands[int(seat)])]
                for seat in Seat
            },
            "biddingHistory": [b.to_dict() for b in final.bid_history],
            "winningBidder": final.winning_bidder.name.lower() if final.winning_bidder is not None else None,
            "contractOwner": final.contract_owner.name.lower() if final.contract_owner is not None else None,
            "purchaseRecipient": final.purchase_recipient.name.lower() if final.purchase_recipient is not None else None,
            "escalation": final.escalation,
            "escalationHistory": list(final.escalation_history),
            "locked": final.locked,
            "projects": [p.to_dict() for p in final.projects],
            "projectWinningTeam": final.project_winning_team.name.lower() if final.project_winning_team is not None else None,
            "pauseBeforePlayIndex": self.pause_before_play_index,
            "analysis": self.analysis.to_dict(),
            "completedTricks": [t.to_dict() for t in final.completed_tricks],
            "roundScore": final.round_score.to_dict(),
            "roundResult": {
                "winner": final.round_score.winner.name.lower(),
                "buyerSucceeded": final.round_score.winner == final.buyer.team,
                "awardedPoints": list(final.round_score.awarded_points),
                "kabout": final.round_score.kabout,
                "gahwa": final.round_score.gahwa,
            },
            "matchScoreBefore": list(self.state_before_pause.match_score),
            "matchScoreAfter": list(final.match_score),
            "matchWinner": final.match_winner.name.lower() if final.match_winner is not None else None,
            "events": [e.to_dict() for e in final.events],
        }

    @staticmethod
    def _mask_cards(mask: int):
        from .cards import mask_to_cards

        return mask_to_cards(mask)


def _strategy_card(
    state: RoundState,
    engine: BalootEngine,
    request: ScenarioRequest,
    *,
    agent_registry: Any | None,
    convention: dict[str, Any],
    seed: int,
    cancel_check: Callable[[], bool] | None = None,
):
    if cancel_check is not None and cancel_check():
        raise InterruptedError("scenario generation cancelled")
    if agent_registry is None or state.current_player is None:
        return HeuristicBot(engine, BotProfile(deterministic_seed=seed)).choose(state)
    from baloot_platform.agents.profiles import AdaptivePlayStrategy

    selected = dict(request.agent_profiles)
    seat_key = state.current_player.name.lower()
    profile_id = selected.get(seat_key, agent_registry.default_lineup.get(seat_key, agent_registry.default_profile))
    strategy = AdaptivePlayStrategy(agent_registry.get(profile_id), convention)
    return strategy.choose_card(
        engine,
        state,
        seed=seed,
        cancel_check=cancel_check,
    )


def _collect_candidates(
    state: RoundState,
    engine: BalootEngine,
    request: ScenarioRequest,
    *,
    agent_registry: Any | None = None,
    convention: dict[str, Any] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[RoundState, ...]:
    """Collect decision states without changing the authoritative round.

    A fourth-hand decision is preferred visually, but third- and second-hand
    decisions are retained because they can be substantially stronger
    strategic puzzles.  Candidate selection happens only after the
    information-safe analyzer has compared legal actions.
    """
    result = state.clone()
    convention = dict(convention or {})
    candidates: list[RoundState] = []
    while result.current_player is not None:
        if cancel_check is not None and cancel_check():
            raise InterruptedError("scenario generation cancelled")
        trick_index = len(result.completed_tricks)
        legal = engine.legal_moves(result)
        if (
            result.current_player == request.viewer
            and request.pause_trick_min <= trick_index <= request.pause_trick_max
            and len(legal) >= 2
            and result.current_trick is not None
            and len(result.current_trick.plays) >= 1
        ):
            candidates.append(result.clone())
        engine.play_card(
            result,
            _strategy_card(
                result,
                engine,
                request,
                agent_registry=agent_registry,
                convention=convention,
                seed=result.seed + len(result.played_cards) * 104729 + int(result.current_player or 0),
                cancel_check=cancel_check,
            ),
        )
    # Visually rich states first, then earlier states with more uncertainty.
    candidates.sort(
        key=lambda item: (
            len(item.current_trick.plays) if item.current_trick else 0,
            -len(item.played_cards),
        ),
        reverse=True,
    )
    return tuple(candidates[: max(1, request.max_candidates_per_seed)])


def _analysis_quality(state: RoundState, result: AnalysisResult, request: ScenarioRequest) -> tuple[bool, float]:
    if len(result.actions) < 2:
        return False, float("-inf")
    best, second = result.actions[0], result.actions[1]
    margin_gap = best.expected_match_margin - second.expected_match_margin
    win_gap = best.win_probability - second.win_probability
    exact_share = min(action.exact_share for action in result.actions[:2])
    passes = (
        result.hidden_deals >= request.minimum_hidden_deals
        and margin_gap >= request.minimum_expected_margin_gap
        and win_gap >= request.minimum_win_probability_gap
        and (not request.require_exact_analysis or exact_share >= 0.999999)
    )
    table_context = len(state.current_trick.plays) if state.current_trick else 0
    score = (
        4.0 * margin_gap
        + 30.0 * win_gap
        + min(8.0, result.hidden_deals / 40.0)
        + 5.0 * exact_share
        + 1.4 * table_context
        - abs(result.debate_score - request.target_debate_score) / 18.0
    )
    return passes, score

def generate_episode(
    rules: RuleBook,
    convention: dict[str, Any],
    *,
    bundle_hash: str,
    request: ScenarioRequest,
    copy: Mapping[str, Any] | None = None,
    agent_registry: Any | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> EpisodeSpec:
    engine = BalootEngine(rules)
    last_error: Exception | None = None
    for offset in range(request.max_seed_attempts):
        if cancel_check is not None and cancel_check():
            raise InterruptedError("scenario generation cancelled")
        seed = request.seed + offset
        try:
            state = engine.create_round(
                seed=seed,
                dealer=request.dealer,
                match_score=request.match_score,
                round_number=request.round_number,
            )
            engine.shuffle_and_initial_deal(
                state,
                cut_position=request.cut_position,
            )
            if not engine.bid_and_complete_deal(
                state,
                forced_contract=request.contract,
                forced_buyer=request.buyer,
                forced_trump=request.trump_suit,
            ):
                continue
            escalations = list(request.escalation)
            if request.escalation_level != "normal" and not escalations:
                assert state.declarer is not None
                opponent = state.declarer.next()
                escalations.append(EscalationRequest("double", opponent, request.escalation_mode))
                if request.escalation_level in {"triple", "four", "gahwa"}:
                    escalations.append(EscalationRequest("triple", state.declarer))
                if request.escalation_level in {"four", "gahwa"}:
                    escalations.append(EscalationRequest("four", opponent, request.escalation_mode))
                if request.escalation_level == "gahwa":
                    escalations.append(EscalationRequest("gahwa", state.declarer))
            for escalation in escalations:
                engine.apply_escalation(
                    state,
                    seat=escalation.seat,
                    action=escalation.action,
                    mode=escalation.mode,
                )
            engine.prepare_projects(state)
            if request.require_project and not state.projects:
                continue
            candidates = _collect_candidates(
                state,
                engine,
                request,
                agent_registry=agent_registry,
                convention=convention,
                cancel_check=cancel_check,
            )
            if not candidates:
                continue
            selected: tuple[RoundState, AnalysisResult, float] | None = None
            for candidate_index, pause in enumerate(candidates):
                if cancel_check is not None and cancel_check():
                    raise InterruptedError(
                        "scenario generation cancelled"
                    )
                info = InformationState.from_complete(pause, request.viewer)
                result = analyze(
                    info,
                    rules,
                    convention,
                    requested_samples=request.requested_samples,
                    exact_allocation_limit=request.exact_allocation_limit,
                    exact_search_max_remaining=request.exact_search_max_remaining,
                    maximum_sampling_attempts=request.maximum_sampling_attempts,
                    exact_search_max_nodes=request.exact_search_max_nodes,
                    exact_search_max_seconds=request.exact_search_max_seconds,
                    exact_search_max_depth=request.exact_search_max_depth,
                    exact_search_max_cache_entries=(
                        request.exact_search_max_cache_entries
                    ),
                    seed=seed * 1009 + candidate_index,
                    cancel_check=cancel_check,
                )
                passes, quality = _analysis_quality(pause, result, request)
                if selected is None or quality > selected[2]:
                    selected = (pause, result, quality)
                if passes:
                    selected = (pause, result, quality)
                    break
            if selected is None:
                continue
            pause, result, _quality = selected
            passes, _ = _analysis_quality(pause, result, request)
            if not passes:
                continue
            final = pause.clone()
            pause_index = len(final.played_cards)
            engine.play_card(final, result.recommended_card)
            while final.current_player is not None:
                engine.play_card(
                    final,
                    _strategy_card(
                        final,
                        engine,
                        request,
                        agent_registry=agent_registry,
                        convention=convention,
                        seed=seed + 99 + len(final.played_cards) * 65537 + int(final.current_player),
                        cancel_check=cancel_check,
                    ),
                )
            episode_id = (
                f"baloot-{seed}-{final.contract.value}-{pause_index}-"
                f"{rules.sha256[:8]}"
            )
            return EpisodeSpec(
                schema_version=4,
                episode_id=episode_id,
                bundle_hash=bundle_hash,
                viewer=request.viewer,
                state_before_pause=pause,
                final_state=final,
                pause_before_play_index=pause_index,
                analysis=result,
                title_ar=governed_episode_text(
                    "video.decision.generated_title",
                    copy=copy,
                    number=str(seed),
                ),
                question_ar=governed_episode_text(
                    "video.decision.generated_question",
                    copy=copy,
                ),
            )
        except (InterruptedError, TimeoutError):
            raise
        except Exception as exc:
            # Candidate search is deterministic; invalid candidates are skipped,
            # but the final error is retained for diagnosis.
            last_error = exc
            continue
    raise RuntimeError(
        "could not generate a publishable scenario after "
        f"{request.max_seed_attempts} seeds: {last_error}"
    )
