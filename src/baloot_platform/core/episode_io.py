"""Validated import/export helpers for immutable EpisodeSpec documents.

This module lets archived games and videos be re-rendered from their canonical
JSON without relying on Python pickle or on an in-memory session.  The loader
legally replays the public card sequence against the active RuleBook before it
constructs an EpisodeSpec.  Presentation metadata is restored only after the
rules engine has accepted the lifecycle and result.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

from baloot_platform.analysis.models import (
    ActionEvaluation,
    AnalysisResult,
    HardDeduction,
    SignalObservation,
)
from baloot_platform.core.cards import Card, cards_to_mask
from baloot_platform.core.episode_copy import governed_episode_text
from baloot_platform.core.enums import Contract, GamePhase, Seat, Suit, Team
from baloot_platform.core.models import BidRecord, DealRecord, Event, Project, RoundScore
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.scenario import EpisodeSpec
from baloot_platform.core.serialization import replay_trusted_context
from baloot_platform.core.shuffle import ShuffleRecord


def _team(value: str | int | Team | None) -> Team | None:
    if value is None or isinstance(value, Team):
        return value
    if isinstance(value, int):
        return Team(value)
    normalized = str(value).strip().lower()
    aliases = {
        "north_south": Team.NORTH_SOUTH,
        "north-south": Team.NORTH_SOUTH,
        "us": Team.NORTH_SOUTH,
        "west_east": Team.WEST_EAST,
        "west-east": Team.WEST_EAST,
        "them": Team.WEST_EAST,
    }
    if normalized not in aliases:
        raise ValueError(f"unknown team: {value}")
    return aliases[normalized]


def _card(value: Any) -> Card:
    return Card.parse(value)


def _cards(values: Any) -> tuple[Card, ...]:
    return tuple(_card(value) for value in (values or ()))


def _trusted_context(data: Mapping[str, Any], play_count: int) -> dict[str, Any]:
    all_plays = [play for trick in data.get("completedTricks", ()) for play in trick.get("plays", ())]
    selected = all_plays[:play_count]
    complete_count = len(selected) // 4
    completed: list[dict[str, Any]] = []
    for index in range(complete_count):
        source = data["completedTricks"][index]
        completed.append({
            "index": index,
            "leader": source["leader"],
            "plays": copy.deepcopy(selected[index * 4 : index * 4 + 4]),
            "winner": source.get("winner"),
            "rawPoints": source.get("rawPoints", 0),
        })
    current = None
    remainder = selected[complete_count * 4 :]
    if remainder:
        source = data["completedTricks"][complete_count]
        current = {
            "index": complete_count,
            "leader": source["leader"],
            "plays": copy.deepcopy(remainder),
            "winner": None,
            "rawPoints": 0,
        }
    return {
        "seed": int((data.get("shuffle") or {}).get("seed", data.get("seed", 0))),
        "roundNumber": int(data.get("roundNumber", 1)),
        "dealer": data["dealer"],
        "buyer": data["buyer"],
        "purchaseRecipient": data.get("purchaseRecipient", data["buyer"]),
        "declarer": data.get("declarer", data["buyer"]),
        "contract": data["contract"],
        "trumpSuit": data.get("trumpSuit"),
        "firstLeader": data["firstLeader"],
        "initialHands": data["initialHands"],
        "completedTricks": completed,
        "currentTrick": current,
        # The authoritative replay derives the next actor from the event prefix.
        # Supplying it here would duplicate state and could conceal tampering.
        "currentPlayer": None,
        "matchScoreBefore": data.get("matchScoreBefore", [0, 0]),
        "escalation": data.get("escalation", "normal"),
        "escalationHistory": data.get("escalationHistory", []),
        "locked": bool(data.get("locked", False)),
    }


def _restore_lifecycle(state: Any, data: Mapping[str, Any], *, final: bool) -> None:
    shuffle = data.get("shuffle") or {}
    if shuffle:
        state.shuffle = ShuffleRecord(
            str(shuffle["algorithm"]),
            int(shuffle["seed"]),
            Seat.parse(shuffle["dealer"]),
            Seat.parse(shuffle["cutter"]) if shuffle.get("cutter") is not None else None,
            int(shuffle["cutPosition"]) if shuffle.get("cutPosition") is not None else None,
            str(shuffle["deckCommitment"]),
            tuple(int(value) for value in shuffle.get("deckOrder", ())),
        )
    state.purchase_card = _card(data["purchaseCard"]) if data.get("purchaseCard") else None
    state.initial_five = [
        cards_to_mask(_cards((data.get("initialFive") or {}).get(seat.name.lower(), ())))
        for seat in Seat
    ]
    state.deal_trace = [
        DealRecord(
            int(item.get("sequence", index)),
            str(item["stage"]),
            int(item.get("packet", 0)),
            Seat.parse(item["seat"]) if item.get("seat") is not None else None,
            _cards(item.get("cards")),
            bool(item.get("faceUp", False)),
        )
        for index, item in enumerate(data.get("dealTrace", ()))
    ]
    state.bid_history = [
        BidRecord(
            int(item.get("round", 1)),
            int(item.get("turn", index)),
            Seat.parse(item["seat"]),
            str(item["action"]),
            Contract.parse(item["contract"]) if item.get("contract") else None,
            Suit.parse(item["trumpSuit"]) if item.get("trumpSuit") else None,
            Seat.parse(item["bidder"]) if item.get("bidder") else None,
            Seat.parse(item["buyer"]) if item.get("buyer") else None,
            Seat.parse(item["purchaseRecipient"])
            if item.get("purchaseRecipient") else None,
        )
        for index, item in enumerate(data.get("biddingHistory", ()))
    ]
    restored_projects: list[Project] = []
    for item in data.get("projects", ()):
        cards = _cards(item.get("cards"))
        restored_projects.append(Project(
            Seat.parse(item["owner"]),
            str(item["kind"]),
            cards,
            int(item.get("value", 0)),
            int(item.get("highRank", 0)),
            int(item.get("categoryPriority", 0)),
            bool(item.get("declared", True)),
            int(item.get("declarationOrder", 999)),
            bool(item.get("scoring", False)),
            bool(item.get("revealed", bool(cards))),
        ))
    if restored_projects:
        state.projects = restored_projects
    state.project_winning_team = _team(data.get("projectWinningTeam"))
    round_score = data.get("roundScore")
    if final and round_score:
        parsed = RoundScore(
            [int(v) for v in round_score.get("rawCardPoints", (0, 0))],
            [int(v) for v in round_score.get("cardGamePoints", (0, 0))],
            [int(v) for v in round_score.get("projectPoints", (0, 0))],
            [int(v) for v in round_score.get("balootPoints", (0, 0))],
            [int(v) for v in round_score.get("totalBeforeAward", (0, 0))],
            _team(round_score["winner"]) or Team.NORTH_SOUTH,
            [int(v) for v in round_score.get("awardedPoints", (0, 0))],
            _team(round_score.get("countingTeam")) or Team.NORTH_SOUTH,
            int(round_score.get("multiplier", 1)),
            bool(round_score.get("kabout", False)),
            bool(round_score.get("gahwa", False)),
            _team(round_score.get("matchWinner")),
            [str(v) for v in round_score.get("explanation", ())],
        )
        # The legally replayed result must agree with the serialized score.
        if state.round_score is not None and state.round_score.awarded_points != parsed.awarded_points:
            raise ValueError("EpisodeSpec score differs from authoritative legal replay")
        state.round_score = parsed
        state.match_score = [int(v) for v in data.get("matchScoreAfter", state.match_score)]
        state.match_winner = _team(data.get("matchWinner"))
        state.phase = GamePhase.COMPLETE
    else:
        state.match_score = [int(v) for v in data.get("matchScoreBefore", state.match_score)]
    state.events = [
        Event(int(item.get("sequence", index)), str(item["type"]), {
            str(key): copy.deepcopy(value)
            for key, value in item.items()
            if key not in {"sequence", "type"}
        })
        for index, item in enumerate(data.get("events", ()))
    ]


def _analysis(data: Mapping[str, Any]) -> AnalysisResult:
    actions: list[ActionEvaluation] = []
    for item in data.get("actions", ()):
        metrics = item.get("metrics") or {}
        confidence = item.get("roundLeadConfidence95", item.get("confidence95", (0.0, 1.0)))
        match_confidence = metrics.get("trueMatchConfidence95", item.get("matchWinConfidence95", (0.0, 1.0)))
        distribution = item.get("distribution") or {}
        actions.append(ActionEvaluation(
            _card(item["card"]),
            float(metrics.get("expectedRoundMargin", item.get("expectedRoundMargin", item.get("expectedMatchMargin", 0.0)))),
            float(metrics.get("expectedOwnAwardedPoints", item.get("expectedTeamAwardedPoints", item.get("expectedTeamPoints", 0.0)))),
            float(metrics.get("expectedOpponentAwardedPoints", item.get("expectedOpponentAwardedPoints", item.get("expectedOpponentPoints", 0.0)))),
            float(metrics.get("roundLeadProbability", item.get("roundLeadProbability", item.get("winProbability", 0.0)))),
            float(item.get("drawProbability", 0.0)),
            float(confidence[0]),
            float(confidence[1]),
            float(item.get("bestShare", 0.0)),
            int(item.get("samples", 0)),
            float(item.get("effectiveSamples", item.get("samples", 0))),
            float(item.get("exactShare", 0.0)),
            float(metrics.get("contractSuccessProbability", item.get("contractSuccessProbability", 0.0))),
            float(metrics.get("trueMatchWinProbability", item.get("matchWinProbability", 0.0))),
            float(match_confidence[0]),
            float(match_confidence[1]),
            str(metrics.get("matchValueMethod", item.get("matchValueModel", "finite_horizon_markov_v1"))),
            tuple(float(v) for v in item.get("dealMargins", ())),
            float(distribution.get("downsideCvar10", item.get("downsideCvar", 0.0))),
            float(distribution.get("volatility", item.get("volatility", 0.0))),
            tuple(str(v) for v in item.get("principalVariation", ())),
            int(item.get("searchNodes", 0)),
            str(
                item.get(
                    "searchCompletionStatus",
                    (
                        "exact"
                        if float(item.get("exactShare", 0.0)) == 1.0
                        else "budget_exhausted"
                    ),
                )
            ),
        ))
    deductions = tuple(HardDeduction(
        str(item["kind"]),
        Seat.parse(item["seat"]),
        Suit.parse(item["suit"]) if item.get("suit") else None,
        float(item.get("certainty", 1.0)),
        str(item.get("source", "episode_import")),
    ) for item in data.get("deductions", ()))
    signals = tuple(SignalObservation(
        str(item["kind"]),
        Seat.parse(item["seat"]),
        str(item.get("relationship", "unknown")),
        _cards(item.get("cards")),
        float(item.get("confidence", 0.0)),
        str(item.get("forcedness", "unknown")),
        tuple(str(v) for v in item.get("interpretations", ())),
        str(item.get("caveat", "hypothesis")),
    ) for item in data.get("signals", ()))
    if not actions:
        raise ValueError("EpisodeSpec analysis must contain legal action evaluations")
    actions.sort(key=lambda action: (action.match_win_probability, action.expected_round_margin), reverse=True)
    return AnalysisResult(
        Seat.parse(data.get("observer", "south")),
        _card(data.get("recommendedCard", actions[0].card)),
        tuple(actions),
        deductions,
        signals,
        str(data.get("method", "episode_import")),
        int(data.get("hiddenDeals", 0)),
        float(data.get("effectiveSampleSize", data.get("hiddenDeals", 0))),
        int(data.get("difficulty", 50)),
        int(data.get("debateScore", 0)),
        bool(data.get("conventionsApplied", False)),
        tuple(str(v) for v in data.get("explanation", ())),
        str(data.get("objective", "match_win_probability")),
        str((data.get("analysisProfile") or {}).get("id", "")),
        str((data.get("analysisProfile") or {}).get("version", "")),
    )


def episode_spec_from_dict(data: Mapping[str, Any], rules: RuleBook) -> EpisodeSpec:
    """Validate and reconstruct an immutable EpisodeSpec from canonical JSON."""
    pause = int(data["pauseBeforePlayIndex"])
    final_context = _trusted_context(data, 32)
    final_context.pop("currentPlayer", None)
    final_state = replay_trusted_context(final_context, rules)
    before_context = _trusted_context(data, pause)
    before_context.pop("currentPlayer", None)
    state_before = replay_trusted_context(before_context, rules)
    _restore_lifecycle(final_state, data, final=True)
    _restore_lifecycle(state_before, data, final=False)
    final_state.validate_conservation()
    state_before.validate_conservation()
    provenance = data.get("provenance") or {}
    return EpisodeSpec(
        int(data.get("schemaVersion", 7)),
        str(data["episodeId"]),
        str(data.get("bundleHash", "")),
        Seat.parse(data.get("viewer", "south")),
        state_before,
        final_state,
        pause,
        _analysis(data["analysis"]),
        str(data.get("titleAr") or governed_episode_text("video.decision.challenge_title")),
        str(data.get("questionAr") or governed_episode_text("video.decision.challenge_question")),
        provenance.get("gameSerial"),
        provenance.get("gameRecordHash"),
        provenance.get("challengeId"),
        provenance.get("analysisId"),
        provenance.get("modelId"),
        provenance.get("auditId"),
        provenance.get("decisionId"),
        tuple((str(item.get("label", "decision")), str(item["decisionId"])) for item in data.get("decisionIndex", ())),
        copy.deepcopy(data.get("auditSummary")),
        str(data.get("playbackMode", "challenge_branch")),
    )


def load_episode_spec(path: str | Path, rules: RuleBook) -> EpisodeSpec:
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("EpisodeSpec document must be a JSON object")
    return episode_spec_from_dict(payload, rules)
