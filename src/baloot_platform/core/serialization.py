from __future__ import annotations

from typing import Any

from .cards import Card, cards_to_mask
from .engine import BalootEngine
from .enums import Contract, GamePhase, Seat, Suit
from .models import RoundState, Trick
from .project_visibility import project_rank_visibility_policy
from .projects import detect_projects
from .rules import RuleBook


def _cards(values) -> tuple[Card, ...]:
    return tuple(Card.parse(v) for v in values or [])


def _trusted_round_shell(
    data: dict[str, Any], rules: RuleBook
) -> tuple[BalootEngine, RoundState]:
    engine = BalootEngine(rules)
    dealer = Seat.parse(data["dealer"])
    contract = Contract.parse(data["contract"])
    buyer = Seat.parse(data["buyer"])
    declarer = Seat.parse(data.get("declarer", data["buyer"]))
    trump = Suit.parse(data["trumpSuit"]) if data.get("trumpSuit") else None
    first_leader = Seat.parse(data["firstLeader"])
    initial_hands_raw = data.get("initialHands")
    if not isinstance(initial_hands_raw, dict):
        raise ValueError("trusted context requires initialHands")
    state = RoundState(
        rules.sha256,
        int(data.get("seed", 0)),
        int(data.get("roundNumber", 1)),
        dealer,
        match_score=list(data.get("matchScoreBefore", data.get("matchScore", [0, 0]))),
        project_rank_visibility=project_rank_visibility_policy(rules),
    )
    state.contract = contract
    state.buyer = buyer
    state.card_taker = Seat.parse(data.get("purchaseRecipient", data["buyer"]))
    state.declarer = declarer
    state.trump_suit = trump
    state.first_leader = first_leader
    state.current_player = first_leader
    state.initial_hands = [
        cards_to_mask(_cards(initial_hands_raw[seat.name.lower()])) for seat in Seat
    ]
    if (
        len(
            {
                card.id
                for seat in Seat
                for card in _cards(initial_hands_raw[seat.name.lower()])
            }
        )
        != 32
    ):
        raise ValueError("initial hands must contain 32 unique cards")
    state.hands = list(state.initial_hands)
    return engine, state


def replay_trusted_context(data: dict[str, Any], rules: RuleBook) -> RoundState:
    """Reconstruct and validate a trusted all-hands play context by legal replay."""
    engine, state = _trusted_round_shell(data, rules)
    contract = Contract.parse(data["contract"])
    dealer = Seat.parse(data["dealer"])
    state.phase = GamePhase.PLAYING
    state.escalation = str(data.get("escalation", "normal"))
    state.locked = bool(data.get("locked", False))
    state.escalation_history = list(data.get("escalationHistory", []))
    state.projects = detect_projects(
        {seat: state.hand(seat) for seat in Seat}, contract, rules, dealer
    )
    project_actions = list(data.get("projectActions") or [])
    if project_actions or engine.project_domain.declaration_mode == "optional":
        state._project_declaration_state = engine.project_domain.declaration_state(
            state.projects,
            dealer=dealer,
        )
    assert state.first_leader is not None
    state.current_trick = Trick(0, state.first_leader)
    plays: list[dict] = []
    for trick in data.get("completedTricks", []):
        plays.extend(trick.get("plays", []))
    if data.get("currentTrick"):
        plays.extend(data["currentTrick"].get("plays", []))
    project_action_index = 0
    for expected in plays:
        seat = Seat.parse(expected["seat"])
        if state.current_player != seat:
            raise ValueError(
                f"turn mismatch: expected {state.current_player}, got {seat}"
            )
        declaration = state._project_declaration_state
        if (
            declaration is not None
            and declaration.current_actor == seat
            and state.current_trick is not None
            and state.current_trick.index == 0
        ):
            if project_action_index >= len(project_actions):
                raise ValueError(
                    "project declaration action is required before playing"
                )
            engine.apply_project_declaration_action(
                state,
                project_actions[project_action_index],
            )
            project_action_index += 1
        engine.play_card(
            state, Card.parse(expected["card"]), ekka=bool(expected.get("ekka", False))
        )
    if project_action_index != len(project_actions):
        raise ValueError("unused project declaration actions remain after replay")
    if data.get("currentPlayer") is not None and state.current_player != Seat.parse(
        data["currentPlayer"]
    ):
        raise ValueError("currentPlayer does not match legal replay")
    state.validate_conservation()
    return state


def replay_trusted_exact_state_context(
    data: dict[str, Any], rules: RuleBook
) -> RoundState:
    """Replay a normalized exact state without advancing a non-play decision."""

    if data.get("phase") != GamePhase.FINAL_DEALT.value:
        return replay_trusted_context(data, rules)
    if data.get("currentTrick") is not None or data.get("completedTricks"):
        raise ValueError("final-dealt exact state cannot contain played cards")
    if data.get("projects"):
        raise ValueError("final-dealt exact state cannot contain prepared projects")

    engine, state = _trusted_round_shell(data, rules)
    state.phase = GamePhase.FINAL_DEALT
    for index, raw in enumerate(data.get("escalationHistory") or []):
        if not isinstance(raw, dict):
            raise TypeError(f"escalation history item {index} must be an object")
        engine.apply_escalation(
            state,
            seat=Seat.parse(raw["seat"]),
            action=str(raw["action"]),
            mode=str(raw["mode"]) if raw.get("mode") is not None else None,
        )
    if state.escalation != str(data.get("escalation", "normal")):
        raise ValueError("escalation level does not match authoritative replay")
    if state.locked is not bool(data.get("locked", False)):
        raise ValueError("escalation lock does not match authoritative replay")
    if state.escalation_history != list(data.get("escalationHistory") or []):
        raise ValueError("escalation history does not match authoritative replay")
    if data.get("currentPlayer") is not None and state.current_player != Seat.parse(
        data["currentPlayer"]
    ):
        raise ValueError("currentPlayer does not match exact-state replay")
    state.validate_conservation()
    return state


def continue_trusted_context(data: dict[str, Any], rules: RuleBook) -> RoundState:
    from .bots import BotProfile, HeuristicBot

    state = replay_trusted_context(data, rules)
    engine = BalootEngine(rules)
    bot = HeuristicBot(engine, BotProfile(state.seed + 7001))
    while state.current_player is not None:
        engine.play_card(state, bot.choose(state))
    return state
