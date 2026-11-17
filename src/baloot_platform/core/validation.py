from __future__ import annotations

from .enums import GamePhase
from .models import RoundState


REQUIRED_LIFECYCLE_EVENTS = {
    "round_created",
    "deck_shuffled",
    "purchase_card_exposed",
    "initial_deal_complete",
    "contract_purchased",
    "final_deal_complete",
    "project_holdings_detected",
    "play_started",
    "round_scored",
}


def validate_completed_round(state: RoundState) -> list[str]:
    errors: list[str] = []
    try:
        state.validate_conservation()
    except AssertionError as exc:
        errors.append(str(exc))
    if state.phase != GamePhase.SCORED:
        errors.append("round is not scored")
    if len(state.completed_tricks) != 8:
        errors.append("round must contain eight tricks")
    if len(state.played_cards) != 32:
        errors.append("round must contain 32 plays")
    event_types = {e.type for e in state.events}
    missing = REQUIRED_LIFECYCLE_EVENTS - event_types
    if missing:
        errors.append(f"missing lifecycle events: {sorted(missing)}")
    if state.shuffle is None or len(state.shuffle.deck_order) != 32:
        errors.append("shuffle record is incomplete")
    if state.buyer is None or state.contract is None or state.first_leader is None:
        errors.append("buyer, contract and first leader are required")
    if state.round_score is None:
        errors.append("round score is missing")
    return errors
