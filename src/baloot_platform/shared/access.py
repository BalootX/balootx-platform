"""Transport-neutral access policy for decisions recorded in shared runs."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from baloot_platform.identity.ownership import owns_subject
from baloot_platform.core.enums import Seat


class SharedDecisionAccessError(RuntimeError):
    """Raised when a known shared decision cannot be authorized safely."""


class SharedDecisionAccessStore(Protocol):
    def get_run_for_decision(self, decision_id: str) -> Any: ...
    def get_run_for_game(self, game_id: str) -> Any: ...
    def get_board(self, identifier: str) -> Any: ...


def observer_relation(actor: str | Seat, observer: str | Seat) -> str:
    """Return the stable seat-relative relationship used by shared review."""

    try:
        actor_seat = actor if isinstance(actor, Seat) else Seat.parse(actor)
        observer_seat = observer if isinstance(observer, Seat) else Seat.parse(observer)
    except Exception:
        return "unknown"
    if actor_seat == observer_seat:
        return "self"
    if actor_seat == observer_seat.partner:
        return "partner"
    if actor_seat == observer_seat.next():
        return "left_opponent"
    return "right_opponent"


def private_decision_review_allowed(
    *,
    actor: str | Seat,
    observer: str | Seat,
    run_completed: bool,
) -> bool:
    """Allow alternatives and analysis only to the actor or after completion."""

    return bool(run_completed) or observer_relation(actor, observer) == "self"


@dataclass(frozen=True, slots=True)
class SharedDecisionAccessV1:
    run_id: str
    run_status: str
    owner_subject_id: str
    observer_seat: Seat
    actor_relation: str

    @property
    def completed(self) -> bool:
        return self.run_status == "completed"

    def can_view_private(self, subject_id: str | None) -> bool:
        if self.completed:
            return True
        return (
            self.run_status == "active"
            and bool(subject_id)
            and owns_subject(subject_id, self.owner_subject_id)
            and self.actor_relation == "self"
        )


def resolve_shared_decision_access(
    store: SharedDecisionAccessStore,
    decision: Mapping[str, Any],
) -> SharedDecisionAccessV1 | None:
    """Resolve shared-run authority, failing closed for a broken known link."""

    decision_id = str(decision.get("decisionId") or "").strip()
    if not decision_id:
        raise SharedDecisionAccessError("decision identity is unavailable")
    lookup = getattr(store, "get_run_for_decision", None)
    if not callable(lookup):
        raise SharedDecisionAccessError("shared decision access lookup is unavailable")
    try:
        loaded = lookup(decision_id)
        if loaded is None:
            game_id = str(decision.get("gameId") or "").strip()
            game_lookup = getattr(store, "get_run_for_game", None)
            if game_id:
                if not callable(game_lookup):
                    raise SharedDecisionAccessError(
                        "shared game access lookup is unavailable"
                    )
                loaded = game_lookup(game_id)
    except Exception as exc:
        raise SharedDecisionAccessError("shared decision access lookup failed") from exc
    if loaded is None:
        return None
    return _resolved_shared_access(store, loaded, str(decision.get("actor") or ""))


def resolve_shared_game_access(
    store: SharedDecisionAccessStore,
    game_id: str,
) -> SharedDecisionAccessV1 | None:
    """Resolve aggregate game authority without treating an archive as completion."""
    if not isinstance(game_id, str) or not game_id.strip():
        raise SharedDecisionAccessError("game identity is unavailable")
    try:
        loaded = store.get_run_for_game(game_id)
    except Exception as exc:
        raise SharedDecisionAccessError("shared game access lookup failed") from exc
    return None if loaded is None else _resolved_shared_access(store, loaded, "")


def _resolved_shared_access(
    store: SharedDecisionAccessStore, loaded: Any, actor: str,
) -> SharedDecisionAccessV1:
    run, _payload = loaded
    try:
        board_item = store.get_board(str(run.board_id))
    except Exception as exc:
        raise SharedDecisionAccessError("shared decision board lookup failed") from exc
    if board_item is None:
        raise SharedDecisionAccessError("shared decision board is unavailable")
    board = board_item[0]
    status = str(run.status)
    owner = str(run.subject_id or "").strip()
    if status not in {"active", "completed"} or not owner:
        raise SharedDecisionAccessError("shared decision run authority is invalid")
    observer = getattr(board, "observer_seat", None)
    if not isinstance(observer, Seat):
        raise SharedDecisionAccessError("shared decision observer identity is invalid")
    return SharedDecisionAccessV1(
        run_id=str(run.run_id),
        run_status=status,
        owner_subject_id=owner,
        observer_seat=observer,
        actor_relation=observer_relation(actor, observer),
    )


def project_shared_decision(
    decision: Mapping[str, Any],
    access: SharedDecisionAccessV1 | None,
    *,
    subject_id: str | None,
) -> dict[str, Any]:
    """Project an active non-actor decision to its already-public action."""

    if access is None or access.can_view_private(subject_id):
        return copy.deepcopy(dict(decision))
    return {
        key: copy.deepcopy(decision[key])
        for key in ("decisionId", "ordinal", "decisionKind", "actor", "selectedAction")
        if key in decision
    } | {
        "actorRelation": access.actor_relation,
        "reviewable": False,
        "privateAnalysisRedacted": True,
    }
