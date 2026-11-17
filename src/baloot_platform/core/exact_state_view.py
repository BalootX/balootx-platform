"""Observer-safe public projection for an immutable exact Baloot state."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from .cards import Card, FULL_DECK
from .models import LegalPlayActionV1
from .project_visibility import redact_project_ranks


_SEATS = ("north", "west", "south", "east")
_SUITS = ("spades", "hearts", "diamonds", "clubs")
_CONTRACTS = ("sun", "hokum", "ashkal")
_HASH = re.compile(r"^[0-9a-f]{64}$")
# Finite protocol identities, derived from the card authority, never caller data.
_CANONICAL_CARD_DOCUMENTS = tuple(MappingProxyType(card.to_dict()) for card in FULL_DECK)
_EXACT_STATE_KEYS = frozenset(
    {
        "schemaVersion",
        "exactStateId",
        "prefixHash",
        "rulesHash",
        "stateHash",
        "phase",
        "decisionKind",
        "nextActor",
        "legalActions",
        "state",
        "projectDeclarationsVerified",
        "projectAssertionsVerified",
        "eventChainHash",
        "sha256",
    }
)
_EXACT_STATE_CORE_KEYS = (
    "schemaVersion",
    "prefixHash",
    "rulesHash",
    "stateHash",
    "phase",
    "decisionKind",
    "nextActor",
    "legalActions",
    "eventChainHash",
)
_PUBLIC_EVENT_TYPES = frozenset(
    {
        "contract_purchased",
        "escalation_announced",
        "project_declared",
        "projects_compared",
        "projects_revealed",
        "baloot_included_in_project",
        "baloot_declared",
        "card_played",
        "trick_completed",
        "round_scored",
        "match_completed",
    }
)
_PUBLIC_EVENT_FIELDS = {
    "contract_purchased": (
        "winningBidder",
        "buyer",
        "contractOwner",
        "purchaseRecipient",
        "declarer",
        "contract",
        "trumpSuit",
        "purchaseCard",
    ),
    "escalation_announced": ("seat", "team", "action", "mode"),
    "project_declared": (
        "owner",
        "kind",
        "value",
        "highRank",
        "declarationOrder",
    ),
    "projects_compared": ("winningTeam", "declarations"),
    "projects_revealed": ("owner", "projects"),
    "baloot_included_in_project": ("owner", "team"),
    "baloot_declared": ("seat", "team"),
    "card_played": (
        "seat",
        "card",
        "trickIndex",
        "position",
        "ekka",
        "baloot",
    ),
    "trick_completed": ("trickIndex", "winner", "rawPoints"),
    "round_scored": ("score", "matchScore", "matchWinner"),
    "match_completed": ("matchScore", "matchWinner"),
}
_PROJECT_DECLARATION_KEYS = (
    "owner",
    "kind",
    "value",
    "highRank",
    "declarationOrder",
)
_PROJECT_PUBLIC_KEYS = (
    *_PROJECT_DECLARATION_KEYS,
    "team",
    "cards",
    "scoring",
    "revealed",
)
_BID_HISTORY_KEYS = (
    "round",
    "turn",
    "seat",
    "action",
    "contract",
    "trumpSuit",
    "bidder",
    "buyer",
    "purchaseRecipient",
)
_BID_HISTORY_KEYS_LEGACY = _BID_HISTORY_KEYS[:-1]
_LEGAL_ACTION_KEYS = {
    "bid": frozenset({"code", "action", "trumpSuit"}),
    "escalation": frozenset({"code", "action", "mode"}),
    "project": frozenset({"code", "action", "owner", "kind", "value", "highRank"}),
    "play": frozenset({"actionId", "code", "card", "ekka"}),
}
_ESCALATION_LEGAL_ACTION_KEYS_V2 = frozenset(
    {"code", "actor", "action", "mode"}
)
_PROJECT_PASS_LEGAL_ACTION_KEYS_V2 = frozenset({"code", "actor", "action"})
_PROJECT_DECLARE_LEGAL_ACTION_KEYS_V2 = frozenset(
    {"code", "actor", "action", "projectId"}
)
_PUBLIC_STATE_KEYS = (
    "roundNumber",
    "phase",
    "dealer",
    "buyer",
    "declarer",
    "contract",
    "trumpSuit",
    "purchaseCard",
    "escalation",
    "firstLeader",
    "currentPlayer",
    "matchScore",
    "balootTeams",
    "roundScore",
    "matchWinner",
)
_ROUND_SCORE_KEYS = frozenset(
    {
        "rawCardPoints",
        "cardGamePoints",
        "projectPoints",
        "balootPoints",
        "totalBeforeAward",
        "winner",
        "awardedPoints",
        "countingTeam",
        "multiplier",
        "kabout",
        "gahwa",
        "matchWinner",
        "explanation",
    }
)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _validate_escalation_actor_state_v1(
    state: Mapping[str, Any], next_actor: str
) -> None:
    declarer = state.get("declarer")
    level = state.get("escalation")
    history = state.get("escalationHistory", [])
    if (
        declarer not in _SEATS
        or level not in {"normal", "double", "triple", "four"}
        or not isinstance(history, list)
        or any(
            not isinstance(item, Mapping)
            or item.get("seat") not in _SEATS
            or not isinstance(item.get("action"), str)
            for item in history
        )
    ):
        raise ValueError("Research V4 exact-state escalation state is invalid")
    declarer_team = (
        "north_south" if declarer in {"north", "south"} else "west_east"
    )
    if level == "normal":
        allowed = {
            seat
            for seat in _SEATS
            if ("north_south" if seat in {"north", "south"} else "west_east")
            != declarer_team
        }
    elif level in {"double", "four"}:
        allowed = {declarer}
    else:
        double_actor = next(
            (
                str(item["seat"])
                for item in reversed(history)
                if item.get("action") == "double"
            ),
            None,
        )
        if double_actor is None:
            raise ValueError("Research V4 exact-state escalation history is invalid")
        allowed = {double_actor}
    if next_actor not in allowed:
        raise ValueError("Research V4 exact-state escalation actor mismatch")


def _validate_exact_state_document_v1(exact_state: Mapping[str, Any]) -> None:
    if set(exact_state) != set(_EXACT_STATE_KEYS):
        raise ValueError("Research V4 exact-state document is not closed")
    if (
        type(exact_state.get("schemaVersion")) is not int
        or exact_state.get("schemaVersion") != 1
        or any(
            not isinstance(exact_state.get(key), str)
            or _HASH.fullmatch(exact_state[key]) is None
            for key in (
                "prefixHash",
                "rulesHash",
                "stateHash",
                "eventChainHash",
                "sha256",
            )
        )
        or any(
            type(exact_state.get(key)) is not int or int(exact_state[key]) < 0
            for key in (
                "projectDeclarationsVerified",
                "projectAssertionsVerified",
            )
        )
    ):
        raise ValueError("Research V4 exact-state document is invalid")
    state = exact_state.get("state")
    if not isinstance(state, Mapping):
        raise ValueError("Research V4 exact-state document lacks its state")
    if (
        _canonical_sha256(state) != exact_state["stateHash"]
        or _canonical_sha256(state.get("events") or []) != exact_state["eventChainHash"]
        or state.get("rulesHash") != exact_state["rulesHash"]
        or state.get("phase") != exact_state.get("phase")
    ):
        raise ValueError("Research V4 exact-state state authority mismatch")
    core = {key: copy.deepcopy(exact_state[key]) for key in _EXACT_STATE_CORE_KEYS}
    digest = str(exact_state.get("sha256") or "")
    accepted_digests = {
        _canonical_sha256(core),
        _canonical_sha256(
            {
                **core,
                "projectDeclarationsVerified": exact_state[
                    "projectDeclarationsVerified"
                ],
                "projectAssertionsVerified": exact_state[
                    "projectAssertionsVerified"
                ],
            }
        ),
    }
    if (
        digest not in accepted_digests
        or exact_state.get("exactStateId") != f"STATE-{digest[:20].upper()}"
    ):
        raise ValueError("Research V4 exact-state semantic identity mismatch")

    next_actor = exact_state.get("nextActor")
    decision_kind = exact_state.get("decisionKind")
    if next_actor not in _SEATS or decision_kind not in {
        "bid",
        "escalation",
        "project",
        "play",
    }:
        raise ValueError("Research V4 exact-state decision authority is invalid")
    if decision_kind == "play" and state.get("currentPlayer") != next_actor:
        raise ValueError("Research V4 exact-state play actor mismatch")
    if decision_kind == "bid":
        bidding = state.get("bidding")
        if not isinstance(bidding, Mapping) or bidding.get("actor") != next_actor:
            raise ValueError("Research V4 exact-state bid actor mismatch")
    if decision_kind == "escalation":
        action_actors = {
            str(action.get("actor"))
            for action in exact_state.get("legalActions") or ()
            if isinstance(action, Mapping) and action.get("actor") in _SEATS
        }
        if action_actors:
            if next_actor not in action_actors:
                raise ValueError("Research V4 exact-state escalation actor mismatch")
        else:
            _validate_escalation_actor_state_v1(state, str(next_actor))
    if decision_kind == "project":
        legal_actions = exact_state.get("legalActions")
        if (
            state.get("currentPlayer") != next_actor
            or not isinstance(legal_actions, list)
            or any(
                not isinstance(action, Mapping)
                or action.get("actor", action.get("owner")) != next_actor
                for action in legal_actions
            )
        ):
            raise ValueError("Research V4 exact-state project actor mismatch")


def _safe_card(value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"id", "code", "suit", "rank", "symbol", "label"}
        or type(value.get("id")) is not int
        or any(
            not isinstance(value.get(key), str)
            for key in ("code", "suit", "rank", "symbol", "label")
        )
    ):
        raise ValueError("Research V4 exact state card is invalid")
    card_id = value["id"]
    if not 0 <= card_id < len(_CANONICAL_CARD_DOCUMENTS):
        raise ValueError(f"invalid card: {dict(value)!r}")
    canonical = _CANONICAL_CARD_DOCUMENTS[card_id]
    if dict(value) != canonical:
        raise ValueError("Research V4 exact state card is not canonical")
    return dict(canonical)


def _safe_score_pair(value: Any, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(item) is not int for item in value)
    ):
        raise ValueError(f"Research V4 exact state {label} is invalid")
    return list(value)


def _safe_round_score(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != set(_ROUND_SCORE_KEYS):
        raise ValueError("Research V4 exact state round score is not closed")
    pair_fields = (
        "rawCardPoints",
        "cardGamePoints",
        "projectPoints",
        "balootPoints",
        "totalBeforeAward",
        "awardedPoints",
    )
    score = {
        key: _safe_score_pair(value[key], f"round score {key}") for key in pair_fields
    }
    if value["winner"] not in {"north_south", "west_east"} or value[
        "countingTeam"
    ] not in {"north_south", "west_east"}:
        raise ValueError("Research V4 exact state round score team is invalid")
    if value["matchWinner"] not in {None, "north_south", "west_east"}:
        raise ValueError("Research V4 exact state round score winner is invalid")
    if type(value["multiplier"]) is not int or any(
        type(value[key]) is not bool for key in ("kabout", "gahwa")
    ):
        raise ValueError("Research V4 exact state round score scalar is invalid")
    explanation = value["explanation"]
    if not isinstance(explanation, list) or any(
        not isinstance(item, str) for item in explanation
    ):
        raise ValueError("Research V4 exact state round score explanation is invalid")
    return {
        **score,
        "winner": value["winner"],
        "countingTeam": value["countingTeam"],
        "multiplier": value["multiplier"],
        "kabout": value["kabout"],
        "gahwa": value["gahwa"],
        "matchWinner": value["matchWinner"],
        "explanation": list(explanation),
    }


def _safe_bid_history(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("Research V4 exact state bidding history is invalid")
    history: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ValueError("Research V4 exact state bid record is not closed")
        keys = set(raw)
        if keys != set(_BID_HISTORY_KEYS) and keys != set(_BID_HISTORY_KEYS_LEGACY):
            raise ValueError("Research V4 exact state bid record is not closed")
        if type(raw["round"]) is not int or type(raw["turn"]) is not int:
            raise ValueError("Research V4 exact state bid ordinal is invalid")
        if raw["seat"] not in _SEATS or not isinstance(raw["action"], str):
            raise ValueError("Research V4 exact state bid action is invalid")
        if raw["contract"] not in {None, *_CONTRACTS}:
            raise ValueError("Research V4 exact state bid contract is invalid")
        if raw["trumpSuit"] not in {None, *_SUITS}:
            raise ValueError("Research V4 exact state bid trump is invalid")
        recipient = raw.get("purchaseRecipient", raw["buyer"])
        if raw["bidder"] not in {None, *_SEATS} or raw["buyer"] not in {
            None,
            *_SEATS,
        } or recipient not in {None, *_SEATS}:
            raise ValueError("Research V4 exact state bid ownership is invalid")
        record = {
            key: copy.deepcopy(raw[key]) for key in _BID_HISTORY_KEYS_LEGACY
        }
        if recipient != raw["buyer"]:
            record["purchaseRecipient"] = copy.deepcopy(recipient)
        history.append(record)
    return history


def _safe_bidding(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"round", "actor", "history"}:
        raise ValueError("Research V4 exact state bidding projection is not closed")
    if type(value["round"]) is not int or value["round"] not in {1, 2}:
        raise ValueError("Research V4 exact state bidding round is invalid")
    if value["actor"] not in _SEATS:
        raise ValueError("Research V4 exact state bidding actor is invalid")
    return {
        "round": value["round"],
        "actor": value["actor"],
        "history": _safe_bid_history(value["history"]),
    }


def _safe_legal_actions(decision_kind: str, value: Any) -> list[dict[str, Any]]:
    expected_keys = _LEGAL_ACTION_KEYS.get(decision_kind)
    if expected_keys is None:
        raise ValueError("Research V4 exact-state decision kind is invalid")
    if not isinstance(value, list) or not value:
        raise ValueError("Research V4 playable exact state requires legal actions")
    actions: list[dict[str, Any]] = []
    for raw in value:
        if (
            decision_kind == "play"
            and isinstance(raw, Mapping)
            and set(raw) == {"id", "code", "suit", "rank", "symbol", "label"}
        ):
            actions.append(
                LegalPlayActionV1(Card.parse(_safe_card(raw)), False).to_dict()
            )
            continue
        if not isinstance(raw, Mapping) or (
            set(raw) != expected_keys
            and not (
                decision_kind == "escalation"
                and set(raw) == _ESCALATION_LEGAL_ACTION_KEYS_V2
            )
            and not (
                decision_kind == "project"
                and set(raw)
                in {
                    _PROJECT_PASS_LEGAL_ACTION_KEYS_V2,
                    _PROJECT_DECLARE_LEGAL_ACTION_KEYS_V2,
                }
            )
        ):
            raise ValueError("Research V4 exact-state legal action is not closed")
        if decision_kind == "play":
            if (
                type(raw["actionId"]) is not int
                or not isinstance(raw["code"], str)
                or type(raw["ekka"]) is not bool
            ):
                raise ValueError("Research V4 exact-state play action is invalid")
            card = Card.parse(_safe_card(raw["card"]))
            canonical = LegalPlayActionV1(card, raw["ekka"]).to_dict()
            if dict(raw) != canonical:
                raise ValueError("Research V4 exact-state play action is not canonical")
        elif decision_kind == "bid":
            action = raw["action"]
            trump = raw["trumpSuit"]
            if (
                not isinstance(raw["code"], str)
                or not isinstance(action, str)
                or trump not in {None, *_SUITS}
            ):
                raise ValueError("Research V4 exact-state bid action is invalid")
            code = action if trump is None else f"{action}:{trump}"
            canonical = {"code": code, "action": action, "trumpSuit": trump}
            if dict(raw) != canonical:
                raise ValueError("Research V4 exact-state bid action is not canonical")
        elif decision_kind == "escalation":
            actor = raw.get("actor")
            action = raw["action"]
            mode = raw["mode"]
            if (
                not isinstance(raw["code"], str)
                or actor not in {None, *_SEATS}
                or not isinstance(action, str)
                or mode not in {None, "open", "locked"}
            ):
                raise ValueError("Research V4 exact-state escalation action is invalid")
            code = action if mode is None else f"{action}:{mode}"
            canonical = {
                "code": code,
                **({"actor": actor} if actor is not None else {}),
                "action": action,
                "mode": mode,
            }
            if dict(raw) != canonical:
                raise ValueError(
                    "Research V4 exact-state escalation action is not canonical"
                )
        else:
            if "actor" in raw:
                actor = raw["actor"]
                action = raw["action"]
                project_id = raw.get("projectId")
                if (
                    actor not in _SEATS
                    or action not in {"pass", "declare"}
                    or (
                        action == "declare"
                        and (
                            not isinstance(project_id, str)
                            or re.fullmatch(r"PROJECT-[0-9A-F]{16}", project_id)
                            is None
                        )
                    )
                    or (action == "pass" and project_id is not None)
                ):
                    raise ValueError(
                        "Research V4 exact-state project action is invalid"
                    )
                canonical = {
                    "code": "pass" if action == "pass" else f"declare:{project_id}",
                    "actor": actor,
                    "action": action,
                    **({"projectId": project_id} if project_id is not None else {}),
                }
            else:
                owner = raw["owner"]
                kind = raw["kind"]
                value = raw["value"]
                high_rank = raw["highRank"]
                if (
                    raw["action"] != "declare"
                    or not isinstance(raw["code"], str)
                    or owner not in _SEATS
                    or not isinstance(kind, str)
                    or type(value) is not int
                    or type(high_rank) is not int
                ):
                    raise ValueError(
                        "Research V4 exact-state project action is invalid"
                    )
                canonical = {
                    "code": f"declare:{kind}:{value}:{high_rank}",
                    "action": "declare",
                    "owner": owner,
                    "kind": kind,
                    "value": value,
                    "highRank": high_rank,
                }
            if dict(raw) != canonical:
                raise ValueError(
                    "Research V4 exact-state project action is not canonical"
                )
        actions.append(copy.deepcopy(canonical))
    return actions


def _project_declared(value: Mapping[str, Any]) -> bool:
    declared = value.get("declared")
    if declared is not None:
        if type(declared) is not bool:
            raise ValueError("Research V4 exact state project declaration is invalid")
        return declared
    order = value.get("declarationOrder", 999)
    if type(order) is not int:
        raise ValueError("Research V4 exact state project declaration order is invalid")
    return order < 999


def _safe_projects(state: Mapping[str, Any], observer: str) -> list[dict[str, Any]]:
    """Mirror ``BalootEngine.observer_view`` project visibility."""

    raw_projects = state.get("projects", [])
    if not isinstance(raw_projects, list):
        raise ValueError("Research V4 exact state projects are invalid")
    projects: list[dict[str, Any]] = []
    for raw in raw_projects:
        if not isinstance(raw, Mapping) or set(raw) != set(_PROJECT_PUBLIC_KEYS):
            raise ValueError("Research V4 exact state project is not closed")
        owner = raw.get("owner")
        if owner not in _SEATS:
            raise ValueError("Research V4 exact state project owner is invalid")
        expected_team = "north_south" if owner in {"north", "south"} else "west_east"
        if (
            raw.get("team") != expected_team
            or not isinstance(raw.get("kind"), str)
            or not raw["kind"]
            or any(
                type(raw.get(key)) is not int
                for key in ("value", "highRank", "declarationOrder")
            )
            or type(raw.get("scoring")) is not bool
        ):
            raise ValueError("Research V4 exact state project is invalid")
        revealed = raw.get("revealed", False)
        if type(revealed) is not bool:
            raise ValueError("Research V4 exact state project reveal is invalid")
        if revealed and not isinstance(raw.get("cards"), list):
            raise ValueError("Research V4 revealed project cards are invalid")
        if not revealed and raw.get("cards") is not None:
            raise ValueError("Research V4 hidden project cards must remain hidden")
        if owner != observer and not _project_declared(raw) and not revealed:
            continue
        project = {
            key: copy.deepcopy(raw[key]) for key in _PROJECT_PUBLIC_KEYS if key in raw
        }
        project["cards"] = (
            [_safe_card(card) for card in raw["cards"]] if revealed else None
        )
        projects.append(project)
    redact_project_ranks(state, observer, projects)
    return projects


def _safe_project_winning_team(
    state: Mapping[str, Any],
    visible_projects: list[dict[str, Any]],
) -> str | None:
    winning_team = state.get("projectWinningTeam")
    if winning_team is not None and winning_team not in {
        "north_south",
        "west_east",
    }:
        raise ValueError("Research V4 exact state project winning team is invalid")
    if not any(
        _project_declared(project) or project.get("revealed") is True
        for project in visible_projects
    ):
        return None
    return winning_team


def _public_project_declaration(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value[key])
        for key in _PROJECT_DECLARATION_KEYS
        if key in value
    }


def _safe_hands(state: Mapping[str, Any], observer: str) -> dict[str, Any]:
    raw = state.get("hands")
    if not isinstance(raw, Mapping) or set(raw) != set(_SEATS):
        raise ValueError("Research V4 exact state has an invalid hand projection")
    hands: dict[str, Any] = {}
    for seat in _SEATS:
        cards = raw[seat]
        if not isinstance(cards, list):
            raise ValueError("Research V4 exact state hand must be a card list")
        hands[seat] = (
            [_safe_card(card) for card in cards]
            if seat == observer
            else {"count": len(cards)}
        )
    return hands


def _public_trick(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or set(raw) != {
        "index",
        "leader",
        "plays",
        "winner",
        "rawPoints",
    }:
        raise ValueError("Research V4 exact state trick projection is not closed")
    if (
        type(raw["index"]) is not int
        or raw["leader"] not in _SEATS
        or raw["winner"] not in {None, *_SEATS}
        or type(raw["rawPoints"]) is not int
    ):
        raise ValueError("Research V4 exact state trick projection is invalid")
    plays = raw.get("plays")
    if not isinstance(plays, list):
        raise ValueError("Research V4 exact state trick plays are invalid")
    public_plays: list[dict[str, Any]] = []
    for play in plays:
        if not isinstance(play, Mapping) or set(play) != {
            "seat",
            "card",
            "trickIndex",
            "position",
            "legalCards",
            "ekka",
            "baloot",
        }:
            raise ValueError("Research V4 exact state trick play is not closed")
        if (
            play["seat"] not in _SEATS
            or type(play["trickIndex"]) is not int
            or type(play["position"]) is not int
            or type(play["ekka"]) is not bool
            or type(play["baloot"]) is not bool
        ):
            raise ValueError("Research V4 exact state trick play is invalid")
        public_plays.append(
            {
                "seat": play["seat"],
                "card": _safe_card(play["card"]),
                "trickIndex": play["trickIndex"],
                "position": play["position"],
                "ekka": play["ekka"],
                "baloot": play["baloot"],
            }
        )
    return {
        **{
            key: copy.deepcopy(raw[key])
            for key in ("index", "leader", "winner", "rawPoints")
            if key in raw
        },
        "plays": public_plays,
    }


def _validate_card_conservation_v1(
    state: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, list[dict[str, Any]]]:
    hands = state.get("hands")
    if not isinstance(hands, Mapping) or set(hands) != set(_SEATS):
        raise ValueError("Research V4 exact state hands are invalid")
    remaining_by_seat: dict[str, set[int]] = {}
    remaining_ids: set[int] = set()
    for seat in _SEATS:
        raw_cards = hands[seat]
        if not isinstance(raw_cards, list):
            raise ValueError("Research V4 exact state hand must be a card list")
        identifiers = {_safe_card(card)["id"] for card in raw_cards}
        if len(identifiers) != len(raw_cards) or remaining_ids & identifiers:
            raise ValueError("Research V4 exact state hands duplicate a card")
        remaining_by_seat[seat] = identifiers
        remaining_ids |= identifiers

    completed = state.get("completedTricks", [])
    if not isinstance(completed, list):
        raise ValueError("Research V4 exact state completed tricks are invalid")
    public_tricks = []
    for raw in completed:
        trick = _public_trick(raw)
        if trick is None:
            raise ValueError("Research V4 exact state completed trick is invalid")
        public_tricks.append(trick)
    current_trick = _public_trick(state.get("currentTrick"))
    public_plays = [play for trick in public_tricks for play in trick["plays"]]
    if current_trick is not None:
        public_plays.extend(current_trick["plays"])
    played_by_seat = {seat: set() for seat in _SEATS}
    played_ids: set[int] = set()
    for play in public_plays:
        identifier = int(play["card"]["id"])
        if identifier in played_ids or identifier in remaining_ids:
            raise ValueError("Research V4 exact state reuses a played card")
        played_ids.add(identifier)
        played_by_seat[str(play["seat"])].add(identifier)

    initial_hands = state.get("initialHands")
    if not isinstance(initial_hands, Mapping) or set(initial_hands) != set(_SEATS):
        raise ValueError("Research V4 exact state initial hands are invalid")
    if any(initial_hands[seat] for seat in _SEATS):
        initial_ids: set[int] = set()
        for seat in _SEATS:
            raw_cards = initial_hands[seat]
            if not isinstance(raw_cards, list):
                raise ValueError("Research V4 exact state initial hand is invalid")
            identifiers = {_safe_card(card)["id"] for card in raw_cards}
            if (
                len(identifiers) != len(raw_cards)
                or initial_ids & identifiers
                or identifiers != remaining_by_seat[seat] | played_by_seat[seat]
            ):
                raise ValueError("Research V4 exact state initial hand is inconsistent")
            initial_ids |= identifiers
        if len(initial_ids) != 32:
            raise ValueError("Research V4 exact state initial deck is incomplete")
    return public_tricks, current_trick, public_plays


def _public_events(
    state: Mapping[str, Any],
    project_winning_team: str | None,
    visible_projects: list[dict[str, Any]],
    *,
    validated_plays: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    actual_plays = validated_plays
    if actual_plays is None:
        actual_plays = [
            play
            for trick in state.get("completedTricks") or ()
            for play in (_public_trick(trick) or {}).get("plays") or ()
        ]
        if state.get("currentTrick") is not None:
            actual_plays.extend(
                (_public_trick(state["currentTrick"]) or {}).get("plays") or ()
            )
    public_declarations = [
        _public_project_declaration(project)
        for project in visible_projects
        if _project_declared(project)
    ]
    revealed_projects = [
        project for project in visible_projects if project.get("revealed") is True
    ]
    for raw in state.get("events") or ():
        if not isinstance(raw, Mapping):
            continue
        event_type = raw.get("type")
        if event_type not in _PUBLIC_EVENT_TYPES:
            continue
        sequence = raw.get("sequence")
        if type(sequence) is not int:
            raise ValueError("Research V4 exact state event sequence is invalid")
        row = {
            "sequence": sequence,
            "type": event_type,
            **{
                key: raw[key]
                for key in _PUBLIC_EVENT_FIELDS[event_type]
                if key in raw
            },
        }
        if event_type == "contract_purchased":
            if any(
                row.get(key) not in {None, *_SEATS}
                for key in (
                    "winningBidder",
                    "buyer",
                    "contractOwner",
                    "purchaseRecipient",
                    "declarer",
                )
            ) or row.get("contract") not in set(_CONTRACTS):
                raise ValueError("Research V4 purchase event is invalid")
            if row.get("trumpSuit") not in {None, *_SUITS}:
                raise ValueError("Research V4 purchase event trump is invalid")
            row["purchaseCard"] = _safe_card(row.get("purchaseCard"))
        elif event_type == "escalation_announced":
            if (
                row.get("seat") not in _SEATS
                or row.get("team") not in {"north_south", "west_east"}
                or not isinstance(row.get("action"), str)
                or row.get("mode") is not None
                and not isinstance(row.get("mode"), str)
            ):
                raise ValueError("Research V4 escalation event is invalid")
        elif event_type == "project_declared":
            if state.get("projectRankVisibility") is not None:
                row["highRank"] = next((item["highRank"] for item in public_declarations if all(item.get(key) == row.get(key) for key in ("owner", "kind", "value", "declarationOrder"))), None)
            if _public_project_declaration(row) not in public_declarations:
                continue
        elif event_type == "projects_compared":
            row["winningTeam"] = project_winning_team
            declarations = raw.get("declarations") or []
            if not isinstance(declarations, list):
                raise ValueError("Research V4 project comparison event is invalid")
            safe_declarations = []
            for item in declarations:
                if not isinstance(item, Mapping):
                    continue
                declaration = _public_project_declaration(item)
                if state.get("projectRankVisibility") is not None:
                    declaration["highRank"] = next((candidate["highRank"] for candidate in public_declarations if all(candidate.get(key) == declaration.get(key) for key in ("owner", "kind", "value", "declarationOrder"))), None)
                if declaration in public_declarations:
                    safe_declarations.append(declaration)
            row["declarations"] = safe_declarations
        elif event_type == "projects_revealed":
            if row.get("owner") not in _SEATS:
                raise ValueError("Research V4 project reveal owner is invalid")
            projects = raw.get("projects") or []
            if not isinstance(projects, list):
                raise ValueError("Research V4 project reveal event is invalid")
            row["projects"] = [
                copy.deepcopy(dict(project))
                for project in projects
                if isinstance(project, Mapping) and dict(project) in revealed_projects
            ]
            if not row["projects"]:
                continue
        elif event_type in {"baloot_included_in_project", "baloot_declared"}:
            seat_key = "owner" if event_type == "baloot_included_in_project" else "seat"
            if row.get(seat_key) not in _SEATS or row.get("team") not in {
                "north_south",
                "west_east",
            }:
                raise ValueError("Research V4 Baloot event is invalid")
        elif event_type == "card_played":
            if (
                row.get("seat") not in _SEATS
                or type(row.get("trickIndex")) is not int
                or type(row.get("position")) is not int
                or type(row.get("ekka")) is not bool
                or type(row.get("baloot")) is not bool
            ):
                raise ValueError("Research V4 card-play event is invalid")
            row["card"] = _safe_card(row.get("card"))
            if {
                key: row[key]
                for key in (
                    "seat",
                    "card",
                    "trickIndex",
                    "position",
                    "ekka",
                    "baloot",
                )
            } not in actual_plays:
                raise ValueError(
                    "Research V4 card-play event lacks a played-card owner"
                )
        elif event_type == "trick_completed":
            if (
                type(row.get("trickIndex")) is not int
                or row.get("winner") not in _SEATS
                or type(row.get("rawPoints")) is not int
            ):
                raise ValueError("Research V4 trick-completion event is invalid")
        elif event_type == "round_scored":
            row["score"] = _safe_round_score(row.get("score"))
            row["matchScore"] = _safe_score_pair(
                row.get("matchScore"),
                "round event match score",
            )
            if row.get("matchWinner") not in {
                None,
                "north_south",
                "west_east",
            }:
                raise ValueError("Research V4 round event winner is invalid")
        elif event_type == "match_completed":
            row["matchScore"] = _safe_score_pair(
                row.get("matchScore"),
                "match event score",
            )
            if row.get("matchWinner") not in {"north_south", "west_east"}:
                raise ValueError("Research V4 match event winner is invalid")
        rows.append(row)
    return rows


def _safe_escalation_history(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("Research V4 observer escalation history is invalid")
    history: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, Mapping) or set(raw) != {
            "seat",
            "team",
            "action",
            "mode",
        }:
            raise ValueError("Research V4 observer escalation record is not closed")
        if (
            raw["seat"] not in _SEATS
            or raw["team"] not in {"north_south", "west_east"}
            or not isinstance(raw["action"], str)
            or raw["mode"] not in {None, "open", "locked"}
        ):
            raise ValueError("Research V4 observer escalation record is invalid")
        history.append(
            {
                "seat": raw["seat"],
                "team": raw["team"],
                "action": raw["action"],
                "mode": raw["mode"],
            }
        )
    return history


def _canonical_observer_hands(
    state: Mapping[str, Any], acting_seat: str
) -> dict[str, Any]:
    hands = _safe_hands(state, acting_seat)
    hands[acting_seat] = sorted(hands[acting_seat], key=lambda card: card["id"])
    return hands


def _canonical_observer_projects(
    state: Mapping[str, Any], acting_seat: str
) -> list[dict[str, Any]]:
    projects = _safe_projects(state, acting_seat)
    return sorted(
        projects,
        key=lambda project: (
            int(project["declarationOrder"]),
            str(project["owner"]),
            str(project["kind"]),
            int(project["value"]),
            int(project["highRank"]) if project["highRank"] is not None else -1,
            tuple(card["id"] for card in project.get("cards") or ()),
        ),
    )


def _canonical_observer_bid_history(value: Any) -> list[dict[str, Any]]:
    history = _safe_bid_history(value)
    keys = [(record["round"], record["turn"]) for record in history]
    if len(keys) != len(set(keys)):
        raise ValueError("Research V4 observer bidding ordinals are duplicated")
    return sorted(history, key=lambda record: (record["round"], record["turn"]))


def _canonical_observer_completed_tricks(
    value: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    # Conservation has already validated and detached every trick.
    indices = [trick["index"] for trick in value]
    if len(indices) != len(set(indices)):
        raise ValueError("Research V4 observer trick indices are duplicated")
    return sorted(value, key=lambda trick: trick["index"])


def _canonical_observer_events(
    state: Mapping[str, Any],
    project_winning_team: str | None,
    visible_projects: list[dict[str, Any]],
    validated_plays: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    events = _public_events(
        state, project_winning_team, visible_projects,
        validated_plays=validated_plays,
    )
    sequences = [event["sequence"] for event in events]
    if len(sequences) != len(set(sequences)):
        raise ValueError("Research V4 observer event sequences are duplicated")
    return sorted(events, key=lambda event: event["sequence"])


def build_canonical_observer_information_v1(
    *,
    complete_state: Mapping[str, Any],
    acting_seat: str,
) -> dict[str, Any]:
    """Build the canonical information available to one acting Baloot seat.

    The input is a complete serialized round state. The returned projection owns
    no runtime state object and deliberately replaces every opponent hand with a
    count, so its hash is shared by worlds that the actor cannot distinguish.
    """

    if not isinstance(complete_state, Mapping):
        raise ValueError("Research V4 observer state must be an object")
    if acting_seat not in _SEATS:
        raise ValueError("Research V4 observer acting seat is invalid")
    completed_tricks, current_trick, public_plays = _validate_card_conservation_v1(complete_state)

    rules_hash = complete_state.get("rulesHash")
    if not isinstance(rules_hash, str) or _HASH.fullmatch(rules_hash) is None:
        raise ValueError("Research V4 observer rules hash is invalid")
    round_number = complete_state.get("roundNumber")
    match_score = _safe_score_pair(complete_state.get("matchScore"), "match score")
    if (
        type(round_number) is not int
        or round_number < 1
        or not isinstance(complete_state.get("phase"), str)
        or complete_state.get("dealer") not in _SEATS
        or any(score < 0 for score in match_score)
    ):
        raise ValueError("Research V4 observer round authority is invalid")

    public_state: dict[str, Any] = {
        "rulesHash": rules_hash,
        "roundNumber": round_number,
        "phase": complete_state["phase"],
        "dealer": complete_state["dealer"],
        "matchScore": match_score,
    }
    purchase_card = complete_state.get("purchaseCard")
    public_state["purchaseCard"] = (
        _safe_card(purchase_card) if purchase_card is not None else None
    )

    if "bidding" in complete_state:
        bidding = _safe_bidding(complete_state["bidding"])
        bidding["history"] = _canonical_observer_bid_history(bidding["history"])
        public_state["bidding"] = bidding
    public_state["biddingHistory"] = _canonical_observer_bid_history(
        complete_state.get("biddingHistory", [])
    )

    for key in (
        "winningBidder",
        "buyer",
        "contractOwner",
        "purchaseRecipient",
        "declarer",
    ):
        value = complete_state.get(key)
        if value not in {None, *_SEATS}:
            raise ValueError(f"Research V4 observer {key} is invalid")
        public_state[key] = value
    contract = complete_state.get("contract")
    trump_suit = complete_state.get("trumpSuit")
    if contract not in {None, *_CONTRACTS} or trump_suit not in {None, *_SUITS}:
        raise ValueError("Research V4 observer contract authority is invalid")
    public_state["contract"] = contract
    public_state["trumpSuit"] = trump_suit

    escalation = complete_state.get("escalation")
    locked = complete_state.get("locked")
    if not isinstance(escalation, str) or type(locked) is not bool:
        raise ValueError("Research V4 observer escalation authority is invalid")
    public_state["escalation"] = escalation
    public_state["escalationHistory"] = _safe_escalation_history(
        complete_state.get("escalationHistory", [])
    )
    public_state["locked"] = locked

    for key in ("firstLeader", "currentPlayer", "annulledBy"):
        value = complete_state.get(key)
        if value not in {None, *_SEATS}:
            raise ValueError(f"Research V4 observer {key} is invalid")
        public_state[key] = value
    annul_reason = complete_state.get("annulReason")
    if annul_reason is not None and not isinstance(annul_reason, str):
        raise ValueError("Research V4 observer annulment reason is invalid")
    public_state["annulReason"] = annul_reason

    public_state["hands"] = _canonical_observer_hands(complete_state, acting_seat)
    public_projects = _canonical_observer_projects(complete_state, acting_seat)
    public_state["projects"] = public_projects
    project_winning_team = _safe_project_winning_team(
        complete_state,
        public_projects,
    )
    public_state["projectWinningTeam"] = project_winning_team

    baloot_teams = complete_state.get("balootTeams")
    if (
        not isinstance(baloot_teams, list)
        or any(team not in {"north_south", "west_east"} for team in baloot_teams)
        or len(baloot_teams) != len(set(baloot_teams))
    ):
        raise ValueError("Research V4 observer Baloot teams are invalid")
    public_state["balootTeams"] = sorted(baloot_teams)

    trump_broken = complete_state.get("trumpBroken")
    if type(trump_broken) is not bool:
        raise ValueError("Research V4 observer trump state is invalid")
    public_state["trumpBroken"] = trump_broken
    public_state["currentTrick"] = current_trick
    public_state["completedTricks"] = _canonical_observer_completed_tricks(
        completed_tricks
    )

    public_state["roundScore"] = _safe_round_score(complete_state.get("roundScore"))
    match_winner = complete_state.get("matchWinner")
    if match_winner not in {None, "north_south", "west_east"}:
        raise ValueError("Research V4 observer match winner is invalid")
    public_state["matchWinner"] = match_winner
    public_state["events"] = _canonical_observer_events(
        complete_state,
        project_winning_team,
        public_projects,
        public_plays,
    )

    core = {
        "schemaVersion": 1,
        "actingSeat": acting_seat,
        "publicState": public_state,
    }
    return {
        **core,
        "observerInformationHash": _canonical_sha256(core),
    }


def canonical_observer_information_hash_v1(
    *,
    complete_state: Mapping[str, Any],
    acting_seat: str,
) -> str:
    """Return the canonical actor-information hash for a complete round state."""

    return str(
        build_canonical_observer_information_v1(
            complete_state=complete_state,
            acting_seat=acting_seat,
        )["observerInformationHash"]
    )


def build_observer_safe_exact_state_v1(
    *,
    exact_state: Mapping[str, Any],
    scenario_projection: Mapping[str, Any],
    observer_seat: str,
    decision_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project one validated exact state for its acting observer only."""

    _validate_exact_state_document_v1(exact_state)
    if observer_seat not in _SEATS:
        raise ValueError("Research V4 exact-state observer is invalid")
    exact_state_id = str(exact_state.get("exactStateId") or "")
    projected_exact_state_id = scenario_projection.get("exactStateId")
    if projected_exact_state_id is not None and exact_state_id != str(
        projected_exact_state_id
    ):
        raise ValueError("Research V4 exact-state projection ownership mismatch")
    decision_kind = str(exact_state.get("decisionKind") or "")
    if decision_kind != str(scenario_projection.get("decisionKind") or ""):
        raise ValueError("Research V4 exact-state decision projection mismatch")
    next_actor = str(exact_state.get("nextActor") or "")
    if next_actor not in _SEATS or observer_seat != next_actor:
        raise ValueError(
            "Research V4 playable exact state requires the acting observer"
        )
    state = exact_state.get("state")
    if not isinstance(state, Mapping):
        raise ValueError("Research V4 exact-state document lacks its state")
    completed_tricks, current_trick, public_plays = _validate_card_conservation_v1(state)
    legal_actions = _safe_legal_actions(
        decision_kind,
        exact_state.get("legalActions"),
    )
    action_codes = [str(action["code"]) for action in legal_actions]
    if len(action_codes) != len(set(action_codes)):
        raise ValueError("Research V4 exact-state legal actions are duplicated")
    if decision_kind == "play":
        actor_hand = state.get("hands")
        if not isinstance(actor_hand, Mapping) or not isinstance(
            actor_hand.get(next_actor), list
        ):
            raise ValueError("Research V4 exact-state acting hand is invalid")
        hand_cards = {_safe_card(card)["code"] for card in actor_hand[next_actor]}
        if any(action["card"]["code"] not in hand_cards for action in legal_actions):
            raise ValueError(
                "Research V4 exact-state play action is outside actor hand"
            )
    public_state = {
        key: copy.deepcopy(state[key]) for key in _PUBLIC_STATE_KEYS if key in state
    }
    if (
        type(public_state.get("roundNumber")) is not int
        or not isinstance(public_state.get("phase"), str)
        or public_state.get("dealer") not in _SEATS
        or public_state.get("buyer") not in {None, *_SEATS}
        or public_state.get("declarer") not in {None, *_SEATS}
        or public_state.get("contract") not in {None, *_CONTRACTS}
        or public_state.get("trumpSuit") not in {None, *_SUITS}
        or not isinstance(public_state.get("escalation"), str)
        or public_state.get("firstLeader") not in {None, *_SEATS}
        or public_state.get("currentPlayer") not in {None, *_SEATS}
        or public_state.get("matchWinner") not in {None, "north_south", "west_east"}
    ):
        raise ValueError("Research V4 exact state public scalar is invalid")
    purchase_card = public_state.get("purchaseCard")
    public_state["purchaseCard"] = (
        _safe_card(purchase_card) if purchase_card is not None else None
    )
    public_state["matchScore"] = _safe_score_pair(
        public_state.get("matchScore"),
        "match score",
    )
    baloot_teams = public_state.get("balootTeams")
    if not isinstance(baloot_teams, list) or any(
        team not in {"north_south", "west_east"} for team in baloot_teams
    ):
        raise ValueError("Research V4 exact state Baloot teams are invalid")
    public_state["balootTeams"] = list(baloot_teams)
    public_state["roundScore"] = _safe_round_score(public_state.get("roundScore"))
    if "bidding" in state:
        public_state["bidding"] = _safe_bidding(state["bidding"])
    if "biddingHistory" in state:
        public_state["biddingHistory"] = _safe_bid_history(state["biddingHistory"])
    public_state["hands"] = _safe_hands(state, observer_seat)
    public_projects = _safe_projects(state, observer_seat)
    public_state["projects"] = public_projects
    project_winning_team = _safe_project_winning_team(
        state,
        public_projects,
    )
    public_state["projectWinningTeam"] = project_winning_team
    if "currentTrick" in state:
        public_state["currentTrick"] = current_trick
    if "completedTricks" in state:
        public_state["completedTricks"] = completed_tricks
    public_state["events"] = _public_events(
        state,
        project_winning_team,
        public_projects,
        validated_plays=public_plays,
    )
    public_projection_hash = _canonical_sha256(
        {
            "schemaVersion": 1,
            "decisionKind": decision_kind,
            "observerSeat": observer_seat,
            "nextActor": next_actor,
            "legalActions": legal_actions,
            "publicState": public_state,
        }
    )
    expected_projection_hash = scenario_projection.get("publicProjectionHash")
    if expected_projection_hash is not None and expected_projection_hash != (
        public_projection_hash
    ):
        raise ValueError("Research V4 public projection authority mismatch")
    raw_reference = decision_reference or scenario_projection.get("decisionReference")
    reference = None
    if raw_reference is not None:
        if (
            not isinstance(raw_reference, Mapping)
            or set(raw_reference) != {"resultCode", "decisionOrdinal"}
            or not isinstance(raw_reference.get("resultCode"), str)
            or re.fullmatch(r"LAF-[A-Z0-9]{10}", raw_reference["resultCode"]) is None
            or type(raw_reference.get("decisionOrdinal")) is not int
            or raw_reference["decisionOrdinal"] != 1
        ):
            raise ValueError("Research V4 decision reference is invalid")
        reference = {
            "resultCode": raw_reference["resultCode"],
            "decisionOrdinal": 1,
        }
    return {
        "schemaVersion": 1,
        "decisionReference": reference,
        "publicProjectionHash": public_projection_hash,
        "phase": str(exact_state.get("phase") or ""),
        "decisionKind": decision_kind,
        "observerSeat": observer_seat,
        "nextActor": next_actor,
        "legalActions": legal_actions,
        "publicState": public_state,
    }


OBSERVER_SAFE_EVENT_TYPES_V1 = _PUBLIC_EVENT_TYPES


__all__ = [
    "OBSERVER_SAFE_EVENT_TYPES_V1",
    "build_canonical_observer_information_v1",
    "build_observer_safe_exact_state_v1",
    "canonical_observer_information_hash_v1",
]
