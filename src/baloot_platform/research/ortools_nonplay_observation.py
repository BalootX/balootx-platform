"""Observer-safe perfect-recall identities for non-play proof grafts."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Seat, Suit
from baloot_platform.core.exact_state_view import (
    canonical_observer_information_hash_v1,
)
from baloot_platform.core.models import Project
from baloot_platform.core.project_ir import project_id_v1
from baloot_platform.core.rules import RuleBook

from .ortools_complete_state_graph import (
    CompleteStateDecisionObservationV1,
    CompleteStateDecisionObserverV1,
)
from .ortools_exact_state import (
    OrToolsNormalizedExactStateV1,
    _effective_prefix_hash,
    _render_native_play_prefix,
    _native_forced_pass_certificate,
    _certified_project_followup_rows,
)
from .ortools_project_fragment import ProjectLifecycleFragmentEvidenceV1
from .ortools_nonplay_graphs import (
    NonPlayDecisionObservationV1,
    NonPlayDecisionObserverV1,
    bind_escalation_recall_v1,
)
from .ortools_play_tail import TailPrefixActionV1
from .scenario_contracts import ExactTracePrefixV1, canonical_sha256


_DECISION_KINDS = frozenset({"bid", "escalation", "project"})
_HISTORY_STAGES = frozenset({"auction", "escalation", "project", "play"})
_HISTORY_KEYS = frozenset(
    {
        "ordinal",
        "stage",
        "actor",
        "action",
        "contract",
        "trumpSuit",
        "mode",
        "owner",
        "kind",
        "value",
        "highRank",
        "ekka",
    }
)
_STAGE_HISTORY_KEYS = {
    "auction": frozenset({"contract", "trumpSuit"}),
    "escalation": frozenset({"mode"}),
    "project": frozenset({"owner", "kind", "value", "highRank"}),
    "play": frozenset({"ekka"}),
}


def _history_row(
    ordinal: int,
    *,
    stage: str,
    actor: Any,
    action: Any,
    **values: Any,
) -> dict[str, Any]:
    if type(action) is not str or not action.strip():
        raise ValueError("public non-play history action is required")
    return {
        "ordinal": ordinal,
        "stage": stage,
        "actor": Seat.parse(actor).name.lower(),
        "action": action.strip(),
        **{key: value for key, value in values.items() if value is not None},
    }


def _strict_public_history(
    history: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for ordinal, value in enumerate(history):
        if not isinstance(value, Mapping):
            raise TypeError("public non-play history items must be objects")
        unknown = set(value) - _HISTORY_KEYS
        if unknown:
            raise ValueError("public non-play history contains private/unknown fields")
        if value.get("ordinal") != ordinal or type(value.get("ordinal")) is not int:
            raise ValueError("public non-play history ordinals must be contiguous")
        stage = value.get("stage")
        actor = value.get("actor")
        action = value.get("action")
        if stage not in _HISTORY_STAGES:
            raise ValueError("public non-play history stage is invalid")
        optional_keys = set(value) - {"ordinal", "stage", "actor", "action"}
        if not optional_keys <= _STAGE_HISTORY_KEYS[str(stage)]:
            raise ValueError("public non-play history fields do not match its stage")
        parsed_actor = Seat.parse(actor).name.lower()
        if type(action) is not str or not action.strip():
            raise ValueError("public non-play history action is required")
        row: dict[str, Any] = {
            "ordinal": ordinal,
            "stage": str(stage),
            "actor": parsed_actor,
            "action": action.strip(),
        }
        for key in sorted(_HISTORY_KEYS - set(row)):
            if key not in value:
                continue
            item = value[key]
            if item is not None and type(item) not in {str, int, bool}:
                raise TypeError("public non-play history values must be scalar")
            row[key] = item
        if stage == "project" and row["action"] == "declare":
            if set(row) != {
                "ordinal",
                "stage",
                "actor",
                "action",
                "owner",
                "kind",
                "value",
                "highRank",
            }:
                raise ValueError(
                    "public project declaration requires its public semantics"
                )
            if row["owner"] != row["actor"]:
                raise ValueError("public project declaration owner is invalid")
            if type(row["value"]) is not int or type(row["highRank"]) is not int:
                raise TypeError("public project declaration values must be integers")
        elif stage == "project" and optional_keys:
            raise ValueError("public project pass cannot contain project semantics")
        if "ekka" in row and type(row["ekka"]) is not bool:
            raise TypeError("public play Ekka marker must be boolean")
        rows.append(row)
    return tuple(rows)


def _require_owned_prefix(
    normalized_exact: OrToolsNormalizedExactStateV1,
    exact_prefix: ExactTracePrefixV1,
) -> None:
    if not isinstance(normalized_exact, OrToolsNormalizedExactStateV1):
        raise TypeError("non-play observation requires normalized exact state")
    if not isinstance(exact_prefix, ExactTracePrefixV1):
        raise TypeError("non-play observation requires exact trace prefix")
    if normalized_exact.prefix_hash != _effective_prefix_hash(exact_prefix):
        raise ValueError("non-play observation state/prefix ownership mismatch")


def _project_lookup(
    projects: Sequence[Project],
) -> dict[str, Project]:
    lookup: dict[str, Project] = {}
    for project in projects:
        if not isinstance(project, Project):
            raise TypeError(
                "non-play observation projects must be canonical Project values"
            )
        identity = project_id_v1(project)
        if identity in lookup:
            raise ValueError("non-play observation project identity is duplicated")
        lookup[identity] = project
    return lookup


def _auction_rows(
    actions: Sequence[Mapping[str, Any]],
    *,
    start_ordinal: int = 0,
    trump_key: str = "trumpSuit",
) -> tuple[dict[str, Any], ...]:
    rows = tuple(
        _history_row(
            start_ordinal + index,
            stage="auction",
            actor=item.get("actor"),
            action=item.get("action"),
            trumpSuit=item.get(trump_key),
        )
        for index, item in enumerate(actions)
    )
    return _strict_public_history(rows)


def _normalized_auction_rows(normalized: OrToolsNormalizedExactStateV1,
                             prefix: ExactTracePrefixV1) -> tuple[dict[str, Any], ...]:
    actions = tuple(dict(action) for action in prefix.auction_actions)
    for index, action in enumerate(actions):
        if index < len(Seat) and action.get("action") == "hokum" and action.get("trumpSuit") is None:
            action["trumpSuit"] = Suit.parse(normalized.state["purchaseCard"]["suit"]).name.lower()
    if all(item.get("actor") is not None for item in actions):
        return _auction_rows(actions)
    history = normalized.state.get("biddingHistory")
    if not history:
        bidding = normalized.state.get("bidding")
        history = bidding.get("history") if isinstance(bidding, Mapping) else None
    if not isinstance(history, list) or len(history) != len(actions):
        raise ValueError("normalized auction history cannot resolve omitted actors")
    resolved = []
    for action, derived in zip(actions, history, strict=True):
        if not isinstance(derived, Mapping) or derived.get("action") != action.get("action"):
            raise ValueError("normalized auction history disagrees with prefix")
        actor = derived.get("seat")
        if actor is None or (action.get("actor") is not None and Seat.parse(action["actor"]) != Seat.parse(actor)):
            raise ValueError("normalized auction actor disagrees with prefix")
        resolved.append({**action, "actor": Seat.parse(actor).name.lower()})
    return _auction_rows(resolved)


def _escalation_rows(
    actions: Sequence[Mapping[str, Any]],
    *,
    start_ordinal: int,
) -> tuple[dict[str, Any], ...]:
    rows = tuple(
        _history_row(
            start_ordinal + index,
            stage="escalation",
            actor=item.get("actor"),
            action=item.get("action"),
            mode=item.get("mode", item.get("requestedMode")),
        )
        for index, item in enumerate(actions)
    )
    # Validate ordinals in the complete sequence at the caller.
    return rows


def _project_rows(
    actions: Sequence[Mapping[str, Any]],
    projects: Mapping[str, Project],
    *,
    start_ordinal: int,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(actions):
        action = item.get("action")
        if action not in {"pass", "declare"}:
            raise ValueError("public project action is invalid")
        values: dict[str, Any] = {}
        if action == "declare":
            project_id = item.get("projectId")
            if type(project_id) is not str or project_id not in projects:
                raise ValueError("public project declaration identity is not governed")
            project = projects[project_id]
            if Seat.parse(item.get("actor")) != project.owner:
                raise ValueError("public project declaration owner is invalid")
            values = {
                "owner": project.owner.name.lower(),
                "kind": project.kind,
                "value": project.value,
                "highRank": project.high_rank,
            }
        elif item.get("projectId") is not None:
            raise ValueError("public project pass cannot identify a project")
        rows.append(
            _history_row(
                start_ordinal + index,
                stage="project",
                actor=item.get("actor"),
                action=action,
                **values,
            )
        )
    return tuple(rows)


def _canonical_prefix_history(
    exact_prefix: ExactTracePrefixV1,
    projects: Sequence[Project],
    auction: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    lookup = _project_lookup(projects)
    escalation = _escalation_rows(
        exact_prefix.escalation_actions,
        start_ordinal=len(auction),
    )
    project = _project_rows(
        exact_prefix.project_actions,
        lookup,
        start_ordinal=len(auction) + len(escalation),
    )
    return _strict_public_history((*auction, *escalation, *project))


@dataclass(frozen=True, slots=True)
class GovernedProjectDisclosureV1:
    owner: Seat
    kind: str
    value: int
    high_rank: int
    cards: tuple[Card, ...]
    trigger_kind: str
    trigger_value: int

    @classmethod
    def build(
        cls,
        value: "GovernedProjectDisclosureV1 | Mapping[str, Any]",
    ) -> "GovernedProjectDisclosureV1":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("governed project disclosure must be an object")
        expected = {
            "owner",
            "kind",
            "value",
            "highRank",
            "cards",
            "revealTrigger",
        }
        if set(value) != expected:
            raise ValueError("governed project disclosure fields are invalid")
        kind = value.get("kind")
        project_value = value.get("value")
        high_rank = value.get("highRank")
        if type(kind) is not str or not kind.strip():
            raise ValueError("governed project disclosure kind is required")
        if type(project_value) is not int or project_value <= 0:
            raise ValueError("governed project disclosure value must be positive")
        if type(high_rank) is not int or high_rank < 0:
            raise ValueError("governed project disclosure high rank is invalid")
        trigger = value.get("revealTrigger")
        if not isinstance(trigger, Mapping) or set(trigger) not in (
            {"kind", "playCount"},
            {"kind", "trickIndex"},
        ):
            raise ValueError("governed project disclosure trigger is invalid")
        trigger_kind = trigger.get("kind")
        if trigger_kind == "play_count" and set(trigger) == {"kind", "playCount"}:
            trigger_value = trigger.get("playCount")
            valid_trigger = type(trigger_value) is int and 0 <= trigger_value <= 32
        elif trigger_kind == "owner_play_in_trick" and set(trigger) == {
            "kind",
            "trickIndex",
        }:
            trigger_value = trigger.get("trickIndex")
            valid_trigger = type(trigger_value) is int and 0 <= trigger_value <= 7
        else:
            valid_trigger = False
            trigger_value = None
        if not valid_trigger:
            raise ValueError("governed project disclosure timing is invalid")
        raw_cards = value.get("cards")
        if not isinstance(raw_cards, list) or not raw_cards:
            raise ValueError("governed project disclosure cards are required")
        cards = tuple(
            sorted((Card.parse(item) for item in raw_cards), key=lambda c: c.id)
        )
        if len(set(cards)) != len(cards):
            raise ValueError("governed project disclosure cards must be unique")
        return cls(
            Seat.parse(value.get("owner")),
            kind.strip(),
            project_value,
            high_rank,
            cards,
            str(trigger_kind),
            int(trigger_value),
        )

    def is_revealed(self, prefix: Sequence[TailPrefixActionV1]) -> bool:
        if self.trigger_kind == "play_count":
            return len(prefix) >= self.trigger_value
        if self.trigger_kind != "owner_play_in_trick":
            raise ValueError("governed project disclosure trigger is unsupported")
        return any(
            index // 4 == self.trigger_value and action.actor == self.owner
            for index, action in enumerate(prefix)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "owner": self.owner.name.lower(),
            "kind": self.kind,
            "value": self.value,
            "highRank": self.high_rank,
            "cards": [card.id for card in self.cards],
            "revealTrigger": (
                {"kind": "play_count", "playCount": self.trigger_value}
                if self.trigger_kind == "play_count"
                else {
                    "kind": "owner_play_in_trick",
                    "trickIndex": self.trigger_value,
                }
            ),
        }


def canonical_nonplay_observation_hash_v1(
    *,
    decision_kind: str,
    actor: Seat | str | int,
    public_action_history: Sequence[Mapping[str, Any]],
    actor_visible_hand: Sequence[Card | int | str | Mapping[str, Any]],
    revealed_projects: Sequence[GovernedProjectDisclosureV1 | Mapping[str, Any]],
) -> str:
    """Hash only information legally available to one non-play actor."""

    kind = str(decision_kind).strip().lower()
    if kind not in _DECISION_KINDS:
        raise ValueError("non-play observation decision kind is invalid")
    seat = Seat.parse(actor)
    history = _strict_public_history(public_action_history)
    hand = tuple(
        sorted(
            (Card.parse(card) for card in actor_visible_hand), key=lambda card: card.id
        )
    )
    if len(hand) > 8 or len(set(hand)) != len(hand):
        raise ValueError("actor-visible hand is invalid")
    projects = tuple(
        sorted(
            (GovernedProjectDisclosureV1.build(item) for item in revealed_projects),
            key=lambda item: (
                item.trigger_kind,
                item.trigger_value,
                int(item.owner),
                item.kind,
                item.value,
                item.high_rank,
                tuple(card.id for card in item.cards),
            ),
        )
    )
    if len({canonical_sha256(item.to_dict()) for item in projects}) != len(projects):
        raise ValueError("governed project disclosures must be unique")
    core = {
        "schemaVersion": 1,
        "kind": "nonplay_actor_observation_v1",
        "decisionKind": kind,
        "actor": seat.name.lower(),
        "publicActionHistory": list(history),
        "actorVisibleHand": [card.id for card in hand],
        "revealedProjects": [item.to_dict() for item in projects],
    }
    return canonical_sha256(core)


def _mapping_sequence(value: Any, label: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, Mapping) for item in value
    ):
        raise ValueError(f"{label} must be a list of objects")
    return tuple(value)


def _projects_from_graph_state(
    value: Any,
) -> tuple[Project, ...]:
    rows = _mapping_sequence(value, "public project graph projects")
    projects: list[Project] = []
    for row in rows:
        expected = {
            "owner",
            "kind",
            "cards",
            "value",
            "highRank",
            "categoryPriority",
            "declared",
            "declarationOrder",
            "scoring",
        }
        if set(row) != expected:
            raise ValueError("public project graph project fields are invalid")
        cards = row.get("cards")
        if not isinstance(cards, list) or not cards:
            raise ValueError("public project graph project cards are invalid")
        kind = row.get("kind")
        project_value = row.get("value")
        high_rank = row.get("highRank")
        priority = row.get("categoryPriority")
        declaration_order = row.get("declarationOrder")
        if type(kind) is not str or not kind:
            raise ValueError("public project graph project kind is invalid")
        if any(
            type(item) is not int
            for item in (project_value, high_rank, priority, declaration_order)
        ):
            raise TypeError("public project graph project values must be integers")
        if (
            type(row.get("declared")) is not bool
            or type(row.get("scoring")) is not bool
        ):
            raise TypeError("public project graph lifecycle must be boolean")
        projects.append(
            Project(
                Seat.parse(row.get("owner")),
                kind,
                tuple(Card.parse(card) for card in cards),
                project_value,
                high_rank,
                priority,
                declared=row["declared"],
                declaration_order=declaration_order,
                scoring=row["scoring"],
            )
        )
    _project_lookup(projects)
    return tuple(projects)


def _root_public_history(
    observation: NonPlayDecisionObservationV1,
    *,
    exact_prefix: ExactTracePrefixV1,
    decision_kind: str,
    auction_prefix: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    if observation.stage != decision_kind:
        raise ValueError("non-play root observation stage is inconsistent")
    public_state = observation.public_state
    if not isinstance(public_state, Mapping):
        raise TypeError("non-play root public state must be an object")
    escalation_prefix = _escalation_rows(
        exact_prefix.escalation_actions,
        start_ordinal=len(auction_prefix),
    )

    if decision_kind == "bid":
        if public_state.get("stage") != "auction":
            raise ValueError("bid root public state stage is invalid")
        state = public_state.get("state")
        if not isinstance(state, Mapping):
            raise ValueError("bid root public state is invalid")
        records = _mapping_sequence(state.get("records"), "public auction records")
        history = _auction_rows(records, trump_key="trump")
        if history[: len(auction_prefix)] != auction_prefix:
            raise ValueError("bid root public history disagrees with exact prefix")
        return history

    if decision_kind == "escalation":
        if public_state.get("stage") != "escalation":
            raise ValueError("escalation root public state stage is invalid")
        actions = _mapping_sequence(
            public_state.get("actions"), "public escalation actions"
        )
        dynamic = _escalation_rows(actions, start_ordinal=len(auction_prefix))
        if dynamic[: len(escalation_prefix)] != escalation_prefix:
            raise ValueError(
                "escalation root public history disagrees with exact prefix"
            )
        return _strict_public_history((*auction_prefix, *dynamic))

    if public_state.get("stage") != "project":
        raise ValueError("project root public state stage is invalid")
    actions = _mapping_sequence(public_state.get("prefix"), "public project actions")
    exact_actions = tuple(dict(item) for item in exact_prefix.project_actions)
    if actions[: len(exact_actions)] != exact_actions:
        raise ValueError("project root public history disagrees with exact prefix")
    projects = _projects_from_graph_state(public_state.get("projects"))
    lookup = _project_lookup(projects)
    dynamic = _project_rows(
        actions,
        lookup,
        start_ordinal=len(auction_prefix) + len(escalation_prefix),
    )
    return _strict_public_history((*auction_prefix, *escalation_prefix, *dynamic))


def _visible_hand(
    state: Mapping[str, Any],
    actor: Seat,
) -> tuple[Card, ...]:
    hands = state.get("hands")
    if not isinstance(hands, Mapping):
        raise ValueError("normalized actor-visible hands are missing")
    value = hands.get(actor.name.lower())
    if not isinstance(value, list):
        raise ValueError("normalized actor-visible hand is missing")
    return tuple(Card.parse(card) for card in value)


def _revealed_projects(
    state: Mapping[str, Any],
) -> tuple[GovernedProjectDisclosureV1, ...]:
    value = state.get("projects") or []
    rows = _mapping_sequence(value, "normalized public projects")
    result: list[GovernedProjectDisclosureV1] = []
    for row in rows:
        if row.get("revealed") is not True:
            continue
        cards = row.get("cards")
        if not isinstance(cards, list) or not cards:
            raise ValueError("revealed normalized project cards are missing")
        result.append(
            GovernedProjectDisclosureV1.build(
                {
                    "owner": row.get("owner"),
                    "kind": row.get("kind"),
                    "value": row.get("value"),
                    "highRank": row.get("highRank"),
                    "cards": cards,
                    "revealTrigger": {"kind": "play_count", "playCount": 0},
                }
            )
        )
    identities = [
        (
            item.owner,
            item.kind,
            item.value,
            item.high_rank,
            item.cards,
        )
        for item in result
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("normalized public project identity is duplicated")
    return tuple(result)


def _root_observation_hash(state, kind, actor, history, revealed, base_hash):
    semantic_hash = canonical_nonplay_observation_hash_v1(
        decision_kind=kind, actor=actor, public_action_history=history,
        actor_visible_hand=_visible_hand(state, actor), revealed_projects=revealed)
    return canonical_sha256({"schemaVersion": 1, "kind": "nonplay_root_decision_observation_v1",
                            "baseObserverInformationHash": base_hash, "semanticObservationHash": semantic_hash})


def _build_native_project_checkpoint_observer(normalized, prefix, *, path_oracle=None):
    if normalized.decision_kind == "project":
        _certified_project_followup_rows(normalized, prefix)
    else:
        _require_owned_prefix(normalized, prefix)
    if normalized.decision_kind not in {"play", "project"} or len(prefix.project_actions) != len(prefix.play_actions) + 1:
        raise ValueError("native checkpoint observer requires an interleaved play root")
    baseline = deepcopy(dict(normalized.state))
    receipt = normalized.normalization_receipt.payload
    certificate = _native_forced_pass_certificate(normalized)
    continuation = None
    branch_model_hash = None
    if path_oracle is not None:
        if (path_oracle.start.source_state_hash != normalized.state_hash
                or path_oracle.project_continuation_actions is None):
            raise ValueError("project branch observer requires its normalized source model")
        continuation = tuple(action.to_dict() for action in path_oracle.project_continuation_actions)
        branch_model_hash = path_oracle.model_hash
    elif certificate is None:
        raise ValueError("native checkpoint observer requires forced declaration evidence")
    if certificate is not None and certificate["modelHash"] != receipt["derivedPlay"]["modelHash"]:
        raise ValueError("native checkpoint observer certificate ownership mismatch")
    auction = _normalized_auction_rows(normalized, prefix)
    authored = _authored_play_prefix(prefix)
    marker = next(index for index, event in enumerate(baseline["events"])
                  if event["type"] == "project_holdings_detected")
    baseline["events"] = baseline["events"][:marker]
    baseline["trumpBroken"] = False

    def observe(value):
        play, checkpoint = value.native_play, value.project_checkpoint
        if (play is None or checkpoint is None or play.model_hash != checkpoint.model_hash
                or (branch_model_hash is not None and play.model_hash != branch_model_hash)
                or not len(authored) < play.prefix_length < len(play.plays)
                or tuple(play.plays[:play.prefix_length]) != value.prefix
                or value.prefix[:len(authored)] != authored
                or value.actor != play.next_actor):
            raise ValueError("native checkpoint observer play evidence mismatch")
        if any(tuple(Card.parse(card) for card in baseline["initialHands"][seat.name.lower()])
               != tuple(play.initial_hands[int(seat)]) for seat in Seat):
            raise ValueError("native checkpoint observer changed hidden-world ownership")
        actions = [dict(action) for action in prefix.project_actions]
        for step in range(len(actions), min(play.prefix_length, 4)):
            actor = play.plays[step].actor.name.lower()
            if continuation is not None:
                index = step - len(prefix.project_actions)
                if index >= len(continuation) or continuation[index]["actor"] != actor:
                    raise ValueError("project branch observer declaration order mismatch")
                actions.append(dict(continuation[index]))
            else:
                if actor not in certificate["seats"]:
                    raise ValueError("native checkpoint observer crossed optional declaration")
                actions.append({"actor": actor, "action": "pass"})
        if continuation is None and value.actor.name.lower() not in certificate["seats"]:
            raise ValueError("native checkpoint observer requires a forced declaration")
        # Projection replaces the play fields and only appends to events.
        # Keep the captured baseline isolated, without recopying fixed hands.
        state = {**baseline, "events": list(baseline["events"])}
        _render_native_play_prefix(state, play=play, projection=checkpoint,
            trump=Suit.parse(state["trumpSuit"]) if state.get("trumpSuit") is not None else None,
            project_actions=actions)
        if play.prefix_length >= 4:
            return canonical_observer_information_hash_v1(complete_state=state, acting_seat=value.actor.name.lower())
        escalation = _escalation_rows(prefix.escalation_actions, start_ordinal=len(auction))
        project_history = _project_rows(actions, _project_lookup(checkpoint.projects),
                                       start_ordinal=len(auction) + len(escalation))
        history = _strict_public_history((*auction, *escalation, *project_history))
        return bind_escalation_recall_v1(_root_observation_hash(state, "project", value.actor,
            history, _revealed_projects(state), canonical_observer_information_hash_v1(
                complete_state=state, acting_seat=value.actor.name.lower())), ())

    return observe


@dataclass(frozen=True, slots=True)
class _RootObserver:
    observe: NonPlayDecisionObserverV1
    native_checkpoint_factory: Callable | None = None

    def __call__(self, value):
        return self.observe(value)


def build_nonplay_root_observer_v1(
    normalized_exact: OrToolsNormalizedExactStateV1,
    exact_prefix: ExactTracePrefixV1,
    decision_kind: str,
) -> NonPlayDecisionObserverV1:
    """Build an observer projector for bid, escalation, or project graph roots."""

    _require_owned_prefix(normalized_exact, exact_prefix)
    kind = str(decision_kind).strip().lower()
    if kind not in _DECISION_KINDS or normalized_exact.decision_kind != kind:
        raise ValueError("non-play root observation decision kind is inconsistent")
    normalized_exact = replace(normalized_exact, state=deepcopy(dict(normalized_exact.state)))
    revealed_projects = _revealed_projects(normalized_exact.state)
    auction_prefix = _normalized_auction_rows(normalized_exact, exact_prefix)
    base_hashes: dict[Seat, str] = {}

    def observe(value: NonPlayDecisionObservationV1) -> str:
        if not isinstance(value, NonPlayDecisionObservationV1):
            raise TypeError("non-play decision observation is required")
        history = _root_public_history(
            value,
            exact_prefix=exact_prefix,
            decision_kind=kind,
            auction_prefix=auction_prefix,
        )
        if value.actor not in base_hashes:
            base_hashes[value.actor] = canonical_observer_information_hash_v1(
                complete_state=normalized_exact.state, acting_seat=value.actor.name.lower())
        return bind_escalation_recall_v1(_root_observation_hash(normalized_exact.state, kind,
            value.actor, history, revealed_projects, base_hashes[value.actor]), value.own_private_windows)

    native = normalized_exact.normalization_receipt.payload.get("backendId") == "ortools_cp_sat_v3"
    return _RootObserver(observe, _build_native_project_checkpoint_observer if native and kind == "project" else None)


def _play_rows(
    prefix: Sequence[TailPrefixActionV1],
    *,
    start_ordinal: int,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "ordinal": start_ordinal + index,
            "stage": "play",
            "actor": action.actor.name.lower(),
            "action": action.card.code,
            "ekka": action.ekka,
        }
        for index, action in enumerate(prefix)
    )


def _authored_play_prefix(
    exact_prefix: ExactTracePrefixV1,
) -> tuple[TailPrefixActionV1, ...]:
    return tuple(
        TailPrefixActionV1.build(
            item.get("actor"),
            item.get("card"),
            ekka=item.get("ekka") is True,
        )
        for item in exact_prefix.play_actions
    )


def _require_native_project_scoring_v1(
    rules: RuleBook,
    normalized: OrToolsNormalizedExactStateV1,
    projects: Sequence[Project],
    evidence: ProjectLifecycleFragmentEvidenceV1 | None,
) -> None:
    """Check owned CP evidence; never recompute project-rule truth in Python."""
    if not isinstance(evidence, ProjectLifecycleFragmentEvidenceV1):
        raise ValueError("native observation requires CP project evidence")
    receipt = dict(normalized.normalization_receipt.payload)
    receipt_hash = receipt.pop("sha256", None)
    if receipt_hash != canonical_sha256(receipt):
        raise ValueError("native observation normalization receipt checksum mismatch")
    if (
        receipt.get("backendId") != "ortools_cp_sat_v3"
        or receipt.get("solverStatus") != "optimal"
        or receipt.get("rulesHash") != rules.sha256
        or normalized.rules_hash != rules.sha256
        or receipt.get("prefixHash") != normalized.prefix_hash
        or receipt.get("exactStateId") != normalized.exact_state_id
        or receipt.get("stateHash") != normalized.state_hash
        or canonical_sha256(normalized.state) != normalized.state_hash
    ):
        raise ValueError("native observation normalization receipt ownership mismatch")
    document = evidence.to_dict()
    digest = document.pop("evidenceHash")
    if digest != canonical_sha256(document):
        raise ValueError("native observation project evidence checksum mismatch")
    owners = {
        Card.parse(card).id: seat
        for seat in Seat
        for card in normalized.state["initialHands"][seat.name.lower()]
    }
    if len(evidence.card_owners) != 32 or owners != dict(enumerate(evidence.card_owners)):
        raise ValueError("native observation project evidence ownership mismatch")
    expected = Counter(
        (item.owner, item.kind, item.cards, item.value, item.high_rank,
         item.category_priority, item.declared,
         item.declaration_order if item.declaration_order is not None else 999, item.awarded)
        for item in evidence.projects if item.selected
    )
    actual = Counter(
        (item.owner, item.kind, item.cards, item.value, item.high_rank,
         item.category_priority, item.declared, item.declaration_order, item.scoring)
        for item in projects
    )
    if actual != expected:
        raise ValueError("trusted projects disagree with native CP project evidence")


def build_nonplay_decision_observer_v1(
    *,
    rules: RuleBook,
    normalized_exact: OrToolsNormalizedExactStateV1,
    exact_prefix: ExactTracePrefixV1,
    decision_kind: str,
    projects: Sequence[Project],
    native_cp_sat: bool = False,
    project_evidence: ProjectLifecycleFragmentEvidenceV1 | None = None,
) -> CompleteStateDecisionObserverV1:
    """Create a perfect-recall callback for complete-state play graph compilation."""

    if not isinstance(rules, RuleBook):
        raise TypeError("non-play observation factory requires RuleBook")
    if type(native_cp_sat) is not bool:
        raise TypeError("native_cp_sat must be boolean")
    _require_owned_prefix(normalized_exact, exact_prefix)
    kind = str(decision_kind).strip().lower()
    if kind not in _DECISION_KINDS or normalized_exact.decision_kind != "play":
        raise ValueError("non-play observation factory decision kind is inconsistent")
    project_rows = tuple(projects)
    history = _canonical_prefix_history(exact_prefix, project_rows,
        _normalized_auction_rows(normalized_exact, exact_prefix))
    initial_hands = normalized_exact.state.get("initialHands")
    if not isinstance(initial_hands, Mapping):
        raise ValueError("normalized initial hands are missing")
    for project in project_rows:
        owner_hand = initial_hands.get(project.owner.name.lower())
        if not isinstance(owner_hand, list):
            raise ValueError("normalized project owner hand is missing")
        owner_cards = {Card.parse(card) for card in owner_hand}
        if not set(project.cards) <= owner_cards:
            raise ValueError("trusted project cards are not owned by its actor")
    normalized_projects = normalized_exact.state.get("projects")
    if normalized_projects is None:
        normalized_projects = []
    if not isinstance(normalized_projects, list) or any(
        not isinstance(item, Mapping) for item in normalized_projects
    ):
        raise ValueError("normalized project state is invalid")
    trusted_public = Counter(
        (
            item.owner.name.lower(),
            item.kind,
            item.value,
            item.high_rank,
            item.revealed,
        )
        for item in project_rows
    )
    normalized_public = Counter(
        (
            str(item.get("owner")),
            str(item.get("kind")),
            item.get("value"),
            item.get("highRank"),
            item.get("revealed") is True,
        )
        for item in normalized_projects
    )
    if normalized_public != trusted_public:
        raise ValueError("trusted projects disagree with the normalized project state")
    trump_raw = normalized_exact.state.get("trumpSuit")
    trump = Suit.parse(trump_raw) if trump_raw is not None else None
    if native_cp_sat:
        _require_native_project_scoring_v1(rules, normalized_exact, project_rows, project_evidence)
    else:
        completed = normalized_exact.state.get("completedTricks") or ()
        rescored, _winner = rules.project_domain.score_team(
            project_rows, trump_suit=trump,
            second_trick_leader=(Seat.parse(completed[0]["winner"]) if completed else None),
        )
        if tuple(item.scoring for item in rescored) != tuple(item.scoring for item in project_rows):
            raise ValueError("trusted project scoring disagrees with Project Domain IR")
    for item in normalized_projects:
        if item.get("revealed") is not True:
            continue
        cards = item.get("cards")
        if not isinstance(cards, list):
            raise ValueError("revealed normalized project cards are missing")
        matching = [
            project
            for project in project_rows
            if project.owner.name.lower() == item.get("owner")
            and project.kind == item.get("kind")
            and project.value == item.get("value")
            and project.high_rank == item.get("highRank")
            and tuple(sorted(card.id for card in project.cards))
            == tuple(sorted(Card.parse(card).id for card in cards))
        ]
        if len(matching) != 1:
            raise ValueError("revealed normalized project cards are not governed")
    scoring_projects = tuple(project for project in project_rows if project.scoring)
    if scoring_projects and (
        not rules.project_domain.should_reveal(1)
        or any(
            rules.project_domain.should_reveal(index)
            for index in range(8)
            if index != 1
        )
    ):
        raise ValueError("project disclosure timing is not governed for observation")
    disclosures = tuple(
        GovernedProjectDisclosureV1(
            project.owner,
            project.kind,
            project.value,
            project.high_rank,
            tuple(project.cards),
            "owner_play_in_trick",
            1,
        )
        for project in scoring_projects
    )
    authored_prefix = _authored_play_prefix(exact_prefix)
    base_play_count = len(authored_prefix)
    revealed_at_base = {
        (
            str(row.get("owner")),
            str(row.get("kind")),
            int(row.get("value")),
            int(row.get("highRank")),
            tuple(sorted(Card.parse(card).id for card in row.get("cards") or ())),
        )
        for row in normalized_exact.state.get("projects") or ()
        if isinstance(row, Mapping) and row.get("revealed") is True
    }
    scheduled_at_base = {
        (
            item.owner.name.lower(),
            item.kind,
            item.value,
            item.high_rank,
            tuple(card.id for card in item.cards),
        )
        for item in disclosures
        if item.is_revealed(authored_prefix)
    }
    if revealed_at_base != scheduled_at_base:
        raise ValueError(
            "project disclosure schedule disagrees with the normalized public state"
        )
    base_observer_hashes = {
        seat: canonical_observer_information_hash_v1(
            complete_state=normalized_exact.state,
            acting_seat=seat.name.lower(),
        )
        for seat in Seat
    }

    def observe(value: CompleteStateDecisionObservationV1) -> str:
        if not isinstance(value, CompleteStateDecisionObservationV1):
            raise TypeError("complete-state decision observation is required")
        prefix = tuple(value.prefix)
        if prefix[:base_play_count] != authored_prefix:
            raise ValueError("decision observation changed the authored play prefix")
        if tuple(value.path) != prefix[base_play_count:]:
            raise ValueError("decision observation path violates perfect recall")
        actor = value.actor
        if len(value.remaining_hands) != 4:
            raise ValueError("decision observation must contain four remaining hands")
        visible_projects = tuple(
            item for item in disclosures if item.is_revealed(prefix)
        )
        public_history = (
            *history,
            *_play_rows(prefix, start_ordinal=len(history)),
        )
        semantic_hash = canonical_nonplay_observation_hash_v1(
            decision_kind=kind,
            actor=actor,
            public_action_history=public_history,
            actor_visible_hand=value.remaining_hands[int(actor)],
            revealed_projects=visible_projects,
        )
        dynamic_public = {
            "currentTrick": [action.to_dict() for action in value.current_trick],
            "trickWinners": [seat.name.lower() for seat in value.trick_winners],
            "trickRawPoints": list(value.trick_raw_points),
            "raw": list(value.raw),
            "tricks": list(value.tricks),
        }
        return canonical_sha256(
            {
                "schemaVersion": 1,
                "kind": "nonplay_complete_state_decision_observation_v1",
                "baseObserverInformationHash": base_observer_hashes[actor],
                "semanticObservationHash": semantic_hash,
                "dynamicPublicState": dynamic_public,
            }
        )

    return observe


__all__ = [
    "GovernedProjectDisclosureV1",
    "build_nonplay_decision_observer_v1",
    "build_nonplay_root_observer_v1",
    "canonical_nonplay_observation_hash_v1",
]
