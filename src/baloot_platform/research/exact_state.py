"""Authoritative normalization for Research V4 exact trace prefixes."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping

from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.cards import Card
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import GamePhase, Seat, Suit
from baloot_platform.core.escalation_ir import EscalationStateV1
from baloot_platform.core.projects import project_lifecycle_matches
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.models import RoundState

from .scenario_contracts import ExactTracePrefixV1, canonical_sha256


def _mapping_path(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise ValueError(f"exact-state assertion path does not exist: {path}")
        current = current[component]
    return current


def _replay_prefix_hash(spec: ExactTracePrefixV1) -> str:
    """Hash only engine-effective input, not optional validation assertions."""

    return canonical_sha256(
        {
            "schemaVersion": 1,
            "deckOrder": list(spec.deck_order),
            "dealer": spec.dealer,
            "roundNumber": spec.round_number,
            "matchScore": list(spec.match_score),
            "auctionActions": [
                {
                    "action": str(action["action"]).strip(),
                    "trumpSuit": (
                        Suit.parse(str(action["trumpSuit"])).name.lower()
                        if action.get("trumpSuit") is not None
                        else None
                    ),
                }
                for action in spec.auction_actions
            ],
            "escalationActions": [
                {
                    "actor": Seat.parse(str(action["actor"])).name.lower(),
                    "action": str(action["action"]).strip(),
                    "mode": (
                        str(action.get("mode") or action.get("requestedMode"))
                        if action.get("mode") is not None
                        or action.get("requestedMode") is not None
                        else None
                    ),
                }
                for action in spec.escalation_actions
            ],
            "playActions": [
                {
                    "card": Card.parse(action["card"]).id,
                    "ekka": action.get("ekka") is True,
                }
                for action in spec.play_actions
            ],
        }
    )


@dataclass(frozen=True, slots=True)
class NormalizedExactStateV1:
    exact_state_id: str
    prefix_hash: str
    rules_hash: str
    state_hash: str
    phase: str
    decision_kind: str
    next_actor: str | None
    legal_actions: tuple[Mapping[str, Any], ...]
    state: Mapping[str, Any]
    project_declarations_verified: int
    project_assertions_verified: int
    event_chain_hash: str
    sha256: str
    round_state: RoundState = field(repr=False, compare=False)
    rules: RuleBook = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "exactStateId": self.exact_state_id,
            "prefixHash": self.prefix_hash,
            "rulesHash": self.rules_hash,
            "stateHash": self.state_hash,
            "phase": self.phase,
            "decisionKind": self.decision_kind,
            "nextActor": self.next_actor,
            "legalActions": [copy.deepcopy(dict(item)) for item in self.legal_actions],
            "state": copy.deepcopy(dict(self.state)),
            "projectDeclarationsVerified": self.project_declarations_verified,
            "projectAssertionsVerified": self.project_assertions_verified,
            "eventChainHash": self.event_chain_hash,
            "sha256": self.sha256,
        }


class ExactStateNormalizerV1:
    """Replay exact input through the canonical bidding and engine transitions."""

    def __init__(self, rules: RuleBook):
        self.rules = rules

    def _verify_project_assertions(
        self,
        state: RoundState,
        assertions: tuple[Mapping[str, Any], ...],
    ) -> int:
        verified = 0
        for assertion in assertions:
            raw = dict(assertion)
            unknown = sorted(
                set(raw)
                - {"owner", "team", "kind", "value", "highRank", "state"}
            )
            if unknown:
                raise ValueError(
                    "exact-state project assertion contains unknown fields: "
                    + ", ".join(unknown)
                )
            requested_state = str(raw.pop("state", "eligible"))
            if requested_state not in {
                "eligible",
                "declared",
                "comparison_winner",
                "awarded",
            }:
                raise ValueError("exact-state project assertion state is invalid")

            def matches(project) -> bool:
                payload = project.to_dict()
                if any(payload.get(key) != value for key, value in raw.items()):
                    return False
                return project_lifecycle_matches(
                    project,
                    requested_state,
                    projects=state.projects,
                    winning_team=state.project_winning_team,
                    trump_suit=state.trump_suit,
                    domain=self.rules.project_domain,
                    second_trick_leader=(state.completed_tricks[0].winner if state.completed_tricks else None),
                )

            if not any(matches(project) for project in state.projects):
                raise ValueError(
                    "exact-state project assertion does not match authoritative projects"
                )
            verified += 1
        return verified

    @staticmethod
    def _verify_project_declarations(
        state_payload: Mapping[str, Any],
        declarations: tuple[Mapping[str, Any], ...],
    ) -> int:
        if not declarations:
            return 0
        actual = [
            {
                key: event[key]
                for key in (
                    "owner",
                    "kind",
                    "value",
                    "highRank",
                    "declarationOrder",
                )
            }
            for event in state_payload.get("events") or ()
            if event.get("type") == "project_declared"
        ]
        expected = [dict(item) for item in declarations]
        if actual != expected:
            raise ValueError(
                "exact-state project declarations do not match authoritative events"
            )
        return len(expected)

    @staticmethod
    def _escalation_legal_actions(
        engine: BalootEngine,
        state: RoundState,
        actor: Seat,
    ) -> tuple[dict[str, Any], ...]:
        if state.declarer is None or state.contract is None:
            raise ValueError("exact escalation state requires a contract and declarer")
        decisions = engine.escalation_domain.legal_decisions(
            EscalationStateV1(
                state.escalation,
                state.locked,
                state.double_seat,
            ),
            declarer=state.declarer,
            contract=state.contract,
            match_score=(int(state.match_score[0]), int(state.match_score[1])),
            actors=(actor,),
        )
        rows: list[dict[str, Any]] = []
        for decision in decisions:
            mode = (
                decision.resolved_mode
                if decision.action in {"double", "four"}
                else None
            )
            code = decision.action if mode is None else f"{decision.action}:{mode}"
            rows.append({"code": code, "action": decision.action, "mode": mode})
        action_order = {
            engine.escalation_domain.closure_action: -1,
            **{
                action: index
                for index, action in enumerate(
                    engine.escalation_domain.supported_actions
                )
            },
        }
        mode_order = {None: -1, "open": 0, "locked": 1}
        return tuple(
            sorted(
                rows,
                key=lambda row: (
                    action_order[str(row["action"])],
                    mode_order[row["mode"]],
                ),
            )
        )

    def normalize(
        self,
        prefix: ExactTracePrefixV1 | Mapping[str, Any],
        *,
        decision_point: Mapping[str, Any] | None = None,
    ) -> NormalizedExactStateV1:
        spec = prefix if isinstance(prefix, ExactTracePrefixV1) else ExactTracePrefixV1.build(prefix)
        decision = dict(decision_point or {"kind": "play", "selector": {}})
        decision_kind = str(decision.get("kind") or "play")
        selector = dict(decision.get("selector") or {})
        prefix_hash = _replay_prefix_hash(spec)
        seed = int(prefix_hash[:16], 16)
        dealer = Seat.parse(spec.dealer)
        engine = BalootEngine(self.rules)
        state = engine.create_round(
            seed=seed,
            dealer=dealer,
            round_number=spec.round_number,
            match_score=spec.match_score,
        )
        engine.initial_deal_from_preordered(state, spec.deck_order)
        if state.purchase_card is None:
            raise AssertionError("authoritative initial deal lacks purchase card")
        state.phase = GamePhase.BIDDING
        bidding = BiddingMachine(
            state.dealer,
            state.purchase_card.suit,
            AuctionDomainIRV1.from_rules(self.rules.data),
            state.purchase_card.rank,
        )
        for index, raw_action in enumerate(spec.auction_actions):
            action = dict(raw_action)
            unknown = sorted(set(action) - {"actor", "action", "trumpSuit"})
            if unknown:
                raise ValueError(
                    f"exact auction action {index} contains unknown fields: {unknown}"
                )
            expected_actor = action.get("actor")
            if expected_actor is not None and bidding.actor != Seat.parse(str(expected_actor)):
                raise ValueError(f"exact auction action {index} actor mismatch")
            selected = str(action.get("action") or "").strip()
            if not selected:
                raise ValueError(f"exact auction action {index} is empty")
            trump_raw = action.get("trumpSuit")
            bidding.submit(
                selected,
                trump=Suit.parse(str(trump_raw)) if trump_raw is not None else None,
            )

        if bidding.result is None:
            if (
                spec.escalation_actions
                or spec.project_declarations
                or spec.project_assertions
                or spec.play_actions
            ):
                raise ValueError(
                    "exact trace cannot contain post-auction actions before auction completion"
                )
            state_payload = state.to_dict(reveal_hands=True)
            state_payload["bidding"] = {
                "round": bidding.round_index,
                "actor": bidding.actor.name.lower() if bidding.actor is not None else None,
                "history": [item.to_dict() for item in bidding.records],
            }
            bid_actions: list[dict[str, Any]] = []
            for action in bidding.legal_actions():
                if action != "hokum":
                    bid_actions.append(
                        {"code": action, "action": action, "trumpSuit": None}
                    )
                elif bidding.round_index == 1:
                    suit = bidding.purchase_suit.name.lower()
                    bid_actions.append(
                        {
                            "code": f"hokum:{suit}",
                            "action": "hokum",
                            "trumpSuit": suit,
                        }
                    )
                else:
                    bid_actions.extend(
                        {
                            "code": f"hokum:{suit.name.lower()}",
                            "action": "hokum",
                            "trumpSuit": suit.name.lower(),
                        }
                        for suit in Suit
                        if suit != bidding.purchase_suit
                    )
            legal_actions = tuple(bid_actions)
            next_actor = bidding.actor.name.lower() if bidding.actor is not None else None
        else:
            purchased = engine.complete_deal_from_bid_result(state, bidding.result)
            if not purchased:
                if (
                    spec.escalation_actions
                    or spec.project_declarations
                    or spec.project_assertions
                    or spec.play_actions
                ):
                    raise ValueError("all-pass exact trace cannot contain post-auction actions")
                legal_actions = ()
                next_actor = None
            else:
                for index, raw_action in enumerate(spec.escalation_actions):
                    action = dict(raw_action)
                    unknown = sorted(
                        set(action) - {"actor", "action", "mode", "requestedMode"}
                    )
                    if unknown:
                        raise ValueError(
                            f"exact escalation action {index} contains unknown fields: {unknown}"
                        )
                    actor = Seat.parse(str(action.get("actor") or ""))
                    engine.apply_escalation(
                        state,
                        seat=actor,
                        action=str(action.get("action") or ""),
                        mode=(
                            str(action.get("mode") or action.get("requestedMode"))
                            if action.get("mode") is not None
                            or action.get("requestedMode") is not None
                            else None
                        ),
                    )
                if decision_kind == "escalation":
                    if (
                        spec.project_declarations
                        or spec.project_assertions
                        or spec.play_actions
                    ):
                        raise ValueError(
                            "escalation decision trace cannot contain project or play actions"
                        )
                    actor_raw = selector.get("actingSeat")
                    if actor_raw is None:
                        raise ValueError(
                            "escalation decisionPoint requires selector.actingSeat"
                        )
                    actor = Seat.parse(str(actor_raw))
                    legal_actions = self._escalation_legal_actions(
                        engine,
                        state,
                        actor,
                    )
                    next_actor = actor.name.lower()
                else:
                    engine.prepare_projects(state)
                    for index, raw_action in enumerate(spec.play_actions):
                        action = dict(raw_action)
                        unknown = sorted(set(action) - {"actor", "card", "ekka"})
                        if unknown:
                            raise ValueError(
                                f"exact play action {index} contains unknown fields: {unknown}"
                            )
                        expected_actor = action.get("actor")
                        if expected_actor is not None and state.current_player != Seat.parse(
                            str(expected_actor)
                        ):
                            raise ValueError(f"exact play action {index} actor mismatch")
                        card_raw = action.get("card")
                        if card_raw is None:
                            raise ValueError(f"exact play action {index} requires card")
                        card = Card.parse(card_raw)
                        legal = engine.legal_moves(state)
                        if card not in legal:
                            raise ValueError(f"exact play action {index} is illegal")
                        ekka = action.get("ekka")
                        if ekka is not None and type(ekka) is not bool:
                            raise TypeError(f"exact play action {index} ekka must be boolean")
                        engine.play_card(state, card, ekka=ekka)
                    if decision_kind == "project":
                        owner_raw = selector.get("owner")
                        if owner_raw is None:
                            raise ValueError(
                                "project decisionPoint requires selector.owner"
                            )
                        owner = Seat.parse(str(owner_raw))
                        if state.current_player != owner:
                            raise ValueError(
                                "project decisionPoint owner must be the current actor"
                            )
                        _updated, announced = engine.project_domain.announce_for_turn(
                            state.projects,
                            owner,
                            trump_suit=state.trump_suit,
                        )
                        if announced is None:
                            raise ValueError(
                                "project decisionPoint has no legal declaration"
                            )
                        project_index = int(selector.get("projectIndex", 0))
                        owned = [
                            project
                            for project in state.projects
                            if project.owner == owner
                        ]
                        authoritative_index = next(
                            (
                                index
                                for index, project in enumerate(owned)
                                if (
                                    project.kind,
                                    project.value,
                                    project.high_rank,
                                )
                                == (
                                    announced.kind,
                                    announced.value,
                                    announced.high_rank,
                                )
                            ),
                            None,
                        )
                        if authoritative_index is None or project_index != authoritative_index:
                            raise ValueError(
                                "project decisionPoint index is not the authoritative declaration"
                            )
                        code = (
                            f"declare:{announced.kind}:{announced.value}:"
                            f"{announced.high_rank}"
                        )
                        legal_actions = (
                            {
                                "code": code,
                                "action": "declare",
                                "owner": owner.name.lower(),
                                "kind": announced.kind,
                                "value": announced.value,
                                "highRank": announced.high_rank,
                            },
                        )
                        next_actor = owner.name.lower()
                    else:
                        legal_actions = tuple(
                            action.to_dict()
                            for action in engine.legal_play_actions(state)
                        )
                        next_actor = (
                            state.current_player.name.lower()
                            if state.current_player is not None
                            else None
                        )
            state_payload = state.to_dict(reveal_hands=True)

        for path, expected in spec.state_assertions.items():
            actual = _mapping_path(state_payload, str(path))
            if canonical_sha256(actual) != canonical_sha256(expected):
                raise ValueError(
                    f"exact-state assertion mismatch at {path}: "
                    f"expected={expected!r}, actual={actual!r}"
                )

        project_declaration_count = self._verify_project_declarations(
            state_payload,
            spec.project_declarations,
        )
        project_count = self._verify_project_assertions(
            state,
            spec.project_assertions,
        )
        state_hash = canonical_sha256(state_payload)
        event_chain_hash = canonical_sha256(state_payload.get("events") or [])
        core = {
            "schemaVersion": 1,
            "prefixHash": prefix_hash,
            "rulesHash": self.rules.sha256,
            "stateHash": state_hash,
            "phase": state.phase.value,
            "decisionKind": decision_kind,
            "nextActor": next_actor,
            "legalActions": [dict(item) for item in legal_actions],
            "eventChainHash": event_chain_hash,
        }
        digest = canonical_sha256(core)
        return NormalizedExactStateV1(
            f"STATE-{digest[:20].upper()}",
            prefix_hash,
            self.rules.sha256,
            state_hash,
            state.phase.value,
            decision_kind,
            next_actor,
            legal_actions,
            state_payload,
            project_declaration_count,
            project_count,
            event_chain_hash,
            digest,
            state.clone(),
            self.rules,
        )


__all__ = ["ExactStateNormalizerV1", "NormalizedExactStateV1"]
