"""Authoritative runtime verification and satisfaction evidence for V4."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

from baloot_platform.core.cards import Card, mask_to_cards
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Rank, Seat, Suit, Team
from baloot_platform.core.projects import project_lifecycle_matches

from .exact_state import NormalizedExactStateV1
from .reference_scoring import ReferenceScoringInterpreterV1
from .proof_service import decode_proof_play_action_v1
from .scenario_contracts import (
    ProofOutcomeV1,
    ResearchScenarioV1,
    ScenarioExpressionV1,
    ScenarioPredicateV1,
    canonical_sha256,
)
from .scenario_registry import ScenarioPredicateRegistryV1


@dataclass(frozen=True, slots=True)
class ScenarioPredicateEvidenceV1:
    predicate_id: str
    predicate_hash: str
    kind: str
    hardness: str
    status: str
    construction_capability: str
    runtime_verifier: str
    observed: Mapping[str, Any]
    exact_state_id: str
    failure_code: str | None
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "predicateId": self.predicate_id,
            "predicateHash": self.predicate_hash,
            "kind": self.kind,
            "hardness": self.hardness,
            "status": self.status,
            "constructionCapability": self.construction_capability,
            "runtimeVerifier": self.runtime_verifier,
            "observed": copy.deepcopy(dict(self.observed)),
            "exactStateId": self.exact_state_id,
            "failureCode": self.failure_code,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ScenarioSatisfactionV1:
    scenario_hash: str
    exact_state_id: str
    constraints_satisfied: bool
    outcome_satisfied: bool
    hard_predicates_satisfied: bool
    status: str
    evidence: tuple[ScenarioPredicateEvidenceV1, ...]
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "scenarioHash": self.scenario_hash,
            "exactStateId": self.exact_state_id,
            "constraintsSatisfied": self.constraints_satisfied,
            "outcomeSatisfied": self.outcome_satisfied,
            "hardPredicatesSatisfied": self.hard_predicates_satisfied,
            "status": self.status,
            "predicates": [item.to_dict() for item in self.evidence],
            "sha256": self.sha256,
        }


def _in_range(value: float, arguments: Mapping[str, Any]) -> bool:
    return float(arguments["minimum"]) <= value <= float(arguments["maximum"])


def _team(value: str) -> Team:
    return Team.NORTH_SOUTH if value == "north_south" else Team.WEST_EAST


class ScenarioRuntimeVerifierV1:
    def __init__(self, registry: ScenarioPredicateRegistryV1 | None = None):
        self.registry = registry or ScenarioPredicateRegistryV1()

    @staticmethod
    def _action_set_matches(
        actual: list[dict[str, Any]],
        expected: list[dict[str, Any]],
        *,
        complete: bool,
    ) -> bool:
        def matches(wanted: Mapping[str, Any], observed: Mapping[str, Any]) -> bool:
            return all(observed.get(key) == value for key, value in wanted.items())

        if complete and len(actual) != len(expected):
            return False
        return all(any(matches(wanted, row) for row in actual) for wanted in expected)

    @staticmethod
    def _tactical_context_matches(
        row: Mapping[str, Any], arguments: Mapping[str, Any]
    ) -> bool:
        aliases = {
            "actingSeat": ("actingSeat", "actor"),
            "observer": ("observer", "observerSeat"),
            "targetSeat": ("targetSeat",),
            "targetTeam": ("targetTeam",),
        }
        for argument, keys in aliases.items():
            if argument not in arguments:
                continue
            actual = next((row[key] for key in keys if row.get(key) is not None), None)
            if actual != arguments[argument]:
                return False
        return True

    def _evaluate_predicate(
        self,
        predicate: ScenarioPredicateV1,
        normalized: NormalizedExactStateV1,
        *,
        proof_outcome: ProofOutcomeV1 | None,
        tactical_evidence: Mapping[str, Any] | None,
    ) -> tuple[str, dict[str, Any], str | None]:
        state = normalized.round_state
        args = dict(predicate.arguments)
        kind = predicate.kind
        bidding_state = normalized.state.get("bidding")
        raw_auction_history = (
            list(bidding_state.get("history", []))
            if isinstance(bidding_state, Mapping)
            else [item.to_dict() for item in state.bid_history]
        )
        auction_history = [
            {
                "round": int(item.get("round", item.get("roundIndex", 0))),
                "turn": int(item.get("turn", 0)),
                "actor": item.get("actor", item.get("seat")),
                "action": item.get("action"),
                "contract": item.get("contract"),
                "trumpSuit": item.get("trumpSuit"),
                "bidder": item.get("bidder"),
                "buyer": item.get("buyer"),
            }
            for item in raw_auction_history
        ]
        satisfied: bool | None
        observed: dict[str, Any] = {}
        if kind in {"deal.card_owner", "deal.card_allowed", "deal.card_forbidden"}:
            card = Card.parse(args["card"])
            hand_masks = (
                state.initial_five
                if args["handStage"] == "initial_five"
                else state.initial_hands
            )
            owner = next(
                (seat for seat in Seat if hand_masks[int(seat)] & (1 << card.id)),
                None,
            )
            if kind == "deal.card_owner":
                satisfied = owner == Seat.parse(args["seat"])
            elif kind == "deal.card_forbidden":
                satisfied = owner != Seat.parse(args["seat"])
            else:
                satisfied = owner is not None and owner.name.lower() in set(
                    args["seats"]
                )
            observed = {
                "card": card.id,
                "owner": owner.name.lower() if owner is not None else None,
                "handStage": args["handStage"],
            }
        elif kind in {
            "deal.suit_count",
            "deal.rank_count",
            "deal.high_card_count",
            "deal.low_card_count",
        }:
            seat = Seat.parse(args["seat"])
            hand_masks = (
                state.initial_five
                if args["handStage"] == "initial_five"
                else state.initial_hands
            )
            cards = mask_to_cards(hand_masks[int(seat)])
            if kind == "deal.suit_count":
                count = sum(card.suit == Suit.parse(args["suit"]) for card in cards)
            elif kind == "deal.rank_count":
                count = sum(card.rank == Rank.parse(args["rank"]) for card in cards)
            elif kind == "deal.high_card_count":
                count = sum(
                    card.rank in {Rank.ACE, Rank.TEN, Rank.KING} for card in cards
                )
            else:
                count = sum(
                    card.rank in {Rank.SEVEN, Rank.EIGHT, Rank.NINE} for card in cards
                )
            satisfied = _in_range(count, args)
            observed = {
                "seat": seat.name.lower(),
                "count": count,
                "handStage": args["handStage"],
            }
        elif kind == "deal.purchase_card":
            card_id = (
                state.purchase_card.id if state.purchase_card is not None else None
            )
            satisfied = card_id == Card.parse(args["card"]).id
            observed = {"card": card_id}
        elif kind == "deal.purchase_suit":
            actual = (
                state.purchase_card.suit.name.lower()
                if state.purchase_card is not None
                else None
            )
            satisfied = actual == args["suit"]
            observed = {"suit": actual}
        elif kind == "deal.remaining_card_owner":
            card = Card.parse(args["card"])
            owners = [seat for seat in Seat if state.hands[int(seat)] & (1 << card.id)]
            owner = owners[0] if len(owners) == 1 else None
            satisfied = owner == Seat.parse(args["seat"])
            observed = {
                "card": card.id,
                "owner": owner.name.lower() if owner is not None else None,
            }
        elif kind in {"deal.remaining_suit_count", "deal.remaining_rank_count"}:
            seat = Seat.parse(args["seat"])
            cards = state.hand(seat)
            if kind == "deal.remaining_suit_count":
                actual = sum(card.suit == Suit.parse(args["suit"]) for card in cards)
            else:
                actual = sum(card.rank == Rank.parse(args["rank"]) for card in cards)
            satisfied = _in_range(actual, args)
            observed = {"seat": args["seat"], "count": actual}
        elif kind == "deal.team_high_card_balance":
            team = _team(args["team"])
            hand_masks = (
                state.initial_five
                if args["handStage"] == "initial_five"
                else state.initial_hands
            )
            seats = [seat for seat in Seat if seat.team == team]
            counts = {
                seat.name.lower(): sum(
                    card.rank in {Rank.ACE, Rank.TEN, Rank.KING}
                    for card in mask_to_cards(hand_masks[int(seat)])
                )
                for seat in seats
            }
            difference = abs(
                counts[seats[0].name.lower()] - counts[seats[1].name.lower()]
            )
            satisfied = difference <= int(args["maximumDifference"])
            observed = {
                "team": args["team"],
                "counts": counts,
                "difference": difference,
            }
        elif kind == "deal.seat_relation":
            seat = Seat.parse(args["seat"])
            other = Seat.parse(args["otherSeat"])
            relation = str(args["relation"])
            actual = {
                "same": other == seat,
                "partner": other == seat.partner,
                "next": other == seat.next(),
                "previous": other == seat.previous(),
                "opponent": other.team != seat.team,
            }[relation]
            satisfied = actual
            observed = {
                "seat": args["seat"],
                "otherSeat": args["otherSeat"],
                "relation": relation,
                "matches": actual,
            }
        elif kind == "auction.dealer":
            satisfied = state.dealer == Seat.parse(args["seat"])
            observed = {"seat": state.dealer.name.lower()}
        elif kind == "auction.buyer":
            satisfied = state.buyer == Seat.parse(args["seat"])
            observed = {
                "seat": state.buyer.name.lower() if state.buyer is not None else None
            }
        elif kind == "auction.ashkal_buyer_relation":
            bidder = state.winning_bidder
            buyer = state.buyer
            actual = None
            if state.contract is not None and state.contract.value == "ashkal":
                if bidder is not None and buyer == bidder:
                    actual = "bidder"
                elif bidder is not None and buyer == bidder.partner:
                    actual = "bidder_partner"
            satisfied = actual == args["relation"]
            observed = {
                "contract": state.contract.value
                if state.contract is not None
                else None,
                "bidder": bidder.name.lower() if bidder is not None else None,
                "buyer": buyer.name.lower() if buyer is not None else None,
                "relation": actual,
            }
        elif kind == "auction.contract":
            actual = state.contract.value if state.contract is not None else None
            satisfied = actual == args["contract"]
            observed = {"contract": actual}
        elif kind == "auction.trump":
            actual = (
                state.trump_suit.name.lower() if state.trump_suit is not None else None
            )
            satisfied = actual == args["suit"]
            observed = {"suit": actual}
        elif kind == "auction.purchase_round":
            rounds = [
                int(item["round"])
                for item in auction_history
                if item.get("contract") is not None
            ]
            actual = rounds[-1] if rounds else None
            satisfied = actual == int(args["round"])
            observed = {"round": actual}
        elif kind == "auction.round":
            actual = (
                int(bidding_state["round"])
                if isinstance(bidding_state, Mapping)
                and bidding_state.get("round") is not None
                else max(
                    (int(item["round"]) for item in auction_history),
                    default=None,
                )
            )
            satisfied = actual == int(args["round"])
            observed = {"round": actual}
        elif kind == "auction.action_sequence":
            actual = copy.deepcopy(auction_history)
            expected = list(args["actions"])
            compared = actual if args["complete"] else actual[: len(expected)]
            satisfied = len(compared) == len(expected) and all(
                observed["action"] == wanted["action"]
                and ("actor" not in wanted or observed["actor"] == wanted["actor"])
                and (
                    "trumpSuit" not in wanted
                    or observed["trumpSuit"] == wanted["trumpSuit"]
                )
                for observed, wanted in zip(compared, expected, strict=True)
            )
            observed = {"actions": actual, "complete": bool(args["complete"])}
        elif kind == "auction.legal_actions":
            if normalized.decision_kind != "bid":
                return (
                    "not_evaluated",
                    {"decisionKind": normalized.decision_kind},
                    "auction_legal_actions_unavailable",
                )
            actual = [
                {
                    "action": str(item["action"]),
                    **(
                        {"trumpSuit": str(item["trumpSuit"])}
                        if item.get("trumpSuit") is not None
                        else {}
                    ),
                }
                for item in normalized.legal_actions
            ]
            expected = [dict(item) for item in args["actions"]]
            satisfied = self._action_set_matches(
                actual,
                expected,
                complete=bool(args["complete"]),
            )
            observed = {"actions": actual, "complete": bool(args["complete"])}
        elif kind == "auction.pass_count":
            actual = sum(item.get("action") == "pass" for item in auction_history)
            satisfied = _in_range(actual, args)
            observed = {"count": actual}
        elif kind == "auction.terminal":
            actual = state.contract is not None or state.phase.value not in {"bidding"}
            satisfied = actual == bool(args["terminal"])
            observed = {"terminal": actual}
        elif kind == "escalation.level":
            satisfied = state.escalation == args["level"]
            observed = {"level": state.escalation}
        elif kind == "escalation.action_sequence":
            actual = [
                {
                    "actor": item["seat"],
                    "action": item["action"],
                    "mode": item.get("mode"),
                }
                for item in state.escalation_history
            ]
            expected = list(args["actions"])
            compared = actual if args.get("complete", True) else actual[: len(expected)]
            satisfied = len(compared) == len(expected) and all(
                observed["actor"] == wanted["actor"]
                and observed["action"] == wanted["action"]
                and ("mode" not in wanted or observed["mode"] == wanted["mode"])
                for observed, wanted in zip(compared, expected, strict=True)
            )
            observed = {
                "actions": actual,
                "complete": bool(args.get("complete", True)),
            }
        elif kind == "escalation.locked":
            satisfied = state.locked == bool(args["locked"])
            observed = {"locked": state.locked}
        elif kind == "escalation.eligible_actor":
            actor = (
                Seat.parse(normalized.next_actor)
                if normalized.next_actor is not None
                else None
            )
            has_escalation = normalized.decision_kind == "escalation" and any(
                item.get("action") != "decline" for item in normalized.legal_actions
            )
            actual = bool(
                has_escalation
                and actor is not None
                and actor.team == _team(args["team"])
                and (args.get("seat") is None or actor == Seat.parse(args["seat"]))
            )
            satisfied = actual == bool(args["eligible"])
            observed = {
                "team": actor.team.name.lower() if actor is not None else None,
                "seat": actor.name.lower() if actor is not None else None,
                "eligible": actual,
            }
        elif kind == "escalation.legal_actions":
            if normalized.decision_kind != "escalation":
                return (
                    "not_evaluated",
                    {"decisionKind": normalized.decision_kind},
                    "escalation_legal_actions_unavailable",
                )
            actual = [
                {
                    "action": str(item["action"]),
                    **(
                        {"mode": str(item["mode"])}
                        if item.get("mode") is not None
                        else {}
                    ),
                }
                for item in normalized.legal_actions
            ]
            expected = [dict(item) for item in args["actions"]]
            satisfied = self._action_set_matches(
                actual,
                expected,
                complete=bool(args["complete"]),
            )
            observed = {"actions": actual, "complete": bool(args["complete"])}
        elif kind == "escalation.terminal":
            actual = normalized.decision_kind != "escalation" or not any(
                item.get("action") != "decline" for item in normalized.legal_actions
            )
            satisfied = actual == bool(args["terminal"])
            observed = {"terminal": actual}
        elif kind in {"project.exists", "project.count", "project.combined_value"}:
            team = _team(args["team"])
            projects = [item for item in state.projects if item.owner.team == team]
            requested_state = str(args.get("state") or "eligible")
            projects = [
                item
                for item in projects
                if project_lifecycle_matches(
                    item,
                    requested_state,
                    projects=state.projects,
                    winning_team=state.project_winning_team,
                    trump_suit=state.trump_suit,
                    domain=normalized.rules.project_domain,
                    second_trick_leader=(state.completed_tricks[0].winner if state.completed_tricks else None),
                )
            ]
            if kind == "project.exists":
                projects = [
                    item
                    for item in projects
                    if item.kind == args["kind"]
                    and item.value == int(args["value"])
                    and ("owner" not in args or item.owner == Seat.parse(args["owner"]))
                ]
                satisfied = bool(projects)
            elif kind == "project.count":
                satisfied = _in_range(len(projects), args)
            else:
                satisfied = _in_range(sum(item.value for item in projects), args)
            observed = {
                "team": args["team"],
                "state": requested_state,
                "count": len(projects),
                "combinedValue": sum(item.value for item in projects),
                "projects": [item.to_dict() for item in projects],
            }
        elif kind == "project.baloot":
            team = _team(args["team"])
            actual = team in state.baloot_teams
            satisfied = actual == bool(args["achieved"])
            observed = {"team": args["team"], "achieved": actual}
        elif kind == "project.overlap_count":
            owner = Seat.parse(args["owner"])
            projects = [item for item in state.projects if item.owner == owner]
            counts: dict[int, int] = {}
            for project in projects:
                for card in project.cards:
                    counts[card.id] = counts.get(card.id, 0) + 1
            actual = sum(count > 1 for count in counts.values())
            satisfied = _in_range(actual, args)
            observed = {"owner": args["owner"], "overlapCardCount": actual}
        elif kind == "play.trick_index":
            actual = (
                state.current_trick.index
                if state.current_trick is not None
                else len(state.completed_tricks)
            )
            satisfied = _in_range(actual, args)
            observed = {"trickIndex": actual}
        elif kind == "play.current_player":
            actual = (
                state.current_player.name.lower()
                if state.current_player is not None
                else None
            )
            satisfied = actual == args["seat"]
            observed = {"seat": actual}
        elif kind == "play.remaining_hand_size":
            seat = Seat.parse(args["seat"])
            actual = state.hands[int(seat)].bit_count()
            satisfied = _in_range(actual, args)
            observed = {"seat": args["seat"], "count": actual}
        elif kind == "play.card_prefix":
            actual = [item.card.id for item in state.played_cards]
            expected = [Card.parse(item).id for item in args["cards"]]
            complete = bool(args.get("complete", False))
            satisfied = (
                actual == expected if complete else actual[: len(expected)] == expected
            )
            observed = {"cardIds": actual, "complete": complete}
        elif kind == "play.action_prefix":
            actual = [
                {"card": item.card.id, "ekka": bool(item.ekka)}
                for item in state.played_cards
            ]
            expected = [
                {
                    "card": Card.parse(item["card"]).id,
                    "ekka": bool(item["ekka"]),
                }
                for item in args["actions"]
            ]
            complete = bool(args.get("complete", False))
            satisfied = (
                actual == expected if complete else actual[: len(expected)] == expected
            )
            observed = {"actions": actual, "complete": complete}
        elif kind == "play.current_trick_leader":
            actual = (
                state.current_trick.leader.name.lower()
                if state.current_trick is not None
                else None
            )
            satisfied = actual == args["seat"]
            observed = {"seat": actual}
        elif kind == "play.current_trick_suit_led":
            suit = (
                state.current_trick.led_suit
                if state.current_trick is not None
                else None
            )
            actual = suit.name.lower() if suit is not None else None
            satisfied = actual == args["suit"]
            observed = {"suit": actual}
        elif kind == "play.trump_use_count":
            tricks = [*state.completed_tricks]
            if state.current_trick is not None:
                tricks.append(state.current_trick)
            if args.get("trickIndex") is not None:
                requested = int(args["trickIndex"])
                tricks = [trick for trick in tricks if trick.index == requested]
                if not tricks:
                    return (
                        "not_evaluated",
                        {"trickIndex": requested},
                        "trick_unavailable",
                    )
            actual = sum(
                play.card.suit == state.trump_suit
                for trick in tricks
                for play in trick.plays
                if state.trump_suit is not None
            )
            satisfied = _in_range(actual, args)
            observed = {
                "trickIndex": args.get("trickIndex"),
                "trumpSuit": (
                    state.trump_suit.name.lower()
                    if state.trump_suit is not None
                    else None
                ),
                "count": actual,
            }
        elif kind == "play.current_trick_card":
            position = int(args["position"])
            play = (
                state.current_trick.plays[position]
                if state.current_trick is not None
                and position < len(state.current_trick.plays)
                else None
            )
            actual_card = play.card.id if play is not None else None
            actual_seat = play.seat.name.lower() if play is not None else None
            satisfied = (
                actual_card == Card.parse(args["card"]).id
                and actual_seat == args["seat"]
                and (
                    "ekka" not in args
                    or (play is not None and bool(play.ekka) == bool(args["ekka"]))
                )
            )
            observed = {
                "position": position,
                "seat": actual_seat,
                "card": actual_card,
                "ekka": bool(play.ekka) if play is not None else None,
            }
        elif kind == "play.completed_trick_winner":
            index = int(args["trickIndex"])
            trick = next(
                (item for item in state.completed_tricks if item.index == index), None
            )
            actual = (
                trick.winner.name.lower()
                if trick is not None and trick.winner is not None
                else None
            )
            satisfied = actual == args["seat"]
            observed = {"trickIndex": index, "seat": actual}
        elif kind == "state.legal_action_count":
            actual = len(normalized.legal_actions)
            satisfied = _in_range(actual, args)
            observed = {"count": actual}
        elif kind == "state.match_score":
            team = _team(args["team"])
            actual = state.match_score[int(team)]
            satisfied = _in_range(actual, args)
            observed = {"team": args["team"], "score": actual}
        elif kind == "state.phase":
            actual = state.phase.value
            satisfied = actual == args["state"]
            observed = {"state": actual}
        elif kind == "state.next_actor":
            actual = normalized.next_actor
            satisfied = actual == args["seat"]
            observed = {"seat": actual}
        elif kind == "state.observer_visibility":
            satisfied = True
            observed = {
                "observer": args["observer"],
                "classification": args["classification"],
            }
        elif kind == "state.observer_card_visibility":
            observer = Seat.parse(args["observer"])
            card = Card.parse(args["card"])
            visible_cards = {
                *state.hand(observer),
                *(played.card for played in state.played_cards),
            }
            if state.purchase_card is not None:
                visible_cards.add(state.purchase_card)
            for project in state.projects:
                if project.owner == observer or project.revealed:
                    visible_cards.update(project.cards)
            actual = "visible" if card in visible_cards else "hidden"
            satisfied = actual == args["visibility"]
            observed = {
                "observer": args["observer"],
                "card": card.id,
                "visibility": actual,
            }
        elif kind in {
            "state.current_raw_points",
            "state.current_project_points",
            "state.current_game_points",
        }:
            team = _team(args["team"])
            if kind == "state.current_raw_points":
                actual = sum(
                    trick.raw_points
                    for trick in state.completed_tricks
                    if trick.winner is not None and trick.winner.team == team
                )
            elif kind == "state.current_project_points":
                actual = sum(
                    project.value
                    for project in state.projects
                    if project.owner.team == team and project.scoring
                )
            elif state.round_score is not None:
                actual = state.round_score.card_game_points[int(team)]
            elif state.contract is None:
                return (
                    "not_evaluated",
                    {"reason": "contract_unavailable"},
                    "current_game_points_unavailable",
                )
            else:
                raw = sum(
                    trick.raw_points
                    for trick in state.completed_tricks
                    if trick.winner is not None and trick.winner.team == team
                )
                actual = ReferenceScoringInterpreterV1.from_rules(
                    normalized.rules.data
                ).convert(raw, state.contract)
            satisfied = _in_range(actual, args)
            observed = {"team": args["team"], "points": actual}
        elif kind == "state.legal_cards_include":
            if normalized.decision_kind != "play":
                return (
                    "not_evaluated",
                    {"decisionKind": normalized.decision_kind},
                    "play_legal_cards_unavailable",
                )
            actual = {
                int(item["card"]["id"] if "card" in item else item["id"])
                for item in normalized.legal_actions
            }
            expected = {Card.parse(item).id for item in args["cards"]}
            complete = bool(args.get("complete", False))
            satisfied = actual == expected if complete else expected.issubset(actual)
            observed = {
                "cardIds": sorted(actual),
                "requiredCardIds": sorted(expected),
                "complete": complete,
            }
        elif kind == "state.legal_play_actions_include":
            if normalized.decision_kind != "play":
                return (
                    "not_evaluated",
                    {"decisionKind": normalized.decision_kind},
                    "play_legal_actions_unavailable",
                )
            actual = {
                (
                    int(item["card"]["id"] if "card" in item else item["id"]),
                    bool(item.get("ekka", False)),
                )
                for item in normalized.legal_actions
            }
            expected = {
                (Card.parse(item["card"]).id, bool(item["ekka"]))
                for item in args["actions"]
            }
            complete = bool(args.get("complete", False))
            satisfied = actual == expected if complete else expected.issubset(actual)
            observed = {
                "actions": [
                    {"card": card_id, "ekka": ekka}
                    for card_id, ekka in sorted(
                        actual, key=lambda item: item[0] + 32 * item[1]
                    )
                ],
                "requiredActions": [
                    {"card": card_id, "ekka": ekka}
                    for card_id, ekka in sorted(
                        expected, key=lambda item: item[0] + 32 * item[1]
                    )
                ],
                "complete": complete,
            }
        elif kind.startswith("outcome."):
            outcome_state = state
            if (
                state.round_score is None
                and normalized.decision_kind == "play"
                and proof_outcome is not None
                and proof_outcome.payload.get("completionStatus") == "exact_complete"
            ):
                outcome_state = state.clone()
                engine = BalootEngine(normalized.rules)
                for code in proof_outcome.payload["principalVariation"]:
                    card, ekka = decode_proof_play_action_v1(code)
                    engine.play_card(
                        outcome_state,
                        card,
                        ekka=ekka,
                        emit_events=False,
                    )
            if outcome_state.round_score is None:
                return (
                    "not_evaluated",
                    {"reason": "round_not_terminal"},
                    "terminal_outcome_unavailable",
                )
            score = outcome_state.round_score
            if kind == "outcome.buyer_contract":
                assert outcome_state.declarer is not None
                won = score.winner == outcome_state.declarer.team
                satisfied = won == (args["status"] == "won")
                observed = {"status": "won" if won else "lost"}
            elif kind in {"outcome.team_round", "outcome.team_match"}:
                team = _team(args["team"])
                winner = (
                    score.winner
                    if kind == "outcome.team_round"
                    else outcome_state.match_winner
                )
                actual = (
                    "won"
                    if winner == team
                    else "lost"
                    if winner == team.other
                    else "tied"
                )
                satisfied = actual == args["status"]
                observed = {"team": args["team"], "status": actual}
            elif kind in {
                "outcome.raw_points",
                "outcome.game_points",
                "outcome.project_points",
            }:
                team = _team(args["team"])
                values = (
                    score.raw_card_points
                    if kind == "outcome.raw_points"
                    else score.project_points
                    if kind == "outcome.project_points"
                    else score.awarded_points
                )
                actual = values[int(team)]
                satisfied = _in_range(actual, args)
                observed = {"team": args["team"], "points": actual}
            elif kind == "outcome.round_margin":
                team = _team(args["team"])
                actual = (
                    score.awarded_points[int(team)]
                    - score.awarded_points[int(team.other)]
                )
                satisfied = _in_range(actual, args)
                observed = {"team": args["team"], "margin": actual}
            elif kind in {"outcome.kabout", "outcome.gahwa"}:
                team = _team(args["team"])
                occurred = bool(
                    score.kabout if kind == "outcome.kabout" else score.gahwa
                )
                actual = occurred and score.winner == team
                satisfied = actual == bool(args["achieved"])
                observed = {
                    "team": args["team"],
                    "achieved": actual,
                    "roundWinner": score.winner.name.lower(),
                }
            elif kind == "outcome.match_target_crossing":
                team = _team(args["team"])
                target = BalootEngine(normalized.rules).scoring_domain.match_target
                actual = (
                    outcome_state.match_winner == team
                    and outcome_state.match_score[int(team)] >= target
                )
                satisfied = actual == bool(args["achieved"])
                observed = {
                    "team": args["team"],
                    "achieved": actual,
                    "score": outcome_state.match_score[int(team)],
                    "target": target,
                }
            else:
                raise ValueError(f"runtime verifier missing for {kind}")
            if outcome_state is not state:
                observed.update(
                    {
                        "stateSource": "proof_principal_variation",
                        "proofHash": proof_outcome.sha256,
                    }
                )
        elif kind == "tactic.verified":
            rows = list((tactical_evidence or {}).get(str(args["kind"])) or ())
            matching = [
                row
                for row in rows
                if self._tactical_context_matches(row, args)
                and float(row.get("score", row.get("magnitude", 0)))
                >= float(args["minimumScore"])
                and (
                    not bool(args.get("causalRequired", False))
                    or bool(row.get("causalProofHash") or row.get("proofHash"))
                )
            ]
            satisfied = len(matching) >= int(args["minimumEvidence"])
            observed = {
                "kind": args["kind"],
                "matchingEvidenceCount": len(matching),
                "matchingEvidence": copy.deepcopy(matching),
            }
        elif kind == "tactic.applicable":
            rows = list((tactical_evidence or {}).get(str(args["kind"])) or ())
            actual = any(
                self._tactical_context_matches(row, args)
                and bool(row.get("applicable", True))
                for row in rows
            )
            satisfied = actual == bool(args["applicable"])
            observed = {
                "kind": args["kind"],
                "applicable": actual,
                "matchingEvidenceCount": sum(
                    self._tactical_context_matches(row, args) for row in rows
                ),
            }
        elif kind in {
            "proof.principal_variation_trick_winner",
            "proof.principal_variation_trick_team_rank_count",
        }:
            if proof_outcome is None:
                return (
                    "not_evaluated",
                    {"reason": "proof_unavailable"},
                    "proof_unavailable",
                )
            proof = proof_outcome.payload
            if proof["completionStatus"] != "exact_complete":
                return (
                    "not_evaluated",
                    {"completionStatus": proof["completionStatus"]},
                    "proof_incomplete",
                )
            clone = state.clone()
            engine = BalootEngine(normalized.rules)
            first_new_trick = len(clone.completed_tricks)
            for code in proof["principalVariation"]:
                card, ekka = decode_proof_play_action_v1(code)
                engine.play_card(clone, card, ekka=ekka, emit_events=False)
            requested_index = int(args["trickIndex"])
            trick = next(
                (
                    item
                    for item in clone.completed_tricks[first_new_trick:]
                    if item.index == requested_index
                ),
                None,
            )
            if trick is None or trick.winner is None:
                return (
                    "not_evaluated",
                    {"trickIndex": requested_index},
                    "principal_variation_trick_unavailable",
                )
            if kind == "proof.principal_variation_trick_winner":
                winning_play = next(
                    item for item in trick.plays if item.seat == trick.winner
                )
                actual_rank = winning_play.card.rank.name.lower()
                satisfied = trick.winner.team == _team(
                    args["team"]
                ) and actual_rank in set(args["rank"])
                observed = {
                    "trickIndex": trick.index,
                    "winner": trick.winner.name.lower(),
                    "team": trick.winner.team.name.lower(),
                    "card": winning_play.card.code,
                    "rank": actual_rank,
                    "proofHash": proof_outcome.sha256,
                }
            else:
                ranks = set(args["rank"])
                matching = [
                    item
                    for item in trick.plays
                    if item.seat.team == _team(args["team"])
                    and item.card.rank.name.lower() in ranks
                ]
                actual = len(matching)
                satisfied = _in_range(actual, args)
                observed = {
                    "trickIndex": trick.index,
                    "team": args["team"],
                    "ranks": sorted(ranks),
                    "count": actual,
                    "matchingCards": [
                        {
                            "seat": item.seat.name.lower(),
                            "card": item.card.code,
                            "rank": item.card.rank.name.lower(),
                        }
                        for item in matching
                    ],
                    "proofHash": proof_outcome.sha256,
                }
        elif kind in {
            "proof.winning_root_action_count",
            "proof.winning_root_card_rank_count",
            "proof.winning_continuation_line_count",
        }:
            if proof_outcome is None:
                return (
                    "not_evaluated",
                    {"reason": "proof_unavailable"},
                    "proof_unavailable",
                )
            payload = proof_outcome.payload
            resources = payload.get("resources")
            if kind in {
                "proof.winning_root_action_count",
                "proof.winning_root_card_rank_count",
            } and (
                payload["winningRootActionCount"] is None
                or (
                    isinstance(resources, Mapping)
                    and resources.get("rootActionCoverageComplete") is False
                )
            ):
                return (
                    "not_evaluated",
                    {"proofHash": proof_outcome.sha256},
                    "proof_incomplete",
                )
            if kind == "proof.winning_root_action_count":
                actual = payload["winningRootActionCount"]
                satisfied = _in_range(actual, args)
                observed = {"count": actual, "proofHash": proof_outcome.sha256}
            elif kind == "proof.winning_root_card_rank_count":
                ranks = set(args["rank"])
                winning = [item for item in payload["rootActions"] if item["satisfies"]]
                try:
                    matching = [
                        item
                        for item in winning
                        if Card.parse(item["action"]).rank.name.lower() in ranks
                    ]
                except ValueError:
                    return (
                        "not_evaluated",
                        {"proofHash": proof_outcome.sha256},
                        "winning_root_action_not_card_play",
                    )
                actual = len(matching)
                satisfied = _in_range(actual, args)
                observed = {
                    "ranks": sorted(ranks),
                    "count": actual,
                    "matchingActions": [item["action"] for item in matching],
                    "winningRootActionCount": len(winning),
                    "proofHash": proof_outcome.sha256,
                }
            else:
                lines = dict(payload["continuationLines"])
                if not lines["complete"]:
                    return (
                        "not_evaluated",
                        {"continuationLines": lines},
                        "line_count_incomplete",
                    )
                actual = int(lines["exactCount"])
                minimum = int(args.get("minimum", 0))
                maximum = int(args.get("maximum", actual))
                satisfied = (
                    minimum <= actual <= maximum
                    and lines["lineUniverse"] == args["lineUniverse"]
                )
                observed = {
                    "count": str(actual),
                    "lineUniverse": lines["lineUniverse"],
                    "proofHash": proof_outcome.sha256,
                }
        else:
            raise ValueError(f"runtime verifier missing for {kind}")
        return (
            "satisfied" if satisfied else "failed",
            observed,
            None if satisfied else "predicate_not_satisfied",
        )

    def verify(
        self,
        scenario: ResearchScenarioV1,
        normalized: NormalizedExactStateV1,
        *,
        proof_outcome: ProofOutcomeV1 | None = None,
        tactical_evidence: Mapping[str, Any] | None = None,
    ) -> ScenarioSatisfactionV1:
        evidence_by_id: dict[str, ScenarioPredicateEvidenceV1] = {}
        for predicate in scenario.predicates():
            definition = self.registry.require(predicate)
            status, observed, failure = self._evaluate_predicate(
                predicate,
                normalized,
                proof_outcome=proof_outcome,
                tactical_evidence=tactical_evidence,
            )
            core = {
                "predicateId": predicate.predicate_id,
                "predicateHash": predicate.sha256,
                "kind": predicate.kind,
                "hardness": predicate.hardness,
                "status": status,
                "constructionCapability": definition.construction_capability,
                "runtimeVerifier": definition.runtime_verifier,
                "observed": observed,
                "exactStateId": normalized.exact_state_id,
                "failureCode": failure,
            }
            evidence_by_id[predicate.predicate_id] = ScenarioPredicateEvidenceV1(
                predicate.predicate_id,
                predicate.sha256,
                predicate.kind,
                predicate.hardness,
                status,
                definition.construction_capability,
                definition.runtime_verifier,
                observed,
                normalized.exact_state_id,
                failure,
                canonical_sha256(core),
            )

        def expression_value(expression: ScenarioExpressionV1) -> bool:
            if expression.predicate is not None:
                return (
                    evidence_by_id[expression.predicate.predicate_id].status
                    == "satisfied"
                )
            values = [expression_value(child) for child in expression.children]
            if expression.op == "all":
                return all(values)
            if expression.op == "any":
                return any(values)
            if expression.op == "not":
                return not values[0]
            if expression.op == "count":
                count = sum(values)
                assert expression.minimum is not None and expression.maximum is not None
                return expression.minimum <= count <= expression.maximum
            raise AssertionError("unsupported normalized scenario expression")

        constraints = expression_value(scenario.constraints)
        outcome = expression_value(scenario.outcome)
        hard_ok = all(
            item.status == "satisfied"
            for item in evidence_by_id.values()
            if item.hardness == "hard"
        )
        status = (
            "satisfied"
            if constraints and outcome and hard_ok
            else "semantic_postcondition_miss"
        )
        rows = tuple(evidence_by_id[key] for key in sorted(evidence_by_id))
        core = {
            "schemaVersion": 1,
            "scenarioHash": scenario.sha256,
            "exactStateId": normalized.exact_state_id,
            "constraintsSatisfied": constraints,
            "outcomeSatisfied": outcome,
            "hardPredicatesSatisfied": hard_ok,
            "status": status,
            "predicateEvidenceHashes": [item.sha256 for item in rows],
        }
        return ScenarioSatisfactionV1(
            scenario.sha256,
            normalized.exact_state_id,
            constraints,
            outcome,
            hard_ok,
            status,
            rows,
            canonical_sha256(core),
        )


__all__ = [
    "ScenarioPredicateEvidenceV1",
    "ScenarioRuntimeVerifierV1",
    "ScenarioSatisfactionV1",
]
