"""Bounded, observer-safe Research V4 search candidate construction."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from baloot_platform.analysis.models import InformationState
from baloot_platform.core.bots import BotProfile, HeuristicBot
from baloot_platform.core.cards import Card, iter_mask
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract, Seat, Suit, Team
from baloot_platform.core.models import LegalPlayActionV1
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.scoring import _compiled_game_points

from .auction_compiler import OrToolsCpSatAuctionCompilerV1
from .escalation_compiler import OrToolsCpSatEscalationCompilerV1
from .exact_state import ExactStateNormalizerV1, NormalizedExactStateV1
from .proof_service import UnifiedScenarioProofServiceV1
from .recipe_v4 import ResearchRecipeV4
from .scenario_construction import (
    ScenarioStructuralCompilerV1,
    partial_trick_bridge_objective,
)
from .scenario_contracts import (
    ProofOutcomeV1,
    ResearchScenarioV1,
    ScenarioProofRequestV1,
    canonical_sha256,
)
from .scenario_evidence import ScenarioRuntimeVerifierV1, ScenarioSatisfactionV1


@dataclass(frozen=True, slots=True)
class ScenarioSearchCandidateV1:
    status: str
    seed: int
    normalized: NormalizedExactStateV1 | None
    proof: ProofOutcomeV1 | None
    satisfaction: ScenarioSatisfactionV1 | None
    construction_receipt: Mapping[str, Any]
    sha256: str


@dataclass(frozen=True, slots=True)
class ScenarioGuidancePlanV1:
    """Public scenario goals that may guide play without becoming evidence."""

    scenario_hash: str
    objectives: tuple[Mapping[str, Any], ...]
    sha256: str

    @classmethod
    def build(
        cls,
        scenario: ResearchScenarioV1,
        proof: ScenarioProofRequestV1,
    ) -> "ScenarioGuidancePlanV1":
        objectives: list[dict[str, Any]] = []
        target_leaders = {
            str(predicate.arguments["seat"])
            for predicate in scenario.predicates()
            if predicate.hardness == "hard"
            and predicate.kind == "play.current_trick_leader"
        }
        target_tricks = {
            int(predicate.arguments["minimum"])
            for predicate in scenario.predicates()
            if predicate.hardness == "hard"
            and predicate.kind == "play.trick_index"
            and predicate.arguments["minimum"] == predicate.arguments["maximum"]
        }
        for predicate in scenario.predicates():
            if predicate.hardness == "report":
                continue
            weight = 4 if predicate.hardness == "hard" else 2
            arguments = dict(predicate.arguments)
            if predicate.kind in {"outcome.team_round", "outcome.team_match"}:
                status = str(arguments["status"])
                if status == "tied":
                    continue
                team = Team[str(arguments["team"]).strip().upper()]
                desired = team if status == "won" else Team(1 - int(team))
                objectives.append(
                    {
                        "sourceId": predicate.predicate_id,
                        "sourceKind": predicate.kind,
                        "desiredTeam": desired.name.lower(),
                        "weight": weight,
                    }
                )
            elif predicate.kind == "outcome.buyer_contract":
                objectives.append(
                    {
                        "sourceId": predicate.predicate_id,
                        "sourceKind": predicate.kind,
                        "buyerStatus": str(arguments["status"]),
                        "weight": weight,
                    }
                )
            elif predicate.kind == "play.current_trick_suit_led":
                objective = {
                    "sourceId": predicate.predicate_id,
                    "sourceKind": predicate.kind,
                    "targetSuit": str(arguments["suit"]),
                    "weight": weight,
                }
                if len(target_leaders) == 1:
                    objective["actingSeat"] = next(iter(target_leaders))
                if len(target_tricks) == 1:
                    objective["targetTrickIndex"] = next(iter(target_tricks))
                objectives.append(objective)
            elif predicate.kind == "play.current_trick_card":
                objective = {
                    "sourceId": predicate.predicate_id,
                    "sourceKind": predicate.kind,
                    "actingSeat": str(arguments["seat"]),
                    "targetPosition": int(arguments["position"]),
                    "targetCard": (
                        Card(int(arguments["card"])).code
                        if isinstance(arguments["card"], int)
                        else Card.parse(str(arguments["card"])).code
                    ),
                    "weight": weight,
                }
                if len(target_tricks) == 1:
                    objective["targetTrickIndex"] = next(iter(target_tricks))
                objectives.append(objective)
            elif (
                predicate.kind == "state.current_raw_points" and len(target_tricks) == 1
            ):
                objectives.append(
                    {
                        "sourceId": predicate.predicate_id,
                        "sourceKind": "actor_visible_raw_points_at_decision",
                        "scoreTeam": str(arguments["team"]),
                        "minimum": int(arguments["minimum"]),
                        "maximum": int(arguments["maximum"]),
                        "targetTrickIndex": next(iter(target_tricks)),
                        "weight": weight,
                    }
                )
            elif predicate.kind == "deal.remaining_card_owner":
                objectives.append(
                    {
                        "sourceId": predicate.predicate_id,
                        "sourceKind": "actor_visible_card_preservation",
                        "actingSeat": str(arguments["seat"]),
                        "protectedCards": [
                            (
                                Card(int(arguments["card"])).code
                                if isinstance(arguments["card"], int)
                                else Card.parse(str(arguments["card"])).code
                            )
                        ],
                        "weight": weight,
                    }
                )
            elif predicate.kind == "proof.principal_variation_trick_winner":
                objectives.append(
                    {
                        "sourceId": predicate.predicate_id,
                        "sourceKind": predicate.kind,
                        "desiredTeam": str(arguments["team"]),
                        "targetTrickIndex": int(arguments["trickIndex"]),
                        "winningRanks": sorted(str(rank) for rank in arguments["rank"]),
                        "weight": weight,
                    }
                )
        decision_seats = {
            str(predicate.arguments["seat"])
            for predicate in scenario.predicates()
            if predicate.hardness == "hard" and predicate.kind == "play.current_player"
        }
        selector = scenario.decision_point["selector"]
        if scenario.decision_point["kind"] == "play" and selector.get("actingSeat"):
            decision_seats.add(str(selector["actingSeat"]))
        if len(decision_seats) == 1 and len(target_tricks) == 1:
            for predicate in scenario.predicates():
                if (
                    predicate.hardness != "report"
                    and predicate.kind == "proof.winning_root_card_rank_count"
                ):
                    objectives.append(
                        {
                            "sourceId": predicate.predicate_id,
                            "sourceKind": "actor_visible_rank_preservation",
                            "actingSeat": next(iter(decision_seats)),
                            "protectedRanks": sorted(
                                str(rank) for rank in predicate.arguments["rank"]
                            ),
                            "targetTrickIndex": next(iter(target_tricks)),
                            "weight": 4 if predicate.hardness == "hard" else 2,
                        }
                    )
            if "remainingCards" in selector:
                bridge_objective = partial_trick_bridge_objective(scenario)
                if bridge_objective is not None:
                    target_leader, target_trick, bridge_suit, turnover_suit = (
                        bridge_objective
                    )
                    objectives.append(
                        {
                            "sourceId": "DECISION-POSITION",
                            "sourceKind": "target_partial_trick_leader",
                            "targetLeader": target_leader.name.lower(),
                            "targetTrickIndex": target_trick,
                            **(
                                {"bridgeSuit": bridge_suit.name.lower()}
                                if bridge_suit is not None
                                else {}
                            ),
                            **(
                                {"turnoverSuit": turnover_suit.name.lower()}
                                if turnover_suit is not None
                                else {}
                            ),
                            "weight": 4,
                        }
                    )
        decision_cards = {
            Card(int(card)).code
            if isinstance(card, int)
            else Card.parse(str(card)).code
            for predicate in scenario.predicates()
            if predicate.hardness == "hard"
            and predicate.kind == "state.legal_cards_include"
            for card in predicate.arguments["cards"]
        }
        decision_cards.update(
            Card.parse(action["card"]).code
            for predicate in scenario.predicates()
            if predicate.hardness == "hard"
            and predicate.kind == "state.legal_play_actions_include"
            for action in predicate.arguments["actions"]
        )
        if proof.selected_action is not None:
            decision_cards.add(
                LegalPlayActionV1.from_code(proof.selected_action).card.code
            )
        if len(decision_seats) == 1 and decision_cards:
            objectives.append(
                {
                    "sourceId": "DECISION-CARDS",
                    "sourceKind": "actor_visible_card_preservation",
                    "actingSeat": next(iter(decision_seats)),
                    "protectedCards": sorted(decision_cards),
                    "weight": 1,
                }
            )
        proof_target = Team[str(proof.target_team).strip().upper()]
        proof_desired = (
            Team(1 - int(proof_target))
            if proof.claim == "target_team_has_no_salvage"
            else proof_target
        )
        objectives.append(
            {
                "sourceId": "PROOF-REQUEST",
                "sourceKind": proof.claim,
                "desiredTeam": proof_desired.name.lower(),
                "weight": 1,
            }
        )
        unique_root_requested = any(
            predicate.hardness == "hard"
            and predicate.kind == "proof.winning_root_action_count"
            and int(predicate.arguments["minimum"]) == 1
            and int(predicate.arguments["maximum"]) == 1
            for predicate in scenario.predicates()
        )
        if (
            unique_root_requested
            and proof.utility == "round_win"
            and len(target_tricks) == 1
        ):
            objectives.append(
                {
                    "sourceId": "PROOF-SCORE-TRAJECTORY",
                    "sourceKind": "actor_visible_round_score_trajectory",
                    "desiredTeam": proof_desired.name.lower(),
                    "targetTrickIndex": next(iter(target_tricks)),
                    "weight": 4,
                }
            )
        ordered = tuple(
            sorted(
                objectives,
                key=lambda item: (
                    str(item["sourceId"]),
                    str(item["sourceKind"]),
                ),
            )
        )
        core = {
            "schemaVersion": 1,
            "scenarioHash": scenario.sha256,
            "objectives": [dict(item) for item in ordered],
        }
        return cls(scenario.sha256, ordered, canonical_sha256(core))

    def desired_team(self, information: InformationState) -> Team:
        votes = {Team.NORTH_SOUTH: 0, Team.WEST_EAST: 0}
        for objective in self.objectives:
            desired = objective.get("desiredTeam")
            if desired is None:
                if "buyerStatus" not in objective:
                    continue
                buyer_team = information.buyer.team
                desired_team = (
                    buyer_team
                    if objective["buyerStatus"] == "won"
                    else Team(1 - int(buyer_team))
                )
            else:
                desired_team = Team[str(desired).strip().upper()]
            votes[desired_team] += int(objective["weight"])
        return max(votes, key=lambda team: (votes[team], -int(team)))


class ScenarioGuidedActorSafePolicyV1:
    """Select one legal action from actor-visible information only."""

    policy_id = "scenario_guided_actor_safe_v13"
    version = "13.0.0"

    def __init__(self, guidance: ScenarioGuidancePlanV1):
        self.guidance = guidance
        self.scenario_hash = guidance.scenario_hash

    @staticmethod
    def _current_trick(information: InformationState) -> tuple[Any, ...]:
        if not information.plays or len(information.plays) % 4 == 0:
            return ()
        trick_index = information.plays[-1].trick_index
        return tuple(
            play for play in information.plays if play.trick_index == trick_index
        )

    @staticmethod
    def _winner_after_action(
        information: InformationState,
        card: Card,
        engine: BalootEngine,
    ) -> Seat:
        plays = [
            (play.seat, play.card)
            for play in ScenarioGuidedActorSafePolicyV1._current_trick(information)
        ]
        plays.append((information.observer, card))
        return ScenarioGuidedActorSafePolicyV1._winner_for_plays(
            information,
            plays,
            engine,
        )

    @staticmethod
    def _winner_for_plays(
        information: InformationState,
        plays: Sequence[tuple[Seat, Card]],
        engine: BalootEngine,
    ) -> Seat:
        led = plays[0][1].suit
        trump = (
            information.trump_suit if information.contract == Contract.HOKUM else None
        )
        eligible = plays
        if trump is not None and any(candidate.suit == trump for _, candidate in plays):
            eligible = [
                (seat, candidate)
                for seat, candidate in plays
                if candidate.suit == trump
            ]
        elif any(candidate.suit == led for _, candidate in plays):
            eligible = [
                (seat, candidate) for seat, candidate in plays if candidate.suit == led
            ]
        winner, _ = max(
            eligible,
            key=lambda item: engine.rules.strength(
                item[1], information.contract, trump
            ),
        )
        return winner

    @staticmethod
    def _public_raw_points(
        information: InformationState,
        engine: BalootEngine,
    ) -> list[int]:
        raw = [0, 0]
        trick_indexes = sorted({play.trick_index for play in information.plays})
        for trick_index in trick_indexes:
            plays = tuple(
                play for play in information.plays if play.trick_index == trick_index
            )
            if len(plays) != 4:
                continue
            winner = ScenarioGuidedActorSafePolicyV1._winner_for_plays(
                information,
                [(play.seat, play.card) for play in plays],
                engine,
            )
            trump = (
                information.trump_suit
                if information.contract == Contract.HOKUM
                else None
            )
            raw[int(winner.team)] += sum(
                int(engine.rules.points(play.card, information.contract, trump))
                for play in plays
            )
        return raw

    @staticmethod
    def _strict_round_win_raw_target(
        information: InformationState,
        desired: Team,
        engine: BalootEngine,
    ) -> int:
        contract = information.contract.effective.value
        card_total = int(engine.scoring_domain.card_game_point_totals[contract])
        raw_total = int(engine.scoring_domain.raw_card_totals[contract])
        projects = [0, 0]
        for project in information.projects:
            if project.scoring:
                projects[int(project.owner.team)] += int(project.value)
        baloot = [0, 0]
        for team in information.baloot_teams:
            baloot[int(team)] += int(engine.project_domain.baloot_value)
        for raw in range(raw_total + 1):
            game_points = _compiled_game_points(
                raw,
                information.contract,
                engine.scoring_domain,
            )
            desired_total = game_points + projects[int(desired)] + baloot[int(desired)]
            other_total = (
                card_total
                - game_points
                + projects[int(desired.other)]
                + baloot[int(desired.other)]
            )
            if desired_total > other_total:
                return raw
        return raw_total

    def _score(
        self,
        information: InformationState,
        card: Card,
        engine: BalootEngine,
    ) -> dict[str, int]:
        desired = self.guidance.desired_team(information)
        winning_seat = self._winner_after_action(information, card, engine)
        winning_team = winning_seat.team
        winner_alignment = 1 if winning_team == desired else -1
        actor_alignment = 1 if information.observer.team == desired else -1
        trump = (
            information.trump_suit if information.contract == Contract.HOKUM else None
        )
        strength = int(engine.rules.strength(card, information.contract, trump))
        points = int(engine.rules.points(card, information.contract, trump))
        current_trick = self._current_trick(information)
        trick_index = (
            current_trick[0].trick_index
            if current_trick
            else len(information.plays) // 4
        )
        leader_objectives = [
            objective
            for objective in self.guidance.objectives
            if "targetLeader" in objective
        ]
        protected_codes = {
            code
            for objective in self.guidance.objectives
            if objective.get("actingSeat") == information.observer.name.lower()
            for code in objective.get("protectedCards", ())
        }
        bridge_preservation = 0
        bridge_setup_strength = 0
        bridge_lead_alignment = 0
        turnover_lead_alignment = 0
        for objective in leader_objectives:
            target_trick = int(objective["targetTrickIndex"])
            target_leader = str(objective["targetLeader"])
            bridge_suit = objective.get("bridgeSuit")
            turnover_suit = objective.get("turnoverSuit")
            weight = int(objective["weight"])
            if bridge_suit is not None and trick_index < target_trick - 1:
                bridge_cards = [
                    candidate
                    for candidate in iter_mask(information.observer_hand)
                    if candidate.code not in protected_codes
                    and candidate.suit.name.lower() == bridge_suit
                ]
                if bridge_cards:
                    ordered_bridge_cards = sorted(
                        bridge_cards,
                        key=lambda candidate: (
                            int(
                                engine.rules.strength(
                                    candidate,
                                    information.contract,
                                    trump,
                                )
                            ),
                            int(
                                engine.rules.points(
                                    candidate,
                                    information.contract,
                                    trump,
                                )
                            ),
                            candidate.id,
                        ),
                    )
                    bridge = ordered_bridge_cards[
                        -1 if information.observer.name.lower() == target_leader else 0
                    ]
                    if card == bridge:
                        bridge_preservation = min(bridge_preservation, -weight)
            if (
                bridge_suit is not None
                and trick_index == target_trick - 1
                and not current_trick
                and card.suit.name.lower() == bridge_suit
            ):
                bridge_lead_alignment = max(bridge_lead_alignment, weight)
            if (
                turnover_suit is not None
                and trick_index == 0
                and not current_trick
                and card.suit.name.lower() == turnover_suit
            ):
                turnover_lead_alignment = max(turnover_lead_alignment, weight)
            if trick_index == target_trick - 1 and not any(
                play.seat.name.lower() == target_leader for play in current_trick
            ):
                direction = (
                    1 if information.observer.name.lower() == target_leader else -1
                )
                bridge_setup_strength += direction * strength * weight
        lead_suit_alignment = max(
            (
                int(
                    not current_trick
                    and objective.get("targetSuit") == card.suit.name.lower()
                    and objective.get("actingSeat", information.observer.name.lower())
                    == information.observer.name.lower()
                    and int(objective.get("targetTrickIndex", trick_index))
                    == trick_index
                )
                * int(objective["weight"])
                for objective in self.guidance.objectives
                if "targetSuit" in objective
            ),
            default=0,
        )
        target_lead_preservation = max(
            (
                int(
                    objective.get("actingSeat") == information.observer.name.lower()
                    and trick_index
                    < int(objective.get("targetTrickIndex", trick_index))
                    and objective.get("targetSuit") == card.suit.name.lower()
                )
                * int(objective["weight"])
                for objective in self.guidance.objectives
                if "targetSuit" in objective
            ),
            default=0,
        )
        proof_rank_alignment = max(
            (
                int(
                    objective.get("desiredTeam") == winning_team.name.lower()
                    and int(objective.get("targetTrickIndex", -1)) == trick_index
                    and card.rank.name.lower() in objective.get("winningRanks", ())
                )
                * int(objective["weight"])
                for objective in self.guidance.objectives
                if "winningRanks" in objective
            ),
            default=0,
        )
        next_trick_leader_alignment = max(
            (
                int(
                    trick_index == int(objective.get("targetTrickIndex", -1)) - 1
                    and objective.get("targetLeader") == winning_seat.name.lower()
                )
                * int(objective["weight"])
                for objective in self.guidance.objectives
                if "targetLeader" in objective
            ),
            default=0,
        )
        target_card_alignment = max(
            (
                int(
                    objective.get("actingSeat") == information.observer.name.lower()
                    and int(objective.get("targetTrickIndex", trick_index))
                    == trick_index
                    and int(objective.get("targetPosition", len(current_trick)))
                    == len(current_trick)
                    and objective.get("targetCard") == card.code
                )
                * int(objective["weight"])
                for objective in self.guidance.objectives
                if "targetCard" in objective
            ),
            default=0,
        )
        target_card_preservation = max(
            (
                int(
                    objective.get("actingSeat") == information.observer.name.lower()
                    and trick_index
                    < int(objective.get("targetTrickIndex", trick_index))
                    and objective.get("targetCard") == card.code
                )
                * int(objective["weight"])
                for objective in self.guidance.objectives
                if "targetCard" in objective
            ),
            default=0,
        )
        score_trajectory = 0
        trajectory_weight = max(
            (
                int(objective["weight"])
                for objective in self.guidance.objectives
                if objective.get("sourceKind") == "actor_visible_round_score_trajectory"
                and objective.get("desiredTeam") == desired.name.lower()
                and trick_index < int(objective["targetTrickIndex"])
            ),
            default=0,
        )
        if trajectory_weight and len(current_trick) == 3:
            projected_raw = self._public_raw_points(information, engine)
            projected_raw[int(winning_team)] += (
                sum(
                    int(engine.rules.points(play.card, information.contract, trump))
                    for play in current_trick
                )
                + points
            )
            completed_tricks = trick_index + 1
            final_target = self._strict_round_win_raw_target(
                information,
                desired,
                engine,
            )
            trajectory_target = (final_target * completed_tricks + 4) // 8
            score_trajectory = (
                -abs(projected_raw[int(desired)] - trajectory_target)
                * trajectory_weight
            )
        decision_raw_trajectory = 0
        raw_objectives = [
            objective
            for objective in self.guidance.objectives
            if objective.get("sourceKind") == "actor_visible_raw_points_at_decision"
            and trick_index < int(objective["targetTrickIndex"])
        ]
        if raw_objectives and len(current_trick) == 3:
            projected_raw = self._public_raw_points(information, engine)
            projected_raw[int(winning_team)] += (
                sum(
                    int(engine.rules.points(play.card, information.contract, trump))
                    for play in current_trick
                )
                + points
            )
            for objective in raw_objectives:
                target_trick = int(objective["targetTrickIndex"])
                if target_trick <= 0:
                    continue
                target = (int(objective["minimum"]) + int(objective["maximum"])) // 2
                completed_tricks = trick_index + 1
                trajectory_target = (
                    target * completed_tricks + target_trick // 2
                ) // target_trick
                team = Team[str(objective["scoreTeam"]).strip().upper()]
                decision_raw_trajectory -= abs(
                    projected_raw[int(team)] - trajectory_target
                ) * int(objective["weight"])
        protected_card = card.code in protected_codes and target_card_alignment == 0
        protected_rank = any(
            objective.get("actingSeat") == information.observer.name.lower()
            and trick_index < int(objective.get("targetTrickIndex", trick_index))
            and card.rank.name.lower() in objective.get("protectedRanks", ())
            for objective in self.guidance.objectives
        )
        return {
            "decisionCardPreservation": -1 if protected_card else 0,
            "decisionRankPreservation": -1 if protected_rank else 0,
            "targetLeaderBridgePreservation": bridge_preservation,
            "targetLeaderBridgeLead": bridge_lead_alignment,
            "targetLeaderTurnoverLead": turnover_lead_alignment,
            "targetLeaderSetupStrength": bridge_setup_strength,
            "leadSuitAlignment": lead_suit_alignment,
            "targetLeadPreservation": -target_lead_preservation,
            "targetCardPreservation": -target_card_preservation,
            "targetCardAlignment": target_card_alignment,
            "proofRankAlignment": proof_rank_alignment,
            "nextTrickLeaderAlignment": next_trick_leader_alignment,
            "decisiveScoreTrajectory": score_trajectory,
            "decisionRawTrajectory": decision_raw_trajectory,
            "winnerAlignment": winner_alignment,
            "pointContribution": points * winner_alignment,
            "strengthDirection": strength * actor_alignment,
            "total": (
                (-1_000_000_000 if protected_card or protected_rank else 0)
                + bridge_preservation * 250_000_000
                + bridge_lead_alignment * 250_000_000
                + turnover_lead_alignment * 250_000_000
                + bridge_setup_strength * 10_000_000
                - target_card_preservation * 1_000_000_000
                + target_card_alignment * 1_000_000_000
                - target_lead_preservation * 100_000_000
                + next_trick_leader_alignment * 100_000_000
                + lead_suit_alignment * 10_000_000
                + proof_rank_alignment * 1_000_000
                + decision_raw_trajectory * 200_000
                + score_trajectory * 100_000
                + winner_alignment * 100_000
                + points * winner_alignment * 100
                + strength * actor_alignment
            ),
        }

    def choose(self, state, engine: BalootEngine) -> Card:
        actor = state.current_player
        if actor is None:
            raise ValueError("scenario-guided policy requires a current actor")
        information = InformationState.from_complete(state, actor)
        information_hash = canonical_sha256(information.to_dict())
        legal = engine.legal_actions(state)
        if not legal:
            raise ValueError("scenario-guided policy has no legal action")
        return max(
            legal,
            key=lambda card: (
                self._score(information, card, engine)["total"],
                canonical_sha256(
                    {
                        "policyId": self.policy_id,
                        "version": self.version,
                        "scenarioHash": self.scenario_hash,
                        "informationStateHash": information_hash,
                        "action": card.code,
                    }
                ),
            ),
        )

    def choose_with_receipt(
        self, state, engine: BalootEngine
    ) -> tuple[Card, dict[str, Any]]:
        actor = state.current_player
        if actor is None:
            raise ValueError("scenario-guided policy requires a current actor")
        information = InformationState.from_complete(state, actor)
        information_hash = canonical_sha256(information.to_dict())
        legal = sorted(engine.legal_actions(state), key=lambda card: card.id)
        components = {
            card.code: self._score(information, card, engine) for card in legal
        }
        tie_breakers = {
            card.code: canonical_sha256(
                {
                    "policyId": self.policy_id,
                    "version": self.version,
                    "scenarioHash": self.scenario_hash,
                    "informationStateHash": information_hash,
                    "action": card.code,
                }
            )
            for card in legal
        }
        selected = max(
            legal,
            key=lambda card: (
                components[card.code]["total"],
                tie_breakers[card.code],
            ),
        )
        desired_team = self.guidance.desired_team(information)
        core = {
            "schemaVersion": 1,
            "policyId": self.policy_id,
            "policyVersion": self.version,
            "actingSeat": actor.name.lower(),
            "informationStateHash": information_hash,
            "guidancePlanHash": self.guidance.sha256,
            "desiredTeam": desired_team.name.lower(),
            "candidateActions": [card.code for card in legal],
            "actionScores": components,
            "tieBreakers": tie_breakers,
            "selectedAction": selected.code,
            "observerSafe": True,
        }
        return selected, {**core, "sha256": canonical_sha256(core)}


class ScenarioSearchExecutorV1:
    """Construct one legal candidate, then verify semantics and proof independently."""

    def __init__(
        self,
        *,
        structural: ScenarioStructuralCompilerV1 | None = None,
        auction: OrToolsCpSatAuctionCompilerV1 | None = None,
        escalation: OrToolsCpSatEscalationCompilerV1 | None = None,
        runtime: ScenarioRuntimeVerifierV1 | None = None,
        proof: UnifiedScenarioProofServiceV1 | None = None,
    ):
        self.structural = structural or ScenarioStructuralCompilerV1()
        self.auction = auction or OrToolsCpSatAuctionCompilerV1()
        self.escalation = escalation or OrToolsCpSatEscalationCompilerV1()
        self.runtime = runtime or ScenarioRuntimeVerifierV1()
        self.proof = proof or UnifiedScenarioProofServiceV1.from_default_authority()

    @staticmethod
    def _hard(recipe: ResearchRecipeV4, kind: str):
        return [
            predicate
            for predicate in recipe.scenario.predicates()
            if predicate.kind == kind and predicate.hardness == "hard"
        ]

    def _dealer(self, recipe: ResearchRecipeV4) -> Seat:
        rows = self._hard(recipe, "auction.dealer")
        return Seat.parse(rows[0].arguments["seat"]) if rows else Seat.EAST

    def _match_score(self, recipe: ResearchRecipeV4) -> list[int]:
        score = [0, 0]
        for predicate in self._hard(recipe, "state.match_score"):
            arguments = dict(predicate.arguments)
            if int(arguments["minimum"]) != int(arguments["maximum"]):
                continue
            team = (
                Team.NORTH_SOUTH
                if arguments["team"] == "north_south"
                else Team.WEST_EAST
            )
            score[int(team)] = int(arguments["minimum"])
        return score

    def _compile_auction(
        self,
        recipe: ResearchRecipeV4,
        *,
        rules: RuleBook,
        dealer: Seat,
        purchase_suit: Suit,
    ):
        buyer_rows = self._hard(recipe, "auction.buyer")
        contract_rows = self._hard(recipe, "auction.contract")
        round_rows = self._hard(recipe, "auction.purchase_round")
        trump_rows = self._hard(recipe, "auction.trump")
        action_rows = self._hard(recipe, "auction.action_sequence")
        required_actions = None
        required_actions_complete = True
        if action_rows:
            actions = list(action_rows[0].arguments["actions"])
            required_actions = actions
            required_actions_complete = bool(action_rows[0].arguments["complete"])
        return self.auction.compile(
            dealer=dealer,
            purchase_suit=purchase_suit,
            rules=rules,
            required_buyer=(
                Seat.parse(buyer_rows[0].arguments["seat"]) if buyer_rows else None
            ),
            required_contracts=(
                [Contract.parse(contract_rows[0].arguments["contract"])]
                if contract_rows
                else []
            ),
            required_purchase_round=(
                int(round_rows[0].arguments["round"]) if round_rows else None
            ),
            required_trump=(
                Suit.parse(trump_rows[0].arguments["suit"]) if trump_rows else None
            ),
            required_actions=required_actions,
            required_actions_complete=required_actions_complete,
            maximum_conflicts=int(recipe.resources["maximumConflicts"]),
            timeout_seconds=float(recipe.resources["solverTimeoutSeconds"]),
        )

    @staticmethod
    def _auction_actions(plan) -> list[dict[str, Any]]:
        return [
            {
                "actor": item.actor.name.lower(),
                "action": item.action,
                **(
                    {"trumpSuit": item.trump.name.lower()}
                    if item.trump is not None
                    else {}
                ),
            }
            for item in plan.actions
        ]

    @staticmethod
    def _escalation_actions(plan) -> list[dict[str, Any]]:
        return [
            {
                "actor": item.actor.name.lower(),
                "action": item.action,
                **(
                    {"requestedMode": item.requested_mode}
                    if item.requested_mode is not None
                    else {}
                ),
            }
            for item in plan.actions
        ]

    def _finish_candidate(
        self,
        recipe: ResearchRecipeV4,
        *,
        normalized: NormalizedExactStateV1,
        bundle_hash: str,
        seed: int,
        receipt: dict[str, Any],
        auction: Any,
        cancel_check: Any | None,
        decision_receipts: Sequence[Mapping[str, Any]] = (),
        policy: Mapping[str, Any],
    ) -> ScenarioSearchCandidateV1:
        selected_action = recipe.proof.selected_action
        legal_actions = {str(item["code"]) for item in normalized.legal_actions}
        if selected_action is not None and selected_action not in legal_actions:
            satisfaction = self.runtime.verify(
                recipe.scenario,
                normalized,
                proof_outcome=None,
            )
            receipt.update(
                {
                    "status": "semantic_postcondition_miss",
                    "failure": "proof_selected_action_not_legal",
                    "auction": auction.to_dict(),
                    "decisionKind": normalized.decision_kind,
                    "decisionReceipts": [
                        copy.deepcopy(dict(item)) for item in decision_receipts
                    ],
                    "normalizedExactStateHash": normalized.sha256,
                    "scenarioSatisfactionHash": satisfaction.sha256,
                    "policy": copy.deepcopy(dict(policy)),
                }
            )
            receipt["sha256"] = canonical_sha256(receipt)
            return ScenarioSearchCandidateV1(
                "semantic_postcondition_miss",
                seed,
                normalized,
                None,
                satisfaction,
                copy.deepcopy(receipt),
                canonical_sha256(receipt),
            )
        proof = self.proof.prove(
            normalized,
            recipe.proof,
            bundle_hash=bundle_hash,
            cancel_check=cancel_check,
        )
        satisfaction = self.runtime.verify(
            recipe.scenario,
            normalized,
            proof_outcome=proof,
        )
        status = satisfaction.status
        if proof.payload["completionStatus"] == "proof_incomplete":
            status = "proof_incomplete"
        elif proof.payload["completionStatus"] == "proof_failed":
            status = "proof_failed"
        receipt.update(
            {
                "status": status,
                "auction": auction.to_dict(),
                "decisionKind": normalized.decision_kind,
                "decisionReceipts": [
                    copy.deepcopy(dict(item)) for item in decision_receipts
                ],
                "normalizedExactStateHash": normalized.sha256,
                "proofOutcomeHash": str(proof.payload["semanticSha256"]),
                "scenarioSatisfactionHash": satisfaction.sha256,
                "policy": copy.deepcopy(dict(policy)),
            }
        )
        receipt["sha256"] = canonical_sha256(receipt)
        return ScenarioSearchCandidateV1(
            status,
            seed,
            normalized,
            proof,
            satisfaction,
            copy.deepcopy(receipt),
            canonical_sha256(receipt),
        )

    def build_candidate(
        self,
        recipe: ResearchRecipeV4,
        *,
        rules: RuleBook,
        bundle_hash: str,
        seed: int,
        cancel_check: Any | None = None,
        excluded_deck_orders: Sequence[Sequence[int]] = (),
    ) -> ScenarioSearchCandidateV1:
        if recipe.scenario.origin.kind != "search":
            raise ValueError("scenario search executor accepts only search origins")
        dealer = self._dealer(recipe)
        structural = self.structural.solve(
            recipe.scenario,
            rules=rules,
            dealer=dealer,
            seed=seed,
            maximum_seconds=float(recipe.resources["solverTimeoutSeconds"]),
            excluded_deck_orders=excluded_deck_orders,
        )
        receipt: dict[str, Any] = {
            "schemaVersion": 1,
            "seed": int(seed),
            "generationPolicy": str(recipe.resources["generationPolicy"]),
            "auctionFeasible": False,
            "structural": structural.to_dict(),
        }
        if structural.status != "satisfied" or structural.deck_order is None:
            receipt["status"] = structural.status
            receipt["sha256"] = canonical_sha256(receipt)
            return ScenarioSearchCandidateV1(
                structural.status,
                seed,
                None,
                None,
                None,
                receipt,
                canonical_sha256(receipt),
            )
        engine = BalootEngine(rules)
        initial = engine.create_round(
            seed=seed,
            dealer=dealer,
            match_score=self._match_score(recipe),
        )
        engine.initial_deal_from_preordered(initial, structural.deck_order)
        if initial.purchase_card is None:
            raise AssertionError("scenario structural result lacks a purchase card")
        auction = self._compile_auction(
            recipe,
            rules=rules,
            dealer=dealer,
            purchase_suit=initial.purchase_card.suit,
        )
        if auction is None:
            receipt.update({"status": "construction_target_miss", "auction": None})
            receipt["sha256"] = canonical_sha256(receipt)
            return ScenarioSearchCandidateV1(
                "construction_target_miss",
                seed,
                None,
                None,
                None,
                receipt,
                canonical_sha256(receipt),
            )
        receipt["auctionFeasible"] = True
        auction_actions = self._auction_actions(auction)
        prefix: dict[str, Any] = {
            "schemaVersion": 1,
            "deckOrder": list(structural.deck_order),
            "dealer": dealer.name.lower(),
            "roundNumber": 1,
            "matchScore": self._match_score(recipe),
            "auctionActions": auction_actions,
            "escalationActions": [],
            "projectAssertions": [],
            "playActions": [],
            "stateAssertions": {},
        }
        decision_kind = str(recipe.scenario.decision_point["kind"])
        selector = dict(recipe.scenario.decision_point["selector"])
        compiled_policy = {
            "id": "scenario_compiled_public_prefix_v1",
            "version": "1.0.0",
            "observerSafe": True,
        }
        normalizer = ExactStateNormalizerV1(rules)
        if decision_kind == "bid":
            action_index = int(selector.get("actionIndex", 0))
            if action_index < 0 or action_index >= len(auction_actions):
                receipt.update(
                    {
                        "status": "construction_target_miss",
                        "failure": "bid_decision_action_index_unavailable",
                    }
                )
                receipt["sha256"] = canonical_sha256(receipt)
                return ScenarioSearchCandidateV1(
                    "construction_target_miss",
                    seed,
                    None,
                    None,
                    None,
                    receipt,
                    canonical_sha256(receipt),
                )
            prefix["auctionActions"] = auction_actions[:action_index]
            normalized = normalizer.normalize(
                prefix,
                decision_point=recipe.scenario.decision_point,
            )
            return self._finish_candidate(
                recipe,
                normalized=normalized,
                bundle_hash=bundle_hash,
                seed=seed,
                receipt=receipt,
                auction=auction,
                cancel_check=cancel_check,
                policy=compiled_policy,
            )

        if decision_kind == "escalation":
            complete = normalizer.normalize(prefix)
            level_rows = self._hard(recipe, "escalation.level")
            locked_rows = self._hard(recipe, "escalation.locked")
            action_rows = self._hard(recipe, "escalation.action_sequence")
            required_level = (
                str(level_rows[0].arguments["level"]) if level_rows else None
            )
            required_mode = None
            if locked_rows and required_level in {"double", "four"}:
                required_mode = (
                    "locked" if bool(locked_rows[0].arguments["locked"]) else "open"
                )
            required_actions = None
            required_actions_complete = True
            if action_rows:
                arguments = action_rows[0].arguments
                required_actions = tuple(
                    (
                        item["actor"],
                        item["action"],
                        item.get("mode"),
                    )
                    for item in arguments["actions"]
                )
                required_actions_complete = bool(arguments.get("complete", True))
            escalation = self.escalation.compile(
                dealer=dealer,
                declarer=complete.round_state.declarer,
                contract=complete.round_state.contract,
                match_score=tuple(complete.round_state.match_score),
                rules=rules,
                required_level=required_level,
                required_mode=required_mode,
                required_actions=required_actions,
                required_actions_complete=required_actions_complete,
                maximum_conflicts=int(recipe.resources["maximumConflicts"]),
                timeout_seconds=float(recipe.resources["solverTimeoutSeconds"]),
            )
            if escalation is None:
                receipt.update(
                    {"status": "construction_target_miss", "escalation": None}
                )
                receipt["sha256"] = canonical_sha256(receipt)
                return ScenarioSearchCandidateV1(
                    "construction_target_miss",
                    seed,
                    None,
                    None,
                    None,
                    receipt,
                    canonical_sha256(receipt),
                )
            escalation_actions = self._escalation_actions(escalation)
            action_index = int(selector.get("actionIndex", 0))
            if action_index < 0 or action_index > len(escalation_actions):
                receipt.update(
                    {
                        "status": "construction_target_miss",
                        "failure": "escalation_decision_action_index_unavailable",
                    }
                )
                receipt["sha256"] = canonical_sha256(receipt)
                return ScenarioSearchCandidateV1(
                    "construction_target_miss",
                    seed,
                    None,
                    None,
                    None,
                    receipt,
                    canonical_sha256(receipt),
                )
            prefix["escalationActions"] = escalation_actions[:action_index]
            normalized = normalizer.normalize(
                prefix,
                decision_point=recipe.scenario.decision_point,
            )
            receipt["escalation"] = escalation.to_dict()
            return self._finish_candidate(
                recipe,
                normalized=normalized,
                bundle_hash=bundle_hash,
                seed=seed,
                receipt=receipt,
                auction=auction,
                cancel_check=cancel_check,
                policy=compiled_policy,
            )

        if decision_kind == "project":
            normalized = normalizer.normalize(
                prefix,
                decision_point=recipe.scenario.decision_point,
            )
            return self._finish_candidate(
                recipe,
                normalized=normalized,
                bundle_hash=bundle_hash,
                seed=seed,
                receipt=receipt,
                auction=auction,
                cancel_check=cancel_check,
                policy=compiled_policy,
            )

        normalized = normalizer.normalize(prefix)
        play_action_prefix_rows = self._hard(recipe, "play.action_prefix")
        play_prefix_rows = self._hard(recipe, "play.card_prefix")
        typed_action_prefix = bool(play_action_prefix_rows)
        required_actions = (
            [
                LegalPlayActionV1(
                    Card.parse(action["card"]),
                    bool(action["ekka"]),
                )
                for action in play_action_prefix_rows[0].arguments["actions"]
            ]
            if play_action_prefix_rows
            else [
                LegalPlayActionV1(Card.parse(card))
                for card in (
                    play_prefix_rows[0].arguments["cards"] if play_prefix_rows else []
                )
            ]
        )
        remaining_target = int(selector.get("remainingCards", 8))
        guidance = ScenarioGuidancePlanV1.build(recipe.scenario, recipe.proof)
        policy = (
            ScenarioGuidedActorSafePolicyV1(guidance)
            if recipe.resources["generationPolicy"] == "scenario_guided_actor_safe"
            else None
        )
        natural = HeuristicBot(engine, BotProfile(int(seed)))
        state = normalized.round_state.clone()
        decision_receipts: list[dict[str, Any]] = []
        while sum(mask.bit_count() for mask in state.hands) > remaining_target:
            if cancel_check is not None and cancel_check():
                receipt.update({"status": "not_found_within_budget", "cancelled": True})
                receipt["sha256"] = canonical_sha256(receipt)
                return ScenarioSearchCandidateV1(
                    "not_found_within_budget",
                    seed,
                    None,
                    None,
                    None,
                    receipt,
                    canonical_sha256(receipt),
                )
            actor = state.current_player
            if actor is None:
                break
            if required_actions:
                required_action = required_actions.pop(0)
                if required_action not in engine.legal_play_actions(state):
                    receipt.update(
                        {
                            "status": "construction_target_miss",
                            "failure": (
                                "required_play_action_prefix_is_illegal"
                                if typed_action_prefix
                                else "required_play_prefix_is_illegal"
                            ),
                        }
                    )
                    receipt["sha256"] = canonical_sha256(receipt)
                    return ScenarioSearchCandidateV1(
                        "construction_target_miss",
                        seed,
                        None,
                        None,
                        None,
                        receipt,
                        canonical_sha256(receipt),
                    )
                card = required_action.card
                ekka = required_action.ekka
            else:
                ekka = False
                if policy is not None:
                    card, decision_receipt = policy.choose_with_receipt(state, engine)
                else:
                    information_hash = canonical_sha256(
                        InformationState.from_complete(state, actor).to_dict()
                    )
                    legal = sorted(
                        engine.legal_actions(state), key=lambda item: item.id
                    )
                    card = natural.choose(state)
                    decision_core = {
                        "schemaVersion": 1,
                        "policyId": "heuristic_bot_v1",
                        "policyVersion": "1.0.0",
                        "actingSeat": actor.name.lower(),
                        "informationStateHash": information_hash,
                        "candidateActions": [item.code for item in legal],
                        "actionScores": None,
                        "selectedAction": card.code,
                        "observerSafe": True,
                    }
                    decision_receipt = {
                        **decision_core,
                        "sha256": canonical_sha256(decision_core),
                    }
                decision_receipts.append(decision_receipt)
            prefix["playActions"].append(
                {
                    "actor": actor.name.lower(),
                    "card": card.code,
                    "ekka": ekka,
                }
            )
            engine.play_card(state, card, ekka=ekka)
        if required_actions:
            receipt.update(
                {
                    "status": "construction_target_miss",
                    "failure": (
                        "required_play_action_prefix_exceeds_decision_point"
                        if typed_action_prefix
                        else "required_play_prefix_exceeds_decision_point"
                    ),
                }
            )
            receipt["sha256"] = canonical_sha256(receipt)
            return ScenarioSearchCandidateV1(
                "construction_target_miss",
                seed,
                None,
                None,
                None,
                receipt,
                canonical_sha256(receipt),
            )
        normalized = normalizer.normalize(prefix)
        acting_seat = selector.get("actingSeat")
        if acting_seat is not None and normalized.next_actor != acting_seat:
            observed_trick = normalized.round_state.current_trick
            receipt.update(
                {
                    "status": "construction_target_miss",
                    "failure": "decision_actor_mismatch",
                    "decisionReceipts": [
                        copy.deepcopy(dict(item)) for item in decision_receipts
                    ],
                    "decisionPointObserved": {
                        "nextActor": normalized.next_actor,
                        "completedTrickCount": len(
                            normalized.round_state.completed_tricks
                        ),
                        "currentTrickLeader": (
                            observed_trick.leader.name.lower()
                            if observed_trick is not None
                            else None
                        ),
                        "currentTrickCards": (
                            [play.card.code for play in observed_trick.plays]
                            if observed_trick is not None
                            else []
                        ),
                    },
                }
            )
            receipt["sha256"] = canonical_sha256(receipt)
            return ScenarioSearchCandidateV1(
                "construction_target_miss",
                seed,
                normalized,
                None,
                None,
                receipt,
                canonical_sha256(receipt),
            )
        receipt["playActionCount"] = len(prefix["playActions"])
        return self._finish_candidate(
            recipe,
            normalized=normalized,
            bundle_hash=bundle_hash,
            seed=seed,
            receipt=receipt,
            auction=auction,
            cancel_check=cancel_check,
            decision_receipts=decision_receipts,
            policy=(
                {
                    "id": policy.policy_id,
                    "version": policy.version,
                    "guidancePlanHash": policy.guidance.sha256,
                    "observerSafe": True,
                }
                if policy is not None
                else {
                    "id": "heuristic_bot_v1",
                    "version": "1.0.0",
                    "observerSafe": True,
                }
            ),
        )


__all__ = [
    "ScenarioGuidancePlanV1",
    "ScenarioGuidedActorSafePolicyV1",
    "ScenarioSearchCandidateV1",
    "ScenarioSearchExecutorV1",
]
