"""Engine-free CP-SAT lowerings for project and terminal-outcome predicates."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from ortools.sat.python import cp_model

from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Seat, Team

from .ortools_project_fragment import (
    OrToolsProjectDeclarationFragmentV1,
    OrToolsProjectEligibilityFragmentV1,
    OrToolsProjectLifecycleFragmentV1,
    ProjectPatternEvidenceV1,
)
from .scenario_contracts import ScenarioPredicateV1, canonical_sha256
from .scenario_predicate_lowering_v2 import CpSatPredicateLoweringV1
from .scenario_registry_v2 import scenario_predicate_definitions_v2


PROJECT_PREDICATE_KINDS_V2 = frozenset(
    {
        "project.exists",
        "project.count",
        "project.combined_value",
        "project.baloot",
        "project.overlap_count",
    }
)
OUTCOME_PREDICATE_KINDS_V2 = frozenset(
    {
        "outcome.buyer_contract",
        "outcome.team_round",
        "outcome.team_match",
        "outcome.raw_points",
        "outcome.game_points",
        "outcome.project_points",
        "outcome.round_margin",
        "outcome.kabout",
        "outcome.gahwa",
        "outcome.match_target_crossing",
    }
)
PROJECT_OUTCOME_PREDICATE_KINDS_V2 = (
    PROJECT_PREDICATE_KINDS_V2 | OUTCOME_PREDICATE_KINDS_V2
)


@dataclass(frozen=True, slots=True)
class CpSatProjectPredicateContextV1:
    model: cp_model.CpModel
    context_id: str
    projects: OrToolsProjectLifecycleFragmentV1 | OrToolsProjectEligibilityFragmentV1

    def __post_init__(self) -> None:
        if not isinstance(self.model, cp_model.CpModel):
            raise TypeError("project predicate context requires a CpModel")
        if type(self.context_id) is not str or not self.context_id:
            raise ValueError("project predicate context_id must be non-empty")
        if not isinstance(self.projects, (OrToolsProjectLifecycleFragmentV1, OrToolsProjectEligibilityFragmentV1)):
            raise TypeError("project predicate context requires a project fragment")

    @property
    def sha256(self) -> str:
        return canonical_sha256(
            {
                "schemaVersion": 1,
                "contextId": self.context_id,
                "projectFragmentHash": self.projects.fragment_hash,
            }
        )


@dataclass(frozen=True, slots=True)
class CpSatOutcomePredicateContextV1:
    """Variables emitted by either the scoring or complete-play CP-SAT stage."""

    model: cp_model.CpModel
    context_id: str
    source_stage: str
    source_hash: str
    buyer_seat: cp_model.IntVar
    round_winner: cp_model.IntVar
    match_winner: cp_model.IntVar
    raw_points: tuple[cp_model.IntVar, cp_model.IntVar]
    game_points: tuple[cp_model.IntVar, cp_model.IntVar]
    project_points: tuple[cp_model.IntVar, cp_model.IntVar]
    final_match_score: tuple[cp_model.IntVar, cp_model.IntVar]
    kabout: cp_model.IntVar
    gahwa: cp_model.IntVar
    match_target: int

    def __post_init__(self) -> None:
        if not isinstance(self.model, cp_model.CpModel):
            raise TypeError("outcome predicate context requires a CpModel")
        if type(self.context_id) is not str or not self.context_id:
            raise ValueError("outcome predicate context_id must be non-empty")
        if self.source_stage not in {"scoring", "full_play"}:
            raise ValueError("outcome source_stage must be scoring or full_play")
        if (
            type(self.source_hash) is not str
            or re.fullmatch(r"[0-9a-f]{64}", self.source_hash) is None
        ):
            raise ValueError("outcome source_hash must be a SHA-256 digest")
        for name in (
            "raw_points",
            "game_points",
            "project_points",
            "final_match_score",
        ):
            values = getattr(self, name)
            if type(values) is not tuple or len(values) != 2:
                raise ValueError(f"outcome {name} requires two team variables")
        if type(self.match_target) is not int or self.match_target <= 0:
            raise ValueError("outcome match_target must be a positive integer")

    @property
    def sha256(self) -> str:
        return canonical_sha256(
            {
                "schemaVersion": 1,
                "contextId": self.context_id,
                "sourceStage": self.source_stage,
                "sourceHash": self.source_hash,
                "variables": {
                    "buyerSeat": self.buyer_seat.Name(),
                    "roundWinner": self.round_winner.Name(),
                    "matchWinner": self.match_winner.Name(),
                    "rawPoints": [item.Name() for item in self.raw_points],
                    "gamePoints": [item.Name() for item in self.game_points],
                    "projectPoints": [item.Name() for item in self.project_points],
                    "finalMatchScore": [item.Name() for item in self.final_match_score],
                    "kabout": self.kabout.Name(),
                    "gahwa": self.gahwa.Name(),
                },
                "matchTarget": self.match_target,
            }
        )


@dataclass(frozen=True, slots=True)
class ProjectOutcomePredicateRegistrationV1:
    kind: str
    context_kind: str
    evidence_extractor_id: str


_EXTRACTORS = MappingProxyType(
    {
        kind: f"cp_sat.{kind.replace('.', '_')}_evidence_v1"
        for kind in sorted(PROJECT_OUTCOME_PREDICATE_KINDS_V2)
    }
)


def project_outcome_predicate_registrations_v1(
) -> tuple[ProjectOutcomePredicateRegistrationV1, ...]:
    """Return stable registrations for the later combined lowering registry."""

    return tuple(
        ProjectOutcomePredicateRegistrationV1(
            kind,
            "project" if kind in PROJECT_PREDICATE_KINDS_V2 else "outcome",
            _EXTRACTORS[kind],
        )
        for kind in sorted(PROJECT_OUTCOME_PREDICATE_KINDS_V2)
    )


def _and(model: cp_model.CpModel, values: Sequence[Any], name: str) -> cp_model.IntVar:
    result = model.new_bool_var(name)
    if not values:
        model.add(result == 1)
    else:
        model.add_bool_and(list(values)).only_enforce_if(result)
        model.add_bool_or([value.Not() for value in values]).only_enforce_if(
            result.Not()
        )
    return result


def _equal(
    model: cp_model.CpModel, left: Any, right: Any, name: str
) -> cp_model.IntVar:
    result = model.new_bool_var(name)
    model.add(left == right).only_enforce_if(result)
    model.add(left != right).only_enforce_if(result.Not())
    return result


def _range(
    model: cp_model.CpModel,
    value: Any,
    minimum: int,
    maximum: int,
    name: str,
) -> cp_model.IntVar:
    lower = model.new_bool_var(f"{name}_minimum")
    upper = model.new_bool_var(f"{name}_maximum")
    model.add(value >= minimum).only_enforce_if(lower)
    model.add(value < minimum).only_enforce_if(lower.Not())
    model.add(value <= maximum).only_enforce_if(upper)
    model.add(value > maximum).only_enforce_if(upper.Not())
    return _and(model, (lower, upper), name)


def _negate(model: cp_model.CpModel, value: Any, name: str) -> cp_model.IntVar:
    result = model.new_bool_var(name)
    model.add(result + value == 1)
    return result


def _team(value: Any) -> Team:
    if type(value) is not str:
        raise TypeError("team must be a string")
    try:
        return Team[value.upper()]
    except KeyError as exc:
        raise ValueError("team is unsupported") from exc


def _project_state(project: ProjectPatternEvidenceV1, state: str) -> bool:
    return bool(getattr(project, state))


class ProjectOutcomePredicateLoweringV2:
    """Closed exact lowerer for the five project and ten outcome V1 kinds."""

    registrations = project_outcome_predicate_registrations_v1()

    def __init__(self) -> None:
        definitions = {
            row.kind: row for row in scenario_predicate_definitions_v2()
        }
        self._definitions = {
            kind: definitions[kind] for kind in PROJECT_OUTCOME_PREDICATE_KINDS_V2
        }

    @property
    def manifest_hash(self) -> str:
        return canonical_sha256(
            {
                "schemaVersion": 1,
                "registrations": [
                    {
                        "kind": row.kind,
                        "contextKind": row.context_kind,
                        "evidenceExtractor": row.evidence_extractor_id,
                    }
                    for row in self.registrations
                ],
            }
        )

    def lower_predicate(
        self,
        predicate: ScenarioPredicateV1,
        context: CpSatProjectPredicateContextV1 | CpSatOutcomePredicateContextV1,
    ) -> CpSatPredicateLoweringV1:
        definition = self._definitions.get(predicate.kind)
        if definition is None or predicate.version != definition.version:
            raise ValueError(
                f"predicate has no project/outcome CP-SAT lowering: "
                f"{predicate.kind}@{predicate.version}"
            )
        definition.validate_arguments(predicate.arguments)
        arguments = dict(predicate.arguments)
        if "minimum" in arguments and "maximum" in arguments:
            minimum = arguments["minimum"]
            maximum = arguments["maximum"]
            if isinstance(minimum, bool) or isinstance(maximum, bool):
                raise TypeError("predicate ranges cannot use booleans")
            if minimum > maximum:
                raise ValueError(f"{predicate.kind} minimum cannot exceed maximum")
        if predicate.kind in PROJECT_PREDICATE_KINDS_V2:
            if not isinstance(context, CpSatProjectPredicateContextV1):
                raise TypeError(f"{predicate.kind} requires the project CP-SAT context")
            literal, extract = self._lower_project(predicate, context)
            context_hash = context.sha256
        else:
            if not isinstance(context, CpSatOutcomePredicateContextV1):
                raise TypeError(f"{predicate.kind} requires the outcome CP-SAT context")
            literal, extract = self._lower_outcome(predicate, context)
            context_hash = context.sha256
        lowering_core = {
            "schemaVersion": 1,
            "predicateHash": predicate.sha256,
            "contextHash": context_hash,
            "evidenceExtractor": _EXTRACTORS[predicate.kind],
        }
        return CpSatPredicateLoweringV1(
            predicate.predicate_id,
            predicate.kind,
            literal,
            _EXTRACTORS[predicate.kind],
            canonical_sha256(lowering_core),
            extract,
        )

    @staticmethod
    def _result_literal(
        model: cp_model.CpModel, predicate: ScenarioPredicateV1, source: Any
    ) -> cp_model.IntVar:
        result = model.new_bool_var(f"scenario_assumption_{predicate.predicate_id}")
        model.add(result == source)
        return result

    def _lower_project(
        self,
        predicate: ScenarioPredicateV1,
        context: CpSatProjectPredicateContextV1,
    ) -> tuple[cp_model.IntVar, Callable[[cp_model.CpSolver], Mapping[str, Any]]]:
        model = context.model
        fragment = context.projects
        args = dict(predicate.arguments)
        kind = predicate.kind
        declaration_projection = isinstance(fragment, OrToolsProjectDeclarationFragmentV1)
        available_states = {'eligible', 'declared'} if declaration_projection else {'eligible'}
        if isinstance(fragment, OrToolsProjectEligibilityFragmentV1) and (
            kind == "project.baloot"
            or (kind != "project.overlap_count" and args.get("state", "eligible") not in available_states)
        ):
            raise ValueError("project construction projection cannot lower unavailable states or Baloot")
        if kind == "project.baloot":
            team = _team(args["team"])
            actual = fragment.baloot_team[int(team)]
            source = actual if args["achieved"] else _negate(
                model, actual, f"{predicate.predicate_id}_not_achieved"
            )

            def extract(solver: cp_model.CpSolver) -> Mapping[str, Any]:
                achieved = bool(solver.boolean_value(actual))
                return {"team": team.name.lower(), "achieved": achieved}

            return self._result_literal(model, predicate, source), extract

        if kind == "project.overlap_count":
            owner = Seat.parse(args["owner"])
            overlaps: list[tuple[Card, cp_model.IntVar, tuple[str, ...]]] = []
            for card_id in range(32):
                candidates = tuple(
                    project
                    for project in fragment.projects
                    if project.owner == owner and any(card.id == card_id for card in project.cards)
                )
                if len(candidates) < 2:
                    continue
                count = model.new_int_var(
                    0, len(candidates), f"{predicate.predicate_id}_card_{card_id}_count"
                )
                model.add(count == sum(project.eligible for project in candidates))
                overlapping = model.new_bool_var(
                    f"{predicate.predicate_id}_card_{card_id}_overlap"
                )
                model.add(count >= 2).only_enforce_if(overlapping)
                model.add(count <= 1).only_enforce_if(overlapping.Not())
                overlaps.append(
                    (Card(card_id), overlapping, tuple(item.project_id for item in candidates))
                )
            overlap_count = model.new_int_var(
                0, len(overlaps), f"{predicate.predicate_id}_overlap_count"
            )
            model.add(overlap_count == sum(item[1] for item in overlaps))
            source = _range(
                model,
                overlap_count,
                int(args["minimum"]),
                int(args["maximum"]),
                f"{predicate.predicate_id}_range",
            )

            def extract(solver: cp_model.CpSolver) -> Mapping[str, Any]:
                return {
                    "owner": owner.name.lower(),
                    "overlapCardCount": int(solver.value(overlap_count)),
                    "overlapCards": [
                        {
                            "card": card.to_dict(),
                            "projectIds": list(project_ids),
                        }
                        for card, literal, project_ids in overlaps
                        if solver.boolean_value(literal)
                    ],
                }

            return self._result_literal(model, predicate, source), extract

        team = _team(args["team"])
        state = str(args.get("state") or "eligible")
        matches = tuple(
            project
            for project in fragment.projects
            if project.team == team
            and (kind != "project.exists" or project.kind == args["kind"])
            and (kind != "project.exists" or project.value == int(args["value"]))
            and (
                kind != "project.exists"
                or args.get("owner") is None
                or project.owner == Seat.parse(args["owner"])
            )
        )
        state_literals = tuple(project.literal(state) for project in matches)
        count = model.new_int_var(0, len(matches), f"{predicate.predicate_id}_count")
        model.add(count == sum(state_literals))
        combined = model.new_int_var(
            0,
            sum(project.value for project in matches),
            f"{predicate.predicate_id}_combined_value",
        )
        model.add(
            combined
            == sum(
                project.value * literal
                for project, literal in zip(matches, state_literals, strict=True)
            )
        )
        if kind == "project.exists":
            source = _range(model, count, 1, len(matches), f"{predicate.predicate_id}_exists")
        elif kind == "project.count":
            source = _range(
                model, count, int(args["minimum"]), int(args["maximum"]),
                f"{predicate.predicate_id}_range",
            )
        else:
            source = _range(
                model, combined, int(args["minimum"]), int(args["maximum"]),
                f"{predicate.predicate_id}_range",
            )

        def extract(solver: cp_model.CpSolver) -> Mapping[str, Any]:
            if isinstance(fragment, OrToolsProjectEligibilityFragmentV1):
                values = {
                    "team": team.name.lower(), "requestedState": state,
                    "count": int(solver.value(count)),
                    "combinedValue": int(solver.value(combined)),
                    "projects": [{"projectId": p.project_id, "owner": p.owner.name.lower(),
                        "kind": p.kind, "value": p.value, "cards": [c.id for c in p.cards],
                        "eligible": True,
                        **({'selected': bool(solver.boolean_value(p.selected)),
                            'declarable': bool(solver.boolean_value(p.declarable)),
                            'declared': bool(solver.boolean_value(p.declared)),
                            'declarationOrder': solver.value(p.declaration_order)
                                if solver.boolean_value(p.declared) else None} if declaration_projection else {})}
                        for p in matches if solver.boolean_value(p.literal(state))],
                    "projectFragmentHash": fragment.fragment_hash,
                    "constructionScope": "project_declarations" if declaration_projection else "project_eligibility",
                }
                return {**values, "projectEvidenceHash": canonical_sha256(values)}
            evidence = fragment.extract(solver)
            observed = tuple(
                project
                for project in evidence.projects
                if project.team == team
                and _project_state(project, state)
                and (kind != "project.exists" or project.kind == args["kind"])
                and (kind != "project.exists" or project.value == int(args["value"]))
                and (
                    kind != "project.exists"
                    or args.get("owner") is None
                    or project.owner == Seat.parse(args["owner"])
                )
            )
            return {
                "team": team.name.lower(),
                "requestedState": state,
                "count": len(observed),
                "combinedValue": sum(project.value for project in observed),
                "projects": [project.to_dict() for project in observed],
                "projectFragmentHash": fragment.fragment_hash,
                "projectEvidenceHash": evidence.evidence_hash,
            }

        return self._result_literal(model, predicate, source), extract

    def _lower_outcome(
        self,
        predicate: ScenarioPredicateV1,
        context: CpSatOutcomePredicateContextV1,
    ) -> tuple[cp_model.IntVar, Callable[[cp_model.CpSolver], Mapping[str, Any]]]:
        model = context.model
        args = dict(predicate.arguments)
        kind = predicate.kind

        if kind == "outcome.buyer_contract":
            buyer_team = model.new_int_var(
                0, 1, f"{predicate.predicate_id}_buyer_team"
            )
            model.add_allowed_assignments(
                [context.buyer_seat, buyer_team],
                [(int(seat), int(seat.team)) for seat in Seat],
            )
            won = _equal(model, context.round_winner, buyer_team, f"{predicate.predicate_id}_won")
            source = won if args["status"] == "won" else _negate(
                model, won, f"{predicate.predicate_id}_lost"
            )
        elif kind in {"outcome.team_round", "outcome.team_match"}:
            team = _team(args["team"])
            winner = context.round_winner if kind == "outcome.team_round" else context.match_winner
            expected = {
                "won": int(team),
                "lost": int(team.other),
                "tied": 2,
            }[args["status"]]
            source = _equal(model, winner, expected, f"{predicate.predicate_id}_status")
        elif kind in {
            "outcome.raw_points",
            "outcome.game_points",
            "outcome.project_points",
            "outcome.round_margin",
        }:
            team = _team(args["team"])
            values = {
                "outcome.raw_points": context.raw_points,
                "outcome.game_points": context.game_points,
                "outcome.project_points": context.project_points,
            }
            value = (
                context.game_points[int(team)] - context.game_points[int(team.other)]
                if kind == "outcome.round_margin"
                else values[kind][int(team)]
            )
            minimum = math.ceil(args["minimum"])
            maximum = math.floor(args["maximum"])
            source = (
                model.new_constant(0)
                if minimum > maximum
                else _range(model, value, minimum, maximum, f"{predicate.predicate_id}_range")
            )
        elif kind in {"outcome.kabout", "outcome.gahwa"}:
            team = _team(args["team"])
            occurred = context.kabout if kind == "outcome.kabout" else context.gahwa
            team_won = _equal(
                model, context.round_winner, int(team), f"{predicate.predicate_id}_team_won"
            )
            achieved = _and(model, (occurred, team_won), f"{predicate.predicate_id}_achieved")
            source = achieved if args["achieved"] else _negate(
                model, achieved, f"{predicate.predicate_id}_not_achieved"
            )
        elif kind == "outcome.match_target_crossing":
            team = _team(args["team"])
            winner = _equal(
                model, context.match_winner, int(team), f"{predicate.predicate_id}_winner"
            )
            reached = _range(
                model,
                context.final_match_score[int(team)],
                context.match_target,
                2_147_483_647,
                f"{predicate.predicate_id}_target",
            )
            achieved = _and(model, (winner, reached), f"{predicate.predicate_id}_achieved")
            source = achieved if args["achieved"] else _negate(
                model, achieved, f"{predicate.predicate_id}_not_achieved"
            )
        else:  # pragma: no cover - closed kind set guards this branch
            raise AssertionError(f"unhandled outcome predicate: {kind}")

        def extract(solver: cp_model.CpSolver) -> Mapping[str, Any]:
            buyer = Seat(int(solver.value(context.buyer_seat)))
            round_winner_value = int(solver.value(context.round_winner))
            match_winner_value = int(solver.value(context.match_winner))
            raw = [int(solver.value(item)) for item in context.raw_points]
            game = [int(solver.value(item)) for item in context.game_points]
            projects = [int(solver.value(item)) for item in context.project_points]
            final_match = [int(solver.value(item)) for item in context.final_match_score]
            snapshot = {
                "sourceStage": context.source_stage,
                "sourceHash": context.source_hash,
                "buyer": buyer.name.lower(),
                "buyerTeam": buyer.team.name.lower(),
                "roundWinner": Team(round_winner_value).name.lower(),
                "matchWinner": (
                    Team(match_winner_value).name.lower()
                    if match_winner_value in {0, 1}
                    else None
                ),
                "rawPoints": raw,
                "gamePoints": game,
                "projectPoints": projects,
                "roundMargin": [game[0] - game[1], game[1] - game[0]],
                "kabout": bool(solver.boolean_value(context.kabout)),
                "gahwa": bool(solver.boolean_value(context.gahwa)),
                "finalMatchScore": final_match,
                "matchTarget": context.match_target,
            }
            if kind == "outcome.buyer_contract":
                snapshot["status"] = (
                    "won" if round_winner_value == int(buyer.team) else "lost"
                )
            elif kind in {"outcome.team_round", "outcome.team_match"}:
                team = _team(args["team"])
                winner = round_winner_value if kind == "outcome.team_round" else match_winner_value
                snapshot.update(
                    {
                        "team": team.name.lower(),
                        "status": "won" if winner == int(team) else "lost" if winner == int(team.other) else "tied",
                    }
                )
            elif kind in {
                "outcome.raw_points",
                "outcome.game_points",
                "outcome.project_points",
                "outcome.round_margin",
            }:
                team = _team(args["team"])
                observed = {
                    "outcome.raw_points": raw[int(team)],
                    "outcome.game_points": game[int(team)],
                    "outcome.project_points": projects[int(team)],
                    "outcome.round_margin": game[int(team)] - game[int(team.other)],
                }[kind]
                snapshot.update(
                    {
                        "team": team.name.lower(),
                        "value": observed,
                        "minimum": args["minimum"],
                        "maximum": args["maximum"],
                    }
                )
            elif kind in {"outcome.kabout", "outcome.gahwa"}:
                team = _team(args["team"])
                event = snapshot["kabout"] if kind == "outcome.kabout" else snapshot["gahwa"]
                snapshot.update(
                    {"team": team.name.lower(), "achieved": event and round_winner_value == int(team)}
                )
            else:
                team = _team(args["team"])
                snapshot.update(
                    {
                        "team": team.name.lower(),
                        "achieved": match_winner_value == int(team) and final_match[int(team)] >= context.match_target,
                    }
                )
            return snapshot

        return self._result_literal(model, predicate, source), extract


__all__ = [
    "CpSatOutcomePredicateContextV1",
    "CpSatProjectPredicateContextV1",
    "OUTCOME_PREDICATE_KINDS_V2",
    "PROJECT_OUTCOME_PREDICATE_KINDS_V2",
    "PROJECT_PREDICATE_KINDS_V2",
    "ProjectOutcomePredicateLoweringV2",
    "ProjectOutcomePredicateRegistrationV1",
    "project_outcome_predicate_registrations_v1",
]
