"""Closed outcome-YAML facade over V4 predicates and native trick variables.

Compilation is not generation or a policy proof. The construction consumer must
bind BOTH the V4 scenario and constrain_play_model before solving.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from baloot_platform.core.cards import Card, FULL_DECK
from baloot_platform.core.enums import Contract, Seat, Suit, Team
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.scoring_ir import ScoringDomainIRV1

from .scenario_contracts import ResearchScenarioV1, canonical_sha256
from .scenario_predicate_lowering_v2 import ScenarioPredicateLoweringRegistryV2
from .scenario_predicate_manifest_v2 import shared_scenario_predicate_catalog_lowering_manifest_v2


PRESETS = frozenset({"big_swing", "comeback", "last_trick_reversal", "declaration", "near_kaboot", "kaboot"})


def _object(value: Any, allowed: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{label} must be an object with string keys")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{label}: unsupported fields {sorted(unknown)}")
    return copy.deepcopy(dict(value))


def _integer(value: Any, minimum: int, maximum: int, label: str) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be an integer in {minimum}..{maximum}")
    return value


def _team(value: Any) -> str:
    if value not in {team.name.lower() for team in Team}:
        raise ValueError("unsupported team")
    return value


@dataclass(frozen=True, slots=True)
class RawTrickMargin:
    """Engine raw-team margin change over [start, stop), including final bonus."""
    team: str
    start: int
    stop: int
    minimum: int
    maximum: int


@dataclass(frozen=True, slots=True)
class CompiledOutcomeSpec:
    source: Mapping[str, Any]
    rules_hash: str
    scenario: ResearchScenarioV1
    trajectory: tuple[RawTrickMargin, ...]
    seat_policies: tuple[tuple[str, str], ...]
    sha256: str

    def require_integrity(self):
        scenario = ResearchScenarioV1.build(self.scenario.to_dict())
        core = dict(source=self.source, rulesHash=self.rules_hash, scenario=scenario.sha256,
                    trajectory=[asdict(row) for row in self.trajectory], seatPolicies=dict(self.seat_policies))
        if canonical_sha256(core) != self.sha256:
            raise ValueError("compiled outcome identity mismatch")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1, "source": copy.deepcopy(dict(self.source)),
            "rulesHash": self.rules_hash, "scenario": self.scenario.to_dict(),
            "trajectory": [asdict(row) for row in self.trajectory],
            "seatPolicies": dict(self.seat_policies), "sha256": self.sha256,
        }

    def constrain_play_model(self, build: Any) -> None:
        """Use native raw-team prefixes, including the scoring authority's bonus."""
        if (len(build.raw_team_prefix_variables) != 9
                or set(build.predicate_context.tricks) != set(range(8))):
            raise ValueError("outcome construction requires a full eight-trick native model")
        lowerer = ScenarioPredicateLoweringRegistryV2()
        for expression in self.scenario.constraints.children:
            if all(predicate.kind.startswith("play.") for predicate in expression.predicates()):
                literal, _ = lowerer.lower_expression(expression, build.predicate_context)
                build.model.add(literal == 1)
        for index, row in enumerate(self.trajectory):
            team = int(Team[row.team.upper()])
            start = build.raw_team_prefix_variables[row.start]
            stop = build.raw_team_prefix_variables[row.stop]
            margin = stop[team] - stop[1-team] - start[team] + start[1-team]
            build.model.add(margin >= row.minimum)
            build.model.add(margin <= row.maximum)


def load_outcome_spec(path: str | Path, rules: RuleBook) -> CompiledOutcomeSpec:
    """YAML is data only; unknown clauses are rejected before model creation."""
    class Loader(yaml.SafeLoader):
        pass

    def mapping(loader, node):
        keys = [loader.construct_object(key) for key, _ in node.value]
        if any(type(key) is not str for key in keys) or len(keys) != len(set(keys)):
            raise ValueError("outcome YAML keys must be unique strings")
        return loader.construct_mapping(node)

    Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    text = Path(path).read_text(encoding="utf-8")
    if len(text.encode("utf-8")) > 131_072:
        raise ValueError("outcome YAML exceeds 128 KiB")
    return compile_outcome_spec(yaml.load(text, Loader=Loader), rules)


def compile_outcome_spec(value: Mapping[str, Any], rules: RuleBook) -> CompiledOutcomeSpec:
    raw = _object(value, {"schemaVersion", "rulesProfile", "category", "mode", "trumpSuit",
                          "team", "buyer", "outcome", "declarations", "policies", "auction", "objective"}, "outcome spec")
    if type(raw.get("schemaVersion")) is not int or raw["schemaVersion"] != 1:
        raise ValueError("outcome spec schemaVersion must be 1")
    if raw.get("rulesProfile") != rules.data["profile"]["id"]:
        raise ValueError("outcome spec rulesProfile must match the supplied rulebook")
    if rules.data["projects"].get("declaration_mode") != "mandatory":
        raise ValueError("outcome generation requires mandatory project declarations")
    if raw.get("objective", "awarded_point_margin") not in {"awarded_point_margin", "raw_point_margin"}:
        raise ValueError("unsupported card-play objective")
    category = raw.get("category")
    if category is not None and category not in PRESETS:
        raise ValueError("unsupported outcome preset")
    mode = Contract.parse(raw["mode"]) if "mode" in raw else None
    if mode is not None and mode.value not in rules.data["contracts"]["allowed"]:
        raise ValueError("requested mode is disabled by the rulebook")
    trump = raw.get("trumpSuit")
    if mode is None and trump is not None:
        mode = Contract.HOKUM
    if mode == Contract.HOKUM:
        if type(trump) is not str or trump not in {suit.name.lower() for suit in Suit}:
            raise ValueError("Hokum requires an explicit trumpSuit")
    elif trump is not None:
        raise ValueError("trumpSuit is Hokum-only")
    outcome = _object(raw.get("outcome", {}), {"winner", "score", "awardedMargin", "decidingTrick",
                                              "swing", "comeback"}, "outcome")
    team = _team(raw.get("team", outcome.get("winner", "north_south")))
    if "winner" in outcome and _team(outcome["winner"]) != team:
        raise ValueError("team and outcome.winner contradict each other")
    scoring = ScoringDomainIRV1.from_rules(rules.data)
    modes = (mode,) if mode is not None else tuple(Contract.parse(item) for item in rules.data["contracts"]["allowed"])
    if mode == Contract.HOKUM and mode.value not in rules.data["contracts"]["allowed"]:
        raise ValueError("requested mode is disabled by the rulebook")
    raw_maximum = max(scoring.raw_card_totals[item.effective.value] for item in modes)
    # A conservative physical-rule bound; feasibility still belongs to native CP-SAT.
    project_domain = rules.project_domain
    project_values = project_domain.to_dict()
    def maximum_award(contract):
        project_maximum = 2 * project_domain.maximum_per_player * max(
            *project_values["sequenceValues"][contract.effective.value].values(),
            *project_values["fourKindValues"][contract.effective.value].values(), 0)
        baloot_maximum = project_domain.baloot_value if contract == Contract.HOKUM else 0
        game_maximum = scoring.card_game_point_totals[contract.effective.value]
        return max(max(scoring.round_award(contract=contract, winner=Team.NORTH_SOUTH,
            declarer_team=declarer, escalation=level, total_before_award=(game_maximum+project_maximum+baloot_maximum, 0),
            project_points=(project_maximum, 0), baloot_points=(baloot_maximum, 0), kabout=sweep))
            for level, _ in scoring.escalation_multipliers for declarer in Team for sweep in (False, True))
    score_maximum = max(maximum_award(item) for item in modes)
    constraints, outcomes, trajectory = [], [], []
    counter = 0
    manifest = shared_scenario_predicate_catalog_lowering_manifest_v2()

    def leaf(kind, arguments):
        nonlocal counter
        definition = manifest.definition(kind)
        if definition is None:
            raise ValueError(f"no native outcome lowering for {kind}")
        definition.validate_arguments(arguments)
        counter += 1
        return dict(predicateId=f"PRED-OUTCOME-{counter:03d}", kind=kind,
                    version=definition.version, hardness="hard", arguments=arguments)

    def winner(trick, side, card=None, seat=None):
        selector = ({"cards": [Card.parse(card).id]} if card else
                    {"owner": {"relation": "team", "team": side}})
        args = dict(trickIndex=trick, selector=selector, team=side)
        if seat is not None:
            if seat not in {s.name.lower() for s in Seat} or Seat[seat.upper()].team != Team[side.upper()]:
                raise ValueError("winningSeat must belong to winningTeam")
            args["seat"] = seat
        return leaf("play.trick_winner_matches", args)

    constraints.append(leaf("auction.contract", {"contract": mode.value}) if mode is not None
                       else leaf("auction.terminal", {"terminal": True}))
    if "buyer" in raw:
        if type(raw["buyer"]) is not str or raw["buyer"] not in {seat.name.lower() for seat in Seat}:
            raise ValueError("buyer must name a seat")
        constraints.append(leaf("auction.buyer", {"seat": raw["buyer"]}))
    if "auction" in raw:
        if type(raw["auction"]) is not list or not 1 <= len(raw["auction"]) <= 8:
            raise ValueError("fixed auction requires one through eight actions")
        constraints.append(leaf("auction.action_sequence", {"actions": raw["auction"], "complete": True}))
    if trump:
        constraints.append(leaf("auction.trump", {"suit": trump}))
    if "winner" in outcome or "comeback" in outcome or category in {"comeback", "last_trick_reversal", "kaboot"}:
        outcomes.append(leaf("outcome.team_round", {"team": _team(outcome.get("winner", team)), "status": "won"}))
    if "score" in outcome:
        scores = _object(outcome["score"], {t.name.lower() for t in Team}, "outcome.score")
        if not scores:
            raise ValueError("score must specify at least one team")
        for side, score in sorted(scores.items()):
            score = _integer(score, 0, score_maximum, "awarded score exceeds rulebook bound")
            outcomes.append(leaf("outcome.game_points", dict(team=side, minimum=score, maximum=score)))
    if "awardedMargin" in outcome:
        bounds = _object(outcome["awardedMargin"], {"minimum", "maximum"}, "awardedMargin")
        low = _integer(bounds.get("minimum", 1), -10000, 10000, "minimum awarded margin")
        high = _integer(bounds.get("maximum", 10000), low, 10000, "maximum awarded margin")
        outcomes.append(leaf("outcome.round_margin", dict(team=team, minimum=low, maximum=high)))
    deciding = _object(outcome.get("decidingTrick", {}), {"index", "winningTeam", "winningSeat", "winningCard",
                          "beatenCard", "beatenCardTeam"}, "decidingTrick")
    if "decidingTrick" in outcome:
        trick = _integer(deciding.get("index"), 1, 8, "deciding trick index") - 1
        side = _team(deciding.get("winningTeam", team))
        if category == "kaboot" and side != team:
            raise ValueError("kaboot contradicts an opposing deciding-trick winner")
        constraints.append(winner(trick, side, deciding.get("winningCard"), deciding.get("winningSeat")))
        if "beatenCard" in deciding:
            opponent = _team(deciding.get("beatenCardTeam", Team[side.upper()].other.name.lower()))
            if opponent == side:
                raise ValueError("beatenCardTeam must be the opposing team")
            if deciding.get("winningCard") is not None and Card.parse(deciding["winningCard"]) == Card.parse(deciding["beatenCard"]):
                raise ValueError("winning and beaten cards must differ")
            constraints.append(leaf("play.trick_card_count", dict(trickIndex=trick, minimum=1, maximum=1,
                selector={"cards": [Card.parse(deciding["beatenCard"]).id], "owner": {"relation": "team", "team": opponent}})))
        elif "beatenCardTeam" in deciding:
            raise ValueError("beatenCardTeam requires beatenCard")
    if category == "near_kaboot":
        constraints.append(dict(op="count", minimum=7, maximum=7,
                                children=[winner(index, team) for index in range(8)]))
    elif category == "kaboot":
        outcomes.append(leaf("outcome.kabout", dict(team=team, achieved=True)))
    if category == "big_swing" or "swing" in outcome:
        swing = _object(outcome.get("swing", {}), {"metric", "minimum", "trickIndex"}, "swing")
        if swing.get("metric", "raw_trick_points") != "raw_trick_points":
            raise ValueError("swing currently supports raw_trick_points only")
        trick = _integer(swing.get("trickIndex", deciding.get("index", 8)), 1, 8, "swing trickIndex")
        minimum = _integer(swing.get("minimum", 20), 1, raw_maximum, "minimum swing")
        trajectory.append(RawTrickMargin(team, trick - 1, trick, minimum, raw_maximum))
    if category in {"comeback", "last_trick_reversal"} or "comeback" in outcome:
        comeback = _object(outcome.get("comeback", {}), {"afterTrick", "minimumDeficit"}, "comeback")
        after = _integer(comeback.get("afterTrick", 7 if category == "last_trick_reversal" else 4), 1, 7, "comeback afterTrick")
        if category == "last_trick_reversal" and after != 7:
            raise ValueError("last_trick_reversal must be behind after trick 7")
        if category == "last_trick_reversal":
            constraints.append(winner(7, team))
        reachable = max(sum(sorted((rules.play_domain.points(card, item,
            (Suit.parse(trump) if trump else Suit.SPADES) if item == Contract.HOKUM else None)
            for card in FULL_DECK), reverse=True)[:4*after]) for item in modes)
        deficit = _integer(comeback.get("minimumDeficit", 1), 1, reachable, "minimumDeficit exceeds reachable bound")
        trajectory.append(RawTrickMargin(team, 0, after, -reachable, -deficit))

    declarations = raw.get("declarations", [])
    if type(declarations) is not list or len(declarations) > 8:
        raise ValueError("declarations must be an array of at most eight requirements")
    if mode is None and declarations:
        raise ValueError("declaration categories require an explicit mode")
    if category == "declaration" and not declarations:
        raise ValueError("declaration preset requires declarations")
    domain = rules.project_domain
    for declaration in declarations:
        row = _object(declaration, {"category", "team", "owner", "state"}, "declaration")
        side = _team(row.get("team", team))
        project = str(row.get("category"))
        if project == "baloot":
            if mode != Contract.HOKUM or not domain.baloot_enabled:
                raise ValueError("Baloot requires Hokum with Baloot enabled")
            if set(row) - {"category", "team"}:
                raise ValueError("Baloot supports achieved team only; owner/state unsupported")
            constraints.append(leaf("project.baloot", dict(team=side, achieved=True)))
            continue
        values = {"sira": [("siri", domain.sequence_value_for(mode, "three"))],
                  "50": [("fifty", domain.sequence_value_for(mode, "four"))],
                  "100": [("hundred_sequence", domain.sequence_value_for(mode, "five_plus"))]}
        quartets = domain.to_dict()["fourKindValues"][mode.effective.value]
        for rank, value in quartets.items():
            bucket = "400" if mode.effective == Contract.SUN and rank == "ace" else "100"
            entry = ("four_hundred" if bucket == "400" else "hundred_four_kind", value)
            if entry not in values.setdefault(bucket, []):
                values[bucket].append(entry)
        if project not in values:
            raise ValueError("unsupported declaration category for this mode")
        alternatives = []
        for kind, value in values[project]:
            args = dict(team=side, kind=kind, value=value, state=row.get("state", "awarded"))
            if "owner" in row:
                if row["owner"] not in {s.name.lower() for s in Seat} or Seat[row["owner"].upper()].team != Team[side.upper()]:
                    raise ValueError("project owner must belong to its team")
                args["owner"] = row["owner"]
            alternatives.append(leaf("project.exists", args))
        constraints.append(alternatives[0] if len(alternatives) == 1 else dict(op="any", children=alternatives))
    policies = _object(raw.get("policies", {}), {"default", "teams", "seats"}, "policies")
    default = policies.get("default", "informed")
    teams = _object(policies.get("teams", {}), {t.name.lower() for t in Team}, "policies.teams")
    seats = _object(policies.get("seats", {}), {s.name.lower() for s in Seat}, "policies.seats")
    if any(policy not in {"informed", "optimal"} for policy in [default, *teams.values(), *seats.values()]):
        raise ValueError("policy must be informed or optimal; no heuristic optimality relabelling")
    seat_policies = tuple((s.name.lower(), seats.get(s.name.lower(), teams.get(s.team.name.lower(), default))) for s in Seat)
    if not outcomes:
        # A mode-only spec is allowed; this tautology adds no requested outcome.
        outcomes.append(leaf("outcome.game_points", dict(team=team, minimum=0, maximum=10000)))
    scenario = ResearchScenarioV1.build(dict(schemaVersion=1, origin={"kind": "search"},
        constraints={"op": "all", "children": constraints},
        decisionPoint={"kind": "play", "selector": {"trickIndex": 0, "position": 0}},
        outcome={"op": "all", "children": outcomes}))
    core = dict(source=raw, rulesHash=rules.sha256, scenario=scenario.sha256,
                trajectory=[asdict(row) for row in trajectory], seatPolicies=dict(seat_policies))
    return CompiledOutcomeSpec(raw, rules.sha256, scenario, tuple(trajectory), seat_policies, canonical_sha256(core))
