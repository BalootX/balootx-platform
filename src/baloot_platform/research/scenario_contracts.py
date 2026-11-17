"""Closed, deterministic contracts for configurable Research V4 scenarios.

This module deliberately contains no solver or transport code.  It is the
single authority for the scenario expression, exact-prefix input, proof
request, diagnostics, and immutable proof outcome wire shapes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

from baloot_platform.adjudication.research import AuthoredAdjudicationEventV1
from baloot_platform.core.enums import Seat
from baloot_platform.tactics import DELAYED_PAYOFF_TACTIC_IDS, TacticalDetectorProfileV1


SCENARIO_SECTIONS = (
    "deal",
    "auction",
    "escalation",
    "projects",
    "playPrefix",
    "state",
    "outcome",
    "tactics",
    "proof",
)
SCENARIO_OPERATORS = frozenset({"all", "any", "not", "count"})
SCENARIO_EXPRESSION_MAXIMUM_DEPTH = 8
SCENARIO_EXPRESSION_MAXIMUM_NODES = 256
SCENARIO_EXPRESSION_MAXIMUM_CHILDREN = 64
SCENARIO_EXPRESSION_MAXIMUM_SERIALIZED_BYTES = 131_072
SCENARIO_EXPRESSION_MAXIMUM_COUNT_RANGE = 64
PREDICATE_HARDNESS = frozenset({"hard", "prefer", "report"})
CONSTRUCTION_CAPABILITIES = frozenset(
    {
        "cp_sat_exact",
        "incremental_actor_safe",
        "guided_search",
        "postcondition_only",
        "exact_input_only",
        "unsupported",
    }
)
PROOF_MODES = ("exact", "bounded", "policy_conditioned")
SCENARIO_DECISION_KINDS = ("bid", "escalation", "project", "play")
PLAYING_QUALITY_MODES = ("budgeted", "exact")
PROOF_VISIBILITIES = ("complete_state", "actor_information_set")
PROOF_STRATEGY_CLASSES = ("pure_deterministic", "frozen_policy")
PROOF_UTILITIES = (
    "awarded_point_margin",
    "contract_success",
    "round_win",
    "match_win",
    "kabout",
)
PROOF_UTILITY_ENCODINGS = {"contract_success": ("signed_win_loss_v1",)}
PROOF_CLAIMS = (
    "outcome_reachable",
    "outcome_forced",
    "winning_root_action_count",
    "winning_continuation_line_count",
    "selected_action_is_unique_winner",
    "target_team_has_no_salvage",
)
LINE_UNIVERSES = ("all_legal", "optimal_strategy", "policy_conditioned")
PROOF_HORIZON_UNITS = ("plies", "decisions", "tricks", "round_end")
PLAYING_QUALITY_HORIZON_UNITS_BY_DECISION_V1 = {
    "bid": ("plies", "decisions"),
    "escalation": ("plies", "decisions"),
    "project": ("plies", "decisions"),
    "play": ("tricks",),
}
PROOF_BOUNDARY_EVALUATOR_DEFINITIONS = {
    "current_raw_point_margin_v1": {
        "operatorVisible": True, "qualityOperatorVisible": True, "decisionKinds": SCENARIO_DECISION_KINDS,
        "kind": "state_value", "horizonUnits": PROOF_HORIZON_UNITS,
    },
    "auction_hand_margin_v1": {
        "operatorVisible": False, "qualityOperatorVisible": True, "decisionKinds": ("bid",), "kind": "heuristic",
        "horizonUnits": ("plies", "decisions"), "horizonValue": 1,
        "utilities": ("awarded_point_margin",),
    },
}
PROOF_BOUNDARY_EVALUATORS = tuple(PROOF_BOUNDARY_EVALUATOR_DEFINITIONS)
OPERATOR_PROOF_BOUNDARY_EVALUATORS = tuple(
    key for key, value in PROOF_BOUNDARY_EVALUATOR_DEFINITIONS.items() if value["operatorVisible"]
)


def validate_proof_boundary_scope_v1(request, decision_kind: str) -> None:
    if request.boundary_evaluator is not None and decision_kind not in (
        PROOF_BOUNDARY_EVALUATOR_DEFINITIONS[request.boundary_evaluator]["decisionKinds"]
    ):
        raise ValueError(f"boundary evaluator {request.boundary_evaluator} does not support {decision_kind}")


def validate_native_delayed_payoff_scope_v1(
    scenario: ResearchScenarioV1, request: ScenarioProofRequestV1,
    profile: TacticalDetectorProfileV1 | None, *, native_authority: bool = True,
) -> None:
    if (profile is not None and profile.native_delayed_gain_metric is not None
            and any(item.kind in {"tactic.applicable", "tactic.verified"}
                and item.arguments["kind"] in DELAYED_PAYOFF_TACTIC_IDS for item in scenario.predicates())
            and (not native_authority or scenario.decision_point["kind"] != "play"
                or request.mode != "exact" or request.visibility != "complete_state"
                or request.utility != "awarded_point_margin")):
        raise ValueError("native delayed payoff requires exact complete-state play with awarded_point_margin")


def hard_tactic_causal_request_issue_v1(
    kind: str, hardness: str, arguments: Mapping[str, Any],
    *, native_delayed_payoff: bool = False,
) -> str | None:
    tactic = arguments.get("kind")
    if (
        kind == "tactic.verified" and hardness == "hard"
        and arguments.get("causalRequired") is True
        and type(tactic) is str and tactic in DELAYED_PAYOFF_TACTIC_IDS
        and not native_delayed_payoff
    ):
        return "tactic_causal_verification_unavailable"
    return None


def hard_continuation_count_request_issue_v1(
    kind: str,
    hardness: str,
    arguments: Mapping[str, Any],
    selected_action: str | None,
    line_universe: str | None,
) -> str | None:
    """One compatibility rule shared by recipe validation and V3 execution."""
    if kind != "proof.winning_continuation_line_count" or hardness != "hard":
        return None
    if selected_action is None:
        return "continuation_count_requires_selected_action"
    if line_universe != arguments.get("lineUniverse"):
        return "continuation_count_line_universe_mismatch"
    return None


DIAGNOSTIC_CLASSES = frozenset(
    {
        "schema_invalid",
        "rules_inapplicable",
        "unsupported_predicate",
        "contradictory_constraints",
        "structurally_infeasible",
        "not_found_within_budget",
        "normalization_incomplete",
        "construction_target_miss",
        "semantic_postcondition_miss",
        "proof_incomplete",
        "proof_failed",
        "satisfied",
    }
)

_PREDICATE_ID = re.compile(r"^PRED-[A-Z0-9][A-Z0-9_-]{2,63}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_SEATS = frozenset({"north", "west", "south", "east"})
_TEAMS = frozenset({"north_south", "west_east"})


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return copy.deepcopy(dict(value))


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be an array")
    return copy.deepcopy(value)


def _strict_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _strict_number(value: Any, label: str) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{label} must be a number")
    result = float(value)
    if result != result or result in {float("inf"), float("-inf")}:
        raise ValueError(f"{label} must be finite")
    return result


def _closed(raw: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {', '.join(unknown)}")


@dataclass(frozen=True, slots=True)
class ScenarioPredicateV1:
    predicate_id: str
    kind: str
    version: int
    hardness: str
    arguments: Mapping[str, Any]
    sha256: str

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "ScenarioPredicateV1":
        raw = _mapping(value, "scenario predicate")
        _closed(
            raw,
            {"predicateId", "kind", "version", "hardness", "arguments", "sha256"},
            "scenario predicate",
        )
        predicate_id = str(raw.get("predicateId") or "")
        if _PREDICATE_ID.fullmatch(predicate_id) is None:
            raise ValueError("scenario predicateId is invalid")
        kind = str(raw.get("kind") or "").strip()
        if not kind or len(kind) > 120:
            raise ValueError("scenario predicate kind is invalid")
        version = _strict_int(
            raw.get("version"), "scenario predicate version", minimum=1
        )
        hardness = str(raw.get("hardness") or "")
        if hardness not in PREDICATE_HARDNESS:
            raise ValueError("scenario predicate hardness is unsupported")
        arguments = _mapping(raw.get("arguments"), "scenario predicate arguments")
        core = {
            "predicateId": predicate_id,
            "kind": kind,
            "version": version,
            "hardness": hardness,
            "arguments": arguments,
        }
        digest = canonical_sha256(core)
        declared = str(raw.get("sha256") or "")
        if declared and declared != digest:
            raise ValueError("scenario predicate checksum mismatch")
        return cls(predicate_id, kind, version, hardness, arguments, digest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicateId": self.predicate_id,
            "kind": self.kind,
            "version": self.version,
            "hardness": self.hardness,
            "arguments": copy.deepcopy(dict(self.arguments)),
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ScenarioExpressionV1:
    op: str | None
    predicate: ScenarioPredicateV1 | None
    children: tuple["ScenarioExpressionV1", ...]
    minimum: int | None
    maximum: int | None
    sha256: str

    @classmethod
    def build(
        cls,
        value: Mapping[str, Any],
        *,
        depth: int = 0,
        counter: list[int] | None = None,
    ) -> "ScenarioExpressionV1":
        if depth == 0:
            serialized_size = len(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            if serialized_size > SCENARIO_EXPRESSION_MAXIMUM_SERIALIZED_BYTES:
                raise ValueError(
                    "scenario expression exceeds maximum serialized size "
                    f"{SCENARIO_EXPRESSION_MAXIMUM_SERIALIZED_BYTES}"
                )
        if depth > SCENARIO_EXPRESSION_MAXIMUM_DEPTH:
            raise ValueError(
                "scenario expression exceeds maximum depth "
                f"{SCENARIO_EXPRESSION_MAXIMUM_DEPTH}"
            )
        if counter is None:
            counter = [0]
        counter[0] += 1
        if counter[0] > SCENARIO_EXPRESSION_MAXIMUM_NODES:
            raise ValueError(
                "scenario expression exceeds maximum node count "
                f"{SCENARIO_EXPRESSION_MAXIMUM_NODES}"
            )
        raw = _mapping(value, "scenario expression")
        is_predicate = "predicateId" in raw or "kind" in raw
        if is_predicate:
            predicate = ScenarioPredicateV1.build(raw)
            core = {"predicate": predicate.to_dict()}
            return cls(None, predicate, (), None, None, canonical_sha256(core))

        _closed(
            raw,
            {"op", "children", "minimum", "maximum", "sha256"},
            "scenario expression",
        )
        op = str(raw.get("op") or "")
        if op not in SCENARIO_OPERATORS:
            raise ValueError("scenario expression operator is unsupported")
        child_values = _list(raw.get("children"), "scenario expression children")
        if not 1 <= len(child_values) <= SCENARIO_EXPRESSION_MAXIMUM_CHILDREN:
            raise ValueError(
                "scenario expression must contain between 1 and "
                f"{SCENARIO_EXPRESSION_MAXIMUM_CHILDREN} children"
            )
        if op == "not" and len(child_values) != 1:
            raise ValueError("scenario not expression requires exactly one child")
        children = tuple(
            cls.build(
                _mapping(item, "scenario expression child"),
                depth=depth + 1,
                counter=counter,
            )
            for item in child_values
        )
        minimum: int | None = None
        maximum: int | None = None
        if op == "count":
            minimum = _strict_int(
                raw.get("minimum"), "scenario count minimum", minimum=0
            )
            maximum = _strict_int(
                raw.get("maximum"), "scenario count maximum", minimum=0
            )
            if (
                minimum > maximum
                or maximum > len(children)
                or maximum - minimum > SCENARIO_EXPRESSION_MAXIMUM_COUNT_RANGE
            ):
                raise ValueError("scenario count bounds are invalid")
        elif "minimum" in raw or "maximum" in raw:
            raise ValueError(
                "scenario count bounds are valid only for count expressions"
            )
        core: dict[str, Any] = {
            "op": op,
            "children": [child.to_dict() for child in children],
        }
        if op == "count":
            core.update({"minimum": minimum, "maximum": maximum})
        digest = canonical_sha256(core)
        declared = str(raw.get("sha256") or "")
        if declared and declared != digest:
            raise ValueError("scenario expression checksum mismatch")
        return cls(op, None, children, minimum, maximum, digest)

    def predicates(self) -> tuple[ScenarioPredicateV1, ...]:
        if self.predicate is not None:
            return (self.predicate,)
        return tuple(item for child in self.children for item in child.predicates())

    def to_dict(self) -> dict[str, Any]:
        if self.predicate is not None:
            return self.predicate.to_dict()
        result: dict[str, Any] = {
            "op": self.op,
            "children": [child.to_dict() for child in self.children],
            "sha256": self.sha256,
        }
        if self.op == "count":
            result.update({"minimum": self.minimum, "maximum": self.maximum})
        return result


@dataclass(frozen=True, slots=True)
class ExactTracePrefixV1:
    deck_order: tuple[int, ...]
    dealer: str
    round_number: int
    match_score: tuple[int, int]
    auction_actions: tuple[Mapping[str, Any], ...]
    escalation_actions: tuple[Mapping[str, Any], ...]
    project_declarations: tuple[Mapping[str, Any], ...]
    project_assertions: tuple[Mapping[str, Any], ...]
    play_actions: tuple[Mapping[str, Any], ...]
    adjudication_events: tuple[AuthoredAdjudicationEventV1, ...]
    state_assertions: Mapping[str, Any]
    sha256: str
    project_actions: tuple[Mapping[str, Any], ...] = ()
    project_actions_present: bool = False

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "ExactTracePrefixV1":
        raw = _mapping(value, "exact trace prefix")
        _closed(
            raw,
            {
                "schemaVersion",
                "deckOrder",
                "dealer",
                "roundNumber",
                "matchScore",
                "auctionActions",
                "escalationActions",
                "projectActions",
                "projectDeclarations",
                "projectAssertions",
                "playActions",
                "adjudicationEvents",
                "stateAssertions",
                "sha256",
            },
            "exact trace prefix",
        )
        if raw.get("schemaVersion") != 1 or type(raw.get("schemaVersion")) is not int:
            raise ValueError("exact trace prefix schemaVersion must be integer 1")
        deck_values = _list(raw.get("deckOrder"), "exact trace deckOrder")
        if any(type(item) is not int for item in deck_values):
            raise TypeError("exact trace deckOrder must contain only integers")
        deck_order = tuple(deck_values)
        if len(deck_order) != 32 or sorted(deck_order) != list(range(32)):
            raise ValueError("exact trace deckOrder must be a permutation of 0..31")
        dealer = str(raw.get("dealer") or "")
        if dealer not in _SEATS:
            raise ValueError("exact trace dealer is invalid")
        round_number = _strict_int(
            raw.get("roundNumber", 1), "exact trace roundNumber", minimum=1
        )
        score_values = _list(raw.get("matchScore", [0, 0]), "exact trace matchScore")
        if len(score_values) != 2:
            raise ValueError("exact trace matchScore must contain two values")
        score = tuple(
            _strict_int(item, "exact trace matchScore item", minimum=0)
            for item in score_values
        )
        sequence_fields = (
            ("auctionActions", "auction actions", {"actor", "action", "trumpSuit"}),
            (
                "escalationActions",
                "escalation actions",
                {"actor", "action", "mode", "requestedMode"},
            ),
            (
                "projectActions",
                "project actions",
                {"actor", "action", "projectId"},
            ),
            (
                "projectDeclarations",
                "project declarations",
                {"owner", "kind", "value", "highRank", "declarationOrder"},
            ),
            (
                "projectAssertions",
                "project assertions",
                {"owner", "team", "kind", "value", "highRank", "state"},
            ),
            ("playActions", "play actions", {"actor", "card", "ekka"}),
        )
        sequences: dict[str, tuple[Mapping[str, Any], ...]] = {}
        for field, label, allowed in sequence_fields:
            values = _list(raw.get(field, []), f"exact trace {label}")
            normalized_items: list[Mapping[str, Any]] = []
            for item in values:
                normalized = _mapping(item, f"exact trace {label} item")
                _closed(normalized, allowed, f"exact trace {label} item")
                actor = normalized.get("actor")
                if actor is not None and actor not in _SEATS:
                    raise ValueError(f"exact trace {label} actor is invalid")
                if field in {"auctionActions", "escalationActions"}:
                    if not str(normalized.get("action") or "").strip():
                        raise ValueError(f"exact trace {label} action is required")
                if field == "projectActions":
                    if actor is None:
                        raise ValueError("exact trace project action requires actor")
                    action = normalized.get("action")
                    if type(action) is not str or action not in {"pass", "declare"}:
                        raise ValueError(
                            "exact trace project action must be pass or declare"
                        )
                    project_id = normalized.get("projectId")
                    if action == "pass":
                        if project_id is not None:
                            raise ValueError(
                                "exact trace project pass cannot contain projectId"
                            )
                    elif (
                        type(project_id) is not str
                        or not project_id.startswith("PROJECT-")
                        or len(project_id) != 24
                        or any(
                            character not in "0123456789ABCDEF"
                            for character in project_id[8:]
                        )
                    ):
                        raise ValueError(
                            "exact trace project declare requires a stable projectId"
                        )
                if field == "escalationActions":
                    if actor is None:
                        raise ValueError("exact trace escalation action requires actor")
                    if (
                        normalized.get("mode") is not None
                        and normalized.get("requestedMode") is not None
                    ):
                        raise ValueError(
                            "exact trace escalation action cannot contain both mode aliases"
                        )
                if field == "projectAssertions":
                    normalized = {"state": "eligible", **normalized}
                    if type(normalized["state"]) is not str:
                        raise TypeError(
                            "exact trace project assertion state must be a string"
                        )
                    if normalized["state"] not in {
                        "eligible",
                        "declared",
                        "comparison_winner",
                        "awarded",
                    }:
                        raise ValueError(
                            "exact trace project assertion state is invalid"
                        )
                    if "owner" in normalized and normalized["owner"] not in _SEATS:
                        raise ValueError(
                            "exact trace project assertion owner is invalid"
                        )
                    if "team" in normalized and normalized["team"] not in _TEAMS:
                        raise ValueError(
                            "exact trace project assertion team is invalid"
                        )
                    if "kind" in normalized:
                        if (
                            type(normalized["kind"]) is not str
                            or not normalized["kind"].strip()
                        ):
                            raise ValueError(
                                "exact trace project assertion kind is invalid"
                            )
                    for numeric_field in ("value", "highRank"):
                        if numeric_field in normalized:
                            _strict_int(
                                normalized[numeric_field],
                                f"exact trace project assertion {numeric_field}",
                                minimum=0,
                            )
                if field == "projectDeclarations":
                    if normalized.get("owner") not in _SEATS:
                        raise ValueError(
                            "exact trace project declaration owner is invalid"
                        )
                    if not str(normalized.get("kind") or "").strip():
                        raise ValueError(
                            "exact trace project declaration kind is required"
                        )
                    _strict_int(
                        normalized.get("value"),
                        "exact trace project declaration value",
                        minimum=0,
                    )
                    _strict_int(
                        normalized.get("highRank"),
                        "exact trace project declaration highRank",
                        minimum=0,
                    )
                    _strict_int(
                        normalized.get("declarationOrder"),
                        "exact trace project declaration declarationOrder",
                        minimum=1,
                    )
                if field == "playActions":
                    if "card" not in normalized:
                        raise ValueError("exact trace play action requires card")
                    if type(normalized["card"]) not in {int, str}:
                        raise TypeError(
                            "exact trace play action card must be an integer or string"
                        )
                    if (
                        normalized.get("ekka") is not None
                        and type(normalized["ekka"]) is not bool
                    ):
                        raise TypeError("exact trace play action ekka must be boolean")
                normalized_items.append(normalized)
            sequences[field] = tuple(normalized_items)
        state_assertions = _mapping(
            raw.get("stateAssertions", {}), "exact trace stateAssertions"
        )
        adjudication_events = tuple(
            AuthoredAdjudicationEventV1.build(
                _mapping(item, "exact trace adjudication event")
            )
            for item in _list(
                raw.get("adjudicationEvents", []),
                "exact trace adjudicationEvents",
            )
        )
        event_hashes = [event.sha256 for event in adjudication_events]
        if len(set(event_hashes)) != len(event_hashes):
            raise ValueError("exact trace adjudicationEvents must be unique")
        core = {
            "schemaVersion": 1,
            "deckOrder": list(deck_order),
            "dealer": dealer,
            "roundNumber": round_number,
            "matchScore": list(score),
            **{
                key: [dict(item) for item in values]
                for key, values in sequences.items()
                if key != "projectActions"
            },
            "stateAssertions": state_assertions,
        }
        if adjudication_events:
            core["adjudicationEvents"] = [
                event.to_dict() for event in adjudication_events
            ]
        project_actions_present = "projectActions" in raw
        if project_actions_present:
            core["projectActions"] = [
                dict(item) for item in sequences["projectActions"]
            ]
        digest = canonical_sha256(core)
        if "sha256" in raw:
            declared = raw["sha256"]
            if type(declared) is not str or _HASH.fullmatch(declared) is None:
                raise ValueError(
                    "exact trace prefix sha256 must be a lowercase SHA-256 string"
                )
            if declared != digest:
                raise ValueError("exact trace prefix checksum mismatch")
        return cls(
            deck_order,
            dealer,
            round_number,
            (score[0], score[1]),
            sequences["auctionActions"],
            sequences["escalationActions"],
            sequences["projectDeclarations"],
            sequences["projectAssertions"],
            sequences["playActions"],
            adjudication_events,
            state_assertions,
            digest,
            sequences["projectActions"],
            project_actions_present,
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schemaVersion": 1,
            "deckOrder": list(self.deck_order),
            "dealer": self.dealer,
            "roundNumber": self.round_number,
            "matchScore": list(self.match_score),
            "auctionActions": [
                copy.deepcopy(dict(item)) for item in self.auction_actions
            ],
            "escalationActions": [
                copy.deepcopy(dict(item)) for item in self.escalation_actions
            ],
            "projectDeclarations": [
                copy.deepcopy(dict(item)) for item in self.project_declarations
            ],
            "projectAssertions": [
                copy.deepcopy(dict(item)) for item in self.project_assertions
            ],
            "playActions": [copy.deepcopy(dict(item)) for item in self.play_actions],
            "stateAssertions": copy.deepcopy(dict(self.state_assertions)),
            "sha256": self.sha256,
        }
        if self.adjudication_events:
            result["adjudicationEvents"] = [
                event.to_dict() for event in self.adjudication_events
            ]
        if self.project_actions_present:
            result["projectActions"] = [
                copy.deepcopy(dict(item)) for item in self.project_actions
            ]
        return result


@dataclass(frozen=True, slots=True)
class ScenarioOriginV1:
    kind: str
    exact_trace_prefix: ExactTracePrefixV1 | None
    sha256: str
    recorded_escalation_opportunities: tuple[Mapping[str, str], ...] | None = None

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "ScenarioOriginV1":
        raw = _mapping(value, "scenario origin")
        _closed(raw, {"kind", "exactTracePrefix", "recordedEscalationOpportunities", "sha256"}, "scenario origin")
        kind = str(raw.get("kind") or "")
        if kind not in {"search", "exact_trace_prefix"}:
            raise ValueError("scenario origin kind is unsupported")
        exact: ExactTracePrefixV1 | None = None
        if kind == "exact_trace_prefix":
            exact = ExactTracePrefixV1.build(
                _mapping(raw.get("exactTracePrefix"), "scenario exactTracePrefix")
            )
        elif "exactTracePrefix" in raw:
            raise ValueError("search scenario origin cannot include exactTracePrefix")
        core: dict[str, Any] = {"kind": kind}
        opportunities = None
        if "recordedEscalationOpportunities" in raw:
            if kind != "exact_trace_prefix":
                raise ValueError("recorded escalation opportunities require exact input")
            events = _list(raw["recordedEscalationOpportunities"], "recorded escalation opportunities")
            normalized_events = []
            for event in events:
                row = _mapping(event, "recorded escalation opportunity")
                if set(row) != {"actor", "actionCode"}:
                    raise ValueError("recorded escalation opportunity requires actor and actionCode")
                if type(row["actor"]) is not str or row["actor"] not in _SEATS:
                    raise ValueError("recorded escalation opportunity actor is invalid")
                code = row["actionCode"]
                if type(code) is not str or not code or code != code.strip():
                    raise ValueError("recorded escalation opportunity actionCode must be canonical")
                normalized_events.append({"actor": row["actor"], "actionCode": code})
            opportunities = tuple(normalized_events)
            core["recordedEscalationOpportunities"] = normalized_events
        if exact is not None:
            core["exactTracePrefix"] = exact.to_dict()
        digest = canonical_sha256(core)
        declared = str(raw.get("sha256") or "")
        if declared and declared != digest:
            raise ValueError("scenario origin checksum mismatch")
        return cls(kind, exact, digest, opportunities)

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"kind": self.kind, "sha256": self.sha256}
        if self.exact_trace_prefix is not None:
            value["exactTracePrefix"] = self.exact_trace_prefix.to_dict()
        if self.recorded_escalation_opportunities is not None:
            value["recordedEscalationOpportunities"] = [dict(event) for event in self.recorded_escalation_opportunities]
        return value


@dataclass(frozen=True, slots=True)
class ScenarioProofRequestV1:
    mode: str
    visibility: str
    strategy_class: str | None
    claim: str
    target_team: str
    utility: str
    success_threshold: float
    horizon: Mapping[str, Any] | None
    boundary_evaluator: str | None
    line_universe: str | None
    policies: Mapping[str, Any] | None
    selected_action: str | None
    budget_profile: str
    sha256: str
    utility_encoding: str | None = None

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "ScenarioProofRequestV1":
        raw = _mapping(value, "scenario proof")
        _closed(
            raw,
            {
                "schemaVersion",
                "mode",
                "visibility",
                "strategyClass",
                "claim",
                "targetTeam",
                "utility",
                "successThreshold",
                "utilityEncoding",
                "horizon",
                "boundaryEvaluator",
                "lineUniverse",
                "policies",
                "selectedAction",
                "budgetProfile",
                "sha256",
            },
            "scenario proof",
        )
        if raw.get("schemaVersion") != 1 or type(raw.get("schemaVersion")) is not int:
            raise ValueError("scenario proof schemaVersion must be integer 1")
        mode = str(raw.get("mode") or "")
        visibility = str(raw.get("visibility") or "")
        strategy_class = raw.get("strategyClass")
        claim = str(raw.get("claim") or "")
        target_team = str(raw.get("targetTeam") or "")
        utility = str(raw.get("utility") or "").strip()
        if mode not in PROOF_MODES:
            raise ValueError("scenario proof mode is unsupported")
        if visibility not in PROOF_VISIBILITIES:
            raise ValueError("scenario proof visibility is unsupported")
        if strategy_class is not None:
            strategy_class = str(strategy_class)
            if strategy_class not in PROOF_STRATEGY_CLASSES:
                raise ValueError("scenario proof strategyClass is unsupported")
        expected_strategy_class = (
            "frozen_policy" if mode == "policy_conditioned" else "pure_deterministic"
        )
        if visibility == "actor_information_set" and strategy_class is None:
            raise ValueError(
                "actor-information-set proof requires strategyClass"
            )
        if strategy_class is not None and strategy_class != expected_strategy_class:
            raise ValueError(
                "scenario proof strategyClass is incompatible with proof mode"
            )
        if claim not in PROOF_CLAIMS:
            raise ValueError("scenario proof claim is unsupported")
        if target_team not in _TEAMS:
            raise ValueError("scenario proof targetTeam is invalid")
        if utility not in PROOF_UTILITIES:
            raise ValueError("scenario proof utility is unsupported")
        utility_encoding = raw.get("utilityEncoding")
        if utility_encoding is not None and (utility != "contract_success"
                or utility_encoding not in PROOF_UTILITY_ENCODINGS[utility] or type(utility_encoding) is not str):
            raise ValueError("utilityEncoding requires contract_success signed_win_loss_v1")
        threshold = _strict_number(
            raw.get("successThreshold", 0.0), "scenario proof successThreshold"
        )
        horizon = None
        boundary = None
        policies = None
        if mode == "bounded":
            horizon = _mapping(raw.get("horizon"), "bounded proof horizon")
            _closed(horizon, {"unit", "value"}, "bounded proof horizon")
            if str(horizon.get("unit") or "") not in PROOF_HORIZON_UNITS:
                raise ValueError("bounded proof horizon unit is unsupported")
            _strict_int(horizon.get("value"), "bounded proof horizon value", minimum=1)
            boundary = str(raw.get("boundaryEvaluator") or "").strip()
            if boundary not in PROOF_BOUNDARY_EVALUATORS:
                raise ValueError("bounded proof boundaryEvaluator is unsupported")
            definition = PROOF_BOUNDARY_EVALUATOR_DEFINITIONS[boundary]
            if (horizon["unit"] not in definition["horizonUnits"]
                or ("horizonValue" in definition and horizon["value"] != definition["horizonValue"])
                or ("utilities" in definition and utility not in definition["utilities"])):
                raise ValueError("bounded proof boundary evaluator does not support this horizon or utility")
        elif "horizon" in raw or "boundaryEvaluator" in raw:
            raise ValueError(
                "horizon and boundaryEvaluator are valid only for bounded proof"
            )
        if mode == "policy_conditioned":
            policies = _mapping(
                raw.get("policies"), "policy-conditioned proof policies"
            )
            if set(policies) != _SEATS:
                raise ValueError(
                    "policy-conditioned proof requires one policy for every seat"
                )
        elif "policies" in raw:
            raise ValueError("policies are valid only for policy-conditioned proof")
        line_universe = raw.get("lineUniverse")
        if line_universe is not None:
            line_universe = str(line_universe)
            if line_universe not in LINE_UNIVERSES:
                raise ValueError("scenario proof lineUniverse is unsupported")
        if claim == "winning_continuation_line_count" and line_universe is None:
            raise ValueError("continuation-line proof requires lineUniverse")
        selected_action = raw.get("selectedAction")
        if selected_action is not None:
            if type(selected_action) is not str:
                raise ValueError("scenario proof selectedAction must be a string")
            selected_action = selected_action.strip()
            if not selected_action:
                raise ValueError("scenario proof selectedAction cannot be empty")
        if claim == "selected_action_is_unique_winner" and selected_action is None:
            raise ValueError("selected_action_is_unique_winner requires selectedAction")
        budget_profile = str(raw.get("budgetProfile") or "").strip()
        if not budget_profile:
            raise ValueError("scenario proof budgetProfile is required")
        core = {
            "schemaVersion": 1,
            "mode": mode,
            "visibility": visibility,
            "claim": claim,
            "targetTeam": target_team,
            "utility": utility,
            "successThreshold": threshold,
            "budgetProfile": budget_profile,
        }
        if strategy_class is not None:
            core["strategyClass"] = strategy_class
        if utility_encoding is not None:
            core["utilityEncoding"] = utility_encoding
        if horizon is not None:
            core.update({"horizon": horizon, "boundaryEvaluator": boundary})
        if line_universe is not None:
            core["lineUniverse"] = line_universe
        if policies is not None:
            core["policies"] = policies
        if selected_action is not None:
            core["selectedAction"] = selected_action
        digest = canonical_sha256(core)
        declared = str(raw.get("sha256") or "")
        if declared and declared != digest:
            raise ValueError("scenario proof checksum mismatch")
        return cls(
            mode,
            visibility,
            strategy_class,
            claim,
            target_team,
            utility,
            threshold,
            horizon,
            boundary,
            line_universe,
            policies,
            selected_action,
            budget_profile,
            digest,
            utility_encoding,
        )

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schemaVersion": 1,
            "mode": self.mode,
            "visibility": self.visibility,
            "claim": self.claim,
            "targetTeam": self.target_team,
            "utility": self.utility,
            "successThreshold": self.success_threshold,
            "budgetProfile": self.budget_profile,
            "sha256": self.sha256,
        }
        if self.strategy_class is not None:
            value["strategyClass"] = self.strategy_class
        if self.utility_encoding is not None:
            value["utilityEncoding"] = self.utility_encoding
        if self.horizon is not None:
            value.update(
                {
                    "horizon": copy.deepcopy(dict(self.horizon)),
                    "boundaryEvaluator": self.boundary_evaluator,
                }
            )
        if self.line_universe is not None:
            value["lineUniverse"] = self.line_universe
        if self.policies is not None:
            value["policies"] = copy.deepcopy(dict(self.policies))
        if self.selected_action is not None:
            value["selectedAction"] = self.selected_action
        return value


_NATIVE_OBSERVED_LIMIT_FIELDS = (
    ("maximumIterations", "maximum_iterations"),
    ("maximumQueries", "maximum_queries"),
    ("maximumObservations", "maximum_observations"),
    ("maximumBoundPrefixes", "maximum_bound_prefixes"),
    ("maximumCachedPrefixesPerBranch", "maximum_cached_prefixes_per_branch"),
)


@dataclass(frozen=True, slots=True)
class NativeObservedProofLimitsV1:
    maximum_iterations: int
    maximum_queries: int
    maximum_observations: int
    maximum_bound_prefixes: int
    maximum_cached_prefixes_per_branch: int
    counterexample_objective: str = "feasibility"
    play_allocation_method: str = "deterministic_cp_sat_prefix_v1"

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """Closed profile-editor schema without importing the service runtime."""
        properties = {key: {"type": "integer", "minimum": 1, "maximum": 2**31 - 1}
                      for key, _ in _NATIVE_OBSERVED_LIMIT_FIELDS}
        properties.update(schemaVersion={"type": "integer", "enum": [1, 2, 3]},
                          counterexampleObjective={"type": "string", "enum": ["minimum_utility"]},
                          playAllocationMethod={"type": "string", "enum": ["deterministic_cp_sat_batched_prefix_v1"]})
        return {"type": "object", "additionalProperties": False, "properties": properties,
            "required": ["schemaVersion", *(key for key, _ in _NATIVE_OBSERVED_LIMIT_FIELDS)],
            "oneOf": [
                {"properties": {"schemaVersion": {"const": 1}},
                 "not": {"anyOf": [{"required": ["counterexampleObjective"]}, {"required": ["playAllocationMethod"]}]}},
                {"properties": {"schemaVersion": {"const": 2}},
                 "required": ["counterexampleObjective"], "not": {"required": ["playAllocationMethod"]}},
                {"properties": {"schemaVersion": {"const": 3}},
                 "required": ["counterexampleObjective", "playAllocationMethod"]}]}

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "NativeObservedProofLimitsV1":
        raw = _mapping(value, "native observed proof limits")
        expected = {"schemaVersion", *(key for key, _ in _NATIVE_OBSERVED_LIMIT_FIELDS)}
        if raw.get("schemaVersion") in {2, 3}:
            expected.add("counterexampleObjective")
        if raw.get("schemaVersion") == 3:
            expected.add("playAllocationMethod")
        if (set(raw) != expected
                or type(raw["schemaVersion"]) is not int or raw["schemaVersion"] not in {1, 2, 3}):
            raise ValueError("native observed proof limits shape or version is invalid")
        objective = raw.get("counterexampleObjective", "feasibility")
        if objective not in ("feasibility", "minimum_utility") or (raw["schemaVersion"] in {2, 3} and objective != "minimum_utility"):
            raise ValueError("native observed counterexample objective is invalid or noncanonical")
        allocation = raw.get("playAllocationMethod", "deterministic_cp_sat_prefix_v1")
        if raw["schemaVersion"] == 3 and allocation != "deterministic_cp_sat_batched_prefix_v1":
            raise ValueError("native observed play allocation method is invalid")
        values = tuple(raw[key] for key, _ in _NATIVE_OBSERVED_LIMIT_FIELDS)
        if any(type(item) is not int or not 0 < item <= 2**31 - 1 for item in values):
            raise ValueError("native observed proof limits must be positive bounded integers")
        return cls(*values, counterexample_objective=objective, play_allocation_method=allocation)

    def to_dict(self) -> dict[str, Any]:
        batched = self.play_allocation_method == "deterministic_cp_sat_batched_prefix_v1"
        return {"schemaVersion": 3 if batched else (1 if self.counterexample_objective == "feasibility" else 2),
                **({"playAllocationMethod": self.play_allocation_method} if batched else {}),
                **({"counterexampleObjective": self.counterexample_objective}
                   if self.counterexample_objective != "feasibility" else {}),
                **{key: getattr(self, attribute) for key, attribute in _NATIVE_OBSERVED_LIMIT_FIELDS}}


@dataclass(frozen=True, slots=True)
class ScenarioProofBudgetV1:
    profile_id: str
    maximum_nodes: int
    maximum_depth: int
    maximum_horizon_actions: int
    maximum_hidden_worlds: int
    maximum_wall_seconds: float
    maximum_cpu_seconds: float
    maximum_cache_entries: int
    maximum_cache_bytes: int
    cancellation_check_interval_nodes: int
    maximum_workers: int
    maximum_line_count_nodes: int
    maximum_state_space: int
    sha256: str
    native_observed: NativeObservedProofLimitsV1 | None = None
    policy_search: str | None = None

    @classmethod
    def build(cls, profile_id: str, value: Mapping[str, Any]) -> "ScenarioProofBudgetV1":
        raw = dict(value)
        expected = {
            "maximumNodes", "maximumDepth", "maximumHorizonActions", "maximumHiddenWorlds",
            "maximumWallSeconds", "maximumCpuSeconds", "maximumCacheEntries", "maximumCacheBytes",
            "cancellationCheckIntervalNodes", "maximumWorkers", "maximumLineCountNodes", "maximumStateSpace",
        }
        if not expected <= set(raw) <= expected | {"nativeObserved", "policySearch"}:
            raise ValueError(f"proof budget {profile_id} shape is invalid")
        if "policySearch" in raw and raw["policySearch"] != "time_budgeted_symbolic_v1":
            raise ValueError(f"proof budget {profile_id} policy search is unsupported")
        integers = {key: raw[key] for key in expected if key not in {"maximumWallSeconds", "maximumCpuSeconds"}}
        if any(type(value) is not int or value < 1 for value in integers.values()):
            raise ValueError(f"proof budget {profile_id} integer bounds are invalid")
        wall = raw["maximumWallSeconds"]
        if type(wall) not in {int, float} or not math.isfinite(wall) or wall <= 0:
            raise ValueError(f"proof budget {profile_id} wall bound is invalid")
        cpu = raw["maximumCpuSeconds"]
        if type(cpu) not in {int, float} or not math.isfinite(cpu) or cpu <= 0:
            raise ValueError(f"proof budget {profile_id} CPU bound is invalid")
        return cls(
            profile_id, int(raw["maximumNodes"]), int(raw["maximumDepth"]),
            int(raw["maximumHorizonActions"]), int(raw["maximumHiddenWorlds"]), float(wall), float(cpu),
            int(raw["maximumCacheEntries"]), int(raw["maximumCacheBytes"]),
            int(raw["cancellationCheckIntervalNodes"]), int(raw["maximumWorkers"]),
            int(raw["maximumLineCountNodes"]), int(raw["maximumStateSpace"]),
            canonical_sha256({"profileId": profile_id, **raw}),
            NativeObservedProofLimitsV1.build(raw["nativeObserved"]) if "nativeObserved" in raw else None,
            raw.get("policySearch"),
        )


NATIVE_BID_WORLD_RANKING_IDS = ("opponent_auction_value_v1",)
NATIVE_BID_REFUTATION_AUCTION_ACTIONS = ("pass", "sun")
NATIVE_BID_INFORMATION_SET_REFUTATION_METHOD = "information_set_refutation_v1"


def build_native_bid_refutation_probe_v1(value, *, maximum_seconds):
    """One budgeted counterworld proposal; configuration is never proof."""
    raw = _mapping(value, "bid threshold refutation")
    fields = {"schemaVersion", "maximumSeconds", "maximumQueries", "worldRankingId", "opponentAuctionAction"}
    if (set(raw) != fields or type(raw["schemaVersion"]) is not int or raw["schemaVersion"] != 1
            or raw["worldRankingId"] not in NATIVE_BID_WORLD_RANKING_IDS
            or raw["opponentAuctionAction"] not in NATIVE_BID_REFUTATION_AUCTION_ACTIONS
            or type(raw["maximumQueries"]) is not int or not 0 < raw["maximumQueries"] < 2**31
            or type(raw["maximumSeconds"]) not in {int, float}
            or not math.isfinite(raw["maximumSeconds"]) or not 0 < raw["maximumSeconds"] < maximum_seconds):
        raise ValueError("bid threshold refutation requires a closed native policy and bounded sub-budget")
    return {**copy.deepcopy(dict(raw)), "maximumSeconds": float(raw["maximumSeconds"])}


def build_native_bid_alternative_search_v1(value, *, maximum_seconds):
    """Closed, bounded proposal configuration; upperBound is not proof evidence."""
    raw = _mapping(value, "bid alternative proof")
    if type(raw.get("schemaVersion")) is int and raw["schemaVersion"] == 2:
        if (set(raw) != {"schemaVersion", "method", "upperBound", "maximumSeconds", "maximumQueries"}
                or raw["method"] != NATIVE_BID_INFORMATION_SET_REFUTATION_METHOD
                or type(raw["upperBound"]) is not int or not -(2**31) <= raw["upperBound"] < 2**31
                or type(raw["maximumQueries"]) is not int or not 0 < raw["maximumQueries"] < 2**31
                or type(raw["maximumSeconds"]) not in {int, float}
                or not math.isfinite(raw["maximumSeconds"])
                or not 0 < raw["maximumSeconds"] < maximum_seconds):
            raise ValueError("bid information-set refutation requires closed configuration and bounded resources")
        return {**copy.deepcopy(dict(raw)), "maximumSeconds": float(raw["maximumSeconds"])}
    fields = {"schemaVersion", "upperBound", "maximumWorlds", "maximumSeconds",
              "maximumQueries", "maximumWorldSeconds", "maximumWorldQueries"}
    if (not fields <= set(raw) <= fields | {"worldRankingId"}
            or type(raw["schemaVersion"]) is not int or raw["schemaVersion"] != 1):
        raise ValueError("bid alternative proof requires its closed versioned configuration")
    if "worldRankingId" in raw and raw["worldRankingId"] not in NATIVE_BID_WORLD_RANKING_IDS:
        raise ValueError("bid alternative proof worldRankingId must name a registered native ordering")
    if type(raw["upperBound"]) is not int or not -(2**31) <= raw["upperBound"] < 2**31:
        raise ValueError("bid alternative proof upperBound must be a bounded integer")
    for key in ("maximumWorlds", "maximumQueries", "maximumWorldQueries"):
        if type(raw[key]) is not int or not 0 < raw[key] < 2**31:
            raise ValueError("bid alternative proof query/world budgets must be positive integers")
    for key in ("maximumSeconds", "maximumWorldSeconds"):
        if type(raw[key]) not in {int, float} or not math.isfinite(raw[key]) or raw[key] <= 0:
            raise ValueError("bid alternative proof time budgets must be finite and positive")
    if (not raw["maximumWorldSeconds"] <= raw["maximumSeconds"] < maximum_seconds
            or raw["maximumWorldQueries"] > raw["maximumQueries"]):
        raise ValueError("bid alternative proof must fit its parent and per-world budgets")
    return {**copy.deepcopy(dict(raw)), "maximumSeconds": float(raw["maximumSeconds"]),
            "maximumWorldSeconds": float(raw["maximumWorldSeconds"])}


def native_bid_nonplay_policy_schema_v1() -> dict[str, Any]:
    """Closed, complete-state incumbent aid; never an optimal-policy claim."""
    from .ortools_project_fragment import NATIVE_FULL_HAND_PROJECT_POLICY_ID
    return {"type": "object", "additionalProperties": False,
        "required": ["schemaVersion", "auctionAction", "escalationAction", "projectPolicyId"],
        "properties": {"schemaVersion": {"type": "integer", "const": 1},
            "auctionAction": {"type": "string", "enum": ["pass"]},
            "escalationAction": {"type": "string", "enum": ["decline"]},
            "projectPolicyId": {"type": "string", "enum": [NATIVE_FULL_HAND_PROJECT_POLICY_ID]}}}


def validate_native_bid_nonplay_policy_v1(value) -> dict[str, Any]:
    raw = _mapping(value, "complete-state bid nonplay policy")
    properties = native_bid_nonplay_policy_schema_v1()["properties"]
    if set(raw) != set(properties) or any(
            type(raw[key]) is not (int if spec["type"] == "integer" else str)
            or raw[key] not in spec.get("enum", [spec.get("const")])
            for key, spec in properties.items()):
        raise ValueError("complete-state bid nonplay policy requires its closed native configuration")
    return copy.deepcopy(dict(raw))


def scenario_video_interestingness_schema_v1(version: int | None = None) -> dict[str, Any]:
    """Closed historical policies and V4 actor-visible content admission."""
    properties = {
        "schemaVersion": {"type": "integer", "const": 1},
        "maximumFinalMargin": {"type": "integer", "minimum": 0, "maximum": 2**31 - 1},
        "minimumPointSwing": {"type": "integer", "minimum": 20, "maximum": 2**31 - 1},
        "maximumFeaturedTrick": {"type": "integer", "minimum": 1, "maximum": 5},
        "minimumMeaningfulDecisions": {"type": "integer", "minimum": 2, "maximum": 128},
    }
    legacy = {"type": "object", "additionalProperties": False,
              "required": list(properties), "properties": properties}
    if version == 1:
        return legacy
    raw_points = copy.deepcopy(legacy)
    raw_properties = raw_points["properties"]
    raw_properties["schemaVersion"]["const"] = 2
    raw_properties["minimumPointSwing"]["minimum"] = 15
    raw_properties["maximumFeaturedTrick"]["maximum"] = 6
    raw_properties["minimumMeaningfulDecisions"]["minimum"] = 1
    if version == 2:
        return raw_points
    late_tail = copy.deepcopy(legacy)
    late_tail["properties"]["schemaVersion"]["const"] = 3
    late_tail["properties"]["maximumFeaturedTrick"]["maximum"] = 7
    late_tail["properties"]["minimumMeaningfulDecisions"]["minimum"] = 1
    if version == 3:
        return late_tail
    actor_interest = copy.deepcopy(late_tail)
    actor_properties = actor_interest["properties"]
    actor_properties["schemaVersion"]["const"] = 4
    actor_properties.update({
        "minimumConsistentWorlds": {"type": "integer", "minimum": 2, "maximum": 256},
        "minimumBaselineRegret": {"type": "integer", "minimum": 1, "maximum": 2**31 - 1},
        "minimumNextBestRegret": {"type": "integer", "minimum": 1, "maximum": 2**31 - 1},
        "baselinePolicyId": {"type": "string", "enum": ["observer_safe_trick_control_v1",
            "observer_safe_partner_feed_established_control_v1", "observer_safe_public_partner_entry_v2",
            "observer_safe_public_stranded_master_v3", "observer_safe_public_master_retention_v4",
            "observer_safe_public_exhausted_suit_v5", "observer_safe_public_two_trick_control_v6"]},
        "allowTieRuleOnlyLessons": {"type": "boolean"},
    })
    actor_interest["required"] = list(actor_properties)
    if version == 4:
        return actor_interest
    if version is not None:
        raise ValueError("video interestingness requires schemaVersion 1, 2, 3 or 4")
    combined = copy.deepcopy(raw_points)
    combined["properties"].update({name: spec for name, spec in actor_properties.items()
                                   if name not in properties})
    combined["required"] = list(properties)
    combined["properties"]["maximumFeaturedTrick"]["maximum"] = 7
    combined["properties"]["schemaVersion"] = {"type": "integer", "enum": [1, 2, 3, 4]}
    combined["oneOf"] = [legacy, raw_points, late_tail, actor_interest]
    return combined


def build_scenario_video_interestingness_v1(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(value, "video interestingness")
    version = raw.get("schemaVersion")
    if type(version) is not int or version not in {1, 2, 3, 4}:
        raise ValueError("video interestingness requires schemaVersion 1, 2, 3 or 4")
    properties = scenario_video_interestingness_schema_v1(version)["properties"]
    _closed(raw, set(properties), "video interestingness")
    if set(raw) != set(properties):
        raise ValueError("video interestingness requires every governed threshold")
    for name, spec in properties.items():
        expected_type = {"integer": int, "string": str, "boolean": bool}[spec["type"]]
        if (type(raw[name]) is not expected_type or ("const" in spec and raw[name] != spec["const"])
                or ("enum" in spec and raw[name] not in spec["enum"])
                or ("minimum" in spec and raw[name] < spec["minimum"])
                or ("maximum" in spec and raw[name] > spec["maximum"])):
            raise ValueError(f"video interestingness {name} is outside its governed bounds")
    return copy.deepcopy(dict(raw))


NATIVE_FAMILY_OPENING_MODE = 'native_project_auction_v1'
NATIVE_FAMILY_PROPOSAL_MODE = 'native_tail_first_v1'


def family_opening_mode_schema_v1():
    return {'type': 'string', 'enum': [NATIVE_FAMILY_OPENING_MODE]}


def family_proposal_mode_schema_v1():
    return {'type': 'string', 'enum': [NATIVE_FAMILY_PROPOSAL_MODE]}


def build_family_opening_evidence_v1(value, *, prefix, rules_hash, source_seed):
    """Bind allocation and fresh auction provenance, never substitute for trace qualification."""
    raw = copy.deepcopy(dict(_mapping(value, 'family opening evidence')))
    digest = raw.pop('sha256', None)
    fields = {'qualified', 'mode', 'status', 'initialPrefix', 'allocation', 'rulesHash', 'seed',
              'familyHash', 'expressionHash', 'constructionNarrowing', 'auctionQualification'}
    if (set(raw) != fields or digest != canonical_sha256(raw)
            or raw['qualified'] is not False or raw['status'] != 'constructed'
            or raw['mode'] != NATIVE_FAMILY_OPENING_MODE or raw['rulesHash'] != rules_hash
            or type(raw['seed']) is not int or raw['seed'] != source_seed):
        raise ValueError('native family opening evidence identity mismatch')
    initial = ExactTracePrefixV1.build(raw['initialPrefix'])
    auction = dict(_mapping(raw['auctionQualification'], 'native family auction evidence'))
    auction_hash = auction.pop('sha256', None)
    allocation = _mapping(raw['allocation'], 'native family allocation')
    allocation_fields = {'schemaVersion', 'status', 'solverStatus', 'dealer', 'buyer', 'deckOrder',
        'positionOfCard', 'ownerAtPosition', 'ownerOfCard', 'purchasePosition', 'purchaseCard', 'purchaseSuit',
        'noGoodHashes', 'minimumDeckHammingDistance', 'modelHash', 'solutionHash', 'branches', 'conflicts',
        'variableCount', 'constraintCount', 'auction'}
    if not allocation_fields <= set(allocation) or set(allocation) - allocation_fields - {'purchaseRecipient'}:
        raise ValueError('native family allocation fields mismatch')
    narrowing = _mapping(raw['constructionNarrowing'], 'native family narrowing')
    narrowing_fields = {'visibleHandAuctionPolicy', 'unknownRevealLeader', 'projectCardOwners'}
    historical_scope = set(narrowing) == narrowing_fields and narrowing.get('unknownRevealLeader') is True
    declaration_scope = (set(narrowing) == narrowing_fields | {'projectScope'}
        and narrowing.get('projectScope') == 'project_declarations' and narrowing.get('unknownRevealLeader') is False)
    if narrowing.get('visibleHandAuctionPolicy') is not True or not (historical_scope or declaration_scope):
        raise ValueError('native family construction narrowing mismatch')
    anchors = _mapping(narrowing['projectCardOwners'], 'native family project anchors')
    if any(card not in {str(index) for index in range(32)} or owner not in {'north', 'west', 'south', 'east'}
           or allocation['ownerOfCard'][int(card)] != owner for card, owner in anchors.items()):
        raise ValueError('native family construction anchors disagree with solved ownership')
    final = ExactTracePrefixV1.build(auction['finalPrefix'])
    winning_bid = next((row for row in reversed(final.auction_actions) if row['action'] != 'pass'), {})
    fresh_roles = {'buyer': winning_bid.get('actor'), 'contract': winning_bid.get('action'),
        'trumpSuit': (winning_bid.get('trumpSuit') or allocation.get('purchaseSuit'))
                     if winning_bid.get('action') == 'hokum' else None}
    if (initial.auction_actions or initial.escalation_actions or initial.project_actions
            or initial.project_declarations or initial.project_assertions or initial.play_actions
            or initial.adjudication_events or initial.state_assertions or initial.round_number != 1
            or any((item.deck_order, item.dealer, item.round_number, item.match_score)
                   != (initial.deck_order, initial.dealer, initial.round_number, initial.match_score)
                   for item in (prefix, final))
            or prefix.auction_actions != final.auction_actions
            or final.escalation_actions or final.project_actions or final.project_declarations
            or final.project_assertions or final.play_actions or final.adjudication_events or final.state_assertions
            or auction_hash != canonical_sha256(auction) or 'executionReceipt' in auction
            or auction.get('qualificationAuthority') != 'cp_sat' or auction.get('operation') != 'generate'
            or auction.get('status') != 'complete' or auction.get('partialReceipt') is not None
            or auction.get('coveredFromBidIndex') != 0 or auction.get('rulesHash') != rules_hash
            or auction.get('sourcePrefixHash') != initial.sha256
            or not auction.get('decisions') or any(row.get('actualActionOptimal') is not True for row in auction['decisions'])
            or allocation.get('status') != 'satisfied' or allocation.get('solverStatus') not in {'feasible', 'optimal'}
            or allocation.get('deckOrder') != list(initial.deck_order) or allocation.get('dealer') != initial.dealer
            or allocation.get('buyer') != allocation.get('auction', {}).get('completion', {}).get('buyer')
            or fresh_roles['contract'] not in {'sun', 'hokum'}
            or any(value != allocation.get('auction', {}).get('completion', {}).get(key)
                   for key, value in fresh_roles.items())
            or any(auction.get('auctionOutcome', {}).get(key) != allocation['auction']['completion'].get(key)
                   for key in ('buyer', 'contract'))):
        raise ValueError('native family opening ownership mismatch')
    from baloot_platform.core.cards import Card
    core = {key: allocation[key] for key in ('schemaVersion', 'modelHash', 'dealer', 'buyer', 'deckOrder',
        'positionOfCard', 'ownerAtPosition', 'ownerOfCard', 'purchasePosition', 'purchaseSuit',
        'noGoodHashes', 'minimumDeckHammingDistance', 'auction')}
    core['purchaseCard'] = Card.parse(allocation['purchaseCard']).id
    if 'purchaseRecipient' in allocation:
        core['purchaseRecipient'] = allocation['purchaseRecipient']
    if canonical_sha256(core) != allocation.get('solutionHash'):
        raise ValueError('native family allocation solution checksum mismatch')
    return {**raw, 'sha256': digest}


@dataclass(frozen=True, slots=True)
class ScenarioQualityExecutionBudgetV1:
    maximum_wall_seconds: float
    maximum_evidence_bytes: int
    maximum_escalation_decisions: int
    sha256: str
    escalation_opportunity_order: tuple[str, ...] | None = None
    maximum_hidden_worlds_per_decision: int | None = None
    early_play_policy_id: str | None = None
    exact_play_from_trick: int | None = None
    featured_play_native_proof: bool = False
    bid_incumbent_proof: Mapping[str, Any] | None = None
    allow_exact_input_quality: bool = False
    complete_state_play_policy_pair: bool = False
    prune_own_project_pass_worlds: bool = False
    bid_opponent_play_policy_id: str | None = None
    bid_alternative_proof: Mapping[str, Any] | None = None
    bid_threshold_refutation: Mapping[str, Any] | None = None
    observed_bid_strategy_play_fallback_id: str | None = None
    audit_constructed_trace: bool = False
    exact_tail_proof: bool = False
    video_interestingness: Mapping[str, Any] | None = None
    candidate_source: str | None = None
    proposal_seed_count: int | None = None
    proposal_maximum_seconds: float | None = None
    proposal_play_indices: tuple[int, ...] | None = None
    accepted_lesson_library: Mapping[str, int | list[str]] | None = None
    play_index_world_caps: Mapping[str, Any] | None = None
    family_target: Mapping[str, Any] | None = None
    family_catalog: Mapping[str, Any] | None = None
    family_opening_mode: str | None = None
    family_proposal_mode: str | None = None
    exact_tail_budget_profile: str | None = None

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "ScenarioQualityExecutionBudgetV1":
        raw = _mapping(value, "quality execution budget")
        fields = {"schemaVersion", "maximumWallSeconds", "maximumEvidenceBytes", "maximumEscalationDecisions"}
        _closed(raw, fields | {"sha256", "familyTarget", "familyCatalog", "familyOpeningMode", "familyProposalMode", "candidateSource", "proposalSeedCount", "proposalMaximumSeconds", "proposalPlayIndices", "acceptedLessonLibrary", "escalationOpportunityOrder", "earlyPlayPolicyId",
                               "exactPlayFromTrick", "maximumHiddenWorldsPerDecision", "playIndexWorldCaps",
                               "featuredPlayNativeProof", "bidIncumbentProof", "allowExactInputQuality",
                               "completeStatePlayPolicyPair", "pruneOwnProjectPassWorlds", "bidOpponentPlayPolicyId",
                               "bidAlternativeProof", "bidThresholdRefutation", "observedBidStrategyPlayFallbackId", "auditConstructedTrace", "exactTailProof", "exactTailBudgetProfile", "videoInterestingness"},
                "quality execution budget")
        if fields - set(raw) or type(raw.get("schemaVersion")) is not int or raw["schemaVersion"] != 1:
            raise ValueError("quality execution budget requires schemaVersion 1 and all limits")
        seconds = raw["maximumWallSeconds"]
        if type(seconds) not in {int, float} or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("quality execution maximumWallSeconds must be finite and positive")
        for field in ("maximumEvidenceBytes", "maximumEscalationDecisions"):
            if type(raw[field]) is not int or not 0 < raw[field] <= 2**31 - 1:
                raise ValueError(f"quality execution {field} must be a positive bounded integer")
        core = {key: raw[key] for key in fields}
        audit_constructed = raw.get("auditConstructedTrace", False)
        if type(audit_constructed) is not bool:
            raise ValueError("quality auditConstructedTrace must be boolean")
        if audit_constructed:
            core["auditConstructedTrace"] = True
        core["maximumWallSeconds"] = float(seconds)
        order = raw.get("escalationOpportunityOrder")
        if "escalationOpportunityOrder" in raw:
            if (type(order) is not list or len(order) != len(_SEATS)
                    or any(type(seat) is not str for seat in order) or set(order) != _SEATS):
                raise ValueError("quality escalationOpportunityOrder must contain every canonical seat once")
            core["escalationOpportunityOrder"] = list(order)
        world_limit = raw.get("maximumHiddenWorldsPerDecision")
        bands = None
        if "playIndexWorldCaps" in raw:
            bands = _mapping(raw["playIndexWorldCaps"], "play index world caps")
            if (set(bands) != {"schemaVersion", "bands"}
                    or type(bands["schemaVersion"]) is not int or bands["schemaVersion"] != 1
                    or type(bands["bands"]) is not list or not bands["bands"]):
                raise ValueError("play index world caps requires closed schemaVersion 1 and nonempty bands")
            covered = set()
            for band in bands["bands"]:
                if (not isinstance(band, Mapping)
                        or set(band) != {"firstPlayIndex", "lastPlayIndex", "maximumHiddenWorlds"}
                        or any(type(band[key]) is not int for key in band)
                        or not 0 <= band["firstPlayIndex"] <= band["lastPlayIndex"] <= 31
                        or not 0 < band["maximumHiddenWorlds"] <= 2**31 - 1):
                    raise ValueError("play index world cap requires integer bounds in 0..31 and a positive world cap")
                indices = set(range(band["firstPlayIndex"], band["lastPlayIndex"] + 1))
                if covered & indices:
                    raise ValueError("play index world cap bands must not overlap")
                covered.update(indices)
            bands = copy.deepcopy(dict(bands))
            bands["bands"].sort(key=lambda band: band["firstPlayIndex"])
            core["playIndexWorldCaps"] = bands
        if world_limit is not None:
            if type(world_limit) is not int or not 0 < world_limit <= 2**31 - 1:
                raise ValueError(
                    "quality execution maximumHiddenWorldsPerDecision must be a positive bounded integer"
                )
            core["maximumHiddenWorldsPerDecision"] = world_limit
        policy_id = raw.get("earlyPlayPolicyId")
        exact_from = raw.get("exactPlayFromTrick")
        if exact_from is None:
            if policy_id is not None:
                raise ValueError("quality early-play policy requires an exact boundary")
        else:
            if type(exact_from) is not int or not 1 <= exact_from <= 8:
                raise ValueError("quality exactPlayFromTrick must be an integer from 1 through 8")
            if (exact_from == 1 and "earlyPlayPolicyId" in raw
                    or exact_from > 1 and policy_id is None):
                raise ValueError("quality early-play policy must cover exactly the tricks before native analysis")
            from .ortools_play_tail import NATIVE_OBSERVER_SAFE_PLAY_POLICY_IDS
            if policy_id is not None and (
                type(policy_id) is not str or policy_id not in NATIVE_OBSERVER_SAFE_PLAY_POLICY_IDS
            ):
                raise ValueError("quality earlyPlayPolicyId is unsupported")
            if policy_id is not None:
                core["earlyPlayPolicyId"] = policy_id
            core["exactPlayFromTrick"] = exact_from
        exact_tail = raw.get("exactTailProof", False)
        if type(exact_tail) is not bool:
            raise ValueError("quality exactTailProof must be boolean")
        if exact_tail:
            if exact_from is None or exact_from <= 1 or policy_id is None:
                raise ValueError("quality exactTailProof requires a declared late boundary and early policy")
            core["exactTailProof"] = True
        tail_budget = raw.get("exactTailBudgetProfile")
        if "exactTailBudgetProfile" in raw:
            if (not exact_tail or type(tail_budget) is not str or not tail_budget
                    or tail_budget != tail_budget.strip()):
                raise ValueError("quality exactTailBudgetProfile requires exactTailProof and a canonical profile ID")
            core["exactTailBudgetProfile"] = tail_budget
        featured = raw.get("featuredPlayNativeProof", False)
        if type(featured) is not bool:
            raise ValueError("quality featuredPlayNativeProof must be boolean")
        if featured:
            core["featuredPlayNativeProof"] = True
        incumbent = None
        if "bidIncumbentProof" in raw:
            from .ortools_proof import DeclarativeActionPolicyV1
            incumbent = _mapping(raw["bidIncumbentProof"], "bid incumbent proof")
            version = incumbent.get("schemaVersion")
            policy_field = "nativePlayPolicyId" if version == 2 else "playPolicies"
            fields = {"schemaVersion", "maximumSeconds", "maximumQueries", policy_field}
            optional = {"observedProjectPolicyId", "observedAuctionPolicyId", "completeStateNonplayPolicy",
                        "completeStateAlternativeFirst"} if version == 2 else set()
            if (not fields <= set(incumbent) <= fields | optional
                    or type(version) is not int or version not in {1, 2}):
                raise ValueError("bid incumbent proof requires its closed versioned configuration")
            allowance = incumbent["maximumSeconds"]
            if (type(allowance) not in {int, float} or not math.isfinite(allowance)
                    or not 0 < allowance < seconds
                    or type(incumbent["maximumQueries"]) is not int
                    or not 0 < incumbent["maximumQueries"] <= 2**31 - 1):
                raise ValueError("bid incumbent proof requires positive bounded allowances below the quality deadline")
            if version == 2:
                if "completeStateAlternativeFirst" in incumbent and type(incumbent["completeStateAlternativeFirst"]) is not bool:
                    raise TypeError("complete-state alternative-first selection must be boolean")
                from .ortools_play_tail import native_observer_safe_policy_identity_v1
                from baloot_platform.core.enums import Team
                native_observer_safe_policy_identity_v1(incumbent[policy_field], Team.NORTH_SOUTH)
                if "observedProjectPolicyId" in incumbent:
                    from .ortools_project_fragment import validate_native_observed_project_policy_id_v1
                    validate_native_observed_project_policy_id_v1(incumbent["observedProjectPolicyId"])
                if "observedAuctionPolicyId" in incumbent:
                    from .ortools_scenario_model import validate_native_observed_auction_policy_id_v1
                    validate_native_observed_auction_policy_id_v1(incumbent["observedAuctionPolicyId"])
                if "completeStateNonplayPolicy" in incumbent:
                    validate_native_bid_nonplay_policy_v1(incumbent["completeStateNonplayPolicy"])
            else:
                policies = _mapping(incumbent[policy_field], "bid incumbent play policies")
                if set(policies) != _SEATS:
                    raise ValueError("bid incumbent play policies must cover every seat")
                for seat, policy in policies.items():
                    parsed = DeclarativeActionPolicyV1.build(seat, policy)
                    if parsed.choices or parsed.materialization is None:
                        raise ValueError("bid incumbent play policies must be declarative templates")
            incumbent = copy.deepcopy(dict(incumbent))
            incumbent["maximumSeconds"] = float(allowance)
            core["bidIncumbentProof"] = incumbent
        allow_exact = raw.get("allowExactInputQuality", False)
        if type(allow_exact) is not bool:
            raise ValueError("quality allowExactInputQuality must be boolean")
        if allow_exact:
            core["allowExactInputQuality"] = True
        paired = raw.get("completeStatePlayPolicyPair", False)
        if type(paired) is not bool:
            raise ValueError("quality completeStatePlayPolicyPair must be boolean")
        if paired:
            core["completeStatePlayPolicyPair"] = True
        prune_pass = raw.get("pruneOwnProjectPassWorlds", False)
        if type(prune_pass) is not bool:
            raise ValueError("quality pruneOwnProjectPassWorlds must be boolean")
        if prune_pass:
            core["pruneOwnProjectPassWorlds"] = True
        bid_policy = raw.get("bidOpponentPlayPolicyId")
        if "bidOpponentPlayPolicyId" in raw:
            from .ortools_play_tail import native_observer_safe_policy_identity_v1
            from baloot_platform.core.enums import Team
            native_observer_safe_policy_identity_v1(bid_policy, Team.NORTH_SOUTH)
            core["bidOpponentPlayPolicyId"] = bid_policy
        strategy_fallback = raw.get("observedBidStrategyPlayFallbackId")
        if "observedBidStrategyPlayFallbackId" in raw:
            from .ortools_play_tail import native_observer_safe_policy_identity_v1
            from baloot_platform.core.enums import Team
            native_observer_safe_policy_identity_v1(strategy_fallback, Team.NORTH_SOUTH)
            core["observedBidStrategyPlayFallbackId"] = strategy_fallback
        alternative = None
        if "bidAlternativeProof" in raw:
            alternative = build_native_bid_alternative_search_v1(raw["bidAlternativeProof"], maximum_seconds=seconds)
            if bid_policy is not None:
                raise ValueError("synthesized bid bounds cannot also select a fixed opponent play policy")
            if world_limit is not None and alternative.get("maximumWorlds", 0) > world_limit:
                raise ValueError("bid alternative proof exceeds maximumHiddenWorldsPerDecision")
            core["bidAlternativeProof"] = alternative
        refutation = None
        if "bidThresholdRefutation" in raw:
            refutation = build_native_bid_refutation_probe_v1(raw["bidThresholdRefutation"], maximum_seconds=seconds)
            core["bidThresholdRefutation"] = refutation
        interestingness = None
        if "videoInterestingness" in raw:
            interestingness = build_scenario_video_interestingness_v1(raw["videoInterestingness"])
            core["videoInterestingness"] = interestingness
        candidate_source = raw.get("candidateSource")
        proposal_seed_count = raw.get("proposalSeedCount")
        proposal_seconds = raw.get("proposalMaximumSeconds")
        indices = raw.get("proposalPlayIndices")
        library = raw.get("acceptedLessonLibrary")
        if "proposalPlayIndices" in raw:
            if (candidate_source is None or type(indices) is not list or not indices
                    or any(type(index) is not int or not 0 <= index <= 31 for index in indices)
                    or len(set(indices)) != len(indices)):
                raise ValueError("proposalPlayIndices requires candidateSource and unique indices in 0..31")
            indices = sorted(indices)
            core["proposalPlayIndices"] = indices
        if "acceptedLessonLibrary" in raw:
            from .lesson_prefilter import lesson_is_repeated_v1
            if candidate_source is None:
                raise ValueError("acceptedLessonLibrary requires candidateSource")
            lesson_is_repeated_v1({"family": None, "deductions": []}, library)
            library = copy.deepcopy(library)
            core["acceptedLessonLibrary"] = library
        if "proposalMaximumSeconds" in raw:
            if (candidate_source is None or type(proposal_seconds) not in {int, float}
                    or not math.isfinite(proposal_seconds) or not 0 < proposal_seconds <= seconds):
                raise ValueError("proposalMaximumSeconds requires candidateSource and a positive limit within maximumWallSeconds")
            core["proposalMaximumSeconds"] = float(proposal_seconds)
        if "proposalSeedCount" in raw and candidate_source is None:
            raise ValueError("proposalSeedCount requires candidateSource")
        if "candidateSource" in raw:
            if candidate_source not in {"runtime_proposal_v1", "runtime_proposal_stream_v1"}:
                raise ValueError("quality candidateSource is unsupported")
            if not audit_constructed or policy_id is None or interestingness is None or interestingness["schemaVersion"] != 4:
                raise ValueError("runtime proposals require auditConstructedTrace, earlyPlayPolicyId and schema-4 videoInterestingness")
            core["candidateSource"] = candidate_source
            if type(proposal_seed_count) is not int or not 1 <= proposal_seed_count <= 1_000_000:
                raise ValueError("proposalSeedCount must be an integer in 1..1000000")
            core["proposalSeedCount"] = proposal_seed_count
        family_target = None
        if "familyTarget" in raw:
            from .family_catalog import build_family_target_v1
            if candidate_source != "runtime_proposal_stream_v1":
                raise ValueError("familyTarget requires runtime_proposal_stream_v1")
            family_target = build_family_target_v1(raw["familyTarget"])
            if indices is not None and indices != [family_target["bindings"]["playIndex"]]:
                raise ValueError("familyTarget must match proposalPlayIndices exactly")
            if library is not None:
                raise ValueError("familyTarget does not support runtime lesson prefiltering")
            core["familyTarget"] = family_target
        family_catalog = None
        if "familyCatalog" in raw:
            from .family_catalog import ResearchFamilyCatalogV1
            if candidate_source != "runtime_proposal_stream_v1" or family_target is not None:
                raise ValueError("familyCatalog requires its own runtime_proposal_stream_v1, not familyTarget")
            if library is not None:
                raise ValueError("familyCatalog does not support runtime lesson prefiltering")
            catalog = ResearchFamilyCatalogV1.build(raw["familyCatalog"])
            catalog.require_sun_history_support()
            family_catalog = catalog.to_dict()
            core["familyCatalog"] = family_catalog
        opening_mode = raw.get('familyOpeningMode')
        if 'familyOpeningMode' in raw:
            if (opening_mode != NATIVE_FAMILY_OPENING_MODE or candidate_source != 'runtime_proposal_stream_v1'
                    or family_target is None and family_catalog is None):
                raise ValueError('familyOpeningMode requires a supported native family target or catalog stream')
            core['familyOpeningMode'] = opening_mode
        proposal_mode = raw.get('familyProposalMode')
        if 'familyProposalMode' in raw:
            if (proposal_mode != NATIVE_FAMILY_PROPOSAL_MODE or family_target is None
                    or family_target['bindings']['contract'] != 'sun' or opening_mode is not None):
                raise ValueError('familyProposalMode requires an explicit Sun familyTarget without familyOpeningMode')
            core['familyProposalMode'] = proposal_mode
        digest = canonical_sha256(core)
        if "sha256" in raw and raw["sha256"] != digest:
            raise ValueError("quality execution budget checksum mismatch")
        return cls(float(seconds), raw["maximumEvidenceBytes"], raw["maximumEscalationDecisions"], digest,
                   tuple(order) if order is not None else None, world_limit, policy_id, exact_from, featured, incumbent, allow_exact, paired, prune_pass, bid_policy, alternative, refutation, strategy_fallback, audit_constructed, exact_tail, interestingness, candidate_source, proposal_seed_count, proposal_seconds,
                   tuple(indices) if indices is not None else None, library, bands, family_target, family_catalog, opening_mode, proposal_mode, tail_budget)

    def resolve_hidden_world_cap(self, maximum_worlds: int, *, play_index: int | None) -> int:
        """Intersect caller/profile limits with the frozen execution cap at the actual play index."""
        if type(maximum_worlds) is not int or maximum_worlds < 1:
            raise ValueError("maximum worlds must be a positive integer")
        if play_index is not None and (type(play_index) is not int or not 0 <= play_index <= 31):
            raise ValueError("play index must be an integer in 0..31")
        cap = min(maximum_worlds, self.maximum_hidden_worlds_per_decision or maximum_worlds)
        if play_index is not None and self.play_index_world_caps is not None:
            for band in self.play_index_world_caps["bands"]:
                if band["firstPlayIndex"] <= play_index <= band["lastPlayIndex"]:
                    cap = min(cap, band["maximumHiddenWorlds"])
        return cap

    @property
    def observed_bid_incumbent_proof(self) -> dict[str, Any] | None:
        """Actor-visible view; the full frozen budget retains complete-state controls."""
        if self.bid_incumbent_proof is None or self.bid_incumbent_proof["schemaVersion"] != 2:
            return None
        return copy.deepcopy({key: value for key, value in self.bid_incumbent_proof.items()
            if key not in {"completeStateNonplayPolicy", "completeStateAlternativeFirst"}})

    def to_dict(self) -> dict[str, Any]:
        return {"schemaVersion": 1, "maximumWallSeconds": self.maximum_wall_seconds,
                "maximumEvidenceBytes": self.maximum_evidence_bytes,
                "maximumEscalationDecisions": self.maximum_escalation_decisions, "sha256": self.sha256,
                **({"playIndexWorldCaps": copy.deepcopy(dict(self.play_index_world_caps))} if self.play_index_world_caps is not None else {}),
                **({"familyTarget": copy.deepcopy(dict(self.family_target))} if self.family_target is not None else {}),
                **({"familyCatalog": copy.deepcopy(dict(self.family_catalog))} if self.family_catalog is not None else {}),
                **({'familyOpeningMode': self.family_opening_mode} if self.family_opening_mode is not None else {}),
                **({'familyProposalMode': self.family_proposal_mode} if self.family_proposal_mode is not None else {}),
                **({"candidateSource": self.candidate_source} if self.candidate_source is not None else {}),
                **({"proposalSeedCount": self.proposal_seed_count} if self.proposal_seed_count is not None else {}),
                **({"proposalPlayIndices": list(self.proposal_play_indices)} if self.proposal_play_indices is not None else {}),
                **({"acceptedLessonLibrary": copy.deepcopy(dict(self.accepted_lesson_library))} if self.accepted_lesson_library is not None else {}),
                **({"proposalMaximumSeconds": float(self.proposal_maximum_seconds)} if self.proposal_maximum_seconds is not None else {}),
                **({"auditConstructedTrace": True} if self.audit_constructed_trace else {}),
                **({"exactTailProof": True} if self.exact_tail_proof else {}),
                **({"exactTailBudgetProfile": self.exact_tail_budget_profile}
                   if self.exact_tail_budget_profile is not None else {}),
                **({"videoInterestingness": copy.deepcopy(dict(self.video_interestingness))}
                   if self.video_interestingness is not None else {}),
                **({"escalationOpportunityOrder": list(self.escalation_opportunity_order)}
                   if self.escalation_opportunity_order is not None else {}),
                **({"maximumHiddenWorldsPerDecision": self.maximum_hidden_worlds_per_decision}
                   if self.maximum_hidden_worlds_per_decision is not None else {}),
                **({"earlyPlayPolicyId": self.early_play_policy_id}
                   if self.early_play_policy_id is not None else {}),
                **({"exactPlayFromTrick": self.exact_play_from_trick}
                   if self.exact_play_from_trick is not None else {}),
                **({"featuredPlayNativeProof": True}
                   if self.featured_play_native_proof else {}),
                **({"allowExactInputQuality": True} if self.allow_exact_input_quality else {}),
                **({"completeStatePlayPolicyPair": True} if self.complete_state_play_policy_pair else {}),
                **({"pruneOwnProjectPassWorlds": True} if self.prune_own_project_pass_worlds else {}),
                **({"bidOpponentPlayPolicyId": self.bid_opponent_play_policy_id}
                   if self.bid_opponent_play_policy_id is not None else {}),
                **({"bidIncumbentProof": copy.deepcopy(dict(self.bid_incumbent_proof))}
                   if self.bid_incumbent_proof is not None else {}),
                **({"bidAlternativeProof": copy.deepcopy(dict(self.bid_alternative_proof))}
                   if self.bid_alternative_proof is not None else {}),
                **({"bidThresholdRefutation": copy.deepcopy(dict(self.bid_threshold_refutation))}
                   if self.bid_threshold_refutation is not None else {}),
                **({"observedBidStrategyPlayFallbackId": self.observed_bid_strategy_play_fallback_id}
                   if self.observed_bid_strategy_play_fallback_id is not None else {})}


@dataclass(frozen=True, slots=True)
class ScenarioPlayingQualityPolicyV1:
    """Actor-relative surrounding-play analysis, separate from a featured claim.

    Declaring a policy does not establish producer coverage or playing strength.
    Resource profile references must be resolved from the recipe's frozen bundle.
    """

    mode: str
    decisions: Mapping[str, Mapping[str, Any]]
    sha256: str
    schema_version: int = 1
    visibility: str = "actor_information_set"

    @property
    def uses_auction_hand_boundary(self) -> bool:
        return (
            self.mode == "budgeted"
            and self.decisions["bid"]["boundaryEvaluator"] == "auction_hand_margin_v1"
        )

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "ScenarioPlayingQualityPolicyV1":
        raw = _mapping(value, "playing quality policy")
        version = raw.get("schemaVersion")
        if type(version) is not int or version not in {1, 2}:
            raise ValueError("playing quality policy schemaVersion must be integer 1 or 2")
        _closed(raw, {"schemaVersion", "mode", "decisions", "sha256"}
                | ({"visibility"} if version == 2 else set()), "playing quality policy")
        visibility = raw.get("visibility") if version == 2 else "actor_information_set"
        if type(visibility) is not str or visibility not in PROOF_VISIBILITIES:
            raise ValueError("playing quality policy requires an explicit supported visibility")
        mode = raw.get("mode")
        if type(mode) is not str or mode not in PLAYING_QUALITY_MODES:
            raise ValueError("playing quality policy mode is unsupported")
        decisions = _mapping(raw.get("decisions"), "playing quality decisions")
        if set(decisions) != set(SCENARIO_DECISION_KINDS):
            raise ValueError("playing quality requires every decision kind")
        fields = {"utility", "budgetProfile"}
        if mode == "budgeted":
            fields |= {"horizon", "boundaryEvaluator"}
        normalized = {}
        for kind in SCENARIO_DECISION_KINDS:
            analysis = _mapping(decisions[kind], f"playing quality {kind}")
            if set(analysis) - {"utilityEncoding"} != fields:
                raise ValueError(f"playing quality {kind} fields do not match its mode")
            for field in fields - {"horizon"}:
                if type(analysis[field]) is not str or not analysis[field] or analysis[field] != analysis[field].strip():
                    raise ValueError(f"playing quality {kind}.{field} must be a canonical string")
            request = cls._request(mode, analysis, "north", visibility)
            validate_proof_boundary_scope_v1(request, kind)
            normalized[kind] = {key: request.to_dict()[key] for key in sorted(fields | (
                {"utilityEncoding"} if request.utility_encoding is not None else set()))}
        core = {"schemaVersion": version, "mode": mode, "decisions": normalized,
                **({"visibility": visibility} if version == 2 else {})}
        digest = canonical_sha256(core)
        if "sha256" in raw and raw["sha256"] != digest:
            raise ValueError("playing quality policy checksum mismatch")
        return cls(mode, normalized, digest, version, visibility)

    @staticmethod
    def _request(mode: str, analysis: Mapping[str, Any], actor: str,
                 visibility: str = "actor_information_set") -> ScenarioProofRequestV1:
        if actor not in _SEATS:
            raise ValueError("playing quality actor must be a seat")
        return ScenarioProofRequestV1.build({
            "schemaVersion": 1, "mode": "bounded" if mode == "budgeted" else "exact",
            "visibility": visibility, "strategyClass": "pure_deterministic",
            "claim": "winning_root_action_count", "successThreshold": 0,
            "targetTeam": Seat.parse(actor).team.name.lower(), **analysis,
        })

    def request_for(self, decision_kind: str, actor: str, *,
                    play_index: int | None = None,
                    execution_budget: ScenarioQualityExecutionBudgetV1 | None = None) -> ScenarioProofRequestV1:
        if decision_kind not in SCENARIO_DECISION_KINDS:
            raise ValueError("playing quality decision kind is unsupported")
        checked = type(self).build(self.to_dict())
        request = self._request(checked.mode, checked.decisions[decision_kind], actor, checked.visibility)
        if play_index is not None and (decision_kind != "play" or type(play_index) is not int
                                       or not 0 <= play_index < 32):
            raise ValueError("quality play index must identify one of the 32 play decisions")
        if execution_budget is not None:
            if not isinstance(execution_budget, ScenarioQualityExecutionBudgetV1):
                raise TypeError("quality request requires a typed execution budget")
            execution_budget = ScenarioQualityExecutionBudgetV1.build(execution_budget.to_dict())
            if execution_budget.exact_tail_proof and decision_kind == "play" and checked.mode == "budgeted":
                if play_index is None:
                    raise ValueError("exact tail quality requires the owned play index")
                if play_index >= 4 * (execution_budget.exact_play_from_trick - 1):
                    raw = request.to_dict()
                    for key in ("sha256", "horizon", "boundaryEvaluator"):
                        raw.pop(key, None)
                    raw["mode"] = "exact"
                    if execution_budget.exact_tail_budget_profile is not None:
                        raw["budgetProfile"] = execution_budget.exact_tail_budget_profile
                    request = ScenarioProofRequestV1.build(raw)
        return request

    def to_dict(self) -> dict[str, Any]:
        return {"schemaVersion": self.schema_version, "mode": self.mode,
                **({"visibility": self.visibility} if self.schema_version == 2 else {}),
                "decisions": copy.deepcopy(dict(self.decisions)), "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class ResearchScenarioV1:
    origin: ScenarioOriginV1
    constraints: ScenarioExpressionV1
    decision_point: Mapping[str, Any]
    outcome: ScenarioExpressionV1
    sha256: str

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "ResearchScenarioV1":
        raw = _mapping(value, "research scenario")
        _closed(
            raw,
            {
                "schemaVersion",
                "origin",
                "constraints",
                "decisionPoint",
                "outcome",
                "sha256",
            },
            "research scenario",
        )
        if raw.get("schemaVersion") != 1 or type(raw.get("schemaVersion")) is not int:
            raise ValueError("research scenario schemaVersion must be integer 1")
        origin = ScenarioOriginV1.build(_mapping(raw.get("origin"), "scenario origin"))
        constraints = ScenarioExpressionV1.build(
            _mapping(raw.get("constraints"), "scenario constraints")
        )
        decision_point = _mapping(raw.get("decisionPoint"), "scenario decisionPoint")
        _closed(decision_point, {"kind", "selector"}, "scenario decisionPoint")
        if str(decision_point.get("kind") or "") not in SCENARIO_DECISION_KINDS:
            raise ValueError("scenario decisionPoint kind is unsupported")
        decision_kind = str(decision_point["kind"])
        selector = _mapping(
            decision_point.get("selector", {}), "scenario decisionPoint selector"
        )
        allowed_selector_fields = {
            "bid": {"actionIndex"},
            "escalation": {"actionIndex", "actingSeat"},
            "project": {"owner", "projectIndex"},
            "play": {"remainingCards", "trickIndex", "position", "actingSeat"},
        }[decision_kind]
        _closed(
            selector,
            allowed_selector_fields,
            f"scenario {decision_kind} decisionPoint selector",
        )
        for key in (
            "actionIndex",
            "projectIndex",
            "remainingCards",
            "trickIndex",
            "position",
        ):
            if key not in selector:
                continue
            minimum = 1 if key == "remainingCards" else 0
            maximum = (
                32
                if key == "remainingCards"
                else 7
                if key == "trickIndex"
                else 3
                if key == "position"
                else 10_000
            )
            value = _strict_int(
                selector[key],
                f"scenario decisionPoint selector {key}",
                minimum=minimum,
            )
            if value > maximum:
                raise ValueError(f"scenario decisionPoint selector {key} is too large")
        if "owner" in selector and str(selector["owner"]) not in _SEATS:
            raise ValueError("scenario project decisionPoint owner is invalid")
        if "actingSeat" in selector and str(selector["actingSeat"]) not in _SEATS:
            raise ValueError("scenario decisionPoint actingSeat is invalid")
        decision_point["selector"] = selector
        outcome = ScenarioExpressionV1.build(
            _mapping(raw.get("outcome"), "scenario outcome")
        )
        all_predicates = (*constraints.predicates(), *outcome.predicates())
        ids = [item.predicate_id for item in all_predicates]
        if len(ids) != len(set(ids)):
            raise ValueError("scenario predicate IDs must be unique")
        core = {
            "schemaVersion": 1,
            "origin": origin.to_dict(),
            "constraints": constraints.to_dict(),
            "decisionPoint": decision_point,
            "outcome": outcome.to_dict(),
        }
        digest = canonical_sha256(core)
        declared = str(raw.get("sha256") or "")
        if declared and declared != digest:
            raise ValueError("research scenario checksum mismatch")
        return cls(origin, constraints, decision_point, outcome, digest)

    def predicates(self) -> tuple[ScenarioPredicateV1, ...]:
        return (*self.constraints.predicates(), *self.outcome.predicates())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "origin": self.origin.to_dict(),
            "constraints": self.constraints.to_dict(),
            "decisionPoint": copy.deepcopy(dict(self.decision_point)),
            "outcome": self.outcome.to_dict(),
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ScenarioDiagnosticV1:
    issue_id: str
    diagnostic_class: str
    severity: str
    code: str
    predicate_ids: tuple[str, ...]
    field_paths: tuple[str, ...]
    message_ar_key: str
    message_en_key: str
    details: Mapping[str, Any]
    sha256: str

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "ScenarioDiagnosticV1":
        raw = _mapping(value, "scenario diagnostic")
        diagnostic_class = str(raw.get("class") or "")
        if diagnostic_class not in DIAGNOSTIC_CLASSES:
            raise ValueError("scenario diagnostic class is unsupported")
        severity = str(raw.get("severity") or "")
        if severity not in {"error", "warning", "info"}:
            raise ValueError("scenario diagnostic severity is unsupported")
        core = {
            "issueId": str(raw.get("issueId") or ""),
            "class": diagnostic_class,
            "severity": severity,
            "code": str(raw.get("code") or ""),
            "predicateIds": [
                str(item)
                for item in _list(
                    raw.get("predicateIds", []), "diagnostic predicateIds"
                )
            ],
            "fieldPaths": [
                str(item)
                for item in _list(raw.get("fieldPaths", []), "diagnostic fieldPaths")
            ],
            "messageArKey": str(raw.get("messageArKey") or ""),
            "messageEnKey": str(raw.get("messageEnKey") or ""),
            "details": _mapping(raw.get("details", {}), "diagnostic details"),
        }
        if (
            not core["issueId"]
            or not core["code"]
            or not core["messageArKey"]
            or not core["messageEnKey"]
        ):
            raise ValueError("scenario diagnostic identities and messages are required")
        digest = canonical_sha256(core)
        return cls(
            core["issueId"],
            diagnostic_class,
            severity,
            core["code"],
            tuple(core["predicateIds"]),
            tuple(core["fieldPaths"]),
            core["messageArKey"],
            core["messageEnKey"],
            core["details"],
            digest,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "issueId": self.issue_id,
            "class": self.diagnostic_class,
            "severity": self.severity,
            "code": self.code,
            "predicateIds": list(self.predicate_ids),
            "fieldPaths": list(self.field_paths),
            "messageArKey": self.message_ar_key,
            "messageEnKey": self.message_en_key,
            "details": copy.deepcopy(dict(self.details)),
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ProofOutcomeV1:
    payload: Mapping[str, Any]
    sha256: str

    @classmethod
    def build(cls, value: Mapping[str, Any]) -> "ProofOutcomeV1":
        raw = _mapping(value, "proof outcome")
        _closed(
            raw,
            {
                "schemaVersion",
                "requestedMode",
                "achievedMode",
                "requestedVisibility",
                "achievedVisibility",
                "strategyClass",
                "claim",
                "targetTeam",
                "utility",
                "successThreshold",
                "horizon",
                "boundaryEvaluator",
                "rootActions",
                "selectedAction",
                "winningRootActionCount",
                "winningRootActionLowerBound",
                "winningRootActionUpperBound",
                "continuationLines",
                "hiddenWorlds",
                "resources",
                "completionStatus",
                "claimSatisfied",
                "principalVariation",
                "identities",
                "semanticSha256",
                "sha256",
            },
            "proof outcome",
        )
        if raw.get("schemaVersion") != 1 or type(raw.get("schemaVersion")) is not int:
            raise ValueError("proof outcome schemaVersion must be integer 1")
        if str(raw.get("requestedMode") or "") not in PROOF_MODES:
            raise ValueError("proof outcome requestedMode is unsupported")
        achieved = str(raw.get("achievedMode") or "")
        if achieved not in {*PROOF_MODES, "none"}:
            raise ValueError("proof outcome achievedMode is unsupported")
        if str(raw.get("requestedVisibility") or "") not in PROOF_VISIBILITIES:
            raise ValueError("proof outcome requestedVisibility is unsupported")
        achieved_visibility = str(raw.get("achievedVisibility") or "")
        if achieved_visibility not in {*PROOF_VISIBILITIES, "none"}:
            raise ValueError("proof outcome achievedVisibility is unsupported")
        strategy_class = raw.get("strategyClass")
        if strategy_class is not None:
            strategy_class = str(strategy_class)
            if strategy_class not in PROOF_STRATEGY_CLASSES:
                raise ValueError("proof outcome strategyClass is unsupported")
        expected_strategy_class = (
            "frozen_policy"
            if str(raw.get("requestedMode") or "") == "policy_conditioned"
            else "pure_deterministic"
        )
        if (
            str(raw.get("requestedVisibility") or "")
            == "actor_information_set"
            and strategy_class is None
        ):
            raise ValueError(
                "actor-information-set proof outcome requires strategyClass"
            )
        if strategy_class is not None and strategy_class != expected_strategy_class:
            raise ValueError(
                "proof outcome strategyClass is incompatible with requested mode"
            )
        if str(raw.get("claim") or "") not in PROOF_CLAIMS:
            raise ValueError("proof outcome claim is unsupported")
        if str(raw.get("targetTeam") or "") not in _TEAMS:
            raise ValueError("proof outcome targetTeam is invalid")
        raw_winning_count = raw.get("winningRootActionCount")
        winning_count = (
            None
            if raw_winning_count is None
            else _strict_int(
                raw_winning_count, "winningRootActionCount", minimum=0
            )
        )
        root_actions = _list(raw.get("rootActions"), "proof outcome rootActions")
        if winning_count is not None and winning_count > len(root_actions):
            raise ValueError("winningRootActionCount exceeds evaluated root actions")
        lower_bound = _strict_int(
            raw.get(
                "winningRootActionLowerBound",
                winning_count if winning_count is not None else 0,
            ),
            "winningRootActionLowerBound",
            minimum=0,
        )
        raw_upper_bound = raw.get(
            "winningRootActionUpperBound",
            winning_count if winning_count is not None else None,
        )
        upper_bound = (
            None
            if raw_upper_bound is None
            else _strict_int(
                raw_upper_bound, "winningRootActionUpperBound", minimum=0
            )
        )
        if upper_bound is not None and lower_bound > upper_bound:
            raise ValueError("winning-root bounds are inverted")
        if winning_count is not None and not (
            lower_bound <= winning_count
            and (upper_bound is None or winning_count <= upper_bound)
        ):
            raise ValueError("winningRootActionCount falls outside its bounds")
        continuation = _mapping(
            raw.get("continuationLines"), "proof outcome continuationLines"
        )
        _closed(
            continuation,
            {"complete", "exactCount", "lowerBound", "upperBound", "lineUniverse"},
            "continuationLines",
        )
        if type(continuation.get("complete")) is not bool:
            raise TypeError("continuationLines.complete must be a boolean")
        if str(continuation.get("lineUniverse") or "") not in LINE_UNIVERSES:
            raise ValueError("continuationLines.lineUniverse is unsupported")
        if continuation["complete"]:
            if not str(continuation.get("exactCount") or "").isdigit():
                raise ValueError(
                    "complete continuation lines require decimal exactCount"
                )
            if (
                continuation.get("lowerBound") is not None
                or continuation.get("upperBound") is not None
            ):
                raise ValueError("complete continuation lines cannot contain bounds")
        else:
            if continuation.get("exactCount") is not None:
                raise ValueError(
                    "incomplete continuation lines cannot contain exactCount"
                )
            if not str(continuation.get("lowerBound") or "").isdigit():
                raise ValueError(
                    "incomplete continuation lines require decimal lowerBound"
                )
            upper = continuation.get("upperBound")
            if upper is not None and not str(upper).isdigit():
                raise ValueError("continuationLines.upperBound must be decimal or null")
        completion = str(raw.get("completionStatus") or "")
        if completion not in {
            "exact_complete",
            "bounded_complete",
            "policy_complete",
            "proof_incomplete",
            "proof_failed",
        }:
            raise ValueError("proof outcome completionStatus is unsupported")
        expected_achieved_mode = {
            "exact_complete": "exact",
            "bounded_complete": "bounded",
            "policy_complete": "policy_conditioned",
        }.get(completion)
        if expected_achieved_mode is not None and achieved != expected_achieved_mode:
            raise ValueError(
                f"{completion} proof must achieve {expected_achieved_mode} mode"
            )
        if expected_achieved_mode is not None and achieved_visibility == "none":
            raise ValueError("completed proof must achieve its declared visibility")
        if expected_achieved_mode is not None and winning_count is None:
            raise ValueError("completed proof requires winningRootActionCount")
        if type(raw.get("claimSatisfied")) is not bool:
            raise TypeError("proof outcome claimSatisfied must be a boolean")
        if completion == "proof_failed" and raw["claimSatisfied"] is not False:
            raise ValueError("proof_failed requires a false claim result")
        if (
            completion in {"exact_complete", "bounded_complete", "policy_complete"}
            and raw["claimSatisfied"] is not True
        ):
            raise ValueError("completed proof requires a satisfied claim")
        identities = _mapping(raw.get("identities"), "proof outcome identities")
        required_identities = {
            "rulesHash",
            "bundleHash",
            "stateHash",
            "proofProfileHash",
        }
        if not required_identities.issubset(identities):
            raise ValueError("proof outcome immutable identities are incomplete")
        for key in required_identities:
            if _HASH.fullmatch(str(identities[key])) is None:
                raise ValueError(f"proof outcome {key} is invalid")
        declared = str(raw.get("sha256") or "")
        declared_semantic = str(raw.get("semanticSha256") or "")
        legacy_declared_receipt = bool(declared) and not declared_semantic
        core = copy.deepcopy(raw)
        core.pop("sha256", None)
        core["successThreshold"] = _strict_number(
            core.get("successThreshold", 0.0), "proof outcome successThreshold"
        )
        if legacy_declared_receipt:
            digest = canonical_sha256(core)
            if declared != digest:
                raise ValueError("proof outcome checksum mismatch")
            return cls(core, digest)
        if not declared and not declared_semantic:
            core.setdefault("winningRootActionLowerBound", lower_bound)
            core.setdefault("winningRootActionUpperBound", upper_bound)
        semantic_core = copy.deepcopy(core)
        semantic_core.pop("semanticSha256", None)
        semantic_resources = semantic_core.get("resources")
        if isinstance(semantic_resources, Mapping):
            stable_resources = copy.deepcopy(dict(semantic_resources))
            stable_resources.pop("elapsedWallSeconds", None)
            stable_resources.pop("elapsedCpuSeconds", None)
            semantic_core["resources"] = stable_resources
        semantic_digest = canonical_sha256(semantic_core)
        if declared_semantic and declared_semantic != semantic_digest:
            raise ValueError("proof outcome semantic checksum mismatch")
        core["semanticSha256"] = semantic_digest
        digest = canonical_sha256(core)
        if declared and declared != digest:
            raise ValueError("proof outcome checksum mismatch")
        return cls(core, digest)

    def to_dict(self) -> dict[str, Any]:
        return {**copy.deepcopy(dict(self.payload)), "sha256": self.sha256}


__all__ = [
    "CONSTRUCTION_CAPABILITIES",
    "DIAGNOSTIC_CLASSES",
    "ExactTracePrefixV1",
    "LINE_UNIVERSES",
    "PREDICATE_HARDNESS",
    "PLAYING_QUALITY_MODES",
    "PROOF_CLAIMS",
    "PROOF_BOUNDARY_EVALUATORS",
    "PROOF_HORIZON_UNITS",
    "PROOF_MODES",
    "PROOF_STRATEGY_CLASSES",
    "PROOF_VISIBILITIES",
    "PROOF_UTILITIES",
    "PROOF_UTILITY_ENCODINGS",
    "ProofOutcomeV1",
    "ResearchScenarioV1",
    "SCENARIO_OPERATORS",
    "SCENARIO_DECISION_KINDS",
    "SCENARIO_SECTIONS",
    "ScenarioDiagnosticV1",
    "ScenarioExpressionV1",
    "ScenarioOriginV1",
    "ScenarioPredicateV1",
    "ScenarioProofRequestV1",
    "ScenarioProofBudgetV1",
    "ScenarioPlayingQualityPolicyV1",
    "canonical_sha256",
    "hard_continuation_count_request_issue_v1",
]
