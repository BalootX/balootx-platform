"""Immutable Research V4 scenario result construction and validation."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping

from baloot_platform.core.auction_ir import AuctionDomainIRV1
from baloot_platform.core.bidding_machine import BiddingMachine
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Seat, Suit
from baloot_platform.core.exact_state_view import (
    OBSERVER_SAFE_EVENT_TYPES_V1,
    build_observer_safe_exact_state_v1,
)

from .exact_state import ExactStateNormalizerV1, NormalizedExactStateV1
from .recipe_v4 import ResearchRecipeV4
from .scenario_contracts import PROOF_HORIZON_UNITS, ProofOutcomeV1, canonical_sha256
from .scenario_evidence import ScenarioSatisfactionV1


SCENARIO_ARTIFACT_MEDIA_TYPES = {
    "exactState": "application/vnd.baloot.research-exact-state-v1+json",
    "scenarioSatisfaction": "application/vnd.baloot.research-scenario-satisfaction-v1+json",
    "constraintEvidence": "application/vnd.baloot.research-constraint-evidence-v1+json",
    "proofOutcome": "application/vnd.baloot.research-proof-outcome-v1+json",
}
_SCENARIO_PROJECTION_KEYS = frozenset(
    {
        "schemaVersion",
        "scenarioTemplateId",
        "origin",
        "status",
        "decisionKind",
        "targetTeam",
        "requestedProofMode",
        "achievedProofMode",
        "proofCompletion",
        "proofVisibility",
        "winningRootActionCount",
        "winningRootActionLowerBound",
        "winningRootActionUpperBound",
        "contractOutcome",
        "projectKinds",
        "projectValues",
        "projectStates",
        "exactStateId",
        "exactStateAvailable",
        "publicProjectionHash",
        "sha256",
    }
)
_SCORE_EVIDENCE_KEYS = frozenset(
    {
        "schemaVersion",
        "family",
        "scenarioSatisfactionHash",
        "proofOutcomeHash",
        "isStructuralObjective",
        "score",
        "sha256",
    }
)
_PUBLIC_ROOT_ACTION_KEYS = frozenset(
    {
        "action",
        "value",
        "satisfies",
        "strategyScope",
        "targetPolicyCount",
        "opponentPolicyCount",
        "targetPolicyHash",
        "worstOpponentPolicyHash",
        "completionStatus",
    }
)
_PUBLIC_RESOURCE_KEYS = frozenset(
    {
        "nodes",
        "depth",
        "cacheEntries",
        "cacheBytes",
        "elapsedWallSeconds",
        "elapsedCpuSeconds",
        "nodeLimit",
        "depthLimit",
        "wallTimeLimitSeconds",
        "cacheEntryLimit",
        "cacheByteLimit",
        "hiddenWorldLimit",
        "horizonActionLimit",
        "cpuTimeLimitSeconds",
        "cancellationCheckIntervalNodes",
        "workerLimit",
        "workers",
        "strategyClass",
        "lineCountNodeLimit",
        "forecastStateSpace",
        "stateSpaceLimit",
        "exhaustionReason",
        "cancelled",
    }
)
_PUBLIC_IDENTITY_HASH_KEYS = frozenset(
    {
        "rulesHash",
        "bundleHash",
        "proofProfileHash",
        "proofBudgetRegistryHash",
    }
)
_PUBLIC_PROOF_KEYS = frozenset(
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
        "boundaryEvaluator",
        "selectedAction",
        "winningRootActionCount",
        "winningRootActionLowerBound",
        "winningRootActionUpperBound",
        "completionStatus",
        "claimSatisfied",
        "semanticSha256",
    }
)


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _proof_semantic_sha256(value: Mapping[str, Any]) -> str:
    return str(value.get("semanticSha256") or value.get("sha256") or "")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_bid_authority_v1(
    normalized: NormalizedExactStateV1,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = normalized.state.get("bidding")
    if not isinstance(raw, Mapping):
        raise ValueError("Research V4 bid state lacks bidding authority")
    state = normalized.round_state
    if state.purchase_card is None:
        raise ValueError("Research V4 bid state lacks the purchase card")
    machine = BiddingMachine(
        state.dealer,
        state.purchase_card.suit,
        AuctionDomainIRV1.from_rules(normalized.rules.data),
        state.purchase_card.rank,
    )
    history = raw.get("history")
    if not isinstance(history, list):
        raise ValueError("Research V4 bid history is invalid")
    for raw_record in history:
        if not isinstance(raw_record, Mapping):
            raise ValueError("Research V4 bid record is invalid")
        record = dict(raw_record)
        if machine.actor is None or record.get("seat") != machine.actor.name.lower():
            raise ValueError("Research V4 bid actor differs from authoritative replay")
        trump_raw = record.get("trumpSuit")
        machine.submit(
            str(record.get("action") or ""),
            trump=Suit.parse(str(trump_raw)) if trump_raw is not None else None,
        )
        if machine.records[-1].to_dict() != record:
            raise ValueError("Research V4 bid record differs from authoritative replay")
    if machine.result is not None or machine.actor is None:
        raise ValueError("Research V4 bid decision is already terminal")
    projection = {
        "round": machine.round_index,
        "actor": machine.actor.name.lower(),
        "history": [item.to_dict() for item in machine.records],
    }
    legal_actions: list[dict[str, Any]] = []
    for action in machine.legal_actions():
        if action != "hokum":
            legal_actions.append({"code": action, "action": action, "trumpSuit": None})
        elif machine.round_index == 1:
            suit = machine.purchase_suit.name.lower()
            legal_actions.append(
                {
                    "code": f"hokum:{suit}",
                    "action": "hokum",
                    "trumpSuit": suit,
                }
            )
        else:
            legal_actions.extend(
                {
                    "code": f"hokum:{suit.name.lower()}",
                    "action": "hokum",
                    "trumpSuit": suit.name.lower(),
                }
                for suit in Suit
                if suit != machine.purchase_suit
            )
    return projection, legal_actions


def _canonical_legal_actions_v1(
    normalized: NormalizedExactStateV1,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if normalized.next_actor is None:
        raise ValueError("Research V4 observer-safe decision lacks an actor")
    actor = Seat.parse(normalized.next_actor)
    if normalized.decision_kind == "bid":
        return _canonical_bid_authority_v1(normalized)
    engine = BalootEngine(normalized.rules)
    state = normalized.round_state
    if normalized.decision_kind == "play":
        if state.current_player != actor:
            raise ValueError("Research V4 play actor differs from authoritative state")
        return None, [action.to_dict() for action in engine.legal_play_actions(state)]
    if normalized.decision_kind == "escalation":
        return None, [
            dict(item)
            for item in ExactStateNormalizerV1._escalation_legal_actions(
                engine,
                state,
                actor,
            )
        ]
    if normalized.decision_kind == "project":
        _updated, announced = engine.project_domain.announce_for_turn(
            state.projects,
            actor,
            trump_suit=state.trump_suit,
        )
        if announced is None:
            return None, []
        return None, [
            {
                "code": (
                    f"declare:{announced.kind}:{announced.value}:{announced.high_rank}"
                ),
                "action": "declare",
                "owner": actor.name.lower(),
                "kind": announced.kind,
                "value": announced.value,
                "highRank": announced.high_rank,
            }
        ]
    raise ValueError("Research V4 observer-safe decision kind is unsupported")


def _observer_safety_evidence_v1(
    normalized: NormalizedExactStateV1,
    scenario_projection: Mapping[str, Any],
    proof: ProofOutcomeV1,
) -> dict[str, Any]:
    checks = {
        "actingObserver": False,
        "proofStateBound": False,
        "exactStateMatchesEngine": False,
        "publicFieldsMatchEngineProjection": False,
        "handsMatchEngineProjection": False,
        "projectsMatchEngineProjection": False,
        "projectWinningTeamMatchesEngineProjection": False,
        "tricksMatchEngineProjection": False,
        "eventsMatchEngineProjection": False,
        "biddingMatchesEngineProjection": False,
        "legalActionsMatchEngineProjection": False,
    }
    observer = str(normalized.next_actor or "")
    projected_hash = None
    engine_hash = None
    failure_code = "observer_projection_unavailable"
    try:
        seat = Seat.parse(observer)
        projected = build_observer_safe_exact_state_v1(
            exact_state=normalized.to_dict(),
            scenario_projection=scenario_projection,
            observer_seat=observer,
        )
        engine_view = BalootEngine.observer_view(normalized.round_state, seat)
        canonical_bidding, canonical_legal_actions = _canonical_legal_actions_v1(
            normalized
        )
        public_state = dict(projected["publicState"])
        identities = dict(proof.payload.get("identities") or {})
        engine_state = normalized.round_state.to_dict(reveal_hands=True)
        if canonical_bidding is not None:
            engine_state["bidding"] = copy.deepcopy(canonical_bidding)
        projected_events = list(public_state.get("events") or [])
        engine_events = [
            item
            for item in engine_view.get("events") or []
            if item.get("type") in OBSERVER_SAFE_EVENT_TYPES_V1
        ]
        engine_public_fields = {
            key: value
            for key, value in public_state.items()
            if key not in {"bidding", "events"}
        }
        checks = {
            "actingObserver": (
                observer
                == seat.name.lower()
                == str(projected.get("observerSeat") or "")
            ),
            "proofStateBound": (
                identities.get("stateHash") == normalized.state_hash
                and identities.get("rulesHash") == normalized.rules_hash
                and identities.get("exactStateId") == normalized.exact_state_id
            ),
            "exactStateMatchesEngine": (
                canonical_sha256(normalized.state) == normalized.state_hash
                and canonical_sha256(engine_state) == normalized.state_hash
            ),
            "publicFieldsMatchEngineProjection": all(
                engine_view.get(key) == value
                for key, value in engine_public_fields.items()
            ),
            "handsMatchEngineProjection": (
                public_state.get("hands") == engine_view.get("hands")
            ),
            "projectsMatchEngineProjection": (
                public_state.get("projects") == engine_view.get("projects")
            ),
            "projectWinningTeamMatchesEngineProjection": (
                public_state.get("projectWinningTeam")
                == engine_view.get("projectWinningTeam")
            ),
            "tricksMatchEngineProjection": (
                public_state.get("currentTrick") == engine_view.get("currentTrick")
                and public_state.get("completedTricks")
                == engine_view.get("completedTricks")
            ),
            "eventsMatchEngineProjection": projected_events == engine_events,
            "biddingMatchesEngineProjection": (
                public_state.get("bidding") == canonical_bidding
            ),
            "legalActionsMatchEngineProjection": (
                projected.get("legalActions") == canonical_legal_actions
            ),
        }
        projected_hash = canonical_sha256(projected)
        engine_hash = canonical_sha256(engine_view)
        failure_code = None if all(checks.values()) else "observer_projection_mismatch"
    except (KeyError, TypeError, ValueError):
        pass
    core = {
        "schemaVersion": 1,
        "kind": "research_v4_observer_projection_v1",
        "exactStateId": normalized.exact_state_id,
        "observerSeat": observer or None,
        "stateHash": normalized.state_hash,
        "proofOutcomeHash": _proof_semantic_sha256(proof.to_dict()),
        "projectionSha256": projected_hash,
        "engineProjectionSha256": engine_hash,
        "checks": checks,
        "failureCode": failure_code,
        "passed": all(checks.values()),
    }
    return {**core, "sha256": canonical_sha256(core)}


def build_constraint_evidence_v1(
    recipe: ResearchRecipeV4,
    normalized: NormalizedExactStateV1,
    satisfaction: ScenarioSatisfactionV1,
    *,
    construction_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    predicates = {item.predicate_id: item for item in satisfaction.evidence}
    rows = []
    for predicate in sorted(
        recipe.scenario.predicates(), key=lambda item: item.predicate_id
    ):
        evidence = predicates[predicate.predicate_id]
        definition = next(
            item
            for item in recipe.capability_plan.rows
            if item["predicateId"] == predicate.predicate_id
        )
        rows.append(
            {
                "predicateId": predicate.predicate_id,
                "predicateHash": predicate.sha256,
                "constructionCapability": definition["constructionCapability"],
                "constructionStatus": (
                    "exact_input_replayed"
                    if recipe.scenario.origin.kind == "exact_trace_prefix"
                    else "constructed_and_replayed"
                    if construction_receipt is not None
                    else "not_constructed"
                ),
                "runtimeEvidenceHash": evidence.sha256,
                "runtimeStatus": evidence.status,
            }
        )
    core = {
        "schemaVersion": 1,
        "scenarioHash": recipe.scenario.sha256,
        "capabilityPlanHash": recipe.capability_plan.sha256,
        "origin": recipe.scenario.origin.kind,
        "exactStateId": normalized.exact_state_id,
        "exactStateHash": normalized.sha256,
        "predicateEvidence": rows,
        "infeasibilityCore": None,
        "constructionStatus": (
            "satisfied"
            if construction_receipt is None
            else str(construction_receipt.get("status") or "satisfied")
        ),
        "constructionReceipt": (
            copy.deepcopy(dict(construction_receipt))
            if construction_receipt is not None
            else None
        ),
        "runtimeStatus": satisfaction.status,
    }
    return {**core, "sha256": canonical_sha256(core)}


def build_scenario_projection_v1(
    recipe: ResearchRecipeV4,
    normalized: NormalizedExactStateV1,
    satisfaction: ScenarioSatisfactionV1,
    proof: ProofOutcomeV1,
) -> dict[str, Any]:
    project_kinds: set[str] = set()
    project_values: set[str] = set()
    project_states: set[str] = set()
    for predicate in recipe.scenario.predicates():
        if not predicate.kind.startswith("project."):
            continue
        arguments = dict(predicate.arguments)
        if arguments.get("kind"):
            project_kinds.add(str(arguments["kind"]))
        if arguments.get("value") is not None:
            project_values.add(str(arguments["value"]))
        if arguments.get("state"):
            project_states.add(str(arguments["state"]))
    payload = proof.payload
    state = normalized.round_state
    contract_outcome = None
    if state.round_score is not None and state.declarer is not None:
        contract_outcome = (
            "won" if state.round_score.winner == state.declarer.team else "lost"
        )
    core = {
        "schemaVersion": 1,
        "scenarioTemplateId": f"SCN-{recipe.scenario.sha256[:12].upper()}",
        "origin": recipe.scenario.origin.kind,
        "status": satisfaction.status,
        "decisionKind": str(recipe.scenario.decision_point["kind"]),
        "targetTeam": str(payload["targetTeam"]),
        "requestedProofMode": str(payload["requestedMode"]),
        "achievedProofMode": str(payload["achievedMode"]),
        "proofCompletion": str(payload["completionStatus"]),
        "proofVisibility": str(payload["achievedVisibility"]),
        "winningRootActionCount": payload["winningRootActionCount"],
        "winningRootActionLowerBound": int(
            payload.get(
                "winningRootActionLowerBound",
                payload["winningRootActionCount"],
            )
        ),
        "winningRootActionUpperBound": payload.get(
            "winningRootActionUpperBound",
            payload["winningRootActionCount"],
        ),
        "contractOutcome": contract_outcome,
        "projectKinds": sorted(project_kinds),
        "projectValues": sorted(project_values),
        "projectStates": sorted(project_states),
        "exactStateId": normalized.exact_state_id,
        "exactStateAvailable": True,
    }
    try:
        public_projection_hash = build_observer_safe_exact_state_v1(
            exact_state=normalized.to_dict(),
            scenario_projection=core,
            observer_seat=str(normalized.next_actor or ""),
        )["publicProjectionHash"]
    except (KeyError, TypeError, ValueError):
        public_projection_hash = None
    core["publicProjectionHash"] = public_projection_hash
    return {**core, "sha256": canonical_sha256(core)}


def public_proof_outcome_v1(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return proof evidence safe for actor-facing reads and rendering.

    Replayable future lines and worst-world evidence remain only in the
    checksum-bound proof artifact.
    """

    raw_proof = copy.deepcopy(dict(value))
    artifact_sha256 = str(
        raw_proof.get("artifactSha256") or canonical_sha256(raw_proof)
    )
    if len(artifact_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in artifact_sha256
    ):
        raise ValueError("Research V4 public proof lacks its artifact identity")
    root_actions = raw_proof.get("rootActions")
    if not isinstance(root_actions, list):
        raise ValueError("Research V4 public proof root actions are invalid")
    public_actions: list[dict[str, Any]] = []
    for raw in root_actions:
        if not isinstance(raw, Mapping):
            raise ValueError("Research V4 public proof root action is invalid")
        action = {
            key: copy.deepcopy(raw[key])
            for key in _PUBLIC_ROOT_ACTION_KEYS
            if key in raw
        }
        if (
            not isinstance(action.get("action"), str)
            or not action["action"]
            or type(action.get("satisfies")) is not bool
            or type(action.get("value")) not in {int, float}
        ):
            raise ValueError("Research V4 public proof root action is invalid")
        if action.get("strategyScope") not in {
            None,
            "complete_state",
            "pure_nonanticipative_maximin",
        }:
            raise ValueError("Research V4 public proof strategy scope is invalid")
        if action.get("completionStatus") not in {
            None,
            "exact",
            "bounded",
            "policy_complete",
        }:
            raise ValueError("Research V4 public proof completion is invalid")
        for key in ("targetPolicyCount", "opponentPolicyCount"):
            if key in action and (type(action[key]) is not int or action[key] < 0):
                raise ValueError("Research V4 public proof policy count is invalid")
        for key in (
            "targetPolicyHash",
            "worstOpponentPolicyHash",
        ):
            if key in action and (
                not isinstance(action[key], str)
                or len(action[key]) != 64
                or any(character not in "0123456789abcdef" for character in action[key])
            ):
                raise ValueError("Research V4 public proof hash is invalid")
        public_actions.append(action)
    hidden_worlds = raw_proof.get("hiddenWorlds")
    if not isinstance(hidden_worlds, Mapping):
        raise ValueError("Research V4 public proof hidden worlds are invalid")
    hidden_count = hidden_worlds.get("count")
    hidden_complete = hidden_worlds.get("complete")
    coverage = hidden_worlds.get("coverage")
    coverage_method = coverage.get("method") if isinstance(coverage, Mapping) else None
    if (not isinstance(coverage_method, str) or not coverage_method) and raw_proof.get(
        "requestedVisibility"
    ) == "complete_state":
        coverage_method = "complete_state"
    if (
        type(hidden_count) is not int
        or hidden_count < 0
        or type(hidden_complete) is not bool
        or not isinstance(coverage, Mapping)
        or not isinstance(coverage_method, str)
        or not coverage_method
    ):
        raise ValueError("Research V4 public proof hidden-world summary is invalid")
    horizon = raw_proof.get("horizon")
    if horizon is not None:
        if (
            not isinstance(horizon, Mapping)
            or horizon.get("unit") not in set(PROOF_HORIZON_UNITS)
            or type(horizon.get("value")) is not int
            or horizon["value"] < 1
        ):
            raise ValueError("Research V4 public proof horizon is invalid")
        public_horizon: dict[str, Any] | None = {
            "unit": horizon["unit"],
            "value": horizon["value"],
        }
    else:
        public_horizon = None
    resources = raw_proof.get("resources", {})
    if not isinstance(resources, Mapping):
        raise ValueError("Research V4 public proof resources are invalid")
    public_resources = {
        key: copy.deepcopy(resources[key])
        for key in _PUBLIC_RESOURCE_KEYS
        if key in resources
        and (resources[key] is None or type(resources[key]) in {bool, int, float, str})
    }
    if public_resources.get("strategyClass") not in {
        None,
        "pure_deterministic",
        "frozen_policy",
    }:
        raise ValueError("Research V4 public proof resource strategy is invalid")
    continuation = raw_proof.get("continuationLines")
    public_continuation = None
    if continuation is not None:
        if not isinstance(continuation, Mapping):
            raise ValueError("Research V4 public proof continuation is invalid")
        public_continuation = {
            key: copy.deepcopy(continuation[key])
            for key in (
                "complete",
                "exactCount",
                "lowerBound",
                "upperBound",
                "lineUniverse",
            )
            if key in continuation
        }
    identities = raw_proof.get("identities")
    if not isinstance(identities, Mapping):
        raise ValueError("Research V4 public proof identities are invalid")
    public_identities: dict[str, Any] = {}
    for key in _PUBLIC_IDENTITY_HASH_KEYS:
        if key not in identities:
            continue
        item = identities[key]
        if (
            not isinstance(item, str)
            or len(item) != 64
            or any(character not in "0123456789abcdef" for character in item)
        ):
            raise ValueError("Research V4 public proof identity hash is invalid")
        public_identities[key] = item
    engine = identities.get("engine")
    if engine is not None:
        if (
            not isinstance(engine, Mapping)
            or set(engine) != {"id", "version"}
            or any(not isinstance(engine.get(key), str) for key in ("id", "version"))
        ):
            raise ValueError("Research V4 public proof engine identity is invalid")
        public_identities["engine"] = dict(engine)
    policy_hashes = identities.get("policyHashes")
    if policy_hashes is not None:
        if not isinstance(policy_hashes, Mapping) or any(
            seat not in {"north", "west", "south", "east"}
            or not isinstance(item, str)
            or len(item) != 64
            or any(character not in "0123456789abcdef" for character in item)
            for seat, item in policy_hashes.items()
        ):
            raise ValueError("Research V4 public proof policy identities are invalid")
        public_identities["policyHashes"] = dict(policy_hashes)
    proof = {
        key: copy.deepcopy(raw_proof[key])
        for key in _PUBLIC_PROOF_KEYS
        if key in raw_proof
    }
    winning_count = proof.get("winningRootActionCount")
    if proof.get("winningRootActionLowerBound") is None:
        proof["winningRootActionLowerBound"] = (
            winning_count
            if type(winning_count) is int
            else sum(action["satisfies"] for action in public_actions)
        )
    if proof.get("winningRootActionUpperBound") is None:
        proof["winningRootActionUpperBound"] = (
            winning_count
            if type(winning_count) is int
            and proof.get("completionStatus")
            in {"exact_complete", "bounded_complete", "policy_complete"}
            else len(public_actions)
        )
    proof["rootActions"] = public_actions
    proof["horizon"] = public_horizon
    if public_continuation is not None:
        proof["continuationLines"] = public_continuation
    proof["hiddenWorlds"] = {
        "count": hidden_count,
        "complete": hidden_complete,
        "coverage": {"method": coverage_method},
    }
    proof["resources"] = public_resources
    proof["identities"] = public_identities
    proof["artifactSha256"] = artifact_sha256
    proof["sha256"] = canonical_sha256(proof)
    return proof


def actor_public_proof_outcome_v1(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove artifact receipts whose preimages include trusted hidden state."""

    proof = public_proof_outcome_v1(value)
    proof.pop("artifactSha256", None)
    proof.pop("semanticSha256", None)
    proof.pop("sha256", None)
    proof["sha256"] = canonical_sha256(proof)
    return proof


def public_scenario_projection_v1(
    value: Mapping[str, Any],
    *,
    result_code: str,
    predicate_evidence_count: int,
) -> dict[str, Any]:
    """Mask full-state identity behind the retained result locator."""

    if (
        not isinstance(result_code, str)
        or not result_code.startswith("LAF-")
        or len(result_code) != 14
    ):
        raise ValueError("Research V4 public decision reference is invalid")
    projection = dict(value)
    public_projection_hash = projection.get("publicProjectionHash")
    if not _is_sha256(public_projection_hash):
        raise ValueError("Research V4 public projection hash is unavailable")
    core = {
        key: copy.deepcopy(projection[key])
        for key in (
            "schemaVersion",
            "scenarioTemplateId",
            "origin",
            "status",
            "decisionKind",
            "targetTeam",
            "requestedProofMode",
            "achievedProofMode",
            "proofCompletion",
            "proofVisibility",
            "winningRootActionCount",
            "winningRootActionLowerBound",
            "winningRootActionUpperBound",
            "contractOutcome",
            "projectKinds",
            "projectValues",
            "projectStates",
            "scenarioCategories",
            "exactStateAvailable",
        )
        if key in projection
    }
    winning_count = core.get("winningRootActionCount")
    core.setdefault(
        "winningRootActionLowerBound",
        winning_count if type(winning_count) is int else 0,
    )
    core.setdefault(
        "winningRootActionUpperBound",
        (
            winning_count
            if type(winning_count) is int
            and core.get("proofCompletion")
            in {"exact_complete", "bounded_complete", "policy_complete"}
            else None
        ),
    )
    if projection.get("exactStateAvailable") is True:
        core["decisionReference"] = {
            "resultCode": result_code,
            "decisionOrdinal": 1,
        }
    if type(predicate_evidence_count) is not int or predicate_evidence_count < 0:
        raise ValueError("Research V4 public predicate evidence count is invalid")
    core["predicateEvidenceCount"] = predicate_evidence_count
    core["publicProjectionHash"] = public_projection_hash
    if 'lessonTags' in projection:
        from .result_library import validate_result_lesson_tags_v1
        document = projection['lessonTags']
        validate_result_lesson_tags_v1(document)
        tags = {key: copy.deepcopy(document[key]) for key in
                ('schemaVersion', 'tags', 'playIndex', 'qualificationEvidence')}
        core['lessonTags'] = {**tags, 'sha256': canonical_sha256(tags)}
    return {**core, "sha256": canonical_sha256(core)}


def public_scenario_result_v4(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the actor-public V4 result without full-state authority fields."""

    result = dict(value)
    if result.get("schemaVersion") != 2:
        raise ValueError("Research V4 public result schema is invalid")
    result_code = str(result.get("resultCode") or "")
    satisfaction = result.get("scenarioSatisfaction")
    predicates = (
        satisfaction.get("predicates")
        if isinstance(satisfaction, Mapping)
        else None
    )
    if not isinstance(predicates, list):
        raise ValueError("Research V4 public predicate evidence is invalid")
    projection = public_scenario_projection_v1(
        dict(result.get("scenarioProjection") or {}),
        result_code=result_code,
        predicate_evidence_count=len(predicates),
    )
    qualification = result.get("qualification")
    if not isinstance(qualification, Mapping):
        raise ValueError("Research V4 public qualification is invalid")
    failures = qualification.get("failures")
    if not isinstance(failures, list) or any(
        not isinstance(item, str) for item in failures
    ):
        raise ValueError("Research V4 public qualification failures are invalid")
    public_qualification = {
        key: copy.deepcopy(qualification[key])
        for key in (
            "schemaVersion",
            "qualified",
            "failures",
            "observerSafe",
            "decisionPointAuditComplete",
            "legacyObserverSafetyUnverified",
        )
        if key in qualification
    }
    if isinstance(qualification.get("interestAdmission"), Mapping):
        public_qualification["interestAdmission"] = {
            key: copy.deepcopy(qualification["interestAdmission"][key])
            for key in ("policyHash", "evidenceHash", "passed", "score")
            if key in qualification["interestAdmission"]
        }
    public = {
        key: copy.deepcopy(result[key])
        for key in (
            "schemaVersion",
            "resultCode",
            "runCode",
            "recipeCode",
            "score",
            "truthLabel",
            "confidence",
            "disposition",
            "classification",
            "interestingSeats",
            "tacticalTags",
            "verifiedTacticalTags",
            "decisionCount",
            "libraryProjection",
            "createdAt",
        )
        if key in result
    }
    public.update(
        {
            "qualification": public_qualification,
            "proofOutcome": actor_public_proof_outcome_v1(
                dict(result.get("proofOutcome") or {})
            ),
            "scenarioProjection": projection,
            "publicProjectionHash": projection["publicProjectionHash"],
        }
    )
    if "decisionReference" in projection:
        public["decisionReference"] = copy.deepcopy(projection["decisionReference"])
    return public


def _proof_completion_coherent_v1(
    proof: Mapping[str, Any],
    legal_action_codes: list[str],
) -> bool:
    completion = proof.get("completionStatus")
    expected_mode = {
        "exact_complete": "exact",
        "bounded_complete": "bounded",
        "policy_complete": "policy_conditioned",
    }.get(completion)
    if expected_mode is None:
        return False
    if (
        proof.get("requestedMode") != expected_mode
        or proof.get("achievedMode") != expected_mode
        or proof.get("requestedVisibility")
        not in {"complete_state", "actor_information_set"}
        or proof.get("achievedVisibility") != proof.get("requestedVisibility")
        or proof.get("claimSatisfied") is not True
    ):
        return False
    hidden_worlds = proof.get("hiddenWorlds")
    if (
        not isinstance(hidden_worlds, Mapping)
        or type(hidden_worlds.get("count")) is not int
        or hidden_worlds["count"] < 1
        or type(hidden_worlds.get("complete")) is not bool
        or (
            proof.get("requestedVisibility") == "actor_information_set"
            and hidden_worlds["complete"] is not True
        )
    ):
        return False
    roots = proof.get("rootActions")
    if not isinstance(roots, list) or not roots:
        return False
    actions: list[str] = []
    winning = 0
    for raw in roots:
        if (
            not isinstance(raw, Mapping)
            or not isinstance(raw.get("action"), str)
            or not raw["action"]
            or type(raw.get("satisfies")) is not bool
            or type(raw.get("value")) not in {int, float}
        ):
            return False
        actions.append(str(raw["action"]))
        winning += int(raw["satisfies"])
    if (
        len(actions) != len(set(actions))
        or set(actions) != set(legal_action_codes)
        or type(proof.get("winningRootActionCount")) is not int
        or proof["winningRootActionCount"] != winning
        or proof.get("selectedAction") not in set(actions)
    ):
        return False
    claim = proof.get("claim")
    selected = next(
        raw for raw in roots if raw.get("action") == proof.get("selectedAction")
    )
    derived_claim = {
        "outcome_reachable": winning >= 1,
        "outcome_forced": winning == len(roots),
        "selected_action_is_unique_winner": (
            winning == 1 and selected.get("satisfies") is True
        ),
        "target_team_has_no_salvage": winning == 0,
    }.get(str(claim), True)
    return derived_claim is True


def _scenario_verdict_v1(
    *,
    satisfaction_status: str,
    satisfaction_hash: str,
    proof: Mapping[str, Any],
    proof_hash: str,
    observer_safety: Mapping[str, Any],
    legal_action_codes: list[str],
) -> dict[str, Any]:
    completion = str(proof.get("completionStatus") or "")
    completion_labelled = completion in {
        "exact_complete",
        "bounded_complete",
        "policy_complete",
    }
    proof_complete = _proof_completion_coherent_v1(proof, legal_action_codes)
    observer_safe = observer_safety.get("passed") is True
    qualified = satisfaction_status == "satisfied" and proof_complete and observer_safe
    failures: list[str] = []
    if satisfaction_status != "satisfied":
        failures.append(satisfaction_status)
    if not proof_complete:
        failures.append(
            "proof_semantics_incoherent" if completion_labelled else completion
        )
    if not observer_safe:
        failures.append("observer_safety_unverified")
    truth = (
        "exact_optimal"
        if proof_complete and completion == "exact_complete"
        else "robust_best_known"
        if proof_complete and completion in {"bounded_complete", "policy_complete"}
        else "rejected"
    )
    return {
        "schemaVersion": 1,
        "score": 100.0 if qualified else 0.0,
        "truthLabel": truth,
        "confidence": 1.0 if proof_complete else 0.0,
        "disposition": "qualified" if qualified else "rejected",
        "qualification": {
            "schemaVersion": 1,
            "qualified": qualified,
            "failures": failures,
            "scenarioSatisfactionHash": satisfaction_hash,
            "proofOutcomeHash": proof_hash,
            "observerSafe": observer_safe,
            "observerSafetyEvidenceHash": observer_safety.get("sha256"),
            "decisionPointAuditComplete": proof_complete,
        },
    }


def build_scenario_result_v4(
    *,
    run: Mapping[str, Any],
    recipe: ResearchRecipeV4,
    normalized: NormalizedExactStateV1,
    satisfaction: ScenarioSatisfactionV1,
    proof: ProofOutcomeV1,
    constraint_evidence: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
    created_at: str,
    generation_lane: str | None = None,
) -> dict[str, Any]:
    lane = generation_lane or (
        "exact_input"
        if recipe.scenario.origin.kind == "exact_trace_prefix"
        else str(recipe.resources["generationLane"])
    )
    projection = build_scenario_projection_v1(
        recipe,
        normalized,
        satisfaction,
        proof,
    )
    observer_safety = _observer_safety_evidence_v1(
        normalized,
        projection,
        proof,
    )
    verdict = _scenario_verdict_v1(
        satisfaction_status=satisfaction.status,
        satisfaction_hash=satisfaction.sha256,
        proof=proof.payload,
        proof_hash=_proof_semantic_sha256(proof.to_dict()),
        observer_safety=observer_safety,
        legal_action_codes=[str(item["code"]) for item in normalized.legal_actions],
    )
    artifact_hashes = {
        key: str(value["sha256"]) for key, value in sorted(artifacts.items())
    }
    score_evidence_core = {
        "schemaVersion": 1,
        "family": "scenario_v4",
        "scenarioSatisfactionHash": satisfaction.sha256,
        "proofOutcomeHash": _proof_semantic_sha256(proof.to_dict()),
        "observerSafetyEvidenceHash": observer_safety["sha256"],
        "isStructuralObjective": False,
        "score": verdict["score"],
    }
    score_evidence = {
        **score_evidence_core,
        "sha256": canonical_sha256(score_evidence_core),
    }
    identity = {
        "schemaVersion": 1,
        "kind": "research_scenario_result_identity_v1",
        "runCode": str(run["runCode"]),
        "runHash": str(run["runHash"]),
        "recipeCode": recipe.code,
        "recipeHash": recipe.sha256,
        "configurationAuthorityHash": canonical_sha256(recipe.configuration),
        "scenarioHash": recipe.scenario.sha256,
        "exactStateHash": normalized.sha256,
        "scenarioSatisfactionHash": satisfaction.sha256,
        "constraintEvidenceHash": str(constraint_evidence["sha256"]),
        "proofOutcomeHash": _proof_semantic_sha256(proof.to_dict()),
        "observerSafetyEvidenceHash": observer_safety["sha256"],
        "scenarioProjectionHash": projection["sha256"],
        "scoreEvidenceHash": score_evidence["sha256"],
        "verdictHash": canonical_sha256(verdict),
        "artifactHashes": artifact_hashes,
    }
    result_hash = canonical_sha256(identity)
    qualification_summary = dict(verdict["qualification"])
    qualification_summary.pop("observerSafetyEvidenceHash")
    return {
        "schemaVersion": 2,
        "resultCode": f"LAF-{result_hash[:10].upper()}",
        "resultHash": result_hash,
        "runCode": str(run["runCode"]),
        "recipeCode": recipe.code,
        "traceHash": normalized.state_hash,
        "gameHash": normalized.prefix_hash,
        "score": verdict["score"],
        "scoreEvidence": score_evidence,
        "truthLabel": verdict["truthLabel"],
        "confidence": verdict["confidence"],
        "disposition": verdict["disposition"],
        "qualification": {
            **qualification_summary,
            "observerSafetyEvidence": observer_safety,
        },
        "classification": "scenario_v4",
        "criticalDecisionIds": [normalized.exact_state_id],
        "interestingSeats": (
            [normalized.next_actor] if normalized.next_actor is not None else []
        ),
        "tacticalTags": [],
        "verifiedTacticalTags": [],
        "decisionCount": 1 if proof.payload["rootActions"] else 0,
        "scenarioSatisfaction": satisfaction.to_dict(),
        "constraintEvidence": copy.deepcopy(dict(constraint_evidence)),
        "proofOutcome": proof.to_dict(),
        "scenarioProjection": projection,
        "artifacts": {
            key: copy.deepcopy(dict(value)) for key, value in artifacts.items()
        },
        "provenance": {
            "schemaVersion": 1,
            "generationLane": lane,
            "recipeHash": recipe.sha256,
            "scenarioHash": recipe.scenario.sha256,
            "exactStateId": normalized.exact_state_id,
            "configurationAuthorityHash": identity["configurationAuthorityHash"],
            "resultIdentity": identity,
        },
        "libraryProjection": {
            "schemaVersion": 1,
            "actingSeats": (
                [normalized.next_actor] if normalized.next_actor is not None else []
            ),
            "observerSeats": (
                [normalized.next_actor] if normalized.next_actor is not None else []
            ),
            "contract": (
                normalized.round_state.contract.value
                if normalized.round_state.contract is not None
                else None
            ),
            "purchaseRound": None,
            "trumpSuit": (
                normalized.round_state.trump_suit.name.lower()
                if normalized.round_state.trump_suit is not None
                else None
            ),
            "escalation": normalized.round_state.escalation,
            "projectKinds": projection["projectKinds"],
            "generationLane": lane,
        },
        "createdAt": str(created_at),
    }


def _validate_score_evidence_v1(
    result: Mapping[str, Any],
    documents: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    evidence = result.get("scoreEvidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("Research V4 score evidence is invalid")
    keys = set(evidence)
    if keys not in (
        set(_SCORE_EVIDENCE_KEYS),
        {*_SCORE_EVIDENCE_KEYS, "observerSafetyEvidenceHash"},
    ):
        raise ValueError("Research V4 score evidence is not closed")
    if (
        type(evidence.get("schemaVersion")) is not int
        or evidence.get("schemaVersion") != 1
        or evidence.get("family") != "scenario_v4"
        or evidence.get("isStructuralObjective") is not False
    ):
        raise ValueError("Research V4 score evidence semantics mismatch")
    core = {
        key: copy.deepcopy(item) for key, item in evidence.items() if key != "sha256"
    }
    if canonical_sha256(core) != str(evidence.get("sha256") or ""):
        raise ValueError("Research V4 score evidence checksum mismatch")
    if (
        evidence.get("score") != result.get("score")
        or evidence.get("scenarioSatisfactionHash")
        != documents["scenarioSatisfaction"].get("sha256")
        or evidence.get("proofOutcomeHash")
        != _proof_semantic_sha256(documents["proofOutcome"])
    ):
        raise ValueError("Research V4 score evidence authority mismatch")
    return evidence


def _validate_observer_safety_qualification_v1(
    *,
    result: Mapping[str, Any],
    exact: Mapping[str, Any],
    proof: Mapping[str, Any],
    identity: Mapping[str, Any],
    score_evidence: Mapping[str, Any],
) -> bool:
    """Validate current evidence; return true for legacy fail-closed reads."""

    qualification = result.get("qualification")
    if not isinstance(qualification, Mapping):
        raise ValueError("Research V4 qualification is invalid")
    observer_evidence = qualification.get("observerSafetyEvidence")
    identity_hash = identity.get("observerSafetyEvidenceHash")
    score_hash = score_evidence.get("observerSafetyEvidenceHash")
    if observer_evidence is None:
        if identity_hash is not None or score_hash is not None:
            raise ValueError("Research V4 observer-safety evidence is missing")
        return True
    if not isinstance(observer_evidence, Mapping) or set(observer_evidence) != {
        "schemaVersion",
        "kind",
        "exactStateId",
        "observerSeat",
        "stateHash",
        "proofOutcomeHash",
        "projectionSha256",
        "engineProjectionSha256",
        "checks",
        "failureCode",
        "passed",
        "sha256",
    }:
        raise ValueError("Research V4 observer-safety evidence is invalid")
    checks = observer_evidence.get("checks")
    if (
        observer_evidence.get("schemaVersion") != 1
        or observer_evidence.get("kind") != "research_v4_observer_projection_v1"
        or observer_evidence.get("failureCode")
        not in {
            None,
            "observer_projection_unavailable",
            "observer_projection_mismatch",
        }
        or not _is_sha256(observer_evidence.get("sha256"))
        or not _is_sha256(observer_evidence.get("stateHash"))
        or not _is_sha256(observer_evidence.get("proofOutcomeHash"))
        or (
            observer_evidence.get("projectionSha256") is not None
            and not _is_sha256(observer_evidence.get("projectionSha256"))
        )
        or (
            observer_evidence.get("engineProjectionSha256") is not None
            and not _is_sha256(observer_evidence.get("engineProjectionSha256"))
        )
    ):
        raise ValueError("Research V4 observer-safety evidence semantics are invalid")
    if (
        not isinstance(checks, Mapping)
        or set(checks)
        != {
            "actingObserver",
            "proofStateBound",
            "exactStateMatchesEngine",
            "publicFieldsMatchEngineProjection",
            "handsMatchEngineProjection",
            "projectsMatchEngineProjection",
            "projectWinningTeamMatchesEngineProjection",
            "tricksMatchEngineProjection",
            "eventsMatchEngineProjection",
            "biddingMatchesEngineProjection",
            "legalActionsMatchEngineProjection",
        }
        or any(type(value) is not bool for value in checks.values())
    ):
        raise ValueError("Research V4 observer-safety checks are invalid")
    observer_passed = observer_evidence.get("passed")
    if (
        type(observer_passed) is not bool
        or observer_passed != all(checks.values())
        or qualification.get("observerSafe") is not observer_passed
    ):
        raise ValueError("Research V4 observer-safety verdict is invalid")
    observer_core = {
        key: copy.deepcopy(item)
        for key, item in observer_evidence.items()
        if key != "sha256"
    }
    if canonical_sha256(observer_core) != str(observer_evidence.get("sha256") or ""):
        raise ValueError("Research V4 observer-safety evidence checksum mismatch")
    failures = qualification.get("failures")
    if not isinstance(failures, list):
        raise ValueError("Research V4 qualification failures are invalid")
    if observer_passed:
        if observer_evidence.get("failureCode") is not None:
            raise ValueError("Research V4 observer-safety success has a failure code")
        if any(
            not _is_sha256(observer_evidence.get(key))
            for key in ("projectionSha256", "engineProjectionSha256")
        ):
            raise ValueError(
                "Research V4 observer-safety projection hashes are invalid"
            )
        if "observer_safety_unverified" in failures:
            raise ValueError("Research V4 observer-safety success retained a failure")
    elif (
        not str(observer_evidence.get("failureCode") or "")
        or "observer_safety_unverified" not in failures
        or qualification.get("qualified") is not False
        or result.get("disposition") != "rejected"
    ):
        raise ValueError("Research V4 observer-safety failure did not fail closed")
    if identity_hash != observer_evidence.get("sha256"):
        raise ValueError("Research V4 observer-safety evidence membership mismatch")
    if score_hash != observer_evidence.get("sha256"):
        raise ValueError("Research V4 score omits observer-safety evidence")
    if (
        observer_evidence.get("exactStateId") != exact.get("exactStateId")
        or observer_evidence.get("observerSeat") != exact.get("nextActor")
        or observer_evidence.get("stateHash") != exact.get("stateHash")
        or observer_evidence.get("proofOutcomeHash") != _proof_semantic_sha256(proof)
    ):
        raise ValueError("Research V4 observer-safety authority mismatch")
    return False


def _validate_current_verdict_v1(
    *,
    result: Mapping[str, Any],
    exact: Mapping[str, Any],
    satisfaction: Mapping[str, Any],
    proof: Mapping[str, Any],
    projection: Mapping[str, Any],
    identity: Mapping[str, Any],
    score_evidence: Mapping[str, Any],
) -> None:
    qualification = result.get("qualification")
    if not isinstance(qualification, Mapping):
        raise ValueError("Research V4 qualification is invalid")
    observer_safety = qualification.get("observerSafetyEvidence")
    if not isinstance(observer_safety, Mapping):
        raise ValueError("Research V4 observer-safety evidence is missing")
    verdict = _scenario_verdict_v1(
        satisfaction_status=str(satisfaction.get("status") or ""),
        satisfaction_hash=str(satisfaction.get("sha256") or ""),
        proof=proof,
        proof_hash=_proof_semantic_sha256(proof),
        observer_safety=observer_safety,
        legal_action_codes=[
            str(item.get("code") or "")
            for item in exact.get("legalActions") or ()
            if isinstance(item, Mapping)
        ],
    )
    expected_qualification = dict(verdict["qualification"])
    expected_qualification.pop("observerSafetyEvidenceHash")
    expected_qualification["observerSafetyEvidence"] = observer_safety
    if dict(qualification) != expected_qualification:
        raise ValueError("Research V4 qualification verdict mismatch")
    if any(
        result.get(key) != verdict[key]
        for key in ("score", "truthLabel", "confidence", "disposition")
    ):
        raise ValueError("Research V4 result verdict mismatch")
    if identity.get("scoreEvidenceHash") != score_evidence.get("sha256"):
        raise ValueError("Research V4 result omits score evidence")
    if identity.get("scenarioProjectionHash") != projection.get("sha256"):
        raise ValueError("Research V4 result omits scenario projection")
    if identity.get("verdictHash") != canonical_sha256(verdict):
        raise ValueError("Research V4 result verdict membership mismatch")
    actor = exact.get("nextActor")
    roots = proof.get("rootActions")
    if (
        actor not in {"north", "west", "south", "east"}
        or result.get("classification") != "scenario_v4"
        or result.get("interestingSeats") != [actor]
        or result.get("criticalDecisionIds") != [exact.get("exactStateId")]
        or result.get("decisionCount")
        != (1 if isinstance(roots, list) and roots else 0)
        or result.get("tacticalTags") != []
        or result.get("verifiedTacticalTags") != []
    ):
        raise ValueError("Research V4 public result metadata mismatch")
    library_projection = result.get("libraryProjection")
    provenance = result.get("provenance")
    state = exact.get("state")
    if (
        not isinstance(library_projection, Mapping)
        or set(library_projection)
        != {
            "schemaVersion",
            "actingSeats",
            "observerSeats",
            "contract",
            "purchaseRound",
            "trumpSuit",
            "escalation",
            "projectKinds",
            "generationLane",
        }
        or not isinstance(provenance, Mapping)
        or not isinstance(state, Mapping)
    ):
        raise ValueError("Research V4 library projection is invalid")
    expected_library_projection = {
        "schemaVersion": 1,
        "actingSeats": [actor],
        "observerSeats": [actor],
        "contract": state.get("contract"),
        "purchaseRound": None,
        "trumpSuit": state.get("trumpSuit"),
        "escalation": state.get("escalation"),
        "projectKinds": projection.get("projectKinds"),
        "generationLane": provenance.get("generationLane"),
    }
    if dict(library_projection) != expected_library_projection:
        raise ValueError("Research V4 library projection authority mismatch")
    if set(projection) != set(_SCENARIO_PROJECTION_KEYS):
        raise ValueError("Research V4 scenario projection is not closed")
    project_lists = (
        projection.get("projectKinds"),
        projection.get("projectValues"),
        projection.get("projectStates"),
    )
    if any(
        not isinstance(items, list)
        or any(not isinstance(item, str) or not item for item in items)
        or items != sorted(set(items))
        for items in project_lists
    ):
        raise ValueError("Research V4 scenario project projection is invalid")
    if any(not item.isdigit() for item in projection["projectValues"]):
        raise ValueError("Research V4 scenario project values are invalid")
    if any(
        item not in {"eligible", "declared", "comparison_winner", "awarded"}
        for item in projection["projectStates"]
    ):
        raise ValueError("Research V4 scenario project states are invalid")
    if not _is_sha256(projection.get("publicProjectionHash")):
        raise ValueError("Research V4 public projection hash is invalid")
    round_score = state.get("roundScore")
    declarer = state.get("declarer")
    contract_outcome = None
    if isinstance(round_score, Mapping) and declarer in {
        "north",
        "west",
        "south",
        "east",
    }:
        declarer_team = "north_south" if declarer in {"north", "south"} else "west_east"
        contract_outcome = (
            "won" if round_score.get("winner") == declarer_team else "lost"
        )
    scenario_hash = str(identity.get("scenarioHash") or "")
    expected_projection = {
        "schemaVersion": 1,
        "scenarioTemplateId": f"SCN-{scenario_hash[:12].upper()}",
        "status": satisfaction.get("status"),
        "decisionKind": exact.get("decisionKind"),
        "targetTeam": proof.get("targetTeam"),
        "requestedProofMode": proof.get("requestedMode"),
        "achievedProofMode": proof.get("achievedMode"),
        "proofCompletion": proof.get("completionStatus"),
        "proofVisibility": proof.get("achievedVisibility"),
        "winningRootActionCount": proof.get("winningRootActionCount"),
        "winningRootActionLowerBound": proof.get("winningRootActionLowerBound"),
        "winningRootActionUpperBound": proof.get("winningRootActionUpperBound"),
        "contractOutcome": contract_outcome,
        "exactStateId": exact.get("exactStateId"),
        "exactStateAvailable": True,
    }
    if any(
        projection.get(key) != expected for key, expected in expected_projection.items()
    ):
        raise ValueError("Research V4 scenario projection authority mismatch")


def validate_scenario_result_v4(
    value: Mapping[str, Any],
    *,
    artifact_store: Any,
) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    if result.get("schemaVersion") != 2:
        raise ValueError("Research V4 scenario result schema is invalid")
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(
        SCENARIO_ARTIFACT_MEDIA_TYPES
    ):
        raise ValueError("Research V4 scenario artifact set is not closed")
    documents: dict[str, dict[str, Any]] = {}
    for key, media_type in SCENARIO_ARTIFACT_MEDIA_TYPES.items():
        reference = artifacts[key]
        if not isinstance(reference, Mapping) or set(reference) != {
            "uri",
            "sha256",
            "bytes",
            "mediaType",
            "name",
        }:
            raise ValueError(f"Research V4 {key} artifact reference is not closed")
        if reference["mediaType"] != media_type:
            raise ValueError(f"Research V4 {key} artifact media type mismatch")
        payload = artifact_store.get(str(reference["uri"]))
        if len(payload) != int(reference["bytes"]) or hashlib.sha256(
            payload
        ).hexdigest() != str(reference["sha256"]):
            raise ValueError(f"Research V4 {key} artifact integrity mismatch")
        try:
            document = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Research V4 {key} artifact is invalid JSON") from exc
        if (
            not isinstance(document, Mapping)
            or canonical_json_bytes(document) != payload
        ):
            raise ValueError(f"Research V4 {key} artifact is not canonical JSON")
        documents[key] = copy.deepcopy(dict(document))
    for key, inline_key in (
        ("scenarioSatisfaction", "scenarioSatisfaction"),
        ("constraintEvidence", "constraintEvidence"),
        ("proofOutcome", "proofOutcome"),
    ):
        expected_inline = (
            public_proof_outcome_v1(documents[key])
            if key == "proofOutcome"
            else documents[key]
        )
        if result.get(inline_key) not in (documents[key], expected_inline):
            raise ValueError(f"Research V4 {key} inline evidence mismatch")
    public_proof = public_proof_outcome_v1(documents["proofOutcome"])
    if public_proof.get("artifactSha256") != artifacts["proofOutcome"].get("sha256"):
        raise ValueError("Research V4 public proof artifact identity mismatch")
    exact = documents["exactState"]
    exact_core = {
        key: copy.deepcopy(exact[key])
        for key in (
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
    }
    legacy_exact_core = {
        **exact_core,
        "projectDeclarationsVerified": exact.get("projectDeclarationsVerified"),
        "projectAssertionsVerified": exact.get("projectAssertionsVerified"),
    }
    if str(exact.get("sha256") or "") not in {
        canonical_sha256(exact_core),
        canonical_sha256(legacy_exact_core),
    }:
        raise ValueError("Research V4 exact-state semantic checksum mismatch")
    projection = dict(result["scenarioProjection"])
    if str(exact.get("decisionKind") or "") != str(
        projection.get("decisionKind") or ""
    ) or str(exact.get("exactStateId") or "") != str(
        projection.get("exactStateId") or ""
    ):
        raise ValueError("Research V4 exact-state decision projection mismatch")
    public_exact_state = build_observer_safe_exact_state_v1(
        exact_state=exact,
        scenario_projection=projection,
        observer_seat=str(exact.get("nextActor") or ""),
    )
    satisfaction = documents["scenarioSatisfaction"]
    predicate_hashes = []
    for row_value in satisfaction.get("predicates") or ():
        row = dict(row_value)
        row_core = {
            key: copy.deepcopy(row[key])
            for key in (
                "predicateId",
                "predicateHash",
                "kind",
                "hardness",
                "status",
                "constructionCapability",
                "runtimeVerifier",
                "observed",
                "exactStateId",
                "failureCode",
            )
        }
        if canonical_sha256(row_core) != str(row.get("sha256") or ""):
            raise ValueError("Research V4 predicate evidence checksum mismatch")
        predicate_hashes.append(str(row["sha256"]))
    satisfaction_core = {
        "schemaVersion": 1,
        "scenarioHash": satisfaction.get("scenarioHash"),
        "exactStateId": satisfaction.get("exactStateId"),
        "constraintsSatisfied": satisfaction.get("constraintsSatisfied"),
        "outcomeSatisfied": satisfaction.get("outcomeSatisfied"),
        "hardPredicatesSatisfied": satisfaction.get("hardPredicatesSatisfied"),
        "status": satisfaction.get("status"),
        "predicateEvidenceHashes": predicate_hashes,
    }
    if canonical_sha256(satisfaction_core) != str(satisfaction.get("sha256") or ""):
        raise ValueError("Research V4 scenario satisfaction checksum mismatch")
    constraint = documents["constraintEvidence"]
    if canonical_sha256(
        {
            key: copy.deepcopy(item)
            for key, item in constraint.items()
            if key != "sha256"
        }
    ) != str(constraint.get("sha256") or ""):
        raise ValueError("Research V4 constraint evidence checksum mismatch")
    ProofOutcomeV1.build(documents["proofOutcome"])
    proof_identities = documents["proofOutcome"].get("identities")
    if not isinstance(proof_identities, Mapping) or (
        proof_identities.get("rulesHash") != exact.get("rulesHash")
        or proof_identities.get("stateHash") != exact.get("stateHash")
        or proof_identities.get("exactStateId") != exact.get("exactStateId")
    ):
        raise ValueError("Research V4 proof and exact-state authority mismatch")
    if canonical_sha256(
        {
            key: copy.deepcopy(value)
            for key, value in projection.items()
            if key != "sha256"
        }
    ) != str(projection["sha256"]):
        raise ValueError("Research V4 scenario projection checksum mismatch")
    identity = dict(dict(result.get("provenance") or {}).get("resultIdentity") or {})
    if not identity or canonical_sha256(identity) != str(
        result.get("resultHash") or ""
    ):
        raise ValueError("Research V4 result identity mismatch")
    score_evidence = _validate_score_evidence_v1(result, documents)
    legacy_observer_safety = _validate_observer_safety_qualification_v1(
        result=result,
        exact=exact,
        proof=documents["proofOutcome"],
        identity=identity,
        score_evidence=score_evidence,
    )
    if not legacy_observer_safety:
        _validate_current_verdict_v1(
            result=result,
            exact=exact,
            satisfaction=documents["scenarioSatisfaction"],
            proof=documents["proofOutcome"],
            projection=projection,
            identity=identity,
            score_evidence=score_evidence,
        )
    expected_artifact_hashes = {
        key: str(reference["sha256"]) for key, reference in sorted(artifacts.items())
    }
    if identity.get("artifactHashes") != expected_artifact_hashes:
        raise ValueError("Research V4 result artifact membership mismatch")
    expected_semantic_hashes = {
        "exactStateHash": documents["exactState"].get("sha256"),
        "scenarioSatisfactionHash": documents["scenarioSatisfaction"].get("sha256"),
        "constraintEvidenceHash": documents["constraintEvidence"].get("sha256"),
        "proofOutcomeHash": _proof_semantic_sha256(documents["proofOutcome"]),
    }
    if any(
        identity.get(key) != expected
        for key, expected in expected_semantic_hashes.items()
    ):
        raise ValueError("Research V4 result semantic evidence membership mismatch")
    if str(result.get("traceHash") or "") != str(
        documents["exactState"].get("stateHash") or ""
    ) or str(result.get("gameHash") or "") != str(
        documents["exactState"].get("prefixHash") or ""
    ):
        raise ValueError("Research V4 exact-state result membership mismatch")
    if legacy_observer_safety:
        if not _is_sha256(projection.get("publicProjectionHash")):
            projection = {
                **projection,
                "publicProjectionHash": public_exact_state["publicProjectionHash"],
            }
            projection["sha256"] = canonical_sha256(
                {key: item for key, item in projection.items() if key != "sha256"}
            )
            result["scenarioProjection"] = projection
        qualification = copy.deepcopy(dict(result["qualification"]))
        failures = list(qualification.get("failures") or [])
        if "observer_safety_unverified" not in failures:
            failures.append("observer_safety_unverified")
        qualification.update(
            {
                "qualified": False,
                "failures": failures,
                "observerSafe": False,
                "observerSafetyEvidence": None,
                "legacyObserverSafetyUnverified": True,
            }
        )
        result["qualification"] = qualification
        result["disposition"] = "rejected"
    result["proofOutcome"] = public_proof_outcome_v1(documents["proofOutcome"])
    return result


def scenario_decision_audit_v1(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Project one observer-safe decision audit from a validated V4 result."""

    result = dict(value)
    proof = actor_public_proof_outcome_v1(dict(result["proofOutcome"]))
    projection = dict(result["scenarioProjection"])
    root_actions = [copy.deepcopy(dict(item)) for item in proof["rootActions"]]
    if not root_actions:
        return []
    selected_code = str(proof.get("selectedAction") or "")
    selected = next(
        (item for item in root_actions if str(item.get("action")) == selected_code),
        root_actions[0],
    )
    actor = (result.get("interestingSeats") or [None])[0]
    if actor not in {"north", "west", "south", "east"}:
        raise ValueError("Research V4 scenario decision actor is unavailable")
    team = "north_south" if actor in {"north", "south"} else "west_east"
    complete = str(proof["completionStatus"]) in {
        "exact_complete",
        "bounded_complete",
        "policy_complete",
    }
    truth = str(result["truthLabel"])
    public_hash = str(projection.get("publicProjectionHash") or "")
    if not _is_sha256(public_hash):
        raise ValueError("Research V4 scenario audit lacks public projection authority")
    decision_reference = {
        "resultCode": str(result["resultCode"]),
        "decisionOrdinal": 1,
    }
    information_hash = canonical_sha256(
        {"publicStateHash": public_hash, "observer": actor}
    )
    return [
        {
            "schemaVersion": 1,
            "decisionId": str(result["resultCode"]),
            "ordinal": 1,
            "decisionOrdinal": 1,
            "actingSeat": actor,
            "actingTeam": team,
            "decisionKind": str(projection["decisionKind"]),
            "phase": "scenario_decision",
            "publicStateHash": public_hash,
            "informationStateHash": information_hash,
            "legalActions": [{"action": item["action"]} for item in root_actions],
            "rulePaths": [],
            "actualAction": {"action": selected["action"]},
            "actionSource": "scenario_proof_principal_variation",
            "recommendedAction": {"action": selected["action"]},
            "recommendationAvailable": complete,
            "actualMatchesRecommendation": complete,
            "forced": len(root_actions) == 1,
            "alternatives": root_actions,
            "expectedRegret": 0.0,
            "probabilityDrop": 0.0,
            "catastrophicCrossing": False,
            "hiddenDealSupport": int(dict(proof["hiddenWorlds"])["count"]),
            "effectiveSampleSize": float(dict(proof["hiddenWorlds"])["count"]),
            "confidence": float(result["confidence"]),
            "convergence": str(proof["completionStatus"]),
            "principalContinuation": [],
            "supportedMetrics": [
                "proof_value",
                "winning_root_action_count",
                "continuation_line_count",
            ],
            "unsupportedMetrics": [],
            "selectionExplanation": {
                "kind": "scenario_proof",
                "proofBoundary": proof["completionStatus"],
            },
            "analyticalExplanation": {
                "claim": proof["claim"],
                "selectedValue": selected.get("value"),
                "satisfies": bool(selected.get("satisfies")),
                "winningRootActionCount": proof["winningRootActionCount"],
            },
            "proof": {
                "requestedMode": proof["requestedMode"],
                "achievedMode": proof["achievedMode"],
                "visibility": proof["achievedVisibility"],
                "completionStatus": proof["completionStatus"],
                "claim": proof["claim"],
                "claimSatisfied": proof["claimSatisfied"],
                "winningRootActionCount": proof["winningRootActionCount"],
            },
            "forcedExplanation": None,
            "tacticalEvidence": [],
            "actualDealOracle": None,
            "truthLabel": truth,
            "eventSequence": None,
            "eventHash": None,
            "provenance": {
                "decisionReference": decision_reference,
                "publicProjectionHash": public_hash,
                "observerSafe": (
                    dict(result.get("qualification") or {}).get("observerSafe") is True
                ),
            },
        }
    ]


__all__ = [
    "SCENARIO_ARTIFACT_MEDIA_TYPES",
    "actor_public_proof_outcome_v1",
    "build_constraint_evidence_v1",
    "build_scenario_projection_v1",
    "build_scenario_result_v4",
    "canonical_json_bytes",
    "public_proof_outcome_v1",
    "public_scenario_projection_v1",
    "public_scenario_result_v4",
    "validate_scenario_result_v4",
    "scenario_decision_audit_v1",
]
