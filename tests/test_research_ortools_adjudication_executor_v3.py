from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from baloot_platform.core.rules import RuleBook
from baloot_platform.research.ortools_research_executor import (
    OrToolsUnifiedResearchExecutorV1,
)
from baloot_platform.research.scenario_contracts import (
    ResearchScenarioV1,
    ScenarioProofRequestV1,
    canonical_sha256,
)


ACTIVE_AUTHORITY_SHA256 = (
    "a347741d373c9de8761aaa08d2ba593b0e9301ce97bcb928ac2a60df7d4695a8"
)
FALSE_EKKA_AUTHORITY_SHA256 = (
    "dc7f00364c176e7c10506b31f649a682bc9af526ccbdbe8310f4161719b3431c"
)


@pytest.fixture(scope="module")
def adjudication_rules(root: Path) -> RuleBook:
    return RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")


def _leaf(
    predicate_id: str,
    kind: str,
    arguments: dict,
    hardness: str = "hard",
) -> dict:
    return {
        "predicateId": predicate_id,
        "kind": kind,
        "version": 1,
        "hardness": hardness,
        "arguments": arguments,
    }


def _policy_proof(*, selected_action: str | None = None) -> ScenarioProofRequestV1:
    policies: dict[str, dict] = {}
    for seed, seat in enumerate(("north", "west", "south", "east"), start=1):
        core = {
            "schemaVersion": 1,
            "seat": seat,
            "policyId": "fixed_action_map_v1",
            "version": "1.0.0",
            "seed": seed,
            "choices": [],
            "materialization": "seeded_legal_action_order_v1",
        }
        policy = {key: value for key, value in core.items() if key != "seat"}
        policy["profileHash"] = canonical_sha256(core)
        policies[seat] = policy
    value = {
        "schemaVersion": 1,
        "mode": "policy_conditioned",
        "visibility": "complete_state",
        "strategyClass": "frozen_policy",
        "claim": "outcome_reachable",
        "targetTeam": "north_south",
        "utility": "awarded_point_margin",
        "successThreshold": -100000,
        "lineUniverse": "policy_conditioned",
        "policies": policies,
        "budgetProfile": "late_game_exact_v1",
    }
    if selected_action is not None:
        value["selectedAction"] = selected_action
    return ScenarioProofRequestV1.build(value)


def _bounded_proof() -> ScenarioProofRequestV1:
    return ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "bounded",
            "visibility": "complete_state",
            "claim": "outcome_reachable",
            "targetTeam": "north_south",
            "utility": "awarded_point_margin",
            "successThreshold": -100000,
            "horizon": {"unit": "plies", "value": 1},
            "boundaryEvaluator": "current_raw_point_margin_v1",
            "budgetProfile": "test_bounded",
        }
    )


def _executor(root: Path, rules: RuleBook) -> OrToolsUnifiedResearchExecutorV1:
    authority = yaml.safe_load(
        (root / "configs/adjudication/research_events_v2.yaml").read_text(
            encoding="utf-8"
        )
    )
    return OrToolsUnifiedResearchExecutorV1(
        rules,
        bundle_revision_id="CBR-TEST-ADJUDICATION-V3",
        bundle_hash="a" * 64,
        backend_id="ortools_cp_sat_v3",
        adjudication_authority=authority,
    )


def _false_ekka_executor(
    root: Path, rules: RuleBook
) -> OrToolsUnifiedResearchExecutorV1:
    authority = yaml.safe_load(
        (
            root
            / "configs/adjudication/research_events_false_ekka_unratified_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    return OrToolsUnifiedResearchExecutorV1(
        rules,
        bundle_revision_id="CBR-TEST-FALSE-EKKA-UNRATIFIED",
        bundle_hash="f" * 64,
        backend_id="ortools_cp_sat_v3",
        adjudication_authority=authority,
    )


def _scenario(*, origin: dict, kind: str, arguments: dict, decision: dict) -> ResearchScenarioV1:
    return ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": origin,
            "constraints": _leaf("PRED-ADJUDICATION", kind, arguments),
            "decisionPoint": decision,
            "outcome": _leaf(
                "PRED-ADJUDICATION-REPORT",
                "outcome.round_margin",
                {
                    "team": "north_south",
                    "minimum": -100000,
                    "maximum": 100000,
                },
                "report",
            ),
        }
    )


def _assert_native_evidence(
    result,
    *,
    kind: str,
    same_actor: bool = False,
    authority_hash: str = ACTIVE_AUTHORITY_SHA256,
) -> dict:
    assert result.status == "qualified", {
        "execution": result.execution_receipt,
        "predicates": (
            result.predicate_evaluation.to_dict()
            if result.predicate_evaluation is not None else None
        ),
    }
    assert result.predicate_evaluation is not None
    evidence = next(
        row
        for row in result.predicate_evaluation.to_dict()["evidence"]
        if row["predicateId"] == "PRED-ADJUDICATION"
    )
    observed = evidence["observed"]
    assert evidence["satisfied"] is True
    assert observed["kind"] == kind
    assert observed["authorityHash"] == authority_hash
    assert observed["ordinaryActionHistoryMutated"] is False
    assert observed["requiresTableReferee"] is True
    assert (observed["expectedActor"] == observed["attemptedActor"]) is same_actor
    return observed


def test_false_ekka_search_is_variant_only_native_rejected_attempt(
    root: Path,
    adjudication_rules: RuleBook,
) -> None:
    official = _executor(root, adjudication_rules)
    assert official.lowering_manifest.definition(
        "adjudication.false_ekka_attempt"
    ) is None

    result = _false_ekka_executor(root, adjudication_rules).execute(
        _scenario(
            origin={"kind": "search"},
            kind="adjudication.false_ekka_attempt",
            arguments={"attemptedActor": "west", "attemptedCardId": 0},
            decision={"kind": "play", "selector": {"trickIndex": 6}},
        ),
        _bounded_proof(),
        maximum_candidates=2,
        maximum_seconds=25,
        maximum_proof_nodes=2_000,
        maximum_proof_depth=64,
    )

    observed = _assert_native_evidence(
        result,
        kind="false_ekka_attempt",
        same_actor=True,
        authority_hash=FALSE_EKKA_AUTHORITY_SHA256,
    )
    assert observed["attemptedAction"] == {"type": "card", "cardId": 0}


def test_v3_executor_fails_closed_without_bundle_bound_adjudication_authority(
    adjudication_rules: RuleBook,
) -> None:
    scenario = _scenario(
        origin={"kind": "search"},
        kind="adjudication.purchase_before_turn_attempt",
        arguments={"attemptedActor": "west", "purchaseToken": "sun"},
        decision={"kind": "bid", "selector": {"actionIndex": 0}},
    )
    result = OrToolsUnifiedResearchExecutorV1(
        adjudication_rules,
        bundle_revision_id="CBR-TEST-ADJUDICATION-MISSING",
        bundle_hash="a" * 64,
        backend_id="ortools_cp_sat_v3",
    ).execute(scenario, _bounded_proof(), maximum_seconds=1)

    assert result.status == "incomplete"
    assert result.candidate is None
    assert result.preflight.unsupported == (
        {
            "predicateId": "PRED-ADJUDICATION",
            "code": "adjudication_authority_unavailable",
        },
    )


def test_selected_adjudication_authority_cannot_substitute_historical_manifest(
    root: Path,
    adjudication_rules: RuleBook,
) -> None:
    active = _executor(root, adjudication_rules)
    historical = yaml.safe_load(
        (root / "configs/adjudication/research_events_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    old = OrToolsUnifiedResearchExecutorV1(
        adjudication_rules,
        bundle_revision_id="CBR-TEST-ADJUDICATION-HISTORICAL",
        bundle_hash="b" * 64,
        backend_id="ortools_cp_sat_v3",
        adjudication_authority=historical,
    )

    assert active.lowering_manifest.catalog()["sha256"] == (
        active.lowering_manifest.manifest_hash
    )
    assert old.lowering_manifest.catalog()["sha256"] == old.lowering_manifest.manifest_hash
    assert active.lowering_manifest.manifest_hash != old.lowering_manifest.manifest_hash
    assert active.adjudication_authority is not None
    assert active.adjudication_authority.sha256 == ACTIVE_AUTHORITY_SHA256


@pytest.mark.parametrize(
    ("kind", "arguments", "decision", "proof", "evidence_kind"),
    (
        (
            "adjudication.purchase_before_turn_attempt",
            {"attemptedActor": "west", "purchaseToken": "sun"},
            {"kind": "bid", "selector": {"actionIndex": 0}},
            _policy_proof(selected_action="sun"),
            "purchase_before_turn",
        ),
        (
            "adjudication.out_of_turn_attempt",
            {"attemptedActor": "west", "attemptedCardId": 0},
            {"kind": "play", "selector": {"trickIndex": 6}},
            _bounded_proof(),
            "out_of_turn",
        ),
        (
            "adjudication.purchase_before_turn_attempt",
            {"attemptedActor": "north", "purchaseToken": "sun"},
            {"kind": "play", "selector": {"trickIndex": 6}},
            _bounded_proof(),
            "purchase_before_turn",
        ),
    ),
)
def test_generated_search_qualifies_only_the_native_attempted_action_branch(
    root: Path,
    adjudication_rules: RuleBook,
    kind: str,
    arguments: dict,
    decision: dict,
    proof: ScenarioProofRequestV1,
    evidence_kind: str,
) -> None:
    result = _executor(root, adjudication_rules).execute(
        _scenario(
            origin={"kind": "search"},
            kind=kind,
            arguments=arguments,
            decision=decision,
        ),
        proof,
        maximum_candidates=2,
        maximum_seconds=25,
        maximum_proof_nodes=2_000,
        maximum_proof_depth=64,
    )

    observed = _assert_native_evidence(result, kind=evidence_kind)
    assert "exactSidecarSha256" not in observed


@pytest.mark.parametrize("decision_kind", ("bid", "play", "historical_purchase"))
def test_exact_input_binds_sidecar_without_mutating_ordinary_history(
    root: Path,
    adjudication_rules: RuleBook,
    decision_kind: str,
) -> None:
    fixture = json.loads(
        (
            root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json"
        ).read_text(encoding="utf-8")
    )
    prefix = copy.deepcopy(fixture["exactTracePrefix"])
    if decision_kind == "bid":
        prefix.update(
            {
                "auctionActions": [],
                "escalationActions": [],
                "projectDeclarations": [],
                "projectAssertions": [],
                "playActions": [],
                "stateAssertions": {},
                "adjudicationEvents": [
                    {
                        "schemaVersion": 1,
                        "kind": "invalid_bid",
                        "actionSequence": 0,
                        "actorSeat": "west",
                        "expectedActorSeat": "north",
                        "attemptedAction": {"token": "sun"},
                        "reasonCode": "purchase_before_turn",
                    }
                ],
            }
        )
        predicate_kind = "adjudication.purchase_before_turn_attempt"
        arguments = {"attemptedActor": "west", "purchaseToken": "sun"}
        decision = {"kind": "bid", "selector": {"actionIndex": 0}}
        proof = _policy_proof(selected_action="sun")
        evidence_kind = "purchase_before_turn"
    elif decision_kind == "historical_purchase":
        prefix["adjudicationEvents"] = [{
            "schemaVersion": 1,
            "kind": "invalid_bid",
            "actionSequence": 0,
            "actorSeat": "west",
            "expectedActorSeat": "north",
            "attemptedAction": {"token": "sun"},
            "reasonCode": "purchase_before_turn",
        }]
        predicate_kind = "adjudication.purchase_before_turn_attempt"
        arguments = {"attemptedActor": "west", "purchaseToken": "sun"}
        decision = {"kind": "play", "selector": {"trickIndex": 5, "actingSeat": "west"}}
        proof = ScenarioProofRequestV1.build(fixture["proof"])
        evidence_kind = "purchase_before_turn"
    else:
        prefix["adjudicationEvents"] = [
            {
                "schemaVersion": 1,
                "kind": "out_of_turn",
                "actionSequence": 23,
                "actorSeat": "south",
                "expectedActorSeat": "west",
                "attemptedAction": {"cardId": 0},
                "reasonCode": "play_before_turn",
            }
        ]
        predicate_kind = "adjudication.out_of_turn_attempt"
        arguments = {"attemptedActor": "south", "attemptedCardId": 0}
        decision = {
            "kind": "play",
            "selector": {"trickIndex": 5, "actingSeat": "west"},
        }
        proof = ScenarioProofRequestV1.build(fixture["proof"])
        evidence_kind = "out_of_turn"
    ordinary_counts = (
        len(prefix["auctionActions"]),
        len(prefix["playActions"]),
    )

    result = _executor(root, adjudication_rules).execute(
        _scenario(
            origin={"kind": "exact_trace_prefix", "exactTracePrefix": prefix},
            kind=predicate_kind,
            arguments=arguments,
            decision=decision,
        ),
        proof,
        maximum_candidates=1,
        maximum_seconds=25,
        maximum_proof_nodes=100_000,
        maximum_proof_depth=64,
    )

    observed = _assert_native_evidence(result, kind=evidence_kind)
    assert observed["exactSidecarSha256"]
    assert result.candidate is not None
    exact = result.candidate.trace["exactTracePrefix"]
    assert (len(exact["auctionActions"]), len(exact["playActions"])) == ordinary_counts
    report = result.candidate.trace["adjudicationReport"]
    assert report["authoritySha256"] == ACTIVE_AUTHORITY_SHA256
    assert report["events"][0]["eventSha256"] == observed["exactSidecarSha256"]
