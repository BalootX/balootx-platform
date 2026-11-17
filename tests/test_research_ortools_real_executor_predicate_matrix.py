from __future__ import annotations

import copy
import json
from enum import Enum
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from deterministic_work import deterministic_work_clock_v1
from ortools.sat.python import cp_model

from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Contract, Seat, Suit
from baloot_platform.core.models import LegalPlayActionV1
from baloot_platform.core.scoring_ir import ScoringDomainIRV1
from baloot_platform.research.ortools_proof import (
    OrToolsScenarioGraphProofV1,
    ProofGraphEdgeV1,
    ProofGraphNodeV1,
    ProofGraphV1,
)
from baloot_platform.research.ortools_escalation_stage import (
    OrToolsEscalationStageV1,
)
from baloot_platform.research.ortools_play_tail import OrToolsPlayTailStageV1
from baloot_platform.research.ortools_project_model import (
    OrToolsProjectLifecycleModelV1,
)
from baloot_platform.research.ortools_research_executor import (
    DefaultStageContextCompilerV1,
    GenericCompleteTraceCompilerV1,
    OrToolsCandidateAttemptV1,
    OrToolsNeutralCandidateV1,
    OrToolsResearchCompilerV1,
    OrToolsUnifiedResearchExecutorV1,
    _native_proof_tactic_configurer_v3,
    _native_play_context_v3,
    _proof_stage_bindings_v1,
    _search_decision_prefix,
    _stage_receipt,
)
from baloot_platform.research.ortools_scenario_model import OrToolsScenarioModelV1
from baloot_platform.research.scenario_contracts import (
    ExactTracePrefixV1,
    ResearchScenarioV1,
    ScenarioProofRequestV1,
    canonical_sha256,
)
from baloot_platform.research.scenario_predicate_manifest_v2 import (
    ScenarioPredicateCatalogLoweringManifestV2,
)
from baloot_platform.research.scenario_adjudication_predicates_v3 import (
    shared_scenario_predicate_catalog_lowering_manifest_v3,
)


_MANIFEST = ScenarioPredicateCatalogLoweringManifestV2()
_CASES = tuple(
    (origin, definition, counterexample)
    for origin in ("search", "exact_trace_prefix")
    for definition in _MANIFEST.exposed_definitions
    for counterexample in (
        (False, True) if definition.kind in {
            "proof.principal_variation_trick_winner",
            "proof.principal_variation_trick_team_rank_count",
        } else (False,)
    )
)
_CONSTRUCTION_CASES = tuple(
    definition
    for definition in _MANIFEST.exposed_definitions
    if definition.cp_sat_support.qualification_source == "construction"
)
_HARD_PREDICATE_GROUPS = tuple(
    (origin, stage, definitions)
    for origin in ("search", "exact_trace_prefix")
    for stage in sorted(
        {
            definition.stage
            for definition in _MANIFEST.exposed_definitions
            if "hard" in definition.allowed_hardness
        }
    )
    if (
        definitions := tuple(
            definition
            for definition in _MANIFEST.exposed_definitions
            if definition.stage == stage
            and "hard" in definition.allowed_hardness
        )
    )
)


def _leaf(
    predicate_id: str,
    kind: str,
    arguments: dict,
    hardness: str,
) -> dict:
    return {
        "predicateId": predicate_id,
        "kind": kind,
        "version": 1,
        "hardness": hardness,
        "arguments": copy.deepcopy(arguments),
    }


def _scenario(
    *,
    origin: dict,
    tested_kind: str,
    tested_arguments: dict,
    tested_hardness: str = "hard",
) -> ResearchScenarioV1:
    tested = _leaf("PRED-MATRIX", tested_kind, tested_arguments, tested_hardness)
    report = _leaf(
        "PRED-MATRIX-REPORT",
        "outcome.team_round",
        {"team": "north_south", "status": "won"},
        "report",
    )
    return ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": origin,
            "constraints": tested,
            "decisionPoint": {"kind": "play", "selector": {"trickIndex": 6}},
            "outcome": report,
        }
    )


def _scenario_with_constraint(
    *,
    origin: dict,
    constraint: dict,
) -> ResearchScenarioV1:
    return ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": origin,
            "constraints": constraint,
            "decisionPoint": {"kind": "play", "selector": {"trickIndex": 6}},
            "outcome": _leaf(
                "PRED-MATRIX-REPORT",
                "outcome.team_round",
                {"team": "north_south", "status": "won"},
                "report",
            ),
        }
    )


def _proof_request(candidate: OrToolsNeutralCandidateV1) -> ScenarioProofRequestV1:
    decision = candidate.trace["play"]["decisionState"]
    first = candidate.trace["play"]["plays"][decision["prefixLength"]]
    selected = LegalPlayActionV1(
        Card.parse(first["card"]), bool(first.get("ekka", False))
    ).code
    return ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "exact",
            "visibility": "complete_state",
            "claim": "selected_action_is_unique_winner",
            "targetTeam": Seat.parse(first["actor"]).team.name.lower(),
            "utility": "round_win",
            "successThreshold": 1,
            "lineUniverse": "all_legal",
            "selectedAction": selected,
            "budgetProfile": "test_exact",
        }
    )


def _proof_graph(candidate: OrToolsNeutralCandidateV1) -> ProofGraphV1:
    # Synthetic utilities exercise stage binding, not whole-round strategy.
    trump = candidate.trace["auction"]["completion"]["trumpSuit"]
    decision = candidate.trace["play"]["decisionState"]
    start = int(decision["prefixLength"])
    rows = candidate.trace["play"]["plays"][start : start + 4]
    trick_index = start // 4
    completed = candidate.trace["play"]["tricks"][trick_index]
    owner_by_card = [0] * 32
    for seat, cards in candidate.trace["play"]["initialHands"].items():
        for card in cards:
            owner_by_card[Card.parse(card).id] = int(Seat.parse(seat))
    base_tricks = [0, 0]
    for trick in candidate.trace["play"]["tricks"][:trick_index]:
        base_tricks[int(Seat.parse(trick["winner"]).team)] += 1
    actions = tuple(
        LegalPlayActionV1(
            Card.parse(row["card"]), bool(row.get("ekka", False))
        ).code
        for row in rows
    )
    alternative = next(code for code in decision["legalActions"] if code != actions[0])
    nodes = [
        ProofGraphNodeV1.decision(
            "root",
            Seat.parse(rows[0]["actor"]).team.name.lower(),
            (
                ProofGraphEdgeV1.build(
                    LegalPlayActionV1.from_code(actions[0]).action_id,
                    actions[0], "pv-1", play_trick_index=trick_index,
                ),
                ProofGraphEdgeV1.build(
                    LegalPlayActionV1.from_code(alternative).action_id,
                    alternative, "lose", play_trick_index=trick_index,
                ),
            ),
            actor_seat=rows[0]["actor"],
        )
    ]
    for index, (row, action) in enumerate(zip(rows[1:], actions[1:], strict=True), 1):
        nodes.append(
            ProofGraphNodeV1.decision(
                f"pv-{index}",
                Seat.parse(row["actor"]).team.name.lower(),
                (ProofGraphEdgeV1.build(
                    LegalPlayActionV1.from_code(action).action_id,
                    action,
                    "win" if index == 3 else f"pv-{index + 1}",
                    play_trick_index=trick_index,
                    **({
                        "completed_trick_index": trick_index,
                        "completed_trick_winner_seat": completed["winner"],
                        "completed_trick_winning_card_id": Card.parse(completed["winningCard"]).id,
                    } if index == 3 else {}),
                ),),
                actor_seat=row["actor"],
            )
        )
    nodes.extend((ProofGraphNodeV1.leaf("win", 1), ProofGraphNodeV1.leaf("lose", -1)))
    return ProofGraphV1.build(
        scope="exact",
        root_node_id="root",
        nodes=tuple(nodes),
        tactic_context={
            "baseTricks": base_tricks,
            "trumpSuit": (
                int(Suit.parse(trump)) if trump is not None else None
            ),
            "ownerByCard": owner_by_card,
        },
    )


@pytest.fixture(scope="module")
def matrix_candidate_and_proof(rules):
    with deterministic_work_clock_v1():
        return _matrix_candidate_and_proof(rules)


def _matrix_candidate_and_proof(rules):
    seed_scenario = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": {"op": "all", "children": [
                _leaf("PRED-SEED-CONTRACT", "auction.contract", {"contract": "sun"}, "hard"),
                _leaf("PRED-SEED-BUYER", "auction.buyer", {"seat": "north"}, "hard"),
                *(
                    _leaf(
                        f"PRED-SEED-HAND-{seat.upper()}", "deal.suit_count",
                        {"seat": seat, "suit": suit, "minimum": 8, "maximum": 8,
                         "handStage": "post_purchase"},
                        "hard",
                    )
                    for seat, suit in (
                        ("north", "spades"), ("west", "hearts"),
                        ("south", "diamonds"), ("east", "clubs"),
                    )
                ),
            ]},
            "decisionPoint": {"kind": "play", "selector": {"trickIndex": 6}},
            "outcome": _leaf(
                "PRED-SEED-OUTCOME",
                "outcome.team_round",
                {"team": "north_south", "status": "won"},
                "report",
            ),
        }
    )
    stage_outputs = {}
    original_deal_solve = OrToolsScenarioModelV1.solve
    original_escalation_solve = OrToolsEscalationStageV1.solve
    original_project_solve = OrToolsProjectLifecycleModelV1.solve
    original_play_solve = OrToolsPlayTailStageV1.solve

    def capture(stage, original):
        def wrapped(instance, *args, **kwargs):
            result = original(instance, *args, **kwargs)
            stage_outputs[stage] = result
            return result

        return wrapped

    with (
        patch.object(
            OrToolsScenarioModelV1,
            "solve",
            capture("deal", original_deal_solve),
        ),
        patch.object(
            OrToolsEscalationStageV1,
            "solve",
            capture("escalation", original_escalation_solve),
        ),
        patch.object(
            OrToolsProjectLifecycleModelV1,
            "solve",
            capture("projects", original_project_solve),
        ),
        patch.object(
            OrToolsPlayTailStageV1,
            "solve",
            capture("playPrefix", original_play_solve),
        ),
    ):
        attempt = GenericCompleteTraceCompilerV1(rules=rules, manifest=_MANIFEST)(
            seed_scenario,
            (),
            seed=0,
            maximum_seconds=15,
        )
    assert attempt.status == "feasible", {"reason": attempt.reason, "timings": attempt.stage_timings}
    assert attempt.candidate is not None
    candidate = attempt.candidate
    purchase_suit = Suit(candidate.trace["deckOrder"][20] // 8).name.lower()
    matching = ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": _leaf(
                "PRED-BASELINE-DEAL",
                "deal.purchase_suit",
                {"suit": purchase_suit},
                "hard",
            ),
            "decisionPoint": {"kind": "play", "selector": {"trickIndex": 6}},
            "outcome": _leaf(
                "PRED-BASELINE-OUTCOME",
                "outcome.team_round",
                {
                    "team": candidate.trace["score"]["roundWinner"],
                    "status": "won",
                },
                "hard",
            ),
        }
    )
    executor = OrToolsUnifiedResearchExecutorV1(
        rules,
        bundle_revision_id="CBR-TEST-ORTOOLS-V4",
        bundle_hash="a" * 64,
    )
    proof_request = _proof_request(candidate)
    proof_graph = _proof_graph(candidate)
    result = executor.execute(
        matching,
        proof_request,
        candidate_compiler=lambda *_args, **_kwargs: OrToolsCandidateAttemptV1(
            "feasible", candidate
        ),
        graph_compiler=lambda *_args, **_kwargs: proof_graph,
        maximum_seconds=15,
    )
    assert result.status == "qualified"
    assert result.candidate is not None
    proof = OrToolsScenarioGraphProofV1().prove(
        proof_graph,
        proof_request,
        maximum_seconds=5,
    )
    assert result.candidate.stage_receipts["proof"].evidence_hash == proof.sha256
    assert set(stage_outputs) == {"deal", "escalation", "projects", "playPrefix"}
    return result.candidate, proof, stage_outputs


def test_native_proof_binding_reconstructs_real_pv_without_python_conversion(
    rules, matrix_candidate_and_proof, monkeypatch,
):
    candidate, proof, _stage_outputs = matrix_candidate_and_proof
    legacy = _proof_stage_bindings_v1(candidate, proof, rules)["proof"].context
    assert legacy.principal_variation_tricks

    def forbidden(*_args, **_kwargs):
        raise AssertionError("native PV reconstruction must not use Python raw conversion")

    monkeypatch.setattr(ScoringDomainIRV1, "convert_raw", forbidden)
    native = _proof_stage_bindings_v1(
        candidate, proof, rules, require_native=True,
    )["proof"].context
    assert native.principal_variation_tricks == legacy.principal_variation_tricks
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 5
    assert solver.solve(native.model) == cp_model.OPTIMAL


def test_reconstructed_proof_binds_absent_individual_target(rules, matrix_candidate_and_proof):
    from dataclasses import replace

    candidate, _proof, _stage_outputs = matrix_candidate_and_proof
    raw = _proof_request(candidate).to_dict()
    raw.pop('sha256')
    raw['targetTeam'] = 'west_east' if raw['targetTeam'] == 'north_south' else 'north_south'
    proof = OrToolsScenarioGraphProofV1().prove(_proof_graph(candidate), ScenarioProofRequestV1.build(raw))
    assert proof.completion_status == 'exact_complete'
    candidate = replace(candidate, stage_receipts={**candidate.stage_receipts,
        'proof': _stage_receipt('proof', model_hash=proof.model_hash, evidence_hash=proof.sha256,
            semantic=proof.to_dict(), solver_status=proof.solver_status)})
    legacy = _proof_stage_bindings_v1(candidate, proof, rules)['proof'].context
    assert legacy.target_seat_variable is None
    context = _proof_stage_bindings_v1(candidate, proof, rules, require_native=True)['proof'].context
    assert context.target_seat_variable is not None
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 5
    assert solver.solve(context.model) == cp_model.OPTIMAL
    assert not solver.boolean_value(context.root_actor_is_target_variable)
    assert solver.value(context.target_seat_variable) == -1


def _candidate_for_origin(
    candidate: OrToolsNeutralCandidateV1,
    origin_kind: str,
    scenario: ResearchScenarioV1,
) -> tuple[dict, OrToolsNeutralCandidateV1]:
    if origin_kind == "search":
        return {"kind": "search"}, candidate
    prefix = _search_decision_prefix(candidate, scenario)
    return (
        {"kind": "exact_trace_prefix", "exactTracePrefix": prefix.to_dict()},
        OrToolsNeutralCandidateV1(
            "exact_trace_prefix",
            {**candidate.trace, "exactTracePrefix": prefix.to_dict()},
            candidate.state,
            candidate.no_good_vector,
            candidate.stage_receipts,
            candidate.exact_state_id,
        ),
    )


def _arguments_for_definition(definition, candidate) -> dict:
    arguments = copy.deepcopy(dict(definition.examples[0]["arguments"]))
    if definition.kind in {
        "proof.principal_variation_trick_winner",
        "proof.principal_variation_trick_team_rank_count",
    }:
        arguments["trickIndex"] = int(
            candidate.trace["play"]["decisionState"]["prefixLength"]
        ) // 4
        trick = candidate.trace["play"]["tricks"][arguments["trickIndex"]]
        if definition.kind == "proof.principal_variation_trick_winner":
            arguments["team"] = Seat.parse(trick["winner"]).team.name.lower()
            arguments["rank"] = [Card.parse(trick["winningCard"]).rank.name.lower()]
        else:
            start = arguments["trickIndex"] * 4
            plays = candidate.trace["play"]["plays"][start : start + 4]
            matching_play = next(
                row for row in plays
                if Seat.parse(row["actor"]).team.name.lower() == arguments["team"]
            )
            arguments["rank"] = [Card.parse(matching_play["card"]).rank.name.lower()]
            count = sum(
                Seat.parse(row["actor"]).team.name.lower() == arguments["team"]
                and Card.parse(row["card"]).rank.name.lower() in arguments["rank"]
                for row in plays
            )
            arguments["minimum"] = arguments["maximum"] = count
    return arguments


@pytest.mark.parametrize(
    ("origin_kind", "definition", "counterexample"),
    _CASES,
    ids=lambda value: (
        value if isinstance(value, str) else
        "counterexample" if value is True else "matching" if value is False else value.kind
    ),
)
@pytest.mark.parametrize("native_authority", [False, True], ids=["legacy", "native-v3"])
def test_every_exposed_hard_predicate_reaches_its_real_executor_stage(
    rules,
    matrix_candidate_and_proof,
    origin_kind,
    definition,
    counterexample,
    native_authority,
):
    candidate, proof, _stage_outputs = matrix_candidate_and_proof
    arguments = _arguments_for_definition(definition, candidate)
    if counterexample:
        if definition.kind == "proof.principal_variation_trick_winner":
            arguments["team"] = (
                "west_east" if arguments["team"] == "north_south" else "north_south"
            )
        else:
            arguments["minimum"] = arguments["maximum"] = (arguments["minimum"] + 1) % 3
    scenario = _scenario(
        origin={"kind": "search"},
        tested_kind=definition.kind,
        tested_arguments=arguments,
    )
    origin, candidate = _candidate_for_origin(candidate, origin_kind, scenario)
    if origin_kind == "exact_trace_prefix":
        scenario = _scenario(
            origin=origin,
            tested_kind=definition.kind,
            tested_arguments=arguments,
        )
    if native_authority and definition.stage in {"proof", "tactics"}:
        graph = _proof_graph(candidate)
        request = _proof_request(candidate)
        proof = OrToolsScenarioGraphProofV1().prove(
            graph,
            request,
            maximum_seconds=5,
            native_line_count=(definition.kind == "proof.winning_continuation_line_count"),
            tactic_configure=_native_proof_tactic_configurer_v3(
                scenario=scenario,
                candidate=candidate,
                graph=graph,
                request=request,
                rules=rules,
                manifest=shared_scenario_predicate_catalog_lowering_manifest_v3(),
            ),
        )
        candidate = OrToolsNeutralCandidateV1(
            candidate.origin,
            candidate.trace,
            candidate.state,
            candidate.no_good_vector,
            {
                **candidate.stage_receipts,
                "proof": _stage_receipt(
                    "proof",
                    model_hash=proof.model_hash,
                    evidence_hash=proof.sha256,
                    semantic=proof.to_dict(),
                    solver_status=proof.solver_status,
                ),
                "tactics": _stage_receipt(
                    "tactics", model_hash=proof.model_hash, evidence_hash=proof.sha256,
                    semantic=proof.to_dict(), solver_status=proof.solver_status,
                ),
            },
            candidate.exact_state_id,
        )
    compiler = OrToolsResearchCompilerV1(
        shared_scenario_predicate_catalog_lowering_manifest_v3()
        if native_authority else _MANIFEST,
        native_cp_sat_authority=native_authority,
    )
    preflight = compiler.preflight(scenario)
    diagnostics = {
        "origin": origin_kind,
        "kind": definition.kind,
        "stage": definition.stage,
        "compiledStages": preflight.compiled_stages,
        "candidateStages": tuple(sorted(candidate.stage_receipts)),
        "unsupported": preflight.unsupported,
    }
    assert preflight.supported, diagnostics
    bindings = DefaultStageContextCompilerV1(
        rules,
        required_stages=preflight.compiled_stages,
        native_cp_sat_authority=native_authority,
    )(candidate, proof)
    diagnostics["boundStages"] = tuple(sorted(bindings))
    assert definition.stage in bindings, diagnostics
    try:
        evaluation = compiler.evaluate(
            scenario,
            candidate,
            bindings,
            maximum_seconds=2,
        )
    except Exception as exc:
        diagnostics["exception"] = f"{type(exc).__name__}: {exc}"
        pytest.fail(str(diagnostics), pytrace=False)
    diagnostics.update(
        {
            "evaluationStatus": evaluation.status,
            "namedCore": evaluation.named_core,
            "evidenceKinds": tuple(
                sorted(str(item["kind"]) for item in evaluation.evidence)
            ),
        }
    )
    assert evaluation.status != "incomplete", diagnostics
    if definition.kind in {
        "proof.principal_variation_trick_winner",
        "proof.principal_variation_trick_team_rank_count",
    }:
        assert evaluation.status == ("target_miss" if counterexample else "satisfied"), diagnostics
    if evaluation.status == "satisfied" or not native_authority:
        assert definition.kind in diagnostics["evidenceKinds"], diagnostics
    if native_authority and definition.stage == "tactics":
        # Removing same-solve evidence must not activate the legacy evaluator.
        saved = dict(proof.native_qualification)
        try:
            proof.native_qualification.clear()
            missing = compiler.evaluate(scenario, candidate, bindings, maximum_seconds=2)
            assert missing.status == "incomplete"
            assert not missing.satisfied
            assert missing.evidence == ()
        finally:
            proof.native_qualification.update(saved)


@pytest.mark.parametrize(
    ("origin_kind", "stage", "definitions"),
    _HARD_PREDICATE_GROUPS,
    ids=lambda value: value if isinstance(value, str) else "predicate-group",
)
@pytest.mark.parametrize("native_authority", [False, True], ids=["legacy", "native-v3"])
def test_every_hard_predicate_evidence_matches_its_native_literal(
    rules,
    matrix_candidate_and_proof,
    origin_kind,
    stage,
    definitions,
    native_authority,
):
    candidate, proof, _stage_outputs = matrix_candidate_and_proof
    leaves = []
    for index, definition in enumerate(definitions):
        support_status = (
            definition.cp_sat_support.search_status
            if origin_kind == "search"
            else definition.cp_sat_support.exact_input_status
        )
        assert support_status in {"exact", "derived"}, {
            "origin": origin_kind,
            "stage": stage,
            "kind": definition.kind,
            "supportStatus": support_status,
        }
        leaves.append(
            _leaf(
                f"PRED-MATRIX-COVERAGE-{index}",
                definition.kind,
                _arguments_for_definition(definition, candidate),
                "hard",
            )
        )
    constraint = {"op": "all", "children": leaves}
    seed_scenario = _scenario_with_constraint(
        origin={"kind": "search"},
        constraint=constraint,
    )
    origin, candidate = _candidate_for_origin(
        candidate,
        origin_kind,
        seed_scenario,
    )
    scenario = _scenario_with_constraint(origin=origin, constraint=constraint)
    compiler = OrToolsResearchCompilerV1(_MANIFEST, native_cp_sat_authority=native_authority)
    preflight = compiler.preflight(scenario)
    assert preflight.supported, {
        "origin": origin_kind,
        "stage": stage,
        "unsupported": preflight.unsupported,
    }
    bindings = DefaultStageContextCompilerV1(
        rules,
        required_stages=preflight.compiled_stages,
        native_cp_sat_authority=native_authority,
    )(candidate, proof)
    context = bindings[stage].context
    predicates = scenario.constraints.predicates()
    lowerings = tuple(
        _MANIFEST.lower_predicate(predicate, context) for predicate in predicates
    )
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 2
    assert solver.solve(context.model) == cp_model.OPTIMAL
    for predicate, lowering in zip(predicates, lowerings, strict=True):
        actual = solver.boolean_value(lowering.literal)
        evidence = lowering.extract(solver)
        assert bool(evidence["satisfied"]) is actual, {
            "origin": origin_kind,
            "stage": stage,
            "kind": predicate.kind,
            "actual": actual,
            "evidence": evidence,
        }
        # Literal negation is a consistency check, not a separate semantic
        # counterexample fixture or permission to expose an unsafe complement.
        assert solver.boolean_value(lowering.literal.Not()) is not actual


@pytest.fixture(scope="module")
def fixed_native_state_candidate(rules, root):
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1

    raw = json.loads((root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json")
                     .read_text(encoding="utf-8"))["exactTracePrefix"]
    raw.pop("stateAssertions", None)
    raw["playActions"].extend(
        {"actor": actor, "card": card} for actor, card in (
            ("west", "CA"), ("east", "C8"), ("north", "C7"),
            ("west", "D7"), ("south", "S7"), ("east", "C9"),
            ("north", "DA"), ("west", "D8"), ("south", "SA"),
        )
    )
    prefix = ExactTracePrefixV1.build(raw)
    normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
        prefix, decision_point={"kind": "play"}, maximum_seconds=60,
    )
    candidate = OrToolsUnifiedResearchExecutorV1._candidate_from_normalized_exact(
        SimpleNamespace(rules=rules), normalized, prefix,
    )
    assert sum(candidate.trace["score"]["projectPoints"]) > 0
    return candidate


@pytest.mark.parametrize("starting_match", [(41, 53), (0, 0)])
@pytest.mark.parametrize("prefix_length", [1, 24, 32])
@pytest.mark.parametrize("origin_kind", ["search", "exact_trace_prefix"])
def test_native_play_state_binds_match_and_awarded_projects(
    rules, fixed_native_state_candidate, starting_match, prefix_length, origin_kind,
    monkeypatch,
):
    original = fixed_native_state_candidate
    trace = copy.deepcopy(dict(original.trace))
    trace["matchScore"] = [7, 9]  # Exact input, not this duplicate, owns the score.
    trace["exactTracePrefix"]["matchScore"] = list(starting_match)
    trace["exactTracePrefix"]["playActions"] = trace["exactTracePrefix"]["playActions"][:prefix_length]
    trace["exactTracePrefix"].pop("sha256")
    trace["exactTracePrefix"] = ExactTracePrefixV1.build(trace["exactTracePrefix"]).to_dict()
    if origin_kind == "search":
        trace.pop("exactTracePrefix")
        trace["matchScore"] = list(starting_match)
    decision = trace["play"]["decisionState"]
    decision["prefixLength"] = prefix_length
    decision["nextActor"] = (
        trace["play"]["plays"][prefix_length]["actor"]
        if prefix_length < 32 else None
    )
    candidate = OrToolsNeutralCandidateV1(
        origin_kind, trace, {**original.state, "currentPlayer": decision["nextActor"]}, original.no_good_vector,
        original.stage_receipts, original.exact_state_id,
    )
    current_raw = [0, 0]
    for trick in trace["play"]["tricks"][:prefix_length // 4]:
        current_raw[int(Seat.parse(trick["winner"]).team)] += trick["rawPoints"]
    scoring = ScoringDomainIRV1.from_rules(rules.data)
    expected_game = (
        trace["score"]["cardGamePoints"] if prefix_length == 32 else
        [scoring.convert_raw(raw, Contract.parse(trace["contract"])) for raw in current_raw]
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("native state qualification must not use Python raw conversion")

    monkeypatch.setattr(ScoringDomainIRV1, "convert_raw", forbidden)
    context = _native_play_context_v3(candidate, rules, dealer=Seat.EAST)
    assert context.context_id == f"candidate-native-play-{candidate.sha256[:16]}"
    terminal = prefix_length == 32
    expected_match = (
        tuple(starting_match[index] + trace["score"]["awardedPoints"][index]
              for index in range(2))
        if terminal else starting_match
    )
    compared = any(rules.project_domain.should_compare(index)
                   for index in range(prefix_length // 4))
    expected_projects = trace["score"]["projectPoints"] if compared else (0, 0)
    claims = []
    for team in ("north_south", "west_east"):
        index = 0 if team == "north_south" else 1
        checks = [
            ("state.match_score", expected_match[index]),
            ("state.current_project_points", expected_projects[index]),
            ("state.current_game_points", expected_game[index]),
        ]
        for kind, expected in checks:
            for value, satisfied in ((expected, True), (expected + 1, False),
                                     *(((0, False),) if expected else ())):
                predicate = ResearchScenarioV1.build({
                    "schemaVersion": 1, "origin": {"kind": "search"},
                    "constraints": _leaf("PRED-STATE-SCORE", kind,
                        {"team": team, "minimum": value, "maximum": value}, "hard"),
                    "decisionPoint": {"kind": "play", "selector": {"trickIndex": 6}},
                    "outcome": _leaf("PRED-REPORT", "outcome.team_round",
                        {"team": "north_south", "status": "won"}, "report"),
                }).constraints.predicates()[0]
                claims.append((_MANIFEST.lower_predicate(predicate, context), satisfied))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 5
    assert solver.solve(context.model) == cp_model.OPTIMAL
    assert [solver.boolean_value(claim.literal) for claim, _ in claims] == [
        expected for _, expected in claims
    ]


def test_native_play_state_requires_project_inputs(rules, fixed_native_state_candidate):
    original = fixed_native_state_candidate
    candidate = OrToolsNeutralCandidateV1(
        original.origin, {**original.trace, "projects": None}, original.state,
        original.no_good_vector, original.stage_receipts, original.exact_state_id,
    )
    with patch.object(OrToolsPlayTailStageV1, "build_model",
                      side_effect=AssertionError("invalid inputs must fail before compiling")):
        with pytest.raises(ValueError, match="requires project declarations"):
            _native_play_context_v3(candidate, rules, dealer=Seat.EAST)


def test_native_play_state_rejects_corrupt_prefix_before_compiling(rules, fixed_native_state_candidate):
    original = fixed_native_state_candidate
    trace = copy.deepcopy(dict(original.trace))
    trace["exactTracePrefix"]["sha256"] = "0" * 64
    candidate = OrToolsNeutralCandidateV1(
        original.origin, trace, original.state,
        original.no_good_vector, original.stage_receipts, original.exact_state_id,
    )
    with patch.object(OrToolsPlayTailStageV1, "build_model",
                      side_effect=AssertionError("invalid inputs must fail before compiling")):
        with pytest.raises(ValueError, match="checksum mismatch"):
            _native_play_context_v3(candidate, rules, dealer=Seat.EAST)


@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize("prefix_length", [4, 7, 8])
def test_native_play_visibility_uses_prefix_not_projection(
    rules, fixed_native_state_candidate, prefix_length, complete,
):
    from baloot_platform.core.dealing import DealLayoutV1

    original = fixed_native_state_candidate
    trace = copy.deepcopy(dict(original.trace))
    rows = trace["play"]["plays"]
    decision = trace["play"]["decisionState"]
    decision["prefixLength"] = prefix_length
    decision["nextActor"] = rows[prefix_length]["actor"]
    trace["exactTracePrefix"]["playActions"] = trace["exactTracePrefix"]["playActions"][:prefix_length]
    trace["exactTracePrefix"].pop("sha256")
    trace["exactTracePrefix"] = ExactTracePrefixV1.build(trace["exactTracePrefix"]).to_dict()
    trace["completePlay"] = copy.deepcopy(trace["play"])
    # Deliberately malicious duplicate projection: every opponent card appears public.
    state = {"currentPlayer": decision["nextActor"],
             "hands": trace["play"]["initialHands"],
             "events": [{"type": "card_played", "card": Card(card).code}
                        for card in range(32)]}
    candidate = OrToolsNeutralCandidateV1(
        original.origin, trace, state, original.no_good_vector,
        original.stage_receipts, original.exact_state_id,
    )
    context = _native_play_context_v3(candidate, rules, dealer=Seat.EAST, complete=complete)
    layout = DealLayoutV1.build(rules.structural_domain, dealer=Seat.EAST)
    public = {Card.parse(trace["deckOrder"][layout.purchase_position]).id}
    public.update(Card.parse(row["card"]).id for row in rows[:prefix_length])
    # North reveals the winning 100 only on completing its trick-two action (ordinal 7).
    if prefix_length >= 8:
        public.update(Card.parse(card).id for card in ("DJ", "DQ", "DK", "D10", "DA"))
    expected = {
        (seat, card): card in public or Card(card).code in trace["play"]["initialHands"][seat.name.lower()]
        for seat in Seat for card in range(32)
    }
    claims = []
    for (seat, card), visible in expected.items():
        predicate = ResearchScenarioV1.build({
            "schemaVersion": 1, "origin": {"kind": "search"},
            "constraints": _leaf("PRED-NATIVE-VISIBILITY", "state.observer_card_visibility",
                {"observer": seat.name.lower(), "card": card,
                 "visibility": "visible" if visible else "hidden"}, "hard"),
            "decisionPoint": {"kind": "play"},
            "outcome": _leaf("PRED-REPORT", "outcome.team_round",
                {"team": "north_south", "status": "won"}, "report"),
        }).constraints.predicates()[0]
        claims.append(_MANIFEST.lower_predicate(predicate, context))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 5
    assert solver.solve(context.model) == cp_model.OPTIMAL
    assert all(solver.boolean_value(claim.literal) for claim in claims), [
        (seat.name, Card(card).code, visible)
        for ((seat, card), visible), claim in zip(expected.items(), claims, strict=True)
        if not solver.boolean_value(claim.literal)
    ]
    assert all(bool(claim.extract(solver)["satisfied"]) for claim in claims)


@pytest.mark.parametrize("prefix_length", [24, 32])
@pytest.mark.parametrize("rules_file", ["saudi_common_v1.yaml", "saudi_federation_2023_v8.yaml"])
def test_native_terminal_hokum_double_baloot_matches_legal_reference(
    root, rules_file, prefix_length, monkeypatch,
):
    from baloot_platform.core.auction_ir import AuctionDomainIRV1
    from baloot_platform.core.bidding_machine import BiddingMachine
    from baloot_platform.core.dealing import DealLayoutV1
    from baloot_platform.core.engine import BalootEngine
    from baloot_platform.core.rules import RuleBook
    from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1

    rules = RuleBook.load(root / "configs/rules" / rules_file)
    layout = DealLayoutV1.build(rules.structural_domain, dealer=Seat.EAST)
    deck = [None] * 32
    for position, code in zip(layout.initial_positions[Seat.NORTH],
                              ("SK", "SQ", "S7", "H7", "D7"), strict=True):
        deck[position] = Card.parse(code).id
    deck[layout.purchase_position] = Card.parse("S8").id
    rest = iter(card for card in range(32) if card not in deck)
    deck = [next(rest) if card is None else card for card in deck]
    starting_match = (41, 53)
    # The independent runtime produces a legal fixture, never Research truth.
    engine = BalootEngine(rules)
    state = engine.create_round(seed=0, dealer=Seat.EAST, match_score=starting_match)
    engine.initial_deal_from_preordered(state, deck)
    auction = BiddingMachine(Seat.EAST, state.purchase_card.suit,
                             AuctionDomainIRV1.from_rules(rules.data), state.purchase_card.rank)
    auction.submit("hokum")
    for _ in range(3):
        auction.submit("pass")
    engine.complete_deal_from_bid_result(state, auction.result)
    engine.apply_escalation(state, seat=Seat.WEST, action="double", mode="open")
    engine.prepare_projects(state)
    actions = []
    for _ in range(32):
        selected = min(engine.legal_play_actions(state), key=lambda action: (action.card.id, action.ekka))
        actions.append({"actor": state.current_player.name.lower(),
                        "card": selected.card.code, "ekka": selected.ekka})
        engine.play_card(state, selected.card, ekka=selected.ekka)
    reference = state.to_dict(reveal_hands=True)
    assert reference["roundScore"]["balootPoints"] == [rules.project_domain.baloot_value, 0]
    prefix = ExactTracePrefixV1.build({
        "schemaVersion": 1, "dealer": "east", "deckOrder": deck,
        "matchScore": list(starting_match),
        "auctionActions": [{"actor": "north", "action": "hokum"},
                           *({"actor": actor, "action": "pass"}
                             for actor in ("west", "south", "east"))],
        "escalationActions": [{"actor": "west", "action": "double", "mode": "open"}],
        "projectDeclarations": [project.public_declaration() for project in state.projects
                                if project.declared], "playActions": actions,
    })
    with patch.object(BalootEngine, "__init__",
                      side_effect=AssertionError("native qualification must not invoke the runtime")):
        normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
            prefix, decision_point={"kind": "play"}, maximum_seconds=60,
        )
        candidate = OrToolsUnifiedResearchExecutorV1._candidate_from_normalized_exact(
            SimpleNamespace(rules=rules), normalized, prefix,
        )
    trace = copy.deepcopy(dict(candidate.trace))
    decision = trace["play"]["decisionState"]
    decision["prefixLength"] = prefix_length
    decision["nextActor"] = (
        trace["play"]["plays"][prefix_length]["actor"] if prefix_length < 32 else None
    )
    trace["exactTracePrefix"]["playActions"] = trace["exactTracePrefix"]["playActions"][:prefix_length]
    trace["exactTracePrefix"].pop("sha256")
    trace["exactTracePrefix"] = ExactTracePrefixV1.build(trace["exactTracePrefix"]).to_dict()
    candidate = OrToolsNeutralCandidateV1(
        candidate.origin, trace, {**candidate.state, "currentPlayer": decision["nextActor"]},
        candidate.no_good_vector, candidate.stage_receipts, candidate.exact_state_id,
    )
    current_raw = [0, 0]
    for trick in trace["play"]["tricks"][:prefix_length // 4]:
        current_raw[int(Seat.parse(trick["winner"]).team)] += trick["rawPoints"]
    scoring = ScoringDomainIRV1.from_rules(rules.data)
    expected_game = (
        reference["roundScore"]["cardGamePoints"] if prefix_length == 32 else
        [scoring.convert_raw(raw, Contract.HOKUM) for raw in current_raw]
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("native state qualification must not use Python raw conversion")

    monkeypatch.setattr(ScoringDomainIRV1, "convert_raw", forbidden)
    with patch.object(BalootEngine, "__init__",
                      side_effect=AssertionError("native qualification must not invoke the runtime")):
        context = DefaultStageContextCompilerV1(
            rules, required_stages=("state",), native_cp_sat_authority=True,
        )(candidate, None)["state"].context
    claims = []
    for index, team in enumerate(("north_south", "west_east")):
        for value, satisfied in ((expected_game[index], True), (expected_game[index] + 1, False)):
            predicate = ResearchScenarioV1.build({
                "schemaVersion": 1, "origin": {"kind": "search"},
                "constraints": _leaf("PRED-HOKUM-GAME", "state.current_game_points",
                    {"team": team, "minimum": value, "maximum": value}, "hard"),
                "decisionPoint": {"kind": "play"},
                "outcome": _leaf("PRED-REPORT", "outcome.team_round",
                    {"team": "north_south", "status": "won"}, "report"),
            }).constraints.predicates()[0]
            claims.append((_MANIFEST.lower_predicate(predicate, context), satisfied))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 5
    assert solver.solve(context.model) == cp_model.OPTIMAL
    assert [solver.boolean_value(claim.literal) for claim, _ in claims] == [
        expected for _, expected in claims
    ]
    for team in ("north_south", "west_east"):
        index = 0 if team == "north_south" else 1
        expected_match = reference["matchScore"] if prefix_length == 32 else starting_match
        assert solver.value(context.match_scores[index]) == expected_match[index]
        assert solver.value(context.game_points[index]) == expected_game[index]
        assert solver.value(context.project_points[index]) == reference["roundScore"]["projectPoints"][index]


def test_frozen_candidate_rebinds_auction_and_projects_without_reconstruction(
    rules,
    matrix_candidate_and_proof,
):
    candidate, proof, _stage_outputs = matrix_candidate_and_proof
    with patch.object(
        OrToolsScenarioModelV1,
        "__init__",
        side_effect=AssertionError("qualification rebuilt the construction model"),
    ):
        bindings = DefaultStageContextCompilerV1(
            rules,
            required_stages=("auction", "projects"),
        )(candidate, proof)

    assert set(bindings) == {"auction", "projects"}
    auction_solver = cp_model.CpSolver()
    assert auction_solver.solve(bindings["auction"].context.model) == cp_model.OPTIMAL
    observed_auction = bindings["auction"].context.fragment.extract(auction_solver)
    assert observed_auction.to_dict()["completion"] == candidate.trace["auction"][
        "completion"
    ]

    project_solver = cp_model.CpSolver()
    assert project_solver.solve(bindings["projects"].context.model) == cp_model.OPTIMAL
    observed_projects = bindings["projects"].context.projects.extract(project_solver)
    assert [item.to_dict() for item in observed_projects.projects] == list(
        candidate.trace["projects"]["projects"]
    )


def _stable_stage_value(value):
    if value is None or type(value) in {bool, int, float, str}:
        return value
    if isinstance(value, Enum):
        return f"{type(value).__name__}.{value.name}"
    if isinstance(value, dict):
        return {
            str(key): _stable_stage_value(item)
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_stable_stage_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_stable_stage_value(item) for item in value), key=str)
    if hasattr(value, "to_dict"):
        return _stable_stage_value(value.to_dict())
    if hasattr(value, "sha256"):
        return {"type": type(value).__name__, "sha256": str(value.sha256)}
    if hasattr(value, "__dict__"):
        return {
            "type": type(value).__name__,
            "fields": _stable_stage_value(vars(value)),
        }
    return {"type": type(value).__name__}


def _construction_stage_signatures(rules, scenario, stage_outputs):
    signatures = {}

    def deal_solve(instance, *_args, **_kwargs):
        signature = canonical_sha256({"modelProto": str(instance.model.proto)})
        signatures["deal"] = signature
        signatures["auction"] = signature
        return stage_outputs["deal"]

    def escalation_solve(instance, *args, **kwargs):
        signatures["escalation"] = canonical_sha256(
            _stable_stage_value(
                {
                    "instance": vars(instance),
                    "args": args,
                    "kwargs": kwargs,
                }
            )
        )
        return stage_outputs["escalation"]

    def project_solve(instance, *args, **kwargs):
        signatures["projects"] = canonical_sha256(
            _stable_stage_value(
                {"instance": vars(instance), "args": args, "kwargs": kwargs}
            )
        )
        return stage_outputs["projects"]

    def play_solve(instance, *args, **kwargs):
        signatures["playPrefix"] = canonical_sha256(
            _stable_stage_value(
                {"instance": vars(instance), "args": args, "kwargs": kwargs}
            )
        )
        return stage_outputs["playPrefix"]

    with (
        patch.object(OrToolsScenarioModelV1, "solve", deal_solve),
        patch.object(OrToolsEscalationStageV1, "solve", escalation_solve),
        patch.object(OrToolsProjectLifecycleModelV1, "solve", project_solve),
        patch.object(OrToolsPlayTailStageV1, "solve", play_solve),
    ):
        attempt = GenericCompleteTraceCompilerV1(rules=rules)(
            scenario,
            (),
            seed=0,
            maximum_seconds=15,
        )
    assert attempt.status == "feasible"
    return signatures


@pytest.mark.parametrize(
    "definition",
    _CONSTRUCTION_CASES,
    ids=lambda definition: f"{definition.stage}-{definition.kind}",
)
def test_construction_qualified_predicate_changes_its_claimed_stage_model(
    rules,
    matrix_candidate_and_proof,
    definition,
):
    _candidate, _proof, stage_outputs = matrix_candidate_and_proof
    arguments = copy.deepcopy(dict(definition.examples[0]["arguments"]))
    hard = _scenario(
        origin={"kind": "search"},
        tested_kind=definition.kind,
        tested_arguments=arguments,
    )
    report = _scenario(
        origin={"kind": "search"},
        tested_kind=definition.kind,
        tested_arguments=arguments,
        tested_hardness="report",
    )
    hard_signatures = _construction_stage_signatures(rules, hard, stage_outputs)
    report_signatures = _construction_stage_signatures(rules, report, stage_outputs)
    diagnostics = {
        "kind": definition.kind,
        "claimedStage": definition.stage,
        "qualificationSource": definition.cp_sat_support.qualification_source,
        "hardStageSignature": hard_signatures.get(definition.stage),
        "reportStageSignature": report_signatures.get(definition.stage),
        "capturedStages": tuple(sorted(hard_signatures)),
    }
    assert definition.stage in hard_signatures, diagnostics
    assert hard_signatures[definition.stage] != report_signatures[definition.stage], (
        diagnostics
    )
