from __future__ import annotations

from dataclasses import replace
import inspect
import json
from types import SimpleNamespace
import time
import pytest
from ortools.sat.python import cp_model

import baloot_platform.research.ortools_tail_first_candidate as tail_candidate_module
import baloot_platform.research.ortools_tail_proof_boundary as tail_boundary_module
from baloot_platform.core.cards import Card
from baloot_platform.core.enums import Contract, Seat, Team
from baloot_platform.core.rules import RuleBook
from baloot_platform.research.reference_play import ReferencePlayInterpreterV1, ReferencePlayV1
from baloot_platform.research.ortools_play_tail import (
    OrToolsPlayTailStageV1,
    PlayTailSolveStopped,
)
from baloot_platform.research.ortools_research_executor import (
    OrToolsCandidateAttemptV1,
    OrToolsNeutralCandidateV1,
    OrToolsResearchStageReceiptV1,
    OrToolsUnifiedResearchExecutorV1,
    _native_proof_tactic_configurer_v3,
)
from baloot_platform.research.ortools_proof import (
    OrToolsScenarioGraphProofV1, ProofGraphEdgeV1, ProofGraphNodeV1, ProofGraphV1,
)
from baloot_platform.research.scenario_predicate_manifest_v2 import (
    shared_scenario_predicate_catalog_lowering_manifest_v2,
)
from baloot_platform.research.ortools_flagship_research import (
    OrToolsFlagshipResearchV1,
)
from baloot_platform.research.ortools_tail_first_candidate import (
    OrToolsTailFirstCandidateCompilerV1,
)
from baloot_platform.research.ortools_tail_first_lift import (
    OrToolsTailFirstBackwardLiftV1,
)
from baloot_platform.research.ortools_tail_proof_boundary import (
    OrToolsTailProofBoundaryV1,
)
from baloot_platform.research.scenario_contracts import (
    ResearchScenarioV1,
    ScenarioProofRequestV1,
)
from baloot_platform.research.tail_first_stage_planner import (
    TailFirstStagePlannerV1,
)


def _leaf(predicate_id: str, kind: str, arguments: dict) -> dict:
    return {
        "predicateId": predicate_id,
        "kind": kind,
        "version": 1,
        "hardness": "hard",
        "arguments": arguments,
    }


def test_tail_proof_tactics_fail_closed_without_shared_native_model() -> None:
    manifest = SimpleNamespace(
        definition=lambda _kind: SimpleNamespace(stage="proof")
    )
    for expected in (True, False):
        scenario = ResearchScenarioV1.build(
            {
                "schemaVersion": 1,
                "origin": {"kind": "search"},
                "constraints": _leaf(
                    "PRED-TAIL-TACTIC",
                    "tactic.applicable",
                    {"kind": "only_winning_move", "applicable": expected},
                ),
                "decisionPoint": {
                    "kind": "play",
                    "selector": {"trickIndex": 5},
                },
                "outcome": _leaf(
                    "PRED-TAIL-TACTIC-OUTCOME",
                    "tactic.applicable",
                    {"kind": "only_winning_move", "applicable": expected},
                ),
            }
        )
        status, evidence = tail_candidate_module._proof_stage_projection_v1(
            scenario,
            None,
            manifest,
            native_cp_sat=True, maximum_seconds=1, maximum_conflicts=1000,
        )
        assert status == "incomplete"
        assert evidence["reason"] == "native_tail_proof_bindings_unavailable"
        for satisfied, expected_status in ((False, "false"), (True, "satisfied")):
            status, evidence = tail_candidate_module._proof_stage_projection_v1(
                scenario,
                SimpleNamespace(native_qualification={
                    predicate.predicate_id: {"satisfied": satisfied}
                    for predicate in scenario.predicates()
                }),
                manifest,
                native_cp_sat=True, maximum_seconds=1, maximum_conflicts=1000,
            )
            assert status == expected_status
            assert evidence["predicateIds"]


@pytest.mark.real_clock
@pytest.mark.parametrize("operator,bounds,projection,expected", [
    ("all", None, None, "false"),
    ("any", None, None, "satisfied"),
    ("not_false", None, None, "satisfied"),
    ("not_true", None, None, "false"),
    ("count", (1, 1), None, "satisfied"),
    ("count", (2, 2), None, "false"),
    ("mixed_any", None, None, "satisfied"),
    ("soft_count", (1, 1), None, "satisfied"),
    ("any", None, "legacy", "false"),
    ("any", None, "cancelled", "incomplete"),
    ("any", None, "expired", "incomplete"),
    ("any", None, cp_model.UNKNOWN, "incomplete"),
    ("any", None, cp_model.MODEL_INVALID, "incomplete"),
    ("any", None, "missing_index", "incomplete"),
    ("any", None, "invalid_bindings", "incomplete"),
    ("any", None, "wrong_name", "incomplete"),
    ("any", None, "missing_proto", "incomplete"),
])
def test_tail_native_proof_projection_preserves_boolean_operators(
    rules, monkeypatch, operator, bounds, projection, expected,
):
    false = _leaf("PRED-ZERO", "proof.winning_root_action_count", {"minimum": 0, "maximum": 0})
    true = _leaf("PRED-ONE", "proof.winning_root_action_count", {"minimum": 1, "maximum": 1})
    children = [false, true]
    if operator.startswith("not_"):
        children = [false if operator == "not_false" else true]
    elif operator == "mixed_any":
        children = [false, _leaf("PRED-DEFERRED", "state.phase", {"state": "playing"})]
    elif operator == "soft_count":
        children.append({**_leaf("PRED-SOFT", "proof.winning_root_action_count",
                                 {"minimum": 1, "maximum": 1}), "hardness": "report"})
    expression = {"op": "not" if operator.startswith("not_") else
                  "any" if operator == "mixed_any" else "count" if operator == "soft_count" else operator,
                  "children": children}
    if bounds is not None:
        expression.update(minimum=bounds[0], maximum=bounds[1])
    scenario = ResearchScenarioV1.build({
        "schemaVersion": 1, "origin": {"kind": "search"}, "constraints": expression,
        "decisionPoint": {"kind": "play", "selector": {
            "trickIndex": 6, "position": 0, "remainingCards": 2, "actingSeat": "north"}},
        "outcome": {**_leaf("PRED-REPORT", "state.phase", {"state": "playing"}), "hardness": "report"},
    })
    graph = ProofGraphV1.build(scope="exact", root_node_id="root", nodes=(
        ProofGraphNodeV1.decision("root", "north_south", (
            ProofGraphEdgeV1.build(0, "S7", "win"),
            ProofGraphEdgeV1.build(1, "S8", "lose")), actor_seat="north"),
        ProofGraphNodeV1.leaf("win", 1), ProofGraphNodeV1.leaf("lose", -1),
    ))
    request = ScenarioProofRequestV1.build({
        "schemaVersion": 1, "mode": "exact", "visibility": "complete_state",
        "claim": "winning_root_action_count", "targetTeam": "north_south",
        "utility": "round_win", "successThreshold": 1, "budgetProfile": "test_exact",
    })
    manifest = shared_scenario_predicate_catalog_lowering_manifest_v2()
    configure = _native_proof_tactic_configurer_v3(scenario=scenario,
        candidate=SimpleNamespace(trace={}, state={}), graph=graph, request=request,
        rules=rules, manifest=manifest)
    proof = OrToolsScenarioGraphProofV1().prove(graph, request, tactic_configure=configure,
                                              native_root_only=True)
    if projection in {"missing_index", "invalid_bindings", "wrong_name", "missing_proto"}:
        document = proof.model_document()
        if projection == "missing_index":
            document["nativeQualificationLiteralIndices"].pop("PRED-ZERO")
        elif projection == "invalid_bindings":
            document["nativeQualificationLiteralIndices"] = None
        elif projection == "wrong_name":
            document["nativeQualificationVariables"]["PRED-ZERO"] = "unowned_literal"
        else:
            document.pop("modelProto")
        proof = replace(proof, _model_document_json=json.dumps(document),
                        model_hash=tail_candidate_module.canonical_sha256(document))
    if projection in {cp_model.UNKNOWN, cp_model.MODEL_INVALID}:
        monkeypatch.setattr(tail_candidate_module, "_solve_cp_sat_with_cancel_v1",
                            lambda *_args: projection)
    elif projection in {"legacy", "cancelled", "expired"}:
        def forbidden_solve(*_args):
            pytest.fail("legacy or cancelled projection must not launch a solve")
        monkeypatch.setattr(tail_candidate_module, "_solve_cp_sat_with_cancel_v1", forbidden_solve)
    before = proof.to_dict()
    status, _ = tail_candidate_module._proof_stage_projection_v1(scenario, proof, manifest,
        native_cp_sat=projection != "legacy", maximum_seconds=0 if projection == "expired" else 1,
        maximum_conflicts=1000,
        cancel_check=(lambda: True) if projection == "cancelled" else None)
    assert status == expected
    assert proof.to_dict() == before
    assert proof.completion_status == "exact_complete"
    assert proof.winning_root_action_count == 1


def test_tail_proof_projection_cannot_reinsert_semantic_truth_constants() -> None:
    source = open(tail_candidate_module.__file__, encoding="utf-8").read()
    start = source.index("def _proof_stage_projection_v1(")
    end = source.index("\ndef _seed_with_target_raw_v1(", start)
    projection = source[start:end]

    assert "_ProofQualificationContextV1(" not in projection
    assert "root_value_" not in projection
    assert "root_actor_is_target_variable" not in projection
    assert "principal_variation_has_continuation_variable" not in projection
    assert "new_constant" not in projection


def _flagship_scenario(
    *, target_team_wins_trick_six: bool = False
) -> ResearchScenarioV1:
    low = {"ranks": ["seven", "eight", "nine"]}
    return ResearchScenarioV1.build(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": {
                "op": "all",
                "children": [
                    _leaf("PRED-CONTRACT-SUN", "auction.contract", {"contract": "sun"}),
                    _leaf("PRED-BUYER-NORTH", "auction.buyer", {"seat": "north"}),
                    _leaf(
                        "PRED-AWARDED-HUNDRED",
                        "project.exists",
                        {
                            "team": "north_south",
                            "owner": "north",
                            "kind": "hundred_sequence",
                            "value": 20,
                            "state": "awarded",
                        },
                    ),
                    _leaf(
                        "PRED-ROOT-ACE",
                        "play.action_matches",
                        {
                            "trickIndex": 5,
                            "position": 3,
                            "actingSeat": "west",
                            "selector": {
                                "ranks": ["ace"],
                                "owner": {"relation": "actor"},
                            },
                        },
                    ),
                    _leaf(
                        (
                            "PRED-TARGET-WINS-SIX-CHANGED"
                            if target_team_wins_trick_six
                            else "PRED-OPPONENT-WINS-SIX"
                        ),
                        "play.trick_winner_matches",
                        {
                            "trickIndex": 5,
                            "team": (
                                "west_east"
                                if target_team_wins_trick_six
                                else "north_south"
                            ),
                            "selector": low,
                        },
                    ),
                    _leaf(
                        "PRED-TEAM-WINS-SEVEN",
                        "play.trick_winner_matches",
                        {"trickIndex": 6, "team": "west_east", "selector": low},
                    ),
                    _leaf(
                        "PRED-TEAM-WINS-EIGHT",
                        "play.trick_winner_matches",
                        {"trickIndex": 7, "team": "west_east", "selector": low},
                    ),
                    _leaf(
                        "PRED-TWO-OPPONENT-ACES-EIGHT",
                        "play.trick_card_count",
                        {
                            "trickIndex": 7,
                            "selector": {
                                "ranks": ["ace"],
                                "owner": {
                                    "relation": "team",
                                    "team": "north_south",
                                },
                            },
                            "minimum": 2,
                            "maximum": 2,
                        },
                    ),
                    _leaf(
                        "PRED-WINNING-ROOT-IS-ACE",
                        "proof.winning_root_action_matches",
                        {
                            "selector": {
                                "ranks": ["ace"],
                                "owner": {"relation": "actor"},
                            }
                        },
                    ),
                    _leaf(
                        "PRED-ONE-WINNING-ROOT",
                        "proof.winning_root_action_count",
                        {"minimum": 1, "maximum": 1},
                    ),
                ],
            },
            "decisionPoint": {
                "kind": "play",
                "selector": {
                    "trickIndex": 5,
                    "position": 3,
                    "remainingCards": 3,
                    "actingSeat": "west",
                },
            },
            "outcome": _leaf(
                "PRED-TARGET-WINS-ROUND",
                "outcome.team_round",
                {"team": "west_east", "status": "won"},
            ),
        }
    )


def _exact_unique_winner_proof() -> ScenarioProofRequestV1:
    return ScenarioProofRequestV1.build(
        {
            "schemaVersion": 1,
            "mode": "exact",
            "visibility": "complete_state",
            "claim": "winning_root_action_count",
            "targetTeam": "west_east",
            "utility": "round_win",
            "successThreshold": 1,
            "budgetProfile": "late_game_exact_v1",
        }
    )


def _nearest_feasible_changed_scenario() -> ResearchScenarioV1:
    """Changed condition: the target team, not the opponent, wins trick six."""

    return _flagship_scenario(target_team_wins_trick_six=True)


@pytest.mark.real_clock
@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        (("infeasible", "infeasible"), ("infeasible", "infeasible")),
        (("infeasible", "feasible"), ("infeasible", "feasible")),
        (("feasible",), ("feasible",)),
        (("unknown",), ("unknown",)),
    ],
)
@pytest.mark.parametrize("native_conversion", [False, True])
def test_tail_policy_screen_cuts_only_when_both_modes_are_exactly_infeasible(
    monkeypatch, rules, statuses, expected, native_conversion,
) -> None:
    calls = []

    def solve(_stage, **kwargs):
        assert _stage.native_conversion is native_conversion
        calls.append(kwargs["locked"])
        status = statuses[len(calls) - 1]
        if status == "feasible":
            return object()
        raise PlayTailSolveStopped(
            status, evidence={"solverStatus": status},
        )

    monkeypatch.setattr(OrToolsPlayTailStageV1, "solve", solve)
    tail = SimpleNamespace(
        initial_hands=((), (), (), ()),
        plays=(),
        evidence_hash="0" * 64,
    )
    assert tail_candidate_module._tail_policy_prefix_screen_v1(
        tail=tail,
        tail_offset=20,
        rules=rules,
        contract=Contract.SUN,
        trump=None,
        first_leader=Seat.NORTH,
        policy_prefix_steps=20,
        policy_id="observer_safe_tactical_category_v1",
        deadline=time.monotonic() + 10,
        maximum_conflicts=1000,
        seed=0,
        cancel_check=lambda: False,
        native_conversion=native_conversion,
    ) == expected
    assert calls == [False, True][:len(expected)]


@pytest.mark.parametrize(
    ("current_raw", "total_raw", "winning_count", "expected"),
    [
        (40, 100, 0, (41, 100)),
        (40, 100, 1, (0, 40)),
        (40, 100, 2, (0, 40)),
        (100, 100, 0, None),
        (0, 100, 1, (0, 0)),
    ],
)
def test_known_root_count_narrows_raw_score_probe_interval(
    current_raw, total_raw, winning_count, expected
) -> None:
    assert tail_candidate_module._first_winning_raw_interval_v1(
        current_raw, total_raw, winning_count
    ) == expected


def test_legacy_exact_user_flagship_returns_named_structural_infeasibility_without_search(
    rules, monkeypatch,
) -> None:
    scenario = _flagship_scenario()
    plan = TailFirstStagePlannerV1().plan(scenario)

    assert plan.planner == "tail_first"
    assert plan.prefix_length == 23
    assert plan.tail_offset == 20
    assert plan.tail_length == 12

    def forbid_search(*_args, **_kwargs):
        pytest.fail("structural infeasibility must stop before planning search")

    monkeypatch.setattr(TailFirstStagePlannerV1, "plan", forbid_search)
    executor = OrToolsUnifiedResearchExecutorV1(
        rules,
        bundle_revision_id="CBR-FLAGSHIP-SEARCH-TEST",
        bundle_hash="a" * 64,
    )
    result = executor.execute(
        scenario,
        _exact_unique_winner_proof(),
        maximum_proof_nodes=100_000,
        maximum_proof_depth=12,
        maximum_candidates=128,
        maximum_seconds=60,
    )
    assert result.status == "infeasible"
    assert result.candidate is None
    assert result.proof is None
    receipt = result.execution_receipt["structuralInfeasibility"]
    assert receipt["status"] == "exact_structural_infeasibility"
    assert receipt["stage"] == "preflight"
    assert receipt["theorem"]["theoremId"] == ("FLAGSHIP-TWO-ACES-TRICK7-LEAD-V1")
    assert receipt["namedCorePredicateIds"] == [
        "PRED-CONTRACT-SUN",
        "PRED-ROOT-ACE",
        "PRED-OPPONENT-WINS-SIX",
        "PRED-TEAM-WINS-SEVEN",
        "PRED-TWO-OPPONENT-ACES-EIGHT",
        "PRED-ONE-WINNING-ROOT",
    ]
    assert receipt["theorem"]["diagnostics"][0]["diagnosticId"] == (
        "ALT-TARGET-WINS-TRICK-6"
    )
    assert result.execution_receipt["completedCandidates"] == 0
    assert result.execution_receipt["stageTimings"]["tailConstruction"] == 0
    assert result.execution_receipt["stageTimings"]["proofBoundary"] == 0
    assert result.execution_receipt["stageTimings"]["prefixLift"] == 0
    artifacts = result.semantic_artifact_documents()
    assert (
        artifacts["constructionModel"]["structuralInfeasibility"]["sha256"]
        == (receipt["sha256"])
    )

    repeated = executor.execute(
        scenario,
        _exact_unique_winner_proof(),
        maximum_proof_nodes=100_000,
        maximum_proof_depth=12,
        maximum_candidates=128,
        maximum_seconds=60,
    )
    assert repeated.semantic_hash == result.semantic_hash
    assert repeated.execution_receipt["structuralInfeasibility"] == receipt


def test_v3_does_not_confuse_a_witnessed_trick_with_a_forced_trick(root) -> None:
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    executor = OrToolsUnifiedResearchExecutorV1(
        rules,
        backend_id="ortools_cp_sat_v3",
        bundle_revision_id="CBR-FLAGSHIP-V3-TEST",
        bundle_hash="a" * 64,
    )
    result = executor.execute(
        _flagship_scenario(),
        _exact_unique_winner_proof(),
        maximum_proof_nodes=100_000,
        maximum_proof_depth=12,
        maximum_prefix_lift_attempts=3,
        maximum_candidates=1,
        maximum_seconds=0.01,
    )
    assert result.status == "incomplete"
    assert "structuralInfeasibility" not in result.execution_receipt


def test_v3_counterstrategy_fails_claim_without_inventing_a_candidate(root, monkeypatch):
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    executor = OrToolsUnifiedResearchExecutorV1(
        rules, backend_id="ortools_cp_sat_v3",
        bundle_revision_id="CBR-FLAGSHIP-V3-TEST", bundle_hash="a" * 64)
    result = executor.execute(
        _flagship_scenario(), _exact_unique_winner_proof(),
        maximum_proof_nodes=100_000, maximum_proof_depth=12,
        maximum_candidates=1, maximum_seconds=15, maximum_conflicts=1_000_000)
    assert result.status == "proof_failed"
    assert result.candidate is None
    assert result.proof is None
    assert result.execution_receipt["counterstrategy"]["status"] == "certified"
    assert "structuralInfeasibility" not in result.execution_receipt
    documents = result.semantic_artifact_documents()
    certificate = documents["proofModel"]["certificate"]
    assert "executionReceipt" not in certificate
    assert documents["proofOutcome"]["certificateHash"] == certificate["sha256"]
    assert documents["proofOutcome"]["stateSpecificRootActionsEvaluated"] is False
    attempts = []
    def execute(*args, **kwargs):
        attempts.append(kwargs)
        return result
    monkeypatch.setattr(executor, "execute", execute)
    campaign = executor._execute_seed_campaign(
        _flagship_scenario(), _exact_unique_winner_proof(), result_count=5,
        symmetry_policy_id="rules_preserving_v1", allow_symmetry_only_variants=False,
        minimum_decision_state_distance=1, maximum_candidates=10, maximum_seconds=15)
    assert campaign.status == "proof_failed"
    assert len(attempts) == 1


def test_sun_ace_sacrifice_and_two_aces_in_last_trick_has_legal_witness_tail(
    rules,
) -> None:
    reference = ReferencePlayInterpreterV1(rules.play_domain, rules.structural_domain)
    hands = {
        Seat.SOUTH: ["C9", "D7", "CA"],
        Seat.EAST: ["C7", "D9", "H9"],
        Seat.NORTH: ["C8", "D8", "SA"],
        Seat.WEST: ["HA", "S7", "H8"],
    }
    tricks = (
        ((Seat.SOUTH, "C9"), (Seat.EAST, "C7"), (Seat.NORTH, "C8"), (Seat.WEST, "HA")),
        ((Seat.SOUTH, "D7"), (Seat.EAST, "D9"), (Seat.NORTH, "D8"), (Seat.WEST, "S7")),
        ((Seat.EAST, "H9"), (Seat.NORTH, "SA"), (Seat.WEST, "H8"), (Seat.SOUTH, "CA")),
    )
    assert len({card for hand in hands.values() for card in hand}) == 12
    for trick, expected_winner in zip(
        tricks, (Seat.SOUTH, Seat.EAST, Seat.EAST), strict=True
    ):
        played = []
        for seat, code in trick:
            card = Card.parse(code)
            legal = reference.legal_moves(
                [Card.parse(value) for value in hands[seat]],
                played,
                actor=seat,
                contract=Contract.SUN,
                trump=None,
            )
            assert card in legal
            hands[seat].remove(code)
            played.append(ReferencePlayV1(seat, card))
        assert reference.trick_winner(
            played, contract=Contract.SUN, trump=None
        ) == expected_winner
    assert all(not hand for hand in hands.values())


@pytest.mark.parametrize("opposing_aces", [1, 2])
def test_native_original_tail_has_ace_counterstrategy_for_every_root_ace(root, opposing_aces):
    """Witnessed trick wins are not forced wins; certify the actual deviation."""
    from ortools.sat.python import cp_model
    from baloot_platform.core.cards import FULL_DECK
    from baloot_platform.core.enums import Rank
    from baloot_platform.research.ortools_play_tail import (
        TailActionConstraintV1, TailTrickWinnerConstraintV1,
        TailTrickCardCountConstraintV1,
    )
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    stage = OrToolsPlayTailStageV1(native_conversion=True)
    options = dict(cards=FULL_DECK, ownership=None, contract=Contract.SUN,
        trump=None, first_leader=Seat.SOUTH, play_domain=rules.play_domain,
        structural_domain=rules.structural_domain, selected_play_count=12,
        decision_prefix_length=3, decision_actor=Seat.WEST,
        action_constraints=(TailActionConstraintV1.build(step=3, ranks=["ace"]),))
    witness = stage.build_model(**options,
        winner_constraints=tuple(TailTrickWinnerConstraintV1.build(
            trick=index, team=team, ranks=["seven", "eight", "nine"])
            for index, team in enumerate((Team.NORTH_SOUTH, Team.WEST_EAST, Team.WEST_EAST))),
        card_count_constraints=(TailTrickCardCountConstraintV1.build(
            trick=2, minimum=opposing_aces, maximum=opposing_aces,
            owner_team=Team.NORTH_SOUTH, ranks=["ace"]),))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 10
    aces = [card for card in FULL_DECK if card.rank == Rank.ACE]
    # The fixed line makes the positive control cheap, not the universal queries.
    line = ("C9", "C7", "C8", "HA", "D7", "D9", "D8", "S7", "H9", "SA", "H8", "CA")
    def positive_control(model):
        control = model.clone()
        for variable, card in zip(witness.selected_play_indexes, line, strict=True):
            control.add(control.get_int_var_from_proto_index(variable.index) == Card.parse(card).id)
        assert solver.solve(control) in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    if opposing_aces == 2:
        positive_control(witness.model)
    retained = witness.model.clone()
    legal = witness.native_legal_action_literals(4)
    retained.add(sum(retained.get_bool_var_from_proto_index(legal[card.code].index)
                     for card in aces) == 0)
    retention_status = solver.solve(retained)
    if opposing_aces == 1:
        # A nearby request need not give the opposing leader an Ace. Never
        # generalize the original contradiction to this different scenario.
        assert retention_status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        return
    assert retention_status == cp_model.INFEASIBLE

    alternative = stage.build_model(**options, shared_model=witness.model,
        shared_owner_of_card=witness.initial_owner_variables)
    for left, right in zip(witness.selected_card_literals, alternative.selected_card_literals, strict=True):
        witness.model.add(left == right)
    for index in range(3):
        witness.model.add(witness.selected_play_indexes[index] == alternative.selected_play_indexes[index])
    positive_control(witness.model)
    changed_winner = witness.model.clone()
    changed_winner.add(changed_winner.get_int_var_from_proto_index(
        alternative.winner_team_variables[0].index) != int(Team.NORTH_SOUTH))
    assert solver.solve(changed_winner) == cp_model.INFEASIBLE

    # Permit every legal response, changing only the opposing leader's choice.
    witness.model.add_allowed_assignments([alternative.selected_play_indexes[4]], [(card.id,) for card in aces])
    positive_control(witness.model)
    failed_lead = witness.model.clone()
    failed_lead.add(failed_lead.get_int_var_from_proto_index(
        alternative.winner_team_variables[1].index) != int(Team.NORTH_SOUTH))
    assert solver.solve(failed_lead) == cp_model.INFEASIBLE
    insufficient_raw = witness.model.clone()
    ace_points = rules.play_domain.points(aces[0], Contract.SUN, None)
    insufficient_raw.add(sum(insufficient_raw.get_int_var_from_proto_index(value.index)
                             for value in alternative.trick_point_variables[:2]) < 2 * ace_points)
    assert solver.solve(insufficient_raw) == cp_model.INFEASIBLE


@pytest.mark.parametrize("profile", ["saudi_federation_2023_v2", "saudi_federation_2023_sweep"])
def test_native_counterstrategy_receipt_binds_recipe_rules_and_all_models(root, profile):
    from baloot_platform.research.scenario_contracts import canonical_sha256
    rules = RuleBook.load(root / f"configs/rules/{profile}.yaml")
    scenario, proof = _flagship_scenario(), _exact_unique_winner_proof()
    receipt = OrToolsFlagshipResearchV1(rules=rules).certify_round_win_counterstrategy(
        scenario, proof, maximum_seconds=15, maximum_conflicts=1_000_000)
    assert receipt["status"] == "certified", json.dumps(receipt, sort_keys=True)
    assert receipt["scenarioHash"] == scenario.sha256
    assert receipt["proofRequestHash"] == proof.sha256
    assert receipt["rulesHash"] == rules.sha256
    assert len(receipt["namedCorePredicateIds"]) == 10
    assert len(receipt["facts"]) == len({item["factId"] for item in receipt["facts"]}) == 6
    assert all(item["solverStatus"] == "infeasible" and item["variableCount"] > 0
               and item["constraintCount"] > 0 and len(item["modelHash"]) == 64
               for item in receipt["facts"])
    assert receipt["sha256"] == canonical_sha256({key: value for key, value in receipt.items()
        if key not in {"sha256", "executionReceipt"}})


@pytest.mark.parametrize("change", ["winner", "ace_count", "eligible_only", "any", "prefer", "extra_rank", "bounded"])
def test_native_counterstrategy_recognizer_does_not_broaden_the_claim(root, monkeypatch, change):
    owner = OrToolsFlagshipResearchV1(rules=RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml"))
    scenario, proof = _flagship_scenario().to_dict(), _exact_unique_winner_proof().to_dict()
    scenario.pop("sha256", None)
    expression = scenario["constraints"]
    expression.pop("sha256", None)
    leaves = {item["predicateId"]: item for item in expression["children"]}
    for item in leaves.values():
        item.pop("sha256", None)
    if change == "winner":
        leaves["PRED-OPPONENT-WINS-SIX"]["arguments"]["team"] = "west_east"
    elif change == "ace_count":
        leaves["PRED-TWO-OPPONENT-ACES-EIGHT"]["arguments"].update(minimum=1, maximum=1)
    elif change == "eligible_only":
        leaves["PRED-AWARDED-HUNDRED"]["arguments"]["state"] = "eligible"
    elif change == "any":
        expression["op"] = "any"
    elif change == "prefer":
        leaves["PRED-AWARDED-HUNDRED"]["hardness"] = "prefer"
    elif change == "extra_rank":
        leaves["PRED-WINNING-ROOT-IS-ACE"]["arguments"]["selector"]["ranks"].append("king")
    else:
        proof.pop("sha256", None)
        proof.update(mode="bounded", horizon={"unit": "tricks", "value": 1},
                     boundaryEvaluator="current_raw_point_margin_v1")
    monkeypatch.setattr(owner, "_round_win_counterstrategy_models",
        lambda *_args: pytest.fail("uncovered request must not launch a theorem solve"))
    assert owner.certify_round_win_counterstrategy(ResearchScenarioV1.build(scenario),
        ScenarioProofRequestV1.build(proof), maximum_seconds=15, maximum_conflicts=1_000_000) is None


@pytest.mark.real_clock
@pytest.mark.parametrize("cancelled", [False, True])
def test_native_counterstrategy_cancelled_is_not_impossible(root, cancelled):
    owner = OrToolsFlagshipResearchV1(rules=RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml"))
    receipt = owner.certify_round_win_counterstrategy(_flagship_scenario(), _exact_unique_winner_proof(),
        maximum_seconds=15 if cancelled else 1e-9, maximum_conflicts=1_000_000,
        cancel_check=lambda: cancelled)
    assert receipt["status"] == "incomplete"
    assert receipt["facts"] == []


@pytest.mark.parametrize("unknown", [False, True])
def test_native_counterstrategy_non_infeasible_query_never_certifies(root, monkeypatch, unknown):
    from ortools.sat.python import cp_model
    import baloot_platform.research.ortools_flagship_research as module
    owner = OrToolsFlagshipResearchV1(rules=RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml"))
    model = cp_model.CpModel()
    model.new_bool_var("counterexample")
    monkeypatch.setattr(owner, "_round_win_counterstrategy_models", lambda *_: iter((("counterexample", model),)))
    if unknown:
        def interrupted(solver, query, _cancel):
            solver.solve(query)  # Populate the solver's telemetry, then simulate interruption.
            return cp_model.UNKNOWN
        monkeypatch.setattr(module, "_solve_cp_sat_with_cancel_v1", interrupted)
    receipt = owner.certify_round_win_counterstrategy(_flagship_scenario(), _exact_unique_winner_proof(),
        maximum_seconds=15, maximum_conflicts=1_000_000)
    assert receipt["status"] == ("incomplete" if unknown else "not_certified")
    assert len(receipt["facts"]) == 1
    assert receipt["facts"][0]["solverStatus"] == ("unknown" if unknown else "optimal")


def test_nearest_feasible_changed_condition_is_not_the_exact_user_scenario(
    rules, root,
) -> None:
    changed = _nearest_feasible_changed_scenario()
    proof = _exact_unique_winner_proof()

    assert (
        OrToolsFlagshipResearchV1(rules=rules).contradiction_core_for_scenario(
            changed, proof
        )
        is None
    )
    changed_winner = next(
        predicate
        for predicate in changed.predicates()
        if predicate.predicate_id == "PRED-TARGET-WINS-SIX-CHANGED"
    )
    assert changed_winner.arguments["team"] == "west_east"
    assert tail_candidate_module._unique_root_suit_release_guidance_v1(
        changed, proof
    ) == (5, 3, (6, 7))
    # Preserve the old artifact, but never advertise its different sixth-trick
    # winner as evidence for the original opponent-wins-six request.
    fixture = json.loads((root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(encoding="utf-8"))
    plays = fixture["exactTracePrefix"]["playActions"][20:]
    trick = [ReferencePlayV1(Seat.parse(item["actor"]), Card.parse(item["card"])) for item in plays]
    trick.append(ReferencePlayV1(Seat.WEST, Card.parse(fixture["proof"]["selectedAction"])))
    assert ReferencePlayInterpreterV1(rules.play_domain, rules.structural_domain).trick_winner(
        trick, contract=Contract.SUN, trump=None,
    ) == Seat.EAST


@pytest.mark.parametrize("minimum,maximum,operator,selector,expected", [
    (1, 1, "all", False, True),
    (1, 2, "all", False, True),
    (1, 3, "all", False, True),
    (2, 3, "all", False, False),
    (1, 2, "any", False, False),
    (1, 2, "all", True, False),
])
def test_first_winning_range_probe_requires_count_only_hard_conjunction(
    minimum, maximum, operator, selector, expected,
) -> None:
    children = [
        _leaf("PRED-COUNT", "proof.winning_root_action_count",
              {"minimum": minimum, "maximum": maximum}),
        _leaf("PRED-SUN", "auction.contract", {"contract": "sun"}),
    ]
    if selector:
        children.append(_leaf("PRED-CARD", "proof.winning_root_action_matches",
                              {"selector": {"ranks": ["ace"]}}))
    scenario = ResearchScenarioV1.build({
        "schemaVersion": 1, "origin": {"kind": "search"},
        "constraints": {"op": operator, "children": children},
        "decisionPoint": {"kind": "play", "selector": {"trickIndex": 4}},
        "outcome": _leaf("PRED-WIN", "outcome.team_round",
                         {"team": "north_south", "status": "won"}),
    })
    predicates = tuple(p for p in scenario.predicates() if p.kind.startswith("proof."))
    assert tail_candidate_module._first_winning_count_range_required_v1(scenario, predicates) is expected
    assert not tail_candidate_module._first_winning_count_range_required_v1(scenario, ())


def test_unique_raw_root_cut_respects_required_winning_card_selector() -> None:
    selector = tail_candidate_module.CardSelectorV1.build(
        {"ranks": ["ace"], "owner": {"relation": "actor"}}
    )
    ace = tail_candidate_module.ProofRootActionV1(0, "CA", 12, True)
    king = tail_candidate_module.ProofRootActionV1(1, "SK", 8, False)
    matches = tail_candidate_module._unique_raw_root_matches_selectors_v1

    assert matches((ace, king), actor=Seat.WEST, selectors=(selector,))
    assert not matches((replace(ace, value=8), replace(king, value=12)),
                       actor=Seat.WEST, selectors=(selector,))
    assert not matches((ace, replace(king, value=12)),
                       actor=Seat.WEST, selectors=(selector,))


@pytest.mark.real_clock
@pytest.mark.parametrize("reject_awarded_boundary", [False, "awarded_point_margin", "round_win", "native_awarded_point_margin", "native_awarded_range"])
def test_tail_tactic_boundary_binds_supported_proof_predicates_in_shared_model(
    monkeypatch,
    rules,
    reject_awarded_boundary,
) -> None:
    scenario = _nearest_feasible_changed_scenario()
    proof_request = _exact_unique_winner_proof()
    native_range = reject_awarded_boundary == "native_awarded_range"
    native_awarded_boundary = reject_awarded_boundary in {"native_awarded_point_margin", "native_awarded_range"}
    if native_range:
        raw = scenario.to_dict()
        raw.pop("sha256")
        raw["constraints"] = {"op": "all", "children": [
            _leaf(p.predicate_id, p.kind,
                  {"minimum": 1, "maximum": 2} if p.kind == "proof.winning_root_action_count" else dict(p.arguments))
            for p in scenario.constraints.predicates()
            if p.kind != "proof.winning_root_action_matches"
        ]}
        scenario = ResearchScenarioV1.build(raw)
    if native_awarded_boundary:
        reject_awarded_boundary = "awarded_point_margin"
    if reject_awarded_boundary:
        request = proof_request.to_dict()
        request.pop("sha256", None)
        request["utility"] = reject_awarded_boundary
        proof_request = ScenarioProofRequestV1.build(request)
    plan = TailFirstStagePlannerV1().plan(scenario)
    counts = {"tail": 0, "boundary": 0, "lift": 0}
    exclusions = []
    decision_state_cuts = []
    requirements = []
    progress = []
    native_qualification = []
    project_ownership = []
    raw_probes = []
    original_tail = OrToolsPlayTailStageV1.solve
    original_boundary = OrToolsTailProofBoundaryV1.solve
    original_projects = tail_boundary_module.attach_project_lifecycle_fragment_v1
    first_boundary = None
    if reject_awarded_boundary:
        original_prove = tail_candidate_module.OrToolsScenarioGraphProofV1.prove

        def reject_awarded(prover, graph, request, **kwargs):
            raw_probe = request.utility == tail_candidate_module.INTERNAL_RAW_POINT_MARGIN_UTILITY_V1
            assert request.utility == reject_awarded_boundary or (raw_probe and (reject_awarded_boundary == "round_win" or native_awarded_boundary)), (
                "public scoring must not be replaced by a raw-margin exclusion"
            )
            proof = original_prove(prover, graph, request, **kwargs)
            if raw_probe:
                raw_probes.append(request.sha256)
                assert first_boundary.seed.contract.effective == Contract.SUN
                assert all(first_boundary.seed.base_tricks)
                return replace(proof, root_actions=tuple(
                    replace(action, value=0) for action in proof.root_actions))
            if reject_awarded_boundary == "round_win" or native_range:
                # Exercise the first-winning-split certificate without making
                # this orchestration test depend on a particular searched deal.
                return replace(proof, claim_satisfied=False, winning_root_action_count=3 if native_range else 2)
            return replace(proof, claim_satisfied=False)

        monkeypatch.setattr(
            tail_candidate_module.OrToolsScenarioGraphProofV1,
            "prove", reject_awarded,
        )
        if reject_awarded_boundary == "round_win" or native_range:
            monkeypatch.setattr(
                tail_candidate_module, "_first_winning_raw_interval_v1",
                lambda *_args: (0, 0),
            )
    monkeypatch.setattr(
        tail_candidate_module,
        "_unique_root_suit_release_guidance_v1",
        lambda *_args: None,
    )

    def solve_tail(stage, **kwargs):
        if kwargs.get("construction_plan_hash") is not None:
            counts["tail"] += 1
            decision_state_cuts.append(
                tuple(kwargs.get("exact_excluded_decision_status_vectors", ()))
            )
            if counts["tail"] > 1:
                raise PlayTailSolveStopped(
                    "test_tail_stop",
                    evidence={"solverStatus": "unknown"},
                )
        return original_tail(stage, **kwargs)

    def solve_boundary(stage, *args, **kwargs):
        nonlocal first_boundary
        counts["boundary"] += 1
        exclusions.append(
            (
                tuple(kwargs.get("excluded", ())),
                tuple(kwargs.get("excluded_proof_states", ())),
            )
        )
        requirements.append(
            (
                kwargs.get("required_history"),
                kwargs.get("required_proof_state"),
            )
        )
        if first_boundary is None:
            first_boundary = original_boundary(stage, *args, **kwargs)
            return first_boundary
        return replace(
            first_boundary,
            status="incomplete",
            seed=None,
            no_good=None,
            reason="test_boundary_stop",
        )

    def unexpected_lift(*_args, **_kwargs):
        counts["lift"] += 1
        raise AssertionError("proof-false boundary reached prefix lift")

    def attach_projects(*args, **kwargs):
        project_ownership.append(kwargs)
        return original_projects(*args, **kwargs)

    def inspect_native_bindings(_scenario, proof, _manifest, **_kwargs):
        native_qualification.append(proof.native_qualification)
        return "incomplete", {"reason": "test_stop_after_native_bindings"}

    monkeypatch.setattr(OrToolsPlayTailStageV1, "solve", solve_tail)
    monkeypatch.setattr(OrToolsTailProofBoundaryV1, "solve", solve_boundary)
    monkeypatch.setattr(
        tail_boundary_module, "attach_project_lifecycle_fragment_v1", attach_projects
    )
    monkeypatch.setattr(OrToolsTailFirstBackwardLiftV1, "lift", unexpected_lift)
    monkeypatch.setattr(
        tail_candidate_module,
        "_proof_stage_projection_v1",
        inspect_native_bindings,
    )
    result = OrToolsTailFirstCandidateCompilerV1(
        rules=rules,
        plan=plan,
        proof_request=proof_request,
        maximum_proof_nodes=100_000,
        maximum_proof_depth=12,
        maximum_line_count_limbs=64,
        maximum_conflicts=10_000_000,
        maximum_prefix_lift_attempts=1,
        progress_callback=progress.append,
        native_cp_sat_scoring=native_awarded_boundary,
    )(
        scenario,
        (),
        seed=1,
        maximum_seconds=15,
        maximum_conflicts=10_000_000,
    )

    assert result.status == "incomplete"
    assert result.reason == (
        "test_boundary_stop" if reject_awarded_boundary
        else "tail_proof_predicate_evidence"
    )
    assert counts == {
        "tail": 1, "boundary": 2 if reject_awarded_boundary else 1, "lift": 0,
    }
    if reject_awarded_boundary:
        assert len(raw_probes) == int(
            (reject_awarded_boundary == "round_win" or native_awarded_boundary) and not native_range
            and all(first_boundary.seed.base_tricks)
            and first_boundary.seed.next_actor.team == Team[proof_request.target_team.upper()]
        )
        rejected = exclusions[1][1]
        original = first_boundary.proof_state_no_good
        if reject_awarded_boundary == "round_win" or native_awarded_boundary and raw_probes or native_range:
            raw_count = sum(first_boundary.seed.base_raw) + 1
            trick_counts = (set(range(1, sum(first_boundary.seed.base_tricks)))
                            if raw_probes else {original.base_tricks_ns})
            assert len(rejected) == raw_count * len(trick_counts)
            assert {(item.base_raw_ns, item.base_tricks_ns) for item in rejected} == {
                (raw, tricks) for raw in range(raw_count) for tricks in trick_counts}
            for item in rejected:
                assert item.verify()
                assert (item.decision_state_hash, item.boundary_combination,
                        item.awarded_project_points,
                        item.baloot_teams, item.escalation_semantics) == (
                            original.decision_state_hash, original.boundary_combination,
                            original.awarded_project_points,
                            original.baloot_teams, original.escalation_semantics)
            if native_range:
                assert not raw_probes
                assert result.partial_receipt["evidence"]["stageCounters"]["proofStateProbes"] > 0
        else:
            assert rejected == (original,)
        assert not native_qualification
        return
    assert set(native_qualification[0]) == {
        "PRED-ONE-WINNING-ROOT",
        "PRED-WINNING-ROOT-IS-ACE",
    }
    assert decision_state_cuts[0] == ()
    assert exclusions[0] == ((), ())
    assert requirements == [(None, None)]
    assert first_boundary is not None
    assert first_boundary.model_build_seconds > 0
    assert first_boundary.solver_seconds > 0
    assert first_boundary.solve_attempts >= 1
    assert project_ownership
    assert sum(owner is not None for owner in project_ownership[0]["fixed_card_owners"]) >= 12
    assert project_ownership[0]["compact_fixed_states"] is True
    assert project_ownership[0]["tighten_value_bounds"] is True
    assert first_boundary.evidence_bindings
    assert set(first_boundary.evidence_bindings[0]) == {
        "predicateId",
        "sourceEvidenceHash",
        "loweringHash",
        "modelHash",
        "literalName",
        "literalIndex",
        "evidenceHash",
    }
    assert all(
        binding["sourceEvidenceHash"] == first_boundary.evidence_hash
        for binding in first_boundary.evidence_bindings
    )


def test_prefix_lift_attempt_cap_is_exact_and_reuses_tail(monkeypatch, rules) -> None:
    scenario = _nearest_feasible_changed_scenario()
    proof_request = _exact_unique_winner_proof()
    plan = TailFirstStagePlannerV1().plan(scenario)
    counts = {"tail": 0, "boundary": 0, "lift": 0}
    exclusions = []
    original_tail = OrToolsPlayTailStageV1.solve
    original_boundary = OrToolsTailProofBoundaryV1.solve
    first_boundary = None

    def solve_tail(stage, **kwargs):
        if kwargs.get("construction_plan_hash") is not None:
            counts["tail"] += 1
        return original_tail(stage, **kwargs)

    def solve_boundary(stage, *args, **kwargs):
        nonlocal first_boundary
        counts["boundary"] += 1
        exclusions.append(
            (
                tuple(kwargs.get("excluded", ())),
                tuple(kwargs.get("excluded_proof_states", ())),
            )
        )
        if first_boundary is None:
            first_boundary = original_boundary(stage, *args, **kwargs)
        return first_boundary

    def failed_lift(*_args, **_kwargs):
        counts["lift"] += 1
        return OrToolsCandidateAttemptV1("infeasible", None, "test_lift_infeasible")

    monkeypatch.setattr(OrToolsPlayTailStageV1, "solve", solve_tail)
    monkeypatch.setattr(OrToolsTailProofBoundaryV1, "solve", solve_boundary)
    monkeypatch.setattr(OrToolsTailFirstBackwardLiftV1, "lift", failed_lift)
    monkeypatch.setattr(
        tail_candidate_module,
        "_proof_stage_projection_v1",
        lambda *_args, **_kwargs: ("satisfied", {}),
    )
    progress = []
    result = OrToolsTailFirstCandidateCompilerV1(
        rules=rules,
        plan=plan,
        proof_request=proof_request,
        maximum_proof_nodes=100_000,
        maximum_proof_depth=12,
        maximum_line_count_limbs=64,
        maximum_conflicts=10_000_000,
        maximum_prefix_lift_attempts=1,
        progress_callback=progress.append,
    )(
        scenario,
        (),
        seed=1,
        maximum_seconds=15,
        maximum_conflicts=10_000_000,
    )

    assert result.status == "incomplete"
    assert result.reason == "prefix_lift_attempt_budget"
    assert any(
        item.get("prefixLiftReason") == "test_lift_infeasible"
        and item.get("prefixLiftStatus") == "infeasible"
        for item in progress
    )
    assert counts == {"tail": 1, "boundary": 2, "lift": 1}
    assert exclusions[0] == ((), ())
    assert len(exclusions[1][0]) == 1
    assert len(exclusions[1][1]) == 1
    assert result.partial_receipt is not None
    assert result.partial_receipt["evidence"]["prefixLiftFailures"] == [{
        "attempt": 1,
        "status": "infeasible",
        "reason": "test_lift_infeasible",
        "proofStateProjectionHash": first_boundary.seed.proof_state_projection_hash,
        "boundaryEvidenceHash": first_boundary.evidence_hash,
    }]
    assert result.partial_receipt["evidence"]["stageCounters"] == {
        "tailCandidates": 1,
        "prefixReachabilityCuts": 0,
        "prefixReachabilityTailCuts": 0,
        "proofBoundaryAttempts": 2,
        "proofBoundarySolverAttempts": 2,
        "proofPredicateRejections": 0,
        "proofRejectedZeroWinningRoots": 0,
        "proofRejectedOneWinningRoot": 0,
        "proofRejectedMultipleWinningRoots": 0,
        "proofStateProbes": 0,
        "rawRootCacheHits": 0,
        "scoreProbeCacheHits": 0,
        "prefixLiftAttempts": 1,
    }


def test_tail_graph_exhaustion_retains_partial_graph_receipt(monkeypatch, rules) -> None:
    scenario = _nearest_feasible_changed_scenario()
    plan = TailFirstStagePlannerV1().plan(scenario)
    graph_receipt = {
        "status": "incomplete",
        "exhaustionReason": "node_budget",
        "exploration": {"nodes": 7, "edges": 9},
    }

    def stopped_graph(*_args, **_kwargs):
        return SimpleNamespace(
            complete=False,
            graph=None,
            receipt=SimpleNamespace(
                exhaustion_reason="node_budget",
                to_dict=lambda: graph_receipt,
            ),
        )

    monkeypatch.setattr(
        tail_candidate_module.OrToolsCompleteStateGraphCompilerV1,
        "compile", stopped_graph,
    )
    attempt = OrToolsTailFirstCandidateCompilerV1(
        rules=rules, plan=plan,
        proof_request=_exact_unique_winner_proof(),
        maximum_proof_nodes=100_000, maximum_proof_depth=12,
        maximum_line_count_limbs=64, maximum_conflicts=10_000_000,
        maximum_prefix_lift_attempts=1,
    )(
        scenario, (), seed=1, maximum_seconds=15,
        maximum_conflicts=10_000_000,
    )

    assert attempt.status == "incomplete"
    assert attempt.reason == "tail_proof_graph_node_budget"
    assert attempt.partial_receipt is not None
    assert attempt.partial_receipt["stage"] == "graph"
    assert attempt.partial_receipt["evidence"]["graphReceipt"] == graph_receipt
    assert attempt.partial_receipt["evidence"]["stageCounters"]["tailCandidates"] == 1


def test_exact_complete_boundary_infeasibility_reconstructs_full_play(
    monkeypatch,
    rules,
) -> None:
    scenario = _nearest_feasible_changed_scenario()
    proof_request = _exact_unique_winner_proof()
    plan = TailFirstStagePlannerV1().plan(scenario)
    counts = {"tail": 0, "boundary": 0, "lift": 0}
    original_tail = OrToolsPlayTailStageV1.solve
    original_boundary = OrToolsTailProofBoundaryV1.solve
    first_play = None

    def solve_tail(stage, **kwargs):
        nonlocal first_play
        if kwargs.get("construction_plan_hash") is None:
            return original_tail(stage, **kwargs)
        counts["tail"] += 1
        if counts["tail"] == 2:
            closure = inspect.getclosurevars(kwargs["model_configurator"])
            excluded = closure.nonlocals["excluded_complete_plays"]
            assert excluded[plan.boundary_combinations[0]] == [first_play]
            raise PlayTailSolveStopped(
                "test_tail_stop",
                evidence={"solverStatus": "unknown"},
            )
        result = original_tail(stage, **kwargs)
        first_play = tuple(play.card.id for play in result.plays)
        return result

    def solve_boundary(stage, *args, **kwargs):
        counts["boundary"] += 1
        assert kwargs["required_base_raw_ns"] is not None
        assert kwargs["required_base_tricks_ns"] is not None
        result = original_boundary(stage, *args, **kwargs)
        return replace(
            result,
            status="infeasible",
            seed=None,
            no_good=None,
            proof_state_no_good=None,
            reason="test_complete_play_infeasible",
        )

    def unexpected_lift(*_args, **_kwargs):
        counts["lift"] += 1
        raise AssertionError("infeasible complete play reached prefix lift")

    monkeypatch.setattr(OrToolsPlayTailStageV1, "solve", solve_tail)
    monkeypatch.setattr(OrToolsTailProofBoundaryV1, "solve", solve_boundary)
    monkeypatch.setattr(OrToolsTailFirstBackwardLiftV1, "lift", unexpected_lift)

    result = OrToolsTailFirstCandidateCompilerV1(
        rules=rules,
        plan=plan,
        proof_request=proof_request,
        maximum_proof_nodes=100_000,
        maximum_proof_depth=12,
        maximum_line_count_limbs=64,
        maximum_conflicts=10_000_000,
        maximum_prefix_lift_attempts=1,
    )(
        scenario,
        (),
        seed=1,
        maximum_seconds=15,
        maximum_conflicts=10_000_000,
    )

    assert result.status == "incomplete"
    assert result.reason == "tail_construction_budget"
    assert counts == {"tail": 2, "boundary": 1, "lift": 0}
    assert result.partial_receipt is not None
    assert result.partial_receipt["evidence"]["stageCounters"]["tailCandidates"] == 1


@pytest.mark.parametrize("policy,ranged_score,native", [(False, False, False), (True, False, False),
    (False, True, True), (False, False, True)])
def test_tail_policy_exhaustion_is_not_global_infeasibility(rules, monkeypatch, policy, ranged_score, native):
    scenario = _nearest_feasible_changed_scenario()
    if ranged_score:
        raw = scenario.to_dict()
        raw.pop("sha256")
        raw["constraints"] = {"op": "all", "children": [raw["constraints"],
            _leaf("PRED-INITIAL-SCORE-RANGE", "state.match_score", {
                "team": "north_south", "minimum": 101, "maximum": 103,
            }), _leaf("PRED-INITIAL-OTHER-SCORE", "state.match_score", {
                "team": "west_east", "minimum": 30, "maximum": 30,
            })]}
        scenario = ResearchScenarioV1.build(raw)
    plan = TailFirstStagePlannerV1().plan(scenario)
    def no_tail(*args, **kwargs):
        raise PlayTailSolveStopped("infeasible", evidence={"solverStatus": "infeasible"})
    monkeypatch.setattr(OrToolsPlayTailStageV1, "solve", no_tail)
    compiler = OrToolsTailFirstCandidateCompilerV1(rules=rules, plan=plan,
        proof_request=_exact_unique_winner_proof(), maximum_proof_nodes=100_000,
        maximum_proof_depth=12, maximum_line_count_limbs=64, maximum_conflicts=10_000_000,
        maximum_prefix_lift_attempts=1, require_optimal_auction_actions=policy,
        native_cp_sat_scoring=native)
    attempt = compiler(scenario, (), seed=0, maximum_seconds=15, maximum_conflicts=10_000_000)
    assert attempt.status == ("not_found" if policy or native else "infeasible")
    assert attempt.reason == ("initial_input_choice_no_witness" if native
        else "tail_quality_constrained_no_witness" if policy else "tail_first_infeasible")


@pytest.mark.real_clock
@pytest.mark.parametrize("stop", ["second", "exhausted", "unknown", "cancelled", "deadline"])
def test_native_tail_retries_initial_tuples_without_resetting_budget_or_leaking_cuts(rules, monkeypatch, stop):
    raw = _nearest_feasible_changed_scenario().to_dict()
    raw.pop("sha256")
    raw["constraints"] = {"op": "all", "children": [raw["constraints"],
        _leaf("PRED-INITIAL-NS", "state.match_score", {"team": "north_south", "minimum": 101, "maximum": 102}),
        _leaf("PRED-INITIAL-WE", "state.match_score", {"team": "west_east", "minimum": 30, "maximum": 30})]}
    scenario = ResearchScenarioV1.build(raw)
    plan = TailFirstStagePlannerV1(native_cp_sat=True).plan(scenario)
    original = tail_candidate_module._required_search_initial_inputs_v3
    seen, cancelled = [], [False]
    selected_deadlines = []

    def select_inputs(*args, **kwargs):
        selected_deadlines.append(kwargs["deadline"])
        if kwargs.get("excluded_inputs") and stop == "unknown":
            raise tail_candidate_module.InformationSetCompilationStopped(
                "expression_projection_incomplete", stage="initial_state_inputs")
        return original(*args, **kwargs)

    def no_tail(_stage, **kwargs):
        state = inspect.currentframe().f_back.f_locals
        assert state["excluded_decks"] == (tuple(range(32)),)
        scores = state["starting_match_score"]
        if not seen or seen[-1] != scores:
            seen.append(scores)
            if len(seen) == 2:
                assert state["counters"]["prefixLiftAttempts"] == 1
                for name in ("excluded_tails", "excluded_complete_plays", "excluded_play_zero_actions"):
                    assert all(not values for values in state[name].values())
                assert state["excluded_boundaries"] == state["excluded_proof_states"] == []
        if scores == (101, 30):
            state["counters"]["prefixLiftAttempts"] = 1
            for name in ("excluded_tails", "excluded_complete_plays", "excluded_play_zero_actions"):
                state[name][plan.boundary_combinations[0]].append("previous-tuple-cut")
            state["excluded_boundaries"].append("previous-tuple-cut")
            state["excluded_proof_states"].append("previous-tuple-cut")
            cancelled[0] = stop == "cancelled"
            if stop == "deadline":
                monkeypatch.setattr(tail_candidate_module, "time", SimpleNamespace(
                    monotonic=lambda: selected_deadlines[0] + 1, perf_counter=time.perf_counter))
        raise PlayTailSolveStopped("infeasible" if scores == (101, 30) or stop == "exhausted" else "unknown",
            evidence={"solverStatus": "infeasible" if scores == (101, 30) or stop == "exhausted" else "unknown"})

    monkeypatch.setattr(tail_candidate_module, "_required_search_initial_inputs_v3", select_inputs)
    monkeypatch.setattr(OrToolsPlayTailStageV1, "solve", no_tail)
    compiler = OrToolsTailFirstCandidateCompilerV1(rules=rules, plan=plan,
        proof_request=_exact_unique_winner_proof(), maximum_proof_nodes=100_000,
        maximum_proof_depth=12, maximum_line_count_limbs=64, maximum_conflicts=10_000_000,
        maximum_prefix_lift_attempts=1, native_cp_sat_scoring=True, cancel_check=lambda: cancelled[0])
    attempt = compiler(scenario, (tuple(range(32)),), seed=0, maximum_seconds=15, maximum_conflicts=10_000_000)
    assert seen == ([(101, 30), (102, 30)] if stop in {"second", "exhausted"} else [(101, 30)])
    assert len(set(selected_deadlines)) == 1
    assert attempt.status == ("not_found" if stop == "exhausted" else "incomplete")
    if stop in {"unknown", "cancelled", "deadline"}:
        assert attempt.partial_receipt["evidence"]["stageCounters"]["prefixLiftAttempts"] == 1
        assert attempt.partial_receipt["exhaustionReason"] == {
            "unknown": "expression_projection_incomplete", "cancelled": "cancelled", "deadline": "tail_first_budget",
        }[stop]


def test_native_tail_cancel_after_initial_inputs_is_incomplete(rules, monkeypatch):
    scenario = _nearest_feasible_changed_scenario()
    cancelled = [False]
    original = tail_candidate_module._required_search_initial_inputs_v3

    def select_inputs(*args, **kwargs):
        result = original(*args, **kwargs)
        cancelled[0] = True
        return result

    def forbidden(*args, **kwargs):
        pytest.fail("cancelled candidate must not launch tail construction")

    monkeypatch.setattr(tail_candidate_module, "_required_search_initial_inputs_v3", select_inputs)
    monkeypatch.setattr(OrToolsPlayTailStageV1, "solve", forbidden)
    compiler = OrToolsTailFirstCandidateCompilerV1(rules=rules,
        plan=TailFirstStagePlannerV1(native_cp_sat=True).plan(scenario),
        proof_request=_exact_unique_winner_proof(), maximum_proof_nodes=100_000,
        maximum_proof_depth=12, maximum_line_count_limbs=64, maximum_conflicts=10_000_000,
        maximum_prefix_lift_attempts=1, native_cp_sat_scoring=True,
        cancel_check=lambda: cancelled[0])
    attempt = compiler(scenario, (), seed=0, maximum_seconds=15, maximum_conflicts=10_000_000)
    assert attempt.status == "incomplete" and attempt.reason == "cancelled"
    assert attempt.partial_receipt["exhaustionReason"] == "cancelled"


def test_tail_first_binds_legal_choice_count_before_search(rules, monkeypatch) -> None:
    raw = _nearest_feasible_changed_scenario().to_dict()
    raw.pop("sha256")
    raw["constraints"].pop("sha256")
    raw["constraints"]["children"].append(
        _leaf("PRED-TAIL-LEGAL-CHOICES", "state.legal_action_count", {
            "minimum": 2, "maximum": 2,
        })
    )
    scenario = ResearchScenarioV1.build(raw)

    class Inspected(Exception):
        pass

    def inspect_tail(stage, **kwargs):
        assert kwargs["random_seed"] == 17
        options = dict(kwargs)
        configure = options.pop("model_configurator")
        for key in (
            "construction_plan_hash", "solver", "maximum_time_seconds",
            "maximum_conflicts", "telemetry_callback", "random_seed",
        ):
            options.pop(key, None)
        build = stage.build_model(**options)
        configure(build)
        assert any(
            variable.name == "PRED-TAIL-LEGAL-CHOICES_count"
            for variable in build.model.proto.variables
        )
        assert any(
            build.model.proto.variables[index].name.startswith(
                "assume_tailConstruction_"
            )
            for index in build.model.proto.assumptions
        )
        raise Inspected

    monkeypatch.setattr(OrToolsPlayTailStageV1, "solve", inspect_tail)
    plan = TailFirstStagePlannerV1().plan(scenario)
    compiler = OrToolsTailFirstCandidateCompilerV1(
        rules=rules, plan=plan, proof_request=_exact_unique_winner_proof(),
        maximum_proof_nodes=100_000, maximum_proof_depth=12,
        maximum_line_count_limbs=64, maximum_conflicts=10_000_000,
        maximum_prefix_lift_attempts=1,
    )
    with pytest.raises(Inspected):
        compiler(
            scenario, (), seed=17, maximum_seconds=15,
            maximum_conflicts=10_000_000,
        )


@pytest.mark.parametrize("policy_prefix_steps,expected_tail_steps", [(0, 0), (10, 0), (19, 0), (23, 3)])
def test_v3_tail_constructs_only_remaining_tricks(
    rules, monkeypatch, policy_prefix_steps, expected_tail_steps
) -> None:
    scenario = _nearest_feasible_changed_scenario()
    plan = TailFirstStagePlannerV1().plan(scenario)
    assert plan.planner == "tail_first"

    class Inspected(Exception):
        pass

    def inspect_tail(stage, **kwargs):
        assert kwargs.get("selected_play_count") == plan.tail_length == 12
        assert "play_step_count" not in kwargs
        assert kwargs["decision_prefix_length"] == 3
        assert kwargs["policy_prefix_steps"] == expected_tail_steps
        assert kwargs["shared_model"] is None
        options = dict(kwargs)
        configure = options.pop("model_configurator")
        for key in (
            "construction_plan_hash", "solver", "maximum_time_seconds",
            "maximum_conflicts", "telemetry_callback", "random_seed",
        ):
            options.pop(key, None)
        build = stage.build_model(**options)
        configure(build)
        assert len(build.selected_play_indexes) == 12
        assert build.model.proto.assumptions
        assert not any(variable.name == "score_round_winner"
                       for variable in build.model.proto.variables)
        assert any(
            variable.name.startswith("tail_project_witness_PRED-AWARDED-HUNDRED_")
            for variable in build.model.proto.variables
        )
        raise Inspected

    monkeypatch.setattr(OrToolsPlayTailStageV1, "solve", inspect_tail)
    compiler = OrToolsTailFirstCandidateCompilerV1(
        rules=rules, plan=plan, proof_request=_exact_unique_winner_proof(),
        maximum_proof_nodes=100_000, maximum_proof_depth=12,
        maximum_line_count_limbs=64, maximum_conflicts=10_000_000,
        maximum_prefix_lift_attempts=1, native_cp_sat_scoring=True,
        policy_prefix_steps=policy_prefix_steps,
        require_optimal_auction_actions=True,
    )
    with pytest.raises(Inspected):
        compiler(
            scenario, (), seed=0, maximum_seconds=15,
            maximum_conflicts=10_000_000,
        )


def test_selective_tail_rejects_native_project_outcome_conflict(root, monkeypatch):
    """The mandatory native boundary rejects a legal but unqualifiable tail."""
    from ortools.sat.python import cp_model
    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    scenario = _flagship_scenario()
    class Inspected(Exception):
        pass
    original_tail = OrToolsPlayTailStageV1.solve
    original_solve = tail_boundary_module._solve_cp_sat_with_cancel_v1

    def fixed_tail(stage, **kwargs):
        configure = kwargs["model_configurator"]
        def pin(build):
            configure(build)
            cards = ("D7", "CK", "D8", "CA", "C7", "SK", "H8", "C8", "C9", "HA", "HK", "DA")
            for variable, card in zip(build.selected_play_indexes, cards, strict=True):
                build.model.add(variable == Card.parse(card).id)
        return original_tail(stage, **{**kwargs, "model_configurator": pin})

    def inspect(solver, model, *args, **kwargs):
        assert original_solve(solver, model, *args, **kwargs) == cp_model.INFEASIBLE
        control = model.clone()
        assumptions = tuple(control.proto.assumptions)
        removed = [index for index in assumptions
                   if control.proto.variables[index].name.startswith("assume_outcome_")]
        assert removed
        control.clear_assumptions()
        control.add_assumptions([control.get_bool_var_from_proto_index(index)
                                 for index in assumptions if index not in removed])
        assert solver.solve(control) in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        raise Inspected
    monkeypatch.setattr(OrToolsPlayTailStageV1, "solve", fixed_tail)
    monkeypatch.setattr(tail_boundary_module, "_solve_cp_sat_with_cancel_v1", inspect)
    compiler = OrToolsTailFirstCandidateCompilerV1(
        rules=rules, plan=TailFirstStagePlannerV1().plan(scenario),
        proof_request=_exact_unique_winner_proof(), maximum_proof_nodes=100_000,
        maximum_proof_depth=12, maximum_line_count_limbs=64,
        maximum_conflicts=10_000_000, maximum_prefix_lift_attempts=1,
        native_cp_sat_scoring=True,
    )
    with pytest.raises(Inspected):
        compiler(scenario, (), seed=0, maximum_seconds=30, maximum_conflicts=10_000_000)


def test_v3_proof_screen_retries_boundary_before_expensive_reachability(root, monkeypatch) -> None:
    from baloot_platform.core.rules import RuleBook

    rules = RuleBook.load(root / "configs/rules/saudi_federation_2023_v2.yaml")
    scenario = _nearest_feasible_changed_scenario()
    plan = TailFirstStagePlannerV1().plan(scenario)

    class Screened(Exception):
        pass

    decision_states = []
    original_solve = OrToolsPlayTailStageV1.solve
    reachability_probes = 0

    def count_reachability_probes(stage, **kwargs):
        nonlocal reachability_probes
        if kwargs.get("tail") is not None:
            reachability_probes += 1
        return original_solve(stage, **kwargs)

    monkeypatch.setattr(OrToolsPlayTailStageV1, "solve", count_reachability_probes)

    def progress(event):
        if event.get("decisionStateHash") and event["currentStage"] == "tailConstruction":
            decision_states.append(event["decisionStateHash"])
        if (event["currentStage"] == "proofBoundary"
                and event["stageCounters"]["proofBoundaryAttempts"] == 2):
            assert len(decision_states) == 1
            assert reachability_probes == 0
            raise Screened

    compiler = OrToolsTailFirstCandidateCompilerV1(
        rules=rules, plan=plan, proof_request=_exact_unique_winner_proof(),
        maximum_proof_nodes=100_000, maximum_proof_depth=12,
        maximum_line_count_limbs=64, maximum_conflicts=10_000_000,
        maximum_prefix_lift_attempts=1, native_cp_sat_scoring=True,
        policy_prefix_steps=plan.tail_offset - 4,
        policy_id="observer_safe_tactical_category_v1",
        require_optimal_auction_actions=True,
        progress_callback=progress,
    )
    with pytest.raises(Screened):
        compiler(
            scenario, (), seed=0, maximum_seconds=15,
            maximum_conflicts=10_000_000,
        )


def test_v3_prior_play_requirement_keeps_full_reachability_model(
    rules, monkeypatch
) -> None:
    raw = _nearest_feasible_changed_scenario().to_dict()
    raw.pop("sha256")
    raw["constraints"].pop("sha256")
    raw["constraints"]["children"].append(_leaf(
        "PRED-EARLY-WINNER", "play.trick_winner_matches", {
            "trickIndex": 0, "team": "north_south",
            "selector": {"ranks": ["ace"]},
        },
    ))
    scenario = ResearchScenarioV1.build(raw)
    plan = TailFirstStagePlannerV1().plan(scenario)

    class Inspected(Exception):
        pass

    def inspect_tail(_stage, **kwargs):
        assert kwargs.get("play_step_count") == 32
        assert "selected_play_count" not in kwargs
        assert kwargs["shared_model"] is not None
        raise Inspected

    monkeypatch.setattr(OrToolsPlayTailStageV1, "solve", inspect_tail)
    compiler = OrToolsTailFirstCandidateCompilerV1(
        rules=rules, plan=plan, proof_request=_exact_unique_winner_proof(),
        maximum_proof_nodes=100_000, maximum_proof_depth=12,
        maximum_line_count_limbs=64, maximum_conflicts=250_000,
        maximum_prefix_lift_attempts=1, native_cp_sat_scoring=True,
        require_optimal_auction_actions=True,
    )
    with pytest.raises(Inspected):
        compiler(
            scenario, (), seed=0, maximum_seconds=15,
            maximum_conflicts=250_000,
        )


def test_v3_small_tail_reaches_boundary_before_budget(rules, monkeypatch) -> None:
    scenario = ResearchScenarioV1.build({
        "schemaVersion": 1,
        "origin": {"kind": "search"},
        "constraints": {"op": "all", "children": [
            _leaf("PRED-CONTRACT", "auction.contract", {"contract": "sun"}),
            _leaf("PRED-CHOICES", "state.legal_action_count", {
                "minimum": 2, "maximum": 2,
            }),
            _leaf("PRED-ROOT", "proof.winning_root_action_count", {
                "minimum": 1, "maximum": 1,
            }),
        ]},
        "decisionPoint": {"kind": "play", "selector": {
            "trickIndex": 6, "position": 0, "remainingCards": 2,
            "actingSeat": "south",
        }},
        "outcome": _leaf("PRED-OUTCOME", "outcome.buyer_contract", {
            "status": "won",
        }),
    })
    plan = TailFirstStagePlannerV1().plan(scenario)
    assert plan.planner == "tail_first"

    class ReachedBoundary(Exception):
        pass

    def boundary(_stage, *_args, **kwargs):
        assert kwargs["fixed_card_owners"] is None
        assert kwargs["required_base_raw_ns"] is None
        raise ReachedBoundary

    monkeypatch.setattr(OrToolsTailProofBoundaryV1, "solve", boundary)
    compiler = OrToolsTailFirstCandidateCompilerV1(
        rules=rules, plan=plan,
        proof_request=ScenarioProofRequestV1.build({
            "schemaVersion": 1, "mode": "exact", "visibility": "complete_state",
            "claim": "winning_root_action_count", "targetTeam": "north_south",
            "utility": "round_win", "successThreshold": 1,
            "budgetProfile": "late_game_exact_v1",
        }),
        maximum_proof_nodes=100_000, maximum_proof_depth=8,
        maximum_line_count_limbs=64, maximum_conflicts=250_000,
        maximum_prefix_lift_attempts=1, native_cp_sat_scoring=True,
        policy_prefix_steps=0,
        require_optimal_auction_actions=True,
    )
    with pytest.raises(ReachedBoundary):
        compiler(
            scenario, (), seed=0, maximum_seconds=15,
            maximum_conflicts=250_000,
        )


@pytest.mark.parametrize("suit_count", [7, 8])
@pytest.mark.parametrize("early_trick", [False, True])
def test_v3_tail_binds_deal_counts_before_boundary(rules, monkeypatch, suit_count, early_trick) -> None:
    suits = ("spades", "hearts", "diamonds", "clubs")
    counted_seats = (Seat.NORTH,) if early_trick else tuple(Seat)
    ignored = _leaf("PRED-PREFERENCE", "deal.suit_count", {
        "seat": "north", "suit": "hearts", "minimum": 8, "maximum": 8,
        "handStage": "post_purchase",
    })
    ignored["hardness"] = "prefer"
    scenario = ResearchScenarioV1.build({
        "schemaVersion": 1, "origin": {"kind": "search"},
        "constraints": {"op": "all", "children": [
            _leaf("PRED-CONTRACT", "auction.contract", {"contract": "sun"}),
            _leaf("PRED-BUYER", "auction.buyer", {"seat": "north"}),
            _leaf("PRED-CHOICES", "state.legal_action_count", {
                "minimum": 2, "maximum": 2,
            }),
            *(_leaf(f"PRED-HAND-{int(seat)}", "deal.suit_count", {
                "seat": seat.name.lower(), "suit": suits[int(seat)],
                "minimum": suit_count, "maximum": suit_count, "handStage": "post_purchase",
            }) for seat in counted_seats),
            *([_leaf("PRED-EARLY-CARD", "play.action_matches", {
                "trickIndex": 0, "position": 0, "actingSeat": "north",
                "selector": {"suits": ["spades"]},
            })] if early_trick else []),
            ignored,
        ]},
        "decisionPoint": {"kind": "play", "selector": {
            "trickIndex": 6, "position": 0, "remainingCards": 2,
            "actingSeat": "north",
        }},
        "outcome": _leaf("PRED-ROOT", "proof.winning_root_action_count", {
            "minimum": 2, "maximum": 2,
        }),
    })
    plan = TailFirstStagePlannerV1().plan(scenario)
    assert plan.planner == "tail_first"

    class ReachedBoundary(Exception):
        pass

    def boundary(_stage, _scenario, _plan, tail, **kwargs):
        owners = kwargs["fixed_card_owners"]
        assert owners is not None, "fully counted hands must start from reachable play"
        assert kwargs["required_base_raw_ns"] is not None
        assert kwargs["required_base_tricks_ns"] is not None
        for seat in counted_seats:
            assert len(tail.initial_hands[int(seat)]) == 2
            assert sum(owner == seat and Card(card).suit.name.lower() == suits[int(seat)]
                       for card, owner in enumerate(owners)) == suit_count
        raise ReachedBoundary

    monkeypatch.setattr(OrToolsTailProofBoundaryV1, "solve", boundary)
    compiler = OrToolsTailFirstCandidateCompilerV1(
        rules=rules, plan=plan,
        proof_request=ScenarioProofRequestV1.build({
            "schemaVersion": 1, "mode": "exact", "visibility": "complete_state",
            "claim": "winning_root_action_count", "targetTeam": "north_south",
            "utility": "awarded_point_margin", "successThreshold": 1,
            "budgetProfile": "late_game_exact_v1",
        }),
        maximum_proof_nodes=100_000, maximum_proof_depth=8,
        maximum_line_count_limbs=64, maximum_conflicts=250_000,
        maximum_prefix_lift_attempts=1, native_cp_sat_scoring=True,
    )
    with pytest.raises(ReachedBoundary):
        compiler(scenario, (), seed=3, maximum_seconds=15,
                 maximum_conflicts=250_000)


def test_policy_full_play_defers_native_project_outcome_to_boundary(
    rules, monkeypatch
) -> None:
    raw = _nearest_feasible_changed_scenario().to_dict()
    raw.pop("sha256")
    raw["constraints"].pop("sha256")
    raw["constraints"]["children"].append(_leaf(
        "PRED-EARLY-CARDS", "play.trick_card_count", {
            "trickIndex": 0,
            "selector": {"ranks": [
                "seven", "eight", "nine", "jack", "queen", "king", "ten", "ace",
            ]},
            "minimum": 4, "maximum": 4,
        },
    ))
    scenario = ResearchScenarioV1.build(raw)
    plan = TailFirstStagePlannerV1().plan(scenario)

    class UnexpectedProjectStage(Exception):
        pass

    def project_stage(*_args, **_kwargs):
        raise UnexpectedProjectStage

    monkeypatch.setattr(
        tail_candidate_module,
        "attach_project_lifecycle_fragment_v1",
        project_stage,
    )
    compiler = OrToolsTailFirstCandidateCompilerV1(
        rules=rules, plan=plan, proof_request=_exact_unique_winner_proof(),
        maximum_proof_nodes=100_000, maximum_proof_depth=12,
        maximum_line_count_limbs=64, maximum_conflicts=1,
        maximum_prefix_lift_attempts=1, native_cp_sat_scoring=True,
        policy_prefix_steps=4,
    )
    attempt = compiler(
        scenario, (), seed=0, maximum_seconds=5, maximum_conflicts=1,
    )
    assert attempt.status == "incomplete"
    assert attempt.reason == "tail_construction_budget"




@pytest.mark.parametrize("after_graph", [False, True])
def test_v3_incomplete_boundary_retains_exact_state_cut(
    rules, monkeypatch, after_graph
) -> None:
    scenario = _nearest_feasible_changed_scenario()
    plan = TailFirstStagePlannerV1().plan(scenario)
    graph_receipts = []
    original_compile = tail_candidate_module.OrToolsCompleteStateGraphCompilerV1.compile

    def recorded_compile(*args, **kwargs):
        result = original_compile(*args, **kwargs)
        graph_receipts.append(result.receipt.to_dict())
        return result

    monkeypatch.setattr(
        tail_candidate_module.OrToolsCompleteStateGraphCompilerV1,
        "compile", recorded_compile,
    )
    original_boundary = OrToolsTailProofBoundaryV1.solve
    original_prove = tail_candidate_module.OrToolsScenarioGraphProofV1.prove

    def stopped_boundary(*args, **kwargs):
        if after_graph and not graph_receipts:
            return original_boundary(*args, **kwargs)
        return SimpleNamespace(
            status="incomplete", reason="tail_boundary_budget",
            model_build_seconds=0.0, solver_seconds=0.0, solve_attempts=0,
        )

    monkeypatch.setattr(OrToolsTailProofBoundaryV1, "solve", stopped_boundary)
    if after_graph:
        monkeypatch.setattr(tail_candidate_module.OrToolsScenarioGraphProofV1, "prove",
            lambda *args, **kwargs: replace(original_prove(*args, **kwargs), claim_satisfied=False))
    progress = []
    compiler = OrToolsTailFirstCandidateCompilerV1(
        rules=rules, plan=plan, proof_request=_exact_unique_winner_proof(),
        maximum_proof_nodes=100_000, maximum_proof_depth=12,
        maximum_line_count_limbs=64, maximum_conflicts=250_000,
        maximum_prefix_lift_attempts=1, native_cp_sat_scoring=True,
        progress_callback=progress.append,
    )
    attempt = compiler(
        scenario, (), seed=0, maximum_seconds=15,
        maximum_conflicts=250_000,
    )
    assert attempt.status == "incomplete"
    assert attempt.reason == "tail_boundary_budget"
    assert any(event.get("proofBoundaryStatus") == "incomplete"
               and event.get("proofBoundaryReason") == "tail_boundary_budget"
               for event in progress)
    assert attempt.partial_receipt is not None
    vector = attempt.partial_receipt["evidence"]["decisionStateStatusVector"]
    assert len(vector) == 32 and set(vector) <= set(range(9))
    if after_graph:
        assert graph_receipts
        assert attempt.partial_receipt["evidence"]["graphReceipt"] == graph_receipts[-1]
    else:
        assert "graphReceipt" not in attempt.partial_receipt["evidence"]


def test_v3_incomplete_lift_retains_stage_counters(rules, monkeypatch) -> None:
    scenario = _nearest_feasible_changed_scenario()
    plan = TailFirstStagePlannerV1().plan(scenario)
    # This is a lift-receipt test. Keep real construction, but isolate it from
    # strategic search yield; separate acceptance tests require actual proofs.
    original_prove = tail_candidate_module.OrToolsScenarioGraphProofV1.prove

    def passing_proof(prover, *args, **kwargs):
        return replace(original_prove(prover, *args, **kwargs), claim_satisfied=True)

    monkeypatch.setattr(tail_candidate_module.OrToolsScenarioGraphProofV1,
                        "prove", passing_proof)
    monkeypatch.setattr(tail_candidate_module, "_proof_stage_projection_v1",
                        lambda *_args, **_kwargs: ("satisfied", {}))
    def incomplete_lift(*_args, **kwargs):
        assert kwargs["maximum_conflicts"] == 10_000_000
        assert kwargs["cancel_check"] is None
        return OrToolsCandidateAttemptV1(
            "incomplete", None, "tail_prefix_lift_budget"
        )
    monkeypatch.setattr(
        OrToolsTailFirstBackwardLiftV1,
        "lift",
        incomplete_lift,
    )
    attempt = OrToolsTailFirstCandidateCompilerV1(
        rules=rules, plan=plan, proof_request=_exact_unique_winner_proof(),
        maximum_proof_nodes=100_000, maximum_proof_depth=12,
        maximum_line_count_limbs=64, maximum_conflicts=10_000_000,
        maximum_prefix_lift_attempts=1, native_cp_sat_scoring=True,
    )(
        # This checks the lift receipt, not search latency; allow the real
        # search to reach the mocked lift on slower test hosts.
        scenario, (), seed=20, maximum_seconds=60,
        maximum_conflicts=10_000_000,
    )
    assert attempt.status == "incomplete"
    assert attempt.partial_receipt is not None
    assert attempt.partial_receipt["stage"] == "prefixLift"
    assert attempt.partial_receipt["evidence"]["stageCounters"]["prefixLiftAttempts"] == 1
    core = dict(attempt.partial_receipt)
    assert core.pop("sha256") == tail_candidate_module.canonical_sha256(core)


@pytest.mark.parametrize("native", [False, True])
def test_quality_rejection_persists_one_complete_play_no_good(
    rules, monkeypatch, native,
) -> None:
    scenario = _nearest_feasible_changed_scenario()
    plan = TailFirstStagePlannerV1().plan(scenario)
    combination = plan.boundary_combinations[0]
    trace = {
        "dealer": combination[0],
        "contract": combination[1],
        "buyer": combination[2],
        "trumpSuit": combination[3],
        "matchScore": [0, 0],
        "tailFirstStagePlan": plan.to_dict(),
        "deckOrder": list(range(32)),
        "play": {
            "initialHands": {
                seat.name.lower(): [
                    card.code
                    for card in tail_candidate_module.FULL_DECK
                    if card.id // 8 == int(seat)
                ]
                for seat in Seat
            },
            "plays": [
                {
                    "actor": Seat(card.id // 8).name.lower(),
                    "card": card.code,
                    "ekka": False,
                }
                for card in tail_candidate_module.FULL_DECK
            ],
            "legalPlayActionsByStep": [
                [
                    {"card": card.code, "ekka": False}
                    for card in tail_candidate_module.FULL_DECK[:8]
                ]
            ],
        },
    }
    receipt = OrToolsResearchStageReceiptV1(
        "test", "a" * 64, "b" * 64, "c" * 64, "optimal"
    )
    candidate = OrToolsNeutralCandidateV1(
        "search", trace, {}, tuple(range(32)), {"test": receipt}
    )
    compiler = OrToolsTailFirstCandidateCompilerV1(
        rules=rules,
        plan=plan,
        proof_request=_exact_unique_winner_proof(),
        maximum_proof_nodes=100_000,
        maximum_proof_depth=12,
        maximum_line_count_limbs=64,
        maximum_conflicts=10_000_000,
        maximum_prefix_lift_attempts=1,
        native_cp_sat_scoring=native,
    )

    compiler._reject_quality_candidate(
        candidate, reject_play_zero_observation=True, reject_initial_input=native,
    )
    compiler._reject_quality_candidate(
        candidate, reject_play_zero_observation=True, reject_initial_input=native,
    )

    key = (*combination, 0, 0) if native else combination
    assert compiler._quality_rejected_complete_plays[key] == [
        tuple(range(32))
    ]
    cut = compiler._quality_rejected_play_zero_actions[key]
    assert len(cut) == 1
    assert cut[0].actor is Seat.NORTH
    assert cut[0].actor_card_ids == tuple(range(8))
    assert cut[0].selected_card_id == 0
    assert cut[0].ekka is False
    assert compiler._quality_rejected_initial_decks == ({key: [tuple(range(32))]} if native else {})
    if native:
        other_trace = {**trace, "matchScore": [1, 0],
            "deckOrder": list(reversed(range(32))),
            "play": {**trace["play"], "plays": list(reversed(trace["play"]["plays"]))}}
        compiler._reject_quality_candidate(replace(candidate, trace=other_trace,
            no_good_vector=tuple(reversed(range(32)))),
            reject_play_zero_observation=True, reject_initial_input=True)
        assert compiler._quality_rejected_complete_plays[(*combination, 1, 0)] == [tuple(reversed(range(32)))]
        assert compiler._quality_rejected_play_zero_actions[(*combination, 1, 0)] != cut
        assert compiler._quality_rejected_initial_decks[(*combination, 1, 0)] == [tuple(reversed(range(32)))]
        with pytest.raises(ValueError, match="native complete deal"):
            compiler._reject_quality_candidate(replace(candidate, no_good_vector=tuple(reversed(range(32)))),
                reject_initial_input=True)
        for invalid in (None, [True, 0], [-1, 0], [0]):
            with pytest.raises(ValueError, match="starting match score"):
                compiler._reject_quality_candidate(replace(candidate, trace={**trace, "matchScore": invalid}))

    hokum_combination = (combination[0], "hokum", combination[2], "clubs")
    compiler._quality_rejected_complete_plays[hokum_combination] = []
    compiler._quality_rejected_play_zero_actions[hokum_combination] = []
    hokum_candidate = OrToolsNeutralCandidateV1(
        "search",
        {
            **trace,
            "contract": "hokum",
            "trumpSuit": "clubs",
        },
        {},
        tuple(range(32)),
        {"test": receipt},
    )
    if native:
        with pytest.raises(ValueError, match="unsupported boundary"):
            compiler._reject_quality_candidate(hokum_candidate, reject_play_zero_observation=True)
    else:
        compiler._reject_quality_candidate(hokum_candidate, reject_play_zero_observation=True)
        assert compiler._quality_rejected_complete_plays[hokum_combination] == []
        assert compiler._quality_rejected_play_zero_actions[hokum_combination] == cut

    observed = []
    models = []
    original_model = tail_candidate_module.OrToolsScenarioModelV1

    def capture_model(**kwargs):
        model = original_model(**kwargs)
        models.append(model)
        return model

    monkeypatch.setattr(tail_candidate_module, "OrToolsScenarioModelV1", capture_model)

    def inspect_tail(_stage, **kwargs):
        closure = inspect.getclosurevars(kwargs["model_configurator"])
        observed.append(
            closure.nonlocals["excluded_play_zero_actions"][combination]
        )
        raise PlayTailSolveStopped(
            "test_owner_cut_observed", evidence={"solverStatus": "unknown"}
        )

    monkeypatch.setattr(OrToolsPlayTailStageV1, "solve", inspect_tail)
    attempt = compiler(
        scenario,
        (),
        seed=1,
        maximum_seconds=15,
        maximum_conflicts=10_000_000,
    )
    assert attempt.status == "incomplete"
    assert observed == [cut]
    if native:
        assert models[0].excluded_deck_orders == (tuple(range(32)),)
        for position, card in enumerate(range(32)):
            models[0].model.add(models[0].card_at_position[position] == card)
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        assert solver.solve(models[0].model) == cp_model.INFEASIBLE

        def exhausted_tail(_stage, **kwargs):
            raise PlayTailSolveStopped("infeasible", evidence={"solverStatus": "infeasible"})

        monkeypatch.setattr(OrToolsPlayTailStageV1, "solve", exhausted_tail)
        monkeypatch.setattr(tail_candidate_module, "_required_search_initial_inputs_v3",
            lambda *_args, **_kwargs: (Seat.parse(combination[0]), (0, 0), False))
        exhausted = compiler(scenario, (), seed=2, maximum_seconds=15,
                             maximum_conflicts=10_000_000)
        assert exhausted.status == "not_found"
        assert exhausted.reason == "tail_quality_constrained_no_witness"
        monkeypatch.setattr(tail_candidate_module, "_required_search_initial_inputs_v3",
            lambda *_args, **_kwargs: (Seat.parse(combination[0]), (1, 0), False))
        compiler(scenario, (), seed=3, maximum_seconds=15, maximum_conflicts=10_000_000)
        assert models[-1].excluded_deck_orders == (tuple(reversed(range(32))),)


def test_flagship_contradiction_recognizer_ignores_missing_acting_seat(rules) -> None:
    raw = _flagship_scenario().to_dict()
    raw.pop("sha256")
    raw["decisionPoint"]["selector"].pop("actingSeat")
    generic = ResearchScenarioV1.build(raw)

    assert (
        OrToolsFlagshipResearchV1(rules=rules).contradiction_core_for_scenario(
            generic, _exact_unique_winner_proof()
        )
        is None
    )


def test_flagship_contradiction_recognizer_accepts_stronger_exact_winner_seats(
    rules,
) -> None:
    raw = _flagship_scenario().to_dict()
    raw.pop("sha256")
    predicates = raw["constraints"]["children"]
    opponent_winner = next(
        item for item in predicates if item["predicateId"] == "PRED-OPPONENT-WINS-SIX"
    )
    opponent_winner["arguments"].pop("team")
    opponent_winner["arguments"]["seat"] = "north"
    target_winner = next(
        item for item in predicates if item["predicateId"] == "PRED-TEAM-WINS-SEVEN"
    )
    target_winner["arguments"].pop("team")
    target_winner["arguments"]["seat"] = "west"
    for item in predicates:
        item.pop("sha256", None)
        selector = item["arguments"].get("selector")
        if selector is not None:
            selector["ekka"] = "any"
    raw["constraints"].pop("sha256", None)
    raw["origin"].pop("sha256", None)
    raw["outcome"].pop("sha256", None)
    ui_scenario = ResearchScenarioV1.build(raw)

    core = OrToolsFlagshipResearchV1(rules=rules).contradiction_core_for_scenario(
        ui_scenario, _exact_unique_winner_proof()
    )

    assert core is not None
    assert "PRED-OPPONENT-WINS-SIX" in core
    assert "PRED-TEAM-WINS-SEVEN" in core

    opponent_winner["arguments"]["seat"] = "west"
    target_winner["arguments"]["seat"] = "north"
    non_contradictory = ResearchScenarioV1.build(raw)
    assert (
        OrToolsFlagshipResearchV1(rules=rules).contradiction_core_for_scenario(
            non_contradictory, _exact_unique_winner_proof()
        )
        is None
    )
